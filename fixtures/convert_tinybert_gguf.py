#!/usr/bin/env python3
"""
Convert ms-marco-TinyBERT-L-2-v2 Cross-Encoder to GGUF format.

This script creates a GGUF file that can be loaded by the Zig TinyBERT module
for re-ranking in WASM.

Usage:
    python convert_tinybert_gguf.py

Output:
    ms-marco-tinybert-l2.gguf (~8MB F16, ~4MB after Q8 quantization)
"""

import torch
import numpy as np
from pathlib import Path

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
    import gguf
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install transformers torch gguf")
    exit(1)

MODEL_ID = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
OUTPUT_FILE = Path(__file__).parent.parent / "models" / "ms-marco-tinybert-l2.gguf"

def main():
    print(f"Loading {MODEL_ID}...")

    # Load model and config
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = model.config

    print(f"Model architecture:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Vocab size: {config.vocab_size}")

    # Create output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Initialize GGUF writer (architecture passed in constructor)
    print(f"\nCreating GGUF file: {OUTPUT_FILE}")
    writer = gguf.GGUFWriter(str(OUTPUT_FILE), "bert")

    # Add metadata
    writer.add_name("ms-marco-TinyBERT-L-2-v2")
    writer.add_description("TinyBERT Cross-Encoder for Re-ranking (2 layers, 128 hidden)")
    writer.add_context_length(512)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_layer_norm_eps(config.layer_norm_eps)
    writer.add_file_type(gguf.GGMLQuantizationType.F16)

    # Add vocabulary
    print("Adding vocabulary...")
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    tokens = [token for token, _ in sorted_vocab]
    writer.add_token_list(tokens)

    # Add special token IDs
    writer.add_bos_token_id(tokenizer.cls_token_id)
    writer.add_eos_token_id(tokenizer.sep_token_id)
    writer.add_pad_token_id(tokenizer.pad_token_id)

    # Add tensors
    print("Adding tensors...")
    state_dict = model.state_dict()

    tensor_count = 0
    classifier_found = False

    for name, param in state_dict.items():
        # Convert to F16
        data = param.detach().cpu().numpy().astype(np.float16)

        # Track classifier head
        if "classifier" in name:
            classifier_found = True
            print(f"  [CLASSIFIER] {name}: {data.shape}")
        else:
            print(f"  {name}: {data.shape}")

        writer.add_tensor(name, data)
        tensor_count += 1

    print(f"\nTotal tensors: {tensor_count}")

    if not classifier_found:
        print("\nWARNING: No classifier weights found!")
        print("This model may not work as a cross-encoder.")
    else:
        print("\nClassifier head found - cross-encoder ready!")

    # Write file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Report file size
    file_size = OUTPUT_FILE.stat().st_size
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Size: {file_size / 1024 / 1024:.2f} MB")

    # Verification command
    print(f"\nVerify classifier tensors:")
    print(f"  python -c \"import gguf; r=gguf.GGUFReader('{OUTPUT_FILE}'); print([t.name for t in r.tensors if 'classifier' in t.name])\"")

if __name__ == "__main__":
    main()
