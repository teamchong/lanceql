#!/usr/bin/env python3
"""
Translate LAION-1M captions to multiple languages.

Usage:
    python translate_captions.py --input laion-1m/images.lance --output translations/ --limit 100000

Requirements:
    pip install lance transformers torch pyarrow
"""

import argparse
import lance
import pyarrow as pa
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from pathlib import Path
from tqdm import tqdm

# Language mappings for NLLB model
LANGUAGES = {
    'en': 'eng_Latn',  # English
    'zh': 'zho_Hans',  # Chinese (Simplified)
    'es': 'spa_Latn',  # Spanish
    'ja': 'jpn_Jpan',  # Japanese
    'fr': 'fra_Latn',  # French
}

class CaptionTranslator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device=None):
        """Initialize translation model."""
        print(f"Loading translation model: {model_name}")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def translate(self, text, src_lang='eng_Latn', tgt_lang='zho_Hans', max_length=512):
        """Translate a single text."""
        if not text or not isinstance(text, str):
            return ""

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                max_length=max_length
            )

            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original on error

    def translate_batch(self, texts, src_lang='eng_Latn', tgt_lang='zho_Hans', batch_size=32):
        """Translate a batch of texts."""
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating to {tgt_lang}"):
            batch = texts[i:i + batch_size]

            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                    max_length=512
                )

                translated = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                results.extend(translated)
            except Exception as e:
                print(f"Batch translation error: {e}")
                results.extend(batch)  # Return originals on error

        return results


def create_caption_dataset(dataset_path, output_dir, target_langs=['zh', 'es'], limit=None, batch_size=32):
    """
    Extract captions and create translated datasets.

    Args:
        dataset_path: Path to input Lance dataset
        output_dir: Directory to save translated datasets
        target_langs: List of target language codes (e.g., ['zh', 'es'])
        limit: Limit number of captions to translate (None = all)
        batch_size: Batch size for translation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")
    ds = lance.dataset(dataset_path)

    # Read captions
    print("Reading captions...")
    table = ds.to_table(columns=['text'])

    if limit:
        print(f"Limiting to {limit} captions")
        table = table.slice(0, limit)

    captions = [row['text'] for row in table.to_pylist()]
    total = len(captions)
    print(f"Loaded {total} captions")

    # Create English dataset (original)
    print("Creating English dataset...")
    en_data = {
        'image_id': list(range(total)),
        'text': captions,
        'language': ['en'] * total
    }
    en_table = pa.Table.from_pydict(en_data)
    en_path = output_dir / 'captions_en.lance'
    lance.write_dataset(en_table, en_path)
    print(f"✓ Saved English dataset to {en_path}")

    # Initialize translator
    translator = CaptionTranslator()

    # Translate to target languages
    for lang_code in target_langs:
        if lang_code not in LANGUAGES:
            print(f"Warning: Unsupported language '{lang_code}', skipping")
            continue

        print(f"\nTranslating to {lang_code.upper()}...")
        src_lang = LANGUAGES['en']
        tgt_lang = LANGUAGES[lang_code]

        translated = translator.translate_batch(
            captions,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=batch_size
        )

        # Create translated dataset
        translated_data = {
            'image_id': list(range(total)),
            'text': translated,
            'language': [lang_code] * total
        }
        translated_table = pa.Table.from_pydict(translated_data)
        translated_path = output_dir / f'captions_{lang_code}.lance'
        lance.write_dataset(translated_table, translated_path)
        print(f"✓ Saved {lang_code.upper()} dataset to {translated_path}")

    print("\n✅ Translation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Generated datasets: captions_en.lance, " + ", ".join([f"captions_{l}.lance" for l in target_langs]))


def main():
    parser = argparse.ArgumentParser(description='Translate LAION captions to multiple languages')
    parser.add_argument('--input', required=True, help='Path to input Lance dataset')
    parser.add_argument('--output', default='translations/', help='Output directory for translated datasets')
    parser.add_argument('--languages', default='zh,es', help='Comma-separated language codes (e.g., zh,es,ja)')
    parser.add_argument('--limit', type=int, default=100000, help='Limit number of captions (default: 100000)')
    parser.add_argument('--batch-size', type=int, default=32, help='Translation batch size (default: 32)')

    args = parser.parse_args()

    target_langs = [lang.strip() for lang in args.languages.split(',')]

    print("=" * 60)
    print("LAION Caption Translation")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Languages: {', '.join(target_langs)}")
    print(f"Limit: {args.limit or 'All'}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    create_caption_dataset(
        dataset_path=args.input,
        output_dir=args.output,
        target_langs=target_langs,
        limit=args.limit,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
