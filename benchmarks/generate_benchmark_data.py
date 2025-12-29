#!/usr/bin/env python3
"""
Generate benchmark datasets for end-to-end @logic_table comparison.

Creates identical data in Lance and Parquet formats:
- 100K rows with embeddings (384-dim like MiniLM)
- Columns: id, amount, embedding

This allows fair comparison between:
- LanceQL reading Lance file
- DuckDB reading Parquet file
- Polars reading Parquet file
"""

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import lance
    HAS_LANCE = True
except ImportError:
    HAS_LANCE = False
    print("Warning: lance not installed, will only generate Parquet")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_ROWS = 100_000  # 100K rows for realistic benchmark
EMBEDDING_DIM = 384  # MiniLM dimension

def generate_data():
    """Generate test data with embeddings."""
    np.random.seed(42)  # Reproducible

    # Generate data
    ids = np.arange(NUM_ROWS, dtype=np.int64)
    amounts = np.random.uniform(10.0, 1000.0, NUM_ROWS).astype(np.float64)

    # Generate normalized embeddings (like real ML embeddings)
    embeddings = np.random.randn(NUM_ROWS, EMBEDDING_DIM).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return ids, amounts, embeddings

def save_parquet(ids, amounts, embeddings):
    """Save as Parquet file."""
    # Flatten embeddings for Parquet (store as list column)
    embedding_list = [emb.tolist() for emb in embeddings]

    table = pa.table({
        'id': pa.array(ids, type=pa.int64()),
        'amount': pa.array(amounts, type=pa.float64()),
        'embedding': pa.array(embedding_list, type=pa.list_(pa.float32())),
    })

    path = os.path.join(OUTPUT_DIR, 'benchmark_e2e.parquet')
    pq.write_table(table, path, compression='snappy')
    print(f"Created {path}")
    print(f"  Rows: {NUM_ROWS}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  Size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")
    return path

def save_lance(ids, amounts, embeddings):
    """Save as Lance file."""
    if not HAS_LANCE:
        print("Skipping Lance (not installed)")
        return None

    # Use fixed_size_list for embeddings (more efficient)
    embedding_type = pa.list_(pa.float32(), EMBEDDING_DIM)
    embedding_list = [emb.tolist() for emb in embeddings]

    table = pa.table({
        'id': pa.array(ids, type=pa.int64()),
        'amount': pa.array(amounts, type=pa.float64()),
        'embedding': pa.array(embedding_list, type=embedding_type),
    })

    path = os.path.join(OUTPUT_DIR, 'benchmark_e2e.lance')
    lance.write_dataset(table, path, mode='overwrite')

    # Get total size of Lance dataset
    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    print(f"Created {path}")
    print(f"  Rows: {NUM_ROWS}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  Size: {total_size / 1024 / 1024:.1f} MB")
    return path

def verify_data():
    """Verify both formats have identical data."""
    print("\n--- Verification ---")

    # Read Parquet
    parquet_path = os.path.join(OUTPUT_DIR, 'benchmark_e2e.parquet')
    pq_table = pq.read_table(parquet_path)
    print(f"Parquet: {len(pq_table)} rows")

    # Read Lance
    if HAS_LANCE:
        lance_path = os.path.join(OUTPUT_DIR, 'benchmark_e2e.lance')
        lance_ds = lance.dataset(lance_path)
        print(f"Lance: {lance_ds.count_rows()} rows")

        # Compare first row embedding
        pq_emb = pq_table['embedding'][0].as_py()
        lance_emb = lance_ds.to_table()['embedding'][0].as_py()

        if pq_emb == lance_emb:
            print("First embedding matches")
        else:
            print("WARNING: Embeddings differ!")

if __name__ == '__main__':
    print(f"Generating benchmark data: {NUM_ROWS} rows, {EMBEDDING_DIM}-dim embeddings\n")

    ids, amounts, embeddings = generate_data()

    save_parquet(ids, amounts, embeddings)
    print()
    save_lance(ids, amounts, embeddings)

    verify_data()
    print("\nDone!")
