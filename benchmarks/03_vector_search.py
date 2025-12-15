#!/usr/bin/env python3
"""
Benchmark: Vector Search
Lance has native ANN (Approximate Nearest Neighbor) search.
Parquet requires external libraries (FAISS, Annoy, etc.).
"""

import time
import tempfile
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import lancedb

# Try to import FAISS, fall back gracefully
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("⚠️  FAISS not installed. Install with: pip install faiss-cpu")

NUM_VECTORS = 100_000
VECTOR_DIM = 384  # Common embedding dimension (e.g., MiniLM)
NUM_QUERIES = 100
TOP_K = 10

def create_data() -> tuple[pa.Table, np.ndarray]:
    """Create sample dataset with embeddings."""
    embeddings = np.random.randn(NUM_VECTORS, VECTOR_DIM).astype(np.float32)
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    table = pa.table({
        "id": range(NUM_VECTORS),
        "text": [f"document_{i}" for i in range(NUM_VECTORS)],
        "embedding": [emb.tolist() for emb in embeddings],
    })
    return table, embeddings

def benchmark_lance_vector_search(tmpdir: str, data: pa.Table, queries: np.ndarray) -> dict:
    """Lance: Native vector search with IVF-PQ index."""
    db_path = os.path.join(tmpdir, "lance_db")
    db = lancedb.connect(db_path)

    # Write data
    start = time.perf_counter()
    tbl = db.create_table("vectors", data)
    write_time = time.perf_counter() - start

    # Create index
    start = time.perf_counter()
    tbl.create_index(
        metric="cosine",
        num_partitions=256,
        num_sub_vectors=48,
    )
    index_time = time.perf_counter() - start

    # Search
    search_times = []
    for query in queries:
        start = time.perf_counter()
        results = tbl.search(query.tolist()).limit(TOP_K).to_list()
        search_times.append(time.perf_counter() - start)

    return {
        "format": "Lance",
        "write_time": write_time,
        "index_time": index_time,
        "search_time_avg": np.mean(search_times),
        "search_time_p99": np.percentile(search_times, 99),
        "qps": len(queries) / sum(search_times),
        "native_vector_search": True,
    }

def benchmark_parquet_faiss(tmpdir: str, data: pa.Table, embeddings: np.ndarray, queries: np.ndarray) -> dict:
    """Parquet + FAISS: External vector index."""
    if not HAS_FAISS:
        return {
            "format": "Parquet + FAISS",
            "error": "FAISS not installed",
        }

    parquet_path = os.path.join(tmpdir, "vectors.parquet")
    index_path = os.path.join(tmpdir, "vectors.faiss")

    # Write parquet (without embeddings for fair comparison, or with)
    start = time.perf_counter()
    pq.write_table(data, parquet_path)
    write_time = time.perf_counter() - start

    # Build FAISS index
    start = time.perf_counter()
    # IVF index similar to Lance's
    nlist = 256  # Number of clusters
    quantizer = faiss.IndexFlatIP(VECTOR_DIM)  # Inner product for cosine
    index = faiss.IndexIVFFlat(quantizer, VECTOR_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 16  # Search 16 clusters
    index_time = time.perf_counter() - start

    # Search
    search_times = []
    for query in queries:
        start = time.perf_counter()
        distances, indices = index.search(query.reshape(1, -1), TOP_K)
        search_times.append(time.perf_counter() - start)

    return {
        "format": "Parquet + FAISS",
        "write_time": write_time,
        "index_time": index_time,
        "search_time_avg": np.mean(search_times),
        "search_time_p99": np.percentile(search_times, 99),
        "qps": len(queries) / sum(search_times),
        "native_vector_search": False,
    }

def main():
    print("=" * 60)
    print("Benchmark: Vector Search")
    print(f"  Vectors: {NUM_VECTORS:,}")
    print(f"  Dimensions: {VECTOR_DIM}")
    print(f"  Queries: {NUM_QUERIES}")
    print(f"  Top-K: {TOP_K}")
    print("=" * 60)

    print("\n⏳ Generating random embeddings...")
    data, embeddings = create_data()
    queries = np.random.randn(NUM_QUERIES, VECTOR_DIM).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n⏳ Benchmarking Lance...")
        lance_results = benchmark_lance_vector_search(tmpdir, data, queries)

        print("⏳ Benchmarking Parquet + FAISS...")
        faiss_results = benchmark_parquet_faiss(tmpdir, data, embeddings, queries)

    print("\nResults:")
    print("-" * 60)
    print(f"{'Metric':<25} {'Lance':>15} {'Parquet+FAISS':>15}")
    print("-" * 60)
    print(f"{'Native Vector Search':<25} {'✅ Yes':>15} {'❌ No':>15}")
    print(f"{'Write Time':<25} {lance_results['write_time']:>13.2f}s {faiss_results.get('write_time', 0):>13.2f}s")
    print(f"{'Index Time':<25} {lance_results['index_time']:>13.2f}s {faiss_results.get('index_time', 0):>13.2f}s")
    print(f"{'Search Time (avg)':<25} {lance_results['search_time_avg']*1000:>11.2f}ms {faiss_results.get('search_time_avg', 0)*1000:>11.2f}ms")
    print(f"{'Search Time (p99)':<25} {lance_results['search_time_p99']*1000:>11.2f}ms {faiss_results.get('search_time_p99', 0)*1000:>11.2f}ms")
    print(f"{'Queries/sec':<25} {lance_results['qps']:>13.0f} {faiss_results.get('qps', 0):>13.0f}")
    print("-" * 60)

    print("\n📝 Key Advantages of Lance:")
    print("   • Single file format - no separate index files to manage")
    print("   • Automatic index updates on data changes")
    print("   • Combined SQL + vector search in one query")
    print("   • Works with HTTP Range requests (browser/remote)")

    print("\n📝 Parquet + FAISS Drawbacks:")
    print("   • Must maintain separate index files")
    print("   • Index must be rebuilt on data changes")
    print("   • Complex deployment (multiple files)")
    print("   • No combined SQL + vector filtering")

if __name__ == "__main__":
    main()
