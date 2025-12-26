#!/usr/bin/env python3
"""
Benchmark: LanceQL vs Polars vs DuckDB

Compares query performance on:
1. Vector similarity search (LanceQL specialty)
2. Aggregations (SUM, AVG, COUNT, GROUP BY)
3. Filtering (WHERE clauses)
4. Sorting (ORDER BY)

Run with: python benchmarks/bench_vs_polars_duckdb.py
"""

import time
import json
import sys
from pathlib import Path

# Check imports
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: pip install numpy")
    sys.exit(1)

HAS_POLARS = False
HAS_DUCKDB = False
HAS_LANCEQL = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    print("WARNING: polars not installed. Skipping Polars benchmarks.")

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    print("WARNING: duckdb not installed. Skipping DuckDB benchmarks.")

# Try to import our LanceQL package
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    from metal0.lanceql.vector import VectorAccelerator, vector_accelerator
    HAS_LANCEQL = True
    print(f"LanceQL backend: {vector_accelerator.backend}")
except ImportError as e:
    print(f"WARNING: LanceQL not available ({e}). Using NumPy baseline.")


def generate_test_data(num_rows: int, embedding_dim: int = 384):
    """Generate test data with embeddings."""
    np.random.seed(42)

    data = {
        "id": np.arange(num_rows),
        "amount": np.random.uniform(10, 10000, num_rows).astype(np.float32),
        "category": np.random.choice(["A", "B", "C", "D", "E"], num_rows),
        "status": np.random.choice(["pending", "completed", "failed"], num_rows),
        "score": np.random.uniform(0, 1, num_rows).astype(np.float32),
    }

    # Add embedding column (high-dimensional float32 vectors)
    embeddings = np.random.randn(num_rows, embedding_dim).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return data, embeddings


def cosine_similarity_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Pure NumPy cosine similarity (baseline)."""
    # query: (dim,), vectors: (n, dim)
    dots = vectors @ query
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    return dots / norms


def benchmark_vector_search(data: dict, embeddings: np.ndarray, num_queries: int = 10):
    """Benchmark vector similarity search."""
    results = {}
    dim = embeddings.shape[1]

    # Generate random query vectors
    np.random.seed(123)
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # NumPy baseline
    start = time.perf_counter()
    for q in queries:
        scores = cosine_similarity_batch(q, embeddings)
        top_k = np.argsort(scores)[-10:][::-1]
    numpy_time = (time.perf_counter() - start) / num_queries * 1000
    results["numpy"] = numpy_time

    # LanceQL (GPU/CPU accelerated)
    if HAS_LANCEQL:
        # Warm up
        _ = vector_accelerator.batch_cosine_similarity(queries[0], embeddings)

        start = time.perf_counter()
        for q in queries:
            scores = vector_accelerator.batch_cosine_similarity(q, embeddings)
            top_k = np.argsort(scores)[-10:][::-1]
        lanceql_time = (time.perf_counter() - start) / num_queries * 1000
        results["lanceql"] = lanceql_time

    # Polars (no native vector search)
    if HAS_POLARS:
        results["polars"] = "N/A (no native vector search)"

    # DuckDB (no native vector search)
    if HAS_DUCKDB:
        results["duckdb"] = "N/A (no native vector search)"

    return results


def benchmark_aggregations(data: dict, num_iterations: int = 100):
    """Benchmark aggregation queries."""
    results = {}

    # Polars
    if HAS_POLARS:
        df = pl.DataFrame(data)

        # Warm up
        _ = df.select(pl.col("amount").sum())

        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = df.select([
                pl.col("amount").sum().alias("sum_amount"),
                pl.col("amount").mean().alias("avg_amount"),
                pl.col("score").max().alias("max_score"),
                pl.len().alias("count")
            ])
        polars_time = (time.perf_counter() - start) / num_iterations * 1000
        results["polars"] = polars_time

    # DuckDB
    if HAS_DUCKDB:
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE test AS SELECT * FROM data", {"data": data})

        # Warm up
        conn.execute("SELECT SUM(amount) FROM test").fetchall()

        start = time.perf_counter()
        for _ in range(num_iterations):
            conn.execute("""
                SELECT SUM(amount), AVG(amount), MAX(score), COUNT(*)
                FROM test
            """).fetchall()
        duckdb_time = (time.perf_counter() - start) / num_iterations * 1000
        results["duckdb"] = duckdb_time
        conn.close()

    # NumPy baseline
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = np.sum(data["amount"])
        _ = np.mean(data["amount"])
        _ = np.max(data["score"])
        _ = len(data["id"])
    numpy_time = (time.perf_counter() - start) / num_iterations * 1000
    results["numpy"] = numpy_time

    return results


def benchmark_filtering(data: dict, num_iterations: int = 100):
    """Benchmark WHERE clause filtering."""
    results = {}

    # Polars
    if HAS_POLARS:
        df = pl.DataFrame(data)

        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = df.filter(
                (pl.col("amount") > 5000) &
                (pl.col("status") == "completed") &
                (pl.col("score") > 0.5)
            )
        polars_time = (time.perf_counter() - start) / num_iterations * 1000
        results["polars"] = polars_time

    # DuckDB
    if HAS_DUCKDB:
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE test AS SELECT * FROM data", {"data": data})

        start = time.perf_counter()
        for _ in range(num_iterations):
            conn.execute("""
                SELECT * FROM test
                WHERE amount > 5000
                AND status = 'completed'
                AND score > 0.5
            """).fetchall()
        duckdb_time = (time.perf_counter() - start) / num_iterations * 1000
        results["duckdb"] = duckdb_time
        conn.close()

    # NumPy baseline
    start = time.perf_counter()
    for _ in range(num_iterations):
        mask = (
            (data["amount"] > 5000) &
            (data["status"] == "completed") &
            (data["score"] > 0.5)
        )
        _ = data["id"][mask]
    numpy_time = (time.perf_counter() - start) / num_iterations * 1000
    results["numpy"] = numpy_time

    return results


def benchmark_group_by(data: dict, num_iterations: int = 100):
    """Benchmark GROUP BY queries."""
    results = {}

    # Polars
    if HAS_POLARS:
        df = pl.DataFrame(data)

        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = df.group_by("category").agg([
                pl.col("amount").sum(),
                pl.col("score").mean(),
                pl.len()
            ])
        polars_time = (time.perf_counter() - start) / num_iterations * 1000
        results["polars"] = polars_time

    # DuckDB
    if HAS_DUCKDB:
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE test AS SELECT * FROM data", {"data": data})

        start = time.perf_counter()
        for _ in range(num_iterations):
            conn.execute("""
                SELECT category, SUM(amount), AVG(score), COUNT(*)
                FROM test
                GROUP BY category
            """).fetchall()
        duckdb_time = (time.perf_counter() - start) / num_iterations * 1000
        results["duckdb"] = duckdb_time
        conn.close()

    return results


def format_result(value):
    """Format benchmark result for display."""
    if isinstance(value, str):
        return value
    return f"{value:.2f} ms"


def main():
    print("=" * 60)
    print("LanceQL vs Polars vs DuckDB Benchmark")
    print("=" * 60)
    print()

    # Test with different data sizes
    sizes = [100_000, 500_000]  # 100K and 500K for comparison

    all_results = {}

    for num_rows in sizes:
        print(f"\n{'='*60}")
        print(f"Dataset: {num_rows:,} rows, 384-dim embeddings")
        print(f"{'='*60}")

        print(f"\nGenerating test data...")
        data, embeddings = generate_test_data(num_rows)
        print(f"Data size: {embeddings.nbytes / 1024 / 1024:.1f} MB embeddings")

        size_results = {}

        # Vector search (LanceQL specialty)
        print(f"\n1. Vector Similarity Search (10 queries, top-10)")
        print("-" * 40)
        vector_results = benchmark_vector_search(data, embeddings, num_queries=10)
        for engine, result in vector_results.items():
            print(f"  {engine:10}: {format_result(result)}")
        size_results["vector_search"] = vector_results

        # Aggregations
        print(f"\n2. Aggregations (SUM, AVG, MAX, COUNT)")
        print("-" * 40)
        agg_results = benchmark_aggregations(data)
        for engine, result in agg_results.items():
            print(f"  {engine:10}: {format_result(result)}")
        size_results["aggregations"] = agg_results

        # Filtering
        print(f"\n3. Filtering (amount > 5000 AND status = 'completed' AND score > 0.5)")
        print("-" * 40)
        filter_results = benchmark_filtering(data)
        for engine, result in filter_results.items():
            print(f"  {engine:10}: {format_result(result)}")
        size_results["filtering"] = filter_results

        # Group by
        print(f"\n4. GROUP BY category")
        print("-" * 40)
        groupby_results = benchmark_group_by(data)
        for engine, result in groupby_results.items():
            print(f"  {engine:10}: {format_result(result)}")
        size_results["group_by"] = groupby_results

        all_results[num_rows] = size_results

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print("""
    Vector Search Comparison:
    - NumPy (with Apple Accelerate): Fast BLAS-optimized matrix ops
    - LanceQL Python (PyTorch MPS): GPU overhead dominates small batches
    - LanceQL Native (Zig + Metal): Zero-copy, ~10x faster (run bench_vector_ops.zig)

    For production vector search:
    - Use LanceQL's native Zig backend (zero Python overhead)
    - IVF-PQ indexing for billion-scale datasets
    - @logic_table compiles Python to native batch code

    Polars/DuckDB excel at:
    - Aggregations, filtering, GROUP BY (columnar optimized)
    - SQL-like query syntax

    LanceQL excels at:
    - Vector similarity search (native GPU/SIMD)
    - Lance format zero-copy reads
    - Embedding-heavy ML workloads
    """)

    # Write JSON results
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
