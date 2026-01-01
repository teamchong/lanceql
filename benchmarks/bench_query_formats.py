#!/usr/bin/env python3
"""
Benchmark: LanceQL vs DuckDB SQL Query Performance

Compares SQL query execution speed on different file formats.
Tests common query patterns:
1. COUNT(*) - row counting
2. SELECT * - full scan
3. WHERE filter - predicate pushdown
4. GROUP BY - aggregation
5. ORDER BY - sorting

Run with: python benchmarks/bench_query_formats.py
"""

import subprocess
import time
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Check for required dependencies
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    print("WARNING: duckdb not installed. Run: pip install duckdb")
    HAS_DUCKDB = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.csv as pa_csv
    HAS_PYARROW = True
except ImportError:
    print("WARNING: pyarrow not installed. Run: pip install pyarrow")
    HAS_PYARROW = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("WARNING: numpy not installed. Run: pip install numpy")
    HAS_NUMPY = False

# Paths
LANCEQL_CLI = Path(__file__).parent.parent / "zig-out" / "bin" / "lanceql"


def generate_test_data(num_rows: int = 100_000) -> Dict:
    """Generate test dataset."""
    if not HAS_NUMPY:
        print("ERROR: numpy required for data generation")
        sys.exit(1)

    np.random.seed(42)
    return {
        "id": np.arange(num_rows, dtype=np.int64),
        "category": np.random.choice(["A", "B", "C", "D", "E"], num_rows),
        "amount": np.random.uniform(10, 10000, num_rows).astype(np.float64),
        "quantity": np.random.randint(1, 100, num_rows).astype(np.int64),
        "active": np.random.choice([True, False], num_rows),
    }


def create_test_files(tmpdir: str, data: Dict) -> Dict[str, str]:
    """Create test files in different formats."""
    files = {}

    if HAS_PYARROW:
        # Create Arrow table
        table = pa.table(data)

        # CSV
        csv_path = os.path.join(tmpdir, "data.csv")
        pa_csv.write_csv(table, csv_path)
        files["csv"] = csv_path

        # Parquet
        parquet_path = os.path.join(tmpdir, "data.parquet")
        pq.write_table(table, parquet_path)
        files["parquet"] = parquet_path

        # Lance (via LanceQL CLI)
        lance_path = os.path.join(tmpdir, "data.lance")
        if LANCEQL_CLI.exists():
            subprocess.run(
                [str(LANCEQL_CLI), "ingest", csv_path, "-o", lance_path],
                capture_output=True
            )
            if os.path.exists(lance_path):
                files["lance"] = lance_path

    return files


def benchmark_duckdb_query(query: str, file_path: str, format_name: str,
                           warmup: int = 2, iterations: int = 10) -> float:
    """Benchmark DuckDB query execution."""
    if not HAS_DUCKDB:
        return 0.0

    conn = duckdb.connect(":memory:")

    try:
        # Build the full query with format-specific reader
        if format_name == "csv":
            full_query = query.replace("{table}", f"read_csv('{file_path}')")
        elif format_name == "parquet":
            full_query = query.replace("{table}", f"read_parquet('{file_path}')")
        elif format_name == "lance":
            # DuckDB doesn't support Lance natively
            return 0.0
        else:
            return 0.0

        # Warmup
        for _ in range(warmup):
            conn.execute(full_query).fetchall()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            conn.execute(full_query).fetchall()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        conn.close()
        return sum(times) / len(times) * 1000  # ms

    except Exception as e:
        print(f"  DuckDB error: {e}")
        return 0.0


def benchmark_lanceql_query(query: str, file_path: str,
                            warmup: int = 2, iterations: int = 10) -> float:
    """Benchmark LanceQL CLI query execution."""
    if not LANCEQL_CLI.exists():
        return 0.0

    # Replace {table} placeholder with actual file path
    full_query = query.replace("{table}", f"'{file_path}'")
    cmd = [str(LANCEQL_CLI), "query", full_query]

    # Warmup
    for _ in range(warmup):
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
        except Exception:
            pass

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            print(f"  LanceQL error: {e}")
            return 0.0

    return sum(times) / len(times) * 1000 if times else 0.0  # ms


def run_benchmarks():
    """Run query benchmarks."""
    print("=" * 80)
    print("LanceQL vs DuckDB: SQL Query Benchmark")
    print("=" * 80)
    print()

    # Generate test data
    num_rows = 100_000
    print(f"Generating test data: {num_rows:,} rows...")
    data = generate_test_data(num_rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        print("Creating test files...")
        files = create_test_files(tmpdir, data)

        for fmt, path in files.items():
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {fmt}: {size_mb:.2f} MB")

        # Define queries to benchmark
        queries = [
            ("COUNT(*)", "SELECT COUNT(*) FROM {table}"),
            ("SELECT LIMIT 100", "SELECT * FROM {table} LIMIT 100"),
            ("WHERE filter", "SELECT * FROM {table} WHERE amount > 5000"),
            ("GROUP BY", "SELECT category, SUM(amount) FROM {table} GROUP BY category"),
            ("ORDER BY", "SELECT * FROM {table} ORDER BY amount DESC LIMIT 100"),
            ("Complex", "SELECT category, COUNT(*), AVG(amount) FROM {table} WHERE quantity > 50 GROUP BY category ORDER BY 2 DESC"),
        ]

        results = {}

        for query_name, query_template in queries:
            print(f"\n{'-'*60}")
            print(f"Query: {query_name}")
            print(f"SQL: {query_template}")
            print(f"{'-'*60}")

            results[query_name] = {}

            for fmt, path in files.items():
                # DuckDB
                duckdb_time = benchmark_duckdb_query(query_template, path, fmt)

                # LanceQL (only for supported formats)
                lanceql_time = 0.0
                if fmt in ["lance", "parquet", "csv"]:
                    lanceql_time = benchmark_lanceql_query(query_template, path)

                # Report
                duckdb_str = f"{duckdb_time:.2f}ms" if duckdb_time > 0 else "N/A"
                lanceql_str = f"{lanceql_time:.2f}ms" if lanceql_time > 0 else "N/A"

                print(f"  {fmt:10}: DuckDB={duckdb_str:10}, LanceQL={lanceql_str}")

                results[query_name][fmt] = {
                    "duckdb_ms": duckdb_time,
                    "lanceql_ms": lanceql_time,
                }

                # Comparison
                if duckdb_time > 0 and lanceql_time > 0:
                    ratio = lanceql_time / duckdb_time
                    if ratio < 1:
                        print(f"             LanceQL is {1/ratio:.1f}x faster")
                    else:
                        print(f"             DuckDB is {ratio:.1f}x faster")

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print()
    print("Key observations:")
    print("  - DuckDB is highly optimized for analytical queries")
    print("  - LanceQL CLI includes process startup overhead (~3-10ms)")
    print("  - For fair comparison, use LanceQL as a library, not CLI")
    print("  - LanceQL supports formats DuckDB doesn't (Avro, ORC, Iceberg)")
    print()

    # Save results
    output_path = Path(__file__).parent / "query_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    run_benchmarks()
