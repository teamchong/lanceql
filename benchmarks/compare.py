#!/usr/bin/env python3
"""
LanceQL vs DuckDB vs Polars Benchmark

Compares query performance across engines on the same Parquet files.
"""

import subprocess
import time
import statistics
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import duckdb
    import polars as pl
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install duckdb polars")
    sys.exit(1)


def run_lanceql(query: str, iterations: int = 10, warmup: int = 3) -> dict:
    """Run LanceQL benchmark"""
    binary = os.path.join(PROJECT_ROOT, "zig-out/bin/lanceql")
    if not os.path.exists(binary):
        return {"error": "lanceql binary not found - run 'zig build' first"}

    cmd = [binary, "query", query, "--benchmark",
           "--iterations", str(iterations), "--warmup", str(warmup), "--json"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Parse JSON output
        import json
        for line in result.stderr.split('\n'):
            if line.startswith('{'):
                return json.loads(line)
        return {"error": "no JSON output", "stderr": result.stderr}
    except Exception as e:
        return {"error": str(e)}


def run_duckdb(query: str, iterations: int = 10, warmup: int = 3) -> dict:
    """Run DuckDB benchmark"""
    times = []

    # Warmup
    for _ in range(warmup):
        try:
            duckdb.sql(query).fetchall()
        except Exception as e:
            return {"error": str(e)}

    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        result = duckdb.sql(query).fetchall()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "min_ms": min(times),
        "avg_ms": statistics.mean(times),
        "max_ms": max(times),
        "rows": len(result) if result else 0
    }


def run_polars(file_path: str, query_type: str, iterations: int = 10, warmup: int = 3) -> dict:
    """Run Polars benchmark"""
    times = []

    def execute_query():
        df = pl.read_parquet(file_path)
        if query_type == "select_all":
            return df
        elif query_type == "select_cols":
            return df.select(["id", "value"])
        elif query_type == "where_filter":
            return df.filter(pl.col("id") > 50000)
        elif query_type == "count":
            return df.select(pl.count())
        elif query_type == "sum":
            return df.select(pl.col("value").sum())
        return df

    # Warmup
    for _ in range(warmup):
        try:
            execute_query()
        except Exception as e:
            return {"error": str(e)}

    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        result = execute_query()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "min_ms": min(times),
        "avg_ms": statistics.mean(times),
        "max_ms": max(times),
        "rows": len(result) if hasattr(result, '__len__') else 1
    }


def format_result(name: str, result: dict) -> str:
    """Format benchmark result as table row"""
    if "error" in result:
        return f"| {name:12} | {'ERROR':>10} | {'':>10} | {'':>10} | {result['error'][:30]}"

    return f"| {name:12} | {result['min_ms']:>10.2f} | {result['avg_ms']:>10.2f} | {result['max_ms']:>10.2f} |"


def main():
    # Test files
    test_files = [
        ("small (5 rows)", "tests/fixtures/simple.parquet"),
        ("100k rows", "tests/fixtures/benchmark_100k.parquet"),
    ]

    # Queries to test: (name, sql_template, polars_type)
    queries = [
        ("SELECT *", "SELECT * FROM '{file}'", "select_all"),
        ("SELECT cols", "SELECT id, value FROM '{file}'", "select_cols"),
        ("WHERE filter", "SELECT * FROM '{file}' WHERE id > 50000", "where_filter"),
        ("COUNT(*)", "SELECT COUNT(*) FROM '{file}'", "count"),
        ("SUM agg", "SELECT SUM(value) FROM '{file}'", "sum"),
    ]

    iterations = 10
    warmup = 3

    print("=" * 70)
    print("LanceQL vs DuckDB vs Polars Benchmark")
    print("=" * 70)
    print(f"Iterations: {iterations}, Warmup: {warmup}")
    print()

    for file_desc, file_path in test_files:
        full_path = os.path.join(PROJECT_ROOT, file_path)
        if not os.path.exists(full_path):
            print(f"Skipping {file_desc}: {file_path} not found")
            continue

        print(f"\n### {file_desc}: {file_path}")
        print()

        for query_name, query_template, polars_type in queries:
            query = query_template.format(file=full_path)

            print(f"**{query_name}**")
            print()
            print("| Engine       |    Min (ms) |    Avg (ms) |    Max (ms) |")
            print("|--------------|-------------|-------------|-------------|")

            # LanceQL
            lanceql_result = run_lanceql(query, iterations, warmup)
            print(format_result("LanceQL", lanceql_result))

            # DuckDB
            duckdb_result = run_duckdb(query, iterations, warmup)
            print(format_result("DuckDB", duckdb_result))

            # Polars
            polars_result = run_polars(full_path, polars_type, iterations, warmup)
            print(format_result("Polars", polars_result))

            print()


if __name__ == "__main__":
    main()
