#!/usr/bin/env python3
"""
Benchmark: LanceQL vs DuckDB Format Ingest Performance

Compares reading/parsing speed of different data formats:
- CSV, TSV, JSON, JSONL
- Parquet
- Arrow IPC
- Avro
- ORC
- Excel (XLSX)
- Delta Lake
- Iceberg

For each format, measures:
1. DuckDB: Native read performance
2. LanceQL: CLI ingest (metadata parsing)

Run with: python benchmarks/bench_ingest_formats.py
"""

import subprocess
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Check for required dependencies
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    print("WARNING: duckdb not installed. Run: pip install duckdb")
    HAS_DUCKDB = False

try:
    import pyarrow.parquet as pq
    import pyarrow.csv as pa_csv
    import pyarrow.feather as feather
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    print("WARNING: pyarrow not installed. Run: pip install pyarrow")
    HAS_PYARROW = False

# Paths
FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
LANCEQL_CLI = Path(__file__).parent.parent / "zig-out" / "bin" / "lanceql"


def get_file_size(path: str) -> int:
    """Get file or directory size in bytes."""
    if os.path.isdir(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total
    return os.path.getsize(path)


def benchmark_duckdb_read(format_name: str, file_path: str, warmup: int = 2, iterations: int = 5) -> Tuple[float, int]:
    """Benchmark DuckDB reading a format."""
    if not HAS_DUCKDB:
        return 0.0, 0

    conn = duckdb.connect(":memory:")

    try:
        # Build read query based on format
        if format_name == "csv":
            query = f"SELECT COUNT(*) FROM read_csv('{file_path}')"
        elif format_name == "tsv":
            query = f"SELECT COUNT(*) FROM read_csv('{file_path}', delim='\\t')"
        elif format_name == "json":
            query = f"SELECT COUNT(*) FROM read_json('{file_path}')"
        elif format_name == "jsonl":
            query = f"SELECT COUNT(*) FROM read_json('{file_path}', format='newline_delimited')"
        elif format_name == "parquet":
            query = f"SELECT COUNT(*) FROM read_parquet('{file_path}')"
        elif format_name == "arrow":
            # DuckDB doesn't support Arrow IPC streaming format directly
            # Try parquet scanner on .arrows files
            if file_path.endswith('.arrows'):
                return 0.0, 0  # Skip streaming format
            # Try to use INSTALL arrow and read
            try:
                conn.execute("INSTALL arrow; LOAD arrow;")
                query = f"SELECT COUNT(*) FROM '{file_path}'"
            except:
                return 0.0, 0
        elif format_name == "avro":
            # DuckDB doesn't have native Avro support yet
            return 0.0, 0
        elif format_name == "orc":
            # DuckDB doesn't have native ORC support yet
            return 0.0, 0
        elif format_name == "xlsx":
            try:
                conn.execute("INSTALL spatial; LOAD spatial;")
                query = f"SELECT COUNT(*) FROM st_read('{file_path}')"
            except:
                return 0.0, 0
        elif format_name == "delta":
            try:
                conn.execute("INSTALL delta; LOAD delta;")
                query = f"SELECT COUNT(*) FROM delta_scan('{file_path}')"
            except:
                return 0.0, 0
        elif format_name == "iceberg":
            try:
                conn.execute("INSTALL iceberg; LOAD iceberg;")
                query = f"SELECT COUNT(*) FROM iceberg_scan('{file_path}')"
            except:
                return 0.0, 0
        else:
            return 0.0, 0

        # Warmup
        for _ in range(warmup):
            try:
                result = conn.execute(query).fetchone()
            except Exception:
                return 0.0, 0

        # Benchmark
        times = []
        row_count = 0
        for _ in range(iterations):
            start = time.perf_counter()
            result = conn.execute(query).fetchone()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            row_count = result[0] if result else 0

        avg_time = sum(times) / len(times) * 1000  # ms
        conn.close()
        return avg_time, row_count

    except Exception as e:
        print(f"  DuckDB error for {format_name}: {e}")
        return 0.0, 0


def benchmark_lanceql_ingest(format_name: str, file_path: str, warmup: int = 2, iterations: int = 5) -> Tuple[float, int]:
    """Benchmark LanceQL CLI ingest."""
    if not LANCEQL_CLI.exists():
        print(f"  LanceQL CLI not found at {LANCEQL_CLI}")
        return 0.0, 0

    output_path = f"/tmp/bench_{format_name}.lance"
    cmd = [str(LANCEQL_CLI), "ingest", file_path, "-o", output_path]

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
            result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            print(f"  LanceQL error: {e}")
            return 0.0, 0

    avg_time = sum(times) / len(times) * 1000  # ms
    return avg_time, 0


def format_result(time_ms: float, file_size: int, row_count: int = 0) -> str:
    """Format benchmark result with throughput."""
    if time_ms <= 0:
        return "N/A"

    throughput_mb = file_size / (1024 * 1024) / (time_ms / 1000)
    result = f"{time_ms:8.2f} ms ({throughput_mb:6.1f} MB/s)"
    if row_count > 0:
        rows_per_sec = row_count / (time_ms / 1000) / 1000
        result += f" [{row_count:,} rows, {rows_per_sec:.1f}K rows/s]"
    return result


def run_benchmarks():
    """Run all format benchmarks."""
    print("=" * 80)
    print("LanceQL vs DuckDB: Format Ingest Benchmark")
    print("=" * 80)
    print()

    # Define test fixtures
    fixtures = [
        # (format_name, fixture_path, description)
        ("csv", FIXTURES_DIR / "simple.csv", "CSV (comma-separated)"),
        ("parquet", FIXTURES_DIR / "simple.parquet", "Parquet"),
        ("arrow", FIXTURES_DIR / "simple.arrow", "Arrow IPC File"),
        ("avro", FIXTURES_DIR / "simple.avro", "Avro"),
        ("orc", FIXTURES_DIR / "simple.orc", "ORC"),
        ("xlsx", FIXTURES_DIR / "simple_uncompressed.xlsx", "Excel XLSX"),
        ("delta", FIXTURES_DIR / "simple.delta", "Delta Lake"),
        ("iceberg", FIXTURES_DIR / "simple.iceberg", "Apache Iceberg"),
    ]

    results = {}

    for format_name, fixture_path, description in fixtures:
        fixture_str = str(fixture_path)

        if not fixture_path.exists() and not fixture_path.is_dir():
            print(f"\n{description}: SKIP (fixture not found: {fixture_path})")
            continue

        file_size = get_file_size(fixture_str)
        size_kb = file_size / 1024

        print(f"\n{'-'*60}")
        print(f"{description} ({size_kb:.1f} KB)")
        print(f"{'-'*60}")

        # DuckDB
        duckdb_time, duckdb_rows = benchmark_duckdb_read(format_name, fixture_str)
        print(f"  DuckDB:  {format_result(duckdb_time, file_size, duckdb_rows)}")

        # LanceQL
        lanceql_time, _ = benchmark_lanceql_ingest(format_name, fixture_str)
        print(f"  LanceQL: {format_result(lanceql_time, file_size)}")

        # Store results
        results[format_name] = {
            "file_size_bytes": file_size,
            "duckdb_ms": duckdb_time,
            "duckdb_rows": duckdb_rows,
            "lanceql_ms": lanceql_time,
        }

        # Comparison
        if duckdb_time > 0 and lanceql_time > 0:
            ratio = lanceql_time / duckdb_time
            if ratio < 1:
                print(f"  LanceQL is {1/ratio:.1f}x faster")
            else:
                print(f"  DuckDB is {ratio:.1f}x faster")

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print()
    print("Formats tested:")
    for fmt, data in results.items():
        duckdb_status = f"{data['duckdb_ms']:.1f}ms" if data['duckdb_ms'] > 0 else "N/A"
        lanceql_status = f"{data['lanceql_ms']:.1f}ms" if data['lanceql_ms'] > 0 else "N/A"
        print(f"  {fmt:12}: DuckDB={duckdb_status:10}, LanceQL={lanceql_status}")

    print()
    print("Notes:")
    print("  - LanceQL currently reports metadata parsing only (not full data conversion)")
    print("  - DuckDB doesn't support Avro, ORC natively")
    print("  - Arrow IPC streaming format (.arrows) not supported by DuckDB")
    print("  - Delta and Iceberg require DuckDB extensions")

    # Save results
    output_path = Path(__file__).parent / "ingest_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_benchmarks()
