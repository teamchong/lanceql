#!/bin/bash
# bench-sql.sh - SQL Clause Benchmark (LanceQL vs DuckDB vs Polars)
#
# Benchmarks: SELECT *, WHERE, GROUP BY, ORDER BY LIMIT, DISTINCT, VECTOR SEARCH, HASH JOIN
# Dataset: 200M rows
# Each benchmark runs 30+ seconds.
#
# Usage: ./scripts/bench-sql.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "SQL Clause Benchmark (LanceQL vs DuckDB vs Polars)"
echo "================================================================================"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Platform: $(uname -s) $(uname -m)"
echo ""

# Check engines
echo "Engines:"
echo "  - LanceQL: native Zig + Metal GPU"

if command -v duckdb &> /dev/null; then
    echo "  - DuckDB: $(duckdb --version 2>/dev/null | head -1 || echo 'available')"
else
    echo "  - DuckDB: not installed (brew install duckdb)"
fi

if python3 -c "import polars" 2>/dev/null; then
    echo "  - Polars: $(python3 -c 'import polars; print(polars.__version__)')"
else
    echo "  - Polars: not installed (pip install polars)"
fi
echo ""

# Build and run
zig build bench-sql 2>&1
