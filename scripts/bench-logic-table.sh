#!/bin/bash
# bench-logic-table.sh - @logic_table ML Workflow Benchmark (LanceQL vs DuckDB vs Polars)
#
# Benchmarks:
#   - Feature Engineering (1B rows): normalize, z-score, log transform
#   - Vector Search (10M docs x 384-dim): cosine similarity, euclidean distance
#   - Fraud Detection (500M transactions): multi-factor risk scoring
#   - Recommendations (5M items x 256-dim): collaborative filtering
#   - SQL Clauses (200M rows): SELECT, WHERE, GROUP BY, ORDER BY
#
# Each benchmark runs 30+ seconds.
#
# Usage: ./scripts/bench-logic-table.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "@logic_table ML Workflow Benchmark (LanceQL vs DuckDB vs Polars)"
echo "================================================================================"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Platform: $(uname -s) $(uname -m)"
echo ""

# Check engines
echo "Engines:"
echo "  - LanceQL: native Zig + Metal GPU (compiled @logic_table)"

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

# Check if vector_ops.a exists (required for @logic_table benchmarks)
if [ ! -f "$PROJECT_DIR/lib/vector_ops.a" ]; then
    echo "================================================================================"
    echo "SKIPPED: lib/vector_ops.a not found"
    echo ""
    echo "The @logic_table benchmark requires a compiled Python logic_table library."
    echo "To generate it, run:"
    echo ""
    echo "  cd deps/metal0 && zig build"
    echo "  ./zig-out/bin/metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a"
    echo ""
    echo "Or run benchmarks locally on a machine with metal0 installed."
    echo "================================================================================"
    exit 0
fi

# Build and run
zig build bench-logic-table 2>&1
