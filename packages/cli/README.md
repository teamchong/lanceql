# @metal0/lanceql

Native query interface for Lance and Parquet files. Designed for apple-to-apple comparison with DuckDB and Polars.

## Installation

```bash
npm install -g @metal0/lanceql
```

## Usage

```bash
# Query Lance files
lanceql "SELECT * FROM 'data.lance' LIMIT 10"

# Query Parquet files
lanceql "SELECT * FROM 'data.parquet' WHERE x > 100"

# Read query from file
lanceql -f query.sql

# Benchmark mode
lanceql --benchmark "SELECT * FROM 'data.lance' WHERE x > 100"

# Output as JSON
lanceql --json "SELECT * FROM 'data.lance' LIMIT 5"

# Output as CSV
lanceql --csv "SELECT * FROM 'data.lance' LIMIT 5"
```

## Comparison with DuckDB/Polars

```bash
# LanceQL
lanceql "SELECT * FROM 'data.lance' LIMIT 10"

# DuckDB (equivalent)
duckdb -c "SELECT * FROM 'data.parquet' LIMIT 10"

# Polars (equivalent)
polars -c "SELECT * FROM read_parquet('data.parquet') LIMIT 10"
```

## Features

- Native Lance file format support
- Parquet file support
- SQL query interface
- GPU acceleration on macOS (Metal)
- Benchmark mode for performance testing
- JSON and CSV output formats

## Building from Source

```bash
# Clone the repository
git clone https://github.com/teamchong/lanceql.git
cd lanceql

# Build the CLI
zig build cli

# Binary is at zig-out/bin/lanceql
```

## License

MIT
