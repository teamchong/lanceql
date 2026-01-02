//! LanceQL CLI Help Messages
//!
//! All command help output functions.

const std = @import("std");

/// Print main help
pub fn printHelp() void {
    std.debug.print(
        \\LanceQL - High-performance data pipeline for Lance files
        \\
        \\Usage: lanceql [command] [options]
        \\
        \\Commands:
        \\  query, q      Execute SQL query on Lance/Parquet files
        \\  ingest, i     Convert CSV/JSON/Parquet to Lance format
        \\  transform, t  Transform Lance data (select, filter, rename)
        \\  enrich, e     Add embeddings and indexes to Lance data
        \\  serve, s      Start interactive web server with WebGPU UI
        \\  help          Show this help message
        \\  version       Show version
        \\
        \\Global Options:
        \\  -c, --config <file>  Use config file (YAML)
        \\  -v, --verbose        Enable verbose output
        \\  -h, --help           Show help
        \\  -V, --version        Show version
        \\
        \\Examples:
        \\  lanceql query "SELECT * FROM 'data.lance' LIMIT 10"
        \\  lanceql ingest data.csv -o dataset.lance
        \\  lanceql enrich dataset.lance --embed text --model minilm
        \\  lanceql serve dataset.lance
        \\  lanceql                    # Auto-detect: config or serve
        \\
        \\Run 'lanceql <command> --help' for command-specific help.
        \\
    , .{});
}

/// Print query command help
pub fn printQueryHelp() void {
    std.debug.print(
        \\Usage: lanceql query [options] "SQL QUERY"
        \\
        \\Execute SQL queries on Lance and Parquet files.
        \\
        \\Options:
        \\  -c, --command <SQL>    Execute SQL query
        \\  -f, --file <PATH>      Read SQL from file
        \\  -b, --benchmark        Run query in benchmark mode
        \\  -i, --iterations <N>   Benchmark iterations (default: 10)
        \\  -w, --warmup <N>       Warmup iterations (default: 3)
        \\      --json             Output results as JSON
        \\      --csv              Output results as CSV
        \\  -h, --help             Show this help
        \\
        \\Examples:
        \\  lanceql query "SELECT * FROM 'users.lance' WHERE age > 25"
        \\  lanceql query -b "SELECT COUNT(*) FROM 'data.parquet'"
        \\  lanceql query -f query.sql --json
        \\
    , .{});
}

/// Print ingest command help
pub fn printIngestHelp() void {
    std.debug.print(
        \\Usage: lanceql ingest <input> -o <output.lance> [options]
        \\
        \\Convert data files to Lance format.
        \\
        \\Supported formats (auto-detected from extension):
        \\  csv, tsv          Delimiter-separated values
        \\  json, jsonl       JSON array or newline-delimited JSON
        \\  parquet           Apache Parquet columnar format
        \\  arrow             Apache Arrow IPC/Feather (.arrow, .arrows, .feather)
        \\  avro              Apache Avro container format
        \\  orc               Apache ORC columnar format
        \\  xlsx              Microsoft Excel (uncompressed)
        \\  delta             Delta Lake table (directory)
        \\  iceberg           Apache Iceberg table (directory)
        \\
        \\Options:
        \\  -o, --output <PATH>    Output Lance file (required)
        \\      --format <FMT>     Override auto-detected format
        \\      --glob <PATTERN>   Glob pattern for directory input (e.g., "*.csv")
        \\  -d, --delimiter <C>    CSV delimiter character (default: auto-detect)
        \\      --no-header        CSV has no header row
        \\      --schema <FILE>    Schema file (JSON)
        \\  -h, --help             Show this help
        \\
        \\Examples:
        \\  lanceql ingest data.csv -o dataset.lance
        \\  lanceql ingest data.json --format jsonl -o dataset.lance
        \\  lanceql ingest data.parquet -o dataset.lance
        \\  lanceql ingest data.arrow -o dataset.lance
        \\  lanceql ingest ./my_delta_table/ --format delta -o dataset.lance
        \\
    , .{});
}

/// Print serve command help
pub fn printServeHelp() void {
    std.debug.print(
        \\Usage: lanceql serve [input] [options]
        \\
        \\Start interactive web server with WebGPU-powered UI.
        \\
        \\Options:
        \\  -p, --port <N>         Server port (default: 3000)
        \\      --host <ADDR>      Host address (default: 127.0.0.1)
        \\      --no-open          Don't auto-open browser
        \\  -h, --help             Show this help
        \\
        \\Features:
        \\  - Infinite scroll table with WebGPU acceleration
        \\  - SQL editor with syntax highlighting
        \\  - Vector search with embedding preview
        \\  - Auto-embedding generation
        \\  - Timeline/version navigation
        \\
        \\Examples:
        \\  lanceql serve dataset.lance
        \\  lanceql serve ./datasets/ --port 8080
        \\  lanceql serve                          # Serve current directory
        \\
    , .{});
}

/// Print transform command help
pub fn printTransformHelp() void {
    std.debug.print(
        \\Usage: lanceql transform <input> -o <output> [options]
        \\
        \\Apply transformations to Lance data.
        \\
        \\NOTE: This command is not yet implemented.
        \\      Use 'lanceql query' with SQL for transformations.
        \\
        \\Planned operations:
        \\  - Column projection and renaming
        \\  - Row filtering
        \\  - Data type conversions
        \\  - Custom expressions
        \\
        \\Options:
        \\  -o, --output <PATH>    Output file (required)
        \\      --select <COLS>    Columns to select
        \\      --filter <EXPR>    Filter expression
        \\  -h, --help             Show this help
        \\
    , .{});
}

/// Print enrich command help
pub fn printEnrichHelp() void {
    std.debug.print(
        \\Usage: lanceql enrich <input> [options]
        \\
        \\Add embeddings and indexes to Lance data.
        \\
        \\NOTE: This command is not yet implemented.
        \\
        \\Planned operations:
        \\  - Text embedding generation
        \\  - Vector index creation (IVF, HNSW)
        \\  - Full-text search indexing
        \\
        \\Options:
        \\      --embed <COLUMN>   Column to embed
        \\      --model <NAME>     Embedding model
        \\      --index <TYPE>     Index type (ivf, hnsw)
        \\  -h, --help             Show this help
        \\
    , .{});
}
