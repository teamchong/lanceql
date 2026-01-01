//! LanceQL CLI Argument Parser
//!
//! Supports subcommand-based architecture:
//!   lanceql query "SELECT ..."
//!   lanceql ingest data.csv -o out.lance
//!   lanceql transform data.lance --select "a,b"
//!   lanceql enrich data.lance --embed text
//!   lanceql serve data.lance
//!   lanceql (no args) - auto-detect config or serve

const std = @import("std");

pub const version = "0.2.0";

/// Subcommands
pub const Command = enum {
    query,
    ingest,
    transform,
    enrich,
    serve,
    help,
    version,
    none, // No command specified - auto-detect mode
};

/// Global options (apply to all commands)
pub const GlobalOptions = struct {
    help: bool = false,
    show_version: bool = false,
    verbose: bool = false,
    config: ?[]const u8 = null, // -c, --config
};

/// Query command options
pub const QueryOptions = struct {
    query: ?[]const u8 = null,
    file: ?[]const u8 = null,
    benchmark: bool = false,
    iterations: usize = 10,
    warmup: usize = 3,
    json: bool = false,
    csv: bool = false,
    help: bool = false,
};

/// Ingest command options
pub const IngestOptions = struct {
    input: ?[]const u8 = null, // Positional: input file/dir
    output: ?[]const u8 = null, // -o, --output
    format: Format = .auto, // --format
    glob: ?[]const u8 = null, // --glob pattern
    delimiter: ?u8 = null, // --delimiter for CSV
    header: bool = true, // --no-header to disable
    schema: ?[]const u8 = null, // --schema file
    help: bool = false,

    pub const Format = enum {
        auto,
        csv,
        tsv,
        json,
        jsonl,
        parquet,
        arrow,
        avro,
        orc,
        xlsx,
        delta,
        iceberg,
    };
};

/// Transform command options
pub const TransformOptions = struct {
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    select: ?[]const u8 = null, // --select "col1,col2"
    filter: ?[]const u8 = null, // --filter "x > 100"
    rename: ?[]const u8 = null, // --rename "old:new"
    cast: ?[]const u8 = null, // --cast "col:type"
    limit: ?usize = null, // --limit N
    help: bool = false,
};

/// Enrich command options
pub const EnrichOptions = struct {
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    embed: ?[]const u8 = null, // --embed column
    model: Model = .minilm, // --model
    index: ?[]const u8 = null, // --index column
    index_type: IndexType = .ivf_pq, // --index-type
    partitions: usize = 256, // --partitions
    help: bool = false,

    pub const Model = enum {
        minilm,
        clip,
    };

    pub const IndexType = enum {
        ivf_pq,
        flat,
    };
};

/// Serve command options
pub const ServeOptions = struct {
    input: ?[]const u8 = null, // Positional: file/dir to serve
    port: u16 = 3000, // --port
    host: []const u8 = "127.0.0.1", // --host
    open: bool = true, // --no-open to disable
    help: bool = false,
};

/// Parsed arguments
pub const Args = struct {
    command: Command,
    global: GlobalOptions,
    query: QueryOptions,
    ingest: IngestOptions,
    transform: TransformOptions,
    enrich: EnrichOptions,
    serve: ServeOptions,
    remaining: []const []const u8, // Unparsed args
};

/// Parse command line arguments
pub fn parse(allocator: std.mem.Allocator) !Args {
    const argv = try std.process.argsAlloc(allocator);
    // Note: Don't free argv - strings are referenced by Args

    var args = Args{
        .command = .none,
        .global = .{},
        .query = .{},
        .ingest = .{},
        .transform = .{},
        .enrich = .{},
        .serve = .{},
        .remaining = &[_][]const u8{},
    };

    if (argv.len < 2) {
        return args; // No arguments = auto-detect mode
    }

    var i: usize = 1;

    // Check for global flags first
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            args.global.help = true;
        } else if (std.mem.eql(u8, arg, "-V") or std.mem.eql(u8, arg, "--version")) {
            args.global.show_version = true;
            args.command = .version;
            return args;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            args.global.verbose = true;
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--config")) {
            i += 1;
            if (i < argv.len) args.global.config = argv[i];
        } else {
            break; // Not a global flag, check for command
        }
    }

    if (i >= argv.len) {
        if (args.global.help) args.command = .help;
        return args;
    }

    // Parse command
    const cmd_str = argv[i];
    if (std.mem.eql(u8, cmd_str, "query") or std.mem.eql(u8, cmd_str, "q")) {
        args.command = .query;
        i += 1;
        try parseQueryOptions(&args.query, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "ingest") or std.mem.eql(u8, cmd_str, "i")) {
        args.command = .ingest;
        i += 1;
        try parseIngestOptions(&args.ingest, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "transform") or std.mem.eql(u8, cmd_str, "t")) {
        args.command = .transform;
        i += 1;
        try parseTransformOptions(&args.transform, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "enrich") or std.mem.eql(u8, cmd_str, "e")) {
        args.command = .enrich;
        i += 1;
        try parseEnrichOptions(&args.enrich, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "serve") or std.mem.eql(u8, cmd_str, "s")) {
        args.command = .serve;
        i += 1;
        try parseServeOptions(&args.serve, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "help")) {
        args.command = .help;
    } else if (std.mem.eql(u8, cmd_str, "version")) {
        args.command = .version;
    } else if (!std.mem.startsWith(u8, cmd_str, "-")) {
        // No recognized command - treat as query (backward compat)
        args.command = .query;
        args.query.query = cmd_str;
        i += 1;
        try parseQueryOptions(&args.query, argv, &i);
    } else {
        // Flags without command - parse as query options for backward compat
        args.command = .query;
        try parseQueryOptions(&args.query, argv, &i);
    }

    return args;
}

fn parseQueryOptions(opts: *QueryOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--command")) {
            i.* += 1;
            if (i.* < argv.len) opts.query = argv[i.*];
        } else if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--file")) {
            i.* += 1;
            if (i.* < argv.len) opts.file = argv[i.*];
        } else if (std.mem.eql(u8, arg, "-b") or std.mem.eql(u8, arg, "--benchmark")) {
            opts.benchmark = true;
        } else if (std.mem.eql(u8, arg, "-i") or std.mem.eql(u8, arg, "--iterations")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.iterations = std.fmt.parseInt(usize, argv[i.*], 10) catch 10;
            }
        } else if (std.mem.eql(u8, arg, "-w") or std.mem.eql(u8, arg, "--warmup")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.warmup = std.fmt.parseInt(usize, argv[i.*], 10) catch 3;
            }
        } else if (std.mem.eql(u8, arg, "--json")) {
            opts.json = true;
        } else if (std.mem.eql(u8, arg, "--csv")) {
            opts.csv = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            // Positional = query string
            if (opts.query == null) opts.query = arg;
        }
    }
}

fn parseIngestOptions(opts: *IngestOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            i.* += 1;
            if (i.* < argv.len) opts.output = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--format")) {
            i.* += 1;
            if (i.* < argv.len) {
                const fmt = argv[i.*];
                if (std.mem.eql(u8, fmt, "csv")) opts.format = .csv
                else if (std.mem.eql(u8, fmt, "tsv")) opts.format = .tsv
                else if (std.mem.eql(u8, fmt, "json")) opts.format = .json
                else if (std.mem.eql(u8, fmt, "jsonl")) opts.format = .jsonl
                else if (std.mem.eql(u8, fmt, "parquet")) opts.format = .parquet
                else if (std.mem.eql(u8, fmt, "arrow")) opts.format = .arrow
                else if (std.mem.eql(u8, fmt, "avro")) opts.format = .avro
                else if (std.mem.eql(u8, fmt, "orc")) opts.format = .orc
                else if (std.mem.eql(u8, fmt, "xlsx") or std.mem.eql(u8, fmt, "excel")) opts.format = .xlsx
                else if (std.mem.eql(u8, fmt, "delta")) opts.format = .delta
                else if (std.mem.eql(u8, fmt, "iceberg")) opts.format = .iceberg;
            }
        } else if (std.mem.eql(u8, arg, "--glob")) {
            i.* += 1;
            if (i.* < argv.len) opts.glob = argv[i.*];
        } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--delimiter")) {
            i.* += 1;
            if (i.* < argv.len and argv[i.*].len > 0) opts.delimiter = argv[i.*][0];
        } else if (std.mem.eql(u8, arg, "--no-header")) {
            opts.header = false;
        } else if (std.mem.eql(u8, arg, "--schema")) {
            i.* += 1;
            if (i.* < argv.len) opts.schema = argv[i.*];
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            // Positional = input file
            if (opts.input == null) opts.input = arg;
        }
    }
}

fn parseTransformOptions(opts: *TransformOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            i.* += 1;
            if (i.* < argv.len) opts.output = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--select")) {
            i.* += 1;
            if (i.* < argv.len) opts.select = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--filter")) {
            i.* += 1;
            if (i.* < argv.len) opts.filter = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--rename")) {
            i.* += 1;
            if (i.* < argv.len) opts.rename = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--cast")) {
            i.* += 1;
            if (i.* < argv.len) opts.cast = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--limit")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.limit = std.fmt.parseInt(usize, argv[i.*], 10) catch null;
            }
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (opts.input == null) opts.input = arg;
        }
    }
}

fn parseEnrichOptions(opts: *EnrichOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            i.* += 1;
            if (i.* < argv.len) opts.output = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--embed")) {
            i.* += 1;
            if (i.* < argv.len) opts.embed = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--model")) {
            i.* += 1;
            if (i.* < argv.len) {
                const m = argv[i.*];
                if (std.mem.eql(u8, m, "minilm")) opts.model = .minilm
                else if (std.mem.eql(u8, m, "clip")) opts.model = .clip;
            }
        } else if (std.mem.eql(u8, arg, "--index")) {
            i.* += 1;
            if (i.* < argv.len) opts.index = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--index-type")) {
            i.* += 1;
            if (i.* < argv.len) {
                const t = argv[i.*];
                if (std.mem.eql(u8, t, "ivf-pq")) opts.index_type = .ivf_pq
                else if (std.mem.eql(u8, t, "flat")) opts.index_type = .flat;
            }
        } else if (std.mem.eql(u8, arg, "--partitions")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.partitions = std.fmt.parseInt(usize, argv[i.*], 10) catch 256;
            }
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (opts.input == null) opts.input = arg;
        }
    }
}

fn parseServeOptions(opts: *ServeOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--port")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.port = std.fmt.parseInt(u16, argv[i.*], 10) catch 3000;
            }
        } else if (std.mem.eql(u8, arg, "--host")) {
            i.* += 1;
            if (i.* < argv.len) opts.host = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--no-open")) {
            opts.open = false;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (opts.input == null) opts.input = arg;
        }
    }
}

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
