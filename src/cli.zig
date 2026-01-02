//! LanceQL CLI - High-performance data pipeline for Lance files
//!
//! Usage:
//!   lanceql query "SELECT * FROM 'data.lance' LIMIT 10"
//!   lanceql ingest data.csv -o out.lance
//!   lanceql transform data.lance --select "a,b"
//!   lanceql enrich data.lance --embed text
//!   lanceql serve data.lance
//!   lanceql (no args) - auto-detect config or serve
//!
//! Designed for apple-to-apple comparison with:
//!   duckdb -c "SELECT * FROM 'data.parquet' LIMIT 10"
//!   polars -c "SELECT * FROM read_parquet('data.parquet') LIMIT 10"

const std = @import("std");
const args = @import("cli/args.zig");
const ingest = @import("cli/ingest.zig");
const enrich = @import("cli/enrich.zig");
const transform = @import("cli/transform.zig");
const serve = @import("cli/serve.zig");
const output = @import("cli/output.zig");
const benchmark = @import("cli/benchmark.zig");
const file_utils = @import("cli/file_utils.zig");
const lanceql = @import("lanceql");
const metal = @import("lanceql.metal");
const Table = @import("lanceql.table").Table;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const DeltaTable = @import("lanceql.delta_table").DeltaTable;
const IcebergTable = @import("lanceql.iceberg_table").IcebergTable;
const ArrowTable = @import("lanceql.arrow_table").ArrowTable;
const AvroTable = @import("lanceql.avro_table").AvroTable;
const OrcTable = @import("lanceql.orc_table").OrcTable;
const XlsxTable = @import("lanceql.xlsx_table").XlsxTable;
const executor = @import("lanceql.sql.executor");
const lexer = @import("lanceql.sql.lexer");
const parser = @import("lanceql.sql.parser");
const ast = @import("lanceql.sql.ast");

/// File type detection
const FileType = enum {
    lance,
    parquet,
    delta,
    iceberg,
    arrow,
    avro,
    orc,
    xlsx,
    unknown,
};

fn detectFileType(path: []const u8, data: []const u8) FileType {
    // Check by extension first
    if (std.mem.endsWith(u8, path, ".parquet")) return .parquet;
    if (std.mem.endsWith(u8, path, ".lance")) return .lance;
    if (std.mem.endsWith(u8, path, ".arrow") or std.mem.endsWith(u8, path, ".arrows") or std.mem.endsWith(u8, path, ".feather")) return .arrow;
    if (std.mem.endsWith(u8, path, ".avro")) return .avro;
    if (std.mem.endsWith(u8, path, ".orc")) return .orc;
    if (std.mem.endsWith(u8, path, ".xlsx")) return .xlsx;

    // Check for Delta directory (has _delta_log/ subdirectory)
    if (std.mem.endsWith(u8, path, ".delta") or isDeltaDirectory(path)) {
        return .delta;
    }

    // Check for Iceberg directory (has metadata/ subdirectory)
    if (std.mem.endsWith(u8, path, ".iceberg") or isIcebergDirectory(path)) {
        return .iceberg;
    }

    // Check magic bytes
    if (data.len >= 6) {
        if (std.mem.eql(u8, data[0..6], "ARROW1")) return .arrow;
    }
    if (data.len >= 4) {
        if (std.mem.eql(u8, data[0..4], "PAR1")) return .parquet;
        if (std.mem.eql(u8, data[0..4], "Obj\x01")) return .avro;
        if (std.mem.eql(u8, data[0..4], "PK\x03\x04")) return .xlsx;
        if (data.len >= 40 and std.mem.eql(u8, data[data.len - 4 ..], "LANC")) return .lance;
    }
    if (data.len >= 3) {
        if (std.mem.eql(u8, data[0..3], "ORC")) return .orc;
    }

    return .unknown;
}

/// Check if path is a Delta Lake table (directory with _delta_log/)
fn isDeltaDirectory(path: []const u8) bool {
    var path_buf: [4096]u8 = undefined;
    const delta_log_path = std.fmt.bufPrint(&path_buf, "{s}/_delta_log", .{path}) catch return false;

    // Try to stat the _delta_log directory
    const stat = std.fs.cwd().statFile(delta_log_path) catch return false;
    return stat.kind == .directory;
}

/// Check if path is an Iceberg table (directory with metadata/)
fn isIcebergDirectory(path: []const u8) bool {
    var path_buf: [4096]u8 = undefined;
    const metadata_path = std.fmt.bufPrint(&path_buf, "{s}/metadata", .{path}) catch return false;

    // Try to stat the metadata directory
    const stat = std.fs.cwd().statFile(metadata_path) catch return false;
    return stat.kind == .directory;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const parsed = try args.parse(allocator);

    // Handle global commands
    switch (parsed.command) {
        .version => {
            std.debug.print("lanceql {s}\n", .{args.version});
            return;
        },
        .help => {
            args.printHelp();
            return;
        },
        .query => {
            if (parsed.global.help or parsed.query.help) {
                args.printQueryHelp();
                return;
            }
            try cmdQuery(allocator, parsed.query);
        },
        .ingest => {
            if (parsed.global.help or parsed.ingest.help) {
                args.printIngestHelp();
                return;
            }
            try cmdIngest(allocator, parsed.ingest);
        },
        .transform => {
            if (parsed.global.help or parsed.transform.help) {
                args.printTransformHelp();
                return;
            }
            try cmdTransform(allocator, parsed.transform);
        },
        .enrich => {
            if (parsed.global.help or parsed.enrich.help) {
                args.printEnrichHelp();
                return;
            }
            try cmdEnrich(allocator, parsed.enrich);
        },
        .serve => {
            if (parsed.global.help or parsed.serve.help) {
                args.printServeHelp();
                return;
            }
            try cmdServe(allocator, parsed.serve);
        },
        .none => {
            // No command - auto-detect mode
            // Check for config file or start serve
            if (parsed.global.config) |config_path| {
                try runConfigFile(allocator, config_path);
            } else if (findConfigFile()) |config_path| {
                try runConfigFile(allocator, config_path);
            } else {
                // Default to help
                args.printHelp();
            }
        },
    }
}

/// Query command - execute SQL on Lance/Parquet files
fn cmdQuery(allocator: std.mem.Allocator, opts: args.QueryOptions) !void {
    // Read query from file if specified
    var query_text = opts.query;
    var file_content: ?[]const u8 = null;

    if (opts.file) |file_path| {
        const f = std.fs.cwd().openFile(file_path, .{}) catch |err| {
            std.debug.print("Error opening file '{s}': {}\n", .{ file_path, err });
            return;
        };
        defer f.close();

        file_content = f.readToEndAlloc(allocator, 1024 * 1024) catch |err| {
            std.debug.print("Error reading file: {}\n", .{err});
            return;
        };
        query_text = file_content;
    }

    defer if (file_content) |fc| allocator.free(fc);

    if (query_text == null) {
        args.printQueryHelp();
        return;
    }

    // Initialize GPU if available
    _ = metal.initGPU();
    defer metal.cleanupGPU();

    const query = query_text.?;

    // Convert QueryOptions to legacy Args for existing functions
    const legacy_args = LegacyArgs{
        .query = query,
        .benchmark = opts.benchmark,
        .iterations = opts.iterations,
        .warmup = opts.warmup,
        .json = opts.json,
        .csv = opts.csv,
    };

    if (opts.benchmark) {
        try benchmark.run(allocator, query, .{
            .iterations = opts.iterations,
            .warmup = opts.warmup,
            .json = opts.json,
        });
    } else {
        try runQuery(allocator, query, legacy_args);
    }
}

/// Ingest command - convert CSV/JSON/Parquet to Lance
fn cmdIngest(allocator: std.mem.Allocator, opts: args.IngestOptions) !void {
    try ingest.run(allocator, opts);
}

/// Transform command - apply transformations to data files
fn cmdTransform(allocator: std.mem.Allocator, opts: args.TransformOptions) !void {
    transform.run(allocator, opts) catch |err| {
        std.debug.print("Transform command failed: {}\n", .{err});
        return err;
    };
}

/// Enrich command - add embeddings and indexes
fn cmdEnrich(allocator: std.mem.Allocator, opts: args.EnrichOptions) !void {
    enrich.run(allocator, opts) catch |err| {
        std.debug.print("Enrich command failed: {}\n", .{err});
        return err;
    };
}

/// Serve command - start interactive web server
fn cmdServe(allocator: std.mem.Allocator, opts: args.ServeOptions) !void {
    serve.run(allocator, opts) catch |err| {
        std.debug.print("Serve command failed: {}\n", .{err});
        return err;
    };
}

/// Run pipeline from config file
fn runConfigFile(allocator: std.mem.Allocator, config_path: []const u8) !void {
    _ = allocator;
    std.debug.print("Error: Config file execution not yet implemented.\n", .{});
    std.debug.print("Config path: {s}\n", .{config_path});
    std.debug.print("\nConfig files will support:\n", .{});
    std.debug.print("  - YAML pipeline definitions\n", .{});
    std.debug.print("  - Multi-step data processing\n", .{});
    std.debug.print("  - Scheduled execution\n", .{});
}

/// Find config file in current directory
fn findConfigFile() ?[]const u8 {
    const config_names = [_][]const u8{
        "lanceql.yaml",
        "lanceql.yml",
        ".lanceqlrc.yaml",
    };

    for (config_names) |name| {
        if (std.fs.cwd().access(name, .{})) |_| {
            return name;
        } else |_| {}
    }
    return null;
}

/// Legacy Args struct for backward compatibility with existing query functions
const LegacyArgs = struct {
    query: ?[]const u8 = null,
    file: ?[]const u8 = null,
    benchmark: bool = false,
    iterations: usize = 10,
    warmup: usize = 3,
    json: bool = false,
    help: bool = false,
    show_version: bool = false,
    csv: bool = false,
};

/// Tokenize and parse a SQL query, returning the parsed statement
fn tokenizeAndParse(allocator: std.mem.Allocator, query: []const u8) !struct { stmt: parser.Statement, tokens: std.ArrayList(lexer.Token) } {
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    errdefer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();
    return .{ .stmt = stmt, .tokens = tokens };
}

/// Output query results in the format specified by legacy_args
fn outputResults(result: *executor.Result, legacy_args: LegacyArgs) void {
    output.outputResults(result, legacy_args.json, legacy_args.csv);
}

/// Execute query on an already-initialized executor and output results
fn executeAndOutput(allocator: std.mem.Allocator, exec: *executor.Executor, query: []const u8, legacy_args: LegacyArgs) !void {
    const parsed = try tokenizeAndParse(allocator, query);
    var tokens = parsed.tokens;
    defer tokens.deinit(allocator);

    var result = try exec.execute(&parsed.stmt.select, &[_]ast.Value{});
    defer result.deinit();

    outputResults(&result, legacy_args);
}



fn runQuery(allocator: std.mem.Allocator, query: []const u8, legacy_args: LegacyArgs) !void {
    // Extract table path from query
    const table_path = file_utils.extractTablePath(query) orelse {
        std.debug.print("Error: Could not extract table path from query\n", .{});
        std.debug.print("Query should be: SELECT ... FROM 'path/to/file.parquet'\n", .{});
        return;
    };

    // Check for Delta first (directory-based, doesn't need to read file data)
    if (isDeltaDirectory(table_path) or std.mem.endsWith(u8, table_path, ".delta")) {
        runDeltaQuery(allocator, table_path, query, legacy_args) catch |err| {
            std.debug.print("Delta query error: {}\n", .{err});
        };
        return;
    }

    // Check for Iceberg (directory-based, doesn't need to read file data)
    if (isIcebergDirectory(table_path) or std.mem.endsWith(u8, table_path, ".iceberg")) {
        runIcebergQuery(allocator, table_path, query, legacy_args) catch |err| {
            std.debug.print("Iceberg query error: {}\n", .{err});
        };
        return;
    }

    // Read file into memory
    const data = file_utils.openFileOrDataset(allocator, table_path) orelse {
        std.debug.print("Error opening '{s}': file not found or unreadable\n", .{table_path});
        return;
    };
    defer allocator.free(data);

    // Detect file type
    const file_type = detectFileType(table_path, data);

    switch (file_type) {
        .parquet => {
            runParquetQuery(allocator, data, query, legacy_args) catch |err| {
                std.debug.print("Parquet query error: {}\n", .{err});
            };
        },
        .lance => {
            runLanceQuery(allocator, data, query, legacy_args) catch |err| {
                std.debug.print("Lance query error: {}\n", .{err});
            };
        },
        .delta => {
            // Should not reach here since we check Delta first, but handle anyway
            runDeltaQuery(allocator, table_path, query, legacy_args) catch |err| {
                std.debug.print("Delta query error: {}\n", .{err});
            };
        },
        .iceberg => {
            // Should not reach here since we check Iceberg first, but handle anyway
            runIcebergQuery(allocator, table_path, query, legacy_args) catch |err| {
                std.debug.print("Iceberg query error: {}\n", .{err});
            };
        },
        .arrow => {
            runArrowQuery(allocator, data, query, legacy_args) catch |err| {
                std.debug.print("Arrow query error: {}\n", .{err});
            };
        },
        .avro => {
            runAvroQuery(allocator, data, query, legacy_args) catch |err| {
                std.debug.print("Avro query error: {}\n", .{err});
            };
        },
        .orc => {
            runOrcQuery(allocator, data, query, legacy_args) catch |err| {
                std.debug.print("ORC query error: {}\n", .{err});
            };
        },
        .xlsx => {
            runXlsxQuery(allocator, data, query, legacy_args) catch |err| {
                std.debug.print("XLSX query error: {}\n", .{err});
            };
        },
        .unknown => {
            // Try Lance first, then Parquet
            runLanceQuery(allocator, data, query, legacy_args) catch {
                runParquetQuery(allocator, data, query, legacy_args) catch |err| {
                    std.debug.print("Query error: {}\n", .{err});
                };
            };
        },
    }
}

fn runParquetQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    var pq_table = try ParquetTable.init(allocator, data);
    defer pq_table.deinit();

    var exec = executor.Executor.initWithParquet(&pq_table, allocator);
    defer exec.deinit();

    try executeAndOutput(allocator, &exec, query, legacy_args);
}

fn runLanceQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    var table = try Table.init(allocator, data);
    defer table.deinit();

    var exec = executor.Executor.init(&table, allocator);
    defer exec.deinit();

    try executeAndOutput(allocator, &exec, query, legacy_args);
}

fn runDeltaQuery(allocator: std.mem.Allocator, path: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    var delta_table = try DeltaTable.init(allocator, path);
    defer delta_table.deinit();

    var exec = executor.Executor.initWithDelta(&delta_table, allocator);
    defer exec.deinit();

    try executeAndOutput(allocator, &exec, query, legacy_args);
}

fn runIcebergQuery(allocator: std.mem.Allocator, path: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    var iceberg_table = try IcebergTable.init(allocator, path);
    defer iceberg_table.deinit();

    if (iceberg_table.numRows() == 0) {
        std.debug.print("Warning: Iceberg table has no data files\n", .{});
        return;
    }

    var exec = executor.Executor.initWithIceberg(&iceberg_table, allocator);
    defer exec.deinit();

    try executeAndOutput(allocator, &exec, query, legacy_args);
}

fn runArrowQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    var arrow_table = try ArrowTable.init(allocator, data);
    defer arrow_table.deinit();

    var exec = executor.Executor.initWithArrow(&arrow_table, allocator);
    defer exec.deinit();

    try executeAndOutput(allocator, &exec, query, legacy_args);
}

fn runAvroQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    var avro_table = try AvroTable.init(allocator, data);
    defer avro_table.deinit();

    var exec = executor.Executor.initWithAvro(&avro_table, allocator);
    defer exec.deinit();

    try executeAndOutput(allocator, &exec, query, legacy_args);
}

fn runOrcQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    var orc_table = try OrcTable.init(allocator, data);
    defer orc_table.deinit();

    var exec = executor.Executor.initWithOrc(&orc_table, allocator);
    defer exec.deinit();

    try executeAndOutput(allocator, &exec, query, legacy_args);
}

fn runXlsxQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    var xlsx_table = try XlsxTable.init(allocator, data);
    defer xlsx_table.deinit();

    var exec = executor.Executor.initWithXlsx(&xlsx_table, allocator);
    defer exec.deinit();

    try executeAndOutput(allocator, &exec, query, legacy_args);
}


