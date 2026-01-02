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
        try runBenchmark(allocator, query, legacy_args);
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
    _ = allocator;
    _ = opts;
    std.debug.print("Error: Serve command not yet implemented.\n", .{});
    std.debug.print("\nServe will provide:\n", .{});
    std.debug.print("  - REST API for SQL queries\n", .{});
    std.debug.print("  - Vector search endpoints\n", .{});
    std.debug.print("  - Web UI for data exploration\n", .{});
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


/// Extract table path from SQL query (finds 'path' in FROM clause)
fn extractTablePath(query: []const u8) ?[]const u8 {
    // Simple extraction: find FROM 'path' or FROM "path"
    const from_pos = std.mem.indexOf(u8, query, "FROM ") orelse
        std.mem.indexOf(u8, query, "from ") orelse return null;

    const after_from = query[from_pos + 5 ..];

    // Skip whitespace
    var start: usize = 0;
    while (start < after_from.len and (after_from[start] == ' ' or after_from[start] == '\t')) {
        start += 1;
    }

    if (start >= after_from.len) return null;

    // Check for quoted path
    const quote_char = after_from[start];
    if (quote_char == '\'' or quote_char == '"') {
        const path_start = start + 1;
        const path_end = std.mem.indexOfScalarPos(u8, after_from, path_start, quote_char) orelse return null;
        return after_from[path_start..path_end];
    }

    // Unquoted identifier
    var end = start;
    while (end < after_from.len and after_from[end] != ' ' and after_from[end] != '\t' and
        after_from[end] != '\n' and after_from[end] != ';' and after_from[end] != ')') {
        end += 1;
    }

    return after_from[start..end];
}

/// Open a file or Lance dataset directory and return its contents
fn openFileOrDataset(allocator: std.mem.Allocator, path: []const u8) ?[]const u8 {
    // Check if path is a file or directory
    const stat = std.fs.cwd().statFile(path) catch {
        // Try as directory - read all .lance fragments
        return readLanceDataset(allocator, path);
    };

    if (stat.kind == .directory) {
        // It's a directory, try to open as Lance dataset
        return readLanceDataset(allocator, path);
    }

    // It's a file, open it directly
    var file = std.fs.cwd().openFile(path, .{}) catch return null;
    defer file.close();

    return file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch null;
}

/// Read all .lance fragments from a Lance dataset directory
fn readLanceDataset(allocator: std.mem.Allocator, path: []const u8) ?[]const u8 {
    var data_path_buf: [4096]u8 = undefined;
    const data_path = std.fmt.bufPrint(&data_path_buf, "{s}/data", .{path}) catch return null;

    var data_dir = std.fs.cwd().openDir(data_path, .{ .iterate = true }) catch return null;
    defer data_dir.close();

    // Collect all .lance files
    var lance_files = std.ArrayList([]const u8){};
    defer {
        for (lance_files.items) |name| allocator.free(name);
        lance_files.deinit(allocator);
    }

    var iter = data_dir.iterate();
    while (iter.next() catch null) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.endsWith(u8, entry.name, ".lance")) {
            lance_files.append(allocator, allocator.dupe(u8, entry.name) catch continue) catch continue;
        }
    }

    if (lance_files.items.len == 0) return null;

    // Sort by filename (0.lance, 1.lance, 2.lance, etc.)
    std.mem.sort([]const u8, lance_files.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            // Extract numeric prefix for proper sorting
            const a_num = extractFragmentNumber(a);
            const b_num = extractFragmentNumber(b);
            if (a_num != null and b_num != null) {
                return a_num.? < b_num.?;
            }
            return std.mem.lessThan(u8, a, b);
        }
        fn extractFragmentNumber(name: []const u8) ?u64 {
            // Parse "0.lance", "1.lance", etc.
            const dot_pos = std.mem.indexOf(u8, name, ".") orelse return null;
            return std.fmt.parseInt(u64, name[0..dot_pos], 10) catch null;
        }
    }.lessThan);

    // Read and concatenate all fragments
    var combined = std.ArrayList(u8){};
    for (lance_files.items) |name| {
        var file = data_dir.openFile(name, .{}) catch continue;
        defer file.close();
        const content = file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch continue;
        defer allocator.free(content);
        combined.appendSlice(allocator, content) catch continue;
    }

    if (combined.items.len == 0) return null;
    return combined.toOwnedSlice(allocator) catch null;
}

fn runQuery(allocator: std.mem.Allocator, query: []const u8, legacy_args: LegacyArgs) !void {
    // Extract table path from query
    const table_path = extractTablePath(query) orelse {
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
    const data = openFileOrDataset(allocator, table_path) orelse {
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
    // Initialize Parquet Table
    var pq_table = ParquetTable.init(allocator, data) catch |err| {
        return err;
    };
    defer pq_table.deinit();

    // Tokenize
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();

    // Execute using Parquet-aware executor
    var exec = executor.Executor.initWithParquet(&pq_table, allocator);
    defer exec.deinit();

    var result = try exec.execute(&stmt.select, &[_]ast.Value{});
    defer result.deinit();

    // Output results
    if (legacy_args.json) {
        printResultsJson(&result);
    } else if (legacy_args.csv) {
        printResultsCsv(&result);
    } else {
        printResultsTable(&result);
    }
}

fn runLanceQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    // Initialize Lance Table
    var table = Table.init(allocator, data) catch |err| {
        return err;
    };
    defer table.deinit();

    // Tokenize
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();

    // Execute
    var exec = executor.Executor.init(&table, allocator);
    defer exec.deinit();

    var result = try exec.execute(&stmt.select, &[_]ast.Value{});
    defer result.deinit();

    // Output results
    if (legacy_args.json) {
        printResultsJson(&result);
    } else if (legacy_args.csv) {
        printResultsCsv(&result);
    } else {
        printResultsTable(&result);
    }
}

fn runDeltaQuery(allocator: std.mem.Allocator, path: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    // Initialize Delta Table (takes directory path, not file data)
    var delta_table = DeltaTable.init(allocator, path) catch |err| {
        return err;
    };
    defer delta_table.deinit();

    // Tokenize
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();

    // Execute using Delta-aware executor
    var exec = executor.Executor.initWithDelta(&delta_table, allocator);
    defer exec.deinit();

    var result = try exec.execute(&stmt.select, &[_]ast.Value{});
    defer result.deinit();

    // Output results
    if (legacy_args.json) {
        printResultsJson(&result);
    } else if (legacy_args.csv) {
        printResultsCsv(&result);
    } else {
        printResultsTable(&result);
    }
}

fn runIcebergQuery(allocator: std.mem.Allocator, path: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    // Initialize Iceberg Table (takes directory path, not file data)
    var iceberg_table = IcebergTable.init(allocator, path) catch |err| {
        return err;
    };
    defer iceberg_table.deinit();

    // Check if table has data
    if (iceberg_table.numRows() == 0) {
        std.debug.print("Warning: Iceberg table has no data files\n", .{});
        return;
    }

    // Tokenize
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();

    // Execute using Iceberg-aware executor
    var exec = executor.Executor.initWithIceberg(&iceberg_table, allocator);
    defer exec.deinit();

    var result = try exec.execute(&stmt.select, &[_]ast.Value{});
    defer result.deinit();

    // Output results
    if (legacy_args.json) {
        printResultsJson(&result);
    } else if (legacy_args.csv) {
        printResultsCsv(&result);
    } else {
        printResultsTable(&result);
    }
}

fn runArrowQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    // Initialize Arrow Table
    var arrow_table = ArrowTable.init(allocator, data) catch |err| {
        return err;
    };
    defer arrow_table.deinit();

    // Tokenize
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();

    // Execute using Arrow-aware executor
    var exec = executor.Executor.initWithArrow(&arrow_table, allocator);
    defer exec.deinit();

    var result = try exec.execute(&stmt.select, &[_]ast.Value{});
    defer result.deinit();

    // Output results
    if (legacy_args.json) {
        printResultsJson(&result);
    } else if (legacy_args.csv) {
        printResultsCsv(&result);
    } else {
        printResultsTable(&result);
    }
}

fn runAvroQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    // Initialize Avro Table
    var avro_table = AvroTable.init(allocator, data) catch |err| {
        return err;
    };
    defer avro_table.deinit();

    // Tokenize
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();

    // Execute using Avro-aware executor
    var exec = executor.Executor.initWithAvro(&avro_table, allocator);
    defer exec.deinit();

    var result = try exec.execute(&stmt.select, &[_]ast.Value{});
    defer result.deinit();

    // Output results
    if (legacy_args.json) {
        printResultsJson(&result);
    } else if (legacy_args.csv) {
        printResultsCsv(&result);
    } else {
        printResultsTable(&result);
    }
}

fn runOrcQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    // Initialize ORC Table
    var orc_table = OrcTable.init(allocator, data) catch |err| {
        return err;
    };
    defer orc_table.deinit();

    // Tokenize
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();

    // Execute using ORC-aware executor
    var exec = executor.Executor.initWithOrc(&orc_table, allocator);
    defer exec.deinit();

    var result = try exec.execute(&stmt.select, &[_]ast.Value{});
    defer result.deinit();

    // Output results
    if (legacy_args.json) {
        printResultsJson(&result);
    } else if (legacy_args.csv) {
        printResultsCsv(&result);
    } else {
        printResultsTable(&result);
    }
}

fn runXlsxQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    // Initialize XLSX Table
    var xlsx_table = XlsxTable.init(allocator, data) catch |err| {
        return err;
    };
    defer xlsx_table.deinit();

    // Tokenize
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();

    // Execute using XLSX-aware executor
    var exec = executor.Executor.initWithXlsx(&xlsx_table, allocator);
    defer exec.deinit();

    var result = try exec.execute(&stmt.select, &[_]ast.Value{});
    defer result.deinit();

    // Output results
    if (legacy_args.json) {
        printResultsJson(&result);
    } else if (legacy_args.csv) {
        printResultsCsv(&result);
    } else {
        printResultsTable(&result);
    }
}

fn printResultsTable(result: *executor.Result) void {
    // Print header
    for (result.columns, 0..) |col, i| {
        if (i > 0) std.debug.print("\t", .{});
        std.debug.print("{s}", .{col.name});
    }
    std.debug.print("\n", .{});

    // Print rows
    for (0..result.row_count) |row| {
        for (result.columns, 0..) |col, i| {
            if (i > 0) std.debug.print("\t", .{});
            printValue(col.data, row);
        }
        std.debug.print("\n", .{});
    }
}

fn printResultsCsv(result: *executor.Result) void {
    // Print header
    for (result.columns, 0..) |col, i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print("{s}", .{col.name});
    }
    std.debug.print("\n", .{});

    // Print rows
    for (0..result.row_count) |row| {
        for (result.columns, 0..) |col, i| {
            if (i > 0) std.debug.print(",", .{});
            printValue(col.data, row);
        }
        std.debug.print("\n", .{});
    }
}

fn printResultsJson(result: *executor.Result) void {
    std.debug.print("[", .{});
    for (0..result.row_count) |row| {
        if (row > 0) std.debug.print(",", .{});
        std.debug.print("{{", .{});
        for (result.columns, 0..) |col, i| {
            if (i > 0) std.debug.print(",", .{});
            std.debug.print("\"{s}\":", .{col.name});
            printValueJson(col.data, row);
        }
        std.debug.print("}}", .{});
    }
    std.debug.print("]\n", .{});
}

fn printValue(data: executor.Result.ColumnData, row: usize) void {
    switch (data) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .int32, .date32 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .float64 => |arr| {
            std.debug.print("{d:.6}", .{arr[row]});
        },
        .float32 => |arr| {
            std.debug.print("{d:.6}", .{arr[row]});
        },
        .bool_ => |arr| {
            std.debug.print("{}", .{arr[row]});
        },
        .string => |arr| {
            std.debug.print("{s}", .{arr[row]});
        },
    }
}

fn printValueJson(data: executor.Result.ColumnData, row: usize) void {
    switch (data) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .int32, .date32 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .float64 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .float32 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .bool_ => |arr| {
            std.debug.print("{}", .{arr[row]});
        },
        .string => |arr| {
            std.debug.print("\"{s}\"", .{arr[row]});
        },
    }
}

fn runBenchmark(allocator: std.mem.Allocator, query: []const u8, legacy_args: LegacyArgs) !void {
    // Extract table path from query
    const table_path = extractTablePath(query) orelse {
        std.debug.print("Error: Could not extract table path from query\n", .{});
        return;
    };

    // Read file into memory
    const data = openFileOrDataset(allocator, table_path) orelse {
        std.debug.print("Error opening '{s}': file not found or unreadable\n", .{table_path});
        return;
    };
    defer allocator.free(data);

    // Initialize Table
    var table = Table.init(allocator, data) catch |err| {
        std.debug.print("Error parsing '{s}': {}\n", .{ table_path, err });
        return;
    };
    defer table.deinit();

    // Tokenize
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = lex.nextToken() catch |err| {
            std.debug.print("Lexer error: {}\n", .{err});
            return;
        };
        tokens.append(allocator, tok) catch {
            std.debug.print("Error: out of memory during tokenization\n", .{});
            return;
        };
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = parse.parseStatement() catch |err| {
        std.debug.print("Parse error: {}\n", .{err});
        return;
    };

    // Get column count
    const num_rows = table.numColumns();

    std.debug.print("LanceQL Benchmark\n", .{});
    std.debug.print("=================\n", .{});
    std.debug.print("Query: {s}\n", .{query});
    std.debug.print("Table: {s} ({d} columns)\n", .{ table_path, num_rows });
    std.debug.print("Warmup: {d}, Iterations: {d}\n\n", .{ legacy_args.warmup, legacy_args.iterations });

    // Warmup
    for (0..legacy_args.warmup) |_| {
        var exec = executor.Executor.init(&table, allocator);
        var result = exec.execute(&stmt.select, &[_]ast.Value{}) catch continue;
        result.deinit();
        exec.deinit();
    }

    // Benchmark
    var times = try allocator.alloc(u64, legacy_args.iterations);
    defer allocator.free(times);

    for (0..legacy_args.iterations) |i| {
        var timer = try std.time.Timer.start();
        var exec = executor.Executor.init(&table, allocator);
        var result = exec.execute(&stmt.select, &[_]ast.Value{}) catch {
            times[i] = 0;
            exec.deinit();
            continue;
        };
        times[i] = timer.read();
        result.deinit();
        exec.deinit();
    }

    // Calculate stats
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;
    var total_ns: u64 = 0;

    for (times) |t| {
        if (t == 0) continue;
        min_ns = @min(min_ns, t);
        max_ns = @max(max_ns, t);
        total_ns += t;
    }

    const avg_ns = total_ns / legacy_args.iterations;
    const avg_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000;
    const min_ms = @as(f64, @floatFromInt(min_ns)) / 1_000_000;
    const max_ms = @as(f64, @floatFromInt(max_ns)) / 1_000_000;
    const throughput = @as(f64, @floatFromInt(num_rows)) / avg_ms / 1000;

    if (legacy_args.json) {
        std.debug.print(
            \\{{"query": "{s}", "columns": {d}, "min_ms": {d:.3}, "avg_ms": {d:.3}, "max_ms": {d:.3}, "throughput_mrows_sec": {d:.2}}}
            \\
        , .{ query, num_rows, min_ms, avg_ms, max_ms, throughput });
    } else {
        std.debug.print("Results:\n", .{});
        std.debug.print("  Columns:    {d}\n", .{num_rows});
        std.debug.print("  Min:        {d:.2} ms\n", .{min_ms});
        std.debug.print("  Avg:        {d:.2} ms\n", .{avg_ms});
        std.debug.print("  Max:        {d:.2} ms\n", .{max_ms});
        std.debug.print("  Throughput: {d:.1}M rows/sec\n", .{throughput});
    }
}
