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
const lanceql = @import("lanceql");
const metal = @import("lanceql.metal");
const Table = @import("lanceql.table").Table;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const executor = @import("lanceql.sql.executor");
const lexer = @import("lanceql.sql.lexer");
const parser = @import("lanceql.sql.parser");
const ast = @import("lanceql.sql.ast");

/// File type detection
const FileType = enum {
    lance,
    parquet,
    unknown,
};

fn detectFileType(path: []const u8, data: []const u8) FileType {
    // Check by extension first
    if (std.mem.endsWith(u8, path, ".parquet")) return .parquet;
    if (std.mem.endsWith(u8, path, ".lance")) return .lance;

    // Check magic bytes
    if (data.len >= 4) {
        if (std.mem.eql(u8, data[0..4], "PAR1")) return .parquet;
        if (data.len >= 40 and std.mem.eql(u8, data[data.len - 4 ..], "LANC")) return .lance;
    }

    return .unknown;
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
                // TODO: printTransformHelp()
                args.printHelp();
                return;
            }
            try cmdTransform(allocator, parsed.transform);
        },
        .enrich => {
            if (parsed.global.help or parsed.enrich.help) {
                // TODO: printEnrichHelp()
                args.printHelp();
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

/// Transform command - apply transformations to Lance data
fn cmdTransform(allocator: std.mem.Allocator, opts: args.TransformOptions) !void {
    _ = allocator;
    if (opts.input == null) {
        std.debug.print("Error: Input file required.\n", .{});
        return;
    }

    std.debug.print("Transform: {s}\n", .{opts.input.?});
    std.debug.print("Note: Transform command not yet implemented.\n", .{});
}

/// Enrich command - add embeddings and indexes
fn cmdEnrich(allocator: std.mem.Allocator, opts: args.EnrichOptions) !void {
    _ = allocator;
    if (opts.input == null) {
        std.debug.print("Error: Input file required.\n", .{});
        return;
    }

    std.debug.print("Enrich: {s}\n", .{opts.input.?});
    std.debug.print("Note: Enrich command not yet implemented.\n", .{});
}

/// Serve command - start interactive web server
fn cmdServe(allocator: std.mem.Allocator, opts: args.ServeOptions) !void {
    _ = allocator;
    std.debug.print("Starting server on {s}:{d}...\n", .{ opts.host, opts.port });
    if (opts.input) |input| {
        std.debug.print("Serving: {s}\n", .{input});
    }
    std.debug.print("Note: Serve command not yet implemented.\n", .{});
}

/// Run pipeline from config file
fn runConfigFile(allocator: std.mem.Allocator, config_path: []const u8) !void {
    _ = allocator;
    std.debug.print("Loading config: {s}\n", .{config_path});
    std.debug.print("Note: Config file execution not yet implemented.\n", .{});
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
        // Try as directory
        var data_path_buf: [4096]u8 = undefined;
        const data_path = std.fmt.bufPrint(&data_path_buf, "{s}/data", .{path}) catch return null;

        var data_dir = std.fs.cwd().openDir(data_path, .{ .iterate = true }) catch return null;
        defer data_dir.close();

        // Find first .lance file in data directory
        var iter = data_dir.iterate();
        while (iter.next() catch null) |entry| {
            if (entry.kind != .file) continue;
            if (std.mem.endsWith(u8, entry.name, ".lance")) {
                var file = data_dir.openFile(entry.name, .{}) catch continue;
                defer file.close();
                return file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch null;
            }
        }
        return null;
    };

    if (stat.kind == .directory) {
        // It's a directory, try to open as Lance dataset
        var data_path_buf: [4096]u8 = undefined;
        const data_path = std.fmt.bufPrint(&data_path_buf, "{s}/data", .{path}) catch return null;

        var data_dir = std.fs.cwd().openDir(data_path, .{ .iterate = true }) catch return null;
        defer data_dir.close();

        // Find first .lance file in data directory
        var iter = data_dir.iterate();
        while (iter.next() catch null) |entry| {
            if (entry.kind != .file) continue;
            if (std.mem.endsWith(u8, entry.name, ".lance")) {
                var file = data_dir.openFile(entry.name, .{}) catch continue;
                defer file.close();
                return file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch null;
            }
        }
        return null;
    }

    // It's a file, open it directly
    var file = std.fs.cwd().openFile(path, .{}) catch return null;
    defer file.close();

    return file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch null;
}

fn runQuery(allocator: std.mem.Allocator, query: []const u8, legacy_args: LegacyArgs) !void {
    // Extract table path from query
    const table_path = extractTablePath(query) orelse {
        std.debug.print("Error: Could not extract table path from query\n", .{});
        std.debug.print("Query should be: SELECT ... FROM 'path/to/file.parquet'\n", .{});
        return;
    };

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

fn runParquetQuery(allocator: std.mem.Allocator, data: []const u8, query: []const u8, legacy_args: LegacyArgs) !void {
    _ = query; // TODO: Parse and execute full SQL

    // Initialize Parquet Table
    var pq_table = ParquetTable.init(allocator, data) catch |err| {
        return err;
    };
    defer pq_table.deinit();

    const col_names = pq_table.getColumnNames();
    const num_rows = pq_table.numRows();

    // For now, do a simple full scan (SELECT *)
    // TODO: Implement full SQL parsing for Parquet

    // Print header
    if (legacy_args.json) {
        std.debug.print("[", .{});
    } else {
        for (col_names, 0..) |name, i| {
            if (i > 0) {
                if (legacy_args.csv) std.debug.print(",", .{}) else std.debug.print("\t", .{});
            }
            std.debug.print("{s}", .{name});
        }
        std.debug.print("\n", .{});
    }

    // Read and print rows (limit to first 1000 for safety)
    const limit = @min(num_rows, 1000);

    // Read all columns
    var col_data = std.ArrayList(ColumnValues){};
    defer {
        for (col_data.items) |*cv| cv.deinit(allocator);
        col_data.deinit(allocator);
    }

    for (0..col_names.len) |col_idx| {
        const col_type = pq_table.getColumnType(col_idx);
        var cv = ColumnValues{};

        if (col_type) |ct| {
            switch (ct) {
                .int64 => cv.int64 = pq_table.readInt64Column(col_idx) catch null,
                .int32 => cv.int32 = pq_table.readInt32Column(col_idx) catch null,
                .double => cv.float64 = pq_table.readFloat64Column(col_idx) catch null,
                .float => cv.float32 = pq_table.readFloat32Column(col_idx) catch null,
                .byte_array => cv.string = pq_table.readStringColumn(col_idx) catch null,
                .boolean => cv.bool_ = pq_table.readBoolColumn(col_idx) catch null,
                else => {},
            }
        }
        col_data.append(allocator, cv) catch {};
    }

    // Print rows
    for (0..limit) |row| {
        if (legacy_args.json) {
            if (row > 0) std.debug.print(",", .{});
            std.debug.print("{{", .{});
        }

        for (col_data.items, 0..) |cv, i| {
            if (legacy_args.json) {
                if (i > 0) std.debug.print(",", .{});
                std.debug.print("\"{s}\":", .{col_names[i]});
            } else if (i > 0) {
                if (legacy_args.csv) std.debug.print(",", .{}) else std.debug.print("\t", .{});
            }
            printParquetValue(cv, row, legacy_args.json);
        }

        if (legacy_args.json) {
            std.debug.print("}}", .{});
        } else {
            std.debug.print("\n", .{});
        }
    }

    if (legacy_args.json) {
        std.debug.print("]\n", .{});
    }
}

const ColumnValues = struct {
    int64: ?[]i64 = null,
    int32: ?[]i32 = null,
    float64: ?[]f64 = null,
    float32: ?[]f32 = null,
    string: ?[][]const u8 = null,
    bool_: ?[]bool = null,

    fn deinit(self: *ColumnValues, allocator: std.mem.Allocator) void {
        if (self.int64) |v| allocator.free(v);
        if (self.int32) |v| allocator.free(v);
        if (self.float64) |v| allocator.free(v);
        if (self.float32) |v| allocator.free(v);
        if (self.string) |v| {
            for (v) |s| allocator.free(s);
            allocator.free(v);
        }
        if (self.bool_) |v| allocator.free(v);
    }
};

fn printParquetValue(cv: ColumnValues, row: usize, json: bool) void {
    if (cv.int64) |arr| {
        if (row < arr.len) std.debug.print("{d}", .{arr[row]}) else std.debug.print("null", .{});
    } else if (cv.int32) |arr| {
        if (row < arr.len) std.debug.print("{d}", .{arr[row]}) else std.debug.print("null", .{});
    } else if (cv.float64) |arr| {
        if (row < arr.len) std.debug.print("{d:.6}", .{arr[row]}) else std.debug.print("null", .{});
    } else if (cv.float32) |arr| {
        if (row < arr.len) std.debug.print("{d:.6}", .{arr[row]}) else std.debug.print("null", .{});
    } else if (cv.string) |arr| {
        if (row < arr.len) {
            if (json) std.debug.print("\"{s}\"", .{arr[row]}) else std.debug.print("{s}", .{arr[row]});
        } else {
            std.debug.print("null", .{});
        }
    } else if (cv.bool_) |arr| {
        if (row < arr.len) std.debug.print("{}", .{arr[row]}) else std.debug.print("null", .{});
    } else {
        std.debug.print("null", .{});
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
