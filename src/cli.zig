//! LanceQL CLI - Native query interface for Lance/Parquet files
//!
//! Usage:
//!   lanceql "SELECT * FROM 'data.lance' LIMIT 10"
//!   lanceql -f query.sql
//!   lanceql -c "SELECT COUNT(*) FROM 'data.parquet'"
//!   lanceql --benchmark "SELECT * FROM 'data.lance' WHERE x > 100"
//!
//! Designed for apple-to-apple comparison with:
//!   duckdb -c "SELECT * FROM 'data.parquet' LIMIT 10"
//!   polars -c "SELECT * FROM read_parquet('data.parquet') LIMIT 10"

const std = @import("std");
const lanceql = @import("lanceql");
const metal = @import("lanceql.metal");
const Table = @import("lanceql.table").Table;
const executor = @import("lanceql.sql.executor");
const lexer = @import("lanceql.sql.lexer");
const parser = @import("lanceql.sql.parser");
const ast = @import("lanceql.sql.ast");

const version = "0.1.0";

const Args = struct {
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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try parseArgs(allocator);

    if (args.show_version) {
        std.debug.print("lanceql {s}\n", .{version});
        return;
    }

    if (args.help or args.query == null) {
        printUsage();
        return;
    }

    // Initialize GPU if available
    _ = metal.initGPU();
    defer metal.cleanupGPU();

    const query = args.query.?;

    if (args.benchmark) {
        try runBenchmark(allocator, query, args);
    } else {
        try runQuery(allocator, query, args);
    }
}

fn parseArgs(allocator: std.mem.Allocator) !Args {
    const argv = try std.process.argsAlloc(allocator);
    // Don't free argv - we need to keep strings alive

    var args = Args{};
    var i: usize = 1;

    while (i < argv.len) : (i += 1) {
        const arg = argv[i];

        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            args.help = true;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--version")) {
            args.show_version = true;
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--command")) {
            i += 1;
            if (i < argv.len) args.query = argv[i];
        } else if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--file")) {
            i += 1;
            if (i < argv.len) args.file = argv[i];
        } else if (std.mem.eql(u8, arg, "-b") or std.mem.eql(u8, arg, "--benchmark")) {
            args.benchmark = true;
        } else if (std.mem.eql(u8, arg, "-i") or std.mem.eql(u8, arg, "--iterations")) {
            i += 1;
            if (i < argv.len) {
                args.iterations = std.fmt.parseInt(usize, argv[i], 10) catch 10;
            }
        } else if (std.mem.eql(u8, arg, "-w") or std.mem.eql(u8, arg, "--warmup")) {
            i += 1;
            if (i < argv.len) {
                args.warmup = std.fmt.parseInt(usize, argv[i], 10) catch 3;
            }
        } else if (std.mem.eql(u8, arg, "--json")) {
            args.json = true;
        } else if (std.mem.eql(u8, arg, "--csv")) {
            args.csv = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            // Positional argument = query
            args.query = argv[i];
        }
    }

    // Read query from file if specified
    if (args.file) |file_path| {
        const f = std.fs.cwd().openFile(file_path, .{}) catch |err| {
            std.debug.print("Error opening file '{s}': {}\n", .{ file_path, err });
            return args;
        };
        defer f.close();

        const content = f.readToEndAlloc(allocator, 1024 * 1024) catch |err| {
            std.debug.print("Error reading file: {}\n", .{err});
            return args;
        };
        args.query = content;
    }

    return args;
}

fn printUsage() void {
    std.debug.print(
        \\LanceQL - Native query engine for Lance/Parquet files
        \\
        \\Usage:
        \\  lanceql [OPTIONS] "SQL QUERY"
        \\  lanceql -c "SELECT * FROM 'data.lance' LIMIT 10"
        \\  lanceql -f query.sql
        \\  lanceql --benchmark "SELECT COUNT(*) FROM 'data.parquet'"
        \\
        \\Options:
        \\  -c, --command <SQL>    Execute SQL query
        \\  -f, --file <PATH>      Read SQL from file
        \\  -b, --benchmark        Run query in benchmark mode (measure time)
        \\  -i, --iterations <N>   Benchmark iterations (default: 10)
        \\  -w, --warmup <N>       Warmup iterations (default: 3)
        \\      --json             Output results as JSON
        \\      --csv              Output results as CSV
        \\  -h, --help             Show this help
        \\  -v, --version          Show version
        \\
        \\Examples:
        \\  # Query Lance file
        \\  lanceql "SELECT * FROM 'users.lance' WHERE age > 25 LIMIT 100"
        \\
        \\  # Query Parquet file
        \\  lanceql "SELECT COUNT(*), AVG(price) FROM 'sales.parquet'"
        \\
        \\  # Vector search
        \\  lanceql "SELECT * FROM 'embeddings.lance' NEAR 'search query' TOPK 20"
        \\
        \\  # Benchmark query
        \\  lanceql -b -i 20 "SELECT * FROM 'data.parquet' WHERE x > 100"
        \\
        \\Comparison with other tools:
        \\  # DuckDB
        \\  duckdb -c "SELECT * FROM 'data.parquet' LIMIT 10"
        \\
        \\  # Polars (Python)
        \\  python -c "import polars as pl; print(pl.read_parquet('data.parquet').head(10))"
        \\
        \\  # LanceQL (this tool)
        \\  lanceql -c "SELECT * FROM 'data.parquet' LIMIT 10"
        \\
    , .{});
}

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

fn runQuery(allocator: std.mem.Allocator, query: []const u8, args: Args) !void {
    // Extract table path from query
    const table_path = extractTablePath(query) orelse {
        std.debug.print("Error: Could not extract table path from query\n", .{});
        std.debug.print("Query should be: SELECT ... FROM 'path/to/file.lance'\n", .{});
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

    // Execute
    var exec = executor.Executor.init(&table, allocator);
    defer exec.deinit();

    var result = exec.execute(&stmt.select, &[_]ast.Value{}) catch |err| {
        std.debug.print("Execution error: {}\n", .{err});
        return;
    };
    defer result.deinit();

    // Output results
    if (args.json) {
        printResultsJson(&result);
    } else if (args.csv) {
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

fn runBenchmark(allocator: std.mem.Allocator, query: []const u8, args: Args) !void {
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
    std.debug.print("Warmup: {d}, Iterations: {d}\n\n", .{ args.warmup, args.iterations });

    // Warmup
    for (0..args.warmup) |_| {
        var exec = executor.Executor.init(&table, allocator);
        var result = exec.execute(&stmt.select, &[_]ast.Value{}) catch continue;
        result.deinit();
        exec.deinit();
    }

    // Benchmark
    var times = try allocator.alloc(u64, args.iterations);
    defer allocator.free(times);

    for (0..args.iterations) |i| {
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

    const avg_ns = total_ns / args.iterations;
    const avg_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000;
    const min_ms = @as(f64, @floatFromInt(min_ns)) / 1_000_000;
    const max_ms = @as(f64, @floatFromInt(max_ns)) / 1_000_000;
    const throughput = @as(f64, @floatFromInt(num_rows)) / avg_ms / 1000;

    if (args.json) {
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
