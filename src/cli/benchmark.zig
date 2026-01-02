//! CLI Benchmark
//!
//! Benchmark execution and statistics for query performance testing.

const std = @import("std");
const Table = @import("lanceql.table").Table;
const lexer = @import("lanceql.sql.lexer");
const parser = @import("lanceql.sql.parser");
const executor = @import("lanceql.sql.executor");
const ast = @import("lanceql.sql.ast");
const file_utils = @import("file_utils.zig");

/// Benchmark options
pub const BenchmarkOptions = struct {
    iterations: usize = 10,
    warmup: usize = 3,
    json: bool = false,
};

/// Run benchmark on a query
pub fn run(allocator: std.mem.Allocator, query: []const u8, opts: BenchmarkOptions) !void {
    // Extract table path from query
    const table_path = file_utils.extractTablePath(query) orelse {
        std.debug.print("Error: Could not extract table path from query\n", .{});
        return;
    };

    // Read file into memory
    const data = file_utils.openFileOrDataset(allocator, table_path) orelse {
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
    std.debug.print("Warmup: {d}, Iterations: {d}\n\n", .{ opts.warmup, opts.iterations });

    // Warmup
    for (0..opts.warmup) |_| {
        var exec = executor.Executor.init(&table, allocator);
        var result = exec.execute(&stmt.select, &[_]ast.Value{}) catch continue;
        result.deinit();
        exec.deinit();
    }

    // Benchmark
    var times = try allocator.alloc(u64, opts.iterations);
    defer allocator.free(times);

    for (0..opts.iterations) |i| {
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

    const avg_ns = total_ns / opts.iterations;
    const avg_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000;
    const min_ms = @as(f64, @floatFromInt(min_ns)) / 1_000_000;
    const max_ms = @as(f64, @floatFromInt(max_ns)) / 1_000_000;
    const throughput = @as(f64, @floatFromInt(num_rows)) / avg_ms / 1000;

    if (opts.json) {
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
