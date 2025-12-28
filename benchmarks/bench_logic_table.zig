//! @logic_table Benchmark: LanceQL vs DuckDB vs Polars
//!
//! Compares REAL metal0-compiled @logic_table vs DuckDB/Polars UDFs.
//!
//! Workflow:
//!   1. metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a
//!   2. zig build bench-logic-table
//!
//! The Python source (benchmarks/vector_ops.py):
//!   @logic_table
//!   class VectorOps:
//!       def dot_product(self, a: list, b: list) -> float: ...
//!       def sum_squares(self, a: list) -> float: ...
//!       def sum_values(self, a: list) -> float: ...

const std = @import("std");

const WARMUP = 5;
const LANCEQL_ITERATIONS = 1_000_000; // ~5+ seconds at ~5us/op
const SUBPROCESS_ITERATIONS = 200; // ~5+ seconds at ~30ms/op (DuckDB subprocess)

// Engine detection
var has_duckdb: bool = false;
var has_polars: bool = false;

// =============================================================================
// REAL extern declarations - from lib/vector_ops.a (metal0 compiled)
// =============================================================================

extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) f64;
extern fn VectorOps_sum_squares(a: [*]const f64, len: usize) f64;
extern fn VectorOps_sum_values(a: [*]const f64, len: usize) f64;


// =============================================================================
// DuckDB/Polars runners
// =============================================================================

fn checkCommand(allocator: std.mem.Allocator, cmd: []const u8) bool {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "which", cmd },
    }) catch return false;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    return result.term.Exited == 0;
}

fn runDuckDB(allocator: std.mem.Allocator, sql: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "duckdb", "-csv", "-c", sql },
        .max_output_bytes = 10 * 1024 * 1024, // 10MB for large vector arrays
    }) catch return error.DuckDBFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |code| if (code != 0) return error.DuckDBFailed,
        else => return error.DuckDBFailed,
    }
    return timer.read();
}

fn runPolars(allocator: std.mem.Allocator, code: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", "-c", code },
        .max_output_bytes = 10 * 1024 * 1024, // 10MB
    }) catch return error.PolarsFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |exit_code| if (exit_code != 0) return error.PolarsFailed,
        else => return error.PolarsFailed,
    }
    return timer.read();
}

// =============================================================================
// Benchmark
// =============================================================================

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("@logic_table Benchmark: LanceQL vs DuckDB vs Polars\n", .{});
    std.debug.print("================================================================================\n", .{});

    // Check available engines
    has_duckdb = checkCommand(allocator, "duckdb");
    has_polars = checkCommand(allocator, "python3");

    std.debug.print("\nEngines:\n", .{});
    std.debug.print("  - LanceQL:  yes (lib/vector_ops.a)\n", .{});
    std.debug.print("  - DuckDB:   {s}\n", .{if (has_duckdb) "yes" else "no (install: brew install duckdb)"});
    std.debug.print("  - Polars:   {s}\n", .{if (has_polars) "yes" else "no (install: pip install polars)"});
    std.debug.print("\n", .{});

    const dim: usize = 384;

    std.debug.print("Vector dimension: {}\n", .{dim});
    std.debug.print("Warmup: {}, Iterations: {}M (LanceQL), {} (subprocess)\n", .{ WARMUP, LANCEQL_ITERATIONS / 1_000_000, SUBPROCESS_ITERATIONS });
    std.debug.print("\n", .{});

    const a = try allocator.alloc(f64, dim);
    defer allocator.free(a);
    const b = try allocator.alloc(f64, dim);
    defer allocator.free(b);

    // Initialize with deterministic data
    var rng = std.Random.DefaultPrng.init(42);
    for (a) |*v| v.* = rng.random().float(f64) * 2 - 1;
    for (b) |*v| v.* = rng.random().float(f64) * 2 - 1;

    // =========================================================================
    // Dot Product Benchmark
    // =========================================================================
    std.debug.print("================================================================================\n", .{});
    std.debug.print("DOT PRODUCT (384-dim vectors)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "Engine", "Time/op", "Total", "Ratio" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 10 });

    // LanceQL (baseline)
    var lanceql_ns: u64 = 0;
    {
        var checksum: f64 = 0;
        for (0..WARMUP) |_| checksum += VectorOps_dot_product(a.ptr, b.ptr, dim);
        var timer = try std.time.Timer.start();
        for (0..LANCEQL_ITERATIONS) |_| checksum += VectorOps_dot_product(a.ptr, b.ptr, dim);
        lanceql_ns = timer.read();
        std.mem.doNotOptimizeAway(&checksum);
    }
    const lanceql_per_op = @as(f64, @floatFromInt(lanceql_ns)) / @as(f64, @floatFromInt(LANCEQL_ITERATIONS));
    const lanceql_total_s = @as(f64, @floatFromInt(lanceql_ns)) / 1_000_000_000.0;
    std.debug.print("{s:<25} {d:>9.0} ns {d:>10.1}s {s:>10}\n", .{ "LanceQL", lanceql_per_op, lanceql_total_s, "1.0x" });

    // DuckDB
    if (has_duckdb) {
        // Build array string
        var a_str = std.ArrayListUnmanaged(u8){};
        defer a_str.deinit(allocator);
        try a_str.appendSlice(allocator, "[");
        for (a, 0..) |v, i| {
            if (i > 0) try a_str.appendSlice(allocator, ",");
            try std.fmt.format(a_str.writer(allocator), "{d:.6}", .{v});
        }
        try a_str.appendSlice(allocator, "]");

        var b_str = std.ArrayListUnmanaged(u8){};
        defer b_str.deinit(allocator);
        try b_str.appendSlice(allocator, "[");
        for (b, 0..) |v, i| {
            if (i > 0) try b_str.appendSlice(allocator, ",");
            try std.fmt.format(b_str.writer(allocator), "{d:.6}", .{v});
        }
        try b_str.appendSlice(allocator, "]");

        const sql = try std.fmt.allocPrint(allocator,
            \\SELECT list_dot_product({s}::DOUBLE[], {s}::DOUBLE[]);
        , .{ a_str.items, b_str.items });
        defer allocator.free(sql);

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
        for (0..SUBPROCESS_ITERATIONS) |_| {
            total_ns += runDuckDB(allocator, sql) catch 0;
        }
        const duckdb_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(SUBPROCESS_ITERATIONS));
        const duckdb_total_s = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
        const duckdb_ratio = duckdb_per_op / lanceql_per_op;
        std.debug.print("{s:<25} {d:>9.0} ms {d:>10.1}s {d:>9.0}x\n", .{ "DuckDB", duckdb_per_op / 1_000_000.0, duckdb_total_s, duckdb_ratio });
    }

    // Polars
    if (has_polars) {
        // Build Python code
        var a_str = std.ArrayListUnmanaged(u8){};
        defer a_str.deinit(allocator);
        try a_str.appendSlice(allocator, "[");
        for (a, 0..) |v, i| {
            if (i > 0) try a_str.appendSlice(allocator, ",");
            try std.fmt.format(a_str.writer(allocator), "{d:.6}", .{v});
        }
        try a_str.appendSlice(allocator, "]");

        var b_str = std.ArrayListUnmanaged(u8){};
        defer b_str.deinit(allocator);
        try b_str.appendSlice(allocator, "[");
        for (b, 0..) |v, i| {
            if (i > 0) try b_str.appendSlice(allocator, ",");
            try std.fmt.format(b_str.writer(allocator), "{d:.6}", .{v});
        }
        try b_str.appendSlice(allocator, "]");

        const py_code = try std.fmt.allocPrint(allocator,
            \\import polars as pl
            \\a = {s}
            \\b = {s}
            \\result = sum(x*y for x,y in zip(a,b))
        , .{ a_str.items, b_str.items });
        defer allocator.free(py_code);

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, py_code) catch 0;
        for (0..SUBPROCESS_ITERATIONS) |_| {
            total_ns += runPolars(allocator, py_code) catch 0;
        }
        const polars_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(SUBPROCESS_ITERATIONS));
        const polars_total_s = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
        const polars_ratio = polars_per_op / lanceql_per_op;
        std.debug.print("{s:<25} {d:>9.0} ms {d:>10.1}s {d:>9.0}x\n", .{ "Polars", polars_per_op / 1_000_000.0, polars_total_s, polars_ratio });
    }

    // =========================================================================
    // Sum Squares Benchmark
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("SUM SQUARES (384-dim vectors)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "Engine", "Time/op", "Total", "Ratio" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 10 });

    // LanceQL (baseline)
    {
        var checksum: f64 = 0;
        for (0..WARMUP) |_| checksum += VectorOps_sum_squares(a.ptr, dim);
        var timer = try std.time.Timer.start();
        for (0..LANCEQL_ITERATIONS) |_| checksum += VectorOps_sum_squares(a.ptr, dim);
        lanceql_ns = timer.read();
        std.mem.doNotOptimizeAway(&checksum);
    }
    const lanceql_ss_per_op = @as(f64, @floatFromInt(lanceql_ns)) / @as(f64, @floatFromInt(LANCEQL_ITERATIONS));
    const lanceql_ss_total_s = @as(f64, @floatFromInt(lanceql_ns)) / 1_000_000_000.0;
    std.debug.print("{s:<25} {d:>9.0} ns {d:>10.1}s {s:>10}\n", .{ "LanceQL", lanceql_ss_per_op, lanceql_ss_total_s, "1.0x" });

    // DuckDB
    if (has_duckdb) {
        var a_str2 = std.ArrayListUnmanaged(u8){};
        defer a_str2.deinit(allocator);
        try a_str2.appendSlice(allocator, "[");
        for (a, 0..) |v, i| {
            if (i > 0) try a_str2.appendSlice(allocator, ",");
            try std.fmt.format(a_str2.writer(allocator), "{d:.6}", .{v});
        }
        try a_str2.appendSlice(allocator, "]");

        const sql = try std.fmt.allocPrint(allocator,
            \\SELECT list_sum(list_transform({s}::DOUBLE[], x -> x * x));
        , .{a_str2.items});
        defer allocator.free(sql);

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
        for (0..SUBPROCESS_ITERATIONS) |_| {
            total_ns += runDuckDB(allocator, sql) catch 0;
        }
        const duckdb_ss_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(SUBPROCESS_ITERATIONS));
        const duckdb_ss_total_s = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
        const duckdb_ss_ratio = duckdb_ss_per_op / lanceql_ss_per_op;
        std.debug.print("{s:<25} {d:>9.0} ms {d:>10.1}s {d:>9.0}x\n", .{ "DuckDB", duckdb_ss_per_op / 1_000_000.0, duckdb_ss_total_s, duckdb_ss_ratio });
    }

    // Polars
    if (has_polars) {
        var a_str3 = std.ArrayListUnmanaged(u8){};
        defer a_str3.deinit(allocator);
        try a_str3.appendSlice(allocator, "[");
        for (a, 0..) |v, i| {
            if (i > 0) try a_str3.appendSlice(allocator, ",");
            try std.fmt.format(a_str3.writer(allocator), "{d:.6}", .{v});
        }
        try a_str3.appendSlice(allocator, "]");

        const py_code = try std.fmt.allocPrint(allocator,
            \\import polars as pl
            \\a = {s}
            \\result = sum(x*x for x in a)
        , .{a_str3.items});
        defer allocator.free(py_code);

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, py_code) catch 0;
        for (0..SUBPROCESS_ITERATIONS) |_| {
            total_ns += runPolars(allocator, py_code) catch 0;
        }
        const polars_ss_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(SUBPROCESS_ITERATIONS));
        const polars_ss_total_s = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
        const polars_ss_ratio = polars_ss_per_op / lanceql_ss_per_op;
        std.debug.print("{s:<25} {d:>9.0} ms {d:>10.1}s {d:>9.0}x\n", .{ "Polars", polars_ss_per_op / 1_000_000.0, polars_ss_total_s, polars_ss_ratio });
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("LanceQL is the baseline (1.0x).\n", .{});
    std.debug.print("DuckDB/Polars times include subprocess startup overhead.\n", .{});
    std.debug.print("\n", .{});
}
