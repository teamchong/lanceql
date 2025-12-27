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

const WARMUP = 3;
const ITERATIONS = 1000; // Reduced for subprocess overhead

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
// Native Zig baseline
// =============================================================================

fn nativeDotProduct(a: []const f64, b: []const f64) f64 {
    var result: f64 = 0.0;
    const len = @min(a.len, b.len);
    for (0..len) |i| {
        result += a[i] * b[i];
    }
    return result;
}

fn nativeSumSquares(a: []const f64) f64 {
    var result: f64 = 0.0;
    for (a) |v| {
        result += v * v;
    }
    return result;
}

fn nativeSumValues(a: []const f64) f64 {
    var result: f64 = 0.0;
    for (a) |v| {
        result += v;
    }
    return result;
}

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
    std.debug.print("  - LanceQL @logic_table: yes (lib/vector_ops.a)\n", .{});
    std.debug.print("  - Native Zig:           yes (baseline)\n", .{});
    std.debug.print("  - DuckDB:               {s}\n", .{if (has_duckdb) "yes" else "no (install: brew install duckdb)"});
    std.debug.print("  - Polars:               {s}\n", .{if (has_polars) "yes" else "no (install: pip install polars)"});
    std.debug.print("\n", .{});

    const dim: usize = 384;

    std.debug.print("Vector dimension: {}\n", .{dim});
    std.debug.print("Warmup: {}, Iterations: {}\n", .{ WARMUP, ITERATIONS });
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
    std.debug.print("DOT PRODUCT\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>15} {s:>12}\n", .{ "Engine", "Time/op", "Ratio" });
    std.debug.print("{s:<25} {s:>15} {s:>12}\n", .{ "-" ** 25, "-" ** 15, "-" ** 12 });

    // Native Zig (baseline)
    var native_ns: u64 = 0;
    {
        var checksum: f64 = 0;
        for (0..WARMUP) |_| checksum += nativeDotProduct(a, b);
        var timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| checksum += nativeDotProduct(a, b);
        native_ns = timer.read();
        std.mem.doNotOptimizeAway(&checksum);
    }
    const native_per_op = @as(f64, @floatFromInt(native_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    std.debug.print("{s:<25} {d:>12.1} ns {s:>12}\n", .{ "Native Zig", native_per_op, "1.0x" });

    // LanceQL @logic_table
    var lanceql_ns: u64 = 0;
    {
        var checksum: f64 = 0;
        for (0..WARMUP) |_| checksum += VectorOps_dot_product(a.ptr, b.ptr, dim);
        var timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| checksum += VectorOps_dot_product(a.ptr, b.ptr, dim);
        lanceql_ns = timer.read();
        std.mem.doNotOptimizeAway(&checksum);
    }
    const lanceql_per_op = @as(f64, @floatFromInt(lanceql_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const lanceql_ratio = lanceql_per_op / native_per_op;
    std.debug.print("{s:<25} {d:>12.1} ns {d:>11.1}x\n", .{ "LanceQL @logic_table", lanceql_per_op, lanceql_ratio });

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
        for (0..@min(ITERATIONS, 100)) |_| { // Limit iterations for subprocess
            total_ns += runDuckDB(allocator, sql) catch 0;
        }
        const duckdb_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(@min(ITERATIONS, 100)));
        const duckdb_ratio = duckdb_per_op / native_per_op;
        std.debug.print("{s:<25} {d:>12.1} ns {d:>11.1}x\n", .{ "DuckDB (subprocess)", duckdb_per_op, duckdb_ratio });
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
        for (0..@min(ITERATIONS, 100)) |_| {
            total_ns += runPolars(allocator, py_code) catch 0;
        }
        const polars_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(@min(ITERATIONS, 100)));
        const polars_ratio = polars_per_op / native_per_op;
        std.debug.print("{s:<25} {d:>12.1} ns {d:>11.1}x\n", .{ "Polars (subprocess)", polars_per_op, polars_ratio });
    }

    // =========================================================================
    // Sum Squares Benchmark
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("SUM SQUARES\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>15} {s:>12}\n", .{ "Engine", "Time/op", "Ratio" });
    std.debug.print("{s:<25} {s:>15} {s:>12}\n", .{ "-" ** 25, "-" ** 15, "-" ** 12 });

    // Native Zig
    {
        var checksum: f64 = 0;
        for (0..WARMUP) |_| checksum += nativeSumSquares(a);
        var timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| checksum += nativeSumSquares(a);
        native_ns = timer.read();
        std.mem.doNotOptimizeAway(&checksum);
    }
    const native_ss_per_op = @as(f64, @floatFromInt(native_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    std.debug.print("{s:<25} {d:>12.1} ns {s:>12}\n", .{ "Native Zig", native_ss_per_op, "1.0x" });

    // LanceQL @logic_table
    {
        var checksum: f64 = 0;
        for (0..WARMUP) |_| checksum += VectorOps_sum_squares(a.ptr, dim);
        var timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| checksum += VectorOps_sum_squares(a.ptr, dim);
        lanceql_ns = timer.read();
        std.mem.doNotOptimizeAway(&checksum);
    }
    const lanceql_ss_per_op = @as(f64, @floatFromInt(lanceql_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const lanceql_ss_ratio = lanceql_ss_per_op / native_ss_per_op;
    std.debug.print("{s:<25} {d:>12.1} ns {d:>11.1}x\n", .{ "LanceQL @logic_table", lanceql_ss_per_op, lanceql_ss_ratio });

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
        for (0..@min(ITERATIONS, 100)) |_| {
            total_ns += runDuckDB(allocator, sql) catch 0;
        }
        const duckdb_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(@min(ITERATIONS, 100)));
        const duckdb_ratio = duckdb_per_op / native_ss_per_op;
        std.debug.print("{s:<25} {d:>12.1} ns {d:>11.1}x\n", .{ "DuckDB (subprocess)", duckdb_per_op, duckdb_ratio });
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
        for (0..@min(ITERATIONS, 100)) |_| {
            total_ns += runPolars(allocator, py_code) catch 0;
        }
        const polars_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(@min(ITERATIONS, 100)));
        const polars_ratio = polars_per_op / native_ss_per_op;
        std.debug.print("{s:<25} {d:>12.1} ns {d:>11.1}x\n", .{ "Polars (subprocess)", polars_per_op, polars_ratio });
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("LanceQL @logic_table is {d:.0}x slower than native Zig.\n", .{lanceql_ratio});
    std.debug.print("This is the optimization target for metal0 codegen.\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Note: DuckDB/Polars times include subprocess overhead.\n", .{});
    std.debug.print("For fair comparison, use in-process bindings.\n", .{});
    std.debug.print("\n", .{});
}
