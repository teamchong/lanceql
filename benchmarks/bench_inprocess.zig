//! In-Process Benchmark: LanceQL vs DuckDB (C API)
//!
//! FAIR apples-to-apples comparison - no subprocess overhead.
//! Both engines run in-process with native code.

const std = @import("std");
const c = @cImport({
    @cInclude("duckdb.h");
});

const WARMUP = 5;
const ITERATIONS = 10_000_000;

// LanceQL @logic_table (metal0 compiled)
extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) f64;
extern fn VectorOps_sum_squares(a: [*]const f64, len: usize) f64;

// Native Zig baseline
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

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("IN-PROCESS Benchmark: LanceQL vs DuckDB (C API) - FAIR COMPARISON\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\nNo subprocess overhead - both run in-process with native code.\n", .{});
    std.debug.print("Iterations: {}M\n", .{ITERATIONS / 1_000_000});
    std.debug.print("\n", .{});

    const dim: usize = 384;

    // Initialize test vectors
    const a = try allocator.alloc(f64, dim);
    defer allocator.free(a);
    const b = try allocator.alloc(f64, dim);
    defer allocator.free(b);

    var rng = std.Random.DefaultPrng.init(42);
    for (a) |*v| v.* = rng.random().float(f64) * 2 - 1;
    for (b) |*v| v.* = rng.random().float(f64) * 2 - 1;

    // Initialize DuckDB
    var db: c.duckdb_database = null;
    var conn: c.duckdb_connection = null;

    if (c.duckdb_open(null, &db) != c.DuckDBSuccess) {
        std.debug.print("Failed to open DuckDB\n", .{});
        return;
    }
    defer c.duckdb_close(&db);

    if (c.duckdb_connect(db, &conn) != c.DuckDBSuccess) {
        std.debug.print("Failed to connect to DuckDB\n", .{});
        return;
    }
    defer c.duckdb_disconnect(&conn);

    std.debug.print("DuckDB initialized (in-process)\n\n", .{});

    // Build array SQL for DuckDB
    var a_sql = std.ArrayListUnmanaged(u8){};
    defer a_sql.deinit(allocator);
    try a_sql.appendSlice(allocator, "[");
    for (a, 0..) |v, i| {
        if (i > 0) try a_sql.appendSlice(allocator, ",");
        try std.fmt.format(a_sql.writer(allocator), "{d:.6}", .{v});
    }
    try a_sql.appendSlice(allocator, "]");

    var b_sql = std.ArrayListUnmanaged(u8){};
    defer b_sql.deinit(allocator);
    try b_sql.appendSlice(allocator, "[");
    for (b, 0..) |v, i| {
        if (i > 0) try b_sql.appendSlice(allocator, ",");
        try std.fmt.format(b_sql.writer(allocator), "{d:.6}", .{v});
    }
    try b_sql.appendSlice(allocator, "]");

    // =========================================================================
    // DOT PRODUCT
    // =========================================================================
    std.debug.print("================================================================================\n", .{});
    std.debug.print("DOT PRODUCT (384-dim)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "Engine", "Time/op", "Total", "Ratio" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 10 });

    // Native Zig
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
    const native_total_s = @as(f64, @floatFromInt(native_ns)) / 1_000_000_000.0;
    std.debug.print("{s:<25} {d:>9.0} ns {d:>10.1}s {s:>10}\n", .{ "Native Zig", native_per_op, native_total_s, "1.0x" });

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
    const lanceql_total_s = @as(f64, @floatFromInt(lanceql_ns)) / 1_000_000_000.0;
    const lanceql_ratio = lanceql_per_op / native_per_op;
    std.debug.print("{s:<25} {d:>9.0} ns {d:>10.1}s {d:>9.1}x\n", .{ "LanceQL @logic_table", lanceql_per_op, lanceql_total_s, lanceql_ratio });

    // DuckDB in-process (fewer iterations - SQL parsing overhead)
    const duckdb_iters: usize = 10000;
    var duckdb_ns: u64 = 0;
    {
        const sql_slice = try std.fmt.allocPrint(allocator,
            "SELECT list_dot_product({s}::DOUBLE[], {s}::DOUBLE[]);",
            .{ a_sql.items, b_sql.items });
        defer allocator.free(sql_slice);

        // Add null terminator for C API
        const sql = try allocator.allocSentinel(u8, sql_slice.len, 0);
        defer allocator.free(sql);
        @memcpy(sql, sql_slice);

        var result: c.duckdb_result = undefined;

        // Warmup
        for (0..WARMUP) |_| {
            if (c.duckdb_query(conn, sql, &result) == c.DuckDBSuccess) {
                c.duckdb_destroy_result(&result);
            }
        }

        var timer = try std.time.Timer.start();
        for (0..duckdb_iters) |_| {
            if (c.duckdb_query(conn, sql, &result) == c.DuckDBSuccess) {
                c.duckdb_destroy_result(&result);
            }
        }
        duckdb_ns = timer.read();
    }
    const duckdb_per_op = @as(f64, @floatFromInt(duckdb_ns)) / @as(f64, @floatFromInt(duckdb_iters));
    const duckdb_total_s = @as(f64, @floatFromInt(duckdb_ns)) / 1_000_000_000.0;
    const duckdb_ratio = duckdb_per_op / native_per_op;
    std.debug.print("{s:<25} {d:>9.0} us {d:>10.1}s {d:>9.0}x\n", .{ "DuckDB (in-process)", duckdb_per_op / 1000.0, duckdb_total_s, duckdb_ratio });

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary (FAIR in-process comparison)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Native Zig:           {d:.0} ns/op (baseline)\n", .{native_per_op});
    std.debug.print("LanceQL @logic_table: {d:.0} ns/op ({d:.1}x slower)\n", .{ lanceql_per_op, lanceql_ratio });
    std.debug.print("DuckDB (in-process):  {d:.0} us/op ({d:.0}x slower)\n", .{ duckdb_per_op / 1000.0, duckdb_ratio });
    std.debug.print("\n", .{});
    std.debug.print("DuckDB overhead is SQL parsing + query planning per call.\n", .{});
    std.debug.print("For batch operations, DuckDB would be much faster.\n", .{});
    std.debug.print("\n", .{});
}
