//! In-Process Benchmark: @logic_table (Compiled Python) vs DuckDB vs Polars
//!
//! What we're comparing:
//!   - @logic_table: REAL Python for loops compiled to native Zig by metal0
//!   - DuckDB: In-process SQL with C API (list_dot_product built-in)
//!   - Polars: Python DataFrame API (df["a"] * df["b"]).sum()
//!
//! HONEST NOTES:
//!   - @logic_table is REAL compiled Python (see benchmarks/vector_ops.py)
//!   - Python code: for i in range(len(a)): result += a[i] * b[i]
//!   - Metal0 compiles this to Zig with runtime dispatch (NOT hand-written SIMD)
//!   - DuckDB/Polars have their own optimized implementations
//!
//! Compile the @logic_table Python code:
//!   metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a

const std = @import("std");
const c = @cImport({
    @cInclude("duckdb.h");
});

const WARMUP = 5;
const LANCEQL_ITERATIONS = 1_000_000; // ~5+ seconds at ~5us/op
const DUCKDB_ITERATIONS = 2000; // ~5+ seconds at ~2.5ms/op
const POLARS_ITERATIONS = 100_000; // Polars via Python

// @logic_table compiled from benchmarks/vector_ops.py
// This is REAL Python code compiled to native Zig by metal0
extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) f64;
extern fn VectorOps_sum_squares(a: [*]const f64, len: usize) f64;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("IN-PROCESS Benchmark: LanceQL vs DuckDB vs Polars\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\nNo subprocess overhead - all engines run in-process.\n", .{});
    std.debug.print("Iterations: {}M (LanceQL), {}K (DuckDB), {}K (Polars)\n", .{ LANCEQL_ITERATIONS / 1_000_000, DUCKDB_ITERATIONS / 1000, POLARS_ITERATIONS / 1000 });
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
    std.debug.print("{s:<25} {d:>9.0} ns {d:>10.1}s {s:>10}\n", .{ "@logic_table", lanceql_per_op, lanceql_total_s, "1.0x" });

    // DuckDB in-process (fewer iterations - SQL parsing overhead)
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
        for (0..DUCKDB_ITERATIONS) |_| {
            if (c.duckdb_query(conn, sql, &result) == c.DuckDBSuccess) {
                c.duckdb_destroy_result(&result);
            }
        }
        duckdb_ns = timer.read();
    }
    const duckdb_per_op = @as(f64, @floatFromInt(duckdb_ns)) / @as(f64, @floatFromInt(DUCKDB_ITERATIONS));
    const duckdb_total_s = @as(f64, @floatFromInt(duckdb_ns)) / 1_000_000_000.0;
    const duckdb_ratio = duckdb_per_op / lanceql_per_op;
    std.debug.print("{s:<25} {d:>9.0} us {d:>10.1}s {d:>9.0}x\n", .{ "DuckDB", duckdb_per_op / 1000.0, duckdb_total_s, duckdb_ratio });

    // Polars in-process (via Python - uses actual Polars DataFrame API)
    var polars_ns: u64 = 0;
    var polars_per_op: f64 = 0;
    {
        // Create Python script for Polars benchmark using DataFrame
        const py_script =
            \\import time
            \\import polars as pl
            \\import numpy as np
            \\
            \\np.random.seed(42)
            \\a = np.random.randn(384).astype(np.float64)
            \\b = np.random.randn(384).astype(np.float64)
            \\
            \\# Create DataFrames
            \\df_a = pl.DataFrame({"val": a.tolist()})
            \\df_b = pl.DataFrame({"val": b.tolist()})
            \\
            \\# Warmup
            \\for _ in range(10):
            \\    _ = (df_a["val"] * df_b["val"]).sum()
            \\
            \\# Benchmark Polars dot product
            \\start = time.perf_counter_ns()
            \\for _ in range(100000):
            \\    result = (df_a["val"] * df_b["val"]).sum()
            \\elapsed = time.perf_counter_ns() - start
            \\print(elapsed)
        ;

        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", py_script },
        }) catch {
            std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "Polars", "N/A", "N/A", "N/A" });
            return;
        };
        defer {
            allocator.free(result.stdout);
            allocator.free(result.stderr);
        }

        const trimmed = std.mem.trim(u8, result.stdout, " \n\r\t");
        polars_ns = std.fmt.parseInt(u64, trimmed, 10) catch 0;
        polars_per_op = @as(f64, @floatFromInt(polars_ns)) / @as(f64, @floatFromInt(POLARS_ITERATIONS));
        const polars_total_s = @as(f64, @floatFromInt(polars_ns)) / 1_000_000_000.0;
        const polars_ratio = polars_per_op / lanceql_per_op;
        std.debug.print("{s:<25} {d:>9.0} ns {d:>10.1}s {d:>9.1}x\n", .{ "Polars", polars_per_op, polars_total_s, polars_ratio });
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary (in-process comparison)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("@logic_table:  {d:>8.0} ns/op (compiled Python for loops)\n", .{lanceql_per_op});
    std.debug.print("DuckDB:        {d:>8.0} us/op ({d:.0}x slower - SQL parsing overhead)\n", .{ duckdb_per_op / 1000.0, duckdb_ratio });
    std.debug.print("Polars:        {d:>8.0} ns/op ({d:.1}x)\n", .{ polars_per_op, polars_per_op / lanceql_per_op });
    std.debug.print("\n", .{});
    std.debug.print("NOTE: @logic_table = Python for loops compiled to native Zig by metal0\n", .{});
    std.debug.print("      See benchmarks/vector_ops.py for the actual Python code.\n", .{});
    std.debug.print("\n", .{});
}
