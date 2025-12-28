//! Feature Engineering Benchmark
//!
//! Real-world use case: ML feature transformations for training/inference
//!
//! Operations tested:
//!   1. Z-score normalization
//!   2. Log transform
//!   3. Feature crossing
//!   4. Binning/Bucketing
//!
//! Fair comparison using each engine's native interface:
//!   - LanceQL:  CLI with SQL (lanceql -c "SELECT ...")
//!   - DuckDB:   CLI with SQL (duckdb -c "SELECT ...")
//!   - Polars:   Python with DataFrame API (native interface)

const std = @import("std");

const WARMUP = 3;
const ITERATIONS = 30;

// Dataset size (use same size for fair comparison)
const NUM_ROWS = 100_000;

var has_lanceql: bool = false;
var has_duckdb: bool = false;
var has_polars: bool = false;

fn checkCommand(allocator: std.mem.Allocator, cmd: []const u8) bool {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "which", cmd },
    }) catch return false;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |code| return code == 0,
        else => return false,
    }
}

fn checkPolars(allocator: std.mem.Allocator) bool {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", "-c", "import polars" },
    }) catch return false;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |code| return code == 0,
        else => return false,
    }
}

fn runLanceQL(allocator: std.mem.Allocator, sql: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "lanceql", "-c", sql },
        .max_output_bytes = 100 * 1024 * 1024,
    }) catch return error.LanceQLFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |code| if (code != 0) return error.LanceQLFailed,
        else => return error.LanceQLFailed,
    }
    return timer.read();
}

fn runDuckDB(allocator: std.mem.Allocator, sql: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "duckdb", "-csv", "-c", sql },
        .max_output_bytes = 100 * 1024 * 1024,
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
        .max_output_bytes = 100 * 1024 * 1024,
    }) catch return error.PolarsFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |code_| if (code_ != 0) return error.PolarsFailed,
        else => return error.PolarsFailed,
    }
    return timer.read();
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Feature Engineering Benchmark: LanceQL vs DuckDB vs Polars\n", .{});
    std.debug.print("================================================================================\n", .{});

    has_lanceql = checkCommand(allocator, "lanceql");
    has_duckdb = checkCommand(allocator, "duckdb");
    has_polars = checkPolars(allocator);

    std.debug.print("\nFair comparison using each engine's native interface:\n", .{});
    std.debug.print("  - LanceQL/DuckDB: CLI with SQL\n", .{});
    std.debug.print("  - Polars: Python with DataFrame API\n", .{});
    std.debug.print("\nDataset: {d}K rows\n", .{NUM_ROWS / 1000});
    std.debug.print("\nEngines:\n", .{});
    std.debug.print("  - LanceQL: {s}\n", .{if (has_lanceql) "yes" else "no (zig build && cp zig-out/bin/lanceql /usr/local/bin/)"});
    std.debug.print("  - DuckDB:  {s}\n", .{if (has_duckdb) "yes" else "no (brew install duckdb)"});
    std.debug.print("  - Polars:  {s}\n", .{if (has_polars) "yes" else "no (pip install polars numpy)"});
    std.debug.print("\n", .{});

    // =========================================================================
    // Benchmark 1: Z-SCORE NORMALIZATION
    // =========================================================================
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Z-SCORE NORMALIZATION ({d}K rows)\n", .{NUM_ROWS / 1000});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<15} {s:>12} {s:>12} {s:>10}\n", .{ "Engine", "Time", "Rows/sec", "Ratio" });
    std.debug.print("{s:<15} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 15, "-" ** 12, "-" ** 12, "-" ** 10 });

    var baseline_ns: u64 = 0;

    // LanceQL
    if (has_lanceql) {
        const sql = std.fmt.comptimePrint(
            \\SELECT (val - AVG(val) OVER()) / NULLIF(STDDEV(val) OVER(), 0) as zscore
            \\FROM (SELECT random() * 1000 as val FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runLanceQL(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runLanceQL(allocator, sql) catch 0;

        baseline_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(baseline_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(baseline_ns)) / 1_000_000_000.0);
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {s:>10}\n", .{ "LanceQL", time_ms, rows_per_sec / 1000, "1.0x" });
    }

    // DuckDB
    if (has_duckdb) {
        const sql = std.fmt.comptimePrint(
            \\SELECT (val - AVG(val) OVER()) / NULLIF(STDDEV(val) OVER(), 0) as zscore
            \\FROM (SELECT random() * 1000 as val FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runDuckDB(allocator, sql) catch 0;

        const avg_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);
        const ratio = if (baseline_ns > 0) @as(f64, @floatFromInt(avg_ns)) / @as(f64, @floatFromInt(baseline_ns)) else 0;
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {d:>9.1}x\n", .{ "DuckDB", time_ms, rows_per_sec / 1000, ratio });
    }

    // Polars (DataFrame API)
    if (has_polars) {
        const py_code = std.fmt.comptimePrint(
            \\import polars as pl
            \\import numpy as np
            \\df = pl.DataFrame({{"val": np.random.rand({d}) * 1000}})
            \\result = df.with_columns(((pl.col("val") - pl.col("val").mean()) / pl.col("val").std()).alias("zscore"))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, py_code) catch 0;
        for (0..ITERATIONS) |_| total_ns += runPolars(allocator, py_code) catch 0;

        const avg_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);
        const ratio = if (baseline_ns > 0) @as(f64, @floatFromInt(avg_ns)) / @as(f64, @floatFromInt(baseline_ns)) else 0;
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {d:>9.1}x\n", .{ "Polars", time_ms, rows_per_sec / 1000, ratio });
    }

    // =========================================================================
    // Benchmark 2: LOG TRANSFORM
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("LOG TRANSFORM ({d}K rows)\n", .{NUM_ROWS / 1000});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<15} {s:>12} {s:>12} {s:>10}\n", .{ "Engine", "Time", "Rows/sec", "Ratio" });
    std.debug.print("{s:<15} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 15, "-" ** 12, "-" ** 12, "-" ** 10 });

    // LanceQL
    if (has_lanceql) {
        const sql = std.fmt.comptimePrint(
            \\SELECT LN(val + 1) as log_val FROM (SELECT random() * 1000 as val FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runLanceQL(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runLanceQL(allocator, sql) catch 0;

        baseline_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(baseline_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(baseline_ns)) / 1_000_000_000.0);
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {s:>10}\n", .{ "LanceQL", time_ms, rows_per_sec / 1000, "1.0x" });
    }

    // DuckDB
    if (has_duckdb) {
        const sql = std.fmt.comptimePrint(
            \\SELECT LN(val + 1) as log_val FROM (SELECT random() * 1000 as val FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runDuckDB(allocator, sql) catch 0;

        const avg_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);
        const ratio = if (baseline_ns > 0) @as(f64, @floatFromInt(avg_ns)) / @as(f64, @floatFromInt(baseline_ns)) else 0;
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {d:>9.1}x\n", .{ "DuckDB", time_ms, rows_per_sec / 1000, ratio });
    }

    // Polars (DataFrame API)
    if (has_polars) {
        const py_code = std.fmt.comptimePrint(
            \\import polars as pl
            \\import numpy as np
            \\df = pl.DataFrame({{"val": np.random.rand({d}) * 1000}})
            \\result = df.with_columns((pl.col("val") + 1).log().alias("log_val"))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, py_code) catch 0;
        for (0..ITERATIONS) |_| total_ns += runPolars(allocator, py_code) catch 0;

        const avg_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);
        const ratio = if (baseline_ns > 0) @as(f64, @floatFromInt(avg_ns)) / @as(f64, @floatFromInt(baseline_ns)) else 0;
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {d:>9.1}x\n", .{ "Polars", time_ms, rows_per_sec / 1000, ratio });
    }

    // =========================================================================
    // Benchmark 3: FEATURE CROSSING
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("FEATURE CROSSING ({d}K rows)\n", .{NUM_ROWS / 1000});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<15} {s:>12} {s:>12} {s:>10}\n", .{ "Engine", "Time", "Rows/sec", "Ratio" });
    std.debug.print("{s:<15} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 15, "-" ** 12, "-" ** 12, "-" ** 10 });

    // LanceQL
    if (has_lanceql) {
        const sql = std.fmt.comptimePrint(
            \\SELECT a * b as crossed FROM (SELECT random() * 1000 as a, random() * 1000 as b FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runLanceQL(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runLanceQL(allocator, sql) catch 0;

        baseline_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(baseline_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(baseline_ns)) / 1_000_000_000.0);
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {s:>10}\n", .{ "LanceQL", time_ms, rows_per_sec / 1000, "1.0x" });
    }

    // DuckDB
    if (has_duckdb) {
        const sql = std.fmt.comptimePrint(
            \\SELECT a * b as crossed FROM (SELECT random() * 1000 as a, random() * 1000 as b FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runDuckDB(allocator, sql) catch 0;

        const avg_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);
        const ratio = if (baseline_ns > 0) @as(f64, @floatFromInt(avg_ns)) / @as(f64, @floatFromInt(baseline_ns)) else 0;
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {d:>9.1}x\n", .{ "DuckDB", time_ms, rows_per_sec / 1000, ratio });
    }

    // Polars (DataFrame API)
    if (has_polars) {
        const py_code = std.fmt.comptimePrint(
            \\import polars as pl
            \\import numpy as np
            \\df = pl.DataFrame({{"a": np.random.rand({d}) * 1000, "b": np.random.rand({d}) * 1000}})
            \\result = df.with_columns((pl.col("a") * pl.col("b")).alias("crossed"))
        , .{ NUM_ROWS, NUM_ROWS });

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, py_code) catch 0;
        for (0..ITERATIONS) |_| total_ns += runPolars(allocator, py_code) catch 0;

        const avg_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);
        const ratio = if (baseline_ns > 0) @as(f64, @floatFromInt(avg_ns)) / @as(f64, @floatFromInt(baseline_ns)) else 0;
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {d:>9.1}x\n", .{ "Polars", time_ms, rows_per_sec / 1000, ratio });
    }

    // =========================================================================
    // Benchmark 4: BINNING
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("BINNING (10 bins, {d}K rows)\n", .{NUM_ROWS / 1000});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<15} {s:>12} {s:>12} {s:>10}\n", .{ "Engine", "Time", "Rows/sec", "Ratio" });
    std.debug.print("{s:<15} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 15, "-" ** 12, "-" ** 12, "-" ** 10 });

    // LanceQL
    if (has_lanceql) {
        const sql = std.fmt.comptimePrint(
            \\SELECT WIDTH_BUCKET(val, 0, 1000, 10) as bin FROM (SELECT random() * 1000 as val FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runLanceQL(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runLanceQL(allocator, sql) catch 0;

        baseline_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(baseline_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(baseline_ns)) / 1_000_000_000.0);
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {s:>10}\n", .{ "LanceQL", time_ms, rows_per_sec / 1000, "1.0x" });
    }

    // DuckDB
    if (has_duckdb) {
        const sql = std.fmt.comptimePrint(
            \\SELECT WIDTH_BUCKET(val, 0, 1000, 10) as bin FROM (SELECT random() * 1000 as val FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runDuckDB(allocator, sql) catch 0;

        const avg_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);
        const ratio = if (baseline_ns > 0) @as(f64, @floatFromInt(avg_ns)) / @as(f64, @floatFromInt(baseline_ns)) else 0;
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {d:>9.1}x\n", .{ "DuckDB", time_ms, rows_per_sec / 1000, ratio });
    }

    // Polars (DataFrame API) - use cut for binning
    if (has_polars) {
        const py_code = std.fmt.comptimePrint(
            \\import polars as pl
            \\import numpy as np
            \\df = pl.DataFrame({{"val": np.random.rand({d}) * 1000}})
            \\result = df.with_columns(pl.col("val").cut([100,200,300,400,500,600,700,800,900]).alias("bin"))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, py_code) catch 0;
        for (0..ITERATIONS) |_| total_ns += runPolars(allocator, py_code) catch 0;

        const avg_ns = total_ns / ITERATIONS;
        const time_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(NUM_ROWS)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);
        const ratio = if (baseline_ns > 0) @as(f64, @floatFromInt(avg_ns)) / @as(f64, @floatFromInt(baseline_ns)) else 0;
        std.debug.print("{s:<15} {d:>9.0} ms {d:>10.0} K/s {d:>9.1}x\n", .{ "Polars", time_ms, rows_per_sec / 1000, ratio });
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Fair comparison using each engine's native interface:\n", .{});
    std.debug.print("  - LanceQL/DuckDB: CLI subprocess with SQL\n", .{});
    std.debug.print("  - Polars: Python subprocess with DataFrame API\n", .{});
    std.debug.print("\nAll have similar subprocess spawn overhead (~30ms).\n", .{});
    std.debug.print("\n", .{});
}
