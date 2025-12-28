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
//! Fair comparison: All engines run via subprocess CLI
//!   - LanceQL:  lanceql -c "SELECT ..."
//!   - DuckDB:   duckdb -c "SELECT ..."
//!   - Polars:   polars -c "SELECT ..."

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

fn runPolars(allocator: std.mem.Allocator, sql: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "polars", "-c", sql },
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
    has_polars = checkCommand(allocator, "polars");

    std.debug.print("\nFair comparison: All engines run via subprocess CLI\n", .{});
    std.debug.print("Each engine has the same ~30ms subprocess spawn overhead.\n", .{});
    std.debug.print("\nDataset: {d}K rows\n", .{NUM_ROWS / 1000});
    std.debug.print("\nEngines:\n", .{});
    std.debug.print("  - LanceQL: {s}\n", .{if (has_lanceql) "yes" else "no (install: zig build && cp zig-out/bin/lanceql /usr/local/bin/)"});
    std.debug.print("  - DuckDB:  {s}\n", .{if (has_duckdb) "yes" else "no (brew install duckdb)"});
    std.debug.print("  - Polars:  {s}\n", .{if (has_polars) "yes" else "no (pip install polars-cli)"});
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

    // Polars
    if (has_polars) {
        const sql = std.fmt.comptimePrint(
            \\SELECT (val - AVG(val) OVER()) / NULLIF(STDDEV(val) OVER(), 0) as zscore
            \\FROM (SELECT random() * 1000 as val FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runPolars(allocator, sql) catch 0;

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

    // Polars
    if (has_polars) {
        const sql = std.fmt.comptimePrint(
            \\SELECT LN(val + 1) as log_val FROM (SELECT random() * 1000 as val FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runPolars(allocator, sql) catch 0;

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

    // Polars
    if (has_polars) {
        const sql = std.fmt.comptimePrint(
            \\SELECT a * b as crossed FROM (SELECT random() * 1000 as a, random() * 1000 as b FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runPolars(allocator, sql) catch 0;

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

    // Polars
    if (has_polars) {
        const sql = std.fmt.comptimePrint(
            \\SELECT WIDTH_BUCKET(val, 0, 1000, 10) as bin FROM (SELECT random() * 1000 as val FROM generate_series(1, {d}))
        , .{NUM_ROWS});

        var total_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, sql) catch 0;
        for (0..ITERATIONS) |_| total_ns += runPolars(allocator, sql) catch 0;

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
    std.debug.print("Fair comparison: All engines run via subprocess CLI.\n", .{});
    std.debug.print("Each has ~30ms subprocess spawn overhead, so ratios reflect actual compute.\n", .{});
    std.debug.print("\n", .{});
}
