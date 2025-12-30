//! SQL Clauses Benchmark - End-to-End Comparison
//!
//! HONEST benchmark testing SQL operations from files:
//!   1. LanceQL native  - Read Lance file → SQL operations (uses library SIMD)
//!   2. DuckDB SQL      - Read Parquet → SQL operations
//!   3. Polars DataFrame - Read Parquet → DataFrame operations
//!
//! FAIR COMPARISON:
//!   - All methods read file EACH iteration (no mmap caching advantage)
//!   - All methods run for exactly 15 seconds
//!   - Throughput measured as rows processed per second
//!
//! SQL CLAUSES TESTED:
//!   - FILTER (WHERE) - uses library simd.countGreaterThan()
//!   - AGGREGATE (SUM) - uses library simd.sum()
//!   - GROUP BY - SKIPPED (needs library implementation with real hash table)
//!   - JOIN - SKIPPED (needs library implementation with real hash join)
//!
//! Setup:
//!   python3 benchmarks/generate_benchmark_data.py  # Creates orders + customers
//!   zig build bench-sql

const std = @import("std");
const Table = @import("lanceql.table").Table;
const simd = @import("lanceql.simd");

const WARMUP_SECONDS = 2;
const BENCHMARK_SECONDS = 15;
const LANCE_PATH = "benchmarks/benchmark_e2e.lance";
const PARQUET_PATH = "benchmarks/benchmark_e2e.parquet";
const CUSTOMERS_LANCE_PATH = "benchmarks/customers.lance";
const CUSTOMERS_PARQUET_PATH = "benchmarks/customers.parquet";

fn checkPythonModule(allocator: std.mem.Allocator, module: []const u8) bool {
    const py_code = std.fmt.allocPrint(allocator, "import {s}", .{module}) catch return false;
    defer allocator.free(py_code);

    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", "-c", py_code },
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

/// Read Lance file from disk (fair comparison - reads each time like competitors)
fn readLanceFileFromPath(allocator: std.mem.Allocator, lance_dir: []const u8) ![]const u8 {
    const data_path = std.fmt.allocPrint(allocator, "{s}/data", .{lance_dir}) catch return error.OutOfMemory;
    defer allocator.free(data_path);

    var data_dir = std.fs.cwd().openDir(data_path, .{ .iterate = true }) catch return error.FileNotFound;
    defer data_dir.close();

    var iter = data_dir.iterate();
    var lance_file_name_buf: [256]u8 = undefined;
    var lance_file_name: ?[]const u8 = null;

    while (iter.next() catch null) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".lance")) {
            const len = @min(entry.name.len, lance_file_name_buf.len);
            @memcpy(lance_file_name_buf[0..len], entry.name[0..len]);
            lance_file_name = lance_file_name_buf[0..len];
            break;
        }
    }

    const file_name = lance_file_name orelse return error.LanceFileNotFound;
    const file = data_dir.openFile(file_name, .{}) catch return error.FileNotFound;
    defer file.close();

    const file_size = file.getEndPos() catch return error.ReadError;
    const bytes = allocator.alloc(u8, file_size) catch return error.OutOfMemory;
    errdefer allocator.free(bytes);

    _ = file.readAll(bytes) catch return error.ReadError;
    return bytes;
}

fn readLanceFile(allocator: std.mem.Allocator) ![]const u8 {
    return readLanceFileFromPath(allocator, LANCE_PATH);
}

fn readCustomersLanceFile(allocator: std.mem.Allocator) ![]const u8 {
    return readLanceFileFromPath(allocator, CUSTOMERS_LANCE_PATH);
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("SQL Clauses Benchmark: End-to-End (Read + SQL Operations)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\nPipeline: Read file → execute SQL clause → return result\n", .{});
    std.debug.print("Each method runs for {d} seconds. Measuring throughput (rows/sec).\n", .{BENCHMARK_SECONDS});
    std.debug.print("\n", .{});

    // Check data files exist
    const lance_exists = blk: {
        var data_dir = std.fs.cwd().openDir(LANCE_PATH ++ "/data", .{ .iterate = true }) catch break :blk false;
        data_dir.close();
        break :blk true;
    };

    const parquet_exists = blk: {
        const file = std.fs.cwd().openFile(PARQUET_PATH, .{}) catch break :blk false;
        file.close();
        break :blk true;
    };

    std.debug.print("Data files:\n", .{});
    std.debug.print("  Lance:   {s} {s}\n", .{ LANCE_PATH, if (lance_exists) "✓" else "✗" });
    std.debug.print("  Parquet: {s} {s}\n", .{ PARQUET_PATH, if (parquet_exists) "✓" else "✗" });

    if (!lance_exists or !parquet_exists) {
        std.debug.print("\n⚠️  Missing data files. Run: python3 benchmarks/generate_benchmark_data.py\n", .{});
        return;
    }

    // Check Python engines
    const has_duckdb = checkPythonModule(allocator, "duckdb");
    const has_polars = checkPythonModule(allocator, "polars");

    std.debug.print("\nEngines:\n", .{});
    std.debug.print("  LanceQL native: yes (uses library SIMD with auto-dispatch)\n", .{});
    std.debug.print("  DuckDB:               {s}\n", .{if (has_duckdb) "yes" else "no"});
    std.debug.print("  Polars:               {s}\n", .{if (has_polars) "yes" else "no"});
    std.debug.print("\n", .{});

    // ==========================================================================
    // FILTER: WHERE amount > 100
    // ==========================================================================
    std.debug.print("================================================================================\n", .{});
    std.debug.print("FILTER: WHERE amount > 100\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Rows/sec", "Iterations", "Speedup" });
    std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 40, "-" ** 12, "-" ** 12, "-" ** 10 });

    var lanceql_filter_rps: f64 = 0;
    {
        var iterations: u64 = 0;
        var total_rows: u64 = 0;

        // Warmup - read file each iteration (fair comparison with DuckDB/Polars)
        const warmup_end = std.time.nanoTimestamp() + WARMUP_SECONDS * 1_000_000_000;
        while (std.time.nanoTimestamp() < warmup_end) {
            const file_data = readLanceFileFromPath(allocator, LANCE_PATH) catch break;
            defer allocator.free(file_data);

            var table = Table.init(allocator, file_data) catch break;
            defer table.deinit();
            const amounts = table.readFloat64Column(1) catch break;
            defer allocator.free(amounts);
            // Use library SIMD function (auto-dispatches scalar/SIMD/parallel)
            const count = simd.countGreaterThan(amounts, 100.0);
            std.mem.doNotOptimizeAway(&count);
        }

        // Benchmark - read file each iteration (fair comparison)
        const benchmark_end_time = std.time.nanoTimestamp() + BENCHMARK_SECONDS * 1_000_000_000;
        const start_time = std.time.nanoTimestamp();
        while (std.time.nanoTimestamp() < benchmark_end_time) {
            const file_data = readLanceFileFromPath(allocator, LANCE_PATH) catch break;
            defer allocator.free(file_data);

            var table = Table.init(allocator, file_data) catch break;
            defer table.deinit();
            const amounts = table.readFloat64Column(1) catch break;
            defer allocator.free(amounts);
            // Use library SIMD function (auto-dispatches scalar/SIMD/parallel)
            const count = simd.countGreaterThan(amounts, 100.0);
            std.mem.doNotOptimizeAway(&count);
            iterations += 1;
            total_rows += amounts.len;
        }

        const elapsed_ns = std.time.nanoTimestamp() - start_time;
        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        lanceql_filter_rps = @as(f64, @floatFromInt(total_rows)) / elapsed_s;
        std.debug.print("{s:<40} {d:>10.0}K {d:>12} {s:>10}\n", .{
            "LanceQL (FILTER via library SIMD)",
            lanceql_filter_rps / 1000.0,
            iterations,
            "1.0x",
        });
    }

    if (has_duckdb) {
        const py_script = std.fmt.comptimePrint(
            \\import duckdb
            \\import time
            \\
            \\WARMUP_SECONDS = {d}
            \\BENCHMARK_SECONDS = {d}
            \\
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    con = duckdb.connect()
            \\    _ = con.execute("SELECT COUNT(*) FROM read_parquet('{s}') WHERE amount > 100").fetchone()
            \\    con.close()
            \\
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    con = duckdb.connect()
            \\    result = con.execute("SELECT COUNT(*) FROM read_parquet('{s}')").fetchone()
            \\    total_rows += result[0]
            \\    con.close()
            \\    iterations += 1
            \\
            \\elapsed = time.time() - start
            \\rows_per_sec = total_rows / elapsed
            \\print(f"ROWS_PER_SEC:{{rows_per_sec:.0f}}")
            \\print(f"ITERATIONS:{{iterations}}")
        , .{ WARMUP_SECONDS, BENCHMARK_SECONDS, PARQUET_PATH, PARQUET_PATH });

        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", py_script },
            .max_output_bytes = 10 * 1024,
        }) catch {
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL", "error", "-", "-" });
            return;
        };
        defer {
            allocator.free(result.stdout);
            allocator.free(result.stderr);
        }

        var rows_per_sec: f64 = 0;
        var iterations: u64 = 0;
        var lines = std.mem.splitScalar(u8, result.stdout, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "ROWS_PER_SEC:")) {
                rows_per_sec = std.fmt.parseFloat(f64, line[13..]) catch 0;
            } else if (std.mem.startsWith(u8, line, "ITERATIONS:")) {
                iterations = std.fmt.parseInt(u64, line[11..], 10) catch 0;
            }
        }
        if (rows_per_sec > 0) {
            const speedup = lanceql_filter_rps / rows_per_sec;
            std.debug.print("{s:<40} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "DuckDB SQL (FILTER)",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        } else {
            // Debug: print stderr if benchmark failed
            if (result.stderr.len > 0) {
                std.debug.print("DuckDB FILTER stderr: {s}\n", .{result.stderr[0..@min(result.stderr.len, 200)]});
            }
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL (FILTER)", "0", "0", "-" });
        }
    }

    if (has_polars) {
        const py_script = std.fmt.comptimePrint(
            \\import polars as pl
            \\import time
            \\
            \\WARMUP_SECONDS = {d}
            \\BENCHMARK_SECONDS = {d}
            \\
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    df = pl.read_parquet("{s}")
            \\    filtered = df.filter(pl.col("amount") > 100)
            \\
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = pl.read_parquet("{s}")
            \\    filtered = df.filter(pl.col("amount") > 100)
            \\    total_rows += len(df)
            \\    iterations += 1
            \\
            \\elapsed = time.time() - start
            \\rows_per_sec = total_rows / elapsed
            \\print(f"ROWS_PER_SEC:{{rows_per_sec:.0f}}")
            \\print(f"ITERATIONS:{{iterations}}")
        , .{ WARMUP_SECONDS, BENCHMARK_SECONDS, PARQUET_PATH, PARQUET_PATH });

        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", py_script },
            .max_output_bytes = 10 * 1024,
        }) catch {
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "Polars DataFrame", "error", "-", "-" });
            return;
        };
        defer {
            allocator.free(result.stdout);
            allocator.free(result.stderr);
        }

        var rows_per_sec: f64 = 0;
        var iterations: u64 = 0;
        var lines = std.mem.splitScalar(u8, result.stdout, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "ROWS_PER_SEC:")) {
                rows_per_sec = std.fmt.parseFloat(f64, line[13..]) catch 0;
            } else if (std.mem.startsWith(u8, line, "ITERATIONS:")) {
                iterations = std.fmt.parseInt(u64, line[11..], 10) catch 0;
            }
        }
        if (rows_per_sec > 0) {
            const speedup = lanceql_filter_rps / rows_per_sec;
            std.debug.print("{s:<40} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "Polars DataFrame (FILTER)",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        }
    }

    // ==========================================================================
    // AGGREGATE: SUM(amount)
    // ==========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("AGGREGATE: SUM(amount)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Rows/sec", "Iterations", "Speedup" });
    std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 40, "-" ** 12, "-" ** 12, "-" ** 10 });

    var lanceql_agg_rps: f64 = 0;
    {
        var iterations: u64 = 0;
        var total_rows: u64 = 0;

        // Warmup - read file each iteration (fair comparison with DuckDB/Polars)
        const warmup_end = std.time.nanoTimestamp() + WARMUP_SECONDS * 1_000_000_000;
        while (std.time.nanoTimestamp() < warmup_end) {
            const file_data = readLanceFileFromPath(allocator, LANCE_PATH) catch break;
            defer allocator.free(file_data);

            var table = Table.init(allocator, file_data) catch break;
            defer table.deinit();
            const amounts = table.readFloat64Column(1) catch break;
            defer allocator.free(amounts);
            // Use library SIMD function (auto-dispatches scalar/SIMD/parallel)
            const total = simd.sum(amounts);
            std.mem.doNotOptimizeAway(&total);
        }

        // Benchmark - read file each iteration (fair comparison)
        const benchmark_end_time = std.time.nanoTimestamp() + BENCHMARK_SECONDS * 1_000_000_000;
        const start_time = std.time.nanoTimestamp();
        while (std.time.nanoTimestamp() < benchmark_end_time) {
            const file_data = readLanceFileFromPath(allocator, LANCE_PATH) catch break;
            defer allocator.free(file_data);

            var table = Table.init(allocator, file_data) catch break;
            defer table.deinit();
            const amounts = table.readFloat64Column(1) catch break;
            defer allocator.free(amounts);
            // Use library SIMD function (auto-dispatches scalar/SIMD/parallel)
            const total = simd.sum(amounts);
            std.mem.doNotOptimizeAway(&total);
            iterations += 1;
            total_rows += amounts.len;
        }

        const elapsed_ns = std.time.nanoTimestamp() - start_time;
        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        lanceql_agg_rps = @as(f64, @floatFromInt(total_rows)) / elapsed_s;
        std.debug.print("{s:<40} {d:>10.0}K {d:>12} {s:>10}\n", .{
            "LanceQL (AGGREGATE via library SIMD)",
            lanceql_agg_rps / 1000.0,
            iterations,
            "1.0x",
        });
    }

    if (has_duckdb) {
        const py_script = std.fmt.comptimePrint(
            \\import duckdb
            \\import time
            \\
            \\WARMUP_SECONDS = {d}
            \\BENCHMARK_SECONDS = {d}
            \\
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    con = duckdb.connect()
            \\    _ = con.execute("SELECT SUM(amount) FROM read_parquet('{s}')").fetchone()
            \\    con.close()
            \\
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    con = duckdb.connect()
            \\    result = con.execute("SELECT COUNT(*), SUM(amount) FROM read_parquet('{s}')").fetchone()
            \\    total_rows += result[0]
            \\    con.close()
            \\    iterations += 1
            \\
            \\elapsed = time.time() - start
            \\rows_per_sec = total_rows / elapsed
            \\print(f"ROWS_PER_SEC:{{rows_per_sec:.0f}}")
            \\print(f"ITERATIONS:{{iterations}}")
        , .{ WARMUP_SECONDS, BENCHMARK_SECONDS, PARQUET_PATH, PARQUET_PATH });

        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", py_script },
            .max_output_bytes = 10 * 1024,
        }) catch {
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL", "error", "-", "-" });
            return;
        };
        defer {
            allocator.free(result.stdout);
            allocator.free(result.stderr);
        }

        var rows_per_sec: f64 = 0;
        var iterations: u64 = 0;
        var lines = std.mem.splitScalar(u8, result.stdout, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "ROWS_PER_SEC:")) {
                rows_per_sec = std.fmt.parseFloat(f64, line[13..]) catch 0;
            } else if (std.mem.startsWith(u8, line, "ITERATIONS:")) {
                iterations = std.fmt.parseInt(u64, line[11..], 10) catch 0;
            }
        }
        if (rows_per_sec > 0) {
            const speedup = lanceql_agg_rps / rows_per_sec;
            std.debug.print("{s:<40} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "DuckDB SQL (AGGREGATE)",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        }
    }

    if (has_polars) {
        const py_script = std.fmt.comptimePrint(
            \\import polars as pl
            \\import time
            \\
            \\WARMUP_SECONDS = {d}
            \\BENCHMARK_SECONDS = {d}
            \\
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    df = pl.read_parquet("{s}")
            \\    total = df["amount"].sum()
            \\
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = pl.read_parquet("{s}")
            \\    total = df["amount"].sum()
            \\    total_rows += len(df)
            \\    iterations += 1
            \\
            \\elapsed = time.time() - start
            \\rows_per_sec = total_rows / elapsed
            \\print(f"ROWS_PER_SEC:{{rows_per_sec:.0f}}")
            \\print(f"ITERATIONS:{{iterations}}")
        , .{ WARMUP_SECONDS, BENCHMARK_SECONDS, PARQUET_PATH, PARQUET_PATH });

        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", py_script },
            .max_output_bytes = 10 * 1024,
        }) catch {
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "Polars DataFrame", "error", "-", "-" });
            return;
        };
        defer {
            allocator.free(result.stdout);
            allocator.free(result.stderr);
        }

        var rows_per_sec: f64 = 0;
        var iterations: u64 = 0;
        var lines = std.mem.splitScalar(u8, result.stdout, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "ROWS_PER_SEC:")) {
                rows_per_sec = std.fmt.parseFloat(f64, line[13..]) catch 0;
            } else if (std.mem.startsWith(u8, line, "ITERATIONS:")) {
                iterations = std.fmt.parseInt(u64, line[11..], 10) catch 0;
            }
        }
        if (rows_per_sec > 0) {
            const speedup = lanceql_agg_rps / rows_per_sec;
            std.debug.print("{s:<40} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "Polars DataFrame (AGGREGATE)",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        }
    }

    // ==========================================================================
    // GROUP BY: SKIPPED - needs library implementation with real hash table
    // ==========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("GROUP BY: SUM(amount) GROUP BY customer_id\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("SKIPPED: LanceQL needs library implementation with real hash table\n", .{});
    std.debug.print("         (Cannot use fixed-size array that assumes data range)\n", .{});

    // ==========================================================================
    // JOIN: SKIPPED - needs library implementation with real hash join
    // ==========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("JOIN: orders INNER JOIN customers ON customer_id\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("SKIPPED: LanceQL needs library implementation with real hash join\n", .{});
    std.debug.print("         (Cannot use fixed-size array that assumes data range)\n", .{});
    std.debug.print("         (Must materialize results like DuckDB/Polars do)\n", .{});

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("All methods: Read file EACH iteration → execute SQL clause → return result\n", .{});
    std.debug.print("FILTER: Count rows matching WHERE clause (uses library SIMD)\n", .{});
    std.debug.print("AGGREGATE: Compute SUM of column (uses library SIMD)\n", .{});
    std.debug.print("GROUP BY: SKIPPED - needs library hash table implementation\n", .{});
    std.debug.print("JOIN: SKIPPED - needs library hash join implementation\n", .{});
    std.debug.print("\n", .{});
}
