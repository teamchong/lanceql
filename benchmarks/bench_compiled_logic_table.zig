//! Compiled @logic_table Benchmark - End-to-End Comparison
//!
//! HONEST benchmark measuring the FULL pipeline from cold start:
//!   1. LanceQL @logic_table  - Read Lance → compute with compiled extern function
//!   2. DuckDB + Python loop  - Read Parquet → per-row Python function calls
//!   3. DuckDB → NumPy batch  - Read Parquet → NumPy vectorized dot product
//!   4. Polars + Python loop  - Read Parquet → per-row Python function calls
//!   5. Polars → NumPy batch  - Read Parquet → NumPy vectorized dot product
//!
//! FAIR COMPARISON:
//!   - All methods read REAL data from disk (Lance or Parquet files)
//!   - All methods run for exactly 15 seconds
//!   - Throughput measured as rows processed per second
//!   - Uses 'amount' column as vector data (since embedding list columns not yet supported)
//!
//! Setup:
//!   python3 benchmarks/generate_benchmark_data.py  # Creates test data
//!   zig build bench-compiled-logic-table

const std = @import("std");
const Table = @import("lanceql.table").Table;

// Extern declarations for compiled @logic_table functions
extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) f64;

const WARMUP_SECONDS = 2;
const BENCHMARK_SECONDS = 15;
const LANCE_PATH = "benchmarks/benchmark_e2e.lance";
const PARQUET_PATH = "benchmarks/benchmark_e2e.parquet";

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

fn readLanceFile(allocator: std.mem.Allocator) ![]const u8 {
    var data_dir = std.fs.cwd().openDir(LANCE_PATH ++ "/data", .{ .iterate = true }) catch return error.FileNotFound;
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

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Compiled @logic_table Benchmark: End-to-End\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\nEach method runs for {d} seconds. Measuring throughput (rows/sec).\n", .{BENCHMARK_SECONDS});
    std.debug.print("Operation: dot product of amount column with query vector\n", .{});
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

    // Check for Python modules
    const has_duckdb = checkPythonModule(allocator, "duckdb");
    const has_polars = checkPythonModule(allocator, "polars");

    std.debug.print("\nEngines:\n", .{});
    std.debug.print("  LanceQL @logic_table: yes (compiled via metal0)\n", .{});
    std.debug.print("  DuckDB:               {s}\n", .{if (has_duckdb) "yes" else "no"});
    std.debug.print("  Polars:               {s}\n", .{if (has_polars) "yes" else "no"});
    std.debug.print("\n", .{});

    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Rows/sec", "Iterations", "Speedup" });
    std.debug.print("================================================================================\n", .{});

    var lanceql_rows_per_sec: f64 = 0;

    // =========================================================================
    // LanceQL @logic_table - Read REAL data from Lance file, use compiled extern
    // =========================================================================
    {
        var iterations: u64 = 0;
        var total_rows: u64 = 0;

        // Pre-allocate query vector (filled once, reused)
        var query_buf: [100000]f64 = undefined;

        // Warmup
        const warmup_end = std.time.nanoTimestamp() + WARMUP_SECONDS * 1_000_000_000;
        while (std.time.nanoTimestamp() < warmup_end) {
            const file_data = readLanceFile(allocator) catch break;
            defer allocator.free(file_data);

            var table = Table.init(allocator, file_data) catch break;
            defer table.deinit();

            // Read REAL 'amount' column from Lance file
            const amounts = table.readFloat64Column(1) catch break;
            defer allocator.free(amounts);

            // Create query vector of same length
            const query_len = @min(amounts.len, query_buf.len);
            for (0..query_len) |i| {
                query_buf[i] = 1.0; // Unit query vector
            }

            // Use REAL compiled @logic_table extern function
            const dot = VectorOps_dot_product(amounts.ptr, &query_buf, query_len);
            std.mem.doNotOptimizeAway(&dot);
        }

        // Benchmark
        const benchmark_end_time = std.time.nanoTimestamp() + BENCHMARK_SECONDS * 1_000_000_000;
        const start_time = std.time.nanoTimestamp();
        while (std.time.nanoTimestamp() < benchmark_end_time) {
            const file_data = readLanceFile(allocator) catch break;
            defer allocator.free(file_data);

            var table = Table.init(allocator, file_data) catch break;
            defer table.deinit();

            // Read REAL 'amount' column from Lance file
            const amounts = table.readFloat64Column(1) catch break;
            defer allocator.free(amounts);

            // Create query vector of same length
            const query_len = @min(amounts.len, query_buf.len);
            for (0..query_len) |i| {
                query_buf[i] = 1.0;
            }

            // Use REAL compiled @logic_table extern function
            const dot = VectorOps_dot_product(amounts.ptr, &query_buf, query_len);
            std.mem.doNotOptimizeAway(&dot);

            iterations += 1;
            total_rows += amounts.len;
        }

        const elapsed_ns = std.time.nanoTimestamp() - start_time;
        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        lanceql_rows_per_sec = @as(f64, @floatFromInt(total_rows)) / elapsed_s;

        std.debug.print("{s:<40} {d:>10.0}K {d:>12} {s:>10}\n", .{
            "@logic_table (compiled)",
            lanceql_rows_per_sec / 1000.0,
            iterations,
            "1.0x",
        });
    }

    // =========================================================================
    // DuckDB + Python loop - Read Parquet, per-row Python function calls
    // =========================================================================
    if (has_duckdb) duckdb_udf: {
        const py_script = std.fmt.comptimePrint(
            \\import duckdb
            \\import numpy as np
            \\import time
            \\
            \\WARMUP_SECONDS = {d}
            \\BENCHMARK_SECONDS = {d}
            \\
            \\con = duckdb.connect()
            \\con.execute("SET enable_progress_bar = false")
            \\
            \\# Python function called per-row (slow due to interpreter overhead)
            \\def multiply_by_one(amount):
            \\    return float(amount * 1.0)
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    amounts = con.execute("SELECT amount FROM read_parquet('{s}') LIMIT 100").fetchnumpy()['amount']
            \\    for a in amounts:
            \\        _ = multiply_by_one(a)
            \\
            \\# Benchmark: Per-row Python function calls (shows interpreter overhead)
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    amounts = con.execute("SELECT amount FROM read_parquet('{s}')").fetchnumpy()['amount']
            \\    total = 0.0
            \\    for a in amounts:
            \\        total += multiply_by_one(a)  # Per-row Python call
            \\    total_rows += len(amounts)
            \\    iterations += 1
            \\
            \\con.close()
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
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB Python UDF", "error", "-", "-" });
            break :duckdb_udf;
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
            const speedup = lanceql_rows_per_sec / rows_per_sec;
            std.debug.print("{s:<40} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "DuckDB + Python loop",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        } else {
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB + Python loop", "error", "-", "-" });
        }
    }

    // =========================================================================
    // DuckDB → NumPy batch - Read Parquet, compute dot product
    // =========================================================================
    if (has_duckdb) duckdb_numpy: {
        const py_script = std.fmt.comptimePrint(
            \\import duckdb
            \\import numpy as np
            \\import time
            \\
            \\WARMUP_SECONDS = {d}
            \\BENCHMARK_SECONDS = {d}
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    con = duckdb.connect()
            \\    con.execute("SET enable_progress_bar = false")
            \\    amounts = con.execute("SELECT amount FROM read_parquet('{s}')").fetchnumpy()['amount']
            \\    query = np.ones(len(amounts))
            \\    dot = np.dot(amounts, query)
            \\    con.close()
            \\
            \\# Benchmark
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    con = duckdb.connect()
            \\    con.execute("SET enable_progress_bar = false")
            \\    amounts = con.execute("SELECT amount FROM read_parquet('{s}')").fetchnumpy()['amount']
            \\    query = np.ones(len(amounts))
            \\    dot = np.dot(amounts, query)
            \\    total_rows += len(amounts)
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
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB → NumPy batch", "error", "-", "-" });
            break :duckdb_numpy;
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
            const speedup = lanceql_rows_per_sec / rows_per_sec;
            std.debug.print("{s:<40} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "DuckDB → NumPy batch",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        } else {
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB → NumPy batch", "error", "-", "-" });
        }
    }

    // =========================================================================
    // Polars + Python loop - Read Parquet, per-row Python function calls
    // =========================================================================
    if (has_polars) polars_udf: {
        const py_script = std.fmt.comptimePrint(
            \\import warnings
            \\warnings.filterwarnings('ignore')
            \\import polars as pl
            \\import time
            \\
            \\WARMUP_SECONDS = {d}
            \\BENCHMARK_SECONDS = {d}
            \\
            \\# Python function called per-row (slow due to interpreter overhead)
            \\def multiply_by_one(amount):
            \\    return float(amount * 1.0)
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    df = pl.read_parquet("{s}")
            \\    amounts = df["amount"].to_list()
            \\    for a in amounts[:100]:
            \\        _ = multiply_by_one(a)
            \\
            \\# Benchmark: Per-row Python function calls (shows interpreter overhead)
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = pl.read_parquet("{s}")
            \\    amounts = df["amount"].to_list()
            \\    total = 0.0
            \\    for a in amounts:
            \\        total += multiply_by_one(a)  # Per-row Python call
            \\    total_rows += len(amounts)
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
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "Polars Python UDF", "error", "-", "-" });
            break :polars_udf;
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
            const speedup = lanceql_rows_per_sec / rows_per_sec;
            std.debug.print("{s:<40} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "Polars + Python loop",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        } else {
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "Polars + Python loop", "error", "-", "-" });
        }
    }

    // =========================================================================
    // Polars → NumPy batch
    // =========================================================================
    if (has_polars) polars_numpy: {
        const py_script = std.fmt.comptimePrint(
            \\import polars as pl
            \\import numpy as np
            \\import time
            \\
            \\WARMUP_SECONDS = {d}
            \\BENCHMARK_SECONDS = {d}
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    df = pl.read_parquet("{s}")
            \\    amounts = df["amount"].to_numpy()
            \\    query = np.ones(len(amounts))
            \\    dot = np.dot(amounts, query)
            \\
            \\# Benchmark
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = pl.read_parquet("{s}")
            \\    amounts = df["amount"].to_numpy()
            \\    query = np.ones(len(amounts))
            \\    dot = np.dot(amounts, query)
            \\    total_rows += len(amounts)
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
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "Polars → NumPy batch", "error", "-", "-" });
            break :polars_numpy;
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
            const speedup = lanceql_rows_per_sec / rows_per_sec;
            std.debug.print("{s:<40} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "Polars → NumPy batch",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        } else {
            std.debug.print("{s:<40} {s:>12} {s:>12} {s:>10}\n", .{ "Polars → NumPy batch", "error", "-", "-" });
        }
    }

    std.debug.print("================================================================================\n", .{});
    std.debug.print("\nNotes:\n", .{});
    std.debug.print("  - @logic_table uses VectorOps_dot_product extern function (compiled Python)\n", .{});
    std.debug.print("  - All methods read REAL 'amount' column from disk (Lance or Parquet files)\n", .{});
    std.debug.print("  - All methods run for exactly 15 seconds\n", .{});
    std.debug.print("  - Throughput = total rows processed / elapsed time\n", .{});
    std.debug.print("\n", .{});
}
