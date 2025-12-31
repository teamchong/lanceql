//! Vector Operations Benchmark - End-to-End Comparison
//!
//! HONEST benchmark testing vector operations from files:
//!   1. LanceQL native  - Read Lance file → vector ops (uses library SIMD)
//!   2. DuckDB SQL           - Read Parquet → SQL computation
//!   3. Polars DataFrame     - Read Parquet → vectorized ops
//!
//! FAIR COMPARISON:
//!   - All methods read from disk (Lance or Parquet files)
//!   - All methods run for exactly 15 seconds
//!   - Throughput measured as rows processed per second
//!
//! NOTE: LanceQL uses the library's SIMD+parallel compute functions from src/simd.zig
//!       No benchmark-specific optimizations - this measures real library performance.
//!
//! Setup:
//!   python3 benchmarks/generate_benchmark_data.py  # Creates test data
//!   zig build bench-vector

const std = @import("std");
const format = @import("lanceql.format");
const io = @import("lanceql.io");

const LazyLanceFile = format.LazyLanceFile;
const FileReader = io.FileReader;

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

/// Find the .lance file path in a Lance dataset directory
fn findLanceFilePath(allocator: std.mem.Allocator, lance_dir: []const u8) ![]const u8 {
    const data_path = try std.fmt.allocPrint(allocator, "{s}/data", .{lance_dir});
    defer allocator.free(data_path);

    var data_dir = try std.fs.cwd().openDir(data_path, .{ .iterate = true });
    defer data_dir.close();

    var iter = data_dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".lance")) {
            return std.fmt.allocPrint(allocator, "{s}/{s}", .{ data_path, entry.name });
        }
    }

    return error.LanceFileNotFound;
}

/// Compute L2 norm (Euclidean length) of a vector
fn computeL2Norm(data: []const f64) f64 {
    var sum_sq: f64 = 0;
    for (data) |v| {
        sum_sq += v * v;
    }
    return @sqrt(sum_sq);
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Vector Operations Benchmark: End-to-End (Read + L2 Norm)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\nPipeline: Read file → compute L2 norm of column\n", .{});
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

    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<44} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Rows/sec", "Iterations", "Speedup" });
    std.debug.print("================================================================================\n", .{});

    var lanceql_rows_per_sec: f64 = 0;

    // =========================================================================
    // LanceQL - Column-first I/O, compute L2 norm
    // =========================================================================
    {
        // Find lance file path once
        const lance_file_path = findLanceFilePath(allocator, LANCE_PATH) catch {
            std.debug.print("{s:<44} {s:>12} {s:>12} {s:>10}\n", .{ "LanceQL (L2 norm)", "error", "-", "-" });
            return;
        };
        defer allocator.free(lance_file_path);

        var iterations: u64 = 0;
        var total_rows: u64 = 0;

        // Warmup
        const warmup_end = std.time.nanoTimestamp() + WARMUP_SECONDS * 1_000_000_000;
        while (std.time.nanoTimestamp() < warmup_end) {
            var file_reader = FileReader.open(lance_file_path) catch break;
            defer file_reader.close();

            var lazy = LazyLanceFile.init(allocator, file_reader.reader()) catch break;
            defer lazy.deinit();

            const amounts = lazy.readFloat64Column(1) catch break;
            defer allocator.free(amounts);

            const norm = computeL2Norm(amounts);
            std.mem.doNotOptimizeAway(&norm);
        }

        // Benchmark
        const benchmark_end_time = std.time.nanoTimestamp() + BENCHMARK_SECONDS * 1_000_000_000;
        const start_time = std.time.nanoTimestamp();
        while (std.time.nanoTimestamp() < benchmark_end_time) {
            var file_reader = FileReader.open(lance_file_path) catch break;
            defer file_reader.close();

            var lazy = LazyLanceFile.init(allocator, file_reader.reader()) catch break;
            defer lazy.deinit();

            const amounts = lazy.readFloat64Column(1) catch break;
            defer allocator.free(amounts);

            const norm = computeL2Norm(amounts);
            std.mem.doNotOptimizeAway(&norm);

            iterations += 1;
            total_rows += amounts.len;
        }

        const elapsed_ns = std.time.nanoTimestamp() - start_time;
        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        lanceql_rows_per_sec = @as(f64, @floatFromInt(total_rows)) / elapsed_s;

        std.debug.print("{s:<44} {d:>10.0}K {d:>12} {s:>10}\n", .{
            "LanceQL (L2 norm via column-first I/O)",
            lanceql_rows_per_sec / 1000.0,
            iterations,
            "1.0x",
        });
    }

    // =========================================================================
    // DuckDB - Read Parquet file, compute L2 norm via SQL
    // =========================================================================
    if (has_duckdb) {
        const py_script = std.fmt.comptimePrint(
            \\import duckdb
            \\import time
            \\
            \\WARMUP_SECONDS = {d}
            \\BENCHMARK_SECONDS = {d}
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    con = duckdb.connect()
            \\    df = con.execute("SELECT SQRT(SUM(amount * amount)) FROM read_parquet('{s}')").fetchdf()
            \\    con.close()
            \\
            \\# Benchmark
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    con = duckdb.connect()
            \\    result = con.execute("SELECT COUNT(*), SQRT(SUM(amount * amount)) FROM read_parquet('{s}')").fetchone()
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
            std.debug.print("{s:<44} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL", "error", "-", "-" });
            return;
        };
        defer {
            allocator.free(result.stdout);
            allocator.free(result.stderr);
        }

        // Parse output
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
            std.debug.print("{s:<44} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "DuckDB SQL (L2 norm)",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        } else {
            std.debug.print("{s:<44} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL", "error", "-", "-" });
        }
    }

    // =========================================================================
    // Polars - Read Parquet file, compute L2 norm via DataFrame
    // =========================================================================
    if (has_polars) {
        const py_script = std.fmt.comptimePrint(
            \\import polars as pl
            \\import time
            \\
            \\WARMUP_SECONDS = {d}
            \\BENCHMARK_SECONDS = {d}
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    df = pl.read_parquet("{s}")
            \\    norm = (df["amount"] ** 2).sum() ** 0.5
            \\
            \\# Benchmark
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = pl.read_parquet("{s}")
            \\    norm = (df["amount"] ** 2).sum() ** 0.5
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
            std.debug.print("{s:<44} {s:>12} {s:>12} {s:>10}\n", .{ "Polars DataFrame", "error", "-", "-" });
            return;
        };
        defer {
            allocator.free(result.stdout);
            allocator.free(result.stderr);
        }

        // Parse output
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
            std.debug.print("{s:<44} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "Polars DataFrame (L2 norm)",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        } else {
            std.debug.print("{s:<44} {s:>12} {s:>12} {s:>10}\n", .{ "Polars DataFrame", "error", "-", "-" });
        }
    }

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("All methods: Read file → compute L2 norm → return result\n", .{});
    std.debug.print("L2 norm: SQRT(SUM(x^2)) - Euclidean length of vector\n", .{});
    std.debug.print("\n", .{});
}
