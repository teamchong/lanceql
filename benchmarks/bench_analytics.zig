//! Analytics Benchmark - End-to-End Comparison
//!
//! HONEST benchmark measuring analytics queries from cold start:
//!   1. LanceQL native  - Read Lance file → aggregate
//!   2. DuckDB SQL           - Read Parquet → SQL aggregations
//!   3. Polars DataFrame     - Read Parquet → DataFrame aggregations
//!
//! FAIR COMPARISON:
//!   - All methods read from disk (Lance or Parquet files)
//!   - All methods run for exactly 15 seconds
//!   - Throughput measured as rows processed per second
//!   - Operations: SUM, AVG, MIN, MAX on amount column
//!
//! Setup:
//!   python3 benchmarks/generate_benchmark_data.py  # Creates test data
//!   zig build bench-analytics

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

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Analytics Benchmark: End-to-End (SUM, AVG, MIN, MAX)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Each method runs for {d} seconds. Measuring throughput (rows/sec).\n", .{BENCHMARK_SECONDS});
    std.debug.print("\n", .{});

    // Check files exist
    const lance_exists = if (std.fs.cwd().access(LANCE_PATH, .{})) true else |_| false;
    const parquet_exists = if (std.fs.cwd().access(PARQUET_PATH, .{})) true else |_| false;

    if (!lance_exists or !parquet_exists) {
        std.debug.print("ERROR: Benchmark data not found. Run:\n", .{});
        std.debug.print("  python3 benchmarks/generate_benchmark_data.py\n", .{});
        return;
    }

    const has_duckdb = checkPythonModule(allocator, "duckdb");
    const has_polars = checkPythonModule(allocator, "polars");

    std.debug.print("Data files:\n", .{});
    std.debug.print("  Lance:   {s} ✓\n", .{LANCE_PATH});
    std.debug.print("  Parquet: {s} ✓\n", .{PARQUET_PATH});
    std.debug.print("\n", .{});
    std.debug.print("Engines:\n", .{});
    std.debug.print("  LanceQL native: yes\n", .{});
    std.debug.print("  DuckDB:               {s}\n", .{if (has_duckdb) "yes" else "no (pip install duckdb)"});
    std.debug.print("  Polars:               {s}\n", .{if (has_polars) "yes" else "no (pip install polars)"});
    std.debug.print("\n", .{});

    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Rows/sec", "Iterations", "Speedup" });
    std.debug.print("================================================================================\n", .{});

    var lanceql_throughput: f64 = 0;

    // 1. LanceQL native (column-first I/O → aggregate)
    {
        // Find lance file path once
        const lance_file_path = findLanceFilePath(allocator, LANCE_PATH) catch {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{ "LanceQL native", "error", "-", "-" });
            return;
        };
        defer allocator.free(lance_file_path);

        const warmup_end = std.time.nanoTimestamp() + WARMUP_SECONDS * std.time.ns_per_s;
        const benchmark_end_time = warmup_end + BENCHMARK_SECONDS * std.time.ns_per_s;

        var iterations: u64 = 0;
        var total_rows: u64 = 0;

        // Warmup
        while (std.time.nanoTimestamp() < warmup_end) {
            var file_reader = FileReader.open(lance_file_path) catch break;
            defer file_reader.close();

            var lazy = LazyLanceFile.init(allocator, file_reader.reader()) catch break;
            defer lazy.deinit();

            const amounts = lazy.readFloat64Column(1) catch break;
            defer allocator.free(amounts);

            var sum: f64 = 0;
            var min_val: f64 = amounts[0];
            var max_val: f64 = amounts[0];
            for (amounts) |v| {
                sum += v;
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
            const avg = sum / @as(f64, @floatFromInt(amounts.len));
            std.mem.doNotOptimizeAway(&sum);
            std.mem.doNotOptimizeAway(&avg);
            std.mem.doNotOptimizeAway(&min_val);
            std.mem.doNotOptimizeAway(&max_val);
        }

        // Benchmark
        const start_time = std.time.nanoTimestamp();
        while (std.time.nanoTimestamp() < benchmark_end_time) {
            var file_reader = FileReader.open(lance_file_path) catch break;
            defer file_reader.close();

            var lazy = LazyLanceFile.init(allocator, file_reader.reader()) catch break;
            defer lazy.deinit();

            const amounts = lazy.readFloat64Column(1) catch break;
            defer allocator.free(amounts);

            var sum: f64 = 0;
            var min_val: f64 = amounts[0];
            var max_val: f64 = amounts[0];
            for (amounts) |v| {
                sum += v;
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
            const avg = sum / @as(f64, @floatFromInt(amounts.len));
            std.mem.doNotOptimizeAway(&sum);
            std.mem.doNotOptimizeAway(&avg);
            std.mem.doNotOptimizeAway(&min_val);
            std.mem.doNotOptimizeAway(&max_val);

            iterations += 1;
            total_rows += amounts.len;
        }
        const elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - start_time);

        lanceql_throughput = @as(f64, @floatFromInt(total_rows)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0);
        std.debug.print("{s:<35} {d:>12.0} {d:>12} {s:>10}\n", .{
            "LanceQL native", lanceql_throughput, iterations, "1.0x",
        });
    }

    // 2. DuckDB SQL aggregations
    if (has_duckdb) duckdb_sql: {
        const script = std.fmt.comptimePrint(
            \\import duckdb
            \\import time
            \\
            \\BENCHMARK_SECONDS = {d}
            \\WARMUP_SECONDS = {d}
            \\PARQUET_PATH = "{s}"
            \\
            \\con = duckdb.connect()
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    con.execute(f"SELECT SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM read_parquet('{{PARQUET_PATH}}')").fetchall()
            \\
            \\# Benchmark
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    result = con.execute(f"SELECT SUM(amount), AVG(amount), MIN(amount), MAX(amount), COUNT(*) FROM read_parquet('{{PARQUET_PATH}}')").fetchall()
            \\    iterations += 1
            \\    total_rows += result[0][4]
            \\elapsed_ns = int((time.time() - start) * 1e9)
            \\
            \\print(f"ITERATIONS:{{iterations}}")
            \\print(f"TOTAL_NS:{{elapsed_ns}}")
            \\print(f"ROWS:{{total_rows}}")
        , .{ BENCHMARK_SECONDS, WARMUP_SECONDS, PARQUET_PATH });

        const py_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", script },
            .max_output_bytes = 10 * 1024 * 1024,
        }) catch {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL", "error", "-", "-" });
            break :duckdb_sql;
        };
        defer {
            allocator.free(py_result.stdout);
            allocator.free(py_result.stderr);
        }

        var iterations: u64 = 0;
        var total_ns: u64 = 0;
        var total_rows: u64 = 0;

        if (std.mem.indexOf(u8, py_result.stdout, "ITERATIONS:")) |idx| {
            const start = idx + 11;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            iterations = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "TOTAL_NS:")) |idx| {
            const start = idx + 9;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            total_ns = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "ROWS:")) |idx| {
            const start = idx + 5;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            total_rows = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }

        if (iterations > 0 and total_ns > 0) {
            const throughput = @as(f64, @floatFromInt(total_rows)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);
            const speedup = lanceql_throughput / throughput;
            var speedup_buf: [16]u8 = undefined;
            const speedup_str = std.fmt.bufPrint(&speedup_buf, "{d:.1}x", .{speedup}) catch "N/A";
            std.debug.print("{s:<35} {d:>12.0} {d:>12} {s:>10}\n", .{
                "DuckDB SQL", throughput, iterations, speedup_str,
            });
        }
    }

    // 3. Polars DataFrame aggregations
    if (has_polars) polars_df: {
        const script = std.fmt.comptimePrint(
            \\import polars as pl
            \\import time
            \\
            \\BENCHMARK_SECONDS = {d}
            \\WARMUP_SECONDS = {d}
            \\PARQUET_PATH = "{s}"
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    df = pl.read_parquet(PARQUET_PATH)
            \\    _ = df.select([
            \\        pl.col("amount").sum().alias("sum"),
            \\        pl.col("amount").mean().alias("avg"),
            \\        pl.col("amount").min().alias("min"),
            \\        pl.col("amount").max().alias("max")
            \\    ])
            \\
            \\# Benchmark
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = pl.read_parquet(PARQUET_PATH)
            \\    result = df.select([
            \\        pl.col("amount").sum().alias("sum"),
            \\        pl.col("amount").mean().alias("avg"),
            \\        pl.col("amount").min().alias("min"),
            \\        pl.col("amount").max().alias("max")
            \\    ])
            \\    iterations += 1
            \\    total_rows += len(df)
            \\elapsed_ns = int((time.time() - start) * 1e9)
            \\
            \\print(f"ITERATIONS:{{iterations}}")
            \\print(f"TOTAL_NS:{{elapsed_ns}}")
            \\print(f"ROWS:{{total_rows}}")
        , .{ BENCHMARK_SECONDS, WARMUP_SECONDS, PARQUET_PATH });

        const py_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", script },
            .max_output_bytes = 10 * 1024 * 1024,
        }) catch {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{ "Polars DataFrame", "error", "-", "-" });
            break :polars_df;
        };
        defer {
            allocator.free(py_result.stdout);
            allocator.free(py_result.stderr);
        }

        var iterations: u64 = 0;
        var total_ns: u64 = 0;
        var total_rows: u64 = 0;

        if (std.mem.indexOf(u8, py_result.stdout, "ITERATIONS:")) |idx| {
            const start = idx + 11;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            iterations = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "TOTAL_NS:")) |idx| {
            const start = idx + 9;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            total_ns = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "ROWS:")) |idx| {
            const start = idx + 5;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            total_rows = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }

        if (iterations > 0 and total_ns > 0) {
            const throughput = @as(f64, @floatFromInt(total_rows)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);
            const speedup = lanceql_throughput / throughput;
            var speedup_buf: [16]u8 = undefined;
            const speedup_str = std.fmt.bufPrint(&speedup_buf, "{d:.1}x", .{speedup}) catch "N/A";
            std.debug.print("{s:<35} {d:>12.0} {d:>12} {s:>10}\n", .{
                "Polars DataFrame", throughput, iterations, speedup_str,
            });
        }
    }

    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Notes:\n", .{});
    std.debug.print("  - Operations: SUM, AVG, MIN, MAX on amount column\n", .{});
    std.debug.print("  - All methods read from disk (Lance or Parquet files)\n", .{});
    std.debug.print("  - All methods run for exactly {d} seconds\n", .{BENCHMARK_SECONDS});
    std.debug.print("  - Throughput = total rows processed / elapsed time\n", .{});
    std.debug.print("\n", .{});
}
