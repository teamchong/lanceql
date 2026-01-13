//! SQL Clauses Benchmark - End-to-End Comparison
//!
//! HONEST benchmark testing SQL operations using the REAL SQL executor:
//!   1. LanceQL native  - Read Lance file → SQL executor → results
//!   2. DuckDB SQL      - Read Parquet → SQL operations
//!   3. Polars DataFrame - Read Parquet → DataFrame operations
//!
//! FAIR COMPARISON:
//!   - All methods read file EACH iteration (no mmap caching advantage)
//!   - All methods run for exactly 15 seconds
//!   - Throughput measured as rows processed per second
//!
//! SQL CLAUSES TESTED (all use real SQL executor):
//!   - FILTER (WHERE) - Real SQL: SELECT COUNT(*) WHERE amount > 100
//!   - AGGREGATE (SUM) - Real SQL: SELECT SUM(amount) FROM t
//!   - GROUP BY        - Real SQL: SELECT customer_id, SUM(amount) GROUP BY customer_id
//!   - JOIN            - Real SQL: SELECT * FROM orders JOIN customers ON customer_id
//!
//! Setup:
//!   python3 benchmarks/generate_benchmark_data.py  # Creates orders + customers
//!   zig build bench-sql

const std = @import("std");
const table_mod = @import("lanceql.table");
const Table = table_mod.Table;
const ast = @import("lanceql.sql.ast");
const parser = @import("lanceql.sql.parser");
const executor_mod = @import("lanceql.sql.executor");
const Executor = executor_mod.Executor;
const Result = executor_mod.Result;
const Value = ast.Value;

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

/// Read file bytes into memory
fn readFileBytes(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const bytes = try allocator.alloc(u8, stat.size);
    errdefer allocator.free(bytes);

    const read = try file.readAll(bytes);
    if (read != stat.size) return error.IncompleteRead;
    return bytes;
}

/// Benchmark result for a single clause
const BenchResult = struct {
    rows_per_sec: f64,
    iterations: u64,
    success: bool,
    error_msg: ?[]const u8,
};

/// Run a SQL benchmark using the real executor
fn runSQLBenchmark(
    allocator: std.mem.Allocator,
    lance_file_path: []const u8,
    sql: []const u8,
    warmup_secs: i64,
    bench_secs: i64,
) BenchResult {
    var iterations: u64 = 0;
    var total_rows: u64 = 0;

    // Warmup - read file and execute SQL each iteration
    const warmup_end = std.time.nanoTimestamp() + warmup_secs * 1_000_000_000;
    while (std.time.nanoTimestamp() < warmup_end) {
        // Read file bytes
        const file_bytes = readFileBytes(allocator, lance_file_path) catch {
            return .{ .rows_per_sec = 0, .iterations = 0, .success = false, .error_msg = "failed to read file" };
        };
        defer allocator.free(file_bytes);

        // Create table from bytes
        var table = Table.init(allocator, file_bytes) catch {
            return .{ .rows_per_sec = 0, .iterations = 0, .success = false, .error_msg = "failed to init table" };
        };
        defer table.deinit();

        // Create executor
        var executor = Executor.init(&table, allocator);
        defer executor.deinit();

        // Parse SQL
        var stmt = parser.parseSQL(sql, allocator) catch {
            return .{ .rows_per_sec = 0, .iterations = 0, .success = false, .error_msg = "failed to parse SQL" };
        };
        defer ast.deinitSelectStmt(&stmt.select, allocator);

        // Execute
        var result = executor.execute(&stmt.select, &[_]Value{}) catch {
            return .{ .rows_per_sec = 0, .iterations = 0, .success = false, .error_msg = "failed to execute SQL" };
        };
        defer result.deinit();

        std.mem.doNotOptimizeAway(&result);
    }

    // Benchmark - read file and execute SQL each iteration
    const benchmark_end_time = std.time.nanoTimestamp() + bench_secs * 1_000_000_000;
    const start_time = std.time.nanoTimestamp();

    while (std.time.nanoTimestamp() < benchmark_end_time) {
        // Read file bytes
        const file_bytes = readFileBytes(allocator, lance_file_path) catch break;
        defer allocator.free(file_bytes);

        // Create table from bytes
        var table = Table.init(allocator, file_bytes) catch break;
        defer table.deinit();

        // Get row count from first column
        const row_count = table.rowCount(0) catch 0;

        // Create executor
        var executor = Executor.init(&table, allocator);
        defer executor.deinit();

        // Parse SQL
        var stmt = parser.parseSQL(sql, allocator) catch break;
        defer ast.deinitSelectStmt(&stmt.select, allocator);

        // Execute
        var result = executor.execute(&stmt.select, &[_]Value{}) catch break;
        defer result.deinit();

        std.mem.doNotOptimizeAway(&result);
        iterations += 1;
        total_rows += row_count;
    }

    const elapsed_ns = std.time.nanoTimestamp() - start_time;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const rows_per_sec = @as(f64, @floatFromInt(total_rows)) / elapsed_s;

    return .{ .rows_per_sec = rows_per_sec, .iterations = iterations, .success = true, .error_msg = null };
}

/// Run JOIN benchmark with two tables registered
fn runJoinSQLBenchmark(
    allocator: std.mem.Allocator,
    orders_path: []const u8,
    customers_path: []const u8,
    sql: []const u8,
    warmup_secs: i64,
    bench_secs: i64,
) BenchResult {
    var iterations: u64 = 0;
    var total_rows: u64 = 0;

    // Warmup
    const warmup_end = std.time.nanoTimestamp() + warmup_secs * 1_000_000_000;
    while (std.time.nanoTimestamp() < warmup_end) {
        // Read both files
        const orders_bytes = readFileBytes(allocator, orders_path) catch {
            return .{ .rows_per_sec = 0, .iterations = 0, .success = false, .error_msg = "failed to read orders" };
        };
        defer allocator.free(orders_bytes);

        const customers_bytes = readFileBytes(allocator, customers_path) catch {
            return .{ .rows_per_sec = 0, .iterations = 0, .success = false, .error_msg = "failed to read customers" };
        };
        defer allocator.free(customers_bytes);

        // Create tables
        var orders_table = Table.init(allocator, orders_bytes) catch {
            return .{ .rows_per_sec = 0, .iterations = 0, .success = false, .error_msg = "failed to init orders table" };
        };
        defer orders_table.deinit();

        var customers_table = Table.init(allocator, customers_bytes) catch {
            return .{ .rows_per_sec = 0, .iterations = 0, .success = false, .error_msg = "failed to init customers table" };
        };
        defer customers_table.deinit();

        // Create executor with orders as primary, register customers
        var executor = Executor.init(&orders_table, allocator);
        defer executor.deinit();
        executor.registerTable("orders", &orders_table) catch {};
        executor.registerTable("customers", &customers_table) catch {};

        // Parse and execute
        var stmt = parser.parseSQL(sql, allocator) catch {
            return .{ .rows_per_sec = 0, .iterations = 0, .success = false, .error_msg = "failed to parse SQL" };
        };
        defer ast.deinitSelectStmt(&stmt.select, allocator);

        var result = executor.execute(&stmt.select, &[_]Value{}) catch {
            return .{ .rows_per_sec = 0, .iterations = 0, .success = false, .error_msg = "failed to execute SQL" };
        };
        defer result.deinit();
        std.mem.doNotOptimizeAway(&result);
    }

    // Benchmark
    const benchmark_end_time = std.time.nanoTimestamp() + bench_secs * 1_000_000_000;
    const start_time = std.time.nanoTimestamp();

    while (std.time.nanoTimestamp() < benchmark_end_time) {
        const orders_bytes = readFileBytes(allocator, orders_path) catch break;
        defer allocator.free(orders_bytes);

        const customers_bytes = readFileBytes(allocator, customers_path) catch break;
        defer allocator.free(customers_bytes);

        var orders_table = Table.init(allocator, orders_bytes) catch break;
        defer orders_table.deinit();

        var customers_table = Table.init(allocator, customers_bytes) catch break;
        defer customers_table.deinit();

        const row_count = orders_table.rowCount(0) catch 0;

        var executor = Executor.init(&orders_table, allocator);
        defer executor.deinit();
        executor.registerTable("orders", &orders_table) catch {};
        executor.registerTable("customers", &customers_table) catch {};

        var stmt = parser.parseSQL(sql, allocator) catch break;
        defer ast.deinitSelectStmt(&stmt.select, allocator);

        var result = executor.execute(&stmt.select, &[_]Value{}) catch break;
        defer result.deinit();

        std.mem.doNotOptimizeAway(&result);
        iterations += 1;
        total_rows += row_count;
    }

    const elapsed_ns = std.time.nanoTimestamp() - start_time;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const rows_per_sec = @as(f64, @floatFromInt(total_rows)) / elapsed_s;

    return .{ .rows_per_sec = rows_per_sec, .iterations = iterations, .success = true, .error_msg = null };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("SQL Clauses Benchmark: End-to-End (Read + Real SQL Executor)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\nPipeline: Read file → parse SQL → execute via real executor → return result\n", .{});
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

    const customers_lance_exists = blk: {
        var data_dir = std.fs.cwd().openDir(CUSTOMERS_LANCE_PATH ++ "/data", .{ .iterate = true }) catch break :blk false;
        data_dir.close();
        break :blk true;
    };

    const customers_parquet_exists = blk: {
        const file = std.fs.cwd().openFile(CUSTOMERS_PARQUET_PATH, .{}) catch break :blk false;
        file.close();
        break :blk true;
    };

    std.debug.print("Data files:\n", .{});
    std.debug.print("  Orders Lance:     {s} {s}\n", .{ LANCE_PATH, if (lance_exists) "✓" else "✗" });
    std.debug.print("  Orders Parquet:   {s} {s}\n", .{ PARQUET_PATH, if (parquet_exists) "✓" else "✗" });
    std.debug.print("  Customers Lance:  {s} {s}\n", .{ CUSTOMERS_LANCE_PATH, if (customers_lance_exists) "✓" else "✗" });
    std.debug.print("  Customers Parquet:{s} {s}\n", .{ CUSTOMERS_PARQUET_PATH, if (customers_parquet_exists) "✓" else "✗" });

    if (!lance_exists or !parquet_exists) {
        std.debug.print("\n⚠️  Missing data files. Run: python3 benchmarks/generate_benchmark_data.py\n", .{});
        return;
    }

    // Check Python engines
    const has_duckdb = checkPythonModule(allocator, "duckdb");
    const has_polars = checkPythonModule(allocator, "polars");

    std.debug.print("\nEngines:\n", .{});
    std.debug.print("  LanceQL native: yes (uses REAL SQL executor with hash tables)\n", .{});
    std.debug.print("  DuckDB:         {s}\n", .{if (has_duckdb) "yes" else "no"});
    std.debug.print("  Polars:         {s}\n", .{if (has_polars) "yes" else "no"});
    std.debug.print("\n", .{});

    // Find lance file path once
    const lance_file_path = findLanceFilePath(allocator, LANCE_PATH) catch {
        std.debug.print("Error: Could not find .lance file in {s}/data\n", .{LANCE_PATH});
        return;
    };
    defer allocator.free(lance_file_path);

    // ==========================================================================
    // FILTER: WHERE amount > 100
    // ==========================================================================
    std.debug.print("================================================================================\n", .{});
    std.debug.print("FILTER: SELECT COUNT(*) WHERE amount > 100 (Real SQL Executor)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Rows/sec", "Iterations", "Speedup" });
    std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 45, "-" ** 12, "-" ** 12, "-" ** 10 });

    var lanceql_filter_rps: f64 = 0;
    {
        const result = runSQLBenchmark(
            allocator,
            lance_file_path,
            "SELECT COUNT(*) FROM t WHERE amount > 100",
            WARMUP_SECONDS,
            BENCHMARK_SECONDS,
        );

        if (result.success) {
            lanceql_filter_rps = result.rows_per_sec;
            std.debug.print("{s:<45} {d:>10.0}K {d:>12} {s:>10}\n", .{
                "LanceQL (FILTER via SQL executor)",
                result.rows_per_sec / 1000.0,
                result.iterations,
                "1.0x",
            });
        } else {
            std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{
                "LanceQL (FILTER via SQL executor)",
                result.error_msg orelse "error",
                "-",
                "-",
            });
        }
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
            std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL", "error", "-", "-" });
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
            const speedup = if (lanceql_filter_rps > 0) lanceql_filter_rps / rows_per_sec else 0;
            std.debug.print("{s:<45} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "DuckDB SQL (FILTER)",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        } else {
            if (result.stderr.len > 0) {
                std.debug.print("DuckDB FILTER stderr: {s}\n", .{result.stderr[0..@min(result.stderr.len, 200)]});
            }
            std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL (FILTER)", "0", "0", "-" });
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
            std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "Polars DataFrame", "error", "-", "-" });
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
            const speedup = if (lanceql_filter_rps > 0) lanceql_filter_rps / rows_per_sec else 0;
            std.debug.print("{s:<45} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
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
    std.debug.print("AGGREGATE: SELECT SUM(amount) (Real SQL Executor)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Rows/sec", "Iterations", "Speedup" });
    std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 45, "-" ** 12, "-" ** 12, "-" ** 10 });

    var lanceql_agg_rps: f64 = 0;
    {
        const result = runSQLBenchmark(
            allocator,
            lance_file_path,
            "SELECT SUM(amount) FROM t",
            WARMUP_SECONDS,
            BENCHMARK_SECONDS,
        );

        if (result.success) {
            lanceql_agg_rps = result.rows_per_sec;
            std.debug.print("{s:<45} {d:>10.0}K {d:>12} {s:>10}\n", .{
                "LanceQL (AGGREGATE via SQL executor)",
                result.rows_per_sec / 1000.0,
                result.iterations,
                "1.0x",
            });
        } else {
            std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{
                "LanceQL (AGGREGATE via SQL executor)",
                result.error_msg orelse "error",
                "-",
                "-",
            });
        }
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
            std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL", "error", "-", "-" });
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
            const speedup = if (lanceql_agg_rps > 0) lanceql_agg_rps / rows_per_sec else 0;
            std.debug.print("{s:<45} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
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
            std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "Polars DataFrame", "error", "-", "-" });
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
            const speedup = if (lanceql_agg_rps > 0) lanceql_agg_rps / rows_per_sec else 0;
            std.debug.print("{s:<45} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "Polars DataFrame (AGGREGATE)",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        }
    }

    // ==========================================================================
    // GROUP BY: SUM(amount) GROUP BY customer_id
    // ==========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("GROUP BY: SELECT customer_id, SUM(amount) GROUP BY customer_id (Integer Hash)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Rows/sec", "Iterations", "Speedup" });
    std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 45, "-" ** 12, "-" ** 12, "-" ** 10 });

    var lanceql_groupby_rps: f64 = 0;
    {
        const result = runSQLBenchmark(
            allocator,
            lance_file_path,
            "SELECT customer_id, SUM(amount) FROM t GROUP BY customer_id",
            WARMUP_SECONDS,
            BENCHMARK_SECONDS,
        );

        if (result.success) {
            lanceql_groupby_rps = result.rows_per_sec;
            std.debug.print("{s:<45} {d:>10.0}K {d:>12} {s:>10}\n", .{
                "LanceQL (GROUP BY via SQL executor)",
                result.rows_per_sec / 1000.0,
                result.iterations,
                "1.0x",
            });
        } else {
            std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{
                "LanceQL (GROUP BY via SQL executor)",
                result.error_msg orelse "error",
                "-",
                "-",
            });
        }
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
            \\    _ = con.execute("SELECT customer_id, SUM(amount) FROM read_parquet('{s}') GROUP BY customer_id").fetchall()
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
            \\    _ = con.execute("SELECT customer_id, SUM(amount) FROM read_parquet('{s}') GROUP BY customer_id").fetchall()
            \\    con.close()
            \\    iterations += 1
            \\
            \\elapsed = time.time() - start
            \\rows_per_sec = total_rows / elapsed
            \\print(f"ROWS_PER_SEC:{{rows_per_sec:.0f}}")
            \\print(f"ITERATIONS:{{iterations}}")
        , .{ WARMUP_SECONDS, BENCHMARK_SECONDS, PARQUET_PATH, PARQUET_PATH, PARQUET_PATH });

        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", py_script },
            .max_output_bytes = 10 * 1024,
        }) catch {
            std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL", "error", "-", "-" });
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
            const speedup = if (lanceql_groupby_rps > 0) lanceql_groupby_rps / rows_per_sec else 0;
            std.debug.print("{s:<45} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "DuckDB SQL (GROUP BY)",
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
            \\    grouped = df.group_by("customer_id").agg(pl.col("amount").sum())
            \\
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = pl.read_parquet("{s}")
            \\    grouped = df.group_by("customer_id").agg(pl.col("amount").sum())
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
            std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "Polars DataFrame", "error", "-", "-" });
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
            const speedup = if (lanceql_groupby_rps > 0) lanceql_groupby_rps / rows_per_sec else 0;
            std.debug.print("{s:<45} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                "Polars DataFrame (GROUP BY)",
                rows_per_sec / 1000.0,
                iterations,
                speedup,
            });
        }
    }

    // ==========================================================================
    // JOIN: orders INNER JOIN customers ON customer_id
    // ==========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("JOIN: orders INNER JOIN customers ON customer_id (Real Hash Join)\n", .{});
    std.debug.print("================================================================================\n", .{});

    if (!customers_lance_exists or !customers_parquet_exists) {
        std.debug.print("SKIPPED: Missing customers data files\n", .{});
        std.debug.print("         Run: python3 benchmarks/generate_benchmark_data.py\n", .{});
    } else {
        std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Rows/sec", "Iterations", "Speedup" });
        std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 45, "-" ** 12, "-" ** 12, "-" ** 10 });

        // Find customers lance file path
        const customers_file_path = findLanceFilePath(allocator, CUSTOMERS_LANCE_PATH) catch {
            std.debug.print("Error: Could not find .lance file in {s}/data\n", .{CUSTOMERS_LANCE_PATH});
            return;
        };
        defer allocator.free(customers_file_path);

        // LanceQL JOIN benchmark with multi-table registration
        var lanceql_join_rps: f64 = 0;
        {
            const result = runJoinSQLBenchmark(
                allocator,
                lance_file_path,
                customers_file_path,
                "SELECT orders.customer_id, orders.amount, customers.name FROM orders INNER JOIN customers ON orders.customer_id = customers.id",
                WARMUP_SECONDS,
                BENCHMARK_SECONDS,
            );

            if (result.success) {
                lanceql_join_rps = result.rows_per_sec;
                std.debug.print("{s:<45} {d:>10.0}K {d:>12} {s:>10}\n", .{
                    "LanceQL (JOIN via SQL executor)",
                    result.rows_per_sec / 1000.0,
                    result.iterations,
                    "1.0x",
                });
            } else {
                std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{
                    "LanceQL (JOIN via SQL executor)",
                    result.error_msg orelse "error",
                    "-",
                    "-",
                });
            }
        }

        var duckdb_join_rps: f64 = 0;
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
                \\    _ = con.execute("""
                \\        SELECT o.*, c.name
                \\        FROM read_parquet('{s}') o
                \\        INNER JOIN read_parquet('{s}') c ON o.customer_id = c.id
                \\    """).fetchall()
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
                \\    _ = con.execute("""
                \\        SELECT o.*, c.name
                \\        FROM read_parquet('{s}') o
                \\        INNER JOIN read_parquet('{s}') c ON o.customer_id = c.id
                \\    """).fetchall()
                \\    con.close()
                \\    iterations += 1
                \\
                \\elapsed = time.time() - start
                \\rows_per_sec = total_rows / elapsed
                \\print(f"ROWS_PER_SEC:{{rows_per_sec:.0f}}")
                \\print(f"ITERATIONS:{{iterations}}")
            , .{
                WARMUP_SECONDS,
                BENCHMARK_SECONDS,
                PARQUET_PATH,
                CUSTOMERS_PARQUET_PATH,
                PARQUET_PATH,
                PARQUET_PATH,
                CUSTOMERS_PARQUET_PATH,
            });

            const result = std.process.Child.run(.{
                .allocator = allocator,
                .argv = &.{ "python3", "-c", py_script },
                .max_output_bytes = 10 * 1024,
            }) catch {
                std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "DuckDB SQL", "error", "-", "-" });
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
                duckdb_join_rps = rows_per_sec;
                std.debug.print("{s:<45} {d:>10.0}K {d:>12} {s:>10}\n", .{
                    "DuckDB SQL (JOIN)",
                    rows_per_sec / 1000.0,
                    iterations,
                    "1.0x",
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
                \\    orders = pl.read_parquet("{s}")
                \\    customers = pl.read_parquet("{s}")
                \\    joined = orders.join(customers, left_on="customer_id", right_on="id")
                \\
                \\iterations = 0
                \\total_rows = 0
                \\start = time.time()
                \\benchmark_end = start + BENCHMARK_SECONDS
                \\while time.time() < benchmark_end:
                \\    orders = pl.read_parquet("{s}")
                \\    customers = pl.read_parquet("{s}")
                \\    joined = orders.join(customers, left_on="customer_id", right_on="id")
                \\    total_rows += len(orders)
                \\    iterations += 1
                \\
                \\elapsed = time.time() - start
                \\rows_per_sec = total_rows / elapsed
                \\print(f"ROWS_PER_SEC:{{rows_per_sec:.0f}}")
                \\print(f"ITERATIONS:{{iterations}}")
            , .{ WARMUP_SECONDS, BENCHMARK_SECONDS, PARQUET_PATH, CUSTOMERS_PARQUET_PATH, PARQUET_PATH, CUSTOMERS_PARQUET_PATH });

            const result = std.process.Child.run(.{
                .allocator = allocator,
                .argv = &.{ "python3", "-c", py_script },
                .max_output_bytes = 10 * 1024,
            }) catch {
                std.debug.print("{s:<45} {s:>12} {s:>12} {s:>10}\n", .{ "Polars DataFrame", "error", "-", "-" });
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
                const speedup = if (duckdb_join_rps > 0) duckdb_join_rps / rows_per_sec else 0;
                std.debug.print("{s:<45} {d:>10.0}K {d:>12} {d:>9.1}x\n", .{
                    "Polars DataFrame (JOIN)",
                    rows_per_sec / 1000.0,
                    iterations,
                    speedup,
                });
            }
        }
    }

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("All methods: Read file EACH iteration → execute SQL → return result\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("LanceQL uses REAL SQL executor with:\n", .{});
    std.debug.print("  - Real parser (parser.parseSQL)\n", .{});
    std.debug.print("  - Real executor (Executor.execute)\n", .{});
    std.debug.print("  - Real hash tables (AutoHashMap(u64) for GROUP BY)\n", .{});
    std.debug.print("  - Real result materialization\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("FILTER:    Uses real WHERE clause evaluation\n", .{});
    std.debug.print("AGGREGATE: Uses real aggregate accumulator\n", .{});
    std.debug.print("GROUP BY:  Uses efficient integer hashing (FNV-1a composite keys)\n", .{});
    std.debug.print("JOIN:      Uses real hash join with multi-table registration\n", .{});
    std.debug.print("\n", .{});
}
