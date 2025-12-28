//! Benchmark: Compiled @logic_table vs DuckDB vs Polars
//!
//! This benchmark compares:
//!   1. Compiled @logic_table (from metal0, lib/vector_ops.a)
//!   2. DuckDB Python UDF
//!   3. DuckDB + NumPy batch
//!   4. Polars Python UDF
//!   5. Polars + NumPy batch
//!
//! Workflow:
//!   1. Compile Python: metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a
//!   2. Run benchmark: zig build bench-compiled-logic-table

const std = @import("std");

// =============================================================================
// Extern declarations for compiled @logic_table functions
// These are exported by lib/vector_ops.a (compiled from benchmarks/vector_ops.py)
// =============================================================================

extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) f64;

// =============================================================================
// Benchmark Configuration
// =============================================================================

const WARMUP = 3;
const BATCH_SIZE = 10_000;
const ZIG_ITERATIONS = 1_000; // Reduced for faster benchmarks
const PYTHON_ITERATIONS = 5;

fn runPythonBenchmark(allocator: std.mem.Allocator, script: []const u8) !u64 {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", "-c", script },
        .max_output_bytes = 10 * 1024 * 1024,
    }) catch return 0;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }

    if (std.mem.indexOf(u8, result.stdout, "RESULT_NS:")) |idx| {
        const start = idx + 10;
        var end = start;
        while (end < result.stdout.len and result.stdout[end] >= '0' and result.stdout[end] <= '9') {
            end += 1;
        }
        return std.fmt.parseInt(u64, result.stdout[start..end], 10) catch 0;
    }
    return 0;
}

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

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Compiled @logic_table vs DuckDB vs Polars\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Vectors: {d} × 384 dimensions\n", .{BATCH_SIZE});
    std.debug.print("\n", .{});

    // Check for Python modules
    const has_duckdb = checkPythonModule(allocator, "duckdb");
    const has_polars = checkPythonModule(allocator, "polars");

    std.debug.print("Available:\n", .{});
    std.debug.print("  - @logic_table: yes (lib/vector_ops.a)\n", .{});
    std.debug.print("  - DuckDB:       {s}\n", .{if (has_duckdb) "yes" else "no"});
    std.debug.print("  - Polars:       {s}\n", .{if (has_polars) "yes" else "no"});
    std.debug.print("\n", .{});

    // Generate test data
    const vec_a = try allocator.alloc(f64, BATCH_SIZE * 384);
    defer allocator.free(vec_a);
    const vec_b = try allocator.alloc(f64, BATCH_SIZE * 384);
    defer allocator.free(vec_b);
    const results = try allocator.alloc(f64, BATCH_SIZE);
    defer allocator.free(results);

    var rng = std.Random.DefaultPrng.init(42);
    for (0..BATCH_SIZE * 384) |i| {
        vec_a[i] = rng.random().float(f64) * 2.0 - 1.0;
        vec_b[i] = rng.random().float(f64) * 2.0 - 1.0;
    }

    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Total (ms)", "Per Vec", "Speedup" });
    std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 35, "-" ** 12, "-" ** 12, "-" ** 10 });

    var logic_table_ns: u64 = 0;

    // 1. Compiled @logic_table (from lib/vector_ops.a)
    {
        // Warmup
        for (0..WARMUP) |_| {
            for (0..BATCH_SIZE) |i| {
                results[i] = VectorOps_dot_product(
                    vec_a.ptr + i * 384,
                    vec_b.ptr + i * 384,
                    384,
                );
            }
        }

        // Benchmark
        var total_ns: u64 = 0;
        for (0..ZIG_ITERATIONS) |_| {
            var timer = try std.time.Timer.start();
            for (0..BATCH_SIZE) |i| {
                results[i] = VectorOps_dot_product(
                    vec_a.ptr + i * 384,
                    vec_b.ptr + i * 384,
                    384,
                );
            }
            std.mem.doNotOptimizeAway(results);
            total_ns += timer.read();
        }
        logic_table_ns = total_ns / ZIG_ITERATIONS;
        const ms = @as(f64, @floatFromInt(logic_table_ns)) / 1_000_000.0;
        const per_vec = @as(f64, @floatFromInt(logic_table_ns)) / @as(f64, @floatFromInt(BATCH_SIZE));
        std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {s:>10}\n", .{
            "@logic_table (compiled)", ms, per_vec, "1.0x",
        });
    }

    // 2. DuckDB Python UDF
    if (has_duckdb) {
        const script = std.fmt.comptimePrint(
            \\import duckdb
            \\import time
            \\import numpy as np
            \\import pandas as pd
            \\
            \\BATCH_SIZE = {d}
            \\ITERATIONS = {d}
            \\
            \\con = duckdb.connect()
            \\
            \\def dot_product_udf(vec_a, vec_b):
            \\    return float(np.dot(vec_a, vec_b))
            \\
            \\con.create_function('dot_product', dot_product_udf,
            \\    [duckdb.list_type('DOUBLE'), duckdb.list_type('DOUBLE')], 'DOUBLE')
            \\
            \\np.random.seed(42)
            \\vec_a = np.random.randn(BATCH_SIZE, 384)
            \\vec_b = np.random.randn(BATCH_SIZE, 384)
            \\
            \\# Batch insert using DataFrame (much faster than row-by-row)
            \\df = pd.DataFrame({{'a': vec_a.tolist(), 'b': vec_b.tolist()}})
            \\con.execute("CREATE TABLE vectors AS SELECT * FROM df")
            \\
            \\# Warmup
            \\con.execute("SELECT dot_product(a, b) FROM vectors LIMIT 10").fetchall()
            \\
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    results = con.execute("SELECT dot_product(a, b) FROM vectors").fetchall()
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\print(f"RESULT_NS:{{sum(times) // len(times)}}")
        , .{ BATCH_SIZE, PYTHON_ITERATIONS });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_vec = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(BATCH_SIZE));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(logic_table_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.0}x\n", .{
                "DuckDB Python UDF", ms, per_vec, speedup,
            });
        } else {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{
                "DuckDB Python UDF", "FAILED", "-", "-",
            });
        }
    }

    // 3. DuckDB + NumPy batch
    if (has_duckdb) {
        const script = std.fmt.comptimePrint(
            \\import duckdb
            \\import time
            \\import numpy as np
            \\import pandas as pd
            \\
            \\BATCH_SIZE = {d}
            \\ITERATIONS = {d}
            \\
            \\con = duckdb.connect()
            \\
            \\np.random.seed(42)
            \\vec_a = np.random.randn(BATCH_SIZE, 384)
            \\vec_b = np.random.randn(BATCH_SIZE, 384)
            \\
            \\# Batch insert using DataFrame (much faster than row-by-row)
            \\df = pd.DataFrame({{'a': vec_a.tolist(), 'b': vec_b.tolist()}})
            \\con.execute("CREATE TABLE vectors AS SELECT * FROM df")
            \\
            \\# Warmup
            \\_ = con.execute("SELECT * FROM vectors LIMIT 10").fetchnumpy()
            \\
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    data = con.execute("SELECT * FROM vectors").fetchnumpy()
            \\    a_arr = np.array([np.array(x) for x in data['a']])
            \\    b_arr = np.array([np.array(x) for x in data['b']])
            \\    results = np.einsum('ij,ij->i', a_arr, b_arr)
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\print(f"RESULT_NS:{{sum(times) // len(times)}}")
        , .{ BATCH_SIZE, PYTHON_ITERATIONS });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_vec = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(BATCH_SIZE));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(logic_table_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.1}x\n", .{
                "DuckDB + NumPy batch", ms, per_vec, speedup,
            });
        } else {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{
                "DuckDB + NumPy batch", "FAILED", "-", "-",
            });
        }
    }

    // 4. Polars Python UDF
    if (has_polars) {
        const script = std.fmt.comptimePrint(
            \\import polars as pl
            \\import time
            \\import numpy as np
            \\
            \\BATCH_SIZE = {d}
            \\ITERATIONS = {d}
            \\
            \\np.random.seed(42)
            \\vec_a = [np.random.randn(384).tolist() for _ in range(BATCH_SIZE)]
            \\vec_b = [np.random.randn(384).tolist() for _ in range(BATCH_SIZE)]
            \\
            \\df = pl.DataFrame({{'a': vec_a, 'b': vec_b}})
            \\
            \\def dot_product_udf(row):
            \\    return float(np.dot(row['a'], row['b']))
            \\
            \\# Warmup
            \\_ = df.head(10).select(
            \\    pl.struct(pl.all()).map_elements(dot_product_udf, return_dtype=pl.Float64)
            \\)
            \\
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    result = df.select(
            \\        pl.struct(pl.all()).map_elements(dot_product_udf, return_dtype=pl.Float64)
            \\    )
            \\    _ = result.to_numpy()
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\print(f"RESULT_NS:{{sum(times) // len(times)}}")
        , .{ BATCH_SIZE, PYTHON_ITERATIONS });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_vec = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(BATCH_SIZE));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(logic_table_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.0}x\n", .{
                "Polars Python UDF", ms, per_vec, speedup,
            });
        } else {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{
                "Polars Python UDF", "FAILED", "-", "-",
            });
        }
    }

    // 5. Polars + NumPy batch
    if (has_polars) {
        const script = std.fmt.comptimePrint(
            \\import polars as pl
            \\import time
            \\import numpy as np
            \\
            \\BATCH_SIZE = {d}
            \\ITERATIONS = {d}
            \\
            \\np.random.seed(42)
            \\vec_a = np.random.randn(BATCH_SIZE, 384)
            \\vec_b = np.random.randn(BATCH_SIZE, 384)
            \\
            \\df = pl.DataFrame({{'a': vec_a.tolist(), 'b': vec_b.tolist()}})
            \\
            \\# Warmup
            \\_ = df.head(10).to_numpy()
            \\
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    a_arr = np.array([np.array(x) for x in df['a'].to_list()])
            \\    b_arr = np.array([np.array(x) for x in df['b'].to_list()])
            \\    results = np.einsum('ij,ij->i', a_arr, b_arr)
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\print(f"RESULT_NS:{{sum(times) // len(times)}}")
        , .{ BATCH_SIZE, PYTHON_ITERATIONS });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_vec = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(BATCH_SIZE));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(logic_table_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.1}x\n", .{
                "Polars + NumPy batch", ms, per_vec, speedup,
            });
        } else {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{
                "Polars + NumPy batch", "FAILED", "-", "-",
            });
        }
    }

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("@logic_table = Python compiled to native code via metal0\n", .{});
    std.debug.print("================================================================================\n", .{});
}
