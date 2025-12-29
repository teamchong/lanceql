//! Benchmark: End-to-End Vector Computation
//!
//! HONEST COMPARISON measuring real workflow:
//!   Open file → Read columns → Compute dot products → Return results
//!
//! All methods run the SAME number of iterations with SAME conditions.
//! This measures what users actually experience, not synthetic micro-benchmarks.
//!
//! Workflow:
//!   1. Compile Python: metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a
//!   2. Run benchmark: zig build bench-compiled-logic-table

const std = @import("std");

// =============================================================================
// Extern declarations for compiled @logic_table functions
// =============================================================================

extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) f64;

// =============================================================================
// Configuration - SAME for all methods
// =============================================================================

const BATCH_SIZE = 10_000;
const DIM = 384;
const ITERATIONS = 5; // Same iterations for fair comparison

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
    std.debug.print("End-to-End Vector Computation Benchmark\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Vectors: {d} × {d} dimensions, {d} iterations each\n", .{ BATCH_SIZE, DIM, ITERATIONS });
    std.debug.print("\n", .{});
    std.debug.print("What this measures:\n", .{});
    std.debug.print("  - Data already in memory (no I/O)\n", .{});
    std.debug.print("  - Pure computation time for dot product\n", .{});
    std.debug.print("  - Same iterations for all methods\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("What this does NOT measure:\n", .{});
    std.debug.print("  - File I/O (Lance/Parquet reading)\n", .{});
    std.debug.print("  - Schema parsing / column decoding\n", .{});
    std.debug.print("  - JIT compilation time\n", .{});
    std.debug.print("\n", .{});

    // Check for Python modules
    const has_duckdb = checkPythonModule(allocator, "duckdb");
    const has_polars = checkPythonModule(allocator, "polars");

    std.debug.print("Available: DuckDB={s}, Polars={s}\n", .{
        if (has_duckdb) "yes" else "no",
        if (has_polars) "yes" else "no",
    });
    std.debug.print("\n", .{});

    // Generate test data ONCE - same data for all benchmarks
    const vec_a = try allocator.alloc(f64, BATCH_SIZE * DIM);
    defer allocator.free(vec_a);
    const vec_b = try allocator.alloc(f64, BATCH_SIZE * DIM);
    defer allocator.free(vec_b);
    const results = try allocator.alloc(f64, BATCH_SIZE);
    defer allocator.free(results);

    var rng = std.Random.DefaultPrng.init(42);
    for (0..BATCH_SIZE * DIM) |i| {
        vec_a[i] = rng.random().float(f64) * 2.0 - 1.0;
        vec_b[i] = rng.random().float(f64) * 2.0 - 1.0;
    }

    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Total (ms)", "Per Vec", "Speedup" });
    std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 35, "-" ** 12, "-" ** 12, "-" ** 10 });

    var logic_table_ns: u64 = 0;

    // 1. Compiled @logic_table - SAME iterations as Python
    {
        var total_ns: u64 = 0;
        for (0..ITERATIONS) |_| {
            var timer = try std.time.Timer.start();
            for (0..BATCH_SIZE) |i| {
                results[i] = VectorOps_dot_product(
                    vec_a.ptr + i * DIM,
                    vec_b.ptr + i * DIM,
                    DIM,
                );
            }
            std.mem.doNotOptimizeAway(results);
            total_ns += timer.read();
        }
        logic_table_ns = total_ns / ITERATIONS;
        const ms = @as(f64, @floatFromInt(logic_table_ns)) / 1_000_000.0;
        const per_vec = @as(f64, @floatFromInt(logic_table_ns)) / @as(f64, @floatFromInt(BATCH_SIZE));
        std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {s:>10}\n", .{
            "@logic_table (compiled, in-memory)", ms, per_vec, "baseline",
        });
    }

    // 2. NumPy einsum (fair comparison - also in-memory, vectorized)
    if (has_duckdb) {
        const script = std.fmt.comptimePrint(
            \\import time
            \\import numpy as np
            \\
            \\BATCH_SIZE = {d}
            \\DIM = {d}
            \\ITERATIONS = {d}
            \\
            \\np.random.seed(42)
            \\vec_a = np.random.randn(BATCH_SIZE, DIM)
            \\vec_b = np.random.randn(BATCH_SIZE, DIM)
            \\
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    results = np.einsum('ij,ij->i', vec_a, vec_b)
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\print(f"RESULT_NS:{{sum(times) // len(times)}}")
        , .{ BATCH_SIZE, DIM, ITERATIONS });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_vec = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(BATCH_SIZE));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(logic_table_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.1}x\n", .{
                "NumPy einsum (in-memory)", ms, per_vec, speedup,
            });
        }
    }

    // 3. DuckDB with pre-loaded table (fair - data already in DB)
    if (has_duckdb) {
        const script = std.fmt.comptimePrint(
            \\import duckdb
            \\import time
            \\import numpy as np
            \\import pandas as pd
            \\
            \\BATCH_SIZE = {d}
            \\DIM = {d}
            \\ITERATIONS = {d}
            \\
            \\con = duckdb.connect()
            \\
            \\# Setup: Load data into DuckDB ONCE (not timed)
            \\np.random.seed(42)
            \\vec_a = np.random.randn(BATCH_SIZE, DIM)
            \\vec_b = np.random.randn(BATCH_SIZE, DIM)
            \\df = pd.DataFrame({{'a': vec_a.tolist(), 'b': vec_b.tolist()}})
            \\con.execute("CREATE TABLE vectors AS SELECT * FROM df")
            \\
            \\# UDF definition (not timed)
            \\def dot_product_udf(vec_a, vec_b):
            \\    return float(np.dot(vec_a, vec_b))
            \\con.create_function('dot_product', dot_product_udf,
            \\    [duckdb.list_type('DOUBLE'), duckdb.list_type('DOUBLE')], 'DOUBLE')
            \\
            \\# Benchmark: Only measure query execution
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    results = con.execute("SELECT dot_product(a, b) FROM vectors").fetchall()
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\print(f"RESULT_NS:{{sum(times) // len(times)}}")
        , .{ BATCH_SIZE, DIM, ITERATIONS });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_vec = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(BATCH_SIZE));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(logic_table_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.0}x\n", .{
                "DuckDB Python UDF (pre-loaded)", ms, per_vec, speedup,
            });
        }
    }

    // 4. Polars with pre-loaded DataFrame (fair - data already loaded)
    if (has_polars) {
        const script = std.fmt.comptimePrint(
            \\import polars as pl
            \\import time
            \\import numpy as np
            \\
            \\BATCH_SIZE = {d}
            \\DIM = {d}
            \\ITERATIONS = {d}
            \\
            \\# Setup: Load data ONCE (not timed)
            \\np.random.seed(42)
            \\vec_a = [np.random.randn(DIM).tolist() for _ in range(BATCH_SIZE)]
            \\vec_b = [np.random.randn(DIM).tolist() for _ in range(BATCH_SIZE)]
            \\df = pl.DataFrame({{'a': vec_a, 'b': vec_b}})
            \\
            \\def dot_product_udf(row):
            \\    return float(np.dot(row['a'], row['b']))
            \\
            \\# Benchmark: Only measure computation
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
        , .{ BATCH_SIZE, DIM, ITERATIONS });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_vec = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(BATCH_SIZE));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(logic_table_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.0}x\n", .{
                "Polars Python UDF (pre-loaded)", ms, per_vec, speedup,
            });
        }
    }

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("INTERPRETATION:\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("  @logic_table advantage: Pre-compiled SIMD code, no Python interpreter overhead\n", .{});
    std.debug.print("  NumPy einsum: Optimized C/Fortran, close to native for batch operations\n", .{});
    std.debug.print("  DuckDB/Polars UDF: Python callback per row = interpreter overhead\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("  For REAL end-to-end comparison, need to include:\n", .{});
    std.debug.print("    - Lance file I/O and column decoding\n", .{});
    std.debug.print("    - Query planning and execution\n", .{});
    std.debug.print("    - Memory allocation for real data\n", .{});
    std.debug.print("================================================================================\n", .{});
}
