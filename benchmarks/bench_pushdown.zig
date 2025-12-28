//! @logic_table Benchmark - HONEST Comparison
//!
//! What we're actually comparing:
//!   1. LanceQL native Zig  - Hand-written Zig code (filter + compute)
//!   2. DuckDB Python UDF   - Row-by-row Python calls through DuckDB
//!   3. DuckDB + NumPy      - Filter in DuckDB, batch compute in NumPy
//!   4. Polars Python UDF   - Row-by-row Python calls (filter DOES pushdown)
//!   5. Polars + NumPy      - Filter in Polars, batch compute in NumPy
//!
//! HONEST NOTES:
//!   - LanceQL is native Zig, NOT JIT-compiled Python (JIT is WIP)
//!   - Polars .filter().apply() DOES pushdown - UDF runs on filtered rows only
//!   - The comparison is: native Zig vs Python, NOT @logic_table vs UDF
//!
//! This benchmark shows the POTENTIAL of @logic_table when JIT is complete.
//! Current implementation uses hand-written Zig as a proxy.

const std = @import("std");

// Hand-written native function (NOT JIT compiled from Python yet)
fn computeRiskScore(amount: f64, days_since_signup: i64, previous_fraud: bool) f64 {
    var score: f64 = 0.0;
    if (amount > 10000) score += @min(0.4, amount / 125000.0);
    if (days_since_signup < 30) score += 0.3;
    if (previous_fraud) score += 0.5;
    return @min(1.0, score);
}

const ROWS: usize = 1_000_000;
const WARMUP: usize = 2;
const ZIG_ITERATIONS: usize = 100; // Fewer iterations, include filter time
const PYTHON_ITERATIONS: usize = 3;

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

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("HONEST Benchmark: Native Zig vs Python (filter + compute)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("NOTE: LanceQL uses hand-written Zig, NOT JIT-compiled Python.\n", .{});
    std.debug.print("      This shows POTENTIAL speedup when JIT is complete.\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Rows: {d} | All timings include filter + compute\n", .{ROWS});
    std.debug.print("\n", .{});

    // Generate test data
    const amounts = try allocator.alloc(f64, ROWS);
    defer allocator.free(amounts);
    const days_since_signup = try allocator.alloc(i64, ROWS);
    defer allocator.free(days_since_signup);
    const previous_fraud = try allocator.alloc(bool, ROWS);
    defer allocator.free(previous_fraud);
    const results = try allocator.alloc(f64, ROWS);
    defer allocator.free(results);

    var rng = std.Random.DefaultPrng.init(42);
    for (0..ROWS) |i| {
        amounts[i] = rng.random().float(f64) * 50000;
        days_since_signup[i] = rng.random().intRangeAtMost(i64, 1, 365);
        previous_fraud[i] = rng.random().float(f64) < 0.05;
    }

    // Count filtered rows for reference
    var filtered_count: usize = 0;
    for (0..ROWS) |i| {
        if (amounts[i] > 25000) filtered_count += 1;
    }

    std.debug.print("Filtered rows: {d} / {d} ({d:.1}%% selectivity)\n", .{
        filtered_count, ROWS,
        @as(f64, @floatFromInt(filtered_count)) / @as(f64, @floatFromInt(ROWS)) * 100,
    });
    std.debug.print("\n", .{});

    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Total (ms)", "Per Row", "Speedup" });
    std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 35, "-" ** 12, "-" ** 12, "-" ** 10 });

    var zig_ns: u64 = 0;

    // 1. Native Zig (filter + compute IN SAME TIMING)
    {
        // Warmup
        for (0..WARMUP) |_| {
            for (0..ROWS) |i| {
                if (amounts[i] > 25000) {
                    results[i] = computeRiskScore(amounts[i], days_since_signup[i], previous_fraud[i]);
                }
            }
        }

        // Benchmark - INCLUDES filter time
        var total_ns: u64 = 0;
        for (0..ZIG_ITERATIONS) |_| {
            var timer = try std.time.Timer.start();
            // Filter AND compute in one pass
            for (0..ROWS) |i| {
                if (amounts[i] > 25000) {
                    results[i] = computeRiskScore(amounts[i], days_since_signup[i], previous_fraud[i]);
                }
            }
            std.mem.doNotOptimizeAway(results);
            total_ns += timer.read();
        }
        zig_ns = total_ns / ZIG_ITERATIONS;
        const ms = @as(f64, @floatFromInt(zig_ns)) / 1_000_000.0;
        const per_row = @as(f64, @floatFromInt(zig_ns)) / @as(f64, @floatFromInt(filtered_count));
        std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {s:>10}\n", .{
            "Native Zig (filter+compute)", ms, per_row, "1.0x",
        });
    }

    // 2. DuckDB Python UDF (filter pushdown, Python UDF per row)
    {
        const script =
            \\import duckdb
            \\import time
            \\import numpy as np
            \\
            \\ROWS = 1000000
            \\ITERATIONS = 3
            \\
            \\con = duckdb.connect()
            \\
            \\np.random.seed(42)
            \\amounts = np.random.uniform(0, 50000, ROWS)
            \\days = np.random.randint(1, 366, ROWS)
            \\fraud = np.random.random(ROWS) < 0.05
            \\
            \\con.execute("""
            \\    CREATE TABLE orders AS
            \\    SELECT * FROM (
            \\        SELECT unnest($1) as amount,
            \\               unnest($2) as days_since_signup,
            \\               unnest($3) as previous_fraud
            \\    )
            \\""", [amounts.tolist(), days.tolist(), fraud.tolist()])
            \\
            \\def risk_score_udf(amount, days, fraud):
            \\    score = 0.0
            \\    if amount > 10000: score += min(0.4, amount / 125000)
            \\    if days < 30: score += 0.3
            \\    if fraud: score += 0.5
            \\    return min(1.0, score)
            \\
            \\con.create_function('risk_score', risk_score_udf,
            \\    parameters=['DOUBLE', 'BIGINT', 'BOOLEAN'], return_type='DOUBLE')
            \\
            \\# Warmup
            \\con.execute("SELECT risk_score(amount, days_since_signup, previous_fraud) FROM orders WHERE amount > 25000 LIMIT 1000").fetchall()
            \\
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    # DuckDB filters first, then calls UDF on filtered rows
            \\    results = con.execute("""
            \\        SELECT risk_score(amount, days_since_signup, previous_fraud)
            \\        FROM orders
            \\        WHERE amount > 25000
            \\    """).fetchall()
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\avg_ns = sum(times) // len(times)
            \\print(f"RESULT_NS:{avg_ns}")
        ;

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_row = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(filtered_count));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(zig_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.0}x\n", .{
                "DuckDB + Python UDF", ms, per_row, speedup,
            });
        } else {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{
                "DuckDB + Python UDF", "SKIP", "-", "-",
            });
        }
    }

    // 3. DuckDB -> NumPy batch
    {
        const script =
            \\import duckdb
            \\import time
            \\import numpy as np
            \\
            \\ROWS = 1000000
            \\ITERATIONS = 10
            \\
            \\con = duckdb.connect()
            \\
            \\np.random.seed(42)
            \\amounts = np.random.uniform(0, 50000, ROWS)
            \\days = np.random.randint(1, 366, ROWS)
            \\fraud = np.random.random(ROWS) < 0.05
            \\
            \\con.execute("""
            \\    CREATE TABLE orders AS
            \\    SELECT * FROM (
            \\        SELECT unnest($1) as amount,
            \\               unnest($2) as days_since_signup,
            \\               unnest($3) as previous_fraud
            \\    )
            \\""", [amounts.tolist(), days.tolist(), fraud.tolist()])
            \\
            \\# Warmup
            \\_ = con.execute("SELECT * FROM orders WHERE amount > 25000 LIMIT 1000").fetchnumpy()
            \\
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    data = con.execute("SELECT * FROM orders WHERE amount > 25000").fetchnumpy()
            \\    amounts_f = data['amount']
            \\    days_f = data['days_since_signup']
            \\    fraud_f = data['previous_fraud']
            \\    scores = np.zeros(len(amounts_f))
            \\    scores += np.where(amounts_f > 10000, np.minimum(0.4, amounts_f / 125000), 0)
            \\    scores += np.where(days_f < 30, 0.3, 0)
            \\    scores += np.where(fraud_f, 0.5, 0)
            \\    scores = np.minimum(1.0, scores)
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\avg_ns = sum(times) // len(times)
            \\print(f"RESULT_NS:{avg_ns}")
        ;

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_row = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(filtered_count));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(zig_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.1}x\n", .{
                "DuckDB + NumPy batch", ms, per_row, speedup,
            });
        } else {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{
                "DuckDB + NumPy batch", "SKIP", "-", "-",
            });
        }
    }

    // 4. Polars Python UDF (filter DOES pushdown, then UDF on filtered rows)
    {
        const script =
            \\import polars as pl
            \\import time
            \\import numpy as np
            \\
            \\ROWS = 1000000
            \\ITERATIONS = 3
            \\
            \\np.random.seed(42)
            \\df = pl.DataFrame({
            \\    'amount': np.random.uniform(0, 50000, ROWS),
            \\    'days_since_signup': np.random.randint(1, 366, ROWS),
            \\    'previous_fraud': np.random.random(ROWS) < 0.05,
            \\})
            \\
            \\def risk_score_udf(row):
            \\    score = 0.0
            \\    if row['amount'] > 10000: score += min(0.4, row['amount'] / 125000)
            \\    if row['days_since_signup'] < 30: score += 0.3
            \\    if row['previous_fraud']: score += 0.5
            \\    return min(1.0, score)
            \\
            \\# Warmup
            \\_ = df.head(1000).select(
            \\    pl.struct(pl.all()).map_elements(risk_score_udf, return_dtype=pl.Float64)
            \\)
            \\
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    # Polars filters FIRST, then UDF runs on filtered rows only
            \\    result = df.filter(pl.col('amount') > 25000).select(
            \\        pl.struct(pl.all()).map_elements(risk_score_udf, return_dtype=pl.Float64)
            \\    )
            \\    _ = result.to_numpy()
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\avg_ns = sum(times) // len(times)
            \\print(f"RESULT_NS:{avg_ns}")
        ;

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_row = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(filtered_count));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(zig_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.0}x\n", .{
                "Polars + Python UDF (pushdown)", ms, per_row, speedup,
            });
        } else {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{
                "Polars + Python UDF (pushdown)", "SKIP", "-", "-",
            });
        }
    }

    // 5. Polars -> NumPy batch
    {
        const script =
            \\import polars as pl
            \\import time
            \\import numpy as np
            \\
            \\ROWS = 1000000
            \\ITERATIONS = 10
            \\
            \\np.random.seed(42)
            \\df = pl.DataFrame({
            \\    'amount': np.random.uniform(0, 50000, ROWS),
            \\    'days_since_signup': np.random.randint(1, 366, ROWS),
            \\    'previous_fraud': np.random.random(ROWS) < 0.05,
            \\})
            \\
            \\# Warmup
            \\_ = df.filter(pl.col('amount') > 25000).head(1000).to_numpy()
            \\
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    filtered = df.filter(pl.col('amount') > 25000)
            \\    amounts_f = filtered['amount'].to_numpy()
            \\    days_f = filtered['days_since_signup'].to_numpy()
            \\    fraud_f = filtered['previous_fraud'].to_numpy()
            \\    scores = np.zeros(len(amounts_f))
            \\    scores += np.where(amounts_f > 10000, np.minimum(0.4, amounts_f / 125000), 0)
            \\    scores += np.where(days_f < 30, 0.3, 0)
            \\    scores += np.where(fraud_f, 0.5, 0)
            \\    scores = np.minimum(1.0, scores)
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\avg_ns = sum(times) // len(times)
            \\print(f"RESULT_NS:{avg_ns}")
        ;

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_row = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(filtered_count));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(zig_ns));
            std.debug.print("{s:<35} {d:>9.2} ms {d:>9.0} ns {d:>9.1}x\n", .{
                "Polars + NumPy batch", ms, per_row, speedup,
            });
        } else {
            std.debug.print("{s:<35} {s:>12} {s:>12} {s:>10}\n", .{
                "Polars + NumPy batch", "SKIP", "-", "-",
            });
        }
    }

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("What This Benchmark Shows\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("This is a comparison of NATIVE ZIG vs PYTHON, not @logic_table vs UDF.\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Native Zig advantages:\n", .{});
    std.debug.print("  - No Python interpreter overhead\n", .{});
    std.debug.print("  - Compiler optimizations (SIMD, inlining)\n", .{});
    std.debug.print("  - No data marshaling between languages\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("@logic_table STATUS: JIT compilation implemented but not used here.\n", .{});
    std.debug.print("  - compileWithPredicate() generates fused Zig code\n", .{});
    std.debug.print("  - jitCompileSource() compiles to .dylib\n", .{});
    std.debug.print("  - TODO: Wire this into the benchmark\n", .{});
    std.debug.print("\n", .{});
}
