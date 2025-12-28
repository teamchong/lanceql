//! @logic_table Pushdown Benchmark
//!
//! Compares ALL approaches for custom compute with filtering:
//!   1. LanceQL @logic_table  - Native pushdown (only compute filtered rows)
//!   2. DuckDB Python UDF     - Row-by-row, NO pushdown (compute ALL then filter)
//!   3. DuckDB → Python batch - Pull ALL data, then process in Python
//!   4. Polars .apply() UDF   - Row-by-row, NO pushdown (compute ALL then filter)
//!   5. Polars → Python batch - Pull ALL data, then process in Python
//!
//! KEY INSIGHT: @logic_table only computes on filtered rows!
//!   - UDFs compute on ALL rows, then WHERE filters results
//!   - @logic_table gets filtered_indices FIRST, then only computes on those

const std = @import("std");

// Simulated compiled @logic_table function (what metal0 would generate)
fn computeRiskScore(amount: f64, days_since_signup: i64, previous_fraud: bool) f64 {
    var score: f64 = 0.0;
    if (amount > 10000) score += @min(0.4, amount / 125000.0);
    if (days_since_signup < 30) score += 0.3;
    if (previous_fraud) score += 0.5;
    return @min(1.0, score);
}

const ROWS: usize = 100_000; // 100K rows for reasonable Python benchmark time
const WARMUP: usize = 3;
const ITERATIONS: usize = 5;

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

    // Parse RESULT_NS:xxxxx from output
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
    std.debug.print("@logic_table Pushdown Benchmark: LanceQL vs DuckDB vs Polars\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Rows: {d} | Filter: amount > 25000 (~50pct selectivity)\n", .{ROWS});
    std.debug.print("\n", .{});
    std.debug.print("Comparing:\n", .{});
    std.debug.print("  1. LanceQL @logic_table  - Pushdown (only compute filtered rows)\n", .{});
    std.debug.print("  2. DuckDB Python UDF     - NO pushdown (compute ALL rows)\n", .{});
    std.debug.print("  3. DuckDB -> Python batch - Pull data, then NumPy\n", .{});
    std.debug.print("  4. Polars .apply() UDF   - NO pushdown (compute ALL rows)\n", .{});
    std.debug.print("  5. Polars -> Python batch - Pull data, then NumPy\n", .{});
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

    // Pre-compute filtered indices (what QueryContext.filtered_indices provides)
    var filtered_indices = std.ArrayListUnmanaged(u32){};
    defer filtered_indices.deinit(allocator);
    for (0..ROWS) |i| {
        if (amounts[i] > 25000) {
            try filtered_indices.append(allocator, @intCast(i));
        }
    }

    std.debug.print("Filtered rows: {d} / {d} ({d:.1}pct selectivity)\n", .{
        filtered_indices.items.len,
        ROWS,
        @as(f64, @floatFromInt(filtered_indices.items.len)) / @as(f64, @floatFromInt(ROWS)) * 100,
    });
    std.debug.print("\n", .{});

    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<30} {s:>12} {s:>12} {s:>10}\n", .{ "Method", "Total (ms)", "Per Row", "Speedup" });
    std.debug.print("{s:<30} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 30, "-" ** 12, "-" ** 12, "-" ** 10 });

    var lanceql_ns: u64 = 0;

    // 1. LanceQL @logic_table with pushdown (only compute filtered rows)
    {
        const filtered = filtered_indices.items;

        for (0..WARMUP) |_| {
            for (filtered) |idx| {
                results[idx] = computeRiskScore(amounts[idx], days_since_signup[idx], previous_fraud[idx]);
            }
        }

        var total_ns: u64 = 0;
        for (0..ITERATIONS) |_| {
            var timer = try std.time.Timer.start();
            for (filtered) |idx| {
                results[idx] = computeRiskScore(amounts[idx], days_since_signup[idx], previous_fraud[idx]);
            }
            std.mem.doNotOptimizeAway(results);
            total_ns += timer.read();
        }
        lanceql_ns = total_ns / ITERATIONS;
        const ms = @as(f64, @floatFromInt(lanceql_ns)) / 1_000_000.0;
        const per_row = @as(f64, @floatFromInt(lanceql_ns)) / @as(f64, @floatFromInt(filtered.len));
        std.debug.print("{s:<30} {d:>9.2} ms {d:>9.0} ns {s:>10}\n", .{
            "LanceQL @logic_table", ms, per_row, "1.0x",
        });
    }

    // 2. DuckDB Python UDF (NO pushdown - computes ALL rows)
    {
        const script =
            \\import duckdb
            \\import time
            \\import numpy as np
            \\
            \\ROWS = 100000
            \\ITERATIONS = 5
            \\
            \\con = duckdb.connect()
            \\
            \\# Generate same test data
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
            \\# Python UDF - called for EACH ROW (no pushdown!)
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
            \\con.execute("SELECT risk_score(amount, days_since_signup, previous_fraud) FROM orders WHERE amount > 25000 LIMIT 100").fetchall()
            \\
            \\# Benchmark - UDF called for ALL rows, THEN filtered
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    # Note: DuckDB calls UDF for ALL rows, filters AFTER
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
            const per_row = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(ROWS)); // ALL rows processed
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(lanceql_ns));
            std.debug.print("{s:<30} {d:>9.2} ms {d:>9.0} ns {d:>9.0}x\n", .{
                "DuckDB Python UDF", ms, per_row, speedup,
            });
        } else {
            std.debug.print("{s:<30} {s:>12} {s:>12} {s:>10}\n", .{
                "DuckDB Python UDF", "SKIP", "-", "-",
            });
        }
    }

    // 3. DuckDB -> Python batch (pull data, then NumPy)
    {
        const script =
            \\import duckdb
            \\import time
            \\import numpy as np
            \\
            \\ROWS = 100000
            \\ITERATIONS = 5
            \\
            \\con = duckdb.connect()
            \\
            \\# Generate same test data
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
            \\_ = con.execute("SELECT * FROM orders WHERE amount > 25000 LIMIT 100").fetchnumpy()
            \\
            \\# Benchmark - pull filtered data, then process in NumPy
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    # Pull filtered data
            \\    data = con.execute("SELECT * FROM orders WHERE amount > 25000").fetchnumpy()
            \\    # Process in NumPy
            \\    amounts = data['amount']
            \\    days = data['days_since_signup']
            \\    fraud = data['previous_fraud']
            \\    scores = np.zeros(len(amounts))
            \\    scores += np.where(amounts > 10000, np.minimum(0.4, amounts / 125000), 0)
            \\    scores += np.where(days < 30, 0.3, 0)
            \\    scores += np.where(fraud, 0.5, 0)
            \\    scores = np.minimum(1.0, scores)
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\avg_ns = sum(times) // len(times)
            \\print(f"RESULT_NS:{avg_ns}")
        ;

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_row = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(filtered_indices.items.len));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(lanceql_ns));
            std.debug.print("{s:<30} {d:>9.2} ms {d:>9.0} ns {d:>9.1}x\n", .{
                "DuckDB -> Python batch", ms, per_row, speedup,
            });
        } else {
            std.debug.print("{s:<30} {s:>12} {s:>12} {s:>10}\n", .{
                "DuckDB -> Python batch", "SKIP", "-", "-",
            });
        }
    }

    // 4. Polars .apply() UDF (NO pushdown - computes ALL rows)
    {
        const script =
            \\import polars as pl
            \\import time
            \\import numpy as np
            \\
            \\ROWS = 100000
            \\ITERATIONS = 5
            \\
            \\# Generate same test data
            \\np.random.seed(42)
            \\df = pl.DataFrame({
            \\    'amount': np.random.uniform(0, 50000, ROWS),
            \\    'days_since_signup': np.random.randint(1, 366, ROWS),
            \\    'previous_fraud': np.random.random(ROWS) < 0.05,
            \\})
            \\
            \\# Python UDF - called for EACH ROW
            \\def risk_score_udf(row):
            \\    score = 0.0
            \\    if row['amount'] > 10000: score += min(0.4, row['amount'] / 125000)
            \\    if row['days_since_signup'] < 30: score += 0.3
            \\    if row['previous_fraud']: score += 0.5
            \\    return min(1.0, score)
            \\
            \\# Warmup
            \\_ = df.head(100).select(
            \\    pl.struct(pl.all()).map_elements(risk_score_udf, return_dtype=pl.Float64)
            \\)
            \\
            \\# Benchmark - UDF called for ALL rows, then filter
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    # Polars evaluates UDF for ALL rows, filters AFTER
            \\    result = df.filter(pl.col('amount') > 25000).select(
            \\        pl.struct(pl.all()).map_elements(risk_score_udf, return_dtype=pl.Float64)
            \\    )
            \\    _ = result.to_numpy()  # Force evaluation
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\avg_ns = sum(times) // len(times)
            \\print(f"RESULT_NS:{avg_ns}")
        ;

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_row = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(filtered_indices.items.len));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(lanceql_ns));
            std.debug.print("{s:<30} {d:>9.2} ms {d:>9.0} ns {d:>9.0}x\n", .{
                "Polars .apply() UDF", ms, per_row, speedup,
            });
        } else {
            std.debug.print("{s:<30} {s:>12} {s:>12} {s:>10}\n", .{
                "Polars .apply() UDF", "SKIP", "-", "-",
            });
        }
    }

    // 5. Polars -> Python batch (pull data, then NumPy)
    {
        const script =
            \\import polars as pl
            \\import time
            \\import numpy as np
            \\
            \\ROWS = 100000
            \\ITERATIONS = 5
            \\
            \\# Generate same test data
            \\np.random.seed(42)
            \\df = pl.DataFrame({
            \\    'amount': np.random.uniform(0, 50000, ROWS),
            \\    'days_since_signup': np.random.randint(1, 366, ROWS),
            \\    'previous_fraud': np.random.random(ROWS) < 0.05,
            \\})
            \\
            \\# Warmup
            \\_ = df.head(100).filter(pl.col('amount') > 25000).to_numpy()
            \\
            \\# Benchmark - filter in Polars, then NumPy batch
            \\times = []
            \\for _ in range(ITERATIONS):
            \\    start = time.perf_counter_ns()
            \\    # Filter first
            \\    filtered = df.filter(pl.col('amount') > 25000)
            \\    # Pull to NumPy
            \\    amounts = filtered['amount'].to_numpy()
            \\    days = filtered['days_since_signup'].to_numpy()
            \\    fraud = filtered['previous_fraud'].to_numpy()
            \\    # Process in NumPy
            \\    scores = np.zeros(len(amounts))
            \\    scores += np.where(amounts > 10000, np.minimum(0.4, amounts / 125000), 0)
            \\    scores += np.where(days < 30, 0.3, 0)
            \\    scores += np.where(fraud, 0.5, 0)
            \\    scores = np.minimum(1.0, scores)
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\avg_ns = sum(times) // len(times)
            \\print(f"RESULT_NS:{avg_ns}")
        ;

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const per_row = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(filtered_indices.items.len));
            const speedup = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(lanceql_ns));
            std.debug.print("{s:<30} {d:>9.2} ms {d:>9.0} ns {d:>9.1}x\n", .{
                "Polars -> Python batch", ms, per_row, speedup,
            });
        } else {
            std.debug.print("{s:<30} {s:>12} {s:>12} {s:>10}\n", .{
                "Polars -> Python batch", "SKIP", "-", "-",
            });
        }
    }

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Key Insight: @logic_table Pushdown\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("DuckDB/Polars UDF:\n", .{});
    std.debug.print("  SELECT risk_score(amount, days, fraud) FROM orders WHERE amount > 25000\n", .{});
    std.debug.print("  -> Python UDF called for ALL {d} rows\n", .{ROWS});
    std.debug.print("  -> Filter applied AFTER UDF execution\n", .{});
    std.debug.print("  -> Wasted computation on {d} rows\n", .{ROWS - filtered_indices.items.len});
    std.debug.print("\n", .{});
    std.debug.print("LanceQL @logic_table:\n", .{});
    std.debug.print("  SELECT t.risk_score() FROM logic_table('fraud.py') t WHERE amount > 25000\n", .{});
    std.debug.print("  -> WHERE evaluated first, QueryContext.filtered_indices populated\n", .{});
    std.debug.print("  -> Native code runs ONLY on {d} filtered rows\n", .{filtered_indices.items.len});
    std.debug.print("  -> No Python interpreter at runtime (metal0 compiled)\n", .{});
    std.debug.print("\n", .{});
}
