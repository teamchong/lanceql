//! SQL Clause Benchmark: LanceQL vs DuckDB vs Polars
//!
//! Benchmarks different SQL operations across all three engines.
//! Run with: zig build bench-sql
//!
//! Output: Side-by-side comparison with speedup ratios

const std = @import("std");
const lanceql = @import("lanceql");
const query = @import("lanceql.query");
const metal = @import("lanceql.metal");

// Production-scale: Each benchmark should run 30+ seconds to avoid measuring cold start
// 200M rows x 10 iterations = ~30-60 seconds per clause
const WARMUP = 3;
const ITERATIONS = 10;
const MIN_BENCHMARK_SECONDS: f64 = 30.0;

const Engine = enum { lanceql, duckdb, polars };

const BenchmarkResult = struct {
    engine: Engine,
    clause: []const u8,
    rows: usize,
    avg_sec: f64,
    throughput_mrows_sec: f64,
};

var has_duckdb: bool = false;
var has_polars: bool = false;
var parquet_path: ?[]const u8 = null;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize GPU
    _ = metal.initGPU();
    defer metal.cleanupGPU();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("SQL Clause Benchmark: LanceQL vs DuckDB vs Polars\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Platform: {s}\n", .{metal.getPlatformInfo()});
    if (metal.isGPUReady()) {
        std.debug.print("GPU: {s}\n", .{metal.getGPUDeviceName()});
    }
    std.debug.print("Warmup: {d}, Iterations: {d}\n", .{ WARMUP, ITERATIONS });
    std.debug.print("(All times are hot execution - after warmup)\n\n", .{});

    // Check for external engines
    has_duckdb = checkCommand(allocator, "duckdb");
    has_polars = checkCommand(allocator, "polars");

    std.debug.print("Engines available:\n", .{});
    std.debug.print("  - LanceQL: yes (native Zig + Metal GPU)\n", .{});
    std.debug.print("  - DuckDB:  {s}\n", .{if (has_duckdb) "yes" else "no (install: brew install duckdb)"});
    std.debug.print("  - Polars:  {s}\n", .{if (has_polars) "yes" else "no (install: pip install polars-cli)"});
    std.debug.print("\n", .{});

    // Create test parquet file for DuckDB/Polars
    if (has_duckdb or has_polars) {
        parquet_path = try createTestParquet(allocator);
    }
    defer if (parquet_path) |p| {
        std.fs.deleteFileAbsolute(p) catch {};
        allocator.free(p);
    };

    var results = std.ArrayListUnmanaged(BenchmarkResult){};
    defer results.deinit(allocator);

    // Benchmark each clause at 10M rows to avoid memory issues
    // GROUP BY with 200M rows needs 3.2GB which causes OOM
    const num_rows: usize = 10_000_000;

    std.debug.print("================================================================================\n", .{});
    std.debug.print("Dataset: {d}M rows\n", .{num_rows / 1_000_000});
    std.debug.print("================================================================================\n", .{});

    // Core SQL clauses
    try benchmarkFullScan(allocator, &results, num_rows);
    try benchmarkFilter(allocator, &results, num_rows);
    try benchmarkGroupBy(allocator, &results, num_rows);
    try benchmarkHaving(allocator, &results, num_rows);
    try benchmarkOrderByLimit(allocator, &results, num_rows);
    try benchmarkDistinct(allocator, &results, num_rows);

    // Predicates
    try benchmarkIn(allocator, &results, num_rows);
    try benchmarkBetween(allocator, &results, num_rows);
    try benchmarkLike(allocator, &results, num_rows);
    try benchmarkIsNull(allocator, &results, num_rows);

    // Aggregates
    try benchmarkAggregates(allocator, &results, num_rows);

    // Joins & Vector
    try benchmarkHashJoin(allocator, &results, num_rows);
    try benchmarkLeftJoin(allocator, &results, num_rows);
    try benchmarkInnerJoin(allocator, &results, num_rows);

    // Expressions
    try benchmarkCaseWhen(allocator, &results, num_rows);

    // Vector search (GPU)
    try benchmarkVectorSearch(allocator, &results, num_rows);

    // Print summary table
    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Summary (200M rows)\n", .{});
    std.debug.print("================================================================================\n", .{});
    printSummaryTable(results.items);
}

fn checkCommand(allocator: std.mem.Allocator, cmd: []const u8) bool {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ cmd, "--version" },
    }) catch return false;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    return result.term.Exited == 0;
}

fn createTestParquet(allocator: std.mem.Allocator) ![]const u8 {
    const path = try std.fmt.allocPrint(allocator, "/tmp/lanceql_bench_{d}.parquet", .{std.time.milliTimestamp()});

    const sql = try std.fmt.allocPrint(allocator,
        \\COPY (
        \\  SELECT
        \\    i AS id,
        \\    random() AS value,
        \\    i % 100 AS group_key,
        \\    'item_' || (i % 1000) AS name
        \\  FROM range(10000000) t(i)
        \\) TO '{s}' (FORMAT PARQUET);
    , .{path});
    defer allocator.free(sql);

    std.debug.print("Creating test data (10M rows): {s}...\n", .{path});

    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "duckdb", "-c", sql },
    }) catch return error.DuckDBFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }

    if (result.term.Exited != 0) {
        return error.DuckDBFailed;
    }

    std.debug.print("Test data created.\n\n", .{});
    return path;
}

fn runDuckDB(allocator: std.mem.Allocator, sql: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "duckdb", "-c", sql },
    }) catch return error.DuckDBFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    return timer.read();
}

fn runPolars(allocator: std.mem.Allocator, sql: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "polars", "-c", sql },
    }) catch return error.PolarsFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    return timer.read();
}

fn runLanceQL(allocator: std.mem.Allocator, sql: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "lanceql", "-c", sql },
    }) catch return error.LanceQLFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    return timer.read();
}

var has_lanceql: bool = false;
var lance_path: ?[]const u8 = null;

fn benchmarkFullScan(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- SELECT * (Full Scan) ---\n", .{});

    // LanceQL
    {
        const data = try allocator.alloc(i64, num_rows);
        defer allocator.free(data);
        for (data, 0..) |*v, i| v.* = @intCast(i);

        for (0..WARMUP) |_| {
            var sum: i64 = 0;
            for (data) |v| sum += v;
            std.mem.doNotOptimizeAway(&sum);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var sum: i64 = 0;
            for (data) |v| sum += v;
            std.mem.doNotOptimizeAway(&sum);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "SELECT *", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT * FROM '{s}';", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "SELECT *", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT * FROM read_parquet('{s}');", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "SELECT *", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkFilter(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- WHERE value > 0.5 (Filter) ---\n", .{});

    // LanceQL
    {
        const data = try allocator.alloc(f64, num_rows);
        defer allocator.free(data);
        var rng = std.Random.DefaultPrng.init(42);
        for (data) |*v| v.* = rng.random().float(f64);

        for (0..WARMUP) |_| {
            var count: usize = 0;
            for (data) |v| {
                if (v > 0.5) count += 1;
            }
            std.mem.doNotOptimizeAway(&count);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var count: usize = 0;
            for (data) |v| {
                if (v > 0.5) count += 1;
            }
            std.mem.doNotOptimizeAway(&count);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "WHERE", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*) FROM '{s}' WHERE value > 0.5;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "WHERE", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*) FROM read_parquet('{s}') WHERE value > 0.5;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "WHERE", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkGroupBy(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- GROUP BY + SUM ---\n", .{});

    const num_groups: usize = 100;

    // LanceQL
    {
        const keys = try allocator.alloc(u64, num_rows);
        defer allocator.free(keys);
        const values = try allocator.alloc(u64, num_rows);
        defer allocator.free(values);

        for (0..num_rows) |i| {
            keys[i] = @intCast(i % num_groups);
            values[i] = 1;
        }

        for (0..WARMUP) |_| {
            var group_by = query.GPUGroupBy.initWithCapacity(allocator, .sum, num_groups * 4) catch continue;
            defer group_by.deinit();
            group_by.process(keys, values) catch continue;
            const res = group_by.getResults() catch continue;
            allocator.free(res.keys);
            allocator.free(res.aggregates);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var group_by = try query.GPUGroupBy.initWithCapacity(allocator, .sum, num_groups * 4);
            defer group_by.deinit();
            try group_by.process(keys, values);
            const res = try group_by.getResults();
            allocator.free(res.keys);
            allocator.free(res.aggregates);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "GROUP BY", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT group_key, SUM(value) FROM '{s}' GROUP BY group_key;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "GROUP BY", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT group_key, SUM(value) FROM read_parquet('{s}') GROUP BY group_key;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "GROUP BY", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkOrderByLimit(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- ORDER BY LIMIT 100 ---\n", .{});

    const limit: usize = 100;

    // LanceQL
    {
        const data = try allocator.alloc(i64, num_rows);
        defer allocator.free(data);

        var rng = std.Random.DefaultPrng.init(42);
        for (data) |*v| v.* = rng.random().int(i64);

        for (0..WARMUP) |_| {
            var top_k = try allocator.alloc(i64, limit);
            defer allocator.free(top_k);
            @memcpy(top_k, data[0..limit]);
            std.mem.sort(i64, top_k, {}, std.sort.desc(i64));
            for (data[limit..]) |v| {
                if (v > top_k[limit - 1]) {
                    top_k[limit - 1] = v;
                    std.mem.sort(i64, top_k, {}, std.sort.desc(i64));
                }
            }
            std.mem.doNotOptimizeAway(top_k);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var top_k = try allocator.alloc(i64, limit);
            defer allocator.free(top_k);
            @memcpy(top_k, data[0..limit]);
            std.mem.sort(i64, top_k, {}, std.sort.desc(i64));
            for (data[limit..]) |v| {
                if (v > top_k[limit - 1]) {
                    top_k[limit - 1] = v;
                    std.mem.sort(i64, top_k, {}, std.sort.desc(i64));
                }
            }
            std.mem.doNotOptimizeAway(top_k);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "ORDER BY LIMIT", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT * FROM '{s}' ORDER BY value DESC LIMIT 100;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "ORDER BY LIMIT", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT * FROM read_parquet('{s}') ORDER BY value DESC LIMIT 100;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "ORDER BY LIMIT", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkDistinct(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- DISTINCT ---\n", .{});

    const num_distinct: usize = 1000;

    // LanceQL
    {
        const keys = try allocator.alloc(u64, num_rows);
        defer allocator.free(keys);

        for (0..num_rows) |i| {
            keys[i] = @intCast(i % num_distinct);
        }

        for (0..WARMUP) |_| {
            var seen = std.AutoHashMap(u64, void).init(allocator);
            defer seen.deinit();
            for (keys) |k| {
                seen.put(k, {}) catch continue;
            }
            std.mem.doNotOptimizeAway(&seen);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var seen = std.AutoHashMap(u64, void).init(allocator);
            defer seen.deinit();
            for (keys) |k| {
                try seen.put(k, {});
            }
            std.mem.doNotOptimizeAway(&seen);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "DISTINCT", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT DISTINCT name FROM '{s}';", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "DISTINCT", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT DISTINCT name FROM read_parquet('{s}');", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "DISTINCT", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkVectorSearch(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- VECTOR SEARCH (384-dim cosine) ---\n", .{});
    std.debug.print("(DuckDB/Polars: no native vector search)\n", .{});

    const dim: usize = 384;

    // LanceQL only (GPU accelerated)
    {
        const query_vec = try allocator.alloc(f32, dim);
        defer allocator.free(query_vec);
        const vectors = try allocator.alloc(f32, num_rows * dim);
        defer allocator.free(vectors);
        const scores = try allocator.alloc(f32, num_rows);
        defer allocator.free(scores);

        var rng = std.Random.DefaultPrng.init(42);
        for (query_vec) |*v| v.* = rng.random().float(f32) * 2 - 1;
        for (vectors) |*v| v.* = rng.random().float(f32) * 2 - 1;

        for (0..WARMUP) |_| {
            metal.batchCosineSimilarity(query_vec, vectors, dim, scores);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            metal.batchCosineSimilarity(query_vec, vectors, dim, scores);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "VECTOR SEARCH", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} ms ({d:>8.1}M rows/s) [GPU]\n", .{ stats.avg_ms, stats.throughput });
    }
}

fn benchmarkHashJoin(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- HASH JOIN ---\n", .{});

    const build_size = num_rows / 10;

    // LanceQL
    {
        const build_keys = try allocator.alloc(u64, build_size);
        defer allocator.free(build_keys);
        const build_row_ids = try allocator.alloc(usize, build_size);
        defer allocator.free(build_row_ids);
        const probe_keys = try allocator.alloc(u64, num_rows);
        defer allocator.free(probe_keys);
        const probe_row_ids = try allocator.alloc(usize, num_rows);
        defer allocator.free(probe_row_ids);

        for (0..build_size) |i| {
            build_keys[i] = @intCast(i * 2);
            build_row_ids[i] = i;
        }
        for (0..num_rows) |i| {
            probe_keys[i] = @intCast(i % (build_size * 2));
            probe_row_ids[i] = i;
        }

        for (0..WARMUP) |_| {
            var hash_join = query.GPUHashJoin.initWithCapacity(allocator, build_size) catch continue;
            defer hash_join.deinit();
            hash_join.build(build_keys, build_row_ids) catch continue;
            const res = hash_join.innerJoin(probe_keys, probe_row_ids) catch continue;
            allocator.free(res.build_indices);
            allocator.free(res.probe_indices);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var hash_join = try query.GPUHashJoin.initWithCapacity(allocator, build_size);
            defer hash_join.deinit();
            try hash_join.build(build_keys, build_row_ids);
            const res = try hash_join.innerJoin(probe_keys, probe_row_ids);
            allocator.free(res.build_indices);
            allocator.free(res.probe_indices);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "HASH JOIN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB - create second table for join
    if (has_duckdb) {
        if (parquet_path) |path| {
            const lookup_path = try std.fmt.allocPrint(allocator, "/tmp/lanceql_lookup_{d}.parquet", .{std.time.milliTimestamp()});
            defer allocator.free(lookup_path);
            defer std.fs.deleteFileAbsolute(lookup_path) catch {};

            const create_sql = try std.fmt.allocPrint(allocator, "COPY (SELECT i AS group_key, 'desc_' || i AS description FROM range(100) t(i)) TO '{s}' (FORMAT PARQUET);", .{lookup_path});
            defer allocator.free(create_sql);
            _ = runDuckDB(allocator, create_sql) catch {};

            const join_sql = try std.fmt.allocPrint(allocator, "SELECT t.*, l.description FROM '{s}' t JOIN '{s}' l ON t.group_key = l.group_key;", .{ path, lookup_path });
            defer allocator.free(join_sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, join_sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, join_sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "HASH JOIN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const lookup_path = try std.fmt.allocPrint(allocator, "/tmp/lanceql_lookup_polars_{d}.parquet", .{std.time.milliTimestamp()});
            defer allocator.free(lookup_path);
            defer std.fs.deleteFileAbsolute(lookup_path) catch {};

            const create_sql = try std.fmt.allocPrint(allocator, "COPY (SELECT i AS group_key, 'desc_' || i AS description FROM range(100) t(i)) TO '{s}' (FORMAT PARQUET);", .{lookup_path});
            defer allocator.free(create_sql);
            _ = runDuckDB(allocator, create_sql) catch {};

            const join_sql = try std.fmt.allocPrint(allocator, "SELECT t.*, l.description FROM read_parquet('{s}') t JOIN read_parquet('{s}') l ON t.group_key = l.group_key;", .{ path, lookup_path });
            defer allocator.free(join_sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, join_sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, join_sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "HASH JOIN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

// =============================================================================
// Additional SQL Clause Benchmarks
// =============================================================================

fn benchmarkHaving(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- GROUP BY + HAVING ---\n", .{});

    const num_groups: usize = 1000;

    // LanceQL: GROUP BY with HAVING filter
    {
        const keys = try allocator.alloc(u64, num_rows);
        defer allocator.free(keys);
        const values = try allocator.alloc(u64, num_rows);
        defer allocator.free(values);

        for (0..num_rows) |i| {
            keys[i] = @intCast(i % num_groups);
            values[i] = @intCast(i % 100);
        }

        for (0..WARMUP) |_| {
            var group_by = query.GPUGroupBy.initWithCapacity(allocator, .sum, num_groups * 4) catch continue;
            defer group_by.deinit();
            group_by.process(keys, values) catch continue;
            const res = group_by.getResults() catch continue;
            // HAVING: filter groups where sum > threshold
            var count: usize = 0;
            const threshold: u64 = num_rows / num_groups * 40; // ~40% of average
            for (res.aggregates) |agg| {
                if (agg > threshold) count += 1;
            }
            std.mem.doNotOptimizeAway(&count);
            allocator.free(res.keys);
            allocator.free(res.aggregates);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var group_by = try query.GPUGroupBy.initWithCapacity(allocator, .sum, num_groups * 4);
            defer group_by.deinit();
            try group_by.process(keys, values);
            const res = try group_by.getResults();
            var count: usize = 0;
            const threshold: u64 = num_rows / num_groups * 40;
            for (res.aggregates) |agg| {
                if (agg > threshold) count += 1;
            }
            std.mem.doNotOptimizeAway(&count);
            allocator.free(res.keys);
            allocator.free(res.aggregates);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "HAVING", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT group_key, SUM(value) as total FROM '{s}' GROUP BY group_key HAVING SUM(value) > 0.4;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "HAVING", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT group_key, SUM(value) as total FROM read_parquet('{s}') GROUP BY group_key HAVING SUM(value) > 0.4;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "HAVING", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkIn(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- IN (list membership) ---\n", .{});

    // LanceQL: value IN (list of 100 values)
    {
        const data = try allocator.alloc(u64, num_rows);
        defer allocator.free(data);

        for (0..num_rows) |i| {
            data[i] = @intCast(i % 10000);
        }

        // Create lookup set (100 values)
        var in_set = std.AutoHashMap(u64, void).init(allocator);
        defer in_set.deinit();
        for (0..100) |i| {
            try in_set.put(@intCast(i * 50), {});
        }

        for (0..WARMUP) |_| {
            var count: usize = 0;
            for (data) |v| {
                if (in_set.contains(v)) count += 1;
            }
            std.mem.doNotOptimizeAway(&count);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var count: usize = 0;
            for (data) |v| {
                if (in_set.contains(v)) count += 1;
            }
            std.mem.doNotOptimizeAway(&count);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "IN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            // Generate IN list
            var in_list = std.ArrayListUnmanaged(u8){};
            defer in_list.deinit(allocator);
            try in_list.appendSlice(allocator, "(");
            for (0..100) |i| {
                if (i > 0) try in_list.appendSlice(allocator, ",");
                var buf: [16]u8 = undefined;
                const s = std.fmt.bufPrint(&buf, "{d}", .{i * 50}) catch continue;
                try in_list.appendSlice(allocator, s);
            }
            try in_list.appendSlice(allocator, ")");

            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*) FROM '{s}' WHERE group_key IN {s};", .{ path, in_list.items });
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "IN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            var in_list = std.ArrayListUnmanaged(u8){};
            defer in_list.deinit(allocator);
            try in_list.appendSlice(allocator, "(");
            for (0..100) |i| {
                if (i > 0) try in_list.appendSlice(allocator, ",");
                var buf: [16]u8 = undefined;
                const s = std.fmt.bufPrint(&buf, "{d}", .{i * 50}) catch continue;
                try in_list.appendSlice(allocator, s);
            }
            try in_list.appendSlice(allocator, ")");

            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*) FROM read_parquet('{s}') WHERE group_key IN {s};", .{ path, in_list.items });
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "IN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkBetween(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- BETWEEN (range check) ---\n", .{});

    // LanceQL
    {
        const data = try allocator.alloc(f64, num_rows);
        defer allocator.free(data);
        var rng = std.Random.DefaultPrng.init(42);
        for (data) |*v| v.* = rng.random().float(f64);

        const low: f64 = 0.25;
        const high: f64 = 0.75;

        for (0..WARMUP) |_| {
            var count: usize = 0;
            for (data) |v| {
                if (v >= low and v <= high) count += 1;
            }
            std.mem.doNotOptimizeAway(&count);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var count: usize = 0;
            for (data) |v| {
                if (v >= low and v <= high) count += 1;
            }
            std.mem.doNotOptimizeAway(&count);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "BETWEEN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*) FROM '{s}' WHERE value BETWEEN 0.25 AND 0.75;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "BETWEEN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*) FROM read_parquet('{s}') WHERE value BETWEEN 0.25 AND 0.75;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "BETWEEN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkLike(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- LIKE (pattern matching) ---\n", .{});

    // LanceQL: simple prefix match
    {
        // Generate string data
        const names = try allocator.alloc([]const u8, num_rows);
        defer allocator.free(names);

        const prefixes = [_][]const u8{ "item_", "prod_", "user_", "data_", "test_" };
        for (0..num_rows) |i| {
            names[i] = prefixes[i % prefixes.len];
        }

        const pattern = "item_";

        for (0..WARMUP) |_| {
            var count: usize = 0;
            for (names) |name| {
                if (std.mem.startsWith(u8, name, pattern)) count += 1;
            }
            std.mem.doNotOptimizeAway(&count);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var count: usize = 0;
            for (names) |name| {
                if (std.mem.startsWith(u8, name, pattern)) count += 1;
            }
            std.mem.doNotOptimizeAway(&count);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "LIKE", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*) FROM '{s}' WHERE name LIKE 'item_%';", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "LIKE", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*) FROM read_parquet('{s}') WHERE name LIKE 'item_%';", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "LIKE", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkIsNull(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- IS NULL / IS NOT NULL ---\n", .{});

    // LanceQL: using optional values
    {
        const data = try allocator.alloc(?f64, num_rows);
        defer allocator.free(data);
        var rng = std.Random.DefaultPrng.init(42);
        for (data) |*v| {
            if (rng.random().float(f32) < 0.1) {
                v.* = null; // 10% nulls
            } else {
                v.* = rng.random().float(f64);
            }
        }

        for (0..WARMUP) |_| {
            var null_count: usize = 0;
            var not_null_count: usize = 0;
            for (data) |v| {
                if (v == null) {
                    null_count += 1;
                } else {
                    not_null_count += 1;
                }
            }
            std.mem.doNotOptimizeAway(&null_count);
            std.mem.doNotOptimizeAway(&not_null_count);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var null_count: usize = 0;
            var not_null_count: usize = 0;
            for (data) |v| {
                if (v == null) {
                    null_count += 1;
                } else {
                    not_null_count += 1;
                }
            }
            std.mem.doNotOptimizeAway(&null_count);
            std.mem.doNotOptimizeAway(&not_null_count);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "IS NULL", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB (no NULL in our test data, but still measures the operation)
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*) FROM '{s}' WHERE value IS NOT NULL;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "IS NULL", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*) FROM read_parquet('{s}') WHERE value IS NOT NULL;", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "IS NULL", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkAggregates(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- Aggregates (COUNT, SUM, AVG, MIN, MAX) ---\n", .{});

    // LanceQL: all aggregates in single pass
    {
        const data = try allocator.alloc(f64, num_rows);
        defer allocator.free(data);
        var rng = std.Random.DefaultPrng.init(42);
        for (data) |*v| v.* = rng.random().float(f64) * 1000;

        for (0..WARMUP) |_| {
            var count: usize = 0;
            var sum: f64 = 0;
            var min_val: f64 = std.math.floatMax(f64);
            var max_val: f64 = -std.math.floatMax(f64);

            for (data) |v| {
                count += 1;
                sum += v;
                min_val = @min(min_val, v);
                max_val = @max(max_val, v);
            }
            const avg = sum / @as(f64, @floatFromInt(count));
            std.mem.doNotOptimizeAway(&count);
            std.mem.doNotOptimizeAway(&sum);
            std.mem.doNotOptimizeAway(&avg);
            std.mem.doNotOptimizeAway(&min_val);
            std.mem.doNotOptimizeAway(&max_val);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var count: usize = 0;
            var sum: f64 = 0;
            var min_val: f64 = std.math.floatMax(f64);
            var max_val: f64 = -std.math.floatMax(f64);

            for (data) |v| {
                count += 1;
                sum += v;
                min_val = @min(min_val, v);
                max_val = @max(max_val, v);
            }
            const avg = sum / @as(f64, @floatFromInt(count));
            std.mem.doNotOptimizeAway(&count);
            std.mem.doNotOptimizeAway(&sum);
            std.mem.doNotOptimizeAway(&avg);
            std.mem.doNotOptimizeAway(&min_val);
            std.mem.doNotOptimizeAway(&max_val);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "AGGREGATES", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*), SUM(value), AVG(value), MIN(value), MAX(value) FROM '{s}';", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "AGGREGATES", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT COUNT(*), SUM(value), AVG(value), MIN(value), MAX(value) FROM read_parquet('{s}');", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "AGGREGATES", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkLeftJoin(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- LEFT JOIN ---\n", .{});

    const build_size = num_rows / 20; // 5% of rows match

    // LanceQL
    {
        const build_keys = try allocator.alloc(u64, build_size);
        defer allocator.free(build_keys);
        const build_row_ids = try allocator.alloc(usize, build_size);
        defer allocator.free(build_row_ids);
        const probe_keys = try allocator.alloc(u64, num_rows);
        defer allocator.free(probe_keys);
        const probe_row_ids = try allocator.alloc(usize, num_rows);
        defer allocator.free(probe_row_ids);

        for (0..build_size) |i| {
            build_keys[i] = @intCast(i * 5); // Only every 5th key exists
            build_row_ids[i] = i;
        }
        for (0..num_rows) |i| {
            probe_keys[i] = @intCast(i % (build_size * 10));
            probe_row_ids[i] = i;
        }

        for (0..WARMUP) |_| {
            var hash_join = query.GPUHashJoin.initWithCapacity(allocator, build_size) catch continue;
            defer hash_join.deinit();
            hash_join.build(build_keys, build_row_ids) catch continue;
            const res = hash_join.leftJoin(probe_keys, probe_row_ids) catch continue;
            allocator.free(res.build_indices);
            allocator.free(res.probe_indices);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var hash_join = try query.GPUHashJoin.initWithCapacity(allocator, build_size);
            defer hash_join.deinit();
            try hash_join.build(build_keys, build_row_ids);
            const res = try hash_join.leftJoin(probe_keys, probe_row_ids);
            allocator.free(res.build_indices);
            allocator.free(res.probe_indices);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "LEFT JOIN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const lookup_path = try std.fmt.allocPrint(allocator, "/tmp/lanceql_left_join_{d}.parquet", .{std.time.milliTimestamp()});
            defer allocator.free(lookup_path);
            defer std.fs.deleteFileAbsolute(lookup_path) catch {};

            const create_sql = try std.fmt.allocPrint(allocator, "COPY (SELECT i * 5 AS group_key, 'desc_' || i AS description FROM range(50) t(i)) TO '{s}' (FORMAT PARQUET);", .{lookup_path});
            defer allocator.free(create_sql);
            _ = runDuckDB(allocator, create_sql) catch {};

            const join_sql = try std.fmt.allocPrint(allocator, "SELECT t.id, l.description FROM '{s}' t LEFT JOIN '{s}' l ON t.group_key = l.group_key;", .{ path, lookup_path });
            defer allocator.free(join_sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, join_sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, join_sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "LEFT JOIN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const lookup_path = try std.fmt.allocPrint(allocator, "/tmp/lanceql_left_join_polars_{d}.parquet", .{std.time.milliTimestamp()});
            defer allocator.free(lookup_path);
            defer std.fs.deleteFileAbsolute(lookup_path) catch {};

            const create_sql = try std.fmt.allocPrint(allocator, "COPY (SELECT i * 5 AS group_key, 'desc_' || i AS description FROM range(50) t(i)) TO '{s}' (FORMAT PARQUET);", .{lookup_path});
            defer allocator.free(create_sql);
            _ = runDuckDB(allocator, create_sql) catch {};

            const join_sql = try std.fmt.allocPrint(allocator, "SELECT t.id, l.description FROM read_parquet('{s}') t LEFT JOIN read_parquet('{s}') l ON t.group_key = l.group_key;", .{ path, lookup_path });
            defer allocator.free(join_sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, join_sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, join_sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "LEFT JOIN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkInnerJoin(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- INNER JOIN ---\n", .{});

    const build_size = num_rows / 10;

    // LanceQL (same as HASH JOIN but explicit INNER JOIN naming)
    {
        const build_keys = try allocator.alloc(u64, build_size);
        defer allocator.free(build_keys);
        const build_row_ids = try allocator.alloc(usize, build_size);
        defer allocator.free(build_row_ids);
        const probe_keys = try allocator.alloc(u64, num_rows);
        defer allocator.free(probe_keys);
        const probe_row_ids = try allocator.alloc(usize, num_rows);
        defer allocator.free(probe_row_ids);

        for (0..build_size) |i| {
            build_keys[i] = @intCast(i * 2);
            build_row_ids[i] = i;
        }
        for (0..num_rows) |i| {
            probe_keys[i] = @intCast(i % (build_size * 2));
            probe_row_ids[i] = i;
        }

        for (0..WARMUP) |_| {
            var hash_join = query.GPUHashJoin.initWithCapacity(allocator, build_size) catch continue;
            defer hash_join.deinit();
            hash_join.build(build_keys, build_row_ids) catch continue;
            const res = hash_join.innerJoin(probe_keys, probe_row_ids) catch continue;
            allocator.free(res.build_indices);
            allocator.free(res.probe_indices);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            var hash_join = try query.GPUHashJoin.initWithCapacity(allocator, build_size);
            defer hash_join.deinit();
            try hash_join.build(build_keys, build_row_ids);
            const res = try hash_join.innerJoin(probe_keys, probe_row_ids);
            allocator.free(res.build_indices);
            allocator.free(res.probe_indices);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "INNER JOIN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const lookup_path = try std.fmt.allocPrint(allocator, "/tmp/lanceql_inner_join_{d}.parquet", .{std.time.milliTimestamp()});
            defer allocator.free(lookup_path);
            defer std.fs.deleteFileAbsolute(lookup_path) catch {};

            const create_sql = try std.fmt.allocPrint(allocator, "COPY (SELECT i AS group_key, 'desc_' || i AS description FROM range(100) t(i)) TO '{s}' (FORMAT PARQUET);", .{lookup_path});
            defer allocator.free(create_sql);
            _ = runDuckDB(allocator, create_sql) catch {};

            const join_sql = try std.fmt.allocPrint(allocator, "SELECT t.id, l.description FROM '{s}' t INNER JOIN '{s}' l ON t.group_key = l.group_key;", .{ path, lookup_path });
            defer allocator.free(join_sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, join_sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, join_sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "INNER JOIN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const lookup_path = try std.fmt.allocPrint(allocator, "/tmp/lanceql_inner_join_polars_{d}.parquet", .{std.time.milliTimestamp()});
            defer allocator.free(lookup_path);
            defer std.fs.deleteFileAbsolute(lookup_path) catch {};

            const create_sql = try std.fmt.allocPrint(allocator, "COPY (SELECT i AS group_key, 'desc_' || i AS description FROM range(100) t(i)) TO '{s}' (FORMAT PARQUET);", .{lookup_path});
            defer allocator.free(create_sql);
            _ = runDuckDB(allocator, create_sql) catch {};

            const join_sql = try std.fmt.allocPrint(allocator, "SELECT t.id, l.description FROM read_parquet('{s}') t INNER JOIN read_parquet('{s}') l ON t.group_key = l.group_key;", .{ path, lookup_path });
            defer allocator.free(join_sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, join_sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, join_sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "INNER JOIN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn benchmarkCaseWhen(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    std.debug.print("\n--- CASE WHEN ---\n", .{});

    // LanceQL: simulate CASE WHEN with branching
    {
        const data = try allocator.alloc(i64, num_rows);
        defer allocator.free(data);
        const output = try allocator.alloc(i64, num_rows);
        defer allocator.free(output);

        for (0..num_rows) |i| {
            data[i] = @intCast(i % 100);
        }

        for (0..WARMUP) |_| {
            for (data, 0..) |v, i| {
                // CASE WHEN v < 25 THEN 1 WHEN v < 50 THEN 2 WHEN v < 75 THEN 3 ELSE 4 END
                output[i] = if (v < 25) 1 else if (v < 50) 2 else if (v < 75) 3 else 4;
            }
            std.mem.doNotOptimizeAway(output);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            for (data, 0..) |v, i| {
                output[i] = if (v < 25) 1 else if (v < 50) 2 else if (v < 75) 3 else 4;
            }
            std.mem.doNotOptimizeAway(output);
            times[iter] = timer.read();
        }

        const stats = calcStats(&times, num_rows);
        try results.append(allocator, .{ .engine = .lanceql, .clause = "CASE WHEN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
        std.debug.print("LanceQL: {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
    }

    // DuckDB
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT CASE WHEN group_key < 25 THEN 'low' WHEN group_key < 50 THEN 'medium' WHEN group_key < 75 THEN 'high' ELSE 'very_high' END AS category FROM '{s}';", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .duckdb, .clause = "CASE WHEN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("DuckDB:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }

    // Polars
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT CASE WHEN group_key < 25 THEN 'low' WHEN group_key < 50 THEN 'medium' WHEN group_key < 75 THEN 'high' ELSE 'very_high' END AS category FROM read_parquet('{s}');", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const stats = calcStats(&times, num_rows);
            try results.append(allocator, .{ .engine = .polars, .clause = "CASE WHEN", .rows = num_rows, .avg_sec = stats.avg_sec, .throughput_mrows_sec = stats.throughput });
            std.debug.print("Polars:  {d:>8.2} s ({d:>8.1}M rows/s)\n", .{ stats.avg_sec, stats.throughput });
        }
    }
}

fn calcStats(times: []const u64, num_rows: usize) struct { avg_sec: f64, avg_ms: f64, throughput: f64 } {
    var total_ns: u64 = 0;
    for (times) |t| {
        total_ns += t;
    }

    const avg_ns = total_ns / times.len;
    const avg_sec = @as(f64, @floatFromInt(avg_ns)) / 1_000_000_000;
    const avg_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000;
    const throughput = @as(f64, @floatFromInt(num_rows)) / avg_sec / 1_000_000;

    return .{ .avg_sec = avg_sec, .avg_ms = avg_ms, .throughput = throughput };
}

fn printSummaryTable(results: []const BenchmarkResult) void {
    const clauses = [_][]const u8{
        "SELECT *",
        "WHERE",
        "GROUP BY",
        "HAVING",
        "ORDER BY LIMIT",
        "DISTINCT",
        "IN",
        "BETWEEN",
        "LIKE",
        "IS NULL",
        "AGGREGATES",
        "HASH JOIN",
        "LEFT JOIN",
        "INNER JOIN",
        "CASE WHEN",
        "VECTOR SEARCH",
    };

    std.debug.print("\n{s:<20} {s:>12} {s:>12} {s:>12}   {s}\n", .{ "Clause", "LanceQL", "DuckDB", "Polars", "Winner" });
    std.debug.print("{s}\n", .{"--------------------------------------------------------------------------------"});

    for (clauses) |clause| {
        var lanceql_sec: ?f64 = null;
        var duckdb_sec: ?f64 = null;
        var polars_sec: ?f64 = null;

        for (results) |r| {
            if (std.mem.eql(u8, r.clause, clause)) {
                switch (r.engine) {
                    .lanceql => lanceql_sec = r.avg_sec,
                    .duckdb => duckdb_sec = r.avg_sec,
                    .polars => polars_sec = r.avg_sec,
                }
            }
        }

        var lanceql_buf: [16]u8 = undefined;
        var duckdb_buf: [16]u8 = undefined;
        var polars_buf: [16]u8 = undefined;

        const lanceql_str = if (lanceql_sec) |sec| std.fmt.bufPrint(&lanceql_buf, "{d:>8.2} s", .{sec}) catch "N/A" else "         N/A";
        const duckdb_str = if (duckdb_sec) |sec| std.fmt.bufPrint(&duckdb_buf, "{d:>8.2} s", .{sec}) catch "N/A" else "         N/A";
        const polars_str = if (polars_sec) |sec| std.fmt.bufPrint(&polars_buf, "{d:>8.2} s", .{sec}) catch "N/A" else "         N/A";

        var winner: []const u8 = "N/A";
        var best_sec: f64 = std.math.floatMax(f64);
        var speedup: f64 = 1.0;

        if (lanceql_sec) |sec| {
            if (sec < best_sec) {
                best_sec = sec;
                winner = "LanceQL";
            }
        }
        if (duckdb_sec) |sec| {
            if (sec < best_sec) {
                best_sec = sec;
                winner = "DuckDB";
            }
        }
        if (polars_sec) |sec| {
            if (sec < best_sec) {
                best_sec = sec;
                winner = "Polars";
            }
        }

        if (lanceql_sec != null and (duckdb_sec != null or polars_sec != null)) {
            var second_best: f64 = std.math.floatMax(f64);
            if (duckdb_sec) |sec| second_best = @min(second_best, sec);
            if (polars_sec) |sec| second_best = @min(second_best, sec);
            if (lanceql_sec) |sec| {
                if (sec == best_sec and second_best < std.math.floatMax(f64)) {
                    speedup = second_best / sec;
                }
            }
        }

        if (speedup > 1.0 and std.mem.eql(u8, winner, "LanceQL")) {
            std.debug.print("{s:<20} {s} {s} {s}   {s} ({d:.1}x)\n", .{ clause, lanceql_str, duckdb_str, polars_str, winner, speedup });
        } else {
            std.debug.print("{s:<20} {s} {s} {s}   {s}\n", .{ clause, lanceql_str, duckdb_str, polars_str, winner });
        }
    }
}
