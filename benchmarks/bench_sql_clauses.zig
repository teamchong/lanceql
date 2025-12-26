//! SQL Clause Benchmark: LanceQL vs DuckDB vs Polars
//!
//! Benchmarks different SQL operations to identify optimization opportunities.
//! Run with: zig build bench-sql
//!
//! Output format: JSON for CI comparison

const std = @import("std");
const lanceql = @import("lanceql");
const query = @import("lanceql.query");
const metal = @import("lanceql.metal");

const BenchmarkResult = struct {
    name: []const u8,
    clause: []const u8,
    rows: usize,
    min_ms: f64,
    avg_ms: f64,
    max_ms: f64,
    throughput_mrows_sec: f64,
};

const WARMUP = 3;
const ITERATIONS = 10;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize GPU
    _ = metal.initGPU();

    std.debug.print("LanceQL SQL Clause Benchmark\n", .{});
    std.debug.print("============================\n", .{});
    std.debug.print("Platform: {s}\n", .{metal.getPlatformInfo()});
    if (metal.isGPUReady()) {
        std.debug.print("GPU: {s}\n", .{metal.getGPUDeviceName()});
    }
    std.debug.print("Warmup: {d}, Iterations: {d}\n\n", .{ WARMUP, ITERATIONS });

    var results = std.ArrayListUnmanaged(BenchmarkResult){};
    defer results.deinit(allocator);

    // Generate test data
    const sizes = [_]usize{ 10_000, 100_000, 1_000_000 };

    for (sizes) |num_rows| {
        std.debug.print("\n=== Dataset: {d} rows ===\n", .{num_rows});

        // Benchmark each clause
        try benchmarkFullScan(allocator, &results, num_rows);
        try benchmarkProjection(allocator, &results, num_rows);
        try benchmarkFilter(allocator, &results, num_rows);
        try benchmarkGroupBy(allocator, &results, num_rows);
        try benchmarkOrderByLimit(allocator, &results, num_rows);
        try benchmarkDistinct(allocator, &results, num_rows);
        try benchmarkVectorSearch(allocator, &results, num_rows);
        try benchmarkHashJoin(allocator, &results, num_rows);
    }

    // Output JSON for CI
    std.debug.print("\n=== JSON Results ===\n", .{});
    outputJSON(results.items);

    metal.cleanupGPU();
}

fn benchmarkFullScan(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    // Simulate: SELECT * FROM table
    const data = try allocator.alloc(i64, num_rows);
    defer allocator.free(data);
    for (data, 0..) |*v, i| v.* = @intCast(i);

    // Warmup
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

    const result = calcStats(&times, num_rows, "SELECT *");
    try results.append(allocator, result);
    printResult(result);
}

fn benchmarkProjection(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    // Simulate: SELECT col1, col2 FROM table (2 of 5 columns)
    const col1 = try allocator.alloc(i64, num_rows);
    defer allocator.free(col1);
    const col2 = try allocator.alloc(f64, num_rows);
    defer allocator.free(col2);

    for (0..num_rows) |i| {
        col1[i] = @intCast(i);
        col2[i] = @floatFromInt(i);
    }

    for (0..WARMUP) |_| {
        var sum: f64 = 0;
        for (col1, col2) |c1, c2| sum += @as(f64, @floatFromInt(c1)) + c2;
        std.mem.doNotOptimizeAway(&sum);
    }

    var times: [ITERATIONS]u64 = undefined;
    for (0..ITERATIONS) |iter| {
        var timer = try std.time.Timer.start();
        var sum: f64 = 0;
        for (col1, col2) |c1, c2| sum += @as(f64, @floatFromInt(c1)) + c2;
        std.mem.doNotOptimizeAway(&sum);
        times[iter] = timer.read();
    }

    const result = calcStats(&times, num_rows, "SELECT col1,col2");
    try results.append(allocator, result);
    printResult(result);
}

fn benchmarkFilter(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    // Simulate: SELECT * FROM table WHERE value > threshold
    const data = try allocator.alloc(i64, num_rows);
    defer allocator.free(data);
    for (data, 0..) |*v, i| v.* = @intCast(i % 1000);

    const threshold: i64 = 500;

    for (0..WARMUP) |_| {
        var count: usize = 0;
        for (data) |v| {
            if (v > threshold) count += 1;
        }
        std.mem.doNotOptimizeAway(&count);
    }

    var times: [ITERATIONS]u64 = undefined;
    for (0..ITERATIONS) |iter| {
        var timer = try std.time.Timer.start();
        var count: usize = 0;
        for (data) |v| {
            if (v > threshold) count += 1;
        }
        std.mem.doNotOptimizeAway(&count);
        times[iter] = timer.read();
    }

    const result = calcStats(&times, num_rows, "WHERE x > 500");
    try results.append(allocator, result);
    printResult(result);
}

fn benchmarkGroupBy(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    // Simulate: SELECT key, SUM(value) FROM table GROUP BY key
    const num_groups: usize = 100;

    const keys = try allocator.alloc(u64, num_rows);
    defer allocator.free(keys);
    const values = try allocator.alloc(u64, num_rows);
    defer allocator.free(values);

    for (0..num_rows) |i| {
        keys[i] = @intCast(i % num_groups);
        values[i] = 1;
    }

    // Use GPU GROUP BY
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

    const result = calcStats(&times, num_rows, "GROUP BY + SUM");
    try results.append(allocator, result);
    printResult(result);
}

fn benchmarkOrderByLimit(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    // Simulate: SELECT * FROM table ORDER BY value DESC LIMIT 100
    const data = try allocator.alloc(i64, num_rows);
    defer allocator.free(data);

    var rng = std.Random.DefaultPrng.init(42);
    for (data) |*v| v.* = rng.random().int(i64);

    const limit: usize = 100;

    for (0..WARMUP) |_| {
        // Partial sort for top-k
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

    const result = calcStats(&times, num_rows, "ORDER BY LIMIT");
    try results.append(allocator, result);
    printResult(result);
}

fn benchmarkDistinct(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    // Simulate: SELECT DISTINCT key FROM table
    const num_distinct: usize = 1000;
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

    const result = calcStats(&times, num_rows, "DISTINCT");
    try results.append(allocator, result);
    printResult(result);
}

fn benchmarkVectorSearch(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    // Simulate: SELECT * FROM table NEAR 'query' TOPK 20
    const dim: usize = 384;
    const top_k: usize = 20;

    const query_vec = try allocator.alloc(f32, dim);
    defer allocator.free(query_vec);
    const vectors = try allocator.alloc(f32, num_rows * dim);
    defer allocator.free(vectors);
    const scores = try allocator.alloc(f32, num_rows);
    defer allocator.free(scores);

    var rng = std.Random.DefaultPrng.init(42);
    for (query_vec) |*v| v.* = rng.random().float(f32) * 2 - 1;
    for (vectors) |*v| v.* = rng.random().float(f32) * 2 - 1;

    // Warmup
    for (0..WARMUP) |_| {
        metal.batchCosineSimilarity(query_vec, vectors, dim, scores);
    }

    var times: [ITERATIONS]u64 = undefined;
    for (0..ITERATIONS) |iter| {
        var timer = try std.time.Timer.start();
        metal.batchCosineSimilarity(query_vec, vectors, dim, scores);
        // Find top-k (simplified - just find max for benchmark)
        var max_score: f32 = scores[0];
        for (scores[1..]) |s| {
            if (s > max_score) max_score = s;
        }
        _ = top_k;
        std.mem.doNotOptimizeAway(&max_score);
        times[iter] = timer.read();
    }

    const result = calcStats(&times, num_rows, "VECTOR SEARCH");
    try results.append(allocator, result);
    printResult(result);
}

fn benchmarkHashJoin(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchmarkResult), num_rows: usize) !void {
    // Simulate: SELECT * FROM left JOIN right ON left.key = right.key
    const build_size = num_rows / 10; // 10% build side

    const build_keys = try allocator.alloc(u64, build_size);
    defer allocator.free(build_keys);
    const build_row_ids = try allocator.alloc(usize, build_size);
    defer allocator.free(build_row_ids);

    const probe_keys = try allocator.alloc(u64, num_rows);
    defer allocator.free(probe_keys);
    const probe_row_ids = try allocator.alloc(usize, num_rows);
    defer allocator.free(probe_row_ids);

    for (0..build_size) |i| {
        build_keys[i] = @intCast(i * 2); // Even keys
        build_row_ids[i] = i;
    }
    for (0..num_rows) |i| {
        probe_keys[i] = @intCast(i % (build_size * 2));
        probe_row_ids[i] = i;
    }

    // Warmup
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

    const result = calcStats(&times, num_rows, "HASH JOIN");
    try results.append(allocator, result);
    printResult(result);
}

fn calcStats(times: []const u64, num_rows: usize, clause: []const u8) BenchmarkResult {
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;
    var total_ns: u64 = 0;

    for (times) |t| {
        min_ns = @min(min_ns, t);
        max_ns = @max(max_ns, t);
        total_ns += t;
    }

    const avg_ns = total_ns / times.len;
    const avg_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000;
    const throughput = @as(f64, @floatFromInt(num_rows)) / avg_ms / 1000; // M rows/sec

    return .{
        .name = "LanceQL",
        .clause = clause,
        .rows = num_rows,
        .min_ms = @as(f64, @floatFromInt(min_ns)) / 1_000_000,
        .avg_ms = avg_ms,
        .max_ms = @as(f64, @floatFromInt(max_ns)) / 1_000_000,
        .throughput_mrows_sec = throughput,
    };
}

fn printResult(r: BenchmarkResult) void {
    std.debug.print("{s:<20} {d:>10} rows  {d:>8.2} ms  {d:>8.1}M rows/s\n", .{
        r.clause,
        r.rows,
        r.avg_ms,
        r.throughput_mrows_sec,
    });
}

fn outputJSON(results: []const BenchmarkResult) void {
    std.debug.print("[\n", .{});
    for (results, 0..) |r, i| {
        std.debug.print(
            \\  {{"name": "{s}", "clause": "{s}", "rows": {d}, "avg_ms": {d:.3}, "throughput_mrows_sec": {d:.2}}}
        , .{ r.name, r.clause, r.rows, r.avg_ms, r.throughput_mrows_sec });
        if (i < results.len - 1) std.debug.print(",", .{});
        std.debug.print("\n", .{});
    }
    std.debug.print("]\n", .{});
}
