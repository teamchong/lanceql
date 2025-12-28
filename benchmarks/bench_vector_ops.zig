//! Benchmark: Vector operations (auto GPU/CPU on Apple Silicon)
//!
//! Compares: LanceQL (GPU/CPU) vs DuckDB vs Polars
//! Run with: zig build bench-vector

const std = @import("std");
const metal = @import("lanceql.metal");

// Each benchmark should run 5+ seconds
const WARMUP = 3;
const BATCH_ITERATIONS = 200; // Batch benchmarks (~5+ seconds at ~50ms/batch)
const SINGLE_ITERATIONS: usize = 100_000_000; // Single vector ops (~5+ seconds at ~50ns)
const SUBPROCESS_ITERATIONS: usize = 200; // Subprocess (~5+ seconds at ~30ms)
const MIN_BENCHMARK_SECONDS: f64 = 5.0;

// Engine availability
var has_duckdb: bool = false;
var has_polars: bool = false;

fn checkCommand(allocator: std.mem.Allocator, cmd: []const u8) bool {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "which", cmd },
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

fn runDuckDB(allocator: std.mem.Allocator, sql: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "duckdb", "-csv", "-c", sql },
        .max_output_bytes = 10 * 1024 * 1024,
    }) catch return error.DuckDBFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |code| if (code != 0) return error.DuckDBFailed,
        else => return error.DuckDBFailed,
    }
    return timer.read();
}

fn runPolars(allocator: std.mem.Allocator, code: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", "-c", code },
        .max_output_bytes = 10 * 1024 * 1024,
    }) catch return error.PolarsFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |exit_code| if (exit_code != 0) return error.PolarsFailed,
        else => return error.PolarsFailed,
    }
    return timer.read();
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Vector Operations Benchmark: LanceQL vs DuckDB vs Polars\n", .{});
    std.debug.print("================================================================================\n", .{});

    // Initialize GPU first
    _ = metal.initGPU();

    // Check available engines
    has_duckdb = checkCommand(allocator, "duckdb");
    has_polars = checkCommand(allocator, "python3");

    std.debug.print("\nEngines:\n", .{});
    std.debug.print("  - LanceQL:  {s}\n", .{metal.getPlatformInfo()});
    if (metal.isGPUReady()) {
        std.debug.print("  - GPU:      {s}\n", .{metal.getGPUDeviceName()});
    }
    std.debug.print("  - DuckDB:   {s}\n", .{if (has_duckdb) "yes" else "no"});
    std.debug.print("  - Polars:   {s}\n", .{if (has_polars) "yes" else "no"});
    std.debug.print("\n", .{});

    const dim: usize = 384;

    // Batch cosine similarity at production scale
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Batch Cosine Similarity ({d}-dim embeddings)\n", .{dim});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<12} {s:>10} {s:>12} {s:>15}\n", .{ "Vectors", "Time", "Path", "Throughput" });
    std.debug.print("{s:<12} {s:>10} {s:>12} {s:>15}\n", .{ "-" ** 12, "-" ** 10, "-" ** 12, "-" ** 15 });

    try benchmarkBatch(allocator, 100_000, dim);    // CPU path (baseline)
    try benchmarkBatch(allocator, 1_000_000, dim);  // GPU path
    try benchmarkBatch(allocator, 10_000_000, dim); // GPU path (production scale)

    // Single vector ops (high iteration count for accurate ns/op measurement)
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Single Vector Operations ({d}-dim)\n", .{dim});
    std.debug.print("================================================================================\n", .{});

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);
    const vec = try allocator.alloc(f32, dim);
    defer allocator.free(vec);

    var rng = std.Random.DefaultPrng.init(42);
    for (query) |*v| v.* = rng.random().float(f32) * 2 - 1;
    for (vec) |*v| v.* = rng.random().float(f32) * 2 - 1;

    std.debug.print("Running {d}M iterations per operation...\n\n", .{SINGLE_ITERATIONS / 1_000_000});

    var timer = try std.time.Timer.start();
    var dot_ns: u64 = 0;
    for (0..SINGLE_ITERATIONS) |_| {
        var t = try std.time.Timer.start();
        _ = metal.dotProduct(query, vec);
        dot_ns += t.read();
    }
    const dot_total = timer.read();
    std.debug.print("Dot product:  {d:.0} ns/op  (total: {d:.1} s)\n", .{ @as(f64, @floatFromInt(dot_ns)) / @as(f64, @floatFromInt(SINGLE_ITERATIONS)), @as(f64, @floatFromInt(dot_total)) / 1_000_000_000 });

    timer = try std.time.Timer.start();
    var cos_ns: u64 = 0;
    for (0..SINGLE_ITERATIONS) |_| {
        var t = try std.time.Timer.start();
        _ = metal.cosineSimilarity(query, vec);
        cos_ns += t.read();
    }
    const cos_total = timer.read();
    std.debug.print("Cosine sim:   {d:.0} ns/op  (total: {d:.1} s)\n", .{ @as(f64, @floatFromInt(cos_ns)) / @as(f64, @floatFromInt(SINGLE_ITERATIONS)), @as(f64, @floatFromInt(cos_total)) / 1_000_000_000 });

    timer = try std.time.Timer.start();
    var l2_ns: u64 = 0;
    for (0..SINGLE_ITERATIONS) |_| {
        var t = try std.time.Timer.start();
        _ = metal.l2DistanceSquared(query, vec);
        l2_ns += t.read();
    }
    const l2_total = timer.read();
    const lanceql_l2_ns = @as(f64, @floatFromInt(l2_ns)) / @as(f64, @floatFromInt(SINGLE_ITERATIONS));
    std.debug.print("L2 distance:  {d:.0} ns/op  (total: {d:.1} s)\n", .{ lanceql_l2_ns, @as(f64, @floatFromInt(l2_total)) / 1_000_000_000 });

    // ==========================================================================
    // DuckDB/Polars Comparison (cosine similarity)
    // ==========================================================================
    if (has_duckdb or has_polars) {
        std.debug.print("\n================================================================================\n", .{});
        std.debug.print("Cosine Similarity Comparison: LanceQL vs DuckDB vs Polars\n", .{});
        std.debug.print("================================================================================\n", .{});

        // Build vector strings for subprocess
        var query_str = std.ArrayListUnmanaged(u8){};
        defer query_str.deinit(allocator);
        try query_str.appendSlice(allocator, "[");
        for (query, 0..) |v, i| {
            if (i > 0) try query_str.appendSlice(allocator, ",");
            try std.fmt.format(query_str.writer(allocator), "{d:.6}", .{v});
        }
        try query_str.appendSlice(allocator, "]");

        var vec_str = std.ArrayListUnmanaged(u8){};
        defer vec_str.deinit(allocator);
        try vec_str.appendSlice(allocator, "[");
        for (vec, 0..) |v, i| {
            if (i > 0) try vec_str.appendSlice(allocator, ",");
            try std.fmt.format(vec_str.writer(allocator), "{d:.6}", .{v});
        }
        try vec_str.appendSlice(allocator, "]");

        const lanceql_cos_ns = @as(f64, @floatFromInt(cos_ns)) / @as(f64, @floatFromInt(SINGLE_ITERATIONS));
        std.debug.print("{s:<25} {d:>12.1} ns   (baseline)\n", .{ "LanceQL", lanceql_cos_ns });

        // DuckDB cosine similarity
        if (has_duckdb) {
            const sql = try std.fmt.allocPrint(allocator,
                \\SELECT list_cosine_similarity({s}::FLOAT[], {s}::FLOAT[]);
            , .{ query_str.items, vec_str.items });
            defer allocator.free(sql);

            var duckdb_ns: u64 = 0;
            for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
            for (0..SUBPROCESS_ITERATIONS) |_| {
                duckdb_ns += runDuckDB(allocator, sql) catch 0;
            }
            const duckdb_per_op = @as(f64, @floatFromInt(duckdb_ns)) / @as(f64, @floatFromInt(SUBPROCESS_ITERATIONS));
            const duckdb_ratio = duckdb_per_op / lanceql_cos_ns;
            std.debug.print("{s:<25} {d:>12.1} ns   {d:.0}x slower\n", .{ "DuckDB (subprocess)", duckdb_per_op, duckdb_ratio });
        }

        // Polars cosine similarity
        if (has_polars) {
            const py_code = try std.fmt.allocPrint(allocator,
                \\import numpy as np
                \\a = np.array({s})
                \\b = np.array({s})
                \\cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            , .{ query_str.items, vec_str.items });
            defer allocator.free(py_code);

            var polars_ns: u64 = 0;
            for (0..WARMUP) |_| _ = runPolars(allocator, py_code) catch 0;
            for (0..SUBPROCESS_ITERATIONS) |_| {
                polars_ns += runPolars(allocator, py_code) catch 0;
            }
            const polars_per_op = @as(f64, @floatFromInt(polars_ns)) / @as(f64, @floatFromInt(SUBPROCESS_ITERATIONS));
            const polars_ratio = polars_per_op / lanceql_cos_ns;
            std.debug.print("{s:<25} {d:>12.1} ns   {d:.0}x slower\n", .{ "Polars (subprocess)", polars_per_op, polars_ratio });
        }

        std.debug.print("\nNote: Subprocess times include process startup overhead.\n", .{});
    }

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Benchmark Complete\n", .{});
    std.debug.print("================================================================================\n", .{});

    metal.cleanupGPU();
}

fn benchmarkBatch(allocator: std.mem.Allocator, num_vectors: usize, dim: usize) !void {
    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);

    const vectors = try allocator.alloc(f32, num_vectors * dim);
    defer allocator.free(vectors);

    const scores = try allocator.alloc(f32, num_vectors);
    defer allocator.free(scores);

    var rng = std.Random.DefaultPrng.init(42);
    for (query) |*v| v.* = rng.random().float(f32) * 2 - 1;
    for (vectors) |*v| v.* = rng.random().float(f32) * 2 - 1;

    // Warmup
    for (0..WARMUP) |_| {
        metal.batchCosineSimilarity(query, vectors, dim, scores);
    }

    // Run enough iterations to get 5+ seconds
    var total_timer = try std.time.Timer.start();
    var total_ns: u64 = 0;
    for (0..BATCH_ITERATIONS) |_| {
        var timer = try std.time.Timer.start();
        metal.batchCosineSimilarity(query, vectors, dim, scores);
        total_ns += timer.read();
    }
    const wall_time = total_timer.read();

    const avg_sec = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(BATCH_ITERATIONS)) / 1_000_000_000;
    const total_sec = @as(f64, @floatFromInt(wall_time)) / 1_000_000_000;
    const mvps = @as(f64, @floatFromInt(num_vectors)) / avg_sec / 1_000_000;

    const path = if (num_vectors >= 100_000) "GPU" else "CPU";
    const scale_str = if (num_vectors >= 10_000_000) "10M" else if (num_vectors >= 1_000_000) " 1M" else "100K";

    std.debug.print("{s:<12} {d:>8.2} s {s:>12} {d:>12.1}M/sec  (total: {d:.1}s)\n", .{ scale_str, avg_sec, path, mvps, total_sec });
}
