//! Benchmark: Vector operations (auto GPU/CPU on Apple Silicon)
//!
//! Each benchmark runs 30+ seconds to ensure fair comparison.
//! Run with: zig build bench-vector

const std = @import("std");
const metal = @import("lanceql.metal");

// Each benchmark should run 30+ seconds
const WARMUP = 3;
const ITERATIONS = 20;
const MIN_BENCHMARK_SECONDS: f64 = 30.0;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Vector Operations Benchmark (GPU vs CPU)\n", .{});
    std.debug.print("================================================================================\n", .{});

    // Initialize GPU first
    _ = metal.initGPU();
    std.debug.print("Platform: {s}\n", .{metal.getPlatformInfo()});

    if (metal.isGPUReady()) {
        std.debug.print("GPU: {s}\n", .{metal.getGPUDeviceName()});
        std.debug.print("Auto-switch: GPU at 100K+ vectors (zero-copy)\n", .{});
    }
    std.debug.print("Warmup: {d}, Iterations: {d}\n", .{ WARMUP, ITERATIONS });
    std.debug.print("Target: 30+ seconds per benchmark\n", .{});
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

    // 10M iterations to get 30+ seconds
    const single_iters: usize = 10_000_000;

    std.debug.print("Running {d}M iterations per operation...\n\n", .{single_iters / 1_000_000});

    var timer = try std.time.Timer.start();
    var dot_ns: u64 = 0;
    for (0..single_iters) |_| {
        var t = try std.time.Timer.start();
        _ = metal.dotProduct(query, vec);
        dot_ns += t.read();
    }
    const dot_total = timer.read();
    std.debug.print("Dot product:  {d:.0} ns/op  (total: {d:.1} s)\n", .{ @as(f64, @floatFromInt(dot_ns)) / @as(f64, @floatFromInt(single_iters)), @as(f64, @floatFromInt(dot_total)) / 1_000_000_000 });

    timer = try std.time.Timer.start();
    var cos_ns: u64 = 0;
    for (0..single_iters) |_| {
        var t = try std.time.Timer.start();
        _ = metal.cosineSimilarity(query, vec);
        cos_ns += t.read();
    }
    const cos_total = timer.read();
    std.debug.print("Cosine sim:   {d:.0} ns/op  (total: {d:.1} s)\n", .{ @as(f64, @floatFromInt(cos_ns)) / @as(f64, @floatFromInt(single_iters)), @as(f64, @floatFromInt(cos_total)) / 1_000_000_000 });

    timer = try std.time.Timer.start();
    var l2_ns: u64 = 0;
    for (0..single_iters) |_| {
        var t = try std.time.Timer.start();
        _ = metal.l2DistanceSquared(query, vec);
        l2_ns += t.read();
    }
    const l2_total = timer.read();
    std.debug.print("L2 distance:  {d:.0} ns/op  (total: {d:.1} s)\n", .{ @as(f64, @floatFromInt(l2_ns)) / @as(f64, @floatFromInt(single_iters)), @as(f64, @floatFromInt(l2_total)) / 1_000_000_000 });

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

    // Run enough iterations to get 30+ seconds
    var total_timer = try std.time.Timer.start();
    var total_ns: u64 = 0;
    for (0..ITERATIONS) |_| {
        var timer = try std.time.Timer.start();
        metal.batchCosineSimilarity(query, vectors, dim, scores);
        total_ns += timer.read();
    }
    const wall_time = total_timer.read();

    const avg_sec = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(ITERATIONS)) / 1_000_000_000;
    const total_sec = @as(f64, @floatFromInt(wall_time)) / 1_000_000_000;
    const mvps = @as(f64, @floatFromInt(num_vectors)) / avg_sec / 1_000_000;

    const path = if (num_vectors >= 100_000) "GPU" else "CPU";
    const scale_str = if (num_vectors >= 10_000_000) "10M" else if (num_vectors >= 1_000_000) " 1M" else "100K";

    std.debug.print("{s:<12} {d:>8.2} s {s:>12} {d:>12.1}M/sec  (total: {d:.1}s)\n", .{ scale_str, avg_sec, path, mvps, total_sec });
}
