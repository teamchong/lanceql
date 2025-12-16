//! Benchmark: Vector operations (auto GPU/CPU on Apple Silicon)
//!
//! Run with: zig build bench-vector

const std = @import("std");
const metal = @import("lanceql.metal");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("Vector Operations Benchmark\n", .{});
    std.debug.print("===========================\n", .{});

    // Initialize GPU first
    _ = metal.initGPU();
    std.debug.print("Platform: {s}\n", .{metal.getPlatformInfo()});

    if (metal.isGPUReady()) {
        std.debug.print("GPU: {s}\n", .{metal.getGPUDeviceName()});
        std.debug.print("Auto-switch: GPU at 100K+ vectors (zero-copy)\n", .{});
    }
    std.debug.print("\n", .{});

    const dim: usize = 384;
    const iterations: usize = 10;

    // Batch cosine similarity at different scales
    std.debug.print("Batch Cosine Similarity ({d} dims):\n", .{dim});
    std.debug.print("-----------------------------------\n", .{});

    try benchmarkBatch(allocator, 10_000, dim, iterations);   // CPU path
    try benchmarkBatch(allocator, 100_000, dim, iterations);  // GPU path (threshold)
    try benchmarkBatch(allocator, 1_000_000, dim, iterations); // GPU path

    // Single vector ops
    std.debug.print("\nSingle Vector Ops ({d} dims):\n", .{dim});
    std.debug.print("-----------------------------\n", .{});

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);
    const vec = try allocator.alloc(f32, dim);
    defer allocator.free(vec);

    var rng = std.Random.DefaultPrng.init(42);
    for (query) |*v| v.* = rng.random().float(f32) * 2 - 1;
    for (vec) |*v| v.* = rng.random().float(f32) * 2 - 1;

    const single_iters: usize = 10000;

    var dot_ns: u64 = 0;
    for (0..single_iters) |_| {
        var timer = try std.time.Timer.start();
        _ = metal.dotProduct(query, vec);
        dot_ns += timer.read();
    }
    std.debug.print("Dot product:  {d:.0} ns/op\n", .{@as(f64, @floatFromInt(dot_ns)) / single_iters});

    var cos_ns: u64 = 0;
    for (0..single_iters) |_| {
        var timer = try std.time.Timer.start();
        _ = metal.cosineSimilarity(query, vec);
        cos_ns += timer.read();
    }
    std.debug.print("Cosine sim:   {d:.0} ns/op\n", .{@as(f64, @floatFromInt(cos_ns)) / single_iters});

    var l2_ns: u64 = 0;
    for (0..single_iters) |_| {
        var timer = try std.time.Timer.start();
        _ = metal.l2DistanceSquared(query, vec);
        l2_ns += timer.read();
    }
    std.debug.print("L2 distance:  {d:.0} ns/op\n", .{@as(f64, @floatFromInt(l2_ns)) / single_iters});

    metal.cleanupGPU();
}

fn benchmarkBatch(allocator: std.mem.Allocator, num_vectors: usize, dim: usize, iterations: usize) !void {
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
    for (0..3) |_| {
        metal.batchCosineSimilarity(query, vectors, dim, scores);
    }

    var min_ns: u64 = std.math.maxInt(u64);
    for (0..iterations) |_| {
        var timer = try std.time.Timer.start();
        metal.batchCosineSimilarity(query, vectors, dim, scores);
        min_ns = @min(min_ns, timer.read());
    }

    const ms = @as(f64, @floatFromInt(min_ns)) / 1_000_000;
    const mvps = @as(f64, @floatFromInt(num_vectors)) / (ms * 1000); // M vectors/sec

    const path = if (num_vectors >= 100_000) "GPU" else "CPU";
    const scale_str = if (num_vectors >= 1_000_000) "  1M" else if (num_vectors >= 100_000) "100K" else " 10K";
    std.debug.print("{s} vectors: {d:6.1} ms ({s}) - {d:.1}M vec/s\n", .{ scale_str, ms, path, mvps });
}
