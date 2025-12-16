//! Benchmark: Vector operations (Accelerate vs pure SIMD)
//!
//! Run with: zig build bench-vector

const std = @import("std");
const metal = @import("lanceql.metal");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("Vector Operations Benchmark\n", .{});
    std.debug.print("===========================\n", .{});
    std.debug.print("Platform: {s}\n\n", .{metal.getPlatformInfo()});

    // Test vectors (384-dim like MiniLM embeddings)
    const dim: usize = 384;
    const num_vectors: usize = 10000;
    const iterations: usize = 100;

    // Allocate test data
    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);

    const vectors = try allocator.alloc(f32, num_vectors * dim);
    defer allocator.free(vectors);

    const scores = try allocator.alloc(f32, num_vectors);
    defer allocator.free(scores);

    // Initialize with random data
    var rng = std.Random.DefaultPrng.init(42);
    for (query) |*v| v.* = rng.random().float(f32) * 2 - 1;
    for (vectors) |*v| v.* = rng.random().float(f32) * 2 - 1;

    // Warmup
    for (0..10) |_| {
        metal.batchCosineSimilarity(query, vectors, dim, scores);
    }

    // Benchmark
    var total_ns: u64 = 0;
    var min_ns: u64 = std.math.maxInt(u64);

    for (0..iterations) |_| {
        var timer = try std.time.Timer.start();
        metal.batchCosineSimilarity(query, vectors, dim, scores);
        const elapsed = timer.read();
        total_ns += elapsed;
        min_ns = @min(min_ns, elapsed);
    }

    const avg_ns = total_ns / iterations;
    const avg_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000;
    const min_ms = @as(f64, @floatFromInt(min_ns)) / 1_000_000;

    // Results
    std.debug.print("Batch Cosine Similarity ({d} vectors x {d} dims)\n", .{ num_vectors, dim });
    std.debug.print("  Min:  {d:.3} ms\n", .{min_ms});
    std.debug.print("  Avg:  {d:.3} ms\n", .{avg_ms});

    const vectors_per_sec = @as(f64, @floatFromInt(num_vectors)) / (avg_ms / 1000);
    std.debug.print("  Throughput: {d:.1}K vectors/sec\n", .{vectors_per_sec / 1000});

    // Single operation benchmarks
    std.debug.print("\nSingle Vector Operations ({d} dims, {d} iterations):\n", .{ dim, iterations * 100 });

    // Dot product
    var dot_ns: u64 = 0;
    for (0..iterations * 100) |_| {
        var timer = try std.time.Timer.start();
        _ = metal.dotProduct(query, vectors[0..dim]);
        dot_ns += timer.read();
    }
    std.debug.print("  Dot product:    {d:.1} ns/op\n", .{@as(f64, @floatFromInt(dot_ns)) / @as(f64, @floatFromInt(iterations * 100))});

    // Cosine similarity
    var cos_ns: u64 = 0;
    for (0..iterations * 100) |_| {
        var timer = try std.time.Timer.start();
        _ = metal.cosineSimilarity(query, vectors[0..dim]);
        cos_ns += timer.read();
    }
    std.debug.print("  Cosine sim:     {d:.1} ns/op\n", .{@as(f64, @floatFromInt(cos_ns)) / @as(f64, @floatFromInt(iterations * 100))});

    // L2 distance
    var l2_ns: u64 = 0;
    for (0..iterations * 100) |_| {
        var timer = try std.time.Timer.start();
        _ = metal.l2DistanceSquared(query, vectors[0..dim]);
        l2_ns += timer.read();
    }
    std.debug.print("  L2 distance:    {d:.1} ns/op\n", .{@as(f64, @floatFromInt(l2_ns)) / @as(f64, @floatFromInt(iterations * 100))});
}
