//! Benchmark: SIMD vs GPU Tiered Dispatch
//!
//! Tests batch vector operations at different scales:
//!   - Small batches: CPU SIMD is optimal (low overhead)
//!   - Large batches (>=10K): GPU Metal would provide higher throughput
//!
//! This benchmark establishes the SIMD baseline for comparison with GPU.
//!
//! Run: zig build bench-tiered

const std = @import("std");

// Direct SIMD implementation for comparison
fn simdDotProduct(a: []const f32, b: []const f32) f32 {
    const Vec4 = @Vector(4, f32);
    var sum_vec: Vec4 = @splat(0);
    var i: usize = 0;
    const simd_end = a.len & ~@as(usize, 3);

    while (i < simd_end) : (i += 4) {
        const a_vec: Vec4 = a[i..][0..4].*;
        const b_vec: Vec4 = b[i..][0..4].*;
        sum_vec += a_vec * b_vec;
    }

    var sum = @reduce(.Add, sum_vec);
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

fn simdBatchDotProduct(
    allocator: std.mem.Allocator,
    query: []const f32,
    vectors: []const f32,
    num_vectors: usize,
    dim: usize,
) ![]f32 {
    const output = try allocator.alloc(f32, num_vectors);

    for (0..num_vectors) |idx| {
        const base = idx * dim;
        const vec = vectors[base..][0..dim];
        output[idx] = simdDotProduct(query[0..dim], vec);
    }

    return output;
}

const WARMUP = 3;
const ITERATIONS = 100;
const DIM = 384;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("SIMD vs GPU Tiered Dispatch Benchmark\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Vector dimension: {d}\n", .{DIM});
    std.debug.print("GPU threshold: 10,000 vectors\n", .{});
    std.debug.print("\n", .{});

    // Test different batch sizes to see crossover point
    const batch_sizes = [_]usize{ 100, 1_000, 5_000, 10_000, 50_000, 100_000 };

    std.debug.print("{s:<15} {s:>15} {s:>15} {s:>12}\n", .{ "Batch Size", "SIMD (ms)", "Per Vec (ns)", "Vecs/sec" });
    std.debug.print("{s:<15} {s:>15} {s:>15} {s:>12}\n", .{ "-" ** 15, "-" ** 15, "-" ** 15, "-" ** 12 });

    for (batch_sizes) |batch_size| {
        // Generate test data
        const query = try allocator.alloc(f32, DIM);
        defer allocator.free(query);
        const vectors = try allocator.alloc(f32, batch_size * DIM);
        defer allocator.free(vectors);

        var rng = std.Random.DefaultPrng.init(42);
        for (query) |*q| q.* = rng.random().float(f32) * 2.0 - 1.0;
        for (vectors) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

        // Warmup
        for (0..WARMUP) |_| {
            const result = try simdBatchDotProduct(allocator, query, vectors, batch_size, DIM);
            defer allocator.free(result);
            std.mem.doNotOptimizeAway(result);
        }

        // Benchmark SIMD
        var total_ns: u64 = 0;
        for (0..ITERATIONS) |_| {
            var timer = try std.time.Timer.start();
            const result = try simdBatchDotProduct(allocator, query, vectors, batch_size, DIM);
            defer allocator.free(result);
            std.mem.doNotOptimizeAway(result);
            total_ns += timer.read();
        }
        const avg_ns = total_ns / ITERATIONS;
        const ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const per_vec = @as(f64, @floatFromInt(avg_ns)) / @as(f64, @floatFromInt(batch_size));
        const vecs_per_sec = @as(f64, @floatFromInt(batch_size)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);

        std.debug.print("{d:<15} {d:>12.2} ms {d:>12.0} ns {d:>10.0}K\n", .{
            batch_size, ms, per_vec, vecs_per_sec / 1000.0,
        });
    }

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Note: GPU dispatch not yet integrated. This shows SIMD-only baseline.\n", .{});
    std.debug.print("GPU would be beneficial for batches >= 10K vectors.\n", .{});
    std.debug.print("================================================================================\n", .{});
}
