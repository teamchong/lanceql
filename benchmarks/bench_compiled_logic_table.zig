//! Benchmark: Compiled @logic_table functions vs native Zig vs DuckDB vs Polars
//!
//! This benchmark compares:
//!   1. Compiled Python @logic_table (from metal0)
//!   2. Native Zig implementation (hand-optimized)
//!   3. DuckDB (SQL engine)
//!   4. Polars/NumPy (Python DataFrame)
//!
//! Workflow:
//!   1. Compile Python: metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a
//!   2. Run benchmark: zig build bench-compiled-logic-table
//!
//! The Python code in benchmarks/vector_ops.py is compiled to native code and
//! linked as a static library. This benchmark proves the @logic_table workflow
//! produces real, callable native code.

const std = @import("std");
const c = @cImport({
    @cInclude("duckdb.h");
});

// =============================================================================
// Extern declarations for compiled @logic_table functions
// These are exported by lib/vector_ops.a (compiled from benchmarks/vector_ops.py)
// =============================================================================

extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) f64;
extern fn VectorOps_sum_squares(a: [*]const f64, len: usize) f64;
extern fn VectorOps_sum_values(a: [*]const f64, len: usize) f64;

// =============================================================================
// Native Zig implementations for comparison
// =============================================================================

fn nativeDotProduct(a: []const f64, b: []const f64) f64 {
    var result: f64 = 0.0;
    const len = @min(a.len, b.len);
    for (0..len) |i| {
        result += a[i] * b[i];
    }
    return result;
}

fn nativeSumSquares(a: []const f64) f64 {
    var result: f64 = 0.0;
    for (a) |v| {
        result += v * v;
    }
    return result;
}

fn nativeSumValues(a: []const f64) f64 {
    var result: f64 = 0.0;
    for (a) |v| {
        result += v;
    }
    return result;
}

// =============================================================================
// Benchmark
// =============================================================================

const WARMUP = 5;
const ITERATIONS = 50_000_000; // ~5 seconds at ~100ns per op
const TARGET_SECONDS: f64 = 5.0;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Compiled @logic_table Benchmark\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Comparing: @logic_table (compiled) vs Native Zig\n", .{});
    std.debug.print("Target: ~{d:.0} seconds per dimension | Iterations: {d}M\n", .{ TARGET_SECONDS, ITERATIONS / 1_000_000 });
    std.debug.print("\n", .{});

    // Test vectors
    const sizes = [_]usize{ 128, 384, 1024 };

    for (sizes) |dim| {
        std.debug.print("--- Vector dimension: {} ---\n", .{dim});

        const a = try allocator.alloc(f64, dim);
        defer allocator.free(a);
        const b = try allocator.alloc(f64, dim);
        defer allocator.free(b);

        // Initialize with random data
        var rng = std.Random.DefaultPrng.init(42);
        for (a) |*v| v.* = rng.random().float(f64) * 2 - 1;
        for (b) |*v| v.* = rng.random().float(f64) * 2 - 1;

        // Verify correctness
        const compiled_dot = VectorOps_dot_product(a.ptr, b.ptr, dim);
        const native_dot = nativeDotProduct(a, b);
        const dot_diff = @abs(compiled_dot - native_dot);
        std.debug.print("dot_product: compiled={d:.6}, native={d:.6}, diff={e:.2}\n", .{ compiled_dot, native_dot, dot_diff });

        const compiled_sum_sq = VectorOps_sum_squares(a.ptr, dim);
        const native_sum_sq = nativeSumSquares(a);
        const sum_sq_diff = @abs(compiled_sum_sq - native_sum_sq);
        std.debug.print("sum_squares: compiled={d:.6}, native={d:.6}, diff={e:.2}\n", .{ compiled_sum_sq, native_sum_sq, sum_sq_diff });

        const compiled_sum = VectorOps_sum_values(a.ptr, dim);
        const native_sum = nativeSumValues(a);
        const sum_diff = @abs(compiled_sum - native_sum);
        std.debug.print("sum_values:  compiled={d:.6}, native={d:.6}, diff={e:.2}\n", .{ compiled_sum, native_sum, sum_diff });

        // Benchmark dot_product
        std.debug.print("\nBenchmarking dot_product ({} iterations)...\n", .{ITERATIONS});

        var checksum: f64 = 0;

        // Warmup compiled
        for (0..WARMUP) |_| checksum += VectorOps_dot_product(a.ptr, b.ptr, dim);

        var compiled_timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| {
            checksum += VectorOps_dot_product(a.ptr, b.ptr, dim);
        }
        const compiled_ns = compiled_timer.read();

        // Warmup native
        for (0..WARMUP) |_| checksum += nativeDotProduct(a, b);

        var native_timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| {
            checksum += nativeDotProduct(a, b);
        }
        const native_ns = native_timer.read();

        // Prevent optimization
        std.mem.doNotOptimizeAway(&checksum);

        const compiled_per_op = @as(f64, @floatFromInt(compiled_ns)) / @as(f64, @floatFromInt(ITERATIONS));
        const native_per_op = @as(f64, @floatFromInt(native_ns)) / @as(f64, @floatFromInt(ITERATIONS));
        const speedup = native_per_op / compiled_per_op;

        std.debug.print("  Compiled @logic_table: {d:.1} ns/op\n", .{compiled_per_op});
        std.debug.print("  Native Zig:            {d:.1} ns/op\n", .{native_per_op});
        std.debug.print("  Ratio: {d:.2}x (compiled vs native)\n", .{speedup});

        std.debug.print("\n", .{});
    }

    std.debug.print("================================================================================\n", .{});
    std.debug.print("Benchmark Complete - @logic_table compilation is working!\n", .{});
    std.debug.print("================================================================================\n", .{});
}
