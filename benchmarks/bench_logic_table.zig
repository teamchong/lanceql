//! @logic_table Benchmark: REAL metal0 compiled Python
//!
//! This benchmark uses ACTUAL metal0-compiled @logic_table functions.
//! No hand-written Zig pretending to be compiled output.
//!
//! Workflow:
//!   1. metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a
//!   2. zig build bench-logic-table
//!
//! The Python source (benchmarks/vector_ops.py):
//!   @logic_table
//!   class VectorOps:
//!       def dot_product(self, a: list, b: list) -> float:
//!           result = 0.0
//!           for i in range(len(a)):
//!               result = result + a[i] * b[i]
//!           return result
//!
//! This is compiled by metal0 to native Zig and linked as lib/vector_ops.a

const std = @import("std");

const WARMUP = 5;
const ITERATIONS = 100_000;

// =============================================================================
// REAL extern declarations - these come from lib/vector_ops.a
// Compiled by: metal0 build --emit-logic-table benchmarks/vector_ops.py
// =============================================================================

extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) f64;
extern fn VectorOps_sum_squares(a: [*]const f64, len: usize) f64;
extern fn VectorOps_sum_values(a: [*]const f64, len: usize) f64;

// =============================================================================
// Native Zig baseline for comparison (what we WANT metal0 to generate)
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

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("@logic_table Benchmark - REAL metal0 compiled Python\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Source: benchmarks/vector_ops.py (compiled by metal0)\n", .{});
    std.debug.print("Library: lib/vector_ops.a\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Comparing:\n", .{});
    std.debug.print("  - Compiled @logic_table (metal0 output)\n", .{});
    std.debug.print("  - Native Zig (optimization target)\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Warmup: {}, Iterations: {}\n", .{ WARMUP, ITERATIONS });
    std.debug.print("\n", .{});

    const dims = [_]usize{ 128, 384, 1024 };

    for (dims) |dim| {
        std.debug.print("================================================================================\n", .{});
        std.debug.print("Vector dimension: {}\n", .{dim});
        std.debug.print("================================================================================\n", .{});

        const a = try allocator.alloc(f64, dim);
        defer allocator.free(a);
        const b = try allocator.alloc(f64, dim);
        defer allocator.free(b);

        // Initialize with deterministic data
        var rng = std.Random.DefaultPrng.init(42);
        for (a) |*v| v.* = rng.random().float(f64) * 2 - 1;
        for (b) |*v| v.* = rng.random().float(f64) * 2 - 1;

        // Verify correctness
        std.debug.print("\nCorrectness check:\n", .{});
        const compiled_dot = VectorOps_dot_product(a.ptr, b.ptr, dim);
        const native_dot = nativeDotProduct(a, b);
        const dot_match = @abs(compiled_dot - native_dot) < 1e-10;
        std.debug.print("  dot_product: compiled={d:.6}, native={d:.6} {s}\n", .{
            compiled_dot,
            native_dot,
            if (dot_match) "✓" else "✗",
        });

        const compiled_sum_sq = VectorOps_sum_squares(a.ptr, dim);
        const native_sum_sq = nativeSumSquares(a);
        const sum_sq_match = @abs(compiled_sum_sq - native_sum_sq) < 1e-10;
        std.debug.print("  sum_squares: compiled={d:.6}, native={d:.6} {s}\n", .{
            compiled_sum_sq,
            native_sum_sq,
            if (sum_sq_match) "✓" else "✗",
        });

        const compiled_sum = VectorOps_sum_values(a.ptr, dim);
        const native_sum = nativeSumValues(a);
        const sum_match = @abs(compiled_sum - native_sum) < 1e-10;
        std.debug.print("  sum_values:  compiled={d:.6}, native={d:.6} {s}\n", .{
            compiled_sum,
            native_sum,
            if (sum_match) "✓" else "✗",
        });

        // Benchmark each function
        std.debug.print("\nPerformance ({} iterations):\n", .{ITERATIONS});
        std.debug.print("{s:<20} {s:>15} {s:>15} {s:>10}\n", .{ "Function", "Compiled", "Native", "Ratio" });
        std.debug.print("{s:<20} {s:>15} {s:>15} {s:>10}\n", .{ "-" ** 20, "-" ** 15, "-" ** 15, "-" ** 10 });

        // dot_product
        try benchmarkDotProduct("dot_product", a, b);

        // sum_squares
        try benchmarkSumSquares("sum_squares", a);

        // sum_values
        try benchmarkSumValues("sum_values", a);

        std.debug.print("\n", .{});
    }

    std.debug.print("================================================================================\n", .{});
    std.debug.print("Benchmark Complete\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("This is the REAL baseline. The ratio shows how much optimization is needed.\n", .{});
    std.debug.print("Goal: Get compiled @logic_table performance close to native Zig.\n", .{});
    std.debug.print("\n", .{});
}

fn benchmarkDotProduct(name: []const u8, a: []const f64, b: []const f64) !void {
    var checksum: f64 = 0;

    // Warmup compiled
    for (0..WARMUP) |_| checksum += VectorOps_dot_product(a.ptr, b.ptr, a.len);

    // Benchmark compiled
    var compiled_timer = try std.time.Timer.start();
    for (0..ITERATIONS) |_| {
        checksum += VectorOps_dot_product(a.ptr, b.ptr, a.len);
    }
    const compiled_ns = compiled_timer.read();

    // Warmup native
    for (0..WARMUP) |_| checksum += nativeDotProduct(a, b);

    // Benchmark native
    var native_timer = try std.time.Timer.start();
    for (0..ITERATIONS) |_| {
        checksum += nativeDotProduct(a, b);
    }
    const native_ns = native_timer.read();

    std.mem.doNotOptimizeAway(&checksum);

    const compiled_per_op = @as(f64, @floatFromInt(compiled_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const native_per_op = @as(f64, @floatFromInt(native_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const ratio = compiled_per_op / native_per_op;

    std.debug.print("{s:<20} {d:>12.1} ns {d:>12.1} ns {d:>9.1}x\n", .{
        name,
        compiled_per_op,
        native_per_op,
        ratio,
    });
}

fn benchmarkSumSquares(name: []const u8, a: []const f64) !void {
    var checksum: f64 = 0;

    // Warmup compiled
    for (0..WARMUP) |_| checksum += VectorOps_sum_squares(a.ptr, a.len);

    // Benchmark compiled
    var compiled_timer = try std.time.Timer.start();
    for (0..ITERATIONS) |_| {
        checksum += VectorOps_sum_squares(a.ptr, a.len);
    }
    const compiled_ns = compiled_timer.read();

    // Warmup native
    for (0..WARMUP) |_| checksum += nativeSumSquares(a);

    // Benchmark native
    var native_timer = try std.time.Timer.start();
    for (0..ITERATIONS) |_| {
        checksum += nativeSumSquares(a);
    }
    const native_ns = native_timer.read();

    std.mem.doNotOptimizeAway(&checksum);

    const compiled_per_op = @as(f64, @floatFromInt(compiled_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const native_per_op = @as(f64, @floatFromInt(native_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const ratio = compiled_per_op / native_per_op;

    std.debug.print("{s:<20} {d:>12.1} ns {d:>12.1} ns {d:>9.1}x\n", .{
        name,
        compiled_per_op,
        native_per_op,
        ratio,
    });
}

fn benchmarkSumValues(name: []const u8, a: []const f64) !void {
    var checksum: f64 = 0;

    // Warmup compiled
    for (0..WARMUP) |_| checksum += VectorOps_sum_values(a.ptr, a.len);

    // Benchmark compiled
    var compiled_timer = try std.time.Timer.start();
    for (0..ITERATIONS) |_| {
        checksum += VectorOps_sum_values(a.ptr, a.len);
    }
    const compiled_ns = compiled_timer.read();

    // Warmup native
    for (0..WARMUP) |_| checksum += nativeSumValues(a);

    // Benchmark native
    var native_timer = try std.time.Timer.start();
    for (0..ITERATIONS) |_| {
        checksum += nativeSumValues(a);
    }
    const native_ns = native_timer.read();

    std.mem.doNotOptimizeAway(&checksum);

    const compiled_per_op = @as(f64, @floatFromInt(compiled_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const native_per_op = @as(f64, @floatFromInt(native_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const ratio = compiled_per_op / native_per_op;

    std.debug.print("{s:<20} {d:>12.1} ns {d:>12.1} ns {d:>9.1}x\n", .{
        name,
        compiled_per_op,
        native_per_op,
        ratio,
    });
}
