//! @logic_table Benchmark: Compiled Native Function Performance
//!
//! Tests REAL metal0-compiled @logic_table functions from lib/vector_ops.a
//!
//! Workflow:
//!   1. metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a
//!   2. zig build bench-logic-table
//!
//! The Python source (benchmarks/vector_ops.py):
//!   @logic_table
//!   class VectorOps:
//!       def dot_product(self, a: list, b: list) -> float: ...
//!       def sum_squares(self, a: list) -> float: ...
//!       def sum_values(self, a: list) -> float: ...
//!
//! NOTE: This benchmark measures compiled @logic_table function performance.
//! For SQL clause benchmarks comparing LanceQL vs DuckDB vs Polars, see bench_sql.sh

const std = @import("std");

const WARMUP = 5;
const ITERATIONS = 1_000_000;

// =============================================================================
// REAL extern declarations - from lib/vector_ops.a (metal0 compiled)
// =============================================================================

extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) f64;
extern fn VectorOps_sum_squares(a: [*]const f64, len: usize) f64;
extern fn VectorOps_sum_values(a: [*]const f64, len: usize) f64;

// =============================================================================
// Benchmark
// =============================================================================

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("@logic_table Benchmark: Compiled Native Function Performance\n", .{});
    std.debug.print("================================================================================\n", .{});

    std.debug.print("\nSource: lib/vector_ops.a (metal0 compiled from Python @logic_table)\n", .{});
    std.debug.print("\n", .{});

    const dim: usize = 384;

    std.debug.print("Vector dimension: {}\n", .{dim});
    std.debug.print("Warmup: {}, Iterations: {}M\n", .{ WARMUP, ITERATIONS / 1_000_000 });
    std.debug.print("\n", .{});

    const a = try allocator.alloc(f64, dim);
    defer allocator.free(a);
    const b = try allocator.alloc(f64, dim);
    defer allocator.free(b);

    // Initialize with deterministic data
    var rng = std.Random.DefaultPrng.init(42);
    for (a) |*v| v.* = rng.random().float(f64) * 2 - 1;
    for (b) |*v| v.* = rng.random().float(f64) * 2 - 1;

    // =========================================================================
    // Dot Product Benchmark
    // =========================================================================
    std.debug.print("================================================================================\n", .{});
    std.debug.print("DOT PRODUCT (384-dim vectors)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>15}\n", .{ "Function", "Time/op", "Total", "Throughput" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>15}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 15 });

    var lanceql_ns: u64 = 0;
    {
        var checksum: f64 = 0;
        for (0..WARMUP) |_| checksum += VectorOps_dot_product(a.ptr, b.ptr, dim);
        var timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| checksum += VectorOps_dot_product(a.ptr, b.ptr, dim);
        lanceql_ns = timer.read();
        std.mem.doNotOptimizeAway(&checksum);
    }
    const dot_per_op = @as(f64, @floatFromInt(lanceql_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const dot_total_s = @as(f64, @floatFromInt(lanceql_ns)) / 1_000_000_000.0;
    const dot_tput = @as(f64, @floatFromInt(ITERATIONS)) / dot_total_s / 1_000_000;
    std.debug.print("{s:<25} {d:>9.0} ns {d:>10.2}s {d:>12.1}M ops/s\n", .{ "dot_product", dot_per_op, dot_total_s, dot_tput });

    // =========================================================================
    // Sum Squares Benchmark
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("SUM SQUARES (384-dim vectors)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>15}\n", .{ "Function", "Time/op", "Total", "Throughput" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>15}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 15 });

    {
        var checksum: f64 = 0;
        for (0..WARMUP) |_| checksum += VectorOps_sum_squares(a.ptr, dim);
        var timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| checksum += VectorOps_sum_squares(a.ptr, dim);
        lanceql_ns = timer.read();
        std.mem.doNotOptimizeAway(&checksum);
    }
    const ss_per_op = @as(f64, @floatFromInt(lanceql_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const ss_total_s = @as(f64, @floatFromInt(lanceql_ns)) / 1_000_000_000.0;
    const ss_tput = @as(f64, @floatFromInt(ITERATIONS)) / ss_total_s / 1_000_000;
    std.debug.print("{s:<25} {d:>9.0} ns {d:>10.2}s {d:>12.1}M ops/s\n", .{ "sum_squares", ss_per_op, ss_total_s, ss_tput });

    // =========================================================================
    // Sum Values Benchmark
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("SUM VALUES (384-dim vectors)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>15}\n", .{ "Function", "Time/op", "Total", "Throughput" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>15}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 15 });

    {
        var checksum: f64 = 0;
        for (0..WARMUP) |_| checksum += VectorOps_sum_values(a.ptr, dim);
        var timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| checksum += VectorOps_sum_values(a.ptr, dim);
        lanceql_ns = timer.read();
        std.mem.doNotOptimizeAway(&checksum);
    }
    const sv_per_op = @as(f64, @floatFromInt(lanceql_ns)) / @as(f64, @floatFromInt(ITERATIONS));
    const sv_total_s = @as(f64, @floatFromInt(lanceql_ns)) / 1_000_000_000.0;
    const sv_tput = @as(f64, @floatFromInt(ITERATIONS)) / sv_total_s / 1_000_000;
    std.debug.print("{s:<25} {d:>9.0} ns {d:>10.2}s {d:>12.1}M ops/s\n", .{ "sum_values", sv_per_op, sv_total_s, sv_tput });

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("@logic_table compiled function throughput:\n", .{});
    std.debug.print("  - dot_product:  {d:.1}M ops/s ({d:.0} ns/op)\n", .{ dot_tput, dot_per_op });
    std.debug.print("  - sum_squares:  {d:.1}M ops/s ({d:.0} ns/op)\n", .{ ss_tput, ss_per_op });
    std.debug.print("  - sum_values:   {d:.1}M ops/s ({d:.0} ns/op)\n", .{ sv_tput, sv_per_op });
    std.debug.print("\n", .{});
    std.debug.print("These are native function calls with ~500ns overhead per call.\n", .{});
    std.debug.print("For SQL clause benchmarks (LanceQL vs DuckDB vs Polars), run:\n", .{});
    std.debug.print("  ./scripts/bench-sql.sh\n", .{});
    std.debug.print("\n", .{});
}
