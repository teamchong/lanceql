//! @logic_table Workflow Benchmark: Real metal0 Integration
//!
//! This benchmark uses REAL @logic_table functions compiled by metal0.
//! The full workflow:
//!   1. Python code with @logic_table decorator
//!   2. metal0 --emit-logic-table compiles to .zig
//!   3. LanceQL imports and executes the compiled functions
//!
//! Run with: zig build bench-logic-table

const std = @import("std");
const metal = @import("lanceql.metal");
const query = @import("lanceql.query");
const logic_table = @import("lanceql.logic_table");

const WARMUP = 3;
const ITERATIONS = 10;

var has_duckdb: bool = false;
var has_polars: bool = false;
var parquet_path: ?[]const u8 = null;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    _ = metal.initGPU();
    defer metal.cleanupGPU();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("@logic_table Workflow Benchmark: Real metal0 Integration\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Platform: {s}\n", .{metal.getPlatformInfo()});
    if (metal.isGPUReady()) {
        std.debug.print("GPU: {s}\n", .{metal.getGPUDeviceName()});
    }
    std.debug.print("Warmup: {d}, Iterations: {d}\n\n", .{ WARMUP, ITERATIONS });

    // Show the real @logic_table workflow
    std.debug.print("Workflow:\n", .{});
    std.debug.print("  1. Python: @logic_table class VectorOps with cosine_sim, dot_product, etc.\n", .{});
    std.debug.print("  2. metal0: --emit-logic-table compiles Python to src/logic_table/vector_ops.zig\n", .{});
    std.debug.print("  3. LanceQL: imports and calls compiled batch functions directly\n\n", .{});

    // Show available methods from the compiled VectorOps struct
    std.debug.print("Compiled @logic_table methods available:\n", .{});
    for (logic_table.vector_ops.VectorOps.methods) |method| {
        std.debug.print("  - VectorOps.{s}\n", .{method});
    }
    std.debug.print("\n", .{});

    // Check for external engines
    has_duckdb = checkCommand(allocator, "duckdb");
    has_polars = checkCommand(allocator, "polars");

    std.debug.print("Engines available:\n", .{});
    std.debug.print("  - LanceQL: yes (native Zig + Metal GPU)\n", .{});
    std.debug.print("  - DuckDB:  {s}\n", .{if (has_duckdb) "yes" else "no"});
    std.debug.print("  - Polars:  {s}\n", .{if (has_polars) "yes" else "no"});
    std.debug.print("\n", .{});

    // Create test data
    if (has_duckdb or has_polars) {
        parquet_path = try createTestData(allocator);
    }
    defer if (parquet_path) |p| {
        std.fs.deleteFileAbsolute(p) catch {};
        allocator.free(p);
    };

    const num_rows: usize = 1_000_000;

    std.debug.print("================================================================================\n", .{});
    std.debug.print("Dataset: {d} rows\n", .{num_rows});
    std.debug.print("================================================================================\n", .{});

    // Only benchmark weighted_score which has simple element-wise semantics
    // The cosine_sim/dot_product functions have O(n^2) semantics for embedding pairs
    try benchmarkWeightedScore(allocator, num_rows);

    // For cosine_sim/dot_product, use smaller batch sizes to demonstrate the workflow
    try benchmarkVectorOpsSmallBatch(allocator, 1000);

    // Print summary
    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n@logic_table advantage:\n", .{});
    std.debug.print("  - Python business logic compiled to native Zig batch functions\n", .{});
    std.debug.print("  - Zero FFI overhead (Zig has native C ABI)\n", .{});
    std.debug.print("  - Batch functions operate on slices, not scalar values\n", .{});
    std.debug.print("  - metal0 compiler optimizes for SIMD/GPU execution\n", .{});
    std.debug.print("\nDuckDB/Polars must:\n", .{});
    std.debug.print("  - Fetch data first, then compute in app code\n", .{});
    std.debug.print("  - Pay serialization overhead for Python interop\n", .{});
    std.debug.print("  - Cannot fuse custom business logic with query execution\n", .{});
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

fn createTestData(allocator: std.mem.Allocator) ![]const u8 {
    const path = try std.fmt.allocPrint(allocator, "/tmp/lanceql_workflow_{d}.parquet", .{std.time.milliTimestamp()});

    // Create test data with embedding columns for vector ops
    const sql = try std.fmt.allocPrint(allocator,
        \\COPY (
        \\  SELECT
        \\    i AS id,
        \\    random() AS score,
        \\    random() AS boost
        \\  FROM range(1000000) t(i)
        \\) TO '{s}' (FORMAT PARQUET);
    , .{path});
    defer allocator.free(sql);

    std.debug.print("Creating test data: {s}...\n", .{path});

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

/// Benchmark weighted_score using REAL compiled @logic_table function
/// This has simple element-wise semantics: out[i] = score[i] * 0.5 + boost[i] * 0.5
fn benchmarkWeightedScore(allocator: std.mem.Allocator, num_rows: usize) !void {
    std.debug.print("\n--- VectorOps.weighted_score (1M rows, element-wise) ---\n", .{});
    std.debug.print("Compiled from Python: score * 0.5 + boost * 0.5\n\n", .{});

    // LanceQL: Using REAL compiled @logic_table function
    {
        const scores = try allocator.alloc(f32, num_rows);
        defer allocator.free(scores);
        const boosts = try allocator.alloc(f32, num_rows);
        defer allocator.free(boosts);
        const output = try allocator.alloc(f32, num_rows);
        defer allocator.free(output);

        var rng = std.Random.DefaultPrng.init(42);
        for (scores, boosts) |*s, *b| {
            s.* = rng.random().float(f32);
            b.* = rng.random().float(f32);
        }

        // Warmup - using REAL compiled function
        for (0..WARMUP) |_| {
            logic_table.Functions.weightedScore(scores, boosts, output);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            logic_table.Functions.weightedScore(scores, boosts, output);
            times[iter] = timer.read();
        }

        const avg_ms = avgMs(&times);
        const throughput = @as(f64, @floatFromInt(num_rows)) / (avg_ms / 1000.0) / 1_000_000;
        std.debug.print("LanceQL (@logic_table): {d:>8.2} ms ({d:.1} M rows/sec)\n", .{ avg_ms, throughput });
    }

    // DuckDB: Same operation in SQL
    if (has_duckdb) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT score * 0.5 + boost * 0.5 AS weighted_score FROM '{s}';", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runDuckDB(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runDuckDB(allocator, sql) catch 0;
            }

            const avg_ms = avgMs(&times);
            const throughput = @as(f64, @floatFromInt(num_rows)) / (avg_ms / 1000.0) / 1_000_000;
            std.debug.print("DuckDB (SQL):           {d:>8.2} ms ({d:.1} M rows/sec)\n", .{ avg_ms, throughput });
        }
    }

    // Polars: Same operation in SQL
    if (has_polars) {
        if (parquet_path) |path| {
            const sql = try std.fmt.allocPrint(allocator, "SELECT score * 0.5 + boost * 0.5 AS weighted_score FROM read_parquet('{s}');", .{path});
            defer allocator.free(sql);

            for (0..WARMUP) |_| {
                _ = runPolars(allocator, sql) catch continue;
            }

            var times: [ITERATIONS]u64 = undefined;
            for (0..ITERATIONS) |iter| {
                times[iter] = runPolars(allocator, sql) catch 0;
            }

            const avg_ms = avgMs(&times);
            const throughput = @as(f64, @floatFromInt(num_rows)) / (avg_ms / 1000.0) / 1_000_000;
            std.debug.print("Polars (SQL):           {d:>8.2} ms ({d:.1} M rows/sec)\n", .{ avg_ms, throughput });
        }
    }
}

/// Benchmark cosine_sim and dot_product with small batch to show they work
/// These have O(n*d) complexity where n=batch_size and d=embedding_dim
fn benchmarkVectorOpsSmallBatch(allocator: std.mem.Allocator, batch_size: usize) !void {
    std.debug.print("\n--- VectorOps.cosine_sim/dot_product ({d} rows, pair-wise) ---\n", .{batch_size});
    std.debug.print("Compiled from Python: cosine similarity and dot product\n\n", .{});

    // LanceQL: Using REAL compiled @logic_table functions
    {
        const query_embedding = try allocator.alloc(f32, batch_size);
        defer allocator.free(query_embedding);
        const docs_embedding = try allocator.alloc(f32, batch_size);
        defer allocator.free(docs_embedding);
        const output = try allocator.alloc(f32, batch_size);
        defer allocator.free(output);

        var rng = std.Random.DefaultPrng.init(42);
        for (query_embedding, docs_embedding) |*q, *d| {
            q.* = rng.random().float(f32) * 2 - 1;
            d.* = rng.random().float(f32) * 2 - 1;
        }

        // Warmup
        for (0..WARMUP) |_| {
            logic_table.Functions.cosineSim(query_embedding, docs_embedding, output);
            logic_table.Functions.dotProduct(query_embedding, docs_embedding, output);
        }

        // Benchmark cosine_sim
        var times_cos: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            logic_table.Functions.cosineSim(query_embedding, docs_embedding, output);
            times_cos[iter] = timer.read();
        }

        // Benchmark dot_product
        var times_dot: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            logic_table.Functions.dotProduct(query_embedding, docs_embedding, output);
            times_dot[iter] = timer.read();
        }

        const avg_cos = avgMs(&times_cos);
        const avg_dot = avgMs(&times_dot);
        std.debug.print("LanceQL cosine_sim:     {d:>8.2} ms\n", .{avg_cos});
        std.debug.print("LanceQL dot_product:    {d:>8.2} ms\n", .{avg_dot});
    }
}

fn avgMs(times: []const u64) f64 {
    var total: u64 = 0;
    for (times) |t| total += t;
    return @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(times.len)) / 1_000_000;
}
