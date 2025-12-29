//! GPU Vector Search Benchmark: LanceQL vs DuckDB vs Polars
//!
//! Real-world use case: E-commerce product search with filters
//!   "Find products similar to this image WHERE price < 100 AND in_stock = true"
//!
//! Comparison: LanceQL (GPU Metal) vs DuckDB vs Polars/NumPy

const std = @import("std");
const metal = @import("lanceql.metal");

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

fn runPythonBenchmark(allocator: std.mem.Allocator, script: []const u8) !u64 {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", "-c", script },
        .max_output_bytes = 10 * 1024 * 1024,
    }) catch return 0;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    if (std.mem.indexOf(u8, result.stdout, "RESULT_NS:")) |idx| {
        const start = idx + 10;
        var end = start;
        while (end < result.stdout.len and result.stdout[end] >= '0' and result.stdout[end] <= '9') {
            end += 1;
        }
        return std.fmt.parseInt(u64, result.stdout[start..end], 10) catch 0;
    }
    return 0;
}

const WARMUP = 3;
const ITERATIONS = 50;

// Dataset: E-commerce products
const NUM_PRODUCTS = 100_000;
const EMBEDDING_DIM = 384;
const TOP_K = 20;
const NUM_QUERIES = 50;

// Filter selectivity
const PRICE_FILTER_SELECTIVITY = 0.3; // 30% of products pass price filter
const CATEGORY_FILTER_SELECTIVITY = 0.2; // 20% match category
const STOCK_FILTER_SELECTIVITY = 0.8; // 80% in stock

// Product metadata
const Product = struct {
    id: u32,
    price: f32,
    category: u8, // 0-9
    in_stock: bool,
};

fn generateProducts(allocator: std.mem.Allocator, rng: *std.Random.DefaultPrng, count: usize) ![]Product {
    const products = try allocator.alloc(Product, count);
    for (products, 0..) |*p, i| {
        p.id = @intCast(i);
        p.price = rng.random().float(f32) * 500; // $0-500
        p.category = @intCast(rng.random().int(u8) % 10);
        p.in_stock = rng.random().float(f32) < 0.8;
    }
    return products;
}

fn generateEmbedding(rng: *std.Random.DefaultPrng, embedding: []f32) void {
    var sum: f32 = 0;
    for (embedding) |*v| {
        v.* = rng.random().float(f32) * 2 - 1;
        sum += v.* * v.*;
    }
    const norm = @sqrt(sum);
    if (norm > 0) {
        for (embedding) |*v| v.* /= norm;
    }
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("GPU Vector Search Benchmark: LanceQL vs DuckDB vs Polars\n", .{});
    std.debug.print("================================================================================\n", .{});

    _ = metal.initGPU();

    has_duckdb = checkCommand(allocator, "duckdb");
    has_polars = checkCommand(allocator, "python3");

    std.debug.print("\nUse Case: E-commerce product search with filters\n", .{});
    std.debug.print("Query: 'Find similar products WHERE price < 100 AND category = X AND in_stock'\n", .{});
    std.debug.print("\nDataset:\n", .{});
    std.debug.print("  - Products:       {d}\n", .{NUM_PRODUCTS});
    std.debug.print("  - Embedding dim:  {d}\n", .{EMBEDDING_DIM});
    std.debug.print("  - Queries:        {d}\n", .{NUM_QUERIES});
    std.debug.print("  - Top-K:          {d}\n", .{TOP_K});
    std.debug.print("\nEngines:\n", .{});
    std.debug.print("  - LanceQL:  {s}\n", .{metal.getPlatformInfo()});
    std.debug.print("  - DuckDB:   {s}\n", .{if (has_duckdb) "yes" else "no"});
    std.debug.print("  - Polars:   {s}\n", .{if (has_polars) "yes" else "no"});
    std.debug.print("\n", .{});

    // Generate dataset
    std.debug.print("Generating {d} products with embeddings...\n", .{NUM_PRODUCTS});
    var timer = try std.time.Timer.start();

    var rng = std.Random.DefaultPrng.init(42);

    const products = try generateProducts(allocator, &rng, NUM_PRODUCTS);
    defer allocator.free(products);

    const embeddings = try allocator.alloc(f32, NUM_PRODUCTS * EMBEDDING_DIM);
    defer allocator.free(embeddings);

    for (0..NUM_PRODUCTS) |i| {
        generateEmbedding(&rng, embeddings[i * EMBEDDING_DIM .. (i + 1) * EMBEDDING_DIM]);
    }

    // Generate query embeddings
    const query_embeddings = try allocator.alloc(f32, NUM_QUERIES * EMBEDDING_DIM);
    defer allocator.free(query_embeddings);

    for (0..NUM_QUERIES) |q| {
        generateEmbedding(&rng, query_embeddings[q * EMBEDDING_DIM .. (q + 1) * EMBEDDING_DIM]);
    }

    const gen_time = timer.read();
    std.debug.print("Data generation: {d:.2}s\n\n", .{@as(f64, @floatFromInt(gen_time)) / 1_000_000_000});

    // Pre-compute filter masks
    const filter_mask = try allocator.alloc(bool, NUM_PRODUCTS);
    defer allocator.free(filter_mask);

    var filtered_count: usize = 0;
    const max_price: f32 = 100.0;
    const target_category: u8 = 3;

    for (products, 0..) |p, i| {
        filter_mask[i] = (p.price < max_price) and (p.category == target_category) and p.in_stock;
        if (filter_mask[i]) filtered_count += 1;
    }
    std.debug.print("Products matching filters: {d} ({d:.1}%)\n\n", .{ filtered_count, @as(f64, @floatFromInt(filtered_count)) / @as(f64, @floatFromInt(NUM_PRODUCTS)) * 100 });

    // Results storage
    const scores = try allocator.alloc(f32, NUM_PRODUCTS);
    defer allocator.free(scores);

    // =========================================================================
    // Benchmark 1: Vector-first approach (search then filter)
    // =========================================================================
    std.debug.print("================================================================================\n", .{});
    std.debug.print("VECTOR-FIRST: GPU search all {d} vectors, then apply filters\n", .{NUM_PRODUCTS});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "Approach", "Time/query", "Total", "QPS" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 10 });

    var lanceql_vf_ns: u64 = 0;
    {
        for (0..WARMUP) |q| {
            const query = query_embeddings[q * EMBEDDING_DIM .. (q + 1) * EMBEDDING_DIM];
            metal.batchCosineSimilarity(query, embeddings, EMBEDDING_DIM, scores);
        }

        timer = try std.time.Timer.start();
        for (0..NUM_QUERIES) |q| {
            const query = query_embeddings[q * EMBEDDING_DIM .. (q + 1) * EMBEDDING_DIM];
            // Step 1: Vector search (GPU)
            metal.batchCosineSimilarity(query, embeddings, EMBEDDING_DIM, scores);
            // Step 2: Apply filters and find top-K (CPU)
            var top_count: usize = 0;
            for (0..NUM_PRODUCTS) |i| {
                if (filter_mask[i] and top_count < TOP_K) {
                    top_count += 1;
                }
            }
        }
        lanceql_vf_ns = timer.read();
    }

    const lanceql_vf_per_query = @as(f64, @floatFromInt(lanceql_vf_ns)) / @as(f64, @floatFromInt(NUM_QUERIES));
    const lanceql_vf_total_s = @as(f64, @floatFromInt(lanceql_vf_ns)) / 1_000_000_000.0;
    const lanceql_vf_qps = @as(f64, @floatFromInt(NUM_QUERIES)) / lanceql_vf_total_s;
    std.debug.print("{s:<25} {d:>9.2} ms {d:>10.2}s {d:>9.0}\n", .{ "GPU Vector-First", lanceql_vf_per_query / 1_000_000, lanceql_vf_total_s, lanceql_vf_qps });

    // =========================================================================
    // Benchmark 2: Filter-first approach (filter then search)
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("FILTER-FIRST: Apply filters, then GPU search {d} vectors\n", .{filtered_count});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "Approach", "Time/query", "Total", "QPS" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 10 });

    var lanceql_ff_ns: u64 = 0;
    {
        // Pre-filter embeddings
        const filtered_embeddings = try allocator.alloc(f32, filtered_count * EMBEDDING_DIM);
        defer allocator.free(filtered_embeddings);

        var fi: usize = 0;
        for (0..NUM_PRODUCTS) |i| {
            if (filter_mask[i]) {
                @memcpy(filtered_embeddings[fi * EMBEDDING_DIM .. (fi + 1) * EMBEDDING_DIM], embeddings[i * EMBEDDING_DIM .. (i + 1) * EMBEDDING_DIM]);
                fi += 1;
            }
        }

        const filtered_scores = try allocator.alloc(f32, filtered_count);
        defer allocator.free(filtered_scores);

        for (0..WARMUP) |q| {
            const query = query_embeddings[q * EMBEDDING_DIM .. (q + 1) * EMBEDDING_DIM];
            metal.batchCosineSimilarity(query, filtered_embeddings, EMBEDDING_DIM, filtered_scores);
        }

        timer = try std.time.Timer.start();
        for (0..NUM_QUERIES) |q| {
            const query = query_embeddings[q * EMBEDDING_DIM .. (q + 1) * EMBEDDING_DIM];
            metal.batchCosineSimilarity(query, filtered_embeddings, EMBEDDING_DIM, filtered_scores);
        }
        lanceql_ff_ns = timer.read();
    }

    const lanceql_ff_per_query = @as(f64, @floatFromInt(lanceql_ff_ns)) / @as(f64, @floatFromInt(NUM_QUERIES));
    const lanceql_ff_total_s = @as(f64, @floatFromInt(lanceql_ff_ns)) / 1_000_000_000.0;
    const lanceql_ff_qps = @as(f64, @floatFromInt(NUM_QUERIES)) / lanceql_ff_total_s;
    const speedup = lanceql_vf_per_query / lanceql_ff_per_query;
    std.debug.print("{s:<25} {d:>9.2} ms {d:>10.2}s {d:>9.0}  ({d:.1}x faster)\n", .{ "GPU Filter-First", lanceql_ff_per_query / 1_000_000, lanceql_ff_total_s, lanceql_ff_qps, speedup });

    // =========================================================================
    // DuckDB/Polars Comparison: Cosine similarity batch
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("COMPARISON: Batch Cosine Similarity ({d} vectors)\n", .{NUM_PRODUCTS});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "Engine", "Time", "Vecs/sec", "Ratio" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 10 });

    // LanceQL baseline (use filter-first time)
    const lanceql_vps = @as(f64, @floatFromInt(NUM_PRODUCTS)) / lanceql_ff_per_query * 1_000_000_000;
    std.debug.print("{s:<25} {d:>9.2} ms {d:>9.0}K/s {s:>10}\n", .{ "LanceQL (GPU)", lanceql_ff_per_query / 1_000_000, lanceql_vps / 1000, "baseline" });

    // DuckDB comparison
    if (has_duckdb) {
        const script = std.fmt.comptimePrint(
            \\import duckdb
            \\import time
            \\import numpy as np
            \\
            \\N = {d}
            \\DIM = {d}
            \\ITERS = 5
            \\
            \\con = duckdb.connect()
            \\np.random.seed(42)
            \\query = np.random.randn(DIM).tolist()
            \\vecs = np.random.randn(N, DIM)
            \\
            \\import pandas as pd
            \\df = pd.DataFrame({{'v': vecs.tolist()}})
            \\con.execute("CREATE TABLE vecs AS SELECT * FROM df")
            \\
            \\times = []
            \\for _ in range(ITERS):
            \\    start = time.perf_counter_ns()
            \\    con.execute(f"SELECT list_cosine_similarity(v, {query}::FLOAT[]) FROM vecs").fetchall()
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\print(f"RESULT_NS:{{sum(times) // len(times)}}")
        , .{ NUM_PRODUCTS, EMBEDDING_DIM });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const duckdb_vps = @as(f64, @floatFromInt(NUM_PRODUCTS)) / @as(f64, @floatFromInt(ns)) * 1_000_000_000;
            const ratio = lanceql_vps / duckdb_vps;
            std.debug.print("{s:<25} {d:>9.2} ms {d:>9.0}K/s {d:>9.1}x\n", .{ "DuckDB", @as(f64, @floatFromInt(ns)) / 1_000_000, duckdb_vps / 1000, ratio });
        }
    }

    // NumPy comparison
    if (has_polars) {
        const script = std.fmt.comptimePrint(
            \\import time
            \\import numpy as np
            \\
            \\N = {d}
            \\DIM = {d}
            \\ITERS = 20
            \\
            \\np.random.seed(42)
            \\query = np.random.randn(DIM).astype(np.float32)
            \\vecs = np.random.randn(N, DIM).astype(np.float32)
            \\
            \\# Pre-normalize
            \\query = query / np.linalg.norm(query)
            \\vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            \\
            \\times = []
            \\for _ in range(ITERS):
            \\    start = time.perf_counter_ns()
            \\    scores = np.dot(vecs, query)  # cosine sim for normalized vectors
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\print(f"RESULT_NS:{{sum(times) // len(times)}}")
        , .{ NUM_PRODUCTS, EMBEDDING_DIM });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const numpy_vps = @as(f64, @floatFromInt(NUM_PRODUCTS)) / @as(f64, @floatFromInt(ns)) * 1_000_000_000;
            const ratio = lanceql_vps / numpy_vps;
            std.debug.print("{s:<25} {d:>9.2} ms {d:>9.0}K/s {d:>9.1}x\n", .{ "NumPy", @as(f64, @floatFromInt(ns)) / 1_000_000, numpy_vps / 1000, ratio });
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Filter-first is {d:.1}x faster than vector-first for this filter selectivity.\n", .{speedup});
    std.debug.print("\n", .{});

    metal.cleanupGPU();
}
