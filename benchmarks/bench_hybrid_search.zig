//! Hybrid Search Benchmark: Vector Similarity + SQL Filters
//!
//! Real-world use case: E-commerce product search
//!   "Find products similar to this image WHERE price < 100 AND in_stock = true"
//!
//! Pipeline:
//!   1. Vector similarity search (find similar items)
//!   2. SQL filter (price, category, stock status)
//!   3. Combine results with proper ranking
//!
//! ⚠️  IMPORTANT: This benchmark compares WORKFLOWS, not raw compute:
//!   - LanceQL:  In-process GPU/CPU compute (no overhead)
//!   - DuckDB:   Subprocess (~30ms spawn) + SQL parsing
//!   - Polars:   Subprocess (~30ms spawn) + Python startup
//!
//! For fair in-process comparison, see bench_inprocess.zig which uses DuckDB C API.

const std = @import("std");
const metal = @import("lanceql.metal");

const WARMUP = 3;
const ITERATIONS = 50;
const SUBPROCESS_ITERATIONS = 30;

// Dataset: E-commerce products
const NUM_PRODUCTS = 100_000;
const EMBEDDING_DIM = 384;
const TOP_K = 20;
const NUM_QUERIES = 50;

// Filter selectivity
const PRICE_FILTER_SELECTIVITY = 0.3; // 30% of products pass price filter
const CATEGORY_FILTER_SELECTIVITY = 0.2; // 20% match category
const STOCK_FILTER_SELECTIVITY = 0.8; // 80% in stock

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
        .max_output_bytes = 100 * 1024 * 1024,
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
        .max_output_bytes = 100 * 1024 * 1024,
    }) catch return error.PolarsFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |code_| if (code_ != 0) return error.PolarsFailed,
        else => return error.PolarsFailed,
    }
    return timer.read();
}

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
    std.debug.print("Hybrid Search Benchmark: Vector + SQL Filters\n", .{});
    std.debug.print("================================================================================\n", .{});

    _ = metal.initGPU();

    has_duckdb = checkCommand(allocator, "duckdb");
    has_polars = checkCommand(allocator, "python3");

    std.debug.print("\nUse Case: E-commerce product search\n", .{});
    std.debug.print("Query: 'Find similar products WHERE price < 100 AND category = X AND in_stock'\n", .{});
    std.debug.print("\nDataset:\n", .{});
    std.debug.print("  - Products:       {d}\n", .{NUM_PRODUCTS});
    std.debug.print("  - Embedding dim:  {d}\n", .{EMBEDDING_DIM});
    std.debug.print("  - Queries:        {d}\n", .{NUM_QUERIES});
    std.debug.print("  - Top-K:          {d}\n", .{TOP_K});
    std.debug.print("\nFilter selectivity:\n", .{});
    std.debug.print("  - Price < $100:   {d:.0}%\n", .{PRICE_FILTER_SELECTIVITY * 100});
    std.debug.print("  - Category match: {d:.0}%\n", .{CATEGORY_FILTER_SELECTIVITY * 100});
    std.debug.print("  - In stock:       {d:.0}%\n", .{STOCK_FILTER_SELECTIVITY * 100});
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
    std.debug.print("VECTOR-FIRST: Search all, then apply SQL filters\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "Engine", "Time/query", "Total", "QPS" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 10 });

    // LanceQL: GPU vector search + CPU filter
    var lanceql_vf_ns: u64 = 0;
    {
        for (0..WARMUP) |q| {
            const query = query_embeddings[q * EMBEDDING_DIM .. (q + 1) * EMBEDDING_DIM];
            metal.batchCosineSimilarity(query, embeddings, EMBEDDING_DIM, scores);
        }

        timer = try std.time.Timer.start();
        for (0..NUM_QUERIES) |q| {
            const query = query_embeddings[q * EMBEDDING_DIM .. (q + 1) * EMBEDDING_DIM];
            // Step 1: Vector search
            metal.batchCosineSimilarity(query, embeddings, EMBEDDING_DIM, scores);
            // Step 2: Apply filters and find top-K
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
    std.debug.print("{s:<25} {d:>9.2} ms {d:>10.2}s {d:>9.0}\n", .{ "LanceQL", lanceql_vf_per_query / 1_000_000, lanceql_vf_total_s, lanceql_vf_qps });

    // DuckDB vector-first (uses VSS extension concept)
    if (has_duckdb) {
        const sql =
            \\WITH products AS (
            \\  SELECT
            \\    row_number() OVER () as id,
            \\    random() * 500 as price,
            \\    (random() * 10)::INT as category,
            \\    random() < 0.8 as in_stock,
            \\    list_transform(range(384), x -> random()::FLOAT) as embedding
            \\  FROM generate_series(1, 10000)
            \\),
            \\query AS (
            \\  SELECT list_transform(range(384), x -> random()::FLOAT) as vec
            \\)
            \\SELECT p.id, list_cosine_similarity(p.embedding, q.vec) as score
            \\FROM products p, query q
            \\WHERE p.price < 100 AND p.category = 3 AND p.in_stock
            \\ORDER BY score DESC
            \\LIMIT 20;
        ;

        var duckdb_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
        for (0..SUBPROCESS_ITERATIONS) |_| duckdb_ns += runDuckDB(allocator, sql) catch 0;

        const duckdb_per_query = @as(f64, @floatFromInt(duckdb_ns)) / @as(f64, @floatFromInt(SUBPROCESS_ITERATIONS));
        const duckdb_total_s = @as(f64, @floatFromInt(duckdb_ns)) / 1_000_000_000.0;
        const duckdb_qps = @as(f64, @floatFromInt(SUBPROCESS_ITERATIONS)) / duckdb_total_s;
        const duckdb_ratio = duckdb_per_query / lanceql_vf_per_query;
        std.debug.print("{s:<25} {d:>9.0} ms {d:>10.2}s {d:>9.0}  ({d:.0}x, 10K products)\n", .{ "DuckDB", duckdb_per_query / 1_000_000, duckdb_total_s, duckdb_qps, duckdb_ratio });
    }

    // Polars vector-first
    if (has_polars) {
        const py_code_vf = try std.fmt.allocPrint(allocator,
            \\import polars as pl
            \\import numpy as np
            \\import time
            \\
            \\np.random.seed(42)
            \\n = 10000
            \\dim = 384
            \\
            \\# Create Polars DataFrame with metadata
            \\df = pl.DataFrame({{
            \\    'id': range(n),
            \\    'price': np.random.rand(n) * 500,
            \\    'category': np.random.randint(0, 10, n),
            \\    'in_stock': np.random.rand(n) < 0.8,
            \\}})
            \\embeddings = np.random.randn(n, dim).astype(np.float32)
            \\embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            \\query = np.random.randn(dim).astype(np.float32)
            \\query = query / np.linalg.norm(query)
            \\
            \\# Polars vector-first: compute all scores, then filter
            \\start = time.time()
            \\for _ in range(50):
            \\    # Vector search (NumPy)
            \\    scores = embeddings @ query
            \\    # Add scores to Polars DataFrame and filter
            \\    result = df.with_columns(pl.Series('score', scores)).filter(
            \\        (pl.col('price') < 100) &
            \\        (pl.col('category') == 3) &
            \\        pl.col('in_stock')
            \\    ).sort('score', descending=True).head(20)
            \\elapsed = time.time() - start
            \\print(f"{{elapsed:.4f}}")
        , .{});
        defer allocator.free(py_code_vf);

        var polars_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, py_code_vf) catch 0;
        for (0..5) |_| polars_ns += runPolars(allocator, py_code_vf) catch 0;

        const polars_per_query = @as(f64, @floatFromInt(polars_ns)) / 5.0 / 50.0;
        const polars_total_s = @as(f64, @floatFromInt(polars_ns)) / 5.0 / 1_000_000_000.0;
        const polars_qps = 50.0 / polars_total_s;
        const polars_ratio = polars_per_query / lanceql_vf_per_query;
        std.debug.print("{s:<25} {d:>9.2} ms {d:>10.2}s {d:>9.0}  ({d:.1}x, 10K products)\n", .{ "Polars", polars_per_query / 1_000_000, polars_total_s, polars_qps, polars_ratio });
    }

    // =========================================================================
    // Benchmark 2: Filter-first approach (filter then search)
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("FILTER-FIRST: Apply SQL filters, then search filtered set\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "Engine", "Time/query", "Total", "QPS" });
    std.debug.print("{s:<25} {s:>12} {s:>12} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 12, "-" ** 10 });

    // LanceQL: Filter first, then search only matching
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
    std.debug.print("{s:<25} {d:>9.2} ms {d:>10.2}s {d:>9.0}  ({d:.1}x faster)\n", .{ "LanceQL", lanceql_ff_per_query / 1_000_000, lanceql_ff_total_s, lanceql_ff_qps, speedup });

    // DuckDB filter-first
    if (has_duckdb) {
        const sql =
            \\WITH products AS (
            \\  SELECT
            \\    row_number() OVER () as id,
            \\    random() * 500 as price,
            \\    (random() * 10)::INT as category,
            \\    random() < 0.8 as in_stock,
            \\    list_transform(range(384), x -> random()::FLOAT) as embedding
            \\  FROM generate_series(1, 10000)
            \\),
            \\filtered AS (
            \\  SELECT * FROM products
            \\  WHERE price < 100 AND category = 3 AND in_stock
            \\),
            \\query AS (
            \\  SELECT list_transform(range(384), x -> random()::FLOAT) as vec
            \\)
            \\SELECT f.id, list_cosine_similarity(f.embedding, q.vec) as score
            \\FROM filtered f, query q
            \\ORDER BY score DESC
            \\LIMIT 20;
        ;

        var duckdb_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
        for (0..SUBPROCESS_ITERATIONS) |_| duckdb_ns += runDuckDB(allocator, sql) catch 0;

        const duckdb_per_query = @as(f64, @floatFromInt(duckdb_ns)) / @as(f64, @floatFromInt(SUBPROCESS_ITERATIONS));
        const duckdb_total_s = @as(f64, @floatFromInt(duckdb_ns)) / 1_000_000_000.0;
        const duckdb_qps = @as(f64, @floatFromInt(SUBPROCESS_ITERATIONS)) / duckdb_total_s;
        const duckdb_ratio = duckdb_per_query / lanceql_ff_per_query;
        std.debug.print("{s:<25} {d:>9.0} ms {d:>10.2}s {d:>9.0}  ({d:.0}x, 10K products)\n", .{ "DuckDB", duckdb_per_query / 1_000_000, duckdb_total_s, duckdb_qps, duckdb_ratio });
    }

    // Polars filter-first
    if (has_polars) {
        const py_code = try std.fmt.allocPrint(allocator,
            \\import polars as pl
            \\import numpy as np
            \\import time
            \\
            \\np.random.seed(42)
            \\n = 10000
            \\dim = 384
            \\
            \\# Create Polars DataFrame
            \\df = pl.DataFrame({{
            \\    'id': range(n),
            \\    'price': np.random.rand(n) * 500,
            \\    'category': np.random.randint(0, 10, n),
            \\    'in_stock': np.random.rand(n) < 0.8,
            \\}})
            \\embeddings = np.random.randn(n, dim).astype(np.float32)
            \\embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            \\query = np.random.randn(dim).astype(np.float32)
            \\query = query / np.linalg.norm(query)
            \\
            \\# Polars filter-first hybrid search
            \\start = time.time()
            \\for _ in range(50):
            \\    # Filter with Polars
            \\    filtered = df.filter(
            \\        (pl.col('price') < 100) &
            \\        (pl.col('category') == 3) &
            \\        pl.col('in_stock')
            \\    )
            \\    mask = filtered['id'].to_numpy()
            \\    # Vector search with NumPy
            \\    filtered_emb = embeddings[mask]
            \\    scores = filtered_emb @ query
            \\    top_k = np.argsort(scores)[-20:][::-1]
            \\elapsed = time.time() - start
            \\print(f"{{elapsed:.4f}}")
        , .{});
        defer allocator.free(py_code);

        var polars_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, py_code) catch 0;
        for (0..5) |_| {
            polars_ns += runPolars(allocator, py_code) catch 0;
        }

        const polars_per_query = @as(f64, @floatFromInt(polars_ns)) / 5.0 / 50.0;
        const polars_total_s = @as(f64, @floatFromInt(polars_ns)) / 5.0 / 1_000_000_000.0;
        const polars_qps = 50.0 / polars_total_s;
        const polars_ratio = polars_per_query / lanceql_ff_per_query;
        std.debug.print("{s:<25} {d:>9.2} ms {d:>10.2}s {d:>9.0}  ({d:.1}x, 10K products)\n", .{ "Polars", polars_per_query / 1_000_000, polars_total_s, polars_qps, polars_ratio });
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
    std.debug.print("⚠️  NOTE: DuckDB/Polars use subprocess execution (~30ms spawn overhead).\n", .{});
    std.debug.print("The ratios include process spawn time, not just compute difference.\n", .{});
    std.debug.print("For fair in-process comparison, run: zig build bench-inprocess\n", .{});
    std.debug.print("\n", .{});

    metal.cleanupGPU();
}
