//! Embedding Pipeline Benchmark: LanceQL vs DuckDB vs Polars
//!
//! Real-world use case: Text-to-vector conversion for semantic search
//!
//! Pipeline stages:
//!   1. Text chunking (split documents into passages)
//!   2. Tokenization (text to token IDs)
//!   3. Embedding generation (tokens to vectors)
//!   4. Normalization (L2 normalize vectors)
//!
//! Comparison: LanceQL native vs DuckDB vs Polars/NumPy

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
const ITERATIONS = 10;

// Pipeline parameters
const NUM_DOCUMENTS = 10_000;
const AVG_DOC_LENGTH = 2000; // characters
const CHUNK_SIZE = 512; // characters per chunk
const CHUNK_OVERLAP = 50;
const EMBEDDING_DIM = 384;
const MAX_TOKENS = 128;

// Generate random text document
fn generateDocument(allocator: std.mem.Allocator, rng: *std.Random.DefaultPrng, length: usize) ![]u8 {
    const words = [_][]const u8{
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "artificial", "intelligence", "neural", "network",
        "deep", "learning", "natural", "language", "processing", "computer",
        "vision", "transformer", "attention", "mechanism", "embedding", "vector",
        "semantic", "search", "retrieval", "augmented", "generation", "model",
    };

    var doc = std.ArrayListUnmanaged(u8){};
    errdefer doc.deinit(allocator);

    while (doc.items.len < length) {
        const word = words[rng.random().int(usize) % words.len];
        try doc.appendSlice(allocator, word);
        try doc.append(allocator, ' ');
    }

    return try doc.toOwnedSlice(allocator);
}

// Chunk document into overlapping passages
fn chunkDocument(allocator: std.mem.Allocator, doc: []const u8, chunk_size: usize, overlap: usize) ![][]const u8 {
    var chunks = std.ArrayListUnmanaged([]const u8){};
    errdefer chunks.deinit(allocator);

    var start: usize = 0;
    while (start < doc.len) {
        const end = @min(start + chunk_size, doc.len);
        const chunk = try allocator.dupe(u8, doc[start..end]);
        try chunks.append(allocator, chunk);

        if (end >= doc.len) break;
        start += chunk_size - overlap;
    }

    return try chunks.toOwnedSlice(allocator);
}

// Simple tokenization (word-based, maps to random token IDs)
fn tokenize(text: []const u8, tokens: []u16, rng: *std.Random.DefaultPrng) usize {
    var token_count: usize = 0;
    var in_word = false;

    for (text) |c| {
        if (c == ' ' or c == '\n' or c == '\t') {
            if (in_word and token_count < tokens.len) {
                tokens[token_count] = rng.random().int(u16) % 30000; // Vocab size
                token_count += 1;
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }

    if (in_word and token_count < tokens.len) {
        tokens[token_count] = rng.random().int(u16) % 30000;
        token_count += 1;
    }

    return token_count;
}

// Simulated embedding lookup (in real use, this would be neural network inference)
fn generateEmbedding(tokens: []const u16, token_count: usize, embedding: []f32, rng: *std.Random.DefaultPrng) void {
    // Initialize with zeros
    @memset(embedding, 0);

    // Simulate attention-weighted sum of token embeddings
    for (0..token_count) |t| {
        const token_id = tokens[t];
        // Deterministic "embedding" based on token ID
        for (embedding, 0..) |*e, d| {
            const hash = @as(u32, token_id) *% 2654435761 +% @as(u32, @intCast(d));
            e.* += @as(f32, @bitCast(hash)) * 0.0000001;
        }
    }

    // Add some randomness
    for (embedding) |*e| {
        e.* += rng.random().float(f32) * 0.01;
    }

    // L2 normalize
    var sum: f32 = 0;
    for (embedding) |e| sum += e * e;
    const norm = @sqrt(sum);
    if (norm > 0) {
        for (embedding) |*e| e.* /= norm;
    }
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Embedding Pipeline Benchmark: LanceQL vs DuckDB vs Polars\n", .{});
    std.debug.print("================================================================================\n", .{});

    _ = metal.initGPU();

    has_duckdb = checkCommand(allocator, "duckdb");
    has_polars = checkCommand(allocator, "python3");

    std.debug.print("\nUse Case: Text-to-vector conversion for semantic search\n", .{});
    std.debug.print("\nDataset:\n", .{});
    std.debug.print("  - Documents:      {d}\n", .{NUM_DOCUMENTS});
    std.debug.print("  - Avg length:     {d} chars\n", .{AVG_DOC_LENGTH});
    std.debug.print("  - Chunk size:     {d} chars\n", .{CHUNK_SIZE});
    std.debug.print("  - Embedding dim:  {d}\n", .{EMBEDDING_DIM});
    std.debug.print("\nEngines:\n", .{});
    std.debug.print("  - LanceQL:  {s}\n", .{metal.getPlatformInfo()});
    std.debug.print("  - DuckDB:   {s}\n", .{if (has_duckdb) "yes" else "no"});
    std.debug.print("  - Polars:   {s}\n", .{if (has_polars) "yes" else "no"});
    std.debug.print("\n", .{});

    var rng = std.Random.DefaultPrng.init(42);

    // =========================================================================
    // Benchmark 1: Text Chunking
    // =========================================================================
    std.debug.print("================================================================================\n", .{});
    std.debug.print("TEXT CHUNKING: {d} documents -> passages\n", .{NUM_DOCUMENTS});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>15}\n", .{ "Stage", "Time", "Throughput" });
    std.debug.print("{s:<25} {s:>12} {s:>15}\n", .{ "-" ** 25, "-" ** 12, "-" ** 15 });

    // Generate sample documents
    const docs = try allocator.alloc([]u8, NUM_DOCUMENTS);
    defer {
        for (docs) |doc| allocator.free(doc);
        allocator.free(docs);
    }

    for (docs) |*doc| {
        doc.* = try generateDocument(allocator, &rng, AVG_DOC_LENGTH);
    }

    // Chunking benchmark
    var lanceql_chunk_ns: u64 = 0;
    var total_chunks: usize = 0;
    {
        var timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| {
            for (docs) |doc| {
                const chunks = try chunkDocument(allocator, doc, CHUNK_SIZE, CHUNK_OVERLAP);
                total_chunks += chunks.len;
                for (chunks) |chunk| allocator.free(chunk);
                allocator.free(chunks);
            }
        }
        lanceql_chunk_ns = timer.read();
    }
    total_chunks /= ITERATIONS;

    const lanceql_chunk_s = @as(f64, @floatFromInt(lanceql_chunk_ns)) / @as(f64, @floatFromInt(ITERATIONS)) / 1_000_000_000.0;
    const lanceql_chunk_tput = @as(f64, @floatFromInt(NUM_DOCUMENTS)) / lanceql_chunk_s;
    std.debug.print("{s:<25} {d:>9.0} ms {d:>12.0} docs/s\n", .{ "Text Chunking", lanceql_chunk_s * 1000, lanceql_chunk_tput });
    std.debug.print("  -> Generated {d} chunks total\n\n", .{total_chunks});

    // =========================================================================
    // Benchmark 2: Tokenization + Embedding
    // =========================================================================
    std.debug.print("================================================================================\n", .{});
    std.debug.print("TOKENIZATION + EMBEDDING: {d} chunks -> vectors\n", .{total_chunks});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>15}\n", .{ "Stage", "Time", "Throughput" });
    std.debug.print("{s:<25} {s:>12} {s:>15}\n", .{ "-" ** 25, "-" ** 12, "-" ** 15 });

    // Tokenize + Embed
    var lanceql_embed_ns: u64 = 0;
    {
        const tokens = try allocator.alloc(u16, MAX_TOKENS);
        defer allocator.free(tokens);
        const embedding = try allocator.alloc(f32, EMBEDDING_DIM);
        defer allocator.free(embedding);

        var timer = try std.time.Timer.start();
        for (0..ITERATIONS) |_| {
            for (docs) |doc| {
                // Chunk
                const chunks = try chunkDocument(allocator, doc, CHUNK_SIZE, CHUNK_OVERLAP);
                defer {
                    for (chunks) |chunk| allocator.free(chunk);
                    allocator.free(chunks);
                }

                // Tokenize + Embed each chunk
                for (chunks) |chunk| {
                    const token_count = tokenize(chunk, tokens, &rng);
                    generateEmbedding(tokens, token_count, embedding, &rng);
                }
            }
        }
        lanceql_embed_ns = timer.read();
    }

    const lanceql_embed_s = @as(f64, @floatFromInt(lanceql_embed_ns)) / @as(f64, @floatFromInt(ITERATIONS)) / 1_000_000_000.0;
    const lanceql_embed_tput = @as(f64, @floatFromInt(total_chunks)) / lanceql_embed_s;
    std.debug.print("{s:<25} {d:>9.0} ms {d:>12.0} chunks/s\n", .{ "Tokenize + Embed", lanceql_embed_s * 1000, lanceql_embed_tput });

    // =========================================================================
    // Benchmark 3: Batch Embedding Normalization
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("L2 NORMALIZATION: {d} vectors\n", .{total_chunks});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>15}\n", .{ "Stage", "Time", "Throughput" });
    std.debug.print("{s:<25} {s:>12} {s:>15}\n", .{ "-" ** 25, "-" ** 12, "-" ** 15 });

    // Generate embeddings to normalize
    const embeddings = try allocator.alloc(f32, total_chunks * EMBEDDING_DIM);
    defer allocator.free(embeddings);

    for (0..total_chunks) |i| {
        for (0..EMBEDDING_DIM) |d| {
            embeddings[i * EMBEDDING_DIM + d] = rng.random().float(f32) * 2 - 1;
        }
    }

    // L2 normalize
    var lanceql_norm_ns: u64 = 0;
    {
        var timer = try std.time.Timer.start();
        for (0..ITERATIONS * 10) |_| {
            for (0..total_chunks) |i| {
                const vec = embeddings[i * EMBEDDING_DIM .. (i + 1) * EMBEDDING_DIM];
                var sum: f32 = 0;
                for (vec) |v| sum += v * v;
                const norm = @sqrt(sum);
                if (norm > 0) {
                    for (vec) |*v| v.* /= norm;
                }
            }
        }
        lanceql_norm_ns = timer.read();
    }

    const lanceql_norm_s = @as(f64, @floatFromInt(lanceql_norm_ns)) / @as(f64, @floatFromInt(ITERATIONS * 10)) / 1_000_000_000.0;
    const lanceql_norm_tput = @as(f64, @floatFromInt(total_chunks)) / lanceql_norm_s / 1_000_000;
    std.debug.print("{s:<25} {d:>9.2} ms {d:>12.1}M vecs/s {s:>10}\n", .{ "LanceQL", lanceql_norm_s * 1000, lanceql_norm_tput, "baseline" });

    // DuckDB L2 normalization comparison
    if (has_duckdb) {
        const script = std.fmt.comptimePrint(
            \\import duckdb
            \\import time
            \\import numpy as np
            \\
            \\CHUNKS = {d}
            \\DIM = {d}
            \\ITERS = 10
            \\
            \\con = duckdb.connect()
            \\np.random.seed(42)
            \\vecs = np.random.randn(CHUNKS, DIM).astype(np.float32)
            \\
            \\# Create table with vectors
            \\con.execute("CREATE TABLE vecs (v FLOAT[])")
            \\for v in vecs:
            \\    con.execute("INSERT INTO vecs VALUES (?)", [v.tolist()])
            \\
            \\times = []
            \\for _ in range(ITERS):
            \\    start = time.perf_counter_ns()
            \\    # L2 normalize using DuckDB array functions
            \\    con.execute("SELECT list_transform(v, x -> x / sqrt(list_sum(list_transform(v, y -> y * y)))) FROM vecs").fetchall()
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\print(f"RESULT_NS:{{sum(times) // len(times)}}")
        , .{ total_chunks, EMBEDDING_DIM });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const duckdb_s = @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
            const duckdb_tput = @as(f64, @floatFromInt(total_chunks)) / duckdb_s / 1_000_000;
            const ratio = lanceql_norm_tput / duckdb_tput;
            std.debug.print("{s:<25} {d:>9.2} ms {d:>12.1}M vecs/s {d:>9.1}x\n", .{ "DuckDB", duckdb_s * 1000, duckdb_tput, ratio });
        }
    }

    // Polars/NumPy L2 normalization comparison
    if (has_polars) {
        const script = std.fmt.comptimePrint(
            \\import time
            \\import numpy as np
            \\
            \\CHUNKS = {d}
            \\DIM = {d}
            \\ITERS = 100
            \\
            \\np.random.seed(42)
            \\vecs = np.random.randn(CHUNKS, DIM).astype(np.float32)
            \\
            \\times = []
            \\for _ in range(ITERS):
            \\    start = time.perf_counter_ns()
            \\    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            \\    normalized = vecs / norms
            \\    times.append(time.perf_counter_ns() - start)
            \\
            \\print(f"RESULT_NS:{{sum(times) // len(times)}}")
        , .{ total_chunks, EMBEDDING_DIM });

        const ns = try runPythonBenchmark(allocator, script);
        if (ns > 0) {
            const polars_s = @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
            const polars_tput = @as(f64, @floatFromInt(total_chunks)) / polars_s / 1_000_000;
            const ratio = lanceql_norm_tput / polars_tput;
            std.debug.print("{s:<25} {d:>9.2} ms {d:>12.1}M vecs/s {d:>9.1}x\n", .{ "NumPy", polars_s * 1000, polars_tput, ratio });
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("LanceQL Pipeline throughput:\n", .{});
    std.debug.print("  - Chunking:       {d:.0} docs/s\n", .{lanceql_chunk_tput});
    std.debug.print("  - Tokenize+Embed: {d:.0} chunks/s\n", .{lanceql_embed_tput});
    std.debug.print("  - Normalization:  {d:.1}M vectors/s\n", .{lanceql_norm_tput});
    std.debug.print("\n", .{});

    metal.cleanupGPU();
}
