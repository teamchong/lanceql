//! Embedding Pipeline Benchmark
//!
//! Real-world use case: Text-to-vector conversion for semantic search
//!
//! Pipeline stages:
//!   1. Text chunking (split documents into passages)
//!   2. Tokenization (text to token IDs)
//!   3. Embedding generation (tokens to vectors)
//!   4. Normalization (L2 normalize vectors)
//!
//! Comparison: LanceQL vs sentence-transformers (Python)

const std = @import("std");
const metal = @import("lanceql.metal");

const WARMUP = 3;
const ITERATIONS = 10;
const SUBPROCESS_ITERATIONS = 10;

// Pipeline parameters
const NUM_DOCUMENTS = 10_000;
const AVG_DOC_LENGTH = 2000; // characters
const CHUNK_SIZE = 512; // characters per chunk
const CHUNK_OVERLAP = 50;
const EMBEDDING_DIM = 384;
const MAX_TOKENS = 128;

var has_python: bool = false;
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

fn runPython(allocator: std.mem.Allocator, code: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", "-c", code },
        .max_output_bytes = 100 * 1024 * 1024,
    }) catch return error.PythonFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |code_| if (code_ != 0) return error.PythonFailed,
        else => return error.PythonFailed,
    }
    return timer.read();
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
    has_python = checkCommand(allocator, "python3");
    has_duckdb = checkCommand(allocator, "duckdb");
    has_polars = has_python; // Polars uses Python

    std.debug.print("\nUse Case: Text-to-vector conversion for semantic search\n", .{});
    std.debug.print("\nPipeline:\n", .{});
    std.debug.print("  Documents → Chunking → Tokenization → Embedding → Normalization\n", .{});
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
    std.debug.print("TEXT CHUNKING: {d} documents → passages\n", .{NUM_DOCUMENTS});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>15} {s:>10}\n", .{ "Engine", "Time", "Throughput", "Ratio" });
    std.debug.print("{s:<25} {s:>12} {s:>15} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 15, "-" ** 10 });

    // Generate sample documents
    const docs = try allocator.alloc([]u8, NUM_DOCUMENTS);
    defer {
        for (docs) |doc| allocator.free(doc);
        allocator.free(docs);
    }

    for (docs) |*doc| {
        doc.* = try generateDocument(allocator, &rng, AVG_DOC_LENGTH);
    }

    // LanceQL: Chunking
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
    std.debug.print("{s:<25} {d:>9.0} ms {d:>12.0} docs/s {s:>10}\n", .{ "LanceQL", lanceql_chunk_s * 1000, lanceql_chunk_tput, "1.0x" });
    std.debug.print("  → Generated {d} chunks total\n\n", .{total_chunks});

    // DuckDB: Text chunking (using string split)
    if (has_duckdb) {
        const sql =
            \\WITH docs AS (
            \\  SELECT
            \\    i as id,
            \\    repeat('the quick brown fox machine learning neural network ', 25) as text
            \\  FROM generate_series(1, 1000) t(i)
            \\)
            \\SELECT id, unnest(regexp_split_to_array(text, '.{512}')) as chunk
            \\FROM docs;
        ;

        var duckdb_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
        for (0..10) |_| duckdb_ns += runDuckDB(allocator, sql) catch 0;

        const duckdb_s = @as(f64, @floatFromInt(duckdb_ns)) / 10.0 / 1_000_000_000.0;
        const duckdb_tput = 1000.0 / duckdb_s;
        const duckdb_ratio = duckdb_s / lanceql_chunk_s;
        std.debug.print("{s:<25} {d:>9.0} ms {d:>12.0} docs/s {d:>9.1}x (1K docs)\n", .{ "DuckDB", duckdb_s * 1000, duckdb_tput, duckdb_ratio });
    }

    // Polars: Chunking
    if (has_polars) {
        const py_code = try std.fmt.allocPrint(allocator,
            \\import polars as pl
            \\import time
            \\import random
            \\
            \\random.seed(42)
            \\words = ['the', 'quick', 'brown', 'fox', 'machine', 'learning', 'neural', 'network']
            \\docs = [' '.join(random.choices(words, k=250)) for _ in range({d})]
            \\df = pl.DataFrame({{'text': docs}})
            \\
            \\def chunk_text(text, chunk_size=512, overlap=50):
            \\    chunks = []
            \\    start = 0
            \\    while start < len(text):
            \\        end = min(start + chunk_size, len(text))
            \\        chunks.append(text[start:end])
            \\        if end >= len(text):
            \\            break
            \\        start += chunk_size - overlap
            \\    return chunks
            \\
            \\start = time.time()
            \\for _ in range({d}):
            \\    result = df.with_columns([
            \\        pl.col('text').map_elements(chunk_text, return_dtype=pl.List(pl.Utf8)).alias('chunks')
            \\    ])
            \\elapsed = time.time() - start
            \\print(f"{{elapsed:.4f}}")
        , .{ NUM_DOCUMENTS, ITERATIONS });
        defer allocator.free(py_code);

        var polars_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, py_code) catch 0;
        polars_ns = runPolars(allocator, py_code) catch 0;

        const polars_s = @as(f64, @floatFromInt(polars_ns)) / @as(f64, @floatFromInt(ITERATIONS)) / 1_000_000_000.0;
        const polars_tput = @as(f64, @floatFromInt(NUM_DOCUMENTS)) / polars_s;
        const polars_ratio = polars_s / lanceql_chunk_s;
        std.debug.print("{s:<25} {d:>9.0} ms {d:>12.0} docs/s {d:>9.1}x\n", .{ "Polars", polars_s * 1000, polars_tput, polars_ratio });
    }

    // =========================================================================
    // Benchmark 2: Tokenization + Embedding
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("TOKENIZATION + EMBEDDING: {d} chunks → vectors\n", .{total_chunks});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>15} {s:>10}\n", .{ "Engine", "Time", "Throughput", "Ratio" });
    std.debug.print("{s:<25} {s:>12} {s:>15} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 15, "-" ** 10 });

    // LanceQL: Tokenize + Embed
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
    std.debug.print("{s:<25} {d:>9.0} ms {d:>12.0} chunks/s {s:>10}\n", .{ "LanceQL", lanceql_embed_s * 1000, lanceql_embed_tput, "1.0x" });

    // =========================================================================
    // Benchmark 3: Batch Embedding Normalization
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("L2 NORMALIZATION: {d} vectors\n", .{total_chunks});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("{s:<25} {s:>12} {s:>15} {s:>10}\n", .{ "Engine", "Time", "Throughput", "Ratio" });
    std.debug.print("{s:<25} {s:>12} {s:>15} {s:>10}\n", .{ "-" ** 25, "-" ** 12, "-" ** 15, "-" ** 10 });

    // Generate embeddings to normalize
    const embeddings = try allocator.alloc(f32, total_chunks * EMBEDDING_DIM);
    defer allocator.free(embeddings);

    for (0..total_chunks) |i| {
        for (0..EMBEDDING_DIM) |d| {
            embeddings[i * EMBEDDING_DIM + d] = rng.random().float(f32) * 2 - 1;
        }
    }

    // LanceQL: L2 normalize
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
    std.debug.print("{s:<25} {d:>9.2} ms {d:>12.1}M vecs/s {s:>10}\n", .{ "LanceQL", lanceql_norm_s * 1000, lanceql_norm_tput, "1.0x" });

    // DuckDB: L2 normalize (list operations)
    if (has_duckdb) {
        const sql =
            \\WITH vectors AS (
            \\  SELECT list_transform(range(384), x -> random()::FLOAT) as vec
            \\  FROM generate_series(1, 10000)
            \\)
            \\SELECT list_transform(vec, x -> x / sqrt(list_sum(list_transform(vec, v -> v*v)))) as normalized
            \\FROM vectors;
        ;

        var duckdb_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runDuckDB(allocator, sql) catch 0;
        for (0..10) |_| duckdb_ns += runDuckDB(allocator, sql) catch 0;

        const duckdb_s = @as(f64, @floatFromInt(duckdb_ns)) / 10.0 / 1_000_000_000.0;
        const duckdb_tput = 10000.0 / duckdb_s / 1_000_000;
        const duckdb_ratio = duckdb_s / lanceql_norm_s;
        std.debug.print("{s:<25} {d:>9.2} ms {d:>12.1}M vecs/s {d:>9.1}x (10K vecs)\n", .{ "DuckDB", duckdb_s * 1000, duckdb_tput, duckdb_ratio });
    }

    // Polars: L2 normalize
    if (has_polars) {
        const py_code = try std.fmt.allocPrint(allocator,
            \\import polars as pl
            \\import numpy as np
            \\import time
            \\np.random.seed(42)
            \\embeddings = np.random.randn({d}, {d}).astype(np.float32)
            \\df = pl.DataFrame({{'vec': embeddings.tolist()}})
            \\start = time.time()
            \\for _ in range(100):
            \\    # Polars doesn't have native vector ops, use NumPy
            \\    vecs = np.array(df['vec'].to_list())
            \\    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            \\    normalized = vecs / norms
            \\elapsed = time.time() - start
            \\print(f"{{elapsed:.4f}}")
        , .{ total_chunks, EMBEDDING_DIM });
        defer allocator.free(py_code);

        var polars_ns: u64 = 0;
        for (0..WARMUP) |_| _ = runPolars(allocator, py_code) catch 0;
        polars_ns = runPolars(allocator, py_code) catch 0;

        const polars_s = @as(f64, @floatFromInt(polars_ns)) / 100.0 / 1_000_000_000.0;
        const polars_tput = @as(f64, @floatFromInt(total_chunks)) / polars_s / 1_000_000;
        const polars_ratio = polars_s / lanceql_norm_s;
        std.debug.print("{s:<25} {d:>9.2} ms {d:>12.1}M vecs/s {d:>9.1}x\n", .{ "Polars", polars_s * 1000, polars_tput, polars_ratio });
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("LanceQL provides end-to-end embedding pipeline without Python overhead.\n", .{});
    std.debug.print("For production, integrate with ONNX runtime for neural network inference.\n", .{});
    std.debug.print("\n", .{});

    metal.cleanupGPU();
}
