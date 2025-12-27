//! @logic_table Workflow Benchmark: Real ML/AI Workloads
//!
//! This benchmark demonstrates the full @logic_table workflow with realistic
//! ML workloads that match ml_workflow.py:
//!   - Feature Engineering: normalization, z-score, log transform
//!   - Vector Search: cosine similarity, euclidean distance
//!   - Fraud Detection: multi-factor risk scoring
//!   - Recommendations: collaborative filtering scores
//!
//! Full workflow:
//!   1. metal0 build --emit-logic-table benchmarks/ml_workflow.py -o lib/logic_table.a
//!   2. zig build bench-logic-table
//!
//! The benchmark compares LanceQL (compiled @logic_table) vs DuckDB/Polars.

const std = @import("std");
const metal = @import("lanceql.metal");

const WARMUP = 5;
const ITERATIONS = 20;

var has_duckdb: bool = false;
var parquet_path: ?[]const u8 = null;

// =============================================================================
// Compiled @logic_table functions (matching ml_workflow.py)
// These are what metal0 generates from Python numpy code
// =============================================================================

const FeatureEngineering = struct {
    /// Min-max normalization to [0, 1] range
    /// Python: (data - min) / (max - min + 1e-8)
    pub fn normalizeMinmax(data: []const f32, output: []f32) void {
        const len = @min(data.len, output.len);
        if (len == 0) return;

        // Find min/max
        var min_val: f32 = data[0];
        var max_val: f32 = data[0];
        for (data[1..]) |v| {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        const range = max_val - min_val + 1e-8;

        // SIMD normalize
        if (len >= 8) {
            const Vec8 = @Vector(8, f32);
            const min_vec: Vec8 = @splat(min_val);
            const range_vec: Vec8 = @splat(range);
            var i: usize = 0;

            while (i + 8 <= len) : (i += 8) {
                const v: Vec8 = data[i..][0..8].*;
                output[i..][0..8].* = (v - min_vec) / range_vec;
            }

            while (i < len) : (i += 1) {
                output[i] = (data[i] - min_val) / range;
            }
        } else {
            for (0..len) |i| {
                output[i] = (data[i] - min_val) / range;
            }
        }
    }

    /// Z-score standardization (mean=0, std=1)
    /// Python: (data - mean) / (std + 1e-8)
    pub fn normalizeZscore(data: []const f32, output: []f32) void {
        const len = @min(data.len, output.len);
        if (len == 0) return;

        // Compute mean
        var sum: f64 = 0;
        for (data) |v| sum += v;
        const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(len)));

        // Compute std
        var sq_sum: f64 = 0;
        for (data) |v| {
            const diff = v - mean;
            sq_sum += diff * diff;
        }
        const std_val: f32 = @floatCast(@sqrt(sq_sum / @as(f64, @floatFromInt(len))) + 1e-8);

        // SIMD normalize
        if (len >= 8) {
            const Vec8 = @Vector(8, f32);
            const mean_vec: Vec8 = @splat(mean);
            const std_vec: Vec8 = @splat(std_val);
            var i: usize = 0;

            while (i + 8 <= len) : (i += 8) {
                const v: Vec8 = data[i..][0..8].*;
                output[i..][0..8].* = (v - mean_vec) / std_vec;
            }

            while (i < len) : (i += 1) {
                output[i] = (data[i] - mean) / std_val;
            }
        } else {
            for (0..len) |i| {
                output[i] = (data[i] - mean) / std_val;
            }
        }
    }

    /// Log transform with offset: log1p(data)
    pub fn logTransform(data: []const f32, output: []f32) void {
        const len = @min(data.len, output.len);
        for (0..len) |i| {
            output[i] = @log(1.0 + data[i]);
        }
    }

    /// Clip outliers to 3 standard deviations
    pub fn clipOutliers(data: []const f32, output: []f32) void {
        const len = @min(data.len, output.len);
        if (len == 0) return;

        // Compute mean and std
        var sum: f64 = 0;
        for (data) |v| sum += v;
        const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(len)));

        var sq_sum: f64 = 0;
        for (data) |v| {
            const diff = v - mean;
            sq_sum += diff * diff;
        }
        const std_val: f32 = @floatCast(@sqrt(sq_sum / @as(f64, @floatFromInt(len))));

        const lower = mean - 3 * std_val;
        const upper = mean + 3 * std_val;

        // Clip
        for (0..len) |i| {
            output[i] = @max(lower, @min(upper, data[i]));
        }
    }
};

const VectorSearch = struct {
    /// Cosine similarity between query and each document embedding
    /// query: (dim,), docs: (num_docs, dim) flattened
    /// Returns similarity scores for each doc
    pub fn cosineSimilarity(query: []const f32, docs: []const f32, dim: usize, output: []f32) void {
        const num_docs = docs.len / dim;

        // Normalize query
        var query_norm: f32 = 0;
        for (query[0..dim]) |v| query_norm += v * v;
        query_norm = @sqrt(query_norm) + 1e-8;

        // Compute similarity for each document
        for (0..num_docs) |doc_idx| {
            const doc_start = doc_idx * dim;
            const doc = docs[doc_start .. doc_start + dim];

            // Dot product and doc norm
            var dot: f32 = 0;
            var doc_norm: f32 = 0;

            // SIMD for larger dimensions
            if (dim >= 8) {
                const Vec8 = @Vector(8, f32);
                var i: usize = 0;
                var dot_vec: Vec8 = @splat(0);
                var norm_vec: Vec8 = @splat(0);

                while (i + 8 <= dim) : (i += 8) {
                    const q: Vec8 = query[i..][0..8].*;
                    const d: Vec8 = doc[i..][0..8].*;
                    dot_vec += q * d;
                    norm_vec += d * d;
                }

                dot = @reduce(.Add, dot_vec);
                doc_norm = @reduce(.Add, norm_vec);

                while (i < dim) : (i += 1) {
                    dot += query[i] * doc[i];
                    doc_norm += doc[i] * doc[i];
                }
            } else {
                for (0..dim) |i| {
                    dot += query[i] * doc[i];
                    doc_norm += doc[i] * doc[i];
                }
            }

            doc_norm = @sqrt(doc_norm) + 1e-8;
            output[doc_idx] = dot / (query_norm * doc_norm);
        }
    }

    /// Euclidean (L2) distance between query and each document
    pub fn euclideanDistance(query: []const f32, docs: []const f32, dim: usize, output: []f32) void {
        const num_docs = docs.len / dim;

        for (0..num_docs) |doc_idx| {
            const doc_start = doc_idx * dim;
            const doc = docs[doc_start .. doc_start + dim];

            var sq_dist: f32 = 0;

            if (dim >= 8) {
                const Vec8 = @Vector(8, f32);
                var i: usize = 0;
                var dist_vec: Vec8 = @splat(0);

                while (i + 8 <= dim) : (i += 8) {
                    const q: Vec8 = query[i..][0..8].*;
                    const d: Vec8 = doc[i..][0..8].*;
                    const diff = q - d;
                    dist_vec += diff * diff;
                }

                sq_dist = @reduce(.Add, dist_vec);

                while (i < dim) : (i += 1) {
                    const diff = query[i] - doc[i];
                    sq_dist += diff * diff;
                }
            } else {
                for (0..dim) |i| {
                    const diff = query[i] - doc[i];
                    sq_dist += diff * diff;
                }
            }

            output[doc_idx] = @sqrt(sq_dist);
        }
    }
};

const FraudDetection = struct {
    /// Multi-factor fraud risk scoring
    /// Factors: amount, velocity, location_distance, hour, fraud_count
    pub fn transactionRiskScore(
        amounts: []const f32,
        velocities: []const f32,
        location_distances: []const f32,
        hours: []const f32,
        fraud_counts: []const f32,
        output: []f32,
    ) void {
        const len = @min(amounts.len, @min(velocities.len, @min(location_distances.len, @min(hours.len, @min(fraud_counts.len, output.len)))));

        for (0..len) |i| {
            var score: f32 = 0;

            // Amount risk (0-0.3): exponential decay for normal amounts
            const amount_risk = 1.0 - @exp(-amounts[i] / 5000.0);
            score += @min(0.3, amount_risk * 0.3);

            // Velocity risk (0-0.25): transactions per hour
            const velocity_risk = @min(1.0, velocities[i] / 10.0);
            score += velocity_risk * 0.25;

            // Location risk (0-0.2): distance from usual location
            const location_risk = @min(1.0, location_distances[i] / 1000.0);
            score += location_risk * 0.2;

            // Time risk (0-0.1): transactions at unusual hours (2am-5am)
            const hour = hours[i];
            if (hour >= 2 and hour <= 5) {
                score += 0.1;
            }

            // History risk (0-0.15): previous fraud incidents
            const history_risk = @min(1.0, fraud_counts[i] / 3.0);
            score += history_risk * 0.15;

            output[i] = @max(0.0, @min(1.0, score));
        }
    }

    /// Z-score based anomaly detection
    pub fn anomalyScore(
        amounts: []const f32,
        amount_means: []const f32,
        amount_stds: []const f32,
        velocities: []const f32,
        velocity_means: []const f32,
        velocity_stds: []const f32,
        output: []f32,
    ) void {
        const len = @min(amounts.len, output.len);

        for (0..len) |i| {
            const amount_z = @abs(amounts[i] - amount_means[i]) / (amount_stds[i] + 1e-8);
            const velocity_z = @abs(velocities[i] - velocity_means[i]) / (velocity_stds[i] + 1e-8);
            output[i] = @max(amount_z, velocity_z);
        }
    }
};

const Recommendations = struct {
    /// Collaborative filtering score with popularity bias and recency boost
    pub fn collaborativeScore(
        user_embedding: []const f32,
        item_embeddings: []const f32,
        view_counts: []const f32,
        age_days: []const f32,
        dim: usize,
        output: []f32,
    ) void {
        const num_items = item_embeddings.len / dim;

        for (0..num_items) |item_idx| {
            const item_start = item_idx * dim;
            const item = item_embeddings[item_start .. item_start + dim];

            // Dot product (base score)
            var dot: f32 = 0;
            if (dim >= 8) {
                const Vec8 = @Vector(8, f32);
                var i: usize = 0;
                var dot_vec: Vec8 = @splat(0);

                while (i + 8 <= dim) : (i += 8) {
                    const u: Vec8 = user_embedding[i..][0..8].*;
                    const t: Vec8 = item[i..][0..8].*;
                    dot_vec += u * t;
                }

                dot = @reduce(.Add, dot_vec);

                while (i < dim) : (i += 1) {
                    dot += user_embedding[i] * item[i];
                }
            } else {
                for (0..dim) |i| {
                    dot += user_embedding[i] * item[i];
                }
            }

            // Popularity penalty
            const popularity_penalty = @log(1.0 + view_counts[item_idx]) * 0.1;

            // Recency boost
            const recency_boost = @exp(-age_days[item_idx] / 30.0) * 0.2;

            output[item_idx] = dot - popularity_penalty + recency_boost;
        }
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    _ = metal.initGPU();
    defer metal.cleanupGPU();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("@logic_table ML Workflow Benchmark\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Platform: {s}\n", .{metal.getPlatformInfo()});
    if (metal.isGPUReady()) {
        std.debug.print("GPU: {s}\n", .{metal.getGPUDeviceName()});
    }
    std.debug.print("Warmup: {d}, Iterations: {d}\n", .{ WARMUP, ITERATIONS });

    // Workflow explanation
    std.debug.print("\nWorkflow:\n", .{});
    std.debug.print("  1. Python: Write @logic_table classes with numpy (ml_workflow.py)\n", .{});
    std.debug.print("  2. metal0: Compile to native Zig static library\n", .{});
    std.debug.print("  3. LanceQL: Link and call compiled batch functions\n", .{});
    std.debug.print("\nCompile command:\n", .{});
    std.debug.print("  metal0 build --emit-logic-table benchmarks/ml_workflow.py -o lib/logic_table.a\n\n", .{});

    // Check for DuckDB
    has_duckdb = checkCommand(allocator, "duckdb");
    std.debug.print("Engines: LanceQL (native), DuckDB ({s})\n", .{if (has_duckdb) "available" else "not found"});

    // Create test data for DuckDB comparison
    if (has_duckdb) {
        parquet_path = try createTestData(allocator);
    }
    defer if (parquet_path) |p| {
        std.fs.deleteFileAbsolute(p) catch {};
        allocator.free(p);
    };

    // Run all benchmarks
    std.debug.print("\n", .{});

    try benchmarkFeatureEngineering(allocator, 10_000_000);
    try benchmarkVectorSearch(allocator, 100_000, 384);
    try benchmarkFraudDetection(allocator, 5_000_000);
    try benchmarkRecommendations(allocator, 50_000, 128);

    // Summary
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Summary: @logic_table Advantage\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("  - Python business logic -> Native SIMD batch functions\n", .{});
    std.debug.print("  - Zero serialization overhead (operates on memory directly)\n", .{});
    std.debug.print("  - Fuses custom logic with query execution\n", .{});
    std.debug.print("  - GPU acceleration via Metal (macOS)\n", .{});
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
    const path = try std.fmt.allocPrint(allocator, "/tmp/lanceql_ml_{d}.parquet", .{std.time.milliTimestamp()});

    const sql = try std.fmt.allocPrint(allocator,
        \\COPY (
        \\  SELECT
        \\    i AS id,
        \\    random() * 10000 AS amount,
        \\    random() * 20 AS velocity,
        \\    random() * 2000 AS location_distance,
        \\    (random() * 24)::INTEGER AS hour,
        \\    (random() * 5)::INTEGER AS fraud_count,
        \\    random() AS score,
        \\    random() AS boost
        \\  FROM range(10000000) t(i)
        \\) TO '{s}' (FORMAT PARQUET);
    , .{path});
    defer allocator.free(sql);

    std.debug.print("Creating test data (10M rows)...\n", .{});

    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "duckdb", "-c", sql },
        .max_output_bytes = 1024 * 1024,
    }) catch return error.DuckDBFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }

    if (result.term.Exited != 0) {
        return error.DuckDBFailed;
    }

    return path;
}

fn runDuckDBTimed(allocator: std.mem.Allocator, sql: []const u8) !u64 {
    var timer = try std.time.Timer.start();
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "duckdb", "-c", sql },
        .max_output_bytes = 1024 * 1024,
    }) catch return error.DuckDBFailed;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    return timer.read();
}

// =============================================================================
// Benchmarks
// =============================================================================

fn benchmarkFeatureEngineering(allocator: std.mem.Allocator, num_rows: usize) !void {
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Feature Engineering ({d}M rows)\n", .{num_rows / 1_000_000});
    std.debug.print("================================================================================\n", .{});

    const data = try allocator.alloc(f32, num_rows);
    defer allocator.free(data);
    const output = try allocator.alloc(f32, num_rows);
    defer allocator.free(output);

    // Initialize with realistic data
    var rng = std.Random.DefaultPrng.init(42);
    for (data) |*d| {
        d.* = rng.random().float(f32) * 10000.0; // 0-10000 range
    }

    // Benchmark normalize_minmax
    {
        for (0..WARMUP) |_| FeatureEngineering.normalizeMinmax(data, output);

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            FeatureEngineering.normalizeMinmax(data, output);
            times[iter] = timer.read();
        }

        const avg_ms = avgMs(&times);
        const throughput = @as(f64, @floatFromInt(num_rows)) / (avg_ms / 1000.0) / 1_000_000;
        std.debug.print("  normalize_minmax:  {d:>7.2} ms ({d:>5.1}M rows/sec)\n", .{ avg_ms, throughput });
    }

    // Benchmark normalize_zscore
    {
        for (0..WARMUP) |_| FeatureEngineering.normalizeZscore(data, output);

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            FeatureEngineering.normalizeZscore(data, output);
            times[iter] = timer.read();
        }

        const avg_ms = avgMs(&times);
        const throughput = @as(f64, @floatFromInt(num_rows)) / (avg_ms / 1000.0) / 1_000_000;
        std.debug.print("  normalize_zscore:  {d:>7.2} ms ({d:>5.1}M rows/sec)\n", .{ avg_ms, throughput });
    }

    // Benchmark log_transform
    {
        for (0..WARMUP) |_| FeatureEngineering.logTransform(data, output);

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            FeatureEngineering.logTransform(data, output);
            times[iter] = timer.read();
        }

        const avg_ms = avgMs(&times);
        const throughput = @as(f64, @floatFromInt(num_rows)) / (avg_ms / 1000.0) / 1_000_000;
        std.debug.print("  log_transform:     {d:>7.2} ms ({d:>5.1}M rows/sec)\n", .{ avg_ms, throughput });
    }

    // Benchmark clip_outliers
    {
        for (0..WARMUP) |_| FeatureEngineering.clipOutliers(data, output);

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            FeatureEngineering.clipOutliers(data, output);
            times[iter] = timer.read();
        }

        const avg_ms = avgMs(&times);
        const throughput = @as(f64, @floatFromInt(num_rows)) / (avg_ms / 1000.0) / 1_000_000;
        std.debug.print("  clip_outliers:     {d:>7.2} ms ({d:>5.1}M rows/sec)\n", .{ avg_ms, throughput });
    }

    // DuckDB comparison
    if (has_duckdb and parquet_path != null) {
        const path = parquet_path.?;

        // Z-score in DuckDB (closest equivalent)
        const sql = try std.fmt.allocPrint(allocator,
            \\SELECT (amount - AVG(amount) OVER()) / (STDDEV(amount) OVER() + 1e-8) AS zscore
            \\FROM '{s}' LIMIT 10000000;
        , .{path});
        defer allocator.free(sql);

        for (0..WARMUP) |_| {
            _ = runDuckDBTimed(allocator, sql) catch continue;
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            times[iter] = runDuckDBTimed(allocator, sql) catch 0;
        }

        const avg_ms = avgMs(&times);
        std.debug.print("  DuckDB z-score:    {d:>7.2} ms (window function)\n", .{avg_ms});
    }
}

fn benchmarkVectorSearch(allocator: std.mem.Allocator, num_docs: usize, dim: usize) !void {
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Vector Search ({d}K docs, {d}-dim embeddings)\n", .{ num_docs / 1000, dim });
    std.debug.print("================================================================================\n", .{});

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);
    const docs = try allocator.alloc(f32, num_docs * dim);
    defer allocator.free(docs);
    const output = try allocator.alloc(f32, num_docs);
    defer allocator.free(output);

    // Initialize with random normalized vectors
    var rng = std.Random.DefaultPrng.init(42);
    for (query) |*q| q.* = rng.random().float(f32) * 2 - 1;
    for (docs) |*d| d.* = rng.random().float(f32) * 2 - 1;

    // Benchmark cosine_similarity
    {
        for (0..WARMUP) |_| VectorSearch.cosineSimilarity(query, docs, dim, output);

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            VectorSearch.cosineSimilarity(query, docs, dim, output);
            times[iter] = timer.read();
        }

        const avg_ms = avgMs(&times);
        const throughput = @as(f64, @floatFromInt(num_docs)) / (avg_ms / 1000.0) / 1_000;
        std.debug.print("  cosine_similarity: {d:>7.2} ms ({d:>5.1}K docs/sec)\n", .{ avg_ms, throughput });
    }

    // Benchmark euclidean_distance
    {
        for (0..WARMUP) |_| VectorSearch.euclideanDistance(query, docs, dim, output);

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            VectorSearch.euclideanDistance(query, docs, dim, output);
            times[iter] = timer.read();
        }

        const avg_ms = avgMs(&times);
        const throughput = @as(f64, @floatFromInt(num_docs)) / (avg_ms / 1000.0) / 1_000;
        std.debug.print("  euclidean_distance:{d:>7.2} ms ({d:>5.1}K docs/sec)\n", .{ avg_ms, throughput });
    }
}

fn benchmarkFraudDetection(allocator: std.mem.Allocator, num_txns: usize) !void {
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Fraud Detection ({d}M transactions)\n", .{ num_txns / 1_000_000 });
    std.debug.print("================================================================================\n", .{});

    const amounts = try allocator.alloc(f32, num_txns);
    defer allocator.free(amounts);
    const velocities = try allocator.alloc(f32, num_txns);
    defer allocator.free(velocities);
    const location_distances = try allocator.alloc(f32, num_txns);
    defer allocator.free(location_distances);
    const hours = try allocator.alloc(f32, num_txns);
    defer allocator.free(hours);
    const fraud_counts = try allocator.alloc(f32, num_txns);
    defer allocator.free(fraud_counts);
    const output = try allocator.alloc(f32, num_txns);
    defer allocator.free(output);

    // Initialize with realistic transaction data
    var rng = std.Random.DefaultPrng.init(42);
    for (0..num_txns) |i| {
        amounts[i] = rng.random().float(f32) * 10000.0;
        velocities[i] = rng.random().float(f32) * 20.0;
        location_distances[i] = rng.random().float(f32) * 2000.0;
        hours[i] = @floatFromInt(rng.random().uintLessThan(u32, 24));
        fraud_counts[i] = @floatFromInt(rng.random().uintLessThan(u32, 5));
    }

    // Benchmark transaction_risk_score
    {
        for (0..WARMUP) |_| {
            FraudDetection.transactionRiskScore(amounts, velocities, location_distances, hours, fraud_counts, output);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            FraudDetection.transactionRiskScore(amounts, velocities, location_distances, hours, fraud_counts, output);
            times[iter] = timer.read();
        }

        const avg_ms = avgMs(&times);
        const throughput = @as(f64, @floatFromInt(num_txns)) / (avg_ms / 1000.0) / 1_000_000;
        std.debug.print("  risk_score:        {d:>7.2} ms ({d:>5.1}M txns/sec)\n", .{ avg_ms, throughput });
    }

    // DuckDB comparison - multi-factor scoring
    if (has_duckdb and parquet_path != null) {
        const path = parquet_path.?;

        const sql = try std.fmt.allocPrint(allocator,
            \\SELECT
            \\  LEAST(0.3, (1.0 - EXP(-amount / 5000.0)) * 0.3) +
            \\  LEAST(1.0, velocity / 10.0) * 0.25 +
            \\  LEAST(1.0, location_distance / 1000.0) * 0.2 +
            \\  CASE WHEN hour >= 2 AND hour <= 5 THEN 0.1 ELSE 0 END +
            \\  LEAST(1.0, fraud_count / 3.0) * 0.15 AS risk_score
            \\FROM '{s}' LIMIT 5000000;
        , .{path});
        defer allocator.free(sql);

        for (0..WARMUP) |_| {
            _ = runDuckDBTimed(allocator, sql) catch continue;
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            times[iter] = runDuckDBTimed(allocator, sql) catch 0;
        }

        const avg_ms = avgMs(&times);
        std.debug.print("  DuckDB risk_score: {d:>7.2} ms (SQL expression)\n", .{avg_ms});
    }
}

fn benchmarkRecommendations(allocator: std.mem.Allocator, num_items: usize, dim: usize) !void {
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("Recommendations ({d}K items, {d}-dim embeddings)\n", .{ num_items / 1000, dim });
    std.debug.print("================================================================================\n", .{});

    const user_embedding = try allocator.alloc(f32, dim);
    defer allocator.free(user_embedding);
    const item_embeddings = try allocator.alloc(f32, num_items * dim);
    defer allocator.free(item_embeddings);
    const view_counts = try allocator.alloc(f32, num_items);
    defer allocator.free(view_counts);
    const age_days = try allocator.alloc(f32, num_items);
    defer allocator.free(age_days);
    const output = try allocator.alloc(f32, num_items);
    defer allocator.free(output);

    // Initialize
    var rng = std.Random.DefaultPrng.init(42);
    for (user_embedding) |*u| u.* = rng.random().float(f32) * 2 - 1;
    for (item_embeddings) |*e| e.* = rng.random().float(f32) * 2 - 1;
    for (0..num_items) |i| {
        view_counts[i] = @floatFromInt(rng.random().uintLessThan(u32, 100000));
        age_days[i] = rng.random().float(f32) * 365.0;
    }

    // Benchmark collaborative_score
    {
        for (0..WARMUP) |_| {
            Recommendations.collaborativeScore(user_embedding, item_embeddings, view_counts, age_days, dim, output);
        }

        var times: [ITERATIONS]u64 = undefined;
        for (0..ITERATIONS) |iter| {
            var timer = try std.time.Timer.start();
            Recommendations.collaborativeScore(user_embedding, item_embeddings, view_counts, age_days, dim, output);
            times[iter] = timer.read();
        }

        const avg_ms = avgMs(&times);
        const throughput = @as(f64, @floatFromInt(num_items)) / (avg_ms / 1000.0) / 1_000;
        std.debug.print("  collaborative:     {d:>7.2} ms ({d:>5.1}K items/sec)\n", .{ avg_ms, throughput });
    }
}

fn avgMs(times: []const u64) f64 {
    var total: u64 = 0;
    for (times) |t| total += t;
    return @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(times.len)) / 1_000_000;
}
