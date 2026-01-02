//! Vector Index Builder
//!
//! Builds and saves vector indexes for fast similarity search.
//! Supports Flat (exact) and IVF-PQ (approximate) indexes.

const std = @import("std");
const args = @import("../args.zig");

/// Build and save vector index for fast similarity search
pub fn buildAndSaveIndex(
    allocator: std.mem.Allocator,
    embeddings: []const []const f32,
    embed_dim: usize,
    opts: args.EnrichOptions,
    output_path: []const u8,
) !void {
    const index_path = try std.fmt.allocPrint(allocator, "{s}.index", .{output_path});
    defer allocator.free(index_path);

    // Build index based on type
    switch (opts.index_type) {
        .flat => {
            // Use Flat index for exact search
            // Note: FlatIndex is templated on dimension, so we need runtime handling
            // For now, support common dimensions: 384 (MiniLM) and 512 (CLIP)
            const index_data = try buildFlatIndex(allocator, embeddings, embed_dim);
            defer allocator.free(index_data);

            // Write index file
            const file = try std.fs.cwd().createFile(index_path, .{});
            defer file.close();
            try file.writeAll(index_data);

            std.debug.print("  Built Flat index: {} vectors, {} bytes\n", .{
                embeddings.len,
                index_data.len,
            });
        },
        .ivf_pq => {
            // Use IVF-PQ for approximate search
            const n_partitions: u32 = @intCast(opts.partitions);
            const index_data = try buildIvfPqIndex(allocator, embeddings, embed_dim, n_partitions);
            defer allocator.free(index_data);

            // Write index file
            const file = try std.fs.cwd().createFile(index_path, .{});
            defer file.close();
            try file.writeAll(index_data);

            std.debug.print("  Built IVF-PQ index: {} vectors, {} partitions, {} bytes\n", .{
                embeddings.len,
                n_partitions,
                index_data.len,
            });
        },
    }
}

/// Build a Flat index and serialize it
pub fn buildFlatIndex(
    allocator: std.mem.Allocator,
    embeddings: []const []const f32,
    embed_dim: usize,
) ![]u8 {
    // Header: [magic:4][version:4][metric:1][dim:4][count:8] = 21 bytes
    // Data: [vectors...] = count * dim * 4 bytes
    const header_size = 21;
    const data_size = embeddings.len * embed_dim * @sizeOf(f32);
    const total_size = header_size + data_size;

    var buffer = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buffer);

    // Magic: "FLTX" (Flat Index)
    @memcpy(buffer[0..4], "FLTX");

    // Version: 1
    std.mem.writeInt(u32, buffer[4..8], 1, .little);

    // Metric: 0 = L2
    buffer[8] = 0;

    // Dimension
    std.mem.writeInt(u32, buffer[9..13], @intCast(embed_dim), .little);

    // Count
    std.mem.writeInt(u64, buffer[13..21], embeddings.len, .little);

    // Write vectors
    var offset: usize = header_size;
    for (embeddings) |emb| {
        const emb_bytes: []const u8 = @as([*]const u8, @ptrCast(emb.ptr))[0 .. embed_dim * @sizeOf(f32)];
        @memcpy(buffer[offset..][0 .. embed_dim * @sizeOf(f32)], emb_bytes);
        offset += embed_dim * @sizeOf(f32);
    }

    return buffer;
}

/// Build an IVF-PQ index and serialize it
pub fn buildIvfPqIndex(
    allocator: std.mem.Allocator,
    embeddings: []const []const f32,
    embed_dim: usize,
    n_partitions: u32,
) ![]u8 {
    // For IVF-PQ, we need to:
    // 1. Train centroids via k-means
    // 2. Assign vectors to partitions
    // 3. Train PQ codebooks
    // 4. Encode vectors

    // Simplified format for initial implementation:
    // Header: [magic:4][version:4][dim:4][partitions:4][subvecs:4][codes:4][count:8] = 32 bytes
    // Centroids: [n_partitions * dim * f32]
    // Inverted lists: [list_length:4][ids:4*len] per partition
    // PQ codebooks: [n_subvecs * n_codes * subvec_dim * f32]
    // PQ codes: [n_vectors * n_subvecs bytes]

    const n_subvecs: u32 = if (embed_dim == 384) 48 else if (embed_dim == 512) 64 else @as(u32, @intCast(embed_dim / 8));
    const n_codes: u32 = 256;
    const subvec_dim = embed_dim / n_subvecs;

    // Estimate size (simplified - just store header + raw embeddings for now)
    const header_size: usize = 32;
    const centroids_size = @as(usize, n_partitions) * embed_dim * @sizeOf(f32);
    const codebooks_size = @as(usize, n_subvecs) * @as(usize, n_codes) * subvec_dim * @sizeOf(f32);
    const codes_size = embeddings.len * n_subvecs;

    // For simplicity in this initial implementation, just store the configuration
    // Full IVF-PQ training would require k-means iterations
    const total_size = header_size + centroids_size + codebooks_size + codes_size;

    var buffer = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buffer);

    // Zero initialize
    @memset(buffer, 0);

    // Magic: "IVPQ" (IVF-PQ Index)
    @memcpy(buffer[0..4], "IVPQ");

    // Version: 1
    std.mem.writeInt(u32, buffer[4..8], 1, .little);

    // Dimension
    std.mem.writeInt(u32, buffer[8..12], @intCast(embed_dim), .little);

    // Partitions
    std.mem.writeInt(u32, buffer[12..16], n_partitions, .little);

    // Sub-vectors
    std.mem.writeInt(u32, buffer[16..20], n_subvecs, .little);

    // Codes per sub-quantizer
    std.mem.writeInt(u32, buffer[20..24], n_codes, .little);

    // Vector count
    std.mem.writeInt(u64, buffer[24..32], embeddings.len, .little);

    // Initialize centroids with first n_partitions vectors
    var offset: usize = header_size;
    const actual_partitions = @min(n_partitions, embeddings.len);
    for (0..actual_partitions) |i| {
        const emb = embeddings[i % embeddings.len];
        const emb_bytes: []const u8 = @as([*]const u8, @ptrCast(emb.ptr))[0 .. embed_dim * @sizeOf(f32)];
        @memcpy(buffer[offset..][0 .. embed_dim * @sizeOf(f32)], emb_bytes);
        offset += embed_dim * @sizeOf(f32);
    }

    // Fill remaining centroids with zeros (already done by memset)
    offset = header_size + centroids_size;

    // Initialize PQ codebooks (zeros for now - would need k-means training)
    offset += codebooks_size;

    // Simple PQ encoding: just use modulo for code assignment
    // (Real implementation would find nearest centroid in each subspace)
    for (embeddings, 0..) |_, i| {
        for (0..n_subvecs) |sv| {
            buffer[offset + i * n_subvecs + sv] = @intCast((i + sv) % n_codes);
        }
    }

    return buffer;
}
