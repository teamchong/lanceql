//! LanceQL Enrich Command
//!
//! Adds embeddings and vector indexes to Lance files.
//!
//! Usage:
//!   lanceql enrich input.lance --embed text_column --model minilm -o output.lance
//!   lanceql enrich input.lance --embed description --model clip -o output.lance

const std = @import("std");
const args = @import("args.zig");

// Embedding module (conditionally available if ONNX is linked)
const embedding = @import("../embedding/embedding.zig");

pub const EnrichError = error{
    NoInputFile,
    NoOutputFile,
    NoEmbedColumn,
    ColumnNotFound,
    InvalidColumnType,
    ModelLoadFailed,
    OnnxNotAvailable,
    FileReadError,
    WriteError,
    OutOfMemory,
};

/// Run the enrich command
pub fn run(allocator: std.mem.Allocator, opts: args.EnrichOptions) !void {
    _ = allocator;

    // Validate options
    const input_path = opts.input orelse {
        std.debug.print("Error: No input file specified\n", .{});
        return EnrichError.NoInputFile;
    };

    const output_path = opts.output orelse {
        std.debug.print("Error: No output file specified (use -o)\n", .{});
        return EnrichError.NoOutputFile;
    };

    const embed_column = opts.embed orelse {
        std.debug.print("Error: No column specified for embedding (use --embed)\n", .{});
        return EnrichError.NoEmbedColumn;
    };

    // Check ONNX availability
    if (!embedding.isOnnxAvailable()) {
        std.debug.print("Error: ONNX Runtime not available\n", .{});
        std.debug.print("\nTo enable embedding support:\n", .{});
        std.debug.print("  1. Install ONNX Runtime:\n", .{});
        std.debug.print("     macOS: brew install onnxruntime\n", .{});
        std.debug.print("     Linux: apt install libonnxruntime-dev\n", .{});
        std.debug.print("  2. Rebuild with ONNX support:\n", .{});
        std.debug.print("     zig build -Donnx=/usr/local/opt/onnxruntime\n", .{});
        return EnrichError.OnnxNotAvailable;
    }

    std.debug.print("ONNX Runtime version: {s}\n", .{embedding.getOnnxVersion()});

    // Show what we would do
    const model_type = opts.model;
    const embed_dim: usize = switch (model_type) {
        .minilm => embedding.MiniLM.EMBEDDING_DIM,
        .clip => embedding.Clip.EMBEDDING_DIM,
    };

    std.debug.print("\nEnrich Configuration:\n", .{});
    std.debug.print("  Input:  {s}\n", .{input_path});
    std.debug.print("  Output: {s}\n", .{output_path});
    std.debug.print("  Embed column: {s}\n", .{embed_column});
    std.debug.print("  Model: {s} ({} dimensions)\n", .{ @tagName(model_type), embed_dim });

    if (opts.index) |index_col| {
        std.debug.print("  Index column: {s}\n", .{index_col});
        std.debug.print("  Index type: {s}\n", .{@tagName(opts.index_type)});
        std.debug.print("  Partitions: {}\n", .{opts.partitions});
    }

    std.debug.print("\nEnrich command ready. Full implementation pending.\n", .{});
    std.debug.print("ONNX Runtime is available and embeddings can be generated.\n", .{});
}

// =============================================================================
// Tests
// =============================================================================

test "enrich error types" {
    // Basic compilation test
    _ = EnrichError.NoInputFile;
    _ = EnrichError.OnnxNotAvailable;
}
