//! Lakehouse Format Ingesters
//!
//! Delta Lake and Apache Iceberg ingestion to Lance format.

const std = @import("std");
const lanceql = @import("lanceql");
const delta = lanceql.encoding.delta;
const iceberg = lanceql.encoding.iceberg;

/// Ingest Delta Lake table to Lance file
pub fn ingestDelta(
    allocator: std.mem.Allocator,
    path: []const u8,
    output_path: []const u8,
) !void {
    _ = output_path;
    std.debug.print("Parsing Delta Lake table: {s}\n", .{path});

    var reader = delta.DeltaReader.init(allocator, path) catch |err| {
        std.debug.print("Error parsing Delta: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const num_cols = reader.columnCount();
    const num_rows = reader.rowCount();
    const num_files = reader.fileCount();

    std.debug.print("  Rows: {d}\n", .{num_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});
    std.debug.print("  Data files: {d}\n", .{num_files});

    std.debug.print("\nError: Delta Lake ingest not yet implemented.\n", .{});
    std.debug.print("Workaround: Use 'lanceql ingest <delta_table>/*.parquet -o output.lance'\n", .{});
}

/// Ingest Iceberg table to Lance file
pub fn ingestIceberg(
    allocator: std.mem.Allocator,
    path: []const u8,
    output_path: []const u8,
) !void {
    _ = output_path;
    std.debug.print("Parsing Iceberg table: {s}\n", .{path});

    var reader = iceberg.IcebergReader.init(allocator, path) catch |err| {
        std.debug.print("Error parsing Iceberg: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const num_cols = reader.columnCount();
    const format_version = reader.formatVersion();
    const snapshot_id = reader.snapshotId();

    std.debug.print("  Format version: {d}\n", .{format_version});
    std.debug.print("  Columns: {d}\n", .{num_cols});
    std.debug.print("  Snapshot ID: {d}\n", .{snapshot_id});

    std.debug.print("\nError: Iceberg ingest not yet implemented.\n", .{});
    std.debug.print("Workaround: Use 'lanceql ingest <iceberg_table>/data/*.parquet -o output.lance'\n", .{});
}
