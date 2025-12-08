//! High-level Table API for reading Lance files.
//!
//! This module provides a convenient interface for reading Lance files,
//! combining file parsing, schema access, and column value decoding.

const std = @import("std");
const format = @import("lanceql.format");
const proto = @import("lanceql.proto");
const encoding = @import("lanceql.encoding");

const LanceFile = format.LanceFile;
const Schema = proto.Schema;
const Field = proto.Field;
const ColumnMetadata = proto.ColumnMetadata;
const PlainDecoder = encoding.PlainDecoder;

/// Errors that can occur when reading a table.
pub const TableError = error{
    NoSchema,
    ColumnNotFound,
    InvalidMetadata,
    UnsupportedType,
    OutOfMemory,
    FileTooSmall,
    InvalidMagic,
    UnsupportedVersion,
    ReadError,
    ColumnOutOfBounds,
};

/// High-level table reader for Lance files.
pub const Table = struct {
    allocator: std.mem.Allocator,
    lance_file: LanceFile,
    schema: ?Schema,

    const Self = @This();

    /// Open a table from a byte slice.
    pub fn init(allocator: std.mem.Allocator, data: []const u8) TableError!Self {
        var lance_file = LanceFile.init(allocator, data) catch |err| {
            return switch (err) {
                error.FileTooSmall => TableError.FileTooSmall,
                error.InvalidMagic => TableError.InvalidMagic,
                error.UnsupportedVersion => TableError.UnsupportedVersion,
                error.InvalidMetadata => TableError.InvalidMetadata,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
        errdefer lance_file.deinit();

        // Parse schema from global buffer 0
        var schema: ?Schema = null;
        if (lance_file.getSchemaBytes()) |schema_bytes| {
            schema = Schema.parse(allocator, schema_bytes) catch null;
        }

        return Self{
            .allocator = allocator,
            .lance_file = lance_file,
            .schema = schema,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.schema) |*s| s.deinit();
        self.lance_file.deinit();
    }

    /// Get the number of columns.
    pub fn numColumns(self: Self) u32 {
        return self.lance_file.numColumns();
    }

    /// Get column names from schema.
    pub fn columnNames(self: Self) TableError![][]const u8 {
        const schema = self.schema orelse return TableError.NoSchema;
        return schema.columnNames(self.allocator) catch return TableError.OutOfMemory;
    }

    /// Get the schema.
    pub fn getSchema(self: Self) ?Schema {
        return self.schema;
    }

    /// Get column index by name.
    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        const schema = self.schema orelse return null;
        return schema.fieldIndex(name);
    }

    /// Get field info by column index.
    pub fn getField(self: Self, col_idx: usize) ?Field {
        const schema = self.schema orelse return null;
        if (col_idx >= schema.fields.len) return null;
        return schema.fields[col_idx];
    }

    /// Get row count for a column.
    pub fn rowCount(self: Self, col_idx: u32) TableError!u64 {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        return col_meta.rowCount();
    }

    // ========================================================================
    // Typed Column Readers
    // ========================================================================

    /// Read all int64 values from a column.
    pub fn readInt64Column(self: Self, col_idx: u32) TableError![]i64 {
        const buffer_data = try self.getColumnBuffer(col_idx);
        const decoder = PlainDecoder.init(buffer_data);
        return decoder.readAllInt64(self.allocator) catch return TableError.OutOfMemory;
    }

    /// Read all int64 values from a column by name.
    pub fn readInt64ColumnByName(self: Self, name: []const u8) TableError![]i64 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readInt64Column(@intCast(idx));
    }

    /// Read all float64 values from a column.
    pub fn readFloat64Column(self: Self, col_idx: u32) TableError![]f64 {
        const buffer_data = try self.getColumnBuffer(col_idx);
        const decoder = PlainDecoder.init(buffer_data);
        return decoder.readAllFloat64(self.allocator) catch return TableError.OutOfMemory;
    }

    /// Read all float64 values from a column by name.
    pub fn readFloat64ColumnByName(self: Self, name: []const u8) TableError![]f64 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readFloat64Column(@intCast(idx));
    }

    /// Read raw column buffer (first page, first buffer).
    pub fn getColumnBuffer(self: Self, col_idx: u32) TableError![]const u8 {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) {
            return TableError.InvalidMetadata;
        }

        const page = col_meta.pages[0];
        if (page.buffer_offsets.len == 0) {
            return TableError.InvalidMetadata;
        }

        const buffer_offset = page.buffer_offsets[0];
        const buffer_size = page.buffer_sizes[0];

        return self.lance_file.readBytes(buffer_offset, buffer_size) catch {
            return TableError.InvalidMetadata;
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "table error enum" {
    // Just verify the error set compiles
    const err: TableError = TableError.NoSchema;
    try std.testing.expect(err == TableError.NoSchema);
}
