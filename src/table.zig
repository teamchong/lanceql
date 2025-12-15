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
    NoPages,
    InvalidBufferIndex,
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

    /// Get column index by name (returns field array index).
    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        const schema = self.schema orelse return null;
        return schema.fieldIndex(name);
    }

    /// Get physical column ID by name (for use with column metadata).
    pub fn physicalColumnId(self: Self, name: []const u8) ?u32 {
        const schema = self.schema orelse return null;
        return schema.physicalColumnId(name);
    }

    /// Get field info by column index (array index in schema.fields).
    pub fn getField(self: Self, col_idx: usize) ?Field {
        const schema = self.schema orelse return null;
        if (col_idx >= schema.fields.len) return null;
        return schema.fields[col_idx];
    }

    /// Get field info by physical column ID.
    pub fn getFieldById(self: Self, field_id: u32) ?Field {
        const schema = self.schema orelse return null;
        for (schema.fields) |field| {
            if (field.id >= 0 and @as(u32, @intCast(field.id)) == field_id) {
                return field;
            }
        }
        return null;
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

    /// Read all int32 values from a column.
    pub fn readInt32Column(self: Self, col_idx: u32) TableError![]i32 {
        const buffer_data = try self.getColumnBuffer(col_idx);
        const decoder = PlainDecoder.init(buffer_data);
        return decoder.readAllInt32(self.allocator) catch return TableError.OutOfMemory;
    }

    /// Read all int32 values from a column by name.
    pub fn readInt32ColumnByName(self: Self, name: []const u8) TableError![]i32 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readInt32Column(@intCast(idx));
    }

    /// Read all float32 values from a column.
    pub fn readFloat32Column(self: Self, col_idx: u32) TableError![]f32 {
        const buffer_data = try self.getColumnBuffer(col_idx);
        const decoder = PlainDecoder.init(buffer_data);
        return decoder.readAllFloat32(self.allocator) catch return TableError.OutOfMemory;
    }

    /// Read all float32 values from a column by name.
    pub fn readFloat32ColumnByName(self: Self, name: []const u8) TableError![]f32 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readFloat32Column(@intCast(idx));
    }

    /// Read all boolean values from a column.
    pub fn readBoolColumn(self: Self, col_idx: u32) TableError![]bool {
        const buffer_data = try self.getColumnBuffer(col_idx);
        const row_count = try self.numRows(col_idx);
        const decoder = PlainDecoder.init(buffer_data);
        return decoder.readAllBool(self.allocator, row_count) catch return TableError.OutOfMemory;
    }

    /// Read all boolean values from a column by name.
    pub fn readBoolColumnByName(self: Self, name: []const u8) TableError![]bool {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readBoolColumn(@intCast(idx));
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

    /// Get multiple buffers for a column (needed for variable-length types like strings).
    /// Returns a slice of buffers corresponding to the requested buffer indices.
    fn getColumnBuffers(self: Self, col_idx: u32, buffer_indices: []const usize) TableError![][]const u8 {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return TableError.NoPages;
        const page = col_meta.pages[0];

        var buffers = try self.allocator.alloc([]const u8, buffer_indices.len);
        errdefer self.allocator.free(buffers);

        for (buffer_indices, 0..) |buf_idx, i| {
            if (buf_idx >= page.buffer_offsets.len) return TableError.InvalidBufferIndex;

            const buffer_offset = page.buffer_offsets[buf_idx];
            const buffer_size = page.buffer_sizes[buf_idx];

            buffers[i] = self.lance_file.readBytes(buffer_offset, buffer_size) catch {
                return TableError.InvalidMetadata;
            };
        }

        return buffers;
    }

    /// Get the number of rows in a specific column.
    /// Reads the column metadata and sums the length across all pages.
    fn numRows(self: Self, col_idx: u32) TableError!usize {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        // Sum length across all pages
        var total_rows: usize = 0;
        for (col_meta.pages) |page| {
            total_rows += @intCast(page.length);
        }

        return total_rows;
    }

    /// Read a string column by index.
    /// Returns a slice of allocated strings (UTF-8 byte slices).
    /// Caller must free each string AND the slice itself using the same allocator.
    pub fn readStringColumn(self: Self, col_idx: u32) TableError![][]const u8 {
        // Get column metadata
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return TableError.NoPages;
        const page = col_meta.pages[0];

        // Lance stores string columns with TWO separate buffers per page:
        // - Buffer 0: offsets array (uint32 or uint64, marking END positions)
        // - Buffer 1: string data (concatenated UTF-8 bytes)
        if (page.buffer_offsets.len < 2) return TableError.InvalidMetadata;

        // Buffer 0 = offsets array
        const offsets_offset = page.buffer_offsets[0];
        const offsets_size = page.buffer_sizes[0];
        const offsets_buffer = self.lance_file.readBytes(offsets_offset, offsets_size) catch {
            return TableError.InvalidMetadata;
        };

        // Buffer 1 = string data
        const data_offset = page.buffer_offsets[1];
        const data_size = page.buffer_sizes[1];
        const data_buffer = self.lance_file.readBytes(data_offset, data_size) catch {
            return TableError.InvalidMetadata;
        };

        // Decode strings (returns slices into data_buffer, not owned copies)
        const string_slices = PlainDecoder.readAllStrings(offsets_buffer, data_buffer, self.allocator) catch {
            return TableError.InvalidMetadata;
        };
        defer self.allocator.free(string_slices);

        // Copy each string into owned memory so caller can safely free
        var owned_strings = self.allocator.alloc([]const u8, string_slices.len) catch {
            return TableError.OutOfMemory;
        };
        errdefer {
            for (owned_strings) |str| {
                if (str.len > 0) self.allocator.free(str);
            }
            self.allocator.free(owned_strings);
        }

        for (string_slices, 0..) |slice, i| {
            const copy = self.allocator.alloc(u8, slice.len) catch {
                // Mark how many we successfully allocated for errdefer
                owned_strings = owned_strings[0..i];
                return TableError.OutOfMemory;
            };
            @memcpy(copy, slice);
            owned_strings[i] = copy;
        }

        return owned_strings;
    }

    /// Read a string column by name.
    pub fn readStringColumnByName(self: Self, name: []const u8) TableError![][]const u8 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readStringColumn(@intCast(idx));
    }

    /// String column buffers for zero-copy Arrow export.
    pub const StringBuffers = struct {
        offsets: []const u8, // Raw Lance offsets buffer (int32 end positions)
        data: []const u8, // Raw string data buffer
    };

    /// Get raw string column buffers for zero-copy Arrow export.
    /// Returns the raw offsets and data buffers without decoding.
    pub fn getStringColumnBuffers(self: Self, col_idx: u32) TableError!StringBuffers {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return TableError.NoPages;
        const page = col_meta.pages[0];

        if (page.buffer_offsets.len < 2) return TableError.InvalidMetadata;

        // Buffer 0 = offsets array
        const offsets_offset = page.buffer_offsets[0];
        const offsets_size = page.buffer_sizes[0];
        const offsets_buffer = self.lance_file.readBytes(offsets_offset, offsets_size) catch {
            return TableError.InvalidMetadata;
        };

        // Buffer 1 = string data
        const data_offset = page.buffer_offsets[1];
        const data_size = page.buffer_sizes[1];
        const data_buffer = self.lance_file.readBytes(data_offset, data_size) catch {
            return TableError.InvalidMetadata;
        };

        return StringBuffers{
            .offsets = offsets_buffer,
            .data = data_buffer,
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
