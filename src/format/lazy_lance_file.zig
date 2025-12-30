//! Lazy Lance file reader - Column-first I/O
//!
//! Only reads the bytes needed for requested columns, not the entire file.
//! This provides 10-100x I/O improvement for queries that only need 1-2 columns
//! from a wide table (e.g., reading 1MB instead of 149MB).
//!
//! ## I/O Pattern
//! 1. Read footer (40 bytes) - get file layout
//! 2. Read column metadata offset table (~KB) - locate columns
//! 3. Read specific column's metadata - get data buffer locations
//! 4. Read only that column's data buffers
//!
//! ## Usage
//! ```zig
//! var file_reader = try FileReader.open("data.lance");
//! defer file_reader.close();
//!
//! var lazy = try LazyLanceFile.init(allocator, file_reader.reader());
//! defer lazy.deinit();
//!
//! // Only reads the bytes for column 1
//! const amounts = try lazy.readFloat64Column(1);
//! defer allocator.free(amounts);
//! ```

const std = @import("std");
const footer_mod = @import("footer.zig");
const proto = @import("lanceql.proto");
const encoding = @import("lanceql.encoding");
const io = @import("lanceql.io");

const Footer = footer_mod.Footer;
const FOOTER_SIZE = footer_mod.FOOTER_SIZE;
const Reader = io.Reader;
const ReadError = io.ReadError;
const ColumnMetadata = proto.ColumnMetadata;
const PlainDecoder = encoding.PlainDecoder;

pub const LazyLanceFileError = error{
    FileTooSmall,
    InvalidMagic,
    UnsupportedVersion,
    InvalidMetadata,
    OutOfMemory,
    ColumnOutOfBounds,
    NoPages,
    IoError,
};

/// Position and length pair from offset tables
pub const OffsetEntry = struct {
    position: u64,
    length: u64,
};

/// Lazy Lance file reader - only reads requested columns
pub const LazyLanceFile = struct {
    allocator: std.mem.Allocator,
    reader: Reader,
    footer: Footer,
    file_size: u64,

    /// Column metadata offset table (cached after first access)
    column_meta_entries: ?[]OffsetEntry = null,

    const Self = @This();

    /// Initialize lazy reader - only reads footer (40 bytes)
    pub fn init(allocator: std.mem.Allocator, reader: Reader) LazyLanceFileError!Self {
        const file_size = reader.size() catch return LazyLanceFileError.IoError;

        if (file_size < FOOTER_SIZE) {
            return LazyLanceFileError.FileTooSmall;
        }

        // Read footer (40 bytes from end)
        var footer_buf: [FOOTER_SIZE]u8 = undefined;
        reader.readExact(file_size - FOOTER_SIZE, &footer_buf) catch {
            return LazyLanceFileError.IoError;
        };

        const footer = Footer.parse(&footer_buf) catch {
            return LazyLanceFileError.InvalidMagic;
        };

        if (!footer.isSupported()) {
            return LazyLanceFileError.UnsupportedVersion;
        }

        return Self{
            .allocator = allocator,
            .reader = reader,
            .footer = footer,
            .file_size = file_size,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.column_meta_entries) |entries| {
            self.allocator.free(entries);
        }
    }

    /// Get number of columns
    pub fn numColumns(self: Self) u32 {
        return self.footer.num_columns;
    }

    /// Load column metadata offset table (lazy - only on first use)
    fn ensureColumnMetaEntries(self: *Self) LazyLanceFileError![]OffsetEntry {
        if (self.column_meta_entries) |entries| {
            return entries;
        }

        const num_cols = self.footer.num_columns;
        const table_size = num_cols * 16; // 16 bytes per entry
        const table_offset = self.footer.column_meta_offsets_start;

        // Read column metadata offset table
        const table_buf = self.allocator.alloc(u8, table_size) catch {
            return LazyLanceFileError.OutOfMemory;
        };
        defer self.allocator.free(table_buf);

        self.reader.readExact(table_offset, table_buf) catch {
            return LazyLanceFileError.IoError;
        };

        // Parse offset entries
        var entries = self.allocator.alloc(OffsetEntry, num_cols) catch {
            return LazyLanceFileError.OutOfMemory;
        };
        errdefer self.allocator.free(entries);

        var i: u32 = 0;
        while (i < num_cols) : (i += 1) {
            const entry_pos = i * 16;
            entries[i] = .{
                .position = std.mem.readInt(u64, table_buf[entry_pos..][0..8], .little),
                .length = std.mem.readInt(u64, table_buf[entry_pos + 8 ..][0..8], .little),
            };
        }

        self.column_meta_entries = entries;
        return entries;
    }

    /// Read column metadata for a specific column
    fn readColumnMetadata(self: *Self, col_idx: u32) LazyLanceFileError!ColumnMetadata {
        const entries = try self.ensureColumnMetaEntries();

        if (col_idx >= entries.len) {
            return LazyLanceFileError.ColumnOutOfBounds;
        }

        const entry = entries[col_idx];

        // Read column metadata bytes
        const meta_buf = self.allocator.alloc(u8, entry.length) catch {
            return LazyLanceFileError.OutOfMemory;
        };
        defer self.allocator.free(meta_buf);

        self.reader.readExact(entry.position, meta_buf) catch {
            return LazyLanceFileError.IoError;
        };

        // Parse column metadata
        return ColumnMetadata.parse(self.allocator, meta_buf) catch {
            return LazyLanceFileError.InvalidMetadata;
        };
    }

    /// Read all float64 values from a column (column-first I/O)
    pub fn readFloat64Column(self: *Self, col_idx: u32) LazyLanceFileError![]f64 {
        var col_meta = try self.readColumnMetadata(col_idx);
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return LazyLanceFileError.NoPages;

        // Calculate total values across all pages
        var total_values: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_sizes.len > 0) {
                total_values += page.buffer_sizes[0] / @sizeOf(f64);
            }
        }

        // Allocate result buffer
        var result = self.allocator.alloc(f64, total_values) catch {
            return LazyLanceFileError.OutOfMemory;
        };
        errdefer self.allocator.free(result);

        // Read each page's buffer and decode
        var offset: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_offsets.len == 0 or page.buffer_sizes.len == 0) continue;

            const buffer_offset = page.buffer_offsets[0];
            const buffer_size = page.buffer_sizes[0];

            // Read only this page's data buffer
            const buffer_data = self.allocator.alloc(u8, buffer_size) catch {
                return LazyLanceFileError.OutOfMemory;
            };
            defer self.allocator.free(buffer_data);

            self.reader.readExact(buffer_offset, buffer_data) catch {
                return LazyLanceFileError.IoError;
            };

            const decoder = PlainDecoder.init(buffer_data);
            const page_values = decoder.readAllFloat64(self.allocator) catch {
                return LazyLanceFileError.OutOfMemory;
            };
            defer self.allocator.free(page_values);

            @memcpy(result[offset .. offset + page_values.len], page_values);
            offset += page_values.len;
        }

        return result;
    }

    /// Read all int64 values from a column (column-first I/O)
    pub fn readInt64Column(self: *Self, col_idx: u32) LazyLanceFileError![]i64 {
        var col_meta = try self.readColumnMetadata(col_idx);
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return LazyLanceFileError.NoPages;

        // Calculate total values across all pages
        var total_values: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_sizes.len > 0) {
                total_values += page.buffer_sizes[0] / @sizeOf(i64);
            }
        }

        // Allocate result buffer
        var result = self.allocator.alloc(i64, total_values) catch {
            return LazyLanceFileError.OutOfMemory;
        };
        errdefer self.allocator.free(result);

        // Read each page's buffer and decode
        var offset: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_offsets.len == 0 or page.buffer_sizes.len == 0) continue;

            const buffer_offset = page.buffer_offsets[0];
            const buffer_size = page.buffer_sizes[0];

            // Read only this page's data buffer
            const buffer_data = self.allocator.alloc(u8, buffer_size) catch {
                return LazyLanceFileError.OutOfMemory;
            };
            defer self.allocator.free(buffer_data);

            self.reader.readExact(buffer_offset, buffer_data) catch {
                return LazyLanceFileError.IoError;
            };

            const decoder = PlainDecoder.init(buffer_data);
            const page_values = decoder.readAllInt64(self.allocator) catch {
                return LazyLanceFileError.OutOfMemory;
            };
            defer self.allocator.free(page_values);

            @memcpy(result[offset .. offset + page_values.len], page_values);
            offset += page_values.len;
        }

        return result;
    }

    /// Get bytes read stats (for benchmarking)
    pub fn getBytesRead(self: Self, col_idx: u32) LazyLanceFileError!u64 {
        var total: u64 = FOOTER_SIZE; // Footer always read

        const entries = self.column_meta_entries orelse {
            // Would need to read offset table
            total += self.footer.num_columns * 16;
            return total;
        };

        if (col_idx >= entries.len) {
            return LazyLanceFileError.ColumnOutOfBounds;
        }

        // Add offset table size
        total += self.footer.num_columns * 16;

        // Add column metadata size
        const entry = entries[col_idx];
        total += entry.length;

        // Would need to parse column metadata to get exact data size
        // For now, return metadata + offset table
        return total;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "lazy lance file error enum" {
    const err: LazyLanceFileError = LazyLanceFileError.FileTooSmall;
    try std.testing.expect(err == LazyLanceFileError.FileTooSmall);
}
