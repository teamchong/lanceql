//! ORC Table wrapper for SQL execution
//!
//! Provides a ParquetTable-like interface for ORC files by:
//! 1. Parsing ORC format to read schema and data
//! 2. Exposing unified column read interface

const std = @import("std");
const OrcReader = @import("lanceql.encoding").OrcReader;
const OrcType = @import("lanceql.encoding").OrcType;
const format = @import("lanceql.format");
const Type = format.parquet_metadata.Type;

pub const OrcTableError = error{
    InvalidOrcFile,
    InvalidOrcMagic,
    InvalidPostScript,
    InvalidFooter,
    InvalidStripeFooter,
    InvalidStream,
    CompressionError,
    UnsupportedCompression,
    NoDataFiles,
    OutOfMemory,
    PathTooLong,
    ReadFailed,
    ColumnNotFound,
    NoRowGroups,
};

/// High-level ORC table reader
pub const OrcTable = struct {
    allocator: std.mem.Allocator,
    reader: OrcReader,
    column_names: [][]const u8, // Cached column names

    const Self = @This();

    /// Open an ORC file from in-memory data
    pub fn init(allocator: std.mem.Allocator, data: []const u8) OrcTableError!Self {
        // Initialize reader
        var reader = OrcReader.init(allocator, data) catch {
            return OrcTableError.InvalidOrcFile;
        };
        errdefer reader.deinit();

        // Cache column names
        const num_cols = reader.columnCount();
        var column_names = allocator.alloc([]const u8, num_cols) catch {
            return OrcTableError.OutOfMemory;
        };
        errdefer allocator.free(column_names);

        // Copy column names from reader
        for (0..num_cols) |i| {
            const name = if (i < reader.column_names.len)
                reader.column_names[i]
            else
                "unknown";
            column_names[i] = allocator.dupe(u8, name) catch {
                // Clean up already allocated names
                for (0..i) |j| {
                    allocator.free(column_names[j]);
                }
                return OrcTableError.OutOfMemory;
            };
        }

        return Self{
            .allocator = allocator,
            .reader = reader,
            .column_names = column_names,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.column_names) |name| {
            self.allocator.free(name);
        }
        self.allocator.free(self.column_names);
        self.reader.deinit();
    }

    // =========================================================================
    // ParquetTable-compatible interface
    // =========================================================================

    /// Get number of columns
    pub fn numColumns(self: Self) u32 {
        return @intCast(self.reader.columnCount());
    }

    /// Get number of rows
    pub fn numRows(self: Self) usize {
        return self.reader.rowCount();
    }

    /// Get column names
    pub fn getColumnNames(self: Self) [][]const u8 {
        return self.column_names;
    }

    /// Get column index by name
    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        for (self.column_names, 0..) |col_name, i| {
            if (std.mem.eql(u8, col_name, name)) {
                return i;
            }
        }
        return null;
    }

    /// Get column type (mapped to Parquet types)
    pub fn getColumnType(self: Self, col_idx: usize) ?Type {
        if (col_idx >= self.reader.column_types.len) return null;
        const orc_type = self.reader.column_types[col_idx];
        return mapOrcToParquetType(orc_type);
    }

    /// Read int64 column data
    pub fn readInt64Column(self: *Self, col_idx: usize) OrcTableError![]i64 {
        // ORC uses 1-based column IDs (0 is struct root)
        return self.reader.readLongColumn(@intCast(col_idx + 1)) catch {
            return OrcTableError.ReadFailed;
        };
    }

    /// Read int32 column data (by reading int64 and converting)
    pub fn readInt32Column(self: *Self, col_idx: usize) OrcTableError![]i32 {
        const values64 = try self.readInt64Column(col_idx);
        defer self.allocator.free(values64);

        var values32 = self.allocator.alloc(i32, values64.len) catch {
            return OrcTableError.OutOfMemory;
        };

        for (values64, 0..) |v, i| {
            values32[i] = @intCast(v);
        }

        return values32;
    }

    /// Read float64 column data
    pub fn readFloat64Column(self: *Self, col_idx: usize) OrcTableError![]f64 {
        return self.reader.readDoubleColumn(@intCast(col_idx + 1)) catch {
            return OrcTableError.ReadFailed;
        };
    }

    /// Read float32 column data (by reading float64 and converting)
    pub fn readFloat32Column(self: *Self, col_idx: usize) OrcTableError![]f32 {
        const values64 = try self.readFloat64Column(col_idx);
        defer self.allocator.free(values64);

        var values32 = self.allocator.alloc(f32, values64.len) catch {
            return OrcTableError.OutOfMemory;
        };

        for (values64, 0..) |v, i| {
            values32[i] = @floatCast(v);
        }

        return values32;
    }

    /// Read string column data
    pub fn readStringColumn(self: *Self, col_idx: usize) OrcTableError![][]const u8 {
        return self.reader.readStringColumn(@intCast(col_idx + 1)) catch {
            return OrcTableError.ReadFailed;
        };
    }

    /// Read bool column data
    pub fn readBoolColumn(self: *Self, col_idx: usize) OrcTableError![]bool {
        const values64 = try self.readInt64Column(col_idx);
        defer self.allocator.free(values64);

        var bools = self.allocator.alloc(bool, values64.len) catch {
            return OrcTableError.OutOfMemory;
        };

        for (values64, 0..) |v, i| {
            bools[i] = v != 0;
        }

        return bools;
    }

    /// Check if path is a valid ORC file
    pub fn isValid(path: []const u8) bool {
        const file = std.fs.cwd().openFile(path, .{}) catch return false;
        defer file.close();

        var header: [3]u8 = undefined;
        _ = file.read(&header) catch return false;

        return std.mem.eql(u8, &header, "ORC");
    }
};

/// Map ORC types to Parquet types
fn mapOrcToParquetType(orc_type: OrcType) ?Type {
    return switch (orc_type) {
        .boolean => .boolean,
        .byte, .short, .int => .int32,
        .long => .int64,
        .float => .float,
        .double => .double,
        .string, .varchar, .char, .binary => .byte_array,
        .date => .int32,
        .timestamp => .int64,
        else => null,
    };
}

test "orc_table: read simple fixture" {
    const allocator = std.testing.allocator;

    // Read file into memory
    const file = std.fs.cwd().openFile("tests/fixtures/simple.orc", .{}) catch |err| {
        std.debug.print("Failed to open ORC file: {}\n", .{err});
        return error.TestFailed;
    };
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch |err| {
        std.debug.print("Failed to read ORC file: {}\n", .{err});
        return error.TestFailed;
    };
    defer allocator.free(data);

    var table = OrcTable.init(allocator, data) catch |err| {
        std.debug.print("Failed to parse ORC file: {}\n", .{err});
        return error.TestFailed;
    };
    defer table.deinit();

    // Check metadata
    try std.testing.expect(table.numColumns() >= 1);
    try std.testing.expect(table.numRows() >= 1);

    // Check column names
    const names = table.getColumnNames();
    try std.testing.expect(names.len >= 1);
}
