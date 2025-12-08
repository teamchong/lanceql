//! Lance-specific protobuf message definitions.
//!
//! Based on Lance file2.proto:
//! https://github.com/lance-format/lance/blob/main/protos/file2.proto
//!
//! These structs represent the metadata stored in Lance files.

const std = @import("std");
const decoder = @import("decoder.zig");

const ProtoDecoder = decoder.ProtoDecoder;
const DecodeError = decoder.DecodeError;

/// Encoding location type
pub const Encoding = union(enum) {
    /// Encoding stored elsewhere in the file (deferred)
    deferred: DeferredEncoding,
    /// Encoding embedded directly in metadata
    direct: DirectEncoding,
    /// No encoding specified
    none: void,
};

/// Reference to encoding data stored elsewhere in the file.
pub const DeferredEncoding = struct {
    /// Byte offset to the encoding buffer
    buffer_location: u64,
    /// Length of the encoding buffer
    buffer_length: u64,
};

/// Encoding data embedded directly in the metadata.
pub const DirectEncoding = struct {
    /// The encoding bytes (protobuf "any" message)
    encoding: []const u8,
};

/// Describes a single page within a column.
pub const Page = struct {
    /// Byte offsets to page data buffers
    buffer_offsets: []const u64,
    /// Sizes of each buffer
    buffer_sizes: []const u64,
    /// Number of logical rows in this page
    length: u64,
    /// Page-specific encoding (if different from column encoding)
    encoding: Encoding,
    /// Priority/ordering value (typically row number for tabular data)
    priority: u64,

    pub fn deinit(self: *Page, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer_offsets);
        allocator.free(self.buffer_sizes);
        if (self.encoding == .direct) {
            allocator.free(self.encoding.direct.encoding);
        }
    }
};

/// Metadata for a single column in a Lance file.
pub const ColumnMetadata = struct {
    /// Column-level encoding description
    encoding: Encoding,
    /// Pages containing this column's data
    pages: []Page,
    /// Byte offsets to column metadata buffers
    buffer_offsets: []const u64,
    /// Sizes of column metadata buffers
    buffer_sizes: []const u64,

    const Self = @This();

    /// Parse ColumnMetadata from protobuf bytes.
    pub fn parse(allocator: std.mem.Allocator, data: []const u8) DecodeError!Self {
        var proto = ProtoDecoder.init(data);

        var encoding: Encoding = .none;
        var pages = std.ArrayListUnmanaged(Page){};
        errdefer {
            for (pages.items) |*page| {
                page.deinit(allocator);
            }
            pages.deinit(allocator);
        }
        var buffer_offsets: []const u64 = &[_]u64{};
        var buffer_sizes: []const u64 = &[_]u64{};

        while (proto.hasMore()) {
            const header = try proto.readFieldHeader();

            switch (header.field_num) {
                1 => { // encoding (oneof)
                    encoding = try parseEncoding(allocator, &proto, header);
                },
                2 => { // pages (repeated)
                    const page_bytes = try proto.readBytes();
                    const page = try parsePage(allocator, page_bytes);
                    pages.append(allocator, page) catch return DecodeError.OutOfMemory;
                },
                3 => { // buffer_offsets (packed repeated uint64)
                    buffer_offsets = try proto.readPackedFixed64(allocator);
                },
                4 => { // buffer_sizes (packed repeated uint64)
                    buffer_sizes = try proto.readPackedFixed64(allocator);
                },
                else => {
                    // Skip unknown fields for forward compatibility
                    try proto.skipField(header.wire_type);
                },
            }
        }

        return Self{
            .encoding = encoding,
            .pages = pages.toOwnedSlice(allocator) catch return DecodeError.OutOfMemory,
            .buffer_offsets = buffer_offsets,
            .buffer_sizes = buffer_sizes,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        for (self.pages) |*page| {
            page.deinit(allocator);
        }
        allocator.free(self.pages);
        allocator.free(self.buffer_offsets);
        allocator.free(self.buffer_sizes);
        if (self.encoding == .direct) {
            allocator.free(self.encoding.direct.encoding);
        }
    }

    /// Get total row count across all pages.
    pub fn rowCount(self: Self) u64 {
        var total: u64 = 0;
        for (self.pages) |page| {
            total += page.length;
        }
        return total;
    }
};

/// Parse encoding from a protobuf field.
fn parseEncoding(allocator: std.mem.Allocator, proto: *ProtoDecoder, header: decoder.FieldHeader) DecodeError!Encoding {
    _ = header;
    // Encoding is typically a nested message
    const encoding_bytes = try proto.readBytes();

    if (encoding_bytes.len == 0) {
        return .none;
    }

    // For now, store as direct encoding
    // TODO: Distinguish between deferred and direct based on message structure
    const encoding_copy = allocator.dupe(u8, encoding_bytes) catch return DecodeError.OutOfMemory;
    return .{ .direct = .{ .encoding = encoding_copy } };
}

/// Parse a Page from protobuf bytes.
fn parsePage(allocator: std.mem.Allocator, data: []const u8) DecodeError!Page {
    var proto = ProtoDecoder.init(data);

    var buffer_offsets: []const u64 = &[_]u64{};
    var buffer_sizes: []const u64 = &[_]u64{};
    var length: u64 = 0;
    var encoding: Encoding = .none;
    var priority: u64 = 0;

    while (proto.hasMore()) {
        const header = try proto.readFieldHeader();

        switch (header.field_num) {
            1 => { // buffer_offsets (packed repeated uint64)
                buffer_offsets = try proto.readPackedFixed64(allocator);
            },
            2 => { // buffer_sizes (packed repeated uint64)
                buffer_sizes = try proto.readPackedFixed64(allocator);
            },
            3 => { // length (uint64)
                length = try proto.readVarint();
            },
            4 => { // encoding
                encoding = try parseEncoding(allocator, &proto, header);
            },
            5 => { // priority (uint64)
                priority = try proto.readVarint();
            },
            else => {
                try proto.skipField(header.wire_type);
            },
        }
    }

    return Page{
        .buffer_offsets = buffer_offsets,
        .buffer_sizes = buffer_sizes,
        .length = length,
        .encoding = encoding,
        .priority = priority,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "parse empty column metadata" {
    const allocator = std.testing.allocator;
    var meta = try ColumnMetadata.parse(allocator, &[_]u8{});
    defer meta.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), meta.pages.len);
    try std.testing.expectEqual(Encoding.none, meta.encoding);
}

test "column metadata row count" {
    var meta = ColumnMetadata{
        .encoding = .none,
        .pages = &[_]Page{
            .{
                .buffer_offsets = &[_]u64{},
                .buffer_sizes = &[_]u64{},
                .length = 100,
                .encoding = .none,
                .priority = 0,
            },
            .{
                .buffer_offsets = &[_]u64{},
                .buffer_sizes = &[_]u64{},
                .length = 200,
                .encoding = .none,
                .priority = 100,
            },
        },
        .buffer_offsets = &[_]u64{},
        .buffer_sizes = &[_]u64{},
    };

    try std.testing.expectEqual(@as(u64, 300), meta.rowCount());
}
