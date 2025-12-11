//! Lance schema protobuf parser.
//!
//! Based on Lance file.proto:
//! https://github.com/lance-format/lance/blob/main/protos/file.proto

const std = @import("std");
const decoder = @import("decoder.zig");

const ProtoDecoder = decoder.ProtoDecoder;
const DecodeError = decoder.DecodeError;

/// Field type enum
pub const FieldType = enum {
    parent,
    repeated,
    leaf,
    unknown,

    pub fn fromInt(value: u64) FieldType {
        return switch (value) {
            0 => .parent,
            1 => .repeated,
            2 => .leaf,
            else => .unknown,
        };
    }
};

/// A field (column) in the schema.
pub const Field = struct {
    name: []const u8,
    id: i32,
    parent_id: i32,
    field_type: FieldType,
    logical_type: []const u8,
    nullable: bool,

    /// Check if this is a top-level (root) field.
    pub fn isTopLevel(self: Field) bool {
        return self.parent_id == -1;
    }
};

/// Lance file schema.
pub const Schema = struct {
    fields: []Field,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Parse schema from protobuf bytes.
    pub fn parse(allocator: std.mem.Allocator, data: []const u8) DecodeError!Self {
        var proto = ProtoDecoder.init(data);

        var fields = std.ArrayListUnmanaged(Field){};
        errdefer {
            for (fields.items) |field| {
                allocator.free(field.name);
                allocator.free(field.logical_type);
            }
            fields.deinit(allocator);
        }

        while (proto.hasMore()) {
            const header = try proto.readFieldHeader();

            switch (header.field_num) {
                1 => { // fields (repeated Field message)
                    const field_bytes = try proto.readBytes();
                    const field = try parseField(allocator, field_bytes);
                    fields.append(allocator, field) catch return DecodeError.OutOfMemory;
                },
                else => {
                    try proto.skipField(header.wire_type);
                },
            }
        }

        return Self{
            .fields = fields.toOwnedSlice(allocator) catch return DecodeError.OutOfMemory,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.fields) |field| {
            // Always free (we always allocate, even for empty strings)
            self.allocator.free(field.name);
            self.allocator.free(field.logical_type);
        }
        self.allocator.free(self.fields);
    }

    /// Get number of top-level columns.
    pub fn columnCount(self: Self) usize {
        var count: usize = 0;
        for (self.fields) |field| {
            if (field.isTopLevel()) count += 1;
        }
        return count;
    }

    /// Get top-level column names.
    pub fn columnNames(self: Self, allocator: std.mem.Allocator) ![][]const u8 {
        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer names.deinit(allocator);

        for (self.fields) |field| {
            if (field.isTopLevel()) {
                names.append(allocator, field.name) catch return error.OutOfMemory;
            }
        }

        return names.toOwnedSlice(allocator) catch return error.OutOfMemory;
    }

    /// Find field by name.
    pub fn findField(self: Self, name: []const u8) ?Field {
        for (self.fields) |field| {
            if (std.mem.eql(u8, field.name, name)) {
                return field;
            }
        }
        return null;
    }

    /// Get field index by name.
    pub fn fieldIndex(self: Self, name: []const u8) ?usize {
        for (self.fields, 0..) |field, i| {
            if (std.mem.eql(u8, field.name, name)) {
                return i;
            }
        }
        return null;
    }
};

/// Parse a single Field from protobuf bytes.
fn parseField(allocator: std.mem.Allocator, data: []const u8) DecodeError!Field {
    var proto = ProtoDecoder.init(data);

    var name: ?[]const u8 = null;
    var id: i32 = 0;
    var parent_id: i32 = -1;
    var field_type: FieldType = .leaf;
    var logical_type: ?[]const u8 = null;
    var nullable: bool = false;

    errdefer {
        if (name) |n| allocator.free(n);
        if (logical_type) |lt| allocator.free(lt);
    }

    while (proto.hasMore()) {
        const header = try proto.readFieldHeader();

        switch (header.field_num) {
            1 => { // type (enum)
                field_type = FieldType.fromInt(try proto.readVarint());
            },
            2 => { // name (string)
                const bytes = try proto.readBytes();
                // Free previous allocation if field appears multiple times
                if (name) |n| allocator.free(n);
                name = allocator.dupe(u8, bytes) catch return DecodeError.OutOfMemory;
            },
            3 => { // id (int32)
                const val = try proto.readVarint();
                id = @bitCast(@as(u32, @truncate(val)));
            },
            4 => { // parent_id (int32) - can be -1 for root fields
                const val = try proto.readVarint();
                parent_id = @bitCast(@as(u32, @truncate(val)));
            },
            5 => { // logical_type (string)
                const bytes = try proto.readBytes();
                // Free previous allocation if field appears multiple times
                if (logical_type) |lt| allocator.free(lt);
                logical_type = allocator.dupe(u8, bytes) catch return DecodeError.OutOfMemory;
            },
            6 => { // nullable (bool)
                nullable = try proto.readVarint() != 0;
            },
            else => {
                try proto.skipField(header.wire_type);
            },
        }
    }

    // Always allocate strings (even empty) for consistent cleanup
    const final_name = name orelse (allocator.alloc(u8, 0) catch return DecodeError.OutOfMemory);
    const final_logical_type = logical_type orelse (allocator.alloc(u8, 0) catch return DecodeError.OutOfMemory);

    return Field{
        .name = final_name,
        .id = id,
        .parent_id = parent_id,
        .field_type = field_type,
        .logical_type = final_logical_type,
        .nullable = nullable,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "parse empty schema" {
    const allocator = std.testing.allocator;
    var schema = try Schema.parse(allocator, &[_]u8{});
    defer schema.deinit();

    try std.testing.expectEqual(@as(usize, 0), schema.fields.len);
}
