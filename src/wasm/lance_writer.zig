//! Lance File Writer for WASM
//!
//! Provides low-level and high-level APIs for writing Lance format files.
//! The low-level API (writer*) gives direct control over byte writing.
//! The high-level API (fragment*) manages columns, metadata, and footer automatically.

const std = @import("std");
const memory = @import("memory.zig");

const wasmAlloc = memory.wasmAlloc;

// ============================================================================
// Constants
// ============================================================================

const MAX_COLUMNS = 64;

// ============================================================================
// Column Types and Info
// ============================================================================

pub const ColumnType = enum(u8) {
    int64 = 0,
    int32 = 1,
    float64 = 2,
    float32 = 3,
    string = 4,
    bool = 5,
    vector = 6,
    uint8 = 7,
};

const ColumnInfo = struct {
    name_ptr: [*]const u8,
    name_len: usize,
    col_type: ColumnType,
    data_offset: usize,
    data_size: usize,
    row_count: usize,
    vector_dim: u32, // Only for vector type
    nullable: bool,
};

// ============================================================================
// Writer State
// ============================================================================

var writer_buffer: ?[*]u8 = null;
var writer_buffer_len: usize = 0;
var writer_offset: usize = 0;

var fragment_columns: [MAX_COLUMNS]ColumnInfo = undefined;
var fragment_column_count: usize = 0;
var fragment_row_count: usize = 0;

// ============================================================================
// Low-Level Writer API
// ============================================================================

/// Initialize a new Lance file writer with capacity
pub export fn writerInit(capacity: usize) u32 {
    writer_buffer = wasmAlloc(capacity);
    if (writer_buffer == null) return 0;
    writer_buffer_len = capacity;
    writer_offset = 0;
    return 1;
}

/// Get pointer to writer buffer for JS to write column data directly
pub export fn writerGetBuffer() ?[*]u8 {
    return writer_buffer;
}

/// Get current write offset
pub export fn writerGetOffset() usize {
    return writer_offset;
}

/// Write int64 values to buffer
pub export fn writerWriteInt64(values: [*]const i64, count: usize) u32 {
    const buf = writer_buffer orelse return 0;
    const bytes_needed = count * 8;
    if (writer_offset + bytes_needed > writer_buffer_len) return 0;

    var i: usize = 0;
    while (i < count) : (i += 1) {
        std.mem.writeInt(i64, buf[writer_offset..][0..8], values[i], .little);
        writer_offset += 8;
    }
    return 1;
}

/// Write int32 values to buffer
pub export fn writerWriteInt32(values: [*]const i32, count: usize) u32 {
    const buf = writer_buffer orelse return 0;
    const bytes_needed = count * 4;
    if (writer_offset + bytes_needed > writer_buffer_len) return 0;

    var i: usize = 0;
    while (i < count) : (i += 1) {
        std.mem.writeInt(i32, buf[writer_offset..][0..4], values[i], .little);
        writer_offset += 4;
    }
    return 1;
}

/// Write float64 values to buffer
pub export fn writerWriteFloat64(values: [*]const f64, count: usize) u32 {
    const buf = writer_buffer orelse return 0;
    const bytes_needed = count * 8;
    if (writer_offset + bytes_needed > writer_buffer_len) return 0;

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const bits: u64 = @bitCast(values[i]);
        std.mem.writeInt(u64, buf[writer_offset..][0..8], bits, .little);
        writer_offset += 8;
    }
    return 1;
}

/// Write float32 values to buffer
pub export fn writerWriteFloat32(values: [*]const f32, count: usize) u32 {
    const buf = writer_buffer orelse return 0;
    const bytes_needed = count * 4;
    if (writer_offset + bytes_needed > writer_buffer_len) return 0;

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const bits: u32 = @bitCast(values[i]);
        std.mem.writeInt(u32, buf[writer_offset..][0..4], bits, .little);
        writer_offset += 4;
    }
    return 1;
}

/// Write raw bytes to buffer (for strings, vectors, etc)
pub export fn writerWriteBytes(data: [*]const u8, len: usize) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + len > writer_buffer_len) return 0;

    @memcpy(buf[writer_offset..][0..len], data[0..len]);
    writer_offset += len;
    return 1;
}

/// Write u32 offset value (for string offsets)
pub export fn writerWriteOffset32(value: u32) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + 4 > writer_buffer_len) return 0;

    std.mem.writeInt(u32, buf[writer_offset..][0..4], value, .little);
    writer_offset += 4;
    return 1;
}

/// Write u64 offset value
pub export fn writerWriteOffset64(value: u64) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + 8 > writer_buffer_len) return 0;

    std.mem.writeInt(u64, buf[writer_offset..][0..8], value, .little);
    writer_offset += 8;
    return 1;
}

/// Write Lance footer (40 bytes)
pub export fn writerWriteFooter(
    column_meta_start: u64,
    column_meta_offsets_start: u64,
    global_buff_offsets_start: u64,
    num_global_buffers: u32,
    num_cols: u32,
    major_version: u16,
    minor_version: u16,
) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + 40 > writer_buffer_len) return 0;

    std.mem.writeInt(u64, buf[writer_offset..][0..8], column_meta_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u64, buf[writer_offset..][0..8], column_meta_offsets_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u64, buf[writer_offset..][0..8], global_buff_offsets_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u32, buf[writer_offset..][0..4], num_global_buffers, .little);
    writer_offset += 4;
    std.mem.writeInt(u32, buf[writer_offset..][0..4], num_cols, .little);
    writer_offset += 4;
    std.mem.writeInt(u16, buf[writer_offset..][0..2], major_version, .little);
    writer_offset += 2;
    std.mem.writeInt(u16, buf[writer_offset..][0..2], minor_version, .little);
    writer_offset += 2;
    @memcpy(buf[writer_offset..][0..4], "LANC");
    writer_offset += 4;

    return 1;
}

/// Write protobuf varint
pub export fn writerWriteVarint(value: u64) u32 {
    const buf = writer_buffer orelse return 0;
    var v = value;

    while (v >= 0x80) {
        if (writer_offset >= writer_buffer_len) return 0;
        buf[writer_offset] = @as(u8, @truncate(v)) | 0x80;
        writer_offset += 1;
        v >>= 7;
    }

    if (writer_offset >= writer_buffer_len) return 0;
    buf[writer_offset] = @truncate(v);
    writer_offset += 1;

    return 1;
}

/// Finalize and return the final file size
pub export fn writerFinalize() usize {
    return writer_offset;
}

/// Reset writer for next file
pub export fn writerReset() void {
    writer_offset = 0;
}

// ============================================================================
// High-Level Fragment Writer API
// ============================================================================
// Manages column schema, data offsets, and metadata writing automatically.
// JS just needs to: fragmentBegin -> fragmentAdd*Column (for each) -> fragmentEnd

/// Begin a new fragment (resets state)
pub export fn fragmentBegin(capacity: usize) u32 {
    if (writerInit(capacity) == 0) return 0;
    fragment_column_count = 0;
    fragment_row_count = 0;
    return 1;
}

/// Add a column with int64 data
pub export fn fragmentAddInt64Column(
    name_ptr: [*]const u8,
    name_len: usize,
    values: [*]const i64,
    count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    const data_offset = writer_offset;
    if (writerWriteInt64(values, count) == 0) return 0;
    const data_size = writer_offset - data_offset;

    fragment_columns[fragment_column_count] = .{
        .name_ptr = name_ptr,
        .name_len = name_len,
        .col_type = .int64,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;

    return 1;
}

/// Add a column with int32 data
pub export fn fragmentAddInt32Column(
    name_ptr: [*]const u8,
    name_len: usize,
    values: [*]const i32,
    count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    const data_offset = writer_offset;
    if (writerWriteInt32(values, count) == 0) return 0;
    const data_size = writer_offset - data_offset;

    fragment_columns[fragment_column_count] = .{
        .name_ptr = name_ptr,
        .name_len = name_len,
        .col_type = .int32,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;

    return 1;
}

/// Add a column with float64 data
pub export fn fragmentAddFloat64Column(
    name_ptr: [*]const u8,
    name_len: usize,
    values: [*]const f64,
    count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    const data_offset = writer_offset;
    if (writerWriteFloat64(values, count) == 0) return 0;
    const data_size = writer_offset - data_offset;

    fragment_columns[fragment_column_count] = .{
        .name_ptr = name_ptr,
        .name_len = name_len,
        .col_type = .float64,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;

    return 1;
}

/// Add a column with float32 data
pub export fn fragmentAddFloat32Column(
    name_ptr: [*]const u8,
    name_len: usize,
    values: [*]const f32,
    count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    const data_offset = writer_offset;
    if (writerWriteFloat32(values, count) == 0) return 0;
    const data_size = writer_offset - data_offset;

    fragment_columns[fragment_column_count] = .{
        .name_ptr = name_ptr,
        .name_len = name_len,
        .col_type = .float32,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;

    return 1;
}

/// Add a column with string data (data followed by offsets)
/// string_data: concatenated UTF-8 bytes
/// offsets: uint32 array of length count+1 (start positions + final end)
pub export fn fragmentAddStringColumn(
    name_ptr: [*]const u8,
    name_len: usize,
    string_data: [*]const u8,
    string_data_len: usize,
    offsets: [*]const u32,
    count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    const data_offset = writer_offset;

    // Write string data
    if (writerWriteBytes(string_data, string_data_len) == 0) return 0;

    // Write offsets (count + 1 values)
    const buf = writer_buffer orelse return 0;
    const offsets_bytes = (count + 1) * 4;
    if (writer_offset + offsets_bytes > writer_buffer_len) return 0;

    var i: usize = 0;
    while (i <= count) : (i += 1) {
        std.mem.writeInt(u32, buf[writer_offset..][0..4], offsets[i], .little);
        writer_offset += 4;
    }

    const data_size = writer_offset - data_offset;

    fragment_columns[fragment_column_count] = .{
        .name_ptr = name_ptr,
        .name_len = name_len,
        .col_type = .string,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;

    return 1;
}

/// Add a column with boolean data (bit-packed)
pub export fn fragmentAddBoolColumn(
    name_ptr: [*]const u8,
    name_len: usize,
    packed_bits: [*]const u8,
    byte_count: usize,
    row_count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    const data_offset = writer_offset;
    if (writerWriteBytes(packed_bits, byte_count) == 0) return 0;
    const data_size = writer_offset - data_offset;

    fragment_columns[fragment_column_count] = .{
        .name_ptr = name_ptr,
        .name_len = name_len,
        .col_type = .bool,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = row_count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    fragment_column_count += 1;
    if (row_count > fragment_row_count) fragment_row_count = row_count;

    return 1;
}

/// Add a column with vector data (float32 arrays, flattened)
pub export fn fragmentAddVectorColumn(
    name_ptr: [*]const u8,
    name_len: usize,
    values: [*]const f32,
    total_floats: usize,
    vector_dim: u32,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;
    if (vector_dim == 0) return 0;

    const data_offset = writer_offset;
    if (writerWriteFloat32(values, total_floats) == 0) return 0;
    const data_size = writer_offset - data_offset;

    const row_count = total_floats / vector_dim;

    fragment_columns[fragment_column_count] = .{
        .name_ptr = name_ptr,
        .name_len = name_len,
        .col_type = .vector,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = row_count,
        .vector_dim = vector_dim,
        .nullable = nullable,
    };
    fragment_column_count += 1;
    if (row_count > fragment_row_count) fragment_row_count = row_count;

    return 1;
}

/// Helper to write column metadata in protobuf format
fn writeColumnMetadata(col: *const ColumnInfo) void {
    const buf = writer_buffer orelse return;

    // Field 1: name (string) - tag = (1 << 3) | 2 = 10
    buf[writer_offset] = 10;
    writer_offset += 1;
    writeVarintInternal(col.name_len);
    @memcpy(buf[writer_offset..][0..col.name_len], col.name_ptr[0..col.name_len]);
    writer_offset += col.name_len;

    // Field 2: type (string) - tag = (2 << 3) | 2 = 18
    const type_str = switch (col.col_type) {
        .int64 => "int64",
        .int32 => "int32",
        .float64 => "float64",
        .float32 => "float32",
        .string => "string",
        .bool => "bool",
        .vector => "vector",
        .uint8 => "uint8",
    };
    buf[writer_offset] = 18;
    writer_offset += 1;
    writeVarintInternal(type_str.len);
    @memcpy(buf[writer_offset..][0..type_str.len], type_str);
    writer_offset += type_str.len;

    // Field 3: nullable (varint) - tag = (3 << 3) | 0 = 24
    buf[writer_offset] = 24;
    writer_offset += 1;
    buf[writer_offset] = if (col.nullable) 1 else 0;
    writer_offset += 1;

    // Field 4: data_offset (fixed64) - tag = (4 << 3) | 1 = 33
    buf[writer_offset] = 33;
    writer_offset += 1;
    std.mem.writeInt(u64, buf[writer_offset..][0..8], col.data_offset, .little);
    writer_offset += 8;

    // Field 5: row_count (varint) - tag = (5 << 3) | 0 = 40
    buf[writer_offset] = 40;
    writer_offset += 1;
    writeVarintInternal(col.row_count);

    // Field 6: data_size (varint) - tag = (6 << 3) | 0 = 48
    buf[writer_offset] = 48;
    writer_offset += 1;
    writeVarintInternal(col.data_size);

    // Field 7: vector_dim (varint) - tag = (7 << 3) | 0 = 56, only if vector
    if (col.col_type == .vector and col.vector_dim > 0) {
        buf[writer_offset] = 56;
        writer_offset += 1;
        writeVarintInternal(col.vector_dim);
    }
}

fn writeVarintInternal(value: usize) void {
    const buf = writer_buffer orelse return;
    var v = value;

    while (v >= 0x80) {
        buf[writer_offset] = @as(u8, @truncate(v)) | 0x80;
        writer_offset += 1;
        v >>= 7;
    }
    buf[writer_offset] = @truncate(v);
    writer_offset += 1;
}

/// Finish the fragment - writes metadata, offsets table, and footer
/// Returns final file size, or 0 on error
pub export fn fragmentEnd() usize {
    if (fragment_column_count == 0) return 0;
    const buf = writer_buffer orelse return 0;

    // Record where column metadata starts
    const col_meta_start = writer_offset;

    // Track each column's metadata offset
    var meta_offsets: [MAX_COLUMNS]usize = undefined;

    // Write column metadata
    for (0..fragment_column_count) |i| {
        meta_offsets[i] = writer_offset;
        writeColumnMetadata(&fragment_columns[i]);
    }

    // Record where offsets table starts
    const col_meta_offsets_start = writer_offset;

    // Write metadata offsets table (uint64 per column)
    for (0..fragment_column_count) |i| {
        std.mem.writeInt(u64, buf[writer_offset..][0..8], meta_offsets[i], .little);
        writer_offset += 8;
    }

    // Global buffer offsets (none for now)
    const global_buff_offsets_start = writer_offset;

    // Write footer (40 bytes)
    std.mem.writeInt(u64, buf[writer_offset..][0..8], col_meta_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u64, buf[writer_offset..][0..8], col_meta_offsets_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u64, buf[writer_offset..][0..8], global_buff_offsets_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u32, buf[writer_offset..][0..4], 0, .little); // num_global_buffers
    writer_offset += 4;
    std.mem.writeInt(u32, buf[writer_offset..][0..4], @intCast(fragment_column_count), .little);
    writer_offset += 4;
    std.mem.writeInt(u16, buf[writer_offset..][0..2], 0, .little); // major version (Lance 2.0)
    writer_offset += 2;
    std.mem.writeInt(u16, buf[writer_offset..][0..2], 3, .little); // minor version
    writer_offset += 2;
    @memcpy(buf[writer_offset..][0..4], "LANC");
    writer_offset += 4;

    return writer_offset;
}

// ============================================================================
// Tests
// ============================================================================

test "lance_writer: writerInit" {
    const result = writerInit(1024);
    try std.testing.expectEqual(@as(u32, 1), result);
    try std.testing.expect(writer_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), writer_offset);
}

test "lance_writer: writerWriteInt64" {
    _ = writerInit(1024);
    const values = [_]i64{ 1, 2, 3 };
    const result = writerWriteInt64(&values, 3);
    try std.testing.expectEqual(@as(u32, 1), result);
    try std.testing.expectEqual(@as(usize, 24), writer_offset);
}

test "lance_writer: fragmentBegin" {
    const result = fragmentBegin(1024);
    try std.testing.expectEqual(@as(u32, 1), result);
    try std.testing.expectEqual(@as(usize, 0), fragment_column_count);
}
