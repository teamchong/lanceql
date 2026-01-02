//! WASM entry point for LanceQL.
//!
//! This module provides exported functions that can be called from JavaScript
//! in the browser. Uses direct byte manipulation for WASM compatibility.

const std = @import("std");

// ============================================================================
// Module imports
// ============================================================================

const memory = @import("wasm/memory.zig");
const format = @import("wasm/format.zig");
const string_column = @import("wasm/string_column.zig");
const vector_column = @import("wasm/vector_column.zig");
const compression = @import("wasm/compression.zig");
const simd_search = @import("wasm/simd_search.zig");
const gguf_utils = @import("wasm/gguf_utils.zig");
const clip_model = @import("wasm/clip_model.zig");
const minilm_model = @import("wasm/minilm_model.zig");

// Module exports are automatic via `pub export fn` in each module.
// Force reference to ensure they're included in WASM binary:
comptime {
    _ = memory;
    _ = format;
    _ = string_column;
    _ = vector_column;
    _ = compression;
    _ = simd_search;
    _ = gguf_utils;
    _ = clip_model;
    _ = minilm_model;
}

// ============================================================================
// Constants (from format module)
// ============================================================================

const FOOTER_SIZE = format.FOOTER_SIZE;

// ============================================================================
// Internal memory functions (from memory module)
// ============================================================================

const wasmAlloc = memory.wasmAlloc;
const wasmReset = memory.wasmReset;

// ============================================================================
// Internal format functions (from format module)
// ============================================================================

const readU64LE = format.readU64LE;
const readU32LE = format.readU32LE;
const readU16LE = format.readU16LE;
const readI64LE = format.readI64LE;
const readI32LE = format.readI32LE;
const readI16LE = format.readI16LE;
const readI8 = format.readI8;
const readU8 = format.readU8;
const readF64LE = format.readF64LE;
const readF32LE = format.readF32LE;
const isValidLanceFile = format.isValidLanceFile;

// ============================================================================
// Global state
// ============================================================================

var file_data: ?[]const u8 = null;
var num_columns: u32 = 0;
var column_meta_offsets_start: u64 = 0;

/// Sync global state to all modules that need it
fn syncStateToModules() void {
    string_column.file_data = file_data;
    string_column.num_columns = num_columns;
    string_column.column_meta_offsets_start = column_meta_offsets_start;
    vector_column.file_data = file_data;
    vector_column.num_columns = num_columns;
    vector_column.column_meta_offsets_start = column_meta_offsets_start;
}

// ============================================================================
// Exported Memory Management
// ============================================================================

export fn resetHeap() void {
    wasmReset();
    file_data = null;
    num_columns = 0;
    column_meta_offsets_start = 0;
    syncStateToModules();
}

// ============================================================================
// File Operations
// ============================================================================

export fn openFile(data: [*]const u8, len: usize) u32 {
    if (isValidLanceFile(data, len) == 0) return 0;

    file_data = data[0..len];

    const footer_start = len - FOOTER_SIZE;
    num_columns = readU32LE(data[0..len], footer_start + 28);
    column_meta_offsets_start = readU64LE(data[0..len], footer_start + 8);

    syncStateToModules();
    return 1;
}

export fn closeFile() void {
    file_data = null;
    num_columns = 0;
    column_meta_offsets_start = 0;
    syncStateToModules();
}

export fn getNumColumns() u32 {
    return num_columns;
}

// ============================================================================
// Column Metadata Parsing
// ============================================================================

fn getColumnOffsetEntry(col_idx: u32) struct { pos: u64, len: u64 } {
    const data = file_data orelse return .{ .pos = 0, .len = 0 };
    if (col_idx >= num_columns) return .{ .pos = 0, .len = 0 };

    const entry_offset: usize = @intCast(column_meta_offsets_start + col_idx * 16);
    if (entry_offset + 16 > data.len) return .{ .pos = 0, .len = 0 };

    return .{
        .pos = readU64LE(data, entry_offset),
        .len = readU64LE(data, entry_offset + 8),
    };
}

// Parse varint from protobuf data
fn readVarint(data: []const u8, offset: *usize) u64 {
    var result: u64 = 0;
    var shift: u6 = 0;

    while (offset.* < data.len) {
        const byte = data[offset.*];
        offset.* += 1;
        result |= @as(u64, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;
        shift +|= 7;
    }
    return result;
}

// Get page buffer info from column metadata protobuf
fn getPageBufferInfo(col_meta: []const u8) struct { offset: u64, size: u64, rows: u64 } {
    var pos: usize = 0;
    var page_offset: u64 = 0;
    var page_size: u64 = 0;
    var page_rows: u64 = 0;

    while (pos < col_meta.len) {
        const tag = readVarint(col_meta, &pos);
        const field_num = tag >> 3;
        const wire_type: u3 = @truncate(tag);

        switch (field_num) {
            1 => { // encoding (length-delimited) - skip it
                if (wire_type == 2) {
                    const skip_len = readVarint(col_meta, &pos);
                    pos += @as(usize, @intCast(skip_len));
                }
            },
            2 => { // pages (length-delimited)
                if (wire_type != 2) break;
                const page_len = readVarint(col_meta, &pos);
                const page_end = pos + @as(usize, @intCast(page_len));

                // Parse page message
                while (pos < page_end and pos < col_meta.len) {
                    const page_tag = readVarint(col_meta, &pos);
                    const page_field = page_tag >> 3;
                    const page_wire: u3 = @truncate(page_tag);

                    switch (page_field) {
                        1 => { // buffer_offsets (packed repeated uint64)
                            if (page_wire == 2) {
                                const packed_len = readVarint(col_meta, &pos);
                                const packed_end = pos + @as(usize, @intCast(packed_len));
                                // Read first offset only
                                if (pos < packed_end) {
                                    page_offset = readVarint(col_meta, &pos);
                                }
                                // Skip rest
                                pos = packed_end;
                            } else {
                                page_offset = readVarint(col_meta, &pos);
                            }
                        },
                        2 => { // buffer_sizes (packed repeated uint64)
                            if (page_wire == 2) {
                                const packed_len = readVarint(col_meta, &pos);
                                const packed_end = pos + @as(usize, @intCast(packed_len));
                                // Read first size only
                                if (pos < packed_end) {
                                    page_size = readVarint(col_meta, &pos);
                                }
                                // Skip rest
                                pos = packed_end;
                            } else {
                                page_size = readVarint(col_meta, &pos);
                            }
                        },
                        3 => { // length (rows)
                            page_rows = readVarint(col_meta, &pos);
                        },
                        else => {
                            // Skip field
                            if (page_wire == 0) {
                                _ = readVarint(col_meta, &pos);
                            } else if (page_wire == 2) {
                                const skip_len = readVarint(col_meta, &pos);
                                pos += @as(usize, @intCast(skip_len));
                            } else if (page_wire == 5) {
                                pos += 4; // 32-bit fixed
                            } else if (page_wire == 1) {
                                pos += 8; // 64-bit fixed
                            }
                        },
                    }
                }
                // Only read first page
                return .{ .offset = page_offset, .size = page_size, .rows = page_rows };
            },
            else => {
                // Skip field
                if (wire_type == 0) {
                    _ = readVarint(col_meta, &pos);
                } else if (wire_type == 2) {
                    const skip_len = readVarint(col_meta, &pos);
                    pos += @as(usize, @intCast(skip_len));
                } else if (wire_type == 5) {
                    pos += 4; // 32-bit fixed
                } else if (wire_type == 1) {
                    pos += 8; // 64-bit fixed
                }
            },
        }
    }

    return .{ .offset = page_offset, .size = page_size, .rows = page_rows };
}

export fn getRowCount(col_idx: u32) u64 {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);
    return info.rows;
}

// Debug: get buffer offset for column
export fn getColumnBufferOffset(col_idx: u32) u64 {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);
    return info.offset;
}

// Debug: get buffer size for column
export fn getColumnBufferSize(col_idx: u32) u64 {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);
    return info.size;
}

// ============================================================================
// Column Reading
// ============================================================================

/// Helper to get column buffer info for reading
pub const ColumnBuffer = struct { data: []const u8, start: usize, size: usize, rows: usize };
pub fn getColumnBuffer(col_idx: u32) ?ColumnBuffer {
    const data = file_data orelse return null;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return null;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return null;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return null;

    return .{ .data = data, .start = buf_start, .size = buf_size, .rows = @intCast(info.rows) };
}

export fn readInt64Column(col_idx: u32, out_ptr: [*]i64, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.size / 8, max_len);
    for (0..count) |i| out_ptr[i] = readI64LE(buf.data, buf.start + i * 8);
    return count;
}

export fn readFloat64Column(col_idx: u32, out_ptr: [*]f64, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.size / 8, max_len);
    for (0..count) |i| out_ptr[i] = readF64LE(buf.data, buf.start + i * 8);
    return count;
}

export fn allocInt64Buffer(count: usize) ?[*]i64 {
    const ptr = wasmAlloc(count * 8) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

export fn freeInt64Buffer(ptr: [*]i64, count: usize) void {
    _ = ptr;
    _ = count;
}

export fn allocFloat64Buffer(count: usize) ?[*]f64 {
    const ptr = wasmAlloc(count * 8) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

export fn freeFloat64Buffer(ptr: [*]f64, count: usize) void {
    _ = ptr;
    _ = count;
}

// ============================================================================
// Query Execution (Simple filter/projection for WASM)
// ============================================================================

/// Filter result indices where column value matches condition
/// op: 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge
export fn filterInt64Column(
    col_idx: u32,
    op: u32,
    value: i64,
    out_indices: [*]u32,
    max_indices: usize,
) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const row_count = buf.size / 8;
    var out_count: usize = 0;

    for (0..row_count) |i| {
        if (out_count >= max_indices) break;
        const col_val = readI64LE(buf.data, buf.start + i * 8);
        const matches = switch (op) {
            0 => col_val == value,
            1 => col_val != value,
            2 => col_val < value,
            3 => col_val <= value,
            4 => col_val > value,
            5 => col_val >= value,
            else => false,
        };
        if (matches) {
            out_indices[out_count] = @intCast(i);
            out_count += 1;
        }
    }
    return out_count;
}

/// Filter float64 column
export fn filterFloat64Column(
    col_idx: u32,
    op: u32,
    value: f64,
    out_indices: [*]u32,
    max_indices: usize,
) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const row_count = buf.size / 8;
    var out_count: usize = 0;

    for (0..row_count) |i| {
        if (out_count >= max_indices) break;
        const col_val = readF64LE(buf.data, buf.start + i * 8);
        const matches = switch (op) {
            0 => col_val == value,
            1 => col_val != value,
            2 => col_val < value,
            3 => col_val <= value,
            4 => col_val > value,
            5 => col_val >= value,
            else => false,
        };
        if (matches) {
            out_indices[out_count] = @intCast(i);
            out_count += 1;
        }
    }
    return out_count;
}

/// Read int64 values at specific indices
export fn readInt64AtIndices(
    col_idx: u32,
    indices: [*]const u32,
    num_indices: usize,
    out_ptr: [*]i64,
) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const max_idx = buf.size / 8;
    for (0..num_indices) |i| {
        const idx = indices[i];
        out_ptr[i] = if (idx >= max_idx) 0 else readI64LE(buf.data, buf.start + idx * 8);
    }
    return num_indices;
}

/// Read float64 values at specific indices
export fn readFloat64AtIndices(
    col_idx: u32,
    indices: [*]const u32,
    num_indices: usize,
    out_ptr: [*]f64,
) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const max_idx = buf.size / 8;
    for (0..num_indices) |i| {
        const idx = indices[i];
        out_ptr[i] = if (idx >= max_idx) 0 else readF64LE(buf.data, buf.start + idx * 8);
    }
    return num_indices;
}

/// Allocate index buffer
export fn allocIndexBuffer(count: usize) ?[*]u32 {
    const ptr = wasmAlloc(count * 4) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

// ============================================================================
// Additional Numeric Type Column Reading
// ============================================================================

export fn readInt32Column(col_idx: u32, out_ptr: [*]i32, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.size / 4, max_len);
    for (0..count) |i| out_ptr[i] = readI32LE(buf.data, buf.start + i * 4);
    return count;
}

export fn readInt16Column(col_idx: u32, out_ptr: [*]i16, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.size / 2, max_len);
    for (0..count) |i| out_ptr[i] = readI16LE(buf.data, buf.start + i * 2);
    return count;
}

export fn readInt8Column(col_idx: u32, out_ptr: [*]i8, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.size, max_len);
    for (0..count) |i| out_ptr[i] = readI8(buf.data, buf.start + i);
    return count;
}

export fn readUint64Column(col_idx: u32, out_ptr: [*]u64, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.size / 8, max_len);
    for (0..count) |i| out_ptr[i] = readU64LE(buf.data, buf.start + i * 8);
    return count;
}

export fn readUint32Column(col_idx: u32, out_ptr: [*]u32, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.size / 4, max_len);
    for (0..count) |i| out_ptr[i] = readU32LE(buf.data, buf.start + i * 4);
    return count;
}

export fn readUint16Column(col_idx: u32, out_ptr: [*]u16, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.size / 2, max_len);
    for (0..count) |i| out_ptr[i] = readU16LE(buf.data, buf.start + i * 2);
    return count;
}

export fn readUint8Column(col_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.size, max_len);
    @memcpy(out_ptr[0..count], buf.data[buf.start..][0..count]);
    return count;
}

export fn readFloat32Column(col_idx: u32, out_ptr: [*]f32, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.size / 4, max_len);
    for (0..count) |i| out_ptr[i] = readF32LE(buf.data, buf.start + i * 4);
    return count;
}

/// Read boolean column (stored as bit-packed in Lance)
export fn readBoolColumn(col_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const count = @min(buf.rows, max_len);

    for (0..count) |i| {
        const byte_idx = i / 8;
        const bit_idx: u3 = @intCast(i % 8);
        if (byte_idx < buf.size) {
            const byte = buf.data[buf.start + byte_idx];
            out_ptr[i] = if ((byte >> bit_idx) & 1 == 1) 1 else 0;
        } else {
            out_ptr[i] = 0;
        }
    }
    return count;
}

/// Allocate int32 buffer
export fn allocInt32Buffer(count: usize) ?[*]i32 {
    const ptr = wasmAlloc(count * 4) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

/// Allocate int16 buffer
export fn allocInt16Buffer(count: usize) ?[*]i16 {
    const ptr = wasmAlloc(count * 2) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

/// Allocate int8 buffer
export fn allocInt8Buffer(count: usize) ?[*]i8 {
    return @ptrCast(wasmAlloc(count));
}

/// Allocate uint64 buffer
export fn allocUint64Buffer(count: usize) ?[*]u64 {
    const ptr = wasmAlloc(count * 8) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

/// Allocate uint16 buffer
export fn allocUint16Buffer(count: usize) ?[*]u16 {
    const ptr = wasmAlloc(count * 2) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

/// Read int32 values at specific indices
export fn readInt32AtIndices(
    col_idx: u32,
    indices: [*]const u32,
    num_indices: usize,
    out_ptr: [*]i32,
) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const max_idx = buf.size / 4;
    for (0..num_indices) |i| {
        const idx = indices[i];
        out_ptr[i] = if (idx >= max_idx) 0 else readI32LE(buf.data, buf.start + idx * 4);
    }
    return num_indices;
}

/// Read float32 values at specific indices
export fn readFloat32AtIndices(
    col_idx: u32,
    indices: [*]const u32,
    num_indices: usize,
    out_ptr: [*]f32,
) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const max_idx = buf.size / 4;
    for (0..num_indices) |i| {
        const idx = indices[i];
        out_ptr[i] = if (idx >= max_idx) 0 else readF32LE(buf.data, buf.start + idx * 4);
    }
    return num_indices;
}

/// Read uint8 values at specific indices
export fn readUint8AtIndices(
    col_idx: u32,
    indices: [*]const u32,
    num_indices: usize,
    out_ptr: [*]u8,
) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    for (0..num_indices) |i| {
        const idx = indices[i];
        out_ptr[i] = if (idx >= buf.size) 0 else buf.data[buf.start + idx];
    }
    return num_indices;
}

/// Read bool values at specific indices
export fn readBoolAtIndices(
    col_idx: u32,
    indices: [*]const u32,
    num_indices: usize,
    out_ptr: [*]u8,
) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    for (0..num_indices) |i| {
        const idx = indices[i];
        if (idx >= buf.rows) {
            out_ptr[i] = 0;
        } else {
            const byte_idx = idx / 8;
            const bit_idx: u3 = @intCast(idx % 8);
            if (byte_idx < buf.size) {
                const byte = buf.data[buf.start + byte_idx];
                out_ptr[i] = if ((byte >> bit_idx) & 1 == 1) 1 else 0;
            } else {
                out_ptr[i] = 0;
            }
        }
    }
    return num_indices;
}

// ============================================================================
// Aggregation Functions
// ============================================================================

/// Sum int64 column
export fn sumInt64Column(col_idx: u32) i64 {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const row_count = buf.size / 8;
    var sum: i64 = 0;
    for (0..row_count) |i| sum += readI64LE(buf.data, buf.start + i * 8);
    return sum;
}

/// Sum float64 column
export fn sumFloat64Column(col_idx: u32) f64 {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const row_count = buf.size / 8;
    var sum: f64 = 0;
    for (0..row_count) |i| sum += readF64LE(buf.data, buf.start + i * 8);
    return sum;
}

/// Min int64 column
export fn minInt64Column(col_idx: u32) i64 {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const row_count = buf.size / 8;
    if (row_count == 0) return 0;
    var min_val: i64 = readI64LE(buf.data, buf.start);
    for (1..row_count) |i| {
        const val = readI64LE(buf.data, buf.start + i * 8);
        if (val < min_val) min_val = val;
    }
    return min_val;
}

/// Max int64 column
export fn maxInt64Column(col_idx: u32) i64 {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const row_count = buf.size / 8;
    if (row_count == 0) return 0;
    var max_val: i64 = readI64LE(buf.data, buf.start);
    for (1..row_count) |i| {
        const val = readI64LE(buf.data, buf.start + i * 8);
        if (val > max_val) max_val = val;
    }
    return max_val;
}

/// Average float64 column
export fn avgFloat64Column(col_idx: u32) f64 {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    const row_count = buf.size / 8;
    if (row_count == 0) return 0;
    var sum: f64 = 0;
    for (0..row_count) |i| sum += readF64LE(buf.data, buf.start + i * 8);
    return sum / @as(f64, @floatFromInt(row_count));
}

// ============================================================================
// Lance File Writer Exports
// ============================================================================

// Writer state
var writer_buffer: ?[*]u8 = null;
var writer_buffer_len: usize = 0;
var writer_offset: usize = 0;

/// Initialize a new Lance file writer with capacity
export fn writerInit(capacity: usize) u32 {
    writer_buffer = wasmAlloc(capacity);
    if (writer_buffer == null) return 0;
    writer_buffer_len = capacity;
    writer_offset = 0;
    return 1;
}

/// Get pointer to writer buffer for JS to write column data directly
export fn writerGetBuffer() ?[*]u8 {
    return writer_buffer;
}

/// Get current write offset
export fn writerGetOffset() usize {
    return writer_offset;
}

/// Write int64 values to buffer
export fn writerWriteInt64(values: [*]const i64, count: usize) u32 {
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
export fn writerWriteInt32(values: [*]const i32, count: usize) u32 {
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
export fn writerWriteFloat64(values: [*]const f64, count: usize) u32 {
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
export fn writerWriteFloat32(values: [*]const f32, count: usize) u32 {
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
export fn writerWriteBytes(data: [*]const u8, len: usize) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + len > writer_buffer_len) return 0;

    @memcpy(buf[writer_offset..][0..len], data[0..len]);
    writer_offset += len;
    return 1;
}

/// Write u32 offset value (for string offsets)
export fn writerWriteOffset32(value: u32) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + 4 > writer_buffer_len) return 0;

    std.mem.writeInt(u32, buf[writer_offset..][0..4], value, .little);
    writer_offset += 4;
    return 1;
}

/// Write u64 offset value
export fn writerWriteOffset64(value: u64) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + 8 > writer_buffer_len) return 0;

    std.mem.writeInt(u64, buf[writer_offset..][0..8], value, .little);
    writer_offset += 8;
    return 1;
}

/// Write Lance footer (40 bytes)
export fn writerWriteFooter(
    column_meta_start: u64,
    column_meta_offsets_start_arg: u64,
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
    std.mem.writeInt(u64, buf[writer_offset..][0..8], column_meta_offsets_start_arg, .little);
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
export fn writerWriteVarint(value: u64) u32 {
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
export fn writerFinalize() usize {
    return writer_offset;
}

/// Reset writer for next file
export fn writerReset() void {
    writer_offset = 0;
}

// ============================================================================
// High-Level Fragment Writer API
// ============================================================================
// Manages column schema, data offsets, and metadata writing automatically.
// JS just needs to: beginFragment -> addColumn (for each) -> endFragment

const MAX_COLUMNS = 64;

const ColumnType = enum(u8) {
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

var fragment_columns: [MAX_COLUMNS]ColumnInfo = undefined;
var fragment_column_count: usize = 0;
var fragment_row_count: usize = 0;

/// Begin a new fragment (resets state)
export fn fragmentBegin(capacity: usize) u32 {
    if (writerInit(capacity) == 0) return 0;
    fragment_column_count = 0;
    fragment_row_count = 0;
    return 1;
}

/// Add a column with int64 data
export fn fragmentAddInt64Column(
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
export fn fragmentAddInt32Column(
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
export fn fragmentAddFloat64Column(
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
export fn fragmentAddFloat32Column(
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
export fn fragmentAddStringColumn(
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
export fn fragmentAddBoolColumn(
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
export fn fragmentAddVectorColumn(
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
export fn fragmentEnd() usize {
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
// High-Level Fragment Reader API
// ============================================================================
// Parse a Lance fragment file and provide access to columns

var reader_data: ?[*]const u8 = null;
var reader_len: usize = 0;
var reader_num_columns: u32 = 0;
var reader_column_meta_start: u64 = 0;
var reader_column_meta_offsets_start: u64 = 0;

const ReaderColumnInfo = struct {
    name: [64]u8,
    name_len: usize,
    col_type: [16]u8,
    type_len: usize,
    nullable: bool,
    data_offset: u64,
    row_count: u64,
    data_size: u64,
    vector_dim: u32,
};

var reader_columns: [MAX_COLUMNS]ReaderColumnInfo = undefined;

/// Load a fragment for reading
export fn fragmentLoad(data: [*]const u8, len: usize) u32 {
    if (len < 40) return 0;

    reader_data = data;
    reader_len = len;

    // Parse footer (last 40 bytes)
    const footer_start = len - 40;

    // Check magic
    if (data[footer_start + 36] != 'L' or
        data[footer_start + 37] != 'A' or
        data[footer_start + 38] != 'N' or
        data[footer_start + 39] != 'C')
    {
        return 0;
    }

    reader_column_meta_start = std.mem.readInt(u64, data[footer_start..][0..8], .little);
    reader_column_meta_offsets_start = std.mem.readInt(u64, data[footer_start + 8 ..][0..8], .little);
    reader_num_columns = std.mem.readInt(u32, data[footer_start + 28 ..][0..4], .little);

    if (reader_num_columns > MAX_COLUMNS) return 0;

    // Parse column metadata
    for (0..reader_num_columns) |i| {
        const offset_pos: usize = @intCast(reader_column_meta_offsets_start + i * 8);
        const meta_offset = std.mem.readInt(u64, data[offset_pos..][0..8], .little);

        const next_offset = if (i + 1 < reader_num_columns)
            std.mem.readInt(u64, data[offset_pos + 8 ..][0..8], .little)
        else
            reader_column_meta_offsets_start;

        parseColumnMeta(data, meta_offset, next_offset, &reader_columns[i]);
    }

    return 1;
}

fn parseColumnMeta(data: [*]const u8, start: u64, end: u64, info: *ReaderColumnInfo) void {
    info.* = .{
        .name = undefined,
        .name_len = 0,
        .col_type = undefined,
        .type_len = 0,
        .nullable = true,
        .data_offset = 0,
        .row_count = 0,
        .data_size = 0,
        .vector_dim = 0,
    };

    var pos: usize = @intCast(start);
    const end_pos: usize = @intCast(end);
    while (pos < end_pos) {
        const tag = data[pos];
        pos += 1;

        const field_num = tag >> 3;
        const wire_type = tag & 0x7;

        switch (field_num) {
            1 => { // name (string)
                if (wire_type == 2) {
                    const len = readVarintAtUsize(data, &pos);
                    const copy_len = @min(len, 64);
                    @memcpy(info.name[0..copy_len], data[pos..][0..copy_len]);
                    info.name_len = copy_len;
                    pos += len;
                }
            },
            2 => { // type (string)
                if (wire_type == 2) {
                    const len = readVarintAtUsize(data, &pos);
                    const copy_len = @min(len, 16);
                    @memcpy(info.col_type[0..copy_len], data[pos..][0..copy_len]);
                    info.type_len = copy_len;
                    pos += len;
                }
            },
            3 => { // nullable (varint)
                if (wire_type == 0) {
                    info.nullable = readVarintAtUsize(data, &pos) != 0;
                }
            },
            4 => { // data_offset (fixed64)
                if (wire_type == 1) {
                    info.data_offset = std.mem.readInt(u64, data[pos..][0..8], .little);
                    pos += 8;
                }
            },
            5 => { // row_count (varint)
                if (wire_type == 0) {
                    info.row_count = readVarintAtUsize(data, &pos);
                }
            },
            6 => { // data_size (varint)
                if (wire_type == 0) {
                    info.data_size = readVarintAtUsize(data, &pos);
                }
            },
            7 => { // vector_dim (varint)
                if (wire_type == 0) {
                    info.vector_dim = @intCast(readVarintAtUsize(data, &pos));
                }
            },
            else => {
                // Skip unknown field
                if (wire_type == 0) {
                    _ = readVarintAtUsize(data, &pos);
                } else if (wire_type == 1) {
                    pos += 8;
                } else if (wire_type == 2) {
                    const len = readVarintAtUsize(data, &pos);
                    pos += len;
                } else if (wire_type == 5) {
                    pos += 4;
                }
            },
        }
    }
}

fn readVarintAtUsize(data: [*]const u8, pos: *usize) usize {
    var value: usize = 0;
    var shift: u5 = 0;

    while (true) {
        const byte = data[pos.*];
        pos.* += 1;
        value |= @as(usize, byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) break;
        shift += 7;
    }

    return value;
}

/// Get number of columns in loaded fragment
export fn fragmentGetColumnCount() u32 {
    return reader_num_columns;
}

/// Get row count from loaded fragment
export fn fragmentGetRowCount() u64 {
    if (reader_num_columns == 0) return 0;
    return reader_columns[0].row_count;
}

/// Get column name (returns length, writes to out_ptr)
export fn fragmentGetColumnName(col_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
    if (col_idx >= reader_num_columns) return 0;
    const info = &reader_columns[col_idx];
    const copy_len = @min(info.name_len, max_len);
    @memcpy(out_ptr[0..copy_len], info.name[0..copy_len]);
    return copy_len;
}

/// Get column type (returns length, writes to out_ptr)
export fn fragmentGetColumnType(col_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
    if (col_idx >= reader_num_columns) return 0;
    const info = &reader_columns[col_idx];
    const copy_len = @min(info.type_len, max_len);
    @memcpy(out_ptr[0..copy_len], info.col_type[0..copy_len]);
    return copy_len;
}

/// Get column vector dimension (0 if not a vector)
export fn fragmentGetColumnVectorDim(col_idx: u32) u32 {
    if (col_idx >= reader_num_columns) return 0;
    return reader_columns[col_idx].vector_dim;
}

/// Read int64 column data
export fn fragmentReadInt64(col_idx: u32, out_ptr: [*]i64, max_count: usize) usize {
    if (col_idx >= reader_num_columns) return 0;
    const data = reader_data orelse return 0;
    const info = &reader_columns[col_idx];

    const count: usize = @intCast(@min(info.row_count, max_count));
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const offset: usize = @intCast(info.data_offset + i * 8);
        out_ptr[i] = std.mem.readInt(i64, data[offset..][0..8], .little);
    }
    return count;
}

/// Read int32 column data
export fn fragmentReadInt32(col_idx: u32, out_ptr: [*]i32, max_count: usize) usize {
    if (col_idx >= reader_num_columns) return 0;
    const data = reader_data orelse return 0;
    const info = &reader_columns[col_idx];

    const count: usize = @intCast(@min(info.row_count, max_count));
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const offset: usize = @intCast(info.data_offset + i * 4);
        out_ptr[i] = std.mem.readInt(i32, data[offset..][0..4], .little);
    }
    return count;
}

/// Read float64 column data
export fn fragmentReadFloat64(col_idx: u32, out_ptr: [*]f64, max_count: usize) usize {
    if (col_idx >= reader_num_columns) return 0;
    const data = reader_data orelse return 0;
    const info = &reader_columns[col_idx];

    const count: usize = @intCast(@min(info.row_count, max_count));
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const offset: usize = @intCast(info.data_offset + i * 8);
        const bits = std.mem.readInt(u64, data[offset..][0..8], .little);
        out_ptr[i] = @bitCast(bits);
    }
    return count;
}

/// Read float32 column data
export fn fragmentReadFloat32(col_idx: u32, out_ptr: [*]f32, max_count: usize) usize {
    if (col_idx >= reader_num_columns) return 0;
    const data = reader_data orelse return 0;
    const info = &reader_columns[col_idx];

    const count: usize = @intCast(@min(info.row_count, max_count));
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const offset: usize = @intCast(info.data_offset + i * 4);
        const bits = std.mem.readInt(u32, data[offset..][0..4], .little);
        out_ptr[i] = @bitCast(bits);
    }
    return count;
}

/// Read bool column data (unpacked from bits)
export fn fragmentReadBool(col_idx: u32, out_ptr: [*]u8, max_count: usize) usize {
    if (col_idx >= reader_num_columns) return 0;
    const data = reader_data orelse return 0;
    const info = &reader_columns[col_idx];

    const count: usize = @intCast(@min(info.row_count, max_count));
    const base_offset: usize = @intCast(info.data_offset);
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const byte_idx = i / 8;
        const bit_idx: u3 = @intCast(i % 8);
        const byte = data[base_offset + byte_idx];
        out_ptr[i] = if ((byte & (@as(u8, 1) << bit_idx)) != 0) 1 else 0;
    }
    return count;
}

/// Get string at index - returns length, writes to out_ptr
export fn fragmentReadStringAt(col_idx: u32, row_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
    if (col_idx >= reader_num_columns) return 0;
    const data = reader_data orelse return 0;
    const info = &reader_columns[col_idx];

    if (row_idx >= info.row_count) return 0;

    // String layout: [string_data][offsets]
    // offsets are at the end: (row_count + 1) * 4 bytes
    const offsets_size: usize = @intCast((info.row_count + 1) * 4);
    const offsets_start: usize = @intCast(info.data_offset + info.data_size - offsets_size);
    const data_start: usize = @intCast(info.data_offset);

    const start_offset = std.mem.readInt(u32, data[offsets_start + row_idx * 4 ..][0..4], .little);
    const end_offset = std.mem.readInt(u32, data[offsets_start + (row_idx + 1) * 4 ..][0..4], .little);

    const str_len = end_offset - start_offset;
    const copy_len = @min(str_len, max_len);

    @memcpy(out_ptr[0..copy_len], data[data_start + start_offset ..][0..copy_len]);
    return copy_len;
}

/// Get string length at index (useful for allocation)
export fn fragmentGetStringLength(col_idx: u32, row_idx: u32) usize {
    if (col_idx >= reader_num_columns) return 0;
    const data = reader_data orelse return 0;
    const info = &reader_columns[col_idx];

    if (row_idx >= info.row_count) return 0;

    const offsets_size: usize = @intCast((info.row_count + 1) * 4);
    const offsets_start: usize = @intCast(info.data_offset + info.data_size - offsets_size);

    const start_offset = std.mem.readInt(u32, data[offsets_start + row_idx * 4 ..][0..4], .little);
    const end_offset = std.mem.readInt(u32, data[offsets_start + (row_idx + 1) * 4 ..][0..4], .little);

    return end_offset - start_offset;
}

/// Read vector at index - returns number of floats written
export fn fragmentReadVectorAt(col_idx: u32, row_idx: u32, out_ptr: [*]f32, max_floats: usize) usize {
    if (col_idx >= reader_num_columns) return 0;
    const data = reader_data orelse return 0;
    const info = &reader_columns[col_idx];

    if (row_idx >= info.row_count) return 0;
    if (info.vector_dim == 0) return 0;

    const dim = info.vector_dim;
    const copy_count: usize = @min(dim, max_floats);
    const base_offset: usize = @intCast(info.data_offset + row_idx * dim * 4);

    var i: usize = 0;
    while (i < copy_count) : (i += 1) {
        const bits = std.mem.readInt(u32, data[base_offset + i * 4 ..][0..4], .little);
        out_ptr[i] = @bitCast(bits);
    }
    return copy_count;
}
