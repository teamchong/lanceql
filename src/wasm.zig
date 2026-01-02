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
const lance_writer = @import("wasm/lance_writer.zig");
const fragment_reader = @import("wasm/fragment_reader.zig");

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
    _ = lance_writer;
    _ = fragment_reader;
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

