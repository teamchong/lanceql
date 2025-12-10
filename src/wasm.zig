//! WASM entry point for LanceQL.
//!
//! This module provides exported functions that can be called from JavaScript
//! in the browser. Uses direct byte manipulation for WASM compatibility.

const std = @import("std");

// ============================================================================
// Constants
// ============================================================================

const FOOTER_SIZE: usize = 40;
const LANCE_MAGIC = "LANC";

// ============================================================================
// Simple bump allocator for WASM (no std.heap.wasm_allocator dependency)
// ============================================================================

var heap: [1024 * 1024]u8 = undefined; // 1MB heap
var heap_offset: usize = 0;

fn wasmAlloc(len: usize) ?[*]u8 {
    // Get current address and align it to 8 bytes
    const heap_base = @intFromPtr(&heap[0]);
    const current_addr = heap_base + heap_offset;
    const aligned_addr = (current_addr + 7) & ~@as(usize, 7);
    const padding = aligned_addr - current_addr;

    heap_offset += padding;

    const aligned_len = (len + 7) & ~@as(usize, 7); // 8-byte align length too
    if (heap_offset + aligned_len > heap.len) return null;
    const ptr: [*]u8 = @ptrCast(&heap[heap_offset]);
    heap_offset += aligned_len;
    return ptr;
}

fn wasmReset() void {
    heap_offset = 0;
}

// ============================================================================
// Global state
// ============================================================================

var file_data: ?[]const u8 = null;
var num_columns: u32 = 0;
var column_meta_offsets_start: u64 = 0;

// ============================================================================
// Exported Memory Management
// ============================================================================

export fn alloc(len: usize) ?[*]u8 {
    return wasmAlloc(len);
}

export fn free(ptr: [*]u8, len: usize) void {
    _ = ptr;
    _ = len;
    // Bump allocator doesn't support individual frees
}

export fn resetHeap() void {
    wasmReset();
    file_data = null;
    num_columns = 0;
}

// ============================================================================
// Footer Parsing Helpers
// ============================================================================

fn readU64LE(data: []const u8, offset: usize) u64 {
    if (offset + 8 > data.len) return 0;
    return std.mem.readInt(u64, data[offset..][0..8], .little);
}

fn readU32LE(data: []const u8, offset: usize) u32 {
    if (offset + 4 > data.len) return 0;
    return std.mem.readInt(u32, data[offset..][0..4], .little);
}

fn readU16LE(data: []const u8, offset: usize) u16 {
    if (offset + 2 > data.len) return 0;
    return std.mem.readInt(u16, data[offset..][0..2], .little);
}

fn readI64LE(data: []const u8, offset: usize) i64 {
    if (offset + 8 > data.len) return 0;
    return std.mem.readInt(i64, data[offset..][0..8], .little);
}

fn readI32LE(data: []const u8, offset: usize) i32 {
    if (offset + 4 > data.len) return 0;
    return std.mem.readInt(i32, data[offset..][0..4], .little);
}

fn readI16LE(data: []const u8, offset: usize) i16 {
    if (offset + 2 > data.len) return 0;
    return std.mem.readInt(i16, data[offset..][0..2], .little);
}

fn readI8(data: []const u8, offset: usize) i8 {
    if (offset >= data.len) return 0;
    return @bitCast(data[offset]);
}

fn readU8(data: []const u8, offset: usize) u8 {
    if (offset >= data.len) return 0;
    return data[offset];
}

fn readF64LE(data: []const u8, offset: usize) f64 {
    const bits = readU64LE(data, offset);
    return @bitCast(bits);
}

fn readF32LE(data: []const u8, offset: usize) f32 {
    const bits = readU32LE(data, offset);
    return @bitCast(bits);
}

// ============================================================================
// Footer Parsing
// ============================================================================

export fn isValidLanceFile(data: [*]const u8, len: usize) u32 {
    if (len < FOOTER_SIZE) return 0;

    // Check magic at end
    const magic_offset = len - 4;
    if (data[magic_offset] != 'L' or data[magic_offset + 1] != 'A' or
        data[magic_offset + 2] != 'N' or data[magic_offset + 3] != 'C')
    {
        return 0;
    }
    return 1;
}

export fn parseFooterGetColumns(data: [*]const u8, len: usize) u32 {
    if (isValidLanceFile(data, len) == 0) return 0;
    const footer_start = len - FOOTER_SIZE;
    return readU32LE(data[0..len], footer_start + 28);
}

export fn parseFooterGetMajorVersion(data: [*]const u8, len: usize) u16 {
    if (isValidLanceFile(data, len) == 0) return 0;
    const footer_start = len - FOOTER_SIZE;
    return readU16LE(data[0..len], footer_start + 32);
}

export fn parseFooterGetMinorVersion(data: [*]const u8, len: usize) u16 {
    if (isValidLanceFile(data, len) == 0) return 0;
    const footer_start = len - FOOTER_SIZE;
    return readU16LE(data[0..len], footer_start + 34);
}

export fn getColumnMetaStart(data: [*]const u8, len: usize) u64 {
    if (isValidLanceFile(data, len) == 0) return 0;
    const footer_start = len - FOOTER_SIZE;
    return readU64LE(data[0..len], footer_start + 0);
}

export fn getColumnMetaOffsetsStart(data: [*]const u8, len: usize) u64 {
    if (isValidLanceFile(data, len) == 0) return 0;
    const footer_start = len - FOOTER_SIZE;
    return readU64LE(data[0..len], footer_start + 8);
}

export fn getVersion() u32 {
    return 0x000100; // v0.1.0
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

    return 1;
}

export fn closeFile() void {
    file_data = null;
    num_columns = 0;
    column_meta_offsets_start = 0;
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

export fn readInt64Column(col_idx: u32, out_ptr: [*]i64, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const count = @min(buf_size / 8, max_len);
    for (0..count) |i| {
        out_ptr[i] = readI64LE(data, buf_start + i * 8);
    }
    return count;
}

export fn readFloat64Column(col_idx: u32, out_ptr: [*]f64, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const count = @min(buf_size / 8, max_len);
    for (0..count) |i| {
        out_ptr[i] = readF64LE(data, buf_start + i * 8);
    }
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
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const row_count = buf_size / 8;
    var out_count: usize = 0;

    for (0..row_count) |i| {
        if (out_count >= max_indices) break;

        const col_val = readI64LE(data, buf_start + i * 8);
        const matches = switch (op) {
            0 => col_val == value, // eq
            1 => col_val != value, // ne
            2 => col_val < value, // lt
            3 => col_val <= value, // le
            4 => col_val > value, // gt
            5 => col_val >= value, // ge
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
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const row_count = buf_size / 8;
    var out_count: usize = 0;

    for (0..row_count) |i| {
        if (out_count >= max_indices) break;

        const col_val = readF64LE(data, buf_start + i * 8);
        const matches = switch (op) {
            0 => col_val == value, // eq
            1 => col_val != value, // ne
            2 => col_val < value, // lt
            3 => col_val <= value, // le
            4 => col_val > value, // gt
            5 => col_val >= value, // ge
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
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const max_idx = buf_size / 8;

    for (0..num_indices) |i| {
        const idx = indices[i];
        if (idx >= max_idx) {
            out_ptr[i] = 0;
        } else {
            out_ptr[i] = readI64LE(data, buf_start + idx * 8);
        }
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
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const max_idx = buf_size / 8;

    for (0..num_indices) |i| {
        const idx = indices[i];
        if (idx >= max_idx) {
            out_ptr[i] = 0;
        } else {
            out_ptr[i] = readF64LE(data, buf_start + idx * 8);
        }
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

/// Read int32 column
export fn readInt32Column(col_idx: u32, out_ptr: [*]i32, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const count = @min(buf_size / 4, max_len);
    for (0..count) |i| {
        out_ptr[i] = readI32LE(data, buf_start + i * 4);
    }
    return count;
}

/// Read int16 column
export fn readInt16Column(col_idx: u32, out_ptr: [*]i16, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const count = @min(buf_size / 2, max_len);
    for (0..count) |i| {
        out_ptr[i] = readI16LE(data, buf_start + i * 2);
    }
    return count;
}

/// Read int8 column
export fn readInt8Column(col_idx: u32, out_ptr: [*]i8, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const count = @min(buf_size, max_len);
    for (0..count) |i| {
        out_ptr[i] = readI8(data, buf_start + i);
    }
    return count;
}

/// Read uint64 column
export fn readUint64Column(col_idx: u32, out_ptr: [*]u64, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const count = @min(buf_size / 8, max_len);
    for (0..count) |i| {
        out_ptr[i] = readU64LE(data, buf_start + i * 8);
    }
    return count;
}

/// Read uint32 column
export fn readUint32Column(col_idx: u32, out_ptr: [*]u32, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const count = @min(buf_size / 4, max_len);
    for (0..count) |i| {
        out_ptr[i] = readU32LE(data, buf_start + i * 4);
    }
    return count;
}

/// Read uint16 column
export fn readUint16Column(col_idx: u32, out_ptr: [*]u16, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const count = @min(buf_size / 2, max_len);
    for (0..count) |i| {
        out_ptr[i] = readU16LE(data, buf_start + i * 2);
    }
    return count;
}

/// Read uint8 column
export fn readUint8Column(col_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const count = @min(buf_size, max_len);
    @memcpy(out_ptr[0..count], data[buf_start..][0..count]);
    return count;
}

/// Read float32 column
export fn readFloat32Column(col_idx: u32, out_ptr: [*]f32, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const count = @min(buf_size / 4, max_len);
    for (0..count) |i| {
        out_ptr[i] = readF32LE(data, buf_start + i * 4);
    }
    return count;
}

/// Read boolean column (stored as bit-packed in Lance)
export fn readBoolColumn(col_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    // Boolean values are bit-packed (8 values per byte)
    const num_bools: usize = @intCast(info.rows);
    const count = @min(num_bools, max_len);

    for (0..count) |i| {
        const byte_idx = i / 8;
        const bit_idx: u3 = @intCast(i % 8);
        if (byte_idx < buf_size) {
            const byte = data[buf_start + byte_idx];
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
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const max_idx = buf_size / 4;

    for (0..num_indices) |i| {
        const idx = indices[i];
        if (idx >= max_idx) {
            out_ptr[i] = 0;
        } else {
            out_ptr[i] = readI32LE(data, buf_start + idx * 4);
        }
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
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const max_idx = buf_size / 4;

    for (0..num_indices) |i| {
        const idx = indices[i];
        if (idx >= max_idx) {
            out_ptr[i] = 0;
        } else {
            out_ptr[i] = readF32LE(data, buf_start + idx * 4);
        }
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
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    for (0..num_indices) |i| {
        const idx = indices[i];
        if (idx >= buf_size) {
            out_ptr[i] = 0;
        } else {
            out_ptr[i] = data[buf_start + idx];
        }
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
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const num_bools: usize = @intCast(info.rows);

    for (0..num_indices) |i| {
        const idx = indices[i];
        if (idx >= num_bools) {
            out_ptr[i] = 0;
        } else {
            const byte_idx = idx / 8;
            const bit_idx: u3 = @intCast(idx % 8);
            if (byte_idx < buf_size) {
                const byte = data[buf_start + byte_idx];
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
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const row_count = buf_size / 8;
    var sum: i64 = 0;

    for (0..row_count) |i| {
        sum += readI64LE(data, buf_start + i * 8);
    }

    return sum;
}

/// Sum float64 column
export fn sumFloat64Column(col_idx: u32) f64 {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const row_count = buf_size / 8;
    var sum: f64 = 0;

    for (0..row_count) |i| {
        sum += readF64LE(data, buf_start + i * 8);
    }

    return sum;
}

/// Min int64 column
export fn minInt64Column(col_idx: u32) i64 {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const row_count = buf_size / 8;
    if (row_count == 0) return 0;

    var min_val: i64 = readI64LE(data, buf_start);
    for (1..row_count) |i| {
        const val = readI64LE(data, buf_start + i * 8);
        if (val < min_val) min_val = val;
    }

    return min_val;
}

/// Max int64 column
export fn maxInt64Column(col_idx: u32) i64 {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const row_count = buf_size / 8;
    if (row_count == 0) return 0;

    var max_val: i64 = readI64LE(data, buf_start);
    for (1..row_count) |i| {
        const val = readI64LE(data, buf_start + i * 8);
        if (val > max_val) max_val = val;
    }

    return max_val;
}

/// Average float64 column
export fn avgFloat64Column(col_idx: u32) f64 {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return 0;

    const row_count = buf_size / 8;
    if (row_count == 0) return 0;

    var sum: f64 = 0;
    for (0..row_count) |i| {
        sum += readF64LE(data, buf_start + i * 8);
    }

    return sum / @as(f64, @floatFromInt(row_count));
}

// ============================================================================
// String Column Support
// ============================================================================

/// Get string buffer info from column metadata (for variable-length data)
/// Lance stores strings with:
///   - buffer[0]: offsets (int32 or int64)
///   - buffer[1]: data (UTF-8 bytes)
fn getStringBufferInfo(col_meta: []const u8) struct {
    offsets_start: u64,
    offsets_size: u64,
    data_start: u64,
    data_size: u64,
    rows: u64,
} {
    var pos: usize = 0;
    var buffer_offsets: [2]u64 = .{ 0, 0 };
    var buffer_sizes: [2]u64 = .{ 0, 0 };
    var page_rows: u64 = 0;
    var buf_idx: usize = 0;

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
                                buf_idx = 0;
                                while (pos < packed_end and buf_idx < 2) {
                                    buffer_offsets[buf_idx] = readVarint(col_meta, &pos);
                                    buf_idx += 1;
                                }
                                pos = packed_end;
                            } else {
                                if (buf_idx < 2) {
                                    buffer_offsets[buf_idx] = readVarint(col_meta, &pos);
                                    buf_idx += 1;
                                }
                            }
                        },
                        2 => { // buffer_sizes (packed repeated uint64)
                            if (page_wire == 2) {
                                const packed_len = readVarint(col_meta, &pos);
                                const packed_end = pos + @as(usize, @intCast(packed_len));
                                buf_idx = 0;
                                while (pos < packed_end and buf_idx < 2) {
                                    buffer_sizes[buf_idx] = readVarint(col_meta, &pos);
                                    buf_idx += 1;
                                }
                                pos = packed_end;
                            } else {
                                if (buf_idx < 2) {
                                    buffer_sizes[buf_idx] = readVarint(col_meta, &pos);
                                    buf_idx += 1;
                                }
                            }
                        },
                        3 => { // length (rows)
                            page_rows = readVarint(col_meta, &pos);
                        },
                        else => {
                            if (page_wire == 0) {
                                _ = readVarint(col_meta, &pos);
                            } else if (page_wire == 2) {
                                const skip_len = readVarint(col_meta, &pos);
                                pos += @as(usize, @intCast(skip_len));
                            } else if (page_wire == 5) {
                                pos += 4;
                            } else if (page_wire == 1) {
                                pos += 8;
                            }
                        },
                    }
                }
                return .{
                    .offsets_start = buffer_offsets[0],
                    .offsets_size = buffer_sizes[0],
                    .data_start = buffer_offsets[1],
                    .data_size = buffer_sizes[1],
                    .rows = page_rows,
                };
            },
            else => {
                if (wire_type == 0) {
                    _ = readVarint(col_meta, &pos);
                } else if (wire_type == 2) {
                    const skip_len = readVarint(col_meta, &pos);
                    pos += @as(usize, @intCast(skip_len));
                } else if (wire_type == 5) {
                    pos += 4;
                } else if (wire_type == 1) {
                    pos += 8;
                }
            },
        }
    }

    return .{
        .offsets_start = 0,
        .offsets_size = 0,
        .data_start = 0,
        .data_size = 0,
        .rows = 0,
    };
}

/// Get number of strings in column
/// Returns 0 if not a string column (string columns have 2 buffers: offsets + data)
export fn getStringCount(col_idx: u32) u64 {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getStringBufferInfo(col_meta);

    // String columns have 2 buffers (offsets + data), non-string columns have 1
    // If data_size is 0, this is not a string column
    if (info.data_size == 0) return 0;

    return info.rows;
}

/// Read a single string at index into output buffer
/// Returns actual string length (may exceed out_max if truncated)
export fn readStringAt(col_idx: u32, row_idx: u32, out_ptr: [*]u8, out_max: usize) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getStringBufferInfo(col_meta);

    if (info.offsets_size == 0 or info.data_size == 0) return 0;
    if (row_idx >= info.rows) return 0;

    const offsets_start: usize = @intCast(info.offsets_start);
    const data_start: usize = @intCast(info.data_start);

    // Lance v2 uses N offsets for N strings (end positions, not N+1 start/end pairs)
    // Check if using 32-bit or 64-bit offsets: offsets_size / rows
    const offset_size = info.offsets_size / info.rows;
    if (offset_size != 4 and offset_size != 8) return 0;

    var str_start: usize = 0;
    var str_end: usize = 0;

    if (offset_size == 4) {
        // 32-bit offsets - each offset is the END position of that string
        str_end = readU32LE(data, offsets_start + row_idx * 4);
        if (row_idx > 0) {
            str_start = readU32LE(data, offsets_start + (row_idx - 1) * 4);
        }
    } else {
        // 64-bit offsets
        str_end = @intCast(readU64LE(data, offsets_start + row_idx * 8));
        if (row_idx > 0) {
            str_start = @intCast(readU64LE(data, offsets_start + (row_idx - 1) * 8));
        }
    }

    if (str_end < str_start) return 0;
    const str_len = str_end - str_start;

    // Check bounds
    if (data_start + str_end > data.len) return 0;

    // Copy string to output buffer
    const copy_len = @min(str_len, out_max);
    const str_data = data[data_start + str_start ..][0..copy_len];
    @memcpy(out_ptr[0..copy_len], str_data);

    return str_len;
}

/// Read multiple strings at indices
/// Returns total bytes written to out_ptr
/// out_lengths receives the length of each string
export fn readStringsAtIndices(
    col_idx: u32,
    indices: [*]const u32,
    num_indices: usize,
    out_ptr: [*]u8,
    out_max: usize,
    out_lengths: [*]u32,
) usize {
    var total_written: usize = 0;

    for (0..num_indices) |i| {
        const remaining = if (total_written < out_max) out_max - total_written else 0;
        const len = readStringAt(col_idx, indices[i], out_ptr + total_written, remaining);
        out_lengths[i] = @intCast(len);
        total_written += @min(len, remaining);
    }

    return total_written;
}

/// Allocate string buffer
export fn allocStringBuffer(size: usize) ?[*]u8 {
    return wasmAlloc(size);
}

/// Allocate u32 buffer for lengths
export fn allocU32Buffer(count: usize) ?[*]u32 {
    const ptr = wasmAlloc(count * 4) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

// ============================================================================
// Vector Column Support (Fixed-size float arrays for embeddings)
// ============================================================================

/// Get vector info from column: dimension and count
/// Vectors are stored as fixed-size arrays of float32
export fn getVectorInfo(col_idx: u32) u64 {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    // Return packed: high 32 bits = rows, low 32 bits = estimated dimension
    // Dimension = buffer_size / (rows * 4) for float32
    if (info.rows == 0) return 0;
    const dim = info.size / (info.rows * 4);
    return (info.rows << 32) | dim;
}

/// Read a single vector at index
/// Returns number of floats written
export fn readVectorAt(
    col_idx: u32,
    row_idx: u32,
    out_ptr: [*]f32,
    max_dim: usize,
) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    if (info.rows == 0) return 0;
    if (row_idx >= info.rows) return 0;

    const dim: usize = @intCast(info.size / (info.rows * 4));
    if (dim == 0) return 0;

    const buf_start: usize = @intCast(info.offset);
    const vec_start = buf_start + @as(usize, row_idx) * dim * 4;

    const actual_dim = @min(dim, max_dim);
    for (0..actual_dim) |i| {
        out_ptr[i] = readF32LE(data, vec_start + i * 4);
    }

    return actual_dim;
}

/// Allocate float32 buffer for vectors
export fn allocFloat32Buffer(count: usize) ?[*]f32 {
    const ptr = wasmAlloc(count * 4) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

/// Compute cosine similarity between two vectors
/// Returns similarity score (-1 to 1, higher is more similar)
export fn cosineSimilarity(
    vec_a: [*]const f32,
    vec_b: [*]const f32,
    dim: usize,
) f32 {
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    for (0..dim) |i| {
        const a = vec_a[i];
        const b = vec_b[i];
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 0;
    return dot / denom;
}

// ============================================================================
// SIMD-Optimized Vector Operations
// ============================================================================

const Vec4 = @Vector(4, f32);

/// SIMD dot product for f32 vectors (4-wide)
fn simdDotProduct(a_ptr: [*]const f32, b_ptr: [*]const f32, dim: usize) f32 {
    var sum: Vec4 = @splat(0);
    var i: usize = 0;

    // Process 4 elements at a time
    while (i + 4 <= dim) : (i += 4) {
        const va: Vec4 = .{ a_ptr[i], a_ptr[i + 1], a_ptr[i + 2], a_ptr[i + 3] };
        const vb: Vec4 = .{ b_ptr[i], b_ptr[i + 1], b_ptr[i + 2], b_ptr[i + 3] };
        sum += va * vb;
    }

    // Horizontal sum
    var result = @reduce(.Add, sum);

    // Handle remainder
    while (i < dim) : (i += 1) {
        result += a_ptr[i] * b_ptr[i];
    }

    return result;
}

/// SIMD L2 norm squared
fn simdNormSquared(ptr: [*]const f32, dim: usize) f32 {
    var sum: Vec4 = @splat(0);
    var i: usize = 0;

    while (i + 4 <= dim) : (i += 4) {
        const v: Vec4 = .{ ptr[i], ptr[i + 1], ptr[i + 2], ptr[i + 3] };
        sum += v * v;
    }

    var result = @reduce(.Add, sum);

    while (i < dim) : (i += 1) {
        result += ptr[i] * ptr[i];
    }

    return result;
}

/// SIMD cosine similarity for pre-normalized vectors (just dot product)
export fn simdCosineSimilarityNormalized(
    vec_a: [*]const f32,
    vec_b: [*]const f32,
    dim: usize,
) f32 {
    return simdDotProduct(vec_a, vec_b, dim);
}

/// SIMD cosine similarity for un-normalized vectors
export fn simdCosineSimilarity(
    vec_a: [*]const f32,
    vec_b: [*]const f32,
    dim: usize,
) f32 {
    const dot = simdDotProduct(vec_a, vec_b, dim);
    const norm_a = @sqrt(simdNormSquared(vec_a, dim));
    const norm_b = @sqrt(simdNormSquared(vec_b, dim));
    const denom = norm_a * norm_b;
    if (denom == 0) return 0;
    return dot / denom;
}

/// Batch compute similarities between query and multiple vectors
/// Much faster than calling simdCosineSimilarity in a loop
/// vectors_ptr: flattened array of [num_vectors * dim] f32
/// out_scores: array of [num_vectors] f32
export fn batchCosineSimilarity(
    query_ptr: [*]const f32,
    vectors_ptr: [*]const f32,
    dim: usize,
    num_vectors: usize,
    out_scores: [*]f32,
    normalized: u32,
) void {
    // Pre-compute query norm if not normalized
    const query_norm = if (normalized == 0) @sqrt(simdNormSquared(query_ptr, dim)) else 1.0;

    for (0..num_vectors) |i| {
        const vec_ptr = vectors_ptr + i * dim;
        const dot = simdDotProduct(query_ptr, vec_ptr, dim);

        if (normalized != 0) {
            // Vectors are L2-normalized, dot product = cosine similarity
            out_scores[i] = dot;
        } else {
            const vec_norm = @sqrt(simdNormSquared(vec_ptr, dim));
            const denom = query_norm * vec_norm;
            out_scores[i] = if (denom == 0) 0 else dot / denom;
        }
    }
}

/// Find top-k from pre-computed scores
/// Uses partial selection for better performance than full sort
fn findTopK(
    scores: [*]const f32,
    num_scores: usize,
    top_k: usize,
    out_indices: [*]u32,
    out_scores: [*]f32,
) usize {
    const actual_k = @min(top_k, num_scores);

    // Initialize with worst scores
    for (0..actual_k) |i| {
        out_indices[i] = 0;
        out_scores[i] = -2.0;
    }

    // Simple insertion sort into top-k (good for small k)
    for (0..num_scores) |i| {
        const score = scores[i];

        if (score > out_scores[actual_k - 1]) {
            // Find insertion point
            var insert_pos: usize = actual_k - 1;
            while (insert_pos > 0 and score > out_scores[insert_pos - 1]) {
                insert_pos -= 1;
            }

            // Shift elements down
            var j: usize = actual_k - 1;
            while (j > insert_pos) {
                out_indices[j] = out_indices[j - 1];
                out_scores[j] = out_scores[j - 1];
                j -= 1;
            }

            // Insert
            out_indices[insert_pos] = @intCast(i);
            out_scores[insert_pos] = score;
        }
    }

    return actual_k;
}

/// Find top-k most similar vectors to query (SIMD optimized)
/// Returns number of results written
/// out_indices: row indices of top matches
/// out_scores: similarity scores
export fn vectorSearchTopK(
    col_idx: u32,
    query_ptr: [*]const f32,
    query_dim: usize,
    top_k: usize,
    out_indices: [*]u32,
    out_scores: [*]f32,
) usize {
    const data = file_data orelse return 0;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return 0;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return 0;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getPageBufferInfo(col_meta);

    if (info.rows == 0) return 0;

    const dim: usize = @intCast(info.size / (info.rows * 4));
    if (dim != query_dim) return 0;

    const buf_start: usize = @intCast(info.offset);
    const num_rows: usize = @intCast(info.rows);
    const actual_k = @min(top_k, num_rows);

    // Initialize with worst scores
    for (0..actual_k) |i| {
        out_indices[i] = 0;
        out_scores[i] = -2.0;
    }

    // Pre-compute query norm (assume vectors may not be normalized)
    const query_norm = @sqrt(simdNormSquared(query_ptr, query_dim));

    // Scan all vectors using SIMD
    for (0..num_rows) |row| {
        const vec_start = buf_start + row * dim * 4;

        // Get pointer to vector data (may be unaligned)
        const vec_ptr: [*]const f32 = @ptrCast(@alignCast(data.ptr + vec_start));

        // SIMD dot product
        const dot = simdDotProduct(query_ptr, vec_ptr, dim);
        const vec_norm = @sqrt(simdNormSquared(vec_ptr, dim));
        const denom = query_norm * vec_norm;
        const score: f32 = if (denom == 0) 0 else dot / denom;

        // Insert into top-k if better than worst
        if (score > out_scores[actual_k - 1]) {
            var insert_pos: usize = actual_k - 1;
            while (insert_pos > 0 and score > out_scores[insert_pos - 1]) {
                insert_pos -= 1;
            }

            var j: usize = actual_k - 1;
            while (j > insert_pos) {
                out_indices[j] = out_indices[j - 1];
                out_scores[j] = out_scores[j - 1];
                j -= 1;
            }

            out_indices[insert_pos] = @intCast(row);
            out_scores[insert_pos] = score;
        }
    }

    return actual_k;
}

/// Vector search on raw buffer (for worker-based processing)
/// Searches vectors directly from a provided buffer, not from file_data
/// normalized: 1 if vectors are L2-normalized (skip norm computation)
export fn vectorSearchBuffer(
    vectors_ptr: [*]const f32,
    num_vectors: usize,
    dim: usize,
    query_ptr: [*]const f32,
    top_k: usize,
    out_indices: [*]u32,
    out_scores: [*]f32,
    normalized: u32,
    start_index: u32,
) usize {
    const actual_k = @min(top_k, num_vectors);

    // Initialize with worst scores
    for (0..actual_k) |i| {
        out_indices[i] = 0;
        out_scores[i] = -2.0;
    }

    // Pre-compute query norm if not normalized
    const query_norm = if (normalized == 0) @sqrt(simdNormSquared(query_ptr, dim)) else 1.0;

    // Scan all vectors using SIMD
    for (0..num_vectors) |row| {
        const vec_ptr = vectors_ptr + row * dim;

        const dot = simdDotProduct(query_ptr, vec_ptr, dim);

        var score: f32 = undefined;
        if (normalized != 0) {
            // For L2-normalized vectors, dot product = cosine similarity
            score = dot;
        } else {
            const vec_norm = @sqrt(simdNormSquared(vec_ptr, dim));
            const denom = query_norm * vec_norm;
            score = if (denom == 0) 0 else dot / denom;
        }

        // Insert into top-k if better than worst
        if (score > out_scores[actual_k - 1]) {
            var insert_pos: usize = actual_k - 1;
            while (insert_pos > 0 and score > out_scores[insert_pos - 1]) {
                insert_pos -= 1;
            }

            var j: usize = actual_k - 1;
            while (j > insert_pos) {
                out_indices[j] = out_indices[j - 1];
                out_scores[j] = out_scores[j - 1];
                j -= 1;
            }

            // Store global index (start_index + local row)
            out_indices[insert_pos] = start_index + @as(u32, @intCast(row));
            out_scores[insert_pos] = score;
        }
    }

    return actual_k;
}

/// Merge multiple top-k results into final top-k
/// Used by main thread to combine results from workers
export fn mergeTopK(
    indices_arrays: [*]const [*]const u32,
    scores_arrays: [*]const [*]const f32,
    num_arrays: usize,
    k_per_array: usize,
    final_k: usize,
    out_indices: [*]u32,
    out_scores: [*]f32,
) usize {
    const actual_k = @min(final_k, num_arrays * k_per_array);

    // Initialize
    for (0..actual_k) |i| {
        out_indices[i] = 0;
        out_scores[i] = -2.0;
    }

    // Merge all results
    for (0..num_arrays) |arr_idx| {
        const indices = indices_arrays[arr_idx];
        const scores = scores_arrays[arr_idx];

        for (0..k_per_array) |i| {
            const score = scores[i];
            if (score <= -2.0) continue; // Skip invalid

            if (score > out_scores[actual_k - 1]) {
                var insert_pos: usize = actual_k - 1;
                while (insert_pos > 0 and score > out_scores[insert_pos - 1]) {
                    insert_pos -= 1;
                }

                var j: usize = actual_k - 1;
                while (j > insert_pos) {
                    out_indices[j] = out_indices[j - 1];
                    out_scores[j] = out_scores[j - 1];
                    j -= 1;
                }

                out_indices[insert_pos] = indices[i];
                out_scores[insert_pos] = score;
            }
        }
    }

    return actual_k;
}

// ============================================================================
// CLIP Text Encoder
// ============================================================================

// CLIP model constants (ViT-B/32)
const CLIP_VOCAB_SIZE: usize = 49408;
const CLIP_MAX_SEQ_LEN: usize = 77;
const CLIP_EMBED_DIM: usize = 512;
const CLIP_NUM_HEADS: usize = 8;
const CLIP_NUM_LAYERS: usize = 12;
const CLIP_MLP_DIM: usize = 2048;
const CLIP_HEAD_DIM: usize = CLIP_EMBED_DIM / CLIP_NUM_HEADS; // 64

// CLIP state
var clip_initialized: bool = false;
var clip_model_loaded: bool = false;

// Buffers for CLIP
var clip_text_buffer: [1024]u8 = undefined;
var clip_output_buffer: [CLIP_EMBED_DIM]f32 = undefined;

// Model weights storage (allocated from model buffer)
var clip_model_buffer: ?[*]u8 = null;
var clip_model_size: usize = 0;

// Weight tensor indices (set after model load)
var token_embedding_idx: ?usize = null;
var position_embedding_idx: ?usize = null;
var ln_final_weight_idx: ?usize = null;
var ln_final_bias_idx: ?usize = null;
var text_projection_idx: ?usize = null;

// Per-layer weight indices (12 layers)
var layer_ln1_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_ln1_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_q_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_q_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_k_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_k_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_v_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_v_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_out_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_out_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_ln2_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_ln2_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_mlp_fc1_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_mlp_fc1_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_mlp_fc2_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_mlp_fc2_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;

// Tokenizer vocab (simplified - just store raw tokens)
var vocab_tokens: ?[*]const u8 = null;
var vocab_offsets: ?[*]const u32 = null;
var vocab_count: usize = 0;

// Scratch buffers for inference (statically allocated)
var scratch_hidden: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;
var scratch_q: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;
var scratch_k: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;
var scratch_v: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;
var scratch_attn: [CLIP_NUM_HEADS * CLIP_MAX_SEQ_LEN * CLIP_MAX_SEQ_LEN]f32 = undefined;
var scratch_mlp: [CLIP_MAX_SEQ_LEN * CLIP_MLP_DIM]f32 = undefined;
var scratch_ln: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;

/// Initialize CLIP module
export fn clip_init() i32 {
    clip_initialized = true;
    clip_model_loaded = false;

    // Clear output buffer
    for (&clip_output_buffer) |*v| {
        v.* = 0;
    }

    return 0;
}

/// Get pointer to text input buffer
export fn clip_get_text_buffer() [*]u8 {
    return &clip_text_buffer;
}

/// Get size of text input buffer
export fn clip_get_text_buffer_size() usize {
    return clip_text_buffer.len;
}

/// Get pointer to output embedding buffer
export fn clip_get_output_buffer() [*]f32 {
    return &clip_output_buffer;
}

/// Get output embedding dimension
export fn clip_get_output_dim() usize {
    return CLIP_EMBED_DIM;
}

/// Allocate buffer for model weights using WASM memory growth
export fn clip_alloc_model_buffer(size: usize) usize {
    // For large allocations like CLIP model (100MB+), grow WASM memory directly
    // WASM pages are 64KB each
    const page_size: usize = 65536;
    const pages_needed = (size + page_size - 1) / page_size;

    // Get current memory size in pages
    const current_pages = @wasmMemorySize(0);
    const current_size = current_pages * page_size;

    // Try to grow memory
    const result = @wasmMemoryGrow(0, pages_needed);
    if (result == @as(usize, @bitCast(@as(isize, -1)))) {
        // Growth failed
        return 0;
    }

    // The new memory starts at the old end
    const ptr: [*]u8 = @ptrFromInt(current_size);
    clip_model_buffer = ptr;
    clip_model_size = size;
    return current_size;
}

/// Check if model weights are loaded
export fn clip_weights_loaded() i32 {
    return if (clip_model_loaded) 1 else 0;
}

// GGUF parsing helpers
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_UINT64: u32 = 10;

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;

// Tensor info storage
const MAX_TENSORS: usize = 256;
var tensor_names: [MAX_TENSORS][64]u8 = undefined;
var tensor_name_lens: [MAX_TENSORS]usize = undefined;
var tensor_offsets: [MAX_TENSORS]u64 = undefined;
var tensor_types: [MAX_TENSORS]u32 = undefined;
var tensor_dims: [MAX_TENSORS][4]u64 = undefined;
var tensor_n_dims: [MAX_TENSORS]u32 = undefined;
var n_tensors_loaded: usize = 0;
var gguf_data_offset: usize = 0;

// Tokenizer storage
var vocab_data: ?[*]const u8 = null;
var vocab_string_offsets: [CLIP_VOCAB_SIZE + 1]u32 = undefined;

fn ggufReadString(data: []const u8, pos: *usize) []const u8 {
    if (pos.* + 8 > data.len) return "";
    const len: usize = @intCast(readU64LE(data, pos.*));
    pos.* += 8;
    if (pos.* + len > data.len) return "";
    const str = data[pos.* .. pos.* + len];
    pos.* += len;
    return str;
}

fn ggufSkipValue(data: []const u8, pos: *usize, vtype: u32) void {
    switch (vtype) {
        GGUF_TYPE_STRING => {
            _ = ggufReadString(data, pos);
        },
        GGUF_TYPE_UINT32, GGUF_TYPE_FLOAT32 => {
            pos.* += 4;
        },
        GGUF_TYPE_BOOL => {
            pos.* += 1;
        },
        GGUF_TYPE_UINT64 => {
            pos.* += 8;
        },
        GGUF_TYPE_ARRAY => {
            if (pos.* + 12 > data.len) return;
            const atype = readU32LE(data, pos.*);
            pos.* += 4;
            const alen: usize = @intCast(readU64LE(data, pos.*));
            pos.* += 8;
            for (0..alen) |_| {
                ggufSkipValue(data, pos, atype);
            }
        },
        else => {},
    }
}

fn strEql(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| {
        if (ca != cb) return false;
    }
    return true;
}

fn findTensor(name: []const u8) ?usize {
    for (0..n_tensors_loaded) |i| {
        if (strEql(tensor_names[i][0..tensor_name_lens[i]], name)) {
            return i;
        }
    }
    return null;
}

fn getTensorF32Ptr(idx: usize) ?[*]const f32 {
    const model_data = clip_model_buffer orelse return null;
    const offset: usize = gguf_data_offset + @as(usize, @intCast(tensor_offsets[idx]));

    if (tensor_types[idx] == GGML_TYPE_F32) {
        return @ptrCast(@alignCast(model_data + offset));
    }
    return null;
}

fn getTensorF16Ptr(idx: usize) ?[*]const u16 {
    const model_data = clip_model_buffer orelse return null;
    const offset: usize = gguf_data_offset + @as(usize, @intCast(tensor_offsets[idx]));

    if (tensor_types[idx] == GGML_TYPE_F16) {
        return @ptrCast(@alignCast(model_data + offset));
    }
    return null;
}

fn getTensorPtr(idx: usize) ?[*]const u8 {
    const model_data = clip_model_buffer orelse return null;
    const offset: usize = gguf_data_offset + @as(usize, @intCast(tensor_offsets[idx]));
    return model_data + offset;
}

// Read a single weight value, handling F16 conversion
fn readWeight(idx: usize, i: usize) f32 {
    const model_data = clip_model_buffer orelse return 0;
    const offset: usize = gguf_data_offset + @as(usize, @intCast(tensor_offsets[idx]));

    if (tensor_types[idx] == GGML_TYPE_F16) {
        const ptr: [*]const u16 = @ptrCast(@alignCast(model_data + offset));
        return f16ToF32(ptr[i]);
    } else {
        const ptr: [*]const f32 = @ptrCast(@alignCast(model_data + offset));
        return ptr[i];
    }
}

fn getTensorF16AsF32(idx: usize, out: []f32) bool {
    const model_data = clip_model_buffer orelse return false;
    const offset: usize = gguf_data_offset + @as(usize, @intCast(tensor_offsets[idx]));

    if (tensor_types[idx] != GGML_TYPE_F16) return false;

    // Calculate total elements
    var total: usize = 1;
    for (0..tensor_n_dims[idx]) |d| {
        total *= @intCast(tensor_dims[idx][d]);
    }

    if (total > out.len) return false;

    // Convert F16 to F32
    const f16_ptr: [*]const u16 = @ptrCast(@alignCast(model_data + offset));
    for (0..total) |i| {
        out[i] = f16ToF32(f16_ptr[i]);
    }
    return true;
}

fn f16ToF32(h: u16) f32 {
    const sign: u32 = @as(u32, h >> 15) << 31;
    const exp: u32 = (h >> 10) & 0x1F;
    const mant: u32 = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            return @bitCast(sign);
        }
        // Subnormal
        var e: u32 = 0;
        var m = mant;
        while ((m & 0x400) == 0) {
            m <<= 1;
            e += 1;
        }
        const new_exp = (127 - 15 - e) << 23;
        const new_mant = (m & 0x3FF) << 13;
        return @bitCast(sign | new_exp | new_mant);
    } else if (exp == 31) {
        // Inf/NaN
        return @bitCast(sign | 0x7F800000 | (mant << 13));
    }

    const new_exp = (exp + 127 - 15) << 23;
    const new_mant = mant << 13;
    return @bitCast(sign | new_exp | new_mant);
}

/// Load GGUF model from buffer
export fn clip_load_model(size: usize) i32 {
    const model_data = clip_model_buffer orelse return -1;
    if (size < 128) return -2;

    const data = model_data[0..size];

    // Check GGUF magic
    if (data[0] != 'G' or data[1] != 'G' or data[2] != 'U' or data[3] != 'F') {
        return -3;
    }

    // GGUF version
    const version = readU32LE(data, 4);
    if (version < 2 or version > 3) {
        return -4;
    }

    // Number of tensors and metadata KV pairs
    const n_tensors: usize = @intCast(readU64LE(data, 8));
    const n_kv: usize = @intCast(readU64LE(data, 16));

    var pos: usize = 24;

    // Parse KV pairs to find tokenizer
    for (0..n_kv) |_| {
        const key = ggufReadString(data, &pos);
        if (pos + 4 > data.len) return -5;
        const vtype = readU32LE(data, pos);
        pos += 4;

        if (strEql(key, "tokenizer.ggml.tokens")) {
            // This is the vocab array
            if (vtype != GGUF_TYPE_ARRAY) {
                ggufSkipValue(data, &pos, vtype);
                continue;
            }
            if (pos + 12 > data.len) return -6;
            const atype = readU32LE(data, pos);
            pos += 4;
            const alen: usize = @intCast(readU64LE(data, pos));
            pos += 8;

            if (atype == GGUF_TYPE_STRING and alen <= CLIP_VOCAB_SIZE) {
                vocab_data = data.ptr + pos;
                vocab_count = alen;

                // Build string offset table
                var str_pos: u32 = 0;
                for (0..alen) |i| {
                    vocab_string_offsets[i] = str_pos;
                    const slen: u32 = @intCast(readU64LE(data, pos));
                    pos += 8;
                    str_pos += slen + 8; // Include length prefix
                    pos += slen;
                }
                vocab_string_offsets[alen] = str_pos;
            } else {
                ggufSkipValue(data, &pos, vtype);
            }
        } else {
            ggufSkipValue(data, &pos, vtype);
        }
    }

    // Parse tensor info
    n_tensors_loaded = @min(n_tensors, MAX_TENSORS);
    for (0..n_tensors_loaded) |i| {
        const name = ggufReadString(data, &pos);
        const name_len = @min(name.len, 63);
        @memcpy(tensor_names[i][0..name_len], name[0..name_len]);
        tensor_name_lens[i] = name_len;

        if (pos + 4 > data.len) return -7;
        const n_dims = readU32LE(data, pos);
        pos += 4;
        tensor_n_dims[i] = n_dims;

        for (0..@min(n_dims, 4)) |d| {
            if (pos + 8 > data.len) return -8;
            tensor_dims[i][d] = readU64LE(data, pos);
            pos += 8;
        }

        if (pos + 12 > data.len) return -9;
        tensor_types[i] = readU32LE(data, pos);
        pos += 4;
        tensor_offsets[i] = readU64LE(data, pos);
        pos += 8;
    }

    // Calculate data offset (aligned to 32 bytes)
    gguf_data_offset = (pos + 31) & ~@as(usize, 31);

    // Store tensor indices for key tensors
    token_embedding_idx = findTensor("t.token_embd.weight");
    position_embedding_idx = findTensor("t.position_embd.weight");
    ln_final_weight_idx = findTensor("t.post_ln.weight");
    ln_final_bias_idx = findTensor("t.post_ln.bias");
    text_projection_idx = findTensor("text_projection.weight");

    // Store per-layer tensor indices
    for (0..CLIP_NUM_LAYERS) |layer| {
        var name_buf: [64]u8 = undefined;

        const ln1_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ln1.weight", .{layer}) catch continue;
        layer_ln1_weight_idx[layer] = findTensor(ln1_w);

        const ln1_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ln1.bias", .{layer}) catch continue;
        layer_ln1_bias_idx[layer] = findTensor(ln1_b);

        const q_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_q.weight", .{layer}) catch continue;
        layer_attn_q_weight_idx[layer] = findTensor(q_w);

        const q_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_q.bias", .{layer}) catch continue;
        layer_attn_q_bias_idx[layer] = findTensor(q_b);

        const k_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_k.weight", .{layer}) catch continue;
        layer_attn_k_weight_idx[layer] = findTensor(k_w);

        const k_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_k.bias", .{layer}) catch continue;
        layer_attn_k_bias_idx[layer] = findTensor(k_b);

        const v_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_v.weight", .{layer}) catch continue;
        layer_attn_v_weight_idx[layer] = findTensor(v_w);

        const v_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_v.bias", .{layer}) catch continue;
        layer_attn_v_bias_idx[layer] = findTensor(v_b);

        const out_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_out.weight", .{layer}) catch continue;
        layer_attn_out_weight_idx[layer] = findTensor(out_w);

        const out_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_out.bias", .{layer}) catch continue;
        layer_attn_out_bias_idx[layer] = findTensor(out_b);

        const ln2_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ln2.weight", .{layer}) catch continue;
        layer_ln2_weight_idx[layer] = findTensor(ln2_w);

        const ln2_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ln2.bias", .{layer}) catch continue;
        layer_ln2_bias_idx[layer] = findTensor(ln2_b);

        const fc1_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ffn_down.weight", .{layer}) catch continue;
        layer_mlp_fc1_weight_idx[layer] = findTensor(fc1_w);

        const fc1_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ffn_down.bias", .{layer}) catch continue;
        layer_mlp_fc1_bias_idx[layer] = findTensor(fc1_b);

        const fc2_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ffn_up.weight", .{layer}) catch continue;
        layer_mlp_fc2_weight_idx[layer] = findTensor(fc2_w);

        const fc2_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ffn_up.bias", .{layer}) catch continue;
        layer_mlp_fc2_bias_idx[layer] = findTensor(fc2_b);
    }

    // Verify essential weights loaded
    if (token_embedding_idx == null) return -10;
    if (position_embedding_idx == null) return -11;
    if (ln_final_weight_idx == null) return -12;
    if (text_projection_idx == null) return -13;

    clip_model_loaded = true;
    return 0;
}

// Layer normalization
fn layerNorm(input: []const f32, weight: []const f32, bias: []const f32, output: []f32) void {
    const dim = weight.len;
    const eps: f32 = 1e-5;

    // Compute mean
    var mean: f32 = 0;
    for (input[0..dim]) |v| {
        mean += v;
    }
    mean /= @floatFromInt(dim);

    // Compute variance
    var variance: f32 = 0;
    for (input[0..dim]) |v| {
        const diff = v - mean;
        variance += diff * diff;
    }
    variance /= @floatFromInt(dim);

    // Normalize
    const inv_std = 1.0 / @sqrt(variance + eps);
    for (0..dim) |i| {
        output[i] = (input[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// Matrix multiply: output[M,N] = input[M,K] @ weight[K,N]
fn matmul(input: []const f32, weight: []const f32, bias: ?[]const f32, output: []f32, M: usize, K: usize, N: usize) void {
    for (0..M) |m| {
        for (0..N) |n| {
            var sum: f32 = if (bias) |b| b[n] else 0;
            for (0..K) |k| {
                sum += input[m * K + k] * weight[k * N + n];
            }
            output[m * N + n] = sum;
        }
    }
}

// GELU activation (approximation using sigmoid)
fn gelu(x: f32) f32 {
    // GELU(x) ≈ x * sigmoid(1.702 * x) - faster approximation
    const scaled = 1.702 * x;
    const sigmoid = 1.0 / (1.0 + @exp(-scaled));
    return x * sigmoid;
}

// Softmax over last dimension
fn softmax(data: []f32, seq_len: usize) void {
    for (0..seq_len) |i| {
        const row = data[i * seq_len .. (i + 1) * seq_len];

        // Find max for numerical stability
        var max_val: f32 = row[0];
        for (row[1..]) |v| {
            if (v > max_val) max_val = v;
        }

        // Exp and sum
        var sum: f32 = 0;
        for (row) |*v| {
            v.* = @exp(v.* - max_val);
            sum += v.*;
        }

        // Normalize
        for (row) |*v| {
            v.* /= sum;
        }
    }
}

// Get vocab token at index
fn getVocabToken(idx: usize) []const u8 {
    if (vocab_data == null or idx >= vocab_count) return "";
    const start = vocab_string_offsets[idx];
    const vdata = vocab_data.?;
    // Read length as little-endian u64
    const len: usize = @intCast(
        @as(u64, vdata[start]) |
            (@as(u64, vdata[start + 1]) << 8) |
            (@as(u64, vdata[start + 2]) << 16) |
            (@as(u64, vdata[start + 3]) << 24) |
            (@as(u64, vdata[start + 4]) << 32) |
            (@as(u64, vdata[start + 5]) << 40) |
            (@as(u64, vdata[start + 6]) << 48) |
            (@as(u64, vdata[start + 7]) << 56),
    );
    return vdata[start + 8 .. start + 8 + len];
}

// Check if text starts with token (case-insensitive for letters)
fn startsWithToken(text: []const u8, token: []const u8) bool {
    if (token.len == 0 or token.len > text.len) return false;
    for (0..token.len) |i| {
        var tc = token[i];
        var xc = text[i];
        // Lowercase both
        if (tc >= 'A' and tc <= 'Z') tc = tc + 32;
        if (xc >= 'A' and xc <= 'Z') xc = xc + 32;
        if (tc != xc) return false;
    }
    return true;
}

// BPE tokenize - greedy longest match
fn tokenize(text: []const u8, tokens: []u32) usize {
    var n_tokens: usize = 0;
    var text_pos: usize = 0;

    // Start token
    tokens[n_tokens] = 49406;
    n_tokens += 1;

    while (text_pos < text.len and n_tokens < CLIP_MAX_SEQ_LEN - 1) {
        if (text[text_pos] == 0) break;

        // Skip leading spaces and add space token
        if (text[text_pos] == ' ') {
            text_pos += 1;
            continue;
        }

        // Find longest matching token
        var best_len: usize = 0;
        var best_id: u32 = 0;

        // Search vocab for longest match
        for (0..vocab_count) |i| {
            const tok = getVocabToken(i);
            if (tok.len > 0 and tok.len > best_len) {
                // Check for </w> suffix (word ending marker)
                var tok_text = tok;
                var is_word_end = false;
                if (tok.len >= 4 and tok[tok.len - 4] == '<' and tok[tok.len - 3] == '/' and tok[tok.len - 2] == 'w' and tok[tok.len - 1] == '>') {
                    tok_text = tok[0 .. tok.len - 4];
                    is_word_end = true;
                }

                if (startsWithToken(text[text_pos..], tok_text)) {
                    // For word-end tokens, check next char is space/end
                    if (is_word_end) {
                        const next_pos = text_pos + tok_text.len;
                        if (next_pos >= text.len or text[next_pos] == ' ' or text[next_pos] == 0) {
                            best_len = tok_text.len;
                            best_id = @intCast(i);
                        }
                    } else {
                        best_len = tok_text.len;
                        best_id = @intCast(i);
                    }
                }
            }
        }

        if (best_len > 0) {
            tokens[n_tokens] = best_id;
            n_tokens += 1;
            text_pos += best_len;
        } else {
            // Unknown char, skip
            text_pos += 1;
        }
    }

    // End token
    tokens[n_tokens] = 49407;
    n_tokens += 1;

    // Pad with end token
    const final_len = n_tokens;
    while (n_tokens < CLIP_MAX_SEQ_LEN) {
        tokens[n_tokens] = 49407;
        n_tokens += 1;
    }

    return final_len;
}

// Multi-head self-attention
// input: normalized hidden states
// residual: original hidden states to add output to
fn multiHeadAttention(
    input: []f32,
    residual: []f32,
    seq_len: usize,
    layer: usize,
    q_out: []f32,
    k_out: []f32,
    v_out: []f32,
    attn_out: []f32,
) void {
    const q_w_idx = layer_attn_q_weight_idx[layer] orelse return;
    const q_b_idx = layer_attn_q_bias_idx[layer] orelse return;
    const k_w_idx = layer_attn_k_weight_idx[layer] orelse return;
    const k_b_idx = layer_attn_k_bias_idx[layer] orelse return;
    const v_w_idx = layer_attn_v_weight_idx[layer] orelse return;
    const v_b_idx = layer_attn_v_bias_idx[layer] orelse return;
    const out_w_idx = layer_attn_out_weight_idx[layer] orelse return;
    const out_b_idx = layer_attn_out_bias_idx[layer] orelse return;

    const scale: f32 = 1.0 / @sqrt(@as(f32, CLIP_HEAD_DIM));

    // Compute Q, K, V for all positions from normalized input
    // Linear layer: y = x @ W^T + b, where W is [out, in]
    // So y[i] = sum_j(x[j] * W[i,j]) + b[i], access W[i,j] = W[i * in + j]
    for (0..seq_len) |pos| {
        const h = input[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM];

        // Q = h @ Wq^T + bq
        for (0..CLIP_EMBED_DIM) |i| {
            var sum: f32 = readWeight(q_b_idx, i);
            for (0..CLIP_EMBED_DIM) |j| {
                sum += h[j] * readWeight(q_w_idx, i * CLIP_EMBED_DIM + j);
            }
            q_out[pos * CLIP_EMBED_DIM + i] = sum;
        }

        // K = h @ Wk^T + bk
        for (0..CLIP_EMBED_DIM) |i| {
            var sum: f32 = readWeight(k_b_idx, i);
            for (0..CLIP_EMBED_DIM) |j| {
                sum += h[j] * readWeight(k_w_idx, i * CLIP_EMBED_DIM + j);
            }
            k_out[pos * CLIP_EMBED_DIM + i] = sum;
        }

        // V = h @ Wv^T + bv
        for (0..CLIP_EMBED_DIM) |i| {
            var sum: f32 = readWeight(v_b_idx, i);
            for (0..CLIP_EMBED_DIM) |j| {
                sum += h[j] * readWeight(v_w_idx, i * CLIP_EMBED_DIM + j);
            }
            v_out[pos * CLIP_EMBED_DIM + i] = sum;
        }
    }

    // Compute attention for each head
    for (0..CLIP_NUM_HEADS) |head| {
        const head_offset = head * CLIP_HEAD_DIM;

        // Compute attention scores: Q @ K^T / sqrt(d)
        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                var dot: f32 = 0;
                for (0..CLIP_HEAD_DIM) |d| {
                    dot += q_out[i * CLIP_EMBED_DIM + head_offset + d] *
                        k_out[j * CLIP_EMBED_DIM + head_offset + d];
                }
                // Causal mask: only attend to previous positions
                if (j > i) {
                    attn_out[head * seq_len * seq_len + i * seq_len + j] = -1e9;
                } else {
                    attn_out[head * seq_len * seq_len + i * seq_len + j] = dot * scale;
                }
            }
        }

        // Softmax per row
        for (0..seq_len) |i| {
            const row_start = head * seq_len * seq_len + i * seq_len;
            var max_val: f32 = attn_out[row_start];
            for (1..seq_len) |j| {
                if (attn_out[row_start + j] > max_val) max_val = attn_out[row_start + j];
            }
            var sum: f32 = 0;
            for (0..seq_len) |j| {
                attn_out[row_start + j] = @exp(attn_out[row_start + j] - max_val);
                sum += attn_out[row_start + j];
            }
            for (0..seq_len) |j| {
                attn_out[row_start + j] /= sum;
            }
        }
    }

    // Compute attention output: softmax(QK^T/sqrt(d)) @ V and add to residual
    // Use a temp buffer for attention output before projection
    var attn_output: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;

    for (0..seq_len) |pos| {
        // Zero output first
        for (0..CLIP_EMBED_DIM) |i| {
            attn_output[pos * CLIP_EMBED_DIM + i] = 0;
        }

        for (0..CLIP_NUM_HEADS) |head| {
            const head_offset = head * CLIP_HEAD_DIM;
            for (0..seq_len) |j| {
                const attn_weight = attn_out[head * seq_len * seq_len + pos * seq_len + j];
                for (0..CLIP_HEAD_DIM) |d| {
                    attn_output[pos * CLIP_EMBED_DIM + head_offset + d] +=
                        attn_weight * v_out[j * CLIP_EMBED_DIM + head_offset + d];
                }
            }
        }

        // Output projection: attn_output @ Wo^T + bo, add to residual
        for (0..CLIP_EMBED_DIM) |i| {
            var sum: f32 = readWeight(out_b_idx, i);
            for (0..CLIP_EMBED_DIM) |j| {
                sum += attn_output[pos * CLIP_EMBED_DIM + j] * readWeight(out_w_idx, i * CLIP_EMBED_DIM + j);
            }
            residual[pos * CLIP_EMBED_DIM + i] += sum;
        }
    }
}

// MLP block: FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
// input: normalized hidden states
// residual: original hidden states to add output to
fn mlpBlock(input: []f32, residual: []f32, seq_len: usize, layer: usize) void {
    const fc1_w_idx = layer_mlp_fc1_weight_idx[layer] orelse return;
    const fc1_b_idx = layer_mlp_fc1_bias_idx[layer] orelse return;
    const fc2_w_idx = layer_mlp_fc2_weight_idx[layer] orelse return;
    const fc2_b_idx = layer_mlp_fc2_bias_idx[layer] orelse return;

    // Linear layer: y = x @ W^T + b, W is [out, in]
    for (0..seq_len) |pos| {
        const h = input[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM];

        // First linear: embed_dim -> mlp_dim with GELU
        // fc1_w is [mlp_dim, embed_dim]
        for (0..CLIP_MLP_DIM) |i| {
            var sum: f32 = readWeight(fc1_b_idx, i);
            for (0..CLIP_EMBED_DIM) |j| {
                sum += h[j] * readWeight(fc1_w_idx, i * CLIP_EMBED_DIM + j);
            }
            // GELU activation (quick approximation)
            const x = sum;
            scratch_mlp[pos * CLIP_MLP_DIM + i] = x * (1.0 / (1.0 + @exp(-1.702 * x)));
        }

        // Second linear: mlp_dim -> embed_dim, add to residual
        // fc2_w is [embed_dim, mlp_dim]
        for (0..CLIP_EMBED_DIM) |i| {
            var sum: f32 = readWeight(fc2_b_idx, i);
            for (0..CLIP_MLP_DIM) |j| {
                sum += scratch_mlp[pos * CLIP_MLP_DIM + j] * readWeight(fc2_w_idx, i * CLIP_MLP_DIM + j);
            }
            residual[pos * CLIP_EMBED_DIM + i] += sum;
        }
    }
}

// Layer norm in-place using tensor indices
fn layerNormInPlace(hidden: []f32, seq_len: usize, weight_idx: usize, bias_idx: usize) void {
    const eps: f32 = 1e-5;
    for (0..seq_len) |pos| {
        const h = hidden[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM];

        // Mean
        var mean: f32 = 0;
        for (h) |v| mean += v;
        mean /= CLIP_EMBED_DIM;

        // Variance
        var variance: f32 = 0;
        for (h) |v| {
            const diff = v - mean;
            variance += diff * diff;
        }
        variance /= CLIP_EMBED_DIM;

        // Normalize
        const inv_std = 1.0 / @sqrt(variance + eps);
        for (0..CLIP_EMBED_DIM) |i| {
            h[i] = (h[i] - mean) * inv_std * readWeight(weight_idx, i) + readWeight(bias_idx, i);
        }
    }
}

/// Encode text to embedding
/// Returns 0 on success, negative on error
export fn clip_encode_text(text_len: usize) i32 {
    if (!clip_initialized) return -1;
    if (!clip_model_loaded) return -2;
    if (text_len == 0 or text_len > clip_text_buffer.len) return -3;

    const tok_emb_idx = token_embedding_idx orelse return -4;
    const pos_emb_idx = position_embedding_idx orelse return -5;
    const ln_w_idx = ln_final_weight_idx orelse return -6;
    const ln_b_idx = ln_final_bias_idx orelse return -7;
    const proj_idx = text_projection_idx orelse return -8;

    // Tokenize
    var tokens: [CLIP_MAX_SEQ_LEN]u32 = undefined;
    const seq_len = tokenize(clip_text_buffer[0..text_len], &tokens);

    // Store for debug
    @memcpy(&debug_tokens, &tokens);
    debug_token_count = seq_len;

    // Initialize hidden states with token + position embeddings
    // GGUF stores embeddings as [vocab_size/seq_len, embed_dim] row-major
    for (0..seq_len) |pos| {
        const tok_id = tokens[pos];
        for (0..CLIP_EMBED_DIM) |i| {
            // Token embedding: [vocab_size, embed_dim] -> tok_id * embed_dim + i
            // Position embedding: [seq_len, embed_dim] -> pos * embed_dim + i
            scratch_hidden[pos * CLIP_EMBED_DIM + i] =
                readWeight(tok_emb_idx, tok_id * CLIP_EMBED_DIM + i) +
                readWeight(pos_emb_idx, pos * CLIP_EMBED_DIM + i);
        }
    }

    // Run through transformer layers
    // CLIP uses pre-norm: LN -> Attn -> residual, LN -> MLP -> residual
    for (0..CLIP_NUM_LAYERS) |layer| {
        // Pre-attention layer norm
        const ln1_w_idx = layer_ln1_weight_idx[layer] orelse continue;
        const ln1_b_idx = layer_ln1_bias_idx[layer] orelse continue;

        // Apply layer norm to scratch_ln (input to attention)
        @memcpy(scratch_ln[0 .. seq_len * CLIP_EMBED_DIM], scratch_hidden[0 .. seq_len * CLIP_EMBED_DIM]);
        layerNormInPlace(&scratch_ln, seq_len, ln1_w_idx, ln1_b_idx);

        // Multi-head attention on normalized input, adds to scratch_hidden (residual)
        multiHeadAttention(&scratch_ln, &scratch_hidden, seq_len, layer, &scratch_q, &scratch_k, &scratch_v, &scratch_attn);

        // Pre-MLP layer norm
        const ln2_w_idx = layer_ln2_weight_idx[layer] orelse continue;
        const ln2_b_idx = layer_ln2_bias_idx[layer] orelse continue;

        // Apply layer norm to scratch_ln (input to MLP)
        @memcpy(scratch_ln[0 .. seq_len * CLIP_EMBED_DIM], scratch_hidden[0 .. seq_len * CLIP_EMBED_DIM]);
        layerNormInPlace(&scratch_ln, seq_len, ln2_w_idx, ln2_b_idx);

        // MLP on normalized input, adds to scratch_hidden (residual)
        mlpBlock(&scratch_ln, &scratch_hidden, seq_len, layer);
    }

    // Final layer norm
    layerNormInPlace(&scratch_hidden, seq_len, ln_w_idx, ln_b_idx);

    // Take embedding at EOS position (last real token before padding)
    const eos_pos = seq_len - 1;
    const final_hidden = scratch_hidden[eos_pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM];

    // Project to output dimension: y = x @ W^T, W is [out, in]
    for (0..CLIP_EMBED_DIM) |i| {
        var sum: f32 = 0;
        for (0..CLIP_EMBED_DIM) |j| {
            sum += final_hidden[j] * readWeight(proj_idx, i * CLIP_EMBED_DIM + j);
        }
        clip_output_buffer[i] = sum;
    }

    // L2 normalize
    var norm_sq: f32 = 0;
    for (clip_output_buffer) |v| norm_sq += v * v;
    const norm = @sqrt(norm_sq);
    if (norm > 0) {
        for (&clip_output_buffer) |*v| v.* /= norm;
    }

    return 0;
}

/// Test function for CLIP
export fn clip_test_add(a: i32, b: i32) i32 {
    return a + b;
}

// Debug: get token at position after encoding
var debug_tokens: [CLIP_MAX_SEQ_LEN]u32 = undefined;
var debug_token_count: usize = 0;

export fn clip_debug_get_token(pos: usize) u32 {
    if (pos < debug_token_count) {
        return debug_tokens[pos];
    }
    return 0;
}

export fn clip_debug_get_token_count() usize {
    return debug_token_count;
}
