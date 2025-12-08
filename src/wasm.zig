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

fn readF64LE(data: []const u8, offset: usize) f64 {
    const bits = readU64LE(data, offset);
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
