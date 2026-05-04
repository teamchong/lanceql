//! Lance File Format Utilities
//!
//! Binary reading helpers and footer parsing for Lance files.

const std = @import("std");

// ============================================================================
// Constants
// ============================================================================

pub const FOOTER_SIZE: usize = 40;
pub const LANCE_MAGIC = "LANC";

// ============================================================================
// Binary Reading Helpers
// ============================================================================

pub fn readU64LE(data: []const u8, offset: usize) u64 {
    if (offset + 8 > data.len) return 0;
    return std.mem.readInt(u64, data[offset..][0..8], .little);
}

pub fn readU32LE(data: []const u8, offset: usize) u32 {
    if (offset + 4 > data.len) return 0;
    return std.mem.readInt(u32, data[offset..][0..4], .little);
}

pub fn readU16LE(data: []const u8, offset: usize) u16 {
    if (offset + 2 > data.len) return 0;
    return std.mem.readInt(u16, data[offset..][0..2], .little);
}

pub fn readI64LE(data: []const u8, offset: usize) i64 {
    if (offset + 8 > data.len) return 0;
    return std.mem.readInt(i64, data[offset..][0..8], .little);
}

pub fn readI32LE(data: []const u8, offset: usize) i32 {
    if (offset + 4 > data.len) return 0;
    return std.mem.readInt(i32, data[offset..][0..4], .little);
}

pub fn readI16LE(data: []const u8, offset: usize) i16 {
    if (offset + 2 > data.len) return 0;
    return std.mem.readInt(i16, data[offset..][0..2], .little);
}

pub fn readI8(data: []const u8, offset: usize) i8 {
    if (offset >= data.len) return 0;
    return @bitCast(data[offset]);
}

pub fn readU8(data: []const u8, offset: usize) u8 {
    if (offset >= data.len) return 0;
    return data[offset];
}

pub fn readF64LE(data: []const u8, offset: usize) f64 {
    const bits = readU64LE(data, offset);
    return @bitCast(bits);
}

pub fn readF32LE(data: []const u8, offset: usize) f32 {
    const bits = readU32LE(data, offset);
    return @bitCast(bits);
}

/// Read varint (up to 64 bits)
pub fn readVarint(data: []const u8, offset: *usize) u64 {
    var result: u64 = 0;
    var shift: u6 = 0;

    while (offset.* < data.len) {
        const byte = data[offset.*];
        offset.* += 1;

        result |= @as(u64, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;

        shift += 7;
        if (shift >= 64) break;
    }

    return result;
}

// ============================================================================
// Footer Parsing
// ============================================================================

/// Check if data contains a valid Lance file
pub fn isValidLanceFileSlice(data: []const u8) bool {
    if (data.len < FOOTER_SIZE) return false;

    // Check magic at end
    const magic_offset = data.len - 4;
    return data[magic_offset] == 'L' and
        data[magic_offset + 1] == 'A' and
        data[magic_offset + 2] == 'N' and
        data[magic_offset + 3] == 'C';
}

/// Parse footer and get column count
pub fn parseFooterGetColumnsSlice(data: []const u8) u32 {
    if (!isValidLanceFileSlice(data)) return 0;
    const footer_start = data.len - FOOTER_SIZE;
    return readU32LE(data, footer_start + 28);
}

/// Parse footer and get major version
pub fn parseFooterGetMajorVersionSlice(data: []const u8) u16 {
    if (!isValidLanceFileSlice(data)) return 0;
    const footer_start = data.len - FOOTER_SIZE;
    return readU16LE(data, footer_start + 32);
}

/// Parse footer and get minor version
pub fn parseFooterGetMinorVersionSlice(data: []const u8) u16 {
    if (!isValidLanceFileSlice(data)) return 0;
    const footer_start = data.len - FOOTER_SIZE;
    return readU16LE(data, footer_start + 34);
}

/// Get column metadata start offset
pub fn getColumnMetaStartSlice(data: []const u8) u64 {
    if (!isValidLanceFileSlice(data)) return 0;
    const footer_start = data.len - FOOTER_SIZE;
    return readU64LE(data, footer_start + 0);
}

/// Get column metadata offsets start
pub fn getColumnMetaOffsetsStartSlice(data: []const u8) u64 {
    if (!isValidLanceFileSlice(data)) return 0;
    const footer_start = data.len - FOOTER_SIZE;
    return readU64LE(data, footer_start + 8);
}

// ============================================================================
// WASM Exports (raw pointer versions for JavaScript)
// ============================================================================

pub export fn isValidLanceFile(data: [*]const u8, len: usize) u32 {
    if (len < FOOTER_SIZE) return 0;
    return if (isValidLanceFileSlice(data[0..len])) 1 else 0;
}

pub export fn parseFooterGetColumns(data: [*]const u8, len: usize) u32 {
    if (len < FOOTER_SIZE) return 0;
    return parseFooterGetColumnsSlice(data[0..len]);
}

pub export fn parseFooterGetMajorVersion(data: [*]const u8, len: usize) u16 {
    if (len < FOOTER_SIZE) return 0;
    return parseFooterGetMajorVersionSlice(data[0..len]);
}

pub export fn parseFooterGetMinorVersion(data: [*]const u8, len: usize) u16 {
    if (len < FOOTER_SIZE) return 0;
    return parseFooterGetMinorVersionSlice(data[0..len]);
}

pub export fn getColumnMetaStart(data: [*]const u8, len: usize) u64 {
    if (len < FOOTER_SIZE) return 0;
    return getColumnMetaStartSlice(data[0..len]);
}

pub export fn getColumnMetaOffsetsStart(data: [*]const u8, len: usize) u64 {
    if (len < FOOTER_SIZE) return 0;
    return getColumnMetaOffsetsStartSlice(data[0..len]);
}

pub export fn getVersion() u32 {
    return 0x000100; // v0.1.0
}

// ============================================================================
// Range-fetch helpers — let JS callers read column metadata + data buffers
// without holding the whole Lance file in memory. The flow is:
//
//   1. Range-fetch the last 40 bytes (the footer) and call
//      parseFooterGetColumns / getColumnMetaStart / getColumnMetaOffsetsStart
//      to learn how many columns there are and where the metadata table
//      lives in the file.
//   2. Range-fetch the column-meta-offsets array (8 bytes per column —
//      consecutive u64 little-endian START positions of each column's
//      metadata in the file). Column N's metadata length is
//      offsets[N+1] - offsets[N], and the last column ends at
//      column_meta_offsets_start. JS can do this directly with a
//      DataView; no wasm helper needed.
//   3. Range-fetch each column's metadata protobuf bytes and call
//      rangeParseColumnMeta to extract data_offset, data_size,
//      row_count, and vector_dim.
//   4. Range-fetch only the data bytes you actually need (e.g. the
//      vec for one HNSW-visited row at data_offset + row_idx*dim*4,
//      the string entry for one matched row).
//
// All the parsing functions below take a buffer that contains JUST the
// relevant slice of the file (not the whole file), so JS can read tiny
// chunks instead of materialising the whole .lance.
// ============================================================================

/// Parse one column's metadata protobuf and emit its data-buffer info.
/// `meta_data` is just that column's metadata bytes (typically 30-200B),
/// not the whole file. Mirrors fragment_reader.zig's parseColumnMeta but
/// is stateless and writes outputs to pointers so JS can collect them.
/// Returns 1 if parsed successfully, 0 if the buffer was empty or
/// malformed enough to abort.
///
/// Field layout (Lance v2.1 column metadata protobuf):
///   1: name (string)         — column name e.g. "edge_id"
///   2: type (string)         — type tag e.g. "string", "vec"
///   3: nullable (varint)
///   4: data_offset (fixed64) — byte offset in the file of column data
///   5: row_count (varint)    — number of rows
///   6: data_size (varint)    — size in bytes of the column data buffer
///   7: vector_dim (varint)   — dim, only present for vec columns
pub export fn rangeParseColumnMeta(
    meta_data: [*]const u8,
    meta_len: usize,
    out_data_offset: *u64,
    out_row_count: *u64,
    out_data_size: *u64,
    out_vector_dim: *u32,
) u32 {
    if (meta_len == 0) return 0;
    out_data_offset.* = 0;
    out_row_count.* = 0;
    out_data_size.* = 0;
    out_vector_dim.* = 0;

    var pos: usize = 0;
    while (pos < meta_len) {
        const tag = meta_data[pos];
        pos += 1;
        const field_num = tag >> 3;
        const wire_type = tag & 0x7;

        switch (field_num) {
            1, 2 => { // name (1) or type (2): length-delimited string, skip
                if (wire_type == 2) {
                    const slen = readVarintAtPtr(meta_data, meta_len, &pos);
                    pos += @intCast(slen);
                }
            },
            3 => { // nullable: varint, skip value
                if (wire_type == 0) _ = readVarintAtPtr(meta_data, meta_len, &pos);
            },
            4 => { // data_offset: fixed64
                if (wire_type == 1 and pos + 8 <= meta_len) {
                    out_data_offset.* = std.mem.readInt(u64, meta_data[pos..][0..8], .little);
                    pos += 8;
                }
            },
            5 => { // row_count: varint
                if (wire_type == 0) out_row_count.* = readVarintAtPtr(meta_data, meta_len, &pos);
            },
            6 => { // data_size: varint
                if (wire_type == 0) out_data_size.* = readVarintAtPtr(meta_data, meta_len, &pos);
            },
            7 => { // vector_dim: varint
                if (wire_type == 0) out_vector_dim.* = @intCast(readVarintAtPtr(meta_data, meta_len, &pos));
            },
            else => {
                // Skip unknown field per its wire type.
                if (wire_type == 0) {
                    _ = readVarintAtPtr(meta_data, meta_len, &pos);
                } else if (wire_type == 1) {
                    pos += 8;
                } else if (wire_type == 2) {
                    const slen = readVarintAtPtr(meta_data, meta_len, &pos);
                    pos += @intCast(slen);
                } else if (wire_type == 5) {
                    pos += 4;
                } else {
                    return 0; // Unsupported wire type
                }
            },
        }
    }
    return 1;
}

fn readVarintAtPtr(data: [*]const u8, data_len: usize, pos: *usize) u64 {
    var result: u64 = 0;
    var shift: u6 = 0;
    while (pos.* < data_len) {
        const byte = data[pos.*];
        pos.* += 1;
        result |= @as(u64, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;
        shift +|= 7;
    }
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "format: readU64LE" {
    const data = [_]u8{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    const result = readU64LE(&data, 0);
    try std.testing.expectEqual(@as(u64, 0x0807060504030201), result);
}

test "format: readU32LE" {
    const data = [_]u8{ 0x01, 0x02, 0x03, 0x04 };
    const result = readU32LE(&data, 0);
    try std.testing.expectEqual(@as(u32, 0x04030201), result);
}

test "format: readVarint" {
    // Single byte: 42
    const data1 = [_]u8{42};
    var offset1: usize = 0;
    try std.testing.expectEqual(@as(u64, 42), readVarint(&data1, &offset1));

    // Multi-byte: 300 = 0xAC 0x02
    const data2 = [_]u8{ 0xAC, 0x02 };
    var offset2: usize = 0;
    try std.testing.expectEqual(@as(u64, 300), readVarint(&data2, &offset2));
}

test "format: isValidLanceFile" {
    // Too short
    const short = [_]u8{ 'L', 'A', 'N', 'C' };
    try std.testing.expect(!isValidLanceFileSlice(&short));

    // Valid (40 bytes with LANC at end)
    var valid: [40]u8 = undefined;
    @memset(&valid, 0);
    valid[36] = 'L';
    valid[37] = 'A';
    valid[38] = 'N';
    valid[39] = 'C';
    try std.testing.expect(isValidLanceFileSlice(&valid));

    // Invalid magic
    var invalid: [40]u8 = undefined;
    @memset(&invalid, 0);
    invalid[36] = 'X';
    invalid[37] = 'X';
    invalid[38] = 'X';
    invalid[39] = 'X';
    try std.testing.expect(!isValidLanceFileSlice(&invalid));
}
