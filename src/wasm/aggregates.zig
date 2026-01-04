//! Aggregation Functions
//!
//! Column aggregation functions (sum, min, max, avg) for WASM.

const format = @import("format.zig");
const column_meta = @import("column_meta.zig");

const readI64LE = format.readI64LE;
const readF64LE = format.readF64LE;

// ============================================================================
// Global state (synced from wasm.zig)
// ============================================================================

pub var file_data: ?[]const u8 = null;
pub var num_columns: u32 = 0;
pub var column_meta_offsets_start: u64 = 0;

// ============================================================================
// Column Buffer Helper
// ============================================================================

const ColumnBuffer = struct { data: []const u8, start: usize, size: usize, rows: usize };

fn getColumnBuffer(col_idx: u32) ?ColumnBuffer {
    const data = file_data orelse return null;
    const entry = column_meta.getColumnOffsetEntry(data, num_columns, column_meta_offsets_start, col_idx);
    if (entry.len == 0) return null;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return null;

    const col_meta_data = data[col_meta_start..][0..col_meta_len];
    const info = column_meta.getPageBufferInfo(col_meta_data);

    const buf_start: usize = @intCast(info.offset);
    const buf_size: usize = @intCast(info.size);
    if (buf_start + buf_size > data.len) return null;

    return .{ .data = data, .start = buf_start, .size = buf_size, .rows = @intCast(info.rows) };
}

// ============================================================================
// Aggregation Exports
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
// Buffer-based Aggregations (for direct use from JS worker)
// These operate on raw typed array buffers passed from JavaScript
// ============================================================================

/// SIMD vector type for 4x f64
const Vec4f64 = @Vector(4, f64);

/// Sum float64 buffer with SIMD acceleration
export fn sumFloat64Buffer(ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0;

    var sum: Vec4f64 = @splat(0);
    var i: usize = 0;

    // Process 4 elements at a time with SIMD
    while (i + 4 <= len) : (i += 4) {
        const v: Vec4f64 = .{ ptr[i], ptr[i + 1], ptr[i + 2], ptr[i + 3] };
        sum += v;
    }

    // Horizontal sum
    var result = @reduce(.Add, sum);

    // Handle remainder
    while (i < len) : (i += 1) {
        result += ptr[i];
    }

    return result;
}

/// Min float64 buffer with SIMD acceleration
export fn minFloat64Buffer(ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0;

    var min_vec: Vec4f64 = @splat(ptr[0]);
    var i: usize = 0;

    // Process 4 elements at a time with SIMD
    while (i + 4 <= len) : (i += 4) {
        const v: Vec4f64 = .{ ptr[i], ptr[i + 1], ptr[i + 2], ptr[i + 3] };
        min_vec = @min(min_vec, v);
    }

    // Horizontal min
    var result = @reduce(.Min, min_vec);

    // Handle remainder
    while (i < len) : (i += 1) {
        if (ptr[i] < result) result = ptr[i];
    }

    return result;
}

/// Max float64 buffer with SIMD acceleration
export fn maxFloat64Buffer(ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0;

    var max_vec: Vec4f64 = @splat(ptr[0]);
    var i: usize = 0;

    // Process 4 elements at a time with SIMD
    while (i + 4 <= len) : (i += 4) {
        const v: Vec4f64 = .{ ptr[i], ptr[i + 1], ptr[i + 2], ptr[i + 3] };
        max_vec = @max(max_vec, v);
    }

    // Horizontal max
    var result = @reduce(.Max, max_vec);

    // Handle remainder
    while (i < len) : (i += 1) {
        if (ptr[i] > result) result = ptr[i];
    }

    return result;
}

/// Average float64 buffer
export fn avgFloat64Buffer(ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0;
    return sumFloat64Buffer(ptr, len) / @as(f64, @floatFromInt(len));
}

/// Count non-null values (for nullable columns)
/// null_bitmap: bit array where 1 = valid, 0 = null
export fn countNonNull(null_bitmap: [*]const u8, len: usize) usize {
    var count: usize = 0;
    const byte_count = (len + 7) / 8;

    for (0..byte_count) |i| {
        // popcount for each byte
        count += @popCount(null_bitmap[i]);
    }

    // Adjust for padding bits in last byte
    const extra_bits = len % 8;
    if (extra_bits > 0) {
        const mask: u8 = @as(u8, 0xFF) >> @intCast(8 - extra_bits);
        count -= @popCount(null_bitmap[byte_count - 1] & ~mask);
    }

    return count;
}

/// Sum int32 buffer
export fn sumInt32Buffer(ptr: [*]const i32, len: usize) i64 {
    if (len == 0) return 0;
    var sum: i64 = 0;
    for (0..len) |i| sum += ptr[i];
    return sum;
}

/// Sum int64 buffer
export fn sumInt64Buffer(ptr: [*]const i64, len: usize) i64 {
    if (len == 0) return 0;
    var sum: i64 = 0;
    for (0..len) |i| sum += ptr[i];
    return sum;
}

/// Min int64 buffer
export fn minInt64Buffer(ptr: [*]const i64, len: usize) i64 {
    if (len == 0) return 0;
    var min_val = ptr[0];
    for (1..len) |i| if (ptr[i] < min_val) { min_val = ptr[i]; };
    return min_val;
}

/// Max int64 buffer
export fn maxInt64Buffer(ptr: [*]const i64, len: usize) i64 {
    if (len == 0) return 0;
    var max_val = ptr[0];
    for (1..len) |i| if (ptr[i] > max_val) { max_val = ptr[i]; };
    return max_val;
}
