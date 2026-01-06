//! Aggregation Functions
//!
//! Column aggregation functions (sum, min, max, avg) for WASM.

const std = @import("std");
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

pub const AggFunc = enum { sum, count, avg, min, max };

pub const AggState = struct {
    val: f64 = 0,
    min: f64 = std.math.floatMax(f64),
    max: f64 = -std.math.floatMax(f64),
    count: usize = 0,

    pub fn update(self: *AggState, v: f64, func: AggFunc) void {
        switch (func) {
            .sum, .avg => self.val += v,
            .min => if (v < self.min) { self.min = v; },
            .max => if (v > self.max) { self.max = v; },
            .count => {},
        }
        self.count += 1;
    }

    pub fn updateVec4(self: *AggState, v: Vec4f64, func: AggFunc) void {
        switch (func) {
            .sum, .avg => self.val += @reduce(.Add, v),
            .min => {
                const m = @reduce(.Min, v);
                if (m < self.min) self.min = m;
            },
            .max => {
                const m = @reduce(.Max, v);
                if (m > self.max) self.max = m;
            },
            .count => {},
        }
        self.count += 4;
    }

    pub fn getResult(self: *AggState, func: AggFunc) f64 {
        return switch (func) {
            .sum => self.val,
            .avg => if (self.count > 0) self.val / @as(f64, @floatFromInt(self.count)) else 0,
            .min => self.min,
            .max => self.max,
            .count => @floatFromInt(self.count),
        };
    }
};

pub const MetricSet = struct {
    sum: bool = false,
    min: bool = false,
    max: bool = false,
    count: bool = false,

    pub fn fromFunc(func: AggFunc) MetricSet {
        var set = MetricSet{};
        switch (func) {
            .sum => set.sum = true,
            .count => set.count = true,
            .avg => { set.sum = true; set.count = true; },
            .min => set.min = true,
            .max => set.max = true,
        }
        return set;
    }
};

pub const MultiAggState = struct {
    sum_vec: Vec4f64 = @splat(0),
    min_vec: Vec4f64 = @splat(std.math.floatMax(f64)),
    max_vec: Vec4f64 = @splat(-std.math.floatMax(f64)),
    sum_scalar: f64 = 0,
    min_scalar: f64 = std.math.floatMax(f64),
    max_scalar: f64 = -std.math.floatMax(f64),
    count: usize = 0,
    metrics: MetricSet = .{},

    pub fn update(self: *MultiAggState, v: f64) void {
        if (self.metrics.sum) self.sum_scalar += v;
        if (self.metrics.min) { if (v < self.min_scalar) self.min_scalar = v; }
        if (self.metrics.max) { if (v > self.max_scalar) self.max_scalar = v; }
        if (self.metrics.count) self.count += 1;
    }

    pub fn updateVec4(self: *MultiAggState, v: Vec4f64) void {
        if (self.metrics.sum) self.sum_vec += v;
        if (self.metrics.min) self.min_vec = @min(self.min_vec, v);
        if (self.metrics.max) self.max_vec = @max(self.max_vec, v);
        if (self.metrics.count) self.count += 4;
    }
    
    pub fn getResult(self: *const MultiAggState, func: AggFunc) f64 {
        const final_sum = self.sum_scalar + @reduce(.Add, self.sum_vec);
        const final_min = @min(self.min_scalar, @reduce(.Min, self.min_vec));
        const final_max = @max(self.max_scalar, @reduce(.Max, self.max_vec));
        
        return switch (func) {
            .sum => final_sum,
            .count => @floatFromInt(self.count),
            .avg => if (self.count > 0) final_sum / @as(f64, @floatFromInt(self.count)) else 0,
            .min => final_min,
            .max => final_max,
        };
    }

    pub fn processBuffer(self: *MultiAggState, ptr: [*]const f64, len: usize) void {
        var i: usize = 0;
        var s1: Vec4f64 = @splat(0);
        var s2: Vec4f64 = @splat(0);
        var s3: Vec4f64 = @splat(0);
        var s4: Vec4f64 = @splat(0);
        
        var m1: Vec4f64 = @splat(std.math.floatMax(f64));
        var m2: Vec4f64 = @splat(std.math.floatMax(f64));
        var m3: Vec4f64 = @splat(std.math.floatMax(f64));
        var m4: Vec4f64 = @splat(std.math.floatMax(f64));
        
        var x1: Vec4f64 = @splat(-std.math.floatMax(f64));
        var x2: Vec4f64 = @splat(-std.math.floatMax(f64));
        var x3: Vec4f64 = @splat(-std.math.floatMax(f64));
        var x4: Vec4f64 = @splat(-std.math.floatMax(f64));

        const has_sum = self.metrics.sum;
        const has_min = self.metrics.min;
        const has_max = self.metrics.max;

        while (i + 16 <= len) : (i += 16) {
            const v1: Vec4f64 = ptr[i..i+4][0..4].*;
            const v2: Vec4f64 = ptr[i+4..i+8][0..4].*;
            const v3: Vec4f64 = ptr[i+8..i+12][0..4].*;
            const v4: Vec4f64 = ptr[i+12..i+16][0..4].*;
            
            if (has_sum) { s1 += v1; s2 += v2; s3 += v3; s4 += v4; }
            if (has_min) { m1 = @min(m1, v1); m2 = @min(m2, v2); m3 = @min(m3, v3); m4 = @min(m4, v4); }
            if (has_max) { x1 = @max(x1, v1); x2 = @max(x2, v2); x3 = @max(x3, v3); x4 = @max(x4, v4); }
        }
        
        if (has_sum) self.sum_vec += (s1 + s2 + s3 + s4);
        if (has_min) self.min_vec = @min(self.min_vec, @min(@min(m1, m2), @min(m3, m4)));
        if (has_max) self.max_vec = @max(self.max_vec, @max(@max(x1, x2), @max(x3, x4)));
        if (self.metrics.count) self.count += i;
        
        while (i + 4 <= len) : (i += 4) {
            const v: Vec4f64 = ptr[i..i+4][0..4].*;
            self.updateVec4(v);
        }
        while (i < len) : (i += 1) {
            self.update(ptr[i]);
        }
    }

    pub fn processBufferI32(self: *MultiAggState, ptr: [*]const i32, len: usize) void {
        var i: usize = 0;
        var s1: Vec4f64 = @splat(0);
        var s2: Vec4f64 = @splat(0);
        var s3: Vec4f64 = @splat(0);
        var s4: Vec4f64 = @splat(0);
        
        const has_sum = self.metrics.sum;
        const has_min = self.metrics.min;
        const has_max = self.metrics.max;

        while (i + 16 <= len) : (i += 16) {
            const v1: @Vector(4, i32) = ptr[i..i+4][0..4].*;
            const v2: @Vector(4, i32) = ptr[i+4..i+8][0..4].*;
            const v3: @Vector(4, i32) = ptr[i+8..i+12][0..4].*;
            const v4: @Vector(4, i32) = ptr[i+12..i+16][0..4].*;
            
            if (has_sum) { s1 += @floatFromInt(v1); s2 += @floatFromInt(v2); s3 += @floatFromInt(v3); s4 += @floatFromInt(v4); }
            if (has_min or has_max) {
                const f1: Vec4f64 = @floatFromInt(v1);
                const f2: Vec4f64 = @floatFromInt(v2);
                const f3: Vec4f64 = @floatFromInt(v3);
                const f4: Vec4f64 = @floatFromInt(v4);
                if (has_min) self.min_vec = @min(self.min_vec, @min(@min(f1, f2), @min(f3, f4)));
                if (has_max) self.max_vec = @max(self.max_vec, @max(@max(f1, f2), @max(f3, f4)));
            }
        }
        if (has_sum) self.sum_vec += (s1 + s2 + s3 + s4);
        if (self.metrics.count) self.count += i;

        while (i + 4 <= len) : (i += 4) {
            const v: @Vector(4, i32) = ptr[i..i+4][0..4].*;
            self.updateVec4(@floatFromInt(v));
        }
        while (i < len) : (i += 1) {
            self.update(@floatFromInt(ptr[i]));
        }
    }

    pub fn processBufferI64(self: *MultiAggState, ptr: [*]const i64, len: usize) void {
        var i: usize = 0;
        var s1: Vec4f64 = @splat(0);
        var s2: Vec4f64 = @splat(0);
        var s3: Vec4f64 = @splat(0);
        var s4: Vec4f64 = @splat(0);
        
        const has_sum = self.metrics.sum;
        const has_min = self.metrics.min;
        const has_max = self.metrics.max;

        while (i + 16 <= len) : (i += 16) {
            const v1: @Vector(4, i64) = ptr[i..i+4][0..4].*;
            const v2: @Vector(4, i64) = ptr[i+4..i+8][0..4].*;
            const v3: @Vector(4, i64) = ptr[i+8..i+12][0..4].*;
            const v4: @Vector(4, i64) = ptr[i+12..i+16][0..4].*;
            
            if (has_sum) { s1 += @floatFromInt(v1); s2 += @floatFromInt(v2); s3 += @floatFromInt(v3); s4 += @floatFromInt(v4); }
            if (has_min or has_max) {
                const f1: Vec4f64 = @floatFromInt(v1);
                const f2: Vec4f64 = @floatFromInt(v2);
                const f3: Vec4f64 = @floatFromInt(v3);
                const f4: Vec4f64 = @floatFromInt(v4);
                if (has_min) self.min_vec = @min(self.min_vec, @min(@min(f1, f2), @min(f3, f4)));
                if (has_max) self.max_vec = @max(self.max_vec, @max(@max(f1, f2), @max(f3, f4)));
            }
        }
        if (has_sum) self.sum_vec += (s1 + s2 + s3 + s4);
        if (self.metrics.count) self.count += i;
        
        while (i + 4 <= len) : (i += 4) {
            const v: @Vector(4, i64) = ptr[i..i+4][0..4].*;
            self.updateVec4(@floatFromInt(v));
        }
        while (i < len) : (i += 1) {
            self.update(@floatFromInt(ptr[i]));
        }
    }
};

/// Sum float64 buffer with SIMD acceleration
pub export fn sumFloat64Buffer(ptr: [*]const f64, len: usize) f64 {
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
pub export fn minFloat64Buffer(ptr: [*]const f64, len: usize) f64 {
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
pub export fn maxFloat64Buffer(ptr: [*]const f64, len: usize) f64 {
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
pub export fn avgFloat64Buffer(ptr: [*]const f64, len: usize) f64 {
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

// ============================================================================
// Buffer-based Filtering (for WHERE clause acceleration)
// op: 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge
// ============================================================================

/// Filter float64 buffer, returns count of matching indices
export fn filterFloat64Buffer(
    data_ptr: [*]const f64,
    len: usize,
    op: u32,
    value: f64,
    out_indices: [*]u32,
    max_indices: usize,
) usize {
    var out_count: usize = 0;

    for (0..len) |i| {
        if (out_count >= max_indices) break;
        const v = data_ptr[i];
        const matches = switch (op) {
            0 => v == value,
            1 => v != value,
            2 => v < value,
            3 => v <= value,
            4 => v > value,
            5 => v >= value,
            else => false,
        };
        if (matches) {
            out_indices[out_count] = @intCast(i);
            out_count += 1;
        }
    }
    return out_count;
}

/// Filter int32 buffer
export fn filterInt32Buffer(
    data_ptr: [*]const i32,
    len: usize,
    op: u32,
    value: i32,
    out_indices: [*]u32,
    max_indices: usize,
) usize {
    var out_count: usize = 0;

    for (0..len) |i| {
        if (out_count >= max_indices) break;
        const v = data_ptr[i];
        const matches = switch (op) {
            0 => v == value,
            1 => v != value,
            2 => v < value,
            3 => v <= value,
            4 => v > value,
            5 => v >= value,
            else => false,
        };
        if (matches) {
            out_indices[out_count] = @intCast(i);
            out_count += 1;
        }
    }
    return out_count;
}

/// Filter with range (BETWEEN): returns indices where low <= val <= high
export fn filterFloat64Range(
    data_ptr: [*]const f64,
    len: usize,
    low: f64,
    high: f64,
    out_indices: [*]u32,
    max_indices: usize,
) usize {
    var out_count: usize = 0;

    for (0..len) |i| {
        if (out_count >= max_indices) break;
        const v = data_ptr[i];
        if (v >= low and v <= high) {
            out_indices[out_count] = @intCast(i);
            out_count += 1;
        }
    }
    return out_count;
}

/// AND two index arrays (intersection)
export fn intersectIndices(
    a: [*]const u32,
    a_len: usize,
    b: [*]const u32,
    b_len: usize,
    out: [*]u32,
    max_out: usize,
) usize {
    // Simple O(n*m) for small arrays; could use hash set for large
    var out_count: usize = 0;
    for (0..a_len) |i| {
        if (out_count >= max_out) break;
        const a_val = a[i];
        for (0..b_len) |j| {
            if (b[j] == a_val) {
                out[out_count] = a_val;
                out_count += 1;
                break;
            }
        }
    }
    return out_count;
}

/// OR two index arrays (union, deduplicated)
export fn unionIndices(
    a: [*]const u32,
    a_len: usize,
    b: [*]const u32,
    b_len: usize,
    out: [*]u32,
    max_out: usize,
) usize {
    var out_count: usize = 0;

    // Add all from a
    for (0..a_len) |i| {
        if (out_count >= max_out) break;
        out[out_count] = a[i];
        out_count += 1;
    }

    // Add from b if not already in out
    outer: for (0..b_len) |i| {
        if (out_count >= max_out) break;
        const b_val = b[i];
        for (0..out_count) |j| {
            if (out[j] == b_val) continue :outer;
        }
        out[out_count] = b_val;
        out_count += 1;
    }

    return out_count;
}
