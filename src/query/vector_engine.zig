//! DuckDB-Style Vectorized Query Engine
//!
//! Key architectural decisions from DuckDB:
//! - VECTOR_SIZE = 2048 tuples per batch
//! - Selection vectors for lazy filtering
//! - Linear probing hash tables
//! - Columnar DataChunk format
//! - Zero-copy column access

const std = @import("std");

// ============================================================================
// Constants (matching DuckDB)
// ============================================================================

/// Standard vector size - all operators process this many rows at once
/// DuckDB uses 2048 as it fits in L3 cache on modern CPUs
pub const VECTOR_SIZE: usize = 2048;

/// Hash table load factor threshold for resize
pub const HASH_LOAD_FACTOR: f32 = 0.7;

/// Null sentinel values
pub const NULL_INT64: i64 = std.math.minInt(i64);
pub const NULL_FLOAT64: f64 = std.math.nan(f64);

// ============================================================================
// SIMD Types (WASM SIMD128 compatible)
// ============================================================================

pub const Vec2i64 = @Vector(2, i64);
pub const Vec2u64 = @Vector(2, u64);
pub const Vec4f32 = @Vector(4, f32);
pub const Vec2f64 = @Vector(2, f64);
pub const Vec4i32 = @Vector(4, i32);
pub const Vec8i16 = @Vector(8, i16);
pub const Vec16i8 = @Vector(16, i8);

// ============================================================================
// Selection Vector - Lazy filtering without materialization
// ============================================================================

/// Selection vector marks which rows are valid without copying data
/// This is KEY to DuckDB's performance - filters don't materialize results
pub const SelectionVector = struct {
    /// Indices of selected rows (null = all rows selected)
    indices: ?[]u32,
    /// Number of selected rows
    count: usize,
    /// Backing storage (owned)
    storage: ?[]u32 = null,

    const Self = @This();

    /// Create selection vector selecting all rows
    pub fn all(row_count: usize) Self {
        return .{ .indices = null, .count = row_count };
    }

    /// Create from explicit indices
    pub fn fromIndices(indices: []u32) Self {
        return .{ .indices = indices, .count = indices.len };
    }

    /// Allocate storage for filtered selection
    pub fn alloc(allocator: std.mem.Allocator, capacity: usize) !Self {
        const storage = try allocator.alloc(u32, capacity);
        return .{ .indices = storage, .count = 0, .storage = storage };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        if (self.storage) |s| allocator.free(s);
        self.storage = null;
        self.indices = null;
    }

    /// Get row index at position (handles both selected and all-rows cases)
    pub inline fn get(self: *const Self, pos: usize) u32 {
        return if (self.indices) |idx| idx[pos] else @intCast(pos);
    }

    /// Check if selecting all rows (no filtering)
    pub inline fn isFlat(self: *const Self) bool {
        return self.indices == null;
    }
};

// ============================================================================
// Vector - Single column of data with validity mask
// ============================================================================

/// Column data types
pub const VectorType = enum(u8) {
    int64,
    int32,
    float64,
    float32,
    bool,
    string,
};

/// Single column vector with optional validity mask
pub const Vector = struct {
    /// Data type
    vtype: VectorType,
    /// Raw data pointer (not owned, points to columnar storage)
    data: [*]u8,
    /// Validity mask (null = all valid). Bit i = 1 means row i is valid
    validity: ?[]u64 = null,
    /// Number of elements
    count: usize,

    const Self = @This();

    /// Create vector from typed slice (zero-copy)
    pub fn fromSlice(comptime T: type, slice: []const T) Self {
        const vtype: VectorType = switch (T) {
            i64 => .int64,
            i32 => .int32,
            f64 => .float64,
            f32 => .float32,
            bool => .bool,
            else => @compileError("Unsupported type"),
        };
        return .{
            .vtype = vtype,
            .data = @constCast(@ptrCast(slice.ptr)),
            .count = slice.len,
        };
    }

    /// Get typed data pointer
    pub inline fn getData(self: *const Self, comptime T: type) [*]const T {
        return @ptrCast(@alignCast(self.data));
    }

    /// Get mutable typed data pointer
    pub inline fn getDataMut(self: *Self, comptime T: type) [*]T {
        return @ptrCast(@alignCast(self.data));
    }

    /// Check if row is valid (not null)
    pub inline fn isValid(self: *const Self, idx: usize) bool {
        if (self.validity) |v| {
            const word = idx / 64;
            const bit = @as(u6, @intCast(idx % 64));
            return (v[word] & (@as(u64, 1) << bit)) != 0;
        }
        return true;
    }
};

// ============================================================================
// DataChunk - Collection of vectors (like a row group)
// ============================================================================

/// DataChunk holds multiple columns for batch processing
/// All vectors have the same row count
pub const DataChunk = struct {
    vectors: []Vector,
    count: usize, // Actual row count (≤ VECTOR_SIZE)
    capacity: usize, // Max capacity (typically VECTOR_SIZE)

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, num_columns: usize) !Self {
        const vectors = try allocator.alloc(Vector, num_columns);
        return .{
            .vectors = vectors,
            .count = 0,
            .capacity = VECTOR_SIZE,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.vectors);
    }

    pub fn setCount(self: *Self, count: usize) void {
        self.count = @min(count, self.capacity);
    }
};

// ============================================================================
// Linear Probing Hash Table (DuckDB style)
// ============================================================================

/// Hash table entry with cached hash for fast comparison
const HashEntry = struct {
    /// Upper bits of hash for fast comparison (avoid key comparison)
    hash_bits: u16,
    /// Row index in source data
    row_idx: u32,
    /// 0 = empty, 1 = occupied, 2 = deleted
    state: u8,
    _pad: u8 = 0,
};

/// Linear probing hash table - much faster than chaining for cache locality
pub const LinearHashTable = struct {
    entries: []HashEntry,
    capacity: usize,
    count: usize,
    mask: u64,

    const Self = @This();
    const EMPTY: u8 = 0;
    const OCCUPIED: u8 = 1;

    pub fn init(allocator: std.mem.Allocator, expected_count: usize) !Self {
        // Size to next power of 2, with load factor headroom
        var capacity: usize = 16;
        const target = @as(usize, @intFromFloat(@as(f32, @floatFromInt(expected_count)) / HASH_LOAD_FACTOR));
        while (capacity < target) capacity *= 2;

        const entries = try allocator.alloc(HashEntry, capacity);
        @memset(entries, HashEntry{ .hash_bits = 0, .row_idx = 0, .state = EMPTY, ._pad = 0 });

        return .{
            .entries = entries,
            .capacity = capacity,
            .count = 0,
            .mask = capacity - 1,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
    }

    /// MurmurHash3 finalizer - excellent distribution
    pub inline fn hash64(key: i64) u64 {
        var h: u64 = @bitCast(key);
        h ^= h >> 33;
        h *%= 0xff51afd7ed558ccd;
        h ^= h >> 33;
        h *%= 0xc4ceb9fe1a85ec53;
        h ^= h >> 33;
        return h;
    }

    /// SIMD hash 2 keys at once
    pub inline fn hash64x2(keys: Vec2i64) Vec2u64 {
        var h: Vec2u64 = @bitCast(keys);
        h ^= h >> @splat(33);
        h *%= @splat(0xff51afd7ed558ccd);
        h ^= h >> @splat(33);
        h *%= @splat(0xc4ceb9fe1a85ec53);
        h ^= h >> @splat(33);
        return h;
    }

    /// Insert key with linear probing
    pub fn insert(self: *Self, key: i64, row_idx: u32) void {
        const h = hash64(key);
        const hash_bits: u16 = @truncate(h >> 48);
        var pos = @as(usize, @intCast(h & self.mask));

        // Linear probe until empty slot
        while (self.entries[pos].state == OCCUPIED) {
            pos = (pos + 1) & @as(usize, @intCast(self.mask));
        }

        self.entries[pos] = .{
            .hash_bits = hash_bits,
            .row_idx = row_idx,
            .state = OCCUPIED,
            ._pad = 0,
        };
        self.count += 1;
    }

    /// Build from i64 column - VECTORIZED
    pub fn buildFromColumn(self: *Self, data: []const i64) void {
        var i: usize = 0;

        // SIMD path: hash 2 at a time
        while (i + 2 <= data.len) : (i += 2) {
            const keys: Vec2i64 = .{ data[i], data[i + 1] };
            const hashes = hash64x2(keys);

            // Insert first
            if (data[i] != NULL_INT64) {
                const h0 = hashes[0];
                const hash_bits0: u16 = @truncate(h0 >> 48);
                var pos0 = @as(usize, @intCast(h0 & self.mask));
                while (self.entries[pos0].state == OCCUPIED) {
                    pos0 = (pos0 + 1) & @as(usize, @intCast(self.mask));
                }
                self.entries[pos0] = .{ .hash_bits = hash_bits0, .row_idx = @intCast(i), .state = OCCUPIED, ._pad = 0 };
                self.count += 1;
            }

            // Insert second
            if (data[i + 1] != NULL_INT64) {
                const h1 = hashes[1];
                const hash_bits1: u16 = @truncate(h1 >> 48);
                var pos1 = @as(usize, @intCast(h1 & self.mask));
                while (self.entries[pos1].state == OCCUPIED) {
                    pos1 = (pos1 + 1) & @as(usize, @intCast(self.mask));
                }
                self.entries[pos1] = .{ .hash_bits = hash_bits1, .row_idx = @intCast(i + 1), .state = OCCUPIED, ._pad = 0 };
                self.count += 1;
            }
        }

        // Scalar remainder
        while (i < data.len) : (i += 1) {
            if (data[i] != NULL_INT64) {
                self.insert(data[i], @intCast(i));
            }
        }
    }

    /// Probe for key, returns row index or null
    pub fn probe(self: *const Self, key: i64) ?u32 {
        const h = hash64(key);
        const hash_bits: u16 = @truncate(h >> 48);
        var pos = @as(usize, @intCast(h & self.mask));

        // Linear probe until empty or found
        while (self.entries[pos].state != EMPTY) {
            if (self.entries[pos].state == OCCUPIED and
                self.entries[pos].hash_bits == hash_bits)
            {
                return self.entries[pos].row_idx;
            }
            pos = (pos + 1) & @as(usize, @intCast(self.mask));
        }
        return null;
    }

    /// Probe and return all matches (for non-unique keys)
    pub fn probeAll(self: *const Self, key: i64, keys_data: []const i64, out: []u32) usize {
        const h = hash64(key);
        const hash_bits: u16 = @truncate(h >> 48);
        var pos = @as(usize, @intCast(h & self.mask));
        var out_count: usize = 0;

        while (self.entries[pos].state != EMPTY and out_count < out.len) {
            if (self.entries[pos].state == OCCUPIED and
                self.entries[pos].hash_bits == hash_bits)
            {
                // Verify actual key match (hash collision possible)
                const row_idx = self.entries[pos].row_idx;
                if (keys_data[row_idx] == key) {
                    out[out_count] = row_idx;
                    out_count += 1;
                }
            }
            pos = (pos + 1) & @as(usize, @intCast(self.mask));
        }
        return out_count;
    }
};

// ============================================================================
// Vectorized Aggregation
// ============================================================================

/// Aggregation state for SUM/COUNT/AVG/MIN/MAX
pub const AggState = struct {
    sum: f64 = 0,
    count: u64 = 0,
    min: f64 = std.math.inf(f64),
    max: f64 = -std.math.inf(f64),

    const Self = @This();

    /// SIMD update from f64 column
    pub fn updateColumnF64(self: *Self, data: []const f64, sel: *const SelectionVector) void {
        if (sel.isFlat()) {
            // Fast path: process entire column with SIMD
            self.updateContiguousF64(data);
        } else {
            // Selection vector path: gather values
            const indices = sel.indices.?;
            for (indices[0..sel.count]) |idx| {
                const v = data[idx];
                self.sum += v;
                self.count += 1;
                if (v < self.min) self.min = v;
                if (v > self.max) self.max = v;
            }
        }
    }

    /// SIMD update from contiguous f64 array
    fn updateContiguousF64(self: *Self, data: []const f64) void {
        var sum_vec: Vec2f64 = @splat(0);
        var min_vec: Vec2f64 = @splat(std.math.inf(f64));
        var max_vec: Vec2f64 = @splat(-std.math.inf(f64));
        var i: usize = 0;

        // SIMD path: 2 elements at a time
        while (i + 2 <= data.len) : (i += 2) {
            const v: Vec2f64 = .{ data[i], data[i + 1] };
            sum_vec += v;
            min_vec = @min(min_vec, v);
            max_vec = @max(max_vec, v);
        }

        self.sum += @reduce(.Add, sum_vec);
        const batch_min = @reduce(.Min, min_vec);
        const batch_max = @reduce(.Max, max_vec);
        if (batch_min < self.min) self.min = batch_min;
        if (batch_max > self.max) self.max = batch_max;
        self.count += i;

        // Scalar remainder
        while (i < data.len) : (i += 1) {
            const v = data[i];
            self.sum += v;
            if (v < self.min) self.min = v;
            if (v > self.max) self.max = v;
        }
        self.count += data.len - (data.len / 2 * 2);
    }

    /// SIMD update from i64 column
    pub fn updateColumnI64(self: *Self, data: []const i64, sel: *const SelectionVector) void {
        if (sel.isFlat()) {
            var sum: i64 = 0;
            var min: i64 = std.math.maxInt(i64);
            var max: i64 = std.math.minInt(i64);
            var i: usize = 0;

            // SIMD path
            while (i + 2 <= data.len) : (i += 2) {
                const v: Vec2i64 = .{ data[i], data[i + 1] };
                sum += @reduce(.Add, v);
                const batch_min = @reduce(.Min, v);
                const batch_max = @reduce(.Max, v);
                if (batch_min < min) min = batch_min;
                if (batch_max > max) max = batch_max;
            }

            // Remainder
            while (i < data.len) : (i += 1) {
                sum += data[i];
                if (data[i] < min) min = data[i];
                if (data[i] > max) max = data[i];
            }

            self.sum += @floatFromInt(sum);
            self.count += data.len;
            if (@as(f64, @floatFromInt(min)) < self.min) self.min = @floatFromInt(min);
            if (@as(f64, @floatFromInt(max)) > self.max) self.max = @floatFromInt(max);
        } else {
            const indices = sel.indices.?;
            for (indices[0..sel.count]) |idx| {
                const v = data[idx];
                self.sum += @floatFromInt(v);
                self.count += 1;
                const vf: f64 = @floatFromInt(v);
                if (vf < self.min) self.min = vf;
                if (vf > self.max) self.max = vf;
            }
        }
    }

    pub fn getAvg(self: *const Self) f64 {
        return if (self.count > 0) self.sum / @as(f64, @floatFromInt(self.count)) else 0;
    }

    /// Update with single value (for row-by-row processing)
    pub inline fn update(self: *Self, val: f64) void {
        self.sum += val;
        self.count += 1;
        if (val < self.min) self.min = val;
        if (val > self.max) self.max = val;
    }

    /// Finalize aggregate result based on function type
    /// Compatible with WASM aggregates.AggFunc enum values
    pub fn finalize(self: *const Self, func_ordinal: u8) f64 {
        // Map ordinal to result: 0=sum, 1=count, 2=avg, 3=min, 4=max
        return switch (func_ordinal) {
            0 => self.sum, // sum
            1 => @floatFromInt(self.count), // count
            2 => self.getAvg(), // avg
            3 => if (self.min == std.math.inf(f64)) 0 else self.min, // min
            4 => if (self.max == -std.math.inf(f64)) 0 else self.max, // max
            else => self.sum,
        };
    }
};

// ============================================================================
// Hash Join Executor
// ============================================================================

/// Execute hash join between two columns
/// Returns (left_indices, right_indices) pairs
pub fn executeHashJoin(
    allocator: std.mem.Allocator,
    left_keys: []const i64,
    right_keys: []const i64,
    max_results: usize,
) !struct { left: []u32, right: []u32 } {
    // Build phase: hash the right (smaller) table
    var ht = try LinearHashTable.init(allocator, right_keys.len);
    defer ht.deinit(allocator);
    ht.buildFromColumn(right_keys);

    // Probe phase: scan left table
    var left_out = try allocator.alloc(u32, max_results);
    errdefer allocator.free(left_out);
    var right_out = try allocator.alloc(u32, max_results);
    errdefer allocator.free(right_out);

    var out_count: usize = 0;
    var match_buf: [64]u32 = undefined;

    for (left_keys, 0..) |key, left_idx| {
        if (key == NULL_INT64) continue;

        // Find all matches in right table
        const matches = ht.probeAll(key, right_keys, &match_buf);
        for (match_buf[0..matches]) |right_idx| {
            if (out_count >= max_results) break;
            left_out[out_count] = @intCast(left_idx);
            right_out[out_count] = right_idx;
            out_count += 1;
        }
    }

    return .{
        .left = left_out[0..out_count],
        .right = right_out[0..out_count],
    };
}

// ============================================================================
// Vectorized Filter Operations
// ============================================================================

/// Filter comparison operators
pub const FilterOp = enum {
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
};

/// Apply filter to i64 column, output selection vector
/// Returns number of matching rows
pub fn filterI64(
    data: []const i64,
    op: FilterOp,
    value: i64,
    input_sel: *const SelectionVector,
    output_sel: *SelectionVector,
) usize {
    const out_indices = output_sel.storage.?;
    var out_count: usize = 0;

    if (input_sel.isFlat()) {
        // Fast path: scan entire column
        var i: usize = 0;

        // SIMD comparison for eq/ne (most common)
        if (op == .eq or op == .ne) {
            const target: Vec2i64 = @splat(value);
            while (i + 2 <= data.len and out_count + 2 <= out_indices.len) : (i += 2) {
                const v: Vec2i64 = .{ data[i], data[i + 1] };
                const cmp = if (op == .eq) v == target else v != target;
                if (cmp[0]) {
                    out_indices[out_count] = @intCast(i);
                    out_count += 1;
                }
                if (cmp[1]) {
                    out_indices[out_count] = @intCast(i + 1);
                    out_count += 1;
                }
            }
        }

        // Scalar remainder / other ops
        while (i < data.len and out_count < out_indices.len) : (i += 1) {
            const match = switch (op) {
                .eq => data[i] == value,
                .ne => data[i] != value,
                .lt => data[i] < value,
                .le => data[i] <= value,
                .gt => data[i] > value,
                .ge => data[i] >= value,
            };
            if (match) {
                out_indices[out_count] = @intCast(i);
                out_count += 1;
            }
        }
    } else {
        // Selection vector path
        const in_indices = input_sel.indices.?;
        for (in_indices[0..input_sel.count]) |idx| {
            if (out_count >= out_indices.len) break;
            const match = switch (op) {
                .eq => data[idx] == value,
                .ne => data[idx] != value,
                .lt => data[idx] < value,
                .le => data[idx] <= value,
                .gt => data[idx] > value,
                .ge => data[idx] >= value,
            };
            if (match) {
                out_indices[out_count] = idx;
                out_count += 1;
            }
        }
    }

    output_sel.count = out_count;
    output_sel.indices = out_indices[0..out_count];
    return out_count;
}

/// Apply filter to f64 column
pub fn filterF64(
    data: []const f64,
    op: FilterOp,
    value: f64,
    input_sel: *const SelectionVector,
    output_sel: *SelectionVector,
) usize {
    const out_indices = output_sel.storage.?;
    var out_count: usize = 0;

    if (input_sel.isFlat()) {
        var i: usize = 0;

        // SIMD for common comparisons
        if (op == .gt or op == .lt) {
            const target: Vec2f64 = @splat(value);
            while (i + 2 <= data.len and out_count + 2 <= out_indices.len) : (i += 2) {
                const v: Vec2f64 = .{ data[i], data[i + 1] };
                const cmp = if (op == .gt) v > target else v < target;
                if (cmp[0]) {
                    out_indices[out_count] = @intCast(i);
                    out_count += 1;
                }
                if (cmp[1]) {
                    out_indices[out_count] = @intCast(i + 1);
                    out_count += 1;
                }
            }
        }

        // Scalar remainder
        while (i < data.len and out_count < out_indices.len) : (i += 1) {
            const match = switch (op) {
                .eq => data[i] == value,
                .ne => data[i] != value,
                .lt => data[i] < value,
                .le => data[i] <= value,
                .gt => data[i] > value,
                .ge => data[i] >= value,
            };
            if (match) {
                out_indices[out_count] = @intCast(i);
                out_count += 1;
            }
        }
    } else {
        const in_indices = input_sel.indices.?;
        for (in_indices[0..input_sel.count]) |idx| {
            if (out_count >= out_indices.len) break;
            const match = switch (op) {
                .eq => data[idx] == value,
                .ne => data[idx] != value,
                .lt => data[idx] < value,
                .le => data[idx] <= value,
                .gt => data[idx] > value,
                .ge => data[idx] >= value,
            };
            if (match) {
                out_indices[out_count] = idx;
                out_count += 1;
            }
        }
    }

    output_sel.count = out_count;
    output_sel.indices = out_indices[0..out_count];
    return out_count;
}

// ============================================================================
// Hash-Based GROUP BY Aggregation
// ============================================================================

/// Group aggregation entry
pub const GroupAggEntry = struct {
    /// First row index for this group (for retrieving group key)
    first_row: u32,
    /// Aggregation state
    agg: AggState,
};

/// Hash-based GROUP BY with aggregation
/// Groups by i64 key column and aggregates f64 value column
pub const HashGroupBy = struct {
    /// Hash table mapping group key hash -> group index
    ht: LinearHashTable,
    /// Group aggregation states
    groups: std.ArrayListUnmanaged(GroupAggEntry),
    /// Allocator
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, expected_groups: usize) !Self {
        return .{
            .ht = try LinearHashTable.init(allocator, expected_groups),
            .groups = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ht.deinit(self.allocator);
        self.groups.deinit(self.allocator);
    }

    /// Build groups from i64 key column
    pub fn buildGroups(self: *Self, keys: []const i64) !void {
        for (keys, 0..) |key, row_idx| {
            if (key == NULL_INT64) continue;

            const h = LinearHashTable.hash64(key);
            const hash_bits: u16 = @truncate(h >> 48);
            var pos = @as(usize, @intCast(h & self.ht.mask));

            // Find existing group or create new one
            var found = false;
            while (self.ht.entries[pos].state == LinearHashTable.OCCUPIED) {
                if (self.ht.entries[pos].hash_bits == hash_bits) {
                    // Check actual key match
                    const group_idx = self.ht.entries[pos].row_idx;
                    const group_first_row = self.groups.items[group_idx].first_row;
                    if (keys[group_first_row] == key) {
                        found = true;
                        break;
                    }
                }
                pos = (pos + 1) & @as(usize, @intCast(self.ht.mask));
            }

            if (!found) {
                // Create new group
                const group_idx: u32 = @intCast(self.groups.items.len);
                try self.groups.append(self.allocator, .{
                    .first_row = @intCast(row_idx),
                    .agg = .{},
                });

                // Insert into hash table
                self.ht.entries[pos] = .{
                    .hash_bits = hash_bits,
                    .row_idx = group_idx,
                    .state = LinearHashTable.OCCUPIED,
                    ._pad = 0,
                };
                self.ht.count += 1;
            }
        }
    }

    /// Aggregate f64 values into groups
    pub fn aggregateF64(self: *Self, keys: []const i64, values: []const f64) void {
        for (keys, values) |key, value| {
            if (key == NULL_INT64) continue;

            // Find group
            const h = LinearHashTable.hash64(key);
            const hash_bits: u16 = @truncate(h >> 48);
            var pos = @as(usize, @intCast(h & self.ht.mask));

            while (self.ht.entries[pos].state == LinearHashTable.OCCUPIED) {
                if (self.ht.entries[pos].hash_bits == hash_bits) {
                    const group_idx = self.ht.entries[pos].row_idx;
                    const group_first_row = self.groups.items[group_idx].first_row;
                    if (keys[group_first_row] == key) {
                        // Update aggregation
                        const agg = &self.groups.items[group_idx].agg;
                        agg.sum += value;
                        agg.count += 1;
                        if (value < agg.min) agg.min = value;
                        if (value > agg.max) agg.max = value;
                        break;
                    }
                }
                pos = (pos + 1) & @as(usize, @intCast(self.ht.mask));
            }
        }
    }

    /// Aggregate i64 values into groups
    pub fn aggregateI64(self: *Self, keys: []const i64, values: []const i64) void {
        for (keys, values) |key, value| {
            if (key == NULL_INT64) continue;

            const h = LinearHashTable.hash64(key);
            const hash_bits: u16 = @truncate(h >> 48);
            var pos = @as(usize, @intCast(h & self.ht.mask));

            while (self.ht.entries[pos].state == LinearHashTable.OCCUPIED) {
                if (self.ht.entries[pos].hash_bits == hash_bits) {
                    const group_idx = self.ht.entries[pos].row_idx;
                    const group_first_row = self.groups.items[group_idx].first_row;
                    if (keys[group_first_row] == key) {
                        const agg = &self.groups.items[group_idx].agg;
                        const vf: f64 = @floatFromInt(value);
                        agg.sum += vf;
                        agg.count += 1;
                        if (vf < agg.min) agg.min = vf;
                        if (vf > agg.max) agg.max = vf;
                        break;
                    }
                }
                pos = (pos + 1) & @as(usize, @intCast(self.ht.mask));
            }
        }
    }

    /// Get number of groups
    pub fn groupCount(self: *const Self) usize {
        return self.groups.items.len;
    }

    /// Get group key at index
    pub fn getGroupKey(self: *const Self, group_idx: usize, keys: []const i64) i64 {
        return keys[self.groups.items[group_idx].first_row];
    }

    /// Get group aggregation state
    pub fn getGroupAgg(self: *const Self, group_idx: usize) *const AggState {
        return &self.groups.items[group_idx].agg;
    }

    /// Get first row index for a group (for representative value lookup)
    pub fn getGroupFirstRow(self: *const Self, group_idx: usize) u32 {
        return self.groups.items[group_idx].first_row;
    }
};

// ============================================================================
// Vectorized Projection (scatter/gather)
// ============================================================================

/// Gather i64 values using selection vector
pub fn gatherI64(data: []const i64, sel: *const SelectionVector, out: []i64) usize {
    const count = @min(sel.count, out.len);
    if (sel.isFlat()) {
        @memcpy(out[0..count], data[0..count]);
    } else {
        const indices = sel.indices.?;
        for (indices[0..count], 0..) |idx, i| {
            out[i] = data[idx];
        }
    }
    return count;
}

/// Gather f64 values using selection vector
pub fn gatherF64(data: []const f64, sel: *const SelectionVector, out: []f64) usize {
    const count = @min(sel.count, out.len);
    if (sel.isFlat()) {
        @memcpy(out[0..count], data[0..count]);
    } else {
        const indices = sel.indices.?;
        for (indices[0..count], 0..) |idx, i| {
            out[i] = data[idx];
        }
    }
    return count;
}

// ============================================================================
// Simple Aggregate (no GROUP BY)
// ============================================================================

/// Compute SUM of f64 column with selection vector
pub fn sumF64(data: []const f64, sel: *const SelectionVector) f64 {
    var state = AggState{};
    state.updateColumnF64(data, sel);
    return state.sum;
}

/// Compute SUM of i64 column with selection vector
pub fn sumI64(data: []const i64, sel: *const SelectionVector) i64 {
    if (sel.isFlat()) {
        var sum: i64 = 0;
        var i: usize = 0;

        // SIMD path
        while (i + 2 <= data.len) : (i += 2) {
            const v: Vec2i64 = .{ data[i], data[i + 1] };
            sum += @reduce(.Add, v);
        }
        while (i < data.len) : (i += 1) {
            sum += data[i];
        }
        return sum;
    } else {
        var sum: i64 = 0;
        const indices = sel.indices.?;
        for (indices[0..sel.count]) |idx| {
            sum += data[idx];
        }
        return sum;
    }
}

/// Count rows with selection vector
pub fn countRows(sel: *const SelectionVector) usize {
    return sel.count;
}

/// Compute MIN of f64 column
pub fn minF64(data: []const f64, sel: *const SelectionVector) f64 {
    var state = AggState{};
    state.updateColumnF64(data, sel);
    return state.min;
}

/// Compute MAX of f64 column
pub fn maxF64(data: []const f64, sel: *const SelectionVector) f64 {
    var state = AggState{};
    state.updateColumnF64(data, sel);
    return state.max;
}

/// Compute AVG of f64 column
pub fn avgF64(data: []const f64, sel: *const SelectionVector) f64 {
    var state = AggState{};
    state.updateColumnF64(data, sel);
    return state.getAvg();
}

// ============================================================================
// Streaming Batch Executor - Single code path for WASM and Native
// ============================================================================

/// Column data types for streaming
pub const ColumnType = enum(u8) {
    int64,
    int32,
    float64,
    float32,
    string,
    bool_,
};

/// A batch of column data - the unit of streaming execution
pub const Batch = struct {
    /// Column data pointers (not owned)
    columns: []ColumnSlice,
    /// Selection vector for this batch
    sel: SelectionVector,
    /// Number of columns
    num_columns: usize,
    /// Offset in source data (for tracking position)
    offset: usize,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, num_columns: usize, capacity: usize) !Self {
        const columns = try allocator.alloc(ColumnSlice, num_columns);
        const sel = try SelectionVector.alloc(allocator, capacity);
        return .{
            .columns = columns,
            .sel = sel,
            .num_columns = num_columns,
            .offset = 0,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.columns);
        self.sel.deinit(allocator);
    }

    pub fn reset(self: *Self, row_count: usize, offset: usize) void {
        self.sel.count = row_count;
        self.sel.indices = null; // Flat selection (all rows)
        self.offset = offset;
    }
};

/// Typed slice for column data
pub const ColumnSlice = union(ColumnType) {
    int64: []const i64,
    int32: []const i32,
    float64: []const f64,
    float32: []const f32,
    string: []const []const u8,
    bool_: []const bool,
};

/// Column reader interface - implemented by both WASM and native
pub const ColumnReader = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Read next batch of rows for a column, returns slice of data
        readBatchI64: *const fn (ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const i64,
        readBatchF64: *const fn (ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const f64,
        readBatchI32: *const fn (ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const i32,
        readBatchF32: *const fn (ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const f32,
        readBatchString: *const fn (ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const []const u8,
        /// Get total row count
        getRowCount: *const fn (ptr: *anyopaque) usize,
        /// Get column type
        getColumnType: *const fn (ptr: *anyopaque, col_idx: usize) ColumnType,
    };

    pub fn readBatchI64(self: *const ColumnReader, col_idx: usize, offset: usize, count: usize) ?[]const i64 {
        return self.vtable.readBatchI64(self.ptr, col_idx, offset, count);
    }

    pub fn readBatchF64(self: *const ColumnReader, col_idx: usize, offset: usize, count: usize) ?[]const f64 {
        return self.vtable.readBatchF64(self.ptr, col_idx, offset, count);
    }

    pub fn readBatchI32(self: *const ColumnReader, col_idx: usize, offset: usize, count: usize) ?[]const i32 {
        return self.vtable.readBatchI32(self.ptr, col_idx, offset, count);
    }

    pub fn readBatchF32(self: *const ColumnReader, col_idx: usize, offset: usize, count: usize) ?[]const f32 {
        return self.vtable.readBatchF32(self.ptr, col_idx, offset, count);
    }

    pub fn readBatchString(self: *const ColumnReader, col_idx: usize, offset: usize, count: usize) ?[]const []const u8 {
        return self.vtable.readBatchString(self.ptr, col_idx, offset, count);
    }

    pub fn getRowCount(self: *const ColumnReader) usize {
        return self.vtable.getRowCount(self.ptr);
    }

    pub fn getColumnType(self: *const ColumnReader, col_idx: usize) ColumnType {
        return self.vtable.getColumnType(self.ptr, col_idx);
    }
};

/// Filter condition for streaming filter
pub const FilterCondition = struct {
    col_idx: usize,
    op: FilterOp,
    value: FilterValue,
};

pub const FilterValue = union(enum) {
    int64: i64,
    float64: f64,
    int32: i32,
    float32: f32,
};

/// Aggregate function type
pub const AggFunc = enum {
    count,
    sum,
    avg,
    min,
    max,
};

/// Aggregate specification
pub const AggSpec = struct {
    func: AggFunc,
    col_idx: usize,
    output_name: []const u8,
};

/// Streaming aggregator - processes batches and accumulates results
pub const StreamingAggregator = struct {
    /// Aggregation states (one per aggregate)
    states: []AggState,
    /// Aggregate specifications
    specs: []const AggSpec,
    /// Allocator
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, specs: []const AggSpec) !Self {
        const states = try allocator.alloc(AggState, specs.len);
        for (states) |*s| {
            s.* = AggState{};
        }
        return .{
            .states = states,
            .specs = specs,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.states);
    }

    /// Process a batch of data
    pub fn processBatch(self: *Self, batch: *const Batch) void {
        for (self.specs, 0..) |spec, i| {
            const col = batch.columns[spec.col_idx];
            switch (col) {
                .int64 => |data| self.states[i].updateColumnI64(data, &batch.sel),
                .float64 => |data| self.states[i].updateColumnF64(data, &batch.sel),
                .int32 => |data| {
                    // Convert to i64 for aggregation
                    if (batch.sel.isFlat()) {
                        for (data) |v| {
                            self.states[i].sum += @floatFromInt(v);
                            self.states[i].count += 1;
                            const vf: f64 = @floatFromInt(v);
                            if (vf < self.states[i].min) self.states[i].min = vf;
                            if (vf > self.states[i].max) self.states[i].max = vf;
                        }
                    } else {
                        const indices = batch.sel.indices.?;
                        for (indices[0..batch.sel.count]) |idx| {
                            const v = data[idx];
                            self.states[i].sum += @floatFromInt(v);
                            self.states[i].count += 1;
                            const vf: f64 = @floatFromInt(v);
                            if (vf < self.states[i].min) self.states[i].min = vf;
                            if (vf > self.states[i].max) self.states[i].max = vf;
                        }
                    }
                },
                .float32 => |data| {
                    if (batch.sel.isFlat()) {
                        for (data) |v| {
                            self.states[i].sum += v;
                            self.states[i].count += 1;
                            if (v < self.states[i].min) self.states[i].min = v;
                            if (v > self.states[i].max) self.states[i].max = v;
                        }
                    } else {
                        const indices = batch.sel.indices.?;
                        for (indices[0..batch.sel.count]) |idx| {
                            const v = data[idx];
                            self.states[i].sum += v;
                            self.states[i].count += 1;
                            if (v < self.states[i].min) self.states[i].min = v;
                            if (v > self.states[i].max) self.states[i].max = v;
                        }
                    }
                },
                else => {}, // String/bool not aggregatable
            }
        }
    }

    /// Get final result for an aggregate
    pub fn getResult(self: *const Self, idx: usize) f64 {
        const state = &self.states[idx];
        return switch (self.specs[idx].func) {
            .count => @floatFromInt(state.count),
            .sum => state.sum,
            .avg => state.getAvg(),
            .min => state.min,
            .max => state.max,
        };
    }

    /// Get count for an aggregate
    pub fn getCount(self: *const Self, idx: usize) u64 {
        return self.states[idx].count;
    }
};

/// Streaming GROUP BY aggregator
pub const StreamingGroupBy = struct {
    /// Hash table for group lookup
    ht: LinearHashTable,
    /// Group aggregation states
    groups: std.ArrayListUnmanaged(GroupState),
    /// Aggregate specifications
    specs: []const AggSpec,
    /// Group key column index
    key_col_idx: usize,
    /// Key data storage (for looking up keys later)
    keys: std.ArrayListUnmanaged(i64),
    /// Allocator
    allocator: std.mem.Allocator,

    const Self = @This();

    const GroupState = struct {
        first_key: i64,
        aggs: []AggState,
    };

    pub fn init(allocator: std.mem.Allocator, key_col_idx: usize, specs: []const AggSpec, expected_groups: usize) !Self {
        return .{
            .ht = try LinearHashTable.init(allocator, expected_groups),
            .groups = .{},
            .specs = specs,
            .key_col_idx = key_col_idx,
            .keys = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ht.deinit(self.allocator);
        for (self.groups.items) |*g| {
            self.allocator.free(g.aggs);
        }
        self.groups.deinit(self.allocator);
        self.keys.deinit(self.allocator);
    }

    /// Process a batch - group keys and aggregate values
    pub fn processBatch(self: *Self, batch: *const Batch) !void {
        const key_col = batch.columns[self.key_col_idx];
        const keys = switch (key_col) {
            .int64 => |d| d,
            else => return, // Only int64 keys supported for now
        };

        // Process each row in the batch
        if (batch.sel.isFlat()) {
            for (keys, 0..) |key, row_idx| {
                try self.processRow(key, batch, row_idx);
            }
        } else {
            const indices = batch.sel.indices.?;
            for (indices[0..batch.sel.count]) |idx| {
                try self.processRow(keys[idx], batch, idx);
            }
        }
    }

    fn processRow(self: *Self, key: i64, batch: *const Batch, row_idx: usize) !void {
        if (key == NULL_INT64) return;

        // Find or create group
        const h = LinearHashTable.hash64(key);
        const hash_bits: u16 = @truncate(h >> 48);
        var pos = @as(usize, @intCast(h & self.ht.mask));

        var group_idx: ?usize = null;
        while (self.ht.entries[pos].state == LinearHashTable.OCCUPIED) {
            if (self.ht.entries[pos].hash_bits == hash_bits) {
                const gidx = self.ht.entries[pos].row_idx;
                if (self.groups.items[gidx].first_key == key) {
                    group_idx = gidx;
                    break;
                }
            }
            pos = (pos + 1) & @as(usize, @intCast(self.ht.mask));
        }

        if (group_idx == null) {
            // Create new group
            const gidx: u32 = @intCast(self.groups.items.len);
            const aggs = try self.allocator.alloc(AggState, self.specs.len);
            for (aggs) |*a| a.* = AggState{};

            try self.groups.append(self.allocator, .{
                .first_key = key,
                .aggs = aggs,
            });
            try self.keys.append(self.allocator, key);

            self.ht.entries[pos] = .{
                .hash_bits = hash_bits,
                .row_idx = gidx,
                .state = LinearHashTable.OCCUPIED,
                ._pad = 0,
            };
            self.ht.count += 1;
            group_idx = gidx;
        }

        // Update aggregates for this group
        const group = &self.groups.items[group_idx.?];
        for (self.specs, 0..) |spec, i| {
            const col = batch.columns[spec.col_idx];
            switch (col) {
                .int64 => |data| {
                    const v = data[row_idx];
                    const vf: f64 = @floatFromInt(v);
                    group.aggs[i].sum += vf;
                    group.aggs[i].count += 1;
                    if (vf < group.aggs[i].min) group.aggs[i].min = vf;
                    if (vf > group.aggs[i].max) group.aggs[i].max = vf;
                },
                .float64 => |data| {
                    const v = data[row_idx];
                    group.aggs[i].sum += v;
                    group.aggs[i].count += 1;
                    if (v < group.aggs[i].min) group.aggs[i].min = v;
                    if (v > group.aggs[i].max) group.aggs[i].max = v;
                },
                else => {},
            }
        }
    }

    pub fn groupCount(self: *const Self) usize {
        return self.groups.items.len;
    }

    pub fn getGroupKey(self: *const Self, group_idx: usize) i64 {
        return self.keys.items[group_idx];
    }

    pub fn getGroupAgg(self: *const Self, group_idx: usize, agg_idx: usize, func: AggFunc) f64 {
        const state = &self.groups.items[group_idx].aggs[agg_idx];
        return switch (func) {
            .count => @floatFromInt(state.count),
            .sum => state.sum,
            .avg => state.getAvg(),
            .min => state.min,
            .max => state.max,
        };
    }
};

/// Streaming executor - processes queries in batches
/// This is the SINGLE CODE PATH for both WASM and Native
pub const StreamingExecutor = struct {
    allocator: std.mem.Allocator,
    batch: Batch,
    filter_sel: SelectionVector,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, num_columns: usize) !Self {
        return .{
            .allocator = allocator,
            .batch = try Batch.init(allocator, num_columns, VECTOR_SIZE),
            .filter_sel = try SelectionVector.alloc(allocator, VECTOR_SIZE),
        };
    }

    pub fn deinit(self: *Self) void {
        self.batch.deinit(self.allocator);
        self.filter_sel.deinit(self.allocator);
    }

    /// Execute a simple aggregate query (no GROUP BY)
    /// Returns: array of aggregate results
    pub fn executeAggregate(
        self: *Self,
        reader: *const ColumnReader,
        filter: ?FilterCondition,
        agg_specs: []const AggSpec,
    ) ![]f64 {
        var agg = try StreamingAggregator.init(self.allocator, agg_specs);
        defer agg.deinit();

        const total_rows = reader.getRowCount();
        var offset: usize = 0;

        // Process in batches
        while (offset < total_rows) {
            const batch_size = @min(VECTOR_SIZE, total_rows - offset);

            // Load columns for this batch
            for (agg_specs, 0..) |spec, i| {
                const col_type = reader.getColumnType(spec.col_idx);
                self.batch.columns[i] = switch (col_type) {
                    .int64 => .{ .int64 = reader.readBatchI64(spec.col_idx, offset, batch_size) orelse &[_]i64{} },
                    .float64 => .{ .float64 = reader.readBatchF64(spec.col_idx, offset, batch_size) orelse &[_]f64{} },
                    .int32 => .{ .int32 = reader.readBatchI32(spec.col_idx, offset, batch_size) orelse &[_]i32{} },
                    .float32 => .{ .float32 = reader.readBatchF32(spec.col_idx, offset, batch_size) orelse &[_]f32{} },
                    else => .{ .int64 = &[_]i64{} },
                };
            }

            // Reset batch selection to all rows
            self.batch.reset(batch_size, offset);

            // Apply filter if present
            if (filter) |f| {
                const col = self.batch.columns[f.col_idx];
                _ = switch (col) {
                    .int64 => |data| filterI64(data, f.op, f.value.int64, &self.batch.sel, &self.filter_sel),
                    .float64 => |data| filterF64(data, f.op, f.value.float64, &self.batch.sel, &self.filter_sel),
                    else => self.batch.sel.count,
                };
                self.batch.sel = self.filter_sel;
            }

            // Process batch
            agg.processBatch(&self.batch);

            offset += batch_size;
        }

        // Return results
        const results = try self.allocator.alloc(f64, agg_specs.len);
        for (0..agg_specs.len) |i| {
            results[i] = agg.getResult(i);
        }
        return results;
    }

    /// Execute a GROUP BY query
    /// Returns: (keys, agg_results) where agg_results[group][agg_idx]
    pub fn executeGroupBy(
        self: *Self,
        reader: *const ColumnReader,
        key_col_idx: usize,
        filter: ?FilterCondition,
        agg_specs: []const AggSpec,
    ) !struct { keys: []i64, results: [][]f64 } {
        var gb = try StreamingGroupBy.init(self.allocator, key_col_idx, agg_specs, 1024);
        defer gb.deinit();

        const total_rows = reader.getRowCount();
        var offset: usize = 0;

        // Determine columns needed
        const num_cols = agg_specs.len + 1; // +1 for key column

        // Process in batches
        while (offset < total_rows) {
            const batch_size = @min(VECTOR_SIZE, total_rows - offset);

            // Load key column
            const key_type = reader.getColumnType(key_col_idx);
            self.batch.columns[0] = switch (key_type) {
                .int64 => .{ .int64 = reader.readBatchI64(key_col_idx, offset, batch_size) orelse &[_]i64{} },
                else => .{ .int64 = &[_]i64{} },
            };

            // Load aggregate columns
            for (agg_specs, 1..) |spec, i| {
                const col_type = reader.getColumnType(spec.col_idx);
                self.batch.columns[i] = switch (col_type) {
                    .int64 => .{ .int64 = reader.readBatchI64(spec.col_idx, offset, batch_size) orelse &[_]i64{} },
                    .float64 => .{ .float64 = reader.readBatchF64(spec.col_idx, offset, batch_size) orelse &[_]f64{} },
                    else => .{ .int64 = &[_]i64{} },
                };
            }
            _ = num_cols;

            // Reset batch
            self.batch.reset(batch_size, offset);

            // Apply filter if present
            if (filter) |f| {
                // Find column index in batch
                const col = if (f.col_idx == key_col_idx)
                    self.batch.columns[0]
                else blk: {
                    for (agg_specs, 1..) |spec, i| {
                        if (spec.col_idx == f.col_idx) break :blk self.batch.columns[i];
                    }
                    break :blk self.batch.columns[0];
                };
                _ = switch (col) {
                    .int64 => |data| filterI64(data, f.op, f.value.int64, &self.batch.sel, &self.filter_sel),
                    .float64 => |data| filterF64(data, f.op, f.value.float64, &self.batch.sel, &self.filter_sel),
                    else => self.batch.sel.count,
                };
                self.batch.sel = self.filter_sel;
            }

            // Adjust column indices for batch layout (key is at 0, aggs start at 1)
            var adjusted_batch = self.batch;
            adjusted_batch.columns = self.batch.columns;

            // Create adjusted specs with batch-local indices
            const adjusted_specs = try self.allocator.alloc(AggSpec, agg_specs.len);
            defer self.allocator.free(adjusted_specs);
            for (agg_specs, 0..) |spec, i| {
                adjusted_specs[i] = .{
                    .func = spec.func,
                    .col_idx = i + 1, // Offset by 1 since key is at 0
                    .output_name = spec.output_name,
                };
            }

            // Process with adjusted group by
            var adjusted_gb = StreamingGroupBy{
                .ht = gb.ht,
                .groups = gb.groups,
                .specs = adjusted_specs,
                .key_col_idx = 0, // Key is at index 0 in batch
                .keys = gb.keys,
                .allocator = self.allocator,
            };
            try adjusted_gb.processBatch(&adjusted_batch);
            gb.ht = adjusted_gb.ht;
            gb.groups = adjusted_gb.groups;
            gb.keys = adjusted_gb.keys;

            offset += batch_size;
        }

        // Build result arrays
        const num_groups = gb.groupCount();
        const keys = try self.allocator.alloc(i64, num_groups);
        const results = try self.allocator.alloc([]f64, num_groups);

        for (0..num_groups) |g| {
            keys[g] = gb.getGroupKey(g);
            results[g] = try self.allocator.alloc(f64, agg_specs.len);
            for (0..agg_specs.len) |a| {
                results[g][a] = gb.getGroupAgg(g, a, agg_specs[a].func);
            }
        }

        return .{ .keys = keys, .results = results };
    }

    /// Execute COUNT(*) with optional filter - optimized path
    pub fn executeCount(
        self: *Self,
        reader: *const ColumnReader,
        filter: ?FilterCondition,
        filter_col_idx: usize,
    ) !u64 {
        var total_count: u64 = 0;
        const total_rows = reader.getRowCount();
        var offset: usize = 0;

        while (offset < total_rows) {
            const batch_size = @min(VECTOR_SIZE, total_rows - offset);

            if (filter) |f| {
                // Load filter column
                const col_type = reader.getColumnType(filter_col_idx);
                const col_data: ColumnSlice = switch (col_type) {
                    .int64 => .{ .int64 = reader.readBatchI64(filter_col_idx, offset, batch_size) orelse &[_]i64{} },
                    .float64 => .{ .float64 = reader.readBatchF64(filter_col_idx, offset, batch_size) orelse &[_]f64{} },
                    else => .{ .int64 = &[_]i64{} },
                };

                self.batch.columns[0] = col_data;
                self.batch.reset(batch_size, offset);

                // Apply filter
                const count = switch (col_data) {
                    .int64 => |data| filterI64(data, f.op, f.value.int64, &self.batch.sel, &self.filter_sel),
                    .float64 => |data| filterF64(data, f.op, f.value.float64, &self.batch.sel, &self.filter_sel),
                    else => batch_size,
                };
                total_count += count;
            } else {
                total_count += batch_size;
            }

            offset += batch_size;
        }

        return total_count;
    }
};

// ============================================================================
// ArrayColumnReader - Wraps pre-loaded column arrays
// ============================================================================

/// Column data holder for ArrayColumnReader
pub const ColumnData = struct {
    col_type: ColumnType,
    int64_data: ?[]const i64 = null,
    int32_data: ?[]const i32 = null,
    float64_data: ?[]const f64 = null,
    float32_data: ?[]const f32 = null,
    string_data: ?[]const []const u8 = null,
};

/// ArrayColumnReader - wraps pre-loaded column arrays for streaming execution
/// Use this when columns are already loaded (e.g., from Lance/Parquet files)
pub const ArrayColumnReader = struct {
    columns: []const ColumnData,
    row_count: usize,

    const Self = @This();

    pub fn init(columns: []const ColumnData, row_count: usize) Self {
        return .{
            .columns = columns,
            .row_count = row_count,
        };
    }

    /// Get a ColumnReader interface for use with StreamingExecutor
    pub fn reader(self: *Self) ColumnReader {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    const vtable = ColumnReader.VTable{
        .readBatchI64 = readBatchI64Fn,
        .readBatchF64 = readBatchF64Fn,
        .readBatchI32 = readBatchI32Fn,
        .readBatchF32 = readBatchF32Fn,
        .readBatchString = readBatchStringFn,
        .getRowCount = getRowCountFn,
        .getColumnType = getColumnTypeFn,
    };

    fn readBatchI64Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const i64 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return null;
        const data = self.columns[col_idx].int64_data orelse return null;
        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchF64Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const f64 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return null;
        const data = self.columns[col_idx].float64_data orelse return null;
        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchI32Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const i32 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return null;
        const data = self.columns[col_idx].int32_data orelse return null;
        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchF32Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const f32 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return null;
        const data = self.columns[col_idx].float32_data orelse return null;
        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchStringFn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const []const u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return null;
        const data = self.columns[col_idx].string_data orelse return null;
        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn getRowCountFn(ptr: *anyopaque) usize {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.row_count;
    }

    fn getColumnTypeFn(ptr: *anyopaque, col_idx: usize) ColumnType {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return .int64;
        return self.columns[col_idx].col_type;
    }
};

// ============================================================================
// Convenience functions for executing queries
// ============================================================================

/// Execute COUNT(*) WHERE filter on pre-loaded column
pub fn executeCountWhere(
    allocator: std.mem.Allocator,
    filter_col: []const f64,
    op: FilterOp,
    value: f64,
) !u64 {
    const columns = [_]ColumnData{
        .{ .col_type = .float64, .float64_data = filter_col },
    };

    var col_reader = ArrayColumnReader.init(&columns, filter_col.len);
    var reader = col_reader.reader();

    var exec = try StreamingExecutor.init(allocator, 1);
    defer exec.deinit();

    return exec.executeCount(&reader, .{
        .col_idx = 0,
        .op = op,
        .value = .{ .float64 = value },
    }, 0);
}

/// Execute SUM(col) on pre-loaded column
pub fn executeSumF64(
    allocator: std.mem.Allocator,
    col: []const f64,
) !f64 {
    const columns = [_]ColumnData{
        .{ .col_type = .float64, .float64_data = col },
    };

    var col_reader = ArrayColumnReader.init(&columns, col.len);
    var reader = col_reader.reader();

    var exec = try StreamingExecutor.init(allocator, 1);
    defer exec.deinit();

    const specs = [_]AggSpec{
        .{ .func = .sum, .col_idx = 0, .output_name = "sum" },
    };

    const results = try exec.executeAggregate(&reader, null, &specs);
    defer allocator.free(results);

    return results[0];
}

/// Execute GROUP BY with SUM on pre-loaded columns
pub fn executeGroupBySumF64(
    allocator: std.mem.Allocator,
    key_col: []const i64,
    value_col: []const f64,
) !struct { keys: []i64, sums: []f64 } {
    const columns = [_]ColumnData{
        .{ .col_type = .int64, .int64_data = key_col },
        .{ .col_type = .float64, .float64_data = value_col },
    };

    var col_reader = ArrayColumnReader.init(&columns, key_col.len);
    var reader = col_reader.reader();

    var exec = try StreamingExecutor.init(allocator, 2);
    defer exec.deinit();

    const specs = [_]AggSpec{
        .{ .func = .sum, .col_idx = 1, .output_name = "sum" },
    };

    const result = try exec.executeGroupBy(&reader, 0, null, &specs);

    // Extract sums from results
    const sums = try allocator.alloc(f64, result.keys.len);
    for (result.results, 0..) |r, i| {
        sums[i] = r[0];
        allocator.free(r);
    }
    allocator.free(result.results);

    return .{ .keys = result.keys, .sums = sums };
}

// ============================================================================
// Tests
// ============================================================================

test "LinearHashTable basic" {
    const allocator = std.testing.allocator;
    var ht = try LinearHashTable.init(allocator, 100);
    defer ht.deinit(allocator);

    ht.insert(42, 0);
    ht.insert(100, 1);
    ht.insert(42, 2); // Duplicate key

    try std.testing.expectEqual(@as(?u32, 0), ht.probe(42)); // First match
    try std.testing.expectEqual(@as(?u32, 1), ht.probe(100));
    try std.testing.expectEqual(@as(?u32, null), ht.probe(999));
}

test "LinearHashTable buildFromColumn" {
    const allocator = std.testing.allocator;
    const data = [_]i64{ 1, 2, 3, 4, 5, NULL_INT64, 7, 8 };

    var ht = try LinearHashTable.init(allocator, data.len);
    defer ht.deinit(allocator);
    ht.buildFromColumn(&data);

    try std.testing.expectEqual(@as(usize, 7), ht.count); // 7 non-null values
    try std.testing.expectEqual(@as(?u32, 0), ht.probe(1));
    try std.testing.expectEqual(@as(?u32, 4), ht.probe(5));
    try std.testing.expectEqual(@as(?u32, null), ht.probe(6)); // Was NULL
}

test "AggState SIMD" {
    var state = AggState{};
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const sel = SelectionVector.all(data.len);

    state.updateColumnF64(&data, &sel);

    try std.testing.expectEqual(@as(f64, 36.0), state.sum);
    try std.testing.expectEqual(@as(u64, 8), state.count);
    try std.testing.expectEqual(@as(f64, 1.0), state.min);
    try std.testing.expectEqual(@as(f64, 8.0), state.max);
}

test "filterI64 basic" {
    const allocator = std.testing.allocator;
    const data = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const input_sel = SelectionVector.all(data.len);

    var output_sel = try SelectionVector.alloc(allocator, data.len);
    defer output_sel.deinit(allocator);

    const count = filterI64(&data, .gt, 5, &input_sel, &output_sel);

    try std.testing.expectEqual(@as(usize, 3), count); // 6, 7, 8
    try std.testing.expectEqual(@as(u32, 5), output_sel.get(0)); // index of 6
    try std.testing.expectEqual(@as(u32, 6), output_sel.get(1)); // index of 7
    try std.testing.expectEqual(@as(u32, 7), output_sel.get(2)); // index of 8
}

test "filterF64 basic" {
    const allocator = std.testing.allocator;
    const data = [_]f64{ 10.0, 50.0, 100.0, 150.0, 200.0 };
    const input_sel = SelectionVector.all(data.len);

    var output_sel = try SelectionVector.alloc(allocator, data.len);
    defer output_sel.deinit(allocator);

    const count = filterF64(&data, .gt, 100.0, &input_sel, &output_sel);

    try std.testing.expectEqual(@as(usize, 2), count); // 150, 200
}

test "HashGroupBy basic" {
    const allocator = std.testing.allocator;
    const keys = [_]i64{ 1, 2, 1, 2, 1, 3 };
    const values = [_]f64{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 };

    var gb = try HashGroupBy.init(allocator, 10);
    defer gb.deinit();

    try gb.buildGroups(&keys);
    gb.aggregateF64(&keys, &values);

    try std.testing.expectEqual(@as(usize, 3), gb.groupCount()); // 3 distinct keys

    // Find group for key=1 and check sum
    for (0..gb.groupCount()) |i| {
        const key = gb.getGroupKey(i, &keys);
        const agg = gb.getGroupAgg(i);
        if (key == 1) {
            try std.testing.expectEqual(@as(f64, 90.0), agg.sum); // 10+30+50
            try std.testing.expectEqual(@as(u64, 3), agg.count);
        }
    }
}

test "sumI64 with selection" {
    const allocator = std.testing.allocator;
    const data = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8 };

    // Test flat selection
    const sel_all = SelectionVector.all(data.len);
    try std.testing.expectEqual(@as(i64, 36), sumI64(&data, &sel_all));

    // Test with selection vector
    var sel = try SelectionVector.alloc(allocator, 3);
    defer sel.deinit(allocator);
    sel.storage.?[0] = 0; // index 0 -> value 1
    sel.storage.?[1] = 2; // index 2 -> value 3
    sel.storage.?[2] = 4; // index 4 -> value 5
    sel.count = 3;
    sel.indices = sel.storage.?[0..3];

    try std.testing.expectEqual(@as(i64, 9), sumI64(&data, &sel)); // 1+3+5
}
