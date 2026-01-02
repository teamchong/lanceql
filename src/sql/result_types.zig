//! Result Types - Data structures for SQL query results
//!
//! Contains Result, CachedColumn, JoinedData types used by the SQL executor.
//! Also includes LanceColumnType for unified type detection across Lance tables.

const std = @import("std");
const Table = @import("lanceql.table").Table;
pub const logic_table_dispatch = @import("logic_table_dispatch.zig");

/// Lance column types - unified type detection for Lance table logical types
pub const LanceColumnType = enum {
    timestamp_ns,
    timestamp_us,
    timestamp_ms,
    timestamp_s,
    date32,
    date64,
    int32,
    float32,
    bool_,
    int64,
    float64,
    string,
    unsupported,

    /// Detect column type from Lance logical_type string
    /// Precise type detection (order matters - check specific before general)
    pub fn fromLogicalType(logical_type: []const u8) LanceColumnType {
        // Timestamp types (check before generic "int" matches)
        if (std.mem.indexOf(u8, logical_type, "timestamp[ns") != null) return .timestamp_ns;
        if (std.mem.indexOf(u8, logical_type, "timestamp[us") != null) return .timestamp_us;
        if (std.mem.indexOf(u8, logical_type, "timestamp[ms") != null) return .timestamp_ms;
        if (std.mem.indexOf(u8, logical_type, "timestamp[s") != null) return .timestamp_s;
        if (std.mem.indexOf(u8, logical_type, "date32") != null) return .date32;
        if (std.mem.indexOf(u8, logical_type, "date64") != null) return .date64;
        // Explicit int32
        if (std.mem.eql(u8, logical_type, "int32")) return .int32;
        // float or float32
        if (std.mem.eql(u8, logical_type, "float") or std.mem.indexOf(u8, logical_type, "float32") != null) return .float32;
        // bool or boolean
        if (std.mem.eql(u8, logical_type, "bool") or std.mem.indexOf(u8, logical_type, "boolean") != null) return .bool_;
        // Default integers (int, int64, integer)
        if (std.mem.indexOf(u8, logical_type, "int") != null) return .int64;
        // double
        if (std.mem.indexOf(u8, logical_type, "double") != null) return .float64;
        // Strings (utf8 or string)
        if (std.mem.indexOf(u8, logical_type, "utf8") != null or std.mem.indexOf(u8, logical_type, "string") != null) return .string;
        return .unsupported;
    }

};

/// Query result in columnar format
pub const Result = struct {
    columns: []Column,
    row_count: usize,
    allocator: std.mem.Allocator,
    /// If false, column data is owned by executor's cache and should not be freed here
    owns_data: bool = true,

    pub const Column = struct {
        name: []const u8,
        data: ColumnData,
    };

    pub const ColumnData = union(enum) {
        int64: []i64,
        int32: []i32,
        float64: []f64,
        float32: []f32,
        bool_: []bool,
        string: [][]const u8,
        // Timestamp types (all stored as integers, semantic meaning differs)
        timestamp_s: []i64, // seconds since epoch
        timestamp_ms: []i64, // milliseconds since epoch
        timestamp_us: []i64, // microseconds since epoch
        timestamp_ns: []i64, // nanoseconds since epoch
        date32: []i32, // days since epoch
        date64: []i64, // milliseconds since epoch

        /// Get the length of the column data
        pub fn len(self: ColumnData) usize {
            return switch (self) {
                inline else => |data| data.len,
            };
        }

        /// Free the column data using the provided allocator
        pub fn free(self: ColumnData, allocator: std.mem.Allocator) void {
            switch (self) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| allocator.free(data),
                .int32, .date32 => |data| allocator.free(data),
                .float64 => |data| allocator.free(data),
                .float32 => |data| allocator.free(data),
                .bool_ => |data| allocator.free(data),
                .string => |data| {
                    for (data) |str| {
                        allocator.free(str);
                    }
                    allocator.free(data);
                },
            }
        }
    };

    pub fn deinit(self: *Result) void {
        if (self.owns_data) {
            for (self.columns) |col| {
                col.data.free(self.allocator);
            }
        }
        self.allocator.free(self.columns);
    }
};

/// Cached column data
pub const CachedColumn = union(enum) {
    int64: []i64,
    int32: []i32,
    float64: []f64,
    float32: []f32,
    bool_: []bool,
    string: [][]const u8,
    // Timestamp types
    timestamp_s: []i64,
    timestamp_ms: []i64,
    timestamp_us: []i64,
    timestamp_ns: []i64,
    date32: []i32,
    date64: []i64,

    /// Get the length of the column data
    pub fn len(self: CachedColumn) usize {
        return switch (self) {
            inline else => |data| data.len,
        };
    }

    /// Free the column data using the provided allocator
    pub fn free(self: CachedColumn, allocator: std.mem.Allocator) void {
        switch (self) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| allocator.free(data),
            .int32, .date32 => |data| allocator.free(data),
            .float64 => |data| allocator.free(data),
            .float32 => |data| allocator.free(data),
            .bool_ => |data| allocator.free(data),
            .string => |data| {
                for (data) |str| {
                    allocator.free(str);
                }
                allocator.free(data);
            },
        }
    }
};

/// Materialized data from a JOIN operation
pub const JoinedData = struct {
    /// Column data by name (qualified with table alias if present)
    columns: std.StringHashMap(CachedColumn),
    /// Column names in order
    column_names: [][]const u8,
    /// Number of rows in the joined result
    row_count: usize,
    /// Allocator for cleanup
    allocator: std.mem.Allocator,
    /// Left table pointer (for schema access)
    left_table: *Table,

    pub fn deinit(self: *JoinedData) void {
        // Free column data
        var iter = self.columns.valueIterator();
        while (iter.next()) |col| {
            col.free(self.allocator);
        }
        self.columns.deinit();
        // Free column names
        for (self.column_names) |name| {
            self.allocator.free(name);
        }
        self.allocator.free(self.column_names);
    }
};

/// Active table source for query execution
/// Tracks whether we're using a direct table or a logic_table
pub const TableSource = union(enum) {
    /// Direct table (existing behavior - table injected at init)
    direct: *Table,
    /// Logic table with loaded data from Python file
    logic_table: struct {
        executor: *logic_table_dispatch.LogicTableExecutor,
        primary_table: *Table,
        alias: ?[]const u8,
    },
    /// Joined table with materialized data
    joined: *JoinedData,

    pub fn getTable(self: TableSource) *Table {
        return switch (self) {
            .direct => |t| t,
            .logic_table => |lt| lt.primary_table,
            .joined => |jd| jd.left_table,
        };
    }
};
