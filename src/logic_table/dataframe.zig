//! LogicTable DataFrame API
//!
//! Provides a fluent interface for querying @logic_table virtual tables:
//!
//! ```python
//! # Python definition
//! @logic_table
//! class FraudDetector:
//!     orders = Table('orders.lance')
//!
//!     def risk_score(self) -> float:
//!         return self.amount_score() * 0.5 + self.velocity_score() * 0.5
//! ```
//!
//! ```zig
//! // Zig usage
//! const df = LogicTableDataFrame.init(allocator, "fraud_detector.py");
//! const results = df
//!     .filter(.{ .method = "risk_score", .op = .gt, .value = 0.7 })
//!     .select(&.{ "order_id", "risk_score" })
//!     .limit(100)
//!     .collect();
//! ```

const std = @import("std");
const logic_table = @import("logic_table.zig");

const LogicTableContext = logic_table.LogicTableContext;
const LogicTableExecutor = @import("executor.zig").LogicTableExecutor;

/// Filter operation for method results
pub const FilterOp = enum {
    eq, // ==
    ne, // !=
    lt, // <
    le, // <=
    gt, // >
    ge, // >=
};

/// Filter specification for method results
pub const MethodFilter = struct {
    /// Method name (e.g., "risk_score")
    method: []const u8,
    /// Comparison operator
    op: FilterOp,
    /// Threshold value
    value: f64,
};

/// Order specification
pub const OrderSpec = struct {
    /// Column or method name
    column: []const u8,
    /// Whether to sort descending
    descending: bool,
};

/// LogicTable DataFrame - fluent API for @logic_table queries
pub const LogicTableDataFrame = struct {
    allocator: std.mem.Allocator,
    executor: *LogicTableExecutor,

    // Query state
    select_columns: ?[]const []const u8 = null,
    method_filters: std.ArrayList(MethodFilter),
    column_filters: std.ArrayList(ColumnFilter),
    order_specs: ?[]const OrderSpec = null,
    limit_value: ?u64 = null,
    offset_value: ?u64 = null,

    const Self = @This();

    /// Column filter for WHERE clause on data columns
    pub const ColumnFilter = struct {
        table: []const u8,
        column: []const u8,
        op: FilterOp,
        value: FilterValue,
    };

    pub const FilterValue = union(enum) {
        int: i64,
        float: f64,
        string: []const u8,
    };

    /// Initialize DataFrame from Python file path
    pub fn init(allocator: std.mem.Allocator, python_file: []const u8) !Self {
        const executor = try allocator.create(LogicTableExecutor);
        executor.* = try LogicTableExecutor.init(allocator, python_file);

        return Self{
            .allocator = allocator,
            .executor = executor,
            .method_filters = std.ArrayList(MethodFilter).init(allocator),
            .column_filters = std.ArrayList(ColumnFilter).init(allocator),
        };
    }

    /// Initialize from existing executor
    pub fn fromExecutor(allocator: std.mem.Allocator, executor: *LogicTableExecutor) Self {
        return Self{
            .allocator = allocator,
            .executor = executor,
            .method_filters = std.ArrayList(MethodFilter).init(allocator),
            .column_filters = std.ArrayList(ColumnFilter).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.method_filters.deinit();
        self.column_filters.deinit();
        self.executor.deinit();
        self.allocator.destroy(self.executor);
    }

    /// Select specific columns/methods
    pub fn select(self: Self, columns: []const []const u8) Self {
        var new = self;
        new.select_columns = columns;
        return new;
    }

    /// Filter by method result
    /// Example: .filterMethod("risk_score", .gt, 0.7)
    pub fn filterMethod(self: Self, method: []const u8, op: FilterOp, value: f64) !Self {
        var new = self;
        try new.method_filters.append(.{
            .method = method,
            .op = op,
            .value = value,
        });
        return new;
    }

    /// Filter by column value
    /// Example: .filterColumn("orders", "amount", .gt, .{ .float = 100.0 })
    pub fn filterColumn(self: Self, table: []const u8, column: []const u8, op: FilterOp, value: FilterValue) !Self {
        var new = self;
        try new.column_filters.append(.{
            .table = table,
            .column = column,
            .op = op,
            .value = value,
        });
        return new;
    }

    /// Order by column/method
    pub fn orderBy(self: Self, column: []const u8, descending: bool) Self {
        var new = self;
        const specs = self.allocator.alloc(OrderSpec, 1) catch return new;
        specs[0] = .{ .column = column, .descending = descending };
        new.order_specs = specs;
        return new;
    }

    /// Limit results
    pub fn limit(self: Self, n: u64) Self {
        var new = self;
        new.limit_value = n;
        return new;
    }

    /// Offset results
    pub fn offset(self: Self, n: u64) Self {
        var new = self;
        new.offset_value = n;
        return new;
    }

    /// Execute the query and return results
    pub fn collect(self: *Self) !QueryResult {
        // Load tables if not already loaded
        try self.executor.loadTables();

        const row_count = self.executor.getRowCount();
        if (row_count == 0) {
            return QueryResult{
                .allocator = self.allocator,
                .columns = &.{},
                .row_count = 0,
            };
        }

        // Build filtered indices based on column filters
        var filtered_indices = try self.allocator.alloc(u32, row_count);
        defer self.allocator.free(filtered_indices);

        var filtered_count: usize = 0;
        for (0..row_count) |i| {
            if (try self.evaluateColumnFilters(@intCast(i))) {
                filtered_indices[filtered_count] = @intCast(i);
                filtered_count += 1;
            }
        }

        // Apply method filters to narrow down results
        var final_indices = try self.allocator.alloc(u32, filtered_count);
        var final_count: usize = 0;

        for (filtered_indices[0..filtered_count]) |idx| {
            if (try self.evaluateMethodFilters(idx)) {
                final_indices[final_count] = idx;
                final_count += 1;
            }
        }

        // Apply limit/offset
        var start: usize = 0;
        var end: usize = final_count;

        if (self.offset_value) |off| {
            start = @min(off, final_count);
        }
        if (self.limit_value) |lim| {
            end = @min(start + lim, final_count);
        }

        const result_count = end - start;

        return QueryResult{
            .allocator = self.allocator,
            .columns = self.select_columns orelse &.{},
            .row_count = result_count,
            .indices = try self.allocator.dupe(u32, final_indices[start..end]),
        };
    }

    /// Evaluate column filters for a row
    fn evaluateColumnFilters(self: *Self, row_idx: u32) !bool {
        for (self.column_filters.items) |filter| {
            const matches = try self.evaluateColumnFilter(filter, row_idx);
            if (!matches) return false;
        }
        return true;
    }

    fn evaluateColumnFilter(self: *Self, filter: ColumnFilter, row_idx: u32) !bool {
        // Get column value from context
        const ctx = self.executor.getContext();

        // Try to get as f32 first
        if (ctx.getF32(filter.table, filter.column)) |data| {
            const value = data[row_idx];
            const threshold = switch (filter.value) {
                .float => |f| @as(f32, @floatCast(f)),
                .int => |i| @as(f32, @floatFromInt(i)),
                else => return false,
            };
            return self.compareValues(f32, value, threshold, filter.op);
        } else |_| {}

        // Try i64
        if (ctx.getI64(filter.table, filter.column)) |data| {
            const value = data[row_idx];
            const threshold = switch (filter.value) {
                .int => |i| i,
                .float => |f| @as(i64, @intFromFloat(f)),
                else => return false,
            };
            return self.compareValues(i64, value, threshold, filter.op);
        } else |_| {}

        return false;
    }

    /// Evaluate method filters for a row
    fn evaluateMethodFilters(self: *Self, row_idx: u32) !bool {
        _ = row_idx;
        // For now, method filters are evaluated at the batch level
        // Individual row evaluation would require row-wise method calls
        for (self.method_filters.items) |_| {
            // TODO: Implement row-wise method evaluation
            // This requires the compiled method to support row indexing
        }
        return true;
    }

    fn compareValues(self: *Self, comptime T: type, a: T, b: T, op: FilterOp) bool {
        _ = self;
        return switch (op) {
            .eq => a == b,
            .ne => a != b,
            .lt => a < b,
            .le => a <= b,
            .gt => a > b,
            .ge => a >= b,
        };
    }

    /// Get the LogicTableContext for direct column access
    pub fn getContext(self: *Self) *LogicTableContext {
        return self.executor.getContext();
    }

    /// Get row count
    pub fn count(self: *Self) !u64 {
        const result = try self.collect();
        return result.row_count;
    }
};

/// Query result from LogicTableDataFrame.collect()
pub const QueryResult = struct {
    allocator: std.mem.Allocator,
    columns: []const []const u8,
    row_count: usize,
    indices: ?[]const u32 = null,

    pub fn deinit(self: *QueryResult) void {
        if (self.indices) |idx| {
            self.allocator.free(idx);
        }
    }
};

// =============================================================================
// Tests
// =============================================================================

test "LogicTableDataFrame basic" {
    const allocator = std.testing.allocator;

    // Create a simple executor
    var executor = try LogicTableExecutor.init(allocator, "test.py");
    defer executor.deinit();

    // Create DataFrame from executor
    var df = LogicTableDataFrame.fromExecutor(allocator, &executor);
    defer {
        df.method_filters.deinit();
        df.column_filters.deinit();
    }

    // Test method chaining
    const filtered = df.select(&.{ "id", "score" }).limit(10);
    try std.testing.expectEqual(@as(u64, 10), filtered.limit_value.?);
}

test "FilterOp comparison" {
    try std.testing.expect(FilterOp.gt != FilterOp.lt);
    try std.testing.expect(FilterOp.eq == FilterOp.eq);
}
