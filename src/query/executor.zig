//! Query Executor - executes parsed SQL statements against Lance tables.
//!
//! Execution pipeline:
//! 1. Column projection - identify needed columns
//! 2. Scan - read column data
//! 3. Filter - apply WHERE clause
//! 4. Group - GROUP BY aggregation
//! 5. Having - filter groups
//! 6. Sort - ORDER BY
//! 7. Limit - truncate results

const std = @import("std");
const Value = @import("lanceql.value").Value;
const ast = @import("ast.zig");
const SelectStmt = ast.SelectStmt;
const SelectItem = ast.SelectItem;
const OrderBy = ast.OrderBy;
const AggregateType = ast.AggregateType;
const expr_mod = @import("expr.zig");
const Expr = expr_mod.Expr;
const agg_mod = @import("aggregates.zig");
const Aggregate = agg_mod.Aggregate;
const GroupKey = agg_mod.GroupKey;
const GroupKeyContext = agg_mod.GroupKeyContext;

/// Query result set.
pub const ResultSet = struct {
    /// Column names
    columns: [][]const u8,

    /// Row data (array of rows, each row is array of values)
    rows: [][]Value,

    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        for (self.rows) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.rows);
        self.allocator.free(self.columns);
    }

    /// Get row count.
    pub fn rowCount(self: Self) usize {
        return self.rows.len;
    }

    /// Get column count.
    pub fn columnCount(self: Self) usize {
        return self.columns.len;
    }

    /// Format as simple table string.
    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        // Header
        for (self.columns, 0..) |col, i| {
            if (i > 0) try writer.writeAll(" | ");
            try writer.writeAll(col);
        }
        try writer.writeAll("\n");

        // Separator
        for (self.columns, 0..) |col, i| {
            if (i > 0) try writer.writeAll("-+-");
            for (0..col.len) |_| try writer.writeAll("-");
        }
        try writer.writeAll("\n");

        // Rows
        for (self.rows) |row| {
            for (row, 0..) |val, i| {
                if (i > 0) try writer.writeAll(" | ");
                try val.format("", .{}, writer);
            }
            try writer.writeAll("\n");
        }
    }
};

/// Query executor.
pub const Executor = struct {
    allocator: std.mem.Allocator,
    stmt: SelectStmt,

    /// Column data provider interface.
    /// Users implement this to provide column data from their source.
    pub const DataProvider = struct {
        ptr: *anyopaque,
        vtable: *const VTable,

        pub const VTable = struct {
            /// Get column names.
            getColumnNames: *const fn (ptr: *anyopaque) [][]const u8,

            /// Get row count.
            getRowCount: *const fn (ptr: *anyopaque) usize,

            /// Read column as int64 values.
            readInt64Column: *const fn (ptr: *anyopaque, col_idx: usize) ?[]i64,

            /// Read column as float64 values.
            readFloat64Column: *const fn (ptr: *anyopaque, col_idx: usize) ?[]f64,
        };

        pub fn getColumnNames(self: DataProvider) [][]const u8 {
            return self.vtable.getColumnNames(self.ptr);
        }

        pub fn getRowCount(self: DataProvider) usize {
            return self.vtable.getRowCount(self.ptr);
        }

        pub fn readInt64Column(self: DataProvider, col_idx: usize) ?[]i64 {
            return self.vtable.readInt64Column(self.ptr, col_idx);
        }

        pub fn readFloat64Column(self: DataProvider, col_idx: usize) ?[]f64 {
            return self.vtable.readFloat64Column(self.ptr, col_idx);
        }
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, stmt: SelectStmt) Self {
        return .{
            .allocator = allocator,
            .stmt = stmt,
        };
    }

    /// Execute the query against provided data.
    pub fn execute(self: *Self, provider: DataProvider) !ResultSet {
        const col_names = provider.getColumnNames();
        const row_count = provider.getRowCount();

        // Build column name -> index map
        var col_map = std.StringHashMap(usize).init(self.allocator);
        defer col_map.deinit();
        for (col_names, 0..) |name, i| {
            try col_map.put(name, i);
        }

        // Step 1: Determine which columns we need to read
        var needed_cols = std.AutoHashMap(usize, void).init(self.allocator);
        defer needed_cols.deinit();

        // Columns from SELECT
        for (self.stmt.columns) |item| {
            switch (item) {
                .star => {
                    // Need all columns
                    for (0..col_names.len) |i| {
                        try needed_cols.put(i, {});
                    }
                },
                .expr => |e| {
                    try self.collectNeededColumns(e.expression, &col_map, &needed_cols);
                },
            }
        }

        // Columns from WHERE
        if (self.stmt.where) |where| {
            try self.collectNeededColumns(where, &col_map, &needed_cols);
        }

        // Columns from GROUP BY
        for (self.stmt.group_by) |col| {
            if (col_map.get(col)) |idx| {
                try needed_cols.put(idx, {});
            }
        }

        // Step 2: Read column data into memory
        var columns_data: std.ArrayListUnmanaged([]Value) = .empty;
        defer {
            for (columns_data.items) |col_data| {
                self.allocator.free(col_data);
            }
            columns_data.deinit(self.allocator);
        }

        for (0..col_names.len) |col_idx| {
            if (needed_cols.contains(col_idx)) {
                const values = try self.readColumnValues(provider, col_idx, row_count);
                try columns_data.append(self.allocator, values);
            } else {
                // Empty placeholder
                try columns_data.append(self.allocator, &.{});
            }
        }

        // Step 3: Build rows and apply WHERE filter
        var filtered_rows: std.ArrayListUnmanaged([]Value) = .empty;
        defer filtered_rows.deinit(self.allocator);

        for (0..row_count) |row_idx| {
            // Build row
            var row = try self.allocator.alloc(Value, col_names.len);
            for (0..col_names.len) |col_idx| {
                if (columns_data.items[col_idx].len > 0) {
                    row[col_idx] = columns_data.items[col_idx][row_idx];
                } else {
                    row[col_idx] = Value.nil();
                }
            }

            // Apply WHERE filter
            var include = true;
            if (self.stmt.where) |where| {
                const result = where.eval(row, col_map) catch Value.nil();
                include = result.toBool() orelse false;
            }

            if (include) {
                try filtered_rows.append(self.allocator, row);
            } else {
                self.allocator.free(row);
            }
        }

        // Step 4: Handle GROUP BY or simple projection
        var result_rows: [][]Value = undefined;
        var result_cols: [][]const u8 = undefined;

        if (self.stmt.group_by.len > 0) {
            const grouped = try self.executeGroupBy(filtered_rows.items, &col_map);
            result_rows = grouped.rows;
            result_cols = grouped.cols;
        } else if (self.hasAggregates()) {
            // Aggregate without GROUP BY (whole table is one group)
            const agg_result = try self.executeAggregateAll(filtered_rows.items, &col_map);
            result_rows = agg_result.rows;
            result_cols = agg_result.cols;
        } else {
            // Simple projection
            const projected = try self.executeProjection(filtered_rows.items, &col_map, col_names);
            result_rows = projected.rows;
            result_cols = projected.cols;
        }

        // Free filtered rows if not used directly
        if (self.stmt.group_by.len > 0 or self.hasAggregates()) {
            for (filtered_rows.items) |row| {
                self.allocator.free(row);
            }
        }

        // Step 5: Apply HAVING (already done in executeGroupBy if needed)

        // Step 6: ORDER BY
        if (self.stmt.order_by.len > 0) {
            try self.applyOrderBy(&result_rows, result_cols);
        }

        // Step 7: OFFSET and LIMIT
        var final_rows = result_rows;
        if (self.stmt.offset) |offset| {
            if (offset < final_rows.len) {
                // Free skipped rows
                for (final_rows[0..offset]) |row| {
                    self.allocator.free(row);
                }
                final_rows = final_rows[offset..];
            } else {
                for (final_rows) |row| {
                    self.allocator.free(row);
                }
                final_rows = &.{};
            }
        }

        if (self.stmt.limit) |limit| {
            if (limit < final_rows.len) {
                // Free excess rows
                for (final_rows[limit..]) |row| {
                    self.allocator.free(row);
                }
                final_rows = final_rows[0..limit];
            }
        }

        // Reallocate to owned slice
        const owned_rows = try self.allocator.alloc([]Value, final_rows.len);
        @memcpy(owned_rows, final_rows);

        return ResultSet{
            .columns = result_cols,
            .rows = owned_rows,
            .allocator = self.allocator,
        };
    }

    fn collectNeededColumns(
        self: *Self,
        e: *Expr,
        col_map: *std.StringHashMap(usize),
        needed: *std.AutoHashMap(usize, void),
    ) !void {
        _ = self;
        switch (e.*) {
            .column => |name| {
                if (col_map.get(name)) |idx| {
                    try needed.put(idx, {});
                }
            },
            .binary => |b| {
                try @constCast(&Executor{ .allocator = needed.allocator, .stmt = undefined }).collectNeededColumns(b.left, col_map, needed);
                try @constCast(&Executor{ .allocator = needed.allocator, .stmt = undefined }).collectNeededColumns(b.right, col_map, needed);
            },
            .unary => |u| {
                try @constCast(&Executor{ .allocator = needed.allocator, .stmt = undefined }).collectNeededColumns(u.operand, col_map, needed);
            },
            .call => |c| {
                for (c.args) |*arg| {
                    try @constCast(&Executor{ .allocator = needed.allocator, .stmt = undefined }).collectNeededColumns(@constCast(arg), col_map, needed);
                }
            },
            .literal, .star => {},
        }
    }

    fn readColumnValues(self: *Self, provider: DataProvider, col_idx: usize, row_count: usize) ![]Value {
        var values = try self.allocator.alloc(Value, row_count);

        // Try int64 first
        if (provider.readInt64Column(col_idx)) |int_data| {
            for (int_data, 0..) |v, i| {
                values[i] = Value.int(v);
            }
            return values;
        }

        // Try float64
        if (provider.readFloat64Column(col_idx)) |float_data| {
            for (float_data, 0..) |v, i| {
                values[i] = Value.float(v);
            }
            return values;
        }

        // No data available
        for (0..row_count) |i| {
            values[i] = Value.nil();
        }
        return values;
    }

    fn hasAggregates(self: *Self) bool {
        for (self.stmt.columns) |item| {
            switch (item) {
                .expr => |e| {
                    if (e.expression.isAggregate()) return true;
                },
                .star => {},
            }
        }
        return false;
    }

    fn executeProjection(
        self: *Self,
        rows: [][]Value,
        col_map: *std.StringHashMap(usize),
        col_names: [][]const u8,
    ) !struct { rows: [][]Value, cols: [][]const u8 } {
        var result_cols: std.ArrayListUnmanaged([]const u8) = .empty;
        var result_rows: std.ArrayListUnmanaged([]Value) = .empty;

        // Determine output columns
        for (self.stmt.columns) |item| {
            switch (item) {
                .star => {
                    for (col_names) |name| {
                        try result_cols.append(self.allocator, name);
                    }
                },
                .expr => |e| {
                    const name = e.alias orelse switch (e.expression.*) {
                        .column => |n| n,
                        else => "?",
                    };
                    try result_cols.append(self.allocator, name);
                },
            }
        }

        // Project each row
        for (rows) |row| {
            var new_row = try self.allocator.alloc(Value, result_cols.items.len);
            var col_idx: usize = 0;

            for (self.stmt.columns) |item| {
                switch (item) {
                    .star => {
                        for (row) |val| {
                            new_row[col_idx] = val;
                            col_idx += 1;
                        }
                    },
                    .expr => |e| {
                        new_row[col_idx] = e.expression.eval(row, col_map.*) catch Value.nil();
                        col_idx += 1;
                    },
                }
            }

            try result_rows.append(self.allocator, new_row);
        }

        return .{
            .rows = try result_rows.toOwnedSlice(self.allocator),
            .cols = try result_cols.toOwnedSlice(self.allocator),
        };
    }

    fn executeAggregateAll(
        self: *Self,
        rows: [][]Value,
        col_map: *std.StringHashMap(usize),
    ) !struct { rows: [][]Value, cols: [][]const u8 } {
        var result_cols: std.ArrayListUnmanaged([]const u8) = .empty;
        var aggs: std.ArrayListUnmanaged(Aggregate) = .empty;
        defer aggs.deinit(self.allocator);

        // Initialize aggregates for each SELECT item
        for (self.stmt.columns) |item| {
            switch (item) {
                .star => {
                    // COUNT(*) as default
                    try result_cols.append(self.allocator, "count");
                    try aggs.append(self.allocator, Aggregate.init(.count, false));
                },
                .expr => |e| {
                    const name = e.alias orelse "?";
                    try result_cols.append(self.allocator, name);

                    switch (e.expression.*) {
                        .call => |c| {
                            const agg_type = AggregateType.fromStr(c.name) orelse .count;
                            try aggs.append(self.allocator, Aggregate.init(agg_type, c.distinct));
                        },
                        else => {
                            // Non-aggregate in aggregate query - use first value
                            try aggs.append(self.allocator, Aggregate.init(.min, false));
                        },
                    }
                },
            }
        }

        // Process all rows
        for (rows) |row| {
            for (self.stmt.columns, 0..) |item, i| {
                switch (item) {
                    .star => {
                        aggs.items[i].addRow();
                    },
                    .expr => |e| {
                        switch (e.expression.*) {
                            .call => |c| {
                                if (c.args.len > 0) {
                                    const arg = &c.args[0];
                                    if (arg.* == .star) {
                                        aggs.items[i].addRow();
                                    } else {
                                        const val = arg.eval(row, col_map.*) catch Value.nil();
                                        aggs.items[i].add(val);
                                    }
                                } else {
                                    aggs.items[i].addRow();
                                }
                            },
                            else => {
                                const val = e.expression.eval(row, col_map.*) catch Value.nil();
                                aggs.items[i].add(val);
                            },
                        }
                    },
                }
            }
        }

        // Build result row
        var result_row = try self.allocator.alloc(Value, aggs.items.len);
        for (aggs.items, 0..) |agg, i| {
            result_row[i] = agg.result();
        }

        var result_rows = try self.allocator.alloc([]Value, 1);
        result_rows[0] = result_row;

        return .{
            .rows = result_rows,
            .cols = try result_cols.toOwnedSlice(self.allocator),
        };
    }

    fn executeGroupBy(
        self: *Self,
        rows: [][]Value,
        col_map: *std.StringHashMap(usize),
    ) !struct { rows: [][]Value, cols: [][]const u8 } {
        _ = rows;
        _ = col_map;
        // TODO: Implement GROUP BY
        // For now, return empty result
        return .{
            .rows = try self.allocator.alloc([]Value, 0),
            .cols = try self.allocator.alloc([]const u8, 0),
        };
    }

    fn applyOrderBy(self: *Self, rows: *[][]Value, col_names: [][]const u8) !void {
        if (self.stmt.order_by.len == 0) return;

        // Find column indices for ORDER BY
        var order_indices = try self.allocator.alloc(usize, self.stmt.order_by.len);
        defer self.allocator.free(order_indices);

        for (self.stmt.order_by, 0..) |ob, i| {
            for (col_names, 0..) |name, idx| {
                if (std.mem.eql(u8, name, ob.column)) {
                    order_indices[i] = idx;
                    break;
                }
            }
        }

        const order_by = self.stmt.order_by;

        // Sort rows
        std.mem.sort([]Value, rows.*, struct {
            order_indices: []usize,
            order_by: []OrderBy,

            pub fn lessThan(ctx: @This(), a: []Value, b: []Value) bool {
                for (ctx.order_indices, ctx.order_by) |idx, ob| {
                    const cmp = Value.compare(a[idx], b[idx]) orelse continue;
                    if (cmp == .eq) continue;

                    if (ob.descending) {
                        return cmp == .gt;
                    } else {
                        return cmp == .lt;
                    }
                }
                return false;
            }
        }{ .order_indices = order_indices, .order_by = order_by });
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ResultSet format" {
    var result = ResultSet{
        .columns = @constCast(&[_][]const u8{ "id", "name" }),
        .rows = @constCast(&[_][]Value{
            @constCast(&[_]Value{ Value.int(1), Value.str("Alice") }),
            @constCast(&[_]Value{ Value.int(2), Value.str("Bob") }),
        }),
        .allocator = std.testing.allocator,
    };

    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try result.format("", .{}, stream.writer());

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "id") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Alice") != null);
}
