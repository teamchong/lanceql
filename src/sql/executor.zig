//! SQL Executor - Execute parsed SQL queries against Lance files
//!
//! Takes a parsed AST and executes it against Lance columnar files,
//! returning results in columnar format compatible with better-sqlite3.

const std = @import("std");
const ast = @import("ast");
const Table = @import("lanceql.table").Table;

const Expr = ast.Expr;
const SelectStmt = ast.SelectStmt;
const Value = ast.Value;
const BinaryOp = ast.BinaryOp;

/// Aggregate function types
pub const AggregateType = enum {
    count,
    count_star,
    sum,
    avg,
    min,
    max,
};

/// Accumulator for aggregate computations
pub const Accumulator = struct {
    agg_type: AggregateType,
    count: i64,
    sum: f64,
    min_int: ?i64,
    max_int: ?i64,
    min_float: ?f64,
    max_float: ?f64,

    pub fn init(agg_type: AggregateType) Accumulator {
        return Accumulator{
            .agg_type = agg_type,
            .count = 0,
            .sum = 0,
            .min_int = null,
            .max_int = null,
            .min_float = null,
            .max_float = null,
        };
    }

    pub fn addInt(self: *Accumulator, value: i64) void {
        self.count += 1;
        self.sum += @as(f64, @floatFromInt(value));
        if (self.min_int == null or value < self.min_int.?) {
            self.min_int = value;
        }
        if (self.max_int == null or value > self.max_int.?) {
            self.max_int = value;
        }
    }

    pub fn addFloat(self: *Accumulator, value: f64) void {
        self.count += 1;
        self.sum += value;
        if (self.min_float == null or value < self.min_float.?) {
            self.min_float = value;
        }
        if (self.max_float == null or value > self.max_float.?) {
            self.max_float = value;
        }
    }

    pub fn addCount(self: *Accumulator) void {
        self.count += 1;
    }

    pub fn getResult(self: Accumulator) f64 {
        return switch (self.agg_type) {
            .count, .count_star => @as(f64, @floatFromInt(self.count)),
            .sum => self.sum,
            .avg => if (self.count > 0) self.sum / @as(f64, @floatFromInt(self.count)) else 0,
            .min => self.min_float orelse @as(f64, @floatFromInt(self.min_int orelse 0)),
            .max => self.max_float orelse @as(f64, @floatFromInt(self.max_int orelse 0)),
        };
    }

    pub fn getIntResult(self: Accumulator) i64 {
        return switch (self.agg_type) {
            .count, .count_star => self.count,
            .sum => @as(i64, @intFromFloat(self.sum)),
            .avg => if (self.count > 0) @as(i64, @intFromFloat(self.sum / @as(f64, @floatFromInt(self.count)))) else 0,
            .min => self.min_int orelse 0,
            .max => self.max_int orelse 0,
        };
    }
};

/// Query result in columnar format
pub const Result = struct {
    columns: []Column,
    row_count: usize,
    allocator: std.mem.Allocator,

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
    };

    pub fn deinit(self: *Result) void {
        for (self.columns) |col| {
            switch (col.data) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| self.allocator.free(data),
                .int32, .date32 => |data| self.allocator.free(data),
                .float64 => |data| self.allocator.free(data),
                .float32 => |data| self.allocator.free(data),
                .bool_ => |data| self.allocator.free(data),
                .string => |data| {
                    for (data) |str| {
                        self.allocator.free(str);
                    }
                    self.allocator.free(data);
                },
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
};

/// SQL Query Executor
pub const Executor = struct {
    table: *Table,
    allocator: std.mem.Allocator,
    column_cache: std.StringHashMap(CachedColumn),

    const Self = @This();

    pub fn init(table: *Table, allocator: std.mem.Allocator) Self {
        return .{
            .table = table,
            .allocator = allocator,
            .column_cache = std.StringHashMap(CachedColumn).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        // Free cached columns
        var iter = self.column_cache.valueIterator();
        while (iter.next()) |col| {
            switch (col.*) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| self.allocator.free(data),
                .int32, .date32 => |data| self.allocator.free(data),
                .float64 => |data| self.allocator.free(data),
                .float32 => |data| self.allocator.free(data),
                .bool_ => |data| self.allocator.free(data),
                .string => |data| {
                    for (data) |str| {
                        self.allocator.free(str);
                    }
                    self.allocator.free(data);
                },
            }
        }
        self.column_cache.deinit();
    }

    // ========================================================================
    // Column Preloading (for WHERE clause optimization)
    // ========================================================================

    /// Extract all column names referenced in an expression
    fn extractColumnNames(self: *Self, expr: *const Expr, list: *std.ArrayList([]const u8)) anyerror!void {
        switch (expr.*) {
            .column => |col| {
                try list.append(self.allocator, col.name);
            },
            .binary => |bin| {
                try self.extractColumnNames(bin.left, list);
                try self.extractColumnNames(bin.right, list);
            },
            .unary => |un| {
                try self.extractColumnNames(un.operand, list);
            },
            else => {},
        }
    }

    /// Preload columns into cache
    fn preloadColumns(self: *Self, col_names: []const []const u8) !void {
        for (col_names) |name| {
            // Skip if already cached
            if (self.column_cache.contains(name)) continue;

            // Use physical column ID (not array index) for column metadata access
            const physical_col_id = self.table.physicalColumnId(name) orelse return error.ColumnNotFound;
            const field = self.table.getFieldById(physical_col_id) orelse return error.InvalidColumn;

            // Read and cache column based on type
            // Precise type detection (order matters - check specific before general)
            const logical_type = field.logical_type;

            // Timestamp types (check before generic "int" matches)
            if (std.mem.indexOf(u8, logical_type, "timestamp[ns") != null) {
                const data = try self.table.readInt64Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .timestamp_ns = data });
            } else if (std.mem.indexOf(u8, logical_type, "timestamp[us") != null) {
                const data = try self.table.readInt64Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .timestamp_us = data });
            } else if (std.mem.indexOf(u8, logical_type, "timestamp[ms") != null) {
                const data = try self.table.readInt64Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .timestamp_ms = data });
            } else if (std.mem.indexOf(u8, logical_type, "timestamp[s") != null) {
                const data = try self.table.readInt64Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .timestamp_s = data });
            } else if (std.mem.indexOf(u8, logical_type, "date32") != null) {
                const data = try self.table.readInt32Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .date32 = data });
            } else if (std.mem.indexOf(u8, logical_type, "date64") != null) {
                const data = try self.table.readInt64Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .date64 = data });
            } else if (std.mem.eql(u8, logical_type, "int32")) {
                // Explicit int32 type
                const data = try self.table.readInt32Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .int32 = data });
            } else if (std.mem.eql(u8, logical_type, "float") or
                std.mem.indexOf(u8, logical_type, "float32") != null)
            {
                // float or float32 → f32
                const data = try self.table.readFloat32Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .float32 = data });
            } else if (std.mem.eql(u8, logical_type, "bool") or
                std.mem.indexOf(u8, logical_type, "boolean") != null)
            {
                // bool or boolean → bool
                const data = try self.table.readBoolColumn(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .bool_ = data });
            } else if (std.mem.indexOf(u8, logical_type, "int") != null) {
                // Default integers (int, int64, integer) to int64
                const data = try self.table.readInt64Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .int64 = data });
            } else if (std.mem.indexOf(u8, logical_type, "double") != null) {
                // double → float64
                const data = try self.table.readFloat64Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .float64 = data });
            } else if (std.mem.indexOf(u8, logical_type, "utf8") != null or
                std.mem.indexOf(u8, logical_type, "string") != null)
            {
                const data = try self.table.readStringColumn(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .string = data });
            } else {
                return error.UnsupportedColumnType;
            }
        }
    }

    /// Execute a SELECT statement
    pub fn execute(self: *Self, stmt: *const SelectStmt, params: []const Value) !Result {
        // 1. Bind parameters to WHERE clause (replace ? with actual values)
        if (stmt.where) |where_expr| {
            _ = where_expr; // TODO: Implement parameter binding
        }

        // 1.5. Extract and preload columns referenced in WHERE clause
        if (stmt.where) |where_expr| {
            var col_names = std.ArrayList([]const u8){};
            defer col_names.deinit(self.allocator);

            try self.extractColumnNames(&where_expr, &col_names);
            try self.preloadColumns(col_names.items);
        }

        // 2. Apply WHERE clause to get filtered row indices
        const indices = if (stmt.where) |where_expr|
            try self.evaluateWhere(&where_expr, params)
        else
            try self.getAllIndices();

        defer self.allocator.free(indices);

        // 3. Check if we need GROUP BY processing
        const has_group_by = stmt.group_by != null;
        const has_aggregates = self.hasAggregates(stmt.columns);

        if (has_group_by or has_aggregates) {
            // Execute with GROUP BY / aggregation
            return self.executeWithGroupBy(stmt, indices);
        }

        // 4. Read columns based on SELECT list (non-aggregate path)
        var columns = try self.readColumns(stmt.columns, indices);
        var row_count = indices.len;

        // 5. Apply DISTINCT if specified
        if (stmt.distinct) {
            const distinct_result = try self.applyDistinct(columns);
            columns = distinct_result.columns;
            row_count = distinct_result.row_count;
        }

        // 6. Apply ORDER BY (in-memory sorting)
        if (stmt.order_by) |order_by| {
            try self.applyOrderBy(columns, order_by);
        }

        // 7. Apply LIMIT and OFFSET
        const final_row_count = self.applyLimitOffset(columns, stmt.limit, stmt.offset);

        return Result{
            .columns = columns,
            .row_count = final_row_count,
            .allocator = self.allocator,
        };
    }

    // ========================================================================
    // GROUP BY / Aggregate Execution
    // ========================================================================

    /// Check if SELECT list contains any aggregate functions
    fn hasAggregates(self: *Self, select_list: []const ast.SelectItem) bool {
        _ = self;
        for (select_list) |item| {
            if (containsAggregate(&item.expr)) {
                return true;
            }
        }
        return false;
    }

    /// Recursively check if expression contains an aggregate function
    fn containsAggregate(expr: *const Expr) bool {
        return switch (expr.*) {
            .call => |call| blk: {
                // Check if this is an aggregate function
                const is_agg = isAggregateFunction(call.name);
                if (is_agg) break :blk true;

                // Check arguments recursively
                for (call.args) |*arg| {
                    if (containsAggregate(arg)) break :blk true;
                }
                break :blk false;
            },
            .binary => |bin| containsAggregate(bin.left) or containsAggregate(bin.right),
            .unary => |un| containsAggregate(un.operand),
            else => false,
        };
    }

    /// Check if function name is an aggregate function
    fn isAggregateFunction(name: []const u8) bool {
        // Case-insensitive comparison
        if (name.len < 3 or name.len > 5) return false;

        var upper_buf: [8]u8 = undefined;
        const upper_name = std.ascii.upperString(&upper_buf, name);

        return std.mem.eql(u8, upper_name, "COUNT") or
            std.mem.eql(u8, upper_name, "SUM") or
            std.mem.eql(u8, upper_name, "AVG") or
            std.mem.eql(u8, upper_name, "MIN") or
            std.mem.eql(u8, upper_name, "MAX");
    }

    /// Parse aggregate function name to AggregateType
    fn parseAggregateType(name: []const u8, args: []const Expr) AggregateType {
        var upper_buf: [8]u8 = undefined;
        const upper_name = std.ascii.upperString(&upper_buf, name);

        if (std.mem.eql(u8, upper_name, "COUNT")) {
            // COUNT(*) vs COUNT(col)
            if (args.len == 1 and args[0] == .column and
                std.mem.eql(u8, args[0].column.name, "*"))
            {
                return .count_star;
            }
            return .count;
        } else if (std.mem.eql(u8, upper_name, "SUM")) {
            return .sum;
        } else if (std.mem.eql(u8, upper_name, "AVG")) {
            return .avg;
        } else if (std.mem.eql(u8, upper_name, "MIN")) {
            return .min;
        } else if (std.mem.eql(u8, upper_name, "MAX")) {
            return .max;
        }
        return .count; // Default fallback
    }

    /// Execute SELECT with GROUP BY and/or aggregates
    fn executeWithGroupBy(self: *Self, stmt: *const SelectStmt, filtered_indices: []const u32) !Result {
        // Preload all columns we'll need for grouping and aggregates
        try self.preloadGroupByColumns(stmt);

        // Get group by column names (empty if no GROUP BY but has aggregates)
        const group_cols = if (stmt.group_by) |gb| gb.columns else &[_][]const u8{};

        // Build groups: maps group key to list of row indices
        var groups = std.StringHashMap(std.ArrayListUnmanaged(u32)).init(self.allocator);
        defer {
            var iter = groups.valueIterator();
            while (iter.next()) |list| {
                list.deinit(self.allocator);
            }
            groups.deinit();
        }

        // Also need to track key strings for proper cleanup
        var key_strings = std.ArrayListUnmanaged([]const u8){};
        defer {
            for (key_strings.items) |key| {
                self.allocator.free(key);
            }
            key_strings.deinit(self.allocator);
        }

        // Group rows by their group key
        for (filtered_indices) |row_idx| {
            const key = try self.buildGroupKey(group_cols, row_idx);

            if (groups.getPtr(key)) |list| {
                // Existing group - add row index and free the duplicate key
                try list.append(self.allocator, row_idx);
                self.allocator.free(key);
            } else {
                // New group
                var list = std.ArrayListUnmanaged(u32){};
                try list.append(self.allocator, row_idx);
                try groups.put(key, list);
                try key_strings.append(self.allocator, key);
            }
        }

        // If no GROUP BY and no matching rows, return single row with 0/null for aggregates
        const num_groups = if (groups.count() == 0 and group_cols.len == 0)
            @as(usize, 1) // Single aggregate result
        else
            groups.count();

        // Build result columns
        var result_columns = std.ArrayListUnmanaged(Result.Column){};
        errdefer {
            for (result_columns.items) |col| {
                switch (col.data) {
                    .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| self.allocator.free(data),
                    .int32, .date32 => |data| self.allocator.free(data),
                    .float64 => |data| self.allocator.free(data),
                    .float32 => |data| self.allocator.free(data),
                    .bool_ => |data| self.allocator.free(data),
                    .string => |data| {
                        for (data) |str| self.allocator.free(str);
                        self.allocator.free(data);
                    },
                }
            }
            result_columns.deinit(self.allocator);
        }

        // Process each SELECT item
        for (stmt.columns) |item| {
            const col = try self.evaluateSelectItemForGroups(item, &groups, group_cols, num_groups);
            try result_columns.append(self.allocator, col);
        }

        var result = Result{
            .columns = try result_columns.toOwnedSlice(self.allocator),
            .row_count = num_groups,
            .allocator = self.allocator,
        };

        // Apply HAVING clause
        if (stmt.group_by) |gb| {
            if (gb.having) |having_expr| {
                try self.applyHaving(&result, &having_expr, stmt.columns);
            }
        }

        // Apply ORDER BY
        if (stmt.order_by) |order_by| {
            try self.applyOrderBy(result.columns, order_by);
        }

        // Apply LIMIT/OFFSET
        result.row_count = self.applyLimitOffset(result.columns, stmt.limit, stmt.offset);

        return result;
    }

    /// Preload columns needed for GROUP BY and aggregates
    fn preloadGroupByColumns(self: *Self, stmt: *const SelectStmt) !void {
        var col_names = std.ArrayList([]const u8){};
        defer col_names.deinit(self.allocator);

        // Add GROUP BY columns
        if (stmt.group_by) |gb| {
            for (gb.columns) |col| {
                try col_names.append(self.allocator, col);
            }
        }

        // Add columns referenced in SELECT expressions
        for (stmt.columns) |item| {
            try self.extractExprColumnNames(&item.expr, &col_names);
        }

        try self.preloadColumns(col_names.items);
    }

    /// Extract column names from any expression
    fn extractExprColumnNames(self: *Self, expr: *const Expr, list: *std.ArrayList([]const u8)) anyerror!void {
        switch (expr.*) {
            .column => |col| {
                if (!std.mem.eql(u8, col.name, "*")) {
                    try list.append(self.allocator, col.name);
                }
            },
            .binary => |bin| {
                try self.extractExprColumnNames(bin.left, list);
                try self.extractExprColumnNames(bin.right, list);
            },
            .unary => |un| {
                try self.extractExprColumnNames(un.operand, list);
            },
            .call => |call| {
                for (call.args) |*arg| {
                    try self.extractExprColumnNames(arg, list);
                }
            },
            else => {},
        }
    }

    /// Build a group key string from GROUP BY column values for a row
    fn buildGroupKey(self: *Self, group_cols: []const []const u8, row_idx: u32) ![]const u8 {
        if (group_cols.len == 0) {
            // No GROUP BY - all rows in one group
            return try self.allocator.dupe(u8, "__all__");
        }

        var key = std.ArrayList(u8){};
        errdefer key.deinit(self.allocator);

        for (group_cols, 0..) |col_name, i| {
            if (i > 0) try key.append(self.allocator, '|');

            const cached = self.column_cache.get(col_name) orelse return error.ColumnNotCached;

            switch (cached) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| {
                    var buf: [64]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .int32, .date32 => |data| {
                    var buf: [32]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .float64 => |data| {
                    var buf: [64]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .float32 => |data| {
                    var buf: [32]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .bool_ => |data| {
                    const str = if (data[row_idx]) "true" else "false";
                    try key.appendSlice(self.allocator, str);
                },
                .string => |data| {
                    try key.appendSlice(self.allocator, data[row_idx]);
                },
            }
        }

        return key.toOwnedSlice(self.allocator);
    }

    /// Evaluate a SELECT item for all groups
    fn evaluateSelectItemForGroups(
        self: *Self,
        item: ast.SelectItem,
        groups: *std.StringHashMap(std.ArrayList(u32)),
        group_cols: []const []const u8,
        num_groups: usize,
    ) !Result.Column {
        const expr = &item.expr;

        // Handle aggregate function
        if (expr.* == .call and isAggregateFunction(expr.call.name)) {
            return self.evaluateAggregateForGroups(item, groups, num_groups);
        }

        // Handle regular column (must be in GROUP BY)
        if (expr.* == .column) {
            const col_name = expr.column.name;

            // Verify column is in GROUP BY
            var in_group_by = false;
            for (group_cols) |gb_col| {
                if (std.mem.eql(u8, gb_col, col_name)) {
                    in_group_by = true;
                    break;
                }
            }
            if (!in_group_by and group_cols.len > 0) {
                return error.ColumnNotInGroupBy;
            }

            return self.evaluateGroupByColumnForGroups(item, groups, num_groups);
        }

        return error.UnsupportedExpression;
    }

    /// Evaluate an aggregate function for all groups
    fn evaluateAggregateForGroups(
        self: *Self,
        item: ast.SelectItem,
        groups: *std.StringHashMap(std.ArrayList(u32)),
        num_groups: usize,
    ) !Result.Column {
        const call = item.expr.call;
        const agg_type = parseAggregateType(call.name, call.args);

        // Determine column name for the aggregate (if not COUNT(*))
        const agg_col_name: ?[]const u8 = if (agg_type != .count_star and call.args.len > 0)
            if (call.args[0] == .column) call.args[0].column.name else null
        else
            null;

        // Allocate result array
        const results = try self.allocator.alloc(i64, num_groups);
        errdefer self.allocator.free(results);

        // Handle case of no groups (aggregate over empty set or no GROUP BY with data)
        if (groups.count() == 0) {
            // Return 0 for COUNT, null would be better for others but use 0
            results[0] = 0;
            return Result.Column{
                .name = item.alias orelse call.name,
                .data = Result.ColumnData{ .int64 = results },
            };
        }

        // Compute aggregate for each group
        var group_idx: usize = 0;
        var iter = groups.iterator();
        while (iter.next()) |entry| {
            const row_indices = entry.value_ptr.items;

            var acc = Accumulator.init(agg_type);

            for (row_indices) |row_idx| {
                if (agg_type == .count_star) {
                    acc.addCount();
                } else if (agg_col_name) |col_name| {
                    const cached = self.column_cache.get(col_name) orelse return error.ColumnNotCached;

                    switch (cached) {
                        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| acc.addInt(data[row_idx]),
                        .int32, .date32 => |data| acc.addInt(data[row_idx]),
                        .float64 => |data| acc.addFloat(data[row_idx]),
                        .float32 => |data| acc.addFloat(data[row_idx]),
                        .bool_ => acc.addCount(), // COUNT for bools
                        .string => acc.addCount(), // COUNT for strings
                    }
                } else {
                    acc.addCount();
                }
            }

            results[group_idx] = acc.getIntResult();
            group_idx += 1;
        }

        return Result.Column{
            .name = item.alias orelse call.name,
            .data = Result.ColumnData{ .int64 = results },
        };
    }

    /// Evaluate a GROUP BY column for all groups (return first value from each group)
    fn evaluateGroupByColumnForGroups(
        self: *Self,
        item: ast.SelectItem,
        groups: *std.StringHashMap(std.ArrayList(u32)),
        num_groups: usize,
    ) !Result.Column {
        const col_name = item.expr.column.name;
        const cached = self.column_cache.get(col_name) orelse return error.ColumnNotCached;

        // Allocate based on column type
        switch (cached) {
            .int64 => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .int64 = results },
                };
            },
            .timestamp_s => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .timestamp_s = results },
                };
            },
            .timestamp_ms => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .timestamp_ms = results },
                };
            },
            .timestamp_us => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .timestamp_us = results },
                };
            },
            .timestamp_ns => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .timestamp_ns = results },
                };
            },
            .date64 => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .date64 = results },
                };
            },
            .int32 => |source_data| {
                const results = try self.allocator.alloc(i32, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .int32 = results },
                };
            },
            .date32 => |source_data| {
                const results = try self.allocator.alloc(i32, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .date32 = results },
                };
            },
            .float64 => |source_data| {
                const results = try self.allocator.alloc(f64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .float64 = results },
                };
            },
            .float32 => |source_data| {
                const results = try self.allocator.alloc(f32, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .float32 = results },
                };
            },
            .bool_ => |source_data| {
                const results = try self.allocator.alloc(bool, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .bool_ = results },
                };
            },
            .string => |source_data| {
                const results = try self.allocator.alloc([]const u8, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = try self.allocator.dupe(u8, source_data[row_indices[0]]);
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .string = results },
                };
            },
        }
    }

    // ========================================================================
    // WHERE Clause Evaluation
    // ========================================================================

    /// Evaluate WHERE clause and return matching row indices
    fn evaluateWhere(self: *Self, where_expr: *const Expr, params: []const Value) ![]u32 {
        // Bind parameters first
        var bound_expr = try self.bindParameters(where_expr, params);
        defer self.freeExpr(&bound_expr);

        // Get total row count
        const row_count = try self.table.rowCount(0);

        // Evaluate expression for each row
        var matching_indices = std.ArrayList(u32){};
        errdefer matching_indices.deinit(self.allocator);

        var row_idx: u32 = 0;
        while (row_idx < row_count) : (row_idx += 1) {
            const matches = try self.evaluateExprForRow(&bound_expr, row_idx);
            if (matches) {
                try matching_indices.append(self.allocator, row_idx);
            }
        }

        return matching_indices.toOwnedSlice(self.allocator);
    }

    /// Bind parameters (replace ? placeholders with actual values)
    fn bindParameters(self: *Self, expr: *const Expr, params: []const Value) !Expr {
        return switch (expr.*) {
            .value => |val| blk: {
                if (val == .parameter) {
                    const param_idx = val.parameter;
                    if (param_idx >= params.len) return error.ParameterOutOfBounds;
                    break :blk Expr{ .value = params[param_idx] };
                }
                break :blk expr.*;
            },
            .column => expr.*,
            .binary => |bin| blk: {
                const left_ptr = try self.allocator.create(Expr);
                errdefer self.allocator.destroy(left_ptr);
                left_ptr.* = try self.bindParameters(bin.left, params);
                errdefer self.freeExpr(left_ptr);

                const right_ptr = try self.allocator.create(Expr);
                errdefer self.allocator.destroy(right_ptr);
                right_ptr.* = try self.bindParameters(bin.right, params);

                break :blk Expr{
                    .binary = .{
                        .op = bin.op,
                        .left = left_ptr,
                        .right = right_ptr,
                    },
                };
            },
            .unary => |un| blk: {
                const operand_ptr = try self.allocator.create(Expr);
                errdefer self.allocator.destroy(operand_ptr);
                operand_ptr.* = try self.bindParameters(un.operand, params);

                break :blk Expr{
                    .unary = .{
                        .op = un.op,
                        .operand = operand_ptr,
                    },
                };
            },
            .call => |call| blk: {
                // Bind parameters in function arguments
                const new_args = try self.allocator.alloc(Expr, call.args.len);
                errdefer self.allocator.free(new_args);

                for (call.args, 0..) |*arg, i| {
                    new_args[i] = try self.bindParameters(arg, params);
                }

                break :blk Expr{
                    .call = .{
                        .name = call.name,
                        .args = new_args,
                        .distinct = call.distinct,
                        .window = call.window,
                    },
                };
            },
            .in_list => |in| blk: {
                // Bind parameters in IN list
                const new_expr = try self.allocator.create(Expr);
                errdefer self.allocator.destroy(new_expr);
                new_expr.* = try self.bindParameters(in.expr, params);
                errdefer self.freeExpr(new_expr);

                const new_values = try self.allocator.alloc(Expr, in.values.len);
                errdefer self.allocator.free(new_values);

                for (in.values, 0..) |*val, i| {
                    new_values[i] = try self.bindParameters(val, params);
                }

                break :blk Expr{
                    .in_list = .{
                        .expr = new_expr,
                        .values = new_values,
                    },
                };
            },
            .between => |bet| blk: {
                const new_expr = try self.allocator.create(Expr);
                errdefer self.allocator.destroy(new_expr);
                new_expr.* = try self.bindParameters(bet.expr, params);
                errdefer self.freeExpr(new_expr);

                const new_low = try self.allocator.create(Expr);
                errdefer self.allocator.destroy(new_low);
                new_low.* = try self.bindParameters(bet.low, params);
                errdefer self.freeExpr(new_low);

                const new_high = try self.allocator.create(Expr);
                errdefer self.allocator.destroy(new_high);
                new_high.* = try self.bindParameters(bet.high, params);

                break :blk Expr{
                    .between = .{
                        .expr = new_expr,
                        .low = new_low,
                        .high = new_high,
                    },
                };
            },
            // New expression types - pass through for now (execution support TODO)
            .case_expr => expr.*,
            .exists => expr.*,
            .cast => expr.*,
        };
    }

    /// Free allocated expression tree
    fn freeExpr(self: *Self, expr: *Expr) void {
        switch (expr.*) {
            .binary => |bin| {
                self.freeExpr(bin.left);
                self.allocator.destroy(bin.left);
                self.freeExpr(bin.right);
                self.allocator.destroy(bin.right);
            },
            .unary => |un| {
                self.freeExpr(un.operand);
                self.allocator.destroy(un.operand);
            },
            .call => |call| {
                for (call.args) |*arg| {
                    self.freeExpr(arg);
                }
                self.allocator.free(call.args);
            },
            .in_list => |in| {
                self.freeExpr(in.expr);
                self.allocator.destroy(in.expr);
                for (in.values) |*val| {
                    self.freeExpr(val);
                }
                self.allocator.free(in.values);
            },
            .between => |bet| {
                self.freeExpr(bet.expr);
                self.allocator.destroy(bet.expr);
                self.freeExpr(bet.low);
                self.allocator.destroy(bet.low);
                self.freeExpr(bet.high);
                self.allocator.destroy(bet.high);
            },
            else => {},
        }
    }

    /// Evaluate expression for a specific row
    fn evaluateExprForRow(self: *Self, expr: *const Expr, row_idx: u32) anyerror!bool {
        return switch (expr.*) {
            .value => |val| blk: {
                // Literal value - interpret as boolean
                break :blk switch (val) {
                    .integer => |i| i != 0,
                    .float => |f| f != 0.0,
                    .null => false,
                    else => true,
                };
            },
            .column => error.ColumnRequiresComparison,
            .binary => |bin| try self.evaluateBinaryOp(bin.op, bin.left, bin.right, row_idx),
            .unary => |un| try self.evaluateUnaryOp(un.op, un.operand, row_idx),
            else => error.UnsupportedExpression,
        };
    }

    /// Evaluate binary operation
    fn evaluateBinaryOp(
        self: *Self,
        op: BinaryOp,
        left: *const Expr,
        right: *const Expr,
        row_idx: u32,
    ) anyerror!bool {
        return switch (op) {
            .@"and" => (try self.evaluateExprForRow(left, row_idx)) and
                       (try self.evaluateExprForRow(right, row_idx)),
            .@"or" => (try self.evaluateExprForRow(left, row_idx)) or
                      (try self.evaluateExprForRow(right, row_idx)),
            .eq, .ne, .lt, .le, .gt, .ge => try self.evaluateComparison(op, left, right, row_idx),
            else => error.UnsupportedOperator,
        };
    }

    /// Evaluate comparison operation
    fn evaluateComparison(
        self: *Self,
        op: BinaryOp,
        left: *const Expr,
        right: *const Expr,
        row_idx: u32,
    ) !bool {
        const left_val = try self.evaluateToValue(left, row_idx);
        const right_val = try self.evaluateToValue(right, row_idx);

        // Type coercion: compare integers and floats
        return switch (left_val) {
            .integer => |left_int| blk: {
                const right_num = switch (right_val) {
                    .integer => |i| @as(f64, @floatFromInt(i)),
                    .float => |f| f,
                    else => return error.TypeMismatch,
                };
                const left_num = @as(f64, @floatFromInt(left_int));
                break :blk self.compareNumbers(op, left_num, right_num);
            },
            .float => |left_float| blk: {
                const right_num = switch (right_val) {
                    .integer => |i| @as(f64, @floatFromInt(i)),
                    .float => |f| f,
                    else => return error.TypeMismatch,
                };
                break :blk self.compareNumbers(op, left_float, right_num);
            },
            .string => |left_str| blk: {
                const right_str = switch (right_val) {
                    .string => |s| s,
                    else => return error.TypeMismatch,
                };
                break :blk self.compareStrings(op, left_str, right_str);
            },
            .null => op == .ne, // NULL != anything is true, NULL == anything is false
            else => error.UnsupportedType,
        };
    }

    /// Compare two numbers
    fn compareNumbers(self: *Self, op: BinaryOp, left: f64, right: f64) bool {
        _ = self;
        return switch (op) {
            .eq => left == right,
            .ne => left != right,
            .lt => left < right,
            .le => left <= right,
            .gt => left > right,
            .ge => left >= right,
            else => unreachable,
        };
    }

    /// Compare two strings
    fn compareStrings(self: *Self, op: BinaryOp, left: []const u8, right: []const u8) bool {
        _ = self;
        const cmp = std.mem.order(u8, left, right);
        return switch (op) {
            .eq => cmp == .eq,
            .ne => cmp != .eq,
            .lt => cmp == .lt,
            .le => cmp == .lt or cmp == .eq,
            .gt => cmp == .gt,
            .ge => cmp == .gt or cmp == .eq,
            else => unreachable,
        };
    }

    /// Evaluate unary operation
    fn evaluateUnaryOp(
        self: *Self,
        op: ast.UnaryOp,
        operand: *const Expr,
        row_idx: u32,
    ) !bool {
        return switch (op) {
            .not => !(try self.evaluateExprForRow(operand, row_idx)),
            .is_null => blk: {
                const val = try self.evaluateToValue(operand, row_idx);
                break :blk val == .null;
            },
            .is_not_null => blk: {
                const val = try self.evaluateToValue(operand, row_idx);
                break :blk val != .null;
            },
            else => error.UnsupportedOperator,
        };
    }

    /// Evaluate expression to a concrete value (for WHERE clause comparisons)
    fn evaluateToValue(self: *Self, expr: *const Expr, row_idx: u32) !Value {
        return switch (expr.*) {
            .value => expr.value,
            .column => |col| blk: {
                // Lookup in cache instead of reading from table
                const cached = self.column_cache.get(col.name) orelse return error.ColumnNotCached;

                break :blk switch (cached) {
                    .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| Value{ .integer = data[row_idx] },
                    .int32, .date32 => |data| Value{ .integer = data[row_idx] },
                    .float64 => |data| Value{ .float = data[row_idx] },
                    .float32 => |data| Value{ .float = data[row_idx] },
                    .bool_ => |data| Value{ .integer = if (data[row_idx]) 1 else 0 },
                    .string => |data| Value{ .string = data[row_idx] },
                };
            },
            else => error.UnsupportedExpression,
        };
    }

    // ========================================================================
    // Expression Evaluation (for SELECT clause)
    // ========================================================================

    /// Evaluate any expression to a concrete Value for a given row
    /// This handles arithmetic, function calls, and nested expressions
    fn evaluateExprToValue(self: *Self, expr: *const Expr, row_idx: u32) !Value {
        return switch (expr.*) {
            .value => expr.value,
            .column => |col| blk: {
                const cached = self.column_cache.get(col.name) orelse return error.ColumnNotCached;
                break :blk switch (cached) {
                    .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| Value{ .integer = data[row_idx] },
                    .int32, .date32 => |data| Value{ .integer = data[row_idx] },
                    .float64 => |data| Value{ .float = data[row_idx] },
                    .float32 => |data| Value{ .float = data[row_idx] },
                    .bool_ => |data| Value{ .integer = if (data[row_idx]) 1 else 0 },
                    .string => |data| Value{ .string = data[row_idx] },
                };
            },
            .binary => |bin| try self.evaluateBinaryToValue(bin, row_idx),
            .unary => |un| try self.evaluateUnaryToValue(un, row_idx),
            .call => |call| try self.evaluateScalarFunction(call, row_idx),
            else => error.UnsupportedExpression,
        };
    }

    /// Evaluate binary expression to a Value (arithmetic operations)
    fn evaluateBinaryToValue(self: *Self, bin: anytype, row_idx: u32) anyerror!Value {
        const left = try self.evaluateExprToValue(bin.left, row_idx);
        const right = try self.evaluateExprToValue(bin.right, row_idx);

        return switch (bin.op) {
            .add => self.addValues(left, right),
            .subtract => self.subtractValues(left, right),
            .multiply => self.multiplyValues(left, right),
            .divide => self.divideValues(left, right),
            .concat => try self.concatStrings(left, right),
            else => error.UnsupportedOperator,
        };
    }

    /// Evaluate unary expression to a Value
    fn evaluateUnaryToValue(self: *Self, un: anytype, row_idx: u32) anyerror!Value {
        const operand = try self.evaluateExprToValue(un.operand, row_idx);

        return switch (un.op) {
            .minus => self.negateValue(operand),
            .not => blk: {
                // Boolean negation
                const bool_val = switch (operand) {
                    .integer => |i| i != 0,
                    .float => |f| f != 0.0,
                    .null => false,
                    else => true,
                };
                break :blk Value{ .integer = if (bool_val) 0 else 1 };
            },
            else => error.UnsupportedOperator,
        };
    }

    /// Negate a numeric value
    fn negateValue(self: *Self, val: Value) Value {
        _ = self;
        return switch (val) {
            .integer => |i| Value{ .integer = -i },
            .float => |f| Value{ .float = -f },
            else => Value{ .null = {} },
        };
    }

    /// Add two values (int + int = int, int + float = float, float + float = float)
    fn addValues(self: *Self, left: Value, right: Value) Value {
        _ = self;
        return switch (left) {
            .integer => |l| switch (right) {
                .integer => |r| Value{ .integer = l + r },
                .float => |r| Value{ .float = @as(f64, @floatFromInt(l)) + r },
                else => Value{ .null = {} },
            },
            .float => |l| switch (right) {
                .integer => |r| Value{ .float = l + @as(f64, @floatFromInt(r)) },
                .float => |r| Value{ .float = l + r },
                else => Value{ .null = {} },
            },
            else => Value{ .null = {} },
        };
    }

    /// Subtract two values
    fn subtractValues(self: *Self, left: Value, right: Value) Value {
        _ = self;
        return switch (left) {
            .integer => |l| switch (right) {
                .integer => |r| Value{ .integer = l - r },
                .float => |r| Value{ .float = @as(f64, @floatFromInt(l)) - r },
                else => Value{ .null = {} },
            },
            .float => |l| switch (right) {
                .integer => |r| Value{ .float = l - @as(f64, @floatFromInt(r)) },
                .float => |r| Value{ .float = l - r },
                else => Value{ .null = {} },
            },
            else => Value{ .null = {} },
        };
    }

    /// Multiply two values
    fn multiplyValues(self: *Self, left: Value, right: Value) Value {
        _ = self;
        return switch (left) {
            .integer => |l| switch (right) {
                .integer => |r| Value{ .integer = l * r },
                .float => |r| Value{ .float = @as(f64, @floatFromInt(l)) * r },
                else => Value{ .null = {} },
            },
            .float => |l| switch (right) {
                .integer => |r| Value{ .float = l * @as(f64, @floatFromInt(r)) },
                .float => |r| Value{ .float = l * r },
                else => Value{ .null = {} },
            },
            else => Value{ .null = {} },
        };
    }

    /// Divide two values (always returns float for precision)
    fn divideValues(self: *Self, left: Value, right: Value) Value {
        _ = self;
        const left_f = switch (left) {
            .integer => |i| @as(f64, @floatFromInt(i)),
            .float => |f| f,
            else => return Value{ .null = {} },
        };
        const right_f = switch (right) {
            .integer => |i| @as(f64, @floatFromInt(i)),
            .float => |f| f,
            else => return Value{ .null = {} },
        };

        if (right_f == 0) return Value{ .null = {} }; // Division by zero
        return Value{ .float = left_f / right_f };
    }

    /// Concatenate two strings (|| operator)
    fn concatStrings(self: *Self, left: Value, right: Value) !Value {
        const left_str = switch (left) {
            .string => |s| s,
            .integer => |i| blk: {
                var buf: [32]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{i}) catch return error.FormatError;
                break :blk str;
            },
            .float => |f| blk: {
                var buf: [32]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{f}) catch return error.FormatError;
                break :blk str;
            },
            else => return Value{ .null = {} },
        };

        const right_str = switch (right) {
            .string => |s| s,
            .integer => |i| blk: {
                var buf: [32]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{i}) catch return error.FormatError;
                break :blk str;
            },
            .float => |f| blk: {
                var buf: [32]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{f}) catch return error.FormatError;
                break :blk str;
            },
            else => return Value{ .null = {} },
        };

        // Allocate new concatenated string
        const result = try self.allocator.alloc(u8, left_str.len + right_str.len);
        @memcpy(result[0..left_str.len], left_str);
        @memcpy(result[left_str.len..], right_str);

        return Value{ .string = result };
    }

    /// Evaluate scalar function call
    fn evaluateScalarFunction(self: *Self, call: anytype, row_idx: u32) anyerror!Value {
        // Skip aggregates - handled elsewhere
        if (isAggregateFunction(call.name)) {
            return error.AggregateInScalarContext;
        }

        // Evaluate all arguments
        var args: [8]Value = undefined;
        const arg_count = @min(call.args.len, 8);
        for (call.args[0..arg_count], 0..) |*arg, i| {
            args[i] = try self.evaluateExprToValue(arg, row_idx);
        }

        // Dispatch by function name (case-insensitive)
        var upper_buf: [32]u8 = undefined;
        const upper_name = std.ascii.upperString(&upper_buf, call.name);

        // String functions
        if (std.mem.eql(u8, upper_name, "UPPER")) {
            return self.funcUpper(args[0]);
        }
        if (std.mem.eql(u8, upper_name, "LOWER")) {
            return self.funcLower(args[0]);
        }
        if (std.mem.eql(u8, upper_name, "LENGTH")) {
            return self.funcLength(args[0]);
        }
        if (std.mem.eql(u8, upper_name, "TRIM")) {
            return self.funcTrim(args[0]);
        }

        // Math functions
        if (std.mem.eql(u8, upper_name, "ABS")) {
            return self.funcAbs(args[0]);
        }
        if (std.mem.eql(u8, upper_name, "ROUND")) {
            const precision: i32 = if (arg_count > 1) switch (args[1]) {
                .integer => |i| @intCast(i),
                else => 0,
            } else 0;
            return self.funcRound(args[0], precision);
        }
        if (std.mem.eql(u8, upper_name, "FLOOR")) {
            return self.funcFloor(args[0]);
        }
        if (std.mem.eql(u8, upper_name, "CEIL") or std.mem.eql(u8, upper_name, "CEILING")) {
            return self.funcCeil(args[0]);
        }

        // Type functions
        if (std.mem.eql(u8, upper_name, "COALESCE")) {
            // Return first non-null value
            for (args[0..arg_count]) |arg| {
                if (arg != .null) return arg;
            }
            return Value{ .null = {} };
        }

        return error.UnknownFunction;
    }

    /// UPPER(string) - Convert string to uppercase
    fn funcUpper(self: *Self, val: Value) Value {
        const str = switch (val) {
            .string => |s| s,
            else => return Value{ .null = {} },
        };

        const result = self.allocator.alloc(u8, str.len) catch return Value{ .null = {} };
        for (str, 0..) |c, i| {
            result[i] = std.ascii.toUpper(c);
        }

        return Value{ .string = result };
    }

    /// LOWER(string) - Convert string to lowercase
    fn funcLower(self: *Self, val: Value) Value {
        const str = switch (val) {
            .string => |s| s,
            else => return Value{ .null = {} },
        };

        const result = self.allocator.alloc(u8, str.len) catch return Value{ .null = {} };
        for (str, 0..) |c, i| {
            result[i] = std.ascii.toLower(c);
        }

        return Value{ .string = result };
    }

    /// LENGTH(string) - Return string length
    fn funcLength(self: *Self, val: Value) Value {
        _ = self;
        return switch (val) {
            .string => |s| Value{ .integer = @intCast(s.len) },
            else => Value{ .null = {} },
        };
    }

    /// TRIM(string) - Remove leading/trailing whitespace
    fn funcTrim(self: *Self, val: Value) Value {
        const str = switch (val) {
            .string => |s| s,
            else => return Value{ .null = {} },
        };

        const trimmed = std.mem.trim(u8, str, " \t\n\r");
        const result = self.allocator.dupe(u8, trimmed) catch return Value{ .null = {} };

        return Value{ .string = result };
    }

    /// ABS(number) - Absolute value
    fn funcAbs(self: *Self, val: Value) Value {
        _ = self;
        return switch (val) {
            .integer => |i| Value{ .integer = if (i < 0) -i else i },
            .float => |f| Value{ .float = @abs(f) },
            else => Value{ .null = {} },
        };
    }

    /// ROUND(number, precision) - Round to precision decimal places
    fn funcRound(self: *Self, val: Value, precision: i32) Value {
        _ = self;
        const f = switch (val) {
            .integer => |i| @as(f64, @floatFromInt(i)),
            .float => |f| f,
            else => return Value{ .null = {} },
        };

        const multiplier = std.math.pow(f64, 10.0, @floatFromInt(precision));
        return Value{ .float = @round(f * multiplier) / multiplier };
    }

    /// FLOOR(number) - Round down
    fn funcFloor(self: *Self, val: Value) Value {
        _ = self;
        return switch (val) {
            .integer => val,
            .float => |f| Value{ .float = @floor(f) },
            else => Value{ .null = {} },
        };
    }

    /// CEIL(number) - Round up
    fn funcCeil(self: *Self, val: Value) Value {
        _ = self;
        return switch (val) {
            .integer => val,
            .float => |f| Value{ .float = @ceil(f) },
            else => Value{ .null = {} },
        };
    }

    // ========================================================================
    // Type Inference for Expressions
    // ========================================================================

    /// Result type enum for expression type inference
    const ResultType = enum {
        int64,
        float64,
        string,
    };

    /// Infer the result type of an expression
    fn inferExpressionType(self: *Self, expr: *const Expr) !ResultType {
        return switch (expr.*) {
            .value => |v| switch (v) {
                .integer => .int64,
                .float => .float64,
                .string => .string,
                else => .string,
            },
            .column => |col| blk: {
                const cached = self.column_cache.get(col.name) orelse {
                    // Not cached yet, look up from table
                    const physical_col_id = self.table.physicalColumnId(col.name) orelse return error.ColumnNotFound;
                    const field = self.table.getFieldById(physical_col_id) orelse return error.InvalidColumn;

                    if (std.mem.indexOf(u8, field.logical_type, "int") != null) {
                        break :blk .int64;
                    } else if (std.mem.indexOf(u8, field.logical_type, "float") != null or
                              std.mem.indexOf(u8, field.logical_type, "double") != null) {
                        break :blk .float64;
                    } else {
                        break :blk .string;
                    }
                };

                break :blk switch (cached) {
                    .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => .int64,
                    .int32, .date32 => .int64, // int32/date32 promoted to int64 for expressions
                    .float64 => .float64,
                    .float32 => .float64, // float32 promoted to float64 for expressions
                    .bool_ => .int64, // bool treated as integer
                    .string => .string,
                };
            },
            .binary => |bin| try self.inferBinaryType(bin),
            .unary => |un| try self.inferExpressionType(un.operand),
            .call => |call| self.inferFunctionReturnType(call.name),
            else => .string,
        };
    }

    /// Infer type of binary expression
    fn inferBinaryType(self: *Self, bin: anytype) anyerror!ResultType {
        // Concat always returns string
        if (bin.op == .concat) return .string;

        const left_type = try self.inferExpressionType(bin.left);
        const right_type = try self.inferExpressionType(bin.right);

        // Division always returns float
        if (bin.op == .divide) return .float64;

        // If either operand is float, result is float
        if (left_type == .float64 or right_type == .float64) return .float64;

        // Both integers -> integer
        if (left_type == .int64 and right_type == .int64) return .int64;

        // Default to float for mixed/unknown types
        return .float64;
    }

    /// Infer return type of a scalar function
    fn inferFunctionReturnType(self: *Self, name: []const u8) ResultType {
        _ = self;
        var upper_buf: [32]u8 = undefined;
        const upper_name = std.ascii.upperString(&upper_buf, name);

        // String functions return string
        if (std.mem.eql(u8, upper_name, "UPPER") or
            std.mem.eql(u8, upper_name, "LOWER") or
            std.mem.eql(u8, upper_name, "TRIM"))
        {
            return .string;
        }

        // LENGTH returns int
        if (std.mem.eql(u8, upper_name, "LENGTH")) {
            return .int64;
        }

        // Math functions typically return float (except ABS which preserves type)
        // For simplicity, return float64 for all math functions
        return .float64;
    }

    /// Evaluate an expression column for all filtered indices
    fn evaluateExpressionColumn(
        self: *Self,
        item: ast.SelectItem,
        indices: []const u32,
    ) !Result.Column {
        // First, preload any columns referenced in the expression
        var col_names = std.ArrayList([]const u8){};
        defer col_names.deinit(self.allocator);
        try self.extractExprColumnNames(&item.expr, &col_names);
        try self.preloadColumns(col_names.items);

        // Infer result type
        const result_type = try self.inferExpressionType(&item.expr);

        // Generate column name from expression if no alias
        const col_name = item.alias orelse "expr";

        // Evaluate expression for each row and store results
        switch (result_type) {
            .int64 => {
                const results = try self.allocator.alloc(i64, indices.len);
                errdefer self.allocator.free(results);

                for (indices, 0..) |row_idx, i| {
                    const val = try self.evaluateExprToValue(&item.expr, row_idx);
                    results[i] = switch (val) {
                        .integer => |v| v,
                        .float => |f| @intFromFloat(f),
                        else => 0,
                    };
                }

                return Result.Column{
                    .name = col_name,
                    .data = Result.ColumnData{ .int64 = results },
                };
            },
            .float64 => {
                const results = try self.allocator.alloc(f64, indices.len);
                errdefer self.allocator.free(results);

                for (indices, 0..) |row_idx, i| {
                    const val = try self.evaluateExprToValue(&item.expr, row_idx);
                    results[i] = switch (val) {
                        .integer => |v| @floatFromInt(v),
                        .float => |f| f,
                        else => 0.0,
                    };
                }

                return Result.Column{
                    .name = col_name,
                    .data = Result.ColumnData{ .float64 = results },
                };
            },
            .string => {
                const results = try self.allocator.alloc([]const u8, indices.len);
                errdefer self.allocator.free(results);

                // Check if expression produces owned strings (e.g., concat, UPPER)
                // by checking if it's a binary concat or a function call
                const expr_produces_owned = switch (item.expr) {
                    .binary => |bin| bin.op == .concat,
                    .call => true, // String functions allocate their results
                    else => false,
                };

                for (indices, 0..) |row_idx, i| {
                    const val = try self.evaluateExprToValue(&item.expr, row_idx);
                    results[i] = switch (val) {
                        .string => |s| blk: {
                            if (expr_produces_owned) {
                                // String was already allocated by concat/UPPER/etc.
                                // Use it directly without duping
                                break :blk s;
                            } else {
                                // String is borrowed from cache, must dupe
                                break :blk try self.allocator.dupe(u8, s);
                            }
                        },
                        .integer => |v| blk: {
                            var buf: [32]u8 = undefined;
                            const str = std.fmt.bufPrint(&buf, "{d}", .{v}) catch "";
                            break :blk try self.allocator.dupe(u8, str);
                        },
                        .float => |f| blk: {
                            var buf: [32]u8 = undefined;
                            const str = std.fmt.bufPrint(&buf, "{d}", .{f}) catch "";
                            break :blk try self.allocator.dupe(u8, str);
                        },
                        else => try self.allocator.dupe(u8, ""),
                    };
                }

                return Result.Column{
                    .name = col_name,
                    .data = Result.ColumnData{ .string = results },
                };
            },
        }
    }

    /// Get all row indices (0, 1, 2, ..., n-1)
    fn getAllIndices(self: *Self) ![]u32 {
        // Get row count from first column
        const row_count = try self.table.rowCount(0);
        const indices = try self.allocator.alloc(u32, @intCast(row_count));

        for (indices, 0..) |*idx, i| {
            idx.* = @intCast(i);
        }

        return indices;
    }

    // ========================================================================
    // Column Reading
    // ========================================================================

    /// Read columns based on SELECT list and filtered indices
    fn readColumns(
        self: *Self,
        select_list: []const ast.SelectItem,
        indices: []const u32,
    ) ![]Result.Column {
        var columns = std.ArrayList(Result.Column){};
        errdefer {
            for (columns.items) |col| {
                switch (col.data) {
                    .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| self.allocator.free(data),
                    .int32, .date32 => |data| self.allocator.free(data),
                    .float64 => |data| self.allocator.free(data),
                    .float32 => |data| self.allocator.free(data),
                    .bool_ => |data| self.allocator.free(data),
                    .string => |data| {
                        for (data) |str| {
                            self.allocator.free(str);
                        }
                        self.allocator.free(data);
                    },
                }
            }
            columns.deinit(self.allocator);
        }

        for (select_list) |item| {
            // Handle SELECT *
            if (item.expr == .column and std.mem.eql(u8, item.expr.column.name, "*")) {
                const col_names = try self.table.columnNames();
                defer self.allocator.free(col_names);

                for (col_names) |col_name| {
                    // Look up the physical column ID from the name
                    // The physical column ID maps to the column metadata index
                    const physical_col_id = self.table.physicalColumnId(col_name) orelse return error.ColumnNotFound;
                    const data = try self.readColumnAtIndices(physical_col_id, indices);

                    try columns.append(self.allocator, Result.Column{
                        .name = col_name,
                        .data = data,
                    });
                }
                break; // SELECT * means we're done
            }

            // Handle regular column
            if (item.expr == .column) {
                const col_name = item.expr.column.name;
                const col_idx = self.table.physicalColumnId(col_name) orelse return error.ColumnNotFound;
                const data = try self.readColumnAtIndices(col_idx, indices);

                try columns.append(self.allocator, Result.Column{
                    .name = item.alias orelse col_name,
                    .data = data,
                });
            } else {
                // Handle expressions (arithmetic, functions, etc.)
                const expr_col = try self.evaluateExpressionColumn(item, indices);
                try columns.append(self.allocator, expr_col);
            }
        }

        return columns.toOwnedSlice(self.allocator);
    }

    /// Read column data at specific row indices
    fn readColumnAtIndices(
        self: *Self,
        col_idx: u32,
        indices: []const u32,
    ) !Result.ColumnData {
        const field = self.table.getFieldById(col_idx) orelse return error.InvalidColumn;

        // Phase 2: Table API now has readAtIndices() methods
        // Current implementation still uses inline filtering for type-specific handling
        // Future: refactor to use Table.readInt64AtIndices() etc. for cleaner code

        const logical_type = field.logical_type;

        // Precise type detection (order matters - check specific before general)
        // Timestamp types (check before generic "int" matches)
        if (std.mem.indexOf(u8, logical_type, "timestamp[ns") != null) {
            const all_data = try self.table.readInt64Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(i64, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .timestamp_ns = filtered };
        } else if (std.mem.indexOf(u8, logical_type, "timestamp[us") != null) {
            const all_data = try self.table.readInt64Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(i64, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .timestamp_us = filtered };
        } else if (std.mem.indexOf(u8, logical_type, "timestamp[ms") != null) {
            const all_data = try self.table.readInt64Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(i64, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .timestamp_ms = filtered };
        } else if (std.mem.indexOf(u8, logical_type, "timestamp[s") != null) {
            const all_data = try self.table.readInt64Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(i64, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .timestamp_s = filtered };
        } else if (std.mem.indexOf(u8, logical_type, "date32") != null) {
            const all_data = try self.table.readInt32Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(i32, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .date32 = filtered };
        } else if (std.mem.indexOf(u8, logical_type, "date64") != null) {
            const all_data = try self.table.readInt64Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(i64, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .date64 = filtered };
        } else if (std.mem.eql(u8, logical_type, "int32")) {
            const all_data = try self.table.readInt32Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(i32, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .int32 = filtered };
        } else if (std.mem.eql(u8, logical_type, "float") or
            std.mem.indexOf(u8, logical_type, "float32") != null) {
            const all_data = try self.table.readFloat32Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(f32, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .float32 = filtered };
        } else if (std.mem.eql(u8, logical_type, "bool") or
            std.mem.indexOf(u8, logical_type, "boolean") != null) {
            const all_data = try self.table.readBoolColumn(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(bool, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .bool_ = filtered };
        } else if (std.mem.indexOf(u8, logical_type, "int") != null) {
            // Default integers to int64
            const all_data = try self.table.readInt64Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(i64, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .int64 = filtered };
        } else if (std.mem.indexOf(u8, logical_type, "double") != null) {
            const all_data = try self.table.readFloat64Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(f64, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .float64 = filtered };
        } else if (std.mem.indexOf(u8, logical_type, "utf8") != null or
            std.mem.indexOf(u8, logical_type, "string") != null) {
            const all_data = try self.table.readStringColumn(col_idx);
            // all_data contains owned strings - must free both array and individual strings
            defer {
                for (all_data) |str| {
                    self.allocator.free(str);
                }
                self.allocator.free(all_data);
            }

            const filtered = try self.allocator.alloc([]const u8, indices.len);
            for (indices, 0..) |idx, i| {
                // Duplicate string for the filtered result
                filtered[i] = try self.allocator.dupe(u8, all_data[idx]);
            }
            return Result.ColumnData{ .string = filtered };
        } else {
            return error.UnsupportedColumnType;
        }
    }

    // ========================================================================
    // DISTINCT Implementation
    // ========================================================================

    /// Apply DISTINCT - remove duplicate rows from result columns
    fn applyDistinct(self: *Self, columns: []Result.Column) !struct {
        columns: []Result.Column,
        row_count: usize,
    } {
        if (columns.len == 0) {
            return .{ .columns = columns, .row_count = 0 };
        }

        const total_rows = switch (columns[0].data) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |d| d.len,
            .int32, .date32 => |d| d.len,
            .float64 => |d| d.len,
            .float32 => |d| d.len,
            .bool_ => |d| d.len,
            .string => |d| d.len,
        };

        if (total_rows == 0) {
            return .{ .columns = columns, .row_count = 0 };
        }

        // Track unique row keys using StringHashMap
        var seen = std.StringHashMap(void).init(self.allocator);
        defer {
            // Free all keys stored in the map
            var key_iter = seen.keyIterator();
            while (key_iter.next()) |key| {
                self.allocator.free(key.*);
            }
            seen.deinit();
        }

        // Track which row indices to keep
        var keep_indices = std.ArrayList(usize){};
        defer keep_indices.deinit(self.allocator);

        // Build row keys and identify unique rows
        for (0..total_rows) |row_idx| {
            const row_key = try self.buildDistinctRowKey(columns, row_idx);

            if (!seen.contains(row_key)) {
                // Store owned copy of key in map
                try seen.put(row_key, {});
                try keep_indices.append(self.allocator, row_idx);
            } else {
                // Key already exists, free the duplicate
                self.allocator.free(row_key);
            }
        }

        // If all rows are unique, return original columns
        if (keep_indices.items.len == total_rows) {
            return .{ .columns = columns, .row_count = total_rows };
        }

        // Build new columns with only unique rows
        const unique_count = keep_indices.items.len;
        const new_columns = try self.allocator.alloc(Result.Column, columns.len);
        errdefer self.allocator.free(new_columns);

        for (columns, 0..) |col, col_idx| {
            new_columns[col_idx] = try self.filterColumnByIndices(col, keep_indices.items);
        }

        // Free original column data
        for (columns) |col| {
            switch (col.data) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |d| self.allocator.free(d),
                .int32, .date32 => |d| self.allocator.free(d),
                .float64 => |d| self.allocator.free(d),
                .float32 => |d| self.allocator.free(d),
                .bool_ => |d| self.allocator.free(d),
                .string => |d| {
                    for (d) |str| self.allocator.free(str);
                    self.allocator.free(d);
                },
            }
        }
        self.allocator.free(columns);

        return .{ .columns = new_columns, .row_count = unique_count };
    }

    /// Build a unique key string for a row across all columns (for DISTINCT)
    fn buildDistinctRowKey(self: *Self, columns: []const Result.Column, row_idx: usize) ![]u8 {
        var key = std.ArrayList(u8){};
        errdefer key.deinit(self.allocator);

        for (columns) |col| {
            try key.append(self.allocator, '|'); // Column separator
            switch (col.data) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |vals| {
                    var buf: [64]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{vals[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .int32, .date32 => |vals| {
                    var buf: [32]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{vals[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .float64 => |vals| {
                    var buf: [64]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{vals[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .float32 => |vals| {
                    var buf: [32]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{vals[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .bool_ => |vals| {
                    try key.appendSlice(self.allocator, if (vals[row_idx]) "true" else "false");
                },
                .string => |vals| {
                    try key.appendSlice(self.allocator, vals[row_idx]);
                },
            }
        }

        return key.toOwnedSlice(self.allocator);
    }

    /// Filter a column to keep only specified row indices
    fn filterColumnByIndices(self: *Self, col: Result.Column, indices: []const usize) !Result.Column {
        const count = indices.len;

        return Result.Column{
            .name = col.name,
            .data = switch (col.data) {
                .int64 => |vals| blk: {
                    const new_vals = try self.allocator.alloc(i64, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .int64 = new_vals };
                },
                .timestamp_s => |vals| blk: {
                    const new_vals = try self.allocator.alloc(i64, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .timestamp_s = new_vals };
                },
                .timestamp_ms => |vals| blk: {
                    const new_vals = try self.allocator.alloc(i64, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .timestamp_ms = new_vals };
                },
                .timestamp_us => |vals| blk: {
                    const new_vals = try self.allocator.alloc(i64, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .timestamp_us = new_vals };
                },
                .timestamp_ns => |vals| blk: {
                    const new_vals = try self.allocator.alloc(i64, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .timestamp_ns = new_vals };
                },
                .date64 => |vals| blk: {
                    const new_vals = try self.allocator.alloc(i64, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .date64 = new_vals };
                },
                .int32 => |vals| blk: {
                    const new_vals = try self.allocator.alloc(i32, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .int32 = new_vals };
                },
                .date32 => |vals| blk: {
                    const new_vals = try self.allocator.alloc(i32, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .date32 = new_vals };
                },
                .float64 => |vals| blk: {
                    const new_vals = try self.allocator.alloc(f64, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .float64 = new_vals };
                },
                .float32 => |vals| blk: {
                    const new_vals = try self.allocator.alloc(f32, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .float32 = new_vals };
                },
                .bool_ => |vals| blk: {
                    const new_vals = try self.allocator.alloc(bool, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = vals[idx];
                    }
                    break :blk Result.ColumnData{ .bool_ = new_vals };
                },
                .string => |vals| blk: {
                    const new_vals = try self.allocator.alloc([]const u8, count);
                    for (indices, 0..) |idx, i| {
                        new_vals[i] = try self.allocator.dupe(u8, vals[idx]);
                    }
                    break :blk Result.ColumnData{ .string = new_vals };
                },
            },
        };
    }

    // ========================================================================
    // ORDER BY Implementation
    // ========================================================================

    fn applyOrderBy(
        self: *Self,
        columns: []Result.Column,
        order_by: []const ast.OrderBy,
    ) !void {
        if (columns.len == 0) return;

        const row_count = switch (columns[0].data) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| data.len,
            .int32, .date32 => |data| data.len,
            .float64 => |data| data.len,
            .float32 => |data| data.len,
            .bool_ => |data| data.len,
            .string => |data| data.len,
        };

        if (row_count == 0) return;

        // Create array of indices [0, 1, 2, ..., n-1]
        const indices = try self.allocator.alloc(usize, row_count);
        defer self.allocator.free(indices);

        for (indices, 0..) |*idx, i| {
            idx.* = i;
        }

        // Sort indices based on order_by columns
        for (order_by) |order| {
            const sort_col_idx = self.findColumnIndex(columns, order.column) orelse continue;
            const sort_col = &columns[sort_col_idx];

            // Sort using comparison function
            const context = SortContext{
                .column = sort_col,
                .direction = order.direction,
            };

            std.mem.sort(usize, indices, context, sortCompare);
        }

        // Reorder all columns based on sorted indices
        for (columns) |*col| {
            try self.reorderColumn(col, indices);
        }
    }

    const SortContext = struct {
        column: *const Result.Column,
        direction: ast.OrderDirection,
    };

    fn sortCompare(context: SortContext, a_idx: usize, b_idx: usize) bool {
        const ascending = context.direction == .asc;

        const cmp = switch (context.column.data) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| blk: {
                const a = data[a_idx];
                const b = data[b_idx];
                if (a < b) break :blk std.math.Order.lt;
                if (a > b) break :blk std.math.Order.gt;
                break :blk std.math.Order.eq;
            },
            .int32, .date32 => |data| blk: {
                const a = data[a_idx];
                const b = data[b_idx];
                if (a < b) break :blk std.math.Order.lt;
                if (a > b) break :blk std.math.Order.gt;
                break :blk std.math.Order.eq;
            },
            .float64 => |data| blk: {
                const a = data[a_idx];
                const b = data[b_idx];
                if (a < b) break :blk std.math.Order.lt;
                if (a > b) break :blk std.math.Order.gt;
                break :blk std.math.Order.eq;
            },
            .float32 => |data| blk: {
                const a = data[a_idx];
                const b = data[b_idx];
                if (a < b) break :blk std.math.Order.lt;
                if (a > b) break :blk std.math.Order.gt;
                break :blk std.math.Order.eq;
            },
            .bool_ => |data| blk: {
                const a: u8 = if (data[a_idx]) 1 else 0;
                const b: u8 = if (data[b_idx]) 1 else 0;
                if (a < b) break :blk std.math.Order.lt;
                if (a > b) break :blk std.math.Order.gt;
                break :blk std.math.Order.eq;
            },
            .string => |data| std.mem.order(u8, data[a_idx], data[b_idx]),
        };

        return if (ascending)
            cmp == .lt
        else
            cmp == .gt;
    }

    fn findColumnIndex(self: *Self, columns: []const Result.Column, name: []const u8) ?usize {
        _ = self;
        for (columns, 0..) |col, i| {
            if (std.mem.eql(u8, col.name, name)) {
                return i;
            }
        }
        return null;
    }

    fn reorderColumn(self: *Self, col: *Result.Column, indices: []const usize) !void {
        switch (col.data) {
            .int64 => |data| {
                const reordered = try self.allocator.alloc(i64, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .int64 = reordered };
            },
            .timestamp_s => |data| {
                const reordered = try self.allocator.alloc(i64, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .timestamp_s = reordered };
            },
            .timestamp_ms => |data| {
                const reordered = try self.allocator.alloc(i64, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .timestamp_ms = reordered };
            },
            .timestamp_us => |data| {
                const reordered = try self.allocator.alloc(i64, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .timestamp_us = reordered };
            },
            .timestamp_ns => |data| {
                const reordered = try self.allocator.alloc(i64, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .timestamp_ns = reordered };
            },
            .date64 => |data| {
                const reordered = try self.allocator.alloc(i64, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .date64 = reordered };
            },
            .int32 => |data| {
                const reordered = try self.allocator.alloc(i32, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .int32 = reordered };
            },
            .date32 => |data| {
                const reordered = try self.allocator.alloc(i32, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .date32 = reordered };
            },
            .float64 => |data| {
                const reordered = try self.allocator.alloc(f64, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .float64 = reordered };
            },
            .float32 => |data| {
                const reordered = try self.allocator.alloc(f32, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .float32 = reordered };
            },
            .bool_ => |data| {
                const reordered = try self.allocator.alloc(bool, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .bool_ = reordered };
            },
            .string => |data| {
                const reordered = try self.allocator.alloc([]const u8, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = try self.allocator.dupe(u8, data[idx]);
                }
                // Free old strings and array
                for (data) |str| {
                    self.allocator.free(str);
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .string = reordered };
            },
        }
    }

    // ========================================================================
    // LIMIT/OFFSET Implementation
    // ========================================================================

    fn applyLimitOffset(
        self: *Self,
        columns: []Result.Column,
        limit: ?u32,
        offset: ?u32,
    ) usize {
        if (columns.len == 0) return 0;

        const row_count = switch (columns[0].data) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| data.len,
            .int32, .date32 => |data| data.len,
            .float64 => |data| data.len,
            .float32 => |data| data.len,
            .bool_ => |data| data.len,
            .string => |data| data.len,
        };

        const start = offset orelse 0;
        if (start >= row_count) {
            // Free all data and return 0
            for (columns) |*col| {
                self.freeColumnData(&col.data);
                col.data = switch (col.data) {
                    .int64 => Result.ColumnData{ .int64 = &[_]i64{} },
                    .timestamp_s => Result.ColumnData{ .timestamp_s = &[_]i64{} },
                    .timestamp_ms => Result.ColumnData{ .timestamp_ms = &[_]i64{} },
                    .timestamp_us => Result.ColumnData{ .timestamp_us = &[_]i64{} },
                    .timestamp_ns => Result.ColumnData{ .timestamp_ns = &[_]i64{} },
                    .date64 => Result.ColumnData{ .date64 = &[_]i64{} },
                    .int32 => Result.ColumnData{ .int32 = &[_]i32{} },
                    .date32 => Result.ColumnData{ .date32 = &[_]i32{} },
                    .float64 => Result.ColumnData{ .float64 = &[_]f64{} },
                    .float32 => Result.ColumnData{ .float32 = &[_]f32{} },
                    .bool_ => Result.ColumnData{ .bool_ = &[_]bool{} },
                    .string => Result.ColumnData{ .string = &[_][]const u8{} },
                };
            }
            return 0;
        }

        const end = if (limit) |l|
            @min(start + l, row_count)
        else
            row_count;

        // Slice each column
        for (columns) |*col| {
            self.sliceColumn(col, start, end) catch {};
        }

        return end - start;
    }

    fn sliceColumn(self: *Self, col: *Result.Column, start: usize, end: usize) !void {
        const new_len = end - start;

        switch (col.data) {
            .int64 => |data| {
                const sliced = try self.allocator.alloc(i64, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .int64 = sliced };
            },
            .timestamp_s => |data| {
                const sliced = try self.allocator.alloc(i64, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .timestamp_s = sliced };
            },
            .timestamp_ms => |data| {
                const sliced = try self.allocator.alloc(i64, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .timestamp_ms = sliced };
            },
            .timestamp_us => |data| {
                const sliced = try self.allocator.alloc(i64, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .timestamp_us = sliced };
            },
            .timestamp_ns => |data| {
                const sliced = try self.allocator.alloc(i64, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .timestamp_ns = sliced };
            },
            .date64 => |data| {
                const sliced = try self.allocator.alloc(i64, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .date64 = sliced };
            },
            .int32 => |data| {
                const sliced = try self.allocator.alloc(i32, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .int32 = sliced };
            },
            .date32 => |data| {
                const sliced = try self.allocator.alloc(i32, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .date32 = sliced };
            },
            .float64 => |data| {
                const sliced = try self.allocator.alloc(f64, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .float64 = sliced };
            },
            .float32 => |data| {
                const sliced = try self.allocator.alloc(f32, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .float32 = sliced };
            },
            .bool_ => |data| {
                const sliced = try self.allocator.alloc(bool, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .bool_ = sliced };
            },
            .string => |data| {
                const sliced = try self.allocator.alloc([]const u8, new_len);
                for (data[start..end], 0..) |str, i| {
                    sliced[i] = try self.allocator.dupe(u8, str);
                }
                // Free old strings
                for (data) |str| {
                    self.allocator.free(str);
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .string = sliced };
            },
        }
    }

    fn freeColumnData(self: *Self, data: *Result.ColumnData) void {
        switch (data.*) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |d| self.allocator.free(d),
            .int32, .date32 => |d| self.allocator.free(d),
            .float64 => |d| self.allocator.free(d),
            .float32 => |d| self.allocator.free(d),
            .bool_ => |d| self.allocator.free(d),
            .string => |d| {
                for (d) |str| {
                    self.allocator.free(str);
                }
                self.allocator.free(d);
            },
        }
    }

    // ========================================================================
    // HAVING Clause Implementation
    // ========================================================================

    /// Apply HAVING filter to result, returning filtered result
    fn applyHaving(
        self: *Self,
        result: *Result,
        having_expr: *const Expr,
        select_items: []const ast.SelectItem,
    ) !void {
        if (result.row_count == 0) return;

        // Collect indices of rows that pass the HAVING filter
        var passing_indices = std.ArrayList(usize){};
        defer passing_indices.deinit(self.allocator);

        for (0..result.row_count) |row_idx| {
            const passes = try self.evaluateHavingExpr(result.columns, select_items, having_expr, row_idx);
            if (passes) {
                try passing_indices.append(self.allocator, row_idx);
            }
        }

        // If all rows pass, nothing to do
        if (passing_indices.items.len == result.row_count) return;

        // Build filtered result columns
        const indices = passing_indices.items;
        var new_columns = try self.allocator.alloc(Result.Column, result.columns.len);
        errdefer self.allocator.free(new_columns);

        for (result.columns, 0..) |col, i| {
            new_columns[i] = try self.filterColumnByIndices(col, indices);
        }

        // Free old column data
        for (result.columns) |col| {
            var data = col.data;
            self.freeColumnData(&data);
        }
        self.allocator.free(result.columns);

        result.columns = new_columns;
        result.row_count = indices.len;
    }

    /// Evaluate HAVING expression for a single result row
    fn evaluateHavingExpr(
        self: *Self,
        columns: []const Result.Column,
        select_items: []const ast.SelectItem,
        expr: *const Expr,
        row_idx: usize,
    ) anyerror!bool {
        return switch (expr.*) {
            .value => |val| switch (val) {
                .integer => |i| i != 0,
                .float => |f| f != 0.0,
                .null => false,
                else => true,
            },
            .binary => |bin| try self.evaluateHavingBinaryOp(columns, select_items, bin.op, bin.left, bin.right, row_idx),
            .unary => |un| switch (un.op) {
                .not => !(try self.evaluateHavingExpr(columns, select_items, un.operand, row_idx)),
                else => error.UnsupportedOperator,
            },
            else => error.UnsupportedExpression,
        };
    }

    /// Evaluate binary operation in HAVING context
    fn evaluateHavingBinaryOp(
        self: *Self,
        columns: []const Result.Column,
        select_items: []const ast.SelectItem,
        op: BinaryOp,
        left: *const Expr,
        right: *const Expr,
        row_idx: usize,
    ) anyerror!bool {
        return switch (op) {
            .@"and" => (try self.evaluateHavingExpr(columns, select_items, left, row_idx)) and
                (try self.evaluateHavingExpr(columns, select_items, right, row_idx)),
            .@"or" => (try self.evaluateHavingExpr(columns, select_items, left, row_idx)) or
                (try self.evaluateHavingExpr(columns, select_items, right, row_idx)),
            .eq, .ne, .lt, .le, .gt, .ge => try self.evaluateHavingComparison(columns, select_items, op, left, right, row_idx),
            else => error.UnsupportedOperator,
        };
    }

    /// Evaluate comparison in HAVING context
    fn evaluateHavingComparison(
        self: *Self,
        columns: []const Result.Column,
        select_items: []const ast.SelectItem,
        op: BinaryOp,
        left: *const Expr,
        right: *const Expr,
        row_idx: usize,
    ) !bool {
        const left_val = try self.getHavingValue(columns, select_items, left, row_idx);
        const right_val = try self.getHavingValue(columns, select_items, right, row_idx);

        // Compare as floats for numeric comparison
        return switch (left_val) {
            .integer => |left_int| blk: {
                const right_num = switch (right_val) {
                    .integer => |i| @as(f64, @floatFromInt(i)),
                    .float => |f| f,
                    else => return error.TypeMismatch,
                };
                const left_num = @as(f64, @floatFromInt(left_int));
                break :blk self.compareNumbers(op, left_num, right_num);
            },
            .float => |left_float| blk: {
                const right_num = switch (right_val) {
                    .integer => |i| @as(f64, @floatFromInt(i)),
                    .float => |f| f,
                    else => return error.TypeMismatch,
                };
                break :blk self.compareNumbers(op, left_float, right_num);
            },
            .string => |left_str| blk: {
                const right_str = switch (right_val) {
                    .string => |s| s,
                    else => return error.TypeMismatch,
                };
                break :blk self.compareStrings(op, left_str, right_str);
            },
            .null => op == .ne,
            else => error.UnsupportedType,
        };
    }

    /// Get value of expression in HAVING context (from result columns)
    fn getHavingValue(
        self: *Self,
        columns: []const Result.Column,
        select_items: []const ast.SelectItem,
        expr: *const Expr,
        row_idx: usize,
    ) !Value {
        return switch (expr.*) {
            .value => expr.value,
            .column => |col| blk: {
                // Look up column by name in result columns
                const col_idx = self.findColumnIndex(columns, col.name) orelse
                    return error.ColumnNotFound;
                break :blk self.getResultColumnValue(columns[col_idx], row_idx);
            },
            .call => |call| blk: {
                // For aggregate functions, find matching SELECT item
                if (isAggregateFunction(call.name)) {
                    const col_idx = try self.findAggregateColumnIndex(columns, select_items, call.name, call.args);
                    break :blk self.getResultColumnValue(columns[col_idx], row_idx);
                }
                return error.UnsupportedExpression;
            },
            .binary => |bin| try self.evaluateHavingBinaryToValue(columns, select_items, bin, row_idx),
            else => error.UnsupportedExpression,
        };
    }

    /// Evaluate binary expression to value in HAVING context
    fn evaluateHavingBinaryToValue(
        self: *Self,
        columns: []const Result.Column,
        select_items: []const ast.SelectItem,
        bin: anytype,
        row_idx: usize,
    ) anyerror!Value {
        const left = try self.getHavingValue(columns, select_items, bin.left, row_idx);
        const right = try self.getHavingValue(columns, select_items, bin.right, row_idx);

        return switch (bin.op) {
            .add => self.addValues(left, right),
            .subtract => self.subtractValues(left, right),
            .multiply => self.multiplyValues(left, right),
            .divide => self.divideValues(left, right),
            else => error.UnsupportedOperator,
        };
    }

    /// Get value from a result column at a given row index
    fn getResultColumnValue(self: *Self, col: Result.Column, row_idx: usize) Value {
        _ = self;
        return switch (col.data) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| Value{ .integer = data[row_idx] },
            .int32, .date32 => |data| Value{ .integer = data[row_idx] },
            .float64 => |data| Value{ .float = data[row_idx] },
            .float32 => |data| Value{ .float = data[row_idx] },
            .bool_ => |data| Value{ .integer = if (data[row_idx]) 1 else 0 },
            .string => |data| Value{ .string = data[row_idx] },
        };
    }

    /// Find the result column index that matches an aggregate function call
    fn findAggregateColumnIndex(
        self: *Self,
        columns: []const Result.Column,
        select_items: []const ast.SelectItem,
        call_name: []const u8,
        call_args: []const Expr,
    ) !usize {
        _ = self;
        // First, try to find by alias matching the function name
        for (columns, 0..) |col, i| {
            if (std.ascii.eqlIgnoreCase(col.name, call_name)) {
                return i;
            }
        }

        // Match by comparing SELECT item expressions
        for (select_items, 0..) |item, i| {
            if (item.expr == .call) {
                const item_call = item.expr.call;
                // Match function name (case insensitive)
                if (std.ascii.eqlIgnoreCase(item_call.name, call_name)) {
                    // Match arguments
                    if (aggregateArgsMatch(item_call.args, call_args)) {
                        return i;
                    }
                }
            }
        }

        return error.ColumnNotFound;
    }
};

/// Check if two aggregate argument lists match
fn aggregateArgsMatch(a: []const Expr, b: []const Expr) bool {
    if (a.len != b.len) return false;

    for (a, b) |arg_a, arg_b| {
        if (!exprEquals(&arg_a, &arg_b)) return false;
    }

    return true;
}

/// Check if two expressions are equal (for aggregate matching)
fn exprEquals(a: *const Expr, b: *const Expr) bool {
    if (std.meta.activeTag(a.*) != std.meta.activeTag(b.*)) return false;

    return switch (a.*) {
        .column => |col_a| blk: {
            const col_b = b.column;
            break :blk std.mem.eql(u8, col_a.name, col_b.name);
        },
        .value => |val_a| blk: {
            const val_b = b.value;
            if (std.meta.activeTag(val_a) != std.meta.activeTag(val_b)) break :blk false;
            break :blk switch (val_a) {
                .integer => |i| i == val_b.integer,
                .float => |f| f == val_b.float,
                .string => |s| std.mem.eql(u8, s, val_b.string),
                .null => true,
                else => false,
            };
        },
        else => false,
    };
}

// Tests are in tests/test_sql_executor.zig
