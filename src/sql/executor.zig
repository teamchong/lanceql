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
        float64: []f64,
        string: [][]const u8,
    };

    pub fn deinit(self: *Result) void {
        for (self.columns) |col| {
            switch (col.data) {
                .int64 => |data| self.allocator.free(data),
                .float64 => |data| self.allocator.free(data),
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
    float64: []f64,
    string: [][]const u8,
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
                .int64 => |data| self.allocator.free(data),
                .float64 => |data| self.allocator.free(data),
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
            if (std.mem.indexOf(u8, field.logical_type, "int") != null) {
                const data = try self.table.readInt64Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .int64 = data });
            } else if (std.mem.indexOf(u8, field.logical_type, "float") != null or
                       std.mem.indexOf(u8, field.logical_type, "double") != null) {
                const data = try self.table.readFloat64Column(physical_col_id);
                try self.column_cache.put(name, CachedColumn{ .float64 = data });
            } else if (std.mem.indexOf(u8, field.logical_type, "utf8") != null or
                       std.mem.indexOf(u8, field.logical_type, "string") != null) {
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
        const columns = try self.readColumns(stmt.columns, indices);

        // 5. Apply ORDER BY (in-memory sorting)
        if (stmt.order_by) |order_by| {
            try self.applyOrderBy(columns, order_by);
        }

        // 6. Apply LIMIT and OFFSET
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
        var groups = std.StringHashMap(std.ArrayList(u32)).init(self.allocator);
        defer {
            var iter = groups.valueIterator();
            while (iter.next()) |list| {
                list.deinit(self.allocator);
            }
            groups.deinit();
        }

        // Also need to track key strings for proper cleanup
        var key_strings = std.ArrayList([]const u8){};
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
                var list = std.ArrayList(u32){};
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
        var result_columns = std.ArrayList(Result.Column){};
        errdefer {
            for (result_columns.items) |col| {
                switch (col.data) {
                    .int64 => |data| self.allocator.free(data),
                    .float64 => |data| self.allocator.free(data),
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
            if (gb.having) |_| {
                // TODO: Apply HAVING filter
                // For now, skip HAVING
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
                .int64 => |data| {
                    var buf: [32]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch unreachable;
                    try key.appendSlice(self.allocator, str);
                },
                .float64 => |data| {
                    var buf: [32]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch unreachable;
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
                        .int64 => |data| acc.addInt(data[row_idx]),
                        .float64 => |data| acc.addFloat(data[row_idx]),
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
                left_ptr.* = try self.bindParameters(bin.left, params);

                const right_ptr = try self.allocator.create(Expr);
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
                operand_ptr.* = try self.bindParameters(un.operand, params);

                break :blk Expr{
                    .unary = .{
                        .op = un.op,
                        .operand = operand_ptr,
                    },
                };
            },
            else => error.UnsupportedExpression,
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

    /// Evaluate expression to a concrete value
    fn evaluateToValue(self: *Self, expr: *const Expr, row_idx: u32) !Value {
        return switch (expr.*) {
            .value => expr.value,
            .column => |col| blk: {
                // Lookup in cache instead of reading from table
                const cached = self.column_cache.get(col.name) orelse return error.ColumnNotCached;

                break :blk switch (cached) {
                    .int64 => |data| Value{ .integer = data[row_idx] },
                    .float64 => |data| Value{ .float = data[row_idx] },
                    .string => |data| Value{ .string = data[row_idx] },
                };
            },
            else => error.UnsupportedExpression,
        };
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
                    .int64 => |data| self.allocator.free(data),
                    .float64 => |data| self.allocator.free(data),
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
                // TODO: Handle expressions, function calls, etc.
                return error.UnsupportedExpression;
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

        // For Phase 1: Read ALL rows, then filter by indices
        // TODO Phase 2: Add readAtIndices() methods to Table for efficiency

        // Determine type from logical_type
        if (std.mem.indexOf(u8, field.logical_type, "int") != null) {
            const all_data = try self.table.readInt64Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(i64, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .int64 = filtered };
        } else if (std.mem.indexOf(u8, field.logical_type, "float") != null or
                   std.mem.indexOf(u8, field.logical_type, "double") != null) {
            const all_data = try self.table.readFloat64Column(col_idx);
            defer self.allocator.free(all_data);

            const filtered = try self.allocator.alloc(f64, indices.len);
            for (indices, 0..) |idx, i| {
                filtered[i] = all_data[idx];
            }
            return Result.ColumnData{ .float64 = filtered };
        } else if (std.mem.indexOf(u8, field.logical_type, "utf8") != null or
                   std.mem.indexOf(u8, field.logical_type, "string") != null) {
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
    // ORDER BY Implementation
    // ========================================================================

    fn applyOrderBy(
        self: *Self,
        columns: []Result.Column,
        order_by: []const ast.OrderBy,
    ) !void {
        if (columns.len == 0) return;

        const row_count = switch (columns[0].data) {
            .int64 => |data| data.len,
            .float64 => |data| data.len,
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
            .int64 => |data| blk: {
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
            .float64 => |data| {
                const reordered = try self.allocator.alloc(f64, data.len);
                for (indices, 0..) |idx, i| {
                    reordered[i] = data[idx];
                }
                self.allocator.free(data);
                col.data = Result.ColumnData{ .float64 = reordered };
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
            .int64 => |data| data.len,
            .float64 => |data| data.len,
            .string => |data| data.len,
        };

        const start = offset orelse 0;
        if (start >= row_count) {
            // Free all data and return 0
            for (columns) |*col| {
                self.freeColumnData(&col.data);
                col.data = switch (col.data) {
                    .int64 => Result.ColumnData{ .int64 = &[_]i64{} },
                    .float64 => Result.ColumnData{ .float64 = &[_]f64{} },
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
            .float64 => |data| {
                const sliced = try self.allocator.alloc(f64, new_len);
                @memcpy(sliced, data[start..end]);
                self.allocator.free(data);
                col.data = Result.ColumnData{ .float64 = sliced };
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
            .int64 => |d| self.allocator.free(d),
            .float64 => |d| self.allocator.free(d),
            .string => |d| {
                for (d) |str| {
                    self.allocator.free(str);
                }
                self.allocator.free(d);
            },
        }
    }
};

// Tests are in tests/test_sql_executor.zig
