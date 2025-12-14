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

        // 3. Read columns based on SELECT list
        const columns = try self.readColumns(stmt.columns, indices);

        // 4. Apply ORDER BY (in-memory sorting)
        if (stmt.order_by) |order_by| {
            try self.applyOrderBy(columns, order_by);
        }

        // 5. Apply LIMIT and OFFSET
        const final_row_count = self.applyLimitOffset(columns, stmt.limit, stmt.offset);

        return Result{
            .columns = columns,
            .row_count = final_row_count,
            .allocator = self.allocator,
        };
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

// ============================================================================
// Tests
// ============================================================================

const parser = @import("parser");

test "execute simple SELECT *" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("tests/fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(lance_data, allocator);
    defer table.deinit();

    // Parse SQL
    const sql = "SELECT * FROM table";
    const stmt = try parser.parseSQL(sql, allocator);
    defer {
        // Free allocated memory
        allocator.free(stmt.select.columns);
    }

    // Execute
    var executor = Executor.init(&table, allocator);
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    // Check column data
    const col = result.columns[0];
    try std.testing.expect(col.data == .int64);
    const values = col.data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 2), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
    try std.testing.expectEqual(@as(i64, 4), values[3]);
    try std.testing.expectEqual(@as(i64, 5), values[4]);
}

test "execute SELECT with WHERE clause" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("tests/fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(lance_data, allocator);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table WHERE id > 2
    const sql = "SELECT * FROM table WHERE id > 2";
    const stmt = try parser.parseSQL(sql, allocator);
    defer {
        allocator.free(stmt.select.columns);
        // Note: WHERE expression cleanup not implemented yet
    }

    // Execute
    var executor = Executor.init(&table, allocator);
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 3 rows (3, 4, 5)
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
    try std.testing.expectEqual(@as(i64, 5), values[2]);
}

test "execute SELECT with ORDER BY DESC" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("tests/fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(lance_data, allocator);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table ORDER BY id DESC
    const sql = "SELECT * FROM table ORDER BY id DESC";
    const stmt = try parser.parseSQL(sql, allocator);
    defer {
        allocator.free(stmt.select.columns);
        if (stmt.select.order_by) |order_by| {
            allocator.free(order_by);
        }
    }

    // Execute
    var executor = Executor.init(&table, allocator);
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should be reversed (5, 4, 3, 2, 1)
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 5), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
    try std.testing.expectEqual(@as(i64, 2), values[3]);
    try std.testing.expectEqual(@as(i64, 1), values[4]);
}

test "execute SELECT with LIMIT" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("tests/fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(lance_data, allocator);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table LIMIT 3
    const sql = "SELECT * FROM table LIMIT 3";
    const stmt = try parser.parseSQL(sql, allocator);
    defer {
        allocator.free(stmt.select.columns);
    }

    // Execute
    var executor = Executor.init(&table, allocator);
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 3 rows (1, 2, 3)
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 2), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
}

test "execute SELECT with OFFSET" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("tests/fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(lance_data, allocator);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table LIMIT 2 OFFSET 2
    const sql = "SELECT * FROM table LIMIT 2 OFFSET 2";
    const stmt = try parser.parseSQL(sql, allocator);
    defer {
        allocator.free(stmt.select.columns);
    }

    // Execute
    var executor = Executor.init(&table, allocator);
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 2 rows (3, 4)
    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
}

test "execute SELECT with float64 column" {
    const allocator = std.testing.allocator;

    // Open test Lance file with float64
    const lance_data = @embedFile("tests/fixtures/simple_float64.lance/data/1001011111000011101110011001df8d2b7e5cdba7b949fb6c85.lance");
    var table = try Table.init(lance_data, allocator);
    defer table.deinit();

    // Parse SQL
    const sql = "SELECT * FROM table WHERE value > 3.0";
    const stmt = try parser.parseSQL(sql, allocator);
    defer {
        allocator.free(stmt.select.columns);
    }

    // Execute
    var executor = Executor.init(&table, allocator);
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 3 rows (3.5, 4.5, 5.5)
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.float64;
    try std.testing.expectEqual(@as(f64, 3.5), values[0]);
    try std.testing.expectEqual(@as(f64, 4.5), values[1]);
    try std.testing.expectEqual(@as(f64, 5.5), values[2]);
}

test "execute SELECT with mixed types" {
    const allocator = std.testing.allocator;

    // Open test Lance file with mixed types
    const lance_data = @embedFile("tests/fixtures/mixed_types.lance/data/11100100001000010010010060d60b4085bd08dcf790581192.lance");
    var table = try Table.init(lance_data, allocator);
    defer table.deinit();

    // Parse SQL
    const sql = "SELECT * FROM table";
    const stmt = try parser.parseSQL(sql, allocator);
    defer {
        allocator.free(stmt.select.columns);
    }

    // Execute
    var executor = Executor.init(&table, allocator);
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results
    try std.testing.expectEqual(@as(usize, 3), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    // Check column types
    try std.testing.expect(result.columns[0].data == .int64);
    try std.testing.expect(result.columns[1].data == .float64);
    try std.testing.expect(result.columns[2].data == .string);
}
