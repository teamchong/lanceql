//! Abstract Syntax Tree definitions for SQL
//!
//! Represents parsed SQL statements as tree structures.
//! Supports SELECT queries with WHERE, ORDER BY, LIMIT, and GROUP BY.

const std = @import("std");

/// SQL Value types
pub const ValueType = enum {
    null,
    integer,
    float,
    string,
    blob,
    parameter, // ? placeholder
};

/// A literal or parameter value
pub const Value = union(ValueType) {
    null: void,
    integer: i64,
    float: f64,
    string: []const u8,
    blob: []const u8,
    parameter: u32, // Parameter index (0-based)
};

/// Binary operators
pub const BinaryOp = enum {
    // Arithmetic
    add,      // +
    subtract, // -
    multiply, // *
    divide,   // /
    concat,   // || (string concatenation)

    // Comparison
    eq,  // =
    ne,  // != or <>
    lt,  // <
    le,  // <=
    gt,  // >
    ge,  // >=

    // Logical
    @"and", // AND
    @"or",  // OR

    // Other
    like,    // LIKE
    in,      // IN
    between, // BETWEEN
};

/// Unary operators
pub const UnaryOp = enum {
    not,    // NOT
    minus,  // -
    is_null, // IS NULL
    is_not_null, // IS NOT NULL
};

/// Expression node (recursive)
pub const Expr = union(enum) {
    /// Literal value
    value: Value,

    /// Column reference (e.g., "users.id" or just "id")
    column: struct {
        table: ?[]const u8, // Optional table qualifier
        name: []const u8,
    },

    /// Binary operation
    binary: struct {
        op: BinaryOp,
        left: *Expr,
        right: *Expr,
    },

    /// Unary operation
    unary: struct {
        op: UnaryOp,
        operand: *Expr,
    },

    /// Function call (e.g., COUNT(*), AVG(salary))
    call: struct {
        name: []const u8,
        args: []Expr,
        distinct: bool,
    },

    /// IN expression: col IN (val1, val2, ...)
    in_list: struct {
        expr: *Expr,
        values: []Expr,
    },

    /// BETWEEN expression: col BETWEEN low AND high
    between: struct {
        expr: *Expr,
        low: *Expr,
        high: *Expr,
    },
};

/// SELECT column specification
pub const SelectItem = struct {
    /// The expression (can be *, column name, or function)
    expr: Expr,

    /// Optional alias (AS name)
    alias: ?[]const u8,
};

/// FROM clause table reference
pub const TableRef = struct {
    /// Table name
    name: []const u8,

    /// Optional alias
    alias: ?[]const u8,
};

/// ORDER BY direction
pub const OrderDirection = enum {
    asc,
    desc,
};

/// ORDER BY clause item
pub const OrderBy = struct {
    /// Column to sort by
    column: []const u8,

    /// Sort direction
    direction: OrderDirection,
};

/// GROUP BY clause
pub const GroupBy = struct {
    /// Columns to group by
    columns: [][]const u8,

    /// Optional HAVING clause
    having: ?Expr,
};

/// Complete SELECT statement
pub const SelectStmt = struct {
    /// SELECT clause
    distinct: bool,
    columns: []SelectItem,

    /// FROM clause
    from: TableRef,

    /// WHERE clause (optional)
    where: ?Expr,

    /// GROUP BY clause (optional)
    group_by: ?GroupBy,

    /// ORDER BY clause (optional)
    order_by: ?[]OrderBy,

    /// LIMIT clause (optional)
    limit: ?u32,

    /// OFFSET clause (optional)
    offset: ?u32,
};

/// Top-level statement (extensible for INSERT/UPDATE/DELETE later)
pub const Statement = union(enum) {
    select: SelectStmt,
    // Future: insert, update, delete, create_table, etc.
};

// ============================================================================
// Memory Management
// ============================================================================

/// Recursively free all heap-allocated Expr pointers in an expression tree.
/// Call this to clean up expressions allocated by the parser.
pub fn deinitExpr(expr: *Expr, allocator: std.mem.Allocator) void {
    switch (expr.*) {
        .binary => |bin| {
            deinitExpr(bin.left, allocator);
            allocator.destroy(bin.left);
            deinitExpr(bin.right, allocator);
            allocator.destroy(bin.right);
        },
        .unary => |un| {
            deinitExpr(un.operand, allocator);
            allocator.destroy(un.operand);
        },
        .call => |call| {
            for (call.args) |*arg| {
                deinitExpr(arg, allocator);
            }
            allocator.free(call.args);
        },
        .in_list => |in| {
            deinitExpr(in.expr, allocator);
            allocator.destroy(in.expr);
            for (in.values) |*val| {
                deinitExpr(val, allocator);
            }
            allocator.free(in.values);
        },
        .between => |between| {
            deinitExpr(between.expr, allocator);
            allocator.destroy(between.expr);
            deinitExpr(between.low, allocator);
            allocator.destroy(between.low);
            deinitExpr(between.high, allocator);
            allocator.destroy(between.high);
        },
        .value, .column => {}, // No heap allocations to free
    }
}

/// Free all heap-allocated memory in a SelectStmt.
/// Call this to clean up statements returned by the parser.
pub fn deinitSelectStmt(stmt: *SelectStmt, allocator: std.mem.Allocator) void {
    // Free column expressions
    for (stmt.columns) |*col| {
        deinitExpr(&col.expr, allocator);
    }
    allocator.free(stmt.columns);

    // Free WHERE clause
    if (stmt.where) |*where| {
        deinitExpr(where, allocator);
    }

    // Free GROUP BY clause
    if (stmt.group_by) |*group_by| {
        allocator.free(group_by.columns);
        if (group_by.having) |*having| {
            deinitExpr(having, allocator);
        }
    }

    // Free ORDER BY clause
    if (stmt.order_by) |order_by| {
        allocator.free(order_by);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if expression is an aggregate function
pub fn isAggregate(expr: *const Expr) bool {
    return switch (expr.*) {
        .call => |call| {
            const name_upper = std.ascii.upperString(call.name);
            return std.mem.eql(u8, name_upper, "COUNT") or
                std.mem.eql(u8, name_upper, "SUM") or
                std.mem.eql(u8, name_upper, "AVG") or
                std.mem.eql(u8, name_upper, "MIN") or
                std.mem.eql(u8, name_upper, "MAX");
        },
        else => false,
    };
}

/// Count parameters in expression tree
pub fn countParameters(expr: *const Expr) u32 {
    return switch (expr.*) {
        .value => |val| if (val == .parameter) 1 else 0,
        .column => 0,
        .binary => |bin| countParameters(bin.left) + countParameters(bin.right),
        .unary => |un| countParameters(un.operand),
        .call => |call| {
            var count: u32 = 0;
            for (call.args) |*arg| {
                count += countParameters(arg);
            }
            return count;
        },
        .in_list => |in| {
            var count = countParameters(in.expr);
            for (in.values) |*val| {
                count += countParameters(val);
            }
            return count;
        },
        .between => |between| {
            return countParameters(between.expr) +
                countParameters(between.low) +
                countParameters(between.high);
        },
    };
}

/// Print expression tree (for debugging)
pub fn printExpr(expr: *const Expr, writer: anytype, indent: usize) !void {
    const spaces = " " ** 80;
    const prefix = spaces[0..@min(indent, spaces.len)];

    switch (expr.*) {
        .value => |val| {
            try writer.print("{s}Value: ", .{prefix});
            switch (val) {
                .null => try writer.writeAll("NULL\n"),
                .integer => |i| try writer.print("{d}\n", .{i}),
                .float => |f| try writer.print("{d}\n", .{f}),
                .string => |s| try writer.print("'{s}'\n", .{s}),
                .blob => |b| try writer.print("BLOB({d} bytes)\n", .{b.len}),
                .parameter => |p| try writer.print("?{d}\n", .{p + 1}),
            }
        },
        .column => |col| {
            if (col.table) |table| {
                try writer.print("{s}Column: {s}.{s}\n", .{ prefix, table, col.name });
            } else {
                try writer.print("{s}Column: {s}\n", .{ prefix, col.name });
            }
        },
        .binary => |bin| {
            try writer.print("{s}Binary: {s}\n", .{ prefix, @tagName(bin.op) });
            try printExpr(bin.left, writer, indent + 2);
            try printExpr(bin.right, writer, indent + 2);
        },
        .unary => |un| {
            try writer.print("{s}Unary: {s}\n", .{ prefix, @tagName(un.op) });
            try printExpr(un.operand, writer, indent + 2);
        },
        .call => |call| {
            try writer.print("{s}Call: {s}(", .{ prefix, call.name });
            if (call.distinct) try writer.writeAll("DISTINCT ");
            try writer.writeAll(")\n");
            for (call.args) |*arg| {
                try printExpr(arg, writer, indent + 2);
            }
        },
        .in_list => |in| {
            try writer.print("{s}IN:\n", .{prefix});
            try printExpr(in.expr, writer, indent + 2);
            try writer.print("{s}  Values:\n", .{prefix});
            for (in.values) |*val| {
                try printExpr(val, writer, indent + 4);
            }
        },
        .between => |between| {
            try writer.print("{s}BETWEEN:\n", .{prefix});
            try printExpr(between.expr, writer, indent + 2);
            try writer.print("{s}  Low:\n", .{prefix});
            try printExpr(between.low, writer, indent + 4);
            try writer.print("{s}  High:\n", .{prefix});
            try printExpr(between.high, writer, indent + 4);
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

test "value types" {
    const val_int = Value{ .integer = 42 };
    const val_str = Value{ .string = "hello" };
    const val_param = Value{ .parameter = 0 };

    try std.testing.expect(val_int == .integer);
    try std.testing.expectEqual(@as(i64, 42), val_int.integer);
    try std.testing.expect(val_str == .string);
    try std.testing.expectEqualStrings("hello", val_str.string);
    try std.testing.expect(val_param == .parameter);
    try std.testing.expectEqual(@as(u32, 0), val_param.parameter);
}

test "column expression" {
    const col = Expr{
        .column = .{
            .table = "users",
            .name = "id",
        },
    };

    try std.testing.expect(col == .column);
    try std.testing.expectEqualStrings("users", col.column.table.?);
    try std.testing.expectEqualStrings("id", col.column.name);
}
