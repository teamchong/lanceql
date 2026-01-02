//! Window Functions - Types and utilities for SQL window operations
//!
//! Contains WindowFunctionType enum and window function detection.

const std = @import("std");
const ast = @import("ast");
const Expr = ast.Expr;

/// Window function types
pub const WindowFunctionType = enum {
    row_number,
    rank,
    dense_rank,
    lag,
    lead,
};

/// Check if expression is a window function (has OVER clause)
pub fn isWindowFunction(expr: *const Expr) bool {
    return switch (expr.*) {
        .call => |call| call.window != null,
        else => false,
    };
}

/// Check if SELECT list contains any window functions
pub fn hasWindowFunctions(select_list: []const ast.SelectItem) bool {
    for (select_list) |item| {
        if (isWindowFunction(&item.expr)) {
            return true;
        }
    }
    return false;
}

/// Parse window function name to WindowFunctionType
pub fn parseWindowFunctionType(name: []const u8) ?WindowFunctionType {
    if (name.len < 3 or name.len > 16) return null;

    var upper_buf: [16]u8 = undefined;
    const len = @min(name.len, upper_buf.len);
    const upper_name = std.ascii.upperString(upper_buf[0..len], name[0..len]);

    if (std.mem.eql(u8, upper_name, "ROW_NUMBER")) return .row_number;
    if (std.mem.eql(u8, upper_name, "RANK")) return .rank;
    if (std.mem.eql(u8, upper_name, "DENSE_RANK")) return .dense_rank;
    if (std.mem.eql(u8, upper_name, "LAG")) return .lag;
    if (std.mem.eql(u8, upper_name, "LEAD")) return .lead;

    return null;
}

/// Check if function name is a window function
pub fn isWindowFunctionName(name: []const u8) bool {
    return parseWindowFunctionType(name) != null;
}
