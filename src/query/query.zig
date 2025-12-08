//! Query module for LanceQL.
//!
//! Provides SQL parsing, expression evaluation, and query execution
//! for Lance files.

const std = @import("std");

pub const value = @import("lanceql.value");
pub const Value = value.Value;

pub const expr = @import("expr.zig");
pub const lexer = @import("lexer.zig");
pub const parser = @import("parser.zig");
pub const ast = @import("ast.zig");
pub const executor = @import("executor.zig");
pub const aggregates = @import("aggregates.zig");

// Re-export main types
pub const Expr = expr.Expr;
pub const BinaryOp = expr.BinaryOp;
pub const UnaryOp = expr.UnaryOp;
pub const Lexer = lexer.Lexer;
pub const Token = lexer.Token;
pub const Parser = parser.Parser;
pub const SelectStmt = ast.SelectStmt;
pub const Executor = executor.Executor;
pub const ResultSet = executor.ResultSet;

test {
    std.testing.refAllDecls(@This());
}
