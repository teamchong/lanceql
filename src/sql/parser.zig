//! SQL Parser - Recursive Descent Parser
//!
//! Converts tokens from the lexer into an Abstract Syntax Tree (AST).
//! Supports SQLite-style SQL with ? parameter placeholders.

const std = @import("std");
const lexer = @import("lexer");
const ast = @import("ast");

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const Expr = ast.Expr;
const SelectStmt = ast.SelectStmt;
const Statement = ast.Statement;

/// SQL Parser
pub const Parser = struct {
    tokens: []const Token,
    position: usize,
    allocator: std.mem.Allocator,
    param_count: u32, // Track parameter count for ?

    const Self = @This();

    pub fn init(tokens: []const Token, allocator: std.mem.Allocator) Self {
        return Self{
            .tokens = tokens,
            .position = 0,
            .allocator = allocator,
            .param_count = 0,
        };
    }

    /// Get current token
    fn current(self: *const Self) ?Token {
        if (self.position >= self.tokens.len) return null;
        return self.tokens[self.position];
    }

    /// Advance to next token
    fn advance(self: *Self) void {
        if (self.position < self.tokens.len) {
            self.position += 1;
        }
    }

    /// Check if current token matches type
    fn check(self: *const Self, token_type: TokenType) bool {
        const tok = self.current() orelse return false;
        return tok.type == token_type;
    }

    /// Consume token if it matches, otherwise error
    fn expect(self: *Self, token_type: TokenType) !Token {
        const tok = self.current() orelse return error.UnexpectedEOF;
        if (tok.type != token_type) {
            return error.UnexpectedToken;
        }
        self.advance();
        return tok;
    }

    /// Check if current token matches any of the given types
    fn match(self: *Self, types: []const TokenType) bool {
        for (types) |t| {
            if (self.check(t)) {
                self.advance();
                return true;
            }
        }
        return false;
    }

    // ========================================================================
    // Top-level parsing
    // ========================================================================

    /// Parse a complete SQL statement
    pub fn parseStatement(self: *Self) !Statement {
        const tok = self.current() orelse return error.EmptyStatement;

        return switch (tok.type) {
            .SELECT => Statement{ .select = try self.parseSelect() },
            else => error.UnsupportedStatement,
        };
    }

    /// Parse SELECT statement
    fn parseSelect(self: *Self) !SelectStmt {
        _ = try self.expect(.SELECT);

        // DISTINCT clause
        const distinct = self.match(&[_]TokenType{.DISTINCT});

        // SELECT columns
        const columns = try self.parseSelectList();

        // FROM clause
        _ = try self.expect(.FROM);
        const from = try self.parseTableRef();

        // WHERE clause (optional)
        const where_clause = if (self.match(&[_]TokenType{.WHERE}))
            try self.parseExpr()
        else
            null;

        // GROUP BY clause (optional)
        const group_by = if (self.match(&[_]TokenType{.GROUP}))
            try self.parseGroupBy()
        else
            null;

        // ORDER BY clause (optional)
        const order_by = if (self.match(&[_]TokenType{.ORDER}))
            try self.parseOrderBy()
        else
            null;

        // LIMIT clause (optional)
        const limit = if (self.match(&[_]TokenType{.LIMIT}))
            try self.parseLimitValue()
        else
            null;

        // OFFSET clause (optional)
        const offset = if (self.match(&[_]TokenType{.OFFSET}))
            try self.parseLimitValue()
        else
            null;

        return SelectStmt{
            .distinct = distinct,
            .columns = columns,
            .from = from,
            .where = where_clause,
            .group_by = group_by,
            .order_by = order_by,
            .limit = limit,
            .offset = offset,
        };
    }

    // ========================================================================
    // SELECT list parsing
    // ========================================================================

    fn parseSelectList(self: *Self) ![]ast.SelectItem {
        var items = std.ArrayList(ast.SelectItem){};
        errdefer items.deinit(self.allocator);

        while (true) {
            try items.append(self.allocator, try self.parseSelectItem());

            if (!self.match(&[_]TokenType{.COMMA})) break;
        }

        return items.toOwnedSlice(self.allocator);
    }

    fn parseSelectItem(self: *Self) !ast.SelectItem {
        // Special case for SELECT *
        if (self.check(.STAR)) {
            self.advance();
            return ast.SelectItem{
                .expr = Expr{
                    .column = .{
                        .table = null,
                        .name = "*",
                    },
                },
                .alias = null,
            };
        }

        const expr = try self.parseExpr();

        // Check for AS alias
        const alias = if (self.match(&[_]TokenType{.AS})) blk: {
            const tok = try self.expect(.IDENTIFIER);
            break :blk tok.lexeme;
        } else null;

        return ast.SelectItem{
            .expr = expr,
            .alias = alias,
        };
    }

    // ========================================================================
    // Expression parsing (operator precedence)
    // ========================================================================

    fn parseExpr(self: *Self) anyerror!Expr {
        return try self.parseOrExpr();
    }

    fn parseOrExpr(self: *Self) anyerror!Expr {
        var left = try self.parseAndExpr();

        while (self.match(&[_]TokenType{.OR})) {
            const right_ptr = try self.allocator.create(Expr);
            right_ptr.* = try self.parseAndExpr();

            const left_ptr = try self.allocator.create(Expr);
            left_ptr.* = left;

            left = Expr{
                .binary = .{
                    .op = .@"or",
                    .left = left_ptr,
                    .right = right_ptr,
                },
            };
        }

        return left;
    }

    fn parseAndExpr(self: *Self) anyerror!Expr {
        var left = try self.parseComparisonExpr();

        while (self.match(&[_]TokenType{.AND})) {
            const right_ptr = try self.allocator.create(Expr);
            right_ptr.* = try self.parseComparisonExpr();

            const left_ptr = try self.allocator.create(Expr);
            left_ptr.* = left;

            left = Expr{
                .binary = .{
                    .op = .@"and",
                    .left = left_ptr,
                    .right = right_ptr,
                },
            };
        }

        return left;
    }

    fn parseComparisonExpr(self: *Self) anyerror!Expr {
        const left = try self.parseAddExpr();

        // Comparison operators
        if (self.match(&[_]TokenType{.EQ})) {
            const right_ptr = try self.allocator.create(Expr);
            right_ptr.* = try self.parseAddExpr();
            const left_ptr = try self.allocator.create(Expr);
            left_ptr.* = left;
            return Expr{ .binary = .{ .op = .eq, .left = left_ptr, .right = right_ptr } };
        } else if (self.match(&[_]TokenType{.NE})) {
            const right_ptr = try self.allocator.create(Expr);
            right_ptr.* = try self.parseAddExpr();
            const left_ptr = try self.allocator.create(Expr);
            left_ptr.* = left;
            return Expr{ .binary = .{ .op = .ne, .left = left_ptr, .right = right_ptr } };
        } else if (self.match(&[_]TokenType{.LT})) {
            const right_ptr = try self.allocator.create(Expr);
            right_ptr.* = try self.parseAddExpr();
            const left_ptr = try self.allocator.create(Expr);
            left_ptr.* = left;
            return Expr{ .binary = .{ .op = .lt, .left = left_ptr, .right = right_ptr } };
        } else if (self.match(&[_]TokenType{.LE})) {
            const right_ptr = try self.allocator.create(Expr);
            right_ptr.* = try self.parseAddExpr();
            const left_ptr = try self.allocator.create(Expr);
            left_ptr.* = left;
            return Expr{ .binary = .{ .op = .le, .left = left_ptr, .right = right_ptr } };
        } else if (self.match(&[_]TokenType{.GT})) {
            const right_ptr = try self.allocator.create(Expr);
            right_ptr.* = try self.parseAddExpr();
            const left_ptr = try self.allocator.create(Expr);
            left_ptr.* = left;
            return Expr{ .binary = .{ .op = .gt, .left = left_ptr, .right = right_ptr } };
        } else if (self.match(&[_]TokenType{.GE})) {
            const right_ptr = try self.allocator.create(Expr);
            right_ptr.* = try self.parseAddExpr();
            const left_ptr = try self.allocator.create(Expr);
            left_ptr.* = left;
            return Expr{ .binary = .{ .op = .ge, .left = left_ptr, .right = right_ptr } };
        }

        return left;
    }

    fn parseAddExpr(self: *Self) anyerror!Expr {
        var left = try self.parseMulExpr();

        while (true) {
            if (self.match(&[_]TokenType{.PLUS})) {
                const right_ptr = try self.allocator.create(Expr);
                right_ptr.* = try self.parseMulExpr();
                const left_ptr = try self.allocator.create(Expr);
                left_ptr.* = left;
                left = Expr{ .binary = .{ .op = .add, .left = left_ptr, .right = right_ptr } };
            } else if (self.match(&[_]TokenType{.MINUS})) {
                const right_ptr = try self.allocator.create(Expr);
                right_ptr.* = try self.parseMulExpr();
                const left_ptr = try self.allocator.create(Expr);
                left_ptr.* = left;
                left = Expr{ .binary = .{ .op = .subtract, .left = left_ptr, .right = right_ptr } };
            } else if (self.match(&[_]TokenType{.CONCAT})) {
                const right_ptr = try self.allocator.create(Expr);
                right_ptr.* = try self.parseMulExpr();
                const left_ptr = try self.allocator.create(Expr);
                left_ptr.* = left;
                left = Expr{ .binary = .{ .op = .concat, .left = left_ptr, .right = right_ptr } };
            } else {
                break;
            }
        }

        return left;
    }

    fn parseMulExpr(self: *Self) anyerror!Expr {
        var left = try self.parsePrimary();

        while (true) {
            if (self.match(&[_]TokenType{.STAR})) {
                const right_ptr = try self.allocator.create(Expr);
                right_ptr.* = try self.parsePrimary();
                const left_ptr = try self.allocator.create(Expr);
                left_ptr.* = left;
                left = Expr{ .binary = .{ .op = .multiply, .left = left_ptr, .right = right_ptr } };
            } else if (self.match(&[_]TokenType{.SLASH})) {
                const right_ptr = try self.allocator.create(Expr);
                right_ptr.* = try self.parsePrimary();
                const left_ptr = try self.allocator.create(Expr);
                left_ptr.* = left;
                left = Expr{ .binary = .{ .op = .divide, .left = left_ptr, .right = right_ptr } };
            } else {
                break;
            }
        }

        return left;
    }

    fn parsePrimary(self: *Self) anyerror!Expr {
        const tok = self.current() orelse return error.UnexpectedEOF;

        switch (tok.type) {
            // Numbers
            .NUMBER => {
                self.advance();
                // Try parsing as integer first
                const value = std.fmt.parseInt(i64, tok.lexeme, 10) catch |err| {
                    if (err == error.InvalidCharacter) {
                        // Parse as float
                        const f = try std.fmt.parseFloat(f64, tok.lexeme);
                        return Expr{ .value = ast.Value{ .float = f } };
                    }
                    return err;
                };
                return Expr{ .value = ast.Value{ .integer = value } };
            },

            // Strings
            .STRING => {
                self.advance();
                // Remove quotes
                const unquoted = tok.lexeme[1 .. tok.lexeme.len - 1];
                return Expr{ .value = ast.Value{ .string = unquoted } };
            },

            // Parameters (?)
            .PARAMETER => {
                self.advance();
                const param_idx = self.param_count;
                self.param_count += 1;
                return Expr{ .value = ast.Value{ .parameter = param_idx } };
            },

            // Identifiers (columns or function calls)
            .IDENTIFIER => {
                const name = tok.lexeme;
                self.advance();

                // Check for function call
                if (self.check(.LPAREN)) {
                    return try self.parseFunctionCall(name);
                }

                // Column reference
                return Expr{
                    .column = .{
                        .table = null,
                        .name = name,
                    },
                };
            },

            // Aggregate function keywords (COUNT, SUM, AVG, MIN, MAX)
            .COUNT, .SUM, .AVG, .MIN, .MAX => {
                const name = tok.lexeme;
                self.advance();
                return try self.parseFunctionCall(name);
            },

            // Parenthesized expression
            .LPAREN => {
                self.advance();
                const expr = try self.parseExpr();
                _ = try self.expect(.RPAREN);
                return expr;
            },

            else => return error.UnexpectedToken,
        }
    }

    fn parseFunctionCall(self: *Self, name: []const u8) anyerror!Expr {
        _ = try self.expect(.LPAREN);

        // Check for DISTINCT
        const distinct = self.match(&[_]TokenType{.DISTINCT});

        // Parse arguments
        var args = std.ArrayList(Expr){};
        errdefer args.deinit(self.allocator);

        // Check for empty argument list or STAR (for COUNT(*))
        if (!self.check(.RPAREN)) {
            while (true) {
                // Special case: STAR inside function call means "all columns" (e.g., COUNT(*))
                if (self.check(.STAR)) {
                    self.advance();
                    try args.append(self.allocator, Expr{
                        .column = .{
                            .table = null,
                            .name = "*",
                        },
                    });
                } else {
                    try args.append(self.allocator, try self.parseExpr());
                }
                if (!self.match(&[_]TokenType{.COMMA})) break;
            }
        }

        _ = try self.expect(.RPAREN);

        return Expr{
            .call = .{
                .name = name,
                .args = try args.toOwnedSlice(self.allocator),
                .distinct = distinct,
            },
        };
    }

    // ========================================================================
    // Clause parsing
    // ========================================================================

    fn parseTableRef(self: *Self) !ast.TableRef {
        const name_tok = try self.expect(.IDENTIFIER);

        // Check for alias
        const alias = if (self.match(&[_]TokenType{.AS})) blk: {
            const tok = try self.expect(.IDENTIFIER);
            break :blk tok.lexeme;
        } else null;

        return ast.TableRef{
            .name = name_tok.lexeme,
            .alias = alias,
        };
    }

    fn parseGroupBy(self: *Self) !ast.GroupBy {
        _ = try self.expect(.BY);

        var columns = std.ArrayList([]const u8){};
        errdefer columns.deinit(self.allocator);

        while (true) {
            const tok = try self.expect(.IDENTIFIER);
            try columns.append(self.allocator, tok.lexeme);

            if (!self.match(&[_]TokenType{.COMMA})) break;
        }

        // HAVING clause (optional)
        const having = if (self.match(&[_]TokenType{.HAVING}))
            try self.parseExpr()
        else
            null;

        return ast.GroupBy{
            .columns = try columns.toOwnedSlice(self.allocator),
            .having = having,
        };
    }

    fn parseOrderBy(self: *Self) ![]ast.OrderBy {
        _ = try self.expect(.BY);

        var items = std.ArrayList(ast.OrderBy){};
        errdefer items.deinit(self.allocator);

        while (true) {
            const tok = try self.expect(.IDENTIFIER);

            const direction = if (self.match(&[_]TokenType{.DESC}))
                ast.OrderDirection.desc
            else blk: {
                _ = self.match(&[_]TokenType{.ASC}); // Optional ASC
                break :blk ast.OrderDirection.asc;
            };

            try items.append(self.allocator, ast.OrderBy{
                .column = tok.lexeme,
                .direction = direction,
            });

            if (!self.match(&[_]TokenType{.COMMA})) break;
        }

        return items.toOwnedSlice(self.allocator);
    }

    fn parseLimitValue(self: *Self) !u32 {
        const tok = try self.expect(.NUMBER);
        return try std.fmt.parseInt(u32, tok.lexeme, 10);
    }
};

// ============================================================================
// Helper function for parsing SQL strings
// ============================================================================

pub fn parseSQL(sql: []const u8, allocator: std.mem.Allocator) !Statement {
    var lex = lexer.Lexer.init(sql);
    const tokens = try lex.tokenize(allocator);
    defer allocator.free(tokens);

    var parser = Parser.init(tokens, allocator);
    return try parser.parseStatement();
}

// ============================================================================
// Tests
// ============================================================================

test "parse simple SELECT" {
    const sql = "SELECT id FROM users";
    const allocator = std.testing.allocator;

    const stmt = try parseSQL(sql, allocator);
    try std.testing.expect(stmt == .select);
    try std.testing.expectEqual(@as(usize, 1), stmt.select.columns.len);
}

test "parse SELECT with WHERE" {
    const sql = "SELECT name FROM users WHERE id = 42";
    const allocator = std.testing.allocator;

    const stmt = try parseSQL(sql, allocator);
    try std.testing.expect(stmt == .select);
    try std.testing.expect(stmt.select.where != null);
}

test "parse SELECT with parameter" {
    const sql = "SELECT * FROM users WHERE id = ?";
    const allocator = std.testing.allocator;

    const stmt = try parseSQL(sql, allocator);
    try std.testing.expect(stmt == .select);
    try std.testing.expect(stmt.select.where != null);

    // Verify the WHERE clause contains a parameter
    const where = stmt.select.where.?;
    try std.testing.expect(where == .binary);
    const right = where.binary.right.*;
    try std.testing.expect(right == .value);
    try std.testing.expect(right.value == .parameter);
    try std.testing.expectEqual(@as(u32, 0), right.value.parameter);
}

test "parse SELECT with multiple parameters" {
    const sql = "SELECT * FROM users WHERE id > ? AND name = ?";
    const allocator = std.testing.allocator;

    const stmt = try parseSQL(sql, allocator);
    try std.testing.expect(stmt == .select);
    try std.testing.expect(stmt.select.where != null);

    // Verify there are two parameters with indices 0 and 1
    const where = stmt.select.where.?;
    try std.testing.expect(where == .binary); // AND expression
    try std.testing.expectEqual(ast.BinaryOp.@"and", where.binary.op);

    // Left side: id > ?  (param index 0)
    const left_binary = where.binary.left.*.binary;
    try std.testing.expectEqual(ast.BinaryOp.gt, left_binary.op);
    try std.testing.expectEqual(@as(u32, 0), left_binary.right.*.value.parameter);

    // Right side: name = ?  (param index 1)
    const right_binary = where.binary.right.*.binary;
    try std.testing.expectEqual(ast.BinaryOp.eq, right_binary.op);
    try std.testing.expectEqual(@as(u32, 1), right_binary.right.*.value.parameter);
}
