//! SQL Lexer (Tokenizer)
//!
//! Converts SQL text into a stream of tokens for parsing.
//! Supports SELECT statements with WHERE, ORDER BY, LIMIT, GROUP BY, and vector search extensions.

const std = @import("std");

/// Token types for SQL lexer
pub const TokenType = enum {
    // Keywords
    SELECT,
    FROM,
    WHERE,
    AND,
    OR,
    NOT,
    IN,
    BETWEEN,
    LIKE,
    IS,
    NULL,
    AS,
    DISTINCT,
    ORDER,
    BY,
    ASC,
    DESC,
    LIMIT,
    OFFSET,
    GROUP,
    HAVING,

    // Vector search extension
    NEAR,
    TOPK,
    FILE,

    // Aggregate functions
    COUNT,
    SUM,
    AVG,
    MIN,
    MAX,

    // Literals
    IDENTIFIER,
    NUMBER,
    STRING,

    // Operators
    STAR,      // *
    EQ,        // =
    NE,        // !=, <>
    LT,        // <
    LE,        // <=
    GT,        // >
    GE,        // >=
    PLUS,      // +
    MINUS,     // -
    SLASH,     // /
    COMMA,     // ,
    DOT,       // .
    LPAREN,    // (
    RPAREN,    // )
    SEMICOLON, // ;

    EOF,
};

/// A token with its type and lexeme
pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    position: usize,
};

/// SQL Lexer
pub const Lexer = struct {
    input: []const u8,
    position: usize,

    const Self = @This();

    /// Create a new lexer
    pub fn init(input: []const u8) Self {
        return Self{
            .input = input,
            .position = 0,
        };
    }

    /// Peek at current character without advancing
    fn peek(self: *const Self) ?u8 {
        if (self.position >= self.input.len) return null;
        return self.input[self.position];
    }

    /// Peek ahead n characters
    fn peekAhead(self: *const Self, n: usize) ?u8 {
        const pos = self.position + n;
        if (pos >= self.input.len) return null;
        return self.input[pos];
    }

    /// Advance and return current character
    fn advance(self: *Self) ?u8 {
        if (self.position >= self.input.len) return null;
        const ch = self.input[self.position];
        self.position += 1;
        return ch;
    }

    /// Skip whitespace
    fn skipWhitespace(self: *Self) void {
        while (self.peek()) |ch| {
            if (!std.ascii.isWhitespace(ch)) break;
            _ = self.advance();
        }
    }

    /// Read identifier or keyword
    fn readIdentifier(self: *Self) !Token {
        const start = self.position;

        while (self.peek()) |ch| {
            if (std.ascii.isAlphanumeric(ch) or ch == '_') {
                _ = self.advance();
            } else {
                break;
            }
        }

        const lexeme = self.input[start..self.position];
        const token_type = keywordOrIdentifier(lexeme);

        return Token{
            .type = token_type,
            .lexeme = lexeme,
            .position = start,
        };
    }

    /// Read numeric literal
    fn readNumber(self: *Self) !Token {
        const start = self.position;

        // Read digits
        while (self.peek()) |ch| {
            if (std.ascii.isDigit(ch)) {
                _ = self.advance();
            } else if (ch == '.') {
                // Handle decimal point
                _ = self.advance();
                // Read fractional part
                while (self.peek()) |digit| {
                    if (std.ascii.isDigit(digit)) {
                        _ = self.advance();
                    } else {
                        break;
                    }
                }
                break;
            } else {
                break;
            }
        }

        const lexeme = self.input[start..self.position];
        return Token{
            .type = .NUMBER,
            .lexeme = lexeme,
            .position = start,
        };
    }

    /// Read string literal (single or double quoted)
    fn readString(self: *Self, quote: u8) !Token {
        const start = self.position - 1; // Include opening quote

        while (self.peek()) |ch| {
            if (ch == quote) {
                _ = self.advance();
                break;
            } else if (ch == '\\') {
                // Handle escape sequences
                _ = self.advance();
                _ = self.advance(); // Skip escaped character
            } else {
                _ = self.advance();
            }
        }

        const lexeme = self.input[start..self.position];
        return Token{
            .type = .STRING,
            .lexeme = lexeme,
            .position = start,
        };
    }

    /// Get next token
    pub fn nextToken(self: *Self) !Token {
        self.skipWhitespace();

        const ch = self.peek() orelse {
            return Token{
                .type = .EOF,
                .lexeme = "",
                .position = self.position,
            };
        };

        const start = self.position;

        // Identifiers and keywords
        if (std.ascii.isAlphabetic(ch) or ch == '_') {
            return self.readIdentifier();
        }

        // Numbers
        if (std.ascii.isDigit(ch)) {
            return self.readNumber();
        }

        // Strings
        if (ch == '\'' or ch == '"') {
            _ = self.advance();
            return self.readString(ch);
        }

        // Operators and punctuation
        _ = self.advance();

        switch (ch) {
            '*' => return Token{ .type = .STAR, .lexeme = "*", .position = start },
            ',' => return Token{ .type = .COMMA, .lexeme = ",", .position = start },
            '.' => return Token{ .type = .DOT, .lexeme = ".", .position = start },
            '(' => return Token{ .type = .LPAREN, .lexeme = "(", .position = start },
            ')' => return Token{ .type = .RPAREN, .lexeme = ")", .position = start },
            ';' => return Token{ .type = .SEMICOLON, .lexeme = ";", .position = start },
            '+' => return Token{ .type = .PLUS, .lexeme = "+", .position = start },
            '-' => return Token{ .type = .MINUS, .lexeme = "-", .position = start },
            '/' => return Token{ .type = .SLASH, .lexeme = "/", .position = start },

            '=' => return Token{ .type = .EQ, .lexeme = "=", .position = start },

            '!' => {
                if (self.peek() == '=') {
                    _ = self.advance();
                    return Token{ .type = .NE, .lexeme = "!=", .position = start };
                }
                return error.UnexpectedCharacter;
            },

            '<' => {
                if (self.peek() == '=') {
                    _ = self.advance();
                    return Token{ .type = .LE, .lexeme = "<=", .position = start };
                } else if (self.peek() == '>') {
                    _ = self.advance();
                    return Token{ .type = .NE, .lexeme = "<>", .position = start };
                }
                return Token{ .type = .LT, .lexeme = "<", .position = start };
            },

            '>' => {
                if (self.peek() == '=') {
                    _ = self.advance();
                    return Token{ .type = .GE, .lexeme = ">=", .position = start };
                }
                return Token{ .type = .GT, .lexeme = ">", .position = start };
            },

            else => return error.UnexpectedCharacter,
        }
    }

    /// Tokenize entire input into array
    pub fn tokenize(self: *Self, allocator: std.mem.Allocator) ![]Token {
        var tokens = std.ArrayList(Token){};
        try tokens.ensureTotalCapacity(allocator, 32);
        errdefer tokens.deinit(allocator);

        while (true) {
            const token = try self.nextToken();
            try tokens.append(allocator, token);
            if (token.type == .EOF) break;
        }

        return tokens.toOwnedSlice(allocator);
    }
};

/// Map string to keyword or return IDENTIFIER
fn keywordOrIdentifier(word: []const u8) TokenType {
    // Convert to uppercase for case-insensitive comparison
    var upper_buf: [64]u8 = undefined;
    if (word.len > upper_buf.len) return .IDENTIFIER;

    for (word, 0..) |ch, i| {
        upper_buf[i] = std.ascii.toUpper(ch);
    }
    const upper = upper_buf[0..word.len];

    // Keywords (alphabetically sorted for binary search)
    if (std.mem.eql(u8, upper, "AND")) return .AND;
    if (std.mem.eql(u8, upper, "AS")) return .AS;
    if (std.mem.eql(u8, upper, "ASC")) return .ASC;
    if (std.mem.eql(u8, upper, "AVG")) return .AVG;
    if (std.mem.eql(u8, upper, "BETWEEN")) return .BETWEEN;
    if (std.mem.eql(u8, upper, "BY")) return .BY;
    if (std.mem.eql(u8, upper, "COUNT")) return .COUNT;
    if (std.mem.eql(u8, upper, "DESC")) return .DESC;
    if (std.mem.eql(u8, upper, "DISTINCT")) return .DISTINCT;
    if (std.mem.eql(u8, upper, "FILE")) return .FILE;
    if (std.mem.eql(u8, upper, "FROM")) return .FROM;
    if (std.mem.eql(u8, upper, "GROUP")) return .GROUP;
    if (std.mem.eql(u8, upper, "HAVING")) return .HAVING;
    if (std.mem.eql(u8, upper, "IN")) return .IN;
    if (std.mem.eql(u8, upper, "IS")) return .IS;
    if (std.mem.eql(u8, upper, "LIKE")) return .LIKE;
    if (std.mem.eql(u8, upper, "LIMIT")) return .LIMIT;
    if (std.mem.eql(u8, upper, "MAX")) return .MAX;
    if (std.mem.eql(u8, upper, "MIN")) return .MIN;
    if (std.mem.eql(u8, upper, "NEAR")) return .NEAR;
    if (std.mem.eql(u8, upper, "NOT")) return .NOT;
    if (std.mem.eql(u8, upper, "NULL")) return .NULL;
    if (std.mem.eql(u8, upper, "OFFSET")) return .OFFSET;
    if (std.mem.eql(u8, upper, "OR")) return .OR;
    if (std.mem.eql(u8, upper, "ORDER")) return .ORDER;
    if (std.mem.eql(u8, upper, "SELECT")) return .SELECT;
    if (std.mem.eql(u8, upper, "SUM")) return .SUM;
    if (std.mem.eql(u8, upper, "TOPK")) return .TOPK;
    if (std.mem.eql(u8, upper, "WHERE")) return .WHERE;

    return .IDENTIFIER;
}

// ============================================================================
// Tests
// ============================================================================

test "lexer basic tokens" {
    const sql = "SELECT * FROM table WHERE id = 42";
    var lexer = Lexer.init(sql);

    const allocator = std.testing.allocator;
    const tokens = try lexer.tokenize(allocator);
    defer allocator.free(tokens);

    try std.testing.expectEqual(TokenType.SELECT, tokens[0].type);
    try std.testing.expectEqual(TokenType.STAR, tokens[1].type);
    try std.testing.expectEqual(TokenType.FROM, tokens[2].type);
    try std.testing.expectEqual(TokenType.IDENTIFIER, tokens[3].type);
    try std.testing.expectEqual(TokenType.WHERE, tokens[4].type);
    try std.testing.expectEqual(TokenType.IDENTIFIER, tokens[5].type);
    try std.testing.expectEqual(TokenType.EQ, tokens[6].type);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[7].type);
}

test "lexer string literals" {
    const sql = "SELECT 'hello' FROM \"world\"";
    var lexer = Lexer.init(sql);

    const allocator = std.testing.allocator;
    const tokens = try lexer.tokenize(allocator);
    defer allocator.free(tokens);

    try std.testing.expectEqual(TokenType.STRING, tokens[1].type);
    try std.testing.expectEqual(TokenType.STRING, tokens[3].type);
}

test "lexer comparison operators" {
    const sql = "a != b AND c <> d AND e <= f AND g >= h";
    var lexer = Lexer.init(sql);

    const allocator = std.testing.allocator;
    const tokens = try lexer.tokenize(allocator);
    defer allocator.free(tokens);

    try std.testing.expectEqual(TokenType.NE, tokens[1].type);
    try std.testing.expectEqual(TokenType.NE, tokens[5].type);
    try std.testing.expectEqual(TokenType.LE, tokens[9].type);
    try std.testing.expectEqual(TokenType.GE, tokens[13].type);
}
