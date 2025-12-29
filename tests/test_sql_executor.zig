//! SQL Executor Integration Tests
//!
//! Tests the SQL executor against real Lance files

const std = @import("std");
const Table = @import("lanceql.table").Table;
const ast = @import("lanceql.sql.ast");
const parser = @import("lanceql.sql.parser");
const Executor = @import("lanceql.sql.executor").Executor;
const Value = ast.Value;

test "execute simple SELECT *" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL
    const sql = "SELECT * FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
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
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table WHERE id > 2
    const sql = "SELECT * FROM table WHERE id > 2";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
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
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table ORDER BY id DESC
    const sql = "SELECT * FROM table ORDER BY id DESC";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
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
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table LIMIT 3
    const sql = "SELECT * FROM table LIMIT 3";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
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
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table LIMIT 2 OFFSET 2
    const sql = "SELECT * FROM table LIMIT 2 OFFSET 2";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
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
    const lance_data = @embedFile("fixtures/simple_float64.lance/data/101000000010001000111010eda0664313bd731181abf5bde4.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL
    const sql = "SELECT * FROM table WHERE value > 3.0";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
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
    const lance_data = @embedFile("fixtures/mixed_types.lance/data/11100100001000010010010060d60b4085bd08dcf790581192.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL
    const sql = "SELECT * FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
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

test "execute SELECT * on better-sqlite3 fixture" {
    const allocator = std.testing.allocator;

    // Read the better-sqlite3 simple fixture (has 'a' string and 'b' int64)
    const data = @embedFile("fixtures/better-sqlite3/simple.lance/data/1010001110011001100010108ba1604433ac0cda4c27f6809f.lance");

    var table = try Table.init(allocator, data);
    defer table.deinit();

    // Verify schema
    const schema = table.schema orelse return error.NoSchema;
    try std.testing.expectEqual(@as(usize, 2), schema.fields.len);

    // Parse SQL
    const sql = "SELECT * FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Debug output
    std.debug.print("\nbetter-sqlite3 SELECT * result:\n", .{});
    std.debug.print("  columns.len = {d}\n", .{result.columns.len});
    std.debug.print("  row_count = {d}\n", .{result.row_count});
    for (result.columns, 0..) |col, i| {
        std.debug.print("  column[{d}]: name='{s}', type={s}\n", .{
            i,
            col.name,
            switch (col.data) {
                .int64 => "int64",
                .int32 => "int32",
                .float64 => "float64",
                .float32 => "float32",
                .bool_ => "bool",
                .string => "string",
                .timestamp_s => "timestamp_s",
                .timestamp_ms => "timestamp_ms",
                .timestamp_us => "timestamp_us",
                .timestamp_ns => "timestamp_ns",
                .date32 => "date32",
                .date64 => "date64",
            },
        });
    }

    // Verify results - should have 2 columns
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 10), result.row_count);

    // Check column types (a is string, b is int64)
    try std.testing.expect(result.columns[0].data == .string);
    try std.testing.expect(result.columns[1].data == .int64);

    // Check string values
    const strings = result.columns[0].data.string;
    try std.testing.expectEqualStrings("foo", strings[0]);
    try std.testing.expectEqualStrings("bar", strings[1]);

    // Check int values
    const ints = result.columns[1].data.int64;
    try std.testing.expectEqual(@as(i64, 1), ints[0]);
    try std.testing.expectEqual(@as(i64, 2), ints[1]);
}

// ============================================================================
// GROUP BY / Aggregate Tests
// ============================================================================

test "execute SELECT COUNT(*)" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT COUNT(*) FROM table
    const sql = "SELECT COUNT(*) FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 1 row with count 5
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.row_count);

    try std.testing.expect(result.columns[0].data == .int64);
    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 5), values[0]);
}

test "execute SELECT SUM(id)" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT SUM(id) FROM table
    const sql = "SELECT SUM(id) FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 1 row with sum 15 (1+2+3+4+5)
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.row_count);

    try std.testing.expect(result.columns[0].data == .int64);
    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 15), values[0]);
}

test "execute SELECT AVG(id)" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT AVG(id) FROM table
    const sql = "SELECT AVG(id) FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 1 row with avg 3 (15/5)
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.row_count);

    try std.testing.expect(result.columns[0].data == .int64);
    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
}

test "execute SELECT MIN/MAX(id)" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Test MIN
    {
        const sql = "SELECT MIN(id) FROM table";
        var stmt = try parser.parseSQL(sql, allocator);
        defer ast.deinitSelectStmt(&stmt.select, allocator);

        var executor = Executor.init(&table, allocator);
        defer executor.deinit();
        var result = try executor.execute(&stmt.select, &[_]Value{});
        defer result.deinit();

        try std.testing.expectEqual(@as(i64, 1), result.columns[0].data.int64[0]);
    }

    // Test MAX
    {
        const sql = "SELECT MAX(id) FROM table";
        var stmt = try parser.parseSQL(sql, allocator);
        defer ast.deinitSelectStmt(&stmt.select, allocator);

        var executor = Executor.init(&table, allocator);
        defer executor.deinit();
        var result = try executor.execute(&stmt.select, &[_]Value{});
        defer result.deinit();

        try std.testing.expectEqual(@as(i64, 5), result.columns[0].data.int64[0]);
    }
}

test "execute SELECT with GROUP BY" {
    const allocator = std.testing.allocator;

    // Use better-sqlite3 fixture which has 'a' (string) and 'b' (int64)
    // Values: a=['foo','bar','baz','qux','quux','corge','grault','garply','waldo','fred'], b=[1..10]
    // All a values are unique, so GROUP BY a gives 10 groups
    const data = @embedFile("fixtures/better-sqlite3/simple.lance/data/1010001110011001100010108ba1604433ac0cda4c27f6809f.lance");

    var table = try Table.init(allocator, data);
    defer table.deinit();

    // Parse SQL: SELECT a, COUNT(*) FROM table GROUP BY a
    const sql = "SELECT a, COUNT(*) FROM table GROUP BY a";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 10 groups (all unique strings)
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 10), result.row_count);

    // Column 0 should be 'a' (string), Column 1 should be COUNT(*) (int64)
    try std.testing.expect(result.columns[0].data == .string);
    try std.testing.expect(result.columns[1].data == .int64);

    // Each group should have exactly 1 row (all unique strings)
    const counts = result.columns[1].data.int64;
    for (counts) |c| {
        try std.testing.expectEqual(@as(i64, 1), c);
    }
}

test "execute SELECT with GROUP BY and SUM" {
    const allocator = std.testing.allocator;

    // Use better-sqlite3 fixture: a is unique strings, b=[1..10]
    // Since each a value is unique, GROUP BY a gives 10 groups with SUM = b value for each
    const data = @embedFile("fixtures/better-sqlite3/simple.lance/data/1010001110011001100010108ba1604433ac0cda4c27f6809f.lance");

    var table = try Table.init(allocator, data);
    defer table.deinit();

    // Parse SQL: SELECT a, SUM(b) FROM table GROUP BY a
    const sql = "SELECT a, SUM(b) FROM table GROUP BY a";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - 10 groups
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 10), result.row_count);

    // Check that we have string and int64 columns
    try std.testing.expect(result.columns[0].data == .string);
    try std.testing.expect(result.columns[1].data == .int64);

    // Verify total sum of all SUM(b) values equals 1+2+3+...+10 = 55
    const sums = result.columns[1].data.int64;
    var total: i64 = 0;
    for (sums) |s| {
        total += s;
    }
    try std.testing.expectEqual(@as(i64, 55), total);
}

// ============================================================================
// Expression Evaluation Tests (Phase 2)
// ============================================================================

test "execute SELECT with arithmetic multiplication" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT id * 2 AS doubled FROM table
    const sql = "SELECT id * 2 AS doubled FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 5 rows with doubled values
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);
    try std.testing.expectEqualStrings("doubled", result.columns[0].name);

    try std.testing.expect(result.columns[0].data == .int64);
    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 2), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
    try std.testing.expectEqual(@as(i64, 6), values[2]);
    try std.testing.expectEqual(@as(i64, 8), values[3]);
    try std.testing.expectEqual(@as(i64, 10), values[4]);
}

test "execute SELECT with arithmetic addition" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT id + 10 AS plus_ten FROM table
    const sql = "SELECT id + 10 AS plus_ten FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 5 rows with +10 values
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    try std.testing.expect(result.columns[0].data == .int64);
    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 11), values[0]);
    try std.testing.expectEqual(@as(i64, 12), values[1]);
    try std.testing.expectEqual(@as(i64, 13), values[2]);
    try std.testing.expectEqual(@as(i64, 14), values[3]);
    try std.testing.expectEqual(@as(i64, 15), values[4]);
}

test "execute SELECT with division" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT id / 2 AS halved FROM table
    const sql = "SELECT id / 2 AS halved FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - division returns float64
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    try std.testing.expect(result.columns[0].data == .float64);
    const values = result.columns[0].data.float64;
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), values[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), values[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), values[4], 0.001);
}

test "execute SELECT with complex expression" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT id * 2 + 1 AS computed FROM table
    const sql = "SELECT id * 2 + 1 AS computed FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - id * 2 + 1: [3, 5, 7, 9, 11]
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    try std.testing.expect(result.columns[0].data == .int64);
    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
    try std.testing.expectEqual(@as(i64, 5), values[1]);
    try std.testing.expectEqual(@as(i64, 7), values[2]);
    try std.testing.expectEqual(@as(i64, 9), values[3]);
    try std.testing.expectEqual(@as(i64, 11), values[4]);
}

test "execute SELECT with mixed column and expression" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT id, id * 2 AS doubled FROM table
    const sql = "SELECT id, id * 2 AS doubled FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - 2 columns
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    // First column is 'id'
    try std.testing.expectEqualStrings("id", result.columns[0].name);
    const ids = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), ids[0]);
    try std.testing.expectEqual(@as(i64, 5), ids[4]);

    // Second column is 'doubled'
    try std.testing.expectEqualStrings("doubled", result.columns[1].name);
    const doubled = result.columns[1].data.int64;
    try std.testing.expectEqual(@as(i64, 2), doubled[0]);
    try std.testing.expectEqual(@as(i64, 10), doubled[4]);
}

test "execute SELECT with UPPER function" {
    const allocator = std.testing.allocator;

    // Use better-sqlite3 fixture which has 'a' (string) column
    const data = @embedFile("fixtures/better-sqlite3/simple.lance/data/1010001110011001100010108ba1604433ac0cda4c27f6809f.lance");

    var table = try Table.init(allocator, data);
    defer table.deinit();

    // Parse SQL: SELECT UPPER(a) AS upper_name FROM table LIMIT 3
    const sql = "SELECT UPPER(a) AS upper_name FROM table LIMIT 3";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    try std.testing.expect(result.columns[0].data == .string);
    const values = result.columns[0].data.string;
    try std.testing.expectEqualStrings("FOO", values[0]);
    try std.testing.expectEqualStrings("BAR", values[1]);
    try std.testing.expectEqualStrings("BAZ", values[2]);
}

test "execute SELECT with LENGTH function" {
    const allocator = std.testing.allocator;

    // Use better-sqlite3 fixture which has 'a' (string) column
    const data = @embedFile("fixtures/better-sqlite3/simple.lance/data/1010001110011001100010108ba1604433ac0cda4c27f6809f.lance");

    var table = try Table.init(allocator, data);
    defer table.deinit();

    // Parse SQL: SELECT a, LENGTH(a) AS len FROM table LIMIT 3
    const sql = "SELECT a, LENGTH(a) AS len FROM table LIMIT 3";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    // First column is 'a' (string)
    try std.testing.expect(result.columns[0].data == .string);

    // Second column is 'len' (int64 from LENGTH)
    try std.testing.expect(result.columns[1].data == .int64);
    const lengths = result.columns[1].data.int64;
    try std.testing.expectEqual(@as(i64, 3), lengths[0]); // "foo" = 3
    try std.testing.expectEqual(@as(i64, 3), lengths[1]); // "bar" = 3
    try std.testing.expectEqual(@as(i64, 3), lengths[2]); // "baz" = 3
}

test "execute SELECT with ABS function" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT ABS(id - 3) AS abs_val FROM table
    // This should give: |1-3|=2, |2-3|=1, |3-3|=0, |4-3|=1, |5-3|=2
    const sql = "SELECT ABS(id - 3) AS abs_val FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    // ABS returns float64 (from inferFunctionReturnType)
    try std.testing.expect(result.columns[0].data == .float64);
    const values = result.columns[0].data.float64;
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), values[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), values[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), values[4], 0.001);
}

test "execute SELECT with string concatenation" {
    const allocator = std.testing.allocator;

    // Use better-sqlite3 fixture which has 'a' (string) column
    const data = @embedFile("fixtures/better-sqlite3/simple.lance/data/1010001110011001100010108ba1604433ac0cda4c27f6809f.lance");

    var table = try Table.init(allocator, data);
    defer table.deinit();

    // Parse SQL: SELECT a || '_suffix' AS with_suffix FROM table LIMIT 3
    const sql = "SELECT a || '_suffix' AS with_suffix FROM table LIMIT 3";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    try std.testing.expect(result.columns[0].data == .string);
    const values = result.columns[0].data.string;
    try std.testing.expectEqualStrings("foo_suffix", values[0]);
    try std.testing.expectEqualStrings("bar_suffix", values[1]);
    try std.testing.expectEqualStrings("baz_suffix", values[2]);
}

// ============================================================================
// Parameter Binding Tests (Phase 3)
// ============================================================================

test "execute SELECT with single integer parameter" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table WHERE id = ?
    const sql = "SELECT * FROM table WHERE id = ?";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute with parameter [3]
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    const params = [_]Value{Value{ .integer = 3 }};
    var result = try executor.execute(&stmt.select, &params);
    defer result.deinit();

    // Verify results - should have 1 row with id=3
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
}

test "execute SELECT with multiple integer parameters" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table WHERE id > ? AND id < ?
    const sql = "SELECT * FROM table WHERE id > ? AND id < ?";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute with parameters [2, 5] - should match id=3, id=4
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    const params = [_]Value{ Value{ .integer = 2 }, Value{ .integer = 5 } };
    var result = try executor.execute(&stmt.select, &params);
    defer result.deinit();

    // Verify results - should have 2 rows with id=3 and id=4
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
}

test "execute SELECT with float parameter" {
    const allocator = std.testing.allocator;

    // Open test Lance file with float64 (value: 1.5, 2.5, 3.5, 4.5, 5.5)
    const lance_data = @embedFile("fixtures/simple_float64.lance/data/101000000010001000111010eda0664313bd731181abf5bde4.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT * FROM table WHERE value > ?
    const sql = "SELECT * FROM table WHERE value > ?";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute with parameter [3.0] - should match 3.5, 4.5, 5.5
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    const params = [_]Value{Value{ .float = 3.0 }};
    var result = try executor.execute(&stmt.select, &params);
    defer result.deinit();

    // Verify results - should have 3 rows
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.float64;
    try std.testing.expectEqual(@as(f64, 3.5), values[0]);
    try std.testing.expectEqual(@as(f64, 4.5), values[1]);
    try std.testing.expectEqual(@as(f64, 5.5), values[2]);
}

test "execute SELECT with string parameter" {
    const allocator = std.testing.allocator;

    // Use better-sqlite3 fixture: a is strings, b=[1..10]
    const data = @embedFile("fixtures/better-sqlite3/simple.lance/data/1010001110011001100010108ba1604433ac0cda4c27f6809f.lance");

    var table = try Table.init(allocator, data);
    defer table.deinit();

    // Parse SQL: SELECT b FROM table WHERE a = ?
    const sql = "SELECT b FROM table WHERE a = ?";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute with parameter ['foo'] - should match the first row where a='foo'
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    const params = [_]Value{Value{ .string = "foo" }};
    var result = try executor.execute(&stmt.select, &params);
    defer result.deinit();

    // Verify results - should have 1 row where a='foo' (first row, b=1)
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
}

test "parameter out of bounds returns error" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL with 2 parameters: SELECT * FROM table WHERE id > ? AND id < ?
    const sql = "SELECT * FROM table WHERE id > ? AND id < ?";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute with only 1 parameter - should fail
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    const params = [_]Value{Value{ .integer = 2 }}; // Only 1 param, need 2
    const result = executor.execute(&stmt.select, &params);

    // Should return ParameterOutOfBounds error
    try std.testing.expectError(error.ParameterOutOfBounds, result);
}

// ============================================================================
// DISTINCT Tests (Phase 4)
// ============================================================================

test "execute SELECT DISTINCT on unique values" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5 - all unique)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT DISTINCT id FROM table
    const sql = "SELECT DISTINCT id FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // All values are unique, so DISTINCT should return all 5 rows
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 2), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
    try std.testing.expectEqual(@as(i64, 4), values[3]);
    try std.testing.expectEqual(@as(i64, 5), values[4]);
}

test "execute SELECT DISTINCT with WHERE clause" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT DISTINCT id FROM table WHERE id > 2
    const sql = "SELECT DISTINCT id FROM table WHERE id > 2";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Should return 3 unique rows (3, 4, 5)
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
    try std.testing.expectEqual(@as(i64, 5), values[2]);
}

test "execute SELECT DISTINCT with ORDER BY" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT DISTINCT id FROM table ORDER BY id DESC
    const sql = "SELECT DISTINCT id FROM table ORDER BY id DESC";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Should return all 5 rows in descending order
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 5), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
    try std.testing.expectEqual(@as(i64, 2), values[3]);
    try std.testing.expectEqual(@as(i64, 1), values[4]);
}

test "execute SELECT DISTINCT with LIMIT" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT DISTINCT id FROM table LIMIT 3
    const sql = "SELECT DISTINCT id FROM table LIMIT 3";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Should return only 3 rows
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);
}

test "execute SELECT DISTINCT on strings" {
    const allocator = std.testing.allocator;

    // Use better-sqlite3 fixture which has 'a' (string) column
    const data = @embedFile("fixtures/better-sqlite3/simple.lance/data/1010001110011001100010108ba1604433ac0cda4c27f6809f.lance");

    var table = try Table.init(allocator, data);
    defer table.deinit();

    // Parse SQL: SELECT DISTINCT a FROM table
    const sql = "SELECT DISTINCT a FROM table";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have unique string values
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expect(result.columns[0].data == .string);

    // The fixture has strings like "foo", "bar", "baz", etc.
    // All should be unique so row count should match original
    try std.testing.expect(result.row_count > 0);
}

// =============================================================================
// @logic_table Integration Tests
// =============================================================================

test "executor registerLogicTableAlias" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    // Register alias
    try executor.registerLogicTableAlias("t", "FraudDetector");

    // Check it's stored
    const class_name = executor.logic_table_aliases.get("t");
    try std.testing.expect(class_name != null);
    try std.testing.expectEqualStrings("FraudDetector", class_name.?);
}

test "executor registerLogicTableAlias rejects duplicates" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/0100110011011011000010005445a8407eb6f52a3c35f80bd3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    // Register alias once
    try executor.registerLogicTableAlias("t", "FraudDetector");

    // Attempt to register again - should fail
    const result = executor.registerLogicTableAlias("t", "OtherClass");
    try std.testing.expectError(error.DuplicateAlias, result);
}
