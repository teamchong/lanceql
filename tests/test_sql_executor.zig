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
                .float64 => "float64",
                .string => "string",
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
