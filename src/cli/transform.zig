//! LanceQL Transform Command
//!
//! Applies SQL-like transformations to data files and outputs to Lance format.
//!
//! Usage:
//!   lanceql transform input.parquet -o output.lance --select "col1,col2"
//!   lanceql transform data.lance -o filtered.lance --filter "x > 100"
//!   lanceql transform input.csv -o output.lance --rename "old:new" --limit 1000

const std = @import("std");
const lanceql = @import("lanceql");
const writer = lanceql.encoding.writer;
const Table = lanceql.Table;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const ArrowTable = @import("lanceql.arrow_table").ArrowTable;
const AvroTable = @import("lanceql.avro_table").AvroTable;
const OrcTable = @import("lanceql.orc_table").OrcTable;
const XlsxTable = @import("lanceql.xlsx_table").XlsxTable;
const lexer = @import("lanceql.sql.lexer");
const parser = @import("lanceql.sql.parser");
const executor = @import("lanceql.sql.executor");
const ast = @import("lanceql.sql.ast");
const Result = executor.Result;
const args = @import("args.zig");

pub const TransformError = error{
    NoInputFile,
    NoOutputFile,
    FileNotFound,
    UnsupportedFormat,
    QueryError,
    WriteError,
    OutOfMemory,
};

/// File types for detection
const FileType = enum {
    lance,
    parquet,
    arrow,
    avro,
    orc,
    xlsx,
    unknown,
};

/// Run the transform command
pub fn run(allocator: std.mem.Allocator, opts: args.TransformOptions) !void {
    // Validate input
    const input_path = opts.input orelse {
        std.debug.print("Error: No input file specified\n", .{});
        std.debug.print("Usage: lanceql transform <input> -o <output> [options]\n", .{});
        return TransformError.NoInputFile;
    };

    // Validate output
    const output_path = opts.output orelse {
        std.debug.print("Error: No output file specified (use -o)\n", .{});
        return TransformError.NoOutputFile;
    };

    // Build SQL query from options
    const sql = try buildSqlQuery(allocator, input_path, opts);
    defer allocator.free(sql);

    std.debug.print("Executing: {s}\n", .{sql});

    // Read input file
    const data = std.fs.cwd().readFileAlloc(allocator, input_path, 500 * 1024 * 1024) catch |err| {
        std.debug.print("Error reading '{s}': {}\n", .{ input_path, err });
        return TransformError.FileNotFound;
    };
    defer allocator.free(data);

    // Detect file type and execute query
    const file_type = detectFileType(input_path, data);
    var result = try executeQuery(allocator, data, sql, file_type);
    defer result.deinit();

    std.debug.print("Query returned {d} rows, {d} columns\n", .{ result.row_count, result.columns.len });

    // Write result to Lance file
    try writeResultToLance(allocator, &result, output_path);
}

/// Build SQL query from transform options
fn buildSqlQuery(allocator: std.mem.Allocator, input_path: []const u8, opts: args.TransformOptions) ![]const u8 {
    var query = std.ArrayListUnmanaged(u8){};
    defer query.deinit(allocator);

    // SELECT clause
    try query.appendSlice(allocator, "SELECT ");

    if (opts.select) |select_cols| {
        // Handle rename: parse "old:new,a:b" format
        if (opts.rename) |rename_spec| {
            // Build select with AS aliases
            var col_iter = std.mem.splitScalar(u8, select_cols, ',');
            var first = true;

            while (col_iter.next()) |col| {
                if (!first) try query.appendSlice(allocator, ", ");
                first = false;

                const trimmed = std.mem.trim(u8, col, " ");

                // Check if this column has a rename
                const new_name = findRename(rename_spec, trimmed);
                if (new_name) |name| {
                    try query.appendSlice(allocator, trimmed);
                    try query.appendSlice(allocator, " AS ");
                    try query.appendSlice(allocator, name);
                } else {
                    try query.appendSlice(allocator, trimmed);
                }
            }
        } else {
            try query.appendSlice(allocator, select_cols);
        }
    } else if (opts.rename) |rename_spec| {
        // SELECT * with renames - need to expand
        // For now, just use * and note rename doesn't work without --select
        _ = rename_spec;
        try query.appendSlice(allocator, "*");
    } else {
        try query.appendSlice(allocator, "*");
    }

    // FROM clause
    try query.appendSlice(allocator, " FROM '");
    try query.appendSlice(allocator, input_path);
    try query.append(allocator, '\'');

    // WHERE clause
    if (opts.filter) |filter| {
        try query.appendSlice(allocator, " WHERE ");
        try query.appendSlice(allocator, filter);
    }

    // LIMIT clause
    if (opts.limit) |limit| {
        try query.appendSlice(allocator, " LIMIT ");
        try std.fmt.format(query.writer(allocator), "{d}", .{limit});
    }

    return try query.toOwnedSlice(allocator);
}

/// Find rename for a column: "old:new,a:b" -> "old" returns "new"
fn findRename(rename_spec: []const u8, col_name: []const u8) ?[]const u8 {
    var pair_iter = std.mem.splitScalar(u8, rename_spec, ',');
    while (pair_iter.next()) |pair| {
        const trimmed = std.mem.trim(u8, pair, " ");
        if (std.mem.indexOf(u8, trimmed, ":")) |colon_pos| {
            const old_name = trimmed[0..colon_pos];
            const new_name = trimmed[colon_pos + 1 ..];
            if (std.mem.eql(u8, old_name, col_name)) {
                return new_name;
            }
        }
    }
    return null;
}

/// Detect file type from extension and magic bytes
fn detectFileType(path: []const u8, data: []const u8) FileType {
    // Check extension first
    if (std.mem.endsWith(u8, path, ".lance")) return .lance;
    if (std.mem.endsWith(u8, path, ".parquet")) return .parquet;
    if (std.mem.endsWith(u8, path, ".arrow") or std.mem.endsWith(u8, path, ".feather") or std.mem.endsWith(u8, path, ".arrows")) return .arrow;
    if (std.mem.endsWith(u8, path, ".avro")) return .avro;
    if (std.mem.endsWith(u8, path, ".orc")) return .orc;
    if (std.mem.endsWith(u8, path, ".xlsx")) return .xlsx;

    // Check magic bytes
    if (data.len >= 4) {
        if (std.mem.eql(u8, data[0..4], "PAR1")) return .parquet;
        if (std.mem.eql(u8, data[0..4], "ORC\x00") or std.mem.eql(u8, data[0..3], "ORC")) return .orc;
        if (std.mem.eql(u8, data[0..4], "Obj\x01")) return .avro;
    }
    if (data.len >= 6 and std.mem.eql(u8, data[0..6], "ARROW1")) return .arrow;

    // Check for Lance magic at end
    if (data.len >= 40 and std.mem.eql(u8, data[data.len - 4 ..], "LANC")) return .lance;

    return .unknown;
}

/// Execute SQL query on data
fn executeQuery(allocator: std.mem.Allocator, data: []const u8, sql: []const u8, file_type: FileType) !Result {
    // Tokenize
    var lex = lexer.Lexer.init(sql);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();

    // Execute based on file type
    switch (file_type) {
        .parquet => {
            var pq_table = try ParquetTable.init(allocator, data);
            defer pq_table.deinit();

            var exec = executor.Executor.initWithParquet(&pq_table, allocator);
            defer exec.deinit();

            return try exec.execute(&stmt.select, &[_]ast.Value{});
        },
        .lance => {
            var table = try Table.init(allocator, data);
            defer table.deinit();

            var exec = executor.Executor.init(&table, allocator);
            defer exec.deinit();

            return try exec.execute(&stmt.select, &[_]ast.Value{});
        },
        .arrow => {
            var arrow_table = try ArrowTable.init(allocator, data);
            defer arrow_table.deinit();

            var exec = executor.Executor.initWithArrow(&arrow_table, allocator);
            defer exec.deinit();

            return try exec.execute(&stmt.select, &[_]ast.Value{});
        },
        .avro => {
            var avro_table = try AvroTable.init(allocator, data);
            defer avro_table.deinit();

            var exec = executor.Executor.initWithAvro(&avro_table, allocator);
            defer exec.deinit();

            return try exec.execute(&stmt.select, &[_]ast.Value{});
        },
        .orc => {
            var orc_table = try OrcTable.init(allocator, data);
            defer orc_table.deinit();

            var exec = executor.Executor.initWithOrc(&orc_table, allocator);
            defer exec.deinit();

            return try exec.execute(&stmt.select, &[_]ast.Value{});
        },
        .xlsx => {
            var xlsx_table = try XlsxTable.init(allocator, data);
            defer xlsx_table.deinit();

            var exec = executor.Executor.initWithXlsx(&xlsx_table, allocator);
            defer exec.deinit();

            return try exec.execute(&stmt.select, &[_]ast.Value{});
        },
        .unknown => {
            // Try Lance first, then Parquet
            if (Table.init(allocator, data)) |table_result| {
                var table = table_result;
                defer table.deinit();
                var exec = executor.Executor.init(&table, allocator);
                defer exec.deinit();
                return try exec.execute(&stmt.select, &[_]ast.Value{});
            } else |_| {
                var pq_table = try ParquetTable.init(allocator, data);
                defer pq_table.deinit();
                var exec = executor.Executor.initWithParquet(&pq_table, allocator);
                defer exec.deinit();
                return try exec.execute(&stmt.select, &[_]ast.Value{});
            }
        },
    }
}

/// Map Result.ColumnData type to writer.DataType
fn columnDataToLanceType(data: Result.ColumnData) writer.DataType {
    return switch (data) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => .int64,
        .int32, .date32 => .int32,
        .float64 => .float64,
        .float32 => .float32,
        .bool_ => .bool,
        .string => .string,
    };
}

/// Write Result to Lance file
fn writeResultToLance(allocator: std.mem.Allocator, result: *Result, output_path: []const u8) !void {
    if (result.columns.len == 0) {
        std.debug.print("Warning: No columns in result, nothing to write\n", .{});
        return;
    }

    // Build Lance schema
    const schema = try allocator.alloc(writer.ColumnSchema, result.columns.len);
    defer allocator.free(schema);

    for (result.columns, 0..) |col, i| {
        schema[i] = .{
            .name = col.name,
            .data_type = columnDataToLanceType(col.data),
        };
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    // Encode each column
    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    for (result.columns, 0..) |col, i| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        var offsets_slice: ?[]const u8 = null;

        switch (col.data) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| {
                try encoder.writeInt64Slice(data);
            },
            .int32, .date32 => |data| {
                // Convert int32 to int64 for Lance
                const as_i64 = try allocator.alloc(i64, data.len);
                defer allocator.free(as_i64);
                for (data, 0..) |v, j| {
                    as_i64[j] = v;
                }
                try encoder.writeInt64Slice(as_i64);
            },
            .float64 => |data| {
                try encoder.writeFloat64Slice(data);
            },
            .float32 => |data| {
                // Convert float32 to float64 for Lance
                const as_f64 = try allocator.alloc(f64, data.len);
                defer allocator.free(as_f64);
                for (data, 0..) |v, j| {
                    as_f64[j] = v;
                }
                try encoder.writeFloat64Slice(as_f64);
            },
            .bool_ => |data| {
                try encoder.writeBools(data);
            },
            .string => |data| {
                try encoder.writeStrings(data, &offsets_buf, allocator);
                offsets_slice = offsets_buf.items;
            },
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(i),
            .data = encoder.getBytes(),
            .row_count = @intCast(col.data.len()),
            .offsets = offsets_slice,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    // Finalize and write file
    const lance_data = try lance_writer.finalize();

    // Write to output file
    const out_file = std.fs.cwd().createFile(output_path, .{}) catch |err| {
        std.debug.print("Error creating output file '{s}': {}\n", .{ output_path, err });
        return TransformError.WriteError;
    };
    defer out_file.close();

    out_file.writeAll(lance_data) catch |err| {
        std.debug.print("Error writing output file: {}\n", .{err});
        return TransformError.WriteError;
    };

    std.debug.print("Created: {s} ({d} rows, {d} columns, {d} bytes)\n", .{
        output_path,
        result.row_count,
        result.columns.len,
        lance_data.len,
    });
}

// =============================================================================
// Tests
// =============================================================================

test "build sql query - select only" {
    const allocator = std.testing.allocator;
    const opts = args.TransformOptions{
        .input = "test.parquet",
        .output = "out.lance",
        .select = "id,name",
        .filter = null,
        .rename = null,
        .cast = null,
        .limit = null,
        .help = false,
    };

    const sql = try buildSqlQuery(allocator, "test.parquet", opts);
    defer allocator.free(sql);

    try std.testing.expectEqualStrings("SELECT id,name FROM 'test.parquet'", sql);
}

test "build sql query - with filter and limit" {
    const allocator = std.testing.allocator;
    const opts = args.TransformOptions{
        .input = "test.parquet",
        .output = "out.lance",
        .select = null,
        .filter = "x > 100",
        .rename = null,
        .cast = null,
        .limit = 50,
        .help = false,
    };

    const sql = try buildSqlQuery(allocator, "test.parquet", opts);
    defer allocator.free(sql);

    try std.testing.expectEqualStrings("SELECT * FROM 'test.parquet' WHERE x > 100 LIMIT 50", sql);
}

test "find rename" {
    try std.testing.expectEqualStrings("new_name", findRename("old:new_name", "old").?);
    try std.testing.expectEqualStrings("b", findRename("a:b,c:d", "a").?);
    try std.testing.expectEqualStrings("d", findRename("a:b,c:d", "c").?);
    try std.testing.expect(findRename("a:b", "x") == null);
}
