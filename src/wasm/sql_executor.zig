//! WASM SQL Executor - Execute SQL queries entirely in WASM
//!
//! Architecture: "One In, One Out"
//! - Input: SQL string (via shared memory)
//! - Output: Packed binary result buffer
//!
//! This eliminates JS object creation and postMessage overhead.

const std = @import("std");
const memory = @import("memory.zig");
const format = @import("format.zig");
const column_meta = @import("column_meta.zig");
const aggregates = @import("aggregates.zig");

// ============================================================================
// Result Buffer Format
// ============================================================================
//
// The result buffer is a packed binary format:
//
// Header (fixed 32 bytes):
//   [0..4]   u32: version (1)
//   [4..8]   u32: column count
//   [8..16]  u64: row count
//   [16..20] u32: header size (32)
//   [20..24] u32: column metadata offset
//   [24..28] u32: data offset
//   [28..32] u32: flags (0=success, 1=error)
//
// Column Metadata (16 bytes per column):
//   [0..4]   u32: column type (0=int64, 1=float64, 2=int32, 3=float32, 4=string)
//   [4..8]   u32: column name offset (into string table)
//   [8..12]  u32: column name length
//   [12..16] u32: data offset (relative to data section start)
//
// Data Section:
//   - For numeric types: packed array of values
//   - For strings: offsets array + data buffer
//
// ============================================================================

const RESULT_VERSION: u32 = 1;
const HEADER_SIZE: u32 = 32;

/// Column types in result buffer
pub const ColumnType = enum(u32) {
    int64 = 0,
    float64 = 1,
    int32 = 2,
    float32 = 3,
    string = 4,
};

// ============================================================================
// In-Memory Table Storage
// ============================================================================

/// Maximum number of tables that can be registered
const MAX_TABLES: usize = 16;

/// Maximum columns per table
const MAX_COLUMNS: usize = 64;

/// Column data storage
pub const ColumnData = struct {
    name: []const u8,
    col_type: ColumnType,
    // Union for different column types
    data: union {
        int64: []const i64,
        float64: []const f64,
        int32: []const i32,
        float32: []const f32,
        // Strings: offsets into string_data
        strings: struct {
            offsets: []const u32, // Start offset for each string
            lengths: []const u32, // Length of each string
            data: []const u8, // Concatenated string data
        },
    },
    row_count: usize,
};

/// Table registration
pub const TableInfo = struct {
    name: []const u8,
    columns: [MAX_COLUMNS]?ColumnData,
    column_count: usize,
    row_count: usize,
};

/// Global table registry (WASM is single-threaded)
var tables: [MAX_TABLES]?TableInfo = .{null} ** MAX_TABLES;
var table_count: usize = 0;

// ============================================================================
// SQL Parsing (Minimal for WASM - subset of full parser)
// ============================================================================

/// Parsed SELECT query (simplified for WASM)
pub const ParsedQuery = struct {
    table_name: []const u8,
    columns: []const []const u8, // Column names or "*"
    is_star: bool,
    where_clause: ?WhereClause,
    order_by: ?OrderByClause,
    limit_value: ?u32,
    offset_value: ?u32,
    aggregates: []const AggregateExpr,
    group_by: ?[]const []const u8,
};

/// WHERE clause operators
pub const WhereOp = enum {
    eq, // =
    ne, // != or <>
    lt, // <
    le, // <=
    gt, // >
    ge, // >=
    between,
    in_list,
    like,
    is_null,
    is_not_null,
    and_op,
    or_op,
};

/// WHERE clause node
pub const WhereClause = struct {
    op: WhereOp,
    column: ?[]const u8,
    // For comparison ops
    value_int: ?i64,
    value_float: ?f64,
    value_string: ?[]const u8,
    // For BETWEEN
    low_int: ?i64,
    high_int: ?i64,
    low_float: ?f64,
    high_float: ?f64,
    // For AND/OR
    left: ?*const WhereClause,
    right: ?*const WhereClause,
};

/// ORDER BY clause
pub const OrderByClause = struct {
    column: []const u8,
    descending: bool,
};

/// Aggregate expression
pub const AggregateExpr = struct {
    func: enum { sum, min, max, avg, count },
    column: []const u8,
    alias: ?[]const u8,
};

// ============================================================================
// Result Buffer
// ============================================================================

/// Result buffer state
var result_buffer: ?[]u8 = null;
var result_size: usize = 0;

/// Allocate result buffer
fn allocResultBuffer(size: usize) ?[]u8 {
    const ptr = memory.wasmAlloc(size) orelse return null;
    result_buffer = ptr[0..size];
    result_size = 0;
    return result_buffer;
}

/// Write to result buffer
fn writeToResult(data: []const u8) bool {
    const buf = result_buffer orelse return false;
    if (result_size + data.len > buf.len) return false;
    @memcpy(buf[result_size..][0..data.len], data);
    result_size += data.len;
    return true;
}

/// Write u32 to result
fn writeU32(value: u32) bool {
    const bytes = std.mem.asBytes(&value);
    return writeToResult(bytes);
}

/// Write u64 to result
fn writeU64(value: u64) bool {
    const bytes = std.mem.asBytes(&value);
    return writeToResult(bytes);
}

// ============================================================================
// Query Execution
// ============================================================================

/// Execute a simple query and return result buffer
fn executeQuery(table: *const TableInfo, query: *const ParsedQuery) !void {
    const row_count = table.row_count;

    // Step 1: Apply WHERE filter to get matching row indices
    var match_indices: [65536]u32 = undefined;
    var match_count: usize = 0;

    if (query.where_clause) |*where| {
        // Evaluate WHERE clause for each row
        for (0..row_count) |i| {
            if (evaluateWhere(table, where, @intCast(i))) {
                if (match_count < match_indices.len) {
                    match_indices[match_count] = @intCast(i);
                    match_count += 1;
                }
            }
        }
    } else {
        // No WHERE clause - all rows match
        for (0..@min(row_count, match_indices.len)) |i| {
            match_indices[i] = @intCast(i);
        }
        match_count = @min(row_count, match_indices.len);
    }

    // Step 2: Apply LIMIT/OFFSET
    var start: usize = 0;
    var end: usize = match_count;

    if (query.offset_value) |offset| {
        start = @min(offset, match_count);
    }
    if (query.limit_value) |limit| {
        end = @min(start + limit, match_count);
    }

    const final_count = end - start;

    // Step 3: Determine columns to output
    var output_columns: [MAX_COLUMNS]usize = undefined;
    var output_col_count: usize = 0;

    if (query.is_star) {
        // SELECT * - output all columns
        for (0..table.column_count) |i| {
            output_columns[output_col_count] = i;
            output_col_count += 1;
        }
    } else {
        // Named columns
        for (query.columns) |col_name| {
            for (table.columns[0..table.column_count], 0..) |maybe_col, i| {
                if (maybe_col) |col| {
                    if (std.mem.eql(u8, col.name, col_name)) {
                        output_columns[output_col_count] = i;
                        output_col_count += 1;
                        break;
                    }
                }
            }
        }
    }

    // Step 4: Calculate result buffer size
    var data_size: usize = 0;
    for (output_columns[0..output_col_count]) |col_idx| {
        if (table.columns[col_idx]) |col| {
            data_size += switch (col.col_type) {
                .int64, .float64 => final_count * 8,
                .int32, .float32 => final_count * 4,
                .string => final_count * 8 + 65536, // offsets + data estimate
            };
        }
    }

    const meta_size = output_col_count * 16;
    const total_size = HEADER_SIZE + meta_size + data_size;

    // Step 5: Allocate and write result
    _ = allocResultBuffer(total_size) orelse return error.OutOfMemory;

    // Write header
    _ = writeU32(RESULT_VERSION);
    _ = writeU32(@intCast(output_col_count));
    _ = writeU64(@intCast(final_count));
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE); // column metadata offset
    _ = writeU32(HEADER_SIZE + @as(u32, @intCast(meta_size))); // data offset
    _ = writeU32(0); // flags = success

    // Write column metadata
    var data_offset: u32 = 0;
    for (output_columns[0..output_col_count]) |col_idx| {
        if (table.columns[col_idx]) |col| {
            _ = writeU32(@intFromEnum(col.col_type));
            _ = writeU32(0); // name offset (TODO: string table)
            _ = writeU32(@intCast(col.name.len));
            _ = writeU32(data_offset);

            // Update data offset for next column
            data_offset += switch (col.col_type) {
                .int64, .float64 => @intCast(final_count * 8),
                .int32, .float32 => @intCast(final_count * 4),
                .string => @intCast(final_count * 8 + 65536),
            };
        }
    }

    // Write column data
    for (output_columns[0..output_col_count]) |col_idx| {
        if (table.columns[col_idx]) |col| {
            switch (col.col_type) {
                .int64 => {
                    for (match_indices[start..end]) |row_idx| {
                        const val = col.data.int64[row_idx];
                        _ = writeToResult(std.mem.asBytes(&val));
                    }
                },
                .float64 => {
                    for (match_indices[start..end]) |row_idx| {
                        const val = col.data.float64[row_idx];
                        _ = writeToResult(std.mem.asBytes(&val));
                    }
                },
                .int32 => {
                    for (match_indices[start..end]) |row_idx| {
                        const val = col.data.int32[row_idx];
                        _ = writeToResult(std.mem.asBytes(&val));
                    }
                },
                .float32 => {
                    for (match_indices[start..end]) |row_idx| {
                        const val = col.data.float32[row_idx];
                        _ = writeToResult(std.mem.asBytes(&val));
                    }
                },
                .string => {
                    // Write string offsets first, then data
                    var str_offset: u32 = 0;
                    for (match_indices[start..end]) |row_idx| {
                        _ = writeU32(str_offset);
                        _ = writeU32(col.data.strings.lengths[row_idx]);
                        str_offset += col.data.strings.lengths[row_idx];
                    }
                    // Write string data
                    for (match_indices[start..end]) |row_idx| {
                        const offset = col.data.strings.offsets[row_idx];
                        const len = col.data.strings.lengths[row_idx];
                        _ = writeToResult(col.data.strings.data[offset..][0..len]);
                    }
                },
            }
        }
    }
}

/// Evaluate WHERE clause for a row
fn evaluateWhere(table: *const TableInfo, where: *const WhereClause, row_idx: u32) bool {
    switch (where.op) {
        .and_op => {
            if (where.left) |left| {
                if (where.right) |right| {
                    return evaluateWhere(table, left, row_idx) and evaluateWhere(table, right, row_idx);
                }
            }
            return false;
        },
        .or_op => {
            if (where.left) |left| {
                if (where.right) |right| {
                    return evaluateWhere(table, left, row_idx) or evaluateWhere(table, right, row_idx);
                }
            }
            return false;
        },
        .eq, .ne, .lt, .le, .gt, .ge => {
            const col_name = where.column orelse return false;

            // Find column
            for (table.columns[0..table.column_count]) |maybe_col| {
                if (maybe_col) |col| {
                    if (std.mem.eql(u8, col.name, col_name)) {
                        return evaluateComparison(col, row_idx, where);
                    }
                }
            }
            return false;
        },
        .between => {
            const col_name = where.column orelse return false;

            for (table.columns[0..table.column_count]) |maybe_col| {
                if (maybe_col) |col| {
                    if (std.mem.eql(u8, col.name, col_name)) {
                        return evaluateBetween(col, row_idx, where);
                    }
                }
            }
            return false;
        },
        .is_null, .is_not_null, .in_list, .like => {
            // TODO: Implement these
            return true;
        },
    }
}

fn evaluateComparison(col: ColumnData, row_idx: u32, where: *const WhereClause) bool {
    switch (col.col_type) {
        .int64 => {
            const val = col.data.int64[row_idx];
            const cmp = where.value_int orelse return false;
            return switch (where.op) {
                .eq => val == cmp,
                .ne => val != cmp,
                .lt => val < cmp,
                .le => val <= cmp,
                .gt => val > cmp,
                .ge => val >= cmp,
                else => false,
            };
        },
        .float64 => {
            const val = col.data.float64[row_idx];
            const cmp = where.value_float orelse return false;
            return switch (where.op) {
                .eq => val == cmp,
                .ne => val != cmp,
                .lt => val < cmp,
                .le => val <= cmp,
                .gt => val > cmp,
                .ge => val >= cmp,
                else => false,
            };
        },
        .int32 => {
            const val = col.data.int32[row_idx];
            const cmp: i32 = @intCast(where.value_int orelse return false);
            return switch (where.op) {
                .eq => val == cmp,
                .ne => val != cmp,
                .lt => val < cmp,
                .le => val <= cmp,
                .gt => val > cmp,
                .ge => val >= cmp,
                else => false,
            };
        },
        .float32 => {
            const val = col.data.float32[row_idx];
            const cmp: f32 = @floatCast(where.value_float orelse return false);
            return switch (where.op) {
                .eq => val == cmp,
                .ne => val != cmp,
                .lt => val < cmp,
                .le => val <= cmp,
                .gt => val > cmp,
                .ge => val >= cmp,
                else => false,
            };
        },
        .string => {
            // TODO: String comparison
            return true;
        },
    }
}

fn evaluateBetween(col: ColumnData, row_idx: u32, where: *const WhereClause) bool {
    switch (col.col_type) {
        .int64 => {
            const val = col.data.int64[row_idx];
            const low = where.low_int orelse return false;
            const high = where.high_int orelse return false;
            return val >= low and val <= high;
        },
        .float64 => {
            const val = col.data.float64[row_idx];
            const low = where.low_float orelse return false;
            const high = where.high_float orelse return false;
            return val >= low and val <= high;
        },
        else => return true,
    }
}

// ============================================================================
// WASM Exports
// ============================================================================

/// SQL input buffer
var sql_input: [4096]u8 = undefined;
var sql_input_len: usize = 0;

/// Get SQL input buffer for JS to write to
pub export fn getSqlInputBuffer() [*]u8 {
    return &sql_input;
}

/// Get SQL input buffer size
pub export fn getSqlInputBufferSize() usize {
    return sql_input.len;
}

/// Set SQL input length (called after JS writes SQL)
pub export fn setSqlInputLength(len: usize) void {
    sql_input_len = @min(len, sql_input.len);
}

/// Register a table with int64 column
pub export fn registerTableInt64(
    table_name_ptr: [*]const u8,
    table_name_len: usize,
    col_name_ptr: [*]const u8,
    col_name_len: usize,
    data_ptr: [*]const i64,
    row_count: usize,
) u32 {
    const table_name = table_name_ptr[0..table_name_len];
    const col_name = col_name_ptr[0..col_name_len];

    // Find or create table
    var table_idx: ?usize = null;
    for (tables[0..table_count], 0..) |maybe_table, i| {
        if (maybe_table) |t| {
            if (std.mem.eql(u8, t.name, table_name)) {
                table_idx = i;
                break;
            }
        }
    }

    if (table_idx == null) {
        if (table_count >= MAX_TABLES) return 1;
        tables[table_count] = TableInfo{
            .name = table_name,
            .columns = .{null} ** MAX_COLUMNS,
            .column_count = 0,
            .row_count = row_count,
        };
        table_idx = table_count;
        table_count += 1;
    }

    const idx = table_idx.?;
    var table = &(tables[idx].?);

    // Add column
    if (table.column_count >= MAX_COLUMNS) return 2;

    table.columns[table.column_count] = ColumnData{
        .name = col_name,
        .col_type = .int64,
        .data = .{ .int64 = data_ptr[0..row_count] },
        .row_count = row_count,
    };
    table.column_count += 1;

    return 0;
}

/// Register a table with float64 column
pub export fn registerTableFloat64(
    table_name_ptr: [*]const u8,
    table_name_len: usize,
    col_name_ptr: [*]const u8,
    col_name_len: usize,
    data_ptr: [*]const f64,
    row_count: usize,
) u32 {
    const table_name = table_name_ptr[0..table_name_len];
    const col_name = col_name_ptr[0..col_name_len];

    // Find or create table
    var table_idx: ?usize = null;
    for (tables[0..table_count], 0..) |maybe_table, i| {
        if (maybe_table) |t| {
            if (std.mem.eql(u8, t.name, table_name)) {
                table_idx = i;
                break;
            }
        }
    }

    if (table_idx == null) {
        if (table_count >= MAX_TABLES) return 1;
        tables[table_count] = TableInfo{
            .name = table_name,
            .columns = .{null} ** MAX_COLUMNS,
            .column_count = 0,
            .row_count = row_count,
        };
        table_idx = table_count;
        table_count += 1;
    }

    const idx = table_idx.?;
    var table = &(tables[idx].?);

    // Add column
    if (table.column_count >= MAX_COLUMNS) return 2;

    table.columns[table.column_count] = ColumnData{
        .name = col_name,
        .col_type = .float64,
        .data = .{ .float64 = data_ptr[0..row_count] },
        .row_count = row_count,
    };
    table.column_count += 1;

    return 0;
}

/// Register string column
pub export fn registerTableString(
    table_name_ptr: [*]const u8,
    table_name_len: usize,
    col_name_ptr: [*]const u8,
    col_name_len: usize,
    offsets_ptr: [*]const u32,
    lengths_ptr: [*]const u32,
    data_ptr: [*]const u8,
    data_len: usize,
    row_count: usize,
) u32 {
    const table_name = table_name_ptr[0..table_name_len];
    const col_name = col_name_ptr[0..col_name_len];

    // Find or create table
    var table_idx: ?usize = null;
    for (tables[0..table_count], 0..) |maybe_table, i| {
        if (maybe_table) |t| {
            if (std.mem.eql(u8, t.name, table_name)) {
                table_idx = i;
                break;
            }
        }
    }

    if (table_idx == null) {
        if (table_count >= MAX_TABLES) return 1;
        tables[table_count] = TableInfo{
            .name = table_name,
            .columns = .{null} ** MAX_COLUMNS,
            .column_count = 0,
            .row_count = row_count,
        };
        table_idx = table_count;
        table_count += 1;
    }

    const idx = table_idx.?;
    var table = &(tables[idx].?);

    // Add column
    if (table.column_count >= MAX_COLUMNS) return 2;

    table.columns[table.column_count] = ColumnData{
        .name = col_name,
        .col_type = .string,
        .data = .{
            .strings = .{
                .offsets = offsets_ptr[0..row_count],
                .lengths = lengths_ptr[0..row_count],
                .data = data_ptr[0..data_len],
            },
        },
        .row_count = row_count,
    };
    table.column_count += 1;

    return 0;
}

/// Clear all registered tables
pub export fn clearTables() void {
    tables = .{null} ** MAX_TABLES;
    table_count = 0;
}

/// Execute SQL and return result buffer pointer
/// Returns 0 on error
pub export fn executeSql() usize {
    const sql = sql_input[0..sql_input_len];

    // Parse SQL (simplified - just extract table name for SELECT * FROM table)
    var query = parseSimpleSql(sql) orelse return 0;

    // Find table
    var found_table: ?*const TableInfo = null;
    for (tables[0..table_count]) |maybe_table| {
        if (maybe_table) |*t| {
            if (std.mem.eql(u8, t.name, query.table_name)) {
                found_table = t;
                break;
            }
        }
    }

    const table = found_table orelse return 0;

    // Execute query
    executeQuery(table, &query) catch return 0;

    // Return result buffer address
    if (result_buffer) |buf| {
        return @intFromPtr(buf.ptr);
    }
    return 0;
}

/// Get result buffer size
pub export fn getResultSize() usize {
    return result_size;
}

/// Reset result buffer for next query
pub export fn resetResult() void {
    memory.wasmReset();
    result_buffer = null;
    result_size = 0;
}

// ============================================================================
// Simple SQL Parser (for common patterns)
// ============================================================================

/// Parse simple SELECT queries
/// Supports: SELECT * FROM table [WHERE col op value] [LIMIT n]
fn parseSimpleSql(sql: []const u8) ?ParsedQuery {
    var result = ParsedQuery{
        .table_name = "",
        .columns = &.{},
        .is_star = false,
        .where_clause = null,
        .order_by = null,
        .limit_value = null,
        .offset_value = null,
        .aggregates = &.{},
        .group_by = null,
    };

    // Skip whitespace and find SELECT
    var pos: usize = 0;
    pos = skipWhitespace(sql, pos);

    if (!startsWithIgnoreCase(sql[pos..], "SELECT")) return null;
    pos += 6;
    pos = skipWhitespace(sql, pos);

    // Check for * or column list
    if (sql.len > pos and sql[pos] == '*') {
        result.is_star = true;
        pos += 1;
    } else {
        // TODO: Parse column list
        return null;
    }

    pos = skipWhitespace(sql, pos);

    // FROM clause
    if (!startsWithIgnoreCase(sql[pos..], "FROM")) return null;
    pos += 4;
    pos = skipWhitespace(sql, pos);

    // Table name
    const table_start = pos;
    while (pos < sql.len and isIdentChar(sql[pos])) {
        pos += 1;
    }
    if (pos == table_start) return null;
    result.table_name = sql[table_start..pos];

    pos = skipWhitespace(sql, pos);

    // Optional WHERE clause
    if (startsWithIgnoreCase(sql[pos..], "WHERE")) {
        pos += 5;
        pos = skipWhitespace(sql, pos);

        // Parse simple comparison: column op value
        result.where_clause = parseWhereClause(sql, &pos);
    }

    // Optional LIMIT clause
    if (startsWithIgnoreCase(sql[pos..], "LIMIT")) {
        pos += 5;
        pos = skipWhitespace(sql, pos);

        // Parse number
        const num_start = pos;
        while (pos < sql.len and std.ascii.isDigit(sql[pos])) {
            pos += 1;
        }
        if (pos > num_start) {
            result.limit_value = std.fmt.parseInt(u32, sql[num_start..pos], 10) catch null;
        }
    }

    return result;
}

fn parseWhereClause(sql: []const u8, pos: *usize) ?WhereClause {
    // Skip whitespace
    pos.* = skipWhitespace(sql, pos.*);

    // Column name
    const col_start = pos.*;
    while (pos.* < sql.len and isIdentChar(sql[pos.*])) {
        pos.* += 1;
    }
    if (pos.* == col_start) return null;
    const column = sql[col_start..pos.*];

    pos.* = skipWhitespace(sql, pos.*);

    // Operator
    var op: WhereOp = .eq;
    if (pos.* < sql.len) {
        if (sql[pos.*] == '=') {
            op = .eq;
            pos.* += 1;
        } else if (sql[pos.*] == '<') {
            pos.* += 1;
            if (pos.* < sql.len and sql[pos.*] == '=') {
                op = .le;
                pos.* += 1;
            } else if (pos.* < sql.len and sql[pos.*] == '>') {
                op = .ne;
                pos.* += 1;
            } else {
                op = .lt;
            }
        } else if (sql[pos.*] == '>') {
            pos.* += 1;
            if (pos.* < sql.len and sql[pos.*] == '=') {
                op = .ge;
                pos.* += 1;
            } else {
                op = .gt;
            }
        } else if (sql[pos.*] == '!' and pos.* + 1 < sql.len and sql[pos.* + 1] == '=') {
            op = .ne;
            pos.* += 2;
        } else {
            return null;
        }
    }

    pos.* = skipWhitespace(sql, pos.*);

    // Value (number or string)
    var value_int: ?i64 = null;
    var value_float: ?f64 = null;

    if (pos.* < sql.len and (std.ascii.isDigit(sql[pos.*]) or sql[pos.*] == '-')) {
        const num_start = pos.*;
        if (sql[pos.*] == '-') pos.* += 1;

        while (pos.* < sql.len and (std.ascii.isDigit(sql[pos.*]) or sql[pos.*] == '.')) {
            pos.* += 1;
        }

        const num_str = sql[num_start..pos.*];
        if (std.mem.indexOf(u8, num_str, ".") != null) {
            value_float = std.fmt.parseFloat(f64, num_str) catch null;
        } else {
            value_int = std.fmt.parseInt(i64, num_str, 10) catch null;
        }
    }

    return WhereClause{
        .op = op,
        .column = column,
        .value_int = value_int,
        .value_float = value_float,
        .value_string = null,
        .low_int = null,
        .high_int = null,
        .low_float = null,
        .high_float = null,
        .left = null,
        .right = null,
    };
}

fn skipWhitespace(sql: []const u8, start: usize) usize {
    var pos = start;
    while (pos < sql.len and std.ascii.isWhitespace(sql[pos])) {
        pos += 1;
    }
    return pos;
}

fn isIdentChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn startsWithIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (haystack.len < needle.len) return false;
    for (haystack[0..needle.len], needle) |h, n| {
        if (std.ascii.toLower(h) != std.ascii.toLower(n)) return false;
    }
    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "parse simple SELECT" {
    const sql = "SELECT * FROM users";
    const query = parseSimpleSql(sql);

    try std.testing.expect(query != null);
    try std.testing.expect(query.?.is_star);
    try std.testing.expectEqualStrings("users", query.?.table_name);
}

test "parse SELECT with WHERE" {
    const sql = "SELECT * FROM orders WHERE id = 42";
    const query = parseSimpleSql(sql);

    try std.testing.expect(query != null);
    try std.testing.expect(query.?.where_clause != null);
    try std.testing.expectEqualStrings("id", query.?.where_clause.?.column.?);
    try std.testing.expectEqual(WhereOp.eq, query.?.where_clause.?.op);
    try std.testing.expectEqual(@as(i64, 42), query.?.where_clause.?.value_int.?);
}

test "parse SELECT with LIMIT" {
    const sql = "SELECT * FROM products LIMIT 100";
    const query = parseSimpleSql(sql);

    try std.testing.expect(query != null);
    try std.testing.expectEqual(@as(u32, 100), query.?.limit_value.?);
}
