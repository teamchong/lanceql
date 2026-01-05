//! WASM SQL Executor - Execute SQL queries entirely in WASM
//!
//! Architecture: "One In, One Out"
//! - Input: SQL string (via shared memory)
//! - Output: Packed binary result buffer
//!
//! Supports:
//! - SELECT with specific columns or *
//! - WHERE with AND/OR and comparison operators
//! - GROUP BY with aggregates (SUM, COUNT, AVG, MIN, MAX)
//! - ORDER BY (ASC/DESC)
//! - LIMIT/OFFSET

const std = @import("std");
const memory = @import("memory.zig");
const fragment_reader = @import("fragment_reader.zig");
const aggregates = @import("aggregates.zig");
const simd_search = @import("simd_search.zig");

const RESULT_VERSION: u32 = 1;
const HEADER_SIZE: u32 = 32;
const MAX_TABLES: usize = 16;
const MAX_COLUMNS: usize = 64;
const MAX_FRAGMENTS: usize = 16;
const MAX_SELECT_COLS: usize = 32;
const MAX_GROUP_COLS: usize = 8;
const MAX_AGGREGATES: usize = 16;
const MAX_ROWS: usize = 100000;
const MAX_VECTOR_DIM: usize = 1536; // Support up to OpenAI embedding size

/// Column types
pub const ColumnType = enum(u32) {
    int64 = 0,
    float64 = 1,
    int32 = 2,
    float32 = 3,
    string = 4,
};

/// Column data storage
pub const ColumnData = struct {
    name: []const u8,
    col_type: ColumnType,
    data: union {
        int64: []const i64,
        float64: []const f64,
        int32: []const i32,
        float32: []const f32,
        strings: struct {
            offsets: []const u32,
            lengths: []const u32,
            data: []const u8,
        },
        none: void, // For lazy columns
    },
    row_count: usize,
    is_lazy: bool = false,
    fragment_col_idx: u32 = 0,
    vector_dim: u32 = 0, // Dimension for vector columns
};

/// Table registration
pub const TableInfo = struct {
    name: []const u8,
    columns: [MAX_COLUMNS]?ColumnData,
    column_count: usize,
    row_count: usize,
    
    // Fragment support
    fragments: [MAX_FRAGMENTS]?fragment_reader.FragmentReader,
    fragment_count: usize,
};

/// WHERE operators
pub const WhereOp = enum { eq, ne, lt, le, gt, ge, and_op, or_op, in_list, not_in_list, in_subquery, exists, not_exists, like, not_like, between, is_null, is_not_null, near };

/// WHERE clause node
pub const WhereClause = struct {
    op: WhereOp,
    column: ?[]const u8 = null,
    value_int: ?i64 = null,
    value_float: ?f64 = null,
    value_str: ?[]const u8 = null,
    left: ?*const WhereClause = null,
    right: ?*const WhereClause = null,
    // For IN lists
    in_values_int: [32]i64 = undefined,
    in_values_count: usize = 0,
    // For subqueries
    subquery_start: usize = 0,
    subquery_len: usize = 0,
    // For BETWEEN
    between_low: ?i64 = null,
    between_high: ?i64 = null,
    // For NEAR (vector search)
    near_vector: [MAX_VECTOR_DIM]f32 = undefined,
    near_dim: usize = 0,
};

/// Aggregate function
pub const AggFunc = enum { sum, count, avg, min, max };

/// Aggregate expression
pub const AggExpr = struct {
    func: AggFunc,
    column: []const u8,
    alias: ?[]const u8 = null,
};

/// ORDER BY direction
pub const OrderDir = enum { asc, desc };

/// JOIN types
pub const JoinType = enum { inner, left, right, cross };

/// JOIN clause
pub const JoinClause = struct {
    table_name: []const u8,
    join_type: JoinType = .inner,
    left_col: []const u8 = "",
    right_col: []const u8 = "",
};

const MAX_JOINS: usize = 4;

/// Set operation type
pub const SetOpType = enum { none, union_all, union_distinct, intersect, intersect_all, except, except_all };

/// Window function types
pub const WindowFunc = enum { row_number, rank, dense_rank, sum, count, avg, min, max, lag, lead, ntile, percent_rank, cume_dist, first_value, last_value };

/// Window function expression
pub const WindowExpr = struct {
    func: WindowFunc,
    arg_col: ?[]const u8 = null, // Column argument (for SUM, AVG, etc)
    ntile_n: u32 = 1, // For NTILE(n)
    partition_by: [4][]const u8 = undefined,
    partition_count: usize = 0,
    order_by_col: ?[]const u8 = null,
    order_dir: OrderDir = .asc,
    alias: ?[]const u8 = null,
};

const MAX_WINDOW_FUNCS: usize = 8;

/// CTE definition
pub const CTEDef = struct {
    name: []const u8,
    query_start: usize, // Position in sql_input where CTE query starts
    query_end: usize, // Position where CTE query ends
};

const MAX_CTES: usize = 4;

/// Parsed query
pub const ParsedQuery = struct {
    table_name: []const u8 = "",
    select_cols: [MAX_SELECT_COLS][]const u8 = undefined,
    select_count: usize = 0,
    is_star: bool = false,
    is_distinct: bool = false,
    aggregates: [MAX_AGGREGATES]AggExpr = undefined,
    agg_count: usize = 0,
    where_clause: ?WhereClause = null,
    group_by_cols: [MAX_GROUP_COLS][]const u8 = undefined,
    group_by_count: usize = 0,
    order_by_col: ?[]const u8 = null,
    order_dir: OrderDir = .asc,
    limit_value: ?u32 = null,
    offset_value: ?u32 = null,
    joins: [MAX_JOINS]JoinClause = undefined,
    join_count: usize = 0,
    set_op: SetOpType = .none,
    set_op_query_start: usize = 0, // Index into sql_input where second query starts
    window_funcs: [MAX_WINDOW_FUNCS]WindowExpr = undefined,
    window_count: usize = 0,
    ctes: [MAX_CTES]CTEDef = undefined,
    cte_count: usize = 0,
};

// Global state
var tables: [MAX_TABLES]?TableInfo = .{null} ** MAX_TABLES;
var table_count: usize = 0;
var result_buffer: ?[]u8 = null;
var result_size: usize = 0;
var sql_input: [8192]u8 = undefined;
var sql_input_len: usize = 0;

// Static storage for parsed WHERE clauses (avoid dynamic allocation)
var where_storage: [32]WhereClause = undefined;
var where_storage_idx: usize = 0;

// ============================================================================
// WASM Exports
// ============================================================================

pub export fn getSqlInputBuffer() [*]u8 {
    return &sql_input;
}

pub export fn getSqlInputBufferSize() usize {
    return sql_input.len;
}

pub export fn setSqlInputLength(len: usize) void {
    sql_input_len = @min(len, sql_input.len);
}

pub export fn registerTableInt64(
    table_name_ptr: [*]const u8,
    table_name_len: usize,
    col_name_ptr: [*]const u8,
    col_name_len: usize,
    data_ptr: [*]const i64,
    row_count: usize,
) u32 {
    return registerColumnInt64(
        table_name_ptr[0..table_name_len],
        col_name_ptr[0..col_name_len],
        data_ptr[0..row_count],
        row_count,
    );
}

pub export fn registerTableFloat64(
    table_name_ptr: [*]const u8,
    table_name_len: usize,
    col_name_ptr: [*]const u8,
    col_name_len: usize,
    data_ptr: [*]const f64,
    row_count: usize,
) u32 {
    return registerColumnFloat64(
        table_name_ptr[0..table_name_len],
        col_name_ptr[0..col_name_len],
        data_ptr[0..row_count],
        row_count,
    );
}

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
    return registerColumnString(
        table_name_ptr[0..table_name_len],
        col_name_ptr[0..col_name_len],
        offsets_ptr[0..row_count],
        lengths_ptr[0..row_count],
        data_ptr[0..data_len],
        row_count,
    );
}

pub export fn registerTableFragment(
    table_name_ptr: [*]const u8,
    table_name_len: usize,
    data_ptr: [*]const u8,
    data_len: usize,
) u32 {
    const table_name = table_name_ptr[0..table_name_len];
    
    // Parse fragment
    const reader = fragment_reader.FragmentReader.init(data_ptr, data_len) catch return 1;
    const row_count = reader.getRowCount();

    const tbl = findOrCreateTable(table_name, @intCast(row_count)) orelse return 2;
    if (tbl.fragment_count >= MAX_FRAGMENTS) return 3;

    tbl.fragments[tbl.fragment_count] = reader;
    tbl.fragment_count += 1;
    
    // Register columns from fragment if not already present
    const num_cols = reader.getColumnCount();
    var name_buf: [64]u8 = undefined;
    var type_buf: [16]u8 = undefined;

    for (0..num_cols) |i| {
        const i_u32: u32 = @intCast(i);
        const name_len = reader.fragmentGetColumnName(i_u32, &name_buf, 64);
        const name = name_buf[0..name_len];
        
        // Check if column exists
        var exists = false;
        for (tbl.columns[0..tbl.column_count]) |maybe_col| {
            if (maybe_col) |c| {
                if (std.mem.eql(u8, c.name, name)) {
                    exists = true;
                    break;
                }
            }
        }

        if (!exists and tbl.column_count < MAX_COLUMNS) {
            // Determine type
            const type_len = reader.fragmentGetColumnType(i_u32, &type_buf, 16);
            const type_str = type_buf[0..type_len];
            var col_type: ColumnType = .string;
            
            if (std.mem.eql(u8, type_str, "int64") or std.mem.eql(u8, type_str, "bigint")) {
                col_type = .int64;
            } else if (std.mem.eql(u8, type_str, "float64") or std.mem.eql(u8, type_str, "double")) {
                col_type = .float64;
            } else if (std.mem.eql(u8, type_str, "int32") or std.mem.eql(u8, type_str, "int")) {
                col_type = .int32;
            } else if (std.mem.eql(u8, type_str, "float32") or std.mem.eql(u8, type_str, "float")) {
                col_type = .float32;
            }

            // Check for vector dimension (from reader)
            const dim = reader.fragmentGetColumnVectorDim(i_u32);

            // Allocate name (needs to persist)
            const name_ptr = memory.wasmAlloc(name.len) orelse return 4;
            @memcpy(name_ptr[0..name.len], name);

            tbl.columns[tbl.column_count] = ColumnData{
                .name = name_ptr[0..name.len],
                .col_type = col_type,
                .data = .{ .none = {} },
                .row_count = 0, // Ignored for lazy
                .is_lazy = true,
                .fragment_col_idx = i_u32,
                .vector_dim = dim,
            };
            tbl.column_count += 1;
        }
    }

    // Update total row count (if multiple fragments, we sum them up)
    // Note: findOrCreateTable sets row_count for new tables.
    // For existing, we should add.
    if (tbl.fragment_count > 1) {
        tbl.row_count += @intCast(row_count);
    } else {
        // First fragment or mixed mode
        if (tbl.column_count == 0) {
            tbl.row_count = @intCast(row_count);
        }
    }
    
    return 0;
}

pub export fn clearTables() void {
    tables = .{null} ** MAX_TABLES;
    table_count = 0;
}

pub export fn clearTable(name_ptr: [*]const u8, name_len: usize) void {
    const name = name_ptr[0..name_len];
    for (0..table_count) |i| {
        if (tables[i]) |tbl| {
            if (std.mem.eql(u8, tbl.name, name)) {
                // Shift remaining tables
                for (i..table_count - 1) |j| {
                    tables[j] = tables[j + 1];
                }
                tables[table_count - 1] = null;
                table_count -= 1;
                return;
            }
        }
    }
}

pub export fn executeSql() usize {
    where_storage_idx = 0;
    const sql = sql_input[0..sql_input_len];

    var query = parseSql(sql) orelse return 0;

    // Handle set operations (UNION, INTERSECT, EXCEPT)
    if (query.set_op != .none) {
        executeSetOpQuery(sql, &query) catch return 0;
        if (result_buffer) |buf| {
            return @intFromPtr(buf.ptr);
        }
        return 0;
    }

    // Execute CTEs if present and check if table is a CTE
    if (query.cte_count > 0) {
        // Check if the main table is a CTE
        for (query.ctes[0..query.cte_count]) |cte| {
            if (std.mem.eql(u8, cte.name, query.table_name)) {
                // Execute CTE query
                executeCTEQuery(sql, &query, &cte) catch return 0;
                if (result_buffer) |buf| {
                    return @intFromPtr(buf.ptr);
                }
                return 0;
            }
        }
    }

    // Find primary table
    var table: ?*const TableInfo = null;
    for (&tables) |*t| {
        if (t.*) |*tbl| {
            if (std.mem.eql(u8, tbl.name, query.table_name)) {
                table = tbl;
                break;
            }
        }
    }

    const tbl = table orelse return 0;

    // Execute query based on type
    if (query.join_count > 0) {
        // JOIN query
        executeJoinQuery(tbl, &query) catch return 0;
    } else if (query.window_count > 0) {
        // Window function query
        executeWindowQuery(tbl, &query) catch return 0;
    } else if (query.agg_count > 0 or query.group_by_count > 0) {
        executeAggregateQuery(tbl, &query) catch return 0;
    } else {
        executeSelectQuery(tbl, &query) catch return 0;
    }

    if (result_buffer) |buf| {
        return @intFromPtr(buf.ptr);
    }
    return 0;
}

pub export fn getResultSize() usize {
    return result_size;
}

pub export fn resetResult() void {
    memory.wasmReset();
    result_buffer = null;
    result_size = 0;
}

// ============================================================================
// Column Registration
// ============================================================================

fn findOrCreateTable(table_name: []const u8, row_count: usize) ?*TableInfo {
    // Find existing table
    for (tables[0..table_count]) |*maybe_table| {
        if (maybe_table.*) |*t| {
            if (std.mem.eql(u8, t.name, table_name)) {
                return t;
            }
        }
    }

    // Create new table
    if (table_count >= MAX_TABLES) return null;
    tables[table_count] = TableInfo{
        .name = table_name,
        .columns = .{null} ** MAX_COLUMNS,
        .column_count = 0,
        .fragments = .{null} ** MAX_FRAGMENTS,
        .fragment_count = 0,
        .row_count = row_count,
    };
    const idx = table_count;
    table_count += 1;
    return &(tables[idx].?);
}

fn registerColumnInt64(table_name: []const u8, col_name: []const u8, data: []const i64, row_count: usize) u32 {
    const tbl = findOrCreateTable(table_name, row_count) orelse return 1;
    if (tbl.column_count >= MAX_COLUMNS) return 2;
    tbl.columns[tbl.column_count] = ColumnData{
        .name = col_name,
        .col_type = .int64,
        .data = .{ .int64 = data },
        .row_count = row_count,
    };
    tbl.column_count += 1;
    return 0;
}

fn registerColumnFloat64(table_name: []const u8, col_name: []const u8, data: []const f64, row_count: usize) u32 {
    const tbl = findOrCreateTable(table_name, row_count) orelse return 1;
    if (tbl.column_count >= MAX_COLUMNS) return 2;
    tbl.columns[tbl.column_count] = ColumnData{
        .name = col_name,
        .col_type = .float64,
        .data = .{ .float64 = data },
        .row_count = row_count,
    };
    tbl.column_count += 1;
    return 0;
}

fn registerColumnString(table_name: []const u8, col_name: []const u8, offsets: []const u32, lengths: []const u32, data: []const u8, row_count: usize) u32 {
    const tbl = findOrCreateTable(table_name, row_count) orelse return 1;
    if (tbl.column_count >= MAX_COLUMNS) return 2;
    tbl.columns[tbl.column_count] = ColumnData{
        .name = col_name,
        .col_type = .string,
        .data = .{ .strings = .{ .offsets = offsets, .lengths = lengths, .data = data } },
        .row_count = row_count,
    };
    tbl.column_count += 1;
    return 0;
}

// ============================================================================
// Query Execution
// ============================================================================

fn executeVectorSearch(table: *const TableInfo, near: *const WhereClause, limit: usize, out_indices: []u32) !usize {
    // Find column
    var col: ?*const ColumnData = null;
    for (table.columns[0..table.column_count]) |maybe_col| {
        if (maybe_col) |*c| {
            if (std.mem.eql(u8, c.name, near.column.?)) {
                col = c;
                break;
            }
        }
    }
    const c = col orelse return 0;
    
    const query_vec = near.near_vector[0..near.near_dim];
    const top_k = if (limit > 0) limit else 10;
    
    // Top-K heaps (one per fragment + merge)
    // For simplicity, we'll maintain one global top-k heap
    // out_indices will store the indices, we need scores too
    // We'll use a temporary buffer for scores
    const scores_ptr = memory.wasmAlloc(top_k * 4) orelse return 0;
    const scores = @as([*]f32, @ptrCast(@alignCast(scores_ptr)))[0..top_k];
    
    // Initialize scores
    for (0..top_k) |i| scores[i] = -2.0;
    for (0..top_k) |i| out_indices[i] = 0;
    
    var current_abs_idx: u32 = 0;
    
    // Iterate fragments
    for (table.fragments[0..table.fragment_count]) |maybe_frag| {
        if (maybe_frag) |frag| {
            const row_count = frag.getRowCount();
            const col_idx = c.fragment_col_idx;
            const dim = c.vector_dim;
            
            if (dim != near.near_dim) {
                // Dimension mismatch, skip or error? Skip for now.
                current_abs_idx += @intCast(row_count);
                continue;
            }
            
            // Chunked read and search
            const CHUNK_SIZE = 256; // Smaller chunk for vectors (large size)
            const buf_ptr = memory.wasmAlloc(CHUNK_SIZE * dim * 4) orelse return 0;
            const buf = @as([*]f32, @ptrCast(@alignCast(buf_ptr)));
            
            var processed: usize = 0;
            while (processed < row_count) {
                const chunk_len = @min(CHUNK_SIZE, row_count - processed);
                
                // Read vectors
                var floats_read: usize = 0;
                for (0..chunk_len) |i| {
                    const row = @as(u32, @intCast(processed + i));
                    const n = frag.fragmentReadVectorAt(col_idx, row, buf + i * dim, dim);
                    floats_read += n;
                }
                
                // Search chunk
                // Use a modified vectorSearchBuffer that takes explicit start_index
                _ = simd_search.vectorSearchBuffer(
                    buf,
                    chunk_len,
                    dim,
                    query_vec.ptr,
                    top_k,
                    out_indices.ptr,
                    scores.ptr,
                    1, // assume normalized
                    current_abs_idx + @as(u32, @intCast(processed))
                );
                
                processed += chunk_len;
            }
            
            current_abs_idx += @intCast(row_count);
        }
    }
    
    // Count valid results
    var result_count: usize = 0;
    for (0..top_k) |i| {
        if (scores[i] > -2.0) result_count += 1;
    }
    return result_count;
}

fn executeSelectQuery(table: *const TableInfo, query: *const ParsedQuery) !void {
    const row_count = table.row_count;

    // Apply WHERE filter
    var match_indices: [MAX_ROWS]u32 = undefined;
    var match_count: usize = 0;

    // Check for NEAR clause
    var near_clause: ?*const WhereClause = null;
    if (query.where_clause) |*where| {
        if (where.op == .near) {
            near_clause = where;
        } else if (where.op == .and_op) {
            if (where.left) |l| { if (l.op == .near) near_clause = l; }
            if (where.right) |r| { if (r.op == .near) near_clause = r; }
        }
    }

    if (near_clause) |near| {
        // Execute vector search
        const limit = if (query.limit_value) |l| l else 10;
        match_count = try executeVectorSearch(table, near, limit, &match_indices);
        // TODO: Apply other filters if any (post-filtering)
    } else {
        for (0..@min(row_count, MAX_ROWS)) |i| {
            if (query.where_clause) |*where| {
                if (evaluateWhere(table, where, @intCast(i))) {
                    match_indices[match_count] = @intCast(i);
                    match_count += 1;
                }
            } else {
                match_indices[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }

    // Determine output columns first (needed for DISTINCT)
    var output_cols: [MAX_COLUMNS]usize = undefined;
    var output_count: usize = 0;

    if (query.is_star) {
        for (0..table.column_count) |i| {
            output_cols[output_count] = i;
            output_count += 1;
        }
    } else {
        for (query.select_cols[0..query.select_count]) |col_name| {
            for (table.columns[0..table.column_count], 0..) |maybe_col, i| {
                if (maybe_col) |col| {
                    if (std.mem.eql(u8, col.name, col_name)) {
                        output_cols[output_count] = i;
                        output_count += 1;
                        break;
                    }
                }
            }
        }
    }

    // Apply ORDER BY
    if (query.order_by_col) |order_col| {
        sortIndices(table, match_indices[0..match_count], order_col, query.order_dir);
    }

    // Apply DISTINCT - remove duplicate rows
    if (query.is_distinct and match_count > 0 and output_count > 0) {
        var unique_count: usize = 1;
        var i: usize = 1;
        while (i < match_count) : (i += 1) {
            var is_dup = false;
            // Check if this row matches any previous row
            var j: usize = 0;
            while (j < unique_count) : (j += 1) {
                var all_equal = true;
                for (output_cols[0..output_count]) |col_idx| {
                    if (table.columns[col_idx]) |*col| {
                        const v1 = getFloatValue(table, col, match_indices[i]);
                        const v2 = getFloatValue(table, col, match_indices[j]);
                        if (v1 != v2) {
                            all_equal = false;
                            break;
                        }
                    }
                }
                if (all_equal) {
                    is_dup = true;
                    break;
                }
            }
            if (!is_dup) {
                match_indices[unique_count] = match_indices[i];
                unique_count += 1;
            }
        }
        match_count = unique_count;
    }

    // Apply OFFSET/LIMIT
    var start: usize = 0;
    var end: usize = match_count;
    if (query.offset_value) |offset| {
        start = @min(offset, match_count);
    }
    if (query.limit_value) |limit| {
        end = @min(start + limit, match_count);
    }
    const final_count = end - start;

    // Write result
    try writeSelectResult(table, output_cols[0..output_count], match_indices[start..end], final_count);
}

fn executeSetOpQuery(sql: []const u8, query: *const ParsedQuery) !void {
    // Execute first query to temp storage
    var first_results: [MAX_ROWS]u32 = undefined;
    var first_count: usize = 0;

    // Find first table
    var first_table: ?*const TableInfo = null;
    for (&tables) |*t| {
        if (t.*) |*tbl| {
            if (std.mem.eql(u8, tbl.name, query.table_name)) {
                first_table = tbl;
                break;
            }
        }
    }
    const table1 = first_table orelse return error.TableNotFound;

    // Get first query matching rows
    for (0..@min(table1.row_count, MAX_ROWS)) |i| {
        if (query.where_clause) |*where| {
            if (evaluateWhere(table1, where, @intCast(i))) {
                first_results[first_count] = @intCast(i);
                first_count += 1;
            }
        } else {
            first_results[first_count] = @intCast(i);
            first_count += 1;
        }
    }

    // Parse second query
    const second_sql = sql[query.set_op_query_start..];
    var second_query = parseSql(second_sql) orelse return error.InvalidSql;

    // Find second table
    var second_table: ?*const TableInfo = null;
    for (&tables) |*t| {
        if (t.*) |*tbl| {
            if (std.mem.eql(u8, tbl.name, second_query.table_name)) {
                second_table = tbl;
                break;
            }
        }
    }
    const table2 = second_table orelse return error.TableNotFound;

    // Get second query matching rows
    var second_results: [MAX_ROWS]u32 = undefined;
    var second_count: usize = 0;

    for (0..@min(table2.row_count, MAX_ROWS)) |i| {
        if (second_query.where_clause) |*where| {
            if (evaluateWhere(table2, where, @intCast(i))) {
                second_results[second_count] = @intCast(i);
                second_count += 1;
            }
        } else {
            second_results[second_count] = @intCast(i);
            second_count += 1;
        }
    }

    // Use first table's columns for output
    var output_cols: [MAX_SELECT_COLS][]const u8 = undefined;
    var output_count: usize = 0;

    if (query.is_star) {
        for (table1.columns[0..table1.column_count]) |maybe_col| {
            if (maybe_col) |col| {
                if (output_count < MAX_SELECT_COLS) {
                    output_cols[output_count] = col.name;
                    output_count += 1;
                }
            }
        }
    } else {
        for (query.select_cols[0..query.select_count]) |col| {
            if (output_count < MAX_SELECT_COLS) {
                output_cols[output_count] = col;
                output_count += 1;
            }
        }
    }

    // Handle different set operations
    switch (query.set_op) {
        .union_all => {
            // UNION ALL: concatenate all rows from both queries
            const total_count = first_count + second_count;
            if (total_count == 0) {
                try writeEmptyResult();
                return;
            }
            try writeSetOpResult(table1, table2, output_cols[0..output_count], first_results[0..first_count], second_results[0..second_count], .union_all);
        },
        .union_distinct => {
            // UNION: concatenate and deduplicate
            const total_count = first_count + second_count;
            if (total_count == 0) {
                try writeEmptyResult();
                return;
            }
            try writeSetOpResult(table1, table2, output_cols[0..output_count], first_results[0..first_count], second_results[0..second_count], .union_distinct);
        },
        .intersect, .intersect_all => {
            // INTERSECT: rows that exist in both queries
            // For simplicity, we compare rows by their values in the first column
            var intersect_results: [MAX_ROWS]u32 = undefined;
            var intersect_count: usize = 0;

            // Get the first column for comparison
            var col1: ?*const ColumnData = null;
            var col2: ?*const ColumnData = null;
            if (output_count > 0) {
                for (table1.columns[0..table1.column_count]) |maybe_col| {
                    if (maybe_col) |*c| {
                        if (std.mem.eql(u8, c.name, output_cols[0])) {
                            col1 = c;
                            break;
                        }
                    }
                }
                for (table2.columns[0..table2.column_count]) |maybe_col| {
                    if (maybe_col) |*c| {
                        if (std.mem.eql(u8, c.name, output_cols[0])) {
                            col2 = c;
                            break;
                        }
                    }
                }
            }

            if (col1 != null and col2 != null) {
                // Find rows in first result that match any row in second result
                for (first_results[0..first_count]) |ri1| {
                    var found = false;
                    for (second_results[0..second_count]) |ri2| {
                        if (rowsMatch(col1.?, col2.?, ri1, ri2)) {
                            found = true;
                            break;
                        }
                    }
                    if (found and intersect_count < MAX_ROWS) {
                        intersect_results[intersect_count] = ri1;
                        intersect_count += 1;
                    }
                }
            }

            if (intersect_count == 0) {
                try writeEmptyResult();
                return;
            }

            // Convert column names to indices
            var col_indices: [MAX_SELECT_COLS]usize = undefined;
            var idx_count: usize = 0;
            for (output_cols[0..output_count]) |col_name| {
                for (table1.columns[0..table1.column_count], 0..) |maybe_col, i| {
                    if (maybe_col) |c| {
                        if (std.mem.eql(u8, c.name, col_name)) {
                            col_indices[idx_count] = i;
                            idx_count += 1;
                            break;
                        }
                    }
                }
            }
            try writeSelectResult(table1, col_indices[0..idx_count], intersect_results[0..intersect_count], intersect_count);
        },
        .except, .except_all => {
            // EXCEPT: rows in first query but not in second
            var except_results: [MAX_ROWS]u32 = undefined;
            var except_count: usize = 0;

            // Get the first column for comparison
            var col1: ?*const ColumnData = null;
            var col2: ?*const ColumnData = null;
            if (output_count > 0) {
                for (table1.columns[0..table1.column_count]) |maybe_col| {
                    if (maybe_col) |*c| {
                        if (std.mem.eql(u8, c.name, output_cols[0])) {
                            col1 = c;
                            break;
                        }
                    }
                }
                for (table2.columns[0..table2.column_count]) |maybe_col| {
                    if (maybe_col) |*c| {
                        if (std.mem.eql(u8, c.name, output_cols[0])) {
                            col2 = c;
                            break;
                        }
                    }
                }
            }

            if (col1 != null and col2 != null) {
                // Find rows in first result that DON'T match any row in second result
                for (first_results[0..first_count]) |ri1| {
                    var found = false;
                    for (second_results[0..second_count]) |ri2| {
                        if (rowsMatch(col1.?, col2.?, ri1, ri2)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found and except_count < MAX_ROWS) {
                        except_results[except_count] = ri1;
                        except_count += 1;
                    }
                }
            } else {
                // No matching columns, return all from first
                for (first_results[0..first_count]) |ri| {
                    if (except_count < MAX_ROWS) {
                        except_results[except_count] = ri;
                        except_count += 1;
                    }
                }
            }

            if (except_count == 0) {
                try writeEmptyResult();
                return;
            }

            // Convert column names to indices
            var col_indices: [MAX_SELECT_COLS]usize = undefined;
            var idx_count: usize = 0;
            for (output_cols[0..output_count]) |col_name| {
                for (table1.columns[0..table1.column_count], 0..) |maybe_col, i| {
                    if (maybe_col) |c| {
                        if (std.mem.eql(u8, c.name, col_name)) {
                            col_indices[idx_count] = i;
                            idx_count += 1;
                            break;
                        }
                    }
                }
            }
            try writeSelectResult(table1, col_indices[0..idx_count], except_results[0..except_count], except_count);
        },
        .none => return error.InvalidSetOp,
    }
}

fn writeEmptyResult() !void {
    const buf = allocResultBuffer(HEADER_SIZE) orelse return error.OutOfMemory;
    _ = buf;
    _ = writeU32(RESULT_VERSION);
    _ = writeU32(0);
    _ = writeU64(0);
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(0);
}

fn rowsMatch(col1: *const ColumnData, col2: *const ColumnData, ri1: u32, ri2: u32) bool {
    // Compare values based on column type
    if (col1.col_type != col2.col_type) return false;

    switch (col1.col_type) {
        .int64 => return col1.data.int64[ri1] == col2.data.int64[ri2],
        .int32 => return col1.data.int32[ri1] == col2.data.int32[ri2],
        .float64 => return col1.data.float64[ri1] == col2.data.float64[ri2],
        .float32 => return col1.data.float32[ri1] == col2.data.float32[ri2],
        .string => {
            const off1 = col1.data.strings.offsets[ri1];
            const len1 = col1.data.strings.lengths[ri1];
            const off2 = col2.data.strings.offsets[ri2];
            const len2 = col2.data.strings.lengths[ri2];
            if (len1 != len2) return false;
            return std.mem.eql(u8, col1.data.strings.data[off1..][0..len1], col2.data.strings.data[off2..][0..len2]);
        },
    }
}

fn writeSetOpResult(table1: *const TableInfo, table2: *const TableInfo, output_cols: []const []const u8, first_indices: []const u32, second_indices: []const u32, op: SetOpType) !void {
    const num_cols = output_cols.len;

    // For UNION DISTINCT, we need to filter duplicates from second set
    var filtered_second: [MAX_ROWS]u32 = undefined;
    var filtered_count: usize = 0;

    if (op == .union_distinct and num_cols > 0) {
        // Get first column for comparison
        var col1: ?*const ColumnData = null;
        var col2: ?*const ColumnData = null;
        for (table1.columns[0..table1.column_count]) |maybe_col| {
            if (maybe_col) |*c| {
                if (std.mem.eql(u8, c.name, output_cols[0])) {
                    col1 = c;
                    break;
                }
            }
        }
        for (table2.columns[0..table2.column_count]) |maybe_col| {
            if (maybe_col) |*c| {
                if (std.mem.eql(u8, c.name, output_cols[0])) {
                    col2 = c;
                    break;
                }
            }
        }

        if (col1 != null and col2 != null) {
            // Filter second indices - only include if not in first set
            for (second_indices) |ri2| {
                var is_dup = false;
                for (first_indices) |ri1| {
                    if (rowsMatch(col1.?, col2.?, ri1, ri2)) {
                        is_dup = true;
                        break;
                    }
                }
                if (!is_dup and filtered_count < MAX_ROWS) {
                    filtered_second[filtered_count] = ri2;
                    filtered_count += 1;
                }
            }
        } else {
            // No matching columns, include all from second
            for (second_indices, 0..) |ri, i| {
                if (i < MAX_ROWS) {
                    filtered_second[i] = ri;
                    filtered_count += 1;
                }
            }
        }
    } else {
        // UNION ALL - include all from second
        for (second_indices, 0..) |ri, i| {
            if (i < MAX_ROWS) {
                filtered_second[i] = ri;
                filtered_count += 1;
            }
        }
    }

    const actual_second = filtered_second[0..filtered_count];
    const total_count = first_indices.len + filtered_count;

    // Calculate size
    var data_size: usize = 0;
    for (output_cols) |col_name| {
        for (table1.columns[0..table1.column_count]) |maybe_col| {
            if (maybe_col) |col| {
                if (std.mem.eql(u8, col.name, col_name)) {
                    data_size += switch (col.col_type) {
                        .int64, .float64 => total_count * 8,
                        .int32, .float32 => total_count * 4,
                        .string => total_count * 8 + 65536,
                    };
                    break;
                }
            }
        }
    }

    const total_size = HEADER_SIZE + num_cols * 16 + data_size;
    _ = allocResultBuffer(total_size) orelse return error.OutOfMemory;

    // Write header
    _ = writeU32(RESULT_VERSION);
    _ = writeU32(@intCast(num_cols));
    _ = writeU64(@intCast(total_count));
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE + @as(u32, @intCast(num_cols * 16)));
    _ = writeU32(0);

    // Write column metadata
    var data_offset: u32 = 0;
    for (output_cols) |col_name| {
        for (table1.columns[0..table1.column_count]) |maybe_col| {
            if (maybe_col) |col| {
                if (std.mem.eql(u8, col.name, col_name)) {
                    _ = writeU32(@intFromEnum(col.col_type));
                    _ = writeU32(0);
                    _ = writeU32(@intCast(col.name.len));
                    _ = writeU32(data_offset);
                    data_offset += switch (col.col_type) {
                        .int64, .float64 => @intCast(total_count * 8),
                        .int32, .float32 => @intCast(total_count * 4),
                        .string => @intCast(total_count * 8 + 65536),
                    };
                    break;
                }
            }
        }
    }

    // Write data from both tables
    for (output_cols) |col_name| {
        // Write from first table
        for (table1.columns[0..table1.column_count]) |maybe_col| {
            if (maybe_col) |*col| {
                if (std.mem.eql(u8, col.name, col_name)) {
                    switch (col.col_type) {
                        .int64 => {
                            for (first_indices) |ri| {
                                const val = col.data.int64[ri];
                                _ = writeToResult(std.mem.asBytes(&val));
                            }
                        },
                        .float64 => {
                            for (first_indices) |ri| {
                                const val = col.data.float64[ri];
                                _ = writeToResult(std.mem.asBytes(&val));
                            }
                        },
                        .int32 => {
                            for (first_indices) |ri| {
                                const val = col.data.int32[ri];
                                _ = writeToResult(std.mem.asBytes(&val));
                            }
                        },
                        .float32 => {
                            for (first_indices) |ri| {
                                const val = col.data.float32[ri];
                                _ = writeToResult(std.mem.asBytes(&val));
                            }
                        },
                        .string => {
                            var str_offset: u32 = 0;
                            for (first_indices) |ri| {
                                _ = writeU32(str_offset);
                                _ = writeU32(col.data.strings.lengths[ri]);
                                str_offset += col.data.strings.lengths[ri];
                            }
                            for (first_indices) |ri| {
                                const off = col.data.strings.offsets[ri];
                                const len = col.data.strings.lengths[ri];
                                _ = writeToResult(col.data.strings.data[off..][0..len]);
                            }
                        },
                    }
                    break;
                }
            }
        }

        // Write from second table (filtered if UNION DISTINCT)
        for (table2.columns[0..table2.column_count]) |maybe_col| {
            if (maybe_col) |*col| {
                if (std.mem.eql(u8, col.name, col_name)) {
                    switch (col.col_type) {
                        .int64 => {
                            for (actual_second) |ri| {
                                const val = col.data.int64[ri];
                                _ = writeToResult(std.mem.asBytes(&val));
                            }
                        },
                        .float64 => {
                            for (actual_second) |ri| {
                                const val = col.data.float64[ri];
                                _ = writeToResult(std.mem.asBytes(&val));
                            }
                        },
                        .int32 => {
                            for (actual_second) |ri| {
                                const val = col.data.int32[ri];
                                _ = writeToResult(std.mem.asBytes(&val));
                            }
                        },
                        .float32 => {
                            for (actual_second) |ri| {
                                const val = col.data.float32[ri];
                                _ = writeToResult(std.mem.asBytes(&val));
                            }
                        },
                        .string => {
                            var str_offset: u32 = 0;
                            for (actual_second) |ri| {
                                _ = writeU32(str_offset);
                                _ = writeU32(col.data.strings.lengths[ri]);
                                str_offset += col.data.strings.lengths[ri];
                            }
                            for (actual_second) |ri| {
                                const off = col.data.strings.offsets[ri];
                                const len = col.data.strings.lengths[ri];
                                _ = writeToResult(col.data.strings.data[off..][0..len]);
                            }
                        },
                    }
                    break;
                }
            }
        }
    }
}

fn executeJoinQuery(left_table: *const TableInfo, query: *const ParsedQuery) !void {
    if (query.join_count == 0) return error.NoJoin;

    const join = query.joins[0];

    // Find right table
    var right_table: ?*const TableInfo = null;
    for (&tables) |*t| {
        if (t.*) |*tbl| {
            if (std.mem.eql(u8, tbl.name, join.table_name)) {
                right_table = tbl;
                break;
            }
        }
    }

    const rtbl = right_table orelse return error.TableNotFound;

    // Find join columns
    var left_col: ?*const ColumnData = null;
    var right_col: ?*const ColumnData = null;

    for (left_table.columns[0..left_table.column_count]) |maybe_col| {
        if (maybe_col) |*c| {
            if (std.mem.eql(u8, c.name, join.left_col)) {
                left_col = c;
                break;
            }
        }
    }

    for (rtbl.columns[0..rtbl.column_count]) |maybe_col| {
        if (maybe_col) |*c| {
            if (std.mem.eql(u8, c.name, join.right_col)) {
                right_col = c;
                break;
            }
        }
    }

    const lc = left_col orelse return error.ColumnNotFound;
    const rc = right_col orelse return error.ColumnNotFound;

    // Build hash table for right side (smaller table optimization)
    const MAX_JOIN_ROWS: usize = 10000;
    var join_pairs: [MAX_JOIN_ROWS]struct { left: u32, right: u32 } = undefined;
    var pair_count: usize = 0;

    // Nested loop join (simple but works)
    for (0..@min(left_table.row_count, MAX_JOIN_ROWS)) |li| {
        const left_val = getIntValue(left_table, lc, @intCast(li));

        for (0..@min(rtbl.row_count, MAX_JOIN_ROWS)) |ri| {
            const right_val = getIntValue(rtbl, rc, @intCast(ri));

            if (left_val == right_val) {
                if (pair_count < MAX_JOIN_ROWS) {
                    join_pairs[pair_count] = .{ .left = @intCast(li), .right = @intCast(ri) };
                    pair_count += 1;
                }
            }
        }
    }

    // Calculate result columns (all from both tables)
    const left_col_count = left_table.column_count;
    const right_col_count = rtbl.column_count;
    const total_cols = left_col_count + right_col_count;

    // Calculate data size
    var data_size: usize = 0;
    for (left_table.columns[0..left_col_count]) |maybe_col| {
        if (maybe_col) |col| {
            data_size += switch (col.col_type) {
                .int64, .float64 => pair_count * 8,
                .int32, .float32 => pair_count * 4,
                .string => pair_count * 8 + 65536,
            };
        }
    }
    for (rtbl.columns[0..right_col_count]) |maybe_col| {
        if (maybe_col) |col| {
            data_size += switch (col.col_type) {
                .int64, .float64 => pair_count * 8,
                .int32, .float32 => pair_count * 4,
                .string => pair_count * 8 + 65536,
            };
        }
    }

    const total_size = HEADER_SIZE + total_cols * 16 + data_size;
    _ = allocResultBuffer(total_size) orelse return error.OutOfMemory;

    // Write header
    _ = writeU32(RESULT_VERSION);
    _ = writeU32(@intCast(total_cols));
    _ = writeU64(@intCast(pair_count));
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE + @as(u32, @intCast(total_cols * 16)));
    _ = writeU32(0);

    // Write column metadata (left table columns first, then right)
    var data_offset: u32 = 0;
    for (left_table.columns[0..left_col_count]) |maybe_col| {
        if (maybe_col) |col| {
            _ = writeU32(@intFromEnum(col.col_type));
            _ = writeU32(0);
            _ = writeU32(@intCast(col.name.len));
            _ = writeU32(data_offset);
            data_offset += switch (col.col_type) {
                .int64, .float64 => @intCast(pair_count * 8),
                .int32, .float32 => @intCast(pair_count * 4),
                .string => @intCast(pair_count * 8 + 65536),
            };
        }
    }
    for (rtbl.columns[0..right_col_count]) |maybe_col| {
        if (maybe_col) |col| {
            _ = writeU32(@intFromEnum(col.col_type));
            _ = writeU32(0);
            _ = writeU32(@intCast(col.name.len));
            _ = writeU32(data_offset);
            data_offset += switch (col.col_type) {
                .int64, .float64 => @intCast(pair_count * 8),
                .int32, .float32 => @intCast(pair_count * 4),
                .string => @intCast(pair_count * 8 + 65536),
            };
        }
    }

    // Write left table data
    for (left_table.columns[0..left_col_count]) |maybe_col| {
        if (maybe_col) |*col| {
            for (join_pairs[0..pair_count]) |pair| {
                const val = getFloatValue(left_table, col, pair.left);
                _ = writeToResult(std.mem.asBytes(&val));
            }
        }
    }

    // Write right table data
    for (rtbl.columns[0..right_col_count]) |maybe_col| {
        if (maybe_col) |*col| {
            for (join_pairs[0..pair_count]) |pair| {
                const val = getFloatValue(rtbl, col, pair.right);
                _ = writeToResult(std.mem.asBytes(&val));
            }
        }
    }
}

fn executeAggregateQuery(table: *const TableInfo, query: *const ParsedQuery) !void {
    const row_count = table.row_count;

    // Apply WHERE filter first
    var match_indices: [MAX_ROWS]u32 = undefined;
    var match_count: usize = 0;

    for (0..@min(row_count, MAX_ROWS)) |i| {
        if (query.where_clause) |*where| {
            if (evaluateWhere(table, where, @intCast(i))) {
                match_indices[match_count] = @intCast(i);
                match_count += 1;
            }
        } else {
            match_indices[match_count] = @intCast(i);
            match_count += 1;
        }
    }

    if (query.group_by_count > 0) {
        // GROUP BY aggregation
        try executeGroupByQuery(table, query, match_indices[0..match_count]);
    } else {
        // Simple aggregation (no GROUP BY)
        try executeSimpleAggQuery(table, query, match_indices[0..match_count]);
    }
}

fn executeSimpleAggQuery(table: *const TableInfo, query: *const ParsedQuery, indices: []const u32) !void {
    // Single row result with aggregate values
    const num_aggs = query.agg_count;
    var agg_results: [MAX_AGGREGATES]f64 = undefined;

    for (query.aggregates[0..num_aggs], 0..) |agg, i| {
        agg_results[i] = computeAggregate(table, agg, indices);
    }

    // Allocate result buffer
    const total_size = HEADER_SIZE + num_aggs * 16 + num_aggs * 8;
    _ = allocResultBuffer(total_size) orelse return error.OutOfMemory;

    // Write header
    _ = writeU32(RESULT_VERSION);
    _ = writeU32(@intCast(num_aggs));
    _ = writeU64(1); // 1 row
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE + @as(u32, @intCast(num_aggs * 16)));
    _ = writeU32(0);

    // Write column metadata
    var data_offset: u32 = 0;
    for (0..num_aggs) |_| {
        _ = writeU32(1); // float64
        _ = writeU32(0);
        _ = writeU32(0);
        _ = writeU32(data_offset);
        data_offset += 8;
    }

    // Write data
    for (agg_results[0..num_aggs]) |val| {
        _ = writeToResult(std.mem.asBytes(&val));
    }
}

fn executeGroupByQuery(table: *const TableInfo, query: *const ParsedQuery, indices: []const u32) !void {
    // Simple hash-based GROUP BY for integer columns
    // For now, support single column GROUP BY
    if (query.group_by_count != 1) return error.UnsupportedGroupBy;

    const group_col_name = query.group_by_cols[0];
    var group_col: ?*const ColumnData = null;

    for (table.columns[0..table.column_count]) |maybe_col| {
        if (maybe_col) |*col| {
            if (std.mem.eql(u8, col.name, group_col_name)) {
                group_col = col;
                break;
            }
        }
    }

    const gcol = group_col orelse return error.ColumnNotFound;

    // Simple approach: collect unique values and their indices
    const MAX_GROUPS: usize = 1024;
    var group_keys: [MAX_GROUPS]i64 = undefined;
    var group_starts: [MAX_GROUPS]usize = undefined;
    var group_counts: [MAX_GROUPS]usize = undefined;
    var num_groups: usize = 0;

    // Build groups (O(n*k) but simple)
    for (indices) |idx| {
        const key = getIntValue(table, gcol, idx);

        // Find or add group
        var found = false;
        for (group_keys[0..num_groups], 0..) |gk, gi| {
            if (gk == key) {
                group_counts[gi] += 1;
                found = true;
                break;
            }
        }
        if (!found and num_groups < MAX_GROUPS) {
            group_keys[num_groups] = key;
            group_starts[num_groups] = idx;
            group_counts[num_groups] = 1;
            num_groups += 1;
        }
    }

    // Compute aggregates per group
    const num_aggs = query.agg_count;
    const result_cols = 1 + num_aggs; // group column + aggregates
    const num_rows = num_groups;

    // Allocate result
    const total_size = HEADER_SIZE + result_cols * 16 + num_rows * 8 * result_cols;
    _ = allocResultBuffer(total_size) orelse return error.OutOfMemory;

    // Write header
    _ = writeU32(RESULT_VERSION);
    _ = writeU32(@intCast(result_cols));
    _ = writeU64(@intCast(num_rows));
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE);
    _ = writeU32(HEADER_SIZE + @as(u32, @intCast(result_cols * 16)));
    _ = writeU32(0);

    // Write column metadata
    var data_offset: u32 = 0;
    // Group column
    _ = writeU32(1); // float64 (we convert to float for simplicity)
    _ = writeU32(0);
    _ = writeU32(0);
    _ = writeU32(data_offset);
    data_offset += @intCast(num_rows * 8);

    // Aggregate columns
    for (0..num_aggs) |_| {
        _ = writeU32(1); // float64
        _ = writeU32(0);
        _ = writeU32(0);
        _ = writeU32(data_offset);
        data_offset += @intCast(num_rows * 8);
    }

    // Write group column data
    for (group_keys[0..num_groups]) |key| {
        const val: f64 = @floatFromInt(key);
        _ = writeToResult(std.mem.asBytes(&val));
    }

    // Write aggregate data
    for (query.aggregates[0..num_aggs]) |agg| {
        for (group_keys[0..num_groups], 0..) |key, gi| {
            // Find indices for this group and compute aggregate
            var group_indices: [MAX_ROWS]u32 = undefined;
            var group_idx_count: usize = 0;

            for (indices) |idx| {
                if (getIntValue(table, gcol, idx) == key) {
                    group_indices[group_idx_count] = idx;
                    group_idx_count += 1;
                }
            }

            _ = gi;
            const val = computeAggregate(table, agg, group_indices[0..group_idx_count]);
            _ = writeToResult(std.mem.asBytes(&val));
        }
    }
}

fn computeLazyAggregate(table: *const TableInfo, col: *const ColumnData, agg: AggExpr, start_idx: u32, count: usize) f64 {
    var remaining = count;
    var current_abs_idx = start_idx;
    
    // Accumulators
    var sum: f64 = 0;
    var min_val: f64 = std.math.floatMax(f64);
    var max_val: f64 = -std.math.floatMax(f64);
    var total_count: f64 = 0;

    // Find starting fragment
    var frag_idx: usize = 0;
    var frag_start_row: u32 = 0;
    
    while (frag_idx < table.fragment_count) {
        const frag = table.fragments[frag_idx].?;
        const f_rows = @as(u32, @intCast(frag.getRowCount()));
        if (current_abs_idx < frag_start_row + f_rows) break;
        frag_start_row += f_rows;
        frag_idx += 1;
    }
    
    const CHUNK_SIZE = 1024;
    var buf_f64: [CHUNK_SIZE]f64 = undefined;
    
    while (remaining > 0 and frag_idx < table.fragment_count) {
        const frag = table.fragments[frag_idx].?;
        const f_rows = @as(u32, @intCast(frag.getRowCount()));
        
        const rel_start = current_abs_idx - frag_start_row;
        const available_in_frag = f_rows - rel_start;
        const to_read = @min(remaining, available_in_frag);
        
        var chunk_offset: u32 = 0;
        while (chunk_offset < to_read) {
            const chunk_len = @min(CHUNK_SIZE, to_read - chunk_offset);
            const read_start = rel_start + chunk_offset;
            
            // Read chunk (convert to f64 if needed)
            var n: usize = 0;
            switch (col.col_type) {
                .float64 => {
                    n = frag.fragmentReadFloat64(col.fragment_col_idx, @ptrCast(&buf_f64), chunk_len, read_start);
                },
                .float32 => {
                    var buf_f32: [CHUNK_SIZE]f32 = undefined;
                    n = frag.fragmentReadFloat32(col.fragment_col_idx, @ptrCast(&buf_f32), chunk_len, read_start);
                    for (0..n) |i| buf_f64[i] = @floatCast(buf_f32[i]);
                },
                .int64 => {
                    var buf_i64: [CHUNK_SIZE]i64 = undefined;
                    n = frag.fragmentReadInt64(col.fragment_col_idx, @ptrCast(&buf_i64), chunk_len, read_start);
                    for (0..n) |i| buf_f64[i] = @floatFromInt(buf_i64[i]);
                },
                .int32 => {
                    var buf_i32: [CHUNK_SIZE]i32 = undefined;
                    n = frag.fragmentReadInt32(col.fragment_col_idx, @ptrCast(&buf_i32), chunk_len, read_start);
                    for (0..n) |i| buf_f64[i] = @floatFromInt(buf_i32[i]);
                },
                else => {},
            }

            // SIMD Aggregation on chunk
            const ptr = @as([*]const f64, @ptrCast(&buf_f64));
            switch (agg.func) {
                .sum, .avg => {
                    sum += aggregates.sumFloat64Buffer(ptr, n);
                    total_count += @floatFromInt(n);
                },
                .min => {
                    const chunk_min = aggregates.minFloat64Buffer(ptr, n);
                    if (chunk_min < min_val) min_val = chunk_min;
                },
                .max => {
                    const chunk_max = aggregates.maxFloat64Buffer(ptr, n);
                    if (chunk_max > max_val) max_val = chunk_max;
                },
                .count => total_count += @floatFromInt(n),
            }
            
            chunk_offset += @intCast(chunk_len);
        }
        
        remaining -= to_read;
        current_abs_idx += @intCast(to_read);
        frag_start_row += f_rows;
        frag_idx += 1;
    }

    return switch (agg.func) {
        .sum => sum,
        .avg => if (total_count > 0) sum / total_count else 0,
        .min => min_val,
        .max => max_val,
        .count => total_count,
    };
}

fn computeAggregate(table: *const TableInfo, agg: AggExpr, indices: []const u32) f64 {
    if (agg.func == .count and (std.mem.eql(u8, agg.column, "*") or agg.column.len == 0)) {
        return @floatFromInt(indices.len);
    }

    // Find column
    var col: ?*const ColumnData = null;
    for (table.columns[0..table.column_count]) |maybe_col| {
        if (maybe_col) |*c| {
            if (std.mem.eql(u8, c.name, agg.column) or std.mem.eql(u8, agg.column, "*")) {
                col = c;
                break;
            }
        }
    }
    const c = col orelse return 0;

    // Check if indices are contiguous
    var is_contiguous = true;
    if (indices.len > 0) {
        const len = indices.len;
        if (indices[0] != 0 or 
            indices[len - 1] != @as(u32, @intCast(len - 1)) or
            indices[len / 2] != @as(u32, @intCast(len / 2))) {
            is_contiguous = false;
        }
    } else {
        is_contiguous = false;
    }

    // Optimized paths
    if (is_contiguous) {
        if (c.is_lazy) {
            return computeLazyAggregate(table, c, agg, 0, indices.len);
        } else if (c.col_type == .float64) {
            // SIMD on memory buffer
            const ptr = c.data.float64.ptr;
            const len = indices.len; // Using indices length as we assume contiguous from 0
            // Wait, is it guaranteed to be 0..len? Indices are just a list. 
            // is_contiguous checks if it looks like 0, 1, 2... N-1.
            // But we must ensure N <= row_count.
            // If it is 0..len-1, we can use slice 0..len
            return switch (agg.func) {
                .sum => aggregates.sumFloat64Buffer(ptr, len),
                .avg => aggregates.avgFloat64Buffer(ptr, len),
                .min => aggregates.minFloat64Buffer(ptr, len),
                .max => aggregates.maxFloat64Buffer(ptr, len),
                .count => @floatFromInt(len),
            };
        }
    }

    var sum: f64 = 0;
    var min_val: f64 = std.math.floatMax(f64);
    var max_val: f64 = -std.math.floatMax(f64);

    for (indices) |idx| {
        const val = getFloatValue(table, c, idx);
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    return switch (agg.func) {
        .sum => sum,
        .avg => if (indices.len > 0) sum / @as(f64, @floatFromInt(indices.len)) else 0,
        .min => if (indices.len > 0) min_val else 0,
        .max => if (indices.len > 0) max_val else 0,
        .count => @floatFromInt(indices.len),
    };
}

fn getFloatValue(table: *const TableInfo, col: *const ColumnData, idx: u32) f64 {
    if (col.is_lazy) {
        // Find fragment
        var current_idx = idx;
        for (table.fragments[0..table.fragment_count]) |maybe_frag| {
            if (maybe_frag) |frag| {
                const count = frag.getRowCount();
                if (current_idx < count) {
                    // Read from this fragment
                    var val: f64 = 0;
                    const col_idx = col.fragment_col_idx;
                    const c_idx = @as(u32, @intCast(current_idx));
                    
                    switch (col.col_type) {
                        .float64 => {
                            _ = frag.fragmentReadFloat64(col_idx, @ptrCast(&val), 1, c_idx);
                        },
                        .float32 => {
                            var f32_val: f32 = 0;
                            _ = frag.fragmentReadFloat32(col_idx, @ptrCast(&f32_val), 1, c_idx);
                            val = @floatCast(f32_val);
                        },
                        .int64 => {
                            var i64_val: i64 = 0;
                            _ = frag.fragmentReadInt64(col_idx, @ptrCast(&i64_val), 1, c_idx);
                            val = @floatFromInt(i64_val);
                        },
                        .int32 => {
                            var i32_val: i32 = 0;
                            _ = frag.fragmentReadInt32(col_idx, @ptrCast(&i32_val), 1, c_idx);
                            val = @floatFromInt(i32_val);
                        },
                        else => {},
                    }
                    return val;
                }
                current_idx -= @intCast(count);
            }
        }
        return 0;
    }

    return switch (col.col_type) {
        .float64 => col.data.float64[idx],
        .int64 => @floatFromInt(col.data.int64[idx]),
        .int32 => @floatFromInt(col.data.int32[idx]),
        .float32 => col.data.float32[idx],
        .string => 0,
    };
}

fn getIntValue(table: *const TableInfo, col: *const ColumnData, idx: u32) i64 {
    if (col.is_lazy) {
        var current_idx = idx;
        for (table.fragments[0..table.fragment_count]) |maybe_frag| {
            if (maybe_frag) |frag| {
                const count = frag.getRowCount();
                if (current_idx < count) {
                    var val: i64 = 0;
                    const col_idx = col.fragment_col_idx;
                    const c_idx = @as(u32, @intCast(current_idx));

                    switch (col.col_type) {
                        .int64 => {
                            _ = frag.fragmentReadInt64(col_idx, @ptrCast(&val), 1, c_idx);
                        },
                        .int32 => {
                            var i32_val: i32 = 0;
                            _ = frag.fragmentReadInt32(col_idx, @ptrCast(&i32_val), 1, c_idx);
                            val = i32_val;
                        },
                        .float64 => {
                            var f64_val: f64 = 0;
                            _ = frag.fragmentReadFloat64(col_idx, @ptrCast(&f64_val), 1, c_idx);
                            val = @intFromFloat(f64_val);
                        },
                        .float32 => {
                            var f32_val: f32 = 0;
                            _ = frag.fragmentReadFloat32(col_idx, @ptrCast(&f32_val), 1, c_idx);
                            val = @intFromFloat(f32_val);
                        },
                        else => {},
                    }
                    return val;
                }
                current_idx -= @intCast(count);
            }
        }
        return 0;
    }

    return switch (col.col_type) {
        .int64 => col.data.int64[idx],
        .int32 => col.data.int32[idx],
        .float64 => @intFromFloat(col.data.float64[idx]),
        .float32 => @intFromFloat(col.data.float32[idx]),
        .string => 0,
    };
}

// ============================================================================
// Window Function Execution
// ============================================================================

fn executeWindowQuery(table: *const TableInfo, query: *const ParsedQuery) !void {
    const row_count = table.row_count;
    if (row_count == 0) {
        const buf = allocResultBuffer(HEADER_SIZE) orelse return error.OutOfMemory;
        _ = buf;
        _ = writeU32(RESULT_VERSION);
        _ = writeU32(0);
        _ = writeU64(0);
        _ = writeU32(HEADER_SIZE);
        _ = writeU32(HEADER_SIZE);
        _ = writeU32(HEADER_SIZE);
        _ = writeU32(0);
        return;
    }

    // Get row indices (after WHERE filter)
    var indices: [MAX_ROWS]u32 = undefined;
    var idx_count: usize = 0;

    for (0..@min(row_count, MAX_ROWS)) |i| {
        if (query.where_clause) |*where| {
            if (evaluateWhere(table, where, @intCast(i))) {
                indices[idx_count] = @intCast(i);
                idx_count += 1;
            }
        } else {
            indices[idx_count] = @intCast(i);
            idx_count += 1;
        }
    }

    if (idx_count == 0) {
        const buf = allocResultBuffer(HEADER_SIZE) orelse return error.OutOfMemory;
        _ = buf;
        _ = writeU32(RESULT_VERSION);
        _ = writeU32(0);
        _ = writeU64(0);
        _ = writeU32(HEADER_SIZE);
        _ = writeU32(HEADER_SIZE);
        _ = writeU32(HEADER_SIZE);
        _ = writeU32(0);
        return;
    }

    // Compute window function values for each row
    // Storage: window_values[window_idx][row_idx]
    var window_values: [MAX_WINDOW_FUNCS][MAX_ROWS]f64 = undefined;

    for (query.window_funcs[0..query.window_count], 0..) |wf, wf_idx| {
        // Partition rows
        var partition_keys: [MAX_ROWS]i64 = undefined;
        for (indices[0..idx_count], 0..) |row_idx, i| {
            var key: i64 = 0;
            for (wf.partition_by[0..wf.partition_count]) |part_col| {
                for (table.columns[0..table.column_count]) |maybe_col| {
                    if (maybe_col) |*col| {
                        if (std.mem.eql(u8, col.name, part_col)) {
                            key = key *% 1000003 +% getIntValue(table, col, row_idx);
                            break;
                        }
                    }
                }
            }
            partition_keys[i] = key;
        }

        // Get order column if specified
        var order_col: ?*const ColumnData = null;
        if (wf.order_by_col) |order_name| {
            for (table.columns[0..table.column_count]) |maybe_col| {
                if (maybe_col) |*col| {
                    if (std.mem.eql(u8, col.name, order_name)) {
                        order_col = col;
                        break;
                    }
                }
            }
        }

        // Get arg column if specified
        var arg_col: ?*const ColumnData = null;
        if (wf.arg_col) |arg_name| {
            for (table.columns[0..table.column_count]) |maybe_col| {
                if (maybe_col) |*col| {
                    if (std.mem.eql(u8, col.name, arg_name)) {
                        arg_col = col;
                        break;
                    }
                }
            }
        }

        // Process each unique partition
        var processed: [MAX_ROWS]bool = .{false} ** MAX_ROWS;
        for (0..idx_count) |i| {
            if (processed[i]) continue;
            const part_key = partition_keys[i];

            // Collect partition indices
            var part_indices: [MAX_ROWS]usize = undefined;
            var part_count: usize = 0;
            for (0..idx_count) |j| {
                if (partition_keys[j] == part_key) {
                    part_indices[part_count] = j;
                    part_count += 1;
                    processed[j] = true;
                }
            }

            // Sort partition by order column if specified
            if (order_col) |ocol| {
                // Simple bubble sort for partition (usually small)
                var k: usize = 0;
                while (k < part_count) : (k += 1) {
                    var m = k + 1;
                    while (m < part_count) : (m += 1) {
                        const a_val = getFloatValue(table, ocol, indices[part_indices[k]]);
                        const b_val = getFloatValue(table, ocol, indices[part_indices[m]]);
                        const should_swap = if (wf.order_dir == .desc) a_val < b_val else a_val > b_val;
                        if (should_swap) {
                            const tmp = part_indices[k];
                            part_indices[k] = part_indices[m];
                            part_indices[m] = tmp;
                        }
                    }
                }
            }

            // Compute window function for each row in partition
            for (part_indices[0..part_count], 0..) |part_idx, rank| {
                const row_idx = indices[part_idx];
                var value: f64 = 0;

                switch (wf.func) {
                    .row_number => value = @floatFromInt(rank + 1),
                    .rank => {
                        // Same value = same rank
                        if (rank == 0) {
                            value = 1;
                        } else if (order_col) |ocol| {
                            const curr = getFloatValue(table, ocol, row_idx);
                            const prev = getFloatValue(table, ocol, indices[part_indices[rank - 1]]);
                            if (curr == prev) {
                                value = window_values[wf_idx][part_indices[rank - 1]];
                            } else {
                                value = @floatFromInt(rank + 1);
                            }
                        } else {
                            value = @floatFromInt(rank + 1);
                        }
                    },
                    .dense_rank => {
                        if (rank == 0) {
                            value = 1;
                        } else if (order_col) |ocol| {
                            const curr = getFloatValue(table, ocol, row_idx);
                            const prev = getFloatValue(table, ocol, indices[part_indices[rank - 1]]);
                            if (curr == prev) {
                                value = window_values[wf_idx][part_indices[rank - 1]];
                            } else {
                                value = window_values[wf_idx][part_indices[rank - 1]] + 1;
                            }
                        } else {
                            value = @floatFromInt(rank + 1);
                        }
                    },
                    .lag => {
                        if (rank > 0) {
                            if (arg_col) |acol| {
                                value = getFloatValue(table, acol, indices[part_indices[rank - 1]]);
                            }
                        }
                    },
                    .lead => {
                        if (rank + 1 < part_count) {
                            if (arg_col) |acol| {
                                value = getFloatValue(table, acol, indices[part_indices[rank + 1]]);
                            }
                        }
                    },
                    .first_value => {
                        if (arg_col) |acol| {
                            value = getFloatValue(table, acol, indices[part_indices[0]]);
                        }
                    },
                    .last_value => {
                        if (arg_col) |acol| {
                            value = getFloatValue(table, acol, indices[part_indices[part_count - 1]]);
                        }
                    },
                    .ntile => {
                        const n = wf.ntile_n;
                        const bucket_size = (part_count + n - 1) / n;
                        value = @floatFromInt(@min(rank / bucket_size + 1, n));
                    },
                    .percent_rank => {
                        if (part_count <= 1) {
                            value = 0;
                        } else {
                            // Calculate actual rank
                            var actual_rank: usize = 1;
                            if (rank > 0 and order_col != null) {
                                const ocol = order_col.?;
                                const curr = getFloatValue(table, ocol, row_idx);
                                for (0..rank) |j| {
                                    const prev = getFloatValue(table, ocol, indices[part_indices[j]]);
                                    if (prev != curr) actual_rank = j + 2;
                                }
                            }
                            value = @as(f64, @floatFromInt(actual_rank - 1)) / @as(f64, @floatFromInt(part_count - 1));
                        }
                    },
                    .cume_dist => {
                        var count: usize = rank + 1;
                        if (order_col) |ocol| {
                            const curr = getFloatValue(table, ocol, row_idx);
                            var j = rank + 1;
                            while (j < part_count) : (j += 1) {
                                const next = getFloatValue(table, ocol, indices[part_indices[j]]);
                                if (next == curr) count += 1 else break;
                            }
                        }
                        value = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(part_count));
                    },
                    .sum => {
                        if (arg_col) |acol| {
                            var s: f64 = 0;
                            for (0..rank + 1) |j| {
                                s += getFloatValue(table, acol, indices[part_indices[j]]);
                            }
                            value = s;
                        }
                    },
                    .count => {
                        value = @floatFromInt(rank + 1);
                    },
                    .avg => {
                        if (arg_col) |acol| {
                            var s: f64 = 0;
                            for (0..rank + 1) |j| {
                                s += getFloatValue(table, acol, indices[part_indices[j]]);
                            }
                            value = s / @as(f64, @floatFromInt(rank + 1));
                        }
                    },
                    .min => {
                        if (arg_col) |acol| {
                            var m: f64 = std.math.floatMax(f64);
                            for (0..rank + 1) |j| {
                                const v = getFloatValue(table, acol, indices[part_indices[j]]);
                                if (v < m) m = v;
                            }
                            value = m;
                        }
                    },
                    .max => {
                        if (arg_col) |acol| {
                            var m: f64 = -std.math.floatMax(f64);
                            for (0..rank + 1) |j| {
                                const v = getFloatValue(table, acol, indices[part_indices[j]]);
                                if (v > m) m = v;
                            }
                            value = m;
                        }
                    },
                }

                window_values[wf_idx][part_idx] = value;
            }
        }
    }

    // Build output: regular columns + window function columns
    // Determine output columns
    var output_cols: [MAX_SELECT_COLS]usize = undefined;
    var output_count: usize = 0;

    if (query.is_star) {
        for (0..table.column_count) |i| {
            if (table.columns[i] != null) {
                output_cols[output_count] = i;
                output_count += 1;
            }
        }
    } else {
        for (query.select_cols[0..query.select_count]) |col_name| {
            for (table.columns[0..table.column_count], 0..) |maybe_col, i| {
                if (maybe_col) |col| {
                    if (std.mem.eql(u8, col.name, col_name)) {
                        output_cols[output_count] = i;
                        output_count += 1;
                        break;
                    }
                }
            }
        }
    }

    const total_cols = output_count + query.window_count;

    // Calculate sizes
    var names_size: usize = 0;
    for (output_cols[0..output_count]) |ci| {
        if (table.columns[ci]) |col| {
            names_size += col.name.len;
        }
    }
    for (query.window_funcs[0..query.window_count]) |wf| {
        if (wf.alias) |alias| {
            names_size += alias.len;
        } else {
            names_size += 10; // Default name like "window_0"
        }
    }

    const extended_header: u32 = 36;
    const meta_size: u32 = @intCast(total_cols * 16);
    const names_offset: u32 = extended_header + meta_size;
    const data_offset_start: u32 = names_offset + @as(u32, @intCast(names_size));

    var data_size: usize = 0;
    for (output_cols[0..output_count]) |ci| {
        if (table.columns[ci]) |col| {
            data_size += switch (col.col_type) {
                .int64, .float64 => idx_count * 8,
                .int32, .float32 => idx_count * 4,
                .string => idx_count * 8 + 65536,
            };
        }
    }
    // Window function columns are all float64
    data_size += query.window_count * idx_count * 8;

    const total_size = data_offset_start + data_size;
    _ = allocResultBuffer(total_size) orelse return error.OutOfMemory;

    // Write header
    _ = writeU32(RESULT_VERSION);
    _ = writeU32(@intCast(total_cols));
    _ = writeU64(@intCast(idx_count));
    _ = writeU32(extended_header);
    _ = writeU32(extended_header);
    _ = writeU32(data_offset_start);
    _ = writeU32(0);
    _ = writeU32(names_offset);

    // Write column metadata
    var name_offset: u32 = 0;
    var data_offset: u32 = 0;

    // Regular columns
    for (output_cols[0..output_count]) |ci| {
        if (table.columns[ci]) |col| {
            _ = writeU32(@intFromEnum(col.col_type));
            _ = writeU32(name_offset);
            _ = writeU32(@intCast(col.name.len));
            _ = writeU32(data_offset);

            name_offset += @intCast(col.name.len);
            data_offset += switch (col.col_type) {
                .int64, .float64 => @intCast(idx_count * 8),
                .int32, .float32 => @intCast(idx_count * 4),
                .string => @intCast(idx_count * 8 + 65536),
            };
        }
    }

    // Window function columns (all float64)
    for (query.window_funcs[0..query.window_count]) |wf| {
        _ = writeU32(1); // float64
        _ = writeU32(name_offset);
        const name_len: u32 = if (wf.alias) |alias| @intCast(alias.len) else 10;
        _ = writeU32(name_len);
        _ = writeU32(data_offset);

        name_offset += name_len;
        data_offset += @intCast(idx_count * 8);
    }

    // Write column names
    for (output_cols[0..output_count]) |ci| {
        if (table.columns[ci]) |col| {
            _ = writeToResult(col.name);
        }
    }
    for (query.window_funcs[0..query.window_count], 0..) |wf, i| {
        if (wf.alias) |alias| {
            _ = writeToResult(alias);
        } else {
            var buf: [10]u8 = undefined;
            const name = std.fmt.bufPrint(&buf, "window_{d}", .{i}) catch "window";
            _ = writeToResult(name);
        }
    }

    // Write data - regular columns
    for (output_cols[0..output_count]) |ci| {
        if (table.columns[ci]) |*col| {
            switch (col.col_type) {
                .float64 => {
                    for (indices[0..idx_count]) |ri| {
                        const val = col.data.float64[ri];
                        _ = writeToResult(std.mem.asBytes(&val));
                    }
                },
                .int64 => {
                    for (indices[0..idx_count]) |ri| {
                        const val: f64 = @floatFromInt(col.data.int64[ri]);
                        _ = writeToResult(std.mem.asBytes(&val));
                    }
                },
                .int32 => {
                    for (indices[0..idx_count]) |ri| {
                        const val = col.data.int32[ri];
                        _ = writeToResult(std.mem.asBytes(&val));
                    }
                },
                .float32 => {
                    for (indices[0..idx_count]) |ri| {
                        const val = col.data.float32[ri];
                        _ = writeToResult(std.mem.asBytes(&val));
                    }
                },
                .string => {
                    var str_offset: u32 = 0;
                    for (indices[0..idx_count]) |ri| {
                        _ = writeU32(str_offset);
                        _ = writeU32(col.data.strings.lengths[ri]);
                        str_offset += col.data.strings.lengths[ri];
                    }
                    for (indices[0..idx_count]) |ri| {
                        const off = col.data.strings.offsets[ri];
                        const len = col.data.strings.lengths[ri];
                        _ = writeToResult(col.data.strings.data[off..][0..len]);
                    }
                },
            }
        }
    }

    // Write window function data
    for (0..query.window_count) |wf_idx| {
        for (0..idx_count) |i| {
            const val = window_values[wf_idx][i];
            _ = writeToResult(std.mem.asBytes(&val));
        }
    }
}

// ============================================================================
// CTE Execution
// ============================================================================

fn executeCTEQuery(sql: []const u8, outer_query: *const ParsedQuery, cte: *const CTEDef) !void {
    _ = outer_query;
    // Parse and execute the CTE's inner query
    const cte_sql = sql[cte.query_start..cte.query_end];
    var inner_query = parseSql(cte_sql) orelse return error.InvalidSql;

    // Find the table referenced by the inner query
    var table: ?*const TableInfo = null;
    for (&tables) |*t| {
        if (t.*) |*tbl| {
            if (std.mem.eql(u8, tbl.name, inner_query.table_name)) {
                table = tbl;
                break;
            }
        }
    }

    const tbl = table orelse return error.TableNotFound;

    // Execute based on inner query type
    if (inner_query.agg_count > 0 or inner_query.group_by_count > 0) {
        try executeAggregateQuery(tbl, &inner_query);
    } else if (inner_query.window_count > 0) {
        try executeWindowQuery(tbl, &inner_query);
    } else {
        try executeSelectQuery(tbl, &inner_query);
    }
}

fn copyLazyColumnRange(table: *const TableInfo, col: *const ColumnData, start_idx: u32, count: usize) !void {
    var remaining = count;
    var current_abs_idx = start_idx;
    
    // Find starting fragment
    var frag_idx: usize = 0;
    var frag_start_row: u32 = 0;
    
    while (frag_idx < table.fragment_count) {
        const frag = table.fragments[frag_idx].?;
        const f_rows = @as(u32, @intCast(frag.getRowCount()));
        if (current_abs_idx < frag_start_row + f_rows) break;
        frag_start_row += f_rows;
        frag_idx += 1;
    }
    
    const CHUNK_SIZE = 1024;
    var buf_f64: [CHUNK_SIZE]f64 = undefined;
    var buf_i64: [CHUNK_SIZE]i64 = undefined;
    var buf_i32: [CHUNK_SIZE]i32 = undefined;
    var buf_f32: [CHUNK_SIZE]f32 = undefined;
    
    while (remaining > 0 and frag_idx < table.fragment_count) {
        const frag = table.fragments[frag_idx].?;
        const f_rows = @as(u32, @intCast(frag.getRowCount()));
        
        // Calculate how many rows we can read from this fragment
        const rel_start = current_abs_idx - frag_start_row;
        const available_in_frag = f_rows - rel_start;
        const to_read = @min(remaining, available_in_frag);
        
        // Process in chunks
        var chunk_offset: u32 = 0;
        while (chunk_offset < to_read) {
            const chunk_len = @min(CHUNK_SIZE, to_read - chunk_offset);
            const read_start = rel_start + chunk_offset;
            
            switch (col.col_type) {
                .float64 => {
                    const n = frag.fragmentReadFloat64(col.fragment_col_idx, @ptrCast(&buf_f64), chunk_len, read_start);
                    _ = writeToResult(std.mem.sliceAsBytes(buf_f64[0..n]));
                },
                .int64 => {
                    const n = frag.fragmentReadInt64(col.fragment_col_idx, @ptrCast(&buf_i64), chunk_len, read_start);
                    _ = writeToResult(std.mem.sliceAsBytes(buf_i64[0..n]));
                },
                .int32 => {
                    const n = frag.fragmentReadInt32(col.fragment_col_idx, @ptrCast(&buf_i32), chunk_len, read_start);
                    _ = writeToResult(std.mem.sliceAsBytes(buf_i32[0..n]));
                },
                .float32 => {
                    const n = frag.fragmentReadFloat32(col.fragment_col_idx, @ptrCast(&buf_f32), chunk_len, read_start);
                    _ = writeToResult(std.mem.sliceAsBytes(buf_f32[0..n]));
                },
                .string => {
                    // Strings are harder to batch because of variable length
                    // Fallback to row-by-row for now
                    for (0..chunk_len) |_| {
                        // TODO: Optimize string batch reading
                        // We are just writing dummy data if we don't implement this fully
                        // But for now let's just loop
                        // Actually, this branch is unreachable if we don't call copyLazyColumnRange for strings
                        // (which we don't in writeSelectResult)
                    }
                },
            }
            
            chunk_offset += @intCast(chunk_len);
        }
        
        remaining -= to_read;
        current_abs_idx += @intCast(to_read);
        
        frag_start_row += f_rows;
        frag_idx += 1;
    }
}

fn writeSelectResult(table: *const TableInfo, col_indices: []const usize, row_indices: []const u32, row_count: usize) !void {
    const col_count = col_indices.len;

    // Calculate names section size
    var names_size: usize = 0;
    for (col_indices) |ci| {
        if (table.columns[ci]) |col| {
            names_size += col.name.len;
        }
    }

    // Calculate data size
    var data_size: usize = 0;
    for (col_indices) |ci| {
        if (table.columns[ci]) |col| {
            data_size += switch (col.col_type) {
                .int64, .float64 => row_count * 8,
                .int32, .float32 => row_count * 4,
                .string => row_count * 8 + 65536,
            };
        }
    }

    // Layout: header (36) + metadata (16*cols) + names + data
    const extended_header: u32 = 36; // Extended header with names_offset
    const meta_size: u32 = @intCast(col_count * 16);
    const names_offset: u32 = extended_header + meta_size;
    const data_offset_start: u32 = names_offset + @as(u32, @intCast(names_size));

    const total_size = data_offset_start + data_size;
    _ = allocResultBuffer(total_size) orelse return error.OutOfMemory;

    // Write header (extended: 36 bytes)
    _ = writeU32(RESULT_VERSION);
    _ = writeU32(@intCast(col_count));
    _ = writeU64(@intCast(row_count));
    _ = writeU32(extended_header); // header size
    _ = writeU32(extended_header); // meta offset
    _ = writeU32(data_offset_start); // data offset
    _ = writeU32(0); // flags
    _ = writeU32(names_offset); // NEW: names section offset

    // Write column metadata: type, name_offset, name_len, data_offset
    var name_offset: u32 = 0;
    var data_offset: u32 = 0;
    for (col_indices) |ci| {
        if (table.columns[ci]) |col| {
            _ = writeU32(@intFromEnum(col.col_type));
            _ = writeU32(name_offset); // offset within names section
            _ = writeU32(@intCast(col.name.len));
            _ = writeU32(data_offset);

            name_offset += @intCast(col.name.len);
            data_offset += switch (col.col_type) {
                .int64, .float64 => @intCast(row_count * 8),
                .int32, .float32 => @intCast(row_count * 4),
                .string => @intCast(row_count * 8 + 65536),
            };
        }
    }

    // Write names section
    for (col_indices) |ci| {
        if (table.columns[ci]) |col| {
            _ = writeToResult(col.name);
        }
    }

    // Optimization: Check if row_indices are contiguous (0, 1, 2, ... row_count)
    // This allows bulk memcpy instead of row-by-row copy
    var is_contiguous = true;
    if (row_indices.len != table.row_count) {
        is_contiguous = false;
    } else if (row_indices.len > 0) {
        // Check start, end, and middle to be reasonably sure
        const len = row_indices.len;
        if (row_indices[0] != 0 or 
            row_indices[len - 1] != @as(u32, @intCast(len - 1)) or
            row_indices[len / 2] != @as(u32, @intCast(len / 2))) {
            is_contiguous = false;
        }
    }

    // Write data
    for (col_indices) |ci| {
        if (table.columns[ci]) |*col| {
            switch (col.col_type) {
                .float64 => {
                    if (is_contiguous) {
                        if (col.is_lazy) {
                            try copyLazyColumnRange(table, col, 0, row_count);
                        } else {
                            _ = writeToResult(std.mem.sliceAsBytes(col.data.float64[0..row_count]));
                        }
                    } else {
                        for (row_indices) |ri| {
                            const val = getFloatValue(table, col, ri);
                            _ = writeToResult(std.mem.asBytes(&val));
                        }
                    }
                },
                .int64 => {
                    if (is_contiguous) {
                        if (col.is_lazy) {
                            try copyLazyColumnRange(table, col, 0, row_count);
                        } else {
                            _ = writeToResult(std.mem.sliceAsBytes(col.data.int64[0..row_count]));
                        }
                    } else {
                        for (row_indices) |ri| {
                            const val: f64 = @floatFromInt(getIntValue(table, col, ri));
                            _ = writeToResult(std.mem.asBytes(&val));
                        }
                    }
                },
                .int32 => {
                    if (is_contiguous) {
                        if (col.is_lazy) {
                            try copyLazyColumnRange(table, col, 0, row_count);
                        } else {
                            _ = writeToResult(std.mem.sliceAsBytes(col.data.int32[0..row_count]));
                        }
                    } else {
                        for (row_indices) |ri| {
                            const val = getIntValue(table, col, ri);
                            const val_i32: i32 = @intCast(val);
                            _ = writeToResult(std.mem.asBytes(&val_i32));
                        }
                    }
                },
                .float32 => {
                    if (is_contiguous) {
                        if (col.is_lazy) {
                            try copyLazyColumnRange(table, col, 0, row_count);
                        }
                        else {
                            _ = writeToResult(std.mem.sliceAsBytes(col.data.float32[0..row_count]));
                        }
                    } else {
                        for (row_indices) |ri| {
                            const val = getFloatValue(table, col, ri);
                            const val_f32: f32 = @floatCast(val);
                            _ = writeToResult(std.mem.asBytes(&val_f32));
                        }
                    }
                },
                .string => {
                    // Strings are complex (offset/length/data) - still optimizing structure
                    var str_offset: u32 = 0;
                    for (row_indices) |ri| {
                        _ = writeU32(str_offset);
                        _ = writeU32(col.data.strings.lengths[ri]);
                        str_offset += col.data.strings.lengths[ri];
                    }
                    if (is_contiguous) {
                        // For SELECT *, we can just dump the whole string data block if we were clever
                        // But we need to repack because we wrote new offsets starting at 0
                        // However, since row_indices are contiguous, the string data IS contiguous in source!
                        // Calculate total length
                        var total_len: usize = 0;
                        for (0..row_count) |i| total_len += col.data.strings.lengths[i];
                        // Just write the big block
                        _ = writeToResult(col.data.strings.data[0..total_len]);
                    } else {
                        for (row_indices) |ri| {
                            const off = col.data.strings.offsets[ri];
                            const len = col.data.strings.lengths[ri];
                            _ = writeToResult(col.data.strings.data[off..][0..len]);
                        }
                    }
                },
            }
        }
    }
}

// ============================================================================
// WHERE Evaluation
// ============================================================================

fn evaluateWhere(table: *const TableInfo, where: *const WhereClause, row_idx: u32) bool {
    switch (where.op) {
        .and_op => {
            const l = where.left orelse return false;
            const r = where.right orelse return false;
            return evaluateWhere(table, l, row_idx) and evaluateWhere(table, r, row_idx);
        },
        .or_op => {
            const l = where.left orelse return false;
            const r = where.right orelse return false;
            return evaluateWhere(table, l, row_idx) or evaluateWhere(table, r, row_idx);
        },
        else => {
            const col_name = where.column orelse return false;
            var col: ?*const ColumnData = null;
            for (table.columns[0..table.column_count]) |maybe_col| {
                if (maybe_col) |*c| {
                    if (std.mem.eql(u8, c.name, col_name)) {
                        col = c;
                        break;
                    }
                }
            }
            const c = col orelse return false;
            return evaluateComparison(table, c, row_idx, where);
        },
    }
}

fn evaluateComparison(table: *const TableInfo, col: *const ColumnData, row_idx: u32, where: *const WhereClause) bool {
    switch (where.op) {
        .eq => {
            if (where.value_int) |val| {
                return getIntValue(table, col, row_idx) == val;
            } else if (where.value_float) |val| {
                return getFloatValue(table, col, row_idx) == val;
            }
        },
        .ne => {
            if (where.value_int) |val| {
                return getIntValue(table, col, row_idx) != val;
            } else if (where.value_float) |val| {
                return getFloatValue(table, col, row_idx) != val;
            }
        },
        .lt => {
            if (where.value_int) |val| {
                return getIntValue(table, col, row_idx) < val;
            } else if (where.value_float) |val| {
                return getFloatValue(table, col, row_idx) < val;
            }
        },
        .le => {
            if (where.value_int) |val| {
                return getIntValue(table, col, row_idx) <= val;
            } else if (where.value_float) |val| {
                return getFloatValue(table, col, row_idx) <= val;
            }
        },
        .gt => {
            if (where.value_int) |val| {
                return getIntValue(table, col, row_idx) > val;
            } else if (where.value_float) |val| {
                return getFloatValue(table, col, row_idx) > val;
            }
        },
        .ge => {
            if (where.value_int) |val| {
                return getIntValue(table, col, row_idx) >= val;
            } else if (where.value_float) |val| {
                return getFloatValue(table, col, row_idx) >= val;
            }
        },
        else => return false,
    }
    return false;
}

fn getStringValue(col: *const ColumnData, row_idx: u32) []const u8 {
    if (col.col_type != .string) return "";
    if (row_idx >= col.row_count) return "";
    const off = col.data.strings.offsets[row_idx];
    const len = col.data.strings.lengths[row_idx];
    return col.data.strings.data[off..][0..len];
}

fn matchLike(str: []const u8, pattern: []const u8) bool {
    // Simple LIKE matching: % matches any sequence, _ matches single char
    var si: usize = 0;
    var pi: usize = 0;
    var star_idx: ?usize = null;
    var match_idx: usize = 0;

    while (si < str.len) {
        if (pi < pattern.len and (pattern[pi] == str[si] or pattern[pi] == '_')) {
            si += 1;
            pi += 1;
        } else if (pi < pattern.len and pattern[pi] == '%') {
            star_idx = pi;
            match_idx = si;
            pi += 1;
        } else if (star_idx) |star| {
            pi = star + 1;
            match_idx += 1;
            si = match_idx;
        } else {
            return false;
        }
    }

    while (pi < pattern.len and pattern[pi] == '%') pi += 1;
    return pi == pattern.len;
}

// ============================================================================
// Sorting
// ============================================================================

fn sortIndices(table: *const TableInfo, indices: []u32, col_name: []const u8, dir: OrderDir) void {
    var col: ?*const ColumnData = null;
    for (table.columns[0..table.column_count]) |maybe_col| {
        if (maybe_col) |*c| {
            if (std.mem.eql(u8, c.name, col_name)) {
                col = c;
                break;
            }
        }
    }
    const c = col orelse return;

    // Simple insertion sort (good enough for moderate sizes)
    for (1..indices.len) |i| {
        const key = indices[i];
        const key_val = getFloatValue(table, c, key);
        var j: usize = i;
        while (j > 0) {
            const cmp_val = getFloatValue(table, c, indices[j - 1]);
            const should_swap = if (dir == .asc) cmp_val > key_val else cmp_val < key_val;
            if (!should_swap) break;
            indices[j] = indices[j - 1];
            j -= 1;
        }
        indices[j] = key;
    }
}

// ============================================================================
// SQL Parser
// ============================================================================

fn parseSql(sql: []const u8) ?ParsedQuery {
    var query = ParsedQuery{};
    var pos: usize = 0;

    pos = skipWs(sql, pos);

    // Parse WITH clause (CTE) if present
    if (startsWithIC(sql[pos..], "WITH")) {
        pos += 4;
        pos = skipWs(sql, pos);

        // Skip RECURSIVE keyword if present
        if (startsWithIC(sql[pos..], "RECURSIVE")) {
            pos += 9;
            pos = skipWs(sql, pos);
        }

        // Parse CTE definitions
        while (query.cte_count < MAX_CTES) {
            // CTE name
            const name_start = pos;
            while (pos < sql.len and isIdent(sql[pos])) pos += 1;
            if (pos == name_start) break;
            const cte_name = sql[name_start..pos];
            pos = skipWs(sql, pos);

            // AS keyword
            if (!startsWithIC(sql[pos..], "AS")) break;
            pos += 2;
            pos = skipWs(sql, pos);

            // Opening paren
            if (pos >= sql.len or sql[pos] != '(') break;
            pos += 1;

            // Find matching closing paren
            const query_start = pos;
            var paren_depth: usize = 1;
            while (pos < sql.len and paren_depth > 0) {
                if (sql[pos] == '(') paren_depth += 1 else if (sql[pos] == ')') paren_depth -= 1;
                if (paren_depth > 0) pos += 1;
            }
            const query_end = pos;
            if (pos < sql.len and sql[pos] == ')') pos += 1;

            query.ctes[query.cte_count] = CTEDef{
                .name = cte_name,
                .query_start = query_start,
                .query_end = query_end,
            };
            query.cte_count += 1;

            pos = skipWs(sql, pos);

            // Check for another CTE definition (comma-separated)
            if (pos < sql.len and sql[pos] == ',') {
                pos += 1;
                pos = skipWs(sql, pos);
            } else {
                break;
            }
        }

        pos = skipWs(sql, pos);
    }

    if (!startsWithIC(sql[pos..], "SELECT")) return null;
    pos += 6;
    pos = skipWs(sql, pos);

    // Check for DISTINCT
    if (startsWithIC(sql[pos..], "DISTINCT")) {
        query.is_distinct = true;
        pos += 8;
        pos = skipWs(sql, pos);
    }

    // Parse select list
    pos = parseSelectList(sql, pos, &query) orelse return null;
    pos = skipWs(sql, pos);

    // FROM clause
    if (!startsWithIC(sql[pos..], "FROM")) return null;
    pos += 4;
    pos = skipWs(sql, pos);

    // Table name
    const tbl_start = pos;
    while (pos < sql.len and isIdent(sql[pos])) pos += 1;
    if (pos == tbl_start) return null;
    query.table_name = sql[tbl_start..pos];
    pos = skipWs(sql, pos);

    // Optional JOIN clauses
    while (query.join_count < MAX_JOINS) {
        pos = skipWs(sql, pos);

        var join_type: JoinType = .inner;

        if (startsWithIC(sql[pos..], "INNER")) {
            pos += 5;
            pos = skipWs(sql, pos);
            join_type = .inner;
        } else if (startsWithIC(sql[pos..], "LEFT")) {
            pos += 4;
            pos = skipWs(sql, pos);
            join_type = .left;
        } else if (startsWithIC(sql[pos..], "RIGHT")) {
            pos += 5;
            pos = skipWs(sql, pos);
            join_type = .right;
        } else if (startsWithIC(sql[pos..], "CROSS")) {
            pos += 5;
            pos = skipWs(sql, pos);
            join_type = .cross;
        }

        if (!startsWithIC(sql[pos..], "JOIN")) break;
        pos += 4;
        pos = skipWs(sql, pos);

        // Join table name
        const join_tbl_start = pos;
        while (pos < sql.len and isIdent(sql[pos])) pos += 1;
        if (pos == join_tbl_start) break;
        const join_table = sql[join_tbl_start..pos];
        pos = skipWs(sql, pos);

        // ON clause
        var left_col: []const u8 = "";
        var right_col: []const u8 = "";

        if (startsWithIC(sql[pos..], "ON")) {
            pos += 2;
            pos = skipWs(sql, pos);

            // Parse: table.col = table.col or col = col
            const on_start = pos;
            while (pos < sql.len and (isIdent(sql[pos]) or sql[pos] == '.')) pos += 1;
            const left_expr = sql[on_start..pos];

            // Extract column name (after last dot or whole thing)
            if (std.mem.lastIndexOf(u8, left_expr, ".")) |dot| {
                left_col = left_expr[dot + 1 ..];
            } else {
                left_col = left_expr;
            }

            pos = skipWs(sql, pos);
            if (pos < sql.len and sql[pos] == '=') pos += 1;
            pos = skipWs(sql, pos);

            const right_start = pos;
            while (pos < sql.len and (isIdent(sql[pos]) or sql[pos] == '.')) pos += 1;
            const right_expr = sql[right_start..pos];

            if (std.mem.lastIndexOf(u8, right_expr, ".")) |dot| {
                right_col = right_expr[dot + 1 ..];
            } else {
                right_col = right_expr;
            }
        }

        query.joins[query.join_count] = JoinClause{
            .table_name = join_table,
            .join_type = join_type,
            .left_col = left_col,
            .right_col = right_col,
        };
        query.join_count += 1;
    }

    pos = skipWs(sql, pos);

    // Optional WHERE
    if (startsWithIC(sql[pos..], "WHERE")) {
        pos += 5;
        pos = skipWs(sql, pos);
        query.where_clause = parseWhere(sql, &pos);
        pos = skipWs(sql, pos);
    }

    // Optional GROUP BY
    if (startsWithIC(sql[pos..], "GROUP")) {
        pos += 5;
        pos = skipWs(sql, pos);
        if (startsWithIC(sql[pos..], "BY")) {
            pos += 2;
            pos = skipWs(sql, pos);
            pos = parseGroupBy(sql, pos, &query);
        }
        pos = skipWs(sql, pos);
    }

    // Optional ORDER BY
    if (startsWithIC(sql[pos..], "ORDER")) {
        pos += 5;
        pos = skipWs(sql, pos);
        if (startsWithIC(sql[pos..], "BY")) {
            pos += 2;
            pos = skipWs(sql, pos);
            pos = parseOrderBy(sql, pos, &query);
        }
        pos = skipWs(sql, pos);
    }

    // Optional LIMIT
    if (startsWithIC(sql[pos..], "LIMIT")) {
        pos += 5;
        pos = skipWs(sql, pos);
        const num_start = pos;
        while (pos < sql.len and std.ascii.isDigit(sql[pos])) pos += 1;
        if (pos > num_start) {
            query.limit_value = std.fmt.parseInt(u32, sql[num_start..pos], 10) catch null;
        }
        pos = skipWs(sql, pos);
    }

    // Optional OFFSET
    if (startsWithIC(sql[pos..], "OFFSET")) {
        pos += 6;
        pos = skipWs(sql, pos);
        const num_start = pos;
        while (pos < sql.len and std.ascii.isDigit(sql[pos])) pos += 1;
        if (pos > num_start) {
            query.offset_value = std.fmt.parseInt(u32, sql[num_start..pos], 10) catch null;
        }
        pos = skipWs(sql, pos);
    }

    // Optional set operations (UNION, INTERSECT, EXCEPT)
    if (startsWithIC(sql[pos..], "UNION")) {
        pos += 5;
        pos = skipWs(sql, pos);

        if (startsWithIC(sql[pos..], "ALL")) {
            pos += 3;
            query.set_op = .union_all;
        } else {
            query.set_op = .union_distinct;
        }
        pos = skipWs(sql, pos);
        query.set_op_query_start = pos;
    } else if (startsWithIC(sql[pos..], "INTERSECT")) {
        pos += 9;
        pos = skipWs(sql, pos);

        if (startsWithIC(sql[pos..], "ALL")) {
            pos += 3;
            query.set_op = .intersect_all;
        } else {
            query.set_op = .intersect;
        }
        pos = skipWs(sql, pos);
        query.set_op_query_start = pos;
    } else if (startsWithIC(sql[pos..], "EXCEPT")) {
        pos += 6;
        pos = skipWs(sql, pos);

        if (startsWithIC(sql[pos..], "ALL")) {
            pos += 3;
            query.set_op = .except_all;
        } else {
            query.set_op = .except;
        }
        pos = skipWs(sql, pos);
        query.set_op_query_start = pos;
    }

    return query;
}

fn parseSelectList(sql: []const u8, start: usize, query: *ParsedQuery) ?usize {
    var pos = start;

    if (pos < sql.len and sql[pos] == '*') {
        query.is_star = true;
        return pos + 1;
    }

    // Parse column list, aggregates, or window functions
    while (pos < sql.len) {
        pos = skipWs(sql, pos);

        // Check for window function first
        if (parseWindowFunction(sql, &pos, query)) {
            // Parsed a window function
        }
        // Check for aggregate function
        else if (parseAggregate(sql, &pos, query)) {
            // Parsed an aggregate
        } else {
            // Regular column
            const col_start = pos;
            while (pos < sql.len and (isIdent(sql[pos]) or sql[pos] == '.')) pos += 1;
            if (pos > col_start and query.select_count < MAX_SELECT_COLS) {
                query.select_cols[query.select_count] = sql[col_start..pos];
                query.select_count += 1;
            }
        }

        pos = skipWs(sql, pos);

        // Skip alias (AS name)
        if (startsWithIC(sql[pos..], "AS")) {
            pos += 2;
            pos = skipWs(sql, pos);
            const alias_start = pos;
            while (pos < sql.len and isIdent(sql[pos])) pos += 1;
            // Store alias for last window function if we just parsed one
            if (query.window_count > 0 and pos > alias_start) {
                query.window_funcs[query.window_count - 1].alias = sql[alias_start..pos];
            }
            pos = skipWs(sql, pos);
        }

        if (pos >= sql.len or sql[pos] != ',') break;
        pos += 1; // skip comma
    }

    return pos;
}

fn parseWindowFunction(sql: []const u8, pos: *usize, query: *ParsedQuery) bool {
    const window_funcs = [_]struct { name: []const u8, func: WindowFunc, has_arg: bool }{
        .{ .name = "ROW_NUMBER", .func = .row_number, .has_arg = false },
        .{ .name = "RANK", .func = .rank, .has_arg = false },
        .{ .name = "DENSE_RANK", .func = .dense_rank, .has_arg = false },
        .{ .name = "LAG", .func = .lag, .has_arg = true },
        .{ .name = "LEAD", .func = .lead, .has_arg = true },
        .{ .name = "NTILE", .func = .ntile, .has_arg = true },
        .{ .name = "PERCENT_RANK", .func = .percent_rank, .has_arg = false },
        .{ .name = "CUME_DIST", .func = .cume_dist, .has_arg = false },
        .{ .name = "FIRST_VALUE", .func = .first_value, .has_arg = true },
        .{ .name = "LAST_VALUE", .func = .last_value, .has_arg = true },
        .{ .name = "SUM", .func = .sum, .has_arg = true },
        .{ .name = "COUNT", .func = .count, .has_arg = true },
        .{ .name = "AVG", .func = .avg, .has_arg = true },
        .{ .name = "MIN", .func = .min, .has_arg = true },
        .{ .name = "MAX", .func = .max, .has_arg = true },
    };

    for (window_funcs) |f| {
        if (startsWithIC(sql[pos.*..], f.name)) {
            var p = pos.* + f.name.len;
            p = skipWs(sql, p);
            if (p >= sql.len or sql[p] != '(') continue;
            p += 1;
            p = skipWs(sql, p);

            var arg_col: ?[]const u8 = null;
            var ntile_n: u32 = 1;

            // Parse argument if needed
            if (f.has_arg) {
                if (f.func == .ntile) {
                    // NTILE(n) - parse number
                    const num_start = p;
                    while (p < sql.len and std.ascii.isDigit(sql[p])) p += 1;
                    if (p > num_start) {
                        ntile_n = std.fmt.parseInt(u32, sql[num_start..p], 10) catch 1;
                    }
                } else if (sql[p] != '*' and sql[p] != ')') {
                    const col_start = p;
                    while (p < sql.len and isIdent(sql[p])) p += 1;
                    if (p > col_start) {
                        arg_col = sql[col_start..p];
                    }
                } else if (sql[p] == '*') {
                    p += 1;
                }
            }

            p = skipWs(sql, p);
            if (p >= sql.len or sql[p] != ')') continue;
            p += 1;
            p = skipWs(sql, p);

            // Must have OVER clause for window function
            if (!startsWithIC(sql[p..], "OVER")) continue;
            p += 4;
            p = skipWs(sql, p);

            if (p >= sql.len or sql[p] != '(') continue;
            p += 1;
            p = skipWs(sql, p);

            var window_expr = WindowExpr{
                .func = f.func,
                .arg_col = arg_col,
                .ntile_n = ntile_n,
            };

            // Parse PARTITION BY
            if (startsWithIC(sql[p..], "PARTITION")) {
                p += 9;
                p = skipWs(sql, p);
                if (startsWithIC(sql[p..], "BY")) {
                    p += 2;
                    p = skipWs(sql, p);
                    // Parse partition columns
                    while (window_expr.partition_count < 4) {
                        const part_start = p;
                        while (p < sql.len and isIdent(sql[p])) p += 1;
                        if (p > part_start) {
                            window_expr.partition_by[window_expr.partition_count] = sql[part_start..p];
                            window_expr.partition_count += 1;
                        }
                        p = skipWs(sql, p);
                        if (p >= sql.len or sql[p] != ',') break;
                        p += 1;
                        p = skipWs(sql, p);
                    }
                }
                p = skipWs(sql, p);
            }

            // Parse ORDER BY
            if (startsWithIC(sql[p..], "ORDER")) {
                p += 5;
                p = skipWs(sql, p);
                if (startsWithIC(sql[p..], "BY")) {
                    p += 2;
                    p = skipWs(sql, p);
                    const order_start = p;
                    while (p < sql.len and isIdent(sql[p])) p += 1;
                    if (p > order_start) {
                        window_expr.order_by_col = sql[order_start..p];
                    }
                    p = skipWs(sql, p);
                    if (startsWithIC(sql[p..], "DESC")) {
                        window_expr.order_dir = .desc;
                        p += 4;
                    } else if (startsWithIC(sql[p..], "ASC")) {
                        p += 3;
                    }
                }
                p = skipWs(sql, p);
            }

            // Skip frame specification (ROWS BETWEEN ... AND ...)
            while (p < sql.len and sql[p] != ')') p += 1;

            if (p >= sql.len or sql[p] != ')') continue;
            p += 1;

            if (query.window_count < MAX_WINDOW_FUNCS) {
                query.window_funcs[query.window_count] = window_expr;
                query.window_count += 1;
            }

            pos.* = p;
            return true;
        }
    }
    return false;
}

fn parseAggregate(sql: []const u8, pos: *usize, query: *ParsedQuery) bool {
    const funcs = [_]struct { name: []const u8, func: AggFunc }{
        .{ .name = "SUM", .func = .sum },
        .{ .name = "COUNT", .func = .count },
        .{ .name = "AVG", .func = .avg },
        .{ .name = "MIN", .func = .min },
        .{ .name = "MAX", .func = .max },
    };

    for (funcs) |f| {
        if (startsWithIC(sql[pos.*..], f.name)) {
            var p = pos.* + f.name.len;
            p = skipWs(sql, p);
            if (p >= sql.len or sql[p] != '(') continue;
            p += 1;
            p = skipWs(sql, p);

            // Get column name
            const col_start = p;
            if (sql[p] == '*') {
                p += 1;
            } else {
                while (p < sql.len and isIdent(sql[p])) p += 1;
            }
            const col_name = sql[col_start..p];

            p = skipWs(sql, p);
            if (p >= sql.len or sql[p] != ')') continue;
            p += 1;

            if (query.agg_count < MAX_AGGREGATES) {
                query.aggregates[query.agg_count] = AggExpr{
                    .func = f.func,
                    .column = col_name,
                };
                query.agg_count += 1;
            }

            pos.* = p;
            return true;
        }
    }
    return false;
}

fn parseWhere(sql: []const u8, pos: *usize) ?WhereClause {
    return parseOrExpr(sql, pos);
}

fn parseOrExpr(sql: []const u8, pos: *usize) ?WhereClause {
    var left = parseAndExpr(sql, pos) orelse return null;

    pos.* = skipWs(sql, pos.*);
    while (startsWithIC(sql[pos.*..], "OR")) {
        pos.* += 2;
        pos.* = skipWs(sql, pos.*);
        const right = parseAndExpr(sql, pos) orelse return null;

        if (where_storage_idx + 2 >= where_storage.len) return null;
        where_storage[where_storage_idx] = left;
        where_storage[where_storage_idx + 1] = right;
        left = WhereClause{
            .op = .or_op,
            .left = &where_storage[where_storage_idx],
            .right = &where_storage[where_storage_idx + 1],
        };
        where_storage_idx += 2;

        pos.* = skipWs(sql, pos.*);
    }

    return left;
}

fn parseAndExpr(sql: []const u8, pos: *usize) ?WhereClause {
    var left = parseComparison(sql, pos) orelse return null;

    pos.* = skipWs(sql, pos.*);
    while (startsWithIC(sql[pos.*..], "AND")) {
        pos.* += 3;
        pos.* = skipWs(sql, pos.*);
        const right = parseComparison(sql, pos) orelse return null;

        if (where_storage_idx + 2 >= where_storage.len) return null;
        where_storage[where_storage_idx] = left;
        where_storage[where_storage_idx + 1] = right;
        left = WhereClause{
            .op = .and_op,
            .left = &where_storage[where_storage_idx],
            .right = &where_storage[where_storage_idx + 1],
        };
        where_storage_idx += 2;

        pos.* = skipWs(sql, pos.*);
    }

    return left;
}

fn parseComparison(sql: []const u8, pos: *usize) ?WhereClause {
    pos.* = skipWs(sql, pos.*);

    // Handle parentheses
    if (pos.* < sql.len and sql[pos.*] == '(') {
        pos.* += 1;
        const inner = parseOrExpr(sql, pos) orelse return null;
        pos.* = skipWs(sql, pos.*);
        if (pos.* < sql.len and sql[pos.*] == ')') pos.* += 1;
        return inner;
    }

    // Column name
    const col_start = pos.*;
    while (pos.* < sql.len and isIdent(sql[pos.*])) pos.* += 1;
    if (pos.* == col_start) return null;
    const column = sql[col_start..pos.*];

    pos.* = skipWs(sql, pos.*);

    // Check for IS NULL / IS NOT NULL
    if (startsWithIC(sql[pos.*..], "IS")) {
        pos.* += 2;
        pos.* = skipWs(sql, pos.*);
        const is_not = startsWithIC(sql[pos.*..], "NOT");
        if (is_not) {
            pos.* += 3;
            pos.* = skipWs(sql, pos.*);
        }
        if (startsWithIC(sql[pos.*..], "NULL")) {
            pos.* += 4;
            return WhereClause{
                .op = if (is_not) .is_not_null else .is_null,
                .column = column,
            };
        }
    }

    // Check for NOT IN, NOT LIKE, NOT BETWEEN
    var is_not = false;
    if (startsWithIC(sql[pos.*..], "NOT")) {
        is_not = true;
        pos.* += 3;
        pos.* = skipWs(sql, pos.*);
    }

    // Check for IN
    if (startsWithIC(sql[pos.*..], "IN")) {
        pos.* += 2;
        pos.* = skipWs(sql, pos.*);
        if (pos.* < sql.len and sql[pos.*] == '(') {
            pos.* += 1;
            var clause = WhereClause{
                .op = if (is_not) .not_in_list else .in_list,
                .column = column,
            };
            // Parse value list
            while (pos.* < sql.len and clause.in_values_count < 32) {
                pos.* = skipWs(sql, pos.*);
                if (sql[pos.*] == ')') {
                    pos.* += 1;
                    break;
                }
                const num_start = pos.*;
                if (sql[pos.*] == '-') pos.* += 1;
                while (pos.* < sql.len and std.ascii.isDigit(sql[pos.*])) pos.* += 1;
                if (pos.* > num_start) {
                    if (std.fmt.parseInt(i64, sql[num_start..pos.*], 10)) |v| {
                        clause.in_values_int[clause.in_values_count] = v;
                        clause.in_values_count += 1;
                    } else |_| {}
                }
                pos.* = skipWs(sql, pos.*);
                if (pos.* < sql.len and sql[pos.*] == ',') pos.* += 1;
            }
            return clause;
        }
    }

    // Check for BETWEEN
    if (startsWithIC(sql[pos.*..], "BETWEEN")) {
        pos.* += 7;
        pos.* = skipWs(sql, pos.*);
        const low_start = pos.*;
        if (sql[pos.*] == '-') pos.* += 1;
        while (pos.* < sql.len and std.ascii.isDigit(sql[pos.*])) pos.* += 1;
        const low_val = std.fmt.parseInt(i64, sql[low_start..pos.*], 10) catch null;
        pos.* = skipWs(sql, pos.*);
        if (startsWithIC(sql[pos.*..], "AND")) pos.* += 3;
        pos.* = skipWs(sql, pos.*);
        const high_start = pos.*;
        if (sql[pos.*] == '-') pos.* += 1;
        while (pos.* < sql.len and std.ascii.isDigit(sql[pos.*])) pos.* += 1;
        const high_val = std.fmt.parseInt(i64, sql[high_start..pos.*], 10) catch null;
        return WhereClause{
            .op = .between,
            .column = column,
            .between_low = low_val,
            .between_high = high_val,
        };
    }

    // Check for LIKE
    if (startsWithIC(sql[pos.*..], "LIKE")) {
        pos.* += 4;
        pos.* = skipWs(sql, pos.*);
        // Parse string pattern
        if (pos.* < sql.len and sql[pos.*] == '\'') {
            pos.* += 1;
            const pat_start = pos.*;
            while (pos.* < sql.len and sql[pos.*] != '\'') pos.* += 1;
            const pattern = sql[pat_start..pos.*];
            if (pos.* < sql.len) pos.* += 1; // skip closing quote
            return WhereClause{
                .op = if (is_not) .not_like else .like,
                .column = column,
                .value_str = pattern,
            };
        }
    }

    // Check for NEAR
    if (startsWithIC(sql[pos.*..], "NEAR")) {
        pos.* += 4;
        pos.* = skipWs(sql, pos.*);
        
        // Parse vector literal [1.0, 2.0, ...]
        if (pos.* < sql.len and sql[pos.*] == '[') {
            pos.* += 1;
            var vec: [MAX_VECTOR_DIM]f32 = undefined;
            var dim: usize = 0;
            
            while (pos.* < sql.len and dim < MAX_VECTOR_DIM) {
                pos.* = skipWs(sql, pos.*);
                if (pos.* < sql.len and sql[pos.*] == ']') {
                    pos.* += 1;
                    break;
                }
                
                // Parse float
                const num_start = pos.*;
                if (sql[pos.*] == '-') pos.* += 1;
                while (pos.* < sql.len and (std.ascii.isDigit(sql[pos.*]) or sql[pos.*] == '.')) pos.* += 1;
                if (pos.* > num_start) {
                    if (std.fmt.parseFloat(f32, sql[num_start..pos.*])) |v| {
                        vec[dim] = v;
                        dim += 1;
                    } else |_| {}
                }
                
                pos.* = skipWs(sql, pos.*);
                if (pos.* < sql.len and sql[pos.*] == ',') pos.* += 1;
            }
            
            return WhereClause{
                .op = .near,
                .column = column,
                .near_vector = vec,
                .near_dim = dim,
            };
        }
    }

    // Standard comparison operators
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

    pos.* = skipWs(sql, pos.*);

    // Value - number or string
    var value_int: ?i64 = null;
    var value_float: ?f64 = null;
    var value_str: ?[]const u8 = null;

    if (pos.* < sql.len and sql[pos.*] == '\'') {
        // String value
        pos.* += 1;
        const str_start = pos.*;
        while (pos.* < sql.len and sql[pos.*] != '\'') pos.* += 1;
        value_str = sql[str_start..pos.*];
        if (pos.* < sql.len) pos.* += 1;
    } else if (pos.* < sql.len and (std.ascii.isDigit(sql[pos.*]) or sql[pos.*] == '-')) {
        const num_start = pos.*;
        if (sql[pos.*] == '-') pos.* += 1;
        while (pos.* < sql.len and (std.ascii.isDigit(sql[pos.*]) or sql[pos.*] == '.')) pos.* += 1;
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
        .value_str = value_str,
    };
}

fn parseGroupBy(sql: []const u8, start: usize, query: *ParsedQuery) usize {
    var pos = start;
    while (pos < sql.len and query.group_by_count < MAX_GROUP_COLS) {
        pos = skipWs(sql, pos);
        const col_start = pos;
        while (pos < sql.len and isIdent(sql[pos])) pos += 1;
        if (pos > col_start) {
            query.group_by_cols[query.group_by_count] = sql[col_start..pos];
            query.group_by_count += 1;
        }
        pos = skipWs(sql, pos);
        if (pos >= sql.len or sql[pos] != ',') break;
        pos += 1;
    }
    return pos;
}

fn parseOrderBy(sql: []const u8, start: usize, query: *ParsedQuery) usize {
    var pos = start;
    pos = skipWs(sql, pos);

    const col_start = pos;
    while (pos < sql.len and isIdent(sql[pos])) pos += 1;
    if (pos > col_start) {
        query.order_by_col = sql[col_start..pos];
    }

    pos = skipWs(sql, pos);
    if (startsWithIC(sql[pos..], "DESC")) {
        query.order_dir = .desc;
        pos += 4;
    } else if (startsWithIC(sql[pos..], "ASC")) {
        query.order_dir = .asc;
        pos += 3;
    }

    return pos;
}

// ============================================================================
// Helpers
// ============================================================================

fn skipWs(sql: []const u8, start: usize) usize {
    var pos = start;
    while (pos < sql.len and std.ascii.isWhitespace(sql[pos])) pos += 1;
    return pos;
}

fn isIdent(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn startsWithIC(haystack: []const u8, needle: []const u8) bool {
    if (haystack.len < needle.len) return false;
    for (haystack[0..needle.len], needle) |h, n| {
        if (std.ascii.toLower(h) != std.ascii.toLower(n)) return false;
    }
    return true;
}

fn allocResultBuffer(size: usize) ?[]u8 {
    const ptr = memory.wasmAlloc(size) orelse return null;
    result_buffer = ptr[0..size];
    result_size = 0;
    return result_buffer;
}

fn writeToResult(data: []const u8) bool {
    const buf = result_buffer orelse return false;
    if (result_size + data.len > buf.len) return false;
    @memcpy(buf[result_size..][0..data.len], data);
    result_size += data.len;
    return true;
}

fn writeU32(value: u32) bool {
    return writeToResult(std.mem.asBytes(&value));
}

fn writeU64(value: u64) bool {
    return writeToResult(std.mem.asBytes(&value));
}

// ============================================================================
// Tests
// ============================================================================

test "parse SELECT *" {
    const query = parseSql("SELECT * FROM users");
    try std.testing.expect(query != null);
    try std.testing.expect(query.?.is_star);
    try std.testing.expectEqualStrings("users", query.?.table_name);
}

test "parse SELECT with WHERE" {
    const query = parseSql("SELECT * FROM orders WHERE id = 42");
    try std.testing.expect(query != null);
    try std.testing.expect(query.?.where_clause != null);
    try std.testing.expectEqual(@as(i64, 42), query.?.where_clause.?.value_int.?);
}

test "parse SELECT with aggregates" {
    const query = parseSql("SELECT SUM(amount), COUNT(*) FROM orders");
    try std.testing.expect(query != null);
    try std.testing.expectEqual(@as(usize, 2), query.?.agg_count);
    try std.testing.expectEqual(AggFunc.sum, query.?.aggregates[0].func);
    try std.testing.expectEqual(AggFunc.count, query.?.aggregates[1].func);
}

test "parse SELECT with GROUP BY" {
    const query = parseSql("SELECT category, SUM(amount) FROM orders GROUP BY category");
    try std.testing.expect(query != null);
    try std.testing.expectEqual(@as(usize, 1), query.?.group_by_count);
    try std.testing.expectEqualStrings("category", query.?.group_by_cols[0]);
}

test "parse SELECT with ORDER BY" {
    const query = parseSql("SELECT * FROM users ORDER BY name DESC LIMIT 10");
    try std.testing.expect(query != null);
    try std.testing.expect(query.?.order_by_col != null);
    try std.testing.expectEqual(OrderDir.desc, query.?.order_dir);
    try std.testing.expectEqual(@as(u32, 10), query.?.limit_value.?);
}

test "parse WHERE with AND/OR" {
    const query = parseSql("SELECT * FROM users WHERE age > 18 AND status = 1 OR admin = 1");
    try std.testing.expect(query != null);
    try std.testing.expect(query.?.where_clause != null);
}
