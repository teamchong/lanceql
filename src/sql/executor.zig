//! SQL Executor - Execute parsed SQL queries against Lance and Parquet files
//!
//! Takes a parsed AST and executes it against Lance or Parquet columnar files,
//! returning results in columnar format compatible with better-sqlite3.

const std = @import("std");
const ast = @import("ast");
const Table = @import("lanceql.table").Table;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const DeltaTable = @import("lanceql.delta_table").DeltaTable;
const IcebergTable = @import("lanceql.iceberg_table").IcebergTable;
const ArrowTable = @import("lanceql.arrow_table").ArrowTable;
const AvroTable = @import("lanceql.avro_table").AvroTable;
const OrcTable = @import("lanceql.orc_table").OrcTable;
const XlsxTable = @import("lanceql.xlsx_table").XlsxTable;
const hash = @import("lanceql.hash");
pub const logic_table_dispatch = @import("logic_table_dispatch.zig");
pub const scalar_functions = @import("scalar_functions.zig");
pub const aggregate_functions = @import("aggregate_functions.zig");
pub const window_functions = @import("window_functions.zig");
pub const result_types = @import("result_types.zig");
pub const result_ops = @import("result_ops.zig");
pub const having_eval = @import("having_eval.zig");
pub const set_ops = @import("set_ops.zig");
pub const window_eval = @import("window_eval.zig");
pub const where_eval = @import("where_eval.zig");
pub const expr_eval = @import("expr_eval.zig");

const Expr = ast.Expr;
const SelectStmt = ast.SelectStmt;
const Value = ast.Value;
const BinaryOp = ast.BinaryOp;

/// Aggregate types (re-exported from aggregate_functions module)
pub const AggregateType = aggregate_functions.AggregateType;
pub const Accumulator = aggregate_functions.Accumulator;
pub const PercentileAccumulator = aggregate_functions.PercentileAccumulator;

/// Result types (re-exported from result_types module)
pub const Result = result_types.Result;
pub const CachedColumn = result_types.CachedColumn;
pub const JoinedData = result_types.JoinedData;
pub const TableSource = result_types.TableSource;
pub const LanceColumnType = result_types.LanceColumnType;

/// SQL Query Executor
pub const Executor = struct {
    /// Default table (used when FROM is a simple table name or not specified)
    table: ?*Table,
    /// Parquet table (alternative to Lance table)
    parquet_table: ?*ParquetTable = null,
    /// Delta table (alternative to Lance table)
    delta_table: ?*DeltaTable = null,
    /// Iceberg table (alternative to Lance table)
    iceberg_table: ?*IcebergTable = null,
    /// Arrow IPC table (alternative to Lance table)
    arrow_table: ?*ArrowTable = null,
    /// Avro table (alternative to Lance table)
    avro_table: ?*AvroTable = null,
    /// ORC table (alternative to Lance table)
    orc_table: ?*OrcTable = null,
    /// XLSX table (alternative to Lance table)
    xlsx_table: ?*XlsxTable = null,
    allocator: std.mem.Allocator,
    column_cache: std.StringHashMap(CachedColumn),
    /// Optional dispatcher for @logic_table method calls
    dispatcher: ?*logic_table_dispatch.Dispatcher = null,
    /// Maps table alias to class name for @logic_table instances
    logic_table_aliases: std.StringHashMap([]const u8),
    /// Currently active table source (set during execute)
    active_source: ?TableSource = null,
    /// Registered tables by name (for JOINs and multi-table queries)
    tables: std.StringHashMap(*Table),
    /// Cache for @logic_table method batch results
    /// Key: "ClassName.methodName", Value: results array
    method_results_cache: std.StringHashMap([]const f64),

    const Self = @This();

    pub fn init(table: ?*Table, allocator: std.mem.Allocator) Self {
        return .{
            .table = table,
            .parquet_table = null,
            .delta_table = null,
            .iceberg_table = null,
            .arrow_table = null,
            .avro_table = null,
            .orc_table = null,
            .xlsx_table = null,
            .allocator = allocator,
            .column_cache = std.StringHashMap(CachedColumn).init(allocator),
            .dispatcher = null,
            .logic_table_aliases = std.StringHashMap([]const u8).init(allocator),
            .active_source = null,
            .tables = std.StringHashMap(*Table).init(allocator),
            .method_results_cache = std.StringHashMap([]const f64).init(allocator),
        };
    }

    /// Initialize executor with a Parquet table
    pub fn initWithParquet(parquet_table: *ParquetTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.parquet_table = parquet_table;
        return self;
    }

    /// Initialize executor with a Delta table
    pub fn initWithDelta(delta_table: *DeltaTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.delta_table = delta_table;
        return self;
    }

    /// Initialize executor with an Iceberg table
    pub fn initWithIceberg(iceberg_table: *IcebergTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.iceberg_table = iceberg_table;
        return self;
    }

    /// Initialize executor with an Arrow IPC table
    pub fn initWithArrow(arrow_table: *ArrowTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.arrow_table = arrow_table;
        return self;
    }

    /// Initialize executor with an Avro table
    pub fn initWithAvro(avro_table: *AvroTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.avro_table = avro_table;
        return self;
    }

    /// Initialize executor with an ORC table
    pub fn initWithOrc(orc_table: *OrcTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.orc_table = orc_table;
        return self;
    }

    /// Initialize executor with an XLSX table
    pub fn initWithXlsx(xlsx_table: *XlsxTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.xlsx_table = xlsx_table;
        return self;
    }

    /// Register a table by name for use in JOINs and multi-table queries
    pub fn registerTable(self: *Self, name: []const u8, table: *Table) !void {
        try self.tables.put(name, table);
    }

    /// Get a registered table by name
    pub fn getRegisteredTable(self: *Self, name: []const u8) ?*Table {
        return self.tables.get(name);
    }

    /// Initialize with a table (convenience for existing code)
    pub fn initWithTable(table: *Table, allocator: std.mem.Allocator) Self {
        return init(table, allocator);
    }

    /// Get the table (must be set before calling this)
    /// This is used by internal methods that expect a table to be configured.
    inline fn tbl(self: *Self) *Table {
        return self.table orelse unreachable;
    }

    /// Check if operating in Parquet mode
    inline fn isParquetMode(self: *Self) bool {
        return self.parquet_table != null;
    }

    /// Check if operating in Delta mode
    inline fn isDeltaMode(self: *Self) bool {
        return self.delta_table != null;
    }

    /// Check if operating in Iceberg mode
    inline fn isIcebergMode(self: *Self) bool {
        return self.iceberg_table != null;
    }

    /// Check if operating in Arrow mode
    inline fn isArrowMode(self: *Self) bool {
        return self.arrow_table != null;
    }

    /// Check if operating in Avro mode
    inline fn isAvroMode(self: *Self) bool {
        return self.avro_table != null;
    }

    /// Check if operating in ORC mode
    inline fn isOrcMode(self: *Self) bool {
        return self.orc_table != null;
    }

    /// Check if operating in XLSX mode
    inline fn isXlsxMode(self: *Self) bool {
        return self.xlsx_table != null;
    }

    /// Field names of typed table pointers for comptime iteration
    const typed_table_fields = .{ "parquet_table", "delta_table", "iceberg_table", "arrow_table", "avro_table", "orc_table", "xlsx_table" };

    /// Check if any typed table is set (non-Lance mode)
    inline fn hasTypedTable(self: *Self) bool {
        inline for (typed_table_fields) |field| {
            if (@field(self, field) != null) return true;
        }
        return false;
    }

    /// Get row count (works with Lance, Parquet, Delta, Iceberg, Arrow, Avro, ORC, or XLSX)
    fn getRowCount(self: *Self) !usize {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| return t.numRows();
        }
        return try self.tbl().rowCount(0);
    }

    /// Get column names (works with Lance, Parquet, Delta, Iceberg, Arrow, Avro, ORC, or XLSX)
    fn getColumnNames(self: *Self) ![][]const u8 {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| return t.getColumnNames();
        }
        return try self.tbl().columnNames();
    }

    /// Get physical column ID by name (works with Lance, Parquet, Delta, Iceberg, Arrow, Avro, ORC, or XLSX)
    fn getPhysicalColumnId(self: *Self, name: []const u8) ?u32 {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                return if (t.columnIndex(name)) |idx| @intCast(idx) else null;
            }
        }
        return self.tbl().physicalColumnId(name);
    }

    /// Filter array by indices - comptime generic for DRY
    fn filterByIndices(self: *Self, comptime T: type, all_data: []const T, indices: []const u32) ![]T {
        const filtered = try self.allocator.alloc(T, indices.len);
        for (indices, 0..) |idx, i| filtered[i] = all_data[idx];
        return filtered;
    }

    /// Method call wrapper conforming to MethodCallFn signature
    /// Bridges expr_eval to executor's evaluateMethodCall
    fn methodCallWrapper(ctx: *anyopaque, expr: *const Expr, row_idx: u32) anyerror!Value {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return switch (expr.*) {
            .method_call => |mc| self.evaluateMethodCall(mc, row_idx),
            else => error.NotAMethodCall,
        };
    }

    /// Build ExprContext for expr_eval module
    fn buildExprContext(self: *Self) expr_eval.ExprContext {
        return .{
            .allocator = self.allocator,
            .column_cache = &self.column_cache,
            .method_ctx = self,
            .method_call_fn = methodCallWrapper,
        };
    }

    /// Set the dispatcher for @logic_table method calls
    pub fn setDispatcher(self: *Self, dispatcher: *logic_table_dispatch.Dispatcher) void {
        self.dispatcher = dispatcher;
    }

    /// Register a @logic_table alias with its class name
    /// Returns error.DuplicateAlias if alias already registered
    pub fn registerLogicTableAlias(self: *Self, alias: []const u8, class_name: []const u8) !void {
        // Check for existing alias first - we don't support overwriting
        if (self.logic_table_aliases.contains(alias)) {
            return error.DuplicateAlias;
        }

        const alias_copy = try self.allocator.dupe(u8, alias);
        errdefer self.allocator.free(alias_copy);
        const class_copy = try self.allocator.dupe(u8, class_name);
        errdefer self.allocator.free(class_copy);
        try self.logic_table_aliases.put(alias_copy, class_copy);
    }

    pub fn deinit(self: *Self) void {
        // Free cached columns
        var iter = self.column_cache.valueIterator();
        while (iter.next()) |col| {
            col.free(self.allocator);
        }
        self.column_cache.deinit();

        // Free logic_table alias keys and values
        var alias_iter = self.logic_table_aliases.iterator();
        while (alias_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.logic_table_aliases.deinit();

        // Free method results cache
        var results_iter = self.method_results_cache.iterator();
        while (results_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.method_results_cache.deinit();

        // Clean up registered tables map (tables are owned by caller, just deinit the map)
        self.tables.deinit();
    }

    // ========================================================================
    // Column Preloading (for WHERE clause optimization)
    // ========================================================================

    /// Extract all column names referenced in an expression
    fn extractColumnNames(self: *Self, expr: *const Expr, list: *std.ArrayList([]const u8)) anyerror!void {
        switch (expr.*) {
            .column => |col| {
                try list.append(self.allocator, col.name);
            },
            .binary => |bin| {
                try self.extractColumnNames(bin.left, list);
                try self.extractColumnNames(bin.right, list);
            },
            .unary => |un| {
                try self.extractColumnNames(un.operand, list);
            },
            .in_list => |in| {
                try self.extractColumnNames(in.expr, list);
                for (in.values) |*val| {
                    try self.extractColumnNames(val, list);
                }
            },
            .in_subquery => |in| {
                try self.extractColumnNames(in.expr, list);
                // Don't extract from subquery - it has its own scope
            },
            .call => |call| {
                for (call.args) |*arg| {
                    try self.extractColumnNames(arg, list);
                }
            },
            else => {},
        }
    }

    /// Preload columns into cache
    fn preloadColumns(self: *Self, col_names: []const []const u8) !void {
        // Use generic preloader for typed tables
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| return self.preloadColumnsFromTable(t, col_names);
        }

        for (col_names) |name| {
            if (self.column_cache.contains(name)) continue;

            const physical_col_id = self.tbl().physicalColumnId(name) orelse return error.ColumnNotFound;
            const fld = self.tbl().getFieldById(physical_col_id) orelse return error.InvalidColumn;
            const col_type = LanceColumnType.fromLogicalType(fld.logical_type);

            const cached = switch (col_type) {
                .timestamp_ns => CachedColumn{ .timestamp_ns = try self.tbl().readInt64Column(physical_col_id) },
                .timestamp_us => CachedColumn{ .timestamp_us = try self.tbl().readInt64Column(physical_col_id) },
                .timestamp_ms => CachedColumn{ .timestamp_ms = try self.tbl().readInt64Column(physical_col_id) },
                .timestamp_s => CachedColumn{ .timestamp_s = try self.tbl().readInt64Column(physical_col_id) },
                .date32 => CachedColumn{ .date32 = try self.tbl().readInt32Column(physical_col_id) },
                .date64 => CachedColumn{ .date64 = try self.tbl().readInt64Column(physical_col_id) },
                .int32 => CachedColumn{ .int32 = try self.tbl().readInt32Column(physical_col_id) },
                .float32 => CachedColumn{ .float32 = try self.tbl().readFloat32Column(physical_col_id) },
                .bool_ => CachedColumn{ .bool_ = try self.tbl().readBoolColumn(physical_col_id) },
                .int64 => CachedColumn{ .int64 = try self.tbl().readInt64Column(physical_col_id) },
                .float64 => CachedColumn{ .float64 = try self.tbl().readFloat64Column(physical_col_id) },
                .string => CachedColumn{ .string = try self.tbl().readStringColumn(physical_col_id) },
                .unsupported => return error.UnsupportedColumnType,
            };
            try self.column_cache.put(name, cached);
        }
    }

    /// Generic column preloader for any table type with standard interface
    fn preloadColumnsFromTable(self: *Self, table: anytype, col_names: []const []const u8) !void {
        const T = @TypeOf(table.*);
        const is_xlsx = T == XlsxTable;

        for (col_names) |name| {
            if (self.column_cache.contains(name)) continue;

            const col_idx = table.columnIndex(name) orelse return error.ColumnNotFound;
            const col_type = table.getColumnType(col_idx) orelse return error.InvalidColumn;

            switch (col_type) {
                .int64 => {
                    const data = table.readInt64Column(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .int64 = data });
                },
                .int32 => {
                    const data = table.readInt32Column(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .int32 = data });
                },
                .double => {
                    const data = table.readFloat64Column(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .float64 = data });
                },
                .float => {
                    const data = table.readFloat32Column(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .float32 = data });
                },
                .boolean => {
                    const data = table.readBoolColumn(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .bool_ = data });
                },
                .byte_array, .fixed_len_byte_array => {
                    const data = table.readStringColumn(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .string = data });
                },
                else => {
                    // XLSX defaults to float64, others error
                    if (is_xlsx) {
                        const data = table.readFloat64Column(col_idx) catch return error.ColumnReadError;
                        try self.column_cache.put(name, CachedColumn{ .float64 = data });
                    } else {
                        return error.UnsupportedColumnType;
                    }
                },
            }
        }
    }

    /// Get the currently active table for query execution
    fn getActiveTable(self: *Self) !*Table {
        if (self.active_source) |source| {
            return source.getTable();
        }
        return self.table orelse error.NoTableConfigured;
    }

    /// Resolve FROM clause to get the table source
    fn resolveTableSource(self: *Self, from: *const ast.TableRef) anyerror!TableSource {
        switch (from.*) {
            .simple => |simple| {
                // First check if table is registered by name
                if (self.tables.get(simple.name)) |registered_table| {
                    return .{ .direct = registered_table };
                }
                // Otherwise use the default table
                const direct_table = self.table orelse return error.NoTableConfigured;
                return .{ .direct = direct_table };
            },
            .function => |func| {
                // Table-valued function (e.g., logic_table('fraud.py'))
                if (std.mem.eql(u8, func.func.name, "logic_table")) {
                    // Extract file path from first argument
                    if (func.func.args.len == 0) {
                        return error.LogicTableRequiresPath;
                    }

                    const path_arg = func.func.args[0];
                    const path = switch (path_arg) {
                        .value => |val| switch (val) {
                            .string => |s| s,
                            else => return error.LogicTablePathMustBeString,
                        },
                        else => return error.LogicTablePathMustBeString,
                    };

                    // Create LogicTableExecutor from file path (heap allocated)
                    const executor = try self.allocator.create(logic_table_dispatch.LogicTableExecutor);
                    errdefer self.allocator.destroy(executor);
                    executor.* = try logic_table_dispatch.LogicTableExecutor.init(self.allocator, path);
                    errdefer executor.deinit();

                    // Load tables referenced in the Python file
                    try executor.loadTables();

                    // Get primary table (first loaded table)
                    const primary_table = executor.getPrimaryTable() orelse {
                        executor.deinit();
                        self.allocator.destroy(executor);
                        return error.NoTablesInLogicTable;
                    };

                    // Register alias for method dispatch
                    if (func.alias) |alias| {
                        try self.registerLogicTableAlias(alias, executor.class_name);
                    }

                    return .{ .logic_table = .{
                        .executor = executor,
                        .primary_table = primary_table,
                        .alias = func.alias,
                    } };
                }
                return error.UnsupportedTableFunction;
            },
            .join => |join| {
                // Execute JOIN by resolving both sides and performing hash join
                return try self.executeJoin(join.left, &join.join_clause);
            },
        }
    }

    /// Execute a JOIN operation using hash join algorithm
    fn executeJoin(self: *Self, left_ref: *const ast.TableRef, join_clause: *const ast.JoinClause) !TableSource {
        // 1. Resolve left table
        var left_source = try self.resolveTableSource(left_ref);
        errdefer self.releaseTableSource(&left_source);
        const left_table = left_source.getTable();

        // 2. Resolve right table
        var right_source = try self.resolveTableSource(join_clause.table);
        defer self.releaseTableSource(&right_source);
        const right_table = right_source.getTable();

        // 3. Extract join key column names from ON condition
        const join_keys = try self.extractJoinKeys(join_clause.on_condition orelse return error.JoinRequiresOnCondition);

        // 4. Get join key columns from both tables
        const left_key_col_idx = left_table.physicalColumnId(join_keys.left_col) orelse return error.JoinColumnNotFound;
        const right_key_col_idx = right_table.physicalColumnId(join_keys.right_col) orelse return error.JoinColumnNotFound;

        // 5. Read join key data
        const left_key_data = try self.readJoinKeyColumn(left_table, left_key_col_idx);
        defer self.freeJoinKeyData(left_key_data);

        const right_key_data = try self.readJoinKeyColumn(right_table, right_key_col_idx);
        defer self.freeJoinKeyData(right_key_data);

        // 6. Build hash table from right table (build phase)
        var hash_table = std.StringHashMap(std.ArrayListUnmanaged(usize)).init(self.allocator);
        defer {
            var iter = hash_table.iterator();
            while (iter.next()) |entry| {
                // Free the key (we allocated it during insertion)
                self.allocator.free(entry.key_ptr.*);
                // Deinit the ArrayList value
                entry.value_ptr.deinit(self.allocator);
            }
            hash_table.deinit();
        }

        for (0..right_key_data.len()) |idx| {
            const key = try self.joinKeyToString(right_key_data, idx);
            defer self.allocator.free(key);

            const result = try hash_table.getOrPut(key);
            if (!result.found_existing) {
                const key_copy = try self.allocator.dupe(u8, key);
                result.key_ptr.* = key_copy;
                result.value_ptr.* = .{};
            }
            try result.value_ptr.append(self.allocator, idx);
        }

        // 7. Probe phase - find matching rows
        var left_indices = std.ArrayListUnmanaged(usize){};
        defer left_indices.deinit(self.allocator);
        var right_indices = std.ArrayListUnmanaged(usize){};
        defer right_indices.deinit(self.allocator);

        // Track matched rows for outer joins
        var matched_right = std.AutoHashMap(usize, void).init(self.allocator);
        defer matched_right.deinit();

        for (0..left_key_data.len()) |left_idx| {
            const key = try self.joinKeyToString(left_key_data, left_idx);
            defer self.allocator.free(key);

            if (hash_table.get(key)) |right_list| {
                for (right_list.items) |right_idx| {
                    try left_indices.append(self.allocator, left_idx);
                    try right_indices.append(self.allocator, right_idx);
                    try matched_right.put(right_idx, {});
                }
            } else if (join_clause.join_type == .left or join_clause.join_type == .full) {
                // LEFT/FULL JOIN: include left row with NULL for right
                try left_indices.append(self.allocator, left_idx);
                try right_indices.append(self.allocator, std.math.maxInt(usize)); // Sentinel for NULL
            }
        }

        // For RIGHT/FULL JOIN: add unmatched right rows
        if (join_clause.join_type == .right or join_clause.join_type == .full) {
            for (0..right_key_data.len()) |right_idx| {
                if (!matched_right.contains(right_idx)) {
                    try left_indices.append(self.allocator, std.math.maxInt(usize)); // Sentinel for NULL
                    try right_indices.append(self.allocator, right_idx);
                }
            }
        }

        // 8. Build joined result with all columns from both tables
        const joined_data = try self.allocator.create(JoinedData);
        errdefer self.allocator.destroy(joined_data);

        joined_data.* = JoinedData{
            .columns = std.StringHashMap(CachedColumn).init(self.allocator),
            .column_names = &[_][]const u8{},
            .row_count = left_indices.items.len,
            .allocator = self.allocator,
            .left_table = left_table,
        };
        errdefer joined_data.deinit();

        // Build column names list and copy data
        var col_names = std.ArrayListUnmanaged([]const u8){};
        errdefer {
            for (col_names.items) |name| {
                self.allocator.free(name);
            }
            col_names.deinit(self.allocator);
        }

        // Add left table columns (with table alias prefix if available)
        const left_alias = switch (left_ref.*) {
            .simple => |s| s.alias orelse s.name,
            else => "left",
        };

        try self.addJoinedColumns(
            left_table,
            left_alias,
            left_indices.items,
            joined_data,
            &col_names,
            false, // isRightSide
        );

        // Add right table columns
        const right_alias = switch (join_clause.table.*) {
            .simple => |s| s.alias orelse s.name,
            else => "right",
        };

        try self.addJoinedColumns(
            right_table,
            right_alias,
            right_indices.items,
            joined_data,
            &col_names,
            true, // isRightSide
        );

        joined_data.column_names = try col_names.toOwnedSlice(self.allocator);

        // Release left_source ownership since it's now managed by joined_data
        // (we keep left_table pointer in joined_data.left_table)
        switch (left_source) {
            .direct => {}, // Nothing to release
            .logic_table => |*lt| {
                lt.executor.deinit();
                self.allocator.destroy(lt.executor);
            },
            .joined => |jd| {
                jd.deinit();
                self.allocator.destroy(jd);
            },
        }

        return .{ .joined = joined_data };
    }

    /// Extract left and right column names from JOIN ON condition
    fn extractJoinKeys(self: *Self, condition: ast.Expr) !struct { left_col: []const u8, right_col: []const u8 } {
        _ = self;
        // ON condition should be: left.col = right.col
        switch (condition) {
            .binary => |bin| {
                if (bin.op != .eq) return error.JoinConditionMustBeEquality;

                const left_col = switch (bin.left.*) {
                    .column => |col| col.name, // column.table is optional qualifier
                    else => return error.JoinConditionMustBeColumn,
                };

                const right_col = switch (bin.right.*) {
                    .column => |col| col.name,
                    else => return error.JoinConditionMustBeColumn,
                };

                return .{ .left_col = left_col, .right_col = right_col };
            },
            else => return error.JoinConditionMustBeBinary,
        }
    }

    /// Join key data union for different column types
    const JoinKeyData = union(enum) {
        int64: []i64,
        int32: []i32,
        float64: []f64,
        string: [][]const u8,

        fn len(self: JoinKeyData) usize {
            return switch (self) {
                .int64 => |d| d.len,
                .int32 => |d| d.len,
                .float64 => |d| d.len,
                .string => |d| d.len,
            };
        }
    };

    /// Read join key column data
    fn readJoinKeyColumn(self: *Self, table: *Table, col_idx: u32) !JoinKeyData {
        _ = self;
        const fld = table.getFieldById(col_idx) orelse return error.InvalidColumn;
        const col_type = LanceColumnType.fromLogicalType(fld.logical_type);

        return switch (col_type) {
            .int64, .timestamp_ns, .timestamp_us, .timestamp_ms, .timestamp_s, .date64 => .{ .int64 = try table.readInt64Column(col_idx) },
            .int32, .date32 => .{ .int32 = try table.readInt32Column(col_idx) },
            .float32, .float64 => .{ .float64 = try table.readFloat64Column(col_idx) },
            .string => .{ .string = try table.readStringColumn(col_idx) },
            .bool_, .unsupported => error.UnsupportedJoinKeyType,
        };
    }

    /// Free join key data
    fn freeJoinKeyData(self: *Self, data: JoinKeyData) void {
        switch (data) {
            .int64 => |d| self.allocator.free(d),
            .int32 => |d| self.allocator.free(d),
            .float64 => |d| self.allocator.free(d),
            .string => |d| {
                for (d) |s| self.allocator.free(s);
                self.allocator.free(d);
            },
        }
    }

    /// Convert join key value at index to string for hashing
    fn joinKeyToString(self: *Self, data: JoinKeyData, idx: usize) ![]u8 {
        var buf: [64]u8 = undefined;
        const result = switch (data) {
            .int64 => |d| std.fmt.bufPrint(&buf, "{d}", .{d[idx]}),
            .int32 => |d| std.fmt.bufPrint(&buf, "{d}", .{d[idx]}),
            .float64 => |d| std.fmt.bufPrint(&buf, "{d:.10}", .{d[idx]}),
            .string => |d| return try self.allocator.dupe(u8, d[idx]),
        };
        return try self.allocator.dupe(u8, result catch return error.FormatError);
    }

    /// Add columns from a table to the joined result
    fn addJoinedColumns(
        self: *Self,
        table: *Table,
        alias: []const u8,
        row_indices: []const usize,
        joined_data: *JoinedData,
        col_names: *std.ArrayListUnmanaged([]const u8),
        is_right_side: bool,
    ) !void {
        const schema = table.getSchema() orelse return error.NoSchema;

        for (schema.fields) |field| {
            if (field.id < 0) continue;
            const col_idx: u32 = @intCast(field.id);

            // Create qualified column name: "alias.column"
            const qualified_name = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ alias, field.name });
            errdefer self.allocator.free(qualified_name);

            // Read and filter column data based on row indices
            const col_data = try self.readJoinedColumnData(table, col_idx, row_indices, is_right_side);

            try joined_data.columns.put(qualified_name, col_data);
            try col_names.append(self.allocator, qualified_name);
        }
    }

    /// Read column data for joined rows (handles NULL for outer joins)
    fn readJoinedColumnData(
        self: *Self,
        table: *Table,
        col_idx: u32,
        row_indices: []const usize,
        is_right_side: bool,
    ) !CachedColumn {
        _ = is_right_side;
        const fld = table.getFieldById(col_idx) orelse return error.InvalidColumn;
        const col_type = LanceColumnType.fromLogicalType(fld.logical_type);
        const null_idx = std.math.maxInt(usize);

        return switch (col_type) {
            .int64, .int32, .timestamp_ns, .timestamp_us, .timestamp_ms, .timestamp_s, .date32, .date64 => blk: {
                const all_data = try table.readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                const result = try self.allocator.alloc(i64, row_indices.len);
                for (row_indices, 0..) |idx, i| result[i] = if (idx == null_idx) 0 else all_data[idx];
                break :blk .{ .int64 = result };
            },
            .float32, .float64 => blk: {
                const all_data = try table.readFloat64Column(col_idx);
                defer self.allocator.free(all_data);
                const result = try self.allocator.alloc(f64, row_indices.len);
                for (row_indices, 0..) |idx, i| result[i] = if (idx == null_idx) std.math.nan(f64) else all_data[idx];
                break :blk .{ .float64 = result };
            },
            .string => blk: {
                const all_data = try table.readStringColumn(col_idx);
                defer {
                    for (all_data) |s| self.allocator.free(s);
                    self.allocator.free(all_data);
                }
                const result = try self.allocator.alloc([]const u8, row_indices.len);
                for (row_indices, 0..) |idx, i| result[i] = try self.allocator.dupe(u8, if (idx == null_idx) "" else all_data[idx]);
                break :blk .{ .string = result };
            },
            .bool_, .unsupported => error.UnsupportedColumnType,
        };
    }

    /// Release resources associated with a table source
    fn releaseTableSource(self: *Self, source: *TableSource) void {
        switch (source.*) {
            .direct => {
                // Nothing to release - table is managed externally
            },
            .logic_table => |*lt| {
                // Clean up executor and free heap allocation
                lt.executor.deinit();
                self.allocator.destroy(lt.executor);
            },
            .joined => |jd| {
                // Clean up joined data
                jd.deinit();
                self.allocator.destroy(jd);
            },
        }
    }

    /// Execute a SELECT statement
    pub fn execute(self: *Self, stmt: *const SelectStmt, params: []const Value) !Result {
        // Check if we have a typed table (skip FROM clause resolution)
        const has_typed_table = self.hasTypedTable();

        // 0. Resolve FROM clause to get table source (only for Lance tables)
        var source: ?TableSource = null;
        var original_table: ?*Table = null;
        if (!has_typed_table) {
            source = try self.resolveTableSource(&stmt.from);
            self.active_source = source;
            original_table = self.table;
            self.table = source.?.getTable();
        }
        defer {
            if (source) |*s| self.releaseTableSource(s);
            if (!has_typed_table) {
                self.active_source = null;
                self.table = original_table;
            }
        }

        // 1. Preload columns referenced in WHERE clause
        if (stmt.where) |where_expr| {
            var col_names = std.ArrayList([]const u8){};
            defer col_names.deinit(self.allocator);

            try self.extractColumnNames(&where_expr, &col_names);
            try self.preloadColumns(col_names.items);
        }

        // 2. Apply WHERE clause to get filtered row indices
        const indices = if (stmt.where) |where_expr|
            try self.evaluateWhere(&where_expr, params)
        else
            try self.getAllIndices();

        defer self.allocator.free(indices);

        // 3. Check if we need GROUP BY processing
        const has_group_by = stmt.group_by != null;
        const has_aggregates = self.hasAggregates(stmt.columns);

        if (has_group_by or has_aggregates) {
            // Execute with GROUP BY / aggregation
            return self.executeWithGroupBy(stmt, indices);
        }

        // 4. Read columns based on SELECT list (non-aggregate path)
        // Window function columns are handled separately
        var columns_list = std.ArrayList(Result.Column){};
        errdefer {
            for (columns_list.items) |*col| {
                self.freeColumnData(&col.data);
            }
            columns_list.deinit(self.allocator);
        }

        const base_columns = try self.readColumns(stmt.columns, indices);
        defer self.allocator.free(base_columns);
        try columns_list.appendSlice(self.allocator, base_columns);

        // 4.5. Evaluate window functions if present
        if (self.hasWindowFunctions(stmt.columns)) {
            try self.evaluateWindowFunctions(&columns_list, stmt.columns, indices);
        }

        var columns = try columns_list.toOwnedSlice(self.allocator);
        var row_count = indices.len;

        // 5. Apply DISTINCT if specified
        if (stmt.distinct) {
            const distinct_result = try result_ops.applyDistinct(self.allocator, columns);
            columns = distinct_result.columns;
            row_count = distinct_result.row_count;
        }

        // 6. Apply ORDER BY (in-memory sorting)
        if (stmt.order_by) |order_by| {
            try result_ops.applyOrderBy(self.allocator, columns, order_by);
        }

        // 7. Apply LIMIT and OFFSET
        const final_row_count = result_ops.applyLimitOffset(self.allocator, columns, stmt.limit, stmt.offset);

        var result = Result{
            .columns = columns,
            .row_count = final_row_count,
            .allocator = self.allocator,
        };

        // 8. Apply set operation (UNION/INTERSECT/EXCEPT) if present
        if (stmt.set_operation) |set_op| {
            result = try self.executeSetOperation(result, set_op, params);
        }

        return result;
    }

    // ========================================================================
    // Set Operation Execution (UNION/INTERSECT/EXCEPT)
    // ========================================================================

    /// Execute a set operation (UNION, INTERSECT, EXCEPT) between two result sets
    /// Note: Takes ownership of left result and frees it after use
    fn executeSetOperation(self: *Self, left_in: Result, set_op_def: ast.SetOperation, params: []const Value) anyerror!Result {
        var left = left_in;
        defer left.deinit();

        // Execute the right-hand SELECT
        var right = try self.execute(set_op_def.right, params);
        defer right.deinit();

        // Verify column count matches
        if (left.columns.len != right.columns.len) {
            return error.SetOperationColumnMismatch;
        }

        return switch (set_op_def.op_type) {
            .union_all => try set_ops.executeUnionAll(self.allocator, left, right),
            .union_distinct => try set_ops.executeUnionDistinct(self.allocator, left, right),
            .intersect => try set_ops.executeIntersect(self.allocator, left, right),
            .except => try set_ops.executeExcept(self.allocator, left, right),
        };
    }

    // ========================================================================
    // GROUP BY / Aggregate Execution
    // ========================================================================

    /// Check if SELECT list contains any aggregate functions
    fn hasAggregates(self: *Self, select_list: []const ast.SelectItem) bool {
        _ = self;
        return aggregate_functions.hasAggregates(select_list);
    }

    /// Recursively check if expression contains an aggregate function
    fn containsAggregate(expr: *const Expr) bool {
        return aggregate_functions.containsAggregate(expr);
    }

    /// Check if function name is an aggregate function
    fn isAggregateFunction(name: []const u8) bool {
        return aggregate_functions.isAggregateFunction(name);
    }

    // ========================================================================
    // Window Function Support
    // ========================================================================

    /// Window function types (re-exported from window_functions module)
    const WindowFunctionType = window_functions.WindowFunctionType;

    /// Check if expression is a window function (has OVER clause)
    fn isWindowFunction(expr: *const Expr) bool {
        return window_functions.isWindowFunction(expr);
    }

    /// Check if SELECT list contains any window functions
    fn hasWindowFunctions(self: *Self, select_list: []const ast.SelectItem) bool {
        _ = self;
        return window_functions.hasWindowFunctions(select_list);
    }

    /// Parse window function name
    fn parseWindowFunctionType(name: []const u8) ?WindowFunctionType {
        return window_functions.parseWindowFunctionType(name);
    }

    /// Evaluate window functions and add result columns
    /// Window functions are evaluated after all base columns are computed
    fn evaluateWindowFunctions(
        self: *Self,
        columns: *std.ArrayList(Result.Column),
        select_list: []const ast.SelectItem,
        indices: []const u32,
    ) !void {
        for (select_list) |item| {
            if (!isWindowFunction(&item.expr)) continue;

            const call = item.expr.call;
            const window_spec = call.window.?;

            // Preload columns needed for window function
            try self.preloadWindowColumns(window_spec, call.args);

            // Create window context
            const ctx = window_eval.WindowContext{
                .allocator = self.allocator,
                .column_cache = &self.column_cache,
            };

            // Evaluate window function
            const results = try window_eval.evaluateWindowFunction(ctx, call, window_spec, indices);

            // Add result column
            const col_name = item.alias orelse call.name;
            try columns.append(self.allocator, Result.Column{
                .name = col_name,
                .data = Result.ColumnData{ .int64 = results },
            });
        }
    }

    /// Preload columns needed for window function evaluation
    fn preloadWindowColumns(self: *Self, window_spec: *const ast.WindowSpec, args: []const Expr) !void {
        var col_names = std.ArrayList([]const u8){};
        defer col_names.deinit(self.allocator);

        // Add PARTITION BY columns
        if (window_spec.partition_by) |partition_cols| {
            for (partition_cols) |col| {
                try col_names.append(self.allocator, col);
            }
        }

        // Add ORDER BY columns
        if (window_spec.order_by) |order_by| {
            for (order_by) |ob| {
                try col_names.append(self.allocator, ob.column);
            }
        }

        // Add LAG/LEAD source column
        if (args.len > 0) {
            if (args[0] == .column) {
                try col_names.append(self.allocator, args[0].column.name);
            }
        }

        try self.preloadColumns(col_names.items);
    }

    /// Parse aggregate function name to AggregateType
    fn parseAggregateType(name: []const u8, args: []const Expr) AggregateType {
        return aggregate_functions.parseAggregateTypeWithArgs(name, args);
    }

    /// Execute SELECT with GROUP BY and/or aggregates
    fn executeWithGroupBy(self: *Self, stmt: *const SelectStmt, filtered_indices: []const u32) !Result {
        // Preload all columns we'll need for grouping and aggregates
        try self.preloadGroupByColumns(stmt);

        // Get group by column names (empty if no GROUP BY but has aggregates)
        const group_cols = if (stmt.group_by) |gb| gb.columns else &[_][]const u8{};

        // Build groups: maps hash key to list of row indices
        // Using integer hashing for O(1) lookups (vs O(n) string comparison)
        var groups = std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)).init(self.allocator);
        defer {
            var iter = groups.valueIterator();
            while (iter.next()) |list| {
                list.deinit(self.allocator);
            }
            groups.deinit();
        }

        // Group rows by their hash key (efficient integer hashing)
        for (filtered_indices) |row_idx| {
            const key = self.hashGroupKey(group_cols, row_idx);

            const entry = try groups.getOrPut(key);
            if (!entry.found_existing) {
                entry.value_ptr.* = .{};
            }
            try entry.value_ptr.append(self.allocator, row_idx);
        }

        // If no GROUP BY and no matching rows, return single row with 0/null for aggregates
        const num_groups = if (groups.count() == 0 and group_cols.len == 0)
            @as(usize, 1) // Single aggregate result
        else
            groups.count();

        // Build result columns
        var result_columns = std.ArrayListUnmanaged(Result.Column){};
        errdefer {
            for (result_columns.items) |col| {
                col.data.free(self.allocator);
            }
            result_columns.deinit(self.allocator);
        }

        // Process each SELECT item
        for (stmt.columns) |item| {
            const col = try self.evaluateSelectItemForGroups(item, &groups, group_cols, num_groups);
            try result_columns.append(self.allocator, col);
        }

        var result = Result{
            .columns = try result_columns.toOwnedSlice(self.allocator),
            .row_count = num_groups,
            .allocator = self.allocator,
        };

        // Apply HAVING clause
        if (stmt.group_by) |gb| {
            if (gb.having) |having_expr| {
                try having_eval.applyHaving(self.allocator, &result, &having_expr, stmt.columns);
            }
        }

        // Apply ORDER BY
        if (stmt.order_by) |order_by| {
            try result_ops.applyOrderBy(self.allocator, result.columns, order_by);
        }

        // Apply LIMIT/OFFSET
        result.row_count = result_ops.applyLimitOffset(self.allocator, result.columns, stmt.limit, stmt.offset);

        return result;
    }

    /// Preload columns needed for GROUP BY and aggregates
    fn preloadGroupByColumns(self: *Self, stmt: *const SelectStmt) !void {
        var col_names = std.ArrayList([]const u8){};
        defer col_names.deinit(self.allocator);

        // Add GROUP BY columns
        if (stmt.group_by) |gb| {
            for (gb.columns) |col| {
                try col_names.append(self.allocator, col);
            }
        }

        // Add columns referenced in SELECT expressions
        for (stmt.columns) |item| {
            try self.extractExprColumnNames(&item.expr, &col_names);
        }

        try self.preloadColumns(col_names.items);
    }

    /// Extract column names from any expression
    fn extractExprColumnNames(self: *Self, expr: *const Expr, list: *std.ArrayList([]const u8)) anyerror!void {
        switch (expr.*) {
            .column => |col| {
                if (!std.mem.eql(u8, col.name, "*")) {
                    try list.append(self.allocator, col.name);
                }
            },
            .binary => |bin| {
                try self.extractExprColumnNames(bin.left, list);
                try self.extractExprColumnNames(bin.right, list);
            },
            .unary => |un| {
                try self.extractExprColumnNames(un.operand, list);
            },
            .call => |call| {
                for (call.args) |*arg| {
                    try self.extractExprColumnNames(arg, list);
                }
            },
            else => {},
        }
    }

    /// Build a group key string from GROUP BY column values for a row
    fn buildGroupKey(self: *Self, group_cols: []const []const u8, row_idx: u32) ![]const u8 {
        if (group_cols.len == 0) {
            // No GROUP BY - all rows in one group
            return try self.allocator.dupe(u8, "__all__");
        }

        var key = std.ArrayList(u8){};
        errdefer key.deinit(self.allocator);

        for (group_cols, 0..) |col_name, i| {
            if (i > 0) try key.append(self.allocator, '|');

            const cached = self.column_cache.get(col_name) orelse return error.ColumnNotCached;

            switch (cached) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| {
                    var buf: [64]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .int32, .date32 => |data| {
                    var buf: [32]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .float64 => |data| {
                    var buf: [64]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .float32 => |data| {
                    var buf: [32]u8 = undefined;
                    const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                    try key.appendSlice(self.allocator, str);
                },
                .bool_ => |data| {
                    const str = if (data[row_idx]) "true" else "false";
                    try key.appendSlice(self.allocator, str);
                },
                .string => |data| {
                    try key.appendSlice(self.allocator, data[row_idx]);
                },
            }
        }

        return key.toOwnedSlice(self.allocator);
    }

    /// Hash a group key from GROUP BY column values (efficient integer hashing)
    ///
    /// This is much faster than buildGroupKey() because:
    /// 1. No string allocation per row
    /// 2. O(1) hash comparison instead of O(n) string comparison
    /// 3. No type conversion overhead
    fn hashGroupKey(self: *Self, group_cols: []const []const u8, row_idx: u32) u64 {
        if (group_cols.len == 0) {
            // No GROUP BY - all rows in one group
            return 0;
        }

        var key_hash: u64 = hash.FNV_OFFSET_BASIS;

        for (group_cols) |col_name| {
            const cached = self.column_cache.get(col_name) orelse continue;

            const col_hash = switch (cached) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| hash.hashI64(data[row_idx]),
                .int32, .date32 => |data| hash.hashI32(data[row_idx]),
                .float64 => |data| hash.hashF64(data[row_idx]),
                .float32 => |data| hash.hashF32(data[row_idx]),
                .bool_ => |data| hash.hashBool(data[row_idx]),
                .string => |data| hash.stringHash(data[row_idx]),
            };

            key_hash = hash.combineHash(key_hash, col_hash);
        }

        return key_hash;
    }

    /// Evaluate a SELECT item for all groups
    fn evaluateSelectItemForGroups(
        self: *Self,
        item: ast.SelectItem,
        groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
        group_cols: []const []const u8,
        num_groups: usize,
    ) !Result.Column {
        const expr = &item.expr;

        // Handle aggregate function
        if (expr.* == .call and isAggregateFunction(expr.call.name)) {
            return self.evaluateAggregateForGroups(item, groups, num_groups);
        }

        // Handle regular column (must be in GROUP BY)
        if (expr.* == .column) {
            const col_name = expr.column.name;

            // Verify column is in GROUP BY
            var in_group_by = false;
            for (group_cols) |gb_col| {
                if (std.mem.eql(u8, gb_col, col_name)) {
                    in_group_by = true;
                    break;
                }
            }
            if (!in_group_by and group_cols.len > 0) {
                return error.ColumnNotInGroupBy;
            }

            return self.evaluateGroupByColumnForGroups(item, groups, num_groups);
        }

        return error.UnsupportedExpression;
    }

    /// Evaluate an aggregate function for all groups
    fn evaluateAggregateForGroups(
        self: *Self,
        item: ast.SelectItem,
        groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
        num_groups: usize,
    ) !Result.Column {
        const call = item.expr.call;
        const agg_type = parseAggregateType(call.name, call.args);

        // Determine column name for the aggregate (if not COUNT(*))
        const agg_col_name: ?[]const u8 = if (agg_type != .count_star and call.args.len > 0)
            if (call.args[0] == .column) call.args[0].column.name else null
        else
            null;

        // Check if this is a percentile-based aggregate (requires storing all values)
        const is_percentile_agg = agg_type == .median or agg_type == .percentile;

        // Check if this is a float-returning aggregate (stddev, variance)
        const is_float_agg = agg_type == .stddev or agg_type == .stddev_pop or
            agg_type == .variance or agg_type == .var_pop or agg_type == .avg;

        if (is_percentile_agg) {
            // Percentile-based aggregates need to store all values
            const results = try self.allocator.alloc(f64, num_groups);
            errdefer self.allocator.free(results);

            // Handle case of no groups
            if (groups.count() == 0) {
                results[0] = 0;
                return Result.Column{
                    .name = item.alias orelse call.name,
                    .data = Result.ColumnData{ .float64 = results },
                };
            }

            // Get percentile value (0.5 for median, from second arg for percentile)
            const percentile_val: f64 = if (agg_type == .median)
                0.5
            else if (call.args.len >= 2 and call.args[1] == .value)
                switch (call.args[1].value) {
                    .float => |f| f,
                    .integer => |i| @as(f64, @floatFromInt(i)),
                    else => 0.5,
                }
            else
                0.5;

            // Compute percentile for each group
            var group_idx: usize = 0;
            var iter = groups.iterator();
            while (iter.next()) |entry| {
                const row_indices = entry.value_ptr.items;

                var acc = PercentileAccumulator.init(self.allocator, percentile_val);
                defer acc.deinit();

                for (row_indices) |row_idx| {
                    if (agg_col_name) |col_name| {
                        const cached = self.column_cache.get(col_name) orelse return error.ColumnNotCached;

                        switch (cached) {
                            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| try acc.addInt(data[row_idx]),
                            .int32, .date32 => |data| try acc.addInt(data[row_idx]),
                            .float64 => |data| try acc.addFloat(data[row_idx]),
                            .float32 => |data| try acc.addFloat(data[row_idx]),
                            .bool_ => |data| try acc.addInt(if (data[row_idx]) 1 else 0),
                            .string => {}, // Skip strings for percentile
                        }
                    }
                }

                results[group_idx] = acc.getResult();
                group_idx += 1;
            }

            return Result.Column{
                .name = item.alias orelse call.name,
                .data = Result.ColumnData{ .float64 = results },
            };
        } else if (is_float_agg) {
            // Allocate float64 result array
            const results = try self.allocator.alloc(f64, num_groups);
            errdefer self.allocator.free(results);

            // Handle case of no groups
            if (groups.count() == 0) {
                results[0] = 0;
                return Result.Column{
                    .name = item.alias orelse call.name,
                    .data = Result.ColumnData{ .float64 = results },
                };
            }

            // Compute aggregate for each group
            var group_idx: usize = 0;
            var iter = groups.iterator();
            while (iter.next()) |entry| {
                const row_indices = entry.value_ptr.items;

                var acc = Accumulator.init(agg_type);

                for (row_indices) |row_idx| {
                    if (agg_col_name) |col_name| {
                        const cached = self.column_cache.get(col_name) orelse return error.ColumnNotCached;

                        switch (cached) {
                            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| acc.addInt(data[row_idx]),
                            .int32, .date32 => |data| acc.addInt(data[row_idx]),
                            .float64 => |data| acc.addFloat(data[row_idx]),
                            .float32 => |data| acc.addFloat(data[row_idx]),
                            .bool_ => acc.addCount(),
                            .string => acc.addCount(),
                        }
                    } else {
                        acc.addCount();
                    }
                }

                results[group_idx] = acc.getResult();
                group_idx += 1;
            }

            return Result.Column{
                .name = item.alias orelse call.name,
                .data = Result.ColumnData{ .float64 = results },
            };
        } else {
            // Allocate int64 result array for count, sum, min, max
            const results = try self.allocator.alloc(i64, num_groups);
            errdefer self.allocator.free(results);

            // Handle case of no groups (aggregate over empty set or no GROUP BY with data)
            if (groups.count() == 0) {
                // Return 0 for COUNT, null would be better for others but use 0
                results[0] = 0;
                return Result.Column{
                    .name = item.alias orelse call.name,
                    .data = Result.ColumnData{ .int64 = results },
                };
            }

            // Compute aggregate for each group
            var group_idx: usize = 0;
            var iter = groups.iterator();
            while (iter.next()) |entry| {
                const row_indices = entry.value_ptr.items;

                var acc = Accumulator.init(agg_type);

                for (row_indices) |row_idx| {
                    if (agg_type == .count_star) {
                        acc.addCount();
                    } else if (agg_col_name) |col_name| {
                        const cached = self.column_cache.get(col_name) orelse return error.ColumnNotCached;

                        switch (cached) {
                            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| acc.addInt(data[row_idx]),
                            .int32, .date32 => |data| acc.addInt(data[row_idx]),
                            .float64 => |data| acc.addFloat(data[row_idx]),
                            .float32 => |data| acc.addFloat(data[row_idx]),
                            .bool_ => acc.addCount(), // COUNT for bools
                            .string => acc.addCount(), // COUNT for strings
                        }
                    } else {
                        acc.addCount();
                    }
                }

                results[group_idx] = acc.getIntResult();
                group_idx += 1;
            }

            return Result.Column{
                .name = item.alias orelse call.name,
                .data = Result.ColumnData{ .int64 = results },
            };
        }
    }

    /// Evaluate a GROUP BY column for all groups (return first value from each group)
    fn evaluateGroupByColumnForGroups(
        self: *Self,
        item: ast.SelectItem,
        groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
        num_groups: usize,
    ) !Result.Column {
        const col_name = item.expr.column.name;
        const cached = self.column_cache.get(col_name) orelse return error.ColumnNotCached;

        // Allocate based on column type
        switch (cached) {
            .int64 => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .int64 = results },
                };
            },
            .timestamp_s => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .timestamp_s = results },
                };
            },
            .timestamp_ms => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .timestamp_ms = results },
                };
            },
            .timestamp_us => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .timestamp_us = results },
                };
            },
            .timestamp_ns => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .timestamp_ns = results },
                };
            },
            .date64 => |source_data| {
                const results = try self.allocator.alloc(i64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .date64 = results },
                };
            },
            .int32 => |source_data| {
                const results = try self.allocator.alloc(i32, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .int32 = results },
                };
            },
            .date32 => |source_data| {
                const results = try self.allocator.alloc(i32, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .date32 = results },
                };
            },
            .float64 => |source_data| {
                const results = try self.allocator.alloc(f64, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .float64 = results },
                };
            },
            .float32 => |source_data| {
                const results = try self.allocator.alloc(f32, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .float32 = results },
                };
            },
            .bool_ => |source_data| {
                const results = try self.allocator.alloc(bool, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = source_data[row_indices[0]];
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .bool_ = results },
                };
            },
            .string => |source_data| {
                const results = try self.allocator.alloc([]const u8, num_groups);
                errdefer self.allocator.free(results);

                var group_idx: usize = 0;
                var iter = groups.iterator();
                while (iter.next()) |entry| {
                    const row_indices = entry.value_ptr.items;
                    if (row_indices.len > 0) {
                        results[group_idx] = try self.allocator.dupe(u8, source_data[row_indices[0]]);
                    }
                    group_idx += 1;
                }

                return Result.Column{
                    .name = item.alias orelse col_name,
                    .data = Result.ColumnData{ .string = results },
                };
            },
        }
    }

    // ========================================================================
    // WHERE Clause Evaluation (delegated to where_eval module)
    // ========================================================================

    /// Static wrapper for evaluateToValue - used as callback from where_eval
    fn evalExprCallback(ctx: *anyopaque, expr: *const Expr, row_idx: u32) anyerror!Value {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.evaluateToValue(expr, row_idx);
    }

    /// Static wrapper for execute - used as callback from where_eval for subqueries
    fn executeCallback(ctx: *anyopaque, stmt: *ast.SelectStmt, params: []const Value) anyerror!Result {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.execute(stmt, params);
    }

    /// Evaluate WHERE clause and return matching row indices
    fn evaluateWhere(self: *Self, where_expr: *const Expr, params: []const Value) ![]u32 {
        const row_count = try self.getRowCount();

        const ctx = where_eval.WhereContext{
            .allocator = self.allocator,
            .column_cache = &self.column_cache,
            .row_count = row_count,
            .execute_fn = executeCallback,
            .eval_ctx = self,
            .eval_expr_fn = evalExprCallback,
        };

        return where_eval.evaluateWhere(ctx, where_expr, params);
    }

    /// Evaluate expression to a concrete value (delegates to expr_eval)
    fn evaluateToValue(self: *Self, expr: *const Expr, row_idx: u32) !Value {
        const ctx = self.buildExprContext();
        return expr_eval.evaluateExprToValue(ctx, expr, row_idx);
    }


    // ========================================================================
    // Method Call Evaluation (for @logic_table)
    // ========================================================================

    /// Evaluate a @logic_table method call (e.g., t.risk_score())
    ///
    /// This uses batch dispatch to compute all method results at once, then caches them.
    /// On subsequent calls with different row indices, the cached results are returned.
    ///
    /// For methods that require column data (Phase 3/4), the inputs need to be populated
    /// with ColumnBinding data from the Lance table.
    ///
    fn evaluateMethodCall(self: *Self, mc: anytype, row_idx: u32) !Value {
        // Get class name from alias
        const class_name = self.logic_table_aliases.get(mc.object) orelse
            return error.TableAliasNotFound;

        // Get dispatcher
        var dispatcher = self.dispatcher orelse
            return error.NoDispatcherConfigured;

        // For now, we only support methods with no runtime arguments
        // The compiled method operates on batch data loaded in the LogicTableContext
        if (mc.args.len > 0) {
            return error.MethodArgsNotSupported;
        }

        // Build cache key: "ClassName.methodName"
        var cache_key_buf: [256]u8 = undefined;
        const cache_key = std.fmt.bufPrint(&cache_key_buf, "{s}.{s}", .{ class_name, mc.method }) catch
            return error.CacheKeyTooLong;

        // Check if we have cached results
        if (self.method_results_cache.get(cache_key)) |cached_results| {
            if (row_idx < cached_results.len) {
                return Value{ .float = cached_results[row_idx] };
            }
            return error.RowIndexOutOfBounds;
        }

        // No cached results - compute batch results

        // Determine row count from the current table
        const row_count: usize = blk: {
            if (self.table) |t| {
                // Get row count from column 0
                const count = t.rowCount(0) catch 1000;
                break :blk @intCast(count);
            }
            // No table - use a default batch size
            break :blk 1000;
        };

        // Check if this is a batch method
        if (dispatcher.isBatchMethod(class_name, mc.method)) {
            // Allocate output buffer
            var output = logic_table_dispatch.ColumnBuffer.initFloat64(self.allocator, row_count) catch
                return error.OutOfMemory;
            errdefer output.deinit(self.allocator);

            // TODO (Phase 3): Get method dependencies and load column data
            // For now, pass empty inputs - methods that don't need column data will work
            const inputs = &[_]logic_table_dispatch.ColumnBinding{};

            // Call batch dispatch
            dispatcher.callMethodBatch(class_name, mc.method, inputs, null, &output, null) catch |err| {
                return switch (err) {
                    logic_table_dispatch.DispatchError.MethodNotFound => error.MethodNotFound,
                    logic_table_dispatch.DispatchError.ArgumentCountMismatch => error.ArgumentCountMismatch,
                    else => error.ExecutionFailed,
                };
            };

            // Cache the results
            const results = output.f64 orelse return error.NoResults;
            const cache_key_copy = self.allocator.dupe(u8, cache_key) catch
                return error.OutOfMemory;
            self.method_results_cache.put(cache_key_copy, results) catch {
                self.allocator.free(cache_key_copy);
                return error.CachePutFailed;
            };

            // Return the result for this row
            if (row_idx < results.len) {
                return Value{ .float = results[row_idx] };
            }
            return error.RowIndexOutOfBounds;
        }

        // Fallback to scalar dispatch for non-batch methods
        const result = dispatcher.callMethod0(class_name, mc.method) catch |err| {
            return switch (err) {
                logic_table_dispatch.DispatchError.MethodNotFound => error.MethodNotFound,
                logic_table_dispatch.DispatchError.ArgumentCountMismatch => error.ArgumentCountMismatch,
                else => error.ExecutionFailed,
            };
        };

        return Value{ .float = result };
    }

    /// Evaluate an expression column for all filtered indices (delegates to expr_eval)
    fn evaluateExpressionColumn(
        self: *Self,
        item: ast.SelectItem,
        indices: []const u32,
    ) !Result.Column {
        // First, preload any columns referenced in the expression
        var col_names = std.ArrayList([]const u8){};
        defer col_names.deinit(self.allocator);
        try self.extractExprColumnNames(&item.expr, &col_names);
        try self.preloadColumns(col_names.items);

        // Delegate to expr_eval module
        const ctx = self.buildExprContext();
        return expr_eval.evaluateExpressionColumn(ctx, item, indices);
    }

    /// Get all row indices (0, 1, 2, ..., n-1)
    fn getAllIndices(self: *Self) ![]u32 {
        // Get row count (works with both Lance and Parquet)
        const row_count = try self.getRowCount();
        const indices = try self.allocator.alloc(u32, @intCast(row_count));

        for (indices, 0..) |*idx, i| {
            idx.* = @intCast(i);
        }

        return indices;
    }

    // ========================================================================
    // Column Reading
    // ========================================================================

    /// Read columns based on SELECT list and filtered indices
    fn readColumns(
        self: *Self,
        select_list: []const ast.SelectItem,
        indices: []const u32,
    ) ![]Result.Column {
        var columns = std.ArrayList(Result.Column){};
        errdefer {
            for (columns.items) |col| {
                col.data.free(self.allocator);
            }
            columns.deinit(self.allocator);
        }

        for (select_list) |item| {
            // Skip window function expressions - they're handled separately
            if (isWindowFunction(&item.expr)) continue;

            // Handle SELECT *
            if (item.expr == .column and std.mem.eql(u8, item.expr.column.name, "*")) {
                const col_names = try self.getColumnNames();
                // Only Lance allocates column names, other table types return stored slices
                defer if (!self.hasTypedTable()) self.allocator.free(col_names);

                for (col_names) |col_name| {
                    // Look up the physical column ID from the name
                    // The physical column ID maps to the column metadata index
                    const physical_col_id = self.getPhysicalColumnId(col_name) orelse return error.ColumnNotFound;
                    const data = try self.readColumnAtIndices(physical_col_id, indices);

                    try columns.append(self.allocator, Result.Column{
                        .name = col_name,
                        .data = data,
                    });
                }
                break; // SELECT * means we're done
            }

            // Handle regular column
            if (item.expr == .column) {
                const col_name = item.expr.column.name;
                const col_idx = self.getPhysicalColumnId(col_name) orelse return error.ColumnNotFound;
                const data = try self.readColumnAtIndices(col_idx, indices);

                try columns.append(self.allocator, Result.Column{
                    .name = item.alias orelse col_name,
                    .data = data,
                });
            } else {
                // Handle expressions (arithmetic, functions, etc.)
                const expr_col = try self.evaluateExpressionColumn(item, indices);
                try columns.append(self.allocator, expr_col);
            }
        }

        return columns.toOwnedSlice(self.allocator);
    }

    /// Read column data at specific row indices
    fn readColumnAtIndices(
        self: *Self,
        col_idx: u32,
        indices: []const u32,
    ) !Result.ColumnData {
        // Use generic reader for typed tables
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| return self.readColumnFromTableAtIndices(t, col_idx, indices);
        }

        const fld = self.tbl().getFieldById(col_idx) orelse return error.InvalidColumn;
        const col_type = LanceColumnType.fromLogicalType(fld.logical_type);

        // String requires special handling for ownership
        if (col_type == .string) {
            const all_data = try self.tbl().readStringColumn(col_idx);
            defer {
                for (all_data) |str| self.allocator.free(str);
                self.allocator.free(all_data);
            }
            const filtered = try self.allocator.alloc([]const u8, indices.len);
            for (indices, 0..) |idx, i| filtered[i] = try self.allocator.dupe(u8, all_data[idx]);
            return Result.ColumnData{ .string = filtered };
        }

        return switch (col_type) {
            .timestamp_ns => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .timestamp_ns = try self.filterByIndices(i64, all_data, indices) };
            },
            .timestamp_us => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .timestamp_us = try self.filterByIndices(i64, all_data, indices) };
            },
            .timestamp_ms => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .timestamp_ms = try self.filterByIndices(i64, all_data, indices) };
            },
            .timestamp_s => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .timestamp_s = try self.filterByIndices(i64, all_data, indices) };
            },
            .date32 => blk: {
                const all_data = try self.tbl().readInt32Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .date32 = try self.filterByIndices(i32, all_data, indices) };
            },
            .date64 => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .date64 = try self.filterByIndices(i64, all_data, indices) };
            },
            .int32 => blk: {
                const all_data = try self.tbl().readInt32Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .int32 = try self.filterByIndices(i32, all_data, indices) };
            },
            .float32 => blk: {
                const all_data = try self.tbl().readFloat32Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .float32 = try self.filterByIndices(f32, all_data, indices) };
            },
            .bool_ => blk: {
                const all_data = try self.tbl().readBoolColumn(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .bool_ = try self.filterByIndices(bool, all_data, indices) };
            },
            .int64 => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .int64 = try self.filterByIndices(i64, all_data, indices) };
            },
            .float64 => blk: {
                const all_data = try self.tbl().readFloat64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .float64 = try self.filterByIndices(f64, all_data, indices) };
            },
            .string => unreachable, // Handled above
            .unsupported => error.UnsupportedColumnType,
        };
    }

    /// Generic column reader for any table type with standard interface
    fn readColumnFromTableAtIndices(
        self: *Self,
        table: anytype,
        col_idx: u32,
        indices: []const u32,
    ) !Result.ColumnData {
        const T = @TypeOf(table.*);
        const is_xlsx = T == XlsxTable;
        const col_type = table.getColumnType(col_idx) orelse return error.InvalidColumn;

        switch (col_type) {
            .int64 => {
                const all_data = table.readInt64Column(col_idx) catch return error.ColumnReadError;
                defer self.allocator.free(all_data);
                return Result.ColumnData{ .int64 = try self.filterByIndices(i64, all_data, indices) };
            },
            .int32 => {
                const all_data = table.readInt32Column(col_idx) catch return error.ColumnReadError;
                defer self.allocator.free(all_data);
                return Result.ColumnData{ .int32 = try self.filterByIndices(i32, all_data, indices) };
            },
            .double => {
                const all_data = table.readFloat64Column(col_idx) catch return error.ColumnReadError;
                defer self.allocator.free(all_data);
                return Result.ColumnData{ .float64 = try self.filterByIndices(f64, all_data, indices) };
            },
            .float => {
                const all_data = table.readFloat32Column(col_idx) catch return error.ColumnReadError;
                defer self.allocator.free(all_data);
                return Result.ColumnData{ .float32 = try self.filterByIndices(f32, all_data, indices) };
            },
            .boolean => {
                const all_data = table.readBoolColumn(col_idx) catch return error.ColumnReadError;
                defer self.allocator.free(all_data);
                return Result.ColumnData{ .bool_ = try self.filterByIndices(bool, all_data, indices) };
            },
            .byte_array, .fixed_len_byte_array => {
                const all_data = table.readStringColumn(col_idx) catch return error.ColumnReadError;
                defer {
                    for (all_data) |str| self.allocator.free(str);
                    self.allocator.free(all_data);
                }
                const filtered = try self.allocator.alloc([]const u8, indices.len);
                for (indices, 0..) |idx, i| filtered[i] = try self.allocator.dupe(u8, all_data[idx]);
                return Result.ColumnData{ .string = filtered };
            },
            else => {
                if (is_xlsx) {
                    const all_data = table.readFloat64Column(col_idx) catch return error.ColumnReadError;
                    defer self.allocator.free(all_data);
                    return Result.ColumnData{ .float64 = try self.filterByIndices(f64, all_data, indices) };
                }
                return error.UnsupportedColumnType;
            },
        }
    }

    fn freeColumnData(self: *Self, data: *Result.ColumnData) void {
        data.free(self.allocator);
    }

};

// Tests are in tests/test_sql_executor.zig
