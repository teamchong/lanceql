// Logic Table Executor - Core execution engine for @logic_table
//
// This module provides the shared execution layer used by both SQL and DataFrame APIs.
//
// Workflow:
// 1. Parse Python file to extract Table() declarations and method names
// 2. Resolve Lance file paths relative to Python file location
// 3. Load required columns from Lance files
// 4. Bind columns to LogicTableContext
// 5. Dispatch method calls to compiled extern functions
//
// Usage:
//   var executor = try LogicTableExecutor.init(allocator, "fraud_detector.py");
//   defer executor.deinit();
//
//   // Bind columns (or auto-load from Table declarations)
//   try executor.loadTables();
//
//   // Execute method on all rows (batch)
//   const scores = try executor.callMethodBatch("risk_score");
//
//   // Execute method on filtered rows
//   const filtered_scores = try executor.callMethodFiltered("risk_score", filtered_indices);

const std = @import("std");
const logic_table = @import("logic_table.zig");
const Table = @import("lanceql.table").Table;

const LogicTableContext = logic_table.LogicTableContext;
const QueryContext = logic_table.QueryContext;
const LogicTableMeta = logic_table.LogicTableMeta;
const MethodMeta = logic_table.MethodMeta;
const LogicTableError = logic_table.LogicTableError;

/// Table declaration extracted from Python @logic_table class
pub const TableDecl = struct {
    /// Variable name in Python (e.g., "orders")
    name: []const u8,
    /// Lance file path (e.g., "orders.lance")
    path: []const u8,
};

/// Loaded table with column data
pub const LoadedTable = struct {
    decl: TableDecl,
    table: *Table,
    /// Columns loaded from this table
    columns: std.StringHashMap(ColumnData),

    pub const ColumnData = union(enum) {
        f32: []const f32,
        f64: []const f64,
        i64: []const i64,
        i32: []const i32,
        bool_: []const bool,
        string: []const []const u8,
    };
};

/// Method function pointer types
pub const MethodFnF64 = *const fn (a: [*]const f64, b: [*]const f64, len: usize) callconv(.c) f64;
pub const MethodFnF64Single = *const fn (a: [*]const f64, len: usize) callconv(.c) f64;

/// Registered method with function pointer
pub const RegisteredMethod = struct {
    name: []const u8,
    class_name: []const u8,
    /// Function pointer (type depends on signature)
    fn_ptr: *const anyopaque,
    /// Number of array parameters
    num_array_params: u8,
};

/// LogicTable Executor - manages @logic_table execution
pub const LogicTableExecutor = struct {
    allocator: std.mem.Allocator,

    /// Path to Python file
    python_file: []const u8,

    /// Directory containing Python file (for relative path resolution)
    base_dir: []const u8,

    /// Class name extracted from Python
    class_name: []const u8,

    /// Table declarations from Python
    table_decls: std.ArrayList(TableDecl),

    /// Loaded tables with data
    loaded_tables: std.StringHashMap(LoadedTable),

    /// Method metadata
    methods: std.ArrayList(MethodMeta),

    /// Registered method function pointers
    registered_methods: std.StringHashMap(RegisteredMethod),

    /// LogicTableContext for column binding
    ctx: LogicTableContext,

    /// QueryContext for pushdown
    query_ctx: ?*QueryContext,

    const Self = @This();

    /// Initialize executor from Python file path
    pub fn init(allocator: std.mem.Allocator, python_file: []const u8) !Self {
        // Extract base directory
        const base_dir = std.fs.path.dirname(python_file) orelse ".";

        return Self{
            .allocator = allocator,
            .python_file = try allocator.dupe(u8, python_file),
            .base_dir = try allocator.dupe(u8, base_dir),
            .class_name = "",
            .table_decls = std.ArrayList(TableDecl).init(allocator),
            .loaded_tables = std.StringHashMap(LoadedTable).init(allocator),
            .methods = std.ArrayList(MethodMeta).init(allocator),
            .registered_methods = std.StringHashMap(RegisteredMethod).init(allocator),
            .ctx = LogicTableContext.init(allocator),
            .query_ctx = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.python_file);
        self.allocator.free(self.base_dir);

        // Free table declarations
        for (self.table_decls.items) |decl| {
            self.allocator.free(decl.name);
            self.allocator.free(decl.path);
        }
        self.table_decls.deinit();

        // Free loaded tables
        var iter = self.loaded_tables.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.table.deinit();
            entry.value_ptr.columns.deinit();
        }
        self.loaded_tables.deinit();

        // Free methods
        for (self.methods.items) |method| {
            self.allocator.free(method.name);
        }
        self.methods.deinit();

        self.registered_methods.deinit();
        self.ctx.deinit();
    }

    /// Set query context for pushdown optimization
    pub fn setQueryContext(self: *Self, query_ctx: *QueryContext) void {
        self.query_ctx = query_ctx;
        self.ctx.query_context = query_ctx;
    }

    /// Add a table declaration manually (alternative to parsing Python)
    pub fn addTableDecl(self: *Self, name: []const u8, path: []const u8) !void {
        try self.table_decls.append(.{
            .name = try self.allocator.dupe(u8, name),
            .path = try self.allocator.dupe(u8, path),
        });
    }

    /// Add a method declaration manually
    pub fn addMethod(self: *Self, name: []const u8, deps: []const []const u8) !void {
        try self.methods.append(.{
            .name = try self.allocator.dupe(u8, name),
            .deps = deps,
        });
    }

    /// Register a compiled method function pointer
    pub fn registerMethod(
        self: *Self,
        class_name: []const u8,
        method_name: []const u8,
        fn_ptr: *const anyopaque,
        num_array_params: u8,
    ) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ class_name, method_name });
        errdefer self.allocator.free(key);

        try self.registered_methods.put(key, .{
            .name = method_name,
            .class_name = class_name,
            .fn_ptr = fn_ptr,
            .num_array_params = num_array_params,
        });
    }

    /// Load all declared tables from Lance files
    pub fn loadTables(self: *Self) !void {
        for (self.table_decls.items) |decl| {
            try self.loadTable(decl);
        }
    }

    /// Load a single table
    fn loadTable(self: *Self, decl: TableDecl) !void {
        // Resolve path relative to Python file
        const full_path = try std.fs.path.join(self.allocator, &.{ self.base_dir, decl.path });
        defer self.allocator.free(full_path);

        // Read file
        const file = std.fs.cwd().openFile(full_path, .{}) catch |err| {
            std.debug.print("Failed to open Lance file: {s} (error: {})\n", .{ full_path, err });
            return LogicTableError.TableNotFound;
        };
        defer file.close();

        const file_size = try file.getEndPos();
        const data = try self.allocator.alloc(u8, file_size);
        _ = try file.readAll(data);

        // Initialize Table
        const table = try self.allocator.create(Table);
        table.* = Table.init(self.allocator, data) catch |err| {
            std.debug.print("Failed to parse Lance file: {s} (error: {})\n", .{ full_path, err });
            return LogicTableError.InvalidLogicTable;
        };

        try self.loaded_tables.put(decl.name, .{
            .decl = decl,
            .table = table,
            .columns = std.StringHashMap(LoadedTable.ColumnData).init(self.allocator),
        });
    }

    /// Load a specific column from a table and bind to context
    pub fn loadColumn(self: *Self, table_name: []const u8, column_name: []const u8) !void {
        const loaded = self.loaded_tables.getPtr(table_name) orelse
            return LogicTableError.TableNotFound;

        const table = loaded.table;

        // Find column index by name
        const col_idx_usize = table.columnIndex(column_name) orelse
            return LogicTableError.ColumnNotFound;
        const col_idx: u32 = @intCast(col_idx_usize);

        // Get field to determine type
        const field = table.getField(col_idx_usize) orelse
            return LogicTableError.ColumnNotFound;

        // Determine column type from logical_type string
        if (std.mem.eql(u8, field.logical_type, "float") or
            std.mem.eql(u8, field.logical_type, "float32"))
        {
            const data = table.readFloat32Column(col_idx) catch
                return LogicTableError.ColumnNotFound;
            try loaded.columns.put(column_name, .{ .f32 = data });
            try self.ctx.bindF32(table_name, column_name, data);
        } else if (std.mem.eql(u8, field.logical_type, "double") or
            std.mem.eql(u8, field.logical_type, "float64"))
        {
            const data = table.readFloat64Column(col_idx) catch
                return LogicTableError.ColumnNotFound;
            try loaded.columns.put(column_name, .{ .f64 = data });
            // Note: LogicTableContext currently only has f32 and i64
        } else if (std.mem.eql(u8, field.logical_type, "int64") or
            std.mem.eql(u8, field.logical_type, "long"))
        {
            const data = table.readInt64Column(col_idx) catch
                return LogicTableError.ColumnNotFound;
            try loaded.columns.put(column_name, .{ .i64 = data });
            try self.ctx.bindI64(table_name, column_name, data);
        } else if (std.mem.eql(u8, field.logical_type, "int32") or
            std.mem.eql(u8, field.logical_type, "int"))
        {
            const data = table.readInt32Column(col_idx) catch
                return LogicTableError.ColumnNotFound;
            try loaded.columns.put(column_name, .{ .i32 = data });
        } else {
            return LogicTableError.TypeMismatch;
        }
    }

    /// Call a registered method in batch mode (all rows)
    pub fn callMethodBatch(
        self: *Self,
        class_name: []const u8,
        method_name: []const u8,
        args: anytype,
    ) !f64 {
        const key = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ class_name, method_name });
        defer self.allocator.free(key);

        const method = self.registered_methods.get(key) orelse
            return LogicTableError.MethodNotFound;

        // Call based on number of array parameters
        switch (method.num_array_params) {
            2 => {
                // Two array params (e.g., dot_product(a, b))
                const fn_ptr: MethodFnF64 = @ptrCast(@alignCast(method.fn_ptr));
                const a = args[0];
                const b = args[1];
                const len = args[2];
                return fn_ptr(a, b, len);
            },
            1 => {
                // Single array param (e.g., sum_squares(a))
                const fn_ptr: MethodFnF64Single = @ptrCast(@alignCast(method.fn_ptr));
                const a = args[0];
                const len = args[1];
                return fn_ptr(a, len);
            },
            else => return LogicTableError.MethodNotFound,
        }
    }

    /// Get column data from context
    pub fn getColumnF32(self: *const Self, table_name: []const u8, column_name: []const u8) ![]const f32 {
        return self.ctx.getF32(table_name, column_name);
    }

    pub fn getColumnI64(self: *const Self, table_name: []const u8, column_name: []const u8) ![]const i64 {
        return self.ctx.getI64(table_name, column_name);
    }

    /// Get row count from first loaded table
    pub fn getRowCount(self: *const Self) usize {
        var iter = self.loaded_tables.iterator();
        if (iter.next()) |entry| {
            // Use first column to get row count
            return entry.value_ptr.table.rowCount(0) catch 0;
        }
        return 0;
    }

    /// Get LogicTableContext for direct access
    pub fn getContext(self: *Self) *LogicTableContext {
        return &self.ctx;
    }

    /// Get metadata about this logic_table
    pub fn getMeta(self: *const Self) LogicTableMeta {
        return .{
            .name = self.class_name,
            .methods = self.methods.items,
        };
    }
};

// =============================================================================
// Extern declarations for compiled @logic_table methods
// These come from lib/vector_ops.a (linked at build time)
// =============================================================================

// VectorOps class methods
pub extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) callconv(.c) f64;
pub extern fn VectorOps_sum_squares(a: [*]const f64, len: usize) callconv(.c) f64;

// =============================================================================
// Convenience functions for common patterns
// =============================================================================

/// Create executor with VectorOps methods pre-registered
pub fn createVectorOpsExecutor(allocator: std.mem.Allocator) !LogicTableExecutor {
    var exec = try LogicTableExecutor.init(allocator, "vector_ops.py");

    // Register compiled extern functions
    try exec.registerMethod("VectorOps", "dot_product", @ptrCast(&VectorOps_dot_product), 2);
    try exec.registerMethod("VectorOps", "sum_squares", @ptrCast(&VectorOps_sum_squares), 1);

    try exec.addMethod("dot_product", &.{ "a", "b" });
    try exec.addMethod("sum_squares", &.{"a"});

    return exec;
}

// =============================================================================
// Tests
// =============================================================================

test "LogicTableExecutor basic" {
    const allocator = std.testing.allocator;

    var executor = try LogicTableExecutor.init(allocator, "test.py");
    defer executor.deinit();

    // Add table declaration
    try executor.addTableDecl("orders", "orders.lance");

    try std.testing.expectEqual(@as(usize, 1), executor.table_decls.items.len);
    try std.testing.expectEqualStrings("orders", executor.table_decls.items[0].name);
}

test "LogicTableExecutor path resolution" {
    const allocator = std.testing.allocator;

    // Test with subdirectory
    var executor = try LogicTableExecutor.init(allocator, "examples/fraud_detector.py");
    defer executor.deinit();

    try std.testing.expectEqualStrings("examples", executor.base_dir);
}
