// Logic Table Integration for LanceQL
// This module provides the interface for @logic_table functions compiled by metal0
//
// Workflow:
// 1. Python code with @logic_table decorator is compiled by metal0
// 2. metal0 --emit-logic-table compiles to .a static library
// 3. LanceQL links the .a file (placed in lib/logic_table.a)
// 4. Compiled @logic_table structs are accessed via extern declarations
//
// The .a file contains:
// - Compiled @logic_table Python classes as Zig structs
// - metal0 runtime (bundled)
// - c_interop modules for numpy/external libs (bundled)

const std = @import("std");

/// LogicTable struct metadata - describes a compiled @logic_table class
pub const LogicTableMeta = struct {
    name: []const u8,
    methods: []const MethodMeta,
};

/// Method metadata - describes a method on a @logic_table class
pub const MethodMeta = struct {
    name: []const u8,
    /// Column dependencies in "table.column" format
    deps: []const []const u8,
};

/// Errors from LogicTable execution
pub const LogicTableError = error{
    ColumnNotBound,
    ColumnNotFound,
    MethodNotFound,
    TypeMismatch,
    DimensionMismatch,
    OutOfMemory,
    GPUError,
    InvalidLogicTable,
    TableNotFound,
};

/// Runtime context for executing LogicTable methods
/// Stores pre-loaded column data for batch operations
pub const LogicTableContext = struct {
    allocator: std.mem.Allocator,

    /// Cached column data: "table.column" → slice
    column_cache_f32: std.StringHashMap([]const f32),
    column_cache_i64: std.StringHashMap([]const i64),

    /// Owned keys that need to be freed
    owned_keys: std.ArrayListUnmanaged([]const u8),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .column_cache_f32 = std.StringHashMap([]const f32).init(allocator),
            .column_cache_i64 = std.StringHashMap([]const i64).init(allocator),
            .owned_keys = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        // Free owned keys
        for (self.owned_keys.items) |key| {
            self.allocator.free(key);
        }
        self.owned_keys.deinit(self.allocator);
        self.column_cache_f32.deinit();
        self.column_cache_i64.deinit();
    }

    /// Bind f32 column data with a key like "table.column"
    pub fn bindColumnF32(self: *Self, key: []const u8, data: []const f32) LogicTableError!void {
        const key_copy = self.allocator.dupe(u8, key) catch return LogicTableError.OutOfMemory;
        errdefer self.allocator.free(key_copy);

        self.column_cache_f32.put(key_copy, data) catch return LogicTableError.OutOfMemory;
        self.owned_keys.append(self.allocator, key_copy) catch return LogicTableError.OutOfMemory;
    }

    /// Bind i64 column data with a key like "table.column"
    pub fn bindColumnI64(self: *Self, key: []const u8, data: []const i64) LogicTableError!void {
        const key_copy = self.allocator.dupe(u8, key) catch return LogicTableError.OutOfMemory;
        errdefer self.allocator.free(key_copy);

        self.column_cache_i64.put(key_copy, data) catch return LogicTableError.OutOfMemory;
        self.owned_keys.append(self.allocator, key_copy) catch return LogicTableError.OutOfMemory;
    }

    /// Bind f32 column data with table alias and column name
    pub fn bindF32(self: *Self, table_alias: []const u8, column_name: []const u8, data: []const f32) LogicTableError!void {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ table_alias, column_name }) catch
            return LogicTableError.OutOfMemory;
        return self.bindColumnF32(key, data);
    }

    /// Bind i64 column data with table alias and column name
    pub fn bindI64(self: *Self, table_alias: []const u8, column_name: []const u8, data: []const i64) LogicTableError!void {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ table_alias, column_name }) catch
            return LogicTableError.OutOfMemory;
        return self.bindColumnI64(key, data);
    }

    /// Get bound f32 column data
    pub fn getColumnF32(self: *const Self, key: []const u8) ?[]const f32 {
        return self.column_cache_f32.get(key);
    }

    /// Get bound i64 column data
    pub fn getColumnI64(self: *const Self, key: []const u8) ?[]const i64 {
        return self.column_cache_i64.get(key);
    }

    /// Get f32 column data by table alias and column name
    pub fn getF32(self: *const Self, table_alias: []const u8, column_name: []const u8) LogicTableError![]const f32 {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ table_alias, column_name }) catch
            return LogicTableError.OutOfMemory;

        return self.column_cache_f32.get(key) orelse LogicTableError.ColumnNotBound;
    }

    /// Get i64 column data by table alias and column name
    pub fn getI64(self: *const Self, table_alias: []const u8, column_name: []const u8) LogicTableError![]const i64 {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ table_alias, column_name }) catch
            return LogicTableError.OutOfMemory;

        return self.column_cache_i64.get(key) orelse LogicTableError.ColumnNotBound;
    }
};

/// Registry of all available logic_table structs
/// The actual structs are linked from the .a static library
pub const LogicTableRegistry = struct {
    allocator: std.mem.Allocator,

    /// Registered tables (populated at runtime or via extern symbols)
    tables: std.StringHashMap(LogicTableMeta),

    pub fn init(allocator: std.mem.Allocator) LogicTableRegistry {
        return .{
            .allocator = allocator,
            .tables = std.StringHashMap(LogicTableMeta).init(allocator),
        };
    }

    pub fn deinit(self: *LogicTableRegistry) void {
        self.tables.deinit();
    }

    /// Register a logic_table struct
    pub fn register(self: *LogicTableRegistry, meta: LogicTableMeta) LogicTableError!void {
        self.tables.put(meta.name, meta) catch return LogicTableError.OutOfMemory;
    }

    /// Get metadata about a logic_table struct by name
    pub fn getTable(self: *const LogicTableRegistry, name: []const u8) ?LogicTableMeta {
        return self.tables.get(name);
    }

    /// List all registered table names
    pub fn listTables(self: *const LogicTableRegistry, allocator: std.mem.Allocator) LogicTableError![]const []const u8 {
        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer names.deinit(allocator);

        var iter = self.tables.keyIterator();
        while (iter.next()) |key| {
            names.append(allocator, key.*) catch return LogicTableError.OutOfMemory;
        }

        return names.toOwnedSlice(allocator) catch LogicTableError.OutOfMemory;
    }
};

// =============================================================================
// Extern declarations for functions from the linked .a static library
// These are populated when lib/logic_table.a is linked
// =============================================================================

// Example: if your Python @logic_table class has a method like:
//   @logic_table
//   class VectorOps:
//       def cosine_similarity(self, query_embedding, doc_embedding) -> float:
//           ...
//
// metal0 will emit an extern function like:
//   export fn VectorOps_cosine_similarity(query: [*]const f32, query_len: usize,
//                                          doc: [*]const f32, doc_len: usize) f32;
//
// You can declare and use it here:
//   extern fn VectorOps_cosine_similarity(query: [*]const f32, query_len: usize,
//                                          doc: [*]const f32, doc_len: usize) f32;

// =============================================================================
// Tests
// =============================================================================

test "LogicTableContext basic" {
    const allocator = std.testing.allocator;
    var ctx = LogicTableContext.init(allocator);
    defer ctx.deinit();

    const f32_data = [_]f32{ 1.0, 2.0, 3.0 };
    const i64_data = [_]i64{ 10, 20, 30 };

    try ctx.bindF32("docs", "embedding", &f32_data);
    try ctx.bindI64("docs", "id", &i64_data);

    const retrieved_f32 = try ctx.getF32("docs", "embedding");
    try std.testing.expectEqual(@as(usize, 3), retrieved_f32.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), retrieved_f32[0], 0.001);

    const retrieved_i64 = try ctx.getI64("docs", "id");
    try std.testing.expectEqual(@as(usize, 3), retrieved_i64.len);
    try std.testing.expectEqual(@as(i64, 10), retrieved_i64[0]);
}

test "LogicTableRegistry basic" {
    const allocator = std.testing.allocator;
    var registry = LogicTableRegistry.init(allocator);
    defer registry.deinit();

    const methods = [_]MethodMeta{
        .{ .name = "cosine_similarity", .deps = &[_][]const u8{ "docs.embedding", "query.embedding" } },
    };

    try registry.register(.{
        .name = "VectorOps",
        .methods = &methods,
    });

    const meta = registry.getTable("VectorOps");
    try std.testing.expect(meta != null);
    try std.testing.expectEqualStrings("VectorOps", meta.?.name);
    try std.testing.expectEqual(@as(usize, 1), meta.?.methods.len);
}
