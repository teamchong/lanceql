/// LanceQL ↔ metal0 Integration
///
/// This module provides JIT compilation of @logic_table Python code
/// using metal0's schema-aware compilation API.
///
/// Architecture:
///   1. Query executor extracts schema from Lance file
///   2. This module passes schema to metal0 compiler
///   3. metal0 generates native Zig code with concrete types
///   4. Code is JIT compiled and executed
///
/// This enables FUSED compilation:
///   - Query predicates + @logic_table methods + Lance schema
///   - All compiled together for maximum optimization
///   - No PyValue wrappers, pure native SIMD code

const std = @import("std");
const metal0 = @import("metal0");

/// Column type from Lance schema
pub const ColumnType = enum {
    i64,
    i32,
    i16,
    i8,
    u64,
    u32,
    u16,
    u8,
    f64,
    f32,
    bool,
    string,
    bytes,
    vec_f32, // Fixed-size list of f32 (embeddings)
    vec_f64,

    /// Convert from Lance physical type
    pub fn fromLanceType(lance_type: []const u8) ?ColumnType {
        const map = std.StaticStringMap(ColumnType).initComptime(.{
            .{ "int64", .i64 },
            .{ "int32", .i32 },
            .{ "int16", .i16 },
            .{ "int8", .i8 },
            .{ "uint64", .u64 },
            .{ "uint32", .u32 },
            .{ "uint16", .u16 },
            .{ "uint8", .u8 },
            .{ "double", .f64 },
            .{ "float", .f32 },
            .{ "boolean", .bool },
            .{ "utf8", .string },
            .{ "binary", .bytes },
        });
        return map.get(lance_type);
    }

    /// Get Zig type string
    pub fn toZigType(self: ColumnType) []const u8 {
        return switch (self) {
            .i64 => "i64",
            .i32 => "i32",
            .i16 => "i16",
            .i8 => "i8",
            .u64 => "u64",
            .u32 => "u32",
            .u16 => "u16",
            .u8 => "u8",
            .f64 => "f64",
            .f32 => "f32",
            .bool => "bool",
            .string => "[]const u8",
            .bytes => "[]const u8",
            .vec_f32 => "[]const f32",
            .vec_f64 => "[]const f64",
        };
    }
};

/// Column definition from Lance schema
pub const ColumnDef = struct {
    name: []const u8,
    column_type: ColumnType,
    nullable: bool = false,
};

/// Schema extracted from Lance file
pub const LanceSchema = struct {
    columns: []const ColumnDef,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LanceSchema) void {
        self.allocator.free(self.columns);
    }
};

/// Compiled @logic_table function
pub const CompiledFunction = struct {
    /// Function pointer (signature depends on method)
    ptr: *const anyopaque,

    /// Zig source code (for debugging)
    source: []const u8,

    /// Method name
    name: []const u8,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *CompiledFunction) void {
        self.allocator.free(self.source);
    }
};

/// JIT compilation context
pub const JitContext = struct {
    allocator: std.mem.Allocator,
    schema: ?LanceSchema = null,
    compiled_functions: std.StringHashMap(CompiledFunction),

    pub fn init(allocator: std.mem.Allocator) JitContext {
        return .{
            .allocator = allocator,
            .compiled_functions = std.StringHashMap(CompiledFunction).init(allocator),
        };
    }

    pub fn deinit(self: *JitContext) void {
        var it = self.compiled_functions.valueIterator();
        while (it.next()) |func| {
            func.deinit();
        }
        self.compiled_functions.deinit();
        if (self.schema) |*schema| {
            schema.deinit();
        }
    }

    /// Load schema from Lance file
    pub fn loadSchema(self: *JitContext, lance_file: anytype) !void {
        _ = lance_file;
        // TODO: Extract schema from LanceFile
        // For now, use a placeholder
        self.schema = LanceSchema{
            .columns = &.{},
            .allocator = self.allocator,
        };
    }

    /// Compile @logic_table Python source with schema
    pub fn compileLogicTable(
        self: *JitContext,
        python_source: []const u8,
        method_name: []const u8,
    ) !CompiledFunction {
        // Convert LanceQL schema to metal0 schema hints
        var metal0_columns: []const metal0.api.ColumnDef = &.{};
        if (self.schema) |schema| {
            var cols = std.ArrayList(metal0.api.ColumnDef).init(self.allocator);

            for (schema.columns) |col| {
                try cols.append(.{
                    .name = col.name,
                    .type = toMetal0Type(col.column_type),
                });
            }
            metal0_columns = try self.allocator.dupe(metal0.api.ColumnDef, cols.items);
            cols.deinit();
        }

        const schema_hints = metal0.SchemaTypeHints{
            .columns = metal0_columns,
            .force_concrete_types = true,
        };

        // Use metal0's schema-aware compilation API
        var result = try metal0.compileWithSchema(
            self.allocator,
            python_source,
            schema_hints,
            .{
                .output = .zig_source,
                .simd = true,
                .inline_all = true,
            },
        );

        // Copy source to our allocator
        const zig_source = try self.allocator.dupe(u8, result.zig_source);
        result.deinit(self.allocator);

        return CompiledFunction{
            .ptr = undefined, // TODO: JIT compile the source
            .source = zig_source,
            .name = method_name,
            .allocator = self.allocator,
        };
    }

    /// Convert LanceQL ColumnType to metal0 ColumnType
    fn toMetal0Type(col_type: ColumnType) metal0.ColumnType {
        return switch (col_type) {
            .i64 => .i64,
            .i32 => .i32,
            .i16 => .i16,
            .i8 => .i8,
            .u64 => .u64,
            .u32 => .u32,
            .u16 => .u16,
            .u8 => .u8,
            .f64 => .f64,
            .f32 => .f32,
            .bool => .bool,
            .string => .string,
            .bytes => .bytes,
            .vec_f32 => .vec_f32,
            .vec_f64 => .vec_f64,
        };
    }
};

/// Generate placeholder Zig code (until metal0 integration complete)
fn generatePlaceholderZig(
    allocator: std.mem.Allocator,
    python_source: []const u8,
    method_name: []const u8,
) ![]const u8 {
    _ = python_source;

    var code = std.ArrayList(u8).init(allocator);
    const writer = code.writer();

    try writer.writeAll("// Generated by LanceQL + metal0 JIT\n");
    try writer.writeAll("// TODO: Full integration pending\n\n");
    try writer.writeAll("const std = @import(\"std\");\n\n");

    try writer.print("pub fn {s}(\n", .{method_name});
    try writer.writeAll("    columns: anytype,\n");
    try writer.writeAll("    filtered_indices: []const u32,\n");
    try writer.writeAll("    results: []f64,\n");
    try writer.writeAll(") void {\n");
    try writer.writeAll("    // SIMD-optimized loop (placeholder)\n");
    try writer.writeAll("    for (filtered_indices) |idx| {\n");
    try writer.writeAll("        _ = columns;\n");
    try writer.writeAll("        results[idx] = 0.0; // TODO: actual computation\n");
    try writer.writeAll("    }\n");
    try writer.writeAll("}\n");

    return code.toOwnedSlice();
}

// =============================================================================
// Usage Example
// =============================================================================
//
// const jit = JitContext.init(allocator);
// defer jit.deinit();
//
// // Load schema from Lance file
// try jit.loadSchema(lance_file);
//
// // Compile @logic_table method
// const compiled = try jit.compileLogicTable(
//     \\@logic_table
//     \\class FraudDetector:
//     \\    def risk_score(self, amount: float, days: int) -> float:
//     \\        score = 0.0
//     \\        if amount > 10000: score += min(0.4, amount / 125000)
//     \\        if days < 30: score += 0.3
//     \\        return score
//     ,
//     "risk_score",
// );
//
// // Execute compiled function
// const func = @ptrCast(*const fn(Columns, []const u32, []f64) void, compiled.ptr);
// func(columns, filtered_indices, results);

test "ColumnType.fromLanceType" {
    try std.testing.expectEqual(ColumnType.i64, ColumnType.fromLanceType("int64").?);
    try std.testing.expectEqual(ColumnType.f64, ColumnType.fromLanceType("double").?);
    try std.testing.expectEqual(ColumnType.string, ColumnType.fromLanceType("utf8").?);
    try std.testing.expectEqual(@as(?ColumnType, null), ColumnType.fromLanceType("unknown"));
}

test "JitContext basic" {
    const allocator = std.testing.allocator;
    var jit = JitContext.init(allocator);
    defer jit.deinit();

    var func = try jit.compileLogicTable("def test(): pass", "test");
    defer func.deinit();

    try std.testing.expect(func.source.len > 0);
}
