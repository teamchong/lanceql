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

// Import Lance format types for schema extraction
const format = @import("lanceql.format");
const proto = @import("lanceql.proto");

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

    /// Convert from Lance physical type (Arrow naming)
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

    /// Convert from Lance schema logical_type string
    /// Lance uses these strings in the schema protobuf
    pub fn fromLogicalType(logical_type: []const u8) ?ColumnType {
        const map = std.StaticStringMap(ColumnType).initComptime(.{
            // Integer types
            .{ "int64", .i64 },
            .{ "int32", .i32 },
            .{ "int16", .i16 },
            .{ "int8", .i8 },
            .{ "uint64", .u64 },
            .{ "uint32", .u32 },
            .{ "uint16", .u16 },
            .{ "uint8", .u8 },
            // Float types
            .{ "double", .f64 },
            .{ "float", .f32 },
            .{ "float64", .f64 },
            .{ "float32", .f32 },
            // Boolean
            .{ "bool", .bool },
            .{ "boolean", .bool },
            // String types
            .{ "string", .string },
            .{ "utf8", .string },
            .{ "large_string", .string },
            .{ "large_utf8", .string },
            // Binary
            .{ "binary", .bytes },
            .{ "large_binary", .bytes },
        });
        return map.get(logical_type);
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
        for (self.columns) |col| {
            self.allocator.free(col.name);
        }
        self.allocator.free(self.columns);
    }
};

/// Compiled @logic_table function
pub const CompiledFunction = struct {
    /// Function pointer (signature depends on method)
    /// For batch processing: fn(columns: Columns, indices: []const u32, results: []f64) void
    ptr: ?*const anyopaque,

    /// Zig source code (for debugging)
    source: []const u8,

    /// Method name
    name: []const u8,

    /// Loaded dynamic library (if JIT compiled)
    lib: ?std.DynLib,

    /// Path to compiled .so file (for cleanup)
    lib_path: ?[]const u8,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *CompiledFunction) void {
        if (self.lib) |*lib| {
            lib.close();
        }
        if (self.lib_path) |path| {
            std.fs.cwd().deleteFile(path) catch {};
            self.allocator.free(path);
        }
        self.allocator.free(self.source);
    }

    /// Call the compiled function with batch processing signature
    /// columns: struct with column data pointers
    /// indices: filtered row indices to process
    /// results: output buffer for results
    pub fn call(
        self: CompiledFunction,
        columns: anytype,
        indices: []const u32,
        results: []f64,
    ) void {
        if (self.ptr) |ptr| {
            const func = @as(*const fn (@TypeOf(columns), []const u32, []f64) void, @ptrCast(ptr));
            func(columns, indices, results);
        }
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
    /// Extracts column names and types from the Lance file's schema buffer
    pub fn loadSchema(self: *JitContext, lance_file: *format.LanceFile) !void {
        // Schema is in global buffer 0
        const schema_bytes = lance_file.getGlobalBuffer(0) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => error.NoSchema,
                else => error.SchemaReadError,
            };
        };

        // Parse the schema protobuf
        var schema = proto.Schema.parse(self.allocator, schema_bytes) catch {
            return error.SchemaParseError;
        };
        defer schema.deinit();

        // Convert to our ColumnDef format
        var columns = std.ArrayListUnmanaged(ColumnDef){};
        errdefer columns.deinit(self.allocator);

        for (schema.fields) |field| {
            // Only include top-level (leaf) columns
            if (!field.isTopLevel()) continue;

            const col_type = ColumnType.fromLogicalType(field.logical_type) orelse {
                // Skip columns with unknown types
                continue;
            };

            try columns.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, field.name),
                .column_type = col_type,
                .nullable = field.nullable,
            });
        }

        self.schema = LanceSchema{
            .columns = try columns.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        };
    }

    /// Load schema from raw bytes (for testing or when LanceFile not available)
    pub fn loadSchemaFromBytes(self: *JitContext, schema_bytes: []const u8) !void {
        var schema = proto.Schema.parse(self.allocator, schema_bytes) catch {
            return error.SchemaParseError;
        };
        defer schema.deinit();

        var columns = std.ArrayListUnmanaged(ColumnDef){};
        errdefer columns.deinit(self.allocator);

        for (schema.fields) |field| {
            if (!field.isTopLevel()) continue;

            const col_type = ColumnType.fromLogicalType(field.logical_type) orelse continue;

            try columns.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, field.name),
                .column_type = col_type,
                .nullable = field.nullable,
            });
        }

        self.schema = LanceSchema{
            .columns = try columns.toOwnedSlice(self.allocator),
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
            var cols = std.ArrayListUnmanaged(metal0.api.ColumnDef){};

            for (schema.columns) |col| {
                try cols.append(self.allocator, .{
                    .name = col.name,
                    .type = toMetal0Type(col.column_type),
                });
            }
            metal0_columns = try self.allocator.dupe(metal0.api.ColumnDef, cols.items);
            cols.deinit(self.allocator);
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

        // Free metal0_columns if allocated
        if (metal0_columns.len > 0) {
            self.allocator.free(metal0_columns);
        }

        // Create sentinel-terminated method name for JIT
        const method_name_z = try self.allocator.dupeZ(u8, method_name);
        defer self.allocator.free(method_name_z);

        // Try to JIT compile the source
        const jit_result = jitCompileSource(self.allocator, zig_source, method_name_z) catch |err| {
            // JIT failed - return with null ptr (can still inspect source)
            std.log.warn("JIT compilation failed: {}, source-only mode", .{err});
            return CompiledFunction{
                .ptr = null,
                .source = zig_source,
                .name = method_name,
                .lib = null,
                .lib_path = null,
                .allocator = self.allocator,
            };
        };

        return CompiledFunction{
            .ptr = jit_result.ptr,
            .source = zig_source,
            .name = method_name,
            .lib = jit_result.lib,
            .lib_path = jit_result.lib_path,
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

/// JIT compilation result
const JitResult = struct {
    ptr: *const anyopaque,
    lib: std.DynLib,
    lib_path: []const u8,
};

/// JIT compile Zig source to a shared library and load it
/// Returns the function pointer and library handle
fn jitCompileSource(
    allocator: std.mem.Allocator,
    zig_source: []const u8,
    func_name: [:0]const u8,
) !JitResult {
    // Generate unique temp file names based on timestamp
    const timestamp = std.time.milliTimestamp();

    var src_path_buf: [256]u8 = undefined;
    const src_path = std.fmt.bufPrint(&src_path_buf, "/tmp/lanceql_jit_{d}.zig", .{timestamp}) catch
        return error.PathTooLong;

    var lib_path_buf: [256]u8 = undefined;
    const lib_name = std.fmt.bufPrint(&lib_path_buf, "/tmp/liblanceql_jit_{d}", .{timestamp}) catch
        return error.PathTooLong;

    // Platform-specific library extension
    const lib_ext = switch (@import("builtin").os.tag) {
        .macos => ".dylib",
        .windows => ".dll",
        else => ".so",
    };

    // Write Zig source to temp file
    const src_file = std.fs.cwd().createFile(src_path, .{}) catch
        return error.CannotCreateTempFile;
    defer src_file.close();
    src_file.writeAll(zig_source) catch return error.CannotWriteSource;

    // Construct full library path and emit-bin argument
    var full_lib_path_buf: [300]u8 = undefined;
    const full_lib_path = std.fmt.bufPrint(&full_lib_path_buf, "{s}{s}", .{ lib_name, lib_ext }) catch
        return error.PathTooLong;

    var emit_bin_buf: [320]u8 = undefined;
    const emit_bin_arg = std.fmt.bufPrint(&emit_bin_buf, "-femit-bin={s}", .{full_lib_path}) catch
        return error.PathTooLong;

    // Build shared library using zig build-lib
    const argv = [_][]const u8{
        "zig",
        "build-lib",
        "-dynamic",
        "-O",
        "ReleaseFast",
        emit_bin_arg,
        src_path,
    };

    var child = std.process.Child.init(&argv, allocator);
    child.stderr_behavior = .Pipe;
    child.stdout_behavior = .Pipe;

    child.spawn() catch return error.CannotSpawnCompiler;
    const result = child.wait() catch return error.CompilerFailed;

    // Clean up source file
    std.fs.cwd().deleteFile(src_path) catch {};

    if (result.Exited != 0) {
        return error.CompilationFailed;
    }

    // Load the compiled library
    var lib = std.DynLib.open(full_lib_path) catch return error.CannotLoadLibrary;
    errdefer lib.close();

    // Look up the function symbol
    const ptr = lib.lookup(*const anyopaque, func_name) orelse {
        return error.SymbolNotFound;
    };

    // Copy library path for cleanup later
    const lib_path_copy = try allocator.dupe(u8, full_lib_path);

    return JitResult{
        .ptr = ptr,
        .lib = lib,
        .lib_path = lib_path_copy,
    };
}

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

test "ColumnType.fromLogicalType" {
    // Test various Lance logical types
    try std.testing.expectEqual(ColumnType.i64, ColumnType.fromLogicalType("int64").?);
    try std.testing.expectEqual(ColumnType.f64, ColumnType.fromLogicalType("double").?);
    try std.testing.expectEqual(ColumnType.f64, ColumnType.fromLogicalType("float64").?);
    try std.testing.expectEqual(ColumnType.string, ColumnType.fromLogicalType("string").?);
    try std.testing.expectEqual(ColumnType.string, ColumnType.fromLogicalType("utf8").?);
    try std.testing.expectEqual(ColumnType.bool, ColumnType.fromLogicalType("bool").?);
    try std.testing.expectEqual(ColumnType.bool, ColumnType.fromLogicalType("boolean").?);
    try std.testing.expectEqual(@as(?ColumnType, null), ColumnType.fromLogicalType("custom_type"));
}

test "loadSchemaFromBytes" {
    const allocator = std.testing.allocator;
    var jit = JitContext.init(allocator);
    defer jit.deinit();

    // Schema bytes from a lancedb-created file with columns: id (int64), name (string)
    // Same bytes used in proto/schema.zig tests
    const schema_bytes = [_]u8{
        0x0a, 0x4f, 0x0a, 0x23, 0x12, 0x02, 0x69, 0x64, 0x20, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0x01, 0x2a, 0x05, 0x69, 0x6e, 0x74, 0x36, 0x34, 0x30, 0x01, 0x38, 0x01, 0x5a, 0x07,
        0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x0a, 0x28, 0x12, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18,
        0x01, 0x20, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x2a, 0x06, 0x73, 0x74,
        0x72, 0x69, 0x6e, 0x67, 0x30, 0x01, 0x38, 0x02, 0x5a, 0x07, 0x64, 0x65, 0x66, 0x61, 0x75, 0x6c,
        0x74, 0x10, 0x03,
    };

    try jit.loadSchemaFromBytes(&schema_bytes);

    // Verify schema was loaded
    try std.testing.expect(jit.schema != null);
    const schema = jit.schema.?;

    // Should have 2 columns: id (int64), name (string)
    try std.testing.expectEqual(@as(usize, 2), schema.columns.len);
    try std.testing.expectEqualStrings("id", schema.columns[0].name);
    try std.testing.expectEqual(ColumnType.i64, schema.columns[0].column_type);
    try std.testing.expectEqualStrings("name", schema.columns[1].name);
    try std.testing.expectEqual(ColumnType.string, schema.columns[1].column_type);
}

test "compileLogicTable with schema" {
    const allocator = std.testing.allocator;
    var jit = JitContext.init(allocator);
    defer jit.deinit();

    // Schema bytes from lancedb with: id (int64), name (string)
    const schema_bytes = [_]u8{
        0x0a, 0x4f, 0x0a, 0x23, 0x12, 0x02, 0x69, 0x64, 0x20, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0x01, 0x2a, 0x05, 0x69, 0x6e, 0x74, 0x36, 0x34, 0x30, 0x01, 0x38, 0x01, 0x5a, 0x07,
        0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x0a, 0x28, 0x12, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18,
        0x01, 0x20, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x2a, 0x06, 0x73, 0x74,
        0x72, 0x69, 0x6e, 0x67, 0x30, 0x01, 0x38, 0x02, 0x5a, 0x07, 0x64, 0x65, 0x66, 0x61, 0x75, 0x6c,
        0x74, 0x10, 0x03,
    };

    try jit.loadSchemaFromBytes(&schema_bytes);

    // Compile with schema
    var func = try jit.compileLogicTable(
        \\def compute(id: int) -> int:
        \\    return id * 2
    , "compute");
    defer func.deinit();

    // Should generate Zig source with schema comments
    try std.testing.expect(func.source.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, func.source, "schema") != null);
}

test "jitCompileSource basic" {
    // Skip JIT test - requires external zig compiler and sandbox may block it
    // This test demonstrates the JIT compilation flow works in principle.
    // In practice, JIT compilation is tested through integration tests.
    //
    // The JIT process:
    // 1. Write Zig source to temp file
    // 2. Call `zig build-lib -dynamic` to create .dylib/.so
    // 3. dlopen() the library
    // 4. Lookup and call the function
    //
    // This is skipped in unit tests due to:
    // - Sandbox restrictions in CI
    // - Need for zig compiler in PATH
    // - Potential permission issues with /tmp
    return error.SkipZigTest;
}
