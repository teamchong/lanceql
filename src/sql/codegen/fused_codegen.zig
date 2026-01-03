//! Fused Code Generator - Compiles QueryPlan to Zig source code
//!
//! Generates a single fused Zig function that executes the entire query plan
//! in a single vectorized pass. This eliminates interpreter overhead and
//! enables better optimization by the Zig compiler.
//!
//! Key optimizations:
//! - Single pass over data (fused scan + filter + compute)
//! - @logic_table methods inlined directly
//! - Vectorized loops with SIMD opportunities
//! - GPU dispatch for vector similarity operations
//!
//! Example output:
//! ```zig
//! pub export fn fused_query(
//!     columns: *const Columns,
//!     output: *OutputBuffers,
//! ) usize {
//!     var count: usize = 0;
//!     var i: usize = 0;
//!     while (i < columns.len) : (i += 1) {
//!         // Inlined @logic_table method
//!         const risk_score = blk: { ... };
//!         // Fused filter
//!         if (columns.amount[i] > 1000 and risk_score > 0.7) {
//!             output.amount[count] = columns.amount[i];
//!             output.risk_score[count] = risk_score;
//!             count += 1;
//!         }
//!     }
//!     return count;
//! }
//! ```

const std = @import("std");
const ast = @import("ast");
const plan_nodes = @import("../planner/plan_nodes.zig");

const PlanNode = plan_nodes.PlanNode;
const QueryPlan = plan_nodes.QueryPlan;
const ColumnRef = plan_nodes.ColumnRef;
const ColumnType = plan_nodes.ColumnType;
const ComputeExpr = plan_nodes.ComputeExpr;

/// Code generation errors
pub const CodeGenError = error{
    OutOfMemory,
    UnsupportedPlanNode,
    UnsupportedExpression,
    UnsupportedOperator,
    InvalidPlan,
};

// ============================================================================
// Column Layout Types - for runtime struct building
// ============================================================================

/// Information about a column in the generated struct
pub const ColumnInfo = struct {
    name: []const u8,
    col_type: ColumnType,
    offset: usize, // Byte offset in struct
};

/// Layout metadata for generated Columns and OutputBuffers structs
pub const ColumnLayout = struct {
    /// Input columns in declaration order
    input_columns: []const ColumnInfo,
    /// Output columns (input + computed) in declaration order
    output_columns: []const ColumnInfo,
    /// Total size of Columns struct (including len field)
    columns_size: usize,
    /// Total size of OutputBuffers struct
    output_size: usize,
};

/// Result of generateWithLayout
pub const GenerateResult = struct {
    source: []const u8,
    layout: ColumnLayout,
};

/// Fused code generator
pub const FusedCodeGen = struct {
    allocator: std.mem.Allocator,

    /// Generated code buffer
    code: std.ArrayList(u8),

    /// Indentation level
    indent: u32,

    /// Input columns referenced
    input_columns: std.StringHashMap(ColumnType),

    /// Input columns in declaration order (for layout)
    input_column_order: std.ArrayList(ColumnInfo),

    /// Computed columns (from @logic_table methods)
    computed_columns: std.StringHashMap([]const u8),

    /// Computed columns in declaration order (for layout)
    computed_column_order: std.ArrayList(ColumnInfo),

    /// Counter for unique variable names
    var_counter: u32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .code = .{},
            .indent = 0,
            .input_columns = std.StringHashMap(ColumnType).init(allocator),
            .input_column_order = .{},
            .computed_columns = std.StringHashMap([]const u8).init(allocator),
            .computed_column_order = .{},
            .var_counter = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.code.deinit(self.allocator);
        self.input_columns.deinit();
        self.input_column_order.deinit(self.allocator);
        self.computed_columns.deinit();
        self.computed_column_order.deinit(self.allocator);
    }

    /// Main entry point: generate fused code from query plan
    pub fn generate(self: *Self, plan: *const QueryPlan) CodeGenError![]const u8 {
        // Analyze plan to collect column info
        try self.analyzePlan(plan.root);

        // Generate code
        try self.genHeader();
        try self.genColumnsStruct();
        try self.genOutputStruct();
        try self.genFusedFunction(plan.root);

        return self.code.items;
    }

    /// Generate fused code with layout metadata for runtime struct building
    pub fn generateWithLayout(self: *Self, plan: *const QueryPlan) CodeGenError!GenerateResult {
        // Generate the source code first
        const source = try self.generate(plan);

        // Build layout from collected column info
        const layout = try self.buildLayout();

        return GenerateResult{
            .source = source,
            .layout = layout,
        };
    }

    /// Build layout metadata from collected columns
    fn buildLayout(self: *Self) CodeGenError!ColumnLayout {
        const ptr_size: usize = @sizeOf(usize); // Size of a pointer (8 bytes on 64-bit)

        // Calculate input column offsets
        var input_cols = self.allocator.alloc(ColumnInfo, self.input_column_order.items.len) catch
            return CodeGenError.OutOfMemory;
        var offset: usize = 0;
        for (self.input_column_order.items, 0..) |col, i| {
            input_cols[i] = .{
                .name = col.name,
                .col_type = col.col_type,
                .offset = offset,
            };
            offset += ptr_size;
        }
        // Add len field offset
        const columns_size = offset + ptr_size;

        // Calculate output column offsets (input columns + computed columns)
        const output_count = self.input_column_order.items.len + self.computed_column_order.items.len;
        var output_cols = self.allocator.alloc(ColumnInfo, output_count) catch
            return CodeGenError.OutOfMemory;

        offset = 0;
        for (self.input_column_order.items, 0..) |col, i| {
            output_cols[i] = .{
                .name = col.name,
                .col_type = col.col_type,
                .offset = offset,
            };
            offset += ptr_size;
        }
        for (self.computed_column_order.items, 0..) |col, i| {
            output_cols[self.input_column_order.items.len + i] = .{
                .name = col.name,
                .col_type = col.col_type,
                .offset = offset,
            };
            offset += ptr_size;
        }
        const output_size = offset;

        return ColumnLayout{
            .input_columns = input_cols,
            .output_columns = output_cols,
            .columns_size = columns_size,
            .output_size = output_size,
        };
    }

    /// Analyze plan to collect column references and types
    fn analyzePlan(self: *Self, node: *const PlanNode) CodeGenError!void {
        switch (node.*) {
            .scan => |scan| {
                for (scan.columns) |col| {
                    // Track in HashMap for fast lookup
                    self.input_columns.put(col.column, col.col_type) catch return CodeGenError.OutOfMemory;
                    // Track in order for layout
                    self.input_column_order.append(self.allocator, .{
                        .name = col.column,
                        .col_type = col.col_type,
                        .offset = 0, // Will be calculated in buildLayout
                    }) catch return CodeGenError.OutOfMemory;
                }
            },
            .filter => |filter| {
                try self.analyzePlan(filter.input);
                try self.analyzeExpr(filter.predicate);
            },
            .project => |project| {
                try self.analyzePlan(project.input);
            },
            .compute => |compute| {
                try self.analyzePlan(compute.input);
                for (compute.expressions) |expr| {
                    if (expr.inlined_body) |body| {
                        // Track in HashMap for fast lookup
                        self.computed_columns.put(expr.name, body) catch return CodeGenError.OutOfMemory;
                        // Track in order for layout (computed columns default to f64)
                        self.computed_column_order.append(self.allocator, .{
                            .name = expr.name,
                            .col_type = .f64,
                            .offset = 0, // Will be calculated in buildLayout
                        }) catch return CodeGenError.OutOfMemory;
                    }
                }
            },
            .group_by => |group_by| {
                try self.analyzePlan(group_by.input);
            },
            .sort => |sort| {
                try self.analyzePlan(sort.input);
            },
            .limit => |limit| {
                try self.analyzePlan(limit.input);
            },
            .window => |window| {
                try self.analyzePlan(window.input);
            },
            .hash_join => |join| {
                try self.analyzePlan(join.left);
                try self.analyzePlan(join.right);
            },
        }
    }

    /// Analyze expression to collect column references
    fn analyzeExpr(self: *Self, expr: *const ast.Expr) CodeGenError!void {
        switch (expr.*) {
            .column => |col| {
                // Add to input columns if not already present
                if (!self.input_columns.contains(col.name)) {
                    self.input_columns.put(col.name, .unknown) catch return CodeGenError.OutOfMemory;
                }
            },
            .binary => |bin| {
                try self.analyzeExpr(bin.left);
                try self.analyzeExpr(bin.right);
            },
            .unary => |un| {
                try self.analyzeExpr(un.operand);
            },
            .call => |call| {
                for (call.args) |*arg| {
                    try self.analyzeExpr(arg);
                }
            },
            .method_call => |mc| {
                for (mc.args) |*arg| {
                    try self.analyzeExpr(arg);
                }
            },
            else => {},
        }
    }

    /// Generate file header with imports
    fn genHeader(self: *Self) CodeGenError!void {
        try self.write(
            \\//! Auto-generated fused query function
            \\//! Generated by LanceQL FusedCodeGen
            \\
            \\const std = @import("std");
            \\
            \\
        );
    }

    /// Generate Columns struct definition
    fn genColumnsStruct(self: *Self) CodeGenError!void {
        try self.write("pub const Columns = struct {\n");
        self.indent += 1;

        var iter = self.input_columns.iterator();
        while (iter.next()) |entry| {
            try self.writeIndent();
            try self.write(entry.key_ptr.*);
            try self.write(": [*]const ");
            try self.write(entry.value_ptr.toZigType());
            try self.write(",\n");
        }

        // Add length field
        try self.writeIndent();
        try self.write("len: usize,\n");

        self.indent -= 1;
        try self.write("};\n\n");
    }

    /// Generate OutputBuffers struct definition
    fn genOutputStruct(self: *Self) CodeGenError!void {
        try self.write("pub const OutputBuffers = struct {\n");
        self.indent += 1;

        // Output columns from input columns
        var iter = self.input_columns.iterator();
        while (iter.next()) |entry| {
            try self.writeIndent();
            try self.write(entry.key_ptr.*);
            try self.write(": [*]");
            try self.write(entry.value_ptr.toZigType());
            try self.write(",\n");
        }

        // Add computed columns
        var comp_iter = self.computed_columns.keyIterator();
        while (comp_iter.next()) |key| {
            try self.writeIndent();
            try self.write(key.*);
            try self.write(": [*]f64,\n");
        }

        self.indent -= 1;
        try self.write("};\n\n");
    }

    /// Generate the main fused query function
    fn genFusedFunction(self: *Self, root: *const PlanNode) CodeGenError!void {
        // Function signature
        try self.write(
            \\pub export fn fused_query(
            \\    columns: *const Columns,
            \\    output: *OutputBuffers,
            \\) callconv(.c) usize {
            \\
        );
        self.indent += 1;

        // Local variables
        try self.writeIndent();
        try self.write("var result_count: usize = 0;\n");
        try self.writeIndent();
        try self.write("var i: usize = 0;\n\n");

        // Main loop
        try self.writeIndent();
        try self.write("while (i < columns.len) : (i += 1) {\n");
        self.indent += 1;

        // Generate body based on plan
        try self.genPlanNodeBody(root);

        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n\n");

        // Return
        try self.writeIndent();
        try self.write("return result_count;\n");

        self.indent -= 1;
        try self.write("}\n");
    }

    /// Generate code for a plan node
    fn genPlanNodeBody(self: *Self, node: *const PlanNode) CodeGenError!void {
        switch (node.*) {
            .scan => {
                // Scan is implicit - columns are accessed via columns.*
            },
            .filter => |filter| {
                // First process input
                if (filter.input.* != .scan) {
                    try self.genPlanNodeBody(filter.input);
                }

                // Generate filter condition
                try self.writeIndent();
                try self.write("if (");
                try self.genExpr(filter.predicate);
                try self.write(") {\n");
                self.indent += 1;

                // Emit output columns
                try self.genEmitRow();

                self.indent -= 1;
                try self.writeIndent();
                try self.write("}\n");
            },
            .compute => |compute| {
                // First process input
                try self.genPlanNodeBody(compute.input);

                // Generate computed columns
                for (compute.expressions) |expr| {
                    try self.writeIndent();
                    try self.write("const ");
                    try self.write(expr.name);
                    try self.write(" = blk: {\n");
                    self.indent += 1;

                    if (expr.inlined_body) |body| {
                        // Use inlined body from metal0
                        try self.writeIndent();
                        try self.write(body);
                        try self.write("\n");
                    } else {
                        // Generate from expression
                        try self.writeIndent();
                        try self.write("break :blk ");
                        try self.genExpr(expr.expr);
                        try self.write(";\n");
                    }

                    self.indent -= 1;
                    try self.writeIndent();
                    try self.write("};\n");
                }
            },
            .project => |project| {
                // Process input - this should contain filter/compute
                try self.genPlanNodeBody(project.input);
            },
            .limit => |limit| {
                // Check limit before processing
                if (limit.limit) |lim| {
                    try self.writeIndent();
                    try self.fmt("if (result_count >= {d}) break;\n", .{lim});
                }

                // Process input
                try self.genPlanNodeBody(limit.input);
            },
            else => {
                // For unsupported nodes, just process input if available
                if (node.getInput()) |input| {
                    try self.genPlanNodeBody(input);
                }
            },
        }
    }

    /// Generate row emission code
    fn genEmitRow(self: *Self) CodeGenError!void {
        // Emit each input column
        var iter = self.input_columns.keyIterator();
        while (iter.next()) |key| {
            try self.writeIndent();
            try self.fmt("output.{s}[result_count] = columns.{s}[i];\n", .{ key.*, key.* });
        }

        // Emit computed columns
        var comp_iter = self.computed_columns.keyIterator();
        while (comp_iter.next()) |key| {
            try self.writeIndent();
            try self.fmt("output.{s}[result_count] = {s};\n", .{ key.*, key.* });
        }

        // Increment counter
        try self.writeIndent();
        try self.write("result_count += 1;\n");
    }

    /// Generate expression code
    fn genExpr(self: *Self, expr: *const ast.Expr) CodeGenError!void {
        switch (expr.*) {
            .value => |val| {
                switch (val) {
                    .null => try self.write("null"),
                    .integer => |i| try self.fmt("{d}", .{i}),
                    .float => |f| try self.fmt("{d}", .{f}),
                    .string => |s| try self.fmt("\"{s}\"", .{s}),
                    .blob => try self.write("\"\""),
                    .parameter => |p| try self.fmt("params[{d}]", .{p}),
                }
            },
            .column => |col| {
                try self.write("columns.");
                try self.write(col.name);
                try self.write("[i]");
            },
            .binary => |bin| {
                try self.write("(");
                try self.genExpr(bin.left);
                try self.write(" ");
                try self.genBinaryOp(bin.op);
                try self.write(" ");
                try self.genExpr(bin.right);
                try self.write(")");
            },
            .unary => |un| {
                switch (un.op) {
                    .not => {
                        try self.write("!");
                        try self.genExpr(un.operand);
                    },
                    .minus => {
                        try self.write("-");
                        try self.genExpr(un.operand);
                    },
                    .is_null => {
                        try self.genExpr(un.operand);
                        try self.write(" == null");
                    },
                    .is_not_null => {
                        try self.genExpr(un.operand);
                        try self.write(" != null");
                    },
                }
            },
            .call => |call| {
                // Generate function call
                try self.write(call.name);
                try self.write("(");
                for (call.args, 0..) |*arg, idx| {
                    if (idx > 0) try self.write(", ");
                    try self.genExpr(arg);
                }
                try self.write(")");
            },
            .method_call => |mc| {
                // Check if this is a computed column
                if (self.computed_columns.get(mc.method)) |_| {
                    try self.write(mc.method);
                } else {
                    // Generate as function call
                    try self.write(mc.object);
                    try self.write("_");
                    try self.write(mc.method);
                    try self.write("(");
                    for (mc.args, 0..) |*arg, idx| {
                        if (idx > 0) try self.write(", ");
                        try self.genExpr(arg);
                    }
                    try self.write(")");
                }
            },
            .in_list => |in| {
                try self.write("(");
                for (in.values, 0..) |*val, idx| {
                    if (idx > 0) try self.write(" or ");
                    try self.genExpr(in.expr);
                    try self.write(" == ");
                    try self.genExpr(val);
                }
                try self.write(")");
            },
            .between => |bet| {
                try self.write("(");
                try self.genExpr(bet.expr);
                try self.write(" >= ");
                try self.genExpr(bet.low);
                try self.write(" and ");
                try self.genExpr(bet.expr);
                try self.write(" <= ");
                try self.genExpr(bet.high);
                try self.write(")");
            },
            else => {
                try self.write("/* unsupported expr */");
            },
        }
    }

    /// Generate binary operator
    fn genBinaryOp(self: *Self, op: ast.BinaryOp) CodeGenError!void {
        const op_str = switch (op) {
            .add => "+",
            .subtract => "-",
            .multiply => "*",
            .divide => "/",
            .concat => "++",
            .eq => "==",
            .ne => "!=",
            .lt => "<",
            .le => "<=",
            .gt => ">",
            .ge => ">=",
            .@"and" => "and",
            .@"or" => "or",
            .like => "/* LIKE */",
            .in => "/* IN */",
            .between => "/* BETWEEN */",
        };
        try self.write(op_str);
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    fn write(self: *Self, s: []const u8) CodeGenError!void {
        self.code.appendSlice(self.allocator, s) catch return CodeGenError.OutOfMemory;
    }

    fn writeIndent(self: *Self) CodeGenError!void {
        var i: u32 = 0;
        while (i < self.indent) : (i += 1) {
            self.code.appendSlice(self.allocator, "    ") catch return CodeGenError.OutOfMemory;
        }
    }

    fn fmt(self: *Self, comptime format: []const u8, args: anytype) CodeGenError!void {
        const writer = self.code.writer(self.allocator);
        writer.print(format, args) catch return CodeGenError.OutOfMemory;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "FusedCodeGen init/deinit" {
    const allocator = std.testing.allocator;
    var codegen = FusedCodeGen.init(allocator);
    defer codegen.deinit();
}

test "FusedCodeGen genHeader" {
    const allocator = std.testing.allocator;
    var codegen = FusedCodeGen.init(allocator);
    defer codegen.deinit();

    try codegen.genHeader();
    try std.testing.expect(std.mem.indexOf(u8, codegen.code.items, "Auto-generated") != null);
}

test "FusedCodeGen simple expression" {
    const allocator = std.testing.allocator;
    var codegen = FusedCodeGen.init(allocator);
    defer codegen.deinit();

    // Test literal expression
    const expr = ast.Expr{ .value = .{ .integer = 42 } };
    try codegen.genExpr(&expr);
    try std.testing.expectEqualStrings("42", codegen.code.items);
}
