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
const WindowFuncType = plan_nodes.WindowFuncType;
const WindowFuncSpec = plan_nodes.WindowFuncSpec;
const OrderBySpec = plan_nodes.OrderBySpec;

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
    /// Allocator used to create the slices
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ColumnLayout) void {
        self.allocator.free(self.input_columns);
        self.allocator.free(self.output_columns);
    }
};

/// Result of generateWithLayout
pub const GenerateResult = struct {
    source: []const u8,
    layout: ColumnLayout,
};

/// Window function info for code generation
pub const WindowFuncInfo = struct {
    /// Output column name
    name: []const u8,
    /// Window function type
    func_type: WindowFuncType,
    /// Partition column names
    partition_cols: []const []const u8,
    /// Order column names
    order_cols: []const []const u8,
    /// Order direction (true = DESC)
    order_desc: []const bool,
    /// Source column for LAG/LEAD (optional)
    source_col: ?[]const u8 = null,
    /// Offset for LAG/LEAD (default 1)
    offset: i64 = 1,
    /// Default value for LAG/LEAD
    default_val: i64 = 0,
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

    /// Window function specifications collected during analysis
    window_specs: std.ArrayList(WindowFuncInfo),

    /// Set of window column names for fast lookup
    window_columns: std.StringHashMap(void),

    /// Counter for unique variable names
    var_counter: u32,

    /// Analyzed plan (stored for generateCode)
    analyzed_plan: ?*const QueryPlan,

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
            .window_specs = .{},
            .window_columns = std.StringHashMap(void).init(allocator),
            .var_counter = 0,
            .analyzed_plan = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.code.deinit(self.allocator);
        self.input_columns.deinit();
        self.input_column_order.deinit(self.allocator);
        self.computed_columns.deinit();
        self.computed_column_order.deinit(self.allocator);
        self.window_specs.deinit(self.allocator);
        self.window_columns.deinit();
    }

    /// Analyze plan to collect column info (call before updateColumnTypes)
    pub fn analyze(self: *Self, plan: *const QueryPlan) CodeGenError!void {
        try self.analyzePlan(plan.root);
        self.analyzed_plan = plan;
    }

    /// Generate code after analysis (and optional updateColumnTypes)
    pub fn generateCode(self: *Self) CodeGenError![]const u8 {
        const plan = self.analyzed_plan orelse return CodeGenError.InvalidPlan;

        // Generate code
        try self.genHeader();
        try self.genColumnsStruct();
        try self.genOutputStruct();
        try self.genWindowFunctions(); // Window helper functions before main function
        try self.genFusedFunction(plan.root);

        return self.code.items;
    }

    /// Main entry point: generate fused code from query plan
    /// (For backwards compatibility - use analyze + updateColumnTypes + generateCode for type resolution)
    pub fn generate(self: *Self, plan: *const QueryPlan) CodeGenError![]const u8 {
        // Analyze plan to collect column info
        try self.analyze(plan);
        // Generate code
        return self.generateCode();
    }

    /// Update column types based on type map
    /// Call this after analyze but before generateCode if column types are unknown
    pub fn updateColumnTypes(self: *Self, type_map: *const std.StringHashMap(ColumnType)) void {
        // Update input_columns HashMap
        var iter = self.input_columns.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* == .unknown) {
                if (type_map.get(entry.key_ptr.*)) |actual_type| {
                    entry.value_ptr.* = actual_type;
                }
            }
        }

        // Update input_column_order ArrayList
        for (self.input_column_order.items) |*col| {
            if (col.col_type == .unknown) {
                if (type_map.get(col.name)) |actual_type| {
                    col.col_type = actual_type;
                }
            }
        }
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

    /// Generate fused code with type resolution from schema
    /// This is the preferred method when column types need to be resolved from table schema
    pub fn generateWithLayoutAndTypes(
        self: *Self,
        plan: *const QueryPlan,
        type_map: *const std.StringHashMap(ColumnType),
    ) CodeGenError!GenerateResult {
        // Step 1: Analyze to collect column info
        try self.analyze(plan);

        // Step 2: Resolve unknown types from schema
        self.updateColumnTypes(type_map);

        // Step 3: Generate code with resolved types
        const source = try self.generateCode();

        // Step 4: Build layout
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

        // Calculate output column offsets (input columns + computed columns + window columns)
        const output_count = self.input_column_order.items.len + self.computed_column_order.items.len + self.window_specs.items.len;
        var output_cols = self.allocator.alloc(ColumnInfo, output_count) catch
            return CodeGenError.OutOfMemory;

        offset = 0;
        var out_idx: usize = 0;
        for (self.input_column_order.items) |col| {
            output_cols[out_idx] = .{
                .name = col.name,
                .col_type = col.col_type,
                .offset = offset,
            };
            offset += ptr_size;
            out_idx += 1;
        }
        for (self.computed_column_order.items) |col| {
            output_cols[out_idx] = .{
                .name = col.name,
                .col_type = col.col_type,
                .offset = offset,
            };
            offset += ptr_size;
            out_idx += 1;
        }
        // Add window columns (all i64 for ranking functions)
        for (self.window_specs.items) |spec| {
            output_cols[out_idx] = .{
                .name = spec.name,
                .col_type = .i64,
                .offset = offset,
            };
            offset += ptr_size;
            out_idx += 1;
        }
        const output_size = offset;

        return ColumnLayout{
            .input_columns = input_cols,
            .output_columns = output_cols,
            .columns_size = columns_size,
            .output_size = output_size,
            .allocator = self.allocator,
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

                // Collect window function info
                for (window.windows) |spec| {
                    // Extract partition column names
                    var partition_names = self.allocator.alloc([]const u8, spec.partition_by.len) catch
                        return CodeGenError.OutOfMemory;
                    for (spec.partition_by, 0..) |col_ref, i| {
                        partition_names[i] = col_ref.column;
                        // Add to input columns
                        try self.addInputColumn(col_ref.column, col_ref.col_type);
                    }

                    // Extract order column names and directions
                    var order_names = self.allocator.alloc([]const u8, spec.order_by.len) catch
                        return CodeGenError.OutOfMemory;
                    var order_desc = self.allocator.alloc(bool, spec.order_by.len) catch
                        return CodeGenError.OutOfMemory;
                    for (spec.order_by, 0..) |order_spec, i| {
                        order_names[i] = order_spec.column.column;
                        order_desc[i] = order_spec.direction == .desc;
                        // Add to input columns
                        try self.addInputColumn(order_spec.column.column, order_spec.column.col_type);
                    }

                    // Track window function
                    self.window_specs.append(self.allocator, .{
                        .name = spec.name,
                        .func_type = spec.func_type,
                        .partition_cols = partition_names,
                        .order_cols = order_names,
                        .order_desc = order_desc,
                    }) catch return CodeGenError.OutOfMemory;

                    // Track as window column for expression generation
                    self.window_columns.put(spec.name, {}) catch return CodeGenError.OutOfMemory;
                }
            },
            .hash_join => |join| {
                try self.analyzePlan(join.left);
                try self.analyzePlan(join.right);
            },
        }
    }

    /// Add a column to input columns (if not already present)
    fn addInputColumn(self: *Self, name: []const u8, col_type: ColumnType) CodeGenError!void {
        if (!self.input_columns.contains(name)) {
            self.input_columns.put(name, col_type) catch return CodeGenError.OutOfMemory;
            self.input_column_order.append(self.allocator, .{
                .name = name,
                .col_type = col_type,
                .offset = 0,
            }) catch return CodeGenError.OutOfMemory;
        }
    }

    /// Analyze expression to collect column references
    fn analyzeExpr(self: *Self, expr: *const ast.Expr) CodeGenError!void {
        switch (expr.*) {
            .column => |col| {
                // Add to input columns if not already present
                if (!self.input_columns.contains(col.name)) {
                    self.input_columns.put(col.name, .unknown) catch return CodeGenError.OutOfMemory;
                    // Also add to order list for layout building
                    self.input_column_order.append(self.allocator, .{
                        .name = col.name,
                        .col_type = .unknown,
                        .offset = 0,
                    }) catch return CodeGenError.OutOfMemory;
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

        // Add window columns (all ranking functions output i64)
        for (self.window_specs.items) |spec| {
            try self.writeIndent();
            try self.write(spec.name);
            try self.write(": [*]i64,\n");
        }

        self.indent -= 1;
        try self.write("};\n\n");
    }

    /// Generate window helper functions
    /// These are called once at the start to pre-compute window values for all rows
    fn genWindowFunctions(self: *Self) CodeGenError!void {
        for (self.window_specs.items) |spec| {
            switch (spec.func_type) {
                .row_number => try self.genWindowRowNumber(spec),
                .rank => try self.genWindowRank(spec, false),
                .dense_rank => try self.genWindowRank(spec, true),
                .lag => try self.genWindowLagLead(spec, true),
                .lead => try self.genWindowLagLead(spec, false),
                else => {}, // Other window functions not yet supported
            }
        }
    }

    /// Generate ROW_NUMBER window function
    fn genWindowRowNumber(self: *Self, spec: WindowFuncInfo) CodeGenError!void {
        try self.fmt("fn computeWindow_{s}(columns: *const Columns, results: []i64) void {{\n", .{spec.name});
        self.indent += 1;

        // Create index array
        try self.writeIndent();
        try self.write("var indices: [4096]u32 = undefined;\n");
        try self.writeIndent();
        try self.write("var idx: u32 = 0;\n");
        try self.writeIndent();
        try self.write("while (idx < columns.len) : (idx += 1) indices[idx] = idx;\n\n");

        // Generate sort - first by partition columns, then by order columns
        if (spec.partition_cols.len > 0 or spec.order_cols.len > 0) {
            try self.genWindowSort(spec);
        }

        // Compute row numbers
        try self.writeIndent();
        if (spec.partition_cols.len > 0) {
            // With partitions
            try self.fmt("var current_partition: i64 = columns.{s}[indices[0]];\n", .{spec.partition_cols[0]});
            try self.writeIndent();
            try self.write("var row_num: i64 = 0;\n");
            try self.writeIndent();
            try self.write("for (indices[0..columns.len]) |i| {\n");
            self.indent += 1;
            try self.writeIndent();
            try self.fmt("if (columns.{s}[i] != current_partition) {{\n", .{spec.partition_cols[0]});
            self.indent += 1;
            try self.writeIndent();
            try self.fmt("current_partition = columns.{s}[i];\n", .{spec.partition_cols[0]});
            try self.writeIndent();
            try self.write("row_num = 0;\n");
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");
            try self.writeIndent();
            try self.write("row_num += 1;\n");
            try self.writeIndent();
            try self.write("results[i] = row_num;\n");
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");
        } else {
            // No partitions - all rows in one group
            try self.write("for (indices[0..columns.len], 1..) |i, row_num| {\n");
            self.indent += 1;
            try self.writeIndent();
            try self.write("results[i] = @intCast(row_num);\n");
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");
        }

        self.indent -= 1;
        try self.write("}\n\n");
    }

    /// Generate RANK or DENSE_RANK window function
    fn genWindowRank(self: *Self, spec: WindowFuncInfo, dense: bool) CodeGenError!void {
        const func_name = if (dense) "dense_rank" else "rank";
        _ = func_name;

        try self.fmt("fn computeWindow_{s}(columns: *const Columns, results: []i64) void {{\n", .{spec.name});
        self.indent += 1;

        // Create index array
        try self.writeIndent();
        try self.write("var indices: [4096]u32 = undefined;\n");
        try self.writeIndent();
        try self.write("var idx: u32 = 0;\n");
        try self.writeIndent();
        try self.write("while (idx < columns.len) : (idx += 1) indices[idx] = idx;\n\n");

        // Generate sort
        if (spec.partition_cols.len > 0 or spec.order_cols.len > 0) {
            try self.genWindowSort(spec);
        }

        // Compute ranks
        try self.writeIndent();
        if (spec.partition_cols.len > 0) {
            try self.fmt("var current_partition: i64 = columns.{s}[indices[0]];\n", .{spec.partition_cols[0]});
        }
        try self.writeIndent();
        try self.write("var current_rank: i64 = 1;\n");
        try self.writeIndent();
        try self.write("var rows_at_rank: i64 = 0;\n");
        if (spec.order_cols.len > 0) {
            try self.writeIndent();
            try self.fmt("var prev_order_val: i64 = columns.{s}[indices[0]];\n", .{spec.order_cols[0]});
        }
        try self.writeIndent();
        try self.write("\n");
        try self.writeIndent();
        try self.write("for (indices[0..columns.len]) |i| {\n");
        self.indent += 1;

        // Check partition change
        if (spec.partition_cols.len > 0) {
            try self.writeIndent();
            try self.fmt("if (columns.{s}[i] != current_partition) {{\n", .{spec.partition_cols[0]});
            self.indent += 1;
            try self.writeIndent();
            try self.fmt("current_partition = columns.{s}[i];\n", .{spec.partition_cols[0]});
            try self.writeIndent();
            try self.write("current_rank = 1;\n");
            try self.writeIndent();
            try self.write("rows_at_rank = 0;\n");
            if (spec.order_cols.len > 0) {
                try self.writeIndent();
                try self.fmt("prev_order_val = columns.{s}[i];\n", .{spec.order_cols[0]});
            }
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");
        }

        // Check order value change for rank update
        if (spec.order_cols.len > 0) {
            try self.writeIndent();
            try self.fmt("const curr_order_val = columns.{s}[i];\n", .{spec.order_cols[0]});
            try self.writeIndent();
            try self.write("if (curr_order_val != prev_order_val) {\n");
            self.indent += 1;
            if (dense) {
                try self.writeIndent();
                try self.write("current_rank += 1;\n");
            } else {
                try self.writeIndent();
                try self.write("current_rank += rows_at_rank;\n");
            }
            try self.writeIndent();
            try self.write("rows_at_rank = 1;\n");
            try self.writeIndent();
            try self.write("prev_order_val = curr_order_val;\n");
            self.indent -= 1;
            try self.writeIndent();
            try self.write("} else {\n");
            self.indent += 1;
            try self.writeIndent();
            try self.write("rows_at_rank += 1;\n");
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");
        }

        try self.writeIndent();
        try self.write("results[i] = current_rank;\n");
        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n");

        self.indent -= 1;
        try self.write("}\n\n");
    }

    /// Generate LAG or LEAD window function
    fn genWindowLagLead(self: *Self, spec: WindowFuncInfo, is_lag: bool) CodeGenError!void {
        _ = is_lag;

        try self.fmt("fn computeWindow_{s}(columns: *const Columns, results: []i64) void {{\n", .{spec.name});
        self.indent += 1;

        // Create index array
        try self.writeIndent();
        try self.write("var indices: [4096]u32 = undefined;\n");
        try self.writeIndent();
        try self.write("var idx: u32 = 0;\n");
        try self.writeIndent();
        try self.write("while (idx < columns.len) : (idx += 1) indices[idx] = idx;\n\n");

        // Generate sort
        if (spec.partition_cols.len > 0 or spec.order_cols.len > 0) {
            try self.genWindowSort(spec);
        }

        // For now, just fill with default (LAG/LEAD needs more work)
        try self.writeIndent();
        try self.fmt("for (indices[0..columns.len]) |i| results[i] = {d};\n", .{spec.default_val});

        self.indent -= 1;
        try self.write("}\n\n");
    }

    /// Generate sort context for window function
    fn genWindowSort(self: *Self, spec: WindowFuncInfo) CodeGenError!void {
        // Generate sort context struct
        try self.writeIndent();
        try self.write("const SortCtx = struct {\n");
        self.indent += 1;
        try self.writeIndent();
        try self.write("cols: *const Columns,\n");
        try self.writeIndent();
        try self.write("fn lessThan(ctx: @This(), a: u32, b: u32) bool {\n");
        self.indent += 1;

        // Compare partition columns first
        for (spec.partition_cols) |col| {
            try self.writeIndent();
            try self.fmt("if (ctx.cols.{s}[a] != ctx.cols.{s}[b]) return ctx.cols.{s}[a] < ctx.cols.{s}[b];\n", .{ col, col, col, col });
        }

        // Then order columns
        for (spec.order_cols, 0..) |col, i| {
            const desc = if (i < spec.order_desc.len) spec.order_desc[i] else false;
            if (desc) {
                try self.writeIndent();
                try self.fmt("if (ctx.cols.{s}[a] != ctx.cols.{s}[b]) return ctx.cols.{s}[a] > ctx.cols.{s}[b];\n", .{ col, col, col, col });
            } else {
                try self.writeIndent();
                try self.fmt("if (ctx.cols.{s}[a] != ctx.cols.{s}[b]) return ctx.cols.{s}[a] < ctx.cols.{s}[b];\n", .{ col, col, col, col });
            }
        }

        try self.writeIndent();
        try self.write("return false;\n");
        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n");
        self.indent -= 1;
        try self.writeIndent();
        try self.write("};\n\n");

        // Call sort
        try self.writeIndent();
        try self.write("std.mem.sort(u32, indices[0..columns.len], SortCtx{ .cols = columns }, SortCtx.lessThan);\n\n");
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

        // Phase 1: Window function preamble (pre-compute window values)
        if (self.window_specs.items.len > 0) {
            try self.writeIndent();
            try self.write("// Phase 1: Pre-compute window function values\n");
            for (self.window_specs.items) |spec| {
                try self.writeIndent();
                try self.fmt("var window_{s}: [4096]i64 = undefined;\n", .{spec.name});
            }
            for (self.window_specs.items) |spec| {
                try self.writeIndent();
                try self.fmt("computeWindow_{s}(columns, &window_{s});\n", .{ spec.name, spec.name });
            }
            try self.write("\n");
        }

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
            .window => |window| {
                // Window values already computed in preamble
                // Just process the input node
                try self.genPlanNodeBody(window.input);
            },
            .group_by, .sort, .hash_join => {
                // These nodes require interpreted execution
                return CodeGenError.UnsupportedPlanNode;
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

        // Emit window columns
        for (self.window_specs.items) |spec| {
            try self.writeIndent();
            try self.fmt("output.{s}[result_count] = window_{s}[i];\n", .{ spec.name, spec.name });
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
                // Check if this is a window column
                if (self.window_columns.contains(col.name)) {
                    try self.fmt("window_{s}[i]", .{col.name});
                } else {
                    try self.write("columns.");
                    try self.write(col.name);
                    try self.write("[i]");
                }
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

test "FusedCodeGen window tracking" {
    const allocator = std.testing.allocator;
    var codegen = FusedCodeGen.init(allocator);
    defer codegen.deinit();

    // Add a window spec manually
    const partition_cols = [_][]const u8{"dept"};
    const order_cols = [_][]const u8{"salary"};
    const order_desc = [_]bool{true};
    try codegen.window_specs.append(allocator, .{
        .name = "rn",
        .func_type = .row_number,
        .partition_cols = &partition_cols,
        .order_cols = &order_cols,
        .order_desc = &order_desc,
    });
    try codegen.window_columns.put("rn", {});

    // Verify window column is tracked
    try std.testing.expect(codegen.window_columns.contains("rn"));
    try std.testing.expectEqual(@as(usize, 1), codegen.window_specs.items.len);
}

test "FusedCodeGen window column expression" {
    const allocator = std.testing.allocator;
    var codegen = FusedCodeGen.init(allocator);
    defer codegen.deinit();

    // Add a window column
    try codegen.window_columns.put("row_num", {});

    // Test that window columns generate window_{name}[i] instead of columns.{name}[i]
    const expr = ast.Expr{ .column = .{ .name = "row_num", .table_alias = null } };
    try codegen.genExpr(&expr);
    try std.testing.expectEqualStrings("window_row_num[i]", codegen.code.items);
}

test "FusedCodeGen regular column expression" {
    const allocator = std.testing.allocator;
    var codegen = FusedCodeGen.init(allocator);
    defer codegen.deinit();

    // Test that regular columns generate columns.{name}[i]
    const expr = ast.Expr{ .column = .{ .name = "id", .table_alias = null } };
    try codegen.genExpr(&expr);
    try std.testing.expectEqualStrings("columns.id[i]", codegen.code.items);
}
