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
const simd_ops = @import("simd_ops.zig");

const PlanNode = plan_nodes.PlanNode;
const QueryPlan = plan_nodes.QueryPlan;
const ColumnRef = plan_nodes.ColumnRef;
const ColumnType = plan_nodes.ColumnType;
const ComputeExpr = plan_nodes.ComputeExpr;
const WindowFuncType = plan_nodes.WindowFuncType;
const WindowFuncSpec = plan_nodes.WindowFuncSpec;
const OrderBySpec = plan_nodes.OrderBySpec;
const AggregateType = plan_nodes.AggregateType;
const AggregateSpec = plan_nodes.AggregateSpec;
const SimdOp = simd_ops.SimdOp;

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

/// Join type for Hash JOIN
pub const JoinType = enum {
    inner,
    left,
    right,
    full,
};

/// Join info for code generation
pub const JoinInfo = struct {
    /// Join type (INNER, LEFT, etc.)
    join_type: JoinType,
    /// Left table join key column
    left_key: ColumnRef,
    /// Right table join key column
    right_key: ColumnRef,
    /// Left side columns to output
    left_output_cols: []const ColumnRef,
    /// Right side columns to output
    right_output_cols: []const ColumnRef,
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

    /// Sort specifications (ORDER BY columns)
    sort_specs: std.ArrayList(OrderBySpec),

    /// GROUP BY key columns
    group_keys: std.ArrayList(ColumnRef),

    /// Aggregate specifications
    aggregate_specs: std.ArrayList(AggregateSpec),

    /// Flag indicating this is a GROUP BY query
    has_group_by: bool,

    /// SIMD vector functions to generate (name -> op type)
    simd_functions: std.StringHashMap(SimdOp),

    /// Hash JOIN specification (if present)
    join_info: ?JoinInfo,

    /// Right-side columns for JOIN (separate from main input columns)
    right_columns: std.StringHashMap(ColumnType),

    /// Right-side column order (for layout)
    right_column_order: std.ArrayList(ColumnInfo),

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
            .sort_specs = .{},
            .group_keys = .{},
            .aggregate_specs = .{},
            .has_group_by = false,
            .simd_functions = std.StringHashMap(SimdOp).init(allocator),
            .join_info = null,
            .right_columns = std.StringHashMap(ColumnType).init(allocator),
            .right_column_order = .{},
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
        self.sort_specs.deinit(self.allocator);
        self.group_keys.deinit(self.allocator);
        self.simd_functions.deinit();
        self.right_columns.deinit();
        self.right_column_order.deinit(self.allocator);
        self.aggregate_specs.deinit(self.allocator);
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

        // Calculate output column offsets
        if (self.has_group_by) {
            // GROUP BY output: group keys + aggregates
            const output_count = self.group_keys.items.len + self.aggregate_specs.items.len;
            var output_cols = self.allocator.alloc(ColumnInfo, output_count) catch
                return CodeGenError.OutOfMemory;

            offset = 0;
            var out_idx: usize = 0;

            // Group keys - use resolved types
            for (self.group_keys.items) |key| {
                const resolved_type = self.input_columns.get(key.column) orelse key.col_type;
                output_cols[out_idx] = .{
                    .name = key.column,
                    .col_type = resolved_type,
                    .offset = offset,
                };
                offset += ptr_size;
                out_idx += 1;
            }

            // Aggregates - use resolved types
            for (self.aggregate_specs.items) |agg| {
                const col_type: ColumnType = switch (agg.agg_type) {
                    .count, .count_distinct => .i64,
                    .sum => if (agg.input_col) |col| blk: {
                        break :blk self.input_columns.get(col.column) orelse col.col_type;
                    } else .i64,
                    .avg => .f64,
                    .min, .max => if (agg.input_col) |col| blk: {
                        break :blk self.input_columns.get(col.column) orelse col.col_type;
                    } else .i64,
                    else => .i64,
                };
                output_cols[out_idx] = .{
                    .name = agg.name,
                    .col_type = col_type,
                    .offset = offset,
                };
                offset += ptr_size;
                out_idx += 1;
            }

            return ColumnLayout{
                .input_columns = input_cols,
                .output_columns = output_cols,
                .columns_size = columns_size,
                .output_size = offset,
                .allocator = self.allocator,
            };
        }

        // Regular output: input columns + computed columns + window columns
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
                self.has_group_by = true;

                // Collect group keys
                for (group_by.group_keys) |key| {
                    self.group_keys.append(self.allocator, key) catch return CodeGenError.OutOfMemory;
                    try self.addInputColumn(key.column, key.col_type);
                }

                // Collect aggregate specifications
                for (group_by.aggregates) |agg| {
                    self.aggregate_specs.append(self.allocator, agg) catch return CodeGenError.OutOfMemory;
                    // Add input column for aggregate if not COUNT(*)
                    if (agg.input_col) |col| {
                        try self.addInputColumn(col.column, col.col_type);
                    }
                }
            },
            .sort => |sort| {
                try self.analyzePlan(sort.input);
                // Collect sort specifications
                for (sort.order_by) |order_spec| {
                    self.sort_specs.append(self.allocator, order_spec) catch return CodeGenError.OutOfMemory;
                    // Ensure sort columns are in input
                    try self.addInputColumn(order_spec.column.column, order_spec.column.col_type);
                }
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
                // Analyze left side (populates input_columns as "left" columns)
                try self.analyzePlan(join.left);

                // Analyze right side - collect columns separately
                try self.analyzeJoinRight(join.right);

                // Add join keys to appropriate column sets
                try self.addInputColumn(join.left_key.column, join.left_key.col_type);
                try self.addRightColumn(join.right_key.column, join.right_key.col_type);

                // Store join info
                self.join_info = .{
                    .join_type = @enumFromInt(@intFromEnum(join.join_type)),
                    .left_key = join.left_key,
                    .right_key = join.right_key,
                    .left_output_cols = &.{}, // Will be populated from SELECT list
                    .right_output_cols = &.{},
                };
            },
        }
    }

    /// Add a column to input columns (if not already present)
    fn addInputColumn(self: *Self, name: []const u8, col_type: ColumnType) CodeGenError!void {
        // Skip invalid column names (like "*" from SELECT *)
        if (name.len == 0 or std.mem.eql(u8, name, "*")) return;

        if (!self.input_columns.contains(name)) {
            self.input_columns.put(name, col_type) catch return CodeGenError.OutOfMemory;
            self.input_column_order.append(self.allocator, .{
                .name = name,
                .col_type = col_type,
                .offset = 0,
            }) catch return CodeGenError.OutOfMemory;
        }
    }

    /// Add a column to right-side columns (for JOIN)
    fn addRightColumn(self: *Self, name: []const u8, col_type: ColumnType) CodeGenError!void {
        // Skip invalid column names
        if (name.len == 0 or std.mem.eql(u8, name, "*")) return;

        if (!self.right_columns.contains(name)) {
            self.right_columns.put(name, col_type) catch return CodeGenError.OutOfMemory;
            self.right_column_order.append(self.allocator, .{
                .name = name,
                .col_type = col_type,
                .offset = 0,
            }) catch return CodeGenError.OutOfMemory;
        }
    }

    /// Analyze right side of JOIN (collects columns into right_columns)
    fn analyzeJoinRight(self: *Self, node: *const PlanNode) CodeGenError!void {
        switch (node.*) {
            .scan => |scan| {
                for (scan.columns) |col| {
                    try self.addRightColumn(col.column, col.col_type);
                }
            },
            .filter => |filter| {
                try self.analyzeJoinRight(filter.input);
            },
            .project => |project| {
                try self.analyzeJoinRight(project.input);
            },
            else => {},
        }
    }

    /// Analyze expression to collect column references
    fn analyzeExpr(self: *Self, expr: *const ast.Expr) CodeGenError!void {
        switch (expr.*) {
            .column => |col| {
                // Skip invalid column names (like "*" from SELECT *)
                if (col.name.len == 0 or std.mem.eql(u8, col.name, "*")) return;

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
                // Track SIMD vector operations
                if (simd_ops.detectSimdOp(call.name)) |op| {
                    self.simd_functions.put(call.name, op) catch return CodeGenError.OutOfMemory;
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

        // Generate SIMD helper functions if any were detected
        try self.genSimdFunctions();
    }

    /// Generate SIMD helper functions for vector operations
    fn genSimdFunctions(self: *Self) CodeGenError!void {
        var iter = self.simd_functions.iterator();
        while (iter.next()) |entry| {
            simd_ops.genSimdFunction(&self.code, self.allocator, entry.value_ptr.*, entry.key_ptr.*) catch
                return CodeGenError.OutOfMemory;
        }
    }

    /// Generate Columns struct definition
    fn genColumnsStruct(self: *Self) CodeGenError!void {
        if (self.join_info != null) {
            // For JOINs, generate LeftColumns and RightColumns
            try self.genJoinColumnsStructs();
            return;
        }

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

    /// Generate LeftColumns and RightColumns for JOIN queries
    fn genJoinColumnsStructs(self: *Self) CodeGenError!void {
        // LeftColumns (from input_columns)
        try self.write("pub const LeftColumns = struct {\n");
        self.indent += 1;

        var left_iter = self.input_columns.iterator();
        while (left_iter.next()) |entry| {
            try self.writeIndent();
            try self.write(entry.key_ptr.*);
            try self.write(": [*]const ");
            try self.write(entry.value_ptr.toZigType());
            try self.write(",\n");
        }
        try self.writeIndent();
        try self.write("len: usize,\n");

        self.indent -= 1;
        try self.write("};\n\n");

        // RightColumns (from right_columns)
        try self.write("pub const RightColumns = struct {\n");
        self.indent += 1;

        var right_iter = self.right_columns.iterator();
        while (right_iter.next()) |entry| {
            try self.writeIndent();
            try self.write(entry.key_ptr.*);
            try self.write(": [*]const ");
            try self.write(entry.value_ptr.toZigType());
            try self.write(",\n");
        }
        try self.writeIndent();
        try self.write("len: usize,\n");

        self.indent -= 1;
        try self.write("};\n\n");
    }

    /// Generate OutputBuffers struct definition
    fn genOutputStruct(self: *Self) CodeGenError!void {
        try self.write("pub const OutputBuffers = struct {\n");
        self.indent += 1;

        if (self.join_info != null) {
            // JOIN output: columns from both left and right sides
            // Left columns with "left_" prefix
            var left_iter = self.input_columns.iterator();
            while (left_iter.next()) |entry| {
                try self.writeIndent();
                try self.write(entry.key_ptr.*);
                try self.write(": [*]");
                try self.write(entry.value_ptr.toZigType());
                try self.write(",\n");
            }

            // Right columns with "right_" prefix (to avoid name collisions)
            var right_iter = self.right_columns.iterator();
            while (right_iter.next()) |entry| {
                try self.writeIndent();
                try self.fmt("right_{s}: [*]{s},\n", .{ entry.key_ptr.*, entry.value_ptr.toZigType() });
            }

            self.indent -= 1;
            try self.write("};\n\n");
            return;
        }

        if (self.has_group_by) {
            // GROUP BY output: group keys + aggregates
            for (self.group_keys.items) |key| {
                // Use resolved type from input_columns if available
                const resolved_type = self.input_columns.get(key.column) orelse key.col_type;
                try self.writeIndent();
                try self.write(key.column);
                try self.write(": [*]");
                try self.write(resolved_type.toZigType());
                try self.write(",\n");
            }

            // Aggregate output columns
            for (self.aggregate_specs.items) |agg| {
                const zig_type: []const u8 = switch (agg.agg_type) {
                    .count, .count_distinct => "i64",
                    .sum => if (agg.input_col) |col| blk: {
                        const resolved = self.input_columns.get(col.column) orelse col.col_type;
                        break :blk resolved.toZigType();
                    } else "i64",
                    .avg => "f64",
                    .min, .max => if (agg.input_col) |col| blk: {
                        const resolved = self.input_columns.get(col.column) orelse col.col_type;
                        break :blk resolved.toZigType();
                    } else "i64",
                    else => "i64",
                };
                try self.writeIndent();
                try self.write(agg.name);
                try self.write(": [*]");
                try self.write(zig_type);
                try self.write(",\n");
            }
        } else {
            // Regular output: input columns + computed + window
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
                .first_value => try self.genWindowFirstLastValue(spec, true),
                .last_value => try self.genWindowFirstLastValue(spec, false),
                .ntile => try self.genWindowNtile(spec),
                else => {}, // nth_value, percent_rank, cume_dist not yet supported
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

    /// Generate FIRST_VALUE or LAST_VALUE window function
    fn genWindowFirstLastValue(self: *Self, spec: WindowFuncInfo, is_first: bool) CodeGenError!void {
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

        // Compute first/last value per partition
        try self.writeIndent();
        if (spec.partition_cols.len > 0) {
            try self.fmt("var current_partition: i64 = columns.{s}[indices[0]];\n", .{spec.partition_cols[0]});
            try self.writeIndent();
            if (is_first) {
                // For FIRST_VALUE, capture first value in partition
                if (spec.order_cols.len > 0) {
                    try self.fmt("var partition_value: i64 = columns.{s}[indices[0]];\n", .{spec.order_cols[0]});
                } else {
                    try self.write("var partition_value: i64 = 0;\n");
                }
            } else {
                // For LAST_VALUE, will update as we go
                try self.write("var partition_value: i64 = 0;\n");
            }
            try self.writeIndent();
            try self.write("var partition_start: usize = 0;\n\n");

            try self.writeIndent();
            try self.write("for (indices[0..columns.len], 0..) |i, pos| {\n");
            self.indent += 1;
            try self.writeIndent();
            try self.fmt("if (columns.{s}[i] != current_partition) {{\n", .{spec.partition_cols[0]});
            self.indent += 1;

            // End of partition - fill results for previous partition
            try self.writeIndent();
            try self.write("for (indices[partition_start..pos]) |pi| results[pi] = partition_value;\n");

            // Start new partition
            try self.writeIndent();
            try self.fmt("current_partition = columns.{s}[i];\n", .{spec.partition_cols[0]});
            try self.writeIndent();
            try self.write("partition_start = pos;\n");
            if (is_first and spec.order_cols.len > 0) {
                try self.writeIndent();
                try self.fmt("partition_value = columns.{s}[i];\n", .{spec.order_cols[0]});
            }
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");

            if (!is_first and spec.order_cols.len > 0) {
                // For LAST_VALUE, keep updating
                try self.writeIndent();
                try self.fmt("partition_value = columns.{s}[i];\n", .{spec.order_cols[0]});
            }
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");

            // Handle last partition
            try self.writeIndent();
            try self.write("for (indices[partition_start..columns.len]) |pi| results[pi] = partition_value;\n");
        } else {
            // No partitions - single value for all rows
            if (spec.order_cols.len > 0) {
                if (is_first) {
                    try self.fmt("const value: i64 = columns.{s}[indices[0]];\n", .{spec.order_cols[0]});
                } else {
                    try self.fmt("const value: i64 = columns.{s}[indices[columns.len - 1]];\n", .{spec.order_cols[0]});
                }
            } else {
                try self.write("const value: i64 = 0;\n");
            }
            try self.writeIndent();
            try self.write("for (indices[0..columns.len]) |i| results[i] = value;\n");
        }

        self.indent -= 1;
        try self.write("}\n\n");
    }

    /// Generate NTILE window function (divide rows into N buckets)
    fn genWindowNtile(self: *Self, spec: WindowFuncInfo) CodeGenError!void {
        const num_buckets: i64 = spec.offset; // Reuse offset field for bucket count

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

        // Compute NTILE per partition
        try self.writeIndent();
        if (spec.partition_cols.len > 0) {
            try self.fmt("var current_partition: i64 = columns.{s}[indices[0]];\n", .{spec.partition_cols[0]});
            try self.writeIndent();
            try self.write("var partition_start: usize = 0;\n\n");

            try self.writeIndent();
            try self.write("for (indices[0..columns.len], 0..) |i, pos| {\n");
            self.indent += 1;
            try self.writeIndent();
            try self.fmt("if (columns.{s}[i] != current_partition) {{\n", .{spec.partition_cols[0]});
            self.indent += 1;

            // End of partition - compute NTILE for previous partition
            try self.writeIndent();
            try self.write("const part_size = pos - partition_start;\n");
            try self.writeIndent();
            try self.fmt("const bucket_size = (part_size + {d} - 1) / {d};\n", .{ num_buckets, num_buckets });
            try self.writeIndent();
            try self.write("for (indices[partition_start..pos], 0..) |pi, offset| {\n");
            self.indent += 1;
            try self.writeIndent();
            try self.write("results[pi] = @intCast(offset / bucket_size + 1);\n");
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");

            // Start new partition
            try self.writeIndent();
            try self.fmt("current_partition = columns.{s}[i];\n", .{spec.partition_cols[0]});
            try self.writeIndent();
            try self.write("partition_start = pos;\n");
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");

            // Handle last partition
            try self.writeIndent();
            try self.write("const part_size = columns.len - partition_start;\n");
            try self.writeIndent();
            try self.fmt("const bucket_size = (part_size + {d} - 1) / {d};\n", .{ num_buckets, num_buckets });
            try self.writeIndent();
            try self.write("for (indices[partition_start..columns.len], 0..) |pi, offset| {\n");
            self.indent += 1;
            try self.writeIndent();
            try self.write("results[pi] = @intCast(offset / bucket_size + 1);\n");
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");
        } else {
            // No partitions - compute NTILE for all rows
            try self.fmt("const bucket_size = (columns.len + {d} - 1) / {d};\n", .{ num_buckets, num_buckets });
            try self.writeIndent();
            try self.write("for (indices[0..columns.len], 0..) |i, offset| {\n");
            self.indent += 1;
            try self.writeIndent();
            try self.write("results[i] = @intCast(offset / bucket_size + 1);\n");
            self.indent -= 1;
            try self.writeIndent();
            try self.write("}\n");
        }

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
        // Use different code generation for GROUP BY queries
        if (self.has_group_by) {
            return self.genGroupByFunction(root);
        }

        // Use different code generation for JOIN queries
        if (self.join_info != null) {
            return self.genHashJoinFunction(root);
        }

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

        // Phase 2: Sort indices if ORDER BY is present
        const has_sort = self.sort_specs.items.len > 0;
        if (has_sort) {
            try self.writeIndent();
            try self.write("// Phase 2: Sort indices for ORDER BY\n");
            try self.writeIndent();
            try self.write("var sorted_indices: [4096]u32 = undefined;\n");
            try self.writeIndent();
            try self.write("var init_idx: u32 = 0;\n");
            try self.writeIndent();
            try self.write("while (init_idx < columns.len) : (init_idx += 1) sorted_indices[init_idx] = init_idx;\n");
            try self.genSortContext();
            try self.write("\n");
        }

        // Local variables
        try self.writeIndent();
        try self.write("var result_count: usize = 0;\n");

        // Main loop - iterate over sorted indices if sorting, otherwise direct indices
        if (has_sort) {
            try self.writeIndent();
            try self.write("for (sorted_indices[0..columns.len]) |i| {\n");
        } else {
            try self.writeIndent();
            try self.write("var i: usize = 0;\n\n");
            try self.writeIndent();
            try self.write("while (i < columns.len) : (i += 1) {\n");
        }
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

    /// Generate sort context and sort call for ORDER BY
    fn genSortContext(self: *Self) CodeGenError!void {
        try self.writeIndent();
        try self.write("const OrderSortCtx = struct {\n");
        self.indent += 1;
        try self.writeIndent();
        try self.write("cols: *const Columns,\n");
        try self.writeIndent();
        try self.write("fn lessThan(ctx: @This(), a: u32, b: u32) bool {\n");
        self.indent += 1;

        // Generate comparison for each ORDER BY column
        for (self.sort_specs.items) |spec| {
            const col = spec.column.column;
            const desc = spec.direction == .desc;
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
        try self.write("};\n");

        // Call sort
        try self.writeIndent();
        try self.write("std.mem.sort(u32, sorted_indices[0..columns.len], OrderSortCtx{ .cols = columns }, OrderSortCtx.lessThan);\n");
    }

    /// Generate GROUP BY function with hash grouping
    fn genGroupByFunction(self: *Self, root: *const PlanNode) CodeGenError!void {
        _ = root; // We use collected group_keys and aggregate_specs

        // Function signature
        try self.write(
            \\pub export fn fused_query(
            \\    columns: *const Columns,
            \\    output: *OutputBuffers,
            \\) callconv(.c) usize {
            \\
        );
        self.indent += 1;

        // Phase 1: Group key storage and aggregate accumulators
        try self.writeIndent();
        try self.write("// Phase 1: Group keys and aggregate accumulators\n");
        try self.writeIndent();
        try self.write("const max_groups: usize = 4096;\n");
        try self.writeIndent();
        try self.write("var num_groups: usize = 0;\n\n");

        // Group key arrays (one per group key column)
        for (self.group_keys.items, 0..) |key, idx| {
            // Use resolved type from input_columns if available
            const resolved_type = self.input_columns.get(key.column) orelse key.col_type;
            try self.writeIndent();
            try self.fmt("var group_key_{d}: [max_groups]{s} = undefined; // {s}\n", .{
                idx,
                resolved_type.toZigType(),
                key.column,
            });
        }
        try self.write("\n");

        // Aggregate accumulator arrays
        for (self.aggregate_specs.items) |agg| {
            const zig_type: []const u8 = switch (agg.agg_type) {
                .count, .count_distinct => "i64",
                .sum => if (agg.input_col) |col| blk: {
                    // Use resolved type from input_columns if available
                    const resolved = self.input_columns.get(col.column) orelse col.col_type;
                    break :blk resolved.toZigType();
                } else "i64",
                .avg => "f64",
                .min, .max => if (agg.input_col) |col| blk: {
                    const resolved = self.input_columns.get(col.column) orelse col.col_type;
                    break :blk resolved.toZigType();
                } else "i64",
                else => "i64", // Default for unsupported
            };
            try self.writeIndent();
            try self.fmt("var agg_{s}: [max_groups]{s} = ", .{ agg.name, zig_type });
            // Initialize with appropriate default
            switch (agg.agg_type) {
                .count, .count_distinct, .sum => try self.write("[_]"),
                .avg => try self.write("[_]"),
                .min => try self.write("[_]"),
                .max => try self.write("[_]"),
                else => try self.write("[_]"),
            }
            try self.fmt("{s}{{0}} ** max_groups;\n", .{zig_type});

            // For AVG, we need a count accumulator too
            if (agg.agg_type == .avg) {
                try self.writeIndent();
                try self.fmt("var agg_{s}_count: [max_groups]i64 = [_]i64{{0}} ** max_groups;\n", .{agg.name});
            }
        }
        try self.write("\n");

        // Phase 2: Grouping loop
        try self.writeIndent();
        try self.write("// Phase 2: Build groups and accumulate aggregates\n");
        try self.writeIndent();
        try self.write("var i: usize = 0;\n");
        try self.writeIndent();
        try self.write("while (i < columns.len) : (i += 1) {\n");
        self.indent += 1;

        // Extract current row's group key values
        for (self.group_keys.items, 0..) |key, idx| {
            try self.writeIndent();
            try self.fmt("const curr_key_{d} = columns.{s}[i];\n", .{ idx, key.column });
        }
        try self.write("\n");

        // Find or create group
        try self.writeIndent();
        try self.write("// Find existing group or create new one\n");
        try self.writeIndent();
        try self.write("var group_idx: usize = 0;\n");
        try self.writeIndent();
        try self.write("var found: bool = false;\n");
        try self.writeIndent();
        try self.write("while (group_idx < num_groups) : (group_idx += 1) {\n");
        self.indent += 1;

        // Compare all group keys
        try self.writeIndent();
        try self.write("if (");
        for (self.group_keys.items, 0..) |_, idx| {
            if (idx > 0) try self.write(" and ");
            try self.fmt("group_key_{d}[group_idx] == curr_key_{d}", .{ idx, idx });
        }
        try self.write(") {\n");
        self.indent += 1;
        try self.writeIndent();
        try self.write("found = true;\n");
        try self.writeIndent();
        try self.write("break;\n");
        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n");
        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n\n");

        // Create new group if not found
        try self.writeIndent();
        try self.write("if (!found) {\n");
        self.indent += 1;
        try self.writeIndent();
        try self.write("group_idx = num_groups;\n");
        for (self.group_keys.items, 0..) |_, idx| {
            try self.writeIndent();
            try self.fmt("group_key_{d}[group_idx] = curr_key_{d};\n", .{ idx, idx });
        }
        try self.writeIndent();
        try self.write("num_groups += 1;\n");
        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n\n");

        // Accumulate aggregates
        try self.writeIndent();
        try self.write("// Accumulate aggregates\n");
        for (self.aggregate_specs.items) |agg| {
            try self.writeIndent();
            switch (agg.agg_type) {
                .count => {
                    if (agg.input_col) |_| {
                        // COUNT(col) - only count non-null values (for now, count all)
                        try self.fmt("agg_{s}[group_idx] += 1;\n", .{agg.name});
                    } else {
                        // COUNT(*)
                        try self.fmt("agg_{s}[group_idx] += 1;\n", .{agg.name});
                    }
                },
                .sum => {
                    if (agg.input_col) |col| {
                        try self.fmt("agg_{s}[group_idx] += columns.{s}[i];\n", .{ agg.name, col.column });
                    }
                },
                .avg => {
                    if (agg.input_col) |col| {
                        try self.fmt("agg_{s}[group_idx] += @as(f64, @floatFromInt(columns.{s}[i]));\n", .{ agg.name, col.column });
                        try self.writeIndent();
                        try self.fmt("agg_{s}_count[group_idx] += 1;\n", .{agg.name});
                    }
                },
                .min => {
                    if (agg.input_col) |col| {
                        try self.fmt("if (!found or columns.{s}[i] < agg_{s}[group_idx]) agg_{s}[group_idx] = columns.{s}[i];\n", .{ col.column, agg.name, agg.name, col.column });
                    }
                },
                .max => {
                    if (agg.input_col) |col| {
                        try self.fmt("if (!found or columns.{s}[i] > agg_{s}[group_idx]) agg_{s}[group_idx] = columns.{s}[i];\n", .{ col.column, agg.name, agg.name, col.column });
                    }
                },
                else => {
                    // Unsupported aggregate - skip
                    try self.write("// Unsupported aggregate\n");
                },
            }
        }

        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n\n");

        // Phase 3: Output results
        try self.writeIndent();
        try self.write("// Phase 3: Output group results\n");
        try self.writeIndent();
        try self.write("var result_idx: usize = 0;\n");
        try self.writeIndent();
        try self.write("while (result_idx < num_groups) : (result_idx += 1) {\n");
        self.indent += 1;

        // Output group keys
        for (self.group_keys.items, 0..) |key, idx| {
            try self.writeIndent();
            try self.fmt("output.{s}[result_idx] = group_key_{d}[result_idx];\n", .{ key.column, idx });
        }

        // Output aggregates
        for (self.aggregate_specs.items) |agg| {
            try self.writeIndent();
            if (agg.agg_type == .avg) {
                // AVG = sum / count
                try self.fmt("output.{s}[result_idx] = agg_{s}[result_idx] / @as(f64, @floatFromInt(agg_{s}_count[result_idx]));\n", .{ agg.name, agg.name, agg.name });
            } else {
                try self.fmt("output.{s}[result_idx] = agg_{s}[result_idx];\n", .{ agg.name, agg.name });
            }
        }

        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n\n");

        // Return
        try self.writeIndent();
        try self.write("return num_groups;\n");

        self.indent -= 1;
        try self.write("}\n");
    }

    /// Generate Hash JOIN function with build/probe phases
    fn genHashJoinFunction(self: *Self, root: *const PlanNode) CodeGenError!void {
        _ = root; // We use collected join_info

        const join = self.join_info orelse return CodeGenError.InvalidPlan;

        // Function signature with two input structs
        try self.write(
            \\pub export fn fused_query(
            \\    left_columns: *const LeftColumns,
            \\    right_columns: *const RightColumns,
            \\    output: *OutputBuffers,
            \\) callconv(.c) usize {
            \\
        );
        self.indent += 1;

        // Phase 1: Build hash table from right input
        try self.writeIndent();
        try self.write("// Phase 1: Build hash table from right input\n");
        try self.writeIndent();
        try self.write("const max_rows: usize = 4096;\n");

        // Get right key type
        const right_key_type = self.right_columns.get(join.right_key.column) orelse join.right_key.col_type;
        try self.writeIndent();
        try self.fmt("var hash_keys: [max_rows]{s} = undefined;\n", .{right_key_type.toZigType()});
        try self.writeIndent();
        try self.write("var hash_indices: [max_rows]u32 = undefined;\n");
        try self.writeIndent();
        try self.write("var hash_count: usize = 0;\n\n");

        try self.writeIndent();
        try self.write("var ri: usize = 0;\n");
        try self.writeIndent();
        try self.write("while (ri < right_columns.len) : (ri += 1) {\n");
        self.indent += 1;
        try self.writeIndent();
        try self.fmt("hash_keys[hash_count] = right_columns.{s}[ri];\n", .{join.right_key.column});
        try self.writeIndent();
        try self.write("hash_indices[hash_count] = @intCast(ri);\n");
        try self.writeIndent();
        try self.write("hash_count += 1;\n");
        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n\n");

        // Phase 2: Probe with left input
        try self.writeIndent();
        try self.write("// Phase 2: Probe with left input\n");
        try self.writeIndent();
        try self.write("var result_count: usize = 0;\n");
        try self.writeIndent();
        try self.write("var li: usize = 0;\n");
        try self.writeIndent();
        try self.write("while (li < left_columns.len) : (li += 1) {\n");
        self.indent += 1;

        try self.writeIndent();
        try self.fmt("const left_key = left_columns.{s}[li];\n\n", .{join.left_key.column});

        // Linear probe (simple for now - can optimize with hash later)
        try self.writeIndent();
        try self.write("// Linear probe for matching keys\n");
        try self.writeIndent();
        try self.write("var hi: usize = 0;\n");
        try self.writeIndent();
        try self.write("while (hi < hash_count) : (hi += 1) {\n");
        self.indent += 1;

        try self.writeIndent();
        try self.write("if (hash_keys[hi] == left_key) {\n");
        self.indent += 1;
        try self.writeIndent();
        try self.write("const ri_match = hash_indices[hi];\n\n");

        // Emit left columns
        try self.writeIndent();
        try self.write("// Emit joined row\n");
        var left_iter = self.input_columns.keyIterator();
        while (left_iter.next()) |key| {
            try self.writeIndent();
            try self.fmt("output.{s}[result_count] = left_columns.{s}[li];\n", .{ key.*, key.* });
        }

        // Emit right columns (with right_ prefix)
        var right_iter = self.right_columns.keyIterator();
        while (right_iter.next()) |key| {
            try self.writeIndent();
            try self.fmt("output.right_{s}[result_count] = right_columns.{s}[ri_match];\n", .{ key.*, key.* });
        }

        try self.writeIndent();
        try self.write("result_count += 1;\n");
        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n");
        self.indent -= 1;
        try self.writeIndent();
        try self.write("}\n");

        // For LEFT JOIN, emit left rows with no match
        if (join.join_type == .left or join.join_type == .full) {
            try self.writeIndent();
            try self.write("// LEFT JOIN: emit unmatched left rows (not implemented yet)\n");
        }

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
            .sort => |sort| {
                // Sort is handled in preamble (sorted_indices)
                // Just process the input node
                try self.genPlanNodeBody(sort.input);
            },
            .group_by, .hash_join => {
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
                // Skip invalid column names (like "*" from SELECT *)
                if (col.name.len == 0 or std.mem.eql(u8, col.name, "*")) {
                    try self.write("0"); // Placeholder for invalid column
                    return;
                }

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
                // Check if this is a SIMD vector operation
                if (self.simd_functions.contains(call.name)) {
                    // Generate SIMD function call: simd_NAME(a, b, dim)
                    try self.fmt("simd_{s}(", .{call.name});
                    for (call.args, 0..) |*arg, idx| {
                        if (idx > 0) try self.write(", ");
                        try self.genExpr(arg);
                    }
                    try self.write(")");
                } else {
                    // Generate regular function call
                    try self.write(call.name);
                    try self.write("(");
                    for (call.args, 0..) |*arg, idx| {
                        if (idx > 0) try self.write(", ");
                        try self.genExpr(arg);
                    }
                    try self.write(")");
                }
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
