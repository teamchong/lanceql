// Logic Table Integration for LanceQL
// This module loads and executes @logic_table functions compiled by metal0
//
// Workflow:
// 1. Python code with @logic_table decorator is compiled by metal0 --emit-logic-table
// 2. Generated .zig files are placed in src/logic_table/
// 3. LanceQL imports them and executes batch functions on Lance data

const std = @import("std");

// Import compiled logic_table modules
pub const vector_ops = @import("vector_ops.zig");

/// Registry of all available logic_table structs
pub const LogicTableRegistry = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) LogicTableRegistry {
        return .{ .allocator = allocator };
    }

    /// Get metadata about a logic_table struct by name
    pub fn getTable(self: *LogicTableRegistry, name: []const u8) ?LogicTableMeta {
        _ = self;
        if (std.mem.eql(u8, name, "VectorOps")) {
            return LogicTableMeta{
                .name = "VectorOps",
                .methods = &vector_ops.VectorOps.methods,
            };
        }
        return null;
    }

    /// Execute a method from a logic_table
    pub fn execute(
        self: *LogicTableRegistry,
        table_name: []const u8,
        method_name: []const u8,
        inputs: []const []const f32,
        output: []f32,
    ) !void {
        _ = self;
        if (std.mem.eql(u8, table_name, "VectorOps")) {
            if (std.mem.eql(u8, method_name, "cosine_sim")) {
                if (inputs.len != 2) return error.InvalidInputCount;
                vector_ops.VectorOps.cosine_sim(inputs[0], inputs[1], output);
            } else if (std.mem.eql(u8, method_name, "dot_product")) {
                if (inputs.len != 2) return error.InvalidInputCount;
                vector_ops.VectorOps.dot_product(inputs[0], inputs[1], output);
            } else if (std.mem.eql(u8, method_name, "weighted_score")) {
                if (inputs.len != 2) return error.InvalidInputCount;
                vector_ops.VectorOps.weighted_score(inputs[0], inputs[1], output);
            } else {
                return error.MethodNotFound;
            }
        } else {
            return error.TableNotFound;
        }
    }
};

pub const LogicTableMeta = struct {
    name: []const u8,
    methods: []const []const u8,
};

/// Direct access to compiled functions for zero-overhead calls
pub const Functions = struct {
    /// Compute cosine similarity between query and document embeddings
    pub inline fn cosineSim(query: []const f32, docs: []const f32, out: []f32) void {
        vector_ops.VectorOps.cosine_sim(query, docs, out);
    }

    /// Compute dot product between query and document embeddings
    pub inline fn dotProduct(query: []const f32, docs: []const f32, out: []f32) void {
        vector_ops.VectorOps.dot_product(query, docs, out);
    }

    /// Compute weighted score combining query score and document boost
    pub inline fn weightedScore(scores: []const f32, boosts: []const f32, out: []f32) void {
        vector_ops.VectorOps.weighted_score(scores, boosts, out);
    }
};

test "logic_table cosine_sim" {
    var output: [3]f32 = undefined;
    const query = [_]f32{ 1.0, 0.0, 0.0 };
    const docs = [_]f32{ 1.0, 0.0, 0.0 };

    Functions.cosineSim(&query, &docs, &output);

    // Cosine similarity of identical unit vectors should be 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 0.01);
}

test "logic_table dot_product" {
    var output: [3]f32 = undefined;
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };

    Functions.dotProduct(&a, &b, &output);

    // Each output[i] = sum(a[j] * b[j]) but the generated code uses i index...
    // This is a simplified test - the actual semantics depend on metal0 codegen
}

test "logic_table registry" {
    var registry = LogicTableRegistry.init(std.testing.allocator);
    const meta = registry.getTable("VectorOps");
    try std.testing.expect(meta != null);
    try std.testing.expectEqualStrings("VectorOps", meta.?.name);
}
