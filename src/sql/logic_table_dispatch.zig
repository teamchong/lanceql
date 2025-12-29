//! LogicTable Dispatch - Links SQL executor to compiled @logic_table methods
//!
//! This module provides the bridge between SQL execution and compiled Python
//! @logic_table methods. It handles:
//! - Registration of compiled extern functions
//! - Dynamic dispatch based on class/method names
//! - Method call execution with proper argument passing
//!
//! Example:
//! ```sql
//! SELECT t.risk_score(), orders.amount
//! FROM logic_table('fraud_detector.py') AS t
//! WHERE t.risk_score() > 0.7
//! ```

const std = @import("std");
const ast = @import("ast");
const logic_table = @import("lanceql.logic_table");

const LogicTableContext = logic_table.LogicTableContext;
const LogicTableExecutor = logic_table.LogicTableExecutor;

/// Error type for logic_table dispatch
pub const DispatchError = error{
    ClassNotFound,
    MethodNotFound,
    ArgumentCountMismatch,
    InvalidArgumentType,
    ExecutionFailed,
};

/// Method function pointer types for C ABI
pub const MethodFnF64_2Args = *const fn ([*]const f64, [*]const f64, usize) callconv(.c) f64;
pub const MethodFnF64_1Arg = *const fn ([*]const f64, usize) callconv(.c) f64;
pub const MethodFnF64_NoArg = *const fn () callconv(.c) f64;

/// Registered method information
pub const RegisteredMethod = struct {
    class_name: []const u8,
    method_name: []const u8,
    fn_ptr: *const anyopaque,
    arg_count: u8,
    return_type: ReturnType,

    pub const ReturnType = enum {
        f64,
        i64,
        bool_,
    };
};

/// LogicTable Dispatch Registry
/// Manages registered @logic_table classes and their methods
pub const Dispatcher = struct {
    allocator: std.mem.Allocator,
    /// Registered methods: "ClassName.method_name" -> RegisteredMethod
    methods: std.StringHashMap(RegisteredMethod),
    /// Logic table executors by alias
    executors: std.StringHashMap(*LogicTableExecutor),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .methods = std.StringHashMap(RegisteredMethod).init(allocator),
            .executors = std.StringHashMap(*LogicTableExecutor).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        // Free method keys
        var iter = self.methods.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.methods.deinit();

        // Free executor keys
        var exec_iter = self.executors.iterator();
        while (exec_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.executors.deinit();
    }

    /// Register a compiled method function pointer
    pub fn registerMethod(
        self: *Self,
        class_name: []const u8,
        method_name: []const u8,
        fn_ptr: *const anyopaque,
        arg_count: u8,
    ) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ class_name, method_name });
        errdefer self.allocator.free(key);

        try self.methods.put(key, .{
            .class_name = class_name,
            .method_name = method_name,
            .fn_ptr = fn_ptr,
            .arg_count = arg_count,
            .return_type = .f64,
        });
    }

    /// Resolve a logic_table() table function
    /// Creates and registers a LogicTableExecutor for the given Python file
    pub fn resolveLogicTable(
        self: *Self,
        python_file: []const u8,
        alias: []const u8,
    ) !*LogicTableExecutor {
        // Check if already resolved
        if (self.executors.get(alias)) |exec| {
            return exec;
        }

        // Create new executor
        const exec = try self.allocator.create(LogicTableExecutor);
        exec.* = try LogicTableExecutor.init(self.allocator, python_file);

        // Store with alias
        const alias_copy = try self.allocator.dupe(u8, alias);
        errdefer self.allocator.free(alias_copy);
        try self.executors.put(alias_copy, exec);

        return exec;
    }

    /// Get executor by alias
    pub fn getExecutor(self: *Self, alias: []const u8) ?*LogicTableExecutor {
        return self.executors.get(alias);
    }

    /// Call a registered method
    /// Returns the method result as f64
    pub fn callMethod(
        self: *Self,
        class_name: []const u8,
        method_name: []const u8,
        args: anytype,
    ) DispatchError!f64 {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ class_name, method_name }) catch
            return DispatchError.MethodNotFound;

        const method = self.methods.get(key) orelse return DispatchError.MethodNotFound;

        // Dispatch based on argument count
        switch (method.arg_count) {
            2 => {
                const fn_ptr: MethodFnF64_2Args = @ptrCast(@alignCast(method.fn_ptr));
                const a = args[0];
                const b = args[1];
                const len = args[2];
                return fn_ptr(a, b, len);
            },
            1 => {
                const fn_ptr: MethodFnF64_1Arg = @ptrCast(@alignCast(method.fn_ptr));
                const a = args[0];
                const len = args[1];
                return fn_ptr(a, len);
            },
            0 => {
                const fn_ptr: MethodFnF64_NoArg = @ptrCast(@alignCast(method.fn_ptr));
                return fn_ptr();
            },
            else => return DispatchError.ArgumentCountMismatch,
        }
    }
};

/// Table source for SQL executor
/// Represents either a Lance file or a @logic_table instance
pub const TableSource = union(enum) {
    /// Regular Lance file
    lance: *anyopaque, // *Table pointer

    /// @logic_table virtual table
    logic_table: struct {
        /// Path to Python file
        python_file: []const u8,
        /// Executor instance
        executor: *LogicTableExecutor,
        /// Class name extracted from Python
        class_name: []const u8,
    },
};

// =============================================================================
// Extern declarations for compiled @logic_table methods
// These are populated when lib/logic_table.a is linked
// =============================================================================

// VectorOps class (from benchmarks/vector_ops.py)
pub extern fn VectorOps_dot_product(a: [*]const f64, b: [*]const f64, len: usize) callconv(.c) f64;
pub extern fn VectorOps_sum_squares(a: [*]const f64, len: usize) callconv(.c) f64;

/// Create a dispatcher with VectorOps methods pre-registered
pub fn createVectorOpsDispatcher(allocator: std.mem.Allocator) !Dispatcher {
    var dispatcher = Dispatcher.init(allocator);
    errdefer dispatcher.deinit();

    try dispatcher.registerMethod("VectorOps", "dot_product", @ptrCast(&VectorOps_dot_product), 2);
    try dispatcher.registerMethod("VectorOps", "sum_squares", @ptrCast(&VectorOps_sum_squares), 1);

    return dispatcher;
}

// =============================================================================
// Tests
// =============================================================================

test "Dispatcher basic" {
    const allocator = std.testing.allocator;

    var dispatcher = Dispatcher.init(allocator);
    defer dispatcher.deinit();

    // Create a test function
    const TestFn = struct {
        fn testAdd(a: [*]const f64, b: [*]const f64, len: usize) callconv(.c) f64 {
            var sum: f64 = 0;
            for (0..len) |i| {
                sum += a[i] + b[i];
            }
            return sum;
        }
    };

    try dispatcher.registerMethod("TestClass", "add", @ptrCast(&TestFn.testAdd), 2);

    // Call the method
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    const b = [_]f64{ 4.0, 5.0, 6.0 };
    const result = try dispatcher.callMethod("TestClass", "add", .{ @as([*]const f64, &a), @as([*]const f64, &b), @as(usize, 3) });

    try std.testing.expectEqual(@as(f64, 21.0), result);
}

test "Dispatcher method not found" {
    const allocator = std.testing.allocator;

    var dispatcher = Dispatcher.init(allocator);
    defer dispatcher.deinit();

    const result = dispatcher.callMethod("Unknown", "method", .{});
    try std.testing.expectError(DispatchError.MethodNotFound, result);
}
