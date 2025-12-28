const std = @import("std");
const runtime = @import("runtime");
const hashmap_helper = runtime.hashmap_helper;
const allocator_helper = runtime.allocator_helper;
const c_interop = @import("c_interop");

const logic_table: ?*anyopaque = null;

const __name__ = "__main__";
const __file__: []const u8 = "/Users/steven_chong/Downloads/repos/lanceql/benchmarks/ml_workflow.py";

// metal0 metadata for runtime eval subprocess
pub const __metal0_source_dir: []const u8 = "/Users/steven_chong/Downloads/repos/lanceql/benchmarks";

pub const FeatureEngineering = struct {
    pub const __logic_table__ = true;

pub fn normalize_minmax(_: std.mem.Allocator, _: runtime.PyValue, data: i64) !i64 {
        _ = &data;
        _ = "Min-max normalization to [0, 1] range.";
        const min_val: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "min", .{data.values}).?)));
        const max_val: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "max", .{data.values}).?)));
        _ = &min_val;
        _ = &max_val;
        return try runtime.divideFloat((runtime.PyValue.from(data.values)).sub(runtime.PyValue.from(min_val)), ((runtime.PyValue.from(max_val)).sub(runtime.PyValue.from(min_val)) + @as(f64, 0.00000001)));
    }
pub fn normalize_zscore(_: std.mem.Allocator, _: runtime.PyValue, data: i64) !i64 {
        _ = &data;
        _ = "Z-score standardization (mean=0, std=1).";
        const mean: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "mean", .{data.values}).?)));
        const std_: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "std", .{data.values}).?)));
        _ = &mean;
        _ = &std_;
        return try runtime.divideFloat((runtime.PyValue.from(data.values)).sub(runtime.PyValue.from(mean)), (runtime.PyValue.from(std_)).add(runtime.PyValue.from(@as(f64, @as(f64, 0.00000001)))));
    }
pub fn log_transform(_: runtime.PyValue, data: i64) i64 {
        _ = &data;
        _ = "Log transform with offset for zero handling.";
        return @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "log1p", .{data.values}).?));
    }
pub fn clip_outliers(_: runtime.PyValue, data: i64) i64 {
        _ = &data;
        _ = "Clip values to 3 standard deviations.";
        const mean: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "mean", .{data.values}).?)));
        const std_: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "std", .{data.values}).?)));
        const lower: runtime.PyValue = (runtime.PyValue.from(mean)).sub(runtime.PyValue.from((runtime.PyValue.from(@as(i64, 3))).mul(runtime.PyValue.from(std_))));
        const upper: runtime.PyValue = (runtime.PyValue.from(mean)).add(runtime.PyValue.from((runtime.PyValue.from(@as(i64, 3))).mul(runtime.PyValue.from(std_))));
        _ = &std_;
        _ = &lower;
        _ = &upper;
        return @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "clip", .{data.values, lower, upper}).?));
    }
pub fn polynomial_features(_: std.mem.Allocator, _: runtime.PyValue, data: i64) !i64 {
        _ = &data;
        _ = "Generate polynomial features (x, x^2, x^3).";
        const x: runtime.PyValue = runtime.PyValue.from(data.values);
        _ = &x;
        return @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "stack", .{__list_rt_0: {
            var __list_var_0 = std.ArrayListUnmanaged(runtime.PyValue){};
            try __list_var_0.append(__global_allocator, x);
            try __list_var_0.append(__global_allocator, (runtime.PyValue.from(x)).pyPow(runtime.PyValue.from(@as(i64, 2))));
            try __list_var_0.append(__global_allocator, (runtime.PyValue.from(x)).pyPow(runtime.PyValue.from(@as(i64, 3))));
            break :__list_rt_0 __list_var_0;
        }}).?));
    }
    pub const methods = [_][]const u8{
        "normalize_minmax",
        "normalize_zscore",
        "log_transform",
        "clip_outliers",
        "polynomial_features",
    };
};

pub const VectorSearch = struct {
    pub const __logic_table__ = true;

pub fn cosine_similarity(_: std.mem.Allocator, _: runtime.PyValue, query: i64, docs: i64) !i64 {
        _ = &query;
        _ = &docs;
        _ = "Cosine similarity between query and document embeddings.\n\n        query: shape (embedding_dim,)\n        docs: shape (num_docs, embedding_dim)\n        returns: shape (num_docs,) similarity scores in [-1, 1]\n        ";
        const query_norm: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy.linalg", "norm", .{query.embedding}).?)));
        const query_normalized: runtime.PyValue = (runtime.PyValue.from(query.embedding)).div(runtime.PyValue.from((runtime.PyValue.from(query_norm)).add(runtime.PyValue.from(@as(f64, @as(f64, 0.00000001))))));
        const docs_norm: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy.linalg", "norm", .{docs.embedding}).?)));
        const docs_normalized: runtime.PyValue = (runtime.PyValue.from(docs.embedding)).div(runtime.PyValue.from((runtime.PyValue.from(docs_norm)).add(runtime.PyValue.from(@as(f64, @as(f64, 0.00000001))))));
        _ = &query_normalized;
        _ = &docs_normalized;
        return @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "dot", .{docs_normalized, query_normalized}).?));
    }
pub fn euclidean_distance(_: runtime.PyValue, query: i64, docs: i64) i64 {
        _ = &query;
        _ = &docs;
        _ = "L2 (Euclidean) distance between query and documents.\n\n        Lower values = more similar.\n        ";
        const diff: runtime.PyValue = (runtime.PyValue.from(docs.embedding)).sub(runtime.PyValue.from(query.embedding));
        _ = &diff;
        return @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "sqrt", .{@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "sum", .{(runtime.PyValue.from(diff)).pyPow(runtime.PyValue.from(@as(i64, 2)))}).?))}).?));
    }
pub fn dot_product(_: runtime.PyValue, query: i64, docs: i64) i64 {
        _ = &query;
        _ = &docs;
        _ = "Raw dot product (for pre-normalized embeddings).";
        return @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "dot", .{docs.embedding, query.embedding}).?));
    }
pub fn manhattan_distance(_: runtime.PyValue, query: i64, docs: i64) i64 {
        _ = &query;
        _ = &docs;
        _ = "L1 (Manhattan) distance between query and documents.";
        const diff: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "abs", .{(runtime.PyValue.from(docs.embedding)).sub(runtime.PyValue.from(query.embedding))}).?)));
        _ = &diff;
        return @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "sum", .{diff}).?));
    }
    pub const methods = [_][]const u8{
        "cosine_similarity",
        "euclidean_distance",
        "dot_product",
        "manhattan_distance",
    };
};

pub const FraudDetection = struct {
    pub const __logic_table__ = true;

pub fn transaction_risk_score(_: std.mem.Allocator, _: runtime.PyValue, txn: i64) !i64 {
        _ = &txn;
        _ = "Multi-factor fraud risk score for transactions.\n\n        Factors:\n        - Amount: large transactions are riskier\n        - Velocity: many transactions in short time\n        - Location: unusual location patterns\n        - Time: unusual time of day\n        - History: past fraud incidents\n        ";
        var score: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "zeros_like", .{txn.amount}).?)));
        _ = &score;
        const amount_risk: runtime.PyValue = runtime.PyValue.from(runtime.subtractNum(@as(f64, 1.0), @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "exp", .{try runtime.divideFloat((unk_0: { const __v = txn.amount; const __T = @TypeOf(__v); break :unk_0 if (@typeInfo(__T) == .@"struct" and @hasDecl(__T, "negate")) (val_1: { var __tmp = __v.clone(__global_allocator) catch @panic("OOM"); __tmp.negate(); break :val_1 __tmp; }) else -__v; }), @as(f64, 5000.0))}).?))));
        score = (runtime.PyValue.from(score)).add(runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "minimum", .{@as(f64, 0.3), (runtime.PyValue.from(amount_risk)).mul(runtime.PyValue.from(@as(f64, @as(f64, 0.3))))}).?))));
        const velocity_risk: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "minimum", .{@as(f64, 1.0), (runtime.PyValue.from(txn.velocity)).div(runtime.PyValue.from(@as(f64, @as(f64, 10.0))))}).?)));
        score = (runtime.PyValue.from(score)).add(runtime.PyValue.from((runtime.PyValue.from(velocity_risk)).mul(runtime.PyValue.from(@as(f64, @as(f64, 0.25))))));
        const location_risk: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "minimum", .{@as(f64, 1.0), (runtime.PyValue.from(txn.location_distance)).div(runtime.PyValue.from(@as(f64, @as(f64, 1000.0))))}).?)));
        score = (runtime.PyValue.from(score)).add(runtime.PyValue.from((runtime.PyValue.from(location_risk)).mul(runtime.PyValue.from(@as(f64, @as(f64, 0.2))))));
        const hour: runtime.PyValue = runtime.PyValue.from(txn.hour);
        const time_risk: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "where", .{((runtime.PyValue.from(hour).ge(runtime.PyValue.from(2))) and (runtime.PyValue.from(hour).le(runtime.PyValue.from(5)))), @as(f64, 0.1), @as(f64, 0.0)}).?)));
        score = (runtime.PyValue.from(score)).add(runtime.PyValue.from(time_risk));
        const history_risk: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "minimum", .{@as(f64, 1.0), (runtime.PyValue.from(txn.fraud_count)).div(runtime.PyValue.from(@as(f64, @as(f64, 3.0))))}).?)));
        score = (runtime.PyValue.from(score)).add(runtime.PyValue.from((runtime.PyValue.from(history_risk)).mul(runtime.PyValue.from(@as(f64, @as(f64, 0.15))))));
        return @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "clip", .{score, @as(f64, 0.0), @as(f64, 1.0)}).?));
    }
pub fn anomaly_score(_: std.mem.Allocator, _: runtime.PyValue, txn: i64) !i64 {
        _ = &txn;
        _ = "Z-score based anomaly detection.";
        const amount_z: runtime.PyValue = runtime.PyValue.from(try runtime.divideFloat(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "abs", .{(runtime.PyValue.from(txn.amount)).sub(runtime.PyValue.from(txn.amount_mean))}).?)), (runtime.PyValue.from(txn.amount_std)).add(runtime.PyValue.from(@as(f64, @as(f64, 0.00000001))))));
        const velocity_z: runtime.PyValue = runtime.PyValue.from(try runtime.divideFloat(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "abs", .{(runtime.PyValue.from(txn.velocity)).sub(runtime.PyValue.from(txn.velocity_mean))}).?)), (runtime.PyValue.from(txn.velocity_std)).add(runtime.PyValue.from(@as(f64, @as(f64, 0.00000001))))));
        _ = &amount_z;
        _ = &velocity_z;
        return @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "maximum", .{amount_z, velocity_z}).?));
    }
    pub const methods = [_][]const u8{
        "transaction_risk_score",
        "anomaly_score",
    };
};

pub const Recommendations = struct {
    pub const __logic_table__ = true;

pub fn collaborative_score(_: std.mem.Allocator, _: runtime.PyValue, user: i64, items: i64) !i64 {
        _ = &user;
        _ = &items;
        _ = "Score items based on user-item embedding similarity.";
        const base_score: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "dot", .{items.embedding, user.embedding}).?)));
        const popularity_penalty: runtime.PyValue = runtime.PyValue.from((runtime.toFloat(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "log1p", .{items.view_count}).?))) * runtime.toFloat(@as(f64, 0.1))));
        const recency_boost: runtime.PyValue = runtime.PyValue.from((runtime.toFloat(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "exp", .{try runtime.divideFloat((unk_2: { const __v = items.age_days; const __T = @TypeOf(__v); break :unk_2 if (@typeInfo(__T) == .@"struct" and @hasDecl(__T, "negate")) (val_3: { var __tmp = __v.clone(__global_allocator) catch @panic("OOM"); __tmp.negate(); break :val_3 __tmp; }) else -__v; }), @as(f64, 30.0))}).?))) * runtime.toFloat(@as(f64, 0.2))));
        _ = &base_score;
        _ = &popularity_penalty;
        _ = &recency_boost;
        return (runtime.PyValue.from((runtime.PyValue.from(base_score)).sub(runtime.PyValue.from(popularity_penalty)))).add(runtime.PyValue.from(recency_boost));
    }
pub fn diversity_score(_: std.mem.Allocator, _: runtime.PyValue, candidates: i64) !i64 {
        _ = &candidates;
        _ = "Compute diversity penalty for result set.\n\n        Penalizes items too similar to earlier items in the list.\n        ";
        const n = @as(i64, @intCast(len_4: { const __obj = candidates.embedding; break :len_4 runtime.builtinLen(__obj); }));
        var diversity: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "ones", .{n}).?)));
        _ = &diversity;
        var i: isize = 1;
        while (i < n) {
            const current: runtime.PyValue = runtime.PyValue.from((sub_5: { const __base = candidates.embedding; break :sub_5 if (@TypeOf(__base) == runtime.PyValue) __base.pyAt(@as(usize, @intCast(i))) else __base[@as(usize, @intCast(i))]; }));
            const previous: runtime.PyValue = runtime.PyValue.from((sub_6: { const __base = candidates.embedding; break :sub_6 __base[0..@as(usize, @intCast(i))]; }));
            const similarities: runtime.PyValue = runtime.PyValue.from(try runtime.divideFloat(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "dot", .{previous, current}).?)), runtime.addNum((mul_7: { const _lhs = @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy.linalg", "norm", .{previous}).?)); const _rhs = @as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy.linalg", "norm", .{current}).?)); break :mul_7 if (@TypeOf(_lhs) == []const u8) (if (_rhs < 0) "" else runtime.strRepeat(__global_allocator, _lhs, @as(usize, @intCast(_rhs)))) else _lhs * _rhs; }), @as(f64, 0.00000001))));
            const max_sim: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "max", .{similarities}).?)));
            diversity[@as(usize, @intCast(i))] = (runtime.PyValue.from(@as(f64, @as(f64, 1.0)))).sub(runtime.PyValue.from(max_sim));
            i += 1;
        }
        return diversity;
    }
pub fn hybrid_score(_: std.mem.Allocator, _: runtime.PyValue, user: i64, items: i64) !i64 {
        _ = &user;
        _ = &items;
        _ = "Hybrid scoring combining multiple signals.";
        var embedding_score: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "dot", .{items.embedding, user.embedding}).?)));
        _ = &embedding_score;
        embedding_score = runtime.PyValue.from(try runtime.divideFloat((runtime.PyValue.from(embedding_score)).add(runtime.PyValue.from(@as(f64, @as(f64, 1.0)))), @as(f64, 2.0)));
        const category_match: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "where", .{(runtime.PyValue.from(items.category).eql(runtime.PyValue.from(user.preferred_category))), @as(f64, 1.0), @as(f64, 0.0)}).?)));
        const price_diff: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "abs", .{(runtime.PyValue.from(items.price)).sub(runtime.PyValue.from(user.preferred_price))}).?)));
        const price_score: runtime.PyValue = runtime.PyValue.from(@as(*runtime.PyObject, @ptrCast(c_interop.callModuleFunction("numpy", "exp", .{(runtime.PyValue.from((price_diff).neg())).div(runtime.PyValue.from(user.price_tolerance))}).?)));
        const rating_score: runtime.PyValue = (runtime.PyValue.from(items.avg_rating)).div(runtime.PyValue.from(@as(f64, @as(f64, 5.0))));
        _ = &category_match;
        _ = &price_score;
        _ = &rating_score;
        return ((((runtime.PyValue.from(@as(f64, @as(f64, 0.4)))).mul(runtime.PyValue.from(embedding_score)) + (runtime.PyValue.from(@as(f64, @as(f64, 0.2)))).mul(runtime.PyValue.from(category_match))) + (runtime.PyValue.from(@as(f64, @as(f64, 0.2)))).mul(runtime.PyValue.from(price_score))) + (runtime.PyValue.from(@as(f64, @as(f64, 0.2)))).mul(runtime.PyValue.from(rating_score)));
    }
    pub const methods = [_][]const u8{
        "collaborative_score",
        "diversity_score",
        "hybrid_score",
    };
};


// Module-level allocator for async functions and f-strings
// Browser WASM: FixedBufferAllocator (no std.Thread), Native: GPA
const __is_freestanding = @import("builtin").os.tag == .freestanding;
// Freestanding uses fixed buffer (64KB), native uses GPA
var __wasm_buffer: [64 * 1024]u8 = undefined;
var __fba = std.heap.FixedBufferAllocator.init(&__wasm_buffer);
var __gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true, .thread_safe = true }){};
var __global_allocator: std.mem.Allocator = undefined;
var __allocator_initialized: bool = false;
var __sys_argv: [][]const u8 = &[_][]const u8{};
var __sys_executable: []const u8 = "";
var __sys_platform: []const u8 = "unknown";
var __sys_byteorder: []const u8 = "little";

pub fn main() !void {
    const allocator = blk: {
        if (comptime __is_freestanding) {
            // Browser WASM: FixedBufferAllocator (no std.Thread)
            break :blk __fba.allocator();
        } else if (comptime allocator_helper.useFastAllocator()) {
            // Release mode: use c_allocator, OS reclaims at exit
            break :blk std.heap.c_allocator;
        } else {
            // Debug: use GPA for leak detection
            break :blk __gpa.allocator();
        }
    };

    __global_allocator = allocator;
    __allocator_initialized = true;
    __sys_argv = blk: {
        // In shared library mode or WASM, std.os.argv is invalid
        const builtin = @import("builtin");
        const is_wasm = builtin.os.tag == .wasi or builtin.os.tag == .freestanding;
        const is_lib = builtin.output_mode == .Lib or builtin.link_mode == .dynamic;
        if (comptime builtin.output_mode == .Exe and !is_wasm and !is_lib) {
            const os_args = std.os.argv;
            var argv_list = std.ArrayListUnmanaged([]const u8){};
            for (os_args) |arg| argv_list.append(allocator, std.mem.span(arg)) catch continue;
            break :blk argv_list.items;
        } else {
            break :blk &[_][]const u8{};
        }
    };
    __sys_executable = (__m1_sys_exec: {
        const __m0_builtin = @import("builtin");
        const is_wasm = __m0_builtin.os.tag == .wasi or __m0_builtin.os.tag == .freestanding;
        const is_lib = __m0_builtin.output_mode == .Lib or __m0_builtin.link_mode == .dynamic;
        if (comptime is_wasm or is_lib) break :__m1_sys_exec "";
        const args = std.os.argv;
        if (args.len > 0) break :__m1_sys_exec std.mem.span(args[0]);
        break :__m1_sys_exec "";
});
    __sys_platform = switch (@import("builtin").os.tag) { .linux => "linux", .macos => "darwin", .windows => "win32", .freebsd => "freebsd", else => "unknown" };
    __sys_byteorder = if (@import("builtin").cpu.arch.endian() == .little) "little" else "big";

    _ = "\nReal ML Workflow for @logic_table Benchmark\n\nThis file contains realistic ML/AI workloads that use numpy for:\n- Feature engineering (normalization, log transforms, z-scores)\n- Similarity search (cosine similarity, dot product, L2 distance)\n- Fraud detection scoring\n- Recommendation scoring\n\nCompiled by metal0: metal0 build --emit-logic-table benchmarks/ml_workflow.py -o lib/logic_table.a\n";
    if ((std.mem.eql(u8, __name__, "__main__"))) {
        runtime.builtins.print(__global_allocator, &.{"Verifying @logic_table classes..."});
        runtime.builtins.print(__global_allocator, &.{(try std.fmt.allocPrint(__global_allocator, "  FeatureEngineering.__logic_table__ = {s}", .{ (try runtime.builtins.pyStr(__global_allocator, runtime.PyValue.from(runtime.eval(__global_allocator, "FeatureEngineering.__logic_table__") catch unreachable))) }))});
        runtime.builtins.print(__global_allocator, &.{(try std.fmt.allocPrint(__global_allocator, "  VectorSearch.__logic_table__ = {s}", .{ (try runtime.builtins.pyStr(__global_allocator, runtime.PyValue.from(runtime.eval(__global_allocator, "VectorSearch.__logic_table__") catch unreachable))) }))});
        runtime.builtins.print(__global_allocator, &.{(try std.fmt.allocPrint(__global_allocator, "  FraudDetection.__logic_table__ = {s}", .{ (try runtime.builtins.pyStr(__global_allocator, runtime.PyValue.from(runtime.eval(__global_allocator, "FraudDetection.__logic_table__") catch unreachable))) }))});
        runtime.builtins.print(__global_allocator, &.{(try std.fmt.allocPrint(__global_allocator, "  Recommendations.__logic_table__ = {s}", .{ (try runtime.builtins.pyStr(__global_allocator, runtime.PyValue.from(runtime.eval(__global_allocator, "Recommendations.__logic_table__") catch unreachable))) }))});
        runtime.builtins.print(__global_allocator, &.{"All @logic_table classes verified!"});
    }
}
