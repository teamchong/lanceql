const std = @import("std");
const runtime = @import("./runtime.zig");
const hashmap_helper = @import("./utils/hashmap_helper.zig");
const allocator_helper = @import("./utils/allocator_helper.zig");


const __name__ = "__main__";
const __file__: []const u8 = "/var/folders/x4/x1pbbx6s6tl3nr7k0ftprwg40000gp/T/tmp8m_mbqtr.py";

// metal0 metadata for runtime eval subprocess
pub const __metal0_source_dir: []const u8 = "/var/folders/x4/x1pbbx6s6tl3nr7k0ftprwg40000gp/T";


// Module-level allocator for f-strings and dynamic allocations
var __gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
var __global_allocator: std.mem.Allocator = __gpa.allocator();

pub const tmp8m_mbqtr = struct {
    const VectorOps = struct {
        // Dynamic attributes dictionary
        __dict__: hashmap_helper.StringHashMap(runtime.PyValue),

        pub fn init(allocator: std.mem.Allocator) @This() {
            return @This(){
                .__dict__ = hashmap_helper.StringHashMap(runtime.PyValue).init(allocator),
            };
        }

        pub fn dot_product(_: *const @This(), a: anytype, b: anytype) f64 {
            const result: f64 = 0.0;
            {
                var i: isize = 0;
                while (i < @as(i64, @intCast(len_0: { const __tmp = a; break :len_0 if (@typeInfo(@TypeOf(__tmp)) == .@"struct" and @hasField(@TypeOf(__tmp), "items")) __tmp.items.len else runtime.pyLen(__tmp); }))) {
                    result = (result + blk: { const _lhs = a[i]; const _rhs = b[i]; break :blk if (@TypeOf(_lhs) == []const u8) runtime.strRepeat(__global_allocator, _lhs, @as(usize, @intCast(_rhs))) else _lhs * _rhs; });
                    i += 1;
                }
            }
            return result;
        }

        pub fn sum_squares(_: *const @This(), a: anytype) f64 {
            const result: f64 = 0.0;
            {
                var i: isize = 0;
                while (i < @as(i64, @intCast(len_1: { const __tmp = a; break :len_1 if (@typeInfo(@TypeOf(__tmp)) == .@"struct" and @hasField(@TypeOf(__tmp), "items")) __tmp.items.len else runtime.pyLen(__tmp); }))) {
                    result = (result + blk: { const _lhs = a[i]; const _rhs = a[i]; break :blk if (@TypeOf(_lhs) == []const u8) runtime.strRepeat(__global_allocator, _lhs, @as(usize, @intCast(_rhs))) else _lhs * _rhs; });
                    i += 1;
                }
            }
            return result;
        }
    };

};
