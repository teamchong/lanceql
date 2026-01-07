const std = @import("std");

// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const wasm_allocator = std.heap.page_allocator;

pub fn wasmAlloc(len: usize) ?[*]u8 {
    const count = (len + 7) / 8;
    const slice = wasm_allocator.alloc(u64, count) catch return null;
    return @ptrCast(slice.ptr);
}

/// No-op as GPA doesn't support global reset, but we shouldn't need it if we free
pub fn wasmReset() void {
    // This was used for the bump allocator. 
    // Now we rely on individual frees.
}

/// Free memory (exported to JavaScript)
pub export fn free(ptr: [*]u8, len: usize) void {
    wasm_allocator.free(ptr[0..len]);
}

/// Get current heap usage (dummy as GPA doesn't track this easily)
pub fn getHeapUsage() usize {
    return 0;
}

/// Get total heap capacity (dummy)
pub fn getHeapCapacity() usize {
    return 64 * 1024 * 1024;
}

// ============================================================================
// WASM Exports
// ============================================================================

/// Allocate memory (exported to JavaScript)
pub export fn alloc(len: usize) ?[*]u8 {
    return wasmAlloc(len);
}

// ============================================================================
// Tests
// ============================================================================

test "memory: basic allocation" {
    wasmReset();
    const ptr1 = wasmAlloc(100);
    try std.testing.expect(ptr1 != null);

    const ptr2 = wasmAlloc(200);
    try std.testing.expect(ptr2 != null);

    // Pointers should be different and properly spaced
    const addr1 = @intFromPtr(ptr1.?);
    const addr2 = @intFromPtr(ptr2.?);
    try std.testing.expect(addr2 > addr1);

    wasmReset();
}

test "memory: alignment" {
    wasmReset();
    _ = wasmAlloc(1); // Allocate 1 byte
    const ptr = wasmAlloc(8);
    try std.testing.expect(ptr != null);

    // Should be 8-byte aligned
    const addr = @intFromPtr(ptr.?);
    try std.testing.expectEqual(@as(usize, 0), addr % 8);

    wasmReset();
}

test "memory: heap capacity" {
    try std.testing.expectEqual(@as(usize, 64 * 1024 * 1024), getHeapCapacity());
}

/// Free memory allocated by wasmAlloc
pub fn wasmFree(ptr: [*]u8, len: usize) void {
    const count = (len + 7) / 8;
    const slice = @as([*]u64, @ptrCast(@alignCast(ptr)))[0..count];
    wasm_allocator.free(slice);
}

