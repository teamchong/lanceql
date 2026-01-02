//! WASM Memory Management
//!
//! Simple bump allocator for WASM environment.
//! No std.heap.wasm_allocator dependency for minimal binary size.

const std = @import("std");

// ============================================================================
// Bump Allocator
// ============================================================================

var heap: [1024 * 1024]u8 = undefined; // 1MB heap
var heap_offset: usize = 0;

/// Allocate memory from the bump allocator (8-byte aligned)
pub fn wasmAlloc(len: usize) ?[*]u8 {
    // Get current address and align it to 8 bytes
    const heap_base = @intFromPtr(&heap[0]);
    const current_addr = heap_base + heap_offset;
    const aligned_addr = (current_addr + 7) & ~@as(usize, 7);
    const padding = aligned_addr - current_addr;

    heap_offset += padding;

    const aligned_len = (len + 7) & ~@as(usize, 7); // 8-byte align length too
    if (heap_offset + aligned_len > heap.len) return null;
    const ptr: [*]u8 = @ptrCast(&heap[heap_offset]);
    heap_offset += aligned_len;
    return ptr;
}

/// Reset the bump allocator (free all memory)
pub fn wasmReset() void {
    heap_offset = 0;
}

/// Get current heap usage
pub fn getHeapUsage() usize {
    return heap_offset;
}

/// Get total heap capacity
pub fn getHeapCapacity() usize {
    return heap.len;
}

// ============================================================================
// WASM Exports
// ============================================================================

/// Allocate memory (exported to JavaScript)
pub export fn alloc(len: usize) ?[*]u8 {
    return wasmAlloc(len);
}

/// Free memory (no-op for bump allocator)
pub export fn free(ptr: [*]u8, len: usize) void {
    _ = ptr;
    _ = len;
    // Bump allocator doesn't support individual frees
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
    try std.testing.expectEqual(@as(usize, 1024 * 1024), getHeapCapacity());
}
