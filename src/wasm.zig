//! WASM entry point for LanceQL.
//!
//! This module provides exported functions that can be called from JavaScript
//! in the browser. It uses extern functions for JavaScript callbacks.

const std = @import("std");
const lanceql = @import("lanceql");

// Use WASM allocator for browser environment
const allocator = std.heap.wasm_allocator;

// ============================================================================
// Exported Memory
// ============================================================================

/// Allocate memory for use by JavaScript.
export fn alloc(len: usize) ?[*]u8 {
    const slice = allocator.alloc(u8, len) catch return null;
    return slice.ptr;
}

/// Free memory allocated by alloc.
export fn free(ptr: [*]u8, len: usize) void {
    allocator.free(ptr[0..len]);
}

// ============================================================================
// Footer Parsing
// ============================================================================

/// Parse a Lance footer and return the number of columns.
/// Returns 0 on error.
export fn parseFooterGetColumns(data: [*]const u8, len: usize) u32 {
    if (len < lanceql.FOOTER_SIZE) return 0;

    const footer_data = data[len - lanceql.FOOTER_SIZE ..][0..lanceql.FOOTER_SIZE];
    const footer = lanceql.Footer.parse(footer_data) catch return 0;

    return footer.num_columns;
}

/// Parse a Lance footer and return the major version.
export fn parseFooterGetMajorVersion(data: [*]const u8, len: usize) u16 {
    if (len < lanceql.FOOTER_SIZE) return 0;

    const footer_data = data[len - lanceql.FOOTER_SIZE ..][0..lanceql.FOOTER_SIZE];
    const footer = lanceql.Footer.parse(footer_data) catch return 0;

    return footer.major_version;
}

/// Parse a Lance footer and return the minor version.
export fn parseFooterGetMinorVersion(data: [*]const u8, len: usize) u16 {
    if (len < lanceql.FOOTER_SIZE) return 0;

    const footer_data = data[len - lanceql.FOOTER_SIZE ..][0..lanceql.FOOTER_SIZE];
    const footer = lanceql.Footer.parse(footer_data) catch return 0;

    return footer.minor_version;
}

/// Check if the footer magic is valid.
export fn isValidLanceFile(data: [*]const u8, len: usize) bool {
    if (len < lanceql.FOOTER_SIZE) return false;

    const footer_data = data[len - lanceql.FOOTER_SIZE ..][0..lanceql.FOOTER_SIZE];
    _ = lanceql.Footer.parse(footer_data) catch return false;

    return true;
}

// ============================================================================
// Column Metadata Offset
// ============================================================================

/// Get the column metadata start offset from the footer.
export fn getColumnMetaStart(data: [*]const u8, len: usize) u64 {
    if (len < lanceql.FOOTER_SIZE) return 0;

    const footer_data = data[len - lanceql.FOOTER_SIZE ..][0..lanceql.FOOTER_SIZE];
    const footer = lanceql.Footer.parse(footer_data) catch return 0;

    return footer.column_meta_start;
}

/// Get the column metadata offsets table start from the footer.
export fn getColumnMetaOffsetsStart(data: [*]const u8, len: usize) u64 {
    if (len < lanceql.FOOTER_SIZE) return 0;

    const footer_data = data[len - lanceql.FOOTER_SIZE ..][0..lanceql.FOOTER_SIZE];
    const footer = lanceql.Footer.parse(footer_data) catch return 0;

    return footer.column_meta_offsets_start;
}

// ============================================================================
// Version Information
// ============================================================================

/// Get the LanceQL library version.
export fn getVersion() u32 {
    // Version 0.1.0 encoded as 0x000100
    return 0x000100;
}
