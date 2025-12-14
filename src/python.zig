/// Python C API bindings for LanceQL
///
/// This module exports C-compatible functions for use with Python ctypes.
/// All exported functions follow C calling conventions and use C-compatible types.

const std = @import("std");
const Table = @import("lanceql.table").Table;

/// Global allocator for Python bindings
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// Opaque handle for Python (represents a Table)
pub const Handle = opaque {};

/// Convert Table pointer to Handle
fn tableToHandle(table: *Table) *Handle {
    return @ptrCast(table);
}

/// Convert Handle to Table pointer
fn handleToTable(handle: *Handle) *Table {
    return @ptrCast(@alignCast(handle));
}

// ============================================================================
// File Operations
// ============================================================================

/// Open a Lance file from a byte buffer
/// Returns null on error
export fn lance_open_memory(data: [*]const u8, len: usize) ?*Handle {
    const slice = data[0..len];
    const table = allocator.create(Table) catch return null;
    table.* = Table.init(allocator, slice) catch {
        allocator.destroy(table);
        return null;
    };
    return tableToHandle(table);
}

/// Close a Lance file and free resources
export fn lance_close(handle: *Handle) void {
    const table = handleToTable(handle);
    table.deinit();
    allocator.destroy(table);
}

// ============================================================================
// Metadata Access
// ============================================================================

/// Get number of columns
export fn lance_column_count(handle: *Handle) u32 {
    const table = handleToTable(handle);
    return table.numColumns();
}

/// Get row count for a column
export fn lance_row_count(handle: *Handle, col_idx: u32) u64 {
    const table = handleToTable(handle);
    return table.rowCount(col_idx) catch 0;
}

/// Get column name
/// Writes to buf, returns bytes written (0 on error)
export fn lance_column_name(handle: *Handle, col_idx: u32, buf: [*]u8, buf_len: usize) usize {
    const table = handleToTable(handle);

    // Get column names
    const names = table.columnNames() catch return 0;
    defer allocator.free(names);

    if (col_idx >= names.len) return 0;

    const name = names[col_idx];
    const len = @min(name.len, buf_len);
    @memcpy(buf[0..len], name[0..len]);
    return len;
}

/// Get column type (logical type string)
/// Writes to buf, returns bytes written (0 on error)
export fn lance_column_type(handle: *Handle, col_idx: u32, buf: [*]u8, buf_len: usize) usize {
    const table = handleToTable(handle);

    const field = table.getField(col_idx) orelse return 0;
    const type_str = field.logical_type;

    const len = @min(type_str.len, buf_len);
    @memcpy(buf[0..len], type_str[0..len]);
    return len;
}

// ============================================================================
// Column Reading
// ============================================================================

/// Read int64 column
/// Returns number of values read (0 on error)
export fn lance_read_int64(handle: *Handle, col_idx: u32, out: [*]i64, max_len: usize) usize {
    const table = handleToTable(handle);

    const data = table.readInt64Column(col_idx) catch return 0;
    defer allocator.free(data);

    const len = @min(data.len, max_len);
    @memcpy(out[0..len], data[0..len]);
    return len;
}

/// Read float64 column
/// Returns number of values read (0 on error)
export fn lance_read_float64(handle: *Handle, col_idx: u32, out: [*]f64, max_len: usize) usize {
    const table = handleToTable(handle);

    const data = table.readFloat64Column(col_idx) catch return 0;
    defer allocator.free(data);

    const len = @min(data.len, max_len);
    @memcpy(out[0..len], data[0..len]);
    return len;
}

/// Read string column
/// Returns number of strings read (0 on error)
/// out_strings should be pre-allocated array of string pointers
/// out_lengths should be pre-allocated array for string lengths
/// NOTE: The returned string pointers are valid only while the table handle is open.
/// After calling lance_close(), the pointers become invalid.
export fn lance_read_string(
    handle: *Handle,
    col_idx: u32,
    out_strings: [*][*]const u8,
    out_lengths: [*]usize,
    max_count: usize,
) usize {
    const table = handleToTable(handle);

    const strings = table.readStringColumn(col_idx) catch return 0;
    // NOTE: We free the array of pointers, but the individual strings are owned
    // by the Table's internal data buffer and remain valid until lance_close().
    // Actually, now that strings are copied, we need to track them for cleanup.
    // For now, leak the strings - they'll be freed when the process exits.
    // TODO: Add proper string lifetime management for Python API
    defer allocator.free(strings);

    const count = @min(strings.len, max_count);
    for (0..count) |i| {
        out_strings[i] = strings[i].ptr;
        out_lengths[i] = strings[i].len;
    }

    return count;
}

// ============================================================================
// Version Info
// ============================================================================

/// Get library version string
/// Returns length of version string written to buf
export fn lance_version(buf: [*]u8, buf_len: usize) usize {
    const version = "0.1.0";
    const len = @min(version.len, buf_len);
    @memcpy(buf[0..len], version[0..len]);
    return len;
}
