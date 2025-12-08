//! Data encoding/decoding for Lance columns.
//!
//! Lance supports various encodings for column data:
//! - Plain: Direct value storage (int64, float64)
//! - Dictionary: Categorical data with lookup table
//! - RLE: Run-length encoding for repeated values
//! - UTF-8: String data with offset arrays

const std = @import("std");

pub const plain = @import("plain.zig");

// Re-export main types
pub const PlainDecoder = plain.PlainDecoder;

test {
    std.testing.refAllDecls(@This());
}
