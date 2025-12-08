//! Protobuf decoding for Lance metadata.
//!
//! This module provides a minimal protobuf wire format decoder
//! tailored for reading Lance file metadata. It only supports
//! decoding (not encoding) since LanceQL is read-only.

const std = @import("std");

pub const decoder = @import("decoder.zig");
pub const lance_messages = @import("lance_messages.zig");

// Re-export main types
pub const ProtoDecoder = decoder.ProtoDecoder;
pub const WireType = decoder.WireType;
pub const DecodeError = decoder.DecodeError;

pub const ColumnMetadata = lance_messages.ColumnMetadata;
pub const Page = lance_messages.Page;
pub const Encoding = lance_messages.Encoding;

test {
    std.testing.refAllDecls(@This());
}
