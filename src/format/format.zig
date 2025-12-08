//! Lance file format parsing module.
//!
//! This module handles parsing the Lance file format structure including:
//! - Footer (40 bytes at end of file)
//! - Column metadata (protobuf encoded)
//! - Version detection and handling
//! - High-level file reading

const std = @import("std");

pub const footer = @import("footer.zig");
pub const version = @import("version.zig");
pub const lance_file = @import("lance_file.zig");

// Re-export main types
pub const Footer = footer.Footer;
pub const Version = version.Version;
pub const LANCE_MAGIC = footer.LANCE_MAGIC;
pub const FOOTER_SIZE = footer.FOOTER_SIZE;
pub const LanceFile = lance_file.LanceFile;
pub const readFile = lance_file.readFile;

test {
    std.testing.refAllDecls(@This());
}
