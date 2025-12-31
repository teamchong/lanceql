//! Data encoding/decoding for Lance columns.
//!
//! Lance supports various encodings for column data:
//! - Plain: Direct value storage (int64, float64)
//! - Dictionary: Categorical data with lookup table
//! - RLE: Run-length encoding for repeated values
//! - UTF-8: String data with offset arrays
//!
//! Also provides parsers for common formats:
//! - CSV/TSV: Delimiter-separated values

const std = @import("std");

pub const plain = @import("plain.zig");
pub const writer = @import("writer.zig");
pub const csv = @import("csv.zig");
pub const json = @import("json.zig");

// Re-export main types
pub const PlainDecoder = plain.PlainDecoder;
pub const PlainEncoder = writer.PlainEncoder;
pub const LanceWriter = writer.LanceWriter;
pub const FooterWriter = writer.FooterWriter;
pub const ProtobufEncoder = writer.ProtobufEncoder;
pub const DataType = writer.DataType;
pub const ColumnSchema = writer.ColumnSchema;
pub const ColumnBatch = writer.ColumnBatch;

// CSV types
pub const CsvParser = csv.CsvParser;
pub const CsvConfig = csv.Config;
pub const CsvField = csv.Field;
pub const CsvColumnType = csv.ColumnType;
pub const CsvColumnData = csv.ColumnData;
pub const readCsv = csv.readCsv;
pub const detectCsvDelimiter = csv.detectDelimiter;

// JSON types
pub const JsonFormat = json.Format;
pub const JsonConfig = json.Config;
pub const JsonColumnType = json.ColumnType;
pub const readJson = json.readJson;
pub const detectJsonFormat = json.detectFormat;

test {
    std.testing.refAllDecls(@This());
}
