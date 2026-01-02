//! LanceQL Ingest Command
//!
//! Converts data files to Lance format.
//! Supports: CSV, TSV, JSON, JSONL, Parquet, Arrow, Avro, ORC, XLSX, Delta, Iceberg
//!
//! Usage:
//!   lanceql ingest data.csv -o output.lance
//!   lanceql ingest data.json --format jsonl -o output.lance
//!   lanceql ingest data.arrow -o output.lance
//!   lanceql ingest ./delta_table/ --format delta -o output.lance

const std = @import("std");
const lanceql = @import("lanceql");
const csv = lanceql.encoding.csv;
const json = lanceql.encoding.json;
const arrow_ipc = lanceql.encoding.arrow_ipc;
const avro = lanceql.encoding.avro;
const orc = lanceql.encoding.orc;
const xlsx = lanceql.encoding.xlsx;
const delta = lanceql.encoding.delta;
const iceberg = lanceql.encoding.iceberg;
const writer = lanceql.encoding.writer;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const parquet_meta = lanceql.format.parquet_metadata;
const args = @import("args.zig");

pub const IngestError = error{
    NoInputFile,
    NoOutputFile,
    UnsupportedFormat,
    FileReadError,
    WriteError,
    OutOfMemory,
};

/// Convert CSV column type to Lance data type
fn csvTypeToLanceType(csv_type: csv.ColumnType) writer.DataType {
    return switch (csv_type) {
        .int64 => .int64,
        .float64 => .float64,
        .bool_ => .bool,
        .string => .string,
    };
}

/// Finalize Lance writer and write to output file
fn finalizeLanceFile(lance_writer: *writer.LanceWriter, output_path: []const u8) !void {
    const lance_data = try lance_writer.finalize();
    const out_file = std.fs.cwd().createFile(output_path, .{}) catch |err| {
        std.debug.print("Error creating output file: {}\n", .{err});
        return error.WriteError;
    };
    defer out_file.close();
    out_file.writeAll(lance_data) catch |err| {
        std.debug.print("Error writing output file: {}\n", .{err});
        return error.WriteError;
    };
    std.debug.print("Created: {s} ({d} bytes)\n", .{ output_path, lance_data.len });
}

/// Detect file format from extension or directory structure
fn detectFormat(allocator: std.mem.Allocator, path: []const u8) args.IngestOptions.Format {
    // File extension detection
    if (std.mem.endsWith(u8, path, ".csv")) return .csv;
    if (std.mem.endsWith(u8, path, ".tsv")) return .tsv;
    if (std.mem.endsWith(u8, path, ".json")) return .json;
    if (std.mem.endsWith(u8, path, ".jsonl") or std.mem.endsWith(u8, path, ".ndjson")) return .jsonl;
    if (std.mem.endsWith(u8, path, ".parquet")) return .parquet;
    if (std.mem.endsWith(u8, path, ".arrow") or std.mem.endsWith(u8, path, ".arrows") or std.mem.endsWith(u8, path, ".feather")) return .arrow;
    if (std.mem.endsWith(u8, path, ".avro")) return .avro;
    if (std.mem.endsWith(u8, path, ".orc")) return .orc;
    if (std.mem.endsWith(u8, path, ".xlsx") or std.mem.endsWith(u8, path, ".xls")) return .xlsx;

    // Directory-based formats - check for Delta Lake (_delta_log) or Iceberg (metadata/)
    if (delta.DeltaReader.isValid(path)) return .delta;
    if (iceberg.IcebergReader.isValid(path)) return .iceberg;

    // Try magic-byte detection for files without extension
    if (detectFormatFromContent(allocator, path)) |fmt| return fmt;

    return .csv; // default to CSV
}

/// Detect format from file magic bytes
fn detectFormatFromContent(allocator: std.mem.Allocator, path: []const u8) ?args.IngestOptions.Format {
    const file = std.fs.cwd().openFile(path, .{}) catch return null;
    defer file.close();

    // Read first few bytes for magic detection
    var header: [16]u8 = undefined;
    const bytes_read = file.read(&header) catch return null;
    if (bytes_read < 4) return null;

    const data = header[0..bytes_read];

    // Arrow IPC: "ARROW1"
    if (bytes_read >= 6 and std.mem.eql(u8, data[0..6], "ARROW1")) return .arrow;

    // Avro: "Obj\x01"
    if (bytes_read >= 4 and std.mem.eql(u8, data[0..4], &[_]u8{ 'O', 'b', 'j', 1 })) return .avro;

    // ORC: "ORC" at start
    if (bytes_read >= 3 and std.mem.eql(u8, data[0..3], "ORC")) return .orc;

    // XLSX/ZIP: "PK\x03\x04"
    if (bytes_read >= 4 and std.mem.readInt(u32, data[0..4], .little) == 0x04034b50) return .xlsx;

    // Parquet: "PAR1"
    if (bytes_read >= 4 and std.mem.eql(u8, data[0..4], "PAR1")) return .parquet;

    // JSON: starts with '{' or '['
    if (data[0] == '{' or data[0] == '[') {
        // Check if it's JSONL (multiple objects separated by newlines)
        const full_data = file.readToEndAlloc(allocator, 1024) catch return .json;
        defer allocator.free(full_data);
        if (std.mem.indexOf(u8, full_data, "}\n{") != null) return .jsonl;
        return .json;
    }

    return null;
}

/// Run the ingest command
pub fn run(allocator: std.mem.Allocator, opts: args.IngestOptions) !void {
    const input_path = opts.input orelse {
        std.debug.print("Error: Input file required.\n", .{});
        args.printIngestHelp();
        return;
    };

    const output_path = opts.output orelse {
        std.debug.print("Error: Output file required. Use -o <output.lance>\n", .{});
        return;
    };

    // Determine format
    const format = if (opts.format == .auto) detectFormat(allocator, input_path) else opts.format;
    std.debug.print("Format: {s}\n", .{@tagName(format)});

    // Directory-based formats don't need file reading
    switch (format) {
        .delta => {
            try ingestDelta(allocator, input_path, output_path);
            return;
        },
        .iceberg => {
            try ingestIceberg(allocator, input_path, output_path);
            return;
        },
        else => {},
    }

    // Read input file
    const file = std.fs.cwd().openFile(input_path, .{}) catch |err| {
        std.debug.print("Error opening '{s}': {}\n", .{ input_path, err });
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch |err| {
        std.debug.print("Error reading file: {}\n", .{err});
        return;
    };
    defer allocator.free(data);

    // Process based on format
    switch (format) {
        .csv, .tsv => {
            try ingestCsv(allocator, data, output_path, .{
                .delimiter = if (format == .tsv) '\t' else opts.delimiter,
                .has_header = opts.header,
            });
        },
        .json, .jsonl => {
            try ingestJson(allocator, data, output_path, .{});
        },
        .parquet => {
            try ingestParquet(allocator, data, output_path);
        },
        .arrow => {
            try ingestArrow(allocator, data, output_path);
        },
        .avro => {
            try ingestAvro(allocator, data, output_path);
        },
        .orc => {
            try ingestOrc(allocator, data, output_path);
        },
        .xlsx => {
            try ingestXlsx(allocator, data, output_path);
        },
        .delta, .iceberg => unreachable, // Handled above
        .auto => unreachable,
    }
}

/// Ingest CSV data to Lance file
pub fn ingestCsv(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
    config: csv.Config,
) !void {
    std.debug.print("Parsing CSV...\n", .{});

    // Parse CSV
    const result = try csv.readCsv(allocator, data, config);
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    std.debug.print("  Rows: {d}\n", .{result.row_count});
    std.debug.print("  Columns: {d}\n", .{result.columns.len});

    // Print column info
    for (result.columns) |col| {
        std.debug.print("    - {s}: {s}\n", .{ col.name, col.col_type.format() });
    }

    // Build Lance schema
    var schema = try allocator.alloc(writer.ColumnSchema, result.columns.len);
    defer allocator.free(schema);

    for (result.columns, 0..) |col, i| {
        schema[i] = .{
            .name = col.name,
            .data_type = csvTypeToLanceType(col.col_type),
        };
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    // Encode and write each column
    std.debug.print("Writing Lance file...\n", .{});

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    for (result.columns, 0..) |col, i| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        var offsets_slice: ?[]const u8 = null;

        switch (col.col_type) {
            .int64 => try encoder.writeInt64Slice(col.int64_values.items),
            .float64 => try encoder.writeFloat64Slice(col.float64_values.items),
            .bool_ => try encoder.writeBools(col.bool_values.items),
            .string => {
                try encoder.writeStrings(col.string_values.items, &offsets_buf, allocator);
                offsets_slice = offsets_buf.items;
            },
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(i),
            .data = encoder.getBytes(),
            .row_count = @intCast(col.len()),
            .offsets = offsets_slice,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    try finalizeLanceFile(&lance_writer, output_path);
}

/// Ingest JSON/JSONL data to Lance file
pub fn ingestJson(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
    config: json.Config,
) !void {
    std.debug.print("Parsing JSON...\n", .{});

    const detected_format = json.detectFormat(data);
    std.debug.print("  Format: {s}\n", .{detected_format.format()});

    // Parse JSON
    const result = json.readJson(allocator, data, config) catch |err| {
        std.debug.print("Error parsing JSON: {}\n", .{err});
        return;
    };
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    std.debug.print("  Rows: {d}\n", .{result.row_count});
    std.debug.print("  Columns: {d}\n", .{result.columns.len});

    // Print column info
    for (result.columns) |col| {
        std.debug.print("    - {s}: {s}\n", .{ col.name, col.col_type.format() });
    }

    // Build Lance schema
    var schema = try allocator.alloc(writer.ColumnSchema, result.columns.len);
    defer allocator.free(schema);

    for (result.columns, 0..) |col, i| {
        schema[i] = .{
            .name = col.name,
            .data_type = csvTypeToLanceType(col.col_type),
        };
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    // Encode and write each column
    std.debug.print("Writing Lance file...\n", .{});

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    for (result.columns, 0..) |col, i| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        var offsets_slice: ?[]const u8 = null;

        switch (col.col_type) {
            .int64 => try encoder.writeInt64Slice(col.int64_values.items),
            .float64 => try encoder.writeFloat64Slice(col.float64_values.items),
            .bool_ => try encoder.writeBools(col.bool_values.items),
            .string => {
                try encoder.writeStrings(col.string_values.items, &offsets_buf, allocator);
                offsets_slice = offsets_buf.items;
            },
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(i),
            .data = encoder.getBytes(),
            .row_count = @intCast(col.len()),
            .offsets = offsets_slice,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    try finalizeLanceFile(&lance_writer, output_path);
}

/// Map Parquet physical type to Lance DataType
fn parquetTypeToLanceType(pq_type: parquet_meta.Type) writer.DataType {
    return switch (pq_type) {
        .boolean => .bool,
        .int32 => .int32,
        .int64 => .int64,
        .float => .float32,
        .double => .float64,
        .byte_array => .string,
        .fixed_len_byte_array => .string,
        .int96 => .int64, // Truncate timestamp
        _ => .string, // Fallback for unknown types
    };
}

/// Ingest Parquet data to Lance file
pub fn ingestParquet(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
) !void {
    std.debug.print("Parsing Parquet file...\n", .{});

    // Open Parquet table
    var pq_table = ParquetTable.init(allocator, data) catch |err| {
        std.debug.print("Error parsing Parquet: {}\n", .{err});
        return;
    };
    defer pq_table.deinit();

    const num_cols = pq_table.numColumns();
    const num_rows = pq_table.numRows();
    const col_names = pq_table.getColumnNames();

    std.debug.print("  Rows: {d}\n", .{num_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});

    // Build Lance schema
    var schema = try allocator.alloc(writer.ColumnSchema, num_cols);
    defer allocator.free(schema);

    for (0..num_cols) |i| {
        const pq_type = pq_table.getColumnType(i) orelse .byte_array;
        const lance_type = parquetTypeToLanceType(pq_type);
        schema[i] = .{
            .name = col_names[i],
            .data_type = lance_type,
        };
        std.debug.print("    - {s}: {s} -> {s}\n", .{
            col_names[i],
            @tagName(pq_type),
            @tagName(lance_type),
        });
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    // Encode each column
    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    std.debug.print("Converting columns...\n", .{});

    for (0..num_cols) |col_idx| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        var offsets_slice: ?[]const u8 = null;
        var row_count: u32 = 0;

        const lance_type = schema[col_idx].data_type;

        switch (lance_type) {
            .int64 => {
                const values = pq_table.readInt64Column(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read int64 column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeInt64Slice(values);
                row_count = @intCast(values.len);
            },
            .int32 => {
                const values = pq_table.readInt32Column(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read int32 column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeInt32Slice(values);
                row_count = @intCast(values.len);
            },
            .float64 => {
                const values = pq_table.readFloat64Column(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read float64 column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeFloat64Slice(values);
                row_count = @intCast(values.len);
            },
            .float32 => {
                const values = pq_table.readFloat32Column(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read float32 column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeFloat32Slice(values);
                row_count = @intCast(values.len);
            },
            .bool => {
                const values = pq_table.readBoolColumn(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read bool column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeBools(values);
                row_count = @intCast(values.len);
            },
            .string => {
                const values = pq_table.readStringColumn(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read string column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer {
                    for (values) |s| allocator.free(s);
                    allocator.free(values);
                }
                try encoder.writeStrings(values, &offsets_buf, allocator);
                offsets_slice = offsets_buf.items;
                row_count = @intCast(values.len);
            },
            else => {
                std.debug.print("  Skipping unsupported type: {s}\n", .{@tagName(lance_type)});
                continue;
            },
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(col_idx),
            .data = encoder.getBytes(),
            .row_count = row_count,
            .offsets = offsets_slice,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    try finalizeLanceFile(&lance_writer, output_path);
}

/// Map Arrow type to Lance DataType
fn arrowTypeToLanceType(arrow_type: arrow_ipc.ArrowType) writer.DataType {
    return switch (arrow_type) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => .int64,
        .float32 => .float32,
        .float64 => .float64,
        .utf8, .large_utf8, .binary, .large_binary => .string,
        .bool_type => .bool,
        else => .string,
    };
}

/// Ingest Arrow IPC data to Lance file
pub fn ingestArrow(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
) !void {
    std.debug.print("Parsing Arrow IPC file...\n", .{});

    var reader = arrow_ipc.ArrowIpcReader.init(allocator, data) catch |err| {
        std.debug.print("Error parsing Arrow: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const num_cols = reader.columnCount();
    const num_rows = reader.rowCount();

    std.debug.print("  Rows: {d}\n", .{num_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});

    if (num_cols == 0 or num_rows == 0) {
        std.debug.print("  No data to convert\n", .{});
        return;
    }

    // First pass: determine which columns we can support
    var supported_cols = std.ArrayListUnmanaged(usize){};
    defer supported_cols.deinit(allocator);

    for (0..num_cols) |i| {
        const arrow_type = reader.getColumnType(i);
        const lance_type = arrowTypeToLanceType(arrow_type);
        switch (lance_type) {
            .int64, .float64, .string => {
                try supported_cols.append(allocator, i);
                std.debug.print("    - {s}: {s} -> {s}\n", .{
                    reader.getColumnName(i),
                    @tagName(arrow_type),
                    @tagName(lance_type),
                });
            },
            else => {
                std.debug.print("    - {s}: {s} (skipped - unsupported type)\n", .{
                    reader.getColumnName(i),
                    @tagName(arrow_type),
                });
            },
        }
    }

    if (supported_cols.items.len == 0) {
        std.debug.print("  No supported columns to convert\n", .{});
        return;
    }

    // Build Lance schema only for supported columns
    var schema = try allocator.alloc(writer.ColumnSchema, supported_cols.items.len);
    defer allocator.free(schema);

    for (supported_cols.items, 0..) |orig_idx, i| {
        const arrow_type = reader.getColumnType(orig_idx);
        schema[i] = .{
            .name = reader.getColumnName(orig_idx),
            .data_type = arrowTypeToLanceType(arrow_type),
        };
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    // Encode each supported column
    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    std.debug.print("Converting columns...\n", .{});

    for (supported_cols.items, 0..) |orig_idx, schema_idx| {
        encoder.reset();

        const arrow_type = reader.getColumnType(orig_idx);
        const lance_type = arrowTypeToLanceType(arrow_type);

        switch (lance_type) {
            .int64 => {
                const values = reader.readInt64Column(orig_idx) catch |err| {
                    std.debug.print("  Error reading int64 column {d}: {}\n", .{orig_idx, err});
                    return;
                };
                defer allocator.free(values);
                try encoder.writeInt64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .float64 => {
                const values = reader.readFloat64Column(orig_idx) catch |err| {
                    std.debug.print("  Error reading float64 column {d}: {}\n", .{orig_idx, err});
                    return;
                };
                defer allocator.free(values);
                try encoder.writeFloat64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .string => {
                const values = reader.readStringColumn(orig_idx) catch |err| {
                    std.debug.print("  Error reading string column {d}: {}\n", .{ orig_idx, err });
                    return;
                };
                defer {
                    for (values) |v| allocator.free(v);
                    allocator.free(values);
                }

                var offsets_buf = std.ArrayListUnmanaged(u8){};
                defer offsets_buf.deinit(allocator);
                try encoder.writeStrings(values, &offsets_buf, allocator);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = offsets_buf.items,
                });
            },
            else => {},
        }
    }

    try finalizeLanceFile(&lance_writer, output_path);
}

/// Ingest Avro data to Lance file
/// Convert Avro type to Lance type
fn avroTypeToLanceType(avro_type: avro.AvroType) writer.DataType {
    return switch (avro_type) {
        .long_type, .int_type => .int64,
        .double_type, .float_type => .float64,
        .string, .bytes => .string,
        .boolean => .bool,
        else => .string, // Default fallback
    };
}

pub fn ingestAvro(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
) !void {
    std.debug.print("Parsing Avro file...\n", .{});

    var reader = avro.AvroReader.init(allocator, data) catch |err| {
        std.debug.print("Error parsing Avro: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const num_cols = reader.columnCount();
    const num_rows = reader.rowCount();

    std.debug.print("  Rows: {d}\n", .{num_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});
    std.debug.print("  Codec: {s}\n", .{@tagName(reader.codec)});

    if (num_cols == 0 or num_rows == 0) {
        std.debug.print("  No data to convert\n", .{});
        return;
    }

    // Check for compressed data - not yet supported
    if (reader.codec != .null) {
        std.debug.print("  Compressed Avro ({s}) not yet supported\n", .{@tagName(reader.codec)});
        return;
    }

    // First pass: determine which columns we can support
    var supported_cols = std.ArrayListUnmanaged(usize){};
    defer supported_cols.deinit(allocator);

    for (0..num_cols) |i| {
        const avro_type = reader.getFieldType(i) orelse continue;
        const lance_type = avroTypeToLanceType(avro_type);
        const name = reader.getFieldName(i) orelse "unknown";

        switch (avro_type) {
            .long_type, .int_type, .double_type, .float_type, .string => {
                try supported_cols.append(allocator, i);
                std.debug.print("    - {s}: {s} -> {s}\n", .{
                    name,
                    @tagName(avro_type),
                    @tagName(lance_type),
                });
            },
            else => {
                std.debug.print("    - {s}: {s} (skipped - unsupported type)\n", .{
                    name,
                    @tagName(avro_type),
                });
            },
        }
    }

    if (supported_cols.items.len == 0) {
        std.debug.print("  No supported columns to convert\n", .{});
        return;
    }

    // Build Lance schema only for supported columns
    var schema = try allocator.alloc(writer.ColumnSchema, supported_cols.items.len);
    defer allocator.free(schema);

    for (supported_cols.items, 0..) |orig_idx, i| {
        const avro_type = reader.getFieldType(orig_idx) orelse {
            std.debug.print("Error: Missing type for supported column {d}\n", .{orig_idx});
            return;
        };
        schema[i] = .{
            .name = reader.getFieldName(orig_idx) orelse "unknown",
            .data_type = avroTypeToLanceType(avro_type),
        };
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    // Encode each supported column
    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    std.debug.print("Converting columns...\n", .{});

    for (supported_cols.items, 0..) |orig_idx, schema_idx| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        const avro_type = reader.getFieldType(orig_idx) orelse continue;

        switch (avro_type) {
            .long_type, .int_type => {
                const values = reader.readLongColumn(orig_idx) catch |err| {
                    std.debug.print("  Error reading long column {d}: {}\n", .{orig_idx, err});
                    return;
                };
                defer allocator.free(values);
                try encoder.writeInt64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .double_type, .float_type => {
                const values = reader.readDoubleColumn(orig_idx) catch |err| {
                    std.debug.print("  Error reading double column {d}: {}\n", .{orig_idx, err});
                    return;
                };
                defer allocator.free(values);
                try encoder.writeFloat64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .string => {
                const values = reader.readStringColumn(orig_idx) catch |err| {
                    std.debug.print("  Error reading string column {d}: {}\n", .{orig_idx, err});
                    return;
                };
                defer {
                    for (values) |s| allocator.free(s);
                    allocator.free(values);
                }

                try encoder.writeStrings(values, &offsets_buf, allocator);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = offsets_buf.items,
                });
            },
            else => {},
        }
    }

    try finalizeLanceFile(&lance_writer, output_path);
}

/// Convert ORC type to Lance type
fn orcTypeToLanceType(orc_type: orc.OrcType) writer.DataType {
    return switch (orc_type) {
        .long, .int, .short, .byte => .int64,
        .double, .float => .float64,
        .string, .varchar, .char, .binary => .string,
        else => .string, // Default fallback
    };
}

/// Ingest ORC data to Lance file
pub fn ingestOrc(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
) !void {
    std.debug.print("Parsing ORC file...\n", .{});

    var reader = orc.OrcReader.init(allocator, data) catch |err| {
        std.debug.print("Error parsing ORC: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const num_cols = reader.columnCount();
    const num_rows = reader.rowCount();
    const num_stripes = reader.stripeCount();

    std.debug.print("  Rows: {d}\n", .{num_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});
    std.debug.print("  Stripes: {d}\n", .{num_stripes});
    std.debug.print("  Compression: {s}\n", .{@tagName(reader.compression)});

    if (num_cols == 0 or num_rows == 0 or num_stripes == 0) {
        std.debug.print("  No data to convert\n", .{});
        return;
    }

    // ORC has a struct column at index 0 that contains all other columns
    // Real columns start at index 1
    // For our test fixtures: id (1), name (2), value (3)
    const column_types = reader.column_types orelse &[_]orc.OrcType{};

    // Determine which columns we can support (skip struct column at index 0)
    var supported_cols = std.ArrayListUnmanaged(usize){};
    defer supported_cols.deinit(allocator);

    for (0..num_cols) |i| {
        const orc_type: orc.OrcType = if (i < column_types.len) column_types[i] else .unknown;
        const lance_type = orcTypeToLanceType(orc_type);

        switch (orc_type) {
            .long, .int, .short, .byte, .double, .float, .string, .varchar, .char => {
                try supported_cols.append(allocator, i);
                std.debug.print("    - col{d}: {s} -> {s}\n", .{
                    i,
                    @tagName(orc_type),
                    @tagName(lance_type),
                });
            },
            .struct_type => {
                // Skip struct column (root container)
                std.debug.print("    - col{d}: struct (skipped - container)\n", .{i});
            },
            else => {
                std.debug.print("    - col{d}: {s} (skipped - unsupported)\n", .{
                    i,
                    @tagName(orc_type),
                });
            },
        }
    }

    if (supported_cols.items.len == 0) {
        std.debug.print("  No supported columns to convert\n", .{});
        return;
    }

    // Build Lance schema
    var schema = try allocator.alloc(writer.ColumnSchema, supported_cols.items.len);
    defer allocator.free(schema);

    const column_names = reader.column_names orelse &[_][]const u8{};

    for (supported_cols.items, 0..) |orig_idx, i| {
        const orc_type: orc.OrcType = if (orig_idx < column_types.len) column_types[orig_idx] else .unknown;

        // Get column name if available
        const col_name = if (orig_idx < column_names.len)
            column_names[orig_idx]
        else
            ""; // Will be replaced with generated name

        // Generate name if empty
        var name_buf: [32]u8 = undefined;
        const name = if (col_name.len > 0)
            col_name
        else blk: {
            const len = std.fmt.bufPrint(&name_buf, "col{d}", .{orig_idx}) catch "unknown";
            break :blk len;
        };

        schema[i] = .{
            .name = name,
            .data_type = orcTypeToLanceType(orc_type),
        };
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    // Encode each supported column
    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    std.debug.print("Converting columns...\n", .{});

    for (supported_cols.items, 0..) |orig_idx, schema_idx| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        const orc_type = if (orig_idx < column_types.len) column_types[orig_idx] else .unknown;

        switch (orc_type) {
            .long, .int, .short, .byte => {
                const values = reader.readLongColumn(@intCast(orig_idx)) catch |err| {
                    std.debug.print("  Error reading long column {d}: {}\n", .{ orig_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeInt64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .double, .float => {
                const values = reader.readDoubleColumn(@intCast(orig_idx)) catch |err| {
                    std.debug.print("  Error reading double column {d}: {}\n", .{ orig_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeFloat64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .string, .varchar, .char => {
                const values = reader.readStringColumn(@intCast(orig_idx)) catch |err| {
                    std.debug.print("  Error reading string column {d}: {}\n", .{ orig_idx, err });
                    return;
                };
                defer {
                    for (values) |s| allocator.free(s);
                    allocator.free(values);
                }

                try encoder.writeStrings(values, &offsets_buf, allocator);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = offsets_buf.items,
                });
            },
            else => {},
        }
    }

    try finalizeLanceFile(&lance_writer, output_path);
}

/// Infer Lance type from XLSX cell values in a column
fn inferXlsxColumnType(reader: *const xlsx.XlsxReader, col: usize) writer.DataType {
    // Check first few non-header rows to infer type
    var has_number = false;
    var has_string = false;

    const start_row: usize = 1; // Skip header row
    const end_row = @min(start_row + 10, reader.rowCount());

    for (start_row..end_row) |row| {
        if (reader.getCell(row, col)) |cell| {
            switch (cell) {
                .number => has_number = true,
                .inline_string, .string, .shared_string => has_string = true,
                else => {},
            }
        }
    }

    if (has_string) return .string;
    if (has_number) return .float64; // XLSX numbers are always f64
    return .string; // Default fallback
}

/// Get cell value as string (for string columns)
fn xlsxCellToString(cell: xlsx.CellValue) ?[]const u8 {
    return switch (cell) {
        .inline_string => |s| s,
        .string => |s| s,
        else => null,
    };
}

/// Ingest XLSX data to Lance file
pub fn ingestXlsx(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
) !void {
    std.debug.print("Parsing XLSX file...\n", .{});

    var reader = xlsx.XlsxReader.init(allocator, data) catch |err| {
        std.debug.print("Error parsing XLSX: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const total_rows = reader.rowCount();
    const num_cols = reader.columnCount();

    if (total_rows == 0 or num_cols == 0) {
        std.debug.print("  No data to convert\n", .{});
        return;
    }

    // First row is header, data rows follow
    const num_data_rows = total_rows - 1;
    std.debug.print("  Rows: {d} (including header)\n", .{total_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});

    if (num_data_rows == 0) {
        std.debug.print("  No data rows (header only)\n", .{});
        return;
    }

    // Build schema from header row and inferred types
    var schema = try allocator.alloc(writer.ColumnSchema, num_cols);
    defer allocator.free(schema);

    var col_names = try allocator.alloc([]const u8, num_cols);
    defer {
        for (col_names) |name| allocator.free(name);
        allocator.free(col_names);
    }

    for (0..num_cols) |col| {
        // Get column name from header
        const header_cell = reader.getCell(0, col);
        const name = if (header_cell) |cell| blk: {
            const cell_name = xlsxCellToString(cell);
            break :blk if (cell_name) |n| try allocator.dupe(u8, n) else try std.fmt.allocPrint(allocator, "col_{d}", .{col});
        } else try std.fmt.allocPrint(allocator, "col_{d}", .{col});
        col_names[col] = name;

        // Infer type from data
        const lance_type = inferXlsxColumnType(&reader, col);
        schema[col] = .{
            .name = name,
            .data_type = lance_type,
        };
        std.debug.print("    - {s}: {s}\n", .{name, @tagName(lance_type)});
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    // Encode each column
    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    std.debug.print("Converting columns...\n", .{});

    for (0..num_cols) |col| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        const lance_type = schema[col].data_type;

        switch (lance_type) {
            .float64 => {
                // Read all number values
                var values = try allocator.alloc(f64, num_data_rows);
                defer allocator.free(values);

                for (0..num_data_rows) |i| {
                    const row = i + 1; // Skip header
                    if (reader.getCell(row, col)) |cell| {
                        values[i] = switch (cell) {
                            .number => |n| n,
                            else => 0.0,
                        };
                    } else {
                        values[i] = 0.0;
                    }
                }

                try encoder.writeFloat64Slice(values);
                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(col),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .string => {
                // Read all string values
                var values = try allocator.alloc([]const u8, num_data_rows);
                defer allocator.free(values);

                for (0..num_data_rows) |i| {
                    const row = i + 1; // Skip header
                    if (reader.getCell(row, col)) |cell| {
                        values[i] = xlsxCellToString(cell) orelse "";
                    } else {
                        values[i] = "";
                    }
                }

                try encoder.writeStrings(values, &offsets_buf, allocator);
                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(col),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = offsets_buf.items,
                });
            },
            else => {},
        }
    }

    try finalizeLanceFile(&lance_writer, output_path);
}

/// Ingest Delta Lake table to Lance file
pub fn ingestDelta(
    allocator: std.mem.Allocator,
    path: []const u8,
    output_path: []const u8,
) !void {
    _ = output_path;
    std.debug.print("Parsing Delta Lake table: {s}\n", .{path});

    var reader = delta.DeltaReader.init(allocator, path) catch |err| {
        std.debug.print("Error parsing Delta: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const num_cols = reader.columnCount();
    const num_rows = reader.rowCount();
    const num_files = reader.fileCount();

    std.debug.print("  Rows: {d}\n", .{num_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});
    std.debug.print("  Data files: {d}\n", .{num_files});

    std.debug.print("\nError: Delta Lake ingest not yet implemented.\n", .{});
    std.debug.print("Workaround: Use 'lanceql ingest <delta_table>/*.parquet -o output.lance'\n", .{});
}

/// Ingest Iceberg table to Lance file
pub fn ingestIceberg(
    allocator: std.mem.Allocator,
    path: []const u8,
    output_path: []const u8,
) !void {
    _ = output_path;
    std.debug.print("Parsing Iceberg table: {s}\n", .{path});

    var reader = iceberg.IcebergReader.init(allocator, path) catch |err| {
        std.debug.print("Error parsing Iceberg: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const num_cols = reader.columnCount();
    const format_version = reader.formatVersion();
    const snapshot_id = reader.snapshotId();

    std.debug.print("  Format version: {d}\n", .{format_version});
    std.debug.print("  Columns: {d}\n", .{num_cols});
    std.debug.print("  Snapshot ID: {d}\n", .{snapshot_id});

    std.debug.print("\nError: Iceberg ingest not yet implemented.\n", .{});
    std.debug.print("Workaround: Use 'lanceql ingest <iceberg_table>/data/*.parquet -o output.lance'\n", .{});
}

test "detect format by extension" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(args.IngestOptions.Format.csv, detectFormat(allocator, "data.csv"));
    try std.testing.expectEqual(args.IngestOptions.Format.tsv, detectFormat(allocator, "data.tsv"));
    try std.testing.expectEqual(args.IngestOptions.Format.json, detectFormat(allocator, "data.json"));
    try std.testing.expectEqual(args.IngestOptions.Format.jsonl, detectFormat(allocator, "data.jsonl"));
    try std.testing.expectEqual(args.IngestOptions.Format.parquet, detectFormat(allocator, "data.parquet"));
    try std.testing.expectEqual(args.IngestOptions.Format.arrow, detectFormat(allocator, "data.arrow"));
    try std.testing.expectEqual(args.IngestOptions.Format.arrow, detectFormat(allocator, "data.feather"));
    try std.testing.expectEqual(args.IngestOptions.Format.avro, detectFormat(allocator, "data.avro"));
    try std.testing.expectEqual(args.IngestOptions.Format.orc, detectFormat(allocator, "data.orc"));
    try std.testing.expectEqual(args.IngestOptions.Format.xlsx, detectFormat(allocator, "data.xlsx"));
}

test "detect format from fixtures" {
    const allocator = std.testing.allocator;

    // Delta Lake directory
    try std.testing.expectEqual(args.IngestOptions.Format.delta, detectFormat(allocator, "tests/fixtures/simple.delta"));

    // Iceberg directory
    try std.testing.expectEqual(args.IngestOptions.Format.iceberg, detectFormat(allocator, "tests/fixtures/simple.iceberg"));
}
