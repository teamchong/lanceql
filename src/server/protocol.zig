const std = @import("std");
const packets = @import("packets.zig");

/// Protocol handler for MySQL wire protocol
pub const Protocol = struct {
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    sequence_id: u8,

    const Self = @This();
    const max_packet_size: usize = 16 * 1024 * 1024; // 16MB

    pub fn init(allocator: std.mem.Allocator, stream: std.net.Stream) Self {
        return .{
            .allocator = allocator,
            .stream = stream,
            .sequence_id = 0,
        };
    }

    /// Reset sequence ID (for new command)
    pub fn resetSequence(self: *Self) void {
        self.sequence_id = 0;
    }

    /// Read a complete MySQL packet
    pub fn readPacket(self: *Self) ![]u8 {
        // Read 4-byte header
        var header_buf: [4]u8 = undefined;
        const header_read = try self.stream.read(&header_buf);
        if (header_read != 4) {
            return error.ConnectionClosed;
        }

        const header = packets.PacketHeader.decode(header_buf);

        // Validate sequence
        if (header.sequence_id != self.sequence_id) {
            return error.SequenceMismatch;
        }
        self.sequence_id +%= 1;

        // Validate size
        if (header.payload_length > max_packet_size) {
            return error.PacketTooLarge;
        }

        // Read payload
        const payload = try self.allocator.alloc(u8, header.payload_length);
        errdefer self.allocator.free(payload);

        var total_read: usize = 0;
        while (total_read < header.payload_length) {
            const n = try self.stream.read(payload[total_read..]);
            if (n == 0) return error.ConnectionClosed;
            total_read += n;
        }

        return payload;
    }

    /// Write a MySQL packet
    pub fn writePacket(self: *Self, payload: []const u8) !void {
        // Write header
        const header = packets.PacketHeader{
            .payload_length = @intCast(payload.len),
            .sequence_id = self.sequence_id,
        };
        self.sequence_id +%= 1;

        _ = try self.stream.write(&header.encode());
        _ = try self.stream.write(payload);
    }

    /// Write an OK packet
    pub fn writeOk(self: *Self, affected_rows: u64, last_insert_id: u64) !void {
        const ok = packets.OkPacket{
            .affected_rows = affected_rows,
            .last_insert_id = last_insert_id,
        };
        const payload = try ok.encode(self.allocator);
        defer self.allocator.free(payload);
        try self.writePacket(payload);
    }

    /// Write an ERR packet
    pub fn writeError(self: *Self, code: u16, message: []const u8) !void {
        const err = packets.ErrPacket{
            .error_code = code,
            .error_message = message,
        };
        const payload = try err.encode(self.allocator);
        defer self.allocator.free(payload);
        try self.writePacket(payload);
    }

    /// Write an EOF packet
    pub fn writeEof(self: *Self) !void {
        const eof = packets.EofPacket{};
        const payload = try eof.encode(self.allocator);
        defer self.allocator.free(payload);
        try self.writePacket(payload);
    }

    /// Write column count for result set
    pub fn writeColumnCount(self: *Self, count: u64) !void {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(self.allocator);
        try packets.encodeLengthEncodedInt(&buf, self.allocator, count);
        try self.writePacket(buf.items);
    }

    /// Write a column definition
    pub fn writeColumnDef(self: *Self, col: packets.ColumnDefinition) !void {
        const payload = try col.encode(self.allocator);
        defer self.allocator.free(payload);
        try self.writePacket(payload);
    }

    /// Write a text result row
    pub fn writeRow(self: *Self, values: []const ?[]const u8) !void {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(self.allocator);

        for (values) |value| {
            if (value) |v| {
                try packets.encodeLengthEncodedString(&buf, self.allocator, v);
            } else {
                try packets.encodeNull(&buf, self.allocator);
            }
        }

        try self.writePacket(buf.items);
    }

    /// Send complete result set
    pub fn sendResultSet(self: *Self, columns: []const packets.ColumnDefinition, rows: []const []const ?[]const u8) !void {
        // Column count
        try self.writeColumnCount(columns.len);

        // Column definitions
        for (columns) |col| {
            try self.writeColumnDef(col);
        }

        // EOF after columns
        try self.writeEof();

        // Rows
        for (rows) |row| {
            try self.writeRow(row);
        }

        // EOF after rows
        try self.writeEof();
    }

    /// Send empty result set (for queries with no results)
    pub fn sendEmptyResultSet(self: *Self, columns: []const packets.ColumnDefinition) !void {
        // Column count
        try self.writeColumnCount(columns.len);

        // Column definitions
        for (columns) |col| {
            try self.writeColumnDef(col);
        }

        // EOF after columns
        try self.writeEof();

        // EOF after rows (no rows)
        try self.writeEof();
    }
};

/// Result set builder helper
pub const ResultSetBuilder = struct {
    allocator: std.mem.Allocator,
    columns: std.ArrayListUnmanaged(packets.ColumnDefinition),
    rows: std.ArrayListUnmanaged([]?[]const u8),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .columns = .empty,
            .rows = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.rows.items) |row| {
            for (row) |cell| {
                if (cell) |c| {
                    self.allocator.free(c);
                }
            }
            self.allocator.free(row);
        }
        self.rows.deinit(self.allocator);
        self.columns.deinit(self.allocator);
    }

    pub fn addColumn(self: *Self, name: []const u8, col_type: packets.ColumnType) !void {
        try self.columns.append(self.allocator, .{
            .name = name,
            .column_type = col_type,
        });
    }

    pub fn addRow(self: *Self, values: []const ?[]const u8) !void {
        const row = try self.allocator.alloc(?[]const u8, values.len);
        errdefer self.allocator.free(row);

        for (values, 0..) |v, i| {
            if (v) |val| {
                row[i] = try self.allocator.dupe(u8, val);
            } else {
                row[i] = null;
            }
        }

        try self.rows.append(self.allocator, row);
    }

    pub fn send(self: *Self, protocol: *Protocol) !void {
        try protocol.sendResultSet(self.columns.items, self.rows.items);
    }
};
