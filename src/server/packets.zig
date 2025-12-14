const std = @import("std");

/// MySQL packet header (4 bytes total)
pub const PacketHeader = struct {
    payload_length: u24,
    sequence_id: u8,

    pub fn encode(self: PacketHeader) [4]u8 {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u24, buf[0..3], self.payload_length, .little);
        buf[3] = self.sequence_id;
        return buf;
    }

    pub fn decode(bytes: [4]u8) PacketHeader {
        return .{
            .payload_length = std.mem.readInt(u24, bytes[0..3], .little),
            .sequence_id = bytes[3],
        };
    }
};

/// Server capabilities flags
pub const Capabilities = struct {
    pub const CLIENT_LONG_PASSWORD: u32 = 1;
    pub const CLIENT_FOUND_ROWS: u32 = 1 << 1;
    pub const CLIENT_LONG_FLAG: u32 = 1 << 2;
    pub const CLIENT_CONNECT_WITH_DB: u32 = 1 << 3;
    pub const CLIENT_NO_SCHEMA: u32 = 1 << 4;
    pub const CLIENT_COMPRESS: u32 = 1 << 5;
    pub const CLIENT_ODBC: u32 = 1 << 6;
    pub const CLIENT_LOCAL_FILES: u32 = 1 << 7;
    pub const CLIENT_IGNORE_SPACE: u32 = 1 << 8;
    pub const CLIENT_PROTOCOL_41: u32 = 1 << 9;
    pub const CLIENT_INTERACTIVE: u32 = 1 << 10;
    pub const CLIENT_SSL: u32 = 1 << 11;
    pub const CLIENT_IGNORE_SIGPIPE: u32 = 1 << 12;
    pub const CLIENT_TRANSACTIONS: u32 = 1 << 13;
    pub const CLIENT_RESERVED: u32 = 1 << 14;
    pub const CLIENT_SECURE_CONNECTION: u32 = 1 << 15;
    pub const CLIENT_MULTI_STATEMENTS: u32 = 1 << 16;
    pub const CLIENT_MULTI_RESULTS: u32 = 1 << 17;
    pub const CLIENT_PS_MULTI_RESULTS: u32 = 1 << 18;
    pub const CLIENT_PLUGIN_AUTH: u32 = 1 << 19;
    pub const CLIENT_CONNECT_ATTRS: u32 = 1 << 20;
    pub const CLIENT_PLUGIN_AUTH_LENENC_CLIENT_DATA: u32 = 1 << 21;
    pub const CLIENT_DEPRECATE_EOF: u32 = 1 << 24;

    /// Default server capabilities
    pub const DEFAULT_SERVER: u32 =
        CLIENT_LONG_PASSWORD |
        CLIENT_FOUND_ROWS |
        CLIENT_LONG_FLAG |
        CLIENT_CONNECT_WITH_DB |
        CLIENT_PROTOCOL_41 |
        CLIENT_TRANSACTIONS |
        CLIENT_SECURE_CONNECTION |
        CLIENT_PLUGIN_AUTH;
};

/// Server status flags
pub const ServerStatus = struct {
    pub const STATUS_IN_TRANS: u16 = 1;
    pub const STATUS_AUTOCOMMIT: u16 = 1 << 1;
    pub const MORE_RESULTS_EXISTS: u16 = 1 << 3;
    pub const STATUS_NO_GOOD_INDEX_USED: u16 = 1 << 4;
    pub const STATUS_NO_INDEX_USED: u16 = 1 << 5;
    pub const STATUS_CURSOR_EXISTS: u16 = 1 << 6;
    pub const STATUS_LAST_ROW_SENT: u16 = 1 << 7;
    pub const STATUS_DB_DROPPED: u16 = 1 << 8;
    pub const STATUS_NO_BACKSLASH_ESCAPES: u16 = 1 << 9;
    pub const STATUS_METADATA_CHANGED: u16 = 1 << 10;
    pub const QUERY_WAS_SLOW: u16 = 1 << 11;
    pub const PS_OUT_PARAMS: u16 = 1 << 12;
    pub const STATUS_IN_TRANS_READONLY: u16 = 1 << 13;
    pub const SESSION_STATE_CHANGED: u16 = 1 << 14;
};

/// Command types (COM_*)
pub const Command = enum(u8) {
    COM_SLEEP = 0x00,
    COM_QUIT = 0x01,
    COM_INIT_DB = 0x02,
    COM_QUERY = 0x03,
    COM_FIELD_LIST = 0x04,
    COM_CREATE_DB = 0x05,
    COM_DROP_DB = 0x06,
    COM_REFRESH = 0x07,
    COM_SHUTDOWN = 0x08,
    COM_STATISTICS = 0x09,
    COM_PROCESS_INFO = 0x0a,
    COM_CONNECT = 0x0b,
    COM_PROCESS_KILL = 0x0c,
    COM_DEBUG = 0x0d,
    COM_PING = 0x0e,
    COM_TIME = 0x0f,
    COM_DELAYED_INSERT = 0x10,
    COM_CHANGE_USER = 0x11,
    COM_BINLOG_DUMP = 0x12,
    COM_TABLE_DUMP = 0x13,
    COM_CONNECT_OUT = 0x14,
    COM_REGISTER_SLAVE = 0x15,
    COM_STMT_PREPARE = 0x16,
    COM_STMT_EXECUTE = 0x17,
    COM_STMT_SEND_LONG_DATA = 0x18,
    COM_STMT_CLOSE = 0x19,
    COM_STMT_RESET = 0x1a,
    COM_SET_OPTION = 0x1b,
    COM_STMT_FETCH = 0x1c,
    COM_DAEMON = 0x1d,
    COM_BINLOG_DUMP_GTID = 0x1e,
    COM_RESET_CONNECTION = 0x1f,
    _,
};

/// Column types
pub const ColumnType = enum(u8) {
    MYSQL_TYPE_DECIMAL = 0x00,
    MYSQL_TYPE_TINY = 0x01,
    MYSQL_TYPE_SHORT = 0x02,
    MYSQL_TYPE_LONG = 0x03,
    MYSQL_TYPE_FLOAT = 0x04,
    MYSQL_TYPE_DOUBLE = 0x05,
    MYSQL_TYPE_NULL = 0x06,
    MYSQL_TYPE_TIMESTAMP = 0x07,
    MYSQL_TYPE_LONGLONG = 0x08,
    MYSQL_TYPE_INT24 = 0x09,
    MYSQL_TYPE_DATE = 0x0a,
    MYSQL_TYPE_TIME = 0x0b,
    MYSQL_TYPE_DATETIME = 0x0c,
    MYSQL_TYPE_YEAR = 0x0d,
    MYSQL_TYPE_NEWDATE = 0x0e,
    MYSQL_TYPE_VARCHAR = 0x0f,
    MYSQL_TYPE_BIT = 0x10,
    MYSQL_TYPE_TIMESTAMP2 = 0x11,
    MYSQL_TYPE_DATETIME2 = 0x12,
    MYSQL_TYPE_TIME2 = 0x13,
    MYSQL_TYPE_NEWDECIMAL = 0xf6,
    MYSQL_TYPE_ENUM = 0xf7,
    MYSQL_TYPE_SET = 0xf8,
    MYSQL_TYPE_TINY_BLOB = 0xf9,
    MYSQL_TYPE_MEDIUM_BLOB = 0xfa,
    MYSQL_TYPE_LONG_BLOB = 0xfb,
    MYSQL_TYPE_BLOB = 0xfc,
    MYSQL_TYPE_VAR_STRING = 0xfd,
    MYSQL_TYPE_STRING = 0xfe,
    MYSQL_TYPE_GEOMETRY = 0xff,
    _,
};

/// Handshake V10 packet (server → client)
pub const HandshakeV10 = struct {
    protocol_version: u8 = 10,
    server_version: []const u8,
    connection_id: u32,
    auth_plugin_data_part1: [8]u8,
    capability_flags: u32,
    character_set: u8 = 0x21, // utf8_general_ci
    status_flags: u16 = ServerStatus.STATUS_AUTOCOMMIT,
    auth_plugin_data_part2: [12]u8,
    auth_plugin_name: []const u8 = "mysql_native_password",

    pub fn encode(self: HandshakeV10, allocator: std.mem.Allocator) ![]u8 {
        var list: std.ArrayListUnmanaged(u8) = .empty;
        errdefer list.deinit(allocator);

        // Protocol version
        try list.append(allocator, self.protocol_version);

        // Server version (NUL-terminated)
        try list.appendSlice(allocator, self.server_version);
        try list.append(allocator, 0);

        // Connection ID (4 bytes, little-endian)
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u32, self.connection_id)));

        // Auth plugin data part 1 (8 bytes)
        try list.appendSlice(allocator, &self.auth_plugin_data_part1);

        // Filler (1 byte)
        try list.append(allocator, 0x00);

        // Capability flags lower 2 bytes
        const cap_lower: u16 = @truncate(self.capability_flags);
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, cap_lower)));

        // Character set
        try list.append(allocator, self.character_set);

        // Status flags
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, self.status_flags)));

        // Capability flags upper 2 bytes
        const cap_upper: u16 = @truncate(self.capability_flags >> 16);
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, cap_upper)));

        // Auth plugin data length (21 = 8 + 12 + 1 for NUL)
        try list.append(allocator, 21);

        // Reserved (10 bytes of zeros)
        try list.appendSlice(allocator, &[_]u8{0} ** 10);

        // Auth plugin data part 2 (12 bytes + NUL)
        try list.appendSlice(allocator, &self.auth_plugin_data_part2);
        try list.append(allocator, 0);

        // Auth plugin name (NUL-terminated)
        try list.appendSlice(allocator, self.auth_plugin_name);
        try list.append(allocator, 0);

        return list.toOwnedSlice(allocator);
    }
};

/// Handshake Response 41 (client → server)
pub const HandshakeResponse41 = struct {
    client_flags: u32,
    max_packet_size: u32,
    character_set: u8,
    username: []const u8,
    auth_response: []const u8,
    database: ?[]const u8,
    auth_plugin_name: ?[]const u8,

    pub fn decode(data: []const u8) !HandshakeResponse41 {
        if (data.len < 32) return error.PacketTooShort;

        var pos: usize = 0;

        // Client flags (4 bytes)
        const client_flags = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;

        // Max packet size (4 bytes)
        const max_packet_size = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;

        // Character set (1 byte)
        const character_set = data[pos];
        pos += 1;

        // Skip 23 bytes of filler
        pos += 23;

        // Username (NUL-terminated)
        const username_end = std.mem.indexOfScalar(u8, data[pos..], 0) orelse return error.InvalidPacket;
        const username = data[pos .. pos + username_end];
        pos += username_end + 1;

        // Auth response (length-prefixed if CLIENT_PLUGIN_AUTH_LENENC_CLIENT_DATA,
        // otherwise 1-byte length prefix or NUL-terminated)
        var auth_response: []const u8 = &.{};
        if (pos < data.len) {
            if (client_flags & Capabilities.CLIENT_SECURE_CONNECTION != 0) {
                const auth_len = data[pos];
                pos += 1;
                if (pos + auth_len <= data.len) {
                    auth_response = data[pos .. pos + auth_len];
                    pos += auth_len;
                }
            }
        }

        // Database (NUL-terminated, if CLIENT_CONNECT_WITH_DB)
        var database: ?[]const u8 = null;
        if (client_flags & Capabilities.CLIENT_CONNECT_WITH_DB != 0 and pos < data.len) {
            const db_end = std.mem.indexOfScalar(u8, data[pos..], 0) orelse data.len - pos;
            if (db_end > 0) {
                database = data[pos .. pos + db_end];
            }
            pos += db_end + 1;
        }

        // Auth plugin name (NUL-terminated, if CLIENT_PLUGIN_AUTH)
        var auth_plugin_name: ?[]const u8 = null;
        if (client_flags & Capabilities.CLIENT_PLUGIN_AUTH != 0 and pos < data.len) {
            const plugin_end = std.mem.indexOfScalar(u8, data[pos..], 0) orelse data.len - pos;
            if (plugin_end > 0) {
                auth_plugin_name = data[pos .. pos + plugin_end];
            }
        }

        return HandshakeResponse41{
            .client_flags = client_flags,
            .max_packet_size = max_packet_size,
            .character_set = character_set,
            .username = username,
            .auth_response = auth_response,
            .database = database,
            .auth_plugin_name = auth_plugin_name,
        };
    }
};

/// OK Packet (server → client)
pub const OkPacket = struct {
    affected_rows: u64 = 0,
    last_insert_id: u64 = 0,
    status_flags: u16 = ServerStatus.STATUS_AUTOCOMMIT,
    warnings: u16 = 0,
    info: ?[]const u8 = null,

    pub fn encode(self: OkPacket, allocator: std.mem.Allocator) ![]u8 {
        var list: std.ArrayListUnmanaged(u8) = .empty;
        errdefer list.deinit(allocator);

        // Header (0x00 for OK)
        try list.append(allocator, 0x00);

        // Affected rows (length-encoded int)
        try encodeLengthEncodedInt(&list, allocator, self.affected_rows);

        // Last insert ID (length-encoded int)
        try encodeLengthEncodedInt(&list, allocator, self.last_insert_id);

        // Status flags
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, self.status_flags)));

        // Warnings
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, self.warnings)));

        // Info (optional)
        if (self.info) |info| {
            try list.appendSlice(allocator, info);
        }

        return list.toOwnedSlice(allocator);
    }
};

/// ERR Packet (server → client)
pub const ErrPacket = struct {
    error_code: u16,
    sql_state: [5]u8 = "HY000".*,
    error_message: []const u8,

    pub fn encode(self: ErrPacket, allocator: std.mem.Allocator) ![]u8 {
        var list: std.ArrayListUnmanaged(u8) = .empty;
        errdefer list.deinit(allocator);

        // Header (0xFF for ERR)
        try list.append(allocator, 0xFF);

        // Error code
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, self.error_code)));

        // SQL state marker
        try list.append(allocator, '#');

        // SQL state
        try list.appendSlice(allocator, &self.sql_state);

        // Error message
        try list.appendSlice(allocator, self.error_message);

        return list.toOwnedSlice(allocator);
    }
};

/// EOF Packet (server → client)
pub const EofPacket = struct {
    warnings: u16 = 0,
    status_flags: u16 = ServerStatus.STATUS_AUTOCOMMIT,

    pub fn encode(self: EofPacket, allocator: std.mem.Allocator) ![]u8 {
        var list: std.ArrayListUnmanaged(u8) = .empty;
        errdefer list.deinit(allocator);

        // Header (0xFE for EOF)
        try list.append(allocator, 0xFE);

        // Warnings
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, self.warnings)));

        // Status flags
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, self.status_flags)));

        return list.toOwnedSlice(allocator);
    }
};

/// Column definition packet
pub const ColumnDefinition = struct {
    catalog: []const u8 = "def",
    schema: []const u8 = "",
    table: []const u8 = "",
    org_table: []const u8 = "",
    name: []const u8,
    org_name: []const u8 = "",
    character_set: u16 = 33, // utf8_general_ci
    column_length: u32 = 255,
    column_type: ColumnType = .MYSQL_TYPE_VAR_STRING,
    flags: u16 = 0,
    decimals: u8 = 0,

    pub fn encode(self: ColumnDefinition, allocator: std.mem.Allocator) ![]u8 {
        var list: std.ArrayListUnmanaged(u8) = .empty;
        errdefer list.deinit(allocator);

        // catalog
        try encodeLengthEncodedString(&list, allocator, self.catalog);
        // schema
        try encodeLengthEncodedString(&list, allocator, self.schema);
        // table (virtual)
        try encodeLengthEncodedString(&list, allocator, self.table);
        // org_table (physical)
        try encodeLengthEncodedString(&list, allocator, self.org_table);
        // name (virtual)
        try encodeLengthEncodedString(&list, allocator, self.name);
        // org_name (physical)
        const org_name = if (self.org_name.len > 0) self.org_name else self.name;
        try encodeLengthEncodedString(&list, allocator, org_name);

        // Fixed length fields (0x0C)
        try list.append(allocator, 0x0C);

        // Character set
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, self.character_set)));

        // Column length
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u32, self.column_length)));

        // Column type
        try list.append(allocator, @intFromEnum(self.column_type));

        // Flags
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, self.flags)));

        // Decimals
        try list.append(allocator, self.decimals);

        // Filler (2 bytes)
        try list.appendSlice(allocator, &[_]u8{ 0, 0 });

        return list.toOwnedSlice(allocator);
    }
};

/// Encode length-encoded integer
pub fn encodeLengthEncodedInt(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: u64) !void {
    if (value < 251) {
        try list.append(allocator, @truncate(value));
    } else if (value < 65536) {
        try list.append(allocator, 0xFC);
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, @truncate(value))));
    } else if (value < 16777216) {
        try list.append(allocator, 0xFD);
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u24, @truncate(value))));
    } else {
        try list.append(allocator, 0xFE);
        try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u64, value)));
    }
}

/// Encode length-encoded string
pub fn encodeLengthEncodedString(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, str: []const u8) !void {
    try encodeLengthEncodedInt(list, allocator, str.len);
    try list.appendSlice(allocator, str);
}

/// Encode a NULL value
pub fn encodeNull(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator) !void {
    try list.append(allocator, 0xFB);
}
