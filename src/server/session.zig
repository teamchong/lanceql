const std = @import("std");
const packets = @import("packets.zig");
const protocol = @import("protocol.zig");
const auth = @import("auth.zig");

/// Session state for a MySQL connection
pub const Session = struct {
    allocator: std.mem.Allocator,
    connection_id: u32,
    protocol: protocol.Protocol,
    username: []const u8,
    current_database: ?[]const u8,
    authenticated: bool,
    scramble: [20]u8,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        stream: std.net.Stream,
        connection_id: u32,
    ) Self {
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));

        return .{
            .allocator = allocator,
            .connection_id = connection_id,
            .protocol = protocol.Protocol.init(allocator, stream),
            .username = "",
            .current_database = null,
            .authenticated = false,
            .scramble = auth.NativePassword.generateScramble(prng.random()),
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.username.len > 0) {
            self.allocator.free(self.username);
        }
        if (self.current_database) |db| {
            self.allocator.free(db);
        }
    }

    /// Perform handshake with client
    pub fn handshake(self: *Self, user_store: *auth.UserStore) !void {
        // Send server handshake
        const handshake_packet = packets.HandshakeV10{
            .server_version = "LanceQL 1.0.0",
            .connection_id = self.connection_id,
            .auth_plugin_data_part1 = self.scramble[0..8].*,
            .auth_plugin_data_part2 = self.scramble[8..20].*,
            .capability_flags = packets.Capabilities.DEFAULT_SERVER,
        };

        const handshake_data = try handshake_packet.encode(self.allocator);
        defer self.allocator.free(handshake_data);
        try self.protocol.writePacket(handshake_data);

        // Read client response
        const response_data = try self.protocol.readPacket();
        defer self.allocator.free(response_data);

        const response = try packets.HandshakeResponse41.decode(response_data);

        // Authenticate
        const auth_result = user_store.authenticate(
            response.username,
            self.scramble,
            response.auth_response,
        );

        if (!auth_result.success) {
            try self.protocol.writeError(1045, auth_result.error_message orelse "Access denied");
            return error.AuthenticationFailed;
        }

        // Store session info
        self.username = try self.allocator.dupe(u8, response.username);
        self.authenticated = true;

        if (response.database) |db| {
            self.current_database = try self.allocator.dupe(u8, db);
        } else if (auth_result.database) |db| {
            self.current_database = try self.allocator.dupe(u8, db);
        }

        // Send OK
        try self.protocol.writeOk(0, 0);
    }

    /// Handle command phase
    pub fn run(self: *Self) !void {
        while (true) {
            self.protocol.resetSequence();

            const packet = self.protocol.readPacket() catch |err| {
                if (err == error.ConnectionClosed) {
                    return;
                }
                return err;
            };
            defer self.allocator.free(packet);

            if (packet.len == 0) continue;

            const cmd: packets.Command = @enumFromInt(packet[0]);
            const payload = packet[1..];

            switch (cmd) {
                .COM_QUIT => {
                    return;
                },
                .COM_PING => {
                    try self.protocol.writeOk(0, 0);
                },
                .COM_INIT_DB => {
                    try self.handleInitDb(payload);
                },
                .COM_QUERY => {
                    try self.handleQuery(payload);
                },
                .COM_FIELD_LIST => {
                    // Not implemented, send empty result
                    try self.protocol.writeEof();
                },
                else => {
                    try self.protocol.writeError(
                        1047,
                        "Unknown command",
                    );
                },
            }
        }
    }

    fn handleInitDb(self: *Self, db_name: []const u8) !void {
        if (self.current_database) |old| {
            self.allocator.free(old);
        }
        self.current_database = try self.allocator.dupe(u8, db_name);
        try self.protocol.writeOk(0, 0);
    }

    fn handleQuery(self: *Self, sql: []const u8) !void {
        // Trim whitespace
        const trimmed = std.mem.trim(u8, sql, " \t\n\r");

        // Handle special queries
        if (std.ascii.startsWithIgnoreCase(trimmed, "SELECT @@version_comment")) {
            try self.sendVersionComment();
            return;
        }

        if (std.ascii.startsWithIgnoreCase(trimmed, "SELECT @@")) {
            try self.sendSystemVariable(trimmed);
            return;
        }

        if (std.ascii.startsWithIgnoreCase(trimmed, "SHOW DATABASES")) {
            try self.sendDatabases();
            return;
        }

        if (std.ascii.startsWithIgnoreCase(trimmed, "SHOW TABLES")) {
            try self.sendTables();
            return;
        }

        if (std.ascii.startsWithIgnoreCase(trimmed, "SET ")) {
            // Acknowledge SET commands
            try self.protocol.writeOk(0, 0);
            return;
        }

        if (std.ascii.startsWithIgnoreCase(trimmed, "USE ")) {
            const db = std.mem.trim(u8, trimmed[4..], " \t;");
            try self.handleInitDb(db);
            return;
        }

        // Parse and execute SQL query
        try self.executeQuery(trimmed);
    }

    fn executeQuery(self: *Self, sql: []const u8) !void {
        // TODO: Connect to lanceql parser and executor
        // For now, return placeholder result for SELECT, error for others
        if (std.ascii.startsWithIgnoreCase(sql, "SELECT")) {
            // Return empty result set
            var builder = protocol.ResultSetBuilder.init(self.allocator);
            defer builder.deinit();
            try builder.addColumn("result", .MYSQL_TYPE_VAR_STRING);
            try builder.addRow(&[_]?[]const u8{"Query received but execution not yet implemented"});
            try builder.send(&self.protocol);
        } else {
            // Non-SELECT queries return OK with 0 affected rows
            try self.protocol.writeOk(0, 0);
        }
    }

    fn sendVersionComment(self: *Self) !void {
        var builder = protocol.ResultSetBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.addColumn("@@version_comment", .MYSQL_TYPE_VAR_STRING);
        try builder.addRow(&[_]?[]const u8{"LanceQL - SQLite SQL over MySQL Protocol"});
        try builder.send(&self.protocol);
    }

    fn sendSystemVariable(self: *Self, sql: []const u8) !void {
        // Extract variable name
        const var_start = std.mem.indexOf(u8, sql, "@@") orelse return self.protocol.writeError(1193, "Unknown system variable");
        var var_end = var_start + 2;
        while (var_end < sql.len and (std.ascii.isAlphanumeric(sql[var_end]) or sql[var_end] == '_')) {
            var_end += 1;
        }
        const var_name = sql[var_start..var_end];

        var builder = protocol.ResultSetBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.addColumn(var_name, .MYSQL_TYPE_VAR_STRING);

        // Return sensible defaults for common variables
        const value: []const u8 = if (std.mem.eql(u8, var_name, "@@version"))
            "8.0.0-LanceQL"
        else if (std.mem.eql(u8, var_name, "@@version_comment"))
            "LanceQL - SQLite SQL over MySQL Protocol"
        else if (std.mem.eql(u8, var_name, "@@max_allowed_packet"))
            "16777216"
        else if (std.mem.eql(u8, var_name, "@@character_set_client"))
            "utf8mb4"
        else if (std.mem.eql(u8, var_name, "@@character_set_connection"))
            "utf8mb4"
        else if (std.mem.eql(u8, var_name, "@@character_set_results"))
            "utf8mb4"
        else if (std.mem.eql(u8, var_name, "@@collation_connection"))
            "utf8mb4_general_ci"
        else if (std.mem.eql(u8, var_name, "@@autocommit"))
            "1"
        else if (std.mem.eql(u8, var_name, "@@session.autocommit"))
            "1"
        else if (std.mem.eql(u8, var_name, "@@sql_mode"))
            ""
        else
            "0";

        try builder.addRow(&[_]?[]const u8{value});
        try builder.send(&self.protocol);
    }

    fn sendDatabases(self: *Self) !void {
        var builder = protocol.ResultSetBuilder.init(self.allocator);
        defer builder.deinit();

        try builder.addColumn("Database", .MYSQL_TYPE_VAR_STRING);

        // List available databases (agent directories)
        // For now, just return current database if set
        if (self.current_database) |db| {
            try builder.addRow(&[_]?[]const u8{db});
        }
        try builder.addRow(&[_]?[]const u8{"information_schema"});

        try builder.send(&self.protocol);
    }

    fn sendTables(self: *Self) !void {
        var builder = protocol.ResultSetBuilder.init(self.allocator);
        defer builder.deinit();

        const col_name = if (self.current_database) |db|
            try std.fmt.allocPrint(self.allocator, "Tables_in_{s}", .{db})
        else
            try self.allocator.dupe(u8, "Tables");
        defer self.allocator.free(col_name);

        try builder.addColumn(col_name, .MYSQL_TYPE_VAR_STRING);

        // TODO: List actual .lance files in agent's directory

        try builder.send(&self.protocol);
    }
};
