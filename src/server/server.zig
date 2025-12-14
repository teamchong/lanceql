const std = @import("std");
const packets = @import("packets.zig");
const protocol = @import("protocol.zig");
const auth = @import("auth.zig");
const session = @import("session.zig");

/// MySQL-compatible server configuration
pub const Config = struct {
    port: u16 = 3306,
    bind_address: []const u8 = "127.0.0.1",
    max_connections: u32 = 100,
    data_dir: []const u8 = "./data",
};

/// MySQL-compatible server
pub const Server = struct {
    allocator: std.mem.Allocator,
    config: Config,
    user_store: auth.UserStore,
    listener: ?std.net.Server,
    next_connection_id: u32,
    running: bool,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: Config) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .user_store = auth.UserStore.init(allocator),
            .listener = null,
            .next_connection_id = 1,
            .running = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.stop();
        self.user_store.deinit();
    }

    /// Add a user to the server
    pub fn addUser(self: *Self, username: []const u8, password: []const u8, database: ?[]const u8) !void {
        try self.user_store.addUser(username, password, database);
    }

    /// Start the server
    pub fn start(self: *Self) !void {
        const address = try std.net.Address.parseIp(self.config.bind_address, self.config.port);
        self.listener = try address.listen(.{
            .reuse_address = true,
        });
        self.running = true;

        std.log.info("LanceQL server listening on {s}:{d}", .{ self.config.bind_address, self.config.port });
    }

    /// Stop the server
    pub fn stop(self: *Self) void {
        self.running = false;
        if (self.listener) |*l| {
            l.deinit();
            self.listener = null;
        }
    }

    /// Accept and handle connections (blocking)
    pub fn run(self: *Self) !void {
        if (self.listener == null) {
            try self.start();
        }

        while (self.running) {
            const conn = self.listener.?.accept() catch |err| {
                if (!self.running) return;
                std.log.err("Accept error: {}", .{err});
                continue;
            };

            // Handle connection in a thread
            const thread = std.Thread.spawn(.{}, handleConnection, .{ self, conn.stream }) catch |err| {
                std.log.err("Failed to spawn thread: {}", .{err});
                conn.stream.close();
                continue;
            };
            thread.detach();
        }
    }

    /// Handle a single connection
    fn handleConnection(self: *Self, stream: std.net.Stream) void {
        const conn_id = @atomicRmw(u32, &self.next_connection_id, .Add, 1, .seq_cst);

        std.log.info("Connection {d} accepted", .{conn_id});

        var sess = session.Session.init(self.allocator, stream, conn_id);
        defer sess.deinit();
        defer stream.close();

        // Handshake
        sess.handshake(&self.user_store) catch |err| {
            std.log.warn("Connection {d} handshake failed: {}", .{ conn_id, err });
            return;
        };

        std.log.info("Connection {d} authenticated as '{s}'", .{ conn_id, sess.username });

        // Command loop
        sess.run() catch |err| {
            std.log.warn("Connection {d} error: {}", .{ conn_id, err });
        };

        std.log.info("Connection {d} closed", .{conn_id});
    }
};

// Re-export types
pub const Session = session.Session;
pub const Protocol = protocol.Protocol;
pub const UserStore = auth.UserStore;
pub const NativePassword = auth.NativePassword;
pub const Packets = packets;
