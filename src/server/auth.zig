const std = @import("std");

/// mysql_native_password authentication
/// Based on: https://dev.mysql.com/doc/dev/mysql-server/latest/page_protocol_connection_phase_authentication_methods_native_password_authentication.html
pub const NativePassword = struct {
    /// Generate 20-byte scramble for handshake
    /// Note: MySQL requires bytes to avoid NUL (0x00)
    pub fn generateScramble(random: std.Random) [20]u8 {
        var scramble: [20]u8 = undefined;
        random.bytes(&scramble);

        // Ensure no NUL bytes (which would terminate strings early)
        for (&scramble) |*b| {
            if (b.* == 0) {
                b.* = random.intRangeAtMost(u8, 1, 255);
            }
        }

        return scramble;
    }

    /// Compute password hash for storage: SHA1(SHA1(password))
    pub fn hashPassword(password: []const u8) [20]u8 {
        const stage1 = sha1Hash(password);
        return sha1Hash(&stage1);
    }

    /// Verify client auth response against stored password hash
    /// Client sends: SHA1(password) XOR SHA1(scramble + SHA1(SHA1(password)))
    /// We have: scramble, stored_hash = SHA1(SHA1(password)), client_response
    pub fn verify(scramble: [20]u8, client_response: []const u8, stored_hash: [20]u8) bool {
        if (client_response.len != 20) return false;

        // Compute SHA1(scramble + stored_hash)
        var hasher = std.crypto.hash.Sha1.init(.{});
        hasher.update(&scramble);
        hasher.update(&stored_hash);
        const stage2 = hasher.finalResult();

        // XOR with client response to recover SHA1(password)
        var recovered_stage1: [20]u8 = undefined;
        for (0..20) |i| {
            recovered_stage1[i] = client_response[i] ^ stage2[i];
        }

        // Verify: SHA1(recovered_stage1) == stored_hash
        const check = sha1Hash(&recovered_stage1);
        return std.mem.eql(u8, &check, &stored_hash);
    }

    /// Compute client auth response (for testing)
    pub fn computeAuthResponse(scramble: [20]u8, password: []const u8) [20]u8 {
        // stage1 = SHA1(password)
        const stage1 = sha1Hash(password);
        // stage2 = SHA1(stage1) = SHA1(SHA1(password))
        const stage2 = sha1Hash(&stage1);

        // SHA1(scramble + stage2)
        var hasher = std.crypto.hash.Sha1.init(.{});
        hasher.update(&scramble);
        hasher.update(&stage2);
        const scramble_hash = hasher.finalResult();

        // result = stage1 XOR SHA1(scramble + stage2)
        var result: [20]u8 = undefined;
        for (0..20) |i| {
            result[i] = stage1[i] ^ scramble_hash[i];
        }

        return result;
    }

    fn sha1Hash(data: []const u8) [20]u8 {
        var hasher = std.crypto.hash.Sha1.init(.{});
        hasher.update(data);
        return hasher.finalResult();
    }
};

/// Simple in-memory user store
pub const UserStore = struct {
    allocator: std.mem.Allocator,
    users: std.StringHashMapUnmanaged(User),

    pub const User = struct {
        password_hash: [20]u8,
        default_database: ?[]const u8,
    };

    pub fn init(allocator: std.mem.Allocator) UserStore {
        return .{
            .allocator = allocator,
            .users = .empty,
        };
    }

    pub fn deinit(self: *UserStore) void {
        var iter = self.users.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            if (entry.value_ptr.default_database) |db| {
                self.allocator.free(db);
            }
        }
        self.users.deinit(self.allocator);
    }

    /// Add a user with plaintext password (hashes it)
    pub fn addUser(self: *UserStore, username: []const u8, password: []const u8, default_database: ?[]const u8) !void {
        const username_copy = try self.allocator.dupe(u8, username);
        errdefer self.allocator.free(username_copy);

        var db_copy: ?[]const u8 = null;
        if (default_database) |db| {
            db_copy = try self.allocator.dupe(u8, db);
        }

        try self.users.put(self.allocator, username_copy, .{
            .password_hash = NativePassword.hashPassword(password),
            .default_database = db_copy,
        });
    }

    /// Add a user with pre-hashed password
    pub fn addUserWithHash(self: *UserStore, username: []const u8, password_hash: [20]u8, default_database: ?[]const u8) !void {
        const username_copy = try self.allocator.dupe(u8, username);
        errdefer self.allocator.free(username_copy);

        var db_copy: ?[]const u8 = null;
        if (default_database) |db| {
            db_copy = try self.allocator.dupe(u8, db);
        }

        try self.users.put(self.allocator, username_copy, .{
            .password_hash = password_hash,
            .default_database = db_copy,
        });
    }

    /// Authenticate a user
    pub fn authenticate(self: *UserStore, username: []const u8, scramble: [20]u8, auth_response: []const u8) AuthResult {
        const user = self.users.get(username) orelse {
            return .{ .success = false, .error_message = "Access denied for user" };
        };

        // Empty password case: auth_response should be empty
        if (auth_response.len == 0) {
            // Check if stored password is also empty (all zeros)
            const empty_hash = NativePassword.hashPassword("");
            if (std.mem.eql(u8, &user.password_hash, &empty_hash)) {
                return .{ .success = true, .database = user.default_database };
            }
            return .{ .success = false, .error_message = "Access denied for user (empty password)" };
        }

        if (NativePassword.verify(scramble, auth_response, user.password_hash)) {
            return .{ .success = true, .database = user.default_database };
        }

        return .{ .success = false, .error_message = "Access denied for user (password mismatch)" };
    }

    pub const AuthResult = struct {
        success: bool,
        database: ?[]const u8 = null,
        error_message: ?[]const u8 = null,
    };
};

// Tests
test "mysql_native_password round trip" {
    const password = "secret123";
    const stored_hash = NativePassword.hashPassword(password);

    var prng = std.Random.DefaultPrng.init(42);
    const scramble = NativePassword.generateScramble(prng.random());

    const auth_response = NativePassword.computeAuthResponse(scramble, password);

    try std.testing.expect(NativePassword.verify(scramble, &auth_response, stored_hash));
}

test "mysql_native_password wrong password" {
    const stored_hash = NativePassword.hashPassword("correct");

    var prng = std.Random.DefaultPrng.init(42);
    const scramble = NativePassword.generateScramble(prng.random());

    const auth_response = NativePassword.computeAuthResponse(scramble, "wrong");

    try std.testing.expect(!NativePassword.verify(scramble, &auth_response, stored_hash));
}

test "user store authentication" {
    const allocator = std.testing.allocator;

    var store = UserStore.init(allocator);
    defer store.deinit();

    try store.addUser("testuser", "testpass", "testdb");

    var prng = std.Random.DefaultPrng.init(42);
    const scramble = NativePassword.generateScramble(prng.random());

    // Correct password
    const good_response = NativePassword.computeAuthResponse(scramble, "testpass");
    const good_result = store.authenticate("testuser", scramble, &good_response);
    try std.testing.expect(good_result.success);
    try std.testing.expectEqualStrings("testdb", good_result.database.?);

    // Wrong password
    const bad_response = NativePassword.computeAuthResponse(scramble, "wrongpass");
    const bad_result = store.authenticate("testuser", scramble, &bad_response);
    try std.testing.expect(!bad_result.success);

    // Unknown user
    const unknown_result = store.authenticate("nobody", scramble, &good_response);
    try std.testing.expect(!unknown_result.success);
}
