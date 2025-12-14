const std = @import("std");
const Server = @import("server.zig").Server;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line args
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var port: u16 = 3306;
    var bind_address: []const u8 = "127.0.0.1";
    var data_dir: []const u8 = "./data";

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i < args.len) {
                port = std.fmt.parseInt(u16, args[i], 10) catch {
                    std.log.err("Invalid port number: {s}", .{args[i]});
                    return error.InvalidArgument;
                };
            }
        } else if (std.mem.eql(u8, arg, "-b") or std.mem.eql(u8, arg, "--bind")) {
            i += 1;
            if (i < args.len) {
                bind_address = args[i];
            }
        } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--data")) {
            i += 1;
            if (i < args.len) {
                data_dir = args[i];
            }
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            printHelp();
            return;
        }
    }

    // Create server
    var server = Server.init(allocator, .{
        .port = port,
        .bind_address = bind_address,
        .data_dir = data_dir,
    });
    defer server.deinit();

    // Add default test user
    try server.addUser("root", "", null); // Empty password for testing
    try server.addUser("test", "test", "testdb");
    try server.addUser("agent", "agent", null);

    std.log.info("LanceQL MySQL-compatible server", .{});
    std.log.info("  Port: {d}", .{port});
    std.log.info("  Bind: {s}", .{bind_address});
    std.log.info("  Data: {s}", .{data_dir});
    std.log.info("", .{});
    std.log.info("Default users:", .{});
    std.log.info("  root (no password)", .{});
    std.log.info("  test/test (database: testdb)", .{});
    std.log.info("  agent/agent", .{});
    std.log.info("", .{});
    std.log.info("Connect with: mysql -h {s} -P {d} -u root", .{ bind_address, port });

    // Start server
    try server.start();
    try server.run();
}

fn printHelp() void {
    const help =
        \\LanceQL - MySQL-compatible server for LanceDB
        \\
        \\Usage: lanceql-server [OPTIONS]
        \\
        \\Options:
        \\  -p, --port PORT     Port to listen on (default: 3306)
        \\  -b, --bind ADDR     Address to bind to (default: 127.0.0.1)
        \\  -d, --data DIR      Data directory (default: ./data)
        \\  -h, --help          Show this help
        \\
        \\Examples:
        \\  lanceql-server
        \\  lanceql-server -p 3307
        \\  lanceql-server -b 0.0.0.0 -p 3306
        \\
        \\Connect with:
        \\  mysql -h localhost -P 3306 -u root
        \\
    ;
    std.debug.print("{s}", .{help});
}
