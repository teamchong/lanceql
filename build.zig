const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // === Core Modules ===
    const proto_mod = b.addModule("lanceql.proto", .{
        .root_source_file = b.path("src/proto/proto.zig"),
    });

    const io_mod = b.addModule("lanceql.io", .{
        .root_source_file = b.path("src/io/io.zig"),
    });

    const encoding_mod = b.addModule("lanceql.encoding", .{
        .root_source_file = b.path("src/encoding/encoding.zig"),
    });

    const format_mod = b.addModule("lanceql.format", .{
        .root_source_file = b.path("src/format/format.zig"),
        .imports = &.{
            .{ .name = "lanceql.proto", .module = proto_mod },
            .{ .name = "lanceql.io", .module = io_mod },
        },
    });

    // Root module exports all
    const lanceql_mod = b.addModule("lanceql", .{
        .root_source_file = b.path("src/lanceql.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.io", .module = io_mod },
            .{ .name = "lanceql.proto", .module = proto_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
        },
    });

    // === Tests ===
    const test_footer = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_footer.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.format", .module = format_mod },
            },
        }),
    });

    const test_proto = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_proto.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.proto", .module = proto_mod },
            },
        }),
    });

    // Run tests
    const run_test_footer = b.addRunArtifact(test_footer);
    const run_test_proto = b.addRunArtifact(test_proto);

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_test_footer.step);
    test_step.dependOn(&run_test_proto.step);

    const test_footer_step = b.step("test-footer", "Run footer tests");
    test_footer_step.dependOn(&run_test_footer.step);

    const test_proto_step = b.step("test-proto", "Run protobuf tests");
    test_proto_step.dependOn(&run_test_proto.step);

    // === WASM Build ===
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const wasm = b.addExecutable(.{
        .name = "lanceql",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wasm.zig"),
            .target = wasm_target,
            .optimize = .ReleaseSmall,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.io", .module = io_mod },
            },
        }),
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;

    const wasm_step = b.step("wasm", "Build WASM module");
    b.installArtifact(wasm);
    wasm_step.dependOn(&wasm.step);
}
