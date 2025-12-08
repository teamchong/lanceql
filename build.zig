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

    const value_mod = b.addModule("lanceql.value", .{
        .root_source_file = b.path("src/value.zig"),
    });

    const query_mod = b.addModule("lanceql.query", .{
        .root_source_file = b.path("src/query/query.zig"),
        .imports = &.{
            .{ .name = "lanceql.value", .module = value_mod },
        },
    });

    const format_mod = b.addModule("lanceql.format", .{
        .root_source_file = b.path("src/format/format.zig"),
        .imports = &.{
            .{ .name = "lanceql.proto", .module = proto_mod },
            .{ .name = "lanceql.io", .module = io_mod },
        },
    });

    const table_mod = b.addModule("lanceql.table", .{
        .root_source_file = b.path("src/table.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.proto", .module = proto_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
        },
    });

    const dataframe_mod = b.addModule("lanceql.dataframe", .{
        .root_source_file = b.path("src/dataframe.zig"),
        .imports = &.{
            .{ .name = "lanceql.value", .module = value_mod },
            .{ .name = "lanceql.query", .module = query_mod },
            .{ .name = "lanceql.table", .module = table_mod },
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
            .{ .name = "lanceql.table", .module = table_mod },
            .{ .name = "lanceql.query", .module = query_mod },
            .{ .name = "lanceql.value", .module = value_mod },
            .{ .name = "lanceql.dataframe", .module = dataframe_mod },
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

    const test_integration = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_integration.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.io", .module = io_mod },
                .{ .name = "lanceql.proto", .module = proto_mod },
                .{ .name = "lanceql.encoding", .module = encoding_mod },
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });

    // Run tests
    const run_test_footer = b.addRunArtifact(test_footer);
    const run_test_proto = b.addRunArtifact(test_proto);
    const run_test_integration = b.addRunArtifact(test_integration);

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_test_footer.step);
    test_step.dependOn(&run_test_proto.step);
    test_step.dependOn(&run_test_integration.step);

    const test_footer_step = b.step("test-footer", "Run footer tests");
    test_footer_step.dependOn(&run_test_footer.step);

    const test_proto_step = b.step("test-proto", "Run protobuf tests");
    test_proto_step.dependOn(&run_test_proto.step);

    const test_integration_step = b.step("test-integration", "Run integration tests with real .lance files");
    test_integration_step.dependOn(&run_test_integration.step);

    // Query module tests
    const test_query = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/query/query.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql.value", .module = value_mod },
            },
        }),
    });

    const run_test_query = b.addRunArtifact(test_query);
    test_step.dependOn(&run_test_query.step);

    const test_query_step = b.step("test-query", "Run query module tests");
    test_query_step.dependOn(&run_test_query.step);

    // Value module tests
    const test_value = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/value.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_value = b.addRunArtifact(test_value);
    test_step.dependOn(&run_test_value.step);

    // DataFrame module tests
    const test_dataframe = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/dataframe.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql.value", .module = value_mod },
                .{ .name = "lanceql.query", .module = query_mod },
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });

    const run_test_dataframe = b.addRunArtifact(test_dataframe);
    test_step.dependOn(&run_test_dataframe.step);

    const test_dataframe_step = b.step("test-dataframe", "Run DataFrame module tests");
    test_dataframe_step.dependOn(&run_test_dataframe.step);

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
            // No imports needed - wasm.zig is self-contained
        }),
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;

    const wasm_step = b.step("wasm", "Build WASM module");
    const install_wasm = b.addInstallArtifact(wasm, .{});
    wasm_step.dependOn(&install_wasm.step);
}
