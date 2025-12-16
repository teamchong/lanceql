//! Simple Parquet reader benchmark
//!
//! Usage: zig build-exe benchmarks/bench_parquet.zig && ./bench_parquet <parquet_file>

const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <parquet_file>\n", .{args[0]});
        return;
    }

    const file_path = args[1];

    // Read file into memory
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const data = try allocator.alloc(u8, file_size);
    defer allocator.free(data);

    _ = try file.readAll(data);

    std.debug.print("Parquet Reader Benchmark\n", .{});
    std.debug.print("========================\n", .{});
    std.debug.print("File: {s}\n", .{file_path});
    std.debug.print("Size: {d:.2} MB\n\n", .{@as(f64, @floatFromInt(file_size)) / 1024 / 1024});

    // Import parquet modules
    const format = @import("lanceql.format");
    const ParquetFile = format.ParquetFile;
    const parquet_enc = @import("lanceql.encoding.parquet");
    const PageReader = parquet_enc.PageReader;

    // Warmup
    const warmup_iterations = 3;
    const bench_iterations = 10;

    std.debug.print("Warming up ({d} iterations)...\n", .{warmup_iterations});
    for (0..warmup_iterations) |_| {
        var pf = try ParquetFile.init(allocator, data);
        defer pf.deinit();

        const rg = pf.getRowGroup(0) orelse continue;
        for (0..rg.columns.len) |col_idx| {
            const col = rg.columns[col_idx];
            const col_meta = col.meta_data orelse continue;
            const col_data = pf.getColumnData(0, col_idx) orelse continue;

            var reader = PageReader.init(col_data, col_meta.type_, null, col_meta.codec, allocator);
            defer reader.deinit();

            var page = reader.readAll() catch continue;
            defer page.deinit(allocator);
        }
    }

    // Benchmark
    std.debug.print("Running benchmark ({d} iterations)...\n\n", .{bench_iterations});

    var total_ns: u64 = 0;
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;
    var total_rows: usize = 0;

    for (0..bench_iterations) |_| {
        var timer = try std.time.Timer.start();

        var pf = try ParquetFile.init(allocator, data);
        defer pf.deinit();

        const rg = pf.getRowGroup(0) orelse continue;
        var rows_in_rg: usize = 0;

        for (0..rg.columns.len) |col_idx| {
            const col = rg.columns[col_idx];
            const col_meta = col.meta_data orelse continue;
            const col_data = pf.getColumnData(0, col_idx) orelse continue;

            var reader = PageReader.init(col_data, col_meta.type_, null, col_meta.codec, allocator);
            defer reader.deinit();

            var page = reader.readAll() catch continue;
            defer page.deinit(allocator);

            if (col_idx == 0) rows_in_rg = page.num_values;
        }

        const elapsed = timer.read();
        total_ns += elapsed;
        min_ns = @min(min_ns, elapsed);
        max_ns = @max(max_ns, elapsed);
        total_rows += rows_in_rg;
    }

    const avg_ns = total_ns / bench_iterations;
    const avg_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000;
    const min_ms = @as(f64, @floatFromInt(min_ns)) / 1_000_000;
    const max_ms = @as(f64, @floatFromInt(max_ns)) / 1_000_000;
    const avg_rows = total_rows / bench_iterations;
    const throughput = @as(f64, @floatFromInt(avg_rows)) / (avg_ms / 1000) / 1_000_000;

    std.debug.print("Results:\n", .{});
    std.debug.print("  Rows per iteration: {d}\n", .{avg_rows});
    std.debug.print("  Min:  {d:.2} ms\n", .{min_ms});
    std.debug.print("  Avg:  {d:.2} ms\n", .{avg_ms});
    std.debug.print("  Max:  {d:.2} ms\n", .{max_ms});
    std.debug.print("  Throughput: {d:.2}M rows/sec\n", .{throughput});
}
