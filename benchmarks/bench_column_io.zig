//! Column-First I/O Benchmark
//!
//! Compares two approaches:
//! 1. Full-file read: Read entire 149MB file, extract 1MB column
//! 2. Column-first: Read only footer + metadata + column data (~1MB total)
//!
//! This demonstrates the I/O improvement from column-first reading.

const std = @import("std");
const format = @import("lanceql.format");
const io = @import("lanceql.io");
const simd = @import("lanceql.simd");

const Table = @import("lanceql.table").Table;
const LazyLanceFile = format.LazyLanceFile;
const FileReader = io.FileReader;

const WARMUP_ITERATIONS = 3;
const BENCH_ITERATIONS = 10;
const LANCE_PATH = "benchmarks/benchmark_e2e.lance";

/// Find the .lance file in the data directory
fn findLanceFile(allocator: std.mem.Allocator, lance_dir: []const u8) ![]const u8 {
    const data_path = try std.fmt.allocPrint(allocator, "{s}/data", .{lance_dir});
    defer allocator.free(data_path);

    var data_dir = try std.fs.cwd().openDir(data_path, .{ .iterate = true });
    defer data_dir.close();

    var iter = data_dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".lance")) {
            return std.fmt.allocPrint(allocator, "{s}/{s}", .{ data_path, entry.name });
        }
    }

    return error.LanceFileNotFound;
}

/// Read entire file into memory (old approach)
fn readFullFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const size = try file.getEndPos();
    const data = try allocator.alloc(u8, size);
    errdefer allocator.free(data);

    _ = try file.readAll(data);
    return data;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Column-First I/O Benchmark\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("\nComparing full-file read vs column-first I/O\n", .{});
    std.debug.print("\n", .{});

    // Find lance file
    const lance_file_path = findLanceFile(allocator, LANCE_PATH) catch {
        std.debug.print("ERROR: Lance file not found. Run:\n", .{});
        std.debug.print("  python3 benchmarks/generate_benchmark_data.py\n", .{});
        return;
    };
    defer allocator.free(lance_file_path);

    // Get file size
    const file_size = blk: {
        const file = try std.fs.cwd().openFile(lance_file_path, .{});
        defer file.close();
        break :blk try file.getEndPos();
    };

    std.debug.print("File: {s}\n", .{lance_file_path});
    std.debug.print("Size: {d:.1} MB\n", .{@as(f64, @floatFromInt(file_size)) / 1024 / 1024});
    std.debug.print("\n", .{});

    // =========================================================================
    // Benchmark 1: Full-file read (old approach)
    // =========================================================================
    std.debug.print("--- Method 1: Full-file read (current approach) ---\n", .{});
    std.debug.print("Reads entire {d:.1} MB file, extracts 1 column\n\n", .{@as(f64, @floatFromInt(file_size)) / 1024 / 1024});

    // Warmup
    for (0..WARMUP_ITERATIONS) |_| {
        const data = readFullFile(allocator, lance_file_path) catch break;
        defer allocator.free(data);

        var table = Table.init(allocator, data) catch break;
        defer table.deinit();

        const amounts = table.readFloat64Column(1) catch break;
        defer allocator.free(amounts);

        const sum_result = simd.sum(amounts);
        std.mem.doNotOptimizeAway(&sum_result);
    }

    var full_total_ns: u64 = 0;
    var full_min_ns: u64 = std.math.maxInt(u64);
    var full_total_bytes: u64 = 0;
    var row_count: usize = 0;

    for (0..BENCH_ITERATIONS) |_| {
        var timer = try std.time.Timer.start();

        const data = readFullFile(allocator, lance_file_path) catch break;
        defer allocator.free(data);

        var table = Table.init(allocator, data) catch break;
        defer table.deinit();

        const amounts = table.readFloat64Column(1) catch break;
        defer allocator.free(amounts);

        const sum_result = simd.sum(amounts);
        std.mem.doNotOptimizeAway(&sum_result);

        const elapsed = timer.read();
        full_total_ns += elapsed;
        full_min_ns = @min(full_min_ns, elapsed);
        full_total_bytes += file_size;
        row_count = amounts.len;
    }

    const full_avg_ms = @as(f64, @floatFromInt(full_total_ns / BENCH_ITERATIONS)) / 1_000_000;
    const full_min_ms = @as(f64, @floatFromInt(full_min_ns)) / 1_000_000;
    const full_throughput = @as(f64, @floatFromInt(row_count)) / (full_min_ms / 1000) / 1_000_000;
    const full_io_mb = @as(f64, @floatFromInt(file_size)) / 1024 / 1024;

    std.debug.print("  Rows: {d}\n", .{row_count});
    std.debug.print("  I/O per iteration: {d:.1} MB (entire file)\n", .{full_io_mb});
    std.debug.print("  Min time: {d:.2} ms\n", .{full_min_ms});
    std.debug.print("  Avg time: {d:.2} ms\n", .{full_avg_ms});
    std.debug.print("  Throughput: {d:.1}M rows/sec\n", .{full_throughput});
    std.debug.print("\n", .{});

    // =========================================================================
    // Benchmark 2: Column-first I/O (new approach)
    // =========================================================================
    std.debug.print("--- Method 2: Column-first I/O (new approach) ---\n", .{});
    std.debug.print("Reads only footer + metadata + column data\n\n", .{});

    // Warmup
    for (0..WARMUP_ITERATIONS) |_| {
        var file_reader = FileReader.open(lance_file_path) catch break;
        defer file_reader.close();

        var lazy = LazyLanceFile.init(allocator, file_reader.reader()) catch break;
        defer lazy.deinit();

        const amounts = lazy.readFloat64Column(1) catch break;
        defer allocator.free(amounts);

        const sum_result = simd.sum(amounts);
        std.mem.doNotOptimizeAway(&sum_result);
    }

    var lazy_total_ns: u64 = 0;
    var lazy_min_ns: u64 = std.math.maxInt(u64);

    for (0..BENCH_ITERATIONS) |_| {
        var timer = try std.time.Timer.start();

        var file_reader = FileReader.open(lance_file_path) catch break;
        defer file_reader.close();

        var lazy = LazyLanceFile.init(allocator, file_reader.reader()) catch break;
        defer lazy.deinit();

        const amounts = lazy.readFloat64Column(1) catch break;
        defer allocator.free(amounts);

        const sum_result = simd.sum(amounts);
        std.mem.doNotOptimizeAway(&sum_result);

        const elapsed = timer.read();
        lazy_total_ns += elapsed;
        lazy_min_ns = @min(lazy_min_ns, elapsed);
    }

    const lazy_avg_ms = @as(f64, @floatFromInt(lazy_total_ns / BENCH_ITERATIONS)) / 1_000_000;
    const lazy_min_ms = @as(f64, @floatFromInt(lazy_min_ns)) / 1_000_000;
    const lazy_throughput = @as(f64, @floatFromInt(row_count)) / (lazy_min_ms / 1000) / 1_000_000;

    // Estimate I/O: footer (40) + offset table (num_cols * 16) + col metadata (~1KB) + data
    const col_data_size = row_count * @sizeOf(f64);
    const estimated_io = 40 + (4 * 16) + 1024 + col_data_size; // rough estimate
    const lazy_io_mb = @as(f64, @floatFromInt(estimated_io)) / 1024 / 1024;

    std.debug.print("  Rows: {d}\n", .{row_count});
    std.debug.print("  I/O per iteration: ~{d:.1} MB (column only)\n", .{lazy_io_mb});
    std.debug.print("  Min time: {d:.2} ms\n", .{lazy_min_ms});
    std.debug.print("  Avg time: {d:.2} ms\n", .{lazy_avg_ms});
    std.debug.print("  Throughput: {d:.1}M rows/sec\n", .{lazy_throughput});
    std.debug.print("\n", .{});

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Summary\n", .{});
    std.debug.print("================================================================================\n", .{});

    const io_reduction = full_io_mb / lazy_io_mb;
    const speedup = full_min_ms / lazy_min_ms;

    std.debug.print("\n", .{});
    std.debug.print("  I/O reduction: {d:.0}x ({d:.1} MB → {d:.1} MB)\n", .{ io_reduction, full_io_mb, lazy_io_mb });
    std.debug.print("  Speed improvement: {d:.1}x ({d:.1} ms → {d:.1} ms)\n", .{ speedup, full_min_ms, lazy_min_ms });
    std.debug.print("\n", .{});

    if (speedup > 1) {
        std.debug.print("  Column-first I/O is {d:.1}x FASTER\n", .{speedup});
    } else {
        std.debug.print("  Column-first I/O is {d:.1}x slower (unexpected!)\n", .{1.0 / speedup});
    }
    std.debug.print("\n", .{});
}
