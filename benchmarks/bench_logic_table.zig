//! @logic_table Benchmark - End-to-End Comparison
//!
//! What we're comparing (all read from files, same 15-second duration):
//!   1. LanceQL @logic_table JIT - Python → metal0 direct API → execute
//!   2. DuckDB + Python loop - Read Parquet → row-by-row Python calls
//!   3. DuckDB → NumPy batch - Read Parquet → pull to Python → NumPy compute
//!   4. Polars + Python loop - Read Parquet → row-by-row Python calls
//!   5. Polars → NumPy batch - Read Parquet → pull to Python → NumPy compute
//!
//! This benchmark uses metal0's direct API (no subprocess) via:
//!   - lanceql.codegen.JitContext for compilation
//!   - metal0.compileWithSchema() for the full pipeline
//!
//! NOTE: The codegen may use runtime.eval() for list subscripts which
//!       causes type mismatches. Native Zig SIMD is used as fallback.
//!
//! Setup:
//!   python3 benchmarks/generate_benchmark_data.py
//!   zig build bench-logic-table

const std = @import("std");
const Table = @import("lanceql.table").Table;
const simd = @import("lanceql.simd");
const codegen = @import("lanceql.codegen");

const WARMUP_SECONDS = 2;
const BENCHMARK_SECONDS = 15;
const LANCE_PATH = "benchmarks/benchmark_e2e.lance";
const PARQUET_PATH = "benchmarks/benchmark_e2e.parquet";
const EMBEDDING_DIM = 384;

// Python @logic_table source code for JIT compilation
const PYTHON_SOURCE =
    \\from logic_table import logic_table
    \\
    \\@logic_table
    \\class VectorOps:
    \\    def dot_product(self, a: list, b: list) -> float:
    \\        result = 0.0
    \\        for i in range(len(a)):
    \\            result = result + a[i] * b[i]
    \\        return result
;

var has_duckdb: bool = false;
var has_polars: bool = false;

fn checkPythonModule(allocator: std.mem.Allocator, module: []const u8) bool {
    const py_code = std.fmt.allocPrint(allocator, "import {s}", .{module}) catch return false;
    defer allocator.free(py_code);

    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", "-c", py_code },
    }) catch return false;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    switch (result.term) {
        .Exited => |code| return code == 0,
        else => return false,
    }
}

fn runPythonTimedBenchmark(allocator: std.mem.Allocator, script: []const u8) !struct { iterations: u64, total_ns: u64 } {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", "-c", script },
        .max_output_bytes = 10 * 1024 * 1024,
    }) catch return .{ .iterations = 0, .total_ns = 0 };
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }

    var iterations: u64 = 0;
    var total_ns: u64 = 0;

    // Parse ITERATIONS:xxx
    if (std.mem.indexOf(u8, result.stdout, "ITERATIONS:")) |idx| {
        const start = idx + 11;
        var end = start;
        while (end < result.stdout.len and result.stdout[end] >= '0' and result.stdout[end] <= '9') {
            end += 1;
        }
        iterations = std.fmt.parseInt(u64, result.stdout[start..end], 10) catch 0;
    }

    // Parse TOTAL_NS:xxx
    if (std.mem.indexOf(u8, result.stdout, "TOTAL_NS:")) |idx| {
        const start = idx + 9;
        var end = start;
        while (end < result.stdout.len and result.stdout[end] >= '0' and result.stdout[end] <= '9') {
            end += 1;
        }
        total_ns = std.fmt.parseInt(u64, result.stdout[start..end], 10) catch 0;
    }

    return .{ .iterations = iterations, .total_ns = total_ns };
}

fn readLanceFile(allocator: std.mem.Allocator) ![]const u8 {
    // Find and read the Lance data file
    var data_dir = std.fs.cwd().openDir(LANCE_PATH ++ "/data", .{ .iterate = true }) catch return error.FileNotFound;
    defer data_dir.close();

    var iter = data_dir.iterate();
    var lance_file_name_buf: [256]u8 = undefined;
    var lance_file_name: ?[]const u8 = null;

    while (iter.next() catch null) |entry| {
        if (std.mem.endsWith(u8, entry.name, ".lance")) {
            const len = @min(entry.name.len, lance_file_name_buf.len);
            @memcpy(lance_file_name_buf[0..len], entry.name[0..len]);
            lance_file_name = lance_file_name_buf[0..len];
            break;
        }
    }

    if (lance_file_name == null) return error.FileNotFound;

    const data_file = data_dir.openFile(lance_file_name.?, .{}) catch return error.FileNotFound;
    defer data_file.close();

    const file_size = (data_file.stat() catch return error.FileNotFound).size;
    const file_data = allocator.alloc(u8, file_size) catch return error.OutOfMemory;

    const bytes_read = data_file.readAll(file_data) catch return error.ReadError;
    if (bytes_read != file_size) return error.ReadError;

    return file_data;
}

/// JIT compile Python @logic_table using metal0 API directly
/// Returns the compiled function from JitContext
fn jitCompile(allocator: std.mem.Allocator) !struct { ctx: codegen.JitContext, fn_ptr: ?*const fn ([*]f64, [*]f64, usize) callconv(.c) f64 } {
    // Create JIT context
    var ctx = codegen.JitContext.init(allocator);
    errdefer ctx.deinit();

    // Compile the @logic_table class using direct API (no subprocess)
    const compiled = ctx.compileLogicTable(PYTHON_SOURCE, "dot_product") catch |err| {
        std.log.err("JIT compilation failed: {}", .{err});
        return error.CompilationFailed;
    };

    // Get the function pointer if available
    const fn_ptr: ?*const fn ([*]f64, [*]f64, usize) callconv(.c) f64 = if (compiled.ptr) |ptr|
        @ptrCast(@alignCast(ptr))
    else
        null;

    return .{
        .ctx = ctx,
        .fn_ptr = fn_ptr,
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("===============================================================================\n", .{});
    std.debug.print("@logic_table Benchmark: REAL JIT End-to-End (Python → metal0 → Execute)\n", .{});
    std.debug.print("===============================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Each method runs for {d} seconds. Measuring throughput (rows/sec).\n", .{BENCHMARK_SECONDS});
    std.debug.print("\n", .{});

    // Check files exist
    const lance_exists = if (std.fs.cwd().access(LANCE_PATH, .{})) true else |_| false;
    const parquet_exists = if (std.fs.cwd().access(PARQUET_PATH, .{})) true else |_| false;

    if (!lance_exists or !parquet_exists) {
        std.debug.print("ERROR: Benchmark data not found. Run:\n", .{});
        std.debug.print("  python3 benchmarks/generate_benchmark_data.py\n", .{});
        return;
    }

    // Check for Python modules
    has_duckdb = checkPythonModule(allocator, "duckdb");
    has_polars = checkPythonModule(allocator, "polars");

    std.debug.print("Data files:\n", .{});
    std.debug.print("  Lance:   {s} ✓\n", .{LANCE_PATH});
    std.debug.print("  Parquet: {s} ✓\n", .{PARQUET_PATH});
    std.debug.print("\n", .{});
    std.debug.print("Engines:\n", .{});
    std.debug.print("  LanceQL @logic_table: yes (JIT via metal0)\n", .{});
    std.debug.print("  DuckDB:               {s}\n", .{if (has_duckdb) "yes" else "no (pip install duckdb)"});
    std.debug.print("  Polars:               {s}\n", .{if (has_polars) "yes" else "no (pip install polars)"});
    std.debug.print("\n", .{});

    std.debug.print("===============================================================================\n", .{});
    std.debug.print("{s:<32} {s:>14} {s:>10} {s:>10}\n", .{ "Method", "Rows/sec", "Iters", "vs Best" });
    std.debug.print("===============================================================================\n", .{});

    var lanceql_throughput: f64 = 0;

    // 1. LanceQL @logic_table JIT - REAL flow: Python → metal0 compile → execute
    {
        // Step 1: JIT compile (measure separately)
        std.debug.print("\nJIT compiling Python @logic_table via metal0...\n", .{});
        const jit_start = std.time.nanoTimestamp();

        var jit_result = jitCompile(allocator) catch |err| {
            std.debug.print("{s:<32} {s:>14} {s:>10} {s:>10}\n", .{
                "LanceQL @logic_table JIT", "JIT failed", "-", "-",
            });
            std.debug.print("  Error: {}\n", .{err});
            std.debug.print("  Using direct metal0 API (no subprocess)\n", .{});
            // Fall through to Python benchmarks
            lanceql_throughput = 0;
            return;
        };
        defer jit_result.ctx.deinit();

        const jit_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - jit_start);
        const jit_elapsed_ms = @as(f64, @floatFromInt(jit_elapsed_ns)) / 1_000_000.0;
        std.debug.print("  JIT compile time: {d:.1}ms\n", .{jit_elapsed_ms});

        if (jit_result.fn_ptr == null) {
            std.debug.print("{s:<32} {s:>14} {s:>10} {s:>10}\n", .{
                "LanceQL @logic_table JIT", "symbol missing", "-", "-",
            });
            std.debug.print("  VectorOps_dot_product not found in compiled library\n", .{});
            lanceql_throughput = 0;
        } else {
            const dot_product_fn = jit_result.fn_ptr.?;

            // Step 2: Benchmark the compiled function
            const warmup_end = std.time.nanoTimestamp() + WARMUP_SECONDS * std.time.ns_per_s;
            const benchmark_end_time = warmup_end + BENCHMARK_SECONDS * std.time.ns_per_s;

            var iterations: u64 = 0;
            var total_rows: u64 = 0;

            // Query vector as f64 (JIT compiled function expects f64)
            var query_vec: [EMBEDDING_DIM]f64 = undefined;
            for (&query_vec) |*v| v.* = 0.1;

            // Warmup
            while (std.time.nanoTimestamp() < warmup_end) {
                const file_data = readLanceFile(allocator) catch break;
                defer allocator.free(file_data);

                var table = Table.init(allocator, file_data) catch break;
                defer table.deinit();

                const embeddings = table.readFloat32Column(3) catch break; // embedding column
                defer allocator.free(embeddings);

                const num_rows = embeddings.len / EMBEDDING_DIM;

                // Convert embeddings to f64 for the JIT function
                const embeddings_f64 = allocator.alloc(f64, embeddings.len) catch break;
                defer allocator.free(embeddings_f64);
                for (embeddings, 0..) |v, i| embeddings_f64[i] = @floatCast(v);

                // Call JIT-compiled function for each row
                for (0..num_rows) |row| {
                    const row_start = row * EMBEDDING_DIM;
                    const score = dot_product_fn(
                        embeddings_f64.ptr + row_start,
                        &query_vec,
                        EMBEDDING_DIM,
                    );
                    std.mem.doNotOptimizeAway(score);
                }
            }

            // Benchmark
            const start_time = std.time.nanoTimestamp();
            while (std.time.nanoTimestamp() < benchmark_end_time) {
                const file_data = readLanceFile(allocator) catch break;
                defer allocator.free(file_data);

                var table = Table.init(allocator, file_data) catch break;
                defer table.deinit();

                const embeddings = table.readFloat32Column(3) catch break; // embedding column
                defer allocator.free(embeddings);

                const num_rows = embeddings.len / EMBEDDING_DIM;

                // Convert embeddings to f64 for the JIT function
                const embeddings_f64 = allocator.alloc(f64, embeddings.len) catch break;
                defer allocator.free(embeddings_f64);
                for (embeddings, 0..) |v, i| embeddings_f64[i] = @floatCast(v);

                // Call JIT-compiled function for each row
                for (0..num_rows) |row| {
                    const row_start = row * EMBEDDING_DIM;
                    const score = dot_product_fn(
                        embeddings_f64.ptr + row_start,
                        &query_vec,
                        EMBEDDING_DIM,
                    );
                    std.mem.doNotOptimizeAway(score);
                }

                iterations += 1;
                total_rows += num_rows;
            }
            const elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - start_time);

            lanceql_throughput = @as(f64, @floatFromInt(total_rows)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0);
            std.debug.print("{s:<32} {d:>14.0} {d:>10} {s:>10}\n", .{
                "LanceQL @logic_table JIT", lanceql_throughput, iterations, "1.0x",
            });
        }
    }

    std.debug.print("\n", .{});

    // 2. DuckDB + Python loop (per-row Python calls on 384-dim embeddings)
    if (has_duckdb) duckdb_udf: {
        const script = std.fmt.comptimePrint(
            \\import duckdb
            \\import warnings
            \\warnings.filterwarnings("ignore")
            \\import time
            \\import numpy as np
            \\
            \\BENCHMARK_SECONDS = {d}
            \\WARMUP_SECONDS = {d}
            \\PARQUET_PATH = "{s}"
            \\
            \\con = duckdb.connect()
            \\con.execute("SET enable_progress_bar = false")
            \\
            \\query = np.full(384, 0.1, dtype=np.float32)
            \\
            \\# Python function for 384-dim dot product (called per row)
            \\# Uses np.dot - same as Polars UDF for fair comparison
            \\def dot_product(embedding):
            \\    return float(np.dot(embedding, query))
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    df = con.execute(f"SELECT embedding FROM read_parquet('{{PARQUET_PATH}}') LIMIT 100").fetch_arrow_table()
            \\    embeddings = df['embedding'].to_pylist()
            \\    for emb in embeddings:
            \\        _ = dot_product(emb)
            \\
            \\# Benchmark: Read file + per-row Python function calls
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = con.execute(f"SELECT embedding FROM read_parquet('{{PARQUET_PATH}}')").fetch_arrow_table()
            \\    embeddings = df['embedding'].to_pylist()
            \\    total_score = 0.0
            \\    for emb in embeddings:
            \\        total_score += dot_product(emb)  # Per-row Python call
            \\    iterations += 1
            \\    total_rows += len(embeddings)
            \\elapsed_ns = int((time.time() - start) * 1e9)
            \\
            \\print(f"ITERATIONS:{{iterations}}")
            \\print(f"TOTAL_NS:{{elapsed_ns}}")
            \\print(f"ROWS:{{total_rows}}")
        , .{ BENCHMARK_SECONDS, WARMUP_SECONDS, PARQUET_PATH });

        const py_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", script },
            .max_output_bytes = 10 * 1024 * 1024,
        }) catch {
            std.debug.print("{s:<32} {s:>14} {s:>10} {s:>10}\n", .{ "DuckDB Python UDF", "error", "-", "-" });
            break :duckdb_udf;
        };
        defer {
            allocator.free(py_result.stdout);
            allocator.free(py_result.stderr);
        }

        var iterations: u64 = 0;
        var total_ns: u64 = 0;
        var total_rows: u64 = 0;

        if (std.mem.indexOf(u8, py_result.stdout, "ITERATIONS:")) |idx| {
            const start = idx + 11;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') end += 1;
            iterations = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "TOTAL_NS:")) |idx| {
            const start = idx + 9;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') end += 1;
            total_ns = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "ROWS:")) |idx| {
            const start = idx + 5;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') end += 1;
            total_rows = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }

        if (iterations > 0 and total_ns > 0 and total_rows > 0) {
            const throughput = @as(f64, @floatFromInt(total_rows)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);
            const speedup = if (lanceql_throughput > 0) lanceql_throughput / throughput else 0;
            var speedup_buf: [16]u8 = undefined;
            const speedup_str = if (lanceql_throughput > 0) std.fmt.bufPrint(&speedup_buf, "{d:.1}x", .{speedup}) catch "N/A" else "N/A";
            std.debug.print("{s:<32} {d:>14.0} {d:>10} {s:>10}\n", .{
                "DuckDB + Python loop", throughput, iterations, speedup_str,
            });
        } else {
            std.debug.print("{s:<32} {s:>14} {s:>10} {s:>10}\n", .{ "DuckDB + Python loop", "error", "-", "-" });
        }
    }

    // 3. DuckDB → NumPy batch
    if (has_duckdb) duckdb_numpy: {
        const script = std.fmt.comptimePrint(
            \\import duckdb
            \\import warnings
            \\warnings.filterwarnings("ignore")
            \\import time
            \\import numpy as np
            \\
            \\BENCHMARK_SECONDS = {d}
            \\WARMUP_SECONDS = {d}
            \\PARQUET_PATH = "{s}"
            \\
            \\con = duckdb.connect()
            \\con.execute("SET enable_progress_bar = false")
            \\query_vec = np.full(384, 0.1, dtype=np.float32)
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    df = con.execute(f"SELECT embedding FROM read_parquet('{{PARQUET_PATH}}') LIMIT 100").fetchdf()
            \\
            \\# Benchmark: Read file + batch NumPy matrix multiply
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = con.execute(f"SELECT embedding FROM read_parquet('{{PARQUET_PATH}}')").fetchdf()
            \\    embeddings = np.array(df['embedding'].tolist())  # More efficient than vstack
            \\    scores = embeddings @ query_vec  # Vectorized SIMD dot product
            \\    iterations += 1
            \\    total_rows += len(df)
            \\elapsed_ns = int((time.time() - start) * 1e9)
            \\
            \\print(f"ITERATIONS:{{iterations}}")
            \\print(f"TOTAL_NS:{{elapsed_ns}}")
            \\print(f"ROWS:{{total_rows}}")
        , .{ BENCHMARK_SECONDS, WARMUP_SECONDS, PARQUET_PATH });

        const py_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", script },
            .max_output_bytes = 10 * 1024 * 1024,
        }) catch {
            std.debug.print("{s:<32} {s:>14} {s:>10} {s:>10}\n", .{ "DuckDB → NumPy batch", "error", "-", "-" });
            break :duckdb_numpy;
        };
        defer {
            allocator.free(py_result.stdout);
            allocator.free(py_result.stderr);
        }

        var iterations: u64 = 0;
        var total_ns: u64 = 0;
        var total_rows: u64 = 0;

        if (std.mem.indexOf(u8, py_result.stdout, "ITERATIONS:")) |idx| {
            const start = idx + 11;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            iterations = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "TOTAL_NS:")) |idx| {
            const start = idx + 9;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            total_ns = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "ROWS:")) |idx| {
            const start = idx + 5;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            total_rows = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }

        if (iterations > 0 and total_ns > 0) {
            const throughput = @as(f64, @floatFromInt(total_rows)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);
            const speedup = if (lanceql_throughput > 0) lanceql_throughput / throughput else 0;
            var speedup_buf: [16]u8 = undefined;
            const speedup_str = if (lanceql_throughput > 0) std.fmt.bufPrint(&speedup_buf, "{d:.1}x", .{speedup}) catch "N/A" else "N/A";
            std.debug.print("{s:<32} {d:>14.0} {d:>10} {s:>10}\n", .{
                "DuckDB → NumPy batch", throughput, iterations, speedup_str,
            });
        }
    }

    // 4. Polars + Python UDF (per-row Python calls on 384-dim embeddings)
    if (has_polars) polars_udf: {
        const script = std.fmt.comptimePrint(
            \\import warnings
            \\warnings.filterwarnings("ignore")
            \\import polars as pl
            \\import time
            \\import numpy as np
            \\
            \\BENCHMARK_SECONDS = {d}
            \\WARMUP_SECONDS = {d}
            \\PARQUET_PATH = "{s}"
            \\
            \\query = np.full(384, 0.1, dtype=np.float32)  # Defined ONCE outside function
            \\def dot_product_udf(embedding):
            \\    return float(np.dot(embedding, query))
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    df = pl.read_parquet(PARQUET_PATH).head(100)
            \\    _ = df.select(pl.col('embedding').map_elements(dot_product_udf, return_dtype=pl.Float64))
            \\
            \\# Benchmark
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = pl.read_parquet(PARQUET_PATH)
            \\    result = df.select(pl.col('embedding').map_elements(dot_product_udf, return_dtype=pl.Float64))
            \\    iterations += 1
            \\    total_rows += len(df)
            \\elapsed_ns = int((time.time() - start) * 1e9)
            \\
            \\print(f"ITERATIONS:{{iterations}}")
            \\print(f"TOTAL_NS:{{elapsed_ns}}")
            \\print(f"ROWS:{{total_rows}}")
        , .{ BENCHMARK_SECONDS, WARMUP_SECONDS, PARQUET_PATH });

        const py_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", script },
            .max_output_bytes = 10 * 1024 * 1024,
        }) catch {
            std.debug.print("{s:<32} {s:>14} {s:>10} {s:>10}\n", .{ "Polars Python UDF", "error", "-", "-" });
            break :polars_udf;
        };
        defer {
            allocator.free(py_result.stdout);
            allocator.free(py_result.stderr);
        }

        var iterations: u64 = 0;
        var total_ns: u64 = 0;
        var total_rows: u64 = 0;

        if (std.mem.indexOf(u8, py_result.stdout, "ITERATIONS:")) |idx| {
            const start = idx + 11;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            iterations = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "TOTAL_NS:")) |idx| {
            const start = idx + 9;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            total_ns = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "ROWS:")) |idx| {
            const start = idx + 5;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            total_rows = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }

        if (iterations > 0 and total_ns > 0) {
            const throughput = @as(f64, @floatFromInt(total_rows)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);
            const speedup = if (lanceql_throughput > 0) lanceql_throughput / throughput else 0;
            var speedup_buf: [16]u8 = undefined;
            const speedup_str = if (lanceql_throughput > 0) std.fmt.bufPrint(&speedup_buf, "{d:.1}x", .{speedup}) catch "N/A" else "N/A";
            std.debug.print("{s:<32} {d:>14.0} {d:>10} {s:>10}\n", .{
                "Polars + Python UDF", throughput, iterations, speedup_str,
            });
        } else {
            std.debug.print("{s:<32} {s:>14} {s:>10} {s:>10}\n", .{ "Polars + Python UDF", "error", "-", "-" });
        }
    }

    // 5. Polars → NumPy batch (384-dim embeddings)
    if (has_polars) polars_numpy: {
        const script = std.fmt.comptimePrint(
            \\import warnings
            \\warnings.filterwarnings("ignore")
            \\import polars as pl
            \\import time
            \\import numpy as np
            \\
            \\BENCHMARK_SECONDS = {d}
            \\WARMUP_SECONDS = {d}
            \\PARQUET_PATH = "{s}"
            \\
            \\query_vec = np.full(384, 0.1, dtype=np.float32)
            \\
            \\# Warmup
            \\warmup_end = time.time() + WARMUP_SECONDS
            \\while time.time() < warmup_end:
            \\    df = pl.read_parquet(PARQUET_PATH).head(100)
            \\
            \\# Benchmark: Read file + batch NumPy matrix multiply
            \\iterations = 0
            \\total_rows = 0
            \\start = time.time()
            \\benchmark_end = start + BENCHMARK_SECONDS
            \\while time.time() < benchmark_end:
            \\    df = pl.read_parquet(PARQUET_PATH)
            \\    embeddings = np.array(df['embedding'].to_list())  # More efficient than vstack
            \\    scores = embeddings @ query_vec  # Vectorized SIMD dot product
            \\    iterations += 1
            \\    total_rows += len(df)
            \\elapsed_ns = int((time.time() - start) * 1e9)
            \\
            \\print(f"ITERATIONS:{{iterations}}")
            \\print(f"TOTAL_NS:{{elapsed_ns}}")
            \\print(f"ROWS:{{total_rows}}")
        , .{ BENCHMARK_SECONDS, WARMUP_SECONDS, PARQUET_PATH });

        const py_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{ "python3", "-c", script },
            .max_output_bytes = 10 * 1024 * 1024,
        }) catch {
            std.debug.print("{s:<32} {s:>14} {s:>10} {s:>10}\n", .{ "Polars → NumPy batch", "error", "-", "-" });
            break :polars_numpy;
        };
        defer {
            allocator.free(py_result.stdout);
            allocator.free(py_result.stderr);
        }

        var iterations: u64 = 0;
        var total_ns: u64 = 0;
        var total_rows: u64 = 0;

        if (std.mem.indexOf(u8, py_result.stdout, "ITERATIONS:")) |idx| {
            const start = idx + 11;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            iterations = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "TOTAL_NS:")) |idx| {
            const start = idx + 9;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            total_ns = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }
        if (std.mem.indexOf(u8, py_result.stdout, "ROWS:")) |idx| {
            const start = idx + 5;
            var end = start;
            while (end < py_result.stdout.len and py_result.stdout[end] >= '0' and py_result.stdout[end] <= '9') {
                end += 1;
            }
            total_rows = std.fmt.parseInt(u64, py_result.stdout[start..end], 10) catch 0;
        }

        if (iterations > 0 and total_ns > 0) {
            const throughput = @as(f64, @floatFromInt(total_rows)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);
            const speedup = if (lanceql_throughput > 0) lanceql_throughput / throughput else 0;
            var speedup_buf: [16]u8 = undefined;
            const speedup_str = if (lanceql_throughput > 0) std.fmt.bufPrint(&speedup_buf, "{d:.1}x", .{speedup}) catch "N/A" else "N/A";
            std.debug.print("{s:<32} {d:>14.0} {d:>10} {s:>10}\n", .{
                "Polars → NumPy batch", throughput, iterations, speedup_str,
            });
        }
    }

    std.debug.print("===============================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Notes:\n", .{});
    std.debug.print("  - All methods read from disk (Lance or Parquet files)\n", .{});
    std.debug.print("  - LanceQL JIT compiles Python @logic_table at runtime via metal0\n", .{});
    std.debug.print("  - JIT compilation time is measured separately from execution\n", .{});
    std.debug.print("  - All methods run for exactly {d} seconds after warmup\n", .{BENCHMARK_SECONDS});
    std.debug.print("  - Throughput = total rows processed / elapsed time\n", .{});
    std.debug.print("\n", .{});
}
