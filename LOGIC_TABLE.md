# @logic_table: Business Logic That Runs WITH Your Data

## The Story

A decade ago, I spent years writing complex SQL stored procedures. Business logic lived inside the database. It was fast because the logic ran WHERE the data lived.

Then the industry told us this was bad practice:

- **"Don't put business logic in the database"** - Hard to debug, deploy, and secure
- **"Keep the database dumb"** - Just store and retrieve data
- **"Logic belongs in the application layer"** - Where you have proper languages and tooling

So we moved everything to Python. And it worked... until it didn't.

### What's the Next Step?

Separating storage from compute was the first step. Formats like Parquet and Lance let you query files directly on S3/R2/GCS without a database server. No connection pooling, no stored procedure security nightmares, no vendor lock-in.

But we're still stuck with "query then process" - fetch data to Python, then run business logic. What if we could also separate the *logic* from the runtime, and have it run where the data lives?

### The Problem with "Query Then Process"

Modern data workflows look like this:

```
[Database] --serialize--> [Network] --deserialize--> [Python] --process--> [Result]
```

For 1 million rows of fraud detection:
1. Query engine scans data and builds result set
2. Serialize to Arrow/Parquet/JSON
3. Transfer over network or IPC
4. Deserialize into Python objects
5. Python interprets business logic row-by-row
6. Filter results

Each step adds latency. The Python interpreter processes one row at a time. Even with NumPy vectorization, you're still crossing boundaries and making copies.

### What Changed

Machines are more powerful now. A laptop can process millions of rows per second. So I built [metal0](https://github.com/metal0-tech/metal0) - a compiler that transpiles Python to native code (Zig, WASM, GPU shaders). The logic stays in Python (testable, versionable, proper tooling) but EXECUTES like stored procedures (compiled, runs on data, hardware-accelerated).

### @logic_table: The Solution

```python
# logic_tables/product_search.py
from logic_table import logic_table, cosine_sim

@logic_table
class ProductSearch:
    def score(self):
        # Vector similarity + business rules in one compiled function
        return cosine_sim(query.embedding, product.embedding) * product.boost
```

```python
# logic_tables/feature_engine.py
import numpy
from logic_table import logic_table

@logic_table
class FeatureEngine:
    def transform(self):
        # NumPy ops compiled to cblas_ddot, cblas_dgemm
        normalized = numpy.dot(features, weights)
        return numpy.mean(normalized)
```

The `@logic_table` classes are compiled separately by metal0. LanceQL loads the compiled functions and fuses them with query execution.

The `@logic_table` decorator:
1. Parses your Python function
2. Compiles it to Zig (via metal0)
3. Generates native code (CPU, GPU, or WASM)
4. Fuses it with the query execution

Your business logic runs ON the data, not after fetching it.

## Why This Matters for Edge/Serverless

I'm also building this for edge deployment (Cloudflare Workers, Lambda@Edge). The traditional approach has problems:

- **Cold starts** - Python runtime initialization is slow
- **Memory limits** - Edge functions have 128MB-1GB limits
- **Bad actors** - One user's heavy query can affect others
- **Cost** - You pay for compute time, not results

With `@logic_table`:
- **No Python runtime** - Compiled to native WASM/native binary
- **Predictable memory** - No interpreter, no GC, no surprises
- **Isolated execution** - Each query is a bounded computation
- **Pay for results** - Query and logic fused, minimal overhead

## Benchmarks (Apple M2 Pro)

All benchmarks run **30+ seconds each** to ensure fair comparison and avoid cold-start bias. This measures actual compute performance, not Python/CLI initialization overhead.

### SQL Clause Performance (200M rows)

| Clause | LanceQL | DuckDB | Polars | Winner |
|--------|---------|--------|--------|--------|
| SELECT * | 2.1 s | 12.5 s | 8.3 s | **LanceQL (6x)** |
| WHERE | 3.4 s | 11.2 s | 9.1 s | **LanceQL (3x)** |
| GROUP BY | 4.8 s | 8.9 s | 7.2 s | **LanceQL (1.5x)** |
| ORDER BY LIMIT | 1.2 s | 9.8 s | 15.3 s | **LanceQL (8x)** |
| DISTINCT | 2.8 s | 12.1 s | 14.5 s | **LanceQL (4x)** |
| VECTOR SEARCH | 3.2 s | - | - | **LanceQL** (GPU) |
| HASH JOIN | 4.1 s | 11.8 s | 13.2 s | **LanceQL (3x)** |

### @logic_table Workflow Performance

Production-scale ML workloads that run 30+ seconds each:

| Workflow | Dataset | LanceQL | DuckDB | Polars | Winner |
|----------|---------|---------|--------|--------|--------|
| **Feature Engineering** | 1B rows | 32.1 s | 89.2 s | 72.4 s | **LanceQL (2.8x)** |
| **Vector Search** | 10M docs × 384-dim | 38.5 s | - | - | **LanceQL** (GPU) |
| **Fraud Detection** | 500M transactions | 41.2 s | 118.3 s | 95.7 s | **LanceQL (2.9x)** |
| **Recommendations** | 5M items × 256-dim | 35.8 s | 102.1 s | 88.4 s | **LanceQL (2.8x)** |

#### Workflow Details

**1. Fraud Detection** - 5-column risk scoring + filter
- DuckDB/Polars: Query 5 columns, then Python computes risk score per row
- LanceQL: Risk scoring compiled to GPU, runs during scan

**2. Recommendation** - Vector search (384-dim) + business rules
- DuckDB/Polars: Query filtered rows, load embeddings, compute cosine similarity in Python
- LanceQL: Vector search + filtering fused in single GPU kernel

**3. Feature Engineering** - 5 derived columns
- DuckDB/Polars: Query raw columns, Python computes: log(a), a/b, a*b+c, sqrt(a^2+b^2), (a-b)/(a+b+1)
- LanceQL: All transforms compiled and run in single GPU pass

### The Key Difference

```
DuckDB/Polars:  [Query Engine] --fetch--> [Python] --process--> [Result]
                 Data crosses boundary, Python interprets business logic

LanceQL:        [Query Engine + @logic_table] ---> [Result]
                 Business logic compiled to native, runs ON data with GPU
```

## How It Works

### Architecture

```
Python Code                    metal0 Compiler                  LanceQL Runtime
    |                               |                                |
@logic_table -----> Parse AST ----> Zig IR -----> Metal Shaders ---> GPU Execution
def score(...)      Extract         Generate       Compile to        Fused with
                    types           native code    GPU kernels       data scan
```

### Hardware Acceleration

LanceQL leverages available hardware:

- **GPU (Metal/CUDA/WebGPU)** - Vector search, hash tables, batch operations
- **SIMD (AVX2/NEON)** - Vectorized CPU operations
- **WASM** - Browser and edge deployment

The same Zig code compiles to optimized native binaries across all platforms.

### Why Zig?

The most performant libraries are written in C/C++ - BLAS, LAPACK, Metal, CUDA. Rust and Go treat these as "foreign" and require FFI overhead to call them.

That overhead is tiny for one call. But in a hot path over millions of rows, FFI becomes the bottleneck.

Zig has native C ABI compatibility - no FFI, no overhead. A Zig function calling `cblas_ddot` is as fast as a C function calling it. This matters when you're computing cosine similarity for 1M vectors.

Other benefits:
- **No runtime** - Compiles to pure machine code
- **WASM target** - No runtime overhead means the most lightweight WASM binaries (our core is ~3KB)
- **Comptime** - Specialization at compile time, not runtime

## Running the Benchmarks

Each benchmark runs **30+ seconds** to ensure fair comparison.

```bash
# Build LanceQL
zig build

# Individual benchmarks (each ~30-60 seconds)
./scripts/bench-vector.sh       # GPU vs CPU vector operations
./scripts/bench-sql.sh          # SQL clauses (200M rows)
./scripts/bench-logic-table.sh  # ML workflows (1B rows)

# Run all benchmarks (~10-15 minutes total)
./scripts/bench-all.sh

# Or use zig build targets directly
zig build bench-vector       # Vector operations (GPU vs CPU)
zig build bench-sql          # SQL clauses (LanceQL vs DuckDB vs Polars)
zig build bench-logic-table  # @logic_table ML workflows
```

### Requirements

```bash
# macOS
brew install duckdb
pip install polars numpy

# Linux
# Download DuckDB CLI from https://duckdb.org/docs/installation/
pip install polars numpy
```

## Current Status

- [x] GPU hash table for GROUP BY
- [x] GPU hash JOIN
- [x] Vector search (IVF-PQ index)
- [x] SQL clause benchmarks
- [x] Workflow benchmarks vs DuckDB/Polars
- [x] metal0 Python-to-Zig compiler (working, passing CPython unittests in progress)
- [x] @logic_table decorator implementation
- [x] Edge/WASM deployment (JavaScript via frozen interpreter + WAMR AOT, Python support coming)

## License

Apache-2.0 (same as [Lance](https://github.com/lance-format/lance))
