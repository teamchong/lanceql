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
        # NumPy ops compiled to direct BLAS calls (cblas_ddot, cblas_dgemm)
        normalized = numpy.dot(features, weights)
        return numpy.mean(normalized)
```

The `@logic_table` classes are compiled **ahead-of-time** by metal0. LanceQL loads the compiled functions and fuses them with query execution.

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

## Protocol: Python ↔ Query Engine Contract

The `@logic_table` protocol defines how Python code and the SQL engine communicate:

### Schema Declaration

```python
from lanceql.protocol import logic_table, DataType, ColumnSchema, TableSchema

@logic_table(
    input_schema={"amount": float, "vendor": str, "user_id": int},
    output_schema={"is_fraud": bool, "confidence": float, "risk_level": str}
)
def fraud_detector(batch: pa.RecordBatch) -> pa.RecordBatch:
    """
    Input:  amount (float64), vendor (string), user_id (int64)
    Output: is_fraud (bool), confidence (float64), risk_level (string)
    """
    amounts = batch.column("amount").to_numpy()
    is_fraud = amounts > 10000
    confidence = np.minimum(amounts / 10000, 1.0)
    risk_level = np.where(confidence > 0.8, "high", "medium")
    return pa.RecordBatch.from_pydict({
        "is_fraud": is_fraud,
        "confidence": confidence,
        "risk_level": risk_level
    })
```

### Supported Data Types

| Type | Python | Arrow | Description |
|------|--------|-------|-------------|
| `INT64` | `int` | `pa.int64()` | 64-bit integer |
| `FLOAT64` | `float` | `pa.float64()` | 64-bit float |
| `STRING` | `str` | `pa.string()` | UTF-8 string |
| `BOOL` | `bool` | `pa.bool_()` | Boolean |
| `TIMESTAMP` | `datetime` | `pa.timestamp('us')` | Microsecond timestamp |
| `BINARY` | `bytes` | `pa.binary()` | Binary data |
| `VECTOR_F32` | `np.ndarray` | `pa.list_(pa.float32())` | Float32 vector |
| `VECTOR_F64` | `np.ndarray` | `pa.list_(pa.float64())` | Float64 vector |

### Window Function Support

Logic tables support SQL window functions via `OVER(PARTITION BY ...)`:

```sql
-- Partition-parallel processing
SELECT fraud_detector(amount, vendor) OVER(PARTITION BY user_id)
FROM transactions

-- With ordering (for time-series)
SELECT anomaly_score(value) OVER(PARTITION BY sensor_id ORDER BY timestamp)
FROM readings

-- Ranking within partitions
SELECT ROW_NUMBER() OVER(PARTITION BY category ORDER BY score DESC)
FROM products
```

Window specification passed to logic table:
```python
@dataclass
class WindowSpec:
    partition_by: List[str]       # PARTITION BY columns
    order_by: List[tuple]         # [(column, 'asc'|'desc'), ...]
    frame_type: str               # 'rows' | 'range'
    frame_start: Any              # Frame start bound
    frame_end: Any                # Frame end bound
```

### Available Window Functions

| Category | Functions | Description |
|----------|-----------|-------------|
| **Ranking** | `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `NTILE(n)` | Row numbering and ranking |
| **Offset** | `LAG(col, n)`, `LEAD(col, n)`, `FIRST_VALUE()`, `LAST_VALUE()` | Access other rows |
| **Aggregate** | `SUM()`, `AVG()`, `COUNT()`, `MIN()`, `MAX()` | Running aggregates |
| **Time** | `TUMBLE()`, `HOP()`, `SESSION()` | Time-based windows |

### Vector-Specific Logic Tables

For ML workloads with embeddings:

```python
@vector_logic_table(
    vector_column="embedding",
    dimensions=384,
    output_schema={"similarity": float, "cluster_id": int}
)
def cluster_vectors(vectors: np.ndarray) -> Dict[str, np.ndarray]:
    """Input: (N, 384) float32 array"""
    similarities = cosine_sim(vectors, centroids)
    cluster_ids = kmeans.predict(vectors)
    return {"similarity": similarities, "cluster_id": cluster_ids}
```

### Time Window Logic Tables

For streaming/time-series:

```python
@time_window_logic_table(
    time_window=TimeWindow(
        timestamp_column="event_time",
        window_duration="1 hour",
        slide_duration="15 minutes"  # Hopping window
    ),
    input_schema={"value": float},
    output_schema={"window_start": datetime, "avg_value": float}
)
def hourly_stats(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Process all events in one time window"""
    ...
```

## Implementation Architecture

### Core Files

```
src/logic_table/
├── logic_table.zig      # Runtime: LogicTableContext, QueryContext, FilterPredicate
├── protocol.py          # Python protocol definitions
└── vector_ops.zig       # metal0-compiled vector operations

src/sql/
├── ast.zig              # AST with WindowSpec support
├── parser.zig           # SQL parser with OVER clause
├── executor.zig         # Query executor
├── column_deps.zig      # Column dependency extraction
└── batch_codegen.zig    # Batch operation code generation

src/query/
└── logic_table.zig      # Query-focused context with GPU dispatch
```

### Execution Flow

```
SQL Query                                     Result
    │                                            ▲
    ▼                                            │
┌─────────────────────────────────────────────────────┐
│ Parser: Extract function calls + window specs       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Column Deps: Identify required columns for @logic_table │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Batch CodeGen: Generate GPU/SIMD dispatch code      │
│   - GPU_THRESHOLD: 10K rows                         │
│   - SIMD_THRESHOLD: 16 elements                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ LogicTableContext: Bind columns to @logic_table     │
│   - Load only needed columns                        │
│   - Pass WHERE predicates for pushdown              │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Execute: Run compiled @logic_table function         │
│   - Metal GPU (macOS)                               │
│   - CUDA (Linux)                                    │
│   - SIMD (CPU fallback)                             │
└─────────────────────────────────────────────────────┘
```

### GPU Dispatch Thresholds

| Threshold | Dispatch | Use Case |
|-----------|----------|----------|
| ≥10K rows | GPU | Large batch vector ops |
| ≥16 elements | SIMD | Small vectors, CPU-bound |
| <16 elements | Scalar | Single values |

### Predicate Pushdown

WHERE clause predicates are exposed to `@logic_table` functions:

```python
# SQL: SELECT fraud_score(...) FROM t WHERE amount > 1000

@logic_table
class FraudDetector:
    def fraud_score(self, ctx: QueryContext):
        # ctx.predicates contains: [FilterPredicate(col="amount", op=">", value=1000)]
        # Skip processing rows that don't match!
        if ctx.has_predicate("amount"):
            threshold = ctx.get_predicate_value("amount")
            # Early exit for filtered rows
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
- [x] Window function syntax (OVER, PARTITION BY, ORDER BY, frame specs)
- [x] Ranking functions (ROW_NUMBER, RANK, DENSE_RANK, NTILE)
- [x] Offset functions (LAG, LEAD, FIRST_VALUE, LAST_VALUE)
- [x] Time window functions (TUMBLE, HOP, SESSION)
- [x] Python ↔ Query protocol with Arrow schema

## License

Apache-2.0 (same as [Lance](https://github.com/lance-format/lance))
