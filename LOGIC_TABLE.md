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

## @logic_table: The Solution

A `@logic_table` is a **Python class that becomes a virtual table** in SQL. It's NOT a UDF (user-defined function) - it's a table source with computed columns.

```python
# fraud_detector.py
from lanceql import logic_table, Table

@logic_table
class FraudDetector:
    """Fraud detection logic combining order and customer data."""

    # Data sources - query engine loads these automatically
    orders = Table('orders.lance')
    customers = Table('customers.lance')

    # Constants
    HIGH_AMOUNT_THRESHOLD = 10000

    def amount_score(self) -> float:
        """Score based on order amount."""
        if self.orders.amount > self.HIGH_AMOUNT_THRESHOLD:
            return min(1.0, self.orders.amount / 50000)
        return 0.0

    def customer_score(self) -> float:
        """Score based on customer risk factors."""
        score = 0.0
        if self.customers.days_since_signup < 30:
            score += 0.3
        if self.customers.previous_fraud:
            score += 0.5
        return min(1.0, score)

    def risk_score(self) -> float:
        """Combined risk score (0-1)."""
        return self.amount_score() * 0.5 + self.customer_score() * 0.5

    def risk_category(self) -> str:
        """Categorize risk level."""
        score = self.risk_score()
        if score > 0.8: return 'critical'
        if score > 0.6: return 'high'
        if score > 0.3: return 'medium'
        return 'low'

    def should_block(self) -> bool:
        """Whether to block this transaction."""
        return self.risk_score() > 0.8
```

### SQL Usage

The `@logic_table` class is used in **FROM or JOIN**, not as a function call:

```sql
-- Basic usage: logic_table as table source
WITH DATA (
    orders = 'orders.lance',
    customers = 'customers.lance'
)
SELECT
    orders.order_id,
    orders.amount,
    t.risk_score(),
    t.risk_category()
FROM logic_table('fraud_detector.py') AS t
WHERE t.risk_score() > 0.7
ORDER BY t.risk_score() DESC
LIMIT 100
```

### Methods in Different SQL Clauses

Every method can be used in ANY SQL clause:

```sql
-- SELECT: computed columns
SELECT t.risk_score(), t.risk_category(), t.should_block()

-- WHERE: filter by method result
WHERE t.risk_score() > 0.5 AND NOT t.should_block()

-- ORDER BY: sort by method result
ORDER BY t.risk_score() DESC

-- GROUP BY: group by method result
GROUP BY t.risk_category()

-- HAVING: filter groups
HAVING AVG(t.risk_score()) > 0.6

-- PARTITION BY: window function partitioning
SUM(amount) OVER(PARTITION BY t.risk_category() ORDER BY t.risk_score())
```

### With Table Alias or Without

```sql
-- With alias
FROM logic_table('fraud_detector.py') AS fraud
WHERE fraud.risk_score() > 0.7

-- Without alias (methods called directly)
FROM logic_table('fraud_detector.py')
WHERE risk_score() > 0.7

-- Join with other tables
SELECT o.*, fraud.risk_score()
FROM orders o
JOIN logic_table('fraud_detector.py') AS fraud
  ON fraud.orders.id = o.id
WHERE fraud.should_block() = false
```

## Key Difference from UDFs

### Traditional UDF (Black Box)

```python
# UDF is called for EACH ROW, no query context
def fraud_udf(amount, customer_id):
    # Can't see WHERE clause
    # Can't skip filtered rows
    # Can't batch operations
    return calculate_score(amount, customer_id)
```

```sql
-- DuckDB/Polars: calls Python for EACH ROW
SELECT fraud_udf(amount, customer_id) FROM orders
-- 100μs+ overhead per row!
```

### @logic_table (Query-Aware)

```python
@logic_table
class FraudDetector:
    def risk_score(self):
        # Has access to QueryContext:
        # - Which rows passed WHERE clause
        # - Pushdown predicates
        # - Can skip filtered rows
        # - Batch operations
        return self.amount_score() + self.customer_score()
```

```sql
-- LanceQL: compiled, runs ON data
SELECT t.risk_score()
FROM logic_table('fraud_detector.py') t
WHERE amount > 1000
-- t.risk_score() only computed for rows where amount > 1000!
```

## Query Context Access

Unlike UDFs, `@logic_table` methods have access to query execution context:

### FilterPredicate (Pushdown from WHERE)

```python
@logic_table
class SmartProcessor:
    def optimized_score(self, ctx: QueryContext):
        # See what's in WHERE clause
        if ctx.has_predicate("amount"):
            threshold = ctx.get_predicate_value("amount")
            # Skip rows below threshold early
            if self.orders.amount < threshold:
                return 0.0  # Early exit

        return self.expensive_calculation()
```

### Filtered Indices

```python
def batch_process(self, ctx: QueryContext):
    # Only process rows that passed WHERE clause
    indices = ctx.filtered_indices  # [5, 10, 15, ...] or None

    if indices is not None:
        # Process only matching rows
        for i in indices:
            yield self.compute(i)
    else:
        # No filter, process all
        for i in range(ctx.total_rows):
            yield self.compute(i)
```

### Selectivity-Based Algorithm Choice

```python
def smart_join(self, ctx: QueryContext):
    selectivity = ctx.get_selectivity()  # matched/total

    if selectivity < 0.01:
        # Very selective: use index lookup
        return self.index_lookup()
    elif selectivity < 0.1:
        # Somewhat selective: use hash join
        return self.hash_join()
    else:
        # Not selective: use merge join
        return self.merge_join()
```

## Window Functions with @logic_table

Methods can be used in PARTITION BY for parallel processing:

```sql
-- Partition by a computed column
SELECT
    order_id,
    amount,
    SUM(amount) OVER(PARTITION BY t.risk_category() ORDER BY t.risk_score())
FROM logic_table('fraud_detector.py') t

-- Use method result to define partitions
SELECT
    ROW_NUMBER() OVER(PARTITION BY t.should_review() ORDER BY t.risk_score() DESC)
FROM logic_table('fraud_detector.py') t
```

### Time-Based Windows

```sql
-- Session windows based on method
SELECT
    t.session_id(),
    SUM(amount) OVER(PARTITION BY t.session_id())
FROM logic_table('session_tracker.py') t

-- Tumbling windows
SELECT
    t.time_bucket('1 hour'),
    AVG(t.metric())
FROM logic_table('metrics.py') t
GROUP BY t.time_bucket('1 hour')
```

## Life Cycle

### Instance Life Cycle

A `@logic_table` instance follows this life cycle:

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUERY EXECUTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. COMPILE (once)                                              │
│     ├─ Parse Python @logic_table class                          │
│     ├─ Extract Table dependencies                               │
│     ├─ Compile methods to Zig IR → native code                  │
│     └─ Register in LogicTableRegistry                           │
│                                                                 │
│  2. INIT (per query)                                            │
│     ├─ Create QueryContext                                      │
│     │   ├─ total_rows, matched_rows                             │
│     │   ├─ filtered_indices (from WHERE)                        │
│     │   ├─ predicates (pushdown)                                │
│     │   └─ result_cache (memoization)                           │
│     │                                                           │
│     └─ Create LogicTableContext                                 │
│         ├─ Link to QueryContext                                 │
│         └─ Initialize column caches                             │
│                                                                 │
│  3. BIND (per table in FROM/JOIN)                               │
│     ├─ Load required columns from Lance files                   │
│     ├─ ctx.bindF32("orders", "amount", [...])                   │
│     └─ ctx.bindI64("customers", "days_since_signup", [...])     │
│                                                                 │
│  4. EXECUTE (per method call)                                   │
│     ├─ Check result_cache for memoized result                   │
│     ├─ If miss: compute using bound columns                     │
│     ├─ GPU dispatch for ≥10K rows                               │
│     ├─ SIMD for 16-10K rows                                     │
│     ├─ Scalar for <16 rows                                      │
│     └─ Cache result for repeated calls                          │
│                                                                 │
│  5. CLEANUP (query end)                                         │
│     ├─ Free result_cache entries                                │
│     ├─ Free QueryContext                                        │
│     └─ Free LogicTableContext                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Code Example

```zig
// Life cycle in executor
pub fn executeQuery(sql: []const u8) !Result {
    var allocator = std.heap.page_allocator;

    // 2. INIT
    var query_ctx = QueryContext.init(allocator);
    defer query_ctx.deinit();  // 5. CLEANUP

    var ctx = LogicTableContext.initWithQuery(allocator, &query_ctx);
    defer ctx.deinit();  // 5. CLEANUP

    // 3. BIND (after parsing SQL and loading tables)
    try ctx.bindF32("orders", "amount", orders_amount_data);
    try ctx.bindI64("customers", "days_since_signup", customer_data);

    // Set query context (after WHERE evaluation)
    query_ctx.total_rows = 100000;
    query_ctx.matched_rows = 1500;
    query_ctx.filtered_indices = filtered;

    // 4. EXECUTE methods on filtered rows
    for (query_ctx.filtered_indices) |idx| {
        const score = compute_risk_score(&ctx, idx);
        // ...
    }
}
```

## Context Sharing

### Shared State Between Methods

All methods in a `@logic_table` share the same `LogicTableContext`, enabling:

1. **Column Data Sharing**: Bound once, used by all methods
2. **Result Memoization**: Cache expensive computations
3. **Query Context**: All methods see same filtered indices

```python
@logic_table
class FraudDetector:
    orders = Table('orders.lance')

    def amount_score(self) -> float:
        # Accesses shared column data
        return min(1.0, self.orders.amount / 50000)

    def velocity_score(self) -> float:
        # Same column data, different computation
        return self.orders.amount / self.orders.avg_amount

    def risk_score(self) -> float:
        # Calls other methods - results may be memoized
        return self.amount_score() * 0.5 + self.velocity_score() * 0.5
```

### Context Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│ QueryContext (per query)                                        │
│   ├─ total_rows: 100,000                                        │
│   ├─ matched_rows: 1,500                                        │
│   ├─ filtered_indices: [5, 12, 45, ...]                         │
│   ├─ predicates: [{column: "amount", op: ">", value: 1000}]     │
│   └─ result_cache: {"risk_score_row_5": 0.85, ...}              │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ LogicTableContext (per logic_table in FROM/JOIN)        │   │
│   │   ├─ query_context: *QueryContext (shared reference)    │   │
│   │   ├─ column_cache_f32: {"orders.amount": [...]}         │   │
│   │   └─ column_cache_i64: {"customers.days_since": [...]}  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ LogicTableContext (another logic_table)                 │   │
│   │   └─ query_context: *QueryContext (SAME reference)      │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Cross-Table Joins

When multiple `@logic_table` classes are joined, they share the same `QueryContext`:

```sql
SELECT
    f.risk_score(),
    p.priority_score()
FROM logic_table('fraud_detector.py') AS f
JOIN logic_table('priority_scorer.py') AS p
  ON f.orders.id = p.orders.id
WHERE f.risk_score() > 0.5
```

Both `f` and `p` see:
- Same `filtered_indices` (rows passing WHERE)
- Same `predicates` (pushdown from WHERE)
- Shared `result_cache` (cross-table memoization)

## External Cache Provider

### Why External Caching?

The default `result_cache` in `QueryContext` is in-memory and query-scoped. For:
- **Cross-query caching**: Reuse expensive computations across queries
- **Persistent caching**: Survive process restarts
- **Distributed caching**: Share across workers/nodes

You need an external cache provider.

### Cache Provider Interface

```python
# Python interface (for @logic_table authors)
from lanceql import CacheProvider

class RedisCacheProvider(CacheProvider):
    """External cache using Redis."""

    def __init__(self, redis_url: str, ttl: int = 3600):
        self.client = redis.from_url(redis_url)
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached value, return None if miss."""
        data = self.client.get(key)
        return pickle.loads(data) if data else None

    def set(self, key: str, value: Any) -> None:
        """Cache a value with TTL."""
        self.client.setex(key, self.ttl, pickle.dumps(value))

    def invalidate(self, pattern: str) -> None:
        """Invalidate keys matching pattern."""
        for key in self.client.scan_iter(pattern):
            self.client.delete(key)

@logic_table(cache=RedisCacheProvider("redis://localhost:6379"))
class ExpensiveProcessor:
    data = Table('large_dataset.lance')

    def expensive_score(self) -> float:
        # Result cached in Redis, survives across queries
        return self.complex_calculation()
```

### Zig Interface

```zig
/// External cache provider interface
pub const CacheProvider = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        get: *const fn (ptr: *anyopaque, key: []const u8) ?CachedResult,
        set: *const fn (ptr: *anyopaque, key: []const u8, value: CachedResult) void,
        invalidate: *const fn (ptr: *anyopaque, pattern: []const u8) void,
    };

    pub fn get(self: CacheProvider, key: []const u8) ?CachedResult {
        return self.vtable.get(self.ptr, key);
    }

    pub fn set(self: CacheProvider, key: []const u8, value: CachedResult) void {
        self.vtable.set(self.ptr, key, value);
    }
};

/// QueryContext with external cache support
pub const QueryContext = struct {
    // ... existing fields ...

    /// In-memory cache (query-scoped)
    result_cache: std.StringHashMap(CachedResult),

    /// External cache provider (optional, cross-query)
    external_cache: ?CacheProvider,

    pub fn getCached(self: *Self, key: []const u8) ?CachedResult {
        // Check in-memory first
        if (self.result_cache.get(key)) |result| {
            return result;
        }
        // Fall back to external cache
        if (self.external_cache) |ext| {
            if (ext.get(key)) |result| {
                // Promote to in-memory for this query
                self.result_cache.put(key, result) catch {};
                return result;
            }
        }
        return null;
    }
};
```

### Built-in Cache Providers

| Provider | Scope | Use Case |
|----------|-------|----------|
| `InMemoryCache` | Query | Default, fast, no persistence |
| `IndexedDBCache` | Browser session | WASM, survives page refresh |
| `FileCache` | Process | CLI tools, local persistence |
| `RedisCache` | Cluster | Distributed, shared state |

### Browser (IndexedDB) Cache

For WASM deployments, we use IndexedDB:

```javascript
// Already implemented in src/core/cache.js
import { metadataCache } from './cache.js';

// Cache schema and metadata
await metadataCache.set(datasetUrl, {
    schema: schema,
    columnTypes: types,
    fragments: fragmentInfo
});

// Retrieve on next visit
const cached = await metadataCache.get(datasetUrl);
if (cached) {
    // Skip expensive schema parsing
}
```

### Cache Key Generation

Cache keys include context to ensure correctness:

```python
def generate_cache_key(method_name: str, ctx: QueryContext) -> str:
    """Generate cache key including query context."""
    components = [
        method_name,
        f"rows:{ctx.total_rows}",
        f"matched:{ctx.matched_rows}",
        # Include predicate hash if method depends on it
        f"pred:{hash(tuple(ctx.predicates))}",
    ]
    return ":".join(components)

# Example keys:
# "risk_score:rows:100000:matched:1500:pred:a1b2c3"
# "amount_score:rows:100000:matched:1500:pred:a1b2c3"
```

## Architecture

### Compilation Flow

```
Python Code                    metal0 Compiler                  LanceQL Runtime
    |                               |                                |
@logic_table -----> Parse AST ----> Zig IR -----> Native Code ----> GPU Execution
class Fraud...      Extract         Generate       Compile to        Fused with
                    deps            functions      GPU kernels       data scan
```

### Runtime Execution

```
SQL Query                                     Result
    │                                            ▲
    ▼                                            │
┌─────────────────────────────────────────────────────┐
│ Parser: Identify logic_table() in FROM/JOIN        │
│         Extract method calls (t.risk_score(), etc) │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Column Deps: Analyze which Table columns needed    │
│   FraudDetector.risk_score() needs:                │
│   - orders.amount                                   │
│   - customers.days_since_signup                    │
│   - customers.previous_fraud                       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Load Data: Only load required columns              │
│   ctx.bindF32("orders", "amount", [...])           │
│   ctx.bindI64("customers", "days_since_signup")    │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Execute WHERE: Get filtered indices                │
│   Push predicates to QueryContext                  │
│   filtered_indices = [5, 12, 45, ...]              │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Execute Methods: Run compiled @logic_table code    │
│   - Only on filtered_indices (not all rows!)       │
│   - GPU for 10K+ rows                              │
│   - SIMD for smaller batches                       │
└─────────────────────────────────────────────────────┘
```

### GPU Dispatch Thresholds

| Threshold | Dispatch | Use Case |
|-----------|----------|----------|
| ≥10K rows | GPU | Large batch operations |
| ≥16 elements | SIMD | Vector operations |
| <16 elements | Scalar | Small data |

## Implementation Files

```
src/logic_table/
├── logic_table.zig      # LogicTableContext, QueryContext, FilterPredicate
│                        # CachedResult, LogicTableRegistry
└── vector_ops.zig       # metal0-compiled example (VectorOps, FeatureEngineering)

src/query/
└── logic_table.zig      # Batch vector operations with GPU/SIMD dispatch

src/sql/
├── ast.zig              # AST with WindowSpec for OVER clause
├── parser.zig           # Parse logic_table() in FROM, method calls
├── executor.zig         # Execute with QueryContext
├── column_deps.zig      # Extract column dependencies from methods
└── batch_codegen.zig    # Generate GPU/SIMD dispatch code

src/core/
└── cache.js             # IndexedDB MetadataCache for browser (WASM)

examples/python/
└── fraud_detector.py    # Complete @logic_table example
```

## Benchmarks (Apple M2 Pro)

### @logic_table vs UDF Performance (100K rows × 384 dims)

| Method | Total (ms) | Per Row (μs) | Notes |
|--------|------------|--------------|-------|
| LanceQL @logic_table | ~10 | 0.1 | Compiled, pushdown |
| DuckDB Python UDF | 10,102 | 101 | Row-by-row Python calls |
| DuckDB → Python Batch | 1,805 | 18 | Pull then process |
| Polars .map_elements() | 9,376 | 94 | Row-by-row Python calls |

**Key insight**: @logic_table is 100-1000x faster than Python UDFs because:
1. No Python interpreter overhead
2. Only processes filtered rows (pushdown)
3. GPU/SIMD acceleration
4. No serialization/deserialization

## Current Status

- [x] @logic_table class decorator
- [x] Table data source declarations
- [x] Method to SQL clause mapping (SELECT, WHERE, ORDER BY, GROUP BY)
- [x] QueryContext with pushdown predicates
- [x] FilteredIndices for row skipping
- [x] LogicTableRegistry for compiled classes
- [x] metal0 Python-to-Zig compiler
- [x] GPU dispatch (Metal/CUDA)
- [x] Window function syntax (OVER, PARTITION BY, ORDER BY)
- [x] WASM deployment

## License

Apache-2.0 (same as [Lance](https://github.com/lance-format/lance))
