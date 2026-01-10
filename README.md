# LanceQL

A browser-based Lance file reader with SQL and vector search support. Query Lance datasets directly in your browser using HTTP Range requests - no server required.

**[@logic_table: Business Logic That Runs WITH Your Data](./LOGIC_TABLE.md)** - Compile Python to native GPU kernels. Eliminates Python interpreter overhead for UDF-style workflows.

**Live Demo:** [https://teamchong.github.io/lanceql](https://teamchong.github.io/lanceql)

**Documentation:**
- [SQL Reference](./docs/SQL_REFERENCE.md) - Complete SQL dialect reference
- [Vector Search Guide](./docs/VECTOR_SEARCH.md) - Semantic search with NEAR clause
- [@logic_table](./LOGIC_TABLE.md) - Compile Python to native GPU kernels

## When to Use LanceQL

| Use Case | LanceQL |
|----------|---------|
| Query columnar files in browser | Direct SQL on Lance/Parquet, no backend needed |
| Semantic search on embeddings | Built-in vector search with IVF-PQ indices |
| Python UDFs that are too slow | @logic_table compiles Python to native code |
| Time-travel queries | Query any historical version |
| Query remote data efficiently | HTTP Range requests - only fetch what you need |

## Features

### All Platforms
- **Vector Search** - Semantic search using `WHERE column NEAR [vector]` syntax
- **Time Travel** - Query historical versions with `read_lance(url, version)`
- **Rich Zig Engine** - Core logic moves to Zig for maximum performance

### Browser (WASM + JavaScript)
- **Hybrid Execution** - Automatically switches between Zig (SIMD) and WebGPU for massive datasets
- **SQL** - Full SQL support powered by Zig parser and executor
- **HTTP Range Requests** - Only fetch the bytes you need, not the entire file
- **Local + Remote** - Drag & drop local files or load from URL

### Node.js/Python (Native)
- **High Performance** - Uses native CPU instructions (AVX-512 where available)
- **Full SQL** - `ORDER BY`, `GROUP BY`, `DISTINCT`, all aggregations
- **Data Types** - int32/64, float32/64, bool, string, timestamp (s/ms/us/ns), date32/64
- **Parameterized Queries** - Bind values with `?` placeholders
- **Drop-in APIs** - better-sqlite3 (Node.js), pyarrow.parquet (Python)

## Installation

### Browser (WASM)
```bash
npm install lanceql  # WIP - not yet published
```

### Node.js (Native) - WIP
```bash
npm install @lanceql/node  # WIP - not yet published
```

Drop-in replacement for **better-sqlite3** with Lance columnar files:
```javascript
// Instead of: const Database = require('better-sqlite3');
const Database = require('@lanceql/node');

const db = new Database('dataset.lance');
const rows = db.prepare('SELECT * FROM data WHERE id > ?').all(100);
```

### Python - WIP
```bash
pip install metal0-lanceql  # WIP - not yet published
```

Drop-in replacement for **pyarrow.parquet** with Lance columnar files:
```python
# Instead of: import pyarrow.parquet as pq
import metal0.lanceql as pq

table = pq.read_table('dataset.lance')
df = table.to_pandas()
```

## Quick Start

### Browser Demo
```bash
cd examples/wasm
python -m http.server 3000
# Open http://localhost:3000
```

Default dataset: 1M LAION images with text embeddings at `https://data.metal0.dev/laion-1m/images.lance`

## SQL Examples

See [SQL Reference](./docs/SQL_REFERENCE.md) for complete documentation.

```sql
-- Local uploaded file
SELECT * FROM read_lance(FILE) LIMIT 50
SELECT * FROM read_lance(FILE, 24) LIMIT 50  -- with version

-- Remote URL
SELECT * FROM read_lance('https://data.metal0.dev/laion-1m/images.lance') LIMIT 50
SELECT * FROM read_lance('https://...', 24) LIMIT 50  -- with version

-- Filter
SELECT url, text, aesthetic FROM read_lance(FILE)
WHERE aesthetic > 0.5
LIMIT 100

-- Aggregations
SELECT COUNT(*), AVG(aesthetic), MAX(aesthetic) FROM read_lance(FILE)

-- Vector search
-- NEAR is now a standard operator in the WHERE clause
SELECT * FROM read_lance(FILE) 
WHERE embedding NEAR [0.1, 0.2, ...] 
LIMIT 20

-- Combined filters (pre-filtering supported)
SELECT * FROM read_lance(FILE) 
WHERE aesthetic > 0.5 AND embedding NEAR [0.1, 0.2, ...] 
LIMIT 30
```

See [Vector Search Guide](./docs/VECTOR_SEARCH.md) for IVF-PQ indices, encoders, and performance tuning.

## CLI Commands

### Query

```bash
lanceql query "SELECT * FROM 'data.lance' LIMIT 10"
lanceql query "SELECT * FROM 'data.lance' WHERE id > 100" --json
```

### Time Travel

```bash
# Show version history
lanceql history data.lance
# version | timestamp                 | operation | rowCount | delta
# --------|---------------------------|-----------|----------|------
#       3 | 2024-01-10T10:30:00.000Z  | INSERT    |        9 | +3
#       2 | 2024-01-10T10:25:00.000Z  | INSERT    |        6 | +3
#       1 | 2024-01-10T10:20:00.000Z  | INSERT    |        3 | +3

# Compare versions (shows actual changed rows)
lanceql diff data.lance --from 1 --to 2
# === Diff v1 → v2 ===
# Summary: +3 added, -0 deleted (from 1 fragments)
#
# --- Added rows (3) ---
# change  id
# ADD     4
# ADD     5
# ADD     6

# JSON output
lanceql diff data.lance --from 1 --to 2 --json
```

### Data Pipeline

```bash
# Convert CSV to Lance
lanceql ingest data.csv -o dataset.lance

# Add embeddings
lanceql enrich dataset.lance --embed text --model minilm

# Start web server
lanceql serve dataset.lance
```

## Vector Index (Auto-Encode)

Create indexes that automatically encode text to embeddings:

```sql
-- Create vector index on text column
CREATE VECTOR INDEX ON docs(content) USING minilm

-- With IF NOT EXISTS
CREATE VECTOR INDEX IF NOT EXISTS ON docs(content) USING clip

-- Custom dimension
CREATE VECTOR INDEX ON docs(content) USING minilm WITH (dim = 384)

-- Drop index
DROP VECTOR INDEX ON docs(content)
DROP VECTOR INDEX IF EXISTS ON docs(content)

-- Show indexes
SHOW VECTOR INDEXES
SHOW VECTOR INDEXES ON docs
```

### Supported Models

| Model | Dimensions | Use Case |
|-------|-----------|----------|
| `minilm` | 384 | Text similarity (default) |
| `clip` | 512 | Image/text cross-modal |
| `bge-small` | 384 | BGE embeddings |
| `bge-base` | 768 | BGE embeddings |
| `bge-large` | 1024 | BGE embeddings |
| `openai` | 1536 | OpenAI embeddings |
| `cohere` | 1024 | Cohere embeddings |

### How It Works

1. **CREATE VECTOR INDEX** - Registers index metadata, creates shadow column `__vec_{column}_{model}`
2. **INSERT** - Text values automatically encoded to embeddings in shadow column
3. **NEAR** - Query rewritten to use shadow column for vector search

```sql
-- User writes:
SELECT * FROM docs WHERE content NEAR 'machine learning'

-- Internally rewritten to:
SELECT * FROM docs WHERE __vec_content_minilm NEAR encode('machine learning') TOPK 10
```

## DataFrame Examples

```python
import lanceql

dataset = lanceql.open("https://data.metal0.dev/laion-1m/images.lance")

# Vector search by text
result = (
    dataset.df()
    .search("cat playing", encoder="minilm", top_k=20)
    .select(["url", "text"])
    .collect()
)

# Vector search by row
result = (
    dataset.df()
    .search_by_row(0, column="embedding", top_k=10)
    .collect()
)

# Filter numeric columns
result = (
    dataset.df()
    .filter("aesthetic", ">", 0.6)
    .limit(50)
    .collect()
)

# Time travel
dataset = lanceql.open("https://...", version=24)
```

## Architecture

### Zig Core (CLI + WASM)

```
src/
├── lanceql.zig          # Root module, re-exports all public APIs
├── wasm.zig             # WASM entry point
├── table.zig            # High-level Table API
│
├── sql/                 # SQL Engine
│   ├── lexer.zig        # Tokenizer (DDL, DML, vector extensions)
│   ├── parser.zig       # Recursive descent parser → AST
│   ├── ast.zig          # Statement types (SELECT, CREATE VECTOR INDEX, etc.)
│   ├── executor.zig     # Query execution, vector index storage
│   ├── planner/         # Query optimizer
│   └── codegen/         # JIT compilation (Metal0)
│
├── format/              # File Format Parsers
│   ├── footer.zig       # Lance footer (40 bytes at EOF)
│   ├── lance_file.zig   # Column access, fragments
│   └── version.zig      # V2_0, V2_1 handling
│
├── proto/               # Protobuf Decoder (hand-rolled)
│   ├── decoder.zig      # Wire format (varint, length-delimited)
│   ├── lance_messages.zig  # ColumnMetadata, PageInfo
│   └── schema.zig       # Schema/field parsing
│
├── io/                  # VFS Abstraction Layer
│   ├── reader.zig       # Reader interface (vtable pattern)
│   ├── file_reader.zig  # Native file system
│   ├── memory_reader.zig # In-memory buffers
│   └── http_reader.zig  # HTTP Range requests
│
└── encoding/            # Column Decoders
    ├── plain.zig        # Plain int64/float64
    └── parquet/         # Parquet support
        ├── page.zig     # PLAIN, RLE_DICTIONARY
        ├── snappy.zig   # Snappy (pure Zig, SIMD)
        └── thrift.zig   # TCompactProtocol
```

### Browser Package

```
packages/browser/
├── src/
│   ├── client/              # Main thread APIs
│   │   ├── store/vault.js   # vault() - KV + SQL API
│   │   ├── lance-ql.js      # LanceQL - remote datasets
│   │   ├── lance/           # DataFrame API
│   │   └── webgpu/          # GPU acceleration
│   │       └── gpu-transformer.js  # Text embeddings
│   │
│   └── worker/              # SharedWorker
│       ├── index.js         # RPC router, DDL sync, vector indexes
│       ├── worker-database.js    # OPFS storage, Lance fragments
│       └── wasm-sql-bridge.js    # WASM ↔ JS bridge
│
└── tests/e2e/               # Playwright tests
```

### Data Flow

```
                          ┌──────────────────────────────┐
                          │       User SQL Query         │
                          │  "SELECT * FROM t WHERE ..."  │
                          └──────────────┬───────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
            ┌───────▼───────┐                         ┌───────▼───────┐
            │   CLI (Zig)   │                         │Browser (WASM) │
            │   lexer.zig   │                         │  worker/      │
            │   parser.zig  │                         │  index.js     │
            │   executor.zig│                         │               │
            └───────┬───────┘                         └───────┬───────┘
                    │                                         │
                    │                                         │
            ┌───────▼───────┐                         ┌───────▼───────┐
            │ VFS Reader    │                         │ WASM Bridge   │
            │ (file/HTTP)   │                         │ + WorkerDB    │
            └───────┬───────┘                         └───────┬───────┘
                    │                                         │
                    └─────────────────┬───────────────────────┘
                                      │
                              ┌───────▼───────┐
                              │  Lance Files  │
                              │ (.lance dir)  │
                              └───────────────┘
```

## WASM Runtime

LanceQL uses an **Immer-style Proxy pattern** for WASM interop:

```javascript
// Traditional WASM interop - verbose, error-prone
const ptr = wasm.alloc(str.length);
const mem = new Uint8Array(wasm.memory.buffer);
mem.set(encoder.encode(str), ptr);
const result = wasm.someFunc(ptr, str.length);
wasm.free(ptr);

// Immer-style - auto marshalling via Proxy (like metal0)
const lanceql = await LanceQL.load('./lanceql.wasm');
lanceql.someFunc("hello");       // strings auto-copied to WASM memory
lanceql.parseData(bytes);        // Uint8Array auto-copied too
lanceql.raw.someFunc(ptr, len);  // raw access when needed
lanceql.memory;                  // WASM memory
```

**How it works:**

```javascript
// ~30 lines of runtime code handles all marshalling
const proxy = new Proxy({}, {
    get(_, name) {
        if (typeof wasm[name] === 'function') {
            return (...args) => wasm[name](...args.flatMap(marshal));
        }
        return wasm[name];
    }
});

// Marshal function - auto-converts strings and Uint8Array
const marshal = arg => {
    if (arg instanceof Uint8Array) {
        // Copy bytes to WASM memory, return [ptr, len]
        buffer.set(arg); return [ptr, arg.length];
    }
    if (typeof arg === 'string') {
        // Encode string to WASM memory, return [ptr, len]
        const bytes = encoder.encode(arg);
        buffer.set(bytes); return [ptr, bytes.length];
    }
    return [arg];  // Numbers pass through
};
```

**Benefits:**
- **Zero boilerplate** - No manual `alloc`/`free`/copy for each call
- **Auto marshalling** - Strings and Uint8Array automatically copied to WASM memory
- **Tiny runtime** - ~30 lines, no dependencies
- **JS debugging** - All logic stays in JS where DevTools works

## Build

Requires [Zig](https://ziglang.org/download/) 0.13.0+

```bash
# Build WASM module
zig build wasm
# Output: zig-out/bin/lanceql.wasm

# Copy to demo
cp zig-out/bin/lanceql.wasm examples/wasm/

# Run tests
zig build test
```

## Usage

```html
<script type="module">
import { LanceQL } from './lanceql.js';

const lanceql = await LanceQL.load('./lanceql.wasm');

// Open remote dataset
const dataset = await lanceql.openDataset('https://data.metal0.dev/laion-1m/images.lance');

// Query
const strings = await dataset.readStrings(0, 50);  // First 50 rows of column 0
</script>
```

For TypeScript, the `lanceql.d.ts` file provides type definitions:

```typescript
import { LanceQL } from './lanceql.js';
// Types are automatically picked up from lanceql.d.ts
```

## Performance Optimizations

- **Zero-copy Arrow** - Direct memory sharing via Arrow C Data Interface
- **Metal GPU (Apple Silicon)** - Zero-copy unified memory, auto-switch at 100K+ vectors
  - Batch cosine similarity: 14.3M vectors/sec (1M × 384-dim)
  - Shaders compile at runtime - no Xcode required
- **Accelerate vDSP (macOS)** - SIMD-optimized for small batches and single-vector ops
  - Dot product: 42 ns/op (384-dim)
  - Cosine similarity: 106 ns/op (384-dim)
- **Comptime SIMD** - 32-byte vectors, bit-width specialization (1-20 bits)
- **IndexedDB Cache** - Schema and column types cached for repeat visits
- **Sidecar Manifest** - Optional `.meta.json` for faster startup
- **Fragment Prefetching** - Parallel metadata loading on dataset open
- **Speculative Prefetch** - Next page loaded in background

### Vector Search Performance (384 dims)

| Scale | Path | Throughput |
|-------|------|------------|
| 10K vectors | CPU (Accelerate) | 13.9M vec/s |
| 100K vectors | GPU (Metal) | 13.5M vec/s |
| 1M vectors | GPU (Metal) | 14.3M vec/s |

*Apple Silicon only - Intel Macs use CPU path (Accelerate still fast)*

### Platform Detection (comptime)

```zig
const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;
const is_apple_silicon = is_macos and builtin.cpu.arch == .aarch64;
// Auto-switch: GPU at 100K+ vectors on Apple Silicon
```

Run vector benchmark: `zig build bench-vector`

Generate sidecar manifest:
```bash
cd fixtures
# Local dataset
python generate_sidecar.py /path/to/dataset.lance

# Remote S3/R2 dataset (requires aws profile 'r2')
python generate_sidecar.py s3://bucket/dataset.lance ./meta.json
aws s3 cp ./meta.json s3://bucket/dataset.lance/.meta.json --profile r2 --endpoint-url https://...
```

## Format Support

| Format | Function | Features |
|--------|----------|----------|
| **Lance** | `read_lance()` | v2.0/v2.1, IVF-PQ indices, time travel, deletion vectors |
| **Parquet** | `read_parquet()` | Pure Zig, Snappy, RLE/PLAIN/DICTIONARY encoding |
| **Delta Lake** | `read_delta()` | Parquet + transaction log |
| **Iceberg** | `read_iceberg()` | Parquet + metadata layer |
| **Arrow IPC** | `read_arrow()` | .arrow, .arrows, .feather files |
| **Avro** | `read_avro()` | Deflate/Snappy compression |
| **ORC** | `read_orc()` | Snappy compression |
| **Excel** | `read_xlsx()` | Multi-sheet support |

**Supported Data Types:** int32/64, float32/64, bool, string, timestamp[s/ms/us/ns], date32/64

See [SQL Reference](./docs/SQL_REFERENCE.md) for complete data source documentation.

## License

Apache-2.0 (same as [Lance](https://github.com/lance-format/lance))
