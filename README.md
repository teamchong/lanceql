# LanceQL

A browser-based Lance file reader with SQL and vector search support. Query Lance datasets directly in your browser using HTTP Range requests - no server required.

**[@logic_table: Business Logic That Runs WITH Your Data](./LOGIC_TABLE.md)** - Compile Python to native GPU kernels. Eliminates Python interpreter overhead for UDF-style workflows.

**Live Demo:** [https://teamchong.github.io/lanceql](https://teamchong.github.io/lanceql)

## Features

### All Platforms
- **Vector Search** - Semantic search with `NEAR` clause using MiniLM/CLIP embeddings
- **Time Travel** - Query historical versions with `read_lance(url, version)`

### Browser (WASM + JavaScript)
- **SQL** - `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, aggregations (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`)
- **HTTP Range Requests** - Only fetch the bytes you need, not the entire file
- **Local + Remote** - Drag & drop local files or load from URL
- **DataFrame API** - `dataset.df().filter(...).select(...).limit(50)`

### Node.js/Python (Native)
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

-- Vector search by text (TOPK optional, default 20)
SELECT * FROM read_lance(FILE) NEAR 'sunset beach'
SELECT * FROM read_lance(FILE) NEAR 'cat' TOPK 50

-- Vector search by row
SELECT * FROM read_lance(FILE) NEAR 0 TOPK 10

-- Specify vector column
SELECT * FROM read_lance(FILE) NEAR embedding 'sunset'

-- Combined with WHERE
SELECT * FROM read_lance(FILE)
WHERE aesthetic > 0.5
NEAR 'beach sunset' TOPK 30
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

```
src/
├── lanceql.zig          # Zig WASM module for Lance parsing
├── format/              # Lance file format (footer, columns)
├── proto/               # Protobuf decoder for manifests
├── io/                  # VFS abstraction (file, memory, HTTP)
└── encoding/
    ├── plain.zig        # Lance column decoders
    └── parquet/         # Parquet file reader
        ├── page.zig     # Page decoder (PLAIN, RLE_DICTIONARY)
        ├── snappy.zig   # Snappy decompression (pure Zig, SIMD)
        └── thrift.zig   # TCompactProtocol decoder

examples/wasm/
├── index.html           # Demo UI
└── lanceql.js           # JS wrapper, SQL parser, vector search
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

### Lance Files
- Lance v2.0 and v2.1 file format
- Multi-fragment datasets with manifest discovery
- IVF-PQ vector indices
- Deletion vectors (logical deletes)
- Version/time-travel queries
- Data types: int32/64, float32/64, bool, string, timestamp[s/ms/us/ns], date32/64

### Parquet Files
- Pure Zig implementation (zero external dependencies)
- Thrift TCompactProtocol metadata decoder
- Encodings: PLAIN, RLE, RLE_DICTIONARY
- Compression: Uncompressed, Snappy
- Data types: boolean, int32/64, float/double, byte_array, fixed_len_byte_array

## License

Apache-2.0 (same as [Lance](https://github.com/lance-format/lance))
