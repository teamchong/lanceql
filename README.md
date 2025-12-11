# LanceQL

A browser-based Lance file reader with SQL and vector search support. Query Lance datasets directly in your browser using HTTP Range requests - no server required.

**Live Demo:** [https://teamchong.github.io/lanceql](https://teamchong.github.io/lanceql)

## Features

- **SQL Queries** - `SELECT`, `WHERE`, `LIMIT`, aggregations (`COUNT`, `SUM`, `AVG`)
- **Vector Search** - Semantic search with `NEAR` clause using MiniLM/CLIP embeddings
- **Time Travel** - Query historical versions with `read_lance(url, version)`
- **DataFrame API** - Python-like syntax: `dataset.df().filter(...).select(...).limit(50)`
- **HTTP Range Requests** - Only fetch the bytes you need, not the entire file
- **Local + Remote** - Drag & drop local files or load from URL
- **Column Statistics** - Click column chips to see min/max/avg/null count

## Quick Start

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
└── encoding/            # Column decoders (plain, dictionary)

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

// Immer-style - auto marshalling via Proxy
const lanceql = await LanceQL.load('./lanceql.wasm');
lanceql._proxy.someFunc("hello");  // strings auto-copied to WASM memory
lanceql._proxy.parseData(bytes);   // Uint8Array auto-copied too
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
- **Type-safe** - Strings and bytes automatically handled
- **Tiny runtime** - ~30 lines, no dependencies
- **JS debugging** - All logic stays in JS where DevTools works

## Build

```bash
# Build WASM module
zig build wasm

# Copy to examples
cp zig-out/bin/lanceql.wasm examples/wasm/

# Run tests
zig build test
```

## Performance Optimizations

- **IndexedDB Cache** - Schema and column types cached for repeat visits
- **Sidecar Manifest** - Optional `.meta.json` for faster startup
- **Fragment Prefetching** - Parallel metadata loading on dataset open
- **Speculative Prefetch** - Next page loaded in background

Generate sidecar manifest:
```bash
cd fixtures
python generate_sidecar.py /path/to/dataset.lance
```

## Lance Format Support

- Lance v2.0 and v2.1 file format
- Multi-fragment datasets with manifest discovery
- IVF-PQ vector indices
- Deletion vectors (logical deletes)
- Version/time-travel queries

## License

Apache-2.0 (same as [Lance](https://github.com/lance-format/lance))
