# LanceQL

A browser-based Lance file reader with SQL and vector search support. Query Lance datasets directly in your browser using HTTP Range requests - no server required.

**Live Demo:** [https://teamchong.github.io/lanceql](https://teamchong.github.io/lanceql)

## Features

- **SQL Queries** - `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, aggregations (`COUNT`, `SUM`, `AVG`)
- **Vector Search** - Semantic search using MiniLM text embeddings with IVF index support
- **Time Travel** - Query historical versions with `AT VERSION N`
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
-- Basic query
SELECT * FROM read_lance('https://data.metal0.dev/laion-1m/images.lance') LIMIT 50

-- Filter and sort
SELECT url, text, aesthetic FROM read_lance('...')
WHERE aesthetic > 0.5
ORDER BY aesthetic DESC
LIMIT 100

-- Aggregations
SELECT COUNT(*), AVG(aesthetic), MAX(aesthetic) FROM read_lance('...')

-- Vector search
SELECT * FROM read_lance('...') SEARCH 'sunset beach' LIMIT 20

-- Time travel
SELECT * FROM read_lance('...') AT VERSION 1 LIMIT 50
```

## DataFrame Examples

```python
import lanceql

dataset = lanceql.open("https://data.metal0.dev/laion-1m/images.lance")

# Vector search
result = (
    dataset.df()
    .search("cat playing", encoder="minilm")
    .select(["url", "text"])
    .limit(20)
    .collect()
)

# Filter numeric columns
result = (
    dataset.df()
    .filter("aesthetic", ">", 0.6)
    .limit(50)
    .collect()
)
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

Apache-2.0 (same as [Lance](https://github.com/lancedb/lance))
