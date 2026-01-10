# @metal0/lanceql

SQL database in the browser with vector search. No server required.

## Features

- **SQL Queries** - Full SQL: SELECT, JOIN, GROUP BY, window functions, CTEs
- **Vector Search** - Semantic search with NEAR clause using MiniLM/CLIP
- **Auto Vector Index** - Automatic text→embedding encoding on INSERT
- **DataFrame API** - fromRecords/fromColumns/saveTo for ETL workflows
- **OPFS Persistence** - Data persists across browser sessions
- **HTTP Range** - Only fetch bytes you need from remote Lance files
- **GPU Accelerated** - Optional WebGPU for JOINs, sorts, vector search
- **Zero Dependencies** - Pure JavaScript + WebAssembly

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser Main Thread                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │    vault()   │  │   LanceQL    │  │     GPU Transformer      │   │
│  │  (client API)│  │  (remote)    │  │   (WebGPU text encode)   │   │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘   │
│         │                 │                       │                  │
│         │    workerRPC()  │                       │                  │
│         └────────┬────────┘                       │                  │
│                  │                                │                  │
└──────────────────┼────────────────────────────────┼──────────────────┘
                   │ MessagePort                    │
┌──────────────────┼────────────────────────────────┼──────────────────┐
│                  ▼           SharedWorker         │                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      Worker Index (index.js)                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐   │  │
│  │  │ Vector Index│  │  DDL Sync   │  │   Query Rewriting    │   │  │
│  │  │  Metadata   │  │ CREATE/DROP │  │  (NEAR → shadow col) │   │  │
│  │  └─────────────┘  └─────────────┘  └──────────────────────┘   │  │
│  │                           │                                    │  │
│  │                           ▼                                    │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │               WorkerDatabase (OPFS storage)              │  │  │
│  │  │  - Tables (Lance fragments)                              │  │  │
│  │  │  - Vector indexes (__vec_* shadow columns)               │  │  │
│  │  │  - KV store (encrypted)                                  │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                           │                                    │  │
│  │                           ▼                                    │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │            WASM SQL Bridge (wasm-sql-bridge.js)          │  │  │
│  │  │  - Schema registration                                   │  │  │
│  │  │  - Query execution via lanceql.wasm                      │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Location | Purpose |
|-----------|----------|---------|
| `vault()` | `client/store/vault.js` | High-level API for KV + SQL |
| `LanceQL` | `client/lance-ql.js` | Remote Lance dataset access |
| `WorkerDatabase` | `worker/worker-database.js` | OPFS persistence, Lance storage |
| `WasmSqlBridge` | `worker/wasm-sql-bridge.js` | Schema→WASM, query execution |
| `GPUTransformer` | `client/webgpu/gpu-transformer.js` | WebGPU text embedding |
| `Worker Index` | `worker/index.js` | RPC router, DDL sync, vector indexes |

## Installation

```bash
npm install @metal0/lanceql
```

## Quick Start

### Local SQL Database

```javascript
import { vault } from '@metal0/lanceql/browser';

const v = await vault();
await v.exec('CREATE TABLE users (id INT, name TEXT)');
await v.exec("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')");
const result = await v.query('SELECT * FROM users');
// → { columns: ['id', 'name'], rows: [[1, 'Alice'], [2, 'Bob']] }
```

### Remote Lance Dataset

```javascript
import { LanceQL } from '@metal0/lanceql/browser';

const lanceql = await LanceQL.load();
const dataset = await lanceql.openDataset('https://example.com/data.lance');
const result = await dataset.executeSQL('SELECT * FROM data LIMIT 10');

// Vector search
const similar = await dataset.executeSQL(`
  SELECT * FROM data WHERE embedding NEAR 'sunset beach' LIMIT 20
`);
```

## Framework Integration

### React

```jsx
import { useState, useEffect } from 'react';
import { vault } from '@metal0/lanceql/browser';

function useQuery(sql) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    vault().then(v => v.query(sql)).then(setData).finally(() => setLoading(false));
  }, [sql]);

  return { data, loading };
}

function UserList() {
  const { data, loading } = useQuery('SELECT * FROM users LIMIT 10');
  if (loading) return <div>Loading...</div>;
  return <ul>{data.rows.map(([id, name]) => <li key={id}>{name}</li>)}</ul>;
}
```

### Vue

```vue
<script setup>
import { ref, onMounted } from 'vue';
import { vault } from '@metal0/lanceql/browser';

const users = ref([]);

onMounted(async () => {
  const v = await vault();
  const result = await v.query('SELECT * FROM users LIMIT 10');
  users.value = result.rows;
});
</script>

<template>
  <ul>
    <li v-for="[id, name] in users" :key="id">{{ name }}</li>
  </ul>
</template>
```

### Svelte

```svelte
<script>
  import { onMount } from 'svelte';
  import { vault } from '@metal0/lanceql/browser';

  let users = [];

  onMount(async () => {
    const v = await vault();
    const result = await v.query('SELECT * FROM users LIMIT 10');
    users = result.rows;
  });
</script>

<ul>
  {#each users as [id, name]}
    <li>{name}</li>
  {/each}
</ul>
```

## API Exports

```javascript
// Core
import { vault, LanceQL } from '@metal0/lanceql/browser';

// Advanced
import {
  Vault,              // Vault class
  TableRef,           // Table reference
  LocalDatabase,      // Lower-level DB
  RemoteLanceDataset  // Direct remote access
} from '@metal0/lanceql/browser';

// GPU Acceleration (optional)
import {
  getGPUJoiner,       // GPU hash joins
  getGPUSorter,       // GPU bitonic sort
  getGPUGrouper,      // GPU GROUP BY
  getGPUVectorSearch, // GPU vector ops
  DistanceMetric      // COSINE, L2, DOT_PRODUCT
} from '@metal0/lanceql/browser';
```

## CSS-Driven (Zero JavaScript)

```html
<!-- Just include the script -->
<script type="module" src="node_modules/lanceql/dist/lanceql.esm.js"></script>

<!-- Query and render with HTML attributes only -->
<div lq-query="SELECT * FROM read_lance('https://example.com/data.lance') LIMIT 10"
     lq-render="table">
</div>
```

That's it! No JavaScript needed. The query executes and renders automatically.

## CSS-Driven Features

### Table Rendering

```html
<div lq-query="SELECT name, value FROM read_lance('https://...') WHERE value > 100 LIMIT 50"
     lq-render="table">
</div>
```

### Image Gallery

```html
<div lq-query="SELECT url, text FROM read_lance('https://...') LIMIT 20"
     lq-render="images">
</div>
```

### Reactive Search

```html
<input type="text" id="search" placeholder="Search...">

<div lq-query="SELECT * FROM read_lance('https://...') WHERE text LIKE '%$value%' LIMIT 20"
     lq-bind="#search"
     lq-render="table">
</div>
```

Types as you search - no JavaScript needed!

### Available Renderers

- `lq-render="table"` - Table with image thumbnails (default)
- `lq-render="images"` - Image grid with captions
- `lq-render="json"` - JSON output
- `lq-render="value"` - Single value (for aggregates)
- `lq-render="list"` - Simple list

## JavaScript API (Optional)

For programmatic control, you can also use the JavaScript API:

```javascript
import LanceQL from 'lanceql';

// Load the WASM module
const lanceql = await LanceQL.load();

// Open a remote dataset
const dataset = await lanceql.openDataset('https://example.com/data.lance');

// Execute SQL
const results = await dataset.executeSQL(`
  SELECT name, value FROM data WHERE value > 100 LIMIT 50
`);

// Vector search
const similar = await dataset.executeSQL(`
  SELECT * FROM data NEAR 'sunset beach' TOPK 20
`);
```

## Usage with Local Files

```javascript
// File input handler
document.getElementById('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  const lanceql = await LanceQL.load();
  const lanceFile = await lanceql.openFile(file);

  // Query the file
  const results = await lanceFile.executeSQL('SELECT * FROM data LIMIT 10');
});
```

## SQL Support

```sql
-- Basic queries
SELECT * FROM data LIMIT 100
SELECT col1, col2 FROM data WHERE col1 > 10

-- Aggregations
SELECT COUNT(*), AVG(value), SUM(value) FROM data
SELECT category, COUNT(*) FROM data GROUP BY category

-- Vector search
SELECT * FROM data NEAR 'search text' TOPK 20
SELECT * FROM data NEAR embedding 'query' WHERE score > 0.5
```

## Auto Vector Index

Create indexes that automatically encode text to embeddings on INSERT:

```javascript
const v = await vault();

// Create table with text column
await v.exec('CREATE TABLE docs (id INT, content TEXT)');

// Create vector index - auto-encodes on INSERT
await v.exec('CREATE VECTOR INDEX ON docs(content) USING minilm');

// Insert text - automatically generates __vec_content_minilm column
await v.exec("INSERT INTO docs VALUES (1, 'Machine learning basics')");
await v.exec("INSERT INTO docs VALUES (2, 'Deep neural networks')");

// Semantic search - uses vector index automatically
const results = await v.query("SELECT * FROM docs WHERE content NEAR 'AI fundamentals' LIMIT 5");
```

### Supported Models

| Model | Dimensions | Use Case |
|-------|-----------|----------|
| `minilm` | 384 | Text similarity (default) |
| `clip` | 512 | Image/text cross-modal |

### Vector Index SQL

```sql
-- Create index
CREATE VECTOR INDEX ON table(column) USING model
CREATE VECTOR INDEX IF NOT EXISTS ON table(column) USING minilm

-- Drop index
DROP VECTOR INDEX ON table(column)
DROP VECTOR INDEX IF EXISTS ON table(column)

-- Show indexes
SHOW VECTOR INDEXES
SHOW VECTOR INDEXES ON table
```

## DataFrame API

Write data from JavaScript objects or columnar arrays:

```javascript
import { WritableDataFrame } from '@metal0/lanceql/browser';

// From array of records
const df = WritableDataFrame.fromRecords([
  { id: 1, name: 'Alice', score: 0.95 },
  { id: 2, name: 'Bob', score: 0.87 },
  { id: 3, name: 'Charlie', score: 0.92 }
]);
await df.saveTo(vault, 'scores');

// From columnar data (efficient for large datasets)
const df2 = WritableDataFrame.fromColumns({
  id: new Int32Array([1, 2, 3]),
  value: new Float64Array([1.5, 2.5, 3.5]),
  name: ['Alice', 'Bob', 'Charlie']
});
await df2.saveTo(vault, 'users', { ifExists: 'replace' });

// Chain from query result
await vault.df('SELECT * FROM source WHERE active = true')
  .saveTo(vault, 'active_users', { ifExists: 'append' });
```

### saveTo Options

| Option | Values | Description |
|--------|--------|-------------|
| `ifExists` | `'fail'` (default) | Error if table exists |
| | `'replace'` | Drop and recreate table |
| | `'append'` | Insert into existing table |
| `schema` | `{col: 'TYPE'}` | Explicit schema (auto-inferred if omitted) |

### Schema Inference

| JavaScript Type | SQL Type |
|-----------------|----------|
| `number` (integer) | `INT` |
| `number` (float) | `FLOAT` |
| `bigint` | `BIGINT` |
| `boolean` | `BOOLEAN` |
| `string` | `TEXT` |
| `Array<number>` | `VECTOR(N)` |
| `Int32Array` | `INT` |
| `Float64Array` | `FLOAT` |
| `BigInt64Array` | `BIGINT` |

## API Reference

### LanceQL.load(wasmUrl?)

Load the WASM module. Returns a LanceQL instance.

```javascript
const lanceql = await LanceQL.load('./lanceql.wasm');
```

### lanceql.openDataset(url, options?)

Open a remote Lance dataset.

```javascript
const dataset = await lanceql.openDataset('https://example.com/data.lance', {
  version: 24,  // Optional: specific version
});
```

### lanceql.openFile(file)

Open a local File object.

```javascript
const lanceFile = await lanceql.openFile(file);
```

### dataset.executeSQL(sql)

Execute a SQL query against the dataset.

```javascript
const results = await dataset.executeSQL('SELECT * FROM data LIMIT 10');
// Returns: { columns: ['col1', 'col2'], values: [[...], [...]] }
```

### dataset.df()

Get a DataFrame API for fluent queries.

```javascript
const result = await dataset.df()
  .filter('value', '>', 100)
  .select(['name', 'value'])
  .limit(50)
  .collect();
```

## Browser Support

- Chrome 89+
- Firefox 89+
- Safari 15+
- Edge 89+

Requires WebAssembly and async/await support.

## License

Apache-2.0
