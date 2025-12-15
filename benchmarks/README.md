# Lance vs Parquet Benchmarks

Benchmarks demonstrating Lance advantages over Parquet for analytics workloads.

## Quick Results

| Feature | Lance | Parquet |
|---------|-------|---------|
| Time Travel | ✅ Native | ❌ Not supported |
| Row Updates | ✅ O(1) | ❌ Full rewrite |
| Row Deletes | ✅ O(1) | ❌ Full rewrite |
| Vector Search | ✅ Native ANN | ❌ Requires external index |
| Query Speed | ✅ Fast | ✅ Fast |
| Compression | ✅ Good | ✅ Good |

## Benchmarks

### 1. Time Travel
Query historical versions without copying data.

```bash
python benchmarks/01_time_travel.py
```

### 2. Updates & Deletes
Update/delete rows without rewriting entire file.

```bash
python benchmarks/02_updates.py
```

### 3. Vector Search
Semantic similarity search with ANN index.

```bash
python benchmarks/03_vector_search.py
```

### 4. Query Speed
Column pruning and predicate pushdown performance.

```bash
python benchmarks/04_query_speed.py
```

### 5. Compression
File size comparison for same data.

```bash
python benchmarks/05_compression.py
```

## Run All

```bash
python benchmarks/run_all.py
```

## Requirements

```bash
pip install pyarrow pandas lancedb faiss-cpu numpy tqdm
```
