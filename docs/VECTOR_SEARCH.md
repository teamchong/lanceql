# Vector Search

## Table of Contents

- [Quick Start](#quick-start)
- [NEAR Clause Syntax](#near-clause-syntax)
- [Text Encoders](#text-encoders)
- [IVF-PQ Index Explained](#ivf-pq-index-explained)
- [Performance Tuning](#performance-tuning)
- [Creating Your Own Index](#creating-your-own-index)

---

## Quick Start

### Text Search

```sql
SELECT * FROM read_lance('https://data.metal0.dev/laion-1m/images.lance')
NEAR 'sunset on the beach'
TOPK 20
```

### Search by Row

```sql
-- Find 10 items similar to row 0
SELECT * FROM read_lance('images.lance') NEAR 0 TOPK 10
```

### Combine with Filters

```sql
SELECT * FROM read_lance('images.lance')
WHERE aesthetic > 0.7
NEAR 'mountain landscape'
TOPK 50
```

---

## NEAR Clause Syntax

### Basic Syntax

```sql
SELECT * FROM table_source
NEAR [column] query_text
[TOPK n]
```

### Examples

```sql
-- Search using default vector column
SELECT * FROM read_lance(FILE) NEAR 'cat playing'

-- Specify TOPK (default is 20)
SELECT * FROM read_lance(FILE) NEAR 'cat playing' TOPK 50

-- Specify vector column
SELECT * FROM read_lance(FILE) NEAR embedding 'cat playing'

-- Search by row index (find similar to row 0)
SELECT * FROM read_lance(FILE) NEAR 0 TOPK 10

-- Combine with WHERE
SELECT * FROM read_lance(FILE)
WHERE category = 'pets'
NEAR 'fluffy dog' TOPK 30

-- With ORDER BY (sorts within top K)
SELECT * FROM read_lance(FILE)
NEAR 'landscape photography'
TOPK 100
ORDER BY aesthetic_score DESC
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `column` | Name of vector column | Auto-detected |
| `query_text` | Text to search for | Required |
| `TOPK n` | Number of results | 20 |

---

## Text Encoders

### MiniLM (Default)

- **Model**: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Best for**: Text-to-text similarity
- **Use case**: Searching text descriptions, captions, articles

```sql
-- MiniLM is the default encoder
SELECT * FROM read_lance(FILE) NEAR 'machine learning tutorial'
```

**Characteristics**:
- Fast inference
- Optimized for semantic textual similarity
- Works well with short to medium text (up to 256 tokens)
- L2-normalized embeddings

### CLIP

- **Model**: OpenAI ViT-B/32
- **Dimensions**: 512
- **Best for**: Text-to-image search
- **Use case**: Finding images from text descriptions

**Characteristics**:
- Trained on image-text pairs
- Works with any image dataset that has CLIP embeddings
- L2-normalized embeddings

### Encoder Comparison

| Feature | MiniLM | CLIP |
|---------|--------|------|
| Dimensions | 384 | 512 |
| Best For | Text similarity | Image search |
| Vocab | WordPiece (30k) | BPE (49k) |

### Choosing an Encoder

- **Text datasets** (articles, documents, captions): Use MiniLM
- **Image datasets**: Use CLIP if images have CLIP embeddings
- **Mixed content**: Match encoder to how the data was indexed

---

## IVF-PQ Index Explained

For datasets with millions of vectors, brute-force search is too slow. LanceQL uses **IVF-PQ** (Inverted File with Product Quantization) for fast approximate search.

### How IVF-PQ Works

#### Step 1: Inverted File (IVF)

Vectors are grouped into **partitions** (clusters):

```
Dataset: 1,000,000 vectors
Partitions: 256 clusters (~4,000 vectors each)

At query time:
1. Find the closest partition centroids (e.g., top 20)
2. Only search vectors in those partitions
3. Skip ~92% of vectors
```

#### Step 2: Product Quantization (PQ)

Vectors are compressed:

```
Original: 384 dimensions × 4 bytes = 1,536 bytes per vector
PQ: 48 sub-vectors × 1 byte = 48 bytes per vector

Compression: 32× smaller
Speed: Compare compressed codes instead of full vectors
```

### Accuracy vs Speed Trade-off

Higher nprobe = more partitions searched = better recall but slower.

**Default nprobe**: 20

---

## Performance Tuning

### Dataset Size Guidelines

| Dataset Size | Recommended Index | Partitions |
|--------------|-------------------|------------|
| < 10,000 | None (brute force) | - |
| 10K - 100K | IVF | 32-64 |
| 100K - 1M | IVF-PQ | 128-256 |
| 1M - 10M | IVF-PQ | 256-1024 |
| > 10M | IVF-PQ + sharding | 1024+ |

### Query Optimization

1. **Filter before search**: Apply WHERE clauses first to reduce search space

```sql
-- Good: Filter then search
SELECT * FROM read_lance(FILE)
WHERE category = 'electronics'
NEAR 'wireless headphones'
TOPK 20

-- Less efficient: Search entire dataset then filter
SELECT * FROM (
    SELECT * FROM read_lance(FILE)
    NEAR 'wireless headphones'
    TOPK 1000
)
WHERE category = 'electronics'
LIMIT 20
```

2. **Limit TOPK**: Request only what you need

```sql
-- Good: Small TOPK
SELECT * FROM read_lance(FILE) NEAR 'query' TOPK 20

-- Wasteful: Large TOPK when you only need 10
SELECT * FROM read_lance(FILE) NEAR 'query' TOPK 1000 LIMIT 10
```

3. **Use appropriate encoder**: Match encoder to your data

---

## Creating Your Own Index

### Using PyLance

```python
import lance
import numpy as np

# Create dataset with vectors
data = {
    "id": [1, 2, 3, ...],
    "text": ["cat", "dog", "bird", ...],
    "embedding": [vec1, vec2, vec3, ...]  # 384-dim float32 vectors
}

# Save as Lance
ds = lance.write_dataset(data, "my_dataset.lance")

# Create IVF-PQ index
ds.create_index(
    "embedding",
    index_type="IVF_PQ",
    num_partitions=256,    # Number of clusters
    num_sub_vectors=48,    # PQ sub-vectors
    metric="cosine"        # or "L2", "dot"
)
```

### Generating Embeddings

#### With MiniLM

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["cat", "dog", "bird"]
embeddings = model.encode(texts, normalize_embeddings=True)
# Shape: (3, 384)
```

#### With CLIP

```python
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

texts = ["cat", "dog", "bird"]
inputs = processor(text=texts, return_tensors="pt", padding=True)

with torch.no_grad():
    embeddings = model.get_text_features(**inputs)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
# Shape: (3, 512)
```

### Index Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `num_partitions` | IVF clusters | sqrt(N) to 4*sqrt(N) |
| `num_sub_vectors` | PQ compression | 8-96 (higher = less compression) |
| `metric` | Distance metric | "cosine" for normalized vectors |

### Example: 1M Image Dataset

```python
import lance
import numpy as np
from tqdm import tqdm

# Load your data
ids = np.arange(1_000_000)
texts = [...]  # 1M text descriptions
embeddings = np.load("embeddings.npy")  # (1M, 384) float32

# Create Lance dataset
ds = lance.write_dataset({
    "id": ids,
    "text": texts,
    "embedding": embeddings.tolist()
}, "images.lance")

# Create IVF-PQ index
ds.create_index(
    "embedding",
    index_type="IVF_PQ",
    num_partitions=256,      # sqrt(1M) ≈ 1000, but 256 works well
    num_sub_vectors=48,      # 384 / 48 = 8 dims per sub-vector
    metric="cosine"
)

print(f"Dataset size: {ds.count_rows():,} rows")
print(f"Index created on 'embedding' column")
```

---

## Troubleshooting

### "No vector column found"

Your dataset doesn't have a recognized vector column. Vector columns should:
- Be named `embedding`, `vector`, or similar
- Contain `fixed_size_list` of floats
- Have consistent dimensions

### "Index not found"

The dataset doesn't have an IVF-PQ index. You can still query (brute force), but it will be slower for large datasets.

### Low accuracy results

1. Check encoder match: Was the dataset indexed with MiniLM or CLIP?
2. Increase TOPK: Try `TOPK 100` to see if relevant results appear
3. Refine query: More specific queries get better results

### Slow queries

1. Check dataset size: Large datasets without an index will be slow
2. Reduce TOPK: Smaller TOPK = faster queries
3. Use filters: `WHERE` clauses reduce the search space
