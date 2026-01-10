# Shared WGSL Shaders

These WGSL shaders are shared between:
- **Browser**: WebGPU via JavaScript adapters
- **CLI**: wgpu-native via Zig bindings

## SQL Operations

| Shader | Purpose |
|--------|---------|
| `join.wgsl` | Hash JOIN (build, probe, init) |
| `group_by.wgsl` | GROUP BY aggregations (count, sum, min, max) |
| `sort.wgsl` | Bitonic sort for ORDER BY |
| `reduce.wgsl` | Aggregate reductions (SUM, MIN, MAX, COUNT) |
| `topk_select.wgsl` | Top-K selection for LIMIT |
| `vector_distance.wgsl` | Vector similarity (cosine, L2, dot product) |

## ML Operations (Browser only)

| Shader | Purpose |
|--------|---------|
| `attention.wgsl` | Transformer attention mechanism |
| `gemm.wgsl` | General matrix multiplication |
| `layernorm.wgsl` | Layer normalization |
| `gelu.wgsl` | GELU activation function |
| `embedding.wgsl` | Embedding lookup |

## Usage

### Browser (via symlink)
```
packages/browser/src/client/webgpu/shaders/ -> ../../../../../shared/gpu/shaders/
```

### CLI (via Zig @embedFile)
```zig
const join_shader = @embedFile("packages/shared/gpu/shaders/join.wgsl");
```
