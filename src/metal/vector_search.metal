//
// Metal GPU Shaders for LanceQL Vector Search
//
// Compile: xcrun -sdk macosx metal -O3 -c vector_search.metal -o vector_search.air
// Link:    xcrun -sdk macosx metallib vector_search.air -o vector_search.metallib
//

#include <metal_stdlib>
using namespace metal;

/// Batch cosine similarity: compute similarity of query against all vectors
/// Each thread processes one vector
kernel void cosine_similarity_batch(
    device const float* query [[buffer(0)]],      // Query vector [dim]
    device const float* vectors [[buffer(1)]],    // All vectors [num_vectors * dim]
    device float* scores [[buffer(2)]],           // Output scores [num_vectors]
    constant uint& dim [[buffer(3)]],             // Vector dimension
    uint gid [[thread_position_in_grid]]          // Vector index
) {
    // Compute dot product and norms
    float dot = 0.0f;
    float query_norm = 0.0f;
    float vec_norm = 0.0f;

    const device float* vec = vectors + gid * dim;

    // Vectorized accumulation (4 floats at a time)
    uint i = 0;
    for (; i + 4 <= dim; i += 4) {
        float4 q = float4(query[i], query[i+1], query[i+2], query[i+3]);
        float4 v = float4(vec[i], vec[i+1], vec[i+2], vec[i+3]);

        dot += dot(q, v);
        query_norm += dot(q, q);
        vec_norm += dot(v, v);
    }

    // Scalar remainder
    for (; i < dim; i++) {
        float q = query[i];
        float v = vec[i];
        dot += q * v;
        query_norm += q * q;
        vec_norm += v * v;
    }

    // Cosine similarity
    float denom = sqrt(query_norm) * sqrt(vec_norm);
    scores[gid] = (denom > 0.0f) ? (dot / denom) : 0.0f;
}

/// Batch dot product
kernel void dot_product_batch(
    device const float* query [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float dot = 0.0f;
    const device float* vec = vectors + gid * dim;

    for (uint i = 0; i < dim; i++) {
        dot += query[i] * vec[i];
    }

    scores[gid] = dot;
}

/// Batch L2 distance squared
kernel void l2_distance_batch(
    device const float* query [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float dist = 0.0f;
    const device float* vec = vectors + gid * dim;

    for (uint i = 0; i < dim; i++) {
        float diff = query[i] - vec[i];
        dist += diff * diff;
    }

    scores[gid] = dist;
}

/// Top-K selection using parallel reduction
/// Each threadgroup finds local top-K, then merged on CPU
kernel void top_k_partial(
    device const float* scores [[buffer(0)]],     // Input scores [num_vectors]
    device float* top_scores [[buffer(1)]],       // Output top scores [num_groups * k]
    device uint* top_indices [[buffer(2)]],       // Output top indices [num_groups * k]
    constant uint& k [[buffer(3)]],               // K value
    constant uint& num_vectors [[buffer(4)]],     // Total vectors
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Each thread maintains its own top-K candidates
    // Simplified: just find max in this thread's chunk
    uint chunk_start = gid;
    uint chunk_stride = tg_size * 256; // Total threads

    float best_score = -INFINITY;
    uint best_idx = 0;

    for (uint i = chunk_start; i < num_vectors; i += chunk_stride) {
        float s = scores[i];
        if (s > best_score) {
            best_score = s;
            best_idx = i;
        }
    }

    // Store this thread's best (full top-K merge done on CPU)
    top_scores[gid] = best_score;
    top_indices[gid] = best_idx;
}
