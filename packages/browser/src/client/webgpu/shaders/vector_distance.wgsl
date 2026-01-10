// GPU-Accelerated Vector Distance Computation
// Supports multiple metrics: cosine, L2, dot product
// Optimized with shared memory caching for query vectors

// ============================================================================
// Distance Computation (Single/Batch Query)
// ============================================================================

struct DistanceParams {
    dim: u32,           // Vector dimension
    num_vectors: u32,   // Number of database vectors
    num_queries: u32,   // Number of query vectors (1 for single query)
    metric: u32,        // 0=cosine, 1=L2, 2=dot_product
}

@group(0) @binding(0) var<uniform> params: DistanceParams;
@group(0) @binding(1) var<storage, read> queries: array<f32>;     // [num_queries * dim]
@group(0) @binding(2) var<storage, read> vectors: array<f32>;     // [num_vectors * dim]
@group(0) @binding(3) var<storage, read_write> distances: array<f32>; // [num_queries * num_vectors]

// Shared memory for query vector caching (max 512 elements = 2KB)
var<workgroup> shared_query: array<f32, 512>;

// Main distance computation kernel
// Dispatch: (ceil(num_vectors/256), num_queries, 1)
@compute @workgroup_size(256)
fn compute_distances(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(workgroup_id) wid: vec3<u32>) {
    let query_idx = wid.y;   // Which query (batch dimension)
    let vec_idx = gid.x;     // Which database vector

    if (vec_idx >= params.num_vectors || query_idx >= params.num_queries) {
        return;
    }

    let dim = params.dim;
    let tid = lid.x;

    // Collaboratively load query vector into shared memory
    // Each thread loads multiple elements if dim > 256
    for (var i = tid; i < dim && i < 512u; i += 256u) {
        shared_query[i] = queries[query_idx * dim + i];
    }
    workgroupBarrier();

    let vec_offset = vec_idx * dim;
    var result: f32 = 0.0;

    switch params.metric {
        case 0u: {
            // Cosine similarity (assumes normalized vectors = just dot product)
            result = compute_dot(vec_offset, dim);
        }
        case 1u: {
            // L2 (Euclidean) distance
            result = compute_l2(vec_offset, dim);
        }
        case 2u: {
            // Raw dot product
            result = compute_dot(vec_offset, dim);
        }
        default: {
            result = compute_dot(vec_offset, dim);
        }
    }

    distances[query_idx * params.num_vectors + vec_idx] = result;
}

// Compute dot product between shared query and vector at offset
fn compute_dot(vec_offset: u32, dim: u32) -> f32 {
    var sum: f32 = 0.0;
    // Process 4 elements at a time for better memory coalescing
    let dim4 = dim / 4u;
    for (var i = 0u; i < dim4; i++) {
        let base = i * 4u;
        sum += shared_query[base] * vectors[vec_offset + base];
        sum += shared_query[base + 1u] * vectors[vec_offset + base + 1u];
        sum += shared_query[base + 2u] * vectors[vec_offset + base + 2u];
        sum += shared_query[base + 3u] * vectors[vec_offset + base + 3u];
    }
    // Handle remainder
    for (var i = dim4 * 4u; i < dim; i++) {
        sum += shared_query[i] * vectors[vec_offset + i];
    }
    return sum;
}

// Compute L2 (Euclidean) distance
fn compute_l2(vec_offset: u32, dim: u32) -> f32 {
    var sum: f32 = 0.0;
    let dim4 = dim / 4u;
    for (var i = 0u; i < dim4; i++) {
        let base = i * 4u;
        let d0 = shared_query[base] - vectors[vec_offset + base];
        let d1 = shared_query[base + 1u] - vectors[vec_offset + base + 1u];
        let d2 = shared_query[base + 2u] - vectors[vec_offset + base + 2u];
        let d3 = shared_query[base + 3u] - vectors[vec_offset + base + 3u];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (var i = dim4 * 4u; i < dim; i++) {
        let d = shared_query[i] - vectors[vec_offset + i];
        sum += d * d;
    }
    return sqrt(sum);
}

// ============================================================================
// Large Dimension Support (dim > 512)
// Uses global memory for query when shared memory is insufficient
// ============================================================================

@compute @workgroup_size(256)
fn compute_distances_large(@builtin(global_invocation_id) gid: vec3<u32>,
                           @builtin(workgroup_id) wid: vec3<u32>) {
    let query_idx = wid.y;
    let vec_idx = gid.x;

    if (vec_idx >= params.num_vectors || query_idx >= params.num_queries) {
        return;
    }

    let dim = params.dim;
    let query_offset = query_idx * dim;
    let vec_offset = vec_idx * dim;
    var result: f32 = 0.0;

    switch params.metric {
        case 0u, 2u: {
            // Dot product (cosine for normalized)
            for (var i = 0u; i < dim; i++) {
                result += queries[query_offset + i] * vectors[vec_offset + i];
            }
        }
        case 1u: {
            // L2 distance
            for (var i = 0u; i < dim; i++) {
                let d = queries[query_offset + i] - vectors[vec_offset + i];
                result += d * d;
            }
            result = sqrt(result);
        }
        default: {}
    }

    distances[query_idx * params.num_vectors + vec_idx] = result;
}
