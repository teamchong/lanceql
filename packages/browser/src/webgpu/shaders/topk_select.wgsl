// GPU-Accelerated Top-K Selection
// Uses partial bitonic sort - only sorts enough to find top-K elements
// Two-phase approach: local top-K per workgroup, then merge

// ============================================================================
// Phase 1: Local Top-K Selection
// Each workgroup processes a chunk and outputs its local top-K
// ============================================================================

struct TopKParams {
    size: u32,        // Total number of elements
    k: u32,           // Number of top elements to find
    descending: u32,  // 1=descending (max top-k), 0=ascending (min top-k)
    num_workgroups: u32, // Total workgroups for intermediate sizing
}

@group(0) @binding(0) var<uniform> params: TopKParams;
@group(0) @binding(1) var<storage, read> input_scores: array<f32>;
@group(0) @binding(2) var<storage, read> input_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_indices: array<u32>;

// Shared memory for local sorting (512 elements per workgroup)
var<workgroup> local_scores: array<f32, 512>;
var<workgroup> local_indices: array<u32, 512>;

// Compare function based on sort order
fn should_swap(a: f32, b: f32, descending: bool) -> bool {
    if (descending) {
        return a < b;  // Swap if a < b (want larger first)
    } else {
        return a > b;  // Swap if a > b (want smaller first)
    }
}

// Sentinel value for empty slots
fn get_sentinel(descending: bool) -> f32 {
    if (descending) {
        return -3.4028235e+38;  // -MAX for descending (pushed to end)
    } else {
        return 3.4028235e+38;   // +MAX for ascending (pushed to end)
    }
}

// Phase 1: Each workgroup finds its local top-K
// Each workgroup processes 512 elements
@compute @workgroup_size(256)
fn local_topk(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let chunk_size = 512u;
    let base = wid.x * chunk_size;
    let tid = lid.x;
    let descending = params.descending == 1u;
    let sentinel = get_sentinel(descending);

    // Each thread loads 2 elements
    let idx1 = base + tid;
    let idx2 = base + tid + 256u;

    if (idx1 < params.size) {
        local_scores[tid] = input_scores[idx1];
        local_indices[tid] = input_indices[idx1];
    } else {
        local_scores[tid] = sentinel;
        local_indices[tid] = 0xFFFFFFFFu;
    }

    if (idx2 < params.size) {
        local_scores[tid + 256u] = input_scores[idx2];
        local_indices[tid + 256u] = input_indices[idx2];
    } else {
        local_scores[tid + 256u] = sentinel;
        local_indices[tid + 256u] = 0xFFFFFFFFu;
    }

    workgroupBarrier();

    // Bitonic sort - full sort of local 512 elements
    // This is O(log^2(512)) = O(81) steps, very fast on GPU
    for (var k = 2u; k <= chunk_size; k = k << 1u) {
        for (var j = k >> 1u; j > 0u; j = j >> 1u) {
            // Each thread handles 2 comparisons
            for (var t = 0u; t < 2u; t++) {
                let i = tid + t * 256u;
                let ixj = i ^ j;

                if (ixj > i && ixj < chunk_size) {
                    let direction = ((i & k) == 0u) == descending;
                    if (should_swap(local_scores[i], local_scores[ixj], direction)) {
                        // Swap scores
                        let tmp_score = local_scores[i];
                        local_scores[i] = local_scores[ixj];
                        local_scores[ixj] = tmp_score;
                        // Swap indices
                        let tmp_idx = local_indices[i];
                        local_indices[i] = local_indices[ixj];
                        local_indices[ixj] = tmp_idx;
                    }
                }
            }
            workgroupBarrier();
        }
    }

    // Write top-K from this workgroup to output
    // First K elements are the best after sorting
    let k_per_wg = min(params.k, chunk_size);
    if (tid < k_per_wg) {
        let out_base = wid.x * params.k;
        output_scores[out_base + tid] = local_scores[tid];
        output_indices[out_base + tid] = local_indices[tid];
    }
    if (tid + 256u < k_per_wg) {
        let out_base = wid.x * params.k;
        output_scores[out_base + tid + 256u] = local_scores[tid + 256u];
        output_indices[out_base + tid + 256u] = local_indices[tid + 256u];
    }
}

// ============================================================================
// Phase 2: Merge Workgroup Results
// Takes all local top-K results and finds global top-K
// ============================================================================

struct MergeParams {
    num_candidates: u32,  // Total candidates (num_workgroups * k)
    k: u32,               // Final K to output
    descending: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> merge_params: MergeParams;
@group(0) @binding(1) var<storage, read> merge_scores: array<f32>;
@group(0) @binding(2) var<storage, read> merge_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> final_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> final_indices: array<u32>;

// For merge phase, we use a simpler approach since candidate count is smaller
// Single workgroup sorts all candidates and outputs top-K
@compute @workgroup_size(256)
fn merge_topk(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = merge_params.num_candidates;
    let k = merge_params.k;
    let descending = merge_params.descending == 1u;
    let sentinel = get_sentinel(descending);

    // Load candidates into shared memory
    // Handle up to 512 candidates in shared memory
    let n_local = min(n, 512u);

    if (tid < n_local) {
        local_scores[tid] = merge_scores[tid];
        local_indices[tid] = merge_indices[tid];
    } else if (tid < 512u) {
        local_scores[tid] = sentinel;
        local_indices[tid] = 0xFFFFFFFFu;
    }

    if (tid + 256u < n_local) {
        local_scores[tid + 256u] = merge_scores[tid + 256u];
        local_indices[tid + 256u] = merge_indices[tid + 256u];
    } else if (tid + 256u < 512u) {
        local_scores[tid + 256u] = sentinel;
        local_indices[tid + 256u] = 0xFFFFFFFFu;
    }

    workgroupBarrier();

    // Bitonic sort the candidates
    let chunk_size = 512u;
    for (var ks = 2u; ks <= chunk_size; ks = ks << 1u) {
        for (var j = ks >> 1u; j > 0u; j = j >> 1u) {
            for (var t = 0u; t < 2u; t++) {
                let i = tid + t * 256u;
                let ixj = i ^ j;

                if (ixj > i && ixj < chunk_size) {
                    let direction = ((i & ks) == 0u) == descending;
                    if (should_swap(local_scores[i], local_scores[ixj], direction)) {
                        let tmp_score = local_scores[i];
                        local_scores[i] = local_scores[ixj];
                        local_scores[ixj] = tmp_score;
                        let tmp_idx = local_indices[i];
                        local_indices[i] = local_indices[ixj];
                        local_indices[ixj] = tmp_idx;
                    }
                }
            }
            workgroupBarrier();
        }
    }

    // Output final top-K
    if (tid < k) {
        final_scores[tid] = local_scores[tid];
        final_indices[tid] = local_indices[tid];
    }
}

// ============================================================================
// Initialize indices (identity mapping)
// ============================================================================

struct InitParams {
    size: u32,
}

@group(0) @binding(0) var<uniform> init_params: InitParams;
@group(0) @binding(1) var<storage, read_write> indices: array<u32>;

@compute @workgroup_size(256)
fn init_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < init_params.size) {
        indices[idx] = idx;
    }
}
