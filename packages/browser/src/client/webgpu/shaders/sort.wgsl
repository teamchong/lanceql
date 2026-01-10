// GPU Bitonic Sort Shader
// Implements parallel bitonic sorting network for ORDER BY operations

// ============================================================================
// Local Bitonic Sort (within workgroup)
// ============================================================================

struct LocalSortParams {
    size: u32,       // Total number of elements
    stage: u32,      // Current bitonic stage
    step: u32,       // Current step within stage
    ascending: u32,  // 1 = ASC, 0 = DESC
}

@group(0) @binding(0) var<uniform> local_params: LocalSortParams;
@group(0) @binding(1) var<storage, read_write> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;

var<workgroup> shared_keys: array<f32, 512>;
var<workgroup> shared_indices: array<u32, 512>;

// Compare and swap for bitonic sort
fn compare_swap(i: u32, j: u32, dir: bool) {
    let ki = shared_keys[i];
    let kj = shared_keys[j];

    // Handle NaN/null (represented as very large values)
    let should_swap = select(ki > kj, ki < kj, dir);

    if (should_swap) {
        shared_keys[i] = kj;
        shared_keys[j] = ki;
        let ti = shared_indices[i];
        shared_indices[i] = shared_indices[j];
        shared_indices[j] = ti;
    }
}

// Local bitonic sort within a workgroup (up to 512 elements)
@compute @workgroup_size(256)
fn local_bitonic_sort(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_size = 512u;
    let base = wid.x * local_size;
    let tid = lid.x;

    // Load two elements per thread into shared memory
    let idx1 = base + tid;
    let idx2 = base + tid + 256u;

    if (idx1 < local_params.size) {
        shared_keys[tid] = keys[idx1];
        shared_indices[tid] = indices[idx1];
    } else {
        shared_keys[tid] = 3.4e38;  // Max float (sorts to end)
        shared_indices[tid] = idx1;
    }

    if (idx2 < local_params.size) {
        shared_keys[tid + 256u] = keys[idx2];
        shared_indices[tid + 256u] = indices[idx2];
    } else {
        shared_keys[tid + 256u] = 3.4e38;
        shared_indices[tid + 256u] = idx2;
    }

    workgroupBarrier();

    // Bitonic sort network for local_size elements
    let ascending = local_params.ascending == 1u;

    // Stage loop: 1, 2, 4, 8, ...
    for (var stage = 1u; stage < local_size; stage = stage << 1u) {
        // Step loop within each stage
        for (var step = stage; step > 0u; step = step >> 1u) {
            let pair_distance = step;
            let block_size = step << 1u;

            // Each thread handles one compare-swap
            let pos = tid;
            if (pos < 256u) {
                // First half of threads
                let block_id = pos / step;
                let in_block = pos % step;
                let i = block_id * block_size + in_block;
                let j = i + pair_distance;

                if (j < local_size) {
                    // Direction alternates for bitonic sequence
                    let dir = ((i / (stage << 1u)) % 2u == 0u) == ascending;
                    compare_swap(i, j, dir);
                }
            }

            workgroupBarrier();
        }
    }

    // Write back to global memory
    if (idx1 < local_params.size) {
        keys[idx1] = shared_keys[tid];
        indices[idx1] = shared_indices[tid];
    }
    if (idx2 < local_params.size) {
        keys[idx2] = shared_keys[tid + 256u];
        indices[idx2] = shared_indices[tid + 256u];
    }
}

// ============================================================================
// Global Bitonic Merge (across workgroups)
// ============================================================================

struct MergeParams {
    size: u32,
    stage: u32,      // 2^stage = block size
    step: u32,       // Current step (distance between compared elements)
    ascending: u32,
}

@group(0) @binding(0) var<uniform> merge_params: MergeParams;
@group(0) @binding(1) var<storage, read_write> merge_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> merge_indices: array<u32>;

// Global bitonic merge step
@compute @workgroup_size(256)
fn bitonic_merge_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let step = merge_params.step;
    let stage = merge_params.stage;
    let ascending = merge_params.ascending == 1u;

    // Each thread handles one element
    let block_size = 1u << (stage + 1u);
    let half_block = 1u << stage;

    // Determine which element this thread compares
    let block_id = tid / half_block;
    let in_half = tid % half_block;

    // Calculate indices to compare
    let i = block_id * block_size + in_half;
    let j = i + step;

    if (j >= merge_params.size) {
        return;
    }

    // Direction based on position in larger bitonic sequence
    let dir = ((i / block_size) % 2u == 0u) == ascending;

    let ki = merge_keys[i];
    let kj = merge_keys[j];

    let should_swap = select(ki > kj, ki < kj, dir);

    if (should_swap) {
        merge_keys[i] = kj;
        merge_keys[j] = ki;
        let ti = merge_indices[i];
        merge_indices[i] = merge_indices[j];
        merge_indices[j] = ti;
    }
}

// ============================================================================
// Initialize indices (0, 1, 2, 3, ...)
// ============================================================================

struct InitParams {
    size: u32,
}

@group(0) @binding(0) var<uniform> init_params: InitParams;
@group(0) @binding(1) var<storage, read_write> init_indices: array<u32>;

@compute @workgroup_size(256)
fn init_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < init_params.size) {
        init_indices[gid.x] = gid.x;
    }
}

// ============================================================================
// Permute rows using sorted indices
// ============================================================================

struct PermuteParams {
    size: u32,
    num_cols: u32,
}

@group(0) @binding(0) var<uniform> perm_params: PermuteParams;
@group(0) @binding(1) var<storage, read> perm_indices: array<u32>;
@group(0) @binding(2) var<storage, read> src_data: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_data: array<f32>;

@compute @workgroup_size(256)
fn permute_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= perm_params.size) {
        return;
    }

    let src_row = perm_indices[row];
    let num_cols = perm_params.num_cols;

    // Copy all columns for this row
    for (var col = 0u; col < num_cols; col++) {
        dst_data[row * num_cols + col] = src_data[src_row * num_cols + col];
    }
}
