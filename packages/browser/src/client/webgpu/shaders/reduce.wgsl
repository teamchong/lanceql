// Parallel Reduction Shaders for SQL Aggregations
// Implements SUM, MIN, MAX, COUNT with workgroup-level shared memory reduction

struct ReduceParams {
    size: u32,           // Number of elements to reduce
    workgroups: u32,     // Number of workgroups for partial results
}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Shared memory for workgroup-level reduction
var<workgroup> shared_data: array<f32, 256>;

// ============================================================================
// SUM Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_sum(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load element or 0 if out of bounds
    if (gid < params.size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0;
    }

    workgroupBarrier();

    // Tree reduction in shared memory
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    // First thread writes workgroup result
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// Final pass: sum the partial results from each workgroup
@compute @workgroup_size(256)
fn reduce_sum_final(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;

    // Load partial result or 0
    if (tid < params.workgroups) {
        shared_data[tid] = input[tid];
    } else {
        shared_data[tid] = 0.0;
    }

    workgroupBarrier();

    // Tree reduction
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    // Write final result
    if (tid == 0u) {
        output[0] = shared_data[0];
    }
}

// ============================================================================
// MIN Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_min(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load element or +inf if out of bounds
    if (gid < params.size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 3.402823466e+38;  // FLT_MAX
    }

    workgroupBarrier();

    // Tree reduction for min
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = min(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

@compute @workgroup_size(256)
fn reduce_min_final(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;

    if (tid < params.workgroups) {
        shared_data[tid] = input[tid];
    } else {
        shared_data[tid] = 3.402823466e+38;
    }

    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = min(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[0] = shared_data[0];
    }
}

// ============================================================================
// MAX Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_max(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = -3.402823466e+38;  // -FLT_MAX
    }

    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

@compute @workgroup_size(256)
fn reduce_max_final(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;

    if (tid < params.workgroups) {
        shared_data[tid] = input[tid];
    } else {
        shared_data[tid] = -3.402823466e+38;
    }

    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[0] = shared_data[0];
    }
}

// ============================================================================
// COUNT (non-null) Reduction
// Uses input as mask: 0.0 = null, 1.0 = present
// ============================================================================

@compute @workgroup_size(256)
fn reduce_count(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Each element contributes 1.0 if present (mask value)
    if (gid < params.size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0;
    }

    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// ============================================================================
// Multi-column reduction (batched)
// Reduces multiple columns in a single dispatch
// ============================================================================

struct MultiReduceParams {
    size: u32,       // Elements per column
    num_cols: u32,   // Number of columns
    stride: u32,     // Stride between columns in input
}

@group(0) @binding(0) var<uniform> multi_params: MultiReduceParams;
@group(0) @binding(1) var<storage, read> multi_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> multi_output: array<f32>;

// Each workgroup handles one column
@compute @workgroup_size(256)
fn reduce_sum_multi(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>,
                    @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let col = workgroup_id.x;

    if (col >= multi_params.num_cols) {
        return;
    }

    // Sum this column's elements in chunks
    var sum: f32 = 0.0;
    let col_offset = col * multi_params.stride;

    for (var i: u32 = tid; i < multi_params.size; i = i + 256u) {
        sum = sum + multi_input[col_offset + i];
    }

    shared_data[tid] = sum;
    workgroupBarrier();

    // Tree reduction
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        multi_output[col] = shared_data[0];
    }
}

@compute @workgroup_size(256)
fn reduce_min_multi(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>,
                    @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let col = workgroup_id.x;

    if (col >= multi_params.num_cols) {
        return;
    }

    var min_val: f32 = 3.402823466e+38;
    let col_offset = col * multi_params.stride;

    for (var i: u32 = tid; i < multi_params.size; i = i + 256u) {
        min_val = min(min_val, multi_input[col_offset + i]);
    }

    shared_data[tid] = min_val;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = min(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        multi_output[col] = shared_data[0];
    }
}

@compute @workgroup_size(256)
fn reduce_max_multi(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>,
                    @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let col = workgroup_id.x;

    if (col >= multi_params.num_cols) {
        return;
    }

    var max_val: f32 = -3.402823466e+38;
    let col_offset = col * multi_params.stride;

    for (var i: u32 = tid; i < multi_params.size; i = i + 256u) {
        max_val = max(max_val, multi_input[col_offset + i]);
    }

    shared_data[tid] = max_val;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        multi_output[col] = shared_data[0];
    }
}
