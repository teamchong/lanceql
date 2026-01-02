// GPU Hash-Based GROUP BY Shader
// Implements parallel hash grouping and per-group aggregation

// ============================================================================
// Phase 1: Build Hash Table (key -> group_id mapping)
// ============================================================================

struct BuildParams {
    size: u32,           // Number of rows
    capacity: u32,       // Hash table capacity
}

@group(0) @binding(0) var<uniform> build_params: BuildParams;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hash_table: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> group_counter: atomic<u32>;

fn fnv_hash(key: u32) -> u32 {
    var h = 2166136261u;
    h ^= (key & 0xFFu); h *= 16777619u;
    h ^= ((key >> 8u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 16u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 24u) & 0xFFu); h *= 16777619u;
    return h;
}

// Build hash table: each unique key gets a unique group_id
@compute @workgroup_size(256)
fn build_groups(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= build_params.size) { return; }

    let key = keys[tid];
    var slot = fnv_hash(key) % build_params.capacity;

    // Linear probing to find or create entry
    for (var p = 0u; p < build_params.capacity; p++) {
        let idx = slot * 2u;

        // Try to claim this slot
        let old_key = atomicCompareExchangeWeak(&hash_table[idx], 0xFFFFFFFFu, key);

        if (old_key.exchanged) {
            // We claimed an empty slot - assign new group_id
            let group_id = atomicAdd(&group_counter, 1u);
            atomicStore(&hash_table[idx + 1u], group_id);
            return;
        }

        if (old_key.old_value == key) {
            // Key already exists - done
            return;
        }

        // Collision - try next slot
        slot = (slot + 1u) % build_params.capacity;
    }
}

// ============================================================================
// Phase 2: Assign Group IDs to All Rows
// ============================================================================

struct AssignParams {
    size: u32,
    capacity: u32,
}

@group(0) @binding(0) var<uniform> assign_params: AssignParams;
@group(0) @binding(1) var<storage, read> assign_keys: array<u32>;
@group(0) @binding(2) var<storage, read> lookup_table: array<u32>;
@group(0) @binding(3) var<storage, read_write> group_ids: array<u32>;

// Probe hash table to get group_id for each row
@compute @workgroup_size(256)
fn assign_groups(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= assign_params.size) { return; }

    let key = assign_keys[tid];
    var slot = fnv_hash(key) % assign_params.capacity;

    // Linear probe to find the key
    for (var p = 0u; p < assign_params.capacity; p++) {
        let idx = slot * 2u;
        let stored_key = lookup_table[idx];

        if (stored_key == key) {
            group_ids[tid] = lookup_table[idx + 1u];
            return;
        }

        if (stored_key == 0xFFFFFFFFu) {
            // Empty slot - key not found (shouldn't happen)
            group_ids[tid] = 0xFFFFFFFFu;
            return;
        }

        slot = (slot + 1u) % assign_params.capacity;
    }

    group_ids[tid] = 0xFFFFFFFFu;
}

// ============================================================================
// Phase 3: Per-Group Aggregation (using atomics)
// ============================================================================

struct AggParams {
    size: u32,           // Number of rows
    num_groups: u32,     // Number of unique groups
    agg_type: u32,       // 0=COUNT, 1=SUM, 2=MIN, 3=MAX
}

@group(0) @binding(0) var<uniform> agg_params: AggParams;
@group(0) @binding(1) var<storage, read> agg_group_ids: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> agg_results: array<atomic<u32>>;

// Convert f32 to sortable u32 (for atomic min/max)
fn float_to_sortable(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    // Flip all bits if negative, otherwise flip sign bit
    return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn sortable_to_float(u: u32) -> f32 {
    // Reverse the transformation
    let bits = select(u ^ 0x80000000u, ~u, (u & 0x80000000u) != 0u);
    return bitcast<f32>(bits);
}

// Per-group COUNT (increment counter for each row in group)
@compute @workgroup_size(256)
fn group_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= agg_params.size) { return; }

    let group_id = agg_group_ids[tid];
    if (group_id < agg_params.num_groups) {
        atomicAdd(&agg_results[group_id], 1u);
    }
}

// Per-group SUM (using atomic add on integer representation)
@compute @workgroup_size(256)
fn group_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= agg_params.size) { return; }

    let group_id = agg_group_ids[tid];
    let val = values[tid];

    if (group_id < agg_params.num_groups && !isNan(val)) {
        // Use fixed-point representation for atomic add
        // Multiply by 1000 for 3 decimal places precision
        let fixed_val = i32(val * 1000.0);
        atomicAdd(&agg_results[group_id], u32(fixed_val));
    }
}

// Per-group MIN
@compute @workgroup_size(256)
fn group_min(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= agg_params.size) { return; }

    let group_id = agg_group_ids[tid];
    let val = values[tid];

    if (group_id < agg_params.num_groups && !isNan(val)) {
        let sortable = float_to_sortable(val);
        atomicMin(&agg_results[group_id], sortable);
    }
}

// Per-group MAX
@compute @workgroup_size(256)
fn group_max(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= agg_params.size) { return; }

    let group_id = agg_group_ids[tid];
    let val = values[tid];

    if (group_id < agg_params.num_groups && !isNan(val)) {
        let sortable = float_to_sortable(val);
        atomicMax(&agg_results[group_id], sortable);
    }
}

// ============================================================================
// Initialize Hash Table
// ============================================================================

struct InitParams {
    capacity: u32,
}

@group(0) @binding(0) var<uniform> init_params: InitParams;
@group(0) @binding(1) var<storage, read_write> init_table: array<u32>;

@compute @workgroup_size(256)
fn init_hash_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= init_params.capacity * 2u) { return; }
    // Keys initialized to 0xFFFFFFFF (empty), values to 0
    init_table[idx] = select(0u, 0xFFFFFFFFu, idx % 2u == 0u);
}

// ============================================================================
// Initialize Aggregation Results
// ============================================================================

struct InitAggParams {
    num_groups: u32,
    init_value: u32,  // 0 for COUNT/SUM, MAX_UINT for MIN, 0 for MAX
}

@group(0) @binding(0) var<uniform> init_agg_params: InitAggParams;
@group(0) @binding(1) var<storage, read_write> init_agg_results: array<u32>;

@compute @workgroup_size(256)
fn init_agg_results(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= init_agg_params.num_groups) { return; }
    init_agg_results[idx] = init_agg_params.init_value;
}
