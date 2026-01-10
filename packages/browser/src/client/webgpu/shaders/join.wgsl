// GPU Hash Join Shaders
// Implements parallel hash table build and probe for SQL JOINs

// ============================================================================
// Hash Table Build Phase
// ============================================================================

struct BuildParams {
    size: u32,           // Number of keys to insert
    capacity: u32,       // Hash table capacity (power of 2)
}

@group(0) @binding(0) var<uniform> build_params: BuildParams;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hash_table: array<atomic<u32>>;

// FNV-1a hash function
fn fnv_hash(key: u32) -> u32 {
    var hash = 2166136261u;
    hash ^= (key & 0xFFu);
    hash *= 16777619u;
    hash ^= ((key >> 8u) & 0xFFu);
    hash *= 16777619u;
    hash ^= ((key >> 16u) & 0xFFu);
    hash *= 16777619u;
    hash ^= ((key >> 24u) & 0xFFu);
    hash *= 16777619u;
    return hash;
}

// Build hash table: insert (key, row_index) pairs
// Hash table layout: [key0, idx0, key1, idx1, ...]
// Empty slots have key = 0xFFFFFFFF
@compute @workgroup_size(256)
fn build_hash_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= build_params.size) {
        return;
    }

    let key = keys[tid];
    let row_idx = tid;

    // Compute hash and find slot using linear probing
    var slot = fnv_hash(key) % build_params.capacity;
    let max_probes = build_params.capacity;

    for (var probe = 0u; probe < max_probes; probe++) {
        let table_idx = slot * 2u;

        // Try to claim this slot with atomic compare-exchange
        // 0xFFFFFFFF means empty
        let old = atomicCompareExchangeWeak(&hash_table[table_idx], 0xFFFFFFFFu, key);

        if (old.exchanged) {
            // Successfully claimed slot - store row index
            atomicStore(&hash_table[table_idx + 1u], row_idx);
            return;
        }

        if (old.old_value == key) {
            // Key already exists - for multi-match, we'd need a chain
            // For now, store in next slot (allows duplicate keys)
            slot = (slot + 1u) % build_params.capacity;
            continue;
        }

        // Collision - linear probe to next slot
        slot = (slot + 1u) % build_params.capacity;
    }
    // Table full - shouldn't happen if capacity >= 2 * size
}

// ============================================================================
// Hash Table Probe Phase
// ============================================================================

struct ProbeParams {
    left_size: u32,      // Number of left keys to probe
    capacity: u32,       // Hash table capacity
    max_matches: u32,    // Maximum matches to output
}

@group(0) @binding(0) var<uniform> probe_params: ProbeParams;
@group(0) @binding(1) var<storage, read> left_keys: array<u32>;
@group(0) @binding(2) var<storage, read> probe_table: array<u32>;  // Non-atomic read
@group(0) @binding(3) var<storage, read_write> matches: array<u32>;  // [left_idx, right_idx, ...]
@group(0) @binding(4) var<storage, read_write> match_count: atomic<u32>;

// Probe hash table for matching right rows
@compute @workgroup_size(256)
fn probe_hash_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let left_idx = gid.x;
    if (left_idx >= probe_params.left_size) {
        return;
    }

    let key = left_keys[left_idx];
    var slot = fnv_hash(key) % probe_params.capacity;
    let max_probes = probe_params.capacity;

    for (var probe = 0u; probe < max_probes; probe++) {
        let table_idx = slot * 2u;
        let stored_key = probe_table[table_idx];

        if (stored_key == 0xFFFFFFFFu) {
            // Empty slot - no match found
            return;
        }

        if (stored_key == key) {
            // Match found! Record the pair
            let right_idx = probe_table[table_idx + 1u];
            let out_idx = atomicAdd(&match_count, 1u);

            if (out_idx * 2u + 1u < probe_params.max_matches * 2u) {
                matches[out_idx * 2u] = left_idx;
                matches[out_idx * 2u + 1u] = right_idx;
            }

            // Continue probing for more matches with same key
        }

        slot = (slot + 1u) % probe_params.capacity;
    }
}

// ============================================================================
// Initialize Hash Table (set all slots to empty)
// ============================================================================

struct InitParams {
    capacity: u32,
}

@group(0) @binding(0) var<uniform> init_params: InitParams;
@group(0) @binding(1) var<storage, read_write> init_table: array<u32>;

@compute @workgroup_size(256)
fn init_hash_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= init_params.capacity * 2u) {
        return;
    }

    // Set key slots to 0xFFFFFFFF (empty marker)
    // Set index slots to 0
    if (idx % 2u == 0u) {
        init_table[idx] = 0xFFFFFFFFu;
    } else {
        init_table[idx] = 0u;
    }
}

// ============================================================================
// Left Outer Join - Mark unmatched left rows
// ============================================================================

struct LeftJoinParams {
    left_size: u32,
    match_count: u32,
}

@group(0) @binding(0) var<uniform> left_params: LeftJoinParams;
@group(0) @binding(1) var<storage, read> left_matched: array<u32>;  // Sorted left indices that matched
@group(0) @binding(2) var<storage, read_write> unmatched: array<u32>;
@group(0) @binding(3) var<storage, read_write> unmatched_count: atomic<u32>;

// Find left rows that had no match (for LEFT JOIN null padding)
@compute @workgroup_size(256)
fn find_unmatched_left(@builtin(global_invocation_id) gid: vec3<u32>) {
    let left_idx = gid.x;
    if (left_idx >= left_params.left_size) {
        return;
    }

    // Binary search in matched list
    var lo = 0u;
    var hi = left_params.match_count;
    var found = false;

    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let val = left_matched[mid];
        if (val == left_idx) {
            found = true;
            break;
        } else if (val < left_idx) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    if (!found) {
        let out_idx = atomicAdd(&unmatched_count, 1u);
        unmatched[out_idx] = left_idx;
    }
}
