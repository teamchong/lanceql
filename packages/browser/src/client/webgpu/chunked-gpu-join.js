/**
 * Chunked GPU Hash Join
 *
 * Memory-safe hash join that processes data in chunks.
 * Uses partition-based approach:
 * 1. Hash-partition both tables into N partitions
 * 2. For each partition: build hash table from right, probe with left
 * 3. Stream results to OPFS buffer
 *
 * This ensures each partition fits in GPU memory, preventing OOM.
 *
 * @example
 * const joiner = new ChunkedGPUJoiner();
 * await joiner.init();
 *
 * // Join with chunked iterators
 * const results = await joiner.hashJoin(
 *     leftChunks,   // AsyncIterator<{key: Uint32Array, indices: Uint32Array}>
 *     rightChunks,  // AsyncIterator<{key: Uint32Array, indices: Uint32Array}>
 *     { gpuMemoryBudget: 256 * 1024 * 1024 }
 * );
 *
 * // Stream results
 * for await (const {leftIdx, rightIdx} of results) {
 *     console.log(leftIdx, rightIdx);
 * }
 */

import { OPFSResultBuffer, createTempBuffer } from '../cache/opfs-result-buffer.js';
import { getBufferPool } from './gpu-buffer-pool.js';

// Partition shader - assigns each key to a partition
const PARTITION_SHADER = `
struct Params {
    size: u32,
    num_partitions: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> partition_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> partition_counts: array<atomic<u32>>;

fn hash_partition(key: u32) -> u32 {
    // FNV-1a hash for partitioning
    var h = 2166136261u;
    h ^= (key & 0xFFu); h *= 16777619u;
    h ^= ((key >> 8u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 16u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 24u) & 0xFFu); h *= 16777619u;
    return h;
}

@compute @workgroup_size(256)
fn assign_partitions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let key = keys[idx];
    let part_id = hash_partition(key) % params.num_partitions;
    partition_ids[idx] = part_id;
    atomicAdd(&partition_counts[part_id], 1u);
}
`;

// Hash join shader (same as gpu-joins.js but extracted for clarity)
const JOIN_SHADER = `
struct BuildParams { size: u32, capacity: u32 }
struct ProbeParams { left_size: u32, capacity: u32, max_matches: u32 }

@group(0) @binding(0) var<uniform> build_params: BuildParams;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hash_table: array<atomic<u32>>;

fn fnv_hash(key: u32) -> u32 {
    var h = 2166136261u;
    h ^= (key & 0xFFu); h *= 16777619u;
    h ^= ((key >> 8u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 16u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 24u) & 0xFFu); h *= 16777619u;
    return h;
}

@compute @workgroup_size(256)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= build_params.size) { return; }
    let key = keys[tid];
    var slot = fnv_hash(key) % build_params.capacity;
    for (var p = 0u; p < build_params.capacity; p++) {
        let idx = slot * 2u;
        let old = atomicCompareExchangeWeak(&hash_table[idx], 0xFFFFFFFFu, key);
        if (old.exchanged) {
            atomicStore(&hash_table[idx + 1u], tid);
            return;
        }
        slot = (slot + 1u) % build_params.capacity;
    }
}

@group(0) @binding(0) var<uniform> probe_params: ProbeParams;
@group(0) @binding(1) var<storage, read> left_keys: array<u32>;
@group(0) @binding(2) var<storage, read> probe_table: array<u32>;
@group(0) @binding(3) var<storage, read_write> matches: array<u32>;
@group(0) @binding(4) var<storage, read_write> match_count: atomic<u32>;

@compute @workgroup_size(256)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let left_idx = gid.x;
    if (left_idx >= probe_params.left_size) { return; }
    let key = left_keys[left_idx];
    var slot = fnv_hash(key) % probe_params.capacity;
    for (var p = 0u; p < probe_params.capacity; p++) {
        let idx = slot * 2u;
        let stored = probe_table[idx];
        if (stored == 0xFFFFFFFFu) { return; }
        if (stored == key) {
            let right_idx = probe_table[idx + 1u];
            let out = atomicAdd(&match_count, 1u);
            if (out * 2u + 1u < probe_params.max_matches * 2u) {
                matches[out * 2u] = left_idx;
                matches[out * 2u + 1u] = right_idx;
            }
        }
        slot = (slot + 1u) % probe_params.capacity;
    }
}

struct InitParams { capacity: u32 }
@group(0) @binding(0) var<uniform> init_params: InitParams;
@group(0) @binding(1) var<storage, read_write> table_data: array<u32>;

@compute @workgroup_size(256)
fn clear_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= init_params.capacity * 2u) { return; }
    table_data[idx] = select(0u, 0xFFFFFFFFu, idx % 2u == 0u);
}
`;

// Default memory budget: 256 MB
const DEFAULT_GPU_MEMORY_BUDGET = 256 * 1024 * 1024;
// Minimum partition size before using partitioning
const MIN_PARTITION_THRESHOLD = 100000;

export class ChunkedGPUJoiner {
    constructor() {
        this.device = null;
        this.pipelines = new Map();
        this.available = false;
        this._initPromise = null;
        this.bufferPool = null;
    }

    async init() {
        if (this._initPromise) return this._initPromise;
        this._initPromise = this._doInit();
        return this._initPromise;
    }

    isAvailable() {
        return this.available;
    }

    async _doInit() {
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            console.log('[ChunkedGPUJoiner] WebGPU not available');
            return false;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return false;

            this.device = await adapter.requestDevice({
                requiredLimits: {
                    maxStorageBufferBindingSize: 256 * 1024 * 1024,
                    maxBufferSize: 256 * 1024 * 1024,
                }
            });

            await this._compileShaders();
            this.bufferPool = getBufferPool(this.device);
            this.available = true;
            console.log('[ChunkedGPUJoiner] Initialized');
            return true;
        } catch (e) {
            console.error('[ChunkedGPUJoiner] Init failed:', e);
            return false;
        }
    }

    async _compileShaders() {
        // Partition shader
        const partitionModule = this.device.createShaderModule({ code: PARTITION_SHADER });
        this.pipelines.set('assign_partitions', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: partitionModule, entryPoint: 'assign_partitions' }
        }));

        // Join shaders
        const joinModule = this.device.createShaderModule({ code: JOIN_SHADER });
        this.pipelines.set('build', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: joinModule, entryPoint: 'build' }
        }));
        this.pipelines.set('probe', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: joinModule, entryPoint: 'probe' }
        }));
        this.pipelines.set('clear_table', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: joinModule, entryPoint: 'clear_table' }
        }));
    }

    /**
     * Perform a chunked hash join.
     *
     * Supports both new chunked API and legacy flat array API:
     * - New: hashJoin(leftChunks, rightChunks, options)
     * - Legacy: hashJoin(leftRows, rightRows, leftKey, rightKey, joinType)
     *
     * @param {AsyncIterable|Array} leftChunks - Async chunks or row array
     * @param {AsyncIterable|Array} rightChunks - Async chunks or row array
     * @param {Object|string} optionsOrLeftKey - Options object or left join key (legacy)
     * @param {string} rightKey - Right join key (legacy only)
     * @param {string} joinType - Join type (legacy only)
     * @returns {Promise<OPFSResultBuffer|Object>} Buffer or match indices
     */
    async hashJoin(leftChunks, rightChunks, optionsOrLeftKey = {}, rightKey, joinType) {
        // Detect legacy API: hashJoin(leftRows, rightRows, leftKey, rightKey, joinType?)
        if (typeof optionsOrLeftKey === 'string') {
            return this.hashJoinFlat(leftChunks, rightChunks, optionsOrLeftKey, rightKey, joinType || 'INNER');
        }

        const options = optionsOrLeftKey;
        const gpuMemoryBudget = options.gpuMemoryBudget || DEFAULT_GPU_MEMORY_BUDGET;
        const effectiveJoinType = options.joinType || 'INNER';

        // Collect all data first to determine partitioning strategy
        const leftData = await this._collectChunks(leftChunks);
        const rightData = await this._collectChunks(rightChunks);

        const totalSize = leftData.keys.length + rightData.keys.length;
        const estimatedMemory = this._estimateMemory(leftData.keys.length, rightData.keys.length);

        // If small enough, use simple GPU join
        if (estimatedMemory < gpuMemoryBudget && totalSize < MIN_PARTITION_THRESHOLD) {
            return this._simpleJoin(leftData, rightData, effectiveJoinType);
        }

        // Use partitioned join for large data
        return this._partitionedJoin(leftData, rightData, gpuMemoryBudget, effectiveJoinType);
    }

    /**
     * Collect async chunks into arrays.
     */
    async _collectChunks(chunks) {
        const allKeys = [];
        const allIndices = [];

        for await (const chunk of chunks) {
            allKeys.push(chunk.keys);
            allIndices.push(chunk.indices);
        }

        // Concatenate
        const totalKeys = allKeys.reduce((s, a) => s + a.length, 0);
        const keys = new Uint32Array(totalKeys);
        const indices = new Uint32Array(totalKeys);

        let offset = 0;
        for (let i = 0; i < allKeys.length; i++) {
            keys.set(allKeys[i], offset);
            indices.set(allIndices[i], offset);
            offset += allKeys[i].length;
        }

        return { keys, indices };
    }

    /**
     * Estimate GPU memory needed for join.
     */
    _estimateMemory(leftSize, rightSize) {
        // Hash table: capacity * 2 * 4 bytes (key + index)
        // capacity = rightSize * 4 (25% load factor)
        const hashTableSize = rightSize * 4 * 2 * 4;
        // Keys: leftSize * 4 + rightSize * 4
        const keysSize = (leftSize + rightSize) * 4;
        // Matches buffer: estimate 10x right size
        const matchesSize = rightSize * 10 * 2 * 4;

        return hashTableSize + keysSize + matchesSize;
    }

    /**
     * Simple non-partitioned GPU join.
     */
    async _simpleJoin(leftData, rightData, joinType) {
        const resultBuffer = createTempBuffer('join');
        await resultBuffer.init();

        if (!this.available) {
            // CPU fallback
            const matches = this._cpuHashJoin(leftData, rightData, joinType);
            await resultBuffer.appendMatches(matches);
            await resultBuffer.finalize();
            return resultBuffer;
        }

        const matches = await this._gpuJoinPartition(
            leftData.keys, leftData.indices,
            rightData.keys, rightData.indices,
            joinType
        );

        await resultBuffer.appendMatches(matches);
        await resultBuffer.finalize();
        return resultBuffer;
    }

    /**
     * Partitioned GPU join for large datasets.
     */
    async _partitionedJoin(leftData, rightData, memoryBudget, joinType) {
        // Calculate number of partitions needed
        const estimatedMemory = this._estimateMemory(leftData.keys.length, rightData.keys.length);
        const numPartitions = Math.max(1, Math.ceil(estimatedMemory / memoryBudget) * 2);

        console.log(`[ChunkedGPUJoiner] Using ${numPartitions} partitions for ${leftData.keys.length} x ${rightData.keys.length} join`);

        // Partition both sides
        const leftPartitions = this._partitionData(leftData, numPartitions);
        const rightPartitions = this._partitionData(rightData, numPartitions);

        // Create result buffer
        const resultBuffer = createTempBuffer('join');
        await resultBuffer.init();

        // Join each partition pair
        for (let p = 0; p < numPartitions; p++) {
            const leftPart = leftPartitions[p];
            const rightPart = rightPartitions[p];

            if (leftPart.keys.length === 0 || rightPart.keys.length === 0) {
                continue; // Skip empty partitions
            }

            console.log(`[ChunkedGPUJoiner] Partition ${p}: ${leftPart.keys.length} x ${rightPart.keys.length}`);

            const matches = this.available
                ? await this._gpuJoinPartition(leftPart.keys, leftPart.indices, rightPart.keys, rightPart.indices, joinType)
                : this._cpuHashJoin(leftPart, rightPart, joinType);

            if (matches.length > 0) {
                await resultBuffer.appendMatches(matches);
            }
        }

        await resultBuffer.finalize();
        return resultBuffer;
    }

    /**
     * Partition data by hash of key.
     */
    _partitionData(data, numPartitions) {
        const partitions = Array.from({ length: numPartitions }, () => ({
            keys: [],
            indices: []
        }));

        for (let i = 0; i < data.keys.length; i++) {
            const key = data.keys[i];
            const partition = this._hashPartition(key, numPartitions);
            partitions[partition].keys.push(key);
            partitions[partition].indices.push(data.indices[i]);
        }

        // Convert to typed arrays
        return partitions.map(p => ({
            keys: new Uint32Array(p.keys),
            indices: new Uint32Array(p.indices)
        }));
    }

    /**
     * FNV-1a hash for partitioning.
     */
    _hashPartition(key, numPartitions) {
        let h = 2166136261;
        h ^= (key & 0xFF); h = Math.imul(h, 16777619);
        h ^= ((key >> 8) & 0xFF); h = Math.imul(h, 16777619);
        h ^= ((key >> 16) & 0xFF); h = Math.imul(h, 16777619);
        h ^= ((key >> 24) & 0xFF); h = Math.imul(h, 16777619);
        return (h >>> 0) % numPartitions;
    }

    /**
     * GPU hash join for a single partition.
     */
    async _gpuJoinPartition(leftKeys, leftIndices, rightKeys, rightIndices, joinType) {
        const capacity = this._nextPowerOf2(rightKeys.length * 4);
        const maxMatches = Math.max(leftKeys.length * 10, 100000);

        // Create buffers
        const rightKeysBuf = this._createBuffer(rightKeys, GPUBufferUsage.STORAGE);
        const leftKeysBuf = this._createBuffer(leftKeys, GPUBufferUsage.STORAGE);
        const hashTableBuf = this.device.createBuffer({
            size: capacity * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const matchesBuf = this.device.createBuffer({
            size: maxMatches * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const matchCountBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const stagingBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        try {
            // Clear hash table
            await this._clearHashTable(hashTableBuf, capacity);

            // Build phase
            await this._buildHashTable(rightKeysBuf, hashTableBuf, rightKeys.length, capacity);

            // Probe phase
            await this._probeHashTable(leftKeysBuf, hashTableBuf, matchesBuf, matchCountBuf, leftKeys.length, capacity, maxMatches);

            // Read match count
            const encoder = this.device.createCommandEncoder();
            encoder.copyBufferToBuffer(matchCountBuf, 0, stagingBuf, 0, 4);
            this.device.queue.submit([encoder.finish()]);

            await stagingBuf.mapAsync(GPUMapMode.READ);
            const matchCount = new Uint32Array(stagingBuf.getMappedRange())[0];
            stagingBuf.unmap();

            if (matchCount === 0) {
                return new Uint32Array(0);
            }

            // Read matches
            const actualMatches = Math.min(matchCount, maxMatches);
            const matchStagingBuf = this.device.createBuffer({
                size: actualMatches * 2 * 4,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            const encoder2 = this.device.createCommandEncoder();
            encoder2.copyBufferToBuffer(matchesBuf, 0, matchStagingBuf, 0, actualMatches * 2 * 4);
            this.device.queue.submit([encoder2.finish()]);

            await matchStagingBuf.mapAsync(GPUMapMode.READ);
            const gpuMatches = new Uint32Array(matchStagingBuf.getMappedRange().slice(0));
            matchStagingBuf.unmap();
            matchStagingBuf.destroy();

            // Convert local indices to original indices
            const result = new Uint32Array(actualMatches * 2);
            for (let i = 0; i < actualMatches; i++) {
                const localLeft = gpuMatches[i * 2];
                const localRight = gpuMatches[i * 2 + 1];
                result[i * 2] = leftIndices[localLeft];
                result[i * 2 + 1] = rightIndices[localRight];
            }

            return result;

        } finally {
            rightKeysBuf.destroy();
            leftKeysBuf.destroy();
            hashTableBuf.destroy();
            matchesBuf.destroy();
            matchCountBuf.destroy();
            stagingBuf.destroy();
        }
    }

    async _clearHashTable(hashTableBuf, capacity) {
        const paramsBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([capacity]));

        const pipeline = this.pipelines.get('clear_table');
        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuf } },
                { binding: 1, resource: { buffer: hashTableBuf } },
            ]
        });

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(capacity * 2 / 256));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        paramsBuf.destroy();
    }

    async _buildHashTable(keysBuf, hashTableBuf, size, capacity) {
        const paramsBuf = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([size, capacity]));

        const pipeline = this.pipelines.get('build');
        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuf } },
                { binding: 1, resource: { buffer: keysBuf } },
                { binding: 2, resource: { buffer: hashTableBuf } },
            ]
        });

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(size / 256));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        paramsBuf.destroy();
    }

    async _probeHashTable(leftKeysBuf, hashTableBuf, matchesBuf, matchCountBuf, leftSize, capacity, maxMatches) {
        // Reset match count
        this.device.queue.writeBuffer(matchCountBuf, 0, new Uint32Array([0]));

        const paramsBuf = this.device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([leftSize, capacity, maxMatches]));

        const pipeline = this.pipelines.get('probe');
        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuf } },
                { binding: 1, resource: { buffer: leftKeysBuf } },
                { binding: 2, resource: { buffer: hashTableBuf } },
                { binding: 3, resource: { buffer: matchesBuf } },
                { binding: 4, resource: { buffer: matchCountBuf } },
            ]
        });

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(leftSize / 256));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        paramsBuf.destroy();
    }

    _createBuffer(data, usage) {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: usage | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }

    _nextPowerOf2(n) {
        let p = 1;
        while (p < n) p *= 2;
        return p;
    }

    /**
     * CPU fallback for hash join.
     */
    _cpuHashJoin(leftData, rightData, joinType) {
        const hashTable = new Map();

        // Build from right
        for (let i = 0; i < rightData.keys.length; i++) {
            const key = rightData.keys[i];
            if (!hashTable.has(key)) {
                hashTable.set(key, []);
            }
            hashTable.get(key).push(rightData.indices[i]);
        }

        // Probe from left
        const matches = [];
        for (let i = 0; i < leftData.keys.length; i++) {
            const key = leftData.keys[i];
            const rightIndices = hashTable.get(key);
            if (rightIndices) {
                for (const rightIdx of rightIndices) {
                    matches.push(leftData.indices[i], rightIdx);
                }
            }
        }

        return new Uint32Array(matches);
    }

    // =========================================================================
    // Backward-compatible API (matches original GPUJoiner)
    // =========================================================================

    /**
     * Hash join with flat arrays (backward-compatible API).
     * Automatically chunks data if needed.
     *
     * @param {Object[]} leftRows - Left table rows
     * @param {Object[]} rightRows - Right table rows
     * @param {string} leftKey - Left join key column name
     * @param {string} rightKey - Right join key column name
     * @param {string} joinType - 'INNER' or 'LEFT'
     * @param {Object} options
     * @returns {Promise<{leftIndices: Uint32Array, rightIndices: Uint32Array}>}
     */
    async hashJoinFlat(leftRows, rightRows, leftKey, rightKey, joinType = 'INNER', options = {}) {
        // Extract keys as numeric values
        const leftKeys = this._extractKeysFromRows(leftRows, leftKey);
        const rightKeys = this._extractKeysFromRows(rightRows, rightKey);
        const leftIndices = new Uint32Array(leftRows.length).map((_, i) => i);
        const rightIndices = new Uint32Array(rightRows.length).map((_, i) => i);

        // Create single-chunk iterators
        const self = this;
        async function* leftChunks() {
            yield { keys: leftKeys, indices: leftIndices };
        }
        async function* rightChunks() {
            yield { keys: rightKeys, indices: rightIndices };
        }

        const resultBuffer = await this.hashJoin(leftChunks(), rightChunks(), { joinType, ...options });
        const matches = await resultBuffer.readAll();
        await resultBuffer.close(true);

        // Convert to left/right index arrays
        const numMatches = matches.length / 2;
        const leftMatchIndices = new Uint32Array(numMatches);
        const rightMatchIndices = new Uint32Array(numMatches);

        for (let i = 0; i < numMatches; i++) {
            leftMatchIndices[i] = matches[i * 2];
            rightMatchIndices[i] = matches[i * 2 + 1];
        }

        return {
            leftIndices: leftMatchIndices,
            rightIndices: rightMatchIndices,
            matchCount: numMatches
        };
    }

    /**
     * Extract numeric keys from row objects.
     */
    _extractKeysFromRows(rows, keyName) {
        const keys = new Uint32Array(rows.length);
        for (let i = 0; i < rows.length; i++) {
            const val = rows[i][keyName];
            keys[i] = typeof val === 'number' ? val : this._hashStringKey(String(val));
        }
        return keys;
    }

    /**
     * Simple string hash for non-numeric keys.
     */
    _hashStringKey(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) - hash) + str.charCodeAt(i);
            hash |= 0;
        }
        return hash >>> 0;
    }
}

// Singleton instance
let chunkedJoinerInstance = null;

export function getChunkedGPUJoiner() {
    if (!chunkedJoinerInstance) {
        chunkedJoinerInstance = new ChunkedGPUJoiner();
    }
    return chunkedJoinerInstance;
}

// Backward-compatible aliases (for drop-in replacement of gpu-joins.js)
export const GPUJoiner = ChunkedGPUJoiner;
export const getGPUJoiner = getChunkedGPUJoiner;

// Threshold for GPU join acceleration
const GPU_JOIN_THRESHOLD = 10000;

export function shouldUseGPUJoin(leftSize, rightSize) {
    return leftSize * rightSize >= GPU_JOIN_THRESHOLD * GPU_JOIN_THRESHOLD;
}
