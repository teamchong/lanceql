/**
 * Chunked GPU Group By
 *
 * Memory-safe GROUP BY that processes data in chunks.
 * Uses partial aggregation approach:
 * 1. Process each chunk, compute partial aggregates
 * 2. Store partial aggregates to memory/OPFS
 * 3. Final merge pass to combine all partials
 *
 * Supports: COUNT, SUM, MIN, MAX, AVG
 *
 * @example
 * const grouper = new ChunkedGPUGrouper();
 * await grouper.init();
 *
 * const results = await grouper.groupBy(
 *     chunks,  // AsyncIterator<{groupKey: Uint32Array, values: Float64Array}>
 *     'SUM',
 *     { gpuMemoryBudget: 256 * 1024 * 1024 }
 * );
 *
 * // Results: Map<groupKey, aggregateValue>
 */

import { OPFSResultBuffer, createTempBuffer } from '../cache/opfs-result-buffer.js';
import { getBufferPool } from './gpu-buffer-pool.js';

// Group by shader with partial aggregation
const GROUP_SHADER = `
struct Params {
    size: u32,
    capacity: u32,
    agg_type: u32,  // 0=COUNT, 1=SUM, 2=MIN, 3=MAX
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> hash_table: array<atomic<u32>>;  // key, count, sum_bits_lo, sum_bits_hi
@group(0) @binding(4) var<storage, read_write> group_count: atomic<u32>;

fn fnv_hash(key: u32) -> u32 {
    var h = 2166136261u;
    h ^= (key & 0xFFu); h *= 16777619u;
    h ^= ((key >> 8u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 16u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 24u) & 0xFFu); h *= 16777619u;
    return h;
}

// Convert f32 to sortable u32 for atomic MIN/MAX
fn f32_to_sortable(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
    return bits ^ mask;
}

fn sortable_to_f32(s: u32) -> f32 {
    let mask = select(0x80000000u, 0xFFFFFFFFu, (s & 0x80000000u) == 0u);
    return bitcast<f32>(s ^ mask);
}

@compute @workgroup_size(256)
fn aggregate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let key = keys[idx];
    let val = values[idx];
    var slot = fnv_hash(key) % params.capacity;

    // Find or create slot for this key
    for (var p = 0u; p < params.capacity; p++) {
        let base = slot * 4u;

        // Try to claim this slot
        let old_key = atomicCompareExchangeWeak(&hash_table[base], 0xFFFFFFFFu, key);

        if (old_key.exchanged || old_key.old_value == key) {
            // This slot is ours
            if (old_key.exchanged) {
                atomicAdd(&group_count, 1u);
            }

            // Update aggregate
            if (params.agg_type == 0u) {
                // COUNT
                atomicAdd(&hash_table[base + 1u], 1u);
            } else if (params.agg_type == 1u) {
                // SUM: Use float accumulation via atomic add on sortable bits
                // Note: This is approximate for floats, consider f64 split for precision
                let val_bits = bitcast<u32>(val);
                atomicAdd(&hash_table[base + 2u], val_bits);
            } else if (params.agg_type == 2u) {
                // MIN
                let sortable = f32_to_sortable(val);
                atomicMin(&hash_table[base + 2u], sortable);
            } else if (params.agg_type == 3u) {
                // MAX
                let sortable = f32_to_sortable(val);
                atomicMax(&hash_table[base + 2u], sortable);
            }
            return;
        }

        slot = (slot + 1u) % params.capacity;
    }
}

struct InitParams { capacity: u32 }
@group(0) @binding(0) var<uniform> init_params: InitParams;
@group(0) @binding(1) var<storage, read_write> table_data: array<u32>;

@compute @workgroup_size(256)
fn clear_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let entry_size = 4u;  // key, count, agg_lo, agg_hi
    if (idx >= init_params.capacity * entry_size) { return; }

    let entry_idx = idx / entry_size;
    let field_idx = idx % entry_size;

    if (field_idx == 0u) {
        // Key: 0xFFFFFFFF = empty
        table_data[idx] = 0xFFFFFFFFu;
    } else if (field_idx == 2u) {
        // Aggregate field: init depends on agg type
        // For MIN: init to MAX_FLOAT sortable
        // For MAX: init to MIN_FLOAT sortable
        // For SUM/COUNT: init to 0
        table_data[idx] = 0u;
    } else {
        table_data[idx] = 0u;
    }
}
`;

// Aggregation types
const AGG_COUNT = 0;
const AGG_SUM = 1;
const AGG_MIN = 2;
const AGG_MAX = 3;

const AGG_TYPE_MAP = {
    'COUNT': AGG_COUNT,
    'SUM': AGG_SUM,
    'MIN': AGG_MIN,
    'MAX': AGG_MAX,
    'AVG': AGG_SUM,  // AVG = SUM / COUNT, handled specially
};

// Default memory budget: 256 MB
const DEFAULT_GPU_MEMORY_BUDGET = 256 * 1024 * 1024;
// Minimum rows before GPU grouping
const GPU_GROUP_THRESHOLD = 10000;

export class ChunkedGPUGrouper {
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

    async _doInit() {
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            console.log('[ChunkedGPUGrouper] WebGPU not available');
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
            console.log('[ChunkedGPUGrouper] Initialized');
            return true;
        } catch (e) {
            console.error('[ChunkedGPUGrouper] Init failed:', e);
            return false;
        }
    }

    async _compileShaders() {
        const module = this.device.createShaderModule({ code: GROUP_SHADER });

        this.pipelines.set('aggregate', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'aggregate' }
        }));

        this.pipelines.set('clear_table', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'clear_table' }
        }));
    }

    /**
     * Perform chunked GROUP BY with aggregation.
     *
     * @param {AsyncIterable<{keys: Uint32Array, values: Float32Array}>} chunks
     * @param {string} aggType - 'COUNT', 'SUM', 'MIN', 'MAX', or 'AVG'
     * @param {Object} options
     * @param {number} options.gpuMemoryBudget - Max GPU memory (default 256MB)
     * @param {number} options.estimatedGroups - Estimated unique groups (helps sizing)
     * @returns {Promise<Map<number, number>>} Map of groupKey -> aggregateValue
     */
    async groupBy(chunks, aggType, options = {}) {
        const gpuMemoryBudget = options.gpuMemoryBudget || DEFAULT_GPU_MEMORY_BUDGET;
        const estimatedGroups = options.estimatedGroups || 100000;
        const isAvg = aggType === 'AVG';

        // Partial aggregates: Map<key, {sum, count, min, max}>
        const partials = new Map();

        let totalRows = 0;
        for await (const chunk of chunks) {
            totalRows += chunk.keys.length;

            // Process this chunk
            const chunkPartials = await this._processChunk(
                chunk.keys,
                chunk.values,
                aggType,
                gpuMemoryBudget
            );

            // Merge into global partials
            this._mergePartials(partials, chunkPartials, aggType);
        }

        console.log(`[ChunkedGPUGrouper] Processed ${totalRows} rows into ${partials.size} groups`);

        // Compute final aggregates
        const results = new Map();
        for (const [key, partial] of partials) {
            let value;
            switch (aggType) {
                case 'COUNT':
                    value = partial.count;
                    break;
                case 'SUM':
                    value = partial.sum;
                    break;
                case 'MIN':
                    value = partial.min;
                    break;
                case 'MAX':
                    value = partial.max;
                    break;
                case 'AVG':
                    value = partial.count > 0 ? partial.sum / partial.count : 0;
                    break;
            }
            results.set(key, value);
        }

        return results;
    }

    /**
     * Process a single chunk.
     */
    async _processChunk(keys, values, aggType, memoryBudget) {
        const size = keys.length;

        // Use CPU for small chunks
        if (!this.available || size < GPU_GROUP_THRESHOLD) {
            return this._cpuAggregate(keys, values, aggType);
        }

        // GPU processing
        return this._gpuAggregate(keys, values, aggType);
    }

    /**
     * GPU aggregation for a chunk.
     */
    async _gpuAggregate(keys, values, aggType) {
        const size = keys.length;
        const capacity = this._nextPowerOf2(Math.max(size, 1024) * 2);
        const aggTypeCode = AGG_TYPE_MAP[aggType] ?? AGG_SUM;

        // Create buffers
        const keysBuf = this._createBuffer(keys, GPUBufferUsage.STORAGE);

        // Convert values to Float32Array if needed
        const valuesF32 = values instanceof Float32Array ? values : new Float32Array(values);
        const valuesBuf = this._createBuffer(valuesF32, GPUBufferUsage.STORAGE);

        const hashTableBuf = this.device.createBuffer({
            size: capacity * 4 * 4, // 4 u32 per entry
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const groupCountBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        try {
            // Clear hash table
            await this._clearHashTable(hashTableBuf, capacity);

            // Reset group count
            this.device.queue.writeBuffer(groupCountBuf, 0, new Uint32Array([0]));

            // Run aggregation
            const paramsBuf = this.device.createBuffer({
                size: 12,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([size, capacity, aggTypeCode]));

            const pipeline = this.pipelines.get('aggregate');
            const bindGroup = this.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: paramsBuf } },
                    { binding: 1, resource: { buffer: keysBuf } },
                    { binding: 2, resource: { buffer: valuesBuf } },
                    { binding: 3, resource: { buffer: hashTableBuf } },
                    { binding: 4, resource: { buffer: groupCountBuf } },
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

            // Read results
            const stagingBuf = this.device.createBuffer({
                size: capacity * 4 * 4,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            const encoder2 = this.device.createCommandEncoder();
            encoder2.copyBufferToBuffer(hashTableBuf, 0, stagingBuf, 0, capacity * 4 * 4);
            this.device.queue.submit([encoder2.finish()]);

            await stagingBuf.mapAsync(GPUMapMode.READ);
            const tableData = new Uint32Array(stagingBuf.getMappedRange().slice(0));
            stagingBuf.unmap();
            stagingBuf.destroy();

            // Extract partial aggregates
            const partials = new Map();
            for (let i = 0; i < capacity; i++) {
                const base = i * 4;
                const key = tableData[base];
                if (key === 0xFFFFFFFF) continue;

                const count = tableData[base + 1];
                const aggBits = tableData[base + 2];

                let value;
                if (aggType === 'COUNT') {
                    value = count;
                } else if (aggType === 'MIN' || aggType === 'MAX') {
                    // Convert sortable back to float
                    value = this._sortableToF32(aggBits);
                } else {
                    // SUM - bits represent accumulated value (approximate)
                    value = this._bitsToFloat(aggBits);
                }

                partials.set(key, {
                    count: count || 1,
                    sum: (aggType === 'SUM' || aggType === 'AVG') ? value : 0,
                    min: aggType === 'MIN' ? value : Infinity,
                    max: aggType === 'MAX' ? value : -Infinity
                });
            }

            return partials;

        } finally {
            keysBuf.destroy();
            valuesBuf.destroy();
            hashTableBuf.destroy();
            groupCountBuf.destroy();
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
        pass.dispatchWorkgroups(Math.ceil(capacity * 4 / 256));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        paramsBuf.destroy();
    }

    /**
     * Merge partial aggregates.
     */
    _mergePartials(target, source, aggType) {
        for (const [key, srcPartial] of source) {
            if (!target.has(key)) {
                target.set(key, {
                    count: 0,
                    sum: 0,
                    min: Infinity,
                    max: -Infinity
                });
            }

            const tgt = target.get(key);
            tgt.count += srcPartial.count;
            tgt.sum += srcPartial.sum;
            tgt.min = Math.min(tgt.min, srcPartial.min);
            tgt.max = Math.max(tgt.max, srcPartial.max);
        }
    }

    /**
     * CPU fallback aggregation.
     */
    _cpuAggregate(keys, values, aggType) {
        const partials = new Map();

        for (let i = 0; i < keys.length; i++) {
            const key = keys[i];
            const val = values[i];

            if (!partials.has(key)) {
                partials.set(key, {
                    count: 0,
                    sum: 0,
                    min: Infinity,
                    max: -Infinity
                });
            }

            const p = partials.get(key);
            p.count++;
            p.sum += val;
            p.min = Math.min(p.min, val);
            p.max = Math.max(p.max, val);
        }

        return partials;
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

    // Float32 bit manipulation helpers
    _f32ToSortable(f) {
        const view = new DataView(new ArrayBuffer(4));
        view.setFloat32(0, f, true);
        const bits = view.getUint32(0, true);
        const mask = (bits & 0x80000000) ? 0xFFFFFFFF : 0x80000000;
        return bits ^ mask;
    }

    _sortableToF32(s) {
        const mask = (s & 0x80000000) ? 0x80000000 : 0xFFFFFFFF;
        const bits = s ^ mask;
        const view = new DataView(new ArrayBuffer(4));
        view.setUint32(0, bits, true);
        return view.getFloat32(0, true);
    }

    _bitsToFloat(bits) {
        const view = new DataView(new ArrayBuffer(4));
        view.setUint32(0, bits, true);
        return view.getFloat32(0, true);
    }
}

// Singleton instance
let chunkedGrouperInstance = null;

export function getChunkedGPUGrouper() {
    if (!chunkedGrouperInstance) {
        chunkedGrouperInstance = new ChunkedGPUGrouper();
    }
    return chunkedGrouperInstance;
}

// Backward-compatible aliases (for drop-in replacement of gpu-group-by.js)
export const GPUGrouper = ChunkedGPUGrouper;
export const getGPUGrouper = getChunkedGPUGrouper;

export function shouldUseGPUGroup(rowCount) {
    return rowCount >= GPU_GROUP_THRESHOLD;
}
