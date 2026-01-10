/**
 * Chunked GPU Sort
 *
 * Memory-safe sorting that handles large datasets.
 *
 * Two modes:
 * 1. ORDER BY ... LIMIT K: Uses top-K selection (very efficient)
 * 2. Full sort: Uses external merge sort via OPFS
 *
 * The LIMIT optimization is key:
 *   ORDER BY score DESC LIMIT 100
 *   → Only need top 100, not full sort of 10M rows
 *   → Process chunks, keep running top-100
 *
 * @example
 * const sorter = new ChunkedGPUSorter();
 * await sorter.init();
 *
 * // With LIMIT (fast)
 * const top100 = await sorter.sortWithLimit(chunks, 100, true);
 *
 * // Full sort (uses OPFS for merge)
 * const sorted = await sorter.fullSort(chunks, true);
 */

import { OPFSResultBuffer, createTempBuffer } from '../cache/opfs-result-buffer.js';
import { getBufferPool } from './gpu-buffer-pool.js';

// Bitonic sort shader
const BITONIC_SORT_SHADER = `
struct Params {
    size: u32,
    stage: u32,
    step: u32,
    descending: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;

@compute @workgroup_size(256)
fn bitonic_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let size = params.size;
    let stage = params.stage;
    let step = params.step;
    let desc = params.descending == 1u;

    let half_step = 1u << step;
    let full_step = half_step << 1u;

    let block = tid / half_step;
    let idx = tid % half_step;
    let i = block * full_step + idx;
    let j = i + half_step;

    if (j >= size) { return; }

    let direction = ((i >> (stage + 1u)) & 1u) == 0u;
    let ascending = direction != desc;

    let ki = keys[i];
    let kj = keys[j];
    let should_swap = select(ki > kj, ki < kj, ascending);

    if (should_swap) {
        keys[i] = kj;
        keys[j] = ki;
        let ii = indices[i];
        indices[i] = indices[j];
        indices[j] = ii;
    }
}
`;

// Default memory budget: 256 MB
const DEFAULT_GPU_MEMORY_BUDGET = 256 * 1024 * 1024;
// Minimum size for GPU sort
const GPU_SORT_THRESHOLD = 10000;
// Bytes per element (f32 key + u32 index)
const BYTES_PER_ELEMENT = 8;

export class ChunkedGPUSorter {
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
            console.log('[ChunkedGPUSorter] WebGPU not available');
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
            console.log('[ChunkedGPUSorter] Initialized');
            return true;
        } catch (e) {
            console.error('[ChunkedGPUSorter] Init failed:', e);
            return false;
        }
    }

    async _compileShaders() {
        const module = this.device.createShaderModule({ code: BITONIC_SORT_SHADER });
        this.pipelines.set('bitonic_step', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'bitonic_step' }
        }));
    }

    /**
     * Calculate optimal chunk size for sorting.
     * @param {number} memoryBudget - GPU memory budget
     * @returns {number} Elements per chunk
     */
    calculateChunkSize(memoryBudget = DEFAULT_GPU_MEMORY_BUDGET) {
        // Need 2x memory for double buffering in bitonic sort
        const availableMemory = memoryBudget * 0.4; // 40% to be safe
        return Math.floor(availableMemory / BYTES_PER_ELEMENT);
    }

    /**
     * Sort with LIMIT - only returns top K elements.
     * Much faster than full sort when K << N.
     *
     * @param {AsyncIterable<{keys: Float32Array, indices: Uint32Array}>} chunks - Data chunks
     * @param {number} limit - Number of results (K)
     * @param {boolean} descending - Sort order
     * @param {Object} options
     * @returns {Promise<{keys: Float32Array, indices: Uint32Array}>}
     */
    async sortWithLimit(chunks, limit, descending = true, options = {}) {
        const k = limit;

        // Running top-K across all chunks
        let runningTopK = {
            keys: new Float32Array(0),
            indices: new Uint32Array(0)
        };

        let totalElements = 0;

        for await (const chunk of chunks) {
            totalElements += chunk.keys.length;

            // Combine with running top-K
            const combined = this._combineChunks(runningTopK, chunk);

            // Sort combined and take top K
            const sorted = this._cpuSort(combined.keys, combined.indices, descending);

            // Keep only top K
            const actualK = Math.min(k, sorted.keys.length);
            runningTopK = {
                keys: sorted.keys.slice(0, actualK),
                indices: sorted.indices.slice(0, actualK)
            };
        }

        console.log(`[ChunkedGPUSorter] sortWithLimit: ${totalElements} elements → top ${runningTopK.keys.length}`);

        return runningTopK;
    }

    /**
     * Full sort using external merge sort.
     * Uses OPFS to store intermediate sorted chunks.
     *
     * @param {AsyncIterable<{keys: Float32Array, indices: Uint32Array}>} chunks - Data chunks
     * @param {boolean} descending - Sort order
     * @param {Object} options
     * @returns {Promise<OPFSResultBuffer>} Buffer containing sorted indices
     */
    async fullSort(chunks, descending = true, options = {}) {
        const gpuMemoryBudget = options.gpuMemoryBudget || DEFAULT_GPU_MEMORY_BUDGET;
        const chunkSize = this.calculateChunkSize(gpuMemoryBudget);

        // Phase 1: Sort each chunk and store to OPFS
        const sortedChunks = [];
        let chunkIndex = 0;

        for await (const chunk of chunks) {
            console.log(`[ChunkedGPUSorter] Sorting chunk ${chunkIndex}: ${chunk.keys.length} elements`);

            // Sort this chunk
            const sorted = await this._sortChunk(chunk.keys, chunk.indices, descending);

            // Store to OPFS
            const chunkBuffer = createTempBuffer(`sort-chunk-${chunkIndex}`);
            await chunkBuffer.init(4); // u32 indices

            // Store as key-index pairs (interleaved for merge)
            const interleaved = new Float32Array(sorted.keys.length * 2);
            for (let i = 0; i < sorted.keys.length; i++) {
                interleaved[i * 2] = sorted.keys[i];
                interleaved[i * 2 + 1] = new DataView(new Uint32Array([sorted.indices[i]]).buffer).getFloat32(0, true);
            }
            await chunkBuffer.appendMatches(new Uint32Array(interleaved.buffer));
            await chunkBuffer.finalize();

            sortedChunks.push({
                buffer: chunkBuffer,
                length: sorted.keys.length
            });

            chunkIndex++;
        }

        // Phase 2: K-way merge
        console.log(`[ChunkedGPUSorter] Merging ${sortedChunks.length} sorted chunks`);

        const resultBuffer = createTempBuffer('sort-result');
        await resultBuffer.init(4);

        await this._kWayMerge(sortedChunks, resultBuffer, descending);

        // Cleanup chunk buffers
        for (const chunk of sortedChunks) {
            await chunk.buffer.close(true);
        }

        await resultBuffer.finalize();
        return resultBuffer;
    }

    /**
     * Sort a single chunk.
     */
    async _sortChunk(keys, indices, descending) {
        const size = keys.length;

        // CPU sort for small chunks or no GPU
        if (!this.available || size < GPU_SORT_THRESHOLD) {
            return this._cpuSort(keys, indices, descending);
        }

        // GPU bitonic sort
        return this._gpuBitonicSort(keys, indices, descending);
    }

    /**
     * GPU bitonic sort.
     */
    async _gpuBitonicSort(keys, indices, descending) {
        const size = keys.length;

        // Pad to power of 2
        const paddedSize = this._nextPowerOf2(size);
        const paddedKeys = new Float32Array(paddedSize);
        const paddedIndices = new Uint32Array(paddedSize);

        paddedKeys.set(keys);
        paddedIndices.set(indices);

        // Fill padding with extreme values
        const padValue = descending ? -Infinity : Infinity;
        for (let i = size; i < paddedSize; i++) {
            paddedKeys[i] = padValue;
            paddedIndices[i] = 0xFFFFFFFF;
        }

        // Create GPU buffers
        const keysBuffer = this.device.createBuffer({
            size: paddedKeys.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.device.queue.writeBuffer(keysBuffer, 0, paddedKeys);

        const indicesBuffer = this.device.createBuffer({
            size: paddedIndices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.device.queue.writeBuffer(indicesBuffer, 0, paddedIndices);

        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        try {
            const pipeline = this.pipelines.get('bitonic_step');

            // Bitonic sort stages
            const numStages = Math.ceil(Math.log2(paddedSize));
            for (let stage = 0; stage < numStages; stage++) {
                for (let step = stage; step >= 0; step--) {
                    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([
                        paddedSize, stage, step, descending ? 1 : 0
                    ]));

                    const bindGroup = this.device.createBindGroup({
                        layout: pipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: paramsBuffer } },
                            { binding: 1, resource: { buffer: keysBuffer } },
                            { binding: 2, resource: { buffer: indicesBuffer } },
                        ]
                    });

                    const encoder = this.device.createCommandEncoder();
                    const pass = encoder.beginComputePass();
                    pass.setPipeline(pipeline);
                    pass.setBindGroup(0, bindGroup);
                    pass.dispatchWorkgroups(Math.ceil(paddedSize / 2 / 256));
                    pass.end();
                    this.device.queue.submit([encoder.finish()]);
                }
            }

            await this.device.queue.onSubmittedWorkDone();

            // Read results
            const keysStagingBuffer = this.device.createBuffer({
                size: size * 4,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
            const indicesStagingBuffer = this.device.createBuffer({
                size: size * 4,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            const encoder = this.device.createCommandEncoder();
            encoder.copyBufferToBuffer(keysBuffer, 0, keysStagingBuffer, 0, size * 4);
            encoder.copyBufferToBuffer(indicesBuffer, 0, indicesStagingBuffer, 0, size * 4);
            this.device.queue.submit([encoder.finish()]);

            await keysStagingBuffer.mapAsync(GPUMapMode.READ);
            await indicesStagingBuffer.mapAsync(GPUMapMode.READ);

            const sortedKeys = new Float32Array(keysStagingBuffer.getMappedRange().slice(0));
            const sortedIndices = new Uint32Array(indicesStagingBuffer.getMappedRange().slice(0));

            keysStagingBuffer.unmap();
            indicesStagingBuffer.unmap();
            keysStagingBuffer.destroy();
            indicesStagingBuffer.destroy();

            return { keys: sortedKeys, indices: sortedIndices };

        } finally {
            keysBuffer.destroy();
            indicesBuffer.destroy();
            paramsBuffer.destroy();
        }
    }

    /**
     * CPU sort fallback.
     */
    _cpuSort(keys, indices, descending) {
        const pairs = [];
        for (let i = 0; i < keys.length; i++) {
            pairs.push({ key: keys[i], index: indices[i] });
        }

        pairs.sort((a, b) => descending ? b.key - a.key : a.key - b.key);

        return {
            keys: new Float32Array(pairs.map(p => p.key)),
            indices: new Uint32Array(pairs.map(p => p.index))
        };
    }

    /**
     * Combine two chunks.
     */
    _combineChunks(a, b) {
        const keys = new Float32Array(a.keys.length + b.keys.length);
        const indices = new Uint32Array(a.indices.length + b.indices.length);

        keys.set(a.keys);
        keys.set(b.keys, a.keys.length);
        indices.set(a.indices);
        indices.set(b.indices, a.indices.length);

        return { keys, indices };
    }

    /**
     * K-way merge of sorted chunks.
     */
    async _kWayMerge(sortedChunks, resultBuffer, descending) {
        // For simplicity, use a priority queue approach
        // Each chunk is read as a stream

        const streams = [];
        for (const chunk of sortedChunks) {
            const data = await chunk.buffer.readAll();
            // Data is interleaved key-index pairs as Float32
            const numPairs = data.length / 2;
            streams.push({
                data,
                length: numPairs,
                pos: 0
            });
        }

        // Simple merge - for production, use a heap
        while (true) {
            // Find best element across all streams
            let bestStream = -1;
            let bestKey = descending ? -Infinity : Infinity;

            for (let i = 0; i < streams.length; i++) {
                const s = streams[i];
                if (s.pos >= s.length) continue;

                const view = new DataView(s.data.buffer, s.data.byteOffset);
                const key = view.getFloat32(s.pos * 8, true);

                const better = descending ? key > bestKey : key < bestKey;
                if (better) {
                    bestKey = key;
                    bestStream = i;
                }
            }

            if (bestStream === -1) break;

            // Output this element's index
            const s = streams[bestStream];
            const view = new DataView(s.data.buffer, s.data.byteOffset);
            const indexBits = view.getUint32(s.pos * 8 + 4, true);
            await resultBuffer.appendMatches(new Uint32Array([indexBits]));

            s.pos++;
        }
    }

    _nextPowerOf2(n) {
        let p = 1;
        while (p < n) p *= 2;
        return p;
    }

    isAvailable() {
        return this.available;
    }
}

// Singleton instance
let chunkedSorterInstance = null;

export function getChunkedGPUSorter() {
    if (!chunkedSorterInstance) {
        chunkedSorterInstance = new ChunkedGPUSorter();
    }
    return chunkedSorterInstance;
}
