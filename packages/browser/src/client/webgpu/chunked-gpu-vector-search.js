/**
 * Chunked GPU Vector Search
 *
 * Memory-safe vector search that processes vectors in chunks.
 * Prevents OOM when searching millions of high-dimensional vectors.
 *
 * Memory calculation:
 *   1M vectors × 384 dims × 4 bytes = 1.5GB (won't fit in GPU)
 *
 * Solution: Process 100K vectors at a time:
 *   100K × 384 × 4 = 150MB per chunk (fits in 256MB budget)
 *
 * Algorithm:
 *   1. For each chunk of vectors:
 *      - Compute distances from query to chunk
 *      - Get chunk's top-K
 *   2. Merge all chunk top-Ks to get global top-K
 *
 * @example
 * const search = new ChunkedGPUVectorSearch();
 * await search.init();
 *
 * // Search with async iterator of vector chunks
 * const results = await search.search(
 *     queryVector,
 *     vectorChunks,  // AsyncIterator<Float32Array[]>
 *     { k: 100, metric: DistanceMetric.COSINE }
 * );
 * // results: { indices: Uint32Array, scores: Float32Array }
 */

import { getBufferPool } from './gpu-buffer-pool.js';

// Distance metrics enum
export const DistanceMetric = {
    COSINE: 0,
    L2: 1,
    DOT_PRODUCT: 2,
};

// Vector distance shader
const DISTANCE_SHADER = `
struct Params {
    dim: u32,
    num_vectors: u32,
    metric: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;

var<workgroup> shared_query: array<f32, 512>;

@compute @workgroup_size(256)
fn compute_distances(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    let vec_idx = gid.x;
    if (vec_idx >= params.num_vectors) { return; }

    let dim = params.dim;
    let tid = lid.x;

    // Load query into shared memory
    for (var i = tid; i < dim && i < 512u; i += 256u) {
        shared_query[i] = query[i];
    }
    workgroupBarrier();

    let vec_offset = vec_idx * dim;
    var result: f32 = 0.0;

    if (params.metric == 1u) {
        // L2 distance
        var sum: f32 = 0.0;
        for (var i = 0u; i < dim; i++) {
            let d = shared_query[i] - vectors[vec_offset + i];
            sum += d * d;
        }
        result = sqrt(sum);
    } else {
        // Cosine / Dot product
        var dot: f32 = 0.0;
        for (var i = 0u; i < dim; i++) {
            dot += shared_query[i] * vectors[vec_offset + i];
        }
        result = dot;
    }

    distances[vec_idx] = result;
}
`;

// Top-K merge shader (for merging chunk results)
const TOPK_MERGE_SHADER = `
struct Params {
    total_candidates: u32,
    k: u32,
    descending: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> scores: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_indices: array<u32>;

// Simple insertion sort for small K
@compute @workgroup_size(1)
fn merge_topk() {
    let k = params.k;
    let n = params.total_candidates;
    let desc = params.descending == 1u;

    // Initialize output with worst values
    let sentinel = select(3.4028235e+38, -3.4028235e+38, desc);
    for (var i = 0u; i < k; i++) {
        out_scores[i] = sentinel;
        out_indices[i] = 0xFFFFFFFFu;
    }

    // Insertion sort each candidate into top-K
    for (var i = 0u; i < n; i++) {
        let score = scores[i];
        let idx = indices[i];

        // Find insertion position
        var pos = k;
        for (var j = 0u; j < k; j++) {
            let better = select(score < out_scores[j], score > out_scores[j], desc);
            if (better) {
                pos = j;
                break;
            }
        }

        // Insert if in top-K
        if (pos < k) {
            // Shift elements down
            for (var j = k - 1u; j > pos; j--) {
                out_scores[j] = out_scores[j - 1u];
                out_indices[j] = out_indices[j - 1u];
            }
            out_scores[pos] = score;
            out_indices[pos] = idx;
        }
    }
}
`;

// Default memory budget: 256 MB
const DEFAULT_GPU_MEMORY_BUDGET = 256 * 1024 * 1024;
// Bytes per f32
const FLOAT_SIZE = 4;

export class ChunkedGPUVectorSearch {
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
            console.log('[ChunkedGPUVectorSearch] WebGPU not available');
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
            console.log('[ChunkedGPUVectorSearch] Initialized');
            return true;
        } catch (e) {
            console.error('[ChunkedGPUVectorSearch] Init failed:', e);
            return false;
        }
    }

    async _compileShaders() {
        const distanceModule = this.device.createShaderModule({ code: DISTANCE_SHADER });
        this.pipelines.set('compute_distances', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: distanceModule, entryPoint: 'compute_distances' }
        }));

        const mergeModule = this.device.createShaderModule({ code: TOPK_MERGE_SHADER });
        this.pipelines.set('merge_topk', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: mergeModule, entryPoint: 'merge_topk' }
        }));
    }

    /**
     * Calculate optimal chunk size based on memory budget and vector dimensions.
     * @param {number} dim - Vector dimensions
     * @param {number} memoryBudget - GPU memory budget in bytes
     * @returns {number} Vectors per chunk
     */
    calculateChunkSize(dim, memoryBudget = DEFAULT_GPU_MEMORY_BUDGET) {
        // Memory per vector: dim floats + 1 distance float + some overhead
        const bytesPerVector = (dim + 1) * FLOAT_SIZE;
        // Add query vector (dim floats) and buffer overhead (~20%)
        const overhead = dim * FLOAT_SIZE + memoryBudget * 0.2;
        const availableMemory = memoryBudget - overhead;

        // Calculate chunk size
        const chunkSize = Math.floor(availableMemory / bytesPerVector);

        // Clamp to reasonable range
        return Math.max(1000, Math.min(chunkSize, 500000));
    }

    /**
     * Search for top-K nearest vectors using chunked processing.
     *
     * @param {Float32Array} queryVec - Query vector
     * @param {AsyncIterable<{vectors: Float32Array[], startIndex: number}>} chunks - Vector chunks
     * @param {Object} options
     * @param {number} options.k - Number of results (default 10)
     * @param {number} options.metric - Distance metric (default COSINE)
     * @param {number} options.gpuMemoryBudget - GPU memory budget (default 256MB)
     * @returns {Promise<{indices: Uint32Array, scores: Float32Array}>}
     */
    async search(queryVec, chunks, options = {}) {
        const k = options.k || 10;
        const metric = options.metric ?? DistanceMetric.COSINE;
        const descending = metric === DistanceMetric.COSINE || metric === DistanceMetric.DOT_PRODUCT;

        // Collect chunk top-Ks
        const chunkResults = [];
        let totalVectors = 0;

        for await (const chunk of chunks) {
            const { vectors, startIndex } = chunk;
            const numVectors = vectors.length;

            console.log(`[ChunkedGPUVectorSearch] Processing chunk: ${numVectors} vectors, startIndex=${startIndex}`);

            // Compute distances for this chunk
            const distances = await this._computeChunkDistances(queryVec, vectors, metric);

            // Get chunk's top-K
            const chunkTopK = this._cpuTopK(distances, startIndex, k, descending);
            chunkResults.push(chunkTopK);

            totalVectors += numVectors;
        }

        console.log(`[ChunkedGPUVectorSearch] Processed ${totalVectors} total vectors in ${chunkResults.length} chunks`);

        // Merge all chunk top-Ks
        return this._mergeTopK(chunkResults, k, descending);
    }

    /**
     * Convenience method for searching a flat array of vectors.
     * Automatically chunks the data.
     *
     * @param {Float32Array} queryVec - Query vector
     * @param {Float32Array[]} vectors - Array of vectors
     * @param {Object} options
     * @returns {Promise<{indices: Uint32Array, scores: Float32Array}>}
     */
    async searchFlat(queryVec, vectors, options = {}) {
        const k = options.k || 10;
        const metric = options.metric ?? DistanceMetric.COSINE;
        const gpuMemoryBudget = options.gpuMemoryBudget || DEFAULT_GPU_MEMORY_BUDGET;

        const dim = queryVec.length;
        const chunkSize = this.calculateChunkSize(dim, gpuMemoryBudget);

        console.log(`[ChunkedGPUVectorSearch] Chunk size: ${chunkSize} vectors (dim=${dim})`);

        // Create async iterator over chunks
        const self = this;
        async function* generateChunks() {
            for (let i = 0; i < vectors.length; i += chunkSize) {
                const end = Math.min(i + chunkSize, vectors.length);
                yield {
                    vectors: vectors.slice(i, end),
                    startIndex: i
                };
            }
        }

        return this.search(queryVec, generateChunks(), { k, metric, gpuMemoryBudget });
    }

    /**
     * Compute distances for a chunk of vectors.
     */
    async _computeChunkDistances(queryVec, vectors, metric) {
        const dim = queryVec.length;
        const numVectors = vectors.length;

        // CPU fallback for small chunks or no GPU
        if (!this.available || numVectors < 1000) {
            return this._cpuDistances(queryVec, vectors, metric);
        }

        // Flatten vectors
        const flatVectors = new Float32Array(numVectors * dim);
        for (let i = 0; i < numVectors; i++) {
            flatVectors.set(vectors[i], i * dim);
        }

        // Create GPU buffers
        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([dim, numVectors, metric, 0]));

        const queryBuffer = this.device.createBuffer({
            size: queryVec.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(queryBuffer, 0, queryVec);

        const vectorsBuffer = this.device.createBuffer({
            size: flatVectors.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(vectorsBuffer, 0, flatVectors);

        const distancesBuffer = this.device.createBuffer({
            size: numVectors * FLOAT_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        try {
            // Run distance computation
            const pipeline = this.pipelines.get('compute_distances');
            const bindGroup = this.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: paramsBuffer } },
                    { binding: 1, resource: { buffer: queryBuffer } },
                    { binding: 2, resource: { buffer: vectorsBuffer } },
                    { binding: 3, resource: { buffer: distancesBuffer } },
                ]
            });

            const encoder = this.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(numVectors / 256));
            pass.end();
            this.device.queue.submit([encoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            // Read distances
            const stagingBuffer = this.device.createBuffer({
                size: numVectors * FLOAT_SIZE,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            const encoder2 = this.device.createCommandEncoder();
            encoder2.copyBufferToBuffer(distancesBuffer, 0, stagingBuffer, 0, numVectors * FLOAT_SIZE);
            this.device.queue.submit([encoder2.finish()]);

            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const distances = new Float32Array(stagingBuffer.getMappedRange().slice(0));
            stagingBuffer.unmap();
            stagingBuffer.destroy();

            return distances;

        } finally {
            paramsBuffer.destroy();
            queryBuffer.destroy();
            vectorsBuffer.destroy();
            distancesBuffer.destroy();
        }
    }

    /**
     * CPU distance computation fallback.
     */
    _cpuDistances(queryVec, vectors, metric) {
        const distances = new Float32Array(vectors.length);
        const dim = queryVec.length;

        for (let i = 0; i < vectors.length; i++) {
            const vec = vectors[i];
            if (metric === DistanceMetric.L2) {
                let sum = 0;
                for (let j = 0; j < dim; j++) {
                    const d = queryVec[j] - vec[j];
                    sum += d * d;
                }
                distances[i] = Math.sqrt(sum);
            } else {
                // Cosine / Dot product
                let dot = 0;
                for (let j = 0; j < dim; j++) {
                    dot += queryVec[j] * vec[j];
                }
                distances[i] = dot;
            }
        }

        return distances;
    }

    /**
     * CPU top-K selection for a chunk.
     */
    _cpuTopK(distances, startIndex, k, descending) {
        // Create index-score pairs
        const pairs = [];
        for (let i = 0; i < distances.length; i++) {
            pairs.push({ index: startIndex + i, score: distances[i] });
        }

        // Sort by score
        pairs.sort((a, b) => descending ? b.score - a.score : a.score - b.score);

        // Take top K
        const topK = pairs.slice(0, Math.min(k, pairs.length));

        return {
            indices: new Uint32Array(topK.map(p => p.index)),
            scores: new Float32Array(topK.map(p => p.score))
        };
    }

    /**
     * Merge top-K results from multiple chunks.
     */
    _mergeTopK(chunkResults, k, descending) {
        if (chunkResults.length === 0) {
            return { indices: new Uint32Array(0), scores: new Float32Array(0) };
        }

        if (chunkResults.length === 1) {
            return chunkResults[0];
        }

        // Collect all candidates
        const candidates = [];
        for (const result of chunkResults) {
            for (let i = 0; i < result.indices.length; i++) {
                candidates.push({
                    index: result.indices[i],
                    score: result.scores[i]
                });
            }
        }

        // Sort all candidates
        candidates.sort((a, b) => descending ? b.score - a.score : a.score - b.score);

        // Take global top-K
        const topK = candidates.slice(0, Math.min(k, candidates.length));

        return {
            indices: new Uint32Array(topK.map(p => p.index)),
            scores: new Float32Array(topK.map(p => p.score))
        };
    }

    /**
     * Check if GPU is available.
     */
    isAvailable() {
        return this.available;
    }
}

// Singleton instance
let chunkedVectorSearchInstance = null;

export function getChunkedGPUVectorSearch() {
    if (!chunkedVectorSearchInstance) {
        chunkedVectorSearchInstance = new ChunkedGPUVectorSearch();
    }
    return chunkedVectorSearchInstance;
}

/**
 * Estimate memory needed for vector search.
 * @param {number} numVectors - Number of vectors
 * @param {number} dim - Vector dimensions
 * @returns {number} Bytes needed
 */
export function estimateVectorSearchMemory(numVectors, dim) {
    // vectors + distances + query + overhead
    return numVectors * dim * FLOAT_SIZE + numVectors * FLOAT_SIZE + dim * FLOAT_SIZE + 1024 * 1024;
}
