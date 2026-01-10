/**
 * GPU-Accelerated Vector Search
 *
 * Provides GPU-based distance computation and top-K selection for vector search.
 * Falls back to CPU for small datasets where GPU overhead exceeds benefit.
 */

// Thresholds for GPU acceleration
const GPU_DISTANCE_THRESHOLD = 5000;    // Min vectors for GPU distance
const GPU_TOPK_THRESHOLD = 10000;       // Min scores for GPU top-K

// Distance metrics enum
export const DistanceMetric = {
    COSINE: 0,
    L2: 1,
    DOT_PRODUCT: 2,
};

// Vector distance shader - embedded for bundler compatibility
const VECTOR_DISTANCE_SHADER = `
struct DistanceParams {
    dim: u32,
    num_vectors: u32,
    num_queries: u32,
    metric: u32,
}

@group(0) @binding(0) var<uniform> params: DistanceParams;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;

var<workgroup> shared_query: array<f32, 512>;

fn compute_dot(vec_offset: u32, dim: u32) -> f32 {
    var sum: f32 = 0.0;
    let dim4 = dim / 4u;
    for (var i = 0u; i < dim4; i++) {
        let base = i * 4u;
        sum += shared_query[base] * vectors[vec_offset + base];
        sum += shared_query[base + 1u] * vectors[vec_offset + base + 1u];
        sum += shared_query[base + 2u] * vectors[vec_offset + base + 2u];
        sum += shared_query[base + 3u] * vectors[vec_offset + base + 3u];
    }
    for (var i = dim4 * 4u; i < dim; i++) {
        sum += shared_query[i] * vectors[vec_offset + i];
    }
    return sum;
}

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

@compute @workgroup_size(256)
fn compute_distances(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(workgroup_id) wid: vec3<u32>) {
    let query_idx = wid.y;
    let vec_idx = gid.x;

    if (vec_idx >= params.num_vectors || query_idx >= params.num_queries) {
        return;
    }

    let dim = params.dim;
    let tid = lid.x;

    for (var i = tid; i < dim && i < 512u; i += 256u) {
        shared_query[i] = queries[query_idx * dim + i];
    }
    workgroupBarrier();

    let vec_offset = vec_idx * dim;
    var result: f32 = 0.0;

    switch params.metric {
        case 0u, 2u: {
            result = compute_dot(vec_offset, dim);
        }
        case 1u: {
            result = compute_l2(vec_offset, dim);
        }
        default: {
            result = compute_dot(vec_offset, dim);
        }
    }

    distances[query_idx * params.num_vectors + vec_idx] = result;
}
`;

// Top-K selection shader
const TOPK_SHADER = `
struct TopKParams {
    size: u32,
    k: u32,
    descending: u32,
    num_workgroups: u32,
}

@group(0) @binding(0) var<uniform> params: TopKParams;
@group(0) @binding(1) var<storage, read> input_scores: array<f32>;
@group(0) @binding(2) var<storage, read> input_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_indices: array<u32>;

var<workgroup> local_scores: array<f32, 512>;
var<workgroup> local_indices: array<u32, 512>;

fn should_swap(a: f32, b: f32, descending: bool) -> bool {
    if (descending) {
        return a < b;
    } else {
        return a > b;
    }
}

fn get_sentinel(descending: bool) -> f32 {
    if (descending) {
        return -3.4028235e+38;
    } else {
        return 3.4028235e+38;
    }
}

@compute @workgroup_size(256)
fn local_topk(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let chunk_size = 512u;
    let base = wid.x * chunk_size;
    let tid = lid.x;
    let descending = params.descending == 1u;
    let sentinel = get_sentinel(descending);

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

    for (var k = 2u; k <= chunk_size; k = k << 1u) {
        for (var j = k >> 1u; j > 0u; j = j >> 1u) {
            for (var t = 0u; t < 2u; t++) {
                let i = tid + t * 256u;
                let ixj = i ^ j;

                if (ixj > i && ixj < chunk_size) {
                    let direction = ((i & k) == 0u) == descending;
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

struct MergeParams {
    num_candidates: u32,
    k: u32,
    descending: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> merge_params: MergeParams;
@group(0) @binding(1) var<storage, read> merge_scores: array<f32>;
@group(0) @binding(2) var<storage, read> merge_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> final_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> final_indices: array<u32>;

@compute @workgroup_size(256)
fn merge_topk(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = merge_params.num_candidates;
    let k = merge_params.k;
    let descending = merge_params.descending == 1u;
    let sentinel = get_sentinel(descending);

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

    if (tid < k) {
        final_scores[tid] = local_scores[tid];
        final_indices[tid] = local_indices[tid];
    }
}

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
`;

/**
 * GPU-accelerated vector search class.
 */
export class GPUVectorSearch {
    constructor() {
        this.device = null;
        this.pipelines = new Map();
        this.available = false;
        this._initPromise = null;
    }

    async init() {
        if (this._initPromise) return this._initPromise;
        this._initPromise = this._doInit();
        return this._initPromise;
    }

    async _doInit() {
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            console.log('[GPUVectorSearch] WebGPU not available');
            return false;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return false;

            this.device = await adapter.requestDevice({
                requiredLimits: {
                    maxStorageBufferBindingSize: 256 * 1024 * 1024,
                    maxBufferSize: 256 * 1024 * 1024,
                },
            });

            await this._compileShaders();
            this.available = true;
            console.log('[GPUVectorSearch] Initialized');
            return true;
        } catch (e) {
            console.error('[GPUVectorSearch] Init failed:', e);
            return false;
        }
    }

    async _compileShaders() {
        // Distance computation shader
        const distanceModule = this.device.createShaderModule({
            code: VECTOR_DISTANCE_SHADER
        });
        this.pipelines.set('distance', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: distanceModule, entryPoint: 'compute_distances' },
        }));

        // Top-K shaders
        const topkModule = this.device.createShaderModule({
            code: TOPK_SHADER
        });
        this.pipelines.set('local_topk', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: topkModule, entryPoint: 'local_topk' },
        }));
        this.pipelines.set('merge_topk', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: topkModule, entryPoint: 'merge_topk' },
        }));
        this.pipelines.set('init_indices', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: topkModule, entryPoint: 'init_indices' },
        }));
    }

    isAvailable() {
        return this.available;
    }

    /**
     * Compute distances between query vector(s) and database vectors.
     */
    async computeDistances(queryVec, vectors, numQueries = 1, metric = DistanceMetric.COSINE) {
        const numVectors = vectors.length;
        const dim = queryVec.length / numQueries;

        // CPU fallback
        if (!this.available || numVectors < GPU_DISTANCE_THRESHOLD) {
            return this._cpuComputeDistances(queryVec, vectors, numQueries, metric);
        }

        // Flatten vectors
        const flatVectors = new Float32Array(numVectors * dim);
        for (let i = 0; i < numVectors; i++) {
            flatVectors.set(vectors[i], i * dim);
        }

        // Create buffers
        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([dim, numVectors, numQueries, metric]));

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

        const distanceBuffer = this.device.createBuffer({
            size: numQueries * numVectors * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const readBuffer = this.device.createBuffer({
            size: numQueries * numVectors * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.pipelines.get('distance').getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: queryBuffer } },
                { binding: 2, resource: { buffer: vectorsBuffer } },
                { binding: 3, resource: { buffer: distanceBuffer } },
            ],
        });

        // Dispatch
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.get('distance'));
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(numVectors / 256), numQueries, 1);
        pass.end();

        encoder.copyBufferToBuffer(distanceBuffer, 0, readBuffer, 0, numQueries * numVectors * 4);
        this.device.queue.submit([encoder.finish()]);

        // Read results
        await readBuffer.mapAsync(GPUMapMode.READ);
        const distances = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();

        // Cleanup
        paramsBuffer.destroy();
        queryBuffer.destroy();
        vectorsBuffer.destroy();
        distanceBuffer.destroy();
        readBuffer.destroy();

        return distances;
    }

    /**
     * Find top-K from scores array.
     */
    async topK(scores, indices = null, k = 10, descending = true) {
        const n = scores.length;

        // CPU fallback
        if (!this.available || n < GPU_TOPK_THRESHOLD) {
            return this._cpuTopK(scores, indices, k, descending);
        }

        // Initialize indices if not provided
        if (!indices) {
            indices = new Uint32Array(n);
            for (let i = 0; i < n; i++) indices[i] = i;
        }

        const numWorkgroups = Math.ceil(n / 512);
        const kPerWg = Math.min(k, 512);
        const numCandidates = numWorkgroups * kPerWg;

        // Phase 1: Local top-K
        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([n, k, descending ? 1 : 0, numWorkgroups]));

        const inputScoresBuffer = this.device.createBuffer({
            size: scores.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(inputScoresBuffer, 0, scores);

        const inputIndicesBuffer = this.device.createBuffer({
            size: indices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(inputIndicesBuffer, 0, indices);

        const intermediateScoresBuffer = this.device.createBuffer({
            size: numCandidates * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const intermediateIndicesBuffer = this.device.createBuffer({
            size: numCandidates * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const localBindGroup = this.device.createBindGroup({
            layout: this.pipelines.get('local_topk').getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: inputScoresBuffer } },
                { binding: 2, resource: { buffer: inputIndicesBuffer } },
                { binding: 3, resource: { buffer: intermediateScoresBuffer } },
                { binding: 4, resource: { buffer: intermediateIndicesBuffer } },
            ],
        });

        let encoder = this.device.createCommandEncoder();
        let pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.get('local_topk'));
        pass.setBindGroup(0, localBindGroup);
        pass.dispatchWorkgroups(numWorkgroups, 1, 1);
        pass.end();
        this.device.queue.submit([encoder.finish()]);

        // Phase 2: Merge results
        const mergeParamsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(mergeParamsBuffer, 0, new Uint32Array([numCandidates, k, descending ? 1 : 0, 0]));

        const finalScoresBuffer = this.device.createBuffer({
            size: k * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const finalIndicesBuffer = this.device.createBuffer({
            size: k * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const readScoresBuffer = this.device.createBuffer({
            size: k * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const readIndicesBuffer = this.device.createBuffer({
            size: k * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const mergeBindGroup = this.device.createBindGroup({
            layout: this.pipelines.get('merge_topk').getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: mergeParamsBuffer } },
                { binding: 1, resource: { buffer: intermediateScoresBuffer } },
                { binding: 2, resource: { buffer: intermediateIndicesBuffer } },
                { binding: 3, resource: { buffer: finalScoresBuffer } },
                { binding: 4, resource: { buffer: finalIndicesBuffer } },
            ],
        });

        encoder = this.device.createCommandEncoder();
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.get('merge_topk'));
        pass.setBindGroup(0, mergeBindGroup);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();

        encoder.copyBufferToBuffer(finalScoresBuffer, 0, readScoresBuffer, 0, k * 4);
        encoder.copyBufferToBuffer(finalIndicesBuffer, 0, readIndicesBuffer, 0, k * 4);
        this.device.queue.submit([encoder.finish()]);

        // Read results
        await Promise.all([
            readScoresBuffer.mapAsync(GPUMapMode.READ),
            readIndicesBuffer.mapAsync(GPUMapMode.READ),
        ]);

        const resultScores = new Float32Array(readScoresBuffer.getMappedRange().slice(0));
        const resultIndices = new Uint32Array(readIndicesBuffer.getMappedRange().slice(0));
        readScoresBuffer.unmap();
        readIndicesBuffer.unmap();

        // Cleanup
        paramsBuffer.destroy();
        inputScoresBuffer.destroy();
        inputIndicesBuffer.destroy();
        intermediateScoresBuffer.destroy();
        intermediateIndicesBuffer.destroy();
        mergeParamsBuffer.destroy();
        finalScoresBuffer.destroy();
        finalIndicesBuffer.destroy();
        readScoresBuffer.destroy();
        readIndicesBuffer.destroy();

        return { indices: resultIndices, scores: resultScores };
    }

    /**
     * Full vector search: distance + top-K.
     */
    async search(queryVec, vectors, k = 10, options = {}) {
        const { metric = DistanceMetric.COSINE } = options;

        const scores = await this.computeDistances(queryVec, vectors, 1, metric);
        const descending = metric === DistanceMetric.COSINE || metric === DistanceMetric.DOT_PRODUCT;

        return await this.topK(scores, null, k, descending);
    }

    /**
     * Batch search: multiple query vectors at once.
     */
    async batchSearch(queryVecs, vectors, k = 10, options = {}) {
        const { metric = DistanceMetric.COSINE } = options;
        const numQueries = queryVecs.length;
        const dim = queryVecs[0].length;

        // Flatten queries
        const flatQueries = new Float32Array(numQueries * dim);
        for (let i = 0; i < numQueries; i++) {
            flatQueries.set(queryVecs[i], i * dim);
        }

        const allScores = await this.computeDistances(flatQueries, vectors, numQueries, metric);
        const descending = metric === DistanceMetric.COSINE || metric === DistanceMetric.DOT_PRODUCT;

        // Extract top-K for each query
        const results = [];
        const numVectors = vectors.length;
        for (let q = 0; q < numQueries; q++) {
            const queryScores = new Float32Array(allScores.buffer, q * numVectors * 4, numVectors);
            const result = await this.topK(new Float32Array(queryScores), null, k, descending);
            results.push(result);
        }

        return results;
    }

    // CPU fallbacks
    _cpuComputeDistances(queryVec, vectors, numQueries, metric) {
        const dim = queryVec.length / numQueries;
        const numVectors = vectors.length;
        const distances = new Float32Array(numQueries * numVectors);

        for (let q = 0; q < numQueries; q++) {
            const queryOffset = q * dim;
            for (let v = 0; v < numVectors; v++) {
                const vec = vectors[v];
                let result = 0;

                if (metric === DistanceMetric.L2) {
                    for (let i = 0; i < dim; i++) {
                        const d = queryVec[queryOffset + i] - vec[i];
                        result += d * d;
                    }
                    result = Math.sqrt(result);
                } else {
                    for (let i = 0; i < dim; i++) {
                        result += queryVec[queryOffset + i] * vec[i];
                    }
                }

                distances[q * numVectors + v] = result;
            }
        }

        return distances;
    }

    _cpuTopK(scores, indices, k, descending) {
        const n = scores.length;
        if (!indices) {
            indices = new Uint32Array(n);
            for (let i = 0; i < n; i++) indices[i] = i;
        }

        const indexed = Array.from(scores).map((score, i) => ({ score, idx: indices[i] }));
        if (descending) {
            indexed.sort((a, b) => b.score - a.score);
        } else {
            indexed.sort((a, b) => a.score - b.score);
        }

        const topK = indexed.slice(0, k);
        return {
            indices: new Uint32Array(topK.map(x => x.idx)),
            scores: new Float32Array(topK.map(x => x.score)),
        };
    }

    dispose() {
        this.pipelines.clear();
        this.device = null;
        this.available = false;
    }
}

// Singleton
let gpuVectorSearchInstance = null;

export function getGPUVectorSearch() {
    if (!gpuVectorSearchInstance) {
        gpuVectorSearchInstance = new GPUVectorSearch();
    }
    return gpuVectorSearchInstance;
}

export function shouldUseGPUVectorSearch(numVectors) {
    return numVectors >= GPU_DISTANCE_THRESHOLD;
}
