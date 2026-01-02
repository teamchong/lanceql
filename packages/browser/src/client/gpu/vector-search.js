/**
 * GPUVectorSearch - GPU vector search for IVF partitions
 */

class GPUVectorSearch {
    constructor() { this.device = null; this.pipelines = new Map(); this.available = false; this._initPromise = null; }
    async init() { if (this._initPromise) return this._initPromise; this._initPromise = this._doInit(); return this._initPromise; }
    async _doInit() {
        if (typeof navigator === 'undefined' || !navigator.gpu) return false;
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return false;
            this.device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: 256 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024 } });
            await this._compileShaders();
            this.available = true;
            return true;
        } catch (e) { return false; }
    }
    async _compileShaders() {
        const distMod = this.device.createShaderModule({ code: VECTOR_DISTANCE_SHADER });
        this.pipelines.set('distance', this.device.createComputePipeline({ layout: 'auto', compute: { module: distMod, entryPoint: 'compute_distances' } }));
        const topkMod = this.device.createShaderModule({ code: TOPK_SHADER });
        this.pipelines.set('local_topk', this.device.createComputePipeline({ layout: 'auto', compute: { module: topkMod, entryPoint: 'local_topk' } }));
        this.pipelines.set('merge_topk', this.device.createComputePipeline({ layout: 'auto', compute: { module: topkMod, entryPoint: 'merge_topk' } }));
    }
    isAvailable() { return this.available; }

    async computeDistances(queryVec, vectors, numQueries = 1, metric = 0) {
        const numVectors = vectors.length;
        const dim = queryVec.length / numQueries;
        if (!this.available || numVectors < GPU_DISTANCE_THRESHOLD) return this._cpuDistances(queryVec, vectors, numQueries, metric);
        const flatVectors = new Float32Array(numVectors * dim);
        for (let i = 0; i < numVectors; i++) flatVectors.set(vectors[i], i * dim);
        const paramsBuffer = this._createUniform(new Uint32Array([dim, numVectors, numQueries, metric]));
        const queryBuffer = this._createStorage(queryVec);
        const vectorsBuffer = this._createStorage(flatVectors);
        const distanceBuffer = this.device.createBuffer({ size: numQueries * numVectors * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const readBuffer = this.device.createBuffer({ size: numQueries * numVectors * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const bindGroup = this.device.createBindGroup({ layout: this.pipelines.get('distance').getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: paramsBuffer } }, { binding: 1, resource: { buffer: queryBuffer } },
            { binding: 2, resource: { buffer: vectorsBuffer } }, { binding: 3, resource: { buffer: distanceBuffer } }
        ]});
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.get('distance'));
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(numVectors / 256), numQueries, 1);
        pass.end();
        encoder.copyBufferToBuffer(distanceBuffer, 0, readBuffer, 0, numQueries * numVectors * 4);
        this.device.queue.submit([encoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ);
        const distances = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();
        [paramsBuffer, queryBuffer, vectorsBuffer, distanceBuffer, readBuffer].forEach(b => b.destroy());
        return distances;
    }

    async topK(scores, indices = null, k = 10, descending = true) {
        const n = scores.length;
        if (!this.available || n < GPU_TOPK_THRESHOLD) return this._cpuTopK(scores, indices, k, descending);
        if (!indices) { indices = new Uint32Array(n); for (let i = 0; i < n; i++) indices[i] = i; }
        const numWorkgroups = Math.ceil(n / 512);
        const kPerWg = Math.min(k, 512);
        const numCandidates = numWorkgroups * kPerWg;
        // Phase 1
        const paramsBuffer = this._createUniform(new Uint32Array([n, k, descending ? 1 : 0, numWorkgroups]));
        const inputScoresBuffer = this._createStorage(scores);
        const inputIndicesBuffer = this._createStorage(indices);
        const intermediateScoresBuffer = this.device.createBuffer({ size: numCandidates * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const intermediateIndicesBuffer = this.device.createBuffer({ size: numCandidates * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const localBG = this.device.createBindGroup({ layout: this.pipelines.get('local_topk').getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: paramsBuffer } }, { binding: 1, resource: { buffer: inputScoresBuffer } },
            { binding: 2, resource: { buffer: inputIndicesBuffer } }, { binding: 3, resource: { buffer: intermediateScoresBuffer } },
            { binding: 4, resource: { buffer: intermediateIndicesBuffer } }
        ]});
        let encoder = this.device.createCommandEncoder();
        let pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.get('local_topk'));
        pass.setBindGroup(0, localBG);
        pass.dispatchWorkgroups(numWorkgroups, 1, 1);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        // Phase 2
        const mergeParamsBuffer = this._createUniform(new Uint32Array([numCandidates, k, descending ? 1 : 0, 0]));
        const finalScoresBuffer = this.device.createBuffer({ size: k * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const finalIndicesBuffer = this.device.createBuffer({ size: k * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const readScoresBuffer = this.device.createBuffer({ size: k * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const readIndicesBuffer = this.device.createBuffer({ size: k * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const mergeBG = this.device.createBindGroup({ layout: this.pipelines.get('merge_topk').getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: mergeParamsBuffer } }, { binding: 1, resource: { buffer: intermediateScoresBuffer } },
            { binding: 2, resource: { buffer: intermediateIndicesBuffer } }, { binding: 3, resource: { buffer: finalScoresBuffer } },
            { binding: 4, resource: { buffer: finalIndicesBuffer } }
        ]});
        encoder = this.device.createCommandEncoder();
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.get('merge_topk'));
        pass.setBindGroup(0, mergeBG);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();
        encoder.copyBufferToBuffer(finalScoresBuffer, 0, readScoresBuffer, 0, k * 4);
        encoder.copyBufferToBuffer(finalIndicesBuffer, 0, readIndicesBuffer, 0, k * 4);
        this.device.queue.submit([encoder.finish()]);
        await Promise.all([readScoresBuffer.mapAsync(GPUMapMode.READ), readIndicesBuffer.mapAsync(GPUMapMode.READ)]);
        const resultScores = new Float32Array(readScoresBuffer.getMappedRange().slice(0));
        const resultIndices = new Uint32Array(readIndicesBuffer.getMappedRange().slice(0));
        readScoresBuffer.unmap(); readIndicesBuffer.unmap();
        [paramsBuffer, inputScoresBuffer, inputIndicesBuffer, intermediateScoresBuffer, intermediateIndicesBuffer,
         mergeParamsBuffer, finalScoresBuffer, finalIndicesBuffer, readScoresBuffer, readIndicesBuffer].forEach(b => b.destroy());
        return { indices: resultIndices, scores: resultScores };
    }

    async search(queryVec, vectors, k = 10, options = {}) {
        const { metric = 0 } = options;
        const scores = await this.computeDistances(queryVec, vectors, 1, metric);
        const descending = metric === 0 || metric === 2;
        return await this.topK(scores, null, k, descending);
    }

    _cpuDistances(queryVec, vectors, numQueries, metric) {
        const dim = queryVec.length / numQueries;
        const numVectors = vectors.length;
        const distances = new Float32Array(numQueries * numVectors);
        for (let q = 0; q < numQueries; q++) {
            const qOff = q * dim;
            for (let v = 0; v < numVectors; v++) {
                const vec = vectors[v];
                let result = 0;
                if (metric === 1) { for (let i = 0; i < dim; i++) { const d = queryVec[qOff + i] - vec[i]; result += d * d; } result = Math.sqrt(result); }
                else { for (let i = 0; i < dim; i++) result += queryVec[qOff + i] * vec[i]; }
                distances[q * numVectors + v] = result;
            }
        }
        return distances;
    }

    _cpuTopK(scores, indices, k, descending) {
        const n = scores.length;
        if (!indices) { indices = new Uint32Array(n); for (let i = 0; i < n; i++) indices[i] = i; }
        const indexed = Array.from(scores).map((score, i) => ({ score, idx: indices[i] }));
        if (descending) indexed.sort((a, b) => b.score - a.score);
        else indexed.sort((a, b) => a.score - b.score);
        const topK = indexed.slice(0, k);
        return { indices: new Uint32Array(topK.map(x => x.idx)), scores: new Float32Array(topK.map(x => x.score)) };
    }

    _createStorage(data) { const buf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); this.device.queue.writeBuffer(buf, 0, data); return buf; }
    _createUniform(data) { const buf = this.device.createBuffer({ size: Math.max(data.byteLength, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.device.queue.writeBuffer(buf, 0, data); return buf; }
}

// Lazy singleton - only instantiated when first accessed
let _gpuVectorSearch = null;
function getGPUVectorSearch() {
    if (!_gpuVectorSearch) _gpuVectorSearch = new GPUVectorSearch();
    return _gpuVectorSearch;
}

export { GPUVectorSearch, getGPUVectorSearch };
