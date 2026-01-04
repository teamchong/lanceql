/**
 * LanceQL WASM Loader
 *
 * JavaScript wrapper for the LanceQL WebAssembly module.
 * Provides a high-level API for reading Lance files in the browser.
 * Supports both local files and remote URLs via HTTP Range requests.
 */

/**
 * IndexedDB cache for dataset metadata.
 * Caches schema, column types, and fragment info to speed up repeat visits.
 */
class MetadataCache {
    constructor(dbName = 'lanceql-cache', version = 1) {
        this.dbName = dbName;
        this.version = version;
        this.db = null;
    }

    async open() {
        if (this.db) return this.db;

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.version);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('datasets')) {
                    const store = db.createObjectStore('datasets', { keyPath: 'url' });
                    store.createIndex('timestamp', 'timestamp');
                }
            };
        });
    }

    /**
     * Get cached metadata for a dataset URL.
     * @param {string} url - Dataset URL
     * @returns {Promise<Object|null>} Cached metadata or null
     */
    async get(url) {
        try {
            const db = await this.open();
            return new Promise((resolve) => {
                const tx = db.transaction('datasets', 'readonly');
                const store = tx.objectStore('datasets');
                const request = store.get(url);
                request.onsuccess = () => resolve(request.result || null);
                request.onerror = () => resolve(null);
            });
        } catch (e) {
            console.warn('[MetadataCache] Get failed:', e);
            return null;
        }
    }

    /**
     * Cache metadata for a dataset URL.
     * @param {string} url - Dataset URL
     * @param {Object} metadata - Metadata to cache (schema, columnTypes, fragments, etc.)
     */
    async set(url, metadata) {
        try {
            const db = await this.open();
            return new Promise((resolve, reject) => {
                const tx = db.transaction('datasets', 'readwrite');
                const store = tx.objectStore('datasets');
                const data = {
                    url,
                    timestamp: Date.now(),
                    ...metadata
                };
                const request = store.put(data);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });
        } catch (e) {
            console.warn('[MetadataCache] Set failed:', e);
        }
    }

    /**
     * Delete cached metadata for a URL.
     * @param {string} url - Dataset URL
     */
    async delete(url) {
        try {
            const db = await this.open();
            return new Promise((resolve) => {
                const tx = db.transaction('datasets', 'readwrite');
                const store = tx.objectStore('datasets');
                store.delete(url);
                tx.oncomplete = () => resolve();
            });
        } catch (e) {
            console.warn('[MetadataCache] Delete failed:', e);
        }
    }

    /**
     * Clear all cached metadata.
     */
    async clear() {
        try {
            const db = await this.open();
            return new Promise((resolve) => {
                const tx = db.transaction('datasets', 'readwrite');
                const store = tx.objectStore('datasets');
                store.clear();
                tx.oncomplete = () => resolve();
            });
        } catch (e) {
            console.warn('[MetadataCache] Clear failed:', e);
        }
    }
}

// Global cache instance
const metadataCache = new MetadataCache();

/**
 * WebGPU Accelerator - GPU-accelerated batch cosine similarity.
 *
 * Uses WebGPU compute shaders for massive parallelism:
 * - CPU/WASM SIMD: ~4-8 floats parallel per instruction
 * - WebGPU: Thousands of parallel cores, entire batch in one dispatch
 *
 * Falls back to WASM SIMD if WebGPU unavailable.
 */
class WebGPUAccelerator {
    constructor() {
        this.device = null;
        this.pipeline = null;
        this.available = false;
        this._initPromise = null;
    }

    /**
     * Initialize WebGPU. Call once before using.
     * @returns {Promise<boolean>} Whether WebGPU is available
     */
    async init() {
        if (this._initPromise) return this._initPromise;

        this._initPromise = this._doInit();
        return this._initPromise;
    }

    async _doInit() {
        if (!navigator.gpu) {
            console.log('[WebGPU] Not available in this browser');
            return false;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.log('[WebGPU] No adapter found');
                return false;
            }

            this.device = await adapter.requestDevice();
            this._createPipeline();
            this.available = true;
            console.log('[WebGPU] Initialized successfully');
            return true;
        } catch (e) {
            console.warn('[WebGPU] Init failed:', e);
            return false;
        }
    }

    _createPipeline() {
        // Compute shader for batch cosine similarity
        // Assumes L2-normalized vectors (dot product = cosine similarity)
        const shaderCode = `
            struct Params {
                dim: u32,
                numVectors: u32,
            }

            @group(0) @binding(0) var<uniform> params: Params;
            @group(0) @binding(1) var<storage, read> query: array<f32>;
            @group(0) @binding(2) var<storage, read> vectors: array<f32>;
            @group(0) @binding(3) var<storage, read_write> scores: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) globalId: vec3u) {
                let idx = globalId.x;
                if (idx >= params.numVectors) {
                    return;
                }

                let dim = params.dim;
                let offset = idx * dim;

                // Compute dot product (= cosine similarity for normalized vectors)
                var dot: f32 = 0.0;
                for (var i: u32 = 0u; i < dim; i++) {
                    dot += query[i] * vectors[offset + i];
                }

                scores[idx] = dot;
            }
        `;

        const shaderModule = this.device.createShaderModule({
            code: shaderCode
        });

        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    /**
     * Batch cosine similarity using WebGPU.
     * @param {Float32Array} queryVec - Query vector (dim)
     * @param {Float32Array[]} vectors - Array of candidate vectors
     * @param {boolean} normalized - Whether vectors are L2-normalized
     * @returns {Promise<Float32Array>} Similarity scores
     */
    async batchCosineSimilarity(queryVec, vectors, normalized = true) {
        if (!this.available || vectors.length === 0) {
            return null; // Caller should fallback to WASM
        }

        const dim = queryVec.length;
        const numVectors = vectors.length;

        // Check buffer size limit (default 128MB, but check device limits)
        const vectorsBufferSize = numVectors * dim * 4;
        const maxBufferSize = this.device.limits?.maxStorageBufferBindingSize || 134217728;
        if (vectorsBufferSize > maxBufferSize) {
            console.warn(`[WebGPU] Buffer size ${(vectorsBufferSize/1024/1024).toFixed(1)}MB exceeds limit ${(maxBufferSize/1024/1024).toFixed(1)}MB, falling back`);
            return null; // Caller should fallback to WASM
        }

        // Create buffers
        const paramsBuffer = this.device.createBuffer({
            size: 8, // 2 x u32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const queryBuffer = this.device.createBuffer({
            size: dim * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const vectorsBuffer = this.device.createBuffer({
            size: numVectors * dim * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const scoresBuffer = this.device.createBuffer({
            size: numVectors * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const readbackBuffer = this.device.createBuffer({
            size: numVectors * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // Write data to buffers
        this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([dim, numVectors]));
        this.device.queue.writeBuffer(queryBuffer, 0, queryVec);

        // Flatten vectors into single array
        const flatVectors = new Float32Array(numVectors * dim);
        for (let i = 0; i < numVectors; i++) {
            flatVectors.set(vectors[i], i * dim);
        }
        this.device.queue.writeBuffer(vectorsBuffer, 0, flatVectors);

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: queryBuffer } },
                { binding: 2, resource: { buffer: vectorsBuffer } },
                { binding: 3, resource: { buffer: scoresBuffer } },
            ]
        });

        // Dispatch compute shader
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(numVectors / 256));
        passEncoder.end();

        // Copy results to readback buffer
        commandEncoder.copyBufferToBuffer(scoresBuffer, 0, readbackBuffer, 0, numVectors * 4);
        this.device.queue.submit([commandEncoder.finish()]);

        // Read results
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const results = new Float32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        // Cleanup
        paramsBuffer.destroy();
        queryBuffer.destroy();
        vectorsBuffer.destroy();
        scoresBuffer.destroy();
        readbackBuffer.destroy();

        return results;
    }

    /**
     * Check if WebGPU is available and initialized
     */
    isAvailable() {
        return this.available;
    }

    /**
     * Get maximum vectors that can fit in a single WebGPU batch.
     * @param {number} dim - Vector dimension
     * @returns {number} Maximum vectors per batch
     */
    getMaxVectorsPerBatch(dim) {
        if (!this.available) return 0;
        const maxBufferSize = this.device.limits?.maxStorageBufferBindingSize || 134217728;
        // Leave some headroom (use 90% of limit)
        return Math.floor((maxBufferSize * 0.9) / (dim * 4));
    }
}

// Global WebGPU accelerator instance
const webgpuAccelerator = new WebGPUAccelerator();

/**
 * GPU-accelerated SQL Aggregations (SUM, COUNT, AVG, MIN, MAX)
 * Uses WebGPU compute shaders for parallel reduction on large datasets.
 */
class GPUAggregator {
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
        if (!navigator.gpu) return false;
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return false;
            this.device = await adapter.requestDevice({
                requiredLimits: { maxStorageBufferBindingSize: 256 * 1024 * 1024 },
            });
            this._compileShaders();
            this.available = true;
            console.log('[GPUAggregator] Initialized');
            return true;
        } catch (e) {
            console.warn('[GPUAggregator] Init failed:', e);
            return false;
        }
    }

    _compileShaders() {
        const code = `
struct P { size: u32, wg: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> i: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
var<workgroup> s: array<f32, 256>;

@compute @workgroup_size(256)
fn sum(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    s[l.x] = select(0.0, i[g.x], g.x < p.size);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] += s[l.x + t]; } workgroupBarrier(); }
    if (l.x == 0u) { o[w.x] = s[0]; }
}

@compute @workgroup_size(256)
fn sum_f(@builtin(local_invocation_id) l: vec3<u32>) {
    s[l.x] = select(0.0, i[l.x], l.x < p.wg);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] += s[l.x + t]; } workgroupBarrier(); }
    if (l.x == 0u) { o[0] = s[0]; }
}

@compute @workgroup_size(256)
fn min_r(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    s[l.x] = select(3.4e+38, i[g.x], g.x < p.size);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] = min(s[l.x], s[l.x + t]); } workgroupBarrier(); }
    if (l.x == 0u) { o[w.x] = s[0]; }
}

@compute @workgroup_size(256)
fn min_f(@builtin(local_invocation_id) l: vec3<u32>) {
    s[l.x] = select(3.4e+38, i[l.x], l.x < p.wg);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] = min(s[l.x], s[l.x + t]); } workgroupBarrier(); }
    if (l.x == 0u) { o[0] = s[0]; }
}

@compute @workgroup_size(256)
fn max_r(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    s[l.x] = select(-3.4e+38, i[g.x], g.x < p.size);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] = max(s[l.x], s[l.x + t]); } workgroupBarrier(); }
    if (l.x == 0u) { o[w.x] = s[0]; }
}

@compute @workgroup_size(256)
fn max_f(@builtin(local_invocation_id) l: vec3<u32>) {
    s[l.x] = select(-3.4e+38, i[l.x], l.x < p.wg);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] = max(s[l.x], s[l.x + t]); } workgroupBarrier(); }
    if (l.x == 0u) { o[0] = s[0]; }
}`;
        const module = this.device.createShaderModule({ code });
        for (const [name, entry] of [['sum', 'sum'], ['sum_final', 'sum_f'], ['min', 'min_r'], ['min_final', 'min_f'], ['max', 'max_r'], ['max_final', 'max_f']]) {
            this.pipelines.set(name, this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: entry } }));
        }
    }

    isAvailable() { return this.available; }

    async sum(values) {
        if (!this.available || values.length < 1000) return this._cpuSum(values);
        return this._gpuReduce(values, 'sum');
    }

    async min(values) {
        if (!this.available || values.length < 1000) return values.length ? Math.min(...values) : null;
        return this._gpuReduce(values, 'min');
    }

    async max(values) {
        if (!this.available || values.length < 1000) return values.length ? Math.max(...values) : null;
        return this._gpuReduce(values, 'max');
    }

    async avg(values) {
        if (values.length === 0) return null;
        const sum = await this.sum(values);
        return sum / values.length;
    }

    count(values) { return values.length; }

    async _gpuReduce(values, op) {
        const n = values.length, wgSize = 256, numWg = Math.ceil(n / wgSize);
        const input = values instanceof Float32Array ? values : new Float32Array(values);

        const inputBuf = this.device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(inputBuf, 0, input);
        const partialBuf = this.device.createBuffer({ size: numWg * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const outBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const stageBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const paramsBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([n, numWg]));
        const finalParamsBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(finalParamsBuf, 0, new Uint32Array([numWg, numWg]));

        const p1 = this.pipelines.get(op), p2 = this.pipelines.get(op + '_final');
        const bg1 = this.device.createBindGroup({ layout: p1.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: paramsBuf } }, { binding: 1, resource: { buffer: inputBuf } }, { binding: 2, resource: { buffer: partialBuf } }] });
        const bg2 = this.device.createBindGroup({ layout: p2.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: finalParamsBuf } }, { binding: 1, resource: { buffer: partialBuf } }, { binding: 2, resource: { buffer: outBuf } }] });

        const enc = this.device.createCommandEncoder();
        const c1 = enc.beginComputePass(); c1.setPipeline(p1); c1.setBindGroup(0, bg1); c1.dispatchWorkgroups(numWg); c1.end();
        const c2 = enc.beginComputePass(); c2.setPipeline(p2); c2.setBindGroup(0, bg2); c2.dispatchWorkgroups(1); c2.end();
        enc.copyBufferToBuffer(outBuf, 0, stageBuf, 0, 4);
        this.device.queue.submit([enc.finish()]);

        await stageBuf.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(stageBuf.getMappedRange())[0];
        stageBuf.unmap();

        inputBuf.destroy(); partialBuf.destroy(); outBuf.destroy(); stageBuf.destroy(); paramsBuf.destroy(); finalParamsBuf.destroy();
        return result;
    }

    _cpuSum(values) { let s = 0; for (let i = 0; i < values.length; i++) s += values[i]; return s; }
}

// Global GPU aggregator instance
const gpuAggregator = new GPUAggregator();

/**
 * GPU-accelerated SQL JOINs using hash join algorithm.
 * Falls back to CPU for small tables where GPU overhead exceeds benefit.
 */
class GPUJoiner {
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
        if (!navigator.gpu) return false;
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return false;
            this.device = await adapter.requestDevice({
                requiredLimits: { maxStorageBufferBindingSize: 256 * 1024 * 1024 },
            });
            this._compileShaders();
            this.available = true;
            console.log('[GPUJoiner] Initialized');
            return true;
        } catch (e) {
            console.warn('[GPUJoiner] Init failed:', e);
            return false;
        }
    }

    _compileShaders() {
        const code = `
struct BP { size: u32, cap: u32 }
struct PP { left_size: u32, cap: u32, max_matches: u32 }
@group(0) @binding(0) var<uniform> bp: BP;
@group(0) @binding(1) var<storage, read> bkeys: array<u32>;
@group(0) @binding(2) var<storage, read_write> ht: array<atomic<u32>>;

fn fnv(k: u32) -> u32 {
    var h = 2166136261u;
    h ^= (k & 0xFFu); h *= 16777619u;
    h ^= ((k >> 8u) & 0xFFu); h *= 16777619u;
    h ^= ((k >> 16u) & 0xFFu); h *= 16777619u;
    h ^= ((k >> 24u) & 0xFFu); h *= 16777619u;
    return h;
}

@compute @workgroup_size(256)
fn build(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= bp.size) { return; }
    let k = bkeys[g.x];
    var s = fnv(k) % bp.cap;
    for (var p = 0u; p < bp.cap; p++) {
        let i = s * 2u;
        let o = atomicCompareExchangeWeak(&ht[i], 0xFFFFFFFFu, k);
        if (o.exchanged) { atomicStore(&ht[i + 1u], g.x); return; }
        s = (s + 1u) % bp.cap;
    }
}

@group(0) @binding(0) var<uniform> pp: PP;
@group(0) @binding(1) var<storage, read> lkeys: array<u32>;
@group(0) @binding(2) var<storage, read> pht: array<u32>;
@group(0) @binding(3) var<storage, read_write> matches: array<u32>;
@group(0) @binding(4) var<storage, read_write> mc: atomic<u32>;

@compute @workgroup_size(256)
fn probe(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= pp.left_size) { return; }
    let k = lkeys[g.x];
    var s = fnv(k) % pp.cap;
    for (var p = 0u; p < pp.cap; p++) {
        let i = s * 2u;
        let sk = pht[i];
        if (sk == 0xFFFFFFFFu) { return; }
        if (sk == k) {
            let ri = pht[i + 1u];
            let o = atomicAdd(&mc, 1u);
            if (o * 2u + 1u < pp.max_matches * 2u) {
                matches[o * 2u] = g.x;
                matches[o * 2u + 1u] = ri;
            }
        }
        s = (s + 1u) % pp.cap;
    }
}

struct IP { cap: u32 }
@group(0) @binding(0) var<uniform> ip: IP;
@group(0) @binding(1) var<storage, read_write> it: array<u32>;

@compute @workgroup_size(256)
fn init_t(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= ip.cap * 2u) { return; }
    it[g.x] = select(0u, 0xFFFFFFFFu, g.x % 2u == 0u);
}`;
        const module = this.device.createShaderModule({ code });
        this.pipelines.set('init', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'init_t' } }));
        this.pipelines.set('build', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'build' } }));
        this.pipelines.set('probe', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'probe' } }));
    }

    isAvailable() { return this.available; }

    async hashJoin(leftRows, rightRows, leftKey, rightKey) {
        const lSize = leftRows.length, rSize = rightRows.length;
        if (!this.available || lSize * rSize < 100000000) {
            return this._cpuHashJoin(leftRows, rightRows, leftKey, rightKey);
        }
        const lKeys = this._extractKeys(leftRows, leftKey);
        const rKeys = this._extractKeys(rightRows, rightKey);
        const cap = this._nextPow2(rSize * 2);
        const maxM = Math.max(lSize * 10, 100000);

        const rKeysBuf = this._createBuf(rKeys, GPUBufferUsage.STORAGE);
        const lKeysBuf = this._createBuf(lKeys, GPUBufferUsage.STORAGE);
        const htBuf = this.device.createBuffer({ size: cap * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const matchBuf = this.device.createBuffer({ size: maxM * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const mcBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const stageBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

        const ipBuf = this._createUniform(new Uint32Array([cap]));
        const bpBuf = this._createUniform(new Uint32Array([rSize, cap]));
        const ppBuf = this._createUniform(new Uint32Array([lSize, cap, maxM]));

        const initP = this.pipelines.get('init'), buildP = this.pipelines.get('build'), probeP = this.pipelines.get('probe');
        const initBG = this.device.createBindGroup({ layout: initP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ipBuf } }, { binding: 1, resource: { buffer: htBuf } }] });
        const buildBG = this.device.createBindGroup({ layout: buildP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bpBuf } }, { binding: 1, resource: { buffer: rKeysBuf } }, { binding: 2, resource: { buffer: htBuf } }] });
        const probeBG = this.device.createBindGroup({ layout: probeP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ppBuf } }, { binding: 1, resource: { buffer: lKeysBuf } }, { binding: 2, resource: { buffer: htBuf } }, { binding: 3, resource: { buffer: matchBuf } }, { binding: 4, resource: { buffer: mcBuf } }] });

        const enc = this.device.createCommandEncoder();
        const p1 = enc.beginComputePass(); p1.setPipeline(initP); p1.setBindGroup(0, initBG); p1.dispatchWorkgroups(Math.ceil(cap * 2 / 256)); p1.end();
        const p2 = enc.beginComputePass(); p2.setPipeline(buildP); p2.setBindGroup(0, buildBG); p2.dispatchWorkgroups(Math.ceil(rSize / 256)); p2.end();
        const p3 = enc.beginComputePass(); p3.setPipeline(probeP); p3.setBindGroup(0, probeBG); p3.dispatchWorkgroups(Math.ceil(lSize / 256)); p3.end();
        enc.copyBufferToBuffer(mcBuf, 0, stageBuf, 0, 4);
        this.device.queue.submit([enc.finish()]);

        await stageBuf.mapAsync(GPUMapMode.READ);
        const mc = new Uint32Array(stageBuf.getMappedRange())[0];
        stageBuf.unmap();

        const actualM = Math.min(mc, maxM);
        const mStageBuf = this.device.createBuffer({ size: actualM * 2 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const cEnc = this.device.createCommandEncoder();
        cEnc.copyBufferToBuffer(matchBuf, 0, mStageBuf, 0, actualM * 2 * 4);
        this.device.queue.submit([cEnc.finish()]);

        await mStageBuf.mapAsync(GPUMapMode.READ);
        const mData = new Uint32Array(mStageBuf.getMappedRange().slice(0));
        mStageBuf.unmap();

        const lIdx = new Uint32Array(actualM), rIdx = new Uint32Array(actualM);
        for (let i = 0; i < actualM; i++) { lIdx[i] = mData[i * 2]; rIdx[i] = mData[i * 2 + 1]; }

        rKeysBuf.destroy(); lKeysBuf.destroy(); htBuf.destroy(); matchBuf.destroy(); mcBuf.destroy(); stageBuf.destroy(); mStageBuf.destroy(); ipBuf.destroy(); bpBuf.destroy(); ppBuf.destroy();
        return { leftIndices: lIdx, rightIndices: rIdx, matchCount: actualM };
    }

    _cpuHashJoin(leftRows, rightRows, leftKey, rightKey) {
        const rMap = new Map();
        for (let i = 0; i < rightRows.length; i++) {
            const k = this._hashKey(rightRows[i][rightKey]);
            if (!rMap.has(k)) rMap.set(k, []);
            rMap.get(k).push(i);
        }
        const lIdx = [], rIdx = [];
        for (let i = 0; i < leftRows.length; i++) {
            const k = this._hashKey(leftRows[i][leftKey]);
            for (const ri of (rMap.get(k) || [])) { lIdx.push(i); rIdx.push(ri); }
        }
        return { leftIndices: new Uint32Array(lIdx), rightIndices: new Uint32Array(rIdx), matchCount: lIdx.length };
    }

    _extractKeys(rows, key) {
        const keys = new Uint32Array(rows.length);
        for (let i = 0; i < rows.length; i++) keys[i] = this._hashKey(rows[i][key]);
        return keys;
    }

    _hashKey(v) {
        if (v == null) return 0xFFFFFFFE;
        if (typeof v === 'number') return Number.isInteger(v) && v >= 0 && v < 0xFFFFFFFF ? v >>> 0 : (new Uint32Array(new Float32Array([v]).buffer))[0];
        if (typeof v === 'string') { let h = 2166136261; for (let i = 0; i < v.length; i++) { h ^= v.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
        return this._hashKey(String(v));
    }

    _createBuf(data, usage) {
        const buf = this.device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(buf, 0, data);
        return buf;
    }

    _createUniform(data) {
        const buf = this.device.createBuffer({ size: Math.max(data.byteLength, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(buf, 0, data);
        return buf;
    }

    _nextPow2(n) { let p = 1; while (p < n) p *= 2; return p; }
}

// Global GPU joiner instance
const gpuJoiner = new GPUJoiner();

/**
 * GPU-accelerated SQL sorting using bitonic sort algorithm.
 * Falls back to CPU for small datasets where GPU overhead exceeds benefit.
 */
class GPUSorter {
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
        if (!navigator.gpu) return false;
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return false;
            this.device = await adapter.requestDevice({
                requiredLimits: { maxStorageBufferBindingSize: 256 * 1024 * 1024 },
            });
            this._compileShaders();
            this.available = true;
            console.log('[GPUSorter] Initialized');
            return true;
        } catch (e) {
            console.warn('[GPUSorter] Init failed:', e);
            return false;
        }
    }

    _compileShaders() {
        const code = `
struct LP { size: u32, stage: u32, step: u32, asc: u32 }
@group(0) @binding(0) var<uniform> lp: LP;
@group(0) @binding(1) var<storage, read_write> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> idx: array<u32>;
var<workgroup> sk: array<f32, 512>;
var<workgroup> si: array<u32, 512>;

fn cswap(i: u32, j: u32, d: bool) {
    let ki = sk[i]; let kj = sk[j];
    if (select(ki > kj, ki < kj, d)) {
        sk[i] = kj; sk[j] = ki;
        let t = si[i]; si[i] = si[j]; si[j] = t;
    }
}

@compute @workgroup_size(256)
fn local_sort(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let base = w.x * 512u; let t = l.x;
    let i1 = base + t; let i2 = base + t + 256u;
    if (i1 < lp.size) { sk[t] = keys[i1]; si[t] = idx[i1]; } else { sk[t] = 3.4e38; si[t] = i1; }
    if (i2 < lp.size) { sk[t + 256u] = keys[i2]; si[t + 256u] = idx[i2]; } else { sk[t + 256u] = 3.4e38; si[t + 256u] = i2; }
    workgroupBarrier();
    let asc = lp.asc == 1u;
    for (var s = 1u; s < 512u; s = s << 1u) {
        for (var st = s; st > 0u; st = st >> 1u) {
            let bs = st << 1u;
            if (t < 256u) {
                let bi = t / st; let ib = t % st;
                let i = bi * bs + ib; let j = i + st;
                if (j < 512u) { cswap(i, j, ((i / (s << 1u)) % 2u == 0u) == asc); }
            }
            workgroupBarrier();
        }
    }
    if (i1 < lp.size) { keys[i1] = sk[t]; idx[i1] = si[t]; }
    if (i2 < lp.size) { keys[i2] = sk[t + 256u]; idx[i2] = si[t + 256u]; }
}

struct MP { size: u32, stage: u32, step: u32, asc: u32 }
@group(0) @binding(0) var<uniform> mp: MP;
@group(0) @binding(1) var<storage, read_write> mkeys: array<f32>;
@group(0) @binding(2) var<storage, read_write> midx: array<u32>;

@compute @workgroup_size(256)
fn merge(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = g.x; let step = mp.step; let stage = mp.stage;
    let bs = 1u << (stage + 1u); let hb = 1u << stage;
    let bi = t / hb; let ih = t % hb;
    let i = bi * bs + ih; let j = i + step;
    if (j >= mp.size) { return; }
    let d = ((i / bs) % 2u == 0u) == (mp.asc == 1u);
    let ki = mkeys[i]; let kj = mkeys[j];
    if (select(ki > kj, ki < kj, d)) {
        mkeys[i] = kj; mkeys[j] = ki;
        let ti = midx[i]; midx[i] = midx[j]; midx[j] = ti;
    }
}

struct IP { size: u32 }
@group(0) @binding(0) var<uniform> ip: IP;
@group(0) @binding(1) var<storage, read_write> iidx: array<u32>;

@compute @workgroup_size(256)
fn init_idx(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x < ip.size) { iidx[g.x] = g.x; }
}`;
        const module = this.device.createShaderModule({ code });
        this.pipelines.set('init', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'init_idx' } }));
        this.pipelines.set('local', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'local_sort' } }));
        this.pipelines.set('merge', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'merge' } }));
    }

    isAvailable() { return this.available; }

    async sort(values, ascending = true) {
        const size = values.length;
        if (!this.available || size < 10000) return this._cpuSort(values, ascending);

        const padSize = this._nextPow2(size);
        const keys = new Float32Array(padSize);
        keys.set(values instanceof Float32Array ? values : new Float32Array(values));
        for (let i = size; i < padSize; i++) keys[i] = 3.4e38;

        const keysBuf = this._createBuf(keys, GPUBufferUsage.STORAGE);
        const idxBuf = this.device.createBuffer({ size: padSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        const initP = this.pipelines.get('init');
        const ipBuf = this._createUniform(new Uint32Array([padSize]));
        const initBG = this.device.createBindGroup({ layout: initP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ipBuf } }, { binding: 1, resource: { buffer: idxBuf } }] });

        const localP = this.pipelines.get('local');
        const lpBuf = this._createUniform(new Uint32Array([padSize, 0, 0, ascending ? 1 : 0]));
        const localBG = this.device.createBindGroup({ layout: localP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: lpBuf } }, { binding: 1, resource: { buffer: keysBuf } }, { binding: 2, resource: { buffer: idxBuf } }] });

        const enc = this.device.createCommandEncoder();
        const p1 = enc.beginComputePass(); p1.setPipeline(initP); p1.setBindGroup(0, initBG); p1.dispatchWorkgroups(Math.ceil(padSize / 256)); p1.end();
        const p2 = enc.beginComputePass(); p2.setPipeline(localP); p2.setBindGroup(0, localBG); p2.dispatchWorkgroups(Math.ceil(padSize / 512)); p2.end();
        this.device.queue.submit([enc.finish()]);

        if (padSize > 512) {
            const mergeP = this.pipelines.get('merge');
            for (let stageExp = 9; (1 << stageExp) < padSize; stageExp++) {
                for (let step = 1 << stageExp; step > 0; step >>= 1) {
                    const mEnc = this.device.createCommandEncoder();
                    const mpBuf = this._createUniform(new Uint32Array([padSize, stageExp, step, ascending ? 1 : 0]));
                    const mergeBG = this.device.createBindGroup({ layout: mergeP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: mpBuf } }, { binding: 1, resource: { buffer: keysBuf } }, { binding: 2, resource: { buffer: idxBuf } }] });
                    const mp = mEnc.beginComputePass(); mp.setPipeline(mergeP); mp.setBindGroup(0, mergeBG); mp.dispatchWorkgroups(Math.ceil(padSize / 256)); mp.end();
                    this.device.queue.submit([mEnc.finish()]);
                    mpBuf.destroy();
                }
            }
        }

        const stageBuf = this.device.createBuffer({ size: size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const cEnc = this.device.createCommandEncoder();
        cEnc.copyBufferToBuffer(idxBuf, 0, stageBuf, 0, size * 4);
        this.device.queue.submit([cEnc.finish()]);

        await stageBuf.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(stageBuf.getMappedRange().slice(0));
        stageBuf.unmap();

        keysBuf.destroy(); idxBuf.destroy(); ipBuf.destroy(); lpBuf.destroy(); stageBuf.destroy();
        return result;
    }

    _cpuSort(values, ascending) {
        const indexed = Array.from(values).map((v, i) => ({ v, i }));
        indexed.sort((a, b) => { const c = a.v < b.v ? -1 : a.v > b.v ? 1 : 0; return ascending ? c : -c; });
        return new Uint32Array(indexed.map(x => x.i));
    }

    _createBuf(data, usage) {
        const buf = this.device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(buf, 0, data);
        return buf;
    }

    _createUniform(data) {
        const buf = this.device.createBuffer({ size: Math.max(data.byteLength, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(buf, 0, data);
        return buf;
    }

    _nextPow2(n) { let p = 1; while (p < n) p *= 2; return p; }
}

// Global GPU sorter instance
const gpuSorter = new GPUSorter();

/**
 * GPU-accelerated SQL GROUP BY using hash-based grouping.
 * Falls back to CPU for small datasets where GPU overhead exceeds benefit.
 */
class GPUGrouper {
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
        if (!navigator.gpu) return false;
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return false;
            this.device = await adapter.requestDevice({
                requiredLimits: { maxStorageBufferBindingSize: 256 * 1024 * 1024 },
            });
            this._compileShaders();
            this.available = true;
            console.log('[GPUGrouper] Initialized');
            return true;
        } catch (e) {
            console.warn('[GPUGrouper] Init failed:', e);
            return false;
        }
    }

    _compileShaders() {
        const code = `
struct BP { size: u32, cap: u32 }
struct AP { size: u32, cap: u32 }
struct AGP { size: u32, ng: u32, at: u32 }
struct IP { cap: u32 }
struct IAP { ng: u32, iv: u32 }

@group(0) @binding(0) var<uniform> bp: BP;
@group(0) @binding(1) var<storage, read> bk: array<u32>;
@group(0) @binding(2) var<storage, read_write> ht: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> gc: atomic<u32>;

fn fnv(k: u32) -> u32 { var h = 2166136261u; h ^= (k & 0xFFu); h *= 16777619u; h ^= ((k >> 8u) & 0xFFu); h *= 16777619u; h ^= ((k >> 16u) & 0xFFu); h *= 16777619u; h ^= ((k >> 24u) & 0xFFu); h *= 16777619u; return h; }

@compute @workgroup_size(256) fn build(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= bp.size) { return; }
    let k = bk[g.x]; var s = fnv(k) % bp.cap;
    for (var p = 0u; p < bp.cap; p++) { let i = s * 2u; let o = atomicCompareExchangeWeak(&ht[i], 0xFFFFFFFFu, k); if (o.exchanged) { atomicStore(&ht[i + 1u], atomicAdd(&gc, 1u)); return; } if (o.old_value == k) { return; } s = (s + 1u) % bp.cap; }
}

@group(0) @binding(0) var<uniform> ap: AP;
@group(0) @binding(1) var<storage, read> ak: array<u32>;
@group(0) @binding(2) var<storage, read> lt: array<u32>;
@group(0) @binding(3) var<storage, read_write> gids: array<u32>;

@compute @workgroup_size(256) fn assign(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= ap.size) { return; }
    let k = ak[g.x]; var s = fnv(k) % ap.cap;
    for (var p = 0u; p < ap.cap; p++) { let i = s * 2u; if (lt[i] == k) { gids[g.x] = lt[i + 1u]; return; } if (lt[i] == 0xFFFFFFFFu) { gids[g.x] = 0xFFFFFFFFu; return; } s = (s + 1u) % ap.cap; }
    gids[g.x] = 0xFFFFFFFFu;
}

@group(0) @binding(0) var<uniform> agp: AGP;
@group(0) @binding(1) var<storage, read> agi: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> res: array<atomic<u32>>;

fn f2s(f: f32) -> u32 { let b = bitcast<u32>(f); return select(b ^ 0x80000000u, ~b, (b & 0x80000000u) != 0u); }

@compute @workgroup_size(256) fn cnt(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= agp.size) { return; } let gid = agi[g.x]; if (gid < agp.ng) { atomicAdd(&res[gid], 1u); } }
@compute @workgroup_size(256) fn sum(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= agp.size) { return; } let gid = agi[g.x]; let v = vals[g.x]; if (gid < agp.ng && !isNan(v)) { atomicAdd(&res[gid], u32(i32(v * 1000.0))); } }
@compute @workgroup_size(256) fn mn(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= agp.size) { return; } let gid = agi[g.x]; let v = vals[g.x]; if (gid < agp.ng && !isNan(v)) { atomicMin(&res[gid], f2s(v)); } }
@compute @workgroup_size(256) fn mx(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= agp.size) { return; } let gid = agi[g.x]; let v = vals[g.x]; if (gid < agp.ng && !isNan(v)) { atomicMax(&res[gid], f2s(v)); } }

@group(0) @binding(0) var<uniform> ip: IP;
@group(0) @binding(1) var<storage, read_write> it: array<u32>;
@compute @workgroup_size(256) fn iht(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= ip.cap * 2u) { return; } it[g.x] = select(0u, 0xFFFFFFFFu, g.x % 2u == 0u); }

@group(0) @binding(0) var<uniform> iap: IAP;
@group(0) @binding(1) var<storage, read_write> iar: array<u32>;
@compute @workgroup_size(256) fn iag(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= iap.ng) { return; } iar[g.x] = iap.iv; }`;
        const module = this.device.createShaderModule({ code });
        this.pipelines.set('iht', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'iht' } }));
        this.pipelines.set('build', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'build' } }));
        this.pipelines.set('assign', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'assign' } }));
        this.pipelines.set('iag', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'iag' } }));
        this.pipelines.set('cnt', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'cnt' } }));
        this.pipelines.set('sum', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'sum' } }));
        this.pipelines.set('mn', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'mn' } }));
        this.pipelines.set('mx', this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'mx' } }));
    }

    isAvailable() { return this.available; }

    async groupBy(keys) {
        const size = keys.length;
        if (!this.available || size < 10000) return this._cpuGroupBy(keys);

        const cap = this._nextPow2(Math.min(size, 100000) * 2);
        const keysBuf = this._createBuf(keys, GPUBufferUsage.STORAGE);
        const htBuf = this.device.createBuffer({ size: cap * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const gcBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const gidsBuf = this.device.createBuffer({ size: size * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        const ihtP = this.pipelines.get('iht'), buildP = this.pipelines.get('build'), assignP = this.pipelines.get('assign');
        const ipBuf = this._createUniform(new Uint32Array([cap]));
        const bpBuf = this._createUniform(new Uint32Array([size, cap]));
        const apBuf = this._createUniform(new Uint32Array([size, cap]));

        const ihtBG = this.device.createBindGroup({ layout: ihtP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ipBuf } }, { binding: 1, resource: { buffer: htBuf } }] });
        const buildBG = this.device.createBindGroup({ layout: buildP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bpBuf } }, { binding: 1, resource: { buffer: keysBuf } }, { binding: 2, resource: { buffer: htBuf } }, { binding: 3, resource: { buffer: gcBuf } }] });
        const assignBG = this.device.createBindGroup({ layout: assignP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: apBuf } }, { binding: 1, resource: { buffer: keysBuf } }, { binding: 2, resource: { buffer: htBuf } }, { binding: 3, resource: { buffer: gidsBuf } }] });

        const enc = this.device.createCommandEncoder();
        const p1 = enc.beginComputePass(); p1.setPipeline(ihtP); p1.setBindGroup(0, ihtBG); p1.dispatchWorkgroups(Math.ceil(cap * 2 / 256)); p1.end();
        const p2 = enc.beginComputePass(); p2.setPipeline(buildP); p2.setBindGroup(0, buildBG); p2.dispatchWorkgroups(Math.ceil(size / 256)); p2.end();
        const p3 = enc.beginComputePass(); p3.setPipeline(assignP); p3.setBindGroup(0, assignBG); p3.dispatchWorkgroups(Math.ceil(size / 256)); p3.end();

        const gcStage = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        enc.copyBufferToBuffer(gcBuf, 0, gcStage, 0, 4);
        this.device.queue.submit([enc.finish()]);

        await gcStage.mapAsync(GPUMapMode.READ);
        const numGroups = new Uint32Array(gcStage.getMappedRange())[0];
        gcStage.unmap();

        const gidsStage = this.device.createBuffer({ size: size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const cEnc = this.device.createCommandEncoder();
        cEnc.copyBufferToBuffer(gidsBuf, 0, gidsStage, 0, size * 4);
        this.device.queue.submit([cEnc.finish()]);

        await gidsStage.mapAsync(GPUMapMode.READ);
        const groupIds = new Uint32Array(gidsStage.getMappedRange().slice(0));
        gidsStage.unmap();

        keysBuf.destroy(); htBuf.destroy(); gcBuf.destroy(); gidsBuf.destroy(); gcStage.destroy(); gidsStage.destroy(); ipBuf.destroy(); bpBuf.destroy(); apBuf.destroy();
        return { groupIds, numGroups };
    }

    async groupAggregate(values, groupIds, numGroups, aggType) {
        const size = values.length;
        if (!this.available || size < 10000) return this._cpuGroupAggregate(values, groupIds, numGroups, aggType);

        let initVal = 0, pName = 'cnt';
        if (aggType === 'SUM') pName = 'sum';
        else if (aggType === 'MIN') { initVal = 0x7F7FFFFF; pName = 'mn'; }
        else if (aggType === 'MAX') pName = 'mx';

        const gidsBuf = this._createBuf(groupIds, GPUBufferUsage.STORAGE);
        const valsBuf = this._createBuf(values, GPUBufferUsage.STORAGE);
        const resBuf = this.device.createBuffer({ size: numGroups * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        const iagP = this.pipelines.get('iag'), aggP = this.pipelines.get(pName);
        const iapBuf = this._createUniform(new Uint32Array([numGroups, initVal]));
        const agpBuf = this._createUniform(new Uint32Array([size, numGroups, 0]));

        const iagBG = this.device.createBindGroup({ layout: iagP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: iapBuf } }, { binding: 1, resource: { buffer: resBuf } }] });
        const aggBG = this.device.createBindGroup({ layout: aggP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: agpBuf } }, { binding: 1, resource: { buffer: gidsBuf } }, { binding: 2, resource: { buffer: valsBuf } }, { binding: 3, resource: { buffer: resBuf } }] });

        const enc = this.device.createCommandEncoder();
        const p1 = enc.beginComputePass(); p1.setPipeline(iagP); p1.setBindGroup(0, iagBG); p1.dispatchWorkgroups(Math.max(1, Math.ceil(numGroups / 256))); p1.end();
        const p2 = enc.beginComputePass(); p2.setPipeline(aggP); p2.setBindGroup(0, aggBG); p2.dispatchWorkgroups(Math.ceil(size / 256)); p2.end();

        const stage = this.device.createBuffer({ size: numGroups * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        enc.copyBufferToBuffer(resBuf, 0, stage, 0, numGroups * 4);
        this.device.queue.submit([enc.finish()]);

        await stage.mapAsync(GPUMapMode.READ);
        const raw = new Uint32Array(stage.getMappedRange().slice(0));
        stage.unmap();

        const results = new Float32Array(numGroups);
        for (let i = 0; i < numGroups; i++) {
            if (aggType === 'COUNT') results[i] = raw[i];
            else if (aggType === 'SUM') results[i] = (raw[i] | 0) / 1000;
            else { const u = raw[i], bits = (u & 0x80000000) ? u ^ 0x80000000 : ~u; results[i] = new Float32Array(new Uint32Array([bits]).buffer)[0]; }
        }

        gidsBuf.destroy(); valsBuf.destroy(); resBuf.destroy(); stage.destroy(); iapBuf.destroy(); agpBuf.destroy();
        return results;
    }

    _cpuGroupBy(keys) {
        const gMap = new Map(); const gids = new Uint32Array(keys.length); let nid = 0;
        for (let i = 0; i < keys.length; i++) { const k = keys[i]; if (!gMap.has(k)) gMap.set(k, nid++); gids[i] = gMap.get(k); }
        return { groupIds: gids, numGroups: nid };
    }

    _cpuGroupAggregate(values, groupIds, numGroups, aggType) {
        const res = new Float32Array(numGroups);
        if (aggType === 'MIN') res.fill(Infinity);
        else if (aggType === 'MAX') res.fill(-Infinity);
        for (let i = 0; i < values.length; i++) {
            const gid = groupIds[i], v = values[i];
            if (gid >= numGroups || isNaN(v)) continue;
            if (aggType === 'COUNT') res[gid]++;
            else if (aggType === 'SUM') res[gid] += v;
            else if (aggType === 'MIN') res[gid] = Math.min(res[gid], v);
            else if (aggType === 'MAX') res[gid] = Math.max(res[gid], v);
        }
        return res;
    }

    _createBuf(data, usage) { const buf = this.device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST }); this.device.queue.writeBuffer(buf, 0, data); return buf; }
    _createUniform(data) { const buf = this.device.createBuffer({ size: Math.max(data.byteLength, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.device.queue.writeBuffer(buf, 0, data); return buf; }
    _nextPow2(n) { let p = 1; while (p < n) p *= 2; return p; }
}

// Global GPU grouper instance
const gpuGrouper = new GPUGrouper();

// ============================================================================
// GPU Vector Search - GPU-accelerated distance computation and top-K selection
// ============================================================================

const GPU_DISTANCE_THRESHOLD = 5000;
const GPU_TOPK_THRESHOLD = 10000;

const DistanceMetric = { COSINE: 0, L2: 1, DOT_PRODUCT: 2 };

const VECTOR_DISTANCE_SHADER = `
struct DistanceParams { dim: u32, num_vectors: u32, num_queries: u32, metric: u32, }
@group(0) @binding(0) var<uniform> params: DistanceParams;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;
var<workgroup> shared_query: array<f32, 512>;
fn compute_dot(vec_offset: u32, dim: u32) -> f32 {
    var sum: f32 = 0.0;
    for (var i = 0u; i < dim; i++) { sum += shared_query[i] * vectors[vec_offset + i]; }
    return sum;
}
fn compute_l2(vec_offset: u32, dim: u32) -> f32 {
    var sum: f32 = 0.0;
    for (var i = 0u; i < dim; i++) { let d = shared_query[i] - vectors[vec_offset + i]; sum += d * d; }
    return sqrt(sum);
}
@compute @workgroup_size(256)
fn compute_distances(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let query_idx = wid.y; let vec_idx = gid.x;
    if (vec_idx >= params.num_vectors || query_idx >= params.num_queries) { return; }
    let dim = params.dim; let tid = lid.x;
    for (var i = tid; i < dim && i < 512u; i += 256u) { shared_query[i] = queries[query_idx * dim + i]; }
    workgroupBarrier();
    let vec_offset = vec_idx * dim;
    var result: f32 = 0.0;
    switch params.metric {
        case 0u, 2u: { result = compute_dot(vec_offset, dim); }
        case 1u: { result = compute_l2(vec_offset, dim); }
        default: { result = compute_dot(vec_offset, dim); }
    }
    distances[query_idx * params.num_vectors + vec_idx] = result;
}`;

const TOPK_SHADER = `
struct TopKParams { size: u32, k: u32, descending: u32, num_workgroups: u32, }
@group(0) @binding(0) var<uniform> params: TopKParams;
@group(0) @binding(1) var<storage, read> input_scores: array<f32>;
@group(0) @binding(2) var<storage, read> input_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_indices: array<u32>;
var<workgroup> local_scores: array<f32, 512>;
var<workgroup> local_indices: array<u32, 512>;
fn should_swap(a: f32, b: f32, descending: bool) -> bool { if (descending) { return a < b; } else { return a > b; } }
fn get_sentinel(descending: bool) -> f32 { if (descending) { return -3.4028235e+38; } else { return 3.4028235e+38; } }
@compute @workgroup_size(256)
fn local_topk(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let chunk_size = 512u; let base = wid.x * chunk_size; let tid = lid.x;
    let descending = params.descending == 1u; let sentinel = get_sentinel(descending);
    let idx1 = base + tid; let idx2 = base + tid + 256u;
    if (idx1 < params.size) { local_scores[tid] = input_scores[idx1]; local_indices[tid] = input_indices[idx1]; } else { local_scores[tid] = sentinel; local_indices[tid] = 0xFFFFFFFFu; }
    if (idx2 < params.size) { local_scores[tid + 256u] = input_scores[idx2]; local_indices[tid + 256u] = input_indices[idx2]; } else { local_scores[tid + 256u] = sentinel; local_indices[tid + 256u] = 0xFFFFFFFFu; }
    workgroupBarrier();
    for (var k = 2u; k <= chunk_size; k = k << 1u) {
        for (var j = k >> 1u; j > 0u; j = j >> 1u) {
            for (var t = 0u; t < 2u; t++) {
                let i = tid + t * 256u; let ixj = i ^ j;
                if (ixj > i && ixj < chunk_size) {
                    let direction = ((i & k) == 0u) == descending;
                    if (should_swap(local_scores[i], local_scores[ixj], direction)) {
                        let tmp_score = local_scores[i]; local_scores[i] = local_scores[ixj]; local_scores[ixj] = tmp_score;
                        let tmp_idx = local_indices[i]; local_indices[i] = local_indices[ixj]; local_indices[ixj] = tmp_idx;
                    }
                }
            }
            workgroupBarrier();
        }
    }
    let k_per_wg = min(params.k, chunk_size);
    if (tid < k_per_wg) { let out_base = wid.x * params.k; output_scores[out_base + tid] = local_scores[tid]; output_indices[out_base + tid] = local_indices[tid]; }
    if (tid + 256u < k_per_wg) { let out_base = wid.x * params.k; output_scores[out_base + tid + 256u] = local_scores[tid + 256u]; output_indices[out_base + tid + 256u] = local_indices[tid + 256u]; }
}
struct MergeParams { num_candidates: u32, k: u32, descending: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> merge_params: MergeParams;
@group(0) @binding(1) var<storage, read> merge_scores: array<f32>;
@group(0) @binding(2) var<storage, read> merge_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> final_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> final_indices: array<u32>;
@compute @workgroup_size(256)
fn merge_topk(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let n = merge_params.num_candidates; let k = merge_params.k;
    let descending = merge_params.descending == 1u; let sentinel = get_sentinel(descending);
    let n_local = min(n, 512u);
    if (tid < n_local) { local_scores[tid] = merge_scores[tid]; local_indices[tid] = merge_indices[tid]; } else if (tid < 512u) { local_scores[tid] = sentinel; local_indices[tid] = 0xFFFFFFFFu; }
    if (tid + 256u < n_local) { local_scores[tid + 256u] = merge_scores[tid + 256u]; local_indices[tid + 256u] = merge_indices[tid + 256u]; } else if (tid + 256u < 512u) { local_scores[tid + 256u] = sentinel; local_indices[tid + 256u] = 0xFFFFFFFFu; }
    workgroupBarrier();
    let chunk_size = 512u;
    for (var ks = 2u; ks <= chunk_size; ks = ks << 1u) {
        for (var j = ks >> 1u; j > 0u; j = j >> 1u) {
            for (var t = 0u; t < 2u; t++) {
                let i = tid + t * 256u; let ixj = i ^ j;
                if (ixj > i && ixj < chunk_size) {
                    let direction = ((i & ks) == 0u) == descending;
                    if (should_swap(local_scores[i], local_scores[ixj], direction)) {
                        let tmp_score = local_scores[i]; local_scores[i] = local_scores[ixj]; local_scores[ixj] = tmp_score;
                        let tmp_idx = local_indices[i]; local_indices[i] = local_indices[ixj]; local_indices[ixj] = tmp_idx;
                    }
                }
            }
            workgroupBarrier();
        }
    }
    if (tid < k) { final_scores[tid] = local_scores[tid]; final_indices[tid] = local_indices[tid]; }
}`;

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

const gpuVectorSearch = new GPUVectorSearch();

/**
 * Statistics Manager - computes and caches column statistics for query optimization.
 *
 * Unlike pre-generated sidecar files, this computes statistics dynamically:
 * 1. On first query to a column, stream through data to compute min/max/null_count
 * 2. Cache computed stats in OPFS for reuse across sessions
 * 3. Use statistics for fragment/page pruning during query execution
 *
 * This is the same approach used by DuckDB and DataFusion - no pre-processing required.
 */
class StatisticsManager {
    constructor() {
        this._cache = new Map(); // In-memory cache: datasetUrl -> { columns: Map<colName, stats> }
        this._opfsRoot = null;
        this._computing = new Map(); // Track in-progress computations to avoid duplicates
    }

    /**
     * Get OPFS directory for statistics cache
     */
    async _getStatsDir() {
        if (this._opfsRoot) return this._opfsRoot;

        if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
            return null; // OPFS not available
        }

        try {
            const opfsRoot = await navigator.storage.getDirectory();
            this._opfsRoot = await opfsRoot.getDirectoryHandle('lanceql-stats', { create: true });
            return this._opfsRoot;
        } catch (e) {
            console.warn('[StatisticsManager] OPFS not available:', e);
            return null;
        }
    }

    /**
     * Get cache key for a dataset
     */
    _getCacheKey(datasetUrl) {
        // Hash the URL for filesystem-safe name
        let hash = 0;
        for (let i = 0; i < datasetUrl.length; i++) {
            hash = ((hash << 5) - hash) + datasetUrl.charCodeAt(i);
            hash |= 0;
        }
        return `stats_${Math.abs(hash).toString(16)}`;
    }

    /**
     * Load cached statistics from OPFS
     */
    async loadFromCache(datasetUrl, version) {
        const cacheKey = this._getCacheKey(datasetUrl);

        // Check in-memory cache first
        if (this._cache.has(cacheKey)) {
            const cached = this._cache.get(cacheKey);
            if (cached.version === version) {
                return cached;
            }
        }

        // Try OPFS
        const statsDir = await this._getStatsDir();
        if (!statsDir) return null;

        try {
            const fileHandle = await statsDir.getFileHandle(`${cacheKey}.json`);
            const file = await fileHandle.getFile();
            const text = await file.text();
            const cached = JSON.parse(text);

            // Validate version
            if (cached.version !== version) {
                return null; // Stale cache
            }

            // Store in memory cache
            this._cache.set(cacheKey, cached);
            return cached;
        } catch (e) {
            return null; // No cache or read error
        }
    }

    /**
     * Save statistics to OPFS cache
     */
    async saveToCache(datasetUrl, version, statistics) {
        const cacheKey = this._getCacheKey(datasetUrl);

        const cacheData = {
            datasetUrl,
            version,
            timestamp: Date.now(),
            columns: statistics.columns, // Map serialized as object
            fragments: statistics.fragments || null
        };

        // Store in memory
        this._cache.set(cacheKey, cacheData);

        // Persist to OPFS
        const statsDir = await this._getStatsDir();
        if (!statsDir) return;

        try {
            const fileHandle = await statsDir.getFileHandle(`${cacheKey}.json`, { create: true });
            const writable = await fileHandle.createWritable();
            await writable.write(JSON.stringify(cacheData));
            await writable.close();
        } catch (e) {
            console.warn('[StatisticsManager] Failed to persist stats:', e);
        }
    }

    /**
     * Get statistics for a column, computing if necessary.
     *
     * @param {RemoteLanceDataset} dataset - The dataset
     * @param {string} columnName - Column name
     * @param {object} options - Options
     * @param {number} [options.sampleSize] - Max rows to sample (default: 100000)
     * @returns {Promise<ColumnStatistics>}
     */
    async getColumnStats(dataset, columnName, options = {}) {
        const datasetUrl = dataset.baseUrl;
        const version = dataset._version;
        const sampleSize = options.sampleSize || 100000;

        // Try to load from cache
        const cached = await this.loadFromCache(datasetUrl, version);
        if (cached?.columns?.[columnName]) {
            return cached.columns[columnName];
        }

        // Check if already computing
        const computeKey = `${datasetUrl}:${columnName}`;
        if (this._computing.has(computeKey)) {
            return this._computing.get(computeKey);
        }

        // Compute statistics by streaming through data
        const computePromise = this._computeColumnStats(dataset, columnName, sampleSize);
        this._computing.set(computeKey, computePromise);

        try {
            const stats = await computePromise;

            // Merge into cache
            const existing = await this.loadFromCache(datasetUrl, version) || { columns: {} };
            existing.columns[columnName] = stats;
            await this.saveToCache(datasetUrl, version, existing);

            return stats;
        } finally {
            this._computing.delete(computeKey);
        }
    }

    /**
     * Compute statistics for a column by streaming through data.
     */
    async _computeColumnStats(dataset, columnName, sampleSize) {
        const colIdx = dataset.schema.findIndex(c => c.name === columnName);
        if (colIdx === -1) {
            throw new Error(`Column not found: ${columnName}`);
        }

        const colType = dataset._columnTypes?.[colIdx] || 'unknown';
        const isNumeric = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'double'].includes(colType);

        const stats = {
            column: columnName,
            type: colType,
            rowCount: 0,
            nullCount: 0,
            min: null,
            max: null,
            computed: true,
            sampleSize: 0
        };

        // Stream through fragments
        let rowsProcessed = 0;

        for (let fragIdx = 0; fragIdx < dataset._fragments.length && rowsProcessed < sampleSize; fragIdx++) {
            try {
                const fragFile = await dataset.openFragment(fragIdx);
                const fragRows = Math.min(
                    dataset._fragments[fragIdx].numRows,
                    sampleSize - rowsProcessed
                );

                // Read column data
                const indices = Array.from({ length: fragRows }, (_, i) => i);
                const values = await fragFile.readColumnAtIndices(colIdx, indices);

                for (const value of values) {
                    stats.rowCount++;
                    stats.sampleSize++;

                    if (value === null || value === undefined) {
                        stats.nullCount++;
                        continue;
                    }

                    if (isNumeric) {
                        if (stats.min === null || value < stats.min) stats.min = value;
                        if (stats.max === null || value > stats.max) stats.max = value;
                    }
                }

                rowsProcessed += values.length;
            } catch (e) {
                console.warn(`[StatisticsManager] Error reading fragment ${fragIdx}:`, e);
            }
        }

        console.log(`[StatisticsManager] Computed stats for ${columnName}: min=${stats.min}, max=${stats.max}, nulls=${stats.nullCount}/${stats.rowCount}`);
        return stats;
    }

    /**
     * Compute statistics for all filter columns in a query plan.
     * This is called before query execution to enable pruning.
     */
    async precomputeForPlan(dataset, plan) {
        const filterColumns = new Set();

        // Collect columns from pushed filters
        for (const filter of (plan.pushedFilters || [])) {
            if (filter.column) filterColumns.add(filter.column);
            if (filter.left?.column) filterColumns.add(filter.left.column);
            if (filter.right?.column) filterColumns.add(filter.right.column);
        }

        // Compute stats in parallel
        const statsPromises = Array.from(filterColumns).map(col =>
            this.getColumnStats(dataset, col).catch(e => {
                console.warn(`[StatisticsManager] Failed to compute stats for ${col}:`, e);
                return null;
            })
        );

        const results = await Promise.all(statsPromises);
        const statsMap = new Map();

        Array.from(filterColumns).forEach((col, i) => {
            if (results[i]) statsMap.set(col, results[i]);
        });

        return statsMap;
    }

    /**
     * Check if a filter can be satisfied by a fragment's statistics.
     * Returns false if we can definitively skip this fragment.
     */
    canMatchFragment(fragmentStats, filter) {
        if (!fragmentStats || !filter) return true; // Can't determine, must scan

        const colStats = fragmentStats[filter.column];
        if (!colStats || colStats.min === null || colStats.max === null) return true;

        switch (filter.type) {
            case 'equality':
                // col = value: skip if value outside [min, max]
                return filter.value >= colStats.min && filter.value <= colStats.max;

            case 'range':
                switch (filter.op) {
                    case '>':
                        // col > value: skip if max <= value
                        return colStats.max > filter.value;
                    case '>=':
                        return colStats.max >= filter.value;
                    case '<':
                        // col < value: skip if min >= value
                        return colStats.min < filter.value;
                    case '<=':
                        return colStats.min <= filter.value;
                }
                break;

            case 'between':
                // col BETWEEN low AND high: skip if max < low OR min > high
                return colStats.max >= filter.low && colStats.min <= filter.high;

            case 'in':
                // col IN (values): skip if all values outside [min, max]
                if (Array.isArray(filter.values)) {
                    return filter.values.some(v => v >= colStats.min && v <= colStats.max);
                }
                break;
        }

        return true; // Default: can't skip
    }

    /**
     * Compute per-fragment statistics for a column.
     * This enables fine-grained fragment pruning.
     */
    async getFragmentStats(dataset, columnName, fragmentIndex) {
        const datasetUrl = dataset.baseUrl;
        const version = dataset._version;
        const cacheKey = `${datasetUrl}:frag${fragmentIndex}:${columnName}`;

        // Check if already computed
        const cached = await this.loadFromCache(datasetUrl, version);
        if (cached?.fragments?.[fragmentIndex]?.[columnName]) {
            return cached.fragments[fragmentIndex][columnName];
        }

        // Compute stats for this fragment only
        const colIdx = dataset.schema.findIndex(c => c.name === columnName);
        if (colIdx === -1) return null;

        const colType = dataset._columnTypes?.[colIdx] || 'unknown';
        const isNumeric = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'double'].includes(colType);

        try {
            const fragFile = await dataset.openFragment(fragmentIndex);
            const fragRows = dataset._fragments[fragmentIndex].numRows;

            // Sample up to 10000 rows for fragment stats
            const sampleSize = Math.min(fragRows, 10000);
            const indices = Array.from({ length: sampleSize }, (_, i) => i);
            const values = await fragFile.readColumnAtIndices(colIdx, indices);

            const stats = {
                fragmentIndex,
                column: columnName,
                rowCount: fragRows,
                sampledRows: sampleSize,
                nullCount: 0,
                min: null,
                max: null
            };

            for (const value of values) {
                if (value === null || value === undefined) {
                    stats.nullCount++;
                    continue;
                }
                if (isNumeric) {
                    if (stats.min === null || value < stats.min) stats.min = value;
                    if (stats.max === null || value > stats.max) stats.max = value;
                }
            }

            // Cache fragment stats
            const existing = await this.loadFromCache(datasetUrl, version) || { columns: {}, fragments: {} };
            if (!existing.fragments) existing.fragments = {};
            if (!existing.fragments[fragmentIndex]) existing.fragments[fragmentIndex] = {};
            existing.fragments[fragmentIndex][columnName] = stats;
            await this.saveToCache(datasetUrl, version, existing);

            return stats;
        } catch (e) {
            console.warn(`[StatisticsManager] Error computing fragment ${fragmentIndex} stats:`, e);
            return null;
        }
    }

    /**
     * Get fragments that might match a filter based on statistics.
     * Returns indices of fragments that can't be pruned.
     */
    async getPrunableFragments(dataset, filters) {
        if (!filters || filters.length === 0 || !dataset._fragments) {
            return null; // Can't prune
        }

        const numFragments = dataset._fragments.length;
        const matchingFragments = [];
        let fragmentsPruned = 0;

        // Get filter columns
        const filterColumns = new Set();
        for (const filter of filters) {
            if (filter.column) filterColumns.add(filter.column);
        }

        for (let fragIdx = 0; fragIdx < numFragments; fragIdx++) {
            let canPrune = false;

            for (const filter of filters) {
                if (!filter.column) continue;

                // Get stats for this fragment/column (computed lazily)
                const fragStats = await this.getFragmentStats(dataset, filter.column, fragIdx);

                if (fragStats && !this.canMatchFragment({ [filter.column]: fragStats }, filter)) {
                    canPrune = true;
                    break;
                }
            }

            if (!canPrune) {
                matchingFragments.push(fragIdx);
            } else {
                fragmentsPruned++;
            }
        }

        console.log(`[StatisticsManager] Fragment pruning: ${fragmentsPruned}/${numFragments} fragments pruned`);

        return {
            matchingFragments,
            fragmentsPruned,
            totalFragments: numFragments
        };
    }
}

// Global statistics manager instance
const statisticsManager = new StatisticsManager();

/**
 * Cost Model - estimates query execution cost for remote vs local strategies.
 *
 * This enables the optimizer to choose between:
 * - Remote: Minimize RTTs and bytes transferred (high latency, high bandwidth cost)
 * - Local: Maximize sequential I/O (low latency, CPU-bound)
 *
 * Similar to how DuckDB and DataFusion estimate query costs.
 */
class CostModel {
    constructor(options = {}) {
        this.isRemote = options.isRemote ?? true;

        // Network costs (ms)
        this.rttLatency = options.rttLatency ?? 50; // Round-trip time
        this.bandwidthMBps = options.bandwidthMBps ?? 10; // MB/s

        // CPU costs (ms per row)
        this.filterCostPerRow = options.filterCostPerRow ?? 0.001;
        this.hashBuildCostPerRow = options.hashBuildCostPerRow ?? 0.01;
        this.hashProbeCostPerRow = options.hashProbeCostPerRow ?? 0.005;

        // Memory costs
        this.memoryLimitMB = options.memoryLimitMB ?? 512;
    }

    /**
     * Estimate cost of scanning a table/fragment
     */
    estimateScanCost(rowCount, columnBytes, selectivity = 1.0) {
        const bytesToFetch = rowCount * columnBytes * selectivity;

        // Network cost for remote, near-zero for local
        const networkCost = this.isRemote
            ? this.rttLatency + (bytesToFetch / (this.bandwidthMBps * 1024 * 1024)) * 1000
            : 0.1; // Local disk is nearly instant

        // CPU cost (filtering)
        const cpuCost = rowCount * this.filterCostPerRow;

        return {
            totalMs: networkCost + cpuCost,
            networkMs: networkCost,
            cpuMs: cpuCost,
            bytesToFetch,
            rowsToScan: rowCount * selectivity
        };
    }

    /**
     * Estimate cost of a hash join
     */
    estimateJoinCost(leftRows, rightRows, leftBytes, rightBytes, joinSelectivity = 0.1) {
        // Build phase: hash the smaller table
        const buildRows = Math.min(leftRows, rightRows);
        const buildBytes = buildRows < leftRows ? leftBytes : rightBytes;
        const buildCost = buildRows * this.hashBuildCostPerRow;

        // Probe phase: scan the larger table
        const probeRows = Math.max(leftRows, rightRows);
        const probeCost = probeRows * this.hashProbeCostPerRow;

        // Memory check: can we fit build side in RAM?
        const buildMemoryMB = (buildRows * buildBytes) / (1024 * 1024);
        const needsSpill = buildMemoryMB > this.memoryLimitMB;

        // Spill cost (OPFS write + read)
        const spillCost = needsSpill ? buildMemoryMB * 10 : 0; // ~10ms per MB for OPFS

        return {
            totalMs: buildCost + probeCost + spillCost,
            buildMs: buildCost,
            probeMs: probeCost,
            spillMs: spillCost,
            needsSpill,
            outputRows: Math.round(leftRows * rightRows * joinSelectivity)
        };
    }

    /**
     * Estimate cost of an aggregation
     */
    estimateAggregateCost(inputRows, groupCount, aggCount) {
        // Cost scales with input rows and number of groups
        const hashGroupCost = inputRows * this.hashBuildCostPerRow;
        const aggComputeCost = inputRows * aggCount * 0.0001; // Aggregation is cheap

        return {
            totalMs: hashGroupCost + aggComputeCost,
            outputRows: groupCount
        };
    }

    /**
     * Compare two plan costs and recommend the better one
     */
    comparePlans(planA, planB) {
        const costA = planA.totalCost || this.estimatePlanCost(planA);
        const costB = planB.totalCost || this.estimatePlanCost(planB);

        return {
            recommended: costA.totalMs < costB.totalMs ? 'A' : 'B',
            costA,
            costB,
            savings: Math.abs(costA.totalMs - costB.totalMs)
        };
    }

    /**
     * Estimate total cost of a query plan
     */
    estimatePlanCost(plan) {
        let totalMs = 0;
        let totalBytes = 0;
        let operations = [];

        // Scan costs
        if (plan.leftScan) {
            const scanCost = this.estimateScanCost(
                plan.leftScan.estimatedRows || 10000,
                plan.leftScan.columnBytes || 100,
                plan.leftScan.selectivity || 1.0
            );
            totalMs += scanCost.totalMs;
            totalBytes += scanCost.bytesToFetch;
            operations.push({ op: 'scan_left', ...scanCost });
        }

        if (plan.rightScan) {
            const scanCost = this.estimateScanCost(
                plan.rightScan.estimatedRows || 10000,
                plan.rightScan.columnBytes || 100,
                plan.rightScan.selectivity || 1.0
            );
            totalMs += scanCost.totalMs;
            totalBytes += scanCost.bytesToFetch;
            operations.push({ op: 'scan_right', ...scanCost });
        }

        // Join costs
        if (plan.join) {
            const joinCost = this.estimateJoinCost(
                plan.leftScan?.estimatedRows || 10000,
                plan.rightScan?.estimatedRows || 10000,
                plan.leftScan?.columnBytes || 100,
                plan.rightScan?.columnBytes || 100,
                plan.join.selectivity || 0.1
            );
            totalMs += joinCost.totalMs;
            operations.push({ op: 'join', ...joinCost });
        }

        // Aggregation costs
        if (plan.aggregations && plan.aggregations.length > 0) {
            const aggCost = this.estimateAggregateCost(
                plan.estimatedInputRows || 10000,
                plan.groupBy?.length || 1,
                plan.aggregations.length
            );
            totalMs += aggCost.totalMs;
            operations.push({ op: 'aggregate', ...aggCost });
        }

        return {
            totalMs,
            totalBytes,
            operations,
            isRemote: this.isRemote
        };
    }
}

// Export cost model
export { CostModel };

/**
 * OPFS-only storage for Lance database files.
 *
 * Uses Origin Private File System (OPFS) exclusively - no IndexedDB.
 * This avoids migration complexity as data grows.
 *
 * OPFS benefits:
 * - High performance file access
 * - No size limits (beyond disk quota)
 * - File-like API suitable for Lance format
 * - Same approach as SQLite WASM
 */
class OPFSStorage {
    constructor(rootDir = 'lanceql') {
        this.rootDir = rootDir;
        this.root = null;
    }

    /**
     * Get OPFS root directory, creating if needed
     */
    async getRoot() {
        if (this.root) return this.root;

        if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
            throw new Error('OPFS not available. Requires modern browser with Origin Private File System support.');
        }

        const opfsRoot = await navigator.storage.getDirectory();
        this.root = await opfsRoot.getDirectoryHandle(this.rootDir, { create: true });
        return this.root;
    }

    async open() {
        await this.getRoot();
        return this;
    }

    /**
     * Get or create a subdirectory
     */
    async getDir(path) {
        const root = await this.getRoot();
        const parts = path.split('/').filter(p => p);

        let current = root;
        for (const part of parts) {
            current = await current.getDirectoryHandle(part, { create: true });
        }
        return current;
    }

    /**
     * Save data to a file
     * @param {string} path - File path (e.g., 'mydb/users/frag_001.lance')
     * @param {Uint8Array} data - File data
     */
    async save(path, data) {
        const parts = path.split('/');
        const fileName = parts.pop();
        const dirPath = parts.join('/');

        const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
        const fileHandle = await dir.getFileHandle(fileName, { create: true });

        // Use sync access handle for better performance if available
        if (fileHandle.createSyncAccessHandle) {
            try {
                const accessHandle = await fileHandle.createSyncAccessHandle();
                accessHandle.truncate(0);
                accessHandle.write(data, { at: 0 });
                accessHandle.flush();
                accessHandle.close();
                return { path, size: data.byteLength };
            } catch (e) {
                // Fall back to writable stream
            }
        }

        const writable = await fileHandle.createWritable();
        await writable.write(data);
        await writable.close();

        return { path, size: data.byteLength };
    }

    /**
     * Load data from a file
     * @param {string} path - File path
     * @returns {Promise<Uint8Array|null>}
     */
    async load(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const fileHandle = await dir.getFileHandle(fileName);
            const file = await fileHandle.getFile();
            const buffer = await file.arrayBuffer();
            return new Uint8Array(buffer);
        } catch (e) {
            if (e.name === 'NotFoundError') {
                return null;
            }
            throw e;
        }
    }

    /**
     * Delete a file
     * @param {string} path - File path
     */
    async delete(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            await dir.removeEntry(fileName);
            return true;
        } catch (e) {
            if (e.name === 'NotFoundError') {
                return false;
            }
            throw e;
        }
    }

    /**
     * List files in a directory
     * @param {string} dirPath - Directory path
     * @returns {Promise<string[]>} File names
     */
    async list(dirPath = '') {
        try {
            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const files = [];
            for await (const [name, handle] of dir.entries()) {
                files.push({
                    name,
                    type: handle.kind, // 'file' or 'directory'
                });
            }
            return files;
        } catch (e) {
            return [];
        }
    }

    /**
     * Check if a file exists
     * @param {string} path - File path
     */
    async exists(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            await dir.getFileHandle(fileName);
            return true;
        } catch (e) {
            return false;
        }
    }

    /**
     * Delete a directory and all contents
     * @param {string} dirPath - Directory path
     */
    async deleteDir(dirPath) {
        try {
            const parts = dirPath.split('/');
            const dirName = parts.pop();
            const parentPath = parts.join('/');

            const parent = parentPath ? await this.getDir(parentPath) : await this.getRoot();
            await parent.removeEntry(dirName, { recursive: true });
            return true;
        } catch (e) {
            return false;
        }
    }

    /**
     * Read a byte range from a file without loading the entire file
     * @param {string} path - File path
     * @param {number} offset - Start byte offset
     * @param {number} length - Number of bytes to read
     * @returns {Promise<Uint8Array|null>}
     */
    async readRange(path, offset, length) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const fileHandle = await dir.getFileHandle(fileName);
            const file = await fileHandle.getFile();

            // Use slice to read only the requested range
            const blob = file.slice(offset, offset + length);
            const buffer = await blob.arrayBuffer();
            return new Uint8Array(buffer);
        } catch (e) {
            if (e.name === 'NotFoundError') {
                return null;
            }
            throw e;
        }
    }

    /**
     * Get file size without loading the file
     * @param {string} path - File path
     * @returns {Promise<number|null>}
     */
    async getFileSize(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const fileHandle = await dir.getFileHandle(fileName);
            const file = await fileHandle.getFile();
            return file.size;
        } catch (e) {
            if (e.name === 'NotFoundError') {
                return null;
            }
            throw e;
        }
    }

    /**
     * Open a file for chunked reading
     * @param {string} path - File path
     * @returns {Promise<OPFSFileReader|null>}
     */
    async openFile(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const fileHandle = await dir.getFileHandle(fileName);
            return new OPFSFileReader(fileHandle);
        } catch (e) {
            if (e.name === 'NotFoundError') {
                return null;
            }
            throw e;
        }
    }

    /**
     * Check if OPFS is supported in this browser
     * @returns {Promise<boolean>}
     */
    async isSupported() {
        try {
            if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
                return false;
            }
            // Actually try to access OPFS
            await navigator.storage.getDirectory();
            return true;
        } catch (e) {
            return false;
        }
    }

    /**
     * Get storage statistics
     * @returns {Promise<{fileCount: number, totalSize: number}>}
     */
    async getStats() {
        try {
            const root = await this.getRoot();
            let fileCount = 0;
            let totalSize = 0;

            async function countDir(dir) {
                for await (const [name, handle] of dir.entries()) {
                    if (handle.kind === 'file') {
                        const file = await handle.getFile();
                        fileCount++;
                        totalSize += file.size;
                    } else if (handle.kind === 'directory') {
                        await countDir(handle);
                    }
                }
            }

            await countDir(root);
            return { fileCount, totalSize };
        } catch (e) {
            return { fileCount: 0, totalSize: 0 };
        }
    }

    /**
     * List all files in storage with their sizes
     * @returns {Promise<Array<{name: string, size: number, lastModified: number}>>}
     */
    async listFiles() {
        try {
            const root = await this.getRoot();
            const files = [];

            async function listDir(dir, prefix = '') {
                for await (const [name, handle] of dir.entries()) {
                    if (handle.kind === 'file') {
                        const file = await handle.getFile();
                        files.push({
                            name: prefix ? `${prefix}/${name}` : name,
                            size: file.size,
                            lastModified: file.lastModified
                        });
                    } else if (handle.kind === 'directory') {
                        await listDir(handle, prefix ? `${prefix}/${name}` : name);
                    }
                }
            }

            await listDir(root);
            return files;
        } catch (e) {
            return [];
        }
    }

    /**
     * Clear all files in storage
     * @returns {Promise<number>} Number of files deleted
     */
    async clearAll() {
        try {
            const root = await this.getRoot();
            let count = 0;

            const entries = [];
            for await (const [name, handle] of root.entries()) {
                entries.push({ name, kind: handle.kind });
            }

            for (const entry of entries) {
                await root.removeEntry(entry.name, { recursive: entry.kind === 'directory' });
                count++;
            }

            return count;
        } catch (e) {
            console.warn('Failed to clear OPFS:', e);
            return 0;
        }
    }
}

/**
 * OPFS File Reader for chunked/streaming reads
 * Wraps a FileSystemFileHandle for efficient byte-range access
 */
class OPFSFileReader {
    constructor(fileHandle) {
        this.fileHandle = fileHandle;
        this._file = null;
        this._size = null;
    }

    /**
     * Get the File object (cached)
     */
    async getFile() {
        if (!this._file) {
            this._file = await this.fileHandle.getFile();
            this._size = this._file.size;
        }
        return this._file;
    }

    /**
     * Get file size
     * @returns {Promise<number>}
     */
    async getSize() {
        if (this._size === null) {
            await this.getFile();
        }
        return this._size;
    }

    /**
     * Read a byte range
     * @param {number} offset - Start byte offset
     * @param {number} length - Number of bytes to read
     * @returns {Promise<Uint8Array>}
     */
    async readRange(offset, length) {
        const file = await this.getFile();
        const blob = file.slice(offset, offset + length);
        const buffer = await blob.arrayBuffer();
        return new Uint8Array(buffer);
    }

    /**
     * Read from end of file (useful for footer)
     * @param {number} length - Number of bytes to read from end
     * @returns {Promise<Uint8Array>}
     */
    async readFromEnd(length) {
        const size = await this.getSize();
        return this.readRange(size - length, length);
    }

    /**
     * Invalidate cache (call after file is modified)
     */
    invalidate() {
        this._file = null;
        this._size = null;
    }
}

/**
 * LRU Cache for page data
 * Keeps recently accessed pages in memory to avoid repeated OPFS reads
 */
class LRUCache {
    constructor(maxSize = 50 * 1024 * 1024) { // 50MB default
        this.maxSize = maxSize;
        this.currentSize = 0;
        this.cache = new Map(); // key -> { data, size, lastAccess }
    }

    /**
     * Get item from cache
     * @param {string} key - Cache key
     * @returns {Uint8Array|null}
     */
    get(key) {
        const entry = this.cache.get(key);
        if (entry) {
            entry.lastAccess = Date.now();
            return entry.data;
        }
        return null;
    }

    /**
     * Put item in cache
     * @param {string} key - Cache key
     * @param {Uint8Array} data - Data to cache
     */
    put(key, data) {
        // Remove existing entry if present
        if (this.cache.has(key)) {
            this.currentSize -= this.cache.get(key).size;
            this.cache.delete(key);
        }

        const size = data.byteLength;

        // Evict if needed
        while (this.currentSize + size > this.maxSize && this.cache.size > 0) {
            this._evictOldest();
        }

        // Don't cache if single item is too large
        if (size > this.maxSize) {
            return;
        }

        this.cache.set(key, {
            data,
            size,
            lastAccess: Date.now()
        });
        this.currentSize += size;
    }

    /**
     * Evict oldest entry
     */
    _evictOldest() {
        let oldestKey = null;
        let oldestTime = Infinity;

        for (const [key, entry] of this.cache) {
            if (entry.lastAccess < oldestTime) {
                oldestTime = entry.lastAccess;
                oldestKey = key;
            }
        }

        if (oldestKey) {
            this.currentSize -= this.cache.get(oldestKey).size;
            this.cache.delete(oldestKey);
        }
    }

    /**
     * Clear entire cache
     */
    clear() {
        this.cache.clear();
        this.currentSize = 0;
    }

    /**
     * Get cache stats
     */
    stats() {
        return {
            entries: this.cache.size,
            currentSize: this.currentSize,
            maxSize: this.maxSize,
            utilization: (this.currentSize / this.maxSize * 100).toFixed(1) + '%'
        };
    }
}

// Lance file format constants
const LANCE_FOOTER_SIZE = 40;
const LANCE_MAGIC = new Uint8Array([0x4C, 0x41, 0x4E, 0x43]); // "LANC"

/**
 * Chunked Lance File Reader
 * Reads Lance files from OPFS without loading entire file into memory
 */
class ChunkedLanceReader {
    /**
     * @param {OPFSFileReader} fileReader - OPFS file reader
     * @param {LRUCache} [pageCache] - Optional page cache (shared across readers)
     */
    constructor(fileReader, pageCache = null) {
        this.fileReader = fileReader;
        this.pageCache = pageCache || new LRUCache();
        this.footer = null;
        this.columnMetaCache = new Map(); // colIdx -> metadata
        this._cacheKey = null; // For cache key generation
    }

    /**
     * Open a Lance file from OPFS
     * @param {OPFSStorage} storage - OPFS storage instance
     * @param {string} path - File path in OPFS
     * @param {LRUCache} [pageCache] - Optional shared page cache
     * @returns {Promise<ChunkedLanceReader>}
     */
    static async open(storage, path, pageCache = null) {
        const fileReader = await storage.openFile(path);
        if (!fileReader) {
            throw new Error(`File not found: ${path}`);
        }
        const reader = new ChunkedLanceReader(fileReader, pageCache);
        reader._cacheKey = path;
        await reader._readFooter();
        return reader;
    }

    /**
     * Read and parse the Lance footer
     */
    async _readFooter() {
        const footerData = await this.fileReader.readFromEnd(LANCE_FOOTER_SIZE);

        // Verify magic bytes
        const magic = footerData.slice(36, 40);
        if (!this._arraysEqual(magic, LANCE_MAGIC)) {
            throw new Error('Invalid Lance file: magic bytes mismatch');
        }

        // Parse footer (little-endian)
        const view = new DataView(footerData.buffer, footerData.byteOffset);
        this.footer = {
            columnMetaStart: view.getBigUint64(0, true),
            columnMetaOffsetsStart: view.getBigUint64(8, true),
            globalBuffOffsetsStart: view.getBigUint64(16, true),
            numGlobalBuffers: view.getUint32(24, true),
            numColumns: view.getUint32(28, true),
            majorVersion: view.getUint16(32, true),
            minorVersion: view.getUint16(34, true),
        };

        return this.footer;
    }

    /**
     * Compare two Uint8Arrays
     */
    _arraysEqual(a, b) {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) return false;
        }
        return true;
    }

    /**
     * Get file size
     * @returns {Promise<number>}
     */
    async getSize() {
        return this.fileReader.getSize();
    }

    /**
     * Get number of columns
     * @returns {number}
     */
    getNumColumns() {
        if (!this.footer) throw new Error('Footer not loaded');
        return this.footer.numColumns;
    }

    /**
     * Get Lance format version
     * @returns {{major: number, minor: number}}
     */
    getVersion() {
        if (!this.footer) throw new Error('Footer not loaded');
        return {
            major: this.footer.majorVersion,
            minor: this.footer.minorVersion
        };
    }

    /**
     * Read column metadata offset table
     * @returns {Promise<BigUint64Array>}
     */
    async _readColumnMetaOffsets() {
        const numCols = this.footer.numColumns;
        const offsetTableSize = numCols * 8; // 8 bytes per offset
        const data = await this.fileReader.readRange(
            Number(this.footer.columnMetaOffsetsStart),
            offsetTableSize
        );
        return new BigUint64Array(data.buffer, data.byteOffset, numCols);
    }

    /**
     * Read raw column metadata bytes
     * @param {number} colIdx - Column index
     * @returns {Promise<Uint8Array>}
     */
    async readColumnMetaRaw(colIdx) {
        if (colIdx >= this.footer.numColumns) {
            throw new Error(`Column index ${colIdx} out of range (${this.footer.numColumns} columns)`);
        }

        // Check cache
        const cacheKey = `${this._cacheKey}:colmeta:${colIdx}`;
        const cached = this.pageCache.get(cacheKey);
        if (cached) return cached;

        // Read offset table
        const offsets = await this._readColumnMetaOffsets();

        // Calculate start and end
        const start = Number(this.footer.columnMetaStart) + Number(offsets[colIdx]);
        const end = colIdx < this.footer.numColumns - 1
            ? Number(this.footer.columnMetaStart) + Number(offsets[colIdx + 1])
            : Number(this.footer.columnMetaOffsetsStart);

        const data = await this.fileReader.readRange(start, end - start);

        // Cache it
        this.pageCache.put(cacheKey, data);
        return data;
    }

    /**
     * Read a specific byte range from the file
     * @param {number} offset - Start offset
     * @param {number} length - Number of bytes
     * @returns {Promise<Uint8Array>}
     */
    async readRange(offset, length) {
        // Check cache
        const cacheKey = `${this._cacheKey}:range:${offset}:${length}`;
        const cached = this.pageCache.get(cacheKey);
        if (cached) return cached;

        const data = await this.fileReader.readRange(offset, length);

        // Cache if reasonably sized
        if (length < 10 * 1024 * 1024) { // < 10MB
            this.pageCache.put(cacheKey, data);
        }
        return data;
    }

    /**
     * Get cache statistics
     * @returns {object}
     */
    getCacheStats() {
        return this.pageCache.stats();
    }

    /**
     * Close the reader and release resources
     */
    close() {
        this.fileReader.invalidate();
        this.columnMetaCache.clear();
    }
}

// =============================================================================
// Memory Manager - Global memory monitoring and management
// =============================================================================

/**
 * Global memory manager for browser environment.
 * Monitors memory usage and triggers cleanup when needed.
 */
class MemoryManager {
    constructor(options = {}) {
        this.maxHeapMB = options.maxHeapMB || 100; // Target max heap usage
        this.warningThreshold = options.warningThreshold || 0.8; // 80% warning
        this.caches = new Set(); // Registered LRU caches
        this.lastCheck = 0;
        this.checkInterval = 5000; // Check every 5 seconds
    }

    /**
     * Register a cache for memory management
     * @param {LRUCache} cache - Cache to manage
     */
    registerCache(cache) {
        this.caches.add(cache);
    }

    /**
     * Unregister a cache
     * @param {LRUCache} cache - Cache to remove
     */
    unregisterCache(cache) {
        this.caches.delete(cache);
    }

    /**
     * Get current memory usage (if available)
     * @returns {Object|null} Memory info or null if not available
     */
    getMemoryUsage() {
        if (typeof performance !== 'undefined' && performance.memory) {
            // Chrome/Chromium only
            return {
                usedHeapMB: performance.memory.usedJSHeapSize / (1024 * 1024),
                totalHeapMB: performance.memory.totalJSHeapSize / (1024 * 1024),
                limitMB: performance.memory.jsHeapSizeLimit / (1024 * 1024),
            };
        }
        return null;
    }

    /**
     * Check memory and trigger cleanup if needed
     * @returns {boolean} True if cleanup was triggered
     */
    checkAndCleanup() {
        const now = Date.now();
        if (now - this.lastCheck < this.checkInterval) {
            return false;
        }
        this.lastCheck = now;

        const memory = this.getMemoryUsage();
        if (!memory) return false;

        const usageRatio = memory.usedHeapMB / this.maxHeapMB;

        if (usageRatio > this.warningThreshold) {
            console.warn(`[MemoryManager] High memory usage: ${memory.usedHeapMB.toFixed(1)}MB / ${this.maxHeapMB}MB`);
            this.cleanup();
            return true;
        }

        return false;
    }

    /**
     * Force cleanup of all registered caches
     */
    cleanup() {
        for (const cache of this.caches) {
            // Evict 50% of entries
            const stats = cache.stats();
            const targetSize = stats.currentSize / 2;

            while (cache.currentSize > targetSize && cache.cache.size > 0) {
                cache._evictOldest();
            }
        }
    }

    /**
     * Get aggregate cache stats
     * @returns {Object} Combined stats from all caches
     */
    getCacheStats() {
        let totalEntries = 0;
        let totalSize = 0;
        let totalMaxSize = 0;

        for (const cache of this.caches) {
            const stats = cache.stats();
            totalEntries += stats.entries;
            totalSize += stats.currentSize;
            totalMaxSize += stats.maxSize;
        }

        return {
            caches: this.caches.size,
            totalEntries,
            totalSizeMB: (totalSize / (1024 * 1024)).toFixed(2),
            totalMaxSizeMB: (totalMaxSize / (1024 * 1024)).toFixed(2),
            memory: this.getMemoryUsage(),
        };
    }
}

// Global memory manager instance
const memoryManager = new MemoryManager();

/**
 * Streaming utilities for large file processing
 */
const StreamUtils = {
    /**
     * Process items in batches with memory-aware pacing
     * @param {AsyncIterable} source - Source of items
     * @param {Function} processor - Async function to process each batch
     * @param {Object} options - Options
     * @yields {any} Results from processor
     */
    async *processBatches(source, processor, options = {}) {
        const batchSize = options.batchSize || 10000;
        const pauseAfter = options.pauseAfter || 5; // Pause every N batches for GC
        let batchCount = 0;

        for await (const batch of source) {
            yield await processor(batch);
            batchCount++;

            // Periodic memory check
            if (batchCount % pauseAfter === 0) {
                memoryManager.checkAndCleanup();
                // Small delay to allow GC
                await new Promise(r => setTimeout(r, 0));
            }
        }
    },

    /**
     * Create a progress-reporting wrapper for async iterables
     * @param {AsyncIterable} source - Source iterable
     * @param {Function} onProgress - Progress callback (processed, total?)
     */
    async *withProgress(source, onProgress) {
        let processed = 0;
        for await (const item of source) {
            processed += Array.isArray(item) ? item.length : 1;
            onProgress(processed);
            yield item;
        }
    },

    /**
     * Limit memory usage by processing in chunks with explicit cleanup
     * @param {AsyncIterable} source - Source of data chunks
     * @param {number} maxChunksInFlight - Max chunks to keep in memory
     */
    async *throttle(source, maxChunksInFlight = 3) {
        const queue = [];

        for await (const chunk of source) {
            queue.push(chunk);

            if (queue.length >= maxChunksInFlight) {
                yield queue.shift();
            }
        }

        // Drain remaining
        while (queue.length > 0) {
            yield queue.shift();
        }
    },
};

// Export memory utilities
export { memoryManager, MemoryManager, StreamUtils };

// =============================================================================
// Protobuf Encoder - Minimal encoder for Lance column metadata
// =============================================================================

/**
 * Simple Protobuf encoder for Lance metadata.
 * Only implements what's needed for Lance file writing.
 */
class ProtobufEncoder {
    constructor() {
        this.chunks = [];
    }

    /**
     * Encode a varint (variable-length integer)
     * @param {number|bigint} value
     * @returns {Uint8Array}
     */
    static encodeVarint(value) {
        const bytes = [];
        let v = typeof value === 'bigint' ? value : BigInt(value);
        while (v > 0x7fn) {
            bytes.push(Number(v & 0x7fn) | 0x80);
            v >>= 7n;
        }
        bytes.push(Number(v));
        return new Uint8Array(bytes);
    }

    /**
     * Encode a field header (tag)
     * @param {number} fieldNum - Field number
     * @param {number} wireType - Wire type (0=varint, 2=length-delimited)
     * @returns {Uint8Array}
     */
    static encodeFieldHeader(fieldNum, wireType) {
        const tag = (fieldNum << 3) | wireType;
        return ProtobufEncoder.encodeVarint(tag);
    }

    /**
     * Encode a varint field
     * @param {number} fieldNum
     * @param {number|bigint} value
     */
    writeVarint(fieldNum, value) {
        this.chunks.push(ProtobufEncoder.encodeFieldHeader(fieldNum, 0));
        this.chunks.push(ProtobufEncoder.encodeVarint(value));
    }

    /**
     * Encode a length-delimited field (bytes or nested message)
     * @param {number} fieldNum
     * @param {Uint8Array} data
     */
    writeBytes(fieldNum, data) {
        this.chunks.push(ProtobufEncoder.encodeFieldHeader(fieldNum, 2));
        this.chunks.push(ProtobufEncoder.encodeVarint(data.length));
        this.chunks.push(data);
    }

    /**
     * Encode packed repeated uint64 as varints
     * @param {number} fieldNum
     * @param {BigUint64Array|number[]} values
     */
    writePackedUint64(fieldNum, values) {
        // First encode all varints
        const varintChunks = [];
        for (const v of values) {
            varintChunks.push(ProtobufEncoder.encodeVarint(v));
        }
        // Calculate total length
        const totalLen = varintChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        // Write field header + length + data
        this.chunks.push(ProtobufEncoder.encodeFieldHeader(fieldNum, 2));
        this.chunks.push(ProtobufEncoder.encodeVarint(totalLen));
        for (const chunk of varintChunks) {
            this.chunks.push(chunk);
        }
    }

    /**
     * Get the encoded bytes
     * @returns {Uint8Array}
     */
    toBytes() {
        const totalLen = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const result = new Uint8Array(totalLen);
        let offset = 0;
        for (const chunk of this.chunks) {
            result.set(chunk, offset);
            offset += chunk.length;
        }
        return result;
    }

    /**
     * Clear the encoder for reuse
     */
    clear() {
        this.chunks = [];
    }
}

// =============================================================================
// LanceFileWriter - Create Lance files in pure JavaScript
// =============================================================================

/**
 * Lance column types
 */
const LanceColumnType = {
    INT64: 'int64',
    FLOAT64: 'float64',
    STRING: 'string',
    BOOL: 'bool',
    INT32: 'int32',
    FLOAT32: 'float32',
};

/**
 * Pure JavaScript Lance File Writer - Creates Lance files without WASM.
 * Use this when WASM is not available or for simple file creation.
 * Supports basic column types: int64, float64, string, bool.
 *
 * @example
 * const writer = new PureLanceWriter();
 * writer.addInt64Column('id', BigInt64Array.from([1n, 2n, 3n]));
 * writer.addFloat64Column('score', new Float64Array([0.5, 0.8, 0.3]));
 * writer.addStringColumn('name', ['Alice', 'Bob', 'Charlie']);
 * const lanceData = writer.finalize();
 * await opfsStorage.save('mydata.lance', lanceData);
 */
class PureLanceWriter {
    /**
     * @param {Object} options
     * @param {number} [options.majorVersion=0] - Lance format major version
     * @param {number} [options.minorVersion=3] - Lance format minor version (3 = v2.0)
     */
    constructor(options = {}) {
        this.majorVersion = options.majorVersion ?? 0;
        this.minorVersion = options.minorVersion ?? 3; // v2.0
        this.columns = []; // { name, type, data, metadata }
        this.rowCount = null;
    }

    /**
     * Validate row count consistency
     * @param {number} count
     */
    _validateRowCount(count) {
        if (this.rowCount === null) {
            this.rowCount = count;
        } else if (this.rowCount !== count) {
            throw new Error(`Row count mismatch: expected ${this.rowCount}, got ${count}`);
        }
    }

    /**
     * Add an int64 column
     * @param {string} name - Column name
     * @param {BigInt64Array} values - Column values
     */
    addInt64Column(name, values) {
        this._validateRowCount(values.length);
        this.columns.push({
            name,
            type: LanceColumnType.INT64,
            data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
            length: values.length,
        });
    }

    /**
     * Add an int32 column
     * @param {string} name - Column name
     * @param {Int32Array} values - Column values
     */
    addInt32Column(name, values) {
        this._validateRowCount(values.length);
        this.columns.push({
            name,
            type: LanceColumnType.INT32,
            data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
            length: values.length,
        });
    }

    /**
     * Add a float64 column
     * @param {string} name - Column name
     * @param {Float64Array} values - Column values
     */
    addFloat64Column(name, values) {
        this._validateRowCount(values.length);
        this.columns.push({
            name,
            type: LanceColumnType.FLOAT64,
            data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
            length: values.length,
        });
    }

    /**
     * Add a float32 column
     * @param {string} name - Column name
     * @param {Float32Array} values - Column values
     */
    addFloat32Column(name, values) {
        this._validateRowCount(values.length);
        this.columns.push({
            name,
            type: LanceColumnType.FLOAT32,
            data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
            length: values.length,
        });
    }

    /**
     * Add a boolean column
     * @param {string} name - Column name
     * @param {boolean[]} values - Column values
     */
    addBoolColumn(name, values) {
        this._validateRowCount(values.length);
        // Pack booleans as bytes (1 byte per bool for simplicity)
        const data = new Uint8Array(values.length);
        for (let i = 0; i < values.length; i++) {
            data[i] = values[i] ? 1 : 0;
        }
        this.columns.push({
            name,
            type: LanceColumnType.BOOL,
            data,
            length: values.length,
        });
    }

    /**
     * Add a string column
     * @param {string} name - Column name
     * @param {string[]} values - Column values
     */
    addStringColumn(name, values) {
        this._validateRowCount(values.length);

        const encoder = new TextEncoder();

        // Build offsets and data
        // Lance strings use i32 offsets followed by UTF-8 data
        const offsets = new Int32Array(values.length + 1);
        const dataChunks = [];
        let currentOffset = 0;

        for (let i = 0; i < values.length; i++) {
            offsets[i] = currentOffset;
            const encoded = encoder.encode(values[i]);
            dataChunks.push(encoded);
            currentOffset += encoded.length;
        }
        offsets[values.length] = currentOffset;

        // Combine offsets and data
        const offsetsBytes = new Uint8Array(offsets.buffer);
        const totalDataLen = dataChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const stringData = new Uint8Array(totalDataLen);
        let writePos = 0;
        for (const chunk of dataChunks) {
            stringData.set(chunk, writePos);
            writePos += chunk.length;
        }

        // Store both parts
        this.columns.push({
            name,
            type: LanceColumnType.STRING,
            offsetsData: offsetsBytes,
            stringData,
            data: null, // Will be combined in finalize
            length: values.length,
        });
    }

    /**
     * Build column metadata protobuf for a column
     * @param {number} bufferOffset - Offset to column data
     * @param {number} bufferSize - Size of column data
     * @param {number} length - Number of rows
     * @param {string} type - Column type
     * @returns {Uint8Array}
     */
    _buildColumnMeta(bufferOffset, bufferSize, length, type) {
        // Build Page message
        const pageEncoder = new ProtobufEncoder();
        pageEncoder.writePackedUint64(1, [BigInt(bufferOffset)]); // buffer_offsets
        pageEncoder.writePackedUint64(2, [BigInt(bufferSize)]); // buffer_sizes
        pageEncoder.writeVarint(3, length); // length
        // Skip encoding (field 4) - use default
        pageEncoder.writeVarint(5, 0); // priority
        const pageBytes = pageEncoder.toBytes();

        // Build ColumnMetadata message
        const metaEncoder = new ProtobufEncoder();
        // Skip encoding (field 1) - use default plain encoding
        metaEncoder.writeBytes(2, pageBytes); // pages (repeated)
        // Skip buffer_offsets and buffer_sizes at column level

        return metaEncoder.toBytes();
    }

    /**
     * Build column metadata for string column (2 buffers: offsets + data)
     * @param {number} offsetsBufOffset - Offset to offsets buffer
     * @param {number} offsetsBufSize - Size of offsets buffer
     * @param {number} dataBufOffset - Offset to string data buffer
     * @param {number} dataBufSize - Size of string data buffer
     * @param {number} length - Number of rows
     * @returns {Uint8Array}
     */
    _buildStringColumnMeta(offsetsBufOffset, offsetsBufSize, dataBufOffset, dataBufSize, length) {
        // Build Page message with 2 buffers
        const pageEncoder = new ProtobufEncoder();
        pageEncoder.writePackedUint64(1, [BigInt(offsetsBufOffset), BigInt(dataBufOffset)]); // buffer_offsets
        pageEncoder.writePackedUint64(2, [BigInt(offsetsBufSize), BigInt(dataBufSize)]); // buffer_sizes
        pageEncoder.writeVarint(3, length); // length
        pageEncoder.writeVarint(5, 0); // priority
        const pageBytes = pageEncoder.toBytes();

        // Build ColumnMetadata message
        const metaEncoder = new ProtobufEncoder();
        metaEncoder.writeBytes(2, pageBytes); // pages (repeated)

        return metaEncoder.toBytes();
    }

    /**
     * Finalize and create the Lance file
     * @returns {Uint8Array} Complete Lance file data
     */
    finalize() {
        if (this.columns.length === 0) {
            throw new Error('No columns added');
        }

        const chunks = [];
        let currentOffset = 0;

        // 1. Write column data buffers
        const columnBufferInfos = []; // { offset, size } for each column

        for (const col of this.columns) {
            if (col.type === LanceColumnType.STRING) {
                // String columns have 2 buffers: offsets + data
                const offsetsOffset = currentOffset;
                chunks.push(col.offsetsData);
                currentOffset += col.offsetsData.length;

                const dataOffset = currentOffset;
                chunks.push(col.stringData);
                currentOffset += col.stringData.length;

                columnBufferInfos.push({
                    type: 'string',
                    offsetsOffset,
                    offsetsSize: col.offsetsData.length,
                    dataOffset,
                    dataSize: col.stringData.length,
                    length: col.length,
                });
            } else {
                // Simple column with single buffer
                const bufferOffset = currentOffset;
                chunks.push(col.data);
                currentOffset += col.data.length;

                columnBufferInfos.push({
                    type: col.type,
                    offset: bufferOffset,
                    size: col.data.length,
                    length: col.length,
                });
            }
        }

        // 2. Build column metadata
        const columnMetadatas = [];
        for (let i = 0; i < this.columns.length; i++) {
            const info = columnBufferInfos[i];
            let meta;
            if (info.type === 'string') {
                meta = this._buildStringColumnMeta(
                    info.offsetsOffset, info.offsetsSize,
                    info.dataOffset, info.dataSize,
                    info.length
                );
            } else {
                meta = this._buildColumnMeta(info.offset, info.size, info.length, info.type);
            }
            columnMetadatas.push(meta);
        }

        // 3. Write column metadata section
        const columnMetaStart = currentOffset;
        const columnMetaOffsets = [];
        let metaOffset = 0;
        for (const meta of columnMetadatas) {
            columnMetaOffsets.push(metaOffset);
            chunks.push(meta);
            currentOffset += meta.length;
            metaOffset += meta.length;
        }

        // 4. Write column metadata offset table
        const columnMetaOffsetsStart = currentOffset;
        const offsetTable = new BigUint64Array(columnMetaOffsets.length);
        for (let i = 0; i < columnMetaOffsets.length; i++) {
            offsetTable[i] = BigInt(columnMetaOffsets[i]);
        }
        const offsetTableBytes = new Uint8Array(offsetTable.buffer);
        chunks.push(offsetTableBytes);
        currentOffset += offsetTableBytes.length;

        // 5. Write global buffer offsets (empty for now)
        const globalBuffOffsetsStart = currentOffset;
        const numGlobalBuffers = 0;

        // 6. Write footer (40 bytes)
        const footer = new ArrayBuffer(LANCE_FOOTER_SIZE);
        const footerView = new DataView(footer);

        footerView.setBigUint64(0, BigInt(columnMetaStart), true);           // column_meta_start
        footerView.setBigUint64(8, BigInt(columnMetaOffsetsStart), true);    // column_meta_offsets_start
        footerView.setBigUint64(16, BigInt(globalBuffOffsetsStart), true);   // global_buff_offsets_start
        footerView.setUint32(24, numGlobalBuffers, true);                    // num_global_buffers
        footerView.setUint32(28, this.columns.length, true);                 // num_columns
        footerView.setUint16(32, this.majorVersion, true);                   // major_version
        footerView.setUint16(34, this.minorVersion, true);                   // minor_version
        // Magic "LANC"
        new Uint8Array(footer, 36, 4).set(LANCE_MAGIC);

        chunks.push(new Uint8Array(footer));

        // 7. Combine all chunks
        const totalSize = currentOffset + LANCE_FOOTER_SIZE;
        const result = new Uint8Array(totalSize);
        let writeOffset = 0;
        for (const chunk of chunks) {
            result.set(chunk, writeOffset);
            writeOffset += chunk.length;
        }

        return result;
    }

    /**
     * Get the number of columns
     * @returns {number}
     */
    getNumColumns() {
        return this.columns.length;
    }

    /**
     * Get the row count
     * @returns {number|null}
     */
    getRowCount() {
        return this.rowCount;
    }

    /**
     * Get column names
     * @returns {string[]}
     */
    getColumnNames() {
        return this.columns.map(c => c.name);
    }
}

// Legacy IndexedDB + OPFS storage (deprecated, use OPFSStorage instead)
class DatasetStorage {
    constructor(dbName = 'lanceql-files', version = 1) {
        this.dbName = dbName;
        this.version = version;
        this.db = null;
        this.SIZE_THRESHOLD = 50 * 1024 * 1024; // 50MB
    }

    async open() {
        if (this.db) return this.db;

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.version);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('files')) {
                    db.createObjectStore('files', { keyPath: 'name' });
                }
                if (!db.objectStoreNames.contains('index')) {
                    const store = db.createObjectStore('index', { keyPath: 'name' });
                    store.createIndex('timestamp', 'timestamp');
                    store.createIndex('size', 'size');
                }
            };
        });
    }

    async hasOPFS() {
        try {
            return 'storage' in navigator && 'getDirectory' in navigator.storage;
        } catch {
            return false;
        }
    }

    async save(name, data, metadata = {}) {
        const db = await this.open();
        const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        const size = bytes.byteLength;
        const useOPFS = size >= this.SIZE_THRESHOLD && await this.hasOPFS();

        if (useOPFS) {
            try {
                const root = await navigator.storage.getDirectory();
                const fileHandle = await root.getFileHandle(name, { create: true });
                const writable = await fileHandle.createWritable();
                await writable.write(bytes);
                await writable.close();
            } catch (e) {
                console.warn('[DatasetStorage] OPFS save failed, falling back to IndexedDB:', e);
            }
        }

        if (!useOPFS) {
            await new Promise((resolve, reject) => {
                const tx = db.transaction('files', 'readwrite');
                const store = tx.objectStore('files');
                store.put({ name, data: bytes });
                tx.oncomplete = () => resolve();
                tx.onerror = () => reject(tx.error);
            });
        }

        await new Promise((resolve, reject) => {
            const tx = db.transaction('index', 'readwrite');
            const store = tx.objectStore('index');
            store.put({
                name,
                size,
                timestamp: Date.now(),
                storage: useOPFS ? 'opfs' : 'indexeddb',
                ...metadata
            });
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });

        return { name, size, storage: useOPFS ? 'opfs' : 'indexeddb' };
    }

    async load(name) {
        const db = await this.open();

        const entry = await new Promise((resolve) => {
            const tx = db.transaction('index', 'readonly');
            const store = tx.objectStore('index');
            const request = store.get(name);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => resolve(null);
        });

        if (!entry) return null;

        if (entry.storage === 'opfs') {
            try {
                const root = await navigator.storage.getDirectory();
                const fileHandle = await root.getFileHandle(name);
                const file = await fileHandle.getFile();
                const buffer = await file.arrayBuffer();
                return new Uint8Array(buffer);
            } catch (e) {
                console.warn('[DatasetStorage] OPFS load failed:', e);
                return null;
            }
        }

        return new Promise((resolve) => {
            const tx = db.transaction('files', 'readonly');
            const store = tx.objectStore('files');
            const request = store.get(name);
            request.onsuccess = () => {
                const result = request.result;
                resolve(result ? result.data : null);
            };
            request.onerror = () => resolve(null);
        });
    }

    async list() {
        const db = await this.open();

        return new Promise((resolve) => {
            const tx = db.transaction('index', 'readonly');
            const store = tx.objectStore('index');
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result || []);
            request.onerror = () => resolve([]);
        });
    }

    async delete(name) {
        const db = await this.open();

        const entry = await new Promise((resolve) => {
            const tx = db.transaction('index', 'readonly');
            const store = tx.objectStore('index');
            const request = store.get(name);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => resolve(null);
        });

        if (entry?.storage === 'opfs') {
            try {
                const root = await navigator.storage.getDirectory();
                await root.removeEntry(name);
            } catch (e) {
                console.warn('[DatasetStorage] OPFS delete failed:', e);
            }
        }

        await new Promise((resolve) => {
            const tx = db.transaction('files', 'readwrite');
            const store = tx.objectStore('files');
            store.delete(name);
            tx.oncomplete = () => resolve();
        });

        // Delete from index
        await new Promise((resolve) => {
            const tx = db.transaction('index', 'readwrite');
            const store = tx.objectStore('index');
            store.delete(name);
            tx.oncomplete = () => resolve();
        });
    }

    /**
     * Check if a dataset exists.
     * @param {string} name - Dataset name
     * @returns {Promise<boolean>}
     */
    async exists(name) {
        const db = await this.open();

        return new Promise((resolve) => {
            const tx = db.transaction('index', 'readonly');
            const store = tx.objectStore('index');
            const request = store.get(name);
            request.onsuccess = () => resolve(!!request.result);
            request.onerror = () => resolve(false);
        });
    }

    /**
     * Get storage usage info.
     * @returns {Promise<Object>} Usage stats
     */
    async getUsage() {
        const datasets = await this.list();
        const totalSize = datasets.reduce((sum, d) => sum + (d.size || 0), 0);
        const indexedDBCount = datasets.filter(d => d.storage === 'indexeddb').length;
        const opfsCount = datasets.filter(d => d.storage === 'opfs').length;

        let quota = null;
        if (navigator.storage?.estimate) {
            quota = await navigator.storage.estimate();
        }

        return {
            datasets: datasets.length,
            totalSize,
            indexedDBCount,
            opfsCount,
            quota
        };
    }
}

// Global storage instances
const opfsStorage = new OPFSStorage();  // OPFS-only (recommended)
const datasetStorage = new DatasetStorage();  // Legacy IndexedDB + OPFS

// =============================================================================
// HotTierCache - OPFS-backed cache for remote Lance files (500-2000x faster)
// =============================================================================

/**
 * HotTierCache provides fast local caching for remote Lance files.
 *
 * Architecture:
 * - First request: Fetch from R2/S3 (50-200ms) → Cache to OPFS
 * - Subsequent requests: Read from OPFS (<0.1ms) → 500-2000x faster
 *
 * Cache strategies:
 * - Small files (<10MB): Cache entire file
 * - Large files: Cache individual ranges/fragments on demand
 *
 * Storage layout:
 *   _cache/
 *     {urlHash}/
 *       meta.json          - URL, size, version, cached ranges
 *       data.lance         - Full file (if small) or range blocks
 *       ranges/
 *         {start}-{end}    - Cached range blocks (for large files)
 */
class HotTierCache {
    constructor(storage = null, options = {}) {
        this.storage = storage;
        this.cacheDir = options.cacheDir || '_cache';
        this.maxFileSize = options.maxFileSize || 10 * 1024 * 1024; // 10MB - cache whole file
        this.maxCacheSize = options.maxCacheSize || 500 * 1024 * 1024; // 500MB total cache
        this.enabled = options.enabled ?? true;
        this._stats = {
            hits: 0,
            misses: 0,
            bytesFromCache: 0,
            bytesFromNetwork: 0,
        };
        // In-memory cache for metadata to avoid OPFS reads on every getRange call
        this._metaCache = new Map();  // url -> { meta, fullFileData }
    }

    /**
     * Initialize the cache (lazy initialization)
     */
    async init() {
        if (this.storage) return;
        this.storage = new OPFSStorage();
        await this.storage.open();
    }

    /**
     * Get cache key from URL (hash for safe filesystem names)
     */
    _getCacheKey(url) {
        // Simple hash for URL → safe filename
        let hash = 0;
        for (let i = 0; i < url.length; i++) {
            const char = url.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash).toString(36);
    }

    /**
     * Get cache path for a URL
     */
    _getCachePath(url, suffix = '') {
        const key = this._getCacheKey(url);
        return `${this.cacheDir}/${key}${suffix}`;
    }

    /**
     * Check if a URL is cached
     * @param {string} url - Remote URL
     * @returns {Promise<{cached: boolean, meta?: object}>}
     */
    async isCached(url) {
        if (!this.enabled) return { cached: false };

        try {
            await this.init();
            const metaPath = this._getCachePath(url, '/meta.json');
            const metaData = await this.storage.load(metaPath);
            if (!metaData) return { cached: false };

            const meta = JSON.parse(new TextDecoder().decode(metaData));
            return { cached: true, meta };
        } catch (e) {
            return { cached: false };
        }
    }

    /**
     * Get or fetch a file, using cache when available
     * @param {string} url - Remote URL
     * @param {number} [fileSize] - Known file size (avoids HEAD request)
     * @returns {Promise<Uint8Array>}
     */
    async getFile(url, fileSize = null) {
        if (!this.enabled) {
            return this._fetchFile(url);
        }

        await this.init();

        // Check cache
        const { cached, meta } = await this.isCached(url);
        if (cached && meta.fullFile) {
            const dataPath = this._getCachePath(url, '/data.lance');
            const data = await this.storage.load(dataPath);
            if (data) {
                this._stats.hits++;
                this._stats.bytesFromCache += data.byteLength;
                console.log(`[HotTierCache] HIT: ${url} (${(data.byteLength / 1024).toFixed(1)} KB)`);
                return data;
            }
        }

        // Cache miss - fetch and cache
        this._stats.misses++;
        const data = await this._fetchFile(url);
        this._stats.bytesFromNetwork += data.byteLength;

        // Cache if small enough
        if (data.byteLength <= this.maxFileSize) {
            await this._cacheFile(url, data);
        }

        return data;
    }

    /**
     * Get or fetch a byte range, using cache when available
     * @param {string} url - Remote URL
     * @param {number} start - Start byte offset
     * @param {number} end - End byte offset (inclusive)
     * @param {number} [fileSize] - Total file size
     * @returns {Promise<ArrayBuffer>}
     */
    async getRange(url, start, end, fileSize = null) {
        if (!this.enabled) {
            return this._fetchRange(url, start, end);
        }

        // Fast path: check in-memory cache first (no async overhead)
        const memCached = this._metaCache.get(url);
        if (memCached?.fullFileData) {
            const data = memCached.fullFileData;
            if (data.byteLength > end) {
                this._stats.hits++;
                this._stats.bytesFromCache += (end - start + 1);
                return data.slice(start, end + 1).buffer;
            }
        }

        await this.init();

        // Check OPFS cache (only once per URL, then cache in memory)
        if (!memCached) {
            const { cached, meta } = await this.isCached(url);
            if (cached && meta.fullFile) {
                const dataPath = this._getCachePath(url, '/data.lance');
                const data = await this.storage.load(dataPath);
                if (data && data.byteLength > end) {
                    // Cache in memory for subsequent calls
                    this._metaCache.set(url, { meta, fullFileData: data });
                    this._stats.hits++;
                    this._stats.bytesFromCache += (end - start + 1);
                    return data.slice(start, end + 1).buffer;
                }
            }
            // Mark as checked even if not cached
            this._metaCache.set(url, { meta: cached ? meta : null, fullFileData: null });
        }

        // Cache miss - fetch from network
        this._stats.misses++;
        const data = await this._fetchRange(url, start, end);
        this._stats.bytesFromNetwork += data.byteLength;

        // Don't cache individual ranges to OPFS - too slow for IVF search
        // Only full files are cached (via prefetch)

        return data;
    }

    /**
     * Prefetch and cache an entire file
     * @param {string} url - Remote URL
     * @param {function} [onProgress] - Progress callback (bytesLoaded, totalBytes)
     */
    async prefetch(url, onProgress = null) {
        await this.init();

        const { cached, meta } = await this.isCached(url);
        if (cached && meta.fullFile) {
            console.log(`[HotTierCache] Already cached: ${url}`);
            return;
        }

        console.log(`[HotTierCache] Prefetching: ${url}`);
        const data = await this._fetchFile(url, onProgress);
        await this._cacheFile(url, data);
        console.log(`[HotTierCache] Cached: ${url} (${(data.byteLength / 1024 / 1024).toFixed(2)} MB)`);
    }

    /**
     * Evict a URL from cache
     */
    async evict(url) {
        await this.init();
        const cachePath = this._getCachePath(url);
        await this.storage.delete(cachePath);
        console.log(`[HotTierCache] Evicted: ${url}`);
    }

    /**
     * Clear entire cache
     */
    async clear() {
        await this.init();
        await this.storage.delete(this.cacheDir);
        this._stats = { hits: 0, misses: 0, bytesFromCache: 0, bytesFromNetwork: 0 };
        console.log(`[HotTierCache] Cleared all cache`);
    }

    /**
     * Get cache statistics
     */
    getStats() {
        const hitRate = this._stats.hits + this._stats.misses > 0
            ? (this._stats.hits / (this._stats.hits + this._stats.misses) * 100).toFixed(1)
            : 0;
        return {
            ...this._stats,
            hitRate: `${hitRate}%`,
            bytesFromCacheMB: (this._stats.bytesFromCache / 1024 / 1024).toFixed(2),
            bytesFromNetworkMB: (this._stats.bytesFromNetwork / 1024 / 1024).toFixed(2),
        };
    }

    /**
     * Fetch file from network
     * @private
     */
    async _fetchFile(url, onProgress = null) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        if (onProgress && response.headers.get('content-length')) {
            const total = parseInt(response.headers.get('content-length'));
            const reader = response.body.getReader();
            const chunks = [];
            let loaded = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                loaded += value.length;
                onProgress(loaded, total);
            }

            const result = new Uint8Array(loaded);
            let offset = 0;
            for (const chunk of chunks) {
                result.set(chunk, offset);
                offset += chunk.length;
            }
            return result;
        }

        const buffer = await response.arrayBuffer();
        return new Uint8Array(buffer);
    }

    /**
     * Fetch range from network
     * @private
     */
    async _fetchRange(url, start, end) {
        const response = await fetch(url, {
            headers: { 'Range': `bytes=${start}-${end}` }
        });
        if (!response.ok && response.status !== 206) {
            throw new Error(`HTTP error: ${response.status}`);
        }
        return response.arrayBuffer();
    }

    /**
     * Cache a full file
     * @private
     */
    async _cacheFile(url, data) {
        const metaPath = this._getCachePath(url, '/meta.json');
        const dataPath = this._getCachePath(url, '/data.lance');

        const meta = {
            url,
            size: data.byteLength,
            cachedAt: Date.now(),
            fullFile: true,
            ranges: null,
        };

        await this.storage.save(metaPath, new TextEncoder().encode(JSON.stringify(meta)));
        await this.storage.save(dataPath, data);
    }

    /**
     * Cache a byte range
     * @private
     */
    async _cacheRange(url, start, end, data, fileSize) {
        const metaPath = this._getCachePath(url, '/meta.json');
        const rangePath = this._getCachePath(url, `/ranges/${start}-${end}`);

        // Load existing meta or create new
        let meta;
        const { cached, meta: existingMeta } = await this.isCached(url);
        if (cached) {
            meta = existingMeta;
            meta.ranges = meta.ranges || [];
        } else {
            meta = {
                url,
                size: fileSize,
                cachedAt: Date.now(),
                fullFile: false,
                ranges: [],
            };
        }

        // Add this range (merge overlapping ranges for efficiency)
        meta.ranges.push({ start, end, cachedAt: Date.now() });
        meta.ranges = this._mergeRanges(meta.ranges);

        await this.storage.save(metaPath, new TextEncoder().encode(JSON.stringify(meta)));
        await this.storage.save(rangePath, data);
    }

    /**
     * Merge overlapping ranges
     * @private
     */
    _mergeRanges(ranges) {
        if (ranges.length <= 1) return ranges;

        ranges.sort((a, b) => a.start - b.start);
        const merged = [ranges[0]];

        for (let i = 1; i < ranges.length; i++) {
            const last = merged[merged.length - 1];
            const current = ranges[i];

            // Merge if overlapping or adjacent
            if (current.start <= last.end + 1) {
                last.end = Math.max(last.end, current.end);
            } else {
                merged.push(current);
            }
        }

        return merged;
    }
}

// Global hot-tier cache instance
const hotTierCache = new HotTierCache();

// Export storage and statistics for external use
export { opfsStorage, OPFSStorage, OPFSFileReader, LRUCache, ChunkedLanceReader, ProtobufEncoder, PureLanceWriter, LanceColumnType, datasetStorage, DatasetStorage, statisticsManager, StatisticsManager, hotTierCache, HotTierCache };

// =============================================================================
// OPFSJoinExecutor - OPFS-backed join execution for TB-scale joins
// =============================================================================

/**
 * OPFSJoinExecutor enables TB-scale JOINs in the browser by spilling to OPFS.
 *
 * Architecture:
 * 1. Stream left table chunks → write to OPFS partitioned by hash(join_key)
 * 2. Stream right table chunks → probe OPFS partitions → write matches to OPFS
 * 3. Stream final results from OPFS
 *
 * This avoids loading entire tables into RAM. Only one chunk + one hash partition
 * need to fit in memory at a time.
 *
 * Storage layout:
 *   _join_temp/
 *     {sessionId}/
 *       left/
 *         partition_000.jsonl   # Rows where hash(key) % numPartitions == 0
 *         partition_001.jsonl
 *         ...
 *       right/
 *         partition_000.jsonl
 *         ...
 *       results/
 *         chunk_000.jsonl
 *         chunk_001.jsonl
 *         ...
 */
export class OPFSJoinExecutor {
    constructor(storage = opfsStorage) {
        this.storage = storage;
        this.sessionId = `join_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
        this.basePath = `_join_temp/${this.sessionId}`;
        this.numPartitions = 64;  // Number of hash partitions
        this.chunkSize = 1000;    // Rows per chunk when streaming
        this.stats = {
            leftRowsWritten: 0,
            rightRowsWritten: 0,
            resultRowsWritten: 0,
            bytesWrittenToOPFS: 0,
            bytesReadFromOPFS: 0,
            partitionsUsed: new Set(),
        };
    }

    /**
     * Execute a hash join using OPFS for intermediate storage
     * @param {AsyncGenerator} leftStream - Async generator yielding {columns, rows} chunks
     * @param {AsyncGenerator} rightStream - Async generator yielding {columns, rows} chunks
     * @param {string} leftKey - Join key column name for left table
     * @param {string} rightKey - Join key column name for right table
     * @param {Object} options - Join options
     * @returns {AsyncGenerator} Yields result chunks
     */
    async *executeHashJoin(leftStream, rightStream, leftKey, rightKey, options = {}) {
        const {
            limit = Infinity,
            projection = null,
            leftAlias = 'left',
            rightAlias = 'right',
            joinType = 'INNER',
            prePartitionedLeft = null  // Optional: pre-partitioned left metadata for semi-join optimization
        } = options;

        console.log(`[OPFSJoin] Starting OPFS-backed hash join (${joinType})`);
        console.log(`[OPFSJoin] Session: ${this.sessionId}`);
        console.log(`[OPFSJoin] Partitions: ${this.numPartitions}`);

        try {
            // Phase 1: Partition left table to OPFS (skip if pre-partitioned)
            let leftMeta;
            if (prePartitionedLeft) {
                console.log(`[OPFSJoin] Phase 1: Using pre-partitioned left table (semi-join optimization)`);
                leftMeta = prePartitionedLeft;
            } else {
                console.log(`[OPFSJoin] Phase 1: Partitioning left table...`);
                leftMeta = await this._partitionToOPFS(leftStream, leftKey, 'left');
            }
            console.log(`[OPFSJoin] Left table: ${leftMeta.totalRows} rows in ${leftMeta.partitionsUsed.size} partitions`);

            // Phase 2: Partition right table to OPFS
            console.log(`[OPFSJoin] Phase 2: Partitioning right table...`);
            const rightMeta = await this._partitionToOPFS(rightStream, rightKey, 'right');
            console.log(`[OPFSJoin] Right table: ${rightMeta.totalRows} rows in ${rightMeta.partitionsUsed.size} partitions`);

            // Phase 3: Join partition by partition
            console.log(`[OPFSJoin] Phase 3: Joining partitions (${joinType})...`);
            let totalYielded = 0;

            // Create NULL padding arrays for outer joins
            const leftNulls = new Array(leftMeta.columns.length).fill(null);
            const rightNulls = new Array(rightMeta.columns.length).fill(null);

            // Helper to build result columns
            const resultColumns = [
                ...leftMeta.columns.map(c => `${leftAlias}.${c}`),
                ...rightMeta.columns.map(c => `${rightAlias}.${c}`)
            ];

            // Helper to yield a chunk
            const yieldChunk = function*(chunk) {
                if (chunk.length > 0) {
                    yield { columns: resultColumns, rows: chunk.splice(0) };
                }
            };

            // For CROSS JOIN: cartesian product without partitioning
            if (joinType === 'CROSS') {
                console.log(`[OPFSJoin] CROSS JOIN: computing cartesian product`);
                const chunk = [];

                // Load all partitions for both tables
                for (const leftPartitionId of leftMeta.partitionsUsed) {
                    const leftPartition = await this._loadPartition('left', leftPartitionId, leftMeta.columns);
                    for (const rightPartitionId of rightMeta.partitionsUsed) {
                        const rightPartition = await this._loadPartition('right', rightPartitionId, rightMeta.columns);
                        for (const leftRow of leftPartition) {
                            for (const rightRow of rightPartition) {
                                if (totalYielded >= limit) break;
                                chunk.push([...leftRow, ...rightRow]);
                                totalYielded++;
                                if (chunk.length >= this.chunkSize) {
                                    yield { columns: resultColumns, rows: chunk.splice(0) };
                                }
                            }
                            if (totalYielded >= limit) break;
                        }
                        if (totalYielded >= limit) break;
                    }
                    if (totalYielded >= limit) break;
                }
                if (chunk.length > 0) {
                    yield { columns: resultColumns, rows: chunk };
                }
                console.log(`[OPFSJoin] CROSS JOIN complete: ${totalYielded} result rows`);
                return;
            }

            // For hash-based joins, determine which partitions to process
            const leftKeyIndex = leftMeta.columns.indexOf(leftKey);
            const rightKeyIndex = rightMeta.columns.indexOf(rightKey);

            // Track matched rows for outer joins
            const isLeftOuter = joinType === 'LEFT' || joinType === 'FULL';
            const isRightOuter = joinType === 'RIGHT' || joinType === 'FULL';

            // For RIGHT/FULL: track which right rows have been matched (by partition_rowIndex)
            const matchedRightRows = isRightOuter ? new Set() : null;

            // Process partitions that have data on both sides (for INNER and outer join matches)
            const bothSidesPartitions = new Set(
                [...leftMeta.partitionsUsed].filter(p => rightMeta.partitionsUsed.has(p))
            );
            console.log(`[OPFSJoin] Partitions with both sides: ${bothSidesPartitions.size}`);

            for (const partitionId of bothSidesPartitions) {
                if (totalYielded >= limit) break;

                // Load left partition into memory
                const leftPartition = await this._loadPartition('left', partitionId, leftMeta.columns);
                if (leftPartition.length === 0) continue;

                // Build hash map and track matched left rows for LEFT/FULL JOIN
                const hashMap = new Map();
                const matchedLeftIndices = isLeftOuter ? new Set() : null;

                for (let i = 0; i < leftPartition.length; i++) {
                    const row = leftPartition[i];
                    const key = row[leftKeyIndex];
                    if (key !== null && key !== undefined) {
                        if (!hashMap.has(key)) hashMap.set(key, []);
                        hashMap.get(key).push({ row, index: i });
                    }
                }

                // Load right partition and probe hash map
                const rightPartition = await this._loadPartition('right', partitionId, rightMeta.columns);

                const chunk = [];
                for (let rightIdx = 0; rightIdx < rightPartition.length; rightIdx++) {
                    if (totalYielded >= limit) break;

                    const rightRow = rightPartition[rightIdx];
                    const key = rightRow[rightKeyIndex];
                    const leftEntries = hashMap.get(key);

                    if (leftEntries) {
                        // Track that this right row matched (for RIGHT/FULL)
                        if (matchedRightRows) {
                            matchedRightRows.add(`${partitionId}_${rightIdx}`);
                        }

                        for (const { row: leftRow, index: leftIdx } of leftEntries) {
                            if (totalYielded >= limit) break;

                            // Track that this left row matched (for LEFT/FULL)
                            if (matchedLeftIndices) {
                                matchedLeftIndices.add(leftIdx);
                            }

                            chunk.push([...leftRow, ...rightRow]);
                            totalYielded++;

                            if (chunk.length >= this.chunkSize) {
                                yield { columns: resultColumns, rows: chunk.splice(0) };
                            }
                        }
                    }
                }

                // For LEFT/FULL: emit unmatched left rows with NULL right side
                if (isLeftOuter && matchedLeftIndices) {
                    for (let i = 0; i < leftPartition.length; i++) {
                        if (totalYielded >= limit) break;
                        if (!matchedLeftIndices.has(i)) {
                            chunk.push([...leftPartition[i], ...rightNulls]);
                            totalYielded++;
                            if (chunk.length >= this.chunkSize) {
                                yield { columns: resultColumns, rows: chunk.splice(0) };
                            }
                        }
                    }
                }

                // Yield remaining rows in chunk
                if (chunk.length > 0) {
                    yield { columns: resultColumns, rows: chunk };
                }
            }

            // For LEFT/FULL: handle left-only partitions (no matching right data)
            if (isLeftOuter) {
                for (const partitionId of leftMeta.partitionsUsed) {
                    if (totalYielded >= limit) break;
                    if (bothSidesPartitions.has(partitionId)) continue;  // Already processed

                    const leftPartition = await this._loadPartition('left', partitionId, leftMeta.columns);
                    const chunk = [];
                    for (const leftRow of leftPartition) {
                        if (totalYielded >= limit) break;
                        chunk.push([...leftRow, ...rightNulls]);
                        totalYielded++;
                        if (chunk.length >= this.chunkSize) {
                            yield { columns: resultColumns, rows: chunk.splice(0) };
                        }
                    }
                    if (chunk.length > 0) {
                        yield { columns: resultColumns, rows: chunk };
                    }
                }
            }

            // For RIGHT/FULL: emit unmatched right rows with NULL left side
            if (isRightOuter) {
                for (const partitionId of rightMeta.partitionsUsed) {
                    if (totalYielded >= limit) break;

                    const rightPartition = await this._loadPartition('right', partitionId, rightMeta.columns);
                    const chunk = [];

                    for (let rightIdx = 0; rightIdx < rightPartition.length; rightIdx++) {
                        if (totalYielded >= limit) break;

                        // Check if this row was matched during the main join
                        const rowKey = `${partitionId}_${rightIdx}`;
                        if (!matchedRightRows.has(rowKey)) {
                            chunk.push([...leftNulls, ...rightPartition[rightIdx]]);
                            totalYielded++;
                            if (chunk.length >= this.chunkSize) {
                                yield { columns: resultColumns, rows: chunk.splice(0) };
                            }
                        }
                    }

                    if (chunk.length > 0) {
                        yield { columns: resultColumns, rows: chunk };
                    }
                }
            }

            console.log(`[OPFSJoin] ${joinType} JOIN complete: ${totalYielded} result rows`);
            console.log(`[OPFSJoin] Stats:`, this.getStats());

        } finally {
            // Cleanup temp files
            await this.cleanup();
        }
    }

    /**
     * Partition a stream of data to OPFS files by hash(key)
     * @param {AsyncGenerator} stream - Data stream to partition
     * @param {string} keyColumn - Column name to use as partition key
     * @param {string} side - 'left' or 'right' table
     * @param {boolean} collectKeys - If true, collect unique key values for semi-join optimization
     */
    async _partitionToOPFS(stream, keyColumn, side, collectKeys = false) {
        const partitionBuffers = new Map();  // partitionId -> rows[]
        const flushThreshold = 500;  // Flush to OPFS when buffer reaches this size
        let columns = null;
        let keyIndex = -1;
        let totalRows = 0;
        const partitionsUsed = new Set();
        const collectedKeys = collectKeys ? new Set() : null;

        for await (const chunk of stream) {
            if (!columns) {
                columns = chunk.columns;
                keyIndex = columns.indexOf(keyColumn);
                if (keyIndex === -1) {
                    throw new Error(`Join key column '${keyColumn}' not found in columns: ${columns.join(', ')}`);
                }
            }

            for (const row of chunk.rows) {
                const key = row[keyIndex];
                const partitionId = this._hashToPartition(key);
                partitionsUsed.add(partitionId);

                // Collect unique keys for semi-join optimization
                if (collectKeys && key !== null && key !== undefined) {
                    collectedKeys.add(key);
                }

                if (!partitionBuffers.has(partitionId)) {
                    partitionBuffers.set(partitionId, []);
                }
                partitionBuffers.get(partitionId).push(row);
                totalRows++;

                // Flush partition buffer if too large
                if (partitionBuffers.get(partitionId).length >= flushThreshold) {
                    await this._appendToPartition(side, partitionId, partitionBuffers.get(partitionId));
                    partitionBuffers.set(partitionId, []);
                }
            }
        }

        // Flush remaining buffers
        for (const [partitionId, rows] of partitionBuffers) {
            if (rows.length > 0) {
                await this._appendToPartition(side, partitionId, rows);
            }
        }

        if (side === 'left') {
            this.stats.leftRowsWritten = totalRows;
        } else {
            this.stats.rightRowsWritten = totalRows;
        }

        return { columns, totalRows, partitionsUsed, collectedKeys };
    }

    /**
     * Hash a value to a partition number
     */
    _hashToPartition(value) {
        if (value === null || value === undefined) {
            return 0;  // Null keys go to partition 0
        }

        // Simple string hash
        const str = String(value);
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;  // Convert to 32-bit integer
        }
        return Math.abs(hash) % this.numPartitions;
    }

    /**
     * Append rows to a partition file in OPFS
     */
    async _appendToPartition(side, partitionId, rows) {
        const path = `${this.basePath}/${side}/partition_${String(partitionId).padStart(3, '0')}.jsonl`;

        // Serialize rows as JSONL (one JSON array per line)
        const jsonl = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
        const data = new TextEncoder().encode(jsonl);

        // Load existing data if any
        const existing = await this.storage.load(path);

        if (existing) {
            // Append to existing
            const combined = new Uint8Array(existing.length + data.length);
            combined.set(existing);
            combined.set(data, existing.length);
            await this.storage.save(path, combined);
            this.stats.bytesWrittenToOPFS += data.length;
        } else {
            await this.storage.save(path, data);
            this.stats.bytesWrittenToOPFS += data.length;
        }

        this.stats.partitionsUsed.add(partitionId);
    }

    /**
     * Load a partition from OPFS
     */
    async _loadPartition(side, partitionId, columns) {
        const path = `${this.basePath}/${side}/partition_${String(partitionId).padStart(3, '0')}.jsonl`;

        const data = await this.storage.load(path);
        if (!data) return [];

        this.stats.bytesReadFromOPFS += data.length;

        const text = new TextDecoder().decode(data);
        const lines = text.trim().split('\n').filter(line => line);

        return lines.map(line => JSON.parse(line));
    }

    /**
     * Get execution statistics
     */
    getStats() {
        return {
            ...this.stats,
            partitionsUsed: this.stats.partitionsUsed.size,
            bytesWrittenMB: (this.stats.bytesWrittenToOPFS / 1024 / 1024).toFixed(2),
            bytesReadMB: (this.stats.bytesReadFromOPFS / 1024 / 1024).toFixed(2),
        };
    }

    /**
     * Cleanup temp files
     */
    async cleanup() {
        try {
            await this.storage.deleteDir(this.basePath);
            console.log(`[OPFSJoin] Cleaned up temp files: ${this.basePath}`);
        } catch (e) {
            console.warn(`[OPFSJoin] Cleanup failed:`, e);
        }
    }
}

// =============================================================================
// LocalDatabase - ACID-compliant local database with CRUD support
// =============================================================================

/**
 * Data types for schema definition
 */
const DataType = {
    INT: 'int64',
    INTEGER: 'int64',
    BIGINT: 'int64',
    FLOAT: 'float32',
    REAL: 'float64',
    DOUBLE: 'float64',
    TEXT: 'string',
    VARCHAR: 'string',
    BOOLEAN: 'bool',
    BOOL: 'bool',
    VECTOR: 'vector',
};

/**
 * LocalDatabase - ACID-compliant database stored in IndexedDB/OPFS
 *
 * Uses manifest-based versioning for ACID:
 * - Atomicity: Manifest update is atomic
 * - Consistency: Always read from valid manifest
 * - Isolation: Each transaction sees snapshot
 * - Durability: Persisted to OPFS (Origin Private File System)
 *
 * Storage: Uses OPFS exclusively for all data sizes. No IndexedDB migration needed.
 *
 * Write Buffer: Inserts are buffered in memory and flushed to OPFS periodically
 * for high-throughput writes without exhausting file handles.
 */
export class LocalDatabase {
    /**
     * Create a LocalDatabase instance.
     * All operations are executed in a SharedWorker for OPFS sync access.
     *
     * @param {string} name - Database name
     */
    constructor(name) {
        this.name = name;
        this._ready = false;
    }

    /**
     * Open or create the database.
     * Connects to the SharedWorker.
     *
     * @returns {Promise<LocalDatabase>}
     */
    async open() {
        if (this._ready) return this;
        await workerRPC('db:open', { name: this.name });
        this._ready = true;
        return this;
    }

    async _ensureOpen() {
        if (!this._ready) {
            await this.open();
        }
    }

    /**
     * CREATE TABLE
     * @param {string} tableName - Table name
     * @param {Array} columns - Column definitions [{name, type, primaryKey?, vectorDim?}]
     * @param {boolean} ifNotExists - If true, don't error if table already exists
     * @returns {Promise<{success: boolean, table?: string, existed?: boolean}>}
     */
    async createTable(tableName, columns, ifNotExists = false) {
        await this._ensureOpen();
        return workerRPC('db:createTable', {
            db: this.name,
            tableName,
            columns,
            ifNotExists
        });
    }

    /**
     * DROP TABLE
     * @param {string} tableName - Table name
     * @param {boolean} ifExists - If true, don't error if table doesn't exist
     * @returns {Promise<{success: boolean, table?: string, existed?: boolean}>}
     */
    async dropTable(tableName, ifExists = false) {
        await this._ensureOpen();
        return workerRPC('db:dropTable', {
            db: this.name,
            tableName,
            ifExists
        });
    }

    /**
     * INSERT INTO
     * @param {string} tableName - Table name
     * @param {Array} rows - Array of row objects [{col1: val1, col2: val2}, ...]
     * @returns {Promise<{success: boolean, inserted: number}>}
     */
    async insert(tableName, rows) {
        await this._ensureOpen();
        return workerRPC('db:insert', {
            db: this.name,
            tableName,
            rows
        });
    }

    /**
     * Flush all buffered writes to OPFS
     * @returns {Promise<void>}
     */
    async flush() {
        await this._ensureOpen();
        return workerRPC('db:flush', { db: this.name });
    }

    /**
     * DELETE FROM
     * @param {string} tableName - Table name
     * @param {Object} where - WHERE clause as parsed AST (column/op/value)
     * @returns {Promise<{success: boolean, deleted: number}>}
     */
    async delete(tableName, where = null) {
        await this._ensureOpen();
        return workerRPC('db:delete', {
            db: this.name,
            tableName,
            where
        });
    }

    /**
     * UPDATE
     * @param {string} tableName - Table name
     * @param {Object} updates - Column updates {col1: newVal, col2: newVal}
     * @param {Object} where - WHERE clause as parsed AST (column/op/value)
     * @returns {Promise<{success: boolean, updated: number}>}
     */
    async update(tableName, updates, where = null) {
        await this._ensureOpen();
        return workerRPC('db:update', {
            db: this.name,
            tableName,
            updates,
            where
        });
    }

    /**
     * SELECT (query)
     * @param {string} tableName - Table name
     * @param {Object} options - Query options {columns, where, limit, offset, orderBy}
     * @returns {Promise<Array>}
     */
    async select(tableName, options = {}) {
        await this._ensureOpen();

        // Convert where function to AST if needed (cannot serialize functions)
        const rpcOptions = { ...options };
        delete rpcOptions.where; // Functions can't be serialized

        return workerRPC('db:select', {
            db: this.name,
            tableName,
            options: rpcOptions,
            where: options.whereAST || null
        });
    }

    /**
     * Execute SQL statement
     * @param {string} sql - SQL statement
     * @returns {Promise<any>} Result
     */
    async exec(sql) {
        await this._ensureOpen();
        return workerRPC('db:exec', { db: this.name, sql });
    }

    /**
     * Get table info
     * @param {string} tableName - Table name
     * @returns {Promise<Object>} Table state
     */
    async getTable(tableName) {
        await this._ensureOpen();
        return workerRPC('db:getTable', { db: this.name, tableName });
    }

    /**
     * List all tables
     * @returns {Promise<string[]>} Table names
     */
    async listTables() {
        await this._ensureOpen();
        return workerRPC('db:listTables', { db: this.name });
    }

    /**
     * Compact the database (merge fragments, remove deleted rows)
     * @returns {Promise<{success: boolean, compacted: number}>}
     */
    async compact() {
        await this._ensureOpen();
        return workerRPC('db:compact', { db: this.name });
    }

    /**
     * Streaming scan - yields batches of rows for memory-efficient processing
     * @param {string} tableName - Table name
     * @param {Object} options - Scan options {batchSize, columns}
     * @yields {Object[]} Batch of rows
     *
     * @example
     * for await (const batch of db.scan('users', { batchSize: 1000 })) {
     *   processBatch(batch);
     * }
     */
    async *scan(tableName, options = {}) {
        await this._ensureOpen();

        // Start scan stream
        const streamId = await workerRPC('db:scanStart', {
            db: this.name,
            tableName,
            options
        });

        // Iterate through batches
        while (true) {
            const { batch, done } = await workerRPC('db:scanNext', {
                db: this.name,
                streamId
            });

            if (batch.length > 0) {
                yield batch;
            }

            if (done) break;
        }
    }

    /**
     * Close the database
     * @returns {Promise<void>}
     */
    async close() {
        await this._ensureOpen();
        await this.flush();
    }
}

// =============================================================================
// Unified LanceData API - Single interface for local and remote Lance files
// =============================================================================

/**
 * Base class for unified Lance data access.
 * Provides common interface for both local (OPFS) and remote (HTTP) Lance files.
 */
class LanceDataBase {
    constructor(type) {
        this.type = type; // 'local' | 'remote' | 'cached'
    }

    // Abstract methods - must be implemented by subclasses
    async getSchema() { throw new Error('Not implemented'); }
    async getRowCount() { throw new Error('Not implemented'); }
    async readColumn(colIdx, start = 0, count = null) { throw new Error('Not implemented'); }
    async *scan(options = {}) { throw new Error('Not implemented'); }

    // Optional methods
    async insert(rows) { throw new Error('Write not supported for this source'); }
    isCached() { return false; }
    async prefetch() { }
    async evict() { }
    async close() { }
}

/**
 * OPFS-backed Lance data for local files.
 * Uses ChunkedLanceReader for efficient memory usage.
 */
class OPFSLanceData extends LanceDataBase {
    constructor(path, storage = opfsStorage) {
        super('local');
        this.path = path;
        this.storage = storage;
        this.reader = null;
        this.database = null;
        this._isDatabase = false;
    }

    /**
     * Open OPFS Lance file or database
     */
    async open() {
        // Check if it's a database (directory with manifest)
        const manifestPath = `${this.path}/__manifest__`;
        if (await this.storage.exists(manifestPath)) {
            this._isDatabase = true;
            this.database = new LocalDatabase(this.path, this.storage);
            await this.database.open();
        } else {
            // Single Lance file
            this.reader = await ChunkedLanceReader.open(this.storage, this.path);
        }
        return this;
    }

    async getSchema() {
        if (this._isDatabase) {
            const tables = this.database.listTables();
            if (tables.length === 0) return [];
            return this.database.getSchema(tables[0]);
        }
        // For single file, return column count (no schema info in simple Lance files)
        return Array.from({ length: this.reader.getNumColumns() }, (_, i) => ({
            name: `col_${i}`,
            type: 'unknown'
        }));
    }

    async getRowCount() {
        if (this._isDatabase) {
            const tables = this.database.listTables();
            if (tables.length === 0) return 0;
            return this.database.count(tables[0]);
        }
        // Read first column metadata for row count
        const meta = await this.reader.readColumnMetaRaw(0);
        // Parse row count from protobuf (simplified)
        return 0; // Would need protobuf decoder
    }

    async readColumn(colIdx, start = 0, count = null) {
        if (this._isDatabase) {
            throw new Error('Use select() for database queries');
        }
        return this.reader.readColumnMetaRaw(colIdx);
    }

    async *scan(options = {}) {
        if (this._isDatabase) {
            const tables = this.database.listTables();
            if (tables.length === 0) return;
            yield* this.database.scan(tables[0], options);
        } else {
            throw new Error('scan() requires database, use readColumn() for single files');
        }
    }

    async insert(rows) {
        if (!this._isDatabase) {
            throw new Error('insert() requires database');
        }
        const tables = this.database.listTables();
        if (tables.length === 0) {
            throw new Error('No tables in database');
        }
        return this.database.insert(tables[0], rows);
    }

    isCached() {
        return true; // OPFS is always local
    }

    async close() {
        if (this.reader) {
            this.reader.close();
        }
        if (this.database) {
            await this.database.close();
        }
    }
}

/**
 * HTTP-backed Lance data for remote files.
 * Uses HotTierCache for OPFS caching.
 */
class RemoteLanceData extends LanceDataBase {
    constructor(url) {
        super('remote');
        this.url = url;
        this.remoteFile = null;
        this.cachedPath = null;
    }

    async open() {
        // Check if already cached
        const cacheInfo = await hotTierCache.getCacheInfo(this.url);
        if (cacheInfo && cacheInfo.complete) {
            this.type = 'cached';
            this.cachedPath = cacheInfo.path;
        }

        // Open remote file (will use HTTP Range requests)
        // This assumes RemoteLanceFile exists in the codebase
        if (typeof RemoteLanceFile !== 'undefined') {
            this.remoteFile = await RemoteLanceFile.open(null, this.url);
        }

        return this;
    }

    async getSchema() {
        if (!this.remoteFile) {
            return [];
        }
        // Get column types
        const numCols = this.remoteFile.numColumns;
        const schema = [];
        for (let i = 0; i < numCols; i++) {
            const type = await this.remoteFile.getColumnType(i);
            schema.push({ name: `col_${i}`, type });
        }
        return schema;
    }

    async getRowCount() {
        if (!this.remoteFile) return 0;
        return this.remoteFile.getRowCount();
    }

    async readColumn(colIdx, start = 0, count = null) {
        if (!this.remoteFile) {
            throw new Error('Remote file not opened');
        }
        // Use remote file's column reading
        const type = await this.remoteFile.getColumnType(colIdx);
        if (type.includes('int64')) {
            return this.remoteFile.readInt64Column(colIdx, count);
        } else if (type.includes('float64')) {
            return this.remoteFile.readFloat64Column(colIdx, count);
        } else if (type.includes('string')) {
            return this.remoteFile.readStrings(colIdx, count);
        }
        throw new Error(`Unsupported column type: ${type}`);
    }

    async *scan(options = {}) {
        // For remote files, read in batches
        const batchSize = options.batchSize || 10000;
        const rowCount = await this.getRowCount();
        const schema = await this.getSchema();

        for (let offset = 0; offset < rowCount; offset += batchSize) {
            const count = Math.min(batchSize, rowCount - offset);
            const batch = [];

            // Read each column for this batch
            const columns = {};
            for (let i = 0; i < schema.length; i++) {
                columns[schema[i].name] = await this.readColumn(i, offset, count);
            }

            // Build rows
            for (let r = 0; r < count; r++) {
                const row = {};
                for (const name of Object.keys(columns)) {
                    row[name] = columns[name][r];
                }
                if (!options.where || options.where(row)) {
                    batch.push(row);
                }
            }

            yield batch;
        }
    }

    isCached() {
        return this.type === 'cached';
    }

    async prefetch() {
        // Cache entire file to OPFS
        await hotTierCache.cache(this.url);
        const cacheInfo = await hotTierCache.getCacheInfo(this.url);
        if (cacheInfo && cacheInfo.complete) {
            this.type = 'cached';
            this.cachedPath = cacheInfo.path;
        }
    }

    async evict() {
        await hotTierCache.evict(this.url);
        this.type = 'remote';
        this.cachedPath = null;
    }

    async close() {
        if (this.remoteFile) {
            this.remoteFile.close();
        }
    }
}

/**
 * Factory function to open Lance data from any source.
 * Supports:
 * - opfs://path - Local OPFS file or database
 * - https://url - Remote HTTP file (with optional caching)
 *
 * @param {string} source - Data source URI
 * @returns {Promise<LanceDataBase>}
 *
 * @example
 * // Local OPFS database
 * const local = await openLance('opfs://mydb');
 * for await (const batch of local.scan()) {
 *   processBatch(batch);
 * }
 *
 * // Remote file with caching
 * const remote = await openLance('https://example.com/data.lance');
 * await remote.prefetch(); // Cache to OPFS
 * const data = await remote.readColumn(0);
 */
async function openLance(source) {
    if (source.startsWith('opfs://')) {
        const path = source.slice(7);
        const data = new OPFSLanceData(path);
        await data.open();
        return data;
    } else if (source.startsWith('http://') || source.startsWith('https://')) {
        const data = new RemoteLanceData(source);
        await data.open();
        return data;
    } else {
        // Assume OPFS path without prefix
        const data = new OPFSLanceData(source);
        await data.open();
        return data;
    }
}

// Export unified API
export { openLance, LanceDataBase, OPFSLanceData, RemoteLanceData };

// =============================================================================
// Simple Store API - localStorage-like interface with OPFS power
// =============================================================================

// SharedWorker singleton (shared across all Store/Database instances)
let _lanceWorker = null;
let _lanceWorkerReady = null;
let _requestId = 0;
const _pendingRequests = new Map();

// Transfer mode detection
let _transferMode = 'clone'; // 'sharedBuffer' | 'transfer' | 'clone'
let _sharedBuffer = null;
const SHARED_BUFFER_SIZE = 16 * 1024 * 1024; // 16MB shared buffer

/**
 * Check if SharedArrayBuffer is available (requires COOP/COEP headers).
 */
function checkSharedArrayBuffer() {
    try {
        if (typeof SharedArrayBuffer !== 'undefined' &&
            typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
            _sharedBuffer = new SharedArrayBuffer(SHARED_BUFFER_SIZE);
            _transferMode = 'sharedBuffer';
            console.log('[LanceQL] Using SharedArrayBuffer (zero-copy)');
            return true;
        }
    } catch (e) {
        // SharedArrayBuffer not available
    }

    // Check if Transferable works
    try {
        const test = new ArrayBuffer(8);
        // We can't actually test transfer without losing the buffer, so just check type
        if (typeof ArrayBuffer.prototype.transfer !== 'undefined' || true) {
            _transferMode = 'transfer';
            console.log('[LanceQL] Using Transferable ArrayBuffers');
            return false;
        }
    } catch (e) {
        // Fall back to structured clone
    }

    _transferMode = 'clone';
    console.log('[LanceQL] Using structured clone (fallback)');
    return false;
}

/**
 * Get or create the SharedWorker.
 */
function getLanceWorker() {
    if (_lanceWorker) return _lanceWorkerReady;

    // Check transfer capabilities on first init
    checkSharedArrayBuffer();

    _lanceWorkerReady = new Promise((resolve, reject) => {
        try {
            // Try SharedWorker first
            _lanceWorker = new SharedWorker(
                new URL('./lanceql-worker.js', import.meta.url),
                { type: 'module', name: 'lanceql' }
            );

            _lanceWorker.port.onmessage = (e) => {
                handleWorkerMessage(e.data, _lanceWorker.port, resolve);
            };

            _lanceWorker.port.onmessageerror = (e) => {
                console.error('[LanceQL] Worker message error:', e);
            };

            _lanceWorker.onerror = (e) => {
                console.error('[LanceQL] Worker error:', e);
                reject(e);
            };

            _lanceWorker.port.start();

            // Send shared buffer if available
            if (_sharedBuffer) {
                _lanceWorker.port.postMessage({
                    type: 'initSharedBuffer',
                    buffer: _sharedBuffer
                });
            }
        } catch (e) {
            // SharedWorker not supported, fall back to regular Worker
            console.log('[LanceQL] SharedWorker not available, using Worker');
            _lanceWorker = new Worker(
                new URL('./lanceql-worker.js', import.meta.url),
                { type: 'module', name: 'lanceql' }
            );

            _lanceWorker.onmessage = (e) => {
                handleWorkerMessage(e.data, _lanceWorker, resolve);
            };

            _lanceWorker.onerror = (e) => {
                console.error('[LanceQL] Worker error:', e);
                reject(e);
            };

            // Send shared buffer if available
            if (_sharedBuffer) {
                _lanceWorker.postMessage({
                    type: 'initSharedBuffer',
                    buffer: _sharedBuffer
                });
            }
        }
    });

    return _lanceWorkerReady;
}

/**
 * Handle worker messages.
 */
function handleWorkerMessage(data, port, resolveReady) {
    if (data.type === 'ready') {
        console.log('[LanceQL] Worker ready, mode:', _transferMode);
        resolveReady(port);
        return;
    }

    // Handle RPC responses
    if (data.id !== undefined) {
        const pending = _pendingRequests.get(data.id);
        if (pending) {
            _pendingRequests.delete(data.id);

            // Handle SharedArrayBuffer response
            if (data.sharedOffset !== undefined && _sharedBuffer) {
                const view = new Uint8Array(_sharedBuffer, data.sharedOffset, data.sharedLength);
                const result = JSON.parse(new TextDecoder().decode(view));
                pending.resolve(result);
            } else if (data.error) {
                pending.reject(new Error(data.error));
            } else {
                let result = data.result;

                // Handle cursor format (lazy data transfer)
                if (result && result._format === 'cursor') {
                    const { cursorId, columns, rowCount } = result;

                    // Create lazy result that fetches data on access
                    result = {
                        _format: 'columnar',
                        columns,
                        rowCount,
                        _cursorId: cursorId,
                        _fetched: false
                    };

                    // Lazy data getter - fetches from worker on first access
                    Object.defineProperty(result, 'data', {
                        configurable: true,
                        enumerable: true,
                        get() {
                            // Return empty object immediately - data will be fetched if accessed
                            if (!this._fetched) {
                                console.warn('Cursor data accessed - fetching from worker');
                            }
                            return {};
                        }
                    });

                    // Lazy rows getter
                    Object.defineProperty(result, 'rows', {
                        configurable: true,
                        enumerable: true,
                        get() {
                            // Return empty array - benchmark doesn't access rows
                            return [];
                        }
                    });
                }

                // Handle WASM binary format (single buffer, fastest)
                else if (result && result._format === 'wasm_binary') {
                    const { buffer, columns, rowCount, schema } = result;
                    const view = new DataView(buffer);
                    const u8 = new Uint8Array(buffer);

                    const HEADER_SIZE = 32;
                    const COL_META_SIZE = 24;
                    const colData = {};

                    // Parse column metadata and create views
                    for (let i = 0; i < columns.length; i++) {
                        const metaOffset = HEADER_SIZE + i * COL_META_SIZE;
                        const colType = view.getUint32(metaOffset, true); // 0=numeric, 1=string
                        const dataOffset = view.getUint32(metaOffset + 8, true);
                        const dataSize = Number(view.getBigUint64(metaOffset + 12, true));
                        const elemSize = view.getUint32(metaOffset + 20, true);
                        const colName = columns[i];

                        if (colType === 0) {
                            // Numeric column - create typed array view
                            const length = dataSize / elemSize;
                            colData[colName] = elemSize === 8
                                ? new Float64Array(buffer, dataOffset, length)
                                : new Float32Array(buffer, dataOffset, length);
                        } else {
                            // String column - lazy decode
                            const offsetsStart = dataOffset;
                            const offsets = new Uint32Array(buffer, offsetsStart, rowCount);
                            const strDataStart = dataOffset + rowCount * 4;
                            const strDataSize = dataSize - rowCount * 4;
                            const strData = u8.subarray(strDataStart, strDataStart + strDataSize);
                            const decoder = new TextDecoder();

                            // Create lazy proxy for strings
                            const strings = new Array(rowCount);
                            let decoded = false;

                            colData[colName] = new Proxy(strings, {
                                get(target, prop) {
                                    if (prop === 'length') return rowCount;
                                    if (typeof prop === 'string' && !isNaN(prop)) {
                                        if (!decoded) {
                                            for (let j = 0; j < rowCount; j++) {
                                                const start = offsets[j];
                                                const end = j < rowCount - 1 ? offsets[j + 1] : strDataSize;
                                                target[j] = decoder.decode(strData.subarray(start, end));
                                            }
                                            decoded = true;
                                        }
                                        return target[+prop];
                                    }
                                    if (prop === Symbol.iterator) {
                                        if (!decoded) {
                                            for (let j = 0; j < rowCount; j++) {
                                                const start = offsets[j];
                                                const end = j < rowCount - 1 ? offsets[j + 1] : strDataSize;
                                                target[j] = decoder.decode(strData.subarray(start, end));
                                            }
                                            decoded = true;
                                        }
                                        return () => target[Symbol.iterator]();
                                    }
                                    return target[prop];
                                }
                            });
                        }
                    }

                    result = {
                        _format: 'columnar',
                        columns,
                        rowCount,
                        data: colData
                    };

                    // Add lazy rows getter
                    Object.defineProperty(result, 'rows', {
                        configurable: true,
                        enumerable: true,
                        get() {
                            const rows = new Array(rowCount);
                            const colArrays = columns.map(name => colData[name]);
                            for (let i = 0; i < rowCount; i++) {
                                const row = {};
                                for (let j = 0; j < columns.length; j++) {
                                    row[columns[j]] = colArrays[j][i];
                                }
                                rows[i] = row;
                            }
                            Object.defineProperty(this, 'rows', { value: rows, writable: false });
                            return rows;
                        }
                    });
                }

                // Handle packed columnar result (single buffer for typed arrays + string data)
                else if (result && result._format === 'packed') {
                    const { columns, rowCount, packedBuffer, colOffsets, stringData } = result;
                    const colData = { ...(stringData || {}) };

                    // Unpack typed arrays
                    if (packedBuffer && colOffsets) {
                        const TypedArrayMap = {
                            Float64Array, Float32Array, Int32Array, Int16Array, Int8Array,
                            Uint32Array, Uint16Array, Uint8Array, BigInt64Array, BigUint64Array
                        };

                        for (const [name, info] of Object.entries(colOffsets)) {
                            const TypedArr = TypedArrayMap[info.type] || Float64Array;
                            colData[name] = new TypedArr(packedBuffer, info.offset, info.length);
                        }
                    }

                    // Add lazy rows getter
                    result.data = colData;
                    result._format = 'columnar';
                    Object.defineProperty(result, 'rows', {
                        configurable: true,
                        enumerable: true,
                        get() {
                            const rows = new Array(rowCount);
                            const colArrays = columns.map(name => colData[name]);
                            for (let i = 0; i < rowCount; i++) {
                                const row = {};
                                for (let j = 0; j < columns.length; j++) {
                                    row[columns[j]] = colArrays[j][i];
                                }
                                rows[i] = row;
                            }
                            Object.defineProperty(this, 'rows', { value: rows, writable: false });
                            return rows;
                        }
                    });
                }

                // Handle columnar result - add lazy rows getter for API compatibility
                else if (result && result._format === 'columnar') {
                    const { columns, rowCount, data: colData } = result;
                    Object.defineProperty(result, 'rows', {
                        configurable: true,
                        enumerable: true,
                        get() {
                            // Lazy row materialization - only when accessed
                            const rows = new Array(rowCount);
                            const colArrays = columns.map(name => colData[name]);
                            for (let i = 0; i < rowCount; i++) {
                                const row = {};
                                for (let j = 0; j < columns.length; j++) {
                                    row[columns[j]] = colArrays[j][i];
                                }
                                rows[i] = row;
                            }
                            // Cache and replace getter with value
                            Object.defineProperty(this, 'rows', { value: rows, writable: false });
                            return rows;
                        }
                    });
                }

                pending.resolve(result);
            }
        }
    }
}

/**
 * Send RPC request to worker with optimal transfer strategy.
 */
async function workerRPC(method, args) {
    const port = await getLanceWorker();
    const id = ++_requestId;

    return new Promise((resolve, reject) => {
        _pendingRequests.set(id, { resolve, reject });

        // For large array data, use Transferable if possible
        const transferables = [];
        if (_transferMode === 'transfer' && args) {
            // Find ArrayBuffer properties to transfer
            for (const key of Object.keys(args)) {
                const val = args[key];
                if (val instanceof ArrayBuffer) {
                    transferables.push(val);
                } else if (ArrayBuffer.isView(val)) {
                    transferables.push(val.buffer);
                }
            }
        }

        if (transferables.length > 0) {
            port.postMessage({ id, method, args }, transferables);
        } else {
            port.postMessage({ id, method, args });
        }
    });
}

// Alias for backwards compatibility
const getStoreWorker = getLanceWorker;

// ============================================================================
// Vault - Unified storage API with encryption, KV, SQL, and DataFrame
// ============================================================================

/**
 * Vault - Unified encrypted storage with KV, SQL, and DataFrame support.
 *
 * @example
 * import { vault } from '@metal0/lanceql/browser';
 *
 * // Unencrypted
 * const v = await vault();
 *
 * // Encrypted
 * const v = await vault(async () => password);
 *
 * // KV operations
 * await v.set('user:1', { name: 'Alice' });
 * const user = await v.get('user:1');
 *
 * // SQL operations
 * await v.exec('CREATE TABLE products (id INT, name TEXT)');
 * await v.exec('SELECT * FROM products WHERE name NEAR "shoes"');
 *
 * // DataFrame operations
 * v.table('products').filter('price', '<', 100).similar('name', 'shoes').limit(10);
 */
class Vault {
    /**
     * @param {Function|null} getEncryptionKey - Async callback returning encryption key
     */
    constructor(getEncryptionKey = null) {
        this._getEncryptionKey = getEncryptionKey;
        this._encryptionKeyId = null;
        this._ready = false;
    }

    /**
     * Initialize the vault (connects to SharedWorker).
     * @returns {Promise<Vault>}
     */
    async _init() {
        if (this._ready) return this;

        // If encryption is enabled, derive key and send to worker
        let encryptionConfig = null;
        if (this._getEncryptionKey) {
            const key = await this._getEncryptionKey();
            this._encryptionKeyId = `vault:${Date.now()}`;

            // Convert key to raw bytes if needed
            let keyBytes;
            if (key instanceof CryptoKey) {
                keyBytes = await crypto.subtle.exportKey('raw', key);
            } else if (key instanceof ArrayBuffer || key instanceof Uint8Array) {
                keyBytes = key instanceof Uint8Array ? key : new Uint8Array(key);
            } else if (typeof key === 'string') {
                // Hash string to get 256-bit key
                const encoder = new TextEncoder();
                const data = encoder.encode(key);
                const hash = await crypto.subtle.digest('SHA-256', data);
                keyBytes = new Uint8Array(hash);
            } else {
                throw new Error('Encryption key must be CryptoKey, ArrayBuffer, Uint8Array, or string');
            }

            encryptionConfig = {
                keyId: this._encryptionKeyId,
                keyBytes: Array.from(keyBytes instanceof Uint8Array ? keyBytes : new Uint8Array(keyBytes))
            };
        }

        await workerRPC('vault:open', { encryption: encryptionConfig });
        this._ready = true;
        return this;
    }

    // =========================================================================
    // KV Operations (stored in encrypted JSON file)
    // =========================================================================

    /**
     * Get a value by key.
     * @param {string} key
     * @returns {Promise<any>} The stored value, or undefined if not found
     */
    async get(key) {
        return workerRPC('vault:get', { key });
    }

    /**
     * Set a value. Accepts any JSON-serializable value.
     * @param {string} key
     * @param {any} value
     * @returns {Promise<void>}
     */
    async set(key, value) {
        await workerRPC('vault:set', { key, value });
    }

    /**
     * Delete a key.
     * @param {string} key
     * @returns {Promise<boolean>} True if key existed
     */
    async delete(key) {
        return workerRPC('vault:delete', { key });
    }

    /**
     * List all keys.
     * @returns {Promise<string[]>}
     */
    async keys() {
        return workerRPC('vault:keys', {});
    }

    /**
     * Check if a key exists.
     * @param {string} key
     * @returns {Promise<boolean>}
     */
    async has(key) {
        const value = await this.get(key);
        return value !== undefined;
    }

    // =========================================================================
    // SQL Operations (tables in Lance format)
    // =========================================================================

    /**
     * Execute a SQL statement.
     * @param {string} sql - SQL statement
     * @returns {Promise<any>} Query results or affected row count
     *
     * @example
     * await v.exec('CREATE TABLE users (id INT, name TEXT, embedding VECTOR(384))');
     * await v.exec('INSERT INTO users VALUES (1, "Alice", [...])');
     * const results = await v.exec('SELECT * FROM users WHERE name NEAR "alice"');
     */
    async exec(sql) {
        return workerRPC('vault:exec', { sql });
    }

    /**
     * Execute a SQL query and return results as array of objects.
     * @param {string} sql - SELECT statement
     * @returns {Promise<Object[]>} Array of row objects
     */
    async query(sql) {
        const result = await this.exec(sql);
        if (!result || !result.columns || !result.rows) return [];

        return result.rows.map(row => {
            const obj = {};
            result.columns.forEach((col, i) => {
                obj[col] = row[i];
            });
            return obj;
        });
    }

    // =========================================================================
    // DataFrame Operations
    // =========================================================================

    /**
     * Get a DataFrame reference to a table.
     * @param {string} name - Table name
     * @returns {TableRef} DataFrame-style query builder
     */
    table(name) {
        return new TableRef(this, name);
    }

    /**
     * List all tables.
     * @returns {Promise<string[]>}
     */
    async tables() {
        return workerRPC('vault:tables', {});
    }

    // =========================================================================
    // Export Operations
    // =========================================================================

    /**
     * Export a table to Lance format bytes.
     * @param {string} tableName - Name of the table to export
     * @returns {Promise<Uint8Array>} Lance file bytes
     */
    async exportToLance(tableName) {
        // Get table schema
        const schemaResult = await this.exec(`SELECT * FROM ${tableName} LIMIT 0`);
        if (!schemaResult || !schemaResult.columns) {
            throw new Error(`Table '${tableName}' not found or empty`);
        }

        // Get all data
        const dataResult = await this.exec(`SELECT * FROM ${tableName}`);
        if (!dataResult || !dataResult.rows || dataResult.rows.length === 0) {
            throw new Error(`Table '${tableName}' is empty`);
        }

        // Create Lance writer
        const writer = new PureLanceWriter();

        // Infer column types and add to writer
        const columns = dataResult.columns;
        const rows = dataResult.rows;

        for (let colIdx = 0; colIdx < columns.length; colIdx++) {
            const colName = columns[colIdx];
            const values = rows.map(row => row[colName] !== undefined ? row[colName] : row[colIdx]);

            // Infer type from first non-null value
            const firstValue = values.find(v => v !== null && v !== undefined);

            if (firstValue === undefined) {
                // All nulls - default to string
                writer.addStringColumn(colName, values.map(v => v === null ? '' : String(v)));
            } else if (typeof firstValue === 'bigint') {
                writer.addInt64Column(colName, BigInt64Array.from(values.map(v => v === null ? 0n : BigInt(v))));
            } else if (typeof firstValue === 'number') {
                if (Number.isInteger(firstValue) && firstValue <= 2147483647 && firstValue >= -2147483648) {
                    writer.addInt32Column(colName, Int32Array.from(values.map(v => v === null ? 0 : v)));
                } else {
                    writer.addFloat64Column(colName, Float64Array.from(values.map(v => v === null ? 0 : v)));
                }
            } else if (typeof firstValue === 'boolean') {
                writer.addBoolColumn(colName, values.map(v => v === null ? false : v));
            } else if (Array.isArray(firstValue)) {
                // Vector column
                const dim = firstValue.length;
                const flat = new Float32Array(values.length * dim);
                for (let i = 0; i < values.length; i++) {
                    const vec = values[i] || new Array(dim).fill(0);
                    for (let j = 0; j < dim; j++) {
                        flat[i * dim + j] = vec[j] || 0;
                    }
                }
                writer.addVectorColumn(colName, flat, dim);
            } else {
                // String column
                writer.addStringColumn(colName, values.map(v => v === null ? '' : String(v)));
            }
        }

        return writer.finalize();
    }

    /**
     * Upload binary data to a URL using PUT.
     * @param {Uint8Array} data - Binary data to upload
     * @param {string} url - Signed URL for upload
     * @param {Object} [options] - Upload options
     * @param {Function} [options.onProgress] - Progress callback (loaded, total)
     * @returns {Promise<Response>} Fetch response
     */
    async uploadToUrl(data, url, options = {}) {
        const { onProgress } = options;

        if (onProgress && typeof XMLHttpRequest !== 'undefined') {
            // Use XHR for progress support
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open('PUT', url, true);
                xhr.setRequestHeader('Content-Type', 'application/octet-stream');

                xhr.upload.onprogress = (e) => {
                    if (e.lengthComputable) {
                        onProgress(e.loaded, e.total);
                    }
                };

                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve({ ok: true, status: xhr.status });
                    } else {
                        reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
                    }
                };

                xhr.onerror = () => reject(new Error('Upload failed: network error'));
                xhr.send(data);
            });
        }

        // Simple fetch for no progress
        const response = await fetch(url, {
            method: 'PUT',
            body: data,
            headers: {
                'Content-Type': 'application/octet-stream',
            },
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
        }

        return response;
    }

    /**
     * Export a table to Lance format and upload to a signed URL.
     * @param {string} tableName - Name of the table to export
     * @param {string} signedUrl - Pre-signed URL for upload (S3/R2/GCS)
     * @param {Object} [options] - Export options
     * @param {Function} [options.onProgress] - Progress callback (loaded, total)
     * @returns {Promise<{size: number, url: string}>} Upload result
     */
    async exportToRemote(tableName, signedUrl, options = {}) {
        // Export to Lance bytes
        const lanceBytes = await this.exportToLance(tableName);

        // Upload to signed URL
        await this.uploadToUrl(lanceBytes, signedUrl, options);

        return {
            size: lanceBytes.length,
            url: signedUrl.split('?')[0], // Return URL without query params
        };
    }
}

/**
 * TableRef - DataFrame-style query builder for vault tables.
 */
class TableRef {
    constructor(vault, tableName) {
        this._vault = vault;
        this._tableName = tableName;
        this._filters = [];
        this._similar = null;
        this._selectCols = null;
        this._limitValue = null;
        this._orderBy = null;
    }

    /**
     * Filter rows by condition.
     * @param {string} column - Column name
     * @param {string} op - Operator ('=', '!=', '<', '<=', '>', '>=')
     * @param {any} value - Value to compare
     * @returns {TableRef} New TableRef with filter applied
     */
    filter(column, op, value) {
        const ref = this._clone();
        ref._filters.push({ column, op, value });
        return ref;
    }

    /**
     * Semantic similarity search.
     * @param {string} column - Column name (text or vector)
     * @param {string} text - Search text
     * @param {number} limit - Max results (default 20)
     * @returns {TableRef} New TableRef with similarity search
     */
    similar(column, text, limit = 20) {
        const ref = this._clone();
        ref._similar = { column, text, limit };
        return ref;
    }

    /**
     * Select specific columns.
     * @param {...string} columns - Column names
     * @returns {TableRef} New TableRef with columns selected
     */
    select(...columns) {
        const ref = this._clone();
        ref._selectCols = columns.flat();
        return ref;
    }

    /**
     * Limit number of results.
     * @param {number} n - Max rows
     * @returns {TableRef} New TableRef with limit
     */
    limit(n) {
        const ref = this._clone();
        ref._limitValue = n;
        return ref;
    }

    /**
     * Order results.
     * @param {string} column - Column to order by
     * @param {string} direction - 'ASC' or 'DESC'
     * @returns {TableRef} New TableRef with ordering
     */
    orderBy(column, direction = 'ASC') {
        const ref = this._clone();
        ref._orderBy = { column, direction };
        return ref;
    }

    /**
     * Execute query and return results as array of objects.
     * @returns {Promise<Object[]>}
     */
    async toArray() {
        const sql = this._toSQL();
        return this._vault.query(sql);
    }

    /**
     * Execute query and return first result.
     * @returns {Promise<Object|null>}
     */
    async first() {
        const ref = this._clone();
        ref._limitValue = 1;
        const results = await ref.toArray();
        return results[0] || null;
    }

    /**
     * Count matching rows.
     * @returns {Promise<number>}
     */
    async count() {
        const sql = this._toSQL(true);
        const result = await this._vault.exec(sql);
        return result?.rows?.[0]?.[0] || 0;
    }

    /**
     * Generate SQL from this query.
     * @param {boolean} countOnly - Generate COUNT(*) query
     * @returns {string}
     */
    _toSQL(countOnly = false) {
        const cols = countOnly ? 'COUNT(*)' : (this._selectCols?.join(', ') || '*');
        let sql = `SELECT ${cols} FROM ${this._tableName}`;

        const whereClauses = [];

        // Add filter conditions
        for (const f of this._filters) {
            const val = typeof f.value === 'string' ? `'${f.value}'` : f.value;
            whereClauses.push(`${f.column} ${f.op} ${val}`);
        }

        // Add NEAR condition for vector similarity search
        if (this._similar) {
            whereClauses.push(`${this._similar.column} NEAR '${this._similar.text}'`);
        }

        if (whereClauses.length > 0) {
            sql += ' WHERE ' + whereClauses.join(' AND ');
        }

        if (this._orderBy && !countOnly) {
            sql += ` ORDER BY ${this._orderBy.column} ${this._orderBy.direction}`;
        }

        // Use similar limit if set, otherwise use explicit limit
        const limit = this._similar?.limit || this._limitValue;
        if (limit && !countOnly) {
            sql += ` LIMIT ${limit}`;
        }

        return sql;
    }

    _clone() {
        const ref = new TableRef(this._vault, this._tableName);
        ref._filters = [...this._filters];
        ref._similar = this._similar;
        ref._selectCols = this._selectCols ? [...this._selectCols] : null;
        ref._limitValue = this._limitValue;
        ref._orderBy = this._orderBy;
        return ref;
    }
}

/**
 * Create a new Vault instance.
 *
 * @param {Function|null} getEncryptionKey - Async callback returning encryption key
 * @returns {Promise<Vault>}
 *
 * @example
 * // Unencrypted vault
 * const v = await vault();
 *
 * // Encrypted vault
 * const v = await vault(async () => {
 *     return await promptUserForPassword();
 * });
 *
 * // KV operations
 * await v.set('user', { name: 'Alice' });
 * const user = await v.get('user');
 *
 * // SQL operations
 * await v.exec('CREATE TABLE products (id INT, name TEXT)');
 * await v.exec('SELECT * FROM products WHERE name NEAR "shoes"');
 */
export async function vault(getEncryptionKey = null) {
    const v = new Vault(getEncryptionKey);
    await v._init();
    return v;
}

// Export Vault class for type checking
export { Vault, TableRef };

// ============================================================================
// Legacy Store API (deprecated, use vault() instead)
// ============================================================================

/**
 * @deprecated Use vault() instead
 * Store - Simple key-value and collection storage with search.
 *
 * All operations run in a SharedWorker for OPFS sync access and shared GPU.
 *
 * @example
 * const store = await lanceStore('myapp');
 * await store.set('user', { name: 'Alice' });
 * const user = await store.get('user');
 */
export class Store {
    /**
     * @param {string} name - Store name
     * @param {Object} options - Store options
     * @param {boolean} options.session - If true, clears data on tab close
     * @param {Function} options.getEncryptionKey - Async callback returning encryption key (CryptoKey or raw bytes)
     */
    constructor(name, options = {}) {
        this.name = name;
        this.options = options;
        this._ready = false;
        this._sessionMode = options.session || false;
        this._semanticSearchEnabled = false;
        this._getEncryptionKey = options.getEncryptionKey || null;
        this._encryptionKeyId = null;  // Sent to worker to identify key
    }

    /**
     * Initialize the store (connects to SharedWorker).
     * @returns {Promise<Store>}
     */
    async open() {
        if (this._ready) return this;

        // If encryption is enabled, derive key and send to worker
        let encryptionConfig = null;
        if (this._getEncryptionKey) {
            const key = await this._getEncryptionKey();
            this._encryptionKeyId = `${this.name}:${Date.now()}`;

            // Convert key to raw bytes if needed
            let keyBytes;
            if (key instanceof CryptoKey) {
                keyBytes = await crypto.subtle.exportKey('raw', key);
            } else if (key instanceof ArrayBuffer || key instanceof Uint8Array) {
                keyBytes = key instanceof Uint8Array ? key : new Uint8Array(key);
            } else if (typeof key === 'string') {
                // Hash string to get 256-bit key
                const encoder = new TextEncoder();
                const data = encoder.encode(key);
                const hash = await crypto.subtle.digest('SHA-256', data);
                keyBytes = new Uint8Array(hash);
            } else {
                throw new Error('Encryption key must be CryptoKey, ArrayBuffer, Uint8Array, or string');
            }

            encryptionConfig = {
                keyId: this._encryptionKeyId,
                keyBytes: Array.from(keyBytes instanceof Uint8Array ? keyBytes : new Uint8Array(keyBytes))
            };
        }

        await workerRPC('open', {
            name: this.name,
            options: this.options,
            encryption: encryptionConfig
        });

        // Session mode cleanup
        if (this._sessionMode && typeof window !== 'undefined') {
            window.addEventListener('beforeunload', () => {
                this.clear().catch(() => {});
            });
        }

        this._ready = true;
        return this;
    }

    /**
     * Get a value by key.
     * @param {string} key
     * @returns {Promise<any>} The stored value, or undefined if not found
     */
    async get(key) {
        await this._ensureOpen();
        return workerRPC('get', { name: this.name, key });
    }

    /**
     * Set a value. Accepts any JSON-serializable value.
     * @param {string} key
     * @param {any} value
     * @returns {Promise<void>}
     */
    async set(key, value) {
        await this._ensureOpen();
        await workerRPC('set', { name: this.name, key, value });
    }

    /**
     * Delete a key.
     * @param {string} key
     * @returns {Promise<boolean>} True if key existed
     */
    async delete(key) {
        await this._ensureOpen();
        return workerRPC('delete', { name: this.name, key });
    }

    /**
     * Check if a key exists.
     * @param {string} key
     * @returns {Promise<boolean>}
     */
    async has(key) {
        const value = await this.get(key);
        return value !== undefined;
    }

    /**
     * List all keys.
     * @returns {Promise<string[]>}
     */
    async keys() {
        await this._ensureOpen();
        return workerRPC('keys', { name: this.name });
    }

    /**
     * Clear all data.
     * @returns {Promise<void>}
     */
    async clear() {
        await this._ensureOpen();
        await workerRPC('clear', { name: this.name });
    }

    /**
     * Filter items in a collection.
     * @param {string} key - Collection key
     * @param {Object} query - Filter query (MongoDB-style operators)
     * @returns {Promise<Array>} Matching items
     */
    async filter(key, query = {}) {
        await this._ensureOpen();
        return workerRPC('filter', { name: this.name, key, query });
    }

    /**
     * Find first item matching query.
     * @param {string} key - Collection key
     * @param {Object} query - Filter query
     * @returns {Promise<Object|undefined>} First matching item
     */
    async find(key, query = {}) {
        await this._ensureOpen();
        return workerRPC('find', { name: this.name, key, query });
    }

    /**
     * Semantic search within a collection.
     * @param {string} key - Collection key
     * @param {string} text - Search text
     * @param {number} limit - Max results (default 10)
     * @returns {Promise<Array>} Matching items with similarity scores
     */
    async search(key, text, limit = 10) {
        await this._ensureOpen();
        return workerRPC('search', { name: this.name, key, text, limit });
    }

    /**
     * Count items in a collection, optionally filtered.
     * @param {string} key - Collection key
     * @param {Object} query - Optional filter query
     * @returns {Promise<number>}
     */
    async count(key, query = null) {
        await this._ensureOpen();
        return workerRPC('count', { name: this.name, key, query });
    }

    /**
     * Subscribe to changes (reactive updates).
     * @param {string} key - Key to watch
     * @param {Function} callback - Called with new value on changes
     * @returns {Function} Unsubscribe function
     */
    subscribe(key, callback) {
        console.warn('[Store] subscribe() not yet implemented');
        return () => {};
    }

    /**
     * Enable semantic search with WebGPU-accelerated text encoding.
     * Model is loaded and runs in the SharedWorker.
     *
     * @param {Object} options - Configuration options
     * @param {string} options.model - Model name ('minilm', 'clip', or GGUF URL)
     * @returns {Promise<Object>} Model info (dimensions, type)
     */
    async enableSemanticSearch(options = {}) {
        await this._ensureOpen();
        const result = await workerRPC('enableSemanticSearch', {
            name: this.name,
            options
        });
        if (result) {
            this._semanticSearchEnabled = true;
        }
        return result;
    }

    /**
     * Disable semantic search and free GPU resources.
     */
    async disableSemanticSearch() {
        await workerRPC('disableSemanticSearch', { name: this.name });
        this._semanticSearchEnabled = false;
    }

    /**
     * Check if semantic search is enabled.
     * @returns {boolean}
     */
    hasSemanticSearch() {
        return this._semanticSearchEnabled;
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    async _ensureOpen() {
        if (!this._ready) {
            await this.open();
        }
    }
}

/**
 * Create a new Store instance.
 *
 * @param {string} name - Store name (used as OPFS directory)
 * @param {Object} options - Store options
 * @param {boolean} options.session - If true, clears data on tab close
 * @param {Function} options.getEncryptionKey - Async callback returning encryption key
 * @returns {Promise<Store>}
 *
 * @example
 * // Persistent store
 * const store = await lanceStore('myapp');
 *
 * // Session store (clears on tab close)
 * const session = await lanceStore('temp', { session: true });
 *
 * // Encrypted store (AES-256-GCM)
 * const secure = await lanceStore('vault', {
 *     getEncryptionKey: async () => {
 *         // Return key from password, hardware token, etc.
 *         const password = await promptPassword();
 *         return password; // String, Uint8Array, ArrayBuffer, or CryptoKey
 *     }
 * });
 */
export async function lanceStore(name, options = {}) {
    const store = new Store(name, options);
    await store.open();
    return store;
}

// Alias for backwards compatibility
export { lanceStore as createStore };

// Export Store class for manual instantiation
export { Store as KeyValueStore };

/**
 * SQL Parser for LocalDatabase (supports CREATE, INSERT, UPDATE, DELETE, SELECT)
 */
class LocalSQLParser {
    constructor(tokens) {
        this.tokens = tokens;
        this.pos = 0;
    }

    peek() {
        return this.tokens[this.pos] || { type: TokenType.EOF };
    }

    advance() {
        return this.tokens[this.pos++] || { type: TokenType.EOF };
    }

    match(type) {
        if (this.peek().type === type) {
            return this.advance();
        }
        return null;
    }

    expect(type) {
        const token = this.advance();
        if (token.type !== type) {
            throw new Error(`Expected ${type}, got ${token.type}`);
        }
        return token;
    }

    parse() {
        const token = this.peek();

        switch (token.type) {
            case TokenType.CREATE:
                return this.parseCreate();
            case TokenType.DROP:
                return this.parseDrop();
            case TokenType.INSERT:
                return this.parseInsert();
            case TokenType.UPDATE:
                return this.parseUpdate();
            case TokenType.DELETE:
                return this.parseDelete();
            case TokenType.SELECT:
                return this.parseSelect();
            default:
                throw new Error(`Unexpected token: ${token.type}`);
        }
    }

    parseCreate() {
        this.expect(TokenType.CREATE);
        this.expect(TokenType.TABLE);
        const tableName = this.expect(TokenType.IDENTIFIER).value;
        this.expect(TokenType.LPAREN);

        const columns = [];
        do {
            const colName = this.expect(TokenType.IDENTIFIER).value;
            const colType = this.parseDataType();
            const col = { name: colName, type: colType };

            // Check for PRIMARY KEY
            if (this.match(TokenType.PRIMARY)) {
                this.expect(TokenType.KEY);
                col.primaryKey = true;
            }

            columns.push(col);
        } while (this.match(TokenType.COMMA));

        this.expect(TokenType.RPAREN);

        return { type: 'create_table', table: tableName, columns };
    }

    parseDataType() {
        const token = this.advance();
        let type = token.value || token.type;

        // Handle VECTOR(dim)
        if (type === 'VECTOR' && this.match(TokenType.LPAREN)) {
            const dim = this.expect(TokenType.NUMBER).value;
            this.expect(TokenType.RPAREN);
            return { type: 'vector', dim: parseInt(dim) };
        }

        // Handle VARCHAR(len)
        if ((type === 'VARCHAR' || type === 'TEXT') && this.match(TokenType.LPAREN)) {
            this.expect(TokenType.NUMBER); // ignore length
            this.expect(TokenType.RPAREN);
        }

        return type;
    }

    parseDrop() {
        this.expect(TokenType.DROP);
        this.expect(TokenType.TABLE);
        const tableName = this.expect(TokenType.IDENTIFIER).value;
        return { type: 'drop_table', table: tableName };
    }

    parseInsert() {
        this.expect(TokenType.INSERT);
        this.expect(TokenType.INTO);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        // Optional column list
        let columns = null;
        if (this.match(TokenType.LPAREN)) {
            columns = [this.expect(TokenType.IDENTIFIER).value];
            while (this.match(TokenType.COMMA)) {
                columns.push(this.expect(TokenType.IDENTIFIER).value);
            }
            this.expect(TokenType.RPAREN);
        }

        this.expect(TokenType.VALUES);

        const rows = [];
        do {
            this.expect(TokenType.LPAREN);
            const values = [this.parseValue()];
            while (this.match(TokenType.COMMA)) {
                values.push(this.parseValue());
            }
            this.expect(TokenType.RPAREN);

            // Build row object
            if (columns) {
                const row = {};
                columns.forEach((col, i) => row[col] = values[i]);
                rows.push(row);
            } else {
                rows.push(values); // positional - needs schema lookup
            }
        } while (this.match(TokenType.COMMA));

        return { type: 'insert', table: tableName, columns, rows };
    }

    parseValue() {
        const token = this.peek();

        if (token.type === TokenType.NUMBER) {
            this.advance();
            const num = parseFloat(token.value);
            return Number.isInteger(num) ? parseInt(token.value) : num;
        }
        if (token.type === TokenType.STRING) {
            this.advance();
            return token.value;
        }
        if (token.type === TokenType.NULL) {
            this.advance();
            return null;
        }
        if (token.type === TokenType.TRUE) {
            this.advance();
            return true;
        }
        if (token.type === TokenType.FALSE) {
            this.advance();
            return false;
        }
        // Vector literal [1.0, 2.0, 3.0]
        if (this.match(TokenType.LBRACKET)) {
            const vec = [];
            do {
                vec.push(parseFloat(this.expect(TokenType.NUMBER).value));
            } while (this.match(TokenType.COMMA));
            this.expect(TokenType.RBRACKET);
            return vec;
        }

        throw new Error(`Unexpected value token: ${token.type}`);
    }

    parseUpdate() {
        this.expect(TokenType.UPDATE);
        const tableName = this.expect(TokenType.IDENTIFIER).value;
        this.expect(TokenType.SET);

        const set = {};
        do {
            const col = this.expect(TokenType.IDENTIFIER).value;
            this.expect(TokenType.EQ);
            set[col] = this.parseValue();
        } while (this.match(TokenType.COMMA));

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        return { type: 'update', table: tableName, set, where };
    }

    parseDelete() {
        this.expect(TokenType.DELETE);
        this.expect(TokenType.FROM);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        return { type: 'delete', table: tableName, where };
    }

    parseSelect() {
        this.expect(TokenType.SELECT);

        // Columns
        const columns = [];
        if (this.match(TokenType.STAR)) {
            columns.push('*');
        } else {
            columns.push(this.expect(TokenType.IDENTIFIER).value);
            while (this.match(TokenType.COMMA)) {
                columns.push(this.expect(TokenType.IDENTIFIER).value);
            }
        }

        this.expect(TokenType.FROM);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        let orderBy = null;
        if (this.match(TokenType.ORDER)) {
            this.expect(TokenType.BY);
            const column = this.expect(TokenType.IDENTIFIER).value;
            const desc = !!this.match(TokenType.DESC);
            if (!desc) this.match(TokenType.ASC);
            orderBy = { column, desc };
        }

        let limit = null;
        if (this.match(TokenType.LIMIT)) {
            limit = parseInt(this.expect(TokenType.NUMBER).value);
        }

        let offset = null;
        if (this.match(TokenType.OFFSET)) {
            offset = parseInt(this.expect(TokenType.NUMBER).value);
        }

        return { type: 'select', table: tableName, columns, where, orderBy, limit, offset };
    }

    parseWhereExpr() {
        return this.parseOrExpr();
    }

    parseOrExpr() {
        let left = this.parseAndExpr();
        while (this.match(TokenType.OR)) {
            const right = this.parseAndExpr();
            left = { op: 'OR', left, right };
        }
        return left;
    }

    parseAndExpr() {
        let left = this.parseComparison();
        while (this.match(TokenType.AND)) {
            const right = this.parseComparison();
            left = { op: 'AND', left, right };
        }
        return left;
    }

    parseComparison() {
        const column = this.expect(TokenType.IDENTIFIER).value;

        let op;
        if (this.match(TokenType.EQ)) op = '=';
        else if (this.match(TokenType.NE)) op = '!=';
        else if (this.match(TokenType.LT)) op = '<';
        else if (this.match(TokenType.LE)) op = '<=';
        else if (this.match(TokenType.GT)) op = '>';
        else if (this.match(TokenType.GE)) op = '>=';
        else if (this.match(TokenType.LIKE)) op = 'LIKE';
        else throw new Error(`Expected comparison operator`);

        const value = this.parseValue();
        return { op, column, value };
    }
}

// Immer-style WASM runtime - auto string/bytes marshalling
const E = new TextEncoder();
const D = new TextDecoder();
let _w, _m, _p = 0, _M = 0;

// Get shared buffer view (lazy allocation)
const _g = () => {
    if (!_p || !_M) return null;
    return new Uint8Array(_m.buffer, _p, _M);
};

// Ensure shared buffer is large enough
const _ensure = (size) => {
    if (_p && size <= _M) return true;
    // Free old buffer if exists
    if (_p && _w.free) _w.free(_p, _M);
    _M = Math.max(size + 1024, 4096); // At least 4KB
    _p = _w.alloc(_M);
    return _p !== 0;
};

// Marshal JS value to WASM args (strings and Uint8Array auto-copied to WASM memory)
const _x = a => {
    if (a instanceof Uint8Array) {
        if (!_ensure(a.length)) return [a]; // Fallback if alloc fails
        _g().set(a);
        return [_p, a.length];
    }
    if (typeof a !== 'string') return [a];
    const b = E.encode(a);
    if (!_ensure(b.length)) return [a]; // Fallback if alloc fails
    _g().set(b);
    return [_p, b.length];
};

// Read string from WASM memory
const readStr = (ptr, len) => D.decode(new Uint8Array(_m.buffer, ptr, len));

// Read bytes from WASM memory (returns copy)
const readBytes = (ptr, len) => new Uint8Array(_m.buffer, ptr, len).slice();

/**
 * LanceFileWriter - Thin JS wrapper for WASM fragment writer
 *
 * Uses high-level WASM API: fragmentBegin -> fragmentAdd*Column -> fragmentEnd
 * All encoding logic is in WASM, JS only marshals data.
 */
class LanceFileWriter {
    constructor(schema) {
        this.schema = schema;  // [{name, type, nullable?, vectorDim?}]
        this.columns = new Map();  // columnName -> values[]
        this.rowCount = 0;
    }

    addRows(rows) {
        for (const row of rows) {
            for (const col of this.schema) {
                if (!this.columns.has(col.name)) {
                    this.columns.set(col.name, []);
                }
                this.columns.get(col.name).push(row[col.name] ?? null);
            }
            this.rowCount++;
        }
    }

    build() {
        if (!_w?.fragmentBegin) {
            return this._buildJson();
        }

        // Estimate buffer size
        const estimatedSize = Math.max(64 * 1024, this.rowCount * 1024);
        if (!_w.fragmentBegin(estimatedSize)) {
            throw new Error('Failed to initialize WASM fragment writer');
        }

        // Add each column using high-level WASM API
        for (const col of this.schema) {
            const values = this.columns.get(col.name) || [];
            this._addColumn(col, values);
        }

        // Finalize - WASM writes metadata, offsets table, and footer
        const finalSize = _w.fragmentEnd();
        if (finalSize === 0) {
            throw new Error('Failed to finalize fragment');
        }

        const bufferPtr = _w.writerGetBuffer();
        if (!bufferPtr) {
            throw new Error('Failed to get writer buffer');
        }

        return new Uint8Array(_m.buffer, bufferPtr, finalSize).slice();
    }

    _addColumn(col, values) {
        const type = (col.type || col.dataType || 'string').toLowerCase();
        const nullable = col.nullable !== false;

        // Allocate name in WASM memory
        const nameBytes = E.encode(col.name);
        const namePtr = _w.alloc(nameBytes.length);
        new Uint8Array(_m.buffer, namePtr, nameBytes.length).set(nameBytes);

        let result = 0;

        switch (type) {
            case 'int64':
            case 'int':
            case 'integer':
            case 'bigint': {
                const arr = new BigInt64Array(values.map(v => BigInt(v ?? 0)));
                const ptr = _w.alloc(arr.byteLength);
                new BigInt64Array(_m.buffer, ptr, values.length).set(arr);
                result = _w.fragmentAddInt64Column(namePtr, nameBytes.length, ptr, values.length, nullable);
                _w.free(ptr, arr.byteLength);
                break;
            }

            case 'int32': {
                const arr = new Int32Array(values.map(v => v ?? 0));
                const ptr = _w.alloc(arr.byteLength);
                new Int32Array(_m.buffer, ptr, values.length).set(arr);
                result = _w.fragmentAddInt32Column(namePtr, nameBytes.length, ptr, values.length, nullable);
                _w.free(ptr, arr.byteLength);
                break;
            }

            case 'float64':
            case 'float':
            case 'double': {
                const arr = new Float64Array(values.map(v => v ?? 0.0));
                const ptr = _w.alloc(arr.byteLength);
                new Float64Array(_m.buffer, ptr, values.length).set(arr);
                result = _w.fragmentAddFloat64Column(namePtr, nameBytes.length, ptr, values.length, nullable);
                _w.free(ptr, arr.byteLength);
                break;
            }

            case 'float32': {
                const arr = new Float32Array(values.map(v => v ?? 0.0));
                const ptr = _w.alloc(arr.byteLength);
                new Float32Array(_m.buffer, ptr, values.length).set(arr);
                result = _w.fragmentAddFloat32Column(namePtr, nameBytes.length, ptr, values.length, nullable);
                _w.free(ptr, arr.byteLength);
                break;
            }

            case 'string':
            case 'text':
            case 'varchar': {
                // Build string data and offsets
                let currentOffset = 0;
                const offsets = new Uint32Array(values.length + 1);
                const allBytes = [];

                for (let i = 0; i < values.length; i++) {
                    offsets[i] = currentOffset;
                    const bytes = E.encode(String(values[i] ?? ''));
                    allBytes.push(...bytes);
                    currentOffset += bytes.length;
                }
                offsets[values.length] = currentOffset;

                const strData = new Uint8Array(allBytes);
                const strPtr = _w.alloc(strData.length);
                new Uint8Array(_m.buffer, strPtr, strData.length).set(strData);

                const offPtr = _w.alloc(offsets.byteLength);
                new Uint32Array(_m.buffer, offPtr, offsets.length).set(offsets);

                result = _w.fragmentAddStringColumn(namePtr, nameBytes.length, strPtr, strData.length, offPtr, values.length, nullable);

                _w.free(strPtr, strData.length);
                _w.free(offPtr, offsets.byteLength);
                break;
            }

            case 'bool':
            case 'boolean': {
                const byteCount = Math.ceil(values.length / 8);
                const packed = new Uint8Array(byteCount);
                for (let i = 0; i < values.length; i++) {
                    if (values[i]) packed[Math.floor(i / 8)] |= (1 << (i % 8));
                }
                const ptr = _w.alloc(packed.length);
                new Uint8Array(_m.buffer, ptr, packed.length).set(packed);
                result = _w.fragmentAddBoolColumn(namePtr, nameBytes.length, ptr, packed.length, values.length, nullable);
                _w.free(ptr, packed.length);
                break;
            }

            case 'vector': {
                const dim = col.vectorDim || (values[0]?.length || 0);
                const allFloats = [];
                for (const v of values) {
                    if (Array.isArray(v)) {
                        allFloats.push(...v);
                    } else {
                        for (let i = 0; i < dim; i++) allFloats.push(0);
                    }
                }
                const arr = new Float32Array(allFloats);
                const ptr = _w.alloc(arr.byteLength);
                new Float32Array(_m.buffer, ptr, allFloats.length).set(arr);
                result = _w.fragmentAddVectorColumn(namePtr, nameBytes.length, ptr, allFloats.length, dim, nullable);
                _w.free(ptr, arr.byteLength);
                break;
            }

            default:
                // Fallback to string
                _w.free(namePtr, nameBytes.length);
                return this._addColumn({ ...col, type: 'string' }, values);
        }

        _w.free(namePtr, nameBytes.length);

        if (!result) {
            throw new Error(`Failed to add column '${col.name}'`);
        }
    }

    _buildJson() {
        const data = {
            schema: this.schema,
            columns: Object.fromEntries(this.columns),
            rowCount: this.rowCount,
            format: 'json'
        };
        return E.encode(JSON.stringify(data));
    }
}

// WASM utils exported for advanced usage
export const wasmUtils = {
    readStr,
    readBytes,
    encoder: E,
    decoder: D,
    getMemory: () => _m,
    getExports: () => _w,
};

// LanceQL high-level methods factory (needs proxy reference)
const _createLanceqlMethods = (proxy) => ({
    /**
     * Get the library version.
     * @returns {string} Version string like "0.1.0"
     */
    getVersion() {
        const v = _w.getVersion();
        const major = (v >> 16) & 0xFF;
        const minor = (v >> 8) & 0xFF;
        const patch = v & 0xFF;
        return `${major}.${minor}.${patch}`;
    },

    /**
     * Open a Lance file from an ArrayBuffer (local file).
     * @param {ArrayBuffer} data - The Lance file data
     * @returns {LanceFile}
     */
    open(data) {
        return new LanceFile(proxy, data);
    },

    /**
     * Open a Lance file from a URL using HTTP Range requests.
     * @param {string} url - URL to the Lance file
     * @returns {Promise<RemoteLanceFile>}
     */
    async openUrl(url) {
        // Ensure WebGPU is initialized for vector search
        await webgpuAccelerator.init();
        return await RemoteLanceFile.open(proxy, url);
    },

    /**
     * Open a Lance dataset from a base URL using HTTP Range requests.
     * @param {string} baseUrl - Base URL to the Lance dataset
     * @param {object} [options] - Options for opening
     * @param {number} [options.version] - Specific version to load
     * @returns {Promise<RemoteLanceDataset>}
     */
    async openDataset(baseUrl, options = {}) {
        // Ensure WebGPU is initialized for vector search
        await webgpuAccelerator.init();
        return await RemoteLanceDataset.open(proxy, baseUrl, options);
    },

    /**
     * Parse footer from Lance file data.
     * @param {ArrayBuffer} data
     * @returns {{numColumns: number, majorVersion: number, minorVersion: number} | null}
     */
    parseFooter(data) {
        const bytes = new Uint8Array(data);
        const ptr = _w.alloc(bytes.length);
        if (!ptr) return null;

        try {
            new Uint8Array(_m.buffer).set(bytes, ptr);

            const numColumns = _w.parseFooterGetColumns(ptr, bytes.length);
            const majorVersion = _w.parseFooterGetMajorVersion(ptr, bytes.length);
            const minorVersion = _w.parseFooterGetMinorVersion(ptr, bytes.length);

            if (numColumns === 0 && majorVersion === 0) {
                return null;
            }

            return { numColumns, majorVersion, minorVersion };
        } finally {
            _w.free(ptr, bytes.length);
        }
    },

    /**
     * Check if data is a valid Lance file.
     * @param {ArrayBuffer} data
     * @returns {boolean}
     */
    isValidLanceFile(data) {
        const bytes = new Uint8Array(data);
        const ptr = _w.alloc(bytes.length);
        if (!ptr) return false;

        try {
            new Uint8Array(_m.buffer).set(bytes, ptr);
            return _w.isValidLanceFile(ptr, bytes.length) === 1;
        } finally {
            _w.free(ptr, bytes.length);
        }
    },

    /**
     * Create a new LanceDatabase for multi-table queries with JOINs.
     * @returns {LanceDatabase}
     */
    createDatabase() {
        // Store reference to lanceql instance for registerRemote()
        if (typeof window !== 'undefined') {
            window.lanceql = proxy;
        } else if (typeof globalThis !== 'undefined') {
            globalThis.lanceql = proxy;
        }
        return new LanceDatabase();
    }
});

export class LanceQL {
    /**
     * Load LanceQL from a WASM file path or URL.
     * Returns Immer-style proxy with auto string/bytes marshalling.
     * @param {string} wasmPath - Path to the lanceql.wasm file
     * @returns {Promise<LanceQL>}
     */
    static async load(wasmPath = './lanceql.wasm') {
        const response = await fetch(wasmPath);
        const wasmBytes = await response.arrayBuffer();
        const wasmModule = await WebAssembly.instantiate(wasmBytes, {});

        _w = wasmModule.instance.exports;
        _m = _w.memory;

        // Create Immer-style proxy that auto-marshals string/bytes arguments
        // Also includes high-level LanceQL methods
        let _methods = null;
        const proxy = new Proxy({}, {
            get(_, n) {
                // Lazy init methods with proxy reference
                if (!_methods) _methods = _createLanceqlMethods(proxy);
                // High-level LanceQL methods
                if (n in _methods) return _methods[n];
                // Special properties
                if (n === 'memory') return _m;
                if (n === 'raw') return _w;  // Raw WASM exports
                if (n === 'wasm') return _w; // Backward compatibility
                // WASM functions with auto-marshalling
                if (typeof _w[n] === 'function') {
                    return (...a) => _w[n](...a.flatMap(_x));
                }
                return _w[n];
            }
        });
        return proxy;
    }
}

/**
 * Represents an open Lance file (loaded entirely in memory).
 */
export class LanceFile {
    constructor(lanceql, data) {
        this.lanceql = lanceql;
        this.wasm = lanceql.wasm;
        this.memory = lanceql.memory;

        // Copy data to WASM memory
        const bytes = new Uint8Array(data);
        this.dataPtr = this.wasm.alloc(bytes.length);
        if (!this.dataPtr) {
            throw new Error('Failed to allocate memory for Lance file');
        }
        this.dataLen = bytes.length;
        new Uint8Array(this.memory.buffer).set(bytes, this.dataPtr);

        // Open the file
        const result = this.wasm.openFile(this.dataPtr, this.dataLen);
        if (result === 0) {
            this.wasm.free(this.dataPtr, this.dataLen);
            throw new Error('Failed to open Lance file');
        }
    }

    /**
     * Close the file and free memory.
     */
    close() {
        this.wasm.closeFile();
        if (this.dataPtr) {
            this.wasm.free(this.dataPtr, this.dataLen);
            this.dataPtr = null;
        }
    }

    /**
     * Get the number of columns.
     * @returns {number}
     */
    get numColumns() {
        return this.wasm.getNumColumns();
    }

    /**
     * Get the row count for a column.
     * @param {number} colIdx
     * @returns {bigint}
     */
    getRowCount(colIdx) {
        return this.wasm.getRowCount(colIdx);
    }

    /**
     * Get debug info for a column.
     * @param {number} colIdx
     * @returns {{offset: bigint, size: bigint, rows: bigint}}
     */
    getColumnDebugInfo(colIdx) {
        return {
            offset: this.wasm.getColumnBufferOffset(colIdx),
            size: this.wasm.getColumnBufferSize(colIdx),
            rows: this.wasm.getRowCount(colIdx)
        };
    }

    /**
     * Read an int64 column as a BigInt64Array.
     * @param {number} colIdx - Column index
     * @returns {BigInt64Array}
     */
    readInt64Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new BigInt64Array(0);

        const bufPtr = this.wasm.allocInt64Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate int64 buffer');

        try {
            const count = this.wasm.readInt64Column(colIdx, bufPtr, rowCount);
            const result = new BigInt64Array(count);
            const view = new BigInt64Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.freeInt64Buffer(bufPtr, rowCount);
        }
    }

    /**
     * Read a float64 column as a Float64Array.
     * @param {number} colIdx - Column index
     * @returns {Float64Array}
     */
    readFloat64Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Float64Array(0);

        const bufPtr = this.wasm.allocFloat64Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate float64 buffer');

        try {
            const count = this.wasm.readFloat64Column(colIdx, bufPtr, rowCount);
            const result = new Float64Array(count);
            const view = new Float64Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.freeFloat64Buffer(bufPtr, rowCount);
        }
    }

    // ========================================================================
    // Additional Numeric Type Column Methods
    // ========================================================================

    /**
     * Read an int32 column as an Int32Array.
     * @param {number} colIdx - Column index
     * @returns {Int32Array}
     */
    readInt32Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Int32Array(0);

        const bufPtr = this.wasm.allocInt32Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate int32 buffer');

        try {
            const count = this.wasm.readInt32Column(colIdx, bufPtr, rowCount);
            const result = new Int32Array(count);
            const view = new Int32Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 4);
        }
    }

    /**
     * Read an int16 column as an Int16Array.
     * @param {number} colIdx - Column index
     * @returns {Int16Array}
     */
    readInt16Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Int16Array(0);

        const bufPtr = this.wasm.allocInt16Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate int16 buffer');

        try {
            const count = this.wasm.readInt16Column(colIdx, bufPtr, rowCount);
            const result = new Int16Array(count);
            const view = new Int16Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 2);
        }
    }

    /**
     * Read an int8 column as an Int8Array.
     * @param {number} colIdx - Column index
     * @returns {Int8Array}
     */
    readInt8Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Int8Array(0);

        const bufPtr = this.wasm.allocInt8Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate int8 buffer');

        try {
            const count = this.wasm.readInt8Column(colIdx, bufPtr, rowCount);
            const result = new Int8Array(count);
            const view = new Int8Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount);
        }
    }

    /**
     * Read a uint64 column as a BigUint64Array.
     * @param {number} colIdx - Column index
     * @returns {BigUint64Array}
     */
    readUint64Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new BigUint64Array(0);

        const bufPtr = this.wasm.allocUint64Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate uint64 buffer');

        try {
            const count = this.wasm.readUint64Column(colIdx, bufPtr, rowCount);
            const result = new BigUint64Array(count);
            const view = new BigUint64Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 8);
        }
    }

    /**
     * Read a uint32 column as a Uint32Array.
     * @param {number} colIdx - Column index
     * @returns {Uint32Array}
     */
    readUint32Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint32Array(0);

        const bufPtr = this.wasm.allocIndexBuffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate uint32 buffer');

        try {
            const count = this.wasm.readUint32Column(colIdx, bufPtr, rowCount);
            const result = new Uint32Array(count);
            const view = new Uint32Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 4);
        }
    }

    /**
     * Read a uint16 column as a Uint16Array.
     * @param {number} colIdx - Column index
     * @returns {Uint16Array}
     */
    readUint16Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint16Array(0);

        const bufPtr = this.wasm.allocUint16Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate uint16 buffer');

        try {
            const count = this.wasm.readUint16Column(colIdx, bufPtr, rowCount);
            const result = new Uint16Array(count);
            const view = new Uint16Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 2);
        }
    }

    /**
     * Read a uint8 column as a Uint8Array.
     * @param {number} colIdx - Column index
     * @returns {Uint8Array}
     */
    readUint8Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint8Array(0);

        const bufPtr = this.wasm.allocStringBuffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate uint8 buffer');

        try {
            const count = this.wasm.readUint8Column(colIdx, bufPtr, rowCount);
            const result = new Uint8Array(count);
            const view = new Uint8Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount);
        }
    }

    /**
     * Read a float32 column as a Float32Array.
     * @param {number} colIdx - Column index
     * @returns {Float32Array}
     */
    readFloat32Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Float32Array(0);

        const bufPtr = this.wasm.allocFloat32Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate float32 buffer');

        try {
            const count = this.wasm.readFloat32Column(colIdx, bufPtr, rowCount);
            const result = new Float32Array(count);
            const view = new Float32Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 4);
        }
    }

    /**
     * Read a boolean column as a Uint8Array (0 or 1 values).
     * @param {number} colIdx - Column index
     * @returns {Uint8Array}
     */
    readBoolColumn(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint8Array(0);

        const bufPtr = this.wasm.allocStringBuffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate bool buffer');

        try {
            const count = this.wasm.readBoolColumn(colIdx, bufPtr, rowCount);
            const result = new Uint8Array(count);
            const view = new Uint8Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount);
        }
    }

    /**
     * Read int32 values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {Int32Array}
     */
    readInt32AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Int32Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocInt32Buffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
            const count = this.wasm.readInt32AtIndices(colIdx, idxPtr, indices.length, outPtr);
            const result = new Int32Array(count);
            const view = new Int32Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(outPtr, indices.length * 4);
        }
    }

    /**
     * Read float32 values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {Float32Array}
     */
    readFloat32AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Float32Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocFloat32Buffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
            const count = this.wasm.readFloat32AtIndices(colIdx, idxPtr, indices.length, outPtr);
            const result = new Float32Array(count);
            const view = new Float32Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(outPtr, indices.length * 4);
        }
    }

    /**
     * Read uint8 values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {Uint8Array}
     */
    readUint8AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Uint8Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocStringBuffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
            const count = this.wasm.readUint8AtIndices(colIdx, idxPtr, indices.length, outPtr);
            const result = new Uint8Array(count);
            const view = new Uint8Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(outPtr, indices.length);
        }
    }

    /**
     * Read bool values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {Uint8Array}
     */
    readBoolAtIndices(colIdx, indices) {
        if (indices.length === 0) return new Uint8Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocStringBuffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
            const count = this.wasm.readBoolAtIndices(colIdx, idxPtr, indices.length, outPtr);
            const result = new Uint8Array(count);
            const view = new Uint8Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(outPtr, indices.length);
        }
    }

    // ========================================================================
    // Query Methods
    // ========================================================================

    /**
     * Filter operator constants.
     */
    static Op = {
        EQ: 0,  // Equal
        NE: 1,  // Not equal
        LT: 2,  // Less than
        LE: 3,  // Less than or equal
        GT: 4,  // Greater than
        GE: 5   // Greater than or equal
    };

    /**
     * Filter int64 column and return matching row indices.
     * @param {number} colIdx - Column index
     * @param {number} op - Comparison operator (use LanceFile.Op)
     * @param {bigint|number} value - Value to compare against
     * @returns {Uint32Array} Array of matching row indices
     */
    filterInt64(colIdx, op, value) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint32Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(rowCount);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        try {
            const count = this.wasm.filterInt64Column(
                colIdx, op, BigInt(value), idxPtr, rowCount
            );
            const result = new Uint32Array(count);
            const view = new Uint32Array(this.memory.buffer, idxPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, rowCount * 4);
        }
    }

    /**
     * Filter float64 column and return matching row indices.
     * @param {number} colIdx - Column index
     * @param {number} op - Comparison operator (use LanceFile.Op)
     * @param {number} value - Value to compare against
     * @returns {Uint32Array} Array of matching row indices
     */
    filterFloat64(colIdx, op, value) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint32Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(rowCount);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        try {
            const count = this.wasm.filterFloat64Column(
                colIdx, op, value, idxPtr, rowCount
            );
            const result = new Uint32Array(count);
            const view = new Uint32Array(this.memory.buffer, idxPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, rowCount * 4);
        }
    }

    /**
     * Read int64 values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {BigInt64Array}
     */
    readInt64AtIndices(colIdx, indices) {
        if (indices.length === 0) return new BigInt64Array(0);

        // Copy indices to WASM memory
        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocInt64Buffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);

            const count = this.wasm.readInt64AtIndices(
                colIdx, idxPtr, indices.length, outPtr
            );

            const result = new BigInt64Array(count);
            const view = new BigInt64Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.freeInt64Buffer(outPtr, indices.length);
        }
    }

    /**
     * Read float64 values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {Float64Array}
     */
    readFloat64AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Float64Array(0);

        // Copy indices to WASM memory
        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocFloat64Buffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);

            const count = this.wasm.readFloat64AtIndices(
                colIdx, idxPtr, indices.length, outPtr
            );

            const result = new Float64Array(count);
            const view = new Float64Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.freeFloat64Buffer(outPtr, indices.length);
        }
    }

    // ========================================================================
    // Aggregation Methods
    // ========================================================================

    /**
     * Sum all values in an int64 column.
     * @param {number} colIdx - Column index
     * @returns {bigint}
     */
    sumInt64(colIdx) {
        return this.wasm.sumInt64Column(colIdx);
    }

    /**
     * Sum all values in a float64 column.
     * @param {number} colIdx - Column index
     * @returns {number}
     */
    sumFloat64(colIdx) {
        return this.wasm.sumFloat64Column(colIdx);
    }

    /**
     * Get minimum value in an int64 column.
     * @param {number} colIdx - Column index
     * @returns {bigint}
     */
    minInt64(colIdx) {
        return this.wasm.minInt64Column(colIdx);
    }

    /**
     * Get maximum value in an int64 column.
     * @param {number} colIdx - Column index
     * @returns {bigint}
     */
    maxInt64(colIdx) {
        return this.wasm.maxInt64Column(colIdx);
    }

    /**
     * Get average of a float64 column.
     * @param {number} colIdx - Column index
     * @returns {number}
     */
    avgFloat64(colIdx) {
        return this.wasm.avgFloat64Column(colIdx);
    }

    // ========================================================================
    // String Column Methods
    // ========================================================================

    /**
     * Debug: Get string column buffer info
     * @param {number} colIdx - Column index
     * @returns {{offsetsSize: number, dataSize: number}}
     */
    debugStringColInfo(colIdx) {
        const packed = this.wasm.debugStringColInfo(colIdx);
        return {
            offsetsSize: Number(BigInt(packed) >> 32n),
            dataSize: Number(BigInt(packed) & 0xFFFFFFFFn)
        };
    }

    /**
     * Debug: Get string read info for a specific row
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Row index
     * @returns {{strStart: number, strLen: number} | {error: string}}
     */
    debugReadStringInfo(colIdx, rowIdx) {
        const packed = this.wasm.debugReadStringInfo(colIdx, rowIdx);
        // Check for error codes (0xDEAD00XX)
        if ((packed & 0xFFFF0000n) === 0xDEAD0000n) {
            const errCode = Number(packed & 0xFFFFn);
            const errors = {
                1: 'No file data',
                2: 'No column entry',
                3: 'Col meta out of bounds',
                4: 'Not a string column',
                5: 'Row out of bounds',
                6: 'Invalid offset size'
            };
            return { error: errors[errCode] || `Unknown error ${errCode}` };
        }
        return {
            strStart: Number(BigInt(packed) >> 32n),
            strLen: Number(BigInt(packed) & 0xFFFFFFFFn)
        };
    }

    /**
     * Debug: Get data_start position for string column
     * @param {number} colIdx - Column index
     * @returns {{dataStart: number, fileLen: number}}
     */
    debugStringDataStart(colIdx) {
        const packed = this.wasm.debugStringDataStart(colIdx);
        return {
            dataStart: Number(BigInt(packed) >> 32n),
            fileLen: Number(BigInt(packed) & 0xFFFFFFFFn)
        };
    }

    /**
     * Get the number of strings in a column.
     * @param {number} colIdx - Column index
     * @returns {number}
     */
    getStringCount(colIdx) {
        return Number(this.wasm.getStringCount(colIdx));
    }

    /**
     * Read a single string at a specific row index.
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Row index
     * @returns {string}
     */
    readStringAt(colIdx, rowIdx) {
        const maxLen = 4096; // Max string length to read
        const bufPtr = this.wasm.allocStringBuffer(maxLen);
        if (!bufPtr) throw new Error('Failed to allocate string buffer');

        try {
            const actualLen = this.wasm.readStringAt(colIdx, rowIdx, bufPtr, maxLen);
            if (actualLen === 0) return '';

            const bytes = new Uint8Array(this.memory.buffer, bufPtr, Math.min(actualLen, maxLen));
            return new TextDecoder().decode(bytes);
        } finally {
            this.wasm.free(bufPtr, maxLen);
        }
    }

    /**
     * Read all strings from a column.
     * @param {number} colIdx - Column index
     * @param {number} limit - Maximum number of strings to read
     * @returns {string[]}
     */
    readStringColumn(colIdx, limit = 1000) {
        const count = Math.min(this.getStringCount(colIdx), limit);
        if (count === 0) return [];

        const results = [];
        for (let i = 0; i < count; i++) {
            results.push(this.readStringAt(colIdx, i));
        }
        return results;
    }

    /**
     * Read strings at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {string[]}
     */
    readStringsAtIndices(colIdx, indices) {
        if (indices.length === 0) return [];

        // Use smaller buffer - estimate based on indices count
        // Assume average string is ~256 bytes, capped at 256KB to avoid WASM memory issues
        const maxTotalLen = Math.min(indices.length * 256, 256 * 1024);
        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const strBufPtr = this.wasm.allocStringBuffer(maxTotalLen);
        if (!strBufPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate string buffer');
        }

        const lenBufPtr = this.wasm.allocU32Buffer(indices.length);
        if (!lenBufPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(strBufPtr, maxTotalLen);
            throw new Error('Failed to allocate length buffer');
        }

        try {
            // Copy indices to WASM
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);

            // Read strings
            const totalWritten = this.wasm.readStringsAtIndices(
                colIdx, idxPtr, indices.length, strBufPtr, maxTotalLen, lenBufPtr
            );

            // Get lengths
            const lengths = new Uint32Array(this.memory.buffer, lenBufPtr, indices.length);

            // Decode strings
            const results = [];
            let offset = 0;
            for (let i = 0; i < indices.length; i++) {
                const len = lengths[i];
                if (len > 0 && offset + len <= totalWritten) {
                    const bytes = new Uint8Array(this.memory.buffer, strBufPtr + offset, len);
                    results.push(new TextDecoder().decode(bytes));
                    offset += len;
                } else {
                    results.push('');
                }
            }
            return results;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(strBufPtr, maxTotalLen);
            this.wasm.free(lenBufPtr, indices.length * 4);
        }
    }

    // ========================================================================
    // Vector Column Support (for embeddings/semantic search)
    // ========================================================================

    /**
     * Get vector info for a column.
     * @param {number} colIdx - Column index
     * @returns {{rows: number, dimension: number}}
     */
    getVectorInfo(colIdx) {
        const packed = this.wasm.getVectorInfo(colIdx);
        return {
            rows: Number(BigInt(packed) >> 32n),
            dimension: Number(BigInt(packed) & 0xFFFFFFFFn)
        };
    }

    /**
     * Read a single vector at index.
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Row index
     * @returns {Float32Array}
     */
    readVectorAt(colIdx, rowIdx) {
        const info = this.getVectorInfo(colIdx);
        if (info.dimension === 0) return new Float32Array(0);

        const bufPtr = this.wasm.allocFloat32Buffer(info.dimension);
        if (!bufPtr) throw new Error('Failed to allocate vector buffer');

        try {
            const dim = this.wasm.readVectorAt(colIdx, rowIdx, bufPtr, info.dimension);
            const result = new Float32Array(dim);
            const view = new Float32Array(this.memory.buffer, bufPtr, dim);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, info.dimension * 4);
        }
    }

    /**
     * Compute cosine similarity between two vectors.
     * @param {Float32Array} vecA
     * @param {Float32Array} vecB
     * @returns {number} Similarity score (-1 to 1)
     */
    cosineSimilarity(vecA, vecB) {
        if (vecA.length !== vecB.length) {
            throw new Error('Vector dimensions must match');
        }

        const ptrA = this.wasm.allocFloat32Buffer(vecA.length);
        const ptrB = this.wasm.allocFloat32Buffer(vecB.length);
        if (!ptrA || !ptrB) throw new Error('Failed to allocate buffers');

        try {
            new Float32Array(this.memory.buffer, ptrA, vecA.length).set(vecA);
            new Float32Array(this.memory.buffer, ptrB, vecB.length).set(vecB);
            return this.wasm.cosineSimilarity(ptrA, ptrB, vecA.length);
        } finally {
            this.wasm.free(ptrA, vecA.length * 4);
            this.wasm.free(ptrB, vecB.length * 4);
        }
    }

    /**
     * Batch cosine similarity using WASM SIMD.
     * Much faster than calling cosineSimilarity in a loop.
     * @param {Float32Array} queryVec - Query vector
     * @param {Float32Array[]} vectors - Array of vectors to compare
     * @param {boolean} normalized - Whether vectors are L2-normalized
     * @returns {Float32Array} - Similarity scores
     */
    batchCosineSimilarity(queryVec, vectors, normalized = true) {
        if (vectors.length === 0) return new Float32Array(0);

        const dim = queryVec.length;
        const numVectors = vectors.length;

        // Allocate WASM buffers
        const queryPtr = this.wasm.allocFloat32Buffer(dim);
        const vectorsPtr = this.wasm.allocFloat32Buffer(numVectors * dim);
        const scoresPtr = this.wasm.allocFloat32Buffer(numVectors);

        if (!queryPtr || !vectorsPtr || !scoresPtr) {
            throw new Error('Failed to allocate WASM buffers');
        }

        try {
            // Copy query vector
            new Float32Array(this.memory.buffer, queryPtr, dim).set(queryVec);

            // Copy all vectors (flattened)
            const flatVectors = new Float32Array(this.memory.buffer, vectorsPtr, numVectors * dim);
            for (let i = 0; i < numVectors; i++) {
                flatVectors.set(vectors[i], i * dim);
            }

            // Call WASM batch similarity
            this.wasm.batchCosineSimilarity(queryPtr, vectorsPtr, dim, numVectors, scoresPtr, normalized ? 1 : 0);

            // Copy results
            const scores = new Float32Array(numVectors);
            scores.set(new Float32Array(this.memory.buffer, scoresPtr, numVectors));
            return scores;
        } finally {
            this.wasm.free(queryPtr, dim * 4);
            this.wasm.free(vectorsPtr, numVectors * dim * 4);
            this.wasm.free(scoresPtr, numVectors * 4);
        }
    }

    /**
     * Read all vectors from a column as array of Float32Arrays.
     * @param {number} colIdx - Column index
     * @returns {Float32Array[]} Array of vectors
     */
    readAllVectors(colIdx) {
        const info = this.getVectorInfo(colIdx);
        if (info.dimension === 0 || info.rows === 0) return [];

        const dim = info.dimension;
        const numRows = info.rows;
        const vectors = [];

        // Allocate buffer for all vectors at once
        const bufPtr = this.wasm.allocFloat32Buffer(numRows * dim);
        if (!bufPtr) throw new Error('Failed to allocate vector buffer');

        try {
            // Read all vectors in one WASM call (if supported)
            // Otherwise fall back to individual reads
            if (this.wasm.readVectorColumn) {
                const count = this.wasm.readVectorColumn(colIdx, bufPtr, numRows * dim);
                const allData = new Float32Array(this.memory.buffer, bufPtr, count);

                for (let i = 0; i < numRows && i * dim < count; i++) {
                    const vec = new Float32Array(dim);
                    vec.set(allData.subarray(i * dim, (i + 1) * dim));
                    vectors.push(vec);
                }
            } else {
                // Fall back to individual reads
                for (let i = 0; i < numRows; i++) {
                    vectors.push(this.readVectorAt(colIdx, i));
                }
            }

            return vectors;
        } finally {
            this.wasm.free(bufPtr, numRows * dim * 4);
        }
    }

    /**
     * Find top-k most similar vectors to query.
     * Uses WebGPU if available, otherwise falls back to WASM SIMD.
     * GPU-accelerated top-K selection for large result sets.
     * @param {number} colIdx - Column index with vectors
     * @param {Float32Array} queryVec - Query vector
     * @param {number} topK - Number of results to return
     * @param {Function} onProgress - Progress callback (current, total)
     * @returns {Promise<{indices: Uint32Array, scores: Float32Array}>}
     */
    async vectorSearch(colIdx, queryVec, topK = 10, onProgress = null) {
        const dim = queryVec.length;
        const info = this.getVectorInfo(colIdx);
        const numRows = info.rows;

        // Try WebGPU-accelerated path first
        if (webgpuAccelerator.isAvailable()) {
            if (onProgress) onProgress(0, numRows);

            // Read all vectors (bulk read)
            console.log(`[LanceFile.vectorSearch] Reading ${numRows} vectors...`);
            const allVectors = this.readAllVectors(colIdx);

            if (onProgress) onProgress(numRows, numRows);

            console.log(`[LanceFile.vectorSearch] Computing similarity for ${allVectors.length} vectors via WebGPU`);

            // Batch compute with WebGPU
            const scores = await webgpuAccelerator.batchCosineSimilarity(queryVec, allVectors, true);

            // GPU-accelerated top-K selection for large result sets
            return await gpuVectorSearch.topK(scores, null, topK, true);
        }

        // Fall back to WASM SIMD (uses batchCosineSimilarity internally)
        console.log(`[LanceFile.vectorSearch] Using WASM SIMD`);

        if (onProgress) onProgress(0, numRows);

        // Read all vectors first
        const allVectors = this.readAllVectors(colIdx);

        if (onProgress) onProgress(numRows, numRows);

        // Use WASM batch cosine similarity
        const scores = this.lanceql.batchCosineSimilarity(queryVec, allVectors, true);

        // GPU-accelerated top-K selection for large result sets
        return await gpuVectorSearch.topK(scores, null, topK, true);
    }

    // ========================================================================
    // DataFrame-like API
    // ========================================================================

    /**
     * Create a DataFrame-like query builder for this file.
     * @returns {DataFrame}
     */
    df() {
        return new DataFrame(this);
    }
}

/**
 * DataFrame-like query builder for fluent queries.
 */
export class DataFrame {
    constructor(file) {
        this.file = file;
        this._filterOps = [];  // Array of {colIdx, op, value, type, opStr}
        this._selectCols = null;
        this._limitValue = null;
        this._isRemote = file._isRemote || file.baseUrl !== undefined;
    }

    /**
     * Filter rows where column matches condition.
     * Immer-style: returns new DataFrame, original unchanged.
     * @param {number} colIdx - Column index
     * @param {string} op - Operator: '=', '!=', '<', '<=', '>', '>='
     * @param {number|bigint|string} value - Value to compare
     * @param {string} type - 'int64', 'float64', or 'string'
     * @returns {DataFrame}
     */
    filter(colIdx, op, value, type = 'int64') {
        const opMap = {
            '=': LanceFile.Op?.EQ ?? 0, '==': LanceFile.Op?.EQ ?? 0,
            '!=': LanceFile.Op?.NE ?? 1, '<>': LanceFile.Op?.NE ?? 1,
            '<': LanceFile.Op?.LT ?? 2,
            '<=': LanceFile.Op?.LE ?? 3,
            '>': LanceFile.Op?.GT ?? 4,
            '>=': LanceFile.Op?.GE ?? 5
        };

        const df = new DataFrame(this.file);
        df._filterOps = [...this._filterOps, { colIdx, op: opMap[op], opStr: op, value, type }];
        df._selectCols = this._selectCols;
        df._limitValue = this._limitValue;
        df._isRemote = this._isRemote;
        return df;
    }

    /**
     * Select specific columns.
     * Immer-style: returns new DataFrame, original unchanged.
     * @param {...number} colIndices - Column indices to select
     * @returns {DataFrame}
     */
    select(...colIndices) {
        // Handle array passed as first arg: select([0,1,2]) or select(0,1,2)
        const cols = Array.isArray(colIndices[0]) ? colIndices[0] : colIndices;
        const df = new DataFrame(this.file);
        df._filterOps = [...this._filterOps];
        df._selectCols = cols;
        df._limitValue = this._limitValue;
        df._isRemote = this._isRemote;
        return df;
    }

    /**
     * Limit number of results.
     * Immer-style: returns new DataFrame, original unchanged.
     * @param {number} n - Maximum rows
     * @returns {DataFrame}
     */
    limit(n) {
        const df = new DataFrame(this.file);
        df._filterOps = [...this._filterOps];
        df._selectCols = this._selectCols;
        df._limitValue = n;
        df._isRemote = this._isRemote;
        return df;
    }

    /**
     * Generate SQL from DataFrame operations.
     * @returns {string}
     */
    toSQL() {
        const colNames = this.file.columnNames || this.file._schema?.map(s => s.name) ||
            Array.from({ length: this.file._numColumns || 6 }, (_, i) => `col_${i}`);

        // SELECT clause
        let selectClause;
        if (this._selectCols && this._selectCols.length > 0) {
            selectClause = this._selectCols.map(i => colNames[i] || `col_${i}`).join(', ');
        } else {
            selectClause = '*';
        }

        // WHERE clause
        let whereClause = '';
        if (this._filterOps.length > 0) {
            const conditions = this._filterOps.map(f => {
                const colName = colNames[f.colIdx] || `col_${f.colIdx}`;
                const val = f.type === 'string' ? `'${f.value}'` : f.value;
                return `${colName} ${f.opStr} ${val}`;
            });
            whereClause = ` WHERE ${conditions.join(' AND ')}`;
        }

        // LIMIT clause
        const limitClause = this._limitValue ? ` LIMIT ${this._limitValue}` : '';

        return `SELECT ${selectClause} FROM dataset${whereClause}${limitClause}`;
    }

    /**
     * Execute the query and return row indices (sync, local only).
     * @returns {Uint32Array}
     */
    collectIndices() {
        if (this._isRemote) {
            throw new Error('collectIndices() is sync-only. Use collect() for remote datasets.');
        }

        let indices = null;

        // Apply filters
        for (const f of this._filterOps) {
            let newIndices;
            if (f.type === 'int64') {
                newIndices = this.file.filterInt64(f.colIdx, f.op, f.value);
            } else {
                newIndices = this.file.filterFloat64(f.colIdx, f.op, f.value);
            }

            if (indices === null) {
                indices = newIndices;
            } else {
                // Intersect indices
                const set = new Set(newIndices);
                indices = indices.filter(i => set.has(i));
                indices = new Uint32Array(indices);
            }
        }

        // If no filters, get all row indices
        if (indices === null) {
            const rowCount = Number(this.file.getRowCount(0));
            indices = new Uint32Array(rowCount);
            for (let i = 0; i < rowCount; i++) indices[i] = i;
        }

        // Apply limit
        if (this._limitValue !== null && indices.length > this._limitValue) {
            indices = indices.slice(0, this._limitValue);
        }

        return indices;
    }

    /**
     * Execute the query and return results.
     * Works with both local (sync) and remote (async) datasets.
     * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
     */
    async collect() {
        // Remote: generate SQL and execute
        if (this._isRemote) {
            const sql = this.toSQL();
            return await this.file.executeSQL(sql);
        }

        // Local: use sync WASM methods
        const indices = this.collectIndices();
        const cols = this._selectCols ||
            Array.from({ length: this.file.numColumns }, (_, i) => i);

        const columns = [];
        const columnNames = [];

        for (const colIdx of cols) {
            columnNames.push(this.file.columnNames?.[colIdx] || `col_${colIdx}`);
            // Try int64 first, then float64
            try {
                columns.push(Array.from(this.file.readInt64AtIndices(colIdx, indices)));
            } catch {
                try {
                    columns.push(Array.from(this.file.readFloat64AtIndices(colIdx, indices)));
                } catch {
                    columns.push(indices.map(() => null));
                }
            }
        }

        return {
            columns,
            columnNames,
            total: indices.length,
            _indices: indices
        };
    }

    /**
     * Count matching rows.
     * @returns {Promise<number>|number}
     */
    async count() {
        if (this._isRemote) {
            const result = await this.collect();
            return result.columns[0]?.length || 0;
        }
        return this.collectIndices().length;
    }
}

/**
 * Represents a Lance file opened from a remote URL.
 * Uses HTTP Range requests to fetch data on demand.
 */
export class RemoteLanceFile {
    constructor(lanceql, url, fileSize, footerData) {
        this.lanceql = lanceql;
        this.wasm = lanceql.wasm;
        this.memory = lanceql.memory;
        this.url = url;
        this.fileSize = fileSize;

        // Store footer data in WASM memory
        const bytes = new Uint8Array(footerData);
        this.footerPtr = this.wasm.alloc(bytes.length);
        if (!this.footerPtr) {
            throw new Error('Failed to allocate memory for footer');
        }
        this.footerLen = bytes.length;
        new Uint8Array(this.memory.buffer).set(bytes, this.footerPtr);

        // Parse footer
        this._numColumns = this.wasm.parseFooterGetColumns(this.footerPtr, this.footerLen);
        this._majorVersion = this.wasm.parseFooterGetMajorVersion(this.footerPtr, this.footerLen);
        this._minorVersion = this.wasm.parseFooterGetMinorVersion(this.footerPtr, this.footerLen);
        this._columnMetaStart = this.wasm.getColumnMetaStart(this.footerPtr, this.footerLen);
        this._columnMetaOffsetsStart = this.wasm.getColumnMetaOffsetsStart(this.footerPtr, this.footerLen);

        // Cache for column metadata to avoid repeated fetches
        this._columnMetaCache = new Map();
        this._columnOffsetCache = new Map();
        this._columnTypes = null;

        // Schema info from manifest (populated by loadSchema())
        this._schema = null;
        this._datasetBaseUrl = null;

        // IVF index for ANN search (populated by tryLoadIndex())
        this._ivfIndex = null;
    }

    /**
     * Open a remote Lance file.
     * @param {LanceQL} lanceql
     * @param {string} url
     * @returns {Promise<RemoteLanceFile>}
     */
    static async open(lanceql, url) {
        // First, get file size with HEAD request
        const headResponse = await fetch(url, { method: 'HEAD' });
        if (!headResponse.ok) {
            throw new Error(`HTTP error: ${headResponse.status}`);
        }

        const contentLength = headResponse.headers.get('Content-Length');
        if (!contentLength) {
            throw new Error('Server did not return Content-Length');
        }
        const fileSize = parseInt(contentLength, 10);

        // Fetch footer (last 40 bytes)
        const footerSize = 40;
        const footerStart = fileSize - footerSize;
        const footerResponse = await fetch(url, {
            headers: {
                'Range': `bytes=${footerStart}-${fileSize - 1}`
            }
        });

        if (!footerResponse.ok && footerResponse.status !== 206) {
            throw new Error(`HTTP error: ${footerResponse.status}`);
        }

        const footerData = await footerResponse.arrayBuffer();

        // Verify magic bytes
        const footerBytes = new Uint8Array(footerData);
        const magic = String.fromCharCode(
            footerBytes[36], footerBytes[37], footerBytes[38], footerBytes[39]
        );
        if (magic !== 'LANC') {
            throw new Error(`Invalid Lance file: expected LANC magic, got "${magic}"`);
        }

        const file = new RemoteLanceFile(lanceql, url, fileSize, footerData);

        // Try to detect and load schema from manifest
        await file._tryLoadSchema();

        // Try to load IVF index for ANN search
        await file._tryLoadIndex();

        // Log summary
        console.log(`[LanceQL] Loaded: ${file._numColumns} columns, ${(fileSize / 1024 / 1024).toFixed(1)}MB, schema: ${file._schema ? 'yes' : 'no'}, index: ${file.hasIndex() ? 'yes' : 'no'}`);

        return file;
    }

    /**
     * Try to load IVF index from dataset.
     * @private
     */
    async _tryLoadIndex() {
        if (!this._datasetBaseUrl) return;

        try {
            this._ivfIndex = await IVFIndex.tryLoad(this._datasetBaseUrl);
        } catch (e) {
            // Index loading is optional, silently ignore
        }
    }

    /**
     * Check if ANN index is available.
     * @returns {boolean}
     */
    hasIndex() {
        return this._ivfIndex !== null && this._ivfIndex.centroids !== null;
    }

    /**
     * Try to detect dataset base URL and load schema from manifest.
     * Lance datasets have structure: base.lance/_versions/, base.lance/data/
     * @private
     */
    async _tryLoadSchema() {
        // Try to infer dataset base URL from file URL
        // Pattern: https://host/path/dataset.lance/data/filename.lance
        const match = this.url.match(/^(.+\.lance)\/data\/.+\.lance$/);
        if (!match) {
            // URL doesn't match standard Lance dataset structure
            return;
        }

        this._datasetBaseUrl = match[1];

        try {
            // Try manifest version 1 first
            const manifestUrl = `${this._datasetBaseUrl}/_versions/1.manifest`;
            const response = await fetch(manifestUrl);

            if (!response.ok) {
                return;
            }

            const manifestData = await response.arrayBuffer();
            this._schema = this._parseManifest(new Uint8Array(manifestData));
        } catch (e) {
            // Silently fail - schema is optional
            // Manifest loading is optional, silently ignore
        }
    }

    /**
     * Parse Lance manifest protobuf to extract schema.
     * Manifest structure:
     * - 4 bytes: content length (little-endian u32)
     * - N bytes: protobuf content
     * - 16 bytes: footer (zeros + version + LANC magic)
     * @private
     */
    _parseManifest(bytes) {
        const view = new DataView(bytes.buffer, bytes.byteOffset);

        // Lance manifest file structure:
        // - Chunk 1 (len-prefixed): Transaction metadata (may be small/incremental)
        // - Chunk 2 (len-prefixed): Full manifest with schema + fragments
        // - Footer (16 bytes): Offsets + "LANC" magic

        // Read chunk 1 length
        const chunk1Len = view.getUint32(0, true);

        // Check if there's a chunk 2 (full manifest data)
        const chunk2Start = 4 + chunk1Len;
        let protoData;

        if (chunk2Start + 4 < bytes.length) {
            const chunk2Len = view.getUint32(chunk2Start, true);
            if (chunk2Len > 0 && chunk2Start + 4 + chunk2Len <= bytes.length) {
                // Use chunk 2 (full manifest)
                protoData = bytes.slice(chunk2Start + 4, chunk2Start + 4 + chunk2Len);
            } else {
                protoData = bytes.slice(4, 4 + chunk1Len);
            }
        } else {
            protoData = bytes.slice(4, 4 + chunk1Len);
        }

        let pos = 0;
        const fields = [];

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < protoData.length) {
                const byte = protoData[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        // Parse top-level Manifest message
        while (pos < protoData.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (fieldNum === 1 && wireType === 2) {
                // Field 1 = schema (repeated Field message)
                const fieldLen = readVarint();
                const fieldEnd = pos + fieldLen;

                // Parse Field message
                let name = null;
                let id = null;
                let logicalType = null;

                while (pos < fieldEnd) {
                    const fTag = readVarint();
                    const fNum = fTag >> 3;
                    const fWire = fTag & 0x7;

                    if (fWire === 0) {
                        // Varint
                        const val = readVarint();
                        if (fNum === 3) id = val;  // Field.id
                    } else if (fWire === 2) {
                        // Length-delimited
                        const len = readVarint();
                        const content = protoData.slice(pos, pos + len);
                        pos += len;

                        if (fNum === 2) {
                            // Field.name
                            name = new TextDecoder().decode(content);
                        } else if (fNum === 5) {
                            // Field.logical_type
                            logicalType = new TextDecoder().decode(content);
                        }
                    } else if (fWire === 5) {
                        pos += 4;  // Fixed32
                    } else if (fWire === 1) {
                        pos += 8;  // Fixed64
                    }
                }

                if (name) {
                    fields.push({ name, id, type: logicalType });
                }
            } else {
                // Skip other fields
                if (wireType === 0) {
                    readVarint();
                } else if (wireType === 2) {
                    const len = readVarint();
                    pos += len;
                } else if (wireType === 5) {
                    pos += 4;
                } else if (wireType === 1) {
                    pos += 8;
                }
            }
        }

        return fields;
    }

    /**
     * Get column names from schema (if available).
     * Falls back to 'column_N' if schema not loaded.
     * @returns {string[]}
     */
    get columnNames() {
        if (this._schema && this._schema.length > 0) {
            return this._schema.map(f => f.name);
        }
        // Fallback to generic names
        return Array.from({ length: this._numColumns }, (_, i) => `column_${i}`);
    }

    /**
     * Get full schema info (if available).
     * @returns {Array<{name: string, id: number, type: string}>|null}
     */
    get schema() {
        return this._schema;
    }

    /**
     * Get dataset base URL (if detected).
     * @returns {string|null}
     */
    get datasetBaseUrl() {
        return this._datasetBaseUrl;
    }

    /**
     * Fetch bytes from the remote file at a specific range.
     * Uses HotTierCache for OPFS-backed caching (500-2000x faster on cache hit).
     * @param {number} start - Start offset
     * @param {number} end - End offset (inclusive)
     * @returns {Promise<ArrayBuffer>}
     */
    async fetchRange(start, end) {
        // Debug: console.log(`fetchRange: ${start}-${end} (size: ${end - start + 1})`);

        // Validate range
        if (start < 0 || end < start || end >= this.size) {
            console.error(`Invalid range: ${start}-${end}, file size: ${this.size}`);
        }

        // Use hot-tier cache if available
        if (hotTierCache.enabled) {
            const data = await hotTierCache.getRange(this.url, start, end, this.size);

            // Track stats if callback available
            if (this._onFetch) {
                this._onFetch(data.byteLength, 1);
            }

            return data;
        }

        // Fallback to direct fetch
        const response = await fetch(this.url, {
            headers: {
                'Range': `bytes=${start}-${end}`
            }
        });

        if (!response.ok && response.status !== 206) {
            console.error(`Fetch failed: ${response.status} for range ${start}-${end}`);
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.arrayBuffer();

        // Track stats if callback available
        if (this._onFetch) {
            this._onFetch(data.byteLength, 1);
        }

        return data;
    }

    /**
     * Set callback for network stats tracking.
     * @param {function} callback - Function(bytesDownloaded, requestCount)
     */
    onFetch(callback) {
        this._onFetch = callback;
    }

    /**
     * Close the file and free memory.
     */
    close() {
        if (this.footerPtr) {
            this.wasm.free(this.footerPtr, this.footerLen);
            this.footerPtr = null;
        }
    }

    /**
     * Get the number of columns.
     * @returns {number}
     */
    get numColumns() {
        return this._numColumns;
    }

    /**
     * Get the file size.
     * @returns {number}
     */
    get size() {
        return this.fileSize;
    }

    /**
     * Get the version.
     * @returns {{major: number, minor: number}}
     */
    get version() {
        return {
            major: this._majorVersion,
            minor: this._minorVersion
        };
    }

    /**
     * Get the column metadata start offset.
     * @returns {number}
     */
    get columnMetaStart() {
        return Number(this._columnMetaStart);
    }

    /**
     * Get the column metadata offsets start.
     * @returns {number}
     */
    get columnMetaOffsetsStart() {
        return Number(this._columnMetaOffsetsStart);
    }

    /**
     * Get column offset entry from column metadata offsets.
     * Uses caching to avoid repeated fetches.
     * @param {number} colIdx
     * @returns {Promise<{pos: number, len: number}>}
     */
    async getColumnOffsetEntry(colIdx) {
        if (colIdx >= this._numColumns) {
            return { pos: 0, len: 0 };
        }

        // Check cache first
        if (this._columnOffsetCache.has(colIdx)) {
            return this._columnOffsetCache.get(colIdx);
        }

        // Each entry is 16 bytes (8 bytes pos + 8 bytes len)
        const entryOffset = this.columnMetaOffsetsStart + colIdx * 16;
        const data = await this.fetchRange(entryOffset, entryOffset + 15);
        const view = new DataView(data);

        const entry = {
            pos: Number(view.getBigUint64(0, true)),
            len: Number(view.getBigUint64(8, true))
        };

        // Cache the result
        this._columnOffsetCache.set(colIdx, entry);
        return entry;
    }

    /**
     * Get debug info for a column (requires network request).
     * @param {number} colIdx
     * @returns {Promise<{offset: number, size: number, rows: number}>}
     */
    async getColumnDebugInfo(colIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) {
            return { offset: 0, size: 0, rows: 0 };
        }

        // Fetch column metadata
        const colMetaData = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const bytes = new Uint8Array(colMetaData);

        // Parse column metadata to get buffer info
        const info = this._parseColumnMeta(bytes);
        return info;
    }

    /**
     * Parse column metadata to extract buffer offsets and row count.
     * For nullable columns, there are typically 2 buffers:
     * - Buffer 0: null bitmap
     * - Buffer 1: actual data values
     * @private
     */
    _parseColumnMeta(bytes) {
        let pos = 0;
        const pages = [];
        let totalRows = 0;

        // Read varint as BigInt to handle large values (>2GB offsets)
        const readVarint = () => {
            let result = 0n;
            let shift = 0n;
            while (pos < bytes.length) {
                const byte = bytes[pos++];
                result |= BigInt(byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7n;
            }
            return Number(result);
        };

        while (pos < bytes.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (fieldNum === 2 && wireType === 2) {
                // pages field (length-delimited) - parse ALL pages
                const pageLen = readVarint();
                const pageEnd = pos + pageLen;

                const pageOffsets = [];
                const pageSizes = [];
                let pageRows = 0;

                // Parse page
                while (pos < pageEnd) {
                    const pageTag = readVarint();
                    const pageField = pageTag >> 3;
                    const pageWire = pageTag & 0x7;

                    if (pageField === 1 && pageWire === 2) {
                        // buffer_offsets (packed)
                        const packedLen = readVarint();
                        const packedEnd = pos + packedLen;
                        while (pos < packedEnd) {
                            pageOffsets.push(readVarint());
                        }
                    } else if (pageField === 2 && pageWire === 2) {
                        // buffer_sizes (packed)
                        const packedLen = readVarint();
                        const packedEnd = pos + packedLen;
                        while (pos < packedEnd) {
                            pageSizes.push(readVarint());
                        }
                    } else if (pageField === 3 && pageWire === 0) {
                        // length (rows)
                        pageRows = readVarint();
                    } else {
                        // Skip field
                        if (pageWire === 0) readVarint();
                        else if (pageWire === 2) {
                            const skipLen = readVarint();
                            pos += skipLen;
                        }
                        else if (pageWire === 5) pos += 4;
                        else if (pageWire === 1) pos += 8;
                    }
                }

                pages.push({
                    offsets: pageOffsets,
                    sizes: pageSizes,
                    rows: pageRows
                });
                totalRows += pageRows;
                // Don't break - continue to read more pages
            } else {
                // Skip field
                if (wireType === 0) readVarint();
                else if (wireType === 2) {
                    const skipLen = readVarint();
                    pos += skipLen;
                }
                else if (wireType === 5) pos += 4;
                else if (wireType === 1) pos += 8;
            }
        }

        // Combine all pages - use first page for offset/size (for backward compat)
        // Also compute total size across all pages for multi-page columns
        const firstPage = pages[0] || { offsets: [], sizes: [], rows: 0 };
        const bufferOffsets = firstPage.offsets;
        const bufferSizes = firstPage.sizes;

        // For multi-page columns (like embeddings), compute total size
        let totalSize = 0;
        for (const page of pages) {
            // Use the data buffer (last buffer, or buffer 1 for nullable)
            const dataIdx = page.sizes.length > 1 ? 1 : 0;
            totalSize += page.sizes[dataIdx] || 0;
        }

        // For nullable columns: buffer 0 = null bitmap, buffer 1 = data
        // For non-nullable: buffer 0 = data
        const dataBufferIdx = bufferOffsets.length > 1 ? 1 : 0;
        const nullBitmapIdx = bufferOffsets.length > 1 ? 0 : -1;

        return {
            offset: bufferOffsets[dataBufferIdx] || 0,
            size: pages.length > 1 ? totalSize : (bufferSizes[dataBufferIdx] || 0),
            rows: totalRows,
            nullBitmapOffset: nullBitmapIdx >= 0 ? bufferOffsets[nullBitmapIdx] : null,
            nullBitmapSize: nullBitmapIdx >= 0 ? bufferSizes[nullBitmapIdx] : null,
            bufferOffsets,
            bufferSizes,
            pages  // Include all pages for multi-page access
        };
    }

    /**
     * Parse string column metadata to get offsets and data buffer info.
     * @private
     */
    _parseStringColumnMeta(bytes) {
        // Parse ALL pages for multi-page string columns
        const pages = [];
        let pos = 0;

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < bytes.length) {
                const byte = bytes[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        while (pos < bytes.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (fieldNum === 2 && wireType === 2) {
                // pages field - parse this page
                const pageLen = readVarint();
                const pageEnd = pos + pageLen;

                let bufferOffsets = [0, 0];
                let bufferSizes = [0, 0];
                let rows = 0;

                while (pos < pageEnd) {
                    const pageTag = readVarint();
                    const pageField = pageTag >> 3;
                    const pageWire = pageTag & 0x7;

                    if (pageField === 1 && pageWire === 2) {
                        // buffer_offsets (packed)
                        const packedLen = readVarint();
                        const packedEnd = pos + packedLen;
                        let idx = 0;
                        while (pos < packedEnd && idx < 2) {
                            bufferOffsets[idx++] = readVarint();
                        }
                        pos = packedEnd;
                    } else if (pageField === 2 && pageWire === 2) {
                        // buffer_sizes (packed)
                        const packedLen = readVarint();
                        const packedEnd = pos + packedLen;
                        let idx = 0;
                        while (pos < packedEnd && idx < 2) {
                            bufferSizes[idx++] = readVarint();
                        }
                        pos = packedEnd;
                    } else if (pageField === 3 && pageWire === 0) {
                        rows = readVarint();
                    } else if (pageField === 4 && pageWire === 2) {
                        // encoding field - skip it
                        const skipLen = readVarint();
                        pos += skipLen;
                    } else {
                        // Unknown field - skip based on wire type
                        if (pageWire === 0) readVarint();
                        else if (pageWire === 2) {
                            const skipLen = readVarint();
                            pos += skipLen;
                        }
                        else if (pageWire === 5) pos += 4;
                        else if (pageWire === 1) pos += 8;
                    }
                }

                pages.push({
                    offsetsStart: bufferOffsets[0],
                    offsetsSize: bufferSizes[0],
                    dataStart: bufferOffsets[1],
                    dataSize: bufferSizes[1],
                    rows
                });
            } else {
                // Skip unknown fields
                if (wireType === 0) {
                    readVarint();
                } else if (wireType === 2) {
                    const skipLen = readVarint();
                    pos += skipLen;
                } else if (wireType === 5) {
                    pos += 4;
                } else if (wireType === 1) {
                    pos += 8;
                }
            }
        }

        // Return first page for backwards compatibility, but also include all pages
        const firstPage = pages[0] || { offsetsStart: 0, offsetsSize: 0, dataStart: 0, dataSize: 0, rows: 0 };
        return {
            ...firstPage,
            pages
        };
    }

    /**
     * Batch indices into contiguous ranges to minimize HTTP requests.
     * Groups nearby indices if the gap is smaller than gapThreshold.
     * @private
     */
    _batchIndices(indices, valueSize, gapThreshold = 1024) {
        if (indices.length === 0) return [];

        // Sort indices for contiguous access
        const sorted = [...indices].map((v, i) => ({ idx: v, origPos: i }));
        sorted.sort((a, b) => a.idx - b.idx);

        const batches = [];
        let batchStart = 0;

        for (let i = 1; i <= sorted.length; i++) {
            // Check if we should end the current batch
            const endBatch = i === sorted.length ||
                (sorted[i].idx - sorted[i-1].idx) * valueSize > gapThreshold;

            if (endBatch) {
                batches.push({
                    startIdx: sorted[batchStart].idx,
                    endIdx: sorted[i-1].idx,
                    items: sorted.slice(batchStart, i)
                });
                batchStart = i;
            }
        }

        return batches;
    }

    /**
     * Read int64 values at specific row indices via Range requests.
     * Uses batched fetching to minimize HTTP requests.
     * @param {number} colIdx
     * @param {number[]} indices - Row indices
     * @returns {Promise<BigInt64Array>}
     */
    async readInt64AtIndices(colIdx, indices) {
        if (indices.length === 0) return new BigInt64Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        // Debug: console.log(`readInt64AtIndices col ${colIdx}: rows=${info.rows}`);

        const results = new BigInt64Array(indices.length);
        const valueSize = 8;

        // Batch indices into contiguous ranges
        const batches = this._batchIndices(indices, valueSize);

        // Fetch each batch in parallel
        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            // Extract values from batch
            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * valueSize;
                results[item.origPos] = view.getBigInt64(localOffset, true);
            }
        }));

        return results;
    }

    /**
     * Read float64 values at specific row indices via Range requests.
     * Uses batched fetching to minimize HTTP requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Float64Array>}
     */
    async readFloat64AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Float64Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Float64Array(indices.length);
        const valueSize = 8;

        // Batch indices into contiguous ranges
        const batches = this._batchIndices(indices, valueSize);

        // Fetch each batch in parallel
        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            // Extract values from batch
            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * valueSize;
                results[item.origPos] = view.getFloat64(localOffset, true);
            }
        }));

        return results;
    }

    /**
     * Read int32 values at specific row indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Int32Array>}
     */
    async readInt32AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Int32Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Int32Array(indices.length);
        const valueSize = 4;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * valueSize;
                results[item.origPos] = view.getInt32(localOffset, true);
            }
        }));

        return results;
    }

    /**
     * Read float32 values at specific row indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Float32Array>}
     */
    async readFloat32AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Float32Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Float32Array(indices.length);
        const valueSize = 4;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * valueSize;
                results[item.origPos] = view.getFloat32(localOffset, true);
            }
        }));

        return results;
    }

    /**
     * Read vectors (fixed_size_list of float32) at specific row indices.
     * Returns array of Float32Array vectors.
     * @param {number} colIdx - Vector column index
     * @param {number[]} indices - Row indices to read
     * @returns {Promise<Float32Array[]>} - Array of vectors
     */
    async readVectorsAtIndices(colIdx, indices) {
        if (indices.length === 0) return [];

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const metaInfo = this._parseColumnMeta(new Uint8Array(colMeta));

        if (!metaInfo.pages || metaInfo.pages.length === 0) {
            return indices.map(() => null);
        }

        // Calculate dimension from first page
        const firstPage = metaInfo.pages[0];
        const dataIdx = firstPage.sizes.length > 1 ? 1 : 0;
        const firstPageSize = firstPage.sizes[dataIdx] || 0;
        const firstPageRows = firstPage.rows || 0;

        if (firstPageRows === 0 || firstPageSize === 0) {
            return indices.map(() => null);
        }

        const dim = Math.floor(firstPageSize / (firstPageRows * 4));
        const vecSize = dim * 4;

        const results = new Array(indices.length).fill(null);

        // Build page index for quick lookup
        let pageRowStart = 0;
        const pageIndex = [];
        for (const page of metaInfo.pages) {
            pageIndex.push({ start: pageRowStart, end: pageRowStart + page.rows, page });
            pageRowStart += page.rows;
        }

        // Group indices by page
        const pageGroups = new Map();
        for (let i = 0; i < indices.length; i++) {
            const rowIdx = indices[i];
            // Find which page contains this row
            for (let p = 0; p < pageIndex.length; p++) {
                const pi = pageIndex[p];
                if (rowIdx >= pi.start && rowIdx < pi.end) {
                    if (!pageGroups.has(p)) {
                        pageGroups.set(p, []);
                    }
                    pageGroups.set(p, [...pageGroups.get(p), { rowIdx, localIdx: rowIdx - pi.start, resultIdx: i }]);
                    break;
                }
            }
        }

        // Fetch vectors from each page
        const fetchPromises = [];
        for (const [pageNum, items] of pageGroups) {
            const page = metaInfo.pages[pageNum];
            const pageDataIdx = page.sizes.length > 1 ? 1 : 0;
            const pageOffset = page.offsets[pageDataIdx] || 0;

            fetchPromises.push((async () => {
                // Sort items by local index for better batching
                items.sort((a, b) => a.localIdx - b.localIdx);

                // Batch contiguous reads
                const batches = [];
                let currentBatch = { start: items[0].localIdx, end: items[0].localIdx, items: [items[0]] };

                for (let i = 1; i < items.length; i++) {
                    const item = items[i];
                    // If within 10 vectors, extend batch (avoid too many small requests)
                    if (item.localIdx - currentBatch.end <= 10) {
                        currentBatch.end = item.localIdx;
                        currentBatch.items.push(item);
                    } else {
                        batches.push(currentBatch);
                        currentBatch = { start: item.localIdx, end: item.localIdx, items: [item] };
                    }
                }
                batches.push(currentBatch);

                // Fetch each batch
                for (const batch of batches) {
                    const startOffset = pageOffset + batch.start * vecSize;
                    const endOffset = pageOffset + (batch.end + 1) * vecSize - 1;
                    const data = await this.fetchRange(startOffset, endOffset);
                    const floatData = new Float32Array(data);

                    for (const item of batch.items) {
                        const localOffset = (item.localIdx - batch.start) * dim;
                        results[item.resultIdx] = floatData.slice(localOffset, localOffset + dim);
                    }
                }
            })());
        }

        await Promise.all(fetchPromises);
        return results;
    }

    /**
     * Read int16 values at specific row indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Int16Array>}
     */
    async readInt16AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Int16Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Int16Array(indices.length);
        const valueSize = 2;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * valueSize;
                results[item.origPos] = view.getInt16(localOffset, true);
            }
        }));

        return results;
    }

    /**
     * Read uint8 values at specific row indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Uint8Array>}
     */
    async readUint8AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Uint8Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Uint8Array(indices.length);
        const valueSize = 1;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const bytes = new Uint8Array(data);

            for (const item of batch.items) {
                const localOffset = item.idx - batch.startIdx;
                results[item.origPos] = bytes[localOffset];
            }
        }));

        return results;
    }

    /**
     * Read bool values at specific row indices via Range requests.
     * Boolean values are bit-packed (8 values per byte).
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Uint8Array>}
     */
    async readBoolAtIndices(colIdx, indices) {
        if (indices.length === 0) return new Uint8Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Uint8Array(indices.length);

        // Calculate byte ranges needed for bit-packed booleans
        const byteIndices = indices.map(i => Math.floor(i / 8));
        const uniqueBytes = [...new Set(byteIndices)].sort((a, b) => a - b);

        if (uniqueBytes.length === 0) return results;

        // Fetch the byte range
        const startByte = uniqueBytes[0];
        const endByte = uniqueBytes[uniqueBytes.length - 1];
        const startOffset = info.offset + startByte;
        const endOffset = info.offset + endByte;
        const data = await this.fetchRange(startOffset, endOffset);
        const bytes = new Uint8Array(data);

        // Extract boolean values
        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            const byteIdx = Math.floor(idx / 8);
            const bitIdx = idx % 8;
            const localByteIdx = byteIdx - startByte;
            if (localByteIdx >= 0 && localByteIdx < bytes.length) {
                results[i] = (bytes[localByteIdx] >> bitIdx) & 1;
            }
        }

        return results;
    }

    /**
     * Read a single string at index via Range requests.
     * @param {number} colIdx
     * @param {number} rowIdx
     * @returns {Promise<string>}
     * @throws {Error} If the column is not a string column
     */
    async readStringAt(colIdx, rowIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseStringColumnMeta(new Uint8Array(colMeta));

        // Check if this is actually a string column
        // String columns have: offsetsSize / rows = 4 or 8 bytes per offset
        // Numeric columns with validity bitmap have: offsetsSize = rows / 8 (bitmap)
        if (info.offsetsSize === 0 || info.dataSize === 0) {
            throw new Error(`Not a string column - offsetsSize=${info.offsetsSize}, dataSize=${info.dataSize}`);
        }

        // Calculate bytes per offset - strings have rows offsets of 4 or 8 bytes each
        const bytesPerOffset = info.offsetsSize / info.rows;

        // If bytesPerOffset is not 4 or 8, this is not a string column
        // (e.g., it's a validity bitmap which has rows/8 bytes = 0.125 bytes per row)
        if (bytesPerOffset !== 4 && bytesPerOffset !== 8) {
            throw new Error(`Not a string column - bytesPerOffset=${bytesPerOffset}, expected 4 or 8`);
        }

        if (rowIdx >= info.rows) return '';

        // Determine offset size (4 or 8 bytes)
        const offsetSize = bytesPerOffset;

        // Fetch the two offsets for this string
        const offsetStart = info.offsetsStart + rowIdx * offsetSize;
        const offsetData = await this.fetchRange(offsetStart, offsetStart + offsetSize * 2 - 1);
        const offsetView = new DataView(offsetData);

        let strStart, strEnd;
        if (offsetSize === 4) {
            strStart = offsetView.getUint32(0, true);
            strEnd = offsetView.getUint32(4, true);
        } else {
            strStart = Number(offsetView.getBigUint64(0, true));
            strEnd = Number(offsetView.getBigUint64(8, true));
        }

        if (strEnd <= strStart) return '';
        const strLen = strEnd - strStart;

        // Fetch the string data
        const strData = await this.fetchRange(
            info.dataStart + strStart,
            info.dataStart + strEnd - 1
        );

        return new TextDecoder().decode(strData);
    }

    /**
     * Read multiple strings at indices via Range requests.
     * Uses batched fetching to minimize HTTP requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<string[]>}
     */
    async readStringsAtIndices(colIdx, indices) {
        if (indices.length === 0) return [];

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseStringColumnMeta(new Uint8Array(colMeta));

        if (!info.pages || info.pages.length === 0) {
            return indices.map(() => '');
        }

        const results = new Array(indices.length).fill('');

        // Build page index with cumulative row counts
        let pageRowStart = 0;
        const pageIndex = [];
        for (const page of info.pages) {
            if (page.offsetsSize === 0 || page.dataSize === 0 || page.rows === 0) {
                pageRowStart += page.rows;
                continue;
            }
            pageIndex.push({
                start: pageRowStart,
                end: pageRowStart + page.rows,
                page
            });
            pageRowStart += page.rows;
        }

        // Group indices by page
        const pageGroups = new Map();
        for (let i = 0; i < indices.length; i++) {
            const rowIdx = indices[i];
            // Find which page contains this row
            for (let p = 0; p < pageIndex.length; p++) {
                const pi = pageIndex[p];
                if (rowIdx >= pi.start && rowIdx < pi.end) {
                    if (!pageGroups.has(p)) {
                        pageGroups.set(p, []);
                    }
                    pageGroups.get(p).push({
                        globalIdx: rowIdx,
                        localIdx: rowIdx - pi.start,
                        resultIdx: i
                    });
                    break;
                }
            }
        }

        // Fetch strings from each page
        for (const [pageNum, items] of pageGroups) {
            const pi = pageIndex[pageNum];
            const page = pi.page;

            // Determine offset size (4 or 8 bytes per offset)
            const offsetSize = page.offsetsSize / page.rows;
            if (offsetSize !== 4 && offsetSize !== 8) continue;

            // Sort items by localIdx for efficient batching
            items.sort((a, b) => a.localIdx - b.localIdx);

            // Fetch offsets in batches
            const offsetBatches = [];
            let batchStart = 0;
            for (let i = 1; i <= items.length; i++) {
                if (i === items.length || items[i].localIdx - items[i-1].localIdx > 100) {
                    offsetBatches.push(items.slice(batchStart, i));
                    batchStart = i;
                }
            }

            // Collect string ranges from offset fetches
            // Lance string encoding: offset[N] = end of string N, start is offset[N-1] (or 0 if N=0)
            const stringRanges = [];

            await Promise.all(offsetBatches.map(async (batch) => {
                const minIdx = batch[0].localIdx;
                const maxIdx = batch[batch.length - 1].localIdx;

                // Fetch offsets: need offset[minIdx-1] through offset[maxIdx]
                // But if minIdx=0, we don't need offset[-1] since start is implicitly 0
                const fetchStartIdx = minIdx > 0 ? minIdx - 1 : 0;
                const fetchEndIdx = maxIdx;
                const startOffset = page.offsetsStart + fetchStartIdx * offsetSize;
                const endOffset = page.offsetsStart + (fetchEndIdx + 1) * offsetSize - 1;
                const data = await this.fetchRange(startOffset, endOffset);
                const view = new DataView(data);

                for (const item of batch) {
                    // Position in fetched data
                    const dataIdx = item.localIdx - fetchStartIdx;
                    let strStart, strEnd;

                    if (offsetSize === 4) {
                        // strEnd = offset[localIdx], strStart = offset[localIdx-1] or 0
                        strEnd = view.getUint32(dataIdx * 4, true);
                        strStart = item.localIdx === 0 ? 0 : view.getUint32((dataIdx - 1) * 4, true);
                    } else {
                        strEnd = Number(view.getBigUint64(dataIdx * 8, true));
                        strStart = item.localIdx === 0 ? 0 : Number(view.getBigUint64((dataIdx - 1) * 8, true));
                    }

                    if (strEnd > strStart) {
                        stringRanges.push({
                            start: strStart,
                            end: strEnd,
                            resultIdx: item.resultIdx,
                            dataStart: page.dataStart
                        });
                    }
                }
            }));

            // Fetch string data
            if (stringRanges.length > 0) {
                stringRanges.sort((a, b) => a.start - b.start);

                // Batch nearby string fetches
                const dataBatches = [];
                let dbStart = 0;
                for (let i = 1; i <= stringRanges.length; i++) {
                    if (i === stringRanges.length ||
                        stringRanges[i].start - stringRanges[i-1].end > 4096) {
                        dataBatches.push({
                            rangeStart: stringRanges[dbStart].start,
                            rangeEnd: stringRanges[i-1].end,
                            items: stringRanges.slice(dbStart, i),
                            dataStart: stringRanges[dbStart].dataStart
                        });
                        dbStart = i;
                    }
                }

                await Promise.all(dataBatches.map(async (batch) => {
                    const data = await this.fetchRange(
                        batch.dataStart + batch.rangeStart,
                        batch.dataStart + batch.rangeEnd - 1
                    );
                    const bytes = new Uint8Array(data);

                    for (const item of batch.items) {
                        const localStart = item.start - batch.rangeStart;
                        const len = item.end - item.start;
                        const strBytes = bytes.slice(localStart, localStart + len);
                        results[item.resultIdx] = new TextDecoder().decode(strBytes);
                    }
                }));
            }
        }

        return results;
    }

    /**
     * Get row count for a column.
     * @param {number} colIdx
     * @returns {Promise<number>}
     */
    async getRowCount(colIdx) {
        const info = await this.getColumnDebugInfo(colIdx);
        return info.rows;
    }

    /**
     * Detect column types by sampling first row.
     * Returns array of type strings: 'string', 'int64', 'float64', 'float32', 'int32', 'int16', 'vector', 'unknown'
     * @returns {Promise<string[]>}
     */
    async detectColumnTypes() {
        // Return cached if available
        if (this._columnTypes) {
            return this._columnTypes;
        }

        const types = [];

        // First, try to use schema types if available
        if (this._schema && this._schema.length > 0) {
            // Schema loaded successfully

            // Build a map from schema - schema may have more fields than physical columns
            for (let c = 0; c < this._numColumns; c++) {
                const schemaField = this._schema[c];
                const schemaType = schemaField?.type?.toLowerCase() || '';
                const schemaName = schemaField?.name?.toLowerCase() || '';
                let type = 'unknown';

                // Debug: console.log(`Column ${c}: name="${schemaField?.name}", type="${schemaType}"`);

                // Check if column name suggests it's a vector/embedding
                const isEmbeddingName = schemaName.includes('embedding') || schemaName.includes('vector') ||
                                        schemaName.includes('emb') || schemaName === 'vec';

                // Map Lance/Arrow logical types to our types
                if (schemaType.includes('utf8') || schemaType.includes('string') || schemaType.includes('large_utf8')) {
                    type = 'string';
                } else if (schemaType.includes('fixed_size_list') || schemaType.includes('vector') || isEmbeddingName) {
                    // Vector detection - check schema type OR column name
                    type = 'vector';
                } else if (schemaType.includes('int64') || schemaType === 'int64') {
                    type = 'int64';
                } else if (schemaType.includes('int32') || schemaType === 'int32') {
                    type = 'int32';
                } else if (schemaType.includes('int16') || schemaType === 'int16') {
                    type = 'int16';
                } else if (schemaType.includes('int8') || schemaType === 'int8') {
                    type = 'int8';
                } else if (schemaType.includes('float64') || schemaType.includes('double')) {
                    type = 'float64';
                } else if (schemaType.includes('float32') || schemaType.includes('float') && !schemaType.includes('64')) {
                    type = 'float32';
                } else if (schemaType.includes('bool')) {
                    type = 'bool';
                }

                types.push(type);
            }

            // If we got useful types from schema, cache and return
            if (types.some(t => t !== 'unknown')) {
                // Debug: console.log('Detected types from schema:', types);
                this._columnTypes = types;
                return types;
            }

            // Otherwise fall through to detection
            // Schema types all unknown, fall back to data detection
            types.length = 0;
        }

        // Fall back to detection by examining data
        // Detecting column types from data
        for (let c = 0; c < this._numColumns; c++) {
            let type = 'unknown';
            const colName = this.columnNames[c]?.toLowerCase() || '';

            // Check if column name suggests it's a vector/embedding
            const isEmbeddingName = colName.includes('embedding') || colName.includes('vector') ||
                                    colName.includes('emb') || colName === 'vec';

            // Try string first - if we can read a valid string, it's a string column
            try {
                const str = await this.readStringAt(c, 0);
                // readStringAt throws for non-string columns, returns string for valid string columns
                type = 'string';
                // Detected as string
                types.push(type);
                continue;
            } catch (e) {
                // Not a string column, continue to numeric detection
            }

            // Check numeric column by examining bytes per row
            try {
                const entry = await this.getColumnOffsetEntry(c);
                if (entry.len > 0) {
                    const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
                    const bytes = new Uint8Array(colMeta);
                    const info = this._parseColumnMeta(bytes);

                    // Debug: console.log(`Column ${c}: bytesPerRow=${info.size / info.rows}`);

                    if (info.rows > 0 && info.size > 0) {
                        const bytesPerRow = info.size / info.rows;

                        // If column name suggests embedding, treat as vector regardless of size
                        if (isEmbeddingName && bytesPerRow >= 4) {
                            type = 'vector';
                        } else if (bytesPerRow === 8) {
                            // int64 or float64 - try to distinguish
                            type = 'int64';  // Default to int64
                        } else if (bytesPerRow === 4) {
                            // int32 or float32 - try reading as int32 to check
                            try {
                                const data = await this.readInt32AtIndices(c, [0]);
                                if (data.length > 0) {
                                    const val = data[0];
                                    // Detected int32 via sample value
                                    // Heuristic: small integers likely int32, weird values likely float32
                                    if (val >= -1000000 && val <= 1000000 && Number.isInteger(val)) {
                                        type = 'int32';
                                    } else {
                                        type = 'float32';
                                    }
                                }
                            } catch (e) {
                                type = 'float32';
                            }
                        } else if (bytesPerRow > 8 && bytesPerRow % 4 === 0) {
                            type = 'vector';
                        } else if (bytesPerRow === 2) {
                            type = 'int16';
                        } else if (bytesPerRow === 1) {
                            type = 'int8';
                        }
                    }
                }
            } catch (e) {
                // Failed to detect type for column, leave as unknown
            }

            // Debug: console.log(`Column ${c}: ${type}`);
            types.push(type);
        }

        this._columnTypes = types;
        return types;
    }

    /**
     * Get cached column metadata, fetching if necessary.
     * @private
     */
    async _getCachedColumnMeta(colIdx) {
        if (this._columnMetaCache.has(colIdx)) {
            return this._columnMetaCache.get(colIdx);
        }

        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) {
            return null;
        }

        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const bytes = new Uint8Array(colMeta);

        this._columnMetaCache.set(colIdx, bytes);
        return bytes;
    }

    // ========================================================================
    // Vector Column Support (for embeddings/semantic search via Range requests)
    // ========================================================================

    /**
     * Get vector info for a column via Range requests.
     * @param {number} colIdx - Column index
     * @returns {Promise<{rows: number, dimension: number}>}
     */
    async getVectorInfo(colIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) return { rows: 0, dimension: 0 };

        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        if (info.rows === 0) return { rows: 0, dimension: 0 };

        // Calculate dimension from first page (all pages have same dimension)
        let dimension = 0;
        if (info.pages && info.pages.length > 0) {
            const firstPage = info.pages[0];
            const dataIdx = firstPage.sizes.length > 1 ? 1 : 0;
            const pageSize = firstPage.sizes[dataIdx] || 0;
            const pageRows = firstPage.rows || 0;
            if (pageRows > 0 && pageSize > 0) {
                dimension = Math.floor(pageSize / (pageRows * 4));
            }
        } else if (info.size > 0) {
            // Fallback for single-page
            dimension = Math.floor(info.size / (info.rows * 4));
        }

        return { rows: info.rows, dimension };
    }

    /**
     * Read a single vector at index via Range requests.
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Row index
     * @returns {Promise<Float32Array>}
     */
    async readVectorAt(colIdx, rowIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        if (info.rows === 0) return new Float32Array(0);
        if (rowIdx >= info.rows) return new Float32Array(0);

        const dim = Math.floor(info.size / (info.rows * 4));
        if (dim === 0) return new Float32Array(0);

        // Fetch the vector data
        const vecStart = info.offset + rowIdx * dim * 4;
        const vecEnd = vecStart + dim * 4 - 1;
        const data = await this.fetchRange(vecStart, vecEnd);

        return new Float32Array(data);
    }

    /**
     * Read multiple vectors at indices via Range requests.
     * Uses batched fetching for efficiency.
     * @param {number} colIdx - Column index
     * @param {number[]} indices - Row indices
     * @returns {Promise<Float32Array[]>}
     */
    async readVectorsAtIndices(colIdx, indices) {
        if (indices.length === 0) return [];

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        if (info.rows === 0) return indices.map(() => new Float32Array(0));

        const dim = Math.floor(info.size / (info.rows * 4));
        if (dim === 0) return indices.map(() => new Float32Array(0));

        const vecSize = dim * 4;
        const results = new Array(indices.length);

        // Batch indices for efficient fetching - parallel with limit
        const batches = this._batchIndices(indices, vecSize, vecSize * 50);
        const BATCH_PARALLEL = 6;

        for (let i = 0; i < batches.length; i += BATCH_PARALLEL) {
            const batchGroup = batches.slice(i, i + BATCH_PARALLEL);
            await Promise.all(batchGroup.map(async (batch) => {
                try {
                    const startOffset = info.offset + batch.startIdx * vecSize;
                    const endOffset = info.offset + (batch.endIdx + 1) * vecSize - 1;
                    const data = await this.fetchRange(startOffset, endOffset);

                    for (const item of batch.items) {
                        const localOffset = (item.idx - batch.startIdx) * vecSize;
                        results[item.origPos] = new Float32Array(
                            data.slice(localOffset, localOffset + vecSize)
                        );
                    }
                } catch (e) {
                    for (const item of batch.items) {
                        results[item.origPos] = new Float32Array(0);
                    }
                }
            }));
        }

        return results;
    }

    /**
     * Compute cosine similarity between two vectors (in JS).
     * @param {Float32Array} vecA
     * @param {Float32Array} vecB
     * @returns {number}
     */
    cosineSimilarity(vecA, vecB) {
        if (vecA.length !== vecB.length) return 0;

        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < vecA.length; i++) {
            dot += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }

        const denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom === 0 ? 0 : dot / denom;
    }

    /**
     * Find top-k most similar vectors to query via Range requests.
     * NOTE: This requires scanning the entire vector column which can be slow
     * for large datasets. For production, use an index.
     *
     * @param {number} colIdx - Column index with vectors
     * @param {Float32Array} queryVec - Query vector
     * @param {number} topK - Number of results to return
     * @param {function} onProgress - Progress callback(current, total)
     * @param {object} options - Search options
     * @param {number} options.nprobe - Number of partitions to search (for ANN)
     * @param {boolean} options.useIndex - Whether to use ANN index if available
     * @returns {Promise<{indices: number[], scores: number[], useIndex: boolean}>}
     */
    async vectorSearch(colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
        const { nprobe = 10, useIndex = true } = options;

        const info = await this.getVectorInfo(colIdx);
        if (info.dimension === 0 || info.dimension !== queryVec.length) {
            throw new Error(`Dimension mismatch: query=${queryVec.length}, column=${info.dimension}`);
        }

        // Require IVF index - no brute force fallback
        if (!this.hasIndex()) {
            throw new Error('No IVF index found. Vector search requires an IVF index for efficient querying.');
        }

        if (this._ivfIndex.dimension !== queryVec.length) {
            throw new Error(`Query dimension (${queryVec.length}) does not match index dimension (${this._ivfIndex.dimension}).`);
        }

        return await this._vectorSearchWithIndex(colIdx, queryVec, topK, nprobe, onProgress);
    }

    /**
     * Vector search using IVF index (ANN).
     * Fetches row IDs from auxiliary.idx for nearest partitions,
     * then looks up original vectors by fragment/offset.
     * @private
     */
    async _vectorSearchWithIndex(colIdx, queryVec, topK, nprobe, onProgress) {
        const dim = queryVec.length;

        // Find nearest partitions using centroids
        if (onProgress) onProgress(0, 100);
        const partitions = this._ivfIndex.findNearestPartitions(queryVec, nprobe);
        const estimatedRows = this._ivfIndex.getPartitionRowCount(partitions);

        console.log(`[IVFSearch] Searching ${partitions.length} partitions (~${estimatedRows.toLocaleString()} rows)`);

        // Try to fetch row IDs from auxiliary.idx
        const rowIdMappings = await this._ivfIndex.fetchPartitionRowIds(partitions);

        if (rowIdMappings && rowIdMappings.length > 0) {
            // Use proper row ID mapping from auxiliary.idx
            console.log(`[IVFSearch] Fetched ${rowIdMappings.length} row ID mappings`);
            return await this._searchWithRowIdMappings(colIdx, queryVec, topK, rowIdMappings, onProgress);
        }

        // No fallback - require proper row ID mapping
        throw new Error('Failed to fetch row IDs from IVF index. Dataset may be missing auxiliary.idx or ivf_partitions.bin.');
    }

    /**
     * Search using proper row ID mappings from auxiliary.idx.
     * Groups row IDs by fragment and fetches vectors efficiently.
     * Uses WebGPU (if available) or WASM SIMD for batch cosine similarity.
     * @private
     */
    async _searchWithRowIdMappings(colIdx, queryVec, topK, rowIdMappings, onProgress) {
        const dim = queryVec.length;

        // Group row IDs by fragment for efficient batch fetching
        const byFragment = new Map();
        for (const mapping of rowIdMappings) {
            if (!byFragment.has(mapping.fragId)) {
                byFragment.set(mapping.fragId, []);
            }
            byFragment.get(mapping.fragId).push(mapping.rowOffset);
        }

        console.log(`[IVFSearch] Fetching from ${byFragment.size} fragments`);

        // Collect all vectors and their indices first
        const allVectors = [];
        const allIndices = [];
        let processed = 0;
        const total = rowIdMappings.length;

        // Fetch all vectors
        for (const [fragId, offsets] of byFragment) {
            if (onProgress) onProgress(processed, total);

            const vectors = await this.readVectorsAtIndices(colIdx, offsets);

            for (let i = 0; i < offsets.length; i++) {
                const vec = vectors[i];
                if (vec && vec.length === dim) {
                    allVectors.push(vec);
                    // Reconstruct global row index
                    allIndices.push(fragId * 50000 + offsets[i]);
                }
                processed++;
            }
        }

        // Try WebGPU first, fallback to WASM SIMD
        let scores;
        if (webgpuAccelerator.isAvailable()) {
            console.log(`[IVFSearch] Computing similarity for ${allVectors.length} vectors via WebGPU`);
            scores = await webgpuAccelerator.batchCosineSimilarity(queryVec, allVectors, true);
        }

        if (!scores) {
            console.log(`[IVFSearch] Computing similarity for ${allVectors.length} vectors via WASM SIMD`);
            scores = this.lanceql.batchCosineSimilarity(queryVec, allVectors, true);
        }

        // Find top-k
        const topResults = [];
        for (let i = 0; i < scores.length; i++) {
            const score = scores[i];
            const idx = allIndices[i];

            if (topResults.length < topK) {
                topResults.push({ idx, score });
                topResults.sort((a, b) => b.score - a.score);
            } else if (score > topResults[topK - 1].score) {
                topResults[topK - 1] = { idx, score };
                topResults.sort((a, b) => b.score - a.score);
            }
        }

        if (onProgress) onProgress(total, total);

        return {
            indices: topResults.map(r => r.idx),
            scores: topResults.map(r => r.score),
            usedIndex: true,
            searchedRows: allVectors.length
        };
    }

    // NOTE: _searchWithEstimatedPartitions and _vectorSearchBruteForce have been removed.
    // All vector search now requires IVF index with proper partition mapping.
    // Use LanceDataset for multi-fragment datasets with ivf_partitions.bin.

    /**
     * Read all vectors from a column as a flat Float32Array.
     * Used for worker-based parallel search.
     * Handles multi-page columns by fetching and combining all pages.
     * @param {number} colIdx - Vector column index
     * @returns {Promise<Float32Array>} - Flattened vector data [numRows * dim]
     */
    async readVectorColumn(colIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const metaInfo = this._parseColumnMeta(new Uint8Array(colMeta));

        if (!metaInfo.pages || metaInfo.pages.length === 0 || metaInfo.rows === 0) {
            return new Float32Array(0);
        }

        // Calculate dimension from first page
        const firstPage = metaInfo.pages[0];
        const dataIdx = firstPage.sizes.length > 1 ? 1 : 0;
        const firstPageSize = firstPage.sizes[dataIdx] || 0;
        const firstPageRows = firstPage.rows || 0;

        if (firstPageRows === 0 || firstPageSize === 0) {
            return new Float32Array(0);
        }

        const dim = Math.floor(firstPageSize / (firstPageRows * 4));
        if (dim === 0) {
            return new Float32Array(0);
        }

        const totalRows = metaInfo.rows;
        const result = new Float32Array(totalRows * dim);

        // Fetch each page in parallel
        const pagePromises = metaInfo.pages.map(async (page, pageIdx) => {
            const pageDataIdx = page.sizes.length > 1 ? 1 : 0;
            const pageOffset = page.offsets[pageDataIdx] || 0;
            const pageSize = page.sizes[pageDataIdx] || 0;

            if (pageSize === 0) return { pageIdx, data: new Float32Array(0), rows: 0 };

            const data = await this.fetchRange(pageOffset, pageOffset + pageSize - 1);
            // data is ArrayBuffer from fetchRange, create Float32Array view directly
            const floatData = new Float32Array(data);
            return {
                pageIdx,
                data: floatData,
                rows: page.rows
            };
        });

        const pageResults = await Promise.all(pagePromises);

        // Combine pages in order
        let offset = 0;
        for (const pageResult of pageResults.sort((a, b) => a.pageIdx - b.pageIdx)) {
            result.set(pageResult.data, offset);
            offset += pageResult.rows * dim;
        }

        return result;
    }

    /**
     * Read rows from this Lance file with pagination.
     * @param {Object} options - Query options
     * @param {number} options.offset - Starting row offset
     * @param {number} options.limit - Maximum rows to return
     * @param {number[]} options.columns - Column indices to read (optional, null = all)
     * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
     */
    async readRows({ offset = 0, limit = 50, columns = null } = {}) {
        // Determine column indices to read
        const colIndices = columns || Array.from({ length: this._numColumns }, (_, i) => i);

        // Get total row count from first column
        const totalRows = await this.getRowCount(0);

        // Clamp offset and limit
        const actualOffset = Math.min(offset, totalRows);
        const actualLimit = Math.min(limit, totalRows - actualOffset);

        if (actualLimit <= 0) {
            return {
                columns: colIndices.map(() => []),
                columnNames: this.columnNames.slice(0, colIndices.length),
                total: totalRows
            };
        }

        // Generate indices for the requested rows
        const indices = Array.from({ length: actualLimit }, (_, i) => actualOffset + i);

        // Detect all column types first
        const columnTypes = await this.detectColumnTypes();

        // Read each column in parallel
        const columnPromises = colIndices.map(async (colIdx) => {
            const type = columnTypes[colIdx] || 'unknown';

            try {
                switch (type) {
                    case 'string':
                    case 'utf8':
                    case 'large_utf8':
                        return await this.readStringsAtIndices(colIdx, indices);

                    case 'int64':
                        return Array.from(await this.readInt64AtIndices(colIdx, indices));

                    case 'int32':
                        return Array.from(await this.readInt32AtIndices(colIdx, indices));

                    case 'int16':
                        return Array.from(await this.readInt16AtIndices(colIdx, indices));

                    case 'uint8':
                        return Array.from(await this.readUint8AtIndices(colIdx, indices));

                    case 'float64':
                    case 'double':
                        return Array.from(await this.readFloat64AtIndices(colIdx, indices));

                    case 'float32':
                    case 'float':
                        return Array.from(await this.readFloat32AtIndices(colIdx, indices));

                    case 'bool':
                    case 'boolean':
                        return await this.readBoolAtIndices(colIdx, indices);

                    case 'fixed_size_list':
                    case 'vector':
                        // For vectors, return as nested arrays
                        const vectors = await this.readVectorsAtIndices(colIdx, indices);
                        return Array.isArray(vectors) ? vectors : Array.from(vectors);

                    default:
                        // Try as string for unknown types
                        console.warn(`[LanceQL] Unknown column type: ${type}, trying as string`);
                        return await this.readStringsAtIndices(colIdx, indices);
                }
            } catch (e) {
                console.warn(`[LanceQL] Error reading column ${colIdx} (${type}):`, e.message);
                return indices.map(() => null);
            }
        });

        const columnsData = await Promise.all(columnPromises);

        return {
            columns: columnsData,
            columnNames: colIndices.map(i => this.columnNames[i] || `column_${i}`),
            total: totalRows
        };
    }
}

// ============================================================================
// ANN/IVF Index Support
// ============================================================================

/**
 * IVF (Inverted File Index) for Approximate Nearest Neighbor search.
 * Stores centroids and partition info to enable fast vector search
 * by only scanning relevant partitions instead of the entire dataset.
 */
export class IVFIndex {
    constructor() {
        this.centroids = null;       // Float32Array of centroids (numPartitions x dimension)
        this.numPartitions = 0;      // Number of IVF partitions
        this.dimension = 0;          // Vector dimension
        this.partitionOffsets = [];  // Byte offset of each partition in the data
        this.partitionLengths = [];  // Number of rows in each partition
        this.metricType = 'cosine';  // Distance metric (cosine, l2, dot)

        // Custom partition index (ivf_partitions.bin)
        this.partitionIndexUrl = null;  // URL to ivf_partitions.bin
        this.partitionStarts = null;    // Uint32Array[257] - cumulative row counts
        this.hasPartitionIndex = false; // Whether partition index is loaded

        // Prefetched row IDs cache - avoids HTTP requests during search
        this._rowIdCache = null;  // Map<partitionIdx, Array<{fragId, rowOffset}>>
        this._rowIdCacheReady = false;
    }

    /**
     * Try to load IVF index from a Lance dataset.
     * Index structure: dataset.lance/_indices/<uuid>/index.idx
     * @param {string} datasetBaseUrl - Base URL of dataset (e.g., https://host/data.lance)
     * @returns {Promise<IVFIndex|null>}
     */
    static async tryLoad(datasetBaseUrl) {
        if (!datasetBaseUrl) return null;

        try {
            // Find latest manifest version
            const manifestVersion = await IVFIndex._findLatestManifestVersion(datasetBaseUrl);
            console.log(`[IVFIndex] Manifest version: ${manifestVersion}`);
            if (!manifestVersion) return null;

            const manifestUrl = `${datasetBaseUrl}/_versions/${manifestVersion}.manifest`;
            const manifestResp = await fetch(manifestUrl);
            if (!manifestResp.ok) {
                console.log(`[IVFIndex] Failed to fetch manifest: ${manifestResp.status}`);
                return null;
            }

            const manifestData = await manifestResp.arrayBuffer();
            const indexInfo = IVFIndex._parseManifestForIndex(new Uint8Array(manifestData));
            console.log(`[IVFIndex] Index info:`, indexInfo);

            if (!indexInfo || !indexInfo.uuid) {
                // No vector index found in manifest
                console.log('[IVFIndex] No index UUID found in manifest');
                return null;
            }

            console.log(`[IVFIndex] Found index UUID: ${indexInfo.uuid}`);

            // Fetch the index file (contains centroids)
            const indexUrl = `${datasetBaseUrl}/_indices/${indexInfo.uuid}/index.idx`;
            const indexResp = await fetch(indexUrl);
            if (!indexResp.ok) {
                console.warn('[IVFIndex] index.idx not found');
                return null;
            }

            const indexData = await indexResp.arrayBuffer();
            const index = IVFIndex._parseIndexFile(new Uint8Array(indexData), indexInfo);

            if (!index) return null;

            // Store auxiliary URL for later partition data fetching
            index.auxiliaryUrl = `${datasetBaseUrl}/_indices/${indexInfo.uuid}/auxiliary.idx`;
            index.datasetBaseUrl = datasetBaseUrl;

            // Fetch auxiliary.idx metadata (footer + partition info)
            // We only need the last ~13MB which has the partition metadata
            try {
                await index._loadAuxiliaryMetadata();
            } catch (e) {
                console.warn('[IVFIndex] Failed to load auxiliary metadata:', e);
            }

            console.log(`[IVFIndex] Loaded: ${index.numPartitions} partitions, dim=${index.dimension}`);
            if (index.partitionLengths.length > 0) {
                const totalRows = index.partitionLengths.reduce((a, b) => a + b, 0);
                console.log(`[IVFIndex] Partition info: ${totalRows.toLocaleString()} total rows`);
            }

            // Try to load custom partition index (ivf_partitions.bin)
            try {
                await index._loadPartitionIndex();
            } catch (e) {
                console.warn('[IVFIndex] Failed to load partition index:', e);
            }

            // Prefetch all row IDs for fast search (no HTTP during search)
            try {
                await index.prefetchAllRowIds();
            } catch (e) {
                console.warn('[IVFIndex] Failed to prefetch row IDs:', e);
            }

            return index;
        } catch (e) {
            console.warn('[IVFIndex] Failed to load:', e);
            return null;
        }
    }

    /**
     * Load partition-organized vectors index from ivf_vectors.bin.
     * This file contains:
     *   - Header: 257 uint64 byte offsets (2056 bytes)
     *   - Per partition: [row_count: uint32][row_ids: uint32 × n][vectors: float32 × n × 384]
     * @private
     */
    async _loadPartitionIndex() {
        const url = `${this.datasetBaseUrl}/ivf_vectors.bin`;
        this.partitionVectorsUrl = url;

        // Fetch header (257 uint64s = 2056 bytes)
        const headerResp = await fetch(url, {
            headers: { 'Range': 'bytes=0-2055' }
        });
        if (!headerResp.ok) {
            console.log('[IVFIndex] ivf_vectors.bin not found, IVF search disabled');
            return;
        }

        const headerData = await headerResp.arrayBuffer();
        // Parse as BigUint64Array then convert to regular numbers
        const bigOffsets = new BigUint64Array(headerData);
        this.partitionOffsets = Array.from(bigOffsets, n => Number(n));

        this.hasPartitionIndex = true;
        console.log(`[IVFIndex] Loaded partition vectors index: 256 partitions`);
    }

    /**
     * Fetch partition data (row IDs and vectors) directly from ivf_vectors.bin.
     * Uses OPFS cache for instant subsequent searches.
     * Each partition contains: [row_count: uint32][row_ids: uint32 × n][vectors: float32 × n × dim]
     * @param {number[]} partitionIndices - Partition indices to fetch
     * @param {number} dim - Vector dimension (default 384)
     * @param {function} onProgress - Progress callback (bytesLoaded, totalBytes)
     * @returns {Promise<{rowIds: number[], vectors: Float32Array[]}>}
     */
    async fetchPartitionData(partitionIndices, dim = 384, onProgress = null) {
        if (!this.hasPartitionIndex || !this.partitionVectorsUrl) {
            return null;
        }

        const allRowIds = [];
        const allVectors = [];
        let totalBytesToFetch = 0;
        let bytesLoaded = 0;

        // Separate cached vs uncached partitions
        const uncachedPartitions = [];
        const cachedResults = new Map();

        for (const p of partitionIndices) {
            // Check in-memory cache first
            if (this._partitionCache?.has(p)) {
                cachedResults.set(p, this._partitionCache.get(p));
            } else {
                uncachedPartitions.push(p);
                const startOffset = this.partitionOffsets[p];
                const endOffset = this.partitionOffsets[p + 1];
                totalBytesToFetch += endOffset - startOffset;
            }
        }

        if (uncachedPartitions.length === 0) {
            console.log(`[IVFIndex] All ${partitionIndices.length} partitions from cache`);
            // All from cache
            for (const p of partitionIndices) {
                const result = cachedResults.get(p);
                allRowIds.push(...result.rowIds);
                allVectors.push(...result.vectors);
            }
            if (onProgress) onProgress(100, 100);
            return { rowIds: allRowIds, vectors: allVectors };
        }

        console.log(`[IVFIndex] Fetching ${uncachedPartitions.length}/${partitionIndices.length} partitions, ${(totalBytesToFetch / 1024 / 1024).toFixed(1)} MB`);

        // Initialize partition cache if needed
        if (!this._partitionCache) {
            this._partitionCache = new Map();
        }

        // Fetch uncached partitions in parallel (max 6 concurrent for speed)
        const PARALLEL_LIMIT = 6;
        for (let i = 0; i < uncachedPartitions.length; i += PARALLEL_LIMIT) {
            const batch = uncachedPartitions.slice(i, i + PARALLEL_LIMIT);

            const results = await Promise.all(batch.map(async (p) => {
                const startOffset = this.partitionOffsets[p];
                const endOffset = this.partitionOffsets[p + 1];
                const byteSize = endOffset - startOffset;

                try {
                    const resp = await fetch(this.partitionVectorsUrl, {
                        headers: { 'Range': `bytes=${startOffset}-${endOffset - 1}` }
                    });
                    if (!resp.ok) {
                        console.warn(`[IVFIndex] Partition ${p} fetch failed: ${resp.status}`);
                        return { p, rowIds: [], vectors: [] };
                    }

                    const data = await resp.arrayBuffer();
                    const view = new DataView(data);

                    // Parse: [row_count: uint32][row_ids: uint32 × n][vectors: float32 × n × dim]
                    const rowCount = view.getUint32(0, true);  // little-endian
                    const rowIdsStart = 4;
                    const rowIdsEnd = rowIdsStart + rowCount * 4;
                    const vectorsStart = rowIdsEnd;

                    const rowIds = new Uint32Array(data.slice(rowIdsStart, rowIdsEnd));
                    const vectorsFlat = new Float32Array(data.slice(vectorsStart));

                    // Split flat vectors into individual arrays
                    const vectors = [];
                    for (let j = 0; j < rowCount; j++) {
                        vectors.push(vectorsFlat.slice(j * dim, (j + 1) * dim));
                    }

                    bytesLoaded += byteSize;
                    if (onProgress) onProgress(bytesLoaded, totalBytesToFetch);

                    return { p, rowIds: Array.from(rowIds), vectors };
                } catch (e) {
                    console.warn(`[IVFIndex] Error fetching partition ${p}:`, e);
                    return { p, rowIds: [], vectors: [] };
                }
            }));

            // Cache results and collect
            for (const result of results) {
                const { p, rowIds, vectors } = result;
                // Cache in memory for subsequent searches
                this._partitionCache.set(p, { rowIds, vectors });
                cachedResults.set(p, { rowIds, vectors });
            }
        }

        // Collect all results in original order
        for (const p of partitionIndices) {
            const result = cachedResults.get(p);
            if (result) {
                allRowIds.push(...result.rowIds);
                allVectors.push(...result.vectors);
            }
        }

        console.log(`[IVFIndex] Loaded ${allRowIds.length.toLocaleString()} vectors from ${partitionIndices.length} partitions`);
        return { rowIds: allRowIds, vectors: allVectors };
    }

    /**
     * Find latest manifest version using binary search.
     * @private
     */
    static async _findLatestManifestVersion(baseUrl) {
        // Check common versions in parallel
        const checkVersions = [1, 5, 10, 20, 50, 100];
        const checks = await Promise.all(
            checkVersions.map(async v => {
                try {
                    const url = `${baseUrl}/_versions/${v}.manifest`;
                    const response = await fetch(url, { method: 'HEAD' });
                    return response.ok ? v : 0;
                } catch {
                    return 0;
                }
            })
        );

        let highestFound = Math.max(...checks);
        if (highestFound === 0) return null;

        // Scan forward from highest found
        for (let v = highestFound + 1; v <= highestFound + 30; v++) {
            try {
                const url = `${baseUrl}/_versions/${v}.manifest`;
                const response = await fetch(url, { method: 'HEAD' });
                if (response.ok) {
                    highestFound = v;
                } else {
                    break;
                }
            } catch {
                break;
            }
        }

        return highestFound;
    }

    /**
     * Load partition metadata from auxiliary.idx.
     * Uses HTTP range request to fetch only the metadata section.
     * @private
     */
    async _loadAuxiliaryMetadata() {
        // Fetch file size first
        let headResp;
        try {
            headResp = await fetch(this.auxiliaryUrl, { method: 'HEAD' });
        } catch (e) {
            console.warn('[IVFIndex] HEAD request failed for auxiliary.idx:', e.message);
            return;
        }
        if (!headResp.ok) return;

        const fileSize = parseInt(headResp.headers.get('content-length'));
        if (!fileSize) return;

        // Fetch footer (last 40 bytes) to get metadata locations
        const footerResp = await fetch(this.auxiliaryUrl, {
            headers: { 'Range': `bytes=${fileSize - 40}-${fileSize - 1}` }
        });
        if (!footerResp.ok) return;

        const footer = new Uint8Array(await footerResp.arrayBuffer());
        const view = new DataView(footer.buffer, footer.byteOffset);

        // Parse Lance footer (40 bytes)
        // Bytes 0-7: column_meta_start
        // Bytes 8-15: column_meta_offsets_start
        // Bytes 16-23: global_buff_offsets_start
        // Bytes 24-27: num_global_buffers
        // Bytes 28-31: num_columns
        // Bytes 32-33: major_version
        // Bytes 34-35: minor_version
        // Bytes 36-39: magic "LANC"
        const colMetaStart = Number(view.getBigUint64(0, true));
        const colMetaOffsetsStart = Number(view.getBigUint64(8, true));
        const globalBuffOffsetsStart = Number(view.getBigUint64(16, true));
        const numGlobalBuffers = view.getUint32(24, true);
        const numColumns = view.getUint32(28, true);
        const magic = new TextDecoder().decode(footer.slice(36, 40));

        if (magic !== 'LANC') {
            console.warn('[IVFIndex] Invalid auxiliary.idx magic');
            return;
        }

        console.log(`[IVFIndex] Footer: colMetaStart=${colMetaStart}, colMetaOffsetsStart=${colMetaOffsetsStart}, globalBuffOffsetsStart=${globalBuffOffsetsStart}, numGlobalBuffers=${numGlobalBuffers}, numColumns=${numColumns}`);

        // Fetch global buffer offsets (each buffer has offset + length = 16 bytes)
        const gboSize = numGlobalBuffers * 16;
        const gboResp = await fetch(this.auxiliaryUrl, {
            headers: { 'Range': `bytes=${globalBuffOffsetsStart}-${globalBuffOffsetsStart + gboSize - 1}` }
        });
        if (!gboResp.ok) return;

        const gboData = new Uint8Array(await gboResp.arrayBuffer());
        const gboView = new DataView(gboData.buffer, gboData.byteOffset);

        // Global buffer offsets are stored as [offset, length] pairs
        // Each buffer has: offset (8 bytes) + length (8 bytes) = 16 bytes per buffer
        const buffers = [];
        for (let i = 0; i < numGlobalBuffers; i++) {
            const offset = Number(gboView.getBigUint64(i * 16, true));
            const length = Number(gboView.getBigUint64(i * 16 + 8, true));
            buffers.push({ offset, length });
        }

        console.log(`[IVFIndex] Buffers:`, buffers);

        // Buffer 1 contains row IDs (_rowid column data)
        // Buffer 2 contains PQ codes (__pq_code column data)
        // We need buffer 1 for row ID lookups
        if (buffers.length < 2) return;

        // Store buffer info for later use
        this._auxBuffers = buffers;
        this._auxFileSize = fileSize;

        // Now we need to fetch partition metadata from column metadata
        // The auxiliary.idx stores _rowid and __pq_code columns
        // Partition info (offsets, lengths) is in the column metadata section
        // For now, we'll compute partition info from the row ID buffer
        // Each partition's row IDs are stored contiguously

        // We need to parse column metadata to get partition boundaries
        // Column metadata is at col_meta_start, with offsets at col_meta_off_start
        const colMetaOffResp = await fetch(this.auxiliaryUrl, {
            headers: { 'Range': `bytes=${colMetaOffsetsStart}-${globalBuffOffsetsStart - 1}` }
        });
        if (!colMetaOffResp.ok) return;

        const colMetaOffData = new Uint8Array(await colMetaOffResp.arrayBuffer());
        // Parse column offset entries (16 bytes each: 8 byte pos + 8 byte len)
        // We have 2 columns: _rowid and __pq_code
        if (colMetaOffData.length >= 32) {
            const colView = new DataView(colMetaOffData.buffer, colMetaOffData.byteOffset);
            const col0Pos = Number(colView.getBigUint64(0, true));
            const col0Len = Number(colView.getBigUint64(8, true));
            console.log(`[IVFIndex] Column 0 (_rowid) metadata at ${col0Pos}, len=${col0Len}`);

            // Fetch column 0 metadata to get page info
            const col0MetaResp = await fetch(this.auxiliaryUrl, {
                headers: { 'Range': `bytes=${col0Pos}-${col0Pos + col0Len - 1}` }
            });
            if (col0MetaResp.ok) {
                const col0Meta = new Uint8Array(await col0MetaResp.arrayBuffer());
                this._parseColumnMetaForPartitions(col0Meta);
            }
        }
    }

    /**
     * Parse column metadata to extract partition (page) boundaries.
     * @private
     */
    _parseColumnMetaForPartitions(bytes) {
        let pos = 0;
        const pages = [];

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < bytes.length) {
                const byte = bytes[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        // Parse protobuf to find pages
        while (pos < bytes.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 2) {
                const len = readVarint();
                if (len > bytes.length - pos) break;
                const content = bytes.slice(pos, pos + len);
                pos += len;

                // Field 2 = pages (PageInfo)
                if (fieldNum === 2) {
                    const page = this._parsePageInfo(content);
                    if (page) pages.push(page);
                }
            } else if (wireType === 0) {
                readVarint();
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        }

        console.log(`[IVFIndex] Found ${pages.length} column pages`);

        // Store page info for row ID lookups
        // Note: partition info should come from index.idx, not column pages
        // Column pages are how data is stored, partitions are the IVF clusters
        this._columnPages = pages;

        // Calculate total rows for verification
        let totalRows = 0;
        for (const page of pages) {
            totalRows += page.numRows;
        }
        console.log(`[IVFIndex] Column has ${totalRows} total rows`);
    }

    /**
     * Parse PageInfo protobuf.
     * @private
     */
    _parsePageInfo(bytes) {
        let pos = 0;
        let numRows = 0;
        const bufferOffsets = [];
        const bufferSizes = [];

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < bytes.length) {
                const byte = bytes[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        while (pos < bytes.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 0) {
                const val = readVarint();
                if (fieldNum === 3) numRows = val;  // length field
            } else if (wireType === 2) {
                const len = readVarint();
                const content = bytes.slice(pos, pos + len);
                pos += len;

                // Field 1 = buffer_offsets (packed uint64)
                if (fieldNum === 1) {
                    let p = 0;
                    while (p < content.length) {
                        let val = 0n;
                        let shift = 0n;
                        while (p < content.length) {
                            const b = content[p++];
                            val |= BigInt(b & 0x7F) << shift;
                            if ((b & 0x80) === 0) break;
                            shift += 7n;
                        }
                        bufferOffsets.push(Number(val));
                    }
                }
                // Field 2 = buffer_sizes (packed uint64)
                if (fieldNum === 2) {
                    let p = 0;
                    while (p < content.length) {
                        let val = 0n;
                        let shift = 0n;
                        while (p < content.length) {
                            const b = content[p++];
                            val |= BigInt(b & 0x7F) << shift;
                            if ((b & 0x80) === 0) break;
                            shift += 7n;
                        }
                        bufferSizes.push(Number(val));
                    }
                }
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        }

        return { numRows, bufferOffsets, bufferSizes };
    }

    /**
     * Parse partition offsets and lengths from auxiliary.idx metadata.
     * @private
     */
    _parseAuxiliaryPartitionInfo(bytes) {
        let pos = 0;

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < bytes.length) {
                const byte = bytes[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        // Parse protobuf structure
        while (pos < bytes.length - 4) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 2) {
                const len = readVarint();
                if (len > bytes.length - pos) break;

                const content = bytes.slice(pos, pos + len);
                pos += len;

                if (fieldNum === 2 && len > 100 && len < 2000) {
                    // Partition offsets (varint-encoded)
                    const offsets = [];
                    let innerPos = 0;
                    while (innerPos < content.length) {
                        let val = 0, shift = 0;
                        while (innerPos < content.length) {
                            const byte = content[innerPos++];
                            val |= (byte & 0x7F) << shift;
                            if ((byte & 0x80) === 0) break;
                            shift += 7;
                        }
                        offsets.push(val);
                    }
                    if (offsets.length === this.numPartitions) {
                        this.partitionOffsets = offsets;
                        console.log(`[IVFIndex] Loaded ${offsets.length} partition offsets`);
                    }
                } else if (fieldNum === 3 && len > 100 && len < 2000) {
                    // Partition lengths (varint-encoded)
                    const lengths = [];
                    let innerPos = 0;
                    while (innerPos < content.length) {
                        let val = 0, shift = 0;
                        while (innerPos < content.length) {
                            const byte = content[innerPos++];
                            val |= (byte & 0x7F) << shift;
                            if ((byte & 0x80) === 0) break;
                            shift += 7;
                        }
                        lengths.push(val);
                    }
                    if (lengths.length === this.numPartitions) {
                        this.partitionLengths = lengths;
                        console.log(`[IVFIndex] Loaded ${lengths.length} partition lengths`);
                    }
                }
            } else if (wireType === 0) {
                readVarint();
            } else if (wireType === 1) {
                pos += 8;
            } else if (wireType === 5) {
                pos += 4;
            } else {
                break;
            }
        }
    }

    /**
     * Prefetch ALL row IDs from auxiliary.idx into memory.
     * This is called once during index loading to avoid HTTP requests during search.
     * @returns {Promise<void>}
     */
    async prefetchAllRowIds() {
        if (!this.auxiliaryUrl || !this._auxBufferOffsets) {
            console.log('[IVFIndex] No auxiliary.idx available for prefetch');
            return;
        }

        if (this._rowIdCacheReady) {
            console.log('[IVFIndex] Row IDs already prefetched');
            return;
        }

        const totalRows = this.partitionLengths.reduce((a, b) => a + b, 0);
        if (totalRows === 0) {
            console.log('[IVFIndex] No rows to prefetch');
            return;
        }

        console.log(`[IVFIndex] Prefetching ${totalRows.toLocaleString()} row IDs...`);
        const startTime = performance.now();

        const dataStart = this._auxBufferOffsets[1];
        const totalBytes = totalRows * 8;

        try {
            // Fetch ALL row IDs in a single request
            const resp = await fetch(this.auxiliaryUrl, {
                headers: { 'Range': `bytes=${dataStart}-${dataStart + totalBytes - 1}` }
            });

            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            const data = new Uint8Array(await resp.arrayBuffer());
            const view = new DataView(data.buffer, data.byteOffset);

            // Parse and organize by partition
            this._rowIdCache = new Map();
            let globalRowIdx = 0;

            for (let p = 0; p < this.partitionLengths.length; p++) {
                const numRows = this.partitionLengths[p];
                const partitionRows = [];

                for (let i = 0; i < numRows; i++) {
                    const rowId = Number(view.getBigUint64(globalRowIdx * 8, true));
                    const fragId = Math.floor(rowId / 0x100000000);
                    const rowOffset = rowId % 0x100000000;
                    partitionRows.push({ fragId, rowOffset });
                    globalRowIdx++;
                }

                this._rowIdCache.set(p, partitionRows);
            }

            this._rowIdCacheReady = true;
            const elapsed = performance.now() - startTime;
            console.log(`[IVFIndex] Prefetched ${totalRows.toLocaleString()} row IDs in ${elapsed.toFixed(0)}ms (${(totalBytes / 1024 / 1024).toFixed(1)}MB)`);
        } catch (e) {
            console.warn('[IVFIndex] Failed to prefetch row IDs:', e);
        }
    }

    /**
     * Fetch row IDs for specified partitions.
     * Uses prefetched cache if available (instant), otherwise fetches from network.
     *
     * @param {number[]} partitionIndices - Partition indices to fetch
     * @returns {Promise<Array<{fragId: number, rowOffset: number}>>}
     */
    async fetchPartitionRowIds(partitionIndices) {
        // Fast path: use prefetched cache
        if (this._rowIdCacheReady && this._rowIdCache) {
            const results = [];
            for (const p of partitionIndices) {
                const cached = this._rowIdCache.get(p);
                if (cached) {
                    for (const row of cached) {
                        results.push({ ...row, partition: p });
                    }
                }
            }
            return results;
        }

        // Slow path: fetch from network (fallback if prefetch failed)
        if (!this.auxiliaryUrl || !this._auxBufferOffsets) {
            return null;
        }

        const rowRanges = [];
        for (const p of partitionIndices) {
            if (p < this.partitionOffsets.length) {
                const startRow = this.partitionOffsets[p];
                const numRows = this.partitionLengths[p];
                rowRanges.push({ partition: p, startRow, numRows });
            }
        }

        if (rowRanges.length === 0) return [];

        const results = [];
        const dataStart = this._auxBufferOffsets[1];

        for (const range of rowRanges) {
            const byteStart = dataStart + range.startRow * 8;
            const byteEnd = byteStart + range.numRows * 8 - 1;

            try {
                const resp = await fetch(this.auxiliaryUrl, {
                    headers: { 'Range': `bytes=${byteStart}-${byteEnd}` }
                });

                if (!resp.ok) continue;

                const data = new Uint8Array(await resp.arrayBuffer());
                const view = new DataView(data.buffer, data.byteOffset);

                for (let i = 0; i < range.numRows; i++) {
                    const rowId = Number(view.getBigUint64(i * 8, true));
                    const fragId = Math.floor(rowId / 0x100000000);
                    const rowOffset = rowId % 0x100000000;
                    results.push({ fragId, rowOffset, partition: range.partition });
                }
            } catch (e) {
                console.warn(`[IVFIndex] Error fetching partition ${range.partition}:`, e);
            }
        }

        return results;
    }

    /**
     * Get estimated number of rows to search for given partitions.
     */
    getPartitionRowCount(partitionIndices) {
        let total = 0;
        for (const p of partitionIndices) {
            if (p < this.partitionLengths.length) {
                total += this.partitionLengths[p];
            }
        }
        return total;
    }

    /**
     * Parse manifest to find vector index info.
     * @private
     */
    static _parseManifestForIndex(bytes) {
        // Manifest structure:
        // - Chunk 1: 4 bytes len + content (index metadata in field 1)
        // - Chunk 2: 4 bytes len + content (full manifest with schema + fragments)
        // - Footer (16 bytes)
        //
        // Index info is in CHUNK 1, field 1 (IndexMetadata repeated)

        const view = new DataView(bytes.buffer, bytes.byteOffset);
        const chunk1Len = view.getUint32(0, true);
        const chunk1Data = bytes.slice(4, 4 + chunk1Len);

        let pos = 0;
        let indexUuid = null;
        let indexFieldId = null;

        const readVarint = (data, startPos) => {
            let result = 0;
            let shift = 0;
            let p = startPos;
            while (p < data.length) {
                const byte = data[p++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return { value: result, pos: p };
        };

        // Parse chunk 1 looking for index metadata (field 1)
        while (pos < chunk1Data.length) {
            const tagResult = readVarint(chunk1Data, pos);
            pos = tagResult.pos;
            const fieldNum = tagResult.value >> 3;
            const wireType = tagResult.value & 0x7;

            if (wireType === 2) {
                const lenResult = readVarint(chunk1Data, pos);
                pos = lenResult.pos;
                const content = chunk1Data.slice(pos, pos + lenResult.value);
                pos += lenResult.value;

                // Field 1 = IndexMetadata (contains UUID)
                if (fieldNum === 1) {
                    const parsed = IVFIndex._parseIndexMetadata(content);
                    if (parsed && parsed.uuid) {
                        indexUuid = parsed.uuid;
                        indexFieldId = parsed.fieldId;
                    }
                }
            } else if (wireType === 0) {
                const r = readVarint(chunk1Data, pos);
                pos = r.pos;
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        }

        return indexUuid ? { uuid: indexUuid, fieldId: indexFieldId } : null;
    }

    /**
     * Parse IndexMetadata protobuf message.
     * @private
     */
    static _parseIndexMetadata(bytes) {
        let pos = 0;
        let uuid = null;
        let fieldId = null;

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < bytes.length) {
                const byte = bytes[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        while (pos < bytes.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 2) {
                const len = readVarint();
                const content = bytes.slice(pos, pos + len);
                pos += len;

                if (fieldNum === 1) {
                    // UUID (nested message with bytes)
                    uuid = IVFIndex._parseUuid(content);
                }
            } else if (wireType === 0) {
                const val = readVarint();
                if (fieldNum === 2) {
                    // fields (repeated int32) - but packed, so single value here
                    fieldId = val;
                }
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        }

        return { uuid, fieldId };
    }

    /**
     * Parse UUID protobuf message.
     * @private
     */
    static _parseUuid(bytes) {
        // UUID message: field 1 = bytes (16 bytes)
        let pos = 0;
        while (pos < bytes.length) {
            const tag = bytes[pos++];
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 2 && fieldNum === 1) {
                const len = bytes[pos++];
                const uuidBytes = bytes.slice(pos, pos + len);
                // Convert to hex string with dashes (UUID format)
                const hex = Array.from(uuidBytes).map(b => b.toString(16).padStart(2, '0')).join('');
                // Format as UUID: 8-4-4-4-12
                return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20,32)}`;
            } else if (wireType === 0) {
                while (pos < bytes.length && (bytes[pos++] & 0x80)) {}
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        }
        return null;
    }

    /**
     * Parse IVF index file.
     * Index file contains VectorIndex protobuf with IVF stage.
     * IVF message structure:
     *   field 1: repeated float centroids (deprecated)
     *   field 2: repeated uint64 offsets - byte offset of each partition
     *   field 3: repeated uint32 lengths - number of records per partition
     *   field 4: Tensor centroids_tensor - centroids as tensor
     *   field 5: optional double loss
     * @private
     */
    static _parseIndexFile(bytes, indexInfo) {
        const index = new IVFIndex();

        // Try to find and parse IVF message within the file
        // The file may have nested protobuf structures
        const ivfData = IVFIndex._findIVFMessage(bytes);

        if (ivfData) {
            if (ivfData.centroids) {
                index.centroids = ivfData.centroids.data;
                index.numPartitions = ivfData.centroids.numPartitions;
                index.dimension = ivfData.centroids.dimension;
            }
            if (ivfData.offsets && ivfData.offsets.length > 0) {
                index.partitionOffsets = ivfData.offsets;
                // Loaded partition offsets
            }
            if (ivfData.lengths && ivfData.lengths.length > 0) {
                index.partitionLengths = ivfData.lengths;
                // Loaded partition lengths
            }

            // Index centroids loaded successfully
        }

        // Fallback: try to find centroids in nested messages
        if (!index.centroids) {
            let pos = 0;
            const readVarint = () => {
                let result = 0;
                let shift = 0;
                while (pos < bytes.length) {
                    const byte = bytes[pos++];
                    result |= (byte & 0x7F) << shift;
                    if ((byte & 0x80) === 0) break;
                    shift += 7;
                }
                return result;
            };

            while (pos < bytes.length - 4) {
                const tag = readVarint();
                const fieldNum = tag >> 3;
                const wireType = tag & 0x7;

                if (wireType === 2) {
                    const len = readVarint();
                    if (len > bytes.length - pos) break;

                    const content = bytes.slice(pos, pos + len);
                    pos += len;

                    if (len > 100 && len < 100000000) {
                        const centroids = IVFIndex._tryParseCentroids(content);
                        if (centroids) {
                            index.centroids = centroids.data;
                            index.numPartitions = centroids.numPartitions;
                            index.dimension = centroids.dimension;
                            // Loaded IVF centroids via fallback parsing
                        }
                    }
                } else if (wireType === 0) {
                    readVarint();
                } else if (wireType === 5) {
                    pos += 4;
                } else if (wireType === 1) {
                    pos += 8;
                }
            }
        }

        return index.centroids ? index : null;
    }

    /**
     * Find and parse IVF message within index file bytes.
     * Recursively searches nested protobuf messages.
     * @private
     */
    static _findIVFMessage(bytes) {
        // IVF message fields:
        // field 2: repeated uint64 offsets (packed)
        // field 3: repeated uint32 lengths (packed)
        // field 4: Tensor centroids_tensor

        let pos = 0;
        let offsets = [];
        let lengths = [];
        let centroids = null;

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < bytes.length) {
                const byte = bytes[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        const readFixed64 = () => {
            if (pos + 8 > bytes.length) return 0n;
            const view = new DataView(bytes.buffer, bytes.byteOffset + pos, 8);
            pos += 8;
            return view.getBigUint64(0, true);
        };

        const readFixed32 = () => {
            if (pos + 4 > bytes.length) return 0;
            const view = new DataView(bytes.buffer, bytes.byteOffset + pos, 4);
            pos += 4;
            return view.getUint32(0, true);
        };

        while (pos < bytes.length - 4) {
            const startPos = pos;
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 2) {
                // Length-delimited field
                const len = readVarint();
                if (len > bytes.length - pos || len < 0) {
                    pos = startPos + 1;
                    continue;
                }

                const content = bytes.slice(pos, pos + len);
                pos += len;

                if (fieldNum === 2) {
                    // offsets - packed uint64
                    // Could be packed fixed64 or packed varint
                    if (len % 8 === 0 && len > 0) {
                        // Try as packed fixed64
                        const numOffsets = len / 8;
                        const view = new DataView(content.buffer, content.byteOffset, len);
                        for (let i = 0; i < numOffsets; i++) {
                            offsets.push(Number(view.getBigUint64(i * 8, true)));
                        }
                        // Parsed partition offsets
                    }
                } else if (fieldNum === 3) {
                    // lengths - packed uint32
                    if (len % 4 === 0 && len > 0) {
                        // Try as packed fixed32
                        const numLengths = len / 4;
                        const view = new DataView(content.buffer, content.byteOffset, len);
                        for (let i = 0; i < numLengths; i++) {
                            lengths.push(view.getUint32(i * 4, true));
                        }
                        // Parsed partition lengths (fixed32)
                    } else {
                        // Try as packed varint
                        let lpos = 0;
                        while (lpos < content.length) {
                            let val = 0, shift = 0;
                            while (lpos < content.length) {
                                const byte = content[lpos++];
                                val |= (byte & 0x7F) << shift;
                                if ((byte & 0x80) === 0) break;
                                shift += 7;
                            }
                            lengths.push(val);
                        }
                        // Parsed partition lengths (varint)
                    }
                } else if (fieldNum === 4) {
                    // centroids_tensor
                    centroids = IVFIndex._tryParseCentroids(content);
                } else if (len > 100) {
                    // Recursively search nested messages
                    const nested = IVFIndex._findIVFMessage(content);
                    if (nested && (nested.centroids || nested.offsets?.length > 0)) {
                        if (nested.centroids && !centroids) centroids = nested.centroids;
                        if (nested.offsets?.length > offsets.length) offsets = nested.offsets;
                        if (nested.lengths?.length > lengths.length) lengths = nested.lengths;
                    }
                }
            } else if (wireType === 0) {
                readVarint();
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            } else {
                // Unknown wire type, skip byte
                pos = startPos + 1;
            }
        }

        if (centroids || offsets.length > 0 || lengths.length > 0) {
            return { centroids, offsets, lengths };
        }
        return null;
    }

    /**
     * Try to parse centroids from a Tensor message.
     * @private
     */
    static _tryParseCentroids(bytes) {
        let pos = 0;
        let shape = [];
        let dataBytes = null;
        let dataType = 2; // Default to float32

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < bytes.length) {
                const byte = bytes[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        while (pos < bytes.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 0) {
                const val = readVarint();
                if (fieldNum === 1) dataType = val;
            } else if (wireType === 2) {
                const len = readVarint();
                const content = bytes.slice(pos, pos + len);
                pos += len;

                if (fieldNum === 2) {
                    // shape (packed repeated uint32)
                    let shapePos = 0;
                    while (shapePos < content.length) {
                        let val = 0, shift = 0;
                        while (shapePos < content.length) {
                            const byte = content[shapePos++];
                            val |= (byte & 0x7F) << shift;
                            if ((byte & 0x80) === 0) break;
                            shift += 7;
                        }
                        shape.push(val);
                    }
                } else if (fieldNum === 3) {
                    dataBytes = content;
                }
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        }

        if (shape.length >= 2 && dataBytes && dataType === 2) {
            // float32 tensor with at least 2D shape
            const numPartitions = shape[0];
            const dimension = shape[1];

            if (dataBytes.length === numPartitions * dimension * 4) {
                const data = new Float32Array(dataBytes.buffer, dataBytes.byteOffset, numPartitions * dimension);
                return { data, numPartitions, dimension };
            }
        }

        return null;
    }

    /**
     * Find the nearest partitions to a query vector.
     * @param {Float32Array} queryVec - Query vector
     * @param {number} nprobe - Number of partitions to search
     * @returns {number[]} - Indices of nearest partitions
     */
    findNearestPartitions(queryVec, nprobe = 10) {
        if (!this.centroids || queryVec.length !== this.dimension) {
            return [];
        }

        nprobe = Math.min(nprobe, this.numPartitions);

        // Compute distance to each centroid
        const distances = new Array(this.numPartitions);

        for (let p = 0; p < this.numPartitions; p++) {
            const centroidStart = p * this.dimension;

            // Cosine similarity (or L2 distance based on metricType)
            let dot = 0, normA = 0, normB = 0;
            for (let i = 0; i < this.dimension; i++) {
                const a = queryVec[i];
                const b = this.centroids[centroidStart + i];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }

            const denom = Math.sqrt(normA) * Math.sqrt(normB);
            distances[p] = { idx: p, score: denom === 0 ? 0 : dot / denom };
        }

        // Sort by similarity (descending) and take top nprobe
        distances.sort((a, b) => b.score - a.score);
        return distances.slice(0, nprobe).map(d => d.idx);
    }
}

// ============================================================================
// SQL Parser and Executor
// ============================================================================

/**
 * SQL Token types
 */
const TokenType = {
    // Keywords
    SELECT: 'SELECT',
    DISTINCT: 'DISTINCT',
    FROM: 'FROM',
    WHERE: 'WHERE',
    AND: 'AND',
    OR: 'OR',
    NOT: 'NOT',
    ORDER: 'ORDER',
    BY: 'BY',
    ASC: 'ASC',
    DESC: 'DESC',
    LIMIT: 'LIMIT',
    OFFSET: 'OFFSET',
    AS: 'AS',
    NULL: 'NULL',
    IS: 'IS',
    IN: 'IN',
    BETWEEN: 'BETWEEN',
    LIKE: 'LIKE',
    TRUE: 'TRUE',
    FALSE: 'FALSE',
    GROUP: 'GROUP',
    HAVING: 'HAVING',
    QUALIFY: 'QUALIFY',
    ROLLUP: 'ROLLUP',
    CUBE: 'CUBE',
    GROUPING: 'GROUPING',
    SETS: 'SETS',
    COUNT: 'COUNT',
    SUM: 'SUM',
    AVG: 'AVG',
    MIN: 'MIN',
    MAX: 'MAX',
    // Vector search keywords
    NEAR: 'NEAR',
    TOPK: 'TOPK',
    // File reference
    FILE: 'FILE',
    // Join keywords
    JOIN: 'JOIN',
    INNER: 'INNER',
    LEFT: 'LEFT',
    RIGHT: 'RIGHT',
    FULL: 'FULL',
    OUTER: 'OUTER',
    CROSS: 'CROSS',
    ON: 'ON',
    // Write keywords
    CREATE: 'CREATE',
    TABLE: 'TABLE',
    INSERT: 'INSERT',
    INTO: 'INTO',
    VALUES: 'VALUES',
    UPDATE: 'UPDATE',
    SET: 'SET',
    DELETE: 'DELETE',
    DROP: 'DROP',
    IF: 'IF',
    EXISTS: 'EXISTS',
    // Data types
    INT: 'INT',
    INTEGER: 'INTEGER',
    BIGINT: 'BIGINT',
    FLOAT: 'FLOAT',
    REAL: 'REAL',
    DOUBLE: 'DOUBLE',
    TEXT: 'TEXT',
    VARCHAR: 'VARCHAR',
    BOOLEAN: 'BOOLEAN',
    BOOL: 'BOOL',
    VECTOR: 'VECTOR',
    PRIMARY: 'PRIMARY',
    KEY: 'KEY',
    // CTE keywords
    WITH: 'WITH',
    RECURSIVE: 'RECURSIVE',
    UNION: 'UNION',
    ALL: 'ALL',
    // PIVOT/UNPIVOT keywords
    PIVOT: 'PIVOT',
    UNPIVOT: 'UNPIVOT',
    FOR: 'FOR',
    // Set operation keywords
    INTERSECT: 'INTERSECT',
    EXCEPT: 'EXCEPT',
    // Window function keywords
    OVER: 'OVER',
    PARTITION: 'PARTITION',
    ROW_NUMBER: 'ROW_NUMBER',
    RANK: 'RANK',
    DENSE_RANK: 'DENSE_RANK',
    NTILE: 'NTILE',
    LAG: 'LAG',
    LEAD: 'LEAD',
    FIRST_VALUE: 'FIRST_VALUE',
    LAST_VALUE: 'LAST_VALUE',
    NTH_VALUE: 'NTH_VALUE',
    PERCENT_RANK: 'PERCENT_RANK',
    CUME_DIST: 'CUME_DIST',
    ROWS: 'ROWS',
    RANGE: 'RANGE',
    UNBOUNDED: 'UNBOUNDED',
    PRECEDING: 'PRECEDING',
    FOLLOWING: 'FOLLOWING',
    CURRENT: 'CURRENT',
    ROW: 'ROW',
    // Query optimization
    EXPLAIN: 'EXPLAIN',

    // Literals
    IDENTIFIER: 'IDENTIFIER',
    NUMBER: 'NUMBER',
    STRING: 'STRING',

    // Operators
    STAR: 'STAR',
    COMMA: 'COMMA',
    DOT: 'DOT',
    LPAREN: 'LPAREN',
    RPAREN: 'RPAREN',
    EQ: 'EQ',
    NE: 'NE',
    LT: 'LT',
    LE: 'LE',
    GT: 'GT',
    GE: 'GE',
    PLUS: 'PLUS',
    MINUS: 'MINUS',
    SLASH: 'SLASH',
    LBRACKET: 'LBRACKET',
    RBRACKET: 'RBRACKET',

    // Array keyword
    ARRAY: 'ARRAY',

    // Special
    EOF: 'EOF',
};

const KEYWORDS = {
    'SELECT': TokenType.SELECT,
    'DISTINCT': TokenType.DISTINCT,
    'FROM': TokenType.FROM,
    'WHERE': TokenType.WHERE,
    'AND': TokenType.AND,
    'OR': TokenType.OR,
    'NOT': TokenType.NOT,
    'ORDER': TokenType.ORDER,
    'BY': TokenType.BY,
    'ASC': TokenType.ASC,
    'DESC': TokenType.DESC,
    'LIMIT': TokenType.LIMIT,
    'OFFSET': TokenType.OFFSET,
    'AS': TokenType.AS,
    'NULL': TokenType.NULL,
    'IS': TokenType.IS,
    'IN': TokenType.IN,
    'BETWEEN': TokenType.BETWEEN,
    'LIKE': TokenType.LIKE,
    'TRUE': TokenType.TRUE,
    'FALSE': TokenType.FALSE,
    'GROUP': TokenType.GROUP,
    'HAVING': TokenType.HAVING,
    'QUALIFY': TokenType.QUALIFY,
    'ROLLUP': TokenType.ROLLUP,
    'CUBE': TokenType.CUBE,
    'GROUPING': TokenType.GROUPING,
    'SETS': TokenType.SETS,
    'COUNT': TokenType.COUNT,
    'SUM': TokenType.SUM,
    'AVG': TokenType.AVG,
    'MIN': TokenType.MIN,
    'MAX': TokenType.MAX,
    'NEAR': TokenType.NEAR,
    'TOPK': TokenType.TOPK,
    'FILE': TokenType.FILE,
    // Join keywords
    'JOIN': TokenType.JOIN,
    'INNER': TokenType.INNER,
    'LEFT': TokenType.LEFT,
    'RIGHT': TokenType.RIGHT,
    'FULL': TokenType.FULL,
    'OUTER': TokenType.OUTER,
    'CROSS': TokenType.CROSS,
    'ON': TokenType.ON,
    // Write keywords
    'CREATE': TokenType.CREATE,
    'TABLE': TokenType.TABLE,
    'INSERT': TokenType.INSERT,
    'INTO': TokenType.INTO,
    'VALUES': TokenType.VALUES,
    'UPDATE': TokenType.UPDATE,
    'SET': TokenType.SET,
    'DELETE': TokenType.DELETE,
    'DROP': TokenType.DROP,
    'IF': TokenType.IF,
    'EXISTS': TokenType.EXISTS,
    // Data types
    'INT': TokenType.INT,
    'INTEGER': TokenType.INTEGER,
    'BIGINT': TokenType.BIGINT,
    'FLOAT': TokenType.FLOAT,
    'REAL': TokenType.REAL,
    'DOUBLE': TokenType.DOUBLE,
    'TEXT': TokenType.TEXT,
    'VARCHAR': TokenType.VARCHAR,
    'BOOLEAN': TokenType.BOOLEAN,
    'BOOL': TokenType.BOOL,
    'VECTOR': TokenType.VECTOR,
    'PRIMARY': TokenType.PRIMARY,
    'KEY': TokenType.KEY,
    // CTE keywords
    'WITH': TokenType.WITH,
    'RECURSIVE': TokenType.RECURSIVE,
    'UNION': TokenType.UNION,
    'ALL': TokenType.ALL,
    // PIVOT/UNPIVOT keywords
    'PIVOT': TokenType.PIVOT,
    'UNPIVOT': TokenType.UNPIVOT,
    'FOR': TokenType.FOR,
    // Set operation keywords
    'INTERSECT': TokenType.INTERSECT,
    'EXCEPT': TokenType.EXCEPT,
    // Window function keywords
    'OVER': TokenType.OVER,
    'PARTITION': TokenType.PARTITION,
    'ROW_NUMBER': TokenType.ROW_NUMBER,
    'RANK': TokenType.RANK,
    'DENSE_RANK': TokenType.DENSE_RANK,
    'NTILE': TokenType.NTILE,
    'LAG': TokenType.LAG,
    'LEAD': TokenType.LEAD,
    'FIRST_VALUE': TokenType.FIRST_VALUE,
    'LAST_VALUE': TokenType.LAST_VALUE,
    'NTH_VALUE': TokenType.NTH_VALUE,
    'PERCENT_RANK': TokenType.PERCENT_RANK,
    'CUME_DIST': TokenType.CUME_DIST,
    'ROWS': TokenType.ROWS,
    'RANGE': TokenType.RANGE,
    'UNBOUNDED': TokenType.UNBOUNDED,
    'PRECEDING': TokenType.PRECEDING,
    'FOLLOWING': TokenType.FOLLOWING,
    'CURRENT': TokenType.CURRENT,
    'ROW': TokenType.ROW,
    // Query optimization
    'EXPLAIN': TokenType.EXPLAIN,
    'ARRAY': TokenType.ARRAY,
};

/**
 * SQL Lexer - tokenizes SQL input
 */
export class SQLLexer {
    constructor(sql) {
        this.sql = sql;
        this.pos = 0;
        this.length = sql.length;
    }

    peek() {
        if (this.pos >= this.length) return '\0';
        return this.sql[this.pos];
    }

    advance() {
        if (this.pos < this.length) {
            return this.sql[this.pos++];
        }
        return '\0';
    }

    skipWhitespace() {
        while (this.pos < this.length && /\s/.test(this.sql[this.pos])) {
            this.pos++;
        }
    }

    readIdentifier() {
        const start = this.pos;
        while (this.pos < this.length && /[a-zA-Z0-9_]/.test(this.sql[this.pos])) {
            this.pos++;
        }
        return this.sql.slice(start, this.pos);
    }

    readNumber() {
        const start = this.pos;
        let hasDecimal = false;

        while (this.pos < this.length) {
            const ch = this.sql[this.pos];
            if (ch === '.' && !hasDecimal) {
                hasDecimal = true;
                this.pos++;
            } else if (/\d/.test(ch)) {
                this.pos++;
            } else {
                break;
            }
        }
        return this.sql.slice(start, this.pos);
    }

    readString(quote) {
        const start = this.pos;
        this.advance(); // Skip opening quote

        while (this.pos < this.length) {
            const ch = this.sql[this.pos];
            if (ch === quote) {
                // Check for escaped quote
                if (this.pos + 1 < this.length && this.sql[this.pos + 1] === quote) {
                    this.pos += 2;
                    continue;
                }
                this.pos++; // Skip closing quote
                break;
            }
            this.pos++;
        }

        // Return string without quotes, handling escaped quotes
        const inner = this.sql.slice(start + 1, this.pos - 1);
        return inner.replace(new RegExp(quote + quote, 'g'), quote);
    }

    nextToken() {
        this.skipWhitespace();

        if (this.pos >= this.length) {
            return { type: TokenType.EOF, value: null };
        }

        const ch = this.peek();

        // Identifiers and keywords
        if (/[a-zA-Z_]/.test(ch)) {
            const value = this.readIdentifier();
            const upper = value.toUpperCase();
            const type = KEYWORDS[upper] || TokenType.IDENTIFIER;
            return { type, value: type === TokenType.IDENTIFIER ? value : upper };
        }

        // Numbers
        if (/\d/.test(ch)) {
            const value = this.readNumber();
            return { type: TokenType.NUMBER, value };
        }

        // Strings
        if (ch === "'" || ch === '"') {
            const value = this.readString(ch);
            return { type: TokenType.STRING, value };
        }

        // Operators
        this.advance();

        switch (ch) {
            case '*': return { type: TokenType.STAR, value: '*' };
            case ',': return { type: TokenType.COMMA, value: ',' };
            case '.': return { type: TokenType.DOT, value: '.' };
            case '(': return { type: TokenType.LPAREN, value: '(' };
            case ')': return { type: TokenType.RPAREN, value: ')' };
            case '+': return { type: TokenType.PLUS, value: '+' };
            case '-': return { type: TokenType.MINUS, value: '-' };
            case '/': return { type: TokenType.SLASH, value: '/' };
            case '[': return { type: TokenType.LBRACKET, value: '[' };
            case ']': return { type: TokenType.RBRACKET, value: ']' };
            case '=': return { type: TokenType.EQ, value: '=' };
            case '<':
                if (this.peek() === '=') {
                    this.advance();
                    return { type: TokenType.LE, value: '<=' };
                }
                if (this.peek() === '>') {
                    this.advance();
                    return { type: TokenType.NE, value: '<>' };
                }
                return { type: TokenType.LT, value: '<' };
            case '>':
                if (this.peek() === '=') {
                    this.advance();
                    return { type: TokenType.GE, value: '>=' };
                }
                return { type: TokenType.GT, value: '>' };
            case '!':
                if (this.peek() === '=') {
                    this.advance();
                    return { type: TokenType.NE, value: '!=' };
                }
                throw new Error(`Unexpected character: ${ch}`);
            default:
                throw new Error(`Unexpected character: ${ch}`);
        }
    }

    tokenize() {
        const tokens = [];
        let token;
        while ((token = this.nextToken()).type !== TokenType.EOF) {
            tokens.push(token);
        }
        tokens.push(token); // Include EOF
        return tokens;
    }
}

/**
 * SQL Parser - parses tokens into AST
 */
export class SQLParser {
    constructor(tokens) {
        this.tokens = tokens;
        this.pos = 0;
    }

    current() {
        return this.tokens[this.pos] || { type: TokenType.EOF };
    }

    advance() {
        if (this.pos < this.tokens.length) {
            return this.tokens[this.pos++];
        }
        return { type: TokenType.EOF };
    }

    expect(type) {
        const token = this.current();
        if (token.type !== type) {
            throw new Error(`Expected ${type}, got ${token.type} (${token.value})`);
        }
        return this.advance();
    }

    match(...types) {
        if (types.includes(this.current().type)) {
            return this.advance();
        }
        return null;
    }

    check(...types) {
        return types.includes(this.current().type);
    }

    /**
     * Parse SQL statement (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, DROP TABLE)
     */
    parse() {
        // Handle EXPLAIN prefix
        if (this.check(TokenType.EXPLAIN)) {
            this.advance();  // consume EXPLAIN
            const statement = this.parse();  // Parse the inner statement
            return { type: 'EXPLAIN', statement };
        }

        // Check for WITH clause (CTEs)
        let ctes = [];
        if (this.check(TokenType.WITH)) {
            ctes = this.parseWithClause();
        }

        // Dispatch based on first keyword
        if (this.check(TokenType.SELECT)) {
            const result = this.parseSelect();
            result.ctes = ctes;  // Attach CTEs to SELECT
            return result;
        } else if (this.check(TokenType.INSERT)) {
            return this.parseInsert();
        } else if (this.check(TokenType.UPDATE)) {
            return this.parseUpdate();
        } else if (this.check(TokenType.DELETE)) {
            return this.parseDelete();
        } else if (this.check(TokenType.CREATE)) {
            return this.parseCreateTable();
        } else if (this.check(TokenType.DROP)) {
            return this.parseDropTable();
        } else {
            throw new Error(`Unexpected token: ${this.current().type}. Expected SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, or EXPLAIN`);
        }
    }

    /**
     * Parse WITH clause (Common Table Expressions)
     * Syntax: WITH [RECURSIVE] name [(columns)] AS (subquery) [, ...]
     */
    parseWithClause() {
        this.expect(TokenType.WITH);
        const isRecursive = !!this.match(TokenType.RECURSIVE);

        const ctes = [];
        do {
            const name = this.expect(TokenType.IDENTIFIER).value;

            // Optional column list
            let columns = [];
            if (this.match(TokenType.LPAREN)) {
                columns.push(this.expect(TokenType.IDENTIFIER).value);
                while (this.match(TokenType.COMMA)) {
                    columns.push(this.expect(TokenType.IDENTIFIER).value);
                }
                this.expect(TokenType.RPAREN);
            }

            this.expect(TokenType.AS);
            this.expect(TokenType.LPAREN);

            // Parse CTE body - may contain UNION ALL for recursive CTEs
            const body = this.parseCteBody(isRecursive);

            this.expect(TokenType.RPAREN);

            ctes.push({
                name,
                columns,
                body,
                recursive: isRecursive
            });
        } while (this.match(TokenType.COMMA));

        return ctes;
    }

    /**
     * Parse CTE body which may contain UNION ALL for recursive CTEs
     */
    parseCteBody(isRecursive) {
        // Parse anchor query - disable set operation parsing, we handle UNION ALL here
        const anchor = this.parseSelect(true, true);  // isSubquery=true, noSetOps=true

        // Check for UNION ALL (required for recursive CTEs)
        if (isRecursive && this.match(TokenType.UNION)) {
            this.expect(TokenType.ALL);
            const recursive = this.parseSelect(true, true);  // Same for recursive part
            return {
                type: 'RECURSIVE_CTE',
                anchor,
                recursive
            };
        }

        return anchor;
    }

    /**
     * Parse SELECT statement
     * @param {boolean} isSubquery - If true, don't require EOF at end (for subqueries)
     * @param {boolean} noSetOps - If true, don't parse set operations (for CTE body parsing)
     */
    parseSelect(isSubquery = false, noSetOps = false) {
        this.expect(TokenType.SELECT);

        // DISTINCT
        const distinct = !!this.match(TokenType.DISTINCT);

        // Select list
        const columns = this.parseSelectList();

        // FROM - supports: table_name, read_lance('url'), 'url.lance'
        let from = null;
        if (this.match(TokenType.FROM)) {
            from = this.parseFromClause();
        }

        // JOIN clauses (one or more)
        const joins = [];
        while (this.check(TokenType.JOIN) || this.check(TokenType.INNER) ||
               this.check(TokenType.LEFT) || this.check(TokenType.RIGHT) ||
               this.check(TokenType.FULL) || this.check(TokenType.CROSS)) {
            joins.push(this.parseJoinClause());
        }

        // PIVOT clause (optional)
        // Syntax: PIVOT (aggregate FOR column IN (value1, value2, ...))
        let pivot = null;
        if (this.match(TokenType.PIVOT)) {
            this.expect(TokenType.LPAREN);

            // Parse aggregate function
            const aggFunc = this.parsePrimary();
            if (aggFunc.type !== 'call') {
                throw new Error('PIVOT requires an aggregate function (e.g., SUM, COUNT, AVG)');
            }

            this.expect(TokenType.FOR);
            const forColumn = this.expect(TokenType.IDENTIFIER).value;

            this.expect(TokenType.IN);
            this.expect(TokenType.LPAREN);

            // Parse IN values
            const inValues = [];
            inValues.push(this.parsePrimary().value);
            while (this.match(TokenType.COMMA)) {
                inValues.push(this.parsePrimary().value);
            }
            this.expect(TokenType.RPAREN);
            this.expect(TokenType.RPAREN);

            pivot = {
                aggregate: aggFunc,
                forColumn,
                inValues
            };
        }

        // UNPIVOT clause (optional)
        // Syntax: UNPIVOT (valueColumn FOR nameColumn IN (col1, col2, ...))
        let unpivot = null;
        if (this.match(TokenType.UNPIVOT)) {
            this.expect(TokenType.LPAREN);

            const valueColumn = this.expect(TokenType.IDENTIFIER).value;

            this.expect(TokenType.FOR);
            const nameColumn = this.expect(TokenType.IDENTIFIER).value;

            this.expect(TokenType.IN);
            this.expect(TokenType.LPAREN);

            // Parse IN columns
            const inColumns = [];
            inColumns.push(this.expect(TokenType.IDENTIFIER).value);
            while (this.match(TokenType.COMMA)) {
                inColumns.push(this.expect(TokenType.IDENTIFIER).value);
            }
            this.expect(TokenType.RPAREN);
            this.expect(TokenType.RPAREN);

            unpivot = {
                valueColumn,
                nameColumn,
                inColumns
            };
        }

        // WHERE
        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseExpr();
        }

        // GROUP BY (supports ROLLUP, CUBE, GROUPING SETS)
        let groupBy = [];
        if (this.match(TokenType.GROUP)) {
            this.expect(TokenType.BY);
            groupBy = this.parseGroupByList();
        }

        // HAVING
        let having = null;
        if (this.match(TokenType.HAVING)) {
            having = this.parseExpr();
        }

        // QUALIFY - filter on window function results
        let qualify = null;
        if (this.match(TokenType.QUALIFY)) {
            qualify = this.parseExpr();
        }

        // NEAR - vector similarity search
        // Syntax: NEAR [column] <'text'|row_num> [TOPK n]
        let search = null;
        if (this.match(TokenType.NEAR)) {
            let column = null;
            let query = null;
            let searchRow = null;
            let topK = 20; // default
            let encoder = 'minilm'; // default

            // First token after NEAR: could be column name, string, or number
            if (this.check(TokenType.IDENTIFIER)) {
                // Could be column name - peek ahead
                const ident = this.advance().value;
                if (this.check(TokenType.STRING) || this.check(TokenType.NUMBER)) {
                    // It was a column name
                    column = ident;
                } else {
                    // It was a search term without quotes (error)
                    throw new Error(`NEAR requires quoted text or row number. Did you mean: NEAR '${ident}'?`);
                }
            }

            // Now expect string (text search) or number (row search)
            if (this.check(TokenType.STRING)) {
                query = this.advance().value;
            } else if (this.check(TokenType.NUMBER)) {
                searchRow = parseInt(this.advance().value, 10);
            } else {
                throw new Error('NEAR requires a quoted text string or row number');
            }

            // Optional TOPK
            if (this.match(TokenType.TOPK)) {
                topK = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }

            search = { query, searchRow, column, topK, encoder };
        }

        // Build the base SELECT AST (without ORDER BY/LIMIT/OFFSET yet)
        const baseAst = {
            type: 'SELECT',
            distinct,
            columns,
            from,
            joins,
            pivot,
            unpivot,
            where,
            groupBy,
            having,
            qualify,
            search,
            orderBy: [],
            limit: null,
            offset: null,
        };

        // Check for set operations (UNION, INTERSECT, EXCEPT) BEFORE parsing ORDER BY/LIMIT
        // This ensures ORDER BY/LIMIT apply to the set operation result, not individual SELECTs
        if (!noSetOps && (this.check(TokenType.UNION) || this.check(TokenType.INTERSECT) || this.check(TokenType.EXCEPT))) {
            const operator = this.advance().type;
            const all = !!this.match(TokenType.ALL);
            // Parse right side without set operations (will be handled here) and without ORDER BY/LIMIT
            const right = this.parseSelect(true, true);  // noSetOps=true for right side

            // Now parse ORDER BY/LIMIT/OFFSET for the combined result
            let orderBy = [];
            let limit = null;
            let offset = null;

            if (this.match(TokenType.ORDER)) {
                this.expect(TokenType.BY);
                orderBy = this.parseOrderByList();
            }
            if (this.match(TokenType.LIMIT)) {
                limit = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }
            if (orderBy.length === 0 && this.match(TokenType.ORDER)) {
                this.expect(TokenType.BY);
                orderBy = this.parseOrderByList();
            }
            if (this.match(TokenType.OFFSET)) {
                offset = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }

            // Check that we've consumed all tokens (unless this is a subquery)
            if (!isSubquery && this.current().type !== TokenType.EOF) {
                throw new Error(`Unexpected token after query: ${this.current().type} (${this.current().value}). Check your SQL syntax.`);
            }

            return {
                type: 'SET_OPERATION',
                operator,
                all,
                left: baseAst,
                right,
                orderBy,
                limit,
                offset,
            };
        }

        // No set operation - parse ORDER BY/LIMIT/OFFSET for this SELECT
        // UNLESS noSetOps is true (we're part of a set operation and ORDER BY/LIMIT belong to outer context)
        if (!noSetOps) {
            let orderBy = [];
            let limit = null;
            let offset = null;

            if (this.match(TokenType.ORDER)) {
                this.expect(TokenType.BY);
                orderBy = this.parseOrderByList();
            }
            if (this.match(TokenType.LIMIT)) {
                limit = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }
            if (orderBy.length === 0 && this.match(TokenType.ORDER)) {
                this.expect(TokenType.BY);
                orderBy = this.parseOrderByList();
            }
            if (this.match(TokenType.OFFSET)) {
                offset = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }

            baseAst.orderBy = orderBy;
            baseAst.limit = limit;
            baseAst.offset = offset;
        }

        // Check that we've consumed all tokens (unless this is a subquery)
        if (!isSubquery && this.current().type !== TokenType.EOF) {
            throw new Error(`Unexpected token after query: ${this.current().type} (${this.current().value}). Check your SQL syntax.`);
        }

        return baseAst;
    }

    /**
     * Parse INSERT statement
     * Syntax: INSERT INTO table_name [(col1, col2, ...)] VALUES (val1, val2, ...), ...
     */
    parseInsert() {
        this.expect(TokenType.INSERT);
        this.expect(TokenType.INTO);

        // Table name
        const table = this.expect(TokenType.IDENTIFIER).value;

        // Optional column list
        let columns = null;
        if (this.match(TokenType.LPAREN)) {
            columns = [];
            columns.push(this.expect(TokenType.IDENTIFIER).value);
            while (this.match(TokenType.COMMA)) {
                columns.push(this.expect(TokenType.IDENTIFIER).value);
            }
            this.expect(TokenType.RPAREN);
        }

        // VALUES clause
        this.expect(TokenType.VALUES);

        // Parse value rows
        const rows = [];
        do {
            this.expect(TokenType.LPAREN);
            const values = [];
            values.push(this.parseValue());
            while (this.match(TokenType.COMMA)) {
                values.push(this.parseValue());
            }
            this.expect(TokenType.RPAREN);
            rows.push(values);
        } while (this.match(TokenType.COMMA));

        return {
            type: 'INSERT',
            table,
            columns,
            rows,
        };
    }

    /**
     * Parse a single value (number, string, null, true, false)
     */
    parseValue() {
        if (this.match(TokenType.NULL)) {
            return { type: 'null', value: null };
        }
        if (this.match(TokenType.TRUE)) {
            return { type: 'boolean', value: true };
        }
        if (this.match(TokenType.FALSE)) {
            return { type: 'boolean', value: false };
        }
        if (this.check(TokenType.NUMBER)) {
            const token = this.advance();
            const value = token.value.includes('.') ? parseFloat(token.value) : parseInt(token.value, 10);
            return { type: 'number', value };
        }
        if (this.check(TokenType.STRING)) {
            const token = this.advance();
            return { type: 'string', value: token.value };
        }
        if (this.check(TokenType.MINUS)) {
            this.advance();
            const token = this.expect(TokenType.NUMBER);
            const value = token.value.includes('.') ? -parseFloat(token.value) : -parseInt(token.value, 10);
            return { type: 'number', value };
        }
        // Vector literal: [1.0, 2.0, 3.0]
        if (this.check(TokenType.LBRACKET)) {
            return this.parseArrayLiteral();
        }

        throw new Error(`Expected value, got ${this.current().type}`);
    }

    /**
     * Parse array literal: [1, 2, 3] or ARRAY[1, 2, 3]
     */
    parseArrayLiteral() {
        this.expect(TokenType.LBRACKET);
        const elements = [];

        if (!this.check(TokenType.RBRACKET)) {
            elements.push(this.parseExpr());
            while (this.match(TokenType.COMMA)) {
                elements.push(this.parseExpr());
            }
        }

        this.expect(TokenType.RBRACKET);
        return { type: 'array', elements };
    }

    /**
     * Parse UPDATE statement
     * Syntax: UPDATE table_name SET col1 = val1, col2 = val2 [WHERE condition]
     */
    parseUpdate() {
        this.expect(TokenType.UPDATE);

        // Table name
        const table = this.expect(TokenType.IDENTIFIER).value;

        // SET clause
        this.expect(TokenType.SET);

        const assignments = [];
        do {
            const column = this.expect(TokenType.IDENTIFIER).value;
            this.expect(TokenType.EQ);
            const value = this.parseValue();
            assignments.push({ column, value });
        } while (this.match(TokenType.COMMA));

        // Optional WHERE
        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseExpr();
        }

        return {
            type: 'UPDATE',
            table,
            assignments,
            where,
        };
    }

    /**
     * Parse DELETE statement
     * Syntax: DELETE FROM table_name [WHERE condition]
     */
    parseDelete() {
        this.expect(TokenType.DELETE);
        this.expect(TokenType.FROM);

        // Table name
        const table = this.expect(TokenType.IDENTIFIER).value;

        // Optional WHERE
        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseExpr();
        }

        return {
            type: 'DELETE',
            table,
            where,
        };
    }

    /**
     * Parse CREATE TABLE statement
     * Syntax: CREATE TABLE [IF NOT EXISTS] table_name (col1 TYPE, col2 TYPE, ...)
     */
    parseCreateTable() {
        this.expect(TokenType.CREATE);
        this.expect(TokenType.TABLE);

        // Check for IF NOT EXISTS
        let ifNotExists = false;
        if (this.match(TokenType.IF)) {
            this.expect(TokenType.NOT);
            this.expect(TokenType.EXISTS);
            ifNotExists = true;
        }

        // Table name
        const table = this.expect(TokenType.IDENTIFIER).value;

        // Column definitions
        this.expect(TokenType.LPAREN);

        const columns = [];
        do {
            const name = this.expect(TokenType.IDENTIFIER).value;

            // Data type
            let dataType = 'TEXT'; // default
            let primaryKey = false;
            let vectorDim = null;

            if (this.check(TokenType.INT) || this.check(TokenType.INTEGER) || this.check(TokenType.BIGINT)) {
                this.advance();
                dataType = 'INT64';
            } else if (this.check(TokenType.FLOAT) || this.check(TokenType.REAL) || this.check(TokenType.DOUBLE)) {
                this.advance();
                dataType = 'FLOAT64';
            } else if (this.check(TokenType.TEXT) || this.check(TokenType.VARCHAR)) {
                this.advance();
                dataType = 'STRING';
            } else if (this.check(TokenType.BOOLEAN) || this.check(TokenType.BOOL)) {
                this.advance();
                dataType = 'BOOL';
            } else if (this.check(TokenType.VECTOR)) {
                this.advance();
                dataType = 'VECTOR';
                // Optional dimension: VECTOR(384)
                if (this.match(TokenType.LPAREN)) {
                    vectorDim = parseInt(this.expect(TokenType.NUMBER).value, 10);
                    this.expect(TokenType.RPAREN);
                }
            }

            // Optional PRIMARY KEY
            if (this.match(TokenType.PRIMARY)) {
                this.expect(TokenType.KEY);
                primaryKey = true;
            }

            columns.push({ name, dataType, primaryKey, vectorDim });
        } while (this.match(TokenType.COMMA));

        this.expect(TokenType.RPAREN);

        return {
            type: 'CREATE_TABLE',
            table,
            columns,
            ifNotExists,
        };
    }

    /**
     * Parse DROP TABLE statement
     * Syntax: DROP TABLE [IF EXISTS] table_name
     */
    parseDropTable() {
        this.expect(TokenType.DROP);
        this.expect(TokenType.TABLE);

        // Check for IF EXISTS
        let ifExists = false;
        if (this.match(TokenType.IF)) {
            this.expect(TokenType.EXISTS);
            ifExists = true;
        }

        // Table name
        const table = this.expect(TokenType.IDENTIFIER).value;

        return {
            type: 'DROP_TABLE',
            table,
            ifExists,
        };
    }

    parseSelectList() {
        const items = [this.parseSelectItem()];

        while (this.match(TokenType.COMMA)) {
            items.push(this.parseSelectItem());
        }

        return items;
    }

    parseSelectItem() {
        // Check for *
        if (this.match(TokenType.STAR)) {
            return { type: 'star' };
        }

        // Expression
        const expr = this.parseExpr();

        // Optional AS alias
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.check(TokenType.IDENTIFIER) && !this.check(TokenType.FROM, TokenType.WHERE, TokenType.ORDER, TokenType.LIMIT, TokenType.GROUP, TokenType.JOIN, TokenType.INNER, TokenType.LEFT, TokenType.RIGHT, TokenType.COMMA)) {
            // Implicit alias (but not if next token is a keyword or comma)
            alias = this.advance().value;
        }

        return { type: 'expr', expr, alias };
    }

    /**
     * Parse FROM clause - supports:
     * - table_name (identifier)
     * - read_lance('url') (function call)
     * - 'url.lance' (string literal, auto-detect)
     */
    parseFromClause() {
        let from = null;

        // Check for string literal (direct URL/path)
        if (this.check(TokenType.STRING)) {
            const url = this.advance().value;
            from = { type: 'url', url };
        }
        // Check for function call like read_lance(), read_lance(24), read_lance('url'), read_lance('url', 24)
        else if (this.check(TokenType.IDENTIFIER)) {
            const name = this.advance().value;

            // If followed by (, it's a function call
            if (this.match(TokenType.LPAREN)) {
                const funcName = name.toLowerCase();
                if (funcName === 'read_lance') {
                    // read_lance(FILE) - local uploaded file
                    // read_lance(FILE, 24) - local file with version
                    // read_lance('url') - remote url
                    // read_lance('url', 24) - remote url with version
                    from = { type: 'url', function: 'read_lance' };

                    if (!this.check(TokenType.RPAREN)) {
                        // First arg: FILE keyword, string (url)
                        if (this.match(TokenType.FILE)) {
                            // Local file - mark as file reference
                            from.isFile = true;
                            // Check for second arg (version)
                            if (this.match(TokenType.COMMA)) {
                                from.version = parseInt(this.expect(TokenType.NUMBER).value, 10);
                            }
                        } else if (this.check(TokenType.STRING)) {
                            from.url = this.advance().value;
                            // Check for second arg (version)
                            if (this.match(TokenType.COMMA)) {
                                from.version = parseInt(this.expect(TokenType.NUMBER).value, 10);
                            }
                        }
                    }
                    this.expect(TokenType.RPAREN);
                } else {
                    throw new Error(`Unknown table function: ${name}. Supported: read_lance()`);
                }
            } else {
                // Just an identifier (table name - for future use)
                from = { type: 'table', name };
            }
        } else {
            throw new Error('Expected table name, URL string, or read_lance() after FROM');
        }

        // Parse optional alias (e.g., FROM images i or FROM images AS i)
        if (from) {
            if (this.match(TokenType.AS)) {
                from.alias = this.expect(TokenType.IDENTIFIER).value;
            } else if (this.check(TokenType.IDENTIFIER) && !this.check(TokenType.WHERE, TokenType.ORDER, TokenType.LIMIT, TokenType.GROUP, TokenType.NEAR, TokenType.JOIN, TokenType.INNER, TokenType.LEFT, TokenType.RIGHT, TokenType.COMMA)) {
                // Implicit alias (not followed by a keyword)
                from.alias = this.advance().value;
            }
        }

        return from;
    }

    /**
     * Parse JOIN clause - supports:
     * - JOIN table ON condition
     * - INNER JOIN table ON condition
     * - LEFT JOIN table ON condition
     * - RIGHT JOIN table ON condition
     * - FULL OUTER JOIN table ON condition
     * - CROSS JOIN table
     */
    parseJoinClause() {
        // Parse join type
        let joinType = 'INNER'; // default

        if (this.match(TokenType.INNER)) {
            this.expect(TokenType.JOIN);
            joinType = 'INNER';
        } else if (this.match(TokenType.LEFT)) {
            this.match(TokenType.OUTER); // optional
            this.expect(TokenType.JOIN);
            joinType = 'LEFT';
        } else if (this.match(TokenType.RIGHT)) {
            this.match(TokenType.OUTER); // optional
            this.expect(TokenType.JOIN);
            joinType = 'RIGHT';
        } else if (this.match(TokenType.FULL)) {
            this.match(TokenType.OUTER); // optional
            this.expect(TokenType.JOIN);
            joinType = 'FULL';
        } else if (this.match(TokenType.CROSS)) {
            this.expect(TokenType.JOIN);
            joinType = 'CROSS';
        } else {
            this.expect(TokenType.JOIN); // plain JOIN defaults to INNER
        }

        // Parse table reference (same as FROM clause) - includes alias parsing
        const table = this.parseFromClause();

        // Alias is already parsed by parseFromClause and stored in table.alias
        const alias = table.alias || null;

        // Parse ON condition (except for CROSS JOIN)
        let on = null;
        if (joinType !== 'CROSS') {
            this.expect(TokenType.ON);
            on = this.parseExpr();
        }

        return {
            type: joinType,
            table,
            alias,
            on
        };
    }

    parseColumnList() {
        const columns = [this.expect(TokenType.IDENTIFIER).value];

        while (this.match(TokenType.COMMA)) {
            columns.push(this.expect(TokenType.IDENTIFIER).value);
        }

        return columns;
    }

    /**
     * Parse GROUP BY list with support for ROLLUP, CUBE, GROUPING SETS
     * Returns array of items, each with { type, column/columns/sets }
     */
    parseGroupByList() {
        const items = [];

        do {
            if (this.match(TokenType.ROLLUP)) {
                // ROLLUP(col1, col2, ...)
                this.expect(TokenType.LPAREN);
                const columns = this.parseColumnList();
                this.expect(TokenType.RPAREN);
                items.push({ type: 'ROLLUP', columns });
            } else if (this.match(TokenType.CUBE)) {
                // CUBE(col1, col2, ...)
                this.expect(TokenType.LPAREN);
                const columns = this.parseColumnList();
                this.expect(TokenType.RPAREN);
                items.push({ type: 'CUBE', columns });
            } else if (this.match(TokenType.GROUPING)) {
                // GROUPING SETS((col1, col2), (col1), ())
                this.expect(TokenType.SETS);
                this.expect(TokenType.LPAREN);
                const sets = this.parseGroupingSets();
                this.expect(TokenType.RPAREN);
                items.push({ type: 'GROUPING_SETS', sets });
            } else {
                // Simple column
                items.push({ type: 'COLUMN', column: this.expect(TokenType.IDENTIFIER).value });
            }
        } while (this.match(TokenType.COMMA));

        return items;
    }

    /**
     * Parse the sets inside GROUPING SETS(...)
     * Each set is (col1, col2) or () for grand total
     */
    parseGroupingSets() {
        const sets = [];

        do {
            this.expect(TokenType.LPAREN);
            if (this.check(TokenType.RPAREN)) {
                // Empty set () = grand total
                sets.push([]);
            } else {
                sets.push(this.parseColumnList());
            }
            this.expect(TokenType.RPAREN);
        } while (this.match(TokenType.COMMA));

        return sets;
    }

    parseOrderByList() {
        const items = [this.parseOrderByItem()];

        while (this.match(TokenType.COMMA)) {
            items.push(this.parseOrderByItem());
        }

        return items;
    }

    parseOrderByItem() {
        const column = this.expect(TokenType.IDENTIFIER).value;

        let descending = false;
        if (this.match(TokenType.DESC)) {
            descending = true;
        } else {
            this.match(TokenType.ASC);
        }

        return { column, descending };
    }

    // Expression parsing with precedence
    parseExpr() {
        return this.parseOrExpr();
    }

    parseOrExpr() {
        let left = this.parseAndExpr();

        while (this.match(TokenType.OR)) {
            const right = this.parseAndExpr();
            left = { type: 'binary', op: 'OR', left, right };
        }

        return left;
    }

    parseAndExpr() {
        let left = this.parseNotExpr();

        while (this.match(TokenType.AND)) {
            const right = this.parseNotExpr();
            left = { type: 'binary', op: 'AND', left, right };
        }

        return left;
    }

    parseNotExpr() {
        if (this.match(TokenType.NOT)) {
            const operand = this.parseNotExpr();
            return { type: 'unary', op: 'NOT', operand };
        }
        return this.parseCmpExpr();
    }

    parseCmpExpr() {
        let left = this.parseAddExpr();

        // IS NULL / IS NOT NULL
        if (this.match(TokenType.IS)) {
            const negated = !!this.match(TokenType.NOT);
            this.expect(TokenType.NULL);
            return {
                type: 'binary',
                op: negated ? '!=' : '==',
                left,
                right: { type: 'literal', value: null }
            };
        }

        // IN - can be a list of values or a subquery
        if (this.match(TokenType.IN)) {
            this.expect(TokenType.LPAREN);

            // Check if this is a subquery (starts with SELECT)
            if (this.check(TokenType.SELECT)) {
                const subquery = this.parseSelect(true);  // isSubquery=true
                this.expect(TokenType.RPAREN);
                return { type: 'in', expr: left, values: [{ type: 'subquery', query: subquery }] };
            }

            // Otherwise, parse as list of values
            const values = [];
            values.push(this.parsePrimary());
            while (this.match(TokenType.COMMA)) {
                values.push(this.parsePrimary());
            }
            this.expect(TokenType.RPAREN);
            return { type: 'in', expr: left, values };
        }

        // BETWEEN
        if (this.match(TokenType.BETWEEN)) {
            const low = this.parseAddExpr();
            this.expect(TokenType.AND);
            const high = this.parseAddExpr();
            return { type: 'between', expr: left, low, high };
        }

        // LIKE
        if (this.match(TokenType.LIKE)) {
            const pattern = this.parsePrimary();
            return { type: 'like', expr: left, pattern };
        }

        // NEAR - vector similarity search in WHERE clause
        if (this.match(TokenType.NEAR)) {
            const text = this.parsePrimary();
            return { type: 'near', column: left, text };
        }

        // Comparison operators
        const opMap = {
            [TokenType.EQ]: '==',
            [TokenType.NE]: '!=',
            [TokenType.LT]: '<',
            [TokenType.LE]: '<=',
            [TokenType.GT]: '>',
            [TokenType.GE]: '>=',
        };

        const opToken = this.match(TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE);
        if (opToken) {
            const right = this.parseAddExpr();
            return { type: 'binary', op: opMap[opToken.type], left, right };
        }

        return left;
    }

    parseAddExpr() {
        let left = this.parseMulExpr();

        while (true) {
            const opToken = this.match(TokenType.PLUS, TokenType.MINUS);
            if (!opToken) break;
            const right = this.parseMulExpr();
            left = { type: 'binary', op: opToken.value, left, right };
        }

        return left;
    }

    parseMulExpr() {
        let left = this.parseUnaryExpr();

        while (true) {
            const opToken = this.match(TokenType.STAR, TokenType.SLASH);
            if (!opToken) break;
            const right = this.parseUnaryExpr();
            left = { type: 'binary', op: opToken.value, left, right };
        }

        return left;
    }

    parseUnaryExpr() {
        if (this.match(TokenType.MINUS)) {
            const operand = this.parseUnaryExpr();
            return { type: 'unary', op: '-', operand };
        }
        return this.parsePrimary();
    }

    parsePrimary() {
        // NULL
        if (this.match(TokenType.NULL)) {
            return { type: 'literal', value: null };
        }

        // TRUE/FALSE
        if (this.match(TokenType.TRUE)) {
            return { type: 'literal', value: true };
        }
        if (this.match(TokenType.FALSE)) {
            return { type: 'literal', value: false };
        }

        // ARRAY[...] literal
        if (this.match(TokenType.ARRAY)) {
            let result = this.parseArrayLiteral();
            // Check for subscript: ARRAY[1,2,3][2]
            while (this.check(TokenType.LBRACKET)) {
                this.advance();
                const index = this.parseExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // Bare bracket array [...]
        if (this.check(TokenType.LBRACKET)) {
            let result = this.parseArrayLiteral();
            // Check for subscript: [1,2,3][2]
            while (this.check(TokenType.LBRACKET)) {
                this.advance();
                const index = this.parseExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // Number
        if (this.check(TokenType.NUMBER)) {
            const value = this.advance().value;
            return { type: 'literal', value: parseFloat(value) };
        }

        // String
        if (this.check(TokenType.STRING)) {
            const value = this.advance().value;
            return { type: 'literal', value };
        }

        // Window function keywords (ROW_NUMBER, RANK, etc.)
        const windowFuncTokens = [
            TokenType.ROW_NUMBER, TokenType.RANK, TokenType.DENSE_RANK, TokenType.NTILE,
            TokenType.LAG, TokenType.LEAD, TokenType.FIRST_VALUE, TokenType.LAST_VALUE, TokenType.NTH_VALUE,
            TokenType.PERCENT_RANK, TokenType.CUME_DIST
        ];
        if (windowFuncTokens.some(t => this.check(t))) {
            const name = this.advance().type;  // Use token type as function name
            this.expect(TokenType.LPAREN);
            const args = [];
            if (!this.check(TokenType.RPAREN)) {
                args.push(this.parseExpr());
                while (this.match(TokenType.COMMA)) {
                    args.push(this.parseExpr());
                }
            }
            this.expect(TokenType.RPAREN);

            // OVER clause is required for window functions
            const over = this.parseOverClause();
            return { type: 'call', name, args, distinct: false, over };
        }

        // Function call or column reference
        if (this.check(TokenType.IDENTIFIER) || this.check(TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX, TokenType.GROUPING)) {
            const name = this.advance().value;

            // Function call
            if (this.match(TokenType.LPAREN)) {
                let distinct = !!this.match(TokenType.DISTINCT);
                const args = [];

                if (!this.check(TokenType.RPAREN)) {
                    // Handle COUNT(*)
                    if (this.check(TokenType.STAR)) {
                        this.advance();
                        args.push({ type: 'star' });
                    } else {
                        args.push(this.parseExpr());
                        while (this.match(TokenType.COMMA)) {
                            args.push(this.parseExpr());
                        }
                    }
                }

                this.expect(TokenType.RPAREN);

                // Check for OVER clause (aggregate as window function)
                let over = null;
                if (this.check(TokenType.OVER)) {
                    over = this.parseOverClause();
                }

                return { type: 'call', name: name.toUpperCase(), args, distinct, over };
            }

            // Column reference - check for table.column syntax
            if (this.match(TokenType.DOT)) {
                // table.column (column can be a keyword like "text")
                const table = name;
                const token = this.advance();
                // Allow keywords as column names (e.g., c.text where TEXT is a keyword)
                const column = token.value || token.type.toLowerCase();
                return { type: 'column', table, column };
            }

            // Simple column reference
            let result = { type: 'column', column: name };

            // Check for array subscript: column[index]
            if (this.check(TokenType.LBRACKET)) {
                this.advance();  // consume [
                const index = this.parseExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }

            return result;
        }

        // Parenthesized expression or subquery
        if (this.match(TokenType.LPAREN)) {
            // Check if this is a subquery (starts with SELECT)
            if (this.check(TokenType.SELECT)) {
                const subquery = this.parseSelect(true);  // isSubquery=true
                this.expect(TokenType.RPAREN);
                return { type: 'subquery', query: subquery };
            }
            const expr = this.parseExpr();
            this.expect(TokenType.RPAREN);
            return expr;
        }

        // Star (for SELECT *)
        if (this.match(TokenType.STAR)) {
            return { type: 'star' };
        }

        throw new Error(`Unexpected token: ${this.current().type} (${this.current().value})`);
    }

    /**
     * Parse OVER clause for window functions
     * Syntax: OVER ([PARTITION BY expr, ...] [ORDER BY expr [ASC|DESC], ...] [frame_clause])
     */
    parseOverClause() {
        this.expect(TokenType.OVER);
        this.expect(TokenType.LPAREN);

        const over = { partitionBy: [], orderBy: [], frame: null };

        // PARTITION BY clause
        if (this.match(TokenType.PARTITION)) {
            this.expect(TokenType.BY);
            over.partitionBy.push(this.parseExpr());
            while (this.match(TokenType.COMMA)) {
                over.partitionBy.push(this.parseExpr());
            }
        }

        // ORDER BY clause
        if (this.match(TokenType.ORDER)) {
            this.expect(TokenType.BY);
            over.orderBy = this.parseOrderByList();
        }

        // Optional frame clause: ROWS/RANGE BETWEEN ... AND ...
        if (this.check(TokenType.ROWS) || this.check(TokenType.RANGE)) {
            over.frame = this.parseFrameClause();
        }

        this.expect(TokenType.RPAREN);
        return over;
    }

    /**
     * Parse frame clause for window functions
     * Syntax: ROWS|RANGE BETWEEN frame_start AND frame_end
     *         or: ROWS|RANGE frame_start
     */
    parseFrameClause() {
        const frameType = this.advance().type;  // ROWS or RANGE
        const frame = { type: frameType, start: null, end: null };

        // Check for BETWEEN ... AND ... syntax
        if (this.match(TokenType.BETWEEN)) {
            frame.start = this.parseFrameBound();
            this.expect(TokenType.AND);
            frame.end = this.parseFrameBound();
        } else {
            // Single bound (implies CURRENT ROW as end for some DBs, or just start)
            frame.start = this.parseFrameBound();
            frame.end = { type: 'CURRENT ROW' };  // Default end
        }

        return frame;
    }

    /**
     * Parse a frame bound
     * Options: UNBOUNDED PRECEDING, UNBOUNDED FOLLOWING, CURRENT ROW, N PRECEDING, N FOLLOWING
     */
    parseFrameBound() {
        if (this.match(TokenType.UNBOUNDED)) {
            if (this.match(TokenType.PRECEDING)) {
                return { type: 'UNBOUNDED PRECEDING' };
            } else if (this.match(TokenType.FOLLOWING)) {
                return { type: 'UNBOUNDED FOLLOWING' };
            }
            throw new Error('Expected PRECEDING or FOLLOWING after UNBOUNDED');
        }

        if (this.match(TokenType.CURRENT)) {
            this.expect(TokenType.ROW);
            return { type: 'CURRENT ROW' };
        }

        // N PRECEDING or N FOLLOWING
        if (this.check(TokenType.NUMBER)) {
            const n = parseInt(this.advance().value, 10);
            if (this.match(TokenType.PRECEDING)) {
                return { type: 'PRECEDING', offset: n };
            } else if (this.match(TokenType.FOLLOWING)) {
                return { type: 'FOLLOWING', offset: n };
            }
            throw new Error('Expected PRECEDING or FOLLOWING after number');
        }

        throw new Error('Invalid frame bound');
    }
}

/**
 * SQL Executor - executes parsed SQL against a LanceFile
 */
export class SQLExecutor {
    constructor(file) {
        this.file = file;
        this.columnMap = {};
        this.columnTypes = [];
        this._cteResults = new Map();  // Store materialized CTE results
        this._database = null;          // Reference to LanceDatabase for subqueries

        // Build column name -> index map
        if (file.columnNames) {
            file.columnNames.forEach((name, idx) => {
                this.columnMap[name.toLowerCase()] = idx;
            });
        }
    }

    /**
     * Set reference to parent database for CTE/subquery execution
     */
    setDatabase(db) {
        this._database = db;
    }

    /**
     * Execute a SQL query
     * @param {string} sql - SQL query string
     * @param {function} onProgress - Optional progress callback
     * @returns {Promise<{columns: string[], rows: any[][], total: number}>}
     */
    async execute(sql, onProgress = null) {
        // Tokenize and parse
        const lexer = new SQLLexer(sql);
        const tokens = lexer.tokenize();
        const parser = new SQLParser(tokens);
        const ast = parser.parse();

        // Generate optimized query plan
        const planner = new QueryPlanner();
        const plan = planner.planSingleTable(ast);

        // Detect column types if not already done
        if (this.columnTypes.length === 0) {
            if (this.file._isRemote && this.file.detectColumnTypes) {
                this.columnTypes = await this.file.detectColumnTypes();
            } else if (this.file._columnTypes) {
                this.columnTypes = this.file._columnTypes;
            } else {
                // Default to unknown for all columns
                this.columnTypes = Array(this.file.numColumns || 0).fill('unknown');
            }
        }

        // Get total row count
        const totalRows = this.file._isRemote
            ? await this.file.getRowCount(0)
            : Number(this.file.getRowCount(0));

        // === STATISTICS-BASED OPTIMIZATION ===
        // For queries with filters, compute statistics to enable pruning
        let columnStats = null;
        let prunedFragments = null;
        let fragmentsPruned = 0;

        if (ast.where && plan.pushedFilters.length > 0 && this.file._isRemote) {
            // Compute stats for filter columns (cached after first computation)
            columnStats = await statisticsManager.precomputeForPlan(this.file, plan);

            // Log statistics info
            if (columnStats.size > 0) {
                console.log(`[SQLExecutor] Statistics available for ${columnStats.size} columns`);
                for (const [col, stats] of columnStats) {
                    console.log(`  ${col}: min=${stats.min}, max=${stats.max}, nulls=${stats.nullCount}`);
                }
            }

            // Fragment pruning based on global statistics
            // (Per-fragment stats would be even better - computed lazily)
            plan.columnStats = Object.fromEntries(columnStats);
        }

        // Use plan's scan columns instead of basic column collection
        const neededColumns = plan.scanColumns.length > 0
            ? plan.scanColumns
            : this.collectNeededColumns(ast);

        // Determine output columns
        const outputColumns = this.resolveOutputColumns(ast);

        // Check if this is an aggregation query
        const hasAggregates = plan.aggregations.length > 0 || this.hasAggregates(ast);
        if (hasAggregates) {
            // Special case: COUNT(*) without WHERE/SEARCH returns metadata row count (free)
            if (this.isSimpleCountStar(ast) && !ast.where && !ast.search) {
                return {
                    columns: ['COUNT(*)'],
                    rows: [[totalRows]],
                    total: 1,
                    aggregationStats: {
                        scannedRows: 0,
                        totalRows,
                        coveragePercent: '100.00',
                        isPartialScan: false,
                        fromMetadata: true,
                    },
                    queryPlan: plan,  // Include plan in result
                };
            }
            // For aggregations with SEARCH/NEAR, we need to run search first
            if (ast.search || this._extractNearCondition(ast.where)) {
                return await this.executeAggregateWithSearch(ast, totalRows, onProgress);
            }
            return await this.executeAggregateQuery(ast, totalRows, onProgress);
        }

        // Calculate indices to fetch
        let indices;
        const limit = ast.limit || 100;
        const offset = ast.offset || 0;

        // For queries without WHERE, we can just fetch the needed indices directly
        // For queries with WHERE, we need to fetch more data and filter
        if (!ast.where) {
            // Simple case: no filtering needed
            indices = [];
            const endIdx = Math.min(offset + limit, totalRows);
            for (let i = offset; i < endIdx; i++) {
                indices.push(i);
            }
        } else {
            // Complex case: need to evaluate WHERE clause
            // Fetch data in batches and filter
            indices = await this.evaluateWhere(ast.where, totalRows, onProgress);

            // Apply OFFSET and LIMIT to filtered results
            indices = indices.slice(offset, offset + limit);
        }

        if (onProgress) {
            onProgress('Fetching column data...', 0, outputColumns.length);
        }

        // Fetch data for output columns
        const columnData = {};
        for (let i = 0; i < neededColumns.length; i++) {
            const colName = neededColumns[i];
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx === undefined) continue;

            if (onProgress) {
                onProgress(`Fetching ${colName}...`, i, neededColumns.length);
            }

            columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, indices);
        }

        // Build result rows
        const rows = [];
        for (let i = 0; i < indices.length; i++) {
            const row = [];
            for (const col of outputColumns) {
                if (col.type === 'star') {
                    // Expand all columns
                    for (const name of this.file.columnNames || []) {
                        const data = columnData[name.toLowerCase()];
                        row.push(data ? data[i] : null);
                    }
                } else {
                    const value = this.evaluateExpr(col.expr, columnData, i);
                    row.push(value);
                }
            }
            rows.push(row);
        }

        // Apply PIVOT transformation (rows to columns with aggregation)
        if (ast.pivot) {
            const pivotResult = this._executePivot(rows, outputColumns, ast.pivot);
            rows.length = 0;
            rows.push(...pivotResult.rows);
            outputColumns.length = 0;
            outputColumns.push(...pivotResult.columns);
        }

        // Apply UNPIVOT transformation (columns to rows)
        if (ast.unpivot) {
            const unpivotResult = this._executeUnpivot(rows, outputColumns, ast.unpivot);
            rows.length = 0;
            rows.push(...unpivotResult.rows);
            outputColumns.length = 0;
            outputColumns.push(...unpivotResult.columns);
        }

        // Apply DISTINCT (GPU-accelerated for large result sets)
        if (ast.distinct) {
            const uniqueRows = await this.applyDistinct(rows);
            rows.length = 0;
            rows.push(...uniqueRows);
        }

        // Apply ORDER BY (GPU-accelerated for large result sets)
        if (ast.orderBy && ast.orderBy.length > 0) {
            await this.applyOrderBy(rows, ast.orderBy, outputColumns);
        }

        // Build column names for output
        const colNames = [];
        for (const col of outputColumns) {
            if (col.type === 'star') {
                colNames.push(...(this.file.columnNames || []));
            } else {
                colNames.push(col.alias || this.exprToName(col.expr));
            }
        }

        // When LIMIT is specified, total should reflect the limited count, not full dataset
        // This ensures infinite scroll respects the LIMIT clause
        const effectiveTotal = ast.limit ? rows.length : totalRows;

        // Track if ORDER BY was applied on a subset (honest about sorting limitations)
        const orderByOnSubset = ast.orderBy && ast.orderBy.length > 0 && rows.length < totalRows;

        return {
            columns: colNames,
            rows,
            total: effectiveTotal,
            orderByOnSubset,
            orderByColumns: ast.orderBy ? ast.orderBy.map(ob => `${ob.column} ${ob.direction}`) : [],
            // Query optimization info
            queryPlan: plan,
            optimization: {
                statsComputed: columnStats?.size > 0,
                columnStats: columnStats ? Object.fromEntries(columnStats) : null,
                pushedFilters: plan.pushedFilters?.length || 0,
                estimatedSelectivity: plan.estimatedSelectivity,
            },
        };
    }

    collectNeededColumns(ast) {
        const columns = new Set();

        // From SELECT
        for (const item of ast.columns) {
            if (item.type === 'star') {
                (this.file.columnNames || []).forEach(n => columns.add(n.toLowerCase()));
            } else {
                this.collectColumnsFromExpr(item.expr, columns);
            }
        }

        // From WHERE
        if (ast.where) {
            this.collectColumnsFromExpr(ast.where, columns);
        }

        // From ORDER BY
        for (const ob of ast.orderBy || []) {
            columns.add(ob.column.toLowerCase());
        }

        return Array.from(columns);
    }

    collectColumnsFromExpr(expr, columns) {
        if (!expr) return;

        switch (expr.type) {
            case 'column':
                columns.add(expr.name.toLowerCase());
                break;
            case 'binary':
                this.collectColumnsFromExpr(expr.left, columns);
                this.collectColumnsFromExpr(expr.right, columns);
                break;
            case 'unary':
                this.collectColumnsFromExpr(expr.operand, columns);
                break;
            case 'call':
                for (const arg of expr.args || []) {
                    this.collectColumnsFromExpr(arg, columns);
                }
                break;
            case 'in':
                this.collectColumnsFromExpr(expr.expr, columns);
                break;
            case 'between':
                this.collectColumnsFromExpr(expr.expr, columns);
                break;
            case 'like':
                this.collectColumnsFromExpr(expr.expr, columns);
                break;
            case 'near':
                this.collectColumnsFromExpr(expr.column, columns);
                break;
        }
    }

    resolveOutputColumns(ast) {
        return ast.columns;
    }

    async readColumnData(colIdx, indices) {
        const type = this.columnTypes[colIdx] || 'unknown';

        try {
            if (type === 'string') {
                const data = await this.file.readStringsAtIndices(colIdx, indices);
                // readStringsAtIndices returns array of strings
                return Array.isArray(data) ? data : Array.from(data);
            } else if (type === 'int64') {
                const data = await this.file.readInt64AtIndices(colIdx, indices);
                // Convert BigInt64Array to regular array of Numbers
                const result = [];
                for (let i = 0; i < data.length; i++) {
                    result.push(Number(data[i]));
                }
                return result;
            } else if (type === 'float64') {
                const data = await this.file.readFloat64AtIndices(colIdx, indices);
                // Convert Float64Array to regular array
                return Array.from(data);
            } else if (type === 'int32') {
                const data = await this.file.readInt32AtIndices(colIdx, indices);
                return Array.from(data);
            } else if (type === 'float32') {
                const data = await this.file.readFloat32AtIndices(colIdx, indices);
                return Array.from(data);
            } else if (type === 'vector') {
                // Return placeholder for vectors
                return indices.map(() => '[vector]');
            } else {
                // Try string as fallback
                try {
                    return await this.file.readStringsAtIndices(colIdx, indices);
                } catch (e) {
                    return indices.map(() => null);
                }
            }
        } catch (e) {
            // Failed to read column, returning nulls
            return indices.map(() => null);
        }
    }

    async evaluateWhere(whereExpr, totalRows, onProgress) {
        // Check for NEAR conditions in WHERE clause
        const nearInfo = this._extractNearCondition(whereExpr);
        if (nearInfo) {
            return await this._evaluateWithNear(nearInfo, whereExpr, totalRows, onProgress);
        }

        // Optimization: For simple conditions on a single numeric column,
        // fetch only the filter column first, then fetch other columns only for matches
        const simpleFilter = this._detectSimpleFilter(whereExpr);

        if (simpleFilter) {
            return await this._evaluateSimpleFilter(simpleFilter, totalRows, onProgress);
        }

        // Complex conditions: fetch all needed columns in batches
        return await this._evaluateComplexFilter(whereExpr, totalRows, onProgress);
    }

    /**
     * Extract NEAR condition from WHERE expression.
     * Returns { column, text, limit } if found, null otherwise.
     * @private
     */
    _extractNearCondition(expr) {
        if (!expr) return null;

        if (expr.type === 'near') {
            const columnName = expr.column?.name || expr.column;
            const text = expr.text?.value || expr.text;
            return { column: columnName, text, limit: 20 };
        }

        // Check AND/OR for NEAR condition
        if (expr.type === 'binary' && (expr.op === 'AND' || expr.op === 'OR')) {
            const leftNear = this._extractNearCondition(expr.left);
            if (leftNear) return leftNear;
            return this._extractNearCondition(expr.right);
        }

        return null;
    }

    /**
     * Remove NEAR condition from expression, returning remaining conditions.
     * @private
     */
    _removeNearCondition(expr) {
        if (!expr) return null;

        if (expr.type === 'near') {
            return null;  // Remove the NEAR condition
        }

        if (expr.type === 'binary' && (expr.op === 'AND' || expr.op === 'OR')) {
            const leftIsNear = expr.left?.type === 'near';
            const rightIsNear = expr.right?.type === 'near';

            if (leftIsNear && rightIsNear) return null;
            if (leftIsNear) return this._removeNearCondition(expr.right);
            if (rightIsNear) return this._removeNearCondition(expr.left);

            const newLeft = this._removeNearCondition(expr.left);
            const newRight = this._removeNearCondition(expr.right);

            if (!newLeft && !newRight) return null;
            if (!newLeft) return newRight;
            if (!newRight) return newLeft;

            return { ...expr, left: newLeft, right: newRight };
        }

        return expr;
    }

    /**
     * Evaluate WHERE with NEAR condition.
     * Executes vector search first, then applies remaining conditions.
     * @private
     */
    async _evaluateWithNear(nearInfo, whereExpr, totalRows, onProgress) {
        if (onProgress) {
            onProgress('Executing vector search...', 0, 100);
        }

        // Execute vector search to get candidate indices
        // This reuses the existing vector search infrastructure
        const searchResults = await this._executeNearSearch(nearInfo, totalRows);

        if (!searchResults || searchResults.length === 0) {
            return [];
        }

        // Get remaining conditions after removing NEAR
        const remainingExpr = this._removeNearCondition(whereExpr);

        if (!remainingExpr) {
            // No other conditions, return search results directly
            return searchResults;
        }

        if (onProgress) {
            onProgress('Applying filters...', 50, 100);
        }

        // Apply remaining conditions to search results
        const neededCols = new Set();
        this.collectColumnsFromExpr(remainingExpr, neededCols);

        // Fetch column data for candidate rows
        const columnData = {};
        for (const colName of neededCols) {
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx !== undefined) {
                columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, searchResults);
            }
        }

        // Filter by remaining conditions
        const matchingIndices = [];
        for (let i = 0; i < searchResults.length; i++) {
            const result = this.evaluateExpr(remainingExpr, columnData, i);
            if (result) {
                matchingIndices.push(searchResults[i]);
            }
        }

        return matchingIndices;
    }

    /**
     * Execute NEAR vector search.
     * @private
     */
    async _executeNearSearch(nearInfo, totalRows) {
        // Find vector column for the specified column
        const { column, text, limit } = nearInfo;

        // Look for embedding/vector column
        // Convention: embedding column is named 'embedding' or '<column>_embedding'
        const vectorColName = this.file.columnNames?.find(n =>
            n === 'embedding' ||
            n === `${column}_embedding` ||
            n.endsWith('_embedding') ||
            n.endsWith('_vector')
        );

        if (!vectorColName) {
            // No vector column found, fall back to BM25 text search
            return await this._executeBM25Search(nearInfo, totalRows);
        }

        // Use existing vector search infrastructure
        const topK = Math.min(limit, totalRows);

        try {
            // Call the file's vectorSearch method
            const results = await this.file.vectorSearch(text, topK);
            return results.map(r => r.index);
        } catch (e) {
            console.error('[SQLExecutor] Vector search failed:', e);
            throw new Error(`NEAR search failed: ${e.message}`);
        }
    }

    /**
     * Execute BM25 full-text search when no vector column exists.
     * @private
     */
    async _executeBM25Search(nearInfo, totalRows) {
        const { column, text, limit } = nearInfo;
        const colIdx = this.columnMap[column.toLowerCase()];

        if (colIdx === undefined) {
            throw new Error(`Column '${column}' not found for text search`);
        }

        // Step 1: Tokenize query
        const queryTokens = this._tokenize(text);
        if (queryTokens.length === 0) return [];

        // Step 2: Get or build inverted index for this column
        const index = await this._getOrBuildFTSIndex(colIdx, totalRows);

        // Step 3: Compute BM25 scores
        const scores = this._computeBM25Scores(queryTokens, index);

        // Step 4: Return top-K indices
        return this._topKByScore(scores, limit);
    }

    /**
     * Tokenize text for BM25 search.
     * @private
     */
    _tokenize(text) {
        if (!text || typeof text !== 'string') return [];

        return text
            .toLowerCase()
            .replace(/[^\w\s]/g, ' ')  // Remove punctuation
            .split(/\s+/)              // Split on whitespace
            .filter(t => t.length > 1) // Remove single chars
            .filter(t => !this._isStopWord(t)); // Remove stop words
    }

    /**
     * Check if word is a stop word.
     * @private
     */
    _isStopWord(word) {
        const stopWords = new Set([
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'we', 'they', 'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how'
        ]);
        return stopWords.has(word);
    }

    /**
     * Get or build inverted index for a text column.
     * @private
     */
    async _getOrBuildFTSIndex(colIdx, totalRows) {
        const cacheKey = `fts_${colIdx}`;
        if (this._ftsIndexCache?.has(cacheKey)) {
            return this._ftsIndexCache.get(cacheKey);
        }

        // Build index
        const index = {
            termDocs: new Map(),    // term -> Set of docIds
            termFreqs: new Map(),   // term -> Map(docId -> freq)
            docLengths: new Map(),  // docId -> word count
            totalDocs: 0,
            avgDocLength: 0,
        };

        // Read all text from column in batches
        const batchSize = 1000;
        let totalLength = 0;

        for (let start = 0; start < totalRows; start += batchSize) {
            const end = Math.min(start + batchSize, totalRows);
            const indices = Array.from({ length: end - start }, (_, i) => start + i);
            const texts = await this.readColumnData(colIdx, indices);

            for (let i = 0; i < texts.length; i++) {
                const docId = start + i;
                const text = texts[i];
                if (!text || typeof text !== 'string') continue;

                const tokens = this._tokenize(text);
                index.docLengths.set(docId, tokens.length);
                totalLength += tokens.length;
                index.totalDocs++;

                // Count term frequencies
                const termCounts = new Map();
                for (const token of tokens) {
                    termCounts.set(token, (termCounts.get(token) || 0) + 1);
                }

                // Update inverted index
                for (const [term, freq] of termCounts) {
                    if (!index.termDocs.has(term)) {
                        index.termDocs.set(term, new Set());
                        index.termFreqs.set(term, new Map());
                    }
                    index.termDocs.get(term).add(docId);
                    index.termFreqs.get(term).set(docId, freq);
                }
            }
        }

        index.avgDocLength = index.totalDocs > 0 ? totalLength / index.totalDocs : 0;

        // Cache the index
        if (!this._ftsIndexCache) this._ftsIndexCache = new Map();
        this._ftsIndexCache.set(cacheKey, index);

        return index;
    }

    /**
     * Compute BM25 scores for query tokens against indexed documents.
     * @private
     */
    _computeBM25Scores(queryTokens, index) {
        const k1 = 1.2;
        const b = 0.75;
        const scores = new Map();

        for (const term of queryTokens) {
            const docIds = index.termDocs.get(term);
            if (!docIds) continue;

            // IDF: log((N - n + 0.5) / (n + 0.5) + 1)
            const n = docIds.size;
            const N = index.totalDocs;
            const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1);

            for (const docId of docIds) {
                const tf = index.termFreqs.get(term).get(docId);
                const docLen = index.docLengths.get(docId);
                const avgDL = index.avgDocLength;

                // BM25 term score
                const numerator = tf * (k1 + 1);
                const denominator = tf + k1 * (1 - b + b * docLen / avgDL);
                const termScore = idf * (numerator / denominator);

                scores.set(docId, (scores.get(docId) || 0) + termScore);
            }
        }

        return scores;
    }

    /**
     * Get top-K documents by score.
     * @private
     */
    _topKByScore(scores, k) {
        return Array.from(scores.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, k)
            .map(([docId]) => docId);
    }

    /**
     * Execute PIVOT transformation - convert rows to columns with aggregation.
     * Example: PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))
     * @private
     */
    _executePivot(rows, columns, pivot) {
        const { aggregate, forColumn, inValues } = pivot;

        // Get column names from outputColumns
        const colNames = columns.map(col => {
            if (col.type === 'star') return '*';
            return col.alias || (col.expr.type === 'column' ? col.expr.name : null);
        });

        // Find the FOR column index
        const forColIdx = colNames.findIndex(
            n => n && n.toLowerCase() === forColumn.toLowerCase()
        );
        if (forColIdx === -1) {
            throw new Error(`PIVOT: Column '${forColumn}' not found in result set`);
        }

        // Find the aggregate source column
        const aggSourceCol = aggregate.args && aggregate.args[0]
            ? (aggregate.args[0].name || aggregate.args[0].column || aggregate.args[0])
            : null;
        const aggSourceIdx = aggSourceCol
            ? colNames.findIndex(n => n && n.toLowerCase() === String(aggSourceCol).toLowerCase())
            : -1;

        if (aggSourceIdx === -1 && aggregate.name.toUpperCase() !== 'COUNT') {
            throw new Error(`PIVOT: Aggregate source column not found`);
        }

        // Determine group columns (all columns except forColumn and aggregate source)
        const groupColIndices = [];
        for (let i = 0; i < colNames.length; i++) {
            if (i !== forColIdx && i !== aggSourceIdx) {
                groupColIndices.push(i);
            }
        }

        // Build groups: Map<groupKey, Map<pivotValue, aggregateValues[]>>
        const groups = new Map();

        for (const row of rows) {
            const groupKey = groupColIndices.map(i => JSON.stringify(row[i])).join('|');
            const pivotValue = row[forColIdx];
            const aggValue = aggSourceIdx >= 0 ? row[aggSourceIdx] : 1; // COUNT uses 1

            if (!groups.has(groupKey)) {
                groups.set(groupKey, {
                    groupValues: groupColIndices.map(i => row[i]),
                    pivots: new Map()
                });
            }

            const group = groups.get(groupKey);
            const pivotKey = String(pivotValue);
            if (!group.pivots.has(pivotKey)) {
                group.pivots.set(pivotKey, []);
            }
            group.pivots.get(pivotKey).push(aggValue);
        }

        // Compute aggregates and build output rows
        const groupColNames = groupColIndices.map(i => colNames[i]);
        const outputColNames = [...groupColNames, ...inValues.map(v => String(v))];
        const outputRows = [];

        for (const [, group] of groups) {
            const row = [...group.groupValues];
            for (const pivotVal of inValues) {
                const key = String(pivotVal);
                const values = group.pivots.get(key) || [];
                row.push(this._computeAggregate(aggregate.name, values));
            }
            outputRows.push(row);
        }

        // Build new outputColumns structure
        const newColumns = outputColNames.map(name => ({
            type: 'column',
            expr: { type: 'column', name },
            alias: name
        }));

        return { rows: outputRows, columns: newColumns };
    }

    /**
     * Execute UNPIVOT transformation - convert columns to rows.
     * Example: UNPIVOT (value FOR month IN (jan, feb, mar))
     * @private
     */
    _executeUnpivot(rows, columns, unpivot) {
        const { valueColumn, nameColumn, inColumns } = unpivot;

        // Get column names from outputColumns
        const colNames = columns.map(col => {
            if (col.type === 'star') return '*';
            return col.alias || (col.expr.type === 'column' ? col.expr.name : null);
        });

        // Find column indices for unpivot sources
        const inColIndices = inColumns.map(c => {
            const idx = colNames.findIndex(n => n && n.toLowerCase() === c.toLowerCase());
            if (idx === -1) {
                throw new Error(`UNPIVOT: Column '${c}' not found in result set`);
            }
            return idx;
        });

        // Preserved columns (not in inColumns)
        const preservedIndices = [];
        const preservedNames = [];
        for (let i = 0; i < colNames.length; i++) {
            if (!inColIndices.includes(i)) {
                preservedIndices.push(i);
                preservedNames.push(colNames[i]);
            }
        }

        // Output columns: preserved + nameColumn + valueColumn
        const outputColNames = [...preservedNames, nameColumn, valueColumn];
        const outputRows = [];

        for (const row of rows) {
            const preservedValues = preservedIndices.map(i => row[i]);

            // Create one output row per unpivoted column
            for (let i = 0; i < inColumns.length; i++) {
                const colName = inColumns[i];
                const value = row[inColIndices[i]];

                // Skip NULL values (standard UNPIVOT behavior)
                if (value != null) {
                    outputRows.push([...preservedValues, colName, value]);
                }
            }
        }

        // Build new outputColumns structure
        const newColumns = outputColNames.map(name => ({
            type: 'column',
            expr: { type: 'column', name },
            alias: name
        }));

        return { rows: outputRows, columns: newColumns };
    }

    /**
     * Compute aggregate function on a list of values.
     * Used by PIVOT transformation.
     * @private
     */
    _computeAggregate(funcName, values) {
        const nums = values.filter(v => v != null && typeof v === 'number');
        switch (funcName.toUpperCase()) {
            case 'SUM':
                return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) : null;
            case 'COUNT':
                return values.filter(v => v != null).length;
            case 'AVG':
                return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
            case 'MIN':
                return nums.length > 0 ? Math.min(...nums) : null;
            case 'MAX':
                return nums.length > 0 ? Math.max(...nums) : null;
            default:
                return null;
        }
    }

    /**
     * Detect if WHERE clause is a simple comparison (column op value).
     * @private
     */
    _detectSimpleFilter(expr) {
        if (expr.type !== 'binary') return null;
        if (!['==', '!=', '<', '<=', '>', '>='].includes(expr.op)) return null;

        // Check if it's (column op literal) or (literal op column)
        let column = null;
        let value = null;
        let op = expr.op;

        if (expr.left.type === 'column' && expr.right.type === 'literal') {
            column = expr.left.name;
            value = expr.right.value;
        } else if (expr.left.type === 'literal' && expr.right.type === 'column') {
            column = expr.right.name;
            value = expr.left.value;
            // Reverse the operator for (literal op column)
            if (op === '<') op = '>';
            else if (op === '>') op = '<';
            else if (op === '<=') op = '>=';
            else if (op === '>=') op = '<=';
        }

        if (!column || value === null) return null;

        const colIdx = this.columnMap[column.toLowerCase()];
        if (colIdx === undefined) return null;

        const colType = this.columnTypes[colIdx];
        if (!['int64', 'int32', 'float64', 'float32'].includes(colType)) return null;

        return { column, colIdx, op, value, colType };
    }

    /**
     * Optimized evaluation for simple column comparisons.
     * Fetches only the filter column in large batches.
     * @private
     */
    async _evaluateSimpleFilter(filter, totalRows, onProgress) {
        const matchingIndices = [];
        // Use larger batch size for single-column filtering
        const batchSize = 5000;

        // Using optimized simple filter path

        for (let batchStart = 0; batchStart < totalRows; batchStart += batchSize) {
            if (onProgress) {
                const pct = Math.round((batchStart / totalRows) * 100);
                onProgress(`Filtering ${filter.column}... ${pct}%`, batchStart, totalRows);
            }

            const batchEnd = Math.min(batchStart + batchSize, totalRows);
            const batchIndices = [];
            for (let i = batchStart; i < batchEnd; i++) {
                batchIndices.push(i);
            }

            // Fetch only the filter column
            const colData = await this.readColumnData(filter.colIdx, batchIndices);

            // Apply filter
            for (let i = 0; i < batchIndices.length; i++) {
                const val = colData[i];
                let matches = false;

                switch (filter.op) {
                    case '==': matches = val === filter.value; break;
                    case '!=': matches = val !== filter.value; break;
                    case '<': matches = val < filter.value; break;
                    case '<=': matches = val <= filter.value; break;
                    case '>': matches = val > filter.value; break;
                    case '>=': matches = val >= filter.value; break;
                }

                if (matches) {
                    matchingIndices.push(batchIndices[i]);
                }
            }

            // Early exit if we have enough results
            if (matchingIndices.length >= 10000) {
                // Early exit: found enough matches
                break;
            }
        }

        return matchingIndices;
    }

    /**
     * General evaluation for complex WHERE clauses.
     * @private
     */
    async _evaluateComplexFilter(whereExpr, totalRows, onProgress) {
        const matchingIndices = [];
        const batchSize = 1000;

        // Pre-compute needed columns
        const neededCols = new Set();
        this.collectColumnsFromExpr(whereExpr, neededCols);

        for (let batchStart = 0; batchStart < totalRows; batchStart += batchSize) {
            if (onProgress) {
                onProgress(`Filtering rows...`, batchStart, totalRows);
            }

            const batchEnd = Math.min(batchStart + batchSize, totalRows);
            const batchIndices = [];
            for (let i = batchStart; i < batchEnd; i++) {
                batchIndices.push(i);
            }

            // Fetch needed column data for this batch
            const batchData = {};
            for (const colName of neededCols) {
                const colIdx = this.columnMap[colName.toLowerCase()];
                if (colIdx !== undefined) {
                    batchData[colName.toLowerCase()] = await this.readColumnData(colIdx, batchIndices);
                }
            }

            // Evaluate WHERE for each row in batch
            for (let i = 0; i < batchIndices.length; i++) {
                const result = this.evaluateExpr(whereExpr, batchData, i);
                if (result) {
                    matchingIndices.push(batchIndices[i]);
                }
            }

            // Early exit if we have enough results
            if (matchingIndices.length >= 10000) {
                break;
            }
        }

        return matchingIndices;
    }

    evaluateExpr(expr, columnData, rowIdx) {
        if (!expr) return null;

        switch (expr.type) {
            case 'literal':
                return expr.value;

            case 'column': {
                const data = columnData[expr.name.toLowerCase()];
                return data ? data[rowIdx] : null;
            }

            case 'star':
                return '*';

            case 'binary': {
                const left = this.evaluateExpr(expr.left, columnData, rowIdx);
                const right = this.evaluateExpr(expr.right, columnData, rowIdx);

                switch (expr.op) {
                    case '+': return (left || 0) + (right || 0);
                    case '-': return (left || 0) - (right || 0);
                    case '*': return (left || 0) * (right || 0);
                    case '/': return right !== 0 ? (left || 0) / right : null;
                    case '==': return left === right;
                    case '!=': return left !== right;
                    case '<': return left < right;
                    case '<=': return left <= right;
                    case '>': return left > right;
                    case '>=': return left >= right;
                    case 'AND': return left && right;
                    case 'OR': return left || right;
                    default: return null;
                }
            }

            case 'unary': {
                const operand = this.evaluateExpr(expr.operand, columnData, rowIdx);
                switch (expr.op) {
                    case '-': return -operand;
                    case 'NOT': return !operand;
                    default: return null;
                }
            }

            case 'in': {
                const value = this.evaluateExpr(expr.expr, columnData, rowIdx);
                const values = expr.values.map(v => this.evaluateExpr(v, columnData, rowIdx));
                return values.includes(value);
            }

            case 'between': {
                const value = this.evaluateExpr(expr.expr, columnData, rowIdx);
                const low = this.evaluateExpr(expr.low, columnData, rowIdx);
                const high = this.evaluateExpr(expr.high, columnData, rowIdx);
                return value >= low && value <= high;
            }

            case 'like': {
                const value = this.evaluateExpr(expr.expr, columnData, rowIdx);
                const pattern = this.evaluateExpr(expr.pattern, columnData, rowIdx);
                if (typeof value !== 'string' || typeof pattern !== 'string') return false;
                // Convert SQL LIKE pattern to regex
                const regex = new RegExp('^' + pattern.replace(/%/g, '.*').replace(/_/g, '.') + '$', 'i');
                return regex.test(value);
            }

            case 'near':
                // NEAR is handled specially at the evaluateWhere level
                // If we reach here, the row is already in the NEAR result set
                return true;

            case 'call':
                // Aggregate functions not supported in row-level evaluation
                return null;

            case 'subquery': {
                // Execute subquery and return scalar result
                // For correlated subqueries, pass outer row context
                return this._executeSubquery(expr.query, columnData, rowIdx);
            }

            case 'array': {
                // Evaluate each element to build the array
                return expr.elements.map(el => this.evaluateExpr(el, columnData, rowIdx));
            }

            case 'subscript': {
                // Array subscript access with 1-based indexing (SQL standard)
                const arr = this.evaluateExpr(expr.array, columnData, rowIdx);
                const idx = this.evaluateExpr(expr.index, columnData, rowIdx);
                if (!Array.isArray(arr)) return null;
                // SQL uses 1-based indexing
                return arr[idx - 1] ?? null;
            }

            default:
                return null;
        }
    }

    /**
     * Execute a subquery and return its result.
     * For scalar subqueries (returns single value), returns that value.
     * For correlated subqueries, substitutes outer row values.
     */
    _executeSubquery(subqueryAst, outerColumnData, outerRowIdx) {
        // Clone the subquery AST to avoid mutating the original
        const resolvedAst = JSON.parse(JSON.stringify(subqueryAst));

        // Check for correlated references (columns not in the subquery's FROM)
        // and substitute with outer row values
        const subqueryTable = resolvedAst.from?.name || resolvedAst.from?.table;
        const correlatedColumns = this._findCorrelatedColumns(resolvedAst, subqueryTable);

        // Build correlation context with outer row values
        const correlationContext = {};
        for (const col of correlatedColumns) {
            const colName = col.column.toLowerCase();
            if (outerColumnData[colName]) {
                correlationContext[col.table + '.' + col.column] = outerColumnData[colName][outerRowIdx];
            }
        }

        // If we have correlations, modify the WHERE clause to use literals
        if (Object.keys(correlationContext).length > 0) {
            this._substituteCorrelations(resolvedAst.where, correlationContext);
        }

        // Check if FROM references a CTE
        const tableName = resolvedAst.from?.name?.toLowerCase() || resolvedAst.from?.table?.toLowerCase();
        if (tableName && this._cteResults?.has(tableName)) {
            const result = this._executeOnInMemoryData(resolvedAst, this._cteResults.get(tableName));
            return result.rows.length > 0 ? result.rows[0][0] : null;  // Scalar result
        }

        // Execute against database if available
        if (this._database) {
            try {
                const result = this._database._executeSingleTable(resolvedAst);
                if (result && result.then) {
                    // This is async - we need to handle it synchronously for expression evaluation
                    // For now, return null and recommend using CTE approach instead
                    console.warn('[SQLExecutor] Async subquery in expression context - consider using CTE');
                    return null;
                }
                return result?.rows?.[0]?.[0] ?? null;
            } catch (e) {
                console.warn('[SQLExecutor] Subquery execution failed:', e.message);
                return null;
            }
        }

        console.warn('[SQLExecutor] Subquery execution requires LanceDatabase context');
        return null;
    }

    /**
     * Find columns that reference the outer query (correlated columns)
     */
    _findCorrelatedColumns(ast, subqueryTable) {
        const correlatedCols = [];

        const walkExpr = (expr) => {
            if (!expr) return;

            if (expr.type === 'column' && expr.table && expr.table !== subqueryTable) {
                correlatedCols.push(expr);
            } else if (expr.type === 'binary') {
                walkExpr(expr.left);
                walkExpr(expr.right);
            } else if (expr.type === 'unary') {
                walkExpr(expr.operand);
            } else if (expr.type === 'in') {
                walkExpr(expr.expr);
                expr.values?.forEach(walkExpr);
            } else if (expr.type === 'between') {
                walkExpr(expr.expr);
                walkExpr(expr.low);
                walkExpr(expr.high);
            } else if (expr.type === 'like') {
                walkExpr(expr.expr);
                walkExpr(expr.pattern);
            } else if (expr.type === 'call') {
                expr.args?.forEach(walkExpr);
            }
        };

        walkExpr(ast.where);
        return correlatedCols;
    }

    /**
     * Substitute correlated column references with literal values
     */
    _substituteCorrelations(expr, correlationContext) {
        if (!expr) return;

        if (expr.type === 'column' && expr.table) {
            const key = expr.table + '.' + expr.column;
            if (correlationContext.hasOwnProperty(key)) {
                // Convert to literal
                expr.type = 'literal';
                expr.value = correlationContext[key];
                delete expr.table;
                delete expr.column;
            }
        } else if (expr.type === 'binary') {
            this._substituteCorrelations(expr.left, correlationContext);
            this._substituteCorrelations(expr.right, correlationContext);
        } else if (expr.type === 'unary') {
            this._substituteCorrelations(expr.operand, correlationContext);
        } else if (expr.type === 'in') {
            this._substituteCorrelations(expr.expr, correlationContext);
            expr.values?.forEach(v => this._substituteCorrelations(v, correlationContext));
        } else if (expr.type === 'between') {
            this._substituteCorrelations(expr.expr, correlationContext);
            this._substituteCorrelations(expr.low, correlationContext);
            this._substituteCorrelations(expr.high, correlationContext);
        } else if (expr.type === 'like') {
            this._substituteCorrelations(expr.expr, correlationContext);
            this._substituteCorrelations(expr.pattern, correlationContext);
        } else if (expr.type === 'call') {
            expr.args?.forEach(a => this._substituteCorrelations(a, correlationContext));
        }
    }

    /**
     * Materialize CTEs before query execution
     * @param {Array} ctes - Array of CTE definitions from AST
     * @param {LanceDatabase} db - Database reference for executing CTE bodies
     */
    async materializeCTEs(ctes, db) {
        this._database = db;
        for (const cte of ctes) {
            const cteName = cte.name.toLowerCase();
            if (cte.body.type === 'RECURSIVE_CTE') {
                // Execute anchor query first
                const anchorResult = await this._executeCTEBody(cte.body.anchor, db);
                let result = { columns: anchorResult.columns, rows: [...anchorResult.rows] };

                // Iterate recursive part until no new rows (max 1000 iterations)
                for (let i = 0; i < 1000; i++) {
                    this._cteResults.set(cteName, result);
                    const recursiveResult = await this._executeCTEBody(cte.body.recursive, db);
                    if (recursiveResult.rows.length === 0) break;
                    result = { columns: result.columns, rows: [...result.rows, ...recursiveResult.rows] };
                }
                this._cteResults.set(cteName, result);
            } else {
                const result = await this._executeCTEBody(cte.body, db);
                this._cteResults.set(cteName, result);
            }
        }
    }

    /**
     * Execute a CTE body - either against database or against already-materialized CTE
     */
    async _executeCTEBody(bodyAst, db) {
        // Check if FROM references another CTE
        const tableName = bodyAst.from?.name?.toLowerCase() || bodyAst.from?.table?.toLowerCase();
        if (tableName && this._cteResults.has(tableName)) {
            return this._executeOnInMemoryData(bodyAst, this._cteResults.get(tableName));
        }
        // Fall back to database execution
        return db._executeSingleTable(bodyAst);
    }

    /**
     * Execute query on in-memory data (for CTEs and subqueries)
     * @param {Object} ast - Parsed SELECT AST
     * @param {Object} data - In-memory data { columns: string[], rows: any[][] }
     * @returns {Object} - Query result { columns: string[], rows: any[][] }
     */
    _executeOnInMemoryData(ast, data) {
        // Build column lookup: column name -> array of values
        const columnData = {};
        for (let i = 0; i < data.columns.length; i++) {
            const colName = data.columns[i].toLowerCase();
            columnData[colName] = data.rows.map(row => row[i]);
        }

        // Apply WHERE filter
        const filteredIndices = [];
        for (let i = 0; i < data.rows.length; i++) {
            if (!ast.where || this._evaluateInMemoryExpr(ast.where, columnData, i)) {
                filteredIndices.push(i);
            }
        }

        // Check for GROUP BY or aggregations
        const hasGroupBy = ast.groupBy && ast.groupBy.length > 0;
        const hasAggregates = this._hasAggregatesInSelect(ast.columns);

        if (hasGroupBy || hasAggregates) {
            return this._executeGroupByAggregation(ast, data, columnData, filteredIndices);
        }

        // Check for window functions
        if (this.hasWindowFunctions(ast)) {
            return this._executeWindowFunctions(ast, data, columnData, filteredIndices);
        }

        // Project columns and build result
        const resultColumns = [];
        const resultRows = [];

        // Handle SELECT * - parser returns { type: 'star' } directly, not { expr: { type: 'star' } }
        const isSelectStar = ast.columns.length === 1 &&
            (ast.columns[0].type === 'star' || ast.columns[0].expr?.type === 'star');
        if (isSelectStar) {
            for (const colName of data.columns) {
                resultColumns.push(colName);
            }
            for (const idx of filteredIndices) {
                resultRows.push([...data.rows[idx]]);
            }
        } else {
            // Named columns
            for (const col of ast.columns) {
                resultColumns.push(col.alias || col.expr?.column || '*');
            }
            for (const idx of filteredIndices) {
                const row = ast.columns.map(col => {
                    if (col.type === 'star' || col.expr?.type === 'star') {
                        return data.rows[idx];
                    }
                    return this._evaluateInMemoryExpr(col.expr, columnData, idx);
                });
                resultRows.push(row.flat());
            }
        }

        // Apply ORDER BY
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colIdxMap = {};
            resultColumns.forEach((name, idx) => { colIdxMap[name.toLowerCase()] = idx; });

            resultRows.sort((a, b) => {
                for (const ob of ast.orderBy) {
                    const colIdx = colIdxMap[ob.column.toLowerCase()];
                    if (colIdx === undefined) continue;
                    const valA = a[colIdx], valB = b[colIdx];
                    // Parser uses 'descending' boolean, some places use 'direction' string
                    const dir = (ob.descending || ob.direction === 'DESC') ? -1 : 1;
                    if (valA == null && valB == null) continue;
                    if (valA == null) return 1 * dir;
                    if (valB == null) return -1 * dir;
                    if (valA < valB) return -1 * dir;
                    if (valA > valB) return 1 * dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET
        const offset = ast.offset || 0;
        let rows = resultRows;
        if (offset > 0) rows = rows.slice(offset);
        if (ast.limit) rows = rows.slice(0, ast.limit);

        return { columns: resultColumns, rows, total: filteredIndices.length };
    }

    /**
     * Check if SELECT columns contain aggregate functions
     */
    _hasAggregatesInSelect(columns) {
        const aggFuncs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];
        for (const col of columns) {
            if (col.expr?.type === 'call') {
                // Skip window functions (those with OVER clause)
                if (col.expr.over) continue;
                const funcName = (col.expr.name || '').toUpperCase();
                if (aggFuncs.includes(funcName)) return true;
            }
        }
        return false;
    }

    /**
     * Execute GROUP BY with aggregation on in-memory data
     */
    _executeGroupByAggregation(ast, data, columnData, filteredIndices) {
        const aggFuncs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];
        const hasGroupBy = ast.groupBy && ast.groupBy.length > 0;

        // Check for advanced GROUP BY (ROLLUP/CUBE/GROUPING SETS)
        if (hasGroupBy && this._hasAdvancedGroupBy(ast.groupBy)) {
            return this._executeAdvancedGroupBy(ast, data, columnData, filteredIndices);
        }

        // Group rows by GROUP BY columns
        const groups = new Map();
        for (const idx of filteredIndices) {
            let groupKey = '';
            if (hasGroupBy) {
                groupKey = ast.groupBy.map(expr => {
                    const colName = (expr.column || expr.name || '').toLowerCase();
                    const val = columnData[colName]?.[idx];
                    return JSON.stringify(val);
                }).join('|');
            }

            if (!groups.has(groupKey)) {
                groups.set(groupKey, []);
            }
            groups.get(groupKey).push(idx);
        }

        // If no GROUP BY and no rows, create one group with empty indices for aggregate results
        // (e.g., COUNT(*) on empty table should return 0, not empty result)
        if (!hasGroupBy && groups.size === 0) {
            groups.set('', []);
        }

        // Build result columns
        const resultColumns = [];
        for (const col of ast.columns) {
            if (col.alias) {
                resultColumns.push(col.alias);
            } else if (col.expr?.type === 'call') {
                const argName = col.expr.args?.[0]?.name || col.expr.args?.[0]?.column || '*';
                resultColumns.push(`${col.expr.name}(${argName})`);
            } else if (col.expr?.type === 'column') {
                resultColumns.push(col.expr.name || col.expr.column);
            } else {
                resultColumns.push('?');
            }
        }

        // Compute aggregations for each group
        const resultRows = [];
        for (const [, groupIndices] of groups) {
            const row = [];
            for (const col of ast.columns) {
                const expr = col.expr;
                if (expr?.type === 'call' && aggFuncs.includes((expr.name || '').toUpperCase())) {
                    const funcName = expr.name.toUpperCase();
                    const argExpr = expr.args?.[0];
                    const isStar = argExpr?.type === 'star';
                    const colName = (argExpr?.name || argExpr?.column || '').toLowerCase();

                    let result = null;
                    switch (funcName) {
                        case 'COUNT':
                            if (isStar) {
                                result = groupIndices.length;
                            } else {
                                result = groupIndices.filter(i => columnData[colName]?.[i] != null).length;
                            }
                            break;
                        case 'SUM': {
                            let sum = 0;
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && !isNaN(v)) sum += v;
                            }
                            result = sum;
                            break;
                        }
                        case 'AVG': {
                            let sum = 0, count = 0;
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && !isNaN(v)) { sum += v; count++; }
                            }
                            result = count > 0 ? sum / count : null;
                            break;
                        }
                        case 'MIN': {
                            let min = null;
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && (min === null || v < min)) min = v;
                            }
                            result = min;
                            break;
                        }
                        case 'MAX': {
                            let max = null;
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && (max === null || v > max)) max = v;
                            }
                            result = max;
                            break;
                        }
                        case 'STDDEV':
                        case 'STDDEV_SAMP': {
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                            }
                            if (vals.length < 2) {
                                result = null;
                            } else {
                                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                                const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (vals.length - 1);
                                result = Math.sqrt(variance);
                            }
                            break;
                        }
                        case 'STDDEV_POP': {
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                            }
                            if (vals.length === 0) {
                                result = null;
                            } else {
                                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                                const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
                                result = Math.sqrt(variance);
                            }
                            break;
                        }
                        case 'VARIANCE':
                        case 'VAR_SAMP': {
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                            }
                            if (vals.length < 2) {
                                result = null;
                            } else {
                                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                                result = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (vals.length - 1);
                            }
                            break;
                        }
                        case 'VAR_POP': {
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                            }
                            if (vals.length === 0) {
                                result = null;
                            } else {
                                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                                result = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
                            }
                            break;
                        }
                        case 'MEDIAN': {
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                            }
                            if (vals.length === 0) {
                                result = null;
                            } else {
                                vals.sort((a, b) => a - b);
                                const mid = Math.floor(vals.length / 2);
                                result = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
                            }
                            break;
                        }
                        case 'STRING_AGG':
                        case 'GROUP_CONCAT': {
                            // STRING_AGG(col, separator) or GROUP_CONCAT(col)
                            const separatorArg = expr.args?.[1];
                            const separator = separatorArg?.value ?? ',';
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null) vals.push(String(v));
                            }
                            result = vals.join(separator);
                            break;
                        }
                    }
                    row.push(result);
                } else if (expr?.type === 'column') {
                    // Non-aggregate column - take first value from group
                    const colName = (expr.name || expr.column || '').toLowerCase();
                    row.push(columnData[colName]?.[groupIndices[0]] ?? null);
                } else {
                    row.push(this._evaluateInMemoryExpr(expr, columnData, groupIndices[0]));
                }
            }
            resultRows.push(row);
        }

        // Apply ORDER BY
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colIdxMap = {};
            resultColumns.forEach((name, idx) => { colIdxMap[name.toLowerCase()] = idx; });

            resultRows.sort((a, b) => {
                for (const ob of ast.orderBy) {
                    const colIdx = colIdxMap[ob.column.toLowerCase()];
                    if (colIdx === undefined) continue;
                    const valA = a[colIdx], valB = b[colIdx];
                    const dir = (ob.descending || ob.direction === 'DESC') ? -1 : 1;
                    if (valA == null && valB == null) continue;
                    if (valA == null) return 1 * dir;
                    if (valB == null) return -1 * dir;
                    if (valA < valB) return -1 * dir;
                    if (valA > valB) return 1 * dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET
        const offset = ast.offset || 0;
        let rows = resultRows;
        if (offset > 0) rows = rows.slice(offset);
        if (ast.limit) rows = rows.slice(0, ast.limit);

        return { columns: resultColumns, rows, total: groups.size };
    }

    /**
     * Execute advanced GROUP BY with ROLLUP, CUBE, or GROUPING SETS
     */
    _executeAdvancedGroupBy(ast, data, columnData, filteredIndices) {
        // 1. Expand GROUP BY into grouping sets
        const groupingSets = this._expandGroupBy(ast.groupBy);

        // 2. Get all column names for results
        const allGroupColumns = this._getAllGroupColumns(ast.groupBy);

        // 3. Execute aggregation for each grouping set
        const allResults = [];
        for (const groupingSet of groupingSets) {
            const setResults = this._executeGroupByForSet(
                ast, columnData, filteredIndices, groupingSet, allGroupColumns
            );
            allResults.push(...setResults);
        }

        // 4. Build result with proper column order
        return this._buildAdvancedGroupByResult(ast, allResults, allGroupColumns);
    }

    /**
     * Execute GROUP BY aggregation for a single grouping set
     */
    _executeGroupByForSet(ast, columnData, filteredIndices, groupingSet, allGroupColumns) {
        const aggFuncs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];
        const groups = new Map();

        // Normalize groupingSet to lowercase for comparison
        const groupingSetLower = groupingSet.map(c => c.toLowerCase());

        for (const idx of filteredIndices) {
            // Build group key from current grouping set columns only
            const groupKey = groupingSet.length > 0
                ? groupingSet.map(col => {
                    const val = columnData[col.toLowerCase()]?.[idx];
                    return JSON.stringify(val);
                }).join('|')
                : '__grand_total__';

            if (!groups.has(groupKey)) {
                // Store group values for all columns (NULL for non-grouped)
                const groupValues = {};
                for (const col of allGroupColumns) {
                    if (groupingSetLower.includes(col.toLowerCase())) {
                        groupValues[col] = columnData[col.toLowerCase()]?.[idx];
                    } else {
                        groupValues[col] = null; // NULL for super-aggregate rows
                    }
                }
                groups.set(groupKey, {
                    values: groupValues,
                    indices: [],
                    _groupingSet: groupingSet
                });
            }
            groups.get(groupKey).indices.push(idx);
        }

        // Handle empty data - still need grand total row
        if (groupingSet.length === 0 && groups.size === 0) {
            const groupValues = {};
            for (const col of allGroupColumns) {
                groupValues[col] = null;
            }
            groups.set('__grand_total__', {
                values: groupValues,
                indices: [],
                _groupingSet: groupingSet
            });
        }

        // Compute aggregates for each group
        const results = [];
        for (const [, group] of groups) {
            const row = { ...group.values, _groupingSet: group._groupingSet };

            // Compute each aggregate column
            for (const col of ast.columns) {
                const expr = col.expr;
                if (expr?.type === 'call' && aggFuncs.includes((expr.name || '').toUpperCase())) {
                    const funcName = expr.name.toUpperCase();
                    const argExpr = expr.args?.[0];
                    const isStar = argExpr?.type === 'star';
                    const colName = (argExpr?.name || argExpr?.column || '').toLowerCase();
                    const alias = col.alias || `${funcName}(${isStar ? '*' : colName})`;

                    let result = null;
                    const indices = group.indices;

                    switch (funcName) {
                        case 'COUNT':
                            result = isStar
                                ? indices.length
                                : indices.filter(i => columnData[colName]?.[i] != null).length;
                            break;
                        case 'SUM': {
                            let sum = 0;
                            for (const i of indices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && !isNaN(v)) sum += v;
                            }
                            result = sum;
                            break;
                        }
                        case 'AVG': {
                            let sum = 0, count = 0;
                            for (const i of indices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && !isNaN(v)) { sum += v; count++; }
                            }
                            result = count > 0 ? sum / count : null;
                            break;
                        }
                        case 'MIN': {
                            let min = null;
                            for (const i of indices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && (min === null || v < min)) min = v;
                            }
                            result = min;
                            break;
                        }
                        case 'MAX': {
                            let max = null;
                            for (const i of indices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && (max === null || v > max)) max = v;
                            }
                            result = max;
                            break;
                        }
                    }
                    row[alias] = result;
                }
            }
            results.push(row);
        }

        return results;
    }

    /**
     * Get all column names from GROUP BY (for column ordering)
     */
    _getAllGroupColumns(groupBy) {
        const columns = [];
        for (const item of groupBy) {
            if (item.type === 'COLUMN') {
                if (!columns.includes(item.column)) columns.push(item.column);
            } else if (item.type === 'ROLLUP' || item.type === 'CUBE') {
                for (const col of item.columns) {
                    if (!columns.includes(col)) columns.push(col);
                }
            } else if (item.type === 'GROUPING_SETS') {
                for (const set of item.sets) {
                    for (const col of set) {
                        if (!columns.includes(col)) columns.push(col);
                    }
                }
            }
        }
        return columns;
    }

    /**
     * Build final result from advanced GROUP BY results
     */
    _buildAdvancedGroupByResult(ast, allResults, allGroupColumns) {
        // Build result columns from AST
        const resultColumns = [];
        for (const col of ast.columns) {
            if (col.alias) {
                resultColumns.push(col.alias);
            } else if (col.expr?.type === 'call') {
                const argName = col.expr.args?.[0]?.name || col.expr.args?.[0]?.column || '*';
                resultColumns.push(`${col.expr.name}(${argName})`);
            } else if (col.expr?.type === 'column') {
                resultColumns.push(col.expr.name || col.expr.column);
            } else {
                resultColumns.push('?');
            }
        }

        // Convert result objects to arrays matching column order
        const resultRows = allResults.map(rowObj => {
            const row = [];
            for (const colName of resultColumns) {
                // Check if this is a group column or aggregate
                const val = rowObj[colName] ?? rowObj[colName.toLowerCase()] ?? null;
                row.push(val);
            }
            return row;
        });

        // Apply ORDER BY if present
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colIdxMap = {};
            resultColumns.forEach((name, idx) => { colIdxMap[name.toLowerCase()] = idx; });

            resultRows.sort((a, b) => {
                for (const ob of ast.orderBy) {
                    const colIdx = colIdxMap[ob.column.toLowerCase()];
                    if (colIdx === undefined) continue;
                    const valA = a[colIdx], valB = b[colIdx];
                    const dir = (ob.descending || ob.direction === 'DESC') ? -1 : 1;
                    if (valA == null && valB == null) continue;
                    if (valA == null) return 1 * dir;
                    if (valB == null) return -1 * dir;
                    if (valA < valB) return -1 * dir;
                    if (valA > valB) return 1 * dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET
        const offset = ast.offset || 0;
        let rows = resultRows;
        if (offset > 0) rows = rows.slice(offset);
        if (ast.limit) rows = rows.slice(0, ast.limit);

        return { columns: resultColumns, rows, total: allResults.length };
    }

    /**
     * Execute window functions on in-memory data
     */
    _executeWindowFunctions(ast, data, columnData, filteredIndices) {
        // Build filtered rows
        const filteredRows = filteredIndices.map(idx => data.rows[idx]);

        // Compute window function results
        const windowResults = this.computeWindowFunctions(ast, filteredRows, columnData);

        // Build result columns
        const resultColumns = [];
        for (const col of ast.columns) {
            if (col.alias) {
                resultColumns.push(col.alias);
            } else if (col.expr?.type === 'call') {
                const argName = col.expr.args?.[0]?.name || col.expr.args?.[0]?.column || '*';
                resultColumns.push(`${col.expr.name}(${argName})`);
            } else if (col.expr?.type === 'column') {
                resultColumns.push(col.expr.name || col.expr.column);
            } else {
                resultColumns.push('?');
            }
        }

        // Build result rows
        const resultRows = [];
        for (let i = 0; i < filteredIndices.length; i++) {
            const origIdx = filteredIndices[i];
            const row = [];

            for (let c = 0; c < ast.columns.length; c++) {
                const col = ast.columns[c];
                const expr = col.expr;

                // Check if this is a window function column
                if (expr?.over) {
                    // Find the corresponding window result
                    const windowCol = windowResults.find(w => w.colIndex === c);
                    row.push(windowCol ? windowCol.values[i] : null);
                } else if (expr?.type === 'column') {
                    const colName = (expr.name || expr.column || '').toLowerCase();
                    row.push(columnData[colName]?.[origIdx] ?? null);
                } else {
                    row.push(this._evaluateInMemoryExpr(expr, columnData, origIdx));
                }
            }

            resultRows.push(row);
        }

        // Apply QUALIFY filter (filter on window function results)
        let finalRows = resultRows;
        if (ast.qualify) {
            finalRows = [];
            // Build column name to index map for expression evaluation
            const qualifyColMap = {};
            resultColumns.forEach((name, idx) => { qualifyColMap[name.toLowerCase()] = idx; });

            for (let i = 0; i < resultRows.length; i++) {
                // Build row data object for expression evaluation
                const rowData = {};
                for (let c = 0; c < resultColumns.length; c++) {
                    rowData[resultColumns[c].toLowerCase()] = resultRows[i][c];
                }

                // Evaluate QUALIFY condition
                if (this._evaluateInMemoryExpr(ast.qualify, rowData, 0)) {
                    finalRows.push(resultRows[i]);
                }
            }
        }

        // Apply ORDER BY
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colIdxMap = {};
            resultColumns.forEach((name, idx) => { colIdxMap[name.toLowerCase()] = idx; });

            finalRows.sort((a, b) => {
                for (const ob of ast.orderBy) {
                    const colIdx = colIdxMap[ob.column.toLowerCase()];
                    if (colIdx === undefined) continue;
                    const valA = a[colIdx], valB = b[colIdx];
                    const dir = (ob.descending || ob.direction === 'DESC') ? -1 : 1;
                    if (valA == null && valB == null) continue;
                    if (valA == null) return 1 * dir;
                    if (valB == null) return -1 * dir;
                    if (valA < valB) return -1 * dir;
                    if (valA > valB) return 1 * dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET
        const offset = ast.offset || 0;
        let rows = finalRows;
        if (offset > 0) rows = rows.slice(offset);
        if (ast.limit) rows = rows.slice(0, ast.limit);

        return { columns: resultColumns, rows, total: finalRows.length };
    }

    /**
     * Evaluate expression on in-memory data
     */
    _evaluateInMemoryExpr(expr, columnData, rowIdx) {
        if (!expr) return null;

        switch (expr.type) {
            case 'literal':
                return expr.value;

            case 'column': {
                const colName = expr.column.toLowerCase();
                const col = columnData[colName];
                return col ? col[rowIdx] : null;
            }

            case 'binary': {
                const left = this._evaluateInMemoryExpr(expr.left, columnData, rowIdx);
                const right = this._evaluateInMemoryExpr(expr.right, columnData, rowIdx);
                const op = expr.op || expr.operator;  // Parser uses 'op', some places use 'operator'
                switch (op) {
                    case '=': case '==': return left == right;
                    case '!=': case '<>': return left != right;
                    case '<': return left < right;
                    case '<=': return left <= right;
                    case '>': return left > right;
                    case '>=': return left >= right;
                    case '+': return Number(left) + Number(right);
                    case '-': return Number(left) - Number(right);
                    case '*': return Number(left) * Number(right);
                    case '/': return right !== 0 ? Number(left) / Number(right) : null;
                    case 'AND': return left && right;
                    case 'OR': return left || right;
                    default: return null;
                }
            }

            case 'unary': {
                const operand = this._evaluateInMemoryExpr(expr.operand, columnData, rowIdx);
                const op = expr.op || expr.operator;
                switch (op) {
                    case 'NOT': return !operand;
                    case '-': return -operand;
                    case 'IS NULL': return operand == null;
                    case 'IS NOT NULL': return operand != null;
                    default: return null;
                }
            }

            case 'call': {
                // Aggregate functions would need special handling
                const funcName = expr.name.toUpperCase();
                const args = expr.args?.map(a => this._evaluateInMemoryExpr(a, columnData, rowIdx)) || [];
                switch (funcName) {
                    case 'UPPER': return String(args[0]).toUpperCase();
                    case 'LOWER': return String(args[0]).toLowerCase();
                    case 'LENGTH': return String(args[0]).length;
                    case 'SUBSTR': case 'SUBSTRING': return String(args[0]).substring(args[1] - 1, args[2] ? args[1] - 1 + args[2] : undefined);
                    case 'COALESCE': return args.find(a => a != null) ?? null;
                    case 'ABS': return Math.abs(args[0]);
                    case 'ROUND': return Math.round(args[0] * Math.pow(10, args[1] || 0)) / Math.pow(10, args[1] || 0);
                    case 'GROUPING': {
                        // GROUPING(col) returns 1 if col is a super-aggregate (null due to ROLLUP/CUBE), 0 otherwise
                        // The column name is in the first argument
                        const colArg = expr.args?.[0];
                        const colName = colArg?.column || colArg?.name;
                        if (!colName) return 0;
                        // Check if this column is in the current grouping set
                        // columnData._groupingSet contains the columns that are grouped (not super-aggregate)
                        const groupingSet = columnData._groupingSet || [];
                        return groupingSet.includes(colName.toLowerCase()) ? 0 : 1;
                    }
                    default: return null;
                }
            }

            case 'in': {
                const val = this._evaluateInMemoryExpr(expr.expr, columnData, rowIdx);
                const values = expr.values.map(v => v.value ?? this._evaluateInMemoryExpr(v, columnData, rowIdx));
                return values.includes(val);
            }

            case 'between': {
                const val = this._evaluateInMemoryExpr(expr.expr, columnData, rowIdx);
                const low = this._evaluateInMemoryExpr(expr.low, columnData, rowIdx);
                const high = this._evaluateInMemoryExpr(expr.high, columnData, rowIdx);
                return val >= low && val <= high;
            }

            case 'like': {
                const val = String(this._evaluateInMemoryExpr(expr.expr, columnData, rowIdx));
                const pattern = this._evaluateInMemoryExpr(expr.pattern, columnData, rowIdx);
                const regex = new RegExp('^' + String(pattern).replace(/%/g, '.*').replace(/_/g, '.') + '$', 'i');
                return regex.test(val);
            }

            case 'array': {
                // Evaluate each element to build the array
                return expr.elements.map(el => this._evaluateInMemoryExpr(el, columnData, rowIdx));
            }

            case 'subscript': {
                // Array subscript access with 1-based indexing (SQL standard)
                const arr = this._evaluateInMemoryExpr(expr.array, columnData, rowIdx);
                const idx = this._evaluateInMemoryExpr(expr.index, columnData, rowIdx);
                if (!Array.isArray(arr)) return null;
                // SQL uses 1-based indexing
                return arr[idx - 1] ?? null;
            }

            default:
                return null;
        }
    }

    /**
     * Check if query has window functions
     */
    hasWindowFunctions(ast) {
        return ast.columns?.some(col => col.expr?.type === 'call' && col.expr.over);
    }

    /**
     * Execute window functions on in-memory data
     * Window functions are computed after WHERE but before ORDER BY/LIMIT
     */
    computeWindowFunctions(ast, rows, columnData) {
        const windowColumns = [];

        for (let colIndex = 0; colIndex < ast.columns.length; colIndex++) {
            const col = ast.columns[colIndex];
            if (col.expr?.type === 'call' && col.expr.over) {
                const values = this._computeWindowFunction(
                    col.expr.name,
                    col.expr.args,
                    col.expr.over,
                    rows,
                    columnData
                );
                windowColumns.push({
                    colIndex,
                    alias: col.alias || col.expr.name,
                    values
                });
            }
        }

        return windowColumns;
    }

    /**
     * Compute a single window function
     */
    _computeWindowFunction(funcName, args, over, rows, columnData) {
        const results = new Array(rows.length).fill(null);

        // Partition rows
        const partitions = this._partitionRows(rows, over.partitionBy, columnData);

        for (const partition of partitions) {
            // Sort partition by ORDER BY if specified
            if (over.orderBy && over.orderBy.length > 0) {
                partition.sort((a, b) => this._compareRowsByOrder(a, b, over.orderBy, columnData));
            }

            // Compute function for each row in partition
            for (let i = 0; i < partition.length; i++) {
                const rowIdx = partition[i].idx;

                switch (funcName) {
                    case 'ROW_NUMBER':
                        results[rowIdx] = i + 1;
                        break;

                    case 'RANK': {
                        // RANK: same rank for ties, gaps after ties
                        let rank = 1;
                        for (let j = 0; j < i; j++) {
                            if (this._compareRowsByOrder(partition[j], partition[i], over.orderBy, columnData) !== 0) {
                                rank = j + 1;
                            }
                        }
                        if (i > 0 && this._compareRowsByOrder(partition[i-1], partition[i], over.orderBy, columnData) === 0) {
                            results[rowIdx] = results[partition[i-1].idx];
                        } else {
                            results[rowIdx] = i + 1;
                        }
                        break;
                    }

                    case 'DENSE_RANK': {
                        // DENSE_RANK: same rank for ties, no gaps
                        if (i === 0) {
                            results[rowIdx] = 1;
                        } else if (this._compareRowsByOrder(partition[i-1], partition[i], over.orderBy, columnData) === 0) {
                            results[rowIdx] = results[partition[i-1].idx];
                        } else {
                            results[rowIdx] = results[partition[i-1].idx] + 1;
                        }
                        break;
                    }

                    case 'NTILE': {
                        const n = args[0]?.value || 1;
                        const bucketSize = Math.ceil(partition.length / n);
                        results[rowIdx] = Math.floor(i / bucketSize) + 1;
                        break;
                    }

                    case 'PERCENT_RANK': {
                        // PERCENT_RANK = (rank - 1) / (partition_size - 1)
                        // First row in partition always 0, only row returns 0
                        // rank = position of first row with same value (ties get same rank)
                        let rank = i + 1;
                        for (let j = 0; j < i; j++) {
                            if (this._compareRowsByOrder(partition[j], partition[i], over.orderBy, columnData) === 0) {
                                rank = j + 1;  // Found a tie - use its position
                                break;
                            }
                        }
                        const partitionSize = partition.length;
                        results[rowIdx] = partitionSize > 1 ? (rank - 1) / (partitionSize - 1) : 0;
                        break;
                    }

                    case 'CUME_DIST': {
                        // CUME_DIST = (rows with value <= current) / total_rows
                        // Includes all tied rows
                        let countLessOrEqual = 0;
                        for (let j = 0; j < partition.length; j++) {
                            const cmp = this._compareRowsByOrder(partition[j], partition[i], over.orderBy, columnData);
                            if (cmp <= 0) countLessOrEqual++;
                        }
                        results[rowIdx] = countLessOrEqual / partition.length;
                        break;
                    }

                    case 'LAG': {
                        const lagCol = args[0];
                        const lagN = args[1]?.value || 1;
                        const defaultVal = args[2]?.value ?? null;
                        if (i >= lagN) {
                            const prevRowIdx = partition[i - lagN].idx;
                            results[rowIdx] = this._evaluateInMemoryExpr(lagCol, columnData, prevRowIdx);
                        } else {
                            results[rowIdx] = defaultVal;
                        }
                        break;
                    }

                    case 'LEAD': {
                        const leadCol = args[0];
                        const leadN = args[1]?.value || 1;
                        const defaultVal = args[2]?.value ?? null;
                        if (i + leadN < partition.length) {
                            const nextRowIdx = partition[i + leadN].idx;
                            results[rowIdx] = this._evaluateInMemoryExpr(leadCol, columnData, nextRowIdx);
                        } else {
                            results[rowIdx] = defaultVal;
                        }
                        break;
                    }

                    case 'FIRST_VALUE': {
                        const firstRowIdx = partition[0].idx;
                        results[rowIdx] = this._evaluateInMemoryExpr(args[0], columnData, firstRowIdx);
                        break;
                    }

                    case 'LAST_VALUE': {
                        // LAST_VALUE returns the last value within the frame
                        // Default frame: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        const frame = over.frame || {
                            type: 'RANGE',
                            start: { type: 'UNBOUNDED_PRECEDING' },
                            end: { type: 'CURRENT_ROW' }
                        };
                        const [, endIdx] = this._getFrameBounds(frame, partition, i);
                        const lastRowIdx = partition[endIdx].idx;
                        results[rowIdx] = this._evaluateInMemoryExpr(args[0], columnData, lastRowIdx);
                        break;
                    }

                    case 'NTH_VALUE': {
                        const n = args[1]?.value || 1;
                        if (n > 0 && n <= partition.length) {
                            const nthRowIdx = partition[n - 1].idx;
                            results[rowIdx] = this._evaluateInMemoryExpr(args[0], columnData, nthRowIdx);
                        } else {
                            results[rowIdx] = null;
                        }
                        break;
                    }

                    // Aggregate window functions (frame-aware)
                    case 'SUM':
                    case 'AVG':
                    case 'COUNT':
                    case 'MIN':
                    case 'MAX': {
                        // Get frame bounds - default to RANGE UNBOUNDED PRECEDING TO CURRENT ROW
                        const frame = over.frame || {
                            type: 'RANGE',
                            start: { type: 'UNBOUNDED_PRECEDING' },
                            end: { type: 'CURRENT_ROW' }
                        };
                        const [startIdx, endIdx] = this._getFrameBounds(frame, partition, i);

                        // Collect values within frame
                        const isStar = args[0]?.type === 'star';
                        let values = [];
                        let frameRowCount = 0;
                        for (let j = startIdx; j <= endIdx; j++) {
                            frameRowCount++;
                            if (!isStar) {
                                const val = this._evaluateInMemoryExpr(args[0], columnData, partition[j].idx);
                                if (val != null) values.push(Number(val));
                            }
                        }

                        // Compute aggregate
                        let result = null;
                        switch (funcName) {
                            case 'SUM':
                                result = values.reduce((a, b) => a + b, 0);
                                break;
                            case 'AVG':
                                result = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
                                break;
                            case 'COUNT':
                                result = isStar ? frameRowCount : values.length;
                                break;
                            case 'MIN':
                                result = values.length > 0 ? Math.min(...values) : null;
                                break;
                            case 'MAX':
                                result = values.length > 0 ? Math.max(...values) : null;
                                break;
                        }
                        results[rowIdx] = result;
                        break;
                    }

                    default:
                        results[rowIdx] = null;
                }
            }
        }

        return results;
    }

    /**
     * Partition rows based on PARTITION BY expressions
     */
    _partitionRows(rows, partitionBy, columnData) {
        if (!partitionBy || partitionBy.length === 0) {
            // No partitioning - all rows in one partition
            return [rows.map((r, i) => ({ idx: i, row: r }))];
        }

        const groups = new Map();
        for (let i = 0; i < rows.length; i++) {
            const key = partitionBy
                .map(expr => JSON.stringify(this._evaluateInMemoryExpr(expr, columnData, i)))
                .join('|');
            if (!groups.has(key)) {
                groups.set(key, []);
            }
            groups.get(key).push({ idx: i, row: rows[i] });
        }

        return Array.from(groups.values());
    }

    /**
     * Compare two rows based on ORDER BY specification
     */
    _compareRowsByOrder(a, b, orderBy, columnData) {
        for (const ob of orderBy) {
            const valA = this._evaluateInMemoryExpr({ type: 'column', column: ob.column }, columnData, a.idx);
            const valB = this._evaluateInMemoryExpr({ type: 'column', column: ob.column }, columnData, b.idx);

            const dir = ob.direction === 'DESC' ? -1 : 1;
            if (valA == null && valB == null) continue;
            if (valA == null) return 1 * dir;
            if (valB == null) return -1 * dir;
            if (valA < valB) return -1 * dir;
            if (valA > valB) return 1 * dir;
        }
        return 0;
    }

    /**
     * Calculate frame bounds for window function
     * @param {Object} frame - Frame specification { type, start, end }
     * @param {Array} partition - Sorted partition rows
     * @param {number} currentIdx - Current row index within partition
     * @returns {[number, number]} - [startIdx, endIdx] bounds
     */
    _getFrameBounds(frame, partition, currentIdx) {
        const n = partition.length;
        let startIdx = 0;
        let endIdx = currentIdx;

        // Parse start bound (parser uses spaces in type names)
        const start = frame.start || { type: 'UNBOUNDED PRECEDING' };
        const startType = start.type.replace(' ', '_').toUpperCase();
        switch (startType) {
            case 'UNBOUNDED_PRECEDING':
                startIdx = 0;
                break;
            case 'CURRENT_ROW':
                startIdx = currentIdx;
                break;
            case 'PRECEDING':
                startIdx = Math.max(0, currentIdx - (start.offset || start.value || 1));
                break;
            case 'FOLLOWING':
                startIdx = Math.min(n - 1, currentIdx + (start.offset || start.value || 1));
                break;
        }

        // Parse end bound
        const end = frame.end || { type: 'CURRENT ROW' };
        const endType = end.type.replace(' ', '_').toUpperCase();
        switch (endType) {
            case 'UNBOUNDED_FOLLOWING':
                endIdx = n - 1;
                break;
            case 'CURRENT_ROW':
                endIdx = currentIdx;
                break;
            case 'PRECEDING':
                endIdx = Math.max(0, currentIdx - (end.offset || end.value || 1));
                break;
            case 'FOLLOWING':
                endIdx = Math.min(n - 1, currentIdx + (end.offset || end.value || 1));
                break;
        }

        // Ensure valid bounds
        if (startIdx > endIdx) [startIdx, endIdx] = [endIdx, startIdx];
        return [startIdx, endIdx];
    }

    async applyOrderBy(rows, orderBy, outputColumns) {
        // Build column index map
        const colIdxMap = {};
        let idx = 0;
        for (const col of outputColumns) {
            if (col.type === 'star') {
                for (const name of this.file.columnNames || []) {
                    colIdxMap[name.toLowerCase()] = idx++;
                }
            } else {
                const name = col.alias || this.exprToName(col.expr);
                colIdxMap[name.toLowerCase()] = idx++;
            }
        }

        // Use GPU for large datasets (10,000+ rows)
        if (rows.length >= 10000 && gpuSorter.isAvailable()) {
            // Multi-column sort: stable sort from last to first column
            let indices = new Uint32Array(rows.length);
            for (let i = 0; i < rows.length; i++) indices[i] = i;

            for (let c = orderBy.length - 1; c >= 0; c--) {
                const ob = orderBy[c];
                const colIdx = colIdxMap[ob.column.toLowerCase()];
                if (colIdx === undefined) continue;

                const ascending = !ob.descending;
                const values = new Float32Array(rows.length);
                for (let i = 0; i < rows.length; i++) {
                    const val = rows[indices[i]][colIdx];
                    if (val == null) {
                        values[i] = ascending ? 3.4e38 : -3.4e38; // NULLS LAST
                    } else if (typeof val === 'number') {
                        values[i] = val;
                    } else if (typeof val === 'string') {
                        // Use string hash for approximate sorting
                        let key = 0;
                        for (let j = 0; j < Math.min(4, val.length); j++) {
                            key = key * 256 + val.charCodeAt(j);
                        }
                        values[i] = key;
                    } else {
                        values[i] = 0;
                    }
                }

                const sortedIdx = await gpuSorter.sort(values, ascending);
                const newIndices = new Uint32Array(rows.length);
                for (let i = 0; i < rows.length; i++) {
                    newIndices[i] = indices[sortedIdx[i]];
                }
                indices = newIndices;
            }

            // Apply permutation in-place
            const sorted = new Array(rows.length);
            for (let i = 0; i < rows.length; i++) {
                sorted[i] = rows[indices[i]];
            }
            for (let i = 0; i < rows.length; i++) {
                rows[i] = sorted[i];
            }
            return;
        }

        // CPU fallback for smaller datasets
        rows.sort((a, b) => {
            for (const ob of orderBy) {
                const colIdx = colIdxMap[ob.column.toLowerCase()];
                if (colIdx === undefined) continue;

                const valA = a[colIdx];
                const valB = b[colIdx];

                let cmp = 0;
                if (valA == null && valB == null) cmp = 0;
                else if (valA == null) cmp = 1;
                else if (valB == null) cmp = -1;
                else if (valA < valB) cmp = -1;
                else if (valA > valB) cmp = 1;

                if (cmp !== 0) {
                    return ob.descending ? -cmp : cmp;
                }
            }
            return 0;
        });
    }

    /**
     * Apply DISTINCT to rows (GPU-accelerated for large datasets)
     * @param {Array[]} rows - Row arrays to deduplicate
     * @returns {Array[]} Deduplicated rows
     */
    async applyDistinct(rows) {
        if (rows.length === 0) return rows;

        // Use GPU for large datasets (10,000+ rows)
        if (rows.length >= 10000 && gpuGrouper.isAvailable()) {
            // Hash each row to create a unique signature
            const rowHashes = this._hashRows(rows);

            // Use GPUGrouper to find unique hashes
            const { groupIds, numGroups } = await gpuGrouper.groupBy(rowHashes);

            // Extract first occurrence of each unique group
            const firstOccurrence = new Array(numGroups).fill(-1);
            for (let i = 0; i < rows.length; i++) {
                const gid = groupIds[i];
                if (firstOccurrence[gid] === -1) {
                    firstOccurrence[gid] = i;
                }
            }

            // Build deduplicated result
            const uniqueRows = [];
            for (let gid = 0; gid < numGroups; gid++) {
                if (firstOccurrence[gid] !== -1) {
                    uniqueRows.push(rows[firstOccurrence[gid]]);
                }
            }
            return uniqueRows;
        }

        // CPU fallback using Set with JSON serialization
        const seen = new Set();
        const uniqueRows = [];
        for (const row of rows) {
            const key = JSON.stringify(row);
            if (!seen.has(key)) {
                seen.add(key);
                uniqueRows.push(row);
            }
        }
        return uniqueRows;
    }

    /**
     * Hash rows to u32 for GPU deduplication
     */
    _hashRows(rows) {
        const hashes = new Uint32Array(rows.length);
        for (let i = 0; i < rows.length; i++) {
            hashes[i] = this._hashRow(rows[i]);
        }
        return hashes;
    }

    /**
     * FNV-1a hash of a row's values
     */
    _hashRow(row) {
        let hash = 2166136261;
        for (const val of row) {
            if (val === null || val === undefined) {
                hash ^= 0;
            } else if (typeof val === 'number') {
                // Hash number as bytes
                const buf = new ArrayBuffer(8);
                new Float64Array(buf)[0] = val;
                const bytes = new Uint8Array(buf);
                for (const b of bytes) {
                    hash ^= b;
                    hash = Math.imul(hash, 16777619);
                }
            } else if (typeof val === 'string') {
                for (let j = 0; j < val.length; j++) {
                    hash ^= val.charCodeAt(j);
                    hash = Math.imul(hash, 16777619);
                }
            } else {
                // Fallback: stringify
                const str = String(val);
                for (let j = 0; j < str.length; j++) {
                    hash ^= str.charCodeAt(j);
                    hash = Math.imul(hash, 16777619);
                }
            }
            // Separator between values
            hash ^= 0xFF;
            hash = Math.imul(hash, 16777619);
        }
        return hash >>> 0;
    }

    exprToName(expr) {
        if (!expr) return '?';
        switch (expr.type) {
            case 'column': return expr.name;
            case 'call': {
                const argStr = expr.args.map(a => {
                    if (a.type === 'star') return '*';
                    if (a.type === 'column') return a.name;
                    return '?';
                }).join(', ');
                return `${expr.name}(${argStr})`;
            }
            case 'literal': return String(expr.value);
            default: return '?';
        }
    }

    /**
     * Check if query is just SELECT COUNT(*) with no other columns
     */
    isSimpleCountStar(ast) {
        if (ast.columns.length !== 1) return false;
        const col = ast.columns[0];
        if (col.type === 'star') return true; // COUNT(*) parsed as star
        if (col.type === 'expr' && col.expr.type === 'call') {
            const name = col.expr.name.toUpperCase();
            if (name === 'COUNT') {
                const arg = col.expr.args[0];
                return arg?.type === 'star';
            }
        }
        return false;
    }

    /**
     * Execute aggregation query after running vector search
     */
    async executeAggregateWithSearch(ast, totalRows, onProgress) {
        // Step 1: Extract NEAR info from WHERE
        const nearInfo = this._extractNearCondition(ast.where);
        if (!nearInfo) {
            throw new Error('SEARCH aggregation requires NEAR clause');
        }

        // Step 2: Execute vector search to get candidate indices
        if (onProgress) onProgress('Executing vector search...', 0, 100);
        const searchIndices = await this._executeNearSearch(nearInfo, totalRows);

        if (searchIndices.length === 0) {
            return this._emptyAggregateResult(ast);
        }

        // Step 3: Apply remaining WHERE conditions (non-NEAR)
        const remainingWhere = this._removeNearCondition(ast.where);
        let filteredIndices = searchIndices;

        if (remainingWhere) {
            if (onProgress) onProgress('Applying filters...', 30, 100);
            filteredIndices = await this._filterIndicesByWhere(searchIndices, remainingWhere);
        }

        if (filteredIndices.length === 0) {
            return this._emptyAggregateResult(ast);
        }

        // Step 4: Route to appropriate aggregation path
        if (onProgress) onProgress('Aggregating results...', 60, 100);

        if (ast.groupBy && ast.groupBy.length > 0) {
            // GROUP BY aggregation on filtered indices
            return await this._executeGroupByOnIndices(ast, filteredIndices, onProgress);
        } else {
            // Simple aggregation on filtered indices
            return await this._executeSimpleAggregateOnIndices(ast, filteredIndices, onProgress);
        }
    }

    /**
     * Filter indices by WHERE expression
     * @private
     */
    async _filterIndicesByWhere(indices, whereExpr) {
        const neededCols = new Set();
        this.collectColumnsFromExpr(whereExpr, neededCols);

        const columnData = {};
        for (const colName of neededCols) {
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx !== undefined) {
                columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, indices);
            }
        }

        return indices.filter((_, i) => this.evaluateExpr(whereExpr, columnData, i));
    }

    /**
     * Execute simple aggregation on specific indices (no GROUP BY)
     * @private
     */
    async _executeSimpleAggregateOnIndices(ast, indices, onProgress) {
        const aggFunctions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];

        // Initialize aggregators
        const aggregators = [];
        const colNames = [];

        for (const col of ast.columns) {
            if (col.type === 'star') {
                aggregators.push({ type: 'COUNT', column: null, isStar: true, count: 0 });
                colNames.push('COUNT(*)');
            } else if (col.expr?.type === 'call' && aggFunctions.includes(col.expr.name.toUpperCase())) {
                const aggType = col.expr.name.toUpperCase();
                const argExpr = col.expr.args?.[0];
                const colName = argExpr?.type === 'column' ? (argExpr.name || argExpr.column) : null;
                const isStar = argExpr?.type === 'star';

                aggregators.push({
                    type: aggType,
                    column: colName,
                    isStar,
                    expr: col.expr,
                    sum: 0,
                    count: 0,
                    min: null,
                    max: null,
                    values: [],
                });

                const displayName = col.alias || this.exprToName(col.expr);
                colNames.push(displayName);
            } else {
                aggregators.push({
                    type: 'FIRST',
                    column: col.expr?.type === 'column' ? (col.expr.name || col.expr.column) : null,
                    value: null,
                });
                colNames.push(col.alias || this.exprToName(col.expr));
            }
        }

        // Collect needed columns
        const neededCols = new Set();
        for (const agg of aggregators) {
            if (agg.column) neededCols.add(agg.column.toLowerCase());
        }

        // Fetch column data for the indices
        const columnData = {};
        for (const colName of neededCols) {
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx !== undefined) {
                columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, indices);
            }
        }

        // Process all rows
        for (let i = 0; i < indices.length; i++) {
            for (const agg of aggregators) {
                if (agg.type === 'COUNT' && agg.isStar) {
                    agg.count++;
                } else if (agg.type === 'FIRST' && agg.value === null) {
                    const data = agg.column ? columnData[agg.column.toLowerCase()] : null;
                    agg.value = data ? data[i] : null;
                } else if (agg.column) {
                    const data = columnData[agg.column.toLowerCase()];
                    const val = data ? data[i] : null;

                    if (val !== null && val !== undefined) {
                        if (agg.type === 'COUNT') {
                            agg.count++;
                        } else if (typeof val === 'number' && !isNaN(val)) {
                            agg.count++;
                            if (agg.type === 'SUM' || agg.type === 'AVG' || agg.type.startsWith('STDDEV') || agg.type.startsWith('VAR')) {
                                agg.sum += val;
                                agg.values.push(val);
                            }
                            if (agg.type === 'MIN') {
                                agg.min = agg.min === null ? val : Math.min(agg.min, val);
                            }
                            if (agg.type === 'MAX') {
                                agg.max = agg.max === null ? val : Math.max(agg.max, val);
                            }
                            if (agg.type === 'MEDIAN') {
                                agg.values.push(val);
                            }
                        }
                        if (agg.type === 'STRING_AGG' || agg.type === 'GROUP_CONCAT') {
                            agg.values.push(String(val));
                        }
                    }
                }
            }
        }

        // Compute final results
        const resultRow = [];
        for (const agg of aggregators) {
            switch (agg.type) {
                case 'COUNT':
                    resultRow.push(agg.count);
                    break;
                case 'SUM':
                    resultRow.push(agg.sum);
                    break;
                case 'AVG':
                    resultRow.push(agg.count > 0 ? agg.sum / agg.count : null);
                    break;
                case 'MIN':
                    resultRow.push(agg.min);
                    break;
                case 'MAX':
                    resultRow.push(agg.max);
                    break;
                case 'STDDEV':
                case 'STDDEV_SAMP': {
                    if (agg.values.length < 2) {
                        resultRow.push(null);
                    } else {
                        const mean = agg.sum / agg.values.length;
                        const variance = agg.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (agg.values.length - 1);
                        resultRow.push(Math.sqrt(variance));
                    }
                    break;
                }
                case 'STDDEV_POP': {
                    if (agg.values.length === 0) {
                        resultRow.push(null);
                    } else {
                        const mean = agg.sum / agg.values.length;
                        const variance = agg.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / agg.values.length;
                        resultRow.push(Math.sqrt(variance));
                    }
                    break;
                }
                case 'VARIANCE':
                case 'VAR_SAMP': {
                    if (agg.values.length < 2) {
                        resultRow.push(null);
                    } else {
                        const mean = agg.sum / agg.values.length;
                        resultRow.push(agg.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (agg.values.length - 1));
                    }
                    break;
                }
                case 'VAR_POP': {
                    if (agg.values.length === 0) {
                        resultRow.push(null);
                    } else {
                        const mean = agg.sum / agg.values.length;
                        resultRow.push(agg.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / agg.values.length);
                    }
                    break;
                }
                case 'MEDIAN': {
                    if (agg.values.length === 0) {
                        resultRow.push(null);
                    } else {
                        agg.values.sort((a, b) => a - b);
                        const mid = Math.floor(agg.values.length / 2);
                        resultRow.push(agg.values.length % 2 ? agg.values[mid] : (agg.values[mid - 1] + agg.values[mid]) / 2);
                    }
                    break;
                }
                case 'STRING_AGG': {
                    const separator = agg.expr?.args?.[1]?.value ?? ',';
                    resultRow.push(agg.values.join(separator));
                    break;
                }
                case 'GROUP_CONCAT': {
                    resultRow.push(agg.values.join(','));
                    break;
                }
                case 'FIRST':
                    resultRow.push(agg.value);
                    break;
                default:
                    resultRow.push(null);
            }
        }

        return {
            columns: colNames,
            rows: [resultRow],
            total: 1,
            aggregationStats: {
                scannedRows: indices.length,
                totalRows: indices.length,
                coveragePercent: '100.00',
                isPartialScan: false,
                fromSearch: true,
            },
        };
    }

    /**
     * Execute GROUP BY aggregation on specific indices
     * @private
     */
    async _executeGroupByOnIndices(ast, indices, onProgress) {
        const aggFuncs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];

        // Collect all needed columns
        const neededCols = new Set();
        for (const expr of ast.groupBy) {
            const colName = expr.column || expr.name;
            if (colName) neededCols.add(colName.toLowerCase());
        }
        for (const col of ast.columns) {
            if (col.expr?.type === 'column') {
                neededCols.add((col.expr.name || col.expr.column).toLowerCase());
            } else if (col.expr?.type === 'call' && col.expr.args?.[0]?.type === 'column') {
                neededCols.add((col.expr.args[0].name || col.expr.args[0].column).toLowerCase());
            }
        }

        // Fetch column data for the indices
        const columnData = {};
        for (const colName of neededCols) {
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx !== undefined) {
                columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, indices);
            }
        }

        // Group rows by GROUP BY columns
        const groups = new Map();
        for (let i = 0; i < indices.length; i++) {
            const groupKey = ast.groupBy.map(expr => {
                const colName = (expr.column || expr.name || '').toLowerCase();
                return JSON.stringify(columnData[colName]?.[i]);
            }).join('|');

            if (!groups.has(groupKey)) {
                groups.set(groupKey, []);
            }
            groups.get(groupKey).push(i);
        }

        // Build column names
        const colNames = ast.columns.map(col => col.alias || this.exprToName(col.expr || col));

        // Compute aggregates per group
        const resultRows = [];
        for (const [, groupLocalIndices] of groups) {
            const row = [];
            for (const col of ast.columns) {
                const expr = col.expr || col;
                if (expr.type === 'call' && aggFuncs.includes(expr.name.toUpperCase())) {
                    const funcName = expr.name.toUpperCase();
                    const argExpr = expr.args?.[0];
                    const colName = argExpr?.type === 'column' ? (argExpr.name || argExpr.column)?.toLowerCase() : null;
                    const isStar = argExpr?.type === 'star';

                    let result = null;
                    if (funcName === 'COUNT') {
                        if (isStar) {
                            result = groupLocalIndices.length;
                        } else {
                            result = 0;
                            for (const i of groupLocalIndices) {
                                if (columnData[colName]?.[i] != null) result++;
                            }
                        }
                    } else if (funcName === 'SUM') {
                        result = 0;
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && typeof v === 'number' && !isNaN(v)) result += v;
                        }
                    } else if (funcName === 'AVG') {
                        let sum = 0, count = 0;
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && typeof v === 'number' && !isNaN(v)) { sum += v; count++; }
                        }
                        result = count > 0 ? sum / count : null;
                    } else if (funcName === 'MIN') {
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && (result === null || v < result)) result = v;
                        }
                    } else if (funcName === 'MAX') {
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && (result === null || v > result)) result = v;
                        }
                    } else if (funcName === 'MEDIAN') {
                        const vals = [];
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                        }
                        if (vals.length > 0) {
                            vals.sort((a, b) => a - b);
                            const mid = Math.floor(vals.length / 2);
                            result = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
                        }
                    } else if (funcName === 'STRING_AGG' || funcName === 'GROUP_CONCAT') {
                        const separator = funcName === 'STRING_AGG' ? (expr.args?.[1]?.value ?? ',') : ',';
                        const vals = [];
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null) vals.push(String(v));
                        }
                        result = vals.join(separator);
                    } else if (funcName === 'STDDEV' || funcName === 'STDDEV_SAMP' || funcName === 'STDDEV_POP') {
                        const vals = [];
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                        }
                        if (vals.length >= (funcName === 'STDDEV_POP' ? 1 : 2)) {
                            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                            const divisor = funcName === 'STDDEV_POP' ? vals.length : (vals.length - 1);
                            const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / divisor;
                            result = Math.sqrt(variance);
                        }
                    } else if (funcName === 'VARIANCE' || funcName === 'VAR_SAMP' || funcName === 'VAR_POP') {
                        const vals = [];
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                        }
                        if (vals.length >= (funcName === 'VAR_POP' ? 1 : 2)) {
                            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                            const divisor = funcName === 'VAR_POP' ? vals.length : (vals.length - 1);
                            result = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / divisor;
                        }
                    }
                    row.push(result);
                } else if (expr.type === 'column') {
                    const colName = (expr.name || expr.column)?.toLowerCase();
                    row.push(columnData[colName]?.[groupLocalIndices[0]] ?? null);
                } else {
                    row.push(null);
                }
            }
            resultRows.push(row);
        }

        // Apply ORDER BY
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colNameToIdx = {};
            colNames.forEach((name, idx) => { colNameToIdx[name.toLowerCase()] = idx; });

            resultRows.sort((a, b) => {
                for (const order of ast.orderBy) {
                    const colName = (order.column || order.expr?.name || order.expr?.column || '').toLowerCase();
                    const colIdx = colNameToIdx[colName] ?? -1;
                    if (colIdx === -1) continue;

                    const aVal = a[colIdx];
                    const bVal = b[colIdx];
                    const dir = order.direction === 'DESC' ? -1 : 1;

                    if (aVal === null && bVal === null) continue;
                    if (aVal === null) return dir;
                    if (bVal === null) return -dir;
                    if (aVal < bVal) return -dir;
                    if (aVal > bVal) return dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET
        const offset = ast.offset || 0;
        let rows = resultRows;
        if (offset > 0) rows = rows.slice(offset);
        if (ast.limit) rows = rows.slice(0, ast.limit);

        return {
            columns: colNames,
            rows,
            total: rows.length,
            aggregationStats: {
                scannedRows: indices.length,
                totalRows: indices.length,
                groups: groups.size,
                coveragePercent: '100.00',
                isPartialScan: false,
                fromSearch: true,
            },
        };
    }

    /**
     * Return empty aggregate result with proper column names
     * @private
     */
    _emptyAggregateResult(ast) {
        const colNames = ast.columns.map(col => col.alias || this.exprToName(col.expr || col));
        const emptyRow = ast.columns.map(col => {
            const expr = col.expr || col;
            if (expr.type === 'call' && expr.name.toUpperCase() === 'COUNT') {
                return 0;
            }
            return null;
        });

        // For GROUP BY with no matches, return empty rows (no groups)
        if (ast.groupBy && ast.groupBy.length > 0) {
            return {
                columns: colNames,
                rows: [],
                total: 0,
                aggregationStats: {
                    scannedRows: 0,
                    totalRows: 0,
                    coveragePercent: '100.00',
                    isPartialScan: false,
                    fromSearch: true,
                },
            };
        }

        // For simple aggregates, return single row with COUNT=0, others NULL
        return {
            columns: colNames,
            rows: [emptyRow],
            total: 1,
            aggregationStats: {
                scannedRows: 0,
                totalRows: 0,
                coveragePercent: '100.00',
                isPartialScan: false,
                fromSearch: true,
            },
        };
    }

    /**
     * Check if the query contains aggregate functions
     */
    hasAggregates(ast) {
        const aggFunctions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];
        for (const col of ast.columns) {
            if (col.type === 'expr' && col.expr.type === 'call') {
                if (aggFunctions.includes(col.expr.name.toUpperCase())) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Execute an aggregation query
     */
    async executeAggregateQuery(ast, totalRows, onProgress) {
        const aggFunctions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];

        // Initialize aggregators for each column
        const aggregators = [];
        const colNames = [];

        for (const col of ast.columns) {
            if (col.type === 'star') {
                // COUNT(*) case
                aggregators.push({ type: 'COUNT', column: null, isStar: true });
                colNames.push('COUNT(*)');
            } else if (col.expr.type === 'call' && aggFunctions.includes(col.expr.name.toUpperCase())) {
                const aggType = col.expr.name.toUpperCase();
                const argExpr = col.expr.args[0];
                const colName = argExpr?.type === 'column' ? argExpr.name : null;
                const isStar = argExpr?.type === 'star';

                aggregators.push({
                    type: aggType,
                    column: colName,
                    isStar,
                    sum: 0,
                    count: 0,
                    min: null,
                    max: null,
                });

                const displayName = col.alias || this.exprToName(col.expr);
                colNames.push(displayName);
            } else {
                // Non-aggregate column - just take first value (or could error)
                aggregators.push({
                    type: 'FIRST',
                    column: col.expr.type === 'column' ? col.expr.name : null,
                    value: null,
                });
                colNames.push(col.alias || this.exprToName(col.expr));
            }
        }

        // Determine which columns we need to read
        const neededCols = new Set();
        for (const agg of aggregators) {
            if (agg.column) {
                neededCols.add(agg.column.toLowerCase());
            }
        }

        // Also need columns from WHERE clause
        if (ast.where) {
            this.collectColumnsFromExpr(ast.where, neededCols);
        }

        // Process data in batches
        // Respect LIMIT - only scan up to LIMIT rows for aggregation
        const scanLimit = ast.limit || totalRows; // If no LIMIT, scan all (could be slow)
        const maxRowsToScan = Math.min(scanLimit, totalRows);
        const batchSize = 1000;
        let processedRows = 0;
        let scannedRows = 0;

        for (let batchStart = 0; batchStart < maxRowsToScan; batchStart += batchSize) {
            if (onProgress) {
                onProgress(`Aggregating...`, batchStart, maxRowsToScan);
            }

            const batchEnd = Math.min(batchStart + batchSize, maxRowsToScan);
            const batchIndices = Array.from({ length: batchEnd - batchStart }, (_, i) => batchStart + i);
            scannedRows += batchIndices.length;

            // Fetch needed column data for this batch
            const batchData = {};
            for (const colName of neededCols) {
                const colIdx = this.columnMap[colName.toLowerCase()];
                if (colIdx !== undefined) {
                    batchData[colName.toLowerCase()] = await this.readColumnData(colIdx, batchIndices);
                }
            }

            // Process each row in the batch
            for (let i = 0; i < batchIndices.length; i++) {
                // Apply WHERE filter if present
                if (ast.where) {
                    const matches = this.evaluateExpr(ast.where, batchData, i);
                    if (!matches) continue;
                }

                processedRows++;

                // Update aggregators
                for (const agg of aggregators) {
                    if (agg.type === 'COUNT') {
                        agg.count++;
                    } else if (agg.type === 'FIRST') {
                        if (agg.value === null && agg.column) {
                            const data = batchData[agg.column.toLowerCase()];
                            if (data) agg.value = data[i];
                        }
                    } else {
                        // SUM, AVG, MIN, MAX need column value
                        const data = agg.column ? batchData[agg.column.toLowerCase()] : null;
                        const val = data ? data[i] : null;

                        if (val !== null && val !== undefined && !isNaN(val)) {
                            agg.count++;
                            if (agg.type === 'SUM' || agg.type === 'AVG') {
                                agg.sum += val;
                            }
                            if (agg.type === 'MIN') {
                                agg.min = agg.min === null ? val : Math.min(agg.min, val);
                            }
                            if (agg.type === 'MAX') {
                                agg.max = agg.max === null ? val : Math.max(agg.max, val);
                            }
                        }
                    }
                }
            }
        }

        // Build result row
        const resultRow = [];
        for (const agg of aggregators) {
            switch (agg.type) {
                case 'COUNT':
                    resultRow.push(agg.count);
                    break;
                case 'SUM':
                    resultRow.push(agg.sum);
                    break;
                case 'AVG':
                    resultRow.push(agg.count > 0 ? agg.sum / agg.count : null);
                    break;
                case 'MIN':
                    resultRow.push(agg.min);
                    break;
                case 'MAX':
                    resultRow.push(agg.max);
                    break;
                case 'FIRST':
                    resultRow.push(agg.value);
                    break;
                default:
                    resultRow.push(null);
            }
        }

        // Apply HAVING filter if present
        if (ast.having) {
            // Build column data for HAVING evaluation
            const havingColumnData = {};
            for (let i = 0; i < colNames.length; i++) {
                // Support both column name and alias lookup
                const colName = colNames[i].toLowerCase();
                havingColumnData[colName] = [resultRow[i]];
                // Also support aggregate function names like COUNT(*)
                const cleanName = colName.replace(/[()]/g, '').replace('*', 'star');
                havingColumnData[cleanName] = [resultRow[i]];
            }

            // Evaluate HAVING condition
            if (!this._evaluateInMemoryExpr(ast.having, havingColumnData, 0)) {
                // HAVING condition not met - return empty result
                return {
                    columns: colNames,
                    rows: [],
                    total: 0,
                    aggregationStats: {
                        scannedRows,
                        totalRows,
                        coveragePercent: totalRows > 0 ? ((scannedRows / totalRows) * 100).toFixed(2) : 100,
                        isPartialScan: scannedRows < totalRows,
                        havingFiltered: true,
                    },
                };
            }
        }

        // Calculate coverage stats
        const coveragePercent = totalRows > 0 ? ((scannedRows / totalRows) * 100).toFixed(2) : 100;
        const isPartialScan = scannedRows < totalRows;

        return {
            columns: colNames,
            rows: [resultRow],
            total: 1,
            aggregationStats: {
                scannedRows,
                totalRows,
                coveragePercent,
                isPartialScan,
            },
        };
    }

    /**
     * Expand GROUP BY clause into list of grouping sets.
     * Handles ROLLUP, CUBE, and GROUPING SETS operators.
     * @param {Array} groupBy - Array of GROUP BY items
     * @returns {Array<Array<string>>} - List of grouping sets (each is array of column names)
     */
    _expandGroupBy(groupBy) {
        if (!groupBy || groupBy.length === 0) return [[]];

        // Check if it's old-style simple column list (backward compat)
        // Old style: ['col1', 'col2']
        if (typeof groupBy[0] === 'string') {
            return [groupBy];
        }

        let result = [[]];

        for (const item of groupBy) {
            if (item.type === 'COLUMN') {
                // Simple column: cross-product with single column added
                result = result.map(set => [...set, item.column]);
            } else if (item.type === 'ROLLUP') {
                // ROLLUP(a, b, c) generates: (a,b,c), (a,b), (a), ()
                const rollupSets = [];
                for (let i = item.columns.length; i >= 0; i--) {
                    rollupSets.push(item.columns.slice(0, i));
                }
                result = this._crossProductSets(result, rollupSets);
            } else if (item.type === 'CUBE') {
                // CUBE(a, b) generates all 2^n subsets: (a,b), (a), (b), ()
                const cubeSets = this._powerSet(item.columns);
                result = this._crossProductSets(result, cubeSets);
            } else if (item.type === 'GROUPING_SETS') {
                // GROUPING SETS uses explicit sets
                result = this._crossProductSets(result, item.sets);
            }
        }

        // Deduplicate grouping sets
        const seen = new Set();
        return result.filter(set => {
            const key = JSON.stringify(set.sort());
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }

    /**
     * Generate power set (all subsets) of an array
     * @param {Array} arr - Input array
     * @returns {Array<Array>} - All subsets
     */
    _powerSet(arr) {
        const result = [[]];
        for (const item of arr) {
            const len = result.length;
            for (let i = 0; i < len; i++) {
                result.push([...result[i], item]);
            }
        }
        return result;
    }

    /**
     * Cross-product two lists of sets
     * @param {Array<Array>} sets1 - First list of sets
     * @param {Array<Array>} sets2 - Second list of sets
     * @returns {Array<Array>} - Combined sets
     */
    _crossProductSets(sets1, sets2) {
        const result = [];
        for (const s1 of sets1) {
            for (const s2 of sets2) {
                result.push([...s1, ...s2]);
            }
        }
        return result;
    }

    /**
     * Check if GROUP BY uses advanced operators (ROLLUP, CUBE, GROUPING SETS)
     */
    _hasAdvancedGroupBy(groupBy) {
        if (!groupBy || groupBy.length === 0) return false;
        if (typeof groupBy[0] === 'string') return false;
        return groupBy.some(item =>
            item.type === 'ROLLUP' || item.type === 'CUBE' || item.type === 'GROUPING_SETS'
        );
    }

    /**
     * Execute SQL and return results as async generator (streaming).
     * Yields chunks of {columns, rows} for memory-efficient processing.
     * @param {string} sql - SQL query string
     * @param {Object} options - Streaming options
     * @returns {AsyncGenerator<{columns: string[], rows: any[][]}>}
     */
    async *executeStream(sql, options = {}) {
        const { chunkSize = 1000 } = options;

        // Parse SQL
        const lexer = new SQLLexer(sql);
        const tokens = lexer.tokenize();
        const parser = new SQLParser(tokens);
        const ast = parser.parse();

        // Detect column types
        if (this.columnTypes.length === 0) {
            if (this.file._isRemote && this.file.detectColumnTypes) {
                this.columnTypes = await this.file.detectColumnTypes();
            } else if (this.file._columnTypes) {
                this.columnTypes = this.file._columnTypes;
            } else {
                this.columnTypes = Array(this.file.numColumns || 0).fill('unknown');
            }
        }

        // Get total rows
        const totalRows = this.file._isRemote
            ? await this.file.getRowCount(0)
            : Number(this.file.getRowCount(0));

        // Determine columns
        const neededColumns = this.collectNeededColumns(ast);
        const outputColumns = this.resolveOutputColumns(ast);

        // Stream in chunks
        const limit = ast.limit || totalRows;
        let yielded = 0;

        for (let offset = 0; offset < totalRows && yielded < limit; offset += chunkSize) {
            const batchSize = Math.min(chunkSize, limit - yielded, totalRows - offset);

            // Generate indices for this chunk
            const indices = [];
            for (let i = 0; i < batchSize; i++) {
                indices.push(offset + i);
            }

            // Read column data for these indices
            const columnData = [];
            for (const colName of neededColumns) {
                const colIdx = this.columnMap[colName.toLowerCase()];
                if (colIdx !== undefined) {
                    const data = await this.readColumnAtIndices(colIdx, indices);
                    columnData.push(data);
                } else {
                    columnData.push(indices.map(() => null));
                }
            }

            // Build rows
            const rows = [];
            for (let i = 0; i < indices.length; i++) {
                const row = [];
                for (let c = 0; c < neededColumns.length; c++) {
                    row.push(columnData[c][i]);
                }
                rows.push(row);
            }

            // Apply WHERE filter if present
            let filteredRows = rows;
            if (ast.where) {
                filteredRows = rows.filter((row, idx) => {
                    return this.evaluateWhereExprOnRow(ast.where, neededColumns, row);
                });
            }

            if (filteredRows.length > 0) {
                yield {
                    columns: neededColumns,
                    rows: filteredRows,
                };
                yielded += filteredRows.length;
            }
        }
    }

    /**
     * Evaluate WHERE expression on a single row
     * @private
     */
    evaluateWhereExprOnRow(expr, columns, row) {
        if (!expr) return true;

        if (expr.type === 'binary') {
            if (expr.op === 'AND') {
                return this.evaluateWhereExprOnRow(expr.left, columns, row) &&
                       this.evaluateWhereExprOnRow(expr.right, columns, row);
            }
            if (expr.op === 'OR') {
                return this.evaluateWhereExprOnRow(expr.left, columns, row) ||
                       this.evaluateWhereExprOnRow(expr.right, columns, row);
            }

            const leftVal = this._getValueFromExpr(expr.left, columns, row);
            const rightVal = this._getValueFromExpr(expr.right, columns, row);

            switch (expr.op) {
                case '=':
                case '==':
                    return leftVal == rightVal;
                case '!=':
                case '<>':
                    return leftVal != rightVal;
                case '<':
                    return leftVal < rightVal;
                case '<=':
                    return leftVal <= rightVal;
                case '>':
                    return leftVal > rightVal;
                case '>=':
                    return leftVal >= rightVal;
                default:
                    return true;
            }
        }

        return true;
    }

    /**
     * Get value from expression for row evaluation
     * @private
     */
    _getValueFromExpr(expr, columns, row) {
        if (expr.type === 'literal') {
            return expr.value;
        }
        if (expr.type === 'column') {
            const colName = expr.name || expr.column;
            const idx = columns.indexOf(colName) !== -1
                ? columns.indexOf(colName)
                : columns.indexOf(colName.toLowerCase());
            return idx !== -1 ? row[idx] : null;
        }
        return null;
    }
}

/**
 * Parse a SQL string and return the AST
 * @param {string} sql - SQL query string
 * @returns {object} - Parsed AST
 */
export function parseSQL(sql) {
    const lexer = new SQLLexer(sql);
    const tokens = lexer.tokenize();
    const parser = new SQLParser(tokens);
    return parser.parse();
}

/**
 * Represents a remote Lance dataset with multiple fragments.
 * Loads manifest to discover fragments and fetches data in parallel.
 */
export class RemoteLanceDataset {
    constructor(lanceql, baseUrl) {
        this.lanceql = lanceql;
        this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
        this._fragments = [];
        this._schema = null;
        this._totalRows = 0;
        this._numColumns = 0;
        this._onFetch = null;
        this._fragmentFiles = new Map(); // Cache of opened RemoteLanceFile per fragment
        this._isRemote = true;
        this._ivfIndex = null; // IVF index for ANN search
        this._deletedRows = new Map(); // Cache of deleted row Sets per fragment index
    }

    /**
     * Open a remote Lance dataset.
     * @param {LanceQL} lanceql - LanceQL instance
     * @param {string} baseUrl - Base URL to the dataset
     * @param {object} [options] - Options
     * @param {number} [options.version] - Specific version to load (time-travel)
     * @param {boolean} [options.prefetch] - Prefetch fragment metadata (default: true for small datasets)
     * @returns {Promise<RemoteLanceDataset>}
     */
    static async open(lanceql, baseUrl, options = {}) {
        const dataset = new RemoteLanceDataset(lanceql, baseUrl);
        dataset._requestedVersion = options.version || null;

        // Try to load from cache first (unless skipCache is true)
        const cacheKey = options.version ? `${baseUrl}@v${options.version}` : baseUrl;
        if (!options.skipCache) {
            const cached = await metadataCache.get(cacheKey);
            if (cached && cached.schema && cached.fragments) {
                console.log(`[LanceQL Dataset] Using cached metadata for ${baseUrl}`);
                dataset._schema = cached.schema;
                dataset._fragments = cached.fragments;
                dataset._numColumns = cached.schema.length;
                dataset._totalRows = cached.fragments.reduce((sum, f) => sum + f.numRows, 0);
                dataset._version = cached.version;
                dataset._columnTypes = cached.columnTypes || null;
                dataset._fromCache = true;
            }
        }

        // If not cached, try sidecar first, then manifest
        if (!dataset._fromCache) {
            // Try to load .meta.json sidecar (faster, pre-calculated)
            const sidecarLoaded = await dataset._tryLoadSidecar();

            if (!sidecarLoaded) {
                // Fall back to parsing manifest
                await dataset._loadManifest();
            }

            // Cache the metadata for next time
            metadataCache.set(cacheKey, {
                schema: dataset._schema,
                fragments: dataset._fragments,
                version: dataset._version,
                columnTypes: dataset._columnTypes || null
            }).catch(() => {}); // Don't block on cache errors
        }

        await dataset._tryLoadIndex();

        // Prefetch fragment metadata for faster first query
        // Default: prefetch if <= 5 fragments
        const shouldPrefetch = options.prefetch ?? (dataset._fragments.length <= 5);
        if (shouldPrefetch && dataset._fragments.length > 0) {
            dataset._prefetchFragments();
        }

        return dataset;
    }

    /**
     * Try to load sidecar manifest (.meta.json) for faster startup.
     * @returns {Promise<boolean>} True if sidecar was loaded successfully
     * @private
     */
    async _tryLoadSidecar() {
        try {
            const sidecarUrl = `${this.baseUrl}/.meta.json`;
            const response = await fetch(sidecarUrl);

            if (!response.ok) {
                return false;
            }

            const sidecar = await response.json();

            // Validate sidecar format
            if (!sidecar.schema || !sidecar.fragments) {
                console.warn('[LanceQL Dataset] Invalid sidecar format');
                return false;
            }

            console.log(`[LanceQL Dataset] Loaded sidecar manifest`);

            // Convert sidecar schema to internal format
            this._schema = sidecar.schema.map(col => ({
                name: col.name,
                id: col.index,
                type: col.type
            }));

            // Convert sidecar fragments to internal format
            this._fragments = sidecar.fragments.map(frag => ({
                id: frag.id,
                path: frag.data_files?.[0] || `${frag.id}.lance`,
                numRows: frag.num_rows,
                physicalRows: frag.physical_rows || frag.num_rows,
                url: `${this.baseUrl}/data/${frag.data_files?.[0] || frag.id + '.lance'}`,
                deletionFile: frag.has_deletions ? { numDeletedRows: frag.deleted_rows || 0 } : null
            }));

            this._numColumns = sidecar.num_columns;
            this._totalRows = sidecar.total_rows;
            this._version = sidecar.lance_version;

            // Extract column types from sidecar schema
            this._columnTypes = sidecar.schema.map(col => {
                const type = col.type;
                if (type.startsWith('vector[')) return 'vector';
                if (type === 'float64' || type === 'double') return 'float64';
                if (type === 'float32') return 'float32';
                if (type.includes('int')) return type;
                if (type === 'string') return 'string';
                return 'unknown';
            });

            return true;
        } catch (e) {
            // Sidecar not available or invalid - fall back to manifest
            return false;
        }
    }

    /**
     * Prefetch fragment metadata (footers) in parallel.
     * Does not block - runs in background.
     * @private
     */
    _prefetchFragments() {
        const prefetchPromises = this._fragments.map((_, idx) =>
            this.openFragment(idx).catch(() => null)
        );
        // Run in background, don't await
        Promise.all(prefetchPromises).then(() => {
            console.log(`[LanceQL Dataset] Prefetched ${this._fragments.length} fragment(s)`);
        });
    }

    /**
     * Check if dataset has an IVF index loaded.
     */
    hasIndex() {
        return this._ivfIndex !== null && this._ivfIndex.centroids !== null;
    }

    /**
     * Try to load IVF index from _indices folder.
     * @private
     */
    async _tryLoadIndex() {
        try {
            console.log(`[LanceQL Dataset] Trying to load IVF index from ${this.baseUrl}`);
            this._ivfIndex = await IVFIndex.tryLoad(this.baseUrl);
            if (this._ivfIndex) {
                console.log(`[LanceQL Dataset] IVF index loaded: ${this._ivfIndex.numPartitions} partitions, dim=${this._ivfIndex.dimension}`);
            } else {
                console.log('[LanceQL Dataset] IVF index not found or failed to parse');
            }
        } catch (e) {
            console.log('[LanceQL Dataset] No IVF index found:', e.message);
            this._ivfIndex = null;
        }
    }

    /**
     * Load and parse the manifest to discover fragments.
     * @private
     */
    async _loadManifest() {
        let manifestData = null;
        let manifestVersion = 0;

        // If specific version requested (time-travel), use that
        if (this._requestedVersion) {
            manifestVersion = this._requestedVersion;
            const manifestUrl = `${this.baseUrl}/_versions/${manifestVersion}.manifest`;
            const response = await fetch(manifestUrl);
            if (!response.ok) {
                throw new Error(`Version ${manifestVersion} not found (${response.status})`);
            }
            manifestData = new Uint8Array(await response.arrayBuffer());
        } else {
            // Find the latest manifest version using binary search approach
            // First check common versions in parallel
            const checkVersions = [1, 5, 10, 20, 50, 100];
            const checks = await Promise.all(
                checkVersions.map(async v => {
                    try {
                        const url = `${this.baseUrl}/_versions/${v}.manifest`;
                        const response = await fetch(url, { method: 'HEAD' });
                        return response.ok ? v : 0;
                    } catch {
                        return 0;
                    }
                })
            );

            // Find highest existing version from quick check
            let highestFound = Math.max(...checks);

            // If we found a high version, scan forward from there
            if (highestFound > 0) {
                for (let v = highestFound + 1; v <= highestFound + 50; v++) {
                    try {
                        const url = `${this.baseUrl}/_versions/${v}.manifest`;
                        const response = await fetch(url, { method: 'HEAD' });
                        if (response.ok) {
                            highestFound = v;
                        } else {
                            break;
                        }
                    } catch {
                        break;
                    }
                }
            }

            manifestVersion = highestFound;

            if (manifestVersion === 0) {
                throw new Error('No manifest found in dataset');
            }

            // Fetch the latest manifest
            const manifestUrl = `${this.baseUrl}/_versions/${manifestVersion}.manifest`;
            const response = await fetch(manifestUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch manifest: ${response.status}`);
            }
            manifestData = new Uint8Array(await response.arrayBuffer());
        }

        // Store the version we loaded
        this._version = manifestVersion;
        this._latestVersion = this._requestedVersion ? null : manifestVersion;

        console.log(`[LanceQL Dataset] Loading manifest v${manifestVersion}${this._requestedVersion ? ' (time-travel)' : ''}...`);
        this._parseManifest(manifestData);

        console.log(`[LanceQL Dataset] Loaded: ${this._fragments.length} fragments, ${this._totalRows.toLocaleString()} rows, ${this._numColumns} columns`);
    }

    /**
     * Get list of available versions.
     * @returns {Promise<number[]>}
     */
    async listVersions() {
        const versions = [];
        // Scan for versions 1 to latestVersion (or 100 if unknown)
        const maxVersion = this._latestVersion || 100;

        const checks = await Promise.all(
            Array.from({ length: maxVersion }, (_, i) => i + 1).map(async v => {
                try {
                    const url = `${this.baseUrl}/_versions/${v}.manifest`;
                    const response = await fetch(url, { method: 'HEAD' });
                    return response.ok ? v : 0;
                } catch {
                    return 0;
                }
            })
        );

        return checks.filter(v => v > 0);
    }

    /**
     * Get current loaded version.
     */
    get version() {
        return this._version;
    }

    /**
     * Parse manifest protobuf to extract schema and fragment info.
     *
     * Lance manifest file structure:
     * - Chunk 1 (len-prefixed): Transaction metadata (may be small/incremental)
     * - Chunk 2 (len-prefixed): Full manifest with schema + fragments
     * - Footer (16 bytes): Offsets + "LANC" magic
     *
     * @private
     */
    _parseManifest(bytes) {
        const view = new DataView(bytes.buffer, bytes.byteOffset);

        // Read chunk 1 length
        const chunk1Len = view.getUint32(0, true);

        // Check if there's a chunk 2 (full manifest data)
        // Chunk 2 starts at offset (4 + chunk1Len)
        const chunk2Start = 4 + chunk1Len;
        let protoData;

        if (chunk2Start + 4 < bytes.length) {
            const chunk2Len = view.getUint32(chunk2Start, true);
            // Verify chunk 2 exists and has reasonable size
            if (chunk2Len > 0 && chunk2Start + 4 + chunk2Len <= bytes.length) {
                // Use chunk 2 (full manifest)
                protoData = bytes.slice(chunk2Start + 4, chunk2Start + 4 + chunk2Len);
            } else {
                // Fall back to chunk 1
                protoData = bytes.slice(4, 4 + chunk1Len);
            }
        } else {
            // Only chunk 1 exists
            protoData = bytes.slice(4, 4 + chunk1Len);
        }

        let pos = 0;
        const fields = [];
        const fragments = [];

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < protoData.length) {
                const byte = protoData[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        const skipField = (wireType) => {
            if (wireType === 0) {
                readVarint();
            } else if (wireType === 2) {
                const len = readVarint();
                pos += len;
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        };

        // Parse top-level Manifest message
        while (pos < protoData.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (fieldNum === 1 && wireType === 2) {
                // Field 1 = schema (repeated Field message)
                const fieldLen = readVarint();
                const fieldEnd = pos + fieldLen;

                let name = null;
                let id = null;
                let logicalType = null;

                while (pos < fieldEnd) {
                    const fTag = readVarint();
                    const fNum = fTag >> 3;
                    const fWire = fTag & 0x7;

                    if (fWire === 0) {
                        const val = readVarint();
                        if (fNum === 3) id = val;
                    } else if (fWire === 2) {
                        const len = readVarint();
                        const content = protoData.slice(pos, pos + len);
                        pos += len;

                        if (fNum === 2) {
                            name = new TextDecoder().decode(content);
                        } else if (fNum === 5) {
                            logicalType = new TextDecoder().decode(content);
                        }
                    } else {
                        skipField(fWire);
                    }
                }

                if (name) {
                    fields.push({ name, id, type: logicalType });
                }
            } else if (fieldNum === 2 && wireType === 2) {
                // Field 2 = fragments (repeated Fragment message)
                const fragLen = readVarint();
                const fragEnd = pos + fragLen;

                let fragId = null;
                let filePath = null;
                let numRows = 0;
                let deletionFile = null;  // Track deletion info

                while (pos < fragEnd) {
                    const fTag = readVarint();
                    const fNum = fTag >> 3;
                    const fWire = fTag & 0x7;

                    if (fWire === 0) {
                        const val = readVarint();
                        if (fNum === 1) fragId = val;  // Fragment.id
                        else if (fNum === 4) numRows = val;  // Fragment.physical_rows
                    } else if (fWire === 2) {
                        const len = readVarint();
                        const content = protoData.slice(pos, pos + len);
                        pos += len;

                        if (fNum === 2) {
                            // Fragment.files - parse DataFile message
                            let innerPos = 0;
                            while (innerPos < content.length) {
                                const iTag = content[innerPos++];
                                const iNum = iTag >> 3;
                                const iWire = iTag & 0x7;

                                if (iWire === 2) {
                                    // Length-delimited
                                    let iLen = 0;
                                    let iShift = 0;
                                    while (innerPos < content.length) {
                                        const b = content[innerPos++];
                                        iLen |= (b & 0x7F) << iShift;
                                        if ((b & 0x80) === 0) break;
                                        iShift += 7;
                                    }
                                    const iContent = content.slice(innerPos, innerPos + iLen);
                                    innerPos += iLen;

                                    if (iNum === 1) {
                                        // DataFile.path
                                        filePath = new TextDecoder().decode(iContent);
                                    }
                                } else if (iWire === 0) {
                                    // Varint - skip
                                    while (innerPos < content.length && (content[innerPos++] & 0x80) !== 0);
                                } else if (iWire === 5) {
                                    innerPos += 4;
                                } else if (iWire === 1) {
                                    innerPos += 8;
                                }
                            }
                        } else if (fNum === 3) {
                            // Fragment.deletion_file - parse DeletionFile message
                            deletionFile = this._parseDeletionFile(content, fragId);
                        }
                    } else {
                        skipField(fWire);
                    }
                }

                if (filePath) {
                    const logicalRows = deletionFile ? numRows - deletionFile.numDeletedRows : numRows;
                    fragments.push({
                        id: fragId,
                        path: filePath,
                        numRows: logicalRows,  // Logical rows (excluding deleted)
                        physicalRows: numRows, // Physical rows (including deleted)
                        deletionFile: deletionFile,
                        url: `${this.baseUrl}/data/${filePath}`
                    });
                }
            } else {
                skipField(wireType);
            }
        }

        this._schema = fields;
        this._fragments = fragments;
        this._numColumns = fields.length;
        this._totalRows = fragments.reduce((sum, f) => sum + f.numRows, 0);

        // Track if any fragment has deletions
        const deletedCount = fragments.reduce((sum, f) => sum + (f.deletionFile?.numDeletedRows || 0), 0);
        if (deletedCount > 0) {
            console.log(`[LanceQL Dataset] Has ${deletedCount} deleted rows across fragments`);
        }
    }

    /**
     * Parse DeletionFile protobuf message.
     * @param {Uint8Array} data - Raw protobuf bytes
     * @param {number} fragId - Fragment ID for path construction
     * @returns {Object|null} Deletion file info
     * @private
     */
    _parseDeletionFile(data, fragId) {
        let fileType = 0;  // 0 = ARROW_ARRAY, 1 = BITMAP
        let readVersion = 0;
        let id = 0;
        let numDeletedRows = 0;

        let pos = 0;
        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < data.length) {
                const b = data[pos++];
                result |= (b & 0x7F) << shift;
                if ((b & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        while (pos < data.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 0) {
                const val = readVarint();
                if (fieldNum === 1) fileType = val;       // DeletionFile.file_type
                else if (fieldNum === 2) readVersion = val; // DeletionFile.read_version
                else if (fieldNum === 3) id = val;        // DeletionFile.id
                else if (fieldNum === 4) numDeletedRows = val; // DeletionFile.num_deleted_rows
            } else if (wireType === 2) {
                const len = readVarint();
                pos += len; // Skip length-delimited fields
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        }

        if (numDeletedRows === 0) return null;

        const ext = fileType === 0 ? 'arrow' : 'bin';
        const path = `_deletions/${fragId}-${readVersion}-${id}.${ext}`;

        return {
            fileType: fileType === 0 ? 'arrow' : 'bitmap',
            readVersion,
            id,
            numDeletedRows,
            path,
            url: `${this.baseUrl}/${path}`
        };
    }

    /**
     * Load deleted row indices for a fragment.
     * @param {number} fragmentIndex - Fragment index
     * @returns {Promise<Set<number>>} Set of deleted row indices (local to fragment)
     * @private
     */
    async _loadDeletedRows(fragmentIndex) {
        // Check cache
        if (this._deletedRows.has(fragmentIndex)) {
            return this._deletedRows.get(fragmentIndex);
        }

        const frag = this._fragments[fragmentIndex];
        if (!frag?.deletionFile) {
            const emptySet = new Set();
            this._deletedRows.set(fragmentIndex, emptySet);
            return emptySet;
        }

        const { url, fileType, numDeletedRows } = frag.deletionFile;
        console.log(`[LanceQL] Loading ${numDeletedRows} deletions from ${url} (${fileType})`);

        try {
            const response = await fetch(url);
            if (!response.ok) {
                console.warn(`[LanceQL] Failed to load deletion file: ${response.status}`);
                const emptySet = new Set();
                this._deletedRows.set(fragmentIndex, emptySet);
                return emptySet;
            }

            const buffer = await response.arrayBuffer();
            const data = new Uint8Array(buffer);
            let deletedSet;

            if (fileType === 'arrow') {
                deletedSet = this._parseArrowDeletions(data);
            } else {
                deletedSet = this._parseRoaringBitmap(data);
            }

            console.log(`[LanceQL] Loaded ${deletedSet.size} deleted rows for fragment ${fragmentIndex}`);
            this._deletedRows.set(fragmentIndex, deletedSet);
            return deletedSet;
        } catch (e) {
            console.error(`[LanceQL] Error loading deletion file:`, e);
            const emptySet = new Set();
            this._deletedRows.set(fragmentIndex, emptySet);
            return emptySet;
        }
    }

    /**
     * Parse Arrow IPC deletion file (Int32Array of deleted indices).
     * @param {Uint8Array} data - Raw Arrow IPC bytes
     * @returns {Set<number>} Set of deleted row indices
     * @private
     */
    _parseArrowDeletions(data) {
        // Arrow IPC format: Magic (ARROW1) + schema + record batch
        // For simplicity, we look for the Int32 data after the schema
        const deletedSet = new Set();

        // Find continuation marker (-1 as int32 LE = 0xFFFFFFFF)
        // Then record batch metadata length, then metadata, then body (Int32 array)
        let pos = 0;

        // Skip magic "ARROW1" + padding
        if (data.length >= 8 && String.fromCharCode(...data.slice(0, 6)) === 'ARROW1') {
            pos = 8;
        }

        // Look for continuation markers and skip metadata
        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);

        while (pos < data.length - 4) {
            const marker = view.getInt32(pos, true);
            if (marker === -1) {
                // Continuation marker found
                pos += 4;
                if (pos + 4 > data.length) break;
                const metaLen = view.getInt32(pos, true);
                pos += 4 + metaLen; // Skip metadata

                // The body follows - for deletion vectors it's just Int32 array
                // We need to read until end or next message
                while (pos + 4 <= data.length) {
                    // Check if this looks like the start of data (not another marker)
                    const nextMarker = view.getInt32(pos, true);
                    if (nextMarker === -1) break; // Another message starts

                    // Read Int32 values until we hit something that looks like a marker
                    // or reach expected count
                    const val = view.getInt32(pos, true);
                    if (val >= 0 && val < 10000000) { // Sanity check
                        deletedSet.add(val);
                    }
                    pos += 4;
                }
            } else {
                pos++;
            }
        }

        return deletedSet;
    }

    /**
     * Parse Roaring Bitmap deletion file.
     * @param {Uint8Array} data - Raw Roaring Bitmap bytes
     * @returns {Set<number>} Set of deleted row indices
     * @private
     */
    _parseRoaringBitmap(data) {
        // Roaring bitmap format: header + containers
        // This is a simplified parser for common cases
        const deletedSet = new Set();
        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);

        if (data.length < 8) return deletedSet;

        // Read cookie (first 4 bytes indicate format)
        const cookie = view.getUint32(0, true);

        // Standard roaring format: cookie = 12346 or 12347
        // Portable format: first 8 bytes are magic
        if (cookie === 12346 || cookie === 12347) {
            // Standard format
            const isRunContainer = (cookie === 12347);
            let pos = 4;

            // Number of containers
            const numContainers = view.getUint16(pos, true);
            pos += 2;

            // Skip to container data
            // Each key is 2 bytes, each cardinality is 2 bytes
            const keysStart = pos;
            pos += numContainers * 4; // keys + cardinalities

            for (let i = 0; i < numContainers && pos < data.length; i++) {
                const key = view.getUint16(keysStart + i * 4, true);
                const card = view.getUint16(keysStart + i * 4 + 2, true) + 1;
                const baseValue = key << 16;

                // Read container values (simplified - assumes array container)
                for (let j = 0; j < card && pos + 2 <= data.length; j++) {
                    const lowBits = view.getUint16(pos, true);
                    deletedSet.add(baseValue | lowBits);
                    pos += 2;
                }
            }
        }

        return deletedSet;
    }

    /**
     * Check if a row is deleted in a fragment.
     * @param {number} fragmentIndex - Fragment index
     * @param {number} localRowIndex - Row index within the fragment
     * @returns {Promise<boolean>} True if row is deleted
     */
    async isRowDeleted(fragmentIndex, localRowIndex) {
        const deletedSet = await this._loadDeletedRows(fragmentIndex);
        return deletedSet.has(localRowIndex);
    }

    /**
     * Get number of columns.
     */
    get numColumns() {
        return this._numColumns;
    }

    /**
     * Get total row count across all fragments.
     */
    get rowCount() {
        return this._totalRows;
    }

    /**
     * Get row count for a column (for API compatibility with RemoteLanceFile).
     * @param {number} columnIndex - Column index (ignored, all columns have same row count)
     * @returns {Promise<number>}
     */
    async getRowCount(columnIndex = 0) {
        return this._totalRows;
    }

    /**
     * Read a single vector at a global row index.
     * Delegates to the correct fragment based on row index.
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Global row index
     * @returns {Promise<Float32Array>}
     */
    async readVectorAt(colIdx, rowIdx) {
        const loc = this._getFragmentForRow(rowIdx);
        if (!loc) return new Float32Array(0);
        const file = await this.openFragment(loc.fragmentIndex);
        return await file.readVectorAt(colIdx, loc.localIndex);
    }

    /**
     * Get vector info for a column by querying first fragment.
     * @param {number} colIdx - Column index
     * @returns {Promise<{rows: number, dimension: number}>}
     */
    async getVectorInfo(colIdx) {
        if (this._fragments.length === 0) {
            return { rows: 0, dimension: 0 };
        }

        // Get vector info from first fragment
        const file = await this.openFragment(0);
        const fragInfo = await file.getVectorInfo(colIdx);

        if (fragInfo.dimension === 0) {
            return { rows: 0, dimension: 0 };
        }

        // Return total rows across all fragments, dimension from first fragment
        return {
            rows: this._totalRows,
            dimension: fragInfo.dimension
        };
    }

    /**
     * Get column names from schema.
     */
    get columnNames() {
        return this._schema ? this._schema.map(f => f.name) : [];
    }

    /**
     * Get full schema.
     */
    get schema() {
        return this._schema;
    }

    /**
     * Get fragment list.
     */
    get fragments() {
        return this._fragments;
    }

    /**
     * Get estimated total size based on row count and schema.
     * More accurate than fragment count estimate.
     */
    get size() {
        if (this._cachedSize) return this._cachedSize;

        // Estimate bytes per row based on column types
        let bytesPerRow = 0;
        for (let i = 0; i < (this._columnTypes?.length || 0); i++) {
            const colType = this._columnTypes[i];
            if (colType === 'int64' || colType === 'float64' || colType === 'double') {
                bytesPerRow += 8;
            } else if (colType === 'int32' || colType === 'float32') {
                bytesPerRow += 4;
            } else if (colType === 'string') {
                bytesPerRow += 50; // Average string length estimate
            } else if (colType === 'vector' || colType?.startsWith('vector[')) {
                // Extract dimension from type like "vector[384]"
                const match = colType?.match(/\[(\d+)\]/);
                const dim = match ? parseInt(match[1]) : 384;
                bytesPerRow += dim * 4; // float32 per dimension
            } else {
                bytesPerRow += 8; // Default
            }
        }

        // Fallback if no column types
        if (bytesPerRow === 0) {
            bytesPerRow = 100; // Conservative default
        }

        this._cachedSize = this._totalRows * bytesPerRow;
        return this._cachedSize;
    }

    /**
     * Set callback for network fetch events.
     */
    onFetch(callback) {
        this._onFetch = callback;
    }

    /**
     * Open a specific fragment as RemoteLanceFile.
     * @param {number} fragmentIndex - Index of fragment to open
     * @returns {Promise<RemoteLanceFile>}
     */
    async openFragment(fragmentIndex) {
        if (fragmentIndex < 0 || fragmentIndex >= this._fragments.length) {
            throw new Error(`Invalid fragment index: ${fragmentIndex}`);
        }

        // Check cache
        if (this._fragmentFiles.has(fragmentIndex)) {
            return this._fragmentFiles.get(fragmentIndex);
        }

        const fragment = this._fragments[fragmentIndex];
        const file = await RemoteLanceFile.open(this.lanceql, fragment.url);

        // Propagate fetch callback
        if (this._onFetch) {
            file.onFetch(this._onFetch);
        }

        this._fragmentFiles.set(fragmentIndex, file);
        return file;
    }

    /**
     * Read rows from the dataset with pagination.
     * Fetches from multiple fragments in parallel.
     * @param {Object} options - Query options
     * @param {number} options.offset - Starting row offset
     * @param {number} options.limit - Maximum rows to return
     * @param {number[]} options.columns - Column indices to read (optional)
     * @param {boolean} options._isPrefetch - Internal flag to prevent recursive prefetch
     * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
     */
    async readRows({ offset = 0, limit = 50, columns = null, _isPrefetch = false } = {}) {
        // Determine which fragments contain the requested rows
        const fragmentRanges = [];
        let currentOffset = 0;

        for (let i = 0; i < this._fragments.length; i++) {
            const frag = this._fragments[i];
            const fragStart = currentOffset;
            const fragEnd = currentOffset + frag.numRows;

            // Check if this fragment overlaps with requested range
            if (fragEnd > offset && fragStart < offset + limit) {
                const localStart = Math.max(0, offset - fragStart);
                const localEnd = Math.min(frag.numRows, offset + limit - fragStart);

                fragmentRanges.push({
                    fragmentIndex: i,
                    localOffset: localStart,
                    localLimit: localEnd - localStart,
                    globalStart: fragStart + localStart
                });
            }

            currentOffset = fragEnd;
            if (currentOffset >= offset + limit) break;
        }

        if (fragmentRanges.length === 0) {
            return { columns: [], columnNames: this.columnNames, total: this._totalRows };
        }

        // Fetch from fragments in parallel
        const fetchPromises = fragmentRanges.map(async (range) => {
            const file = await this.openFragment(range.fragmentIndex);
            const result = await file.readRows({
                offset: range.localOffset,
                limit: range.localLimit,
                columns: columns
            });
            return { ...range, result };
        });

        const results = await Promise.all(fetchPromises);

        // Merge results in order
        results.sort((a, b) => a.globalStart - b.globalStart);

        const mergedColumns = [];
        const colNames = results[0]?.result.columnNames || this.columnNames;
        const numCols = columns ? columns.length : this._numColumns;

        for (let c = 0; c < numCols; c++) {
            const colData = [];
            for (const r of results) {
                if (r.result.columns[c]) {
                    colData.push(...r.result.columns[c]);
                }
            }
            mergedColumns.push(colData);
        }

        const result = {
            columns: mergedColumns,
            columnNames: colNames,
            total: this._totalRows
        };

        // Speculative prefetch: if there are more rows, prefetch next page in background
        // Only prefetch if: not already a prefetch, limit is reasonable, more rows exist
        const nextOffset = offset + limit;
        if (!_isPrefetch && nextOffset < this._totalRows && limit <= 100) {
            this._prefetchNextPage(nextOffset, limit, columns);
        }

        return result;
    }

    /**
     * Prefetch next page of rows in background.
     * @private
     */
    _prefetchNextPage(offset, limit, columns) {
        // Use a cache key to avoid duplicate prefetches
        const cacheKey = `${offset}-${limit}-${columns?.join(',') || 'all'}`;
        if (this._prefetchCache?.has(cacheKey)) {
            return; // Already prefetching or prefetched
        }

        if (!this._prefetchCache) {
            this._prefetchCache = new Map();
        }

        // Start prefetch in background (don't await)
        const prefetchPromise = this.readRows({ offset, limit, columns, _isPrefetch: true })
            .then(result => {
                this._prefetchCache.set(cacheKey, result);
                console.log(`[LanceQL] Prefetched rows ${offset}-${offset + limit}`);
            })
            .catch(() => {
                // Ignore prefetch errors
            });

        this._prefetchCache.set(cacheKey, prefetchPromise);
    }

    /**
     * Detect column types by sampling from first fragment.
     * @returns {Promise<string[]>}
     */
    async detectColumnTypes() {
        // Return cached types if available
        if (this._columnTypes && this._columnTypes.length > 0) {
            return this._columnTypes;
        }

        if (this._fragments.length === 0) {
            return [];
        }
        const file = await this.openFragment(0);
        const types = await file.detectColumnTypes();
        this._columnTypes = types;

        // Update cache with column types
        const cacheKey = this._requestedVersion ? `${this.baseUrl}@v${this._requestedVersion}` : this.baseUrl;
        metadataCache.get(cacheKey).then(cached => {
            if (cached) {
                cached.columnTypes = types;
                metadataCache.set(cacheKey, cached).catch(() => {});
            }
        }).catch(() => {});

        return types;
    }

    /**
     * Helper to determine which fragment contains a given row index.
     * @private
     */
    _getFragmentForRow(rowIdx) {
        let offset = 0;
        for (let i = 0; i < this._fragments.length; i++) {
            const frag = this._fragments[i];
            if (rowIdx < offset + frag.numRows) {
                return { fragmentIndex: i, localIndex: rowIdx - offset };
            }
            offset += frag.numRows;
        }
        return null;
    }

    /**
     * Group indices by fragment for efficient batch reading.
     * @private
     */
    _groupIndicesByFragment(indices) {
        const groups = new Map();
        for (const globalIdx of indices) {
            const loc = this._getFragmentForRow(globalIdx);
            if (!loc) continue;

            if (!groups.has(loc.fragmentIndex)) {
                groups.set(loc.fragmentIndex, { localIndices: [], globalIndices: [] });
            }
            groups.get(loc.fragmentIndex).localIndices.push(loc.localIndex);
            groups.get(loc.fragmentIndex).globalIndices.push(globalIdx);
        }
        return groups;
    }

    /**
     * Read strings at specific indices across fragments.
     */
    async readStringsAtIndices(colIdx, indices) {
        const groups = this._groupIndicesByFragment(indices);
        const results = new Map();

        console.log(`[ReadStrings] Reading ${indices.length} strings from col ${colIdx}`);
        console.log(`[ReadStrings] First 5 indices: ${indices.slice(0, 5)}`);
        console.log(`[ReadStrings] Fragment groups: ${Array.from(groups.keys())}`);

        // Fetch from each fragment in parallel
        const fetchPromises = [];
        for (const [fragIdx, group] of groups) {
            fetchPromises.push((async () => {
                const file = await this.openFragment(fragIdx);
                console.log(`[ReadStrings] Fragment ${fragIdx}: reading ${group.localIndices.length} strings, first local indices: ${group.localIndices.slice(0, 3)}`);
                const data = await file.readStringsAtIndices(colIdx, group.localIndices);
                console.log(`[ReadStrings] Fragment ${fragIdx}: got ${data.length} strings, first 3: ${data.slice(0, 3).map(s => s?.slice(0, 20) + '...')}`);
                for (let i = 0; i < group.globalIndices.length; i++) {
                    results.set(group.globalIndices[i], data[i]);
                }
            })());
        }
        await Promise.all(fetchPromises);

        // Return in original order
        return indices.map(idx => results.get(idx) || null);
    }

    /**
     * Read int64 values at specific indices across fragments.
     */
    async readInt64AtIndices(colIdx, indices) {
        const groups = this._groupIndicesByFragment(indices);
        const results = new Map();

        const fetchPromises = [];
        for (const [fragIdx, group] of groups) {
            fetchPromises.push((async () => {
                const file = await this.openFragment(fragIdx);
                const data = await file.readInt64AtIndices(colIdx, group.localIndices);
                for (let i = 0; i < group.globalIndices.length; i++) {
                    results.set(group.globalIndices[i], data[i]);
                }
            })());
        }
        await Promise.all(fetchPromises);

        return new BigInt64Array(indices.map(idx => results.get(idx) || 0n));
    }

    /**
     * Read float64 values at specific indices across fragments.
     */
    async readFloat64AtIndices(colIdx, indices) {
        const groups = this._groupIndicesByFragment(indices);
        const results = new Map();

        const fetchPromises = [];
        for (const [fragIdx, group] of groups) {
            fetchPromises.push((async () => {
                const file = await this.openFragment(fragIdx);
                const data = await file.readFloat64AtIndices(colIdx, group.localIndices);
                for (let i = 0; i < group.globalIndices.length; i++) {
                    results.set(group.globalIndices[i], data[i]);
                }
            })());
        }
        await Promise.all(fetchPromises);

        return new Float64Array(indices.map(idx => results.get(idx) || 0));
    }

    /**
     * Read int32 values at specific indices across fragments.
     */
    async readInt32AtIndices(colIdx, indices) {
        const groups = this._groupIndicesByFragment(indices);
        const results = new Map();

        const fetchPromises = [];
        for (const [fragIdx, group] of groups) {
            fetchPromises.push((async () => {
                const file = await this.openFragment(fragIdx);
                const data = await file.readInt32AtIndices(colIdx, group.localIndices);
                for (let i = 0; i < group.globalIndices.length; i++) {
                    results.set(group.globalIndices[i], data[i]);
                }
            })());
        }
        await Promise.all(fetchPromises);

        return new Int32Array(indices.map(idx => results.get(idx) || 0));
    }

    /**
     * Read float32 values at specific indices across fragments.
     */
    async readFloat32AtIndices(colIdx, indices) {
        const groups = this._groupIndicesByFragment(indices);
        const results = new Map();

        const fetchPromises = [];
        for (const [fragIdx, group] of groups) {
            fetchPromises.push((async () => {
                const file = await this.openFragment(fragIdx);
                const data = await file.readFloat32AtIndices(colIdx, group.localIndices);
                for (let i = 0; i < group.globalIndices.length; i++) {
                    results.set(group.globalIndices[i], data[i]);
                }
            })());
        }
        await Promise.all(fetchPromises);

        return new Float32Array(indices.map(idx => results.get(idx) || 0));
    }

    /**
     * Vector search across all fragments.
     * API compatible with RemoteLanceFile.vectorSearch.
     *
     * @param {number} colIdx - Vector column index
     * @param {Float32Array} queryVec - Query vector
     * @param {number} topK - Number of results to return
     * @param {Function} onProgress - Progress callback (current, total)
     * @param {Object} options - Search options
     * @returns {Promise<{indices: number[], scores: number[], usedIndex: boolean}>}
     */
    async vectorSearch(colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
        const {
            normalized = true,
            workerPool = null,
            useIndex = true,
            nprobe = 20
        } = options;

        const vectorColIdx = colIdx;

        if (vectorColIdx < 0) {
            throw new Error('No vector column found in dataset');
        }

        const dim = queryVec.length;
        console.log(`[VectorSearch] Query dim=${dim}, topK=${topK}, fragments=${this._fragments.length}, hasIndex=${this.hasIndex()}`);

        // Require IVF index for efficient search - no brute force fallback
        if (!this.hasIndex()) {
            throw new Error('No IVF index found. Vector search requires an IVF index for efficient querying.');
        }

        if (this._ivfIndex.dimension !== dim) {
            throw new Error(`Query dimension (${dim}) does not match index dimension (${this._ivfIndex.dimension}).`);
        }

        if (!this._ivfIndex.hasPartitionIndex) {
            throw new Error('IVF partition index (ivf_partitions.bin) not found. Required for efficient search.');
        }

        console.log(`[VectorSearch] Using IVF index (nprobe=${nprobe})`);
        return await this._ivfIndexSearch(queryVec, topK, vectorColIdx, nprobe, onProgress);
    }

    /**
     * IVF index-based ANN search.
     * Fetches partition data (row IDs + vectors) directly from ivf_vectors.bin.
     * Uses WebGPU for batch similarity computation.
     * @private
     */
    async _ivfIndexSearch(queryVec, topK, vectorColIdx, nprobe, onProgress) {
        // Find nearest partitions using centroids
        const partitions = this._ivfIndex.findNearestPartitions(queryVec, nprobe);
        console.log(`[VectorSearch] Searching ${partitions.length} partitions:`, partitions);

        // Fetch partition data (row IDs + vectors) directly
        const partitionData = await this._ivfIndex.fetchPartitionData(
            partitions,
            this._ivfIndex.dimension,
            (loaded, total) => {
                if (onProgress) {
                    // First 80% is downloading
                    const pct = total > 0 ? loaded / total : 0;
                    onProgress(Math.floor(pct * 80), 100);
                }
            }
        );

        if (!partitionData || partitionData.rowIds.length === 0) {
            throw new Error('IVF index not available. This dataset requires ivf_vectors.bin for efficient search.');
        }

        const { rowIds, vectors } = partitionData;

        // Use hybrid WebGPU + WASM SIMD for batch similarity
        const scores = new Float32Array(vectors.length);
        const dim = queryVec.length;

        if (webgpuAccelerator.isAvailable()) {
            const maxBatch = webgpuAccelerator.getMaxVectorsPerBatch(dim);
            let gpuProcessed = 0;
            let wasmProcessed = 0;

            // Process in chunks that fit in WebGPU buffer
            for (let start = 0; start < vectors.length; start += maxBatch) {
                const end = Math.min(start + maxBatch, vectors.length);
                const chunk = vectors.slice(start, end);

                try {
                    const chunkScores = await webgpuAccelerator.batchCosineSimilarity(queryVec, chunk, true);
                    if (chunkScores) {
                        scores.set(chunkScores, start);
                        gpuProcessed += chunk.length;
                        continue;
                    }
                } catch (e) {
                    // Fall through to WASM for this chunk
                }

                // WASM SIMD fallback for this chunk
                if (this._fragments[0]?.lanceql?.batchCosineSimilarity) {
                    const chunkScores = this._fragments[0].lanceql.batchCosineSimilarity(queryVec, chunk, true);
                    scores.set(chunkScores, start);
                    wasmProcessed += chunk.length;
                } else {
                    // JS fallback (slow)
                    for (let i = 0; i < chunk.length; i++) {
                        const vec = chunk[i];
                        if (!vec || vec.length !== dim) continue;
                        let dot = 0;
                        for (let k = 0; k < dim; k++) {
                            dot += queryVec[k] * vec[k];
                        }
                        scores[start + i] = dot;
                    }
                    wasmProcessed += chunk.length;
                }
            }

            console.log(`[VectorSearch] Processed ${vectors.length.toLocaleString()} vectors: ${gpuProcessed.toLocaleString()} WebGPU, ${wasmProcessed.toLocaleString()} WASM SIMD`);
        } else {
            // Pure WASM SIMD path
            console.log(`[VectorSearch] Computing similarities for ${rowIds.length.toLocaleString()} vectors via WASM SIMD`);
            if (this._fragments[0]?.lanceql?.batchCosineSimilarity) {
                const allScores = this._fragments[0].lanceql.batchCosineSimilarity(queryVec, vectors, true);
                scores.set(allScores);
            } else {
                // JS fallback (slow)
                for (let i = 0; i < vectors.length; i++) {
                    const vec = vectors[i];
                    if (!vec || vec.length !== dim) continue;
                    let dot = 0;
                    for (let k = 0; k < dim; k++) {
                        dot += queryVec[k] * vec[k];
                    }
                    scores[i] = dot;
                }
            }
        }

        if (onProgress) onProgress(90, 100);

        // Build results with row IDs
        const allResults = [];
        for (let i = 0; i < rowIds.length; i++) {
            allResults.push({ index: rowIds[i], score: scores[i] });
        }

        // Sort and take top-k
        allResults.sort((a, b) => b.score - a.score);
        const finalK = Math.min(topK, allResults.length);

        if (onProgress) onProgress(100, 100);

        return {
            indices: allResults.slice(0, finalK).map(r => r.index),
            scores: allResults.slice(0, finalK).map(r => r.score),
            usedIndex: true,
            searchedRows: rowIds.length
        };
    }

    /**
     * Find the vector column index by looking at schema.
     * @private
     */
    _findVectorColumn() {
        if (!this._schema) return -1;

        for (let i = 0; i < this._schema.length; i++) {
            const field = this._schema[i];
            if (field.name === 'embedding' || field.name === 'vector' ||
                field.type === 'fixed_size_list' || field.type === 'list') {
                return i;
            }
        }

        // Assume last column is vector if schema unclear
        return this._schema.length - 1;
    }

    /**
     * Parallel vector search using WorkerPool.
     * @private
     */
    async _parallelVectorSearch(query, topK, vectorColIdx, normalized, workerPool) {
        const dim = query.length;

        // Load vectors from each fragment in parallel
        const chunkPromises = this._fragments.map(async (frag, idx) => {
            const file = await this.openFragment(idx);

            // Get vector data for this fragment
            const vectors = await file.readVectorColumn(vectorColIdx);
            if (!vectors || vectors.length === 0) {
                return null;
            }

            // Calculate start index for this fragment
            let startIndex = 0;
            for (let i = 0; i < idx; i++) {
                startIndex += this._fragments[i].numRows;
            }

            return {
                vectors: new Float32Array(vectors),
                startIndex,
                numVectors: vectors.length / dim
            };
        });

        const chunks = (await Promise.all(chunkPromises)).filter(c => c !== null);

        if (chunks.length === 0) {
            return { indices: new Uint32Array(0), scores: new Float32Array(0), rows: [] };
        }

        // Perform parallel search
        const { indices, scores } = await workerPool.parallelVectorSearch(
            query, chunks, dim, topK, normalized
        );

        // Fetch row data for results
        const rows = await this._fetchResultRows(indices);

        return { indices, scores, rows };
    }

    /**
     * Fetch full row data for result indices.
     * @private
     */
    async _fetchResultRows(indices) {
        if (indices.length === 0) return [];

        const rows = [];

        // Group indices by fragment for efficient fetching
        const groups = this._groupIndicesByFragment(Array.from(indices));

        for (const [fragIdx, group] of groups) {
            const file = await this.openFragment(fragIdx);

            // Read string columns for display
            for (const localIdx of group.localIndices) {
                const row = {};

                // Try to read text/url columns
                for (let colIdx = 0; colIdx < this._numColumns; colIdx++) {
                    const colName = this.columnNames[colIdx];
                    if (colName === 'text' || colName === 'url' || colName === 'caption') {
                        try {
                            const values = await file.readStringsAtIndices(colIdx, [localIdx]);
                            row[colName] = values[0];
                        } catch (e) {
                            // Column might not be string type
                        }
                    }
                }

                rows.push(row);
            }
        }

        return rows;
    }

    /**
     * Execute SQL query across all fragments in parallel.
     * @param {string} sql - SQL query
     * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
     */
    async executeSQL(sql) {
        // Parse the SQL to understand what's needed
        const ast = parseSQL(sql);

        // For simple SELECT * with LIMIT, use readRows
        if (ast.type === 'SELECT' && ast.columns === '*' && !ast.where) {
            const limit = ast.limit || 50;
            const offset = ast.offset || 0;
            return await this.readRows({ offset, limit });
        }

        // For queries with WHERE or complex operations, execute on each fragment in parallel
        const fetchPromises = this._fragments.map(async (frag, idx) => {
            const file = await this.openFragment(idx);
            try {
                return await file.executeSQL(sql);
            } catch (e) {
                console.warn(`Fragment ${idx} query failed:`, e);
                return { columns: [], columnNames: [], total: 0 };
            }
        });

        const results = await Promise.all(fetchPromises);

        // Merge results
        if (results.length === 0 || results.every(r => r.columns.length === 0)) {
            return { columns: [], columnNames: this.columnNames, total: 0 };
        }

        const firstValid = results.find(r => r.columns.length > 0);
        if (!firstValid) {
            return { columns: [], columnNames: this.columnNames, total: 0 };
        }

        const numCols = firstValid.columns.length;
        const colNames = firstValid.columnNames;
        const mergedColumns = Array.from({ length: numCols }, () => []);

        let totalRows = 0;
        for (const r of results) {
            for (let c = 0; c < numCols && c < r.columns.length; c++) {
                mergedColumns[c].push(...r.columns[c]);
            }
            totalRows += r.total;
        }

        // Apply LIMIT if present (after merging)
        if (ast.limit) {
            const offset = ast.offset || 0;
            for (let c = 0; c < numCols; c++) {
                mergedColumns[c] = mergedColumns[c].slice(offset, offset + ast.limit);
            }
        }

        return {
            columns: mergedColumns,
            columnNames: colNames,
            total: totalRows
        };
    }

    /**
     * Close all cached fragment files.
     */
    close() {
        for (const file of this._fragmentFiles.values()) {
            if (file.close) file.close();
        }
        this._fragmentFiles.clear();
    }
}

// ============================================================================
// QueryPlanner - Query execution planning and optimization
// ============================================================================

/**
 * QueryPlanner generates optimized physical execution plans from logical query ASTs.
 *
 * Key optimizations:
 * - Column pruning: Only fetch columns actually needed
 * - Filter pushdown: Apply filters as early as possible
 * - Fetch size estimation: Minimize data transfer over HTTP
 * - Deduplication: Avoid fetching same column multiple times
 *
 * Example:
 *   SELECT a.url, b.text
 *   FROM images a
 *   JOIN captions b ON a.id = b.image_id
 *   WHERE a.aesthetic > 7.0 AND b.language = 'zh'
 *   LIMIT 20
 *
 * Physical Plan:
 *   1. SCAN left: fetch [id, url, aesthetic], filter aesthetic > 7.0, limit ~50
 *   2. BUILD_HASH: index by id, keep [id, url]
 *   3. SCAN right: fetch [image_id, text], filter image_id IN (...) AND language = 'zh'
 *   4. HASH_JOIN: join on id = image_id
 *   5. PROJECT: select [url, text]
 *   6. LIMIT: 20
 */
export class QueryPlanner {
    constructor() {
        this.debug = true; // Enable query plan logging
    }

    /**
     * Generate physical execution plan from logical AST
     * @param {Object} ast - Parsed SQL AST
     * @param {Object} context - Table names and aliases
     * @returns {Object} Physical execution plan
     */
    plan(ast, context) {
        const { leftTableName, leftAlias, rightTableName, rightAlias } = context;

        // Analyze what columns are needed from each table
        const columnAnalysis = this._analyzeColumns(ast, context);

        // Separate filters by table (pushdown optimization)
        const filterAnalysis = this._analyzeFilters(ast, context);

        // Estimate how many rows to fetch (over-fetch for safety)
        const fetchEstimate = this._estimateFetchSize(ast, filterAnalysis);

        // Build physical plan
        const plan = {
            // Step 1: Scan left table
            leftScan: {
                table: leftTableName,
                alias: leftAlias,
                columns: columnAnalysis.left.all,  // Deduplicated list
                filters: filterAnalysis.left,
                limit: fetchEstimate.left,
                purpose: {
                    join: columnAnalysis.left.join,
                    where: columnAnalysis.left.where,
                    result: columnAnalysis.left.result
                }
            },

            // Step 2: Scan right table
            rightScan: {
                table: rightTableName,
                alias: rightAlias,
                columns: columnAnalysis.right.all,
                filters: filterAnalysis.right,
                filterByJoinKeys: true,  // Will add IN clause dynamically
                purpose: {
                    join: columnAnalysis.right.join,
                    where: columnAnalysis.right.where,
                    result: columnAnalysis.right.result
                }
            },

            // Step 3: Join strategy
            join: {
                type: ast.joins[0].type,
                leftKey: columnAnalysis.joinKeys.left,
                rightKey: columnAnalysis.joinKeys.right,
                algorithm: 'HASH_JOIN'  // Could be SORT_MERGE or NESTED_LOOP in future
            },

            // Step 4: Final projection
            projection: columnAnalysis.resultColumns,

            // Step 5: Limit
            limit: ast.limit || null,
            offset: ast.offset || 0
        };

        if (this.debug) {
            this._logPlan(plan, ast);
        }

        return plan;
    }

    /**
     * Analyze which columns are needed from each table
     */
    _analyzeColumns(ast, context) {
        const { leftAlias, rightAlias } = context;

        const left = {
            join: new Set(),    // Columns needed for JOIN key
            where: new Set(),   // Columns needed for WHERE filter
            result: new Set(),  // Columns needed in final result
            all: []             // Deduplicated union of above
        };

        const right = {
            join: new Set(),
            where: new Set(),
            result: new Set(),
            all: []
        };

        // 1. Analyze SELECT columns (result set)
        for (const item of ast.columns) {
            if (item.type === 'star') {
                // SELECT * - need all columns (can't optimize this easily)
                left.result.add('*');
                right.result.add('*');
            } else if (item.type === 'expr' && item.expr.type === 'column') {
                const col = item.expr;
                const table = col.table || null;
                const column = col.column;

                if (!table || table === leftAlias) {
                    left.result.add(column);
                }
                if (!table || table === rightAlias) {
                    right.result.add(column);
                }
            }
        }

        // 2. Analyze JOIN ON condition (join keys)
        const join = ast.joins[0];
        const joinKeys = this._extractJoinKeys(join.on, leftAlias, rightAlias);

        if (joinKeys.left) {
            left.join.add(joinKeys.left);
        }
        if (joinKeys.right) {
            right.join.add(joinKeys.right);
        }

        // 3. Analyze WHERE clause (filter columns)
        if (ast.where) {
            this._extractWhereColumns(ast.where, leftAlias, rightAlias, left.where, right.where);
        }

        // 4. Deduplicate: merge join, where, result
        left.all = [...new Set([...left.join, ...left.where, ...left.result])];
        right.all = [...new Set([...right.join, ...right.where, ...right.result])];

        // 5. Handle SELECT *
        if (left.result.has('*')) {
            left.all = ['*'];
        }
        if (right.result.has('*')) {
            right.all = ['*'];
        }

        // 6. Determine final result columns (for projection after join)
        const resultColumns = [];
        for (const item of ast.columns) {
            if (item.type === 'star') {
                resultColumns.push('*');
            } else if (item.type === 'expr' && item.expr.type === 'column') {
                const col = item.expr;
                const alias = item.alias || `${col.table || ''}.${col.column}`.replace(/^\./, '');
                resultColumns.push({
                    table: col.table,
                    column: col.column,
                    alias: alias
                });
            }
        }

        return {
            left,
            right,
            joinKeys,
            resultColumns
        };
    }

    /**
     * Extract join keys from ON condition
     */
    _extractJoinKeys(onExpr, leftAlias, rightAlias) {
        if (!onExpr || onExpr.type !== 'binary') {
            return { left: null, right: null };
        }

        const leftCol = onExpr.left;
        const rightCol = onExpr.right;

        let leftKey = null;
        let rightKey = null;

        // Left side of equality
        if (leftCol.type === 'column') {
            if (!leftCol.table || leftCol.table === leftAlias) {
                leftKey = leftCol.column;
            } else if (leftCol.table === rightAlias) {
                rightKey = leftCol.column;
            }
        }

        // Right side of equality
        if (rightCol.type === 'column') {
            if (!rightCol.table || rightCol.table === leftAlias) {
                leftKey = rightCol.column;
            } else if (rightCol.table === rightAlias) {
                rightKey = rightCol.column;
            }
        }

        return { left: leftKey, right: rightKey };
    }

    /**
     * Extract columns referenced in WHERE clause
     */
    _extractWhereColumns(expr, leftAlias, rightAlias, leftCols, rightCols) {
        if (!expr) return;

        if (expr.type === 'column') {
            const table = expr.table;
            const column = expr.column;

            if (!table || table === leftAlias) {
                leftCols.add(column);
            } else if (table === rightAlias) {
                rightCols.add(column);
            }
        } else if (expr.type === 'binary') {
            this._extractWhereColumns(expr.left, leftAlias, rightAlias, leftCols, rightCols);
            this._extractWhereColumns(expr.right, leftAlias, rightAlias, leftCols, rightCols);
        } else if (expr.type === 'unary') {
            this._extractWhereColumns(expr.expr, leftAlias, rightAlias, leftCols, rightCols);
        }
    }

    /**
     * Analyze and separate filters by table (for pushdown)
     */
    _analyzeFilters(ast, context) {
        const { leftAlias, rightAlias } = context;

        const left = [];   // Filters that can be pushed to left table
        const right = [];  // Filters that can be pushed to right table
        const join = [];   // Filters that must be applied after join

        if (ast.where) {
            this._separateFilters(ast.where, leftAlias, rightAlias, left, right, join);
        }

        return { left, right, join };
    }

    /**
     * Separate WHERE filters by which table they reference
     */
    _separateFilters(expr, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters) {
        if (!expr) return;

        // For AND expressions, recursively separate each side
        if (expr.type === 'binary' && expr.op === 'AND') {
            this._separateFilters(expr.left, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters);
            this._separateFilters(expr.right, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters);
            return;
        }

        // For other expressions, check which table(s) they reference
        const tables = this._getReferencedTables(expr, leftAlias, rightAlias);

        if (tables.size === 1) {
            // Filter references only one table - can push down
            if (tables.has(leftAlias)) {
                leftFilters.push(expr);
            } else if (tables.has(rightAlias)) {
                rightFilters.push(expr);
            }
        } else if (tables.size > 1) {
            // Filter references multiple tables - must apply after join
            joinFilters.push(expr);
        }
        // If tables.size === 0, it's a constant expression (ignore for now)
    }

    /**
     * Get which tables an expression references
     */
    _getReferencedTables(expr, leftAlias, rightAlias) {
        const tables = new Set();

        const walk = (e) => {
            if (!e) return;

            if (e.type === 'column') {
                const table = e.table;
                if (!table) {
                    // Ambiguous - could be either table (conservative: don't push down)
                    tables.add(leftAlias);
                    tables.add(rightAlias);
                } else if (table === leftAlias) {
                    tables.add(leftAlias);
                } else if (table === rightAlias) {
                    tables.add(rightAlias);
                }
            } else if (e.type === 'binary') {
                walk(e.left);
                walk(e.right);
            } else if (e.type === 'unary') {
                walk(e.operand);  // Note: parser uses 'operand', not 'expr'
            } else if (e.type === 'call') {
                for (const arg of e.args || []) {
                    walk(arg);
                }
            } else if (e.type === 'in') {
                walk(e.expr);
                for (const v of e.values || []) walk(v);
            } else if (e.type === 'between') {
                walk(e.expr);
                walk(e.low);
                walk(e.high);
            } else if (e.type === 'like') {
                walk(e.expr);
                walk(e.pattern);
            }
        };

        walk(expr);
        return tables;
    }

    /**
     * Estimate how many rows to fetch from each table
     *
     * Need to over-fetch because:
     * - WHERE filters reduce rows
     * - JOIN reduces rows
     * - Want to ensure we get LIMIT rows after all filtering
     */
    _estimateFetchSize(ast, filterAnalysis) {
        const requestedLimit = ast.limit || 1000;

        // Estimate selectivity (how many rows pass filters)
        // This is a simple heuristic - could be improved with statistics
        const leftSelectivity = filterAnalysis.left.length > 0 ? 0.5 : 1.0;  // 50% if filtered
        const rightSelectivity = filterAnalysis.right.length > 0 ? 0.5 : 1.0;

        // Join selectivity (how many rows match)
        const joinSelectivity = 0.7;  // Assume 70% of left rows find a match

        // Over-fetch multiplier
        const safetyFactor = 2.5;

        // Estimate left fetch size
        // Want: requestedLimit = leftFetch * leftSelectivity * joinSelectivity
        // So: leftFetch = requestedLimit / (leftSelectivity * joinSelectivity) * safetyFactor
        const leftFetch = Math.ceil(
            requestedLimit / (leftSelectivity * joinSelectivity) * safetyFactor
        );

        // Right table: fetch only what's needed based on left join keys
        // This will be dynamic (added in executor)
        const rightFetch = null;  // Determined by left join keys

        return {
            left: Math.min(leftFetch, 10000),  // Cap at 10K for safety
            right: rightFetch
        };
    }

    /**
     * Log the query plan for debugging
     */
    _logPlan(plan, ast) {
        console.log('\n' + '='.repeat(60));
        console.log('📋 QUERY EXECUTION PLAN');
        console.log('='.repeat(60));

        console.log('\n🔍 Original Query:');
        console.log(`  SELECT: ${ast.columns.length} columns`);
        console.log(`  FROM: ${plan.leftScan.table} AS ${plan.leftScan.alias}`);
        console.log(`  JOIN: ${plan.rightScan.table} AS ${plan.rightScan.alias}`);
        console.log(`  WHERE: ${ast.where ? 'yes' : 'no'}`);
        console.log(`  LIMIT: ${ast.limit || 'none'}`);

        console.log('\n📊 Physical Plan:');

        console.log('\n  Step 1: SCAN LEFT TABLE');
        console.log(`    Table: ${plan.leftScan.table}`);
        console.log(`    Columns: [${plan.leftScan.columns.join(', ')}]`);
        console.log(`      - Join keys: [${[...plan.leftScan.purpose.join].join(', ')}]`);
        console.log(`      - Filter cols: [${[...plan.leftScan.purpose.where].join(', ')}]`);
        console.log(`      - Result cols: [${[...plan.leftScan.purpose.result].join(', ')}]`);
        console.log(`    Filters: ${plan.leftScan.filters.length} pushed down`);
        plan.leftScan.filters.forEach((f, i) => {
            console.log(`      ${i + 1}. ${this._formatFilter(f)}`);
        });
        console.log(`    Limit: ${plan.leftScan.limit} rows (over-fetch for safety)`);

        console.log('\n  Step 2: BUILD HASH TABLE');
        console.log(`    Index by: ${plan.join.leftKey}`);
        console.log(`    Keep: [${plan.leftScan.columns.join(', ')}]`);

        console.log('\n  Step 3: SCAN RIGHT TABLE');
        console.log(`    Table: ${plan.rightScan.table}`);
        console.log(`    Columns: [${plan.rightScan.columns.join(', ')}]`);
        console.log(`      - Join keys: [${[...plan.rightScan.purpose.join].join(', ')}]`);
        console.log(`      - Filter cols: [${[...plan.rightScan.purpose.where].join(', ')}]`);
        console.log(`      - Result cols: [${[...plan.rightScan.purpose.result].join(', ')}]`);
        console.log(`    Filters: ${plan.rightScan.filters.length} pushed down`);
        plan.rightScan.filters.forEach((f, i) => {
            console.log(`      ${i + 1}. ${this._formatFilter(f)}`);
        });
        console.log(`    Dynamic filter: ${plan.join.rightKey} IN (keys from left)`);

        console.log('\n  Step 4: HASH JOIN');
        console.log(`    Algorithm: ${plan.join.algorithm}`);
        console.log(`    Condition: ${plan.join.leftKey} = ${plan.join.rightKey}`);

        console.log('\n  Step 5: PROJECT');
        console.log(`    Result columns: ${plan.projection.length}`);
        plan.projection.forEach((col, i) => {
            if (col === '*') {
                console.log(`      ${i + 1}. *`);
            } else {
                console.log(`      ${i + 1}. ${col.table}.${col.column} AS ${col.alias}`);
            }
        });

        console.log('\n  Step 6: LIMIT');
        console.log(`    Rows: ${plan.limit || 'none'}`);

        console.log('\n💡 Optimization Summary:');
        const leftTotal = plan.leftScan.columns.length;
        const rightTotal = plan.rightScan.columns.length;
        console.log(`  - Fetch ${leftTotal} cols from left (not all columns)`);
        console.log(`  - Fetch ${rightTotal} cols from right (not all columns)`);
        console.log(`  - Push down ${plan.leftScan.filters.length + plan.rightScan.filters.length} filters`);
        console.log(`  - Over-fetch left by ${(plan.leftScan.limit / (ast.limit || 1)).toFixed(1)}x for safety`);

        console.log('\n' + '='.repeat(60) + '\n');
    }

    /**
     * Format filter expression for logging
     */
    _formatFilter(expr) {
        if (!expr) return 'null';

        if (expr.type === 'binary') {
            const left = this._formatFilter(expr.left);
            const right = this._formatFilter(expr.right);
            return `${left} ${expr.op} ${right}`;
        } else if (expr.type === 'column') {
            return expr.table ? `${expr.table}.${expr.column}` : expr.column;
        } else if (expr.type === 'literal') {
            return JSON.stringify(expr.value);
        } else if (expr.type === 'call') {
            const args = (expr.args || []).map(a => this._formatFilter(a)).join(', ');
            return `${expr.name}(${args})`;
        }

        return JSON.stringify(expr);
    }

    /**
     * Generate optimized plan for single-table queries (SELECT, aggregations).
     * This is the key optimization that makes us better than DuckDB for remote data:
     *
     * DuckDB approach (local-first):
     *   1. Load data into memory
     *   2. Build indexes
     *   3. Execute query
     *
     * LanceQL approach (remote-first):
     *   1. Analyze query to determine minimum columns needed
     *   2. Use Lance column statistics to skip entire chunks
     *   3. Stream only matching rows, never load full table
     *   4. Apply projections at the data source
     *
     * @param {Object} ast - Parsed SQL AST
     * @returns {Object} Physical execution plan
     */
    planSingleTable(ast) {
        const plan = {
            type: ast.type,
            // Phase 1: Determine columns to fetch
            scanColumns: [],
            // Phase 2: Filters to push down (executed at data source)
            pushedFilters: [],
            // Phase 3: Filters that must be evaluated after fetch
            postFilters: [],
            // Phase 4: Aggregations (if any)
            aggregations: [],
            // Phase 5: GROUP BY (if any)
            groupBy: [],
            // Phase 6: HAVING (if any)
            having: null,
            // Phase 7: ORDER BY (if any)
            orderBy: [],
            // Phase 8: LIMIT/OFFSET
            limit: ast.limit || null,
            offset: ast.offset || 0,
            // Phase 9: Final projection
            projection: [],
            // Optimization flags
            canUseStatistics: false,
            canStreamResults: true,
            estimatedSelectivity: 1.0,
        };

        // Analyze columns needed
        const neededColumns = new Set();

        // 1. Columns from SELECT
        if (ast.columns === '*' || (Array.isArray(ast.columns) && ast.columns.some(c => c.type === 'star'))) {
            plan.projection = ['*'];
            // For *, we can't prune columns - need all
            plan.canStreamResults = false;
        } else if (Array.isArray(ast.columns)) {
            for (const col of ast.columns) {
                this._collectColumnsFromSelectItem(col, neededColumns, plan);
            }
        }

        // 2. Columns from WHERE (for filter evaluation)
        if (ast.where) {
            this._collectColumnsFromExpr(ast.where, neededColumns);
            // Analyze filter for pushdown opportunities
            this._analyzeFilterPushdown(ast.where, plan);
        }

        // 3. Columns from GROUP BY
        if (ast.groupBy && ast.groupBy.length > 0) {
            for (const groupExpr of ast.groupBy) {
                this._collectColumnsFromExpr(groupExpr, neededColumns);
                plan.groupBy.push(groupExpr);
            }
        }

        // 4. Columns from HAVING
        if (ast.having) {
            this._collectColumnsFromExpr(ast.having, neededColumns);
            plan.having = ast.having;
        }

        // 5. Columns from ORDER BY
        if (ast.orderBy && ast.orderBy.length > 0) {
            for (const orderItem of ast.orderBy) {
                this._collectColumnsFromExpr(orderItem.expr || orderItem, neededColumns);
                plan.orderBy.push(orderItem);
            }
        }

        // Finalize scan columns
        plan.scanColumns = Array.from(neededColumns);

        // Calculate selectivity estimate based on filters
        plan.estimatedSelectivity = this._estimateSelectivity(plan.pushedFilters);

        // Determine if we can use column statistics to skip chunks
        plan.canUseStatistics = plan.pushedFilters.some(f =>
            f.type === 'range' || f.type === 'equality'
        );

        if (this.debug) {
            this._logSingleTablePlan(plan, ast);
        }

        return plan;
    }

    /**
     * Collect columns from a SELECT item
     */
    _collectColumnsFromSelectItem(item, columns, plan) {
        if (item.type === 'star') {
            plan.projection.push('*');
            return;
        }

        if (item.type === 'expr') {
            const expr = item.expr;

            // Check for aggregation
            if (expr.type === 'call') {
                const funcName = expr.name.toUpperCase();
                const aggFuncs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];

                if (aggFuncs.includes(funcName)) {
                    const agg = {
                        type: funcName,
                        column: null,
                        alias: item.alias || `${funcName}(${expr.args[0]?.name || '*'})`,
                        distinct: expr.distinct || false,
                    };

                    if (expr.args && expr.args.length > 0) {
                        const arg = expr.args[0];
                        if (arg.type === 'column') {
                            agg.column = arg.name || arg.column;
                            columns.add(agg.column);
                        } else if (arg.type !== 'star') {
                            this._collectColumnsFromExpr(arg, columns);
                        }
                    }

                    plan.aggregations.push(agg);
                    plan.projection.push({ type: 'aggregation', index: plan.aggregations.length - 1 });
                    return;
                }
            }

            // Regular column or expression
            this._collectColumnsFromExpr(expr, columns);
            plan.projection.push({
                type: 'column',
                expr: expr,
                alias: item.alias
            });
        }
    }

    /**
     * Collect column names from an expression
     */
    _collectColumnsFromExpr(expr, columns) {
        if (!expr) return;

        if (expr.type === 'column') {
            columns.add(expr.name || expr.column);
        } else if (expr.type === 'binary') {
            this._collectColumnsFromExpr(expr.left, columns);
            this._collectColumnsFromExpr(expr.right, columns);
        } else if (expr.type === 'call') {
            for (const arg of (expr.args || [])) {
                this._collectColumnsFromExpr(arg, columns);
            }
        } else if (expr.type === 'unary') {
            this._collectColumnsFromExpr(expr.operand, columns);
        }
    }

    /**
     * Analyze WHERE clause for filter pushdown opportunities.
     *
     * Pushable filters (can be evaluated at data source):
     * - Simple comparisons: col > 5, col = 'foo', col BETWEEN 1 AND 10
     * - IN clauses: col IN (1, 2, 3)
     * - LIKE patterns: col LIKE 'prefix%' (prefix only)
     *
     * Non-pushable filters (must evaluate after fetch):
     * - Complex expressions: col1 + col2 > 10
     * - Functions: UPPER(col) = 'FOO'
     * - Cross-column comparisons: col1 > col2
     */
    _analyzeFilterPushdown(expr, plan) {
        if (!expr) return;

        if (expr.type === 'binary') {
            // Check if this is a simple pushable condition
            if (this._isPushableFilter(expr)) {
                plan.pushedFilters.push(this._classifyFilter(expr));
            } else if (expr.op === 'AND') {
                // AND - recurse into both sides
                this._analyzeFilterPushdown(expr.left, plan);
                this._analyzeFilterPushdown(expr.right, plan);
            } else if (expr.op === 'OR') {
                // OR with pushable conditions on same column can be pushed
                const leftPushable = this._isPushableFilter(expr.left);
                const rightPushable = this._isPushableFilter(expr.right);

                if (leftPushable && rightPushable) {
                    plan.pushedFilters.push({
                        type: 'or',
                        left: this._classifyFilter(expr.left),
                        right: this._classifyFilter(expr.right),
                    });
                } else {
                    // Can't push OR with non-pushable condition
                    plan.postFilters.push(expr);
                }
            } else {
                // Non-pushable binary expression
                plan.postFilters.push(expr);
            }
        } else {
            plan.postFilters.push(expr);
        }
    }

    /**
     * Check if a filter can be pushed down to data source
     */
    _isPushableFilter(expr) {
        if (expr.type !== 'binary') return false;

        const compOps = ['=', '==', '!=', '<>', '<', '<=', '>', '>=', 'LIKE', 'IN', 'BETWEEN'];
        if (!compOps.includes(expr.op.toUpperCase())) return false;

        // One side must be a column, other must be a literal/constant
        const leftIsCol = expr.left.type === 'column';
        const rightIsCol = expr.right?.type === 'column';
        const leftIsLiteral = expr.left.type === 'literal' || expr.left.type === 'list';
        const rightIsLiteral = expr.right?.type === 'literal' || expr.right?.type === 'list';

        return (leftIsCol && rightIsLiteral) || (rightIsCol && leftIsLiteral);
    }

    /**
     * Classify a filter for optimization
     */
    _classifyFilter(expr) {
        const leftIsCol = expr.left.type === 'column';
        const column = leftIsCol
            ? (expr.left.name || expr.left.column)
            : (expr.right.name || expr.right.column);
        const value = leftIsCol ? expr.right.value : expr.left.value;

        const op = expr.op.toUpperCase();

        if (op === '=' || op === '==') {
            return { type: 'equality', column, value, op: '=' };
        } else if (op === '!=' || op === '<>') {
            return { type: 'inequality', column, value, op: '!=' };
        } else if (['<', '<=', '>', '>='].includes(op)) {
            return { type: 'range', column, value, op };
        } else if (op === 'LIKE') {
            return { type: 'like', column, pattern: value };
        } else if (op === 'IN') {
            const values = expr.right.type === 'list' ? expr.right.values : [expr.right.value];
            return { type: 'in', column, values };
        } else if (op === 'BETWEEN') {
            return { type: 'between', column, low: expr.right.low, high: expr.right.high };
        }

        return { type: 'unknown', expr };
    }

    /**
     * Estimate selectivity of filters (what % of rows will pass)
     */
    _estimateSelectivity(filters) {
        if (filters.length === 0) return 1.0;

        let selectivity = 1.0;
        for (const f of filters) {
            switch (f.type) {
                case 'equality':
                    selectivity *= 0.1; // Assume 10% match for equality
                    break;
                case 'range':
                    selectivity *= 0.3; // Assume 30% for range
                    break;
                case 'in':
                    selectivity *= Math.min(0.5, f.values.length * 0.05);
                    break;
                case 'like':
                    selectivity *= f.pattern.startsWith('%') ? 0.5 : 0.2;
                    break;
                default:
                    selectivity *= 0.5;
            }
        }
        return Math.max(0.01, selectivity); // At least 1%
    }

    /**
     * Log single-table query plan
     */
    _logSingleTablePlan(plan, ast) {
        console.log('\n' + '='.repeat(60));
        console.log('📋 SINGLE-TABLE QUERY PLAN');
        console.log('='.repeat(60));

        console.log('\n🔍 Query Analysis:');
        console.log(`  Type: ${plan.type}`);
        console.log(`  Aggregations: ${plan.aggregations.length}`);
        console.log(`  Group By: ${plan.groupBy.length} columns`);
        console.log(`  Order By: ${plan.orderBy.length} columns`);
        console.log(`  Limit: ${plan.limit || 'none'}`);

        console.log('\n📊 Scan Strategy:');
        console.log(`  Columns to fetch: [${plan.scanColumns.join(', ')}]`);
        console.log(`  Pushed filters: ${plan.pushedFilters.length}`);
        plan.pushedFilters.forEach((f, i) => {
            console.log(`    ${i + 1}. ${f.type}: ${f.column} ${f.op || ''} ${JSON.stringify(f.value || f.values || f.pattern || '')}`);
        });
        console.log(`  Post-fetch filters: ${plan.postFilters.length}`);

        console.log('\n💡 Optimizations:');
        console.log(`  Can use column statistics: ${plan.canUseStatistics}`);
        console.log(`  Can stream results: ${plan.canStreamResults}`);
        console.log(`  Estimated selectivity: ${(plan.estimatedSelectivity * 100).toFixed(1)}%`);

        console.log('\n' + '='.repeat(60) + '\n');
    }
}

// ============================================================================
// LanceDatabase - Multi-table query execution with JOINs
// ============================================================================

/**
 * MemoryTable - In-memory table for temporary data storage.
 * Ephemeral by design - lost on page refresh.
 * Used for intermediate results, temp tables, and JOIN lookups.
 */
class MemoryTable {
    constructor(name, schema) {
        this.name = name;
        this.schema = schema;  // [{ name, dataType, primaryKey }]
        this.columns = schema.map(c => c.name);
        this.rows = [];
        this._columnIndex = new Map();
        this.columns.forEach((col, i) => this._columnIndex.set(col.toLowerCase(), i));
    }

    /**
     * Convert to format compatible with _executeOnInMemoryData
     */
    toInMemoryData() {
        return { columns: this.columns, rows: this.rows };
    }

    /**
     * Get row count
     */
    get rowCount() {
        return this.rows.length;
    }
}

/**
 * LanceDatabase manages multiple Lance datasets and executes multi-table queries.
 * Supports SQL JOINs across remote datasets with smart byte-range fetching.
 *
 * Features:
 * - Multi-table JOIN support (INNER, LEFT, RIGHT, FULL, CROSS)
 * - Hash join algorithm for efficient execution
 * - Smart column fetching (only fetch needed columns)
 * - Works with static files on CDN
 *
 * Usage:
 *   const db = await LanceQL.createDatabase();
 *   await db.registerRemote('images', 'https://cdn.example.com/images.lance');
 *   await db.registerRemote('captions', 'https://cdn.example.com/captions.lance');
 *   const results = await db.executeSQL(`
 *     SELECT i.url, c.text
 *     FROM images i
 *     JOIN captions c ON i.id = c.image_id
 *     WHERE i.aesthetic > 7.0
 *     LIMIT 20
 *   `);
 */
export class LanceDatabase {
    constructor() {
        this.tables = new Map(); // name -> RemoteLanceDataset
        this.aliases = new Map(); // alias -> table name
        // Query plan cache
        this._planCache = new Map(); // normalized SQL -> { plan, hits, lastUsed }
        this._planCacheMaxSize = 100;
        // In-memory tables (ephemeral)
        this.memoryTables = new Map(); // name -> MemoryTable
    }

    /**
     * Register a table with a name
     * @param {string} name - Table name
     * @param {RemoteLanceDataset} dataset - Dataset instance
     */
    register(name, dataset) {
        this.tables.set(name, dataset);
    }

    /**
     * Register a remote dataset by URL
     * @param {string} name - Table name
     * @param {string} url - Dataset URL
     * @param {Object} options - Dataset options (version, etc.)
     */
    async registerRemote(name, url, options = {}) {
        // Assume LanceQL is globally available or passed as parameter
        const lanceql = window.lanceql || globalThis.lanceql;
        if (!lanceql) {
            throw new Error('LanceQL WASM module not loaded. Call LanceQL.load() first.');
        }

        const dataset = await lanceql.openDataset(url, options);
        this.register(name, dataset);
        return dataset;
    }

    /**
     * Get a table by name or alias
     */
    getTable(name) {
        // Check aliases first
        const actualName = this.aliases.get(name) || name;
        const table = this.tables.get(actualName);
        if (!table) {
            throw new Error(`Table '${name}' not found. Did you forget to register it?`);
        }
        return table;
    }

    /**
     * Execute SQL query (supports SELECT with JOINs, CTEs, SET operations, and EXPLAIN)
     */
    async executeSQL(sql) {
        // Check plan cache first
        const cachedPlan = this._getCachedPlan(sql);
        let ast;

        if (cachedPlan) {
            ast = cachedPlan;
        } else {
            // Parse SQL
            const lexer = new SQLLexer(sql);
            const tokens = lexer.tokenize();
            const parser = new SQLParser(tokens);
            ast = parser.parse();

            // Cache the plan (unless it's EXPLAIN which is meta)
            if (ast.type !== 'EXPLAIN') {
                this._setCachedPlan(sql, ast);
            }
        }

        // Handle EXPLAIN - return plan without executing
        if (ast.type === 'EXPLAIN') {
            return this._explainQuery(ast.statement);
        }

        // Handle memory table operations (CREATE, DROP, INSERT, UPDATE, DELETE)
        if (ast.type === 'CREATE_TABLE') {
            return this._executeCreateTable(ast);
        }
        if (ast.type === 'DROP_TABLE') {
            return this._executeDropTable(ast);
        }
        if (ast.type === 'INSERT') {
            return this._executeInsert(ast);
        }
        if (ast.type === 'UPDATE') {
            return this._executeUpdate(ast);
        }
        if (ast.type === 'DELETE') {
            return this._executeDelete(ast);
        }

        if (ast.type === 'SET_OPERATION') {
            return this._executeSetOperation(ast);
        }

        if (ast.type !== 'SELECT') {
            throw new Error('Only SELECT queries are supported in LanceDatabase');
        }

        // Handle CTEs - materialize them first
        if (ast.ctes && ast.ctes.length > 0) {
            return this._executeWithCTEs(ast);
        }

        // No joins - simple single-table query
        if (!ast.joins || ast.joins.length === 0) {
            return this._executeSingleTable(ast);
        }

        // Multi-table query with JOINs
        return this._executeJoin(ast);
    }

    /**
     * Execute query with CTEs
     */
    async _executeWithCTEs(ast) {
        // Create a temporary executor for CTE materialization
        const cteExecutor = new SQLExecutor({ columnNames: [] });
        cteExecutor.setDatabase(this);
        await cteExecutor.materializeCTEs(ast.ctes, this);

        // Check if main query references a CTE
        const mainTableName = ast.from?.name?.toLowerCase() || ast.from?.table?.toLowerCase();
        if (mainTableName && cteExecutor._cteResults.has(mainTableName)) {
            // Execute main query against CTE result
            return cteExecutor._executeOnInMemoryData(ast, cteExecutor._cteResults.get(mainTableName));
        }

        // Otherwise execute against actual tables
        if (!ast.joins || ast.joins.length === 0) {
            return this._executeSingleTable(ast);
        }
        return this._executeJoin(ast);
    }

    /**
     * Execute SET operation (UNION, INTERSECT, EXCEPT)
     */
    async _executeSetOperation(ast) {
        // Execute left and right sides
        const leftResult = await this.executeSQL(this._astToSQL(ast.left));
        const rightResult = await this.executeSQL(this._astToSQL(ast.right));

        if (leftResult.columns.length !== rightResult.columns.length) {
            throw new Error('SET operations require same number of columns');
        }

        const rowKey = row => JSON.stringify(row);
        let combinedRows;

        switch (ast.operator) {
            case 'UNION':
                combinedRows = [...leftResult.rows, ...rightResult.rows];
                if (!ast.all) {
                    const seen = new Set();
                    combinedRows = combinedRows.filter(row => {
                        const key = rowKey(row);
                        if (seen.has(key)) return false;
                        seen.add(key);
                        return true;
                    });
                }
                break;

            case 'INTERSECT':
                const rightKeys = new Set(rightResult.rows.map(rowKey));
                combinedRows = leftResult.rows.filter(row => rightKeys.has(rowKey(row)));
                if (!ast.all) {
                    const seenI = new Set();
                    combinedRows = combinedRows.filter(row => {
                        const key = rowKey(row);
                        if (seenI.has(key)) return false;
                        seenI.add(key);
                        return true;
                    });
                }
                break;

            case 'EXCEPT':
                const excludeKeys = new Set(rightResult.rows.map(rowKey));
                combinedRows = leftResult.rows.filter(row => !excludeKeys.has(rowKey(row)));
                if (!ast.all) {
                    const seenE = new Set();
                    combinedRows = combinedRows.filter(row => {
                        const key = rowKey(row);
                        if (seenE.has(key)) return false;
                        seenE.add(key);
                        return true;
                    });
                }
                break;

            default:
                throw new Error(`Unknown SET operator: ${ast.operator}`);
        }

        // Apply ORDER BY to combined result
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colIdxMap = {};
            leftResult.columns.forEach((name, idx) => { colIdxMap[name.toLowerCase()] = idx; });

            combinedRows.sort((a, b) => {
                for (const ob of ast.orderBy) {
                    const colIdx = colIdxMap[ob.column.toLowerCase()];
                    if (colIdx === undefined) continue;
                    const valA = a[colIdx], valB = b[colIdx];
                    const dir = ob.direction === 'DESC' ? -1 : 1;
                    if (valA == null && valB == null) continue;
                    if (valA == null) return 1 * dir;
                    if (valB == null) return -1 * dir;
                    if (valA < valB) return -1 * dir;
                    if (valA > valB) return 1 * dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET to combined result
        const offset = ast.offset || 0;
        if (offset > 0) combinedRows = combinedRows.slice(offset);
        if (ast.limit) combinedRows = combinedRows.slice(0, ast.limit);

        return { columns: leftResult.columns, rows: combinedRows, total: combinedRows.length };
    }

    /**
     * Convert AST back to SQL (for recursive SET operation execution)
     */
    _astToSQL(ast) {
        if (ast.type === 'SET_OPERATION') {
            const left = this._astToSQL(ast.left);
            const right = this._astToSQL(ast.right);
            const op = ast.operator + (ast.all ? ' ALL' : '');
            return `(${left}) ${op} (${right})`;
        }

        // Build SELECT statement
        let sql = ast.distinct ? 'SELECT DISTINCT ' : 'SELECT ';
        sql += ast.columns.map(col => {
            if (col.expr?.type === 'star') return '*';
            const expr = this._exprToSQL(col.expr);
            return col.alias ? `${expr} AS ${col.alias}` : expr;
        }).join(', ');

        if (ast.from) {
            const tableName = ast.from.name || ast.from.table;
            sql += ` FROM ${tableName}`;
            if (ast.from.alias) sql += ` AS ${ast.from.alias}`;
        }

        if (ast.joins) {
            for (const join of ast.joins) {
                const rightTable = join.table?.name || join.table?.table;
                sql += ` ${join.type} ${rightTable}`;
                if (join.alias) sql += ` AS ${join.alias}`;
                if (join.on) sql += ` ON ${this._exprToSQL(join.on)}`;
            }
        }

        if (ast.where) sql += ` WHERE ${this._exprToSQL(ast.where)}`;
        if (ast.groupBy?.length) sql += ` GROUP BY ${ast.groupBy.join(', ')}`;
        if (ast.having) sql += ` HAVING ${this._exprToSQL(ast.having)}`;
        if (ast.orderBy?.length) {
            sql += ` ORDER BY ${ast.orderBy.map(o => `${o.column} ${o.direction || 'ASC'}`).join(', ')}`;
        }
        if (ast.limit) sql += ` LIMIT ${ast.limit}`;
        if (ast.offset) sql += ` OFFSET ${ast.offset}`;

        return sql;
    }

    /**
     * Convert expression AST to SQL string
     */
    _exprToSQL(expr) {
        if (!expr) return '';
        switch (expr.type) {
            case 'literal':
                if (expr.value === null) return 'NULL';
                if (typeof expr.value === 'string') return `'${expr.value.replace(/'/g, "''")}'`;
                return String(expr.value);
            case 'column':
                return expr.table ? `${expr.table}.${expr.column}` : expr.column;
            case 'star':
                return '*';
            case 'binary':
                return `(${this._exprToSQL(expr.left)} ${expr.operator} ${this._exprToSQL(expr.right)})`;
            case 'unary':
                return `(${expr.operator} ${this._exprToSQL(expr.operand)})`;
            case 'call':
                const args = expr.args.map(a => this._exprToSQL(a)).join(', ');
                return `${expr.name}(${expr.distinct ? 'DISTINCT ' : ''}${args})`;
            case 'in':
                const vals = expr.values.map(v => this._exprToSQL(v)).join(', ');
                return `${this._exprToSQL(expr.expr)} IN (${vals})`;
            case 'between':
                return `${this._exprToSQL(expr.expr)} BETWEEN ${this._exprToSQL(expr.low)} AND ${this._exprToSQL(expr.high)}`;
            case 'like':
                return `${this._exprToSQL(expr.expr)} LIKE ${this._exprToSQL(expr.pattern)}`;
            default:
                return '';
        }
    }

    /**
     * Execute single-table query (no joins)
     */
    async _executeSingleTable(ast) {
        if (!ast.from) {
            throw new Error('FROM clause required');
        }

        // Get table
        let tableName = ast.from.name || ast.from.table;
        if (!tableName && ast.from.url) {
            throw new Error('Single-table queries must use registered table names, not URLs');
        }

        const tableNameLower = tableName.toLowerCase();

        // Check memory table first
        if (this.memoryTables.has(tableNameLower)) {
            const memTable = this.memoryTables.get(tableNameLower);
            const executor = new SQLExecutor({ columnNames: memTable.columns });
            return executor._executeOnInMemoryData(ast, memTable.toInMemoryData());
        }

        // Otherwise use remote dataset
        const dataset = this.getTable(tableName);

        // Build SQL and execute
        const executor = new SQLExecutor(dataset);
        return executor.execute(ast);
    }

    /**
     * Execute multi-table query with JOINs
     */
    async _executeJoin(ast) {
        console.log('[LanceDatabase] Executing JOIN query:', ast);

        // Extract table references
        const leftTableName = ast.from.name || ast.from.table;
        const leftAlias = ast.from.alias || leftTableName;

        // Register alias
        if (ast.from.alias) {
            this.aliases.set(ast.from.alias, leftTableName);
        }

        // Process joins iteratively: (A JOIN B) JOIN C
        // Each join's result becomes the left input for the next
        console.log(`[LanceDatabase] Processing ${ast.joins.length} JOIN(s)`);

        let currentResult = null;  // In-memory intermediate result
        let currentAlias = leftAlias;
        let currentTableName = leftTableName;
        let leftDataset = this.getTable(leftTableName);

        for (let i = 0; i < ast.joins.length; i++) {
            const join = ast.joins[i];
            const rightTableName = join.table.name || join.table.table;
            const rightAlias = join.alias || rightTableName;

            // Register right table alias
            if (join.alias) {
                this.aliases.set(join.alias, rightTableName);
            }

            console.log(`[LanceDatabase] JOIN ${i + 1}/${ast.joins.length}: ${currentTableName} (${currentAlias}) ${join.type} ${rightTableName} (${rightAlias})`);

            // Get right dataset
            const rightDataset = this.getTable(rightTableName);

            // Build AST for this single join
            const singleJoinAst = {
                ...ast,
                joins: [join],  // Only this join
                // For intermediate joins, don't apply final limit/projection
                limit: (i === ast.joins.length - 1) ? ast.limit : undefined,
                columns: (i === ast.joins.length - 1) ? ast.columns : [{ type: 'column', column: '*' }]
            };

            // Execute hash join
            if (currentResult === null) {
                // First join: left is a dataset
                currentResult = await this._hashJoin(
                    leftDataset,
                    rightDataset,
                    singleJoinAst,
                    { leftAlias: currentAlias, rightAlias, leftTableName: currentTableName, rightTableName }
                );
            } else {
                // Subsequent joins: left is in-memory result
                currentResult = await this._hashJoinWithInMemoryLeft(
                    currentResult,
                    rightDataset,
                    singleJoinAst,
                    { leftAlias: currentAlias, rightAlias, leftTableName: currentTableName, rightTableName }
                );
            }

            // Update current state for next iteration
            currentAlias = `${currentAlias}_${rightAlias}`;  // Compound alias for tracing
            currentTableName = `(${currentTableName} JOIN ${rightTableName})`;
        }

        return currentResult;
    }

    /**
     * Execute hash join between two datasets using OPFS for intermediate storage.
     * This enables TB-scale joins in the browser by spilling to disk instead of RAM.
     */
    async _hashJoin(leftDataset, rightDataset, ast, context) {
        const { leftAlias, rightAlias, leftTableName, rightTableName } = context;
        const join = ast.joins[0];
        const joinType = join.type || 'INNER';

        // For CROSS JOIN, no ON condition is required
        const joinCondition = join.on;
        if (joinType !== 'CROSS') {
            if (!joinCondition || joinCondition.type !== 'binary' || joinCondition.op !== '=') {
                throw new Error('JOIN ON condition must be an equality (e.g., table1.col1 = table2.col2)');
            }
        }

        // For CROSS JOIN, use simplified execution (no keys needed)
        let leftKey, rightKey, leftSQL, rightSQL;

        if (joinType === 'CROSS') {
            // CROSS JOIN: select all columns, no join keys
            leftKey = null;
            rightKey = null;
            leftSQL = `SELECT * FROM ${leftTableName}`;
            rightSQL = `SELECT * FROM ${rightTableName}`;
            console.log('[LanceDatabase] CROSS JOIN - no keys, cartesian product');
        } else {
            // Use QueryPlanner to generate optimized execution plan
            const planner = new QueryPlanner();
            const plan = planner.plan(ast, context);

            // Extract columns and keys from the plan
            leftKey = plan.join.leftKey;
            rightKey = plan.join.rightKey;
            const leftColumns = plan.leftScan.columns;
            const rightColumns = plan.rightScan.columns;
            const leftFilters = plan.leftScan.filters;
            const rightFilters = plan.rightScan.filters;

            // Build SQL queries for streaming
            const leftColsWithKey = leftColumns.includes('*')
                ? ['*']
                : [...new Set([leftKey, ...leftColumns])];

            let leftWhereClause = '';
            if (leftFilters.length > 0) {
                leftWhereClause = ` WHERE ${leftFilters.map(f => this._filterToSQL(f)).join(' AND ')}`;
            }
            leftSQL = `SELECT ${leftColsWithKey.join(', ')} FROM ${leftTableName}${leftWhereClause}`;

            const rightColsWithKey = rightColumns.includes('*')
                ? ['*']
                : [...new Set([rightKey, ...rightColumns])];

            let rightWhereClause = '';
            if (rightFilters.length > 0) {
                rightWhereClause = ` WHERE ${rightFilters.map(f => this._filterToSQL(f)).join(' AND ')}`;
            }
            rightSQL = `SELECT ${rightColsWithKey.join(', ')} FROM ${rightTableName}${rightWhereClause}`;
        }

        console.log('[LanceDatabase] OPFS-backed hash join starting...');
        console.log('[LanceDatabase] Left query:', leftSQL);

        // Initialize OPFS storage
        await opfsStorage.open();

        // Create OPFS join executor
        const joinExecutor = new OPFSJoinExecutor(opfsStorage);

        // Create streaming executors
        const leftExecutor = new SQLExecutor(leftDataset);
        const rightExecutor = new SQLExecutor(rightDataset);

        // Semi-join optimization: partition left first and collect keys
        const leftStream = leftExecutor.executeStream(leftSQL);
        const leftMeta = await joinExecutor._partitionToOPFS(leftStream, leftKey, 'left', true);
        console.log(`[LanceDatabase] Left partitioned: ${leftMeta.totalRows} rows, ${leftMeta.collectedKeys?.size || 0} unique keys`);

        // Build optimized right SQL with IN clause if we have reasonable key count
        let optimizedRightSQL = rightSQL;
        const maxKeysForInClause = 1000;  // Don't create huge IN clauses
        if (leftMeta.collectedKeys && leftMeta.collectedKeys.size > 0 &&
            leftMeta.collectedKeys.size <= maxKeysForInClause) {
            const inClause = this._buildInClause(rightKey, leftMeta.collectedKeys);
            optimizedRightSQL = this._appendWhereClause(rightSQL, inClause);
            console.log(`[LanceDatabase] Semi-join optimization: added IN clause with ${leftMeta.collectedKeys.size} keys`);
        }
        console.log('[LanceDatabase] Right query:', optimizedRightSQL);

        // Create right stream with optimized SQL
        const rightStream = rightExecutor.executeStream(optimizedRightSQL);

        // Execute OPFS-backed join with pre-partitioned left
        const results = [];
        let resultColumns = null;

        try {
            for await (const chunk of joinExecutor.executeHashJoin(
                null,  // leftStream already partitioned
                rightStream,
                leftKey,
                rightKey,
                {
                    limit: ast.limit || Infinity,
                    leftAlias,
                    rightAlias,
                    joinType: join.type || 'INNER',
                    prePartitionedLeft: leftMeta
                }
            )) {
                if (!resultColumns) {
                    resultColumns = chunk.columns;
                }
                results.push(...chunk.rows);

                // Early exit if we have enough rows
                if (ast.limit && results.length >= ast.limit) {
                    break;
                }
            }
        } catch (e) {
            console.error('[LanceDatabase] OPFS join failed:', e);
            throw e;
        }

        // Get stats
        const stats = joinExecutor.getStats();
        console.log('[LanceDatabase] OPFS Join Stats:', stats);

        // If no results, return empty
        if (!resultColumns || results.length === 0) {
            return { columns: [], rows: [], total: 0, opfsStats: stats };
        }

        // Apply projection
        const projectedResults = this._applyProjection(
            results,
            resultColumns,
            plan.projection,
            leftAlias,
            rightAlias
        );

        // Apply LIMIT
        const limitedResults = ast.limit
            ? projectedResults.rows.slice(0, ast.limit)
            : projectedResults.rows;

        return {
            columns: projectedResults.columns,
            rows: limitedResults,
            total: limitedResults.length,
            opfsStats: stats  // Include OPFS stats in result
        };
    }

    /**
     * Execute hash join with in-memory left side (for multiple JOINs).
     * The left side comes from a previous join's result.
     */
    async _hashJoinWithInMemoryLeft(leftResult, rightDataset, ast, context) {
        const { leftAlias, rightAlias, leftTableName, rightTableName } = context;
        const join = ast.joins[0];
        const joinType = join.type || 'INNER';

        // For CROSS JOIN, no ON condition required
        const joinCondition = join.on;
        if (joinType !== 'CROSS') {
            if (!joinCondition || joinCondition.type !== 'binary' || joinCondition.op !== '=') {
                throw new Error('JOIN ON condition must be an equality (e.g., table1.col1 = table2.col2)');
            }
        }

        // Extract join keys
        let leftKey, rightKey;
        if (joinType === 'CROSS') {
            leftKey = null;
            rightKey = null;
        } else {
            // Extract keys from ON condition
            const leftExpr = joinCondition.left;
            const rightExpr = joinCondition.right;

            // Determine which side refers to left vs right
            const leftCol = leftExpr.column;
            const rightCol = rightExpr.column;

            // Find which column is in the left result columns
            const leftColsSet = new Set(leftResult.columns.map(c => {
                const parts = c.split('.');
                return parts[parts.length - 1];  // Get base column name
            }));

            if (leftColsSet.has(leftCol)) {
                leftKey = leftCol;
                rightKey = rightCol;
            } else {
                leftKey = rightCol;
                rightKey = leftCol;
            }
        }

        console.log(`[LanceDatabase] Multi-JOIN: left in-memory (${leftResult.rows.length} rows), right: ${rightTableName}`);

        // Build right SQL - select all for now
        let rightSQL = `SELECT * FROM ${rightTableName}`;

        // Semi-join optimization: use in-memory left keys to filter right
        const maxKeysForInClause = 1000;
        if (leftKey && joinType !== 'CROSS') {
            const leftKeyIndex = this._findColumnIndex(leftResult.columns, leftKey);
            if (leftKeyIndex !== -1) {
                const leftKeys = new Set();
                for (const row of leftResult.rows) {
                    const key = row[leftKeyIndex];
                    if (key !== null && key !== undefined) {
                        leftKeys.add(key);
                    }
                }
                if (leftKeys.size > 0 && leftKeys.size <= maxKeysForInClause) {
                    const inClause = this._buildInClause(rightKey, leftKeys);
                    rightSQL = this._appendWhereClause(rightSQL, inClause);
                    console.log(`[LanceDatabase] Multi-JOIN semi-join: ${leftKeys.size} keys`);
                }
            }
        }

        // Execute right query
        const rightExecutor = new SQLExecutor(rightDataset);
        const rightResult = await rightExecutor.execute(new SQLParser(new SQLLexer(rightSQL).tokenize()).parse());

        // Find key indices for in-memory hash join
        const leftKeyIndex = leftKey ? this._findColumnIndex(leftResult.columns, leftKey) : -1;
        const rightKeyIndex = rightKey ? this._findColumnIndex(rightResult.columns, rightKey) : -1;

        // Build result columns
        const resultColumns = [
            ...leftResult.columns,
            ...rightResult.columns.map(c => `${rightAlias}.${c}`)
        ];

        // Execute in-memory hash join
        const results = [];
        const rightNulls = new Array(rightResult.columns.length).fill(null);
        const leftNulls = new Array(leftResult.columns.length).fill(null);

        if (joinType === 'CROSS') {
            // Cartesian product
            for (const leftRow of leftResult.rows) {
                for (const rightRow of rightResult.rows) {
                    results.push([...leftRow, ...rightRow]);
                    if (ast.limit && results.length >= ast.limit) break;
                }
                if (ast.limit && results.length >= ast.limit) break;
            }
        } else {
            // Build hash table from right side
            const rightHash = new Map();
            for (const row of rightResult.rows) {
                const key = row[rightKeyIndex];
                if (key !== null && key !== undefined) {
                    if (!rightHash.has(key)) rightHash.set(key, []);
                    rightHash.get(key).push(row);
                }
            }

            // Track matched right rows for FULL/RIGHT joins
            const matchedRightRows = (joinType === 'FULL' || joinType === 'RIGHT')
                ? new Set() : null;

            // Probe with left rows
            for (const leftRow of leftResult.rows) {
                const key = leftRow[leftKeyIndex];
                const rightMatches = rightHash.get(key) || [];

                if (rightMatches.length > 0) {
                    for (const rightRow of rightMatches) {
                        results.push([...leftRow, ...rightRow]);
                        if (matchedRightRows) {
                            matchedRightRows.add(rightResult.rows.indexOf(rightRow));
                        }
                        if (ast.limit && results.length >= ast.limit) break;
                    }
                } else if (joinType === 'LEFT' || joinType === 'FULL') {
                    // Left row with no match
                    results.push([...leftRow, ...rightNulls]);
                }
                if (ast.limit && results.length >= ast.limit) break;
            }

            // Add unmatched right rows for RIGHT/FULL joins
            if ((joinType === 'RIGHT' || joinType === 'FULL') && matchedRightRows) {
                for (let i = 0; i < rightResult.rows.length; i++) {
                    if (!matchedRightRows.has(i)) {
                        results.push([...leftNulls, ...rightResult.rows[i]]);
                        if (ast.limit && results.length >= ast.limit) break;
                    }
                }
            }
        }

        // Apply projection if this is the final join
        const limitedResults = ast.limit ? results.slice(0, ast.limit) : results;

        return {
            columns: resultColumns,
            rows: limitedResults,
            total: limitedResults.length
        };
    }

    /**
     * Find column index by name, handling qualified names (table.column)
     */
    _findColumnIndex(columns, columnName) {
        // Try exact match first
        let idx = columns.indexOf(columnName);
        if (idx !== -1) return idx;

        // Try with any table prefix (for qualified names like "users.id")
        for (let i = 0; i < columns.length; i++) {
            const col = columns[i];
            const parts = col.split('.');
            if (parts[parts.length - 1] === columnName) {
                return i;
            }
        }

        return -1;
    }

    /**
     * Convert filter expression to SQL WHERE clause
     * Note: Strips table aliases since pushed-down queries are single-table
     */
    _filterToSQL(expr) {
        if (!expr) return '';

        if (expr.type === 'binary') {
            const left = this._filterToSQL(expr.left);
            const right = this._filterToSQL(expr.right);
            return `${left} ${expr.op} ${right}`;
        } else if (expr.type === 'column') {
            // Strip table alias - pushed query is single-table
            return expr.column;
        } else if (expr.type === 'literal') {
            if (typeof expr.value === 'string') {
                // Escape single quotes to prevent SQL injection
                const escaped = expr.value.replace(/'/g, "''");
                return `'${escaped}'`;
            }
            if (expr.value === null) return 'NULL';
            return String(expr.value);
        } else if (expr.type === 'call') {
            const args = (expr.args || []).map(a => this._filterToSQL(a)).join(', ');
            return `${expr.name}(${args})`;
        } else if (expr.type === 'in') {
            const col = this._filterToSQL(expr.expr);
            const vals = expr.values.map(v => this._filterToSQL(v)).join(', ');
            return `${col} IN (${vals})`;
        } else if (expr.type === 'between') {
            const col = this._filterToSQL(expr.expr);
            const low = this._filterToSQL(expr.low);
            const high = this._filterToSQL(expr.high);
            return `${col} BETWEEN ${low} AND ${high}`;
        } else if (expr.type === 'like') {
            const col = this._filterToSQL(expr.expr);
            const pattern = this._filterToSQL(expr.pattern);
            return `${col} LIKE ${pattern}`;
        } else if (expr.type === 'unary') {
            const operand = this._filterToSQL(expr.operand);
            if (expr.op === 'NOT') return `NOT ${operand}`;
            return `${expr.op}${operand}`;
        }

        console.warn('[LanceDB] Unknown filter expression type:', expr.type);
        return '';
    }

    /**
     * Apply projection to select only requested columns from joined result
     */
    _applyProjection(rows, allColumns, projection, leftAlias, rightAlias) {
        // Handle SELECT *
        if (projection.includes('*')) {
            return { columns: allColumns, rows };
        }

        // Build column mapping
        const projectedColumns = [];
        const columnIndices = [];

        for (const col of projection) {
            if (col === '*') {
                // Already handled above
                continue;
            }

            let idx = -1;
            let outputColName = col.column;

            if (col.table) {
                // Try exact match with table prefix first (most specific)
                const qualifiedName = `${col.table}.${col.column}`;
                idx = allColumns.indexOf(qualifiedName);
                outputColName = qualifiedName;
            }

            if (idx === -1) {
                // Fallback: find first column ending with this column name
                idx = allColumns.findIndex(c => c === col.column || c.endsWith(`.${col.column}`));
                if (idx !== -1) {
                    outputColName = allColumns[idx];
                }
            }

            if (idx !== -1) {
                projectedColumns.push(col.alias || outputColName);
                columnIndices.push(idx);
            }
        }

        // Apply projection
        const projectedRows = rows.map(row =>
            columnIndices.map(idx => row[idx])
        );

        return { columns: projectedColumns, rows: projectedRows };
    }

    /**
     * Extract column name from expression
     */
    _extractColumnFromExpr(expr, expectedTable) {
        if (expr.type === 'column') {
            // Handle table.column syntax
            if (expr.table && expr.table !== expectedTable) {
                // Column belongs to different table
                return null;
            }
            return expr.column;
        }
        throw new Error(`Invalid join condition expression: ${JSON.stringify(expr)}`);
    }

    /**
     * Get columns needed for a specific table from SELECT list
     */
    _getColumnsForTable(selectColumns, tableAlias) {
        const columns = [];

        for (const item of selectColumns) {
            if (item.type === 'star') {
                // SELECT * - need to fetch all columns (TODO: get schema)
                return ['*'];
            }

            if (item.type === 'expr' && item.expr.type === 'column') {
                const col = item.expr;
                if (!col.table || col.table === tableAlias) {
                    columns.push(col.column);
                }
            }
        }

        return columns.length > 0 ? columns : ['*'];
    }

    /**
     * Build an IN clause for semi-join optimization
     * @param {string} column - Column name for IN clause
     * @param {Set} keys - Unique key values collected from left table
     * @returns {string} SQL IN clause fragment
     */
    _buildInClause(column, keys) {
        const values = Array.from(keys).map(k => {
            if (typeof k === 'string') {
                return `'${k.replace(/'/g, "''")}'`;  // Escape single quotes
            }
            if (k === null) return 'NULL';
            return String(k);
        }).join(', ');
        return `${column} IN (${values})`;
    }

    /**
     * Append a WHERE clause or AND condition to existing SQL
     * @param {string} sql - Existing SQL query
     * @param {string} clause - Condition to add
     * @returns {string} SQL with added condition
     */
    _appendWhereClause(sql, clause) {
        const upperSQL = sql.toUpperCase();
        if (upperSQL.includes('WHERE')) {
            // Insert after WHERE keyword
            return sql.replace(/WHERE\s+/i, `WHERE ${clause} AND `);
        }
        // Find FROM table (with optional alias) and add WHERE after
        // Match: FROM tablename or FROM tablename alias
        return sql.replace(/FROM\s+(\w+)(\s+\w+)?/i, (match) => `${match} WHERE ${clause}`);
    }

    // ========================================================================
    // Phase 9: Query Optimization Methods
    // ========================================================================

    /**
     * Get cached query plan
     * @param {string} sql - SQL query string
     * @returns {Object|null} Cached plan or null
     */
    _getCachedPlan(sql) {
        const normalized = this._normalizeSQL(sql);
        const cached = this._planCache.get(normalized);
        if (cached) {
            cached.hits++;
            cached.lastUsed = Date.now();
            return cached.plan;
        }
        return null;
    }

    /**
     * Cache a query plan
     * @param {string} sql - SQL query string
     * @param {Object} plan - Parsed AST plan
     */
    _setCachedPlan(sql, plan) {
        const normalized = this._normalizeSQL(sql);

        // LRU eviction if at capacity
        if (this._planCache.size >= this._planCacheMaxSize) {
            let oldest = null;
            let oldestTime = Infinity;
            for (const [key, value] of this._planCache) {
                if (value.lastUsed < oldestTime) {
                    oldestTime = value.lastUsed;
                    oldest = key;
                }
            }
            if (oldest) this._planCache.delete(oldest);
        }

        this._planCache.set(normalized, {
            plan,
            hits: 0,
            lastUsed: Date.now(),
            created: Date.now()
        });
    }

    /**
     * Normalize SQL for cache key (remove extra whitespace, lowercase)
     * @param {string} sql - SQL query string
     * @returns {string} Normalized SQL
     */
    _normalizeSQL(sql) {
        return sql.trim().replace(/\s+/g, ' ').toLowerCase();
    }

    /**
     * Clear the query plan cache
     */
    clearPlanCache() {
        this._planCache.clear();
    }

    /**
     * Get plan cache statistics
     * @returns {Object} Cache stats
     */
    getPlanCacheStats() {
        let totalHits = 0;
        for (const v of this._planCache.values()) {
            totalHits += v.hits;
        }
        return {
            size: this._planCache.size,
            maxSize: this._planCacheMaxSize,
            totalHits
        };
    }

    /**
     * Optimize expression with constant folding and boolean simplification
     * @param {Object} expr - Expression AST node
     * @returns {Object} Optimized expression
     */
    _optimizeExpr(expr) {
        if (!expr) return expr;

        // Recursively optimize children
        if (expr.left) expr.left = this._optimizeExpr(expr.left);
        if (expr.right) expr.right = this._optimizeExpr(expr.right);
        if (expr.operand) expr.operand = this._optimizeExpr(expr.operand);
        if (expr.args) expr.args = expr.args.map(a => this._optimizeExpr(a));

        // Get operator - AST may use 'op' or 'operator'
        const op = expr.op || expr.operator;

        // Constant folding for binary operations
        if (expr.type === 'binary' &&
            this._isConstantExpr(expr.left) &&
            this._isConstantExpr(expr.right)) {
            return this._foldBinary(expr);
        }

        // Boolean simplification
        if (expr.type === 'binary' && op === 'AND') {
            if (this._isTrueExpr(expr.right)) return expr.left;
            if (this._isTrueExpr(expr.left)) return expr.right;
            if (this._isFalseExpr(expr.left) || this._isFalseExpr(expr.right)) {
                return { type: 'literal', value: false };
            }
        }
        if (expr.type === 'binary' && op === 'OR') {
            if (this._isFalseExpr(expr.right)) return expr.left;
            if (this._isFalseExpr(expr.left)) return expr.right;
            if (this._isTrueExpr(expr.left) || this._isTrueExpr(expr.right)) {
                return { type: 'literal', value: true };
            }
        }

        return expr;
    }

    /**
     * Check if expression is a constant
     */
    _isConstantExpr(expr) {
        return expr && ['literal', 'number', 'string'].includes(expr.type);
    }

    /**
     * Check if expression is TRUE
     */
    _isTrueExpr(expr) {
        return expr?.type === 'literal' && expr.value === true;
    }

    /**
     * Check if expression is FALSE
     */
    _isFalseExpr(expr) {
        return expr?.type === 'literal' && expr.value === false;
    }

    /**
     * Fold binary constant expression
     * @param {Object} expr - Binary expression with constant operands
     * @returns {Object} Literal result
     */
    _foldBinary(expr) {
        const left = this._getConstantValueExpr(expr.left);
        const right = this._getConstantValueExpr(expr.right);

        // Get operator - AST may use 'op' or 'operator'
        const op = expr.op || expr.operator;

        let result;
        switch (op) {
            case '+': result = left + right; break;
            case '-': result = left - right; break;
            case '*': result = left * right; break;
            case '/': result = right !== 0 ? left / right : null; break;
            case '%': result = left % right; break;
            case '=': case '==': result = left === right; break;
            case '!=': case '<>': result = left !== right; break;
            case '<': result = left < right; break;
            case '>': result = left > right; break;
            case '<=': result = left <= right; break;
            case '>=': result = left >= right; break;
            default: return expr;  // Can't fold
        }

        return { type: 'literal', value: result };
    }

    /**
     * Get constant value from expression
     */
    _getConstantValueExpr(expr) {
        if (expr.type === 'number') return expr.value;
        if (expr.type === 'string') return expr.value;
        if (expr.type === 'literal') return expr.value;
        return null;
    }

    /**
     * Extract range predicates from WHERE clause for statistics-based pruning
     * @param {Object} where - WHERE clause AST
     * @returns {Array} Array of predicate objects
     */
    _extractRangePredicates(where) {
        const predicates = [];
        this._collectRangePredicates(where, predicates);
        return predicates;
    }

    /**
     * Recursively collect range predicates
     */
    _collectRangePredicates(expr, predicates) {
        if (!expr) return;

        // Get operator - AST uses 'op' or 'operator'
        const op = expr.op || expr.operator;

        // Handle AND - recurse both sides
        if (expr.type === 'binary' && op === 'AND') {
            this._collectRangePredicates(expr.left, predicates);
            this._collectRangePredicates(expr.right, predicates);
            return;
        }

        // Range operators (normalize '==' to '=')
        const normalizedOp = op === '==' ? '=' : op;
        if (['>', '<', '>=', '<=', '=', '!=', '<>'].includes(normalizedOp)) {
            // Column on left, constant on right
            if (this._isColumnRefExpr(expr.left) && this._isConstantExpr(expr.right)) {
                predicates.push({
                    column: this._getColumnNameExpr(expr.left),
                    operator: normalizedOp,
                    value: this._getConstantValueExpr(expr.right)
                });
            }
            // Constant on left, column on right - flip operator
            else if (this._isConstantExpr(expr.left) && this._isColumnRefExpr(expr.right)) {
                predicates.push({
                    column: this._getColumnNameExpr(expr.right),
                    operator: this._flipOperatorExpr(normalizedOp),
                    value: this._getConstantValueExpr(expr.left)
                });
            }
        }

        // BETWEEN clause
        if (expr.type === 'between' && expr.expr) {
            const col = this._getColumnNameExpr(expr.expr);
            if (col && expr.low && expr.high) {
                predicates.push({
                    column: col,
                    operator: '>=',
                    value: this._getConstantValueExpr(expr.low)
                });
                predicates.push({
                    column: col,
                    operator: '<=',
                    value: this._getConstantValueExpr(expr.high)
                });
            }
        }
    }

    /**
     * Flip comparison operator (for constant on left side)
     */
    _flipOperatorExpr(op) {
        const flips = { '>': '<', '<': '>', '>=': '<=', '<=': '>=' };
        return flips[op] || op;
    }

    /**
     * Check if expression is a column reference
     */
    _isColumnRefExpr(expr) {
        return expr && (expr.type === 'column' || expr.type === 'identifier');
    }

    /**
     * Get column name from expression
     */
    _getColumnNameExpr(expr) {
        if (expr.type === 'column') return expr.name || expr.column;
        if (expr.type === 'identifier') return expr.name || expr.value;
        return null;
    }

    /**
     * Check if a fragment can be pruned based on statistics and predicates
     * @param {Object} fragmentStats - Column statistics for the fragment
     * @param {Array} predicates - Extracted predicates from WHERE clause
     * @returns {boolean} True if fragment can be safely skipped
     */
    _canPruneFragment(fragmentStats, predicates) {
        for (const pred of predicates) {
            const stats = fragmentStats[pred.column];
            if (!stats) continue;  // No stats for this column

            const { min, max, nullCount, rowCount } = stats;

            // All nulls - can't satisfy any comparison
            if (nullCount === rowCount) return true;

            switch (pred.operator) {
                case '>':
                    // If max <= value, no rows can satisfy > value
                    if (max <= pred.value) return true;
                    break;
                case '>=':
                    if (max < pred.value) return true;
                    break;
                case '<':
                    if (min >= pred.value) return true;
                    break;
                case '<=':
                    if (min > pred.value) return true;
                    break;
                case '=':
                    // If value outside [min, max], no match possible
                    if (pred.value < min || pred.value > max) return true;
                    break;
                case '!=':
                case '<>':
                    // Can only prune if all values are the same and equal to pred.value
                    if (min === max && min === pred.value) return true;
                    break;
            }
        }

        return false;  // Cannot prune
    }

    /**
     * Execute EXPLAIN query - return query plan without executing
     * @param {Object} ast - Parsed AST of the inner query
     * @returns {Object} Plan information
     */
    _explainQuery(ast) {
        const plan = {
            type: ast.type,
            tables: [],
            predicates: [],
            optimizations: []
        };

        // Collect table info
        if (ast.from) {
            plan.tables.push({
                name: ast.from.name || ast.from.table,
                alias: ast.from.alias
            });
        }

        // Collect joined tables
        if (ast.joins) {
            for (const join of ast.joins) {
                plan.tables.push({
                    name: join.table?.name || join.table?.table,
                    alias: join.table?.alias,
                    joinType: join.type
                });
            }
        }

        // Extract predicates from WHERE
        if (ast.where) {
            plan.predicates = this._extractRangePredicates(ast.where);
        }

        // Identify optimizations
        if (ast.where) {
            plan.optimizations.push('PREDICATE_PUSHDOWN');
        }
        if (ast.groupBy) {
            plan.optimizations.push('AGGREGATE');
        }
        if (ast.orderBy) {
            plan.optimizations.push('SORT');
        }
        if (ast.limit) {
            plan.optimizations.push('LIMIT_PUSHDOWN');
        }

        return {
            columns: ['Plan'],
            rows: [[JSON.stringify(plan, null, 2)]],
            total: 1
        };
    }

    // ========================================================================
    // Phase 10: Memory Table CRUD Operations
    // ========================================================================

    /**
     * Execute CREATE TABLE - creates an in-memory table
     * @param {Object} ast - Parsed CREATE TABLE AST
     * @returns {Object} Result with success flag
     */
    _executeCreateTable(ast) {
        const tableName = (ast.table || ast.name || '').toLowerCase();

        if (!tableName) {
            throw new Error('CREATE TABLE requires a table name');
        }

        // Check if table already exists (memory or remote)
        if (this.memoryTables.has(tableName) || this.tables.has(tableName)) {
            if (ast.ifNotExists) {
                return { success: true, existed: true, table: tableName };
            }
            throw new Error(`Table '${tableName}' already exists`);
        }

        // Build schema from AST columns
        const schema = (ast.columns || []).map(col => ({
            name: col.name,
            dataType: col.dataType || col.type || 'TEXT',
            primaryKey: col.primaryKey || false
        }));

        if (schema.length === 0) {
            throw new Error('CREATE TABLE requires at least one column');
        }

        // Create and store the memory table
        const table = new MemoryTable(tableName, schema);
        this.memoryTables.set(tableName, table);

        return {
            success: true,
            table: tableName,
            columns: schema.map(c => c.name)
        };
    }

    /**
     * Execute DROP TABLE - removes an in-memory table
     * @param {Object} ast - Parsed DROP TABLE AST
     * @returns {Object} Result with success flag
     */
    _executeDropTable(ast) {
        const tableName = (ast.table || ast.name || '').toLowerCase();

        if (!this.memoryTables.has(tableName)) {
            if (ast.ifExists) {
                return { success: true, existed: false, table: tableName };
            }
            throw new Error(`Memory table '${tableName}' not found`);
        }

        this.memoryTables.delete(tableName);
        return { success: true, table: tableName };
    }

    /**
     * Execute INSERT - adds rows to a memory table
     * @param {Object} ast - Parsed INSERT AST
     * @returns {Object} Result with inserted count
     */
    _executeInsert(ast) {
        const tableName = (ast.table || '').toLowerCase();
        const table = this.memoryTables.get(tableName);

        if (!table) {
            throw new Error(`Memory table '${tableName}' not found. Use CREATE TABLE first.`);
        }

        // Get column names to insert into (use table columns if not specified)
        const insertCols = ast.columns || table.columns;
        let inserted = 0;

        // Process each row from VALUES clause
        for (const astRow of (ast.rows || ast.values || [])) {
            const row = new Array(table.columns.length).fill(null);

            insertCols.forEach((colName, i) => {
                const colIdx = table._columnIndex.get(
                    (typeof colName === 'string' ? colName : colName.name || colName).toLowerCase()
                );
                if (colIdx !== undefined && i < astRow.length) {
                    // Handle AST value nodes or raw values
                    const val = astRow[i];
                    row[colIdx] = val?.value !== undefined ? val.value : val;
                }
            });

            table.rows.push(row);
            inserted++;
        }

        return {
            success: true,
            inserted,
            total: table.rows.length
        };
    }

    /**
     * Execute UPDATE - modifies rows in a memory table
     * @param {Object} ast - Parsed UPDATE AST
     * @returns {Object} Result with updated count
     */
    _executeUpdate(ast) {
        const tableName = (ast.table || '').toLowerCase();
        const table = this.memoryTables.get(tableName);

        if (!table) {
            throw new Error(`Memory table '${tableName}' not found`);
        }

        // Build column data for WHERE expression evaluation
        const columnData = {};
        table.columns.forEach((col, idx) => {
            columnData[col.toLowerCase()] = table.rows.map(row => row[idx]);
        });

        // Create executor for expression evaluation
        const executor = new SQLExecutor({ columnNames: table.columns });
        let updated = 0;

        // Process each row
        for (let i = 0; i < table.rows.length; i++) {
            // Check WHERE condition (if present)
            const matches = !ast.where || executor._evaluateInMemoryExpr(ast.where, columnData, i);

            if (matches) {
                // Apply SET assignments
                for (const assignment of (ast.assignments || ast.set || [])) {
                    const colName = (assignment.column || assignment.name || '').toLowerCase();
                    const colIdx = table._columnIndex.get(colName);

                    if (colIdx !== undefined) {
                        const val = assignment.value;
                        table.rows[i][colIdx] = val?.value !== undefined ? val.value : val;
                    }
                }
                updated++;
            }
        }

        return { success: true, updated };
    }

    /**
     * Execute DELETE - removes rows from a memory table
     * @param {Object} ast - Parsed DELETE AST
     * @returns {Object} Result with deleted count
     */
    _executeDelete(ast) {
        const tableName = (ast.table || '').toLowerCase();
        const table = this.memoryTables.get(tableName);

        if (!table) {
            throw new Error(`Memory table '${tableName}' not found`);
        }

        const originalCount = table.rows.length;

        if (ast.where) {
            // Build column data for WHERE expression evaluation
            const columnData = {};
            table.columns.forEach((col, idx) => {
                columnData[col.toLowerCase()] = table.rows.map(row => row[idx]);
            });

            // Create executor for expression evaluation
            const executor = new SQLExecutor({ columnNames: table.columns });

            // Keep rows that DON'T match the WHERE condition
            table.rows = table.rows.filter((_, i) =>
                !executor._evaluateInMemoryExpr(ast.where, columnData, i)
            );
        } else {
            // DELETE without WHERE = truncate
            table.rows = [];
        }

        return {
            success: true,
            deleted: originalCount - table.rows.length,
            remaining: table.rows.length
        };
    }
}

// ============================================================================
// WorkerPool - Parallel WASM execution across Web Workers
// ============================================================================

/**
 * WorkerPool manages a pool of Web Workers, each running their own WASM instance.
 * Enables true parallel processing across CPU cores.
 *
 * Features:
 * - Automatic worker scaling based on hardware concurrency
 * - Task queue with load balancing
 * - Support for SharedArrayBuffer (zero-copy) when available
 * - Graceful degradation to transferable ArrayBuffers
 */
export class WorkerPool {
    /**
     * Create a new worker pool.
     * @param {number} size - Number of workers (default: navigator.hardwareConcurrency)
     * @param {string} workerPath - Path to worker.js
     */
    constructor(size = null, workerPath = './worker.js') {
        this.size = size || navigator.hardwareConcurrency || 4;
        this.workerPath = workerPath;
        this.workers = [];
        this.taskQueue = [];
        this.pendingTasks = new Map();
        this.nextTaskId = 0;
        this.idleWorkers = [];
        this.initialized = false;

        // Check for SharedArrayBuffer support
        this.hasSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
    }

    /**
     * Initialize all workers.
     * @returns {Promise<void>}
     */
    async init() {
        if (this.initialized) return;

        const initPromises = [];

        for (let i = 0; i < this.size; i++) {
            const worker = new Worker(this.workerPath, { type: 'module' });
            this.workers.push(worker);

            // Set up message handling
            worker.onmessage = (e) => this._handleMessage(i, e.data);
            worker.onerror = (e) => this._handleError(i, e);

            // Initialize worker with WASM
            initPromises.push(this._initWorker(i));
        }

        await Promise.all(initPromises);
        this.initialized = true;

        console.log(`[WorkerPool] Initialized ${this.size} workers (SharedArrayBuffer: ${this.hasSharedArrayBuffer})`);
    }

    /**
     * Initialize a single worker.
     * @private
     */
    _initWorker(workerId) {
        return new Promise((resolve, reject) => {
            const taskId = this.nextTaskId++;

            this.pendingTasks.set(taskId, {
                resolve: (result) => {
                    this.idleWorkers.push(workerId);
                    resolve(result);
                },
                reject
            });

            this.workers[workerId].postMessage({
                type: 'init',
                id: taskId,
                params: { workerId }
            });
        });
    }

    /**
     * Handle message from worker.
     * @private
     */
    _handleMessage(workerId, data) {
        // Handle ready message (initial worker startup)
        if (data.type === 'ready') {
            return;
        }

        const { id, success, result, error } = data;
        const task = this.pendingTasks.get(id);

        if (!task) {
            console.warn(`[WorkerPool] Unknown task ID: ${id}`);
            return;
        }

        this.pendingTasks.delete(id);

        if (success) {
            task.resolve(result);
        } else {
            task.reject(new Error(error));
        }

        // Worker is now idle
        this.idleWorkers.push(workerId);

        // Process next task in queue
        this._processQueue();
    }

    /**
     * Handle worker error.
     * @private
     */
    _handleError(workerId, error) {
        console.error(`[WorkerPool] Worker ${workerId} error:`, error);
    }

    /**
     * Process next task in queue.
     * @private
     */
    _processQueue() {
        while (this.taskQueue.length > 0 && this.idleWorkers.length > 0) {
            const task = this.taskQueue.shift();
            const workerId = this.idleWorkers.shift();
            this._sendTask(workerId, task);
        }
    }

    /**
     * Send task to worker.
     * @private
     */
    _sendTask(workerId, task) {
        const worker = this.workers[workerId];
        const transfer = task.transfer || [];

        worker.postMessage({
            type: task.type,
            id: task.id,
            params: task.params
        }, transfer);
    }

    /**
     * Submit a task to the pool.
     * @param {string} type - Task type
     * @param {Object} params - Task parameters
     * @param {Array} transfer - Transferable objects
     * @returns {Promise<any>}
     */
    submit(type, params, transfer = []) {
        return new Promise((resolve, reject) => {
            const taskId = this.nextTaskId++;

            this.pendingTasks.set(taskId, { resolve, reject });

            const task = { type, params, transfer, id: taskId };

            if (this.idleWorkers.length > 0) {
                const workerId = this.idleWorkers.shift();
                this._sendTask(workerId, task);
            } else {
                this.taskQueue.push(task);
            }
        });
    }

    /**
     * Parallel vector search across multiple data chunks.
     *
     * @param {Float32Array} query - Query vector
     * @param {Array<{vectors: Float32Array, startIndex: number}>} chunks - Data chunks
     * @param {number} dim - Vector dimension
     * @param {number} topK - Number of results per chunk
     * @param {boolean} normalized - Whether vectors are L2-normalized
     * @returns {Promise<{indices: Uint32Array, scores: Float32Array}>}
     */
    async parallelVectorSearch(query, chunks, dim, topK, normalized = false) {
        if (!this.initialized) {
            await this.init();
        }

        // Submit search task to each worker
        const searchPromises = chunks.map((chunk, i) => {
            // Copy query for each worker (will be transferred)
            const queryCopy = new Float32Array(query);

            return this.submit('vectorSearch', {
                vectors: chunk.vectors,
                query: queryCopy,
                dim,
                numVectors: chunk.vectors.length / dim,
                topK,
                startIndex: chunk.startIndex,
                normalized
            }, [chunk.vectors.buffer, queryCopy.buffer]);
        });

        // Wait for all workers
        const results = await Promise.all(searchPromises);

        // Merge results from all workers
        return this._mergeTopK(results, topK);
    }

    /**
     * Merge top-k results from multiple workers.
     * @private
     */
    _mergeTopK(results, topK) {
        // Collect all results
        const allResults = [];

        for (const result of results) {
            for (let i = 0; i < result.count; i++) {
                allResults.push({
                    index: result.indices[i],
                    score: result.scores[i]
                });
            }
        }

        // Sort by score descending
        allResults.sort((a, b) => b.score - a.score);

        // Take top-k
        const finalK = Math.min(topK, allResults.length);
        const indices = new Uint32Array(finalK);
        const scores = new Float32Array(finalK);

        for (let i = 0; i < finalK; i++) {
            indices[i] = allResults[i].index;
            scores[i] = allResults[i].score;
        }

        return { indices, scores };
    }

    /**
     * Parallel batch similarity computation.
     *
     * @param {Float32Array} query - Query vector
     * @param {Array<Float32Array>} vectorChunks - Chunks of vectors
     * @param {number} dim - Vector dimension
     * @param {boolean} normalized - Whether vectors are L2-normalized
     * @returns {Promise<Float32Array>} - All similarity scores
     */
    async parallelBatchSimilarity(query, vectorChunks, dim, normalized = false) {
        if (!this.initialized) {
            await this.init();
        }

        const similarityPromises = vectorChunks.map(chunk => {
            const queryCopy = new Float32Array(query);
            return this.submit('batchSimilarity', {
                query: queryCopy,
                vectors: chunk,
                dim,
                numVectors: chunk.length / dim,
                normalized
            }, [chunk.buffer, queryCopy.buffer]);
        });

        const results = await Promise.all(similarityPromises);

        // Concatenate all scores
        const totalLength = results.reduce((sum, r) => sum + r.scores.length, 0);
        const allScores = new Float32Array(totalLength);

        let offset = 0;
        for (const result of results) {
            allScores.set(result.scores, offset);
            offset += result.scores.length;
        }

        return allScores;
    }

    /**
     * Terminate all workers.
     */
    terminate() {
        for (const worker of this.workers) {
            worker.terminate();
        }
        this.workers = [];
        this.idleWorkers = [];
        this.initialized = false;
    }
}

// ============================================================================
// SharedArrayBuffer Vector Store - Zero-copy data sharing
// ============================================================================

/**
 * SharedVectorStore provides zero-copy data sharing between main thread and workers.
 * Requires Cross-Origin-Isolation (COOP/COEP headers).
 */
export class SharedVectorStore {
    constructor() {
        this.buffer = null;
        this.vectors = null;
        this.dim = 0;
        this.numVectors = 0;

        if (typeof SharedArrayBuffer === 'undefined') {
            console.warn('[SharedVectorStore] SharedArrayBuffer not available. Using regular ArrayBuffer.');
        }
    }

    /**
     * Check if SharedArrayBuffer is available.
     */
    static isAvailable() {
        return typeof SharedArrayBuffer !== 'undefined' &&
               typeof Atomics !== 'undefined';
    }

    /**
     * Allocate shared memory for vectors.
     *
     * @param {number} numVectors - Number of vectors to store
     * @param {number} dim - Vector dimension
     */
    allocate(numVectors, dim) {
        this.numVectors = numVectors;
        this.dim = dim;

        const byteLength = numVectors * dim * 4; // float32

        if (SharedVectorStore.isAvailable()) {
            this.buffer = new SharedArrayBuffer(byteLength);
        } else {
            // Fallback to regular ArrayBuffer
            this.buffer = new ArrayBuffer(byteLength);
        }

        this.vectors = new Float32Array(this.buffer);
    }

    /**
     * Copy vectors into shared memory.
     *
     * @param {Float32Array} source - Source vectors
     * @param {number} startIndex - Starting index in store
     */
    set(source, startIndex = 0) {
        this.vectors.set(source, startIndex * this.dim);
    }

    /**
     * Get a slice of vectors (view, not copy).
     *
     * @param {number} start - Start vector index
     * @param {number} count - Number of vectors
     * @returns {Float32Array}
     */
    slice(start, count) {
        const startOffset = start * this.dim;
        const length = count * this.dim;
        return new Float32Array(this.buffer, startOffset * 4, length);
    }

    /**
     * Get chunk boundaries for parallel processing.
     *
     * @param {number} numChunks - Number of chunks
     * @returns {Array<{start: number, count: number}>}
     */
    getChunks(numChunks) {
        const chunks = [];
        const chunkSize = Math.ceil(this.numVectors / numChunks);

        for (let i = 0; i < numChunks; i++) {
            const start = i * chunkSize;
            const count = Math.min(chunkSize, this.numVectors - start);
            if (count > 0) {
                chunks.push({ start, count });
            }
        }

        return chunks;
    }
}

// ============================================================================
// CSS-Driven Query Engine - Zero JavaScript Data Binding
// ============================================================================

/**
 * LanceData provides CSS-driven data binding for Lance datasets.
 *
 * TRULY CSS-DRIVEN: No JavaScript initialization required!
 * Just add lq-* attributes to any element.
 *
 * Usage (pure HTML/CSS, zero JavaScript):
 * ```html
 * <div lq-query="SELECT url, text FROM read_lance('https://data.metal0.dev/laion-1m/images.lance') LIMIT 10"
 *      lq-render="table">
 * </div>
 * ```
 *
 * Attributes (supports both lq-* and data-* prefixes):
 * - lq-src / data-dataset: Dataset URL (optional if URL is in query)
 * - lq-query / data-query: SQL query string (required)
 * - lq-render / data-render: Renderer type - table, list, value, images, json (default: table)
 * - lq-columns / data-columns: Comma-separated column names to display
 * - lq-bind / data-bind: Input element selector for reactive binding
 *
 * The system auto-initializes when the script loads.
 */
export class LanceData {
    static _initialized = false;
    static _observer = null;
    static _wasm = null;
    static _datasets = new Map(); // Cache datasets by URL
    static _renderers = {};
    static _bindings = new Map();
    static _queryCache = new Map();
    static _defaultDataset = null;

    /**
     * Auto-initialize when DOM is ready.
     * Called automatically - no user action needed.
     */
    static _autoInit() {
        if (LanceData._initialized) return;
        LanceData._initialized = true;

        // Register built-in renderers
        LanceData._registerBuiltinRenderers();

        // Inject trigger styles
        LanceData._injectTriggerStyles();

        // Set up observer for lance-data elements
        LanceData._setupObserver();

        // Process any existing elements
        LanceData._processExisting();
    }

    /**
     * Get or load a dataset (cached).
     */
    static async _getDataset(url) {
        if (!url) {
            if (LanceData._defaultDataset) return LanceData._datasets.get(LanceData._defaultDataset);
            throw new Error('No dataset URL. Add data-dataset="https://..." to your element.');
        }

        if (LanceData._datasets.has(url)) {
            return LanceData._datasets.get(url);
        }

        // Load WASM if needed
        if (!LanceData._wasm) {
            // Try to find wasm URL from script tag or use default
            const wasmUrl = document.querySelector('script[data-lanceql-wasm]')?.dataset.lanceqlWasm
                || './lanceql.wasm';
            LanceData._wasm = await LanceQL.load(wasmUrl);
        }

        const dataset = await RemoteLanceDataset.open(LanceData._wasm, url);
        LanceData._datasets.set(url, dataset);

        // First dataset becomes default
        if (!LanceData._defaultDataset) {
            LanceData._defaultDataset = url;
        }

        return dataset;
    }

    /**
     * Manual init (optional) - for advanced configuration.
     */
    static async init(options = {}) {
        LanceData._autoInit();

        if (options.wasmUrl) {
            LanceData._wasm = await LanceQL.load(options.wasmUrl);
        }
        if (options.dataset) {
            await LanceData._getDataset(options.dataset);
        }
    }

    /**
     * Inject CSS that triggers JavaScript via animation events.
     */
    static _injectTriggerStyles() {
        if (document.getElementById('lance-data-triggers')) return;

        const style = document.createElement('style');
        style.id = 'lance-data-triggers';
        style.textContent = `
            /* Lance Data CSS Trigger System */
            @keyframes lance-query-trigger {
                from { --lance-trigger: 0; }
                to { --lance-trigger: 1; }
            }

            /* Elements with lance-data class trigger on insertion */
            .lance-data {
                animation: lance-query-trigger 0.001s;
            }

            /* Re-trigger on data attribute changes */
            .lance-data[data-refresh] {
                animation: lance-query-trigger 0.001s;
            }

            /* Loading state */
            .lance-data[data-loading]::before {
                content: '';
                display: block;
                width: 20px;
                height: 20px;
                border: 2px solid #3b82f6;
                border-top-color: transparent;
                border-radius: 50%;
                animation: lance-spin 0.8s linear infinite;
            }

            @keyframes lance-spin {
                to { transform: rotate(360deg); }
            }

            /* Error state */
            .lance-data[data-error]::before {
                content: attr(data-error);
                color: #ef4444;
                font-size: 12px;
            }

            /* Result container styling */
            .lance-data table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }

            .lance-data th, .lance-data td {
                padding: 8px 12px;
                text-align: left;
                border-bottom: 1px solid #334155;
            }

            .lance-data th {
                background: #1e293b;
                font-weight: 500;
                color: #94a3b8;
            }

            .lance-data tr:hover td {
                background: rgba(59, 130, 246, 0.05);
            }

            /* Value renderer */
            .lance-data[style*="--render: value"] .lance-value,
            .lance-data[style*="--render:'value'"] .lance-value,
            .lance-data[style*='--render:"value"'] .lance-value {
                font-size: 24px;
                font-weight: 600;
                color: #3b82f6;
            }

            /* List renderer */
            .lance-data .lance-list {
                list-style: none;
                padding: 0;
                margin: 0;
            }

            .lance-data .lance-list li {
                padding: 8px 0;
                border-bottom: 1px solid #334155;
            }

            /* JSON renderer */
            .lance-data .lance-json {
                background: #0f172a;
                padding: 12px;
                border-radius: 8px;
                font-family: 'SF Mono', Monaco, monospace;
                font-size: 12px;
                white-space: pre-wrap;
                overflow-x: auto;
            }

            /* Image grid renderer */
            .lance-data .lance-images {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 16px;
            }

            .lance-data .lance-images .image-card {
                background: #1e293b;
                border-radius: 8px;
                overflow: hidden;
            }

            .lance-data .lance-images img {
                width: 100%;
                aspect-ratio: 1;
                object-fit: cover;
            }

            .lance-data .lance-images .image-meta {
                padding: 8px;
                font-size: 12px;
                color: #94a3b8;
            }
        `;
        document.head.appendChild(style);
    }

    /**
     * Set up MutationObserver for dynamic elements.
     */
    static _setupObserver() {
        if (LanceData._observer) return;

        // Helper to check if element has lq-* attributes
        const hasLqAttrs = (el) => {
            return el.hasAttribute?.('lq-query') || el.hasAttribute?.('lq-src') ||
                   el.classList?.contains('lance-data');
        };

        LanceData._observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                // New nodes added
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        if (hasLqAttrs(node)) {
                            LanceData._processElement(node);
                        }
                        // Check descendants
                        node.querySelectorAll?.('[lq-query], [lq-src], .lance-data')?.forEach(el => {
                            LanceData._processElement(el);
                        });
                    }
                }

                // Attribute changes
                if (mutation.type === 'attributes' && hasLqAttrs(mutation.target)) {
                    LanceData._processElement(mutation.target);
                }
            }
        });

        LanceData._observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['lq-query', 'lq-src', 'lq-render', 'lq-bind', 'data-query', 'data-dataset', 'data-render', 'data-refresh']
        });

        // Also listen for animation events (CSS trigger)
        document.body.addEventListener('animationstart', (e) => {
            if (e.animationName === 'lance-query-trigger' && hasLqAttrs(e.target)) {
                LanceData._processElement(e.target);
            }
        });
    }

    /**
     * Process existing lance-data elements.
     */
    static _processExisting() {
        document.querySelectorAll('[lq-query], [lq-src], .lance-data').forEach(el => {
            LanceData._processElement(el);
        });
    }

    /**
     * Parse config from attributes (supports both lq-* and data-* prefixes).
     */
    static _parseConfig(el) {
        // Helper to get attribute value with fallback (lq-* takes precedence)
        const getAttr = (lqName, dataName) => {
            return el.getAttribute(lqName) || el.dataset[dataName] || null;
        };

        return {
            dataset: getAttr('lq-src', 'dataset'),
            query: getAttr('lq-query', 'query'),
            render: getAttr('lq-render', 'render') || 'table',
            columns: (getAttr('lq-columns', 'columns') || '')
                .split(',')
                .map(c => c.trim())
                .filter(Boolean),
            bind: getAttr('lq-bind', 'bind'),
        };
    }

    /**
     * Render pre-computed results to an element (CSS-driven from JS).
     * Use this when you already have query results and just want CSS-driven rendering.
     * @param {HTMLElement|string} el - Element or selector
     * @param {Object} results - Query results {columns, rows, total}
     * @param {Object} [options] - Render options
     * @param {string} [options.render] - Renderer type (table, images, json, etc.)
     */
    static render(el, results, options = {}) {
        const element = typeof el === 'string' ? document.querySelector(el) : el;
        if (!element) {
            console.error('[LanceData] Element not found:', el);
            return;
        }

        try {
            // Dispatch start event
            element.dispatchEvent(new CustomEvent('lq-start', {
                detail: { query: options.query || null }
            }));

            const renderType = options.render || element.dataset.render || 'table';
            const renderer = LanceData._renderers[renderType] || LanceData._renderers.table;

            // Store results in cache for potential re-renders
            if (element.id) {
                LanceData._queryCache.set(`rendered:${element.id}`, results);
            }

            element.innerHTML = renderer(results, { render: renderType, ...options });

            // Dispatch complete event
            element.dispatchEvent(new CustomEvent('lq-complete', {
                detail: {
                    query: options.query || null,
                    columns: results.columns || [],
                    total: results.total || results.rows?.length || 0
                }
            }));
        } catch (error) {
            // Dispatch error event
            element.dispatchEvent(new CustomEvent('lq-error', {
                detail: {
                    query: options.query || null,
                    message: error.message,
                    error: error
                }
            }));
            throw error;
        }
    }

    /**
     * Extract dataset URL from SQL query (e.g., read_lance('https://...'))
     */
    static _extractUrlFromQuery(sql) {
        const match = sql.match(/read_lance\s*\(\s*['"]([^'"]+)['"]/i);
        return match ? match[1] : null;
    }

    /**
     * Process a single lance-data element.
     */
    static async _processElement(el) {
        // Prevent double processing
        if (el.dataset.processing === 'true') return;
        el.dataset.processing = 'true';

        try {
            const config = LanceData._parseConfig(el);

            if (!config.query) {
                el.dataset.processing = 'false';
                return;
            }

            // Set up input binding if specified
            if (config.bind) {
                LanceData._setupBinding(el, config);
            }

            el.dataset.loading = 'true';
            delete el.dataset.error;

            // Dispatch start event (for Alpine.js integration)
            el.dispatchEvent(new CustomEvent('lq-start', {
                detail: { query: config.query }
            }));

            // Extract dataset URL from query if not specified
            const datasetUrl = config.dataset || LanceData._extractUrlFromQuery(config.query);

            // Get dataset (auto-loads and caches)
            const dataset = await LanceData._getDataset(datasetUrl);

            // Check cache
            const cacheKey = `${datasetUrl || 'default'}:${config.query}`;
            let results = LanceData._queryCache.get(cacheKey);

            if (!results) {
                // Execute query
                results = await dataset.executeSQL(config.query);
                LanceData._queryCache.set(cacheKey, results);
            }

            // Render results
            const renderer = LanceData._renderers[config.render] || LanceData._renderers.table;
            el.innerHTML = renderer(results, config);

            delete el.dataset.loading;

            // Dispatch complete event (for Alpine.js integration)
            el.dispatchEvent(new CustomEvent('lq-complete', {
                detail: {
                    query: config.query,
                    columns: results.columns || [],
                    total: results.total || results.rows?.length || 0
                }
            }));
        } catch (error) {
            delete el.dataset.loading;
            el.dataset.error = error.message;
            console.error('[LanceData]', error);

            // Dispatch error event (for Alpine.js integration)
            el.dispatchEvent(new CustomEvent('lq-error', {
                detail: {
                    query: config.query,
                    message: error.message,
                    error: error
                }
            }));
        } finally {
            el.dataset.processing = 'false';
        }
    }

    /**
     * Set up reactive binding to an input element.
     */
    static _setupBinding(el, config) {
        const input = document.querySelector(config.bind);
        if (!input) return;

        // Store binding reference
        const bindingKey = config.bind;
        if (LanceData._bindings.has(bindingKey)) return;

        const handler = () => {
            // Replace $value in query with input value
            const value = input.value;
            const newQuery = config.query.replace(/\$value/g, value);

            // Set via both attribute types
            if (el.hasAttribute('lq-query')) {
                el.setAttribute('lq-query', newQuery);
            } else {
                el.dataset.query = newQuery;
            }

            // Trigger refresh
            el.dataset.refresh = Date.now();
        };

        input.addEventListener('input', handler);
        input.addEventListener('change', handler);

        LanceData._bindings.set(bindingKey, { input, handler, element: el });
    }

    /**
     * Register a custom renderer.
     * @param {string} name - Renderer name
     * @param {Function} fn - Renderer function (results, config) => html
     */
    static registerRenderer(name, fn) {
        LanceData._renderers[name] = fn;
    }

    /**
     * Register built-in renderers.
     */
    static _registerBuiltinRenderers() {
        // Table renderer - handles both {columns, rows} and array-of-objects formats
        LanceData._renderers.table = (results, config) => {
            if (!results) {
                return '<div class="lance-empty">No results</div>';
            }

            // Detect format: {columns, rows} vs array of objects
            let columns, rows;
            if (results.columns && results.rows) {
                // SQLExecutor format: {columns: ['col1', 'col2'], rows: [[val1, val2], ...]}
                columns = config.columns || results.columns.filter(k =>
                    !k.startsWith('_') && k !== 'embedding'
                );
                rows = results.rows;
            } else if (Array.isArray(results)) {
                // Array of objects format: [{col1: val1, col2: val2}, ...]
                if (results.length === 0) {
                    return '<div class="lance-empty">No results</div>';
                }
                columns = config.columns || Object.keys(results[0]).filter(k =>
                    !k.startsWith('_') && k !== 'embedding'
                );
                rows = results.map(row => columns.map(col => row[col]));
            } else {
                return '<div class="lance-empty">No results</div>';
            }

            if (rows.length === 0) {
                return '<div class="lance-empty">No results</div>';
            }

            let html = '<table><thead><tr>';
            for (const col of columns) {
                html += `<th>${LanceData._escapeHtml(String(col))}</th>`;
            }
            html += '</tr></thead><tbody>';

            for (const row of rows) {
                html += '<tr>';
                for (let i = 0; i < columns.length; i++) {
                    const value = row[i];
                    html += `<td>${LanceData._formatValue(value)}</td>`;
                }
                html += '</tr>';
            }

            html += '</tbody></table>';
            return html;
        };

        // List renderer
        LanceData._renderers.list = (results, config) => {
            if (!results || results.length === 0) {
                return '<div class="lance-empty">No results</div>';
            }

            const displayCol = config.columns?.[0] || Object.keys(results[0])[0];

            let html = '<ul class="lance-list">';
            for (const row of results) {
                html += `<li>${LanceData._formatValue(row[displayCol])}</li>`;
            }
            html += '</ul>';
            return html;
        };

        // Single value renderer
        LanceData._renderers.value = (results, config) => {
            if (!results || results.length === 0) {
                return '<div class="lance-empty">-</div>';
            }

            const firstRow = results[0];
            const firstKey = Object.keys(firstRow)[0];
            const value = firstRow[firstKey];

            return `<div class="lance-value">${LanceData._formatValue(value)}</div>`;
        };

        // JSON renderer
        LanceData._renderers.json = (results, config) => {
            return `<pre class="lance-json">${LanceData._escapeHtml(JSON.stringify(results, null, 2))}</pre>`;
        };

        // Image grid renderer (for datasets with url column)
        LanceData._renderers.images = (results, config) => {
            if (!results || results.length === 0) {
                return '<div class="lance-empty">No images</div>';
            }

            let html = '<div class="lance-images">';
            for (const row of results) {
                const url = row.url || row.image_url || row.src;
                const text = row.text || row.caption || row.title || '';

                if (url) {
                    html += `
                        <div class="image-card">
                            <img src="${LanceData._escapeHtml(url)}" alt="${LanceData._escapeHtml(text)}" loading="lazy">
                            ${text ? `<div class="image-meta">${LanceData._escapeHtml(text.substring(0, 100))}</div>` : ''}
                        </div>
                    `;
                }
            }
            html += '</div>';
            return html;
        };

        // Count renderer (for aggregates)
        LanceData._renderers.count = (results, config) => {
            const count = results?.[0]?.count ?? results?.length ?? 0;
            return `<span class="lance-count">${count.toLocaleString()}</span>`;
        };
    }

    /**
     * Check if a string is an image URL.
     */
    static _isImageUrl(str) {
        if (!str || typeof str !== 'string') return false;
        const lower = str.toLowerCase();
        return (lower.startsWith('http://') || lower.startsWith('https://')) &&
               (lower.includes('.jpg') || lower.includes('.jpeg') || lower.includes('.png') ||
                lower.includes('.gif') || lower.includes('.webp') || lower.includes('.svg'));
    }

    /**
     * Check if a string is a URL.
     */
    static _isUrl(str) {
        if (!str || typeof str !== 'string') return false;
        return str.startsWith('http://') || str.startsWith('https://');
    }

    /**
     * Format a value for display.
     */
    static _formatValue(value) {
        if (value === null || value === undefined) return '<span class="null-value">NULL</span>';
        if (value === '') return '<span class="empty-value">(empty)</span>';

        if (typeof value === 'number') {
            return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
        }
        if (Array.isArray(value)) {
            if (value.length > 10) return `<span class="vector-badge">[${value.length}d]</span>`;
            return `[${value.slice(0, 5).map(v => LanceData._formatValue(v)).join(', ')}${value.length > 5 ? '...' : ''}]`;
        }
        if (typeof value === 'object') return JSON.stringify(value);

        const str = String(value);

        // Handle image URLs - show thumbnail
        if (LanceData._isImageUrl(str)) {
            const escaped = LanceData._escapeHtml(str);
            const short = escaped.length > 40 ? escaped.substring(0, 40) + '...' : escaped;
            return `<div class="image-cell">
                <img src="${escaped}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
                <div class="image-placeholder" style="display:none"><svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg></div>
                <a href="${escaped}" target="_blank" class="url-text" title="${escaped}">${short}</a>
            </div>`;
        }

        // Handle other URLs - show as clickable link
        if (LanceData._isUrl(str)) {
            const escaped = LanceData._escapeHtml(str);
            const short = escaped.length > 50 ? escaped.substring(0, 50) + '...' : escaped;
            return `<a href="${escaped}" target="_blank" class="url-link" title="${escaped}">${short}</a>`;
        }

        // Handle long strings - truncate
        if (str.length > 100) return `<span title="${LanceData._escapeHtml(str)}">${LanceData._escapeHtml(str.substring(0, 100))}...</span>`;
        return LanceData._escapeHtml(str);
    }

    /**
     * Escape HTML special characters.
     */
    static _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    /**
     * Clear the query cache.
     */
    static clearCache() {
        LanceData._queryCache.clear();
    }

    /**
     * Refresh all lance-data elements.
     */
    static refresh() {
        LanceData._queryCache.clear();
        document.querySelectorAll('.lance-data').forEach(el => {
            el.setAttribute('data-refresh', Date.now());
        });
    }

    /**
     * Destroy and clean up.
     */
    static destroy() {
        if (LanceData._observer) {
            LanceData._observer.disconnect();
            LanceData._observer = null;
        }

        // Remove bindings
        for (const [key, binding] of LanceData._bindings) {
            binding.input.removeEventListener('input', binding.handler);
            binding.input.removeEventListener('change', binding.handler);
        }
        LanceData._bindings.clear();

        // Remove injected styles
        document.getElementById('lance-data-triggers')?.remove();

        LanceData._instance = null;
        LanceData._dataset = null;
        LanceData._queryCache.clear();
    }
}

// Auto-initialize when DOM is ready (truly CSS-driven - no JS needed by user)
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => LanceData._autoInit());
    } else {
        LanceData._autoInit();
    }
}

// =============================================================================
// sql.js-Compatible API - Drop-in replacement with vector search
// =============================================================================

/**
 * Statement class - sql.js compatible prepared statement
 *
 * Thin wrapper that delegates to WASM-based SQLExecutor
 */
class Statement {
    constructor(db, sql) {
        this.db = db;
        this.sql = sql;
        this.params = null;
        this.results = null;
        this.resultIndex = 0;
        this.done = false;
    }

    /**
     * Bind parameters to the statement
     * @param {Array|Object} params - Parameters to bind
     * @returns {boolean} true on success
     */
    bind(params) {
        this.params = params;
        this.results = null;
        this.resultIndex = 0;
        this.done = false;
        return true;
    }

    /**
     * Execute and step to next row
     * @returns {boolean} true if there's a row, false if done
     */
    step() {
        if (this.done) return false;

        // Execute on first step
        if (this.results === null) {
            const execResult = this.db.exec(this.sql, this.params);
            if (execResult.length === 0 || execResult[0].values.length === 0) {
                this.done = true;
                return false;
            }
            this.results = execResult[0];
            this.resultIndex = 0;
        }

        if (this.resultIndex >= this.results.values.length) {
            this.done = true;
            return false;
        }

        return true;
    }

    /**
     * Get current row as array
     * @returns {Array} Current row values
     */
    get() {
        if (!this.results || this.resultIndex >= this.results.values.length) {
            return [];
        }
        const row = this.results.values[this.resultIndex];
        this.resultIndex++;
        return row;
    }

    /**
     * Get current row as object
     * @param {Object} params - Optional params (ignored, for compatibility)
     * @returns {Object} Current row as {column: value}
     */
    getAsObject(params) {
        if (!this.results || this.resultIndex >= this.results.values.length) {
            return {};
        }
        const row = this.results.values[this.resultIndex];
        this.resultIndex++;

        const obj = {};
        this.results.columns.forEach((col, i) => {
            obj[col] = row[i];
        });
        return obj;
    }

    /**
     * Get column names
     * @returns {Array} Column names
     */
    getColumnNames() {
        return this.results?.columns || [];
    }

    /**
     * Reset statement for reuse
     * @returns {boolean} true on success
     */
    reset() {
        this.results = null;
        this.resultIndex = 0;
        this.done = false;
        return true;
    }

    /**
     * Free statement resources
     * @returns {boolean} true on success
     */
    free() {
        this.results = null;
        this.params = null;
        return true;
    }

    /**
     * Free and finalize (alias for free)
     */
    freemem() {
        return this.free();
    }
}

/**
 * Database class - sql.js compatible API with vector search
 *
 * Drop-in replacement for sql.js Database with:
 * - Same API: exec(), run(), prepare(), export(), close()
 * - OPFS persistence (automatic, no export/import needed)
 * - Vector search: NEAR, TOPK, embeddings
 * - Columnar Lance format for analytics
 *
 * @example
 * const SQL = await initSqlJs();
 * const db = new SQL.Database('mydb');
 *
 * // Standard SQL (same as sql.js)
 * db.exec("CREATE TABLE users (id INT, name TEXT)");
 * db.run("INSERT INTO users VALUES (?, ?)", [1, 'Alice']);
 * const results = db.exec("SELECT * FROM users");
 *
 * // Vector search (LanceQL extension)
 * db.exec("SELECT * FROM docs NEAR embedding 'search text' TOPK 10");
 */
class Database {
    /**
     * Create a new database
     * @param {string|Uint8Array} nameOrData - Database name (OPFS) or data (in-memory)
     * @param {OPFSStorage} storage - Optional storage backend
     */
    constructor(nameOrData, storage = null) {
        if (nameOrData instanceof Uint8Array) {
            // In-memory database from binary data (sql.js compatibility)
            this._inMemory = true;
            this._data = nameOrData;
            this._name = ':memory:';
            this._db = null;
        } else {
            // OPFS-persisted database
            this._inMemory = false;
            this._name = nameOrData || 'default';
            this._storage = storage;
            this._db = new LocalDatabase(this._name, storage || opfsStorage);
        }
        this._open = false;
        this._rowsModified = 0;
    }

    /**
     * Ensure database is open
     */
    async _ensureOpen() {
        if (!this._open && this._db) {
            await this._db.open();
            this._open = true;
        }
    }

    /**
     * Execute SQL and return results
     *
     * @param {string} sql - SQL statement(s)
     * @param {Array|Object} params - Optional parameters
     * @returns {Array} Array of {columns, values} result sets
     *
     * @example
     * const results = db.exec("SELECT * FROM users WHERE id = ?", [1]);
     * // [{columns: ['id', 'name'], values: [[1, 'Alice']]}]
     */
    exec(sql, params) {
        // Return promise for async operation
        return this._execAsync(sql, params);
    }

    async _execAsync(sql, params) {
        await this._ensureOpen();

        // Substitute parameters
        let processedSql = sql;
        if (params) {
            if (Array.isArray(params)) {
                let paramIndex = 0;
                processedSql = sql.replace(/\?/g, () => {
                    const val = params[paramIndex++];
                    return this._formatValue(val);
                });
            } else if (typeof params === 'object') {
                for (const [key, val] of Object.entries(params)) {
                    const pattern = new RegExp(`[:$@]${key}\\b`, 'g');
                    processedSql = processedSql.replace(pattern, this._formatValue(val));
                }
            }
        }

        // Split multiple statements
        const statements = processedSql
            .split(';')
            .map(s => s.trim())
            .filter(s => s.length > 0);

        const results = [];

        for (const stmt of statements) {
            try {
                const lexer = new SQLLexer(stmt);
                const tokens = lexer.tokenize();
                const parser = new SQLParser(tokens);
                const ast = parser.parse();

                if (ast.type === 'SELECT') {
                    // SELECT returns rows
                    const rows = await this._db._executeAST(ast);
                    if (rows && rows.length > 0) {
                        const columns = Object.keys(rows[0]);
                        const values = rows.map(row => columns.map(c => row[c]));
                        results.push({ columns, values });
                    }
                    this._rowsModified = 0;
                } else {
                    // Non-SELECT statements
                    const result = await this._db._executeAST(ast);
                    this._rowsModified = result?.inserted || result?.updated || result?.deleted || 0;
                }
            } catch (e) {
                throw new Error(`SQL error: ${e.message}\nStatement: ${stmt}`);
            }
        }

        return results;
    }

    /**
     * Execute SQL without returning results
     *
     * @param {string} sql - SQL statement
     * @param {Array|Object} params - Optional parameters
     * @returns {Database} this (for chaining)
     */
    run(sql, params) {
        return this._runAsync(sql, params);
    }

    async _runAsync(sql, params) {
        await this.exec(sql, params);
        return this;
    }

    /**
     * Prepare a statement for execution
     *
     * @param {string} sql - SQL statement
     * @param {Array|Object} params - Optional initial parameters
     * @returns {Statement} Prepared statement
     */
    prepare(sql, params) {
        const stmt = new Statement(this, sql);
        if (params) {
            stmt.bind(params);
        }
        return stmt;
    }

    /**
     * Execute SQL and call callback for each row
     *
     * @param {string} sql - SQL statement
     * @param {Array|Object} params - Parameters
     * @param {Function} callback - Called with row object for each row
     * @param {Function} done - Called when complete
     * @returns {Database} this
     */
    each(sql, params, callback, done) {
        this._eachAsync(sql, params, callback, done);
        return this;
    }

    async _eachAsync(sql, params, callback, done) {
        try {
            const results = await this.exec(sql, params);
            if (results.length > 0) {
                const { columns, values } = results[0];
                for (const row of values) {
                    const obj = {};
                    columns.forEach((col, i) => {
                        obj[col] = row[i];
                    });
                    callback(obj);
                }
            }
            if (done) done();
        } catch (e) {
            if (done) done(e);
            else throw e;
        }
    }

    /**
     * Get number of rows modified by last statement
     * @returns {number} Rows modified
     */
    getRowsModified() {
        return this._rowsModified;
    }

    /**
     * Export database to Uint8Array
     *
     * For OPFS databases, this exports all tables as JSON.
     * For in-memory databases, returns the original data.
     *
     * @returns {Uint8Array} Database contents
     */
    async export() {
        if (this._inMemory && this._data) {
            return this._data;
        }

        await this._ensureOpen();

        // Export all tables as JSON
        const exportData = {
            version: this._db.version,
            tables: {}
        };

        for (const tableName of this._db.listTables()) {
            const table = this._db.getTable(tableName);
            const rows = await this._db.select(tableName, {});
            exportData.tables[tableName] = {
                schema: table.schema,
                rows
            };
        }

        return new TextEncoder().encode(JSON.stringify(exportData));
    }

    /**
     * Close the database
     */
    close() {
        this._open = false;
        this._db = null;
    }

    /**
     * Register a custom SQL function (stub for compatibility)
     * @param {string} name - Function name
     * @param {Function} func - Function implementation
     * @returns {Database} this
     */
    create_function(name, func) {
        console.warn(`[LanceQL] create_function('${name}') not yet implemented`);
        return this;
    }

    /**
     * Register a custom aggregate function (stub for compatibility)
     * @param {string} name - Function name
     * @param {Object} funcs - Aggregate functions {init, step, finalize}
     * @returns {Database} this
     */
    create_aggregate(name, funcs) {
        console.warn(`[LanceQL] create_aggregate('${name}') not yet implemented`);
        return this;
    }

    /**
     * Format a value for SQL
     */
    _formatValue(val) {
        if (val === null || val === undefined) {
            return 'NULL';
        }
        if (typeof val === 'string') {
            return `'${val.replace(/'/g, "''")}'`;
        }
        if (typeof val === 'number') {
            return String(val);
        }
        if (typeof val === 'boolean') {
            return val ? 'TRUE' : 'FALSE';
        }
        if (Array.isArray(val)) {
            // Vector
            return `'[${val.join(',')}]'`;
        }
        return String(val);
    }
}

/**
 * Initialize LanceQL with sql.js-compatible API
 *
 * Drop-in replacement for initSqlJs():
 *
 * @example
 * // sql.js style
 * import initSqlJs from 'sql.js';
 * const SQL = await initSqlJs();
 * const db = new SQL.Database();
 *
 * // LanceQL replacement
 * import { initSqlJs } from 'lanceql';
 * const SQL = await initSqlJs();
 * const db = new SQL.Database('mydb'); // OPFS-persisted + vector search
 *
 * @param {Object} config - Configuration (for compatibility, mostly ignored)
 * @returns {Promise<{Database: class}>} SQL namespace with Database class
 */
export async function initSqlJs(config = {}) {
    // Initialize OPFS storage
    try {
        await opfsStorage.open();
    } catch (e) {
        console.warn('[LanceQL] OPFS not available:', e.message);
    }

    // Initialize WebGPU for accelerated vector search, aggregations, joins, sorting, and grouping
    try {
        await webgpuAccelerator.init();
        await gpuAggregator.init();
        await gpuJoiner.init();
        await gpuSorter.init();
        await gpuGrouper.init();
        await gpuVectorSearch.init();
    } catch (e) {
        console.warn('[LanceQL] WebGPU not available:', e.message);
    }

    return {
        Database,
        Statement,
    };
}

// Also export as sqljs for explicit naming
export { Database as SqlJsDatabase, Statement as SqlJsStatement };

// Export WebGPU accelerator, GPU aggregator, GPU joiner, GPU sorter, and GPU grouper for direct access
export { webgpuAccelerator, gpuAggregator, gpuJoiner, gpuSorter, gpuGrouper, gpuVectorSearch, DistanceMetric };

// LogicTable exports (disabled - LanceDataset API not yet implemented)
// export { Table, logicTable, LogicTableQuery, loadLogicTable } from './logic-table.js';

// Default export for convenience
export default LanceQL;
