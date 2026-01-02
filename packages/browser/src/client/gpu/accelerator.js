/**
 * WebGPUAccelerator - GPU-accelerated batch cosine similarity
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

// Lazy singleton - only instantiated when first accessed
let _webgpuAccelerator = null;
function getWebGPUAccelerator() {
    if (!_webgpuAccelerator) _webgpuAccelerator = new WebGPUAccelerator();
    return _webgpuAccelerator;
}

export { WebGPUAccelerator, getWebGPUAccelerator };
