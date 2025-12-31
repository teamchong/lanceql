/**
 * GPU Buffer Manager
 *
 * Manages WebGPU buffer allocation, reuse, and lifecycle.
 * Supports lazy loading of model weights to GPU memory.
 */

export class GPUBufferManager {
    constructor(device) {
        this.device = device;
        this.buffers = new Map();  // name → GPUBuffer
        this.uniformBuffers = new Map();
        this.bindGroups = new Map();
        this.pipelines = new Map();

        // Memory tracking
        this.allocatedBytes = 0;
        this.maxBytes = 512 * 1024 * 1024;  // 512MB default limit
    }

    /**
     * Create or get a storage buffer.
     * @param {string} name - Buffer identifier
     * @param {number} size - Size in bytes
     * @param {number} usage - GPUBufferUsage flags
     * @returns {GPUBuffer}
     */
    getOrCreateBuffer(name, size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
        const key = `${name}_${size}`;

        if (this.buffers.has(key)) {
            return this.buffers.get(key);
        }

        const buffer = this.device.createBuffer({
            size,
            usage,
            label: name,
        });

        this.buffers.set(key, buffer);
        this.allocatedBytes += size;

        return buffer;
    }

    /**
     * Create a buffer and upload data.
     * @param {string} name - Buffer identifier
     * @param {Float32Array|Uint32Array} data - Data to upload
     * @param {number} usage - GPUBufferUsage flags
     * @returns {GPUBuffer}
     */
    createBufferWithData(name, data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage,
            label: name,
            mappedAtCreation: true,
        });

        const dst = new (data.constructor)(buffer.getMappedRange());
        dst.set(data);
        buffer.unmap();

        this.buffers.set(name, buffer);
        this.allocatedBytes += data.byteLength;

        return buffer;
    }

    /**
     * Create uniform buffer with struct data.
     * @param {string} name - Buffer identifier
     * @param {ArrayBuffer} data - Uniform data
     * @returns {GPUBuffer}
     */
    createUniformBuffer(name, data) {
        // Align to 16 bytes (WebGPU requirement)
        const alignedSize = Math.ceil(data.byteLength / 16) * 16;

        const buffer = this.device.createBuffer({
            size: alignedSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: `uniform_${name}`,
        });

        this.device.queue.writeBuffer(buffer, 0, data);
        this.uniformBuffers.set(name, buffer);

        return buffer;
    }

    /**
     * Update uniform buffer data.
     * @param {string} name - Buffer identifier
     * @param {ArrayBuffer} data - New data
     */
    updateUniform(name, data) {
        const buffer = this.uniformBuffers.get(name);
        if (buffer) {
            this.device.queue.writeBuffer(buffer, 0, data);
        }
    }

    /**
     * Create bind group for a shader.
     * @param {string} name - Bind group identifier
     * @param {GPUBindGroupLayout} layout - Bind group layout
     * @param {Array} entries - Bind group entries
     * @returns {GPUBindGroup}
     */
    createBindGroup(name, layout, entries) {
        const bindGroup = this.device.createBindGroup({
            layout,
            entries,
            label: name,
        });

        this.bindGroups.set(name, bindGroup);
        return bindGroup;
    }

    /**
     * Read data back from GPU buffer.
     * @param {GPUBuffer} buffer - Source buffer
     * @param {number} size - Bytes to read
     * @returns {Promise<Float32Array>}
     */
    async readBuffer(buffer, size) {
        const readBuffer = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
        this.device.queue.submit([encoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();
        readBuffer.destroy();

        return data;
    }

    /**
     * Free a buffer.
     * @param {string} name - Buffer identifier
     */
    freeBuffer(name) {
        const buffer = this.buffers.get(name);
        if (buffer) {
            this.allocatedBytes -= buffer.size;
            buffer.destroy();
            this.buffers.delete(name);
        }
    }

    /**
     * Free all buffers.
     */
    freeAll() {
        for (const buffer of this.buffers.values()) {
            buffer.destroy();
        }
        for (const buffer of this.uniformBuffers.values()) {
            buffer.destroy();
        }
        this.buffers.clear();
        this.uniformBuffers.clear();
        this.bindGroups.clear();
        this.allocatedBytes = 0;
    }

    /**
     * Get memory usage info.
     */
    getMemoryInfo() {
        return {
            allocated: this.allocatedBytes,
            max: this.maxBytes,
            usage: this.allocatedBytes / this.maxBytes,
            bufferCount: this.buffers.size,
        };
    }
}

/**
 * Model weight storage on GPU.
 * Handles lazy loading and caching of model weights.
 */
export class ModelWeightCache {
    constructor(bufferManager) {
        this.bufferManager = bufferManager;
        this.loadedModels = new Map();  // modelId → { buffers, metadata }
        this.loading = new Map();  // modelId → Promise
    }

    /**
     * Check if model is loaded.
     * @param {string} modelId - Model identifier
     * @returns {boolean}
     */
    isLoaded(modelId) {
        return this.loadedModels.has(modelId);
    }

    /**
     * Get loaded model buffers.
     * @param {string} modelId - Model identifier
     * @returns {Object|null}
     */
    getModel(modelId) {
        return this.loadedModels.get(modelId) || null;
    }

    /**
     * Load model weights to GPU (lazy, cached).
     * @param {string} modelId - Model identifier
     * @param {Function} loadFn - Async function that returns weight tensors
     * @returns {Promise<Object>}
     */
    async loadModel(modelId, loadFn) {
        // Already loaded
        if (this.loadedModels.has(modelId)) {
            return this.loadedModels.get(modelId);
        }

        // Currently loading
        if (this.loading.has(modelId)) {
            return this.loading.get(modelId);
        }

        // Start loading
        const loadPromise = (async () => {
            console.log(`[ModelWeightCache] Loading model: ${modelId}`);
            const startTime = performance.now();

            const { weights, metadata } = await loadFn();

            // Upload weights to GPU
            const gpuBuffers = {};
            for (const [name, data] of Object.entries(weights)) {
                gpuBuffers[name] = this.bufferManager.createBufferWithData(
                    `${modelId}_${name}`,
                    data
                );
            }

            const loadTime = performance.now() - startTime;
            console.log(`[ModelWeightCache] Loaded ${modelId} in ${loadTime.toFixed(0)}ms`);

            const model = { buffers: gpuBuffers, metadata };
            this.loadedModels.set(modelId, model);
            this.loading.delete(modelId);

            return model;
        })();

        this.loading.set(modelId, loadPromise);
        return loadPromise;
    }

    /**
     * Unload model from GPU.
     * @param {string} modelId - Model identifier
     */
    unloadModel(modelId) {
        const model = this.loadedModels.get(modelId);
        if (model) {
            for (const [name, buffer] of Object.entries(model.buffers)) {
                this.bufferManager.freeBuffer(`${modelId}_${name}`);
            }
            this.loadedModels.delete(modelId);
            console.log(`[ModelWeightCache] Unloaded model: ${modelId}`);
        }
    }

    /**
     * Unload all models.
     */
    unloadAll() {
        for (const modelId of this.loadedModels.keys()) {
            this.unloadModel(modelId);
        }
    }

    /**
     * Get loaded model IDs.
     * @returns {string[]}
     */
    getLoadedModels() {
        return Array.from(this.loadedModels.keys());
    }
}
