/**
 * GPU-accelerated SQL Aggregations
 *
 * Provides GPU-based SUM, COUNT, AVG, MIN, MAX for large datasets.
 * Falls back to CPU for small datasets where GPU overhead exceeds benefit.
 */

// Reduction shader - embedded for bundler compatibility
const REDUCE_SHADER = `
struct ReduceParams {
    size: u32,
    workgroups: u32,
}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_sum(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(0.0, input[gid.x], gid.x < params.size);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] += shared_data[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { output[wid.x] = shared_data[0]; }
}

@compute @workgroup_size(256)
fn reduce_sum_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(0.0, input[tid], tid < params.workgroups);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] += shared_data[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { output[0] = shared_data[0]; }
}

@compute @workgroup_size(256)
fn reduce_min(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(3.4e+38, input[gid.x], gid.x < params.size);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] = min(shared_data[tid], shared_data[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { output[wid.x] = shared_data[0]; }
}

@compute @workgroup_size(256)
fn reduce_min_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(3.4e+38, input[tid], tid < params.workgroups);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] = min(shared_data[tid], shared_data[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { output[0] = shared_data[0]; }
}

@compute @workgroup_size(256)
fn reduce_max(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(-3.4e+38, input[gid.x], gid.x < params.size);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] = max(shared_data[tid], shared_data[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { output[wid.x] = shared_data[0]; }
}

@compute @workgroup_size(256)
fn reduce_max_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(-3.4e+38, input[tid], tid < params.workgroups);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] = max(shared_data[tid], shared_data[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { output[0] = shared_data[0]; }
}
`;

// Minimum rows to benefit from GPU acceleration
// GPU has overhead - use typed arrays for medium data, GPU for large
const GPU_THRESHOLD = 10000;

/**
 * GPU Aggregator for SQL operations
 */
export class GPUAggregator {
    constructor() {
        this.device = null;
        this.pipelines = new Map();
        this.available = false;
    }

    /**
     * Initialize WebGPU device and compile shaders.
     * @returns {Promise<boolean>} Whether GPU is available
     */
    async init() {
        if (this.device) return this.available;

        if (typeof navigator === 'undefined' || !navigator.gpu) {
            console.log('[GPUAggregator] WebGPU not available');
            return false;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.log('[GPUAggregator] No WebGPU adapter');
                return false;
            }

            this.device = await adapter.requestDevice({
                requiredLimits: {
                    maxStorageBufferBindingSize: 256 * 1024 * 1024,  // 256MB
                    maxBufferSize: 256 * 1024 * 1024,
                },
            });

            await this._compileShaders();
            this.available = true;
            console.log('[GPUAggregator] Initialized');
            return true;
        } catch (e) {
            console.error('[GPUAggregator] Init failed:', e);
            return false;
        }
    }

    /**
     * Check if GPU aggregation is available.
     * @returns {boolean}
     */
    isAvailable() {
        return this.available;
    }

    /**
     * Compile reduction shaders.
     * @private
     */
    async _compileShaders() {
        const module = this.device.createShaderModule({ code: REDUCE_SHADER });

        // Create pipelines for each reduction type
        for (const op of ['sum', 'min', 'max']) {
            // First pass pipeline
            this.pipelines.set(`reduce_${op}`, this.device.createComputePipeline({
                layout: 'auto',
                compute: { module, entryPoint: `reduce_${op}` },
            }));

            // Final pass pipeline
            this.pipelines.set(`reduce_${op}_final`, this.device.createComputePipeline({
                layout: 'auto',
                compute: { module, entryPoint: `reduce_${op}_final` },
            }));
        }
    }

    /**
     * Compute SUM of numeric values.
     * @param {number[]|Float32Array|Float64Array} values - Input values
     * @returns {Promise<number>} Sum result
     */
    async sum(values) {
        if (values.length < GPU_THRESHOLD || !this.available) {
            return this._cpuSum(values);
        }
        return this._gpuReduce(values, 'sum');
    }

    /**
     * Compute COUNT of non-null values.
     * @param {any[]} values - Input values (nulls excluded)
     * @returns {number} Count result
     */
    count(values) {
        // COUNT is always fast on CPU - just return length
        return values.length;
    }

    /**
     * Compute AVG of numeric values.
     * @param {number[]|Float32Array|Float64Array} values - Input values
     * @returns {Promise<number>} Average result
     */
    async avg(values) {
        if (values.length === 0) return null;
        const sum = await this.sum(values);
        return sum / values.length;
    }

    /**
     * Compute MIN of numeric values.
     * @param {number[]|Float32Array|Float64Array} values - Input values
     * @returns {Promise<number>} Minimum result
     */
    async min(values) {
        if (values.length === 0) return null;
        if (values.length < GPU_THRESHOLD || !this.available) {
            return this._cpuMin(values);
        }
        return this._gpuReduce(values, 'min');
    }

    /**
     * Compute MAX of numeric values.
     * @param {number[]|Float32Array|Float64Array} values - Input values
     * @returns {Promise<number>} Maximum result
     */
    async max(values) {
        if (values.length === 0) return null;
        if (values.length < GPU_THRESHOLD || !this.available) {
            return this._cpuMax(values);
        }
        return this._gpuReduce(values, 'max');
    }

    /**
     * Batch compute multiple aggregations on the same column.
     * More efficient than individual calls.
     * @param {number[]} values - Input values
     * @param {string[]} ops - Operations to compute: 'sum', 'min', 'max', 'count', 'avg'
     * @returns {Promise<Object>} Results keyed by operation
     */
    async batch(values, ops) {
        const results = {};

        // Compute each requested aggregation
        for (const op of ops) {
            switch (op) {
                case 'sum':
                    results.sum = await this.sum(values);
                    break;
                case 'count':
                    results.count = await this.count(values);
                    break;
                case 'avg':
                    // Optimize: reuse sum if already computed
                    if (results.sum !== undefined) {
                        results.avg = values.length > 0 ? results.sum / values.length : null;
                    } else {
                        results.avg = await this.avg(values);
                    }
                    break;
                case 'min':
                    results.min = await this.min(values);
                    break;
                case 'max':
                    results.max = await this.max(values);
                    break;
            }
        }

        return results;
    }

    /**
     * Run GPU reduction.
     * @private
     */
    async _gpuReduce(values, op) {
        const n = values.length;
        const workgroupSize = 256;
        const numWorkgroups = Math.ceil(n / workgroupSize);

        // Convert to Float32Array
        const inputData = values instanceof Float32Array
            ? values
            : new Float32Array(values);

        // Create buffers
        const inputBuffer = this.device.createBuffer({
            size: inputData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(inputBuffer, 0, inputData);

        // Intermediate buffer for partial results
        const partialBuffer = this.device.createBuffer({
            size: numWorkgroups * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Output buffer
        const outputBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Staging buffer for reading results
        const stagingBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // Params buffer for first pass
        const paramsData = new Uint32Array([n, numWorkgroups]);
        const paramsBuffer = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        // First pass: reduce to partial results
        const firstPipeline = this.pipelines.get(`reduce_${op}`);
        const firstBindGroup = this.device.createBindGroup({
            layout: firstPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: inputBuffer } },
                { binding: 2, resource: { buffer: partialBuffer } },
            ],
        });

        // Final pass params
        const finalParamsData = new Uint32Array([numWorkgroups, numWorkgroups]);
        const finalParamsBuffer = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(finalParamsBuffer, 0, finalParamsData);

        // Final pass: reduce partial results to single value
        const finalPipeline = this.pipelines.get(`reduce_${op}_final`);
        const finalBindGroup = this.device.createBindGroup({
            layout: finalPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: finalParamsBuffer } },
                { binding: 1, resource: { buffer: partialBuffer } },
                { binding: 2, resource: { buffer: outputBuffer } },
            ],
        });

        // Encode commands
        const encoder = this.device.createCommandEncoder();

        // First pass
        const pass1 = encoder.beginComputePass();
        pass1.setPipeline(firstPipeline);
        pass1.setBindGroup(0, firstBindGroup);
        pass1.dispatchWorkgroups(numWorkgroups);
        pass1.end();

        // Final pass
        const pass2 = encoder.beginComputePass();
        pass2.setPipeline(finalPipeline);
        pass2.setBindGroup(0, finalBindGroup);
        pass2.dispatchWorkgroups(1);
        pass2.end();

        // Copy result to staging
        encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4);

        this.device.queue.submit([encoder.finish()]);

        // Read result
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(stagingBuffer.getMappedRange())[0];
        stagingBuffer.unmap();

        // Cleanup
        inputBuffer.destroy();
        partialBuffer.destroy();
        outputBuffer.destroy();
        stagingBuffer.destroy();
        paramsBuffer.destroy();
        finalParamsBuffer.destroy();

        return result;
    }

    /**
     * CPU fallback for sum.
     * @private
     */
    _cpuSum(values) {
        let sum = 0;
        for (let i = 0; i < values.length; i++) {
            sum += values[i];
        }
        return sum;
    }

    /**
     * CPU fallback for min.
     * @private
     */
    _cpuMin(values) {
        let min = values[0];
        for (let i = 1; i < values.length; i++) {
            if (values[i] < min) min = values[i];
        }
        return min;
    }

    /**
     * CPU fallback for max.
     * @private
     */
    _cpuMax(values) {
        let max = values[0];
        for (let i = 1; i < values.length; i++) {
            if (values[i] > max) max = values[i];
        }
        return max;
    }

    /**
     * Dispose GPU resources.
     */
    dispose() {
        this.pipelines.clear();
        this.device = null;
        this.available = false;
    }
}

// Singleton instance
let gpuAggregatorInstance = null;

/**
 * Get or create the GPU aggregator instance.
 * @returns {GPUAggregator}
 */
export function getGPUAggregator() {
    if (!gpuAggregatorInstance) {
        gpuAggregatorInstance = new GPUAggregator();
    }
    return gpuAggregatorInstance;
}

/**
 * Check if GPU aggregation is beneficial for a dataset size.
 * @param {number} size - Number of rows
 * @returns {boolean}
 */
export function shouldUseGPU(size) {
    return size >= GPU_THRESHOLD;
}
