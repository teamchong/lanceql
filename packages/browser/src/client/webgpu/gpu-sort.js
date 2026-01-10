/**
 * GPU-accelerated SQL Sorting
 *
 * Provides GPU-based bitonic sort for ORDER BY operations on large result sets.
 * Falls back to CPU for small datasets where GPU overhead exceeds benefit.
 */

// Bitonic sort shaders - embedded for bundler compatibility
const SORT_SHADER = `
// GPU Bitonic Sort Shader
// Implements parallel bitonic sorting network for ORDER BY operations

struct LocalSortParams {
    size: u32,
    stage: u32,
    step: u32,
    ascending: u32,
}

@group(0) @binding(0) var<uniform> local_params: LocalSortParams;
@group(0) @binding(1) var<storage, read_write> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;

var<workgroup> shared_keys: array<f32, 512>;
var<workgroup> shared_indices: array<u32, 512>;

fn compare_swap(i: u32, j: u32, dir: bool) {
    let ki = shared_keys[i];
    let kj = shared_keys[j];
    let should_swap = select(ki > kj, ki < kj, dir);
    if (should_swap) {
        shared_keys[i] = kj;
        shared_keys[j] = ki;
        let ti = shared_indices[i];
        shared_indices[i] = shared_indices[j];
        shared_indices[j] = ti;
    }
}

@compute @workgroup_size(256)
fn local_bitonic_sort(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_size = 512u;
    let base = wid.x * local_size;
    let tid = lid.x;

    let idx1 = base + tid;
    let idx2 = base + tid + 256u;

    if (idx1 < local_params.size) {
        shared_keys[tid] = keys[idx1];
        shared_indices[tid] = indices[idx1];
    } else {
        shared_keys[tid] = 3.4e38;
        shared_indices[tid] = idx1;
    }

    if (idx2 < local_params.size) {
        shared_keys[tid + 256u] = keys[idx2];
        shared_indices[tid + 256u] = indices[idx2];
    } else {
        shared_keys[tid + 256u] = 3.4e38;
        shared_indices[tid + 256u] = idx2;
    }

    workgroupBarrier();

    let ascending = local_params.ascending == 1u;

    for (var stage = 1u; stage < local_size; stage = stage << 1u) {
        for (var step = stage; step > 0u; step = step >> 1u) {
            let pair_distance = step;
            let block_size = step << 1u;
            let pos = tid;
            if (pos < 256u) {
                let block_id = pos / step;
                let in_block = pos % step;
                let i = block_id * block_size + in_block;
                let j = i + pair_distance;
                if (j < local_size) {
                    let dir = ((i / (stage << 1u)) % 2u == 0u) == ascending;
                    compare_swap(i, j, dir);
                }
            }
            workgroupBarrier();
        }
    }

    if (idx1 < local_params.size) {
        keys[idx1] = shared_keys[tid];
        indices[idx1] = shared_indices[tid];
    }
    if (idx2 < local_params.size) {
        keys[idx2] = shared_keys[tid + 256u];
        indices[idx2] = shared_indices[tid + 256u];
    }
}

struct MergeParams {
    size: u32,
    stage: u32,
    step: u32,
    ascending: u32,
}

@group(0) @binding(0) var<uniform> merge_params: MergeParams;
@group(0) @binding(1) var<storage, read_write> merge_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> merge_indices: array<u32>;

@compute @workgroup_size(256)
fn bitonic_merge_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let step = merge_params.step;
    let stage = merge_params.stage;
    let ascending = merge_params.ascending == 1u;

    let block_size = 1u << (stage + 1u);
    let half_block = 1u << stage;

    let block_id = tid / half_block;
    let in_half = tid % half_block;

    let i = block_id * block_size + in_half;
    let j = i + step;

    if (j >= merge_params.size) { return; }

    let dir = ((i / block_size) % 2u == 0u) == ascending;

    let ki = merge_keys[i];
    let kj = merge_keys[j];
    let should_swap = select(ki > kj, ki < kj, dir);

    if (should_swap) {
        merge_keys[i] = kj;
        merge_keys[j] = ki;
        let ti = merge_indices[i];
        merge_indices[i] = merge_indices[j];
        merge_indices[j] = ti;
    }
}

struct InitParams { size: u32 }

@group(0) @binding(0) var<uniform> init_params: InitParams;
@group(0) @binding(1) var<storage, read_write> init_indices: array<u32>;

@compute @workgroup_size(256)
fn init_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < init_params.size) {
        init_indices[gid.x] = gid.x;
    }
}
`;

// Minimum rows to benefit from GPU acceleration
const GPU_SORT_THRESHOLD = 10000;

/**
 * GPU Sorter for SQL ORDER BY operations
 */
export class GPUSorter {
    constructor() {
        this.device = null;
        this.pipelines = new Map();
        this.available = false;
        this._initPromise = null;
    }

    /**
     * Initialize WebGPU device and compile shaders.
     * @returns {Promise<boolean>} Whether GPU is available
     */
    async init() {
        if (this._initPromise) return this._initPromise;
        this._initPromise = this._doInit();
        return this._initPromise;
    }

    async _doInit() {
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            console.log('[GPUSorter] WebGPU not available');
            return false;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.log('[GPUSorter] No WebGPU adapter');
                return false;
            }

            this.device = await adapter.requestDevice({
                requiredLimits: {
                    maxStorageBufferBindingSize: 256 * 1024 * 1024,
                    maxBufferSize: 256 * 1024 * 1024,
                },
            });

            await this._compileShaders();
            this.available = true;
            console.log('[GPUSorter] Initialized');
            return true;
        } catch (e) {
            console.error('[GPUSorter] Init failed:', e);
            return false;
        }
    }

    async _compileShaders() {
        const module = this.device.createShaderModule({ code: SORT_SHADER });

        this.pipelines.set('init', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'init_indices' },
        }));

        this.pipelines.set('local_sort', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'local_bitonic_sort' },
        }));

        this.pipelines.set('merge', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'bitonic_merge_step' },
        }));
    }

    isAvailable() { return this.available; }

    /**
     * Sort an array of numeric values and return sorted indices.
     * @param {Float32Array|number[]} values - Values to sort
     * @param {boolean} ascending - Sort direction (true = ASC, false = DESC)
     * @returns {Promise<Uint32Array>} Sorted indices
     */
    async sort(values, ascending = true) {
        const size = values.length;

        // Use CPU for small arrays
        if (!this.available || size < GPU_SORT_THRESHOLD) {
            return this._cpuSort(values, ascending);
        }

        // Pad to power of 2 for bitonic sort
        const paddedSize = this._nextPowerOf2(size);
        const keys = new Float32Array(paddedSize);
        keys.set(values instanceof Float32Array ? values : new Float32Array(values));
        // Pad with max float (sorts to end)
        for (let i = size; i < paddedSize; i++) {
            keys[i] = 3.4e38;
        }

        // Create GPU buffers
        const keysBuf = this._createBuffer(keys, GPUBufferUsage.STORAGE);
        const indicesBuf = this.device.createBuffer({
            size: paddedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Initialize indices
        const initParamsBuf = this._createUniformBuffer(new Uint32Array([paddedSize]));
        const initPipeline = this.pipelines.get('init');
        const initBindGroup = this.device.createBindGroup({
            layout: initPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: initParamsBuf } },
                { binding: 1, resource: { buffer: indicesBuf } },
            ],
        });

        const encoder = this.device.createCommandEncoder();

        // Init pass
        const initWg = Math.ceil(paddedSize / 256);
        const initPass = encoder.beginComputePass();
        initPass.setPipeline(initPipeline);
        initPass.setBindGroup(0, initBindGroup);
        initPass.dispatchWorkgroups(initWg);
        initPass.end();

        // Local sort pass (within workgroups of 512)
        const localSortPipeline = this.pipelines.get('local_sort');
        const localParamsBuf = this._createUniformBuffer(new Uint32Array([paddedSize, 0, 0, ascending ? 1 : 0]));
        const localBindGroup = this.device.createBindGroup({
            layout: localSortPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: localParamsBuf } },
                { binding: 1, resource: { buffer: keysBuf } },
                { binding: 2, resource: { buffer: indicesBuf } },
            ],
        });

        const localWg = Math.ceil(paddedSize / 512);
        const localPass = encoder.beginComputePass();
        localPass.setPipeline(localSortPipeline);
        localPass.setBindGroup(0, localBindGroup);
        localPass.dispatchWorkgroups(localWg);
        localPass.end();

        this.device.queue.submit([encoder.finish()]);

        // Global merge passes (for sizes > 512)
        if (paddedSize > 512) {
            const mergePipeline = this.pipelines.get('merge');

            // Start from stage 9 (512 elements already sorted locally)
            for (let stageExp = 9; (1 << stageExp) < paddedSize; stageExp++) {
                const stage = 1 << stageExp;

                for (let step = stage; step > 0; step >>= 1) {
                    const mergeEncoder = this.device.createCommandEncoder();
                    const mergeParamsBuf = this._createUniformBuffer(
                        new Uint32Array([paddedSize, stageExp, step, ascending ? 1 : 0])
                    );

                    const mergeBindGroup = this.device.createBindGroup({
                        layout: mergePipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: mergeParamsBuf } },
                            { binding: 1, resource: { buffer: keysBuf } },
                            { binding: 2, resource: { buffer: indicesBuf } },
                        ],
                    });

                    const mergeWg = Math.ceil(paddedSize / 256);
                    const mergePass = mergeEncoder.beginComputePass();
                    mergePass.setPipeline(mergePipeline);
                    mergePass.setBindGroup(0, mergeBindGroup);
                    mergePass.dispatchWorkgroups(mergeWg);
                    mergePass.end();

                    this.device.queue.submit([mergeEncoder.finish()]);
                    mergeParamsBuf.destroy();
                }
            }
        }

        // Read back sorted indices
        const stagingBuf = this.device.createBuffer({
            size: size * 4, // Only read original size
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(indicesBuf, 0, stagingBuf, 0, size * 4);
        this.device.queue.submit([copyEncoder.finish()]);

        await stagingBuf.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(stagingBuf.getMappedRange().slice(0));
        stagingBuf.unmap();

        // Cleanup
        keysBuf.destroy();
        indicesBuf.destroy();
        initParamsBuf.destroy();
        localParamsBuf.destroy();
        stagingBuf.destroy();

        return result;
    }

    /**
     * Sort rows by ORDER BY clause.
     * @param {Object[]} rows - Rows to sort
     * @param {Object[]} orderBy - ORDER BY clauses [{column, direction}]
     * @param {Map<string,number>} colMap - Column name to index mapping
     * @returns {Promise<Object[]>} Sorted rows
     */
    async sortRows(rows, orderBy, colMap) {
        if (rows.length < GPU_SORT_THRESHOLD || !this.available) {
            return this._cpuSortRows(rows, orderBy, colMap);
        }

        // Multi-column sort: stable sort from last to first column
        let indices = new Uint32Array(rows.length);
        for (let i = 0; i < rows.length; i++) indices[i] = i;

        // Process columns in reverse order for stable multi-column sort
        for (let i = orderBy.length - 1; i >= 0; i--) {
            const { column, direction } = orderBy[i];
            const colName = typeof column === 'string' ? column : column.name;
            const ascending = direction !== 'DESC';

            // Extract values for this column in current index order
            const values = new Float32Array(rows.length);
            for (let j = 0; j < rows.length; j++) {
                const row = rows[indices[j]];
                const val = row[colName];
                if (val === null || val === undefined) {
                    values[j] = ascending ? 3.4e38 : -3.4e38; // NULLS LAST
                } else if (typeof val === 'number') {
                    values[j] = val;
                } else if (typeof val === 'string') {
                    // Hash string for sorting (approximate)
                    values[j] = this._stringToSortKey(val);
                } else {
                    values[j] = 0;
                }
            }

            // Sort and get new permutation
            const sortedIndices = await this.sort(values, ascending);

            // Apply permutation to indices
            const newIndices = new Uint32Array(rows.length);
            for (let j = 0; j < rows.length; j++) {
                newIndices[j] = indices[sortedIndices[j]];
            }
            indices = newIndices;
        }

        // Apply final permutation to rows
        const sortedRows = new Array(rows.length);
        for (let i = 0; i < rows.length; i++) {
            sortedRows[i] = rows[indices[i]];
        }
        return sortedRows;
    }

    /**
     * CPU fallback sort
     */
    _cpuSort(values, ascending) {
        const indexed = Array.from(values).map((v, i) => ({ v, i }));
        indexed.sort((a, b) => {
            if (a.v === b.v) return 0;
            const cmp = a.v < b.v ? -1 : 1;
            return ascending ? cmp : -cmp;
        });
        return new Uint32Array(indexed.map(x => x.i));
    }

    /**
     * CPU fallback for row sorting
     */
    _cpuSortRows(rows, orderBy, colMap) {
        const sorted = [...rows];
        sorted.sort((a, b) => {
            for (const { column, direction } of orderBy) {
                const colName = typeof column === 'string' ? column : column.name;
                const aVal = a[colName];
                const bVal = b[colName];

                // NULLS LAST
                if (aVal === null || aVal === undefined) {
                    if (bVal !== null && bVal !== undefined) return 1;
                    continue;
                }
                if (bVal === null || bVal === undefined) return -1;

                let cmp = 0;
                if (typeof aVal === 'string' && typeof bVal === 'string') {
                    cmp = aVal.localeCompare(bVal);
                } else {
                    cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
                }

                if (cmp !== 0) {
                    return direction === 'DESC' ? -cmp : cmp;
                }
            }
            return 0;
        });
        return sorted;
    }

    /**
     * Convert string to numeric sort key (approximate lexicographic order)
     */
    _stringToSortKey(str) {
        // Use first 4 characters as sort key
        let key = 0;
        for (let i = 0; i < Math.min(4, str.length); i++) {
            key = key * 256 + str.charCodeAt(i);
        }
        return key;
    }

    _createBuffer(data, usage) {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: usage | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }

    _createUniformBuffer(data) {
        const buffer = this.device.createBuffer({
            size: Math.max(data.byteLength, 16),
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }

    _nextPowerOf2(n) {
        let p = 1;
        while (p < n) p *= 2;
        return p;
    }

    dispose() {
        this.pipelines.clear();
        this.device = null;
        this.available = false;
    }
}

// Singleton instance
let gpuSorterInstance = null;

/**
 * Get or create the GPU sorter instance.
 * @returns {GPUSorter}
 */
export function getGPUSorter() {
    if (!gpuSorterInstance) {
        gpuSorterInstance = new GPUSorter();
    }
    return gpuSorterInstance;
}

/**
 * Check if GPU sort is beneficial for given row count.
 * @param {number} rowCount - Number of rows
 * @returns {boolean}
 */
export function shouldUseGPUSort(rowCount) {
    return rowCount >= GPU_SORT_THRESHOLD;
}
