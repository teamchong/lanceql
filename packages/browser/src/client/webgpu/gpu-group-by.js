/**
 * GPU-accelerated SQL GROUP BY
 *
 * Provides GPU-based hash grouping and per-group aggregation for large datasets.
 * Falls back to CPU for small datasets where GPU overhead exceeds benefit.
 */

import { getBufferPool } from './gpu-buffer-pool.js';

// Hash grouping shaders - embedded for bundler compatibility
const GROUP_BY_SHADER = `
struct BP { size: u32, cap: u32 }
struct AP { size: u32, cap: u32 }
struct AGP { size: u32, num_groups: u32, agg_type: u32 }
struct IP { cap: u32 }
struct IAP { num_groups: u32, init_value: u32 }

@group(0) @binding(0) var<uniform> bp: BP;
@group(0) @binding(1) var<storage, read> bkeys: array<u32>;
@group(0) @binding(2) var<storage, read_write> ht: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> gc: atomic<u32>;

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
        if (o.exchanged) { atomicStore(&ht[i + 1u], atomicAdd(&gc, 1u)); return; }
        if (o.old_value == k) { return; }
        s = (s + 1u) % bp.cap;
    }
}

@group(0) @binding(0) var<uniform> ap: AP;
@group(0) @binding(1) var<storage, read> akeys: array<u32>;
@group(0) @binding(2) var<storage, read> lt: array<u32>;
@group(0) @binding(3) var<storage, read_write> gids: array<u32>;

@compute @workgroup_size(256)
fn assign(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= ap.size) { return; }
    let k = akeys[g.x];
    var s = fnv(k) % ap.cap;
    for (var p = 0u; p < ap.cap; p++) {
        let i = s * 2u;
        if (lt[i] == k) { gids[g.x] = lt[i + 1u]; return; }
        if (lt[i] == 0xFFFFFFFFu) { gids[g.x] = 0xFFFFFFFFu; return; }
        s = (s + 1u) % ap.cap;
    }
    gids[g.x] = 0xFFFFFFFFu;
}

@group(0) @binding(0) var<uniform> agp: AGP;
@group(0) @binding(1) var<storage, read> agids: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> res: array<atomic<u32>>;

fn f2s(f: f32) -> u32 { let b = bitcast<u32>(f); return select(b ^ 0x80000000u, ~b, (b & 0x80000000u) != 0u); }

@compute @workgroup_size(256)
fn count_agg(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= agp.size) { return; }
    let gid = agids[g.x];
    if (gid < agp.num_groups) { atomicAdd(&res[gid], 1u); }
}

@compute @workgroup_size(256)
fn sum_agg(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= agp.size) { return; }
    let gid = agids[g.x]; let v = vals[g.x];
    if (gid < agp.num_groups && !isNan(v)) { atomicAdd(&res[gid], u32(i32(v * 1000.0))); }
}

@compute @workgroup_size(256)
fn min_agg(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= agp.size) { return; }
    let gid = agids[g.x]; let v = vals[g.x];
    if (gid < agp.num_groups && !isNan(v)) { atomicMin(&res[gid], f2s(v)); }
}

@compute @workgroup_size(256)
fn max_agg(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= agp.size) { return; }
    let gid = agids[g.x]; let v = vals[g.x];
    if (gid < agp.num_groups && !isNan(v)) { atomicMax(&res[gid], f2s(v)); }
}

@group(0) @binding(0) var<uniform> ip: IP;
@group(0) @binding(1) var<storage, read_write> it: array<u32>;

@compute @workgroup_size(256)
fn init_ht(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= ip.cap * 2u) { return; }
    it[g.x] = select(0u, 0xFFFFFFFFu, g.x % 2u == 0u);
}

@group(0) @binding(0) var<uniform> iap: IAP;
@group(0) @binding(1) var<storage, read_write> iar: array<u32>;

@compute @workgroup_size(256)
fn init_agg(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= iap.num_groups) { return; }
    iar[g.x] = iap.init_value;
}
`;

// Minimum rows to benefit from GPU acceleration
// GPU has overhead - use hash grouping for medium data, GPU for large
const GPU_GROUP_THRESHOLD = 10000;

/**
 * GPU Grouper for SQL GROUP BY operations
 */
export class GPUGrouper {
    constructor() {
        this.device = null;
        this.pipelines = new Map();
        this.available = false;
        this._initPromise = null;
        this.bufferPool = null;
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
            console.log('[GPUGrouper] WebGPU not available');
            return false;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.log('[GPUGrouper] No WebGPU adapter');
                return false;
            }

            this.device = await adapter.requestDevice({
                requiredLimits: {
                    maxStorageBufferBindingSize: 256 * 1024 * 1024,
                    maxBufferSize: 256 * 1024 * 1024,
                },
            });

            await this._compileShaders();
            this.bufferPool = getBufferPool(this.device);
            this.available = true;
            console.log('[GPUGrouper] Initialized');
            return true;
        } catch (e) {
            console.error('[GPUGrouper] Init failed:', e);
            return false;
        }
    }

    async _compileShaders() {
        const module = this.device.createShaderModule({ code: GROUP_BY_SHADER });

        this.pipelines.set('init_ht', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'init_ht' },
        }));

        this.pipelines.set('build', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'build' },
        }));

        this.pipelines.set('assign', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'assign' },
        }));

        this.pipelines.set('init_agg', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'init_agg' },
        }));

        this.pipelines.set('count', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'count_agg' },
        }));

        this.pipelines.set('sum', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'sum_agg' },
        }));

        this.pipelines.set('min', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'min_agg' },
        }));

        this.pipelines.set('max', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'max_agg' },
        }));
    }

    isAvailable() { return this.available; }

    /**
     * Group rows by key and return group assignments.
     * @param {Uint32Array} keys - Hash keys for each row
     * @returns {Promise<{groupIds: Uint32Array, numGroups: number}>}
     */
    async groupBy(keys) {
        const size = keys.length;

        if (!this.available || size < GPU_GROUP_THRESHOLD) {
            return this._cpuGroupBy(keys);
        }

        // Hash table capacity (power of 2, at least 2x estimated groups)
        const estimatedGroups = Math.min(size, 100000);
        const capacity = this._nextPowerOf2(estimatedGroups * 2);

        // Create GPU buffers
        const keysBuf = this._createBuffer(keys, GPUBufferUsage.STORAGE);
        const htBuf = this.device.createBuffer({
            size: capacity * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const gcBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const gidsBuf = this.device.createBuffer({
            size: size * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Initialize hash table
        const initP = this.pipelines.get('init_ht');
        const initParamsBuf = this._createUniformBuffer(new Uint32Array([capacity]));
        const initBG = this.device.createBindGroup({
            layout: initP.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: initParamsBuf } },
                { binding: 1, resource: { buffer: htBuf } },
            ],
        });

        // Build groups
        const buildP = this.pipelines.get('build');
        const buildParamsBuf = this._createUniformBuffer(new Uint32Array([size, capacity]));
        const buildBG = this.device.createBindGroup({
            layout: buildP.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: buildParamsBuf } },
                { binding: 1, resource: { buffer: keysBuf } },
                { binding: 2, resource: { buffer: htBuf } },
                { binding: 3, resource: { buffer: gcBuf } },
            ],
        });

        // Assign groups
        const assignP = this.pipelines.get('assign');
        const assignParamsBuf = this._createUniformBuffer(new Uint32Array([size, capacity]));
        const assignBG = this.device.createBindGroup({
            layout: assignP.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: assignParamsBuf } },
                { binding: 1, resource: { buffer: keysBuf } },
                { binding: 2, resource: { buffer: htBuf } },
                { binding: 3, resource: { buffer: gidsBuf } },
            ],
        });

        // Execute GPU commands
        const encoder = this.device.createCommandEncoder();

        // Init pass
        const initWg = Math.ceil((capacity * 2) / 256);
        const initPass = encoder.beginComputePass();
        initPass.setPipeline(initP);
        initPass.setBindGroup(0, initBG);
        initPass.dispatchWorkgroups(initWg);
        initPass.end();

        // Build pass
        const buildWg = Math.ceil(size / 256);
        const buildPass = encoder.beginComputePass();
        buildPass.setPipeline(buildP);
        buildPass.setBindGroup(0, buildBG);
        buildPass.dispatchWorkgroups(buildWg);
        buildPass.end();

        // Assign pass
        const assignPass = encoder.beginComputePass();
        assignPass.setPipeline(assignP);
        assignPass.setBindGroup(0, assignBG);
        assignPass.dispatchWorkgroups(buildWg);
        assignPass.end();

        // Copy results to staging
        const gcStagingBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        encoder.copyBufferToBuffer(gcBuf, 0, gcStagingBuf, 0, 4);

        this.device.queue.submit([encoder.finish()]);

        // Read group count
        await gcStagingBuf.mapAsync(GPUMapMode.READ);
        const numGroups = new Uint32Array(gcStagingBuf.getMappedRange())[0];
        gcStagingBuf.unmap();

        // Read group IDs
        const gidsStagingBuf = this.device.createBuffer({
            size: size * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(gidsBuf, 0, gidsStagingBuf, 0, size * 4);
        this.device.queue.submit([copyEncoder.finish()]);

        await gidsStagingBuf.mapAsync(GPUMapMode.READ);
        const groupIds = new Uint32Array(gidsStagingBuf.getMappedRange().slice(0));
        gidsStagingBuf.unmap();

        // Cleanup
        keysBuf.destroy();
        htBuf.destroy();
        gcBuf.destroy();
        gidsBuf.destroy();
        gcStagingBuf.destroy();
        gidsStagingBuf.destroy();
        initParamsBuf.destroy();
        buildParamsBuf.destroy();
        assignParamsBuf.destroy();

        return { groupIds, numGroups };
    }

    /**
     * Compute per-group aggregation.
     * @param {Float32Array} values - Values to aggregate
     * @param {Uint32Array} groupIds - Group ID for each row
     * @param {number} numGroups - Total number of groups
     * @param {string} aggType - 'COUNT', 'SUM', 'MIN', 'MAX'
     * @returns {Promise<Float32Array>} Aggregated values per group
     */
    async groupAggregate(values, groupIds, numGroups, aggType) {
        const size = values.length;

        if (!this.available || size < GPU_GROUP_THRESHOLD) {
            return this._cpuGroupAggregate(values, groupIds, numGroups, aggType);
        }

        // Determine init value and pipeline
        let initValue = 0;
        let pipelineName = 'count';
        if (aggType === 'SUM') {
            pipelineName = 'sum';
        } else if (aggType === 'MIN') {
            initValue = 0x7F7FFFFF; // Max positive float as sortable
            pipelineName = 'min';
        } else if (aggType === 'MAX') {
            initValue = 0;
            pipelineName = 'max';
        }

        // Create GPU buffers
        const groupIdsBuf = this._createBuffer(groupIds, GPUBufferUsage.STORAGE);
        const valuesBuf = this._createBuffer(values, GPUBufferUsage.STORAGE);
        const resultsBuf = this.device.createBuffer({
            size: numGroups * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Initialize results
        const initAggP = this.pipelines.get('init_agg');
        const initAggParamsBuf = this._createUniformBuffer(new Uint32Array([numGroups, initValue]));
        const initAggBG = this.device.createBindGroup({
            layout: initAggP.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: initAggParamsBuf } },
                { binding: 1, resource: { buffer: resultsBuf } },
            ],
        });

        // Aggregate
        const aggP = this.pipelines.get(pipelineName);
        const aggParamsBuf = this._createUniformBuffer(new Uint32Array([size, numGroups, 0]));
        const aggBG = this.device.createBindGroup({
            layout: aggP.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: aggParamsBuf } },
                { binding: 1, resource: { buffer: groupIdsBuf } },
                { binding: 2, resource: { buffer: valuesBuf } },
                { binding: 3, resource: { buffer: resultsBuf } },
            ],
        });

        const encoder = this.device.createCommandEncoder();

        // Init pass
        const initWg = Math.ceil(numGroups / 256);
        const initPass = encoder.beginComputePass();
        initPass.setPipeline(initAggP);
        initPass.setBindGroup(0, initAggBG);
        initPass.dispatchWorkgroups(Math.max(1, initWg));
        initPass.end();

        // Aggregate pass
        const aggWg = Math.ceil(size / 256);
        const aggPass = encoder.beginComputePass();
        aggPass.setPipeline(aggP);
        aggPass.setBindGroup(0, aggBG);
        aggPass.dispatchWorkgroups(aggWg);
        aggPass.end();

        // Copy results
        const stagingBuf = this.device.createBuffer({
            size: numGroups * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        encoder.copyBufferToBuffer(resultsBuf, 0, stagingBuf, 0, numGroups * 4);

        this.device.queue.submit([encoder.finish()]);

        await stagingBuf.mapAsync(GPUMapMode.READ);
        const rawResults = new Uint32Array(stagingBuf.getMappedRange().slice(0));
        stagingBuf.unmap();

        // Convert results based on aggregation type
        const results = new Float32Array(numGroups);
        for (let i = 0; i < numGroups; i++) {
            if (aggType === 'COUNT') {
                results[i] = rawResults[i];
            } else if (aggType === 'SUM') {
                // Convert from fixed-point (scaled by 1000)
                results[i] = (rawResults[i] | 0) / 1000;
            } else if (aggType === 'MIN' || aggType === 'MAX') {
                // Convert from sortable representation
                const u = rawResults[i];
                const bits = (u & 0x80000000) ? u ^ 0x80000000 : ~u;
                results[i] = new Float32Array(new Uint32Array([bits]).buffer)[0];
            }
        }

        // Cleanup
        groupIdsBuf.destroy();
        valuesBuf.destroy();
        resultsBuf.destroy();
        stagingBuf.destroy();
        initAggParamsBuf.destroy();
        aggParamsBuf.destroy();

        return results;
    }

    /**
     * CPU fallback for groupBy
     */
    _cpuGroupBy(keys) {
        const groupMap = new Map();
        const groupIds = new Uint32Array(keys.length);
        let nextGroupId = 0;

        for (let i = 0; i < keys.length; i++) {
            const key = keys[i];
            if (!groupMap.has(key)) {
                groupMap.set(key, nextGroupId++);
            }
            groupIds[i] = groupMap.get(key);
        }

        return { groupIds, numGroups: nextGroupId };
    }

    /**
     * CPU fallback for groupAggregate
     */
    _cpuGroupAggregate(values, groupIds, numGroups, aggType) {
        const results = new Float32Array(numGroups);

        if (aggType === 'MIN') {
            results.fill(Infinity);
        } else if (aggType === 'MAX') {
            results.fill(-Infinity);
        }

        for (let i = 0; i < values.length; i++) {
            const gid = groupIds[i];
            const val = values[i];

            if (gid >= numGroups || isNaN(val)) continue;

            if (aggType === 'COUNT') {
                results[gid]++;
            } else if (aggType === 'SUM') {
                results[gid] += val;
            } else if (aggType === 'MIN') {
                results[gid] = Math.min(results[gid], val);
            } else if (aggType === 'MAX') {
                results[gid] = Math.max(results[gid], val);
            }
        }

        return results;
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

    /**
     * Get the buffer pool for external cache management.
     * @returns {GPUBufferPool|null}
     */
    getBufferPool() {
        return this.bufferPool;
    }

    /**
     * Invalidate cached buffers for a table.
     * @param {string} tableId
     */
    invalidateTable(tableId) {
        if (this.bufferPool) {
            this.bufferPool.invalidatePrefix(tableId + ':');
        }
    }

    dispose() {
        if (this.bufferPool) {
            this.bufferPool.clear();
            this.bufferPool = null;
        }
        this.pipelines.clear();
        this.device = null;
        this.available = false;
    }
}

// Singleton instance
let gpuGrouperInstance = null;

/**
 * Get or create the GPU grouper instance.
 * @returns {GPUGrouper}
 */
export function getGPUGrouper() {
    if (!gpuGrouperInstance) {
        gpuGrouperInstance = new GPUGrouper();
    }
    return gpuGrouperInstance;
}

/**
 * Check if GPU GROUP BY is beneficial for given row count.
 * @param {number} rowCount - Number of rows
 * @returns {boolean}
 */
export function shouldUseGPUGroup(rowCount) {
    return rowCount >= GPU_GROUP_THRESHOLD;
}
