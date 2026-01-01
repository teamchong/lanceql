/**
 * GPU-accelerated SQL JOINs
 *
 * Provides GPU-based hash join for large table operations.
 * Falls back to CPU for small datasets where GPU overhead exceeds benefit.
 */

// Hash join shaders - embedded for bundler compatibility
const JOIN_SHADER = `
struct BuildParams { size: u32, capacity: u32 }
struct ProbeParams { left_size: u32, capacity: u32, max_matches: u32 }
struct InitParams { capacity: u32 }

@group(0) @binding(0) var<uniform> build_params: BuildParams;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hash_table: array<atomic<u32>>;

fn fnv_hash(key: u32) -> u32 {
    var h = 2166136261u;
    h ^= (key & 0xFFu); h *= 16777619u;
    h ^= ((key >> 8u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 16u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 24u) & 0xFFu); h *= 16777619u;
    return h;
}

@compute @workgroup_size(256)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= build_params.size) { return; }
    let key = keys[tid];
    var slot = fnv_hash(key) % build_params.capacity;
    for (var p = 0u; p < build_params.capacity; p++) {
        let idx = slot * 2u;
        let old = atomicCompareExchangeWeak(&hash_table[idx], 0xFFFFFFFFu, key);
        if (old.exchanged) { atomicStore(&hash_table[idx + 1u], tid); return; }
        slot = (slot + 1u) % build_params.capacity;
    }
}

@group(0) @binding(0) var<uniform> probe_params: ProbeParams;
@group(0) @binding(1) var<storage, read> left_keys: array<u32>;
@group(0) @binding(2) var<storage, read> probe_table: array<u32>;
@group(0) @binding(3) var<storage, read_write> matches: array<u32>;
@group(0) @binding(4) var<storage, read_write> match_count: atomic<u32>;

@compute @workgroup_size(256)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let left_idx = gid.x;
    if (left_idx >= probe_params.left_size) { return; }
    let key = left_keys[left_idx];
    var slot = fnv_hash(key) % probe_params.capacity;
    for (var p = 0u; p < probe_params.capacity; p++) {
        let idx = slot * 2u;
        let stored = probe_table[idx];
        if (stored == 0xFFFFFFFFu) { return; }
        if (stored == key) {
            let right_idx = probe_table[idx + 1u];
            let out = atomicAdd(&match_count, 1u);
            if (out * 2u + 1u < probe_params.max_matches * 2u) {
                matches[out * 2u] = left_idx;
                matches[out * 2u + 1u] = right_idx;
            }
        }
        slot = (slot + 1u) % probe_params.capacity;
    }
}

@group(0) @binding(0) var<uniform> init_params: InitParams;
@group(0) @binding(1) var<storage, read_write> init_table: array<u32>;

@compute @workgroup_size(256)
fn init_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= init_params.capacity * 2u) { return; }
    init_table[idx] = select(0u, 0xFFFFFFFFu, idx % 2u == 0u);
}
`;

// Minimum rows to benefit from GPU acceleration
const GPU_JOIN_THRESHOLD = 10000;

/**
 * GPU Joiner for SQL hash join operations
 */
export class GPUJoiner {
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
            console.log('[GPUJoiner] WebGPU not available');
            return false;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.log('[GPUJoiner] No WebGPU adapter');
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
            console.log('[GPUJoiner] Initialized');
            return true;
        } catch (e) {
            console.error('[GPUJoiner] Init failed:', e);
            return false;
        }
    }

    async _compileShaders() {
        const module = this.device.createShaderModule({ code: JOIN_SHADER });

        this.pipelines.set('init', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'init_table' },
        }));

        this.pipelines.set('build', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'build' },
        }));

        this.pipelines.set('probe', this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'probe' },
        }));
    }

    isAvailable() { return this.available; }

    /**
     * Perform hash join between two tables.
     * @param {Object[]} leftRows - Left table rows
     * @param {Object[]} rightRows - Right table rows (build side)
     * @param {string} leftKey - Left join column name
     * @param {string} rightKey - Right join column name
     * @param {string} joinType - 'INNER', 'LEFT', 'RIGHT'
     * @returns {Promise<{leftIndices: Uint32Array, rightIndices: Uint32Array}>}
     */
    async hashJoin(leftRows, rightRows, leftKey, rightKey, joinType = 'INNER') {
        const leftSize = leftRows.length;
        const rightSize = rightRows.length;

        // Use CPU for small tables
        if (!this.available || leftSize * rightSize < GPU_JOIN_THRESHOLD * GPU_JOIN_THRESHOLD) {
            return this._cpuHashJoin(leftRows, rightRows, leftKey, rightKey, joinType);
        }

        // Extract keys as numeric values
        const leftKeys = this._extractKeys(leftRows, leftKey);
        const rightKeys = this._extractKeys(rightRows, rightKey);

        // Hash table capacity (power of 2, at least 2x size)
        const capacity = this._nextPowerOf2(rightSize * 2);
        const maxMatches = Math.max(leftSize * 10, 100000); // Allow many-to-many

        // Create GPU buffers
        const rightKeysBuf = this._createBuffer(rightKeys, GPUBufferUsage.STORAGE);
        const leftKeysBuf = this._createBuffer(leftKeys, GPUBufferUsage.STORAGE);
        const hashTableBuf = this.device.createBuffer({
            size: capacity * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const matchesBuf = this.device.createBuffer({
            size: maxMatches * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const matchCountBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const stagingBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // Initialize hash table
        const initParamsBuf = this._createUniformBuffer(new Uint32Array([capacity]));
        const initPipeline = this.pipelines.get('init');
        const initBindGroup = this.device.createBindGroup({
            layout: initPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: initParamsBuf } },
                { binding: 1, resource: { buffer: hashTableBuf } },
            ],
        });

        // Build hash table
        const buildParamsBuf = this._createUniformBuffer(new Uint32Array([rightSize, capacity]));
        const buildPipeline = this.pipelines.get('build');
        const buildBindGroup = this.device.createBindGroup({
            layout: buildPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: buildParamsBuf } },
                { binding: 1, resource: { buffer: rightKeysBuf } },
                { binding: 2, resource: { buffer: hashTableBuf } },
            ],
        });

        // Probe hash table
        const probeParamsBuf = this._createUniformBuffer(new Uint32Array([leftSize, capacity, maxMatches]));
        const probePipeline = this.pipelines.get('probe');
        const probeBindGroup = this.device.createBindGroup({
            layout: probePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: probeParamsBuf } },
                { binding: 1, resource: { buffer: leftKeysBuf } },
                { binding: 2, resource: { buffer: hashTableBuf } },
                { binding: 3, resource: { buffer: matchesBuf } },
                { binding: 4, resource: { buffer: matchCountBuf } },
            ],
        });

        // Execute GPU commands
        const encoder = this.device.createCommandEncoder();

        // Init pass
        const initWg = Math.ceil((capacity * 2) / 256);
        const initPass = encoder.beginComputePass();
        initPass.setPipeline(initPipeline);
        initPass.setBindGroup(0, initBindGroup);
        initPass.dispatchWorkgroups(initWg);
        initPass.end();

        // Build pass
        const buildWg = Math.ceil(rightSize / 256);
        const buildPass = encoder.beginComputePass();
        buildPass.setPipeline(buildPipeline);
        buildPass.setBindGroup(0, buildBindGroup);
        buildPass.dispatchWorkgroups(buildWg);
        buildPass.end();

        // Probe pass
        const probeWg = Math.ceil(leftSize / 256);
        const probePass = encoder.beginComputePass();
        probePass.setPipeline(probePipeline);
        probePass.setBindGroup(0, probeBindGroup);
        probePass.dispatchWorkgroups(probeWg);
        probePass.end();

        // Copy match count to staging
        encoder.copyBufferToBuffer(matchCountBuf, 0, stagingBuf, 0, 4);
        this.device.queue.submit([encoder.finish()]);

        // Read match count
        await stagingBuf.mapAsync(GPUMapMode.READ);
        const matchCount = new Uint32Array(stagingBuf.getMappedRange())[0];
        stagingBuf.unmap();

        // Read matched pairs
        const actualMatches = Math.min(matchCount, maxMatches);
        const matchStagingBuf = this.device.createBuffer({
            size: actualMatches * 2 * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(matchesBuf, 0, matchStagingBuf, 0, actualMatches * 2 * 4);
        this.device.queue.submit([copyEncoder.finish()]);

        await matchStagingBuf.mapAsync(GPUMapMode.READ);
        const matchData = new Uint32Array(matchStagingBuf.getMappedRange().slice(0));
        matchStagingBuf.unmap();

        // Extract left and right indices
        const leftIndices = new Uint32Array(actualMatches);
        const rightIndices = new Uint32Array(actualMatches);
        for (let i = 0; i < actualMatches; i++) {
            leftIndices[i] = matchData[i * 2];
            rightIndices[i] = matchData[i * 2 + 1];
        }

        // Cleanup
        rightKeysBuf.destroy();
        leftKeysBuf.destroy();
        hashTableBuf.destroy();
        matchesBuf.destroy();
        matchCountBuf.destroy();
        stagingBuf.destroy();
        matchStagingBuf.destroy();
        initParamsBuf.destroy();
        buildParamsBuf.destroy();
        probeParamsBuf.destroy();

        return { leftIndices, rightIndices, matchCount: actualMatches };
    }

    /**
     * CPU fallback hash join
     */
    _cpuHashJoin(leftRows, rightRows, leftKey, rightKey, joinType) {
        // Build hash table from right rows
        const rightMap = new Map();
        for (let i = 0; i < rightRows.length; i++) {
            const key = this._hashKey(rightRows[i][rightKey]);
            if (!rightMap.has(key)) {
                rightMap.set(key, []);
            }
            rightMap.get(key).push(i);
        }

        // Probe with left rows
        const leftIndices = [];
        const rightIndices = [];

        for (let i = 0; i < leftRows.length; i++) {
            const key = this._hashKey(leftRows[i][leftKey]);
            const matches = rightMap.get(key) || [];
            for (const rightIdx of matches) {
                leftIndices.push(i);
                rightIndices.push(rightIdx);
            }
        }

        return {
            leftIndices: new Uint32Array(leftIndices),
            rightIndices: new Uint32Array(rightIndices),
            matchCount: leftIndices.length,
        };
    }

    /**
     * Extract keys from rows as Uint32Array
     */
    _extractKeys(rows, keyColumn) {
        const keys = new Uint32Array(rows.length);
        for (let i = 0; i < rows.length; i++) {
            keys[i] = this._hashKey(rows[i][keyColumn]);
        }
        return keys;
    }

    /**
     * Convert any value to a u32 hash
     */
    _hashKey(value) {
        if (value === null || value === undefined) {
            return 0xFFFFFFFE; // Special null marker
        }
        if (typeof value === 'number') {
            // For integers, use directly; for floats, use bit representation
            if (Number.isInteger(value) && value >= 0 && value < 0xFFFFFFFF) {
                return value >>> 0;
            }
            // Hash float as bytes
            const buf = new ArrayBuffer(4);
            new Float32Array(buf)[0] = value;
            return new Uint32Array(buf)[0];
        }
        if (typeof value === 'string') {
            // FNV-1a string hash
            let hash = 2166136261;
            for (let i = 0; i < value.length; i++) {
                hash ^= value.charCodeAt(i);
                hash = Math.imul(hash, 16777619);
            }
            return hash >>> 0;
        }
        // Fallback: stringify and hash
        return this._hashKey(String(value));
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
            size: Math.max(data.byteLength, 16), // Min 16 bytes for uniform
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
let gpuJoinerInstance = null;

/**
 * Get or create the GPU joiner instance.
 * @returns {GPUJoiner}
 */
export function getGPUJoiner() {
    if (!gpuJoinerInstance) {
        gpuJoinerInstance = new GPUJoiner();
    }
    return gpuJoinerInstance;
}

/**
 * Check if GPU join is beneficial for given table sizes.
 * @param {number} leftSize - Left table row count
 * @param {number} rightSize - Right table row count
 * @returns {boolean}
 */
export function shouldUseGPUJoin(leftSize, rightSize) {
    return leftSize * rightSize >= GPU_JOIN_THRESHOLD * GPU_JOIN_THRESHOLD;
}
