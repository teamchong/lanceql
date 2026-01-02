/**
 * GPUJoiner - GPU-accelerated join operations
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

// Lazy singleton - only instantiated when first accessed
let _gpuJoiner = null;
function getGPUJoiner() {
    if (!_gpuJoiner) _gpuJoiner = new GPUJoiner();
    return _gpuJoiner;
}

export { GPUJoiner, getGPUJoiner };
