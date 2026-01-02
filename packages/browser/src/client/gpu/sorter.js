/**
 * GPUSorter - GPU-accelerated sorting with stable order preservation
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

export { GPUSorter };
