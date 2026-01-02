/**
 * GPUGrouper - GPU GROUP BY with multi-level partitioning
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


export { GPUGrouper };
