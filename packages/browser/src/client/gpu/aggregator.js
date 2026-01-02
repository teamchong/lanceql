/**
 * GPUAggregator - GPU-accelerated aggregation (SUM, COUNT, AVG, MIN, MAX)
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

export { GPUAggregator };
