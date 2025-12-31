/**
 * WebGPU Shader Unit Tests
 *
 * Run in browser: open test/webgpu-test.html
 * Run in Node: node --experimental-webgpu test/webgpu.test.js (requires Dawn)
 */

import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';

// Shader sources (inline for testing)
const GEMM_SHADER = `
struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(16, 16)
fn gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
        acc = acc + A[row * dims.K + k] * B[k * dims.N + col];
    }

    C[row * dims.N + col] = dims.alpha * acc;
}
`;

const GELU_SHADER = `
struct Params {
    size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn gelu_fast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    let x = input[idx];
    let sigmoid = 1.0 / (1.0 + exp(-1.702 * x));
    output[idx] = x * sigmoid;
}
`;

const LAYERNORM_SHADER = `
struct Params {
    size: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> sharedSum: array<f32, 256>;

@compute @workgroup_size(256)
fn layernorm(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>) {
    let localIdx = local_id.x;

    // Compute mean
    var localSum: f32 = 0.0;
    for (var i: u32 = localIdx; i < params.size; i = i + 256u) {
        localSum = localSum + input[i];
    }
    sharedSum[localIdx] = localSum;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (localIdx < stride) {
            sharedSum[localIdx] = sharedSum[localIdx] + sharedSum[localIdx + stride];
        }
        workgroupBarrier();
    }
    let mean = sharedSum[0] / f32(params.size);
    workgroupBarrier();

    // Compute variance
    var localSumSq: f32 = 0.0;
    for (var i: u32 = localIdx; i < params.size; i = i + 256u) {
        let diff = input[i] - mean;
        localSumSq = localSumSq + diff * diff;
    }
    sharedSum[localIdx] = localSumSq;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (localIdx < stride) {
            sharedSum[localIdx] = sharedSum[localIdx] + sharedSum[localIdx + stride];
        }
        workgroupBarrier();
    }
    let variance = sharedSum[0] / f32(params.size);
    let invStd = 1.0 / sqrt(variance + params.eps);
    workgroupBarrier();

    // Normalize
    for (var i: u32 = localIdx; i < params.size; i = i + 256u) {
        let normalized = (input[i] - mean) * invStd;
        output[i] = normalized * gamma[i] + beta[i];
    }
}
`;

// Test utilities
class WebGPUTestContext {
    constructor() {
        this.device = null;
        this.available = false;
    }

    async init() {
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            console.log('WebGPU not available - skipping GPU tests');
            return false;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.log('No WebGPU adapter found');
                return false;
            }

            this.device = await adapter.requestDevice();
            this.available = true;
            return true;
        } catch (e) {
            console.log('WebGPU init failed:', e);
            return false;
        }
    }

    createBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage,
            mappedAtCreation: true,
        });
        new Float32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
        return buffer;
    }

    createUniformBuffer(data) {
        const alignedSize = Math.ceil(data.byteLength / 16) * 16;
        const buffer = this.device.createBuffer({
            size: alignedSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }

    async readBuffer(buffer, size) {
        const readBuffer = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
        this.device.queue.submit([encoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();
        readBuffer.destroy();

        return result;
    }

    createPipeline(shaderCode, entryPoint = 'main') {
        const module = this.device.createShaderModule({ code: shaderCode });
        return this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint },
        });
    }

    destroy() {
        if (this.device) {
            this.device.destroy();
        }
    }
}

// Helper to check if arrays are approximately equal
function assertArrayClose(actual, expected, tolerance = 1e-5) {
    assert.strictEqual(actual.length, expected.length, 'Array lengths differ');
    for (let i = 0; i < actual.length; i++) {
        const diff = Math.abs(actual[i] - expected[i]);
        assert.ok(diff < tolerance, `Element ${i}: ${actual[i]} != ${expected[i]} (diff: ${diff})`);
    }
}

// CPU reference implementations
function cpuGemm(A, B, M, N, K, alpha = 1.0) {
    const C = new Float32Array(M * N);
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum;
        }
    }
    return C;
}

function cpuGelu(input) {
    const output = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
        const x = input[i];
        const sigmoid = 1 / (1 + Math.exp(-1.702 * x));
        output[i] = x * sigmoid;
    }
    return output;
}

function cpuLayerNorm(input, gamma, beta, eps = 1e-5) {
    const mean = input.reduce((a, b) => a + b, 0) / input.length;
    const variance = input.reduce((a, b) => a + (b - mean) ** 2, 0) / input.length;
    const invStd = 1 / Math.sqrt(variance + eps);

    const output = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
        output[i] = ((input[i] - mean) * invStd) * gamma[i] + beta[i];
    }
    return output;
}

// Tests
describe('WebGPU Shader Tests', () => {
    let ctx;

    before(async () => {
        ctx = new WebGPUTestContext();
        const available = await ctx.init();
        if (!available) {
            console.log('Skipping WebGPU tests - not available');
        }
    });

    after(() => {
        if (ctx) ctx.destroy();
    });

    describe('GEMM Shader', () => {
        it('should compute 2x2 matrix multiplication', async function() {
            if (!ctx.available) return this.skip();

            const M = 2, N = 2, K = 2;
            const A = new Float32Array([1, 2, 3, 4]);
            const B = new Float32Array([5, 6, 7, 8]);
            const expected = cpuGemm(A, B, M, N, K);

            // Create buffers
            const uniformData = new ArrayBuffer(16);
            const view = new DataView(uniformData);
            view.setUint32(0, M, true);
            view.setUint32(4, N, true);
            view.setUint32(8, K, true);
            view.setFloat32(12, 1.0, true);

            const uniformBuffer = ctx.createUniformBuffer(uniformData);
            const bufferA = ctx.createBuffer(A);
            const bufferB = ctx.createBuffer(B);
            const bufferC = ctx.createBuffer(new Float32Array(M * N));

            // Create pipeline
            const pipeline = ctx.createPipeline(GEMM_SHADER, 'gemm');

            // Create bind group
            const bindGroup = ctx.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer } },
                    { binding: 1, resource: { buffer: bufferA } },
                    { binding: 2, resource: { buffer: bufferB } },
                    { binding: 3, resource: { buffer: bufferC } },
                ],
            });

            // Run shader
            const encoder = ctx.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(1, 1);
            pass.end();
            ctx.device.queue.submit([encoder.finish()]);

            // Read result
            const result = await ctx.readBuffer(bufferC, M * N * 4);
            assertArrayClose(result, expected);
        });

        it('should compute 4x4 matrix multiplication', async function() {
            if (!ctx.available) return this.skip();

            const M = 4, N = 4, K = 4;
            const A = new Float32Array(M * K).map(() => Math.random());
            const B = new Float32Array(K * N).map(() => Math.random());
            const expected = cpuGemm(A, B, M, N, K);

            const uniformData = new ArrayBuffer(16);
            const view = new DataView(uniformData);
            view.setUint32(0, M, true);
            view.setUint32(4, N, true);
            view.setUint32(8, K, true);
            view.setFloat32(12, 1.0, true);

            const uniformBuffer = ctx.createUniformBuffer(uniformData);
            const bufferA = ctx.createBuffer(A);
            const bufferB = ctx.createBuffer(B);
            const bufferC = ctx.createBuffer(new Float32Array(M * N));

            const pipeline = ctx.createPipeline(GEMM_SHADER, 'gemm');
            const bindGroup = ctx.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer } },
                    { binding: 1, resource: { buffer: bufferA } },
                    { binding: 2, resource: { buffer: bufferB } },
                    { binding: 3, resource: { buffer: bufferC } },
                ],
            });

            const encoder = ctx.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(1, 1);
            pass.end();
            ctx.device.queue.submit([encoder.finish()]);

            const result = await ctx.readBuffer(bufferC, M * N * 4);
            assertArrayClose(result, expected, 1e-4);
        });
    });

    describe('GELU Shader', () => {
        it('should compute GELU activation', async function() {
            if (!ctx.available) return this.skip();

            const input = new Float32Array([-2, -1, 0, 1, 2]);
            const expected = cpuGelu(input);

            const uniformData = new ArrayBuffer(16);
            new DataView(uniformData).setUint32(0, input.length, true);

            const uniformBuffer = ctx.createUniformBuffer(uniformData);
            const inputBuffer = ctx.createBuffer(input);
            const outputBuffer = ctx.createBuffer(new Float32Array(input.length));

            const pipeline = ctx.createPipeline(GELU_SHADER, 'gelu_fast');
            const bindGroup = ctx.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer } },
                    { binding: 1, resource: { buffer: inputBuffer } },
                    { binding: 2, resource: { buffer: outputBuffer } },
                ],
            });

            const encoder = ctx.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(1);
            pass.end();
            ctx.device.queue.submit([encoder.finish()]);

            const result = await ctx.readBuffer(outputBuffer, input.length * 4);
            assertArrayClose(result, expected, 1e-5);
        });

        it('should handle large arrays', async function() {
            if (!ctx.available) return this.skip();

            const size = 1024;
            const input = new Float32Array(size).map(() => (Math.random() - 0.5) * 4);
            const expected = cpuGelu(input);

            const uniformData = new ArrayBuffer(16);
            new DataView(uniformData).setUint32(0, size, true);

            const uniformBuffer = ctx.createUniformBuffer(uniformData);
            const inputBuffer = ctx.createBuffer(input);
            const outputBuffer = ctx.createBuffer(new Float32Array(size));

            const pipeline = ctx.createPipeline(GELU_SHADER, 'gelu_fast');
            const bindGroup = ctx.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer } },
                    { binding: 1, resource: { buffer: inputBuffer } },
                    { binding: 2, resource: { buffer: outputBuffer } },
                ],
            });

            const encoder = ctx.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(size / 256));
            pass.end();
            ctx.device.queue.submit([encoder.finish()]);

            const result = await ctx.readBuffer(outputBuffer, size * 4);
            assertArrayClose(result, expected, 1e-5);
        });
    });

    describe('LayerNorm Shader', () => {
        it('should normalize input', async function() {
            if (!ctx.available) return this.skip();

            const size = 256;
            const input = new Float32Array(size).map(() => Math.random() * 10);
            const gamma = new Float32Array(size).fill(1.0);
            const beta = new Float32Array(size).fill(0.0);
            const expected = cpuLayerNorm(input, gamma, beta);

            const uniformData = new ArrayBuffer(16);
            const view = new DataView(uniformData);
            view.setUint32(0, size, true);
            view.setFloat32(4, 1e-5, true);

            const uniformBuffer = ctx.createUniformBuffer(uniformData);
            const inputBuffer = ctx.createBuffer(input);
            const gammaBuffer = ctx.createBuffer(gamma);
            const betaBuffer = ctx.createBuffer(beta);
            const outputBuffer = ctx.createBuffer(new Float32Array(size));

            const pipeline = ctx.createPipeline(LAYERNORM_SHADER, 'layernorm');
            const bindGroup = ctx.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer } },
                    { binding: 1, resource: { buffer: inputBuffer } },
                    { binding: 2, resource: { buffer: gammaBuffer } },
                    { binding: 3, resource: { buffer: betaBuffer } },
                    { binding: 4, resource: { buffer: outputBuffer } },
                ],
            });

            const encoder = ctx.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(1);
            pass.end();
            ctx.device.queue.submit([encoder.finish()]);

            const result = await ctx.readBuffer(outputBuffer, size * 4);
            assertArrayClose(result, expected, 1e-4);
        });

        it('should apply scale and bias', async function() {
            if (!ctx.available) return this.skip();

            const size = 256;
            const input = new Float32Array(size).map(() => Math.random() * 10);
            const gamma = new Float32Array(size).map(() => Math.random() * 2);
            const beta = new Float32Array(size).map(() => Math.random() - 0.5);
            const expected = cpuLayerNorm(input, gamma, beta);

            const uniformData = new ArrayBuffer(16);
            const view = new DataView(uniformData);
            view.setUint32(0, size, true);
            view.setFloat32(4, 1e-5, true);

            const uniformBuffer = ctx.createUniformBuffer(uniformData);
            const inputBuffer = ctx.createBuffer(input);
            const gammaBuffer = ctx.createBuffer(gamma);
            const betaBuffer = ctx.createBuffer(beta);
            const outputBuffer = ctx.createBuffer(new Float32Array(size));

            const pipeline = ctx.createPipeline(LAYERNORM_SHADER, 'layernorm');
            const bindGroup = ctx.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer } },
                    { binding: 1, resource: { buffer: inputBuffer } },
                    { binding: 2, resource: { buffer: gammaBuffer } },
                    { binding: 3, resource: { buffer: betaBuffer } },
                    { binding: 4, resource: { buffer: outputBuffer } },
                ],
            });

            const encoder = ctx.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(1);
            pass.end();
            ctx.device.queue.submit([encoder.finish()]);

            const result = await ctx.readBuffer(outputBuffer, size * 4);
            assertArrayClose(result, expected, 1e-4);
        });
    });
});

// CPU-only tests (always run)
describe('CPU Reference Tests', () => {
    it('cpuGemm produces correct results', () => {
        const A = new Float32Array([1, 2, 3, 4]);
        const B = new Float32Array([5, 6, 7, 8]);
        const result = cpuGemm(A, B, 2, 2, 2);
        assertArrayClose(result, new Float32Array([19, 22, 43, 50]));
    });

    it('cpuGelu handles negative values', () => {
        const input = new Float32Array([-2, -1, 0, 1, 2]);
        const result = cpuGelu(input);

        // GELU(-2) ≈ -0.0454, GELU(-1) ≈ -0.159, GELU(0) = 0
        assert.ok(result[0] < 0, 'GELU(-2) should be negative');
        assert.ok(result[1] < 0, 'GELU(-1) should be negative');
        assert.ok(Math.abs(result[2]) < 1e-6, 'GELU(0) should be ~0');
        assert.ok(result[3] > 0.5, 'GELU(1) should be > 0.5');
        assert.ok(result[4] > 1.5, 'GELU(2) should be > 1.5');
    });

    it('cpuLayerNorm normalizes to zero mean unit variance', () => {
        const input = new Float32Array([1, 2, 3, 4, 5]);
        const gamma = new Float32Array(5).fill(1);
        const beta = new Float32Array(5).fill(0);
        const result = cpuLayerNorm(input, gamma, beta);

        // Mean should be ~0
        const mean = result.reduce((a, b) => a + b, 0) / result.length;
        assert.ok(Math.abs(mean) < 1e-5, `Mean should be ~0, got ${mean}`);

        // Variance should be ~1
        const variance = result.reduce((a, b) => a + b * b, 0) / result.length;
        assert.ok(Math.abs(variance - 1) < 1e-5, `Variance should be ~1, got ${variance}`);
    });
});

console.log('Run with: node --test test/webgpu.test.js');
