// Layer Normalization
// Normalizes input: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Uses parallel reduction for mean/variance computation

struct LayerNormParams {
    batchSize: u32,   // Number of sequences
    seqLen: u32,      // Sequence length
    hiddenSize: u32,  // Hidden dimension to normalize over
    eps: f32,         // Small constant for numerical stability (1e-5)
}

@group(0) @binding(0) var<uniform> params: LayerNormParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;  // Scale
@group(0) @binding(3) var<storage, read> beta: array<f32>;   // Bias
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> sharedSum: array<f32, 256>;
var<workgroup> sharedSumSq: array<f32, 256>;

// Parallel reduction to compute sum
fn workgroupReduceSum(localIdx: u32, value: f32) -> f32 {
    sharedSum[localIdx] = value;
    workgroupBarrier();

    // Tree reduction
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (localIdx < stride) {
            sharedSum[localIdx] = sharedSum[localIdx] + sharedSum[localIdx + stride];
        }
        workgroupBarrier();
    }

    return sharedSum[0];
}

// Compute mean and variance, then normalize
@compute @workgroup_size(256)
fn layernorm(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>,
             @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let batchIdx = workgroup_id.y;
    let seqIdx = workgroup_id.x;
    let localIdx = local_id.x;

    if (batchIdx >= params.batchSize || seqIdx >= params.seqLen) {
        return;
    }

    let offset = (batchIdx * params.seqLen + seqIdx) * params.hiddenSize;

    // Step 1: Compute sum for mean
    var localSum: f32 = 0.0;
    for (var i: u32 = localIdx; i < params.hiddenSize; i = i + WORKGROUP_SIZE) {
        localSum = localSum + input[offset + i];
    }

    let totalSum = workgroupReduceSum(localIdx, localSum);
    let mean = totalSum / f32(params.hiddenSize);

    workgroupBarrier();

    // Step 2: Compute sum of squared differences for variance
    var localSumSq: f32 = 0.0;
    for (var i: u32 = localIdx; i < params.hiddenSize; i = i + WORKGROUP_SIZE) {
        let diff = input[offset + i] - mean;
        localSumSq = localSumSq + diff * diff;
    }

    sharedSumSq[localIdx] = localSumSq;
    workgroupBarrier();

    // Tree reduction for variance
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (localIdx < stride) {
            sharedSumSq[localIdx] = sharedSumSq[localIdx] + sharedSumSq[localIdx + stride];
        }
        workgroupBarrier();
    }

    let variance = sharedSumSq[0] / f32(params.hiddenSize);
    let invStd = 1.0 / sqrt(variance + params.eps);

    workgroupBarrier();

    // Step 3: Normalize and apply scale/bias
    for (var i: u32 = localIdx; i < params.hiddenSize; i = i + WORKGROUP_SIZE) {
        let normalized = (input[offset + i] - mean) * invStd;
        output[offset + i] = normalized * gamma[i] + beta[i];
    }
}

// Fused LayerNorm + Linear: y = LayerNorm(x) @ W + b
// More efficient by avoiding intermediate memory writes
struct FusedParams {
    batchSize: u32,
    seqLen: u32,
    inputSize: u32,   // Input hidden dimension
    outputSize: u32,  // Output dimension after linear
    eps: f32,
}

@group(0) @binding(0) var<uniform> fusedParams: FusedParams;
@group(0) @binding(1) var<storage, read> fusedInput: array<f32>;
@group(0) @binding(2) var<storage, read> lnGamma: array<f32>;
@group(0) @binding(3) var<storage, read> lnBeta: array<f32>;
@group(0) @binding(4) var<storage, read> weight: array<f32>;  // [inputSize × outputSize]
@group(0) @binding(5) var<storage, read> linearBias: array<f32>;
@group(0) @binding(6) var<storage, read_write> fusedOutput: array<f32>;

var<workgroup> normalizedCache: array<f32, 1024>;  // Cache normalized values

@compute @workgroup_size(256)
fn layernorm_linear(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>,
                    @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let batchIdx = workgroup_id.y;
    let seqIdx = workgroup_id.x;
    let localIdx = local_id.x;

    if (batchIdx >= fusedParams.batchSize || seqIdx >= fusedParams.seqLen) {
        return;
    }

    let inputOffset = (batchIdx * fusedParams.seqLen + seqIdx) * fusedParams.inputSize;
    let outputOffset = (batchIdx * fusedParams.seqLen + seqIdx) * fusedParams.outputSize;

    // Compute mean
    var localSum: f32 = 0.0;
    for (var i: u32 = localIdx; i < fusedParams.inputSize; i = i + WORKGROUP_SIZE) {
        localSum = localSum + fusedInput[inputOffset + i];
    }
    let mean = workgroupReduceSum(localIdx, localSum) / f32(fusedParams.inputSize);
    workgroupBarrier();

    // Compute variance
    var localSumSq: f32 = 0.0;
    for (var i: u32 = localIdx; i < fusedParams.inputSize; i = i + WORKGROUP_SIZE) {
        let diff = fusedInput[inputOffset + i] - mean;
        localSumSq = localSumSq + diff * diff;
    }
    sharedSumSq[localIdx] = localSumSq;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (localIdx < stride) {
            sharedSumSq[localIdx] = sharedSumSq[localIdx] + sharedSumSq[localIdx + stride];
        }
        workgroupBarrier();
    }

    let invStd = 1.0 / sqrt(sharedSumSq[0] / f32(fusedParams.inputSize) + fusedParams.eps);
    workgroupBarrier();

    // Normalize and cache (for hidden sizes up to 1024)
    for (var i: u32 = localIdx; i < min(fusedParams.inputSize, 1024u); i = i + WORKGROUP_SIZE) {
        let normalized = (fusedInput[inputOffset + i] - mean) * invStd;
        normalizedCache[i] = normalized * lnGamma[i] + lnBeta[i];
    }
    workgroupBarrier();

    // Linear projection: each thread computes one output element
    for (var outIdx: u32 = localIdx; outIdx < fusedParams.outputSize; outIdx = outIdx + WORKGROUP_SIZE) {
        var acc: f32 = linearBias[outIdx];
        for (var i: u32 = 0u; i < fusedParams.inputSize; i = i + 1u) {
            let normalized = (fusedInput[inputOffset + i] - mean) * invStd * lnGamma[i] + lnBeta[i];
            acc = acc + normalized * weight[i * fusedParams.outputSize + outIdx];
        }
        fusedOutput[outputOffset + outIdx] = acc;
    }
}
