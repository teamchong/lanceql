// Fused Multi-Head Self-Attention
// Computes: Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
// Fused kernel reduces memory bandwidth by computing in-place

struct AttentionParams {
    batchSize: u32,
    seqLen: u32,
    numHeads: u32,
    headDim: u32,      // d_k = hiddenSize / numHeads
    hiddenSize: u32,   // Total hidden dimension
}

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> query: array<f32>;   // [batch × seq × hidden]
@group(0) @binding(2) var<storage, read> key: array<f32>;     // [batch × seq × hidden]
@group(0) @binding(3) var<storage, read> value: array<f32>;   // [batch × seq × hidden]
@group(0) @binding(4) var<storage, read_write> output: array<f32>;  // [batch × seq × hidden]

const TILE_SIZE: u32 = 16u;
const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> sharedQ: array<f32, 256>;  // Query tile
var<workgroup> sharedK: array<f32, 256>;  // Key tile
var<workgroup> sharedScores: array<f32, 256>;  // Attention scores
var<workgroup> sharedMax: array<f32, 16>;  // For stable softmax

// Compute attention for one head, one query position
// This is a simplified version - full optimization would use flash attention
@compute @workgroup_size(16, 16)
fn attention(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>,
             @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let batchIdx = workgroup_id.z / params.numHeads;
    let headIdx = workgroup_id.z % params.numHeads;
    let queryPos = global_id.y;
    let dimIdx = global_id.x;

    if (batchIdx >= params.batchSize || queryPos >= params.seqLen || dimIdx >= params.headDim) {
        return;
    }

    let scale = 1.0 / sqrt(f32(params.headDim));

    // Compute Q[queryPos] × K^T to get attention scores
    var scores: array<f32, 512>;  // Max sequence length 512
    var maxScore: f32 = -1e10;

    for (var keyPos: u32 = 0u; keyPos < params.seqLen; keyPos = keyPos + 1u) {
        var dotProduct: f32 = 0.0;

        // Compute dot product Q[queryPos] · K[keyPos]
        for (var d: u32 = 0u; d < params.headDim; d = d + 1u) {
            let qIdx = batchIdx * params.seqLen * params.hiddenSize +
                       queryPos * params.hiddenSize +
                       headIdx * params.headDim + d;
            let kIdx = batchIdx * params.seqLen * params.hiddenSize +
                       keyPos * params.hiddenSize +
                       headIdx * params.headDim + d;

            dotProduct = dotProduct + query[qIdx] * key[kIdx];
        }

        let score = dotProduct * scale;
        scores[keyPos] = score;
        maxScore = max(maxScore, score);
    }

    // Softmax: exp(score - max) / sum(exp(scores - max))
    var sumExp: f32 = 0.0;
    for (var keyPos: u32 = 0u; keyPos < params.seqLen; keyPos = keyPos + 1u) {
        scores[keyPos] = exp(scores[keyPos] - maxScore);
        sumExp = sumExp + scores[keyPos];
    }

    for (var keyPos: u32 = 0u; keyPos < params.seqLen; keyPos = keyPos + 1u) {
        scores[keyPos] = scores[keyPos] / sumExp;
    }

    // Compute weighted sum of values
    var attnOutput: f32 = 0.0;
    for (var keyPos: u32 = 0u; keyPos < params.seqLen; keyPos = keyPos + 1u) {
        let vIdx = batchIdx * params.seqLen * params.hiddenSize +
                   keyPos * params.hiddenSize +
                   headIdx * params.headDim + dimIdx;
        attnOutput = attnOutput + scores[keyPos] * value[vIdx];
    }

    // Write output
    let outIdx = batchIdx * params.seqLen * params.hiddenSize +
                 queryPos * params.hiddenSize +
                 headIdx * params.headDim + dimIdx;
    output[outIdx] = attnOutput;
}

// Softmax for attention scores (standalone)
// Used when attention is computed in separate passes
struct SoftmaxParams {
    batchSize: u32,
    seqLen: u32,
    numHeads: u32,
}

@group(0) @binding(0) var<uniform> softmaxParams: SoftmaxParams;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;  // [batch × heads × seq × seq]

var<workgroup> wgMax: array<f32, 256>;
var<workgroup> wgSum: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) global_id: vec3<u32>,
           @builtin(local_invocation_id) local_id: vec3<u32>,
           @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let batchHeadIdx = workgroup_id.y;
    let queryPos = workgroup_id.x;
    let localIdx = local_id.x;

    let batchIdx = batchHeadIdx / softmaxParams.numHeads;
    let headIdx = batchHeadIdx % softmaxParams.numHeads;

    if (batchIdx >= softmaxParams.batchSize || queryPos >= softmaxParams.seqLen) {
        return;
    }

    let rowOffset = (batchHeadIdx * softmaxParams.seqLen + queryPos) * softmaxParams.seqLen;

    // Find max for numerical stability
    var localMax: f32 = -1e10;
    for (var i: u32 = localIdx; i < softmaxParams.seqLen; i = i + WORKGROUP_SIZE) {
        localMax = max(localMax, scores[rowOffset + i]);
    }
    wgMax[localIdx] = localMax;
    workgroupBarrier();

    // Reduce to find global max
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (localIdx < stride) {
            wgMax[localIdx] = max(wgMax[localIdx], wgMax[localIdx + stride]);
        }
        workgroupBarrier();
    }
    let globalMax = wgMax[0];
    workgroupBarrier();

    // Compute exp(x - max) and sum
    var localSum: f32 = 0.0;
    for (var i: u32 = localIdx; i < softmaxParams.seqLen; i = i + WORKGROUP_SIZE) {
        let expVal = exp(scores[rowOffset + i] - globalMax);
        scores[rowOffset + i] = expVal;
        localSum = localSum + expVal;
    }
    wgSum[localIdx] = localSum;
    workgroupBarrier();

    // Reduce to find global sum
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (localIdx < stride) {
            wgSum[localIdx] = wgSum[localIdx] + wgSum[localIdx + stride];
        }
        workgroupBarrier();
    }
    let globalSum = wgSum[0];
    workgroupBarrier();

    // Normalize
    let invSum = 1.0 / globalSum;
    for (var i: u32 = localIdx; i < softmaxParams.seqLen; i = i + WORKGROUP_SIZE) {
        scores[rowOffset + i] = scores[rowOffset + i] * invSum;
    }
}

// QKV projection: compute Q, K, V from input in one pass
// More efficient than three separate GEMM calls
struct QKVParams {
    batchSize: u32,
    seqLen: u32,
    hiddenSize: u32,
}

@group(0) @binding(0) var<uniform> qkvParams: QKVParams;
@group(0) @binding(1) var<storage, read> qkvInput: array<f32>;     // [batch × seq × hidden]
@group(0) @binding(2) var<storage, read> qWeight: array<f32>;      // [hidden × hidden]
@group(0) @binding(3) var<storage, read> kWeight: array<f32>;      // [hidden × hidden]
@group(0) @binding(4) var<storage, read> vWeight: array<f32>;      // [hidden × hidden]
@group(0) @binding(5) var<storage, read> qBias: array<f32>;        // [hidden]
@group(0) @binding(6) var<storage, read> kBias: array<f32>;        // [hidden]
@group(0) @binding(7) var<storage, read> vBias: array<f32>;        // [hidden]
@group(0) @binding(8) var<storage, read_write> qOut: array<f32>;   // [batch × seq × hidden]
@group(0) @binding(9) var<storage, read_write> kOut: array<f32>;   // [batch × seq × hidden]
@group(0) @binding(10) var<storage, read_write> vOut: array<f32>;  // [batch × seq × hidden]

@compute @workgroup_size(256)
fn qkv_projection(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let totalElements = qkvParams.batchSize * qkvParams.seqLen * qkvParams.hiddenSize;

    if (idx >= totalElements) {
        return;
    }

    let dimIdx = idx % qkvParams.hiddenSize;
    let seqIdx = (idx / qkvParams.hiddenSize) % qkvParams.seqLen;
    let batchIdx = idx / (qkvParams.seqLen * qkvParams.hiddenSize);

    let inputOffset = (batchIdx * qkvParams.seqLen + seqIdx) * qkvParams.hiddenSize;

    // Compute Q, K, V projections
    var qVal: f32 = qBias[dimIdx];
    var kVal: f32 = kBias[dimIdx];
    var vVal: f32 = vBias[dimIdx];

    for (var i: u32 = 0u; i < qkvParams.hiddenSize; i = i + 1u) {
        let inputVal = qkvInput[inputOffset + i];
        let weightIdx = i * qkvParams.hiddenSize + dimIdx;

        qVal = qVal + inputVal * qWeight[weightIdx];
        kVal = kVal + inputVal * kWeight[weightIdx];
        vVal = vVal + inputVal * vWeight[weightIdx];
    }

    qOut[idx] = qVal;
    kOut[idx] = kVal;
    vOut[idx] = vVal;
}

// Mean pooling over sequence dimension (for sentence embeddings)
struct PoolParams {
    batchSize: u32,
    seqLen: u32,
    hiddenSize: u32,
}

@group(0) @binding(0) var<uniform> poolParams: PoolParams;
@group(0) @binding(1) var<storage, read> poolInput: array<f32>;     // [batch × seq × hidden]
@group(0) @binding(2) var<storage, read> attentionMask: array<f32>; // [batch × seq] - 1.0 for valid, 0.0 for padding
@group(0) @binding(3) var<storage, read_write> poolOutput: array<f32>;  // [batch × hidden]

@compute @workgroup_size(256)
fn mean_pool(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let totalElements = poolParams.batchSize * poolParams.hiddenSize;

    if (idx >= totalElements) {
        return;
    }

    let dimIdx = idx % poolParams.hiddenSize;
    let batchIdx = idx / poolParams.hiddenSize;

    var sum: f32 = 0.0;
    var count: f32 = 0.0;

    for (var seqIdx: u32 = 0u; seqIdx < poolParams.seqLen; seqIdx = seqIdx + 1u) {
        let mask = attentionMask[batchIdx * poolParams.seqLen + seqIdx];
        let inputIdx = (batchIdx * poolParams.seqLen + seqIdx) * poolParams.hiddenSize + dimIdx;

        sum = sum + poolInput[inputIdx] * mask;
        count = count + mask;
    }

    // Average over valid tokens
    if (count > 0.0) {
        poolOutput[idx] = sum / count;
    } else {
        poolOutput[idx] = 0.0;
    }
}

// L2 normalize embeddings (for cosine similarity)
@group(0) @binding(0) var<uniform> normParams: PoolParams;
@group(0) @binding(1) var<storage, read_write> embeddings: array<f32>;  // [batch × hidden]

var<workgroup> wgNorm: array<f32, 256>;

@compute @workgroup_size(256)
fn l2_normalize(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let batchIdx = workgroup_id.x;
    let localIdx = local_id.x;

    if (batchIdx >= normParams.batchSize) {
        return;
    }

    let offset = batchIdx * normParams.hiddenSize;

    // Compute squared sum
    var localSumSq: f32 = 0.0;
    for (var i: u32 = localIdx; i < normParams.hiddenSize; i = i + WORKGROUP_SIZE) {
        let val = embeddings[offset + i];
        localSumSq = localSumSq + val * val;
    }
    wgNorm[localIdx] = localSumSq;
    workgroupBarrier();

    // Reduce
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (localIdx < stride) {
            wgNorm[localIdx] = wgNorm[localIdx] + wgNorm[localIdx + stride];
        }
        workgroupBarrier();
    }

    let invNorm = 1.0 / sqrt(wgNorm[0] + 1e-12);
    workgroupBarrier();

    // Normalize
    for (var i: u32 = localIdx; i < normParams.hiddenSize; i = i + WORKGROUP_SIZE) {
        embeddings[offset + i] = embeddings[offset + i] * invNorm;
    }
}
