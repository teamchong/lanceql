// Token Embedding Lookup
// Maps token IDs to embedding vectors

struct EmbeddingParams {
    vocabSize: u32,    // Vocabulary size
    embeddingDim: u32, // Embedding dimension
    seqLen: u32,       // Sequence length
    batchSize: u32,    // Batch size
}

@group(0) @binding(0) var<uniform> params: EmbeddingParams;
@group(0) @binding(1) var<storage, read> tokenIds: array<u32>;  // [batch × seqLen]
@group(0) @binding(2) var<storage, read> embeddings: array<f32>;  // [vocab × embeddingDim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>;  // [batch × seqLen × embeddingDim]

@compute @workgroup_size(256)
fn embed_tokens(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let totalTokens = params.batchSize * params.seqLen;

    if (idx >= totalTokens * params.embeddingDim) {
        return;
    }

    // Calculate which token and which dimension
    let tokenIdx = idx / params.embeddingDim;
    let dimIdx = idx % params.embeddingDim;

    // Get token ID
    let tokenId = tokenIds[tokenIdx];

    // Bounds check
    if (tokenId >= params.vocabSize) {
        output[idx] = 0.0;
        return;
    }

    // Look up embedding
    output[idx] = embeddings[tokenId * params.embeddingDim + dimIdx];
}

// Token + Position embedding (BERT-style)
// output = token_embeddings + position_embeddings + segment_embeddings
struct BertEmbeddingParams {
    vocabSize: u32,
    embeddingDim: u32,
    maxPositions: u32,
    seqLen: u32,
    batchSize: u32,
}

@group(0) @binding(0) var<uniform> bertParams: BertEmbeddingParams;
@group(0) @binding(1) var<storage, read> bertTokenIds: array<u32>;
@group(0) @binding(2) var<storage, read> tokenEmbeddings: array<f32>;     // [vocab × dim]
@group(0) @binding(3) var<storage, read> positionEmbeddings: array<f32>;  // [maxPos × dim]
@group(0) @binding(4) var<storage, read_write> bertOutput: array<f32>;

@compute @workgroup_size(256)
fn embed_bert(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let totalElements = bertParams.batchSize * bertParams.seqLen * bertParams.embeddingDim;

    if (idx >= totalElements) {
        return;
    }

    // Calculate indices
    let dimIdx = idx % bertParams.embeddingDim;
    let seqIdx = (idx / bertParams.embeddingDim) % bertParams.seqLen;
    let batchIdx = idx / (bertParams.seqLen * bertParams.embeddingDim);

    // Get token ID
    let tokenIdx = batchIdx * bertParams.seqLen + seqIdx;
    let tokenId = bertTokenIds[tokenIdx];

    // Look up embeddings
    var tokenEmb: f32 = 0.0;
    if (tokenId < bertParams.vocabSize) {
        tokenEmb = tokenEmbeddings[tokenId * bertParams.embeddingDim + dimIdx];
    }

    var posEmb: f32 = 0.0;
    if (seqIdx < bertParams.maxPositions) {
        posEmb = positionEmbeddings[seqIdx * bertParams.embeddingDim + dimIdx];
    }

    // Sum embeddings (segment embedding typically 0 for single sentence)
    bertOutput[idx] = tokenEmb + posEmb;
}

// CLIP-style embedding with [CLS] token handling
// Adds positional embeddings (learned) to patch/token embeddings
struct ClipEmbeddingParams {
    vocabSize: u32,
    embeddingDim: u32,
    contextLen: u32,  // Max context length (77 for CLIP)
    seqLen: u32,      // Actual sequence length
    batchSize: u32,
}

@group(0) @binding(0) var<uniform> clipParams: ClipEmbeddingParams;
@group(0) @binding(1) var<storage, read> clipTokenIds: array<u32>;
@group(0) @binding(2) var<storage, read> clipTokenEmbeddings: array<f32>;
@group(0) @binding(3) var<storage, read> clipPositionalEmbeddings: array<f32>;  // [contextLen × dim]
@group(0) @binding(4) var<storage, read_write> clipOutput: array<f32>;

@compute @workgroup_size(256)
fn embed_clip(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let totalElements = clipParams.batchSize * clipParams.seqLen * clipParams.embeddingDim;

    if (idx >= totalElements) {
        return;
    }

    let dimIdx = idx % clipParams.embeddingDim;
    let seqIdx = (idx / clipParams.embeddingDim) % clipParams.seqLen;
    let batchIdx = idx / (clipParams.seqLen * clipParams.embeddingDim);

    let tokenIdx = batchIdx * clipParams.seqLen + seqIdx;
    let tokenId = clipTokenIds[tokenIdx];

    // Token embedding
    var tokenEmb: f32 = 0.0;
    if (tokenId < clipParams.vocabSize) {
        tokenEmb = clipTokenEmbeddings[tokenId * clipParams.embeddingDim + dimIdx];
    }

    // Positional embedding
    var posEmb: f32 = 0.0;
    if (seqIdx < clipParams.contextLen) {
        posEmb = clipPositionalEmbeddings[seqIdx * clipParams.embeddingDim + dimIdx];
    }

    clipOutput[idx] = tokenEmb + posEmb;
}
