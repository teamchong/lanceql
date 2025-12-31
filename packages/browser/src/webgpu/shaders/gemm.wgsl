// GEMM - General Matrix Multiply
// C = A × B where A is [M×K], B is [K×N], C is [M×N]
// Uses tiled algorithm with shared memory for cache efficiency

struct Dimensions {
    M: u32,  // rows of A and C
    N: u32,  // cols of B and C
    K: u32,  // cols of A, rows of B
    alpha: f32,  // scaling factor (usually 1.0)
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE_SIZE: u32 = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn gemm(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let row = global_id.y;
    let col = global_id.x;
    let localRow = local_id.y;
    let localCol = local_id.x;

    var acc: f32 = 0.0;

    // Number of tiles needed to cover K dimension
    let numTiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        // Load tile of A into shared memory
        let aRow = row;
        let aCol = t * TILE_SIZE + localCol;
        if (aRow < dims.M && aCol < dims.K) {
            tileA[localRow][localCol] = A[aRow * dims.K + aCol];
        } else {
            tileA[localRow][localCol] = 0.0;
        }

        // Load tile of B into shared memory
        let bRow = t * TILE_SIZE + localRow;
        let bCol = col;
        if (bRow < dims.K && bCol < dims.N) {
            tileB[localRow][localCol] = B[bRow * dims.N + bCol];
        } else {
            tileB[localRow][localCol] = 0.0;
        }

        // Synchronize to ensure tile is loaded
        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = acc + tileA[localRow][k] * tileB[k][localCol];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = dims.alpha * acc;
    }
}

// GEMM with bias addition: C = A × B + bias
// bias is a 1D array of size N (broadcast across rows)
struct DimensionsWithBias {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
}

@group(0) @binding(4) var<storage, read> bias: array<f32>;

@compute @workgroup_size(16, 16)
fn gemm_bias(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>) {

    let row = global_id.y;
    let col = global_id.x;
    let localRow = local_id.y;
    let localCol = local_id.x;

    var acc: f32 = 0.0;

    let numTiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let aRow = row;
        let aCol = t * TILE_SIZE + localCol;
        if (aRow < dims.M && aCol < dims.K) {
            tileA[localRow][localCol] = A[aRow * dims.K + aCol];
        } else {
            tileA[localRow][localCol] = 0.0;
        }

        let bRow = t * TILE_SIZE + localRow;
        let bCol = col;
        if (bRow < dims.K && bCol < dims.N) {
            tileB[localRow][localCol] = B[bRow * dims.N + bCol];
        } else {
            tileB[localRow][localCol] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = acc + tileA[localRow][k] * tileB[k][localCol];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = dims.alpha * acc + bias[col];
    }
}

// Batched GEMM for processing multiple inputs
// A is [batch × M × K], B is [K × N], C is [batch × M × N]
struct BatchDimensions {
    batch: u32,
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<uniform> batchDims: BatchDimensions;

@compute @workgroup_size(16, 16)
fn gemm_batched(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>) {

    let batchIdx = global_id.z;
    let row = global_id.y;
    let col = global_id.x;
    let localRow = local_id.y;
    let localCol = local_id.x;

    if (batchIdx >= batchDims.batch) {
        return;
    }

    var acc: f32 = 0.0;

    let numTiles = (batchDims.K + TILE_SIZE - 1u) / TILE_SIZE;
    let aOffset = batchIdx * batchDims.M * batchDims.K;
    let cOffset = batchIdx * batchDims.M * batchDims.N;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let aRow = row;
        let aCol = t * TILE_SIZE + localCol;
        if (aRow < batchDims.M && aCol < batchDims.K) {
            tileA[localRow][localCol] = A[aOffset + aRow * batchDims.K + aCol];
        } else {
            tileA[localRow][localCol] = 0.0;
        }

        let bRow = t * TILE_SIZE + localRow;
        let bCol = col;
        if (bRow < batchDims.K && bCol < batchDims.N) {
            tileB[localRow][localCol] = B[bRow * batchDims.N + bCol];
        } else {
            tileB[localRow][localCol] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = acc + tileA[localRow][k] * tileB[k][localCol];
        }

        workgroupBarrier();
    }

    if (row < batchDims.M && col < batchDims.N) {
        C[cOffset + row * batchDims.N + col] = acc;
    }
}
