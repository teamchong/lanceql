// GELU Activation and other elementwise operations
// GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Fast approximation: x * sigmoid(1.702 * x)

struct ElementwiseParams {
    size: u32,  // Total number of elements
}

@group(0) @binding(0) var<uniform> params: ElementwiseParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608;  // sqrt(2/pi)

// Standard GELU using tanh approximation
@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let x = input[idx];
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + 0.044715 * x3);

    // tanh approximation: (exp(2x) - 1) / (exp(2x) + 1)
    let exp2x = exp(2.0 * inner);
    let tanh_val = (exp2x - 1.0) / (exp2x + 1.0);

    output[idx] = 0.5 * x * (1.0 + tanh_val);
}

// Fast GELU approximation: x * sigmoid(1.702 * x)
// Faster than tanh version, very close accuracy
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

// In-place GELU (input and output are same buffer)
@group(0) @binding(1) var<storage, read_write> inplace: array<f32>;

@compute @workgroup_size(256)
fn gelu_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let x = inplace[idx];
    let sigmoid = 1.0 / (1.0 + exp(-1.702 * x));
    inplace[idx] = x * sigmoid;
}

// ReLU activation
@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    output[idx] = max(0.0, input[idx]);
}

// SiLU (Swish) activation: x * sigmoid(x)
@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let x = input[idx];
    output[idx] = x / (1.0 + exp(-x));
}

// Vectorized element-wise add: C = A + B
@group(0) @binding(3) var<storage, read> addB: array<f32>;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    output[idx] = input[idx] + addB[idx];
}

// Scale: y = x * scale
struct ScaleParams {
    size: u32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> scaleParams: ScaleParams;

@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scaleParams.size) {
        return;
    }

    output[idx] = input[idx] * scaleParams.scale;
}
