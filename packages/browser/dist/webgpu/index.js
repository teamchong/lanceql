/**
 * LanceQL WebGPU Module
 *
 * GPU-accelerated text and image encoding for semantic search.
 */

export { GPUBufferManager, ModelWeightCache } from './gpu-buffers.js';
export { GGUFLoader, MODEL_REGISTRY, resolveModelUrl } from './gguf-loader.js';
export {
    GPUTransformer,
    getGPUTransformer,
    encodeText,
    encodeImage,
} from './gpu-transformer.js';

// Shader sources (for advanced users)
export const SHADERS = {
    GEMM: './shaders/gemm.wgsl',
    LAYERNORM: './shaders/layernorm.wgsl',
    GELU: './shaders/gelu.wgsl',
    EMBEDDING: './shaders/embedding.wgsl',
    ATTENTION: './shaders/attention.wgsl',
};

/**
 * Check if WebGPU is available.
 * @returns {boolean}
 */
export function isWebGPUAvailable() {
    return typeof navigator !== 'undefined' && !!navigator.gpu;
}

/**
 * Get WebGPU device info.
 * @returns {Promise<Object|null>}
 */
export async function getWebGPUInfo() {
    if (!isWebGPUAvailable()) return null;

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return null;

        const info = await adapter.requestAdapterInfo();
        const features = Array.from(adapter.features);
        const limits = {};

        // Copy relevant limits
        for (const key of ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupsPerDimension']) {
            limits[key] = adapter.limits[key];
        }

        return {
            vendor: info.vendor,
            architecture: info.architecture,
            device: info.device,
            description: info.description,
            features,
            limits,
        };
    } catch (e) {
        return null;
    }
}
