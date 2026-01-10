/**
 * LanceQL WebGPU Module
 *
 * GPU-accelerated text and image encoding for semantic search.
 * GPU-accelerated SQL aggregations (SUM, COUNT, AVG, MIN, MAX).
 * GPU-accelerated SQL JOINs (hash join for large tables).
 * GPU-accelerated SQL ORDER BY (bitonic sort for large result sets).
 * GPU-accelerated SQL GROUP BY (hash-based grouping for large datasets).
 * GPU-accelerated vector search (distance computation and top-K selection).
 */

export { GPUBufferPool, getBufferPool } from './gpu-buffer-pool.js';
export { GPUBufferManager, ModelWeightCache } from './gpu-buffers.js';
export { WebGPUAccelerator, getWebGPUAccelerator } from './accelerator.js';
export { GGUFLoader, MODEL_REGISTRY, resolveModelUrl } from './gguf-loader.js';
export {
    GPUTransformer,
    getGPUTransformer,
    encodeText,
    encodeImage,
} from './gpu-transformer.js';
export {
    GPUAggregator,
    getGPUAggregator,
    shouldUseGPU,
} from './gpu-aggregations.js';

// GPU operations using chunked implementations (memory-safe for large datasets)
// These provide backward-compatible aliases (GPUJoiner, GPUSorter, etc.)
export {
    ChunkedGPUJoiner,
    getChunkedGPUJoiner,
    GPUJoiner,
    getGPUJoiner,
    shouldUseGPUJoin,
} from './chunked-gpu-join.js';
export {
    ChunkedGPUSorter,
    getChunkedGPUSorter,
    GPUSorter,
    getGPUSorter,
    shouldUseGPUSort,
} from './chunked-gpu-sort.js';
export {
    ChunkedGPUGrouper,
    getChunkedGPUGrouper,
    GPUGrouper,
    getGPUGrouper,
    shouldUseGPUGroup,
} from './chunked-gpu-group.js';
export {
    ChunkedGPUVectorSearch,
    getChunkedGPUVectorSearch,
    GPUVectorSearch,
    getGPUVectorSearch,
    shouldUseGPUVectorSearch,
    DistanceMetric,
    estimateVectorSearchMemory,
} from './chunked-gpu-vector-search.js';

// OPFS result buffer for streaming large results
export {
    OPFSResultBuffer,
    createTempBuffer,
} from '../cache/opfs-result-buffer.js';

// Shader sources (for advanced users)
export const SHADERS = {
    GEMM: './shaders/gemm.wgsl',
    LAYERNORM: './shaders/layernorm.wgsl',
    GELU: './shaders/gelu.wgsl',
    EMBEDDING: './shaders/embedding.wgsl',
    ATTENTION: './shaders/attention.wgsl',
    REDUCE: './shaders/reduce.wgsl',
    JOIN: './shaders/join.wgsl',
    SORT: './shaders/sort.wgsl',
    GROUP_BY: './shaders/group_by.wgsl',
    VECTOR_DISTANCE: './shaders/vector_distance.wgsl',
    TOPK_SELECT: './shaders/topk_select.wgsl',
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
