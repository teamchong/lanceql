//
// Metal backend for LanceQL - Objective-C wrapper
//
// Provides C API for Zig to call Metal GPU operations
// Uses PRECOMPILED Metal shaders (.metallib) for faster startup
//
// Build shaders: make metal-shaders (or zig build metal-shaders)
//

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Global Metal state
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_cosine_pipeline = nil;
static id<MTLComputePipelineState> g_dot_pipeline = nil;
static id<MTLComputePipelineState> g_l2_pipeline = nil;
static bool g_initialized = false;

// Path to precompiled Metal library (set by build system)
static const char* g_metallib_path = NULL;

// Set path to precompiled .metallib file
void lanceql_metal_set_library_path(const char* path) {
    g_metallib_path = path;
}

// Initialize Metal with precompiled shaders
int lanceql_metal_init(void) {
    @autoreleasepool {
        if (g_initialized) return 0;

        // Get default Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            NSLog(@"LanceQL Metal: No GPU device found");
            return -1;
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            NSLog(@"LanceQL Metal: Failed to create command queue");
            return -2;
        }

        NSError* error = nil;

        // Try to load precompiled .metallib
        if (g_metallib_path) {
            NSString* path = [NSString stringWithUTF8String:g_metallib_path];
            NSURL* url = [NSURL fileURLWithPath:path];
            g_library = [g_device newLibraryWithURL:url error:&error];
            if (g_library) {
                NSLog(@"LanceQL Metal: Loaded precompiled library from %@", path);
            }
        }

        // Search common paths if not found
        if (!g_library) {
            NSArray* searchPaths = @[
                @"vector_search.metallib",
                @"zig-out/lib/vector_search.metallib",
                @"src/metal/vector_search.metallib",
                [[NSBundle mainBundle] pathForResource:@"vector_search" ofType:@"metallib"] ?: @""
            ];

            for (NSString* path in searchPaths) {
                if ([path length] == 0) continue;
                NSURL* url = [NSURL fileURLWithPath:path];
                g_library = [g_device newLibraryWithURL:url error:&error];
                if (g_library) {
                    NSLog(@"LanceQL Metal: Loaded precompiled library from %@", path);
                    break;
                }
            }
        }

        // Fallback: compile at runtime if no precompiled library found
        if (!g_library) {
            NSLog(@"LanceQL Metal: No precompiled .metallib found, compiling at runtime...");

            NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void cosine_similarity_batch(
    device const float* query [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float dot = 0.0f;
    float query_norm = 0.0f;
    float vec_norm = 0.0f;
    device const float* vec = vectors + gid * dim;
    for (uint i = 0; i < dim; i++) {
        float q = query[i];
        float v = vec[i];
        dot += q * v;
        query_norm += q * q;
        vec_norm += v * v;
    }
    float denom = sqrt(query_norm) * sqrt(vec_norm);
    scores[gid] = (denom > 0.0f) ? (dot / denom) : 0.0f;
}

kernel void dot_product_batch(
    device const float* query [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float dot = 0.0f;
    device const float* vec = vectors + gid * dim;
    for (uint i = 0; i < dim; i++) {
        dot += query[i] * vec[i];
    }
    scores[gid] = dot;
}

kernel void l2_distance_batch(
    device const float* query [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float dist = 0.0f;
    device const float* vec = vectors + gid * dim;
    for (uint i = 0; i < dim; i++) {
        float diff = query[i] - vec[i];
        dist += diff * diff;
    }
    scores[gid] = dist;
}
)";
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.fastMathEnabled = YES;
            g_library = [g_device newLibraryWithSource:shaderSource options:options error:&error];

            if (!g_library) {
                NSLog(@"LanceQL Metal: Shader compile error: %@", error);
                return -3;
            }
        }

        // Create compute pipelines
        id<MTLFunction> cosine_fn = [g_library newFunctionWithName:@"cosine_similarity_batch"];
        id<MTLFunction> dot_fn = [g_library newFunctionWithName:@"dot_product_batch"];
        id<MTLFunction> l2_fn = [g_library newFunctionWithName:@"l2_distance_batch"];

        if (cosine_fn) {
            g_cosine_pipeline = [g_device newComputePipelineStateWithFunction:cosine_fn error:&error];
            if (!g_cosine_pipeline) {
                NSLog(@"LanceQL Metal: Pipeline error: %@", error);
            }
        }
        if (dot_fn) {
            g_dot_pipeline = [g_device newComputePipelineStateWithFunction:dot_fn error:&error];
        }
        if (l2_fn) {
            g_l2_pipeline = [g_device newComputePipelineStateWithFunction:l2_fn error:&error];
        }

        g_initialized = true;
        NSLog(@"LanceQL Metal: Initialized GPU '%@' with %d kernels",
              g_device.name,
              (g_cosine_pipeline ? 1 : 0) + (g_dot_pipeline ? 1 : 0) + (g_l2_pipeline ? 1 : 0));

        return 0;
    }
}

// Load precompiled .metallib from specific path
int lanceql_metal_load_library(const char* path) {
    lanceql_metal_set_library_path(path);
    return lanceql_metal_init();
}

// Batch cosine similarity on GPU (zero-copy on Apple Silicon unified memory)
int lanceql_metal_cosine_batch(
    const float* query,
    const float* vectors,
    float* scores,
    unsigned int dim,
    unsigned int num_vectors
) {
    @autoreleasepool {
        if (!g_initialized) {
            if (lanceql_metal_init() != 0) return -1;
        }
        if (!g_cosine_pipeline) return -1;

        size_t query_size = dim * sizeof(float);
        size_t vectors_size = (size_t)num_vectors * dim * sizeof(float);
        size_t scores_size = num_vectors * sizeof(float);

        // Zero-copy buffers - unified memory on Apple Silicon
        id<MTLBuffer> query_buf = [g_device newBufferWithBytesNoCopy:(void*)query
                                                              length:query_size
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
        id<MTLBuffer> vectors_buf = [g_device newBufferWithBytesNoCopy:(void*)vectors
                                                                length:vectors_size
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];
        id<MTLBuffer> scores_buf = [g_device newBufferWithBytesNoCopy:(void*)scores
                                                               length:scores_size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        if (!query_buf || !vectors_buf || !scores_buf) {
            // Fallback to copying if zero-copy fails (alignment issues)
            query_buf = [g_device newBufferWithBytes:query length:query_size options:MTLResourceStorageModeShared];
            vectors_buf = [g_device newBufferWithBytes:vectors length:vectors_size options:MTLResourceStorageModeShared];
            scores_buf = [g_device newBufferWithLength:scores_size options:MTLResourceStorageModeShared];
        }

        id<MTLCommandBuffer> cmd_buf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:g_cosine_pipeline];
        [encoder setBuffer:query_buf offset:0 atIndex:0];
        [encoder setBuffer:vectors_buf offset:0 atIndex:1];
        [encoder setBuffer:scores_buf offset:0 atIndex:2];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:3];

        MTLSize grid_size = MTLSizeMake(num_vectors, 1, 1);
        NSUInteger thread_group_size = MIN(g_cosine_pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)num_vectors);
        thread_group_size = MIN(thread_group_size, 256);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        // Copy results if we fell back to non-zero-copy buffers
        if (scores_buf.contents != scores) {
            memcpy(scores, scores_buf.contents, scores_size);
        }

        return 0;
    }
}

// Batch dot product on GPU
int lanceql_metal_dot_batch(
    const float* query,
    const float* vectors,
    float* scores,
    unsigned int dim,
    unsigned int num_vectors
) {
    @autoreleasepool {
        if (!g_initialized) {
            if (lanceql_metal_init() != 0) return -1;
        }
        if (!g_dot_pipeline) return -1;

        size_t query_size = dim * sizeof(float);
        size_t vectors_size = (size_t)num_vectors * dim * sizeof(float);
        size_t scores_size = num_vectors * sizeof(float);

        id<MTLBuffer> query_buf = [g_device newBufferWithBytes:query length:query_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> vectors_buf = [g_device newBufferWithBytes:vectors length:vectors_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buf = [g_device newBufferWithLength:scores_size options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd_buf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:g_dot_pipeline];
        [encoder setBuffer:query_buf offset:0 atIndex:0];
        [encoder setBuffer:vectors_buf offset:0 atIndex:1];
        [encoder setBuffer:scores_buf offset:0 atIndex:2];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:3];

        MTLSize grid_size = MTLSizeMake(num_vectors, 1, 1);
        NSUInteger thread_group_size = MIN(g_dot_pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)num_vectors);
        thread_group_size = MIN(thread_group_size, 256);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        memcpy(scores, scores_buf.contents, scores_size);
        return 0;
    }
}

// Batch L2 distance on GPU
int lanceql_metal_l2_batch(
    const float* query,
    const float* vectors,
    float* scores,
    unsigned int dim,
    unsigned int num_vectors
) {
    @autoreleasepool {
        if (!g_initialized) {
            if (lanceql_metal_init() != 0) return -1;
        }
        if (!g_l2_pipeline) return -1;

        size_t query_size = dim * sizeof(float);
        size_t vectors_size = (size_t)num_vectors * dim * sizeof(float);
        size_t scores_size = num_vectors * sizeof(float);

        id<MTLBuffer> query_buf = [g_device newBufferWithBytes:query length:query_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> vectors_buf = [g_device newBufferWithBytes:vectors length:vectors_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buf = [g_device newBufferWithLength:scores_size options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd_buf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:g_l2_pipeline];
        [encoder setBuffer:query_buf offset:0 atIndex:0];
        [encoder setBuffer:vectors_buf offset:0 atIndex:1];
        [encoder setBuffer:scores_buf offset:0 atIndex:2];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:3];

        MTLSize grid_size = MTLSizeMake(num_vectors, 1, 1);
        NSUInteger thread_group_size = MIN(g_l2_pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)num_vectors);
        thread_group_size = MIN(thread_group_size, 256);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        memcpy(scores, scores_buf.contents, scores_size);
        return 0;
    }
}

// Cleanup
void lanceql_metal_cleanup(void) {
    g_cosine_pipeline = nil;
    g_dot_pipeline = nil;
    g_l2_pipeline = nil;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
    g_initialized = false;
}

// Check if Metal is available
int lanceql_metal_available(void) {
    if (!g_initialized) {
        lanceql_metal_init();
    }
    return g_initialized ? 1 : 0;
}

// Get device name
const char* lanceql_metal_device_name(void) {
    if (!g_device) return "No device";
    return [g_device.name UTF8String];
}

// Check if using precompiled shaders
int lanceql_metal_is_precompiled(void) {
    return (g_metallib_path != NULL) ? 1 : 0;
}
