//
// Metal backend for LanceQL - Objective-C wrapper
//
// Provides C API for Zig to call Metal GPU operations
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

// Initialize Metal
int lanceql_metal_init(void) {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            NSLog(@"Metal: No GPU device found");
            return -1;
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            NSLog(@"Metal: Failed to create command queue");
            return -2;
        }

        NSLog(@"Metal: Initialized with %@", g_device.name);
        return 0;
    }
}

// Load shader library from .metallib file
int lanceql_metal_load_library(const char* path) {
    @autoreleasepool {
        if (!g_device) return -1;

        NSError* error = nil;
        NSString* nsPath = [NSString stringWithUTF8String:path];
        NSURL* url = [NSURL fileURLWithPath:nsPath];

        g_library = [g_device newLibraryWithURL:url error:&error];
        if (!g_library) {
            NSLog(@"Metal: Failed to load library: %@", error);
            return -2;
        }

        // Create pipelines
        id<MTLFunction> cosine_fn = [g_library newFunctionWithName:@"cosine_similarity_batch"];
        id<MTLFunction> dot_fn = [g_library newFunctionWithName:@"dot_product_batch"];
        id<MTLFunction> l2_fn = [g_library newFunctionWithName:@"l2_distance_batch"];

        if (cosine_fn) {
            g_cosine_pipeline = [g_device newComputePipelineStateWithFunction:cosine_fn error:&error];
        }
        if (dot_fn) {
            g_dot_pipeline = [g_device newComputePipelineStateWithFunction:dot_fn error:&error];
        }
        if (l2_fn) {
            g_l2_pipeline = [g_device newComputePipelineStateWithFunction:l2_fn error:&error];
        }

        NSLog(@"Metal: Loaded library with %d functions",
              (g_cosine_pipeline ? 1 : 0) + (g_dot_pipeline ? 1 : 0) + (g_l2_pipeline ? 1 : 0));
        return 0;
    }
}

// Batch cosine similarity on GPU
int lanceql_metal_cosine_batch(
    const float* query,      // [dim]
    const float* vectors,    // [num_vectors * dim]
    float* scores,           // [num_vectors]
    unsigned int dim,
    unsigned int num_vectors
) {
    @autoreleasepool {
        if (!g_device || !g_queue || !g_cosine_pipeline) {
            return -1;
        }

        // Create buffers
        size_t query_size = dim * sizeof(float);
        size_t vectors_size = num_vectors * dim * sizeof(float);
        size_t scores_size = num_vectors * sizeof(float);

        id<MTLBuffer> query_buf = [g_device newBufferWithBytes:query
                                                        length:query_size
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> vectors_buf = [g_device newBufferWithBytes:vectors
                                                          length:vectors_size
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buf = [g_device newBufferWithLength:scores_size
                                                         options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> cmd_buf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:g_cosine_pipeline];
        [encoder setBuffer:query_buf offset:0 atIndex:0];
        [encoder setBuffer:vectors_buf offset:0 atIndex:1];
        [encoder setBuffer:scores_buf offset:0 atIndex:2];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:3];

        // Dispatch threads
        MTLSize grid_size = MTLSizeMake(num_vectors, 1, 1);
        NSUInteger thread_group_size = MIN(g_cosine_pipeline.maxTotalThreadsPerThreadgroup, num_vectors);
        MTLSize threadgroup_size = MTLSizeMake(thread_group_size, 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        // Execute and wait
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        // Copy results back
        memcpy(scores, scores_buf.contents, scores_size);

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
        if (!g_device || !g_queue || !g_dot_pipeline) {
            return -1;
        }

        size_t query_size = dim * sizeof(float);
        size_t vectors_size = num_vectors * dim * sizeof(float);
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
        NSUInteger thread_group_size = MIN(g_dot_pipeline.maxTotalThreadsPerThreadgroup, num_vectors);
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
}

// Check if Metal is available
int lanceql_metal_available(void) {
    return (g_device != nil && g_queue != nil) ? 1 : 0;
}

// Get device name
const char* lanceql_metal_device_name(void) {
    if (!g_device) return "No device";
    return [g_device.name UTF8String];
}
