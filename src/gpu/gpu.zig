//! LanceQL GPU Module - Cross-platform GPU acceleration via wgpu-native
//!
//! Replaces Metal-only backend with cross-platform WebGPU.
//! Shares WGSL shaders with browser for single codebase.
//!
//! Supported platforms:
//! - macOS: Metal backend
//! - Linux: Vulkan backend
//! - Windows: DirectX 12 / Vulkan backend

const std = @import("std");

pub const gpu_context = @import("gpu_context.zig");
pub const GPUContext = gpu_context.GPUContext;
pub const getGlobalContext = gpu_context.getGlobalContext;
pub const releaseGlobalContext = gpu_context.releaseGlobalContext;
pub const isGPUAvailable = gpu_context.isGPUAvailable;
pub const shaders = gpu_context.shaders;

// Re-export wgpu types for convenience
pub const wgpu = @import("wgpu");

/// Threshold for GPU acceleration (use GPU for large datasets)
pub const GPU_THRESHOLD: usize = 10_000;

/// Check if GPU should be used for given data size
pub fn shouldUseGPU(size: usize) bool {
    return size >= GPU_THRESHOLD and isGPUAvailable();
}

/// Platform info string
pub fn getPlatformInfo() []const u8 {
    const builtin = @import("builtin");
    return switch (builtin.os.tag) {
        .macos => "wgpu-native (Metal backend)",
        .linux => "wgpu-native (Vulkan backend)",
        .windows => "wgpu-native (DirectX 12 backend)",
        else => "wgpu-native (unknown backend)",
    };
}

test "platform info" {
    const info = getPlatformInfo();
    std.debug.print("Platform: {s}\n", .{info});
}
