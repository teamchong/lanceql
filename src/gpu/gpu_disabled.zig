//! Inactive GPU module body, used when the build is invoked without -Dgpu=true.
//!
//! When `enable_gpu = false`, build.zig points lanceql.gpu at this file instead of
//! `gpu.zig` so that no source file in the GPU subsystem is parsed (those files
//! `@import("wgpu")`, which requires the wgpu_native_zig dependency).
//!
//! The wasm target does not transitively pull lanceql.gpu, so this module is
//! never compiled in the wasm build path. Native targets that DO transitively
//! pull lanceql.gpu will fail to compile when the project is configured without
//! GPU — that is the intentional contract: native+GPU requires -Dgpu=true.

// Sentinel symbol so any accidental use surfaces a clear compile error.
pub const gpu_disabled: void = {};
