# LanceQL Architecture

## Core Philosophy: "Rich Zig, Thin JS"

LanceQL moves the database engine logic out of JavaScript and into **Zig**. JavaScript acts only as a coordinator and I/O bridge.

-   **Zig:** Parses SQL, plans execution, manages memory, executes queries, performs vector search (SIMD).
-   **JavaScript:** Dispatches queries, handles file handles (OPFS/Node fs), manages WebGPU context.

---

## The Threshold Design (Hybrid Execution)

To handle datasets ranging from kilobytes to gigabytes in the browser, LanceQL employs a **Tiered Execution Strategy**.

### Tier 1: Zig WASM Engine (The Default)
*   **Target:** Small to Medium Data (< 1M rows).
*   **Technology:** WebAssembly + SIMD (128-bit).
*   **Mechanism:**
    *   Zig reads data directly from FileSystem handles (OPFS).
    *   Performs SIMD-accelerated scanning, filtering, and aggregation.
    *   Zero-copy input (reads directly from disk buffers).
*   **Why?**
    *   **Instant Start:** No initialization or data transfer overhead.
    *   **Low Latency:** For datasets under 1M rows, the cost of moving data to the GPU outweighs the compute benefit.
    *   **Universal:** Works in all environments (Browser, Node.js, CLI).

### Tier 2: WebGPU Accelerator (Massive Data)
*   **Target:** Massive Data (> 1M rows).
*   **Technology:** WebGPU (Compute Shaders).
*   **Mechanism:**
    *   Zig detects "Massive" threshold.
    *   Engine delegates specific operators (Vector Search, heavy Matrix Mult) to WebGPU.
    *   Data is streamed to GPU VRAM.
*   **Why?**
    *   **Throughput:** GPU parallelization crushes massive vector scans (Cosine Similarity).
    *   **Off-Main-Thread:** Keeps the UI responsive during long queries.

---

## Execution Modes

### 1. Browser Mode
Designed for high-performance interactive apps.
*   **Engine:** `lanceql.wasm` (Zig).
*   **Storage:** OPFS (Origin Private File System) / HTTP.
*   **Acceleration:** **Enabled**.
    *   Automatic fallback to Tier 2 (WebGPU) when `row_count > 1,000,000` (configurable) and query contains `NEAR`.
    *   Uses **SIMD** for standard aggregations.

### 2. CLI / Node.js Mode
Designed for scripting, build steps, and server-side usage.
*   **Engine:** `lanceql.wasm` (Zig) running in V8/Node.
*   **Storage:** Local Filesystem (`fs`).
*   **Acceleration:** **Disabled** (WebGPU usually unavailable).
    *   Runs exclusively in **Tier 1**.
    *   Relies entirely on CPU SIMD optimization.
    *   *Future:* Native binary builds (bypassing WASM) for AVX-512 support.

---

## Data Flow

```mermaid
graph TD
    SQL[SQL Query] --> JS[JS Coordinator]
    JS -->|Select Mode| Router{Row Count > 1M?}
    
    Router -->|No / CLI| Zig[Zig WASM Engine]
    Router -->|Yes (Browser)| GPU[WebGPU Engine]
    
    subgraph "Tier 1: Zig WASM"
        Zig --> Parser
        Parser --> Planner
        Planner -->|Read| OPFS[File System]
        Planner -->|SIMD| CPU[CPU Execution]
    end
    
    subgraph "Tier 2: WebGPU"
        GPU -->|Upload| VRAM
        VRAM -->|Compute Shader| Cores[GPU Cores]
    end
    
    Zig --> Result
    GPU --> Result
```
