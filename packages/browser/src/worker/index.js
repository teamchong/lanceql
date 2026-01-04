/**
 * LanceQL Unified SharedWorker
 *
 * Handles all OPFS operations and heavy computation in a single worker shared across tabs:
 * - Store operations (key-value storage)
 * - LocalDatabase operations (SQL, Lance tables)
 * - WebGPU transformer (shared model for embeddings)
 *
 * Uses Zig WASM for SIMD-accelerated aggregations (sum, min, max, avg).
 * Benefits: shared GPU model, responsive UI, efficient OPFS access via createSyncAccessHandle
 */

import { WorkerStore } from './worker-store.js';
import { WorkerDatabase } from './worker-database.js';
import { WorkerVault } from './worker-vault.js';
import { executeSQL, evalWhere } from './sql/executor.js';
import { E } from './data-types.js';

// ============================================================================
// WASM Module (Zig SIMD aggregations)
// ============================================================================

let wasm = null;
let wasmMemory = null;

async function loadWasm() {
    if (wasm) return wasm;
    try {
        // Load WASM from same directory as worker
        const response = await fetch(new URL('./lanceql.wasm', import.meta.url));
        const bytes = await response.arrayBuffer();
        const module = await WebAssembly.instantiate(bytes, {});
        wasm = module.instance.exports;
        wasmMemory = wasm.memory;
        console.log('[LanceQLWorker] WASM loaded');
        return wasm;
    } catch (e) {
        console.warn('[LanceQLWorker] WASM not available:', e.message);
        return null;
    }
}

// Export WASM for executor
export function getWasm() { return wasm; }
export function getWasmMemory() { return wasmMemory; }

// Initialize WASM on load
loadWasm();

// ============================================================================
// Shared State
// ============================================================================

// Store instances (one per store name)
const stores = new Map();

// Database instances (one per database name)
const databases = new Map();

// Vault instance (singleton)
let vaultInstance = null;

// Connected ports
const ports = new Set();

// SharedArrayBuffer for zero-copy responses (set by main thread)
let sharedBuffer = null;
let sharedOffset = 0;

// Large response threshold (use shared buffer for responses > 1KB)
const SHARED_THRESHOLD = 1024;

// ============================================================================
// Get or create instances
// ============================================================================

async function getVault(encryptionConfig = null) {
    if (!vaultInstance) {
        vaultInstance = new WorkerVault();
    }
    // Re-open with encryption if not already done
    await vaultInstance.open(encryptionConfig);
    return vaultInstance;
}

async function getStore(name, options = {}, encryptionConfig = null) {
    // Include encryption key ID in cache key (but not the actual key bytes)
    const encKeyId = encryptionConfig?.keyId || 'none';
    const key = `${name}:${encKeyId}:${JSON.stringify(options)}`;
    if (!stores.has(key)) {
        const store = new WorkerStore(name, options);
        await store.open(encryptionConfig);
        stores.set(key, store);
    }
    return stores.get(key);
}

async function getDatabase(name) {
    if (!databases.has(name)) {
        const db = new WorkerDatabase(name);
        await db.open();
        databases.set(name, db);
    }
    return databases.get(name);
}

// ============================================================================
// Message Handler
// ============================================================================

/**
 * Send response, using SharedArrayBuffer for large data if available.
 */
function sendResponse(port, id, result) {
    // Try SharedArrayBuffer for large responses
    if (sharedBuffer && result !== undefined) {
        const json = JSON.stringify(result);
        if (json.length > SHARED_THRESHOLD) {
            const bytes = E.encode(json);
            if (sharedOffset + bytes.length <= sharedBuffer.byteLength) {
                // Write to shared buffer
                const view = new Uint8Array(sharedBuffer, sharedOffset, bytes.length);
                view.set(bytes);

                port.postMessage({
                    id,
                    sharedOffset,
                    sharedLength: bytes.length
                });

                sharedOffset += bytes.length;
                // Reset offset if we're past halfway (simple ring buffer)
                if (sharedOffset > sharedBuffer.byteLength / 2) {
                    sharedOffset = 0;
                }
                return;
            }
        }
    }

    // Fall back to regular postMessage
    port.postMessage({ id, result });
}

async function handleMessage(port, data) {
    // Handle SharedArrayBuffer initialization
    if (data.type === 'initSharedBuffer') {
        sharedBuffer = data.buffer;
        sharedOffset = 0;
        console.log('[LanceQLWorker] SharedArrayBuffer initialized:', sharedBuffer.byteLength, 'bytes');
        return;
    }

    const { id, method, args } = data;

    try {
        let result;

        // Store operations
        if (method === 'ping') {
            result = 'pong';
        } else if (method === 'open') {
            await getStore(args.name, args.options, args.encryption);
            result = true;
        } else if (method === 'get') {
            result = await (await getStore(args.name)).get(args.key);
        } else if (method === 'set') {
            await (await getStore(args.name)).set(args.key, args.value);
            result = true;
        } else if (method === 'delete') {
            await (await getStore(args.name)).delete(args.key);
            result = true;
        } else if (method === 'keys') {
            result = await (await getStore(args.name)).keys();
        } else if (method === 'clear') {
            await (await getStore(args.name)).clear();
            result = true;
        } else if (method === 'filter') {
            result = await (await getStore(args.name)).filter(args.key, args.query);
        } else if (method === 'find') {
            result = await (await getStore(args.name)).find(args.key, args.query);
        } else if (method === 'search') {
            result = await (await getStore(args.name)).search(args.key, args.text, args.limit);
        } else if (method === 'count') {
            result = await (await getStore(args.name)).count(args.key, args.query);
        } else if (method === 'enableSemanticSearch') {
            result = await (await getStore(args.name)).enableSemanticSearch(args.options);
        } else if (method === 'disableSemanticSearch') {
            (await getStore(args.name)).disableSemanticSearch();
            result = true;
        } else if (method === 'hasSemanticSearch') {
            result = (await getStore(args.name)).hasSemanticSearch();
        }
        // Database operations
        else if (method === 'db:open') {
            await getDatabase(args.name);
            result = true;
        } else if (method === 'db:createTable') {
            result = await (await getDatabase(args.db)).createTable(args.tableName, args.columns, args.ifNotExists);
        } else if (method === 'db:dropTable') {
            result = await (await getDatabase(args.db)).dropTable(args.tableName, args.ifExists);
        } else if (method === 'db:insert') {
            result = await (await getDatabase(args.db)).insert(args.tableName, args.rows);
        } else if (method === 'db:delete') {
            // Note: predicate function is serialized - need to recreate
            const db = await getDatabase(args.db);
            const predicate = args.where
                ? (row) => evalWhere(args.where, row)
                : () => true;
            result = await db.delete(args.tableName, predicate);
        } else if (method === 'db:update') {
            const db = await getDatabase(args.db);
            const predicate = args.where
                ? (row) => evalWhere(args.where, row)
                : () => true;
            result = await db.update(args.tableName, args.updates, predicate);
        } else if (method === 'db:select') {
            const db = await getDatabase(args.db);
            const options = { ...args.options };
            if (args.where) {
                options.where = (row) => evalWhere(args.where, row);
            }
            result = await db.select(args.tableName, options);
        } else if (method === 'db:exec') {
            result = await executeSQL(await getDatabase(args.db), args.sql);
        } else if (method === 'db:flush') {
            await (await getDatabase(args.db)).flush();
            result = true;
        } else if (method === 'db:compact') {
            result = await (await getDatabase(args.db)).compact();
        } else if (method === 'db:listTables') {
            result = (await getDatabase(args.db)).listTables();
        } else if (method === 'db:getTable') {
            result = (await getDatabase(args.db)).getTable(args.tableName);
        } else if (method === 'db:scanStart') {
            result = await (await getDatabase(args.db)).scanStart(args.tableName, args.options);
        } else if (method === 'db:scanNext') {
            const db = await getDatabase(args.db);
            result = db.scanNext(args.streamId);
        }
        // Vault operations
        else if (method === 'vault:open') {
            await getVault(args.encryption);
            result = true;
        } else if (method === 'vault:get') {
            result = await (await getVault()).get(args.key);
        } else if (method === 'vault:set') {
            await (await getVault()).set(args.key, args.value);
            result = true;
        } else if (method === 'vault:delete') {
            await (await getVault()).delete(args.key);
            result = true;
        } else if (method === 'vault:keys') {
            result = await (await getVault()).keys();
        } else if (method === 'vault:has') {
            result = await (await getVault()).has(args.key);
        } else if (method === 'vault:exec') {
            result = await (await getVault()).exec(args.sql);
        }
        // Unknown method
        else {
            throw new Error(`Unknown method: ${method}`);
        }

        sendResponse(port, id, result);
    } catch (error) {
        port.postMessage({ id, error: error.message });
    }
}

// ============================================================================
// Worker Entry Points
// ============================================================================

// SharedWorker connection handler
self.onconnect = (event) => {
    const port = event.ports[0];
    ports.add(port);

    port.onmessage = (e) => {
        handleMessage(port, e.data);
    };

    port.onmessageerror = (e) => {
        console.error('[LanceQLWorker] Message error:', e);
    };

    // Worker is ready
    port.postMessage({ type: 'ready' });
    port.start();

    console.log('[LanceQLWorker] New connection, total ports:', ports.size);
};

// Regular Worker fallback (when SharedWorker not available)
self.onmessage = (e) => {
    handleMessage(self, e.data);
};

console.log('[LanceQLWorker] Initialized');
