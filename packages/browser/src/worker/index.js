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

import { BufferPool } from './buffer-pool.js';
import { executeSQL, evalWhere } from './sql/executor.js';
import { getWasmSqlExecutor, shouldUseWasmSql } from './wasm-sql-bridge.js';
import { E } from './data-types.js';

// ============================================================================
// WASM SQL Execution (execute queries entirely in WASM for zero-copy transfer)
// ============================================================================

/**
 * Execute SQL query entirely in WASM using WasmSqlExecutor
 * Supports full SQL: JOINs, CTEs, Window functions, Set operations
 * @param {WorkerDatabase} db - Database instance
 * @param {string} sql - SQL query
 * @returns {Object} - Result with columnar format
 */
async function executeWasmSqlFull(db, sql) {
    if (!wasm) {
        await loadWasm();
        if (!wasm) throw new Error('WASM not loaded');
    }

    const executor = getWasmSqlExecutor();

    // Extract all table names from SQL (FROM, JOIN, WITH clauses)
    const tableNames = extractTableNames(sql);

    // Register all tables with the executor
    // Reworked Logic Body
    for (const tableName of tableNames) {
        const table = db.tables.get(tableName);
        if (!table) continue;

        const colBuf = db._columnarBuffer?.get(tableName);
        const bufLen = colBuf?.__length || 0;
        const version = `${tableName}:${table.fragments?.length || 0}:${bufLen}:${table.deletionVector?.length || 0}`;

        const hasFiles = table.fragments.length > 0;
        const hasMemory = bufLen > 0;

        if (hasFiles) {
            // 1. Register Files (Zero-Copy)
            const handles = [];
            for (const fragPath of table.fragments) {
                const handleId = await registerOPFSFile(fragPath);
                if (handleId) handles.push(handleId);
            }
            executor.registerTableFromFiles(tableName, table.fragments, version);

            // 2. Append Memory (Hybrid)
            if (hasMemory) {
                const columns = {};
                for (const c of table.schema) {
                    const arr = colBuf[c.name];
                    if (arr && (ArrayBuffer.isView(arr))) {
                        columns[c.name] = arr.subarray(0, bufLen);
                    }
                }
                executor.appendTableMemory(tableName, columns, bufLen);
            }
        } else if (hasMemory) {
            // Pure Memory Table -> Full Register
            const columnarData = await db.selectColumnar(tableName);
            if (columnarData) {
                const { columns, rowCount } = columnarData;
                executor.registerTable(tableName, columns, rowCount, version);
            }
        }
    }

    // Execute SQL in WASM
    return executor.execute(sql);
}

/**
 * Extract all table names referenced in SQL query
 */
function extractTableNames(sql) {
    const names = new Set();
    const upper = sql.toUpperCase();

    // Match FROM tableName
    const fromMatches = sql.matchAll(/FROM\s+(\w+)/gi);
    for (const m of fromMatches) names.add(m[1].toLowerCase());

    // Match JOIN tableName
    const joinMatches = sql.matchAll(/JOIN\s+(\w+)/gi);
    for (const m of joinMatches) names.add(m[1].toLowerCase());

    // Match UPDATE tableName
    const updateMatch = sql.match(/UPDATE\s+(\w+)/i);
    if (updateMatch) names.add(updateMatch[1].toLowerCase());

    // Match INSERT INTO tableName
    const insertMatch = sql.match(/INSERT\s+INTO\s+(\w+)/i);
    if (insertMatch) names.add(insertMatch[1].toLowerCase());

    // Match DELETE FROM tableName
    const deleteMatch = sql.match(/DELETE\s+FROM\s+(\w+)/i);
    if (deleteMatch) names.add(deleteMatch[1].toLowerCase());

    return Array.from(names);
}

// ============================================================================
// WASM Module (Zig SIMD aggregations with direct OPFS access)
// ============================================================================

let wasm = null;
let wasmMemory = null;

// OPFS sync access handles (indexed by handle ID)
const opfsHandles = new Map();
let nextHandleId = 1;

/**
 * Get directory handle for path, creating directories as needed
 */
async function getOPFSDir(pathParts) {
    let current = await navigator.storage.getDirectory();
    current = await current.getDirectoryHandle('lanceql', { create: true });
    for (const part of pathParts) {
        current = await current.getDirectoryHandle(part, { create: true });
    }
    return current;
}

/**
 * OPFS imports for WASM - provides synchronous file access via FileSystemSyncAccessHandle
 */
function createWasmImports() {
    return {
        env: {
            // Open file, returns handle ID (0 = error)
            opfs_open: (pathPtr, pathLen) => {
                try {
                    // Read path from WASM memory
                    const pathBytes = new Uint8Array(wasmMemory.buffer, pathPtr, pathLen);
                    const path = new TextDecoder().decode(pathBytes);

                    // Look up pre-opened handle ID
                    for (const [id, handle] of opfsHandles.entries()) {
                        // We store the path on the handle object for lookup
                        if (handle._path === path) {
                            return id;
                        }
                    }
                    console.warn('[LanceQLWorker] WASM tried to open unregistered path:', path);
                    return 0;
                } catch (e) {
                    // This catch block is within the WASM import function, not a message handler.
                    // 'port' and 'id' are not defined here.
                    // The instruction seems to imply a message handler context, but the snippet
                    // targets this specific location.
                    // To maintain syntactical correctness and faithfulness to the snippet's location,
                    // I'm applying the console.error change.
                    // If 'port' and 'id' were intended to be available, the scope would need
                    // to be adjusted, which is a larger change than just modifying the catch block.
                    console.error('[LanceQLWorker] Error:', e);
                    // port.postMessage({ id, error: e.stack || e.toString() }); // This line would cause a ReferenceError
                    return 0;
                }
            },
            // Read from file at offset into buffer
            opfs_read: (handle, bufPtr, bufLen, offset) => {
                const accessHandle = opfsHandles.get(handle);
                if (!accessHandle) return 0;
                try {
                    const buf = new Uint8Array(wasmMemory.buffer, bufPtr, bufLen);
                    return accessHandle.read(buf, { at: Number(offset) });
                } catch (e) {
                    return 0;
                }
            },
            // Get file size
            opfs_size: (handle) => {
                const accessHandle = opfsHandles.get(handle);
                if (!accessHandle) return BigInt(0);
                try {
                    return BigInt(accessHandle.getSize());
                } catch (e) {
                    return BigInt(0);
                }
            },
            // Close file handle
            opfs_close: (handle) => {
                // We keep handles open for the duration of the query/registration
                // and manage them via registerOPFSFile/closeOPFSFile
            },
            js_log: (ptr, len) => {
                try {
                    const bytes = new Uint8Array(wasmMemory.buffer, ptr, len);
                    const msg = new TextDecoder().decode(bytes);
                    console.log(`[LanceQLWasm] ${msg}`);
                } catch (e) {
                    console.error('[LanceQLWasm] Log error:', e);
                }
            },
            __assert_fail: (msgPtr, filePtr, line, funcPtr) => {
                const decoder = new TextDecoder();
                const msg = decoder.decode(new Uint8Array(wasmMemory.buffer, msgPtr).subarray(0, 100));
                console.error(`[WASM ASSERT] ${msg} at line ${line}`);
            }
        }
    };
}

/**
 * Pre-open OPFS file and register handle for WASM access
 * Call this before WASM operations that need the file
 */
export async function registerOPFSFile(path) {
    try {
        const parts = path.split('/').filter(p => p);
        const fileName = parts.pop();
        const dir = await getOPFSDir(parts);
        const fileHandle = await dir.getFileHandle(fileName);
        const accessHandle = await fileHandle.createSyncAccessHandle();

        const handleId = nextHandleId++;
        accessHandle._path = path; // Attach path for lookup in opfs_open
        opfsHandles.set(handleId, accessHandle);
        return handleId;
    } catch (e) {
        console.warn('[LanceQLWorker] Failed to register OPFS file:', path, e);
        return 0;
    }
}

/**
 * Close registered OPFS file
 */
export function closeOPFSFile(handleId) {
    const accessHandle = opfsHandles.get(handleId);
    if (accessHandle) {
        try { accessHandle.close(); } catch (e) { }
        opfsHandles.delete(handleId);
    }
}

async function loadWasm() {
    if (wasm) return wasm;
    try {
        const response = await fetch(new URL('./lanceql.wasm', import.meta.url));
        const bytes = await response.arrayBuffer();

        // Instantiate with OPFS imports
        const imports = createWasmImports();
        const module = await WebAssembly.instantiate(bytes, imports);
        wasm = module.instance.exports;
        wasmMemory = wasm.memory;

        console.log('[LanceQLWorker] WASM loaded with OPFS support');
        return wasm;
    } catch (e) {
        console.warn('[LanceQLWorker] WASM not available:', e.message);
        return null;
    }
}

// Export WASM for executor
export function getWasm() { return wasm; }
export function getWasmMemory() { return wasmMemory; }

// Reusable WASM buffer pool to reduce allocations
let wasmBufferPtr = 0;
let wasmBufferSize = 0;
const MIN_BUFFER_SIZE = 1024 * 1024; // 1MB minimum

/**
 * Get a WASM buffer of at least the specified size (reuses existing if big enough)
 */
function getWasmBuffer(size) {
    if (!wasm) return 0;
    if (size <= wasmBufferSize && wasmBufferPtr !== 0) {
        return wasmBufferPtr;
    }
    // Allocate new buffer (at least MIN_BUFFER_SIZE)
    const newSize = Math.max(size, MIN_BUFFER_SIZE);
    const ptr = wasm.alloc(newSize);
    if (ptr) {
        wasmBufferPtr = ptr;
        wasmBufferSize = newSize;
    }
    return ptr;
}

/**
 * Load fragment directly into WASM memory from OPFS
 * Returns column count on success, 0 on failure
 */
export async function loadFragmentToWasm(fragPath) {
    const w = await loadWasm();
    if (!w) return 0;

    try {
        // Open OPFS file with sync access handle
        const parts = fragPath.split('/').filter(p => p);
        const fileName = parts.pop();
        const dir = await getOPFSDir(parts);
        const fileHandle = await dir.getFileHandle(fileName);
        const accessHandle = await fileHandle.createSyncAccessHandle();

        // Get size and get reusable buffer
        const size = accessHandle.getSize();
        const ptr = getWasmBuffer(size);
        if (!ptr) {
            accessHandle.close();
            return 0;
        }

        // Read directly into WASM memory
        const wasmBuf = new Uint8Array(wasmMemory.buffer, ptr, size);
        const bytesRead = accessHandle.read(wasmBuf, { at: 0 });
        accessHandle.close();

        if (bytesRead !== size) return 0;

        // Open as Lance file
        return w.openFile(ptr, size);
    } catch (e) {
        console.warn('[LanceQLWorker] Failed to load fragment:', fragPath, e);
        return 0;
    }
}

// Helper to get a file handle from a path
async function getFileHandle(root, path) {
    const parts = path.split('/').filter(p => p);
    let currentDir = root;
    for (let i = 0; i < parts.length - 1; i++) {
        currentDir = await currentDir.getDirectoryHandle(parts[i], { create: false });
    }
    return await currentDir.getFileHandle(parts[parts.length - 1], { create: false });
}

let opfsRoot = null;

/**
 * Load fragment with caching
 */
async function loadFragmentCached(fragPath) {
    // Check global buffer pool
    let loaded = bufferPool.get(fragPath);
    if (loaded) return loaded;

    if (!opfsRoot) {
        opfsRoot = await navigator.storage.getDirectory();
    }

    // Load from OPFS
    const handle = await getFileHandle(opfsRoot, fragPath);
    const file = await handle.getFile();
    const buffer = await file.arrayBuffer();
    loaded = new Uint8Array(buffer);

    if (loaded) {
        // Store in pool
        bufferPool.set(fragPath, loaded, loaded.byteLength);
    }
    return loaded;
}

/**
 * Aggregate column directly in WASM (no row conversion)
 * @param {string} fragPath - Path to fragment file in OPFS
 * @param {number} colIdx - Column index to aggregate
 * @param {string} func - Aggregate function: 'sum', 'min', 'max', 'avg', 'count'
 */
export async function wasmAggregate(fragPath, colIdx, func) {
    const loaded = await loadFragmentCached(fragPath);
    if (!loaded) return null;

    const w = wasm;
    switch (func) {
        case 'sum': return w.opfsSumFloat64Column(colIdx);
        case 'min': return w.opfsMinFloat64Column(colIdx);
        case 'max': return w.opfsMaxFloat64Column(colIdx);
        case 'avg': return w.opfsAvgFloat64Column(colIdx);
        case 'count': return Number(w.opfsCountRows());
        default: return null;
    }
}

// Initialize WASM on load
loadWasm();

// ============================================================================
// Shared State
// ============================================================================

// Global Buffer Pool (Shared Memory Cache)
const bufferPool = new BufferPool();

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

// Cursor storage for lazy data transfer
const cursors = new Map();
let nextCursorId = 1;

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
        const db = new WorkerDatabase(name, bufferPool);
        await db.open();
        databases.set(name, db);
    }
    return databases.get(name);
}

// ============================================================================
// Message Handler
// ============================================================================

/**
 * Send response, using Transferable arrays for columnar data.
 * For small results (< 1000 rows), use direct transfer.
 * For larger results, pack into single buffer for efficiency.
 */
function sendResponse(port, id, result) {
    // WASM binary format - single buffer transfer (fastest)
    if (result && result._format === 'wasm_binary') {
        port.postMessage({
            id,
            result: {
                _format: 'wasm_binary',
                buffer: result.buffer,
                columns: result.columns,
                rowCount: result.rowCount,
                schema: result.schema
            }
        }, [result.buffer]);
        return;
    }

    // Fast path: columnar data
    if (result && result._format === 'columnar' && result.data) {
        const colNames = result.columns;
        const rowCount = result.rowCount;

        // For results < 100k rows, use simple direct transfer (no packing overhead)
        if (rowCount < 100000) {
            const transferables = [];
            const serializedData = {};
            const usedBuffers = new Set();

            for (const name of colNames) {
                const arr = result.data[name];
                if (ArrayBuffer.isView(arr)) {
                    const isView = arr.byteOffset !== 0 || arr.byteLength < arr.buffer.byteLength;
                    const bufferAlreadyUsed = usedBuffers.has(arr.buffer);

                    if (isView || bufferAlreadyUsed) {
                        const copy = new arr.constructor(arr);
                        serializedData[name] = copy;
                        transferables.push(copy.buffer);
                    } else {
                        serializedData[name] = arr;
                        transferables.push(arr.buffer);
                        usedBuffers.add(arr.buffer);
                    }
                } else if (arr && arr._arrowString) {
                    // Handle Arrow String structure (offsets + bytes) - Zero Copy Transfer!
                    serializedData[name] = arr;
                    // Transfer the underlying buffers provided they are not already transferred or views
                    if (arr.offsets && arr.offsets.buffer && !usedBuffers.has(arr.offsets.buffer)) {
                        transferables.push(arr.offsets.buffer);
                        usedBuffers.add(arr.offsets.buffer);
                    }
                    if (arr.bytes && arr.bytes.buffer && !usedBuffers.has(arr.bytes.buffer)) {
                        transferables.push(arr.bytes.buffer);
                        usedBuffers.add(arr.bytes.buffer);
                    }
                } else {
                    serializedData[name] = arr;
                }
            }

            port.postMessage({
                id,
                result: { _format: 'columnar', columns: colNames, rowCount, data: serializedData }
            }, transferables);
            return;
        }

        // For larger results, pack typed arrays into single buffer
        const typedCols = [];
        const stringCols = [];
        let numericBytes = 0;

        for (const name of colNames) {
            const arr = result.data[name];
            if (ArrayBuffer.isView(arr)) {
                typedCols.push({ name, arr });
                numericBytes += arr.byteLength;
            } else if (Array.isArray(arr)) {
                stringCols.push({ name, arr });
            }
        }

        const packedBuffer = numericBytes > 0 ? new ArrayBuffer(numericBytes) : null;
        const colOffsets = {};
        let offset = 0;

        if (packedBuffer) {
            const packedView = new Uint8Array(packedBuffer);
            for (const { name, arr } of typedCols) {
                const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
                packedView.set(bytes, offset);
                colOffsets[name] = { offset, length: arr.length, type: arr.constructor.name };
                offset += arr.byteLength;
            }
        }

        // For string columns, use structured clone - V8's native implementation is highly optimized
        const stringData = {};
        for (const { name, arr } of stringCols) {
            stringData[name] = arr;
        }

        const transferables = [];
        if (packedBuffer) transferables.push(packedBuffer);

        port.postMessage({
            id,
            result: {
                _format: 'packed',
                columns: colNames,
                rowCount: rowCount,
                packedBuffer: packedBuffer,
                colOffsets: colOffsets,
                stringData: stringData
            }
        }, transferables);
        return;
    }

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
            console.log(`[LanceQLWorker] db:open ${args.name}`);
            await getDatabase(args.name);
            result = true;
        } else if (method === 'db:createTable') {
            console.log(`[LanceQLWorker] db:createTable ${args.tableName}`);
            result = await (await getDatabase(args.db)).createTable(args.tableName, args.columns, args.ifNotExists);
            console.log(`[LanceQLWorker] db:createTable ${args.tableName} done`);
        } else if (method === 'db:dropTable') {
            console.log(`[LanceQLWorker] db:dropTable ${args.tableName}`);
            const db = await getDatabase(args.db);
            result = await db.dropTable(args.tableName, args.ifExists);
            // Clear from WASM executor too
            const nameBytes = E.encode(args.tableName);
            getWasmSqlExecutor().clearTable(nameBytes, nameBytes.length);
        } else if (method === 'db:insert') {
            console.log(`[LanceQLWorker] db:insert into ${args.tableName}, rows: ${args.rows?.length}`);
            result = await (await getDatabase(args.db)).insert(args.tableName, args.rows);
            console.log(`[LanceQLWorker] db:insert done`);
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
            const db = await getDatabase(args.db);

            // Try WASM SQL executor first for SELECT queries on large tables
            const tableNames = extractTableNames(args.sql);
            const primaryTable = tableNames[0] ? db.tables.get(tableNames[0]) : null;
            const rowCount = primaryTable ? primaryTable.rowCount : 0;

            console.log(`[LanceQLWorker] Query: "${args.sql}", rowCount: ${rowCount}, useWasm: ${shouldUseWasmSql(args.sql, rowCount)}`);
            if (shouldUseWasmSql(args.sql, rowCount)) {
                try {
                    console.log('[LanceQLWorker] Attempting WASM SQL execution...');
                    result = await executeWasmSqlFull(db, args.sql);
                    console.log('[LanceQLWorker] WASM SQL execution successful');
                } catch (e) {
                    console.warn('[LanceQLWorker] WASM SQL failed, falling back to JS:', e.stack || e.message);
                    result = await executeSQL(db, args.sql);
                }
            } else {
                result = await executeSQL(db, args.sql);
            }

            // For large results (>= 100k rows), use lazy transfer - store data in worker, return handle
            // This avoids blocking the main thread with massive message deserialization
            if (result && result._format === 'columnar' && result.rowCount >= 100000) {
                const cursorId = nextCursorId++;
                cursors.set(cursorId, result);
                result = {
                    _format: 'cursor',
                    cursorId,
                    columns: result.columns,
                    rowCount: result.rowCount
                };
            }
        } else if (method === 'cursor:fetch') {
            // Fetch data from stored cursor
            const cursor = cursors.get(args.cursorId);
            if (!cursor) throw new Error('Cursor not found');
            result = cursor;
            cursors.delete(args.cursorId); // One-time fetch
        } else if (method === 'db:flush') {
            console.log(`[LanceQLWorker] db:flush ${args.db}`);
            await (await getDatabase(args.db)).flush();
            console.log(`[LanceQLWorker] db:flush ${args.db} done`);
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
            const vault = await getVault();

            // Try WASM SQL executor first for SELECT queries on large tables
            const tableNames = extractTableNames(args.sql);
            const primaryTable = tableNames[0] ? vault._db.tables.get(tableNames[0]) : null;
            const rowCount = primaryTable ? primaryTable.rowCount : 0;

            if (shouldUseWasmSql(args.sql, rowCount)) {
                try {
                    result = await executeWasmSqlFull(vault._db, args.sql);
                } catch (e) {
                    console.warn('[LanceQLWorker] Vault WASM SQL failed, falling back to JS:', e.message);
                    result = await vault.exec(args.sql);
                }
            } else {
                result = await vault.exec(args.sql);
            }

            // For large results (>= 100k rows), use lazy transfer
            if (result && result._format === 'columnar' && result.rowCount >= 100000) {
                const cursorId = nextCursorId++;
                cursors.set(cursorId, result);
                result = {
                    _format: 'cursor',
                    cursorId,
                    columns: result.columns,
                    rowCount: result.rowCount
                };
            }
        }
        // Unknown method
        else {
            throw new Error(`Unknown method: ${method}`);
        }

        sendResponse(port, id, result);
    } catch (error) {
        port.postMessage({ id, error: error.stack || error.message });
    }
}

// ============================================================================
// Worker Entry Points
// ============================================================================

// Detect environment
const isSharedWorker = typeof SharedWorkerGlobalScope !== 'undefined' && self instanceof SharedWorkerGlobalScope;

if (isSharedWorker) {
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

        // Worker is ready after WASM is loaded
        loadWasm().then(() => {
            port.postMessage({ type: 'ready' });
        }).catch(err => {
            console.error('[LanceQLWorker] Failed to load WASM:', err);
            port.postMessage({ type: 'ready', error: 'WASM load failed' });
        });
        port.start();

        console.log('[LanceQLWorker] New connection, total ports:', ports.size);
    };
} else {
    // Regular Worker
    self.onmessage = (e) => {
        handleMessage(self, e.data);
    };

    // Send ready for regular worker after WASM is loaded
    loadWasm().then(() => {
        self.postMessage({ type: 'ready' });
    }).catch(err => {
        console.error('[LanceQLWorker] Failed to load WASM:', err);
        self.postMessage({ type: 'ready', error: 'WASM load failed' });
    });
}

console.log('[LanceQLWorker] Initialized');
