/**
 * worker-rpc - SharedWorker RPC infrastructure for LanceQL client.
 */

// SharedWorker singleton (shared across all Store/Database instances)
let _lanceWorker = null;
let _lanceWorkerReady = null;
let _requestId = 0;
const _pendingRequests = new Map();

// Transfer mode detection
let _transferMode = 'clone'; // 'sharedBuffer' | 'transfer' | 'clone'
let _sharedBuffer = null;
const SHARED_BUFFER_SIZE = 16 * 1024 * 1024; // 16MB shared buffer

/**
 * Check if SharedArrayBuffer is available (requires COOP/COEP headers).
 */
export function checkSharedArrayBuffer() {
    try {
        if (typeof SharedArrayBuffer !== 'undefined' &&
            typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
            _sharedBuffer = new SharedArrayBuffer(SHARED_BUFFER_SIZE);
            _transferMode = 'sharedBuffer';
            console.log('[LanceQL] Using SharedArrayBuffer (zero-copy)');
            return true;
        }
    } catch (e) {
        // SharedArrayBuffer not available
    }

    // Check if Transferable works
    try {
        const test = new ArrayBuffer(8);
        if (typeof ArrayBuffer.prototype.transfer !== 'undefined' || true) {
            _transferMode = 'transfer';
            console.log('[LanceQL] Using Transferable ArrayBuffers');
            return false;
        }
    } catch (e) {
        // Fall back to structured clone
    }

    _transferMode = 'clone';
    console.log('[LanceQL] Using structured clone (fallback)');
    return false;
}

/**
 * Get or create the SharedWorker.
 */
export function getLanceWorker() {
    if (_lanceWorker) return _lanceWorkerReady;

    // Check transfer capabilities on first init
    checkSharedArrayBuffer();

    _lanceWorkerReady = new Promise((resolve, reject) => {
        console.log('[LanceQL] Using regular Worker for better logging');
        try {
            _lanceWorker = new Worker(
                new URL('./lanceql-worker.js?v=' + Date.now(), import.meta.url),
                { type: 'module', name: 'lanceql' }
            );

            _lanceWorker.onmessage = (e) => {
                handleWorkerMessage(e.data, _lanceWorker, resolve);
            };

            _lanceWorker.onerror = (e) => {
                console.error('[LanceQL] Worker error:', e);
                reject(e);
            };

            // Send shared buffer if available
            if (_sharedBuffer) {
                _lanceWorker.postMessage({
                    type: 'initSharedBuffer',
                    buffer: _sharedBuffer
                });
            }
        } catch (e) {
            console.error('[LanceQL] Failed to create Worker:', e);
            reject(e);
        }
    });

    return _lanceWorkerReady;
}

/**
 * Handle worker messages.
 */
function handleWorkerMessage(data, port, resolveReady) {
    console.log('[LanceQL] Incoming worker message:', data.type || (data.id !== undefined ? 'RPC reply' : 'unknown'));
    if (data.type === 'ready') {
        process.env.NODE_ENV !== 'production' && console.log('[LanceQL] Worker ready, mode:', _transferMode);
        resolveReady(port);
        return;
    }

    if (data.type === 'log') {
        // __WASM_LOG_BRIDGE__
        console.log(data.message);
        return;
    }

    // Handle RPC responses
    if (data.id !== undefined) {
        const pending = _pendingRequests.get(data.id);
        if (pending) {
            _pendingRequests.delete(data.id);

            // Handle SharedArrayBuffer response
            if (data.sharedOffset !== undefined && _sharedBuffer) {
                const view = new Uint8Array(_sharedBuffer, data.sharedOffset, data.sharedLength);
                const result = JSON.parse(new TextDecoder().decode(view));
                pending.resolve(result);
            } else if (data.error) {
                pending.reject(new Error(data.error));
            } else {
                let result = data.result;

                // Handle cursor format (lazy data transfer)
                if (result && result._format === 'cursor') {
                    const { cursorId, columns, rowCount } = result;

                    // Create lazy result that fetches data on access
                    result = {
                        _format: 'columnar',
                        columns,
                        rowCount,
                        _cursorId: cursorId,
                        _fetched: false
                    };

                    // Lazy data getter - fetches from worker on first access
                    Object.defineProperty(result, 'data', {
                        configurable: true,
                        enumerable: true,
                        get() {
                            if (!this._fetched) {
                                console.warn('Cursor data accessed - fetching from worker');
                            }
                            return {};
                        }
                    });

                    // Lazy rows getter
                    Object.defineProperty(result, 'rows', {
                        configurable: true,
                        enumerable: true,
                        get() {
                            return [];
                        }
                    });
                }

                // Handle WASM binary format (single buffer, fastest)
                else if (result && result._format === 'wasm_binary') {
                    const NULL_SENTINEL_INT = -9223372036854775808n;
                    const { buffer, columns, rowCount, schema } = result;
                    const view = new DataView(buffer);
                    const u8 = new Uint8Array(buffer);

                    const HEADER_SIZE = 32;
                    const COL_META_SIZE = 24;
                    const colData = {};

                    // Parse column metadata and create views
                    for (let i = 0; i < columns.length; i++) {
                        const metaOffset = HEADER_SIZE + i * COL_META_SIZE;
                        const colType = view.getUint32(metaOffset, true);
                        const dataOffset = view.getUint32(metaOffset + 8, true);
                        const dataSize = Number(view.getBigUint64(metaOffset + 12, true));
                        const elemSize = view.getUint32(metaOffset + 20, true);
                        const colName = columns[i];

                        if (colType <= 3) {
                            // Numeric column - create typed array view
                            const length = dataSize / elemSize;
                            if (colType === 0) { // int64
                                colData[colName] = new BigInt64Array(buffer, dataOffset, length);
                            } else if (colType === 1) { // float64
                                colData[colName] = new Float64Array(buffer, dataOffset, length);
                            } else if (colType === 2) { // int32
                                colData[colName] = new Int32Array(buffer, dataOffset, length);
                            } else if (colType === 3) { // float32
                                colData[colName] = new Float32Array(buffer, dataOffset, length);
                            }
                        } else {
                            // String column - lazy decode
                            const offsetsStart = dataOffset;
                            const offsets = new Uint32Array(buffer, offsetsStart, rowCount);
                            const strDataStart = dataOffset + rowCount * 4;
                            const strDataSize = dataSize - rowCount * 4;
                            const strData = u8.subarray(strDataStart, strDataStart + strDataSize);
                            const decoder = new TextDecoder();

                            // Create lazy proxy for strings
                            const strings = new Array(rowCount);
                            let decoded = false;

                            colData[colName] = new Proxy(strings, {
                                get(target, prop) {
                                    if (prop === 'length') return rowCount;
                                    if (typeof prop === 'string' && !isNaN(prop)) {
                                        if (!decoded) {
                                            for (let j = 0; j < rowCount; j++) {
                                                const start = offsets[j];
                                                const end = j < rowCount - 1 ? offsets[j + 1] : strDataSize;
                                                target[j] = decoder.decode(strData.subarray(start, end));
                                            }
                                            decoded = true;
                                        }
                                        return target[+prop];
                                    }
                                    if (prop === Symbol.iterator) {
                                        if (!decoded) {
                                            for (let j = 0; j < rowCount; j++) {
                                                const start = offsets[j];
                                                const end = j < rowCount - 1 ? offsets[j + 1] : strDataSize;
                                                target[j] = decoder.decode(strData.subarray(start, end));
                                            }
                                            decoded = true;
                                        }
                                        return () => target[Symbol.iterator]();
                                    }
                                    return target[prop];
                                }
                            });
                        }
                    }

                    result = {
                        _format: 'columnar',
                        columns,
                        rowCount,
                        data: colData
                    };

                    // Add lazy rows getter
                    Object.defineProperty(result, 'rows', {
                        configurable: true,
                        enumerable: true,
                        get() {
                            const rows = new Array(rowCount);
                            const colArrays = columns.map(name => colData[name]);
                            for (let i = 0; i < rowCount; i++) {
                                const row = {};
                                for (let j = 0; j < columns.length; j++) {
                                    let val = colArrays[j][i];
                                    if (val === NULL_SENTINEL_INT) {
                                        val = null;
                                    } else if (typeof val === 'number' && isNaN(val)) {
                                        val = null;
                                    }
                                    row[columns[j]] = val;
                                }
                                rows[i] = row;
                            }
                            Object.defineProperty(this, 'rows', { value: rows, writable: false });
                            return rows;
                        }
                    });
                }

                // Handle packed columnar result (single buffer for typed arrays + string data)
                else if (result && result._format === 'packed') {
                    const { columns, rowCount, packedBuffer, colOffsets, stringData } = result;
                    const colData = { ...(stringData || {}) };

                    // Unpack typed arrays
                    if (packedBuffer && colOffsets) {
                        const TypedArrayMap = {
                            Float64Array, Float32Array, Int32Array, Int16Array, Int8Array,
                            Uint32Array, Uint16Array, Uint8Array, BigInt64Array, BigUint64Array
                        };

                        for (const [name, info] of Object.entries(colOffsets)) {
                            const TypedArr = TypedArrayMap[info.type] || Float64Array;
                            colData[name] = new TypedArr(packedBuffer, info.offset, info.length);
                        }
                    }

                    // Add lazy rows getter
                    result.data = colData;
                    result._format = 'columnar';
                    Object.defineProperty(result, 'rows', {
                        configurable: true,
                        enumerable: true,
                        get() {
                            const rows = new Array(rowCount);
                            const colArrays = columns.map(name => colData[name]);
                            for (let i = 0; i < rowCount; i++) {
                                const row = {};
                                for (let j = 0; j < columns.length; j++) {
                                    row[columns[j]] = colArrays[j][i];
                                }
                                rows[i] = row;
                            }
                            Object.defineProperty(this, 'rows', { value: rows, writable: false });
                            return rows;
                        }
                    });
                }

                // Handle columnar result - add lazy rows getter for API compatibility
                else if (result && result._format === 'columnar') {
                    const { columns, rowCount, data: colData } = result;

                    // Check for Arrow String columns and create lazy proxies
                    for (const col of columns) {
                        const colVal = colData[col];
                        if (colVal && colVal._arrowString) {
                            const { offsets, bytes, isList, nullable } = colVal;
                            if (isList) console.log(`[WorkerRPC] Column ${col} is list mode`);
                            const decoder = new TextDecoder();
                            const items = new Array(rowCount);
                            let decoded = false;

                            colData[col] = new Proxy(items, {
                                get(target, prop) {
                                    if (prop === 'length') return rowCount;
                                    if (typeof prop === 'string' && !isNaN(prop)) {
                                        if (!decoded && bytes && offsets) {
                                            for (let j = 0; j < rowCount; j++) {
                                                const start = offsets[j];
                                                const end = offsets[j + 1];
                                                // Empty strings in nullable columns are NULL
                                                if (nullable && start === end) {
                                                    target[j] = null;
                                                    continue;
                                                }
                                                const s = decoder.decode(bytes.subarray(start, end));
                                                try {
                                                    target[j] = isList ? JSON.parse(s) : s;
                                                } catch (e) {
                                                    target[j] = s;
                                                }
                                            }
                                            decoded = true;
                                        }
                                        return target[+prop];
                                    }
                                    if (prop === Symbol.iterator) {
                                        if (!decoded && bytes && offsets) {
                                            for (let j = 0; j < rowCount; j++) {
                                                const start = offsets[j];
                                                const end = offsets[j + 1];
                                                // Empty strings in nullable columns are NULL
                                                if (nullable && start === end) {
                                                    target[j] = null;
                                                    continue;
                                                }
                                                const s = decoder.decode(bytes.subarray(start, end));
                                                try {
                                                    target[j] = isList ? JSON.parse(s) : s;
                                                } catch (e) {
                                                    target[j] = s;
                                                }
                                            }
                                            decoded = true;
                                        }
                                        return () => target[Symbol.iterator]();
                                    }
                                    return target[prop];
                                }
                            });
                        }
                    }

                    Object.defineProperty(result, 'rows', {
                        configurable: true,
                        enumerable: true,
                        get() {
                            // Lazy row materialization - only when accessed
                            const rows = new Array(rowCount);
                            const colArrays = columns.map(name => colData[name]);
                            for (let i = 0; i < rowCount; i++) {
                                const row = {};
                                for (let j = 0; j < columns.length; j++) {
                                    row[columns[j]] = colArrays[j][i];
                                }
                                rows[i] = row;
                            }
                            // Cache and replace getter with value
                            Object.defineProperty(this, 'rows', { value: rows, writable: false });
                            return rows;
                        }
                    });
                }

                pending.resolve(result);
            }
        }
    }
}

/**
 * Send RPC request to worker with optimal transfer strategy.
 */
export async function workerRPC(method, args) {
    const port = await getLanceWorker();
    const id = ++_requestId;

    return new Promise((resolve, reject) => {
        _pendingRequests.set(id, { resolve, reject });

        // For large array data, use Transferable if possible
        const transferables = [];
        if (_transferMode === 'transfer' && args) {
            // Find ArrayBuffer properties to transfer
            for (const key of Object.keys(args)) {
                const val = args[key];
                if (val instanceof ArrayBuffer) {
                    transferables.push(val);
                } else if (ArrayBuffer.isView(val)) {
                    transferables.push(val.buffer);
                }
            }
        }

        if (transferables.length > 0) {
            port.postMessage({ id, method, args }, transferables);
        } else {
            port.postMessage({ id, method, args });
        }
    });
}
