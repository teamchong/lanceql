/**
 * WASM SQL Bridge - Execute SQL entirely in WASM
 *
 * Architecture: "One In, One Out"
 * - Input: SQL string (written to WASM memory)
 * - Output: Packed binary result buffer
 *
 * This eliminates JS object creation and postMessage overhead.
 */

import { getWasm, getWasmMemory } from './index.js';

// Result buffer format constants (must match sql_executor.zig)
const RESULT_VERSION = 1;
const HEADER_SIZE = 36;

// Column types
const ColumnType = {
    INT64: 0,
    FLOAT64: 1,
    INT32: 2,
    FLOAT32: 3,
    STRING: 4,
    LIST: 5,
};

/**
 * WASM SQL Executor
 * Manages table registration and SQL execution in WASM
 */
export class WasmSqlExecutor {
    constructor() {
        this._registered = new Map(); // tableName -> { version: string, columns: Set }
    }

    getLastError() {
        const wasm = getWasm();
        if (!wasm) return "WASM not loaded";

        // Alloc temp buffer for error
        const ptr = wasm.alloc(4096);
        console.log(`[WASM LOG] getLastError alloc ptr: ${ptr}`);
        const len = wasm.getLastError(ptr, 4096);
        console.log(`[WASM LOG] getLastError len: ${len}`);
        if (len === 0) return "Unknown Error";

        // Re-acquire memory after alloc (may have grown)
        const bytes = new Uint8Array(getWasmMemory().buffer, ptr, len);
        const msg = new TextDecoder().decode(bytes);
        console.log(`[WASM LOG] getLastError msg: ${msg}`);
        return msg;
    }

    /**
     * Parse SQL in WASM to identify required table names
     * @param {string} sql
     * @returns {string[]}
     */
    getTableNames(sql) {
        const wasm = getWasm();
        if (!wasm) throw new Error('WASM not loaded');

        // Write SQL to WASM
        const sqlBytes = new TextEncoder().encode(sql);
        const sqlPtr = wasm.alloc(sqlBytes.length);
        // Re-acquire memory after alloc (may have grown)
        new Uint8Array(getWasmMemory().buffer, sqlPtr, sqlBytes.length).set(sqlBytes);

        // Call Zig to get comma-separated table names
        const namesPtr = wasm.getTableNames(sqlPtr, sqlBytes.length);
        if (namesPtr === 0) return [];

        // Read result (null-terminated string)
        const view = new Uint8Array(getWasmMemory().buffer, namesPtr);
        let len = 0;
        while (view[len] !== 0 && len < 1024) len++;

        const namesStr = new TextDecoder().decode(view.subarray(0, len));
        return namesStr ? namesStr.split(',').filter(n => n) : [];
    }

    /**
     * Check if table exists in WASM
     * @param {string} tableName
     * @returns {boolean}
     */
    hasTable(tableName) {
        const wasm = getWasm();
        if (!wasm) return false;

        const bytes = new TextEncoder().encode(tableName);
        const ptr = wasm.alloc(bytes.length);
        // Re-acquire memory after alloc (may have grown)
        new Uint8Array(getWasmMemory().buffer, ptr, bytes.length).set(bytes);

        const res = wasm.hasTable(ptr, bytes.length);
        return res === 1;
    }

    /**
     * Register table data in WASM for SQL execution
     * @param {string} tableName
     * @param {Object} columns - Map of column name to typed array
     * @param {number} rowCount
     * @param {string} version - Table version string to detect changes
     */
    registerTable(tableName, columns, rowCount, version = '') {
        const wasm = getWasm();
        if (!wasm) throw new Error('WASM not loaded');

        // Check if table is already registered with the same version
        const existing = this._registered.get(tableName);
        if (existing && existing.version === version) {
            return;
        }

        // Clear existing table in WASM if it exists
        if (existing) {
            const nameBytes = new TextEncoder().encode(tableName);
            wasm.clearTable(nameBytes, nameBytes.length);
        }

        // Helper to get fresh memory buffer (important: WASM memory may grow after allocs)
        const getMemBuf = () => getWasmMemory().buffer;

        if (!getMemBuf()) {
            throw new Error('WASM memory not available');
        }

        // Write table name to WASM memory
        const tableNameBytes = new TextEncoder().encode(tableName);
        const tableNamePtr = wasm.alloc(tableNameBytes.length);
        if (tableNamePtr < 0 || tableNamePtr >= getMemBuf().byteLength) {
            throw new Error(`Invalid alloc result: ${tableNamePtr} for ${tableNameBytes.length} bytes`);
        }
        new Uint8Array(getMemBuf(), tableNamePtr, tableNameBytes.length).set(tableNameBytes);

        const registeredCols = new Set();

        // Register each column
        for (const [colName, data] of Object.entries(columns)) {
            if (colName.startsWith('__')) continue; // Skip metadata
            // Skip empty arrays (e.g., vector columns we can't handle yet)
            if (Array.isArray(data) && data.length === 0) continue;

            const colNameBytes = new TextEncoder().encode(colName);
            const colNamePtr = wasm.alloc(colNameBytes.length);
            if (colNamePtr < 0 || colNamePtr >= getMemBuf().byteLength) {
                throw new Error(`Invalid colName alloc: ${colNamePtr}`);
            }
            new Uint8Array(getMemBuf(), colNamePtr, colNameBytes.length).set(colNameBytes);

            if (data instanceof Float64Array) {
                // Copy data to WASM memory
                const dataPtr = wasm.allocFloat64Buffer(data.length);
                if (dataPtr === 0 || dataPtr < 0 || dataPtr >= getMemBuf().byteLength) {
                    throw new Error(`Invalid Float64 alloc: ${dataPtr} for ${data.length} elements`);
                }
                new Float64Array(getMemBuf(), dataPtr, data.length).set(data);

                wasm.registerTableFloat64(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, data.length
                );
                registeredCols.add(colName);
            } else if (data instanceof BigInt64Array) {
                // Convert to Int64 array
                const dataPtr = wasm.allocInt64Buffer(data.length);
                new BigInt64Array(getMemBuf(), dataPtr, data.length).set(data);

                wasm.registerTableInt64(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, data.length
                );
                registeredCols.add(colName);
            } else if (data instanceof Int32Array) {
                // For int32, we need to add support in WASM
                // For now, convert to Float64
                const f64Data = new Float64Array(data.length);
                for (let i = 0; i < data.length; i++) f64Data[i] = data[i];

                const dataPtr = wasm.allocFloat64Buffer(f64Data.length);
                new Float64Array(getMemBuf(), dataPtr, f64Data.length).set(f64Data);

                wasm.registerTableFloat64(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, f64Data.length
                );
                registeredCols.add(colName);
            } else if (data instanceof Float32Array) {
                // Vector column (Float32Array with dimension metadata)
                // Check for dimension metadata: __colName_dim
                const dimKey = `__${colName}_dim`;
                const vectorDim = columns[dimKey] || 0;
                const actualRowCount = vectorDim > 0 ? Math.floor(data.length / vectorDim) : data.length;

                // Allocate and copy Float32 data to WASM
                const dataPtr = wasm.allocFloat32Buffer ? wasm.allocFloat32Buffer(data.length) : null;
                if (dataPtr) {
                    new Float32Array(getMemBuf(), dataPtr, data.length).set(data);

                    if (wasm.registerTableFloat32Vector) {
                        wasm.registerTableFloat32Vector(
                            tableNamePtr, tableNameBytes.length,
                            colNamePtr, colNameBytes.length,
                            dataPtr, actualRowCount, vectorDim
                        );
                        console.log(`[Bridge] Registered vector column ${colName}: ${actualRowCount} rows, dim=${vectorDim}`);
                    } else {
                        console.warn(`[Bridge] No registerTableFloat32Vector, skipping ${colName}`);
                    }
                    registeredCols.add(colName);
                } else {
                    console.warn(`[Bridge] Failed to alloc Float32 buffer for ${colName}`);
                }
            } else if (Array.isArray(data)) {
                // String column - encode as offsets + data
                const offsets = new Uint32Array(data.length);
                const lengths = new Uint32Array(data.length);
                const encoder = new TextEncoder();

                // First pass: encode all strings to get byte lengths
                const encodedStrings = [];
                let totalLen = 0;
                for (let i = 0; i < data.length; i++) {
                    const str = String(data[i] || '');
                    const bytes = encoder.encode(str);
                    encodedStrings.push(bytes);
                    lengths[i] = bytes.length; // Use byte length, not char length
                    offsets[i] = totalLen;
                    totalLen += bytes.length;
                }

                // Concatenate all strings
                const stringData = new Uint8Array(totalLen);
                let offset = 0;
                for (let i = 0; i < encodedStrings.length; i++) {
                    stringData.set(encodedStrings[i], offset);
                    offset += encodedStrings[i].length;
                }

                // Copy to WASM
                const offsetsPtr = wasm.alloc(offsets.byteLength);
                if (offsetsPtr < 0 || offsetsPtr >= getMemBuf().byteLength) {
                    throw new Error(`Invalid offsetsPtr alloc: ${offsetsPtr}`);
                }
                new Uint32Array(getMemBuf(), offsetsPtr, offsets.length).set(offsets);

                const lengthsPtr = wasm.alloc(lengths.byteLength);
                if (lengthsPtr < 0 || lengthsPtr >= getMemBuf().byteLength) {
                    throw new Error(`Invalid lengthsPtr alloc: ${lengthsPtr}`);
                }
                new Uint32Array(getMemBuf(), lengthsPtr, lengths.length).set(lengths);

                const dataPtr = wasm.alloc(stringData.length || 1); // Ensure at least 1 byte
                if (dataPtr < 0 || dataPtr >= getMemBuf().byteLength) {
                    throw new Error(`Invalid dataPtr alloc: ${dataPtr}`);
                }
                new Uint8Array(getMemBuf(), dataPtr, stringData.length).set(stringData);

                wasm.registerTableString(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    offsetsPtr, lengthsPtr, dataPtr, totalLen, data.length
                );
                registeredCols.add(colName);
            }
        }

        this._registered.set(tableName, { version, columns: registeredCols, rowCount });
    }

    /**
     * Create an alias for an existing table (reuses table data including shadow columns)
     * @param {string} sourceName - The existing table name
     * @param {string} aliasName - The alias name to create
     */
    aliasTable(sourceName, aliasName) {
        const wasm = getWasm();
        if (!wasm) throw new Error('WASM not loaded');

        const encoder = new TextEncoder();
        const sourceBytes = encoder.encode(sourceName);
        const aliasBytes = encoder.encode(aliasName);

        const sourcePtr = wasm.alloc(sourceBytes.length);
        const aliasPtr = wasm.alloc(aliasBytes.length);

        // Re-acquire memory after allocs (may have grown)
        const memBuf = getWasmMemory().buffer;
        new Uint8Array(memBuf, sourcePtr, sourceBytes.length).set(sourceBytes);
        new Uint8Array(memBuf, aliasPtr, aliasBytes.length).set(aliasBytes);

        if (wasm.aliasTable) {
            wasm.aliasTable(sourcePtr, sourceBytes.length, aliasPtr, aliasBytes.length);
        }

        // Copy registration info
        const existing = this._registered.get(sourceName);
        if (existing) {
            this._registered.set(aliasName, { ...existing, aliasOf: sourceName });
        }
    }

    /**
     * Register table from OPFS file paths
     * @param {string} tableName
     * @param {string[]} filePaths - Array of OPFS paths
     * @param {string} version - Table version string
     */
    registerTableFromFiles(tableName, filePaths, version = '') {
        const wasm = getWasm();
        if (!wasm) throw new Error('WASM not loaded');

        // Check if table is already registered with the same version
        const existing = this._registered.get(tableName);
        if (existing && existing.version === version) {
            return;
        }

        // Clear existing table in WASM if it exists
        if (existing) {
            const nameBytes = new TextEncoder().encode(tableName);
            wasm.clearTable(nameBytes, nameBytes.length);
        }

        const encoder = new TextEncoder();
        const tableNameBytes = encoder.encode(tableName);
        const tableNamePtr = wasm.alloc(tableNameBytes.length);
        new Uint8Array(getWasmMemory().buffer, tableNamePtr, tableNameBytes.length).set(tableNameBytes);

        // Register each fragment
        for (const path of filePaths) {
            const pathBytes = encoder.encode(path);
            const pathPtr = wasm.alloc(pathBytes.length);
            new Uint8Array(getWasmMemory().buffer, pathPtr, pathBytes.length).set(pathBytes);

            const result = wasm.registerTableFromOPFS(
                tableNamePtr, tableNameBytes.length,
                pathPtr, pathBytes.length
            );

            if (result !== 0) {
                console.warn(`Failed to register fragment ${path} for table ${tableName}: error ${result}`);
            }
        }

        this._registered.set(tableName, { version, type: 'files' });
    }

    /**
     * Append in-memory batch to existing registered table (Hybrid Scan)
     * @param {string} tableName
     * @param {Object} columns - Map of column name to typed array
     * @param {number} rowCount
     */
    appendTableMemory(tableName, columns, rowCount) {
        const wasm = getWasm();
        if (!wasm) throw new Error('WASM not loaded');

        // Helper to get fresh memory buffer after allocs
        const getMemBuf = () => getWasmMemory().buffer;

        // Write table name
        const tableNameBytes = new TextEncoder().encode(tableName);
        const tableNamePtr = wasm.alloc(tableNameBytes.length);
        new Uint8Array(getMemBuf(), tableNamePtr, tableNameBytes.length).set(tableNameBytes);

        for (const [colName, data] of Object.entries(columns)) {
            if (colName.startsWith('__')) continue;

            const colNameBytes = new TextEncoder().encode(colName);
            const colNamePtr = wasm.alloc(colNameBytes.length);
            new Uint8Array(getMemBuf(), colNamePtr, colNameBytes.length).set(colNameBytes);

            if (data instanceof Float64Array) {
                const dataPtr = wasm.allocFloat64Buffer(data.length);
                new Float64Array(getMemBuf(), dataPtr, data.length).set(data);

                // type_code: 4 (float64)
                wasm.appendTableMemory(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, 4, rowCount
                );
            } else if (data instanceof BigInt64Array) {
                const dataPtr = wasm.allocInt64Buffer(data.length);
                new BigInt64Array(getMemBuf(), dataPtr, data.length).set(data);

                // type_code: 2 (int64)
                wasm.appendTableMemory(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, 2, rowCount
                );
            } else if (data instanceof Int32Array) {
                const dataPtr = wasm.alloc(data.byteLength);
                new Int32Array(getMemBuf(), dataPtr, data.length).set(data);

                // type_code: 1 (int32)
                wasm.appendTableMemory(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, 1, rowCount
                );
            } else if (data instanceof Float32Array) {
                const dataPtr = wasm.alloc(data.byteLength);
                new Float32Array(getMemBuf(), dataPtr, data.length).set(data);

                // type_code: 3 (float32)
                wasm.appendTableMemory(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, 3, rowCount
                );
            }
        }
    }

    /**
     * Execute SQL and return result as columnar data
     * @param {string} sql - SQL query string
     * @returns {Object} - { columns: string[], rowCount: number, data: Object }
     */
    execute(sql) {
        const wasm = getWasm();
        if (!wasm) throw new Error('WASM not loaded');

        // Set current timestamp for NOW(), CURRENT_DATE() functions
        // Note: WASM i64 accepts BigInt in modern browsers
        if (wasm.setCurrentTimestamp) {
            try {
                wasm.setCurrentTimestamp(BigInt(Date.now()));
            } catch (e) {
                // Fallback for environments that don't support BigInt with WASM
                console.warn('setCurrentTimestamp failed:', e);
            }
        }

        // Get SQL input buffer from WASM
        const sqlInputPtr = wasm.getSqlInputBuffer();
        const sqlInputSize = wasm.getSqlInputBufferSize();

        // Write SQL to WASM memory
        const sqlBytes = new TextEncoder().encode(sql);
        if (sqlBytes.length > sqlInputSize) {
            throw new Error(`SQL too long: ${sqlBytes.length} > ${sqlInputSize}`);
        }
        new Uint8Array(getWasmMemory().buffer, sqlInputPtr, sqlBytes.length).set(sqlBytes);
        wasm.setSqlInputLength(sqlBytes.length);

        // Execute SQL in WASM
        const resultPtr = wasm.executeSql();
        if (resultPtr === 0) {
            const errMsg = this.getLastError();
            console.log(`[WASM LOG] execute throwing: "${errMsg}"`);
            throw new Error(errMsg);
        }

        const resultSize = wasm.getResultSize();
        const debugMsg = this.getLastError();
        if (debugMsg && debugMsg.length > 0) {
            console.log(`[WASM DEBUG CAPTURED] ${debugMsg}`);
        }
        if (debugMsg.length > 0 && debugMsg.startsWith("DEBUG:")) {
            // Throw to clear visible logs
            // throw new Error(`[WASM DEBUG] ${debugMsg}`);
        }

        const result = this._parseResult(getWasmMemory().buffer, resultPtr, resultSize);

        // Reset WASM state for next query
        wasm.resetResult();

        return result;
    }

    /**
     * Parse WASM result buffer (Lance File/Fragment format)
     */
    _parseResult(buffer, ptr, size) {
        const view = new DataView(buffer, ptr, size);
        const decoder = new TextDecoder();

        // Check for Lance Footer first (Standard Format)
        // Footer is at the end (40 bytes)
        if (size >= 40) {
            const footerOffset = size - 40;
            const magicVals = [
                view.getUint8(footerOffset + 36),
                view.getUint8(footerOffset + 37),
                view.getUint8(footerOffset + 38),
                view.getUint8(footerOffset + 39)
            ];
            const magic = String.fromCharCode(...magicVals);
            if (magic === 'LANC') {
                // It's a Lance file
                return this._parseLanceResult(buffer, ptr, size, footerOffset, view, decoder);
            }
        }

        // Check for legacy format (RESULT_VERSION = 1 at the very beginning)
        if (size >= 36 && view.getUint32(0, true) === RESULT_VERSION) {
            return this._parseLegacyResult(buffer, ptr, size);
        }

        throw new Error(`Invalid result format (Size: ${size}). Not a Lance file.`);
    }

    _parseLanceResult(buffer, ptr, size, footerOffset, view, decoder) {
        // Read Footer

        // const colMetaStart = Number(view.getBigUint64(footerOffset, true));
        const colMetaOffsetsStart = Number(view.getBigUint64(footerOffset + 8, true));
        // const globalBuffOffsetsStart = Number(view.getBigUint64(footerOffset + 16, true));
        // const numGlobalBuffers = view.getUint32(footerOffset + 24, true);
        const numCols = view.getUint32(footerOffset + 28, true);

        const columns = [];
        const colData = {};
        let resultRowCount = 0;

        // Read Column Metadata Offsets
        // Note: These might not be 8-byte aligned in the file relative to global ptr
        // So we use DataView to read them safely.

        for (let i = 0; i < numCols; i++) {
            const offsetPos = colMetaOffsetsStart + i * 8;
            const metaPos = Number(view.getBigUint64(offsetPos, true));

            // Parse Column Metadata (Protobuf)
            let localOffset = metaPos;

            // 1. Name
            // tag 10 (field 1, wire 2)
            view.getUint8(localOffset++); // skip tag
            const [nameLen, lenBytes] = this._readVarint(view, localOffset);
            localOffset += lenBytes;
            const nameBytes = new Uint8Array(buffer, ptr + localOffset, nameLen);
            const colName = decoder.decode(nameBytes);
            localOffset += nameLen;
            columns.push(colName);

            // 2. Type
            // tag 18 (field 2, wire 2)
            view.getUint8(localOffset++); // skip tag
            const [typeLen, typeLenBytes] = this._readVarint(view, localOffset);
            localOffset += typeLenBytes;
            const typeBytes = new Uint8Array(buffer, ptr + localOffset, typeLen);
            const typeStr = decoder.decode(typeBytes);
            localOffset += typeLen;

            // 3. Nullable
            // tag 24 (field 3, wire 0)
            view.getUint8(localOffset++);
            const [nullable, nullBytes] = this._readVarint(view, localOffset);
            localOffset += nullBytes;

            // 4. Data Offset
            // tag 33 (field 4, wire 1 - fixed64)
            view.getUint8(localOffset++);
            const dataOffset = Number(view.getBigUint64(localOffset, true));
            localOffset += 8;

            // 5. Row Count
            // tag 40 (field 5, wire 0)
            view.getUint8(localOffset++);
            const [rowCount, rowBytes] = this._readVarint(view, localOffset);
            localOffset += rowBytes;
            resultRowCount = rowCount; // consistent across columns

            // 6. Data Size
            // tag 48 (field 6, wire 0) -- wait, zig code says 48
            view.getUint8(localOffset++);
            const [dataSize, sizeBytes] = this._readVarint(view, localOffset);
            localOffset += sizeBytes;

            // Read Data
            const absDataOffset = ptr + dataOffset;

            // Map types
            if (typeStr === 'float64' || typeStr === 'int64' || typeStr === 'int32' || typeStr === 'float32') {
                if (typeStr === 'float64') {
                    // Zero-ish copy: slice the buffer
                    const arr = new Float64Array(buffer, absDataOffset, rowCount).slice();
                    // Check for NaN implies NULL (SQL semantics for this engine)
                    // We only convert if NaN exists to save perf
                    let hasNan = false;
                    for (let k = 0; k < rowCount; k++) {
                        if (Number.isNaN(arr[k])) { hasNan = true; break; }
                    }
                    if (hasNan) {
                        console.log(`[WASM LOG] Column ${colName} has NaNs, converting to nulls`);
                        const nullArr = new Array(rowCount);
                        for (let k = 0; k < rowCount; k++) {
                            const v = arr[k];
                            nullArr[k] = Number.isNaN(v) ? null : v;
                        }
                        colData[colName] = nullArr;
                    } else {
                        colData[colName] = arr;
                    }
                } else if (typeStr === 'int64') {
                    // Conversion needed or return BigInt64Array?
                    const src = new BigInt64Array(buffer, absDataOffset, rowCount);
                    const dst = new Array(rowCount);
                    const NULL_SENTINEL_INT = -9223372036854775808n;
                    for (let j = 0; j < rowCount; j++) {
                        const v = src[j];
                        dst[j] = (v === NULL_SENTINEL_INT) ? null : Number(v);
                    }
                    colData[colName] = dst;
                } else if (typeStr === 'int32') {
                    const src = new Int32Array(buffer, absDataOffset, rowCount);
                    const dst = new Array(rowCount);
                    // For int32, we currently use the same i64 sentinel if we were appending from JS,
                    // but if it's from file it might be different. 
                    // However, getIntValueOptimized returns i64, so it might be promoted.
                    // Let's assume for now.
                    for (let j = 0; j < rowCount; j++) {
                        const v = src[j];
                        // check for both int32 min and int64 min (cast to int32)
                        if (v === -2147483648) dst[j] = null;
                        else dst[j] = v;
                    }
                    colData[colName] = dst;
                } else { // float32
                    const src = new Float32Array(buffer, absDataOffset, rowCount);
                    const dst = new Array(rowCount);
                    for (let j = 0; j < rowCount; j++) {
                        const v = src[j];
                        dst[j] = isNaN(v) ? null : v;
                    }
                    colData[colName] = dst;
                }
            } else if (typeStr === 'string' || typeStr === 'list') {
                const isList = typeStr === 'list';
                // Separating Data and Offsets
                // Data Size includes both data bytes and offsets array
                const offsetsLen = (rowCount + 1) * 4;
                const dataBytesLen = dataSize - offsetsLen;

                // Copy bytes
                const bytes = new Uint8Array(buffer, absDataOffset, dataBytesLen).slice();
                // Copy offsets
                // Offsets start after data bytes
                const offsets = new Uint32Array(buffer, absDataOffset + dataBytesLen, rowCount + 1).slice();

                // If column is nullable, mark empty strings as null by setting nullable flag
                colData[colName] = { _arrowString: true, offsets, bytes, isList: typeStr === 'list', nullable: nullable === 1 };
            }
        }

        return {
            _format: 'columnar',
            columns,
            rowCount: resultRowCount,
            data: colData,
        };
    }

    _readVarint(view, offset) {
        let result = 0;
        let shift = 0;
        let bytesRead = 0;
        while (true) {
            const byte = view.getUint8(offset + bytesRead);
            bytesRead++;
            result |= (byte & 0x7F) << shift;
            if ((byte & 0x80) === 0) break;
            shift += 7;
        }
        return [result, bytesRead];
    }

    /**
     * Clear all registered tables
     */
    /**
     * Parse legacy result format (used by window functions)
     */
    _parseLegacyResult(buffer, ptr, size) {
        const view = new DataView(buffer, ptr, size);
        const decoder = new TextDecoder();

        // Header (36 bytes)
        // 0: RESULT_VERSION (4)
        // 4: total_cols (4)
        // 8: row_count (8)
        // 16: extended_header (4)
        // 20: extended_header (4)
        // 24: data_offset_start (4)
        // 28: 0 (4)
        // 32: names_offset (4)

        const numCols = view.getUint32(4, true);
        const rowCount = Number(view.getBigUint64(8, true));
        const dataOffsetStart = view.getUint32(24, true);
        const namesOffset = view.getUint32(32, true);

        const columns = [];
        const colData = {};

        for (let i = 0; i < numCols; i++) {
            const metaPos = 36 + i * 16;
            const typeEnum = view.getUint32(metaPos, true);
            const nameOff = view.getUint32(metaPos + 4, true);
            const nameLen = view.getUint32(metaPos + 8, true);
            const dataOff = view.getUint32(metaPos + 12, true);

            // Read Name
            const nameBytes = new Uint8Array(buffer, ptr + namesOffset + nameOff, nameLen);
            const colName = decoder.decode(nameBytes);
            columns.push(colName);

            // Read Data
            const typeStr = Object.keys(ColumnType).find(key => ColumnType[key] === typeEnum).toLowerCase();
            const absDataOffset = ptr + dataOffsetStart + dataOff;

            if (typeStr === 'float64' || typeStr === 'int64' || typeStr === 'int32' || typeStr === 'float32') {
                if (typeStr === 'float64') {
                    colData[colName] = new Float64Array(buffer, absDataOffset, rowCount).slice();
                } else if (typeStr === 'int64') {
                    // Legacy format often stores int64 as float64 for JS
                    colData[colName] = new Float64Array(buffer, absDataOffset, rowCount).slice();
                } else if (typeStr === 'int32') {
                    colData[colName] = new Int32Array(buffer, absDataOffset, rowCount).slice();
                } else if (typeStr === 'float32') {
                    colData[colName] = new Float32Array(buffer, absDataOffset, rowCount).slice();
                }
            } else if (typeStr === 'string' || typeStr === 'list') {
                // Legacy string/list format: offsets(u32 * rows) + len(u32 * rows) + data
                // Wait! Let's check Zig again for legacy string format.
                // Zig 4668 reg columns: 
                // for ri: writeU32(str_offset), writeU32(len)
                // for ri: writeToResult(data)

                const offsets = new Uint32Array(rowCount + 1);
                let currentOffset = 0;
                offsets[0] = 0;

                // Pass 1: results are stored as [off0, len0, off1, len1, ...]
                const metaDataOffset = absDataOffset;
                let totalBytes = 0;
                for (let j = 0; j < rowCount; j++) {
                    const len = view.getUint32(metaDataOffset + j * 8 + 4, true);
                    totalBytes += len;
                    offsets[j + 1] = totalBytes;
                }

                const dataBytesOffset = absDataOffset + rowCount * 8;
                const bytes = new Uint8Array(buffer, dataBytesOffset, totalBytes).slice();

                colData[colName] = { _arrowString: true, offsets, bytes, isList: typeStr === 'list' };
            }
        }

        return {
            _format: 'columnar',
            columns,
            rowCount,
            data: colData,
        };
    }

    clear() {
        const wasm = getWasm();
        if (wasm) {
            wasm.clearTables();
        }
        this._registered.clear();
    }

}

// Singleton instance
let instance = null;

export function getWasmSqlExecutor() {
    if (!instance) {
        instance = new WasmSqlExecutor();
    }
    return instance;
}

/**
 * Check if WASM SQL execution is available and beneficial
 * @param {string} sql - SQL query
 * @param {number} rowCount - Estimated row count
 * @returns {boolean}
 */
