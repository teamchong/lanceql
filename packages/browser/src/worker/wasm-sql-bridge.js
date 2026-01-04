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
const HEADER_SIZE = 32;

// Column types
const ColumnType = {
    INT64: 0,
    FLOAT64: 1,
    INT32: 2,
    FLOAT32: 3,
    STRING: 4,
};

/**
 * WASM SQL Executor
 * Manages table registration and SQL execution in WASM
 */
export class WasmSqlExecutor {
    constructor() {
        this._registered = new Map(); // tableName -> { columns: Map }
    }

    /**
     * Register table data in WASM for SQL execution
     * @param {string} tableName
     * @param {Object} columns - Map of column name to typed array
     * @param {number} rowCount
     */
    registerTable(tableName, columns, rowCount) {
        const wasm = getWasm();
        if (!wasm) throw new Error('WASM not loaded');

        const memory = getWasmMemory();

        // Write table name to WASM memory
        const tableNameBytes = new TextEncoder().encode(tableName);
        const tableNamePtr = wasm.wasmAlloc(tableNameBytes.length);
        new Uint8Array(memory.buffer, tableNamePtr, tableNameBytes.length).set(tableNameBytes);

        // Register each column
        for (const [colName, data] of Object.entries(columns)) {
            if (colName.startsWith('__')) continue; // Skip metadata

            const colNameBytes = new TextEncoder().encode(colName);
            const colNamePtr = wasm.wasmAlloc(colNameBytes.length);
            new Uint8Array(memory.buffer, colNamePtr, colNameBytes.length).set(colNameBytes);

            if (data instanceof Float64Array) {
                // Copy data to WASM memory
                const dataPtr = wasm.allocFloat64Buffer(data.length);
                new Float64Array(memory.buffer, dataPtr, data.length).set(data);

                wasm.registerTableFloat64(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, data.length
                );
            } else if (data instanceof BigInt64Array) {
                // Convert to Int64 array
                const dataPtr = wasm.allocInt64Buffer(data.length);
                new BigInt64Array(memory.buffer, dataPtr, data.length).set(data);

                wasm.registerTableInt64(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, data.length
                );
            } else if (data instanceof Int32Array) {
                // For int32, we need to add support in WASM
                // For now, convert to Float64
                const f64Data = new Float64Array(data.length);
                for (let i = 0; i < data.length; i++) f64Data[i] = data[i];

                const dataPtr = wasm.allocFloat64Buffer(f64Data.length);
                new Float64Array(memory.buffer, dataPtr, f64Data.length).set(f64Data);

                wasm.registerTableFloat64(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, f64Data.length
                );
            } else if (Array.isArray(data)) {
                // String column - encode as offsets + data
                const offsets = new Uint32Array(data.length);
                const lengths = new Uint32Array(data.length);

                // Calculate total string length
                let totalLen = 0;
                for (let i = 0; i < data.length; i++) {
                    const str = String(data[i] || '');
                    lengths[i] = str.length;
                    offsets[i] = totalLen;
                    totalLen += str.length;
                }

                // Concatenate all strings
                const stringData = new Uint8Array(totalLen);
                let offset = 0;
                for (let i = 0; i < data.length; i++) {
                    const str = String(data[i] || '');
                    const bytes = new TextEncoder().encode(str);
                    stringData.set(bytes, offset);
                    offset += bytes.length;
                }

                // Copy to WASM
                const offsetsPtr = wasm.wasmAlloc(offsets.byteLength);
                new Uint32Array(memory.buffer, offsetsPtr, offsets.length).set(offsets);

                const lengthsPtr = wasm.wasmAlloc(lengths.byteLength);
                new Uint32Array(memory.buffer, lengthsPtr, lengths.length).set(lengths);

                const dataPtr = wasm.wasmAlloc(stringData.length);
                new Uint8Array(memory.buffer, dataPtr, stringData.length).set(stringData);

                wasm.registerTableString(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    offsetsPtr, lengthsPtr, dataPtr, totalLen, data.length
                );
            }
        }

        this._registered.set(tableName, { columns: Object.keys(columns), rowCount });
    }

    /**
     * Execute SQL and return result as columnar data
     * @param {string} sql - SQL query string
     * @returns {Object} - { columns: string[], rowCount: number, data: Object }
     */
    execute(sql) {
        const wasm = getWasm();
        if (!wasm) throw new Error('WASM not loaded');

        const memory = getWasmMemory();

        // Get SQL input buffer from WASM
        const sqlInputPtr = wasm.getSqlInputBuffer();
        const sqlInputSize = wasm.getSqlInputBufferSize();

        // Write SQL to WASM memory
        const sqlBytes = new TextEncoder().encode(sql);
        if (sqlBytes.length > sqlInputSize) {
            throw new Error(`SQL too long: ${sqlBytes.length} > ${sqlInputSize}`);
        }
        new Uint8Array(memory.buffer, sqlInputPtr, sqlBytes.length).set(sqlBytes);
        wasm.setSqlInputLength(sqlBytes.length);

        // Execute SQL in WASM
        const resultPtr = wasm.executeSql();
        if (resultPtr === 0) {
            throw new Error('WASM SQL execution failed');
        }

        const resultSize = wasm.getResultSize();

        // Parse result buffer
        const result = this._parseResult(memory.buffer, resultPtr, resultSize);

        // Reset WASM state for next query
        wasm.resetResult();

        return result;
    }

    /**
     * Parse WASM result buffer into columnar data
     */
    _parseResult(buffer, ptr, size) {
        const view = new DataView(buffer, ptr, size);

        // Parse header
        const version = view.getUint32(0, true);
        if (version !== RESULT_VERSION) {
            throw new Error(`Unsupported result version: ${version}`);
        }

        const columnCount = view.getUint32(4, true);
        const rowCount = Number(view.getBigUint64(8, true));
        const headerSize = view.getUint32(16, true);
        const metaOffset = view.getUint32(20, true);
        const dataOffset = view.getUint32(24, true);
        const flags = view.getUint32(28, true);

        if (flags !== 0) {
            throw new Error('WASM SQL execution returned error');
        }

        // Parse column metadata
        const columns = [];
        const colData = {};
        let curDataOffset = dataOffset;

        for (let i = 0; i < columnCount; i++) {
            const metaPos = metaOffset + i * 16;
            const colType = view.getUint32(metaPos, true);
            const colNameOffset = view.getUint32(metaPos + 4, true);
            const colNameLen = view.getUint32(metaPos + 8, true);
            const colDataOffset = view.getUint32(metaPos + 12, true);

            // Column name (TODO: implement string table)
            const colName = `col_${i}`;
            columns.push(colName);

            // Parse column data based on type
            const absDataOffset = dataOffset + colDataOffset;

            switch (colType) {
                case ColumnType.INT64: {
                    const arr = new BigInt64Array(buffer, ptr + absDataOffset, rowCount);
                    // Convert to Float64 for JS compatibility
                    const f64Arr = new Float64Array(rowCount);
                    for (let j = 0; j < rowCount; j++) f64Arr[j] = Number(arr[j]);
                    colData[colName] = f64Arr;
                    break;
                }
                case ColumnType.FLOAT64: {
                    colData[colName] = new Float64Array(buffer, ptr + absDataOffset, rowCount).slice();
                    break;
                }
                case ColumnType.INT32: {
                    const arr = new Int32Array(buffer, ptr + absDataOffset, rowCount);
                    const f64Arr = new Float64Array(rowCount);
                    for (let j = 0; j < rowCount; j++) f64Arr[j] = arr[j];
                    colData[colName] = f64Arr;
                    break;
                }
                case ColumnType.FLOAT32: {
                    const arr = new Float32Array(buffer, ptr + absDataOffset, rowCount);
                    const f64Arr = new Float64Array(rowCount);
                    for (let j = 0; j < rowCount; j++) f64Arr[j] = arr[j];
                    colData[colName] = f64Arr;
                    break;
                }
                case ColumnType.STRING: {
                    // String format: [offset0, len0, offset1, len1, ...] + data
                    // Copy raw bytes + offsets for lazy decoding on main thread
                    const offsetsAndLens = new Uint32Array(buffer, ptr + absDataOffset, rowCount * 2);
                    const strDataStart = absDataOffset + rowCount * 8;

                    // Calculate total string bytes
                    let totalBytes = 0;
                    for (let j = 0; j < rowCount; j++) {
                        totalBytes += offsetsAndLens[j * 2 + 1]; // Add length
                    }

                    // Create Arrow-like offsets (cumulative byte positions)
                    const offsets = new Uint32Array(rowCount + 1);
                    const bytes = new Uint8Array(totalBytes);

                    let bytePos = 0;
                    for (let j = 0; j < rowCount; j++) {
                        offsets[j] = bytePos;
                        const strOffset = offsetsAndLens[j * 2];
                        const strLen = offsetsAndLens[j * 2 + 1];
                        const srcBytes = new Uint8Array(buffer, ptr + strDataStart + strOffset, strLen);
                        bytes.set(srcBytes, bytePos);
                        bytePos += strLen;
                    }
                    offsets[rowCount] = totalBytes;

                    // Mark as Arrow string buffer for Transferable transfer
                    colData[colName] = { _arrowString: true, offsets, bytes };
                    break;
                }
            }
        }

        return {
            _format: 'columnar',
            columns,
            rowCount,
            data: colData,
        };
    }

    /**
     * Clear all registered tables
     */
    clear() {
        const wasm = getWasm();
        if (wasm) {
            wasm.clearTables();
        }
        this._registered.clear();
    }

    /**
     * Check if a table is registered
     */
    hasTable(tableName) {
        return this._registered.has(tableName);
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
export function shouldUseWasmSql(sql, rowCount) {
    // Use WASM for larger datasets where the overhead is worth it
    if (rowCount < 1000) return false;

    // Check for simple patterns that WASM handles well
    const sqlUpper = sql.toUpperCase().trim();

    // SELECT * FROM table
    if (/^SELECT\s+\*\s+FROM\s+\w+(\s+WHERE|\s+LIMIT|\s*$)/i.test(sql)) {
        return true;
    }

    // Simple aggregations
    if (/^SELECT\s+(SUM|AVG|MIN|MAX|COUNT)\s*\(/i.test(sql)) {
        return true;
    }

    // Simple WHERE clause
    if (/WHERE\s+\w+\s*[<>=!]+\s*\d+/i.test(sql)) {
        return true;
    }

    return false;
}
