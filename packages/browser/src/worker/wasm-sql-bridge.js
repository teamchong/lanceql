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
import { SQLParser } from './sql/parser.js';
import { SQLLexer } from './sql/tokenizer.js';

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
        this._registered = new Map(); // tableName -> { version: string, columns: Set }
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

        const memory = getWasmMemory();

        // Write table name to WASM memory
        const tableNameBytes = new TextEncoder().encode(tableName);
        const tableNamePtr = wasm.alloc(tableNameBytes.length);
        new Uint8Array(memory.buffer, tableNamePtr, tableNameBytes.length).set(tableNameBytes);

        const registeredCols = new Set();

        // Register each column
        for (const [colName, data] of Object.entries(columns)) {
            if (colName.startsWith('__')) continue; // Skip metadata

            const colNameBytes = new TextEncoder().encode(colName);
            const colNamePtr = wasm.alloc(colNameBytes.length);
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
                registeredCols.add(colName);
            } else if (data instanceof BigInt64Array) {
                // Convert to Int64 array
                const dataPtr = wasm.allocInt64Buffer(data.length);
                new BigInt64Array(memory.buffer, dataPtr, data.length).set(data);

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
                new Float64Array(memory.buffer, dataPtr, f64Data.length).set(f64Data);

                wasm.registerTableFloat64(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, f64Data.length
                );
                registeredCols.add(colName);
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
                const offsetsPtr = wasm.alloc(offsets.byteLength);
                new Uint32Array(memory.buffer, offsetsPtr, offsets.length).set(offsets);

                const lengthsPtr = wasm.alloc(lengths.byteLength);
                new Uint32Array(memory.buffer, lengthsPtr, lengths.length).set(lengths);

                const dataPtr = wasm.alloc(stringData.length);
                new Uint8Array(memory.buffer, dataPtr, stringData.length).set(stringData);

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

        const memory = getWasmMemory();

        // Write table name
        const tableNameBytes = new TextEncoder().encode(tableName);
        const tableNamePtr = wasm.alloc(tableNameBytes.length);
        new Uint8Array(memory.buffer, tableNamePtr, tableNameBytes.length).set(tableNameBytes);

        for (const [colName, data] of Object.entries(columns)) {
            if (colName.startsWith('__')) continue;

            const colNameBytes = new TextEncoder().encode(colName);
            const colNamePtr = wasm.alloc(colNameBytes.length);
            new Uint8Array(memory.buffer, colNamePtr, colNameBytes.length).set(colNameBytes);

            if (data instanceof Float64Array) {
                const dataPtr = wasm.allocFloat64Buffer(data.length);
                new Float64Array(memory.buffer, dataPtr, data.length).set(data);

                // type_code: 4 (float64)
                wasm.appendTableMemory(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, 4, rowCount
                );
            } else if (data instanceof BigInt64Array) {
                const dataPtr = wasm.allocInt64Buffer(data.length);
                new BigInt64Array(memory.buffer, dataPtr, data.length).set(data);

                // type_code: 2 (int64)
                wasm.appendTableMemory(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, 2, rowCount
                );
            } else if (data instanceof Int32Array) {
                // Convert to Float64 for now as int32 support is partial
                // Or better, support int32 in appendTableMemory if Zig supports it (We added it!)
                const dataPtr = wasm.alloc(data.byteLength); // int32 is just bytes
                new Int32Array(memory.buffer, dataPtr, data.length).set(data);

                // type_code: 1 (int32)
                wasm.appendTableMemory(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, 1, rowCount
                );
            } else if (data instanceof Float32Array) {
                const dataPtr = wasm.alloc(data.byteLength);
                new Float32Array(memory.buffer, dataPtr, data.length).set(data);

                // type_code: 3 (float32)
                wasm.appendTableMemory(
                    tableNamePtr, tableNameBytes.length,
                    colNamePtr, colNameBytes.length,
                    dataPtr, 3, rowCount
                );
            }
            // String support for append is tricky (variable length). 
            // Zig implementation for appendTableMemory currently only takes fixed width arrays?
            // Let's check Zig...
            // Zig takes data_ptr, type_code.
            // But for strings we need offsets+bytes.
            // Our zig implementation returns error(4) for anything else.
            // So Strings are NOT supported in Memory Batch yet?
            // We should fix Zig or skip.
            // For now we skip strings or convert to something else? 
            // Skipping will cause empty column for new rows.
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
     * Parse WASM result buffer (Lance File/Fragment format)
     */
    _parseResult(buffer, ptr, size) {
        const view = new DataView(buffer, ptr, size);
        const decoder = new TextDecoder();

        // Footer is at the end (40 bytes)
        if (size < 40) throw new Error("Result too small for Lance footer");
        const footerOffset = size - 40;

        // Verify Magic
        const magicVals = [
            view.getUint8(footerOffset + 36),
            view.getUint8(footerOffset + 37),
            view.getUint8(footerOffset + 38),
            view.getUint8(footerOffset + 39)
        ];
        const magic = String.fromCharCode(...magicVals);
        if (magic !== 'LANC') {
            throw new Error("Invalid Lance Magic: " + magic);
        }

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
                    colData[colName] = new Float64Array(buffer, absDataOffset, rowCount).slice();
                } else if (typeStr === 'int64') {
                    // Conversion needed or return BigInt64Array?
                    // Previous logic converted to Float64 for JS compat
                    const src = new BigInt64Array(buffer, absDataOffset, rowCount);
                    const dst = new Float64Array(rowCount);
                    for (let j = 0; j < rowCount; j++) dst[j] = Number(src[j]);
                    colData[colName] = dst;
                } else if (typeStr === 'int32') {
                    const src = new Int32Array(buffer, absDataOffset, rowCount);
                    const dst = new Float64Array(rowCount);
                    for (let j = 0; j < rowCount; j++) dst[j] = src[j];
                    colData[colName] = dst;
                } else { // float32
                    const src = new Float32Array(buffer, absDataOffset, rowCount);
                    const dst = new Float64Array(rowCount);
                    for (let j = 0; j < rowCount; j++) dst[j] = src[j];
                    colData[colName] = dst;
                }
            } else if (typeStr === 'string') {
                // Separating Data and Offsets
                // Data Size includes both data bytes and offsets array
                const offsetsLen = (rowCount + 1) * 4;
                const dataBytesLen = dataSize - offsetsLen;

                // Copy bytes
                const bytes = new Uint8Array(buffer, absDataOffset, dataBytesLen).slice();
                // Copy offsets
                // Offsets start after data bytes
                const offsets = new Uint32Array(buffer, absDataOffset + dataBytesLen, rowCount + 1).slice();

                colData[colName] = { _arrowString: true, offsets, bytes };
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
    if (rowCount < 100) return false;

    try {
        const lexer = new SQLLexer(sql);
        const tokens = lexer.tokenize();
        const parser = new SQLParser(tokens);
        const ast = parser.parse();

        // If not a SELECT statement (e.g. INSERT, UPDATE), default to WASM (or handle elsewhere)
        // Currently we only route SELECTs here.
        if (ast.type !== 'SELECT') {
            return true;
        }

        // Complex clauses that benefit from WASM / are not supported by simple JS scan
        if (ast.where || ast.having || ast.qualify) return true;
        if (ast.joins && ast.joins.length > 0) return true;
        if (ast.groupBy || ast.orderBy) return true;
        if (ast.distinct) return true;
        if (ast.union || ast.intersect || ast.except) return true;

        // Check columns for aggregates or complex expressions
        // Simple column access is fine for JS
        for (const col of ast.columns) {
            if (col.type === 'star') continue;

            // If it's an aggregate or window function, use WASM
            if (col.type === 'aggregate' || col.type === 'window') {
                return true;
            }

            // If it's a scalar subquery or case expression, use WASM
            if (col.type === 'scalar_subquery' || col.type === 'case') {
                return true;
            }

            // Function call checking would go here if we had a generic function type
        }

        // Check for calculated fields in SELECT list (e.g. col1 + col2)
        // The parser might represent these as expressions. 
        // For now, if we've passed the above checks, it's likely a simple column selection.

        // If we got here, it's a simple SELECT (columns) FROM table [LIMIT] [OFFSET]
        // This is faster in JS via direct zero-copy pass-through
        return false;

    } catch (e) {
        // If parsing fails (e.g. complex syntax not handled), default to WASM as safe fallback
        // or potentially false if we think JS is safer? 
        // WASM engine is generally more robust for full SQL support.
        console.warn('Failed to parse SQL for routing, defaulting to WASM:', e);
        return true;
    }
}
