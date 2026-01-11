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
import { getWasmSqlExecutor } from './wasm-sql-bridge.js';
import { E } from './data-types.js';

// Simple regex patterns for time travel commands (no heavy SQL parser needed)
const TIME_TRAVEL_PATTERNS = {
    // SHOW VERSIONS FOR table_name
    showVersions: /^\s*SHOW\s+VERSIONS\s+FOR\s+(\w+)\s*$/i,
    // RESTORE table_name TO VERSION n
    restoreTable: /^\s*RESTORE\s+(\w+)\s+TO\s+VERSION\s+(\d+)\s*$/i,
    // SELECT ... FROM table VERSION AS OF n
    versionAsOf: /FROM\s+(\w+)\s+VERSION\s+AS\s+OF\s+(\d+)/i,
};

function parseTimeTravelCommand(sql) {
    let match;
    if ((match = sql.match(TIME_TRAVEL_PATTERNS.showVersions))) {
        return { type: 'SHOW_VERSIONS', table: match[1] };
    }
    if ((match = sql.match(TIME_TRAVEL_PATTERNS.restoreTable))) {
        return { type: 'RESTORE_TABLE', table: match[1], version: parseInt(match[2], 10) };
    }
    if ((match = sql.match(TIME_TRAVEL_PATTERNS.versionAsOf))) {
        return { type: 'SELECT_VERSION', table: match[1], version: parseInt(match[2], 10), sql };
    }
    return null;
}

// ============================================================================
// read_lance() URL Extraction and SQL Rewriting
// ============================================================================

/**
 * Extract read_lance('url') patterns from SQL
 * @param {string} sql - SQL query
 * @returns {Array<{fullMatch: string, url: string, alias: string}>}
 */
function extractReadLanceUrls(sql) {
    // Match read_lance('url') with optional alias (AS alias or just alias)
    // The alias must NOT be a SQL keyword
    const sqlKeywords = /^(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|AND|OR|NOT|IN|LIKE|BETWEEN|GROUP|ORDER|BY|HAVING|LIMIT|OFFSET|UNION|EXCEPT|INTERSECT|AS|NULL|TRUE|FALSE|IS|CASE|WHEN|THEN|ELSE|END|DISTINCT|ALL|ASC|DESC|CREATE|DROP|INSERT|UPDATE|DELETE|INTO|VALUES|TABLE|INDEX|VIEW|SET|WITH|RECURSIVE)$/i;

    // Match read_lance('url') followed optionally by AS alias or just alias
    // But only capture the alias if it's after AS, to avoid matching SQL keywords
    const pattern = /read_lance\s*\(\s*'([^']+)'\s*\)(?:\s+AS\s+(\w+)|\s+(\w+))?/gi;
    const urls = [];
    let match;
    while ((match = pattern.exec(sql)) !== null) {
        // Check both capture groups - AS alias (match[2]) or bare alias (match[3])
        let explicitAlias = match[2];
        let bareAlias = match[3];

        // Only use bare alias if it's not a SQL keyword
        if (bareAlias && sqlKeywords.test(bareAlias)) {
            bareAlias = null;
        }

        const alias = explicitAlias || bareAlias || `_tbl${urls.length}`;

        // Determine what was actually matched (for replacement)
        // If bare alias was matched but is a keyword, don't include it in fullMatch
        let fullMatch = match[0];
        if (match[3] && sqlKeywords.test(match[3])) {
            // Remove the keyword from the match - it's not part of the table reference
            fullMatch = `read_lance('${match[1]}')`;
        }

        console.log(`[Worker] extractReadLanceUrls: found "${fullMatch}" -> alias "${alias}"`);
        urls.push({
            fullMatch,
            url: match[1],
            alias
        });
    }
    console.log(`[Worker] extractReadLanceUrls: found ${urls.length} URLs`);
    return urls;
}

/**
 * Rewrite SQL: replace read_lance('url') with table alias
 * @param {string} sql - Original SQL
 * @param {Array<{fullMatch: string, alias: string}>} urlMappings - URL mappings
 * @returns {string} - Rewritten SQL
 */
function rewriteSqlWithAliases(sql, urlMappings) {
    let rewritten = sql;
    for (const { fullMatch, alias } of urlMappings) {
        console.log(`[Worker] rewriteSqlWithAliases: replacing "${fullMatch}" with "${alias}"`);
        rewritten = rewritten.replace(fullMatch, alias);
    }
    console.log(`[Worker] rewriteSqlWithAliases: result = "${rewritten}"`);
    return rewritten;
}

/**
 * Fetch and parse remote Lance dataset
 * @param {string} url - Remote URL
 * @param {number} limit - Max rows to fetch
 * @returns {Promise<{columns: Object, rowCount: number, columnNames: string[]}|null>}
 */
async function fetchRemoteLance(url, limit = 10000) {
    try {
        console.log(`[Worker] Fetching remote Lance: ${url}`);

        // Try to load .meta.json sidecar first (fastest path)
        const sidecarUrl = `${url}/.meta.json`;
        let schema = null;
        let fragments = [];
        let columnTypes = [];

        try {
            const sidecarResponse = await fetch(sidecarUrl);
            if (sidecarResponse.ok) {
                const sidecar = await sidecarResponse.json();
                schema = sidecar.schema;
                fragments = sidecar.fragments || [];
                columnTypes = schema.map(col => {
                    const type = col.type;
                    if (type.startsWith('vector[')) return 'vector';
                    if (type === 'float64' || type === 'double') return 'float64';
                    if (type === 'float32') return 'float32';
                    if (type.includes('int64')) return 'int64';
                    if (type.includes('int')) return 'int32';
                    if (type === 'string') return 'string';
                    return 'unknown';
                });
                console.log(`[Worker] Loaded sidecar: ${schema.length} columns, ${fragments.length} fragments`);
            }
        } catch (e) {
            console.log(`[Worker] No sidecar available: ${e.message}`);
        }

        // If no sidecar, try to find manifest
        if (!schema) {
            // Find latest manifest by probing versions
            let manifestVersion = 0;
            const checkVersions = [1, 5, 10, 20, 50, 100];
            for (const v of checkVersions) {
                try {
                    const response = await fetch(`${url}/_versions/${v}.manifest`, { method: 'HEAD' });
                    if (response.ok) manifestVersion = v;
                } catch {}
            }
            // Scan forward from highest found
            if (manifestVersion > 0) {
                for (let v = manifestVersion + 1; v <= manifestVersion + 50; v++) {
                    try {
                        const response = await fetch(`${url}/_versions/${v}.manifest`, { method: 'HEAD' });
                        if (response.ok) manifestVersion = v;
                        else break;
                    } catch { break; }
                }
            }

            if (manifestVersion === 0) {
                console.error(`[Worker] No manifest found for ${url}`);
                return null;
            }

            // Parse manifest (simplified - get basic info)
            const manifestResponse = await fetch(`${url}/_versions/${manifestVersion}.manifest`);
            if (!manifestResponse.ok) return null;
            const manifestData = new Uint8Array(await manifestResponse.arrayBuffer());
            const parsed = parseManifestBasic(manifestData);
            schema = parsed.schema;
            fragments = parsed.fragments;
            columnTypes = schema.map(col => col.type || 'unknown');
        }

        if (!schema || schema.length === 0) {
            console.error(`[Worker] No schema found for ${url}`);
            return null;
        }

        // Now fetch actual data from fragments
        // For simplicity, fetch first fragment and limit rows
        if (fragments.length === 0) {
            console.error(`[Worker] No fragments found for ${url}`);
            return null;
        }

        // Fetch and parse fragment data
        const fragmentPath = fragments[0].data_files?.[0] || `${fragments[0].id}.lance`;
        const fragmentUrl = `${url}/data/${fragmentPath}`;
        const result = await fetchAndParseFragment(fragmentUrl, schema, columnTypes, limit);

        return result;
    } catch (e) {
        console.error(`[Worker] Failed to fetch remote Lance:`, e);
        return null;
    }
}

/**
 * Parse manifest to extract basic schema and fragment info
 * @param {Uint8Array} bytes - Manifest data
 * @returns {{schema: Array, fragments: Array}}
 */
function parseManifestBasic(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset);
    const schema = [];
    const fragments = [];

    // Read chunk 1 length
    const chunk1Len = view.getUint32(0, true);
    const chunk2Start = 4 + chunk1Len;

    let protoData;
    if (chunk2Start + 4 < bytes.length) {
        const chunk2Len = view.getUint32(chunk2Start, true);
        if (chunk2Len > 0 && chunk2Start + 4 + chunk2Len <= bytes.length) {
            protoData = bytes.slice(chunk2Start + 4, chunk2Start + 4 + chunk2Len);
        } else {
            protoData = bytes.slice(4, 4 + chunk1Len);
        }
    } else {
        protoData = bytes.slice(4, 4 + chunk1Len);
    }

    let pos = 0;

    const readVarint = () => {
        let result = 0, shift = 0;
        while (pos < protoData.length) {
            const byte = protoData[pos++];
            result |= (byte & 0x7F) << shift;
            if ((byte & 0x80) === 0) break;
            shift += 7;
        }
        return result;
    };

    const skipField = (wireType) => {
        if (wireType === 0) readVarint();
        else if (wireType === 2) pos += readVarint();
        else if (wireType === 5) pos += 4;
        else if (wireType === 1) pos += 8;
    };

    while (pos < protoData.length) {
        const tag = readVarint();
        const fieldNum = tag >> 3;
        const wireType = tag & 0x7;

        if (fieldNum === 1 && wireType === 2) {
            // Field 1 = schema (repeated Field message)
            const fieldLen = readVarint();
            const fieldEnd = pos + fieldLen;
            let name = null, id = null, logicalType = null;

            while (pos < fieldEnd) {
                const fTag = readVarint();
                const fNum = fTag >> 3;
                const fWire = fTag & 0x7;

                if (fWire === 0) {
                    const val = readVarint();
                    if (fNum === 3) id = val;
                } else if (fWire === 2) {
                    const len = readVarint();
                    const content = protoData.slice(pos, pos + len);
                    pos += len;
                    if (fNum === 2) name = new TextDecoder().decode(content);
                    else if (fNum === 5) logicalType = new TextDecoder().decode(content);
                } else {
                    skipField(fWire);
                }
            }
            if (name) schema.push({ name, id, type: logicalType });
        } else if (fieldNum === 2 && wireType === 2) {
            // Field 2 = fragments
            const fragLen = readVarint();
            const fragEnd = pos + fragLen;
            let fragId = null, filePath = null, numRows = 0;

            while (pos < fragEnd) {
                const fTag = readVarint();
                const fNum = fTag >> 3;
                const fWire = fTag & 0x7;

                if (fWire === 0) {
                    const val = readVarint();
                    if (fNum === 1) fragId = val;
                    else if (fNum === 4) numRows = val;
                } else if (fWire === 2) {
                    const len = readVarint();
                    const content = protoData.slice(pos, pos + len);
                    pos += len;
                    if (fNum === 2) {
                        // Parse DataFile message for path
                        let innerPos = 0;
                        while (innerPos < content.length) {
                            const iTag = content[innerPos++];
                            const iNum = iTag >> 3;
                            const iWire = iTag & 0x7;
                            if (iWire === 2) {
                                let iLen = 0, iShift = 0;
                                while (innerPos < content.length) {
                                    const b = content[innerPos++];
                                    iLen |= (b & 0x7F) << iShift;
                                    if ((b & 0x80) === 0) break;
                                    iShift += 7;
                                }
                                if (iNum === 1) filePath = new TextDecoder().decode(content.slice(innerPos, innerPos + iLen));
                                innerPos += iLen;
                            } else if (iWire === 0) {
                                while (innerPos < content.length && (content[innerPos++] & 0x80) !== 0);
                            } else if (iWire === 5) innerPos += 4;
                            else if (iWire === 1) innerPos += 8;
                        }
                    }
                } else {
                    skipField(fWire);
                }
            }
            if (filePath) {
                fragments.push({ id: fragId, data_files: [filePath], num_rows: numRows });
            }
        } else {
            skipField(wireType);
        }
    }

    return { schema, fragments };
}

/**
 * Fetch and parse a Lance fragment file to extract columnar data
 * @param {string} url - Fragment URL
 * @param {Array} schema - Column schema
 * @param {Array} columnTypes - Column types
 * @param {number} limit - Max rows
 * @returns {Promise<{columns: Object, rowCount: number, columnNames: string[]}>}
 */
async function fetchAndParseFragment(url, schema, columnTypes, limit) {
    console.log(`[Worker] Fetching fragment: ${url}`);

    // Ensure WASM is loaded
    if (!wasm) {
        await loadWasm();
        if (!wasm) throw new Error('WASM not loaded');
    }

    // Check file size first with HEAD request
    const headResponse = await fetch(url, { method: 'HEAD' });
    if (!headResponse.ok) {
        throw new Error(`Failed to check fragment size: HTTP ${headResponse.status}`);
    }
    const contentLength = parseInt(headResponse.headers.get('Content-Length') || '0', 10);
    const MAX_FRAGMENT_SIZE = 100 * 1024 * 1024; // 100MB max for full download

    if (contentLength > MAX_FRAGMENT_SIZE) {
        console.warn(`[Worker] Fragment too large (${(contentLength / 1024 / 1024).toFixed(1)}MB) - using schema-based placeholder data`);
        // Return placeholder data for large files
        const columns = {};
        const columnNames = [];
        const actualRows = Math.min(limit, 100);
        for (let i = 0; i < schema.length; i++) {
            const name = schema[i].name;
            const type = columnTypes[i];
            columnNames.push(name);
            if (type === 'float64' || type === 'double' || type === 'int64' || type === 'int32' || type === 'float32') {
                columns[name] = new Float64Array(actualRows).fill(0);
            } else if (type === 'string') {
                columns[name] = Array(actualRows).fill('(large file - use RemoteLanceDataset)');
            } else {
                columns[name] = [];
            }
        }
        return { columns, rowCount: actualRows, columnNames };
    }

    // Fetch the entire fragment file (only for small files)
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch fragment: HTTP ${response.status}`);
    }
    const data = new Uint8Array(await response.arrayBuffer());
    console.log(`[Worker] Fetched fragment: ${data.length} bytes`);

    // Verify Lance magic
    if (data.length < 40) {
        throw new Error('Fragment too small');
    }
    const magic = String.fromCharCode(data[data.length - 4], data[data.length - 3], data[data.length - 2], data[data.length - 1]);
    if (magic !== 'LANC') {
        throw new Error(`Invalid Lance magic: ${magic}`);
    }

    // Load fragment into WASM memory
    const dataPtr = wasm.alloc(data.length);
    if (!dataPtr || dataPtr < 0) {
        throw new Error('Failed to allocate WASM memory for fragment');
    }
    new Uint8Array(wasmMemory.buffer, dataPtr, data.length).set(data);

    // Open the file in WASM
    const openResult = wasm.openFile(dataPtr, data.length);
    if (openResult !== 0) {
        throw new Error(`WASM openFile failed: ${openResult}`);
    }

    // Get actual row count from WASM
    const numColumns = wasm.getNumColumns();
    const rowCount = Number(wasm.getRowCount(0)); // Get row count from first column
    const actualRows = Math.min(rowCount, limit);

    console.log(`[Worker] Fragment has ${numColumns} columns, ${rowCount} rows, reading ${actualRows}`);

    const columns = {};
    const columnNames = [];

    // Read each column using WASM
    for (let i = 0; i < schema.length && i < numColumns; i++) {
        const name = schema[i].name;
        const type = columnTypes[i];
        columnNames.push(name);

        try {
            if (type === 'float64' || type === 'double') {
                const outPtr = wasm.allocFloat64Buffer(actualRows);
                if (outPtr) {
                    const count = wasm.readFloat64Column(i, outPtr, actualRows);
                    columns[name] = new Float64Array(wasmMemory.buffer, outPtr, count).slice();
                } else {
                    columns[name] = new Float64Array(actualRows).fill(0);
                }
            } else if (type === 'int64') {
                const outPtr = wasm.allocInt64Buffer(actualRows);
                if (outPtr) {
                    const count = wasm.readInt64Column(i, outPtr, actualRows);
                    // Convert BigInt64Array to Float64Array for WASM SQL registration
                    const bigArr = new BigInt64Array(wasmMemory.buffer, outPtr, count);
                    const floatArr = new Float64Array(count);
                    for (let j = 0; j < count; j++) {
                        floatArr[j] = Number(bigArr[j]);
                    }
                    columns[name] = floatArr;
                } else {
                    columns[name] = new Float64Array(actualRows).fill(0);
                }
            } else if (type === 'int32') {
                const outPtr = wasm.allocInt32Buffer(actualRows);
                if (outPtr) {
                    const count = wasm.readInt32Column(i, outPtr, actualRows);
                    const intArr = new Int32Array(wasmMemory.buffer, outPtr, count);
                    // Convert to Float64Array for WASM SQL registration
                    const floatArr = new Float64Array(count);
                    for (let j = 0; j < count; j++) {
                        floatArr[j] = intArr[j];
                    }
                    columns[name] = floatArr;
                } else {
                    columns[name] = new Float64Array(actualRows).fill(0);
                }
            } else if (type === 'float32') {
                // Use regular alloc for float32 (4 bytes per element)
                const outPtr = wasm.alloc(actualRows * 4);
                if (outPtr && outPtr > 0) {
                    const count = wasm.readFloat32Column(i, outPtr, actualRows);
                    const floatArr = new Float32Array(wasmMemory.buffer, outPtr, count);
                    // Convert to Float64Array for WASM SQL registration
                    const f64Arr = new Float64Array(count);
                    for (let j = 0; j < count; j++) {
                        f64Arr[j] = floatArr[j];
                    }
                    columns[name] = f64Arr;
                } else {
                    columns[name] = new Float64Array(actualRows).fill(0);
                }
            } else if (type === 'string') {
                // String columns are more complex - use the string reading functions
                // For now, create placeholder - strings need offset/length arrays
                columns[name] = Array(actualRows).fill('(string data)');
            } else if (type === 'vector') {
                // Skip vectors - too large
                columns[name] = [];
            } else {
                columns[name] = [];
            }
        } catch (e) {
            console.warn(`[Worker] Failed to read column ${name}:`, e);
            columns[name] = [];
        }
    }

    // Close the file to free WASM resources
    wasm.closeFile();

    console.log(`[Worker] Read ${actualRows} rows from ${columnNames.length} columns`);

    return { columns, rowCount: actualRows, columnNames };
}

/**
 * Read a protobuf varint from data at offset
 * Returns [value, bytesRead]
 */
function readVarint(data, offset) {
    let result = 0;
    let shift = 0;
    let bytesRead = 0;
    while (offset + bytesRead < data.length) {
        const byte = data[offset + bytesRead];
        bytesRead++;
        result |= (byte & 0x7F) << shift;
        if ((byte & 0x80) === 0) break;
        shift += 7;
        if (shift > 35) break; // Prevent infinite loop
    }
    return [result, bytesRead];
}

/**
 * Parse Lance file data into columnar format
 * @param {Uint8Array} data - Full Lance file data
 * @param {Array} schema - Column schema
 * @param {Array} columnTypes - Column types
 * @param {number} limit - Max rows
 * @returns {{columns: Object, rowCount: number, columnNames: string[]}}
 */
function parseLanceFileData(data, schema, columnTypes, limit) {
    const view = new DataView(data.buffer, data.byteOffset);
    const footerOffset = data.length - 40;

    // Read footer
    const colMetaStart = Number(view.getBigUint64(footerOffset, true));
    const colMetaOffsetsStart = Number(view.getBigUint64(footerOffset + 8, true));
    const numColumns = view.getUint32(footerOffset + 28, true);

    console.log(`[Worker] parseLanceFileData: ${numColumns} columns, metaStart=${colMetaStart}, offsetsStart=${colMetaOffsetsStart}`);

    const columns = {};
    const columnNames = [];
    let rowCount = 0;

    // Parse each column metadata
    for (let i = 0; i < numColumns && i < schema.length; i++) {
        const offsetPos = colMetaOffsetsStart + i * 8;
        if (offsetPos + 8 > data.length) {
            console.warn(`[Worker] Column ${i} offset position out of bounds`);
            continue;
        }
        const metaPos = Number(view.getBigUint64(offsetPos, true));

        if (metaPos >= data.length) {
            console.warn(`[Worker] Column ${i} metadata position ${metaPos} out of bounds`);
            continue;
        }

        // Parse column metadata using protobuf format
        let localOffset = metaPos;
        let name = schema[i]?.name || `col_${i}`;
        let typeStr = columnTypes[i] || 'unknown';
        let dataOffset = 0;
        let colRowCount = 0;
        let dataSize = 0;

        // Parse protobuf fields
        while (localOffset < data.length && localOffset < metaPos + 500) { // Safety limit
            const tagByte = data[localOffset];
            const fieldNum = tagByte >> 3;
            const wireType = tagByte & 0x7;
            localOffset++;

            if (fieldNum === 0) break; // End of message

            if (wireType === 0) { // Varint
                const [val, bytes] = readVarint(data, localOffset);
                localOffset += bytes;
                if (fieldNum === 3) { /* nullable */ }
                else if (fieldNum === 5) colRowCount = val;
                else if (fieldNum === 6) dataSize = val;
            } else if (wireType === 1) { // Fixed 64-bit
                if (localOffset + 8 > data.length) break;
                if (fieldNum === 4) {
                    dataOffset = Number(view.getBigUint64(localOffset, true));
                }
                localOffset += 8;
            } else if (wireType === 2) { // Length-delimited
                const [len, lenBytes] = readVarint(data, localOffset);
                localOffset += lenBytes;
                if (localOffset + len > data.length) break;

                if (fieldNum === 1) { // name
                    name = new TextDecoder().decode(data.slice(localOffset, localOffset + len));
                } else if (fieldNum === 2) { // type
                    typeStr = new TextDecoder().decode(data.slice(localOffset, localOffset + len));
                }
                localOffset += len;
            } else if (wireType === 5) { // Fixed 32-bit
                localOffset += 4;
            } else {
                break; // Unknown wire type
            }
        }

        columnNames.push(name);
        if (rowCount === 0) rowCount = colRowCount;

        console.log(`[Worker] Column ${i}: name=${name}, type=${typeStr}, dataOffset=${dataOffset}, rowCount=${colRowCount}, dataSize=${dataSize}`);

        // Read column data based on type
        const actualRows = Math.min(rowCount, limit);
        if (actualRows === 0 || dataOffset === 0) {
            console.log(`[Worker] Column ${name}: skipping (actualRows=${actualRows}, dataOffset=${dataOffset})`);
            columns[name] = [];
            continue;
        }

        const type = columnTypes[i] || typeStr;

        try {
            if (type === 'float64' || typeStr === 'float64' || typeStr === 'double') {
                columns[name] = new Float64Array(data.buffer, data.byteOffset + dataOffset, actualRows).slice();
            } else if (type === 'int64' || typeStr === 'int64') {
                const bigIntArr = new BigInt64Array(data.buffer, data.byteOffset + dataOffset, actualRows);
                columns[name] = new BigInt64Array(bigIntArr);
            } else if (type === 'int32' || typeStr === 'int32') {
                columns[name] = new Int32Array(data.buffer, data.byteOffset + dataOffset, actualRows).slice();
            } else if (type === 'float32' || typeStr === 'float32') {
                columns[name] = new Float32Array(data.buffer, data.byteOffset + dataOffset, actualRows).slice();
            } else if (type === 'string' || typeStr === 'string') {
                // String data: bytes first, then offsets
                const offsetsLen = (actualRows + 1) * 4;
                const dataBytesLen = dataSize - offsetsLen;
                if (dataBytesLen > 0 && dataOffset + dataSize <= data.length) {
                    const offsets = new Uint32Array(data.buffer, data.byteOffset + dataOffset + dataBytesLen, actualRows + 1);
                    const bytes = data.slice(dataOffset, dataOffset + dataBytesLen);
                    const strings = [];
                    for (let j = 0; j < actualRows; j++) {
                        const start = offsets[j];
                        const end = offsets[j + 1];
                        if (start <= end && end <= bytes.length) {
                            strings.push(new TextDecoder().decode(bytes.slice(start, end)));
                        } else {
                            strings.push('');
                        }
                    }
                    columns[name] = strings;
                } else {
                    columns[name] = [];
                }
            } else if (type === 'vector' || typeStr.startsWith('fixed_size_list')) {
                // Skip vectors for now - they're large
                columns[name] = [];
            } else {
                columns[name] = [];
            }
        } catch (e) {
            console.warn(`[Worker] Failed to read column ${name}:`, e);
            columns[name] = [];
        }
    }

    return { columns, rowCount: Math.min(rowCount, limit), columnNames };
}

/**
 * Load and parse OPFS Lance file
 * @param {string} path - OPFS path (without opfs:// prefix)
 * @param {number} limit - Max rows to read
 * @returns {Promise<{columns: Object, rowCount: number, columnNames: string[]}|null>}
 */
async function loadOPFSLance(path, limit = 10000) {
    try {
        console.log(`[Worker] Loading OPFS Lance: ${path}`);

        // Load file from OPFS
        const parts = path.split('/').filter(p => p);
        const fileName = parts.pop();
        const dir = await getOPFSDir(parts);
        const fileHandle = await dir.getFileHandle(fileName);
        const file = await fileHandle.getFile();
        const data = new Uint8Array(await file.arrayBuffer());

        if (!data || data.length === 0) {
            console.error(`[Worker] Empty OPFS file: ${path}`);
            return null;
        }

        // Verify Lance magic
        if (data.length < 40) {
            console.error(`[Worker] File too small for Lance format: ${path}`);
            return null;
        }

        const magic = String.fromCharCode(data[data.length - 4], data[data.length - 3], data[data.length - 2], data[data.length - 1]);
        if (magic !== 'LANC') {
            console.error(`[Worker] Invalid Lance magic in ${path}: ${magic}`);
            return null;
        }

        // Parse the Lance file - extract schema from metadata
        const view = new DataView(data.buffer, data.byteOffset);
        const footerOffset = data.length - 40;
        const colMetaOffsetsStart = Number(view.getBigUint64(footerOffset + 8, true));
        const numColumns = view.getUint32(footerOffset + 28, true);

        // Build schema from column metadata
        const schema = [];
        const columnTypes = [];
        for (let i = 0; i < numColumns; i++) {
            const offsetPos = colMetaOffsetsStart + i * 8;
            const metaPos = Number(view.getBigUint64(offsetPos, true));

            let localOffset = metaPos;
            // Name
            data[localOffset++]; // tag
            const nameLen = data[localOffset++];
            const name = new TextDecoder().decode(data.slice(localOffset, localOffset + nameLen));
            localOffset += nameLen;

            // Type
            data[localOffset++]; // tag
            const typeLen = data[localOffset++];
            const typeStr = new TextDecoder().decode(data.slice(localOffset, localOffset + typeLen));

            schema.push({ name, type: typeStr });
            columnTypes.push(typeStr);
        }

        return parseLanceFileData(data, schema, columnTypes, limit);
    } catch (e) {
        console.error(`[Worker] Failed to load OPFS Lance:`, e);
        return null;
    }
}

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

    // Strip SQL comments (-- line comments and /* block comments */)
    const cleanSql = sql
        .replace(/--[^\n]*/g, '') // Remove -- line comments
        .replace(/\/\*[\s\S]*?\*\//g, '') // Remove /* block comments */
        .trim();

    const executor = getWasmSqlExecutor();

    // Step 1: Extract and handle read_lance('url') URLs BEFORE parsing
    const urlMappings = extractReadLanceUrls(cleanSql);

    for (const { url, alias } of urlMappings) {
        // Skip if already registered
        if (executor.hasTable(alias)) {
            console.log(`[Worker] Table ${alias} already registered`);
            continue;
        }

        let data = null;
        let loadError = null;
        try {
            if (url.startsWith('https://') || url.startsWith('http://')) {
                console.log(`[Worker] Fetching remote Lance: ${url} as ${alias}`);
                data = await fetchRemoteLance(url);
            } else if (url.startsWith('opfs://')) {
                const opfsPath = url.replace('opfs://', '');
                console.log(`[Worker] Loading OPFS Lance: ${opfsPath} as ${alias}`);
                data = await loadOPFSLance(opfsPath);
            } else {
                // Treat as OPFS path
                console.log(`[Worker] Loading OPFS Lance (no prefix): ${url} as ${alias}`);
                data = await loadOPFSLance(url);
            }
        } catch (e) {
            loadError = e;
            console.error(`[Worker] Failed to load ${url}:`, e);
        }

        if (data) {
            console.log(`[Worker] Registering table ${alias} with ${data.rowCount} rows, columns: ${Object.keys(data.columns).join(', ')}`);
            executor.registerTable(alias, data.columns, data.rowCount, url);
            // Verify registration succeeded
            if (!executor.hasTable(alias)) {
                throw new Error(`Failed to register table ${alias} from ${url}`);
            }
        } else {
            const reason = loadError ? loadError.message : 'file may not exist or returned empty data';
            throw new Error(`Could not load read_lance('${url}'): ${reason}`);
        }
    }

    // Step 2: Rewrite SQL to use aliases instead of read_lance() calls
    const rewrittenSql = urlMappings.length > 0 ? rewriteSqlWithAliases(cleanSql, urlMappings) : cleanSql;

    // Step 3: Extract remaining table names from rewritten SQL
    const tableNames = executor.getTableNames(rewrittenSql);

    // Step 4: Register local DB tables
    for (const tableName of tableNames) {
        // Skip already registered tables (including our aliases)
        const exists = executor.hasTable(tableName);
        if (exists) continue;

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

    // Execute SQL in WASM (use rewritten SQL with aliases instead of read_lance() calls)
    return executor.execute(rewrittenSql);
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

            __assert_fail: (msgPtr, filePtr, line, funcPtr) => {
                const decoder = new TextDecoder();
                const msg = decoder.decode(new Uint8Array(wasmMemory.buffer, msgPtr).subarray(0, 100));
                console.error(`[WASM ASSERT] ${msg} at line ${line}`);
            },
            js_log: (ptr, len) => {
                const decoder = new TextDecoder();
                const msg = decoder.decode(new Uint8Array(wasmMemory.buffer, ptr, len));
                console.log(`[WASM LOG] ${msg}`);
                for (const port of ports) {
                    port.postMessage({ type: 'log', message: msg, marker: '__WASM_LOG_BRIDGE__' });
                }
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
        const url = new URL('./lanceql.wasm', import.meta.url);
        url.searchParams.set('v', Date.now().toString());
        const response = await fetch(url);
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

            // Check for time travel commands (simple regex, no heavy parser)
            const ttCmd = parseTimeTravelCommand(args.sql);

            if (ttCmd?.type === 'SHOW_VERSIONS') {
                const versions = await db.listVersions(ttCmd.table);
                result = {
                    _format: 'columnar',
                    columns: ['version', 'timestamp', 'operation', 'rowCount'],
                    rowCount: versions.length,
                    data: {
                        version: new Float64Array(versions.map(v => v.version)),
                        timestamp: versions.map(v => new Date(v.timestamp).toISOString()),
                        operation: versions.map(v => v.operation),
                        rowCount: new Float64Array(versions.map(v => v.rowCount))
                    }
                };
            } else if (ttCmd?.type === 'RESTORE_TABLE') {
                const restoreResult = await db.restoreToVersion(ttCmd.table, ttCmd.version);
                result = {
                    _format: 'columnar',
                    columns: ['status', 'newVersion'],
                    rowCount: 1,
                    data: {
                        status: ['restored'],
                        newVersion: new Float64Array([restoreResult.newVersion])
                    }
                };
            } else if (ttCmd?.type === 'SELECT_VERSION') {
                // SELECT with VERSION AS OF - route to selectAtVersion
                const rows = await db.selectAtVersion(ttCmd.table, ttCmd.version, {});
                // Convert rows to columnar format
                if (rows.length > 0) {
                    const columns = Object.keys(rows[0]);
                    const data = {};
                    for (const col of columns) {
                        data[col] = rows.map(r => r[col]);
                    }
                    result = {
                        _format: 'columnar',
                        columns,
                        rowCount: rows.length,
                        data
                    };
                } else {
                    result = { _format: 'columnar', columns: [], rowCount: 0, data: {} };
                }
            } else {
                // Standard SQL - use WASM executor
                result = await executeWasmSqlFull(db, args.sql);
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
        // Time travel (versioning) operations
        else if (method === 'db:listVersions') {
            const db = await getDatabase(args.db);
            result = await db.listVersions(args.tableName);
        } else if (method === 'db:selectAtVersion') {
            const db = await getDatabase(args.db);
            const options = { ...args.options };
            if (args.where) {
                options.where = (row) => evalWhere(args.where, row);
            }
            result = await db.selectAtVersion(args.tableName, args.version, options);
        } else if (method === 'db:restoreTable') {
            const db = await getDatabase(args.db);
            result = await db.restoreToVersion(args.tableName, args.version);
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
        } else if (method === 'vault:tables') {
            const vault = await getVault();
            result = vault._db ? vault._db.listTables() : [];
        } else if (method === 'vault:exec') {
            const vault = await getVault();
            const db = vault._db;

            // Check for time travel commands (simple regex, no heavy parser)
            const ttCmd = parseTimeTravelCommand(args.sql);

            if (ttCmd?.type === 'SHOW_VERSIONS') {
                const versions = await db.listVersions(ttCmd.table);
                result = {
                    _format: 'columnar',
                    columns: ['version', 'timestamp', 'operation', 'rowCount'],
                    rowCount: versions.length,
                    data: {
                        version: new Float64Array(versions.map(v => v.version)),
                        timestamp: versions.map(v => new Date(v.timestamp).toISOString()),
                        operation: versions.map(v => v.operation),
                        rowCount: new Float64Array(versions.map(v => v.rowCount))
                    }
                };
            } else if (ttCmd?.type === 'RESTORE_TABLE') {
                const restoreResult = await db.restoreToVersion(ttCmd.table, ttCmd.version);
                result = {
                    _format: 'columnar',
                    columns: ['status', 'newVersion'],
                    rowCount: 1,
                    data: {
                        status: ['restored'],
                        newVersion: new Float64Array([restoreResult.newVersion])
                    }
                };
            } else if (ttCmd?.type === 'SELECT_VERSION') {
                // SELECT with VERSION AS OF - route to selectAtVersion
                const rows = await db.selectAtVersion(ttCmd.table, ttCmd.version, {});
                // Convert rows to columnar format
                if (rows.length > 0) {
                    const columns = Object.keys(rows[0]);
                    const data = {};
                    for (const col of columns) {
                        data[col] = rows.map(r => r[col]);
                    }
                    result = {
                        _format: 'columnar',
                        columns,
                        rowCount: rows.length,
                        data
                    };
                } else {
                    result = { _format: 'columnar', columns: [], rowCount: 0, data: {} };
                }
            } else {
                // Standard SQL - use WASM executor
                result = await executeWasmSqlFull(db, args.sql);
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
        // Map WASM error names to human-readable messages
        let errorMsg = error.stack || error.message;
        if (errorMsg.includes('TableDoesNotExist')) {
            errorMsg = 'Table does not exist';
        } else if (errorMsg.includes('ColumnDoesNotExist')) {
            errorMsg = 'Column does not exist';
        } else if (errorMsg.includes('UnknownEmbeddingModel')) {
            errorMsg = 'Unknown embedding model';
        }
        port.postMessage({ id, error: errorMsg });
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
