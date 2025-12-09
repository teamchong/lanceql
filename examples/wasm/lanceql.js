/**
 * LanceQL WASM Loader
 *
 * JavaScript wrapper for the LanceQL WebAssembly module.
 * Provides a high-level API for reading Lance files in the browser.
 * Supports both local files and remote URLs via HTTP Range requests.
 */

export class LanceQL {
    constructor(wasmInstance) {
        this.wasm = wasmInstance.exports;
        this.memory = this.wasm.memory;
    }

    /**
     * Load LanceQL from a WASM file path or URL.
     * @param {string} wasmPath - Path to the lanceql.wasm file
     * @returns {Promise<LanceQL>}
     */
    static async load(wasmPath = './lanceql.wasm') {
        const response = await fetch(wasmPath);
        const wasmBytes = await response.arrayBuffer();
        const wasmModule = await WebAssembly.instantiate(wasmBytes, {});
        return new LanceQL(wasmModule.instance);
    }

    /**
     * Get the library version.
     * @returns {string} Version string like "0.1.0"
     */
    getVersion() {
        const v = this.wasm.getVersion();
        const major = (v >> 16) & 0xFF;
        const minor = (v >> 8) & 0xFF;
        const patch = v & 0xFF;
        return `${major}.${minor}.${patch}`;
    }

    /**
     * Open a Lance file from an ArrayBuffer (local file).
     * @param {ArrayBuffer} data - The Lance file data
     * @returns {LanceFile}
     */
    open(data) {
        return new LanceFile(this, data);
    }

    /**
     * Open a Lance file from a URL using HTTP Range requests.
     * Only fetches metadata initially - column data is fetched on demand.
     * @param {string} url - URL to the Lance file
     * @returns {Promise<RemoteLanceFile>}
     */
    async openUrl(url) {
        return await RemoteLanceFile.open(this, url);
    }

    /**
     * Parse footer from Lance file data (without opening).
     * @param {ArrayBuffer} data
     * @returns {{numColumns: number, majorVersion: number, minorVersion: number} | null}
     */
    parseFooter(data) {
        const bytes = new Uint8Array(data);
        const ptr = this.wasm.alloc(bytes.length);
        if (!ptr) return null;

        try {
            new Uint8Array(this.memory.buffer).set(bytes, ptr);

            const numColumns = this.wasm.parseFooterGetColumns(ptr, bytes.length);
            const majorVersion = this.wasm.parseFooterGetMajorVersion(ptr, bytes.length);
            const minorVersion = this.wasm.parseFooterGetMinorVersion(ptr, bytes.length);

            if (numColumns === 0 && majorVersion === 0) {
                return null; // Invalid file
            }

            return { numColumns, majorVersion, minorVersion };
        } finally {
            this.wasm.free(ptr, bytes.length);
        }
    }

    /**
     * Check if data is a valid Lance file.
     * @param {ArrayBuffer} data
     * @returns {boolean}
     */
    isValid(data) {
        const bytes = new Uint8Array(data);
        const ptr = this.wasm.alloc(bytes.length);
        if (!ptr) return false;

        try {
            new Uint8Array(this.memory.buffer).set(bytes, ptr);
            return this.wasm.isValidLanceFile(ptr, bytes.length) === 1;
        } finally {
            this.wasm.free(ptr, bytes.length);
        }
    }
}

/**
 * Represents an open Lance file (loaded entirely in memory).
 */
export class LanceFile {
    constructor(lanceql, data) {
        this.lanceql = lanceql;
        this.wasm = lanceql.wasm;
        this.memory = lanceql.memory;

        // Copy data to WASM memory
        const bytes = new Uint8Array(data);
        this.dataPtr = this.wasm.alloc(bytes.length);
        if (!this.dataPtr) {
            throw new Error('Failed to allocate memory for Lance file');
        }
        this.dataLen = bytes.length;
        new Uint8Array(this.memory.buffer).set(bytes, this.dataPtr);

        // Open the file
        const result = this.wasm.openFile(this.dataPtr, this.dataLen);
        if (result === 0) {
            this.wasm.free(this.dataPtr, this.dataLen);
            throw new Error('Failed to open Lance file');
        }
    }

    /**
     * Close the file and free memory.
     */
    close() {
        this.wasm.closeFile();
        if (this.dataPtr) {
            this.wasm.free(this.dataPtr, this.dataLen);
            this.dataPtr = null;
        }
    }

    /**
     * Get the number of columns.
     * @returns {number}
     */
    get numColumns() {
        return this.wasm.getNumColumns();
    }

    /**
     * Get the row count for a column.
     * @param {number} colIdx
     * @returns {bigint}
     */
    getRowCount(colIdx) {
        return this.wasm.getRowCount(colIdx);
    }

    /**
     * Get debug info for a column.
     * @param {number} colIdx
     * @returns {{offset: bigint, size: bigint, rows: bigint}}
     */
    getColumnDebugInfo(colIdx) {
        return {
            offset: this.wasm.getColumnBufferOffset(colIdx),
            size: this.wasm.getColumnBufferSize(colIdx),
            rows: this.wasm.getRowCount(colIdx)
        };
    }

    /**
     * Read an int64 column as a BigInt64Array.
     * @param {number} colIdx - Column index
     * @returns {BigInt64Array}
     */
    readInt64Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new BigInt64Array(0);

        const bufPtr = this.wasm.allocInt64Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate int64 buffer');

        try {
            const count = this.wasm.readInt64Column(colIdx, bufPtr, rowCount);
            const result = new BigInt64Array(count);
            const view = new BigInt64Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.freeInt64Buffer(bufPtr, rowCount);
        }
    }

    /**
     * Read a float64 column as a Float64Array.
     * @param {number} colIdx - Column index
     * @returns {Float64Array}
     */
    readFloat64Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Float64Array(0);

        const bufPtr = this.wasm.allocFloat64Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate float64 buffer');

        try {
            const count = this.wasm.readFloat64Column(colIdx, bufPtr, rowCount);
            const result = new Float64Array(count);
            const view = new Float64Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.freeFloat64Buffer(bufPtr, rowCount);
        }
    }

    // ========================================================================
    // Query Methods
    // ========================================================================

    /**
     * Filter operator constants.
     */
    static Op = {
        EQ: 0,  // Equal
        NE: 1,  // Not equal
        LT: 2,  // Less than
        LE: 3,  // Less than or equal
        GT: 4,  // Greater than
        GE: 5   // Greater than or equal
    };

    /**
     * Filter int64 column and return matching row indices.
     * @param {number} colIdx - Column index
     * @param {number} op - Comparison operator (use LanceFile.Op)
     * @param {bigint|number} value - Value to compare against
     * @returns {Uint32Array} Array of matching row indices
     */
    filterInt64(colIdx, op, value) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint32Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(rowCount);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        try {
            const count = this.wasm.filterInt64Column(
                colIdx, op, BigInt(value), idxPtr, rowCount
            );
            const result = new Uint32Array(count);
            const view = new Uint32Array(this.memory.buffer, idxPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, rowCount * 4);
        }
    }

    /**
     * Filter float64 column and return matching row indices.
     * @param {number} colIdx - Column index
     * @param {number} op - Comparison operator (use LanceFile.Op)
     * @param {number} value - Value to compare against
     * @returns {Uint32Array} Array of matching row indices
     */
    filterFloat64(colIdx, op, value) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint32Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(rowCount);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        try {
            const count = this.wasm.filterFloat64Column(
                colIdx, op, value, idxPtr, rowCount
            );
            const result = new Uint32Array(count);
            const view = new Uint32Array(this.memory.buffer, idxPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, rowCount * 4);
        }
    }

    /**
     * Read int64 values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {BigInt64Array}
     */
    readInt64AtIndices(colIdx, indices) {
        if (indices.length === 0) return new BigInt64Array(0);

        // Copy indices to WASM memory
        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocInt64Buffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);

            const count = this.wasm.readInt64AtIndices(
                colIdx, idxPtr, indices.length, outPtr
            );

            const result = new BigInt64Array(count);
            const view = new BigInt64Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.freeInt64Buffer(outPtr, indices.length);
        }
    }

    /**
     * Read float64 values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {Float64Array}
     */
    readFloat64AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Float64Array(0);

        // Copy indices to WASM memory
        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocFloat64Buffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);

            const count = this.wasm.readFloat64AtIndices(
                colIdx, idxPtr, indices.length, outPtr
            );

            const result = new Float64Array(count);
            const view = new Float64Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.freeFloat64Buffer(outPtr, indices.length);
        }
    }

    // ========================================================================
    // Aggregation Methods
    // ========================================================================

    /**
     * Sum all values in an int64 column.
     * @param {number} colIdx - Column index
     * @returns {bigint}
     */
    sumInt64(colIdx) {
        return this.wasm.sumInt64Column(colIdx);
    }

    /**
     * Sum all values in a float64 column.
     * @param {number} colIdx - Column index
     * @returns {number}
     */
    sumFloat64(colIdx) {
        return this.wasm.sumFloat64Column(colIdx);
    }

    /**
     * Get minimum value in an int64 column.
     * @param {number} colIdx - Column index
     * @returns {bigint}
     */
    minInt64(colIdx) {
        return this.wasm.minInt64Column(colIdx);
    }

    /**
     * Get maximum value in an int64 column.
     * @param {number} colIdx - Column index
     * @returns {bigint}
     */
    maxInt64(colIdx) {
        return this.wasm.maxInt64Column(colIdx);
    }

    /**
     * Get average of a float64 column.
     * @param {number} colIdx - Column index
     * @returns {number}
     */
    avgFloat64(colIdx) {
        return this.wasm.avgFloat64Column(colIdx);
    }

    // ========================================================================
    // String Column Methods
    // ========================================================================

    /**
     * Get the number of strings in a column.
     * @param {number} colIdx - Column index
     * @returns {number}
     */
    getStringCount(colIdx) {
        return Number(this.wasm.getStringCount(colIdx));
    }

    /**
     * Read a single string at a specific row index.
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Row index
     * @returns {string}
     */
    readStringAt(colIdx, rowIdx) {
        const maxLen = 4096; // Max string length to read
        const bufPtr = this.wasm.allocStringBuffer(maxLen);
        if (!bufPtr) throw new Error('Failed to allocate string buffer');

        try {
            const actualLen = this.wasm.readStringAt(colIdx, rowIdx, bufPtr, maxLen);
            if (actualLen === 0) return '';

            const bytes = new Uint8Array(this.memory.buffer, bufPtr, Math.min(actualLen, maxLen));
            return new TextDecoder().decode(bytes);
        } finally {
            this.wasm.free(bufPtr, maxLen);
        }
    }

    /**
     * Read all strings from a column.
     * @param {number} colIdx - Column index
     * @param {number} limit - Maximum number of strings to read
     * @returns {string[]}
     */
    readStringColumn(colIdx, limit = 1000) {
        const count = Math.min(this.getStringCount(colIdx), limit);
        if (count === 0) return [];

        const results = [];
        for (let i = 0; i < count; i++) {
            results.push(this.readStringAt(colIdx, i));
        }
        return results;
    }

    /**
     * Read strings at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {string[]}
     */
    readStringsAtIndices(colIdx, indices) {
        if (indices.length === 0) return [];

        const maxTotalLen = 1024 * 1024; // 1MB total buffer
        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const strBufPtr = this.wasm.allocStringBuffer(maxTotalLen);
        if (!strBufPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate string buffer');
        }

        const lenBufPtr = this.wasm.allocU32Buffer(indices.length);
        if (!lenBufPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(strBufPtr, maxTotalLen);
            throw new Error('Failed to allocate length buffer');
        }

        try {
            // Copy indices to WASM
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);

            // Read strings
            const totalWritten = this.wasm.readStringsAtIndices(
                colIdx, idxPtr, indices.length, strBufPtr, maxTotalLen, lenBufPtr
            );

            // Get lengths
            const lengths = new Uint32Array(this.memory.buffer, lenBufPtr, indices.length);

            // Decode strings
            const results = [];
            let offset = 0;
            for (let i = 0; i < indices.length; i++) {
                const len = lengths[i];
                if (len > 0 && offset + len <= totalWritten) {
                    const bytes = new Uint8Array(this.memory.buffer, strBufPtr + offset, len);
                    results.push(new TextDecoder().decode(bytes));
                    offset += len;
                } else {
                    results.push('');
                }
            }
            return results;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(strBufPtr, maxTotalLen);
            this.wasm.free(lenBufPtr, indices.length * 4);
        }
    }

    // ========================================================================
    // DataFrame-like API
    // ========================================================================

    /**
     * Create a DataFrame-like query builder for this file.
     * @returns {DataFrame}
     */
    df() {
        return new DataFrame(this);
    }
}

/**
 * DataFrame-like query builder for fluent queries.
 */
export class DataFrame {
    constructor(file) {
        this.file = file;
        this._filterOps = [];  // Array of {colIdx, op, value, type}
        this._selectCols = null;
        this._limitValue = null;
    }

    /**
     * Filter rows where column matches condition.
     * @param {number} colIdx - Column index
     * @param {string} op - Operator: '=', '!=', '<', '<=', '>', '>='
     * @param {number|bigint} value - Value to compare
     * @param {string} type - 'int64' or 'float64'
     * @returns {DataFrame}
     */
    filter(colIdx, op, value, type = 'int64') {
        const opMap = {
            '=': LanceFile.Op.EQ, '==': LanceFile.Op.EQ,
            '!=': LanceFile.Op.NE, '<>': LanceFile.Op.NE,
            '<': LanceFile.Op.LT,
            '<=': LanceFile.Op.LE,
            '>': LanceFile.Op.GT,
            '>=': LanceFile.Op.GE
        };

        const df = new DataFrame(this.file);
        df._filterOps = [...this._filterOps, { colIdx, op: opMap[op], value, type }];
        df._selectCols = this._selectCols;
        df._limitValue = this._limitValue;
        return df;
    }

    /**
     * Select specific columns.
     * @param {...number} colIndices - Column indices to select
     * @returns {DataFrame}
     */
    select(...colIndices) {
        const df = new DataFrame(this.file);
        df._filterOps = [...this._filterOps];
        df._selectCols = colIndices;
        df._limitValue = this._limitValue;
        return df;
    }

    /**
     * Limit number of results.
     * @param {number} n - Maximum rows
     * @returns {DataFrame}
     */
    limit(n) {
        const df = new DataFrame(this.file);
        df._filterOps = [...this._filterOps];
        df._selectCols = this._selectCols;
        df._limitValue = n;
        return df;
    }

    /**
     * Execute the query and return row indices.
     * @returns {Uint32Array}
     */
    collectIndices() {
        let indices = null;

        // Apply filters
        for (const f of this._filterOps) {
            let newIndices;
            if (f.type === 'int64') {
                newIndices = this.file.filterInt64(f.colIdx, f.op, f.value);
            } else {
                newIndices = this.file.filterFloat64(f.colIdx, f.op, f.value);
            }

            if (indices === null) {
                indices = newIndices;
            } else {
                // Intersect indices
                const set = new Set(newIndices);
                indices = indices.filter(i => set.has(i));
                indices = new Uint32Array(indices);
            }
        }

        // If no filters, get all row indices
        if (indices === null) {
            const rowCount = Number(this.file.getRowCount(0));
            indices = new Uint32Array(rowCount);
            for (let i = 0; i < rowCount; i++) indices[i] = i;
        }

        // Apply limit
        if (this._limitValue !== null && indices.length > this._limitValue) {
            indices = indices.slice(0, this._limitValue);
        }

        return indices;
    }

    /**
     * Execute the query and return results as arrays.
     * @returns {Object} Object with column data arrays
     */
    collect() {
        const indices = this.collectIndices();
        const result = { _indices: indices };

        const cols = this._selectCols ||
            Array.from({ length: this.file.numColumns }, (_, i) => i);

        for (const colIdx of cols) {
            // Try int64 first, then float64
            try {
                result[`col${colIdx}`] = this.file.readInt64AtIndices(colIdx, indices);
            } catch {
                try {
                    result[`col${colIdx}`] = this.file.readFloat64AtIndices(colIdx, indices);
                } catch {
                    result[`col${colIdx}`] = null;
                }
            }
        }

        return result;
    }

    /**
     * Count matching rows.
     * @returns {number}
     */
    count() {
        return this.collectIndices().length;
    }
}

/**
 * Represents a Lance file opened from a remote URL.
 * Uses HTTP Range requests to fetch data on demand.
 */
export class RemoteLanceFile {
    constructor(lanceql, url, fileSize, footerData) {
        this.lanceql = lanceql;
        this.wasm = lanceql.wasm;
        this.memory = lanceql.memory;
        this.url = url;
        this.fileSize = fileSize;

        // Store footer data in WASM memory
        const bytes = new Uint8Array(footerData);
        this.footerPtr = this.wasm.alloc(bytes.length);
        if (!this.footerPtr) {
            throw new Error('Failed to allocate memory for footer');
        }
        this.footerLen = bytes.length;
        new Uint8Array(this.memory.buffer).set(bytes, this.footerPtr);

        // Parse footer
        this._numColumns = this.wasm.parseFooterGetColumns(this.footerPtr, this.footerLen);
        this._majorVersion = this.wasm.parseFooterGetMajorVersion(this.footerPtr, this.footerLen);
        this._minorVersion = this.wasm.parseFooterGetMinorVersion(this.footerPtr, this.footerLen);
        this._columnMetaStart = this.wasm.getColumnMetaStart(this.footerPtr, this.footerLen);
        this._columnMetaOffsetsStart = this.wasm.getColumnMetaOffsetsStart(this.footerPtr, this.footerLen);
    }

    /**
     * Open a remote Lance file.
     * @param {LanceQL} lanceql
     * @param {string} url
     * @returns {Promise<RemoteLanceFile>}
     */
    static async open(lanceql, url) {
        // First, get file size with HEAD request
        const headResponse = await fetch(url, { method: 'HEAD' });
        if (!headResponse.ok) {
            throw new Error(`HTTP error: ${headResponse.status}`);
        }

        const contentLength = headResponse.headers.get('Content-Length');
        if (!contentLength) {
            throw new Error('Server did not return Content-Length');
        }
        const fileSize = parseInt(contentLength, 10);

        // Fetch footer (last 40 bytes)
        const footerSize = 40;
        const footerStart = fileSize - footerSize;
        const footerResponse = await fetch(url, {
            headers: {
                'Range': `bytes=${footerStart}-${fileSize - 1}`
            }
        });

        if (!footerResponse.ok && footerResponse.status !== 206) {
            throw new Error(`HTTP error: ${footerResponse.status}`);
        }

        const footerData = await footerResponse.arrayBuffer();

        // Verify magic bytes
        const footerBytes = new Uint8Array(footerData);
        const magic = String.fromCharCode(
            footerBytes[36], footerBytes[37], footerBytes[38], footerBytes[39]
        );
        if (magic !== 'LANC') {
            throw new Error(`Invalid Lance file: expected LANC magic, got "${magic}"`);
        }

        return new RemoteLanceFile(lanceql, url, fileSize, footerData);
    }

    /**
     * Fetch bytes from the remote file at a specific range.
     * @param {number} start - Start offset
     * @param {number} end - End offset (inclusive)
     * @returns {Promise<ArrayBuffer>}
     */
    async fetchRange(start, end) {
        const response = await fetch(this.url, {
            headers: {
                'Range': `bytes=${start}-${end}`
            }
        });

        if (!response.ok && response.status !== 206) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.arrayBuffer();

        // Track stats if callback available
        if (this._onFetch) {
            this._onFetch(data.byteLength, 1);
        }

        return data;
    }

    /**
     * Set callback for network stats tracking.
     * @param {function} callback - Function(bytesDownloaded, requestCount)
     */
    onFetch(callback) {
        this._onFetch = callback;
    }

    /**
     * Close the file and free memory.
     */
    close() {
        if (this.footerPtr) {
            this.wasm.free(this.footerPtr, this.footerLen);
            this.footerPtr = null;
        }
    }

    /**
     * Get the number of columns.
     * @returns {number}
     */
    get numColumns() {
        return this._numColumns;
    }

    /**
     * Get the file size.
     * @returns {number}
     */
    get size() {
        return this.fileSize;
    }

    /**
     * Get the version.
     * @returns {{major: number, minor: number}}
     */
    get version() {
        return {
            major: this._majorVersion,
            minor: this._minorVersion
        };
    }

    /**
     * Get the column metadata start offset.
     * @returns {number}
     */
    get columnMetaStart() {
        return Number(this._columnMetaStart);
    }

    /**
     * Get the column metadata offsets start.
     * @returns {number}
     */
    get columnMetaOffsetsStart() {
        return Number(this._columnMetaOffsetsStart);
    }

    /**
     * Get column offset entry from column metadata offsets.
     * @param {number} colIdx
     * @returns {Promise<{pos: number, len: number}>}
     */
    async getColumnOffsetEntry(colIdx) {
        if (colIdx >= this._numColumns) {
            return { pos: 0, len: 0 };
        }

        // Each entry is 16 bytes (8 bytes pos + 8 bytes len)
        const entryOffset = this.columnMetaOffsetsStart + colIdx * 16;
        const data = await this.fetchRange(entryOffset, entryOffset + 15);
        const view = new DataView(data);

        return {
            pos: Number(view.getBigUint64(0, true)),
            len: Number(view.getBigUint64(8, true))
        };
    }

    /**
     * Get debug info for a column (requires network request).
     * @param {number} colIdx
     * @returns {Promise<{offset: number, size: number, rows: number}>}
     */
    async getColumnDebugInfo(colIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) {
            return { offset: 0, size: 0, rows: 0 };
        }

        // Fetch column metadata
        const colMetaData = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const bytes = new Uint8Array(colMetaData);

        // Parse column metadata to get buffer info
        const info = this._parseColumnMeta(bytes);
        return info;
    }

    /**
     * Parse column metadata to extract buffer offsets and row count.
     * @private
     */
    _parseColumnMeta(bytes) {
        let pos = 0;
        let bufferOffset = 0;
        let bufferSize = 0;
        let rows = 0;

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < bytes.length) {
                const byte = bytes[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        while (pos < bytes.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (fieldNum === 2 && wireType === 2) {
                // pages field (length-delimited)
                const pageLen = readVarint();
                const pageEnd = pos + pageLen;

                // Parse page
                while (pos < pageEnd) {
                    const pageTag = readVarint();
                    const pageField = pageTag >> 3;
                    const pageWire = pageTag & 0x7;

                    if (pageField === 1 && pageWire === 2) {
                        // buffer_offsets (packed)
                        const packedLen = readVarint();
                        const packedEnd = pos + packedLen;
                        if (pos < packedEnd) {
                            bufferOffset = readVarint();
                        }
                        pos = packedEnd;
                    } else if (pageField === 2 && pageWire === 2) {
                        // buffer_sizes (packed)
                        const packedLen = readVarint();
                        const packedEnd = pos + packedLen;
                        if (pos < packedEnd) {
                            bufferSize = readVarint();
                        }
                        pos = packedEnd;
                    } else if (pageField === 3 && pageWire === 0) {
                        // length (rows)
                        rows = readVarint();
                    } else {
                        // Skip field
                        if (pageWire === 0) readVarint();
                        else if (pageWire === 2) pos += readVarint();
                        else if (pageWire === 5) pos += 4;
                        else if (pageWire === 1) pos += 8;
                    }
                }
                break;
            } else {
                // Skip field
                if (wireType === 0) readVarint();
                else if (wireType === 2) pos += readVarint();
                else if (wireType === 5) pos += 4;
                else if (wireType === 1) pos += 8;
            }
        }

        return { offset: bufferOffset, size: bufferSize, rows };
    }

    /**
     * Parse string column metadata to get offsets and data buffer info.
     * @private
     */
    _parseStringColumnMeta(bytes) {
        let pos = 0;
        let bufferOffsets = [0, 0];
        let bufferSizes = [0, 0];
        let rows = 0;

        const readVarint = () => {
            let result = 0;
            let shift = 0;
            while (pos < bytes.length) {
                const byte = bytes[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        };

        while (pos < bytes.length) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (fieldNum === 2 && wireType === 2) {
                // pages field
                const pageLen = readVarint();
                const pageEnd = pos + pageLen;

                while (pos < pageEnd) {
                    const pageTag = readVarint();
                    const pageField = pageTag >> 3;
                    const pageWire = pageTag & 0x7;

                    if (pageField === 1 && pageWire === 2) {
                        // buffer_offsets (packed)
                        const packedLen = readVarint();
                        const packedEnd = pos + packedLen;
                        let idx = 0;
                        while (pos < packedEnd && idx < 2) {
                            bufferOffsets[idx++] = readVarint();
                        }
                        pos = packedEnd;
                    } else if (pageField === 2 && pageWire === 2) {
                        // buffer_sizes (packed)
                        const packedLen = readVarint();
                        const packedEnd = pos + packedLen;
                        let idx = 0;
                        while (pos < packedEnd && idx < 2) {
                            bufferSizes[idx++] = readVarint();
                        }
                        pos = packedEnd;
                    } else if (pageField === 3 && pageWire === 0) {
                        rows = readVarint();
                    } else {
                        if (pageWire === 0) readVarint();
                        else if (pageWire === 2) pos += readVarint();
                        else if (pageWire === 5) pos += 4;
                        else if (pageWire === 1) pos += 8;
                    }
                }
                break;
            } else {
                if (wireType === 0) readVarint();
                else if (wireType === 2) pos += readVarint();
                else if (wireType === 5) pos += 4;
                else if (wireType === 1) pos += 8;
            }
        }

        return {
            offsetsStart: bufferOffsets[0],
            offsetsSize: bufferSizes[0],
            dataStart: bufferOffsets[1],
            dataSize: bufferSizes[1],
            rows
        };
    }

    /**
     * Read int64 values at specific row indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices - Row indices
     * @returns {Promise<BigInt64Array>}
     */
    async readInt64AtIndices(colIdx, indices) {
        if (indices.length === 0) return new BigInt64Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new BigInt64Array(indices.length);
        const view = new DataView(new ArrayBuffer(8));

        // Fetch each value (could optimize with batch fetching)
        for (let i = 0; i < indices.length; i++) {
            const offset = info.offset + indices[i] * 8;
            const data = await this.fetchRange(offset, offset + 7);
            results[i] = new DataView(data).getBigInt64(0, true);
        }

        return results;
    }

    /**
     * Read float64 values at specific row indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Float64Array>}
     */
    async readFloat64AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Float64Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Float64Array(indices.length);

        for (let i = 0; i < indices.length; i++) {
            const offset = info.offset + indices[i] * 8;
            const data = await this.fetchRange(offset, offset + 7);
            results[i] = new DataView(data).getFloat64(0, true);
        }

        return results;
    }

    /**
     * Read a single string at index via Range requests.
     * @param {number} colIdx
     * @param {number} rowIdx
     * @returns {Promise<string>}
     */
    async readStringAt(colIdx, rowIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseStringColumnMeta(new Uint8Array(colMeta));

        if (info.offsetsSize === 0 || info.dataSize === 0) return '';
        if (rowIdx >= info.rows) return '';

        // Determine offset size (4 or 8 bytes)
        const offsetSize = info.offsetsSize / (info.rows + 1);
        if (offsetSize !== 4 && offsetSize !== 8) return '';

        // Fetch the two offsets for this string
        const offsetStart = info.offsetsStart + rowIdx * offsetSize;
        const offsetData = await this.fetchRange(offsetStart, offsetStart + offsetSize * 2 - 1);
        const offsetView = new DataView(offsetData);

        let strStart, strEnd;
        if (offsetSize === 4) {
            strStart = offsetView.getUint32(0, true);
            strEnd = offsetView.getUint32(4, true);
        } else {
            strStart = Number(offsetView.getBigUint64(0, true));
            strEnd = Number(offsetView.getBigUint64(8, true));
        }

        if (strEnd <= strStart) return '';
        const strLen = strEnd - strStart;

        // Fetch the string data
        const strData = await this.fetchRange(
            info.dataStart + strStart,
            info.dataStart + strEnd - 1
        );

        return new TextDecoder().decode(strData);
    }

    /**
     * Read multiple strings at indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<string[]>}
     */
    async readStringsAtIndices(colIdx, indices) {
        const results = [];
        for (const idx of indices) {
            results.push(await this.readStringAt(colIdx, idx));
        }
        return results;
    }

    /**
     * Get row count for a column.
     * @param {number} colIdx
     * @returns {Promise<number>}
     */
    async getRowCount(colIdx) {
        const info = await this.getColumnDebugInfo(colIdx);
        return info.rows;
    }
}

// Default export for convenience
export default LanceQL;
