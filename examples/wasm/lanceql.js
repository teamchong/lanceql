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
    // Additional Numeric Type Column Methods
    // ========================================================================

    /**
     * Read an int32 column as an Int32Array.
     * @param {number} colIdx - Column index
     * @returns {Int32Array}
     */
    readInt32Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Int32Array(0);

        const bufPtr = this.wasm.allocInt32Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate int32 buffer');

        try {
            const count = this.wasm.readInt32Column(colIdx, bufPtr, rowCount);
            const result = new Int32Array(count);
            const view = new Int32Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 4);
        }
    }

    /**
     * Read an int16 column as an Int16Array.
     * @param {number} colIdx - Column index
     * @returns {Int16Array}
     */
    readInt16Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Int16Array(0);

        const bufPtr = this.wasm.allocInt16Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate int16 buffer');

        try {
            const count = this.wasm.readInt16Column(colIdx, bufPtr, rowCount);
            const result = new Int16Array(count);
            const view = new Int16Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 2);
        }
    }

    /**
     * Read an int8 column as an Int8Array.
     * @param {number} colIdx - Column index
     * @returns {Int8Array}
     */
    readInt8Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Int8Array(0);

        const bufPtr = this.wasm.allocInt8Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate int8 buffer');

        try {
            const count = this.wasm.readInt8Column(colIdx, bufPtr, rowCount);
            const result = new Int8Array(count);
            const view = new Int8Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount);
        }
    }

    /**
     * Read a uint64 column as a BigUint64Array.
     * @param {number} colIdx - Column index
     * @returns {BigUint64Array}
     */
    readUint64Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new BigUint64Array(0);

        const bufPtr = this.wasm.allocUint64Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate uint64 buffer');

        try {
            const count = this.wasm.readUint64Column(colIdx, bufPtr, rowCount);
            const result = new BigUint64Array(count);
            const view = new BigUint64Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 8);
        }
    }

    /**
     * Read a uint32 column as a Uint32Array.
     * @param {number} colIdx - Column index
     * @returns {Uint32Array}
     */
    readUint32Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint32Array(0);

        const bufPtr = this.wasm.allocIndexBuffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate uint32 buffer');

        try {
            const count = this.wasm.readUint32Column(colIdx, bufPtr, rowCount);
            const result = new Uint32Array(count);
            const view = new Uint32Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 4);
        }
    }

    /**
     * Read a uint16 column as a Uint16Array.
     * @param {number} colIdx - Column index
     * @returns {Uint16Array}
     */
    readUint16Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint16Array(0);

        const bufPtr = this.wasm.allocUint16Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate uint16 buffer');

        try {
            const count = this.wasm.readUint16Column(colIdx, bufPtr, rowCount);
            const result = new Uint16Array(count);
            const view = new Uint16Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 2);
        }
    }

    /**
     * Read a uint8 column as a Uint8Array.
     * @param {number} colIdx - Column index
     * @returns {Uint8Array}
     */
    readUint8Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint8Array(0);

        const bufPtr = this.wasm.allocStringBuffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate uint8 buffer');

        try {
            const count = this.wasm.readUint8Column(colIdx, bufPtr, rowCount);
            const result = new Uint8Array(count);
            const view = new Uint8Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount);
        }
    }

    /**
     * Read a float32 column as a Float32Array.
     * @param {number} colIdx - Column index
     * @returns {Float32Array}
     */
    readFloat32Column(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Float32Array(0);

        const bufPtr = this.wasm.allocFloat32Buffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate float32 buffer');

        try {
            const count = this.wasm.readFloat32Column(colIdx, bufPtr, rowCount);
            const result = new Float32Array(count);
            const view = new Float32Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount * 4);
        }
    }

    /**
     * Read a boolean column as a Uint8Array (0 or 1 values).
     * @param {number} colIdx - Column index
     * @returns {Uint8Array}
     */
    readBoolColumn(colIdx) {
        const rowCount = Number(this.getRowCount(colIdx));
        if (rowCount === 0) return new Uint8Array(0);

        const bufPtr = this.wasm.allocStringBuffer(rowCount);
        if (!bufPtr) throw new Error('Failed to allocate bool buffer');

        try {
            const count = this.wasm.readBoolColumn(colIdx, bufPtr, rowCount);
            const result = new Uint8Array(count);
            const view = new Uint8Array(this.memory.buffer, bufPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, rowCount);
        }
    }

    /**
     * Read int32 values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {Int32Array}
     */
    readInt32AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Int32Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocInt32Buffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
            const count = this.wasm.readInt32AtIndices(colIdx, idxPtr, indices.length, outPtr);
            const result = new Int32Array(count);
            const view = new Int32Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(outPtr, indices.length * 4);
        }
    }

    /**
     * Read float32 values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {Float32Array}
     */
    readFloat32AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Float32Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocFloat32Buffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
            const count = this.wasm.readFloat32AtIndices(colIdx, idxPtr, indices.length, outPtr);
            const result = new Float32Array(count);
            const view = new Float32Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(outPtr, indices.length * 4);
        }
    }

    /**
     * Read uint8 values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {Uint8Array}
     */
    readUint8AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Uint8Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocStringBuffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
            const count = this.wasm.readUint8AtIndices(colIdx, idxPtr, indices.length, outPtr);
            const result = new Uint8Array(count);
            const view = new Uint8Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(outPtr, indices.length);
        }
    }

    /**
     * Read bool values at specific row indices.
     * @param {number} colIdx - Column index
     * @param {Uint32Array} indices - Row indices to read
     * @returns {Uint8Array}
     */
    readBoolAtIndices(colIdx, indices) {
        if (indices.length === 0) return new Uint8Array(0);

        const idxPtr = this.wasm.allocIndexBuffer(indices.length);
        if (!idxPtr) throw new Error('Failed to allocate index buffer');

        const outPtr = this.wasm.allocStringBuffer(indices.length);
        if (!outPtr) {
            this.wasm.free(idxPtr, indices.length * 4);
            throw new Error('Failed to allocate output buffer');
        }

        try {
            new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
            const count = this.wasm.readBoolAtIndices(colIdx, idxPtr, indices.length, outPtr);
            const result = new Uint8Array(count);
            const view = new Uint8Array(this.memory.buffer, outPtr, count);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(idxPtr, indices.length * 4);
            this.wasm.free(outPtr, indices.length);
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
    // Vector Column Support (for embeddings/semantic search)
    // ========================================================================

    /**
     * Get vector info for a column.
     * @param {number} colIdx - Column index
     * @returns {{rows: number, dimension: number}}
     */
    getVectorInfo(colIdx) {
        const packed = this.wasm.getVectorInfo(colIdx);
        return {
            rows: Number(BigInt(packed) >> 32n),
            dimension: Number(BigInt(packed) & 0xFFFFFFFFn)
        };
    }

    /**
     * Read a single vector at index.
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Row index
     * @returns {Float32Array}
     */
    readVectorAt(colIdx, rowIdx) {
        const info = this.getVectorInfo(colIdx);
        if (info.dimension === 0) return new Float32Array(0);

        const bufPtr = this.wasm.allocFloat32Buffer(info.dimension);
        if (!bufPtr) throw new Error('Failed to allocate vector buffer');

        try {
            const dim = this.wasm.readVectorAt(colIdx, rowIdx, bufPtr, info.dimension);
            const result = new Float32Array(dim);
            const view = new Float32Array(this.memory.buffer, bufPtr, dim);
            result.set(view);
            return result;
        } finally {
            this.wasm.free(bufPtr, info.dimension * 4);
        }
    }

    /**
     * Compute cosine similarity between two vectors.
     * @param {Float32Array} vecA
     * @param {Float32Array} vecB
     * @returns {number} Similarity score (-1 to 1)
     */
    cosineSimilarity(vecA, vecB) {
        if (vecA.length !== vecB.length) {
            throw new Error('Vector dimensions must match');
        }

        const ptrA = this.wasm.allocFloat32Buffer(vecA.length);
        const ptrB = this.wasm.allocFloat32Buffer(vecB.length);
        if (!ptrA || !ptrB) throw new Error('Failed to allocate buffers');

        try {
            new Float32Array(this.memory.buffer, ptrA, vecA.length).set(vecA);
            new Float32Array(this.memory.buffer, ptrB, vecB.length).set(vecB);
            return this.wasm.cosineSimilarity(ptrA, ptrB, vecA.length);
        } finally {
            this.wasm.free(ptrA, vecA.length * 4);
            this.wasm.free(ptrB, vecB.length * 4);
        }
    }

    /**
     * Find top-k most similar vectors to query.
     * @param {number} colIdx - Column index with vectors
     * @param {Float32Array} queryVec - Query vector
     * @param {number} topK - Number of results to return
     * @returns {{indices: Uint32Array, scores: Float32Array}}
     */
    vectorSearch(colIdx, queryVec, topK = 10) {
        const queryPtr = this.wasm.allocFloat32Buffer(queryVec.length);
        const indicesPtr = this.wasm.allocIndexBuffer(topK);
        const scoresPtr = this.wasm.allocFloat32Buffer(topK);

        if (!queryPtr || !indicesPtr || !scoresPtr) {
            throw new Error('Failed to allocate buffers');
        }

        try {
            new Float32Array(this.memory.buffer, queryPtr, queryVec.length).set(queryVec);

            const count = this.wasm.vectorSearchTopK(
                colIdx, queryPtr, queryVec.length, topK, indicesPtr, scoresPtr
            );

            const indices = new Uint32Array(count);
            const scores = new Float32Array(count);

            indices.set(new Uint32Array(this.memory.buffer, indicesPtr, count));
            scores.set(new Float32Array(this.memory.buffer, scoresPtr, count));

            return { indices, scores };
        } finally {
            this.wasm.free(queryPtr, queryVec.length * 4);
            this.wasm.free(indicesPtr, topK * 4);
            this.wasm.free(scoresPtr, topK * 4);
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

        // Cache for column metadata to avoid repeated fetches
        this._columnMetaCache = new Map();
        this._columnOffsetCache = new Map();
        this._columnTypes = null;
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
     * Uses caching to avoid repeated fetches.
     * @param {number} colIdx
     * @returns {Promise<{pos: number, len: number}>}
     */
    async getColumnOffsetEntry(colIdx) {
        if (colIdx >= this._numColumns) {
            return { pos: 0, len: 0 };
        }

        // Check cache first
        if (this._columnOffsetCache.has(colIdx)) {
            return this._columnOffsetCache.get(colIdx);
        }

        // Each entry is 16 bytes (8 bytes pos + 8 bytes len)
        const entryOffset = this.columnMetaOffsetsStart + colIdx * 16;
        const data = await this.fetchRange(entryOffset, entryOffset + 15);
        const view = new DataView(data);

        const entry = {
            pos: Number(view.getBigUint64(0, true)),
            len: Number(view.getBigUint64(8, true))
        };

        // Cache the result
        this._columnOffsetCache.set(colIdx, entry);
        return entry;
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
     * Batch indices into contiguous ranges to minimize HTTP requests.
     * Groups nearby indices if the gap is smaller than gapThreshold.
     * @private
     */
    _batchIndices(indices, valueSize, gapThreshold = 1024) {
        if (indices.length === 0) return [];

        // Sort indices for contiguous access
        const sorted = [...indices].map((v, i) => ({ idx: v, origPos: i }));
        sorted.sort((a, b) => a.idx - b.idx);

        const batches = [];
        let batchStart = 0;

        for (let i = 1; i <= sorted.length; i++) {
            // Check if we should end the current batch
            const endBatch = i === sorted.length ||
                (sorted[i].idx - sorted[i-1].idx) * valueSize > gapThreshold;

            if (endBatch) {
                batches.push({
                    startIdx: sorted[batchStart].idx,
                    endIdx: sorted[i-1].idx,
                    items: sorted.slice(batchStart, i)
                });
                batchStart = i;
            }
        }

        return batches;
    }

    /**
     * Read int64 values at specific row indices via Range requests.
     * Uses batched fetching to minimize HTTP requests.
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
        const valueSize = 8;

        // Batch indices into contiguous ranges
        const batches = this._batchIndices(indices, valueSize);

        // Fetch each batch in parallel
        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            // Extract values from batch
            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * valueSize;
                results[item.origPos] = view.getBigInt64(localOffset, true);
            }
        }));

        return results;
    }

    /**
     * Read float64 values at specific row indices via Range requests.
     * Uses batched fetching to minimize HTTP requests.
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
        const valueSize = 8;

        // Batch indices into contiguous ranges
        const batches = this._batchIndices(indices, valueSize);

        // Fetch each batch in parallel
        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            // Extract values from batch
            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * valueSize;
                results[item.origPos] = view.getFloat64(localOffset, true);
            }
        }));

        return results;
    }

    /**
     * Read int32 values at specific row indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Int32Array>}
     */
    async readInt32AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Int32Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Int32Array(indices.length);
        const valueSize = 4;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * valueSize;
                results[item.origPos] = view.getInt32(localOffset, true);
            }
        }));

        return results;
    }

    /**
     * Read float32 values at specific row indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Float32Array>}
     */
    async readFloat32AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Float32Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Float32Array(indices.length);
        const valueSize = 4;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * valueSize;
                results[item.origPos] = view.getFloat32(localOffset, true);
            }
        }));

        return results;
    }

    /**
     * Read uint8 values at specific row indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Uint8Array>}
     */
    async readUint8AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Uint8Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Uint8Array(indices.length);
        const valueSize = 1;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const bytes = new Uint8Array(data);

            for (const item of batch.items) {
                const localOffset = item.idx - batch.startIdx;
                results[item.origPos] = bytes[localOffset];
            }
        }));

        return results;
    }

    /**
     * Read bool values at specific row indices via Range requests.
     * Boolean values are bit-packed (8 values per byte).
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Uint8Array>}
     */
    async readBoolAtIndices(colIdx, indices) {
        if (indices.length === 0) return new Uint8Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Uint8Array(indices.length);

        // Calculate byte ranges needed for bit-packed booleans
        const byteIndices = indices.map(i => Math.floor(i / 8));
        const uniqueBytes = [...new Set(byteIndices)].sort((a, b) => a - b);

        if (uniqueBytes.length === 0) return results;

        // Fetch the byte range
        const startByte = uniqueBytes[0];
        const endByte = uniqueBytes[uniqueBytes.length - 1];
        const startOffset = info.offset + startByte;
        const endOffset = info.offset + endByte;
        const data = await this.fetchRange(startOffset, endOffset);
        const bytes = new Uint8Array(data);

        // Extract boolean values
        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            const byteIdx = Math.floor(idx / 8);
            const bitIdx = idx % 8;
            const localByteIdx = byteIdx - startByte;
            if (localByteIdx >= 0 && localByteIdx < bytes.length) {
                results[i] = (bytes[localByteIdx] >> bitIdx) & 1;
            }
        }

        return results;
    }

    /**
     * Read a single string at index via Range requests.
     * @param {number} colIdx
     * @param {number} rowIdx
     * @returns {Promise<string>}
     * @throws {Error} If the column is not a string column
     */
    async readStringAt(colIdx, rowIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseStringColumnMeta(new Uint8Array(colMeta));

        // Check if this is actually a string column
        // String columns have: offsetsSize / rows = 4 or 8 bytes per offset
        // Numeric columns with validity bitmap have: offsetsSize = rows / 8 (bitmap)
        if (info.offsetsSize === 0 || info.dataSize === 0) {
            throw new Error('Not a string column');
        }

        // Calculate bytes per offset - strings have rows offsets of 4 or 8 bytes each
        const bytesPerOffset = info.offsetsSize / info.rows;

        // If bytesPerOffset is not 4 or 8, this is not a string column
        // (e.g., it's a validity bitmap which has rows/8 bytes = 0.125 bytes per row)
        if (bytesPerOffset !== 4 && bytesPerOffset !== 8) {
            throw new Error(`Not a string column - bytesPerOffset=${bytesPerOffset}, expected 4 or 8`);
        }

        if (rowIdx >= info.rows) return '';

        // Determine offset size (4 or 8 bytes)
        const offsetSize = bytesPerOffset;

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
     * Uses batched fetching to minimize HTTP requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<string[]>}
     */
    async readStringsAtIndices(colIdx, indices) {
        if (indices.length === 0) return [];

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseStringColumnMeta(new Uint8Array(colMeta));

        if (info.offsetsSize === 0 || info.dataSize === 0) {
            return indices.map(() => '');
        }

        // Determine offset size (4 or 8 bytes)
        const offsetSize = info.offsetsSize / info.rows;
        if (offsetSize !== 4 && offsetSize !== 8) {
            return indices.map(() => '');
        }

        const results = new Array(indices.length).fill('');

        // Batch offset fetching - each string needs 2 consecutive offsets
        const offsetBatches = this._batchIndices(indices, offsetSize, 2048);

        // First pass: fetch all offsets in batches
        const stringRanges = new Array(indices.length);

        await Promise.all(offsetBatches.map(async (batch) => {
            // Fetch offsets for entire batch range (need +1 for end offset)
            const startOffset = info.offsetsStart + batch.startIdx * offsetSize;
            const endOffset = info.offsetsStart + (batch.endIdx + 2) * offsetSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            // Extract string ranges from offsets
            for (const item of batch.items) {
                if (item.idx >= info.rows) continue;

                const localIdx = item.idx - batch.startIdx;
                let strStart, strEnd;

                if (offsetSize === 4) {
                    strStart = view.getUint32(localIdx * 4, true);
                    strEnd = view.getUint32((localIdx + 1) * 4, true);
                } else {
                    strStart = Number(view.getBigUint64(localIdx * 8, true));
                    strEnd = Number(view.getBigUint64((localIdx + 1) * 8, true));
                }

                if (strEnd > strStart) {
                    stringRanges[item.origPos] = { start: strStart, end: strEnd };
                }
            }
        }));

        // Second pass: batch fetch string data
        // Group strings that are close together in the data buffer
        const dataItems = stringRanges
            .map((range, origPos) => range ? { ...range, origPos } : null)
            .filter(Boolean)
            .sort((a, b) => a.start - b.start);

        if (dataItems.length === 0) return results;

        // Batch string data fetches
        const dataBatches = [];
        let batchStart = 0;

        for (let i = 1; i <= dataItems.length; i++) {
            const endBatch = i === dataItems.length ||
                (dataItems[i].start - dataItems[i-1].end) > 4096;

            if (endBatch) {
                dataBatches.push({
                    rangeStart: dataItems[batchStart].start,
                    rangeEnd: dataItems[i-1].end,
                    items: dataItems.slice(batchStart, i)
                });
                batchStart = i;
            }
        }

        // Fetch string data batches in parallel
        await Promise.all(dataBatches.map(async (batch) => {
            const data = await this.fetchRange(
                info.dataStart + batch.rangeStart,
                info.dataStart + batch.rangeEnd - 1
            );
            const bytes = new Uint8Array(data);

            for (const item of batch.items) {
                const localStart = item.start - batch.rangeStart;
                const len = item.end - item.start;
                const strBytes = bytes.slice(localStart, localStart + len);
                results[item.origPos] = new TextDecoder().decode(strBytes);
            }
        }));

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

    /**
     * Detect column types by sampling first row.
     * Returns array of type strings: 'string', 'int64', 'float64', 'float32', 'vector', 'unknown'
     * @returns {Promise<string[]>}
     */
    async detectColumnTypes() {
        // Return cached if available
        if (this._columnTypes) {
            return this._columnTypes;
        }

        const types = [];

        for (let c = 0; c < this._numColumns; c++) {
            let type = 'unknown';

            // Try string first - if we can read a valid string, it's a string column
            try {
                const str = await this.readStringAt(c, 0);
                // readStringAt throws for non-string columns, returns string for valid string columns
                type = 'string';
                types.push(type);
                continue;
            } catch (e) {
                // Not a string column, continue to numeric detection
            }

            // Check numeric column by examining bytes per row
            try {
                const entry = await this.getColumnOffsetEntry(c);
                if (entry.len > 0) {
                    const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
                    const bytes = new Uint8Array(colMeta);
                    const info = this._parseColumnMeta(bytes);

                    if (info.rows > 0 && info.size > 0) {
                        const bytesPerRow = info.size / info.rows;

                        if (bytesPerRow === 8) {
                            // int64 or float64
                            try {
                                const data = await this.readInt64AtIndices(c, [0]);
                                if (data.length > 0) {
                                    const val = data[0];
                                    // Heuristic: if value looks like reasonable int
                                    if (val >= -1e15 && val <= 1e15) {
                                        type = 'int64';
                                    } else {
                                        type = 'float64';
                                    }
                                }
                            } catch (e) {
                                type = 'float64';
                            }
                        } else if (bytesPerRow === 4) {
                            type = 'float32';
                        } else if (bytesPerRow > 8 && bytesPerRow % 4 === 0) {
                            type = 'vector';
                        } else if (bytesPerRow === 2) {
                            type = 'int16';
                        } else if (bytesPerRow === 1) {
                            type = 'int8';
                        }
                    }
                }
            } catch (e) {
                console.warn(`Failed to detect type for column ${c}:`, e);
            }

            types.push(type);
        }

        this._columnTypes = types;
        return types;
    }

    /**
     * Get cached column metadata, fetching if necessary.
     * @private
     */
    async _getCachedColumnMeta(colIdx) {
        if (this._columnMetaCache.has(colIdx)) {
            return this._columnMetaCache.get(colIdx);
        }

        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) {
            return null;
        }

        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const bytes = new Uint8Array(colMeta);

        this._columnMetaCache.set(colIdx, bytes);
        return bytes;
    }

    // ========================================================================
    // Vector Column Support (for embeddings/semantic search via Range requests)
    // ========================================================================

    /**
     * Get vector info for a column via Range requests.
     * @param {number} colIdx - Column index
     * @returns {Promise<{rows: number, dimension: number}>}
     */
    async getVectorInfo(colIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) return { rows: 0, dimension: 0 };

        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        if (info.rows === 0 || info.size === 0) return { rows: 0, dimension: 0 };

        // Dimension = buffer_size / (rows * 4) for float32
        const dimension = Math.floor(info.size / (info.rows * 4));

        return { rows: info.rows, dimension };
    }

    /**
     * Read a single vector at index via Range requests.
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Row index
     * @returns {Promise<Float32Array>}
     */
    async readVectorAt(colIdx, rowIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        if (info.rows === 0) return new Float32Array(0);
        if (rowIdx >= info.rows) return new Float32Array(0);

        const dim = Math.floor(info.size / (info.rows * 4));
        if (dim === 0) return new Float32Array(0);

        // Fetch the vector data
        const vecStart = info.offset + rowIdx * dim * 4;
        const vecEnd = vecStart + dim * 4 - 1;
        const data = await this.fetchRange(vecStart, vecEnd);

        return new Float32Array(data);
    }

    /**
     * Read multiple vectors at indices via Range requests.
     * Uses batched fetching for efficiency.
     * @param {number} colIdx - Column index
     * @param {number[]} indices - Row indices
     * @returns {Promise<Float32Array[]>}
     */
    async readVectorsAtIndices(colIdx, indices) {
        if (indices.length === 0) return [];

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        if (info.rows === 0) return indices.map(() => new Float32Array(0));

        const dim = Math.floor(info.size / (info.rows * 4));
        if (dim === 0) return indices.map(() => new Float32Array(0));

        const vecSize = dim * 4;
        const results = new Array(indices.length);

        // Batch indices for efficient fetching
        const batches = this._batchIndices(indices, vecSize, vecSize * 10);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * vecSize;
            const endOffset = info.offset + (batch.endIdx + 1) * vecSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);

            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * vecSize;
                results[item.origPos] = new Float32Array(
                    data.slice(localOffset, localOffset + vecSize)
                );
            }
        }));

        return results;
    }

    /**
     * Compute cosine similarity between two vectors (in JS).
     * @param {Float32Array} vecA
     * @param {Float32Array} vecB
     * @returns {number}
     */
    cosineSimilarity(vecA, vecB) {
        if (vecA.length !== vecB.length) return 0;

        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < vecA.length; i++) {
            dot += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }

        const denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom === 0 ? 0 : dot / denom;
    }

    /**
     * Find top-k most similar vectors to query via Range requests.
     * NOTE: This requires scanning the entire vector column which can be slow
     * for large datasets. For production, use an index.
     *
     * @param {number} colIdx - Column index with vectors
     * @param {Float32Array} queryVec - Query vector
     * @param {number} topK - Number of results to return
     * @param {function} onProgress - Progress callback(current, total)
     * @returns {Promise<{indices: number[], scores: number[]}>}
     */
    async vectorSearch(colIdx, queryVec, topK = 10, onProgress = null) {
        const info = await this.getVectorInfo(colIdx);
        if (info.dimension === 0 || info.dimension !== queryVec.length) {
            throw new Error(`Dimension mismatch: query=${queryVec.length}, column=${info.dimension}`);
        }

        const dim = info.dimension;
        const vecSize = dim * 4;
        const numRows = info.rows;

        // Batch size for fetching (fetch multiple vectors at once)
        const batchSize = Math.min(100, numRows);

        // Top-k heap (min-heap by score)
        const topResults = [];

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const metaInfo = this._parseColumnMeta(new Uint8Array(colMeta));

        for (let batchStart = 0; batchStart < numRows; batchStart += batchSize) {
            const batchEnd = Math.min(batchStart + batchSize, numRows);

            if (onProgress) {
                onProgress(batchStart, numRows);
            }

            // Fetch batch of vectors
            const startOffset = metaInfo.offset + batchStart * vecSize;
            const endOffset = metaInfo.offset + batchEnd * vecSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);

            // Compute similarities for this batch
            for (let i = 0; i < batchEnd - batchStart; i++) {
                const rowIdx = batchStart + i;
                const vecData = new Float32Array(data.slice(i * vecSize, (i + 1) * vecSize));

                // Compute cosine similarity
                let dot = 0, normA = 0, normB = 0;
                for (let j = 0; j < dim; j++) {
                    dot += queryVec[j] * vecData[j];
                    normA += queryVec[j] * queryVec[j];
                    normB += vecData[j] * vecData[j];
                }
                const denom = Math.sqrt(normA) * Math.sqrt(normB);
                const score = denom === 0 ? 0 : dot / denom;

                // Insert into top-k
                if (topResults.length < topK) {
                    topResults.push({ idx: rowIdx, score });
                    topResults.sort((a, b) => b.score - a.score);
                } else if (score > topResults[topK - 1].score) {
                    topResults[topK - 1] = { idx: rowIdx, score };
                    topResults.sort((a, b) => b.score - a.score);
                }
            }
        }

        if (onProgress) {
            onProgress(numRows, numRows);
        }

        return {
            indices: topResults.map(r => r.idx),
            scores: topResults.map(r => r.score)
        };
    }
}

// Default export for convenience
export default LanceQL;
