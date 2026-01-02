/**
 * LanceFile - In-memory Lance file API
 */

class LanceFile {
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
     * Debug: Get string column buffer info
     * @param {number} colIdx - Column index
     * @returns {{offsetsSize: number, dataSize: number}}
     */
    debugStringColInfo(colIdx) {
        const packed = this.wasm.debugStringColInfo(colIdx);
        return {
            offsetsSize: Number(BigInt(packed) >> 32n),
            dataSize: Number(BigInt(packed) & 0xFFFFFFFFn)
        };
    }

    /**
     * Debug: Get string read info for a specific row
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Row index
     * @returns {{strStart: number, strLen: number} | {error: string}}
     */
    debugReadStringInfo(colIdx, rowIdx) {
        const packed = this.wasm.debugReadStringInfo(colIdx, rowIdx);
        // Check for error codes (0xDEAD00XX)
        if ((packed & 0xFFFF0000n) === 0xDEAD0000n) {
            const errCode = Number(packed & 0xFFFFn);
            const errors = {
                1: 'No file data',
                2: 'No column entry',
                3: 'Col meta out of bounds',
                4: 'Not a string column',
                5: 'Row out of bounds',
                6: 'Invalid offset size'
            };
            return { error: errors[errCode] || `Unknown error ${errCode}` };
        }
        return {
            strStart: Number(BigInt(packed) >> 32n),
            strLen: Number(BigInt(packed) & 0xFFFFFFFFn)
        };
    }

    /**
     * Debug: Get data_start position for string column
     * @param {number} colIdx - Column index
     * @returns {{dataStart: number, fileLen: number}}
     */
    debugStringDataStart(colIdx) {
        const packed = this.wasm.debugStringDataStart(colIdx);
        return {
            dataStart: Number(BigInt(packed) >> 32n),
            fileLen: Number(BigInt(packed) & 0xFFFFFFFFn)
        };
    }

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

        // Use smaller buffer - estimate based on indices count
        // Assume average string is ~256 bytes, capped at 256KB to avoid WASM memory issues
        const maxTotalLen = Math.min(indices.length * 256, 256 * 1024);
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
     * Batch cosine similarity using WASM SIMD.
     * Much faster than calling cosineSimilarity in a loop.
     * @param {Float32Array} queryVec - Query vector
     * @param {Float32Array[]} vectors - Array of vectors to compare
     * @param {boolean} normalized - Whether vectors are L2-normalized
     * @returns {Float32Array} - Similarity scores
     */
    batchCosineSimilarity(queryVec, vectors, normalized = true) {
        if (vectors.length === 0) return new Float32Array(0);

        const dim = queryVec.length;
        const numVectors = vectors.length;

        // Allocate WASM buffers
        const queryPtr = this.wasm.allocFloat32Buffer(dim);
        const vectorsPtr = this.wasm.allocFloat32Buffer(numVectors * dim);
        const scoresPtr = this.wasm.allocFloat32Buffer(numVectors);

        if (!queryPtr || !vectorsPtr || !scoresPtr) {
            throw new Error('Failed to allocate WASM buffers');
        }

        try {
            // Copy query vector
            new Float32Array(this.memory.buffer, queryPtr, dim).set(queryVec);

            // Copy all vectors (flattened)
            const flatVectors = new Float32Array(this.memory.buffer, vectorsPtr, numVectors * dim);
            for (let i = 0; i < numVectors; i++) {
                flatVectors.set(vectors[i], i * dim);
            }

            // Call WASM batch similarity
            this.wasm.batchCosineSimilarity(queryPtr, vectorsPtr, dim, numVectors, scoresPtr, normalized ? 1 : 0);

            // Copy results
            const scores = new Float32Array(numVectors);
            scores.set(new Float32Array(this.memory.buffer, scoresPtr, numVectors));
            return scores;
        } finally {
            this.wasm.free(queryPtr, dim * 4);
            this.wasm.free(vectorsPtr, numVectors * dim * 4);
            this.wasm.free(scoresPtr, numVectors * 4);
        }
    }

    /**
     * Read all vectors from a column as array of Float32Arrays.
     * @param {number} colIdx - Column index
     * @returns {Float32Array[]} Array of vectors
     */
    readAllVectors(colIdx) {
        const info = this.getVectorInfo(colIdx);
        if (info.dimension === 0 || info.rows === 0) return [];

        const dim = info.dimension;
        const numRows = info.rows;
        const vectors = [];

        // Allocate buffer for all vectors at once
        const bufPtr = this.wasm.allocFloat32Buffer(numRows * dim);
        if (!bufPtr) throw new Error('Failed to allocate vector buffer');

        try {
            // Read all vectors in one WASM call (if supported)
            // Otherwise fall back to individual reads
            if (this.wasm.readVectorColumn) {
                const count = this.wasm.readVectorColumn(colIdx, bufPtr, numRows * dim);
                const allData = new Float32Array(this.memory.buffer, bufPtr, count);

                for (let i = 0; i < numRows && i * dim < count; i++) {
                    const vec = new Float32Array(dim);
                    vec.set(allData.subarray(i * dim, (i + 1) * dim));
                    vectors.push(vec);
                }
            } else {
                // Fall back to individual reads
                for (let i = 0; i < numRows; i++) {
                    vectors.push(this.readVectorAt(colIdx, i));
                }
            }

            return vectors;
        } finally {
            this.wasm.free(bufPtr, numRows * dim * 4);
        }
    }

    /**
     * Find top-k most similar vectors to query.
     * Uses WebGPU if available, otherwise falls back to WASM SIMD.
     * GPU-accelerated top-K selection for large result sets.
     * @param {number} colIdx - Column index with vectors
     * @param {Float32Array} queryVec - Query vector
     * @param {number} topK - Number of results to return
     * @param {Function} onProgress - Progress callback (current, total)
     * @returns {Promise<{indices: Uint32Array, scores: Float32Array}>}
     */
    async vectorSearch(colIdx, queryVec, topK = 10, onProgress = null) {
        const dim = queryVec.length;
        const info = this.getVectorInfo(colIdx);
        const numRows = info.rows;

        // Try WebGPU-accelerated path first
        if (webgpuAccelerator.isAvailable()) {
            if (onProgress) onProgress(0, numRows);

            // Read all vectors (bulk read)
            console.log(`[LanceFile.vectorSearch] Reading ${numRows} vectors...`);
            const allVectors = this.readAllVectors(colIdx);

            if (onProgress) onProgress(numRows, numRows);

            console.log(`[LanceFile.vectorSearch] Computing similarity for ${allVectors.length} vectors via WebGPU`);

            // Batch compute with WebGPU
            const scores = await webgpuAccelerator.batchCosineSimilarity(queryVec, allVectors, true);

            // GPU-accelerated top-K selection for large result sets
            return await gpuVectorSearch.topK(scores, null, topK, true);
        }

        // Fall back to WASM SIMD (uses batchCosineSimilarity internally)
        console.log(`[LanceFile.vectorSearch] Using WASM SIMD`);

        if (onProgress) onProgress(0, numRows);

        // Read all vectors first
        const allVectors = this.readAllVectors(colIdx);

        if (onProgress) onProgress(numRows, numRows);

        // Use WASM batch cosine similarity
        const scores = this.lanceql.batchCosineSimilarity(queryVec, allVectors, true);

        // GPU-accelerated top-K selection for large result sets
        return await gpuVectorSearch.topK(scores, null, topK, true);
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

export { LanceFile };
