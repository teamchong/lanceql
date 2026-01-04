/**
 * Data Types and Lance File Writer
 *
 * Supports both JSON (fallback) and binary columnar format for performance.
 */

// Text encoder/decoder
export const E = new TextEncoder();
export const D = new TextDecoder();

export const DataType = {
    INT64: 'int64',
    INT32: 'int32',
    FLOAT64: 'float64',
    FLOAT32: 'float32',
    STRING: 'string',
    BOOL: 'bool',
    VECTOR: 'vector',
};

// Type codes for binary format
const TYPE_INT32 = 1;
const TYPE_INT64 = 2;
const TYPE_FLOAT32 = 3;
const TYPE_FLOAT64 = 4;
const TYPE_STRING = 5;
const TYPE_BOOL = 6;

/**
 * Get typed array constructor for a data type
 */
function getTypedArrayForType(dataType) {
    switch (dataType) {
        case 'int32':
        case 'integer':
            return Int32Array;
        case 'int64':
            return BigInt64Array;
        case 'float32':
        case 'real':
            return Float32Array;
        case 'float64':
        case 'double':
            return Float64Array;
        default:
            return null; // Use plain array for strings, etc.
    }
}

/**
 * Get type code for binary serialization
 */
function getTypeCode(dataType) {
    switch (dataType) {
        case 'int32':
        case 'integer':
            return TYPE_INT32;
        case 'int64':
            return TYPE_INT64;
        case 'float32':
        case 'real':
            return TYPE_FLOAT32;
        case 'float64':
        case 'double':
            return TYPE_FLOAT64;
        case 'string':
        case 'text':
            return TYPE_STRING;
        case 'bool':
        case 'boolean':
            return TYPE_BOOL;
        default:
            return TYPE_STRING;
    }
}

export class LanceFileWriter {
    constructor(schema) {
        this.schema = schema;
        this.columns = new Map();
        this.rowCount = 0;
        this._useBinary = true; // Use binary format by default
    }

    addRows(rows) {
        if (rows.length === 0) return;

        // Initialize columns on first add
        if (this.rowCount === 0) {
            for (const col of this.schema) {
                const TypedArray = getTypedArrayForType(col.dataType);
                if (TypedArray && TypedArray !== BigInt64Array) {
                    // Pre-allocate typed array (will grow if needed)
                    this.columns.set(col.name, {
                        type: 'typed',
                        dataType: col.dataType,
                        data: new TypedArray(Math.max(rows.length, 1024)),
                        length: 0
                    });
                } else {
                    // Plain array for strings and bigints
                    this.columns.set(col.name, {
                        type: 'array',
                        dataType: col.dataType,
                        data: [],
                        length: 0
                    });
                }
            }
        }

        // Add rows in batch
        for (const col of this.schema) {
            const column = this.columns.get(col.name);

            if (column.type === 'typed') {
                // Grow typed array if needed
                const newLength = column.length + rows.length;
                if (newLength > column.data.length) {
                    const newSize = Math.max(newLength, column.data.length * 2);
                    const newData = new column.data.constructor(newSize);
                    newData.set(column.data);
                    column.data = newData;
                }

                // Batch copy values
                for (let i = 0; i < rows.length; i++) {
                    const val = rows[i][col.name];
                    column.data[column.length + i] = val ?? 0;
                }
                column.length += rows.length;
            } else {
                // Plain array
                for (let i = 0; i < rows.length; i++) {
                    column.data.push(rows[i][col.name] ?? null);
                }
                column.length += rows.length;
            }
        }

        this.rowCount += rows.length;
    }

    /**
     * Build binary columnar format
     * Format: [magic(4)] [version(4)] [numCols(4)] [rowCount(4)]
     *         [colName1Len(4)] [colName1] [colType(1)] [colDataLen(4)] [colData]
     *         ...
     */
    buildBinary() {
        const chunks = [];
        let totalSize = 16; // Header: magic + version + numCols + rowCount

        // Calculate total size and prepare column data
        const columnChunks = [];
        for (const col of this.schema) {
            const column = this.columns.get(col.name);
            const nameBytes = E.encode(col.name);
            const typeCode = getTypeCode(col.dataType);

            let dataBytes;
            if (column.type === 'typed') {
                // Trim typed array to actual length
                const trimmed = column.data.subarray(0, column.length);
                dataBytes = new Uint8Array(trimmed.buffer, trimmed.byteOffset, trimmed.byteLength);
            } else {
                // Serialize strings/mixed as JSON
                dataBytes = E.encode(JSON.stringify(column.data));
            }

            columnChunks.push({ nameBytes, typeCode, dataBytes, dataType: col.dataType });
            totalSize += 4 + nameBytes.length + 1 + 4 + dataBytes.length;
        }

        // Build binary buffer
        const buffer = new ArrayBuffer(totalSize);
        const view = new DataView(buffer);
        const bytes = new Uint8Array(buffer);
        let offset = 0;

        // Header
        view.setUint32(offset, 0x4C414E43, false); // 'LANC' magic
        offset += 4;
        view.setUint32(offset, 2, false); // Version 2 (binary)
        offset += 4;
        view.setUint32(offset, this.schema.length, false);
        offset += 4;
        view.setUint32(offset, this.rowCount, false);
        offset += 4;

        // Columns
        for (const chunk of columnChunks) {
            // Column name
            view.setUint32(offset, chunk.nameBytes.length, false);
            offset += 4;
            bytes.set(chunk.nameBytes, offset);
            offset += chunk.nameBytes.length;

            // Type code
            view.setUint8(offset, chunk.typeCode);
            offset += 1;

            // Data
            view.setUint32(offset, chunk.dataBytes.length, false);
            offset += 4;
            bytes.set(chunk.dataBytes, offset);
            offset += chunk.dataBytes.length;
        }

        return new Uint8Array(buffer);
    }

    build() {
        if (this._useBinary && this.rowCount > 0) {
            try {
                return this.buildBinary();
            } catch (e) {
                console.warn('[LanceFileWriter] Binary build failed, falling back to JSON:', e);
            }
        }

        // JSON columnar format (fallback)
        const data = {
            format: 'json',
            schema: this.schema,
            columns: {},
            rowCount: this.rowCount,
        };

        for (const [name, column] of this.columns) {
            if (column.type === 'typed') {
                // Convert typed array to plain array for JSON
                data.columns[name] = Array.from(column.data.subarray(0, column.length));
            } else {
                data.columns[name] = column.data;
            }
        }

        return E.encode(JSON.stringify(data));
    }
}

/**
 * Parse binary columnar format
 */
export function parseBinaryColumnar(buffer) {
    const view = new DataView(buffer.buffer || buffer);
    const bytes = new Uint8Array(buffer.buffer || buffer);
    let offset = 0;

    // Check magic
    const magic = view.getUint32(offset, false);
    if (magic !== 0x4C414E43) {
        return null; // Not binary format
    }
    offset += 4;

    const version = view.getUint32(offset, false);
    offset += 4;
    const numCols = view.getUint32(offset, false);
    offset += 4;
    const rowCount = view.getUint32(offset, false);
    offset += 4;

    const schema = [];
    const columns = {};

    for (let i = 0; i < numCols; i++) {
        // Column name
        const nameLen = view.getUint32(offset, false);
        offset += 4;
        const name = D.decode(bytes.subarray(offset, offset + nameLen));
        offset += nameLen;

        // Type code
        const typeCode = view.getUint8(offset);
        offset += 1;

        // Data
        const dataLen = view.getUint32(offset, false);
        offset += 4;
        const dataBytes = bytes.subarray(offset, offset + dataLen);
        offset += dataLen;

        // Parse based on type
        let data, dataType;
        switch (typeCode) {
            case TYPE_INT32:
                dataType = 'int32';
                data = new Int32Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 4);
                break;
            case TYPE_FLOAT32:
                dataType = 'float32';
                data = new Float32Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 4);
                break;
            case TYPE_FLOAT64:
                dataType = 'float64';
                data = new Float64Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 8);
                break;
            case TYPE_STRING:
            case TYPE_BOOL:
            default:
                dataType = typeCode === TYPE_BOOL ? 'bool' : 'string';
                data = JSON.parse(D.decode(dataBytes));
                break;
        }

        schema.push({ name, dataType });
        columns[name] = data;
    }

    return { schema, columns, rowCount, format: 'binary' };
}
