/**
 * Lance Writer - Protobuf encoding and Lance file format writer
 */

import { LANCE_FOOTER_SIZE, LANCE_MAGIC } from '../lance/constants.js';

class ProtobufEncoder {
    constructor() {
        this.chunks = [];
    }

    /**
     * Encode a varint (variable-length integer)
     * @param {number|bigint} value
     * @returns {Uint8Array}
     */
    static encodeVarint(value) {
        const bytes = [];
        let v = typeof value === 'bigint' ? value : BigInt(value);
        while (v > 0x7fn) {
            bytes.push(Number(v & 0x7fn) | 0x80);
            v >>= 7n;
        }
        bytes.push(Number(v));
        return new Uint8Array(bytes);
    }

    /**
     * Encode a field header (tag)
     * @param {number} fieldNum - Field number
     * @param {number} wireType - Wire type (0=varint, 2=length-delimited)
     * @returns {Uint8Array}
     */
    static encodeFieldHeader(fieldNum, wireType) {
        const tag = (fieldNum << 3) | wireType;
        return ProtobufEncoder.encodeVarint(tag);
    }

    /**
     * Encode a varint field
     * @param {number} fieldNum
     * @param {number|bigint} value
     */
    writeVarint(fieldNum, value) {
        this.chunks.push(ProtobufEncoder.encodeFieldHeader(fieldNum, 0));
        this.chunks.push(ProtobufEncoder.encodeVarint(value));
    }

    /**
     * Encode a length-delimited field (bytes or nested message)
     * @param {number} fieldNum
     * @param {Uint8Array} data
     */
    writeBytes(fieldNum, data) {
        this.chunks.push(ProtobufEncoder.encodeFieldHeader(fieldNum, 2));
        this.chunks.push(ProtobufEncoder.encodeVarint(data.length));
        this.chunks.push(data);
    }

    /**
     * Encode packed repeated uint64 as varints
     * @param {number} fieldNum
     * @param {BigUint64Array|number[]} values
     */
    writePackedUint64(fieldNum, values) {
        // First encode all varints
        const varintChunks = [];
        for (const v of values) {
            varintChunks.push(ProtobufEncoder.encodeVarint(v));
        }
        // Calculate total length
        const totalLen = varintChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        // Write field header + length + data
        this.chunks.push(ProtobufEncoder.encodeFieldHeader(fieldNum, 2));
        this.chunks.push(ProtobufEncoder.encodeVarint(totalLen));
        for (const chunk of varintChunks) {
            this.chunks.push(chunk);
        }
    }

    /**
     * Get the encoded bytes
     * @returns {Uint8Array}
     */
    toBytes() {
        const totalLen = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const result = new Uint8Array(totalLen);
        let offset = 0;
        for (const chunk of this.chunks) {
            result.set(chunk, offset);
            offset += chunk.length;
        }
        return result;
    }

    /**
     * Clear the encoder for reuse
     */
    clear() {
        this.chunks = [];
    }
}

// =============================================================================
// LanceFileWriter - Create Lance files in pure JavaScript
// =============================================================================

/**
 * Lance column types
 */
const LanceColumnType = {
    INT64: 'int64',
    FLOAT64: 'float64',
    STRING: 'string',
    BOOL: 'bool',
    INT32: 'int32',
    FLOAT32: 'float32',
};

/**
 * Pure JavaScript Lance File Writer - Creates Lance files without WASM.
 * Use this when WASM is not available or for simple file creation.
 * Supports basic column types: int64, float64, string, bool.
 *
 * @example
 * const writer = new PureLanceWriter();
 * writer.addInt64Column('id', BigInt64Array.from([1n, 2n, 3n]));
 * writer.addFloat64Column('score', new Float64Array([0.5, 0.8, 0.3]));
 * writer.addStringColumn('name', ['Alice', 'Bob', 'Charlie']);
 * const lanceData = writer.finalize();
 * await opfsStorage.save('mydata.lance', lanceData);
 */
class PureLanceWriter {
    /**
     * @param {Object} options
     * @param {number} [options.majorVersion=0] - Lance format major version
     * @param {number} [options.minorVersion=3] - Lance format minor version (3 = v2.0)
     */
    constructor(options = {}) {
        this.majorVersion = options.majorVersion ?? 0;
        this.minorVersion = options.minorVersion ?? 3; // v2.0
        this.columns = []; // { name, type, data, metadata }
        this.rowCount = null;
    }

    /**
     * Validate row count consistency
     * @param {number} count
     */
    _validateRowCount(count) {
        if (this.rowCount === null) {
            this.rowCount = count;
        } else if (this.rowCount !== count) {
            throw new Error(`Row count mismatch: expected ${this.rowCount}, got ${count}`);
        }
    }

    /**
     * Add an int64 column
     * @param {string} name - Column name
     * @param {BigInt64Array} values - Column values
     */
    addInt64Column(name, values) {
        this._validateRowCount(values.length);
        this.columns.push({
            name,
            type: LanceColumnType.INT64,
            data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
            length: values.length,
        });
    }

    /**
     * Add an int32 column
     * @param {string} name - Column name
     * @param {Int32Array} values - Column values
     */
    addInt32Column(name, values) {
        this._validateRowCount(values.length);
        this.columns.push({
            name,
            type: LanceColumnType.INT32,
            data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
            length: values.length,
        });
    }

    /**
     * Add a float64 column
     * @param {string} name - Column name
     * @param {Float64Array} values - Column values
     */
    addFloat64Column(name, values) {
        this._validateRowCount(values.length);
        this.columns.push({
            name,
            type: LanceColumnType.FLOAT64,
            data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
            length: values.length,
        });
    }

    /**
     * Add a float32 column
     * @param {string} name - Column name
     * @param {Float32Array} values - Column values
     */
    addFloat32Column(name, values) {
        this._validateRowCount(values.length);
        this.columns.push({
            name,
            type: LanceColumnType.FLOAT32,
            data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
            length: values.length,
        });
    }

    /**
     * Add a boolean column
     * @param {string} name - Column name
     * @param {boolean[]} values - Column values
     */
    addBoolColumn(name, values) {
        this._validateRowCount(values.length);
        // Pack booleans as bytes (1 byte per bool for simplicity)
        const data = new Uint8Array(values.length);
        for (let i = 0; i < values.length; i++) {
            data[i] = values[i] ? 1 : 0;
        }
        this.columns.push({
            name,
            type: LanceColumnType.BOOL,
            data,
            length: values.length,
        });
    }

    /**
     * Add a string column
     * @param {string} name - Column name
     * @param {string[]} values - Column values
     */
    addStringColumn(name, values) {
        this._validateRowCount(values.length);

        const encoder = new TextEncoder();

        // Build offsets and data
        // Lance strings use i32 offsets followed by UTF-8 data
        const offsets = new Int32Array(values.length + 1);
        const dataChunks = [];
        let currentOffset = 0;

        for (let i = 0; i < values.length; i++) {
            offsets[i] = currentOffset;
            const encoded = encoder.encode(values[i]);
            dataChunks.push(encoded);
            currentOffset += encoded.length;
        }
        offsets[values.length] = currentOffset;

        // Combine offsets and data
        const offsetsBytes = new Uint8Array(offsets.buffer);
        const totalDataLen = dataChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const stringData = new Uint8Array(totalDataLen);
        let writePos = 0;
        for (const chunk of dataChunks) {
            stringData.set(chunk, writePos);
            writePos += chunk.length;
        }

        // Store both parts
        this.columns.push({
            name,
            type: LanceColumnType.STRING,
            offsetsData: offsetsBytes,
            stringData,
            data: null, // Will be combined in finalize
            length: values.length,
        });
    }

    /**
     * Build column metadata protobuf for a column
     * @param {number} bufferOffset - Offset to column data
     * @param {number} bufferSize - Size of column data
     * @param {number} length - Number of rows
     * @param {string} type - Column type
     * @returns {Uint8Array}
     */
    _buildColumnMeta(bufferOffset, bufferSize, length, type) {
        // Build Page message
        const pageEncoder = new ProtobufEncoder();
        pageEncoder.writePackedUint64(1, [BigInt(bufferOffset)]); // buffer_offsets
        pageEncoder.writePackedUint64(2, [BigInt(bufferSize)]); // buffer_sizes
        pageEncoder.writeVarint(3, length); // length
        // Skip encoding (field 4) - use default
        pageEncoder.writeVarint(5, 0); // priority
        const pageBytes = pageEncoder.toBytes();

        // Build ColumnMetadata message
        const metaEncoder = new ProtobufEncoder();
        // Skip encoding (field 1) - use default plain encoding
        metaEncoder.writeBytes(2, pageBytes); // pages (repeated)
        // Skip buffer_offsets and buffer_sizes at column level

        return metaEncoder.toBytes();
    }

    /**
     * Build column metadata for string column (2 buffers: offsets + data)
     * @param {number} offsetsBufOffset - Offset to offsets buffer
     * @param {number} offsetsBufSize - Size of offsets buffer
     * @param {number} dataBufOffset - Offset to string data buffer
     * @param {number} dataBufSize - Size of string data buffer
     * @param {number} length - Number of rows
     * @returns {Uint8Array}
     */
    _buildStringColumnMeta(offsetsBufOffset, offsetsBufSize, dataBufOffset, dataBufSize, length) {
        // Build Page message with 2 buffers
        const pageEncoder = new ProtobufEncoder();
        pageEncoder.writePackedUint64(1, [BigInt(offsetsBufOffset), BigInt(dataBufOffset)]); // buffer_offsets
        pageEncoder.writePackedUint64(2, [BigInt(offsetsBufSize), BigInt(dataBufSize)]); // buffer_sizes
        pageEncoder.writeVarint(3, length); // length
        pageEncoder.writeVarint(5, 0); // priority
        const pageBytes = pageEncoder.toBytes();

        // Build ColumnMetadata message
        const metaEncoder = new ProtobufEncoder();
        metaEncoder.writeBytes(2, pageBytes); // pages (repeated)

        return metaEncoder.toBytes();
    }

    /**
     * Finalize and create the Lance file
     * @returns {Uint8Array} Complete Lance file data
     */
    finalize() {
        if (this.columns.length === 0) {
            throw new Error('No columns added');
        }

        const chunks = [];
        let currentOffset = 0;

        // 1. Write column data buffers
        const columnBufferInfos = []; // { offset, size } for each column

        for (const col of this.columns) {
            if (col.type === LanceColumnType.STRING) {
                // String columns have 2 buffers: offsets + data
                const offsetsOffset = currentOffset;
                chunks.push(col.offsetsData);
                currentOffset += col.offsetsData.length;

                const dataOffset = currentOffset;
                chunks.push(col.stringData);
                currentOffset += col.stringData.length;

                columnBufferInfos.push({
                    type: 'string',
                    offsetsOffset,
                    offsetsSize: col.offsetsData.length,
                    dataOffset,
                    dataSize: col.stringData.length,
                    length: col.length,
                });
            } else {
                // Simple column with single buffer
                const bufferOffset = currentOffset;
                chunks.push(col.data);
                currentOffset += col.data.length;

                columnBufferInfos.push({
                    type: col.type,
                    offset: bufferOffset,
                    size: col.data.length,
                    length: col.length,
                });
            }
        }

        // 2. Build column metadata
        const columnMetadatas = [];
        for (let i = 0; i < this.columns.length; i++) {
            const info = columnBufferInfos[i];
            let meta;
            if (info.type === 'string') {
                meta = this._buildStringColumnMeta(
                    info.offsetsOffset, info.offsetsSize,
                    info.dataOffset, info.dataSize,
                    info.length
                );
            } else {
                meta = this._buildColumnMeta(info.offset, info.size, info.length, info.type);
            }
            columnMetadatas.push(meta);
        }

        // 3. Write column metadata section
        const columnMetaStart = currentOffset;
        const columnMetaOffsets = [];
        let metaOffset = 0;
        for (const meta of columnMetadatas) {
            columnMetaOffsets.push(metaOffset);
            chunks.push(meta);
            currentOffset += meta.length;
            metaOffset += meta.length;
        }

        // 4. Write column metadata offset table
        const columnMetaOffsetsStart = currentOffset;
        const offsetTable = new BigUint64Array(columnMetaOffsets.length);
        for (let i = 0; i < columnMetaOffsets.length; i++) {
            offsetTable[i] = BigInt(columnMetaOffsets[i]);
        }
        const offsetTableBytes = new Uint8Array(offsetTable.buffer);
        chunks.push(offsetTableBytes);
        currentOffset += offsetTableBytes.length;

        // 5. Write global buffer offsets (empty for now)
        const globalBuffOffsetsStart = currentOffset;
        const numGlobalBuffers = 0;

        // 6. Write footer (40 bytes)
        const footer = new ArrayBuffer(LANCE_FOOTER_SIZE);
        const footerView = new DataView(footer);

        footerView.setBigUint64(0, BigInt(columnMetaStart), true);           // column_meta_start
        footerView.setBigUint64(8, BigInt(columnMetaOffsetsStart), true);    // column_meta_offsets_start
        footerView.setBigUint64(16, BigInt(globalBuffOffsetsStart), true);   // global_buff_offsets_start
        footerView.setUint32(24, numGlobalBuffers, true);                    // num_global_buffers
        footerView.setUint32(28, this.columns.length, true);                 // num_columns
        footerView.setUint16(32, this.majorVersion, true);                   // major_version
        footerView.setUint16(34, this.minorVersion, true);                   // minor_version
        // Magic "LANC"
        new Uint8Array(footer, 36, 4).set(LANCE_MAGIC);

        chunks.push(new Uint8Array(footer));

        // 7. Combine all chunks
        const totalSize = currentOffset + LANCE_FOOTER_SIZE;
        const result = new Uint8Array(totalSize);
        let writeOffset = 0;
        for (const chunk of chunks) {
            result.set(chunk, writeOffset);
            writeOffset += chunk.length;
        }

        return result;
    }

    /**
     * Get the number of columns
     * @returns {number}
     */
    getNumColumns() {
        return this.columns.length;
    }

    /**
     * Get the row count
     * @returns {number|null}
     */
    getRowCount() {
        return this.rowCount;
    }

    /**
     * Get column names
     * @returns {string[]}
     */
    getColumnNames() {
        return this.columns.map(c => c.name);
    }
}

// Legacy IndexedDB + OPFS storage (deprecated, use OPFSStorage instead)

export { ProtobufEncoder, PureLanceWriter };
