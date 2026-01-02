/**
 * RemoteLanceFile - Numeric column readers
 * Extracted from remote-file.js for modularity
 */

import { batchIndices } from './remote-file-proto.js';

/**
 * Read int64 values at specific row indices via Range requests.
 * Uses batched fetching to minimize HTTP requests.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Row indices
 * @returns {Promise<BigInt64Array>}
 */
export async function readInt64AtIndices(file, colIdx, indices) {
    if (indices.length === 0) return new BigInt64Array(0);

    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = file._parseColumnMeta(new Uint8Array(colMeta));

    const results = new BigInt64Array(indices.length);
    const valueSize = 8;

    const batches = batchIndices(indices, valueSize);

    await Promise.all(batches.map(async (batch) => {
        const startOffset = info.offset + batch.startIdx * valueSize;
        const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
        const data = await file.fetchRange(startOffset, endOffset);
        const view = new DataView(data);

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
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Row indices
 * @returns {Promise<Float64Array>}
 */
export async function readFloat64AtIndices(file, colIdx, indices) {
    if (indices.length === 0) return new Float64Array(0);

    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = file._parseColumnMeta(new Uint8Array(colMeta));

    const results = new Float64Array(indices.length);
    const valueSize = 8;

    const batches = batchIndices(indices, valueSize);

    await Promise.all(batches.map(async (batch) => {
        const startOffset = info.offset + batch.startIdx * valueSize;
        const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
        const data = await file.fetchRange(startOffset, endOffset);
        const view = new DataView(data);

        for (const item of batch.items) {
            const localOffset = (item.idx - batch.startIdx) * valueSize;
            results[item.origPos] = view.getFloat64(localOffset, true);
        }
    }));

    return results;
}

/**
 * Read int32 values at specific row indices via Range requests.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Row indices
 * @returns {Promise<Int32Array>}
 */
export async function readInt32AtIndices(file, colIdx, indices) {
    if (indices.length === 0) return new Int32Array(0);

    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = file._parseColumnMeta(new Uint8Array(colMeta));

    const results = new Int32Array(indices.length);
    const valueSize = 4;

    const batches = batchIndices(indices, valueSize);

    await Promise.all(batches.map(async (batch) => {
        const startOffset = info.offset + batch.startIdx * valueSize;
        const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
        const data = await file.fetchRange(startOffset, endOffset);
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
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Row indices
 * @returns {Promise<Float32Array>}
 */
export async function readFloat32AtIndices(file, colIdx, indices) {
    if (indices.length === 0) return new Float32Array(0);

    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = file._parseColumnMeta(new Uint8Array(colMeta));

    const results = new Float32Array(indices.length);
    const valueSize = 4;

    const batches = batchIndices(indices, valueSize);

    await Promise.all(batches.map(async (batch) => {
        const startOffset = info.offset + batch.startIdx * valueSize;
        const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
        const data = await file.fetchRange(startOffset, endOffset);
        const view = new DataView(data);

        for (const item of batch.items) {
            const localOffset = (item.idx - batch.startIdx) * valueSize;
            results[item.origPos] = view.getFloat32(localOffset, true);
        }
    }));

    return results;
}

/**
 * Read int16 values at specific row indices via Range requests.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Row indices
 * @returns {Promise<Int16Array>}
 */
export async function readInt16AtIndices(file, colIdx, indices) {
    if (indices.length === 0) return new Int16Array(0);

    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = file._parseColumnMeta(new Uint8Array(colMeta));

    const results = new Int16Array(indices.length);
    const valueSize = 2;

    const batches = batchIndices(indices, valueSize);

    await Promise.all(batches.map(async (batch) => {
        const startOffset = info.offset + batch.startIdx * valueSize;
        const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
        const data = await file.fetchRange(startOffset, endOffset);
        const view = new DataView(data);

        for (const item of batch.items) {
            const localOffset = (item.idx - batch.startIdx) * valueSize;
            results[item.origPos] = view.getInt16(localOffset, true);
        }
    }));

    return results;
}

/**
 * Read uint8 values at specific row indices via Range requests.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Row indices
 * @returns {Promise<Uint8Array>}
 */
export async function readUint8AtIndices(file, colIdx, indices) {
    if (indices.length === 0) return new Uint8Array(0);

    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = file._parseColumnMeta(new Uint8Array(colMeta));

    const results = new Uint8Array(indices.length);
    const valueSize = 1;

    const batches = batchIndices(indices, valueSize);

    await Promise.all(batches.map(async (batch) => {
        const startOffset = info.offset + batch.startIdx * valueSize;
        const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
        const data = await file.fetchRange(startOffset, endOffset);
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
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Row indices
 * @returns {Promise<Uint8Array>}
 */
export async function readBoolAtIndices(file, colIdx, indices) {
    if (indices.length === 0) return new Uint8Array(0);

    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = file._parseColumnMeta(new Uint8Array(colMeta));

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
    const data = await file.fetchRange(startOffset, endOffset);
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
