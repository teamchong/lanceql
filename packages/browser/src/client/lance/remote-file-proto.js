/**
 * RemoteLanceFile - Protobuf column metadata parsing
 * Extracted from remote-file.js for modularity
 */

/**
 * Parse column metadata to extract buffer offsets and row count.
 * For nullable columns, there are typically 2 buffers:
 * - Buffer 0: null bitmap
 * - Buffer 1: actual data values
 * @param {Uint8Array} bytes - Column metadata bytes
 * @returns {Object} - Parsed metadata info
 */
export function parseColumnMeta(bytes) {
    let pos = 0;
    const pages = [];
    let totalRows = 0;

    // Read varint as BigInt to handle large values (>2GB offsets)
    const readVarint = () => {
        let result = 0n;
        let shift = 0n;
        while (pos < bytes.length) {
            const byte = bytes[pos++];
            result |= BigInt(byte & 0x7F) << shift;
            if ((byte & 0x80) === 0) break;
            shift += 7n;
        }
        return Number(result);
    };

    while (pos < bytes.length) {
        const tag = readVarint();
        const fieldNum = tag >> 3;
        const wireType = tag & 0x7;

        if (fieldNum === 2 && wireType === 2) {
            // pages field (length-delimited) - parse ALL pages
            const pageLen = readVarint();
            const pageEnd = pos + pageLen;

            const pageOffsets = [];
            const pageSizes = [];
            let pageRows = 0;

            // Parse page
            while (pos < pageEnd) {
                const pageTag = readVarint();
                const pageField = pageTag >> 3;
                const pageWire = pageTag & 0x7;

                if (pageField === 1 && pageWire === 2) {
                    // buffer_offsets (packed)
                    const packedLen = readVarint();
                    const packedEnd = pos + packedLen;
                    while (pos < packedEnd) {
                        pageOffsets.push(readVarint());
                    }
                } else if (pageField === 2 && pageWire === 2) {
                    // buffer_sizes (packed)
                    const packedLen = readVarint();
                    const packedEnd = pos + packedLen;
                    while (pos < packedEnd) {
                        pageSizes.push(readVarint());
                    }
                } else if (pageField === 3 && pageWire === 0) {
                    // length (rows)
                    pageRows = readVarint();
                } else {
                    // Skip field
                    if (pageWire === 0) readVarint();
                    else if (pageWire === 2) {
                        const skipLen = readVarint();
                        pos += skipLen;
                    }
                    else if (pageWire === 5) pos += 4;
                    else if (pageWire === 1) pos += 8;
                }
            }

            pages.push({
                offsets: pageOffsets,
                sizes: pageSizes,
                rows: pageRows
            });
            totalRows += pageRows;
        } else {
            // Skip field
            if (wireType === 0) readVarint();
            else if (wireType === 2) {
                const skipLen = readVarint();
                pos += skipLen;
            }
            else if (wireType === 5) pos += 4;
            else if (wireType === 1) pos += 8;
        }
    }

    // Combine all pages - use first page for offset/size (for backward compat)
    const firstPage = pages[0] || { offsets: [], sizes: [], rows: 0 };
    const bufferOffsets = firstPage.offsets;
    const bufferSizes = firstPage.sizes;

    // For multi-page columns, compute total size
    let totalSize = 0;
    for (const page of pages) {
        const dataIdx = page.sizes.length > 1 ? 1 : 0;
        totalSize += page.sizes[dataIdx] || 0;
    }

    // For nullable columns: buffer 0 = null bitmap, buffer 1 = data
    // For non-nullable: buffer 0 = data
    const dataBufferIdx = bufferOffsets.length > 1 ? 1 : 0;
    const nullBitmapIdx = bufferOffsets.length > 1 ? 0 : -1;

    return {
        offset: bufferOffsets[dataBufferIdx] || 0,
        size: pages.length > 1 ? totalSize : (bufferSizes[dataBufferIdx] || 0),
        rows: totalRows,
        nullBitmapOffset: nullBitmapIdx >= 0 ? bufferOffsets[nullBitmapIdx] : null,
        nullBitmapSize: nullBitmapIdx >= 0 ? bufferSizes[nullBitmapIdx] : null,
        bufferOffsets,
        bufferSizes,
        pages
    };
}

/**
 * Parse string column metadata to get offsets and data buffer info.
 * @param {Uint8Array} bytes - Column metadata bytes
 * @returns {Object} - String column metadata
 */
export function parseStringColumnMeta(bytes) {
    const pages = [];
    let pos = 0;

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
            // pages field - parse this page
            const pageLen = readVarint();
            const pageEnd = pos + pageLen;

            let bufferOffsets = [0, 0];
            let bufferSizes = [0, 0];
            let rows = 0;

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
                } else if (pageField === 4 && pageWire === 2) {
                    // encoding field - skip it
                    const skipLen = readVarint();
                    pos += skipLen;
                } else {
                    // Unknown field - skip based on wire type
                    if (pageWire === 0) readVarint();
                    else if (pageWire === 2) {
                        const skipLen = readVarint();
                        pos += skipLen;
                    }
                    else if (pageWire === 5) pos += 4;
                    else if (pageWire === 1) pos += 8;
                }
            }

            pages.push({
                offsetsStart: bufferOffsets[0],
                offsetsSize: bufferSizes[0],
                dataStart: bufferOffsets[1],
                dataSize: bufferSizes[1],
                rows
            });
        } else {
            // Skip unknown fields
            if (wireType === 0) {
                readVarint();
            } else if (wireType === 2) {
                const skipLen = readVarint();
                pos += skipLen;
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        }
    }

    // Return first page for backwards compatibility, but also include all pages
    const firstPage = pages[0] || { offsetsStart: 0, offsetsSize: 0, dataStart: 0, dataSize: 0, rows: 0 };
    return {
        ...firstPage,
        pages
    };
}

/**
 * Batch indices into contiguous ranges to minimize HTTP requests.
 * Groups nearby indices if the gap is smaller than gapThreshold.
 * @param {number[]} indices - Row indices to batch
 * @param {number} valueSize - Size of each value in bytes
 * @param {number} gapThreshold - Max gap to merge (default 1024)
 * @returns {Array} - Batched index groups
 */
export function batchIndices(indices, valueSize, gapThreshold = 1024) {
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
