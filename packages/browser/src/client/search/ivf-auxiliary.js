/**
 * IVF Auxiliary Metadata Parsing
 * Handles loading and parsing auxiliary.idx metadata
 */

/**
 * Load partition metadata from auxiliary.idx.
 * Uses HTTP range request to fetch only the metadata section.
 * @param {IVFIndex} index - The IVF index instance
 */
export async function loadAuxiliaryMetadata(index) {
    // Fetch file size first
    let headResp;
    try {
        headResp = await fetch(index.auxiliaryUrl, { method: 'HEAD' });
    } catch (e) {
        console.warn('[IVFIndex] HEAD request failed for auxiliary.idx:', e.message);
        return;
    }
    if (!headResp.ok) return;

    const fileSize = parseInt(headResp.headers.get('content-length'));
    if (!fileSize) return;

    // Fetch footer (last 40 bytes) to get metadata locations
    const footerResp = await fetch(index.auxiliaryUrl, {
        headers: { 'Range': `bytes=${fileSize - 40}-${fileSize - 1}` }
    });
    if (!footerResp.ok) return;

    const footer = new Uint8Array(await footerResp.arrayBuffer());
    const view = new DataView(footer.buffer, footer.byteOffset);

    // Parse Lance footer (40 bytes)
    const colMetaStart = Number(view.getBigUint64(0, true));
    const colMetaOffsetsStart = Number(view.getBigUint64(8, true));
    const globalBuffOffsetsStart = Number(view.getBigUint64(16, true));
    const numGlobalBuffers = view.getUint32(24, true);
    const numColumns = view.getUint32(28, true);
    const magic = new TextDecoder().decode(footer.slice(36, 40));

    if (magic !== 'LANC') {
        console.warn('[IVFIndex] Invalid auxiliary.idx magic');
        return;
    }

    console.log(`[IVFIndex] Footer: colMetaStart=${colMetaStart}, colMetaOffsetsStart=${colMetaOffsetsStart}, globalBuffOffsetsStart=${globalBuffOffsetsStart}, numGlobalBuffers=${numGlobalBuffers}, numColumns=${numColumns}`);

    // Fetch global buffer offsets (each buffer has offset + length = 16 bytes)
    const gboSize = numGlobalBuffers * 16;
    const gboResp = await fetch(index.auxiliaryUrl, {
        headers: { 'Range': `bytes=${globalBuffOffsetsStart}-${globalBuffOffsetsStart + gboSize - 1}` }
    });
    if (!gboResp.ok) return;

    const gboData = new Uint8Array(await gboResp.arrayBuffer());
    const gboView = new DataView(gboData.buffer, gboData.byteOffset);

    // Global buffer offsets are stored as [offset, length] pairs
    const buffers = [];
    for (let i = 0; i < numGlobalBuffers; i++) {
        const offset = Number(gboView.getBigUint64(i * 16, true));
        const length = Number(gboView.getBigUint64(i * 16 + 8, true));
        buffers.push({ offset, length });
    }

    console.log(`[IVFIndex] Buffers:`, buffers);

    if (buffers.length < 2) return;

    // Store buffer info for later use
    index._auxBuffers = buffers;
    index._auxFileSize = fileSize;

    // Parse column metadata to get partition boundaries
    const colMetaOffResp = await fetch(index.auxiliaryUrl, {
        headers: { 'Range': `bytes=${colMetaOffsetsStart}-${globalBuffOffsetsStart - 1}` }
    });
    if (!colMetaOffResp.ok) return;

    const colMetaOffData = new Uint8Array(await colMetaOffResp.arrayBuffer());
    if (colMetaOffData.length >= 32) {
        const colView = new DataView(colMetaOffData.buffer, colMetaOffData.byteOffset);
        const col0Pos = Number(colView.getBigUint64(0, true));
        const col0Len = Number(colView.getBigUint64(8, true));
        console.log(`[IVFIndex] Column 0 (_rowid) metadata at ${col0Pos}, len=${col0Len}`);

        // Fetch column 0 metadata to get page info
        const col0MetaResp = await fetch(index.auxiliaryUrl, {
            headers: { 'Range': `bytes=${col0Pos}-${col0Pos + col0Len - 1}` }
        });
        if (col0MetaResp.ok) {
            const col0Meta = new Uint8Array(await col0MetaResp.arrayBuffer());
            parseColumnMetaForPartitions(index, col0Meta);
        }
    }
}

/**
 * Parse column metadata to extract partition (page) boundaries.
 * @param {IVFIndex} index - The IVF index instance
 * @param {Uint8Array} bytes - Column metadata bytes
 */
export function parseColumnMetaForPartitions(index, bytes) {
    let pos = 0;
    const pages = [];

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

    // Parse protobuf to find pages
    while (pos < bytes.length) {
        const tag = readVarint();
        const fieldNum = tag >> 3;
        const wireType = tag & 0x7;

        if (wireType === 2) {
            const len = readVarint();
            if (len > bytes.length - pos) break;
            const content = bytes.slice(pos, pos + len);
            pos += len;

            // Field 2 = pages (PageInfo)
            if (fieldNum === 2) {
                const page = parsePageInfo(content);
                if (page) pages.push(page);
            }
        } else if (wireType === 0) {
            readVarint();
        } else if (wireType === 5) {
            pos += 4;
        } else if (wireType === 1) {
            pos += 8;
        }
    }

    console.log(`[IVFIndex] Found ${pages.length} column pages`);

    // Store page info for row ID lookups
    index._columnPages = pages;

    // Calculate total rows for verification
    let totalRows = 0;
    for (const page of pages) {
        totalRows += page.numRows;
    }
    console.log(`[IVFIndex] Column has ${totalRows} total rows`);
}

/**
 * Parse PageInfo protobuf.
 * @param {Uint8Array} bytes - PageInfo bytes
 * @returns {Object|null}
 */
export function parsePageInfo(bytes) {
    let pos = 0;
    let numRows = 0;
    const bufferOffsets = [];
    const bufferSizes = [];

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

        if (wireType === 0) {
            const val = readVarint();
            if (fieldNum === 3) numRows = val;  // length field
        } else if (wireType === 2) {
            const len = readVarint();
            const content = bytes.slice(pos, pos + len);
            pos += len;

            // Field 1 = buffer_offsets (packed uint64)
            if (fieldNum === 1) {
                let p = 0;
                while (p < content.length) {
                    let val = 0n;
                    let shift = 0n;
                    while (p < content.length) {
                        const b = content[p++];
                        val |= BigInt(b & 0x7F) << shift;
                        if ((b & 0x80) === 0) break;
                        shift += 7n;
                    }
                    bufferOffsets.push(Number(val));
                }
            }
            // Field 2 = buffer_sizes (packed uint64)
            if (fieldNum === 2) {
                let p = 0;
                while (p < content.length) {
                    let val = 0n;
                    let shift = 0n;
                    while (p < content.length) {
                        const b = content[p++];
                        val |= BigInt(b & 0x7F) << shift;
                        if ((b & 0x80) === 0) break;
                        shift += 7n;
                    }
                    bufferSizes.push(Number(val));
                }
            }
        } else if (wireType === 5) {
            pos += 4;
        } else if (wireType === 1) {
            pos += 8;
        }
    }

    return { numRows, bufferOffsets, bufferSizes };
}

/**
 * Parse partition offsets and lengths from auxiliary.idx metadata.
 * @param {IVFIndex} index - The IVF index instance
 * @param {Uint8Array} bytes - Metadata bytes
 */
export function parseAuxiliaryPartitionInfo(index, bytes) {
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

    // Parse protobuf structure
    while (pos < bytes.length - 4) {
        const tag = readVarint();
        const fieldNum = tag >> 3;
        const wireType = tag & 0x7;

        if (wireType === 2) {
            const len = readVarint();
            if (len > bytes.length - pos) break;

            const content = bytes.slice(pos, pos + len);
            pos += len;

            if (fieldNum === 2 && len > 100 && len < 2000) {
                // Partition offsets (varint-encoded)
                const offsets = [];
                let innerPos = 0;
                while (innerPos < content.length) {
                    let val = 0, shift = 0;
                    while (innerPos < content.length) {
                        const byte = content[innerPos++];
                        val |= (byte & 0x7F) << shift;
                        if ((byte & 0x80) === 0) break;
                        shift += 7;
                    }
                    offsets.push(val);
                }
                if (offsets.length === index.numPartitions) {
                    index.partitionOffsets = offsets;
                    console.log(`[IVFIndex] Loaded ${offsets.length} partition offsets`);
                }
            } else if (fieldNum === 3 && len > 100 && len < 2000) {
                // Partition lengths (varint-encoded)
                const lengths = [];
                let innerPos = 0;
                while (innerPos < content.length) {
                    let val = 0, shift = 0;
                    while (innerPos < content.length) {
                        const byte = content[innerPos++];
                        val |= (byte & 0x7F) << shift;
                        if ((byte & 0x80) === 0) break;
                        shift += 7;
                    }
                    lengths.push(val);
                }
                if (lengths.length === index.numPartitions) {
                    index.partitionLengths = lengths;
                    console.log(`[IVFIndex] Loaded ${lengths.length} partition lengths`);
                }
            }
        } else if (wireType === 0) {
            readVarint();
        } else if (wireType === 1) {
            pos += 8;
        } else if (wireType === 5) {
            pos += 4;
        } else {
            break;
        }
    }
}
