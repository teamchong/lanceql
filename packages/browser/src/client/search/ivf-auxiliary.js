export async function loadAuxiliaryMetadata(index) {
    let headResp;
    try {
        headResp = await fetch(index.auxiliaryUrl, { method: 'HEAD' });
    } catch {
        return;
    }
    if (!headResp.ok) return;

    const fileSize = parseInt(headResp.headers.get('content-length'));
    if (!fileSize) return;

    const footerResp = await fetch(index.auxiliaryUrl, {
        headers: { 'Range': `bytes=${fileSize - 40}-${fileSize - 1}` }
    });
    if (!footerResp.ok) return;

    const footer = new Uint8Array(await footerResp.arrayBuffer());
    const view = new DataView(footer.buffer, footer.byteOffset);

    const colMetaStart = Number(view.getBigUint64(0, true));
    const colMetaOffsetsStart = Number(view.getBigUint64(8, true));
    const globalBuffOffsetsStart = Number(view.getBigUint64(16, true));
    const numGlobalBuffers = view.getUint32(24, true);
    const magic = new TextDecoder().decode(footer.slice(36, 40));

    if (magic !== 'LANC') return;

    const gboSize = numGlobalBuffers * 16;
    const gboResp = await fetch(index.auxiliaryUrl, {
        headers: { 'Range': `bytes=${globalBuffOffsetsStart}-${globalBuffOffsetsStart + gboSize - 1}` }
    });
    if (!gboResp.ok) return;

    const gboData = new Uint8Array(await gboResp.arrayBuffer());
    const gboView = new DataView(gboData.buffer, gboData.byteOffset);

    const buffers = [];
    for (let i = 0; i < numGlobalBuffers; i++) {
        const offset = Number(gboView.getBigUint64(i * 16, true));
        const length = Number(gboView.getBigUint64(i * 16 + 8, true));
        buffers.push({ offset, length });
    }

    if (buffers.length < 2) return;

    index._auxBuffers = buffers;
    index._auxFileSize = fileSize;

    const colMetaOffResp = await fetch(index.auxiliaryUrl, {
        headers: { 'Range': `bytes=${colMetaOffsetsStart}-${globalBuffOffsetsStart - 1}` }
    });
    if (!colMetaOffResp.ok) return;

    const colMetaOffData = new Uint8Array(await colMetaOffResp.arrayBuffer());
    if (colMetaOffData.length >= 32) {
        const colView = new DataView(colMetaOffData.buffer, colMetaOffData.byteOffset);
        const col0Pos = Number(colView.getBigUint64(0, true));
        const col0Len = Number(colView.getBigUint64(8, true));

        const col0MetaResp = await fetch(index.auxiliaryUrl, {
            headers: { 'Range': `bytes=${col0Pos}-${col0Pos + col0Len - 1}` }
        });
        if (col0MetaResp.ok) {
            const col0Meta = new Uint8Array(await col0MetaResp.arrayBuffer());
            parseColumnMetaForPartitions(index, col0Meta);
        }
    }
}

export function parseColumnMetaForPartitions(index, bytes) {
    let pos = 0;
    const pages = [];

    const readVarint = () => {
        let result = 0, shift = 0;
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

        if (wireType === 2) {
            const len = readVarint();
            if (len > bytes.length - pos) break;
            const content = bytes.slice(pos, pos + len);
            pos += len;
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

    index._columnPages = pages;
}

export function parsePageInfo(bytes) {
    let pos = 0;
    let numRows = 0;
    const bufferOffsets = [];
    const bufferSizes = [];

    const readVarint = () => {
        let result = 0, shift = 0;
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
            if (fieldNum === 3) numRows = val;
        } else if (wireType === 2) {
            const len = readVarint();
            const content = bytes.slice(pos, pos + len);
            pos += len;

            if (fieldNum === 1) {
                let p = 0;
                while (p < content.length) {
                    let val = 0n, shift = 0n;
                    while (p < content.length) {
                        const b = content[p++];
                        val |= BigInt(b & 0x7F) << shift;
                        if ((b & 0x80) === 0) break;
                        shift += 7n;
                    }
                    bufferOffsets.push(Number(val));
                }
            }
            if (fieldNum === 2) {
                let p = 0;
                while (p < content.length) {
                    let val = 0n, shift = 0n;
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

export function parseAuxiliaryPartitionInfo(index, bytes) {
    let pos = 0;

    const readVarint = () => {
        let result = 0, shift = 0;
        while (pos < bytes.length) {
            const byte = bytes[pos++];
            result |= (byte & 0x7F) << shift;
            if ((byte & 0x80) === 0) break;
            shift += 7;
        }
        return result;
    };

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
                }
            } else if (fieldNum === 3 && len > 100 && len < 2000) {
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
