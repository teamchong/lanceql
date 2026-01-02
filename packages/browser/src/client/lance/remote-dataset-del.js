/**
 * RemoteLanceDataset - Deletion file parsing
 * Extracted from remote-dataset.js for modularity
 */

/**
 * Parse DeletionFile protobuf message.
 * @param {Uint8Array} data - Raw protobuf bytes
 * @param {number} fragId - Fragment ID for path construction
 * @param {string} baseUrl - Dataset base URL
 * @returns {Object|null} Deletion file info
 */
export function parseDeletionFile(data, fragId, baseUrl) {
    let fileType = 0;  // 0 = ARROW_ARRAY, 1 = BITMAP
    let readVersion = 0;
    let id = 0;
    let numDeletedRows = 0;

    let pos = 0;
    const readVarint = () => {
        let result = 0;
        let shift = 0;
        while (pos < data.length) {
            const b = data[pos++];
            result |= (b & 0x7F) << shift;
            if ((b & 0x80) === 0) break;
            shift += 7;
        }
        return result;
    };

    while (pos < data.length) {
        const tag = readVarint();
        const fieldNum = tag >> 3;
        const wireType = tag & 0x7;

        if (wireType === 0) {
            const val = readVarint();
            if (fieldNum === 1) fileType = val;
            else if (fieldNum === 2) readVersion = val;
            else if (fieldNum === 3) id = val;
            else if (fieldNum === 4) numDeletedRows = val;
        } else if (wireType === 2) {
            const len = readVarint();
            pos += len;
        } else if (wireType === 5) {
            pos += 4;
        } else if (wireType === 1) {
            pos += 8;
        }
    }

    if (numDeletedRows === 0) return null;

    const ext = fileType === 0 ? 'arrow' : 'bin';
    const path = `_deletions/${fragId}-${readVersion}-${id}.${ext}`;

    return {
        fileType: fileType === 0 ? 'arrow' : 'bitmap',
        readVersion,
        id,
        numDeletedRows,
        path,
        url: `${baseUrl}/${path}`
    };
}

/**
 * Parse Arrow IPC deletion file (Int32Array of deleted indices).
 * @param {Uint8Array} data - Raw Arrow IPC bytes
 * @returns {Set<number>} Set of deleted row indices
 */
export function parseArrowDeletions(data) {
    const deletedSet = new Set();
    let pos = 0;

    // Skip magic "ARROW1" + padding
    if (data.length >= 8 && String.fromCharCode(...data.slice(0, 6)) === 'ARROW1') {
        pos = 8;
    }

    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);

    while (pos < data.length - 4) {
        const marker = view.getInt32(pos, true);
        if (marker === -1) {
            pos += 4;
            if (pos + 4 > data.length) break;
            const metaLen = view.getInt32(pos, true);
            pos += 4 + metaLen;

            while (pos + 4 <= data.length) {
                const nextMarker = view.getInt32(pos, true);
                if (nextMarker === -1) break;

                const val = view.getInt32(pos, true);
                if (val >= 0 && val < 10000000) {
                    deletedSet.add(val);
                }
                pos += 4;
            }
        } else {
            pos++;
        }
    }

    return deletedSet;
}

/**
 * Parse Roaring Bitmap deletion file.
 * @param {Uint8Array} data - Raw Roaring Bitmap bytes
 * @returns {Set<number>} Set of deleted row indices
 */
export function parseRoaringBitmap(data) {
    const deletedSet = new Set();
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);

    if (data.length < 8) return deletedSet;

    const cookie = view.getUint32(0, true);

    if (cookie === 12346 || cookie === 12347) {
        const isRunContainer = (cookie === 12347);
        let pos = 4;

        const numContainers = view.getUint16(pos, true);
        pos += 2;

        const keysStart = pos;
        pos += numContainers * 4;

        for (let i = 0; i < numContainers && pos < data.length; i++) {
            const key = view.getUint16(keysStart + i * 4, true);
            const card = view.getUint16(keysStart + i * 4 + 2, true) + 1;
            const baseValue = key << 16;

            for (let j = 0; j < card && pos + 2 <= data.length; j++) {
                const lowBits = view.getUint16(pos, true);
                deletedSet.add(baseValue | lowBits);
                pos += 2;
            }
        }
    }

    return deletedSet;
}

/**
 * Load deleted row indices for a fragment.
 * @param {Object} dataset - Dataset instance
 * @param {number} fragmentIndex - Fragment index
 * @returns {Promise<Set<number>>} Set of deleted row indices
 */
export async function loadDeletedRows(dataset, fragmentIndex) {
    if (dataset._deletedRows.has(fragmentIndex)) {
        return dataset._deletedRows.get(fragmentIndex);
    }

    const frag = dataset._fragments[fragmentIndex];
    if (!frag?.deletionFile) {
        const emptySet = new Set();
        dataset._deletedRows.set(fragmentIndex, emptySet);
        return emptySet;
    }

    const { url, fileType, numDeletedRows } = frag.deletionFile;
    console.log(`[LanceQL] Loading ${numDeletedRows} deletions from ${url} (${fileType})`);

    try {
        const response = await fetch(url);
        if (!response.ok) {
            console.warn(`[LanceQL] Failed to load deletion file: ${response.status}`);
            const emptySet = new Set();
            dataset._deletedRows.set(fragmentIndex, emptySet);
            return emptySet;
        }

        const buffer = await response.arrayBuffer();
        const data = new Uint8Array(buffer);
        let deletedSet;

        if (fileType === 'arrow') {
            deletedSet = parseArrowDeletions(data);
        } else {
            deletedSet = parseRoaringBitmap(data);
        }

        console.log(`[LanceQL] Loaded ${deletedSet.size} deleted rows for fragment ${fragmentIndex}`);
        dataset._deletedRows.set(fragmentIndex, deletedSet);
        return deletedSet;
    } catch (e) {
        console.error(`[LanceQL] Error loading deletion file:`, e);
        const emptySet = new Set();
        dataset._deletedRows.set(fragmentIndex, emptySet);
        return emptySet;
    }
}
