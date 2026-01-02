/**
 * IVF Manifest and Index Parsing
 * Static methods for parsing Lance manifest and index files
 */

/**
 * Find latest manifest version using binary search.
 * @param {string} baseUrl - Dataset base URL
 * @returns {Promise<number|null>}
 */
export async function findLatestManifestVersion(baseUrl) {
    // Check common versions in parallel
    const checkVersions = [1, 5, 10, 20, 50, 100];
    const checks = await Promise.all(
        checkVersions.map(async v => {
            try {
                const url = `${baseUrl}/_versions/${v}.manifest`;
                const response = await fetch(url, { method: 'HEAD' });
                return response.ok ? v : 0;
            } catch {
                return 0;
            }
        })
    );

    let highestFound = Math.max(...checks);
    if (highestFound === 0) return null;

    // Scan forward from highest found
    for (let v = highestFound + 1; v <= highestFound + 30; v++) {
        try {
            const url = `${baseUrl}/_versions/${v}.manifest`;
            const response = await fetch(url, { method: 'HEAD' });
            if (response.ok) {
                highestFound = v;
            } else {
                break;
            }
        } catch {
            break;
        }
    }

    return highestFound;
}

/**
 * Parse manifest to find vector index info.
 * @param {Uint8Array} bytes - Manifest bytes
 * @returns {Object|null}
 */
export function parseManifestForIndex(bytes) {
    // Manifest structure:
    // - Chunk 1: 4 bytes len + content (index metadata in field 1)
    // - Chunk 2: 4 bytes len + content (full manifest with schema + fragments)
    // - Footer (16 bytes)

    const view = new DataView(bytes.buffer, bytes.byteOffset);
    const chunk1Len = view.getUint32(0, true);
    const chunk1Data = bytes.slice(4, 4 + chunk1Len);

    let pos = 0;
    let indexUuid = null;
    let indexFieldId = null;

    const readVarint = (data, startPos) => {
        let result = 0;
        let shift = 0;
        let p = startPos;
        while (p < data.length) {
            const byte = data[p++];
            result |= (byte & 0x7F) << shift;
            if ((byte & 0x80) === 0) break;
            shift += 7;
        }
        return { value: result, pos: p };
    };

    // Parse chunk 1 looking for index metadata (field 1)
    while (pos < chunk1Data.length) {
        const tagResult = readVarint(chunk1Data, pos);
        pos = tagResult.pos;
        const fieldNum = tagResult.value >> 3;
        const wireType = tagResult.value & 0x7;

        if (wireType === 2) {
            const lenResult = readVarint(chunk1Data, pos);
            pos = lenResult.pos;
            const content = chunk1Data.slice(pos, pos + lenResult.value);
            pos += lenResult.value;

            // Field 1 = IndexMetadata (contains UUID)
            if (fieldNum === 1) {
                const parsed = parseIndexMetadata(content);
                if (parsed && parsed.uuid) {
                    indexUuid = parsed.uuid;
                    indexFieldId = parsed.fieldId;
                }
            }
        } else if (wireType === 0) {
            const r = readVarint(chunk1Data, pos);
            pos = r.pos;
        } else if (wireType === 5) {
            pos += 4;
        } else if (wireType === 1) {
            pos += 8;
        }
    }

    return indexUuid ? { uuid: indexUuid, fieldId: indexFieldId } : null;
}

/**
 * Parse IndexMetadata protobuf message.
 * @param {Uint8Array} bytes - IndexMetadata bytes
 * @returns {Object}
 */
export function parseIndexMetadata(bytes) {
    let pos = 0;
    let uuid = null;
    let fieldId = null;

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

        if (wireType === 2) {
            const len = readVarint();
            const content = bytes.slice(pos, pos + len);
            pos += len;

            if (fieldNum === 1) {
                // UUID (nested message with bytes)
                uuid = parseUuid(content);
            }
        } else if (wireType === 0) {
            const val = readVarint();
            if (fieldNum === 2) {
                fieldId = val;
            }
        } else if (wireType === 5) {
            pos += 4;
        } else if (wireType === 1) {
            pos += 8;
        }
    }

    return { uuid, fieldId };
}

/**
 * Parse UUID protobuf message.
 * @param {Uint8Array} bytes - UUID bytes
 * @returns {string|null}
 */
export function parseUuid(bytes) {
    // UUID message: field 1 = bytes (16 bytes)
    let pos = 0;
    while (pos < bytes.length) {
        const tag = bytes[pos++];
        const fieldNum = tag >> 3;
        const wireType = tag & 0x7;

        if (wireType === 2 && fieldNum === 1) {
            const len = bytes[pos++];
            const uuidBytes = bytes.slice(pos, pos + len);
            // Convert to hex string with dashes (UUID format)
            const hex = Array.from(uuidBytes).map(b => b.toString(16).padStart(2, '0')).join('');
            // Format as UUID: 8-4-4-4-12
            return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20,32)}`;
        } else if (wireType === 0) {
            while (pos < bytes.length && (bytes[pos++] & 0x80)) {}
        } else if (wireType === 5) {
            pos += 4;
        } else if (wireType === 1) {
            pos += 8;
        }
    }
    return null;
}

/**
 * Parse IVF index file.
 * @param {Uint8Array} bytes - Index file bytes
 * @param {Object} indexInfo - Index info from manifest
 * @param {function} IVFIndexClass - IVFIndex constructor
 * @returns {IVFIndex|null}
 */
export function parseIndexFile(bytes, indexInfo, IVFIndexClass) {
    const index = new IVFIndexClass();

    // Try to find and parse IVF message within the file
    const ivfData = findIVFMessage(bytes);

    if (ivfData) {
        if (ivfData.centroids) {
            index.centroids = ivfData.centroids.data;
            index.numPartitions = ivfData.centroids.numPartitions;
            index.dimension = ivfData.centroids.dimension;
        }
        if (ivfData.offsets && ivfData.offsets.length > 0) {
            index.partitionOffsets = ivfData.offsets;
        }
        if (ivfData.lengths && ivfData.lengths.length > 0) {
            index.partitionLengths = ivfData.lengths;
        }
    }

    // Fallback: try to find centroids in nested messages
    if (!index.centroids) {
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

        while (pos < bytes.length - 4) {
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 2) {
                const len = readVarint();
                if (len > bytes.length - pos) break;

                const content = bytes.slice(pos, pos + len);
                pos += len;

                if (len > 100 && len < 100000000) {
                    const centroids = tryParseCentroids(content);
                    if (centroids) {
                        index.centroids = centroids.data;
                        index.numPartitions = centroids.numPartitions;
                        index.dimension = centroids.dimension;
                    }
                }
            } else if (wireType === 0) {
                readVarint();
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        }
    }

    return index.centroids ? index : null;
}

/**
 * Find and parse IVF message within index file bytes.
 * @param {Uint8Array} bytes - Index file bytes
 * @returns {Object|null}
 */
export function findIVFMessage(bytes) {
    let pos = 0;
    let offsets = [];
    let lengths = [];
    let centroids = null;

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

    while (pos < bytes.length - 4) {
        const startPos = pos;
        const tag = readVarint();
        const fieldNum = tag >> 3;
        const wireType = tag & 0x7;

        if (wireType === 2) {
            const len = readVarint();
            if (len > bytes.length - pos || len < 0) {
                pos = startPos + 1;
                continue;
            }

            const content = bytes.slice(pos, pos + len);
            pos += len;

            if (fieldNum === 2) {
                // offsets - packed uint64
                if (len % 8 === 0 && len > 0) {
                    const numOffsets = len / 8;
                    const view = new DataView(content.buffer, content.byteOffset, len);
                    for (let i = 0; i < numOffsets; i++) {
                        offsets.push(Number(view.getBigUint64(i * 8, true)));
                    }
                }
            } else if (fieldNum === 3) {
                // lengths - packed uint32
                if (len % 4 === 0 && len > 0) {
                    const numLengths = len / 4;
                    const view = new DataView(content.buffer, content.byteOffset, len);
                    for (let i = 0; i < numLengths; i++) {
                        lengths.push(view.getUint32(i * 4, true));
                    }
                } else {
                    // Try as packed varint
                    let lpos = 0;
                    while (lpos < content.length) {
                        let val = 0, shift = 0;
                        while (lpos < content.length) {
                            const byte = content[lpos++];
                            val |= (byte & 0x7F) << shift;
                            if ((byte & 0x80) === 0) break;
                            shift += 7;
                        }
                        lengths.push(val);
                    }
                }
            } else if (fieldNum === 4) {
                // centroids_tensor
                centroids = tryParseCentroids(content);
            } else if (len > 100) {
                // Recursively search nested messages
                const nested = findIVFMessage(content);
                if (nested && (nested.centroids || nested.offsets?.length > 0)) {
                    if (nested.centroids && !centroids) centroids = nested.centroids;
                    if (nested.offsets?.length > offsets.length) offsets = nested.offsets;
                    if (nested.lengths?.length > lengths.length) lengths = nested.lengths;
                }
            }
        } else if (wireType === 0) {
            readVarint();
        } else if (wireType === 5) {
            pos += 4;
        } else if (wireType === 1) {
            pos += 8;
        } else {
            pos = startPos + 1;
        }
    }

    if (centroids || offsets.length > 0 || lengths.length > 0) {
        return { centroids, offsets, lengths };
    }
    return null;
}

/**
 * Try to parse centroids from a Tensor message.
 * @param {Uint8Array} bytes - Tensor bytes
 * @returns {Object|null}
 */
export function tryParseCentroids(bytes) {
    let pos = 0;
    let shape = [];
    let dataBytes = null;
    let dataType = 2; // Default to float32

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
            if (fieldNum === 1) dataType = val;
        } else if (wireType === 2) {
            const len = readVarint();
            const content = bytes.slice(pos, pos + len);
            pos += len;

            if (fieldNum === 2) {
                // shape (packed repeated uint32)
                let shapePos = 0;
                while (shapePos < content.length) {
                    let val = 0, shift = 0;
                    while (shapePos < content.length) {
                        const byte = content[shapePos++];
                        val |= (byte & 0x7F) << shift;
                        if ((byte & 0x80) === 0) break;
                        shift += 7;
                    }
                    shape.push(val);
                }
            } else if (fieldNum === 3) {
                dataBytes = content;
            }
        } else if (wireType === 5) {
            pos += 4;
        } else if (wireType === 1) {
            pos += 8;
        }
    }

    if (shape.length >= 2 && dataBytes && dataType === 2) {
        // float32 tensor with at least 2D shape
        const numPartitions = shape[0];
        const dimension = shape[1];

        if (dataBytes.length === numPartitions * dimension * 4) {
            const data = new Float32Array(dataBytes.buffer, dataBytes.byteOffset, numPartitions * dimension);
            return { data, numPartitions, dimension };
        }
    }

    return null;
}
