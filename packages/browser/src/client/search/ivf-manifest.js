export async function findLatestManifestVersion(baseUrl) {
    const checkVersions = [1, 5, 10, 20, 50, 100];
    const checks = await Promise.all(
        checkVersions.map(async v => {
            try {
                const response = await fetch(`${baseUrl}/_versions/${v}.manifest`, { method: 'HEAD' });
                return response.ok ? v : 0;
            } catch {
                return 0;
            }
        })
    );

    let highestFound = Math.max(...checks);
    if (highestFound === 0) return null;

    for (let v = highestFound + 1; v <= highestFound + 30; v++) {
        try {
            const response = await fetch(`${baseUrl}/_versions/${v}.manifest`, { method: 'HEAD' });
            if (response.ok) highestFound = v;
            else break;
        } catch {
            break;
        }
    }

    return highestFound;
}

export function parseManifestForIndex(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset);
    const chunk1Len = view.getUint32(0, true);
    const chunk1Data = bytes.slice(4, 4 + chunk1Len);

    let pos = 0;
    let indexUuid = null;
    let indexFieldId = null;

    const readVarint = (data, startPos) => {
        let result = 0, shift = 0, p = startPos;
        while (p < data.length) {
            const byte = data[p++];
            result |= (byte & 0x7F) << shift;
            if ((byte & 0x80) === 0) break;
            shift += 7;
        }
        return { value: result, pos: p };
    };

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

            if (fieldNum === 1) {
                const parsed = parseIndexMetadata(content);
                if (parsed?.uuid) {
                    indexUuid = parsed.uuid;
                    indexFieldId = parsed.fieldId;
                }
            }
        } else if (wireType === 0) {
            pos = readVarint(chunk1Data, pos).pos;
        } else if (wireType === 5) {
            pos += 4;
        } else if (wireType === 1) {
            pos += 8;
        }
    }

    return indexUuid ? { uuid: indexUuid, fieldId: indexFieldId } : null;
}

export function parseIndexMetadata(bytes) {
    let pos = 0, uuid = null, fieldId = null;

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
            const content = bytes.slice(pos, pos + len);
            pos += len;
            if (fieldNum === 1) uuid = parseUuid(content);
        } else if (wireType === 0) {
            const val = readVarint();
            if (fieldNum === 2) fieldId = val;
        } else if (wireType === 5) {
            pos += 4;
        } else if (wireType === 1) {
            pos += 8;
        }
    }

    return { uuid, fieldId };
}

export function parseUuid(bytes) {
    let pos = 0;
    while (pos < bytes.length) {
        const tag = bytes[pos++];
        const fieldNum = tag >> 3;
        const wireType = tag & 0x7;

        if (wireType === 2 && fieldNum === 1) {
            const len = bytes[pos++];
            const uuidBytes = bytes.slice(pos, pos + len);
            const hex = Array.from(uuidBytes).map(b => b.toString(16).padStart(2, '0')).join('');
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

export function parseIndexFile(bytes, indexInfo, IVFIndexClass) {
    const index = new IVFIndexClass();
    const ivfData = findIVFMessage(bytes);

    if (ivfData) {
        if (ivfData.centroids) {
            index.centroids = ivfData.centroids.data;
            index.numPartitions = ivfData.centroids.numPartitions;
            index.dimension = ivfData.centroids.dimension;
        }
        if (ivfData.offsets?.length > 0) index.partitionOffsets = ivfData.offsets;
        if (ivfData.lengths?.length > 0) index.partitionLengths = ivfData.lengths;
    }

    if (!index.centroids) {
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

export function findIVFMessage(bytes) {
    let pos = 0, offsets = [], lengths = [], centroids = null;

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

            if (fieldNum === 2 && len % 8 === 0 && len > 0) {
                const view = new DataView(content.buffer, content.byteOffset, len);
                for (let i = 0; i < len / 8; i++) {
                    offsets.push(Number(view.getBigUint64(i * 8, true)));
                }
            } else if (fieldNum === 3) {
                if (len % 4 === 0 && len > 0) {
                    const view = new DataView(content.buffer, content.byteOffset, len);
                    for (let i = 0; i < len / 4; i++) {
                        lengths.push(view.getUint32(i * 4, true));
                    }
                } else {
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
                centroids = tryParseCentroids(content);
            } else if (len > 100) {
                const nested = findIVFMessage(content);
                if (nested?.centroids || nested?.offsets?.length > 0) {
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

    return (centroids || offsets.length > 0 || lengths.length > 0)
        ? { centroids, offsets, lengths }
        : null;
}

export function tryParseCentroids(bytes) {
    let pos = 0, shape = [], dataBytes = null, dataType = 2;

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
            if (fieldNum === 1) dataType = val;
        } else if (wireType === 2) {
            const len = readVarint();
            const content = bytes.slice(pos, pos + len);
            pos += len;

            if (fieldNum === 2) {
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
        const numPartitions = shape[0];
        const dimension = shape[1];
        if (dataBytes.length === numPartitions * dimension * 4) {
            return {
                data: new Float32Array(dataBytes.buffer, dataBytes.byteOffset, numPartitions * dimension),
                numPartitions,
                dimension
            };
        }
    }

    return null;
}
