/**
 * RemoteLanceFile - Schema and manifest parsing
 * Extracted from remote-file.js for modularity
 */

/**
 * Try to detect dataset base URL and load schema from manifest.
 * Lance datasets have structure: base.lance/_versions/, base.lance/data/
 * @param {RemoteLanceFile} file - File instance
 * @returns {Promise<void>}
 */
export async function tryLoadSchema(file) {
    // Try to infer dataset base URL from file URL
    // Pattern: https://host/path/dataset.lance/data/filename.lance
    const match = file.url.match(/^(.+\.lance)\/data\/.+\.lance$/);
    if (!match) {
        // URL doesn't match standard Lance dataset structure
        return;
    }

    file._datasetBaseUrl = match[1];

    try {
        // Try manifest version 1 first
        const manifestUrl = `${file._datasetBaseUrl}/_versions/1.manifest`;
        const response = await fetch(manifestUrl);

        if (!response.ok) {
            return;
        }

        const manifestData = await response.arrayBuffer();
        file._schema = parseManifest(new Uint8Array(manifestData));
    } catch (e) {
        // Silently fail - schema is optional
    }
}

/**
 * Parse Lance manifest protobuf to extract schema.
 * Manifest structure:
 * - 4 bytes: content length (little-endian u32)
 * - N bytes: protobuf content
 * - 16 bytes: footer (zeros + version + LANC magic)
 * @param {Uint8Array} bytes - Manifest data
 * @returns {Array<{name: string, id: number, type: string}>}
 */
export function parseManifest(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset);

    // Lance manifest file structure:
    // - Chunk 1 (len-prefixed): Transaction metadata (may be small/incremental)
    // - Chunk 2 (len-prefixed): Full manifest with schema + fragments
    // - Footer (16 bytes): Offsets + "LANC" magic

    // Read chunk 1 length
    const chunk1Len = view.getUint32(0, true);

    // Check if there's a chunk 2 (full manifest data)
    const chunk2Start = 4 + chunk1Len;
    let protoData;

    if (chunk2Start + 4 < bytes.length) {
        const chunk2Len = view.getUint32(chunk2Start, true);
        if (chunk2Len > 0 && chunk2Start + 4 + chunk2Len <= bytes.length) {
            // Use chunk 2 (full manifest)
            protoData = bytes.slice(chunk2Start + 4, chunk2Start + 4 + chunk2Len);
        } else {
            protoData = bytes.slice(4, 4 + chunk1Len);
        }
    } else {
        protoData = bytes.slice(4, 4 + chunk1Len);
    }

    let pos = 0;
    const fields = [];

    const readVarint = () => {
        let result = 0;
        let shift = 0;
        while (pos < protoData.length) {
            const byte = protoData[pos++];
            result |= (byte & 0x7F) << shift;
            if ((byte & 0x80) === 0) break;
            shift += 7;
        }
        return result;
    };

    // Parse top-level Manifest message
    while (pos < protoData.length) {
        const tag = readVarint();
        const fieldNum = tag >> 3;
        const wireType = tag & 0x7;

        if (fieldNum === 1 && wireType === 2) {
            // Field 1 = schema (repeated Field message)
            const fieldLen = readVarint();
            const fieldEnd = pos + fieldLen;

            // Parse Field message
            let name = null;
            let id = null;
            let logicalType = null;

            while (pos < fieldEnd) {
                const fTag = readVarint();
                const fNum = fTag >> 3;
                const fWire = fTag & 0x7;

                if (fWire === 0) {
                    // Varint
                    const val = readVarint();
                    if (fNum === 3) id = val;  // Field.id
                } else if (fWire === 2) {
                    // Length-delimited
                    const len = readVarint();
                    const content = protoData.slice(pos, pos + len);
                    pos += len;

                    if (fNum === 2) {
                        // Field.name
                        name = new TextDecoder().decode(content);
                    } else if (fNum === 5) {
                        // Field.logical_type
                        logicalType = new TextDecoder().decode(content);
                    }
                } else if (fWire === 5) {
                    pos += 4;  // Fixed32
                } else if (fWire === 1) {
                    pos += 8;  // Fixed64
                }
            }

            if (name) {
                fields.push({ name, id, type: logicalType });
            }
        } else {
            // Skip other fields
            if (wireType === 0) {
                readVarint();
            } else if (wireType === 2) {
                const len = readVarint();
                pos += len;
            } else if (wireType === 5) {
                pos += 4;
            } else if (wireType === 1) {
                pos += 8;
            }
        }
    }

    return fields;
}

/**
 * Get column names from schema (if available).
 * Falls back to 'column_N' if schema not loaded.
 * @param {RemoteLanceFile} file - File instance
 * @returns {string[]}
 */
export function getColumnNames(file) {
    if (file._schema && file._schema.length > 0) {
        return file._schema.map(f => f.name);
    }
    // Fallback to generic names
    return Array.from({ length: file._numColumns }, (_, i) => `column_${i}`);
}

/**
 * Detect column types by sampling first row.
 * Returns array of type strings: 'string', 'int64', 'float64', 'float32', 'int32', 'int16', 'vector', 'unknown'
 * @param {RemoteLanceFile} file - File instance
 * @returns {Promise<string[]>}
 */
export async function detectColumnTypes(file) {
    // Return cached if available
    if (file._columnTypes) {
        return file._columnTypes;
    }

    const types = [];

    // First, try to use schema types if available
    if (file._schema && file._schema.length > 0) {
        // Build a map from schema - schema may have more fields than physical columns
        for (let c = 0; c < file._numColumns; c++) {
            const schemaField = file._schema[c];
            const schemaType = schemaField?.type?.toLowerCase() || '';
            const schemaName = schemaField?.name?.toLowerCase() || '';
            let type = 'unknown';

            // Check if column name suggests it's a vector/embedding
            const isEmbeddingName = schemaName.includes('embedding') || schemaName.includes('vector') ||
                                    schemaName.includes('emb') || schemaName === 'vec';

            // Map Lance/Arrow logical types to our types
            if (schemaType.includes('utf8') || schemaType.includes('string') || schemaType.includes('large_utf8')) {
                type = 'string';
            } else if (schemaType.includes('fixed_size_list') || schemaType.includes('vector') || isEmbeddingName) {
                type = 'vector';
            } else if (schemaType.includes('int64') || schemaType === 'int64') {
                type = 'int64';
            } else if (schemaType.includes('int32') || schemaType === 'int32') {
                type = 'int32';
            } else if (schemaType.includes('int16') || schemaType === 'int16') {
                type = 'int16';
            } else if (schemaType.includes('int8') || schemaType === 'int8') {
                type = 'int8';
            } else if (schemaType.includes('float64') || schemaType.includes('double')) {
                type = 'float64';
            } else if (schemaType.includes('float32') || schemaType.includes('float') && !schemaType.includes('64')) {
                type = 'float32';
            } else if (schemaType.includes('bool')) {
                type = 'bool';
            }

            types.push(type);
        }

        // If we got useful types from schema, cache and return
        if (types.some(t => t !== 'unknown')) {
            file._columnTypes = types;
            return types;
        }

        // Otherwise fall through to detection
        types.length = 0;
    }

    // Fall back to detection by examining data
    for (let c = 0; c < file._numColumns; c++) {
        let type = 'unknown';
        const colName = file.columnNames[c]?.toLowerCase() || '';

        // Check if column name suggests it's a vector/embedding
        const isEmbeddingName = colName.includes('embedding') || colName.includes('vector') ||
                                colName.includes('emb') || colName === 'vec';

        // Try string first - if we can read a valid string, it's a string column
        try {
            await file.readStringAt(c, 0);
            type = 'string';
            types.push(type);
            continue;
        } catch (e) {
            // Not a string column, continue to numeric detection
        }

        // Check numeric column by examining bytes per row
        try {
            const entry = await file.getColumnOffsetEntry(c);
            if (entry.len > 0) {
                const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
                const bytes = new Uint8Array(colMeta);
                const info = file._parseColumnMeta(bytes);

                if (info.rows > 0 && info.size > 0) {
                    const bytesPerRow = info.size / info.rows;

                    // If column name suggests embedding, treat as vector regardless of size
                    if (isEmbeddingName && bytesPerRow >= 4) {
                        type = 'vector';
                    } else if (bytesPerRow === 8) {
                        type = 'int64';
                    } else if (bytesPerRow === 4) {
                        // int32 or float32 - try reading as int32 to check
                        try {
                            const data = await file.readInt32AtIndices(c, [0]);
                            if (data.length > 0) {
                                const val = data[0];
                                if (val >= -1000000 && val <= 1000000 && Number.isInteger(val)) {
                                    type = 'int32';
                                } else {
                                    type = 'float32';
                                }
                            }
                        } catch (e) {
                            type = 'float32';
                        }
                    } else if (bytesPerRow > 8 && bytesPerRow % 4 === 0) {
                        type = 'vector';
                    } else if (bytesPerRow === 2) {
                        type = 'int16';
                    } else if (bytesPerRow === 1) {
                        type = 'int8';
                    }
                }
            }
        } catch (e) {
            // Failed to detect type for column, leave as unknown
        }

        types.push(type);
    }

    file._columnTypes = types;
    return types;
}
