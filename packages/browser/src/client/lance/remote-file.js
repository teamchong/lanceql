/**
 * RemoteLanceFile - Remote HTTP file reader with streaming
 */

class RemoteLanceFile {
    constructor(lanceql, url, fileSize, footerData) {
        this.lanceql = lanceql;
        this.wasm = lanceql.wasm;
        this.memory = lanceql.memory;
        this.url = url;
        this.fileSize = fileSize;

        // Store footer data in WASM memory
        const bytes = new Uint8Array(footerData);
        this.footerPtr = this.wasm.alloc(bytes.length);
        if (!this.footerPtr) {
            throw new Error('Failed to allocate memory for footer');
        }
        this.footerLen = bytes.length;
        new Uint8Array(this.memory.buffer).set(bytes, this.footerPtr);

        // Parse footer
        this._numColumns = this.wasm.parseFooterGetColumns(this.footerPtr, this.footerLen);
        this._majorVersion = this.wasm.parseFooterGetMajorVersion(this.footerPtr, this.footerLen);
        this._minorVersion = this.wasm.parseFooterGetMinorVersion(this.footerPtr, this.footerLen);
        this._columnMetaStart = this.wasm.getColumnMetaStart(this.footerPtr, this.footerLen);
        this._columnMetaOffsetsStart = this.wasm.getColumnMetaOffsetsStart(this.footerPtr, this.footerLen);

        // Cache for column metadata to avoid repeated fetches
        this._columnMetaCache = new Map();
        this._columnOffsetCache = new Map();
        this._columnTypes = null;

        // Schema info from manifest (populated by loadSchema())
        this._schema = null;
        this._datasetBaseUrl = null;

        // IVF index for ANN search (populated by tryLoadIndex())
        this._ivfIndex = null;
    }

    /**
     * Open a remote Lance file.
     * @param {LanceQL} lanceql
     * @param {string} url
     * @returns {Promise<RemoteLanceFile>}
     */
    static async open(lanceql, url) {
        // First, get file size with HEAD request
        const headResponse = await fetch(url, { method: 'HEAD' });
        if (!headResponse.ok) {
            throw new Error(`HTTP error: ${headResponse.status}`);
        }

        const contentLength = headResponse.headers.get('Content-Length');
        if (!contentLength) {
            throw new Error('Server did not return Content-Length');
        }
        const fileSize = parseInt(contentLength, 10);

        // Fetch footer (last 40 bytes)
        const footerSize = 40;
        const footerStart = fileSize - footerSize;
        const footerResponse = await fetch(url, {
            headers: {
                'Range': `bytes=${footerStart}-${fileSize - 1}`
            }
        });

        if (!footerResponse.ok && footerResponse.status !== 206) {
            throw new Error(`HTTP error: ${footerResponse.status}`);
        }

        const footerData = await footerResponse.arrayBuffer();

        // Verify magic bytes
        const footerBytes = new Uint8Array(footerData);
        const magic = String.fromCharCode(
            footerBytes[36], footerBytes[37], footerBytes[38], footerBytes[39]
        );
        if (magic !== 'LANC') {
            throw new Error(`Invalid Lance file: expected LANC magic, got "${magic}"`);
        }

        const file = new RemoteLanceFile(lanceql, url, fileSize, footerData);

        // Try to detect and load schema from manifest
        await file._tryLoadSchema();

        // Try to load IVF index for ANN search
        await file._tryLoadIndex();

        // Log summary
        console.log(`[LanceQL] Loaded: ${file._numColumns} columns, ${(fileSize / 1024 / 1024).toFixed(1)}MB, schema: ${file._schema ? 'yes' : 'no'}, index: ${file.hasIndex() ? 'yes' : 'no'}`);

        return file;
    }

    /**
     * Try to load IVF index from dataset.
     * @private
     */
    async _tryLoadIndex() {
        if (!this._datasetBaseUrl) return;

        try {
            this._ivfIndex = await IVFIndex.tryLoad(this._datasetBaseUrl);
        } catch (e) {
            // Index loading is optional, silently ignore
        }
    }

    /**
     * Check if ANN index is available.
     * @returns {boolean}
     */
    hasIndex() {
        return this._ivfIndex !== null && this._ivfIndex.centroids !== null;
    }

    /**
     * Try to detect dataset base URL and load schema from manifest.
     * Lance datasets have structure: base.lance/_versions/, base.lance/data/
     * @private
     */
    async _tryLoadSchema() {
        // Try to infer dataset base URL from file URL
        // Pattern: https://host/path/dataset.lance/data/filename.lance
        const match = this.url.match(/^(.+\.lance)\/data\/.+\.lance$/);
        if (!match) {
            // URL doesn't match standard Lance dataset structure
            return;
        }

        this._datasetBaseUrl = match[1];

        try {
            // Try manifest version 1 first
            const manifestUrl = `${this._datasetBaseUrl}/_versions/1.manifest`;
            const response = await fetch(manifestUrl);

            if (!response.ok) {
                return;
            }

            const manifestData = await response.arrayBuffer();
            this._schema = this._parseManifest(new Uint8Array(manifestData));
        } catch (e) {
            // Silently fail - schema is optional
            // Manifest loading is optional, silently ignore
        }
    }

    /**
     * Parse Lance manifest protobuf to extract schema.
     * Manifest structure:
     * - 4 bytes: content length (little-endian u32)
     * - N bytes: protobuf content
     * - 16 bytes: footer (zeros + version + LANC magic)
     * @private
     */
    _parseManifest(bytes) {
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
     * @returns {string[]}
     */
    get columnNames() {
        if (this._schema && this._schema.length > 0) {
            return this._schema.map(f => f.name);
        }
        // Fallback to generic names
        return Array.from({ length: this._numColumns }, (_, i) => `column_${i}`);
    }

    /**
     * Get full schema info (if available).
     * @returns {Array<{name: string, id: number, type: string}>|null}
     */
    get schema() {
        return this._schema;
    }

    /**
     * Get dataset base URL (if detected).
     * @returns {string|null}
     */
    get datasetBaseUrl() {
        return this._datasetBaseUrl;
    }

    /**
     * Fetch bytes from the remote file at a specific range.
     * Uses HotTierCache for OPFS-backed caching (500-2000x faster on cache hit).
     * @param {number} start - Start offset
     * @param {number} end - End offset (inclusive)
     * @returns {Promise<ArrayBuffer>}
     */
    async fetchRange(start, end) {
        // Debug: console.log(`fetchRange: ${start}-${end} (size: ${end - start + 1})`);

        // Validate range
        if (start < 0 || end < start || end >= this.size) {
            console.error(`Invalid range: ${start}-${end}, file size: ${this.size}`);
        }

        // Use hot-tier cache if available
        if (hotTierCache.enabled) {
            const data = await hotTierCache.getRange(this.url, start, end, this.size);

            // Track stats if callback available
            if (this._onFetch) {
                this._onFetch(data.byteLength, 1);
            }

            return data;
        }

        // Fallback to direct fetch
        const response = await fetch(this.url, {
            headers: {
                'Range': `bytes=${start}-${end}`
            }
        });

        if (!response.ok && response.status !== 206) {
            console.error(`Fetch failed: ${response.status} for range ${start}-${end}`);
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.arrayBuffer();

        // Track stats if callback available
        if (this._onFetch) {
            this._onFetch(data.byteLength, 1);
        }

        return data;
    }

    /**
     * Set callback for network stats tracking.
     * @param {function} callback - Function(bytesDownloaded, requestCount)
     */
    onFetch(callback) {
        this._onFetch = callback;
    }

    /**
     * Close the file and free memory.
     */
    close() {
        if (this.footerPtr) {
            this.wasm.free(this.footerPtr, this.footerLen);
            this.footerPtr = null;
        }
    }

    /**
     * Get the number of columns.
     * @returns {number}
     */
    get numColumns() {
        return this._numColumns;
    }

    /**
     * Get the file size.
     * @returns {number}
     */
    get size() {
        return this.fileSize;
    }

    /**
     * Get the version.
     * @returns {{major: number, minor: number}}
     */
    get version() {
        return {
            major: this._majorVersion,
            minor: this._minorVersion
        };
    }

    /**
     * Get the column metadata start offset.
     * @returns {number}
     */
    get columnMetaStart() {
        return Number(this._columnMetaStart);
    }

    /**
     * Get the column metadata offsets start.
     * @returns {number}
     */
    get columnMetaOffsetsStart() {
        return Number(this._columnMetaOffsetsStart);
    }

    /**
     * Get column offset entry from column metadata offsets.
     * Uses caching to avoid repeated fetches.
     * @param {number} colIdx
     * @returns {Promise<{pos: number, len: number}>}
     */
    async getColumnOffsetEntry(colIdx) {
        if (colIdx >= this._numColumns) {
            return { pos: 0, len: 0 };
        }

        // Check cache first
        if (this._columnOffsetCache.has(colIdx)) {
            return this._columnOffsetCache.get(colIdx);
        }

        // Each entry is 16 bytes (8 bytes pos + 8 bytes len)
        const entryOffset = this.columnMetaOffsetsStart + colIdx * 16;
        const data = await this.fetchRange(entryOffset, entryOffset + 15);
        const view = new DataView(data);

        const entry = {
            pos: Number(view.getBigUint64(0, true)),
            len: Number(view.getBigUint64(8, true))
        };

        // Cache the result
        this._columnOffsetCache.set(colIdx, entry);
        return entry;
    }

    /**
     * Get debug info for a column (requires network request).
     * @param {number} colIdx
     * @returns {Promise<{offset: number, size: number, rows: number}>}
     */
    async getColumnDebugInfo(colIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) {
            return { offset: 0, size: 0, rows: 0 };
        }

        // Fetch column metadata
        const colMetaData = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const bytes = new Uint8Array(colMetaData);

        // Parse column metadata to get buffer info
        const info = this._parseColumnMeta(bytes);
        return info;
    }

    /**
     * Parse column metadata to extract buffer offsets and row count.
     * For nullable columns, there are typically 2 buffers:
     * - Buffer 0: null bitmap
     * - Buffer 1: actual data values
     * @private
     */
    _parseColumnMeta(bytes) {
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
                // Don't break - continue to read more pages
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
        // Also compute total size across all pages for multi-page columns
        const firstPage = pages[0] || { offsets: [], sizes: [], rows: 0 };
        const bufferOffsets = firstPage.offsets;
        const bufferSizes = firstPage.sizes;

        // For multi-page columns (like embeddings), compute total size
        let totalSize = 0;
        for (const page of pages) {
            // Use the data buffer (last buffer, or buffer 1 for nullable)
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
            pages  // Include all pages for multi-page access
        };
    }

    /**
     * Parse string column metadata to get offsets and data buffer info.
     * @private
     */
    _parseStringColumnMeta(bytes) {
        // Parse ALL pages for multi-page string columns
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
     * @private
     */
    _batchIndices(indices, valueSize, gapThreshold = 1024) {
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

    /**
     * Read int64 values at specific row indices via Range requests.
     * Uses batched fetching to minimize HTTP requests.
     * @param {number} colIdx
     * @param {number[]} indices - Row indices
     * @returns {Promise<BigInt64Array>}
     */
    async readInt64AtIndices(colIdx, indices) {
        if (indices.length === 0) return new BigInt64Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        // Debug: console.log(`readInt64AtIndices col ${colIdx}: rows=${info.rows}`);

        const results = new BigInt64Array(indices.length);
        const valueSize = 8;

        // Batch indices into contiguous ranges
        const batches = this._batchIndices(indices, valueSize);

        // Fetch each batch in parallel
        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            // Extract values from batch
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
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Float64Array>}
     */
    async readFloat64AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Float64Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Float64Array(indices.length);
        const valueSize = 8;

        // Batch indices into contiguous ranges
        const batches = this._batchIndices(indices, valueSize);

        // Fetch each batch in parallel
        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
            const view = new DataView(data);

            // Extract values from batch
            for (const item of batch.items) {
                const localOffset = (item.idx - batch.startIdx) * valueSize;
                results[item.origPos] = view.getFloat64(localOffset, true);
            }
        }));

        return results;
    }

    /**
     * Read int32 values at specific row indices via Range requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Int32Array>}
     */
    async readInt32AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Int32Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Int32Array(indices.length);
        const valueSize = 4;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
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
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Float32Array>}
     */
    async readFloat32AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Float32Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Float32Array(indices.length);
        const valueSize = 4;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
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
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Int16Array>}
     */
    async readInt16AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Int16Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Int16Array(indices.length);
        const valueSize = 2;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
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
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Uint8Array>}
     */
    async readUint8AtIndices(colIdx, indices) {
        if (indices.length === 0) return new Uint8Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        const results = new Uint8Array(indices.length);
        const valueSize = 1;

        const batches = this._batchIndices(indices, valueSize);

        await Promise.all(batches.map(async (batch) => {
            const startOffset = info.offset + batch.startIdx * valueSize;
            const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
            const data = await this.fetchRange(startOffset, endOffset);
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
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<Uint8Array>}
     */
    async readBoolAtIndices(colIdx, indices) {
        if (indices.length === 0) return new Uint8Array(0);

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

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
        const data = await this.fetchRange(startOffset, endOffset);
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

    /**
     * Read a single string at index via Range requests.
     * @param {number} colIdx
     * @param {number} rowIdx
     * @returns {Promise<string>}
     * @throws {Error} If the column is not a string column
     */
    async readStringAt(colIdx, rowIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseStringColumnMeta(new Uint8Array(colMeta));

        // Check if this is actually a string column
        // String columns have: offsetsSize / rows = 4 or 8 bytes per offset
        // Numeric columns with validity bitmap have: offsetsSize = rows / 8 (bitmap)
        if (info.offsetsSize === 0 || info.dataSize === 0) {
            throw new Error(`Not a string column - offsetsSize=${info.offsetsSize}, dataSize=${info.dataSize}`);
        }

        // Calculate bytes per offset - strings have rows offsets of 4 or 8 bytes each
        const bytesPerOffset = info.offsetsSize / info.rows;

        // If bytesPerOffset is not 4 or 8, this is not a string column
        // (e.g., it's a validity bitmap which has rows/8 bytes = 0.125 bytes per row)
        if (bytesPerOffset !== 4 && bytesPerOffset !== 8) {
            throw new Error(`Not a string column - bytesPerOffset=${bytesPerOffset}, expected 4 or 8`);
        }

        if (rowIdx >= info.rows) return '';

        // Determine offset size (4 or 8 bytes)
        const offsetSize = bytesPerOffset;

        // Fetch the two offsets for this string
        const offsetStart = info.offsetsStart + rowIdx * offsetSize;
        const offsetData = await this.fetchRange(offsetStart, offsetStart + offsetSize * 2 - 1);
        const offsetView = new DataView(offsetData);

        let strStart, strEnd;
        if (offsetSize === 4) {
            strStart = offsetView.getUint32(0, true);
            strEnd = offsetView.getUint32(4, true);
        } else {
            strStart = Number(offsetView.getBigUint64(0, true));
            strEnd = Number(offsetView.getBigUint64(8, true));
        }

        if (strEnd <= strStart) return '';
        const strLen = strEnd - strStart;

        // Fetch the string data
        const strData = await this.fetchRange(
            info.dataStart + strStart,
            info.dataStart + strEnd - 1
        );

        return new TextDecoder().decode(strData);
    }

    /**
     * Read multiple strings at indices via Range requests.
     * Uses batched fetching to minimize HTTP requests.
     * @param {number} colIdx
     * @param {number[]} indices
     * @returns {Promise<string[]>}
     */
    async readStringsAtIndices(colIdx, indices) {
        if (indices.length === 0) return [];

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseStringColumnMeta(new Uint8Array(colMeta));

        if (!info.pages || info.pages.length === 0) {
            return indices.map(() => '');
        }

        const results = new Array(indices.length).fill('');

        // Build page index with cumulative row counts
        let pageRowStart = 0;
        const pageIndex = [];
        for (const page of info.pages) {
            if (page.offsetsSize === 0 || page.dataSize === 0 || page.rows === 0) {
                pageRowStart += page.rows;
                continue;
            }
            pageIndex.push({
                start: pageRowStart,
                end: pageRowStart + page.rows,
                page
            });
            pageRowStart += page.rows;
        }

        // Group indices by page
        const pageGroups = new Map();
        for (let i = 0; i < indices.length; i++) {
            const rowIdx = indices[i];
            // Find which page contains this row
            for (let p = 0; p < pageIndex.length; p++) {
                const pi = pageIndex[p];
                if (rowIdx >= pi.start && rowIdx < pi.end) {
                    if (!pageGroups.has(p)) {
                        pageGroups.set(p, []);
                    }
                    pageGroups.get(p).push({
                        globalIdx: rowIdx,
                        localIdx: rowIdx - pi.start,
                        resultIdx: i
                    });
                    break;
                }
            }
        }

        // Fetch strings from each page
        for (const [pageNum, items] of pageGroups) {
            const pi = pageIndex[pageNum];
            const page = pi.page;

            // Determine offset size (4 or 8 bytes per offset)
            const offsetSize = page.offsetsSize / page.rows;
            if (offsetSize !== 4 && offsetSize !== 8) continue;

            // Sort items by localIdx for efficient batching
            items.sort((a, b) => a.localIdx - b.localIdx);

            // Fetch offsets in batches
            const offsetBatches = [];
            let batchStart = 0;
            for (let i = 1; i <= items.length; i++) {
                if (i === items.length || items[i].localIdx - items[i-1].localIdx > 100) {
                    offsetBatches.push(items.slice(batchStart, i));
                    batchStart = i;
                }
            }

            // Collect string ranges from offset fetches
            // Lance string encoding: offset[N] = end of string N, start is offset[N-1] (or 0 if N=0)
            const stringRanges = [];

            await Promise.all(offsetBatches.map(async (batch) => {
                const minIdx = batch[0].localIdx;
                const maxIdx = batch[batch.length - 1].localIdx;

                // Fetch offsets: need offset[minIdx-1] through offset[maxIdx]
                // But if minIdx=0, we don't need offset[-1] since start is implicitly 0
                const fetchStartIdx = minIdx > 0 ? minIdx - 1 : 0;
                const fetchEndIdx = maxIdx;
                const startOffset = page.offsetsStart + fetchStartIdx * offsetSize;
                const endOffset = page.offsetsStart + (fetchEndIdx + 1) * offsetSize - 1;
                const data = await this.fetchRange(startOffset, endOffset);
                const view = new DataView(data);

                for (const item of batch) {
                    // Position in fetched data
                    const dataIdx = item.localIdx - fetchStartIdx;
                    let strStart, strEnd;

                    if (offsetSize === 4) {
                        // strEnd = offset[localIdx], strStart = offset[localIdx-1] or 0
                        strEnd = view.getUint32(dataIdx * 4, true);
                        strStart = item.localIdx === 0 ? 0 : view.getUint32((dataIdx - 1) * 4, true);
                    } else {
                        strEnd = Number(view.getBigUint64(dataIdx * 8, true));
                        strStart = item.localIdx === 0 ? 0 : Number(view.getBigUint64((dataIdx - 1) * 8, true));
                    }

                    if (strEnd > strStart) {
                        stringRanges.push({
                            start: strStart,
                            end: strEnd,
                            resultIdx: item.resultIdx,
                            dataStart: page.dataStart
                        });
                    }
                }
            }));

            // Fetch string data
            if (stringRanges.length > 0) {
                stringRanges.sort((a, b) => a.start - b.start);

                // Batch nearby string fetches
                const dataBatches = [];
                let dbStart = 0;
                for (let i = 1; i <= stringRanges.length; i++) {
                    if (i === stringRanges.length ||
                        stringRanges[i].start - stringRanges[i-1].end > 4096) {
                        dataBatches.push({
                            rangeStart: stringRanges[dbStart].start,
                            rangeEnd: stringRanges[i-1].end,
                            items: stringRanges.slice(dbStart, i),
                            dataStart: stringRanges[dbStart].dataStart
                        });
                        dbStart = i;
                    }
                }

                await Promise.all(dataBatches.map(async (batch) => {
                    const data = await this.fetchRange(
                        batch.dataStart + batch.rangeStart,
                        batch.dataStart + batch.rangeEnd - 1
                    );
                    const bytes = new Uint8Array(data);

                    for (const item of batch.items) {
                        const localStart = item.start - batch.rangeStart;
                        const len = item.end - item.start;
                        const strBytes = bytes.slice(localStart, localStart + len);
                        results[item.resultIdx] = new TextDecoder().decode(strBytes);
                    }
                }));
            }
        }

        return results;
    }

    /**
     * Get row count for a column.
     * @param {number} colIdx
     * @returns {Promise<number>}
     */
    async getRowCount(colIdx) {
        const info = await this.getColumnDebugInfo(colIdx);
        return info.rows;
    }

    /**
     * Detect column types by sampling first row.
     * Returns array of type strings: 'string', 'int64', 'float64', 'float32', 'int32', 'int16', 'vector', 'unknown'
     * @returns {Promise<string[]>}
     */
    async detectColumnTypes() {
        // Return cached if available
        if (this._columnTypes) {
            return this._columnTypes;
        }

        const types = [];

        // First, try to use schema types if available
        if (this._schema && this._schema.length > 0) {
            // Schema loaded successfully

            // Build a map from schema - schema may have more fields than physical columns
            for (let c = 0; c < this._numColumns; c++) {
                const schemaField = this._schema[c];
                const schemaType = schemaField?.type?.toLowerCase() || '';
                const schemaName = schemaField?.name?.toLowerCase() || '';
                let type = 'unknown';

                // Debug: console.log(`Column ${c}: name="${schemaField?.name}", type="${schemaType}"`);

                // Check if column name suggests it's a vector/embedding
                const isEmbeddingName = schemaName.includes('embedding') || schemaName.includes('vector') ||
                                        schemaName.includes('emb') || schemaName === 'vec';

                // Map Lance/Arrow logical types to our types
                if (schemaType.includes('utf8') || schemaType.includes('string') || schemaType.includes('large_utf8')) {
                    type = 'string';
                } else if (schemaType.includes('fixed_size_list') || schemaType.includes('vector') || isEmbeddingName) {
                    // Vector detection - check schema type OR column name
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
                // Debug: console.log('Detected types from schema:', types);
                this._columnTypes = types;
                return types;
            }

            // Otherwise fall through to detection
            // Schema types all unknown, fall back to data detection
            types.length = 0;
        }

        // Fall back to detection by examining data
        // Detecting column types from data
        for (let c = 0; c < this._numColumns; c++) {
            let type = 'unknown';
            const colName = this.columnNames[c]?.toLowerCase() || '';

            // Check if column name suggests it's a vector/embedding
            const isEmbeddingName = colName.includes('embedding') || colName.includes('vector') ||
                                    colName.includes('emb') || colName === 'vec';

            // Try string first - if we can read a valid string, it's a string column
            try {
                const str = await this.readStringAt(c, 0);
                // readStringAt throws for non-string columns, returns string for valid string columns
                type = 'string';
                // Detected as string
                types.push(type);
                continue;
            } catch (e) {
                // Not a string column, continue to numeric detection
            }

            // Check numeric column by examining bytes per row
            try {
                const entry = await this.getColumnOffsetEntry(c);
                if (entry.len > 0) {
                    const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
                    const bytes = new Uint8Array(colMeta);
                    const info = this._parseColumnMeta(bytes);

                    // Debug: console.log(`Column ${c}: bytesPerRow=${info.size / info.rows}`);

                    if (info.rows > 0 && info.size > 0) {
                        const bytesPerRow = info.size / info.rows;

                        // If column name suggests embedding, treat as vector regardless of size
                        if (isEmbeddingName && bytesPerRow >= 4) {
                            type = 'vector';
                        } else if (bytesPerRow === 8) {
                            // int64 or float64 - try to distinguish
                            type = 'int64';  // Default to int64
                        } else if (bytesPerRow === 4) {
                            // int32 or float32 - try reading as int32 to check
                            try {
                                const data = await this.readInt32AtIndices(c, [0]);
                                if (data.length > 0) {
                                    const val = data[0];
                                    // Detected int32 via sample value
                                    // Heuristic: small integers likely int32, weird values likely float32
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

            // Debug: console.log(`Column ${c}: ${type}`);
            types.push(type);
        }

        this._columnTypes = types;
        return types;
    }

    /**
     * Get cached column metadata, fetching if necessary.
     * @private
     */
    async _getCachedColumnMeta(colIdx) {
        if (this._columnMetaCache.has(colIdx)) {
            return this._columnMetaCache.get(colIdx);
        }

        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) {
            return null;
        }

        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const bytes = new Uint8Array(colMeta);

        this._columnMetaCache.set(colIdx, bytes);
        return bytes;
    }

    // ========================================================================
    // Vector Column Support (for embeddings/semantic search via Range requests)
    // ========================================================================

    /**
     * Get vector info for a column via Range requests.
     * @param {number} colIdx - Column index
     * @returns {Promise<{rows: number, dimension: number}>}
     */
    async getVectorInfo(colIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) return { rows: 0, dimension: 0 };

        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        if (info.rows === 0) return { rows: 0, dimension: 0 };

        // Calculate dimension from first page (all pages have same dimension)
        let dimension = 0;
        if (info.pages && info.pages.length > 0) {
            const firstPage = info.pages[0];
            const dataIdx = firstPage.sizes.length > 1 ? 1 : 0;
            const pageSize = firstPage.sizes[dataIdx] || 0;
            const pageRows = firstPage.rows || 0;
            if (pageRows > 0 && pageSize > 0) {
                dimension = Math.floor(pageSize / (pageRows * 4));
            }
        } else if (info.size > 0) {
            // Fallback for single-page
            dimension = Math.floor(info.size / (info.rows * 4));
        }

        return { rows: info.rows, dimension };
    }

    /**
     * Read a single vector at index via Range requests.
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Row index
     * @returns {Promise<Float32Array>}
     */
    async readVectorAt(colIdx, rowIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        if (info.rows === 0) return new Float32Array(0);
        if (rowIdx >= info.rows) return new Float32Array(0);

        const dim = Math.floor(info.size / (info.rows * 4));
        if (dim === 0) return new Float32Array(0);

        // Fetch the vector data
        const vecStart = info.offset + rowIdx * dim * 4;
        const vecEnd = vecStart + dim * 4 - 1;
        const data = await this.fetchRange(vecStart, vecEnd);

        return new Float32Array(data);
    }

    /**
     * Read multiple vectors at indices via Range requests.
     * Uses batched fetching for efficiency.
     * @param {number} colIdx - Column index
     * @param {number[]} indices - Row indices
     * @returns {Promise<Float32Array[]>}
     */
    async readVectorsAtIndices(colIdx, indices) {
        if (indices.length === 0) return [];

        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const info = this._parseColumnMeta(new Uint8Array(colMeta));

        if (info.rows === 0) return indices.map(() => new Float32Array(0));

        const dim = Math.floor(info.size / (info.rows * 4));
        if (dim === 0) return indices.map(() => new Float32Array(0));

        const vecSize = dim * 4;
        const results = new Array(indices.length);

        // Batch indices for efficient fetching - parallel with limit
        const batches = this._batchIndices(indices, vecSize, vecSize * 50);
        const BATCH_PARALLEL = 6;

        for (let i = 0; i < batches.length; i += BATCH_PARALLEL) {
            const batchGroup = batches.slice(i, i + BATCH_PARALLEL);
            await Promise.all(batchGroup.map(async (batch) => {
                try {
                    const startOffset = info.offset + batch.startIdx * vecSize;
                    const endOffset = info.offset + (batch.endIdx + 1) * vecSize - 1;
                    const data = await this.fetchRange(startOffset, endOffset);

                    for (const item of batch.items) {
                        const localOffset = (item.idx - batch.startIdx) * vecSize;
                        results[item.origPos] = new Float32Array(
                            data.slice(localOffset, localOffset + vecSize)
                        );
                    }
                } catch (e) {
                    for (const item of batch.items) {
                        results[item.origPos] = new Float32Array(0);
                    }
                }
            }));
        }

        return results;
    }

    /**
     * Compute cosine similarity between two vectors (in JS).
     * @param {Float32Array} vecA
     * @param {Float32Array} vecB
     * @returns {number}
     */
    cosineSimilarity(vecA, vecB) {
        if (vecA.length !== vecB.length) return 0;

        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < vecA.length; i++) {
            dot += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }

        const denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom === 0 ? 0 : dot / denom;
    }

    /**
     * Find top-k most similar vectors to query via Range requests.
     * NOTE: This requires scanning the entire vector column which can be slow
     * for large datasets. For production, use an index.
     *
     * @param {number} colIdx - Column index with vectors
     * @param {Float32Array} queryVec - Query vector
     * @param {number} topK - Number of results to return
     * @param {function} onProgress - Progress callback(current, total)
     * @param {object} options - Search options
     * @param {number} options.nprobe - Number of partitions to search (for ANN)
     * @param {boolean} options.useIndex - Whether to use ANN index if available
     * @returns {Promise<{indices: number[], scores: number[], useIndex: boolean}>}
     */
    async vectorSearch(colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
        const { nprobe = 10, useIndex = true } = options;

        const info = await this.getVectorInfo(colIdx);
        if (info.dimension === 0 || info.dimension !== queryVec.length) {
            throw new Error(`Dimension mismatch: query=${queryVec.length}, column=${info.dimension}`);
        }

        // Require IVF index - no brute force fallback
        if (!this.hasIndex()) {
            throw new Error('No IVF index found. Vector search requires an IVF index for efficient querying.');
        }

        if (this._ivfIndex.dimension !== queryVec.length) {
            throw new Error(`Query dimension (${queryVec.length}) does not match index dimension (${this._ivfIndex.dimension}).`);
        }

        return await this._vectorSearchWithIndex(colIdx, queryVec, topK, nprobe, onProgress);
    }

    /**
     * Vector search using IVF index (ANN).
     * Fetches row IDs from auxiliary.idx for nearest partitions,
     * then looks up original vectors by fragment/offset.
     * @private
     */
    async _vectorSearchWithIndex(colIdx, queryVec, topK, nprobe, onProgress) {
        const dim = queryVec.length;

        // Find nearest partitions using centroids
        if (onProgress) onProgress(0, 100);
        const partitions = this._ivfIndex.findNearestPartitions(queryVec, nprobe);
        const estimatedRows = this._ivfIndex.getPartitionRowCount(partitions);

        console.log(`[IVFSearch] Searching ${partitions.length} partitions (~${estimatedRows.toLocaleString()} rows)`);

        // Try to fetch row IDs from auxiliary.idx
        const rowIdMappings = await this._ivfIndex.fetchPartitionRowIds(partitions);

        if (rowIdMappings && rowIdMappings.length > 0) {
            // Use proper row ID mapping from auxiliary.idx
            console.log(`[IVFSearch] Fetched ${rowIdMappings.length} row ID mappings`);
            return await this._searchWithRowIdMappings(colIdx, queryVec, topK, rowIdMappings, onProgress);
        }

        // No fallback - require proper row ID mapping
        throw new Error('Failed to fetch row IDs from IVF index. Dataset may be missing auxiliary.idx or ivf_partitions.bin.');
    }

    /**
     * Search using proper row ID mappings from auxiliary.idx.
     * Groups row IDs by fragment and fetches vectors efficiently.
     * Uses WebGPU (if available) or WASM SIMD for batch cosine similarity.
     * @private
     */
    async _searchWithRowIdMappings(colIdx, queryVec, topK, rowIdMappings, onProgress) {
        const dim = queryVec.length;

        // Group row IDs by fragment for efficient batch fetching
        const byFragment = new Map();
        for (const mapping of rowIdMappings) {
            if (!byFragment.has(mapping.fragId)) {
                byFragment.set(mapping.fragId, []);
            }
            byFragment.get(mapping.fragId).push(mapping.rowOffset);
        }

        console.log(`[IVFSearch] Fetching from ${byFragment.size} fragments`);

        // Collect all vectors and their indices first
        const allVectors = [];
        const allIndices = [];
        let processed = 0;
        const total = rowIdMappings.length;

        // Fetch all vectors
        for (const [fragId, offsets] of byFragment) {
            if (onProgress) onProgress(processed, total);

            const vectors = await this.readVectorsAtIndices(colIdx, offsets);

            for (let i = 0; i < offsets.length; i++) {
                const vec = vectors[i];
                if (vec && vec.length === dim) {
                    allVectors.push(vec);
                    // Reconstruct global row index
                    allIndices.push(fragId * 50000 + offsets[i]);
                }
                processed++;
            }
        }

        // Try WebGPU first, fallback to WASM SIMD
        let scores;
        if (webgpuAccelerator.isAvailable()) {
            console.log(`[IVFSearch] Computing similarity for ${allVectors.length} vectors via WebGPU`);
            scores = await webgpuAccelerator.batchCosineSimilarity(queryVec, allVectors, true);
        }

        if (!scores) {
            console.log(`[IVFSearch] Computing similarity for ${allVectors.length} vectors via WASM SIMD`);
            scores = this.lanceql.batchCosineSimilarity(queryVec, allVectors, true);
        }

        // Find top-k
        const topResults = [];
        for (let i = 0; i < scores.length; i++) {
            const score = scores[i];
            const idx = allIndices[i];

            if (topResults.length < topK) {
                topResults.push({ idx, score });
                topResults.sort((a, b) => b.score - a.score);
            } else if (score > topResults[topK - 1].score) {
                topResults[topK - 1] = { idx, score };
                topResults.sort((a, b) => b.score - a.score);
            }
        }

        if (onProgress) onProgress(total, total);

        return {
            indices: topResults.map(r => r.idx),
            scores: topResults.map(r => r.score),
            usedIndex: true,
            searchedRows: allVectors.length
        };
    }

    // NOTE: _searchWithEstimatedPartitions and _vectorSearchBruteForce have been removed.
    // All vector search now requires IVF index with proper partition mapping.
    // Use LanceDataset for multi-fragment datasets with ivf_partitions.bin.

    /**
     * Read all vectors from a column as a flat Float32Array.
     * Used for worker-based parallel search.
     * Handles multi-page columns by fetching and combining all pages.
     * @param {number} colIdx - Vector column index
     * @returns {Promise<Float32Array>} - Flattened vector data [numRows * dim]
     */
    async readVectorColumn(colIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const metaInfo = this._parseColumnMeta(new Uint8Array(colMeta));

        if (!metaInfo.pages || metaInfo.pages.length === 0 || metaInfo.rows === 0) {
            return new Float32Array(0);
        }

        // Calculate dimension from first page
        const firstPage = metaInfo.pages[0];
        const dataIdx = firstPage.sizes.length > 1 ? 1 : 0;
        const firstPageSize = firstPage.sizes[dataIdx] || 0;
        const firstPageRows = firstPage.rows || 0;

        if (firstPageRows === 0 || firstPageSize === 0) {
            return new Float32Array(0);
        }

        const dim = Math.floor(firstPageSize / (firstPageRows * 4));
        if (dim === 0) {
            return new Float32Array(0);
        }

        const totalRows = metaInfo.rows;
        const result = new Float32Array(totalRows * dim);

        // Fetch each page in parallel
        const pagePromises = metaInfo.pages.map(async (page, pageIdx) => {
            const pageDataIdx = page.sizes.length > 1 ? 1 : 0;
            const pageOffset = page.offsets[pageDataIdx] || 0;
            const pageSize = page.sizes[pageDataIdx] || 0;

            if (pageSize === 0) return { pageIdx, data: new Float32Array(0), rows: 0 };

            const data = await this.fetchRange(pageOffset, pageOffset + pageSize - 1);
            // data is ArrayBuffer from fetchRange, create Float32Array view directly
            const floatData = new Float32Array(data);
            return {
                pageIdx,
                data: floatData,
                rows: page.rows
            };
        });

        const pageResults = await Promise.all(pagePromises);

        // Combine pages in order
        let offset = 0;
        for (const pageResult of pageResults.sort((a, b) => a.pageIdx - b.pageIdx)) {
            result.set(pageResult.data, offset);
            offset += pageResult.rows * dim;
        }

        return result;
    }

    /**
     * Read rows from this Lance file with pagination.
     * @param {Object} options - Query options
     * @param {number} options.offset - Starting row offset
     * @param {number} options.limit - Maximum rows to return
     * @param {number[]} options.columns - Column indices to read (optional, null = all)
     * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
     */
    async readRows({ offset = 0, limit = 50, columns = null } = {}) {
        // Determine column indices to read
        const colIndices = columns || Array.from({ length: this._numColumns }, (_, i) => i);

        // Get total row count from first column
        const totalRows = await this.getRowCount(0);

        // Clamp offset and limit
        const actualOffset = Math.min(offset, totalRows);
        const actualLimit = Math.min(limit, totalRows - actualOffset);

        if (actualLimit <= 0) {
            return {
                columns: colIndices.map(() => []),
                columnNames: this.columnNames.slice(0, colIndices.length),
                total: totalRows
            };
        }

        // Generate indices for the requested rows
        const indices = Array.from({ length: actualLimit }, (_, i) => actualOffset + i);

        // Detect all column types first
        const columnTypes = await this.detectColumnTypes();

        // Read each column in parallel
        const columnPromises = colIndices.map(async (colIdx) => {
            const type = columnTypes[colIdx] || 'unknown';

            try {
                switch (type) {
                    case 'string':
                    case 'utf8':
                    case 'large_utf8':
                        return await this.readStringsAtIndices(colIdx, indices);

                    case 'int64':
                        return Array.from(await this.readInt64AtIndices(colIdx, indices));

                    case 'int32':
                        return Array.from(await this.readInt32AtIndices(colIdx, indices));

                    case 'int16':
                        return Array.from(await this.readInt16AtIndices(colIdx, indices));

                    case 'uint8':
                        return Array.from(await this.readUint8AtIndices(colIdx, indices));

                    case 'float64':
                    case 'double':
                        return Array.from(await this.readFloat64AtIndices(colIdx, indices));

                    case 'float32':
                    case 'float':
                        return Array.from(await this.readFloat32AtIndices(colIdx, indices));

                    case 'bool':
                    case 'boolean':
                        return await this.readBoolAtIndices(colIdx, indices);

                    case 'fixed_size_list':
                    case 'vector':
                        // For vectors, return as nested arrays
                        const vectors = await this.readVectorsAtIndices(colIdx, indices);
                        return Array.isArray(vectors) ? vectors : Array.from(vectors);

                    default:
                        // Try as string for unknown types
                        console.warn(`[LanceQL] Unknown column type: ${type}, trying as string`);
                        return await this.readStringsAtIndices(colIdx, indices);
                }
            } catch (e) {
                console.warn(`[LanceQL] Error reading column ${colIdx} (${type}):`, e.message);
                return indices.map(() => null);
            }
        });

        const columnsData = await Promise.all(columnPromises);

        return {
            columns: columnsData,
            columnNames: colIndices.map(i => this.columnNames[i] || `column_${i}`),
            total: totalRows
        };
    }
}


export { RemoteLanceFile };
