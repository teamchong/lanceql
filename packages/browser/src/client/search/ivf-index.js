/**
 * IVFIndex - IVF-PQ vector index for approximate nearest neighbor search
 */

class IVFIndex {
    constructor() {
        this.centroids = null;       // Float32Array of centroids (numPartitions x dimension)
        this.numPartitions = 0;      // Number of IVF partitions
        this.dimension = 0;          // Vector dimension
        this.partitionOffsets = [];  // Byte offset of each partition in the data
        this.partitionLengths = [];  // Number of rows in each partition
        this.metricType = 'cosine';  // Distance metric (cosine, l2, dot)

        // Custom partition index (ivf_partitions.bin)
        this.partitionIndexUrl = null;  // URL to ivf_partitions.bin
        this.partitionStarts = null;    // Uint32Array[257] - cumulative row counts
        this.hasPartitionIndex = false; // Whether partition index is loaded

        // Prefetched row IDs cache - avoids HTTP requests during search
        this._rowIdCache = null;  // Map<partitionIdx, Array<{fragId, rowOffset}>>
        this._rowIdCacheReady = false;
    }

    /**
     * Try to load IVF index from a Lance dataset.
     * Index structure: dataset.lance/_indices/<uuid>/index.idx
     * @param {string} datasetBaseUrl - Base URL of dataset (e.g., https://host/data.lance)
     * @returns {Promise<IVFIndex|null>}
     */
    static async tryLoad(datasetBaseUrl) {
        if (!datasetBaseUrl) return null;

        try {
            // Find latest manifest version
            const manifestVersion = await IVFIndex._findLatestManifestVersion(datasetBaseUrl);
            console.log(`[IVFIndex] Manifest version: ${manifestVersion}`);
            if (!manifestVersion) return null;

            const manifestUrl = `${datasetBaseUrl}/_versions/${manifestVersion}.manifest`;
            const manifestResp = await fetch(manifestUrl);
            if (!manifestResp.ok) {
                console.log(`[IVFIndex] Failed to fetch manifest: ${manifestResp.status}`);
                return null;
            }

            const manifestData = await manifestResp.arrayBuffer();
            const indexInfo = IVFIndex._parseManifestForIndex(new Uint8Array(manifestData));
            console.log(`[IVFIndex] Index info:`, indexInfo);

            if (!indexInfo || !indexInfo.uuid) {
                // No vector index found in manifest
                console.log('[IVFIndex] No index UUID found in manifest');
                return null;
            }

            console.log(`[IVFIndex] Found index UUID: ${indexInfo.uuid}`);

            // Fetch the index file (contains centroids)
            const indexUrl = `${datasetBaseUrl}/_indices/${indexInfo.uuid}/index.idx`;
            const indexResp = await fetch(indexUrl);
            if (!indexResp.ok) {
                console.warn('[IVFIndex] index.idx not found');
                return null;
            }

            const indexData = await indexResp.arrayBuffer();
            const index = IVFIndex._parseIndexFile(new Uint8Array(indexData), indexInfo);

            if (!index) return null;

            // Store auxiliary URL for later partition data fetching
            index.auxiliaryUrl = `${datasetBaseUrl}/_indices/${indexInfo.uuid}/auxiliary.idx`;
            index.datasetBaseUrl = datasetBaseUrl;

            // Fetch auxiliary.idx metadata (footer + partition info)
            // We only need the last ~13MB which has the partition metadata
            try {
                await index._loadAuxiliaryMetadata();
            } catch (e) {
                console.warn('[IVFIndex] Failed to load auxiliary metadata:', e);
            }

            console.log(`[IVFIndex] Loaded: ${index.numPartitions} partitions, dim=${index.dimension}`);
            if (index.partitionLengths.length > 0) {
                const totalRows = index.partitionLengths.reduce((a, b) => a + b, 0);
                console.log(`[IVFIndex] Partition info: ${totalRows.toLocaleString()} total rows`);
            }

            // Try to load custom partition index (ivf_partitions.bin)
            try {
                await index._loadPartitionIndex();
            } catch (e) {
                console.warn('[IVFIndex] Failed to load partition index:', e);
            }

            // Prefetch all row IDs for fast search (no HTTP during search)
            try {
                await index.prefetchAllRowIds();
            } catch (e) {
                console.warn('[IVFIndex] Failed to prefetch row IDs:', e);
            }

            return index;
        } catch (e) {
            console.warn('[IVFIndex] Failed to load:', e);
            return null;
        }
    }

    /**
     * Load partition-organized vectors index from ivf_vectors.bin.
     * This file contains:
     *   - Header: 257 uint64 byte offsets (2056 bytes)
     *   - Per partition: [row_count: uint32][row_ids: uint32 × n][vectors: float32 × n × 384]
     * @private
     */
    async _loadPartitionIndex() {
        const url = `${this.datasetBaseUrl}/ivf_vectors.bin`;
        this.partitionVectorsUrl = url;

        // Fetch header (257 uint64s = 2056 bytes)
        const headerResp = await fetch(url, {
            headers: { 'Range': 'bytes=0-2055' }
        });
        if (!headerResp.ok) {
            console.log('[IVFIndex] ivf_vectors.bin not found, IVF search disabled');
            return;
        }

        const headerData = await headerResp.arrayBuffer();
        // Parse as BigUint64Array then convert to regular numbers
        const bigOffsets = new BigUint64Array(headerData);
        this.partitionOffsets = Array.from(bigOffsets, n => Number(n));

        this.hasPartitionIndex = true;
        console.log(`[IVFIndex] Loaded partition vectors index: 256 partitions`);
    }

    /**
     * Fetch partition data (row IDs and vectors) directly from ivf_vectors.bin.
     * Uses OPFS cache for instant subsequent searches.
     * Each partition contains: [row_count: uint32][row_ids: uint32 × n][vectors: float32 × n × dim]
     * @param {number[]} partitionIndices - Partition indices to fetch
     * @param {number} dim - Vector dimension (default 384)
     * @param {function} onProgress - Progress callback (bytesLoaded, totalBytes)
     * @returns {Promise<{rowIds: number[], vectors: Float32Array[]}>}
     */
    async fetchPartitionData(partitionIndices, dim = 384, onProgress = null) {
        if (!this.hasPartitionIndex || !this.partitionVectorsUrl) {
            return null;
        }

        const allRowIds = [];
        const allVectors = [];
        let totalBytesToFetch = 0;
        let bytesLoaded = 0;

        // Separate cached vs uncached partitions
        const uncachedPartitions = [];
        const cachedResults = new Map();

        for (const p of partitionIndices) {
            // Check in-memory cache first
            if (this._partitionCache?.has(p)) {
                cachedResults.set(p, this._partitionCache.get(p));
            } else {
                uncachedPartitions.push(p);
                const startOffset = this.partitionOffsets[p];
                const endOffset = this.partitionOffsets[p + 1];
                totalBytesToFetch += endOffset - startOffset;
            }
        }

        if (uncachedPartitions.length === 0) {
            console.log(`[IVFIndex] All ${partitionIndices.length} partitions from cache`);
            // All from cache
            for (const p of partitionIndices) {
                const result = cachedResults.get(p);
                allRowIds.push(...result.rowIds);
                allVectors.push(...result.vectors);
            }
            if (onProgress) onProgress(100, 100);
            return { rowIds: allRowIds, vectors: allVectors };
        }

        console.log(`[IVFIndex] Fetching ${uncachedPartitions.length}/${partitionIndices.length} partitions, ${(totalBytesToFetch / 1024 / 1024).toFixed(1)} MB`);

        // Initialize partition cache if needed
        if (!this._partitionCache) {
            this._partitionCache = new Map();
        }

        // Fetch uncached partitions in parallel (max 6 concurrent for speed)
        const PARALLEL_LIMIT = 6;
        for (let i = 0; i < uncachedPartitions.length; i += PARALLEL_LIMIT) {
            const batch = uncachedPartitions.slice(i, i + PARALLEL_LIMIT);

            const results = await Promise.all(batch.map(async (p) => {
                const startOffset = this.partitionOffsets[p];
                const endOffset = this.partitionOffsets[p + 1];
                const byteSize = endOffset - startOffset;

                try {
                    const resp = await fetch(this.partitionVectorsUrl, {
                        headers: { 'Range': `bytes=${startOffset}-${endOffset - 1}` }
                    });
                    if (!resp.ok) {
                        console.warn(`[IVFIndex] Partition ${p} fetch failed: ${resp.status}`);
                        return { p, rowIds: [], vectors: [] };
                    }

                    const data = await resp.arrayBuffer();
                    const view = new DataView(data);

                    // Parse: [row_count: uint32][row_ids: uint32 × n][vectors: float32 × n × dim]
                    const rowCount = view.getUint32(0, true);  // little-endian
                    const rowIdsStart = 4;
                    const rowIdsEnd = rowIdsStart + rowCount * 4;
                    const vectorsStart = rowIdsEnd;

                    const rowIds = new Uint32Array(data.slice(rowIdsStart, rowIdsEnd));
                    const vectorsFlat = new Float32Array(data.slice(vectorsStart));

                    // Split flat vectors into individual arrays
                    const vectors = [];
                    for (let j = 0; j < rowCount; j++) {
                        vectors.push(vectorsFlat.slice(j * dim, (j + 1) * dim));
                    }

                    bytesLoaded += byteSize;
                    if (onProgress) onProgress(bytesLoaded, totalBytesToFetch);

                    return { p, rowIds: Array.from(rowIds), vectors };
                } catch (e) {
                    console.warn(`[IVFIndex] Error fetching partition ${p}:`, e);
                    return { p, rowIds: [], vectors: [] };
                }
            }));

            // Cache results and collect
            for (const result of results) {
                const { p, rowIds, vectors } = result;
                // Cache in memory for subsequent searches
                this._partitionCache.set(p, { rowIds, vectors });
                cachedResults.set(p, { rowIds, vectors });
            }
        }

        // Collect all results in original order
        for (const p of partitionIndices) {
            const result = cachedResults.get(p);
            if (result) {
                allRowIds.push(...result.rowIds);
                allVectors.push(...result.vectors);
            }
        }

        console.log(`[IVFIndex] Loaded ${allRowIds.length.toLocaleString()} vectors from ${partitionIndices.length} partitions`);
        return { rowIds: allRowIds, vectors: allVectors };
    }

    /**
     * Find latest manifest version using binary search.
     * @private
     */
    static async _findLatestManifestVersion(baseUrl) {
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
     * Load partition metadata from auxiliary.idx.
     * Uses HTTP range request to fetch only the metadata section.
     * @private
     */
    async _loadAuxiliaryMetadata() {
        // Fetch file size first
        let headResp;
        try {
            headResp = await fetch(this.auxiliaryUrl, { method: 'HEAD' });
        } catch (e) {
            console.warn('[IVFIndex] HEAD request failed for auxiliary.idx:', e.message);
            return;
        }
        if (!headResp.ok) return;

        const fileSize = parseInt(headResp.headers.get('content-length'));
        if (!fileSize) return;

        // Fetch footer (last 40 bytes) to get metadata locations
        const footerResp = await fetch(this.auxiliaryUrl, {
            headers: { 'Range': `bytes=${fileSize - 40}-${fileSize - 1}` }
        });
        if (!footerResp.ok) return;

        const footer = new Uint8Array(await footerResp.arrayBuffer());
        const view = new DataView(footer.buffer, footer.byteOffset);

        // Parse Lance footer (40 bytes)
        // Bytes 0-7: column_meta_start
        // Bytes 8-15: column_meta_offsets_start
        // Bytes 16-23: global_buff_offsets_start
        // Bytes 24-27: num_global_buffers
        // Bytes 28-31: num_columns
        // Bytes 32-33: major_version
        // Bytes 34-35: minor_version
        // Bytes 36-39: magic "LANC"
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
        const gboResp = await fetch(this.auxiliaryUrl, {
            headers: { 'Range': `bytes=${globalBuffOffsetsStart}-${globalBuffOffsetsStart + gboSize - 1}` }
        });
        if (!gboResp.ok) return;

        const gboData = new Uint8Array(await gboResp.arrayBuffer());
        const gboView = new DataView(gboData.buffer, gboData.byteOffset);

        // Global buffer offsets are stored as [offset, length] pairs
        // Each buffer has: offset (8 bytes) + length (8 bytes) = 16 bytes per buffer
        const buffers = [];
        for (let i = 0; i < numGlobalBuffers; i++) {
            const offset = Number(gboView.getBigUint64(i * 16, true));
            const length = Number(gboView.getBigUint64(i * 16 + 8, true));
            buffers.push({ offset, length });
        }

        console.log(`[IVFIndex] Buffers:`, buffers);

        // Buffer 1 contains row IDs (_rowid column data)
        // Buffer 2 contains PQ codes (__pq_code column data)
        // We need buffer 1 for row ID lookups
        if (buffers.length < 2) return;

        // Store buffer info for later use
        this._auxBuffers = buffers;
        this._auxFileSize = fileSize;

        // Now we need to fetch partition metadata from column metadata
        // The auxiliary.idx stores _rowid and __pq_code columns
        // Partition info (offsets, lengths) is in the column metadata section
        // For now, we'll compute partition info from the row ID buffer
        // Each partition's row IDs are stored contiguously

        // We need to parse column metadata to get partition boundaries
        // Column metadata is at col_meta_start, with offsets at col_meta_off_start
        const colMetaOffResp = await fetch(this.auxiliaryUrl, {
            headers: { 'Range': `bytes=${colMetaOffsetsStart}-${globalBuffOffsetsStart - 1}` }
        });
        if (!colMetaOffResp.ok) return;

        const colMetaOffData = new Uint8Array(await colMetaOffResp.arrayBuffer());
        // Parse column offset entries (16 bytes each: 8 byte pos + 8 byte len)
        // We have 2 columns: _rowid and __pq_code
        if (colMetaOffData.length >= 32) {
            const colView = new DataView(colMetaOffData.buffer, colMetaOffData.byteOffset);
            const col0Pos = Number(colView.getBigUint64(0, true));
            const col0Len = Number(colView.getBigUint64(8, true));
            console.log(`[IVFIndex] Column 0 (_rowid) metadata at ${col0Pos}, len=${col0Len}`);

            // Fetch column 0 metadata to get page info
            const col0MetaResp = await fetch(this.auxiliaryUrl, {
                headers: { 'Range': `bytes=${col0Pos}-${col0Pos + col0Len - 1}` }
            });
            if (col0MetaResp.ok) {
                const col0Meta = new Uint8Array(await col0MetaResp.arrayBuffer());
                this._parseColumnMetaForPartitions(col0Meta);
            }
        }
    }

    /**
     * Parse column metadata to extract partition (page) boundaries.
     * @private
     */
    _parseColumnMetaForPartitions(bytes) {
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
                    const page = this._parsePageInfo(content);
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
        // Note: partition info should come from index.idx, not column pages
        // Column pages are how data is stored, partitions are the IVF clusters
        this._columnPages = pages;

        // Calculate total rows for verification
        let totalRows = 0;
        for (const page of pages) {
            totalRows += page.numRows;
        }
        console.log(`[IVFIndex] Column has ${totalRows} total rows`);
    }

    /**
     * Parse PageInfo protobuf.
     * @private
     */
    _parsePageInfo(bytes) {
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
     * @private
     */
    _parseAuxiliaryPartitionInfo(bytes) {
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
                    if (offsets.length === this.numPartitions) {
                        this.partitionOffsets = offsets;
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
                    if (lengths.length === this.numPartitions) {
                        this.partitionLengths = lengths;
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

    /**
     * Prefetch ALL row IDs from auxiliary.idx into memory.
     * This is called once during index loading to avoid HTTP requests during search.
     * @returns {Promise<void>}
     */
    async prefetchAllRowIds() {
        if (!this.auxiliaryUrl || !this._auxBufferOffsets) {
            console.log('[IVFIndex] No auxiliary.idx available for prefetch');
            return;
        }

        if (this._rowIdCacheReady) {
            console.log('[IVFIndex] Row IDs already prefetched');
            return;
        }

        const totalRows = this.partitionLengths.reduce((a, b) => a + b, 0);
        if (totalRows === 0) {
            console.log('[IVFIndex] No rows to prefetch');
            return;
        }

        console.log(`[IVFIndex] Prefetching ${totalRows.toLocaleString()} row IDs...`);
        const startTime = performance.now();

        const dataStart = this._auxBufferOffsets[1];
        const totalBytes = totalRows * 8;

        try {
            // Fetch ALL row IDs in a single request
            const resp = await fetch(this.auxiliaryUrl, {
                headers: { 'Range': `bytes=${dataStart}-${dataStart + totalBytes - 1}` }
            });

            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            const data = new Uint8Array(await resp.arrayBuffer());
            const view = new DataView(data.buffer, data.byteOffset);

            // Parse and organize by partition
            this._rowIdCache = new Map();
            let globalRowIdx = 0;

            for (let p = 0; p < this.partitionLengths.length; p++) {
                const numRows = this.partitionLengths[p];
                const partitionRows = [];

                for (let i = 0; i < numRows; i++) {
                    const rowId = Number(view.getBigUint64(globalRowIdx * 8, true));
                    const fragId = Math.floor(rowId / 0x100000000);
                    const rowOffset = rowId % 0x100000000;
                    partitionRows.push({ fragId, rowOffset });
                    globalRowIdx++;
                }

                this._rowIdCache.set(p, partitionRows);
            }

            this._rowIdCacheReady = true;
            const elapsed = performance.now() - startTime;
            console.log(`[IVFIndex] Prefetched ${totalRows.toLocaleString()} row IDs in ${elapsed.toFixed(0)}ms (${(totalBytes / 1024 / 1024).toFixed(1)}MB)`);
        } catch (e) {
            console.warn('[IVFIndex] Failed to prefetch row IDs:', e);
        }
    }

    /**
     * Fetch row IDs for specified partitions.
     * Uses prefetched cache if available (instant), otherwise fetches from network.
     *
     * @param {number[]} partitionIndices - Partition indices to fetch
     * @returns {Promise<Array<{fragId: number, rowOffset: number}>>}
     */
    async fetchPartitionRowIds(partitionIndices) {
        // Fast path: use prefetched cache
        if (this._rowIdCacheReady && this._rowIdCache) {
            const results = [];
            for (const p of partitionIndices) {
                const cached = this._rowIdCache.get(p);
                if (cached) {
                    for (const row of cached) {
                        results.push({ ...row, partition: p });
                    }
                }
            }
            return results;
        }

        // Slow path: fetch from network (fallback if prefetch failed)
        if (!this.auxiliaryUrl || !this._auxBufferOffsets) {
            return null;
        }

        const rowRanges = [];
        for (const p of partitionIndices) {
            if (p < this.partitionOffsets.length) {
                const startRow = this.partitionOffsets[p];
                const numRows = this.partitionLengths[p];
                rowRanges.push({ partition: p, startRow, numRows });
            }
        }

        if (rowRanges.length === 0) return [];

        const results = [];
        const dataStart = this._auxBufferOffsets[1];

        for (const range of rowRanges) {
            const byteStart = dataStart + range.startRow * 8;
            const byteEnd = byteStart + range.numRows * 8 - 1;

            try {
                const resp = await fetch(this.auxiliaryUrl, {
                    headers: { 'Range': `bytes=${byteStart}-${byteEnd}` }
                });

                if (!resp.ok) continue;

                const data = new Uint8Array(await resp.arrayBuffer());
                const view = new DataView(data.buffer, data.byteOffset);

                for (let i = 0; i < range.numRows; i++) {
                    const rowId = Number(view.getBigUint64(i * 8, true));
                    const fragId = Math.floor(rowId / 0x100000000);
                    const rowOffset = rowId % 0x100000000;
                    results.push({ fragId, rowOffset, partition: range.partition });
                }
            } catch (e) {
                console.warn(`[IVFIndex] Error fetching partition ${range.partition}:`, e);
            }
        }

        return results;
    }

    /**
     * Get estimated number of rows to search for given partitions.
     */
    getPartitionRowCount(partitionIndices) {
        let total = 0;
        for (const p of partitionIndices) {
            if (p < this.partitionLengths.length) {
                total += this.partitionLengths[p];
            }
        }
        return total;
    }

    /**
     * Parse manifest to find vector index info.
     * @private
     */
    static _parseManifestForIndex(bytes) {
        // Manifest structure:
        // - Chunk 1: 4 bytes len + content (index metadata in field 1)
        // - Chunk 2: 4 bytes len + content (full manifest with schema + fragments)
        // - Footer (16 bytes)
        //
        // Index info is in CHUNK 1, field 1 (IndexMetadata repeated)

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
                    const parsed = IVFIndex._parseIndexMetadata(content);
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
     * @private
     */
    static _parseIndexMetadata(bytes) {
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
                    uuid = IVFIndex._parseUuid(content);
                }
            } else if (wireType === 0) {
                const val = readVarint();
                if (fieldNum === 2) {
                    // fields (repeated int32) - but packed, so single value here
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
     * @private
     */
    static _parseUuid(bytes) {
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
     * Index file contains VectorIndex protobuf with IVF stage.
     * IVF message structure:
     *   field 1: repeated float centroids (deprecated)
     *   field 2: repeated uint64 offsets - byte offset of each partition
     *   field 3: repeated uint32 lengths - number of records per partition
     *   field 4: Tensor centroids_tensor - centroids as tensor
     *   field 5: optional double loss
     * @private
     */
    static _parseIndexFile(bytes, indexInfo) {
        const index = new IVFIndex();

        // Try to find and parse IVF message within the file
        // The file may have nested protobuf structures
        const ivfData = IVFIndex._findIVFMessage(bytes);

        if (ivfData) {
            if (ivfData.centroids) {
                index.centroids = ivfData.centroids.data;
                index.numPartitions = ivfData.centroids.numPartitions;
                index.dimension = ivfData.centroids.dimension;
            }
            if (ivfData.offsets && ivfData.offsets.length > 0) {
                index.partitionOffsets = ivfData.offsets;
                // Loaded partition offsets
            }
            if (ivfData.lengths && ivfData.lengths.length > 0) {
                index.partitionLengths = ivfData.lengths;
                // Loaded partition lengths
            }

            // Index centroids loaded successfully
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
                        const centroids = IVFIndex._tryParseCentroids(content);
                        if (centroids) {
                            index.centroids = centroids.data;
                            index.numPartitions = centroids.numPartitions;
                            index.dimension = centroids.dimension;
                            // Loaded IVF centroids via fallback parsing
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
     * Recursively searches nested protobuf messages.
     * @private
     */
    static _findIVFMessage(bytes) {
        // IVF message fields:
        // field 2: repeated uint64 offsets (packed)
        // field 3: repeated uint32 lengths (packed)
        // field 4: Tensor centroids_tensor

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

        const readFixed64 = () => {
            if (pos + 8 > bytes.length) return 0n;
            const view = new DataView(bytes.buffer, bytes.byteOffset + pos, 8);
            pos += 8;
            return view.getBigUint64(0, true);
        };

        const readFixed32 = () => {
            if (pos + 4 > bytes.length) return 0;
            const view = new DataView(bytes.buffer, bytes.byteOffset + pos, 4);
            pos += 4;
            return view.getUint32(0, true);
        };

        while (pos < bytes.length - 4) {
            const startPos = pos;
            const tag = readVarint();
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 2) {
                // Length-delimited field
                const len = readVarint();
                if (len > bytes.length - pos || len < 0) {
                    pos = startPos + 1;
                    continue;
                }

                const content = bytes.slice(pos, pos + len);
                pos += len;

                if (fieldNum === 2) {
                    // offsets - packed uint64
                    // Could be packed fixed64 or packed varint
                    if (len % 8 === 0 && len > 0) {
                        // Try as packed fixed64
                        const numOffsets = len / 8;
                        const view = new DataView(content.buffer, content.byteOffset, len);
                        for (let i = 0; i < numOffsets; i++) {
                            offsets.push(Number(view.getBigUint64(i * 8, true)));
                        }
                        // Parsed partition offsets
                    }
                } else if (fieldNum === 3) {
                    // lengths - packed uint32
                    if (len % 4 === 0 && len > 0) {
                        // Try as packed fixed32
                        const numLengths = len / 4;
                        const view = new DataView(content.buffer, content.byteOffset, len);
                        for (let i = 0; i < numLengths; i++) {
                            lengths.push(view.getUint32(i * 4, true));
                        }
                        // Parsed partition lengths (fixed32)
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
                        // Parsed partition lengths (varint)
                    }
                } else if (fieldNum === 4) {
                    // centroids_tensor
                    centroids = IVFIndex._tryParseCentroids(content);
                } else if (len > 100) {
                    // Recursively search nested messages
                    const nested = IVFIndex._findIVFMessage(content);
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
                // Unknown wire type, skip byte
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
     * @private
     */
    static _tryParseCentroids(bytes) {
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

    /**
     * Find the nearest partitions to a query vector.
     * @param {Float32Array} queryVec - Query vector
     * @param {number} nprobe - Number of partitions to search
     * @returns {number[]} - Indices of nearest partitions
     */
    findNearestPartitions(queryVec, nprobe = 10) {
        if (!this.centroids || queryVec.length !== this.dimension) {
            return [];
        }

        nprobe = Math.min(nprobe, this.numPartitions);

        // Compute distance to each centroid
        const distances = new Array(this.numPartitions);

        for (let p = 0; p < this.numPartitions; p++) {
            const centroidStart = p * this.dimension;

            // Cosine similarity (or L2 distance based on metricType)
            let dot = 0, normA = 0, normB = 0;
            for (let i = 0; i < this.dimension; i++) {
                const a = queryVec[i];
                const b = this.centroids[centroidStart + i];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }

            const denom = Math.sqrt(normA) * Math.sqrt(normB);
            distances[p] = { idx: p, score: denom === 0 ? 0 : dot / denom };
        }

        // Sort by similarity (descending) and take top nprobe
        distances.sort((a, b) => b.score - a.score);
        return distances.slice(0, nprobe).map(d => d.idx);
    }
}


export { IVFIndex };
