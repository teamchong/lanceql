/**
 * RemoteLanceDataset - Multi-fragment remote dataset
 */

class RemoteLanceDataset {
    constructor(lanceql, baseUrl) {
        this.lanceql = lanceql;
        this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
        this._fragments = [];
        this._schema = null;
        this._totalRows = 0;
        this._numColumns = 0;
        this._onFetch = null;
        this._fragmentFiles = new Map(); // Cache of opened RemoteLanceFile per fragment
        this._isRemote = true;
        this._ivfIndex = null; // IVF index for ANN search
        this._deletedRows = new Map(); // Cache of deleted row Sets per fragment index
    }

    /**
     * Open a remote Lance dataset.
     * @param {LanceQL} lanceql - LanceQL instance
     * @param {string} baseUrl - Base URL to the dataset
     * @param {object} [options] - Options
     * @param {number} [options.version] - Specific version to load (time-travel)
     * @param {boolean} [options.prefetch] - Prefetch fragment metadata (default: true for small datasets)
     * @returns {Promise<RemoteLanceDataset>}
     */
    static async open(lanceql, baseUrl, options = {}) {
        const dataset = new RemoteLanceDataset(lanceql, baseUrl);
        dataset._requestedVersion = options.version || null;

        // Try to load from cache first (unless skipCache is true)
        const cacheKey = options.version ? `${baseUrl}@v${options.version}` : baseUrl;
        if (!options.skipCache) {
            const cached = await metadataCache.get(cacheKey);
            if (cached && cached.schema && cached.fragments) {
                console.log(`[LanceQL Dataset] Using cached metadata for ${baseUrl}`);
                dataset._schema = cached.schema;
                dataset._fragments = cached.fragments;
                dataset._numColumns = cached.schema.length;
                dataset._totalRows = cached.fragments.reduce((sum, f) => sum + f.numRows, 0);
                dataset._version = cached.version;
                dataset._columnTypes = cached.columnTypes || null;
                dataset._fromCache = true;
            }
        }

        // If not cached, try sidecar first, then manifest
        if (!dataset._fromCache) {
            // Try to load .meta.json sidecar (faster, pre-calculated)
            const sidecarLoaded = await dataset._tryLoadSidecar();

            if (!sidecarLoaded) {
                // Fall back to parsing manifest
                await dataset._loadManifest();
            }

            // Cache the metadata for next time
            metadataCache.set(cacheKey, {
                schema: dataset._schema,
                fragments: dataset._fragments,
                version: dataset._version,
                columnTypes: dataset._columnTypes || null
            }).catch(() => {}); // Don't block on cache errors
        }

        await dataset._tryLoadIndex();

        // Prefetch fragment metadata for faster first query
        // Default: prefetch if <= 5 fragments
        const shouldPrefetch = options.prefetch ?? (dataset._fragments.length <= 5);
        if (shouldPrefetch && dataset._fragments.length > 0) {
            dataset._prefetchFragments();
        }

        return dataset;
    }

    /**
     * Try to load sidecar manifest (.meta.json) for faster startup.
     * @returns {Promise<boolean>} True if sidecar was loaded successfully
     * @private
     */
    async _tryLoadSidecar() {
        try {
            const sidecarUrl = `${this.baseUrl}/.meta.json`;
            const response = await fetch(sidecarUrl);

            if (!response.ok) {
                return false;
            }

            const sidecar = await response.json();

            // Validate sidecar format
            if (!sidecar.schema || !sidecar.fragments) {
                console.warn('[LanceQL Dataset] Invalid sidecar format');
                return false;
            }

            console.log(`[LanceQL Dataset] Loaded sidecar manifest`);

            // Convert sidecar schema to internal format
            this._schema = sidecar.schema.map(col => ({
                name: col.name,
                id: col.index,
                type: col.type
            }));

            // Convert sidecar fragments to internal format
            this._fragments = sidecar.fragments.map(frag => ({
                id: frag.id,
                path: frag.data_files?.[0] || `${frag.id}.lance`,
                numRows: frag.num_rows,
                physicalRows: frag.physical_rows || frag.num_rows,
                url: `${this.baseUrl}/data/${frag.data_files?.[0] || frag.id + '.lance'}`,
                deletionFile: frag.has_deletions ? { numDeletedRows: frag.deleted_rows || 0 } : null
            }));

            this._numColumns = sidecar.num_columns;
            this._totalRows = sidecar.total_rows;
            this._version = sidecar.lance_version;

            // Extract column types from sidecar schema
            this._columnTypes = sidecar.schema.map(col => {
                const type = col.type;
                if (type.startsWith('vector[')) return 'vector';
                if (type === 'float64' || type === 'double') return 'float64';
                if (type === 'float32') return 'float32';
                if (type.includes('int')) return type;
                if (type === 'string') return 'string';
                return 'unknown';
            });

            return true;
        } catch (e) {
            // Sidecar not available or invalid - fall back to manifest
            return false;
        }
    }

    /**
     * Prefetch fragment metadata (footers) in parallel.
     * Does not block - runs in background.
     * @private
     */
    _prefetchFragments() {
        const prefetchPromises = this._fragments.map((_, idx) =>
            this.openFragment(idx).catch(() => null)
        );
        // Run in background, don't await
        Promise.all(prefetchPromises).then(() => {
            console.log(`[LanceQL Dataset] Prefetched ${this._fragments.length} fragment(s)`);
        });
    }

    /**
     * Check if dataset has an IVF index loaded.
     */
    hasIndex() {
        return this._ivfIndex !== null && this._ivfIndex.centroids !== null;
    }

    /**
     * Try to load IVF index from _indices folder.
     * @private
     */
    async _tryLoadIndex() {
        try {
            console.log(`[LanceQL Dataset] Trying to load IVF index from ${this.baseUrl}`);
            this._ivfIndex = await IVFIndex.tryLoad(this.baseUrl);
            if (this._ivfIndex) {
                console.log(`[LanceQL Dataset] IVF index loaded: ${this._ivfIndex.numPartitions} partitions, dim=${this._ivfIndex.dimension}`);
            } else {
                console.log('[LanceQL Dataset] IVF index not found or failed to parse');
            }
        } catch (e) {
            console.log('[LanceQL Dataset] No IVF index found:', e.message);
            this._ivfIndex = null;
        }
    }

    /**
     * Load and parse the manifest to discover fragments.
     * @private
     */
    async _loadManifest() {
        let manifestData = null;
        let manifestVersion = 0;

        // If specific version requested (time-travel), use that
        if (this._requestedVersion) {
            manifestVersion = this._requestedVersion;
            const manifestUrl = `${this.baseUrl}/_versions/${manifestVersion}.manifest`;
            const response = await fetch(manifestUrl);
            if (!response.ok) {
                throw new Error(`Version ${manifestVersion} not found (${response.status})`);
            }
            manifestData = new Uint8Array(await response.arrayBuffer());
        } else {
            // Find the latest manifest version using binary search approach
            // First check common versions in parallel
            const checkVersions = [1, 5, 10, 20, 50, 100];
            const checks = await Promise.all(
                checkVersions.map(async v => {
                    try {
                        const url = `${this.baseUrl}/_versions/${v}.manifest`;
                        const response = await fetch(url, { method: 'HEAD' });
                        return response.ok ? v : 0;
                    } catch {
                        return 0;
                    }
                })
            );

            // Find highest existing version from quick check
            let highestFound = Math.max(...checks);

            // If we found a high version, scan forward from there
            if (highestFound > 0) {
                for (let v = highestFound + 1; v <= highestFound + 50; v++) {
                    try {
                        const url = `${this.baseUrl}/_versions/${v}.manifest`;
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
            }

            manifestVersion = highestFound;

            if (manifestVersion === 0) {
                throw new Error('No manifest found in dataset');
            }

            // Fetch the latest manifest
            const manifestUrl = `${this.baseUrl}/_versions/${manifestVersion}.manifest`;
            const response = await fetch(manifestUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch manifest: ${response.status}`);
            }
            manifestData = new Uint8Array(await response.arrayBuffer());
        }

        // Store the version we loaded
        this._version = manifestVersion;
        this._latestVersion = this._requestedVersion ? null : manifestVersion;

        console.log(`[LanceQL Dataset] Loading manifest v${manifestVersion}${this._requestedVersion ? ' (time-travel)' : ''}...`);
        this._parseManifest(manifestData);

        console.log(`[LanceQL Dataset] Loaded: ${this._fragments.length} fragments, ${this._totalRows.toLocaleString()} rows, ${this._numColumns} columns`);
    }

    /**
     * Get list of available versions.
     * @returns {Promise<number[]>}
     */
    async listVersions() {
        const versions = [];
        // Scan for versions 1 to latestVersion (or 100 if unknown)
        const maxVersion = this._latestVersion || 100;

        const checks = await Promise.all(
            Array.from({ length: maxVersion }, (_, i) => i + 1).map(async v => {
                try {
                    const url = `${this.baseUrl}/_versions/${v}.manifest`;
                    const response = await fetch(url, { method: 'HEAD' });
                    return response.ok ? v : 0;
                } catch {
                    return 0;
                }
            })
        );

        return checks.filter(v => v > 0);
    }

    /**
     * Get current loaded version.
     */
    get version() {
        return this._version;
    }

    /**
     * Parse manifest protobuf to extract schema and fragment info.
     *
     * Lance manifest file structure:
     * - Chunk 1 (len-prefixed): Transaction metadata (may be small/incremental)
     * - Chunk 2 (len-prefixed): Full manifest with schema + fragments
     * - Footer (16 bytes): Offsets + "LANC" magic
     *
     * @private
     */
    _parseManifest(bytes) {
        const view = new DataView(bytes.buffer, bytes.byteOffset);

        // Read chunk 1 length
        const chunk1Len = view.getUint32(0, true);

        // Check if there's a chunk 2 (full manifest data)
        // Chunk 2 starts at offset (4 + chunk1Len)
        const chunk2Start = 4 + chunk1Len;
        let protoData;

        if (chunk2Start + 4 < bytes.length) {
            const chunk2Len = view.getUint32(chunk2Start, true);
            // Verify chunk 2 exists and has reasonable size
            if (chunk2Len > 0 && chunk2Start + 4 + chunk2Len <= bytes.length) {
                // Use chunk 2 (full manifest)
                protoData = bytes.slice(chunk2Start + 4, chunk2Start + 4 + chunk2Len);
            } else {
                // Fall back to chunk 1
                protoData = bytes.slice(4, 4 + chunk1Len);
            }
        } else {
            // Only chunk 1 exists
            protoData = bytes.slice(4, 4 + chunk1Len);
        }

        let pos = 0;
        const fields = [];
        const fragments = [];

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

        const skipField = (wireType) => {
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

                let name = null;
                let id = null;
                let logicalType = null;

                while (pos < fieldEnd) {
                    const fTag = readVarint();
                    const fNum = fTag >> 3;
                    const fWire = fTag & 0x7;

                    if (fWire === 0) {
                        const val = readVarint();
                        if (fNum === 3) id = val;
                    } else if (fWire === 2) {
                        const len = readVarint();
                        const content = protoData.slice(pos, pos + len);
                        pos += len;

                        if (fNum === 2) {
                            name = new TextDecoder().decode(content);
                        } else if (fNum === 5) {
                            logicalType = new TextDecoder().decode(content);
                        }
                    } else {
                        skipField(fWire);
                    }
                }

                if (name) {
                    fields.push({ name, id, type: logicalType });
                }
            } else if (fieldNum === 2 && wireType === 2) {
                // Field 2 = fragments (repeated Fragment message)
                const fragLen = readVarint();
                const fragEnd = pos + fragLen;

                let fragId = null;
                let filePath = null;
                let numRows = 0;
                let deletionFile = null;  // Track deletion info

                while (pos < fragEnd) {
                    const fTag = readVarint();
                    const fNum = fTag >> 3;
                    const fWire = fTag & 0x7;

                    if (fWire === 0) {
                        const val = readVarint();
                        if (fNum === 1) fragId = val;  // Fragment.id
                        else if (fNum === 4) numRows = val;  // Fragment.physical_rows
                    } else if (fWire === 2) {
                        const len = readVarint();
                        const content = protoData.slice(pos, pos + len);
                        pos += len;

                        if (fNum === 2) {
                            // Fragment.files - parse DataFile message
                            let innerPos = 0;
                            while (innerPos < content.length) {
                                const iTag = content[innerPos++];
                                const iNum = iTag >> 3;
                                const iWire = iTag & 0x7;

                                if (iWire === 2) {
                                    // Length-delimited
                                    let iLen = 0;
                                    let iShift = 0;
                                    while (innerPos < content.length) {
                                        const b = content[innerPos++];
                                        iLen |= (b & 0x7F) << iShift;
                                        if ((b & 0x80) === 0) break;
                                        iShift += 7;
                                    }
                                    const iContent = content.slice(innerPos, innerPos + iLen);
                                    innerPos += iLen;

                                    if (iNum === 1) {
                                        // DataFile.path
                                        filePath = new TextDecoder().decode(iContent);
                                    }
                                } else if (iWire === 0) {
                                    // Varint - skip
                                    while (innerPos < content.length && (content[innerPos++] & 0x80) !== 0);
                                } else if (iWire === 5) {
                                    innerPos += 4;
                                } else if (iWire === 1) {
                                    innerPos += 8;
                                }
                            }
                        } else if (fNum === 3) {
                            // Fragment.deletion_file - parse DeletionFile message
                            deletionFile = this._parseDeletionFile(content, fragId);
                        }
                    } else {
                        skipField(fWire);
                    }
                }

                if (filePath) {
                    const logicalRows = deletionFile ? numRows - deletionFile.numDeletedRows : numRows;
                    fragments.push({
                        id: fragId,
                        path: filePath,
                        numRows: logicalRows,  // Logical rows (excluding deleted)
                        physicalRows: numRows, // Physical rows (including deleted)
                        deletionFile: deletionFile,
                        url: `${this.baseUrl}/data/${filePath}`
                    });
                }
            } else {
                skipField(wireType);
            }
        }

        this._schema = fields;
        this._fragments = fragments;
        this._numColumns = fields.length;
        this._totalRows = fragments.reduce((sum, f) => sum + f.numRows, 0);

        // Track if any fragment has deletions
        const deletedCount = fragments.reduce((sum, f) => sum + (f.deletionFile?.numDeletedRows || 0), 0);
        if (deletedCount > 0) {
            console.log(`[LanceQL Dataset] Has ${deletedCount} deleted rows across fragments`);
        }
    }

    /**
     * Parse DeletionFile protobuf message.
     * @param {Uint8Array} data - Raw protobuf bytes
     * @param {number} fragId - Fragment ID for path construction
     * @returns {Object|null} Deletion file info
     * @private
     */
    _parseDeletionFile(data, fragId) {
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
                if (fieldNum === 1) fileType = val;       // DeletionFile.file_type
                else if (fieldNum === 2) readVersion = val; // DeletionFile.read_version
                else if (fieldNum === 3) id = val;        // DeletionFile.id
                else if (fieldNum === 4) numDeletedRows = val; // DeletionFile.num_deleted_rows
            } else if (wireType === 2) {
                const len = readVarint();
                pos += len; // Skip length-delimited fields
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
            url: `${this.baseUrl}/${path}`
        };
    }

    /**
     * Load deleted row indices for a fragment.
     * @param {number} fragmentIndex - Fragment index
     * @returns {Promise<Set<number>>} Set of deleted row indices (local to fragment)
     * @private
     */
    async _loadDeletedRows(fragmentIndex) {
        // Check cache
        if (this._deletedRows.has(fragmentIndex)) {
            return this._deletedRows.get(fragmentIndex);
        }

        const frag = this._fragments[fragmentIndex];
        if (!frag?.deletionFile) {
            const emptySet = new Set();
            this._deletedRows.set(fragmentIndex, emptySet);
            return emptySet;
        }

        const { url, fileType, numDeletedRows } = frag.deletionFile;
        console.log(`[LanceQL] Loading ${numDeletedRows} deletions from ${url} (${fileType})`);

        try {
            const response = await fetch(url);
            if (!response.ok) {
                console.warn(`[LanceQL] Failed to load deletion file: ${response.status}`);
                const emptySet = new Set();
                this._deletedRows.set(fragmentIndex, emptySet);
                return emptySet;
            }

            const buffer = await response.arrayBuffer();
            const data = new Uint8Array(buffer);
            let deletedSet;

            if (fileType === 'arrow') {
                deletedSet = this._parseArrowDeletions(data);
            } else {
                deletedSet = this._parseRoaringBitmap(data);
            }

            console.log(`[LanceQL] Loaded ${deletedSet.size} deleted rows for fragment ${fragmentIndex}`);
            this._deletedRows.set(fragmentIndex, deletedSet);
            return deletedSet;
        } catch (e) {
            console.error(`[LanceQL] Error loading deletion file:`, e);
            const emptySet = new Set();
            this._deletedRows.set(fragmentIndex, emptySet);
            return emptySet;
        }
    }

    /**
     * Parse Arrow IPC deletion file (Int32Array of deleted indices).
     * @param {Uint8Array} data - Raw Arrow IPC bytes
     * @returns {Set<number>} Set of deleted row indices
     * @private
     */
    _parseArrowDeletions(data) {
        // Arrow IPC format: Magic (ARROW1) + schema + record batch
        // For simplicity, we look for the Int32 data after the schema
        const deletedSet = new Set();

        // Find continuation marker (-1 as int32 LE = 0xFFFFFFFF)
        // Then record batch metadata length, then metadata, then body (Int32 array)
        let pos = 0;

        // Skip magic "ARROW1" + padding
        if (data.length >= 8 && String.fromCharCode(...data.slice(0, 6)) === 'ARROW1') {
            pos = 8;
        }

        // Look for continuation markers and skip metadata
        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);

        while (pos < data.length - 4) {
            const marker = view.getInt32(pos, true);
            if (marker === -1) {
                // Continuation marker found
                pos += 4;
                if (pos + 4 > data.length) break;
                const metaLen = view.getInt32(pos, true);
                pos += 4 + metaLen; // Skip metadata

                // The body follows - for deletion vectors it's just Int32 array
                // We need to read until end or next message
                while (pos + 4 <= data.length) {
                    // Check if this looks like the start of data (not another marker)
                    const nextMarker = view.getInt32(pos, true);
                    if (nextMarker === -1) break; // Another message starts

                    // Read Int32 values until we hit something that looks like a marker
                    // or reach expected count
                    const val = view.getInt32(pos, true);
                    if (val >= 0 && val < 10000000) { // Sanity check
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
     * @private
     */
    _parseRoaringBitmap(data) {
        // Roaring bitmap format: header + containers
        // This is a simplified parser for common cases
        const deletedSet = new Set();
        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);

        if (data.length < 8) return deletedSet;

        // Read cookie (first 4 bytes indicate format)
        const cookie = view.getUint32(0, true);

        // Standard roaring format: cookie = 12346 or 12347
        // Portable format: first 8 bytes are magic
        if (cookie === 12346 || cookie === 12347) {
            // Standard format
            const isRunContainer = (cookie === 12347);
            let pos = 4;

            // Number of containers
            const numContainers = view.getUint16(pos, true);
            pos += 2;

            // Skip to container data
            // Each key is 2 bytes, each cardinality is 2 bytes
            const keysStart = pos;
            pos += numContainers * 4; // keys + cardinalities

            for (let i = 0; i < numContainers && pos < data.length; i++) {
                const key = view.getUint16(keysStart + i * 4, true);
                const card = view.getUint16(keysStart + i * 4 + 2, true) + 1;
                const baseValue = key << 16;

                // Read container values (simplified - assumes array container)
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
     * Check if a row is deleted in a fragment.
     * @param {number} fragmentIndex - Fragment index
     * @param {number} localRowIndex - Row index within the fragment
     * @returns {Promise<boolean>} True if row is deleted
     */
    async isRowDeleted(fragmentIndex, localRowIndex) {
        const deletedSet = await this._loadDeletedRows(fragmentIndex);
        return deletedSet.has(localRowIndex);
    }

    /**
     * Get number of columns.
     */
    get numColumns() {
        return this._numColumns;
    }

    /**
     * Get total row count across all fragments.
     */
    get rowCount() {
        return this._totalRows;
    }

    /**
     * Get row count for a column (for API compatibility with RemoteLanceFile).
     * @param {number} columnIndex - Column index (ignored, all columns have same row count)
     * @returns {Promise<number>}
     */
    async getRowCount(columnIndex = 0) {
        return this._totalRows;
    }

    /**
     * Read a single vector at a global row index.
     * Delegates to the correct fragment based on row index.
     * @param {number} colIdx - Column index
     * @param {number} rowIdx - Global row index
     * @returns {Promise<Float32Array>}
     */
    async readVectorAt(colIdx, rowIdx) {
        const loc = this._getFragmentForRow(rowIdx);
        if (!loc) return new Float32Array(0);
        const file = await this.openFragment(loc.fragmentIndex);
        return await file.readVectorAt(colIdx, loc.localIndex);
    }

    /**
     * Get vector info for a column by querying first fragment.
     * @param {number} colIdx - Column index
     * @returns {Promise<{rows: number, dimension: number}>}
     */
    async getVectorInfo(colIdx) {
        if (this._fragments.length === 0) {
            return { rows: 0, dimension: 0 };
        }

        // Get vector info from first fragment
        const file = await this.openFragment(0);
        const fragInfo = await file.getVectorInfo(colIdx);

        if (fragInfo.dimension === 0) {
            return { rows: 0, dimension: 0 };
        }

        // Return total rows across all fragments, dimension from first fragment
        return {
            rows: this._totalRows,
            dimension: fragInfo.dimension
        };
    }

    /**
     * Get column names from schema.
     */
    get columnNames() {
        return this._schema ? this._schema.map(f => f.name) : [];
    }

    /**
     * Get full schema.
     */
    get schema() {
        return this._schema;
    }

    /**
     * Get fragment list.
     */
    get fragments() {
        return this._fragments;
    }

    /**
     * Get estimated total size based on row count and schema.
     * More accurate than fragment count estimate.
     */
    get size() {
        if (this._cachedSize) return this._cachedSize;

        // Estimate bytes per row based on column types
        let bytesPerRow = 0;
        for (let i = 0; i < (this._columnTypes?.length || 0); i++) {
            const colType = this._columnTypes[i];
            if (colType === 'int64' || colType === 'float64' || colType === 'double') {
                bytesPerRow += 8;
            } else if (colType === 'int32' || colType === 'float32') {
                bytesPerRow += 4;
            } else if (colType === 'string') {
                bytesPerRow += 50; // Average string length estimate
            } else if (colType === 'vector' || colType?.startsWith('vector[')) {
                // Extract dimension from type like "vector[384]"
                const match = colType?.match(/\[(\d+)\]/);
                const dim = match ? parseInt(match[1]) : 384;
                bytesPerRow += dim * 4; // float32 per dimension
            } else {
                bytesPerRow += 8; // Default
            }
        }

        // Fallback if no column types
        if (bytesPerRow === 0) {
            bytesPerRow = 100; // Conservative default
        }

        this._cachedSize = this._totalRows * bytesPerRow;
        return this._cachedSize;
    }

    /**
     * Set callback for network fetch events.
     */
    onFetch(callback) {
        this._onFetch = callback;
    }

    /**
     * Open a specific fragment as RemoteLanceFile.
     * @param {number} fragmentIndex - Index of fragment to open
     * @returns {Promise<RemoteLanceFile>}
     */
    async openFragment(fragmentIndex) {
        if (fragmentIndex < 0 || fragmentIndex >= this._fragments.length) {
            throw new Error(`Invalid fragment index: ${fragmentIndex}`);
        }

        // Check cache
        if (this._fragmentFiles.has(fragmentIndex)) {
            return this._fragmentFiles.get(fragmentIndex);
        }

        const fragment = this._fragments[fragmentIndex];
        const file = await RemoteLanceFile.open(this.lanceql, fragment.url);

        // Propagate fetch callback
        if (this._onFetch) {
            file.onFetch(this._onFetch);
        }

        this._fragmentFiles.set(fragmentIndex, file);
        return file;
    }

    /**
     * Read rows from the dataset with pagination.
     * Fetches from multiple fragments in parallel.
     * @param {Object} options - Query options
     * @param {number} options.offset - Starting row offset
     * @param {number} options.limit - Maximum rows to return
     * @param {number[]} options.columns - Column indices to read (optional)
     * @param {boolean} options._isPrefetch - Internal flag to prevent recursive prefetch
     * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
     */
    async readRows({ offset = 0, limit = 50, columns = null, _isPrefetch = false } = {}) {
        // Determine which fragments contain the requested rows
        const fragmentRanges = [];
        let currentOffset = 0;

        for (let i = 0; i < this._fragments.length; i++) {
            const frag = this._fragments[i];
            const fragStart = currentOffset;
            const fragEnd = currentOffset + frag.numRows;

            // Check if this fragment overlaps with requested range
            if (fragEnd > offset && fragStart < offset + limit) {
                const localStart = Math.max(0, offset - fragStart);
                const localEnd = Math.min(frag.numRows, offset + limit - fragStart);

                fragmentRanges.push({
                    fragmentIndex: i,
                    localOffset: localStart,
                    localLimit: localEnd - localStart,
                    globalStart: fragStart + localStart
                });
            }

            currentOffset = fragEnd;
            if (currentOffset >= offset + limit) break;
        }

        if (fragmentRanges.length === 0) {
            return { columns: [], columnNames: this.columnNames, total: this._totalRows };
        }

        // Fetch from fragments in parallel
        const fetchPromises = fragmentRanges.map(async (range) => {
            const file = await this.openFragment(range.fragmentIndex);
            const result = await file.readRows({
                offset: range.localOffset,
                limit: range.localLimit,
                columns: columns
            });
            return { ...range, result };
        });

        const results = await Promise.all(fetchPromises);

        // Merge results in order
        results.sort((a, b) => a.globalStart - b.globalStart);

        const mergedColumns = [];
        const colNames = results[0]?.result.columnNames || this.columnNames;
        const numCols = columns ? columns.length : this._numColumns;

        for (let c = 0; c < numCols; c++) {
            const colData = [];
            for (const r of results) {
                if (r.result.columns[c]) {
                    colData.push(...r.result.columns[c]);
                }
            }
            mergedColumns.push(colData);
        }

        const result = {
            columns: mergedColumns,
            columnNames: colNames,
            total: this._totalRows
        };

        // Speculative prefetch: if there are more rows, prefetch next page in background
        // Only prefetch if: not already a prefetch, limit is reasonable, more rows exist
        const nextOffset = offset + limit;
        if (!_isPrefetch && nextOffset < this._totalRows && limit <= 100) {
            this._prefetchNextPage(nextOffset, limit, columns);
        }

        return result;
    }

    /**
     * Prefetch next page of rows in background.
     * @private
     */
    _prefetchNextPage(offset, limit, columns) {
        // Use a cache key to avoid duplicate prefetches
        const cacheKey = `${offset}-${limit}-${columns?.join(',') || 'all'}`;
        if (this._prefetchCache?.has(cacheKey)) {
            return; // Already prefetching or prefetched
        }

        if (!this._prefetchCache) {
            this._prefetchCache = new Map();
        }

        // Start prefetch in background (don't await)
        const prefetchPromise = this.readRows({ offset, limit, columns, _isPrefetch: true })
            .then(result => {
                this._prefetchCache.set(cacheKey, result);
                console.log(`[LanceQL] Prefetched rows ${offset}-${offset + limit}`);
            })
            .catch(() => {
                // Ignore prefetch errors
            });

        this._prefetchCache.set(cacheKey, prefetchPromise);
    }

    /**
     * Detect column types by sampling from first fragment.
     * @returns {Promise<string[]>}
     */
    async detectColumnTypes() {
        // Return cached types if available
        if (this._columnTypes && this._columnTypes.length > 0) {
            return this._columnTypes;
        }

        if (this._fragments.length === 0) {
            return [];
        }
        const file = await this.openFragment(0);
        const types = await file.detectColumnTypes();
        this._columnTypes = types;

        // Update cache with column types
        const cacheKey = this._requestedVersion ? `${this.baseUrl}@v${this._requestedVersion}` : this.baseUrl;
        metadataCache.get(cacheKey).then(cached => {
            if (cached) {
                cached.columnTypes = types;
                metadataCache.set(cacheKey, cached).catch(() => {});
            }
        }).catch(() => {});

        return types;
    }

    /**
     * Helper to determine which fragment contains a given row index.
     * @private
     */
    _getFragmentForRow(rowIdx) {
        let offset = 0;
        for (let i = 0; i < this._fragments.length; i++) {
            const frag = this._fragments[i];
            if (rowIdx < offset + frag.numRows) {
                return { fragmentIndex: i, localIndex: rowIdx - offset };
            }
            offset += frag.numRows;
        }
        return null;
    }

    /**
     * Group indices by fragment for efficient batch reading.
     * @private
     */
    _groupIndicesByFragment(indices) {
        const groups = new Map();
        for (const globalIdx of indices) {
            const loc = this._getFragmentForRow(globalIdx);
            if (!loc) continue;

            if (!groups.has(loc.fragmentIndex)) {
                groups.set(loc.fragmentIndex, { localIndices: [], globalIndices: [] });
            }
            groups.get(loc.fragmentIndex).localIndices.push(loc.localIndex);
            groups.get(loc.fragmentIndex).globalIndices.push(globalIdx);
        }
        return groups;
    }

    /**
     * Read strings at specific indices across fragments.
     */
    async readStringsAtIndices(colIdx, indices) {
        const groups = this._groupIndicesByFragment(indices);
        const results = new Map();

        console.log(`[ReadStrings] Reading ${indices.length} strings from col ${colIdx}`);
        console.log(`[ReadStrings] First 5 indices: ${indices.slice(0, 5)}`);
        console.log(`[ReadStrings] Fragment groups: ${Array.from(groups.keys())}`);

        // Fetch from each fragment in parallel
        const fetchPromises = [];
        for (const [fragIdx, group] of groups) {
            fetchPromises.push((async () => {
                const file = await this.openFragment(fragIdx);
                console.log(`[ReadStrings] Fragment ${fragIdx}: reading ${group.localIndices.length} strings, first local indices: ${group.localIndices.slice(0, 3)}`);
                const data = await file.readStringsAtIndices(colIdx, group.localIndices);
                console.log(`[ReadStrings] Fragment ${fragIdx}: got ${data.length} strings, first 3: ${data.slice(0, 3).map(s => s?.slice(0, 20) + '...')}`);
                for (let i = 0; i < group.globalIndices.length; i++) {
                    results.set(group.globalIndices[i], data[i]);
                }
            })());
        }
        await Promise.all(fetchPromises);

        // Return in original order
        return indices.map(idx => results.get(idx) || null);
    }

    /**
     * Read int64 values at specific indices across fragments.
     */
    async readInt64AtIndices(colIdx, indices) {
        const groups = this._groupIndicesByFragment(indices);
        const results = new Map();

        const fetchPromises = [];
        for (const [fragIdx, group] of groups) {
            fetchPromises.push((async () => {
                const file = await this.openFragment(fragIdx);
                const data = await file.readInt64AtIndices(colIdx, group.localIndices);
                for (let i = 0; i < group.globalIndices.length; i++) {
                    results.set(group.globalIndices[i], data[i]);
                }
            })());
        }
        await Promise.all(fetchPromises);

        return new BigInt64Array(indices.map(idx => results.get(idx) || 0n));
    }

    /**
     * Read float64 values at specific indices across fragments.
     */
    async readFloat64AtIndices(colIdx, indices) {
        const groups = this._groupIndicesByFragment(indices);
        const results = new Map();

        const fetchPromises = [];
        for (const [fragIdx, group] of groups) {
            fetchPromises.push((async () => {
                const file = await this.openFragment(fragIdx);
                const data = await file.readFloat64AtIndices(colIdx, group.localIndices);
                for (let i = 0; i < group.globalIndices.length; i++) {
                    results.set(group.globalIndices[i], data[i]);
                }
            })());
        }
        await Promise.all(fetchPromises);

        return new Float64Array(indices.map(idx => results.get(idx) || 0));
    }

    /**
     * Read int32 values at specific indices across fragments.
     */
    async readInt32AtIndices(colIdx, indices) {
        const groups = this._groupIndicesByFragment(indices);
        const results = new Map();

        const fetchPromises = [];
        for (const [fragIdx, group] of groups) {
            fetchPromises.push((async () => {
                const file = await this.openFragment(fragIdx);
                const data = await file.readInt32AtIndices(colIdx, group.localIndices);
                for (let i = 0; i < group.globalIndices.length; i++) {
                    results.set(group.globalIndices[i], data[i]);
                }
            })());
        }
        await Promise.all(fetchPromises);

        return new Int32Array(indices.map(idx => results.get(idx) || 0));
    }

    /**
     * Read float32 values at specific indices across fragments.
     */
    async readFloat32AtIndices(colIdx, indices) {
        const groups = this._groupIndicesByFragment(indices);
        const results = new Map();

        const fetchPromises = [];
        for (const [fragIdx, group] of groups) {
            fetchPromises.push((async () => {
                const file = await this.openFragment(fragIdx);
                const data = await file.readFloat32AtIndices(colIdx, group.localIndices);
                for (let i = 0; i < group.globalIndices.length; i++) {
                    results.set(group.globalIndices[i], data[i]);
                }
            })());
        }
        await Promise.all(fetchPromises);

        return new Float32Array(indices.map(idx => results.get(idx) || 0));
    }

    /**
     * Vector search across all fragments.
     * API compatible with RemoteLanceFile.vectorSearch.
     *
     * @param {number} colIdx - Vector column index
     * @param {Float32Array} queryVec - Query vector
     * @param {number} topK - Number of results to return
     * @param {Function} onProgress - Progress callback (current, total)
     * @param {Object} options - Search options
     * @returns {Promise<{indices: number[], scores: number[], usedIndex: boolean}>}
     */
    async vectorSearch(colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
        const {
            normalized = true,
            workerPool = null,
            useIndex = true,
            nprobe = 20
        } = options;

        const vectorColIdx = colIdx;

        if (vectorColIdx < 0) {
            throw new Error('No vector column found in dataset');
        }

        const dim = queryVec.length;
        console.log(`[VectorSearch] Query dim=${dim}, topK=${topK}, fragments=${this._fragments.length}, hasIndex=${this.hasIndex()}`);

        // Require IVF index for efficient search - no brute force fallback
        if (!this.hasIndex()) {
            throw new Error('No IVF index found. Vector search requires an IVF index for efficient querying.');
        }

        if (this._ivfIndex.dimension !== dim) {
            throw new Error(`Query dimension (${dim}) does not match index dimension (${this._ivfIndex.dimension}).`);
        }

        if (!this._ivfIndex.hasPartitionIndex) {
            throw new Error('IVF partition index (ivf_partitions.bin) not found. Required for efficient search.');
        }

        console.log(`[VectorSearch] Using IVF index (nprobe=${nprobe})`);
        return await this._ivfIndexSearch(queryVec, topK, vectorColIdx, nprobe, onProgress);
    }

    /**
     * IVF index-based ANN search.
     * Fetches partition data (row IDs + vectors) directly from ivf_vectors.bin.
     * Uses WebGPU for batch similarity computation.
     * @private
     */
    async _ivfIndexSearch(queryVec, topK, vectorColIdx, nprobe, onProgress) {
        // Find nearest partitions using centroids
        const partitions = this._ivfIndex.findNearestPartitions(queryVec, nprobe);
        console.log(`[VectorSearch] Searching ${partitions.length} partitions:`, partitions);

        // Fetch partition data (row IDs + vectors) directly
        const partitionData = await this._ivfIndex.fetchPartitionData(
            partitions,
            this._ivfIndex.dimension,
            (loaded, total) => {
                if (onProgress) {
                    // First 80% is downloading
                    const pct = total > 0 ? loaded / total : 0;
                    onProgress(Math.floor(pct * 80), 100);
                }
            }
        );

        if (!partitionData || partitionData.rowIds.length === 0) {
            throw new Error('IVF index not available. This dataset requires ivf_vectors.bin for efficient search.');
        }

        const { rowIds, vectors } = partitionData;

        // Use hybrid WebGPU + WASM SIMD for batch similarity
        const scores = new Float32Array(vectors.length);
        const dim = queryVec.length;

        if (webgpuAccelerator.isAvailable()) {
            const maxBatch = webgpuAccelerator.getMaxVectorsPerBatch(dim);
            let gpuProcessed = 0;
            let wasmProcessed = 0;

            // Process in chunks that fit in WebGPU buffer
            for (let start = 0; start < vectors.length; start += maxBatch) {
                const end = Math.min(start + maxBatch, vectors.length);
                const chunk = vectors.slice(start, end);

                try {
                    const chunkScores = await webgpuAccelerator.batchCosineSimilarity(queryVec, chunk, true);
                    if (chunkScores) {
                        scores.set(chunkScores, start);
                        gpuProcessed += chunk.length;
                        continue;
                    }
                } catch (e) {
                    // Fall through to WASM for this chunk
                }

                // WASM SIMD fallback for this chunk
                if (this._fragments[0]?.lanceql?.batchCosineSimilarity) {
                    const chunkScores = this._fragments[0].lanceql.batchCosineSimilarity(queryVec, chunk, true);
                    scores.set(chunkScores, start);
                    wasmProcessed += chunk.length;
                } else {
                    // JS fallback (slow)
                    for (let i = 0; i < chunk.length; i++) {
                        const vec = chunk[i];
                        if (!vec || vec.length !== dim) continue;
                        let dot = 0;
                        for (let k = 0; k < dim; k++) {
                            dot += queryVec[k] * vec[k];
                        }
                        scores[start + i] = dot;
                    }
                    wasmProcessed += chunk.length;
                }
            }

            console.log(`[VectorSearch] Processed ${vectors.length.toLocaleString()} vectors: ${gpuProcessed.toLocaleString()} WebGPU, ${wasmProcessed.toLocaleString()} WASM SIMD`);
        } else {
            // Pure WASM SIMD path
            console.log(`[VectorSearch] Computing similarities for ${rowIds.length.toLocaleString()} vectors via WASM SIMD`);
            if (this._fragments[0]?.lanceql?.batchCosineSimilarity) {
                const allScores = this._fragments[0].lanceql.batchCosineSimilarity(queryVec, vectors, true);
                scores.set(allScores);
            } else {
                // JS fallback (slow)
                for (let i = 0; i < vectors.length; i++) {
                    const vec = vectors[i];
                    if (!vec || vec.length !== dim) continue;
                    let dot = 0;
                    for (let k = 0; k < dim; k++) {
                        dot += queryVec[k] * vec[k];
                    }
                    scores[i] = dot;
                }
            }
        }

        if (onProgress) onProgress(90, 100);

        // Build results with row IDs
        const allResults = [];
        for (let i = 0; i < rowIds.length; i++) {
            allResults.push({ index: rowIds[i], score: scores[i] });
        }

        // Sort and take top-k
        allResults.sort((a, b) => b.score - a.score);
        const finalK = Math.min(topK, allResults.length);

        if (onProgress) onProgress(100, 100);

        return {
            indices: allResults.slice(0, finalK).map(r => r.index),
            scores: allResults.slice(0, finalK).map(r => r.score),
            usedIndex: true,
            searchedRows: rowIds.length
        };
    }

    /**
     * Find the vector column index by looking at schema.
     * @private
     */
    _findVectorColumn() {
        if (!this._schema) return -1;

        for (let i = 0; i < this._schema.length; i++) {
            const field = this._schema[i];
            if (field.name === 'embedding' || field.name === 'vector' ||
                field.type === 'fixed_size_list' || field.type === 'list') {
                return i;
            }
        }

        // Assume last column is vector if schema unclear
        return this._schema.length - 1;
    }

    /**
     * Parallel vector search using WorkerPool.
     * @private
     */
    async _parallelVectorSearch(query, topK, vectorColIdx, normalized, workerPool) {
        const dim = query.length;

        // Load vectors from each fragment in parallel
        const chunkPromises = this._fragments.map(async (frag, idx) => {
            const file = await this.openFragment(idx);

            // Get vector data for this fragment
            const vectors = await file.readVectorColumn(vectorColIdx);
            if (!vectors || vectors.length === 0) {
                return null;
            }

            // Calculate start index for this fragment
            let startIndex = 0;
            for (let i = 0; i < idx; i++) {
                startIndex += this._fragments[i].numRows;
            }

            return {
                vectors: new Float32Array(vectors),
                startIndex,
                numVectors: vectors.length / dim
            };
        });

        const chunks = (await Promise.all(chunkPromises)).filter(c => c !== null);

        if (chunks.length === 0) {
            return { indices: new Uint32Array(0), scores: new Float32Array(0), rows: [] };
        }

        // Perform parallel search
        const { indices, scores } = await workerPool.parallelVectorSearch(
            query, chunks, dim, topK, normalized
        );

        // Fetch row data for results
        const rows = await this._fetchResultRows(indices);

        return { indices, scores, rows };
    }

    /**
     * Fetch full row data for result indices.
     * @private
     */
    async _fetchResultRows(indices) {
        if (indices.length === 0) return [];

        const rows = [];

        // Group indices by fragment for efficient fetching
        const groups = this._groupIndicesByFragment(Array.from(indices));

        for (const [fragIdx, group] of groups) {
            const file = await this.openFragment(fragIdx);

            // Read string columns for display
            for (const localIdx of group.localIndices) {
                const row = {};

                // Try to read text/url columns
                for (let colIdx = 0; colIdx < this._numColumns; colIdx++) {
                    const colName = this.columnNames[colIdx];
                    if (colName === 'text' || colName === 'url' || colName === 'caption') {
                        try {
                            const values = await file.readStringsAtIndices(colIdx, [localIdx]);
                            row[colName] = values[0];
                        } catch (e) {
                            // Column might not be string type
                        }
                    }
                }

                rows.push(row);
            }
        }

        return rows;
    }

    /**
     * Execute SQL query across all fragments in parallel.
     * @param {string} sql - SQL query
     * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
     */
    async executeSQL(sql) {
        // Parse the SQL to understand what's needed
        const ast = parseSQL(sql);

        // For simple SELECT * with LIMIT, use readRows
        if (ast.type === 'SELECT' && ast.columns === '*' && !ast.where) {
            const limit = ast.limit || 50;
            const offset = ast.offset || 0;
            return await this.readRows({ offset, limit });
        }

        // For queries with WHERE or complex operations, execute on each fragment in parallel
        const fetchPromises = this._fragments.map(async (frag, idx) => {
            const file = await this.openFragment(idx);
            try {
                return await file.executeSQL(sql);
            } catch (e) {
                console.warn(`Fragment ${idx} query failed:`, e);
                return { columns: [], columnNames: [], total: 0 };
            }
        });

        const results = await Promise.all(fetchPromises);

        // Merge results
        if (results.length === 0 || results.every(r => r.columns.length === 0)) {
            return { columns: [], columnNames: this.columnNames, total: 0 };
        }

        const firstValid = results.find(r => r.columns.length > 0);
        if (!firstValid) {
            return { columns: [], columnNames: this.columnNames, total: 0 };
        }

        const numCols = firstValid.columns.length;
        const colNames = firstValid.columnNames;
        const mergedColumns = Array.from({ length: numCols }, () => []);

        let totalRows = 0;
        for (const r of results) {
            for (let c = 0; c < numCols && c < r.columns.length; c++) {
                mergedColumns[c].push(...r.columns[c]);
            }
            totalRows += r.total;
        }

        // Apply LIMIT if present (after merging)
        if (ast.limit) {
            const offset = ast.offset || 0;
            for (let c = 0; c < numCols; c++) {
                mergedColumns[c] = mergedColumns[c].slice(offset, offset + ast.limit);
            }
        }

        return {
            columns: mergedColumns,
            columnNames: colNames,
            total: totalRows
        };
    }

    /**
     * Close all cached fragment files.
     */
    close() {
        for (const file of this._fragmentFiles.values()) {
            if (file.close) file.close();
        }
        this._fragmentFiles.clear();
    }
}


export { RemoteLanceDataset };
