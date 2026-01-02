/**
 * RemoteLanceFile - Remote HTTP file reader with streaming
 * Main class that coordinates extracted modules.
 */

import { hotTierCache } from '../cache/hot-tier-cache.js';
import { IVFIndex } from '../search/ivf-index.js';
import { tryLoadSchema, getColumnNames, detectColumnTypes, parseManifest } from './remote-file-meta.js';
import { parseColumnMeta, parseStringColumnMeta, batchIndices } from './remote-file-proto.js';
import * as NumericModule from './remote-file-numeric.js';
import * as StringModule from './remote-file-string.js';
import * as VectorModule from './remote-file-vector.js';

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
        await tryLoadSchema(file);

        // Try to load IVF index for ANN search
        await file._tryLoadIndex();

        // Log summary
        console.log(`[LanceQL] Loaded: ${file._numColumns} columns, ${(fileSize / 1024 / 1024).toFixed(1)}MB, schema: ${file._schema ? 'yes' : 'no'}, index: ${file.hasIndex() ? 'yes' : 'no'}`);

        return file;
    }

    /**
     * Try to load IVF index from dataset.
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
     */
    hasIndex() {
        return this._ivfIndex !== null && this._ivfIndex.centroids !== null;
    }

    // === Properties ===

    get columnNames() {
        return getColumnNames(this);
    }

    get schema() {
        return this._schema;
    }

    get datasetBaseUrl() {
        return this._datasetBaseUrl;
    }

    get numColumns() {
        return this._numColumns;
    }

    get size() {
        return this.fileSize;
    }

    get version() {
        return {
            major: this._majorVersion,
            minor: this._minorVersion
        };
    }

    get columnMetaStart() {
        return Number(this._columnMetaStart);
    }

    get columnMetaOffsetsStart() {
        return Number(this._columnMetaOffsetsStart);
    }

    // === Core Methods ===

    /**
     * Fetch bytes from the remote file at a specific range.
     * Uses HotTierCache for OPFS-backed caching.
     */
    async fetchRange(start, end) {
        // Validate range
        if (start < 0 || end < start || end >= this.size) {
            console.error(`Invalid range: ${start}-${end}, file size: ${this.size}`);
        }

        // Use hot-tier cache if available
        if (hotTierCache.enabled) {
            const data = await hotTierCache.getRange(this.url, start, end, this.size);

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

        if (this._onFetch) {
            this._onFetch(data.byteLength, 1);
        }

        return data;
    }

    /**
     * Set callback for network stats tracking.
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

    // === Column Metadata ===

    /**
     * Get column offset entry from column metadata offsets.
     */
    async getColumnOffsetEntry(colIdx) {
        if (colIdx >= this._numColumns) {
            return { pos: 0, len: 0 };
        }

        if (this._columnOffsetCache.has(colIdx)) {
            return this._columnOffsetCache.get(colIdx);
        }

        const entryOffset = this.columnMetaOffsetsStart + colIdx * 16;
        const data = await this.fetchRange(entryOffset, entryOffset + 15);
        const view = new DataView(data);

        const entry = {
            pos: Number(view.getBigUint64(0, true)),
            len: Number(view.getBigUint64(8, true))
        };

        this._columnOffsetCache.set(colIdx, entry);
        return entry;
    }

    /**
     * Get debug info for a column.
     */
    async getColumnDebugInfo(colIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) {
            return { offset: 0, size: 0, rows: 0 };
        }

        const colMetaData = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const bytes = new Uint8Array(colMetaData);
        return this._parseColumnMeta(bytes);
    }

    _parseColumnMeta(bytes) {
        return parseColumnMeta(bytes);
    }

    _parseStringColumnMeta(bytes) {
        return parseStringColumnMeta(bytes);
    }

    _batchIndices(indices, valueSize, gapThreshold = 1024) {
        return batchIndices(indices, valueSize, gapThreshold);
    }

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

    // === Numeric Column Readers ===

    readInt64AtIndices(colIdx, indices) {
        return NumericModule.readInt64AtIndices(this, colIdx, indices);
    }

    readFloat64AtIndices(colIdx, indices) {
        return NumericModule.readFloat64AtIndices(this, colIdx, indices);
    }

    readInt32AtIndices(colIdx, indices) {
        return NumericModule.readInt32AtIndices(this, colIdx, indices);
    }

    readFloat32AtIndices(colIdx, indices) {
        return NumericModule.readFloat32AtIndices(this, colIdx, indices);
    }

    readInt16AtIndices(colIdx, indices) {
        return NumericModule.readInt16AtIndices(this, colIdx, indices);
    }

    readUint8AtIndices(colIdx, indices) {
        return NumericModule.readUint8AtIndices(this, colIdx, indices);
    }

    readBoolAtIndices(colIdx, indices) {
        return NumericModule.readBoolAtIndices(this, colIdx, indices);
    }

    // === String Column Readers ===

    readStringAt(colIdx, rowIdx) {
        return StringModule.readStringAt(this, colIdx, rowIdx);
    }

    readStringsAtIndices(colIdx, indices) {
        return StringModule.readStringsAtIndices(this, colIdx, indices);
    }

    // === Row Count ===

    async getRowCount(colIdx) {
        const info = await this.getColumnDebugInfo(colIdx);
        return info.rows;
    }

    // === Type Detection ===

    detectColumnTypes() {
        return detectColumnTypes(this);
    }

    // === Vector Operations ===

    getVectorInfo(colIdx) {
        return VectorModule.getVectorInfo(this, colIdx);
    }

    readVectorAt(colIdx, rowIdx) {
        return VectorModule.readVectorAt(this, colIdx, rowIdx);
    }

    readVectorsAtIndices(colIdx, indices) {
        return VectorModule.readVectorsAtIndices(this, colIdx, indices);
    }

    cosineSimilarity(vecA, vecB) {
        return VectorModule.cosineSimilarity(vecA, vecB);
    }

    vectorSearch(colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
        return VectorModule.vectorSearch(this, colIdx, queryVec, topK, onProgress, options);
    }

    readVectorColumn(colIdx) {
        return VectorModule.readVectorColumn(this, colIdx);
    }

    readRows(options = {}) {
        return VectorModule.readRows(this, options);
    }
}


export { RemoteLanceFile };
