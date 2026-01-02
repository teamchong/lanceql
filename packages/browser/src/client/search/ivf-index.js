/**
 * IVFIndex - IVF-PQ vector index for approximate nearest neighbor search
 */

import * as Manifest from './ivf-manifest.js';
import * as Auxiliary from './ivf-auxiliary.js';
import * as Partitions from './ivf-partitions.js';

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
            const manifestVersion = await Manifest.findLatestManifestVersion(datasetBaseUrl);
            console.log(`[IVFIndex] Manifest version: ${manifestVersion}`);
            if (!manifestVersion) return null;

            const manifestUrl = `${datasetBaseUrl}/_versions/${manifestVersion}.manifest`;
            const manifestResp = await fetch(manifestUrl);
            if (!manifestResp.ok) {
                console.log(`[IVFIndex] Failed to fetch manifest: ${manifestResp.status}`);
                return null;
            }

            const manifestData = await manifestResp.arrayBuffer();
            const indexInfo = Manifest.parseManifestForIndex(new Uint8Array(manifestData));
            console.log(`[IVFIndex] Index info:`, indexInfo);

            if (!indexInfo || !indexInfo.uuid) {
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
            const index = Manifest.parseIndexFile(new Uint8Array(indexData), indexInfo, IVFIndex);

            if (!index) return null;

            // Store auxiliary URL for later partition data fetching
            index.auxiliaryUrl = `${datasetBaseUrl}/_indices/${indexInfo.uuid}/auxiliary.idx`;
            index.datasetBaseUrl = datasetBaseUrl;

            // Fetch auxiliary.idx metadata (footer + partition info)
            try {
                await Auxiliary.loadAuxiliaryMetadata(index);
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
                await Partitions.loadPartitionIndex(index);
            } catch (e) {
                console.warn('[IVFIndex] Failed to load partition index:', e);
            }

            // Prefetch all row IDs for fast search (no HTTP during search)
            try {
                await Partitions.prefetchAllRowIds(index);
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
     * Load partition index from ivf_vectors.bin.
     * @returns {Promise<void>}
     */
    async _loadPartitionIndex() {
        return Partitions.loadPartitionIndex(this);
    }

    /**
     * Fetch partition data (row IDs and vectors).
     * @param {number[]} partitionIndices - Partition indices to fetch
     * @param {number} dim - Vector dimension (default 384)
     * @param {function} onProgress - Progress callback
     * @returns {Promise<{rowIds: number[], vectors: Float32Array[]}>}
     */
    fetchPartitionData(partitionIndices, dim = 384, onProgress = null) {
        return Partitions.fetchPartitionData(this, partitionIndices, dim, onProgress);
    }

    /**
     * Load auxiliary metadata from auxiliary.idx.
     * @returns {Promise<void>}
     */
    async _loadAuxiliaryMetadata() {
        return Auxiliary.loadAuxiliaryMetadata(this);
    }

    /**
     * Parse column metadata for partitions.
     * @param {Uint8Array} bytes - Column metadata bytes
     */
    _parseColumnMetaForPartitions(bytes) {
        return Auxiliary.parseColumnMetaForPartitions(this, bytes);
    }

    /**
     * Parse auxiliary partition info.
     * @param {Uint8Array} bytes - Partition info bytes
     */
    _parseAuxiliaryPartitionInfo(bytes) {
        return Auxiliary.parseAuxiliaryPartitionInfo(this, bytes);
    }

    /**
     * Prefetch all row IDs for fast search.
     * @returns {Promise<void>}
     */
    async prefetchAllRowIds() {
        return Partitions.prefetchAllRowIds(this);
    }

    /**
     * Fetch row IDs for specified partitions.
     * @param {number[]} partitionIndices - Partition indices to fetch
     * @returns {Promise<Array<{fragId: number, rowOffset: number}>>}
     */
    fetchPartitionRowIds(partitionIndices) {
        return Partitions.fetchPartitionRowIds(this, partitionIndices);
    }

    /**
     * Get estimated number of rows to search for given partitions.
     * @param {number[]} partitionIndices - Partition indices
     * @returns {number}
     */
    getPartitionRowCount(partitionIndices) {
        return Partitions.getPartitionRowCount(this, partitionIndices);
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

    // Static methods delegate to manifest module
    static async _findLatestManifestVersion(baseUrl) {
        return Manifest.findLatestManifestVersion(baseUrl);
    }

    static _parseManifestForIndex(bytes) {
        return Manifest.parseManifestForIndex(bytes);
    }

    static _parseIndexMetadata(bytes) {
        return Manifest.parseIndexMetadata(bytes);
    }

    static _parseUuid(bytes) {
        return Manifest.parseUuid(bytes);
    }

    static _parseIndexFile(bytes, indexInfo) {
        return Manifest.parseIndexFile(bytes, indexInfo, IVFIndex);
    }

    static _findIVFMessage(bytes) {
        return Manifest.findIVFMessage(bytes);
    }

    static _tryParseCentroids(bytes) {
        return Manifest.tryParseCentroids(bytes);
    }
}

export { IVFIndex };
