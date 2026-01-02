import * as Manifest from './ivf-manifest.js';
import * as Auxiliary from './ivf-auxiliary.js';
import * as Partitions from './ivf-partitions.js';

/**
 * Quickselect algorithm to find top-k elements in O(n) average time.
 * Partitions array in-place so indices 0..k-1 contain the k largest elements.
 * @param {Array} arr - Array of objects with score property
 * @param {number} k - Number of top elements to select
 * @returns {Array} The top k elements (unordered)
 */
function quickselectTopK(arr, k) {
    if (k >= arr.length) return arr;
    if (k <= 0) return [];

    let left = 0;
    let right = arr.length - 1;

    while (left < right) {
        // Choose pivot using median-of-three for better performance
        const mid = (left + right) >> 1;
        if (arr[mid].score > arr[left].score) swap(arr, left, mid);
        if (arr[right].score > arr[left].score) swap(arr, left, right);
        if (arr[mid].score > arr[right].score) swap(arr, mid, right);
        const pivot = arr[right].score;

        let i = left;
        for (let j = left; j < right; j++) {
            if (arr[j].score >= pivot) {
                swap(arr, i, j);
                i++;
            }
        }
        swap(arr, i, right);

        if (i === k - 1) break;
        if (i < k - 1) left = i + 1;
        else right = i - 1;
    }

    return arr.slice(0, k);
}

function swap(arr, i, j) {
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
}

class IVFIndex {
    constructor() {
        this.centroids = null;
        this.numPartitions = 0;
        this.dimension = 0;
        this.partitionOffsets = [];
        this.partitionLengths = [];
        this.metricType = 'cosine';
        this.partitionIndexUrl = null;
        this.partitionStarts = null;
        this.hasPartitionIndex = false;
        this._rowIdCache = null;
        this._rowIdCacheReady = false;
        this._accessCounts = new Map();  // partition -> access count for prefetching
    }

    static async tryLoad(datasetBaseUrl) {
        if (!datasetBaseUrl) return null;

        try {
            const manifestVersion = await Manifest.findLatestManifestVersion(datasetBaseUrl);
            if (!manifestVersion) return null;

            const manifestUrl = `${datasetBaseUrl}/_versions/${manifestVersion}.manifest`;
            const manifestResp = await fetch(manifestUrl);
            if (!manifestResp.ok) return null;

            const manifestData = await manifestResp.arrayBuffer();
            const indexInfo = Manifest.parseManifestForIndex(new Uint8Array(manifestData));
            if (!indexInfo?.uuid) return null;

            const indexUrl = `${datasetBaseUrl}/_indices/${indexInfo.uuid}/index.idx`;
            const indexResp = await fetch(indexUrl);
            if (!indexResp.ok) return null;

            const indexData = await indexResp.arrayBuffer();
            const index = Manifest.parseIndexFile(new Uint8Array(indexData), indexInfo, IVFIndex);
            if (!index) return null;

            index.auxiliaryUrl = `${datasetBaseUrl}/_indices/${indexInfo.uuid}/auxiliary.idx`;
            index.datasetBaseUrl = datasetBaseUrl;

            try { await Auxiliary.loadAuxiliaryMetadata(index); } catch {}
            try { await Partitions.loadPartitionIndex(index); } catch {}
            try { await Partitions.prefetchAllRowIds(index); } catch {}

            return index;
        } catch {
            return null;
        }
    }

    async _loadPartitionIndex() {
        return Partitions.loadPartitionIndex(this);
    }

    fetchPartitionData(partitionIndices, dim = 384, onProgress = null) {
        return Partitions.fetchPartitionData(this, partitionIndices, dim, onProgress);
    }

    async _loadAuxiliaryMetadata() {
        return Auxiliary.loadAuxiliaryMetadata(this);
    }

    _parseColumnMetaForPartitions(bytes) {
        return Auxiliary.parseColumnMetaForPartitions(this, bytes);
    }

    _parseAuxiliaryPartitionInfo(bytes) {
        return Auxiliary.parseAuxiliaryPartitionInfo(this, bytes);
    }

    async prefetchAllRowIds() {
        return Partitions.prefetchAllRowIds(this);
    }

    fetchPartitionRowIds(partitionIndices) {
        return Partitions.fetchPartitionRowIds(this, partitionIndices);
    }

    getPartitionRowCount(partitionIndices) {
        return Partitions.getPartitionRowCount(this, partitionIndices);
    }

    findNearestPartitions(queryVec, nprobe = 10) {
        if (!this.centroids || queryVec.length !== this.dimension) return [];

        nprobe = Math.min(nprobe, this.numPartitions);
        const distances = new Array(this.numPartitions);

        // Pre-compute query norm once
        let normA = 0;
        for (let i = 0; i < this.dimension; i++) {
            normA += queryVec[i] * queryVec[i];
        }
        const sqrtNormA = Math.sqrt(normA);

        for (let p = 0; p < this.numPartitions; p++) {
            const start = p * this.dimension;
            let dot = 0, normB = 0;
            for (let i = 0; i < this.dimension; i++) {
                const b = this.centroids[start + i];
                dot += queryVec[i] * b;
                normB += b * b;
            }
            const denom = sqrtNormA * Math.sqrt(normB);
            distances[p] = { idx: p, score: denom === 0 ? 0 : dot / denom };
        }

        // Use O(n) quickselect instead of O(n log n) sort
        const topK = quickselectTopK(distances, nprobe);
        const partitionIndices = topK.map(d => d.idx);

        // Track partition access counts for prefetching
        for (const idx of partitionIndices) {
            this._accessCounts.set(idx, (this._accessCounts.get(idx) || 0) + 1);
        }

        return partitionIndices;
    }

    /**
     * Prefetch frequently accessed partitions into cache.
     * Call after several searches to warm the cache with hot partitions.
     * @param {number} topN - Number of top partitions to prefetch (default 10)
     * @param {number} minAccesses - Minimum access count to be considered hot (default 3)
     */
    async prefetchHotPartitions(topN = 10, minAccesses = 3) {
        if (!this._accessCounts || this._accessCounts.size === 0) return;

        const hotPartitions = [...this._accessCounts.entries()]
            .filter(([_, count]) => count >= minAccesses)
            .sort((a, b) => b[1] - a[1])
            .slice(0, topN)
            .map(([p]) => p);

        if (hotPartitions.length === 0) return;

        // Prefetch partitions into LRU cache
        await this.fetchPartitionData(hotPartitions, this.dimension);
    }

    /**
     * Get access statistics for debugging/monitoring.
     */
    getAccessStats() {
        return {
            totalPartitions: this.numPartitions,
            accessedPartitions: this._accessCounts.size,
            topPartitions: [...this._accessCounts.entries()]
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10)
        };
    }
}

export { IVFIndex };
