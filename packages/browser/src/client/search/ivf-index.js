import * as Manifest from './ivf-manifest.js';
import * as Auxiliary from './ivf-auxiliary.js';
import * as Partitions from './ivf-partitions.js';

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

        for (let p = 0; p < this.numPartitions; p++) {
            const start = p * this.dimension;
            let dot = 0, normA = 0, normB = 0;
            for (let i = 0; i < this.dimension; i++) {
                const a = queryVec[i];
                const b = this.centroids[start + i];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
            const denom = Math.sqrt(normA) * Math.sqrt(normB);
            distances[p] = { idx: p, score: denom === 0 ? 0 : dot / denom };
        }

        distances.sort((a, b) => b.score - a.score);
        return distances.slice(0, nprobe).map(d => d.idx);
    }
}

export { IVFIndex };
