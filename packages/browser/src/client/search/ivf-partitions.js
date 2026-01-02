import { LRUCache } from '../cache/lru-cache.js';

// Default 50MB cache for partition data
const PARTITION_CACHE_SIZE = 50 * 1024 * 1024;

export async function loadPartitionIndex(index) {
    const url = `${index.datasetBaseUrl}/ivf_vectors.bin`;
    index.partitionVectorsUrl = url;

    const headerResp = await fetch(url, { headers: { 'Range': 'bytes=0-2055' } });
    if (!headerResp.ok) return;

    const headerData = await headerResp.arrayBuffer();
    const bigOffsets = new BigUint64Array(headerData);
    index.partitionOffsets = Array.from(bigOffsets, n => Number(n));
    index.hasPartitionIndex = true;
}

export async function fetchPartitionData(index, partitionIndices, dim = 384, onProgress = null) {
    if (!index.hasPartitionIndex || !index.partitionVectorsUrl) return null;

    let totalBytesToFetch = 0;
    let bytesLoaded = 0;

    const uncachedPartitions = [];
    const cachedResults = new Map();

    // Initialize LRU cache if not present
    if (!index._partitionCache) {
        index._partitionCache = new LRUCache({ maxSize: PARTITION_CACHE_SIZE });
    }

    for (const p of partitionIndices) {
        const cached = index._partitionCache.get(p);
        if (cached !== undefined) {
            cachedResults.set(p, cached);
        } else {
            uncachedPartitions.push(p);
            totalBytesToFetch += index.partitionOffsets[p + 1] - index.partitionOffsets[p];
        }
    }

    if (uncachedPartitions.length === 0) {
        return assembleResults(partitionIndices, cachedResults, dim, onProgress);
    }

    // Initialize adaptive fetch stats if not present
    if (!index._fetchStats) {
        index._fetchStats = {
            concurrency: 6,
            recentLatencies: [],  // Rolling window of last 10 batches
            minConcurrency: 2,
            maxConcurrency: 12
        };
    }

    const stats = index._fetchStats;
    for (let i = 0; i < uncachedPartitions.length; i += stats.concurrency) {
        const batch = uncachedPartitions.slice(i, i + stats.concurrency);
        const batchStart = performance.now();

        const results = await Promise.all(batch.map(async (p) => {
            const startOffset = index.partitionOffsets[p];
            const endOffset = index.partitionOffsets[p + 1];
            const byteSize = endOffset - startOffset;

            try {
                const resp = await fetch(index.partitionVectorsUrl, {
                    headers: { 'Range': `bytes=${startOffset}-${endOffset - 1}` }
                });
                if (!resp.ok) return { p, rowIds: [], vectors: [] };

                const data = await resp.arrayBuffer();
                const view = new DataView(data);
                const rowCount = view.getUint32(0, true);
                const rowIdsEnd = 4 + rowCount * 4;

                const rowIds = new Uint32Array(data.slice(4, rowIdsEnd));
                // Keep vectors as flat Float32Array to avoid allocation overhead
                const vectorsFlat = new Float32Array(data.slice(rowIdsEnd));

                bytesLoaded += byteSize;
                if (onProgress) onProgress(bytesLoaded, totalBytesToFetch);

                return { p, rowIds: Array.from(rowIds), vectors: vectorsFlat, numVectors: rowCount };
            } catch {
                return { p, rowIds: [], vectors: [] };
            }
        }));

        for (const result of results) {
            const data = {
                rowIds: result.rowIds,
                vectors: result.vectors,
                numVectors: result.numVectors ?? result.rowIds.length
            };
            // Estimate size: rowIds (4 bytes each) + flat vectors buffer
            const size = result.rowIds.length * 4 + (result.vectors.byteLength || result.vectors.length * 4);
            index._partitionCache.set(result.p, data, size);
            cachedResults.set(result.p, data);
        }

        // Adaptive concurrency adjustment based on batch latency
        const batchLatency = performance.now() - batchStart;
        stats.recentLatencies.push(batchLatency);
        if (stats.recentLatencies.length > 10) stats.recentLatencies.shift();

        const avgLatency = stats.recentLatencies.reduce((a, b) => a + b, 0) / stats.recentLatencies.length;
        if (avgLatency < 50 && stats.concurrency < stats.maxConcurrency) {
            stats.concurrency++;
        } else if (avgLatency > 200 && stats.concurrency > stats.minConcurrency) {
            stats.concurrency--;
        }
    }

    return assembleResults(partitionIndices, cachedResults, dim, onProgress);
}

/**
 * Assemble results from cached partitions into a single flat output.
 * Returns flat Float32Array for vectors to avoid allocation overhead.
 */
function assembleResults(partitionIndices, cachedResults, dim, onProgress) {
    // Calculate total size
    let totalRowIds = 0;
    let totalVectorElements = 0;

    for (const p of partitionIndices) {
        const result = cachedResults.get(p);
        if (result) {
            totalRowIds += result.rowIds.length;
            totalVectorElements += result.vectors.length;
        }
    }

    // Pre-allocate arrays
    const allRowIds = new Array(totalRowIds);
    const allVectors = new Float32Array(totalVectorElements);

    let rowIdOffset = 0;
    let vectorOffset = 0;

    for (const p of partitionIndices) {
        const result = cachedResults.get(p);
        if (result) {
            // Copy row IDs
            for (let i = 0; i < result.rowIds.length; i++) {
                allRowIds[rowIdOffset++] = result.rowIds[i];
            }
            // Copy vectors (already flat)
            allVectors.set(result.vectors, vectorOffset);
            vectorOffset += result.vectors.length;
        }
    }

    if (onProgress) onProgress(100, 100);
    return { rowIds: allRowIds, vectors: allVectors, preFlattened: true };
}

export async function prefetchAllRowIds(index) {
    if (!index.auxiliaryUrl || !index._auxBufferOffsets) return;
    if (index._rowIdCacheReady) return;

    const totalRows = index.partitionLengths.reduce((a, b) => a + b, 0);
    if (totalRows === 0) return;

    const dataStart = index._auxBufferOffsets[1];
    const totalBytes = totalRows * 8;

    try {
        const resp = await fetch(index.auxiliaryUrl, {
            headers: { 'Range': `bytes=${dataStart}-${dataStart + totalBytes - 1}` }
        });
        if (!resp.ok) return;

        const data = new Uint8Array(await resp.arrayBuffer());
        const view = new DataView(data.buffer, data.byteOffset);

        index._rowIdCache = new Map();
        let globalRowIdx = 0;

        for (let p = 0; p < index.partitionLengths.length; p++) {
            const numRows = index.partitionLengths[p];
            const partitionRows = [];

            for (let i = 0; i < numRows; i++) {
                const rowId = Number(view.getBigUint64(globalRowIdx * 8, true));
                partitionRows.push({
                    fragId: Math.floor(rowId / 0x100000000),
                    rowOffset: rowId % 0x100000000
                });
                globalRowIdx++;
            }

            index._rowIdCache.set(p, partitionRows);
        }

        index._rowIdCacheReady = true;
    } catch {}
}

export async function fetchPartitionRowIds(index, partitionIndices) {
    if (index._rowIdCacheReady && index._rowIdCache) {
        const results = [];
        for (const p of partitionIndices) {
            const cached = index._rowIdCache.get(p);
            if (cached) {
                for (const row of cached) {
                    results.push({ ...row, partition: p });
                }
            }
        }
        return results;
    }

    if (!index.auxiliaryUrl || !index._auxBufferOffsets) return null;

    const rowRanges = [];
    for (const p of partitionIndices) {
        if (p < index.partitionOffsets.length) {
            rowRanges.push({
                partition: p,
                startRow: index.partitionOffsets[p],
                numRows: index.partitionLengths[p]
            });
        }
    }

    if (rowRanges.length === 0) return [];

    const results = [];
    const dataStart = index._auxBufferOffsets[1];

    for (const range of rowRanges) {
        const byteStart = dataStart + range.startRow * 8;
        const byteEnd = byteStart + range.numRows * 8 - 1;

        try {
            const resp = await fetch(index.auxiliaryUrl, {
                headers: { 'Range': `bytes=${byteStart}-${byteEnd}` }
            });
            if (!resp.ok) continue;

            const data = new Uint8Array(await resp.arrayBuffer());
            const view = new DataView(data.buffer, data.byteOffset);

            for (let i = 0; i < range.numRows; i++) {
                const rowId = Number(view.getBigUint64(i * 8, true));
                results.push({
                    fragId: Math.floor(rowId / 0x100000000),
                    rowOffset: rowId % 0x100000000,
                    partition: range.partition
                });
            }
        } catch {}
    }

    return results;
}

export function getPartitionRowCount(index, partitionIndices) {
    let total = 0;
    for (const p of partitionIndices) {
        if (p < index.partitionLengths.length) {
            total += index.partitionLengths[p];
        }
    }
    return total;
}
