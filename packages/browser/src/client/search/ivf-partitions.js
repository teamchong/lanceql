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

    const allRowIds = [];
    const allVectors = [];
    let totalBytesToFetch = 0;
    let bytesLoaded = 0;

    const uncachedPartitions = [];
    const cachedResults = new Map();

    for (const p of partitionIndices) {
        if (index._partitionCache?.has(p)) {
            cachedResults.set(p, index._partitionCache.get(p));
        } else {
            uncachedPartitions.push(p);
            totalBytesToFetch += index.partitionOffsets[p + 1] - index.partitionOffsets[p];
        }
    }

    if (uncachedPartitions.length === 0) {
        for (const p of partitionIndices) {
            const result = cachedResults.get(p);
            allRowIds.push(...result.rowIds);
            allVectors.push(...result.vectors);
        }
        if (onProgress) onProgress(100, 100);
        return { rowIds: allRowIds, vectors: allVectors };
    }

    if (!index._partitionCache) index._partitionCache = new Map();

    const PARALLEL_LIMIT = 6;
    for (let i = 0; i < uncachedPartitions.length; i += PARALLEL_LIMIT) {
        const batch = uncachedPartitions.slice(i, i + PARALLEL_LIMIT);

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
                const vectorsFlat = new Float32Array(data.slice(rowIdsEnd));

                const vectors = [];
                for (let j = 0; j < rowCount; j++) {
                    vectors.push(vectorsFlat.slice(j * dim, (j + 1) * dim));
                }

                bytesLoaded += byteSize;
                if (onProgress) onProgress(bytesLoaded, totalBytesToFetch);

                return { p, rowIds: Array.from(rowIds), vectors };
            } catch {
                return { p, rowIds: [], vectors: [] };
            }
        }));

        for (const result of results) {
            index._partitionCache.set(result.p, { rowIds: result.rowIds, vectors: result.vectors });
            cachedResults.set(result.p, { rowIds: result.rowIds, vectors: result.vectors });
        }
    }

    for (const p of partitionIndices) {
        const result = cachedResults.get(p);
        if (result) {
            allRowIds.push(...result.rowIds);
            allVectors.push(...result.vectors);
        }
    }

    return { rowIds: allRowIds, vectors: allVectors };
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
