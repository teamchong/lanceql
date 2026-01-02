/**
 * IVF Partition Data Fetching
 * Handles loading partition vectors and row IDs
 */

/**
 * Load partition-organized vectors index from ivf_vectors.bin.
 * This file contains:
 *   - Header: 257 uint64 byte offsets (2056 bytes)
 *   - Per partition: [row_count: uint32][row_ids: uint32 × n][vectors: float32 × n × 384]
 * @param {IVFIndex} index - The IVF index instance
 */
export async function loadPartitionIndex(index) {
    const url = `${index.datasetBaseUrl}/ivf_vectors.bin`;
    index.partitionVectorsUrl = url;

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
    index.partitionOffsets = Array.from(bigOffsets, n => Number(n));

    index.hasPartitionIndex = true;
    console.log(`[IVFIndex] Loaded partition vectors index: 256 partitions`);
}

/**
 * Fetch partition data (row IDs and vectors) directly from ivf_vectors.bin.
 * Uses in-memory cache for instant subsequent searches.
 * Each partition contains: [row_count: uint32][row_ids: uint32 × n][vectors: float32 × n × dim]
 * @param {IVFIndex} index - The IVF index instance
 * @param {number[]} partitionIndices - Partition indices to fetch
 * @param {number} dim - Vector dimension (default 384)
 * @param {function} onProgress - Progress callback (bytesLoaded, totalBytes)
 * @returns {Promise<{rowIds: number[], vectors: Float32Array[]}>}
 */
export async function fetchPartitionData(index, partitionIndices, dim = 384, onProgress = null) {
    if (!index.hasPartitionIndex || !index.partitionVectorsUrl) {
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
        if (index._partitionCache?.has(p)) {
            cachedResults.set(p, index._partitionCache.get(p));
        } else {
            uncachedPartitions.push(p);
            const startOffset = index.partitionOffsets[p];
            const endOffset = index.partitionOffsets[p + 1];
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
    if (!index._partitionCache) {
        index._partitionCache = new Map();
    }

    // Fetch uncached partitions in parallel (max 6 concurrent for speed)
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
            index._partitionCache.set(p, { rowIds, vectors });
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
 * Prefetch ALL row IDs from auxiliary.idx into memory.
 * This is called once during index loading to avoid HTTP requests during search.
 * @param {IVFIndex} index - The IVF index instance
 * @returns {Promise<void>}
 */
export async function prefetchAllRowIds(index) {
    if (!index.auxiliaryUrl || !index._auxBufferOffsets) {
        console.log('[IVFIndex] No auxiliary.idx available for prefetch');
        return;
    }

    if (index._rowIdCacheReady) {
        console.log('[IVFIndex] Row IDs already prefetched');
        return;
    }

    const totalRows = index.partitionLengths.reduce((a, b) => a + b, 0);
    if (totalRows === 0) {
        console.log('[IVFIndex] No rows to prefetch');
        return;
    }

    console.log(`[IVFIndex] Prefetching ${totalRows.toLocaleString()} row IDs...`);
    const startTime = performance.now();

    const dataStart = index._auxBufferOffsets[1];
    const totalBytes = totalRows * 8;

    try {
        // Fetch ALL row IDs in a single request
        const resp = await fetch(index.auxiliaryUrl, {
            headers: { 'Range': `bytes=${dataStart}-${dataStart + totalBytes - 1}` }
        });

        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }

        const data = new Uint8Array(await resp.arrayBuffer());
        const view = new DataView(data.buffer, data.byteOffset);

        // Parse and organize by partition
        index._rowIdCache = new Map();
        let globalRowIdx = 0;

        for (let p = 0; p < index.partitionLengths.length; p++) {
            const numRows = index.partitionLengths[p];
            const partitionRows = [];

            for (let i = 0; i < numRows; i++) {
                const rowId = Number(view.getBigUint64(globalRowIdx * 8, true));
                const fragId = Math.floor(rowId / 0x100000000);
                const rowOffset = rowId % 0x100000000;
                partitionRows.push({ fragId, rowOffset });
                globalRowIdx++;
            }

            index._rowIdCache.set(p, partitionRows);
        }

        index._rowIdCacheReady = true;
        const elapsed = performance.now() - startTime;
        console.log(`[IVFIndex] Prefetched ${totalRows.toLocaleString()} row IDs in ${elapsed.toFixed(0)}ms (${(totalBytes / 1024 / 1024).toFixed(1)}MB)`);
    } catch (e) {
        console.warn('[IVFIndex] Failed to prefetch row IDs:', e);
    }
}

/**
 * Fetch row IDs for specified partitions.
 * Uses prefetched cache if available (instant), otherwise fetches from network.
 * @param {IVFIndex} index - The IVF index instance
 * @param {number[]} partitionIndices - Partition indices to fetch
 * @returns {Promise<Array<{fragId: number, rowOffset: number}>>}
 */
export async function fetchPartitionRowIds(index, partitionIndices) {
    // Fast path: use prefetched cache
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

    // Slow path: fetch from network (fallback if prefetch failed)
    if (!index.auxiliaryUrl || !index._auxBufferOffsets) {
        return null;
    }

    const rowRanges = [];
    for (const p of partitionIndices) {
        if (p < index.partitionOffsets.length) {
            const startRow = index.partitionOffsets[p];
            const numRows = index.partitionLengths[p];
            rowRanges.push({ partition: p, startRow, numRows });
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
 * @param {IVFIndex} index - The IVF index instance
 * @param {number[]} partitionIndices - Partition indices
 * @returns {number}
 */
export function getPartitionRowCount(index, partitionIndices) {
    let total = 0;
    for (const p of partitionIndices) {
        if (p < index.partitionLengths.length) {
            total += index.partitionLengths[p];
        }
    }
    return total;
}
