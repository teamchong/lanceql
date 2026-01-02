/**
 * RemoteLanceDataset - Fragment operations
 * Extracted from remote-dataset.js for modularity
 */

/**
 * Helper to determine which fragment contains a given row index.
 * @param {Object} dataset - Dataset instance
 * @param {number} rowIdx - Global row index
 * @returns {{fragmentIndex: number, localIndex: number}|null}
 */
export function getFragmentForRow(dataset, rowIdx) {
    let offset = 0;
    for (let i = 0; i < dataset._fragments.length; i++) {
        const frag = dataset._fragments[i];
        if (rowIdx < offset + frag.numRows) {
            return { fragmentIndex: i, localIndex: rowIdx - offset };
        }
        offset += frag.numRows;
    }
    return null;
}

/**
 * Group indices by fragment for efficient batch reading.
 * @param {Object} dataset - Dataset instance
 * @param {number[]} indices - Global row indices
 * @returns {Map<number, {localIndices: number[], globalIndices: number[]}>}
 */
export function groupIndicesByFragment(dataset, indices) {
    const groups = new Map();
    for (const globalIdx of indices) {
        const loc = getFragmentForRow(dataset, globalIdx);
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
 * Read rows from the dataset with pagination.
 * Fetches from multiple fragments in parallel.
 * @param {Object} dataset - Dataset instance
 * @param {Object} options - Query options
 * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
 */
export async function readRows(dataset, { offset = 0, limit = 50, columns = null, _isPrefetch = false } = {}) {
    // Determine which fragments contain the requested rows
    const fragmentRanges = [];
    let currentOffset = 0;

    for (let i = 0; i < dataset._fragments.length; i++) {
        const frag = dataset._fragments[i];
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
        return { columns: [], columnNames: dataset.columnNames, total: dataset._totalRows };
    }

    // Fetch from fragments in parallel
    const fetchPromises = fragmentRanges.map(async (range) => {
        const file = await dataset.openFragment(range.fragmentIndex);
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
    const colNames = results[0]?.result.columnNames || dataset.columnNames;
    const numCols = columns ? columns.length : dataset._numColumns;

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
        total: dataset._totalRows
    };

    // Speculative prefetch: if there are more rows, prefetch next page in background
    const nextOffset = offset + limit;
    if (!_isPrefetch && nextOffset < dataset._totalRows && limit <= 100) {
        prefetchNextPage(dataset, nextOffset, limit, columns);
    }

    return result;
}

/**
 * Prefetch next page of rows in background.
 * @param {Object} dataset - Dataset instance
 * @param {number} offset - Next page offset
 * @param {number} limit - Page size
 * @param {number[]|null} columns - Column indices
 */
export function prefetchNextPage(dataset, offset, limit, columns) {
    // Use a cache key to avoid duplicate prefetches
    const cacheKey = `${offset}-${limit}-${columns?.join(',') || 'all'}`;
    if (dataset._prefetchCache?.has(cacheKey)) {
        return; // Already prefetching or prefetched
    }

    if (!dataset._prefetchCache) {
        dataset._prefetchCache = new Map();
    }

    // Start prefetch in background (don't await)
    const prefetchPromise = readRows(dataset, { offset, limit, columns, _isPrefetch: true })
        .then(result => {
            dataset._prefetchCache.set(cacheKey, result);
            console.log(`[LanceQL] Prefetched rows ${offset}-${offset + limit}`);
        })
        .catch(() => {
            // Ignore prefetch errors
        });

    dataset._prefetchCache.set(cacheKey, prefetchPromise);
}

/**
 * Read strings at specific indices across fragments.
 * @param {Object} dataset - Dataset instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Global row indices
 * @returns {Promise<string[]>}
 */
export async function readStringsAtIndices(dataset, colIdx, indices) {
    const groups = groupIndicesByFragment(dataset, indices);
    const results = new Map();

    console.log(`[ReadStrings] Reading ${indices.length} strings from col ${colIdx}`);
    console.log(`[ReadStrings] First 5 indices: ${indices.slice(0, 5)}`);
    console.log(`[ReadStrings] Fragment groups: ${Array.from(groups.keys())}`);

    // Fetch from each fragment in parallel
    const fetchPromises = [];
    for (const [fragIdx, group] of groups) {
        fetchPromises.push((async () => {
            const file = await dataset.openFragment(fragIdx);
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
 * @param {Object} dataset - Dataset instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Global row indices
 * @returns {Promise<BigInt64Array>}
 */
export async function readInt64AtIndices(dataset, colIdx, indices) {
    const groups = groupIndicesByFragment(dataset, indices);
    const results = new Map();

    const fetchPromises = [];
    for (const [fragIdx, group] of groups) {
        fetchPromises.push((async () => {
            const file = await dataset.openFragment(fragIdx);
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
 * @param {Object} dataset - Dataset instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Global row indices
 * @returns {Promise<Float64Array>}
 */
export async function readFloat64AtIndices(dataset, colIdx, indices) {
    const groups = groupIndicesByFragment(dataset, indices);
    const results = new Map();

    const fetchPromises = [];
    for (const [fragIdx, group] of groups) {
        fetchPromises.push((async () => {
            const file = await dataset.openFragment(fragIdx);
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
 * @param {Object} dataset - Dataset instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Global row indices
 * @returns {Promise<Int32Array>}
 */
export async function readInt32AtIndices(dataset, colIdx, indices) {
    const groups = groupIndicesByFragment(dataset, indices);
    const results = new Map();

    const fetchPromises = [];
    for (const [fragIdx, group] of groups) {
        fetchPromises.push((async () => {
            const file = await dataset.openFragment(fragIdx);
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
 * @param {Object} dataset - Dataset instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Global row indices
 * @returns {Promise<Float32Array>}
 */
export async function readFloat32AtIndices(dataset, colIdx, indices) {
    const groups = groupIndicesByFragment(dataset, indices);
    const results = new Map();

    const fetchPromises = [];
    for (const [fragIdx, group] of groups) {
        fetchPromises.push((async () => {
            const file = await dataset.openFragment(fragIdx);
            const data = await file.readFloat32AtIndices(colIdx, group.localIndices);
            for (let i = 0; i < group.globalIndices.length; i++) {
                results.set(group.globalIndices[i], data[i]);
            }
        })());
    }
    await Promise.all(fetchPromises);

    return new Float32Array(indices.map(idx => results.get(idx) || 0));
}
