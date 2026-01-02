/**
 * RemoteLanceFile - Vector operations and ANN search
 * Extracted from remote-file.js for modularity
 */

import { batchIndices } from './remote-file-proto.js';
import { webgpuAccelerator } from '../gpu/accelerator.js';

/**
 * Get vector info for a column via Range requests.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @returns {Promise<{rows: number, dimension: number}>}
 */
export async function getVectorInfo(file, colIdx) {
    const entry = await file.getColumnOffsetEntry(colIdx);
    if (entry.len === 0) return { rows: 0, dimension: 0 };

    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = file._parseColumnMeta(new Uint8Array(colMeta));

    if (info.rows === 0) return { rows: 0, dimension: 0 };

    // Calculate dimension from first page (all pages have same dimension)
    let dimension = 0;
    if (info.pages && info.pages.length > 0) {
        const firstPage = info.pages[0];
        const dataIdx = firstPage.sizes.length > 1 ? 1 : 0;
        const pageSize = firstPage.sizes[dataIdx] || 0;
        const pageRows = firstPage.rows || 0;
        if (pageRows > 0 && pageSize > 0) {
            dimension = Math.floor(pageSize / (pageRows * 4));
        }
    } else if (info.size > 0) {
        // Fallback for single-page
        dimension = Math.floor(info.size / (info.rows * 4));
    }

    return { rows: info.rows, dimension };
}

/**
 * Read a single vector at index via Range requests.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number} rowIdx - Row index
 * @returns {Promise<Float32Array>}
 */
export async function readVectorAt(file, colIdx, rowIdx) {
    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = file._parseColumnMeta(new Uint8Array(colMeta));

    if (info.rows === 0) return new Float32Array(0);
    if (rowIdx >= info.rows) return new Float32Array(0);

    const dim = Math.floor(info.size / (info.rows * 4));
    if (dim === 0) return new Float32Array(0);

    // Fetch the vector data
    const vecStart = info.offset + rowIdx * dim * 4;
    const vecEnd = vecStart + dim * 4 - 1;
    const data = await file.fetchRange(vecStart, vecEnd);

    return new Float32Array(data);
}

/**
 * Read multiple vectors at indices via Range requests.
 * Uses batched fetching for efficiency.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {number[]} indices - Row indices
 * @returns {Promise<Float32Array[]>}
 */
export async function readVectorsAtIndices(file, colIdx, indices) {
    if (indices.length === 0) return [];

    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const info = file._parseColumnMeta(new Uint8Array(colMeta));

    if (info.rows === 0) return indices.map(() => new Float32Array(0));

    const dim = Math.floor(info.size / (info.rows * 4));
    if (dim === 0) return indices.map(() => new Float32Array(0));

    const vecSize = dim * 4;
    const results = new Array(indices.length);

    // Batch indices for efficient fetching - parallel with limit
    const batches = batchIndices(indices, vecSize, vecSize * 50);
    const BATCH_PARALLEL = 6;

    for (let i = 0; i < batches.length; i += BATCH_PARALLEL) {
        const batchGroup = batches.slice(i, i + BATCH_PARALLEL);
        await Promise.all(batchGroup.map(async (batch) => {
            try {
                const startOffset = info.offset + batch.startIdx * vecSize;
                const endOffset = info.offset + (batch.endIdx + 1) * vecSize - 1;
                const data = await file.fetchRange(startOffset, endOffset);

                for (const item of batch.items) {
                    const localOffset = (item.idx - batch.startIdx) * vecSize;
                    results[item.origPos] = new Float32Array(
                        data.slice(localOffset, localOffset + vecSize)
                    );
                }
            } catch (e) {
                for (const item of batch.items) {
                    results[item.origPos] = new Float32Array(0);
                }
            }
        }));
    }

    return results;
}

/**
 * Compute cosine similarity between two vectors.
 * @param {Float32Array} vecA - First vector
 * @param {Float32Array} vecB - Second vector
 * @returns {number} - Similarity score
 */
export function cosineSimilarity(vecA, vecB) {
    if (vecA.length !== vecB.length) return 0;

    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dot += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
}

/**
 * Find top-k most similar vectors to query via Range requests.
 * NOTE: Requires IVF index for efficient querying.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index with vectors
 * @param {Float32Array} queryVec - Query vector
 * @param {number} topK - Number of results to return
 * @param {function} onProgress - Progress callback(current, total)
 * @param {object} options - Search options
 * @returns {Promise<{indices: number[], scores: number[], usedIndex: boolean}>}
 */
export async function vectorSearch(file, colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
    const { nprobe = 10 } = options;

    const info = await getVectorInfo(file, colIdx);
    if (info.dimension === 0 || info.dimension !== queryVec.length) {
        throw new Error(`Dimension mismatch: query=${queryVec.length}, column=${info.dimension}`);
    }

    // Require IVF index - no brute force fallback
    if (!file.hasIndex()) {
        throw new Error('No IVF index found. Vector search requires an IVF index for efficient querying.');
    }

    if (file._ivfIndex.dimension !== queryVec.length) {
        throw new Error(`Query dimension (${queryVec.length}) does not match index dimension (${file._ivfIndex.dimension}).`);
    }

    return await vectorSearchWithIndex(file, colIdx, queryVec, topK, nprobe, onProgress);
}

/**
 * Vector search using IVF index (ANN).
 * Fetches row IDs from auxiliary.idx for nearest partitions,
 * then looks up original vectors by fragment/offset.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {Float32Array} queryVec - Query vector
 * @param {number} topK - Number of results
 * @param {number} nprobe - Number of partitions to search
 * @param {function} onProgress - Progress callback
 * @returns {Promise<Object>}
 */
async function vectorSearchWithIndex(file, colIdx, queryVec, topK, nprobe, onProgress) {
    // Find nearest partitions using centroids
    if (onProgress) onProgress(0, 100);
    const partitions = file._ivfIndex.findNearestPartitions(queryVec, nprobe);
    const estimatedRows = file._ivfIndex.getPartitionRowCount(partitions);

    console.log(`[IVFSearch] Searching ${partitions.length} partitions (~${estimatedRows.toLocaleString()} rows)`);

    // Try to fetch row IDs from auxiliary.idx
    const rowIdMappings = await file._ivfIndex.fetchPartitionRowIds(partitions);

    if (rowIdMappings && rowIdMappings.length > 0) {
        console.log(`[IVFSearch] Fetched ${rowIdMappings.length} row ID mappings`);
        return await searchWithRowIdMappings(file, colIdx, queryVec, topK, rowIdMappings, onProgress);
    }

    throw new Error('Failed to fetch row IDs from IVF index. Dataset may be missing auxiliary.idx or ivf_partitions.bin.');
}

/**
 * Search using proper row ID mappings from auxiliary.idx.
 * Groups row IDs by fragment and fetches vectors efficiently.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Column index
 * @param {Float32Array} queryVec - Query vector
 * @param {number} topK - Number of results
 * @param {Array} rowIdMappings - Row ID mappings
 * @param {function} onProgress - Progress callback
 * @returns {Promise<Object>}
 */
async function searchWithRowIdMappings(file, colIdx, queryVec, topK, rowIdMappings, onProgress) {
    const dim = queryVec.length;

    // Group row IDs by fragment for efficient batch fetching
    const byFragment = new Map();
    for (const mapping of rowIdMappings) {
        if (!byFragment.has(mapping.fragId)) {
            byFragment.set(mapping.fragId, []);
        }
        byFragment.get(mapping.fragId).push(mapping.rowOffset);
    }

    console.log(`[IVFSearch] Fetching from ${byFragment.size} fragments`);

    // Collect all vectors and their indices first
    const allVectors = [];
    const allIndices = [];
    let processed = 0;
    const total = rowIdMappings.length;

    // Fetch all vectors
    for (const [fragId, offsets] of byFragment) {
        if (onProgress) onProgress(processed, total);

        const vectors = await readVectorsAtIndices(file, colIdx, offsets);

        for (let i = 0; i < offsets.length; i++) {
            const vec = vectors[i];
            if (vec && vec.length === dim) {
                allVectors.push(vec);
                // Reconstruct global row index
                allIndices.push(fragId * 50000 + offsets[i]);
            }
            processed++;
        }
    }

    // Try WebGPU first, fallback to WASM SIMD
    let scores;
    if (webgpuAccelerator.isAvailable()) {
        console.log(`[IVFSearch] Computing similarity for ${allVectors.length} vectors via WebGPU`);
        scores = await webgpuAccelerator.batchCosineSimilarity(queryVec, allVectors, true);
    }

    if (!scores) {
        console.log(`[IVFSearch] Computing similarity for ${allVectors.length} vectors via WASM SIMD`);
        scores = file.lanceql.batchCosineSimilarity(queryVec, allVectors, true);
    }

    // Find top-k
    const topResults = [];
    for (let i = 0; i < scores.length; i++) {
        const score = scores[i];
        const idx = allIndices[i];

        if (topResults.length < topK) {
            topResults.push({ idx, score });
            topResults.sort((a, b) => b.score - a.score);
        } else if (score > topResults[topK - 1].score) {
            topResults[topK - 1] = { idx, score };
            topResults.sort((a, b) => b.score - a.score);
        }
    }

    if (onProgress) onProgress(total, total);

    return {
        indices: topResults.map(r => r.idx),
        scores: topResults.map(r => r.score),
        usedIndex: true,
        searchedRows: allVectors.length
    };
}

/**
 * Read all vectors from a column as a flat Float32Array.
 * Used for worker-based parallel search.
 * Handles multi-page columns by fetching and combining all pages.
 * @param {RemoteLanceFile} file - File instance
 * @param {number} colIdx - Vector column index
 * @returns {Promise<Float32Array>} - Flattened vector data [numRows * dim]
 */
export async function readVectorColumn(file, colIdx) {
    const entry = await file.getColumnOffsetEntry(colIdx);
    const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
    const metaInfo = file._parseColumnMeta(new Uint8Array(colMeta));

    if (!metaInfo.pages || metaInfo.pages.length === 0 || metaInfo.rows === 0) {
        return new Float32Array(0);
    }

    // Calculate dimension from first page
    const firstPage = metaInfo.pages[0];
    const dataIdx = firstPage.sizes.length > 1 ? 1 : 0;
    const firstPageSize = firstPage.sizes[dataIdx] || 0;
    const firstPageRows = firstPage.rows || 0;

    if (firstPageRows === 0 || firstPageSize === 0) {
        return new Float32Array(0);
    }

    const dim = Math.floor(firstPageSize / (firstPageRows * 4));
    if (dim === 0) {
        return new Float32Array(0);
    }

    const totalRows = metaInfo.rows;
    const result = new Float32Array(totalRows * dim);

    // Fetch each page in parallel
    const pagePromises = metaInfo.pages.map(async (page, pageIdx) => {
        const pageDataIdx = page.sizes.length > 1 ? 1 : 0;
        const pageOffset = page.offsets[pageDataIdx] || 0;
        const pageSize = page.sizes[pageDataIdx] || 0;

        if (pageSize === 0) return { pageIdx, data: new Float32Array(0), rows: 0 };

        const data = await file.fetchRange(pageOffset, pageOffset + pageSize - 1);
        const floatData = new Float32Array(data);
        return {
            pageIdx,
            data: floatData,
            rows: page.rows
        };
    });

    const pageResults = await Promise.all(pagePromises);

    // Combine pages in order
    let offset = 0;
    for (const pageResult of pageResults.sort((a, b) => a.pageIdx - b.pageIdx)) {
        result.set(pageResult.data, offset);
        offset += pageResult.rows * dim;
    }

    return result;
}

/**
 * Read rows from this Lance file with pagination.
 * @param {RemoteLanceFile} file - File instance
 * @param {Object} options - Query options
 * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
 */
export async function readRows(file, { offset = 0, limit = 50, columns = null } = {}) {
    // Determine column indices to read
    const colIndices = columns || Array.from({ length: file._numColumns }, (_, i) => i);

    // Get total row count from first column
    const totalRows = await file.getRowCount(0);

    // Clamp offset and limit
    const actualOffset = Math.min(offset, totalRows);
    const actualLimit = Math.min(limit, totalRows - actualOffset);

    if (actualLimit <= 0) {
        return {
            columns: colIndices.map(() => []),
            columnNames: file.columnNames.slice(0, colIndices.length),
            total: totalRows
        };
    }

    // Generate indices for the requested rows
    const indices = Array.from({ length: actualLimit }, (_, i) => actualOffset + i);

    // Detect all column types first
    const columnTypes = await file.detectColumnTypes();

    // Read each column in parallel
    const columnPromises = colIndices.map(async (colIdx) => {
        const type = columnTypes[colIdx] || 'unknown';

        try {
            switch (type) {
                case 'string':
                case 'utf8':
                case 'large_utf8':
                    return await file.readStringsAtIndices(colIdx, indices);

                case 'int64':
                    return Array.from(await file.readInt64AtIndices(colIdx, indices));

                case 'int32':
                    return Array.from(await file.readInt32AtIndices(colIdx, indices));

                case 'int16':
                    return Array.from(await file.readInt16AtIndices(colIdx, indices));

                case 'uint8':
                    return Array.from(await file.readUint8AtIndices(colIdx, indices));

                case 'float64':
                case 'double':
                    return Array.from(await file.readFloat64AtIndices(colIdx, indices));

                case 'float32':
                case 'float':
                    return Array.from(await file.readFloat32AtIndices(colIdx, indices));

                case 'bool':
                case 'boolean':
                    return await file.readBoolAtIndices(colIdx, indices);

                case 'fixed_size_list':
                case 'vector':
                    const vectors = await file.readVectorsAtIndices(colIdx, indices);
                    return Array.isArray(vectors) ? vectors : Array.from(vectors);

                default:
                    console.warn(`[LanceQL] Unknown column type: ${type}, trying as string`);
                    return await file.readStringsAtIndices(colIdx, indices);
            }
        } catch (e) {
            console.warn(`[LanceQL] Error reading column ${colIdx} (${type}):`, e.message);
            return indices.map(() => null);
        }
    });

    const columnsData = await Promise.all(columnPromises);

    return {
        columns: columnsData,
        columnNames: colIndices.map(i => file.columnNames[i] || `column_${i}`),
        total: totalRows
    };
}
