import { getWebGPUAccelerator } from '../webgpu/index.js';

export async function vectorSearch(dataset, colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
    const { normalized = true, workerPool = null, useIndex = true, nprobe = 20 } = options;

    const vectorColIdx = colIdx;

    if (vectorColIdx < 0) {
        throw new Error('No vector column found in dataset');
    }

    const dim = queryVec.length;

    if (!dataset.hasIndex()) {
        throw new Error('No IVF index found. Vector search requires an IVF index for efficient querying.');
    }

    if (dataset._ivfIndex.dimension !== dim) {
        throw new Error(`Query dimension (${dim}) does not match index dimension (${dataset._ivfIndex.dimension}).`);
    }

    if (!dataset._ivfIndex.hasPartitionIndex) {
        throw new Error('IVF partition index (ivf_partitions.bin) not found. Required for efficient search.');
    }

    return await ivfIndexSearch(dataset, queryVec, topK, vectorColIdx, nprobe, onProgress);
}

async function ivfIndexSearch(dataset, queryVec, topK, vectorColIdx, nprobe, onProgress) {
    const partitions = dataset._ivfIndex.findNearestPartitions(queryVec, nprobe);

    const partitionData = await dataset._ivfIndex.fetchPartitionData(
        partitions,
        dataset._ivfIndex.dimension,
        (loaded, total) => {
            if (onProgress) {
                const pct = total > 0 ? loaded / total : 0;
                onProgress(Math.floor(pct * 80), 100);
            }
        }
    );

    if (!partitionData || partitionData.rowIds.length === 0) {
        throw new Error('IVF index not available. This dataset requires ivf_vectors.bin for efficient search.');
    }

    const { rowIds, vectors, preFlattened } = partitionData;
    const dim = queryVec.length;
    const numVectors = preFlattened ? vectors.length / dim : vectors.length;
    const scores = new Float32Array(numVectors);

    const accelerator = getWebGPUAccelerator();
    if (accelerator.isAvailable()) {
        const maxBatch = accelerator.getMaxVectorsPerBatch(dim);

        if (preFlattened) {
            // Process flat vectors in batches
            for (let vecStart = 0; vecStart < numVectors; vecStart += maxBatch) {
                const vecEnd = Math.min(vecStart + maxBatch, numVectors);
                const batchCount = vecEnd - vecStart;
                const chunk = vectors.subarray(vecStart * dim, vecEnd * dim);

                try {
                    const chunkScores = await accelerator.batchCosineSimilarity(queryVec, chunk, true, true);
                    if (chunkScores) {
                        scores.set(chunkScores, vecStart);
                        continue;
                    }
                } catch (e) {
                    // Fall through to WASM
                }

                // WASM SIMD fallback with flat vectors
                if (dataset.lanceql?.batchCosineSimilarityFlat) {
                    const chunkScores = dataset.lanceql.batchCosineSimilarityFlat(queryVec, chunk, dim, true);
                    scores.set(chunkScores, vecStart);
                } else {
                    // JS fallback for flat vectors
                    for (let i = 0; i < batchCount; i++) {
                        const offset = i * dim;
                        let dot = 0;
                        for (let k = 0; k < dim; k++) {
                            dot += queryVec[k] * chunk[offset + k];
                        }
                        scores[vecStart + i] = dot;
                    }
                }
            }
        } else {
            // Legacy path for array-of-arrays
            for (let start = 0; start < numVectors; start += maxBatch) {
                const end = Math.min(start + maxBatch, numVectors);
                const chunk = vectors.slice(start, end);

                try {
                    const chunkScores = await accelerator.batchCosineSimilarity(queryVec, chunk, true, false);
                    if (chunkScores) {
                        scores.set(chunkScores, start);
                        continue;
                    }
                } catch (e) {
                    // Fall through to JS
                }

                for (let i = 0; i < chunk.length; i++) {
                    const vec = chunk[i];
                    if (!vec || vec.length !== dim) continue;
                    let dot = 0;
                    for (let k = 0; k < dim; k++) {
                        dot += queryVec[k] * vec[k];
                    }
                    scores[start + i] = dot;
                }
            }
        }
    } else {
        // Non-GPU path
        if (preFlattened) {
            for (let i = 0; i < numVectors; i++) {
                const offset = i * dim;
                let dot = 0;
                for (let k = 0; k < dim; k++) {
                    dot += queryVec[k] * vectors[offset + k];
                }
                scores[i] = dot;
            }
        } else {
            for (let i = 0; i < numVectors; i++) {
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

    // Build results and use quickselect for top-K (O(n) vs O(n log n) sort)
    const allResults = new Array(rowIds.length);
    for (let i = 0; i < rowIds.length; i++) {
        allResults[i] = { index: rowIds[i], score: scores[i] };
    }

    const finalK = Math.min(topK, allResults.length);
    quickselectTopK(allResults, finalK);

    if (onProgress) onProgress(100, 100);

    return {
        indices: allResults.slice(0, finalK).map(r => r.index),
        scores: allResults.slice(0, finalK).map(r => r.score),
        usedIndex: true,
        searchedRows: rowIds.length
    };
}

/**
 * Quickselect to find top-k elements by score in O(n) average time.
 */
function quickselectTopK(arr, k) {
    if (k >= arr.length || k <= 0) return;

    let left = 0;
    let right = arr.length - 1;

    while (left < right) {
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
}

function swap(arr, i, j) {
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
}

export function findVectorColumn(dataset) {
    if (!dataset._schema) return -1;

    for (let i = 0; i < dataset._schema.length; i++) {
        const field = dataset._schema[i];
        if (field.name === 'embedding' || field.name === 'vector' ||
            field.type === 'fixed_size_list' || field.type === 'list') {
            return i;
        }
    }

    return dataset._schema.length - 1;
}

export async function parallelVectorSearch(dataset, query, topK, vectorColIdx, normalized, workerPool) {
    const dim = query.length;

    const chunkPromises = dataset._fragments.map(async (frag, idx) => {
        const file = await dataset.openFragment(idx);
        const vectors = await file.readVectorColumn(vectorColIdx);
        if (!vectors || vectors.length === 0) {
            return null;
        }

        let startIndex = 0;
        for (let i = 0; i < idx; i++) {
            startIndex += dataset._fragments[i].numRows;
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

    const { indices, scores } = await workerPool.parallelVectorSearch(
        query, chunks, dim, topK, normalized
    );

    const rows = await fetchResultRows(dataset, indices);

    return { indices, scores, rows };
}

async function fetchResultRows(dataset, indices) {
    if (indices.length === 0) return [];

    const rows = [];
    const groups = dataset._groupIndicesByFragment(Array.from(indices));

    for (const [fragIdx, group] of groups) {
        const file = await dataset.openFragment(fragIdx);

        for (const localIdx of group.localIndices) {
            const row = {};

            for (let colIdx = 0; colIdx < dataset._numColumns; colIdx++) {
                const colName = dataset.columnNames[colIdx];
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
