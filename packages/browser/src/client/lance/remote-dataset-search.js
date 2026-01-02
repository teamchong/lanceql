import { webgpuAccelerator } from '../gpu/accelerator.js';

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

    const { rowIds, vectors } = partitionData;
    const scores = new Float32Array(vectors.length);
    const dim = queryVec.length;

    if (webgpuAccelerator.isAvailable()) {
        const maxBatch = webgpuAccelerator.getMaxVectorsPerBatch(dim);
        let gpuProcessed = 0;
        let wasmProcessed = 0;

        for (let start = 0; start < vectors.length; start += maxBatch) {
            const end = Math.min(start + maxBatch, vectors.length);
            const chunk = vectors.slice(start, end);

            try {
                const chunkScores = await webgpuAccelerator.batchCosineSimilarity(queryVec, chunk, true);
                if (chunkScores) {
                    scores.set(chunkScores, start);
                    gpuProcessed += chunk.length;
                    continue;
                }
            } catch (e) {
                // Fall through to WASM
            }

            // WASM SIMD fallback
            if (dataset.lanceql?.batchCosineSimilarity) {
                const chunkScores = dataset.lanceql.batchCosineSimilarity(queryVec, chunk, true);
                scores.set(chunkScores, start);
                wasmProcessed += chunk.length;
            } else {
                // JS fallback
                for (let i = 0; i < chunk.length; i++) {
                    const vec = chunk[i];
                    if (!vec || vec.length !== dim) continue;
                    let dot = 0;
                    for (let k = 0; k < dim; k++) {
                        dot += queryVec[k] * vec[k];
                    }
                    scores[start + i] = dot;
                }
                wasmProcessed += chunk.length;
            }
        }

    } else {
        if (dataset.lanceql?.batchCosineSimilarity) {
            const allScores = dataset.lanceql.batchCosineSimilarity(queryVec, vectors, true);
            scores.set(allScores);
        } else {
            for (let i = 0; i < vectors.length; i++) {
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

    // Build and sort results
    const allResults = [];
    for (let i = 0; i < rowIds.length; i++) {
        allResults.push({ index: rowIds[i], score: scores[i] });
    }

    allResults.sort((a, b) => b.score - a.score);
    const finalK = Math.min(topK, allResults.length);

    if (onProgress) onProgress(100, 100);

    return {
        indices: allResults.slice(0, finalK).map(r => r.index),
        scores: allResults.slice(0, finalK).map(r => r.score),
        usedIndex: true,
        searchedRows: rowIds.length
    };
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
