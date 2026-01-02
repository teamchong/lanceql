/**
 * SQL Executor - Search functionality (NEAR vector search, BM25, FTS)
 * Extracted from executor.js for modularity
 */

/**
 * Extract NEAR condition from WHERE expression.
 * Returns { column, text, limit } if found, null otherwise.
 */
export function extractNearCondition(expr) {
    if (!expr) return null;

    if (expr.type === 'near') {
        const columnName = expr.column?.name || expr.column;
        const text = expr.text?.value || expr.text;
        return { column: columnName, text, limit: 20 };
    }

    // Check AND/OR for NEAR condition
    if (expr.type === 'binary' && (expr.op === 'AND' || expr.op === 'OR')) {
        const leftNear = extractNearCondition(expr.left);
        if (leftNear) return leftNear;
        return extractNearCondition(expr.right);
    }

    return null;
}

/**
 * Remove NEAR condition from expression, returning remaining conditions.
 */
export function removeNearCondition(expr) {
    if (!expr) return null;

    if (expr.type === 'near') {
        return null;  // Remove the NEAR condition
    }

    if (expr.type === 'binary' && (expr.op === 'AND' || expr.op === 'OR')) {
        const leftIsNear = expr.left?.type === 'near';
        const rightIsNear = expr.right?.type === 'near';

        if (leftIsNear && rightIsNear) return null;
        if (leftIsNear) return removeNearCondition(expr.right);
        if (rightIsNear) return removeNearCondition(expr.left);

        const newLeft = removeNearCondition(expr.left);
        const newRight = removeNearCondition(expr.right);

        if (!newLeft && !newRight) return null;
        if (!newLeft) return newRight;
        if (!newRight) return newLeft;

        return { ...expr, left: newLeft, right: newRight };
    }

    return expr;
}

/**
 * Tokenize text for BM25 search.
 */
export function tokenize(text) {
    if (!text || typeof text !== 'string') return [];

    return text
        .toLowerCase()
        .replace(/[^\w\s]/g, ' ')  // Remove punctuation
        .split(/\s+/)              // Split on whitespace
        .filter(t => t.length > 1) // Remove single chars
        .filter(t => !isStopWord(t)); // Remove stop words
}

/**
 * Check if word is a stop word.
 */
export function isStopWord(word) {
    const stopWords = new Set([
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'we', 'they', 'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how'
    ]);
    return stopWords.has(word);
}

/**
 * Compute BM25 scores for query tokens against indexed documents.
 */
export function computeBM25Scores(queryTokens, index) {
    const k1 = 1.2;
    const b = 0.75;
    const scores = new Map();

    for (const term of queryTokens) {
        const docIds = index.termDocs.get(term);
        if (!docIds) continue;

        // IDF: log((N - n + 0.5) / (n + 0.5) + 1)
        const n = docIds.size;
        const N = index.totalDocs;
        const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1);

        for (const docId of docIds) {
            const tf = index.termFreqs.get(term).get(docId);
            const docLen = index.docLengths.get(docId);
            const avgDL = index.avgDocLength;

            // BM25 term score
            const numerator = tf * (k1 + 1);
            const denominator = tf + k1 * (1 - b + b * docLen / avgDL);
            const termScore = idf * (numerator / denominator);

            scores.set(docId, (scores.get(docId) || 0) + termScore);
        }
    }

    return scores;
}

/**
 * Get top-K documents by score.
 */
export function topKByScore(scores, k) {
    return Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, k)
        .map(([docId]) => docId);
}

/**
 * Build inverted index for a text column.
 * @param {Function} readColumnData - Function to read column data at indices
 * @param {number} colIdx - Column index
 * @param {number} totalRows - Total number of rows
 * @returns {Promise<Object>} - Inverted index
 */
export async function buildFTSIndex(readColumnData, colIdx, totalRows) {
    const index = {
        termDocs: new Map(),    // term -> Set of docIds
        termFreqs: new Map(),   // term -> Map(docId -> freq)
        docLengths: new Map(),  // docId -> word count
        totalDocs: 0,
        avgDocLength: 0,
    };

    // Read all text from column in batches
    const batchSize = 1000;
    let totalLength = 0;

    for (let start = 0; start < totalRows; start += batchSize) {
        const end = Math.min(start + batchSize, totalRows);
        const indices = Array.from({ length: end - start }, (_, i) => start + i);
        const texts = await readColumnData(colIdx, indices);

        for (let i = 0; i < texts.length; i++) {
            const docId = start + i;
            const text = texts[i];
            if (!text || typeof text !== 'string') continue;

            const tokens = tokenize(text);
            index.docLengths.set(docId, tokens.length);
            totalLength += tokens.length;
            index.totalDocs++;

            // Count term frequencies
            const termCounts = new Map();
            for (const token of tokens) {
                termCounts.set(token, (termCounts.get(token) || 0) + 1);
            }

            // Update inverted index
            for (const [term, freq] of termCounts) {
                if (!index.termDocs.has(term)) {
                    index.termDocs.set(term, new Set());
                    index.termFreqs.set(term, new Map());
                }
                index.termDocs.get(term).add(docId);
                index.termFreqs.get(term).set(docId, freq);
            }
        }
    }

    index.avgDocLength = index.totalDocs > 0 ? totalLength / index.totalDocs : 0;
    return index;
}

/**
 * Execute BM25 full-text search when no vector column exists.
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} nearInfo - { column, text, limit }
 * @param {number} totalRows - Total rows to search
 * @returns {Promise<number[]>} - Matching row indices
 */
export async function executeBM25Search(executor, nearInfo, totalRows) {
    const { column, text, limit } = nearInfo;
    const colIdx = executor.columnMap[column.toLowerCase()];

    if (colIdx === undefined) {
        throw new Error(`Column '${column}' not found for text search`);
    }

    // Step 1: Tokenize query
    const queryTokens = tokenize(text);
    if (queryTokens.length === 0) return [];

    // Step 2: Get or build inverted index for this column
    const cacheKey = `fts_${colIdx}`;
    if (!executor._ftsIndexCache) executor._ftsIndexCache = new Map();

    let index = executor._ftsIndexCache.get(cacheKey);
    if (!index) {
        index = await buildFTSIndex(
            (idx, indices) => executor.readColumnData(idx, indices),
            colIdx,
            totalRows
        );
        executor._ftsIndexCache.set(cacheKey, index);
    }

    // Step 3: Compute BM25 scores
    const scores = computeBM25Scores(queryTokens, index);

    // Step 4: Return top-K indices
    return topKByScore(scores, limit);
}

/**
 * Execute NEAR vector search.
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} nearInfo - { column, text, limit }
 * @param {number} totalRows - Total rows to search
 * @returns {Promise<number[]>} - Matching row indices
 */
export async function executeNearSearch(executor, nearInfo, totalRows) {
    // Find vector column for the specified column
    const { column, text, limit } = nearInfo;

    // Look for embedding/vector column
    // Convention: embedding column is named 'embedding' or '<column>_embedding'
    const vectorColName = executor.file.columnNames?.find(n =>
        n === 'embedding' ||
        n === `${column}_embedding` ||
        n.endsWith('_embedding') ||
        n.endsWith('_vector')
    );

    if (!vectorColName) {
        // No vector column found, fall back to BM25 text search
        return await executeBM25Search(executor, nearInfo, totalRows);
    }

    // Use existing vector search infrastructure
    const topK = Math.min(limit, totalRows);

    try {
        // Call the file's vectorSearch method
        const results = await executor.file.vectorSearch(text, topK);
        return results.map(r => r.index);
    } catch (e) {
        console.error('[SQLExecutor] Vector search failed:', e);
        throw new Error(`NEAR search failed: ${e.message}`);
    }
}

/**
 * Evaluate WHERE with NEAR condition.
 * Executes vector search first, then applies remaining conditions.
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} nearInfo - { column, text, limit }
 * @param {Object} whereExpr - WHERE expression AST
 * @param {number} totalRows - Total rows
 * @param {Function} onProgress - Progress callback
 * @returns {Promise<number[]>} - Matching row indices
 */
export async function evaluateWithNear(executor, nearInfo, whereExpr, totalRows, onProgress) {
    if (onProgress) {
        onProgress('Executing vector search...', 0, 100);
    }

    // Execute vector search to get candidate indices
    const searchResults = await executeNearSearch(executor, nearInfo, totalRows);

    if (!searchResults || searchResults.length === 0) {
        return [];
    }

    // Get remaining conditions after removing NEAR
    const remainingExpr = removeNearCondition(whereExpr);

    if (!remainingExpr) {
        // No other conditions, return search results directly
        return searchResults;
    }

    if (onProgress) {
        onProgress('Applying filters...', 50, 100);
    }

    // Apply remaining conditions to search results
    const neededCols = new Set();
    executor.collectColumnsFromExpr(remainingExpr, neededCols);

    // Fetch column data for candidate rows
    const columnData = {};
    for (const colName of neededCols) {
        const colIdx = executor.columnMap[colName.toLowerCase()];
        if (colIdx !== undefined) {
            columnData[colName.toLowerCase()] = await executor.readColumnData(colIdx, searchResults);
        }
    }

    // Filter by remaining conditions
    const matchingIndices = [];
    for (let i = 0; i < searchResults.length; i++) {
        const result = executor.evaluateExpr(remainingExpr, columnData, i);
        if (result) {
            matchingIndices.push(searchResults[i]);
        }
    }

    return matchingIndices;
}

/**
 * Filter indices by WHERE expression
 * @param {Object} executor - SQLExecutor instance
 * @param {number[]} indices - Row indices to filter
 * @param {Object} whereExpr - WHERE expression AST
 * @returns {Promise<number[]>} - Filtered indices
 */
export async function filterIndicesByWhere(executor, indices, whereExpr) {
    const neededCols = new Set();
    executor.collectColumnsFromExpr(whereExpr, neededCols);

    const columnData = {};
    for (const colName of neededCols) {
        const colIdx = executor.columnMap[colName.toLowerCase()];
        if (colIdx !== undefined) {
            columnData[colName.toLowerCase()] = await executor.readColumnData(colIdx, indices);
        }
    }

    return indices.filter((_, i) => executor.evaluateExpr(whereExpr, columnData, i));
}

/**
 * Execute aggregate query with search (NEAR + aggregation)
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} ast - Parsed SQL AST
 * @param {number} totalRows - Total rows
 * @param {Function} onProgress - Progress callback
 * @returns {Promise<Object>} - Query result
 */
export async function executeAggregateWithSearch(executor, ast, totalRows, onProgress) {
    // Extract NEAR condition
    const nearInfo = extractNearCondition(ast.where);
    if (!nearInfo) {
        return await executor.executeAggregateQuery(ast, totalRows, onProgress);
    }

    // Execute search first
    const searchIndices = await executeNearSearch(executor, nearInfo, totalRows);

    if (searchIndices.length === 0) {
        return executor._emptyAggregateResult(ast);
    }

    // Remove NEAR from WHERE for remaining filtering
    const remainingWhere = removeNearCondition(ast.where);
    let filteredIndices = searchIndices;

    if (remainingWhere) {
        filteredIndices = await filterIndicesByWhere(executor, searchIndices, remainingWhere);
    }

    if (filteredIndices.length === 0) {
        return executor._emptyAggregateResult(ast);
    }

    // Execute aggregation on filtered indices
    if (ast.groupBy && ast.groupBy.length > 0) {
        return await executor._executeGroupByOnIndices(ast, filteredIndices, onProgress);
    }

    return await executor._executeSimpleAggregateOnIndices(ast, filteredIndices, onProgress);
}
