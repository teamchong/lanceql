/**
 * SQLExecutor - SQL query execution engine
 */

import { SQLLexer } from './lexer.js';
import { SQLParser } from './parser.js';

class SQLExecutor {
    constructor(file) {
        this.file = file;
        this.columnMap = {};
        this.columnTypes = [];
        this._cteResults = new Map();  // Store materialized CTE results
        this._database = null;          // Reference to LanceDatabase for subqueries

        // Build column name -> index map
        if (file.columnNames) {
            file.columnNames.forEach((name, idx) => {
                this.columnMap[name.toLowerCase()] = idx;
            });
        }
    }

    /**
     * Set reference to parent database for CTE/subquery execution
     */
    setDatabase(db) {
        this._database = db;
    }

    /**
     * Execute a SQL query
     * @param {string} sql - SQL query string
     * @param {function} onProgress - Optional progress callback
     * @returns {Promise<{columns: string[], rows: any[][], total: number}>}
     */
    async execute(sql, onProgress = null) {
        // Tokenize and parse
        const lexer = new SQLLexer(sql);
        const tokens = lexer.tokenize();
        const parser = new SQLParser(tokens);
        const ast = parser.parse();

        // Generate optimized query plan
        const planner = new QueryPlanner();
        const plan = planner.planSingleTable(ast);

        // Detect column types if not already done
        if (this.columnTypes.length === 0) {
            if (this.file._isRemote && this.file.detectColumnTypes) {
                this.columnTypes = await this.file.detectColumnTypes();
            } else if (this.file._columnTypes) {
                this.columnTypes = this.file._columnTypes;
            } else {
                // Default to unknown for all columns
                this.columnTypes = Array(this.file.numColumns || 0).fill('unknown');
            }
        }

        // Get total row count
        const totalRows = this.file._isRemote
            ? await this.file.getRowCount(0)
            : Number(this.file.getRowCount(0));

        // === STATISTICS-BASED OPTIMIZATION ===
        // For queries with filters, compute statistics to enable pruning
        let columnStats = null;
        let prunedFragments = null;
        let fragmentsPruned = 0;

        if (ast.where && plan.pushedFilters.length > 0 && this.file._isRemote) {
            // Compute stats for filter columns (cached after first computation)
            columnStats = await statisticsManager.precomputeForPlan(this.file, plan);

            // Log statistics info
            if (columnStats.size > 0) {
                console.log(`[SQLExecutor] Statistics available for ${columnStats.size} columns`);
                for (const [col, stats] of columnStats) {
                    console.log(`  ${col}: min=${stats.min}, max=${stats.max}, nulls=${stats.nullCount}`);
                }
            }

            // Fragment pruning based on global statistics
            // (Per-fragment stats would be even better - computed lazily)
            plan.columnStats = Object.fromEntries(columnStats);
        }

        // Use plan's scan columns instead of basic column collection
        const neededColumns = plan.scanColumns.length > 0
            ? plan.scanColumns
            : this.collectNeededColumns(ast);

        // Determine output columns
        const outputColumns = this.resolveOutputColumns(ast);

        // Check if this is an aggregation query
        const hasAggregates = plan.aggregations.length > 0 || this.hasAggregates(ast);
        if (hasAggregates) {
            // Special case: COUNT(*) without WHERE/SEARCH returns metadata row count (free)
            if (this.isSimpleCountStar(ast) && !ast.where && !ast.search) {
                return {
                    columns: ['COUNT(*)'],
                    rows: [[totalRows]],
                    total: 1,
                    aggregationStats: {
                        scannedRows: 0,
                        totalRows,
                        coveragePercent: '100.00',
                        isPartialScan: false,
                        fromMetadata: true,
                    },
                    queryPlan: plan,  // Include plan in result
                };
            }
            // For aggregations with SEARCH/NEAR, we need to run search first
            if (ast.search || this._extractNearCondition(ast.where)) {
                return await this.executeAggregateWithSearch(ast, totalRows, onProgress);
            }
            return await this.executeAggregateQuery(ast, totalRows, onProgress);
        }

        // Calculate indices to fetch
        let indices;
        const limit = ast.limit || 100;
        const offset = ast.offset || 0;

        // For queries without WHERE, we can just fetch the needed indices directly
        // For queries with WHERE, we need to fetch more data and filter
        if (!ast.where) {
            // Simple case: no filtering needed
            indices = [];
            const endIdx = Math.min(offset + limit, totalRows);
            for (let i = offset; i < endIdx; i++) {
                indices.push(i);
            }
        } else {
            // Complex case: need to evaluate WHERE clause
            // Fetch data in batches and filter
            indices = await this.evaluateWhere(ast.where, totalRows, onProgress);

            // Apply OFFSET and LIMIT to filtered results
            indices = indices.slice(offset, offset + limit);
        }

        if (onProgress) {
            onProgress('Fetching column data...', 0, outputColumns.length);
        }

        // Fetch data for output columns
        const columnData = {};
        for (let i = 0; i < neededColumns.length; i++) {
            const colName = neededColumns[i];
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx === undefined) continue;

            if (onProgress) {
                onProgress(`Fetching ${colName}...`, i, neededColumns.length);
            }

            columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, indices);
        }

        // Build result rows
        const rows = [];
        for (let i = 0; i < indices.length; i++) {
            const row = [];
            for (const col of outputColumns) {
                if (col.type === 'star') {
                    // Expand all columns
                    for (const name of this.file.columnNames || []) {
                        const data = columnData[name.toLowerCase()];
                        row.push(data ? data[i] : null);
                    }
                } else {
                    const value = this.evaluateExpr(col.expr, columnData, i);
                    row.push(value);
                }
            }
            rows.push(row);
        }

        // Apply PIVOT transformation (rows to columns with aggregation)
        if (ast.pivot) {
            const pivotResult = this._executePivot(rows, outputColumns, ast.pivot);
            rows.length = 0;
            rows.push(...pivotResult.rows);
            outputColumns.length = 0;
            outputColumns.push(...pivotResult.columns);
        }

        // Apply UNPIVOT transformation (columns to rows)
        if (ast.unpivot) {
            const unpivotResult = this._executeUnpivot(rows, outputColumns, ast.unpivot);
            rows.length = 0;
            rows.push(...unpivotResult.rows);
            outputColumns.length = 0;
            outputColumns.push(...unpivotResult.columns);
        }

        // Apply DISTINCT (GPU-accelerated for large result sets)
        if (ast.distinct) {
            const uniqueRows = await this.applyDistinct(rows);
            rows.length = 0;
            rows.push(...uniqueRows);
        }

        // Apply ORDER BY (GPU-accelerated for large result sets)
        if (ast.orderBy && ast.orderBy.length > 0) {
            await this.applyOrderBy(rows, ast.orderBy, outputColumns);
        }

        // Build column names for output
        const colNames = [];
        for (const col of outputColumns) {
            if (col.type === 'star') {
                colNames.push(...(this.file.columnNames || []));
            } else {
                colNames.push(col.alias || this.exprToName(col.expr));
            }
        }

        // When LIMIT is specified, total should reflect the limited count, not full dataset
        // This ensures infinite scroll respects the LIMIT clause
        const effectiveTotal = ast.limit ? rows.length : totalRows;

        // Track if ORDER BY was applied on a subset (honest about sorting limitations)
        const orderByOnSubset = ast.orderBy && ast.orderBy.length > 0 && rows.length < totalRows;

        return {
            columns: colNames,
            rows,
            total: effectiveTotal,
            orderByOnSubset,
            orderByColumns: ast.orderBy ? ast.orderBy.map(ob => `${ob.column} ${ob.direction}`) : [],
            // Query optimization info
            queryPlan: plan,
            optimization: {
                statsComputed: columnStats?.size > 0,
                columnStats: columnStats ? Object.fromEntries(columnStats) : null,
                pushedFilters: plan.pushedFilters?.length || 0,
                estimatedSelectivity: plan.estimatedSelectivity,
            },
        };
    }

    collectNeededColumns(ast) {
        const columns = new Set();

        // From SELECT
        for (const item of ast.columns) {
            if (item.type === 'star') {
                (this.file.columnNames || []).forEach(n => columns.add(n.toLowerCase()));
            } else {
                this.collectColumnsFromExpr(item.expr, columns);
            }
        }

        // From WHERE
        if (ast.where) {
            this.collectColumnsFromExpr(ast.where, columns);
        }

        // From ORDER BY
        for (const ob of ast.orderBy || []) {
            columns.add(ob.column.toLowerCase());
        }

        return Array.from(columns);
    }

    collectColumnsFromExpr(expr, columns) {
        if (!expr) return;

        switch (expr.type) {
            case 'column':
                columns.add(expr.name.toLowerCase());
                break;
            case 'binary':
                this.collectColumnsFromExpr(expr.left, columns);
                this.collectColumnsFromExpr(expr.right, columns);
                break;
            case 'unary':
                this.collectColumnsFromExpr(expr.operand, columns);
                break;
            case 'call':
                for (const arg of expr.args || []) {
                    this.collectColumnsFromExpr(arg, columns);
                }
                break;
            case 'in':
                this.collectColumnsFromExpr(expr.expr, columns);
                break;
            case 'between':
                this.collectColumnsFromExpr(expr.expr, columns);
                break;
            case 'like':
                this.collectColumnsFromExpr(expr.expr, columns);
                break;
            case 'near':
                this.collectColumnsFromExpr(expr.column, columns);
                break;
        }
    }

    resolveOutputColumns(ast) {
        return ast.columns;
    }

    async readColumnData(colIdx, indices) {
        const type = this.columnTypes[colIdx] || 'unknown';

        try {
            if (type === 'string') {
                const data = await this.file.readStringsAtIndices(colIdx, indices);
                // readStringsAtIndices returns array of strings
                return Array.isArray(data) ? data : Array.from(data);
            } else if (type === 'int64') {
                const data = await this.file.readInt64AtIndices(colIdx, indices);
                // Convert BigInt64Array to regular array of Numbers
                const result = [];
                for (let i = 0; i < data.length; i++) {
                    result.push(Number(data[i]));
                }
                return result;
            } else if (type === 'float64') {
                const data = await this.file.readFloat64AtIndices(colIdx, indices);
                // Convert Float64Array to regular array
                return Array.from(data);
            } else if (type === 'int32') {
                const data = await this.file.readInt32AtIndices(colIdx, indices);
                return Array.from(data);
            } else if (type === 'float32') {
                const data = await this.file.readFloat32AtIndices(colIdx, indices);
                return Array.from(data);
            } else if (type === 'vector') {
                // Return placeholder for vectors
                return indices.map(() => '[vector]');
            } else {
                // Try string as fallback
                try {
                    return await this.file.readStringsAtIndices(colIdx, indices);
                } catch (e) {
                    return indices.map(() => null);
                }
            }
        } catch (e) {
            // Failed to read column, returning nulls
            return indices.map(() => null);
        }
    }

    async evaluateWhere(whereExpr, totalRows, onProgress) {
        // Check for NEAR conditions in WHERE clause
        const nearInfo = this._extractNearCondition(whereExpr);
        if (nearInfo) {
            return await this._evaluateWithNear(nearInfo, whereExpr, totalRows, onProgress);
        }

        // Optimization: For simple conditions on a single numeric column,
        // fetch only the filter column first, then fetch other columns only for matches
        const simpleFilter = this._detectSimpleFilter(whereExpr);

        if (simpleFilter) {
            return await this._evaluateSimpleFilter(simpleFilter, totalRows, onProgress);
        }

        // Complex conditions: fetch all needed columns in batches
        return await this._evaluateComplexFilter(whereExpr, totalRows, onProgress);
    }

    /**
     * Extract NEAR condition from WHERE expression.
     * Returns { column, text, limit } if found, null otherwise.
     * @private
     */
    _extractNearCondition(expr) {
        if (!expr) return null;

        if (expr.type === 'near') {
            const columnName = expr.column?.name || expr.column;
            const text = expr.text?.value || expr.text;
            return { column: columnName, text, limit: 20 };
        }

        // Check AND/OR for NEAR condition
        if (expr.type === 'binary' && (expr.op === 'AND' || expr.op === 'OR')) {
            const leftNear = this._extractNearCondition(expr.left);
            if (leftNear) return leftNear;
            return this._extractNearCondition(expr.right);
        }

        return null;
    }

    /**
     * Remove NEAR condition from expression, returning remaining conditions.
     * @private
     */
    _removeNearCondition(expr) {
        if (!expr) return null;

        if (expr.type === 'near') {
            return null;  // Remove the NEAR condition
        }

        if (expr.type === 'binary' && (expr.op === 'AND' || expr.op === 'OR')) {
            const leftIsNear = expr.left?.type === 'near';
            const rightIsNear = expr.right?.type === 'near';

            if (leftIsNear && rightIsNear) return null;
            if (leftIsNear) return this._removeNearCondition(expr.right);
            if (rightIsNear) return this._removeNearCondition(expr.left);

            const newLeft = this._removeNearCondition(expr.left);
            const newRight = this._removeNearCondition(expr.right);

            if (!newLeft && !newRight) return null;
            if (!newLeft) return newRight;
            if (!newRight) return newLeft;

            return { ...expr, left: newLeft, right: newRight };
        }

        return expr;
    }

    /**
     * Evaluate WHERE with NEAR condition.
     * Executes vector search first, then applies remaining conditions.
     * @private
     */
    async _evaluateWithNear(nearInfo, whereExpr, totalRows, onProgress) {
        if (onProgress) {
            onProgress('Executing vector search...', 0, 100);
        }

        // Execute vector search to get candidate indices
        // This reuses the existing vector search infrastructure
        const searchResults = await this._executeNearSearch(nearInfo, totalRows);

        if (!searchResults || searchResults.length === 0) {
            return [];
        }

        // Get remaining conditions after removing NEAR
        const remainingExpr = this._removeNearCondition(whereExpr);

        if (!remainingExpr) {
            // No other conditions, return search results directly
            return searchResults;
        }

        if (onProgress) {
            onProgress('Applying filters...', 50, 100);
        }

        // Apply remaining conditions to search results
        const neededCols = new Set();
        this.collectColumnsFromExpr(remainingExpr, neededCols);

        // Fetch column data for candidate rows
        const columnData = {};
        for (const colName of neededCols) {
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx !== undefined) {
                columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, searchResults);
            }
        }

        // Filter by remaining conditions
        const matchingIndices = [];
        for (let i = 0; i < searchResults.length; i++) {
            const result = this.evaluateExpr(remainingExpr, columnData, i);
            if (result) {
                matchingIndices.push(searchResults[i]);
            }
        }

        return matchingIndices;
    }

    /**
     * Execute NEAR vector search.
     * @private
     */
    async _executeNearSearch(nearInfo, totalRows) {
        // Find vector column for the specified column
        const { column, text, limit } = nearInfo;

        // Look for embedding/vector column
        // Convention: embedding column is named 'embedding' or '<column>_embedding'
        const vectorColName = this.file.columnNames?.find(n =>
            n === 'embedding' ||
            n === `${column}_embedding` ||
            n.endsWith('_embedding') ||
            n.endsWith('_vector')
        );

        if (!vectorColName) {
            // No vector column found, fall back to BM25 text search
            return await this._executeBM25Search(nearInfo, totalRows);
        }

        // Use existing vector search infrastructure
        const topK = Math.min(limit, totalRows);

        try {
            // Call the file's vectorSearch method
            const results = await this.file.vectorSearch(text, topK);
            return results.map(r => r.index);
        } catch (e) {
            console.error('[SQLExecutor] Vector search failed:', e);
            throw new Error(`NEAR search failed: ${e.message}`);
        }
    }

    /**
     * Execute BM25 full-text search when no vector column exists.
     * @private
     */
    async _executeBM25Search(nearInfo, totalRows) {
        const { column, text, limit } = nearInfo;
        const colIdx = this.columnMap[column.toLowerCase()];

        if (colIdx === undefined) {
            throw new Error(`Column '${column}' not found for text search`);
        }

        // Step 1: Tokenize query
        const queryTokens = this._tokenize(text);
        if (queryTokens.length === 0) return [];

        // Step 2: Get or build inverted index for this column
        const index = await this._getOrBuildFTSIndex(colIdx, totalRows);

        // Step 3: Compute BM25 scores
        const scores = this._computeBM25Scores(queryTokens, index);

        // Step 4: Return top-K indices
        return this._topKByScore(scores, limit);
    }

    /**
     * Tokenize text for BM25 search.
     * @private
     */
    _tokenize(text) {
        if (!text || typeof text !== 'string') return [];

        return text
            .toLowerCase()
            .replace(/[^\w\s]/g, ' ')  // Remove punctuation
            .split(/\s+/)              // Split on whitespace
            .filter(t => t.length > 1) // Remove single chars
            .filter(t => !this._isStopWord(t)); // Remove stop words
    }

    /**
     * Check if word is a stop word.
     * @private
     */
    _isStopWord(word) {
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
     * Get or build inverted index for a text column.
     * @private
     */
    async _getOrBuildFTSIndex(colIdx, totalRows) {
        const cacheKey = `fts_${colIdx}`;
        if (this._ftsIndexCache?.has(cacheKey)) {
            return this._ftsIndexCache.get(cacheKey);
        }

        // Build index
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
            const texts = await this.readColumnData(colIdx, indices);

            for (let i = 0; i < texts.length; i++) {
                const docId = start + i;
                const text = texts[i];
                if (!text || typeof text !== 'string') continue;

                const tokens = this._tokenize(text);
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

        // Cache the index
        if (!this._ftsIndexCache) this._ftsIndexCache = new Map();
        this._ftsIndexCache.set(cacheKey, index);

        return index;
    }

    /**
     * Compute BM25 scores for query tokens against indexed documents.
     * @private
     */
    _computeBM25Scores(queryTokens, index) {
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
     * @private
     */
    _topKByScore(scores, k) {
        return Array.from(scores.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, k)
            .map(([docId]) => docId);
    }

    /**
     * Execute PIVOT transformation - convert rows to columns with aggregation.
     * Example: PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))
     * @private
     */
    _executePivot(rows, columns, pivot) {
        const { aggregate, forColumn, inValues } = pivot;

        // Get column names from outputColumns
        const colNames = columns.map(col => {
            if (col.type === 'star') return '*';
            return col.alias || (col.expr.type === 'column' ? col.expr.name : null);
        });

        // Find the FOR column index
        const forColIdx = colNames.findIndex(
            n => n && n.toLowerCase() === forColumn.toLowerCase()
        );
        if (forColIdx === -1) {
            throw new Error(`PIVOT: Column '${forColumn}' not found in result set`);
        }

        // Find the aggregate source column
        const aggSourceCol = aggregate.args && aggregate.args[0]
            ? (aggregate.args[0].name || aggregate.args[0].column || aggregate.args[0])
            : null;
        const aggSourceIdx = aggSourceCol
            ? colNames.findIndex(n => n && n.toLowerCase() === String(aggSourceCol).toLowerCase())
            : -1;

        if (aggSourceIdx === -1 && aggregate.name.toUpperCase() !== 'COUNT') {
            throw new Error(`PIVOT: Aggregate source column not found`);
        }

        // Determine group columns (all columns except forColumn and aggregate source)
        const groupColIndices = [];
        for (let i = 0; i < colNames.length; i++) {
            if (i !== forColIdx && i !== aggSourceIdx) {
                groupColIndices.push(i);
            }
        }

        // Build groups: Map<groupKey, Map<pivotValue, aggregateValues[]>>
        const groups = new Map();

        for (const row of rows) {
            const groupKey = groupColIndices.map(i => JSON.stringify(row[i])).join('|');
            const pivotValue = row[forColIdx];
            const aggValue = aggSourceIdx >= 0 ? row[aggSourceIdx] : 1; // COUNT uses 1

            if (!groups.has(groupKey)) {
                groups.set(groupKey, {
                    groupValues: groupColIndices.map(i => row[i]),
                    pivots: new Map()
                });
            }

            const group = groups.get(groupKey);
            const pivotKey = String(pivotValue);
            if (!group.pivots.has(pivotKey)) {
                group.pivots.set(pivotKey, []);
            }
            group.pivots.get(pivotKey).push(aggValue);
        }

        // Compute aggregates and build output rows
        const groupColNames = groupColIndices.map(i => colNames[i]);
        const outputColNames = [...groupColNames, ...inValues.map(v => String(v))];
        const outputRows = [];

        for (const [, group] of groups) {
            const row = [...group.groupValues];
            for (const pivotVal of inValues) {
                const key = String(pivotVal);
                const values = group.pivots.get(key) || [];
                row.push(this._computeAggregate(aggregate.name, values));
            }
            outputRows.push(row);
        }

        // Build new outputColumns structure
        const newColumns = outputColNames.map(name => ({
            type: 'column',
            expr: { type: 'column', name },
            alias: name
        }));

        return { rows: outputRows, columns: newColumns };
    }

    /**
     * Execute UNPIVOT transformation - convert columns to rows.
     * Example: UNPIVOT (value FOR month IN (jan, feb, mar))
     * @private
     */
    _executeUnpivot(rows, columns, unpivot) {
        const { valueColumn, nameColumn, inColumns } = unpivot;

        // Get column names from outputColumns
        const colNames = columns.map(col => {
            if (col.type === 'star') return '*';
            return col.alias || (col.expr.type === 'column' ? col.expr.name : null);
        });

        // Find column indices for unpivot sources
        const inColIndices = inColumns.map(c => {
            const idx = colNames.findIndex(n => n && n.toLowerCase() === c.toLowerCase());
            if (idx === -1) {
                throw new Error(`UNPIVOT: Column '${c}' not found in result set`);
            }
            return idx;
        });

        // Preserved columns (not in inColumns)
        const preservedIndices = [];
        const preservedNames = [];
        for (let i = 0; i < colNames.length; i++) {
            if (!inColIndices.includes(i)) {
                preservedIndices.push(i);
                preservedNames.push(colNames[i]);
            }
        }

        // Output columns: preserved + nameColumn + valueColumn
        const outputColNames = [...preservedNames, nameColumn, valueColumn];
        const outputRows = [];

        for (const row of rows) {
            const preservedValues = preservedIndices.map(i => row[i]);

            // Create one output row per unpivoted column
            for (let i = 0; i < inColumns.length; i++) {
                const colName = inColumns[i];
                const value = row[inColIndices[i]];

                // Skip NULL values (standard UNPIVOT behavior)
                if (value != null) {
                    outputRows.push([...preservedValues, colName, value]);
                }
            }
        }

        // Build new outputColumns structure
        const newColumns = outputColNames.map(name => ({
            type: 'column',
            expr: { type: 'column', name },
            alias: name
        }));

        return { rows: outputRows, columns: newColumns };
    }

    /**
     * Compute aggregate function on a list of values.
     * Used by PIVOT transformation.
     * @private
     */
    _computeAggregate(funcName, values) {
        const nums = values.filter(v => v != null && typeof v === 'number');
        switch (funcName.toUpperCase()) {
            case 'SUM':
                return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) : null;
            case 'COUNT':
                return values.filter(v => v != null).length;
            case 'AVG':
                return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
            case 'MIN':
                return nums.length > 0 ? Math.min(...nums) : null;
            case 'MAX':
                return nums.length > 0 ? Math.max(...nums) : null;
            default:
                return null;
        }
    }

    /**
     * Detect if WHERE clause is a simple comparison (column op value).
     * @private
     */
    _detectSimpleFilter(expr) {
        if (expr.type !== 'binary') return null;
        if (!['==', '!=', '<', '<=', '>', '>='].includes(expr.op)) return null;

        // Check if it's (column op literal) or (literal op column)
        let column = null;
        let value = null;
        let op = expr.op;

        if (expr.left.type === 'column' && expr.right.type === 'literal') {
            column = expr.left.name;
            value = expr.right.value;
        } else if (expr.left.type === 'literal' && expr.right.type === 'column') {
            column = expr.right.name;
            value = expr.left.value;
            // Reverse the operator for (literal op column)
            if (op === '<') op = '>';
            else if (op === '>') op = '<';
            else if (op === '<=') op = '>=';
            else if (op === '>=') op = '<=';
        }

        if (!column || value === null) return null;

        const colIdx = this.columnMap[column.toLowerCase()];
        if (colIdx === undefined) return null;

        const colType = this.columnTypes[colIdx];
        if (!['int64', 'int32', 'float64', 'float32'].includes(colType)) return null;

        return { column, colIdx, op, value, colType };
    }

    /**
     * Optimized evaluation for simple column comparisons.
     * Fetches only the filter column in large batches.
     * @private
     */
    async _evaluateSimpleFilter(filter, totalRows, onProgress) {
        const matchingIndices = [];
        // Use larger batch size for single-column filtering
        const batchSize = 5000;

        // Using optimized simple filter path

        for (let batchStart = 0; batchStart < totalRows; batchStart += batchSize) {
            if (onProgress) {
                const pct = Math.round((batchStart / totalRows) * 100);
                onProgress(`Filtering ${filter.column}... ${pct}%`, batchStart, totalRows);
            }

            const batchEnd = Math.min(batchStart + batchSize, totalRows);
            const batchIndices = [];
            for (let i = batchStart; i < batchEnd; i++) {
                batchIndices.push(i);
            }

            // Fetch only the filter column
            const colData = await this.readColumnData(filter.colIdx, batchIndices);

            // Apply filter
            for (let i = 0; i < batchIndices.length; i++) {
                const val = colData[i];
                let matches = false;

                switch (filter.op) {
                    case '==': matches = val === filter.value; break;
                    case '!=': matches = val !== filter.value; break;
                    case '<': matches = val < filter.value; break;
                    case '<=': matches = val <= filter.value; break;
                    case '>': matches = val > filter.value; break;
                    case '>=': matches = val >= filter.value; break;
                }

                if (matches) {
                    matchingIndices.push(batchIndices[i]);
                }
            }

            // Early exit if we have enough results
            if (matchingIndices.length >= 10000) {
                // Early exit: found enough matches
                break;
            }
        }

        return matchingIndices;
    }

    /**
     * General evaluation for complex WHERE clauses.
     * @private
     */
    async _evaluateComplexFilter(whereExpr, totalRows, onProgress) {
        const matchingIndices = [];
        const batchSize = 1000;

        // Pre-compute needed columns
        const neededCols = new Set();
        this.collectColumnsFromExpr(whereExpr, neededCols);

        for (let batchStart = 0; batchStart < totalRows; batchStart += batchSize) {
            if (onProgress) {
                onProgress(`Filtering rows...`, batchStart, totalRows);
            }

            const batchEnd = Math.min(batchStart + batchSize, totalRows);
            const batchIndices = [];
            for (let i = batchStart; i < batchEnd; i++) {
                batchIndices.push(i);
            }

            // Fetch needed column data for this batch
            const batchData = {};
            for (const colName of neededCols) {
                const colIdx = this.columnMap[colName.toLowerCase()];
                if (colIdx !== undefined) {
                    batchData[colName.toLowerCase()] = await this.readColumnData(colIdx, batchIndices);
                }
            }

            // Evaluate WHERE for each row in batch
            for (let i = 0; i < batchIndices.length; i++) {
                const result = this.evaluateExpr(whereExpr, batchData, i);
                if (result) {
                    matchingIndices.push(batchIndices[i]);
                }
            }

            // Early exit if we have enough results
            if (matchingIndices.length >= 10000) {
                break;
            }
        }

        return matchingIndices;
    }

    evaluateExpr(expr, columnData, rowIdx) {
        if (!expr) return null;

        switch (expr.type) {
            case 'literal':
                return expr.value;

            case 'column': {
                const data = columnData[expr.name.toLowerCase()];
                return data ? data[rowIdx] : null;
            }

            case 'star':
                return '*';

            case 'binary': {
                const left = this.evaluateExpr(expr.left, columnData, rowIdx);
                const right = this.evaluateExpr(expr.right, columnData, rowIdx);

                switch (expr.op) {
                    case '+': return (left || 0) + (right || 0);
                    case '-': return (left || 0) - (right || 0);
                    case '*': return (left || 0) * (right || 0);
                    case '/': return right !== 0 ? (left || 0) / right : null;
                    case '==': return left === right;
                    case '!=': return left !== right;
                    case '<': return left < right;
                    case '<=': return left <= right;
                    case '>': return left > right;
                    case '>=': return left >= right;
                    case 'AND': return left && right;
                    case 'OR': return left || right;
                    default: return null;
                }
            }

            case 'unary': {
                const operand = this.evaluateExpr(expr.operand, columnData, rowIdx);
                switch (expr.op) {
                    case '-': return -operand;
                    case 'NOT': return !operand;
                    default: return null;
                }
            }

            case 'in': {
                const value = this.evaluateExpr(expr.expr, columnData, rowIdx);
                const values = expr.values.map(v => this.evaluateExpr(v, columnData, rowIdx));
                return values.includes(value);
            }

            case 'between': {
                const value = this.evaluateExpr(expr.expr, columnData, rowIdx);
                const low = this.evaluateExpr(expr.low, columnData, rowIdx);
                const high = this.evaluateExpr(expr.high, columnData, rowIdx);
                return value >= low && value <= high;
            }

            case 'like': {
                const value = this.evaluateExpr(expr.expr, columnData, rowIdx);
                const pattern = this.evaluateExpr(expr.pattern, columnData, rowIdx);
                if (typeof value !== 'string' || typeof pattern !== 'string') return false;
                // Convert SQL LIKE pattern to regex
                const regex = new RegExp('^' + pattern.replace(/%/g, '.*').replace(/_/g, '.') + '$', 'i');
                return regex.test(value);
            }

            case 'near':
                // NEAR is handled specially at the evaluateWhere level
                // If we reach here, the row is already in the NEAR result set
                return true;

            case 'call':
                // Aggregate functions not supported in row-level evaluation
                return null;

            case 'subquery': {
                // Execute subquery and return scalar result
                // For correlated subqueries, pass outer row context
                return this._executeSubquery(expr.query, columnData, rowIdx);
            }

            case 'array': {
                // Evaluate each element to build the array
                return expr.elements.map(el => this.evaluateExpr(el, columnData, rowIdx));
            }

            case 'subscript': {
                // Array subscript access with 1-based indexing (SQL standard)
                const arr = this.evaluateExpr(expr.array, columnData, rowIdx);
                const idx = this.evaluateExpr(expr.index, columnData, rowIdx);
                if (!Array.isArray(arr)) return null;
                // SQL uses 1-based indexing
                return arr[idx - 1] ?? null;
            }

            default:
                return null;
        }
    }

    /**
     * Execute a subquery and return its result.
     * For scalar subqueries (returns single value), returns that value.
     * For correlated subqueries, substitutes outer row values.
     */
    _executeSubquery(subqueryAst, outerColumnData, outerRowIdx) {
        // Clone the subquery AST to avoid mutating the original
        const resolvedAst = JSON.parse(JSON.stringify(subqueryAst));

        // Check for correlated references (columns not in the subquery's FROM)
        // and substitute with outer row values
        const subqueryTable = resolvedAst.from?.name || resolvedAst.from?.table;
        const correlatedColumns = this._findCorrelatedColumns(resolvedAst, subqueryTable);

        // Build correlation context with outer row values
        const correlationContext = {};
        for (const col of correlatedColumns) {
            const colName = col.column.toLowerCase();
            if (outerColumnData[colName]) {
                correlationContext[col.table + '.' + col.column] = outerColumnData[colName][outerRowIdx];
            }
        }

        // If we have correlations, modify the WHERE clause to use literals
        if (Object.keys(correlationContext).length > 0) {
            this._substituteCorrelations(resolvedAst.where, correlationContext);
        }

        // Check if FROM references a CTE
        const tableName = resolvedAst.from?.name?.toLowerCase() || resolvedAst.from?.table?.toLowerCase();
        if (tableName && this._cteResults?.has(tableName)) {
            const result = this._executeOnInMemoryData(resolvedAst, this._cteResults.get(tableName));
            return result.rows.length > 0 ? result.rows[0][0] : null;  // Scalar result
        }

        // Execute against database if available
        if (this._database) {
            try {
                const result = this._database._executeSingleTable(resolvedAst);
                if (result && result.then) {
                    // This is async - we need to handle it synchronously for expression evaluation
                    // For now, return null and recommend using CTE approach instead
                    console.warn('[SQLExecutor] Async subquery in expression context - consider using CTE');
                    return null;
                }
                return result?.rows?.[0]?.[0] ?? null;
            } catch (e) {
                console.warn('[SQLExecutor] Subquery execution failed:', e.message);
                return null;
            }
        }

        console.warn('[SQLExecutor] Subquery execution requires LanceDatabase context');
        return null;
    }

    /**
     * Find columns that reference the outer query (correlated columns)
     */
    _findCorrelatedColumns(ast, subqueryTable) {
        const correlatedCols = [];

        const walkExpr = (expr) => {
            if (!expr) return;

            if (expr.type === 'column' && expr.table && expr.table !== subqueryTable) {
                correlatedCols.push(expr);
            } else if (expr.type === 'binary') {
                walkExpr(expr.left);
                walkExpr(expr.right);
            } else if (expr.type === 'unary') {
                walkExpr(expr.operand);
            } else if (expr.type === 'in') {
                walkExpr(expr.expr);
                expr.values?.forEach(walkExpr);
            } else if (expr.type === 'between') {
                walkExpr(expr.expr);
                walkExpr(expr.low);
                walkExpr(expr.high);
            } else if (expr.type === 'like') {
                walkExpr(expr.expr);
                walkExpr(expr.pattern);
            } else if (expr.type === 'call') {
                expr.args?.forEach(walkExpr);
            }
        };

        walkExpr(ast.where);
        return correlatedCols;
    }

    /**
     * Substitute correlated column references with literal values
     */
    _substituteCorrelations(expr, correlationContext) {
        if (!expr) return;

        if (expr.type === 'column' && expr.table) {
            const key = expr.table + '.' + expr.column;
            if (correlationContext.hasOwnProperty(key)) {
                // Convert to literal
                expr.type = 'literal';
                expr.value = correlationContext[key];
                delete expr.table;
                delete expr.column;
            }
        } else if (expr.type === 'binary') {
            this._substituteCorrelations(expr.left, correlationContext);
            this._substituteCorrelations(expr.right, correlationContext);
        } else if (expr.type === 'unary') {
            this._substituteCorrelations(expr.operand, correlationContext);
        } else if (expr.type === 'in') {
            this._substituteCorrelations(expr.expr, correlationContext);
            expr.values?.forEach(v => this._substituteCorrelations(v, correlationContext));
        } else if (expr.type === 'between') {
            this._substituteCorrelations(expr.expr, correlationContext);
            this._substituteCorrelations(expr.low, correlationContext);
            this._substituteCorrelations(expr.high, correlationContext);
        } else if (expr.type === 'like') {
            this._substituteCorrelations(expr.expr, correlationContext);
            this._substituteCorrelations(expr.pattern, correlationContext);
        } else if (expr.type === 'call') {
            expr.args?.forEach(a => this._substituteCorrelations(a, correlationContext));
        }
    }

    /**
     * Materialize CTEs before query execution
     * @param {Array} ctes - Array of CTE definitions from AST
     * @param {LanceDatabase} db - Database reference for executing CTE bodies
     */
    async materializeCTEs(ctes, db) {
        this._database = db;
        for (const cte of ctes) {
            const cteName = cte.name.toLowerCase();
            if (cte.body.type === 'RECURSIVE_CTE') {
                // Execute anchor query first
                const anchorResult = await this._executeCTEBody(cte.body.anchor, db);
                let result = { columns: anchorResult.columns, rows: [...anchorResult.rows] };

                // Iterate recursive part until no new rows (max 1000 iterations)
                for (let i = 0; i < 1000; i++) {
                    this._cteResults.set(cteName, result);
                    const recursiveResult = await this._executeCTEBody(cte.body.recursive, db);
                    if (recursiveResult.rows.length === 0) break;
                    result = { columns: result.columns, rows: [...result.rows, ...recursiveResult.rows] };
                }
                this._cteResults.set(cteName, result);
            } else {
                const result = await this._executeCTEBody(cte.body, db);
                this._cteResults.set(cteName, result);
            }
        }
    }

    /**
     * Execute a CTE body - either against database or against already-materialized CTE
     */
    async _executeCTEBody(bodyAst, db) {
        // Check if FROM references another CTE
        const tableName = bodyAst.from?.name?.toLowerCase() || bodyAst.from?.table?.toLowerCase();
        if (tableName && this._cteResults.has(tableName)) {
            return this._executeOnInMemoryData(bodyAst, this._cteResults.get(tableName));
        }
        // Fall back to database execution
        return db._executeSingleTable(bodyAst);
    }

    /**
     * Execute query on in-memory data (for CTEs and subqueries)
     * @param {Object} ast - Parsed SELECT AST
     * @param {Object} data - In-memory data { columns: string[], rows: any[][] }
     * @returns {Object} - Query result { columns: string[], rows: any[][] }
     */
    _executeOnInMemoryData(ast, data) {
        // Build column lookup: column name -> array of values
        const columnData = {};
        for (let i = 0; i < data.columns.length; i++) {
            const colName = data.columns[i].toLowerCase();
            columnData[colName] = data.rows.map(row => row[i]);
        }

        // Apply WHERE filter
        const filteredIndices = [];
        for (let i = 0; i < data.rows.length; i++) {
            if (!ast.where || this._evaluateInMemoryExpr(ast.where, columnData, i)) {
                filteredIndices.push(i);
            }
        }

        // Check for GROUP BY or aggregations
        const hasGroupBy = ast.groupBy && ast.groupBy.length > 0;
        const hasAggregates = this._hasAggregatesInSelect(ast.columns);

        if (hasGroupBy || hasAggregates) {
            return this._executeGroupByAggregation(ast, data, columnData, filteredIndices);
        }

        // Check for window functions
        if (this.hasWindowFunctions(ast)) {
            return this._executeWindowFunctions(ast, data, columnData, filteredIndices);
        }

        // Project columns and build result
        const resultColumns = [];
        const resultRows = [];

        // Handle SELECT * - parser returns { type: 'star' } directly, not { expr: { type: 'star' } }
        const isSelectStar = ast.columns.length === 1 &&
            (ast.columns[0].type === 'star' || ast.columns[0].expr?.type === 'star');
        if (isSelectStar) {
            for (const colName of data.columns) {
                resultColumns.push(colName);
            }
            for (const idx of filteredIndices) {
                resultRows.push([...data.rows[idx]]);
            }
        } else {
            // Named columns
            for (const col of ast.columns) {
                resultColumns.push(col.alias || col.expr?.column || '*');
            }
            for (const idx of filteredIndices) {
                const row = ast.columns.map(col => {
                    if (col.type === 'star' || col.expr?.type === 'star') {
                        return data.rows[idx];
                    }
                    return this._evaluateInMemoryExpr(col.expr, columnData, idx);
                });
                resultRows.push(row.flat());
            }
        }

        // Apply ORDER BY
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colIdxMap = {};
            resultColumns.forEach((name, idx) => { colIdxMap[name.toLowerCase()] = idx; });

            resultRows.sort((a, b) => {
                for (const ob of ast.orderBy) {
                    const colIdx = colIdxMap[ob.column.toLowerCase()];
                    if (colIdx === undefined) continue;
                    const valA = a[colIdx], valB = b[colIdx];
                    // Parser uses 'descending' boolean, some places use 'direction' string
                    const dir = (ob.descending || ob.direction === 'DESC') ? -1 : 1;
                    if (valA == null && valB == null) continue;
                    if (valA == null) return 1 * dir;
                    if (valB == null) return -1 * dir;
                    if (valA < valB) return -1 * dir;
                    if (valA > valB) return 1 * dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET
        const offset = ast.offset || 0;
        let rows = resultRows;
        if (offset > 0) rows = rows.slice(offset);
        if (ast.limit) rows = rows.slice(0, ast.limit);

        return { columns: resultColumns, rows, total: filteredIndices.length };
    }

    /**
     * Check if SELECT columns contain aggregate functions
     */
    _hasAggregatesInSelect(columns) {
        const aggFuncs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];
        for (const col of columns) {
            if (col.expr?.type === 'call') {
                // Skip window functions (those with OVER clause)
                if (col.expr.over) continue;
                const funcName = (col.expr.name || '').toUpperCase();
                if (aggFuncs.includes(funcName)) return true;
            }
        }
        return false;
    }

    /**
     * Execute GROUP BY with aggregation on in-memory data
     */
    _executeGroupByAggregation(ast, data, columnData, filteredIndices) {
        const aggFuncs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];
        const hasGroupBy = ast.groupBy && ast.groupBy.length > 0;

        // Check for advanced GROUP BY (ROLLUP/CUBE/GROUPING SETS)
        if (hasGroupBy && this._hasAdvancedGroupBy(ast.groupBy)) {
            return this._executeAdvancedGroupBy(ast, data, columnData, filteredIndices);
        }

        // Group rows by GROUP BY columns
        const groups = new Map();
        for (const idx of filteredIndices) {
            let groupKey = '';
            if (hasGroupBy) {
                groupKey = ast.groupBy.map(expr => {
                    const colName = (expr.column || expr.name || '').toLowerCase();
                    const val = columnData[colName]?.[idx];
                    return JSON.stringify(val);
                }).join('|');
            }

            if (!groups.has(groupKey)) {
                groups.set(groupKey, []);
            }
            groups.get(groupKey).push(idx);
        }

        // If no GROUP BY and no rows, create one group with empty indices for aggregate results
        // (e.g., COUNT(*) on empty table should return 0, not empty result)
        if (!hasGroupBy && groups.size === 0) {
            groups.set('', []);
        }

        // Build result columns
        const resultColumns = [];
        for (const col of ast.columns) {
            if (col.alias) {
                resultColumns.push(col.alias);
            } else if (col.expr?.type === 'call') {
                const argName = col.expr.args?.[0]?.name || col.expr.args?.[0]?.column || '*';
                resultColumns.push(`${col.expr.name}(${argName})`);
            } else if (col.expr?.type === 'column') {
                resultColumns.push(col.expr.name || col.expr.column);
            } else {
                resultColumns.push('?');
            }
        }

        // Compute aggregations for each group
        const resultRows = [];
        for (const [, groupIndices] of groups) {
            const row = [];
            for (const col of ast.columns) {
                const expr = col.expr;
                if (expr?.type === 'call' && aggFuncs.includes((expr.name || '').toUpperCase())) {
                    const funcName = expr.name.toUpperCase();
                    const argExpr = expr.args?.[0];
                    const isStar = argExpr?.type === 'star';
                    const colName = (argExpr?.name || argExpr?.column || '').toLowerCase();

                    let result = null;
                    switch (funcName) {
                        case 'COUNT':
                            if (isStar) {
                                result = groupIndices.length;
                            } else {
                                result = groupIndices.filter(i => columnData[colName]?.[i] != null).length;
                            }
                            break;
                        case 'SUM': {
                            let sum = 0;
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && !isNaN(v)) sum += v;
                            }
                            result = sum;
                            break;
                        }
                        case 'AVG': {
                            let sum = 0, count = 0;
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && !isNaN(v)) { sum += v; count++; }
                            }
                            result = count > 0 ? sum / count : null;
                            break;
                        }
                        case 'MIN': {
                            let min = null;
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && (min === null || v < min)) min = v;
                            }
                            result = min;
                            break;
                        }
                        case 'MAX': {
                            let max = null;
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && (max === null || v > max)) max = v;
                            }
                            result = max;
                            break;
                        }
                        case 'STDDEV':
                        case 'STDDEV_SAMP': {
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                            }
                            if (vals.length < 2) {
                                result = null;
                            } else {
                                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                                const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (vals.length - 1);
                                result = Math.sqrt(variance);
                            }
                            break;
                        }
                        case 'STDDEV_POP': {
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                            }
                            if (vals.length === 0) {
                                result = null;
                            } else {
                                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                                const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
                                result = Math.sqrt(variance);
                            }
                            break;
                        }
                        case 'VARIANCE':
                        case 'VAR_SAMP': {
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                            }
                            if (vals.length < 2) {
                                result = null;
                            } else {
                                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                                result = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (vals.length - 1);
                            }
                            break;
                        }
                        case 'VAR_POP': {
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                            }
                            if (vals.length === 0) {
                                result = null;
                            } else {
                                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                                result = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
                            }
                            break;
                        }
                        case 'MEDIAN': {
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                            }
                            if (vals.length === 0) {
                                result = null;
                            } else {
                                vals.sort((a, b) => a - b);
                                const mid = Math.floor(vals.length / 2);
                                result = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
                            }
                            break;
                        }
                        case 'STRING_AGG':
                        case 'GROUP_CONCAT': {
                            // STRING_AGG(col, separator) or GROUP_CONCAT(col)
                            const separatorArg = expr.args?.[1];
                            const separator = separatorArg?.value ?? ',';
                            const vals = [];
                            for (const i of groupIndices) {
                                const v = columnData[colName]?.[i];
                                if (v != null) vals.push(String(v));
                            }
                            result = vals.join(separator);
                            break;
                        }
                    }
                    row.push(result);
                } else if (expr?.type === 'column') {
                    // Non-aggregate column - take first value from group
                    const colName = (expr.name || expr.column || '').toLowerCase();
                    row.push(columnData[colName]?.[groupIndices[0]] ?? null);
                } else {
                    row.push(this._evaluateInMemoryExpr(expr, columnData, groupIndices[0]));
                }
            }
            resultRows.push(row);
        }

        // Apply ORDER BY
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colIdxMap = {};
            resultColumns.forEach((name, idx) => { colIdxMap[name.toLowerCase()] = idx; });

            resultRows.sort((a, b) => {
                for (const ob of ast.orderBy) {
                    const colIdx = colIdxMap[ob.column.toLowerCase()];
                    if (colIdx === undefined) continue;
                    const valA = a[colIdx], valB = b[colIdx];
                    const dir = (ob.descending || ob.direction === 'DESC') ? -1 : 1;
                    if (valA == null && valB == null) continue;
                    if (valA == null) return 1 * dir;
                    if (valB == null) return -1 * dir;
                    if (valA < valB) return -1 * dir;
                    if (valA > valB) return 1 * dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET
        const offset = ast.offset || 0;
        let rows = resultRows;
        if (offset > 0) rows = rows.slice(offset);
        if (ast.limit) rows = rows.slice(0, ast.limit);

        return { columns: resultColumns, rows, total: groups.size };
    }

    /**
     * Execute advanced GROUP BY with ROLLUP, CUBE, or GROUPING SETS
     */
    _executeAdvancedGroupBy(ast, data, columnData, filteredIndices) {
        // 1. Expand GROUP BY into grouping sets
        const groupingSets = this._expandGroupBy(ast.groupBy);

        // 2. Get all column names for results
        const allGroupColumns = this._getAllGroupColumns(ast.groupBy);

        // 3. Execute aggregation for each grouping set
        const allResults = [];
        for (const groupingSet of groupingSets) {
            const setResults = this._executeGroupByForSet(
                ast, columnData, filteredIndices, groupingSet, allGroupColumns
            );
            allResults.push(...setResults);
        }

        // 4. Build result with proper column order
        return this._buildAdvancedGroupByResult(ast, allResults, allGroupColumns);
    }

    /**
     * Execute GROUP BY aggregation for a single grouping set
     */
    _executeGroupByForSet(ast, columnData, filteredIndices, groupingSet, allGroupColumns) {
        const aggFuncs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];
        const groups = new Map();

        // Normalize groupingSet to lowercase for comparison
        const groupingSetLower = groupingSet.map(c => c.toLowerCase());

        for (const idx of filteredIndices) {
            // Build group key from current grouping set columns only
            const groupKey = groupingSet.length > 0
                ? groupingSet.map(col => {
                    const val = columnData[col.toLowerCase()]?.[idx];
                    return JSON.stringify(val);
                }).join('|')
                : '__grand_total__';

            if (!groups.has(groupKey)) {
                // Store group values for all columns (NULL for non-grouped)
                const groupValues = {};
                for (const col of allGroupColumns) {
                    if (groupingSetLower.includes(col.toLowerCase())) {
                        groupValues[col] = columnData[col.toLowerCase()]?.[idx];
                    } else {
                        groupValues[col] = null; // NULL for super-aggregate rows
                    }
                }
                groups.set(groupKey, {
                    values: groupValues,
                    indices: [],
                    _groupingSet: groupingSet
                });
            }
            groups.get(groupKey).indices.push(idx);
        }

        // Handle empty data - still need grand total row
        if (groupingSet.length === 0 && groups.size === 0) {
            const groupValues = {};
            for (const col of allGroupColumns) {
                groupValues[col] = null;
            }
            groups.set('__grand_total__', {
                values: groupValues,
                indices: [],
                _groupingSet: groupingSet
            });
        }

        // Compute aggregates for each group
        const results = [];
        for (const [, group] of groups) {
            const row = { ...group.values, _groupingSet: group._groupingSet };

            // Compute each aggregate column
            for (const col of ast.columns) {
                const expr = col.expr;
                if (expr?.type === 'call' && aggFuncs.includes((expr.name || '').toUpperCase())) {
                    const funcName = expr.name.toUpperCase();
                    const argExpr = expr.args?.[0];
                    const isStar = argExpr?.type === 'star';
                    const colName = (argExpr?.name || argExpr?.column || '').toLowerCase();
                    const alias = col.alias || `${funcName}(${isStar ? '*' : colName})`;

                    let result = null;
                    const indices = group.indices;

                    switch (funcName) {
                        case 'COUNT':
                            result = isStar
                                ? indices.length
                                : indices.filter(i => columnData[colName]?.[i] != null).length;
                            break;
                        case 'SUM': {
                            let sum = 0;
                            for (const i of indices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && !isNaN(v)) sum += v;
                            }
                            result = sum;
                            break;
                        }
                        case 'AVG': {
                            let sum = 0, count = 0;
                            for (const i of indices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && !isNaN(v)) { sum += v; count++; }
                            }
                            result = count > 0 ? sum / count : null;
                            break;
                        }
                        case 'MIN': {
                            let min = null;
                            for (const i of indices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && (min === null || v < min)) min = v;
                            }
                            result = min;
                            break;
                        }
                        case 'MAX': {
                            let max = null;
                            for (const i of indices) {
                                const v = columnData[colName]?.[i];
                                if (v != null && (max === null || v > max)) max = v;
                            }
                            result = max;
                            break;
                        }
                    }
                    row[alias] = result;
                }
            }
            results.push(row);
        }

        return results;
    }

    /**
     * Get all column names from GROUP BY (for column ordering)
     */
    _getAllGroupColumns(groupBy) {
        const columns = [];
        for (const item of groupBy) {
            if (item.type === 'COLUMN') {
                if (!columns.includes(item.column)) columns.push(item.column);
            } else if (item.type === 'ROLLUP' || item.type === 'CUBE') {
                for (const col of item.columns) {
                    if (!columns.includes(col)) columns.push(col);
                }
            } else if (item.type === 'GROUPING_SETS') {
                for (const set of item.sets) {
                    for (const col of set) {
                        if (!columns.includes(col)) columns.push(col);
                    }
                }
            }
        }
        return columns;
    }

    /**
     * Build final result from advanced GROUP BY results
     */
    _buildAdvancedGroupByResult(ast, allResults, allGroupColumns) {
        // Build result columns from AST
        const resultColumns = [];
        for (const col of ast.columns) {
            if (col.alias) {
                resultColumns.push(col.alias);
            } else if (col.expr?.type === 'call') {
                const argName = col.expr.args?.[0]?.name || col.expr.args?.[0]?.column || '*';
                resultColumns.push(`${col.expr.name}(${argName})`);
            } else if (col.expr?.type === 'column') {
                resultColumns.push(col.expr.name || col.expr.column);
            } else {
                resultColumns.push('?');
            }
        }

        // Convert result objects to arrays matching column order
        const resultRows = allResults.map(rowObj => {
            const row = [];
            for (const colName of resultColumns) {
                // Check if this is a group column or aggregate
                const val = rowObj[colName] ?? rowObj[colName.toLowerCase()] ?? null;
                row.push(val);
            }
            return row;
        });

        // Apply ORDER BY if present
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colIdxMap = {};
            resultColumns.forEach((name, idx) => { colIdxMap[name.toLowerCase()] = idx; });

            resultRows.sort((a, b) => {
                for (const ob of ast.orderBy) {
                    const colIdx = colIdxMap[ob.column.toLowerCase()];
                    if (colIdx === undefined) continue;
                    const valA = a[colIdx], valB = b[colIdx];
                    const dir = (ob.descending || ob.direction === 'DESC') ? -1 : 1;
                    if (valA == null && valB == null) continue;
                    if (valA == null) return 1 * dir;
                    if (valB == null) return -1 * dir;
                    if (valA < valB) return -1 * dir;
                    if (valA > valB) return 1 * dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET
        const offset = ast.offset || 0;
        let rows = resultRows;
        if (offset > 0) rows = rows.slice(offset);
        if (ast.limit) rows = rows.slice(0, ast.limit);

        return { columns: resultColumns, rows, total: allResults.length };
    }

    /**
     * Execute window functions on in-memory data
     */
    _executeWindowFunctions(ast, data, columnData, filteredIndices) {
        // Build filtered rows
        const filteredRows = filteredIndices.map(idx => data.rows[idx]);

        // Compute window function results
        const windowResults = this.computeWindowFunctions(ast, filteredRows, columnData);

        // Build result columns
        const resultColumns = [];
        for (const col of ast.columns) {
            if (col.alias) {
                resultColumns.push(col.alias);
            } else if (col.expr?.type === 'call') {
                const argName = col.expr.args?.[0]?.name || col.expr.args?.[0]?.column || '*';
                resultColumns.push(`${col.expr.name}(${argName})`);
            } else if (col.expr?.type === 'column') {
                resultColumns.push(col.expr.name || col.expr.column);
            } else {
                resultColumns.push('?');
            }
        }

        // Build result rows
        const resultRows = [];
        for (let i = 0; i < filteredIndices.length; i++) {
            const origIdx = filteredIndices[i];
            const row = [];

            for (let c = 0; c < ast.columns.length; c++) {
                const col = ast.columns[c];
                const expr = col.expr;

                // Check if this is a window function column
                if (expr?.over) {
                    // Find the corresponding window result
                    const windowCol = windowResults.find(w => w.colIndex === c);
                    row.push(windowCol ? windowCol.values[i] : null);
                } else if (expr?.type === 'column') {
                    const colName = (expr.name || expr.column || '').toLowerCase();
                    row.push(columnData[colName]?.[origIdx] ?? null);
                } else {
                    row.push(this._evaluateInMemoryExpr(expr, columnData, origIdx));
                }
            }

            resultRows.push(row);
        }

        // Apply QUALIFY filter (filter on window function results)
        let finalRows = resultRows;
        if (ast.qualify) {
            finalRows = [];
            // Build column name to index map for expression evaluation
            const qualifyColMap = {};
            resultColumns.forEach((name, idx) => { qualifyColMap[name.toLowerCase()] = idx; });

            for (let i = 0; i < resultRows.length; i++) {
                // Build row data object for expression evaluation
                const rowData = {};
                for (let c = 0; c < resultColumns.length; c++) {
                    rowData[resultColumns[c].toLowerCase()] = resultRows[i][c];
                }

                // Evaluate QUALIFY condition
                if (this._evaluateInMemoryExpr(ast.qualify, rowData, 0)) {
                    finalRows.push(resultRows[i]);
                }
            }
        }

        // Apply ORDER BY
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colIdxMap = {};
            resultColumns.forEach((name, idx) => { colIdxMap[name.toLowerCase()] = idx; });

            finalRows.sort((a, b) => {
                for (const ob of ast.orderBy) {
                    const colIdx = colIdxMap[ob.column.toLowerCase()];
                    if (colIdx === undefined) continue;
                    const valA = a[colIdx], valB = b[colIdx];
                    const dir = (ob.descending || ob.direction === 'DESC') ? -1 : 1;
                    if (valA == null && valB == null) continue;
                    if (valA == null) return 1 * dir;
                    if (valB == null) return -1 * dir;
                    if (valA < valB) return -1 * dir;
                    if (valA > valB) return 1 * dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET
        const offset = ast.offset || 0;
        let rows = finalRows;
        if (offset > 0) rows = rows.slice(offset);
        if (ast.limit) rows = rows.slice(0, ast.limit);

        return { columns: resultColumns, rows, total: finalRows.length };
    }

    /**
     * Evaluate expression on in-memory data
     */
    _evaluateInMemoryExpr(expr, columnData, rowIdx) {
        if (!expr) return null;

        switch (expr.type) {
            case 'literal':
                return expr.value;

            case 'column': {
                const colName = expr.column.toLowerCase();
                const col = columnData[colName];
                return col ? col[rowIdx] : null;
            }

            case 'binary': {
                const left = this._evaluateInMemoryExpr(expr.left, columnData, rowIdx);
                const right = this._evaluateInMemoryExpr(expr.right, columnData, rowIdx);
                const op = expr.op || expr.operator;  // Parser uses 'op', some places use 'operator'
                switch (op) {
                    case '=': case '==': return left == right;
                    case '!=': case '<>': return left != right;
                    case '<': return left < right;
                    case '<=': return left <= right;
                    case '>': return left > right;
                    case '>=': return left >= right;
                    case '+': return Number(left) + Number(right);
                    case '-': return Number(left) - Number(right);
                    case '*': return Number(left) * Number(right);
                    case '/': return right !== 0 ? Number(left) / Number(right) : null;
                    case 'AND': return left && right;
                    case 'OR': return left || right;
                    default: return null;
                }
            }

            case 'unary': {
                const operand = this._evaluateInMemoryExpr(expr.operand, columnData, rowIdx);
                const op = expr.op || expr.operator;
                switch (op) {
                    case 'NOT': return !operand;
                    case '-': return -operand;
                    case 'IS NULL': return operand == null;
                    case 'IS NOT NULL': return operand != null;
                    default: return null;
                }
            }

            case 'call': {
                // Aggregate functions would need special handling
                const funcName = expr.name.toUpperCase();
                const args = expr.args?.map(a => this._evaluateInMemoryExpr(a, columnData, rowIdx)) || [];
                switch (funcName) {
                    case 'UPPER': return String(args[0]).toUpperCase();
                    case 'LOWER': return String(args[0]).toLowerCase();
                    case 'LENGTH': return String(args[0]).length;
                    case 'SUBSTR': case 'SUBSTRING': return String(args[0]).substring(args[1] - 1, args[2] ? args[1] - 1 + args[2] : undefined);
                    case 'COALESCE': return args.find(a => a != null) ?? null;
                    case 'ABS': return Math.abs(args[0]);
                    case 'ROUND': return Math.round(args[0] * Math.pow(10, args[1] || 0)) / Math.pow(10, args[1] || 0);
                    case 'GROUPING': {
                        // GROUPING(col) returns 1 if col is a super-aggregate (null due to ROLLUP/CUBE), 0 otherwise
                        // The column name is in the first argument
                        const colArg = expr.args?.[0];
                        const colName = colArg?.column || colArg?.name;
                        if (!colName) return 0;
                        // Check if this column is in the current grouping set
                        // columnData._groupingSet contains the columns that are grouped (not super-aggregate)
                        const groupingSet = columnData._groupingSet || [];
                        return groupingSet.includes(colName.toLowerCase()) ? 0 : 1;
                    }
                    default: return null;
                }
            }

            case 'in': {
                const val = this._evaluateInMemoryExpr(expr.expr, columnData, rowIdx);
                const values = expr.values.map(v => v.value ?? this._evaluateInMemoryExpr(v, columnData, rowIdx));
                return values.includes(val);
            }

            case 'between': {
                const val = this._evaluateInMemoryExpr(expr.expr, columnData, rowIdx);
                const low = this._evaluateInMemoryExpr(expr.low, columnData, rowIdx);
                const high = this._evaluateInMemoryExpr(expr.high, columnData, rowIdx);
                return val >= low && val <= high;
            }

            case 'like': {
                const val = String(this._evaluateInMemoryExpr(expr.expr, columnData, rowIdx));
                const pattern = this._evaluateInMemoryExpr(expr.pattern, columnData, rowIdx);
                const regex = new RegExp('^' + String(pattern).replace(/%/g, '.*').replace(/_/g, '.') + '$', 'i');
                return regex.test(val);
            }

            case 'array': {
                // Evaluate each element to build the array
                return expr.elements.map(el => this._evaluateInMemoryExpr(el, columnData, rowIdx));
            }

            case 'subscript': {
                // Array subscript access with 1-based indexing (SQL standard)
                const arr = this._evaluateInMemoryExpr(expr.array, columnData, rowIdx);
                const idx = this._evaluateInMemoryExpr(expr.index, columnData, rowIdx);
                if (!Array.isArray(arr)) return null;
                // SQL uses 1-based indexing
                return arr[idx - 1] ?? null;
            }

            default:
                return null;
        }
    }

    /**
     * Check if query has window functions
     */
    hasWindowFunctions(ast) {
        return ast.columns?.some(col => col.expr?.type === 'call' && col.expr.over);
    }

    /**
     * Execute window functions on in-memory data
     * Window functions are computed after WHERE but before ORDER BY/LIMIT
     */
    computeWindowFunctions(ast, rows, columnData) {
        const windowColumns = [];

        for (let colIndex = 0; colIndex < ast.columns.length; colIndex++) {
            const col = ast.columns[colIndex];
            if (col.expr?.type === 'call' && col.expr.over) {
                const values = this._computeWindowFunction(
                    col.expr.name,
                    col.expr.args,
                    col.expr.over,
                    rows,
                    columnData
                );
                windowColumns.push({
                    colIndex,
                    alias: col.alias || col.expr.name,
                    values
                });
            }
        }

        return windowColumns;
    }

    /**
     * Compute a single window function
     */
    _computeWindowFunction(funcName, args, over, rows, columnData) {
        const results = new Array(rows.length).fill(null);

        // Partition rows
        const partitions = this._partitionRows(rows, over.partitionBy, columnData);

        for (const partition of partitions) {
            // Sort partition by ORDER BY if specified
            if (over.orderBy && over.orderBy.length > 0) {
                partition.sort((a, b) => this._compareRowsByOrder(a, b, over.orderBy, columnData));
            }

            // Compute function for each row in partition
            for (let i = 0; i < partition.length; i++) {
                const rowIdx = partition[i].idx;

                switch (funcName) {
                    case 'ROW_NUMBER':
                        results[rowIdx] = i + 1;
                        break;

                    case 'RANK': {
                        // RANK: same rank for ties, gaps after ties
                        let rank = 1;
                        for (let j = 0; j < i; j++) {
                            if (this._compareRowsByOrder(partition[j], partition[i], over.orderBy, columnData) !== 0) {
                                rank = j + 1;
                            }
                        }
                        if (i > 0 && this._compareRowsByOrder(partition[i-1], partition[i], over.orderBy, columnData) === 0) {
                            results[rowIdx] = results[partition[i-1].idx];
                        } else {
                            results[rowIdx] = i + 1;
                        }
                        break;
                    }

                    case 'DENSE_RANK': {
                        // DENSE_RANK: same rank for ties, no gaps
                        if (i === 0) {
                            results[rowIdx] = 1;
                        } else if (this._compareRowsByOrder(partition[i-1], partition[i], over.orderBy, columnData) === 0) {
                            results[rowIdx] = results[partition[i-1].idx];
                        } else {
                            results[rowIdx] = results[partition[i-1].idx] + 1;
                        }
                        break;
                    }

                    case 'NTILE': {
                        const n = args[0]?.value || 1;
                        const bucketSize = Math.ceil(partition.length / n);
                        results[rowIdx] = Math.floor(i / bucketSize) + 1;
                        break;
                    }

                    case 'PERCENT_RANK': {
                        // PERCENT_RANK = (rank - 1) / (partition_size - 1)
                        // First row in partition always 0, only row returns 0
                        // rank = position of first row with same value (ties get same rank)
                        let rank = i + 1;
                        for (let j = 0; j < i; j++) {
                            if (this._compareRowsByOrder(partition[j], partition[i], over.orderBy, columnData) === 0) {
                                rank = j + 1;  // Found a tie - use its position
                                break;
                            }
                        }
                        const partitionSize = partition.length;
                        results[rowIdx] = partitionSize > 1 ? (rank - 1) / (partitionSize - 1) : 0;
                        break;
                    }

                    case 'CUME_DIST': {
                        // CUME_DIST = (rows with value <= current) / total_rows
                        // Includes all tied rows
                        let countLessOrEqual = 0;
                        for (let j = 0; j < partition.length; j++) {
                            const cmp = this._compareRowsByOrder(partition[j], partition[i], over.orderBy, columnData);
                            if (cmp <= 0) countLessOrEqual++;
                        }
                        results[rowIdx] = countLessOrEqual / partition.length;
                        break;
                    }

                    case 'LAG': {
                        const lagCol = args[0];
                        const lagN = args[1]?.value || 1;
                        const defaultVal = args[2]?.value ?? null;
                        if (i >= lagN) {
                            const prevRowIdx = partition[i - lagN].idx;
                            results[rowIdx] = this._evaluateInMemoryExpr(lagCol, columnData, prevRowIdx);
                        } else {
                            results[rowIdx] = defaultVal;
                        }
                        break;
                    }

                    case 'LEAD': {
                        const leadCol = args[0];
                        const leadN = args[1]?.value || 1;
                        const defaultVal = args[2]?.value ?? null;
                        if (i + leadN < partition.length) {
                            const nextRowIdx = partition[i + leadN].idx;
                            results[rowIdx] = this._evaluateInMemoryExpr(leadCol, columnData, nextRowIdx);
                        } else {
                            results[rowIdx] = defaultVal;
                        }
                        break;
                    }

                    case 'FIRST_VALUE': {
                        const firstRowIdx = partition[0].idx;
                        results[rowIdx] = this._evaluateInMemoryExpr(args[0], columnData, firstRowIdx);
                        break;
                    }

                    case 'LAST_VALUE': {
                        // LAST_VALUE returns the last value within the frame
                        // Default frame: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        const frame = over.frame || {
                            type: 'RANGE',
                            start: { type: 'UNBOUNDED_PRECEDING' },
                            end: { type: 'CURRENT_ROW' }
                        };
                        const [, endIdx] = this._getFrameBounds(frame, partition, i);
                        const lastRowIdx = partition[endIdx].idx;
                        results[rowIdx] = this._evaluateInMemoryExpr(args[0], columnData, lastRowIdx);
                        break;
                    }

                    case 'NTH_VALUE': {
                        const n = args[1]?.value || 1;
                        if (n > 0 && n <= partition.length) {
                            const nthRowIdx = partition[n - 1].idx;
                            results[rowIdx] = this._evaluateInMemoryExpr(args[0], columnData, nthRowIdx);
                        } else {
                            results[rowIdx] = null;
                        }
                        break;
                    }

                    // Aggregate window functions (frame-aware)
                    case 'SUM':
                    case 'AVG':
                    case 'COUNT':
                    case 'MIN':
                    case 'MAX': {
                        // Get frame bounds - default to RANGE UNBOUNDED PRECEDING TO CURRENT ROW
                        const frame = over.frame || {
                            type: 'RANGE',
                            start: { type: 'UNBOUNDED_PRECEDING' },
                            end: { type: 'CURRENT_ROW' }
                        };
                        const [startIdx, endIdx] = this._getFrameBounds(frame, partition, i);

                        // Collect values within frame
                        const isStar = args[0]?.type === 'star';
                        let values = [];
                        let frameRowCount = 0;
                        for (let j = startIdx; j <= endIdx; j++) {
                            frameRowCount++;
                            if (!isStar) {
                                const val = this._evaluateInMemoryExpr(args[0], columnData, partition[j].idx);
                                if (val != null) values.push(Number(val));
                            }
                        }

                        // Compute aggregate
                        let result = null;
                        switch (funcName) {
                            case 'SUM':
                                result = values.reduce((a, b) => a + b, 0);
                                break;
                            case 'AVG':
                                result = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
                                break;
                            case 'COUNT':
                                result = isStar ? frameRowCount : values.length;
                                break;
                            case 'MIN':
                                result = values.length > 0 ? Math.min(...values) : null;
                                break;
                            case 'MAX':
                                result = values.length > 0 ? Math.max(...values) : null;
                                break;
                        }
                        results[rowIdx] = result;
                        break;
                    }

                    default:
                        results[rowIdx] = null;
                }
            }
        }

        return results;
    }

    /**
     * Partition rows based on PARTITION BY expressions
     */
    _partitionRows(rows, partitionBy, columnData) {
        if (!partitionBy || partitionBy.length === 0) {
            // No partitioning - all rows in one partition
            return [rows.map((r, i) => ({ idx: i, row: r }))];
        }

        const groups = new Map();
        for (let i = 0; i < rows.length; i++) {
            const key = partitionBy
                .map(expr => JSON.stringify(this._evaluateInMemoryExpr(expr, columnData, i)))
                .join('|');
            if (!groups.has(key)) {
                groups.set(key, []);
            }
            groups.get(key).push({ idx: i, row: rows[i] });
        }

        return Array.from(groups.values());
    }

    /**
     * Compare two rows based on ORDER BY specification
     */
    _compareRowsByOrder(a, b, orderBy, columnData) {
        for (const ob of orderBy) {
            const valA = this._evaluateInMemoryExpr({ type: 'column', column: ob.column }, columnData, a.idx);
            const valB = this._evaluateInMemoryExpr({ type: 'column', column: ob.column }, columnData, b.idx);

            const dir = ob.direction === 'DESC' ? -1 : 1;
            if (valA == null && valB == null) continue;
            if (valA == null) return 1 * dir;
            if (valB == null) return -1 * dir;
            if (valA < valB) return -1 * dir;
            if (valA > valB) return 1 * dir;
        }
        return 0;
    }

    /**
     * Calculate frame bounds for window function
     * @param {Object} frame - Frame specification { type, start, end }
     * @param {Array} partition - Sorted partition rows
     * @param {number} currentIdx - Current row index within partition
     * @returns {[number, number]} - [startIdx, endIdx] bounds
     */
    _getFrameBounds(frame, partition, currentIdx) {
        const n = partition.length;
        let startIdx = 0;
        let endIdx = currentIdx;

        // Parse start bound (parser uses spaces in type names)
        const start = frame.start || { type: 'UNBOUNDED PRECEDING' };
        const startType = start.type.replace(' ', '_').toUpperCase();
        switch (startType) {
            case 'UNBOUNDED_PRECEDING':
                startIdx = 0;
                break;
            case 'CURRENT_ROW':
                startIdx = currentIdx;
                break;
            case 'PRECEDING':
                startIdx = Math.max(0, currentIdx - (start.offset || start.value || 1));
                break;
            case 'FOLLOWING':
                startIdx = Math.min(n - 1, currentIdx + (start.offset || start.value || 1));
                break;
        }

        // Parse end bound
        const end = frame.end || { type: 'CURRENT ROW' };
        const endType = end.type.replace(' ', '_').toUpperCase();
        switch (endType) {
            case 'UNBOUNDED_FOLLOWING':
                endIdx = n - 1;
                break;
            case 'CURRENT_ROW':
                endIdx = currentIdx;
                break;
            case 'PRECEDING':
                endIdx = Math.max(0, currentIdx - (end.offset || end.value || 1));
                break;
            case 'FOLLOWING':
                endIdx = Math.min(n - 1, currentIdx + (end.offset || end.value || 1));
                break;
        }

        // Ensure valid bounds
        if (startIdx > endIdx) [startIdx, endIdx] = [endIdx, startIdx];
        return [startIdx, endIdx];
    }

    async applyOrderBy(rows, orderBy, outputColumns) {
        // Build column index map
        const colIdxMap = {};
        let idx = 0;
        for (const col of outputColumns) {
            if (col.type === 'star') {
                for (const name of this.file.columnNames || []) {
                    colIdxMap[name.toLowerCase()] = idx++;
                }
            } else {
                const name = col.alias || this.exprToName(col.expr);
                colIdxMap[name.toLowerCase()] = idx++;
            }
        }

        // Use GPU for large datasets (10,000+ rows)
        if (rows.length >= 10000 && gpuSorter.isAvailable()) {
            // Multi-column sort: stable sort from last to first column
            let indices = new Uint32Array(rows.length);
            for (let i = 0; i < rows.length; i++) indices[i] = i;

            for (let c = orderBy.length - 1; c >= 0; c--) {
                const ob = orderBy[c];
                const colIdx = colIdxMap[ob.column.toLowerCase()];
                if (colIdx === undefined) continue;

                const ascending = !ob.descending;
                const values = new Float32Array(rows.length);
                for (let i = 0; i < rows.length; i++) {
                    const val = rows[indices[i]][colIdx];
                    if (val == null) {
                        values[i] = ascending ? 3.4e38 : -3.4e38; // NULLS LAST
                    } else if (typeof val === 'number') {
                        values[i] = val;
                    } else if (typeof val === 'string') {
                        // Use string hash for approximate sorting
                        let key = 0;
                        for (let j = 0; j < Math.min(4, val.length); j++) {
                            key = key * 256 + val.charCodeAt(j);
                        }
                        values[i] = key;
                    } else {
                        values[i] = 0;
                    }
                }

                const sortedIdx = await gpuSorter.sort(values, ascending);
                const newIndices = new Uint32Array(rows.length);
                for (let i = 0; i < rows.length; i++) {
                    newIndices[i] = indices[sortedIdx[i]];
                }
                indices = newIndices;
            }

            // Apply permutation in-place
            const sorted = new Array(rows.length);
            for (let i = 0; i < rows.length; i++) {
                sorted[i] = rows[indices[i]];
            }
            for (let i = 0; i < rows.length; i++) {
                rows[i] = sorted[i];
            }
            return;
        }

        // CPU fallback for smaller datasets
        rows.sort((a, b) => {
            for (const ob of orderBy) {
                const colIdx = colIdxMap[ob.column.toLowerCase()];
                if (colIdx === undefined) continue;

                const valA = a[colIdx];
                const valB = b[colIdx];

                let cmp = 0;
                if (valA == null && valB == null) cmp = 0;
                else if (valA == null) cmp = 1;
                else if (valB == null) cmp = -1;
                else if (valA < valB) cmp = -1;
                else if (valA > valB) cmp = 1;

                if (cmp !== 0) {
                    return ob.descending ? -cmp : cmp;
                }
            }
            return 0;
        });
    }

    /**
     * Apply DISTINCT to rows (GPU-accelerated for large datasets)
     * @param {Array[]} rows - Row arrays to deduplicate
     * @returns {Array[]} Deduplicated rows
     */
    async applyDistinct(rows) {
        if (rows.length === 0) return rows;

        // Use GPU for large datasets (10,000+ rows)
        if (rows.length >= 10000 && gpuGrouper.isAvailable()) {
            // Hash each row to create a unique signature
            const rowHashes = this._hashRows(rows);

            // Use GPUGrouper to find unique hashes
            const { groupIds, numGroups } = await gpuGrouper.groupBy(rowHashes);

            // Extract first occurrence of each unique group
            const firstOccurrence = new Array(numGroups).fill(-1);
            for (let i = 0; i < rows.length; i++) {
                const gid = groupIds[i];
                if (firstOccurrence[gid] === -1) {
                    firstOccurrence[gid] = i;
                }
            }

            // Build deduplicated result
            const uniqueRows = [];
            for (let gid = 0; gid < numGroups; gid++) {
                if (firstOccurrence[gid] !== -1) {
                    uniqueRows.push(rows[firstOccurrence[gid]]);
                }
            }
            return uniqueRows;
        }

        // CPU fallback using Set with JSON serialization
        const seen = new Set();
        const uniqueRows = [];
        for (const row of rows) {
            const key = JSON.stringify(row);
            if (!seen.has(key)) {
                seen.add(key);
                uniqueRows.push(row);
            }
        }
        return uniqueRows;
    }

    /**
     * Hash rows to u32 for GPU deduplication
     */
    _hashRows(rows) {
        const hashes = new Uint32Array(rows.length);
        for (let i = 0; i < rows.length; i++) {
            hashes[i] = this._hashRow(rows[i]);
        }
        return hashes;
    }

    /**
     * FNV-1a hash of a row's values
     */
    _hashRow(row) {
        let hash = 2166136261;
        for (const val of row) {
            if (val === null || val === undefined) {
                hash ^= 0;
            } else if (typeof val === 'number') {
                // Hash number as bytes
                const buf = new ArrayBuffer(8);
                new Float64Array(buf)[0] = val;
                const bytes = new Uint8Array(buf);
                for (const b of bytes) {
                    hash ^= b;
                    hash = Math.imul(hash, 16777619);
                }
            } else if (typeof val === 'string') {
                for (let j = 0; j < val.length; j++) {
                    hash ^= val.charCodeAt(j);
                    hash = Math.imul(hash, 16777619);
                }
            } else {
                // Fallback: stringify
                const str = String(val);
                for (let j = 0; j < str.length; j++) {
                    hash ^= str.charCodeAt(j);
                    hash = Math.imul(hash, 16777619);
                }
            }
            // Separator between values
            hash ^= 0xFF;
            hash = Math.imul(hash, 16777619);
        }
        return hash >>> 0;
    }

    exprToName(expr) {
        if (!expr) return '?';
        switch (expr.type) {
            case 'column': return expr.name;
            case 'call': {
                const argStr = expr.args.map(a => {
                    if (a.type === 'star') return '*';
                    if (a.type === 'column') return a.name;
                    return '?';
                }).join(', ');
                return `${expr.name}(${argStr})`;
            }
            case 'literal': return String(expr.value);
            default: return '?';
        }
    }

    /**
     * Check if query is just SELECT COUNT(*) with no other columns
     */
    isSimpleCountStar(ast) {
        if (ast.columns.length !== 1) return false;
        const col = ast.columns[0];
        if (col.type === 'star') return true; // COUNT(*) parsed as star
        if (col.type === 'expr' && col.expr.type === 'call') {
            const name = col.expr.name.toUpperCase();
            if (name === 'COUNT') {
                const arg = col.expr.args[0];
                return arg?.type === 'star';
            }
        }
        return false;
    }

    /**
     * Execute aggregation query after running vector search
     */
    async executeAggregateWithSearch(ast, totalRows, onProgress) {
        // Step 1: Extract NEAR info from WHERE
        const nearInfo = this._extractNearCondition(ast.where);
        if (!nearInfo) {
            throw new Error('SEARCH aggregation requires NEAR clause');
        }

        // Step 2: Execute vector search to get candidate indices
        if (onProgress) onProgress('Executing vector search...', 0, 100);
        const searchIndices = await this._executeNearSearch(nearInfo, totalRows);

        if (searchIndices.length === 0) {
            return this._emptyAggregateResult(ast);
        }

        // Step 3: Apply remaining WHERE conditions (non-NEAR)
        const remainingWhere = this._removeNearCondition(ast.where);
        let filteredIndices = searchIndices;

        if (remainingWhere) {
            if (onProgress) onProgress('Applying filters...', 30, 100);
            filteredIndices = await this._filterIndicesByWhere(searchIndices, remainingWhere);
        }

        if (filteredIndices.length === 0) {
            return this._emptyAggregateResult(ast);
        }

        // Step 4: Route to appropriate aggregation path
        if (onProgress) onProgress('Aggregating results...', 60, 100);

        if (ast.groupBy && ast.groupBy.length > 0) {
            // GROUP BY aggregation on filtered indices
            return await this._executeGroupByOnIndices(ast, filteredIndices, onProgress);
        } else {
            // Simple aggregation on filtered indices
            return await this._executeSimpleAggregateOnIndices(ast, filteredIndices, onProgress);
        }
    }

    /**
     * Filter indices by WHERE expression
     * @private
     */
    async _filterIndicesByWhere(indices, whereExpr) {
        const neededCols = new Set();
        this.collectColumnsFromExpr(whereExpr, neededCols);

        const columnData = {};
        for (const colName of neededCols) {
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx !== undefined) {
                columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, indices);
            }
        }

        return indices.filter((_, i) => this.evaluateExpr(whereExpr, columnData, i));
    }

    /**
     * Execute simple aggregation on specific indices (no GROUP BY)
     * @private
     */
    async _executeSimpleAggregateOnIndices(ast, indices, onProgress) {
        const aggFunctions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];

        // Initialize aggregators
        const aggregators = [];
        const colNames = [];

        for (const col of ast.columns) {
            if (col.type === 'star') {
                aggregators.push({ type: 'COUNT', column: null, isStar: true, count: 0 });
                colNames.push('COUNT(*)');
            } else if (col.expr?.type === 'call' && aggFunctions.includes(col.expr.name.toUpperCase())) {
                const aggType = col.expr.name.toUpperCase();
                const argExpr = col.expr.args?.[0];
                const colName = argExpr?.type === 'column' ? (argExpr.name || argExpr.column) : null;
                const isStar = argExpr?.type === 'star';

                aggregators.push({
                    type: aggType,
                    column: colName,
                    isStar,
                    expr: col.expr,
                    sum: 0,
                    count: 0,
                    min: null,
                    max: null,
                    values: [],
                });

                const displayName = col.alias || this.exprToName(col.expr);
                colNames.push(displayName);
            } else {
                aggregators.push({
                    type: 'FIRST',
                    column: col.expr?.type === 'column' ? (col.expr.name || col.expr.column) : null,
                    value: null,
                });
                colNames.push(col.alias || this.exprToName(col.expr));
            }
        }

        // Collect needed columns
        const neededCols = new Set();
        for (const agg of aggregators) {
            if (agg.column) neededCols.add(agg.column.toLowerCase());
        }

        // Fetch column data for the indices
        const columnData = {};
        for (const colName of neededCols) {
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx !== undefined) {
                columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, indices);
            }
        }

        // Process all rows
        for (let i = 0; i < indices.length; i++) {
            for (const agg of aggregators) {
                if (agg.type === 'COUNT' && agg.isStar) {
                    agg.count++;
                } else if (agg.type === 'FIRST' && agg.value === null) {
                    const data = agg.column ? columnData[agg.column.toLowerCase()] : null;
                    agg.value = data ? data[i] : null;
                } else if (agg.column) {
                    const data = columnData[agg.column.toLowerCase()];
                    const val = data ? data[i] : null;

                    if (val !== null && val !== undefined) {
                        if (agg.type === 'COUNT') {
                            agg.count++;
                        } else if (typeof val === 'number' && !isNaN(val)) {
                            agg.count++;
                            if (agg.type === 'SUM' || agg.type === 'AVG' || agg.type.startsWith('STDDEV') || agg.type.startsWith('VAR')) {
                                agg.sum += val;
                                agg.values.push(val);
                            }
                            if (agg.type === 'MIN') {
                                agg.min = agg.min === null ? val : Math.min(agg.min, val);
                            }
                            if (agg.type === 'MAX') {
                                agg.max = agg.max === null ? val : Math.max(agg.max, val);
                            }
                            if (agg.type === 'MEDIAN') {
                                agg.values.push(val);
                            }
                        }
                        if (agg.type === 'STRING_AGG' || agg.type === 'GROUP_CONCAT') {
                            agg.values.push(String(val));
                        }
                    }
                }
            }
        }

        // Compute final results
        const resultRow = [];
        for (const agg of aggregators) {
            switch (agg.type) {
                case 'COUNT':
                    resultRow.push(agg.count);
                    break;
                case 'SUM':
                    resultRow.push(agg.sum);
                    break;
                case 'AVG':
                    resultRow.push(agg.count > 0 ? agg.sum / agg.count : null);
                    break;
                case 'MIN':
                    resultRow.push(agg.min);
                    break;
                case 'MAX':
                    resultRow.push(agg.max);
                    break;
                case 'STDDEV':
                case 'STDDEV_SAMP': {
                    if (agg.values.length < 2) {
                        resultRow.push(null);
                    } else {
                        const mean = agg.sum / agg.values.length;
                        const variance = agg.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (agg.values.length - 1);
                        resultRow.push(Math.sqrt(variance));
                    }
                    break;
                }
                case 'STDDEV_POP': {
                    if (agg.values.length === 0) {
                        resultRow.push(null);
                    } else {
                        const mean = agg.sum / agg.values.length;
                        const variance = agg.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / agg.values.length;
                        resultRow.push(Math.sqrt(variance));
                    }
                    break;
                }
                case 'VARIANCE':
                case 'VAR_SAMP': {
                    if (agg.values.length < 2) {
                        resultRow.push(null);
                    } else {
                        const mean = agg.sum / agg.values.length;
                        resultRow.push(agg.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (agg.values.length - 1));
                    }
                    break;
                }
                case 'VAR_POP': {
                    if (agg.values.length === 0) {
                        resultRow.push(null);
                    } else {
                        const mean = agg.sum / agg.values.length;
                        resultRow.push(agg.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / agg.values.length);
                    }
                    break;
                }
                case 'MEDIAN': {
                    if (agg.values.length === 0) {
                        resultRow.push(null);
                    } else {
                        agg.values.sort((a, b) => a - b);
                        const mid = Math.floor(agg.values.length / 2);
                        resultRow.push(agg.values.length % 2 ? agg.values[mid] : (agg.values[mid - 1] + agg.values[mid]) / 2);
                    }
                    break;
                }
                case 'STRING_AGG': {
                    const separator = agg.expr?.args?.[1]?.value ?? ',';
                    resultRow.push(agg.values.join(separator));
                    break;
                }
                case 'GROUP_CONCAT': {
                    resultRow.push(agg.values.join(','));
                    break;
                }
                case 'FIRST':
                    resultRow.push(agg.value);
                    break;
                default:
                    resultRow.push(null);
            }
        }

        return {
            columns: colNames,
            rows: [resultRow],
            total: 1,
            aggregationStats: {
                scannedRows: indices.length,
                totalRows: indices.length,
                coveragePercent: '100.00',
                isPartialScan: false,
                fromSearch: true,
            },
        };
    }

    /**
     * Execute GROUP BY aggregation on specific indices
     * @private
     */
    async _executeGroupByOnIndices(ast, indices, onProgress) {
        const aggFuncs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];

        // Collect all needed columns
        const neededCols = new Set();
        for (const expr of ast.groupBy) {
            const colName = expr.column || expr.name;
            if (colName) neededCols.add(colName.toLowerCase());
        }
        for (const col of ast.columns) {
            if (col.expr?.type === 'column') {
                neededCols.add((col.expr.name || col.expr.column).toLowerCase());
            } else if (col.expr?.type === 'call' && col.expr.args?.[0]?.type === 'column') {
                neededCols.add((col.expr.args[0].name || col.expr.args[0].column).toLowerCase());
            }
        }

        // Fetch column data for the indices
        const columnData = {};
        for (const colName of neededCols) {
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx !== undefined) {
                columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, indices);
            }
        }

        // Group rows by GROUP BY columns
        const groups = new Map();
        for (let i = 0; i < indices.length; i++) {
            const groupKey = ast.groupBy.map(expr => {
                const colName = (expr.column || expr.name || '').toLowerCase();
                return JSON.stringify(columnData[colName]?.[i]);
            }).join('|');

            if (!groups.has(groupKey)) {
                groups.set(groupKey, []);
            }
            groups.get(groupKey).push(i);
        }

        // Build column names
        const colNames = ast.columns.map(col => col.alias || this.exprToName(col.expr || col));

        // Compute aggregates per group
        const resultRows = [];
        for (const [, groupLocalIndices] of groups) {
            const row = [];
            for (const col of ast.columns) {
                const expr = col.expr || col;
                if (expr.type === 'call' && aggFuncs.includes(expr.name.toUpperCase())) {
                    const funcName = expr.name.toUpperCase();
                    const argExpr = expr.args?.[0];
                    const colName = argExpr?.type === 'column' ? (argExpr.name || argExpr.column)?.toLowerCase() : null;
                    const isStar = argExpr?.type === 'star';

                    let result = null;
                    if (funcName === 'COUNT') {
                        if (isStar) {
                            result = groupLocalIndices.length;
                        } else {
                            result = 0;
                            for (const i of groupLocalIndices) {
                                if (columnData[colName]?.[i] != null) result++;
                            }
                        }
                    } else if (funcName === 'SUM') {
                        result = 0;
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && typeof v === 'number' && !isNaN(v)) result += v;
                        }
                    } else if (funcName === 'AVG') {
                        let sum = 0, count = 0;
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && typeof v === 'number' && !isNaN(v)) { sum += v; count++; }
                        }
                        result = count > 0 ? sum / count : null;
                    } else if (funcName === 'MIN') {
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && (result === null || v < result)) result = v;
                        }
                    } else if (funcName === 'MAX') {
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && (result === null || v > result)) result = v;
                        }
                    } else if (funcName === 'MEDIAN') {
                        const vals = [];
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                        }
                        if (vals.length > 0) {
                            vals.sort((a, b) => a - b);
                            const mid = Math.floor(vals.length / 2);
                            result = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
                        }
                    } else if (funcName === 'STRING_AGG' || funcName === 'GROUP_CONCAT') {
                        const separator = funcName === 'STRING_AGG' ? (expr.args?.[1]?.value ?? ',') : ',';
                        const vals = [];
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null) vals.push(String(v));
                        }
                        result = vals.join(separator);
                    } else if (funcName === 'STDDEV' || funcName === 'STDDEV_SAMP' || funcName === 'STDDEV_POP') {
                        const vals = [];
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                        }
                        if (vals.length >= (funcName === 'STDDEV_POP' ? 1 : 2)) {
                            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                            const divisor = funcName === 'STDDEV_POP' ? vals.length : (vals.length - 1);
                            const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / divisor;
                            result = Math.sqrt(variance);
                        }
                    } else if (funcName === 'VARIANCE' || funcName === 'VAR_SAMP' || funcName === 'VAR_POP') {
                        const vals = [];
                        for (const i of groupLocalIndices) {
                            const v = columnData[colName]?.[i];
                            if (v != null && typeof v === 'number' && !isNaN(v)) vals.push(v);
                        }
                        if (vals.length >= (funcName === 'VAR_POP' ? 1 : 2)) {
                            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                            const divisor = funcName === 'VAR_POP' ? vals.length : (vals.length - 1);
                            result = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / divisor;
                        }
                    }
                    row.push(result);
                } else if (expr.type === 'column') {
                    const colName = (expr.name || expr.column)?.toLowerCase();
                    row.push(columnData[colName]?.[groupLocalIndices[0]] ?? null);
                } else {
                    row.push(null);
                }
            }
            resultRows.push(row);
        }

        // Apply ORDER BY
        if (ast.orderBy && ast.orderBy.length > 0) {
            const colNameToIdx = {};
            colNames.forEach((name, idx) => { colNameToIdx[name.toLowerCase()] = idx; });

            resultRows.sort((a, b) => {
                for (const order of ast.orderBy) {
                    const colName = (order.column || order.expr?.name || order.expr?.column || '').toLowerCase();
                    const colIdx = colNameToIdx[colName] ?? -1;
                    if (colIdx === -1) continue;

                    const aVal = a[colIdx];
                    const bVal = b[colIdx];
                    const dir = order.direction === 'DESC' ? -1 : 1;

                    if (aVal === null && bVal === null) continue;
                    if (aVal === null) return dir;
                    if (bVal === null) return -dir;
                    if (aVal < bVal) return -dir;
                    if (aVal > bVal) return dir;
                }
                return 0;
            });
        }

        // Apply LIMIT/OFFSET
        const offset = ast.offset || 0;
        let rows = resultRows;
        if (offset > 0) rows = rows.slice(offset);
        if (ast.limit) rows = rows.slice(0, ast.limit);

        return {
            columns: colNames,
            rows,
            total: rows.length,
            aggregationStats: {
                scannedRows: indices.length,
                totalRows: indices.length,
                groups: groups.size,
                coveragePercent: '100.00',
                isPartialScan: false,
                fromSearch: true,
            },
        };
    }

    /**
     * Return empty aggregate result with proper column names
     * @private
     */
    _emptyAggregateResult(ast) {
        const colNames = ast.columns.map(col => col.alias || this.exprToName(col.expr || col));
        const emptyRow = ast.columns.map(col => {
            const expr = col.expr || col;
            if (expr.type === 'call' && expr.name.toUpperCase() === 'COUNT') {
                return 0;
            }
            return null;
        });

        // For GROUP BY with no matches, return empty rows (no groups)
        if (ast.groupBy && ast.groupBy.length > 0) {
            return {
                columns: colNames,
                rows: [],
                total: 0,
                aggregationStats: {
                    scannedRows: 0,
                    totalRows: 0,
                    coveragePercent: '100.00',
                    isPartialScan: false,
                    fromSearch: true,
                },
            };
        }

        // For simple aggregates, return single row with COUNT=0, others NULL
        return {
            columns: colNames,
            rows: [emptyRow],
            total: 1,
            aggregationStats: {
                scannedRows: 0,
                totalRows: 0,
                coveragePercent: '100.00',
                isPartialScan: false,
                fromSearch: true,
            },
        };
    }

    /**
     * Check if the query contains aggregate functions
     */
    hasAggregates(ast) {
        const aggFunctions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];
        for (const col of ast.columns) {
            if (col.type === 'expr' && col.expr.type === 'call') {
                if (aggFunctions.includes(col.expr.name.toUpperCase())) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Execute an aggregation query
     */
    async executeAggregateQuery(ast, totalRows, onProgress) {
        const aggFunctions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];

        // Initialize aggregators for each column
        const aggregators = [];
        const colNames = [];

        for (const col of ast.columns) {
            if (col.type === 'star') {
                // COUNT(*) case
                aggregators.push({ type: 'COUNT', column: null, isStar: true });
                colNames.push('COUNT(*)');
            } else if (col.expr.type === 'call' && aggFunctions.includes(col.expr.name.toUpperCase())) {
                const aggType = col.expr.name.toUpperCase();
                const argExpr = col.expr.args[0];
                const colName = argExpr?.type === 'column' ? argExpr.name : null;
                const isStar = argExpr?.type === 'star';

                aggregators.push({
                    type: aggType,
                    column: colName,
                    isStar,
                    sum: 0,
                    count: 0,
                    min: null,
                    max: null,
                });

                const displayName = col.alias || this.exprToName(col.expr);
                colNames.push(displayName);
            } else {
                // Non-aggregate column - just take first value (or could error)
                aggregators.push({
                    type: 'FIRST',
                    column: col.expr.type === 'column' ? col.expr.name : null,
                    value: null,
                });
                colNames.push(col.alias || this.exprToName(col.expr));
            }
        }

        // Determine which columns we need to read
        const neededCols = new Set();
        for (const agg of aggregators) {
            if (agg.column) {
                neededCols.add(agg.column.toLowerCase());
            }
        }

        // Also need columns from WHERE clause
        if (ast.where) {
            this.collectColumnsFromExpr(ast.where, neededCols);
        }

        // Process data in batches
        // Respect LIMIT - only scan up to LIMIT rows for aggregation
        const scanLimit = ast.limit || totalRows; // If no LIMIT, scan all (could be slow)
        const maxRowsToScan = Math.min(scanLimit, totalRows);
        const batchSize = 1000;
        let processedRows = 0;
        let scannedRows = 0;

        for (let batchStart = 0; batchStart < maxRowsToScan; batchStart += batchSize) {
            if (onProgress) {
                onProgress(`Aggregating...`, batchStart, maxRowsToScan);
            }

            const batchEnd = Math.min(batchStart + batchSize, maxRowsToScan);
            const batchIndices = Array.from({ length: batchEnd - batchStart }, (_, i) => batchStart + i);
            scannedRows += batchIndices.length;

            // Fetch needed column data for this batch
            const batchData = {};
            for (const colName of neededCols) {
                const colIdx = this.columnMap[colName.toLowerCase()];
                if (colIdx !== undefined) {
                    batchData[colName.toLowerCase()] = await this.readColumnData(colIdx, batchIndices);
                }
            }

            // Process each row in the batch
            for (let i = 0; i < batchIndices.length; i++) {
                // Apply WHERE filter if present
                if (ast.where) {
                    const matches = this.evaluateExpr(ast.where, batchData, i);
                    if (!matches) continue;
                }

                processedRows++;

                // Update aggregators
                for (const agg of aggregators) {
                    if (agg.type === 'COUNT') {
                        agg.count++;
                    } else if (agg.type === 'FIRST') {
                        if (agg.value === null && agg.column) {
                            const data = batchData[agg.column.toLowerCase()];
                            if (data) agg.value = data[i];
                        }
                    } else {
                        // SUM, AVG, MIN, MAX need column value
                        const data = agg.column ? batchData[agg.column.toLowerCase()] : null;
                        const val = data ? data[i] : null;

                        if (val !== null && val !== undefined && !isNaN(val)) {
                            agg.count++;
                            if (agg.type === 'SUM' || agg.type === 'AVG') {
                                agg.sum += val;
                            }
                            if (agg.type === 'MIN') {
                                agg.min = agg.min === null ? val : Math.min(agg.min, val);
                            }
                            if (agg.type === 'MAX') {
                                agg.max = agg.max === null ? val : Math.max(agg.max, val);
                            }
                        }
                    }
                }
            }
        }

        // Build result row
        const resultRow = [];
        for (const agg of aggregators) {
            switch (agg.type) {
                case 'COUNT':
                    resultRow.push(agg.count);
                    break;
                case 'SUM':
                    resultRow.push(agg.sum);
                    break;
                case 'AVG':
                    resultRow.push(agg.count > 0 ? agg.sum / agg.count : null);
                    break;
                case 'MIN':
                    resultRow.push(agg.min);
                    break;
                case 'MAX':
                    resultRow.push(agg.max);
                    break;
                case 'FIRST':
                    resultRow.push(agg.value);
                    break;
                default:
                    resultRow.push(null);
            }
        }

        // Apply HAVING filter if present
        if (ast.having) {
            // Build column data for HAVING evaluation
            const havingColumnData = {};
            for (let i = 0; i < colNames.length; i++) {
                // Support both column name and alias lookup
                const colName = colNames[i].toLowerCase();
                havingColumnData[colName] = [resultRow[i]];
                // Also support aggregate function names like COUNT(*)
                const cleanName = colName.replace(/[()]/g, '').replace('*', 'star');
                havingColumnData[cleanName] = [resultRow[i]];
            }

            // Evaluate HAVING condition
            if (!this._evaluateInMemoryExpr(ast.having, havingColumnData, 0)) {
                // HAVING condition not met - return empty result
                return {
                    columns: colNames,
                    rows: [],
                    total: 0,
                    aggregationStats: {
                        scannedRows,
                        totalRows,
                        coveragePercent: totalRows > 0 ? ((scannedRows / totalRows) * 100).toFixed(2) : 100,
                        isPartialScan: scannedRows < totalRows,
                        havingFiltered: true,
                    },
                };
            }
        }

        // Calculate coverage stats
        const coveragePercent = totalRows > 0 ? ((scannedRows / totalRows) * 100).toFixed(2) : 100;
        const isPartialScan = scannedRows < totalRows;

        return {
            columns: colNames,
            rows: [resultRow],
            total: 1,
            aggregationStats: {
                scannedRows,
                totalRows,
                coveragePercent,
                isPartialScan,
            },
        };
    }

    /**
     * Expand GROUP BY clause into list of grouping sets.
     * Handles ROLLUP, CUBE, and GROUPING SETS operators.
     * @param {Array} groupBy - Array of GROUP BY items
     * @returns {Array<Array<string>>} - List of grouping sets (each is array of column names)
     */
    _expandGroupBy(groupBy) {
        if (!groupBy || groupBy.length === 0) return [[]];

        // Check if it's old-style simple column list (backward compat)
        // Old style: ['col1', 'col2']
        if (typeof groupBy[0] === 'string') {
            return [groupBy];
        }

        let result = [[]];

        for (const item of groupBy) {
            if (item.type === 'COLUMN') {
                // Simple column: cross-product with single column added
                result = result.map(set => [...set, item.column]);
            } else if (item.type === 'ROLLUP') {
                // ROLLUP(a, b, c) generates: (a,b,c), (a,b), (a), ()
                const rollupSets = [];
                for (let i = item.columns.length; i >= 0; i--) {
                    rollupSets.push(item.columns.slice(0, i));
                }
                result = this._crossProductSets(result, rollupSets);
            } else if (item.type === 'CUBE') {
                // CUBE(a, b) generates all 2^n subsets: (a,b), (a), (b), ()
                const cubeSets = this._powerSet(item.columns);
                result = this._crossProductSets(result, cubeSets);
            } else if (item.type === 'GROUPING_SETS') {
                // GROUPING SETS uses explicit sets
                result = this._crossProductSets(result, item.sets);
            }
        }

        // Deduplicate grouping sets
        const seen = new Set();
        return result.filter(set => {
            const key = JSON.stringify(set.sort());
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }

    /**
     * Generate power set (all subsets) of an array
     * @param {Array} arr - Input array
     * @returns {Array<Array>} - All subsets
     */
    _powerSet(arr) {
        const result = [[]];
        for (const item of arr) {
            const len = result.length;
            for (let i = 0; i < len; i++) {
                result.push([...result[i], item]);
            }
        }
        return result;
    }

    /**
     * Cross-product two lists of sets
     * @param {Array<Array>} sets1 - First list of sets
     * @param {Array<Array>} sets2 - Second list of sets
     * @returns {Array<Array>} - Combined sets
     */
    _crossProductSets(sets1, sets2) {
        const result = [];
        for (const s1 of sets1) {
            for (const s2 of sets2) {
                result.push([...s1, ...s2]);
            }
        }
        return result;
    }

    /**
     * Check if GROUP BY uses advanced operators (ROLLUP, CUBE, GROUPING SETS)
     */
    _hasAdvancedGroupBy(groupBy) {
        if (!groupBy || groupBy.length === 0) return false;
        if (typeof groupBy[0] === 'string') return false;
        return groupBy.some(item =>
            item.type === 'ROLLUP' || item.type === 'CUBE' || item.type === 'GROUPING_SETS'
        );
    }

    /**
     * Execute SQL and return results as async generator (streaming).
     * Yields chunks of {columns, rows} for memory-efficient processing.
     * @param {string} sql - SQL query string
     * @param {Object} options - Streaming options
     * @returns {AsyncGenerator<{columns: string[], rows: any[][]}>}
     */
    async *executeStream(sql, options = {}) {
        const { chunkSize = 1000 } = options;

        // Parse SQL
        const lexer = new SQLLexer(sql);
        const tokens = lexer.tokenize();
        const parser = new SQLParser(tokens);
        const ast = parser.parse();

        // Detect column types
        if (this.columnTypes.length === 0) {
            if (this.file._isRemote && this.file.detectColumnTypes) {
                this.columnTypes = await this.file.detectColumnTypes();
            } else if (this.file._columnTypes) {
                this.columnTypes = this.file._columnTypes;
            } else {
                this.columnTypes = Array(this.file.numColumns || 0).fill('unknown');
            }
        }

        // Get total rows
        const totalRows = this.file._isRemote
            ? await this.file.getRowCount(0)
            : Number(this.file.getRowCount(0));

        // Determine columns
        const neededColumns = this.collectNeededColumns(ast);
        const outputColumns = this.resolveOutputColumns(ast);

        // Stream in chunks
        const limit = ast.limit || totalRows;
        let yielded = 0;

        for (let offset = 0; offset < totalRows && yielded < limit; offset += chunkSize) {
            const batchSize = Math.min(chunkSize, limit - yielded, totalRows - offset);

            // Generate indices for this chunk
            const indices = [];
            for (let i = 0; i < batchSize; i++) {
                indices.push(offset + i);
            }

            // Read column data for these indices
            const columnData = [];
            for (const colName of neededColumns) {
                const colIdx = this.columnMap[colName.toLowerCase()];
                if (colIdx !== undefined) {
                    const data = await this.readColumnAtIndices(colIdx, indices);
                    columnData.push(data);
                } else {
                    columnData.push(indices.map(() => null));
                }
            }

            // Build rows
            const rows = [];
            for (let i = 0; i < indices.length; i++) {
                const row = [];
                for (let c = 0; c < neededColumns.length; c++) {
                    row.push(columnData[c][i]);
                }
                rows.push(row);
            }

            // Apply WHERE filter if present
            let filteredRows = rows;
            if (ast.where) {
                filteredRows = rows.filter((row, idx) => {
                    return this.evaluateWhereExprOnRow(ast.where, neededColumns, row);
                });
            }

            if (filteredRows.length > 0) {
                yield {
                    columns: neededColumns,
                    rows: filteredRows,
                };
                yielded += filteredRows.length;
            }
        }
    }

    /**
     * Evaluate WHERE expression on a single row
     * @private
     */
    evaluateWhereExprOnRow(expr, columns, row) {
        if (!expr) return true;

        if (expr.type === 'binary') {
            if (expr.op === 'AND') {
                return this.evaluateWhereExprOnRow(expr.left, columns, row) &&
                       this.evaluateWhereExprOnRow(expr.right, columns, row);
            }
            if (expr.op === 'OR') {
                return this.evaluateWhereExprOnRow(expr.left, columns, row) ||
                       this.evaluateWhereExprOnRow(expr.right, columns, row);
            }

            const leftVal = this._getValueFromExpr(expr.left, columns, row);
            const rightVal = this._getValueFromExpr(expr.right, columns, row);

            switch (expr.op) {
                case '=':
                case '==':
                    return leftVal == rightVal;
                case '!=':
                case '<>':
                    return leftVal != rightVal;
                case '<':
                    return leftVal < rightVal;
                case '<=':
                    return leftVal <= rightVal;
                case '>':
                    return leftVal > rightVal;
                case '>=':
                    return leftVal >= rightVal;
                default:
                    return true;
            }
        }

        return true;
    }

    /**
     * Get value from expression for row evaluation
     * @private
     */
    _getValueFromExpr(expr, columns, row) {
        if (expr.type === 'literal') {
            return expr.value;
        }
        if (expr.type === 'column') {
            const colName = expr.name || expr.column;
            const idx = columns.indexOf(colName) !== -1
                ? columns.indexOf(colName)
                : columns.indexOf(colName.toLowerCase());
            return idx !== -1 ? row[idx] : null;
        }
        return null;
    }
}

export { SQLExecutor };
