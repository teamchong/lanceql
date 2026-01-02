/**
 * QueryPlanner - Cost-based query plan generation
 */

class StatisticsManager {
    constructor() {
        this._cache = new Map(); // In-memory cache: datasetUrl -> { columns: Map<colName, stats> }
        this._opfsRoot = null;
        this._computing = new Map(); // Track in-progress computations to avoid duplicates
    }

    /**
     * Get OPFS directory for statistics cache
     */
    async _getStatsDir() {
        if (this._opfsRoot) return this._opfsRoot;

        if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
            return null; // OPFS not available
        }

        try {
            const opfsRoot = await navigator.storage.getDirectory();
            this._opfsRoot = await opfsRoot.getDirectoryHandle('lanceql-stats', { create: true });
            return this._opfsRoot;
        } catch (e) {
            console.warn('[StatisticsManager] OPFS not available:', e);
            return null;
        }
    }

    /**
     * Get cache key for a dataset
     */
    _getCacheKey(datasetUrl) {
        // Hash the URL for filesystem-safe name
        let hash = 0;
        for (let i = 0; i < datasetUrl.length; i++) {
            hash = ((hash << 5) - hash) + datasetUrl.charCodeAt(i);
            hash |= 0;
        }
        return `stats_${Math.abs(hash).toString(16)}`;
    }

    /**
     * Load cached statistics from OPFS
     */
    async loadFromCache(datasetUrl, version) {
        const cacheKey = this._getCacheKey(datasetUrl);

        // Check in-memory cache first
        if (this._cache.has(cacheKey)) {
            const cached = this._cache.get(cacheKey);
            if (cached.version === version) {
                return cached;
            }
        }

        // Try OPFS
        const statsDir = await this._getStatsDir();
        if (!statsDir) return null;

        try {
            const fileHandle = await statsDir.getFileHandle(`${cacheKey}.json`);
            const file = await fileHandle.getFile();
            const text = await file.text();
            const cached = JSON.parse(text);

            // Validate version
            if (cached.version !== version) {
                return null; // Stale cache
            }

            // Store in memory cache
            this._cache.set(cacheKey, cached);
            return cached;
        } catch (e) {
            return null; // No cache or read error
        }
    }

    /**
     * Save statistics to OPFS cache
     */
    async saveToCache(datasetUrl, version, statistics) {
        const cacheKey = this._getCacheKey(datasetUrl);

        const cacheData = {
            datasetUrl,
            version,
            timestamp: Date.now(),
            columns: statistics.columns, // Map serialized as object
            fragments: statistics.fragments || null
        };

        // Store in memory
        this._cache.set(cacheKey, cacheData);

        // Persist to OPFS
        const statsDir = await this._getStatsDir();
        if (!statsDir) return;

        try {
            const fileHandle = await statsDir.getFileHandle(`${cacheKey}.json`, { create: true });
            const writable = await fileHandle.createWritable();
            await writable.write(JSON.stringify(cacheData));
            await writable.close();
        } catch (e) {
            console.warn('[StatisticsManager] Failed to persist stats:', e);
        }
    }

    /**
     * Get statistics for a column, computing if necessary.
     *
     * @param {RemoteLanceDataset} dataset - The dataset
     * @param {string} columnName - Column name
     * @param {object} options - Options
     * @param {number} [options.sampleSize] - Max rows to sample (default: 100000)
     * @returns {Promise<ColumnStatistics>}
     */
    async getColumnStats(dataset, columnName, options = {}) {
        const datasetUrl = dataset.baseUrl;
        const version = dataset._version;
        const sampleSize = options.sampleSize || 100000;

        // Try to load from cache
        const cached = await this.loadFromCache(datasetUrl, version);
        if (cached?.columns?.[columnName]) {
            return cached.columns[columnName];
        }

        // Check if already computing
        const computeKey = `${datasetUrl}:${columnName}`;
        if (this._computing.has(computeKey)) {
            return this._computing.get(computeKey);
        }

        // Compute statistics by streaming through data
        const computePromise = this._computeColumnStats(dataset, columnName, sampleSize);
        this._computing.set(computeKey, computePromise);

        try {
            const stats = await computePromise;

            // Merge into cache
            const existing = await this.loadFromCache(datasetUrl, version) || { columns: {} };
            existing.columns[columnName] = stats;
            await this.saveToCache(datasetUrl, version, existing);

            return stats;
        } finally {
            this._computing.delete(computeKey);
        }
    }

    /**
     * Compute statistics for a column by streaming through data.
     */
    async _computeColumnStats(dataset, columnName, sampleSize) {
        const colIdx = dataset.schema.findIndex(c => c.name === columnName);
        if (colIdx === -1) {
            throw new Error(`Column not found: ${columnName}`);
        }

        const colType = dataset._columnTypes?.[colIdx] || 'unknown';
        const isNumeric = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'double'].includes(colType);

        const stats = {
            column: columnName,
            type: colType,
            rowCount: 0,
            nullCount: 0,
            min: null,
            max: null,
            computed: true,
            sampleSize: 0
        };

        // Stream through fragments
        let rowsProcessed = 0;

        for (let fragIdx = 0; fragIdx < dataset._fragments.length && rowsProcessed < sampleSize; fragIdx++) {
            try {
                const fragFile = await dataset.openFragment(fragIdx);
                const fragRows = Math.min(
                    dataset._fragments[fragIdx].numRows,
                    sampleSize - rowsProcessed
                );

                // Read column data
                const indices = Array.from({ length: fragRows }, (_, i) => i);
                const values = await fragFile.readColumnAtIndices(colIdx, indices);

                for (const value of values) {
                    stats.rowCount++;
                    stats.sampleSize++;

                    if (value === null || value === undefined) {
                        stats.nullCount++;
                        continue;
                    }

                    if (isNumeric) {
                        if (stats.min === null || value < stats.min) stats.min = value;
                        if (stats.max === null || value > stats.max) stats.max = value;
                    }
                }

                rowsProcessed += values.length;
            } catch (e) {
                console.warn(`[StatisticsManager] Error reading fragment ${fragIdx}:`, e);
            }
        }

        console.log(`[StatisticsManager] Computed stats for ${columnName}: min=${stats.min}, max=${stats.max}, nulls=${stats.nullCount}/${stats.rowCount}`);
        return stats;
    }

    /**
     * Compute statistics for all filter columns in a query plan.
     * This is called before query execution to enable pruning.
     */
    async precomputeForPlan(dataset, plan) {
        const filterColumns = new Set();

        // Collect columns from pushed filters
        for (const filter of (plan.pushedFilters || [])) {
            if (filter.column) filterColumns.add(filter.column);
            if (filter.left?.column) filterColumns.add(filter.left.column);
            if (filter.right?.column) filterColumns.add(filter.right.column);
        }

        // Compute stats in parallel
        const statsPromises = Array.from(filterColumns).map(col =>
            this.getColumnStats(dataset, col).catch(e => {
                console.warn(`[StatisticsManager] Failed to compute stats for ${col}:`, e);
                return null;
            })
        );

        const results = await Promise.all(statsPromises);
        const statsMap = new Map();

        Array.from(filterColumns).forEach((col, i) => {
            if (results[i]) statsMap.set(col, results[i]);
        });

        return statsMap;
    }

    /**
     * Check if a filter can be satisfied by a fragment's statistics.
     * Returns false if we can definitively skip this fragment.
     */
    canMatchFragment(fragmentStats, filter) {
        if (!fragmentStats || !filter) return true; // Can't determine, must scan

        const colStats = fragmentStats[filter.column];
        if (!colStats || colStats.min === null || colStats.max === null) return true;

        switch (filter.type) {
            case 'equality':
                // col = value: skip if value outside [min, max]
                return filter.value >= colStats.min && filter.value <= colStats.max;

            case 'range':
                switch (filter.op) {
                    case '>':
                        // col > value: skip if max <= value
                        return colStats.max > filter.value;
                    case '>=':
                        return colStats.max >= filter.value;
                    case '<':
                        // col < value: skip if min >= value
                        return colStats.min < filter.value;
                    case '<=':
                        return colStats.min <= filter.value;
                }
                break;

            case 'between':
                // col BETWEEN low AND high: skip if max < low OR min > high
                return colStats.max >= filter.low && colStats.min <= filter.high;

            case 'in':
                // col IN (values): skip if all values outside [min, max]
                if (Array.isArray(filter.values)) {
                    return filter.values.some(v => v >= colStats.min && v <= colStats.max);
                }
                break;
        }

        return true; // Default: can't skip
    }

    /**
     * Compute per-fragment statistics for a column.
     * This enables fine-grained fragment pruning.
     */
    async getFragmentStats(dataset, columnName, fragmentIndex) {
        const datasetUrl = dataset.baseUrl;
        const version = dataset._version;
        const cacheKey = `${datasetUrl}:frag${fragmentIndex}:${columnName}`;

        // Check if already computed
        const cached = await this.loadFromCache(datasetUrl, version);
        if (cached?.fragments?.[fragmentIndex]?.[columnName]) {
            return cached.fragments[fragmentIndex][columnName];
        }

        // Compute stats for this fragment only
        const colIdx = dataset.schema.findIndex(c => c.name === columnName);
        if (colIdx === -1) return null;

        const colType = dataset._columnTypes?.[colIdx] || 'unknown';
        const isNumeric = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'double'].includes(colType);

        try {
            const fragFile = await dataset.openFragment(fragmentIndex);
            const fragRows = dataset._fragments[fragmentIndex].numRows;

            // Sample up to 10000 rows for fragment stats
            const sampleSize = Math.min(fragRows, 10000);
            const indices = Array.from({ length: sampleSize }, (_, i) => i);
            const values = await fragFile.readColumnAtIndices(colIdx, indices);

            const stats = {
                fragmentIndex,
                column: columnName,
                rowCount: fragRows,
                sampledRows: sampleSize,
                nullCount: 0,
                min: null,
                max: null
            };

            for (const value of values) {
                if (value === null || value === undefined) {
                    stats.nullCount++;
                    continue;
                }
                if (isNumeric) {
                    if (stats.min === null || value < stats.min) stats.min = value;
                    if (stats.max === null || value > stats.max) stats.max = value;
                }
            }

            // Cache fragment stats
            const existing = await this.loadFromCache(datasetUrl, version) || { columns: {}, fragments: {} };
            if (!existing.fragments) existing.fragments = {};
            if (!existing.fragments[fragmentIndex]) existing.fragments[fragmentIndex] = {};
            existing.fragments[fragmentIndex][columnName] = stats;
            await this.saveToCache(datasetUrl, version, existing);

            return stats;
        } catch (e) {
            console.warn(`[StatisticsManager] Error computing fragment ${fragmentIndex} stats:`, e);
            return null;
        }
    }

    /**
     * Get fragments that might match a filter based on statistics.
     * Returns indices of fragments that can't be pruned.
     */
    async getPrunableFragments(dataset, filters) {
        if (!filters || filters.length === 0 || !dataset._fragments) {
            return null; // Can't prune
        }

        const numFragments = dataset._fragments.length;
        const matchingFragments = [];
        let fragmentsPruned = 0;

        // Get filter columns
        const filterColumns = new Set();
        for (const filter of filters) {
            if (filter.column) filterColumns.add(filter.column);
        }

        for (let fragIdx = 0; fragIdx < numFragments; fragIdx++) {
            let canPrune = false;

            for (const filter of filters) {
                if (!filter.column) continue;

                // Get stats for this fragment/column (computed lazily)
                const fragStats = await this.getFragmentStats(dataset, filter.column, fragIdx);

                if (fragStats && !this.canMatchFragment({ [filter.column]: fragStats }, filter)) {
                    canPrune = true;
                    break;
                }
            }

            if (!canPrune) {
                matchingFragments.push(fragIdx);
            } else {
                fragmentsPruned++;
            }
        }

        console.log(`[StatisticsManager] Fragment pruning: ${fragmentsPruned}/${numFragments} fragments pruned`);

        return {
            matchingFragments,
            fragmentsPruned,
            totalFragments: numFragments
        };
    }
}

// Global statistics manager instance
const statisticsManager = new StatisticsManager();

/**
 * Cost Model - estimates query execution cost for remote vs local strategies.
 *
 * This enables the optimizer to choose between:
 * - Remote: Minimize RTTs and bytes transferred (high latency, high bandwidth cost)
 * - Local: Maximize sequential I/O (low latency, CPU-bound)
 *
 * Similar to how DuckDB and DataFusion estimate query costs.
 */

class CostModel {
    constructor(options = {}) {
        this.isRemote = options.isRemote ?? true;

        // Network costs (ms)
        this.rttLatency = options.rttLatency ?? 50; // Round-trip time
        this.bandwidthMBps = options.bandwidthMBps ?? 10; // MB/s

        // CPU costs (ms per row)
        this.filterCostPerRow = options.filterCostPerRow ?? 0.001;
        this.hashBuildCostPerRow = options.hashBuildCostPerRow ?? 0.01;
        this.hashProbeCostPerRow = options.hashProbeCostPerRow ?? 0.005;

        // Memory costs
        this.memoryLimitMB = options.memoryLimitMB ?? 512;
    }

    /**
     * Estimate cost of scanning a table/fragment
     */
    estimateScanCost(rowCount, columnBytes, selectivity = 1.0) {
        const bytesToFetch = rowCount * columnBytes * selectivity;

        // Network cost for remote, near-zero for local
        const networkCost = this.isRemote
            ? this.rttLatency + (bytesToFetch / (this.bandwidthMBps * 1024 * 1024)) * 1000
            : 0.1; // Local disk is nearly instant

        // CPU cost (filtering)
        const cpuCost = rowCount * this.filterCostPerRow;

        return {
            totalMs: networkCost + cpuCost,
            networkMs: networkCost,
            cpuMs: cpuCost,
            bytesToFetch,
            rowsToScan: rowCount * selectivity
        };
    }

    /**
     * Estimate cost of a hash join
     */
    estimateJoinCost(leftRows, rightRows, leftBytes, rightBytes, joinSelectivity = 0.1) {
        // Build phase: hash the smaller table
        const buildRows = Math.min(leftRows, rightRows);
        const buildBytes = buildRows < leftRows ? leftBytes : rightBytes;
        const buildCost = buildRows * this.hashBuildCostPerRow;

        // Probe phase: scan the larger table
        const probeRows = Math.max(leftRows, rightRows);
        const probeCost = probeRows * this.hashProbeCostPerRow;

        // Memory check: can we fit build side in RAM?
        const buildMemoryMB = (buildRows * buildBytes) / (1024 * 1024);
        const needsSpill = buildMemoryMB > this.memoryLimitMB;

        // Spill cost (OPFS write + read)
        const spillCost = needsSpill ? buildMemoryMB * 10 : 0; // ~10ms per MB for OPFS

        return {
            totalMs: buildCost + probeCost + spillCost,
            buildMs: buildCost,
            probeMs: probeCost,
            spillMs: spillCost,
            needsSpill,
            outputRows: Math.round(leftRows * rightRows * joinSelectivity)
        };
    }

    /**
     * Estimate cost of an aggregation
     */
    estimateAggregateCost(inputRows, groupCount, aggCount) {
        // Cost scales with input rows and number of groups
        const hashGroupCost = inputRows * this.hashBuildCostPerRow;
        const aggComputeCost = inputRows * aggCount * 0.0001; // Aggregation is cheap

        return {
            totalMs: hashGroupCost + aggComputeCost,
            outputRows: groupCount
        };
    }

    /**
     * Compare two plan costs and recommend the better one
     */
    comparePlans(planA, planB) {
        const costA = planA.totalCost || this.estimatePlanCost(planA);
        const costB = planB.totalCost || this.estimatePlanCost(planB);

        return {
            recommended: costA.totalMs < costB.totalMs ? 'A' : 'B',
            costA,
            costB,
            savings: Math.abs(costA.totalMs - costB.totalMs)
        };
    }

    /**
     * Estimate total cost of a query plan
     */
    estimatePlanCost(plan) {
        let totalMs = 0;
        let totalBytes = 0;
        let operations = [];

        // Scan costs
        if (plan.leftScan) {
            const scanCost = this.estimateScanCost(
                plan.leftScan.estimatedRows || 10000,
                plan.leftScan.columnBytes || 100,
                plan.leftScan.selectivity || 1.0
            );
            totalMs += scanCost.totalMs;
            totalBytes += scanCost.bytesToFetch;
            operations.push({ op: 'scan_left', ...scanCost });
        }

        if (plan.rightScan) {
            const scanCost = this.estimateScanCost(
                plan.rightScan.estimatedRows || 10000,
                plan.rightScan.columnBytes || 100,
                plan.rightScan.selectivity || 1.0
            );
            totalMs += scanCost.totalMs;
            totalBytes += scanCost.bytesToFetch;
            operations.push({ op: 'scan_right', ...scanCost });
        }

        // Join costs
        if (plan.join) {
            const joinCost = this.estimateJoinCost(
                plan.leftScan?.estimatedRows || 10000,
                plan.rightScan?.estimatedRows || 10000,
                plan.leftScan?.columnBytes || 100,
                plan.rightScan?.columnBytes || 100,
                plan.join.selectivity || 0.1
            );
            totalMs += joinCost.totalMs;
            operations.push({ op: 'join', ...joinCost });
        }

        // Aggregation costs
        if (plan.aggregations && plan.aggregations.length > 0) {
            const aggCost = this.estimateAggregateCost(
                plan.estimatedInputRows || 10000,
                plan.groupBy?.length || 1,
                plan.aggregations.length
            );
            totalMs += aggCost.totalMs;
            operations.push({ op: 'aggregate', ...aggCost });
        }

        return {
            totalMs,
            totalBytes,
            operations,
            isRemote: this.isRemote
        };
    }
}

// Export cost model

/**
 * OPFS-only storage for Lance database files.
 *
 * Uses Origin Private File System (OPFS) exclusively - no IndexedDB.
 * This avoids migration complexity as data grows.
 *
 * OPFS benefits:
 * - High performance file access
 * - No size limits (beyond disk quota)
 * - File-like API suitable for Lance format
 * - Same approach as SQLite WASM
 */

class QueryPlanner {
    constructor() {
        this.debug = true; // Enable query plan logging
    }

    /**
     * Generate physical execution plan from logical AST
     * @param {Object} ast - Parsed SQL AST
     * @param {Object} context - Table names and aliases
     * @returns {Object} Physical execution plan
     */
    plan(ast, context) {
        const { leftTableName, leftAlias, rightTableName, rightAlias } = context;

        // Analyze what columns are needed from each table
        const columnAnalysis = this._analyzeColumns(ast, context);

        // Separate filters by table (pushdown optimization)
        const filterAnalysis = this._analyzeFilters(ast, context);

        // Estimate how many rows to fetch (over-fetch for safety)
        const fetchEstimate = this._estimateFetchSize(ast, filterAnalysis);

        // Build physical plan
        const plan = {
            // Step 1: Scan left table
            leftScan: {
                table: leftTableName,
                alias: leftAlias,
                columns: columnAnalysis.left.all,  // Deduplicated list
                filters: filterAnalysis.left,
                limit: fetchEstimate.left,
                purpose: {
                    join: columnAnalysis.left.join,
                    where: columnAnalysis.left.where,
                    result: columnAnalysis.left.result
                }
            },

            // Step 2: Scan right table
            rightScan: {
                table: rightTableName,
                alias: rightAlias,
                columns: columnAnalysis.right.all,
                filters: filterAnalysis.right,
                filterByJoinKeys: true,  // Will add IN clause dynamically
                purpose: {
                    join: columnAnalysis.right.join,
                    where: columnAnalysis.right.where,
                    result: columnAnalysis.right.result
                }
            },

            // Step 3: Join strategy
            join: {
                type: ast.joins[0].type,
                leftKey: columnAnalysis.joinKeys.left,
                rightKey: columnAnalysis.joinKeys.right,
                algorithm: 'HASH_JOIN'  // Could be SORT_MERGE or NESTED_LOOP in future
            },

            // Step 4: Final projection
            projection: columnAnalysis.resultColumns,

            // Step 5: Limit
            limit: ast.limit || null,
            offset: ast.offset || 0
        };

        if (this.debug) {
            this._logPlan(plan, ast);
        }

        return plan;
    }

    /**
     * Analyze which columns are needed from each table
     */
    _analyzeColumns(ast, context) {
        const { leftAlias, rightAlias } = context;

        const left = {
            join: new Set(),    // Columns needed for JOIN key
            where: new Set(),   // Columns needed for WHERE filter
            result: new Set(),  // Columns needed in final result
            all: []             // Deduplicated union of above
        };

        const right = {
            join: new Set(),
            where: new Set(),
            result: new Set(),
            all: []
        };

        // 1. Analyze SELECT columns (result set)
        for (const item of ast.columns) {
            if (item.type === 'star') {
                // SELECT * - need all columns (can't optimize this easily)
                left.result.add('*');
                right.result.add('*');
            } else if (item.type === 'expr' && item.expr.type === 'column') {
                const col = item.expr;
                const table = col.table || null;
                const column = col.column;

                if (!table || table === leftAlias) {
                    left.result.add(column);
                }
                if (!table || table === rightAlias) {
                    right.result.add(column);
                }
            }
        }

        // 2. Analyze JOIN ON condition (join keys)
        const join = ast.joins[0];
        const joinKeys = this._extractJoinKeys(join.on, leftAlias, rightAlias);

        if (joinKeys.left) {
            left.join.add(joinKeys.left);
        }
        if (joinKeys.right) {
            right.join.add(joinKeys.right);
        }

        // 3. Analyze WHERE clause (filter columns)
        if (ast.where) {
            this._extractWhereColumns(ast.where, leftAlias, rightAlias, left.where, right.where);
        }

        // 4. Deduplicate: merge join, where, result
        left.all = [...new Set([...left.join, ...left.where, ...left.result])];
        right.all = [...new Set([...right.join, ...right.where, ...right.result])];

        // 5. Handle SELECT *
        if (left.result.has('*')) {
            left.all = ['*'];
        }
        if (right.result.has('*')) {
            right.all = ['*'];
        }

        // 6. Determine final result columns (for projection after join)
        const resultColumns = [];
        for (const item of ast.columns) {
            if (item.type === 'star') {
                resultColumns.push('*');
            } else if (item.type === 'expr' && item.expr.type === 'column') {
                const col = item.expr;
                const alias = item.alias || `${col.table || ''}.${col.column}`.replace(/^\./, '');
                resultColumns.push({
                    table: col.table,
                    column: col.column,
                    alias: alias
                });
            }
        }

        return {
            left,
            right,
            joinKeys,
            resultColumns
        };
    }

    /**
     * Extract join keys from ON condition
     */
    _extractJoinKeys(onExpr, leftAlias, rightAlias) {
        if (!onExpr || onExpr.type !== 'binary') {
            return { left: null, right: null };
        }

        const leftCol = onExpr.left;
        const rightCol = onExpr.right;

        let leftKey = null;
        let rightKey = null;

        // Left side of equality
        if (leftCol.type === 'column') {
            if (!leftCol.table || leftCol.table === leftAlias) {
                leftKey = leftCol.column;
            } else if (leftCol.table === rightAlias) {
                rightKey = leftCol.column;
            }
        }

        // Right side of equality
        if (rightCol.type === 'column') {
            if (!rightCol.table || rightCol.table === leftAlias) {
                leftKey = rightCol.column;
            } else if (rightCol.table === rightAlias) {
                rightKey = rightCol.column;
            }
        }

        return { left: leftKey, right: rightKey };
    }

    /**
     * Extract columns referenced in WHERE clause
     */
    _extractWhereColumns(expr, leftAlias, rightAlias, leftCols, rightCols) {
        if (!expr) return;

        if (expr.type === 'column') {
            const table = expr.table;
            const column = expr.column;

            if (!table || table === leftAlias) {
                leftCols.add(column);
            } else if (table === rightAlias) {
                rightCols.add(column);
            }
        } else if (expr.type === 'binary') {
            this._extractWhereColumns(expr.left, leftAlias, rightAlias, leftCols, rightCols);
            this._extractWhereColumns(expr.right, leftAlias, rightAlias, leftCols, rightCols);
        } else if (expr.type === 'unary') {
            this._extractWhereColumns(expr.expr, leftAlias, rightAlias, leftCols, rightCols);
        }
    }

    /**
     * Analyze and separate filters by table (for pushdown)
     */
    _analyzeFilters(ast, context) {
        const { leftAlias, rightAlias } = context;

        const left = [];   // Filters that can be pushed to left table
        const right = [];  // Filters that can be pushed to right table
        const join = [];   // Filters that must be applied after join

        if (ast.where) {
            this._separateFilters(ast.where, leftAlias, rightAlias, left, right, join);
        }

        return { left, right, join };
    }

    /**
     * Separate WHERE filters by which table they reference
     */
    _separateFilters(expr, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters) {
        if (!expr) return;

        // For AND expressions, recursively separate each side
        if (expr.type === 'binary' && expr.op === 'AND') {
            this._separateFilters(expr.left, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters);
            this._separateFilters(expr.right, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters);
            return;
        }

        // For other expressions, check which table(s) they reference
        const tables = this._getReferencedTables(expr, leftAlias, rightAlias);

        if (tables.size === 1) {
            // Filter references only one table - can push down
            if (tables.has(leftAlias)) {
                leftFilters.push(expr);
            } else if (tables.has(rightAlias)) {
                rightFilters.push(expr);
            }
        } else if (tables.size > 1) {
            // Filter references multiple tables - must apply after join
            joinFilters.push(expr);
        }
        // If tables.size === 0, it's a constant expression (ignore for now)
    }

    /**
     * Get which tables an expression references
     */
    _getReferencedTables(expr, leftAlias, rightAlias) {
        const tables = new Set();

        const walk = (e) => {
            if (!e) return;

            if (e.type === 'column') {
                const table = e.table;
                if (!table) {
                    // Ambiguous - could be either table (conservative: don't push down)
                    tables.add(leftAlias);
                    tables.add(rightAlias);
                } else if (table === leftAlias) {
                    tables.add(leftAlias);
                } else if (table === rightAlias) {
                    tables.add(rightAlias);
                }
            } else if (e.type === 'binary') {
                walk(e.left);
                walk(e.right);
            } else if (e.type === 'unary') {
                walk(e.operand);  // Note: parser uses 'operand', not 'expr'
            } else if (e.type === 'call') {
                for (const arg of e.args || []) {
                    walk(arg);
                }
            } else if (e.type === 'in') {
                walk(e.expr);
                for (const v of e.values || []) walk(v);
            } else if (e.type === 'between') {
                walk(e.expr);
                walk(e.low);
                walk(e.high);
            } else if (e.type === 'like') {
                walk(e.expr);
                walk(e.pattern);
            }
        };

        walk(expr);
        return tables;
    }

    /**
     * Estimate how many rows to fetch from each table
     *
     * Need to over-fetch because:
     * - WHERE filters reduce rows
     * - JOIN reduces rows
     * - Want to ensure we get LIMIT rows after all filtering
     */
    _estimateFetchSize(ast, filterAnalysis) {
        const requestedLimit = ast.limit || 1000;

        // Estimate selectivity (how many rows pass filters)
        // This is a simple heuristic - could be improved with statistics
        const leftSelectivity = filterAnalysis.left.length > 0 ? 0.5 : 1.0;  // 50% if filtered
        const rightSelectivity = filterAnalysis.right.length > 0 ? 0.5 : 1.0;

        // Join selectivity (how many rows match)
        const joinSelectivity = 0.7;  // Assume 70% of left rows find a match

        // Over-fetch multiplier
        const safetyFactor = 2.5;

        // Estimate left fetch size
        // Want: requestedLimit = leftFetch * leftSelectivity * joinSelectivity
        // So: leftFetch = requestedLimit / (leftSelectivity * joinSelectivity) * safetyFactor
        const leftFetch = Math.ceil(
            requestedLimit / (leftSelectivity * joinSelectivity) * safetyFactor
        );

        // Right table: fetch only what's needed based on left join keys
        // This will be dynamic (added in executor)
        const rightFetch = null;  // Determined by left join keys

        return {
            left: Math.min(leftFetch, 10000),  // Cap at 10K for safety
            right: rightFetch
        };
    }

    /**
     * Log the query plan for debugging
     */
    _logPlan(plan, ast) {
        console.log('\n' + '='.repeat(60));
        console.log('📋 QUERY EXECUTION PLAN');
        console.log('='.repeat(60));

        console.log('\n🔍 Original Query:');
        console.log(`  SELECT: ${ast.columns.length} columns`);
        console.log(`  FROM: ${plan.leftScan.table} AS ${plan.leftScan.alias}`);
        console.log(`  JOIN: ${plan.rightScan.table} AS ${plan.rightScan.alias}`);
        console.log(`  WHERE: ${ast.where ? 'yes' : 'no'}`);
        console.log(`  LIMIT: ${ast.limit || 'none'}`);

        console.log('\n📊 Physical Plan:');

        console.log('\n  Step 1: SCAN LEFT TABLE');
        console.log(`    Table: ${plan.leftScan.table}`);
        console.log(`    Columns: [${plan.leftScan.columns.join(', ')}]`);
        console.log(`      - Join keys: [${[...plan.leftScan.purpose.join].join(', ')}]`);
        console.log(`      - Filter cols: [${[...plan.leftScan.purpose.where].join(', ')}]`);
        console.log(`      - Result cols: [${[...plan.leftScan.purpose.result].join(', ')}]`);
        console.log(`    Filters: ${plan.leftScan.filters.length} pushed down`);
        plan.leftScan.filters.forEach((f, i) => {
            console.log(`      ${i + 1}. ${this._formatFilter(f)}`);
        });
        console.log(`    Limit: ${plan.leftScan.limit} rows (over-fetch for safety)`);

        console.log('\n  Step 2: BUILD HASH TABLE');
        console.log(`    Index by: ${plan.join.leftKey}`);
        console.log(`    Keep: [${plan.leftScan.columns.join(', ')}]`);

        console.log('\n  Step 3: SCAN RIGHT TABLE');
        console.log(`    Table: ${plan.rightScan.table}`);
        console.log(`    Columns: [${plan.rightScan.columns.join(', ')}]`);
        console.log(`      - Join keys: [${[...plan.rightScan.purpose.join].join(', ')}]`);
        console.log(`      - Filter cols: [${[...plan.rightScan.purpose.where].join(', ')}]`);
        console.log(`      - Result cols: [${[...plan.rightScan.purpose.result].join(', ')}]`);
        console.log(`    Filters: ${plan.rightScan.filters.length} pushed down`);
        plan.rightScan.filters.forEach((f, i) => {
            console.log(`      ${i + 1}. ${this._formatFilter(f)}`);
        });
        console.log(`    Dynamic filter: ${plan.join.rightKey} IN (keys from left)`);

        console.log('\n  Step 4: HASH JOIN');
        console.log(`    Algorithm: ${plan.join.algorithm}`);
        console.log(`    Condition: ${plan.join.leftKey} = ${plan.join.rightKey}`);

        console.log('\n  Step 5: PROJECT');
        console.log(`    Result columns: ${plan.projection.length}`);
        plan.projection.forEach((col, i) => {
            if (col === '*') {
                console.log(`      ${i + 1}. *`);
            } else {
                console.log(`      ${i + 1}. ${col.table}.${col.column} AS ${col.alias}`);
            }
        });

        console.log('\n  Step 6: LIMIT');
        console.log(`    Rows: ${plan.limit || 'none'}`);

        console.log('\n💡 Optimization Summary:');
        const leftTotal = plan.leftScan.columns.length;
        const rightTotal = plan.rightScan.columns.length;
        console.log(`  - Fetch ${leftTotal} cols from left (not all columns)`);
        console.log(`  - Fetch ${rightTotal} cols from right (not all columns)`);
        console.log(`  - Push down ${plan.leftScan.filters.length + plan.rightScan.filters.length} filters`);
        console.log(`  - Over-fetch left by ${(plan.leftScan.limit / (ast.limit || 1)).toFixed(1)}x for safety`);

        console.log('\n' + '='.repeat(60) + '\n');
    }

    /**
     * Format filter expression for logging
     */
    _formatFilter(expr) {
        if (!expr) return 'null';

        if (expr.type === 'binary') {
            const left = this._formatFilter(expr.left);
            const right = this._formatFilter(expr.right);
            return `${left} ${expr.op} ${right}`;
        } else if (expr.type === 'column') {
            return expr.table ? `${expr.table}.${expr.column}` : expr.column;
        } else if (expr.type === 'literal') {
            return JSON.stringify(expr.value);
        } else if (expr.type === 'call') {
            const args = (expr.args || []).map(a => this._formatFilter(a)).join(', ');
            return `${expr.name}(${args})`;
        }

        return JSON.stringify(expr);
    }

    /**
     * Generate optimized plan for single-table queries (SELECT, aggregations).
     * This is the key optimization that makes us better than DuckDB for remote data:
     *
     * DuckDB approach (local-first):
     *   1. Load data into memory
     *   2. Build indexes
     *   3. Execute query
     *
     * LanceQL approach (remote-first):
     *   1. Analyze query to determine minimum columns needed
     *   2. Use Lance column statistics to skip entire chunks
     *   3. Stream only matching rows, never load full table
     *   4. Apply projections at the data source
     *
     * @param {Object} ast - Parsed SQL AST
     * @returns {Object} Physical execution plan
     */
    planSingleTable(ast) {
        const plan = {
            type: ast.type,
            // Phase 1: Determine columns to fetch
            scanColumns: [],
            // Phase 2: Filters to push down (executed at data source)
            pushedFilters: [],
            // Phase 3: Filters that must be evaluated after fetch
            postFilters: [],
            // Phase 4: Aggregations (if any)
            aggregations: [],
            // Phase 5: GROUP BY (if any)
            groupBy: [],
            // Phase 6: HAVING (if any)
            having: null,
            // Phase 7: ORDER BY (if any)
            orderBy: [],
            // Phase 8: LIMIT/OFFSET
            limit: ast.limit || null,
            offset: ast.offset || 0,
            // Phase 9: Final projection
            projection: [],
            // Optimization flags
            canUseStatistics: false,
            canStreamResults: true,
            estimatedSelectivity: 1.0,
        };

        // Analyze columns needed
        const neededColumns = new Set();

        // 1. Columns from SELECT
        if (ast.columns === '*' || (Array.isArray(ast.columns) && ast.columns.some(c => c.type === 'star'))) {
            plan.projection = ['*'];
            // For *, we can't prune columns - need all
            plan.canStreamResults = false;
        } else if (Array.isArray(ast.columns)) {
            for (const col of ast.columns) {
                this._collectColumnsFromSelectItem(col, neededColumns, plan);
            }
        }

        // 2. Columns from WHERE (for filter evaluation)
        if (ast.where) {
            this._collectColumnsFromExpr(ast.where, neededColumns);
            // Analyze filter for pushdown opportunities
            this._analyzeFilterPushdown(ast.where, plan);
        }

        // 3. Columns from GROUP BY
        if (ast.groupBy && ast.groupBy.length > 0) {
            for (const groupExpr of ast.groupBy) {
                this._collectColumnsFromExpr(groupExpr, neededColumns);
                plan.groupBy.push(groupExpr);
            }
        }

        // 4. Columns from HAVING
        if (ast.having) {
            this._collectColumnsFromExpr(ast.having, neededColumns);
            plan.having = ast.having;
        }

        // 5. Columns from ORDER BY
        if (ast.orderBy && ast.orderBy.length > 0) {
            for (const orderItem of ast.orderBy) {
                this._collectColumnsFromExpr(orderItem.expr || orderItem, neededColumns);
                plan.orderBy.push(orderItem);
            }
        }

        // Finalize scan columns
        plan.scanColumns = Array.from(neededColumns);

        // Calculate selectivity estimate based on filters
        plan.estimatedSelectivity = this._estimateSelectivity(plan.pushedFilters);

        // Determine if we can use column statistics to skip chunks
        plan.canUseStatistics = plan.pushedFilters.some(f =>
            f.type === 'range' || f.type === 'equality'
        );

        if (this.debug) {
            this._logSingleTablePlan(plan, ast);
        }

        return plan;
    }

    /**
     * Collect columns from a SELECT item
     */
    _collectColumnsFromSelectItem(item, columns, plan) {
        if (item.type === 'star') {
            plan.projection.push('*');
            return;
        }

        if (item.type === 'expr') {
            const expr = item.expr;

            // Check for aggregation
            if (expr.type === 'call') {
                const funcName = expr.name.toUpperCase();
                const aggFuncs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];

                if (aggFuncs.includes(funcName)) {
                    const agg = {
                        type: funcName,
                        column: null,
                        alias: item.alias || `${funcName}(${expr.args[0]?.name || '*'})`,
                        distinct: expr.distinct || false,
                    };

                    if (expr.args && expr.args.length > 0) {
                        const arg = expr.args[0];
                        if (arg.type === 'column') {
                            agg.column = arg.name || arg.column;
                            columns.add(agg.column);
                        } else if (arg.type !== 'star') {
                            this._collectColumnsFromExpr(arg, columns);
                        }
                    }

                    plan.aggregations.push(agg);
                    plan.projection.push({ type: 'aggregation', index: plan.aggregations.length - 1 });
                    return;
                }
            }

            // Regular column or expression
            this._collectColumnsFromExpr(expr, columns);
            plan.projection.push({
                type: 'column',
                expr: expr,
                alias: item.alias
            });
        }
    }

    /**
     * Collect column names from an expression
     */
    _collectColumnsFromExpr(expr, columns) {
        if (!expr) return;

        if (expr.type === 'column') {
            columns.add(expr.name || expr.column);
        } else if (expr.type === 'binary') {
            this._collectColumnsFromExpr(expr.left, columns);
            this._collectColumnsFromExpr(expr.right, columns);
        } else if (expr.type === 'call') {
            for (const arg of (expr.args || [])) {
                this._collectColumnsFromExpr(arg, columns);
            }
        } else if (expr.type === 'unary') {
            this._collectColumnsFromExpr(expr.operand, columns);
        }
    }

    /**
     * Analyze WHERE clause for filter pushdown opportunities.
     *
     * Pushable filters (can be evaluated at data source):
     * - Simple comparisons: col > 5, col = 'foo', col BETWEEN 1 AND 10
     * - IN clauses: col IN (1, 2, 3)
     * - LIKE patterns: col LIKE 'prefix%' (prefix only)
     *
     * Non-pushable filters (must evaluate after fetch):
     * - Complex expressions: col1 + col2 > 10
     * - Functions: UPPER(col) = 'FOO'
     * - Cross-column comparisons: col1 > col2
     */
    _analyzeFilterPushdown(expr, plan) {
        if (!expr) return;

        if (expr.type === 'binary') {
            // Check if this is a simple pushable condition
            if (this._isPushableFilter(expr)) {
                plan.pushedFilters.push(this._classifyFilter(expr));
            } else if (expr.op === 'AND') {
                // AND - recurse into both sides
                this._analyzeFilterPushdown(expr.left, plan);
                this._analyzeFilterPushdown(expr.right, plan);
            } else if (expr.op === 'OR') {
                // OR with pushable conditions on same column can be pushed
                const leftPushable = this._isPushableFilter(expr.left);
                const rightPushable = this._isPushableFilter(expr.right);

                if (leftPushable && rightPushable) {
                    plan.pushedFilters.push({
                        type: 'or',
                        left: this._classifyFilter(expr.left),
                        right: this._classifyFilter(expr.right),
                    });
                } else {
                    // Can't push OR with non-pushable condition
                    plan.postFilters.push(expr);
                }
            } else {
                // Non-pushable binary expression
                plan.postFilters.push(expr);
            }
        } else {
            plan.postFilters.push(expr);
        }
    }

    /**
     * Check if a filter can be pushed down to data source
     */
    _isPushableFilter(expr) {
        if (expr.type !== 'binary') return false;

        const compOps = ['=', '==', '!=', '<>', '<', '<=', '>', '>=', 'LIKE', 'IN', 'BETWEEN'];
        if (!compOps.includes(expr.op.toUpperCase())) return false;

        // One side must be a column, other must be a literal/constant
        const leftIsCol = expr.left.type === 'column';
        const rightIsCol = expr.right?.type === 'column';
        const leftIsLiteral = expr.left.type === 'literal' || expr.left.type === 'list';
        const rightIsLiteral = expr.right?.type === 'literal' || expr.right?.type === 'list';

        return (leftIsCol && rightIsLiteral) || (rightIsCol && leftIsLiteral);
    }

    /**
     * Classify a filter for optimization
     */
    _classifyFilter(expr) {
        const leftIsCol = expr.left.type === 'column';
        const column = leftIsCol
            ? (expr.left.name || expr.left.column)
            : (expr.right.name || expr.right.column);
        const value = leftIsCol ? expr.right.value : expr.left.value;

        const op = expr.op.toUpperCase();

        if (op === '=' || op === '==') {
            return { type: 'equality', column, value, op: '=' };
        } else if (op === '!=' || op === '<>') {
            return { type: 'inequality', column, value, op: '!=' };
        } else if (['<', '<=', '>', '>='].includes(op)) {
            return { type: 'range', column, value, op };
        } else if (op === 'LIKE') {
            return { type: 'like', column, pattern: value };
        } else if (op === 'IN') {
            const values = expr.right.type === 'list' ? expr.right.values : [expr.right.value];
            return { type: 'in', column, values };
        } else if (op === 'BETWEEN') {
            return { type: 'between', column, low: expr.right.low, high: expr.right.high };
        }

        return { type: 'unknown', expr };
    }

    /**
     * Estimate selectivity of filters (what % of rows will pass)
     */
    _estimateSelectivity(filters) {
        if (filters.length === 0) return 1.0;

        let selectivity = 1.0;
        for (const f of filters) {
            switch (f.type) {
                case 'equality':
                    selectivity *= 0.1; // Assume 10% match for equality
                    break;
                case 'range':
                    selectivity *= 0.3; // Assume 30% for range
                    break;
                case 'in':
                    selectivity *= Math.min(0.5, f.values.length * 0.05);
                    break;
                case 'like':
                    selectivity *= f.pattern.startsWith('%') ? 0.5 : 0.2;
                    break;
                default:
                    selectivity *= 0.5;
            }
        }
        return Math.max(0.01, selectivity); // At least 1%
    }

    /**
     * Log single-table query plan
     */
    _logSingleTablePlan(plan, ast) {
        console.log('\n' + '='.repeat(60));
        console.log('📋 SINGLE-TABLE QUERY PLAN');
        console.log('='.repeat(60));

        console.log('\n🔍 Query Analysis:');
        console.log(`  Type: ${plan.type}`);
        console.log(`  Aggregations: ${plan.aggregations.length}`);
        console.log(`  Group By: ${plan.groupBy.length} columns`);
        console.log(`  Order By: ${plan.orderBy.length} columns`);
        console.log(`  Limit: ${plan.limit || 'none'}`);

        console.log('\n📊 Scan Strategy:');
        console.log(`  Columns to fetch: [${plan.scanColumns.join(', ')}]`);
        console.log(`  Pushed filters: ${plan.pushedFilters.length}`);
        plan.pushedFilters.forEach((f, i) => {
            console.log(`    ${i + 1}. ${f.type}: ${f.column} ${f.op || ''} ${JSON.stringify(f.value || f.values || f.pattern || '')}`);
        });
        console.log(`  Post-fetch filters: ${plan.postFilters.length}`);

        console.log('\n💡 Optimizations:');
        console.log(`  Can use column statistics: ${plan.canUseStatistics}`);
        console.log(`  Can stream results: ${plan.canStreamResults}`);
        console.log(`  Estimated selectivity: ${(plan.estimatedSelectivity * 100).toFixed(1)}%`);

        console.log('\n' + '='.repeat(60) + '\n');
    }
}


export { StatisticsManager, CostModel, QueryPlanner };
