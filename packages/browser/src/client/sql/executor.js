/**
 * SQLExecutor - SQL query execution engine
 * Main class that coordinates extracted modules for query execution.
 */

import { SQLLexer } from './lexer.js';
import { SQLParser } from './parser.js';
import { QueryPlanner, statisticsManager } from './query-planner.js';
import { getGPUSorter } from '../gpu/sorter.js';

// Import extracted modules
import * as SearchModule from './executor-search.js';
import * as WindowModule from './executor-window.js';
import * as FilterModule from './executor-filters.js';
import * as AggModule from './executor-aggregates.js';
import * as SubqueryModule from './executor-subquery.js';
import * as UtilsModule from './executor-utils.js';

class SQLExecutor {
    constructor(file, options = {}) {
        this.file = file;
        this.columnMap = {};
        this.columnTypes = [];
        this._cteResults = new Map();
        this._database = null;
        this._ftsIndexCache = null;
        this._debug = options.debug ?? false;

        // Build column name -> index map
        if (file.columnNames) {
            file.columnNames.forEach((name, idx) => {
                this.columnMap[name.toLowerCase()] = idx;
            });
        }
    }

    setDatabase(db) {
        this._database = db;
    }

    /**
     * Execute a SQL query
     */
    async execute(sql, onProgress = null) {
        const lexer = new SQLLexer(sql);
        const tokens = lexer.tokenize();
        const parser = new SQLParser(tokens);
        const ast = parser.parse();

        const planner = new QueryPlanner();
        const plan = planner.planSingleTable(ast);

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

        const totalRows = this.file._isRemote
            ? await this.file.getRowCount(0)
            : Number(this.file.getRowCount(0));

        // Statistics-based optimization
        let columnStats = null;
        if (ast.where && plan.pushedFilters.length > 0 && this.file._isRemote) {
            columnStats = await statisticsManager.precomputeForPlan(this.file, plan);
            plan.columnStats = Object.fromEntries(columnStats);
        }

        const neededColumns = plan.scanColumns.length > 0
            ? plan.scanColumns
            : this.collectNeededColumns(ast);
        const outputColumns = this.resolveOutputColumns(ast);

        // Check for aggregation
        const hasAggregates = plan.aggregations.length > 0 || AggModule.hasAggregates(ast);
        if (hasAggregates) {
            if (AggModule.isSimpleCountStar(ast) && !ast.where && !ast.search) {
                return {
                    columns: ['COUNT(*)'],
                    rows: [[totalRows]],
                    total: 1,
                    aggregationStats: { scannedRows: 0, totalRows, coveragePercent: '100.00', isPartialScan: false, fromMetadata: true },
                    queryPlan: plan,
                };
            }
            if (ast.search || SearchModule.extractNearCondition(ast.where)) {
                return await SearchModule.executeAggregateWithSearch(this, ast, totalRows, onProgress);
            }
            return await this.executeAggregateQuery(ast, totalRows, onProgress);
        }

        // Calculate indices
        let indices;
        const limit = ast.limit || 100;
        const offset = ast.offset || 0;

        if (!ast.where) {
            indices = [];
            const endIdx = Math.min(offset + limit, totalRows);
            for (let i = offset; i < endIdx; i++) indices.push(i);
        } else {
            indices = await this.evaluateWhere(ast.where, totalRows, onProgress);
            indices = indices.slice(offset, offset + limit);
        }

        if (onProgress) onProgress('Fetching column data...', 0, outputColumns.length);

        // Fetch column data
        const columnData = {};
        for (let i = 0; i < neededColumns.length; i++) {
            const colName = neededColumns[i];
            const colIdx = this.columnMap[colName.toLowerCase()];
            if (colIdx === undefined) continue;
            if (onProgress) onProgress(`Fetching ${colName}...`, i, neededColumns.length);
            columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, indices);
        }

        // Build result rows
        const rows = [];
        for (let i = 0; i < indices.length; i++) {
            const row = [];
            for (const col of outputColumns) {
                if (col.type === 'star') {
                    for (const name of this.file.columnNames || []) {
                        row.push(columnData[name.toLowerCase()]?.[i] ?? null);
                    }
                } else {
                    row.push(this.evaluateExpr(col.expr, columnData, i));
                }
            }
            rows.push(row);
        }

        // Apply transformations
        if (ast.pivot) {
            const pivotResult = UtilsModule.executePivot(rows, outputColumns, ast.pivot);
            rows.length = 0;
            rows.push(...pivotResult.rows);
            outputColumns.length = 0;
            outputColumns.push(...pivotResult.columns);
        }

        if (ast.unpivot) {
            const unpivotResult = UtilsModule.executeUnpivot(rows, outputColumns, ast.unpivot);
            rows.length = 0;
            rows.push(...unpivotResult.rows);
            outputColumns.length = 0;
            outputColumns.push(...unpivotResult.columns);
        }

        if (ast.distinct) {
            const uniqueRows = await UtilsModule.applyDistinct(this, rows, gpuSorter);
            rows.length = 0;
            rows.push(...uniqueRows);
        }

        if (ast.orderBy && ast.orderBy.length > 0) {
            await UtilsModule.applyOrderBy(this, rows, ast.orderBy, outputColumns, gpuSorter);
        }

        // Build column names
        const colNames = [];
        for (const col of outputColumns) {
            if (col.type === 'star') {
                colNames.push(...(this.file.columnNames || []));
            } else {
                colNames.push(col.alias || UtilsModule.exprToName(col.expr));
            }
        }

        return {
            columns: colNames,
            rows,
            total: ast.limit ? rows.length : totalRows,
            orderByOnSubset: ast.orderBy && ast.orderBy.length > 0 && rows.length < totalRows,
            orderByColumns: ast.orderBy ? ast.orderBy.map(ob => `${ob.column} ${ob.direction}`) : [],
            queryPlan: plan,
            optimization: {
                statsComputed: columnStats?.size > 0,
                columnStats: columnStats ? Object.fromEntries(columnStats) : null,
                pushedFilters: plan.pushedFilters?.length || 0,
                estimatedSelectivity: plan.estimatedSelectivity,
            },
        };
    }

    // === Column Collection ===
    collectNeededColumns(ast) {
        const columns = new Set();
        for (const item of ast.columns) {
            if (item.type === 'star') {
                (this.file.columnNames || []).forEach(n => columns.add(n.toLowerCase()));
            } else {
                FilterModule.collectColumnsFromExpr(item.expr, columns);
            }
        }
        if (ast.where) FilterModule.collectColumnsFromExpr(ast.where, columns);
        for (const ob of ast.orderBy || []) columns.add(ob.column.toLowerCase());
        return Array.from(columns);
    }

    collectColumnsFromExpr(expr, columns) {
        FilterModule.collectColumnsFromExpr(expr, columns);
    }

    resolveOutputColumns(ast) {
        return ast.columns;
    }

    // === Data Reading ===
    async readColumnData(colIdx, indices) {
        const type = this.columnTypes[colIdx] || 'unknown';
        try {
            if (type === 'string') {
                const data = await this.file.readStringsAtIndices(colIdx, indices);
                return Array.isArray(data) ? data : Array.from(data);
            } else if (type === 'int64') {
                const data = await this.file.readInt64AtIndices(colIdx, indices);
                return Array.from(data, v => Number(v));
            } else if (type === 'float64') {
                return Array.from(await this.file.readFloat64AtIndices(colIdx, indices));
            } else if (type === 'int32') {
                return Array.from(await this.file.readInt32AtIndices(colIdx, indices));
            } else if (type === 'float32') {
                return Array.from(await this.file.readFloat32AtIndices(colIdx, indices));
            } else if (type === 'vector') {
                return indices.map(() => '[vector]');
            } else {
                try {
                    return await this.file.readStringsAtIndices(colIdx, indices);
                } catch (e) {
                    if (this._debug) console.warn(`[SQLExecutor] readColumnData col ${colIdx} fallback failed:`, e.message);
                    return indices.map(() => null);
                }
            }
        } catch (e) {
            if (this._debug) console.warn(`[SQLExecutor] readColumnData col ${colIdx} failed:`, e.message);
            return indices.map(() => null);
        }
    }

    // === WHERE Evaluation ===
    async evaluateWhere(whereExpr, totalRows, onProgress) {
        const nearInfo = SearchModule.extractNearCondition(whereExpr);
        if (nearInfo) {
            return await SearchModule.evaluateWithNear(this, nearInfo, whereExpr, totalRows, onProgress);
        }

        const simpleFilter = FilterModule.detectSimpleFilter(whereExpr, this.columnMap, this.columnTypes);
        if (simpleFilter) {
            return await FilterModule.evaluateSimpleFilter(this, simpleFilter, totalRows, onProgress);
        }

        return await FilterModule.evaluateComplexFilter(this, whereExpr, totalRows, onProgress);
    }

    evaluateExpr(expr, columnData, rowIdx) {
        return FilterModule.evaluateExpr(this, expr, columnData, rowIdx);
    }

    // === Subquery/CTE Support ===
    _executeSubquery(subqueryAst, outerColumnData, outerRowIdx) {
        return SubqueryModule.executeSubquery(this, subqueryAst, outerColumnData, outerRowIdx);
    }

    _executeCTEBody(bodyAst, db) {
        return SubqueryModule.executeCTEBody(this, bodyAst, db);
    }

    async materializeCTEs(ctes, db) {
        return SubqueryModule.materializeCTEs(this, ctes, db);
    }

    _executeOnInMemoryData(ast, data) {
        return SubqueryModule.executeOnInMemoryData(this, ast, data);
    }

    _evaluateInMemoryExpr(expr, columnData, rowIdx) {
        return FilterModule.evaluateInMemoryExpr(expr, columnData, rowIdx);
    }

    // === Window Functions ===
    hasWindowFunctions(ast) {
        return WindowModule.hasWindowFunctions(ast);
    }

    computeWindowFunctions(ast, rows, columnData) {
        return WindowModule.computeWindowFunctions(ast, rows, columnData,
            (expr, colData, idx) => this._evaluateInMemoryExpr(expr, colData, idx));
    }

    _executeWindowFunctions(ast, data, columnData, filteredIndices) {
        return WindowModule.executeWindowFunctions(this, ast, data, columnData, filteredIndices);
    }

    _partitionRows(rows, partitionBy, columnData, evaluateExpr) {
        return WindowModule.partitionRows(rows, partitionBy, columnData, evaluateExpr);
    }

    _compareRowsByOrder(a, b, orderBy, columnData, evaluateExpr) {
        return WindowModule.compareRowsByOrder(a, b, orderBy, columnData, evaluateExpr);
    }

    // === Aggregation ===
    hasAggregates(ast) {
        return AggModule.hasAggregates(ast);
    }

    _executeGroupByAggregation(ast, data, columnData, filteredIndices) {
        return AggModule.executeGroupByAggregation(this, ast, data, columnData, filteredIndices);
    }

    _hasAdvancedGroupBy(groupBy) {
        return AggModule.hasAdvancedGroupBy(groupBy);
    }

    _emptyAggregateResult(ast) {
        return AggModule.emptyAggregateResult(ast);
    }

    async executeAggregateQuery(ast, totalRows, onProgress) {
        // This method is kept inline due to its complexity and file-specific logic
        const aggFunctions = AggModule.AGG_FUNCS;
        const aggregators = [];
        const colNames = [];

        for (const col of ast.columns) {
            if (col.type === 'star') {
                aggregators.push({ type: 'COUNT', column: null, isStar: true });
                colNames.push('COUNT(*)');
            } else if (col.expr.type === 'call' && aggFunctions.includes(col.expr.name.toUpperCase())) {
                const aggType = col.expr.name.toUpperCase();
                const argExpr = col.expr.args[0];
                aggregators.push({
                    type: aggType,
                    column: argExpr?.type === 'column' ? argExpr.name : null,
                    isStar: argExpr?.type === 'star',
                    sum: 0, count: 0, min: null, max: null, values: []
                });
                colNames.push(col.alias || UtilsModule.exprToName(col.expr));
            } else {
                aggregators.push({
                    type: 'FIRST',
                    column: col.expr.type === 'column' ? col.expr.name : null,
                    value: null
                });
                colNames.push(col.alias || UtilsModule.exprToName(col.expr));
            }
        }

        const neededCols = new Set();
        for (const agg of aggregators) {
            if (agg.column) neededCols.add(agg.column.toLowerCase());
        }
        if (ast.where) this.collectColumnsFromExpr(ast.where, neededCols);

        const maxRowsToScan = ast.limit ? Math.min(ast.limit, totalRows) : totalRows;
        const batchSize = 1000;
        let scannedRows = 0;

        for (let batchStart = 0; batchStart < maxRowsToScan; batchStart += batchSize) {
            if (onProgress) onProgress(`Aggregating...`, batchStart, maxRowsToScan);

            const batchEnd = Math.min(batchStart + batchSize, maxRowsToScan);
            const batchIndices = Array.from({ length: batchEnd - batchStart }, (_, i) => batchStart + i);
            scannedRows += batchIndices.length;

            const batchData = {};
            for (const colName of neededCols) {
                const colIdx = this.columnMap[colName.toLowerCase()];
                if (colIdx !== undefined) {
                    batchData[colName.toLowerCase()] = await this.readColumnData(colIdx, batchIndices);
                }
            }

            for (let i = 0; i < batchIndices.length; i++) {
                if (ast.where && !this.evaluateExpr(ast.where, batchData, i)) continue;

                for (const agg of aggregators) {
                    if (agg.type === 'COUNT') {
                        agg.count++;
                    } else if (agg.type === 'FIRST' && agg.value === null && agg.column) {
                        agg.value = batchData[agg.column.toLowerCase()]?.[i];
                    } else if (agg.column) {
                        const val = batchData[agg.column.toLowerCase()]?.[i];
                        if (val != null && !isNaN(val)) {
                            agg.count++;
                            if (agg.type === 'SUM' || agg.type === 'AVG') agg.sum += val;
                            if (agg.type === 'MIN') agg.min = agg.min === null ? val : Math.min(agg.min, val);
                            if (agg.type === 'MAX') agg.max = agg.max === null ? val : Math.max(agg.max, val);
                        }
                    }
                }
            }
        }

        const resultRow = aggregators.map(agg => {
            switch (agg.type) {
                case 'COUNT': return agg.count;
                case 'SUM': return agg.sum;
                case 'AVG': return agg.count > 0 ? agg.sum / agg.count : null;
                case 'MIN': return agg.min;
                case 'MAX': return agg.max;
                case 'FIRST': return agg.value;
                default: return null;
            }
        });

        if (ast.having) {
            const havingData = {};
            colNames.forEach((name, i) => { havingData[name.toLowerCase()] = [resultRow[i]]; });
            if (!this._evaluateInMemoryExpr(ast.having, havingData, 0)) {
                return { columns: colNames, rows: [], total: 0, aggregationStats: { scannedRows, totalRows } };
            }
        }

        return {
            columns: colNames,
            rows: [resultRow],
            total: 1,
            aggregationStats: {
                scannedRows,
                totalRows,
                coveragePercent: totalRows > 0 ? ((scannedRows / totalRows) * 100).toFixed(2) : '100.00',
                isPartialScan: scannedRows < totalRows
            }
        };
    }

    async _executeSimpleAggregateOnIndices(ast, indices, onProgress) {
        // Simplified aggregate on specific indices (used by search)
        const colNames = ast.columns.map(col => col.alias || UtilsModule.exprToName(col.expr || col));
        const neededCols = new Set();
        for (const col of ast.columns) {
            if (col.expr?.args?.[0]?.type === 'column') {
                neededCols.add((col.expr.args[0].name || col.expr.args[0].column).toLowerCase());
            }
        }

        const columnData = {};
        for (const colName of neededCols) {
            const colIdx = this.columnMap[colName];
            if (colIdx !== undefined) {
                columnData[colName] = await this.readColumnData(colIdx, indices);
            }
        }

        const resultRow = ast.columns.map(col => {
            if (col.expr?.type === 'call') {
                const funcName = col.expr.name.toUpperCase();
                const argExpr = col.expr.args?.[0];
                const isStar = argExpr?.type === 'star';
                if (funcName === 'COUNT' && isStar) return indices.length;
                const colName = (argExpr?.name || argExpr?.column)?.toLowerCase();
                const values = indices.map((_, i) => columnData[colName]?.[i]);
                return AggModule.computeAggregate(funcName, values);
            }
            return null;
        });

        return { columns: colNames, rows: [resultRow], total: 1 };
    }

    async _executeGroupByOnIndices(ast, indices, onProgress) {
        const neededCols = new Set();
        for (const expr of ast.groupBy) {
            neededCols.add((expr.column || expr.name).toLowerCase());
        }
        for (const col of ast.columns) {
            if (col.expr?.type === 'column') neededCols.add((col.expr.name || col.expr.column).toLowerCase());
            if (col.expr?.args?.[0]?.type === 'column') neededCols.add((col.expr.args[0].name || col.expr.args[0].column).toLowerCase());
        }

        const columnData = {};
        for (const colName of neededCols) {
            const colIdx = this.columnMap[colName];
            if (colIdx !== undefined) {
                columnData[colName] = await this.readColumnData(colIdx, indices);
            }
        }

        const groups = new Map();
        for (let i = 0; i < indices.length; i++) {
            const key = ast.groupBy.map(expr => JSON.stringify(columnData[(expr.column || expr.name).toLowerCase()]?.[i])).join('|');
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(i);
        }

        const colNames = ast.columns.map(col => col.alias || UtilsModule.exprToName(col.expr || col));
        const resultRows = [];

        for (const [, groupIndices] of groups) {
            const row = ast.columns.map(col => {
                const expr = col.expr || col;
                if (expr.type === 'call' && AggModule.AGG_FUNCS.includes(expr.name.toUpperCase())) {
                    const colName = (expr.args?.[0]?.name || expr.args?.[0]?.column)?.toLowerCase();
                    const isStar = expr.args?.[0]?.type === 'star';
                    if (expr.name.toUpperCase() === 'COUNT' && isStar) return groupIndices.length;
                    const values = groupIndices.map(i => columnData[colName]?.[i]);
                    return AggModule.computeAggregate(expr.name.toUpperCase(), values);
                } else if (expr.type === 'column') {
                    return columnData[(expr.name || expr.column).toLowerCase()]?.[groupIndices[0]];
                }
                return null;
            });
            resultRows.push(row);
        }

        return { columns: colNames, rows: resultRows, total: resultRows.length };
    }

    // === Utility Methods ===
    exprToName(expr) {
        return UtilsModule.exprToName(expr);
    }

    isSimpleCountStar(ast) {
        return AggModule.isSimpleCountStar(ast);
    }

    async applyOrderBy(rows, orderBy, outputColumns) {
        return UtilsModule.applyOrderBy(this, rows, orderBy, outputColumns, gpuSorter);
    }

    async applyDistinct(rows) {
        return UtilsModule.applyDistinct(this, rows, gpuSorter);
    }

    // === Streaming ===
    async *executeStream(sql, options = {}) {
        const { chunkSize = 1000 } = options;
        const lexer = new SQLLexer(sql);
        const tokens = lexer.tokenize();
        const parser = new SQLParser(tokens);
        const ast = parser.parse();

        if (this.columnTypes.length === 0) {
            if (this.file._isRemote && this.file.detectColumnTypes) {
                this.columnTypes = await this.file.detectColumnTypes();
            } else if (this.file._columnTypes) {
                this.columnTypes = this.file._columnTypes;
            } else {
                this.columnTypes = Array(this.file.numColumns || 0).fill('unknown');
            }
        }

        const totalRows = this.file._isRemote ? await this.file.getRowCount(0) : Number(this.file.getRowCount(0));
        const neededColumns = this.collectNeededColumns(ast);
        const limit = ast.limit || totalRows;
        let yielded = 0;

        for (let offset = 0; offset < totalRows && yielded < limit; offset += chunkSize) {
            const batchSize = Math.min(chunkSize, limit - yielded, totalRows - offset);
            const indices = Array.from({ length: batchSize }, (_, i) => offset + i);

            const columnData = [];
            for (const colName of neededColumns) {
                const colIdx = this.columnMap[colName.toLowerCase()];
                columnData.push(colIdx !== undefined ? await this.readColumnData(colIdx, indices) : indices.map(() => null));
            }

            const rows = indices.map((_, i) => neededColumns.map((_, c) => columnData[c][i]));
            let filteredRows = rows;

            if (ast.where) {
                filteredRows = rows.filter((row) =>
                    UtilsModule.evaluateWhereExprOnRow(ast.where, neededColumns, row, UtilsModule.getValueFromExpr)
                );
            }

            if (filteredRows.length > 0) {
                yield { columns: neededColumns, rows: filteredRows };
                yielded += filteredRows.length;
            }
        }
    }
}

export { SQLExecutor };
