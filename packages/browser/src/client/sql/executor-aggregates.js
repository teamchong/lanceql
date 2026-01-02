/**
 * SQL Executor - Aggregation support (GROUP BY, ROLLUP, CUBE, GROUPING SETS)
 * Extracted from executor.js for modularity
 */

import { evaluateInMemoryExpr } from './executor-filters.js';
import { exprToName } from './executor-utils.js';

/** List of recognized aggregate function names */
export const AGG_FUNCS = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];

/**
 * Check if GROUP BY uses advanced operators (ROLLUP, CUBE, GROUPING SETS)
 * @param {Array} groupBy - GROUP BY clause items
 * @returns {boolean} - True if advanced grouping
 */
export function hasAdvancedGroupBy(groupBy) {
    if (!groupBy || groupBy.length === 0) return false;
    if (typeof groupBy[0] === 'string') return false;
    return groupBy.some(item =>
        item.type === 'ROLLUP' || item.type === 'CUBE' || item.type === 'GROUPING_SETS'
    );
}

/**
 * Get all column names from GROUP BY (for column ordering)
 * @param {Array} groupBy - GROUP BY clause items
 * @returns {Array<string>} - Column names
 */
export function getAllGroupColumns(groupBy) {
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
 * Generate power set (all subsets) of an array
 * @param {Array} arr - Input array
 * @returns {Array<Array>} - All subsets
 */
export function powerSet(arr) {
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
export function crossProductSets(sets1, sets2) {
    const result = [];
    for (const s1 of sets1) {
        for (const s2 of sets2) {
            result.push([...s1, ...s2]);
        }
    }
    return result;
}

/**
 * Expand GROUP BY clause into list of grouping sets
 * @param {Array} groupBy - GROUP BY clause items
 * @returns {Array<Array<string>>} - List of grouping sets
 */
export function expandGroupBy(groupBy) {
    if (!groupBy || groupBy.length === 0) return [[]];

    // Check if it's old-style simple column list (backward compat)
    if (typeof groupBy[0] === 'string') {
        return [groupBy];
    }

    let result = [[]];

    for (const item of groupBy) {
        if (item.type === 'COLUMN') {
            result = result.map(set => [...set, item.column]);
        } else if (item.type === 'ROLLUP') {
            // ROLLUP(a, b, c) generates: (a,b,c), (a,b), (a), ()
            const rollupSets = [];
            for (let i = item.columns.length; i >= 0; i--) {
                rollupSets.push(item.columns.slice(0, i));
            }
            result = crossProductSets(result, rollupSets);
        } else if (item.type === 'CUBE') {
            // CUBE(a, b) generates all 2^n subsets
            const cubeSets = powerSet(item.columns);
            result = crossProductSets(result, cubeSets);
        } else if (item.type === 'GROUPING_SETS') {
            result = crossProductSets(result, item.sets);
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
 * Compute an aggregate value for a set of row values
 * @param {string} funcName - Aggregate function name
 * @param {Array} values - Row values
 * @param {Object} options - Additional options (separator for STRING_AGG)
 * @returns {*} - Aggregated value
 */
export function computeAggregate(funcName, values, options = {}) {
    const nums = values.filter(v => v != null && typeof v === 'number' && !isNaN(v));

    switch (funcName.toUpperCase()) {
        case 'COUNT':
            return values.filter(v => v != null).length;
        case 'SUM':
            return nums.reduce((a, b) => a + b, 0);
        case 'AVG':
            return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
        case 'MIN':
            return nums.length > 0 ? Math.min(...nums) : null;
        case 'MAX':
            return nums.length > 0 ? Math.max(...nums) : null;
        case 'STDDEV':
        case 'STDDEV_SAMP': {
            if (nums.length < 2) return null;
            const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
            const variance = nums.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (nums.length - 1);
            return Math.sqrt(variance);
        }
        case 'STDDEV_POP': {
            if (nums.length === 0) return null;
            const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
            const variance = nums.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / nums.length;
            return Math.sqrt(variance);
        }
        case 'VARIANCE':
        case 'VAR_SAMP': {
            if (nums.length < 2) return null;
            const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
            return nums.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (nums.length - 1);
        }
        case 'VAR_POP': {
            if (nums.length === 0) return null;
            const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
            return nums.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / nums.length;
        }
        case 'MEDIAN': {
            if (nums.length === 0) return null;
            const sorted = [...nums].sort((a, b) => a - b);
            const mid = Math.floor(sorted.length / 2);
            return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        }
        case 'STRING_AGG':
        case 'GROUP_CONCAT': {
            const separator = options.separator ?? ',';
            return values.filter(v => v != null).map(String).join(separator);
        }
        default:
            return null;
    }
}

/**
 * Check if query has aggregate functions
 * @param {Object} ast - Parsed SQL AST
 * @returns {boolean} - True if aggregates found
 */
export function hasAggregates(ast) {
    for (const col of ast.columns) {
        if (col.type === 'expr' && col.expr.type === 'call') {
            if (AGG_FUNCS.includes(col.expr.name.toUpperCase())) {
                return true;
            }
        }
    }
    return false;
}

/**
 * Check if query is simple COUNT(*) without filters
 * @param {Object} ast - Parsed SQL AST
 * @returns {boolean} - True if simple count star
 */
export function isSimpleCountStar(ast) {
    if (ast.columns.length !== 1) return false;
    const col = ast.columns[0];
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
 * Return empty aggregate result with proper column names
 * @param {Object} ast - Parsed SQL AST
 * @returns {Object} - Empty result structure
 */
export function emptyAggregateResult(ast) {
    const colNames = ast.columns.map(col => col.alias || exprToName(col.expr || col));
    const emptyRow = ast.columns.map(col => {
        const expr = col.expr || col;
        if (expr.type === 'call' && expr.name.toUpperCase() === 'COUNT') {
            return 0;
        }
        return null;
    });

    // For GROUP BY with no matches, return empty rows
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

    // For simple aggregates, return single row
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
 * Execute GROUP BY aggregation for a single grouping set
 * @param {Object} ast - Parsed SQL AST
 * @param {Object} columnData - Column data lookup
 * @param {Array<number>} filteredIndices - Row indices
 * @param {Array<string>} groupingSet - Current grouping set columns
 * @param {Array<string>} allGroupColumns - All group columns
 * @returns {Array<Object>} - Result rows as objects
 */
export function executeGroupByForSet(ast, columnData, filteredIndices, groupingSet, allGroupColumns) {
    const groups = new Map();
    const groupingSetLower = groupingSet.map(c => c.toLowerCase());

    for (const idx of filteredIndices) {
        const groupKey = groupingSet.length > 0
            ? groupingSet.map(col => {
                const val = columnData[col.toLowerCase()]?.[idx];
                return JSON.stringify(val);
            }).join('|')
            : '__grand_total__';

        if (!groups.has(groupKey)) {
            const groupValues = {};
            for (const col of allGroupColumns) {
                if (groupingSetLower.includes(col.toLowerCase())) {
                    groupValues[col] = columnData[col.toLowerCase()]?.[idx];
                } else {
                    groupValues[col] = null;
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

    // Handle empty data
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

    // Compute aggregates
    const results = [];
    for (const [, group] of groups) {
        const row = { ...group.values, _groupingSet: group._groupingSet };

        for (const col of ast.columns) {
            const expr = col.expr;
            if (expr?.type === 'call' && AGG_FUNCS.includes((expr.name || '').toUpperCase())) {
                const funcName = expr.name.toUpperCase();
                const argExpr = expr.args?.[0];
                const isStar = argExpr?.type === 'star';
                const colName = (argExpr?.name || argExpr?.column || '').toLowerCase();
                const alias = col.alias || `${funcName}(${isStar ? '*' : colName})`;

                const indices = group.indices;
                let values = [];

                if (isStar) {
                    values = indices.map(() => 1); // COUNT(*)
                } else {
                    values = indices.map(i => columnData[colName]?.[i]).filter(v => v != null);
                }

                row[alias] = funcName === 'COUNT' && isStar
                    ? indices.length
                    : computeAggregate(funcName, values);
            }
        }
        results.push(row);
    }

    return results;
}

/**
 * Build final result from advanced GROUP BY results
 * @param {Object} ast - Parsed SQL AST
 * @param {Array<Object>} allResults - Result rows as objects
 * @param {Array<string>} allGroupColumns - All group columns
 * @returns {Object} - Query result { columns, rows, total }
 */
export function buildAdvancedGroupByResult(ast, allResults, allGroupColumns) {
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

    const resultRows = allResults.map(rowObj => {
        return resultColumns.map(colName => rowObj[colName] ?? rowObj[colName.toLowerCase()] ?? null);
    });

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

    return { columns: resultColumns, rows, total: allResults.length };
}

/**
 * Execute advanced GROUP BY with ROLLUP/CUBE/GROUPING SETS
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} ast - Parsed SQL AST
 * @param {Object} data - In-memory data
 * @param {Object} columnData - Column data lookup
 * @param {Array<number>} filteredIndices - Filtered row indices
 * @returns {Object} - Query result
 */
export function executeAdvancedGroupBy(executor, ast, data, columnData, filteredIndices) {
    const groupingSets = expandGroupBy(ast.groupBy);
    const allGroupColumns = getAllGroupColumns(ast.groupBy);

    const allResults = [];
    for (const groupingSet of groupingSets) {
        const setResults = executeGroupByForSet(
            ast, columnData, filteredIndices, groupingSet, allGroupColumns
        );
        allResults.push(...setResults);
    }

    return buildAdvancedGroupByResult(ast, allResults, allGroupColumns);
}

/**
 * Execute GROUP BY with aggregation on in-memory data
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} ast - Parsed SQL AST
 * @param {Object} data - In-memory data
 * @param {Object} columnData - Column data lookup
 * @param {Array<number>} filteredIndices - Filtered row indices
 * @returns {Object} - Query result { columns, rows, total }
 */
export function executeGroupByAggregation(executor, ast, data, columnData, filteredIndices) {
    const hasGroupByClause = ast.groupBy && ast.groupBy.length > 0;

    // Check for advanced GROUP BY
    if (hasGroupByClause && hasAdvancedGroupBy(ast.groupBy)) {
        return executeAdvancedGroupBy(executor, ast, data, columnData, filteredIndices);
    }

    // Group rows
    const groups = new Map();
    for (const idx of filteredIndices) {
        let groupKey = '';
        if (hasGroupByClause) {
            groupKey = ast.groupBy.map(expr => {
                const colName = (expr.column || expr.name || '').toLowerCase();
                return JSON.stringify(columnData[colName]?.[idx]);
            }).join('|');
        }

        if (!groups.has(groupKey)) {
            groups.set(groupKey, []);
        }
        groups.get(groupKey).push(idx);
    }

    // Handle empty result set with no groups
    if (!hasGroupByClause && groups.size === 0) {
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

    // Compute aggregations
    const resultRows = [];
    for (const [, groupIndices] of groups) {
        const row = [];
        for (const col of ast.columns) {
            const expr = col.expr;
            if (expr?.type === 'call' && AGG_FUNCS.includes((expr.name || '').toUpperCase())) {
                const funcName = expr.name.toUpperCase();
                const argExpr = expr.args?.[0];
                const isStar = argExpr?.type === 'star';
                const colName = (argExpr?.name || argExpr?.column || '').toLowerCase();

                if (funcName === 'COUNT' && isStar) {
                    row.push(groupIndices.length);
                } else {
                    const values = groupIndices.map(i => columnData[colName]?.[i]);
                    const separator = expr.args?.[1]?.value;
                    row.push(computeAggregate(funcName, values, { separator }));
                }
            } else if (expr?.type === 'column') {
                const colName = (expr.name || expr.column || '').toLowerCase();
                row.push(columnData[colName]?.[groupIndices[0]] ?? null);
            } else {
                row.push(evaluateInMemoryExpr(expr, columnData, groupIndices[0]));
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
