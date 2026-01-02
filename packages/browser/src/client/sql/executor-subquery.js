/**
 * SQL Executor - CTE and subquery support
 * Extracted from executor.js for modularity
 */

import { evaluateInMemoryExpr } from './executor-filters.js';

/**
 * Execute a subquery and return its result
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} subqueryAst - Subquery AST
 * @param {Object} outerColumnData - Outer query column data
 * @param {number} outerRowIdx - Outer query row index
 * @returns {*} - Scalar result from subquery
 */
export function executeSubquery(executor, subqueryAst, outerColumnData, outerRowIdx) {
    // Clone the subquery AST to avoid mutating the original
    // structuredClone is faster than JSON.parse(JSON.stringify())
    const resolvedAst = structuredClone(subqueryAst);

    // Check for correlated references
    const subqueryTable = resolvedAst.from?.name || resolvedAst.from?.table;
    const correlatedColumns = findCorrelatedColumns(resolvedAst, subqueryTable);

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
        substituteCorrelations(resolvedAst.where, correlationContext);
    }

    // Check if FROM references a CTE
    const tableName = resolvedAst.from?.name?.toLowerCase() || resolvedAst.from?.table?.toLowerCase();
    if (tableName && executor._cteResults?.has(tableName)) {
        const result = executeOnInMemoryData(executor, resolvedAst, executor._cteResults.get(tableName));
        return result.rows.length > 0 ? result.rows[0][0] : null;
    }

    // Execute against database if available
    if (executor._database) {
        try {
            const result = executor._database._executeSingleTable(resolvedAst);
            if (result && result.then) {
                return null; // Async subquery in expression context not supported
            }
            return result?.rows?.[0]?.[0] ?? null;
        } catch {
            return null;
        }
    }

    return null; // Requires LanceDatabase context
}

/**
 * Find columns that reference the outer query (correlated columns)
 * @param {Object} ast - Query AST
 * @param {string} subqueryTable - Subquery's FROM table
 * @returns {Array} - Array of correlated column references
 */
export function findCorrelatedColumns(ast, subqueryTable) {
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
 * @param {Object} expr - Expression AST (mutated)
 * @param {Object} correlationContext - Map of column refs to values
 */
export function substituteCorrelations(expr, correlationContext) {
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
        substituteCorrelations(expr.left, correlationContext);
        substituteCorrelations(expr.right, correlationContext);
    } else if (expr.type === 'unary') {
        substituteCorrelations(expr.operand, correlationContext);
    } else if (expr.type === 'in') {
        substituteCorrelations(expr.expr, correlationContext);
        expr.values?.forEach(v => substituteCorrelations(v, correlationContext));
    } else if (expr.type === 'between') {
        substituteCorrelations(expr.expr, correlationContext);
        substituteCorrelations(expr.low, correlationContext);
        substituteCorrelations(expr.high, correlationContext);
    } else if (expr.type === 'like') {
        substituteCorrelations(expr.expr, correlationContext);
        substituteCorrelations(expr.pattern, correlationContext);
    } else if (expr.type === 'call') {
        expr.args?.forEach(a => substituteCorrelations(a, correlationContext));
    }
}

/**
 * Materialize CTEs before query execution
 * @param {Object} executor - SQLExecutor instance
 * @param {Array} ctes - Array of CTE definitions
 * @param {Object} db - Database reference
 */
export async function materializeCTEs(executor, ctes, db) {
    executor._database = db;
    for (const cte of ctes) {
        const cteName = cte.name.toLowerCase();
        if (cte.body.type === 'RECURSIVE_CTE') {
            // Execute anchor query first
            const anchorResult = await executeCTEBody(executor, cte.body.anchor, db);
            let result = { columns: anchorResult.columns, rows: [...anchorResult.rows] };

            // Iterate recursive part until no new rows (max 1000 iterations)
            for (let i = 0; i < 1000; i++) {
                executor._cteResults.set(cteName, result);
                const recursiveResult = await executeCTEBody(executor, cte.body.recursive, db);
                if (recursiveResult.rows.length === 0) break;
                result = { columns: result.columns, rows: [...result.rows, ...recursiveResult.rows] };
            }
            executor._cteResults.set(cteName, result);
        } else {
            const result = await executeCTEBody(executor, cte.body, db);
            executor._cteResults.set(cteName, result);
        }
    }
}

/**
 * Execute a CTE body
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} bodyAst - CTE body AST
 * @param {Object} db - Database reference
 * @returns {Promise<Object>} - Query result
 */
export async function executeCTEBody(executor, bodyAst, db) {
    // Check if FROM references another CTE
    const tableName = bodyAst.from?.name?.toLowerCase() || bodyAst.from?.table?.toLowerCase();
    if (tableName && executor._cteResults.has(tableName)) {
        return executeOnInMemoryData(executor, bodyAst, executor._cteResults.get(tableName));
    }
    // Fall back to database execution
    return db._executeSingleTable(bodyAst);
}

/**
 * Check if SELECT columns contain aggregate functions
 * @param {Array} columns - SELECT columns
 * @returns {boolean} - True if aggregates found
 */
export function hasAggregatesInSelect(columns) {
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
 * Execute query on in-memory data (for CTEs and subqueries)
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} ast - Parsed SELECT AST
 * @param {Object} data - In-memory data { columns, rows }
 * @returns {Object} - Query result { columns, rows, total }
 */
export function executeOnInMemoryData(executor, ast, data) {
    // Build column lookup: column name -> array of values
    const columnData = {};
    for (let i = 0; i < data.columns.length; i++) {
        const colName = data.columns[i].toLowerCase();
        columnData[colName] = data.rows.map(row => row[i]);
    }

    // Apply WHERE filter
    const filteredIndices = [];
    for (let i = 0; i < data.rows.length; i++) {
        if (!ast.where || evaluateInMemoryExpr(ast.where, columnData, i)) {
            filteredIndices.push(i);
        }
    }

    // Check for GROUP BY or aggregations
    const hasGroupBy = ast.groupBy && ast.groupBy.length > 0;
    const hasAggregates = hasAggregatesInSelect(ast.columns);

    if (hasGroupBy || hasAggregates) {
        return executor._executeGroupByAggregation(ast, data, columnData, filteredIndices);
    }

    // Check for window functions
    if (executor.hasWindowFunctions(ast)) {
        return executor._executeWindowFunctions(ast, data, columnData, filteredIndices);
    }

    // Project columns and build result
    const resultColumns = [];
    const resultRows = [];

    // Handle SELECT *
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
                return evaluateInMemoryExpr(col.expr, columnData, idx);
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
