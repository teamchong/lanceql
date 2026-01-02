/**
 * LanceDatabase - Join operations
 * Extracted from lance-database.js for modularity
 */

import { SQLLexer } from '../sql/lexer.js';
import { SQLParser } from '../sql/parser.js';
import { SQLExecutor } from '../sql/executor.js';
import { QueryPlanner } from '../sql/query-planner.js';
import { OPFSJoinExecutor } from './local-database.js';
import { opfsStorage } from '../storage/opfs.js';

/**
 * Execute multi-table query with JOINs
 * @param {LanceDatabase} db - Database instance
 * @param {Object} ast - Parsed AST
 * @returns {Promise<Object>} Query result
 */
export async function executeJoin(db, ast) {
    console.log('[LanceDatabase] Executing JOIN query:', ast);

    // Extract table references
    const leftTableName = ast.from.name || ast.from.table;
    const leftAlias = ast.from.alias || leftTableName;

    // Register alias
    if (ast.from.alias) {
        db.aliases.set(ast.from.alias, leftTableName);
    }

    // Process joins iteratively: (A JOIN B) JOIN C
    console.log(`[LanceDatabase] Processing ${ast.joins.length} JOIN(s)`);

    let currentResult = null;
    let currentAlias = leftAlias;
    let currentTableName = leftTableName;
    let leftDataset = db.getTable(leftTableName);

    for (let i = 0; i < ast.joins.length; i++) {
        const join = ast.joins[i];
        const rightTableName = join.table.name || join.table.table;
        const rightAlias = join.alias || rightTableName;

        // Register right table alias
        if (join.alias) {
            db.aliases.set(join.alias, rightTableName);
        }

        console.log(`[LanceDatabase] JOIN ${i + 1}/${ast.joins.length}: ${currentTableName} (${currentAlias}) ${join.type} ${rightTableName} (${rightAlias})`);

        // Get right dataset
        const rightDataset = db.getTable(rightTableName);

        // Build AST for this single join
        const singleJoinAst = {
            ...ast,
            joins: [join],
            limit: (i === ast.joins.length - 1) ? ast.limit : undefined,
            columns: (i === ast.joins.length - 1) ? ast.columns : [{ type: 'column', column: '*' }]
        };

        // Execute hash join
        if (currentResult === null) {
            currentResult = await hashJoin(
                db,
                leftDataset,
                rightDataset,
                singleJoinAst,
                { leftAlias: currentAlias, rightAlias, leftTableName: currentTableName, rightTableName }
            );
        } else {
            currentResult = await hashJoinWithInMemoryLeft(
                db,
                currentResult,
                rightDataset,
                singleJoinAst,
                { leftAlias: currentAlias, rightAlias, leftTableName: currentTableName, rightTableName }
            );
        }

        currentAlias = `${currentAlias}_${rightAlias}`;
        currentTableName = `(${currentTableName} JOIN ${rightTableName})`;
    }

    return currentResult;
}

/**
 * Execute hash join between two datasets using OPFS for intermediate storage.
 */
export async function hashJoin(db, leftDataset, rightDataset, ast, context) {
    const { leftAlias, rightAlias, leftTableName, rightTableName } = context;
    const join = ast.joins[0];
    const joinType = join.type || 'INNER';

    const joinCondition = join.on;
    if (joinType !== 'CROSS') {
        if (!joinCondition || joinCondition.type !== 'binary' || joinCondition.op !== '=') {
            throw new Error('JOIN ON condition must be an equality (e.g., table1.col1 = table2.col2)');
        }
    }

    let leftKey, rightKey, leftSQL, rightSQL, plan;

    if (joinType === 'CROSS') {
        leftKey = null;
        rightKey = null;
        leftSQL = `SELECT * FROM ${leftTableName}`;
        rightSQL = `SELECT * FROM ${rightTableName}`;
        console.log('[LanceDatabase] CROSS JOIN - no keys, cartesian product');
    } else {
        const planner = new QueryPlanner();
        plan = planner.plan(ast, context);

        leftKey = plan.join.leftKey;
        rightKey = plan.join.rightKey;
        const leftColumns = plan.leftScan.columns;
        const rightColumns = plan.rightScan.columns;
        const leftFilters = plan.leftScan.filters;
        const rightFilters = plan.rightScan.filters;

        const leftColsWithKey = leftColumns.includes('*')
            ? ['*']
            : [...new Set([leftKey, ...leftColumns])];

        let leftWhereClause = '';
        if (leftFilters.length > 0) {
            leftWhereClause = ` WHERE ${leftFilters.map(f => filterToSQL(f)).join(' AND ')}`;
        }
        leftSQL = `SELECT ${leftColsWithKey.join(', ')} FROM ${leftTableName}${leftWhereClause}`;

        const rightColsWithKey = rightColumns.includes('*')
            ? ['*']
            : [...new Set([rightKey, ...rightColumns])];

        let rightWhereClause = '';
        if (rightFilters.length > 0) {
            rightWhereClause = ` WHERE ${rightFilters.map(f => filterToSQL(f)).join(' AND ')}`;
        }
        rightSQL = `SELECT ${rightColsWithKey.join(', ')} FROM ${rightTableName}${rightWhereClause}`;
    }

    console.log('[LanceDatabase] OPFS-backed hash join starting...');
    console.log('[LanceDatabase] Left query:', leftSQL);

    await opfsStorage.open();

    const joinExecutor = new OPFSJoinExecutor(opfsStorage);
    const leftExecutor = new SQLExecutor(leftDataset);
    const rightExecutor = new SQLExecutor(rightDataset);

    const leftStream = leftExecutor.executeStream(leftSQL);
    const leftMeta = await joinExecutor._partitionToOPFS(leftStream, leftKey, 'left', true);
    console.log(`[LanceDatabase] Left partitioned: ${leftMeta.totalRows} rows, ${leftMeta.collectedKeys?.size || 0} unique keys`);

    let optimizedRightSQL = rightSQL;
    const maxKeysForInClause = 1000;
    if (leftMeta.collectedKeys && leftMeta.collectedKeys.size > 0 &&
        leftMeta.collectedKeys.size <= maxKeysForInClause) {
        const inClause = buildInClause(rightKey, leftMeta.collectedKeys);
        optimizedRightSQL = appendWhereClause(rightSQL, inClause);
        console.log(`[LanceDatabase] Semi-join optimization: added IN clause with ${leftMeta.collectedKeys.size} keys`);
    }
    console.log('[LanceDatabase] Right query:', optimizedRightSQL);

    const rightStream = rightExecutor.executeStream(optimizedRightSQL);

    const results = [];
    let resultColumns = null;

    try {
        for await (const chunk of joinExecutor.executeHashJoin(
            null,
            rightStream,
            leftKey,
            rightKey,
            {
                limit: ast.limit || Infinity,
                leftAlias,
                rightAlias,
                joinType: join.type || 'INNER',
                prePartitionedLeft: leftMeta
            }
        )) {
            if (!resultColumns) {
                resultColumns = chunk.columns;
            }
            results.push(...chunk.rows);

            if (ast.limit && results.length >= ast.limit) {
                break;
            }
        }
    } catch (e) {
        console.error('[LanceDatabase] OPFS join failed:', e);
        throw e;
    }

    const stats = joinExecutor.getStats();
    console.log('[LanceDatabase] OPFS Join Stats:', stats);

    if (!resultColumns || results.length === 0) {
        return { columns: [], rows: [], total: 0, opfsStats: stats };
    }

    const projectedResults = applyProjection(
        results,
        resultColumns,
        plan.projection,
        leftAlias,
        rightAlias
    );

    const limitedResults = ast.limit
        ? projectedResults.rows.slice(0, ast.limit)
        : projectedResults.rows;

    return {
        columns: projectedResults.columns,
        rows: limitedResults,
        total: limitedResults.length,
        opfsStats: stats
    };
}

/**
 * Execute hash join with in-memory left side (for multiple JOINs).
 */
export async function hashJoinWithInMemoryLeft(db, leftResult, rightDataset, ast, context) {
    const { leftAlias, rightAlias, leftTableName, rightTableName } = context;
    const join = ast.joins[0];
    const joinType = join.type || 'INNER';

    const joinCondition = join.on;
    if (joinType !== 'CROSS') {
        if (!joinCondition || joinCondition.type !== 'binary' || joinCondition.op !== '=') {
            throw new Error('JOIN ON condition must be an equality (e.g., table1.col1 = table2.col2)');
        }
    }

    let leftKey, rightKey;
    if (joinType === 'CROSS') {
        leftKey = null;
        rightKey = null;
    } else {
        const leftExpr = joinCondition.left;
        const rightExpr = joinCondition.right;
        const leftCol = leftExpr.column;
        const rightCol = rightExpr.column;

        const leftColsSet = new Set(leftResult.columns.map(c => {
            const parts = c.split('.');
            return parts[parts.length - 1];
        }));

        if (leftColsSet.has(leftCol)) {
            leftKey = leftCol;
            rightKey = rightCol;
        } else {
            leftKey = rightCol;
            rightKey = leftCol;
        }
    }

    console.log(`[LanceDatabase] Multi-JOIN: left in-memory (${leftResult.rows.length} rows), right: ${rightTableName}`);

    let rightSQL = `SELECT * FROM ${rightTableName}`;

    const maxKeysForInClause = 1000;
    if (leftKey && joinType !== 'CROSS') {
        const leftKeyIndex = findColumnIndex(leftResult.columns, leftKey);
        if (leftKeyIndex !== -1) {
            const leftKeys = new Set();
            for (const row of leftResult.rows) {
                const key = row[leftKeyIndex];
                if (key !== null && key !== undefined) {
                    leftKeys.add(key);
                }
            }
            if (leftKeys.size > 0 && leftKeys.size <= maxKeysForInClause) {
                const inClause = buildInClause(rightKey, leftKeys);
                rightSQL = appendWhereClause(rightSQL, inClause);
                console.log(`[LanceDatabase] Multi-JOIN semi-join: ${leftKeys.size} keys`);
            }
        }
    }

    const rightExecutor = new SQLExecutor(rightDataset);
    const rightResult = await rightExecutor.execute(new SQLParser(new SQLLexer(rightSQL).tokenize()).parse());

    const leftKeyIndex = leftKey ? findColumnIndex(leftResult.columns, leftKey) : -1;
    const rightKeyIndex = rightKey ? findColumnIndex(rightResult.columns, rightKey) : -1;

    const resultColumns = [
        ...leftResult.columns,
        ...rightResult.columns.map(c => `${rightAlias}.${c}`)
    ];

    const results = [];
    const rightNulls = new Array(rightResult.columns.length).fill(null);
    const leftNulls = new Array(leftResult.columns.length).fill(null);

    if (joinType === 'CROSS') {
        for (const leftRow of leftResult.rows) {
            for (const rightRow of rightResult.rows) {
                results.push([...leftRow, ...rightRow]);
                if (ast.limit && results.length >= ast.limit) break;
            }
            if (ast.limit && results.length >= ast.limit) break;
        }
    } else {
        const rightHash = new Map();
        for (const row of rightResult.rows) {
            const key = row[rightKeyIndex];
            if (key !== null && key !== undefined) {
                if (!rightHash.has(key)) rightHash.set(key, []);
                rightHash.get(key).push(row);
            }
        }

        const matchedRightRows = (joinType === 'FULL' || joinType === 'RIGHT')
            ? new Set() : null;

        for (const leftRow of leftResult.rows) {
            const key = leftRow[leftKeyIndex];
            const rightMatches = rightHash.get(key) || [];

            if (rightMatches.length > 0) {
                for (const rightRow of rightMatches) {
                    results.push([...leftRow, ...rightRow]);
                    if (matchedRightRows) {
                        matchedRightRows.add(rightResult.rows.indexOf(rightRow));
                    }
                    if (ast.limit && results.length >= ast.limit) break;
                }
            } else if (joinType === 'LEFT' || joinType === 'FULL') {
                results.push([...leftRow, ...rightNulls]);
            }
            if (ast.limit && results.length >= ast.limit) break;
        }

        if ((joinType === 'RIGHT' || joinType === 'FULL') && matchedRightRows) {
            for (let i = 0; i < rightResult.rows.length; i++) {
                if (!matchedRightRows.has(i)) {
                    results.push([...leftNulls, ...rightResult.rows[i]]);
                    if (ast.limit && results.length >= ast.limit) break;
                }
            }
        }
    }

    const limitedResults = ast.limit ? results.slice(0, ast.limit) : results;

    return {
        columns: resultColumns,
        rows: limitedResults,
        total: limitedResults.length
    };
}

/**
 * Find column index by name, handling qualified names
 */
export function findColumnIndex(columns, columnName) {
    let idx = columns.indexOf(columnName);
    if (idx !== -1) return idx;

    for (let i = 0; i < columns.length; i++) {
        const col = columns[i];
        const parts = col.split('.');
        if (parts[parts.length - 1] === columnName) {
            return i;
        }
    }

    return -1;
}

/**
 * Convert filter expression to SQL WHERE clause
 */
export function filterToSQL(expr) {
    if (!expr) return '';

    if (expr.type === 'binary') {
        const left = filterToSQL(expr.left);
        const right = filterToSQL(expr.right);
        return `${left} ${expr.op} ${right}`;
    } else if (expr.type === 'column') {
        return expr.column;
    } else if (expr.type === 'literal') {
        if (typeof expr.value === 'string') {
            const escaped = expr.value.replace(/'/g, "''");
            return `'${escaped}'`;
        }
        if (expr.value === null) return 'NULL';
        return String(expr.value);
    } else if (expr.type === 'call') {
        const args = (expr.args || []).map(a => filterToSQL(a)).join(', ');
        return `${expr.name}(${args})`;
    } else if (expr.type === 'in') {
        const col = filterToSQL(expr.expr);
        const vals = expr.values.map(v => filterToSQL(v)).join(', ');
        return `${col} IN (${vals})`;
    } else if (expr.type === 'between') {
        const col = filterToSQL(expr.expr);
        const low = filterToSQL(expr.low);
        const high = filterToSQL(expr.high);
        return `${col} BETWEEN ${low} AND ${high}`;
    } else if (expr.type === 'like') {
        const col = filterToSQL(expr.expr);
        const pattern = filterToSQL(expr.pattern);
        return `${col} LIKE ${pattern}`;
    } else if (expr.type === 'unary') {
        const operand = filterToSQL(expr.operand);
        if (expr.op === 'NOT') return `NOT ${operand}`;
        return `${expr.op}${operand}`;
    }

    console.warn('[LanceDB] Unknown filter expression type:', expr.type);
    return '';
}

/**
 * Apply projection to select only requested columns from joined result
 */
export function applyProjection(rows, allColumns, projection, leftAlias, rightAlias) {
    if (projection.includes('*')) {
        return { columns: allColumns, rows };
    }

    const projectedColumns = [];
    const columnIndices = [];

    for (const col of projection) {
        if (col === '*') continue;

        let idx = -1;
        let outputColName = col.column;

        if (col.table) {
            const qualifiedName = `${col.table}.${col.column}`;
            idx = allColumns.indexOf(qualifiedName);
            outputColName = qualifiedName;
        }

        if (idx === -1) {
            idx = allColumns.findIndex(c => c === col.column || c.endsWith(`.${col.column}`));
            if (idx !== -1) {
                outputColName = allColumns[idx];
            }
        }

        if (idx !== -1) {
            projectedColumns.push(col.alias || outputColName);
            columnIndices.push(idx);
        }
    }

    const projectedRows = rows.map(row =>
        columnIndices.map(idx => row[idx])
    );

    return { columns: projectedColumns, rows: projectedRows };
}

/**
 * Build an IN clause for semi-join optimization
 */
export function buildInClause(column, keys) {
    const values = Array.from(keys).map(k => {
        if (typeof k === 'string') {
            return `'${k.replace(/'/g, "''")}'`;
        }
        if (k === null) return 'NULL';
        return String(k);
    }).join(', ');
    return `${column} IN (${values})`;
}

/**
 * Append a WHERE clause or AND condition to existing SQL
 */
export function appendWhereClause(sql, clause) {
    const upperSQL = sql.toUpperCase();
    if (upperSQL.includes('WHERE')) {
        return sql.replace(/WHERE\s+/i, `WHERE ${clause} AND `);
    }
    return sql.replace(/FROM\s+(\w+)(\s+\w+)?/i, (match) => `${match} WHERE ${clause}`);
}
