/**
 * Single-table query planning
 * Extracted from query-planner.js for modularity
 */

/**
 * Generate optimized plan for single-table queries.
 * @param {Object} planner - QueryPlanner instance
 * @param {Object} ast - Parsed SQL AST
 * @returns {Object} Physical execution plan
 */
export function planSingleTable(planner, ast) {
    const plan = {
        type: ast.type,
        scanColumns: [],
        pushedFilters: [],
        postFilters: [],
        aggregations: [],
        groupBy: [],
        having: null,
        orderBy: [],
        limit: ast.limit || null,
        offset: ast.offset || 0,
        projection: [],
        canUseStatistics: false,
        canStreamResults: true,
        estimatedSelectivity: 1.0,
    };

    const neededColumns = new Set();

    // 1. Columns from SELECT
    if (ast.columns === '*' || (Array.isArray(ast.columns) && ast.columns.some(c => c.type === 'star'))) {
        plan.projection = ['*'];
        plan.canStreamResults = false;
    } else if (Array.isArray(ast.columns)) {
        for (const col of ast.columns) {
            collectColumnsFromSelectItem(col, neededColumns, plan);
        }
    }

    // 2. Columns from WHERE
    if (ast.where) {
        collectColumnsFromExpr(ast.where, neededColumns);
        analyzeFilterPushdown(ast.where, plan);
    }

    // 3. Columns from GROUP BY
    if (ast.groupBy && ast.groupBy.length > 0) {
        for (const groupExpr of ast.groupBy) {
            collectColumnsFromExpr(groupExpr, neededColumns);
            plan.groupBy.push(groupExpr);
        }
    }

    // 4. Columns from HAVING
    if (ast.having) {
        collectColumnsFromExpr(ast.having, neededColumns);
        plan.having = ast.having;
    }

    // 5. Columns from ORDER BY
    if (ast.orderBy && ast.orderBy.length > 0) {
        for (const orderItem of ast.orderBy) {
            collectColumnsFromExpr(orderItem.expr || orderItem, neededColumns);
            plan.orderBy.push(orderItem);
        }
    }

    plan.scanColumns = Array.from(neededColumns);
    plan.estimatedSelectivity = estimateSelectivity(plan.pushedFilters);
    plan.canUseStatistics = plan.pushedFilters.some(f =>
        f.type === 'range' || f.type === 'equality'
    );

    return plan;
}

/**
 * Collect columns from a SELECT item
 */
export function collectColumnsFromSelectItem(item, columns, plan) {
    if (item.type === 'star') {
        plan.projection.push('*');
        return;
    }

    if (item.type === 'expr') {
        const expr = item.expr;

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
                        collectColumnsFromExpr(arg, columns);
                    }
                }

                plan.aggregations.push(agg);
                plan.projection.push({ type: 'aggregation', index: plan.aggregations.length - 1 });
                return;
            }
        }

        collectColumnsFromExpr(expr, columns);
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
export function collectColumnsFromExpr(expr, columns) {
    if (!expr) return;

    if (expr.type === 'column') {
        columns.add(expr.name || expr.column);
    } else if (expr.type === 'binary') {
        collectColumnsFromExpr(expr.left, columns);
        collectColumnsFromExpr(expr.right, columns);
    } else if (expr.type === 'call') {
        for (const arg of (expr.args || [])) {
            collectColumnsFromExpr(arg, columns);
        }
    } else if (expr.type === 'unary') {
        collectColumnsFromExpr(expr.operand, columns);
    }
}

/**
 * Analyze WHERE clause for filter pushdown
 */
export function analyzeFilterPushdown(expr, plan) {
    if (!expr) return;

    if (expr.type === 'binary') {
        if (isPushableFilter(expr)) {
            plan.pushedFilters.push(classifyFilter(expr));
        } else if (expr.op === 'AND') {
            analyzeFilterPushdown(expr.left, plan);
            analyzeFilterPushdown(expr.right, plan);
        } else if (expr.op === 'OR') {
            const leftPushable = isPushableFilter(expr.left);
            const rightPushable = isPushableFilter(expr.right);

            if (leftPushable && rightPushable) {
                plan.pushedFilters.push({
                    type: 'or',
                    left: classifyFilter(expr.left),
                    right: classifyFilter(expr.right),
                });
            } else {
                plan.postFilters.push(expr);
            }
        } else {
            plan.postFilters.push(expr);
        }
    } else {
        plan.postFilters.push(expr);
    }
}

/**
 * Check if a filter can be pushed down
 */
export function isPushableFilter(expr) {
    if (expr.type !== 'binary') return false;

    const compOps = ['=', '==', '!=', '<>', '<', '<=', '>', '>=', 'LIKE', 'IN', 'BETWEEN'];
    if (!compOps.includes(expr.op.toUpperCase())) return false;

    const leftIsCol = expr.left.type === 'column';
    const rightIsCol = expr.right?.type === 'column';
    const leftIsLiteral = expr.left.type === 'literal' || expr.left.type === 'list';
    const rightIsLiteral = expr.right?.type === 'literal' || expr.right?.type === 'list';

    return (leftIsCol && rightIsLiteral) || (rightIsCol && leftIsLiteral);
}

/**
 * Classify a filter for optimization
 */
export function classifyFilter(expr) {
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
 * Estimate selectivity of filters
 */
export function estimateSelectivity(filters) {
    if (filters.length === 0) return 1.0;

    let selectivity = 1.0;
    for (const f of filters) {
        switch (f.type) {
            case 'equality':
                selectivity *= 0.1;
                break;
            case 'range':
                selectivity *= 0.3;
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
    return Math.max(0.01, selectivity);
}

