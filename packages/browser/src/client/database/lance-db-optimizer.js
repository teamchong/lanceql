/**
 * LanceDatabase - Query optimization utilities
 * Extracted from lance-database.js for modularity
 */

/**
 * Get cached query plan
 * @param {LanceDatabase} db - Database instance
 * @param {string} sql - SQL query string
 * @returns {Object|null} Cached plan or null
 */
export function getCachedPlan(db, sql) {
    const normalized = normalizeSQL(sql);
    const cached = db._planCache.get(normalized);
    if (cached) {
        cached.hits++;
        cached.lastUsed = Date.now();
        return cached.plan;
    }
    return null;
}

/**
 * Cache a query plan
 * @param {LanceDatabase} db - Database instance
 * @param {string} sql - SQL query string
 * @param {Object} plan - Parsed AST plan
 */
export function setCachedPlan(db, sql, plan) {
    const normalized = normalizeSQL(sql);

    // LRU eviction if at capacity
    if (db._planCache.size >= db._planCacheMaxSize) {
        let oldest = null;
        let oldestTime = Infinity;
        for (const [key, value] of db._planCache) {
            if (value.lastUsed < oldestTime) {
                oldestTime = value.lastUsed;
                oldest = key;
            }
        }
        if (oldest) db._planCache.delete(oldest);
    }

    db._planCache.set(normalized, {
        plan,
        hits: 0,
        lastUsed: Date.now(),
        created: Date.now()
    });
}

/**
 * Normalize SQL for cache key (remove extra whitespace, lowercase)
 * @param {string} sql - SQL query string
 * @returns {string} Normalized SQL
 */
export function normalizeSQL(sql) {
    return sql.trim().replace(/\s+/g, ' ').toLowerCase();
}

/**
 * Get plan cache statistics
 * @param {LanceDatabase} db - Database instance
 * @returns {Object} Cache stats
 */
export function getPlanCacheStats(db) {
    let totalHits = 0;
    for (const v of db._planCache.values()) {
        totalHits += v.hits;
    }
    return {
        size: db._planCache.size,
        maxSize: db._planCacheMaxSize,
        totalHits
    };
}

/**
 * Optimize expression with constant folding and boolean simplification
 * @param {Object} expr - Expression AST node
 * @returns {Object} Optimized expression
 */
export function optimizeExpr(expr) {
    if (!expr) return expr;

    // Recursively optimize children
    if (expr.left) expr.left = optimizeExpr(expr.left);
    if (expr.right) expr.right = optimizeExpr(expr.right);
    if (expr.operand) expr.operand = optimizeExpr(expr.operand);
    if (expr.args) expr.args = expr.args.map(a => optimizeExpr(a));

    const op = expr.op || expr.operator;

    // Constant folding for binary operations
    if (expr.type === 'binary' &&
        isConstantExpr(expr.left) &&
        isConstantExpr(expr.right)) {
        return foldBinary(expr);
    }

    // Boolean simplification
    if (expr.type === 'binary' && op === 'AND') {
        if (isTrueExpr(expr.right)) return expr.left;
        if (isTrueExpr(expr.left)) return expr.right;
        if (isFalseExpr(expr.left) || isFalseExpr(expr.right)) {
            return { type: 'literal', value: false };
        }
    }
    if (expr.type === 'binary' && op === 'OR') {
        if (isFalseExpr(expr.right)) return expr.left;
        if (isFalseExpr(expr.left)) return expr.right;
        if (isTrueExpr(expr.left) || isTrueExpr(expr.right)) {
            return { type: 'literal', value: true };
        }
    }

    return expr;
}

/**
 * Check if expression is a constant
 */
export function isConstantExpr(expr) {
    return expr && ['literal', 'number', 'string'].includes(expr.type);
}

/**
 * Check if expression is TRUE
 */
export function isTrueExpr(expr) {
    return expr?.type === 'literal' && expr.value === true;
}

/**
 * Check if expression is FALSE
 */
export function isFalseExpr(expr) {
    return expr?.type === 'literal' && expr.value === false;
}

/**
 * Fold binary constant expression
 * @param {Object} expr - Binary expression with constant operands
 * @returns {Object} Literal result
 */
export function foldBinary(expr) {
    const left = getConstantValueExpr(expr.left);
    const right = getConstantValueExpr(expr.right);
    const op = expr.op || expr.operator;

    let result;
    switch (op) {
        case '+': result = left + right; break;
        case '-': result = left - right; break;
        case '*': result = left * right; break;
        case '/': result = right !== 0 ? left / right : null; break;
        case '%': result = left % right; break;
        case '=': case '==': result = left === right; break;
        case '!=': case '<>': result = left !== right; break;
        case '<': result = left < right; break;
        case '>': result = left > right; break;
        case '<=': result = left <= right; break;
        case '>=': result = left >= right; break;
        default: return expr;
    }

    return { type: 'literal', value: result };
}

/**
 * Get constant value from expression
 */
export function getConstantValueExpr(expr) {
    if (expr.type === 'number') return expr.value;
    if (expr.type === 'string') return expr.value;
    if (expr.type === 'literal') return expr.value;
    return null;
}

/**
 * Extract range predicates from WHERE clause for statistics-based pruning
 * @param {Object} where - WHERE clause AST
 * @returns {Array} Array of predicate objects
 */
export function extractRangePredicates(where) {
    const predicates = [];
    collectRangePredicates(where, predicates);
    return predicates;
}

/**
 * Recursively collect range predicates
 */
export function collectRangePredicates(expr, predicates) {
    if (!expr) return;

    const op = expr.op || expr.operator;

    // Handle AND - recurse both sides
    if (expr.type === 'binary' && op === 'AND') {
        collectRangePredicates(expr.left, predicates);
        collectRangePredicates(expr.right, predicates);
        return;
    }

    const normalizedOp = op === '==' ? '=' : op;
    if (['>', '<', '>=', '<=', '=', '!=', '<>'].includes(normalizedOp)) {
        if (isColumnRefExpr(expr.left) && isConstantExpr(expr.right)) {
            predicates.push({
                column: getColumnNameExpr(expr.left),
                operator: normalizedOp,
                value: getConstantValueExpr(expr.right)
            });
        }
        else if (isConstantExpr(expr.left) && isColumnRefExpr(expr.right)) {
            predicates.push({
                column: getColumnNameExpr(expr.right),
                operator: flipOperatorExpr(normalizedOp),
                value: getConstantValueExpr(expr.left)
            });
        }
    }

    // BETWEEN clause
    if (expr.type === 'between' && expr.expr) {
        const col = getColumnNameExpr(expr.expr);
        if (col && expr.low && expr.high) {
            predicates.push({
                column: col,
                operator: '>=',
                value: getConstantValueExpr(expr.low)
            });
            predicates.push({
                column: col,
                operator: '<=',
                value: getConstantValueExpr(expr.high)
            });
        }
    }
}

/**
 * Flip comparison operator (for constant on left side)
 */
export function flipOperatorExpr(op) {
    const flips = { '>': '<', '<': '>', '>=': '<=', '<=': '>=' };
    return flips[op] || op;
}

/**
 * Check if expression is a column reference
 */
export function isColumnRefExpr(expr) {
    return expr && (expr.type === 'column' || expr.type === 'identifier');
}

/**
 * Get column name from expression
 */
export function getColumnNameExpr(expr) {
    if (expr.type === 'column') return expr.name || expr.column;
    if (expr.type === 'identifier') return expr.name || expr.value;
    return null;
}

/**
 * Check if a fragment can be pruned based on statistics and predicates
 * @param {Object} fragmentStats - Column statistics for the fragment
 * @param {Array} predicates - Extracted predicates from WHERE clause
 * @returns {boolean} True if fragment can be safely skipped
 */
export function canPruneFragment(fragmentStats, predicates) {
    for (const pred of predicates) {
        const stats = fragmentStats[pred.column];
        if (!stats) continue;

        const { min, max, nullCount, rowCount } = stats;

        if (nullCount === rowCount) return true;

        switch (pred.operator) {
            case '>':
                if (max <= pred.value) return true;
                break;
            case '>=':
                if (max < pred.value) return true;
                break;
            case '<':
                if (min >= pred.value) return true;
                break;
            case '<=':
                if (min > pred.value) return true;
                break;
            case '=':
                if (pred.value < min || pred.value > max) return true;
                break;
            case '!=':
            case '<>':
                if (min === max && min === pred.value) return true;
                break;
        }
    }

    return false;
}

/**
 * Execute EXPLAIN query - return query plan without executing
 * @param {LanceDatabase} db - Database instance
 * @param {Object} ast - Parsed AST of the inner query
 * @returns {Object} Plan information
 */
export function explainQuery(db, ast) {
    const plan = {
        type: ast.type,
        tables: [],
        predicates: [],
        optimizations: []
    };

    if (ast.from) {
        plan.tables.push({
            name: ast.from.name || ast.from.table,
            alias: ast.from.alias
        });
    }

    if (ast.joins) {
        for (const join of ast.joins) {
            plan.tables.push({
                name: join.table?.name || join.table?.table,
                alias: join.table?.alias,
                joinType: join.type
            });
        }
    }

    if (ast.where) {
        plan.predicates = extractRangePredicates(ast.where);
    }

    if (ast.where) {
        plan.optimizations.push('PREDICATE_PUSHDOWN');
    }
    if (ast.groupBy) {
        plan.optimizations.push('AGGREGATE');
    }
    if (ast.orderBy) {
        plan.optimizations.push('SORT');
    }
    if (ast.limit) {
        plan.optimizations.push('LIMIT_PUSHDOWN');
    }

    return {
        columns: ['Plan'],
        rows: [[JSON.stringify(plan, null, 2)]],
        total: 1
    };
}
