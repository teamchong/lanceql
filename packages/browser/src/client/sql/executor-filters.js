/**
 * SQL Executor - Filter and expression evaluation
 * Extracted from executor.js for modularity
 */

/**
 * Detect if WHERE clause is a simple comparison (column op value)
 * @param {Object} expr - WHERE expression AST
 * @param {Object} columnMap - Column name to index mapping
 * @param {Array<string>} columnTypes - Column type array
 * @returns {Object|null} - Simple filter info or null
 */
export function detectSimpleFilter(expr, columnMap, columnTypes) {
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

    const colIdx = columnMap[column.toLowerCase()];
    if (colIdx === undefined) return null;

    const colType = columnTypes[colIdx];
    if (!['int64', 'int32', 'float64', 'float32'].includes(colType)) return null;

    return { column, colIdx, op, value, colType };
}

/**
 * Optimized evaluation for simple column comparisons
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} filter - Simple filter info
 * @param {number} totalRows - Total rows to scan
 * @param {Function} onProgress - Progress callback
 * @returns {Promise<number[]>} - Matching row indices
 */
export async function evaluateSimpleFilter(executor, filter, totalRows, onProgress) {
    const matchingIndices = [];
    const batchSize = 5000;

    for (let batchStart = 0; batchStart < totalRows; batchStart += batchSize) {
        if (onProgress) {
            const pct = Math.round((batchStart / totalRows) * 100);
            onProgress(`Filtering ${filter.column}... ${pct}%`, batchStart, totalRows);
        }

        const batchEnd = Math.min(batchStart + batchSize, totalRows);
        const batchIndices = Array.from(
            { length: batchEnd - batchStart },
            (_, i) => batchStart + i
        );

        // Fetch only the filter column
        const colData = await executor.readColumnData(filter.colIdx, batchIndices);

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
            break;
        }
    }

    return matchingIndices;
}

/**
 * General evaluation for complex WHERE clauses
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} whereExpr - WHERE expression AST
 * @param {number} totalRows - Total rows to scan
 * @param {Function} onProgress - Progress callback
 * @returns {Promise<number[]>} - Matching row indices
 */
export async function evaluateComplexFilter(executor, whereExpr, totalRows, onProgress) {
    const matchingIndices = [];
    const batchSize = 1000;

    // Pre-compute needed columns
    const neededCols = new Set();
    executor.collectColumnsFromExpr(whereExpr, neededCols);

    for (let batchStart = 0; batchStart < totalRows; batchStart += batchSize) {
        if (onProgress) {
            onProgress(`Filtering rows...`, batchStart, totalRows);
        }

        const batchEnd = Math.min(batchStart + batchSize, totalRows);
        const batchIndices = Array.from(
            { length: batchEnd - batchStart },
            (_, i) => batchStart + i
        );

        // Fetch needed column data for this batch
        const batchData = {};
        for (const colName of neededCols) {
            const colIdx = executor.columnMap[colName.toLowerCase()];
            if (colIdx !== undefined) {
                batchData[colName.toLowerCase()] = await executor.readColumnData(colIdx, batchIndices);
            }
        }

        // Evaluate WHERE for each row in batch
        for (let i = 0; i < batchIndices.length; i++) {
            const result = evaluateExpr(executor, whereExpr, batchData, i);
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

/**
 * Evaluate an expression for a specific row
 * @param {Object} executor - SQLExecutor instance (for subquery support)
 * @param {Object} expr - Expression AST
 * @param {Object} columnData - Column data lookup
 * @param {number} rowIdx - Row index in columnData arrays
 * @returns {*} - Expression result
 */
export function evaluateExpr(executor, expr, columnData, rowIdx) {
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
            const left = evaluateExpr(executor, expr.left, columnData, rowIdx);
            const right = evaluateExpr(executor, expr.right, columnData, rowIdx);

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
            const operand = evaluateExpr(executor, expr.operand, columnData, rowIdx);
            switch (expr.op) {
                case '-': return -operand;
                case 'NOT': return !operand;
                default: return null;
            }
        }

        case 'in': {
            const value = evaluateExpr(executor, expr.expr, columnData, rowIdx);
            const values = expr.values.map(v => evaluateExpr(executor, v, columnData, rowIdx));
            return values.includes(value);
        }

        case 'between': {
            const value = evaluateExpr(executor, expr.expr, columnData, rowIdx);
            const low = evaluateExpr(executor, expr.low, columnData, rowIdx);
            const high = evaluateExpr(executor, expr.high, columnData, rowIdx);
            // SQL NULL semantics: NULL in any operand returns NULL
            if (value == null || low == null || high == null) return null;
            return value >= low && value <= high;
        }

        case 'like': {
            const value = evaluateExpr(executor, expr.expr, columnData, rowIdx);
            const pattern = evaluateExpr(executor, expr.pattern, columnData, rowIdx);
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
            return executor._executeSubquery(expr.query, columnData, rowIdx);
        }

        case 'array': {
            // Evaluate each element to build the array
            return expr.elements.map(el => evaluateExpr(executor, el, columnData, rowIdx));
        }

        case 'subscript': {
            // Array subscript access with 1-based indexing (SQL standard)
            const arr = evaluateExpr(executor, expr.array, columnData, rowIdx);
            const idx = evaluateExpr(executor, expr.index, columnData, rowIdx);
            if (!Array.isArray(arr)) return null;
            // SQL uses 1-based indexing
            return arr[idx - 1] ?? null;
        }

        default:
            return null;
    }
}

/**
 * Evaluate expression on in-memory data
 * @param {Object} expr - Expression AST
 * @param {Object} columnData - Column data lookup (arrays or single values)
 * @param {number} rowIdx - Row index
 * @returns {*} - Expression result
 */
export function evaluateInMemoryExpr(expr, columnData, rowIdx) {
    if (!expr) return null;

    switch (expr.type) {
        case 'literal':
            return expr.value;

        case 'column': {
            const colName = expr.column.toLowerCase();
            const col = columnData[colName];
            return col ? (Array.isArray(col) ? col[rowIdx] : col) : null;
        }

        case 'binary': {
            const left = evaluateInMemoryExpr(expr.left, columnData, rowIdx);
            const right = evaluateInMemoryExpr(expr.right, columnData, rowIdx);
            const op = expr.op || expr.operator;
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
            const operand = evaluateInMemoryExpr(expr.operand, columnData, rowIdx);
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
            const funcName = expr.name.toUpperCase();
            const args = expr.args?.map(a => evaluateInMemoryExpr(a, columnData, rowIdx)) || [];
            switch (funcName) {
                case 'UPPER': return String(args[0]).toUpperCase();
                case 'LOWER': return String(args[0]).toLowerCase();
                case 'LENGTH': return String(args[0]).length;
                case 'SUBSTR': case 'SUBSTRING': {
                    if (args[0] == null || args[1] == null) return null;
                    const start = Number(args[1]);
                    if (isNaN(start)) return null;
                    const len = args[2] != null ? Number(args[2]) : undefined;
                    if (len !== undefined && (isNaN(len) || len < 0)) return null;
                    return String(args[0]).substring(start - 1, len !== undefined ? start - 1 + len : undefined);
                }
                case 'COALESCE': return args.find(a => a != null) ?? null;
                case 'ABS': return Math.abs(args[0]);
                case 'ROUND': return Math.round(args[0] * Math.pow(10, args[1] || 0)) / Math.pow(10, args[1] || 0);
                case 'GROUPING': {
                    // GROUPING(col) returns 1 if col is super-aggregate, 0 otherwise
                    const colArg = expr.args?.[0];
                    const colName = colArg?.column || colArg?.name;
                    if (!colName) return 0;
                    const groupingSet = columnData._groupingSet || [];
                    return groupingSet.includes(colName.toLowerCase()) ? 0 : 1;
                }
                default: return null;
            }
        }

        case 'in': {
            const val = evaluateInMemoryExpr(expr.expr, columnData, rowIdx);
            const values = expr.values.map(v => v.value ?? evaluateInMemoryExpr(v, columnData, rowIdx));
            return values.includes(val);
        }

        case 'between': {
            const val = evaluateInMemoryExpr(expr.expr, columnData, rowIdx);
            const low = evaluateInMemoryExpr(expr.low, columnData, rowIdx);
            const high = evaluateInMemoryExpr(expr.high, columnData, rowIdx);
            if (val == null || low == null || high == null) return null;
            return val >= low && val <= high;
        }

        case 'like': {
            const val = String(evaluateInMemoryExpr(expr.expr, columnData, rowIdx));
            const pattern = evaluateInMemoryExpr(expr.pattern, columnData, rowIdx);
            const regex = new RegExp('^' + String(pattern).replace(/%/g, '.*').replace(/_/g, '.') + '$', 'i');
            return regex.test(val);
        }

        case 'array': {
            return expr.elements.map(el => evaluateInMemoryExpr(el, columnData, rowIdx));
        }

        case 'subscript': {
            const arr = evaluateInMemoryExpr(expr.array, columnData, rowIdx);
            const idx = evaluateInMemoryExpr(expr.index, columnData, rowIdx);
            if (!Array.isArray(arr)) return null;
            return arr[idx - 1] ?? null;
        }

        default:
            return null;
    }
}

/**
 * Collect column names from an expression
 * @param {Object} expr - Expression AST
 * @param {Set<string>} columns - Set to add column names to
 */
export function collectColumnsFromExpr(expr, columns) {
    if (!expr) return;

    switch (expr.type) {
        case 'column':
            columns.add(expr.name.toLowerCase());
            break;
        case 'binary':
            collectColumnsFromExpr(expr.left, columns);
            collectColumnsFromExpr(expr.right, columns);
            break;
        case 'unary':
            collectColumnsFromExpr(expr.operand, columns);
            break;
        case 'call':
            for (const arg of expr.args || []) {
                collectColumnsFromExpr(arg, columns);
            }
            break;
        case 'in':
            collectColumnsFromExpr(expr.expr, columns);
            break;
        case 'between':
            collectColumnsFromExpr(expr.expr, columns);
            break;
        case 'like':
            collectColumnsFromExpr(expr.expr, columns);
            break;
        case 'near':
            collectColumnsFromExpr(expr.column, columns);
            break;
    }
}
