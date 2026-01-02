/**
 * SQL Executor - Utility functions (DISTINCT, PIVOT, UNPIVOT, hashing, streaming)
 * Extracted from executor.js for modularity
 */

/**
 * Hash a row for deduplication
 * @param {Array} row - Data row
 * @returns {number} - Hash value
 */
export function hashRow(row) {
    let hash = 0;
    for (const val of row) {
        if (val === null || val === undefined) {
            hash = (hash * 31 + 0) >>> 0;
        } else if (typeof val === 'number') {
            hash = (hash * 31 + val) >>> 0;
        } else if (val instanceof Uint8Array) {
            for (const b of val) {
                hash = (hash * 31 + b) >>> 0;
            }
        } else if (Array.isArray(val)) {
            for (let j = 0; j < val.length; j++) {
                hash = (hash * 31 + (val[j] || 0)) >>> 0;
            }
        } else {
            const str = String(val);
            for (let j = 0; j < str.length; j++) {
                hash = (hash * 31 + str.charCodeAt(j)) >>> 0;
            }
        }
    }
    return hash;
}

/**
 * Hash all rows for grouping/deduplication
 * @param {Array<Array>} rows - Data rows
 * @returns {Uint32Array} - Hash values for each row
 */
export function hashRows(rows) {
    const hashes = new Uint32Array(rows.length);
    for (let i = 0; i < rows.length; i++) {
        hashes[i] = hashRow(rows[i]);
    }
    return hashes;
}

/**
 * Compute aggregate function on a list of values
 * Used by PIVOT transformation
 * @param {string} funcName - Aggregate function name
 * @param {Array} values - Values to aggregate
 * @returns {*} - Aggregated result
 */
export function computeAggregate(funcName, values) {
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
 * Execute PIVOT transformation - convert rows to columns with aggregation
 * @param {Array<Array>} rows - Data rows
 * @param {Array} columns - Column definitions
 * @param {Object} pivot - PIVOT specification { aggregate, forColumn, inValues }
 * @returns {Object} - { rows, columns } transformed result
 */
export function executePivot(rows, columns, pivot) {
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

    // Build groups: Map<groupKey, { groupValues, pivots }>
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
            row.push(computeAggregate(aggregate.name, values));
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
 * Execute UNPIVOT transformation - convert columns to rows
 * @param {Array<Array>} rows - Data rows
 * @param {Array} columns - Column definitions
 * @param {Object} unpivot - UNPIVOT specification { valueColumn, nameColumn, inColumns }
 * @returns {Object} - { rows, columns } transformed result
 */
export function executeUnpivot(rows, columns, unpivot) {
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
 * Apply DISTINCT to rows using GPU acceleration when available
 * @param {Object} executor - SQLExecutor instance
 * @param {Array<Array>} rows - Data rows
 * @param {Object} gpuSorter - GPU sorter instance (optional)
 * @returns {Promise<Array<Array>>} - Unique rows
 */
export async function applyDistinct(executor, rows, gpuSorter = null) {
    if (rows.length === 0) return [];

    // Use GPU for large datasets
    if (rows.length >= 10000 && gpuSorter?.isAvailable?.()) {
        const hashes = hashRows(rows);
        const numGroups = await gpuSorter.groupBy(hashes);

        // Get first occurrence of each group
        const firstOccurrence = new Int32Array(numGroups).fill(-1);
        for (let i = 0; i < rows.length; i++) {
            const gid = hashes[i] % numGroups;
            if (firstOccurrence[gid] === -1) {
                firstOccurrence[gid] = i;
            }
        }

        const uniqueRows = [];
        for (let gid = 0; gid < numGroups; gid++) {
            if (firstOccurrence[gid] !== -1) {
                uniqueRows.push(rows[firstOccurrence[gid]]);
            }
        }
        return uniqueRows;
    }

    // CPU fallback using Set
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
 * Apply ORDER BY to rows using GPU acceleration when available
 * @param {Object} executor - SQLExecutor instance
 * @param {Array<Array>} rows - Data rows (mutated in place)
 * @param {Array} orderBy - ORDER BY specification
 * @param {Array} outputColumns - Output column definitions
 * @param {Object} gpuSorter - GPU sorter instance (optional)
 * @returns {Promise<void>}
 */
export async function applyOrderBy(executor, rows, orderBy, outputColumns, gpuSorter = null) {
    // Build column index map
    const colIdxMap = {};
    let idx = 0;
    for (const col of outputColumns) {
        if (col.type === 'star') {
            for (const name of executor.file.columnNames || []) {
                colIdxMap[name.toLowerCase()] = idx++;
            }
        } else {
            const name = col.alias || executor.exprToName(col.expr);
            colIdxMap[name.toLowerCase()] = idx++;
        }
    }

    // Use GPU for large datasets (10,000+ rows)
    if (rows.length >= 10000 && gpuSorter?.isAvailable?.()) {
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

        // Reorder rows
        const sorted = [];
        for (let i = 0; i < rows.length; i++) {
            sorted.push(rows[indices[i]]);
        }
        rows.length = 0;
        rows.push(...sorted);
        return;
    }

    // CPU fallback
    rows.sort((a, b) => {
        for (const ob of orderBy) {
            const colIdx = colIdxMap[ob.column.toLowerCase()];
            if (colIdx === undefined) continue;

            const valA = a[colIdx];
            const valB = b[colIdx];
            const dir = ob.descending ? -1 : 1;

            if (valA == null && valB == null) continue;
            if (valA == null) return 1 * dir;
            if (valB == null) return -1 * dir;
            if (valA < valB) return -1 * dir;
            if (valA > valB) return 1 * dir;
        }
        return 0;
    });
}

/**
 * Evaluate WHERE expression on a single row (for streaming)
 * @param {Object} expr - WHERE expression AST
 * @param {Array<string>} columns - Column names
 * @param {Array} row - Data row
 * @param {Function} getValueFromExpr - Value extraction function
 * @returns {boolean} - True if row matches
 */
export function evaluateWhereExprOnRow(expr, columns, row, getValueFromExpr) {
    if (!expr) return true;

    if (expr.type === 'binary') {
        if (expr.op === 'AND') {
            return evaluateWhereExprOnRow(expr.left, columns, row, getValueFromExpr) &&
                   evaluateWhereExprOnRow(expr.right, columns, row, getValueFromExpr);
        }
        if (expr.op === 'OR') {
            return evaluateWhereExprOnRow(expr.left, columns, row, getValueFromExpr) ||
                   evaluateWhereExprOnRow(expr.right, columns, row, getValueFromExpr);
        }

        const leftVal = getValueFromExpr(expr.left, columns, row);
        const rightVal = getValueFromExpr(expr.right, columns, row);

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
 * @param {Object} expr - Expression AST
 * @param {Array<string>} columns - Column names
 * @param {Array} row - Data row
 * @returns {*} - Expression value
 */
export function getValueFromExpr(expr, columns, row) {
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

/**
 * Convert expression to display name
 * @param {Object} expr - Expression AST
 * @returns {string} - Display name
 */
export function exprToName(expr) {
    if (!expr) return '?';
    switch (expr.type) {
        case 'column':
            return expr.name || expr.column || '?';
        case 'literal':
            return String(expr.value);
        case 'call':
            const argNames = (expr.args || [])
                .map(a => exprToName(a))
                .join(', ');
            return `${expr.name}(${argNames})`;
        case 'binary':
            return `${exprToName(expr.left)} ${expr.op} ${exprToName(expr.right)}`;
        case 'star':
            return '*';
        default:
            return '?';
    }
}
