/**
 * SQL Executor - Window function support
 * Extracted from executor.js for modularity
 */

/**
 * Check if query has window functions
 * @param {Object} ast - Parsed SQL AST
 * @returns {boolean} - True if query contains window functions
 */
export function hasWindowFunctions(ast) {
    return ast.columns?.some(col => col.expr?.type === 'call' && col.expr.over);
}

/**
 * Partition rows based on PARTITION BY expressions
 * @param {Array} rows - Data rows
 * @param {Array} partitionBy - PARTITION BY expressions
 * @param {Object} columnData - Column data lookup
 * @param {Function} evaluateExpr - Expression evaluator function
 * @returns {Array<Array>} - Partitioned row groups
 */
export function partitionRows(rows, partitionBy, columnData, evaluateExpr) {
    if (!partitionBy || partitionBy.length === 0) {
        // No partitioning - all rows in one partition
        return [rows.map((r, i) => ({ idx: i, row: r }))];
    }

    const groups = new Map();
    for (let i = 0; i < rows.length; i++) {
        const key = partitionBy
            .map(expr => JSON.stringify(evaluateExpr(expr, columnData, i)))
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
 * @param {Object} a - First row with idx property
 * @param {Object} b - Second row with idx property
 * @param {Array} orderBy - ORDER BY specification
 * @param {Object} columnData - Column data lookup
 * @param {Function} evaluateExpr - Expression evaluator function
 * @returns {number} - Comparison result (-1, 0, 1)
 */
export function compareRowsByOrder(a, b, orderBy, columnData, evaluateExpr) {
    for (const ob of orderBy) {
        const valA = evaluateExpr({ type: 'column', column: ob.column }, columnData, a.idx);
        const valB = evaluateExpr({ type: 'column', column: ob.column }, columnData, b.idx);

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
export function getFrameBounds(frame, partition, currentIdx) {
    const n = partition.length;
    let startIdx = 0;
    let endIdx = currentIdx;

    // Parse start bound (parser uses spaces in type names)
    const start = frame.start || { type: 'UNBOUNDED PRECEDING' };
    const startType = start.type.replace(' ', '_').toUpperCase();
    // Coerce offset to number with fallback to 1
    const startOffset = Number(start.offset ?? start.value ?? 1) || 1;
    switch (startType) {
        case 'UNBOUNDED_PRECEDING':
            startIdx = 0;
            break;
        case 'CURRENT_ROW':
            startIdx = currentIdx;
            break;
        case 'PRECEDING':
            startIdx = Math.max(0, currentIdx - startOffset);
            break;
        case 'FOLLOWING':
            startIdx = Math.min(n - 1, currentIdx + startOffset);
            break;
    }

    // Parse end bound
    const end = frame.end || { type: 'CURRENT ROW' };
    const endType = end.type.replace(' ', '_').toUpperCase();
    const endOffset = Number(end.offset ?? end.value ?? 1) || 1;
    switch (endType) {
        case 'UNBOUNDED_FOLLOWING':
            endIdx = n - 1;
            break;
        case 'CURRENT_ROW':
            endIdx = currentIdx;
            break;
        case 'PRECEDING':
            endIdx = Math.max(0, currentIdx - endOffset);
            break;
        case 'FOLLOWING':
            endIdx = Math.min(n - 1, currentIdx + endOffset);
            break;
    }

    // Ensure valid bounds
    if (startIdx > endIdx) [startIdx, endIdx] = [endIdx, startIdx];
    return [startIdx, endIdx];
}

/**
 * Compute a single window function
 * @param {string} funcName - Function name (ROW_NUMBER, RANK, etc.)
 * @param {Array} args - Function arguments
 * @param {Object} over - OVER clause specification
 * @param {Array} rows - Data rows
 * @param {Object} columnData - Column data lookup
 * @param {Function} evaluateExpr - Expression evaluator function
 * @returns {Array} - Computed values for each row
 */
export function computeWindowFunction(funcName, args, over, rows, columnData, evaluateExpr) {
    const results = new Array(rows.length).fill(null);

    // Partition rows
    const partitions = partitionRows(rows, over.partitionBy, columnData, evaluateExpr);

    for (const partition of partitions) {
        // Sort partition by ORDER BY if specified
        if (over.orderBy && over.orderBy.length > 0) {
            partition.sort((a, b) => compareRowsByOrder(a, b, over.orderBy, columnData, evaluateExpr));
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
                    if (i > 0 && compareRowsByOrder(partition[i-1], partition[i], over.orderBy, columnData, evaluateExpr) === 0) {
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
                    } else if (compareRowsByOrder(partition[i-1], partition[i], over.orderBy, columnData, evaluateExpr) === 0) {
                        results[rowIdx] = results[partition[i-1].idx];
                    } else {
                        results[rowIdx] = results[partition[i-1].idx] + 1;
                    }
                    break;
                }

                case 'NTILE': {
                    // Clamp N to valid range and use SQL standard algorithm
                    const requestedN = Math.max(1, Number(args[0]?.value) || 1);
                    const n = Math.min(requestedN, partition.length);
                    results[rowIdx] = Math.floor(i * n / partition.length) + 1;
                    break;
                }

                case 'PERCENT_RANK': {
                    // PERCENT_RANK = (rank - 1) / (partition_size - 1)
                    let rank = i + 1;
                    for (let j = 0; j < i; j++) {
                        if (compareRowsByOrder(partition[j], partition[i], over.orderBy, columnData, evaluateExpr) === 0) {
                            rank = j + 1;
                            break;
                        }
                    }
                    const partitionSize = partition.length;
                    results[rowIdx] = partitionSize > 1 ? (rank - 1) / (partitionSize - 1) : 0;
                    break;
                }

                case 'CUME_DIST': {
                    // CUME_DIST = (rows with value <= current) / total_rows
                    let countLessOrEqual = 0;
                    for (let j = 0; j < partition.length; j++) {
                        const cmp = compareRowsByOrder(partition[j], partition[i], over.orderBy, columnData, evaluateExpr);
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
                        results[rowIdx] = evaluateExpr(lagCol, columnData, prevRowIdx);
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
                        results[rowIdx] = evaluateExpr(leadCol, columnData, nextRowIdx);
                    } else {
                        results[rowIdx] = defaultVal;
                    }
                    break;
                }

                case 'FIRST_VALUE': {
                    const firstRowIdx = partition[0].idx;
                    results[rowIdx] = evaluateExpr(args[0], columnData, firstRowIdx);
                    break;
                }

                case 'LAST_VALUE': {
                    // LAST_VALUE returns the last value within the frame
                    const frame = over.frame || {
                        type: 'RANGE',
                        start: { type: 'UNBOUNDED_PRECEDING' },
                        end: { type: 'CURRENT_ROW' }
                    };
                    const [, endIdx] = getFrameBounds(frame, partition, i);
                    const lastRowIdx = partition[endIdx].idx;
                    results[rowIdx] = evaluateExpr(args[0], columnData, lastRowIdx);
                    break;
                }

                case 'NTH_VALUE': {
                    const n = Number(args[1]?.value) || 1;
                    // Use frame bounds, not partition length
                    const frame = over.frame || { type: 'RANGE', start: { type: 'UNBOUNDED_PRECEDING' }, end: { type: 'CURRENT_ROW' } };
                    const [startIdx, endIdx] = getFrameBounds(frame, partition, i);
                    const frameSize = endIdx - startIdx + 1;
                    if (n > 0 && n <= frameSize) {
                        const nthRowIdx = partition[startIdx + n - 1].idx;
                        results[rowIdx] = evaluateExpr(args[0], columnData, nthRowIdx);
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
                    // Get frame bounds
                    const frame = over.frame || {
                        type: 'RANGE',
                        start: { type: 'UNBOUNDED_PRECEDING' },
                        end: { type: 'CURRENT_ROW' }
                    };
                    const [startIdx, endIdx] = getFrameBounds(frame, partition, i);

                    // Collect values within frame
                    const isStar = args[0]?.type === 'star';
                    let values = [];
                    let frameRowCount = 0;
                    for (let j = startIdx; j <= endIdx; j++) {
                        frameRowCount++;
                        if (!isStar) {
                            const val = evaluateExpr(args[0], columnData, partition[j].idx);
                            const numVal = Number(val);
                            if (val != null && !isNaN(numVal)) values.push(numVal);
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
 * Execute window functions on in-memory data
 * @param {Object} ast - Parsed SQL AST
 * @param {Array} rows - Data rows
 * @param {Object} columnData - Column data lookup
 * @param {Function} evaluateExpr - Expression evaluator function
 * @returns {Array} - Window function results for each column
 */
export function computeWindowFunctions(ast, rows, columnData, evaluateExpr) {
    const windowColumns = [];

    for (let colIndex = 0; colIndex < ast.columns.length; colIndex++) {
        const col = ast.columns[colIndex];
        if (col.expr?.type === 'call' && col.expr.over) {
            const values = computeWindowFunction(
                col.expr.name,
                col.expr.args,
                col.expr.over,
                rows,
                columnData,
                evaluateExpr
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
 * Execute window functions on in-memory data and build result
 * @param {Object} executor - SQLExecutor instance
 * @param {Object} ast - Parsed SQL AST
 * @param {Object} data - In-memory data { columns, rows }
 * @param {Object} columnData - Column data lookup
 * @param {Array<number>} filteredIndices - Filtered row indices
 * @returns {Object} - Query result { columns, rows, total }
 */
export function executeWindowFunctions(executor, ast, data, columnData, filteredIndices) {
    // Build filtered rows
    const filteredRows = filteredIndices.map(idx => data.rows[idx]);

    // Compute window function results
    const windowResults = computeWindowFunctions(
        ast,
        filteredRows,
        columnData,
        (expr, colData, rowIdx) => executor._evaluateInMemoryExpr(expr, colData, rowIdx)
    );

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
                const windowCol = windowResults.find(w => w.colIndex === c);
                row.push(windowCol ? windowCol.values[i] : null);
            } else if (expr?.type === 'column') {
                const colName = (expr.name || expr.column || '').toLowerCase();
                row.push(columnData[colName]?.[origIdx] ?? null);
            } else {
                row.push(executor._evaluateInMemoryExpr(expr, columnData, origIdx));
            }
        }

        resultRows.push(row);
    }

    // Apply QUALIFY filter (filter on window function results)
    let finalRows = resultRows;
    if (ast.qualify) {
        finalRows = [];
        const qualifyColMap = {};
        resultColumns.forEach((name, idx) => { qualifyColMap[name.toLowerCase()] = idx; });

        for (let i = 0; i < resultRows.length; i++) {
            const rowData = {};
            for (let c = 0; c < resultColumns.length; c++) {
                rowData[resultColumns[c].toLowerCase()] = resultRows[i][c];
            }

            if (executor._evaluateInMemoryExpr(ast.qualify, rowData, 0)) {
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
                if (colIdx === undefined) {
                    console.warn(`[SQLExecutor] ORDER BY column '${ob.column}' not found in result columns`);
                    continue;
                }
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
