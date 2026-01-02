/**
 * RemoteLanceDataset - SQL execution helpers
 * Extracted from remote-dataset.js for modularity
 */

/**
 * Simple SQL parser for basic queries.
 * @param {string} sql - SQL query string
 * @returns {Object} Parsed AST
 */
export function parseSQL(sql) {
    const ast = { type: 'SELECT', columns: '*', limit: null, offset: null, where: null };
    const upper = sql.toUpperCase();

    if (upper.includes('LIMIT')) {
        const match = sql.match(/LIMIT\s+(\d+)/i);
        if (match) ast.limit = parseInt(match[1]);
    }
    if (upper.includes('OFFSET')) {
        const match = sql.match(/OFFSET\s+(\d+)/i);
        if (match) ast.offset = parseInt(match[1]);
    }
    if (upper.includes('WHERE')) {
        ast.where = true;
    }

    return ast;
}

/**
 * Execute SQL query across all fragments in parallel.
 * @param {Object} dataset - Dataset instance
 * @param {string} sql - SQL query
 * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
 */
export async function executeSQL(dataset, sql) {
    // Parse the SQL to understand what's needed
    const ast = parseSQL(sql);

    // For simple SELECT * with LIMIT, use readRows
    if (ast.type === 'SELECT' && ast.columns === '*' && !ast.where) {
        const limit = ast.limit || 50;
        const offset = ast.offset || 0;
        return await dataset.readRows({ offset, limit });
    }

    // For queries with WHERE or complex operations, execute on each fragment in parallel
    const fetchPromises = dataset._fragments.map(async (frag, idx) => {
        const file = await dataset.openFragment(idx);
        try {
            return await file.executeSQL(sql);
        } catch (e) {
            console.warn(`Fragment ${idx} query failed:`, e);
            return { columns: [], columnNames: [], total: 0 };
        }
    });

    const results = await Promise.all(fetchPromises);

    // Merge results
    if (results.length === 0 || results.every(r => r.columns.length === 0)) {
        return { columns: [], columnNames: dataset.columnNames, total: 0 };
    }

    const firstValid = results.find(r => r.columns.length > 0);
    if (!firstValid) {
        return { columns: [], columnNames: dataset.columnNames, total: 0 };
    }

    const numCols = firstValid.columns.length;
    const colNames = firstValid.columnNames;
    const mergedColumns = Array.from({ length: numCols }, () => []);

    let totalRows = 0;
    for (const r of results) {
        for (let c = 0; c < numCols && c < r.columns.length; c++) {
            mergedColumns[c].push(...r.columns[c]);
        }
        totalRows += r.total;
    }

    // Apply LIMIT if present (after merging)
    if (ast.limit) {
        const offset = ast.offset || 0;
        for (let c = 0; c < numCols; c++) {
            mergedColumns[c] = mergedColumns[c].slice(offset, offset + ast.limit);
        }
    }

    return {
        columns: mergedColumns,
        columnNames: colNames,
        total: totalRows
    };
}
