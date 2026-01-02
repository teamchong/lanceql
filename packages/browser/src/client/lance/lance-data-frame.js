/**
 * LanceData - DataFrame query builder
 * Extracted from lance-data.js for modularity
 */

// Forward declaration for LanceFile.Op enum
let LanceFile;

/**
 * DataFrame query builder with Immer-style immutability.
 * Supports filter, select, limit operations.
 */
class DataFrame {
    constructor(file) {
        this.file = file;
        this._filterOps = [];  // Array of {colIdx, op, value, type, opStr}
        this._selectCols = null;
        this._limitValue = null;
        this._isRemote = file._isRemote || file.baseUrl !== undefined;
    }

    /**
     * Filter rows where column matches condition.
     * Immer-style: returns new DataFrame, original unchanged.
     * @param {number} colIdx - Column index
     * @param {string} op - Operator: '=', '!=', '<', '<=', '>', '>='
     * @param {number|bigint|string} value - Value to compare
     * @param {string} type - 'int64', 'float64', or 'string'
     * @returns {DataFrame}
     */
    filter(colIdx, op, value, type = 'int64') {
        const opMap = {
            '=': LanceFile?.Op?.EQ ?? 0, '==': LanceFile?.Op?.EQ ?? 0,
            '!=': LanceFile?.Op?.NE ?? 1, '<>': LanceFile?.Op?.NE ?? 1,
            '<': LanceFile?.Op?.LT ?? 2,
            '<=': LanceFile?.Op?.LE ?? 3,
            '>': LanceFile?.Op?.GT ?? 4,
            '>=': LanceFile?.Op?.GE ?? 5
        };

        const df = new DataFrame(this.file);
        df._filterOps = [...this._filterOps, { colIdx, op: opMap[op], opStr: op, value, type }];
        df._selectCols = this._selectCols;
        df._limitValue = this._limitValue;
        df._isRemote = this._isRemote;
        return df;
    }

    /**
     * Select specific columns.
     * Immer-style: returns new DataFrame, original unchanged.
     * @param {...number} colIndices - Column indices to select
     * @returns {DataFrame}
     */
    select(...colIndices) {
        const cols = Array.isArray(colIndices[0]) ? colIndices[0] : colIndices;
        const df = new DataFrame(this.file);
        df._filterOps = [...this._filterOps];
        df._selectCols = cols;
        df._limitValue = this._limitValue;
        df._isRemote = this._isRemote;
        return df;
    }

    /**
     * Limit number of results.
     * Immer-style: returns new DataFrame, original unchanged.
     * @param {number} n - Maximum rows
     * @returns {DataFrame}
     */
    limit(n) {
        const df = new DataFrame(this.file);
        df._filterOps = [...this._filterOps];
        df._selectCols = this._selectCols;
        df._limitValue = n;
        df._isRemote = this._isRemote;
        return df;
    }

    /**
     * Generate SQL from DataFrame operations.
     * @returns {string}
     */
    toSQL() {
        const colNames = this.file.columnNames || this.file._schema?.map(s => s.name) ||
            Array.from({ length: this.file._numColumns || 6 }, (_, i) => `col_${i}`);

        // SELECT clause
        let selectClause;
        if (this._selectCols && this._selectCols.length > 0) {
            selectClause = this._selectCols.map(i => colNames[i] || `col_${i}`).join(', ');
        } else {
            selectClause = '*';
        }

        // WHERE clause
        let whereClause = '';
        if (this._filterOps.length > 0) {
            const conditions = this._filterOps.map(f => {
                const colName = colNames[f.colIdx] || `col_${f.colIdx}`;
                const val = f.type === 'string' ? `'${f.value}'` : f.value;
                return `${colName} ${f.opStr} ${val}`;
            });
            whereClause = ` WHERE ${conditions.join(' AND ')}`;
        }

        // LIMIT clause
        const limitClause = this._limitValue ? ` LIMIT ${this._limitValue}` : '';

        return `SELECT ${selectClause} FROM dataset${whereClause}${limitClause}`;
    }

    /**
     * Execute the query and return row indices (sync, local only).
     * @returns {Uint32Array}
     */
    collectIndices() {
        if (this._isRemote) {
            throw new Error('collectIndices() is sync-only. Use collect() for remote datasets.');
        }

        let indices = null;

        // Apply filters
        for (const f of this._filterOps) {
            let newIndices;
            if (f.type === 'int64') {
                newIndices = this.file.filterInt64(f.colIdx, f.op, f.value);
            } else {
                newIndices = this.file.filterFloat64(f.colIdx, f.op, f.value);
            }

            if (indices === null) {
                indices = newIndices;
            } else {
                // Intersect indices
                const set = new Set(newIndices);
                indices = indices.filter(i => set.has(i));
                indices = new Uint32Array(indices);
            }
        }

        // If no filters, get all row indices
        if (indices === null) {
            const rowCount = Number(this.file.getRowCount(0));
            indices = new Uint32Array(rowCount);
            for (let i = 0; i < rowCount; i++) indices[i] = i;
        }

        // Apply limit
        if (this._limitValue !== null && indices.length > this._limitValue) {
            indices = indices.slice(0, this._limitValue);
        }

        return indices;
    }

    /**
     * Execute the query and return results.
     * Works with both local (sync) and remote (async) datasets.
     * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
     */
    async collect() {
        // Remote: generate SQL and execute
        if (this._isRemote) {
            const sql = this.toSQL();
            return await this.file.executeSQL(sql);
        }

        // Local: use sync WASM methods
        const indices = this.collectIndices();
        const cols = this._selectCols ||
            Array.from({ length: this.file.numColumns }, (_, i) => i);

        const columns = [];
        const columnNames = [];

        for (const colIdx of cols) {
            columnNames.push(this.file.columnNames?.[colIdx] || `col_${colIdx}`);
            // Try int64 first, then float64
            try {
                columns.push(Array.from(this.file.readInt64AtIndices(colIdx, indices)));
            } catch {
                try {
                    columns.push(Array.from(this.file.readFloat64AtIndices(colIdx, indices)));
                } catch {
                    columns.push(indices.map(() => null));
                }
            }
        }

        return {
            columns,
            columnNames,
            total: indices.length,
            _indices: indices
        };
    }

    /**
     * Count matching rows.
     * @returns {Promise<number>|number}
     */
    async count() {
        if (this._isRemote) {
            const result = await this.collect();
            return result.columns[0]?.length || 0;
        }
        return this.collectIndices().length;
    }
}


export { DataFrame };
