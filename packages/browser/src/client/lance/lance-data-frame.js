/**
 * LanceData - DataFrame query builder
 * Extracted from lance-data.js for modularity
 *
 * API aligned with CLI DataFrame (src/dataframe.zig) for consistency.
 */

// Forward declaration for LanceFile.Op enum
let LanceFile;

/**
 * DataFrame query builder with Immer-style immutability.
 * Supports filter, select, limit, orderBy, offset, distinct, groupBy operations.
 */
class DataFrame {
    constructor(file) {
        this.file = file;
        this._filterOps = [];  // Array of {colIdx, op, value, type, opStr}
        this._selectCols = null;
        this._limitValue = null;
        this._offsetValue = null;
        this._orderByCols = [];  // Array of {colIdx, descending}
        this._distinct = false;
        this._groupByCols = null;  // Array of column indices
        this._aggregates = [];  // Array of {func, colIdx, alias}
        this._isRemote = file._isRemote || file.baseUrl !== undefined;
    }

    /**
     * Clone this DataFrame with all state preserved.
     * @returns {DataFrame}
     */
    _clone() {
        const df = new DataFrame(this.file);
        df._filterOps = [...this._filterOps];
        df._selectCols = this._selectCols;
        df._limitValue = this._limitValue;
        df._offsetValue = this._offsetValue;
        df._orderByCols = [...this._orderByCols];
        df._distinct = this._distinct;
        df._groupByCols = this._groupByCols;
        df._aggregates = [...this._aggregates];
        df._isRemote = this._isRemote;
        return df;
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

        const df = this._clone();
        df._filterOps.push({ colIdx, op: opMap[op], opStr: op, value, type });
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
        const df = this._clone();
        df._selectCols = cols;
        return df;
    }

    /**
     * Limit number of results.
     * @param {number} n - Maximum rows
     * @returns {DataFrame}
     */
    limit(n) {
        const df = this._clone();
        df._limitValue = n;
        return df;
    }

    /**
     * Offset results (skip first n rows).
     * @param {number} n - Rows to skip
     * @returns {DataFrame}
     */
    offset(n) {
        const df = this._clone();
        df._offsetValue = n;
        return df;
    }

    /**
     * Order by column.
     * @param {number} colIdx - Column index
     * @param {boolean} descending - Sort descending (default: false)
     * @returns {DataFrame}
     */
    orderBy(colIdx, descending = false) {
        const df = this._clone();
        df._orderByCols.push({ colIdx, descending });
        return df;
    }

    /**
     * Select distinct rows.
     * @returns {DataFrame}
     */
    distinct() {
        const df = this._clone();
        df._distinct = true;
        return df;
    }

    /**
     * Group by columns.
     * @param {...number} colIndices - Column indices to group by
     * @returns {GroupedFrame}
     */
    groupBy(...colIndices) {
        const cols = Array.isArray(colIndices[0]) ? colIndices[0] : colIndices;
        return new GroupedFrame(this, cols);
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
        if (this._aggregates.length > 0) {
            // GROUP BY with aggregates
            const parts = [];
            if (this._groupByCols) {
                parts.push(...this._groupByCols.map(i => colNames[i] || `col_${i}`));
            }
            parts.push(...this._aggregates.map(a => {
                const col = a.colIdx !== null ? (colNames[a.colIdx] || `col_${a.colIdx}`) : '*';
                const alias = a.alias ? ` AS ${a.alias}` : '';
                return `${a.func}(${col})${alias}`;
            }));
            selectClause = parts.join(', ');
        } else if (this._selectCols && this._selectCols.length > 0) {
            selectClause = this._selectCols.map(i => colNames[i] || `col_${i}`).join(', ');
        } else {
            selectClause = '*';
        }

        // DISTINCT
        const distinctClause = this._distinct ? 'DISTINCT ' : '';

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

        // GROUP BY clause
        let groupByClause = '';
        if (this._groupByCols && this._groupByCols.length > 0) {
            groupByClause = ` GROUP BY ${this._groupByCols.map(i => colNames[i] || `col_${i}`).join(', ')}`;
        }

        // ORDER BY clause
        let orderByClause = '';
        if (this._orderByCols.length > 0) {
            const orders = this._orderByCols.map(o => {
                const colName = colNames[o.colIdx] || `col_${o.colIdx}`;
                return o.descending ? `${colName} DESC` : colName;
            });
            orderByClause = ` ORDER BY ${orders.join(', ')}`;
        }

        // LIMIT clause
        const limitClause = this._limitValue ? ` LIMIT ${this._limitValue}` : '';

        // OFFSET clause
        const offsetClause = this._offsetValue ? ` OFFSET ${this._offsetValue}` : '';

        return `SELECT ${distinctClause}${selectClause} FROM dataset${whereClause}${groupByClause}${orderByClause}${limitClause}${offsetClause}`;
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


/**
 * GroupedFrame - DataFrame with GROUP BY columns set.
 * Returned by DataFrame.groupBy(), provides aggregate methods.
 */
class GroupedFrame {
    constructor(df, groupByCols) {
        this._df = df;
        this._groupByCols = groupByCols;
    }

    /**
     * Apply aggregates to grouped data.
     * @param {Array<{func: string, colIdx: number|null, alias?: string}>} specs - Aggregate specifications
     * @returns {DataFrame}
     */
    agg(specs) {
        const df = this._df._clone();
        df._groupByCols = this._groupByCols;
        df._aggregates = specs;
        return df;
    }

    /**
     * Count rows per group.
     * @param {string} [alias] - Optional alias for the count column
     * @returns {DataFrame}
     */
    count(alias = 'count') {
        return this.agg([{ func: 'COUNT', colIdx: null, alias }]);
    }

    /**
     * Sum a column per group.
     * @param {number} colIdx - Column index to sum
     * @param {string} [alias] - Optional alias
     * @returns {DataFrame}
     */
    sum(colIdx, alias) {
        return this.agg([{ func: 'SUM', colIdx, alias }]);
    }

    /**
     * Average a column per group.
     * @param {number} colIdx - Column index to average
     * @param {string} [alias] - Optional alias
     * @returns {DataFrame}
     */
    avg(colIdx, alias) {
        return this.agg([{ func: 'AVG', colIdx, alias }]);
    }

    /**
     * Min of a column per group.
     * @param {number} colIdx - Column index
     * @param {string} [alias] - Optional alias
     * @returns {DataFrame}
     */
    min(colIdx, alias) {
        return this.agg([{ func: 'MIN', colIdx, alias }]);
    }

    /**
     * Max of a column per group.
     * @param {number} colIdx - Column index
     * @param {string} [alias] - Optional alias
     * @returns {DataFrame}
     */
    max(colIdx, alias) {
        return this.agg([{ func: 'MAX', colIdx, alias }]);
    }
}


export { DataFrame, GroupedFrame };
