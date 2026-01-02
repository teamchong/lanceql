/**
 * LanceData - sql.js-Compatible API
 * Extracted from lance-data.js for modularity
 *
 * Drop-in replacement for sql.js with vector search support.
 */

import { SQLLexer } from '../sql/lexer.js';
import { SQLParser } from '../sql/parser.js';
import { getWebGPUAccelerator } from '../gpu/accelerator.js';
import { GPUAggregator } from '../gpu/aggregator.js';
import { GPUJoiner } from '../gpu/joiner.js';
import { GPUSorter, getGPUSorter } from '../gpu/sorter.js';
import { GPUGrouper } from '../gpu/grouper.js';
import { GPUVectorSearch } from '../gpu/vector-search.js';

// Forward declarations
let LocalDatabase, opfsStorage;

// Singleton GPU instances
let gpuAggregator, gpuJoiner, gpuGrouper, gpuVectorSearch;

/**
 * Statement class - sql.js compatible prepared statement
 */
class Statement {
    constructor(db, sql) {
        this.db = db;
        this.sql = sql;
        this.params = null;
        this.results = null;
        this.resultIndex = 0;
        this.done = false;
    }

    bind(params) {
        this.params = params;
        this.results = null;
        this.resultIndex = 0;
        this.done = false;
        return true;
    }

    step() {
        if (this.done) return false;

        if (this.results === null) {
            const execResult = this.db.exec(this.sql, this.params);
            if (execResult.length === 0 || execResult[0].values.length === 0) {
                this.done = true;
                return false;
            }
            this.results = execResult[0];
            this.resultIndex = 0;
        }

        if (this.resultIndex >= this.results.values.length) {
            this.done = true;
            return false;
        }

        return true;
    }

    get() {
        if (!this.results || this.resultIndex >= this.results.values.length) {
            return [];
        }
        const row = this.results.values[this.resultIndex];
        this.resultIndex++;
        return row;
    }

    getAsObject(params) {
        if (!this.results || this.resultIndex >= this.results.values.length) {
            return {};
        }
        const row = this.results.values[this.resultIndex];
        this.resultIndex++;

        const obj = {};
        this.results.columns.forEach((col, i) => {
            obj[col] = row[i];
        });
        return obj;
    }

    getColumnNames() {
        return this.results?.columns || [];
    }

    reset() {
        this.results = null;
        this.resultIndex = 0;
        this.done = false;
        return true;
    }

    free() {
        this.results = null;
        this.params = null;
        return true;
    }

    freemem() {
        return this.free();
    }
}

/**
 * Database class - sql.js compatible API with vector search
 */
class Database {
    constructor(nameOrData, storage = null) {
        if (nameOrData instanceof Uint8Array) {
            this._inMemory = true;
            this._data = nameOrData;
            this._name = ':memory:';
            this._db = null;
        } else {
            this._inMemory = false;
            this._name = nameOrData || 'default';
            this._storage = storage;
            this._db = null;
            this._pendingInit = true;
        }
        this._open = false;
        this._rowsModified = 0;
    }

    async _ensureOpen() {
        if (!this._open) {
            if (!LocalDatabase) {
                const dbModule = await import('../database/local-database.js');
                LocalDatabase = dbModule.LocalDatabase;
            }
            if (!opfsStorage) {
                const opfsModule = await import('../storage/opfs.js');
                opfsStorage = opfsModule.opfsStorage;
            }

            if (!this._db && !this._inMemory) {
                this._db = new LocalDatabase(this._name, this._storage || opfsStorage);
            }

            if (this._db) {
                await this._db.open();
            }
            this._open = true;
        }
    }

    exec(sql, params) {
        return this._execAsync(sql, params);
    }

    async _execAsync(sql, params) {
        await this._ensureOpen();

        let processedSql = sql;
        if (params) {
            if (Array.isArray(params)) {
                let paramIndex = 0;
                processedSql = sql.replace(/\?/g, () => {
                    const val = params[paramIndex++];
                    return this._formatValue(val);
                });
            } else if (typeof params === 'object') {
                for (const [key, val] of Object.entries(params)) {
                    const pattern = new RegExp(`[:$@]${key}\\b`, 'g');
                    processedSql = processedSql.replace(pattern, this._formatValue(val));
                }
            }
        }

        const statements = processedSql
            .split(';')
            .map(s => s.trim())
            .filter(s => s.length > 0);

        const results = [];

        for (const stmt of statements) {
            try {
                const lexer = new SQLLexer(stmt);
                const tokens = lexer.tokenize();
                const parser = new SQLParser(tokens);
                const ast = parser.parse();

                if (ast.type === 'SELECT') {
                    const rows = await this._db._executeAST(ast);
                    if (rows && rows.length > 0) {
                        const columns = Object.keys(rows[0]);
                        const values = rows.map(row => columns.map(c => row[c]));
                        results.push({ columns, values });
                    }
                    this._rowsModified = 0;
                } else {
                    const result = await this._db._executeAST(ast);
                    this._rowsModified = result?.inserted || result?.updated || result?.deleted || 0;
                }
            } catch (e) {
                throw new Error(`SQL error: ${e.message}\nStatement: ${stmt}`);
            }
        }

        return results;
    }

    run(sql, params) {
        return this._runAsync(sql, params);
    }

    async _runAsync(sql, params) {
        await this.exec(sql, params);
        return this;
    }

    prepare(sql, params) {
        const stmt = new Statement(this, sql);
        if (params) {
            stmt.bind(params);
        }
        return stmt;
    }

    each(sql, params, callback, done) {
        this._eachAsync(sql, params, callback, done);
        return this;
    }

    async _eachAsync(sql, params, callback, done) {
        try {
            const results = await this.exec(sql, params);
            if (results.length > 0) {
                const { columns, values } = results[0];
                for (const row of values) {
                    const obj = {};
                    columns.forEach((col, i) => {
                        obj[col] = row[i];
                    });
                    callback(obj);
                }
            }
            if (done) done();
        } catch (e) {
            if (done) done(e);
            else throw e;
        }
    }

    getRowsModified() {
        return this._rowsModified;
    }

    async export() {
        if (this._inMemory && this._data) {
            return this._data;
        }

        await this._ensureOpen();

        const exportData = {
            version: this._db?.version,
            tables: {}
        };

        if (this._db) {
            for (const tableName of this._db.listTables()) {
                const table = this._db.getTable(tableName);
                const rows = await this._db.select(tableName, {});
                exportData.tables[tableName] = {
                    schema: table?.schema,
                    rows
                };
            }
        }

        return new TextEncoder().encode(JSON.stringify(exportData));
    }

    close() {
        this._open = false;
        this._db = null;
    }

    create_function(name, func) {
        console.warn(`[LanceQL] create_function('${name}') not yet implemented`);
        return this;
    }

    create_aggregate(name, funcs) {
        console.warn(`[LanceQL] create_aggregate('${name}') not yet implemented`);
        return this;
    }

    _formatValue(val) {
        if (val === null || val === undefined) {
            return 'NULL';
        }
        if (typeof val === 'string') {
            return `'${val.replace(/'/g, "''")}'`;
        }
        if (typeof val === 'number') {
            return String(val);
        }
        if (typeof val === 'boolean') {
            return val ? 'TRUE' : 'FALSE';
        }
        if (Array.isArray(val)) {
            return `'[${val.join(',')}]'`;
        }
        return String(val);
    }
}

/**
 * Initialize LanceQL with sql.js-compatible API
 *
 * @param {Object} config - Configuration
 * @returns {Promise<{Database: class}>}
 */
async function initSqlJs(config = {}) {
    // Initialize OPFS storage
    try {
        if (!opfsStorage) {
            const opfsModule = await import('../storage/opfs.js');
            opfsStorage = opfsModule.opfsStorage;
        }
        await opfsStorage.open();
    } catch (e) {
        console.warn('[LanceQL] OPFS not available:', e.message);
    }

    // Initialize WebGPU
    try {
        const accelerator = getWebGPUAccelerator();
        await accelerator.init();

        if (!gpuAggregator) gpuAggregator = new GPUAggregator();
        if (!gpuJoiner) gpuJoiner = new GPUJoiner();
        if (!gpuGrouper) gpuGrouper = new GPUGrouper();
        if (!gpuVectorSearch) gpuVectorSearch = new GPUVectorSearch();

        await gpuAggregator.init();
        await gpuJoiner.init();
        await getGPUSorter().init();
        await gpuGrouper.init();
        await gpuVectorSearch.init();
    } catch (e) {
        console.warn('[LanceQL] WebGPU not available:', e.message);
    }

    return {
        Database,
        Statement,
    };
}


export { Statement, Database, initSqlJs };
export { Database as SqlJsDatabase, Statement as SqlJsStatement };
