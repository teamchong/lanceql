/**
 * Vault - Encrypted key-value store and Table reference
 */
import { workerRPC } from '../rpc/worker-rpc.js';
import { PureLanceWriter } from '../storage/lance-writer.js';
import { openExplorer } from '../explorer/index.js';

class Vault {
    /**
     * @param {Function|null} getEncryptionKey - Async callback returning encryption key
     */
    constructor(getEncryptionKey = null) {
        this._getEncryptionKey = getEncryptionKey;
        this._encryptionKeyId = null;
        this._ready = false;
    }

    /**
     * Initialize the vault (connects to SharedWorker).
     * @returns {Promise<Vault>}
     */
    async _init() {
        if (this._ready) return this;

        // If encryption is enabled, derive key and send to worker
        let encryptionConfig = null;
        if (this._getEncryptionKey) {
            const key = await this._getEncryptionKey();
            this._encryptionKeyId = `vault:${Date.now()}`;

            // Convert key to raw bytes if needed
            let keyBytes;
            if (key instanceof CryptoKey) {
                keyBytes = await crypto.subtle.exportKey('raw', key);
            } else if (key instanceof ArrayBuffer || key instanceof Uint8Array) {
                keyBytes = key instanceof Uint8Array ? key : new Uint8Array(key);
            } else if (typeof key === 'string') {
                // Hash string to get 256-bit key
                const encoder = new TextEncoder();
                const data = encoder.encode(key);
                const hash = await crypto.subtle.digest('SHA-256', data);
                keyBytes = new Uint8Array(hash);
            } else {
                throw new Error('Encryption key must be CryptoKey, ArrayBuffer, Uint8Array, or string');
            }

            encryptionConfig = {
                keyId: this._encryptionKeyId,
                keyBytes: Array.from(keyBytes instanceof Uint8Array ? keyBytes : new Uint8Array(keyBytes))
            };
        }

        await workerRPC('vault:open', { encryption: encryptionConfig });
        this._ready = true;
        return this;
    }

    // =========================================================================
    // KV Operations (stored in encrypted JSON file)
    // =========================================================================

    /**
     * Get a value by key.
     * @param {string} key
     * @returns {Promise<any>} The stored value, or undefined if not found
     */
    async get(key) {
        return workerRPC('vault:get', { key });
    }

    /**
     * Set a value. Accepts any JSON-serializable value.
     * @param {string} key
     * @param {any} value
     * @returns {Promise<void>}
     */
    async set(key, value) {
        await workerRPC('vault:set', { key, value });
    }

    /**
     * Delete a key.
     * @param {string} key
     * @returns {Promise<boolean>} True if key existed
     */
    async delete(key) {
        return workerRPC('vault:delete', { key });
    }

    /**
     * List all keys.
     * @returns {Promise<string[]>}
     */
    async keys() {
        return workerRPC('vault:keys', {});
    }

    /**
     * Check if a key exists.
     * @param {string} key
     * @returns {Promise<boolean>}
     */
    async has(key) {
        const value = await this.get(key);
        return value !== undefined;
    }

    // =========================================================================
    // SQL Operations (tables in Lance format)
    // =========================================================================

    /**
     * Execute a SQL statement.
     * @param {string} sql - SQL statement
     * @returns {Promise<any>} Query results or affected row count
     *
     * @example
     * await v.exec('CREATE TABLE users (id INT, name TEXT, embedding VECTOR(384))');
     * await v.exec('INSERT INTO users VALUES (1, "Alice", [...])');
     * const results = await v.exec('SELECT * FROM users WHERE name NEAR "alice"');
     */
    async exec(sql) {
        return workerRPC('vault:exec', { sql });
    }

    /**
     * Load the MiniLM model into the worker for CREATE VECTOR INDEX operations.
     * @returns {Promise<{loaded: boolean, cached: boolean}>}
     */
    async loadMinilmModel() {
        return workerRPC('loadMinilmModel', {});
    }

    /**
     * Execute a SQL query and return results as array of objects.
     * @param {string} sql - SELECT statement
     * @returns {Promise<Object[]>} Array of row objects
     */
    async query(sql) {
        const result = await this.exec(sql);
        if (!result || !result.columns || !result.rows) return [];

        return result.rows.map(row => {
            const obj = {};
            result.columns.forEach((col, i) => {
                obj[col] = row[i];
            });
            return obj;
        });
    }

    // =========================================================================
    // DataFrame Operations
    // =========================================================================

    /**
     * Get a DataFrame reference to a table.
     * @param {string} name - Table name
     * @returns {TableRef} DataFrame-style query builder
     */
    table(name) {
        return new TableRef(this, name);
    }

    /**
     * List all tables.
     * @returns {Promise<string[]>}
     */
    async tables() {
        return workerRPC('vault:tables', {});
    }

    /**
     * Execute a SQL query and return a WritableDataFrame.
     * @param {string} sql - SELECT statement
     * @returns {Promise<WritableDataFrame>} DataFrame with query results
     *
     * @example
     * const df = await v.df('SELECT * FROM users WHERE active = 1');
     * await df.saveTo(v, 'active_users');
     */
    async df(sql) {
        const result = await this.exec(sql);
        if (!result || !result.columns || !result.rows) {
            return new WritableDataFrame({}, []);
        }

        const columns = {};
        const columnNames = result.columns;

        // Initialize column arrays
        for (const name of columnNames) {
            columns[name] = [];
        }

        // Fill column arrays from rows
        for (const row of result.rows) {
            for (let i = 0; i < columnNames.length; i++) {
                const name = columnNames[i];
                columns[name].push(row[name] !== undefined ? row[name] : row[i]);
            }
        }

        return new WritableDataFrame(columns, columnNames);
    }

    // =========================================================================
    // Export Operations
    // =========================================================================

    /**
     * Export a table to Lance format bytes.
     * @param {string} tableName - Name of the table to export
     * @returns {Promise<Uint8Array>} Lance file bytes
     */
    async exportToLance(tableName) {
        // Get table schema
        const schemaResult = await this.exec(`SELECT * FROM ${tableName} LIMIT 0`);
        if (!schemaResult || !schemaResult.columns) {
            throw new Error(`Table '${tableName}' not found or empty`);
        }

        // Get all data
        const dataResult = await this.exec(`SELECT * FROM ${tableName}`);
        if (!dataResult || !dataResult.rows || dataResult.rows.length === 0) {
            throw new Error(`Table '${tableName}' is empty`);
        }

        // Create Lance writer
        const writer = new PureLanceWriter();

        // Infer column types and add to writer
        const columns = dataResult.columns;
        const rows = dataResult.rows;

        for (let colIdx = 0; colIdx < columns.length; colIdx++) {
            const colName = columns[colIdx];
            const values = rows.map(row => row[colName] !== undefined ? row[colName] : row[colIdx]);

            // Infer type from first non-null value
            const firstValue = values.find(v => v !== null && v !== undefined);

            if (firstValue === undefined) {
                // All nulls - default to string
                writer.addStringColumn(colName, values.map(v => v === null ? '' : String(v)));
            } else if (typeof firstValue === 'bigint') {
                writer.addInt64Column(colName, BigInt64Array.from(values.map(v => v === null ? 0n : BigInt(v))));
            } else if (typeof firstValue === 'number') {
                if (Number.isInteger(firstValue) && firstValue <= 2147483647 && firstValue >= -2147483648) {
                    writer.addInt32Column(colName, Int32Array.from(values.map(v => v === null ? 0 : v)));
                } else {
                    writer.addFloat64Column(colName, Float64Array.from(values.map(v => v === null ? 0 : v)));
                }
            } else if (typeof firstValue === 'boolean') {
                writer.addBoolColumn(colName, values.map(v => v === null ? false : v));
            } else if (Array.isArray(firstValue)) {
                // Vector column
                const dim = firstValue.length;
                const flat = new Float32Array(values.length * dim);
                for (let i = 0; i < values.length; i++) {
                    const vec = values[i] || new Array(dim).fill(0);
                    for (let j = 0; j < dim; j++) {
                        flat[i * dim + j] = vec[j] || 0;
                    }
                }
                writer.addVectorColumn(colName, flat, dim);
            } else {
                // String column
                writer.addStringColumn(colName, values.map(v => v === null ? '' : String(v)));
            }
        }

        return writer.finalize();
    }

    /**
     * Upload binary data to a URL using PUT.
     * @param {Uint8Array} data - Binary data to upload
     * @param {string} url - Signed URL for upload
     * @param {Object} [options] - Upload options
     * @param {Function} [options.onProgress] - Progress callback (loaded, total)
     * @returns {Promise<Response>} Fetch response
     */
    async uploadToUrl(data, url, options = {}) {
        const { onProgress } = options;

        if (onProgress && typeof XMLHttpRequest !== 'undefined') {
            // Use XHR for progress support
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open('PUT', url, true);
                xhr.setRequestHeader('Content-Type', 'application/octet-stream');

                xhr.upload.onprogress = (e) => {
                    if (e.lengthComputable) {
                        onProgress(e.loaded, e.total);
                    }
                };

                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve({ ok: true, status: xhr.status });
                    } else {
                        reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
                    }
                };

                xhr.onerror = () => reject(new Error('Upload failed: network error'));
                xhr.send(data);
            });
        }

        // Simple fetch for no progress
        const response = await fetch(url, {
            method: 'PUT',
            body: data,
            headers: {
                'Content-Type': 'application/octet-stream',
            },
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
        }

        return response;
    }

    /**
     * Export a table to Lance format and upload to a signed URL.
     * @param {string} tableName - Name of the table to export
     * @param {string} signedUrl - Pre-signed URL for upload (S3/R2/GCS)
     * @param {Object} [options] - Export options
     * @param {Function} [options.onProgress] - Progress callback (loaded, total)
     * @returns {Promise<{size: number, url: string}>} Upload result
     */
    async exportToRemote(tableName, signedUrl, options = {}) {
        // Export to Lance bytes
        const lanceBytes = await this.exportToLance(tableName);

        // Upload to signed URL
        await this.uploadToUrl(lanceBytes, signedUrl, options);

        return {
            size: lanceBytes.length,
            url: signedUrl.split('?')[0], // Return URL without query params
        };
    }

    // =========================================================================
    // Developer Tools
    // =========================================================================

    /**
     * Open the Data Explorer in a new window.
     *
     * DevTools-style debugging UI with:
     * - SQL editor
     * - DataView (virtualized table)
     * - Timeline (version history)
     *
     * @param {Object} [options] - Explorer options
     * @param {string} [options.table] - Pre-select table
     * @param {number} [options.version] - Start at specific version
     * @param {number} [options.width=1200] - Window width
     * @param {number} [options.height=800] - Window height
     * @returns {Object} Controller for the explorer window
     *
     * @example
     * const v = await vault();
     * v.explorer(); // Opens debug window
     *
     * // With options
     * v.explorer({ table: 'users', width: 1400 });
     *
     * // Use controller
     * const exp = v.explorer();
     * exp.runSQL('SELECT * FROM users LIMIT 10');
     * exp.close();
     */
    explorer(options = {}) {
        return openExplorer(this, options);
    }

    // =========================================================================
    // Time Travel APIs
    // =========================================================================

    /**
     * Get version timeline for a table.
     * @param {string} table - Table name
     * @returns {Promise<Array>} Version history with deltas
     *
     * @example
     * const history = await v.timeline('users');
     * // [{ version: 3, timestamp: ..., operation: 'INSERT', delta: '+5' }, ...]
     */
    async timeline(table) {
        const result = await this.exec(`SHOW VERSIONS FOR ${table}`);
        return result?.rows?.map(row => ({
            version: row[0],
            timestamp: row[1],
            operation: row[2],
            rowCount: row[3],
            delta: row[4]
        })) || [];
    }

    /**
     * Get diff between two versions.
     * @param {string} table - Table name
     * @param {Object} options - Diff options
     * @param {number} options.from - From version
     * @param {number} [options.to] - To version (defaults to HEAD)
     * @param {number} [options.limit=100] - Max rows
     * @returns {Promise<Object>} Diff result with added/deleted rows
     *
     * @example
     * const diff = await v.diff('users', { from: 2, to: 3 });
     * // { added: [...], deleted: [...], fromVersion: 2, toVersion: 3 }
     */
    async diff(table, { from, to, limit = 100 } = {}) {
        let sql = `DIFF ${table} VERSION ${from}`;
        if (to !== undefined) sql += ` AND VERSION ${to}`;
        sql += ` LIMIT ${limit}`;

        const result = await this.exec(sql);
        return {
            added: result?.added || [],
            deleted: result?.deleted || [],
            fromVersion: result?.from_version || from,
            toVersion: result?.to_version || to || 'HEAD'
        };
    }

    /**
     * Get what changed in the last version.
     * @param {string} table - Table name
     * @returns {Promise<Object>} Last change diff
     */
    async lastChange(table) {
        return this.diff(table, { from: -1 });
    }
}

/**
 * TableRef - DataFrame-style query builder for vault tables.
 */
class TableRef {
    constructor(vault, tableName) {
        this._vault = vault;
        this._tableName = tableName;
        this._filters = [];
        this._similar = null;
        this._selectCols = null;
        this._limitValue = null;
        this._orderBy = null;
    }

    /**
     * Filter rows by condition.
     * @param {string} column - Column name
     * @param {string} op - Operator ('=', '!=', '<', '<=', '>', '>=')
     * @param {any} value - Value to compare
     * @returns {TableRef} New TableRef with filter applied
     */
    filter(column, op, value) {
        const ref = this._clone();
        ref._filters.push({ column, op, value });
        return ref;
    }

    /**
     * Semantic similarity search.
     * @param {string} column - Column name (text or vector)
     * @param {string} text - Search text
     * @param {number} limit - Max results (default 20)
     * @returns {TableRef} New TableRef with similarity search
     */
    similar(column, text, limit = 20) {
        const ref = this._clone();
        ref._similar = { column, text, limit };
        return ref;
    }

    /**
     * Select specific columns.
     * @param {...string} columns - Column names
     * @returns {TableRef} New TableRef with columns selected
     */
    select(...columns) {
        const ref = this._clone();
        ref._selectCols = columns.flat();
        return ref;
    }

    /**
     * Limit number of results.
     * @param {number} n - Max rows
     * @returns {TableRef} New TableRef with limit
     */
    limit(n) {
        const ref = this._clone();
        ref._limitValue = n;
        return ref;
    }

    /**
     * Order results.
     * @param {string} column - Column to order by
     * @param {string} direction - 'ASC' or 'DESC'
     * @returns {TableRef} New TableRef with ordering
     */
    orderBy(column, direction = 'ASC') {
        const ref = this._clone();
        ref._orderBy = { column, direction };
        return ref;
    }

    /**
     * Execute query and return results as array of objects.
     * @returns {Promise<Object[]>}
     */
    async toArray() {
        const sql = this._toSQL();
        return this._vault.query(sql);
    }

    /**
     * Execute query and return first result.
     * @returns {Promise<Object|null>}
     */
    async first() {
        const ref = this._clone();
        ref._limitValue = 1;
        const results = await ref.toArray();
        return results[0] || null;
    }

    /**
     * Count matching rows.
     * @returns {Promise<number>}
     */
    async count() {
        const sql = this._toSQL(true);
        const result = await this._vault.exec(sql);
        return result?.rows?.[0]?.[0] || 0;
    }

    /**
     * Generate SQL from this query.
     * @param {boolean} countOnly - Generate COUNT(*) query
     * @returns {string}
     */
    _toSQL(countOnly = false) {
        const cols = countOnly ? 'COUNT(*)' : (this._selectCols?.join(', ') || '*');
        let sql = `SELECT ${cols} FROM ${this._tableName}`;

        const whereClauses = [];

        // Add filter conditions
        for (const f of this._filters) {
            const val = typeof f.value === 'string' ? `'${f.value}'` : f.value;
            whereClauses.push(`${f.column} ${f.op} ${val}`);
        }

        // Add NEAR condition for vector similarity search
        if (this._similar) {
            whereClauses.push(`${this._similar.column} NEAR '${this._similar.text}'`);
        }

        if (whereClauses.length > 0) {
            sql += ' WHERE ' + whereClauses.join(' AND ');
        }

        if (this._orderBy && !countOnly) {
            sql += ` ORDER BY ${this._orderBy.column} ${this._orderBy.direction}`;
        }

        // Use similar limit if set, otherwise use explicit limit
        const limit = this._similar?.limit || this._limitValue;
        if (limit && !countOnly) {
            sql += ` LIMIT ${limit}`;
        }

        return sql;
    }

    _clone() {
        const ref = new TableRef(this._vault, this._tableName);
        ref._filters = [...this._filters];
        ref._similar = this._similar;
        ref._selectCols = this._selectCols ? [...this._selectCols] : null;
        ref._limitValue = this._limitValue;
        ref._orderBy = this._orderBy;
        return ref;
    }
}

/**
 * WritableDataFrame - DataFrame for creating and saving data to vault tables.
 *
 * @example
 * const df = WritableDataFrame.fromRecords([
 *   { id: 1, name: 'Alice' },
 *   { id: 2, name: 'Bob' }
 * ]);
 * await df.saveTo(vault, 'users');
 */
class WritableDataFrame {
    /**
     * Create a WritableDataFrame with columnar data.
     * @param {Object} columns - Column name -> array of values
     * @param {string[]} columnNames - Column names in order
     */
    constructor(columns, columnNames) {
        this._columns = columns;
        this._columnNames = columnNames;
        this._rowCount = columnNames.length > 0 ? (columns[columnNames[0]]?.length || 0) : 0;
    }

    /**
     * Number of rows in the DataFrame.
     */
    get rowCount() {
        return this._rowCount;
    }

    /**
     * Column names.
     */
    get columns() {
        return [...this._columnNames];
    }

    /**
     * Create a DataFrame from an array of record objects.
     * @param {Object[]} records - Array of row objects
     * @returns {WritableDataFrame}
     *
     * @example
     * const df = WritableDataFrame.fromRecords([
     *   { id: 1, name: 'Alice', age: 30 },
     *   { id: 2, name: 'Bob', age: 25 }
     * ]);
     */
    static fromRecords(records) {
        if (!records || records.length === 0) {
            return new WritableDataFrame({}, []);
        }

        // Get column names from first record
        const columnNames = Object.keys(records[0]);
        const columns = {};

        // Initialize column arrays
        for (const name of columnNames) {
            columns[name] = [];
        }

        // Fill column arrays
        for (const record of records) {
            for (const name of columnNames) {
                columns[name].push(record[name] !== undefined ? record[name] : null);
            }
        }

        return new WritableDataFrame(columns, columnNames);
    }

    /**
     * Create a DataFrame from columnar data.
     * @param {Object} columns - Column name -> array or TypedArray
     * @returns {WritableDataFrame}
     *
     * @example
     * const df = WritableDataFrame.fromColumns({
     *   id: [1, 2, 3],
     *   name: ['Alice', 'Bob', 'Charlie'],
     *   score: new Float64Array([0.9, 0.8, 0.7])
     * });
     */
    static fromColumns(columns) {
        const columnNames = Object.keys(columns);
        const normalizedColumns = {};

        for (const name of columnNames) {
            const data = columns[name];
            // Convert TypedArrays to regular arrays for uniform handling
            if (ArrayBuffer.isView(data) && !(data instanceof Uint8Array)) {
                normalizedColumns[name] = Array.from(data);
            } else {
                normalizedColumns[name] = Array.isArray(data) ? data : [data];
            }
        }

        return new WritableDataFrame(normalizedColumns, columnNames);
    }

    /**
     * Save the DataFrame to a vault table.
     * @param {Vault} vault - The vault instance
     * @param {string} tableName - Target table name
     * @param {Object} [options] - Save options
     * @param {string} [options.ifExists='fail'] - Action if table exists: 'fail', 'replace', 'append'
     * @returns {Promise<{rowCount: number}>}
     *
     * @example
     * const df = WritableDataFrame.fromRecords([{ id: 1, name: 'Alice' }]);
     * await df.saveTo(v, 'users');
     * await df.saveTo(v, 'users', { ifExists: 'replace' });
     * await df.saveTo(v, 'users', { ifExists: 'append' });
     */
    async saveTo(vault, tableName, options = {}) {
        const { ifExists = 'fail' } = options;

        switch (ifExists) {
            case 'replace':
                // Drop if exists, then create
                await vault.exec(`DROP TABLE IF EXISTS ${tableName}`);
                await this._createTableAndInsert(vault, tableName);
                return { rowCount: this._rowCount, tableName };

            case 'append':
                // Just insert data (assume table exists)
                try {
                    await this._insertInto(vault, tableName);
                    return { rowCount: this._rowCount, tableName };
                } catch (e) {
                    // Table doesn't exist, create it
                    if (e.message && e.message.includes('not found')) {
                        await this._createTableAndInsert(vault, tableName);
                        return { rowCount: this._rowCount, tableName };
                    }
                    throw e;
                }

            case 'fail':
            default:
                // Try to create - will fail if exists
                try {
                    await this._createTableAndInsert(vault, tableName);
                    return { rowCount: this._rowCount, tableName };
                } catch (e) {
                    // If table already exists, throw with friendly message
                    if (e.message && (e.message.includes('AlreadyExists') || e.message.includes('already exists'))) {
                        throw new Error(`Table '${tableName}' already exists`);
                    }
                    throw e;
                }
        }
    }

    /**
     * Infer SQL type from value.
     * @private
     */
    _inferType(values) {
        for (const v of values) {
            if (v === null || v === undefined) continue;
            if (typeof v === 'bigint') return 'INT64';
            if (typeof v === 'number') {
                if (Number.isInteger(v) && v <= 2147483647 && v >= -2147483648) {
                    return 'INT';
                }
                return 'FLOAT64';
            }
            if (typeof v === 'boolean') return 'INT';
            if (Array.isArray(v)) return `FLOAT32[${v.length}]`;
            return 'TEXT';
        }
        return 'TEXT'; // Default for all nulls
    }

    /**
     * Format value for SQL.
     * @private
     */
    _formatValue(value) {
        if (value === null || value === undefined) return 'NULL';
        if (typeof value === 'number' || typeof value === 'bigint') return String(value);
        if (typeof value === 'boolean') return value ? '1' : '0';
        if (Array.isArray(value)) return `[${value.join(',')}]`;
        // Escape single quotes in strings
        return `'${String(value).replace(/'/g, "''")}'`;
    }

    /**
     * Create table and insert data.
     * @private
     */
    async _createTableAndInsert(vault, tableName) {
        // Build CREATE TABLE statement
        const columnDefs = this._columnNames.map(name => {
            const type = this._inferType(this._columns[name]);
            return `${name} ${type}`;
        }).join(', ');

        await vault.exec(`CREATE TABLE ${tableName} (${columnDefs})`);
        await this._insertInto(vault, tableName);
    }

    /**
     * Insert data into existing table.
     * @private
     */
    async _insertInto(vault, tableName) {
        if (this._rowCount === 0) return { rowCount: 0 };

        // Build INSERT statement with all values
        const colList = this._columnNames.join(', ');
        const valueRows = [];

        for (let i = 0; i < this._rowCount; i++) {
            const values = this._columnNames.map(name =>
                this._formatValue(this._columns[name][i])
            ).join(', ');
            valueRows.push(`(${values})`);
        }

        await vault.exec(`INSERT INTO ${tableName} (${colList}) VALUES ${valueRows.join(', ')}`);
        return { rowCount: this._rowCount };
    }

    /**
     * Get a row as an object.
     * @param {number} index - Row index
     * @returns {Object}
     */
    row(index) {
        if (index < 0 || index >= this._rowCount) return null;
        const obj = {};
        for (const name of this._columnNames) {
            obj[name] = this._columns[name][index];
        }
        return obj;
    }

    /**
     * Get column data.
     * @param {string} name - Column name
     * @returns {Array}
     */
    column(name) {
        return this._columns[name] || null;
    }

    /**
     * Convert to array of record objects.
     * @returns {Object[]}
     */
    toRecords() {
        const records = [];
        for (let i = 0; i < this._rowCount; i++) {
            records.push(this.row(i));
        }
        return records;
    }
}

/**
 * Create a new Vault instance.
 *
 * @param {Function|null} getEncryptionKey - Async callback returning encryption key
 * @returns {Promise<Vault>}
 *
 * @example
 * // Unencrypted vault
 * const v = await vault();
 *
 * // Encrypted vault
 * const v = await vault(async () => {
 *     return await promptUserForPassword();
 * });
 *
 * // KV operations
 * await v.set('user', { name: 'Alice' });
 * const user = await v.get('user');
 *
 * // SQL operations
 * await v.exec('CREATE TABLE products (id INT, name TEXT)');
 * await v.exec('SELECT * FROM products WHERE name NEAR "shoes"');
 */
async function vault(getEncryptionKey = null) {
    const v = new Vault(getEncryptionKey);
    await v._init();
    return v;
}

// Export Vault class for type checking

// ============================================================================
// Legacy Store API (deprecated, use vault() instead)
// ============================================================================

/**
 * @deprecated Use vault() instead
 * Store - Simple key-value and collection storage with search.
 *
 * All operations run in a SharedWorker for OPFS sync access and shared GPU.
 *
 * @example
 * const store = await lanceStore('myapp');
 * await store.set('user', { name: 'Alice' });
 * const user = await store.get('user');
 */

export { Vault, TableRef, WritableDataFrame, vault };
