const binding = require('../build/Release/lanceql.node');

// ============================================================================
// SqliteError - Custom error class matching better-sqlite3
// ============================================================================

class SqliteError extends Error {
    constructor(message, code) {
        super(message);
        this.name = 'SqliteError';
        this.code = code || 'SQLITE_ERROR';
    }
}

// ============================================================================
// Database - Main database class matching better-sqlite3 API
// ============================================================================

class Database {
    constructor(filename, options = {}) {
        // Validate inputs like better-sqlite3
        if (filename === ':memory:' || filename === '') {
            throw new SqliteError('In-memory databases not supported', 'SQLITE_ERROR');
        }

        // Support both file paths and Buffer objects
        try {
            if (Buffer.isBuffer(filename)) {
                this._db = new binding.Database(filename, options);
                this._name = 'buffer';
            } else {
                this._db = new binding.Database(filename, options);
                this._name = filename;
            }
        } catch (err) {
            throw new SqliteError(err.message || 'Failed to open database', 'SQLITE_CANTOPEN');
        }

        this._open = true;
        // Lance files are always read-only, default to true unless explicitly set to false
        this._readonly = options.readonly !== false;
    }

    // Properties (getters matching better-sqlite3)
    get open() { return this._open; }
    get inTransaction() { return false; } // Always false for read-only Lance
    get readonly() { return this._readonly; }
    get memory() { return false; }
    get name() { return this._name; }

    prepare(sql) {
        if (!this._open) {
            throw new SqliteError('The database connection is not open', 'SQLITE_MISUSE');
        }

        if (typeof sql !== 'string') {
            throw new SqliteError('SQL must be a string', 'SQLITE_MISUSE');
        }

        try {
            const nativeStmt = this._db.prepare(sql);
            return new Statement(nativeStmt, this, sql);
        } catch (err) {
            throw new SqliteError(err.message || 'Failed to prepare statement', 'SQLITE_ERROR');
        }
    }

    exec(sql) {
        if (!this._open) {
            throw new SqliteError('The database connection is not open', 'SQLITE_MISUSE');
        }

        // For v0.1.0, exec is a no-op (Lance is read-only)
        // Just validate SQL syntax by trying to parse it
        if (typeof sql !== 'string') {
            throw new SqliteError('SQL must be a string', 'SQLITE_MISUSE');
        }

        return this; // Match better-sqlite3 chaining
    }

    pragma(sql, options) {
        if (!this._open) {
            throw new SqliteError('The database connection is not open', 'SQLITE_MISUSE');
        }

        // PRAGMA commands for metadata queries
        // For v0.1.0, only table_info is supported
        if (sql.startsWith('table_info')) {
            // Return empty array for now (would need lance_pragma_table_info export)
            return [];
        }

        return [];
    }

    transaction(fn) {
        if (!this._open) {
            throw new SqliteError('The database connection is not open', 'SQLITE_MISUSE');
        }

        // Transactions not supported for read-only Lance files
        throw new SqliteError('Transactions not supported in v0.1.0', 'SQLITE_ERROR');
    }

    close() {
        if (!this._open) return this;

        try {
            this._db.close();
            this._open = false;
        } catch (err) {
            throw new SqliteError(err.message || 'Failed to close database', 'SQLITE_ERROR');
        }

        return this; // Match better-sqlite3 chaining
    }

    // Stub methods that throw errors (not supported in v0.1.0)
    // Note: 'function' is a reserved word, so we use quoted property syntax
    'function'() {
        throw new SqliteError('User-defined functions not supported', 'SQLITE_ERROR');
    }

    aggregate() {
        throw new SqliteError('Aggregate functions not supported', 'SQLITE_ERROR');
    }

    table() {
        throw new SqliteError('Virtual tables not supported', 'SQLITE_ERROR');
    }

    loadExtension() {
        throw new SqliteError('Extensions not supported', 'SQLITE_ERROR');
    }

    backup() {
        throw new SqliteError('Backup not supported', 'SQLITE_ERROR');
    }

    serialize() {
        throw new SqliteError('Serialize not supported', 'SQLITE_ERROR');
    }

    defaultSafeIntegers(toggle) {
        // No-op for v0.1.0 (always uses JavaScript numbers)
        return this;
    }

    unsafeMode(toggle) {
        // No-op (always safe)
        return this;
    }
}

// ============================================================================
// Statement - Prepared statement class matching better-sqlite3 API
// ============================================================================

class Statement {
    constructor(nativeStmt, db, sql) {
        this._stmt = nativeStmt;
        this._db = db;
        this._sql = sql;
        this._pluck = false;
        this._expand = false;
        this._raw = false;
    }

    // Properties (getters matching better-sqlite3)
    get source() { return this._sql; }
    get readonly() { return this._sql.trim().toUpperCase().startsWith('SELECT'); }
    get database() { return this._db; }
    get reader() { return this.readonly; }
    get busy() { return false; } // Synchronous execution

    all(...params) {
        try {
            let rows = this._stmt.all();

            // Apply formatting modifiers
            if (this._raw) {
                // Convert {id: 1, name: 'John'} to [1, 'John']
                return rows.map(row => Object.values(row));
            }
            if (this._pluck) {
                // Return first column only
                return rows.map(row => Object.values(row)[0]);
            }
            if (this._expand) {
                // Namespace by table (if column metadata available)
                // For v0.1.0, just return as-is since we don't track table names yet
                return rows;
            }

            return rows;
        } catch (err) {
            throw new SqliteError(err.message || 'Failed to execute statement', 'SQLITE_ERROR');
        }
    }

    get(...params) {
        const rows = this.all(...params);
        return rows.length > 0 ? rows[0] : undefined;
    }

    run(...params) {
        if (!this.readonly) {
            throw new SqliteError('Write operations not supported in v0.1.0', 'SQLITE_READONLY');
        }

        // For SELECT queries, execute but return dummy write result
        try {
            this.all(...params);
            return { changes: 0, lastInsertRowid: 0 };
        } catch (err) {
            throw new SqliteError(err.message || 'Failed to execute statement', 'SQLITE_ERROR');
        }
    }

    iterate(...params) {
        // Return iterator matching better-sqlite3
        const rows = this.all(...params);
        return rows[Symbol.iterator]();
    }

    // Formatting methods (return this for chaining)
    pluck(toggle = true) {
        this._pluck = toggle;
        return this; // Enable chaining
    }

    expand(toggle = true) {
        this._expand = toggle;
        return this; // Enable chaining
    }

    raw(toggle = true) {
        this._raw = toggle;
        return this; // Enable chaining
    }

    columns() {
        // Return column metadata
        // For v0.1.0, execute query and extract column names from first row
        try {
            const rows = this._stmt.all();
            if (rows.length === 0) {
                return [];
            }

            // Return array of {name, column, table?, database?, type?}
            const firstRow = rows[0];
            return Object.keys(firstRow).map((name, index) => ({
                name: name,
                column: name,
                table: null,
                database: null,
                type: null
            }));
        } catch (err) {
            return [];
        }
    }

    bind(...params) {
        // Bind parameters (for v0.1.0, this is a no-op since we don't support params yet)
        // Just return this for chaining
        return this;
    }

    safeIntegers(toggle = true) {
        // No-op for v0.1.0 (always uses JavaScript numbers)
        return this;
    }

    finalize() {
        // Explicitly release native statement resources
        // This is important for long-running applications or when creating many statements
        if (this._stmt && typeof this._stmt.finalize === 'function') {
            this._stmt.finalize();
        }
        this._stmt = null;
    }
}

// ============================================================================
// Exports - Match better-sqlite3 export format exactly
// ============================================================================

module.exports = Database;
module.exports.Database = Database;
module.exports.SqliteError = SqliteError;
