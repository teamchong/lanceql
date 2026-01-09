/**
 * WorkerDatabase - SQL database operations with OPFS persistence
 */

import { opfsStorage } from './opfs-storage.js';
import { DataType, LanceFileWriter, E, D, parseBinaryColumnar } from './data-types.js';
import { getWasm, getWasmMemory } from './index.js';
import { BufferPool } from './buffer-pool.js';

// Streaming scan state
const scanStreams = new Map();
let nextScanId = 1;

export class WorkerDatabase {
    constructor(name, bufferPool) {
        this.name = name;
        this.tables = new Map();
        this.version = 0;
        this.manifestKey = `${name}/__manifest__`;

        // Use shared buffer pool or create local one
        this.bufferPool = bufferPool || new BufferPool();

        // Write buffer for fast inserts
        this._writeBuffer = new Map();
        this._flushTimer = null;
        this._flushInterval = 2000;       // Increased from 1000
        this._flushThreshold = 10000;     // Increased from 1000 to reduce fragments
        // Columnar write buffer: { colName: Array }
        this._columnarBuffer = new Map();
        this._flushing = false;

        // Read cache
        this._readCache = new Map();
        // Columnar merged cache: tableName -> { version: string, data: result }
        this._columnarCache = new Map();
    }

    async open() {
        // Load manifest from storage
        const manifestData = await opfsStorage.load(this.manifestKey);
        if (manifestData) {
            const manifest = JSON.parse(D.decode(manifestData));
            this.version = manifest.version || 0;
            this.tables = new Map(Object.entries(manifest.tables || {}));

            // Migrate legacy tables to versioned format
            for (const [tableName, tableState] of this.tables) {
                const hasVersions = await this._getLatestVersion(tableName);
                if (hasVersions === 0 && tableState.fragments?.length > 0) {
                    // Create version 1 from current state (migration)
                    await this._createVersion(tableName, 'MIGRATE');
                }
            }
        }
        return this;
    }

    async _saveManifest() {
        this.version++;
        const manifest = {
            version: this.version,
            timestamp: Date.now(),
            tables: Object.fromEntries(this.tables),
        };
        const data = E.encode(JSON.stringify(manifest));
        await opfsStorage.save(this.manifestKey, data);
    }

    // ==================== Time Travel (Versioning) ====================

    /**
     * Get the latest version number for a table
     */
    async _getLatestVersion(tableName) {
        const path = `${this.name}/${tableName}/_latest`;
        try {
            const data = await opfsStorage.load(path);
            if (!data) return 0;
            return parseInt(D.decode(data), 10);
        } catch {
            return 0;
        }
    }

    /**
     * Set the latest version number for a table
     */
    async _setLatestVersion(tableName, version) {
        const path = `${this.name}/${tableName}/_latest`;
        await opfsStorage.save(path, E.encode(String(version)));
    }

    /**
     * Load a specific version's manifest
     */
    async _loadTableVersion(tableName, version) {
        const path = `${this.name}/${tableName}/_versions/${version}.manifest`;
        const data = await opfsStorage.load(path);
        if (!data) {
            throw new Error(`Version ${version} not found for table '${tableName}'`);
        }
        return JSON.parse(D.decode(data));
    }

    /**
     * Save a version manifest
     */
    async _saveTableVersion(tableName, versionManifest) {
        const path = `${this.name}/${tableName}/_versions/${versionManifest.version}.manifest`;
        await opfsStorage.save(path, E.encode(JSON.stringify(versionManifest)));
    }

    /**
     * Create a new version for a table
     */
    async _createVersion(tableName, operation) {
        const table = this.tables.get(tableName);
        if (!table) return 0;

        const currentVersion = await this._getLatestVersion(tableName);
        const newVersion = currentVersion + 1;

        const versionManifest = {
            version: newVersion,
            timestamp: Date.now(),
            parentVersion: currentVersion,
            operation,
            schema: table.schema,
            fragments: [...table.fragments],
            deletionVector: [...table.deletionVector],
            rowCount: table.rowCount,
            nextRowId: table.nextRowId
        };

        await this._saveTableVersion(tableName, versionManifest);
        await this._setLatestVersion(tableName, newVersion);
        return newVersion;
    }

    /**
     * List all versions for a table
     */
    async listVersions(tableName) {
        const latest = await this._getLatestVersion(tableName);
        const versions = [];
        for (let v = 1; v <= latest; v++) {
            try {
                const manifest = await this._loadTableVersion(tableName, v);
                versions.push({
                    version: manifest.version,
                    timestamp: manifest.timestamp,
                    operation: manifest.operation,
                    rowCount: manifest.rowCount
                });
            } catch {
                // Version file missing, skip
            }
        }
        return versions;
    }

    /**
     * Select data at a specific version (time travel)
     */
    async selectAtVersion(tableName, version, options = {}) {
        const versionManifest = await this._loadTableVersion(tableName, version);
        const deletedSet = new Set(versionManifest.deletionVector);
        const table = this.tables.get(tableName);

        // Load rows from version's fragments
        const rows = [];
        for (const fragPath of versionManifest.fragments) {
            const fragData = await opfsStorage.load(fragPath);
            if (fragData) {
                const fragRows = this._parseFragment(fragData, versionManifest.schema);
                for (const row of fragRows) {
                    if (!deletedSet.has(row.__rowId)) {
                        rows.push(row);
                    }
                }
            }
        }

        // Apply options (WHERE, ORDER BY, LIMIT, OFFSET, columns)
        let result = rows;

        if (options.where) {
            result = result.filter(options.where);
        }

        if (options.orderBy) {
            const { column, desc } = options.orderBy;
            result.sort((a, b) => {
                const cmp = a[column] < b[column] ? -1 : a[column] > b[column] ? 1 : 0;
                return desc ? -cmp : cmp;
            });
        }

        if (options.offset) {
            result = result.slice(options.offset);
        }

        if (options.limit) {
            result = result.slice(0, options.limit);
        }

        // Column projection and __rowId removal
        const projectCols = options.columns && options.columns.length > 0 && options.columns[0] !== '*'
            ? options.columns : null;

        return result.map(row => {
            if (projectCols) {
                const projected = {};
                for (const col of projectCols) {
                    projected[col] = row[col];
                }
                return projected;
            } else {
                const { __rowId, ...rest } = row;
                return rest;
            }
        });
    }

    /**
     * Restore table to a previous version (creates a new version)
     */
    async restoreToVersion(tableName, targetVersion) {
        const targetManifest = await this._loadTableVersion(tableName, targetVersion);
        const currentVersion = await this._getLatestVersion(tableName);
        const newVersion = currentVersion + 1;

        // Create new version with target's state
        const restoredManifest = {
            ...targetManifest,
            version: newVersion,
            timestamp: Date.now(),
            parentVersion: currentVersion,
            operation: `RESTORE_FROM_${targetVersion}`
        };

        await this._saveTableVersion(tableName, restoredManifest);
        await this._setLatestVersion(tableName, newVersion);

        // Update in-memory state
        const table = this.tables.get(tableName);
        if (table) {
            table.fragments = [...targetManifest.fragments];
            table.deletionVector = [...targetManifest.deletionVector];
            table.rowCount = targetManifest.rowCount;
            table.nextRowId = targetManifest.nextRowId;

            // Clear write buffer
            this._columnarBuffer.delete(tableName);
            this._writeBuffer.delete(tableName);

            // Save to global manifest
            await this._saveManifest();
        }

        return { restored: true, newVersion };
    }

    // ==================== End Time Travel ====================

    async createTable(tableName, columns, ifNotExists = false) {
        if (this.tables.has(tableName)) {
            if (ifNotExists) {
                return { success: true, existed: true };
            }
            throw new Error(`Table '${tableName}' already exists`);
        }

        const schema = columns.map(col => ({
            name: col.name,
            type: DataType[(col.dataType || col.type)?.toUpperCase()] || col.dataType || col.type || 'string',
            primaryKey: col.primaryKey || false,
            vectorDim: col.vectorDim || null,
        }));

        const tableState = {
            name: tableName,
            schema,
            fragments: [],
            deletionVector: [],
            rowCount: 0,
            nextRowId: 0,
            version: 0,
            createdAt: Date.now(),
        };

        this.tables.set(tableName, tableState);
        await this._saveManifest();

        // Create version 1 for new table (CREATE operation)
        await this._createVersion(tableName, 'CREATE');

        return { success: true, table: tableName };
    }

    async dropTable(tableName, ifExists = false) {
        if (!this.tables.has(tableName)) {
            if (ifExists) {
                this._writeBuffer.delete(tableName);
                return { success: true, existed: false };
            }
            throw new Error(`Table '${tableName}' does not exist`);
        }

        const table = this.tables.get(tableName);

        this._writeBuffer.delete(tableName);

        for (const fragKey of table.fragments) {
            this._readCache.delete(fragKey);
            await opfsStorage.delete(fragKey);
        }

        this.tables.delete(tableName);
        await this._saveManifest();

        return { success: true, table: tableName };
    }

    async insert(tableName, rows) {
        if (!this.tables.has(tableName)) {
            throw new Error(`Table '${tableName}' does not exist`);
        }

        const table = this.tables.get(tableName);
        const n = rows.length;

        // Initialize columnar buffer with pre-allocated typed arrays
        if (!this._columnarBuffer.has(tableName)) {
            const initialCapacity = Math.max(1024, n * 2);
            const cols = {
                __rowId: new Float64Array(initialCapacity),
                __length: 0,
                __capacity: initialCapacity,
                __schema: table.schema
            };
            for (const c of table.schema) {
                const type = (c.dataType || c.type || '').toLowerCase();
                if (type === 'text' || type === 'string' || type === 'varchar') {
                    cols[c.name] = new Array(initialCapacity);
                } else if (type === 'int64' || type === 'bigint') {
                    cols[c.name] = new BigInt64Array(initialCapacity);
                } else {
                    cols[c.name] = new Float64Array(initialCapacity);
                }
            }
            this._columnarBuffer.set(tableName, cols);
        }

        const colBuf = this._columnarBuffer.get(tableName);
        let len = colBuf.__length;
        let cap = colBuf.__capacity;

        // Grow arrays if needed
        if (len + n > cap) {
            const newCap = Math.max(cap * 2, len + n);
            const newRowId = new Float64Array(newCap);
            newRowId.set(colBuf.__rowId.subarray(0, len));
            colBuf.__rowId = newRowId;

            for (const c of table.schema) {
                const old = colBuf[c.name];
                if (old instanceof Float64Array) {
                    const newArr = new Float64Array(newCap);
                    newArr.set(old.subarray(0, len));
                    colBuf[c.name] = newArr;
                } else if (old instanceof BigInt64Array) {
                    const newArr = new BigInt64Array(newCap);
                    newArr.set(old.subarray(0, len));
                    colBuf[c.name] = newArr;
                } else {
                    // String array - just extend
                    colBuf[c.name].length = newCap;
                }
            }
            colBuf.__capacity = newCap;
        }

        // Append directly to typed arrays (fast!)
        for (let i = 0; i < n; i++) {
            const row = rows[i];
            colBuf.__rowId[len + i] = table.nextRowId++;
            for (const c of table.schema) {
                let val = row[c.name];
                // Use NaN to represent null in numeric typed arrays
                if (colBuf[c.name] instanceof Float64Array) {
                    colBuf[c.name][len + i] = (val !== null && val !== undefined) ? Number(val) : NaN;
                } else if (colBuf[c.name] instanceof BigInt64Array) {
                    colBuf[c.name][len + i] = (val !== null && val !== undefined) ? BigInt(val) : 0n;
                } else {
                    colBuf[c.name][len + i] = val ?? null;
                }
            }
        }
        colBuf.__length = len + n;

        table.rowCount += n;
        table.version = (table.version || 0) + 1;
        this._scheduleFlush();

        if (colBuf.__length >= this._flushThreshold) {
            await this._flushTable(tableName);
        }

        return { success: true, inserted: n };
    }

    _scheduleFlush() {
        if (this._flushTimer) return;
        this._flushTimer = setTimeout(() => {
            this._flushTimer = null;
            this.flush().catch(e => console.warn('[WorkerDatabase] Flush error:', e));
        }, this._flushInterval);
    }

    async flush() {
        if (this._flushing) return;
        this._flushing = true;

        try {
            const tables = [...this._columnarBuffer.keys()];
            for (const tableName of tables) {
                await this._flushTable(tableName);
            }
        } finally {
            this._flushing = false;
        }
    }

    async _flushTable(tableName) {
        const colBuf = this._columnarBuffer.get(tableName);
        const bufLen = colBuf?.__length || 0;
        if (!colBuf || bufLen === 0) return;

        const table = this.tables.get(tableName);
        if (!table) return;

        // Build schema with __rowId and correct types
        const schemaWithRowId = [
            { name: '__rowId', type: 'int64', dataType: 'float64', primaryKey: true },
            ...table.schema.filter(c => c.name !== '__rowId').map(c => {
                const type = (c.dataType || c.type || '').toLowerCase();
                if (type === 'int64' || type === 'bigint') {
                    return { ...c, dataType: 'int64' };
                }
                const isNumeric = type === 'float64' || type === 'float32' || type === 'int32' || type === 'integer' || type === 'real' || type === 'double';
                return {
                    ...c,
                    dataType: isNumeric ? 'float64' : (c.dataType || c.type || 'float64')
                };
            })
        ];

        // Already typed arrays - just slice to actual length (no copy needed for writing)
        const columnarData = {};
        for (const col of schemaWithRowId) {
            const arr = colBuf[col.name];
            if (!arr) continue;

            if (arr instanceof Float64Array || arr instanceof BigInt64Array) {
                // Typed array - slice to actual length
                columnarData[col.name] = arr.subarray(0, bufLen);
            } else {
                // String array - slice to actual length
                columnarData[col.name] = arr.slice(0, bufLen);
            }
        }

        // Reset buffer length (reuse the pre-allocated arrays)
        colBuf.__length = 0;

        // Use setColumnarData - no row conversion!
        const writer = new LanceFileWriter(schemaWithRowId);
        writer.setColumnarData(columnarData);
        const lanceData = writer.build();

        const fragKey = `${this.name}/${tableName}/frag_${Date.now()}_${Math.random().toString(36).slice(2)}.lance`;
        await opfsStorage.save(fragKey, lanceData);

        table.fragments.push(fragKey);
        await this._saveManifest();

        // Create a new version for this flush (INSERT operation)
        await this._createVersion(tableName, 'INSERT');
    }

    async delete(tableName, predicateFn) {
        if (!this.tables.has(tableName)) {
            throw new Error(`Table '${tableName}' does not exist`);
        }

        const table = this.tables.get(tableName);
        let deletedCount = 0;

        // Delete from columnar buffer (mark rows for deletion)
        const colBuf = this._columnarBuffer.get(tableName);
        const bufLen = colBuf?.__length || 0;
        if (colBuf && bufLen > 0) {
            const colNames = table.schema.map(c => c.name);
            for (let i = 0; i < bufLen; i++) {
                // Build row object for predicate evaluation
                const row = { __rowId: colBuf.__rowId[i] };
                for (const name of colNames) {
                    const v = colBuf[name][i];
                    row[name] = Number.isNaN(v) ? null : v;
                }

                if (predicateFn(row)) {
                    // Mark row as deleted by adding to deletion vector
                    if (!table.deletionVector.includes(colBuf.__rowId[i])) {
                        table.deletionVector.push(colBuf.__rowId[i]);
                        deletedCount++;
                    }
                }
            }
        }

        // Delete from persisted fragments
        for (const fragKey of table.fragments) {
            const fragData = await opfsStorage.load(fragKey);
            if (fragData) {
                const rows = this._parseFragment(fragData, table.schema);
                for (const row of rows) {
                    if (!table.deletionVector.includes(row.__rowId) && predicateFn(row)) {
                        table.deletionVector.push(row.__rowId);
                        deletedCount++;
                    }
                }
            }
        }

        table.rowCount -= deletedCount;
        table.version = (table.version || 0) + 1;
        await this._saveManifest();

        // Create a new version for this delete operation
        if (deletedCount > 0) {
            await this._createVersion(tableName, 'DELETE');
        }

        return { success: true, deleted: deletedCount };
    }

    async update(tableName, updates, predicateFn) {
        if (!this.tables.has(tableName)) {
            throw new Error(`Table '${tableName}' does not exist`);
        }

        const table = this.tables.get(tableName);
        let updatedCount = 0;

        // Update buffered rows
        const buffer = this._writeBuffer.get(tableName);
        if (buffer && buffer.length > 0) {
            for (const row of buffer) {
                if (predicateFn(row)) {
                    Object.assign(row, updates);
                    updatedCount++;
                }
            }
        }

        // Update persisted rows
        const persistedUpdates = [];
        for (const fragKey of table.fragments) {
            const fragData = await opfsStorage.load(fragKey);
            if (fragData) {
                const rows = this._parseFragment(fragData, table.schema);
                for (const row of rows) {
                    if (!table.deletionVector.includes(row.__rowId) && predicateFn(row)) {
                        table.deletionVector.push(row.__rowId);
                        table.rowCount--;

                        const newRow = { ...row, ...updates };
                        delete newRow.__rowId;
                        persistedUpdates.push(newRow);
                        updatedCount++;
                    }
                }
            }
        }

        if (persistedUpdates.length > 0) {
            await this.insert(tableName, persistedUpdates);
        } else {
            await this._saveManifest();
        }

        // Create a new version for this update operation
        // Note: insert already creates a version, so only create if no inserts
        if (updatedCount > 0 && persistedUpdates.length === 0) {
            await this._createVersion(tableName, 'UPDATE');
        }

        return { success: true, updated: updatedCount };
    }

    async updateWithExpr(tableName, updateExprs, predicateFn, evalExpr) {
        if (!this.tables.has(tableName)) {
            throw new Error(`Table '${tableName}' does not exist`);
        }

        const table = this.tables.get(tableName);
        let updatedCount = 0;

        // Update columnar buffer (where most recent data is stored)
        const colBuf = this._columnarBuffer.get(tableName);
        const bufLen = colBuf?.__length || 0;
        if (colBuf && bufLen > 0) {
            const colNames = table.schema.map(c => c.name);
            for (let i = 0; i < bufLen; i++) {
                // Build row object for predicate evaluation
                const row = { __rowId: colBuf.__rowId[i] };
                for (const name of colNames) {
                    const v = colBuf[name][i];
                    row[name] = Number.isNaN(v) ? null : v;
                }

                if (predicateFn(row)) {
                    // Apply updates directly to columnar buffer
                    for (const [col, expr] of Object.entries(updateExprs)) {
                        const newVal = evalExpr(expr, row);
                        if (colBuf[col] !== undefined) {
                            colBuf[col][i] = newVal ?? (colBuf[col] instanceof Float64Array ? NaN : null);
                        }
                    }
                    table.version = (table.version || 0) + 1;
                    updatedCount++;
                }
            }
        }

        // Update persisted rows
        const persistedUpdates = [];
        for (const fragKey of table.fragments) {
            const fragData = await opfsStorage.load(fragKey);
            if (fragData) {
                const rows = this._parseFragment(fragData, table.schema);
                for (const row of rows) {
                    if (!table.deletionVector.includes(row.__rowId) && predicateFn(row)) {
                        table.deletionVector.push(row.__rowId);
                        table.rowCount--;

                        const newRow = { ...row };
                        for (const [col, expr] of Object.entries(updateExprs)) {
                            newRow[col] = evalExpr(expr, row);
                        }
                        delete newRow.__rowId;
                        persistedUpdates.push(newRow);
                        updatedCount++;
                    }
                }
            }
        }

        if (persistedUpdates.length > 0) {
            await this.insert(tableName, persistedUpdates);
        } else {
            await this._saveManifest();
        }

        // Create a new version for this update operation
        // Note: insert already creates a version, so only create if no inserts
        if (updatedCount > 0 && persistedUpdates.length === 0) {
            await this._createVersion(tableName, 'UPDATE');
        }

        return { success: true, updated: updatedCount };
    }

    async select(tableName, options = {}) {
        if (!this.tables.has(tableName)) {
            throw new Error(`Table '${tableName}' does not exist`);
        }

        let rows = await this._readAllRows(tableName);

        // WHERE filter
        if (options.where) {
            rows = rows.filter(options.where);
        }

        // ORDER BY
        if (options.orderBy) {
            const { column, desc } = options.orderBy;
            rows.sort((a, b) => {
                const cmp = a[column] < b[column] ? -1 : a[column] > b[column] ? 1 : 0;
                return desc ? -cmp : cmp;
            });
        }

        // OFFSET
        if (options.offset) {
            rows = rows.slice(options.offset);
        }

        // LIMIT
        if (options.limit) {
            rows = rows.slice(0, options.limit);
        }

        // Column projection and __rowId removal in single pass
        const projectCols = options.columns && options.columns.length > 0 && options.columns[0] !== '*'
            ? options.columns
            : null;

        // Single pass: project columns and remove __rowId
        const result = new Array(rows.length);
        for (let i = 0; i < rows.length; i++) {
            const row = rows[i];
            if (projectCols) {
                const projected = {};
                for (const col of projectCols) {
                    projected[col] = row[col];
                }
                result[i] = projected;
            } else {
                // Clone without __rowId
                const { __rowId, ...rest } = row;
                result[i] = rest;
            }
        }

        return result;
    }

    async _readAllRows(tableName) {
        const table = this.tables.get(tableName);
        const colBuf = this._columnarBuffer.get(tableName);
        const hasDeleted = table.deletionVector.length > 0;
        const deletedSet = hasDeleted ? new Set(table.deletionVector) : null;
        const allRows = [];

        // Read from persisted fragments
        for (const fragKey of table.fragments) {
            // Check Buffer Pool for binary columnar data
            let binary = this.bufferPool.get(fragKey);
            let rows = null;

            if (!binary) {
                // Load from OPFS
                const fragData = await opfsStorage.load(fragKey);
                if (fragData) {
                    binary = parseBinaryColumnar(fragData);
                    if (binary) {
                        // Store in pool (approx size)
                        this.bufferPool.set(fragKey, binary, fragData.byteLength);
                    } else {
                        // Fallback for JSON/Legacy
                        rows = this._parseFragment(fragData, table.schema);
                        // We don't cache legacy JSON rows in the pool to avoid complexity
                    }
                }
            }

            // Hydrate rows from binary if available
            if (binary && !rows) {
                rows = this._hydrateRowsFromBinary(binary, table.schema);
            }
            // Ensure rows is an array
            rows = rows || [];

            // Optimize: if no deletions, push all at once
            if (!deletedSet) {
                allRows.push(...rows);
            } else {
                for (const row of rows) {
                    if (!deletedSet.has(row.__rowId)) {
                        allRows.push(row);
                    }
                }
            }
        }

        // Include columnar buffer (convert to rows for compatibility)
        const bufLen = colBuf?.__length || 0;
        if (colBuf && bufLen > 0) {
            const colNames = table.schema.map(c => c.name);
            for (let i = 0; i < bufLen; i++) {
                if (deletedSet && deletedSet.has(colBuf.__rowId[i])) continue;
                const row = { __rowId: colBuf.__rowId[i] };
                for (const name of colNames) {
                    const v = colBuf[name][i];
                    // Convert NaN back to null (NaN represents null in typed arrays)
                    row[name] = Number.isNaN(v) ? null : v;
                }
                allRows.push(row);
            }
        }

        return allRows;
    }

    _hydrateRowsFromBinary(binary, schema) {
        const { columns, rowCount } = binary;
        const colNames = schema.map(c => c.name);
        const rows = new Array(rowCount);

        for (let i = 0; i < rowCount; i++) {
            const row = { __rowId: columns.__rowId[i] };
            for (const name of colNames) {
                if (columns[name]) {
                    row[name] = columns[name][i];
                }
            }
            rows[i] = row;
        }
        return rows;
    }

    _parseFragment(data, schema) {
        try {
            // Try binary format first (faster)
            const binary = parseBinaryColumnar(data);
            if (binary) {
                return this._parseBinaryColumnar(binary);
            }

            // Fall back to JSON
            const text = D.decode(data);
            const parsed = JSON.parse(text);

            if (parsed.format === 'json' && parsed.columns) {
                return this._parseJsonColumnar(parsed);
            }

            return Array.isArray(parsed) ? parsed : [parsed];
        } catch (e) {
            console.warn('[WorkerDatabase] Failed to parse fragment:', e);
            return [];
        }
    }

    /**
     * Select data in COLUMNAR format (no row conversion).
     * Returns { schema, columns: { colName: TypedArray }, rowCount }
     */
    async selectColumnar(tableName) {
        const table = this.tables.get(tableName);
        if (!table) return null;

        // Version string based on table state and explicit version counter
        const bufLen = this._columnarBuffer.get(tableName)?.__length || 0;
        const version = `${table.fragments.length}:${bufLen}:${table.deletionVector.length}:${table.version || 0}`;

        // Check cache
        const cached = this._columnarCache.get(tableName);
        if (cached && cached.version === version) {
            // Need to return NEW views of the buffers so Transferable doesn't detach the cached buffers
            const columns = {};
            for (const [name, arr] of Object.entries(cached.data.columns)) {
                if (ArrayBuffer.isView(arr)) {
                    columns[name] = new arr.constructor(arr.buffer, arr.byteOffset, arr.length);
                } else {
                    columns[name] = arr; // Strings are just arrays
                }
            }
            return { schema: table.schema, columns, rowCount: cached.data.rowCount };
        }

        const hasDeleted = table.deletionVector.length > 0;
        const deletedSet = hasDeleted ? new Set(table.deletionVector) : null;

        // Collect columnar data from all fragments
        const allColumns = {};
        const colNames = table.schema.map(c => c.name);
        for (const name of colNames) allColumns[name] = [];
        allColumns.__rowId = [];

        // Read from persisted fragments - use cache, STAY COLUMNAR
        for (const fragKey of table.fragments) {
            // Check Buffer Pool
            let binary = this.bufferPool.get(fragKey);
            if (!binary) {
                const fragData = await opfsStorage.load(fragKey);
                if (!fragData) continue;
                binary = parseBinaryColumnar(fragData);
                if (binary) {
                    this.bufferPool.set(fragKey, binary, fragData.byteLength);
                }
            }
            if (!binary) continue;

            const { columns, rowCount } = binary;

            // If no deletions, append directly
            if (!deletedSet) {
                for (const name of colNames) {
                    if (columns[name]) allColumns[name].push(columns[name]);
                }
                if (columns.__rowId) allColumns.__rowId.push(columns.__rowId);
            } else {
                // Filter by deletion vector - need to filter each column
                const rowIds = columns.__rowId;
                const validIndices = [];
                for (let i = 0; i < rowCount; i++) {
                    if (!deletedSet.has(rowIds[i])) validIndices.push(i);
                }
                for (const name of colNames) {
                    if (columns[name]) {
                        const arr = columns[name];
                        const filtered = new arr.constructor(validIndices.length);
                        for (let j = 0; j < validIndices.length; j++) {
                            filtered[j] = arr[validIndices[j]];
                        }
                        allColumns[name].push(filtered);
                    }
                }
            }
        }

        // Include columnar buffer directly - already typed arrays!
        // Copy to owned arrays so Transferable works without extra copy in sendResponse
        const colBuf = this._columnarBuffer.get(tableName);
        const bufLen2 = colBuf?.__length || 0;
        if (colBuf && bufLen2 > 0) {
            if (!deletedSet) {
                // No deletions - copy to fresh owned arrays
                for (const col of table.schema) {
                    const arr = colBuf[col.name];
                    if (!arr) continue;
                    if (arr instanceof Float64Array) {
                        // Copy to owned array (enables zero-copy transfer)
                        const owned = new Float64Array(bufLen2);
                        owned.set(arr.subarray(0, bufLen2));
                        allColumns[col.name].push(owned);
                    } else {
                        allColumns[col.name].push(arr.slice(0, bufLen2));
                    }
                }
                const ownedRowId = new Float64Array(bufLen2);
                ownedRowId.set(colBuf.__rowId.subarray(0, bufLen2));
                allColumns.__rowId.push(ownedRowId);
            } else {
                // Filter by deletion vector
                const validIndices = [];
                for (let i = 0; i < bufLen2; i++) {
                    if (!deletedSet.has(colBuf.__rowId[i])) validIndices.push(i);
                }
                const vLen = validIndices.length;
                for (const col of table.schema) {
                    const arr = colBuf[col.name];
                    if (!arr) continue;
                    if (arr instanceof Float64Array) {
                        const filtered = new Float64Array(vLen);
                        for (let j = 0; j < vLen; j++) filtered[j] = arr[validIndices[j]];
                        allColumns[col.name].push(filtered);
                    } else {
                        allColumns[col.name].push(validIndices.map(i => arr[i]));
                    }
                }
                const filteredRowId = new Float64Array(vLen);
                for (let j = 0; j < vLen; j++) filteredRowId[j] = colBuf.__rowId[validIndices[j]];
                allColumns.__rowId.push(filteredRowId);
            }
        }

        // Merge arrays for each column
        const mergedColumns = {};
        let totalRows = 0;
        for (const name of [...colNames, '__rowId']) {
            const arrays = allColumns[name];
            if (arrays.length === 0) {
                mergedColumns[name] = new Float64Array(0);
            } else if (arrays.length === 1) {
                mergedColumns[name] = arrays[0];
                // Track totalRows from any column that has data
                if (totalRows === 0) totalRows = arrays[0].length;
            } else {
                // Merge multiple arrays
                const totalLen = arrays.reduce((sum, a) => sum + a.length, 0);
                // Track totalRows from any column that has data
                if (totalRows === 0) totalRows = totalLen;

                // Determine type from first array
                const first = arrays[0];
                const merged = ArrayBuffer.isView(first)
                    ? new first.constructor(totalLen)
                    : new Array(totalLen);

                let offset = 0;
                for (const arr of arrays) {
                    if (ArrayBuffer.isView(merged)) {
                        merged.set(arr, offset);
                    } else {
                        for (let i = 0; i < arr.length; i++) merged[offset + i] = arr[i];
                    }
                    offset += arr.length;
                }
                mergedColumns[name] = merged;
            }
        }

        const result = { schema: table.schema, columns: mergedColumns, rowCount: totalRows };
        this._columnarCache.set(tableName, { version, data: result });
        return result;
    }

    /**
     * Read column data directly (no row conversion) for fast aggregation.
     * Returns typed array for numeric columns.
     */
    async _readColumn(tableName, colName) {
        const table = this.tables.get(tableName);
        if (!table) return null;

        const buffer = this._writeBuffer.get(tableName);
        const colData = [];

        // Read from persisted fragments (columnar)
        for (const fragKey of table.fragments) {
            const fragData = await opfsStorage.load(fragKey);
            if (!fragData) continue;

            const binary = parseBinaryColumnar(fragData);
            if (binary && binary.columns[colName]) {
                // Direct typed array access - no row conversion!
                const arr = binary.columns[colName];
                if (arr.length > 0) colData.push(arr);
            }
        }

        // Include buffered rows (need to extract column)
        if (buffer && buffer.length > 0) {
            const bufCol = new Float64Array(buffer.length);
            for (let i = 0; i < buffer.length; i++) {
                const v = buffer[i][colName];
                bufCol[i] = typeof v === 'number' ? v : 0;
            }
            colData.push(bufCol);
        }

        // Merge all fragments into single typed array
        if (colData.length === 0) return new Float64Array(0);
        if (colData.length === 1) return colData[0];

        // Merge multiple fragments
        const totalLen = colData.reduce((sum, arr) => sum + arr.length, 0);
        const merged = new Float64Array(totalLen);
        let offset = 0;
        for (const arr of colData) {
            merged.set(arr, offset);
            offset += arr.length;
        }
        return merged;
    }

    /**
     * Parse binary columnar format - optimized for typed arrays
     */
    _parseBinaryColumnar(data) {
        const { schema, columns, rowCount } = data;

        // Pre-allocate array
        const rows = new Array(rowCount);
        const colNames = schema.map(c => c.name);
        const colArrays = colNames.map(name => columns[name]);
        const numCols = colNames.length;

        // Build rows - typed arrays provide direct indexed access
        for (let i = 0; i < rowCount; i++) {
            const row = {};
            for (let j = 0; j < numCols; j++) {
                row[colNames[j]] = colArrays[j][i] ?? null;
            }
            rows[i] = row;
        }

        return rows;
    }

    _parseJsonColumnar(data) {
        const { schema, columns, rowCount } = data;

        // Pre-allocate array for better performance
        const rows = new Array(rowCount);
        const colNames = schema.map(c => c.name);
        const colArrays = colNames.map(name => columns[name] || []);
        const numCols = colNames.length;

        // Build rows with direct array access (faster than object property lookup)
        for (let i = 0; i < rowCount; i++) {
            const row = {};
            for (let j = 0; j < numCols; j++) {
                row[colNames[j]] = colArrays[j][i] ?? null;
            }
            rows[i] = row;
        }

        return rows;
    }

    getTable(tableName) {
        return this.tables.get(tableName);
    }

    listTables() {
        return Array.from(this.tables.keys());
    }

    /**
     * Get fragment paths for direct WASM access (no row conversion)
     */
    getFragmentPaths(tableName) {
        const table = this.tables.get(tableName);
        if (!table) return [];
        return table.fragments;
    }

    /**
     * Get column index by name for a table
     */
    getColumnIndex(tableName, colName) {
        const table = this.tables.get(tableName);
        if (!table) return -1;
        // Schema includes __rowId at index 0
        const idx = table.schema.findIndex(c => c.name === colName);
        return idx >= 0 ? idx + 1 : -1; // +1 for __rowId
    }

    /**
     * Check if table has buffered (unflushed) data
     */
    hasBufferedData(tableName) {
        const colBuf = this._columnarBuffer.get(tableName);
        return colBuf && (colBuf.__length || 0) > 0;
    }

    async compact() {
        for (const [tableName, table] of this.tables) {
            const allRows = await this._readAllRows(tableName);

            for (const fragKey of table.fragments) {
                await opfsStorage.delete(fragKey);
            }

            table.fragments = [];
            table.deletionVector = [];
            table.rowCount = 0;
            table.nextRowId = 0;

            if (allRows.length > 0) {
                const cleanRows = allRows.map(({ __rowId, ...rest }) => rest);
                await this.insert(tableName, cleanRows);
            }
        }

        return { success: true, compacted: this.tables.size };
    }

    // Start a scan stream
    async scanStart(tableName, options = {}) {
        if (!this.tables.has(tableName)) {
            throw new Error(`Table '${tableName}' does not exist`);
        }

        const streamId = nextScanId++;
        const table = this.tables.get(tableName);
        const deletedSet = new Set(table.deletionVector);

        // Collect all rows
        const allRows = [];
        for (const fragKey of table.fragments) {
            const fragData = await opfsStorage.load(fragKey);
            if (fragData) {
                const rows = this._parseFragment(fragData, table.schema);
                for (const row of rows) {
                    if (!deletedSet.has(row.__rowId)) {
                        allRows.push(row);
                    }
                }
            }
        }

        // Add buffered rows
        const buffer = this._writeBuffer.get(tableName);
        if (buffer) {
            for (const row of buffer) {
                if (!deletedSet.has(row.__rowId)) {
                    allRows.push(row);
                }
            }
        }

        scanStreams.set(streamId, {
            rows: allRows,
            index: 0,
            batchSize: options.batchSize || 10000,
            columns: options.columns,
        });

        return streamId;
    }

    // Get next batch from scan stream
    scanNext(streamId) {
        const stream = scanStreams.get(streamId);
        if (!stream) {
            return { batch: [], done: true };
        }

        const batch = [];
        const end = Math.min(stream.index + stream.batchSize, stream.rows.length);

        for (let i = stream.index; i < end; i++) {
            const row = stream.rows[i];
            let projectedRow;

            if (stream.columns && stream.columns.length > 0 && stream.columns[0] !== '*') {
                projectedRow = {};
                for (const col of stream.columns) {
                    projectedRow[col] = row[col];
                }
            } else {
                const { __rowId, ...rest } = row;
                projectedRow = rest;
            }

            batch.push(projectedRow);
        }

        stream.index = end;

        const done = stream.index >= stream.rows.length;
        if (done) {
            scanStreams.delete(streamId);
        }

        return { batch, done };
    }
}
