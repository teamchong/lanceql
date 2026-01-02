/**
 * WorkerDatabase - SQL database operations with OPFS persistence
 */

import { opfsStorage } from './opfs-storage.js';
import { DataType, LanceFileWriter, E, D } from './data-types.js';

// Streaming scan state
const scanStreams = new Map();
let nextScanId = 1;

export class WorkerDatabase {
    constructor(name) {
        this.name = name;
        this.tables = new Map();
        this.version = 0;
        this.manifestKey = `${name}/__manifest__`;

        // Write buffer for fast inserts
        this._writeBuffer = new Map();
        this._flushTimer = null;
        this._flushInterval = 1000;
        this._flushThreshold = 1000;
        this._flushing = false;

        // Read cache
        this._readCache = new Map();
    }

    async open() {
        // Load manifest from storage
        const manifestData = await opfsStorage.load(this.manifestKey);
        if (manifestData) {
            const manifest = JSON.parse(D.decode(manifestData));
            this.version = manifest.version || 0;
            this.tables = new Map(Object.entries(manifest.tables || {}));
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
            createdAt: Date.now(),
        };

        this.tables.set(tableName, tableState);
        await this._saveManifest();

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

        const rowsWithIds = rows.map(row => ({
            __rowId: table.nextRowId++,
            ...row,
        }));

        if (!this._writeBuffer.has(tableName)) {
            this._writeBuffer.set(tableName, []);
        }
        this._writeBuffer.get(tableName).push(...rowsWithIds);
        table.rowCount += rows.length;

        this._scheduleFlush();

        const bufferSize = this._writeBuffer.get(tableName).length;
        if (bufferSize >= this._flushThreshold) {
            await this._flushTable(tableName);
        }

        return { success: true, inserted: rows.length };
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
            const tables = [...this._writeBuffer.keys()];
            for (const tableName of tables) {
                await this._flushTable(tableName);
            }
        } finally {
            this._flushing = false;
        }
    }

    async _flushTable(tableName) {
        const buffer = this._writeBuffer.get(tableName);
        if (!buffer || buffer.length === 0) return;

        const table = this.tables.get(tableName);
        if (!table) return;

        const rowsToFlush = buffer.splice(0, buffer.length);

        const schemaWithRowId = [
            { name: '__rowId', type: 'int64', primaryKey: true },
            ...table.schema.filter(c => c.name !== '__rowId')
        ];

        const writer = new LanceFileWriter(schemaWithRowId);
        writer.addRows(rowsToFlush);
        const lanceData = writer.build();

        const fragKey = `${this.name}/${tableName}/frag_${Date.now()}_${Math.random().toString(36).slice(2)}.lance`;
        await opfsStorage.save(fragKey, lanceData);

        table.fragments.push(fragKey);
        await this._saveManifest();
    }

    async delete(tableName, predicateFn) {
        if (!this.tables.has(tableName)) {
            throw new Error(`Table '${tableName}' does not exist`);
        }

        const table = this.tables.get(tableName);
        let deletedCount = 0;

        // Delete from buffer
        const buffer = this._writeBuffer.get(tableName);
        if (buffer && buffer.length > 0) {
            const originalLen = buffer.length;
            const remaining = buffer.filter(row => !predicateFn(row));
            buffer.length = 0;
            buffer.push(...remaining);
            deletedCount += (originalLen - remaining.length);
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
        await this._saveManifest();

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

        return { success: true, updated: updatedCount };
    }

    async updateWithExpr(tableName, updateExprs, predicateFn, evalExpr) {
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
                    for (const [col, expr] of Object.entries(updateExprs)) {
                        row[col] = evalExpr(expr, row);
                    }
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

        // Column projection
        if (options.columns && options.columns.length > 0 && options.columns[0] !== '*') {
            rows = rows.map(row => {
                const projected = {};
                for (const col of options.columns) {
                    projected[col] = row[col];
                }
                return projected;
            });
        }

        // Remove internal __rowId
        rows = rows.map(row => {
            const { __rowId, ...rest } = row;
            return rest;
        });

        return rows;
    }

    async _readAllRows(tableName) {
        const table = this.tables.get(tableName);
        const deletedSet = new Set(table.deletionVector);
        const allRows = [];

        // Read from persisted fragments
        for (const fragKey of table.fragments) {
            let rows = this._readCache.get(fragKey);

            if (!rows) {
                const fragData = await opfsStorage.load(fragKey);
                if (fragData) {
                    rows = this._parseFragment(fragData, table.schema);
                    this._readCache.set(fragKey, rows);
                } else {
                    rows = [];
                }
            }

            for (const row of rows) {
                if (!deletedSet.has(row.__rowId)) {
                    allRows.push(row);
                }
            }
        }

        // Include buffered rows
        const buffer = this._writeBuffer.get(tableName);
        if (buffer && buffer.length > 0) {
            for (const row of buffer) {
                if (!deletedSet.has(row.__rowId)) {
                    allRows.push(row);
                }
            }
        }

        return allRows;
    }

    _parseFragment(data, schema) {
        try {
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

    _parseJsonColumnar(data) {
        const { schema, columns, rowCount } = data;
        const rows = [];

        for (let i = 0; i < rowCount; i++) {
            const row = {};
            for (const col of schema) {
                row[col.name] = columns[col.name]?.[i] ?? null;
            }
            rows.push(row);
        }

        return rows;
    }

    getTable(tableName) {
        return this.tables.get(tableName);
    }

    listTables() {
        return Array.from(this.tables.keys());
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
