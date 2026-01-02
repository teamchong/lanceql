/**
 * LanceData - Base classes and factory
 * Extracted from lance-data.js for modularity
 */

import { hotTierCache } from '../cache/hot-tier-cache.js';

// Forward declarations - resolved at runtime
let ChunkedLanceReader, LocalDatabase, opfsStorage, RemoteLanceFile;

/**
 * Load dependencies lazily to avoid circular imports
 */
async function loadDeps() {
    if (!ChunkedLanceReader) {
        const storageModule = await import('../storage/lance-reader.js');
        ChunkedLanceReader = storageModule.ChunkedLanceReader;
    }
    if (!LocalDatabase) {
        const dbModule = await import('../database/local-database.js');
        LocalDatabase = dbModule.LocalDatabase;
    }
    if (!opfsStorage) {
        const opfsModule = await import('../storage/opfs.js');
        opfsStorage = opfsModule.opfsStorage;
    }
    if (!RemoteLanceFile) {
        const remoteModule = await import('./remote-file.js');
        RemoteLanceFile = remoteModule.RemoteLanceFile;
    }
}

/**
 * Base class for Lance data sources.
 * Defines the common interface for local and remote data.
 */
class LanceDataBase {
    constructor(type) {
        this.type = type; // 'local' | 'remote' | 'cached'
    }

    // Abstract methods - must be implemented by subclasses
    async getSchema() { throw new Error('Not implemented'); }
    async getRowCount() { throw new Error('Not implemented'); }
    async readColumn(colIdx, start = 0, count = null) { throw new Error('Not implemented'); }
    async *scan(options = {}) { throw new Error('Not implemented'); }

    // Optional methods
    async insert(rows) { throw new Error('Write not supported for this source'); }
    isCached() { return false; }
    async prefetch() { }
    async evict() { }
    async close() { }
}

/**
 * OPFS-backed Lance data for local files.
 * Uses ChunkedLanceReader for efficient memory usage.
 */
class OPFSLanceData extends LanceDataBase {
    constructor(path, storage = null) {
        super('local');
        this.path = path;
        this.storage = storage;
        this.reader = null;
        this.database = null;
        this._isDatabase = false;
    }

    async open() {
        await loadDeps();
        this.storage = this.storage || opfsStorage;

        // Check if it's a database (directory with manifest)
        const manifestPath = `${this.path}/__manifest__`;
        if (await this.storage.exists(manifestPath)) {
            this._isDatabase = true;
            this.database = new LocalDatabase(this.path, this.storage);
            await this.database.open();
        } else {
            // Single Lance file
            this.reader = await ChunkedLanceReader.open(this.storage, this.path);
        }
        return this;
    }

    async getSchema() {
        if (this._isDatabase) {
            const tables = this.database.listTables();
            if (tables.length === 0) return [];
            return this.database.getSchema(tables[0]);
        }
        return Array.from({ length: this.reader.getNumColumns() }, (_, i) => ({
            name: `col_${i}`,
            type: 'unknown'
        }));
    }

    async getRowCount() {
        if (this._isDatabase) {
            const tables = this.database.listTables();
            if (tables.length === 0) return 0;
            return this.database.count(tables[0]);
        }
        const meta = await this.reader.readColumnMetaRaw(0);
        return 0;
    }

    async readColumn(colIdx, start = 0, count = null) {
        if (this._isDatabase) {
            throw new Error('Use select() for database queries');
        }
        return this.reader.readColumnMetaRaw(colIdx);
    }

    async *scan(options = {}) {
        if (this._isDatabase) {
            const tables = this.database.listTables();
            if (tables.length === 0) return;
            yield* this.database.scan(tables[0], options);
        } else {
            throw new Error('scan() requires database, use readColumn() for single files');
        }
    }

    async insert(rows) {
        if (!this._isDatabase) {
            throw new Error('insert() requires database');
        }
        const tables = this.database.listTables();
        if (tables.length === 0) {
            throw new Error('No tables in database');
        }
        return this.database.insert(tables[0], rows);
    }

    isCached() {
        return true;
    }

    async close() {
        if (this.reader) {
            this.reader.close();
        }
        if (this.database) {
            await this.database.close();
        }
    }
}

/**
 * HTTP-backed Lance data for remote files.
 * Uses HotTierCache for OPFS caching.
 */
class RemoteLanceData extends LanceDataBase {
    constructor(url) {
        super('remote');
        this.url = url;
        this.remoteFile = null;
        this.cachedPath = null;
    }

    async open() {
        await loadDeps();

        // Check if already cached
        const cacheInfo = await hotTierCache.getCacheInfo(this.url);
        if (cacheInfo && cacheInfo.complete) {
            this.type = 'cached';
            this.cachedPath = cacheInfo.path;
        }

        // Open remote file
        if (RemoteLanceFile) {
            this.remoteFile = await RemoteLanceFile.open(null, this.url);
        }

        return this;
    }

    async getSchema() {
        if (!this.remoteFile) {
            return [];
        }
        const numCols = this.remoteFile.numColumns;
        const schema = [];
        for (let i = 0; i < numCols; i++) {
            const type = await this.remoteFile.getColumnType?.(i) || 'unknown';
            schema.push({ name: `col_${i}`, type });
        }
        return schema;
    }

    async getRowCount() {
        if (!this.remoteFile) return 0;
        return this.remoteFile.getRowCount?.(0) || 0;
    }

    async readColumn(colIdx, start = 0, count = null) {
        if (!this.remoteFile) {
            throw new Error('Remote file not opened');
        }
        const type = await this.remoteFile.getColumnType?.(colIdx) || 'unknown';
        if (type.includes('int64')) {
            return this.remoteFile.readInt64Column?.(colIdx, count);
        } else if (type.includes('float64')) {
            return this.remoteFile.readFloat64Column?.(colIdx, count);
        } else if (type.includes('string')) {
            return this.remoteFile.readStrings?.(colIdx, count);
        }
        throw new Error(`Unsupported column type: ${type}`);
    }

    async *scan(options = {}) {
        const batchSize = options.batchSize || 10000;
        const rowCount = await this.getRowCount();
        const schema = await this.getSchema();

        for (let offset = 0; offset < rowCount; offset += batchSize) {
            const count = Math.min(batchSize, rowCount - offset);
            const batch = [];

            const columns = {};
            for (let i = 0; i < schema.length; i++) {
                columns[schema[i].name] = await this.readColumn(i, offset, count);
            }

            for (let r = 0; r < count; r++) {
                const row = {};
                for (const name of Object.keys(columns)) {
                    row[name] = columns[name][r];
                }
                if (!options.where || options.where(row)) {
                    batch.push(row);
                }
            }

            yield batch;
        }
    }

    isCached() {
        return this.type === 'cached';
    }

    async prefetch() {
        await hotTierCache.cache(this.url);
        const cacheInfo = await hotTierCache.getCacheInfo(this.url);
        if (cacheInfo && cacheInfo.complete) {
            this.type = 'cached';
            this.cachedPath = cacheInfo.path;
        }
    }

    async evict() {
        await hotTierCache.evict(this.url);
        this.type = 'remote';
        this.cachedPath = null;
    }

    async close() {
        if (this.remoteFile) {
            this.remoteFile.close();
        }
    }
}

/**
 * Factory function to open Lance data from any source.
 * @param {string} source - Data source URI
 * @returns {Promise<LanceDataBase>}
 */
async function openLance(source) {
    if (source.startsWith('opfs://')) {
        const path = source.slice(7);
        const data = new OPFSLanceData(path);
        await data.open();
        return data;
    } else if (source.startsWith('http://') || source.startsWith('https://')) {
        const data = new RemoteLanceData(source);
        await data.open();
        return data;
    } else {
        const data = new OPFSLanceData(source);
        await data.open();
        return data;
    }
}


export { LanceDataBase, OPFSLanceData, RemoteLanceData, openLance };
