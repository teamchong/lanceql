/**
 * LanceQL Unified SharedWorker
 *
 * Handles all OPFS operations and heavy computation in a single worker shared across tabs:
 * - Store operations (key-value storage)
 * - LocalDatabase operations (SQL, Lance tables)
 * - WebGPU transformer (shared model for embeddings)
 *
 * Benefits: shared GPU model, responsive UI, efficient OPFS access via createSyncAccessHandle
 */

// ============================================================================
// Shared State
// ============================================================================

// Store instances (one per store name)
const stores = new Map();

// Database instances (one per database name)
const databases = new Map();

// Connected ports
const ports = new Set();

// WebGPU transformer (lazy loaded, shared across all tabs)
let gpuTransformer = null;
let gpuTransformerPromise = null;

// Embedding cache (shared across all tabs)
const embeddingCache = new Map();

// Streaming scan state
const scanStreams = new Map();
let nextScanId = 1;

// Text encoder/decoder
const E = new TextEncoder();
const D = new TextDecoder();

// SharedArrayBuffer for zero-copy responses (set by main thread)
let sharedBuffer = null;
let sharedOffset = 0;

// Large response threshold (use shared buffer for responses > 1KB)
const SHARED_THRESHOLD = 1024;

// ============================================================================
// OPFSStorage - File system operations
// ============================================================================

class OPFSStorage {
    constructor(rootDir = 'lanceql') {
        this.rootDir = rootDir;
        this.root = null;
    }

    async getRoot() {
        if (this.root) return this.root;

        if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
            throw new Error('OPFS not available');
        }

        const opfsRoot = await navigator.storage.getDirectory();
        this.root = await opfsRoot.getDirectoryHandle(this.rootDir, { create: true });
        return this.root;
    }

    async open() {
        await this.getRoot();
        return this;
    }

    async getDir(path) {
        const root = await this.getRoot();
        const parts = path.split('/').filter(p => p);

        let current = root;
        for (const part of parts) {
            current = await current.getDirectoryHandle(part, { create: true });
        }
        return current;
    }

    async save(path, data) {
        const parts = path.split('/');
        const fileName = parts.pop();
        const dirPath = parts.join('/');

        const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
        const fileHandle = await dir.getFileHandle(fileName, { create: true });

        // Try sync access handle (faster, requires worker context)
        if (fileHandle.createSyncAccessHandle) {
            try {
                const accessHandle = await fileHandle.createSyncAccessHandle();
                accessHandle.truncate(0);
                accessHandle.write(data, { at: 0 });
                accessHandle.flush();
                accessHandle.close();
                return { path, size: data.byteLength };
            } catch (e) {
                // Fall back to writable stream
            }
        }

        const writable = await fileHandle.createWritable();
        await writable.write(data);
        await writable.close();

        return { path, size: data.byteLength };
    }

    async load(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const fileHandle = await dir.getFileHandle(fileName);
            const file = await fileHandle.getFile();
            const buffer = await file.arrayBuffer();
            return new Uint8Array(buffer);
        } catch (e) {
            if (e.name === 'NotFoundError') return null;
            throw e;
        }
    }

    async delete(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            await dir.removeEntry(fileName);
            return true;
        } catch (e) {
            if (e.name === 'NotFoundError') return false;
            throw e;
        }
    }

    async list(dirPath = '') {
        try {
            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            const files = [];
            for await (const [name, handle] of dir.entries()) {
                files.push({ name, type: handle.kind });
            }
            return files;
        } catch (e) {
            return [];
        }
    }

    async exists(path) {
        try {
            const parts = path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/');

            const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
            await dir.getFileHandle(fileName);
            return true;
        } catch (e) {
            return false;
        }
    }

    async deleteDir(dirPath) {
        try {
            const parts = dirPath.split('/');
            const dirName = parts.pop();
            const parentPath = parts.join('/');

            const parent = parentPath ? await this.getDir(parentPath) : await this.getRoot();
            await parent.removeEntry(dirName, { recursive: true });
            return true;
        } catch (e) {
            return false;
        }
    }
}

// Shared OPFS storage instance
const opfsStorage = new OPFSStorage();

// ============================================================================
// Encryption helpers (AES-256-GCM)
// ============================================================================

const encryptionKeys = new Map();  // keyId -> CryptoKey

async function importEncryptionKey(keyBytes) {
    return crypto.subtle.importKey(
        'raw',
        new Uint8Array(keyBytes),
        { name: 'AES-GCM', length: 256 },
        false,
        ['encrypt', 'decrypt']
    );
}

async function encryptData(data, cryptoKey) {
    const iv = crypto.getRandomValues(new Uint8Array(12));  // 96-bit IV for GCM
    const encoder = new TextEncoder();
    const plaintext = encoder.encode(JSON.stringify(data));

    const ciphertext = await crypto.subtle.encrypt(
        { name: 'AES-GCM', iv },
        cryptoKey,
        plaintext
    );

    // Format: [iv (12 bytes)][ciphertext]
    const result = new Uint8Array(12 + ciphertext.byteLength);
    result.set(iv, 0);
    result.set(new Uint8Array(ciphertext), 12);
    return result;
}

async function decryptData(encrypted, cryptoKey) {
    const iv = encrypted.slice(0, 12);
    const ciphertext = encrypted.slice(12);

    const plaintext = await crypto.subtle.decrypt(
        { name: 'AES-GCM', iv },
        cryptoKey,
        ciphertext
    );

    const decoder = new TextDecoder();
    return JSON.parse(decoder.decode(plaintext));
}

// ============================================================================
// WorkerStore - Key-value storage (for Store API)
// ============================================================================

class WorkerStore {
    constructor(name, options = {}) {
        this.name = name;
        this.options = options;
        this._root = null;
        this._ready = false;
        this._embedder = null;
        this._encryptionKeyId = null;
    }

    async open(encryptionConfig = null) {
        if (this._ready) return this;

        // Set up encryption if provided
        if (encryptionConfig) {
            const { keyId, keyBytes } = encryptionConfig;
            if (!encryptionKeys.has(keyId)) {
                const cryptoKey = await importEncryptionKey(keyBytes);
                encryptionKeys.set(keyId, cryptoKey);
            }
            this._encryptionKeyId = keyId;
        }

        try {
            const opfsRoot = await navigator.storage.getDirectory();
            this._root = await opfsRoot.getDirectoryHandle(`lanceql-${this.name}`, { create: true });
            this._ready = true;
        } catch (e) {
            console.error('[WorkerStore] Failed to open OPFS:', e);
            throw e;
        }

        return this;
    }

    _getCryptoKey() {
        return this._encryptionKeyId ? encryptionKeys.get(this._encryptionKeyId) : null;
    }

    async get(key) {
        await this._ensureOpen();

        try {
            const cryptoKey = this._getCryptoKey();
            const ext = cryptoKey ? '.enc' : '.json';
            const fileHandle = await this._root.getFileHandle(`${key}${ext}`);
            const file = await fileHandle.getFile();

            if (cryptoKey) {
                const buffer = await file.arrayBuffer();
                return decryptData(new Uint8Array(buffer), cryptoKey);
            } else {
                const text = await file.text();
                return JSON.parse(text);
            }
        } catch (e) {
            if (e.name === 'NotFoundError') return undefined;
            throw e;
        }
    }

    async set(key, value) {
        await this._ensureOpen();

        const cryptoKey = this._getCryptoKey();
        const ext = cryptoKey ? '.enc' : '.json';
        const fileHandle = await this._root.getFileHandle(`${key}${ext}`, { create: true });
        const writable = await fileHandle.createWritable();

        if (cryptoKey) {
            const encrypted = await encryptData(value, cryptoKey);
            await writable.write(encrypted);
        } else {
            await writable.write(JSON.stringify(value));
        }

        await writable.close();
    }

    async delete(key) {
        await this._ensureOpen();

        const cryptoKey = this._getCryptoKey();
        const ext = cryptoKey ? '.enc' : '.json';

        try {
            await this._root.removeEntry(`${key}${ext}`);
        } catch (e) {
            if (e.name !== 'NotFoundError') throw e;
        }
    }

    async keys() {
        await this._ensureOpen();

        const cryptoKey = this._getCryptoKey();
        const ext = cryptoKey ? '.enc' : '.json';

        const keys = [];
        for await (const [name] of this._root.entries()) {
            if (name.endsWith(ext)) {
                keys.push(name.slice(0, -ext.length));
            }
        }
        return keys;
    }

    async clear() {
        await this._ensureOpen();

        const entries = [];
        for await (const [name] of this._root.entries()) {
            entries.push(name);
        }

        for (const name of entries) {
            await this._root.removeEntry(name);
        }
    }

    async filter(key, query) {
        const value = await this.get(key);
        if (!Array.isArray(value)) {
            throw new Error(`Key '${key}' is not a collection`);
        }

        return value.filter(item => this._matchQuery(item, query));
    }

    async find(key, query) {
        const value = await this.get(key);
        if (!Array.isArray(value)) {
            throw new Error(`Key '${key}' is not a collection`);
        }

        return value.find(item => this._matchQuery(item, query));
    }

    async search(key, text, limit = 10) {
        const value = await this.get(key);
        if (!Array.isArray(value)) {
            throw new Error(`Key '${key}' is not a collection`);
        }

        if (value.length === 0) return [];

        // Use semantic search if embedder is available
        if (this._embedder) {
            return this._semanticSearch(value, key, text, limit);
        }

        // Fallback to text matching
        const textLower = text.toLowerCase();
        const scored = value.map(item => {
            const itemText = this._extractText(item).toLowerCase();
            const words = textLower.split(/\s+/);
            const matchCount = words.filter(w => itemText.includes(w)).length;
            return { item, score: matchCount / words.length };
        });

        return scored
            .filter(s => s.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    }

    async _semanticSearch(value, key, text, limit) {
        const queryVec = await this._embedder.embed(text);
        const scored = [];

        // Batch encode items that aren't cached
        const textsToEmbed = [];
        const itemIndices = [];

        for (let i = 0; i < value.length; i++) {
            const item = value[i];
            const itemText = this._extractText(item);
            const cacheKey = `${this.name}:${key}:${itemText}`;

            if (embeddingCache.has(cacheKey)) {
                const cachedVec = embeddingCache.get(cacheKey);
                const score = this._cosineSimilarity(queryVec, cachedVec);
                scored.push({ item, score });
            } else {
                textsToEmbed.push(itemText);
                itemIndices.push(i);
            }
        }

        // Batch encode
        if (textsToEmbed.length > 0) {
            let itemVecs;
            if (textsToEmbed.length > 1 && this._embedder.embedBatch) {
                itemVecs = await this._embedder.embedBatch(textsToEmbed);
            } else {
                itemVecs = await Promise.all(textsToEmbed.map(t => this._embedder.embed(t)));
            }

            for (let j = 0; j < itemVecs.length; j++) {
                const idx = itemIndices[j];
                const item = value[idx];
                const itemText = textsToEmbed[j];
                const itemVec = itemVecs[j];
                const cacheKey = `${this.name}:${key}:${itemText}`;

                embeddingCache.set(cacheKey, itemVec);

                const score = this._cosineSimilarity(queryVec, itemVec);
                scored.push({ item, score });
            }
        }

        return scored
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    }

    async enableSemanticSearch(options = {}) {
        const { model = 'minilm', onProgress } = options;

        // Initialize WebGPU transformer (shared)
        if (!gpuTransformer) {
            if (!gpuTransformerPromise) {
                gpuTransformerPromise = this._initGPUTransformer();
            }
            gpuTransformer = await gpuTransformerPromise;
        }

        if (!gpuTransformer) {
            return null; // WebGPU not available
        }

        // Load model
        const modelConfig = await gpuTransformer.loadModel(model, onProgress);

        this._embedder = {
            model,
            dimensions: modelConfig.hiddenSize,
            embed: async (text) => gpuTransformer.encodeText(text, model),
            embedBatch: async (texts) => gpuTransformer.encodeTextBatch(texts, model),
        };

        return {
            model,
            dimensions: modelConfig.hiddenSize,
            type: modelConfig.modelType || 'text',
        };
    }

    async _initGPUTransformer() {
        try {
            // Dynamic import of WebGPU module
            const webgpu = await import('./webgpu/index.js');

            if (!webgpu.isWebGPUAvailable()) {
                console.log('[WorkerStore] WebGPU not available');
                return null;
            }

            const transformer = webgpu.getGPUTransformer();
            const available = await transformer.init();

            if (!available) {
                console.log('[WorkerStore] WebGPU init failed');
                return null;
            }

            console.log('[WorkerStore] WebGPU initialized');
            return transformer;
        } catch (e) {
            console.error('[WorkerStore] WebGPU init error:', e);
            return null;
        }
    }

    disableSemanticSearch() {
        if (this._embedder && gpuTransformer) {
            gpuTransformer.unloadModel(this._embedder.model);
            this._embedder = null;
        }
    }

    hasSemanticSearch() {
        return this._embedder !== null;
    }

    async count(key, query = null) {
        const value = await this.get(key);
        if (!Array.isArray(value)) {
            throw new Error(`Key '${key}' is not a collection`);
        }

        if (!query) return value.length;
        return value.filter(item => this._matchQuery(item, query)).length;
    }

    async _ensureOpen() {
        if (!this._ready) {
            await this.open();
        }
    }

    _matchQuery(item, query) {
        for (const [field, condition] of Object.entries(query)) {
            const value = item[field];

            if (typeof condition === 'object' && condition !== null) {
                for (const [op, opVal] of Object.entries(condition)) {
                    switch (op) {
                        case '$eq': if (value !== opVal) return false; break;
                        case '$ne': if (value === opVal) return false; break;
                        case '$lt': if (!(value < opVal)) return false; break;
                        case '$lte': if (!(value <= opVal)) return false; break;
                        case '$gt': if (!(value > opVal)) return false; break;
                        case '$gte': if (!(value >= opVal)) return false; break;
                        case '$in': if (!Array.isArray(opVal) || !opVal.includes(value)) return false; break;
                        case '$nin': if (Array.isArray(opVal) && opVal.includes(value)) return false; break;
                        case '$contains': if (typeof value !== 'string' || !value.includes(opVal)) return false; break;
                        case '$regex': if (typeof value !== 'string' || !new RegExp(opVal).test(value)) return false; break;
                    }
                }
            } else {
                if (value !== condition) return false;
            }
        }
        return true;
    }

    _extractText(item) {
        if (typeof item === 'string') return item;
        const texts = [];
        for (const [key, val] of Object.entries(item)) {
            if (typeof val === 'string') texts.push(val);
        }
        return texts.join(' ');
    }

    _cosineSimilarity(a, b) {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}

// ============================================================================
// Data Types
// ============================================================================

const DataType = {
    INT64: 'int64',
    INT32: 'int32',
    FLOAT64: 'float64',
    FLOAT32: 'float32',
    STRING: 'string',
    BOOL: 'bool',
    VECTOR: 'vector',
};

// ============================================================================
// LanceFileWriter - Write Lance file format
// ============================================================================

class LanceFileWriter {
    constructor(schema) {
        this.schema = schema;
        this.columns = new Map();
        this.rowCount = 0;
    }

    addRows(rows) {
        for (const row of rows) {
            for (const col of this.schema) {
                if (!this.columns.has(col.name)) {
                    this.columns.set(col.name, []);
                }
                this.columns.get(col.name).push(row[col.name] ?? null);
            }
            this.rowCount++;
        }
    }

    build() {
        // JSON columnar format (reliable fallback)
        const data = {
            format: 'json',
            schema: this.schema,
            columns: {},
            rowCount: this.rowCount,
        };

        for (const [name, values] of this.columns) {
            data.columns[name] = values;
        }

        return E.encode(JSON.stringify(data));
    }
}

// ============================================================================
// WorkerDatabase - SQL database operations
// ============================================================================

class WorkerDatabase {
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

// ============================================================================
// SQL Lexer and Parser
// ============================================================================

const TokenType = {
    // Keywords
    SELECT: 'SELECT', FROM: 'FROM', WHERE: 'WHERE', INSERT: 'INSERT',
    INTO: 'INTO', VALUES: 'VALUES', UPDATE: 'UPDATE', SET: 'SET',
    DELETE: 'DELETE', CREATE: 'CREATE', DROP: 'DROP', TABLE: 'TABLE',
    IF: 'IF', EXISTS: 'EXISTS', NOT: 'NOT', PRIMARY: 'PRIMARY',
    KEY: 'KEY', ORDER: 'ORDER', BY: 'BY', ASC: 'ASC', DESC: 'DESC',
    LIMIT: 'LIMIT', OFFSET: 'OFFSET', AND: 'AND', OR: 'OR',
    NULL: 'NULL', TRUE: 'TRUE', FALSE: 'FALSE', LIKE: 'LIKE',
    // JOIN keywords
    JOIN: 'JOIN', LEFT: 'LEFT', RIGHT: 'RIGHT', INNER: 'INNER', ON: 'ON', AS: 'AS',
    // GROUP BY / HAVING
    GROUP: 'GROUP', HAVING: 'HAVING',
    // Aggregate functions
    COUNT: 'COUNT', SUM: 'SUM', AVG: 'AVG', MIN: 'MIN', MAX: 'MAX',
    // Additional operators
    DISTINCT: 'DISTINCT', BETWEEN: 'BETWEEN', IN: 'IN',
    // Vector search
    NEAR: 'NEAR', TOPK: 'TOPK',

    // Literals
    IDENTIFIER: 'IDENTIFIER', STRING: 'STRING', NUMBER: 'NUMBER',

    // Operators
    EQ: '=', NE: '!=', LT: '<', LE: '<=', GT: '>', GE: '>=',
    STAR: '*', COMMA: ',', LPAREN: '(', RPAREN: ')',
    LBRACKET: '[', RBRACKET: ']', DOT: '.',

    // Special
    EOF: 'EOF',
};

class SQLLexer {
    constructor(sql) {
        this.sql = sql;
        this.pos = 0;
    }

    tokenize() {
        const tokens = [];

        while (this.pos < this.sql.length) {
            this._skipWhitespace();
            if (this.pos >= this.sql.length) break;

            const token = this._nextToken();
            if (token) tokens.push(token);
        }

        tokens.push({ type: TokenType.EOF });
        return tokens;
    }

    _skipWhitespace() {
        while (this.pos < this.sql.length && /\s/.test(this.sql[this.pos])) {
            this.pos++;
        }
    }

    _nextToken() {
        const ch = this.sql[this.pos];

        // Single character tokens
        const singleChars = {
            '*': TokenType.STAR,
            ',': TokenType.COMMA,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            '=': TokenType.EQ,
            '<': TokenType.LT,
            '>': TokenType.GT,
            '.': TokenType.DOT,
        };

        if (singleChars[ch]) {
            this.pos++;

            // Check for multi-char operators
            if (ch === '<' && this.sql[this.pos] === '=') {
                this.pos++;
                return { type: TokenType.LE };
            }
            if (ch === '>' && this.sql[this.pos] === '=') {
                this.pos++;
                return { type: TokenType.GE };
            }
            if (ch === '!' && this.sql[this.pos] === '=') {
                this.pos++;
                return { type: TokenType.NE };
            }
            if (ch === '<' && this.sql[this.pos] === '>') {
                this.pos++;
                return { type: TokenType.NE };
            }

            return { type: singleChars[ch] };
        }

        if (ch === '!') {
            this.pos++;
            if (this.sql[this.pos] === '=') {
                this.pos++;
                return { type: TokenType.NE };
            }
            throw new Error(`Unexpected character: !`);
        }

        // String literal
        if (ch === "'" || ch === '"') {
            return this._readString(ch);
        }

        // Number
        if (/\d/.test(ch) || (ch === '-' && /\d/.test(this.sql[this.pos + 1]))) {
            return this._readNumber();
        }

        // Identifier or keyword
        if (/[a-zA-Z_]/.test(ch)) {
            return this._readIdentifier();
        }

        throw new Error(`Unexpected character: ${ch}`);
    }

    _readString(quote) {
        this.pos++; // Skip opening quote
        let value = '';

        while (this.pos < this.sql.length && this.sql[this.pos] !== quote) {
            if (this.sql[this.pos] === '\\') {
                this.pos++;
                if (this.pos < this.sql.length) {
                    value += this.sql[this.pos];
                }
            } else {
                value += this.sql[this.pos];
            }
            this.pos++;
        }

        this.pos++; // Skip closing quote
        return { type: TokenType.STRING, value };
    }

    _readNumber() {
        let value = '';
        if (this.sql[this.pos] === '-') {
            value += this.sql[this.pos++];
        }

        while (this.pos < this.sql.length && /[\d.]/.test(this.sql[this.pos])) {
            value += this.sql[this.pos++];
        }

        return { type: TokenType.NUMBER, value };
    }

    _readIdentifier() {
        let value = '';

        while (this.pos < this.sql.length && /[a-zA-Z0-9_]/.test(this.sql[this.pos])) {
            value += this.sql[this.pos++];
        }

        const upper = value.toUpperCase();
        if (TokenType[upper]) {
            return { type: TokenType[upper], value };
        }

        return { type: TokenType.IDENTIFIER, value };
    }
}

class SQLParser {
    constructor(tokens) {
        this.tokens = tokens;
        this.pos = 0;
    }

    peek() {
        return this.tokens[this.pos] || { type: TokenType.EOF };
    }

    advance() {
        return this.tokens[this.pos++] || { type: TokenType.EOF };
    }

    match(type) {
        if (this.peek().type === type) {
            return this.advance();
        }
        return null;
    }

    expect(type) {
        const token = this.advance();
        if (token.type !== type) {
            throw new Error(`Expected ${type}, got ${token.type}`);
        }
        return token;
    }

    parse() {
        const token = this.peek();

        switch (token.type) {
            case TokenType.CREATE:
                return this.parseCreate();
            case TokenType.DROP:
                return this.parseDrop();
            case TokenType.INSERT:
                return this.parseInsert();
            case TokenType.UPDATE:
                return this.parseUpdate();
            case TokenType.DELETE:
                return this.parseDelete();
            case TokenType.SELECT:
                return this.parseSelect();
            default:
                throw new Error(`Unexpected token: ${token.type}`);
        }
    }

    parseCreate() {
        this.expect(TokenType.CREATE);
        this.expect(TokenType.TABLE);

        let ifNotExists = false;
        if (this.match(TokenType.IF)) {
            this.expect(TokenType.NOT);
            this.expect(TokenType.EXISTS);
            ifNotExists = true;
        }

        const tableName = this.expect(TokenType.IDENTIFIER).value;
        this.expect(TokenType.LPAREN);

        const columns = [];
        do {
            const colName = this.expect(TokenType.IDENTIFIER).value;
            const colType = this.parseDataType();
            const col = { name: colName, type: colType };

            if (this.match(TokenType.PRIMARY)) {
                this.expect(TokenType.KEY);
                col.primaryKey = true;
            }

            columns.push(col);
        } while (this.match(TokenType.COMMA));

        this.expect(TokenType.RPAREN);

        return { type: 'CREATE_TABLE', table: tableName, columns, ifNotExists };
    }

    parseDataType() {
        const token = this.advance();
        let type = token.value || token.type;

        // Handle VECTOR(dim)
        if (type.toUpperCase() === 'VECTOR' && this.match(TokenType.LPAREN)) {
            const dim = this.expect(TokenType.NUMBER).value;
            this.expect(TokenType.RPAREN);
            return { type: 'vector', dim: parseInt(dim) };
        }

        // Handle VARCHAR(len)
        if ((type.toUpperCase() === 'VARCHAR' || type.toUpperCase() === 'TEXT') && this.match(TokenType.LPAREN)) {
            this.expect(TokenType.NUMBER);
            this.expect(TokenType.RPAREN);
        }

        return type;
    }

    parseDrop() {
        this.expect(TokenType.DROP);
        this.expect(TokenType.TABLE);

        let ifExists = false;
        if (this.match(TokenType.IF)) {
            this.expect(TokenType.EXISTS);
            ifExists = true;
        }

        const tableName = this.expect(TokenType.IDENTIFIER).value;
        return { type: 'DROP_TABLE', table: tableName, ifExists };
    }

    parseInsert() {
        this.expect(TokenType.INSERT);
        this.expect(TokenType.INTO);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        let columns = null;
        if (this.match(TokenType.LPAREN)) {
            columns = [this.expect(TokenType.IDENTIFIER).value];
            while (this.match(TokenType.COMMA)) {
                columns.push(this.expect(TokenType.IDENTIFIER).value);
            }
            this.expect(TokenType.RPAREN);
        }

        this.expect(TokenType.VALUES);

        const rows = [];
        do {
            this.expect(TokenType.LPAREN);
            const values = [this.parseValue()];
            while (this.match(TokenType.COMMA)) {
                values.push(this.parseValue());
            }
            this.expect(TokenType.RPAREN);

            if (columns) {
                const row = {};
                columns.forEach((col, i) => row[col] = values[i]);
                rows.push(row);
            } else {
                rows.push(values);
            }
        } while (this.match(TokenType.COMMA));

        return { type: 'INSERT', table: tableName, columns, rows };
    }

    parseValue() {
        const token = this.peek();

        if (token.type === TokenType.NUMBER) {
            this.advance();
            const num = parseFloat(token.value);
            return Number.isInteger(num) ? parseInt(token.value) : num;
        }
        if (token.type === TokenType.STRING) {
            this.advance();
            return token.value;
        }
        if (token.type === TokenType.NULL) {
            this.advance();
            return null;
        }
        if (token.type === TokenType.TRUE) {
            this.advance();
            return true;
        }
        if (token.type === TokenType.FALSE) {
            this.advance();
            return false;
        }
        // Vector literal
        if (this.match(TokenType.LBRACKET)) {
            const vec = [];
            do {
                vec.push(parseFloat(this.expect(TokenType.NUMBER).value));
            } while (this.match(TokenType.COMMA));
            this.expect(TokenType.RBRACKET);
            return vec;
        }

        throw new Error(`Unexpected value token: ${token.type}`);
    }

    parseUpdate() {
        this.expect(TokenType.UPDATE);
        const tableName = this.expect(TokenType.IDENTIFIER).value;
        this.expect(TokenType.SET);

        const updates = {};
        do {
            const col = this.expect(TokenType.IDENTIFIER).value;
            this.expect(TokenType.EQ);
            updates[col] = this.parseValue();
        } while (this.match(TokenType.COMMA));

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        return { type: 'UPDATE', table: tableName, updates, where };
    }

    parseDelete() {
        this.expect(TokenType.DELETE);
        this.expect(TokenType.FROM);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        return { type: 'DELETE', table: tableName, where };
    }

    parseSelect() {
        this.expect(TokenType.SELECT);

        // Check for DISTINCT
        const distinct = !!this.match(TokenType.DISTINCT);

        // Parse column list (may include aggregate functions, table.column, etc.)
        const columns = [];
        columns.push(this.parseSelectColumn());
        while (this.match(TokenType.COMMA)) {
            columns.push(this.parseSelectColumn());
        }

        this.expect(TokenType.FROM);

        // Parse table with optional alias
        const tables = [];
        tables.push(this.parseTableRef());

        // Parse JOINs
        const joins = [];
        while (this.peek().type === TokenType.JOIN ||
               this.peek().type === TokenType.LEFT ||
               this.peek().type === TokenType.RIGHT ||
               this.peek().type === TokenType.INNER) {
            joins.push(this.parseJoin());
        }

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        // Parse GROUP BY
        let groupBy = null;
        if (this.match(TokenType.GROUP)) {
            this.expect(TokenType.BY);
            groupBy = [this.parseColumnRef()];
            while (this.match(TokenType.COMMA)) {
                groupBy.push(this.parseColumnRef());
            }
        }

        // Parse HAVING
        let having = null;
        if (this.match(TokenType.HAVING)) {
            having = this.parseWhereExpr();
        }

        let orderBy = null;
        if (this.match(TokenType.ORDER)) {
            this.expect(TokenType.BY);
            orderBy = [];
            do {
                const column = this.parseColumnRef();
                const desc = !!this.match(TokenType.DESC);
                if (!desc) this.match(TokenType.ASC);
                orderBy.push({ column, desc });
            } while (this.match(TokenType.COMMA));
        }

        let limit = null;
        if (this.match(TokenType.LIMIT)) {
            limit = parseInt(this.expect(TokenType.NUMBER).value);
        }

        let offset = null;
        if (this.match(TokenType.OFFSET)) {
            offset = parseInt(this.expect(TokenType.NUMBER).value);
        }

        // For backwards compatibility, use first table name as 'table'
        const tableName = tables[0].name;

        return {
            type: 'SELECT',
            table: tableName,
            tables,
            columns,
            distinct,
            joins,
            where,
            groupBy,
            having,
            orderBy,
            limit,
            offset
        };
    }

    // Parse a single column in SELECT clause
    parseSelectColumn() {
        // Check for * first
        if (this.match(TokenType.STAR)) {
            return { type: 'star', value: '*' };
        }

        // Check for aggregate functions
        const aggFuncs = [TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX];
        for (const funcType of aggFuncs) {
            if (this.match(funcType)) {
                const funcName = funcType;
                this.expect(TokenType.LPAREN);
                let arg;
                if (this.match(TokenType.STAR)) {
                    arg = '*';
                } else if (this.match(TokenType.DISTINCT)) {
                    const col = this.parseColumnRef();
                    arg = { distinct: true, column: col };
                } else {
                    arg = this.parseColumnRef();
                }
                this.expect(TokenType.RPAREN);

                // Optional alias
                let alias = null;
                if (this.match(TokenType.AS)) {
                    alias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.peek().type === TokenType.IDENTIFIER) {
                    alias = this.advance().value;
                }

                return { type: 'aggregate', func: funcName.toLowerCase(), arg, alias };
            }
        }

        // Regular column (may be table.column)
        const col = this.parseColumnRef();

        // Optional alias
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.peek().type === TokenType.IDENTIFIER &&
                   this.peek().type !== TokenType.FROM &&
                   this.peek().type !== TokenType.COMMA) {
            // Peek next to avoid consuming FROM as alias
            const nextType = this.peek().type;
            if (nextType !== TokenType.FROM && nextType !== TokenType.COMMA &&
                nextType !== TokenType.WHERE && nextType !== TokenType.ORDER &&
                nextType !== TokenType.GROUP && nextType !== TokenType.LIMIT) {
                alias = this.advance().value;
            }
        }

        return { type: 'column', value: col, alias };
    }

    // Parse column reference (may be table.column or just column)
    parseColumnRef() {
        const first = this.expect(TokenType.IDENTIFIER).value;
        if (this.match(TokenType.DOT)) {
            const second = this.expect(TokenType.IDENTIFIER).value;
            return { table: first, column: second };
        }
        return first;
    }

    // Parse table reference with optional alias
    parseTableRef() {
        const name = this.expect(TokenType.IDENTIFIER).value;
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.peek().type === TokenType.IDENTIFIER) {
            // Check if next token looks like an alias (not a keyword)
            const nextType = this.peek().type;
            if (nextType !== TokenType.WHERE && nextType !== TokenType.ORDER &&
                nextType !== TokenType.GROUP && nextType !== TokenType.LIMIT &&
                nextType !== TokenType.JOIN && nextType !== TokenType.LEFT &&
                nextType !== TokenType.RIGHT && nextType !== TokenType.INNER &&
                nextType !== TokenType.ON) {
                alias = this.advance().value;
            }
        }
        return { name, alias };
    }

    // Parse JOIN clause
    parseJoin() {
        let joinType = 'INNER';
        if (this.match(TokenType.LEFT)) {
            joinType = 'LEFT';
            this.match(TokenType.JOIN); // LEFT JOIN
        } else if (this.match(TokenType.RIGHT)) {
            joinType = 'RIGHT';
            this.match(TokenType.JOIN); // RIGHT JOIN
        } else if (this.match(TokenType.INNER)) {
            joinType = 'INNER';
            this.match(TokenType.JOIN); // INNER JOIN
        } else {
            this.expect(TokenType.JOIN); // Just JOIN
        }

        const table = this.parseTableRef();
        this.expect(TokenType.ON);
        const on = this.parseJoinCondition();

        return { type: joinType, table, on };
    }

    // Parse JOIN ON condition (left.col = right.col)
    parseJoinCondition() {
        const left = this.parseColumnRef();
        this.expect(TokenType.EQ);
        const right = this.parseColumnRef();
        return { left, right };
    }

    parseWhereExpr() {
        return this.parseOrExpr();
    }

    parseOrExpr() {
        let left = this.parseAndExpr();
        while (this.match(TokenType.OR)) {
            const right = this.parseAndExpr();
            left = { op: 'OR', left, right };
        }
        return left;
    }

    parseAndExpr() {
        let left = this.parseComparison();
        while (this.match(TokenType.AND)) {
            const right = this.parseComparison();
            left = { op: 'AND', left, right };
        }
        return left;
    }

    parseComparison() {
        // Handle parenthesized expressions
        if (this.match(TokenType.LPAREN)) {
            const expr = this.parseOrExpr();
            this.expect(TokenType.RPAREN);
            return expr;
        }

        // Handle aggregate functions in HAVING clause
        const aggFuncs = [TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX];
        let column;
        let isAggregate = false;
        let aggFunc = null;
        let aggArg = null;

        for (const funcType of aggFuncs) {
            if (this.match(funcType)) {
                isAggregate = true;
                aggFunc = funcType.toLowerCase();
                this.expect(TokenType.LPAREN);
                if (this.match(TokenType.STAR)) {
                    aggArg = '*';
                } else {
                    aggArg = this.parseColumnRef();
                }
                this.expect(TokenType.RPAREN);
                column = { type: 'aggregate', func: aggFunc, arg: aggArg };
                break;
            }
        }

        if (!isAggregate) {
            // Parse column (may be dotted like table.column)
            column = this.parseColumnRef();
        }

        // Handle BETWEEN
        if (this.match(TokenType.BETWEEN)) {
            const low = this.parseValue();
            this.expect(TokenType.AND);
            const high = this.parseValue();
            return { op: 'BETWEEN', column, low, high };
        }

        // Handle IN
        if (this.match(TokenType.IN)) {
            this.expect(TokenType.LPAREN);
            const values = [this.parseValue()];
            while (this.match(TokenType.COMMA)) {
                values.push(this.parseValue());
            }
            this.expect(TokenType.RPAREN);
            return { op: 'IN', column, values };
        }

        // Handle NEAR (vector similarity search)
        // Syntax: column NEAR 'text' [TOPK n]
        if (this.match(TokenType.NEAR)) {
            const text = this.expect(TokenType.STRING).value;
            let topK = null;
            if (this.match(TokenType.TOPK)) {
                topK = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }
            return { op: 'NEAR', column, text, topK };
        }

        let op;
        if (this.match(TokenType.EQ)) op = '=';
        else if (this.match(TokenType.NE)) op = '!=';
        else if (this.match(TokenType.LT)) op = '<';
        else if (this.match(TokenType.LE)) op = '<=';
        else if (this.match(TokenType.GT)) op = '>';
        else if (this.match(TokenType.GE)) op = '>=';
        else if (this.match(TokenType.LIKE)) op = 'LIKE';
        else throw new Error(`Expected comparison operator`);

        const value = this.parseValue();
        return { op, column, value };
    }
}

// SQL Executor helper - get column value from row
function getColumnValue(row, column, tableAliases = {}) {
    if (typeof column === 'string') {
        return row[column];
    }
    if (column && column.table && column.column) {
        // Try table.column directly first (e.g., u.id)
        const fullKey = `${column.table}.${column.column}`;
        if (fullKey in row) return row[fullKey];
        // Try using alias mapping to get actual table name
        const tableName = tableAliases[column.table] || column.table;
        const aliasKey = `${tableName}.${column.column}`;
        if (aliasKey in row) return row[aliasKey];
        // Try just the column name (fallback for non-JOIN queries)
        if (column.column in row) return row[column.column];
    }
    return undefined;
}

function evalWhere(where, row, tableAliases = {}) {
    if (!where) return true;

    switch (where.op) {
        case 'AND':
            return evalWhere(where.left, row, tableAliases) && evalWhere(where.right, row, tableAliases);
        case 'OR':
            return evalWhere(where.left, row, tableAliases) || evalWhere(where.right, row, tableAliases);
        case '=': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val === where.value;
        }
        case '!=':
        case '<>': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val !== where.value;
        }
        case '<': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val < where.value;
        }
        case '<=': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val <= where.value;
        }
        case '>': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val > where.value;
        }
        case '>=': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val >= where.value;
        }
        case 'LIKE': {
            const val = getColumnValue(row, where.column, tableAliases);
            const pattern = where.value.replace(/%/g, '.*').replace(/_/g, '.');
            return new RegExp(`^${pattern}$`, 'i').test(val);
        }
        case 'BETWEEN': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val >= where.low && val <= where.high;
        }
        case 'IN': {
            const val = getColumnValue(row, where.column, tableAliases);
            return where.values.includes(val);
        }
        default:
            return true;
    }
}

// Calculate aggregate function value
function calculateAggregate(func, arg, rows) {
    if (rows.length === 0) return func === 'count' ? 0 : null;

    const colName = arg === '*' ? null : (typeof arg === 'string' ? arg : (arg.column || arg));

    switch (func) {
        case 'count':
            if (arg === '*') return rows.length;
            return rows.filter(r => r[colName] != null).length;
        case 'sum': {
            let sum = 0;
            for (const row of rows) {
                const val = row[colName];
                if (typeof val === 'number') sum += val;
            }
            return sum;
        }
        case 'avg': {
            let sum = 0, count = 0;
            for (const row of rows) {
                const val = row[colName];
                if (typeof val === 'number') {
                    sum += val;
                    count++;
                }
            }
            return count > 0 ? sum / count : null;
        }
        case 'min': {
            let min = Infinity;
            for (const row of rows) {
                const val = row[colName];
                if (val != null && val < min) min = val;
            }
            return min === Infinity ? null : min;
        }
        case 'max': {
            let max = -Infinity;
            for (const row of rows) {
                const val = row[colName];
                if (val != null && val > max) max = val;
            }
            return max === -Infinity ? null : max;
        }
        default:
            return null;
    }
}

// Evaluate HAVING clause (similar to WHERE but works on aggregated values)
function evalHaving(having, row) {
    if (!having) return true;

    switch (having.op) {
        case 'AND':
            return evalHaving(having.left, row) && evalHaving(having.right, row);
        case 'OR':
            return evalHaving(having.left, row) || evalHaving(having.right, row);
        default: {
            // For aggregate comparisons, the column is already in the row
            let val;
            if (having.column && having.column.type === 'aggregate') {
                const aggName = `${having.column.func}(${having.column.arg === '*' ? '*' : (typeof having.column.arg === 'string' ? having.column.arg : having.column.arg.column)})`;
                val = row[aggName];
            } else {
                const colName = typeof having.column === 'string' ? having.column : having.column.column;
                val = row[colName];
            }

            switch (having.op) {
                case '=': return val === having.value;
                case '!=':
                case '<>': return val !== having.value;
                case '<': return val < having.value;
                case '<=': return val <= having.value;
                case '>': return val > having.value;
                case '>=': return val >= having.value;
                default: return true;
            }
        }
    }
}

// Extract NEAR condition from WHERE expression
function extractNearCondition(expr) {
    if (!expr) return null;
    if (expr.op === 'NEAR') {
        return expr;
    }
    if (expr.op === 'AND' || expr.op === 'OR') {
        const leftNear = extractNearCondition(expr.left);
        if (leftNear) return leftNear;
        return extractNearCondition(expr.right);
    }
    return null;
}

// Remove NEAR condition from expression, returning remaining conditions
function removeNearCondition(expr) {
    if (!expr) return null;
    if (expr.op === 'NEAR') return null;
    if (expr.op === 'AND' || expr.op === 'OR') {
        const left = removeNearCondition(expr.left);
        const right = removeNearCondition(expr.right);
        if (!left && !right) return null;
        if (!left) return right;
        if (!right) return left;
        return { op: expr.op, left, right };
    }
    return expr;
}

// Cosine similarity between two vectors
function cosineSimilarity(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0, magA = 0, magB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }
    return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

// Execute NEAR vector search
async function executeNearSearch(rows, nearCondition, limit) {
    const column = typeof nearCondition.column === 'string' ? nearCondition.column : nearCondition.column.column;
    const text = nearCondition.text;
    const topK = nearCondition.topK || limit || 10;

    // Check if gpuTransformer is available and has a loaded model
    if (!gpuTransformer) {
        throw new Error('NEAR requires a text encoder model. Load a model first with store.loadModel()');
    }

    // Generate query embedding
    let queryVec;
    try {
        // Try to get any loaded model
        const models = gpuTransformer.getLoadedModels?.() || [];
        if (models.length === 0) {
            throw new Error('No text encoder model loaded');
        }
        queryVec = await gpuTransformer.encodeText(text, models[0]);
    } catch (e) {
        throw new Error(`NEAR failed to encode query: ${e.message}`);
    }

    // Score each row
    const scored = [];
    for (const row of rows) {
        const colValue = row[column];

        // If column is already a vector (array of numbers), use it directly
        if (Array.isArray(colValue) && typeof colValue[0] === 'number') {
            const score = cosineSimilarity(queryVec, colValue);
            scored.push({ row, score });
        }
        // If column is text, we need to embed it (expensive)
        else if (typeof colValue === 'string') {
            const cacheKey = `sql:${column}:${colValue}`;
            let itemVec = embeddingCache.get(cacheKey);
            if (!itemVec) {
                const models = gpuTransformer.getLoadedModels?.() || [];
                itemVec = await gpuTransformer.encodeText(colValue, models[0]);
                embeddingCache.set(cacheKey, itemVec);
            }
            const score = cosineSimilarity(queryVec, itemVec);
            scored.push({ row, score });
        }
    }

    // Sort by score descending and take top K
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map(s => ({ ...s.row, _score: s.score }));
}

async function executeSQL(db, sql) {
    const lexer = new SQLLexer(sql);
    const tokens = lexer.tokenize();
    const parser = new SQLParser(tokens);
    const ast = parser.parse();

    const type = ast.type.toUpperCase();

    switch (type) {
        case 'CREATE_TABLE':
            return db.createTable(ast.table, ast.columns, ast.ifNotExists);

        case 'DROP_TABLE':
            return db.dropTable(ast.table, ast.ifExists);

        case 'INSERT': {
            // If rows are arrays (no column names in INSERT), convert to objects using table schema
            let rows = ast.rows;
            if (rows.length > 0 && Array.isArray(rows[0])) {
                const tableState = db.tables.get(ast.table);
                if (tableState && tableState.schema) {
                    rows = rows.map(valueArray => {
                        const row = {};
                        tableState.schema.forEach((col, i) => {
                            row[col.name] = valueArray[i];
                        });
                        return row;
                    });
                }
            }
            return db.insert(ast.table, rows);
        }

        case 'DELETE': {
            const predicate = ast.where
                ? (row) => evalWhere(ast.where, row)
                : () => true;
            return db.delete(ast.table, predicate);
        }

        case 'UPDATE': {
            const predicate = ast.where
                ? (row) => evalWhere(ast.where, row)
                : () => true;
            return db.update(ast.table, ast.updates, predicate);
        }

        case 'SELECT': {
            // Build table alias mapping
            const tableAliases = {};
            if (ast.tables) {
                for (const t of ast.tables) {
                    if (t.alias) tableAliases[t.alias] = t.name;
                }
            }
            if (ast.joins) {
                for (const j of ast.joins) {
                    if (j.table.alias) tableAliases[j.table.alias] = j.table.name;
                }
            }

            // Fetch data from main table
            let rows = await db.select(ast.table, {});

            // Process JOINs
            if (ast.joins && ast.joins.length > 0) {
                // Add prefixes for the first table before any JOINs
                const firstTableName = ast.tables[0].alias || ast.tables[0].name;
                rows = rows.map(row => {
                    const prefixed = { ...row };
                    for (const key of Object.keys(row)) {
                        prefixed[`${firstTableName}.${key}`] = row[key];
                    }
                    return prefixed;
                });

                for (const join of ast.joins) {
                    const rightTable = join.table.name;
                    const rightRows = await db.select(rightTable, {});
                    const newRows = [];
                    const matchedRightIndices = new Set();

                    // Get right table info for namespacing
                    const rightTableName = join.table.alias || join.table.name;

                    for (const leftRow of rows) {
                        let matched = false;
                        for (let ri = 0; ri < rightRows.length; ri++) {
                            const rightRow = rightRows[ri];
                            // Evaluate ON condition
                            const leftVal = getColumnValue(leftRow, join.on.left, tableAliases);
                            const rightVal = getColumnValue(rightRow, join.on.right, tableAliases);
                            if (leftVal === rightVal) {
                                matched = true;
                                matchedRightIndices.add(ri);
                                // Merge rows - keep left row as-is, add right row with prefix
                                const merged = { ...leftRow };
                                // Add right row columns with table prefix
                                for (const key of Object.keys(rightRow)) {
                                    // Don't overwrite with plain key if it already exists
                                    if (!(key in merged)) merged[key] = rightRow[key];
                                    merged[`${rightTableName}.${key}`] = rightRow[key];
                                }
                                newRows.push(merged);
                            }
                        }
                        // LEFT JOIN: include left row even if no match
                        if (!matched && join.type === 'LEFT') {
                            newRows.push({ ...leftRow });
                        }
                    }

                    // RIGHT JOIN: include unmatched right rows
                    if (join.type === 'RIGHT') {
                        for (let ri = 0; ri < rightRows.length; ri++) {
                            if (!matchedRightIndices.has(ri)) {
                                const merged = {};
                                for (const key of Object.keys(rightRows[ri])) {
                                    merged[key] = rightRows[ri][key];
                                    merged[`${rightTableName}.${key}`] = rightRows[ri][key];
                                }
                                newRows.push(merged);
                            }
                        }
                    }
                    rows = newRows;
                }
            }

            // Check for NEAR condition in WHERE clause
            const nearCondition = extractNearCondition(ast.where);
            if (nearCondition) {
                // Execute NEAR search first
                rows = await executeNearSearch(rows, nearCondition, ast.limit);
                // Apply remaining WHERE conditions
                const remainingWhere = removeNearCondition(ast.where);
                if (remainingWhere) {
                    rows = rows.filter(row => evalWhere(remainingWhere, row, tableAliases));
                }
            } else if (ast.where) {
                // Apply regular WHERE clause
                rows = rows.filter(row => evalWhere(ast.where, row, tableAliases));
            }

            // Apply GROUP BY with aggregations
            if (ast.groupBy && ast.groupBy.length > 0) {
                const groups = new Map();
                for (const row of rows) {
                    const keyParts = ast.groupBy.map(col => {
                        const colName = typeof col === 'string' ? col : col.column;
                        return String(row[colName]);
                    });
                    const key = keyParts.join('|');
                    if (!groups.has(key)) {
                        groups.set(key, []);
                    }
                    groups.get(key).push(row);
                }

                // Process each group
                const groupedRows = [];
                for (const [, groupRows] of groups) {
                    const resultRow = {};
                    // Add GROUP BY columns
                    for (const col of ast.groupBy) {
                        const colName = typeof col === 'string' ? col : col.column;
                        resultRow[colName] = groupRows[0][colName];
                    }
                    // Calculate aggregates
                    for (const col of ast.columns) {
                        if (col.type === 'aggregate') {
                            const rawName = `${col.func}(${col.arg === '*' ? '*' : (typeof col.arg === 'string' ? col.arg : col.arg.column)})`;
                            const aggValue = calculateAggregate(col.func, col.arg, groupRows);
                            // Store with raw name for HAVING clause lookups
                            resultRow[rawName] = aggValue;
                            // Also store with alias if different
                            if (col.alias && col.alias !== rawName) {
                                resultRow[col.alias] = aggValue;
                            }
                        }
                    }
                    groupedRows.push(resultRow);
                }
                rows = groupedRows;

                // Apply HAVING
                if (ast.having) {
                    rows = rows.filter(row => evalHaving(ast.having, row));
                }
            }
            // Handle aggregates without GROUP BY (whole table aggregate)
            else if (ast.columns.some(c => c.type === 'aggregate')) {
                const resultRow = {};
                for (const col of ast.columns) {
                    if (col.type === 'aggregate') {
                        const aggName = col.alias || `${col.func}(${col.arg === '*' ? '*' : (typeof col.arg === 'string' ? col.arg : col.arg.column)})`;
                        resultRow[aggName] = calculateAggregate(col.func, col.arg, rows);
                    } else if (col.type === 'column') {
                        const colName = typeof col.value === 'string' ? col.value : col.value.column;
                        if (rows.length > 0) resultRow[colName] = rows[0][colName];
                    }
                }
                rows = [resultRow];
            }

            // Apply ORDER BY first (before projection, using original column values)
            if (ast.orderBy && ast.orderBy.length > 0) {
                rows.sort((a, b) => {
                    for (const order of ast.orderBy) {
                        const col = typeof order.column === 'string' ? order.column : order.column.column;
                        const aVal = a[col];
                        const bVal = b[col];
                        if (aVal < bVal) return order.desc ? 1 : -1;
                        if (aVal > bVal) return order.desc ? -1 : 1;
                    }
                    return 0;
                });
            }

            // Apply OFFSET
            if (ast.offset) {
                rows = rows.slice(ast.offset);
            }

            // Apply LIMIT
            if (ast.limit) {
                rows = rows.slice(0, ast.limit);
            }

            // Project columns (after ORDER BY, OFFSET, LIMIT)
            let columnNames = [];
            if (!ast.columns.some(c => c.type === 'star') && !ast.groupBy) {
                rows = rows.map(row => {
                    const result = {};
                    for (const col of ast.columns) {
                        if (col.type === 'column') {
                            const colName = typeof col.value === 'string' ? col.value : col.value.column;
                            const alias = col.alias || colName;
                            // Use getColumnValue to properly resolve table-prefixed columns
                            result[alias] = getColumnValue(row, col.value, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'aggregate') {
                            const aggName = col.alias || `${col.func}(${col.arg === '*' ? '*' : (typeof col.arg === 'string' ? col.arg : col.arg.column)})`;
                            result[aggName] = row[aggName];
                            if (!columnNames.includes(aggName)) columnNames.push(aggName);
                        }
                    }
                    return result;
                });
            } else if (rows.length > 0) {
                columnNames = Object.keys(rows[0]);
            }

            // Apply DISTINCT (after projection)
            if (ast.distinct) {
                const seen = new Set();
                rows = rows.filter(row => {
                    const key = JSON.stringify(row);
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                });
            }

            return { rows, columns: columnNames };
        }

        default:
            throw new Error(`Unknown statement type: ${ast.type}`);
    }
}

// ============================================================================
// WorkerVault - Unified vault storage (KV + tables)
// ============================================================================

class WorkerVault {
    constructor() {
        this._root = null;
        this._ready = false;
        this._kv = {};  // KV data loaded from JSON
        this._encryptionKeyId = null;
        this._db = null;  // Embedded database for SQL tables
    }

    async open(encryptionConfig = null) {
        if (this._ready) return this;

        // Set up encryption if provided
        if (encryptionConfig) {
            const { keyId, keyBytes } = encryptionConfig;
            if (!encryptionKeys.has(keyId)) {
                const cryptoKey = await importEncryptionKey(keyBytes);
                encryptionKeys.set(keyId, cryptoKey);
            }
            this._encryptionKeyId = keyId;
        }

        try {
            const opfsRoot = await navigator.storage.getDirectory();
            this._root = await opfsRoot.getDirectoryHandle('vault', { create: true });
            await this._loadKV();

            // Initialize embedded database for SQL tables
            this._db = new WorkerDatabase('vault');
            await this._db.open();

            this._ready = true;
        } catch (e) {
            console.error('[WorkerVault] Failed to open OPFS:', e);
            throw e;
        }

        return this;
    }

    _getCryptoKey() {
        return this._encryptionKeyId ? encryptionKeys.get(this._encryptionKeyId) : null;
    }

    async _loadKV() {
        try {
            const cryptoKey = this._getCryptoKey();
            const filename = cryptoKey ? '_vault.json.enc' : '_vault.json';
            const fileHandle = await this._root.getFileHandle(filename);
            const file = await fileHandle.getFile();

            if (cryptoKey) {
                const buffer = await file.arrayBuffer();
                this._kv = await decryptData(new Uint8Array(buffer), cryptoKey);
            } else {
                const text = await file.text();
                this._kv = JSON.parse(text);
            }
        } catch (e) {
            if (e.name === 'NotFoundError') {
                this._kv = {};
            } else {
                throw e;
            }
        }
    }

    async _saveKV() {
        const cryptoKey = this._getCryptoKey();
        const filename = cryptoKey ? '_vault.json.enc' : '_vault.json';
        const fileHandle = await this._root.getFileHandle(filename, { create: true });
        const writable = await fileHandle.createWritable();

        if (cryptoKey) {
            const encrypted = await encryptData(this._kv, cryptoKey);
            await writable.write(encrypted);
        } else {
            await writable.write(JSON.stringify(this._kv));
        }

        await writable.close();
    }

    async get(key) {
        return this._kv[key];
    }

    async set(key, value) {
        this._kv[key] = value;
        await this._saveKV();
    }

    async delete(key) {
        delete this._kv[key];
        await this._saveKV();
    }

    async keys() {
        return Object.keys(this._kv);
    }

    async has(key) {
        return key in this._kv;
    }

    async exec(sql) {
        // Delegate SQL execution to embedded database
        return executeSQL(this._db, sql);
    }
}

// Vault singleton
let vaultInstance = null;

async function getVault(encryptionConfig = null) {
    if (!vaultInstance) {
        vaultInstance = new WorkerVault();
    }
    // Re-open with encryption if not already done
    await vaultInstance.open(encryptionConfig);
    return vaultInstance;
}

// ============================================================================
// Get or create instances
// ============================================================================

async function getStore(name, options = {}, encryptionConfig = null) {
    // Include encryption key ID in cache key (but not the actual key bytes)
    const encKeyId = encryptionConfig?.keyId || 'none';
    const key = `${name}:${encKeyId}:${JSON.stringify(options)}`;
    if (!stores.has(key)) {
        const store = new WorkerStore(name, options);
        await store.open(encryptionConfig);
        stores.set(key, store);
    }
    return stores.get(key);
}

async function getDatabase(name) {
    if (!databases.has(name)) {
        const db = new WorkerDatabase(name);
        await db.open();
        databases.set(name, db);
    }
    return databases.get(name);
}

// ============================================================================
// Message Handler
// ============================================================================

/**
 * Send response, using SharedArrayBuffer for large data if available.
 */
function sendResponse(port, id, result) {
    // Try SharedArrayBuffer for large responses
    if (sharedBuffer && result !== undefined) {
        const json = JSON.stringify(result);
        if (json.length > SHARED_THRESHOLD) {
            const bytes = E.encode(json);
            if (sharedOffset + bytes.length <= sharedBuffer.byteLength) {
                // Write to shared buffer
                const view = new Uint8Array(sharedBuffer, sharedOffset, bytes.length);
                view.set(bytes);

                port.postMessage({
                    id,
                    sharedOffset,
                    sharedLength: bytes.length
                });

                sharedOffset += bytes.length;
                // Reset offset if we're past halfway (simple ring buffer)
                if (sharedOffset > sharedBuffer.byteLength / 2) {
                    sharedOffset = 0;
                }
                return;
            }
        }
    }

    // Fall back to regular postMessage
    port.postMessage({ id, result });
}

async function handleMessage(port, data) {
    // Handle SharedArrayBuffer initialization
    if (data.type === 'initSharedBuffer') {
        sharedBuffer = data.buffer;
        sharedOffset = 0;
        console.log('[LanceQLWorker] SharedArrayBuffer initialized:', sharedBuffer.byteLength, 'bytes');
        return;
    }

    const { id, method, args } = data;

    try {
        let result;

        // Store operations
        if (method === 'ping') {
            result = 'pong';
        } else if (method === 'open') {
            await getStore(args.name, args.options, args.encryption);
            result = true;
        } else if (method === 'get') {
            result = await (await getStore(args.name)).get(args.key);
        } else if (method === 'set') {
            await (await getStore(args.name)).set(args.key, args.value);
            result = true;
        } else if (method === 'delete') {
            await (await getStore(args.name)).delete(args.key);
            result = true;
        } else if (method === 'keys') {
            result = await (await getStore(args.name)).keys();
        } else if (method === 'clear') {
            await (await getStore(args.name)).clear();
            result = true;
        } else if (method === 'filter') {
            result = await (await getStore(args.name)).filter(args.key, args.query);
        } else if (method === 'find') {
            result = await (await getStore(args.name)).find(args.key, args.query);
        } else if (method === 'search') {
            result = await (await getStore(args.name)).search(args.key, args.text, args.limit);
        } else if (method === 'count') {
            result = await (await getStore(args.name)).count(args.key, args.query);
        } else if (method === 'enableSemanticSearch') {
            result = await (await getStore(args.name)).enableSemanticSearch(args.options);
        } else if (method === 'disableSemanticSearch') {
            (await getStore(args.name)).disableSemanticSearch();
            result = true;
        } else if (method === 'hasSemanticSearch') {
            result = (await getStore(args.name)).hasSemanticSearch();
        }
        // Database operations
        else if (method === 'db:open') {
            await getDatabase(args.name);
            result = true;
        } else if (method === 'db:createTable') {
            result = await (await getDatabase(args.db)).createTable(args.tableName, args.columns, args.ifNotExists);
        } else if (method === 'db:dropTable') {
            result = await (await getDatabase(args.db)).dropTable(args.tableName, args.ifExists);
        } else if (method === 'db:insert') {
            result = await (await getDatabase(args.db)).insert(args.tableName, args.rows);
        } else if (method === 'db:delete') {
            // Note: predicate function is serialized - need to recreate
            const db = await getDatabase(args.db);
            const predicate = args.where
                ? (row) => evalWhere(args.where, row)
                : () => true;
            result = await db.delete(args.tableName, predicate);
        } else if (method === 'db:update') {
            const db = await getDatabase(args.db);
            const predicate = args.where
                ? (row) => evalWhere(args.where, row)
                : () => true;
            result = await db.update(args.tableName, args.updates, predicate);
        } else if (method === 'db:select') {
            const db = await getDatabase(args.db);
            const options = { ...args.options };
            if (args.where) {
                options.where = (row) => evalWhere(args.where, row);
            }
            result = await db.select(args.tableName, options);
        } else if (method === 'db:exec') {
            result = await executeSQL(await getDatabase(args.db), args.sql);
        } else if (method === 'db:flush') {
            await (await getDatabase(args.db)).flush();
            result = true;
        } else if (method === 'db:compact') {
            result = await (await getDatabase(args.db)).compact();
        } else if (method === 'db:listTables') {
            result = (await getDatabase(args.db)).listTables();
        } else if (method === 'db:getTable') {
            result = (await getDatabase(args.db)).getTable(args.tableName);
        } else if (method === 'db:scanStart') {
            result = await (await getDatabase(args.db)).scanStart(args.tableName, args.options);
        } else if (method === 'db:scanNext') {
            const db = await getDatabase(args.db);
            result = db.scanNext(args.streamId);
        }
        // Vault operations
        else if (method === 'vault:open') {
            await getVault(args.encryption);
            result = true;
        } else if (method === 'vault:get') {
            result = await (await getVault()).get(args.key);
        } else if (method === 'vault:set') {
            await (await getVault()).set(args.key, args.value);
            result = true;
        } else if (method === 'vault:delete') {
            await (await getVault()).delete(args.key);
            result = true;
        } else if (method === 'vault:keys') {
            result = await (await getVault()).keys();
        } else if (method === 'vault:has') {
            result = await (await getVault()).has(args.key);
        } else if (method === 'vault:exec') {
            result = await (await getVault()).exec(args.sql);
        }
        // Unknown method
        else {
            throw new Error(`Unknown method: ${method}`);
        }

        sendResponse(port, id, result);
    } catch (error) {
        port.postMessage({ id, error: error.message });
    }
}

// ============================================================================
// Worker Entry Points
// ============================================================================

// SharedWorker connection handler
self.onconnect = (event) => {
    const port = event.ports[0];
    ports.add(port);

    port.onmessage = (e) => {
        handleMessage(port, e.data);
    };

    port.onmessageerror = (e) => {
        console.error('[LanceQLWorker] Message error:', e);
    };

    // Worker is ready
    port.postMessage({ type: 'ready' });
    port.start();

    console.log('[LanceQLWorker] New connection, total ports:', ports.size);
};

// Regular Worker fallback (when SharedWorker not available)
self.onmessage = (e) => {
    handleMessage(self, e.data);
};

console.log('[LanceQLWorker] Initialized');
