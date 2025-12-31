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

    // Literals
    IDENTIFIER: 'IDENTIFIER', STRING: 'STRING', NUMBER: 'NUMBER',

    // Operators
    EQ: '=', NE: '!=', LT: '<', LE: '<=', GT: '>', GE: '>=',
    STAR: '*', COMMA: ',', LPAREN: '(', RPAREN: ')',
    LBRACKET: '[', RBRACKET: ']',

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

        const columns = [];
        if (this.match(TokenType.STAR)) {
            columns.push('*');
        } else {
            columns.push(this.expect(TokenType.IDENTIFIER).value);
            while (this.match(TokenType.COMMA)) {
                columns.push(this.expect(TokenType.IDENTIFIER).value);
            }
        }

        this.expect(TokenType.FROM);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        let orderBy = null;
        if (this.match(TokenType.ORDER)) {
            this.expect(TokenType.BY);
            const column = this.expect(TokenType.IDENTIFIER).value;
            const desc = !!this.match(TokenType.DESC);
            if (!desc) this.match(TokenType.ASC);
            orderBy = { column, desc };
        }

        let limit = null;
        if (this.match(TokenType.LIMIT)) {
            limit = parseInt(this.expect(TokenType.NUMBER).value);
        }

        let offset = null;
        if (this.match(TokenType.OFFSET)) {
            offset = parseInt(this.expect(TokenType.NUMBER).value);
        }

        return { type: 'SELECT', table: tableName, columns, where, orderBy, limit, offset };
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
        const column = this.expect(TokenType.IDENTIFIER).value;

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

// SQL Executor helper
function evalWhere(where, row) {
    if (!where) return true;

    switch (where.op) {
        case 'AND':
            return evalWhere(where.left, row) && evalWhere(where.right, row);
        case 'OR':
            return evalWhere(where.left, row) || evalWhere(where.right, row);
        case '=':
            return row[where.column] === where.value;
        case '!=':
        case '<>':
            return row[where.column] !== where.value;
        case '<':
            return row[where.column] < where.value;
        case '<=':
            return row[where.column] <= where.value;
        case '>':
            return row[where.column] > where.value;
        case '>=':
            return row[where.column] >= where.value;
        case 'LIKE':
            const pattern = where.value.replace(/%/g, '.*').replace(/_/g, '.');
            return new RegExp(`^${pattern}$`, 'i').test(row[where.column]);
        default:
            return true;
    }
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
            return db.insert(ast.table, ast.rows);
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
            const options = {
                columns: ast.columns,
                where: ast.where ? (row) => evalWhere(ast.where, row) : null,
                limit: ast.limit,
                offset: ast.offset,
                orderBy: ast.orderBy,
            };
            return db.select(ast.table, options);
        }

        default:
            throw new Error(`Unknown statement type: ${ast.type}`);
    }
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
