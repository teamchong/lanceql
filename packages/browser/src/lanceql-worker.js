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
    FULL: 'FULL', OUTER: 'OUTER', CROSS: 'CROSS',
    // GROUP BY / HAVING / QUALIFY / ROLLUP / CUBE / GROUPING SETS
    GROUP: 'GROUP', HAVING: 'HAVING', QUALIFY: 'QUALIFY',
    ROLLUP: 'ROLLUP', CUBE: 'CUBE', GROUPING: 'GROUPING', SETS: 'SETS',
    // Aggregate functions
    COUNT: 'COUNT', SUM: 'SUM', AVG: 'AVG', MIN: 'MIN', MAX: 'MAX',
    // Additional operators
    DISTINCT: 'DISTINCT', BETWEEN: 'BETWEEN', IN: 'IN',
    // Vector search
    NEAR: 'NEAR', TOPK: 'TOPK',
    // CASE expression
    CASE: 'CASE', WHEN: 'WHEN', THEN: 'THEN', ELSE: 'ELSE', END: 'END',
    // Type casting and IS NULL
    CAST: 'CAST', IS: 'IS',
    // Set operations
    UNION: 'UNION', INTERSECT: 'INTERSECT', EXCEPT: 'EXCEPT', ALL: 'ALL',
    // CTEs
    WITH: 'WITH',
    // Window functions
    OVER: 'OVER', PARTITION: 'PARTITION', ROW_NUMBER: 'ROW_NUMBER', RANK: 'RANK',
    DENSE_RANK: 'DENSE_RANK', LAG: 'LAG', LEAD: 'LEAD',
    NTILE: 'NTILE', PERCENT_RANK: 'PERCENT_RANK', CUME_DIST: 'CUME_DIST',
    FIRST_VALUE: 'FIRST_VALUE', LAST_VALUE: 'LAST_VALUE', NTH_VALUE: 'NTH_VALUE',
    // Window frame specifications
    ROWS: 'ROWS', RANGE: 'RANGE', UNBOUNDED: 'UNBOUNDED',
    PRECEDING: 'PRECEDING', FOLLOWING: 'FOLLOWING', CURRENT: 'CURRENT', ROW: 'ROW',
    // NULLS FIRST/LAST for ORDER BY
    NULLS: 'NULLS', FIRST: 'FIRST', LAST: 'LAST',
    // Date/Time keywords (used in EXTRACT)
    YEAR: 'YEAR', MONTH: 'MONTH', DAY: 'DAY',
    HOUR: 'HOUR', MINUTE: 'MINUTE', SECOND: 'SECOND',
    // Array keyword
    ARRAY: 'ARRAY',
    // DML enhancement keywords
    CONFLICT: 'CONFLICT', DO: 'DO', NOTHING: 'NOTHING', EXCLUDED: 'EXCLUDED', USING: 'USING',
    // EXPLAIN
    EXPLAIN: 'EXPLAIN', ANALYZE: 'ANALYZE',
    // PIVOT/UNPIVOT
    PIVOT: 'PIVOT', UNPIVOT: 'UNPIVOT', FOR: 'FOR',

    // Literals
    IDENTIFIER: 'IDENTIFIER', STRING: 'STRING', NUMBER: 'NUMBER',

    // Operators
    EQ: '=', NE: '!=', LT: '<', LE: '<=', GT: '>', GE: '>=',
    STAR: '*', COMMA: ',', LPAREN: '(', RPAREN: ')',
    LBRACKET: '[', RBRACKET: ']', DOT: '.',
    // Arithmetic operators
    PLUS: '+', MINUS: '-', SLASH: '/',
    // Bitwise operators
    AMPERSAND: '&', PIPE: '|', CARET: '^', TILDE: '~', LSHIFT: '<<', RSHIFT: '>>',

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
            '+': TokenType.PLUS,
            '/': TokenType.SLASH,
            '&': TokenType.AMPERSAND,
            '|': TokenType.PIPE,
            '^': TokenType.CARET,
            '~': TokenType.TILDE,
        };

        if (singleChars[ch]) {
            this.pos++;

            // Check for multi-char operators
            if (ch === '<' && this.sql[this.pos] === '=') {
                this.pos++;
                return { type: TokenType.LE };
            }
            if (ch === '<' && this.sql[this.pos] === '<') {
                this.pos++;
                return { type: TokenType.LSHIFT };
            }
            if (ch === '>' && this.sql[this.pos] === '=') {
                this.pos++;
                return { type: TokenType.GE };
            }
            if (ch === '>' && this.sql[this.pos] === '>') {
                this.pos++;
                return { type: TokenType.RSHIFT };
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

        // Minus sign: could be negative number or subtraction operator
        if (ch === '-') {
            // Check if it's a negative number (minus followed by digit)
            // Only treat as negative number at start or after operator/open paren
            const prevChar = this.pos > 0 ? this.sql[this.pos - 1] : ' ';
            const isAfterOperand = /[a-zA-Z0-9_)\]]/.test(prevChar.trim() || ' ');
            if (!isAfterOperand && /\d/.test(this.sql[this.pos + 1])) {
                return this._readNumber();
            }
            // Otherwise treat as minus operator
            this.pos++;
            return { type: TokenType.MINUS };
        }

        // Number
        if (/\d/.test(ch)) {
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

    check(type) {
        return this.peek().type === type;
    }

    isKeyword(keyword) {
        const token = this.peek();
        const upper = keyword.toUpperCase();
        return token.type === TokenType[upper] ||
               (token.type === TokenType.IDENTIFIER && token.value.toUpperCase() === upper);
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
            case TokenType.EXPLAIN:
                return this.parseExplain();
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
            case TokenType.WITH:
                return this.parseWithClause();
            default:
                throw new Error(`Unexpected token: ${token.type}`);
        }
    }

    parseExplain() {
        this.expect(TokenType.EXPLAIN);
        const analyze = !!this.match(TokenType.ANALYZE);

        // Parse the statement to explain (SELECT, UPDATE, DELETE, INSERT)
        const stmtToken = this.peek();
        let statement;
        switch (stmtToken.type) {
            case TokenType.SELECT:
                statement = this.parseSelect();
                break;
            case TokenType.WITH:
                statement = this.parseWithClause();
                break;
            case TokenType.UPDATE:
                statement = this.parseUpdate();
                break;
            case TokenType.DELETE:
                statement = this.parseDelete();
                break;
            case TokenType.INSERT:
                statement = this.parseInsert();
                break;
            default:
                throw new Error(`EXPLAIN not supported for: ${stmtToken.type}`);
        }

        return { type: 'EXPLAIN', analyze, statement };
    }

    parseWithClause() {
        this.expect(TokenType.WITH);

        // Parse CTEs
        const ctes = [];
        do {
            const cteName = this.expect(TokenType.IDENTIFIER).value;
            this.expect(TokenType.AS);
            this.expect(TokenType.LPAREN);
            const cteQuery = this.parseSelect();
            this.expect(TokenType.RPAREN);
            ctes.push({ name: cteName, query: cteQuery });
        } while (this.match(TokenType.COMMA));

        // Parse main SELECT
        const mainQuery = this.parseSelect();

        return {
            type: 'WITH',
            ctes,
            query: mainQuery
        };
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

        // Check for INSERT...SELECT vs INSERT...VALUES
        if (this.check(TokenType.SELECT)) {
            const selectQuery = this.parseSelect();
            return { type: 'INSERT', table: tableName, columns, select: selectQuery };
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

        // Check for ON CONFLICT clause (UPSERT)
        let onConflict = null;
        if (this.match(TokenType.ON)) {
            this.expect(TokenType.CONFLICT);

            // Optional conflict target: (column1, column2)
            let conflictColumns = null;
            if (this.match(TokenType.LPAREN)) {
                conflictColumns = [this.expect(TokenType.IDENTIFIER).value];
                while (this.match(TokenType.COMMA)) {
                    conflictColumns.push(this.expect(TokenType.IDENTIFIER).value);
                }
                this.expect(TokenType.RPAREN);
            }

            this.expect(TokenType.DO);

            if (this.match(TokenType.NOTHING)) {
                onConflict = { action: 'nothing', columns: conflictColumns };
            } else if (this.match(TokenType.UPDATE)) {
                this.expect(TokenType.SET);
                const updates = {};
                do {
                    const col = this.expect(TokenType.IDENTIFIER).value;
                    this.expect(TokenType.EQ);
                    updates[col] = this.parseArithmeticExpr();
                } while (this.match(TokenType.COMMA));
                onConflict = { action: 'update', columns: conflictColumns, updates };
            }
        }

        return { type: 'INSERT', table: tableName, columns, rows, onConflict };
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

        // Optional alias
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.check(TokenType.IDENTIFIER) && !this.isKeyword('SET')) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        }

        this.expect(TokenType.SET);

        const updates = {};
        do {
            const col = this.expect(TokenType.IDENTIFIER).value;
            this.expect(TokenType.EQ);
            updates[col] = this.parseArithmeticExpr();
        } while (this.match(TokenType.COMMA));

        // Optional FROM clause for JOINs
        let from = null;
        if (this.match(TokenType.FROM)) {
            from = [];
            do {
                const tbl = this.expect(TokenType.IDENTIFIER).value;
                let tblAlias = null;
                if (this.match(TokenType.AS)) {
                    tblAlias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.check(TokenType.IDENTIFIER) && !this.isKeyword('WHERE')) {
                    tblAlias = this.expect(TokenType.IDENTIFIER).value;
                }
                from.push({ name: tbl, alias: tblAlias });
            } while (this.match(TokenType.COMMA));
        }

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        return { type: 'UPDATE', table: tableName, alias, updates, from, where };
    }

    parseDelete() {
        this.expect(TokenType.DELETE);
        this.expect(TokenType.FROM);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        // Optional alias
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.check(TokenType.IDENTIFIER) && !this.isKeyword('USING') && !this.isKeyword('WHERE')) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        }

        // Optional USING clause for JOINs
        let using = null;
        if (this.match(TokenType.USING)) {
            using = [];
            do {
                const tbl = this.expect(TokenType.IDENTIFIER).value;
                let tblAlias = null;
                if (this.match(TokenType.AS)) {
                    tblAlias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.check(TokenType.IDENTIFIER) && !this.isKeyword('WHERE')) {
                    tblAlias = this.expect(TokenType.IDENTIFIER).value;
                }
                using.push({ name: tbl, alias: tblAlias });
            } while (this.match(TokenType.COMMA));
        }

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        return { type: 'DELETE', table: tableName, alias, using, where };
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

        // FROM clause is optional (for SELECT without tables like SELECT 1+1, SELECT JSON_OBJECT(...))
        const tables = [];
        const joins = [];
        let table = null;

        if (this.match(TokenType.FROM)) {
            // Parse table with optional alias
            tables.push(this.parseTableRef());

            // Parse JOINs
            while (this.peek().type === TokenType.JOIN ||
                   this.peek().type === TokenType.LEFT ||
                   this.peek().type === TokenType.RIGHT ||
                   this.peek().type === TokenType.INNER ||
                   this.peek().type === TokenType.FULL ||
                   this.peek().type === TokenType.CROSS) {
                joins.push(this.parseJoin());
            }
            table = tables[0].name;
        }

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        // Parse GROUP BY (with ROLLUP/CUBE/GROUPING SETS support)
        let groupBy = null;
        if (this.match(TokenType.GROUP)) {
            this.expect(TokenType.BY);

            // Check for ROLLUP
            if (this.match(TokenType.ROLLUP)) {
                this.expect(TokenType.LPAREN);
                const columns = [this.parseColumnRef()];
                while (this.match(TokenType.COMMA)) {
                    columns.push(this.parseColumnRef());
                }
                this.expect(TokenType.RPAREN);
                groupBy = { type: 'ROLLUP', columns };
            }
            // Check for CUBE
            else if (this.match(TokenType.CUBE)) {
                this.expect(TokenType.LPAREN);
                const columns = [this.parseColumnRef()];
                while (this.match(TokenType.COMMA)) {
                    columns.push(this.parseColumnRef());
                }
                this.expect(TokenType.RPAREN);
                groupBy = { type: 'CUBE', columns };
            }
            // Check for GROUPING SETS
            else if (this.match(TokenType.GROUPING)) {
                this.expect(TokenType.SETS);
                this.expect(TokenType.LPAREN);
                const sets = [];
                do {
                    this.expect(TokenType.LPAREN);
                    const setCols = [];
                    if (!this.check(TokenType.RPAREN)) {
                        setCols.push(this.parseColumnRef());
                        while (this.match(TokenType.COMMA)) {
                            setCols.push(this.parseColumnRef());
                        }
                    }
                    this.expect(TokenType.RPAREN);
                    sets.push(setCols);
                } while (this.match(TokenType.COMMA));
                this.expect(TokenType.RPAREN);
                groupBy = { type: 'GROUPING_SETS', sets };
            }
            // Standard GROUP BY
            else {
                groupBy = [this.parseColumnRef()];
                while (this.match(TokenType.COMMA)) {
                    groupBy.push(this.parseColumnRef());
                }
            }
        }

        // Parse HAVING
        let having = null;
        if (this.match(TokenType.HAVING)) {
            having = this.parseWhereExpr();
        }

        // Parse QUALIFY (filter on window function results)
        let qualify = null;
        if (this.match(TokenType.QUALIFY)) {
            qualify = this.parseWhereExpr();
        }

        let orderBy = null;
        if (this.match(TokenType.ORDER)) {
            this.expect(TokenType.BY);
            orderBy = [];
            do {
                const column = this.parseColumnRef();
                const desc = !!this.match(TokenType.DESC);
                if (!desc) this.match(TokenType.ASC);
                // Parse NULLS FIRST/LAST
                let nullsFirst = null;
                if (this.match(TokenType.NULLS)) {
                    if (this.match(TokenType.FIRST)) {
                        nullsFirst = true;
                    } else if (this.match(TokenType.LAST)) {
                        nullsFirst = false;
                    }
                }
                orderBy.push({ column, desc, nullsFirst });
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

        // For backwards compatibility, use first table name as 'table' (null if no FROM)
        const tableName = tables.length > 0 ? tables[0].name : null;

        const selectAst = {
            type: 'SELECT',
            table: tableName,
            tables,
            columns,
            distinct,
            joins,
            where,
            groupBy,
            having,
            qualify,
            orderBy,
            limit,
            offset
        };

        // Check for set operations (UNION, INTERSECT, EXCEPT)
        if (this.match(TokenType.UNION)) {
            const all = this.match(TokenType.ALL);
            const right = this.parseSelect();
            return { type: 'UNION', all: !!all, left: selectAst, right };
        }
        if (this.match(TokenType.INTERSECT)) {
            const all = this.match(TokenType.ALL);
            const right = this.parseSelect();
            return { type: 'INTERSECT', all: !!all, left: selectAst, right };
        }
        if (this.match(TokenType.EXCEPT)) {
            const all = this.match(TokenType.ALL);
            const right = this.parseSelect();
            return { type: 'EXCEPT', all: !!all, left: selectAst, right };
        }

        // Check for PIVOT/UNPIVOT transformations
        if (this.match(TokenType.PIVOT)) {
            return this.parsePivot(selectAst);
        }
        if (this.match(TokenType.UNPIVOT)) {
            return this.parseUnpivot(selectAst);
        }

        return selectAst;
    }

    parsePivot(selectAst) {
        this.expect(TokenType.LPAREN);
        const aggToken = this.advance();
        const aggFunc = (aggToken.value || aggToken.type).toUpperCase();
        this.expect(TokenType.LPAREN);
        const valueColumn = this.parseColumnRef();
        this.expect(TokenType.RPAREN);
        this.expect(TokenType.FOR);
        const pivotColumn = this.parseColumnRef();
        this.expect(TokenType.IN);
        this.expect(TokenType.LPAREN);
        const pivotValues = [];
        do {
            pivotValues.push(this.expect(TokenType.STRING).value);
        } while (this.match(TokenType.COMMA));
        this.expect(TokenType.RPAREN);
        this.expect(TokenType.RPAREN);

        return {
            type: 'PIVOT',
            select: selectAst,
            aggFunc,
            valueColumn: typeof valueColumn === 'string' ? valueColumn : valueColumn.column,
            pivotColumn: typeof pivotColumn === 'string' ? pivotColumn : pivotColumn.column,
            pivotValues
        };
    }

    parseUnpivot(selectAst) {
        // Helper to get identifier (allows keywords to be used as column names)
        const getIdentifier = () => {
            const token = this.advance();
            // Accept IDENTIFIER or any keyword that has a value (keywords store their value)
            return token.value || token.type;
        };

        this.expect(TokenType.LPAREN);
        const valueColumn = getIdentifier();
        this.expect(TokenType.FOR);
        const nameColumn = getIdentifier();
        this.expect(TokenType.IN);
        this.expect(TokenType.LPAREN);
        const unpivotColumns = [];
        do {
            unpivotColumns.push(getIdentifier());
        } while (this.match(TokenType.COMMA));
        this.expect(TokenType.RPAREN);
        this.expect(TokenType.RPAREN);

        return {
            type: 'UNPIVOT',
            select: selectAst,
            valueColumn,
            nameColumn,
            unpivotColumns
        };
    }

    // Parse a single column in SELECT clause
    parseSelectColumn() {
        // Check for * first
        if (this.match(TokenType.STAR)) {
            return { type: 'star', value: '*' };
        }

        // Check for scalar subquery: (SELECT ...)
        if (this.check(TokenType.LPAREN)) {
            const savedPos = this.pos;
            this.advance(); // consume (
            if (this.check(TokenType.SELECT)) {
                const subquery = this.parseSelect();
                this.expect(TokenType.RPAREN);
                let alias = null;
                if (this.match(TokenType.AS)) {
                    alias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.peek().type === TokenType.IDENTIFIER) {
                    const nextType = this.peek().type;
                    if (nextType !== TokenType.FROM && nextType !== TokenType.COMMA &&
                        nextType !== TokenType.WHERE && nextType !== TokenType.ORDER &&
                        nextType !== TokenType.GROUP && nextType !== TokenType.LIMIT) {
                        alias = this.advance().value;
                    }
                }
                return { type: 'scalar_subquery', subquery, alias };
            }
            // Not a subquery, restore position
            this.pos = savedPos;
        }

        // Check for CASE expression
        if (this.match(TokenType.CASE)) {
            const caseExpr = this.parseCaseExpr();
            let alias = null;
            if (this.match(TokenType.AS)) {
                alias = this.expect(TokenType.IDENTIFIER).value;
            }
            return { type: 'case', expr: caseExpr, alias };
        }

        // Check for window functions: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, NTILE, PERCENT_RANK, CUME_DIST, FIRST_VALUE, LAST_VALUE, NTH_VALUE
        const windowFuncs = [TokenType.ROW_NUMBER, TokenType.RANK, TokenType.DENSE_RANK, TokenType.LAG, TokenType.LEAD,
                             TokenType.NTILE, TokenType.PERCENT_RANK, TokenType.CUME_DIST, TokenType.FIRST_VALUE, TokenType.LAST_VALUE, TokenType.NTH_VALUE];
        for (const funcType of windowFuncs) {
            if (this.match(funcType)) {
                const funcName = funcType.toLowerCase();
                this.expect(TokenType.LPAREN);
                let arg = null;
                let args = [];
                // Handle function arguments
                if (funcName === 'lag' || funcName === 'lead' || funcName === 'first_value' || funcName === 'last_value') {
                    if (!this.check(TokenType.RPAREN)) {
                        arg = this.parseColumnRef();
                    }
                } else if (funcName === 'ntile') {
                    // NTILE takes a number argument
                    arg = parseInt(this.expect(TokenType.NUMBER).value, 10);
                } else if (funcName === 'nth_value') {
                    // NTH_VALUE(col, n)
                    arg = this.parseColumnRef();
                    if (this.match(TokenType.COMMA)) {
                        args.push(arg);
                        args.push(parseInt(this.expect(TokenType.NUMBER).value, 10));
                    }
                }
                this.expect(TokenType.RPAREN);
                this.expect(TokenType.OVER);
                const windowSpec = this.parseOverSpec();

                // Optional alias
                let alias = null;
                if (this.match(TokenType.AS)) {
                    alias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.peek().type === TokenType.IDENTIFIER) {
                    alias = this.advance().value;
                }

                return { type: 'window', func: funcName, arg, args: args.length ? args : null, over: windowSpec, alias };
            }
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

                // Check for OVER (window function)
                if (this.match(TokenType.OVER)) {
                    const windowSpec = this.parseOverSpec();
                    let alias = null;
                    if (this.match(TokenType.AS)) {
                        alias = this.expect(TokenType.IDENTIFIER).value;
                    } else if (this.peek().type === TokenType.IDENTIFIER) {
                        alias = this.advance().value;
                    }
                    return { type: 'window', func: funcName.toLowerCase(), arg, over: windowSpec, alias };
                }

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

        // Check for string-based aggregate functions (STDDEV, VARIANCE, MEDIAN, STRING_AGG)
        const stringAggFuncs = ['STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];
        if (this.peek().type === TokenType.IDENTIFIER) {
            const funcName = this.peek().value.toUpperCase();
            if (stringAggFuncs.includes(funcName)) {
                this.advance();
                this.expect(TokenType.LPAREN);
                let arg;
                let separator = null;

                if (this.match(TokenType.DISTINCT)) {
                    const col = this.parseColumnRef();
                    arg = { distinct: true, column: col };
                } else {
                    arg = this.parseColumnRef();
                }

                // STRING_AGG has a second argument for separator
                if ((funcName === 'STRING_AGG' || funcName === 'GROUP_CONCAT') && this.match(TokenType.COMMA)) {
                    separator = this.expect(TokenType.STRING).value;
                    arg = { column: arg, separator };
                }

                this.expect(TokenType.RPAREN);

                // Check for OVER (window function) - some aggregates can be windowed
                if (this.match(TokenType.OVER)) {
                    const windowSpec = this.parseOverSpec();
                    let alias = null;
                    if (this.match(TokenType.AS)) {
                        alias = this.expect(TokenType.IDENTIFIER).value;
                    } else if (this.peek().type === TokenType.IDENTIFIER) {
                        alias = this.advance().value;
                    }
                    return { type: 'window', func: funcName.toLowerCase(), arg, over: windowSpec, alias };
                }

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

        // Parse arithmetic expression (handles columns, literals, functions, and arithmetic operators)
        const expr = this.parseArithmeticExpr();

        // Optional alias
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.peek().type === TokenType.IDENTIFIER) {
            const nextType = this.peek().type;
            if (nextType !== TokenType.FROM && nextType !== TokenType.COMMA &&
                nextType !== TokenType.WHERE && nextType !== TokenType.ORDER &&
                nextType !== TokenType.GROUP && nextType !== TokenType.LIMIT) {
                alias = this.advance().value;
            }
        }

        // Convert result to appropriate type for backward compatibility
        if (expr.type === 'column') {
            return { type: 'column', value: expr.value, alias };
        } else if (expr.type === 'function') {
            return { type: 'function', func: expr.func, args: expr.args, alias };
        } else if (expr.type === 'literal') {
            return { type: 'literal', value: expr.value, alias };
        } else if (expr.type === 'arithmetic') {
            return { type: 'arithmetic', expr, alias };
        }

        // Fallback
        return { type: 'arithmetic', expr, alias };
    }

    // Parse function arguments (comma-separated expressions)
    parseFunctionArgs() {
        const args = [];
        if (this.peek().type !== TokenType.RPAREN) {
            args.push(this.parseFunctionArg());
            while (this.match(TokenType.COMMA)) {
                args.push(this.parseFunctionArg());
            }
        }
        return args;
    }

    // Parse a single function argument (can be column, literal, or nested function)
    parseFunctionArg() {
        // Check for AS keyword (for CAST ... AS type)
        if (this.peek().type === TokenType.AS) {
            return null; // Will be handled by CAST specially
        }

        // Check for nested function call
        const scalarFuncs = ['COALESCE', 'NULLIF', 'UPPER', 'LOWER', 'LENGTH', 'SUBSTR', 'SUBSTRING',
                            'TRIM', 'LTRIM', 'RTRIM', 'CONCAT', 'REPLACE', 'ABS', 'ROUND', 'CEIL',
                            'CEILING', 'FLOOR', 'MOD', 'POWER', 'POW', 'SQRT',
                            // Date/Time functions
                            'NOW', 'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP',
                            'DATE', 'TIME', 'STRFTIME', 'DATE_DIFF', 'DATE_ADD', 'DATE_SUB',
                            'EXTRACT', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND',
                            // Additional string functions
                            'SPLIT', 'LEFT', 'RIGHT', 'LPAD', 'RPAD', 'POSITION', 'INSTR', 'REPEAT', 'REVERSE',
                            // Conditional functions
                            'GREATEST', 'LEAST', 'IIF', 'IF',
                            // Additional math functions
                            'LOG', 'LOG10', 'LN', 'EXP', 'SIN', 'COS', 'TAN', 'ASIN', 'ACOS', 'ATAN', 'ATAN2',
                            'PI', 'RANDOM', 'RAND', 'SIGN', 'DEGREES', 'RADIANS', 'TRUNCATE', 'TRUNC',
                            // REGEXP functions
                            'REGEXP_MATCHES', 'REGEXP_REPLACE', 'REGEXP_EXTRACT', 'REGEXP_SUBSTR',
                            'REGEXP_SPLIT', 'REGEXP_COUNT',
                            // JSON functions
                            'JSON_EXTRACT', 'JSON_VALUE', 'JSON_OBJECT', 'JSON_ARRAY',
                            'JSON_KEYS', 'JSON_LENGTH', 'JSON_TYPE', 'JSON_VALID',
                            // Array functions
                            'ARRAY_LENGTH', 'ARRAY_CONTAINS', 'ARRAY_POSITION', 'ARRAY_APPEND',
                            'ARRAY_REMOVE', 'ARRAY_SLICE', 'ARRAY_CONCAT', 'UNNEST',
                            // UUID functions
                            'UUID', 'GEN_RANDOM_UUID', 'UUID_STRING', 'IS_UUID',
                            // Binary/Bit functions
                            'BIT_COUNT', 'HEX', 'UNHEX', 'ENCODE', 'DECODE'];
        if (this.peek().type === TokenType.IDENTIFIER) {
            const funcName = this.peek().value.toUpperCase();
            if (scalarFuncs.includes(funcName)) {
                this.advance();
                this.expect(TokenType.LPAREN);
                const args = this.parseFunctionArgs();
                this.expect(TokenType.RPAREN);
                return { type: 'function', func: funcName.toLowerCase(), args };
            }
        }

        // String literal
        if (this.peek().type === TokenType.STRING) {
            const left = { type: 'literal', value: this.advance().value };
            return this.tryParseComparisonExpr(left);
        }

        // Number literal
        if (this.peek().type === TokenType.NUMBER) {
            const left = { type: 'literal', value: parseFloat(this.advance().value) };
            return this.tryParseComparisonExpr(left);
        }

        // NULL
        if (this.match(TokenType.NULL)) {
            return { type: 'literal', value: null };
        }

        // ARRAY literal: ARRAY[1, 2, 3]
        if (this.match(TokenType.ARRAY)) {
            const elements = [];
            if (this.match(TokenType.LBRACKET)) {
                if (!this.check(TokenType.RBRACKET)) {
                    elements.push(this.parseArithmeticExpr());
                    while (this.match(TokenType.COMMA)) {
                        elements.push(this.parseArithmeticExpr());
                    }
                }
                this.expect(TokenType.RBRACKET);
            }
            let result = { type: 'array_literal', elements };
            // Check for subscript
            while (this.match(TokenType.LBRACKET)) {
                const index = this.parseArithmeticExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // Bare bracket array: [1, 2, 3]
        if (this.match(TokenType.LBRACKET)) {
            const elements = [];
            if (!this.check(TokenType.RBRACKET)) {
                elements.push(this.parseArithmeticExpr());
                while (this.match(TokenType.COMMA)) {
                    elements.push(this.parseArithmeticExpr());
                }
            }
            this.expect(TokenType.RBRACKET);
            let result = { type: 'array_literal', elements };
            // Check for subscript
            while (this.match(TokenType.LBRACKET)) {
                const index = this.parseArithmeticExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // Column reference
        const col = this.parseColumnRef();
        const left = { type: 'column', value: col };
        return this.tryParseComparisonExpr(left);
    }

    // Check if there's a comparison operator and parse as comparison expression
    tryParseComparisonExpr(left) {
        // Check for comparison operators
        let op = null;
        if (this.match(TokenType.EQ)) op = '=';
        else if (this.match(TokenType.NE)) op = '!=';
        else if (this.match(TokenType.LT)) op = '<';
        else if (this.match(TokenType.LE)) op = '<=';
        else if (this.match(TokenType.GT)) op = '>';
        else if (this.match(TokenType.GE)) op = '>=';

        if (!op) return left;

        // Parse right side of comparison
        let right;
        if (this.peek().type === TokenType.STRING) {
            right = { type: 'literal', value: this.advance().value };
        } else if (this.peek().type === TokenType.NUMBER) {
            right = { type: 'literal', value: parseFloat(this.advance().value) };
        } else if (this.match(TokenType.NULL)) {
            right = { type: 'literal', value: null };
        } else {
            const col = this.parseColumnRef();
            right = { type: 'column', value: col };
        }

        return { type: 'comparison', op, left, right };
    }

    // Parse CASE WHEN ... THEN ... ELSE ... END
    parseCaseExpr() {
        const branches = [];

        // Simple CASE: CASE expr WHEN value THEN result
        // Searched CASE: CASE WHEN condition THEN result
        let caseExpr = null;
        if (this.peek().type !== TokenType.WHEN) {
            caseExpr = this.parseFunctionArg();
        }

        while (this.match(TokenType.WHEN)) {
            const condition = this.parseFunctionArg();
            this.expect(TokenType.THEN);
            const result = this.parseFunctionArg();
            branches.push({ condition, result });
        }

        let elseResult = null;
        if (this.match(TokenType.ELSE)) {
            elseResult = this.parseFunctionArg();
        }

        this.expect(TokenType.END);
        return { caseExpr, branches, elseResult };
    }

    // ========== Arithmetic Expression Parsing ==========
    // Parse arithmetic expression with proper precedence: () > * / > + -
    parseArithmeticExpr() {
        return this.parseAddSub();
    }

    parseAddSub() {
        let left = this.parseMulDiv();
        while (this.peek().type === TokenType.PLUS || this.peek().type === TokenType.MINUS) {
            const op = this.advance().type === TokenType.PLUS ? '+' : '-';
            const right = this.parseMulDiv();
            left = { type: 'arithmetic', op, left, right };
        }
        return left;
    }

    parseMulDiv() {
        let left = this.parseBitwise();
        while (this.peek().type === TokenType.STAR || this.peek().type === TokenType.SLASH) {
            const op = this.advance().type === TokenType.STAR ? '*' : '/';
            const right = this.parseBitwise();
            left = { type: 'arithmetic', op, left, right };
        }
        return left;
    }

    parseBitwise() {
        let left = this.parseUnary();
        const bitwiseOps = [TokenType.AMPERSAND, TokenType.PIPE, TokenType.CARET, TokenType.LSHIFT, TokenType.RSHIFT];
        while (bitwiseOps.includes(this.peek().type)) {
            const token = this.advance();
            let op;
            switch (token.type) {
                case TokenType.AMPERSAND: op = '&'; break;
                case TokenType.PIPE: op = '|'; break;
                case TokenType.CARET: op = '^'; break;
                case TokenType.LSHIFT: op = '<<'; break;
                case TokenType.RSHIFT: op = '>>'; break;
            }
            const right = this.parseUnary();
            left = { type: 'arithmetic', op, left, right };
        }
        return left;
    }

    parseUnary() {
        // Handle unary minus
        if (this.match(TokenType.MINUS)) {
            const operand = this.parseUnary();
            return { type: 'arithmetic', op: 'unary-', operand };
        }
        // Handle bitwise NOT
        if (this.match(TokenType.TILDE)) {
            const operand = this.parseUnary();
            return { type: 'arithmetic', op: 'unary~', operand };
        }
        return this.parseArithmeticPrimary();
    }

    parseArithmeticPrimary() {
        // Parenthesized expression
        if (this.match(TokenType.LPAREN)) {
            const expr = this.parseArithmeticExpr();
            this.expect(TokenType.RPAREN);
            return expr;
        }

        // Number literal
        if (this.peek().type === TokenType.NUMBER) {
            return { type: 'literal', value: parseFloat(this.advance().value) };
        }

        // String literal
        if (this.peek().type === TokenType.STRING) {
            return { type: 'literal', value: this.advance().value };
        }

        // NULL literal
        if (this.match(TokenType.NULL)) {
            return { type: 'literal', value: null };
        }

        // ARRAY constructor: ARRAY[1, 2, 3] with optional subscript ARRAY[1,2,3][1]
        if (this.match(TokenType.ARRAY)) {
            const elements = [];
            if (this.match(TokenType.LBRACKET)) {
                if (!this.check(TokenType.RBRACKET)) {
                    elements.push(this.parseArithmeticExpr());
                    while (this.match(TokenType.COMMA)) {
                        elements.push(this.parseArithmeticExpr());
                    }
                }
                this.expect(TokenType.RBRACKET);
            }
            let result = { type: 'array_literal', elements };
            // Check for subscript
            while (this.match(TokenType.LBRACKET)) {
                const index = this.parseArithmeticExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // Bare bracket array: [1, 2, 3] with optional subscript
        if (this.match(TokenType.LBRACKET)) {
            const elements = [];
            if (!this.check(TokenType.RBRACKET)) {
                elements.push(this.parseArithmeticExpr());
                while (this.match(TokenType.COMMA)) {
                    elements.push(this.parseArithmeticExpr());
                }
            }
            this.expect(TokenType.RBRACKET);
            let result = { type: 'array_literal', elements };
            // Check for subscript
            while (this.match(TokenType.LBRACKET)) {
                const index = this.parseArithmeticExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // EXCLUDED.column for UPSERT
        if (this.match(TokenType.EXCLUDED)) {
            this.expect(TokenType.DOT);
            const col = this.expect(TokenType.IDENTIFIER).value;
            return { type: 'column', value: `EXCLUDED.${col}` };
        }

        // CAST function has special syntax: CAST(value AS type)
        if (this.match(TokenType.CAST)) {
            this.expect(TokenType.LPAREN);
            const value = this.parseArithmeticExpr();
            this.expect(TokenType.AS);
            const targetType = this.parseDataType();
            this.expect(TokenType.RPAREN);
            return { type: 'function', func: 'cast', args: [value, { type: 'literal', value: targetType }] };
        }

        // Tokens that can also be function names (Date/Time: YEAR, MONTH, etc.; String: LEFT, RIGHT)
        const funcTokens = [TokenType.YEAR, TokenType.MONTH, TokenType.DAY, TokenType.HOUR, TokenType.MINUTE, TokenType.SECOND,
                           TokenType.LEFT, TokenType.RIGHT];
        for (const funcType of funcTokens) {
            if (this.peek().type === funcType) {
                const funcName = this.advance().type.toLowerCase();
                this.expect(TokenType.LPAREN);
                const args = this.parseFunctionArgs();
                this.expect(TokenType.RPAREN);
                return { type: 'function', func: funcName, args };
            }
        }

        // Check for function call (identifier followed by LPAREN)
        if (this.peek().type === TokenType.IDENTIFIER) {
            const nextPos = this.pos + 1;
            if (nextPos < this.tokens.length && this.tokens[nextPos].type === TokenType.LPAREN) {
                // It's a function call - use parseFunctionArg's logic
                const funcName = this.advance().value.toUpperCase();
                this.expect(TokenType.LPAREN);
                const args = this.parseFunctionArgs();
                this.expect(TokenType.RPAREN);
                return { type: 'function', func: funcName.toLowerCase(), args };
            }
        }

        // Column reference (possibly table.column) with optional array subscript
        if (this.peek().type === TokenType.IDENTIFIER) {
            const col = this.parseColumnRef();
            let result = { type: 'column', value: col };
            // Check for array subscript: col[index]
            while (this.match(TokenType.LBRACKET)) {
                const index = this.parseArithmeticExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        throw new Error(`Unexpected token in arithmetic expression: ${this.peek().type}`);
    }

    // Parse window specification: (PARTITION BY ... ORDER BY ...) - OVER already consumed
    parseOverSpec() {
        this.expect(TokenType.LPAREN);

        let partitionBy = null;
        let orderBy = null;

        // Parse PARTITION BY
        if (this.match(TokenType.PARTITION)) {
            this.expect(TokenType.BY);
            partitionBy = [this.parseColumnRef()];
            while (this.match(TokenType.COMMA)) {
                partitionBy.push(this.parseColumnRef());
            }
        }

        // Parse ORDER BY
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

        // Parse window frame specification (ROWS/RANGE BETWEEN ... AND ...)
        let frame = null;
        let frameType = null;
        if (this.match(TokenType.ROWS)) {
            frameType = 'rows';
        } else if (this.match(TokenType.RANGE)) {
            frameType = 'range';
        }
        if (frameType) {
            this.expect(TokenType.BETWEEN);
            const start = this.parseFrameBound();
            this.expect(TokenType.AND);
            const end = this.parseFrameBound();
            frame = { type: frameType, start, end };
        }

        this.expect(TokenType.RPAREN);
        return { partitionBy, orderBy, frame };
    }

    // Parse window frame bound (UNBOUNDED PRECEDING, CURRENT ROW, N PRECEDING, etc.)
    parseFrameBound() {
        if (this.match(TokenType.UNBOUNDED)) {
            if (this.match(TokenType.PRECEDING)) return { type: 'unbounded', direction: 'preceding' };
            if (this.match(TokenType.FOLLOWING)) return { type: 'unbounded', direction: 'following' };
            throw new Error('Expected PRECEDING or FOLLOWING after UNBOUNDED');
        }
        if (this.match(TokenType.CURRENT)) {
            this.expect(TokenType.ROW);
            return { type: 'current' };
        }
        // N PRECEDING or N FOLLOWING
        const n = parseInt(this.expect(TokenType.NUMBER).value, 10);
        if (this.match(TokenType.PRECEDING)) return { type: 'offset', value: n, direction: 'preceding' };
        if (this.match(TokenType.FOLLOWING)) return { type: 'offset', value: n, direction: 'following' };
        throw new Error('Expected PRECEDING or FOLLOWING after number');
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

    // Parse table reference with optional alias (supports subqueries)
    parseTableRef() {
        // Check for subquery: (SELECT ...)
        if (this.match(TokenType.LPAREN)) {
            if (this.peek().type === TokenType.SELECT) {
                const subquery = this.parseSelect();
                this.expect(TokenType.RPAREN);
                // Alias is required for derived tables
                let alias = null;
                if (this.match(TokenType.AS)) {
                    alias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.peek().type === TokenType.IDENTIFIER) {
                    alias = this.advance().value;
                }
                return { type: 'subquery', query: subquery, alias: alias || '__derived' };
            }
            // Not a subquery, put back the LPAREN
            this.pos--;
        }

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
                nextType !== TokenType.FULL && nextType !== TokenType.OUTER &&
                nextType !== TokenType.CROSS && nextType !== TokenType.ON) {
                alias = this.advance().value;
            }
        }
        return { name, alias };
    }

    // Parse JOIN clause
    parseJoin() {
        let joinType = 'INNER';

        if (this.match(TokenType.FULL)) {
            this.match(TokenType.OUTER); // OUTER is optional
            this.expect(TokenType.JOIN);
            joinType = 'FULL';
        } else if (this.match(TokenType.CROSS)) {
            this.expect(TokenType.JOIN);
            joinType = 'CROSS';
        } else if (this.match(TokenType.LEFT)) {
            this.match(TokenType.OUTER); // OUTER is optional
            this.expect(TokenType.JOIN);
            joinType = 'LEFT';
        } else if (this.match(TokenType.RIGHT)) {
            this.match(TokenType.OUTER); // OUTER is optional
            this.expect(TokenType.JOIN);
            joinType = 'RIGHT';
        } else if (this.match(TokenType.INNER)) {
            this.expect(TokenType.JOIN);
            joinType = 'INNER';
        } else {
            this.expect(TokenType.JOIN); // Just JOIN
        }

        const table = this.parseTableRef();

        // CROSS JOIN has no ON clause
        let on = null;
        if (joinType !== 'CROSS') {
            this.expect(TokenType.ON);
            on = this.parseJoinCondition();
        }

        return { type: joinType, table, on };
    }

    // Parse JOIN ON condition with compound expressions (AND/OR)
    parseJoinCondition() {
        return this.parseJoinOrExpr();
    }

    parseJoinOrExpr() {
        let left = this.parseJoinAndExpr();
        while (this.match(TokenType.OR)) {
            left = { op: 'OR', left, right: this.parseJoinAndExpr() };
        }
        return left;
    }

    parseJoinAndExpr() {
        let left = this.parseJoinComparison();
        while (this.match(TokenType.AND)) {
            left = { op: 'AND', left, right: this.parseJoinComparison() };
        }
        return left;
    }

    parseJoinComparison() {
        // Handle parenthesized expressions
        if (this.match(TokenType.LPAREN)) {
            const expr = this.parseJoinOrExpr();
            this.expect(TokenType.RPAREN);
            return expr;
        }

        const left = this.parseColumnRef();
        let op;
        if (this.match(TokenType.EQ)) op = '=';
        else if (this.match(TokenType.NE)) op = '!=';
        else if (this.match(TokenType.LT)) op = '<';
        else if (this.match(TokenType.LE)) op = '<=';
        else if (this.match(TokenType.GT)) op = '>';
        else if (this.match(TokenType.GE)) op = '>=';
        else throw new Error('Expected comparison operator in JOIN condition');

        const right = this.parseColumnRef();
        return { op, left, right };
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
        // Handle NOT EXISTS first
        if (this.match(TokenType.NOT)) {
            if (this.match(TokenType.EXISTS)) {
                this.expect(TokenType.LPAREN);
                const subquery = this.parseSelect();
                this.expect(TokenType.RPAREN);
                return { op: 'NOT EXISTS', subquery };
            }
            // Put back NOT - it might be part of NOT IN, NOT LIKE, etc.
            this.pos--;
        }

        // Handle EXISTS
        if (this.match(TokenType.EXISTS)) {
            this.expect(TokenType.LPAREN);
            const subquery = this.parseSelect();
            this.expect(TokenType.RPAREN);
            return { op: 'EXISTS', subquery };
        }

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

        // Handle IS NULL / IS NOT NULL
        if (this.match(TokenType.IS)) {
            const isNot = this.match(TokenType.NOT);
            this.expect(TokenType.NULL);
            return { op: isNot ? 'IS NOT NULL' : 'IS NULL', column };
        }

        // Handle NOT BETWEEN, NOT IN, NOT LIKE
        const isNot = this.match(TokenType.NOT);

        // Handle BETWEEN / NOT BETWEEN
        if (this.match(TokenType.BETWEEN)) {
            const low = this.parseValue();
            this.expect(TokenType.AND);
            const high = this.parseValue();
            return { op: isNot ? 'NOT BETWEEN' : 'BETWEEN', column, low, high };
        }

        // Handle IN / NOT IN (with subquery support)
        if (this.match(TokenType.IN)) {
            this.expect(TokenType.LPAREN);
            // Check if it's a subquery
            if (this.check(TokenType.SELECT)) {
                const subquery = this.parseSelect();
                this.expect(TokenType.RPAREN);
                return { op: isNot ? 'NOT IN SUBQUERY' : 'IN SUBQUERY', column, subquery };
            }
            // Otherwise it's a list of values
            const values = [this.parseValue()];
            while (this.match(TokenType.COMMA)) {
                values.push(this.parseValue());
            }
            this.expect(TokenType.RPAREN);
            return { op: isNot ? 'NOT IN' : 'IN', column, values };
        }

        // Handle LIKE / NOT LIKE (move before other operators)
        if (this.match(TokenType.LIKE)) {
            const value = this.parseValue();
            return { op: isNot ? 'NOT LIKE' : 'LIKE', column, value };
        }

        // If we consumed NOT but didn't match BETWEEN/IN/LIKE, error
        if (isNot) {
            throw new Error('Expected BETWEEN, IN, or LIKE after NOT');
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
        else throw new Error(`Expected comparison operator`);

        // Parse value - can be literal value OR column reference (for correlated subqueries)
        let value;
        const nextToken = this.peek();
        if (nextToken.type === TokenType.IDENTIFIER) {
            // Could be a column reference (e.g., d.id in correlated subquery)
            value = this.parseColumnRef();
        } else {
            value = this.parseValue();
        }
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

// Flatten joined row for DML with JOIN support (UPDATE...FROM, DELETE...USING)
function flattenJoinedRow(jr) {
    const flat = {};
    for (const [alias, row] of Object.entries(jr)) {
        if (alias === '__idx') continue;
        if (typeof row === 'object' && row !== null) {
            for (const [col, val] of Object.entries(row)) {
                flat[`${alias}.${col}`] = val;
                if (!(col in flat)) flat[col] = val;
            }
        }
    }
    return flat;
}

// Evaluate compound JOIN condition (supports AND/OR and multiple comparison operators)
function evalJoinCondition(condition, leftRow, rightRow, tableAliases) {
    if (!condition) return true;

    // Handle AND/OR compound conditions
    if (condition.op === 'AND') {
        return evalJoinCondition(condition.left, leftRow, rightRow, tableAliases) &&
               evalJoinCondition(condition.right, leftRow, rightRow, tableAliases);
    }
    if (condition.op === 'OR') {
        return evalJoinCondition(condition.left, leftRow, rightRow, tableAliases) ||
               evalJoinCondition(condition.right, leftRow, rightRow, tableAliases);
    }

    // Handle comparison - try both orderings since we don't know which row has which column
    const leftVal = getColumnValue(leftRow, condition.left, tableAliases) ??
                    getColumnValue(rightRow, condition.left, tableAliases);
    const rightVal = getColumnValue(rightRow, condition.right, tableAliases) ??
                     getColumnValue(leftRow, condition.right, tableAliases);

    switch (condition.op) {
        case '=': return leftVal === rightVal;
        case '!=': return leftVal !== rightVal;
        case '<': return leftVal < rightVal;
        case '<=': return leftVal <= rightVal;
        case '>': return leftVal > rightVal;
        case '>=': return leftVal >= rightVal;
        default: return false;
    }
}

function evalWhere(where, row, tableAliases = {}) {
    if (!where) return true;

    // Helper to resolve value - could be literal or column reference
    const resolveValue = (val) => {
        if (val && typeof val === 'object' && (val.table || val.column)) {
            // It's a column reference
            return getColumnValue(row, val, tableAliases);
        }
        return val;
    };

    switch (where.op) {
        case 'AND':
            return evalWhere(where.left, row, tableAliases) && evalWhere(where.right, row, tableAliases);
        case 'OR':
            return evalWhere(where.left, row, tableAliases) || evalWhere(where.right, row, tableAliases);
        case '=': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val === compareVal;
        }
        case '!=':
        case '<>': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val !== compareVal;
        }
        case '<': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val < compareVal;
        }
        case '<=': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val <= compareVal;
        }
        case '>': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val > compareVal;
        }
        case '>=': {
            const val = getColumnValue(row, where.column, tableAliases);
            const compareVal = resolveValue(where.value);
            return val >= compareVal;
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
        case 'NOT IN': {
            const val = getColumnValue(row, where.column, tableAliases);
            return !where.values.includes(val);
        }
        case 'NOT BETWEEN': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val < where.low || val > where.high;
        }
        case 'NOT LIKE': {
            const val = getColumnValue(row, where.column, tableAliases);
            const pattern = where.value.replace(/%/g, '.*').replace(/_/g, '.');
            return !new RegExp(`^${pattern}$`, 'i').test(val);
        }
        case 'IS NULL': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val === null || val === undefined;
        }
        case 'IS NOT NULL': {
            const val = getColumnValue(row, where.column, tableAliases);
            return val !== null && val !== undefined;
        }
        case 'IN SUBQUERY': {
            const val = getColumnValue(row, where.column, tableAliases);
            // subqueryValues should be pre-populated before eval
            return where.subqueryValues ? where.subqueryValues.includes(val) : false;
        }
        case 'NOT IN SUBQUERY': {
            const val = getColumnValue(row, where.column, tableAliases);
            return where.subqueryValues ? !where.subqueryValues.includes(val) : true;
        }
        case 'EXISTS': {
            // existsResult should be pre-populated before eval
            return where.existsResult === true;
        }
        case 'NOT EXISTS': {
            return where.existsResult === false;
        }
        default:
            return true;
    }
}

// Pre-execute subqueries in WHERE clause and populate subqueryValues
async function preExecuteSubqueries(where, db) {
    if (!where) return;

    if (where.op === 'AND' || where.op === 'OR') {
        await preExecuteSubqueries(where.left, db);
        await preExecuteSubqueries(where.right, db);
        return;
    }

    if (where.op === 'IN SUBQUERY' || where.op === 'NOT IN SUBQUERY') {
        // Execute the subquery and extract values
        const subAst = where.subquery;
        const subRows = await db.select(subAst.table);

        // Get the first column from each row
        const firstCol = subAst.columns[0];
        const colName = firstCol.type === 'column'
            ? (firstCol.value.column || firstCol.value)
            : (typeof firstCol === 'string' ? firstCol : firstCol.value);

        where.subqueryValues = subRows.map(row => row[colName]);
    }

    if (where.op === 'EXISTS' || where.op === 'NOT EXISTS') {
        // Execute the subquery and check if any rows exist
        const result = await executeAST(db, where.subquery);
        where.existsResult = result.rows.length > 0;
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
        case 'stddev': case 'stddev_samp': {
            const vals = rows.map(r => r[colName]).filter(v => v != null && typeof v === 'number');
            if (vals.length < 2) return null;
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (vals.length - 1);
            return Math.sqrt(variance);
        }
        case 'stddev_pop': {
            const vals = rows.map(r => r[colName]).filter(v => v != null && typeof v === 'number');
            if (vals.length === 0) return null;
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
            return Math.sqrt(variance);
        }
        case 'variance': case 'var_samp': {
            const vals = rows.map(r => r[colName]).filter(v => v != null && typeof v === 'number');
            if (vals.length < 2) return null;
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            return vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (vals.length - 1);
        }
        case 'var_pop': {
            const vals = rows.map(r => r[colName]).filter(v => v != null && typeof v === 'number');
            if (vals.length === 0) return null;
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            return vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
        }
        case 'median': {
            const vals = rows.map(r => r[colName]).filter(v => v != null && typeof v === 'number').sort((a, b) => a - b);
            if (vals.length === 0) return null;
            const mid = Math.floor(vals.length / 2);
            return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
        }
        case 'string_agg': case 'group_concat': {
            const actualCol = typeof arg === 'object' && arg.column ? arg.column : colName;
            const separator = typeof arg === 'object' && arg.separator != null ? arg.separator : ',';
            const vals = rows.map(r => r[actualCol]).filter(v => v != null).map(String);
            return vals.join(separator);
        }
        default:
            return null;
    }
}

// Evaluate arithmetic expression for a single row
function evaluateArithmeticExpr(expr, row, tableAliases = {}, excluded = null) {
    if (!expr) return null;

    switch (expr.type) {
        case 'literal':
            return expr.value;
        case 'column': {
            // Handle EXCLUDED.column for UPSERT
            const colName = typeof expr.value === 'string' ? expr.value : expr.value?.column;
            if (colName && colName.toUpperCase().startsWith('EXCLUDED.')) {
                const field = colName.substring(9);
                return excluded?.[field] ?? null;
            }
            return getColumnValue(row, expr.value, tableAliases);
        }
        case 'function':
            return evaluateScalarFunction(expr.func, expr.args, row, tableAliases);
        case 'array_literal':
            return expr.elements.map(el => evaluateArithmeticExpr(el, row, tableAliases));
        case 'subscript': {
            const arr = evaluateArithmeticExpr(expr.array, row, tableAliases);
            const idx = evaluateArithmeticExpr(expr.index, row, tableAliases);
            if (!Array.isArray(arr) || idx == null) return null;
            return arr[idx - 1] ?? null; // SQL uses 1-based indexing
        }
        case 'arithmetic':
            if (expr.op === 'unary-') {
                const operand = evaluateArithmeticExpr(expr.operand, row, tableAliases);
                return operand != null ? -operand : null;
            }
            if (expr.op === 'unary~') {
                const operand = evaluateArithmeticExpr(expr.operand, row, tableAliases);
                return operand != null ? ~(operand | 0) : null;
            }
            const left = evaluateArithmeticExpr(expr.left, row, tableAliases);
            const right = evaluateArithmeticExpr(expr.right, row, tableAliases);
            if (left == null || right == null) return null;
            switch (expr.op) {
                case '+': return left + right;
                case '-': return left - right;
                case '*': return left * right;
                case '/': return right !== 0 ? left / right : null;
                // Bitwise operators
                case '&': return (left | 0) & (right | 0);
                case '|': return (left | 0) | (right | 0);
                case '^': return (left | 0) ^ (right | 0);
                case '<<': return (left | 0) << (right | 0);
                case '>>': return (left | 0) >> (right | 0);
                default: return null;
            }
        default:
            return null;
    }
}

// Evaluate scalar function for a single row
function evaluateScalarFunction(func, args, row, tableAliases = {}) {
    const evalArg = (arg) => {
        if (!arg) return null;
        if (arg.type === 'literal') return arg.value;
        if (arg.type === 'column') return getColumnValue(row, arg.value, tableAliases);
        if (arg.type === 'function') return evaluateScalarFunction(arg.func, arg.args, row, tableAliases);
        if (arg.type === 'array_literal') return arg.elements.map(el => evalArg(el));
        if (arg.type === 'subscript') {
            const arr = evalArg(arg.array);
            const idx = evalArg(arg.index);
            if (!Array.isArray(arr) || idx == null) return null;
            return arr[idx - 1] ?? null; // SQL 1-indexed
        }
        if (arg.type === 'arithmetic') {
            if (arg.op === 'unary-') return -evalArg(arg.operand);
            if (arg.op === 'unary~') return ~(evalArg(arg.operand) | 0);
            const left = evalArg(arg.left);
            const right = evalArg(arg.right);
            switch (arg.op) {
                case '+': return left + right;
                case '-': return left - right;
                case '*': return left * right;
                case '/': return right !== 0 ? left / right : null;
                case '&': return (left | 0) & (right | 0);
                case '|': return (left | 0) | (right | 0);
                case '^': return (left | 0) ^ (right | 0);
                case '<<': return (left | 0) << (right | 0);
                case '>>': return (left | 0) >> (right | 0);
                default: return null;
            }
        }
        if (arg.type === 'comparison') {
            const left = evalArg(arg.left);
            const right = evalArg(arg.right);
            switch (arg.op) {
                case '=': return left === right;
                case '!=': case '<>': return left !== right;
                case '<': return left < right;
                case '<=': return left <= right;
                case '>': return left > right;
                case '>=': return left >= right;
                default: return false;
            }
        }
        return null;
    };

    switch (func) {
        // COALESCE - return first non-null
        case 'coalesce': {
            for (const arg of args) {
                const val = evalArg(arg);
                if (val !== null && val !== undefined) return val;
            }
            return null;
        }
        // NULLIF - return null if args are equal
        case 'nullif': {
            const a = evalArg(args[0]);
            const b = evalArg(args[1]);
            return a === b ? null : a;
        }
        // String functions
        case 'upper':
            return String(evalArg(args[0]) ?? '').toUpperCase();
        case 'lower':
            return String(evalArg(args[0]) ?? '').toLowerCase();
        case 'length':
            return String(evalArg(args[0]) ?? '').length;
        case 'substr':
        case 'substring': {
            const str = String(evalArg(args[0]) ?? '');
            const start = (evalArg(args[1]) ?? 1) - 1; // SQL is 1-indexed
            const len = args[2] ? evalArg(args[2]) : undefined;
            return len !== undefined ? str.substr(start, len) : str.substr(start);
        }
        case 'trim':
            return String(evalArg(args[0]) ?? '').trim();
        case 'ltrim':
            return String(evalArg(args[0]) ?? '').trimStart();
        case 'rtrim':
            return String(evalArg(args[0]) ?? '').trimEnd();
        case 'concat':
            return args.map(a => String(evalArg(a) ?? '')).join('');
        case 'replace': {
            const str = String(evalArg(args[0]) ?? '');
            const from = String(evalArg(args[1]) ?? '');
            const to = String(evalArg(args[2]) ?? '');
            return str.split(from).join(to);
        }
        // Math functions
        case 'abs':
            return Math.abs(evalArg(args[0]) ?? 0);
        case 'round': {
            const val = evalArg(args[0]) ?? 0;
            const decimals = args[1] ? evalArg(args[1]) : 0;
            const factor = Math.pow(10, decimals);
            return Math.round(val * factor) / factor;
        }
        case 'ceil':
        case 'ceiling':
            return Math.ceil(evalArg(args[0]) ?? 0);
        case 'floor':
            return Math.floor(evalArg(args[0]) ?? 0);
        case 'mod':
            return (evalArg(args[0]) ?? 0) % (evalArg(args[1]) ?? 1);
        case 'power':
        case 'pow':
            return Math.pow(evalArg(args[0]) ?? 0, evalArg(args[1]) ?? 1);
        case 'sqrt':
            return Math.sqrt(evalArg(args[0]) ?? 0);
        case 'truncate':
        case 'trunc': {
            const val = evalArg(args[0]) ?? 0;
            const scale = args[1] ? evalArg(args[1]) : 0;
            if (scale === 0) return Math.trunc(val);
            const factor = Math.pow(10, scale);
            return Math.trunc(val * factor) / factor;
        }
        case 'sign': {
            const v = evalArg(args[0]);
            if (v == null) return null;
            return v > 0 ? 1 : v < 0 ? -1 : 0;
        }
        case 'log':
        case 'ln':
            return Math.log(evalArg(args[0]) ?? 0);
        case 'log10':
            return Math.log10(evalArg(args[0]) ?? 0);
        case 'exp':
            return Math.exp(evalArg(args[0]) ?? 0);
        case 'sin':
            return Math.sin(evalArg(args[0]) ?? 0);
        case 'cos':
            return Math.cos(evalArg(args[0]) ?? 0);
        case 'tan':
            return Math.tan(evalArg(args[0]) ?? 0);
        case 'asin':
            return Math.asin(evalArg(args[0]) ?? 0);
        case 'acos':
            return Math.acos(evalArg(args[0]) ?? 0);
        case 'atan':
            return Math.atan(evalArg(args[0]) ?? 0);
        case 'atan2':
            return Math.atan2(evalArg(args[0]) ?? 0, evalArg(args[1]) ?? 0);
        case 'pi':
            return Math.PI;
        case 'random':
        case 'rand':
            return Math.random();
        case 'degrees':
            return (evalArg(args[0]) ?? 0) * (180 / Math.PI);
        case 'radians':
            return (evalArg(args[0]) ?? 0) * (Math.PI / 180);

        // ========== Conditional Functions ==========
        case 'greatest': {
            const values = args.map(evalArg).filter(v => v != null);
            return values.length ? Math.max(...values) : null;
        }
        case 'least': {
            const values = args.map(evalArg).filter(v => v != null);
            return values.length ? Math.min(...values) : null;
        }
        case 'iif':
        case 'if': {
            const condition = evalArg(args[0]);
            return condition ? evalArg(args[1]) : evalArg(args[2]);
        }

        // ========== Type Casting ==========
        case 'cast': {
            const value = evalArg(args[0]);
            const targetType = String(args[1]?.value || args[1] || '').toUpperCase();
            if (value == null) return null;
            switch (targetType) {
                case 'INTEGER': case 'INT': case 'BIGINT': return Math.trunc(Number(value));
                case 'REAL': case 'FLOAT': case 'DOUBLE': return Number(value);
                case 'TEXT': case 'VARCHAR': case 'STRING': return String(value);
                case 'BOOLEAN': case 'BOOL': return Boolean(value);
                default: return value;
            }
        }

        // ========== Date/Time Functions ==========
        case 'now':
        case 'current_timestamp':
            return new Date().toISOString();
        case 'current_date':
            return new Date().toISOString().split('T')[0];
        case 'current_time':
            return new Date().toISOString().split('T')[1].split('.')[0];
        case 'date': {
            const val = evalArg(args[0]);
            if (!val) return null;
            const d = new Date(val);
            return isNaN(d.getTime()) ? null : d.toISOString().split('T')[0];
        }
        case 'time': {
            const val = evalArg(args[0]);
            if (!val) return null;
            const d = new Date(val);
            return isNaN(d.getTime()) ? null : d.toISOString().split('T')[1].split('.')[0];
        }
        case 'strftime': {
            const format = String(evalArg(args[0]) ?? '');
            const dateVal = evalArg(args[1]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            if (isNaN(d.getTime())) return null;
            // Use UTC methods for consistency with date-only strings
            return format
                .replace(/%Y/g, d.getUTCFullYear())
                .replace(/%m/g, String(d.getUTCMonth() + 1).padStart(2, '0'))
                .replace(/%d/g, String(d.getUTCDate()).padStart(2, '0'))
                .replace(/%H/g, String(d.getUTCHours()).padStart(2, '0'))
                .replace(/%M/g, String(d.getUTCMinutes()).padStart(2, '0'))
                .replace(/%S/g, String(d.getUTCSeconds()).padStart(2, '0'))
                .replace(/%w/g, d.getUTCDay())
                .replace(/%j/g, Math.floor((d - new Date(Date.UTC(d.getUTCFullYear(), 0, 0))) / 86400000));
        }
        case 'date_diff': {
            const unit = String(evalArg(args[0]) ?? 'day').toLowerCase();
            const d1 = new Date(evalArg(args[1]));
            const d2 = new Date(evalArg(args[2]));
            if (isNaN(d1.getTime()) || isNaN(d2.getTime())) return null;
            const diffMs = d2.getTime() - d1.getTime();
            switch (unit) {
                case 'second': case 'seconds': return Math.floor(diffMs / 1000);
                case 'minute': case 'minutes': return Math.floor(diffMs / 60000);
                case 'hour': case 'hours': return Math.floor(diffMs / 3600000);
                case 'day': case 'days': return Math.floor(diffMs / 86400000);
                case 'week': case 'weeks': return Math.floor(diffMs / 604800000);
                case 'month': case 'months':
                    return (d2.getFullYear() - d1.getFullYear()) * 12 + (d2.getMonth() - d1.getMonth());
                case 'year': case 'years':
                    return d2.getFullYear() - d1.getFullYear();
                default: return Math.floor(diffMs / 86400000);
            }
        }
        case 'date_add':
        case 'date_sub': {
            const dateVal = evalArg(args[0]);
            const amount = evalArg(args[1]) ?? 0;
            const unit = String(evalArg(args[2]) ?? 'day').toLowerCase();
            if (!dateVal) return null;
            const d = new Date(dateVal);
            if (isNaN(d.getTime())) return null;
            const sign = func === 'date_add' ? 1 : -1;
            switch (unit) {
                case 'second': case 'seconds': d.setSeconds(d.getSeconds() + sign * amount); break;
                case 'minute': case 'minutes': d.setMinutes(d.getMinutes() + sign * amount); break;
                case 'hour': case 'hours': d.setHours(d.getHours() + sign * amount); break;
                case 'day': case 'days': d.setDate(d.getDate() + sign * amount); break;
                case 'week': case 'weeks': d.setDate(d.getDate() + sign * amount * 7); break;
                case 'month': case 'months': d.setMonth(d.getMonth() + sign * amount); break;
                case 'year': case 'years': d.setFullYear(d.getFullYear() + sign * amount); break;
            }
            return d.toISOString();
        }
        case 'extract': {
            const unit = String(evalArg(args[0]) ?? '').toUpperCase();
            const dateVal = evalArg(args[1]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            if (isNaN(d.getTime())) return null;
            switch (unit) {
                case 'YEAR': return d.getFullYear();
                case 'MONTH': return d.getMonth() + 1;
                case 'DAY': return d.getDate();
                case 'HOUR': return d.getHours();
                case 'MINUTE': return d.getMinutes();
                case 'SECOND': return d.getSeconds();
                case 'DOW': case 'DAYOFWEEK': return d.getDay();
                case 'DOY': case 'DAYOFYEAR':
                    return Math.floor((d - new Date(d.getFullYear(), 0, 0)) / 86400000);
                default: return null;
            }
        }
        // Shorthand date extractors (use UTC to avoid timezone issues with date-only strings)
        case 'year': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCFullYear();
        }
        case 'month': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCMonth() + 1;
        }
        case 'day': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCDate();
        }
        case 'hour': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCHours();
        }
        case 'minute': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCMinutes();
        }
        case 'second': {
            const dateVal = evalArg(args[0]);
            if (!dateVal) return null;
            const d = new Date(dateVal);
            return isNaN(d.getTime()) ? null : d.getUTCSeconds();
        }

        // ========== Additional String Functions ==========
        case 'split': {
            const str = String(evalArg(args[0]) ?? '');
            const delimiter = String(evalArg(args[1]) ?? ',');
            return str.split(delimiter);
        }
        case 'left': {
            const str = String(evalArg(args[0]) ?? '');
            const n = evalArg(args[1]) ?? 0;
            return str.substring(0, n);
        }
        case 'right': {
            const str = String(evalArg(args[0]) ?? '');
            const n = evalArg(args[1]) ?? 0;
            return str.substring(Math.max(0, str.length - n));
        }
        case 'lpad': {
            const str = String(evalArg(args[0]) ?? '');
            const len = evalArg(args[1]) ?? 0;
            const pad = String(evalArg(args[2]) ?? ' ');
            return str.padStart(len, pad);
        }
        case 'rpad': {
            const str = String(evalArg(args[0]) ?? '');
            const len = evalArg(args[1]) ?? 0;
            const pad = String(evalArg(args[2]) ?? ' ');
            return str.padEnd(len, pad);
        }
        case 'position':
        case 'instr': {
            const str = String(evalArg(args[0]) ?? '');
            const substr = String(evalArg(args[1]) ?? '');
            const pos = str.indexOf(substr);
            return pos === -1 ? 0 : pos + 1; // SQL uses 1-based indexing
        }
        case 'repeat': {
            const str = String(evalArg(args[0]) ?? '');
            const n = evalArg(args[1]) ?? 0;
            return str.repeat(Math.max(0, n));
        }
        case 'reverse': {
            const str = String(evalArg(args[0]) ?? '');
            return str.split('').reverse().join('');
        }

        // ========== REGEXP Functions ==========
        case 'regexp_matches': {
            const str = String(evalArg(args[0]) ?? '');
            const pattern = String(evalArg(args[1]) ?? '');
            const flags = args[2] ? String(evalArg(args[2])) : '';
            try {
                return new RegExp(pattern, flags).test(str) ? 1 : 0;
            } catch (e) {
                return 0;
            }
        }
        case 'regexp_replace': {
            const str = String(evalArg(args[0]) ?? '');
            const pattern = String(evalArg(args[1]) ?? '');
            const replacement = String(evalArg(args[2]) ?? '');
            const flags = args[3] ? String(evalArg(args[3])) : 'g';
            try {
                return str.replace(new RegExp(pattern, flags), replacement);
            } catch (e) {
                return str;
            }
        }
        case 'regexp_extract':
        case 'regexp_substr': {
            const str = String(evalArg(args[0]) ?? '');
            const pattern = String(evalArg(args[1]) ?? '');
            const groupIndex = args[2] ? parseInt(evalArg(args[2]), 10) : 0;
            try {
                const match = str.match(new RegExp(pattern));
                return match ? (match[groupIndex] ?? null) : null;
            } catch (e) {
                return null;
            }
        }
        case 'regexp_count': {
            const str = String(evalArg(args[0]) ?? '');
            const pattern = String(evalArg(args[1]) ?? '');
            const flags = (args[2] ? String(evalArg(args[2])) : '') + 'g';
            try {
                const matches = str.match(new RegExp(pattern, flags));
                return matches ? matches.length : 0;
            } catch (e) {
                return 0;
            }
        }
        case 'regexp_split': {
            const str = String(evalArg(args[0]) ?? '');
            const pattern = String(evalArg(args[1]) ?? '');
            try {
                return JSON.stringify(str.split(new RegExp(pattern)));
            } catch (e) {
                return JSON.stringify([str]);
            }
        }

        // ========== JSON Functions ==========
        case 'json_extract':
        case 'json_value': {
            const jsonStr = String(evalArg(args[0]) ?? '{}');
            const path = String(evalArg(args[1]) ?? '$');
            try {
                const obj = JSON.parse(jsonStr);
                return navigateJsonPath(obj, path);
            } catch (e) {
                return null;
            }
        }
        case 'json_object': {
            const result = {};
            for (let i = 0; i < args.length; i += 2) {
                const key = String(evalArg(args[i]) ?? '');
                const value = evalArg(args[i + 1]);
                result[key] = value;
            }
            return JSON.stringify(result);
        }
        case 'json_array': {
            const result = args.map(evalArg);
            return JSON.stringify(result);
        }
        case 'json_keys': {
            const jsonStr = String(evalArg(args[0]) ?? '{}');
            try {
                const obj = JSON.parse(jsonStr);
                if (typeof obj === 'object' && obj !== null && !Array.isArray(obj)) {
                    return JSON.stringify(Object.keys(obj));
                }
                return null;
            } catch (e) {
                return null;
            }
        }
        case 'json_length': {
            const jsonStr = String(evalArg(args[0]) ?? '{}');
            try {
                const obj = JSON.parse(jsonStr);
                if (Array.isArray(obj)) return obj.length;
                if (typeof obj === 'object' && obj !== null) return Object.keys(obj).length;
                return null;
            } catch (e) {
                return null;
            }
        }
        case 'json_type': {
            const jsonStr = String(evalArg(args[0]) ?? 'null');
            try {
                const obj = JSON.parse(jsonStr);
                if (obj === null) return 'NULL';
                if (Array.isArray(obj)) return 'ARRAY';
                if (typeof obj === 'object') return 'OBJECT';
                if (typeof obj === 'string') return 'STRING';
                if (typeof obj === 'number') return 'NUMBER';
                if (typeof obj === 'boolean') return 'BOOLEAN';
                return null;
            } catch (e) {
                return null;
            }
        }
        case 'json_valid': {
            const jsonStr = String(evalArg(args[0]) ?? '');
            try {
                JSON.parse(jsonStr);
                return 1;
            } catch (e) {
                return 0;
            }
        }

        // ========== Array Functions ==========
        case 'array_length': {
            const arr = evalArg(args[0]);
            return Array.isArray(arr) ? arr.length : null;
        }
        case 'array_contains': {
            const arr = evalArg(args[0]);
            const value = evalArg(args[1]);
            if (!Array.isArray(arr)) return null;
            return arr.includes(value) ? 1 : 0;
        }
        case 'array_position': {
            const arr = evalArg(args[0]);
            const value = evalArg(args[1]);
            if (!Array.isArray(arr)) return null;
            const idx = arr.indexOf(value);
            return idx === -1 ? null : idx + 1; // SQL 1-based indexing
        }
        case 'array_append': {
            const arr = evalArg(args[0]);
            const value = evalArg(args[1]);
            if (!Array.isArray(arr)) return null;
            return [...arr, value];
        }
        case 'array_remove': {
            const arr = evalArg(args[0]);
            const value = evalArg(args[1]);
            if (!Array.isArray(arr)) return null;
            return arr.filter(el => el !== value);
        }
        case 'array_slice': {
            const arr = evalArg(args[0]);
            const start = (evalArg(args[1]) ?? 1) - 1; // SQL 1-based to JS 0-based
            const end = args[2] ? evalArg(args[2]) - 1 : arr?.length; // SQL 1-based to JS 0-based (exclusive)
            if (!Array.isArray(arr)) return null;
            return arr.slice(start, end);
        }
        case 'array_concat': {
            const arr1 = evalArg(args[0]);
            const arr2 = evalArg(args[1]);
            if (!Array.isArray(arr1) || !Array.isArray(arr2)) return null;
            return [...arr1, ...arr2];
        }
        case 'unnest': {
            // UNNEST typically expands array to rows - in scalar context, return first element
            const arr = evalArg(args[0]);
            return Array.isArray(arr) && arr.length > 0 ? arr[0] : null;
        }

        // ========== UUID Functions ==========
        case 'uuid':
        case 'gen_random_uuid': {
            // Generate UUID v4
            if (typeof crypto !== 'undefined' && crypto.randomUUID) {
                return crypto.randomUUID();
            }
            // Fallback for environments without crypto.randomUUID
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
                const r = Math.random() * 16 | 0;
                return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
            });
        }
        case 'uuid_string': {
            const val = evalArg(args[0]);
            if (val == null) return null;
            return String(val);
        }
        case 'is_uuid': {
            const val = evalArg(args[0]);
            if (val == null) return 0;
            const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
            return uuidRegex.test(String(val)) ? 1 : 0;
        }

        // ========== Binary/Bit Functions ==========
        case 'bit_count': {
            const val = evalArg(args[0]);
            if (val == null) return null;
            let n = val | 0;
            let count = 0;
            // Handle negative numbers by using unsigned right shift
            n = n >>> 0;
            while (n) {
                count += n & 1;
                n >>>= 1;
            }
            return count;
        }
        case 'hex': {
            const val = evalArg(args[0]);
            if (val == null) return null;
            if (typeof val === 'number') {
                return (val >>> 0).toString(16).toUpperCase();
            }
            // String to hex
            return String(val).split('').map(c => c.charCodeAt(0).toString(16).padStart(2, '0')).join('').toUpperCase();
        }
        case 'unhex': {
            const val = evalArg(args[0]);
            if (val == null) return null;
            const hex = String(val);
            let result = '';
            for (let i = 0; i < hex.length; i += 2) {
                result += String.fromCharCode(parseInt(hex.substr(i, 2), 16));
            }
            return result;
        }
        case 'encode': {
            const val = evalArg(args[0]);
            const encoding = String(evalArg(args[1]) ?? 'base64').toLowerCase();
            if (val == null) return null;
            if (encoding === 'base64') {
                return btoa(String(val));
            }
            if (encoding === 'hex') {
                return String(val).split('').map(c => c.charCodeAt(0).toString(16).padStart(2, '0')).join('');
            }
            return val;
        }
        case 'decode': {
            const val = evalArg(args[0]);
            const encoding = String(evalArg(args[1]) ?? 'base64').toLowerCase();
            if (val == null) return null;
            if (encoding === 'base64') {
                try {
                    return atob(String(val));
                } catch (e) {
                    return null;
                }
            }
            if (encoding === 'hex') {
                const hex = String(val);
                let result = '';
                for (let i = 0; i < hex.length; i += 2) {
                    result += String.fromCharCode(parseInt(hex.substr(i, 2), 16));
                }
                return result;
            }
            return val;
        }

        // GROUPING function for ROLLUP/CUBE/GROUPING SETS
        case 'grouping': {
            // GROUPING(column) returns 1 if column is a super-aggregate (rolled up), 0 otherwise
            const arg = args[0];
            if (arg && arg.type === 'column') {
                const colName = typeof arg.value === 'string' ? arg.value : arg.value.column;
                return row[`__grouping_${colName}`] ?? 0;
            }
            return 0;
        }

        default:
            return null;
    }
}

// JSON path navigation helper (supports $.key.subkey and $.array[0])
function parseJsonPath(path) {
    if (!path || !path.startsWith('$')) return [];
    const segments = [];
    let remaining = path.substring(1);
    while (remaining.length > 0) {
        if (remaining.startsWith('.')) {
            remaining = remaining.substring(1);
            const match = remaining.match(/^([a-zA-Z_][a-zA-Z0-9_]*)/);
            if (match) {
                segments.push({ type: 'key', value: match[1] });
                remaining = remaining.substring(match[1].length);
            }
        } else if (remaining.startsWith('[')) {
            const endBracket = remaining.indexOf(']');
            if (endBracket === -1) break;
            const content = remaining.substring(1, endBracket);
            if (/^\d+$/.test(content)) {
                segments.push({ type: 'index', value: parseInt(content, 10) });
            } else if (content.startsWith("'") || content.startsWith('"')) {
                segments.push({ type: 'key', value: content.slice(1, -1) });
            }
            remaining = remaining.substring(endBracket + 1);
        } else {
            break;
        }
    }
    return segments;
}

function navigateJsonPath(obj, path) {
    let current = obj;
    for (const seg of parseJsonPath(path)) {
        if (current == null) return null;
        current = seg.type === 'key' ? current[seg.value] :
                  (Array.isArray(current) ? current[seg.value] : null);
    }
    return typeof current === 'object' ? JSON.stringify(current) : current;
}

// Evaluate scalar subquery for a single outer row (correlated subquery support)
async function evaluateScalarSubquery(subquery, outerRow, db, tableAliases) {
    // Bind outer row references into the subquery's WHERE clause
    const boundSubquery = JSON.parse(JSON.stringify(subquery)); // Deep clone
    if (boundSubquery.where) {
        bindOuterReferences(boundSubquery.where, outerRow, tableAliases);
    }

    const result = await executeAST(db, boundSubquery);
    if (result.rows.length === 0) return null;
    if (result.rows.length > 1) throw new Error('Scalar subquery returned more than one row');

    // Return the first (and only) column value of the first row
    const keys = Object.keys(result.rows[0]);
    return keys.length > 0 ? result.rows[0][keys[0]] : null;
}

// Bind outer row references in WHERE clause for correlated subqueries
function bindOuterReferences(where, outerRow, tableAliases) {
    if (!where) return;

    if (where.op === 'AND' || where.op === 'OR') {
        bindOuterReferences(where.left, outerRow, tableAliases);
        bindOuterReferences(where.right, outerRow, tableAliases);
        return;
    }

    // Check if the value references an outer table column
    if (where.value && typeof where.value === 'object' && where.value.table) {
        const val = getColumnValue(outerRow, where.value, tableAliases);
        if (val !== undefined) {
            where.value = val; // Replace with actual value
        }
    }

    // Also check the column side for cases like outer.col = inner.col
    if (where.column && typeof where.column === 'object' && where.column.table) {
        const val = getColumnValue(outerRow, where.column, tableAliases);
        if (val !== undefined && where.value !== undefined) {
            // Swap: column becomes the inner column, value becomes the outer value
            where.column = where.value;
            where.value = val;
        }
    }
}

// Evaluate CASE WHEN expression for a single row
function evaluateCaseExpr(caseExpr, row, tableAliases = {}) {
    const evalArg = (arg) => {
        if (!arg) return null;
        if (arg.type === 'literal') return arg.value;
        if (arg.type === 'column') return getColumnValue(row, arg.value, tableAliases);
        if (arg.type === 'function') return evaluateScalarFunction(arg.func, arg.args, row, tableAliases);
        if (arg.type === 'array_literal') return arg.elements.map(el => evalArg(el));
        if (arg.type === 'subscript') {
            const arr = evalArg(arg.array);
            const idx = evalArg(arg.index);
            if (!Array.isArray(arr) || idx == null) return null;
            return arr[idx - 1] ?? null;
        }
        if (arg.type === 'arithmetic') {
            if (arg.op === 'unary-') return -evalArg(arg.operand);
            if (arg.op === 'unary~') return ~(evalArg(arg.operand) | 0);
            const left = evalArg(arg.left);
            const right = evalArg(arg.right);
            switch (arg.op) {
                case '+': return left + right;
                case '-': return left - right;
                case '*': return left * right;
                case '/': return right !== 0 ? left / right : null;
                case '&': return (left | 0) & (right | 0);
                case '|': return (left | 0) | (right | 0);
                case '^': return (left | 0) ^ (right | 0);
                case '<<': return (left | 0) << (right | 0);
                case '>>': return (left | 0) >> (right | 0);
                default: return null;
            }
        }
        if (arg.type === 'comparison') {
            const left = evalArg(arg.left);
            const right = evalArg(arg.right);
            switch (arg.op) {
                case '=': return left === right;
                case '!=': case '<>': return left !== right;
                case '<': return left < right;
                case '<=': return left <= right;
                case '>': return left > right;
                case '>=': return left >= right;
                default: return false;
            }
        }
        return null;
    };

    // Simple CASE: CASE expr WHEN value THEN result
    if (caseExpr.caseExpr) {
        const caseVal = evalArg(caseExpr.caseExpr);
        for (const branch of caseExpr.branches) {
            const whenVal = evalArg(branch.condition);
            if (caseVal === whenVal) {
                return evalArg(branch.result);
            }
        }
    } else {
        // Searched CASE: CASE WHEN condition THEN result
        for (const branch of caseExpr.branches) {
            const cond = evalArg(branch.condition);
            if (cond) {
                return evalArg(branch.result);
            }
        }
    }

    // ELSE clause
    return caseExpr.elseResult ? evalArg(caseExpr.elseResult) : null;
}

// Compute window functions for all rows
function computeWindowFunctions(rows, windowCols, tableAliases = {}) {
    if (rows.length === 0) return rows;

    for (const col of windowCols) {
        const alias = col.alias || `${col.func}(...)`;
        const windowKey = `__window_${alias}`;
        const { partitionBy, orderBy } = col.over;

        // Group rows into partitions
        const partitions = new Map();
        for (let i = 0; i < rows.length; i++) {
            const row = rows[i];
            const partKey = partitionBy
                ? partitionBy.map(p => {
                    const colName = typeof p === 'string' ? p : p.column;
                    return String(row[colName]);
                }).join('|')
                : '__all__';
            if (!partitions.has(partKey)) {
                partitions.set(partKey, []);
            }
            partitions.get(partKey).push({ index: i, row });
        }

        // Process each partition
        for (const [, partRows] of partitions) {
            // Sort partition by ORDER BY if specified
            if (orderBy && orderBy.length > 0) {
                partRows.sort((a, b) => {
                    for (const order of orderBy) {
                        const colName = typeof order.column === 'string' ? order.column : order.column.column;
                        const aVal = a.row[colName];
                        const bVal = b.row[colName];
                        if (aVal < bVal) return order.desc ? 1 : -1;
                        if (aVal > bVal) return order.desc ? -1 : 1;
                    }
                    return 0;
                });
            }

            // Helper to get frame bounds
            const getFrameBounds = (i, frame) => {
                if (!frame) {
                    // Default: if ORDER BY exists, UNBOUNDED PRECEDING to CURRENT ROW
                    // Otherwise, entire partition
                    return orderBy && orderBy.length > 0 ? [0, i] : [0, partRows.length - 1];
                }
                let start = 0, end = partRows.length - 1;
                // Calculate start
                if (frame.start.type === 'unbounded' && frame.start.direction === 'preceding') {
                    start = 0;
                } else if (frame.start.type === 'current') {
                    start = i;
                } else if (frame.start.type === 'offset') {
                    start = frame.start.direction === 'preceding' ? i - frame.start.value : i + frame.start.value;
                }
                // Calculate end
                if (frame.end.type === 'unbounded' && frame.end.direction === 'following') {
                    end = partRows.length - 1;
                } else if (frame.end.type === 'current') {
                    end = i;
                } else if (frame.end.type === 'offset') {
                    end = frame.end.direction === 'preceding' ? i - frame.end.value : i + frame.end.value;
                }
                return [Math.max(0, start), Math.min(partRows.length - 1, end)];
            };

            // Compute window function for each row in partition
            for (let i = 0; i < partRows.length; i++) {
                const { index, row } = partRows[i];
                let value;
                const frame = col.over.frame;

                switch (col.func) {
                    case 'row_number':
                        value = i + 1;
                        break;
                    case 'rank': {
                        // Same rank for same ORDER BY values
                        if (i === 0) {
                            value = 1;
                        } else {
                            const prevRow = partRows[i - 1].row;
                            const sameAsPrev = orderBy ? orderBy.every(order => {
                                const colName = typeof order.column === 'string' ? order.column : order.column.column;
                                return row[colName] === prevRow[colName];
                            }) : false;
                            value = sameAsPrev ? rows[partRows[i - 1].index][windowKey] : i + 1;
                        }
                        break;
                    }
                    case 'dense_rank': {
                        if (i === 0) {
                            value = 1;
                        } else {
                            const prevRow = partRows[i - 1].row;
                            const sameAsPrev = orderBy ? orderBy.every(order => {
                                const colName = typeof order.column === 'string' ? order.column : order.column.column;
                                return row[colName] === prevRow[colName];
                            }) : false;
                            value = sameAsPrev ? rows[partRows[i - 1].index][windowKey] : rows[partRows[i - 1].index][windowKey] + 1;
                        }
                        break;
                    }
                    case 'sum': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        const [frameStart, frameEnd] = getFrameBounds(i, frame);
                        const windowRows = partRows.slice(frameStart, frameEnd + 1);
                        value = windowRows.reduce((acc, p) => {
                            const v = p.row[argCol];
                            return acc + (typeof v === 'number' ? v : 0);
                        }, 0);
                        break;
                    }
                    case 'count': {
                        const [frameStart, frameEnd] = getFrameBounds(i, frame);
                        value = frameEnd - frameStart + 1;
                        break;
                    }
                    case 'avg': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        const [frameStart, frameEnd] = getFrameBounds(i, frame);
                        const windowRows = partRows.slice(frameStart, frameEnd + 1);
                        const vals = windowRows.map(p => p.row[argCol]).filter(v => typeof v === 'number');
                        value = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
                        break;
                    }
                    case 'min': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        const [frameStart, frameEnd] = getFrameBounds(i, frame);
                        const windowRows = partRows.slice(frameStart, frameEnd + 1);
                        const vals = windowRows.map(p => p.row[argCol]).filter(v => v != null);
                        value = vals.length > 0 ? Math.min(...vals) : null;
                        break;
                    }
                    case 'max': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        const [frameStart, frameEnd] = getFrameBounds(i, frame);
                        const windowRows = partRows.slice(frameStart, frameEnd + 1);
                        const vals = windowRows.map(p => p.row[argCol]).filter(v => v != null);
                        value = vals.length > 0 ? Math.max(...vals) : null;
                        break;
                    }
                    case 'lag': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        value = i > 0 ? partRows[i - 1].row[argCol] : null;
                        break;
                    }
                    case 'lead': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        value = i < partRows.length - 1 ? partRows[i + 1].row[argCol] : null;
                        break;
                    }
                    case 'ntile': {
                        // NTILE(n) divides partition into n buckets
                        const n = col.arg || 1;
                        const bucketSize = Math.ceil(partRows.length / n);
                        value = Math.min(Math.floor(i / bucketSize) + 1, n);
                        break;
                    }
                    case 'percent_rank': {
                        // (rank - 1) / (partition_size - 1)
                        if (partRows.length <= 1) {
                            value = 0;
                        } else {
                            // Calculate rank first
                            let rank = 1;
                            if (i > 0 && orderBy) {
                                for (let j = 0; j < i; j++) {
                                    const jRow = partRows[j].row;
                                    const isSame = orderBy.every(order => {
                                        const colName = typeof order.column === 'string' ? order.column : order.column.column;
                                        return row[colName] === jRow[colName];
                                    });
                                    if (!isSame) rank = j + 2;
                                }
                            }
                            value = (rank - 1) / (partRows.length - 1);
                        }
                        break;
                    }
                    case 'cume_dist': {
                        // count of rows <= current / partition_size
                        let count = i + 1;
                        if (orderBy) {
                            // Count rows with same or lower order value
                            for (let j = i + 1; j < partRows.length; j++) {
                                const jRow = partRows[j].row;
                                const isSame = orderBy.every(order => {
                                    const colName = typeof order.column === 'string' ? order.column : order.column.column;
                                    return row[colName] === jRow[colName];
                                });
                                if (isSame) count++;
                                else break;
                            }
                        }
                        value = count / partRows.length;
                        break;
                    }
                    case 'first_value': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        value = partRows[0].row[argCol];
                        break;
                    }
                    case 'last_value': {
                        const argCol = typeof col.arg === 'string' ? col.arg : col.arg?.column;
                        // Default frame is UNBOUNDED PRECEDING to CURRENT ROW, so last_value = current row
                        // With explicit UNBOUNDED FOLLOWING, it's partition last
                        const frame = col.over.frame;
                        if (frame && frame.end.type === 'unbounded' && frame.end.direction === 'following') {
                            value = partRows[partRows.length - 1].row[argCol];
                        } else {
                            // Default: current row
                            value = row[argCol];
                        }
                        break;
                    }
                    case 'nth_value': {
                        // NTH_VALUE(col, n) - returns nth value in partition
                        const argCol = col.args ? (typeof col.args[0] === 'string' ? col.args[0] : col.args[0]?.column) : (typeof col.arg === 'string' ? col.arg : col.arg?.column);
                        const n = col.args ? col.args[1] : 1;
                        value = n > 0 && n <= partRows.length ? partRows[n - 1].row[argCol] : null;
                        break;
                    }
                    default:
                        value = null;
                }

                rows[index][windowKey] = value;
            }
        }
    }

    return rows;
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

// Generate query plan from AST for EXPLAIN
function generateQueryPlan(ast) {
    const type = ast.type?.toUpperCase() || 'SELECT';

    // Handle JOINs - they become HASH_JOIN operations
    if (type === 'SELECT' && ast.joins && ast.joins.length > 0) {
        const children = [];

        // Main table scan
        const mainTable = ast.tables?.[0];
        children.push({
            operation: 'SELECT',
            table: typeof mainTable === 'string' ? mainTable : (mainTable?.name || mainTable?.alias || 'unknown'),
            access: 'FULL_SCAN'
        });

        // Join tables
        for (const join of ast.joins) {
            const joinTable = join.table;
            children.push({
                operation: 'SELECT',
                table: typeof joinTable === 'string' ? joinTable : (joinTable?.name || joinTable?.alias || 'unknown'),
                access: 'FULL_SCAN'
            });
        }

        return {
            operation: 'HASH_JOIN',
            joinType: ast.joins[0].type || 'INNER',
            children
        };
    }

    // Standard SELECT/UPDATE/DELETE/INSERT
    const plan = {
        operation: type,
        table: getTableName(ast),
        access: 'FULL_SCAN'
    };

    // Detect optimizations
    const optimizations = [];

    if (ast.where) {
        optimizations.push('PREDICATE_PUSHDOWN');
        plan.filter = stringifyWhereClause(ast.where);
    }

    if (ast.groupBy) {
        optimizations.push('AGGREGATE');
    }

    if (ast.orderBy && ast.orderBy.length > 0) {
        optimizations.push('SORT');
    }

    if (ast.limit !== undefined) {
        optimizations.push('LIMIT');
    }

    if (optimizations.length > 0) {
        plan.optimizations = optimizations;
    }

    return plan;
}

function getTableName(ast) {
    if (ast.table) return ast.table;
    if (ast.tables && ast.tables.length > 0) {
        const t = ast.tables[0];
        return typeof t === 'string' ? t : (t.name || t.alias || 'unknown');
    }
    return 'unknown';
}

function stringifyWhereClause(where) {
    if (!where) return '';
    if (where.type === 'comparison') {
        const left = stringifyExpr(where.left);
        const right = stringifyExpr(where.right);
        return `${left} ${where.op} ${right}`;
    }
    if (where.type === 'logical') {
        const left = stringifyWhereClause(where.left);
        const right = stringifyWhereClause(where.right);
        return `(${left} ${where.op} ${right})`;
    }
    return JSON.stringify(where);
}

function stringifyExpr(expr) {
    if (!expr) return 'null';
    if (expr.type === 'literal') return String(expr.value);
    if (expr.type === 'column') {
        return typeof expr.value === 'string' ? expr.value : expr.value.column;
    }
    return JSON.stringify(expr);
}

async function executeSQL(db, sql) {
    const lexer = new SQLLexer(sql);
    const tokens = lexer.tokenize();
    const parser = new SQLParser(tokens);
    const ast = parser.parse();
    return executeAST(db, ast);
}

async function executeAST(db, ast) {
    const type = ast.type.toUpperCase();

    switch (type) {
        case 'CREATE_TABLE':
            return db.createTable(ast.table, ast.columns, ast.ifNotExists);

        case 'DROP_TABLE':
            return db.dropTable(ast.table, ast.ifExists);

        case 'INSERT': {
            let rows = ast.rows || [];

            // Handle INSERT...SELECT
            if (ast.select) {
                const selectResult = await executeAST(db, ast.select);
                rows = selectResult.rows;

                // Map select columns to insert columns if specified
                if (ast.columns && rows.length > 0) {
                    const selectCols = Object.keys(rows[0]);
                    rows = rows.map(row => {
                        const newRow = {};
                        ast.columns.forEach((col, i) => {
                            newRow[col] = row[selectCols[i]];
                        });
                        return newRow;
                    });
                }
            } else if (rows.length > 0 && Array.isArray(rows[0])) {
                // If rows are arrays (no column names in INSERT), convert to objects using table schema
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

            // Handle ON CONFLICT (UPSERT)
            if (ast.onConflict) {
                const tableState = db.tables.get(ast.table);
                const existingRows = await db.select(ast.table, {});
                const conflictCols = ast.onConflict.columns ||
                    tableState?.schema?.filter(c => c.primaryKey).map(c => c.name) || ['id'];

                const insertedRows = [];
                const updatedRows = [];

                for (const row of rows) {
                    const existingRow = existingRows.find(existing =>
                        conflictCols.every(col => existing[col] === row[col])
                    );

                    if (existingRow) {
                        if (ast.onConflict.action === 'update') {
                            // Evaluate update expressions
                            const updates = {};
                            for (const [col, expr] of Object.entries(ast.onConflict.updates)) {
                                updates[col] = evaluateArithmeticExpr(expr, existingRow, {}, row);
                            }
                            // Apply updates to existing row
                            await db.updateWithExpr(ast.table, ast.onConflict.updates,
                                (r) => conflictCols.every(col => r[col] === row[col]),
                                (expr, r) => evaluateArithmeticExpr(expr, r, {}, row));
                        }
                        // 'nothing': skip insertion
                    } else {
                        insertedRows.push(row);
                    }
                }
                if (insertedRows.length > 0) {
                    return db.insert(ast.table, insertedRows);
                }
                return { success: true };
            }

            return db.insert(ast.table, rows);
        }

        case 'DELETE': {
            // Handle DELETE with USING clause (JOIN-based delete)
            if (ast.using) {
                const mainRows = await db.select(ast.table, {});
                const tableAliases = { [ast.alias || ast.table]: ast.table };

                // Build joined rows
                let joinedRows = mainRows.map((r) => ({ [ast.alias || ast.table]: { ...r } }));
                for (const t of ast.using) {
                    const rightRows = await db.select(t.name, {});
                    tableAliases[t.alias || t.name] = t.name;
                    const newJoined = [];
                    for (const left of joinedRows) {
                        for (const right of rightRows) {
                            newJoined.push({ ...left, [t.alias || t.name]: right });
                        }
                    }
                    joinedRows = newJoined;
                }

                // Filter with WHERE to find matching rows
                if (ast.where) {
                    joinedRows = joinedRows.filter(jr => {
                        const flatRow = flattenJoinedRow(jr);
                        return evalWhere(ast.where, flatRow);
                    });
                }

                // Collect main table rows to delete
                const rowsToDelete = joinedRows.map(jr => jr[ast.alias || ast.table]);
                const tableSchema = db.tables.get(ast.table)?.schema || [];

                // Delete matching rows by comparing all columns
                return db.delete(ast.table, (row) => {
                    return rowsToDelete.some(delRow =>
                        tableSchema.every(col => row[col.name] === delRow[col.name])
                    );
                });
            }

            const predicate = ast.where
                ? (row) => evalWhere(ast.where, row)
                : () => true;
            return db.delete(ast.table, predicate);
        }

        case 'UPDATE': {
            // Handle UPDATE with FROM clause (JOIN-based update)
            if (ast.from) {
                const mainRows = await db.select(ast.table, {});
                const tableAliases = { [ast.alias || ast.table]: ast.table };

                // Build joined rows
                let joinedRows = mainRows.map((r) => ({ [ast.alias || ast.table]: { ...r } }));
                for (const t of ast.from) {
                    const rightRows = await db.select(t.name, {});
                    tableAliases[t.alias || t.name] = t.name;
                    const newJoined = [];
                    for (const left of joinedRows) {
                        for (const right of rightRows) {
                            newJoined.push({ ...left, [t.alias || t.name]: right });
                        }
                    }
                    joinedRows = newJoined;
                }

                // Filter with WHERE
                if (ast.where) {
                    joinedRows = joinedRows.filter(jr => {
                        const flatRow = flattenJoinedRow(jr);
                        return evalWhere(ast.where, flatRow);
                    });
                }

                // Collect matched rows and their update contexts
                const matchedContexts = [];
                for (const jr of joinedRows) {
                    const mainRow = jr[ast.alias || ast.table];
                    matchedContexts.push({
                        mainRow,
                        context: flattenJoinedRow(jr)
                    });
                }

                // Use updateWithExpr - check if row matches any in matchedContexts
                const seen = new Set();
                return db.updateWithExpr(ast.table, ast.updates,
                    (row) => {
                        // Check if this row matches any of the main rows
                        const tableSchema = db.tables.get(ast.table)?.schema || [];
                        for (const m of matchedContexts) {
                            const matches = tableSchema.every(col =>
                                row[col.name] === m.mainRow[col.name]
                            );
                            if (matches && !seen.has(JSON.stringify(m.mainRow))) {
                                seen.add(JSON.stringify(m.mainRow));
                                row.__updateContext = m.context;
                                return true;
                            }
                        }
                        return false;
                    },
                    (expr, row) => {
                        const context = row.__updateContext || row;
                        delete row.__updateContext;
                        return evaluateArithmeticExpr(expr, context, tableAliases);
                    });
            }

            // Simple UPDATE (no FROM) - use updateWithExpr which handles expressions
            const predicate = ast.where
                ? (row) => evalWhere(ast.where, row)
                : () => true;

            return db.updateWithExpr(ast.table, ast.updates, predicate, (expr, row) => evaluateArithmeticExpr(expr, row, {}));
        }

        case 'SELECT': {
            // Build table alias mapping and handle derived tables (subqueries)
            const tableAliases = {};
            const derivedTables = new Map();

            if (ast.tables) {
                for (const t of ast.tables) {
                    if (t.type === 'subquery') {
                        // Execute subquery and store as derived table
                        const subResult = await executeAST(db, t.query);
                        derivedTables.set(t.alias, subResult.rows);
                        tableAliases[t.alias] = t.alias;
                    } else if (t.alias) {
                        tableAliases[t.alias] = t.name;
                    }
                }
            }
            if (ast.joins) {
                for (const j of ast.joins) {
                    if (j.table.type === 'subquery') {
                        const subResult = await executeAST(db, j.table.query);
                        derivedTables.set(j.table.alias, subResult.rows);
                        tableAliases[j.table.alias] = j.table.alias;
                    } else if (j.table.alias) {
                        tableAliases[j.table.alias] = j.table.name;
                    }
                }
            }

            // Fetch data from main table (or derived table)
            // SELECT without FROM (e.g., SELECT 1+1, SELECT JSON_OBJECT(...)) returns single row
            let rows;
            const mainTable = ast.tables?.[0];
            if (!mainTable) {
                // No FROM clause - create a single empty row for expression evaluation
                rows = [{}];
            } else if (mainTable.type === 'subquery') {
                rows = derivedTables.get(mainTable.alias) || [];
            } else {
                rows = await db.select(ast.table, {});
            }

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
                    // Get right table rows (from database or derived table)
                    let rightRows;
                    if (join.table.type === 'subquery') {
                        rightRows = derivedTables.get(join.table.alias) || [];
                    } else {
                        rightRows = await db.select(join.table.name, {});
                    }
                    const newRows = [];
                    const matchedRightIndices = new Set();

                    // Get right table info for namespacing
                    const rightTableName = join.table.alias || join.table.name;

                    // CROSS JOIN: Cartesian product (no ON condition)
                    if (join.type === 'CROSS') {
                        for (const leftRow of rows) {
                            for (const rightRow of rightRows) {
                                const merged = { ...leftRow };
                                for (const key of Object.keys(rightRow)) {
                                    if (!(key in merged)) merged[key] = rightRow[key];
                                    merged[`${rightTableName}.${key}`] = rightRow[key];
                                }
                                newRows.push(merged);
                            }
                        }
                    } else {
                        // INNER, LEFT, RIGHT, FULL JOINs with ON condition
                        for (const leftRow of rows) {
                            let matched = false;
                            for (let ri = 0; ri < rightRows.length; ri++) {
                                const rightRow = rightRows[ri];
                                // Evaluate ON condition (supports compound AND/OR conditions)
                                if (evalJoinCondition(join.on, leftRow, rightRow, tableAliases)) {
                                    matched = true;
                                    matchedRightIndices.add(ri);
                                    // Merge rows - keep left row as-is, add right row with prefix
                                    const merged = { ...leftRow };
                                    for (const key of Object.keys(rightRow)) {
                                        if (!(key in merged)) merged[key] = rightRow[key];
                                        merged[`${rightTableName}.${key}`] = rightRow[key];
                                    }
                                    newRows.push(merged);
                                }
                            }
                            // LEFT JOIN or FULL OUTER JOIN: include left row even if no match
                            if (!matched && (join.type === 'LEFT' || join.type === 'FULL')) {
                                newRows.push({ ...leftRow });
                            }
                        }

                        // RIGHT JOIN or FULL OUTER JOIN: include unmatched right rows
                        if (join.type === 'RIGHT' || join.type === 'FULL') {
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
                    }
                    rows = newRows;
                }
            }

            // Pre-execute any subqueries in WHERE clause
            if (ast.where) {
                await preExecuteSubqueries(ast.where, db);
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

            // Apply GROUP BY with aggregations (supports ROLLUP/CUBE/GROUPING SETS)
            if (ast.groupBy && (Array.isArray(ast.groupBy) ? ast.groupBy.length > 0 : ast.groupBy.type)) {
                // Determine grouping sets based on type
                let groupingSets = [];
                let allColumns = [];

                if (Array.isArray(ast.groupBy)) {
                    // Standard GROUP BY: single grouping set with all columns
                    allColumns = ast.groupBy.map(col => typeof col === 'string' ? col : col.column);
                    groupingSets = [allColumns];
                } else if (ast.groupBy.type === 'ROLLUP') {
                    // ROLLUP(a, b, c) generates: (a,b,c), (a,b), (a), ()
                    allColumns = ast.groupBy.columns.map(col => typeof col === 'string' ? col : col.column);
                    for (let i = allColumns.length; i >= 0; i--) {
                        groupingSets.push(allColumns.slice(0, i));
                    }
                } else if (ast.groupBy.type === 'CUBE') {
                    // CUBE(a, b) generates all 2^n combinations: (a,b), (a), (b), ()
                    allColumns = ast.groupBy.columns.map(col => typeof col === 'string' ? col : col.column);
                    const n = allColumns.length;
                    for (let mask = (1 << n) - 1; mask >= 0; mask--) {
                        const set = [];
                        for (let i = 0; i < n; i++) {
                            if (mask & (1 << i)) set.push(allColumns[i]);
                        }
                        groupingSets.push(set);
                    }
                } else if (ast.groupBy.type === 'GROUPING_SETS') {
                    // Explicit grouping sets
                    groupingSets = ast.groupBy.sets.map(set =>
                        set.map(col => typeof col === 'string' ? col : col.column)
                    );
                    // Collect all unique columns
                    const colSet = new Set();
                    for (const set of groupingSets) {
                        for (const col of set) colSet.add(col);
                    }
                    allColumns = [...colSet];
                }

                // Execute aggregation for each grouping set
                const allGroupedRows = [];
                for (const groupingSet of groupingSets) {
                    const groups = new Map();
                    for (const row of rows) {
                        const keyParts = groupingSet.map(col => String(row[col]));
                        const key = keyParts.join('|');
                        if (!groups.has(key)) {
                            groups.set(key, []);
                        }
                        groups.get(key).push(row);
                    }

                    // Process each group for this grouping set
                    for (const [, groupRows] of groups) {
                        const resultRow = {};
                        // Add all columns (null for rolled-up columns)
                        for (const col of allColumns) {
                            if (groupingSet.includes(col)) {
                                resultRow[col] = groupRows[0][col];
                                resultRow[`__grouping_${col}`] = 0; // Not rolled up
                            } else {
                                resultRow[col] = null;
                                resultRow[`__grouping_${col}`] = 1; // Rolled up (super-aggregate)
                            }
                        }
                        // Calculate aggregates
                        for (const col of ast.columns) {
                            if (col.type === 'aggregate') {
                                const rawName = `${col.func}(${col.arg === '*' ? '*' : (typeof col.arg === 'string' ? col.arg : col.arg.column)})`;
                                const aggValue = calculateAggregate(col.func, col.arg, groupRows);
                                resultRow[rawName] = aggValue;
                                if (col.alias && col.alias !== rawName) {
                                    resultRow[col.alias] = aggValue;
                                }
                            }
                        }
                        allGroupedRows.push(resultRow);
                    }
                }
                rows = allGroupedRows;

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
                    } else if (col.type === 'function') {
                        const alias = col.alias || `${col.func}(...)`;
                        if (rows.length > 0) resultRow[alias] = evaluateScalarFunction(col.func, col.args, rows[0], tableAliases);
                    } else if (col.type === 'arithmetic') {
                        const alias = col.alias || 'expr';
                        if (rows.length > 0) resultRow[alias] = evaluateArithmeticExpr(col.expr, rows[0], tableAliases);
                    } else if (col.type === 'literal') {
                        const alias = col.alias || 'value';
                        resultRow[alias] = col.value;
                    }
                }
                rows = [resultRow];
            }

            // Compute window functions (before ORDER BY but after all data is collected)
            const windowCols = ast.columns.filter(c => c.type === 'window');
            if (windowCols.length > 0) {
                rows = computeWindowFunctions(rows, windowCols, tableAliases);
            }

            // Apply QUALIFY (filter on window function results)
            if (ast.qualify) {
                rows = rows.filter(row => {
                    // QUALIFY can reference window function results by their alias
                    // The window function values are stored with __window_alias prefix
                    // Create a row view that maps aliases to window values
                    const rowWithWindowCols = { ...row };
                    for (const col of windowCols) {
                        const alias = col.alias || `${col.func}(...)`;
                        if (row[`__window_${alias}`] !== undefined) {
                            rowWithWindowCols[alias] = row[`__window_${alias}`];
                        }
                    }
                    return evalWhere(ast.qualify, rowWithWindowCols, tableAliases);
                });
            }

            // Apply ORDER BY first (before projection, using original column values)
            if (ast.orderBy && ast.orderBy.length > 0) {
                rows.sort((a, b) => {
                    for (const order of ast.orderBy) {
                        const col = typeof order.column === 'string' ? order.column : order.column.column;
                        const aVal = a[col];
                        const bVal = b[col];
                        const aNull = aVal == null;
                        const bNull = bVal == null;

                        // Handle NULL values
                        if (aNull || bNull) {
                            if (aNull && bNull) continue; // Both null, move to next column
                            // Determine nulls ordering: explicit setting or default (ASC=NULLS LAST, DESC=NULLS FIRST)
                            const nullsFirst = order.nullsFirst ?? order.desc;
                            if (aNull) return nullsFirst ? -1 : 1;
                            if (bNull) return nullsFirst ? 1 : -1;
                        }

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
            const hasScalarSubquery = ast.columns.some(c => c.type === 'scalar_subquery');

            if (!ast.columns.some(c => c.type === 'star') && !ast.groupBy) {
                // Use async projection if there are scalar subqueries
                const projectRow = async (row) => {
                    const result = {};
                    for (const col of ast.columns) {
                        if (col.type === 'column') {
                            const colName = typeof col.value === 'string' ? col.value : col.value.column;
                            const alias = col.alias || colName;
                            result[alias] = getColumnValue(row, col.value, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'aggregate') {
                            const aggName = col.alias || `${col.func}(${col.arg === '*' ? '*' : (typeof col.arg === 'string' ? col.arg : col.arg.column)})`;
                            result[aggName] = row[aggName];
                            if (!columnNames.includes(aggName)) columnNames.push(aggName);
                        } else if (col.type === 'function') {
                            const alias = col.alias || `${col.func}(...)`;
                            result[alias] = evaluateScalarFunction(col.func, col.args, row, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'case') {
                            const alias = col.alias || 'case';
                            result[alias] = evaluateCaseExpr(col.expr, row, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'window') {
                            const alias = col.alias || `${col.func}(...)`;
                            result[alias] = row[`__window_${alias}`];
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'arithmetic') {
                            const alias = col.alias || 'expr';
                            result[alias] = evaluateArithmeticExpr(col.expr, row, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'literal') {
                            const alias = col.alias || 'value';
                            result[alias] = col.value;
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        } else if (col.type === 'scalar_subquery') {
                            // Correlated scalar subquery - execute for each row
                            const alias = col.alias || 'subquery';
                            result[alias] = await evaluateScalarSubquery(col.subquery, row, db, tableAliases);
                            if (!columnNames.includes(alias)) columnNames.push(alias);
                        }
                    }
                    return result;
                };

                if (hasScalarSubquery) {
                    rows = await Promise.all(rows.map(projectRow));
                } else {
                    rows = rows.map(row => {
                        const result = {};
                        for (const col of ast.columns) {
                            if (col.type === 'column') {
                                const colName = typeof col.value === 'string' ? col.value : col.value.column;
                                const alias = col.alias || colName;
                                result[alias] = getColumnValue(row, col.value, tableAliases);
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            } else if (col.type === 'aggregate') {
                                const aggName = col.alias || `${col.func}(${col.arg === '*' ? '*' : (typeof col.arg === 'string' ? col.arg : col.arg.column)})`;
                                result[aggName] = row[aggName];
                                if (!columnNames.includes(aggName)) columnNames.push(aggName);
                            } else if (col.type === 'function') {
                                const alias = col.alias || `${col.func}(...)`;
                                result[alias] = evaluateScalarFunction(col.func, col.args, row, tableAliases);
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            } else if (col.type === 'case') {
                                const alias = col.alias || 'case';
                                result[alias] = evaluateCaseExpr(col.expr, row, tableAliases);
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            } else if (col.type === 'window') {
                                const alias = col.alias || `${col.func}(...)`;
                                result[alias] = row[`__window_${alias}`];
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            } else if (col.type === 'arithmetic') {
                                const alias = col.alias || 'expr';
                                result[alias] = evaluateArithmeticExpr(col.expr, row, tableAliases);
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            } else if (col.type === 'literal') {
                                const alias = col.alias || 'value';
                                result[alias] = col.value;
                                if (!columnNames.includes(alias)) columnNames.push(alias);
                            }
                        }
                        return result;
                    });
                }
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

        case 'UNION': {
            // Execute both sides recursively
            const leftResult = await executeAST(db, ast.left);
            const rightResult = await executeAST(db, ast.right);
            let rows = [...leftResult.rows, ...rightResult.rows];

            // UNION (without ALL) removes duplicates
            if (!ast.all) {
                const seen = new Set();
                rows = rows.filter(row => {
                    const key = JSON.stringify(row);
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                });
            }

            return { rows, columns: leftResult.columns };
        }

        case 'INTERSECT': {
            const leftResult = await executeAST(db, ast.left);
            const rightResult = await executeAST(db, ast.right);
            const rightKeys = new Set(rightResult.rows.map(r => JSON.stringify(r)));
            let rows = leftResult.rows.filter(row => rightKeys.has(JSON.stringify(row)));

            // INTERSECT (without ALL) removes duplicates
            if (!ast.all) {
                const seen = new Set();
                rows = rows.filter(row => {
                    const key = JSON.stringify(row);
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                });
            }

            return { rows, columns: leftResult.columns };
        }

        case 'EXCEPT': {
            const leftResult = await executeAST(db, ast.left);
            const rightResult = await executeAST(db, ast.right);
            const rightKeys = new Set(rightResult.rows.map(r => JSON.stringify(r)));
            let rows = leftResult.rows.filter(row => !rightKeys.has(JSON.stringify(row)));

            // EXCEPT (without ALL) removes duplicates
            if (!ast.all) {
                const seen = new Set();
                rows = rows.filter(row => {
                    const key = JSON.stringify(row);
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                });
            }

            return { rows, columns: leftResult.columns };
        }

        case 'WITH': {
            // Execute CTEs and create temporary tables
            const cteNames = [];
            for (const cte of ast.ctes) {
                const cteResult = await executeAST(db, cte.query);
                // Create a temporary table with CTE results
                const schema = cteResult.columns.map(col => ({ name: col, type: 'TEXT' }));
                // Add __rowId to each row for compatibility
                const rows = cteResult.rows.map((row, i) => ({ ...row, __rowId: i }));
                db.tables.set(cte.name, {
                    name: cte.name,
                    schema,
                    fragments: [],
                    deletionVector: [],
                    rowCount: rows.length,
                    nextRowId: rows.length,
                    isCTE: true
                });
                // Store rows in write buffer for compatibility
                db._writeBuffer.set(cte.name, rows);
                cteNames.push(cte.name);
            }

            try {
                // Execute main query
                const result = await executeAST(db, ast.query);
                return result;
            } finally {
                // Clean up CTE tables
                for (const name of cteNames) {
                    db.tables.delete(name);
                    db._writeBuffer.delete(name);
                }
            }
        }

        case 'EXPLAIN': {
            const plan = generateQueryPlan(ast.statement);

            if (!ast.analyze) {
                // EXPLAIN without ANALYZE - just return the plan
                return { columns: ['plan'], rows: [[JSON.stringify(plan)]] };
            }

            // EXPLAIN ANALYZE - execute and measure timing
            const startTime = performance.now();
            const result = await executeAST(db, ast.statement);
            const elapsed = performance.now() - startTime;

            const execution = {
                actualTimeMs: Math.round(elapsed * 1000) / 1000, // Round to 3 decimal places
                rowsReturned: result.rows ? result.rows.length : 0,
                rowsTotal: result.rows ? result.rows.length : 0
            };

            return {
                columns: ['plan_with_execution'],
                rows: [[JSON.stringify({ plan, execution })]]
            };
        }

        case 'PIVOT': {
            // Execute the SELECT first
            const selectResult = await executeAST(db, ast.select);
            const rows = selectResult.rows;

            // Determine grouping columns (all SELECT columns except pivot and value columns)
            const selectCols = ast.select.columns
                .filter(c => c.type === 'column')
                .map(c => typeof c.value === 'string' ? c.value : c.value.column);
            const groupCols = selectCols.filter(
                c => c !== ast.pivotColumn && c !== ast.valueColumn
            );

            // Group rows by grouping columns
            const groups = new Map();
            for (const row of rows) {
                const keyParts = groupCols.map(c => String(row[c]));
                const key = keyParts.join('|');
                if (!groups.has(key)) {
                    groups.set(key, { keyRow: row, pivotData: {} });
                }
                const pivotVal = row[ast.pivotColumn];
                const dataVal = row[ast.valueColumn];
                if (!groups.get(key).pivotData[pivotVal]) {
                    groups.get(key).pivotData[pivotVal] = [];
                }
                groups.get(key).pivotData[pivotVal].push(dataVal);
            }

            // Aggregate function helper
            const aggregate = (func, values) => {
                if (!values || values.length === 0) return 0;
                const nums = values.filter(v => v != null).map(v => Number(v));
                switch (func) {
                    case 'SUM': return nums.reduce((a, b) => a + b, 0);
                    case 'COUNT': return nums.length;
                    case 'AVG': return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
                    case 'MIN': return Math.min(...nums);
                    case 'MAX': return Math.max(...nums);
                    default: return nums[0];
                }
            };

            // Build result rows
            const resultRows = [];
            for (const [, { keyRow, pivotData }] of groups) {
                const newRow = {};
                for (const col of groupCols) {
                    newRow[col] = keyRow[col];
                }
                for (const pv of ast.pivotValues) {
                    newRow[pv] = aggregate(ast.aggFunc, pivotData[pv] || []);
                }
                resultRows.push(newRow);
            }

            return { columns: [...groupCols, ...ast.pivotValues], rows: resultRows };
        }

        case 'UNPIVOT': {
            // Execute the SELECT first
            const selectResult = await executeAST(db, ast.select);
            const rows = selectResult.rows;

            // Determine preserved columns (all SELECT columns except unpivot columns)
            const selectCols = ast.select.columns
                .filter(c => c.type === 'column')
                .map(c => typeof c.value === 'string' ? c.value : c.value.column);
            const preservedCols = selectCols.filter(c => !ast.unpivotColumns.includes(c));

            // Build result rows
            const resultRows = [];
            for (const row of rows) {
                for (const col of ast.unpivotColumns) {
                    const val = row[col];
                    if (val != null) { // Skip NULL values
                        const newRow = {};
                        for (const pc of preservedCols) {
                            newRow[pc] = row[pc];
                        }
                        newRow[ast.nameColumn] = col;
                        newRow[ast.valueColumn] = val;
                        resultRows.push(newRow);
                    }
                }
            }

            return { columns: [...preservedCols, ast.nameColumn, ast.valueColumn], rows: resultRows };
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
