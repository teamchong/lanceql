/**
 * WorkerStore - Key-value storage for Store API
 */

import { encryptionKeys, importEncryptionKey, encryptData, decryptData } from './encryption.js';

// Shared state (set by index.js)
let gpuTransformer = null;
let gpuTransformerPromise = null;
const embeddingCache = new Map();

export function setGPUTransformer(transformer) {
    gpuTransformer = transformer;
}

export function setGPUTransformerPromise(promise) {
    gpuTransformerPromise = promise;
}

export function getEmbeddingCache() {
    return embeddingCache;
}

export function getGPUTransformerState() {
    return gpuTransformer;
}

export class WorkerStore {
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
                setGPUTransformerPromise(gpuTransformerPromise);
            }
            gpuTransformer = await gpuTransformerPromise;
            setGPUTransformer(gpuTransformer);
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
