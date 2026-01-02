/**
 * Store - Client-side key-value store (IndexedDB + OPFS)
 */

class Store {
    /**
     * @param {string} name - Store name
     * @param {Object} options - Store options
     * @param {boolean} options.session - If true, clears data on tab close
     * @param {Function} options.getEncryptionKey - Async callback returning encryption key (CryptoKey or raw bytes)
     */
    constructor(name, options = {}) {
        this.name = name;
        this.options = options;
        this._ready = false;
        this._sessionMode = options.session || false;
        this._semanticSearchEnabled = false;
        this._getEncryptionKey = options.getEncryptionKey || null;
        this._encryptionKeyId = null;  // Sent to worker to identify key
    }

    /**
     * Initialize the store (connects to SharedWorker).
     * @returns {Promise<Store>}
     */
    async open() {
        if (this._ready) return this;

        // If encryption is enabled, derive key and send to worker
        let encryptionConfig = null;
        if (this._getEncryptionKey) {
            const key = await this._getEncryptionKey();
            this._encryptionKeyId = `${this.name}:${Date.now()}`;

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

        await workerRPC('open', {
            name: this.name,
            options: this.options,
            encryption: encryptionConfig
        });

        // Session mode cleanup
        if (this._sessionMode && typeof window !== 'undefined') {
            window.addEventListener('beforeunload', () => {
                this.clear().catch(() => {});
            });
        }

        this._ready = true;
        return this;
    }

    /**
     * Get a value by key.
     * @param {string} key
     * @returns {Promise<any>} The stored value, or undefined if not found
     */
    async get(key) {
        await this._ensureOpen();
        return workerRPC('get', { name: this.name, key });
    }

    /**
     * Set a value. Accepts any JSON-serializable value.
     * @param {string} key
     * @param {any} value
     * @returns {Promise<void>}
     */
    async set(key, value) {
        await this._ensureOpen();
        await workerRPC('set', { name: this.name, key, value });
    }

    /**
     * Delete a key.
     * @param {string} key
     * @returns {Promise<boolean>} True if key existed
     */
    async delete(key) {
        await this._ensureOpen();
        return workerRPC('delete', { name: this.name, key });
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

    /**
     * List all keys.
     * @returns {Promise<string[]>}
     */
    async keys() {
        await this._ensureOpen();
        return workerRPC('keys', { name: this.name });
    }

    /**
     * Clear all data.
     * @returns {Promise<void>}
     */
    async clear() {
        await this._ensureOpen();
        await workerRPC('clear', { name: this.name });
    }

    /**
     * Filter items in a collection.
     * @param {string} key - Collection key
     * @param {Object} query - Filter query (MongoDB-style operators)
     * @returns {Promise<Array>} Matching items
     */
    async filter(key, query = {}) {
        await this._ensureOpen();
        return workerRPC('filter', { name: this.name, key, query });
    }

    /**
     * Find first item matching query.
     * @param {string} key - Collection key
     * @param {Object} query - Filter query
     * @returns {Promise<Object|undefined>} First matching item
     */
    async find(key, query = {}) {
        await this._ensureOpen();
        return workerRPC('find', { name: this.name, key, query });
    }

    /**
     * Semantic search within a collection.
     * @param {string} key - Collection key
     * @param {string} text - Search text
     * @param {number} limit - Max results (default 10)
     * @returns {Promise<Array>} Matching items with similarity scores
     */
    async search(key, text, limit = 10) {
        await this._ensureOpen();
        return workerRPC('search', { name: this.name, key, text, limit });
    }

    /**
     * Count items in a collection, optionally filtered.
     * @param {string} key - Collection key
     * @param {Object} query - Optional filter query
     * @returns {Promise<number>}
     */
    async count(key, query = null) {
        await this._ensureOpen();
        return workerRPC('count', { name: this.name, key, query });
    }

    /**
     * Subscribe to changes (reactive updates).
     * @param {string} key - Key to watch
     * @param {Function} callback - Called with new value on changes
     * @returns {Function} Unsubscribe function
     */
    subscribe(key, callback) {
        console.warn('[Store] subscribe() not yet implemented');
        return () => {};
    }

    /**
     * Enable semantic search with WebGPU-accelerated text encoding.
     * Model is loaded and runs in the SharedWorker.
     *
     * @param {Object} options - Configuration options
     * @param {string} options.model - Model name ('minilm', 'clip', or GGUF URL)
     * @returns {Promise<Object>} Model info (dimensions, type)
     */
    async enableSemanticSearch(options = {}) {
        await this._ensureOpen();
        const result = await workerRPC('enableSemanticSearch', {
            name: this.name,
            options
        });
        if (result) {
            this._semanticSearchEnabled = true;
        }
        return result;
    }

    /**
     * Disable semantic search and free GPU resources.
     */
    async disableSemanticSearch() {
        await workerRPC('disableSemanticSearch', { name: this.name });
        this._semanticSearchEnabled = false;
    }

    /**
     * Check if semantic search is enabled.
     * @returns {boolean}
     */
    hasSemanticSearch() {
        return this._semanticSearchEnabled;
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    async _ensureOpen() {
        if (!this._ready) {
            await this.open();
        }
    }
}

/**
 * Create a new Store instance.
 *
 * @param {string} name - Store name (used as OPFS directory)
 * @param {Object} options - Store options
 * @param {boolean} options.session - If true, clears data on tab close
 * @param {Function} options.getEncryptionKey - Async callback returning encryption key
 * @returns {Promise<Store>}
 *
 * @example
 * // Persistent store
 * const store = await lanceStore('myapp');
 *
 * // Session store (clears on tab close)
 * const session = await lanceStore('temp', { session: true });
 *
 * // Encrypted store (AES-256-GCM)
 * const secure = await lanceStore('vault', {
 *     getEncryptionKey: async () => {
 *         // Return key from password, hardware token, etc.
 *         const password = await promptPassword();
 *         return password; // String, Uint8Array, ArrayBuffer, or CryptoKey
 *     }
 * });
 */
async function lanceStore(name, options = {}) {
    const store = new Store(name, options);
    await store.open();
    return store;
}

// Alias for backwards compatibility
export { lanceStore as createStore };

// Export Store class for manual instantiation
export { Store as KeyValueStore };

/**
 * SQL Parser for LocalDatabase (supports CREATE, INSERT, UPDATE, DELETE, SELECT)
 */

export { Store, lanceStore };
