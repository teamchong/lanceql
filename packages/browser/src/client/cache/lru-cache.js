/**
 * LRUCache - Page caching for Lance file reads
 */

class LRUCache {
    constructor(options = {}) {
        this.maxSize = options.maxSize ?? 50 * 1024 * 1024; // 50MB default
        this.currentSize = 0;
        this.cache = new Map(); // key -> { data, size, lastAccess }
        this._accessCounter = 0; // Monotonic counter for LRU tracking
    }

    /**
     * Get item from cache
     * @param {string} key - Cache key
     * @returns {Uint8Array|null}
     */
    get(key) {
        const entry = this.cache.get(key);
        if (entry) {
            entry.lastAccess = ++this._accessCounter;
            return entry.data;
        }
        return undefined;
    }

    delete(key) {
        const entry = this.cache.get(key);
        if (entry) {
            this.currentSize -= entry.size;
            this.cache.delete(key);
            return true;
        }
        return false;
    }

    /**
     * Put item in cache
     * @param {string} key - Cache key
     * @param {Uint8Array} data - Data to cache
     */
    put(key, data) {
        // Remove existing entry if present
        if (this.cache.has(key)) {
            this.currentSize -= this.cache.get(key).size;
            this.cache.delete(key);
        }

        // Calculate size based on data type
        let size = 0;
        if (data === null || data === undefined) {
            size = 0;
        } else if (data.byteLength !== undefined) {
            size = data.byteLength;
        } else if (typeof data === 'string') {
            size = data.length * 2; // UTF-16
        } else if (typeof data === 'object') {
            size = JSON.stringify(data).length * 2;
        } else {
            size = 8; // primitive
        }

        // Evict if needed
        while (this.currentSize + size > this.maxSize && this.cache.size > 0) {
            this._evictOldest();
        }

        // Don't cache if single item is too large
        if (size > this.maxSize) {
            return;
        }

        this.cache.set(key, {
            data,
            size,
            lastAccess: ++this._accessCounter
        });
        this.currentSize += size;
    }

    /**
     * Evict oldest entry
     */
    _evictOldest() {
        let oldestKey = null;
        let oldestTime = Infinity;

        for (const [key, entry] of this.cache) {
            if (entry.lastAccess < oldestTime) {
                oldestTime = entry.lastAccess;
                oldestKey = key;
            }
        }

        if (oldestKey) {
            this.currentSize -= this.cache.get(oldestKey).size;
            this.cache.delete(oldestKey);
        }
    }

    /**
     * Clear entire cache
     */
    clear() {
        this.cache.clear();
        this.currentSize = 0;
    }

    /**
     * Get cache stats
     */
    stats() {
        return {
            entries: this.cache.size,
            currentSize: this.currentSize,
            maxSize: this.maxSize,
            utilization: (this.currentSize / this.maxSize * 100).toFixed(1) + '%'
        };
    }
}

// Lance file format constants
const LANCE_FOOTER_SIZE = 40;
const LANCE_MAGIC = new Uint8Array([0x4C, 0x41, 0x4E, 0x43]); // "LANC"

/**
 * Chunked Lance File Reader
 * Reads Lance files from OPFS without loading entire file into memory
 */

export { LRUCache };
