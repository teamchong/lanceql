/**
 * LRUCache - Page caching for Lance file reads
 */

class LRUCache {
    constructor(maxSize = 50 * 1024 * 1024) { // 50MB default
        this.maxSize = maxSize;
        this.currentSize = 0;
        this.cache = new Map(); // key -> { data, size, lastAccess }
    }

    /**
     * Get item from cache
     * @param {string} key - Cache key
     * @returns {Uint8Array|null}
     */
    get(key) {
        const entry = this.cache.get(key);
        if (entry) {
            entry.lastAccess = Date.now();
            return entry.data;
        }
        return null;
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

        const size = data.byteLength;

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
            lastAccess: Date.now()
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
