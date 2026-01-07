/**
 * LRUCache - Page caching for Lance file reads
 * Uses doubly-linked list for O(1) eviction
 */

class LRUCache {
    constructor(options = {}) {
        this.maxSize = options.maxSize ?? 50 * 1024 * 1024; // 50MB default
        this.currentSize = 0;
        this.cache = new Map(); // key -> node
        // Doubly-linked list: head = MRU, tail = LRU
        this._head = null;
        this._tail = null;
    }

    /**
     * Get item from cache - O(1)
     * @param {string} key - Cache key
     * @returns {*} Cached data or undefined
     */
    get(key) {
        const node = this.cache.get(key);
        if (node) {
            this._moveToHead(node);
            return node.data;
        }
        return undefined;
    }

    /**
     * Delete item from cache - O(1)
     */
    delete(key) {
        const node = this.cache.get(key);
        if (node) {
            this._removeNode(node);
            this.cache.delete(key);
            this.currentSize -= node.size;
            return true;
        }
        return false;
    }

    /**
     * Set item in cache with explicit size
     * @param {string} key - Cache key
     * @param {*} data - Data to cache
     * @param {number} size - Size in bytes (optional, auto-calculated if not provided)
     */
    set(key, data, size = null) {
        return this.put(key, data, size);
    }

    /**
     * Put item in cache - O(1) amortized
     * @param {string} key - Cache key
     * @param {*} data - Data to cache
     * @param {number} explicitSize - Optional explicit size in bytes
     */
    put(key, data, explicitSize = null) {
        // Remove existing entry if present
        const existing = this.cache.get(key);
        if (existing) {
            this._removeNode(existing);
            this.currentSize -= existing.size;
            this.cache.delete(key);
        }

        // Calculate size
        let size = explicitSize;
        if (size === null) {
            if (data === null || data === undefined) {
                size = 0;
            } else if (data.byteLength !== undefined) {
                size = data.byteLength;
            } else if (typeof data === 'string') {
                size = data.length * 2;
            } else if (typeof data === 'object') {
                size = JSON.stringify(data).length * 2;
            } else {
                size = 8;
            }
        }

        // Evict LRU entries until we have space - O(1) per eviction
        while (this.currentSize + size > this.maxSize && this._tail) {
            this._evictTail();
        }

        // Don't cache if single item is too large
        if (size > this.maxSize) {
            return;
        }

        // Create new node and add to head (MRU)
        const node = { key, data, size, prev: null, next: null };
        this._addToHead(node);
        this.cache.set(key, node);
        this.currentSize += size;
    }

    /** Add node to head of list (MRU position) - O(1) */
    _addToHead(node) {
        node.prev = null;
        node.next = this._head;
        if (this._head) {
            this._head.prev = node;
        }
        this._head = node;
        if (!this._tail) {
            this._tail = node;
        }
    }

    /** Remove node from list - O(1) */
    _removeNode(node) {
        if (node.prev) {
            node.prev.next = node.next;
        } else {
            this._head = node.next;
        }
        if (node.next) {
            node.next.prev = node.prev;
        } else {
            this._tail = node.prev;
        }
        node.prev = null;
        node.next = null;
    }

    /** Move existing node to head (MRU) - O(1) */
    _moveToHead(node) {
        if (node === this._head) return;
        this._removeNode(node);
        this._addToHead(node);
    }

    /** Evict tail node (LRU) - O(1) */
    _evictTail() {
        if (!this._tail) return;
        const node = this._tail;
        this._removeNode(node);
        this.cache.delete(node.key);
        this.currentSize -= node.size;
    }

    /**
     * Clear entire cache
     */
    clear() {
        this.cache.clear();
        this._head = null;
        this._tail = null;
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



/**
 * Chunked Lance File Reader
 * Reads Lance files from OPFS without loading entire file into memory
 */

export { LRUCache };
