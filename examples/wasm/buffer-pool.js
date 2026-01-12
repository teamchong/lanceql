/**
 * LRU Buffer Pool
 * Manages memory usage by evicting cold data when a byte limit is reached.
 */

export class BufferPool {
    /**
     * @param {number} maxBytes - Maximum memory in bytes (default 512MB)
     */
    constructor(maxBytes = 512 * 1024 * 1024) {
        this.maxBytes = maxBytes;
        this.currentBytes = 0;
        this.cache = new Map(); // key -> Node
        this.head = null; // Most recently used
        this.tail = null; // Least recently used
    }

    /**
     * Get an item from the pool. Updates recency.
     * @param {string} key
     * @returns {any} value or undefined
     */
    get(key) {
        const node = this.cache.get(key);
        if (!node) return undefined;

        this._moveToHead(node);
        return node.value;
    }

    /**
     * Add or update an item in the pool. Evicts if necessary.
     * @param {string} key
     * @param {any} value
     * @param {number} size - Approximate size in bytes
     */
    set(key, value, size = 0) {
        const existingNode = this.cache.get(key);

        if (existingNode) {
            // Update existing
            this.currentBytes -= existingNode.size;
            this.currentBytes += size;
            existingNode.value = value;
            existingNode.size = size;
            this._moveToHead(existingNode);
        } else {
            // Add new
            const newNode = { key, value, size, prev: null, next: null };
            this.cache.set(key, newNode);
            this._addToHead(newNode);
            this.currentBytes += size;
        }

        this._evictIfNeeded();
    }

    /**
     * Check if a key exists without updating recency.
     * @param {string} key
     */
    has(key) {
        return this.cache.has(key);
    }

    /**
     * Remove an item from the pool.
     * @param {string} key
     */
    delete(key) {
        const node = this.cache.get(key);
        if (node) {
            this._removeNode(node);
            this.cache.delete(key);
            this.currentBytes -= node.size;
        }
    }

    /**
     * Clear the pool.
     */
    clear() {
        this.cache.clear();
        this.head = null;
        this.tail = null;
        this.currentBytes = 0;
    }

    _moveToHead(node) {
        if (node === this.head) return;

        this._removeNode(node);
        this._addToHead(node);
    }

    _addToHead(node) {
        node.next = this.head;
        node.prev = null;

        if (this.head) {
            this.head.prev = node;
        }
        this.head = node;

        if (!this.tail) {
            this.tail = node;
        }
    }

    _removeNode(node) {
        if (node.prev) {
            node.prev.next = node.next;
        } else {
            this.head = node.next;
        }

        if (node.next) {
            node.next.prev = node.prev;
        } else {
            this.tail = node.prev;
        }
    }

    _evictIfNeeded() {
        while (this.currentBytes > this.maxBytes && this.tail) {
            const node = this.tail;
            this._removeNode(node);
            this.cache.delete(node.key);
            this.currentBytes -= node.size;
            // console.debug(`[BufferPool] Evicted ${node.key}, freed ${node.size} bytes. Current: ${this.currentBytes}/${this.maxBytes}`);
        }
    }

    /**
     * Helper to estimate size of common objects
     * @param {any} val
     * @returns {number}
     */
    static estimateSize(val) {
        if (!val) return 0;
        if (val.byteLength) return val.byteLength; // ArrayBuffer / TypedArray
        if (Array.isArray(val)) {
            // Rough estimate for array of objects/strings
            return val.length * 100;
        }
        if (val.buffer && val.buffer.byteLength) return val.buffer.byteLength;
        // Basic object estimate
        return 1000;
    }
}
