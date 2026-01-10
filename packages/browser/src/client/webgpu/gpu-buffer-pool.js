/**
 * GPU Buffer Pool - Persistent buffer management with LRU eviction
 *
 * Reduces GPU buffer creation overhead by caching frequently-used buffers.
 * Buffers are keyed by content hash and evicted when pool exceeds maxSize.
 */

/**
 * Simple hash for buffer keys
 * @param {string} key
 * @returns {string}
 */
function hashKey(key) {
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
        hash = ((hash << 5) - hash) + key.charCodeAt(i);
        hash |= 0;
    }
    return hash.toString(36);
}

/**
 * GPU Buffer Pool with LRU eviction
 */
export class GPUBufferPool {
    /**
     * @param {GPUDevice} device - WebGPU device
     * @param {number} maxPoolSize - Maximum pool size in bytes (default 256MB)
     */
    constructor(device, maxPoolSize = 256 * 1024 * 1024) {
        this.device = device;
        this.maxPoolSize = maxPoolSize;
        this.currentSize = 0;
        this.buffers = new Map(); // key -> { buffer, size, lastAccess, usage }
        this.accessOrder = []; // LRU tracking
    }

    /**
     * Get or create a buffer for the given data.
     * @param {string} key - Unique key for this buffer (e.g., "table:column:hash")
     * @param {Float32Array|Uint32Array|Int32Array} data - Data to upload
     * @param {number} usage - GPUBufferUsage flags
     * @returns {GPUBuffer}
     */
    getOrCreate(key, data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
        const hashedKey = hashKey(key);
        const cached = this.buffers.get(hashedKey);

        if (cached) {
            // Update LRU order
            this._touch(hashedKey);

            // Check if data has changed (size mismatch = invalidate)
            if (cached.size === data.byteLength) {
                return cached.buffer;
            }

            // Size changed - invalidate and recreate
            this.invalidate(key);
        }

        // Ensure space for new buffer
        this._ensureSpace(data.byteLength);

        // Create new buffer
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage,
            mappedAtCreation: true,
        });

        // Copy data
        const arrayType = data.constructor;
        new arrayType(buffer.getMappedRange()).set(data);
        buffer.unmap();

        // Store in pool
        this.buffers.set(hashedKey, {
            buffer,
            size: data.byteLength,
            lastAccess: Date.now(),
            usage,
            originalKey: key,
        });

        this.currentSize += data.byteLength;
        this.accessOrder.push(hashedKey);

        return buffer;
    }

    /**
     * Get or create a storage buffer (most common usage).
     * @param {string} key - Unique key
     * @param {Float32Array|Uint32Array} data - Data to upload
     * @returns {GPUBuffer}
     */
    getStorageBuffer(key, data) {
        return this.getOrCreate(key, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    }

    /**
     * Get or create a uniform buffer.
     * @param {string} key - Unique key
     * @param {Float32Array|Uint32Array} data - Data to upload
     * @returns {GPUBuffer}
     */
    getUniformBuffer(key, data) {
        return this.getOrCreate(key, data, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    }

    /**
     * Check if buffer exists in pool.
     * @param {string} key
     * @returns {boolean}
     */
    has(key) {
        return this.buffers.has(hashKey(key));
    }

    /**
     * Get buffer without creating.
     * @param {string} key
     * @returns {GPUBuffer|null}
     */
    get(key) {
        const hashedKey = hashKey(key);
        const cached = this.buffers.get(hashedKey);
        if (cached) {
            this._touch(hashedKey);
            return cached.buffer;
        }
        return null;
    }

    /**
     * Invalidate (remove) a specific buffer.
     * @param {string} key
     */
    invalidate(key) {
        const hashedKey = hashKey(key);
        const cached = this.buffers.get(hashedKey);
        if (cached) {
            cached.buffer.destroy();
            this.currentSize -= cached.size;
            this.buffers.delete(hashedKey);
            this.accessOrder = this.accessOrder.filter(k => k !== hashedKey);
        }
    }

    /**
     * Invalidate all buffers matching a prefix.
     * @param {string} prefix - Key prefix to match
     */
    invalidatePrefix(prefix) {
        const toRemove = [];
        for (const [hashedKey, entry] of this.buffers) {
            if (entry.originalKey.startsWith(prefix)) {
                toRemove.push(hashedKey);
            }
        }
        for (const hashedKey of toRemove) {
            const entry = this.buffers.get(hashedKey);
            entry.buffer.destroy();
            this.currentSize -= entry.size;
            this.buffers.delete(hashedKey);
        }
        this.accessOrder = this.accessOrder.filter(k => !toRemove.includes(k));
    }

    /**
     * Clear all buffers.
     */
    clear() {
        for (const entry of this.buffers.values()) {
            entry.buffer.destroy();
        }
        this.buffers.clear();
        this.accessOrder = [];
        this.currentSize = 0;
    }

    /**
     * Get pool statistics.
     * @returns {{entries: number, currentSize: number, maxSize: number, utilization: number}}
     */
    stats() {
        return {
            entries: this.buffers.size,
            currentSize: this.currentSize,
            maxSize: this.maxPoolSize,
            utilization: this.currentSize / this.maxPoolSize,
        };
    }

    /**
     * Update LRU access order.
     * @private
     */
    _touch(hashedKey) {
        const idx = this.accessOrder.indexOf(hashedKey);
        if (idx !== -1) {
            this.accessOrder.splice(idx, 1);
        }
        this.accessOrder.push(hashedKey);

        const cached = this.buffers.get(hashedKey);
        if (cached) {
            cached.lastAccess = Date.now();
        }
    }

    /**
     * Evict LRU buffers until we have space.
     * @private
     */
    _ensureSpace(requiredBytes) {
        while (this.currentSize + requiredBytes > this.maxPoolSize && this.accessOrder.length > 0) {
            const oldest = this.accessOrder.shift();
            const entry = this.buffers.get(oldest);
            if (entry) {
                entry.buffer.destroy();
                this.currentSize -= entry.size;
                this.buffers.delete(oldest);
            }
        }
    }
}

// Singleton pool per device
const devicePools = new WeakMap();

/**
 * Get or create buffer pool for a device.
 * @param {GPUDevice} device
 * @param {number} maxSize - Max pool size in bytes
 * @returns {GPUBufferPool}
 */
export function getBufferPool(device, maxSize = 256 * 1024 * 1024) {
    let pool = devicePools.get(device);
    if (!pool) {
        pool = new GPUBufferPool(device, maxSize);
        devicePools.set(device, pool);
    }
    return pool;
}
