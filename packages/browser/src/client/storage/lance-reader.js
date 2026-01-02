/**
 * ChunkedLanceReader - Lance file format parser with memory management
 */

class ChunkedLanceReader {
    /**
     * @param {OPFSFileReader} fileReader - OPFS file reader
     * @param {LRUCache} [pageCache] - Optional page cache (shared across readers)
     */
    constructor(fileReader, pageCache = null) {
        this.fileReader = fileReader;
        this.pageCache = pageCache || new LRUCache();
        this.footer = null;
        this.columnMetaCache = new Map(); // colIdx -> metadata
        this._cacheKey = null; // For cache key generation
    }

    /**
     * Open a Lance file from OPFS
     * @param {OPFSStorage} storage - OPFS storage instance
     * @param {string} path - File path in OPFS
     * @param {LRUCache} [pageCache] - Optional shared page cache
     * @returns {Promise<ChunkedLanceReader>}
     */
    static async open(storage, path, pageCache = null) {
        const fileReader = await storage.openFile(path);
        if (!fileReader) {
            throw new Error(`File not found: ${path}`);
        }
        const reader = new ChunkedLanceReader(fileReader, pageCache);
        reader._cacheKey = path;
        await reader._readFooter();
        return reader;
    }

    /**
     * Read and parse the Lance footer
     */
    async _readFooter() {
        const footerData = await this.fileReader.readFromEnd(LANCE_FOOTER_SIZE);

        // Verify magic bytes
        const magic = footerData.slice(36, 40);
        if (!this._arraysEqual(magic, LANCE_MAGIC)) {
            throw new Error('Invalid Lance file: magic bytes mismatch');
        }

        // Parse footer (little-endian)
        const view = new DataView(footerData.buffer, footerData.byteOffset);
        this.footer = {
            columnMetaStart: view.getBigUint64(0, true),
            columnMetaOffsetsStart: view.getBigUint64(8, true),
            globalBuffOffsetsStart: view.getBigUint64(16, true),
            numGlobalBuffers: view.getUint32(24, true),
            numColumns: view.getUint32(28, true),
            majorVersion: view.getUint16(32, true),
            minorVersion: view.getUint16(34, true),
        };

        return this.footer;
    }

    /**
     * Compare two Uint8Arrays
     */
    _arraysEqual(a, b) {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) return false;
        }
        return true;
    }

    /**
     * Get file size
     * @returns {Promise<number>}
     */
    async getSize() {
        return this.fileReader.getSize();
    }

    /**
     * Get number of columns
     * @returns {number}
     */
    getNumColumns() {
        if (!this.footer) throw new Error('Footer not loaded');
        return this.footer.numColumns;
    }

    /**
     * Get Lance format version
     * @returns {{major: number, minor: number}}
     */
    getVersion() {
        if (!this.footer) throw new Error('Footer not loaded');
        return {
            major: this.footer.majorVersion,
            minor: this.footer.minorVersion
        };
    }

    /**
     * Read column metadata offset table
     * @returns {Promise<BigUint64Array>}
     */
    async _readColumnMetaOffsets() {
        const numCols = this.footer.numColumns;
        const offsetTableSize = numCols * 8; // 8 bytes per offset
        const data = await this.fileReader.readRange(
            Number(this.footer.columnMetaOffsetsStart),
            offsetTableSize
        );
        return new BigUint64Array(data.buffer, data.byteOffset, numCols);
    }

    /**
     * Read raw column metadata bytes
     * @param {number} colIdx - Column index
     * @returns {Promise<Uint8Array>}
     */
    async readColumnMetaRaw(colIdx) {
        if (colIdx >= this.footer.numColumns) {
            throw new Error(`Column index ${colIdx} out of range (${this.footer.numColumns} columns)`);
        }

        // Check cache
        const cacheKey = `${this._cacheKey}:colmeta:${colIdx}`;
        const cached = this.pageCache.get(cacheKey);
        if (cached) return cached;

        // Read offset table
        const offsets = await this._readColumnMetaOffsets();

        // Calculate start and end
        const start = Number(this.footer.columnMetaStart) + Number(offsets[colIdx]);
        const end = colIdx < this.footer.numColumns - 1
            ? Number(this.footer.columnMetaStart) + Number(offsets[colIdx + 1])
            : Number(this.footer.columnMetaOffsetsStart);

        const data = await this.fileReader.readRange(start, end - start);

        // Cache it
        this.pageCache.put(cacheKey, data);
        return data;
    }

    /**
     * Read a specific byte range from the file
     * @param {number} offset - Start offset
     * @param {number} length - Number of bytes
     * @returns {Promise<Uint8Array>}
     */
    async readRange(offset, length) {
        // Check cache
        const cacheKey = `${this._cacheKey}:range:${offset}:${length}`;
        const cached = this.pageCache.get(cacheKey);
        if (cached) return cached;

        const data = await this.fileReader.readRange(offset, length);

        // Cache if reasonably sized
        if (length < 10 * 1024 * 1024) { // < 10MB
            this.pageCache.put(cacheKey, data);
        }
        return data;
    }

    /**
     * Get cache statistics
     * @returns {object}
     */
    getCacheStats() {
        return this.pageCache.stats();
    }

    /**
     * Close the reader and release resources
     */
    close() {
        this.fileReader.invalidate();
        this.columnMetaCache.clear();
    }
}

// =============================================================================
// Memory Manager - Global memory monitoring and management
// =============================================================================

/**
 * Global memory manager for browser environment.
 * Monitors memory usage and triggers cleanup when needed.
 */
class MemoryManager {
    constructor(options = {}) {
        this.maxHeapMB = options.maxHeapMB || 100; // Target max heap usage
        this.warningThreshold = options.warningThreshold || 0.8; // 80% warning
        this.caches = new Set(); // Registered LRU caches
        this.lastCheck = 0;
        this.checkInterval = 5000; // Check every 5 seconds
    }

    /**
     * Register a cache for memory management
     * @param {LRUCache} cache - Cache to manage
     */
    registerCache(cache) {
        this.caches.add(cache);
    }

    /**
     * Unregister a cache
     * @param {LRUCache} cache - Cache to remove
     */
    unregisterCache(cache) {
        this.caches.delete(cache);
    }

    /**
     * Get current memory usage (if available)
     * @returns {Object|null} Memory info or null if not available
     */
    getMemoryUsage() {
        if (typeof performance !== 'undefined' && performance.memory) {
            // Chrome/Chromium only
            return {
                usedHeapMB: performance.memory.usedJSHeapSize / (1024 * 1024),
                totalHeapMB: performance.memory.totalJSHeapSize / (1024 * 1024),
                limitMB: performance.memory.jsHeapSizeLimit / (1024 * 1024),
            };
        }
        return null;
    }

    /**
     * Check memory and trigger cleanup if needed
     * @returns {boolean} True if cleanup was triggered
     */
    checkAndCleanup() {
        const now = Date.now();
        if (now - this.lastCheck < this.checkInterval) {
            return false;
        }
        this.lastCheck = now;

        const memory = this.getMemoryUsage();
        if (!memory) return false;

        const usageRatio = memory.usedHeapMB / this.maxHeapMB;

        if (usageRatio > this.warningThreshold) {
            console.warn(`[MemoryManager] High memory usage: ${memory.usedHeapMB.toFixed(1)}MB / ${this.maxHeapMB}MB`);
            this.cleanup();
            return true;
        }

        return false;
    }

    /**
     * Force cleanup of all registered caches
     */
    cleanup() {
        for (const cache of this.caches) {
            // Evict 50% of entries
            const stats = cache.stats();
            const targetSize = stats.currentSize / 2;

            while (cache.currentSize > targetSize && cache.cache.size > 0) {
                cache._evictOldest();
            }
        }
    }

    /**
     * Get aggregate cache stats
     * @returns {Object} Combined stats from all caches
     */
    getCacheStats() {
        let totalEntries = 0;
        let totalSize = 0;
        let totalMaxSize = 0;

        for (const cache of this.caches) {
            const stats = cache.stats();
            totalEntries += stats.entries;
            totalSize += stats.currentSize;
            totalMaxSize += stats.maxSize;
        }

        return {
            caches: this.caches.size,
            totalEntries,
            totalSizeMB: (totalSize / (1024 * 1024)).toFixed(2),
            totalMaxSizeMB: (totalMaxSize / (1024 * 1024)).toFixed(2),
            memory: this.getMemoryUsage(),
        };
    }
}

// Global memory manager instance
const memoryManager = new MemoryManager();

/**
 * Streaming utilities for large file processing
 */
const StreamUtils = {
    /**
     * Process items in batches with memory-aware pacing
     * @param {AsyncIterable} source - Source of items
     * @param {Function} processor - Async function to process each batch
     * @param {Object} options - Options
     * @yields {any} Results from processor
     */
    async *processBatches(source, processor, options = {}) {
        const batchSize = options.batchSize || 10000;
        const pauseAfter = options.pauseAfter || 5; // Pause every N batches for GC
        let batchCount = 0;

        for await (const batch of source) {
            yield await processor(batch);
            batchCount++;

            // Periodic memory check
            if (batchCount % pauseAfter === 0) {
                memoryManager.checkAndCleanup();
                // Small delay to allow GC
                await new Promise(r => setTimeout(r, 0));
            }
        }
    },

    /**
     * Create a progress-reporting wrapper for async iterables
     * @param {AsyncIterable} source - Source iterable
     * @param {Function} onProgress - Progress callback (processed, total?)
     */
    async *withProgress(source, onProgress) {
        let processed = 0;
        for await (const item of source) {
            processed += Array.isArray(item) ? item.length : 1;
            onProgress(processed);
            yield item;
        }
    },

    /**
     * Limit memory usage by processing in chunks with explicit cleanup
     * @param {AsyncIterable} source - Source of data chunks
     * @param {number} maxChunksInFlight - Max chunks to keep in memory
     */
    async *throttle(source, maxChunksInFlight = 3) {
        const queue = [];

        for await (const chunk of source) {
            queue.push(chunk);

            if (queue.length >= maxChunksInFlight) {
                yield queue.shift();
            }
        }

        // Drain remaining
        while (queue.length > 0) {
            yield queue.shift();
        }
    },
};

// Export memory utilities


export { ChunkedLanceReader, MemoryManager };
