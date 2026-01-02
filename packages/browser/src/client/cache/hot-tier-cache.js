/**
 * HotTierCache - Two-tier caching (hot/cold) for frequent data
 */

class HotTierCache {
    constructor(storage = null, options = {}) {
        this.storage = storage;
        this.cacheDir = options.cacheDir || '_cache';
        this.maxFileSize = options.maxFileSize || 10 * 1024 * 1024; // 10MB - cache whole file
        this.maxCacheSize = options.maxCacheSize || 500 * 1024 * 1024; // 500MB total cache
        this.enabled = options.enabled ?? true;
        this._stats = {
            hits: 0,
            misses: 0,
            bytesFromCache: 0,
            bytesFromNetwork: 0,
        };
        // In-memory cache for metadata to avoid OPFS reads on every getRange call
        this._metaCache = new Map();  // url -> { meta, fullFileData }
    }

    /**
     * Initialize the cache (lazy initialization)
     */
    async init() {
        if (this.storage) return;
        this.storage = new OPFSStorage();
        await this.storage.open();
    }

    /**
     * Get cache key from URL (hash for safe filesystem names)
     */
    _getCacheKey(url) {
        // Simple hash for URL → safe filename
        let hash = 0;
        for (let i = 0; i < url.length; i++) {
            const char = url.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash).toString(36);
    }

    /**
     * Get cache path for a URL
     */
    _getCachePath(url, suffix = '') {
        const key = this._getCacheKey(url);
        return `${this.cacheDir}/${key}${suffix}`;
    }

    /**
     * Check if a URL is cached
     * @param {string} url - Remote URL
     * @returns {Promise<{cached: boolean, meta?: object}>}
     */
    async isCached(url) {
        if (!this.enabled) return { cached: false };

        try {
            await this.init();
            const metaPath = this._getCachePath(url, '/meta.json');
            const metaData = await this.storage.load(metaPath);
            if (!metaData) return { cached: false };

            const meta = JSON.parse(new TextDecoder().decode(metaData));
            return { cached: true, meta };
        } catch (e) {
            return { cached: false };
        }
    }

    /**
     * Get or fetch a file, using cache when available
     * @param {string} url - Remote URL
     * @param {number} [fileSize] - Known file size (avoids HEAD request)
     * @returns {Promise<Uint8Array>}
     */
    async getFile(url, fileSize = null) {
        if (!this.enabled) {
            return this._fetchFile(url);
        }

        await this.init();

        // Check cache
        const { cached, meta } = await this.isCached(url);
        if (cached && meta.fullFile) {
            const dataPath = this._getCachePath(url, '/data.lance');
            const data = await this.storage.load(dataPath);
            if (data) {
                this._stats.hits++;
                this._stats.bytesFromCache += data.byteLength;
                console.log(`[HotTierCache] HIT: ${url} (${(data.byteLength / 1024).toFixed(1)} KB)`);
                return data;
            }
        }

        // Cache miss - fetch and cache
        this._stats.misses++;
        const data = await this._fetchFile(url);
        this._stats.bytesFromNetwork += data.byteLength;

        // Cache if small enough
        if (data.byteLength <= this.maxFileSize) {
            await this._cacheFile(url, data);
        }

        return data;
    }

    /**
     * Get or fetch a byte range, using cache when available
     * @param {string} url - Remote URL
     * @param {number} start - Start byte offset
     * @param {number} end - End byte offset (inclusive)
     * @param {number} [fileSize] - Total file size
     * @returns {Promise<ArrayBuffer>}
     */
    async getRange(url, start, end, fileSize = null) {
        if (!this.enabled) {
            return this._fetchRange(url, start, end);
        }

        // Fast path: check in-memory cache first (no async overhead)
        const memCached = this._metaCache.get(url);
        if (memCached?.fullFileData) {
            const data = memCached.fullFileData;
            if (data.byteLength > end) {
                this._stats.hits++;
                this._stats.bytesFromCache += (end - start + 1);
                return data.slice(start, end + 1).buffer;
            }
        }

        await this.init();

        // Check OPFS cache (only once per URL, then cache in memory)
        if (!memCached) {
            const { cached, meta } = await this.isCached(url);
            if (cached && meta.fullFile) {
                const dataPath = this._getCachePath(url, '/data.lance');
                const data = await this.storage.load(dataPath);
                if (data && data.byteLength > end) {
                    // Cache in memory for subsequent calls
                    this._metaCache.set(url, { meta, fullFileData: data });
                    this._stats.hits++;
                    this._stats.bytesFromCache += (end - start + 1);
                    return data.slice(start, end + 1).buffer;
                }
            }
            // Mark as checked even if not cached
            this._metaCache.set(url, { meta: cached ? meta : null, fullFileData: null });
        }

        // Cache miss - fetch from network
        this._stats.misses++;
        const data = await this._fetchRange(url, start, end);
        this._stats.bytesFromNetwork += data.byteLength;

        // Don't cache individual ranges to OPFS - too slow for IVF search
        // Only full files are cached (via prefetch)

        return data;
    }

    /**
     * Prefetch and cache an entire file
     * @param {string} url - Remote URL
     * @param {function} [onProgress] - Progress callback (bytesLoaded, totalBytes)
     */
    async prefetch(url, onProgress = null) {
        await this.init();

        const { cached, meta } = await this.isCached(url);
        if (cached && meta.fullFile) {
            console.log(`[HotTierCache] Already cached: ${url}`);
            return;
        }

        console.log(`[HotTierCache] Prefetching: ${url}`);
        const data = await this._fetchFile(url, onProgress);
        await this._cacheFile(url, data);
        console.log(`[HotTierCache] Cached: ${url} (${(data.byteLength / 1024 / 1024).toFixed(2)} MB)`);
    }

    /**
     * Evict a URL from cache
     */
    async evict(url) {
        await this.init();
        const cachePath = this._getCachePath(url);
        await this.storage.delete(cachePath);
        console.log(`[HotTierCache] Evicted: ${url}`);
    }

    /**
     * Clear entire cache
     */
    async clear() {
        await this.init();
        await this.storage.delete(this.cacheDir);
        this._stats = { hits: 0, misses: 0, bytesFromCache: 0, bytesFromNetwork: 0 };
        console.log(`[HotTierCache] Cleared all cache`);
    }

    /**
     * Get cache statistics
     */
    getStats() {
        const hitRate = this._stats.hits + this._stats.misses > 0
            ? (this._stats.hits / (this._stats.hits + this._stats.misses) * 100).toFixed(1)
            : 0;
        return {
            ...this._stats,
            hitRate: `${hitRate}%`,
            bytesFromCacheMB: (this._stats.bytesFromCache / 1024 / 1024).toFixed(2),
            bytesFromNetworkMB: (this._stats.bytesFromNetwork / 1024 / 1024).toFixed(2),
        };
    }

    /**
     * Fetch file from network
     * @private
     */
    async _fetchFile(url, onProgress = null) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        if (onProgress && response.headers.get('content-length')) {
            const total = parseInt(response.headers.get('content-length'));
            const reader = response.body.getReader();
            const chunks = [];
            let loaded = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                loaded += value.length;
                onProgress(loaded, total);
            }

            const result = new Uint8Array(loaded);
            let offset = 0;
            for (const chunk of chunks) {
                result.set(chunk, offset);
                offset += chunk.length;
            }
            return result;
        }

        const buffer = await response.arrayBuffer();
        return new Uint8Array(buffer);
    }

    /**
     * Fetch range from network
     * @private
     */
    async _fetchRange(url, start, end) {
        const response = await fetch(url, {
            headers: { 'Range': `bytes=${start}-${end}` }
        });
        if (!response.ok && response.status !== 206) {
            throw new Error(`HTTP error: ${response.status}`);
        }
        return response.arrayBuffer();
    }

    /**
     * Cache a full file
     * @private
     */
    async _cacheFile(url, data) {
        const metaPath = this._getCachePath(url, '/meta.json');
        const dataPath = this._getCachePath(url, '/data.lance');

        const meta = {
            url,
            size: data.byteLength,
            cachedAt: Date.now(),
            fullFile: true,
            ranges: null,
        };

        await this.storage.save(metaPath, new TextEncoder().encode(JSON.stringify(meta)));
        await this.storage.save(dataPath, data);
    }

    /**
     * Cache a byte range
     * @private
     */
    async _cacheRange(url, start, end, data, fileSize) {
        const metaPath = this._getCachePath(url, '/meta.json');
        const rangePath = this._getCachePath(url, `/ranges/${start}-${end}`);

        // Load existing meta or create new
        let meta;
        const { cached, meta: existingMeta } = await this.isCached(url);
        if (cached) {
            meta = existingMeta;
            meta.ranges = meta.ranges || [];
        } else {
            meta = {
                url,
                size: fileSize,
                cachedAt: Date.now(),
                fullFile: false,
                ranges: [],
            };
        }

        // Add this range (merge overlapping ranges for efficiency)
        meta.ranges.push({ start, end, cachedAt: Date.now() });
        meta.ranges = this._mergeRanges(meta.ranges);

        await this.storage.save(metaPath, new TextEncoder().encode(JSON.stringify(meta)));
        await this.storage.save(rangePath, data);
    }

    /**
     * Merge overlapping ranges
     * @private
     */
    _mergeRanges(ranges) {
        if (ranges.length <= 1) return ranges;

        ranges.sort((a, b) => a.start - b.start);
        const merged = [ranges[0]];

        for (let i = 1; i < ranges.length; i++) {
            const last = merged[merged.length - 1];
            const current = ranges[i];

            // Merge if overlapping or adjacent
            if (current.start <= last.end + 1) {
                last.end = Math.max(last.end, current.end);
            } else {
                merged.push(current);
            }
        }

        return merged;
    }
}

// Global hot-tier cache instance
const hotTierCache = new HotTierCache();

// Export storage and statistics for external use


export { HotTierCache, hotTierCache };
