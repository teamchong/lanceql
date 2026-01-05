var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __esm = (fn, res) => function __init() {
  return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
};
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);

// src/client/cache/metadata-cache.js
var MetadataCache, metadataCache;
var init_metadata_cache = __esm({
  "src/client/cache/metadata-cache.js"() {
    MetadataCache = class {
      constructor(dbName = "lanceql-cache", version = 1) {
        this.dbName = dbName;
        this.version = version;
        this.db = null;
      }
      async open() {
        if (this.db) return this.db;
        return new Promise((resolve, reject) => {
          const request = indexedDB.open(this.dbName, this.version);
          request.onerror = () => reject(request.error);
          request.onsuccess = () => {
            this.db = request.result;
            resolve(this.db);
          };
          request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains("datasets")) {
              const store = db.createObjectStore("datasets", { keyPath: "url" });
              store.createIndex("timestamp", "timestamp");
            }
          };
        });
      }
      /**
       * Get cached metadata for a dataset URL.
       * @param {string} url - Dataset URL
       * @returns {Promise<Object|null>} Cached metadata or null
       */
      async get(url) {
        try {
          const db = await this.open();
          return new Promise((resolve) => {
            const tx = db.transaction("datasets", "readonly");
            const store = tx.objectStore("datasets");
            const request = store.get(url);
            request.onsuccess = () => resolve(request.result || null);
            request.onerror = () => resolve(null);
          });
        } catch (e) {
          console.warn("[MetadataCache] Get failed:", e);
          return null;
        }
      }
      /**
       * Cache metadata for a dataset URL.
       * @param {string} url - Dataset URL
       * @param {Object} metadata - Metadata to cache (schema, columnTypes, fragments, etc.)
       */
      async set(url, metadata) {
        try {
          const db = await this.open();
          return new Promise((resolve, reject) => {
            const tx = db.transaction("datasets", "readwrite");
            const store = tx.objectStore("datasets");
            const data = {
              url,
              timestamp: Date.now(),
              ...metadata
            };
            const request = store.put(data);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
          });
        } catch (e) {
          console.warn("[MetadataCache] Set failed:", e);
        }
      }
      /**
       * Delete cached metadata for a URL.
       * @param {string} url - Dataset URL
       */
      async delete(url) {
        try {
          const db = await this.open();
          return new Promise((resolve) => {
            const tx = db.transaction("datasets", "readwrite");
            const store = tx.objectStore("datasets");
            store.delete(url);
            tx.oncomplete = () => resolve();
          });
        } catch (e) {
          console.warn("[MetadataCache] Delete failed:", e);
        }
      }
      /**
       * Clear all cached metadata.
       */
      async clear() {
        try {
          const db = await this.open();
          return new Promise((resolve) => {
            const tx = db.transaction("datasets", "readwrite");
            const store = tx.objectStore("datasets");
            store.clear();
            tx.oncomplete = () => resolve();
          });
        } catch (e) {
          console.warn("[MetadataCache] Clear failed:", e);
        }
      }
    };
    metadataCache = new MetadataCache();
  }
});

// src/client/cache/lru-cache.js
var LRUCache2, LANCE_MAGIC2;
var init_lru_cache = __esm({
  "src/client/cache/lru-cache.js"() {
    LRUCache2 = class {
      constructor(options = {}) {
        this.maxSize = options.maxSize ?? 50 * 1024 * 1024;
        this.currentSize = 0;
        this.cache = /* @__PURE__ */ new Map();
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
        return void 0;
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
        const existing = this.cache.get(key);
        if (existing) {
          this._removeNode(existing);
          this.currentSize -= existing.size;
          this.cache.delete(key);
        }
        let size = explicitSize;
        if (size === null) {
          if (data === null || data === void 0) {
            size = 0;
          } else if (data.byteLength !== void 0) {
            size = data.byteLength;
          } else if (typeof data === "string") {
            size = data.length * 2;
          } else if (typeof data === "object") {
            size = JSON.stringify(data).length * 2;
          } else {
            size = 8;
          }
        }
        while (this.currentSize + size > this.maxSize && this._tail) {
          this._evictTail();
        }
        if (size > this.maxSize) {
          return;
        }
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
          utilization: (this.currentSize / this.maxSize * 100).toFixed(1) + "%"
        };
      }
    };
    LANCE_MAGIC2 = new Uint8Array([76, 65, 78, 67]);
  }
});

// src/client/cache/hot-tier-cache.js
function getHotTierCache() {
  if (!_hotTierCache) _hotTierCache = new HotTierCache();
  return _hotTierCache;
}
var HotTierCache, _hotTierCache;
var init_hot_tier_cache = __esm({
  "src/client/cache/hot-tier-cache.js"() {
    HotTierCache = class {
      constructor(storage = null, options = {}) {
        this.storage = storage;
        this.cacheDir = options.cacheDir || "_cache";
        this.maxFileSize = options.maxFileSize || 10 * 1024 * 1024;
        this.maxCacheSize = options.maxCacheSize || 500 * 1024 * 1024;
        this.enabled = options.enabled ?? true;
        this._stats = {
          hits: 0,
          misses: 0,
          bytesFromCache: 0,
          bytesFromNetwork: 0
        };
        this._metaCache = /* @__PURE__ */ new Map();
        this._metaCacheOrder = [];
        this.maxMetaCacheEntries = options.maxMetaCacheEntries || 100;
      }
      /**
       * Add to metadata cache with LRU eviction
       * @private
       */
      _setMetaCache(url, data) {
        const existingIdx = this._metaCacheOrder.indexOf(url);
        if (existingIdx !== -1) {
          this._metaCacheOrder.splice(existingIdx, 1);
        }
        while (this._metaCacheOrder.length >= this.maxMetaCacheEntries) {
          const oldestUrl = this._metaCacheOrder.shift();
          this._metaCache.delete(oldestUrl);
        }
        this._metaCache.set(url, data);
        this._metaCacheOrder.push(url);
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
        let hash = 0;
        for (let i = 0; i < url.length; i++) {
          const char = url.charCodeAt(i);
          hash = (hash << 5) - hash + char;
          hash = hash & hash;
        }
        return Math.abs(hash).toString(36);
      }
      /**
       * Get cache path for a URL
       */
      _getCachePath(url, suffix = "") {
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
          const metaPath = this._getCachePath(url, "/meta.json");
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
        const { cached, meta } = await this.isCached(url);
        if (cached && meta.fullFile) {
          const dataPath = this._getCachePath(url, "/data.lance");
          const data2 = await this.storage.load(dataPath);
          if (data2) {
            this._stats.hits++;
            this._stats.bytesFromCache += data2.byteLength;
            console.log(`[HotTierCache] HIT: ${url} (${(data2.byteLength / 1024).toFixed(1)} KB)`);
            return data2;
          }
        }
        this._stats.misses++;
        const data = await this._fetchFile(url);
        this._stats.bytesFromNetwork += data.byteLength;
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
        const memCached = this._metaCache.get(url);
        if (memCached?.fullFileData) {
          const data2 = memCached.fullFileData;
          if (data2.byteLength > end) {
            this._stats.hits++;
            this._stats.bytesFromCache += end - start + 1;
            return data2.slice(start, end + 1).buffer;
          }
        }
        await this.init();
        if (!memCached) {
          const { cached, meta } = await this.isCached(url);
          if (cached && meta.fullFile) {
            const dataPath = this._getCachePath(url, "/data.lance");
            const data2 = await this.storage.load(dataPath);
            if (data2 && data2.byteLength > end) {
              this._setMetaCache(url, { meta, fullFileData: data2 });
              this._stats.hits++;
              this._stats.bytesFromCache += end - start + 1;
              return data2.slice(start, end + 1).buffer;
            }
          }
          this._setMetaCache(url, { meta: cached ? meta : null, fullFileData: null });
        }
        this._stats.misses++;
        const data = await this._fetchRange(url, start, end);
        this._stats.bytesFromNetwork += data.byteLength;
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
        const hitRate = this._stats.hits + this._stats.misses > 0 ? (this._stats.hits / (this._stats.hits + this._stats.misses) * 100).toFixed(1) : 0;
        return {
          ...this._stats,
          hitRate: `${hitRate}%`,
          bytesFromCacheMB: (this._stats.bytesFromCache / 1024 / 1024).toFixed(2),
          bytesFromNetworkMB: (this._stats.bytesFromNetwork / 1024 / 1024).toFixed(2)
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
        if (onProgress && response.headers.get("content-length")) {
          const total = parseInt(response.headers.get("content-length"));
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
          headers: { "Range": `bytes=${start}-${end}` }
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
        const metaPath = this._getCachePath(url, "/meta.json");
        const dataPath = this._getCachePath(url, "/data.lance");
        const meta = {
          url,
          size: data.byteLength,
          cachedAt: Date.now(),
          fullFile: true,
          ranges: null
        };
        await this.storage.save(metaPath, new TextEncoder().encode(JSON.stringify(meta)));
        await this.storage.save(dataPath, data);
      }
      /**
       * Cache a byte range
       * @private
       */
      async _cacheRange(url, start, end, data, fileSize) {
        const metaPath = this._getCachePath(url, "/meta.json");
        const rangePath = this._getCachePath(url, `/ranges/${start}-${end}`);
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
            ranges: []
          };
        }
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
          if (current.start <= last.end + 1) {
            last.end = Math.max(last.end, current.end);
          } else {
            merged.push(current);
          }
        }
        return merged;
      }
    };
    _hotTierCache = null;
  }
});

// src/client/gpu/accelerator.js
function getWebGPUAccelerator() {
  if (!_webgpuAccelerator) _webgpuAccelerator = new WebGPUAccelerator();
  return _webgpuAccelerator;
}
var WebGPUAccelerator, _webgpuAccelerator;
var init_accelerator = __esm({
  "src/client/gpu/accelerator.js"() {
    WebGPUAccelerator = class {
      constructor() {
        this.device = null;
        this.pipeline = null;
        this.available = false;
        this._initPromise = null;
      }
      /**
       * Initialize WebGPU. Call once before using.
       * @returns {Promise<boolean>} Whether WebGPU is available
       */
      async init() {
        if (this._initPromise) return this._initPromise;
        this._initPromise = this._doInit();
        return this._initPromise;
      }
      async _doInit() {
        if (!navigator.gpu) {
          console.log("[WebGPU] Not available in this browser");
          return false;
        }
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (!adapter) {
            console.log("[WebGPU] No adapter found");
            return false;
          }
          this.device = await adapter.requestDevice();
          this._createPipeline();
          this.available = true;
          console.log("[WebGPU] Initialized successfully");
          return true;
        } catch (e) {
          console.warn("[WebGPU] Init failed:", e);
          return false;
        }
      }
      _createPipeline() {
        const shaderCode = `
            struct Params {
                dim: u32,
                numVectors: u32,
            }

            @group(0) @binding(0) var<uniform> params: Params;
            @group(0) @binding(1) var<storage, read> query: array<f32>;
            @group(0) @binding(2) var<storage, read> vectors: array<f32>;
            @group(0) @binding(3) var<storage, read_write> scores: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) globalId: vec3u) {
                let idx = globalId.x;
                if (idx >= params.numVectors) {
                    return;
                }

                let dim = params.dim;
                let offset = idx * dim;

                // Compute dot product (= cosine similarity for normalized vectors)
                var dot: f32 = 0.0;
                for (var i: u32 = 0u; i < dim; i++) {
                    dot += query[i] * vectors[offset + i];
                }

                scores[idx] = dot;
            }
        `;
        const shaderModule = this.device.createShaderModule({
          code: shaderCode
        });
        this.pipeline = this.device.createComputePipeline({
          layout: "auto",
          compute: {
            module: shaderModule,
            entryPoint: "main"
          }
        });
      }
      /**
       * Batch cosine similarity using WebGPU.
       * @param {Float32Array} queryVec - Query vector (dim)
       * @param {Float32Array|Float32Array[]} vectors - Candidate vectors (flat or array of arrays)
       * @param {boolean} normalized - Whether vectors are L2-normalized
       * @param {boolean} preFlattened - If true, vectors is a flat Float32Array
       * @returns {Promise<Float32Array>} Similarity scores
       */
      async batchCosineSimilarity(queryVec, vectors, normalized = true, preFlattened = false) {
        if (!this.available || vectors.length === 0) {
          return null;
        }
        const dim = queryVec.length;
        const numVectors = preFlattened ? vectors.length / dim : vectors.length;
        const vectorsBufferSize = numVectors * dim * 4;
        const maxBufferSize = this.device.limits?.maxStorageBufferBindingSize || 134217728;
        if (vectorsBufferSize > maxBufferSize) {
          console.warn(`[WebGPU] Buffer size ${(vectorsBufferSize / 1024 / 1024).toFixed(1)}MB exceeds limit ${(maxBufferSize / 1024 / 1024).toFixed(1)}MB, falling back`);
          return null;
        }
        const paramsBuffer = this.device.createBuffer({
          size: 8,
          // 2 x u32
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const queryBuffer = this.device.createBuffer({
          size: dim * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const vectorsBuffer = this.device.createBuffer({
          size: numVectors * dim * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const scoresBuffer = this.device.createBuffer({
          size: numVectors * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const readbackBuffer = this.device.createBuffer({
          size: numVectors * 4,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([dim, numVectors]));
        this.device.queue.writeBuffer(queryBuffer, 0, queryVec);
        if (preFlattened) {
          this.device.queue.writeBuffer(vectorsBuffer, 0, vectors);
        } else {
          const flatVectors = new Float32Array(numVectors * dim);
          for (let i = 0; i < numVectors; i++) {
            flatVectors.set(vectors[i], i * dim);
          }
          this.device.queue.writeBuffer(vectorsBuffer, 0, flatVectors);
        }
        const bindGroup = this.device.createBindGroup({
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: queryBuffer } },
            { binding: 2, resource: { buffer: vectorsBuffer } },
            { binding: 3, resource: { buffer: scoresBuffer } }
          ]
        });
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(numVectors / 256));
        passEncoder.end();
        commandEncoder.copyBufferToBuffer(scoresBuffer, 0, readbackBuffer, 0, numVectors * 4);
        this.device.queue.submit([commandEncoder.finish()]);
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const results = new Float32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();
        paramsBuffer.destroy();
        queryBuffer.destroy();
        vectorsBuffer.destroy();
        scoresBuffer.destroy();
        readbackBuffer.destroy();
        return results;
      }
      /**
       * Check if WebGPU is available and initialized
       */
      isAvailable() {
        return this.available;
      }
      /**
       * Get maximum vectors that can fit in a single WebGPU batch.
       * @param {number} dim - Vector dimension
       * @returns {number} Maximum vectors per batch
       */
      getMaxVectorsPerBatch(dim) {
        if (!this.available) return 0;
        const maxBufferSize = this.device.limits?.maxStorageBufferBindingSize || 134217728;
        return Math.floor(maxBufferSize * 0.9 / (dim * 4));
      }
    };
    _webgpuAccelerator = null;
  }
});

// src/client/storage/opfs.js
var opfs_exports = {};
__export(opfs_exports, {
  OPFSFileReader: () => OPFSFileReader,
  OPFSStorage: () => OPFSStorage2,
  opfsStorage: () => opfsStorage2
});
var OPFSStorage2, OPFSFileReader, opfsStorage2;
var init_opfs = __esm({
  "src/client/storage/opfs.js"() {
    OPFSStorage2 = class {
      constructor(rootDir = "lanceql") {
        this.rootDir = rootDir;
        this.root = null;
      }
      /**
       * Get OPFS root directory, creating if needed
       */
      async getRoot() {
        if (this.root) return this.root;
        if (typeof navigator === "undefined" || !navigator.storage?.getDirectory) {
          throw new Error("OPFS not available. Requires modern browser with Origin Private File System support.");
        }
        const opfsRoot = await navigator.storage.getDirectory();
        this.root = await opfsRoot.getDirectoryHandle(this.rootDir, { create: true });
        return this.root;
      }
      async open() {
        await this.getRoot();
        return this;
      }
      /**
       * Get or create a subdirectory
       */
      async getDir(path) {
        const root = await this.getRoot();
        const parts = path.split("/").filter((p) => p);
        let current = root;
        for (const part of parts) {
          current = await current.getDirectoryHandle(part, { create: true });
        }
        return current;
      }
      /**
       * Save data to a file
       * @param {string} path - File path (e.g., 'mydb/users/frag_001.lance')
       * @param {Uint8Array} data - File data
       */
      async save(path, data) {
        const parts = path.split("/");
        const fileName = parts.pop();
        const dirPath = parts.join("/");
        const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
        const fileHandle = await dir.getFileHandle(fileName, { create: true });
        if (fileHandle.createSyncAccessHandle) {
          try {
            const accessHandle = await fileHandle.createSyncAccessHandle();
            accessHandle.truncate(0);
            accessHandle.write(data, { at: 0 });
            accessHandle.flush();
            accessHandle.close();
            return { path, size: data.byteLength };
          } catch (e) {
          }
        }
        const writable = await fileHandle.createWritable();
        await writable.write(data);
        await writable.close();
        return { path, size: data.byteLength };
      }
      /**
       * Load data from a file
       * @param {string} path - File path
       * @returns {Promise<Uint8Array|null>}
       */
      async load(path) {
        try {
          const parts = path.split("/");
          const fileName = parts.pop();
          const dirPath = parts.join("/");
          const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
          const fileHandle = await dir.getFileHandle(fileName);
          const file = await fileHandle.getFile();
          const buffer = await file.arrayBuffer();
          return new Uint8Array(buffer);
        } catch (e) {
          if (e.name === "NotFoundError") {
            return null;
          }
          throw e;
        }
      }
      /**
       * Delete a file
       * @param {string} path - File path
       */
      async delete(path) {
        try {
          const parts = path.split("/");
          const fileName = parts.pop();
          const dirPath = parts.join("/");
          const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
          await dir.removeEntry(fileName);
          return true;
        } catch (e) {
          if (e.name === "NotFoundError") {
            return false;
          }
          throw e;
        }
      }
      /**
       * List files in a directory
       * @param {string} dirPath - Directory path
       * @returns {Promise<string[]>} File names
       */
      async list(dirPath = "") {
        try {
          const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
          const files = [];
          for await (const [name, handle] of dir.entries()) {
            files.push({
              name,
              type: handle.kind
              // 'file' or 'directory'
            });
          }
          return files;
        } catch (e) {
          return [];
        }
      }
      /**
       * Check if a file exists
       * @param {string} path - File path
       */
      async exists(path) {
        try {
          const parts = path.split("/");
          const fileName = parts.pop();
          const dirPath = parts.join("/");
          const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
          await dir.getFileHandle(fileName);
          return true;
        } catch (e) {
          return false;
        }
      }
      /**
       * Delete a directory and all contents
       * @param {string} dirPath - Directory path
       */
      async deleteDir(dirPath) {
        try {
          const parts = dirPath.split("/");
          const dirName = parts.pop();
          const parentPath = parts.join("/");
          const parent = parentPath ? await this.getDir(parentPath) : await this.getRoot();
          await parent.removeEntry(dirName, { recursive: true });
          return true;
        } catch (e) {
          return false;
        }
      }
      /**
       * Read a byte range from a file without loading the entire file
       * @param {string} path - File path
       * @param {number} offset - Start byte offset
       * @param {number} length - Number of bytes to read
       * @returns {Promise<Uint8Array|null>}
       */
      async readRange(path, offset, length) {
        try {
          const parts = path.split("/");
          const fileName = parts.pop();
          const dirPath = parts.join("/");
          const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
          const fileHandle = await dir.getFileHandle(fileName);
          const file = await fileHandle.getFile();
          const blob = file.slice(offset, offset + length);
          const buffer = await blob.arrayBuffer();
          return new Uint8Array(buffer);
        } catch (e) {
          if (e.name === "NotFoundError") {
            return null;
          }
          throw e;
        }
      }
      /**
       * Get file size without loading the file
       * @param {string} path - File path
       * @returns {Promise<number|null>}
       */
      async getFileSize(path) {
        try {
          const parts = path.split("/");
          const fileName = parts.pop();
          const dirPath = parts.join("/");
          const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
          const fileHandle = await dir.getFileHandle(fileName);
          const file = await fileHandle.getFile();
          return file.size;
        } catch (e) {
          if (e.name === "NotFoundError") {
            return null;
          }
          throw e;
        }
      }
      /**
       * Open a file for chunked reading
       * @param {string} path - File path
       * @returns {Promise<OPFSFileReader|null>}
       */
      async openFile(path) {
        try {
          const parts = path.split("/");
          const fileName = parts.pop();
          const dirPath = parts.join("/");
          const dir = dirPath ? await this.getDir(dirPath) : await this.getRoot();
          const fileHandle = await dir.getFileHandle(fileName);
          return new OPFSFileReader(fileHandle);
        } catch (e) {
          if (e.name === "NotFoundError") {
            return null;
          }
          throw e;
        }
      }
      /**
       * Check if OPFS is supported in this browser
       * @returns {Promise<boolean>}
       */
      async isSupported() {
        try {
          if (typeof navigator === "undefined" || !navigator.storage?.getDirectory) {
            return false;
          }
          await navigator.storage.getDirectory();
          return true;
        } catch (e) {
          return false;
        }
      }
      /**
       * Get storage statistics
       * @returns {Promise<{fileCount: number, totalSize: number}>}
       */
      async getStats() {
        try {
          const root = await this.getRoot();
          let fileCount = 0;
          let totalSize = 0;
          async function countDir(dir) {
            for await (const [name, handle] of dir.entries()) {
              if (handle.kind === "file") {
                const file = await handle.getFile();
                fileCount++;
                totalSize += file.size;
              } else if (handle.kind === "directory") {
                await countDir(handle);
              }
            }
          }
          await countDir(root);
          return { fileCount, totalSize };
        } catch (e) {
          return { fileCount: 0, totalSize: 0 };
        }
      }
      /**
       * List all files in storage with their sizes
       * @returns {Promise<Array<{name: string, size: number, lastModified: number}>>}
       */
      async listFiles() {
        try {
          const root = await this.getRoot();
          const files = [];
          async function listDir(dir, prefix = "") {
            for await (const [name, handle] of dir.entries()) {
              if (handle.kind === "file") {
                const file = await handle.getFile();
                files.push({
                  name: prefix ? `${prefix}/${name}` : name,
                  size: file.size,
                  lastModified: file.lastModified
                });
              } else if (handle.kind === "directory") {
                await listDir(handle, prefix ? `${prefix}/${name}` : name);
              }
            }
          }
          await listDir(root);
          return files;
        } catch (e) {
          return [];
        }
      }
      /**
       * Clear all files in storage
       * @returns {Promise<number>} Number of files deleted
       */
      async clearAll() {
        try {
          const root = await this.getRoot();
          let count = 0;
          const entries = [];
          for await (const [name, handle] of root.entries()) {
            entries.push({ name, kind: handle.kind });
          }
          for (const entry of entries) {
            await root.removeEntry(entry.name, { recursive: entry.kind === "directory" });
            count++;
          }
          return count;
        } catch (e) {
          console.warn("Failed to clear OPFS:", e);
          return 0;
        }
      }
    };
    OPFSFileReader = class {
      constructor(fileHandle) {
        this.fileHandle = fileHandle;
        this._file = null;
        this._size = null;
      }
      /**
       * Get the File object (cached)
       */
      async getFile() {
        if (!this._file) {
          this._file = await this.fileHandle.getFile();
          this._size = this._file.size;
        }
        return this._file;
      }
      /**
       * Get file size
       * @returns {Promise<number>}
       */
      async getSize() {
        if (this._size === null) {
          await this.getFile();
        }
        return this._size;
      }
      /**
       * Read a byte range
       * @param {number} offset - Start byte offset
       * @param {number} length - Number of bytes to read
       * @returns {Promise<Uint8Array>}
       */
      async readRange(offset, length) {
        const file = await this.getFile();
        const blob = file.slice(offset, offset + length);
        const buffer = await blob.arrayBuffer();
        return new Uint8Array(buffer);
      }
      /**
       * Read from end of file (useful for footer)
       * @param {number} length - Number of bytes to read from end
       * @returns {Promise<Uint8Array>}
       */
      async readFromEnd(length) {
        const size = await this.getSize();
        return this.readRange(size - length, length);
      }
      /**
       * Invalidate cache (call after file is modified)
       */
      invalidate() {
        this._file = null;
        this._size = null;
      }
    };
    opfsStorage2 = new OPFSStorage2();
  }
});

// src/client/storage/lance-reader.js
var lance_reader_exports = {};
__export(lance_reader_exports, {
  ChunkedLanceReader: () => ChunkedLanceReader,
  MemoryManager: () => MemoryManager
});
var ChunkedLanceReader, MemoryManager, memoryManager;
var init_lance_reader = __esm({
  "src/client/storage/lance-reader.js"() {
    ChunkedLanceReader = class _ChunkedLanceReader {
      /**
       * @param {OPFSFileReader} fileReader - OPFS file reader
       * @param {LRUCache} [pageCache] - Optional page cache (shared across readers)
       */
      constructor(fileReader, pageCache = null) {
        this.fileReader = fileReader;
        this.pageCache = pageCache || new LRUCache();
        this.footer = null;
        this.columnMetaCache = /* @__PURE__ */ new Map();
        this._cacheKey = null;
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
        const reader = new _ChunkedLanceReader(fileReader, pageCache);
        reader._cacheKey = path;
        await reader._readFooter();
        return reader;
      }
      /**
       * Read and parse the Lance footer
       */
      async _readFooter() {
        const footerData = await this.fileReader.readFromEnd(LANCE_FOOTER_SIZE);
        const magic = footerData.slice(36, 40);
        if (!this._arraysEqual(magic, LANCE_MAGIC)) {
          throw new Error("Invalid Lance file: magic bytes mismatch");
        }
        const view = new DataView(footerData.buffer, footerData.byteOffset);
        this.footer = {
          columnMetaStart: view.getBigUint64(0, true),
          columnMetaOffsetsStart: view.getBigUint64(8, true),
          globalBuffOffsetsStart: view.getBigUint64(16, true),
          numGlobalBuffers: view.getUint32(24, true),
          numColumns: view.getUint32(28, true),
          majorVersion: view.getUint16(32, true),
          minorVersion: view.getUint16(34, true)
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
        if (!this.footer) throw new Error("Footer not loaded");
        return this.footer.numColumns;
      }
      /**
       * Get Lance format version
       * @returns {{major: number, minor: number}}
       */
      getVersion() {
        if (!this.footer) throw new Error("Footer not loaded");
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
        const offsetTableSize = numCols * 8;
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
        const cacheKey = `${this._cacheKey}:colmeta:${colIdx}`;
        const cached = this.pageCache.get(cacheKey);
        if (cached) return cached;
        const offsets = await this._readColumnMetaOffsets();
        const start = Number(this.footer.columnMetaStart) + Number(offsets[colIdx]);
        const end = colIdx < this.footer.numColumns - 1 ? Number(this.footer.columnMetaStart) + Number(offsets[colIdx + 1]) : Number(this.footer.columnMetaOffsetsStart);
        const data = await this.fileReader.readRange(start, end - start);
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
        const cacheKey = `${this._cacheKey}:range:${offset}:${length}`;
        const cached = this.pageCache.get(cacheKey);
        if (cached) return cached;
        const data = await this.fileReader.readRange(offset, length);
        if (length < 10 * 1024 * 1024) {
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
    };
    MemoryManager = class {
      constructor(options = {}) {
        this.maxHeapMB = options.maxHeapMB || 100;
        this.warningThreshold = options.warningThreshold || 0.8;
        this.caches = /* @__PURE__ */ new Set();
        this.lastCheck = 0;
        this.checkInterval = 5e3;
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
        if (typeof performance !== "undefined" && performance.memory) {
          return {
            usedHeapMB: performance.memory.usedJSHeapSize / (1024 * 1024),
            totalHeapMB: performance.memory.totalJSHeapSize / (1024 * 1024),
            limitMB: performance.memory.jsHeapSizeLimit / (1024 * 1024)
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
          memory: this.getMemoryUsage()
        };
      }
    };
    memoryManager = new MemoryManager();
  }
});

// src/client/search/ivf-manifest.js
async function findLatestManifestVersion(baseUrl) {
  const checkVersions = [1, 5, 10, 20, 50, 100];
  const checks = await Promise.all(
    checkVersions.map(async (v) => {
      try {
        const response = await fetch(`${baseUrl}/_versions/${v}.manifest`, { method: "HEAD" });
        return response.ok ? v : 0;
      } catch {
        return 0;
      }
    })
  );
  let highestFound = Math.max(...checks);
  if (highestFound === 0) return null;
  for (let v = highestFound + 1; v <= highestFound + 30; v++) {
    try {
      const response = await fetch(`${baseUrl}/_versions/${v}.manifest`, { method: "HEAD" });
      if (response.ok) highestFound = v;
      else break;
    } catch {
      break;
    }
  }
  return highestFound;
}
function parseManifestForIndex(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset);
  const chunk1Len = view.getUint32(0, true);
  const chunk1Data = bytes.slice(4, 4 + chunk1Len);
  let pos = 0;
  let indexUuid = null;
  let indexFieldId = null;
  const readVarint = (data, startPos) => {
    let result = 0, shift = 0, p = startPos;
    while (p < data.length) {
      const byte = data[p++];
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return { value: result, pos: p };
  };
  while (pos < chunk1Data.length) {
    const tagResult = readVarint(chunk1Data, pos);
    pos = tagResult.pos;
    const fieldNum = tagResult.value >> 3;
    const wireType = tagResult.value & 7;
    if (wireType === 2) {
      const lenResult = readVarint(chunk1Data, pos);
      pos = lenResult.pos;
      const content = chunk1Data.slice(pos, pos + lenResult.value);
      pos += lenResult.value;
      if (fieldNum === 1) {
        const parsed = parseIndexMetadata(content);
        if (parsed?.uuid) {
          indexUuid = parsed.uuid;
          indexFieldId = parsed.fieldId;
        }
      }
    } else if (wireType === 0) {
      pos = readVarint(chunk1Data, pos).pos;
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    }
  }
  return indexUuid ? { uuid: indexUuid, fieldId: indexFieldId } : null;
}
function parseIndexMetadata(bytes) {
  let pos = 0, uuid = null, fieldId = null;
  const readVarint = () => {
    let result = 0, shift = 0;
    while (pos < bytes.length) {
      const byte = bytes[pos++];
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return result;
  };
  while (pos < bytes.length) {
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (wireType === 2) {
      const len = readVarint();
      const content = bytes.slice(pos, pos + len);
      pos += len;
      if (fieldNum === 1) uuid = parseUuid(content);
    } else if (wireType === 0) {
      const val = readVarint();
      if (fieldNum === 2) fieldId = val;
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    }
  }
  return { uuid, fieldId };
}
function parseUuid(bytes) {
  let pos = 0;
  while (pos < bytes.length) {
    const tag = bytes[pos++];
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (wireType === 2 && fieldNum === 1) {
      const len = bytes[pos++];
      const uuidBytes = bytes.slice(pos, pos + len);
      const hex = Array.from(uuidBytes).map((b) => b.toString(16).padStart(2, "0")).join("");
      return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
    } else if (wireType === 0) {
      while (pos < bytes.length && bytes[pos++] & 128) {
      }
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    }
  }
  return null;
}
function parseIndexFile(bytes, indexInfo, IVFIndexClass) {
  const index = new IVFIndexClass();
  const ivfData = findIVFMessage(bytes);
  if (ivfData) {
    if (ivfData.centroids) {
      index.centroids = ivfData.centroids.data;
      index.numPartitions = ivfData.centroids.numPartitions;
      index.dimension = ivfData.centroids.dimension;
    }
    if (ivfData.offsets?.length > 0) index.partitionOffsets = ivfData.offsets;
    if (ivfData.lengths?.length > 0) index.partitionLengths = ivfData.lengths;
  }
  if (!index.centroids) {
    let pos = 0;
    const readVarint = () => {
      let result = 0, shift = 0;
      while (pos < bytes.length) {
        const byte = bytes[pos++];
        result |= (byte & 127) << shift;
        if ((byte & 128) === 0) break;
        shift += 7;
      }
      return result;
    };
    while (pos < bytes.length - 4) {
      const tag = readVarint();
      const wireType = tag & 7;
      if (wireType === 2) {
        const len = readVarint();
        if (len > bytes.length - pos) break;
        const content = bytes.slice(pos, pos + len);
        pos += len;
        if (len > 100 && len < 1e8) {
          const centroids = tryParseCentroids(content);
          if (centroids) {
            index.centroids = centroids.data;
            index.numPartitions = centroids.numPartitions;
            index.dimension = centroids.dimension;
          }
        }
      } else if (wireType === 0) {
        readVarint();
      } else if (wireType === 5) {
        pos += 4;
      } else if (wireType === 1) {
        pos += 8;
      }
    }
  }
  return index.centroids ? index : null;
}
function findIVFMessage(bytes) {
  let pos = 0, offsets = [], lengths = [], centroids = null;
  const readVarint = () => {
    let result = 0, shift = 0;
    while (pos < bytes.length) {
      const byte = bytes[pos++];
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return result;
  };
  while (pos < bytes.length - 4) {
    const startPos = pos;
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (wireType === 2) {
      const len = readVarint();
      if (len > bytes.length - pos || len < 0) {
        pos = startPos + 1;
        continue;
      }
      const content = bytes.slice(pos, pos + len);
      pos += len;
      if (fieldNum === 2 && len % 8 === 0 && len > 0) {
        const view = new DataView(content.buffer, content.byteOffset, len);
        for (let i = 0; i < len / 8; i++) {
          offsets.push(Number(view.getBigUint64(i * 8, true)));
        }
      } else if (fieldNum === 3) {
        if (len % 4 === 0 && len > 0) {
          const view = new DataView(content.buffer, content.byteOffset, len);
          for (let i = 0; i < len / 4; i++) {
            lengths.push(view.getUint32(i * 4, true));
          }
        } else {
          let lpos = 0;
          while (lpos < content.length) {
            let val = 0, shift = 0;
            while (lpos < content.length) {
              const byte = content[lpos++];
              val |= (byte & 127) << shift;
              if ((byte & 128) === 0) break;
              shift += 7;
            }
            lengths.push(val);
          }
        }
      } else if (fieldNum === 4) {
        centroids = tryParseCentroids(content);
      } else if (len > 100) {
        const nested = findIVFMessage(content);
        if (nested?.centroids || nested?.offsets?.length > 0) {
          if (nested.centroids && !centroids) centroids = nested.centroids;
          if (nested.offsets?.length > offsets.length) offsets = nested.offsets;
          if (nested.lengths?.length > lengths.length) lengths = nested.lengths;
        }
      }
    } else if (wireType === 0) {
      readVarint();
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    } else {
      pos = startPos + 1;
    }
  }
  return centroids || offsets.length > 0 || lengths.length > 0 ? { centroids, offsets, lengths } : null;
}
function tryParseCentroids(bytes) {
  let pos = 0, shape = [], dataBytes = null, dataType = 2;
  const readVarint = () => {
    let result = 0, shift = 0;
    while (pos < bytes.length) {
      const byte = bytes[pos++];
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return result;
  };
  while (pos < bytes.length) {
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (wireType === 0) {
      const val = readVarint();
      if (fieldNum === 1) dataType = val;
    } else if (wireType === 2) {
      const len = readVarint();
      const content = bytes.slice(pos, pos + len);
      pos += len;
      if (fieldNum === 2) {
        let shapePos = 0;
        while (shapePos < content.length) {
          let val = 0, shift = 0;
          while (shapePos < content.length) {
            const byte = content[shapePos++];
            val |= (byte & 127) << shift;
            if ((byte & 128) === 0) break;
            shift += 7;
          }
          shape.push(val);
        }
      } else if (fieldNum === 3) {
        dataBytes = content;
      }
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    }
  }
  if (shape.length >= 2 && dataBytes && dataType === 2) {
    const numPartitions = shape[0];
    const dimension = shape[1];
    if (dataBytes.length === numPartitions * dimension * 4) {
      return {
        data: new Float32Array(dataBytes.buffer, dataBytes.byteOffset, numPartitions * dimension),
        numPartitions,
        dimension
      };
    }
  }
  return null;
}
var init_ivf_manifest = __esm({
  "src/client/search/ivf-manifest.js"() {
  }
});

// src/client/search/ivf-auxiliary.js
async function loadAuxiliaryMetadata(index) {
  let headResp;
  try {
    headResp = await fetch(index.auxiliaryUrl, { method: "HEAD" });
  } catch {
    return;
  }
  if (!headResp.ok) return;
  const fileSize = parseInt(headResp.headers.get("content-length"));
  if (!fileSize) return;
  const footerResp = await fetch(index.auxiliaryUrl, {
    headers: { "Range": `bytes=${fileSize - 40}-${fileSize - 1}` }
  });
  if (!footerResp.ok) return;
  const footer = new Uint8Array(await footerResp.arrayBuffer());
  const view = new DataView(footer.buffer, footer.byteOffset);
  const colMetaStart = Number(view.getBigUint64(0, true));
  const colMetaOffsetsStart = Number(view.getBigUint64(8, true));
  const globalBuffOffsetsStart = Number(view.getBigUint64(16, true));
  const numGlobalBuffers = view.getUint32(24, true);
  const magic = new TextDecoder().decode(footer.slice(36, 40));
  if (magic !== "LANC") return;
  const gboSize = numGlobalBuffers * 16;
  const gboResp = await fetch(index.auxiliaryUrl, {
    headers: { "Range": `bytes=${globalBuffOffsetsStart}-${globalBuffOffsetsStart + gboSize - 1}` }
  });
  if (!gboResp.ok) return;
  const gboData = new Uint8Array(await gboResp.arrayBuffer());
  const gboView = new DataView(gboData.buffer, gboData.byteOffset);
  const buffers = [];
  for (let i = 0; i < numGlobalBuffers; i++) {
    const offset = Number(gboView.getBigUint64(i * 16, true));
    const length = Number(gboView.getBigUint64(i * 16 + 8, true));
    buffers.push({ offset, length });
  }
  if (buffers.length < 2) return;
  index._auxBuffers = buffers;
  index._auxFileSize = fileSize;
  const colMetaOffResp = await fetch(index.auxiliaryUrl, {
    headers: { "Range": `bytes=${colMetaOffsetsStart}-${globalBuffOffsetsStart - 1}` }
  });
  if (!colMetaOffResp.ok) return;
  const colMetaOffData = new Uint8Array(await colMetaOffResp.arrayBuffer());
  if (colMetaOffData.length >= 32) {
    const colView = new DataView(colMetaOffData.buffer, colMetaOffData.byteOffset);
    const col0Pos = Number(colView.getBigUint64(0, true));
    const col0Len = Number(colView.getBigUint64(8, true));
    const col0MetaResp = await fetch(index.auxiliaryUrl, {
      headers: { "Range": `bytes=${col0Pos}-${col0Pos + col0Len - 1}` }
    });
    if (col0MetaResp.ok) {
      const col0Meta = new Uint8Array(await col0MetaResp.arrayBuffer());
      parseColumnMetaForPartitions(index, col0Meta);
    }
  }
}
function parseColumnMetaForPartitions(index, bytes) {
  let pos = 0;
  const pages = [];
  const readVarint = () => {
    let result = 0, shift = 0;
    while (pos < bytes.length) {
      const byte = bytes[pos++];
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return result;
  };
  while (pos < bytes.length) {
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (wireType === 2) {
      const len = readVarint();
      if (len > bytes.length - pos) break;
      const content = bytes.slice(pos, pos + len);
      pos += len;
      if (fieldNum === 2) {
        const page = parsePageInfo(content);
        if (page) pages.push(page);
      }
    } else if (wireType === 0) {
      readVarint();
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    }
  }
  index._columnPages = pages;
}
function parsePageInfo(bytes) {
  let pos = 0;
  let numRows = 0;
  const bufferOffsets = [];
  const bufferSizes = [];
  const readVarint = () => {
    let result = 0, shift = 0;
    while (pos < bytes.length) {
      const byte = bytes[pos++];
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return result;
  };
  while (pos < bytes.length) {
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (wireType === 0) {
      const val = readVarint();
      if (fieldNum === 3) numRows = val;
    } else if (wireType === 2) {
      const len = readVarint();
      const content = bytes.slice(pos, pos + len);
      pos += len;
      if (fieldNum === 1) {
        let p = 0;
        while (p < content.length) {
          let val = 0n, shift = 0n;
          while (p < content.length) {
            const b = content[p++];
            val |= BigInt(b & 127) << shift;
            if ((b & 128) === 0) break;
            shift += 7n;
          }
          bufferOffsets.push(Number(val));
        }
      }
      if (fieldNum === 2) {
        let p = 0;
        while (p < content.length) {
          let val = 0n, shift = 0n;
          while (p < content.length) {
            const b = content[p++];
            val |= BigInt(b & 127) << shift;
            if ((b & 128) === 0) break;
            shift += 7n;
          }
          bufferSizes.push(Number(val));
        }
      }
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    }
  }
  return { numRows, bufferOffsets, bufferSizes };
}
function parseAuxiliaryPartitionInfo(index, bytes) {
  let pos = 0;
  const readVarint = () => {
    let result = 0, shift = 0;
    while (pos < bytes.length) {
      const byte = bytes[pos++];
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return result;
  };
  while (pos < bytes.length - 4) {
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (wireType === 2) {
      const len = readVarint();
      if (len > bytes.length - pos) break;
      const content = bytes.slice(pos, pos + len);
      pos += len;
      if (fieldNum === 2 && len > 100 && len < 2e3) {
        const offsets = [];
        let innerPos = 0;
        while (innerPos < content.length) {
          let val = 0, shift = 0;
          while (innerPos < content.length) {
            const byte = content[innerPos++];
            val |= (byte & 127) << shift;
            if ((byte & 128) === 0) break;
            shift += 7;
          }
          offsets.push(val);
        }
        if (offsets.length === index.numPartitions) {
          index.partitionOffsets = offsets;
        }
      } else if (fieldNum === 3 && len > 100 && len < 2e3) {
        const lengths = [];
        let innerPos = 0;
        while (innerPos < content.length) {
          let val = 0, shift = 0;
          while (innerPos < content.length) {
            const byte = content[innerPos++];
            val |= (byte & 127) << shift;
            if ((byte & 128) === 0) break;
            shift += 7;
          }
          lengths.push(val);
        }
        if (lengths.length === index.numPartitions) {
          index.partitionLengths = lengths;
        }
      }
    } else if (wireType === 0) {
      readVarint();
    } else if (wireType === 1) {
      pos += 8;
    } else if (wireType === 5) {
      pos += 4;
    } else {
      break;
    }
  }
}
var init_ivf_auxiliary = __esm({
  "src/client/search/ivf-auxiliary.js"() {
  }
});

// src/client/search/ivf-partitions.js
async function loadPartitionIndex(index) {
  const url = `${index.datasetBaseUrl}/ivf_vectors.bin`;
  index.partitionVectorsUrl = url;
  const headerResp = await fetch(url, { headers: { "Range": "bytes=0-2055" } });
  if (!headerResp.ok) return;
  const headerData = await headerResp.arrayBuffer();
  const bigOffsets = new BigUint64Array(headerData);
  index.partitionOffsets = Array.from(bigOffsets, (n) => Number(n));
  index.hasPartitionIndex = true;
}
async function fetchPartitionData(index, partitionIndices, dim = 384, onProgress = null) {
  if (!index.hasPartitionIndex || !index.partitionVectorsUrl) return null;
  let totalBytesToFetch = 0;
  let bytesLoaded = 0;
  const uncachedPartitions = [];
  const cachedResults = /* @__PURE__ */ new Map();
  if (!index._partitionCache) {
    index._partitionCache = new LRUCache2({ maxSize: PARTITION_CACHE_SIZE });
  }
  for (const p of partitionIndices) {
    const cached = index._partitionCache.get(p);
    if (cached !== void 0) {
      cachedResults.set(p, cached);
    } else {
      uncachedPartitions.push(p);
      totalBytesToFetch += index.partitionOffsets[p + 1] - index.partitionOffsets[p];
    }
  }
  if (uncachedPartitions.length === 0) {
    return assembleResults(partitionIndices, cachedResults, dim, onProgress);
  }
  if (!index._fetchStats) {
    index._fetchStats = {
      concurrency: 6,
      recentLatencies: [],
      // Rolling window of last 10 batches
      minConcurrency: 2,
      maxConcurrency: 12
    };
  }
  const stats = index._fetchStats;
  for (let i = 0; i < uncachedPartitions.length; i += stats.concurrency) {
    const batch = uncachedPartitions.slice(i, i + stats.concurrency);
    const batchStart = performance.now();
    const results = await Promise.all(batch.map(async (p) => {
      const startOffset = index.partitionOffsets[p];
      const endOffset = index.partitionOffsets[p + 1];
      const byteSize = endOffset - startOffset;
      try {
        const resp = await fetch(index.partitionVectorsUrl, {
          headers: { "Range": `bytes=${startOffset}-${endOffset - 1}` }
        });
        if (!resp.ok) return { p, rowIds: [], vectors: [] };
        const data = await resp.arrayBuffer();
        const view = new DataView(data);
        const rowCount = view.getUint32(0, true);
        const rowIdsEnd = 4 + rowCount * 4;
        const rowIds = new Uint32Array(data.slice(4, rowIdsEnd));
        const vectorsFlat = new Float32Array(data.slice(rowIdsEnd));
        bytesLoaded += byteSize;
        if (onProgress) onProgress(bytesLoaded, totalBytesToFetch);
        return { p, rowIds: Array.from(rowIds), vectors: vectorsFlat, numVectors: rowCount };
      } catch {
        return { p, rowIds: [], vectors: [] };
      }
    }));
    for (const result of results) {
      const data = {
        rowIds: result.rowIds,
        vectors: result.vectors,
        numVectors: result.numVectors ?? result.rowIds.length
      };
      const size = result.rowIds.length * 4 + (result.vectors.byteLength || result.vectors.length * 4);
      index._partitionCache.set(result.p, data, size);
      cachedResults.set(result.p, data);
    }
    const batchLatency = performance.now() - batchStart;
    stats.recentLatencies.push(batchLatency);
    if (stats.recentLatencies.length > 10) stats.recentLatencies.shift();
    const avgLatency = stats.recentLatencies.reduce((a, b) => a + b, 0) / stats.recentLatencies.length;
    if (avgLatency < 50 && stats.concurrency < stats.maxConcurrency) {
      stats.concurrency++;
    } else if (avgLatency > 200 && stats.concurrency > stats.minConcurrency) {
      stats.concurrency--;
    }
  }
  return assembleResults(partitionIndices, cachedResults, dim, onProgress);
}
function assembleResults(partitionIndices, cachedResults, dim, onProgress) {
  let totalRowIds = 0;
  let totalVectorElements = 0;
  for (const p of partitionIndices) {
    const result = cachedResults.get(p);
    if (result) {
      totalRowIds += result.rowIds.length;
      totalVectorElements += result.vectors.length;
    }
  }
  const allRowIds = new Array(totalRowIds);
  const allVectors = new Float32Array(totalVectorElements);
  let rowIdOffset = 0;
  let vectorOffset = 0;
  for (const p of partitionIndices) {
    const result = cachedResults.get(p);
    if (result) {
      for (let i = 0; i < result.rowIds.length; i++) {
        allRowIds[rowIdOffset++] = result.rowIds[i];
      }
      allVectors.set(result.vectors, vectorOffset);
      vectorOffset += result.vectors.length;
    }
  }
  if (onProgress) onProgress(100, 100);
  return { rowIds: allRowIds, vectors: allVectors, preFlattened: true };
}
async function prefetchAllRowIds(index) {
  if (!index.auxiliaryUrl || !index._auxBufferOffsets) return;
  if (index._rowIdCacheReady) return;
  const totalRows = index.partitionLengths.reduce((a, b) => a + b, 0);
  if (totalRows === 0) return;
  const dataStart = index._auxBufferOffsets[1];
  const totalBytes = totalRows * 8;
  try {
    const resp = await fetch(index.auxiliaryUrl, {
      headers: { "Range": `bytes=${dataStart}-${dataStart + totalBytes - 1}` }
    });
    if (!resp.ok) return;
    const data = new Uint8Array(await resp.arrayBuffer());
    const view = new DataView(data.buffer, data.byteOffset);
    index._rowIdCache = /* @__PURE__ */ new Map();
    let globalRowIdx = 0;
    for (let p = 0; p < index.partitionLengths.length; p++) {
      const numRows = index.partitionLengths[p];
      const partitionRows2 = [];
      for (let i = 0; i < numRows; i++) {
        const rowId = Number(view.getBigUint64(globalRowIdx * 8, true));
        partitionRows2.push({
          fragId: Math.floor(rowId / 4294967296),
          rowOffset: rowId % 4294967296
        });
        globalRowIdx++;
      }
      index._rowIdCache.set(p, partitionRows2);
    }
    index._rowIdCacheReady = true;
  } catch {
  }
}
async function fetchPartitionRowIds(index, partitionIndices) {
  if (index._rowIdCacheReady && index._rowIdCache) {
    const results2 = [];
    for (const p of partitionIndices) {
      const cached = index._rowIdCache.get(p);
      if (cached) {
        for (const row of cached) {
          results2.push({ ...row, partition: p });
        }
      }
    }
    return results2;
  }
  if (!index.auxiliaryUrl || !index._auxBufferOffsets) return null;
  const rowRanges = [];
  for (const p of partitionIndices) {
    if (p < index.partitionOffsets.length) {
      rowRanges.push({
        partition: p,
        startRow: index.partitionOffsets[p],
        numRows: index.partitionLengths[p]
      });
    }
  }
  if (rowRanges.length === 0) return [];
  const results = [];
  const dataStart = index._auxBufferOffsets[1];
  for (const range of rowRanges) {
    const byteStart = dataStart + range.startRow * 8;
    const byteEnd = byteStart + range.numRows * 8 - 1;
    try {
      const resp = await fetch(index.auxiliaryUrl, {
        headers: { "Range": `bytes=${byteStart}-${byteEnd}` }
      });
      if (!resp.ok) continue;
      const data = new Uint8Array(await resp.arrayBuffer());
      const view = new DataView(data.buffer, data.byteOffset);
      for (let i = 0; i < range.numRows; i++) {
        const rowId = Number(view.getBigUint64(i * 8, true));
        results.push({
          fragId: Math.floor(rowId / 4294967296),
          rowOffset: rowId % 4294967296,
          partition: range.partition
        });
      }
    } catch {
    }
  }
  return results;
}
function getPartitionRowCount(index, partitionIndices) {
  let total = 0;
  for (const p of partitionIndices) {
    if (p < index.partitionLengths.length) {
      total += index.partitionLengths[p];
    }
  }
  return total;
}
var PARTITION_CACHE_SIZE;
var init_ivf_partitions = __esm({
  "src/client/search/ivf-partitions.js"() {
    init_lru_cache();
    PARTITION_CACHE_SIZE = 50 * 1024 * 1024;
  }
});

// src/client/search/ivf-index.js
function quickselectTopK(arr, k) {
  if (k >= arr.length) return arr;
  if (k <= 0) return [];
  let left = 0;
  let right = arr.length - 1;
  while (left < right) {
    const mid = left + right >> 1;
    if (arr[mid].score > arr[left].score) swap(arr, left, mid);
    if (arr[right].score > arr[left].score) swap(arr, left, right);
    if (arr[mid].score > arr[right].score) swap(arr, mid, right);
    const pivot = arr[right].score;
    let i = left;
    for (let j = left; j < right; j++) {
      if (arr[j].score >= pivot) {
        swap(arr, i, j);
        i++;
      }
    }
    swap(arr, i, right);
    if (i === k - 1) break;
    if (i < k - 1) left = i + 1;
    else right = i - 1;
  }
  return arr.slice(0, k);
}
function swap(arr, i, j) {
  const tmp = arr[i];
  arr[i] = arr[j];
  arr[j] = tmp;
}
var IVFIndex;
var init_ivf_index = __esm({
  "src/client/search/ivf-index.js"() {
    init_ivf_manifest();
    init_ivf_auxiliary();
    init_ivf_partitions();
    IVFIndex = class _IVFIndex {
      constructor() {
        this.centroids = null;
        this.numPartitions = 0;
        this.dimension = 0;
        this.partitionOffsets = [];
        this.partitionLengths = [];
        this.metricType = "cosine";
        this.partitionIndexUrl = null;
        this.partitionStarts = null;
        this.hasPartitionIndex = false;
        this._rowIdCache = null;
        this._rowIdCacheReady = false;
        this._accessCounts = /* @__PURE__ */ new Map();
      }
      static async tryLoad(datasetBaseUrl) {
        if (!datasetBaseUrl) return null;
        try {
          const manifestVersion = await findLatestManifestVersion(datasetBaseUrl);
          if (!manifestVersion) return null;
          const manifestUrl = `${datasetBaseUrl}/_versions/${manifestVersion}.manifest`;
          const manifestResp = await fetch(manifestUrl);
          if (!manifestResp.ok) return null;
          const manifestData = await manifestResp.arrayBuffer();
          const indexInfo = parseManifestForIndex(new Uint8Array(manifestData));
          if (!indexInfo?.uuid) return null;
          const indexUrl = `${datasetBaseUrl}/_indices/${indexInfo.uuid}/index.idx`;
          const indexResp = await fetch(indexUrl);
          if (!indexResp.ok) return null;
          const indexData = await indexResp.arrayBuffer();
          const index = parseIndexFile(new Uint8Array(indexData), indexInfo, _IVFIndex);
          if (!index) return null;
          index.auxiliaryUrl = `${datasetBaseUrl}/_indices/${indexInfo.uuid}/auxiliary.idx`;
          index.datasetBaseUrl = datasetBaseUrl;
          try {
            await loadAuxiliaryMetadata(index);
          } catch {
          }
          try {
            await loadPartitionIndex(index);
          } catch {
          }
          try {
            await prefetchAllRowIds(index);
          } catch {
          }
          return index;
        } catch {
          return null;
        }
      }
      async _loadPartitionIndex() {
        return loadPartitionIndex(this);
      }
      fetchPartitionData(partitionIndices, dim = 384, onProgress = null) {
        return fetchPartitionData(this, partitionIndices, dim, onProgress);
      }
      async _loadAuxiliaryMetadata() {
        return loadAuxiliaryMetadata(this);
      }
      _parseColumnMetaForPartitions(bytes) {
        return parseColumnMetaForPartitions(this, bytes);
      }
      _parseAuxiliaryPartitionInfo(bytes) {
        return parseAuxiliaryPartitionInfo(this, bytes);
      }
      async prefetchAllRowIds() {
        return prefetchAllRowIds(this);
      }
      fetchPartitionRowIds(partitionIndices) {
        return fetchPartitionRowIds(this, partitionIndices);
      }
      getPartitionRowCount(partitionIndices) {
        return getPartitionRowCount(this, partitionIndices);
      }
      findNearestPartitions(queryVec, nprobe = 10) {
        if (!this.centroids || queryVec.length !== this.dimension) return [];
        nprobe = Math.min(nprobe, this.numPartitions);
        const distances = new Array(this.numPartitions);
        let normA = 0;
        for (let i = 0; i < this.dimension; i++) {
          normA += queryVec[i] * queryVec[i];
        }
        const sqrtNormA = Math.sqrt(normA);
        for (let p = 0; p < this.numPartitions; p++) {
          const start = p * this.dimension;
          let dot = 0, normB = 0;
          for (let i = 0; i < this.dimension; i++) {
            const b = this.centroids[start + i];
            dot += queryVec[i] * b;
            normB += b * b;
          }
          const denom = sqrtNormA * Math.sqrt(normB);
          distances[p] = { idx: p, score: denom === 0 ? 0 : dot / denom };
        }
        const topK = quickselectTopK(distances, nprobe);
        const partitionIndices = topK.map((d) => d.idx);
        for (const idx of partitionIndices) {
          this._accessCounts.set(idx, (this._accessCounts.get(idx) || 0) + 1);
        }
        return partitionIndices;
      }
      /**
       * Prefetch frequently accessed partitions into cache.
       * Call after several searches to warm the cache with hot partitions.
       * @param {number} topN - Number of top partitions to prefetch (default 10)
       * @param {number} minAccesses - Minimum access count to be considered hot (default 3)
       */
      async prefetchHotPartitions(topN = 10, minAccesses = 3) {
        if (!this._accessCounts || this._accessCounts.size === 0) return;
        const hotPartitions = [...this._accessCounts.entries()].filter(([_, count]) => count >= minAccesses).sort((a, b) => b[1] - a[1]).slice(0, topN).map(([p]) => p);
        if (hotPartitions.length === 0) return;
        await this.fetchPartitionData(hotPartitions, this.dimension);
      }
      /**
       * Get access statistics for debugging/monitoring.
       */
      getAccessStats() {
        return {
          totalPartitions: this.numPartitions,
          accessedPartitions: this._accessCounts.size,
          topPartitions: [...this._accessCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)
        };
      }
    };
  }
});

// src/client/lance/remote-file-meta.js
async function tryLoadSchema(file) {
  const match = file.url.match(/^(.+\.lance)\/data\/.+\.lance$/);
  if (!match) {
    return;
  }
  file._datasetBaseUrl = match[1];
  try {
    const manifestUrl = `${file._datasetBaseUrl}/_versions/1.manifest`;
    const response = await fetch(manifestUrl);
    if (!response.ok) {
      return;
    }
    const manifestData = await response.arrayBuffer();
    file._schema = parseManifest(new Uint8Array(manifestData));
  } catch (e) {
  }
}
function parseManifest(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset);
  const chunk1Len = view.getUint32(0, true);
  const chunk2Start = 4 + chunk1Len;
  let protoData;
  if (chunk2Start + 4 < bytes.length) {
    const chunk2Len = view.getUint32(chunk2Start, true);
    if (chunk2Len > 0 && chunk2Start + 4 + chunk2Len <= bytes.length) {
      protoData = bytes.slice(chunk2Start + 4, chunk2Start + 4 + chunk2Len);
    } else {
      protoData = bytes.slice(4, 4 + chunk1Len);
    }
  } else {
    protoData = bytes.slice(4, 4 + chunk1Len);
  }
  let pos = 0;
  const fields = [];
  const readVarint = () => {
    let result = 0;
    let shift = 0;
    while (pos < protoData.length) {
      const byte = protoData[pos++];
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return result;
  };
  while (pos < protoData.length) {
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (fieldNum === 1 && wireType === 2) {
      const fieldLen = readVarint();
      const fieldEnd = pos + fieldLen;
      let name = null;
      let id = null;
      let logicalType = null;
      while (pos < fieldEnd) {
        const fTag = readVarint();
        const fNum = fTag >> 3;
        const fWire = fTag & 7;
        if (fWire === 0) {
          const val = readVarint();
          if (fNum === 3) id = val;
        } else if (fWire === 2) {
          const len = readVarint();
          const content = protoData.slice(pos, pos + len);
          pos += len;
          if (fNum === 2) {
            name = new TextDecoder().decode(content);
          } else if (fNum === 5) {
            logicalType = new TextDecoder().decode(content);
          }
        } else if (fWire === 5) {
          pos += 4;
        } else if (fWire === 1) {
          pos += 8;
        }
      }
      if (name) {
        fields.push({ name, id, type: logicalType });
      }
    } else {
      if (wireType === 0) {
        readVarint();
      } else if (wireType === 2) {
        const len = readVarint();
        pos += len;
      } else if (wireType === 5) {
        pos += 4;
      } else if (wireType === 1) {
        pos += 8;
      }
    }
  }
  return fields;
}
function getColumnNames(file) {
  if (file._schema && file._schema.length > 0) {
    return file._schema.map((f) => f.name);
  }
  return Array.from({ length: file._numColumns }, (_, i) => `column_${i}`);
}
async function detectColumnTypes(file) {
  if (file._columnTypes) {
    return file._columnTypes;
  }
  const types = [];
  if (file._schema && file._schema.length > 0) {
    for (let c = 0; c < file._numColumns; c++) {
      const schemaField = file._schema[c];
      const schemaType = schemaField?.type?.toLowerCase() || "";
      const schemaName = schemaField?.name?.toLowerCase() || "";
      let type = "unknown";
      const isEmbeddingName = schemaName.includes("embedding") || schemaName.includes("vector") || schemaName.includes("emb") || schemaName === "vec";
      if (schemaType.includes("utf8") || schemaType.includes("string") || schemaType.includes("large_utf8")) {
        type = "string";
      } else if (schemaType.includes("fixed_size_list") || schemaType.includes("vector") || isEmbeddingName) {
        type = "vector";
      } else if (schemaType.includes("int64") || schemaType === "int64") {
        type = "int64";
      } else if (schemaType.includes("int32") || schemaType === "int32") {
        type = "int32";
      } else if (schemaType.includes("int16") || schemaType === "int16") {
        type = "int16";
      } else if (schemaType.includes("int8") || schemaType === "int8") {
        type = "int8";
      } else if (schemaType.includes("float64") || schemaType.includes("double")) {
        type = "float64";
      } else if (schemaType.includes("float32") || schemaType.includes("float") && !schemaType.includes("64")) {
        type = "float32";
      } else if (schemaType.includes("bool")) {
        type = "bool";
      }
      types.push(type);
    }
    if (types.some((t) => t !== "unknown")) {
      file._columnTypes = types;
      return types;
    }
    types.length = 0;
  }
  for (let c = 0; c < file._numColumns; c++) {
    let type = "unknown";
    const colName = file.columnNames[c]?.toLowerCase() || "";
    const isEmbeddingName = colName.includes("embedding") || colName.includes("vector") || colName.includes("emb") || colName === "vec";
    try {
      await file.readStringAt(c, 0);
      type = "string";
      types.push(type);
      continue;
    } catch (e) {
    }
    try {
      const entry = await file.getColumnOffsetEntry(c);
      if (entry.len > 0) {
        const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const bytes = new Uint8Array(colMeta);
        const info = file._parseColumnMeta(bytes);
        if (info.rows > 0 && info.size > 0) {
          const bytesPerRow = info.size / info.rows;
          if (isEmbeddingName && bytesPerRow >= 4) {
            type = "vector";
          } else if (bytesPerRow === 8) {
            type = "int64";
          } else if (bytesPerRow === 4) {
            try {
              const data = await file.readInt32AtIndices(c, [0]);
              if (data.length > 0) {
                const val = data[0];
                if (val >= -1e6 && val <= 1e6 && Number.isInteger(val)) {
                  type = "int32";
                } else {
                  type = "float32";
                }
              }
            } catch (e) {
              type = "float32";
            }
          } else if (bytesPerRow > 8 && bytesPerRow % 4 === 0) {
            type = "vector";
          } else if (bytesPerRow === 2) {
            type = "int16";
          } else if (bytesPerRow === 1) {
            type = "int8";
          }
        }
      }
    } catch (e) {
    }
    types.push(type);
  }
  file._columnTypes = types;
  return types;
}
var init_remote_file_meta = __esm({
  "src/client/lance/remote-file-meta.js"() {
  }
});

// src/client/lance/remote-file-proto.js
function parseColumnMeta(bytes) {
  let pos = 0;
  const pages = [];
  let totalRows = 0;
  const readVarint = () => {
    let result = 0n;
    let shift = 0n;
    while (pos < bytes.length) {
      const byte = bytes[pos++];
      result |= BigInt(byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7n;
    }
    return Number(result);
  };
  while (pos < bytes.length) {
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (fieldNum === 2 && wireType === 2) {
      const pageLen = readVarint();
      const pageEnd = pos + pageLen;
      const pageOffsets = [];
      const pageSizes = [];
      let pageRows = 0;
      while (pos < pageEnd) {
        const pageTag = readVarint();
        const pageField = pageTag >> 3;
        const pageWire = pageTag & 7;
        if (pageField === 1 && pageWire === 2) {
          const packedLen = readVarint();
          const packedEnd = pos + packedLen;
          while (pos < packedEnd) {
            pageOffsets.push(readVarint());
          }
        } else if (pageField === 2 && pageWire === 2) {
          const packedLen = readVarint();
          const packedEnd = pos + packedLen;
          while (pos < packedEnd) {
            pageSizes.push(readVarint());
          }
        } else if (pageField === 3 && pageWire === 0) {
          pageRows = readVarint();
        } else {
          if (pageWire === 0) readVarint();
          else if (pageWire === 2) {
            const skipLen = readVarint();
            pos += skipLen;
          } else if (pageWire === 5) pos += 4;
          else if (pageWire === 1) pos += 8;
        }
      }
      pages.push({
        offsets: pageOffsets,
        sizes: pageSizes,
        rows: pageRows
      });
      totalRows += pageRows;
    } else {
      if (wireType === 0) readVarint();
      else if (wireType === 2) {
        const skipLen = readVarint();
        pos += skipLen;
      } else if (wireType === 5) pos += 4;
      else if (wireType === 1) pos += 8;
    }
  }
  const firstPage = pages[0] || { offsets: [], sizes: [], rows: 0 };
  const bufferOffsets = firstPage.offsets;
  const bufferSizes = firstPage.sizes;
  let totalSize = 0;
  for (const page of pages) {
    const dataIdx = page.sizes.length > 1 ? 1 : 0;
    totalSize += page.sizes[dataIdx] || 0;
  }
  const dataBufferIdx = bufferOffsets.length > 1 ? 1 : 0;
  const nullBitmapIdx = bufferOffsets.length > 1 ? 0 : -1;
  return {
    offset: bufferOffsets[dataBufferIdx] || 0,
    size: pages.length > 1 ? totalSize : bufferSizes[dataBufferIdx] || 0,
    rows: totalRows,
    nullBitmapOffset: nullBitmapIdx >= 0 ? bufferOffsets[nullBitmapIdx] : null,
    nullBitmapSize: nullBitmapIdx >= 0 ? bufferSizes[nullBitmapIdx] : null,
    bufferOffsets,
    bufferSizes,
    pages
  };
}
function parseStringColumnMeta(bytes) {
  const pages = [];
  let pos = 0;
  const readVarint = () => {
    let result = 0;
    let shift = 0;
    while (pos < bytes.length) {
      const byte = bytes[pos++];
      result |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    return result;
  };
  while (pos < bytes.length) {
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (fieldNum === 2 && wireType === 2) {
      const pageLen = readVarint();
      const pageEnd = pos + pageLen;
      let bufferOffsets = [0, 0];
      let bufferSizes = [0, 0];
      let rows = 0;
      while (pos < pageEnd) {
        const pageTag = readVarint();
        const pageField = pageTag >> 3;
        const pageWire = pageTag & 7;
        if (pageField === 1 && pageWire === 2) {
          const packedLen = readVarint();
          const packedEnd = pos + packedLen;
          let idx = 0;
          while (pos < packedEnd && idx < 2) {
            bufferOffsets[idx++] = readVarint();
          }
          pos = packedEnd;
        } else if (pageField === 2 && pageWire === 2) {
          const packedLen = readVarint();
          const packedEnd = pos + packedLen;
          let idx = 0;
          while (pos < packedEnd && idx < 2) {
            bufferSizes[idx++] = readVarint();
          }
          pos = packedEnd;
        } else if (pageField === 3 && pageWire === 0) {
          rows = readVarint();
        } else if (pageField === 4 && pageWire === 2) {
          const skipLen = readVarint();
          pos += skipLen;
        } else {
          if (pageWire === 0) readVarint();
          else if (pageWire === 2) {
            const skipLen = readVarint();
            pos += skipLen;
          } else if (pageWire === 5) pos += 4;
          else if (pageWire === 1) pos += 8;
        }
      }
      pages.push({
        offsetsStart: bufferOffsets[0],
        offsetsSize: bufferSizes[0],
        dataStart: bufferOffsets[1],
        dataSize: bufferSizes[1],
        rows
      });
    } else {
      if (wireType === 0) {
        readVarint();
      } else if (wireType === 2) {
        const skipLen = readVarint();
        pos += skipLen;
      } else if (wireType === 5) {
        pos += 4;
      } else if (wireType === 1) {
        pos += 8;
      }
    }
  }
  const firstPage = pages[0] || { offsetsStart: 0, offsetsSize: 0, dataStart: 0, dataSize: 0, rows: 0 };
  return {
    ...firstPage,
    pages
  };
}
function batchIndices(indices, valueSize, gapThreshold = 1024) {
  if (indices.length === 0) return [];
  const sorted = [...indices].map((v, i) => ({ idx: v, origPos: i }));
  sorted.sort((a, b) => a.idx - b.idx);
  const batches = [];
  let batchStart = 0;
  for (let i = 1; i <= sorted.length; i++) {
    const endBatch = i === sorted.length || (sorted[i].idx - sorted[i - 1].idx) * valueSize > gapThreshold;
    if (endBatch) {
      batches.push({
        startIdx: sorted[batchStart].idx,
        endIdx: sorted[i - 1].idx,
        items: sorted.slice(batchStart, i)
      });
      batchStart = i;
    }
  }
  return batches;
}
var init_remote_file_proto = __esm({
  "src/client/lance/remote-file-proto.js"() {
  }
});

// src/client/lance/remote-file-numeric.js
async function readInt64AtIndices(file, colIdx, indices) {
  if (indices.length === 0) return new BigInt64Array(0);
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = file._parseColumnMeta(new Uint8Array(colMeta));
  const results = new BigInt64Array(indices.length);
  const valueSize = 8;
  const batches = batchIndices(indices, valueSize);
  await Promise.all(batches.map(async (batch) => {
    const startOffset = info.offset + batch.startIdx * valueSize;
    const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
    const data = await file.fetchRange(startOffset, endOffset);
    const view = new DataView(data);
    for (const item of batch.items) {
      const localOffset = (item.idx - batch.startIdx) * valueSize;
      results[item.origPos] = view.getBigInt64(localOffset, true);
    }
  }));
  return results;
}
async function readFloat64AtIndices(file, colIdx, indices) {
  if (indices.length === 0) return new Float64Array(0);
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = file._parseColumnMeta(new Uint8Array(colMeta));
  const results = new Float64Array(indices.length);
  const valueSize = 8;
  const batches = batchIndices(indices, valueSize);
  await Promise.all(batches.map(async (batch) => {
    const startOffset = info.offset + batch.startIdx * valueSize;
    const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
    const data = await file.fetchRange(startOffset, endOffset);
    const view = new DataView(data);
    for (const item of batch.items) {
      const localOffset = (item.idx - batch.startIdx) * valueSize;
      results[item.origPos] = view.getFloat64(localOffset, true);
    }
  }));
  return results;
}
async function readInt32AtIndices(file, colIdx, indices) {
  if (indices.length === 0) return new Int32Array(0);
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = file._parseColumnMeta(new Uint8Array(colMeta));
  const results = new Int32Array(indices.length);
  const valueSize = 4;
  const batches = batchIndices(indices, valueSize);
  await Promise.all(batches.map(async (batch) => {
    const startOffset = info.offset + batch.startIdx * valueSize;
    const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
    const data = await file.fetchRange(startOffset, endOffset);
    const view = new DataView(data);
    for (const item of batch.items) {
      const localOffset = (item.idx - batch.startIdx) * valueSize;
      results[item.origPos] = view.getInt32(localOffset, true);
    }
  }));
  return results;
}
async function readFloat32AtIndices(file, colIdx, indices) {
  if (indices.length === 0) return new Float32Array(0);
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = file._parseColumnMeta(new Uint8Array(colMeta));
  const results = new Float32Array(indices.length);
  const valueSize = 4;
  const batches = batchIndices(indices, valueSize);
  await Promise.all(batches.map(async (batch) => {
    const startOffset = info.offset + batch.startIdx * valueSize;
    const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
    const data = await file.fetchRange(startOffset, endOffset);
    const view = new DataView(data);
    for (const item of batch.items) {
      const localOffset = (item.idx - batch.startIdx) * valueSize;
      results[item.origPos] = view.getFloat32(localOffset, true);
    }
  }));
  return results;
}
async function readInt16AtIndices(file, colIdx, indices) {
  if (indices.length === 0) return new Int16Array(0);
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = file._parseColumnMeta(new Uint8Array(colMeta));
  const results = new Int16Array(indices.length);
  const valueSize = 2;
  const batches = batchIndices(indices, valueSize);
  await Promise.all(batches.map(async (batch) => {
    const startOffset = info.offset + batch.startIdx * valueSize;
    const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
    const data = await file.fetchRange(startOffset, endOffset);
    const view = new DataView(data);
    for (const item of batch.items) {
      const localOffset = (item.idx - batch.startIdx) * valueSize;
      results[item.origPos] = view.getInt16(localOffset, true);
    }
  }));
  return results;
}
async function readUint8AtIndices(file, colIdx, indices) {
  if (indices.length === 0) return new Uint8Array(0);
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = file._parseColumnMeta(new Uint8Array(colMeta));
  const results = new Uint8Array(indices.length);
  const valueSize = 1;
  const batches = batchIndices(indices, valueSize);
  await Promise.all(batches.map(async (batch) => {
    const startOffset = info.offset + batch.startIdx * valueSize;
    const endOffset = info.offset + (batch.endIdx + 1) * valueSize - 1;
    const data = await file.fetchRange(startOffset, endOffset);
    const bytes = new Uint8Array(data);
    for (const item of batch.items) {
      const localOffset = item.idx - batch.startIdx;
      results[item.origPos] = bytes[localOffset];
    }
  }));
  return results;
}
async function readBoolAtIndices(file, colIdx, indices) {
  if (indices.length === 0) return new Uint8Array(0);
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = file._parseColumnMeta(new Uint8Array(colMeta));
  const results = new Uint8Array(indices.length);
  const byteIndices = indices.map((i) => Math.floor(i / 8));
  const uniqueBytes = [...new Set(byteIndices)].sort((a, b) => a - b);
  if (uniqueBytes.length === 0) return results;
  const startByte = uniqueBytes[0];
  const endByte = uniqueBytes[uniqueBytes.length - 1];
  const startOffset = info.offset + startByte;
  const endOffset = info.offset + endByte;
  const data = await file.fetchRange(startOffset, endOffset);
  const bytes = new Uint8Array(data);
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    const byteIdx = Math.floor(idx / 8);
    const bitIdx = idx % 8;
    const localByteIdx = byteIdx - startByte;
    if (localByteIdx >= 0 && localByteIdx < bytes.length) {
      results[i] = bytes[localByteIdx] >> bitIdx & 1;
    }
  }
  return results;
}
var init_remote_file_numeric = __esm({
  "src/client/lance/remote-file-numeric.js"() {
    init_remote_file_proto();
  }
});

// src/client/lance/remote-file-string.js
async function readStringAt(file, colIdx, rowIdx) {
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = parseStringColumnMeta(new Uint8Array(colMeta));
  if (info.offsetsSize === 0 || info.dataSize === 0) {
    throw new Error(`Not a string column - offsetsSize=${info.offsetsSize}, dataSize=${info.dataSize}`);
  }
  const bytesPerOffset = info.offsetsSize / info.rows;
  if (bytesPerOffset !== 4 && bytesPerOffset !== 8) {
    throw new Error(`Not a string column - bytesPerOffset=${bytesPerOffset}, expected 4 or 8`);
  }
  if (rowIdx >= info.rows) return "";
  const offsetSize = bytesPerOffset;
  const offsetStart = info.offsetsStart + rowIdx * offsetSize;
  const offsetData = await file.fetchRange(offsetStart, offsetStart + offsetSize * 2 - 1);
  const offsetView = new DataView(offsetData);
  let strStart, strEnd;
  if (offsetSize === 4) {
    strStart = offsetView.getUint32(0, true);
    strEnd = offsetView.getUint32(4, true);
  } else {
    strStart = Number(offsetView.getBigUint64(0, true));
    strEnd = Number(offsetView.getBigUint64(8, true));
  }
  if (strEnd <= strStart) return "";
  const strLen = strEnd - strStart;
  const strData = await file.fetchRange(
    info.dataStart + strStart,
    info.dataStart + strEnd - 1
  );
  return new TextDecoder().decode(strData);
}
async function readStringsAtIndices(file, colIdx, indices) {
  if (indices.length === 0) return [];
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = parseStringColumnMeta(new Uint8Array(colMeta));
  if (!info.pages || info.pages.length === 0) {
    return indices.map(() => "");
  }
  const results = new Array(indices.length).fill("");
  let pageRowStart = 0;
  const pageIndex = [];
  for (const page of info.pages) {
    if (page.offsetsSize === 0 || page.dataSize === 0 || page.rows === 0) {
      pageRowStart += page.rows;
      continue;
    }
    pageIndex.push({
      start: pageRowStart,
      end: pageRowStart + page.rows,
      page
    });
    pageRowStart += page.rows;
  }
  const pageGroups = /* @__PURE__ */ new Map();
  for (let i = 0; i < indices.length; i++) {
    const rowIdx = indices[i];
    for (let p = 0; p < pageIndex.length; p++) {
      const pi = pageIndex[p];
      if (rowIdx >= pi.start && rowIdx < pi.end) {
        if (!pageGroups.has(p)) {
          pageGroups.set(p, []);
        }
        pageGroups.get(p).push({
          globalIdx: rowIdx,
          localIdx: rowIdx - pi.start,
          resultIdx: i
        });
        break;
      }
    }
  }
  for (const [pageNum, items] of pageGroups) {
    const pi = pageIndex[pageNum];
    const page = pi.page;
    const offsetSize = page.offsetsSize / page.rows;
    if (offsetSize !== 4 && offsetSize !== 8) continue;
    items.sort((a, b) => a.localIdx - b.localIdx);
    const offsetBatches = [];
    let batchStart = 0;
    for (let i = 1; i <= items.length; i++) {
      if (i === items.length || items[i].localIdx - items[i - 1].localIdx > 100) {
        offsetBatches.push(items.slice(batchStart, i));
        batchStart = i;
      }
    }
    const stringRanges = [];
    await Promise.all(offsetBatches.map(async (batch) => {
      const minIdx = batch[0].localIdx;
      const maxIdx = batch[batch.length - 1].localIdx;
      const fetchStartIdx = minIdx > 0 ? minIdx - 1 : 0;
      const fetchEndIdx = maxIdx;
      const startOffset = page.offsetsStart + fetchStartIdx * offsetSize;
      const endOffset = page.offsetsStart + (fetchEndIdx + 1) * offsetSize - 1;
      const data = await file.fetchRange(startOffset, endOffset);
      const view = new DataView(data);
      for (const item of batch) {
        const dataIdx = item.localIdx - fetchStartIdx;
        let strStart, strEnd;
        if (offsetSize === 4) {
          strEnd = view.getUint32(dataIdx * 4, true);
          strStart = item.localIdx === 0 ? 0 : view.getUint32((dataIdx - 1) * 4, true);
        } else {
          strEnd = Number(view.getBigUint64(dataIdx * 8, true));
          strStart = item.localIdx === 0 ? 0 : Number(view.getBigUint64((dataIdx - 1) * 8, true));
        }
        if (strEnd > strStart) {
          stringRanges.push({
            start: strStart,
            end: strEnd,
            resultIdx: item.resultIdx,
            dataStart: page.dataStart
          });
        }
      }
    }));
    if (stringRanges.length > 0) {
      stringRanges.sort((a, b) => a.start - b.start);
      const dataBatches = [];
      let dbStart = 0;
      for (let i = 1; i <= stringRanges.length; i++) {
        if (i === stringRanges.length || stringRanges[i].start - stringRanges[i - 1].end > 4096) {
          dataBatches.push({
            rangeStart: stringRanges[dbStart].start,
            rangeEnd: stringRanges[i - 1].end,
            items: stringRanges.slice(dbStart, i),
            dataStart: stringRanges[dbStart].dataStart
          });
          dbStart = i;
        }
      }
      await Promise.all(dataBatches.map(async (batch) => {
        const data = await file.fetchRange(
          batch.dataStart + batch.rangeStart,
          batch.dataStart + batch.rangeEnd - 1
        );
        const bytes = new Uint8Array(data);
        for (const item of batch.items) {
          const localStart = item.start - batch.rangeStart;
          const len = item.end - item.start;
          const strBytes = bytes.slice(localStart, localStart + len);
          results[item.resultIdx] = new TextDecoder().decode(strBytes);
        }
      }));
    }
  }
  return results;
}
var init_remote_file_string = __esm({
  "src/client/lance/remote-file-string.js"() {
    init_remote_file_proto();
  }
});

// src/client/lance/remote-file-vector.js
async function getVectorInfo(file, colIdx) {
  const entry = await file.getColumnOffsetEntry(colIdx);
  if (entry.len === 0) return { rows: 0, dimension: 0 };
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = file._parseColumnMeta(new Uint8Array(colMeta));
  if (info.rows === 0) return { rows: 0, dimension: 0 };
  let dimension = 0;
  if (info.pages && info.pages.length > 0) {
    const firstPage = info.pages[0];
    const dataIdx = firstPage.sizes.length > 1 ? 1 : 0;
    const pageSize = firstPage.sizes[dataIdx] || 0;
    const pageRows = firstPage.rows || 0;
    if (pageRows > 0 && pageSize > 0) {
      dimension = Math.floor(pageSize / (pageRows * 4));
    }
  } else if (info.size > 0) {
    dimension = Math.floor(info.size / (info.rows * 4));
  }
  return { rows: info.rows, dimension };
}
async function readVectorAt(file, colIdx, rowIdx) {
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = file._parseColumnMeta(new Uint8Array(colMeta));
  if (info.rows === 0) return new Float32Array(0);
  if (rowIdx >= info.rows) return new Float32Array(0);
  const dim = Math.floor(info.size / (info.rows * 4));
  if (dim === 0) return new Float32Array(0);
  const vecStart = info.offset + rowIdx * dim * 4;
  const vecEnd = vecStart + dim * 4 - 1;
  const data = await file.fetchRange(vecStart, vecEnd);
  return new Float32Array(data);
}
async function readVectorsAtIndices(file, colIdx, indices) {
  if (indices.length === 0) return [];
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const info = file._parseColumnMeta(new Uint8Array(colMeta));
  if (info.rows === 0) return indices.map(() => new Float32Array(0));
  const dim = Math.floor(info.size / (info.rows * 4));
  if (dim === 0) return indices.map(() => new Float32Array(0));
  const vecSize = dim * 4;
  const results = new Array(indices.length);
  const batches = batchIndices(indices, vecSize, vecSize * 50);
  const BATCH_PARALLEL = 6;
  for (let i = 0; i < batches.length; i += BATCH_PARALLEL) {
    const batchGroup = batches.slice(i, i + BATCH_PARALLEL);
    await Promise.all(batchGroup.map(async (batch) => {
      try {
        const startOffset = info.offset + batch.startIdx * vecSize;
        const endOffset = info.offset + (batch.endIdx + 1) * vecSize - 1;
        const data = await file.fetchRange(startOffset, endOffset);
        for (const item of batch.items) {
          const localOffset = (item.idx - batch.startIdx) * vecSize;
          results[item.origPos] = new Float32Array(
            data.slice(localOffset, localOffset + vecSize)
          );
        }
      } catch (e) {
        for (const item of batch.items) {
          results[item.origPos] = new Float32Array(0);
        }
      }
    }));
  }
  return results;
}
function cosineSimilarity(vecA, vecB) {
  if (vecA.length !== vecB.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
async function vectorSearch(file, colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
  const { nprobe = 10 } = options;
  const info = await getVectorInfo(file, colIdx);
  if (info.dimension === 0 || info.dimension !== queryVec.length) {
    throw new Error(`Dimension mismatch: query=${queryVec.length}, column=${info.dimension}`);
  }
  if (!file.hasIndex()) {
    throw new Error("No IVF index found. Vector search requires an IVF index for efficient querying.");
  }
  if (file._ivfIndex.dimension !== queryVec.length) {
    throw new Error(`Query dimension (${queryVec.length}) does not match index dimension (${file._ivfIndex.dimension}).`);
  }
  return await vectorSearchWithIndex(file, colIdx, queryVec, topK, nprobe, onProgress);
}
async function vectorSearchWithIndex(file, colIdx, queryVec, topK, nprobe, onProgress) {
  if (onProgress) onProgress(0, 100);
  const partitions = file._ivfIndex.findNearestPartitions(queryVec, nprobe);
  const rowIdMappings = await file._ivfIndex.fetchPartitionRowIds(partitions);
  if (rowIdMappings && rowIdMappings.length > 0) {
    return await searchWithRowIdMappings(file, colIdx, queryVec, topK, rowIdMappings, onProgress);
  }
  throw new Error("Failed to fetch row IDs from IVF index. Dataset may be missing auxiliary.idx or ivf_partitions.bin.");
}
async function searchWithRowIdMappings(file, colIdx, queryVec, topK, rowIdMappings, onProgress) {
  const dim = queryVec.length;
  const byFragment = /* @__PURE__ */ new Map();
  for (const mapping of rowIdMappings) {
    if (!byFragment.has(mapping.fragId)) {
      byFragment.set(mapping.fragId, []);
    }
    byFragment.get(mapping.fragId).push(mapping.rowOffset);
  }
  const allVectors = [];
  const allIndices = [];
  let processed = 0;
  const total = rowIdMappings.length;
  for (const [fragId, offsets] of byFragment) {
    if (onProgress) onProgress(processed, total);
    const vectors = await readVectorsAtIndices(file, colIdx, offsets);
    for (let i = 0; i < offsets.length; i++) {
      const vec = vectors[i];
      if (vec && vec.length === dim) {
        allVectors.push(vec);
        allIndices.push(fragId * 5e4 + offsets[i]);
      }
      processed++;
    }
  }
  let scores;
  const accelerator = getWebGPUAccelerator();
  if (accelerator.isAvailable()) {
    scores = await accelerator.batchCosineSimilarity(queryVec, allVectors, true);
  }
  if (!scores) {
    scores = file.lanceql.batchCosineSimilarity(queryVec, allVectors, true);
  }
  const topResults = [];
  for (let i = 0; i < scores.length; i++) {
    const score = scores[i];
    const idx = allIndices[i];
    if (topResults.length < topK) {
      topResults.push({ idx, score });
      topResults.sort((a, b) => b.score - a.score);
    } else if (score > topResults[topK - 1].score) {
      topResults[topK - 1] = { idx, score };
      topResults.sort((a, b) => b.score - a.score);
    }
  }
  if (onProgress) onProgress(total, total);
  return {
    indices: topResults.map((r) => r.idx),
    scores: topResults.map((r) => r.score),
    usedIndex: true,
    searchedRows: allVectors.length
  };
}
async function readVectorColumn(file, colIdx) {
  const entry = await file.getColumnOffsetEntry(colIdx);
  const colMeta = await file.fetchRange(entry.pos, entry.pos + entry.len - 1);
  const metaInfo = file._parseColumnMeta(new Uint8Array(colMeta));
  if (!metaInfo.pages || metaInfo.pages.length === 0 || metaInfo.rows === 0) {
    return new Float32Array(0);
  }
  const firstPage = metaInfo.pages[0];
  const dataIdx = firstPage.sizes.length > 1 ? 1 : 0;
  const firstPageSize = firstPage.sizes[dataIdx] || 0;
  const firstPageRows = firstPage.rows || 0;
  if (firstPageRows === 0 || firstPageSize === 0) {
    return new Float32Array(0);
  }
  const dim = Math.floor(firstPageSize / (firstPageRows * 4));
  if (dim === 0) {
    return new Float32Array(0);
  }
  const totalRows = metaInfo.rows;
  const result = new Float32Array(totalRows * dim);
  const pagePromises = metaInfo.pages.map(async (page, pageIdx) => {
    const pageDataIdx = page.sizes.length > 1 ? 1 : 0;
    const pageOffset = page.offsets[pageDataIdx] || 0;
    const pageSize = page.sizes[pageDataIdx] || 0;
    if (pageSize === 0) return { pageIdx, data: new Float32Array(0), rows: 0 };
    const data = await file.fetchRange(pageOffset, pageOffset + pageSize - 1);
    const floatData = new Float32Array(data);
    return {
      pageIdx,
      data: floatData,
      rows: page.rows
    };
  });
  const pageResults = await Promise.all(pagePromises);
  let offset = 0;
  for (const pageResult of pageResults.sort((a, b) => a.pageIdx - b.pageIdx)) {
    result.set(pageResult.data, offset);
    offset += pageResult.rows * dim;
  }
  return result;
}
async function readRows(file, { offset = 0, limit = 50, columns = null } = {}) {
  const colIndices = columns || Array.from({ length: file._numColumns }, (_, i) => i);
  const totalRows = await file.getRowCount(0);
  const actualOffset = Math.min(offset, totalRows);
  const actualLimit = Math.min(limit, totalRows - actualOffset);
  if (actualLimit <= 0) {
    return {
      columns: colIndices.map(() => []),
      columnNames: file.columnNames.slice(0, colIndices.length),
      total: totalRows
    };
  }
  const indices = Array.from({ length: actualLimit }, (_, i) => actualOffset + i);
  const columnTypes = await file.detectColumnTypes();
  const columnPromises = colIndices.map(async (colIdx) => {
    const type = columnTypes[colIdx] || "unknown";
    try {
      switch (type) {
        case "string":
        case "utf8":
        case "large_utf8":
          return await file.readStringsAtIndices(colIdx, indices);
        case "int64":
          return Array.from(await file.readInt64AtIndices(colIdx, indices));
        case "int32":
          return Array.from(await file.readInt32AtIndices(colIdx, indices));
        case "int16":
          return Array.from(await file.readInt16AtIndices(colIdx, indices));
        case "uint8":
          return Array.from(await file.readUint8AtIndices(colIdx, indices));
        case "float64":
        case "double":
          return Array.from(await file.readFloat64AtIndices(colIdx, indices));
        case "float32":
        case "float":
          return Array.from(await file.readFloat32AtIndices(colIdx, indices));
        case "bool":
        case "boolean":
          return await file.readBoolAtIndices(colIdx, indices);
        case "fixed_size_list":
        case "vector":
          const vectors = await file.readVectorsAtIndices(colIdx, indices);
          return Array.isArray(vectors) ? vectors : Array.from(vectors);
        default:
          return await file.readStringsAtIndices(colIdx, indices);
      }
    } catch {
      return indices.map(() => null);
    }
  });
  const columnsData = await Promise.all(columnPromises);
  return {
    columns: columnsData,
    columnNames: colIndices.map((i) => file.columnNames[i] || `column_${i}`),
    total: totalRows
  };
}
var init_remote_file_vector = __esm({
  "src/client/lance/remote-file-vector.js"() {
    init_remote_file_proto();
    init_accelerator();
  }
});

// src/client/lance/remote-file.js
var remote_file_exports = {};
__export(remote_file_exports, {
  RemoteLanceFile: () => RemoteLanceFile2
});
var RemoteLanceFile2;
var init_remote_file = __esm({
  "src/client/lance/remote-file.js"() {
    init_hot_tier_cache();
    init_ivf_index();
    init_remote_file_meta();
    init_remote_file_proto();
    init_remote_file_numeric();
    init_remote_file_string();
    init_remote_file_vector();
    RemoteLanceFile2 = class _RemoteLanceFile {
      constructor(lanceql, url, fileSize, footerData) {
        this.lanceql = lanceql;
        this.wasm = lanceql.wasm;
        this.memory = lanceql.memory;
        this.url = url;
        this.fileSize = fileSize;
        const bytes = new Uint8Array(footerData);
        this.footerPtr = this.wasm.alloc(bytes.length);
        if (!this.footerPtr) {
          throw new Error("Failed to allocate memory for footer");
        }
        this.footerLen = bytes.length;
        new Uint8Array(this.memory.buffer).set(bytes, this.footerPtr);
        this._numColumns = this.wasm.parseFooterGetColumns(this.footerPtr, this.footerLen);
        this._majorVersion = this.wasm.parseFooterGetMajorVersion(this.footerPtr, this.footerLen);
        this._minorVersion = this.wasm.parseFooterGetMinorVersion(this.footerPtr, this.footerLen);
        this._columnMetaStart = this.wasm.getColumnMetaStart(this.footerPtr, this.footerLen);
        this._columnMetaOffsetsStart = this.wasm.getColumnMetaOffsetsStart(this.footerPtr, this.footerLen);
        this._columnMetaCache = /* @__PURE__ */ new Map();
        this._columnOffsetCache = /* @__PURE__ */ new Map();
        this._columnTypes = null;
        this._schema = null;
        this._datasetBaseUrl = null;
        this._ivfIndex = null;
      }
      /**
       * Open a remote Lance file.
       */
      static async open(lanceql, url) {
        const headResponse = await fetch(url, { method: "HEAD" });
        if (!headResponse.ok) {
          throw new Error(`HTTP error: ${headResponse.status}`);
        }
        const contentLength = headResponse.headers.get("Content-Length");
        if (!contentLength) {
          throw new Error("Server did not return Content-Length");
        }
        const fileSize = parseInt(contentLength, 10);
        const footerSize = 40;
        const footerStart = fileSize - footerSize;
        const footerResponse = await fetch(url, {
          headers: {
            "Range": `bytes=${footerStart}-${fileSize - 1}`
          }
        });
        if (!footerResponse.ok && footerResponse.status !== 206) {
          throw new Error(`HTTP error: ${footerResponse.status}`);
        }
        const footerData = await footerResponse.arrayBuffer();
        const footerBytes = new Uint8Array(footerData);
        const magic = String.fromCharCode(
          footerBytes[36],
          footerBytes[37],
          footerBytes[38],
          footerBytes[39]
        );
        if (magic !== "LANC") {
          throw new Error(`Invalid Lance file: expected LANC magic, got "${magic}"`);
        }
        const file = new _RemoteLanceFile(lanceql, url, fileSize, footerData);
        await tryLoadSchema(file);
        await file._tryLoadIndex();
        console.log(`[LanceQL] Loaded: ${file._numColumns} columns, ${(fileSize / 1024 / 1024).toFixed(1)}MB, schema: ${file._schema ? "yes" : "no"}, index: ${file.hasIndex() ? "yes" : "no"}`);
        return file;
      }
      /**
       * Try to load IVF index from dataset.
       */
      async _tryLoadIndex() {
        if (!this._datasetBaseUrl) return;
        try {
          this._ivfIndex = await IVFIndex.tryLoad(this._datasetBaseUrl);
        } catch (e) {
        }
      }
      /**
       * Check if ANN index is available.
       */
      hasIndex() {
        return this._ivfIndex !== null && this._ivfIndex.centroids !== null;
      }
      // === Properties ===
      get columnNames() {
        return getColumnNames(this);
      }
      get schema() {
        return this._schema;
      }
      get datasetBaseUrl() {
        return this._datasetBaseUrl;
      }
      get numColumns() {
        return this._numColumns;
      }
      get size() {
        return this.fileSize;
      }
      get version() {
        return {
          major: this._majorVersion,
          minor: this._minorVersion
        };
      }
      get columnMetaStart() {
        return Number(this._columnMetaStart);
      }
      get columnMetaOffsetsStart() {
        return Number(this._columnMetaOffsetsStart);
      }
      // === Core Methods ===
      /**
       * Fetch bytes from the remote file at a specific range.
       * Uses HotTierCache for OPFS-backed caching.
       */
      async fetchRange(start, end) {
        if (start < 0 || end < start || end >= this.size) {
          console.error(`Invalid range: ${start}-${end}, file size: ${this.size}`);
        }
        const cache = getHotTierCache();
        if (cache.enabled) {
          const data2 = await cache.getRange(this.url, start, end, this.size);
          if (this._onFetch) {
            this._onFetch(data2.byteLength, 1);
          }
          return data2;
        }
        const response = await fetch(this.url, {
          headers: {
            "Range": `bytes=${start}-${end}`
          }
        });
        if (!response.ok && response.status !== 206) {
          console.error(`Fetch failed: ${response.status} for range ${start}-${end}`);
          throw new Error(`HTTP error: ${response.status}`);
        }
        const data = await response.arrayBuffer();
        if (this._onFetch) {
          this._onFetch(data.byteLength, 1);
        }
        return data;
      }
      /**
       * Set callback for network stats tracking.
       */
      onFetch(callback) {
        this._onFetch = callback;
      }
      /**
       * Close the file and free memory.
       */
      close() {
        if (this.footerPtr) {
          this.wasm.free(this.footerPtr, this.footerLen);
          this.footerPtr = null;
        }
      }
      // === Column Metadata ===
      /**
       * Get column offset entry from column metadata offsets.
       */
      async getColumnOffsetEntry(colIdx) {
        if (colIdx >= this._numColumns) {
          return { pos: 0, len: 0 };
        }
        if (this._columnOffsetCache.has(colIdx)) {
          return this._columnOffsetCache.get(colIdx);
        }
        const entryOffset = this.columnMetaOffsetsStart + colIdx * 16;
        const data = await this.fetchRange(entryOffset, entryOffset + 15);
        const view = new DataView(data);
        const entry = {
          pos: Number(view.getBigUint64(0, true)),
          len: Number(view.getBigUint64(8, true))
        };
        this._columnOffsetCache.set(colIdx, entry);
        return entry;
      }
      /**
       * Get debug info for a column.
       */
      async getColumnDebugInfo(colIdx) {
        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) {
          return { offset: 0, size: 0, rows: 0 };
        }
        const colMetaData = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const bytes = new Uint8Array(colMetaData);
        return this._parseColumnMeta(bytes);
      }
      _parseColumnMeta(bytes) {
        return parseColumnMeta(bytes);
      }
      _parseStringColumnMeta(bytes) {
        return parseStringColumnMeta(bytes);
      }
      _batchIndices(indices, valueSize, gapThreshold = 1024) {
        return batchIndices(indices, valueSize, gapThreshold);
      }
      async _getCachedColumnMeta(colIdx) {
        if (this._columnMetaCache.has(colIdx)) {
          return this._columnMetaCache.get(colIdx);
        }
        const entry = await this.getColumnOffsetEntry(colIdx);
        if (entry.len === 0) {
          return null;
        }
        const colMeta = await this.fetchRange(entry.pos, entry.pos + entry.len - 1);
        const bytes = new Uint8Array(colMeta);
        this._columnMetaCache.set(colIdx, bytes);
        return bytes;
      }
      // === Numeric Column Readers ===
      readInt64AtIndices(colIdx, indices) {
        return readInt64AtIndices(this, colIdx, indices);
      }
      readFloat64AtIndices(colIdx, indices) {
        return readFloat64AtIndices(this, colIdx, indices);
      }
      readInt32AtIndices(colIdx, indices) {
        return readInt32AtIndices(this, colIdx, indices);
      }
      readFloat32AtIndices(colIdx, indices) {
        return readFloat32AtIndices(this, colIdx, indices);
      }
      readInt16AtIndices(colIdx, indices) {
        return readInt16AtIndices(this, colIdx, indices);
      }
      readUint8AtIndices(colIdx, indices) {
        return readUint8AtIndices(this, colIdx, indices);
      }
      readBoolAtIndices(colIdx, indices) {
        return readBoolAtIndices(this, colIdx, indices);
      }
      // === String Column Readers ===
      readStringAt(colIdx, rowIdx) {
        return readStringAt(this, colIdx, rowIdx);
      }
      readStringsAtIndices(colIdx, indices) {
        return readStringsAtIndices(this, colIdx, indices);
      }
      // === Row Count ===
      async getRowCount(colIdx) {
        const info = await this.getColumnDebugInfo(colIdx);
        return info.rows;
      }
      // === Type Detection ===
      detectColumnTypes() {
        return detectColumnTypes(this);
      }
      // === Vector Operations ===
      getVectorInfo(colIdx) {
        return getVectorInfo(this, colIdx);
      }
      readVectorAt(colIdx, rowIdx) {
        return readVectorAt(this, colIdx, rowIdx);
      }
      readVectorsAtIndices(colIdx, indices) {
        return readVectorsAtIndices(this, colIdx, indices);
      }
      cosineSimilarity(vecA, vecB) {
        return cosineSimilarity(vecA, vecB);
      }
      vectorSearch(colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
        return vectorSearch(this, colIdx, queryVec, topK, onProgress, options);
      }
      readVectorColumn(colIdx) {
        return readVectorColumn(this, colIdx);
      }
      readRows(options = {}) {
        return readRows(this, options);
      }
    };
  }
});

// src/client/lance/remote-dataset-del.js
function parseDeletionFile(data, fragId, baseUrl) {
  let fileType = 0;
  let readVersion = 0;
  let id = 0;
  let numDeletedRows = 0;
  let pos = 0;
  const readVarint = () => {
    let result = 0;
    let shift = 0;
    while (pos < data.length) {
      const b = data[pos++];
      result |= (b & 127) << shift;
      if ((b & 128) === 0) break;
      shift += 7;
    }
    return result;
  };
  while (pos < data.length) {
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 7;
    if (wireType === 0) {
      const val = readVarint();
      if (fieldNum === 1) fileType = val;
      else if (fieldNum === 2) readVersion = val;
      else if (fieldNum === 3) id = val;
      else if (fieldNum === 4) numDeletedRows = val;
    } else if (wireType === 2) {
      const len = readVarint();
      pos += len;
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    }
  }
  if (numDeletedRows === 0) return null;
  const ext = fileType === 0 ? "arrow" : "bin";
  const path = `_deletions/${fragId}-${readVersion}-${id}.${ext}`;
  return {
    fileType: fileType === 0 ? "arrow" : "bitmap",
    readVersion,
    id,
    numDeletedRows,
    path,
    url: `${baseUrl}/${path}`
  };
}
function parseArrowDeletions(data) {
  const deletedSet = /* @__PURE__ */ new Set();
  let pos = 0;
  if (data.length >= 8 && String.fromCharCode(...data.slice(0, 6)) === "ARROW1") {
    pos = 8;
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  while (pos < data.length - 4) {
    const marker = view.getInt32(pos, true);
    if (marker === -1) {
      pos += 4;
      if (pos + 4 > data.length) break;
      const metaLen = view.getInt32(pos, true);
      pos += 4 + metaLen;
      while (pos + 4 <= data.length) {
        const nextMarker = view.getInt32(pos, true);
        if (nextMarker === -1) break;
        const val = view.getInt32(pos, true);
        if (val >= 0 && val < 1e7) {
          deletedSet.add(val);
        }
        pos += 4;
      }
    } else {
      pos++;
    }
  }
  return deletedSet;
}
function parseRoaringBitmap(data) {
  const deletedSet = /* @__PURE__ */ new Set();
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  if (data.length < 8) return deletedSet;
  const cookie = view.getUint32(0, true);
  if (cookie === 12346 || cookie === 12347) {
    const isRunContainer = cookie === 12347;
    let pos = 4;
    const numContainers = view.getUint16(pos, true);
    pos += 2;
    const keysStart = pos;
    pos += numContainers * 4;
    for (let i = 0; i < numContainers && pos < data.length; i++) {
      const key = view.getUint16(keysStart + i * 4, true);
      const card = view.getUint16(keysStart + i * 4 + 2, true) + 1;
      const baseValue = key << 16;
      for (let j = 0; j < card && pos + 2 <= data.length; j++) {
        const lowBits = view.getUint16(pos, true);
        deletedSet.add(baseValue | lowBits);
        pos += 2;
      }
    }
  }
  return deletedSet;
}
async function loadDeletedRows(dataset, fragmentIndex) {
  if (dataset._deletedRows.has(fragmentIndex)) {
    return dataset._deletedRows.get(fragmentIndex);
  }
  const frag = dataset._fragments[fragmentIndex];
  if (!frag?.deletionFile) {
    const emptySet = /* @__PURE__ */ new Set();
    dataset._deletedRows.set(fragmentIndex, emptySet);
    return emptySet;
  }
  const { url, fileType, numDeletedRows } = frag.deletionFile;
  console.log(`[LanceQL] Loading ${numDeletedRows} deletions from ${url} (${fileType})`);
  try {
    const response = await fetch(url);
    if (!response.ok) {
      console.warn(`[LanceQL] Failed to load deletion file: ${response.status}`);
      const emptySet = /* @__PURE__ */ new Set();
      dataset._deletedRows.set(fragmentIndex, emptySet);
      return emptySet;
    }
    const buffer = await response.arrayBuffer();
    const data = new Uint8Array(buffer);
    let deletedSet;
    if (fileType === "arrow") {
      deletedSet = parseArrowDeletions(data);
    } else {
      deletedSet = parseRoaringBitmap(data);
    }
    console.log(`[LanceQL] Loaded ${deletedSet.size} deleted rows for fragment ${fragmentIndex}`);
    dataset._deletedRows.set(fragmentIndex, deletedSet);
    return deletedSet;
  } catch (e) {
    console.error(`[LanceQL] Error loading deletion file:`, e);
    const emptySet = /* @__PURE__ */ new Set();
    dataset._deletedRows.set(fragmentIndex, emptySet);
    return emptySet;
  }
}
var init_remote_dataset_del = __esm({
  "src/client/lance/remote-dataset-del.js"() {
  }
});

// src/client/lance/remote-dataset-search.js
async function vectorSearch2(dataset, colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
  const { normalized = true, workerPool = null, useIndex = true, nprobe = 20 } = options;
  const vectorColIdx = colIdx;
  if (vectorColIdx < 0) {
    throw new Error("No vector column found in dataset");
  }
  const dim = queryVec.length;
  if (!dataset.hasIndex()) {
    throw new Error("No IVF index found. Vector search requires an IVF index for efficient querying.");
  }
  if (dataset._ivfIndex.dimension !== dim) {
    throw new Error(`Query dimension (${dim}) does not match index dimension (${dataset._ivfIndex.dimension}).`);
  }
  if (!dataset._ivfIndex.hasPartitionIndex) {
    throw new Error("IVF partition index (ivf_partitions.bin) not found. Required for efficient search.");
  }
  return await ivfIndexSearch(dataset, queryVec, topK, vectorColIdx, nprobe, onProgress);
}
async function ivfIndexSearch(dataset, queryVec, topK, vectorColIdx, nprobe, onProgress) {
  const partitions = dataset._ivfIndex.findNearestPartitions(queryVec, nprobe);
  const partitionData = await dataset._ivfIndex.fetchPartitionData(
    partitions,
    dataset._ivfIndex.dimension,
    (loaded, total) => {
      if (onProgress) {
        const pct = total > 0 ? loaded / total : 0;
        onProgress(Math.floor(pct * 80), 100);
      }
    }
  );
  if (!partitionData || partitionData.rowIds.length === 0) {
    throw new Error("IVF index not available. This dataset requires ivf_vectors.bin for efficient search.");
  }
  const { rowIds, vectors, preFlattened } = partitionData;
  const dim = queryVec.length;
  const numVectors = preFlattened ? vectors.length / dim : vectors.length;
  const scores = new Float32Array(numVectors);
  const accelerator = getWebGPUAccelerator();
  if (accelerator.isAvailable()) {
    const maxBatch = accelerator.getMaxVectorsPerBatch(dim);
    if (preFlattened) {
      for (let vecStart = 0; vecStart < numVectors; vecStart += maxBatch) {
        const vecEnd = Math.min(vecStart + maxBatch, numVectors);
        const batchCount = vecEnd - vecStart;
        const chunk = vectors.subarray(vecStart * dim, vecEnd * dim);
        try {
          const chunkScores = await accelerator.batchCosineSimilarity(queryVec, chunk, true, true);
          if (chunkScores) {
            scores.set(chunkScores, vecStart);
            continue;
          }
        } catch (e) {
        }
        if (dataset.lanceql?.batchCosineSimilarityFlat) {
          const chunkScores = dataset.lanceql.batchCosineSimilarityFlat(queryVec, chunk, dim, true);
          scores.set(chunkScores, vecStart);
        } else {
          for (let i = 0; i < batchCount; i++) {
            const offset = i * dim;
            let dot = 0;
            for (let k = 0; k < dim; k++) {
              dot += queryVec[k] * chunk[offset + k];
            }
            scores[vecStart + i] = dot;
          }
        }
      }
    } else {
      for (let start = 0; start < numVectors; start += maxBatch) {
        const end = Math.min(start + maxBatch, numVectors);
        const chunk = vectors.slice(start, end);
        try {
          const chunkScores = await accelerator.batchCosineSimilarity(queryVec, chunk, true, false);
          if (chunkScores) {
            scores.set(chunkScores, start);
            continue;
          }
        } catch (e) {
        }
        for (let i = 0; i < chunk.length; i++) {
          const vec = chunk[i];
          if (!vec || vec.length !== dim) continue;
          let dot = 0;
          for (let k = 0; k < dim; k++) {
            dot += queryVec[k] * vec[k];
          }
          scores[start + i] = dot;
        }
      }
    }
  } else {
    if (preFlattened) {
      for (let i = 0; i < numVectors; i++) {
        const offset = i * dim;
        let dot = 0;
        for (let k = 0; k < dim; k++) {
          dot += queryVec[k] * vectors[offset + k];
        }
        scores[i] = dot;
      }
    } else {
      for (let i = 0; i < numVectors; i++) {
        const vec = vectors[i];
        if (!vec || vec.length !== dim) continue;
        let dot = 0;
        for (let k = 0; k < dim; k++) {
          dot += queryVec[k] * vec[k];
        }
        scores[i] = dot;
      }
    }
  }
  if (onProgress) onProgress(90, 100);
  const allResults = new Array(rowIds.length);
  for (let i = 0; i < rowIds.length; i++) {
    allResults[i] = { index: rowIds[i], score: scores[i] };
  }
  const finalK = Math.min(topK, allResults.length);
  quickselectTopK2(allResults, finalK);
  if (onProgress) onProgress(100, 100);
  return {
    indices: allResults.slice(0, finalK).map((r) => r.index),
    scores: allResults.slice(0, finalK).map((r) => r.score),
    usedIndex: true,
    searchedRows: rowIds.length
  };
}
function quickselectTopK2(arr, k) {
  if (k >= arr.length || k <= 0) return;
  let left = 0;
  let right = arr.length - 1;
  while (left < right) {
    const mid = left + right >> 1;
    if (arr[mid].score > arr[left].score) swap2(arr, left, mid);
    if (arr[right].score > arr[left].score) swap2(arr, left, right);
    if (arr[mid].score > arr[right].score) swap2(arr, mid, right);
    const pivot = arr[right].score;
    let i = left;
    for (let j = left; j < right; j++) {
      if (arr[j].score >= pivot) {
        swap2(arr, i, j);
        i++;
      }
    }
    swap2(arr, i, right);
    if (i === k - 1) break;
    if (i < k - 1) left = i + 1;
    else right = i - 1;
  }
}
function swap2(arr, i, j) {
  const tmp = arr[i];
  arr[i] = arr[j];
  arr[j] = tmp;
}
function findVectorColumn(dataset) {
  if (!dataset._schema) return -1;
  for (let i = 0; i < dataset._schema.length; i++) {
    const field = dataset._schema[i];
    if (field.name === "embedding" || field.name === "vector" || field.type === "fixed_size_list" || field.type === "list") {
      return i;
    }
  }
  return dataset._schema.length - 1;
}
var init_remote_dataset_search = __esm({
  "src/client/lance/remote-dataset-search.js"() {
    init_accelerator();
  }
});

// src/client/lance/remote-dataset-frag.js
function getFragmentForRow(dataset, rowIdx) {
  let offset = 0;
  for (let i = 0; i < dataset._fragments.length; i++) {
    const frag = dataset._fragments[i];
    if (rowIdx < offset + frag.numRows) {
      return { fragmentIndex: i, localIndex: rowIdx - offset };
    }
    offset += frag.numRows;
  }
  return null;
}
function groupIndicesByFragment(dataset, indices) {
  const groups = /* @__PURE__ */ new Map();
  for (const globalIdx of indices) {
    const loc = getFragmentForRow(dataset, globalIdx);
    if (!loc) continue;
    if (!groups.has(loc.fragmentIndex)) {
      groups.set(loc.fragmentIndex, { localIndices: [], globalIndices: [] });
    }
    groups.get(loc.fragmentIndex).localIndices.push(loc.localIndex);
    groups.get(loc.fragmentIndex).globalIndices.push(globalIdx);
  }
  return groups;
}
async function readRows2(dataset, { offset = 0, limit = 50, columns = null, _isPrefetch = false } = {}) {
  const fragmentRanges = [];
  let currentOffset = 0;
  for (let i = 0; i < dataset._fragments.length; i++) {
    const frag = dataset._fragments[i];
    const fragStart = currentOffset;
    const fragEnd = currentOffset + frag.numRows;
    if (fragEnd > offset && fragStart < offset + limit) {
      const localStart = Math.max(0, offset - fragStart);
      const localEnd = Math.min(frag.numRows, offset + limit - fragStart);
      fragmentRanges.push({
        fragmentIndex: i,
        localOffset: localStart,
        localLimit: localEnd - localStart,
        globalStart: fragStart + localStart
      });
    }
    currentOffset = fragEnd;
    if (currentOffset >= offset + limit) break;
  }
  if (fragmentRanges.length === 0) {
    return { columns: [], columnNames: dataset.columnNames, total: dataset._totalRows };
  }
  const fetchPromises = fragmentRanges.map(async (range) => {
    const file = await dataset.openFragment(range.fragmentIndex);
    const result2 = await file.readRows({
      offset: range.localOffset,
      limit: range.localLimit,
      columns
    });
    return { ...range, result: result2 };
  });
  const results = await Promise.all(fetchPromises);
  results.sort((a, b) => a.globalStart - b.globalStart);
  const mergedColumns = [];
  const colNames = results[0]?.result.columnNames || dataset.columnNames;
  const numCols = columns ? columns.length : dataset._numColumns;
  for (let c = 0; c < numCols; c++) {
    const colData = [];
    for (const r of results) {
      if (r.result.columns[c]) {
        colData.push(...r.result.columns[c]);
      }
    }
    mergedColumns.push(colData);
  }
  const result = {
    columns: mergedColumns,
    columnNames: colNames,
    total: dataset._totalRows
  };
  const nextOffset = offset + limit;
  if (!_isPrefetch && nextOffset < dataset._totalRows && limit <= 100) {
    prefetchNextPage(dataset, nextOffset, limit, columns);
  }
  return result;
}
function prefetchNextPage(dataset, offset, limit, columns) {
  const cacheKey = `${offset}-${limit}-${columns?.join(",") || "all"}`;
  if (dataset._prefetchCache?.has(cacheKey)) {
    return;
  }
  if (!dataset._prefetchCache) {
    dataset._prefetchCache = /* @__PURE__ */ new Map();
  }
  const prefetchPromise = readRows2(dataset, { offset, limit, columns, _isPrefetch: true }).then((result) => {
    dataset._prefetchCache.set(cacheKey, result);
  }).catch(() => {
  });
  dataset._prefetchCache.set(cacheKey, prefetchPromise);
}
async function readStringsAtIndices2(dataset, colIdx, indices) {
  const groups = groupIndicesByFragment(dataset, indices);
  const results = /* @__PURE__ */ new Map();
  const fetchPromises = [];
  for (const [fragIdx, group] of groups) {
    fetchPromises.push((async () => {
      const file = await dataset.openFragment(fragIdx);
      const data = await file.readStringsAtIndices(colIdx, group.localIndices);
      for (let i = 0; i < group.globalIndices.length; i++) {
        results.set(group.globalIndices[i], data[i]);
      }
    })());
  }
  await Promise.all(fetchPromises);
  return indices.map((idx) => results.get(idx) || null);
}
async function readInt64AtIndices2(dataset, colIdx, indices) {
  const groups = groupIndicesByFragment(dataset, indices);
  const results = /* @__PURE__ */ new Map();
  const fetchPromises = [];
  for (const [fragIdx, group] of groups) {
    fetchPromises.push((async () => {
      const file = await dataset.openFragment(fragIdx);
      const data = await file.readInt64AtIndices(colIdx, group.localIndices);
      for (let i = 0; i < group.globalIndices.length; i++) {
        results.set(group.globalIndices[i], data[i]);
      }
    })());
  }
  await Promise.all(fetchPromises);
  return new BigInt64Array(indices.map((idx) => results.get(idx) || 0n));
}
async function readFloat64AtIndices2(dataset, colIdx, indices) {
  const groups = groupIndicesByFragment(dataset, indices);
  const results = /* @__PURE__ */ new Map();
  const fetchPromises = [];
  for (const [fragIdx, group] of groups) {
    fetchPromises.push((async () => {
      const file = await dataset.openFragment(fragIdx);
      const data = await file.readFloat64AtIndices(colIdx, group.localIndices);
      for (let i = 0; i < group.globalIndices.length; i++) {
        results.set(group.globalIndices[i], data[i]);
      }
    })());
  }
  await Promise.all(fetchPromises);
  return new Float64Array(indices.map((idx) => results.get(idx) || 0));
}
async function readInt32AtIndices2(dataset, colIdx, indices) {
  const groups = groupIndicesByFragment(dataset, indices);
  const results = /* @__PURE__ */ new Map();
  const fetchPromises = [];
  for (const [fragIdx, group] of groups) {
    fetchPromises.push((async () => {
      const file = await dataset.openFragment(fragIdx);
      const data = await file.readInt32AtIndices(colIdx, group.localIndices);
      for (let i = 0; i < group.globalIndices.length; i++) {
        results.set(group.globalIndices[i], data[i]);
      }
    })());
  }
  await Promise.all(fetchPromises);
  return new Int32Array(indices.map((idx) => results.get(idx) || 0));
}
async function readFloat32AtIndices2(dataset, colIdx, indices) {
  const groups = groupIndicesByFragment(dataset, indices);
  const results = /* @__PURE__ */ new Map();
  const fetchPromises = [];
  for (const [fragIdx, group] of groups) {
    fetchPromises.push((async () => {
      const file = await dataset.openFragment(fragIdx);
      const data = await file.readFloat32AtIndices(colIdx, group.localIndices);
      for (let i = 0; i < group.globalIndices.length; i++) {
        results.set(group.globalIndices[i], data[i]);
      }
    })());
  }
  await Promise.all(fetchPromises);
  return new Float32Array(indices.map((idx) => results.get(idx) || 0));
}
var init_remote_dataset_frag = __esm({
  "src/client/lance/remote-dataset-frag.js"() {
  }
});

// src/client/lance/remote-dataset-sql.js
function parseSQL(sql) {
  const ast = { type: "SELECT", columns: "*", limit: null, offset: null, where: null };
  const upper = sql.toUpperCase();
  if (upper.includes("LIMIT")) {
    const match = sql.match(/LIMIT\s+(\d+)/i);
    if (match) ast.limit = parseInt(match[1]);
  }
  if (upper.includes("OFFSET")) {
    const match = sql.match(/OFFSET\s+(\d+)/i);
    if (match) ast.offset = parseInt(match[1]);
  }
  if (upper.includes("WHERE")) {
    ast.where = true;
  }
  return ast;
}
async function executeSQL(dataset, sql) {
  const ast = parseSQL(sql);
  if (ast.type === "SELECT" && ast.columns === "*" && !ast.where) {
    const limit = ast.limit || 50;
    const offset = ast.offset || 0;
    return await dataset.readRows({ offset, limit });
  }
  const fetchPromises = dataset._fragments.map(async (frag, idx) => {
    const file = await dataset.openFragment(idx);
    try {
      return await file.executeSQL(sql);
    } catch (e) {
      console.warn(`Fragment ${idx} query failed:`, e);
      return { columns: [], columnNames: [], total: 0 };
    }
  });
  const results = await Promise.all(fetchPromises);
  if (results.length === 0 || results.every((r) => r.columns.length === 0)) {
    return { columns: [], columnNames: dataset.columnNames, total: 0 };
  }
  const firstValid = results.find((r) => r.columns.length > 0);
  if (!firstValid) {
    return { columns: [], columnNames: dataset.columnNames, total: 0 };
  }
  const numCols = firstValid.columns.length;
  const colNames = firstValid.columnNames;
  const mergedColumns = Array.from({ length: numCols }, () => []);
  let totalRows = 0;
  for (const r of results) {
    for (let c = 0; c < numCols && c < r.columns.length; c++) {
      mergedColumns[c].push(...r.columns[c]);
    }
    totalRows += r.total;
  }
  if (ast.limit) {
    const offset = ast.offset || 0;
    for (let c = 0; c < numCols; c++) {
      mergedColumns[c] = mergedColumns[c].slice(offset, offset + ast.limit);
    }
  }
  return {
    columns: mergedColumns,
    columnNames: colNames,
    total: totalRows
  };
}
var init_remote_dataset_sql = __esm({
  "src/client/lance/remote-dataset-sql.js"() {
  }
});

// src/client/lance/remote-dataset.js
var remote_dataset_exports = {};
__export(remote_dataset_exports, {
  RemoteLanceDataset: () => RemoteLanceDataset2
});
var metadataCache2, RemoteLanceDataset2;
var init_remote_dataset = __esm({
  "src/client/lance/remote-dataset.js"() {
    init_remote_file();
    init_ivf_index();
    init_metadata_cache();
    init_remote_dataset_del();
    init_remote_dataset_search();
    init_remote_dataset_frag();
    init_remote_dataset_sql();
    metadataCache2 = new MetadataCache();
    RemoteLanceDataset2 = class _RemoteLanceDataset {
      constructor(lanceql, baseUrl) {
        this.lanceql = lanceql;
        this.baseUrl = baseUrl.replace(/\/$/, "");
        this._fragments = [];
        this._schema = null;
        this._totalRows = 0;
        this._numColumns = 0;
        this._onFetch = null;
        this._fragmentFiles = /* @__PURE__ */ new Map();
        this._isRemote = true;
        this._ivfIndex = null;
        this._deletedRows = /* @__PURE__ */ new Map();
      }
      /**
       * Open a remote Lance dataset.
       * @param {LanceQL} lanceql - LanceQL instance
       * @param {string} baseUrl - Base URL to the dataset
       * @param {object} [options] - Options
       * @param {number} [options.version] - Specific version to load (time-travel)
       * @param {boolean} [options.prefetch] - Prefetch fragment metadata (default: true for small datasets)
       * @returns {Promise<RemoteLanceDataset>}
       */
      static async open(lanceql, baseUrl, options = {}) {
        const dataset = new _RemoteLanceDataset(lanceql, baseUrl);
        dataset._requestedVersion = options.version || null;
        const cacheKey = options.version ? `${baseUrl}@v${options.version}` : baseUrl;
        if (!options.skipCache) {
          const cached = await metadataCache2.get(cacheKey);
          if (cached && cached.schema && cached.fragments) {
            dataset._schema = cached.schema;
            dataset._fragments = cached.fragments;
            dataset._numColumns = cached.schema.length;
            dataset._totalRows = cached.fragments.reduce((sum, f) => sum + f.numRows, 0);
            dataset._version = cached.version;
            dataset._columnTypes = cached.columnTypes || null;
            dataset._fromCache = true;
          }
        }
        if (!dataset._fromCache) {
          const sidecarLoaded = await dataset._tryLoadSidecar();
          if (!sidecarLoaded) {
            await dataset._loadManifest();
          }
          metadataCache2.set(cacheKey, {
            schema: dataset._schema,
            fragments: dataset._fragments,
            version: dataset._version,
            columnTypes: dataset._columnTypes || null
          }).catch(() => {
          });
        }
        await dataset._tryLoadIndex();
        const shouldPrefetch = options.prefetch ?? dataset._fragments.length <= 5;
        if (shouldPrefetch && dataset._fragments.length > 0) {
          dataset._prefetchFragments();
        }
        return dataset;
      }
      /**
       * Try to load sidecar manifest (.meta.json) for faster startup.
       * @returns {Promise<boolean>} True if sidecar was loaded successfully
       * @private
       */
      async _tryLoadSidecar() {
        try {
          const sidecarUrl = `${this.baseUrl}/.meta.json`;
          const response = await fetch(sidecarUrl);
          if (!response.ok) {
            return false;
          }
          const sidecar = await response.json();
          if (!sidecar.schema || !sidecar.fragments) {
            return false;
          }
          this._schema = sidecar.schema.map((col) => ({
            name: col.name,
            id: col.index,
            type: col.type
          }));
          this._fragments = sidecar.fragments.map((frag) => ({
            id: frag.id,
            path: frag.data_files?.[0] || `${frag.id}.lance`,
            numRows: frag.num_rows,
            physicalRows: frag.physical_rows || frag.num_rows,
            url: `${this.baseUrl}/data/${frag.data_files?.[0] || frag.id + ".lance"}`,
            deletionFile: frag.has_deletions ? { numDeletedRows: frag.deleted_rows || 0 } : null
          }));
          this._numColumns = sidecar.num_columns;
          this._totalRows = sidecar.total_rows;
          this._version = sidecar.lance_version;
          this._columnTypes = sidecar.schema.map((col) => {
            const type = col.type;
            if (type.startsWith("vector[")) return "vector";
            if (type === "float64" || type === "double") return "float64";
            if (type === "float32") return "float32";
            if (type.includes("int")) return type;
            if (type === "string") return "string";
            return "unknown";
          });
          return true;
        } catch (e) {
          return false;
        }
      }
      /**
       * Prefetch fragment metadata (footers) in parallel.
       * Does not block - runs in background.
       * @private
       */
      _prefetchFragments() {
        const prefetchPromises = this._fragments.map(
          (_, idx) => this.openFragment(idx).catch(() => null)
        );
        Promise.all(prefetchPromises).catch(() => {
        });
      }
      /**
       * Check if dataset has an IVF index loaded.
       */
      hasIndex() {
        return this._ivfIndex !== null && this._ivfIndex.centroids !== null;
      }
      /**
       * Try to load IVF index from _indices folder.
       * @private
       */
      async _tryLoadIndex() {
        try {
          this._ivfIndex = await IVFIndex.tryLoad(this.baseUrl);
        } catch {
          this._ivfIndex = null;
        }
      }
      /**
       * Load and parse the manifest to discover fragments.
       * @private
       */
      async _loadManifest() {
        let manifestData = null;
        let manifestVersion = 0;
        if (this._requestedVersion) {
          manifestVersion = this._requestedVersion;
          const manifestUrl = `${this.baseUrl}/_versions/${manifestVersion}.manifest`;
          const response = await fetch(manifestUrl);
          if (!response.ok) {
            throw new Error(`Version ${manifestVersion} not found (${response.status})`);
          }
          manifestData = new Uint8Array(await response.arrayBuffer());
        } else {
          const checkVersions = [1, 5, 10, 20, 50, 100];
          const checks = await Promise.all(
            checkVersions.map(async (v) => {
              try {
                const url = `${this.baseUrl}/_versions/${v}.manifest`;
                const response2 = await fetch(url, { method: "HEAD" });
                return response2.ok ? v : 0;
              } catch {
                return 0;
              }
            })
          );
          let highestFound = Math.max(...checks);
          if (highestFound > 0) {
            for (let v = highestFound + 1; v <= highestFound + 50; v++) {
              try {
                const url = `${this.baseUrl}/_versions/${v}.manifest`;
                const response2 = await fetch(url, { method: "HEAD" });
                if (response2.ok) {
                  highestFound = v;
                } else {
                  break;
                }
              } catch {
                break;
              }
            }
          }
          manifestVersion = highestFound;
          if (manifestVersion === 0) {
            throw new Error("No manifest found in dataset");
          }
          const manifestUrl = `${this.baseUrl}/_versions/${manifestVersion}.manifest`;
          const response = await fetch(manifestUrl);
          if (!response.ok) {
            throw new Error(`Failed to fetch manifest: ${response.status}`);
          }
          manifestData = new Uint8Array(await response.arrayBuffer());
        }
        this._version = manifestVersion;
        this._latestVersion = this._requestedVersion ? null : manifestVersion;
        this._parseManifest(manifestData);
      }
      /**
       * Get list of available versions.
       * @returns {Promise<number[]>}
       */
      async listVersions() {
        const versions = [];
        const maxVersion = this._latestVersion || 100;
        const checks = await Promise.all(
          Array.from({ length: maxVersion }, (_, i) => i + 1).map(async (v) => {
            try {
              const url = `${this.baseUrl}/_versions/${v}.manifest`;
              const response = await fetch(url, { method: "HEAD" });
              return response.ok ? v : 0;
            } catch {
              return 0;
            }
          })
        );
        return checks.filter((v) => v > 0);
      }
      /**
       * Get current loaded version.
       */
      get version() {
        return this._version;
      }
      /**
       * Parse manifest protobuf to extract schema and fragment info.
       *
       * Lance manifest file structure:
       * - Chunk 1 (len-prefixed): Transaction metadata (may be small/incremental)
       * - Chunk 2 (len-prefixed): Full manifest with schema + fragments
       * - Footer (16 bytes): Offsets + "LANC" magic
       *
       * @private
       */
      _parseManifest(bytes) {
        const view = new DataView(bytes.buffer, bytes.byteOffset);
        const chunk1Len = view.getUint32(0, true);
        const chunk2Start = 4 + chunk1Len;
        let protoData;
        if (chunk2Start + 4 < bytes.length) {
          const chunk2Len = view.getUint32(chunk2Start, true);
          if (chunk2Len > 0 && chunk2Start + 4 + chunk2Len <= bytes.length) {
            protoData = bytes.slice(chunk2Start + 4, chunk2Start + 4 + chunk2Len);
          } else {
            protoData = bytes.slice(4, 4 + chunk1Len);
          }
        } else {
          protoData = bytes.slice(4, 4 + chunk1Len);
        }
        let pos = 0;
        const fields = [];
        const fragments = [];
        const readVarint = () => {
          let result = 0;
          let shift = 0;
          while (pos < protoData.length) {
            const byte = protoData[pos++];
            result |= (byte & 127) << shift;
            if ((byte & 128) === 0) break;
            shift += 7;
          }
          return result;
        };
        const skipField = (wireType) => {
          if (wireType === 0) {
            readVarint();
          } else if (wireType === 2) {
            const len = readVarint();
            pos += len;
          } else if (wireType === 5) {
            pos += 4;
          } else if (wireType === 1) {
            pos += 8;
          }
        };
        while (pos < protoData.length) {
          const tag = readVarint();
          const fieldNum = tag >> 3;
          const wireType = tag & 7;
          if (fieldNum === 1 && wireType === 2) {
            const fieldLen = readVarint();
            const fieldEnd = pos + fieldLen;
            let name = null;
            let id = null;
            let logicalType = null;
            while (pos < fieldEnd) {
              const fTag = readVarint();
              const fNum = fTag >> 3;
              const fWire = fTag & 7;
              if (fWire === 0) {
                const val = readVarint();
                if (fNum === 3) id = val;
              } else if (fWire === 2) {
                const len = readVarint();
                const content = protoData.slice(pos, pos + len);
                pos += len;
                if (fNum === 2) {
                  name = new TextDecoder().decode(content);
                } else if (fNum === 5) {
                  logicalType = new TextDecoder().decode(content);
                }
              } else {
                skipField(fWire);
              }
            }
            if (name) {
              fields.push({ name, id, type: logicalType });
            }
          } else if (fieldNum === 2 && wireType === 2) {
            const fragLen = readVarint();
            const fragEnd = pos + fragLen;
            let fragId = null;
            let filePath = null;
            let numRows = 0;
            let deletionFile = null;
            while (pos < fragEnd) {
              const fTag = readVarint();
              const fNum = fTag >> 3;
              const fWire = fTag & 7;
              if (fWire === 0) {
                const val = readVarint();
                if (fNum === 1) fragId = val;
                else if (fNum === 4) numRows = val;
              } else if (fWire === 2) {
                const len = readVarint();
                const content = protoData.slice(pos, pos + len);
                pos += len;
                if (fNum === 2) {
                  let innerPos = 0;
                  while (innerPos < content.length) {
                    const iTag = content[innerPos++];
                    const iNum = iTag >> 3;
                    const iWire = iTag & 7;
                    if (iWire === 2) {
                      let iLen = 0;
                      let iShift = 0;
                      while (innerPos < content.length) {
                        const b = content[innerPos++];
                        iLen |= (b & 127) << iShift;
                        if ((b & 128) === 0) break;
                        iShift += 7;
                      }
                      const iContent = content.slice(innerPos, innerPos + iLen);
                      innerPos += iLen;
                      if (iNum === 1) {
                        filePath = new TextDecoder().decode(iContent);
                      }
                    } else if (iWire === 0) {
                      while (innerPos < content.length && (content[innerPos++] & 128) !== 0) ;
                    } else if (iWire === 5) {
                      innerPos += 4;
                    } else if (iWire === 1) {
                      innerPos += 8;
                    }
                  }
                } else if (fNum === 3) {
                  deletionFile = this._parseDeletionFile(content, fragId);
                }
              } else {
                skipField(fWire);
              }
            }
            if (filePath) {
              const logicalRows = deletionFile ? numRows - deletionFile.numDeletedRows : numRows;
              fragments.push({
                id: fragId,
                path: filePath,
                numRows: logicalRows,
                // Logical rows (excluding deleted)
                physicalRows: numRows,
                // Physical rows (including deleted)
                deletionFile,
                url: `${this.baseUrl}/data/${filePath}`
              });
            }
          } else {
            skipField(wireType);
          }
        }
        this._schema = fields;
        this._fragments = fragments;
        this._numColumns = fields.length;
        this._totalRows = fragments.reduce((sum, f) => sum + f.numRows, 0);
      }
      /**
       * Parse DeletionFile protobuf message.
       * @private
       */
      _parseDeletionFile(data, fragId) {
        return parseDeletionFile(data, fragId, this.baseUrl);
      }
      /**
       * Load deleted row indices for a fragment.
       * @param {number} fragmentIndex - Fragment index
       * @returns {Promise<Set<number>>} Set of deleted row indices (local to fragment)
       * @private
       */
      async _loadDeletedRows(fragmentIndex) {
        return loadDeletedRows(this, fragmentIndex);
      }
      /**
       * Check if a row is deleted in a fragment.
       * @param {number} fragmentIndex - Fragment index
       * @param {number} localRowIndex - Row index within the fragment
       * @returns {Promise<boolean>} True if row is deleted
       */
      async isRowDeleted(fragmentIndex, localRowIndex) {
        const deletedSet = await this._loadDeletedRows(fragmentIndex);
        return deletedSet.has(localRowIndex);
      }
      /**
       * Get number of columns.
       */
      get numColumns() {
        return this._numColumns;
      }
      /**
       * Get total row count across all fragments.
       */
      get rowCount() {
        return this._totalRows;
      }
      /**
       * Get row count for a column (for API compatibility with RemoteLanceFile).
       * @param {number} columnIndex - Column index (ignored, all columns have same row count)
       * @returns {Promise<number>}
       */
      async getRowCount(columnIndex = 0) {
        return this._totalRows;
      }
      /**
       * Read a single vector at a global row index.
       * Delegates to the correct fragment based on row index.
       * @param {number} colIdx - Column index
       * @param {number} rowIdx - Global row index
       * @returns {Promise<Float32Array>}
       */
      async readVectorAt(colIdx, rowIdx) {
        const loc = this._getFragmentForRow(rowIdx);
        if (!loc) return new Float32Array(0);
        const file = await this.openFragment(loc.fragmentIndex);
        return await file.readVectorAt(colIdx, loc.localIndex);
      }
      /**
       * Get vector info for a column by querying first fragment.
       * @param {number} colIdx - Column index
       * @returns {Promise<{rows: number, dimension: number}>}
       */
      async getVectorInfo(colIdx) {
        if (this._fragments.length === 0) {
          return { rows: 0, dimension: 0 };
        }
        const file = await this.openFragment(0);
        const fragInfo = await file.getVectorInfo(colIdx);
        if (fragInfo.dimension === 0) {
          return { rows: 0, dimension: 0 };
        }
        return {
          rows: this._totalRows,
          dimension: fragInfo.dimension
        };
      }
      /**
       * Get column names from schema.
       */
      get columnNames() {
        return this._schema ? this._schema.map((f) => f.name) : [];
      }
      /**
       * Get full schema.
       */
      get schema() {
        return this._schema;
      }
      /**
       * Get fragment list.
       */
      get fragments() {
        return this._fragments;
      }
      /**
       * Get estimated total size based on row count and schema.
       * More accurate than fragment count estimate.
       */
      get size() {
        if (this._cachedSize) return this._cachedSize;
        let bytesPerRow = 0;
        for (let i = 0; i < (this._columnTypes?.length || 0); i++) {
          const colType = this._columnTypes[i];
          if (colType === "int64" || colType === "float64" || colType === "double") {
            bytesPerRow += 8;
          } else if (colType === "int32" || colType === "float32") {
            bytesPerRow += 4;
          } else if (colType === "string") {
            bytesPerRow += 50;
          } else if (colType === "vector" || colType?.startsWith("vector[")) {
            const match = colType?.match(/\[(\d+)\]/);
            const dim = match ? parseInt(match[1]) : 384;
            bytesPerRow += dim * 4;
          } else {
            bytesPerRow += 8;
          }
        }
        if (bytesPerRow === 0) {
          bytesPerRow = 100;
        }
        this._cachedSize = this._totalRows * bytesPerRow;
        return this._cachedSize;
      }
      /**
       * Set callback for network fetch events.
       */
      onFetch(callback) {
        this._onFetch = callback;
      }
      /**
       * Open a specific fragment as RemoteLanceFile.
       * @param {number} fragmentIndex - Index of fragment to open
       * @returns {Promise<RemoteLanceFile>}
       */
      async openFragment(fragmentIndex) {
        if (fragmentIndex < 0 || fragmentIndex >= this._fragments.length) {
          throw new Error(`Invalid fragment index: ${fragmentIndex}`);
        }
        if (this._fragmentFiles.has(fragmentIndex)) {
          return this._fragmentFiles.get(fragmentIndex);
        }
        const fragment = this._fragments[fragmentIndex];
        const file = await RemoteLanceFile2.open(this.lanceql, fragment.url);
        if (this._onFetch) {
          file.onFetch(this._onFetch);
        }
        this._fragmentFiles.set(fragmentIndex, file);
        return file;
      }
      /**
       * Read rows from the dataset with pagination.
       * @param {Object} options - Query options
       * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
       */
      async readRows(options = {}) {
        return readRows2(this, options);
      }
      /**
       * Detect column types by sampling from first fragment.
       * @returns {Promise<string[]>}
       */
      async detectColumnTypes() {
        if (this._columnTypes && this._columnTypes.length > 0) {
          return this._columnTypes;
        }
        if (this._fragments.length === 0) {
          return [];
        }
        const file = await this.openFragment(0);
        const types = await file.detectColumnTypes();
        this._columnTypes = types;
        const cacheKey = this._requestedVersion ? `${this.baseUrl}@v${this._requestedVersion}` : this.baseUrl;
        metadataCache2.get(cacheKey).then((cached) => {
          if (cached) {
            cached.columnTypes = types;
            metadataCache2.set(cacheKey, cached).catch(() => {
            });
          }
        }).catch(() => {
        });
        return types;
      }
      /**
       * Helper to determine which fragment contains a given row index.
       * @private
       */
      _getFragmentForRow(rowIdx) {
        return getFragmentForRow(this, rowIdx);
      }
      /**
       * Group indices by fragment for efficient batch reading.
       * @private
       */
      _groupIndicesByFragment(indices) {
        return groupIndicesByFragment(this, indices);
      }
      /**
       * Read strings at specific indices across fragments.
       */
      async readStringsAtIndices(colIdx, indices) {
        return readStringsAtIndices2(this, colIdx, indices);
      }
      /**
       * Read int64 values at specific indices across fragments.
       */
      async readInt64AtIndices(colIdx, indices) {
        return readInt64AtIndices2(this, colIdx, indices);
      }
      /**
       * Read float64 values at specific indices across fragments.
       */
      async readFloat64AtIndices(colIdx, indices) {
        return readFloat64AtIndices2(this, colIdx, indices);
      }
      /**
       * Read int32 values at specific indices across fragments.
       */
      async readInt32AtIndices(colIdx, indices) {
        return readInt32AtIndices2(this, colIdx, indices);
      }
      /**
       * Read float32 values at specific indices across fragments.
       */
      async readFloat32AtIndices(colIdx, indices) {
        return readFloat32AtIndices2(this, colIdx, indices);
      }
      /**
       * Vector search across all fragments.
       * API compatible with RemoteLanceFile.vectorSearch.
       *
       * @param {number} colIdx - Vector column index
       * @param {Float32Array} queryVec - Query vector
       * @param {number} topK - Number of results to return
       * @param {Function} onProgress - Progress callback (current, total)
       * @param {Object} options - Search options
       * @returns {Promise<{indices: number[], scores: number[], usedIndex: boolean}>}
       */
      async vectorSearch(colIdx, queryVec, topK = 10, onProgress = null, options = {}) {
        return vectorSearch2(this, colIdx, queryVec, topK, onProgress, options);
      }
      /**
       * Find the vector column index by looking at schema.
       * @private
       */
      _findVectorColumn() {
        return findVectorColumn(this);
      }
      /**
       * Execute SQL query across all fragments in parallel.
       * @param {string} sql - SQL query
       * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
       */
      async executeSQL(sql) {
        return executeSQL(this, sql);
      }
      /**
       * Close all cached fragment files.
       */
      close() {
        for (const file of this._fragmentFiles.values()) {
          if (file.close) file.close();
        }
        this._fragmentFiles.clear();
      }
    };
  }
});

// src/client/rpc/worker-rpc.js
function checkSharedArrayBuffer() {
  try {
    if (typeof SharedArrayBuffer !== "undefined" && typeof crossOriginIsolated !== "undefined" && crossOriginIsolated) {
      _sharedBuffer = new SharedArrayBuffer(SHARED_BUFFER_SIZE);
      _transferMode = "sharedBuffer";
      console.log("[LanceQL] Using SharedArrayBuffer (zero-copy)");
      return true;
    }
  } catch (e) {
  }
  try {
    const test = new ArrayBuffer(8);
    if (typeof ArrayBuffer.prototype.transfer !== "undefined" || true) {
      _transferMode = "transfer";
      console.log("[LanceQL] Using Transferable ArrayBuffers");
      return false;
    }
  } catch (e) {
  }
  _transferMode = "clone";
  console.log("[LanceQL] Using structured clone (fallback)");
  return false;
}
function getLanceWorker() {
  if (_lanceWorker) return _lanceWorkerReady;
  checkSharedArrayBuffer();
  _lanceWorkerReady = new Promise((resolve, reject) => {
    console.log("[LanceQL] Using regular Worker for better logging");
    try {
      _lanceWorker = new Worker(
        new URL("./lanceql-worker.js", import_meta.url),
        { type: "module", name: "lanceql" }
      );
      _lanceWorker.onmessage = (e) => {
        handleWorkerMessage(e.data, _lanceWorker, resolve);
      };
      _lanceWorker.onerror = (e) => {
        console.error("[LanceQL] Worker error:", e);
        reject(e);
      };
      if (_sharedBuffer) {
        _lanceWorker.postMessage({
          type: "initSharedBuffer",
          buffer: _sharedBuffer
        });
      }
    } catch (e) {
      console.error("[LanceQL] Failed to create Worker:", e);
      reject(e);
    }
  });
  return _lanceWorkerReady;
}
function handleWorkerMessage(data, port, resolveReady) {
  console.log("[LanceQL] Incoming worker message:", data.type || (data.id !== void 0 ? "RPC reply" : "unknown"));
  if (data.type === "ready") {
    console.log("[LanceQL] Worker ready, mode:", _transferMode);
    resolveReady(port);
    return;
  }
  if (data.id !== void 0) {
    const pending = _pendingRequests.get(data.id);
    if (pending) {
      _pendingRequests.delete(data.id);
      if (data.sharedOffset !== void 0 && _sharedBuffer) {
        const view = new Uint8Array(_sharedBuffer, data.sharedOffset, data.sharedLength);
        const result = JSON.parse(new TextDecoder().decode(view));
        pending.resolve(result);
      } else if (data.error) {
        pending.reject(new Error(data.error));
      } else {
        let result = data.result;
        if (result && result._format === "cursor") {
          const { cursorId, columns, rowCount } = result;
          result = {
            _format: "columnar",
            columns,
            rowCount,
            _cursorId: cursorId,
            _fetched: false
          };
          Object.defineProperty(result, "data", {
            configurable: true,
            enumerable: true,
            get() {
              if (!this._fetched) {
                console.warn("Cursor data accessed - fetching from worker");
              }
              return {};
            }
          });
          Object.defineProperty(result, "rows", {
            configurable: true,
            enumerable: true,
            get() {
              return [];
            }
          });
        } else if (result && result._format === "wasm_binary") {
          const { buffer, columns, rowCount, schema } = result;
          const view = new DataView(buffer);
          const u8 = new Uint8Array(buffer);
          const HEADER_SIZE = 32;
          const COL_META_SIZE = 24;
          const colData = {};
          for (let i = 0; i < columns.length; i++) {
            const metaOffset = HEADER_SIZE + i * COL_META_SIZE;
            const colType = view.getUint32(metaOffset, true);
            const dataOffset = view.getUint32(metaOffset + 8, true);
            const dataSize = Number(view.getBigUint64(metaOffset + 12, true));
            const elemSize = view.getUint32(metaOffset + 20, true);
            const colName = columns[i];
            if (colType === 0) {
              const length = dataSize / elemSize;
              colData[colName] = elemSize === 8 ? new Float64Array(buffer, dataOffset, length) : new Float32Array(buffer, dataOffset, length);
            } else {
              const offsetsStart = dataOffset;
              const offsets = new Uint32Array(buffer, offsetsStart, rowCount);
              const strDataStart = dataOffset + rowCount * 4;
              const strDataSize = dataSize - rowCount * 4;
              const strData = u8.subarray(strDataStart, strDataStart + strDataSize);
              const decoder = new TextDecoder();
              const strings = new Array(rowCount);
              let decoded = false;
              colData[colName] = new Proxy(strings, {
                get(target, prop) {
                  if (prop === "length") return rowCount;
                  if (typeof prop === "string" && !isNaN(prop)) {
                    if (!decoded) {
                      for (let j = 0; j < rowCount; j++) {
                        const start = offsets[j];
                        const end = j < rowCount - 1 ? offsets[j + 1] : strDataSize;
                        target[j] = decoder.decode(strData.subarray(start, end));
                      }
                      decoded = true;
                    }
                    return target[+prop];
                  }
                  if (prop === Symbol.iterator) {
                    if (!decoded) {
                      for (let j = 0; j < rowCount; j++) {
                        const start = offsets[j];
                        const end = j < rowCount - 1 ? offsets[j + 1] : strDataSize;
                        target[j] = decoder.decode(strData.subarray(start, end));
                      }
                      decoded = true;
                    }
                    return () => target[Symbol.iterator]();
                  }
                  return target[prop];
                }
              });
            }
          }
          result = {
            _format: "columnar",
            columns,
            rowCount,
            data: colData
          };
          Object.defineProperty(result, "rows", {
            configurable: true,
            enumerable: true,
            get() {
              const rows = new Array(rowCount);
              const colArrays = columns.map((name) => colData[name]);
              for (let i = 0; i < rowCount; i++) {
                const row = {};
                for (let j = 0; j < columns.length; j++) {
                  row[columns[j]] = colArrays[j][i];
                }
                rows[i] = row;
              }
              Object.defineProperty(this, "rows", { value: rows, writable: false });
              return rows;
            }
          });
        } else if (result && result._format === "packed") {
          const { columns, rowCount, packedBuffer, colOffsets, stringData } = result;
          const colData = { ...stringData || {} };
          if (packedBuffer && colOffsets) {
            const TypedArrayMap = {
              Float64Array,
              Float32Array,
              Int32Array,
              Int16Array,
              Int8Array,
              Uint32Array,
              Uint16Array,
              Uint8Array,
              BigInt64Array,
              BigUint64Array
            };
            for (const [name, info] of Object.entries(colOffsets)) {
              const TypedArr = TypedArrayMap[info.type] || Float64Array;
              colData[name] = new TypedArr(packedBuffer, info.offset, info.length);
            }
          }
          result.data = colData;
          result._format = "columnar";
          Object.defineProperty(result, "rows", {
            configurable: true,
            enumerable: true,
            get() {
              const rows = new Array(rowCount);
              const colArrays = columns.map((name) => colData[name]);
              for (let i = 0; i < rowCount; i++) {
                const row = {};
                for (let j = 0; j < columns.length; j++) {
                  row[columns[j]] = colArrays[j][i];
                }
                rows[i] = row;
              }
              Object.defineProperty(this, "rows", { value: rows, writable: false });
              return rows;
            }
          });
        } else if (result && result._format === "columnar") {
          const { columns, rowCount, data: colData } = result;
          Object.defineProperty(result, "rows", {
            configurable: true,
            enumerable: true,
            get() {
              const rows = new Array(rowCount);
              const colArrays = columns.map((name) => colData[name]);
              for (let i = 0; i < rowCount; i++) {
                const row = {};
                for (let j = 0; j < columns.length; j++) {
                  row[columns[j]] = colArrays[j][i];
                }
                rows[i] = row;
              }
              Object.defineProperty(this, "rows", { value: rows, writable: false });
              return rows;
            }
          });
        }
        pending.resolve(result);
      }
    }
  }
}
async function workerRPC(method, args) {
  const port = await getLanceWorker();
  const id = ++_requestId;
  return new Promise((resolve, reject) => {
    _pendingRequests.set(id, { resolve, reject });
    const transferables = [];
    if (_transferMode === "transfer" && args) {
      for (const key of Object.keys(args)) {
        const val = args[key];
        if (val instanceof ArrayBuffer) {
          transferables.push(val);
        } else if (ArrayBuffer.isView(val)) {
          transferables.push(val.buffer);
        }
      }
    }
    if (transferables.length > 0) {
      port.postMessage({ id, method, args }, transferables);
    } else {
      port.postMessage({ id, method, args });
    }
  });
}
var import_meta, _lanceWorker, _lanceWorkerReady, _requestId, _pendingRequests, _transferMode, _sharedBuffer, SHARED_BUFFER_SIZE;
var init_worker_rpc = __esm({
  "src/client/rpc/worker-rpc.js"() {
    import_meta = {};
    _lanceWorker = null;
    _lanceWorkerReady = null;
    _requestId = 0;
    _pendingRequests = /* @__PURE__ */ new Map();
    _transferMode = "clone";
    _sharedBuffer = null;
    SHARED_BUFFER_SIZE = 16 * 1024 * 1024;
  }
});

// src/client/database/local-database.js
var local_database_exports = {};
__export(local_database_exports, {
  LocalDatabase: () => LocalDatabase,
  OPFSJoinExecutor: () => OPFSJoinExecutor
});
var OPFSJoinExecutor, LocalDatabase;
var init_local_database = __esm({
  "src/client/database/local-database.js"() {
    init_worker_rpc();
    OPFSJoinExecutor = class {
      constructor(storage = opfsStorage) {
        this.storage = storage;
        this.sessionId = `join_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
        this.basePath = `_join_temp/${this.sessionId}`;
        this.numPartitions = 64;
        this.chunkSize = 1e3;
        this.stats = {
          leftRowsWritten: 0,
          rightRowsWritten: 0,
          resultRowsWritten: 0,
          bytesWrittenToOPFS: 0,
          bytesReadFromOPFS: 0,
          partitionsUsed: /* @__PURE__ */ new Set()
        };
      }
      /**
       * Execute a hash join using OPFS for intermediate storage
       * @param {AsyncGenerator} leftStream - Async generator yielding {columns, rows} chunks
       * @param {AsyncGenerator} rightStream - Async generator yielding {columns, rows} chunks
       * @param {string} leftKey - Join key column name for left table
       * @param {string} rightKey - Join key column name for right table
       * @param {Object} options - Join options
       * @returns {AsyncGenerator} Yields result chunks
       */
      async *executeHashJoin(leftStream, rightStream, leftKey, rightKey, options = {}) {
        const {
          limit = Infinity,
          projection = null,
          leftAlias = "left",
          rightAlias = "right",
          joinType = "INNER",
          prePartitionedLeft = null
          // Optional: pre-partitioned left metadata for semi-join optimization
        } = options;
        try {
          let leftMeta;
          if (prePartitionedLeft) {
            leftMeta = prePartitionedLeft;
          } else {
            leftMeta = await this._partitionToOPFS(leftStream, leftKey, "left");
          }
          const rightMeta = await this._partitionToOPFS(rightStream, rightKey, "right");
          let totalYielded = 0;
          const leftNulls = new Array(leftMeta.columns.length).fill(null);
          const rightNulls = new Array(rightMeta.columns.length).fill(null);
          const resultColumns = [
            ...leftMeta.columns.map((c) => `${leftAlias}.${c}`),
            ...rightMeta.columns.map((c) => `${rightAlias}.${c}`)
          ];
          const yieldChunk = function* (chunk) {
            if (chunk.length > 0) {
              yield { columns: resultColumns, rows: chunk.splice(0) };
            }
          };
          if (joinType === "CROSS") {
            const chunk = [];
            for (const leftPartitionId of leftMeta.partitionsUsed) {
              const leftPartition = await this._loadPartition("left", leftPartitionId, leftMeta.columns);
              for (const rightPartitionId of rightMeta.partitionsUsed) {
                const rightPartition = await this._loadPartition("right", rightPartitionId, rightMeta.columns);
                for (const leftRow of leftPartition) {
                  for (const rightRow of rightPartition) {
                    if (totalYielded >= limit) break;
                    chunk.push([...leftRow, ...rightRow]);
                    totalYielded++;
                    if (chunk.length >= this.chunkSize) {
                      yield { columns: resultColumns, rows: chunk.splice(0) };
                    }
                  }
                  if (totalYielded >= limit) break;
                }
                if (totalYielded >= limit) break;
              }
              if (totalYielded >= limit) break;
            }
            if (chunk.length > 0) {
              yield { columns: resultColumns, rows: chunk };
            }
            return;
          }
          const leftKeyIndex = leftMeta.columns.indexOf(leftKey);
          const rightKeyIndex = rightMeta.columns.indexOf(rightKey);
          const isLeftOuter = joinType === "LEFT" || joinType === "FULL";
          const isRightOuter = joinType === "RIGHT" || joinType === "FULL";
          const matchedRightRows = isRightOuter ? /* @__PURE__ */ new Set() : null;
          const bothSidesPartitions = new Set(
            [...leftMeta.partitionsUsed].filter((p) => rightMeta.partitionsUsed.has(p))
          );
          for (const partitionId of bothSidesPartitions) {
            if (totalYielded >= limit) break;
            const leftPartition = await this._loadPartition("left", partitionId, leftMeta.columns);
            if (leftPartition.length === 0) continue;
            const hashMap = /* @__PURE__ */ new Map();
            const matchedLeftIndices = isLeftOuter ? /* @__PURE__ */ new Set() : null;
            for (let i = 0; i < leftPartition.length; i++) {
              const row = leftPartition[i];
              const key = row[leftKeyIndex];
              if (key !== null && key !== void 0) {
                if (!hashMap.has(key)) hashMap.set(key, []);
                hashMap.get(key).push({ row, index: i });
              }
            }
            const rightPartition = await this._loadPartition("right", partitionId, rightMeta.columns);
            const chunk = [];
            for (let rightIdx = 0; rightIdx < rightPartition.length; rightIdx++) {
              if (totalYielded >= limit) break;
              const rightRow = rightPartition[rightIdx];
              const key = rightRow[rightKeyIndex];
              const leftEntries = hashMap.get(key);
              if (leftEntries) {
                if (matchedRightRows) {
                  matchedRightRows.add(`${partitionId}_${rightIdx}`);
                }
                for (const { row: leftRow, index: leftIdx } of leftEntries) {
                  if (totalYielded >= limit) break;
                  if (matchedLeftIndices) {
                    matchedLeftIndices.add(leftIdx);
                  }
                  chunk.push([...leftRow, ...rightRow]);
                  totalYielded++;
                  if (chunk.length >= this.chunkSize) {
                    yield { columns: resultColumns, rows: chunk.splice(0) };
                  }
                }
              }
            }
            if (isLeftOuter && matchedLeftIndices) {
              for (let i = 0; i < leftPartition.length; i++) {
                if (totalYielded >= limit) break;
                if (!matchedLeftIndices.has(i)) {
                  chunk.push([...leftPartition[i], ...rightNulls]);
                  totalYielded++;
                  if (chunk.length >= this.chunkSize) {
                    yield { columns: resultColumns, rows: chunk.splice(0) };
                  }
                }
              }
            }
            if (chunk.length > 0) {
              yield { columns: resultColumns, rows: chunk };
            }
          }
          if (isLeftOuter) {
            for (const partitionId of leftMeta.partitionsUsed) {
              if (totalYielded >= limit) break;
              if (bothSidesPartitions.has(partitionId)) continue;
              const leftPartition = await this._loadPartition("left", partitionId, leftMeta.columns);
              const chunk = [];
              for (const leftRow of leftPartition) {
                if (totalYielded >= limit) break;
                chunk.push([...leftRow, ...rightNulls]);
                totalYielded++;
                if (chunk.length >= this.chunkSize) {
                  yield { columns: resultColumns, rows: chunk.splice(0) };
                }
              }
              if (chunk.length > 0) {
                yield { columns: resultColumns, rows: chunk };
              }
            }
          }
          if (isRightOuter) {
            for (const partitionId of rightMeta.partitionsUsed) {
              if (totalYielded >= limit) break;
              const rightPartition = await this._loadPartition("right", partitionId, rightMeta.columns);
              const chunk = [];
              for (let rightIdx = 0; rightIdx < rightPartition.length; rightIdx++) {
                if (totalYielded >= limit) break;
                const rowKey = `${partitionId}_${rightIdx}`;
                if (!matchedRightRows.has(rowKey)) {
                  chunk.push([...leftNulls, ...rightPartition[rightIdx]]);
                  totalYielded++;
                  if (chunk.length >= this.chunkSize) {
                    yield { columns: resultColumns, rows: chunk.splice(0) };
                  }
                }
              }
              if (chunk.length > 0) {
                yield { columns: resultColumns, rows: chunk };
              }
            }
          }
        } finally {
          await this.cleanup();
        }
      }
      /**
       * Partition a stream of data to OPFS files by hash(key)
       * @param {AsyncGenerator} stream - Data stream to partition
       * @param {string} keyColumn - Column name to use as partition key
       * @param {string} side - 'left' or 'right' table
       * @param {boolean} collectKeys - If true, collect unique key values for semi-join optimization
       */
      async _partitionToOPFS(stream, keyColumn, side, collectKeys = false) {
        const partitionBuffers = /* @__PURE__ */ new Map();
        const flushThreshold = 500;
        let columns = null;
        let keyIndex = -1;
        let totalRows = 0;
        const partitionsUsed = /* @__PURE__ */ new Set();
        const collectedKeys = collectKeys ? /* @__PURE__ */ new Set() : null;
        for await (const chunk of stream) {
          if (!columns) {
            columns = chunk.columns;
            keyIndex = columns.indexOf(keyColumn);
            if (keyIndex === -1) {
              throw new Error(`Join key column '${keyColumn}' not found in columns: ${columns.join(", ")}`);
            }
          }
          for (const row of chunk.rows) {
            const key = row[keyIndex];
            const partitionId = this._hashToPartition(key);
            partitionsUsed.add(partitionId);
            if (collectKeys && key !== null && key !== void 0) {
              collectedKeys.add(key);
            }
            if (!partitionBuffers.has(partitionId)) {
              partitionBuffers.set(partitionId, []);
            }
            partitionBuffers.get(partitionId).push(row);
            totalRows++;
            if (partitionBuffers.get(partitionId).length >= flushThreshold) {
              await this._appendToPartition(side, partitionId, partitionBuffers.get(partitionId));
              partitionBuffers.set(partitionId, []);
            }
          }
        }
        for (const [partitionId, rows] of partitionBuffers) {
          if (rows.length > 0) {
            await this._appendToPartition(side, partitionId, rows);
          }
        }
        if (side === "left") {
          this.stats.leftRowsWritten = totalRows;
        } else {
          this.stats.rightRowsWritten = totalRows;
        }
        return { columns, totalRows, partitionsUsed, collectedKeys };
      }
      /**
       * Hash a value to a partition number
       */
      _hashToPartition(value) {
        if (value === null || value === void 0) {
          return 0;
        }
        const str = String(value);
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
          const char = str.charCodeAt(i);
          hash = (hash << 5) - hash + char;
          hash = hash & hash;
        }
        return Math.abs(hash) % this.numPartitions;
      }
      /**
       * Append rows to a partition file in OPFS
       */
      async _appendToPartition(side, partitionId, rows) {
        const path = `${this.basePath}/${side}/partition_${String(partitionId).padStart(3, "0")}.jsonl`;
        const jsonl = rows.map((row) => JSON.stringify(row)).join("\n") + "\n";
        const data = new TextEncoder().encode(jsonl);
        const existing = await this.storage.load(path);
        if (existing) {
          const combined = new Uint8Array(existing.length + data.length);
          combined.set(existing);
          combined.set(data, existing.length);
          await this.storage.save(path, combined);
          this.stats.bytesWrittenToOPFS += data.length;
        } else {
          await this.storage.save(path, data);
          this.stats.bytesWrittenToOPFS += data.length;
        }
        this.stats.partitionsUsed.add(partitionId);
      }
      /**
       * Load a partition from OPFS
       */
      async _loadPartition(side, partitionId, columns) {
        const path = `${this.basePath}/${side}/partition_${String(partitionId).padStart(3, "0")}.jsonl`;
        const data = await this.storage.load(path);
        if (!data) return [];
        this.stats.bytesReadFromOPFS += data.length;
        const text = new TextDecoder().decode(data);
        const lines = text.trim().split("\n").filter((line) => line);
        return lines.map((line) => JSON.parse(line));
      }
      /**
       * Get execution statistics
       */
      getStats() {
        return {
          ...this.stats,
          partitionsUsed: this.stats.partitionsUsed.size,
          bytesWrittenMB: (this.stats.bytesWrittenToOPFS / 1024 / 1024).toFixed(2),
          bytesReadMB: (this.stats.bytesReadFromOPFS / 1024 / 1024).toFixed(2)
        };
      }
      /**
       * Cleanup temp files
       */
      async cleanup() {
        try {
          await this.storage.deleteDir(this.basePath);
        } catch {
        }
      }
    };
    LocalDatabase = class {
      /**
       * Create a LocalDatabase instance.
       * All operations are executed in a SharedWorker for OPFS sync access.
       *
       * @param {string} name - Database name
       */
      constructor(name) {
        this.name = name;
        this._ready = false;
      }
      /**
       * Open or create the database.
       * Connects to the SharedWorker.
       *
       * @returns {Promise<LocalDatabase>}
       */
      async open() {
        if (this._ready) return this;
        await workerRPC("db:open", { name: this.name });
        this._ready = true;
        return this;
      }
      async _ensureOpen() {
        if (!this._ready) {
          await this.open();
        }
      }
      /**
       * CREATE TABLE
       * @param {string} tableName - Table name
       * @param {Array} columns - Column definitions [{name, type, primaryKey?, vectorDim?}]
       * @param {boolean} ifNotExists - If true, don't error if table already exists
       * @returns {Promise<{success: boolean, table?: string, existed?: boolean}>}
       */
      async createTable(tableName, columns, ifNotExists = false) {
        await this._ensureOpen();
        return workerRPC("db:createTable", {
          db: this.name,
          tableName,
          columns,
          ifNotExists
        });
      }
      /**
       * DROP TABLE
       * @param {string} tableName - Table name
       * @param {boolean} ifExists - If true, don't error if table doesn't exist
       * @returns {Promise<{success: boolean, table?: string, existed?: boolean}>}
       */
      async dropTable(tableName, ifExists = false) {
        await this._ensureOpen();
        return workerRPC("db:dropTable", {
          db: this.name,
          tableName,
          ifExists
        });
      }
      /**
       * INSERT INTO
       * @param {string} tableName - Table name
       * @param {Array} rows - Array of row objects [{col1: val1, col2: val2}, ...]
       * @returns {Promise<{success: boolean, inserted: number}>}
       */
      async insert(tableName, rows) {
        await this._ensureOpen();
        return workerRPC("db:insert", {
          db: this.name,
          tableName,
          rows
        });
      }
      /**
       * Flush all buffered writes to OPFS
       * @returns {Promise<void>}
       */
      async flush() {
        await this._ensureOpen();
        return workerRPC("db:flush", { db: this.name });
      }
      /**
       * DELETE FROM
       * @param {string} tableName - Table name
       * @param {Object} where - WHERE clause as parsed AST (column/op/value)
       * @returns {Promise<{success: boolean, deleted: number}>}
       */
      async delete(tableName, where = null) {
        await this._ensureOpen();
        return workerRPC("db:delete", {
          db: this.name,
          tableName,
          where
        });
      }
      /**
       * UPDATE
       * @param {string} tableName - Table name
       * @param {Object} updates - Column updates {col1: newVal, col2: newVal}
       * @param {Object} where - WHERE clause as parsed AST (column/op/value)
       * @returns {Promise<{success: boolean, updated: number}>}
       */
      async update(tableName, updates, where = null) {
        await this._ensureOpen();
        return workerRPC("db:update", {
          db: this.name,
          tableName,
          updates,
          where
        });
      }
      /**
       * SELECT (query)
       * @param {string} tableName - Table name
       * @param {Object} options - Query options {columns, where, limit, offset, orderBy}
       * @returns {Promise<Array>}
       */
      async select(tableName, options = {}) {
        await this._ensureOpen();
        const rpcOptions = { ...options };
        delete rpcOptions.where;
        return workerRPC("db:select", {
          db: this.name,
          tableName,
          options: rpcOptions,
          where: options.whereAST || null
        });
      }
      /**
       * Execute SQL statement
       * @param {string} sql - SQL statement
       * @returns {Promise<any>} Result
       */
      async exec(sql) {
        await this._ensureOpen();
        return workerRPC("db:exec", { db: this.name, sql });
      }
      /**
       * Get table info
       * @param {string} tableName - Table name
       * @returns {Promise<Object>} Table state
       */
      async getTable(tableName) {
        await this._ensureOpen();
        return workerRPC("db:getTable", { db: this.name, tableName });
      }
      /**
       * List all tables
       * @returns {Promise<string[]>} Table names
       */
      async listTables() {
        await this._ensureOpen();
        return workerRPC("db:listTables", { db: this.name });
      }
      /**
       * Compact the database (merge fragments, remove deleted rows)
       * @returns {Promise<{success: boolean, compacted: number}>}
       */
      async compact() {
        await this._ensureOpen();
        return workerRPC("db:compact", { db: this.name });
      }
      /**
       * Streaming scan - yields batches of rows for memory-efficient processing
       * @param {string} tableName - Table name
       * @param {Object} options - Scan options {batchSize, columns}
       * @yields {Object[]} Batch of rows
       *
       * @example
       * for await (const batch of db.scan('users', { batchSize: 1000 })) {
       *   processBatch(batch);
       * }
       */
      async *scan(tableName, options = {}) {
        await this._ensureOpen();
        const streamId = await workerRPC("db:scanStart", {
          db: this.name,
          tableName,
          options
        });
        while (true) {
          const { batch, done } = await workerRPC("db:scanNext", {
            db: this.name,
            streamId
          });
          if (batch.length > 0) {
            yield batch;
          }
          if (done) break;
        }
      }
      /**
       * Close the database
       * @returns {Promise<void>}
       */
      async close() {
        await this._ensureOpen();
        await this.flush();
      }
    };
  }
});

// src/client/wasm/lanceql.js
var lanceql_exports = {};
__export(lanceql_exports, {
  LanceFileWriter: () => LanceFileWriter,
  LanceQL: () => LanceQL,
  LocalSQLParser: () => LocalSQLParser,
  wasmUtils: () => wasmUtils
});
var LocalSQLParser, E, D, _w, _m, _p, _M, _g, _ensure, _x, readStr, readBytes, LanceFileWriter, wasmUtils, _createLanceqlMethods, LanceQL;
var init_lanceql = __esm({
  "src/client/wasm/lanceql.js"() {
    init_accelerator();
    LocalSQLParser = class {
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
          default:
            throw new Error(`Unexpected token: ${token.type}`);
        }
      }
      parseCreate() {
        this.expect(TokenType.CREATE);
        this.expect(TokenType.TABLE);
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
        return { type: "create_table", table: tableName, columns };
      }
      parseDataType() {
        const token = this.advance();
        let type = token.value || token.type;
        if (type === "VECTOR" && this.match(TokenType.LPAREN)) {
          const dim = this.expect(TokenType.NUMBER).value;
          this.expect(TokenType.RPAREN);
          return { type: "vector", dim: parseInt(dim) };
        }
        if ((type === "VARCHAR" || type === "TEXT") && this.match(TokenType.LPAREN)) {
          this.expect(TokenType.NUMBER);
          this.expect(TokenType.RPAREN);
        }
        return type;
      }
      parseDrop() {
        this.expect(TokenType.DROP);
        this.expect(TokenType.TABLE);
        const tableName = this.expect(TokenType.IDENTIFIER).value;
        return { type: "drop_table", table: tableName };
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
        return { type: "insert", table: tableName, columns, rows };
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
        this.expect(TokenType.SET);
        const set = {};
        do {
          const col = this.expect(TokenType.IDENTIFIER).value;
          this.expect(TokenType.EQ);
          set[col] = this.parseValue();
        } while (this.match(TokenType.COMMA));
        let where = null;
        if (this.match(TokenType.WHERE)) {
          where = this.parseWhereExpr();
        }
        return { type: "update", table: tableName, set, where };
      }
      parseDelete() {
        this.expect(TokenType.DELETE);
        this.expect(TokenType.FROM);
        const tableName = this.expect(TokenType.IDENTIFIER).value;
        let where = null;
        if (this.match(TokenType.WHERE)) {
          where = this.parseWhereExpr();
        }
        return { type: "delete", table: tableName, where };
      }
      parseSelect() {
        this.expect(TokenType.SELECT);
        const columns = [];
        if (this.match(TokenType.STAR)) {
          columns.push("*");
        } else {
          columns.push(this.expect(TokenType.IDENTIFIER).value);
          while (this.match(TokenType.COMMA)) {
            columns.push(this.expect(TokenType.IDENTIFIER).value);
          }
        }
        this.expect(TokenType.FROM);
        const tableName = this.expect(TokenType.IDENTIFIER).value;
        let where = null;
        if (this.match(TokenType.WHERE)) {
          where = this.parseWhereExpr();
        }
        let orderBy = null;
        if (this.match(TokenType.ORDER)) {
          this.expect(TokenType.BY);
          const column = this.expect(TokenType.IDENTIFIER).value;
          const desc = !!this.match(TokenType.DESC);
          if (!desc) this.match(TokenType.ASC);
          orderBy = { column, desc };
        }
        let limit = null;
        if (this.match(TokenType.LIMIT)) {
          limit = parseInt(this.expect(TokenType.NUMBER).value);
        }
        let offset = null;
        if (this.match(TokenType.OFFSET)) {
          offset = parseInt(this.expect(TokenType.NUMBER).value);
        }
        return { type: "select", table: tableName, columns, where, orderBy, limit, offset };
      }
      parseWhereExpr() {
        return this.parseOrExpr();
      }
      parseOrExpr() {
        let left = this.parseAndExpr();
        while (this.match(TokenType.OR)) {
          const right = this.parseAndExpr();
          left = { op: "OR", left, right };
        }
        return left;
      }
      parseAndExpr() {
        let left = this.parseComparison();
        while (this.match(TokenType.AND)) {
          const right = this.parseComparison();
          left = { op: "AND", left, right };
        }
        return left;
      }
      parseComparison() {
        const column = this.expect(TokenType.IDENTIFIER).value;
        let op;
        if (this.match(TokenType.EQ)) op = "=";
        else if (this.match(TokenType.NE)) op = "!=";
        else if (this.match(TokenType.LT)) op = "<";
        else if (this.match(TokenType.LE)) op = "<=";
        else if (this.match(TokenType.GT)) op = ">";
        else if (this.match(TokenType.GE)) op = ">=";
        else if (this.match(TokenType.LIKE)) op = "LIKE";
        else throw new Error(`Expected comparison operator`);
        const value = this.parseValue();
        return { op, column, value };
      }
    };
    E = new TextEncoder();
    D = new TextDecoder();
    _p = 0;
    _M = 0;
    _g = () => {
      if (!_p || !_M) return null;
      return new Uint8Array(_m.buffer, _p, _M);
    };
    _ensure = (size) => {
      if (_p && size <= _M) return true;
      if (_p && _w.free) _w.free(_p, _M);
      _M = Math.max(size + 1024, 4096);
      _p = _w.alloc(_M);
      return _p !== 0;
    };
    _x = (a) => {
      if (a instanceof Uint8Array) {
        if (!_ensure(a.length)) return [a];
        _g().set(a);
        return [_p, a.length];
      }
      if (typeof a !== "string") return [a];
      const b = E.encode(a);
      if (!_ensure(b.length)) return [a];
      _g().set(b);
      return [_p, b.length];
    };
    readStr = (ptr, len) => D.decode(new Uint8Array(_m.buffer, ptr, len));
    readBytes = (ptr, len) => new Uint8Array(_m.buffer, ptr, len).slice();
    LanceFileWriter = class {
      constructor(schema) {
        this.schema = schema;
        this.columns = /* @__PURE__ */ new Map();
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
        if (!_w?.fragmentBegin) {
          return this._buildJson();
        }
        const estimatedSize = Math.max(64 * 1024, this.rowCount * 1024);
        if (!_w.fragmentBegin(estimatedSize)) {
          throw new Error("Failed to initialize WASM fragment writer");
        }
        for (const col of this.schema) {
          const values = this.columns.get(col.name) || [];
          this._addColumn(col, values);
        }
        const finalSize = _w.fragmentEnd();
        if (finalSize === 0) {
          throw new Error("Failed to finalize fragment");
        }
        const bufferPtr = _w.writerGetBuffer();
        if (!bufferPtr) {
          throw new Error("Failed to get writer buffer");
        }
        return new Uint8Array(_m.buffer, bufferPtr, finalSize).slice();
      }
      _addColumn(col, values) {
        const type = (col.type || col.dataType || "string").toLowerCase();
        const nullable = col.nullable !== false;
        const nameBytes = E.encode(col.name);
        const namePtr = _w.alloc(nameBytes.length);
        new Uint8Array(_m.buffer, namePtr, nameBytes.length).set(nameBytes);
        let result = 0;
        switch (type) {
          case "int64":
          case "int":
          case "integer":
          case "bigint": {
            const arr = new BigInt64Array(values.map((v) => BigInt(v ?? 0)));
            const ptr = _w.alloc(arr.byteLength);
            new BigInt64Array(_m.buffer, ptr, values.length).set(arr);
            result = _w.fragmentAddInt64Column(namePtr, nameBytes.length, ptr, values.length, nullable);
            _w.free(ptr, arr.byteLength);
            break;
          }
          case "int32": {
            const arr = new Int32Array(values.map((v) => v ?? 0));
            const ptr = _w.alloc(arr.byteLength);
            new Int32Array(_m.buffer, ptr, values.length).set(arr);
            result = _w.fragmentAddInt32Column(namePtr, nameBytes.length, ptr, values.length, nullable);
            _w.free(ptr, arr.byteLength);
            break;
          }
          case "float64":
          case "float":
          case "double": {
            const arr = new Float64Array(values.map((v) => v ?? 0));
            const ptr = _w.alloc(arr.byteLength);
            new Float64Array(_m.buffer, ptr, values.length).set(arr);
            result = _w.fragmentAddFloat64Column(namePtr, nameBytes.length, ptr, values.length, nullable);
            _w.free(ptr, arr.byteLength);
            break;
          }
          case "float32": {
            const arr = new Float32Array(values.map((v) => v ?? 0));
            const ptr = _w.alloc(arr.byteLength);
            new Float32Array(_m.buffer, ptr, values.length).set(arr);
            result = _w.fragmentAddFloat32Column(namePtr, nameBytes.length, ptr, values.length, nullable);
            _w.free(ptr, arr.byteLength);
            break;
          }
          case "string":
          case "text":
          case "varchar": {
            let currentOffset = 0;
            const offsets = new Uint32Array(values.length + 1);
            const allBytes = [];
            for (let i = 0; i < values.length; i++) {
              offsets[i] = currentOffset;
              const bytes = E.encode(String(values[i] ?? ""));
              allBytes.push(...bytes);
              currentOffset += bytes.length;
            }
            offsets[values.length] = currentOffset;
            const strData = new Uint8Array(allBytes);
            const strPtr = _w.alloc(strData.length);
            new Uint8Array(_m.buffer, strPtr, strData.length).set(strData);
            const offPtr = _w.alloc(offsets.byteLength);
            new Uint32Array(_m.buffer, offPtr, offsets.length).set(offsets);
            result = _w.fragmentAddStringColumn(namePtr, nameBytes.length, strPtr, strData.length, offPtr, values.length, nullable);
            _w.free(strPtr, strData.length);
            _w.free(offPtr, offsets.byteLength);
            break;
          }
          case "bool":
          case "boolean": {
            const byteCount = Math.ceil(values.length / 8);
            const packed = new Uint8Array(byteCount);
            for (let i = 0; i < values.length; i++) {
              if (values[i]) packed[Math.floor(i / 8)] |= 1 << i % 8;
            }
            const ptr = _w.alloc(packed.length);
            new Uint8Array(_m.buffer, ptr, packed.length).set(packed);
            result = _w.fragmentAddBoolColumn(namePtr, nameBytes.length, ptr, packed.length, values.length, nullable);
            _w.free(ptr, packed.length);
            break;
          }
          case "vector": {
            const dim = col.vectorDim || (values[0]?.length || 0);
            const allFloats = [];
            for (const v of values) {
              if (Array.isArray(v)) {
                allFloats.push(...v);
              } else {
                for (let i = 0; i < dim; i++) allFloats.push(0);
              }
            }
            const arr = new Float32Array(allFloats);
            const ptr = _w.alloc(arr.byteLength);
            new Float32Array(_m.buffer, ptr, allFloats.length).set(arr);
            result = _w.fragmentAddVectorColumn(namePtr, nameBytes.length, ptr, allFloats.length, dim, nullable);
            _w.free(ptr, arr.byteLength);
            break;
          }
          default:
            _w.free(namePtr, nameBytes.length);
            return this._addColumn({ ...col, type: "string" }, values);
        }
        _w.free(namePtr, nameBytes.length);
        if (!result) {
          throw new Error(`Failed to add column '${col.name}'`);
        }
      }
      _buildJson() {
        const data = {
          schema: this.schema,
          columns: Object.fromEntries(this.columns),
          rowCount: this.rowCount,
          format: "json"
        };
        return E.encode(JSON.stringify(data));
      }
    };
    wasmUtils = {
      readStr,
      readBytes,
      encoder: E,
      decoder: D,
      getMemory: () => _m,
      getExports: () => _w
    };
    _createLanceqlMethods = (proxy) => ({
      /**
       * Get the library version.
       * @returns {string} Version string like "0.1.0"
       */
      getVersion() {
        const v = _w.getVersion();
        const major = v >> 16 & 255;
        const minor = v >> 8 & 255;
        const patch = v & 255;
        return `${major}.${minor}.${patch}`;
      },
      /**
       * Open a Lance file from an ArrayBuffer (local file).
       * @param {ArrayBuffer} data - The Lance file data
       * @returns {LanceFile}
       */
      open(data) {
        return new LanceFile(proxy, data);
      },
      /**
       * Open a Lance file from a URL using HTTP Range requests.
       * @param {string} url - URL to the Lance file
       * @returns {Promise<RemoteLanceFile>}
       */
      async openUrl(url) {
        await getWebGPUAccelerator().init();
        return await RemoteLanceFile.open(proxy, url);
      },
      /**
       * Open a Lance dataset from a base URL using HTTP Range requests.
       * @param {string} baseUrl - Base URL to the Lance dataset
       * @param {object} [options] - Options for opening
       * @param {number} [options.version] - Specific version to load
       * @returns {Promise<RemoteLanceDataset>}
       */
      async openDataset(baseUrl, options = {}) {
        await getWebGPUAccelerator().init();
        return await RemoteLanceDataset.open(proxy, baseUrl, options);
      },
      /**
       * Parse footer from Lance file data.
       * @param {ArrayBuffer} data
       * @returns {{numColumns: number, majorVersion: number, minorVersion: number} | null}
       */
      parseFooter(data) {
        const bytes = new Uint8Array(data);
        const ptr = _w.alloc(bytes.length);
        if (!ptr) return null;
        try {
          new Uint8Array(_m.buffer).set(bytes, ptr);
          const numColumns = _w.parseFooterGetColumns(ptr, bytes.length);
          const majorVersion = _w.parseFooterGetMajorVersion(ptr, bytes.length);
          const minorVersion = _w.parseFooterGetMinorVersion(ptr, bytes.length);
          if (numColumns === 0 && majorVersion === 0) {
            return null;
          }
          return { numColumns, majorVersion, minorVersion };
        } finally {
          _w.free(ptr, bytes.length);
        }
      },
      /**
       * Check if data is a valid Lance file.
       * @param {ArrayBuffer} data
       * @returns {boolean}
       */
      isValidLanceFile(data) {
        const bytes = new Uint8Array(data);
        const ptr = _w.alloc(bytes.length);
        if (!ptr) return false;
        try {
          new Uint8Array(_m.buffer).set(bytes, ptr);
          return _w.isValidLanceFile(ptr, bytes.length) === 1;
        } finally {
          _w.free(ptr, bytes.length);
        }
      },
      /**
       * Create a new LanceDatabase for multi-table queries with JOINs.
       * @returns {LanceDatabase}
       */
      createDatabase() {
        if (typeof window !== "undefined") {
          window.lanceql = proxy;
        } else if (typeof globalThis !== "undefined") {
          globalThis.lanceql = proxy;
        }
        return new LanceDatabase();
      }
    });
    LanceQL = class {
      /**
       * Load LanceQL from a WASM file path or URL.
       * Returns Immer-style proxy with auto string/bytes marshalling.
       * @param {string} wasmPath - Path to the lanceql.wasm file
       * @returns {Promise<LanceQL>}
       */
      static async load(wasmPath = "./lanceql.wasm") {
        const response = await fetch(wasmPath);
        const wasmBytes = await response.arrayBuffer();
        const wasmModule = await WebAssembly.instantiate(wasmBytes, {});
        _w = wasmModule.instance.exports;
        _m = _w.memory;
        let _methods = null;
        const proxy = new Proxy({}, {
          get(_, n) {
            if (!_methods) _methods = _createLanceqlMethods(proxy);
            if (n in _methods) return _methods[n];
            if (n === "memory") return _m;
            if (n === "raw") return _w;
            if (n === "wasm") return _w;
            if (typeof _w[n] === "function") {
              return (...a) => _w[n](...a.flatMap(_x));
            }
            return _w[n];
          }
        });
        return proxy;
      }
    };
  }
});

// src/client/index.js
var index_exports = {};
__export(index_exports, {
  ChunkedLanceReader: () => ChunkedLanceReader,
  CostModel: () => CostModel,
  DataFrame: () => DataFrame2,
  Database: () => Database,
  DatasetStorage: () => DatasetStorage,
  GPUAggregator: () => GPUAggregator,
  GPUGrouper: () => GPUGrouper,
  GPUJoiner: () => GPUJoiner,
  GPUSorter: () => GPUSorter,
  GPUVectorSearch: () => GPUVectorSearch,
  HotTierCache: () => HotTierCache,
  IVFIndex: () => IVFIndex,
  LRUCache: () => LRUCache2,
  LanceData: () => LanceData,
  LanceDataBase: () => LanceDataBase,
  LanceDatabase: () => LanceDatabase2,
  LanceFile: () => LanceFile2,
  LanceFileWriter: () => LanceFileWriter,
  LanceQL: () => LanceQL,
  LocalDatabase: () => LocalDatabase,
  LocalSQLParser: () => LocalSQLParser,
  MemoryManager: () => MemoryManager,
  MemoryTable: () => MemoryTable2,
  MetadataCache: () => MetadataCache,
  OPFSFileReader: () => OPFSFileReader,
  OPFSJoinExecutor: () => OPFSJoinExecutor,
  OPFSLanceData: () => OPFSLanceData,
  OPFSStorage: () => OPFSStorage2,
  ProtobufEncoder: () => ProtobufEncoder,
  PureLanceWriter: () => PureLanceWriter2,
  QueryPlanner: () => QueryPlanner,
  RemoteLanceData: () => RemoteLanceData,
  RemoteLanceDataset: () => RemoteLanceDataset2,
  RemoteLanceFile: () => RemoteLanceFile2,
  SQLExecutor: () => SQLExecutor,
  SQLLexer: () => SQLLexer,
  SQLParser: () => SQLParser,
  SharedVectorStore: () => SharedVectorStore,
  Statement: () => Statement,
  StatisticsManager: () => StatisticsManager,
  Store: () => Store,
  TableRef: () => TableRef,
  Vault: () => Vault,
  WebGPUAccelerator: () => WebGPUAccelerator,
  WorkerPool: () => WorkerPool,
  default: () => LanceQL,
  getGPUAggregator: () => getGPUAggregator,
  getGPUGrouper: () => getGPUGrouper,
  getGPUJoiner: () => getGPUJoiner,
  getGPUSorter: () => getGPUSorter,
  getGPUVectorSearch: () => getGPUVectorSearch,
  getHotTierCache: () => getHotTierCache,
  getWebGPUAccelerator: () => getWebGPUAccelerator,
  initSqlJs: () => initSqlJs,
  lanceStore: () => lanceStore,
  vault: () => vault,
  wasmUtils: () => wasmUtils
});
module.exports = __toCommonJS(index_exports);
init_metadata_cache();
init_lru_cache();
init_hot_tier_cache();
init_accelerator();

// src/client/gpu/aggregator.js
var GPUAggregator = class {
  constructor() {
    this.device = null;
    this.pipelines = /* @__PURE__ */ new Map();
    this.available = false;
    this._initPromise = null;
  }
  async init() {
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._doInit();
    return this._initPromise;
  }
  async _doInit() {
    if (!navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return false;
      this.device = await adapter.requestDevice({
        requiredLimits: { maxStorageBufferBindingSize: 256 * 1024 * 1024 }
      });
      this._compileShaders();
      this.available = true;
      console.log("[GPUAggregator] Initialized");
      return true;
    } catch (e) {
      console.warn("[GPUAggregator] Init failed:", e);
      return false;
    }
  }
  _compileShaders() {
    const code = `
struct P { size: u32, wg: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> i: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
var<workgroup> s: array<f32, 256>;

@compute @workgroup_size(256)
fn sum(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    s[l.x] = select(0.0, i[g.x], g.x < p.size);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] += s[l.x + t]; } workgroupBarrier(); }
    if (l.x == 0u) { o[w.x] = s[0]; }
}

@compute @workgroup_size(256)
fn sum_f(@builtin(local_invocation_id) l: vec3<u32>) {
    s[l.x] = select(0.0, i[l.x], l.x < p.wg);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] += s[l.x + t]; } workgroupBarrier(); }
    if (l.x == 0u) { o[0] = s[0]; }
}

@compute @workgroup_size(256)
fn min_r(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    s[l.x] = select(3.4e+38, i[g.x], g.x < p.size);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] = min(s[l.x], s[l.x + t]); } workgroupBarrier(); }
    if (l.x == 0u) { o[w.x] = s[0]; }
}

@compute @workgroup_size(256)
fn min_f(@builtin(local_invocation_id) l: vec3<u32>) {
    s[l.x] = select(3.4e+38, i[l.x], l.x < p.wg);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] = min(s[l.x], s[l.x + t]); } workgroupBarrier(); }
    if (l.x == 0u) { o[0] = s[0]; }
}

@compute @workgroup_size(256)
fn max_r(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    s[l.x] = select(-3.4e+38, i[g.x], g.x < p.size);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] = max(s[l.x], s[l.x + t]); } workgroupBarrier(); }
    if (l.x == 0u) { o[w.x] = s[0]; }
}

@compute @workgroup_size(256)
fn max_f(@builtin(local_invocation_id) l: vec3<u32>) {
    s[l.x] = select(-3.4e+38, i[l.x], l.x < p.wg);
    workgroupBarrier();
    for (var t: u32 = 128u; t > 0u; t >>= 1u) { if (l.x < t) { s[l.x] = max(s[l.x], s[l.x + t]); } workgroupBarrier(); }
    if (l.x == 0u) { o[0] = s[0]; }
}`;
    const module2 = this.device.createShaderModule({ code });
    for (const [name, entry] of [["sum", "sum"], ["sum_final", "sum_f"], ["min", "min_r"], ["min_final", "min_f"], ["max", "max_r"], ["max_final", "max_f"]]) {
      this.pipelines.set(name, this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: entry } }));
    }
  }
  isAvailable() {
    return this.available;
  }
  async sum(values) {
    if (!this.available || values.length < 1e3) return this._cpuSum(values);
    return this._gpuReduce(values, "sum");
  }
  async min(values) {
    if (!this.available || values.length < 1e3) return values.length ? Math.min(...values) : null;
    return this._gpuReduce(values, "min");
  }
  async max(values) {
    if (!this.available || values.length < 1e3) return values.length ? Math.max(...values) : null;
    return this._gpuReduce(values, "max");
  }
  async avg(values) {
    if (values.length === 0) return null;
    const sum = await this.sum(values);
    return sum / values.length;
  }
  count(values) {
    return values.length;
  }
  async _gpuReduce(values, op) {
    const n = values.length, wgSize = 256, numWg = Math.ceil(n / wgSize);
    const input = values instanceof Float32Array ? values : new Float32Array(values);
    const inputBuf = this.device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(inputBuf, 0, input);
    const partialBuf = this.device.createBuffer({ size: numWg * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const outBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const stageBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const paramsBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([n, numWg]));
    const finalParamsBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(finalParamsBuf, 0, new Uint32Array([numWg, numWg]));
    const p1 = this.pipelines.get(op), p2 = this.pipelines.get(op + "_final");
    const bg1 = this.device.createBindGroup({ layout: p1.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: paramsBuf } }, { binding: 1, resource: { buffer: inputBuf } }, { binding: 2, resource: { buffer: partialBuf } }] });
    const bg2 = this.device.createBindGroup({ layout: p2.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: finalParamsBuf } }, { binding: 1, resource: { buffer: partialBuf } }, { binding: 2, resource: { buffer: outBuf } }] });
    const enc = this.device.createCommandEncoder();
    const c1 = enc.beginComputePass();
    c1.setPipeline(p1);
    c1.setBindGroup(0, bg1);
    c1.dispatchWorkgroups(numWg);
    c1.end();
    const c2 = enc.beginComputePass();
    c2.setPipeline(p2);
    c2.setBindGroup(0, bg2);
    c2.dispatchWorkgroups(1);
    c2.end();
    enc.copyBufferToBuffer(outBuf, 0, stageBuf, 0, 4);
    this.device.queue.submit([enc.finish()]);
    await stageBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stageBuf.getMappedRange())[0];
    stageBuf.unmap();
    inputBuf.destroy();
    partialBuf.destroy();
    outBuf.destroy();
    stageBuf.destroy();
    paramsBuf.destroy();
    finalParamsBuf.destroy();
    return result;
  }
  _cpuSum(values) {
    let s = 0;
    for (let i = 0; i < values.length; i++) s += values[i];
    return s;
  }
};
var _gpuAggregator = null;
function getGPUAggregator() {
  if (!_gpuAggregator) _gpuAggregator = new GPUAggregator();
  return _gpuAggregator;
}

// src/client/gpu/joiner.js
var GPUJoiner = class {
  constructor() {
    this.device = null;
    this.pipelines = /* @__PURE__ */ new Map();
    this.available = false;
    this._initPromise = null;
  }
  async init() {
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._doInit();
    return this._initPromise;
  }
  async _doInit() {
    if (!navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return false;
      this.device = await adapter.requestDevice({
        requiredLimits: { maxStorageBufferBindingSize: 256 * 1024 * 1024 }
      });
      this._compileShaders();
      this.available = true;
      console.log("[GPUJoiner] Initialized");
      return true;
    } catch (e) {
      console.warn("[GPUJoiner] Init failed:", e);
      return false;
    }
  }
  _compileShaders() {
    const code = `
struct BP { size: u32, cap: u32 }
struct PP { left_size: u32, cap: u32, max_matches: u32 }
@group(0) @binding(0) var<uniform> bp: BP;
@group(0) @binding(1) var<storage, read> bkeys: array<u32>;
@group(0) @binding(2) var<storage, read_write> ht: array<atomic<u32>>;

fn fnv(k: u32) -> u32 {
    var h = 2166136261u;
    h ^= (k & 0xFFu); h *= 16777619u;
    h ^= ((k >> 8u) & 0xFFu); h *= 16777619u;
    h ^= ((k >> 16u) & 0xFFu); h *= 16777619u;
    h ^= ((k >> 24u) & 0xFFu); h *= 16777619u;
    return h;
}

@compute @workgroup_size(256)
fn build(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= bp.size) { return; }
    let k = bkeys[g.x];
    var s = fnv(k) % bp.cap;
    for (var p = 0u; p < bp.cap; p++) {
        let i = s * 2u;
        let o = atomicCompareExchangeWeak(&ht[i], 0xFFFFFFFFu, k);
        if (o.exchanged) { atomicStore(&ht[i + 1u], g.x); return; }
        s = (s + 1u) % bp.cap;
    }
}

@group(0) @binding(0) var<uniform> pp: PP;
@group(0) @binding(1) var<storage, read> lkeys: array<u32>;
@group(0) @binding(2) var<storage, read> pht: array<u32>;
@group(0) @binding(3) var<storage, read_write> matches: array<u32>;
@group(0) @binding(4) var<storage, read_write> mc: atomic<u32>;

@compute @workgroup_size(256)
fn probe(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= pp.left_size) { return; }
    let k = lkeys[g.x];
    var s = fnv(k) % pp.cap;
    for (var p = 0u; p < pp.cap; p++) {
        let i = s * 2u;
        let sk = pht[i];
        if (sk == 0xFFFFFFFFu) { return; }
        if (sk == k) {
            let ri = pht[i + 1u];
            let o = atomicAdd(&mc, 1u);
            if (o * 2u + 1u < pp.max_matches * 2u) {
                matches[o * 2u] = g.x;
                matches[o * 2u + 1u] = ri;
            }
        }
        s = (s + 1u) % pp.cap;
    }
}

struct IP { cap: u32 }
@group(0) @binding(0) var<uniform> ip: IP;
@group(0) @binding(1) var<storage, read_write> it: array<u32>;

@compute @workgroup_size(256)
fn init_t(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= ip.cap * 2u) { return; }
    it[g.x] = select(0u, 0xFFFFFFFFu, g.x % 2u == 0u);
}`;
    const module2 = this.device.createShaderModule({ code });
    this.pipelines.set("init", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "init_t" } }));
    this.pipelines.set("build", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "build" } }));
    this.pipelines.set("probe", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "probe" } }));
  }
  isAvailable() {
    return this.available;
  }
  async hashJoin(leftRows, rightRows, leftKey, rightKey) {
    const lSize = leftRows.length, rSize = rightRows.length;
    if (!this.available || lSize * rSize < 1e8) {
      return this._cpuHashJoin(leftRows, rightRows, leftKey, rightKey);
    }
    const lKeys = this._extractKeys(leftRows, leftKey);
    const rKeys = this._extractKeys(rightRows, rightKey);
    const cap = this._nextPow2(rSize * 2);
    const maxM = Math.max(lSize * 10, 1e5);
    const rKeysBuf = this._createBuf(rKeys, GPUBufferUsage.STORAGE);
    const lKeysBuf = this._createBuf(lKeys, GPUBufferUsage.STORAGE);
    const htBuf = this.device.createBuffer({ size: cap * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const matchBuf = this.device.createBuffer({ size: maxM * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const mcBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const stageBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const ipBuf = this._createUniform(new Uint32Array([cap]));
    const bpBuf = this._createUniform(new Uint32Array([rSize, cap]));
    const ppBuf = this._createUniform(new Uint32Array([lSize, cap, maxM]));
    const initP = this.pipelines.get("init"), buildP = this.pipelines.get("build"), probeP = this.pipelines.get("probe");
    const initBG = this.device.createBindGroup({ layout: initP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ipBuf } }, { binding: 1, resource: { buffer: htBuf } }] });
    const buildBG = this.device.createBindGroup({ layout: buildP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bpBuf } }, { binding: 1, resource: { buffer: rKeysBuf } }, { binding: 2, resource: { buffer: htBuf } }] });
    const probeBG = this.device.createBindGroup({ layout: probeP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ppBuf } }, { binding: 1, resource: { buffer: lKeysBuf } }, { binding: 2, resource: { buffer: htBuf } }, { binding: 3, resource: { buffer: matchBuf } }, { binding: 4, resource: { buffer: mcBuf } }] });
    const enc = this.device.createCommandEncoder();
    const p1 = enc.beginComputePass();
    p1.setPipeline(initP);
    p1.setBindGroup(0, initBG);
    p1.dispatchWorkgroups(Math.ceil(cap * 2 / 256));
    p1.end();
    const p2 = enc.beginComputePass();
    p2.setPipeline(buildP);
    p2.setBindGroup(0, buildBG);
    p2.dispatchWorkgroups(Math.ceil(rSize / 256));
    p2.end();
    const p3 = enc.beginComputePass();
    p3.setPipeline(probeP);
    p3.setBindGroup(0, probeBG);
    p3.dispatchWorkgroups(Math.ceil(lSize / 256));
    p3.end();
    enc.copyBufferToBuffer(mcBuf, 0, stageBuf, 0, 4);
    this.device.queue.submit([enc.finish()]);
    await stageBuf.mapAsync(GPUMapMode.READ);
    const mc = new Uint32Array(stageBuf.getMappedRange())[0];
    stageBuf.unmap();
    const actualM = Math.min(mc, maxM);
    const mStageBuf = this.device.createBuffer({ size: actualM * 2 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const cEnc = this.device.createCommandEncoder();
    cEnc.copyBufferToBuffer(matchBuf, 0, mStageBuf, 0, actualM * 2 * 4);
    this.device.queue.submit([cEnc.finish()]);
    await mStageBuf.mapAsync(GPUMapMode.READ);
    const mData = new Uint32Array(mStageBuf.getMappedRange().slice(0));
    mStageBuf.unmap();
    const lIdx = new Uint32Array(actualM), rIdx = new Uint32Array(actualM);
    for (let i = 0; i < actualM; i++) {
      lIdx[i] = mData[i * 2];
      rIdx[i] = mData[i * 2 + 1];
    }
    rKeysBuf.destroy();
    lKeysBuf.destroy();
    htBuf.destroy();
    matchBuf.destroy();
    mcBuf.destroy();
    stageBuf.destroy();
    mStageBuf.destroy();
    ipBuf.destroy();
    bpBuf.destroy();
    ppBuf.destroy();
    return { leftIndices: lIdx, rightIndices: rIdx, matchCount: actualM };
  }
  _cpuHashJoin(leftRows, rightRows, leftKey, rightKey) {
    const rMap = /* @__PURE__ */ new Map();
    for (let i = 0; i < rightRows.length; i++) {
      const k = this._hashKey(rightRows[i][rightKey]);
      if (!rMap.has(k)) rMap.set(k, []);
      rMap.get(k).push(i);
    }
    const lIdx = [], rIdx = [];
    for (let i = 0; i < leftRows.length; i++) {
      const k = this._hashKey(leftRows[i][leftKey]);
      for (const ri of rMap.get(k) || []) {
        lIdx.push(i);
        rIdx.push(ri);
      }
    }
    return { leftIndices: new Uint32Array(lIdx), rightIndices: new Uint32Array(rIdx), matchCount: lIdx.length };
  }
  _extractKeys(rows, key) {
    const keys = new Uint32Array(rows.length);
    for (let i = 0; i < rows.length; i++) keys[i] = this._hashKey(rows[i][key]);
    return keys;
  }
  _hashKey(v) {
    if (v == null) return 4294967294;
    if (typeof v === "number") return Number.isInteger(v) && v >= 0 && v < 4294967295 ? v >>> 0 : new Uint32Array(new Float32Array([v]).buffer)[0];
    if (typeof v === "string") {
      let h = 2166136261;
      for (let i = 0; i < v.length; i++) {
        h ^= v.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }
      return h >>> 0;
    }
    return this._hashKey(String(v));
  }
  _createBuf(data, usage) {
    const buf = this.device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }
  _createUniform(data) {
    const buf = this.device.createBuffer({ size: Math.max(data.byteLength, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }
  _nextPow2(n) {
    let p = 1;
    while (p < n) p *= 2;
    return p;
  }
};
var _gpuJoiner = null;
function getGPUJoiner() {
  if (!_gpuJoiner) _gpuJoiner = new GPUJoiner();
  return _gpuJoiner;
}

// src/client/gpu/sorter.js
var GPUSorter = class {
  constructor() {
    this.device = null;
    this.pipelines = /* @__PURE__ */ new Map();
    this.available = false;
    this._initPromise = null;
  }
  async init() {
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._doInit();
    return this._initPromise;
  }
  async _doInit() {
    if (!navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return false;
      this.device = await adapter.requestDevice({
        requiredLimits: { maxStorageBufferBindingSize: 256 * 1024 * 1024 }
      });
      this._compileShaders();
      this.available = true;
      console.log("[GPUSorter] Initialized");
      return true;
    } catch (e) {
      console.warn("[GPUSorter] Init failed:", e);
      return false;
    }
  }
  _compileShaders() {
    const code = `
struct LP { size: u32, stage: u32, step: u32, asc: u32 }
@group(0) @binding(0) var<uniform> lp: LP;
@group(0) @binding(1) var<storage, read_write> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> idx: array<u32>;
var<workgroup> sk: array<f32, 512>;
var<workgroup> si: array<u32, 512>;

fn cswap(i: u32, j: u32, d: bool) {
    let ki = sk[i]; let kj = sk[j];
    if (select(ki > kj, ki < kj, d)) {
        sk[i] = kj; sk[j] = ki;
        let t = si[i]; si[i] = si[j]; si[j] = t;
    }
}

@compute @workgroup_size(256)
fn local_sort(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let base = w.x * 512u; let t = l.x;
    let i1 = base + t; let i2 = base + t + 256u;
    if (i1 < lp.size) { sk[t] = keys[i1]; si[t] = idx[i1]; } else { sk[t] = 3.4e38; si[t] = i1; }
    if (i2 < lp.size) { sk[t + 256u] = keys[i2]; si[t + 256u] = idx[i2]; } else { sk[t + 256u] = 3.4e38; si[t + 256u] = i2; }
    workgroupBarrier();
    let asc = lp.asc == 1u;
    for (var s = 1u; s < 512u; s = s << 1u) {
        for (var st = s; st > 0u; st = st >> 1u) {
            let bs = st << 1u;
            if (t < 256u) {
                let bi = t / st; let ib = t % st;
                let i = bi * bs + ib; let j = i + st;
                if (j < 512u) { cswap(i, j, ((i / (s << 1u)) % 2u == 0u) == asc); }
            }
            workgroupBarrier();
        }
    }
    if (i1 < lp.size) { keys[i1] = sk[t]; idx[i1] = si[t]; }
    if (i2 < lp.size) { keys[i2] = sk[t + 256u]; idx[i2] = si[t + 256u]; }
}

struct MP { size: u32, stage: u32, step: u32, asc: u32 }
@group(0) @binding(0) var<uniform> mp: MP;
@group(0) @binding(1) var<storage, read_write> mkeys: array<f32>;
@group(0) @binding(2) var<storage, read_write> midx: array<u32>;

@compute @workgroup_size(256)
fn merge(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = g.x; let step = mp.step; let stage = mp.stage;
    let bs = 1u << (stage + 1u); let hb = 1u << stage;
    let bi = t / hb; let ih = t % hb;
    let i = bi * bs + ih; let j = i + step;
    if (j >= mp.size) { return; }
    let d = ((i / bs) % 2u == 0u) == (mp.asc == 1u);
    let ki = mkeys[i]; let kj = mkeys[j];
    if (select(ki > kj, ki < kj, d)) {
        mkeys[i] = kj; mkeys[j] = ki;
        let ti = midx[i]; midx[i] = midx[j]; midx[j] = ti;
    }
}

struct IP { size: u32 }
@group(0) @binding(0) var<uniform> ip: IP;
@group(0) @binding(1) var<storage, read_write> iidx: array<u32>;

@compute @workgroup_size(256)
fn init_idx(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x < ip.size) { iidx[g.x] = g.x; }
}`;
    const module2 = this.device.createShaderModule({ code });
    this.pipelines.set("init", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "init_idx" } }));
    this.pipelines.set("local", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "local_sort" } }));
    this.pipelines.set("merge", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "merge" } }));
  }
  isAvailable() {
    return this.available;
  }
  async sort(values, ascending = true) {
    const size = values.length;
    if (!this.available || size < 1e4) return this._cpuSort(values, ascending);
    const padSize = this._nextPow2(size);
    const keys = new Float32Array(padSize);
    keys.set(values instanceof Float32Array ? values : new Float32Array(values));
    for (let i = size; i < padSize; i++) keys[i] = 34e37;
    const keysBuf = this._createBuf(keys, GPUBufferUsage.STORAGE);
    const idxBuf = this.device.createBuffer({ size: padSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const initP = this.pipelines.get("init");
    const ipBuf = this._createUniform(new Uint32Array([padSize]));
    const initBG = this.device.createBindGroup({ layout: initP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ipBuf } }, { binding: 1, resource: { buffer: idxBuf } }] });
    const localP = this.pipelines.get("local");
    const lpBuf = this._createUniform(new Uint32Array([padSize, 0, 0, ascending ? 1 : 0]));
    const localBG = this.device.createBindGroup({ layout: localP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: lpBuf } }, { binding: 1, resource: { buffer: keysBuf } }, { binding: 2, resource: { buffer: idxBuf } }] });
    const enc = this.device.createCommandEncoder();
    const p1 = enc.beginComputePass();
    p1.setPipeline(initP);
    p1.setBindGroup(0, initBG);
    p1.dispatchWorkgroups(Math.ceil(padSize / 256));
    p1.end();
    const p2 = enc.beginComputePass();
    p2.setPipeline(localP);
    p2.setBindGroup(0, localBG);
    p2.dispatchWorkgroups(Math.ceil(padSize / 512));
    p2.end();
    this.device.queue.submit([enc.finish()]);
    if (padSize > 512) {
      const mergeP = this.pipelines.get("merge");
      for (let stageExp = 9; 1 << stageExp < padSize; stageExp++) {
        for (let step = 1 << stageExp; step > 0; step >>= 1) {
          const mEnc = this.device.createCommandEncoder();
          const mpBuf = this._createUniform(new Uint32Array([padSize, stageExp, step, ascending ? 1 : 0]));
          const mergeBG = this.device.createBindGroup({ layout: mergeP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: mpBuf } }, { binding: 1, resource: { buffer: keysBuf } }, { binding: 2, resource: { buffer: idxBuf } }] });
          const mp = mEnc.beginComputePass();
          mp.setPipeline(mergeP);
          mp.setBindGroup(0, mergeBG);
          mp.dispatchWorkgroups(Math.ceil(padSize / 256));
          mp.end();
          this.device.queue.submit([mEnc.finish()]);
          mpBuf.destroy();
        }
      }
    }
    const stageBuf = this.device.createBuffer({ size: size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const cEnc = this.device.createCommandEncoder();
    cEnc.copyBufferToBuffer(idxBuf, 0, stageBuf, 0, size * 4);
    this.device.queue.submit([cEnc.finish()]);
    await stageBuf.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(stageBuf.getMappedRange().slice(0));
    stageBuf.unmap();
    keysBuf.destroy();
    idxBuf.destroy();
    ipBuf.destroy();
    lpBuf.destroy();
    stageBuf.destroy();
    return result;
  }
  _cpuSort(values, ascending) {
    const indexed = Array.from(values).map((v, i) => ({ v, i }));
    indexed.sort((a, b) => {
      const c = a.v < b.v ? -1 : a.v > b.v ? 1 : 0;
      return ascending ? c : -c;
    });
    return new Uint32Array(indexed.map((x) => x.i));
  }
  _createBuf(data, usage) {
    const buf = this.device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }
  _createUniform(data) {
    const buf = this.device.createBuffer({ size: Math.max(data.byteLength, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }
  _nextPow2(n) {
    let p = 1;
    while (p < n) p *= 2;
    return p;
  }
};
var _gpuSorter = null;
function getGPUSorter() {
  if (!_gpuSorter) _gpuSorter = new GPUSorter();
  return _gpuSorter;
}

// src/client/gpu/grouper.js
var GPUGrouper = class {
  constructor() {
    this.device = null;
    this.pipelines = /* @__PURE__ */ new Map();
    this.available = false;
    this._initPromise = null;
  }
  async init() {
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._doInit();
    return this._initPromise;
  }
  async _doInit() {
    if (!navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return false;
      this.device = await adapter.requestDevice({
        requiredLimits: { maxStorageBufferBindingSize: 256 * 1024 * 1024 }
      });
      this._compileShaders();
      this.available = true;
      console.log("[GPUGrouper] Initialized");
      return true;
    } catch (e) {
      console.warn("[GPUGrouper] Init failed:", e);
      return false;
    }
  }
  _compileShaders() {
    const code = `
struct BP { size: u32, cap: u32 }
struct AP { size: u32, cap: u32 }
struct AGP { size: u32, ng: u32, at: u32 }
struct IP { cap: u32 }
struct IAP { ng: u32, iv: u32 }

@group(0) @binding(0) var<uniform> bp: BP;
@group(0) @binding(1) var<storage, read> bk: array<u32>;
@group(0) @binding(2) var<storage, read_write> ht: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> gc: atomic<u32>;

fn fnv(k: u32) -> u32 { var h = 2166136261u; h ^= (k & 0xFFu); h *= 16777619u; h ^= ((k >> 8u) & 0xFFu); h *= 16777619u; h ^= ((k >> 16u) & 0xFFu); h *= 16777619u; h ^= ((k >> 24u) & 0xFFu); h *= 16777619u; return h; }

@compute @workgroup_size(256) fn build(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= bp.size) { return; }
    let k = bk[g.x]; var s = fnv(k) % bp.cap;
    for (var p = 0u; p < bp.cap; p++) { let i = s * 2u; let o = atomicCompareExchangeWeak(&ht[i], 0xFFFFFFFFu, k); if (o.exchanged) { atomicStore(&ht[i + 1u], atomicAdd(&gc, 1u)); return; } if (o.old_value == k) { return; } s = (s + 1u) % bp.cap; }
}

@group(0) @binding(0) var<uniform> ap: AP;
@group(0) @binding(1) var<storage, read> ak: array<u32>;
@group(0) @binding(2) var<storage, read> lt: array<u32>;
@group(0) @binding(3) var<storage, read_write> gids: array<u32>;

@compute @workgroup_size(256) fn assign(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= ap.size) { return; }
    let k = ak[g.x]; var s = fnv(k) % ap.cap;
    for (var p = 0u; p < ap.cap; p++) { let i = s * 2u; if (lt[i] == k) { gids[g.x] = lt[i + 1u]; return; } if (lt[i] == 0xFFFFFFFFu) { gids[g.x] = 0xFFFFFFFFu; return; } s = (s + 1u) % ap.cap; }
    gids[g.x] = 0xFFFFFFFFu;
}

@group(0) @binding(0) var<uniform> agp: AGP;
@group(0) @binding(1) var<storage, read> agi: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> res: array<atomic<u32>>;

fn f2s(f: f32) -> u32 { let b = bitcast<u32>(f); return select(b ^ 0x80000000u, ~b, (b & 0x80000000u) != 0u); }

@compute @workgroup_size(256) fn cnt(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= agp.size) { return; } let gid = agi[g.x]; if (gid < agp.ng) { atomicAdd(&res[gid], 1u); } }
@compute @workgroup_size(256) fn sum(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= agp.size) { return; } let gid = agi[g.x]; let v = vals[g.x]; if (gid < agp.ng && !isNan(v)) { atomicAdd(&res[gid], u32(i32(v * 1000.0))); } }
@compute @workgroup_size(256) fn mn(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= agp.size) { return; } let gid = agi[g.x]; let v = vals[g.x]; if (gid < agp.ng && !isNan(v)) { atomicMin(&res[gid], f2s(v)); } }
@compute @workgroup_size(256) fn mx(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= agp.size) { return; } let gid = agi[g.x]; let v = vals[g.x]; if (gid < agp.ng && !isNan(v)) { atomicMax(&res[gid], f2s(v)); } }

@group(0) @binding(0) var<uniform> ip: IP;
@group(0) @binding(1) var<storage, read_write> it: array<u32>;
@compute @workgroup_size(256) fn iht(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= ip.cap * 2u) { return; } it[g.x] = select(0u, 0xFFFFFFFFu, g.x % 2u == 0u); }

@group(0) @binding(0) var<uniform> iap: IAP;
@group(0) @binding(1) var<storage, read_write> iar: array<u32>;
@compute @workgroup_size(256) fn iag(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= iap.ng) { return; } iar[g.x] = iap.iv; }`;
    const module2 = this.device.createShaderModule({ code });
    this.pipelines.set("iht", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "iht" } }));
    this.pipelines.set("build", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "build" } }));
    this.pipelines.set("assign", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "assign" } }));
    this.pipelines.set("iag", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "iag" } }));
    this.pipelines.set("cnt", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "cnt" } }));
    this.pipelines.set("sum", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "sum" } }));
    this.pipelines.set("mn", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "mn" } }));
    this.pipelines.set("mx", this.device.createComputePipeline({ layout: "auto", compute: { module: module2, entryPoint: "mx" } }));
  }
  isAvailable() {
    return this.available;
  }
  async groupBy(keys) {
    const size = keys.length;
    if (!this.available || size < 1e4) return this._cpuGroupBy(keys);
    const cap = this._nextPow2(Math.min(size, 1e5) * 2);
    const keysBuf = this._createBuf(keys, GPUBufferUsage.STORAGE);
    const htBuf = this.device.createBuffer({ size: cap * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const gcBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const gidsBuf = this.device.createBuffer({ size: size * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const ihtP = this.pipelines.get("iht"), buildP = this.pipelines.get("build"), assignP = this.pipelines.get("assign");
    const ipBuf = this._createUniform(new Uint32Array([cap]));
    const bpBuf = this._createUniform(new Uint32Array([size, cap]));
    const apBuf = this._createUniform(new Uint32Array([size, cap]));
    const ihtBG = this.device.createBindGroup({ layout: ihtP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ipBuf } }, { binding: 1, resource: { buffer: htBuf } }] });
    const buildBG = this.device.createBindGroup({ layout: buildP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bpBuf } }, { binding: 1, resource: { buffer: keysBuf } }, { binding: 2, resource: { buffer: htBuf } }, { binding: 3, resource: { buffer: gcBuf } }] });
    const assignBG = this.device.createBindGroup({ layout: assignP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: apBuf } }, { binding: 1, resource: { buffer: keysBuf } }, { binding: 2, resource: { buffer: htBuf } }, { binding: 3, resource: { buffer: gidsBuf } }] });
    const enc = this.device.createCommandEncoder();
    const p1 = enc.beginComputePass();
    p1.setPipeline(ihtP);
    p1.setBindGroup(0, ihtBG);
    p1.dispatchWorkgroups(Math.ceil(cap * 2 / 256));
    p1.end();
    const p2 = enc.beginComputePass();
    p2.setPipeline(buildP);
    p2.setBindGroup(0, buildBG);
    p2.dispatchWorkgroups(Math.ceil(size / 256));
    p2.end();
    const p3 = enc.beginComputePass();
    p3.setPipeline(assignP);
    p3.setBindGroup(0, assignBG);
    p3.dispatchWorkgroups(Math.ceil(size / 256));
    p3.end();
    const gcStage = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(gcBuf, 0, gcStage, 0, 4);
    this.device.queue.submit([enc.finish()]);
    await gcStage.mapAsync(GPUMapMode.READ);
    const numGroups = new Uint32Array(gcStage.getMappedRange())[0];
    gcStage.unmap();
    const gidsStage = this.device.createBuffer({ size: size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const cEnc = this.device.createCommandEncoder();
    cEnc.copyBufferToBuffer(gidsBuf, 0, gidsStage, 0, size * 4);
    this.device.queue.submit([cEnc.finish()]);
    await gidsStage.mapAsync(GPUMapMode.READ);
    const groupIds = new Uint32Array(gidsStage.getMappedRange().slice(0));
    gidsStage.unmap();
    keysBuf.destroy();
    htBuf.destroy();
    gcBuf.destroy();
    gidsBuf.destroy();
    gcStage.destroy();
    gidsStage.destroy();
    ipBuf.destroy();
    bpBuf.destroy();
    apBuf.destroy();
    return { groupIds, numGroups };
  }
  async groupAggregate(values, groupIds, numGroups, aggType) {
    const size = values.length;
    if (!this.available || size < 1e4) return this._cpuGroupAggregate(values, groupIds, numGroups, aggType);
    let initVal = 0, pName = "cnt";
    if (aggType === "SUM") pName = "sum";
    else if (aggType === "MIN") {
      initVal = 2139095039;
      pName = "mn";
    } else if (aggType === "MAX") pName = "mx";
    const gidsBuf = this._createBuf(groupIds, GPUBufferUsage.STORAGE);
    const valsBuf = this._createBuf(values, GPUBufferUsage.STORAGE);
    const resBuf = this.device.createBuffer({ size: numGroups * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const iagP = this.pipelines.get("iag"), aggP = this.pipelines.get(pName);
    const iapBuf = this._createUniform(new Uint32Array([numGroups, initVal]));
    const agpBuf = this._createUniform(new Uint32Array([size, numGroups, 0]));
    const iagBG = this.device.createBindGroup({ layout: iagP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: iapBuf } }, { binding: 1, resource: { buffer: resBuf } }] });
    const aggBG = this.device.createBindGroup({ layout: aggP.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: agpBuf } }, { binding: 1, resource: { buffer: gidsBuf } }, { binding: 2, resource: { buffer: valsBuf } }, { binding: 3, resource: { buffer: resBuf } }] });
    const enc = this.device.createCommandEncoder();
    const p1 = enc.beginComputePass();
    p1.setPipeline(iagP);
    p1.setBindGroup(0, iagBG);
    p1.dispatchWorkgroups(Math.max(1, Math.ceil(numGroups / 256)));
    p1.end();
    const p2 = enc.beginComputePass();
    p2.setPipeline(aggP);
    p2.setBindGroup(0, aggBG);
    p2.dispatchWorkgroups(Math.ceil(size / 256));
    p2.end();
    const stage = this.device.createBuffer({ size: numGroups * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(resBuf, 0, stage, 0, numGroups * 4);
    this.device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const raw = new Uint32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    const results = new Float32Array(numGroups);
    for (let i = 0; i < numGroups; i++) {
      if (aggType === "COUNT") results[i] = raw[i];
      else if (aggType === "SUM") results[i] = (raw[i] | 0) / 1e3;
      else {
        const u = raw[i], bits = u & 2147483648 ? u ^ 2147483648 : ~u;
        results[i] = new Float32Array(new Uint32Array([bits]).buffer)[0];
      }
    }
    gidsBuf.destroy();
    valsBuf.destroy();
    resBuf.destroy();
    stage.destroy();
    iapBuf.destroy();
    agpBuf.destroy();
    return results;
  }
  _cpuGroupBy(keys) {
    const gMap = /* @__PURE__ */ new Map();
    const gids = new Uint32Array(keys.length);
    let nid = 0;
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      if (!gMap.has(k)) gMap.set(k, nid++);
      gids[i] = gMap.get(k);
    }
    return { groupIds: gids, numGroups: nid };
  }
  _cpuGroupAggregate(values, groupIds, numGroups, aggType) {
    const res = new Float32Array(numGroups);
    if (aggType === "MIN") res.fill(Infinity);
    else if (aggType === "MAX") res.fill(-Infinity);
    for (let i = 0; i < values.length; i++) {
      const gid = groupIds[i], v = values[i];
      if (gid >= numGroups || isNaN(v)) continue;
      if (aggType === "COUNT") res[gid]++;
      else if (aggType === "SUM") res[gid] += v;
      else if (aggType === "MIN") res[gid] = Math.min(res[gid], v);
      else if (aggType === "MAX") res[gid] = Math.max(res[gid], v);
    }
    return res;
  }
  _createBuf(data, usage) {
    const buf = this.device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }
  _createUniform(data) {
    const buf = this.device.createBuffer({ size: Math.max(data.byteLength, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }
  _nextPow2(n) {
    let p = 1;
    while (p < n) p *= 2;
    return p;
  }
};
var _gpuGrouper = null;
function getGPUGrouper() {
  if (!_gpuGrouper) _gpuGrouper = new GPUGrouper();
  return _gpuGrouper;
}

// src/client/gpu/vector-search.js
var GPUVectorSearch = class {
  constructor() {
    this.device = null;
    this.pipelines = /* @__PURE__ */ new Map();
    this.available = false;
    this._initPromise = null;
  }
  async init() {
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._doInit();
    return this._initPromise;
  }
  async _doInit() {
    if (typeof navigator === "undefined" || !navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return false;
      this.device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: 256 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024 } });
      await this._compileShaders();
      this.available = true;
      return true;
    } catch (e) {
      return false;
    }
  }
  async _compileShaders() {
    const distMod = this.device.createShaderModule({ code: VECTOR_DISTANCE_SHADER });
    this.pipelines.set("distance", this.device.createComputePipeline({ layout: "auto", compute: { module: distMod, entryPoint: "compute_distances" } }));
    const topkMod = this.device.createShaderModule({ code: TOPK_SHADER });
    this.pipelines.set("local_topk", this.device.createComputePipeline({ layout: "auto", compute: { module: topkMod, entryPoint: "local_topk" } }));
    this.pipelines.set("merge_topk", this.device.createComputePipeline({ layout: "auto", compute: { module: topkMod, entryPoint: "merge_topk" } }));
  }
  isAvailable() {
    return this.available;
  }
  async computeDistances(queryVec, vectors, numQueries = 1, metric = 0) {
    const numVectors = vectors.length;
    const dim = queryVec.length / numQueries;
    if (!this.available || numVectors < GPU_DISTANCE_THRESHOLD) return this._cpuDistances(queryVec, vectors, numQueries, metric);
    const flatVectors = new Float32Array(numVectors * dim);
    for (let i = 0; i < numVectors; i++) flatVectors.set(vectors[i], i * dim);
    const paramsBuffer = this._createUniform(new Uint32Array([dim, numVectors, numQueries, metric]));
    const queryBuffer = this._createStorage(queryVec);
    const vectorsBuffer = this._createStorage(flatVectors);
    const distanceBuffer = this.device.createBuffer({ size: numQueries * numVectors * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuffer = this.device.createBuffer({ size: numQueries * numVectors * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bindGroup = this.device.createBindGroup({ layout: this.pipelines.get("distance").getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: queryBuffer } },
      { binding: 2, resource: { buffer: vectorsBuffer } },
      { binding: 3, resource: { buffer: distanceBuffer } }
    ] });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.get("distance"));
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(numVectors / 256), numQueries, 1);
    pass.end();
    encoder.copyBufferToBuffer(distanceBuffer, 0, readBuffer, 0, numQueries * numVectors * 4);
    this.device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const distances = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    [paramsBuffer, queryBuffer, vectorsBuffer, distanceBuffer, readBuffer].forEach((b) => b.destroy());
    return distances;
  }
  async topK(scores, indices = null, k = 10, descending = true) {
    const n = scores.length;
    if (!this.available || n < GPU_TOPK_THRESHOLD) return this._cpuTopK(scores, indices, k, descending);
    if (!indices) {
      indices = new Uint32Array(n);
      for (let i = 0; i < n; i++) indices[i] = i;
    }
    const numWorkgroups = Math.ceil(n / 512);
    const kPerWg = Math.min(k, 512);
    const numCandidates = numWorkgroups * kPerWg;
    const paramsBuffer = this._createUniform(new Uint32Array([n, k, descending ? 1 : 0, numWorkgroups]));
    const inputScoresBuffer = this._createStorage(scores);
    const inputIndicesBuffer = this._createStorage(indices);
    const intermediateScoresBuffer = this.device.createBuffer({ size: numCandidates * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const intermediateIndicesBuffer = this.device.createBuffer({ size: numCandidates * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const localBG = this.device.createBindGroup({ layout: this.pipelines.get("local_topk").getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: inputScoresBuffer } },
      { binding: 2, resource: { buffer: inputIndicesBuffer } },
      { binding: 3, resource: { buffer: intermediateScoresBuffer } },
      { binding: 4, resource: { buffer: intermediateIndicesBuffer } }
    ] });
    let encoder = this.device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.get("local_topk"));
    pass.setBindGroup(0, localBG);
    pass.dispatchWorkgroups(numWorkgroups, 1, 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    const mergeParamsBuffer = this._createUniform(new Uint32Array([numCandidates, k, descending ? 1 : 0, 0]));
    const finalScoresBuffer = this.device.createBuffer({ size: k * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const finalIndicesBuffer = this.device.createBuffer({ size: k * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readScoresBuffer = this.device.createBuffer({ size: k * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const readIndicesBuffer = this.device.createBuffer({ size: k * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const mergeBG = this.device.createBindGroup({ layout: this.pipelines.get("merge_topk").getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: mergeParamsBuffer } },
      { binding: 1, resource: { buffer: intermediateScoresBuffer } },
      { binding: 2, resource: { buffer: intermediateIndicesBuffer } },
      { binding: 3, resource: { buffer: finalScoresBuffer } },
      { binding: 4, resource: { buffer: finalIndicesBuffer } }
    ] });
    encoder = this.device.createCommandEncoder();
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.get("merge_topk"));
    pass.setBindGroup(0, mergeBG);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();
    encoder.copyBufferToBuffer(finalScoresBuffer, 0, readScoresBuffer, 0, k * 4);
    encoder.copyBufferToBuffer(finalIndicesBuffer, 0, readIndicesBuffer, 0, k * 4);
    this.device.queue.submit([encoder.finish()]);
    await Promise.all([readScoresBuffer.mapAsync(GPUMapMode.READ), readIndicesBuffer.mapAsync(GPUMapMode.READ)]);
    const resultScores = new Float32Array(readScoresBuffer.getMappedRange().slice(0));
    const resultIndices = new Uint32Array(readIndicesBuffer.getMappedRange().slice(0));
    readScoresBuffer.unmap();
    readIndicesBuffer.unmap();
    [
      paramsBuffer,
      inputScoresBuffer,
      inputIndicesBuffer,
      intermediateScoresBuffer,
      intermediateIndicesBuffer,
      mergeParamsBuffer,
      finalScoresBuffer,
      finalIndicesBuffer,
      readScoresBuffer,
      readIndicesBuffer
    ].forEach((b) => b.destroy());
    return { indices: resultIndices, scores: resultScores };
  }
  async search(queryVec, vectors, k = 10, options = {}) {
    const { metric = 0 } = options;
    const scores = await this.computeDistances(queryVec, vectors, 1, metric);
    const descending = metric === 0 || metric === 2;
    return await this.topK(scores, null, k, descending);
  }
  _cpuDistances(queryVec, vectors, numQueries, metric) {
    const dim = queryVec.length / numQueries;
    const numVectors = vectors.length;
    const distances = new Float32Array(numQueries * numVectors);
    for (let q = 0; q < numQueries; q++) {
      const qOff = q * dim;
      for (let v = 0; v < numVectors; v++) {
        const vec = vectors[v];
        let result = 0;
        if (metric === 1) {
          for (let i = 0; i < dim; i++) {
            const d = queryVec[qOff + i] - vec[i];
            result += d * d;
          }
          result = Math.sqrt(result);
        } else {
          for (let i = 0; i < dim; i++) result += queryVec[qOff + i] * vec[i];
        }
        distances[q * numVectors + v] = result;
      }
    }
    return distances;
  }
  _cpuTopK(scores, indices, k, descending) {
    const n = scores.length;
    if (!indices) {
      indices = new Uint32Array(n);
      for (let i = 0; i < n; i++) indices[i] = i;
    }
    const indexed = Array.from(scores).map((score, i) => ({ score, idx: indices[i] }));
    if (descending) indexed.sort((a, b) => b.score - a.score);
    else indexed.sort((a, b) => a.score - b.score);
    const topK = indexed.slice(0, k);
    return { indices: new Uint32Array(topK.map((x) => x.idx)), scores: new Float32Array(topK.map((x) => x.score)) };
  }
  _createStorage(data) {
    const buf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }
  _createUniform(data) {
    const buf = this.device.createBuffer({ size: Math.max(data.byteLength, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }
};
var _gpuVectorSearch = null;
function getGPUVectorSearch() {
  if (!_gpuVectorSearch) _gpuVectorSearch = new GPUVectorSearch();
  return _gpuVectorSearch;
}

// src/client/index.js
init_opfs();
init_lance_reader();

// src/client/storage/lance-writer.js
var ProtobufEncoder = class _ProtobufEncoder {
  constructor() {
    this.chunks = [];
  }
  /**
   * Encode a varint (variable-length integer)
   * @param {number|bigint} value
   * @returns {Uint8Array}
   */
  static encodeVarint(value) {
    const bytes = [];
    let v = typeof value === "bigint" ? value : BigInt(value);
    while (v > 0x7fn) {
      bytes.push(Number(v & 0x7fn) | 128);
      v >>= 7n;
    }
    bytes.push(Number(v));
    return new Uint8Array(bytes);
  }
  /**
   * Encode a field header (tag)
   * @param {number} fieldNum - Field number
   * @param {number} wireType - Wire type (0=varint, 2=length-delimited)
   * @returns {Uint8Array}
   */
  static encodeFieldHeader(fieldNum, wireType) {
    const tag = fieldNum << 3 | wireType;
    return _ProtobufEncoder.encodeVarint(tag);
  }
  /**
   * Encode a varint field
   * @param {number} fieldNum
   * @param {number|bigint} value
   */
  writeVarint(fieldNum, value) {
    this.chunks.push(_ProtobufEncoder.encodeFieldHeader(fieldNum, 0));
    this.chunks.push(_ProtobufEncoder.encodeVarint(value));
  }
  /**
   * Encode a length-delimited field (bytes or nested message)
   * @param {number} fieldNum
   * @param {Uint8Array} data
   */
  writeBytes(fieldNum, data) {
    this.chunks.push(_ProtobufEncoder.encodeFieldHeader(fieldNum, 2));
    this.chunks.push(_ProtobufEncoder.encodeVarint(data.length));
    this.chunks.push(data);
  }
  /**
   * Encode packed repeated uint64 as varints
   * @param {number} fieldNum
   * @param {BigUint64Array|number[]} values
   */
  writePackedUint64(fieldNum, values) {
    const varintChunks = [];
    for (const v of values) {
      varintChunks.push(_ProtobufEncoder.encodeVarint(v));
    }
    const totalLen = varintChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    this.chunks.push(_ProtobufEncoder.encodeFieldHeader(fieldNum, 2));
    this.chunks.push(_ProtobufEncoder.encodeVarint(totalLen));
    for (const chunk of varintChunks) {
      this.chunks.push(chunk);
    }
  }
  /**
   * Get the encoded bytes
   * @returns {Uint8Array}
   */
  toBytes() {
    const totalLen = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const result = new Uint8Array(totalLen);
    let offset = 0;
    for (const chunk of this.chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }
    return result;
  }
  /**
   * Clear the encoder for reuse
   */
  clear() {
    this.chunks = [];
  }
};
var LanceColumnType = {
  INT64: "int64",
  FLOAT64: "float64",
  STRING: "string",
  BOOL: "bool",
  INT32: "int32",
  FLOAT32: "float32"
};
var PureLanceWriter2 = class {
  /**
   * @param {Object} options
   * @param {number} [options.majorVersion=0] - Lance format major version
   * @param {number} [options.minorVersion=3] - Lance format minor version (3 = v2.0)
   */
  constructor(options = {}) {
    this.majorVersion = options.majorVersion ?? 0;
    this.minorVersion = options.minorVersion ?? 3;
    this.columns = [];
    this.rowCount = null;
  }
  /**
   * Validate row count consistency
   * @param {number} count
   */
  _validateRowCount(count) {
    if (this.rowCount === null) {
      this.rowCount = count;
    } else if (this.rowCount !== count) {
      throw new Error(`Row count mismatch: expected ${this.rowCount}, got ${count}`);
    }
  }
  /**
   * Add an int64 column
   * @param {string} name - Column name
   * @param {BigInt64Array} values - Column values
   */
  addInt64Column(name, values) {
    this._validateRowCount(values.length);
    this.columns.push({
      name,
      type: LanceColumnType.INT64,
      data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
      length: values.length
    });
  }
  /**
   * Add an int32 column
   * @param {string} name - Column name
   * @param {Int32Array} values - Column values
   */
  addInt32Column(name, values) {
    this._validateRowCount(values.length);
    this.columns.push({
      name,
      type: LanceColumnType.INT32,
      data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
      length: values.length
    });
  }
  /**
   * Add a float64 column
   * @param {string} name - Column name
   * @param {Float64Array} values - Column values
   */
  addFloat64Column(name, values) {
    this._validateRowCount(values.length);
    this.columns.push({
      name,
      type: LanceColumnType.FLOAT64,
      data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
      length: values.length
    });
  }
  /**
   * Add a float32 column
   * @param {string} name - Column name
   * @param {Float32Array} values - Column values
   */
  addFloat32Column(name, values) {
    this._validateRowCount(values.length);
    this.columns.push({
      name,
      type: LanceColumnType.FLOAT32,
      data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
      length: values.length
    });
  }
  /**
   * Add a boolean column
   * @param {string} name - Column name
   * @param {boolean[]} values - Column values
   */
  addBoolColumn(name, values) {
    this._validateRowCount(values.length);
    const data = new Uint8Array(values.length);
    for (let i = 0; i < values.length; i++) {
      data[i] = values[i] ? 1 : 0;
    }
    this.columns.push({
      name,
      type: LanceColumnType.BOOL,
      data,
      length: values.length
    });
  }
  /**
   * Add a string column
   * @param {string} name - Column name
   * @param {string[]} values - Column values
   */
  addStringColumn(name, values) {
    this._validateRowCount(values.length);
    const encoder = new TextEncoder();
    const offsets = new Int32Array(values.length + 1);
    const dataChunks = [];
    let currentOffset = 0;
    for (let i = 0; i < values.length; i++) {
      offsets[i] = currentOffset;
      const encoded = encoder.encode(values[i]);
      dataChunks.push(encoded);
      currentOffset += encoded.length;
    }
    offsets[values.length] = currentOffset;
    const offsetsBytes = new Uint8Array(offsets.buffer);
    const totalDataLen = dataChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const stringData = new Uint8Array(totalDataLen);
    let writePos = 0;
    for (const chunk of dataChunks) {
      stringData.set(chunk, writePos);
      writePos += chunk.length;
    }
    this.columns.push({
      name,
      type: LanceColumnType.STRING,
      offsetsData: offsetsBytes,
      stringData,
      data: null,
      // Will be combined in finalize
      length: values.length
    });
  }
  /**
   * Build column metadata protobuf for a column
   * @param {number} bufferOffset - Offset to column data
   * @param {number} bufferSize - Size of column data
   * @param {number} length - Number of rows
   * @param {string} type - Column type
   * @returns {Uint8Array}
   */
  _buildColumnMeta(bufferOffset, bufferSize, length, type) {
    const pageEncoder = new ProtobufEncoder();
    pageEncoder.writePackedUint64(1, [BigInt(bufferOffset)]);
    pageEncoder.writePackedUint64(2, [BigInt(bufferSize)]);
    pageEncoder.writeVarint(3, length);
    pageEncoder.writeVarint(5, 0);
    const pageBytes = pageEncoder.toBytes();
    const metaEncoder = new ProtobufEncoder();
    metaEncoder.writeBytes(2, pageBytes);
    return metaEncoder.toBytes();
  }
  /**
   * Build column metadata for string column (2 buffers: offsets + data)
   * @param {number} offsetsBufOffset - Offset to offsets buffer
   * @param {number} offsetsBufSize - Size of offsets buffer
   * @param {number} dataBufOffset - Offset to string data buffer
   * @param {number} dataBufSize - Size of string data buffer
   * @param {number} length - Number of rows
   * @returns {Uint8Array}
   */
  _buildStringColumnMeta(offsetsBufOffset, offsetsBufSize, dataBufOffset, dataBufSize, length) {
    const pageEncoder = new ProtobufEncoder();
    pageEncoder.writePackedUint64(1, [BigInt(offsetsBufOffset), BigInt(dataBufOffset)]);
    pageEncoder.writePackedUint64(2, [BigInt(offsetsBufSize), BigInt(dataBufSize)]);
    pageEncoder.writeVarint(3, length);
    pageEncoder.writeVarint(5, 0);
    const pageBytes = pageEncoder.toBytes();
    const metaEncoder = new ProtobufEncoder();
    metaEncoder.writeBytes(2, pageBytes);
    return metaEncoder.toBytes();
  }
  /**
   * Finalize and create the Lance file
   * @returns {Uint8Array} Complete Lance file data
   */
  finalize() {
    if (this.columns.length === 0) {
      throw new Error("No columns added");
    }
    const chunks = [];
    let currentOffset = 0;
    const columnBufferInfos = [];
    for (const col of this.columns) {
      if (col.type === LanceColumnType.STRING) {
        const offsetsOffset = currentOffset;
        chunks.push(col.offsetsData);
        currentOffset += col.offsetsData.length;
        const dataOffset = currentOffset;
        chunks.push(col.stringData);
        currentOffset += col.stringData.length;
        columnBufferInfos.push({
          type: "string",
          offsetsOffset,
          offsetsSize: col.offsetsData.length,
          dataOffset,
          dataSize: col.stringData.length,
          length: col.length
        });
      } else {
        const bufferOffset = currentOffset;
        chunks.push(col.data);
        currentOffset += col.data.length;
        columnBufferInfos.push({
          type: col.type,
          offset: bufferOffset,
          size: col.data.length,
          length: col.length
        });
      }
    }
    const columnMetadatas = [];
    for (let i = 0; i < this.columns.length; i++) {
      const info = columnBufferInfos[i];
      let meta;
      if (info.type === "string") {
        meta = this._buildStringColumnMeta(
          info.offsetsOffset,
          info.offsetsSize,
          info.dataOffset,
          info.dataSize,
          info.length
        );
      } else {
        meta = this._buildColumnMeta(info.offset, info.size, info.length, info.type);
      }
      columnMetadatas.push(meta);
    }
    const columnMetaStart = currentOffset;
    const columnMetaOffsets = [];
    let metaOffset = 0;
    for (const meta of columnMetadatas) {
      columnMetaOffsets.push(metaOffset);
      chunks.push(meta);
      currentOffset += meta.length;
      metaOffset += meta.length;
    }
    const columnMetaOffsetsStart = currentOffset;
    const offsetTable = new BigUint64Array(columnMetaOffsets.length);
    for (let i = 0; i < columnMetaOffsets.length; i++) {
      offsetTable[i] = BigInt(columnMetaOffsets[i]);
    }
    const offsetTableBytes = new Uint8Array(offsetTable.buffer);
    chunks.push(offsetTableBytes);
    currentOffset += offsetTableBytes.length;
    const globalBuffOffsetsStart = currentOffset;
    const numGlobalBuffers = 0;
    const footer = new ArrayBuffer(LANCE_FOOTER_SIZE);
    const footerView = new DataView(footer);
    footerView.setBigUint64(0, BigInt(columnMetaStart), true);
    footerView.setBigUint64(8, BigInt(columnMetaOffsetsStart), true);
    footerView.setBigUint64(16, BigInt(globalBuffOffsetsStart), true);
    footerView.setUint32(24, numGlobalBuffers, true);
    footerView.setUint32(28, this.columns.length, true);
    footerView.setUint16(32, this.majorVersion, true);
    footerView.setUint16(34, this.minorVersion, true);
    new Uint8Array(footer, 36, 4).set(LANCE_MAGIC);
    chunks.push(new Uint8Array(footer));
    const totalSize = currentOffset + LANCE_FOOTER_SIZE;
    const result = new Uint8Array(totalSize);
    let writeOffset = 0;
    for (const chunk of chunks) {
      result.set(chunk, writeOffset);
      writeOffset += chunk.length;
    }
    return result;
  }
  /**
   * Get the number of columns
   * @returns {number}
   */
  getNumColumns() {
    return this.columns.length;
  }
  /**
   * Get the row count
   * @returns {number|null}
   */
  getRowCount() {
    return this.rowCount;
  }
  /**
   * Get column names
   * @returns {string[]}
   */
  getColumnNames() {
    return this.columns.map((c) => c.name);
  }
};

// src/client/storage/dataset-storage.js
init_opfs();
var DatasetStorage = class {
  constructor(dbName = "lanceql-files", version = 1) {
    this.dbName = dbName;
    this.version = version;
    this.db = null;
    this.SIZE_THRESHOLD = 50 * 1024 * 1024;
  }
  async open() {
    if (this.db) return this.db;
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains("files")) {
          db.createObjectStore("files", { keyPath: "name" });
        }
        if (!db.objectStoreNames.contains("index")) {
          const store = db.createObjectStore("index", { keyPath: "name" });
          store.createIndex("timestamp", "timestamp");
          store.createIndex("size", "size");
        }
      };
    });
  }
  async hasOPFS() {
    try {
      return "storage" in navigator && "getDirectory" in navigator.storage;
    } catch {
      return false;
    }
  }
  async save(name, data, metadata = {}) {
    const db = await this.open();
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
    const size = bytes.byteLength;
    const useOPFS = size >= this.SIZE_THRESHOLD && await this.hasOPFS();
    if (useOPFS) {
      try {
        const root = await navigator.storage.getDirectory();
        const fileHandle = await root.getFileHandle(name, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(bytes);
        await writable.close();
      } catch (e) {
        console.warn("[DatasetStorage] OPFS save failed, falling back to IndexedDB:", e);
      }
    }
    if (!useOPFS) {
      await new Promise((resolve, reject) => {
        const tx = db.transaction("files", "readwrite");
        const store = tx.objectStore("files");
        store.put({ name, data: bytes });
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    }
    await new Promise((resolve, reject) => {
      const tx = db.transaction("index", "readwrite");
      const store = tx.objectStore("index");
      store.put({
        name,
        size,
        timestamp: Date.now(),
        storage: useOPFS ? "opfs" : "indexeddb",
        ...metadata
      });
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    return { name, size, storage: useOPFS ? "opfs" : "indexeddb" };
  }
  async load(name) {
    const db = await this.open();
    const entry = await new Promise((resolve) => {
      const tx = db.transaction("index", "readonly");
      const store = tx.objectStore("index");
      const request = store.get(name);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
    });
    if (!entry) return null;
    if (entry.storage === "opfs") {
      try {
        const root = await navigator.storage.getDirectory();
        const fileHandle = await root.getFileHandle(name);
        const file = await fileHandle.getFile();
        const buffer = await file.arrayBuffer();
        return new Uint8Array(buffer);
      } catch (e) {
        console.warn("[DatasetStorage] OPFS load failed:", e);
        return null;
      }
    }
    return new Promise((resolve) => {
      const tx = db.transaction("files", "readonly");
      const store = tx.objectStore("files");
      const request = store.get(name);
      request.onsuccess = () => {
        const result = request.result;
        resolve(result ? result.data : null);
      };
      request.onerror = () => resolve(null);
    });
  }
  async list() {
    const db = await this.open();
    return new Promise((resolve) => {
      const tx = db.transaction("index", "readonly");
      const store = tx.objectStore("index");
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => resolve([]);
    });
  }
  async delete(name) {
    const db = await this.open();
    const entry = await new Promise((resolve) => {
      const tx = db.transaction("index", "readonly");
      const store = tx.objectStore("index");
      const request = store.get(name);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
    });
    if (entry?.storage === "opfs") {
      try {
        const root = await navigator.storage.getDirectory();
        await root.removeEntry(name);
      } catch (e) {
        console.warn("[DatasetStorage] OPFS delete failed:", e);
      }
    }
    await new Promise((resolve) => {
      const tx = db.transaction("files", "readwrite");
      const store = tx.objectStore("files");
      store.delete(name);
      tx.oncomplete = () => resolve();
    });
    await new Promise((resolve) => {
      const tx = db.transaction("index", "readwrite");
      const store = tx.objectStore("index");
      store.delete(name);
      tx.oncomplete = () => resolve();
    });
  }
  /**
   * Check if a dataset exists.
   * @param {string} name - Dataset name
   * @returns {Promise<boolean>}
   */
  async exists(name) {
    const db = await this.open();
    return new Promise((resolve) => {
      const tx = db.transaction("index", "readonly");
      const store = tx.objectStore("index");
      const request = store.get(name);
      request.onsuccess = () => resolve(!!request.result);
      request.onerror = () => resolve(false);
    });
  }
  /**
   * Get storage usage info.
   * @returns {Promise<Object>} Usage stats
   */
  async getUsage() {
    const datasets = await this.list();
    const totalSize = datasets.reduce((sum, d) => sum + (d.size || 0), 0);
    const indexedDBCount = datasets.filter((d) => d.storage === "indexeddb").length;
    const opfsCount = datasets.filter((d) => d.storage === "opfs").length;
    let quota = null;
    if (navigator.storage?.estimate) {
      quota = await navigator.storage.estimate();
    }
    return {
      datasets: datasets.length,
      totalSize,
      indexedDBCount,
      opfsCount,
      quota
    };
  }
};
var opfsStorage3 = new OPFSStorage2();
var datasetStorage = new DatasetStorage();

// src/client/sql/lexer.js
var TokenType2 = {
  // Keywords
  SELECT: "SELECT",
  DISTINCT: "DISTINCT",
  FROM: "FROM",
  WHERE: "WHERE",
  AND: "AND",
  OR: "OR",
  NOT: "NOT",
  ORDER: "ORDER",
  BY: "BY",
  ASC: "ASC",
  DESC: "DESC",
  LIMIT: "LIMIT",
  OFFSET: "OFFSET",
  AS: "AS",
  NULL: "NULL",
  IS: "IS",
  IN: "IN",
  BETWEEN: "BETWEEN",
  LIKE: "LIKE",
  TRUE: "TRUE",
  FALSE: "FALSE",
  GROUP: "GROUP",
  HAVING: "HAVING",
  QUALIFY: "QUALIFY",
  ROLLUP: "ROLLUP",
  CUBE: "CUBE",
  GROUPING: "GROUPING",
  SETS: "SETS",
  COUNT: "COUNT",
  SUM: "SUM",
  AVG: "AVG",
  MIN: "MIN",
  MAX: "MAX",
  NEAR: "NEAR",
  TOPK: "TOPK",
  FILE: "FILE",
  JOIN: "JOIN",
  INNER: "INNER",
  LEFT: "LEFT",
  RIGHT: "RIGHT",
  FULL: "FULL",
  OUTER: "OUTER",
  CROSS: "CROSS",
  ON: "ON",
  CREATE: "CREATE",
  TABLE: "TABLE",
  INSERT: "INSERT",
  INTO: "INTO",
  VALUES: "VALUES",
  UPDATE: "UPDATE",
  SET: "SET",
  DELETE: "DELETE",
  DROP: "DROP",
  IF: "IF",
  EXISTS: "EXISTS",
  INT: "INT",
  INTEGER: "INTEGER",
  BIGINT: "BIGINT",
  FLOAT: "FLOAT",
  REAL: "REAL",
  DOUBLE: "DOUBLE",
  TEXT: "TEXT",
  VARCHAR: "VARCHAR",
  BOOLEAN: "BOOLEAN",
  BOOL: "BOOL",
  VECTOR: "VECTOR",
  PRIMARY: "PRIMARY",
  KEY: "KEY",
  WITH: "WITH",
  RECURSIVE: "RECURSIVE",
  UNION: "UNION",
  ALL: "ALL",
  PIVOT: "PIVOT",
  UNPIVOT: "UNPIVOT",
  FOR: "FOR",
  INTERSECT: "INTERSECT",
  EXCEPT: "EXCEPT",
  OVER: "OVER",
  PARTITION: "PARTITION",
  ROW_NUMBER: "ROW_NUMBER",
  RANK: "RANK",
  DENSE_RANK: "DENSE_RANK",
  NTILE: "NTILE",
  LAG: "LAG",
  LEAD: "LEAD",
  FIRST_VALUE: "FIRST_VALUE",
  LAST_VALUE: "LAST_VALUE",
  NTH_VALUE: "NTH_VALUE",
  PERCENT_RANK: "PERCENT_RANK",
  CUME_DIST: "CUME_DIST",
  ROWS: "ROWS",
  RANGE: "RANGE",
  UNBOUNDED: "UNBOUNDED",
  PRECEDING: "PRECEDING",
  FOLLOWING: "FOLLOWING",
  CURRENT: "CURRENT",
  ROW: "ROW",
  EXPLAIN: "EXPLAIN",
  ARRAY: "ARRAY",
  CASE: "CASE",
  WHEN: "WHEN",
  THEN: "THEN",
  ELSE: "ELSE",
  END: "END",
  CAST: "CAST",
  COALESCE: "COALESCE",
  NULLIF: "NULLIF",
  // Literals & Operators
  IDENTIFIER: "IDENTIFIER",
  NUMBER: "NUMBER",
  STRING: "STRING",
  STAR: "STAR",
  COMMA: "COMMA",
  DOT: "DOT",
  LPAREN: "LPAREN",
  RPAREN: "RPAREN",
  EQ: "EQ",
  NE: "NE",
  LT: "LT",
  LE: "LE",
  GT: "GT",
  GE: "GE",
  PLUS: "PLUS",
  MINUS: "MINUS",
  SLASH: "SLASH",
  LBRACKET: "LBRACKET",
  RBRACKET: "RBRACKET",
  EOF: "EOF"
};
var KEYWORDS = {
  "SELECT": TokenType2.SELECT,
  "DISTINCT": TokenType2.DISTINCT,
  "FROM": TokenType2.FROM,
  "WHERE": TokenType2.WHERE,
  "AND": TokenType2.AND,
  "OR": TokenType2.OR,
  "NOT": TokenType2.NOT,
  "ORDER": TokenType2.ORDER,
  "BY": TokenType2.BY,
  "ASC": TokenType2.ASC,
  "DESC": TokenType2.DESC,
  "LIMIT": TokenType2.LIMIT,
  "OFFSET": TokenType2.OFFSET,
  "AS": TokenType2.AS,
  "NULL": TokenType2.NULL,
  "IS": TokenType2.IS,
  "IN": TokenType2.IN,
  "BETWEEN": TokenType2.BETWEEN,
  "LIKE": TokenType2.LIKE,
  "TRUE": TokenType2.TRUE,
  "FALSE": TokenType2.FALSE,
  "GROUP": TokenType2.GROUP,
  "HAVING": TokenType2.HAVING,
  "QUALIFY": TokenType2.QUALIFY,
  "ROLLUP": TokenType2.ROLLUP,
  "CUBE": TokenType2.CUBE,
  "GROUPING": TokenType2.GROUPING,
  "SETS": TokenType2.SETS,
  "COUNT": TokenType2.COUNT,
  "SUM": TokenType2.SUM,
  "AVG": TokenType2.AVG,
  "MIN": TokenType2.MIN,
  "MAX": TokenType2.MAX,
  "NEAR": TokenType2.NEAR,
  "TOPK": TokenType2.TOPK,
  "FILE": TokenType2.FILE,
  "JOIN": TokenType2.JOIN,
  "INNER": TokenType2.INNER,
  "LEFT": TokenType2.LEFT,
  "RIGHT": TokenType2.RIGHT,
  "FULL": TokenType2.FULL,
  "OUTER": TokenType2.OUTER,
  "CROSS": TokenType2.CROSS,
  "ON": TokenType2.ON,
  "CREATE": TokenType2.CREATE,
  "TABLE": TokenType2.TABLE,
  "INSERT": TokenType2.INSERT,
  "INTO": TokenType2.INTO,
  "VALUES": TokenType2.VALUES,
  "UPDATE": TokenType2.UPDATE,
  "SET": TokenType2.SET,
  "DELETE": TokenType2.DELETE,
  "DROP": TokenType2.DROP,
  "IF": TokenType2.IF,
  "EXISTS": TokenType2.EXISTS,
  "INT": TokenType2.INT,
  "INTEGER": TokenType2.INTEGER,
  "BIGINT": TokenType2.BIGINT,
  "FLOAT": TokenType2.FLOAT,
  "REAL": TokenType2.REAL,
  "DOUBLE": TokenType2.DOUBLE,
  "TEXT": TokenType2.TEXT,
  "VARCHAR": TokenType2.VARCHAR,
  "BOOLEAN": TokenType2.BOOLEAN,
  "BOOL": TokenType2.BOOL,
  "VECTOR": TokenType2.VECTOR,
  "PRIMARY": TokenType2.PRIMARY,
  "KEY": TokenType2.KEY,
  "WITH": TokenType2.WITH,
  "RECURSIVE": TokenType2.RECURSIVE,
  "UNION": TokenType2.UNION,
  "ALL": TokenType2.ALL,
  "PIVOT": TokenType2.PIVOT,
  "UNPIVOT": TokenType2.UNPIVOT,
  "FOR": TokenType2.FOR,
  "INTERSECT": TokenType2.INTERSECT,
  "EXCEPT": TokenType2.EXCEPT,
  "OVER": TokenType2.OVER,
  "PARTITION": TokenType2.PARTITION,
  "ROW_NUMBER": TokenType2.ROW_NUMBER,
  "RANK": TokenType2.RANK,
  "DENSE_RANK": TokenType2.DENSE_RANK,
  "NTILE": TokenType2.NTILE,
  "LAG": TokenType2.LAG,
  "LEAD": TokenType2.LEAD,
  "FIRST_VALUE": TokenType2.FIRST_VALUE,
  "LAST_VALUE": TokenType2.LAST_VALUE,
  "NTH_VALUE": TokenType2.NTH_VALUE,
  "PERCENT_RANK": TokenType2.PERCENT_RANK,
  "CUME_DIST": TokenType2.CUME_DIST,
  "ROWS": TokenType2.ROWS,
  "RANGE": TokenType2.RANGE,
  "UNBOUNDED": TokenType2.UNBOUNDED,
  "PRECEDING": TokenType2.PRECEDING,
  "FOLLOWING": TokenType2.FOLLOWING,
  "CURRENT": TokenType2.CURRENT,
  "ROW": TokenType2.ROW,
  "EXPLAIN": TokenType2.EXPLAIN,
  "ARRAY": TokenType2.ARRAY,
  "CASE": TokenType2.CASE,
  "WHEN": TokenType2.WHEN,
  "THEN": TokenType2.THEN,
  "ELSE": TokenType2.ELSE,
  "END": TokenType2.END,
  "CAST": TokenType2.CAST,
  "COALESCE": TokenType2.COALESCE,
  "NULLIF": TokenType2.NULLIF
};
var SQLLexer = class {
  constructor(sql) {
    this.sql = sql;
    this.pos = 0;
    this.length = sql.length;
  }
  peek() {
    if (this.pos >= this.length) return "\0";
    return this.sql[this.pos];
  }
  advance() {
    if (this.pos < this.length) {
      return this.sql[this.pos++];
    }
    return "\0";
  }
  skipWhitespace() {
    while (this.pos < this.length && /\s/.test(this.sql[this.pos])) {
      this.pos++;
    }
  }
  readIdentifier() {
    const start = this.pos;
    while (this.pos < this.length && /[a-zA-Z0-9_]/.test(this.sql[this.pos])) {
      this.pos++;
    }
    return this.sql.slice(start, this.pos);
  }
  readNumber() {
    const start = this.pos;
    let hasDecimal = false;
    while (this.pos < this.length) {
      const ch = this.sql[this.pos];
      if (ch === "." && !hasDecimal) {
        hasDecimal = true;
        this.pos++;
      } else if (/\d/.test(ch)) {
        this.pos++;
      } else {
        break;
      }
    }
    return this.sql.slice(start, this.pos);
  }
  readString(quote) {
    const start = this.pos;
    this.advance();
    while (this.pos < this.length) {
      const ch = this.sql[this.pos];
      if (ch === quote) {
        if (this.pos + 1 < this.length && this.sql[this.pos + 1] === quote) {
          this.pos += 2;
          continue;
        }
        this.pos++;
        break;
      }
      this.pos++;
    }
    const inner = this.sql.slice(start + 1, this.pos - 1);
    return inner.replace(new RegExp(quote + quote, "g"), quote);
  }
  nextToken() {
    this.skipWhitespace();
    if (this.pos >= this.length) {
      return { type: TokenType2.EOF, value: null };
    }
    const ch = this.peek();
    if (/[a-zA-Z_]/.test(ch)) {
      const value = this.readIdentifier();
      const upper = value.toUpperCase();
      const type = KEYWORDS[upper] || TokenType2.IDENTIFIER;
      return { type, value: type === TokenType2.IDENTIFIER ? value : upper };
    }
    if (/\d/.test(ch)) {
      const value = this.readNumber();
      return { type: TokenType2.NUMBER, value };
    }
    if (ch === "'" || ch === '"') {
      const value = this.readString(ch);
      return { type: TokenType2.STRING, value };
    }
    this.advance();
    switch (ch) {
      case "*":
        return { type: TokenType2.STAR, value: "*" };
      case ",":
        return { type: TokenType2.COMMA, value: "," };
      case ".":
        return { type: TokenType2.DOT, value: "." };
      case "(":
        return { type: TokenType2.LPAREN, value: "(" };
      case ")":
        return { type: TokenType2.RPAREN, value: ")" };
      case "+":
        return { type: TokenType2.PLUS, value: "+" };
      case "-":
        return { type: TokenType2.MINUS, value: "-" };
      case "/":
        return { type: TokenType2.SLASH, value: "/" };
      case "[":
        return { type: TokenType2.LBRACKET, value: "[" };
      case "]":
        return { type: TokenType2.RBRACKET, value: "]" };
      case "=":
        return { type: TokenType2.EQ, value: "=" };
      case "<":
        if (this.peek() === "=") {
          this.advance();
          return { type: TokenType2.LE, value: "<=" };
        }
        if (this.peek() === ">") {
          this.advance();
          return { type: TokenType2.NE, value: "<>" };
        }
        return { type: TokenType2.LT, value: "<" };
      case ">":
        if (this.peek() === "=") {
          this.advance();
          return { type: TokenType2.GE, value: ">=" };
        }
        return { type: TokenType2.GT, value: ">" };
      case "!":
        if (this.peek() === "=") {
          this.advance();
          return { type: TokenType2.NE, value: "!=" };
        }
        throw new Error(`Unexpected character: ${ch}`);
      default:
        throw new Error(`Unexpected character: ${ch}`);
    }
  }
  tokenize() {
    const tokens = [];
    let token;
    while ((token = this.nextToken()).type !== TokenType2.EOF) {
      tokens.push(token);
    }
    tokens.push(token);
    return tokens;
  }
};

// src/client/sql/parser-expr.js
function parseExpr(parser) {
  return parseOrExpr(parser);
}
function parseOrExpr(parser) {
  let left = parseAndExpr(parser);
  while (parser.match(TokenType2.OR)) {
    const right = parseAndExpr(parser);
    left = { type: "binary", op: "OR", left, right };
  }
  return left;
}
function parseAndExpr(parser) {
  let left = parseNotExpr(parser);
  while (parser.match(TokenType2.AND)) {
    const right = parseNotExpr(parser);
    left = { type: "binary", op: "AND", left, right };
  }
  return left;
}
function parseNotExpr(parser) {
  if (parser.match(TokenType2.NOT)) {
    const operand = parseNotExpr(parser);
    return { type: "unary", op: "NOT", operand };
  }
  return parseCmpExpr(parser);
}
function parseCmpExpr(parser) {
  let left = parseAddExpr(parser);
  if (parser.match(TokenType2.IS)) {
    const negated = !!parser.match(TokenType2.NOT);
    parser.expect(TokenType2.NULL);
    return {
      type: "binary",
      op: negated ? "!=" : "==",
      left,
      right: { type: "literal", value: null }
    };
  }
  if (parser.match(TokenType2.IN)) {
    parser.expect(TokenType2.LPAREN);
    if (parser.check(TokenType2.SELECT)) {
      const subquery = parser.parseSelect(true);
      parser.expect(TokenType2.RPAREN);
      return { type: "in", expr: left, values: [{ type: "subquery", query: subquery }] };
    }
    const values = [];
    values.push(parsePrimary(parser));
    while (parser.match(TokenType2.COMMA)) {
      values.push(parsePrimary(parser));
    }
    parser.expect(TokenType2.RPAREN);
    return { type: "in", expr: left, values };
  }
  if (parser.match(TokenType2.BETWEEN)) {
    const low = parseAddExpr(parser);
    parser.expect(TokenType2.AND);
    const high = parseAddExpr(parser);
    return { type: "between", expr: left, low, high };
  }
  if (parser.match(TokenType2.LIKE)) {
    const pattern = parsePrimary(parser);
    return { type: "like", expr: left, pattern };
  }
  if (parser.match(TokenType2.NEAR)) {
    const value = parsePrimary(parser);
    return { type: "near", column: left, value };
  }
  const opMap = {
    [TokenType2.EQ]: "==",
    [TokenType2.NE]: "!=",
    [TokenType2.LT]: "<",
    [TokenType2.LE]: "<=",
    [TokenType2.GT]: ">",
    [TokenType2.GE]: ">="
  };
  const opToken = parser.match(TokenType2.EQ, TokenType2.NE, TokenType2.LT, TokenType2.LE, TokenType2.GT, TokenType2.GE);
  if (opToken) {
    const right = parseAddExpr(parser);
    return { type: "binary", op: opMap[opToken.type], left, right };
  }
  return left;
}
function parseAddExpr(parser) {
  let left = parseMulExpr(parser);
  while (true) {
    const opToken = parser.match(TokenType2.PLUS, TokenType2.MINUS);
    if (!opToken) break;
    const right = parseMulExpr(parser);
    left = { type: "binary", op: opToken.value, left, right };
  }
  return left;
}
function parseMulExpr(parser) {
  let left = parseUnaryExpr(parser);
  while (true) {
    const opToken = parser.match(TokenType2.STAR, TokenType2.SLASH);
    if (!opToken) break;
    const right = parseUnaryExpr(parser);
    left = { type: "binary", op: opToken.value, left, right };
  }
  return left;
}
function parseUnaryExpr(parser) {
  if (parser.match(TokenType2.MINUS)) {
    const operand = parseUnaryExpr(parser);
    return { type: "unary", op: "-", operand };
  }
  return parsePrimary(parser);
}
function parsePrimary(parser) {
  if (parser.match(TokenType2.NULL)) {
    return { type: "literal", value: null };
  }
  if (parser.match(TokenType2.TRUE)) {
    return { type: "literal", value: true };
  }
  if (parser.match(TokenType2.FALSE)) {
    return { type: "literal", value: false };
  }
  if (parser.match(TokenType2.ARRAY)) {
    let result = parseArrayLiteral(parser);
    while (parser.check(TokenType2.LBRACKET)) {
      parser.advance();
      const index = parseExpr(parser);
      parser.expect(TokenType2.RBRACKET);
      result = { type: "subscript", array: result, index };
    }
    return result;
  }
  if (parser.check(TokenType2.LBRACKET)) {
    let result = parseArrayLiteral(parser);
    while (parser.check(TokenType2.LBRACKET)) {
      parser.advance();
      const index = parseExpr(parser);
      parser.expect(TokenType2.RBRACKET);
      result = { type: "subscript", array: result, index };
    }
    return result;
  }
  if (parser.check(TokenType2.NUMBER)) {
    const value = parser.advance().value;
    return { type: "literal", value: parseFloat(value) };
  }
  if (parser.check(TokenType2.STRING)) {
    const value = parser.advance().value;
    return { type: "literal", value };
  }
  const windowFuncTokens = [
    TokenType2.ROW_NUMBER,
    TokenType2.RANK,
    TokenType2.DENSE_RANK,
    TokenType2.NTILE,
    TokenType2.LAG,
    TokenType2.LEAD,
    TokenType2.FIRST_VALUE,
    TokenType2.LAST_VALUE,
    TokenType2.NTH_VALUE,
    TokenType2.PERCENT_RANK,
    TokenType2.CUME_DIST
  ];
  if (windowFuncTokens.some((t) => parser.check(t))) {
    const name = parser.advance().type;
    parser.expect(TokenType2.LPAREN);
    const args = [];
    if (!parser.check(TokenType2.RPAREN)) {
      args.push(parseExpr(parser));
      while (parser.match(TokenType2.COMMA)) {
        args.push(parseExpr(parser));
      }
    }
    parser.expect(TokenType2.RPAREN);
    const over = parser.parseOverClause();
    return { type: "call", name, args, distinct: false, over };
  }
  if (parser.check(TokenType2.IDENTIFIER) || parser.check(TokenType2.COUNT, TokenType2.SUM, TokenType2.AVG, TokenType2.MIN, TokenType2.MAX, TokenType2.GROUPING)) {
    const name = parser.advance().value;
    if (parser.match(TokenType2.LPAREN)) {
      let distinct = !!parser.match(TokenType2.DISTINCT);
      const args = [];
      if (!parser.check(TokenType2.RPAREN)) {
        if (parser.check(TokenType2.STAR)) {
          parser.advance();
          args.push({ type: "star" });
        } else {
          args.push(parseExpr(parser));
          while (parser.match(TokenType2.COMMA)) {
            args.push(parseExpr(parser));
          }
        }
      }
      parser.expect(TokenType2.RPAREN);
      let over = null;
      if (parser.check(TokenType2.OVER)) {
        over = parser.parseOverClause();
      }
      return { type: "call", name: name.toUpperCase(), args, distinct, over };
    }
    if (parser.match(TokenType2.DOT)) {
      const table = name;
      const token = parser.advance();
      const column = token.value || token.type.toLowerCase();
      return { type: "column", table, column };
    }
    let result = { type: "column", column: name };
    if (parser.check(TokenType2.LBRACKET)) {
      parser.advance();
      const index = parseExpr(parser);
      parser.expect(TokenType2.RBRACKET);
      result = { type: "subscript", array: result, index };
    }
    return result;
  }
  if (parser.match(TokenType2.LPAREN)) {
    if (parser.check(TokenType2.SELECT)) {
      const subquery = parser.parseSelect(true);
      parser.expect(TokenType2.RPAREN);
      return { type: "subquery", query: subquery };
    }
    const expr = parseExpr(parser);
    parser.expect(TokenType2.RPAREN);
    return expr;
  }
  if (parser.match(TokenType2.STAR)) {
    return { type: "star" };
  }
  throw new Error(`Unexpected token: ${parser.current().type} (${parser.current().value})`);
}
function parseValue(parser) {
  if (parser.match(TokenType2.NULL)) {
    return { type: "null", value: null };
  }
  if (parser.match(TokenType2.TRUE)) {
    return { type: "boolean", value: true };
  }
  if (parser.match(TokenType2.FALSE)) {
    return { type: "boolean", value: false };
  }
  if (parser.check(TokenType2.NUMBER)) {
    const token = parser.advance();
    const value = token.value.includes(".") ? parseFloat(token.value) : parseInt(token.value, 10);
    return { type: "number", value };
  }
  if (parser.check(TokenType2.STRING)) {
    const token = parser.advance();
    return { type: "string", value: token.value };
  }
  if (parser.check(TokenType2.MINUS)) {
    parser.advance();
    const token = parser.expect(TokenType2.NUMBER);
    const value = token.value.includes(".") ? -parseFloat(token.value) : -parseInt(token.value, 10);
    return { type: "number", value };
  }
  if (parser.check(TokenType2.LBRACKET)) {
    return parseArrayLiteral(parser);
  }
  throw new Error(`Expected value, got ${parser.current().type}`);
}
function parseArrayLiteral(parser) {
  parser.expect(TokenType2.LBRACKET);
  const elements = [];
  if (!parser.check(TokenType2.RBRACKET)) {
    elements.push(parseExpr(parser));
    while (parser.match(TokenType2.COMMA)) {
      elements.push(parseExpr(parser));
    }
  }
  parser.expect(TokenType2.RBRACKET);
  return { type: "array", elements };
}

// src/client/sql/parser-advanced.js
function parseWithClause(parser) {
  parser.expect(TokenType2.WITH);
  const isRecursive = !!parser.match(TokenType2.RECURSIVE);
  const ctes = [];
  do {
    const name = parser.expect(TokenType2.IDENTIFIER).value;
    let columns = [];
    if (parser.match(TokenType2.LPAREN)) {
      columns.push(parser.expect(TokenType2.IDENTIFIER).value);
      while (parser.match(TokenType2.COMMA)) {
        columns.push(parser.expect(TokenType2.IDENTIFIER).value);
      }
      parser.expect(TokenType2.RPAREN);
    }
    parser.expect(TokenType2.AS);
    parser.expect(TokenType2.LPAREN);
    const body = parseCteBody(parser, isRecursive);
    parser.expect(TokenType2.RPAREN);
    ctes.push({
      name,
      columns,
      body,
      recursive: isRecursive
    });
  } while (parser.match(TokenType2.COMMA));
  return ctes;
}
function parseCteBody(parser, isRecursive) {
  const anchor = parser.parseSelect(true, true);
  if (isRecursive && parser.match(TokenType2.UNION)) {
    parser.expect(TokenType2.ALL);
    const recursive = parser.parseSelect(true, true);
    return {
      type: "RECURSIVE_CTE",
      anchor,
      recursive
    };
  }
  return anchor;
}
function parseGroupByList(parser) {
  const items = [];
  do {
    if (parser.match(TokenType2.ROLLUP)) {
      parser.expect(TokenType2.LPAREN);
      const columns = parseColumnList(parser);
      parser.expect(TokenType2.RPAREN);
      items.push({ type: "ROLLUP", columns });
    } else if (parser.match(TokenType2.CUBE)) {
      parser.expect(TokenType2.LPAREN);
      const columns = parseColumnList(parser);
      parser.expect(TokenType2.RPAREN);
      items.push({ type: "CUBE", columns });
    } else if (parser.match(TokenType2.GROUPING)) {
      parser.expect(TokenType2.SETS);
      parser.expect(TokenType2.LPAREN);
      const sets = parseGroupingSets(parser);
      parser.expect(TokenType2.RPAREN);
      items.push({ type: "GROUPING_SETS", sets });
    } else {
      items.push({ type: "COLUMN", column: parser.expect(TokenType2.IDENTIFIER).value });
    }
  } while (parser.match(TokenType2.COMMA));
  return items;
}
function parseGroupingSets(parser) {
  const sets = [];
  do {
    parser.expect(TokenType2.LPAREN);
    if (parser.check(TokenType2.RPAREN)) {
      sets.push([]);
    } else {
      sets.push(parseColumnList(parser));
    }
    parser.expect(TokenType2.RPAREN);
  } while (parser.match(TokenType2.COMMA));
  return sets;
}
function parseColumnList(parser) {
  const columns = [parser.expect(TokenType2.IDENTIFIER).value];
  while (parser.match(TokenType2.COMMA)) {
    columns.push(parser.expect(TokenType2.IDENTIFIER).value);
  }
  return columns;
}
function parseOverClause(parser) {
  parser.expect(TokenType2.OVER);
  parser.expect(TokenType2.LPAREN);
  const over = { partitionBy: [], orderBy: [], frame: null };
  if (parser.match(TokenType2.PARTITION)) {
    parser.expect(TokenType2.BY);
    over.partitionBy.push(parser.parseExpr());
    while (parser.match(TokenType2.COMMA)) {
      over.partitionBy.push(parser.parseExpr());
    }
  }
  if (parser.match(TokenType2.ORDER)) {
    parser.expect(TokenType2.BY);
    over.orderBy = parser.parseOrderByList();
  }
  if (parser.check(TokenType2.ROWS) || parser.check(TokenType2.RANGE)) {
    over.frame = parseFrameClause(parser);
  }
  parser.expect(TokenType2.RPAREN);
  return over;
}
function parseFrameClause(parser) {
  const frameType = parser.advance().type;
  const frame = { type: frameType, start: null, end: null };
  if (parser.match(TokenType2.BETWEEN)) {
    frame.start = parseFrameBound(parser);
    parser.expect(TokenType2.AND);
    frame.end = parseFrameBound(parser);
  } else {
    frame.start = parseFrameBound(parser);
    frame.end = { type: "CURRENT ROW" };
  }
  return frame;
}
function parseFrameBound(parser) {
  if (parser.match(TokenType2.UNBOUNDED)) {
    if (parser.match(TokenType2.PRECEDING)) {
      return { type: "UNBOUNDED PRECEDING" };
    } else if (parser.match(TokenType2.FOLLOWING)) {
      return { type: "UNBOUNDED FOLLOWING" };
    }
    throw new Error("Expected PRECEDING or FOLLOWING after UNBOUNDED");
  }
  if (parser.match(TokenType2.CURRENT)) {
    parser.expect(TokenType2.ROW);
    return { type: "CURRENT ROW" };
  }
  if (parser.check(TokenType2.NUMBER)) {
    const n = parseInt(parser.advance().value, 10);
    if (parser.match(TokenType2.PRECEDING)) {
      return { type: "PRECEDING", offset: n };
    } else if (parser.match(TokenType2.FOLLOWING)) {
      return { type: "FOLLOWING", offset: n };
    }
    throw new Error("Expected PRECEDING or FOLLOWING after number");
  }
  throw new Error("Invalid frame bound");
}
function parsePivotClause(parser, parsePrimaryFn) {
  parser.expect(TokenType2.LPAREN);
  const aggFunc = parsePrimaryFn(parser);
  if (aggFunc.type !== "call") {
    throw new Error("PIVOT requires an aggregate function (e.g., SUM, COUNT, AVG)");
  }
  parser.expect(TokenType2.FOR);
  const forColumn = parser.expect(TokenType2.IDENTIFIER).value;
  parser.expect(TokenType2.IN);
  parser.expect(TokenType2.LPAREN);
  const inValues = [];
  inValues.push(parsePrimaryFn(parser).value);
  while (parser.match(TokenType2.COMMA)) {
    inValues.push(parsePrimaryFn(parser).value);
  }
  parser.expect(TokenType2.RPAREN);
  parser.expect(TokenType2.RPAREN);
  return {
    aggregate: aggFunc,
    forColumn,
    inValues
  };
}
function parseUnpivotClause(parser) {
  parser.expect(TokenType2.LPAREN);
  const valueColumn = parser.expect(TokenType2.IDENTIFIER).value;
  parser.expect(TokenType2.FOR);
  const nameColumn = parser.expect(TokenType2.IDENTIFIER).value;
  parser.expect(TokenType2.IN);
  parser.expect(TokenType2.LPAREN);
  const inColumns = [];
  inColumns.push(parser.expect(TokenType2.IDENTIFIER).value);
  while (parser.match(TokenType2.COMMA)) {
    inColumns.push(parser.expect(TokenType2.IDENTIFIER).value);
  }
  parser.expect(TokenType2.RPAREN);
  parser.expect(TokenType2.RPAREN);
  return {
    valueColumn,
    nameColumn,
    inColumns
  };
}
function parseNearClause(parser) {
  let column = null;
  let query = null;
  let searchRow = null;
  let topK = 20;
  let encoder = "minilm";
  if (parser.check(TokenType2.IDENTIFIER)) {
    const ident = parser.advance().value;
    if (parser.check(TokenType2.STRING) || parser.check(TokenType2.NUMBER)) {
      column = ident;
    } else {
      throw new Error(`NEAR requires quoted text or row number. Did you mean: NEAR '${ident}'?`);
    }
  }
  if (parser.check(TokenType2.STRING)) {
    query = parser.advance().value;
  } else if (parser.check(TokenType2.NUMBER)) {
    searchRow = parseInt(parser.advance().value, 10);
  } else {
    throw new Error("NEAR requires a quoted text string or row number");
  }
  if (parser.match(TokenType2.TOPK)) {
    topK = parseInt(parser.expect(TokenType2.NUMBER).value, 10);
  }
  return { query, searchRow, column, topK, encoder };
}

// src/client/sql/parser.js
var SQLParser = class {
  constructor(tokens) {
    this.tokens = tokens;
    this.pos = 0;
  }
  current() {
    return this.tokens[this.pos] || { type: TokenType2.EOF };
  }
  advance() {
    if (this.pos < this.tokens.length) {
      return this.tokens[this.pos++];
    }
    return { type: TokenType2.EOF };
  }
  expect(type) {
    const token = this.current();
    if (token.type !== type) {
      throw new Error(`Expected ${type}, got ${token.type} (${token.value})`);
    }
    return this.advance();
  }
  match(...types) {
    if (types.includes(this.current().type)) {
      return this.advance();
    }
    return null;
  }
  check(...types) {
    return types.includes(this.current().type);
  }
  /**
   * Parse SQL statement (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, DROP TABLE)
   */
  parse() {
    if (this.check(TokenType2.EXPLAIN)) {
      this.advance();
      const statement = this.parse();
      return { type: "EXPLAIN", statement };
    }
    let ctes = [];
    if (this.check(TokenType2.WITH)) {
      ctes = parseWithClause(this);
    }
    if (this.check(TokenType2.SELECT)) {
      const result = this.parseSelect();
      result.ctes = ctes;
      return result;
    } else if (this.check(TokenType2.INSERT)) {
      return this.parseInsert();
    } else if (this.check(TokenType2.UPDATE)) {
      return this.parseUpdate();
    } else if (this.check(TokenType2.DELETE)) {
      return this.parseDelete();
    } else if (this.check(TokenType2.CREATE)) {
      return this.parseCreateTable();
    } else if (this.check(TokenType2.DROP)) {
      return this.parseDropTable();
    } else {
      throw new Error(`Unexpected token: ${this.current().type}. Expected SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, or EXPLAIN`);
    }
  }
  /**
   * Parse SELECT statement
   * @param {boolean} isSubquery - If true, don't require EOF at end (for subqueries)
   * @param {boolean} noSetOps - If true, don't parse set operations (for CTE body parsing)
   */
  parseSelect(isSubquery = false, noSetOps = false) {
    this.expect(TokenType2.SELECT);
    const distinct = !!this.match(TokenType2.DISTINCT);
    const columns = this.parseSelectList();
    let from = null;
    if (this.match(TokenType2.FROM)) {
      from = this.parseFromClause();
    }
    const joins = [];
    while (this.check(TokenType2.JOIN) || this.check(TokenType2.INNER) || this.check(TokenType2.LEFT) || this.check(TokenType2.RIGHT) || this.check(TokenType2.FULL) || this.check(TokenType2.CROSS)) {
      joins.push(this.parseJoinClause());
    }
    let pivot = null;
    if (this.match(TokenType2.PIVOT)) {
      pivot = parsePivotClause(this, parsePrimary);
    }
    let unpivot = null;
    if (this.match(TokenType2.UNPIVOT)) {
      unpivot = parseUnpivotClause(this);
    }
    let where = null;
    if (this.match(TokenType2.WHERE)) {
      where = this.parseExpr();
    }
    let groupBy = [];
    if (this.match(TokenType2.GROUP)) {
      this.expect(TokenType2.BY);
      groupBy = parseGroupByList(this);
    }
    let having = null;
    if (this.match(TokenType2.HAVING)) {
      having = this.parseExpr();
    }
    let qualify = null;
    if (this.match(TokenType2.QUALIFY)) {
      qualify = this.parseExpr();
    }
    let search = null;
    if (this.match(TokenType2.NEAR)) {
      search = parseNearClause(this);
    }
    const baseAst = {
      type: "SELECT",
      distinct,
      columns,
      from,
      joins,
      pivot,
      unpivot,
      where,
      groupBy,
      having,
      qualify,
      search,
      orderBy: [],
      limit: null,
      offset: null
    };
    if (!noSetOps && (this.check(TokenType2.UNION) || this.check(TokenType2.INTERSECT) || this.check(TokenType2.EXCEPT))) {
      const operator = this.advance().type;
      const all = !!this.match(TokenType2.ALL);
      const right = this.parseSelect(true, true);
      let orderBy = [];
      let limit = null;
      let offset = null;
      if (this.match(TokenType2.ORDER)) {
        this.expect(TokenType2.BY);
        orderBy = this.parseOrderByList();
      }
      if (this.match(TokenType2.LIMIT)) {
        limit = parseInt(this.expect(TokenType2.NUMBER).value, 10);
      }
      if (orderBy.length === 0 && this.match(TokenType2.ORDER)) {
        this.expect(TokenType2.BY);
        orderBy = this.parseOrderByList();
      }
      if (this.match(TokenType2.OFFSET)) {
        offset = parseInt(this.expect(TokenType2.NUMBER).value, 10);
      }
      if (!isSubquery && this.current().type !== TokenType2.EOF) {
        throw new Error(`Unexpected token after query: ${this.current().type} (${this.current().value}). Check your SQL syntax.`);
      }
      return {
        type: "SET_OPERATION",
        operator,
        all,
        left: baseAst,
        right,
        orderBy,
        limit,
        offset
      };
    }
    if (!noSetOps) {
      let orderBy = [];
      let limit = null;
      let offset = null;
      if (this.match(TokenType2.ORDER)) {
        this.expect(TokenType2.BY);
        orderBy = this.parseOrderByList();
      }
      if (this.match(TokenType2.LIMIT)) {
        limit = parseInt(this.expect(TokenType2.NUMBER).value, 10);
      }
      if (orderBy.length === 0 && this.match(TokenType2.ORDER)) {
        this.expect(TokenType2.BY);
        orderBy = this.parseOrderByList();
      }
      if (this.match(TokenType2.OFFSET)) {
        offset = parseInt(this.expect(TokenType2.NUMBER).value, 10);
      }
      baseAst.orderBy = orderBy;
      baseAst.limit = limit;
      baseAst.offset = offset;
    }
    if (!isSubquery && this.current().type !== TokenType2.EOF) {
      throw new Error(`Unexpected token after query: ${this.current().type} (${this.current().value}). Check your SQL syntax.`);
    }
    return baseAst;
  }
  /**
   * Parse INSERT statement
   */
  parseInsert() {
    this.expect(TokenType2.INSERT);
    this.expect(TokenType2.INTO);
    const table = this.expect(TokenType2.IDENTIFIER).value;
    let columns = null;
    if (this.match(TokenType2.LPAREN)) {
      columns = [];
      columns.push(this.expect(TokenType2.IDENTIFIER).value);
      while (this.match(TokenType2.COMMA)) {
        columns.push(this.expect(TokenType2.IDENTIFIER).value);
      }
      this.expect(TokenType2.RPAREN);
    }
    this.expect(TokenType2.VALUES);
    const rows = [];
    do {
      this.expect(TokenType2.LPAREN);
      const values = [];
      values.push(parseValue(this));
      while (this.match(TokenType2.COMMA)) {
        values.push(parseValue(this));
      }
      this.expect(TokenType2.RPAREN);
      rows.push(values);
    } while (this.match(TokenType2.COMMA));
    return { type: "INSERT", table, columns, rows };
  }
  /**
   * Parse UPDATE statement
   */
  parseUpdate() {
    this.expect(TokenType2.UPDATE);
    const table = this.expect(TokenType2.IDENTIFIER).value;
    this.expect(TokenType2.SET);
    const assignments = [];
    do {
      const column = this.expect(TokenType2.IDENTIFIER).value;
      this.expect(TokenType2.EQ);
      const value = parseValue(this);
      assignments.push({ column, value });
    } while (this.match(TokenType2.COMMA));
    let where = null;
    if (this.match(TokenType2.WHERE)) {
      where = this.parseExpr();
    }
    return { type: "UPDATE", table, assignments, where };
  }
  /**
   * Parse DELETE statement
   */
  parseDelete() {
    this.expect(TokenType2.DELETE);
    this.expect(TokenType2.FROM);
    const table = this.expect(TokenType2.IDENTIFIER).value;
    let where = null;
    if (this.match(TokenType2.WHERE)) {
      where = this.parseExpr();
    }
    return { type: "DELETE", table, where };
  }
  /**
   * Parse CREATE TABLE statement
   */
  parseCreateTable() {
    this.expect(TokenType2.CREATE);
    this.expect(TokenType2.TABLE);
    let ifNotExists = false;
    if (this.match(TokenType2.IF)) {
      this.expect(TokenType2.NOT);
      this.expect(TokenType2.EXISTS);
      ifNotExists = true;
    }
    const table = this.expect(TokenType2.IDENTIFIER).value;
    this.expect(TokenType2.LPAREN);
    const columns = [];
    do {
      const name = this.expect(TokenType2.IDENTIFIER).value;
      let dataType = "TEXT";
      let primaryKey = false;
      let vectorDim = null;
      if (this.check(TokenType2.INT) || this.check(TokenType2.INTEGER) || this.check(TokenType2.BIGINT)) {
        this.advance();
        dataType = "INT64";
      } else if (this.check(TokenType2.FLOAT) || this.check(TokenType2.REAL) || this.check(TokenType2.DOUBLE)) {
        this.advance();
        dataType = "FLOAT64";
      } else if (this.check(TokenType2.TEXT) || this.check(TokenType2.VARCHAR)) {
        this.advance();
        dataType = "STRING";
      } else if (this.check(TokenType2.BOOLEAN) || this.check(TokenType2.BOOL)) {
        this.advance();
        dataType = "BOOL";
      } else if (this.check(TokenType2.VECTOR)) {
        this.advance();
        dataType = "VECTOR";
        if (this.match(TokenType2.LPAREN)) {
          vectorDim = parseInt(this.expect(TokenType2.NUMBER).value, 10);
          this.expect(TokenType2.RPAREN);
        }
      }
      if (this.match(TokenType2.PRIMARY)) {
        this.expect(TokenType2.KEY);
        primaryKey = true;
      }
      columns.push({ name, dataType, primaryKey, vectorDim });
    } while (this.match(TokenType2.COMMA));
    this.expect(TokenType2.RPAREN);
    return { type: "CREATE_TABLE", table, columns, ifNotExists };
  }
  /**
   * Parse DROP TABLE statement
   */
  parseDropTable() {
    this.expect(TokenType2.DROP);
    this.expect(TokenType2.TABLE);
    let ifExists = false;
    if (this.match(TokenType2.IF)) {
      this.expect(TokenType2.EXISTS);
      ifExists = true;
    }
    const table = this.expect(TokenType2.IDENTIFIER).value;
    return { type: "DROP_TABLE", table, ifExists };
  }
  parseSelectList() {
    const items = [this.parseSelectItem()];
    while (this.match(TokenType2.COMMA)) {
      items.push(this.parseSelectItem());
    }
    return items;
  }
  parseSelectItem() {
    if (this.match(TokenType2.STAR)) {
      return { type: "star" };
    }
    const expr = this.parseExpr();
    let alias = null;
    if (this.match(TokenType2.AS)) {
      alias = this.expect(TokenType2.IDENTIFIER).value;
    } else if (this.check(TokenType2.IDENTIFIER) && !this.check(TokenType2.FROM, TokenType2.WHERE, TokenType2.ORDER, TokenType2.LIMIT, TokenType2.GROUP, TokenType2.JOIN, TokenType2.INNER, TokenType2.LEFT, TokenType2.RIGHT, TokenType2.COMMA)) {
      alias = this.advance().value;
    }
    return { type: "expr", expr, alias };
  }
  /**
   * Parse FROM clause
   */
  parseFromClause() {
    let from = null;
    if (this.check(TokenType2.STRING)) {
      const url = this.advance().value;
      from = { type: "url", url };
    } else if (this.check(TokenType2.IDENTIFIER)) {
      const name = this.advance().value;
      if (this.match(TokenType2.LPAREN)) {
        const funcName = name.toLowerCase();
        if (funcName === "read_lance") {
          from = { type: "url", function: "read_lance" };
          if (!this.check(TokenType2.RPAREN)) {
            if (this.match(TokenType2.FILE)) {
              from.isFile = true;
              if (this.match(TokenType2.COMMA)) {
                from.version = parseInt(this.expect(TokenType2.NUMBER).value, 10);
              }
            } else if (this.check(TokenType2.STRING)) {
              from.url = this.advance().value;
              if (this.match(TokenType2.COMMA)) {
                from.version = parseInt(this.expect(TokenType2.NUMBER).value, 10);
              }
            }
          }
          this.expect(TokenType2.RPAREN);
        } else {
          throw new Error(`Unknown table function: ${name}. Supported: read_lance()`);
        }
      } else {
        from = { type: "table", name };
      }
    } else {
      throw new Error("Expected table name, URL string, or read_lance() after FROM");
    }
    if (from) {
      if (this.match(TokenType2.AS)) {
        from.alias = this.expect(TokenType2.IDENTIFIER).value;
      } else if (this.check(TokenType2.IDENTIFIER) && !this.check(TokenType2.WHERE, TokenType2.ORDER, TokenType2.LIMIT, TokenType2.GROUP, TokenType2.NEAR, TokenType2.JOIN, TokenType2.INNER, TokenType2.LEFT, TokenType2.RIGHT, TokenType2.COMMA)) {
        from.alias = this.advance().value;
      }
    }
    return from;
  }
  /**
   * Parse JOIN clause
   */
  parseJoinClause() {
    let joinType = "INNER";
    if (this.match(TokenType2.INNER)) {
      this.expect(TokenType2.JOIN);
      joinType = "INNER";
    } else if (this.match(TokenType2.LEFT)) {
      this.match(TokenType2.OUTER);
      this.expect(TokenType2.JOIN);
      joinType = "LEFT";
    } else if (this.match(TokenType2.RIGHT)) {
      this.match(TokenType2.OUTER);
      this.expect(TokenType2.JOIN);
      joinType = "RIGHT";
    } else if (this.match(TokenType2.FULL)) {
      this.match(TokenType2.OUTER);
      this.expect(TokenType2.JOIN);
      joinType = "FULL";
    } else if (this.match(TokenType2.CROSS)) {
      this.expect(TokenType2.JOIN);
      joinType = "CROSS";
    } else {
      this.expect(TokenType2.JOIN);
    }
    const table = this.parseFromClause();
    const alias = table.alias || null;
    let on = null;
    if (joinType !== "CROSS") {
      this.expect(TokenType2.ON);
      on = this.parseExpr();
    }
    return { type: joinType, table, alias, on };
  }
  parseOrderByList() {
    const items = [this.parseOrderByItem()];
    while (this.match(TokenType2.COMMA)) {
      items.push(this.parseOrderByItem());
    }
    return items;
  }
  parseOrderByItem() {
    const column = this.expect(TokenType2.IDENTIFIER).value;
    let descending = false;
    if (this.match(TokenType2.DESC)) {
      descending = true;
    } else {
      this.match(TokenType2.ASC);
    }
    return { column, descending };
  }
  // Expression parsing - delegate to parser-expr.js
  parseExpr() {
    return parseExpr(this);
  }
  // Window function OVER clause - delegate to parser-advanced.js
  parseOverClause() {
    return parseOverClause(this);
  }
};

// src/client/sql/statistics-manager.js
var StatisticsManager = class {
  constructor() {
    this._cache = /* @__PURE__ */ new Map();
    this._opfsRoot = null;
    this._computing = /* @__PURE__ */ new Map();
  }
  async _getStatsDir() {
    if (this._opfsRoot) return this._opfsRoot;
    if (typeof navigator === "undefined" || !navigator.storage?.getDirectory) {
      return null;
    }
    try {
      const opfsRoot = await navigator.storage.getDirectory();
      this._opfsRoot = await opfsRoot.getDirectoryHandle("lanceql-stats", { create: true });
      return this._opfsRoot;
    } catch {
      return null;
    }
  }
  _getCacheKey(datasetUrl) {
    let hash = 0;
    for (let i = 0; i < datasetUrl.length; i++) {
      hash = (hash << 5) - hash + datasetUrl.charCodeAt(i);
      hash |= 0;
    }
    return `stats_${Math.abs(hash).toString(16)}`;
  }
  async loadFromCache(datasetUrl, version) {
    const cacheKey = this._getCacheKey(datasetUrl);
    if (this._cache.has(cacheKey)) {
      const cached = this._cache.get(cacheKey);
      if (cached.version === version) {
        return cached;
      }
    }
    const statsDir = await this._getStatsDir();
    if (!statsDir) return null;
    try {
      const fileHandle = await statsDir.getFileHandle(`${cacheKey}.json`);
      const file = await fileHandle.getFile();
      const text = await file.text();
      const cached = JSON.parse(text);
      if (cached.version !== version) {
        return null;
      }
      this._cache.set(cacheKey, cached);
      return cached;
    } catch (e) {
      return null;
    }
  }
  async saveToCache(datasetUrl, version, statistics) {
    const cacheKey = this._getCacheKey(datasetUrl);
    const cacheData = {
      datasetUrl,
      version,
      timestamp: Date.now(),
      columns: statistics.columns,
      fragments: statistics.fragments || null
    };
    this._cache.set(cacheKey, cacheData);
    const statsDir = await this._getStatsDir();
    if (!statsDir) return;
    try {
      const fileHandle = await statsDir.getFileHandle(`${cacheKey}.json`, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(JSON.stringify(cacheData));
      await writable.close();
    } catch {
    }
  }
  async getColumnStats(dataset, columnName, options = {}) {
    const datasetUrl = dataset.baseUrl;
    const version = dataset._version;
    const sampleSize = options.sampleSize || 1e5;
    const cached = await this.loadFromCache(datasetUrl, version);
    if (cached?.columns?.[columnName]) {
      return cached.columns[columnName];
    }
    const computeKey = `${datasetUrl}:${columnName}`;
    if (this._computing.has(computeKey)) {
      return this._computing.get(computeKey);
    }
    const computePromise = this._computeColumnStats(dataset, columnName, sampleSize);
    this._computing.set(computeKey, computePromise);
    try {
      const stats = await computePromise;
      const existing = await this.loadFromCache(datasetUrl, version) || { columns: {} };
      existing.columns[columnName] = stats;
      await this.saveToCache(datasetUrl, version, existing);
      return stats;
    } finally {
      this._computing.delete(computeKey);
    }
  }
  async _computeColumnStats(dataset, columnName, sampleSize) {
    const colIdx = dataset.schema.findIndex((c) => c.name === columnName);
    if (colIdx === -1) {
      throw new Error(`Column not found: ${columnName}`);
    }
    const colType = dataset._columnTypes?.[colIdx] || "unknown";
    const isNumeric = ["int8", "int16", "int32", "int64", "float32", "float64", "double"].includes(colType);
    const stats = {
      column: columnName,
      type: colType,
      rowCount: 0,
      nullCount: 0,
      min: null,
      max: null,
      computed: true,
      sampleSize: 0
    };
    let rowsProcessed = 0;
    for (let fragIdx = 0; fragIdx < dataset._fragments.length && rowsProcessed < sampleSize; fragIdx++) {
      try {
        const fragFile = await dataset.openFragment(fragIdx);
        const fragRows = Math.min(
          dataset._fragments[fragIdx].numRows,
          sampleSize - rowsProcessed
        );
        const indices = Array.from({ length: fragRows }, (_, i) => i);
        const values = await fragFile.readColumnAtIndices(colIdx, indices);
        for (const value of values) {
          stats.rowCount++;
          stats.sampleSize++;
          if (value === null || value === void 0) {
            stats.nullCount++;
            continue;
          }
          if (isNumeric) {
            if (stats.min === null || value < stats.min) stats.min = value;
            if (stats.max === null || value > stats.max) stats.max = value;
          }
        }
        rowsProcessed += values.length;
      } catch {
      }
    }
    return stats;
  }
  async precomputeForPlan(dataset, plan) {
    const filterColumns = /* @__PURE__ */ new Set();
    for (const filter of plan.pushedFilters || []) {
      if (filter.column) filterColumns.add(filter.column);
      if (filter.left?.column) filterColumns.add(filter.left.column);
      if (filter.right?.column) filterColumns.add(filter.right.column);
    }
    const statsPromises = Array.from(filterColumns).map(
      (col) => this.getColumnStats(dataset, col).catch(() => null)
    );
    const results = await Promise.all(statsPromises);
    const statsMap = /* @__PURE__ */ new Map();
    Array.from(filterColumns).forEach((col, i) => {
      if (results[i]) statsMap.set(col, results[i]);
    });
    return statsMap;
  }
  canMatchFragment(fragmentStats, filter) {
    if (!fragmentStats || !filter) return true;
    const colStats = fragmentStats[filter.column];
    if (!colStats || colStats.min === null || colStats.max === null) return true;
    switch (filter.type) {
      case "equality":
        return filter.value >= colStats.min && filter.value <= colStats.max;
      case "range":
        switch (filter.op) {
          case ">":
            return colStats.max > filter.value;
          case ">=":
            return colStats.max >= filter.value;
          case "<":
            return colStats.min < filter.value;
          case "<=":
            return colStats.min <= filter.value;
        }
        break;
      case "between":
        return colStats.max >= filter.low && colStats.min <= filter.high;
      case "in":
        if (Array.isArray(filter.values)) {
          return filter.values.some((v) => v >= colStats.min && v <= colStats.max);
        }
        break;
    }
    return true;
  }
  async getFragmentStats(dataset, columnName, fragmentIndex) {
    const datasetUrl = dataset.baseUrl;
    const version = dataset._version;
    const cached = await this.loadFromCache(datasetUrl, version);
    if (cached?.fragments?.[fragmentIndex]?.[columnName]) {
      return cached.fragments[fragmentIndex][columnName];
    }
    const colIdx = dataset.schema.findIndex((c) => c.name === columnName);
    if (colIdx === -1) return null;
    const colType = dataset._columnTypes?.[colIdx] || "unknown";
    const isNumeric = ["int8", "int16", "int32", "int64", "float32", "float64", "double"].includes(colType);
    try {
      const fragFile = await dataset.openFragment(fragmentIndex);
      const fragRows = dataset._fragments[fragmentIndex].numRows;
      const sampleSize = Math.min(fragRows, 1e4);
      const indices = Array.from({ length: sampleSize }, (_, i) => i);
      const values = await fragFile.readColumnAtIndices(colIdx, indices);
      const stats = {
        fragmentIndex,
        column: columnName,
        rowCount: fragRows,
        sampledRows: sampleSize,
        nullCount: 0,
        min: null,
        max: null
      };
      for (const value of values) {
        if (value === null || value === void 0) {
          stats.nullCount++;
          continue;
        }
        if (isNumeric) {
          if (stats.min === null || value < stats.min) stats.min = value;
          if (stats.max === null || value > stats.max) stats.max = value;
        }
      }
      const existing = await this.loadFromCache(datasetUrl, version) || { columns: {}, fragments: {} };
      if (!existing.fragments) existing.fragments = {};
      if (!existing.fragments[fragmentIndex]) existing.fragments[fragmentIndex] = {};
      existing.fragments[fragmentIndex][columnName] = stats;
      await this.saveToCache(datasetUrl, version, existing);
      return stats;
    } catch {
      return null;
    }
  }
  async getPrunableFragments(dataset, filters) {
    if (!filters || filters.length === 0 || !dataset._fragments) {
      return null;
    }
    const numFragments = dataset._fragments.length;
    const matchingFragments = [];
    let fragmentsPruned = 0;
    const filterColumns = /* @__PURE__ */ new Set();
    for (const filter of filters) {
      if (filter.column) filterColumns.add(filter.column);
    }
    for (let fragIdx = 0; fragIdx < numFragments; fragIdx++) {
      let canPrune = false;
      for (const filter of filters) {
        if (!filter.column) continue;
        const fragStats = await this.getFragmentStats(dataset, filter.column, fragIdx);
        if (fragStats && !this.canMatchFragment({ [filter.column]: fragStats }, filter)) {
          canPrune = true;
          break;
        }
      }
      if (!canPrune) {
        matchingFragments.push(fragIdx);
      } else {
        fragmentsPruned++;
      }
    }
    return {
      matchingFragments,
      fragmentsPruned,
      totalFragments: numFragments
    };
  }
};
var statisticsManager = new StatisticsManager();

// src/client/sql/cost-model.js
var CostModel = class {
  constructor(options = {}) {
    this.isRemote = options.isRemote ?? true;
    this.rttLatency = options.rttLatency ?? 50;
    this.bandwidthMBps = options.bandwidthMBps ?? 10;
    this.filterCostPerRow = options.filterCostPerRow ?? 1e-3;
    this.hashBuildCostPerRow = options.hashBuildCostPerRow ?? 0.01;
    this.hashProbeCostPerRow = options.hashProbeCostPerRow ?? 5e-3;
    this.memoryLimitMB = options.memoryLimitMB ?? 512;
  }
  estimateScanCost(rowCount, columnBytes, selectivity = 1) {
    const bytesToFetch = rowCount * columnBytes * selectivity;
    const networkCost = this.isRemote ? this.rttLatency + bytesToFetch / (this.bandwidthMBps * 1024 * 1024) * 1e3 : 0.1;
    const cpuCost = rowCount * this.filterCostPerRow;
    return {
      totalMs: networkCost + cpuCost,
      networkMs: networkCost,
      cpuMs: cpuCost,
      bytesToFetch,
      rowsToScan: rowCount * selectivity
    };
  }
  estimateJoinCost(leftRows, rightRows, leftBytes, rightBytes, joinSelectivity = 0.1) {
    const buildRows = Math.min(leftRows, rightRows);
    const buildBytes = buildRows < leftRows ? leftBytes : rightBytes;
    const buildCost = buildRows * this.hashBuildCostPerRow;
    const probeRows = Math.max(leftRows, rightRows);
    const probeCost = probeRows * this.hashProbeCostPerRow;
    const buildMemoryMB = buildRows * buildBytes / (1024 * 1024);
    const needsSpill = buildMemoryMB > this.memoryLimitMB;
    const spillCost = needsSpill ? buildMemoryMB * 10 : 0;
    return {
      totalMs: buildCost + probeCost + spillCost,
      buildMs: buildCost,
      probeMs: probeCost,
      spillMs: spillCost,
      needsSpill,
      outputRows: Math.round(leftRows * rightRows * joinSelectivity)
    };
  }
  estimateAggregateCost(inputRows, groupCount, aggCount) {
    const hashGroupCost = inputRows * this.hashBuildCostPerRow;
    const aggComputeCost = inputRows * aggCount * 1e-4;
    return {
      totalMs: hashGroupCost + aggComputeCost,
      outputRows: groupCount
    };
  }
  comparePlans(planA, planB) {
    const costA = planA.totalCost || this.estimatePlanCost(planA);
    const costB = planB.totalCost || this.estimatePlanCost(planB);
    return {
      recommended: costA.totalMs < costB.totalMs ? "A" : "B",
      costA,
      costB,
      savings: Math.abs(costA.totalMs - costB.totalMs)
    };
  }
  estimatePlanCost(plan) {
    let totalMs = 0;
    let totalBytes = 0;
    let operations = [];
    if (plan.leftScan) {
      const scanCost = this.estimateScanCost(
        plan.leftScan.estimatedRows || 1e4,
        plan.leftScan.columnBytes || 100,
        plan.leftScan.selectivity || 1
      );
      totalMs += scanCost.totalMs;
      totalBytes += scanCost.bytesToFetch;
      operations.push({ op: "scan_left", ...scanCost });
    }
    if (plan.rightScan) {
      const scanCost = this.estimateScanCost(
        plan.rightScan.estimatedRows || 1e4,
        plan.rightScan.columnBytes || 100,
        plan.rightScan.selectivity || 1
      );
      totalMs += scanCost.totalMs;
      totalBytes += scanCost.bytesToFetch;
      operations.push({ op: "scan_right", ...scanCost });
    }
    if (plan.join) {
      const joinCost = this.estimateJoinCost(
        plan.leftScan?.estimatedRows || 1e4,
        plan.rightScan?.estimatedRows || 1e4,
        plan.leftScan?.columnBytes || 100,
        plan.rightScan?.columnBytes || 100,
        plan.join.selectivity || 0.1
      );
      totalMs += joinCost.totalMs;
      operations.push({ op: "join", ...joinCost });
    }
    if (plan.aggregations && plan.aggregations.length > 0) {
      const aggCost = this.estimateAggregateCost(
        plan.estimatedInputRows || 1e4,
        plan.groupBy?.length || 1,
        plan.aggregations.length
      );
      totalMs += aggCost.totalMs;
      operations.push({ op: "aggregate", ...aggCost });
    }
    return {
      totalMs,
      totalBytes,
      operations,
      isRemote: this.isRemote
    };
  }
};

// src/client/sql/planner-single.js
function planSingleTable(planner, ast) {
  const plan = {
    type: ast.type,
    scanColumns: [],
    pushedFilters: [],
    postFilters: [],
    aggregations: [],
    groupBy: [],
    having: null,
    orderBy: [],
    limit: ast.limit || null,
    offset: ast.offset || 0,
    projection: [],
    canUseStatistics: false,
    canStreamResults: true,
    estimatedSelectivity: 1
  };
  const neededColumns = /* @__PURE__ */ new Set();
  if (ast.columns === "*" || Array.isArray(ast.columns) && ast.columns.some((c) => c.type === "star")) {
    plan.projection = ["*"];
    plan.canStreamResults = false;
  } else if (Array.isArray(ast.columns)) {
    for (const col of ast.columns) {
      collectColumnsFromSelectItem(col, neededColumns, plan);
    }
  }
  if (ast.where) {
    collectColumnsFromExpr(ast.where, neededColumns);
    analyzeFilterPushdown(ast.where, plan);
  }
  if (ast.groupBy && ast.groupBy.length > 0) {
    for (const groupExpr of ast.groupBy) {
      collectColumnsFromExpr(groupExpr, neededColumns);
      plan.groupBy.push(groupExpr);
    }
  }
  if (ast.having) {
    collectColumnsFromExpr(ast.having, neededColumns);
    plan.having = ast.having;
  }
  if (ast.orderBy && ast.orderBy.length > 0) {
    for (const orderItem of ast.orderBy) {
      collectColumnsFromExpr(orderItem.expr || orderItem, neededColumns);
      plan.orderBy.push(orderItem);
    }
  }
  plan.scanColumns = Array.from(neededColumns);
  plan.estimatedSelectivity = estimateSelectivity(plan.pushedFilters);
  plan.canUseStatistics = plan.pushedFilters.some(
    (f) => f.type === "range" || f.type === "equality"
  );
  return plan;
}
function collectColumnsFromSelectItem(item, columns, plan) {
  if (item.type === "star") {
    plan.projection.push("*");
    return;
  }
  if (item.type === "expr") {
    const expr = item.expr;
    if (expr.type === "call") {
      const funcName = expr.name.toUpperCase();
      const aggFuncs = ["COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "STDDEV_SAMP", "STDDEV_POP", "VARIANCE", "VAR_SAMP", "VAR_POP", "MEDIAN", "STRING_AGG", "GROUP_CONCAT"];
      if (aggFuncs.includes(funcName)) {
        const agg = {
          type: funcName,
          column: null,
          alias: item.alias || `${funcName}(${expr.args[0]?.name || "*"})`,
          distinct: expr.distinct || false
        };
        if (expr.args && expr.args.length > 0) {
          const arg = expr.args[0];
          if (arg.type === "column") {
            agg.column = arg.name || arg.column;
            columns.add(agg.column);
          } else if (arg.type !== "star") {
            collectColumnsFromExpr(arg, columns);
          }
        }
        plan.aggregations.push(agg);
        plan.projection.push({ type: "aggregation", index: plan.aggregations.length - 1 });
        return;
      }
    }
    collectColumnsFromExpr(expr, columns);
    plan.projection.push({
      type: "column",
      expr,
      alias: item.alias
    });
  }
}
function collectColumnsFromExpr(expr, columns) {
  if (!expr) return;
  if (expr.type === "column") {
    columns.add(expr.name || expr.column);
  } else if (expr.type === "binary") {
    collectColumnsFromExpr(expr.left, columns);
    collectColumnsFromExpr(expr.right, columns);
  } else if (expr.type === "call") {
    for (const arg of expr.args || []) {
      collectColumnsFromExpr(arg, columns);
    }
  } else if (expr.type === "unary") {
    collectColumnsFromExpr(expr.operand, columns);
  }
}
function analyzeFilterPushdown(expr, plan) {
  if (!expr) return;
  if (expr.type === "binary") {
    if (isPushableFilter(expr)) {
      plan.pushedFilters.push(classifyFilter(expr));
    } else if (expr.op === "AND") {
      analyzeFilterPushdown(expr.left, plan);
      analyzeFilterPushdown(expr.right, plan);
    } else if (expr.op === "OR") {
      const leftPushable = isPushableFilter(expr.left);
      const rightPushable = isPushableFilter(expr.right);
      if (leftPushable && rightPushable) {
        plan.pushedFilters.push({
          type: "or",
          left: classifyFilter(expr.left),
          right: classifyFilter(expr.right)
        });
      } else {
        plan.postFilters.push(expr);
      }
    } else {
      plan.postFilters.push(expr);
    }
  } else {
    plan.postFilters.push(expr);
  }
}
function isPushableFilter(expr) {
  if (expr.type !== "binary") return false;
  const compOps = ["=", "==", "!=", "<>", "<", "<=", ">", ">=", "LIKE", "IN", "BETWEEN"];
  if (!compOps.includes(expr.op.toUpperCase())) return false;
  const leftIsCol = expr.left.type === "column";
  const rightIsCol = expr.right?.type === "column";
  const leftIsLiteral = expr.left.type === "literal" || expr.left.type === "list";
  const rightIsLiteral = expr.right?.type === "literal" || expr.right?.type === "list";
  return leftIsCol && rightIsLiteral || rightIsCol && leftIsLiteral;
}
function classifyFilter(expr) {
  const leftIsCol = expr.left.type === "column";
  const column = leftIsCol ? expr.left.name || expr.left.column : expr.right.name || expr.right.column;
  const value = leftIsCol ? expr.right.value : expr.left.value;
  const op = expr.op.toUpperCase();
  if (op === "=" || op === "==") {
    return { type: "equality", column, value, op: "=" };
  } else if (op === "!=" || op === "<>") {
    return { type: "inequality", column, value, op: "!=" };
  } else if (["<", "<=", ">", ">="].includes(op)) {
    return { type: "range", column, value, op };
  } else if (op === "LIKE") {
    return { type: "like", column, pattern: value };
  } else if (op === "IN") {
    const values = expr.right.type === "list" ? expr.right.values : [expr.right.value];
    return { type: "in", column, values };
  } else if (op === "BETWEEN") {
    return { type: "between", column, low: expr.right.low, high: expr.right.high };
  }
  return { type: "unknown", expr };
}
function estimateSelectivity(filters) {
  if (filters.length === 0) return 1;
  let selectivity = 1;
  for (const f of filters) {
    switch (f.type) {
      case "equality":
        selectivity *= 0.1;
        break;
      case "range":
        selectivity *= 0.3;
        break;
      case "in":
        selectivity *= Math.min(0.5, f.values.length * 0.05);
        break;
      case "like":
        selectivity *= f.pattern.startsWith("%") ? 0.5 : 0.2;
        break;
      default:
        selectivity *= 0.5;
    }
  }
  return Math.max(0.01, selectivity);
}

// src/client/sql/query-planner.js
var QueryPlanner = class {
  plan(ast, context) {
    const { leftTableName, leftAlias, rightTableName, rightAlias } = context;
    const columnAnalysis = this._analyzeColumns(ast, context);
    const filterAnalysis = this._analyzeFilters(ast, context);
    const fetchEstimate = this._estimateFetchSize(ast, filterAnalysis);
    return {
      leftScan: {
        table: leftTableName,
        alias: leftAlias,
        columns: columnAnalysis.left.all,
        filters: filterAnalysis.left,
        limit: fetchEstimate.left,
        purpose: {
          join: columnAnalysis.left.join,
          where: columnAnalysis.left.where,
          result: columnAnalysis.left.result
        }
      },
      rightScan: {
        table: rightTableName,
        alias: rightAlias,
        columns: columnAnalysis.right.all,
        filters: filterAnalysis.right,
        filterByJoinKeys: true,
        purpose: {
          join: columnAnalysis.right.join,
          where: columnAnalysis.right.where,
          result: columnAnalysis.right.result
        }
      },
      join: {
        type: ast.joins[0].type,
        leftKey: columnAnalysis.joinKeys.left,
        rightKey: columnAnalysis.joinKeys.right,
        algorithm: "HASH_JOIN"
      },
      projection: columnAnalysis.resultColumns,
      limit: ast.limit || null,
      offset: ast.offset || 0
    };
  }
  planSingleTable(ast) {
    return planSingleTable(this, ast);
  }
  _analyzeColumns(ast, context) {
    const { leftAlias, rightAlias } = context;
    const left = { join: /* @__PURE__ */ new Set(), where: /* @__PURE__ */ new Set(), result: /* @__PURE__ */ new Set(), all: [] };
    const right = { join: /* @__PURE__ */ new Set(), where: /* @__PURE__ */ new Set(), result: /* @__PURE__ */ new Set(), all: [] };
    for (const item of ast.columns) {
      if (item.type === "star") {
        left.result.add("*");
        right.result.add("*");
      } else if (item.type === "expr" && item.expr.type === "column") {
        const col = item.expr;
        const table = col.table || null;
        const column = col.column;
        if (!table || table === leftAlias) left.result.add(column);
        if (!table || table === rightAlias) right.result.add(column);
      }
    }
    const join = ast.joins[0];
    const joinKeys = this._extractJoinKeys(join.on, leftAlias, rightAlias);
    if (joinKeys.left) left.join.add(joinKeys.left);
    if (joinKeys.right) right.join.add(joinKeys.right);
    if (ast.where) {
      this._extractWhereColumns(ast.where, leftAlias, rightAlias, left.where, right.where);
    }
    left.all = [.../* @__PURE__ */ new Set([...left.join, ...left.where, ...left.result])];
    right.all = [.../* @__PURE__ */ new Set([...right.join, ...right.where, ...right.result])];
    if (left.result.has("*")) left.all = ["*"];
    if (right.result.has("*")) right.all = ["*"];
    const resultColumns = [];
    for (const item of ast.columns) {
      if (item.type === "star") {
        resultColumns.push("*");
      } else if (item.type === "expr" && item.expr.type === "column") {
        const col = item.expr;
        const alias = item.alias || `${col.table || ""}.${col.column}`.replace(/^\./, "");
        resultColumns.push({ table: col.table, column: col.column, alias });
      }
    }
    return { left, right, joinKeys, resultColumns };
  }
  _extractJoinKeys(onExpr, leftAlias, rightAlias) {
    if (!onExpr || onExpr.type !== "binary") return { left: null, right: null };
    const leftCol = onExpr.left;
    const rightCol = onExpr.right;
    let leftKey = null, rightKey = null;
    if (leftCol.type === "column") {
      if (!leftCol.table || leftCol.table === leftAlias) leftKey = leftCol.column;
      else if (leftCol.table === rightAlias) rightKey = leftCol.column;
    }
    if (rightCol.type === "column") {
      if (!rightCol.table || rightCol.table === leftAlias) leftKey = rightCol.column;
      else if (rightCol.table === rightAlias) rightKey = rightCol.column;
    }
    return { left: leftKey, right: rightKey };
  }
  _extractWhereColumns(expr, leftAlias, rightAlias, leftCols, rightCols) {
    if (!expr) return;
    if (expr.type === "column") {
      const table = expr.table;
      const column = expr.column;
      if (!table || table === leftAlias) leftCols.add(column);
      else if (table === rightAlias) rightCols.add(column);
    } else if (expr.type === "binary") {
      this._extractWhereColumns(expr.left, leftAlias, rightAlias, leftCols, rightCols);
      this._extractWhereColumns(expr.right, leftAlias, rightAlias, leftCols, rightCols);
    } else if (expr.type === "unary") {
      this._extractWhereColumns(expr.expr, leftAlias, rightAlias, leftCols, rightCols);
    }
  }
  _analyzeFilters(ast, context) {
    const { leftAlias, rightAlias } = context;
    const left = [], right = [], join = [];
    if (ast.where) {
      this._separateFilters(ast.where, leftAlias, rightAlias, left, right, join);
    }
    return { left, right, join };
  }
  _separateFilters(expr, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters) {
    if (!expr) return;
    if (expr.type === "binary" && expr.op === "AND") {
      this._separateFilters(expr.left, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters);
      this._separateFilters(expr.right, leftAlias, rightAlias, leftFilters, rightFilters, joinFilters);
      return;
    }
    const tables = this._getReferencedTables(expr, leftAlias, rightAlias);
    if (tables.size === 1) {
      if (tables.has(leftAlias)) leftFilters.push(expr);
      else if (tables.has(rightAlias)) rightFilters.push(expr);
    } else if (tables.size > 1) {
      joinFilters.push(expr);
    }
  }
  _getReferencedTables(expr, leftAlias, rightAlias) {
    const tables = /* @__PURE__ */ new Set();
    const walk = (e) => {
      if (!e) return;
      if (e.type === "column") {
        const table = e.table;
        if (!table) {
          tables.add(leftAlias);
          tables.add(rightAlias);
        } else if (table === leftAlias) {
          tables.add(leftAlias);
        } else if (table === rightAlias) {
          tables.add(rightAlias);
        }
      } else if (e.type === "binary") {
        walk(e.left);
        walk(e.right);
      } else if (e.type === "unary") {
        walk(e.operand);
      } else if (e.type === "call") {
        for (const arg of e.args || []) walk(arg);
      } else if (e.type === "in") {
        walk(e.expr);
        for (const v of e.values || []) walk(v);
      } else if (e.type === "between") {
        walk(e.expr);
        walk(e.low);
        walk(e.high);
      } else if (e.type === "like") {
        walk(e.expr);
        walk(e.pattern);
      }
    };
    walk(expr);
    return tables;
  }
  _estimateFetchSize(ast, filterAnalysis) {
    const requestedLimit = ast.limit || 1e3;
    const leftSelectivity = filterAnalysis.left.length > 0 ? 0.5 : 1;
    const joinSelectivity = 0.7;
    const safetyFactor = 2.5;
    const leftFetch = Math.ceil(
      requestedLimit / (leftSelectivity * joinSelectivity) * safetyFactor
    );
    return { left: Math.min(leftFetch, 1e4), right: null };
  }
};

// src/client/sql/executor-search.js
function extractNearCondition(expr) {
  if (!expr) return null;
  if (expr.type === "near") {
    const columnName = expr.column?.name || expr.column;
    const value = expr.value?.value ?? expr.value;
    return { column: columnName, value, limit: 20 };
  }
  if (expr.type === "binary" && (expr.op === "AND" || expr.op === "OR")) {
    const leftNear = extractNearCondition(expr.left);
    if (leftNear) return leftNear;
    return extractNearCondition(expr.right);
  }
  return null;
}
function removeNearCondition(expr) {
  if (!expr) return null;
  if (expr.type === "near") {
    return null;
  }
  if (expr.type === "binary" && (expr.op === "AND" || expr.op === "OR")) {
    const leftIsNear = expr.left?.type === "near";
    const rightIsNear = expr.right?.type === "near";
    if (leftIsNear && rightIsNear) return null;
    if (leftIsNear) return removeNearCondition(expr.right);
    if (rightIsNear) return removeNearCondition(expr.left);
    const newLeft = removeNearCondition(expr.left);
    const newRight = removeNearCondition(expr.right);
    if (!newLeft && !newRight) return null;
    if (!newLeft) return newRight;
    if (!newRight) return newLeft;
    return { ...expr, left: newLeft, right: newRight };
  }
  return expr;
}
function tokenize(text) {
  if (!text || typeof text !== "string") return [];
  return text.toLowerCase().replace(/[^\w\s]/g, " ").split(/\s+/).filter((t) => t.length > 1).filter((t) => !isStopWord(t));
}
function isStopWord(word) {
  const stopWords = /* @__PURE__ */ new Set([
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "what",
    "which",
    "who",
    "whom",
    "where",
    "when",
    "why",
    "how"
  ]);
  return stopWords.has(word);
}
function computeBM25Scores(queryTokens, index) {
  const k1 = 1.2;
  const b = 0.75;
  const scores = /* @__PURE__ */ new Map();
  for (const term of queryTokens) {
    const docIds = index.termDocs.get(term);
    if (!docIds) continue;
    const n = docIds.size;
    const N = index.totalDocs;
    const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1);
    for (const docId of docIds) {
      const tf = index.termFreqs.get(term).get(docId);
      const docLen = index.docLengths.get(docId);
      const avgDL = index.avgDocLength;
      const numerator = tf * (k1 + 1);
      const denominator = tf + k1 * (1 - b + b * docLen / avgDL);
      const termScore = idf * (numerator / denominator);
      scores.set(docId, (scores.get(docId) || 0) + termScore);
    }
  }
  return scores;
}
function topKByScore(scores, k) {
  return Array.from(scores.entries()).sort((a, b) => b[1] - a[1]).slice(0, k).map(([docId]) => docId);
}
async function buildFTSIndex(readColumnData, colIdx, totalRows) {
  const index = {
    termDocs: /* @__PURE__ */ new Map(),
    // term -> Set of docIds
    termFreqs: /* @__PURE__ */ new Map(),
    // term -> Map(docId -> freq)
    docLengths: /* @__PURE__ */ new Map(),
    // docId -> word count
    totalDocs: 0,
    avgDocLength: 0
  };
  const batchSize = 1e3;
  let totalLength = 0;
  for (let start = 0; start < totalRows; start += batchSize) {
    const end = Math.min(start + batchSize, totalRows);
    const indices = Array.from({ length: end - start }, (_, i) => start + i);
    const texts = await readColumnData(colIdx, indices);
    for (let i = 0; i < texts.length; i++) {
      const docId = start + i;
      const text = texts[i];
      if (!text || typeof text !== "string") continue;
      const tokens = tokenize(text);
      index.docLengths.set(docId, tokens.length);
      totalLength += tokens.length;
      index.totalDocs++;
      const termCounts = /* @__PURE__ */ new Map();
      for (const token of tokens) {
        termCounts.set(token, (termCounts.get(token) || 0) + 1);
      }
      for (const [term, freq] of termCounts) {
        if (!index.termDocs.has(term)) {
          index.termDocs.set(term, /* @__PURE__ */ new Set());
          index.termFreqs.set(term, /* @__PURE__ */ new Map());
        }
        index.termDocs.get(term).add(docId);
        index.termFreqs.get(term).set(docId, freq);
      }
    }
  }
  index.avgDocLength = index.totalDocs > 0 ? totalLength / index.totalDocs : 0;
  return index;
}
async function executeBM25Search(executor, nearInfo, totalRows) {
  const { column, value, limit } = nearInfo;
  const colIdx = executor.columnMap[column.toLowerCase()];
  if (colIdx === void 0) {
    throw new Error(`Column '${column}' not found for text search`);
  }
  const queryTokens = tokenize(value);
  if (queryTokens.length === 0) return [];
  const cacheKey = `fts_${colIdx}_${totalRows}`;
  if (!executor.file._ftsIndexCache) executor.file._ftsIndexCache = /* @__PURE__ */ new Map();
  let index = executor.file._ftsIndexCache.get(cacheKey);
  if (!index) {
    index = await buildFTSIndex(
      (idx, indices) => executor.readColumnData(idx, indices),
      colIdx,
      totalRows
    );
    executor.file._ftsIndexCache.set(cacheKey, index);
  }
  const scores = computeBM25Scores(queryTokens, index);
  return topKByScore(scores, limit);
}
async function executeNearSearch(executor, nearInfo, totalRows) {
  const { column, value, limit } = nearInfo;
  const vectorColName = executor.file.columnNames?.find(
    (n) => n === "embedding" || n === `${column}_embedding` || n.endsWith("_embedding") || n.endsWith("_vector")
  );
  if (!vectorColName) {
    return await executeBM25Search(executor, nearInfo, totalRows);
  }
  const vectorColIdx = executor.columnMap[vectorColName.toLowerCase()];
  if (vectorColIdx === void 0) {
    throw new Error(`Vector column '${vectorColName}' found but index missing`);
  }
  const topK = Math.min(limit, totalRows);
  try {
    const results = await executor.file.vectorSearch(vectorColIdx, value, topK);
    return results.map((r) => r.index);
  } catch (e) {
    console.error("[SQLExecutor] Vector search failed:", e);
    throw new Error(`NEAR search failed: ${e.message}`);
  }
}
async function evaluateWithNear(executor, nearInfo, whereExpr, totalRows, onProgress) {
  if (onProgress) {
    onProgress("Executing vector search...", 0, 100);
  }
  const searchResults = await executeNearSearch(executor, nearInfo, totalRows);
  if (!searchResults || searchResults.length === 0) {
    return [];
  }
  const remainingExpr = removeNearCondition(whereExpr);
  if (!remainingExpr) {
    return searchResults;
  }
  if (onProgress) {
    onProgress("Applying filters...", 50, 100);
  }
  const neededCols = /* @__PURE__ */ new Set();
  executor.collectColumnsFromExpr(remainingExpr, neededCols);
  const columnData = {};
  for (const colName of neededCols) {
    const colIdx = executor.columnMap[colName.toLowerCase()];
    if (colIdx !== void 0) {
      columnData[colName.toLowerCase()] = await executor.readColumnData(colIdx, searchResults);
    }
  }
  const matchingIndices = [];
  for (let i = 0; i < searchResults.length; i++) {
    const result = executor.evaluateExpr(remainingExpr, columnData, i);
    if (result) {
      matchingIndices.push(searchResults[i]);
    }
  }
  return matchingIndices;
}
async function filterIndicesByWhere(executor, indices, whereExpr) {
  const neededCols = /* @__PURE__ */ new Set();
  executor.collectColumnsFromExpr(whereExpr, neededCols);
  const columnData = {};
  for (const colName of neededCols) {
    const colIdx = executor.columnMap[colName.toLowerCase()];
    if (colIdx !== void 0) {
      columnData[colName.toLowerCase()] = await executor.readColumnData(colIdx, indices);
    }
  }
  return indices.filter((_, i) => executor.evaluateExpr(whereExpr, columnData, i));
}
async function executeAggregateWithSearch(executor, ast, totalRows, onProgress) {
  const nearInfo = extractNearCondition(ast.where);
  if (!nearInfo) {
    return await executor.executeAggregateQuery(ast, totalRows, onProgress);
  }
  const searchIndices = await executeNearSearch(executor, nearInfo, totalRows);
  if (searchIndices.length === 0) {
    return executor._emptyAggregateResult(ast);
  }
  const remainingWhere = removeNearCondition(ast.where);
  let filteredIndices = searchIndices;
  if (remainingWhere) {
    filteredIndices = await filterIndicesByWhere(executor, searchIndices, remainingWhere);
  }
  if (filteredIndices.length === 0) {
    return executor._emptyAggregateResult(ast);
  }
  if (ast.groupBy && ast.groupBy.length > 0) {
    return await executor._executeGroupByOnIndices(ast, filteredIndices, onProgress);
  }
  return await executor._executeSimpleAggregateOnIndices(ast, filteredIndices, onProgress);
}

// src/client/sql/executor-window.js
function hasWindowFunctions(ast) {
  return ast.columns?.some((col) => col.expr?.type === "call" && col.expr.over);
}
function partitionRows(rows, partitionBy, columnData, evaluateExpr2) {
  if (!partitionBy || partitionBy.length === 0) {
    return [rows.map((r, i) => ({ idx: i, row: r }))];
  }
  const groups = /* @__PURE__ */ new Map();
  for (let i = 0; i < rows.length; i++) {
    const key = partitionBy.map((expr) => JSON.stringify(evaluateExpr2(expr, columnData, i))).join("|");
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push({ idx: i, row: rows[i] });
  }
  return Array.from(groups.values());
}
function compareRowsByOrder(a, b, orderBy, columnData, evaluateExpr2) {
  for (const ob of orderBy) {
    const valA = evaluateExpr2({ type: "column", column: ob.column }, columnData, a.idx);
    const valB = evaluateExpr2({ type: "column", column: ob.column }, columnData, b.idx);
    const dir = ob.direction === "DESC" ? -1 : 1;
    if (valA == null && valB == null) continue;
    if (valA == null) return 1 * dir;
    if (valB == null) return -1 * dir;
    if (valA < valB) return -1 * dir;
    if (valA > valB) return 1 * dir;
  }
  return 0;
}
function getFrameBounds(frame, partition, currentIdx) {
  const n = partition.length;
  let startIdx = 0;
  let endIdx = currentIdx;
  const start = frame.start || { type: "UNBOUNDED PRECEDING" };
  const startType = start.type.replace(" ", "_").toUpperCase();
  const startOffset = Number(start.offset ?? start.value ?? 1) || 1;
  switch (startType) {
    case "UNBOUNDED_PRECEDING":
      startIdx = 0;
      break;
    case "CURRENT_ROW":
      startIdx = currentIdx;
      break;
    case "PRECEDING":
      startIdx = Math.max(0, currentIdx - startOffset);
      break;
    case "FOLLOWING":
      startIdx = Math.min(n - 1, currentIdx + startOffset);
      break;
  }
  const end = frame.end || { type: "CURRENT ROW" };
  const endType = end.type.replace(" ", "_").toUpperCase();
  const endOffset = Number(end.offset ?? end.value ?? 1) || 1;
  switch (endType) {
    case "UNBOUNDED_FOLLOWING":
      endIdx = n - 1;
      break;
    case "CURRENT_ROW":
      endIdx = currentIdx;
      break;
    case "PRECEDING":
      endIdx = Math.max(0, currentIdx - endOffset);
      break;
    case "FOLLOWING":
      endIdx = Math.min(n - 1, currentIdx + endOffset);
      break;
  }
  if (startIdx > endIdx) [startIdx, endIdx] = [endIdx, startIdx];
  return [startIdx, endIdx];
}
function computeWindowFunction(funcName, args, over, rows, columnData, evaluateExpr2) {
  const results = new Array(rows.length).fill(null);
  const partitions = partitionRows(rows, over.partitionBy, columnData, evaluateExpr2);
  for (const partition of partitions) {
    if (over.orderBy && over.orderBy.length > 0) {
      partition.sort((a, b) => compareRowsByOrder(a, b, over.orderBy, columnData, evaluateExpr2));
    }
    for (let i = 0; i < partition.length; i++) {
      const rowIdx = partition[i].idx;
      switch (funcName) {
        case "ROW_NUMBER":
          results[rowIdx] = i + 1;
          break;
        case "RANK": {
          if (i > 0 && compareRowsByOrder(partition[i - 1], partition[i], over.orderBy, columnData, evaluateExpr2) === 0) {
            results[rowIdx] = results[partition[i - 1].idx];
          } else {
            results[rowIdx] = i + 1;
          }
          break;
        }
        case "DENSE_RANK": {
          if (i === 0) {
            results[rowIdx] = 1;
          } else if (compareRowsByOrder(partition[i - 1], partition[i], over.orderBy, columnData, evaluateExpr2) === 0) {
            results[rowIdx] = results[partition[i - 1].idx];
          } else {
            results[rowIdx] = results[partition[i - 1].idx] + 1;
          }
          break;
        }
        case "NTILE": {
          const requestedN = Math.max(1, Number(args[0]?.value) || 1);
          const n = Math.min(requestedN, partition.length);
          results[rowIdx] = Math.floor(i * n / partition.length) + 1;
          break;
        }
        case "PERCENT_RANK": {
          let rank = i + 1;
          for (let j = 0; j < i; j++) {
            if (compareRowsByOrder(partition[j], partition[i], over.orderBy, columnData, evaluateExpr2) === 0) {
              rank = j + 1;
              break;
            }
          }
          const partitionSize = partition.length;
          results[rowIdx] = partitionSize > 1 ? (rank - 1) / (partitionSize - 1) : 0;
          break;
        }
        case "CUME_DIST": {
          let countLessOrEqual = 0;
          for (let j = 0; j < partition.length; j++) {
            const cmp = compareRowsByOrder(partition[j], partition[i], over.orderBy, columnData, evaluateExpr2);
            if (cmp <= 0) countLessOrEqual++;
          }
          results[rowIdx] = countLessOrEqual / partition.length;
          break;
        }
        case "LAG": {
          const lagCol = args[0];
          const lagN = args[1]?.value || 1;
          const defaultVal = args[2]?.value ?? null;
          if (i >= lagN) {
            const prevRowIdx = partition[i - lagN].idx;
            results[rowIdx] = evaluateExpr2(lagCol, columnData, prevRowIdx);
          } else {
            results[rowIdx] = defaultVal;
          }
          break;
        }
        case "LEAD": {
          const leadCol = args[0];
          const leadN = args[1]?.value || 1;
          const defaultVal = args[2]?.value ?? null;
          if (i + leadN < partition.length) {
            const nextRowIdx = partition[i + leadN].idx;
            results[rowIdx] = evaluateExpr2(leadCol, columnData, nextRowIdx);
          } else {
            results[rowIdx] = defaultVal;
          }
          break;
        }
        case "FIRST_VALUE": {
          const firstRowIdx = partition[0].idx;
          results[rowIdx] = evaluateExpr2(args[0], columnData, firstRowIdx);
          break;
        }
        case "LAST_VALUE": {
          const frame = over.frame || {
            type: "RANGE",
            start: { type: "UNBOUNDED_PRECEDING" },
            end: { type: "CURRENT_ROW" }
          };
          const [, endIdx] = getFrameBounds(frame, partition, i);
          const lastRowIdx = partition[endIdx].idx;
          results[rowIdx] = evaluateExpr2(args[0], columnData, lastRowIdx);
          break;
        }
        case "NTH_VALUE": {
          const n = Number(args[1]?.value) || 1;
          const frame = over.frame || { type: "RANGE", start: { type: "UNBOUNDED_PRECEDING" }, end: { type: "CURRENT_ROW" } };
          const [startIdx, endIdx] = getFrameBounds(frame, partition, i);
          const frameSize = endIdx - startIdx + 1;
          if (n > 0 && n <= frameSize) {
            const nthRowIdx = partition[startIdx + n - 1].idx;
            results[rowIdx] = evaluateExpr2(args[0], columnData, nthRowIdx);
          } else {
            results[rowIdx] = null;
          }
          break;
        }
        // Aggregate window functions (frame-aware)
        case "SUM":
        case "AVG":
        case "COUNT":
        case "MIN":
        case "MAX": {
          const frame = over.frame || {
            type: "RANGE",
            start: { type: "UNBOUNDED_PRECEDING" },
            end: { type: "CURRENT_ROW" }
          };
          const [startIdx, endIdx] = getFrameBounds(frame, partition, i);
          const isStar = args[0]?.type === "star";
          let values = [];
          let frameRowCount = 0;
          for (let j = startIdx; j <= endIdx; j++) {
            frameRowCount++;
            if (!isStar) {
              const val = evaluateExpr2(args[0], columnData, partition[j].idx);
              const numVal = Number(val);
              if (val != null && !isNaN(numVal)) values.push(numVal);
            }
          }
          let result = null;
          switch (funcName) {
            case "SUM":
              result = values.reduce((a, b) => a + b, 0);
              break;
            case "AVG":
              result = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
              break;
            case "COUNT":
              result = isStar ? frameRowCount : values.length;
              break;
            case "MIN":
              result = values.length > 0 ? Math.min(...values) : null;
              break;
            case "MAX":
              result = values.length > 0 ? Math.max(...values) : null;
              break;
          }
          results[rowIdx] = result;
          break;
        }
        default:
          results[rowIdx] = null;
      }
    }
  }
  return results;
}
function computeWindowFunctions(ast, rows, columnData, evaluateExpr2) {
  const windowColumns = [];
  for (let colIndex = 0; colIndex < ast.columns.length; colIndex++) {
    const col = ast.columns[colIndex];
    if (col.expr?.type === "call" && col.expr.over) {
      const values = computeWindowFunction(
        col.expr.name,
        col.expr.args,
        col.expr.over,
        rows,
        columnData,
        evaluateExpr2
      );
      windowColumns.push({
        colIndex,
        alias: col.alias || col.expr.name,
        values
      });
    }
  }
  return windowColumns;
}
function executeWindowFunctions(executor, ast, data, columnData, filteredIndices) {
  const filteredRows = filteredIndices.map((idx) => data.rows[idx]);
  const windowResults = computeWindowFunctions(
    ast,
    filteredRows,
    columnData,
    (expr, colData, rowIdx) => executor._evaluateInMemoryExpr(expr, colData, rowIdx)
  );
  const resultColumns = [];
  for (const col of ast.columns) {
    if (col.alias) {
      resultColumns.push(col.alias);
    } else if (col.expr?.type === "call") {
      const argName = col.expr.args?.[0]?.name || col.expr.args?.[0]?.column || "*";
      resultColumns.push(`${col.expr.name}(${argName})`);
    } else if (col.expr?.type === "column") {
      resultColumns.push(col.expr.name || col.expr.column);
    } else {
      resultColumns.push("?");
    }
  }
  const resultRows = [];
  for (let i = 0; i < filteredIndices.length; i++) {
    const origIdx = filteredIndices[i];
    const row = [];
    for (let c = 0; c < ast.columns.length; c++) {
      const col = ast.columns[c];
      const expr = col.expr;
      if (expr?.over) {
        const windowCol = windowResults.find((w) => w.colIndex === c);
        row.push(windowCol ? windowCol.values[i] : null);
      } else if (expr?.type === "column") {
        const colName = (expr.name || expr.column || "").toLowerCase();
        row.push(columnData[colName]?.[origIdx] ?? null);
      } else {
        row.push(executor._evaluateInMemoryExpr(expr, columnData, origIdx));
      }
    }
    resultRows.push(row);
  }
  let finalRows = resultRows;
  if (ast.qualify) {
    finalRows = [];
    const qualifyColMap = {};
    resultColumns.forEach((name, idx) => {
      qualifyColMap[name.toLowerCase()] = idx;
    });
    for (let i = 0; i < resultRows.length; i++) {
      const rowData = {};
      for (let c = 0; c < resultColumns.length; c++) {
        rowData[resultColumns[c].toLowerCase()] = resultRows[i][c];
      }
      if (executor._evaluateInMemoryExpr(ast.qualify, rowData, 0)) {
        finalRows.push(resultRows[i]);
      }
    }
  }
  if (ast.orderBy && ast.orderBy.length > 0) {
    const colIdxMap = {};
    resultColumns.forEach((name, idx) => {
      colIdxMap[name.toLowerCase()] = idx;
    });
    finalRows.sort((a, b) => {
      for (const ob of ast.orderBy) {
        const colIdx = colIdxMap[ob.column.toLowerCase()];
        if (colIdx === void 0) {
          console.warn(`[SQLExecutor] ORDER BY column '${ob.column}' not found in result columns`);
          continue;
        }
        const valA = a[colIdx], valB = b[colIdx];
        const dir = ob.descending || ob.direction === "DESC" ? -1 : 1;
        if (valA == null && valB == null) continue;
        if (valA == null) return 1 * dir;
        if (valB == null) return -1 * dir;
        if (valA < valB) return -1 * dir;
        if (valA > valB) return 1 * dir;
      }
      return 0;
    });
  }
  const offset = ast.offset || 0;
  let rows = finalRows;
  if (offset > 0) rows = rows.slice(offset);
  if (ast.limit) rows = rows.slice(0, ast.limit);
  return { columns: resultColumns, rows, total: finalRows.length };
}

// src/client/sql/executor-filters.js
function detectSimpleFilter(expr, columnMap, columnTypes) {
  if (expr.type !== "binary") return null;
  if (!["==", "!=", "<", "<=", ">", ">="].includes(expr.op)) return null;
  let column = null;
  let value = null;
  let op = expr.op;
  if (expr.left.type === "column" && expr.right.type === "literal") {
    column = expr.left.name;
    value = expr.right.value;
  } else if (expr.left.type === "literal" && expr.right.type === "column") {
    column = expr.right.name;
    value = expr.left.value;
    if (op === "<") op = ">";
    else if (op === ">") op = "<";
    else if (op === "<=") op = ">=";
    else if (op === ">=") op = "<=";
  }
  if (!column || value === null) return null;
  const colIdx = columnMap[column.toLowerCase()];
  if (colIdx === void 0) return null;
  const colType = columnTypes[colIdx];
  if (!["int64", "int32", "float64", "float32"].includes(colType)) return null;
  return { column, colIdx, op, value, colType };
}
async function evaluateSimpleFilter(executor, filter, totalRows, onProgress) {
  const matchingIndices = [];
  const batchSize = 5e3;
  for (let batchStart = 0; batchStart < totalRows; batchStart += batchSize) {
    if (onProgress) {
      const pct = Math.round(batchStart / totalRows * 100);
      onProgress(`Filtering ${filter.column}... ${pct}%`, batchStart, totalRows);
    }
    const batchEnd = Math.min(batchStart + batchSize, totalRows);
    const batchIndices2 = Array.from(
      { length: batchEnd - batchStart },
      (_, i) => batchStart + i
    );
    const colData = await executor.readColumnData(filter.colIdx, batchIndices2);
    for (let i = 0; i < batchIndices2.length; i++) {
      const val = colData[i];
      let matches = false;
      switch (filter.op) {
        case "==":
          matches = val === filter.value;
          break;
        case "!=":
          matches = val !== filter.value;
          break;
        case "<":
          matches = val < filter.value;
          break;
        case "<=":
          matches = val <= filter.value;
          break;
        case ">":
          matches = val > filter.value;
          break;
        case ">=":
          matches = val >= filter.value;
          break;
      }
      if (matches) {
        matchingIndices.push(batchIndices2[i]);
      }
    }
    if (matchingIndices.length >= 1e4) {
      break;
    }
  }
  return matchingIndices;
}
async function evaluateComplexFilter(executor, whereExpr, totalRows, onProgress) {
  const matchingIndices = [];
  const batchSize = 1e3;
  const neededCols = /* @__PURE__ */ new Set();
  executor.collectColumnsFromExpr(whereExpr, neededCols);
  for (let batchStart = 0; batchStart < totalRows; batchStart += batchSize) {
    if (onProgress) {
      onProgress(`Filtering rows...`, batchStart, totalRows);
    }
    const batchEnd = Math.min(batchStart + batchSize, totalRows);
    const batchIndices2 = Array.from(
      { length: batchEnd - batchStart },
      (_, i) => batchStart + i
    );
    const batchData = {};
    for (const colName of neededCols) {
      const colIdx = executor.columnMap[colName.toLowerCase()];
      if (colIdx !== void 0) {
        batchData[colName.toLowerCase()] = await executor.readColumnData(colIdx, batchIndices2);
      }
    }
    for (let i = 0; i < batchIndices2.length; i++) {
      const result = evaluateExpr(executor, whereExpr, batchData, i);
      if (result) {
        matchingIndices.push(batchIndices2[i]);
      }
    }
    if (matchingIndices.length >= 1e4) {
      break;
    }
  }
  return matchingIndices;
}
function evaluateExpr(executor, expr, columnData, rowIdx) {
  if (!expr) return null;
  switch (expr.type) {
    case "literal":
      return expr.value;
    case "column": {
      const data = columnData[expr.name.toLowerCase()];
      return data ? data[rowIdx] : null;
    }
    case "star":
      return "*";
    case "binary": {
      const left = evaluateExpr(executor, expr.left, columnData, rowIdx);
      const right = evaluateExpr(executor, expr.right, columnData, rowIdx);
      switch (expr.op) {
        case "+":
          return (left || 0) + (right || 0);
        case "-":
          return (left || 0) - (right || 0);
        case "*":
          return (left || 0) * (right || 0);
        case "/":
          return right !== 0 ? (left || 0) / right : null;
        case "==":
          return left === right;
        case "!=":
          return left !== right;
        case "<":
          return left < right;
        case "<=":
          return left <= right;
        case ">":
          return left > right;
        case ">=":
          return left >= right;
        case "AND":
          return left && right;
        case "OR":
          return left || right;
        default:
          return null;
      }
    }
    case "unary": {
      const operand = evaluateExpr(executor, expr.operand, columnData, rowIdx);
      switch (expr.op) {
        case "-":
          return -operand;
        case "NOT":
          return !operand;
        default:
          return null;
      }
    }
    case "in": {
      const value = evaluateExpr(executor, expr.expr, columnData, rowIdx);
      const values = expr.values.map((v) => evaluateExpr(executor, v, columnData, rowIdx));
      return values.includes(value);
    }
    case "between": {
      const value = evaluateExpr(executor, expr.expr, columnData, rowIdx);
      const low = evaluateExpr(executor, expr.low, columnData, rowIdx);
      const high = evaluateExpr(executor, expr.high, columnData, rowIdx);
      if (value == null || low == null || high == null) return null;
      return value >= low && value <= high;
    }
    case "like": {
      const value = evaluateExpr(executor, expr.expr, columnData, rowIdx);
      const pattern = evaluateExpr(executor, expr.pattern, columnData, rowIdx);
      if (typeof value !== "string" || typeof pattern !== "string") return false;
      const regex = new RegExp("^" + pattern.replace(/%/g, ".*").replace(/_/g, ".") + "$", "i");
      return regex.test(value);
    }
    case "near":
      return true;
    case "call":
      return null;
    case "subquery": {
      return executor._executeSubquery(expr.query, columnData, rowIdx);
    }
    case "array": {
      return expr.elements.map((el) => evaluateExpr(executor, el, columnData, rowIdx));
    }
    case "subscript": {
      const arr = evaluateExpr(executor, expr.array, columnData, rowIdx);
      const idx = evaluateExpr(executor, expr.index, columnData, rowIdx);
      if (!Array.isArray(arr)) return null;
      return arr[idx - 1] ?? null;
    }
    default:
      return null;
  }
}
function evaluateInMemoryExpr(expr, columnData, rowIdx) {
  if (!expr) return null;
  switch (expr.type) {
    case "literal":
      return expr.value;
    case "column": {
      const colName = expr.column.toLowerCase();
      const col = columnData[colName];
      return col ? Array.isArray(col) ? col[rowIdx] : col : null;
    }
    case "binary": {
      const left = evaluateInMemoryExpr(expr.left, columnData, rowIdx);
      const right = evaluateInMemoryExpr(expr.right, columnData, rowIdx);
      const op = expr.op || expr.operator;
      switch (op) {
        case "=":
        case "==":
          return left == right;
        case "!=":
        case "<>":
          return left != right;
        case "<":
          return left < right;
        case "<=":
          return left <= right;
        case ">":
          return left > right;
        case ">=":
          return left >= right;
        case "+":
          return Number(left) + Number(right);
        case "-":
          return Number(left) - Number(right);
        case "*":
          return Number(left) * Number(right);
        case "/":
          return right !== 0 ? Number(left) / Number(right) : null;
        case "AND":
          return left && right;
        case "OR":
          return left || right;
        default:
          return null;
      }
    }
    case "unary": {
      const operand = evaluateInMemoryExpr(expr.operand, columnData, rowIdx);
      const op = expr.op || expr.operator;
      switch (op) {
        case "NOT":
          return !operand;
        case "-":
          return -operand;
        case "IS NULL":
          return operand == null;
        case "IS NOT NULL":
          return operand != null;
        default:
          return null;
      }
    }
    case "call": {
      const funcName = expr.name.toUpperCase();
      const args = expr.args?.map((a) => evaluateInMemoryExpr(a, columnData, rowIdx)) || [];
      switch (funcName) {
        case "UPPER":
          return String(args[0]).toUpperCase();
        case "LOWER":
          return String(args[0]).toLowerCase();
        case "LENGTH":
          return String(args[0]).length;
        case "SUBSTR":
        case "SUBSTRING": {
          if (args[0] == null || args[1] == null) return null;
          const start = Number(args[1]);
          if (isNaN(start)) return null;
          const len = args[2] != null ? Number(args[2]) : void 0;
          if (len !== void 0 && (isNaN(len) || len < 0)) return null;
          return String(args[0]).substring(start - 1, len !== void 0 ? start - 1 + len : void 0);
        }
        case "COALESCE":
          return args.find((a) => a != null) ?? null;
        case "ABS":
          return Math.abs(args[0]);
        case "ROUND":
          return Math.round(args[0] * Math.pow(10, args[1] || 0)) / Math.pow(10, args[1] || 0);
        case "GROUPING": {
          const colArg = expr.args?.[0];
          const colName = colArg?.column || colArg?.name;
          if (!colName) return 0;
          const groupingSet = columnData._groupingSet || [];
          return groupingSet.includes(colName.toLowerCase()) ? 0 : 1;
        }
        default:
          return null;
      }
    }
    case "in": {
      const val = evaluateInMemoryExpr(expr.expr, columnData, rowIdx);
      const values = expr.values.map((v) => v.value ?? evaluateInMemoryExpr(v, columnData, rowIdx));
      return values.includes(val);
    }
    case "between": {
      const val = evaluateInMemoryExpr(expr.expr, columnData, rowIdx);
      const low = evaluateInMemoryExpr(expr.low, columnData, rowIdx);
      const high = evaluateInMemoryExpr(expr.high, columnData, rowIdx);
      if (val == null || low == null || high == null) return null;
      return val >= low && val <= high;
    }
    case "like": {
      const val = String(evaluateInMemoryExpr(expr.expr, columnData, rowIdx));
      const pattern = evaluateInMemoryExpr(expr.pattern, columnData, rowIdx);
      const regex = new RegExp("^" + String(pattern).replace(/%/g, ".*").replace(/_/g, ".") + "$", "i");
      return regex.test(val);
    }
    case "array": {
      return expr.elements.map((el) => evaluateInMemoryExpr(el, columnData, rowIdx));
    }
    case "subscript": {
      const arr = evaluateInMemoryExpr(expr.array, columnData, rowIdx);
      const idx = evaluateInMemoryExpr(expr.index, columnData, rowIdx);
      if (!Array.isArray(arr)) return null;
      return arr[idx - 1] ?? null;
    }
    default:
      return null;
  }
}
function collectColumnsFromExpr2(expr, columns) {
  if (!expr) return;
  switch (expr.type) {
    case "column":
      columns.add(expr.name.toLowerCase());
      break;
    case "binary":
      collectColumnsFromExpr2(expr.left, columns);
      collectColumnsFromExpr2(expr.right, columns);
      break;
    case "unary":
      collectColumnsFromExpr2(expr.operand, columns);
      break;
    case "call":
      for (const arg of expr.args || []) {
        collectColumnsFromExpr2(arg, columns);
      }
      break;
    case "in":
      collectColumnsFromExpr2(expr.expr, columns);
      break;
    case "between":
      collectColumnsFromExpr2(expr.expr, columns);
      break;
    case "like":
      collectColumnsFromExpr2(expr.expr, columns);
      break;
    case "near":
      collectColumnsFromExpr2(expr.column, columns);
      break;
  }
}

// src/client/sql/executor-utils.js
function hashRow(row) {
  let hash = 0;
  for (const val of row) {
    if (val === null || val === void 0) {
      hash = hash * 31 + 0 >>> 0;
    } else if (typeof val === "number") {
      hash = hash * 31 + val >>> 0;
    } else if (val instanceof Uint8Array) {
      for (const b of val) {
        hash = hash * 31 + b >>> 0;
      }
    } else if (Array.isArray(val)) {
      for (let j = 0; j < val.length; j++) {
        hash = hash * 31 + (val[j] || 0) >>> 0;
      }
    } else {
      const str = String(val);
      for (let j = 0; j < str.length; j++) {
        hash = hash * 31 + str.charCodeAt(j) >>> 0;
      }
    }
  }
  return hash;
}
function hashRows(rows) {
  const hashes = new Uint32Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    hashes[i] = hashRow(rows[i]);
  }
  return hashes;
}
function computeAggregate(funcName, values) {
  const nums = values.filter((v) => v != null && typeof v === "number");
  switch (funcName.toUpperCase()) {
    case "SUM":
      return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) : null;
    case "COUNT":
      return values.filter((v) => v != null).length;
    case "AVG":
      return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
    case "MIN":
      return nums.length > 0 ? Math.min(...nums) : null;
    case "MAX":
      return nums.length > 0 ? Math.max(...nums) : null;
    default:
      return null;
  }
}
function executePivot(rows, columns, pivot) {
  const { aggregate, forColumn, inValues } = pivot;
  const colNames = columns.map((col) => {
    if (col.type === "star") return "*";
    return col.alias || (col.expr.type === "column" ? col.expr.name : null);
  });
  const forColIdx = colNames.findIndex(
    (n) => n && n.toLowerCase() === forColumn.toLowerCase()
  );
  if (forColIdx === -1) {
    throw new Error(`PIVOT: Column '${forColumn}' not found in result set`);
  }
  const aggSourceCol = aggregate.args && aggregate.args[0] ? aggregate.args[0].name || aggregate.args[0].column || aggregate.args[0] : null;
  const aggSourceIdx = aggSourceCol ? colNames.findIndex((n) => n && n.toLowerCase() === String(aggSourceCol).toLowerCase()) : -1;
  if (aggSourceIdx === -1 && aggregate.name.toUpperCase() !== "COUNT") {
    throw new Error(`PIVOT: Aggregate source column not found`);
  }
  const groupColIndices = [];
  for (let i = 0; i < colNames.length; i++) {
    if (i !== forColIdx && i !== aggSourceIdx) {
      groupColIndices.push(i);
    }
  }
  const groups = /* @__PURE__ */ new Map();
  for (const row of rows) {
    const groupKey = groupColIndices.map((i) => JSON.stringify(row[i])).join("|");
    const pivotValue = row[forColIdx];
    const aggValue = aggSourceIdx >= 0 ? row[aggSourceIdx] : 1;
    if (!groups.has(groupKey)) {
      groups.set(groupKey, {
        groupValues: groupColIndices.map((i) => row[i]),
        pivots: /* @__PURE__ */ new Map()
      });
    }
    const group = groups.get(groupKey);
    const pivotKey = String(pivotValue);
    if (!group.pivots.has(pivotKey)) {
      group.pivots.set(pivotKey, []);
    }
    group.pivots.get(pivotKey).push(aggValue);
  }
  const groupColNames = groupColIndices.map((i) => colNames[i]);
  const outputColNames = [...groupColNames, ...inValues.map((v) => String(v))];
  const outputRows = [];
  for (const [, group] of groups) {
    const row = [...group.groupValues];
    for (const pivotVal of inValues) {
      const key = String(pivotVal);
      const values = group.pivots.get(key) || [];
      row.push(computeAggregate(aggregate.name, values));
    }
    outputRows.push(row);
  }
  const newColumns = outputColNames.map((name) => ({
    type: "column",
    expr: { type: "column", name },
    alias: name
  }));
  return { rows: outputRows, columns: newColumns };
}
function executeUnpivot(rows, columns, unpivot) {
  const { valueColumn, nameColumn, inColumns } = unpivot;
  const colNames = columns.map((col) => {
    if (col.type === "star") return "*";
    return col.alias || (col.expr.type === "column" ? col.expr.name : null);
  });
  const inColIndices = inColumns.map((c) => {
    const idx = colNames.findIndex((n) => n && n.toLowerCase() === c.toLowerCase());
    if (idx === -1) {
      throw new Error(`UNPIVOT: Column '${c}' not found in result set`);
    }
    return idx;
  });
  const preservedIndices = [];
  const preservedNames = [];
  for (let i = 0; i < colNames.length; i++) {
    if (!inColIndices.includes(i)) {
      preservedIndices.push(i);
      preservedNames.push(colNames[i]);
    }
  }
  const outputColNames = [...preservedNames, nameColumn, valueColumn];
  const outputRows = [];
  for (const row of rows) {
    const preservedValues = preservedIndices.map((i) => row[i]);
    for (let i = 0; i < inColumns.length; i++) {
      const colName = inColumns[i];
      const value = row[inColIndices[i]];
      if (value != null) {
        outputRows.push([...preservedValues, colName, value]);
      }
    }
  }
  const newColumns = outputColNames.map((name) => ({
    type: "column",
    expr: { type: "column", name },
    alias: name
  }));
  return { rows: outputRows, columns: newColumns };
}
async function applyDistinct(executor, rows, gpuSorter2 = null) {
  if (rows.length === 0) return [];
  if (rows.length >= 1e4 && gpuSorter2?.isAvailable?.()) {
    const hashes = hashRows(rows);
    const numGroups = await gpuSorter2.groupBy(hashes);
    const firstOccurrence = new Int32Array(numGroups).fill(-1);
    for (let i = 0; i < rows.length; i++) {
      const gid = hashes[i] % numGroups;
      if (firstOccurrence[gid] === -1) {
        firstOccurrence[gid] = i;
      }
    }
    const uniqueRows2 = [];
    for (let gid = 0; gid < numGroups; gid++) {
      if (firstOccurrence[gid] !== -1) {
        uniqueRows2.push(rows[firstOccurrence[gid]]);
      }
    }
    return uniqueRows2;
  }
  const seen = /* @__PURE__ */ new Set();
  const uniqueRows = [];
  for (const row of rows) {
    const key = JSON.stringify(row);
    if (!seen.has(key)) {
      seen.add(key);
      uniqueRows.push(row);
    }
  }
  return uniqueRows;
}
async function applyOrderBy(executor, rows, orderBy, outputColumns, gpuSorter2 = null) {
  const colIdxMap = {};
  let idx = 0;
  for (const col of outputColumns) {
    if (col.type === "star") {
      for (const name of executor.file.columnNames || []) {
        colIdxMap[name.toLowerCase()] = idx++;
      }
    } else {
      const name = col.alias || executor.exprToName(col.expr);
      colIdxMap[name.toLowerCase()] = idx++;
    }
  }
  if (rows.length >= 1e4 && gpuSorter2?.isAvailable?.()) {
    let indices = new Uint32Array(rows.length);
    for (let i = 0; i < rows.length; i++) indices[i] = i;
    for (let c = orderBy.length - 1; c >= 0; c--) {
      const ob = orderBy[c];
      const colIdx = colIdxMap[ob.column.toLowerCase()];
      if (colIdx === void 0) continue;
      const ascending = !ob.descending;
      const values = new Float32Array(rows.length);
      for (let i = 0; i < rows.length; i++) {
        const val = rows[indices[i]][colIdx];
        if (val == null) {
          values[i] = ascending ? 34e37 : -34e37;
        } else if (typeof val === "number") {
          values[i] = val;
        } else if (typeof val === "string") {
          let key = 0;
          for (let j = 0; j < Math.min(4, val.length); j++) {
            key = key * 256 + val.charCodeAt(j);
          }
          values[i] = key;
        } else {
          values[i] = 0;
        }
      }
      const sortedIdx = await gpuSorter2.sort(values, ascending);
      const newIndices = new Uint32Array(rows.length);
      for (let i = 0; i < rows.length; i++) {
        newIndices[i] = indices[sortedIdx[i]];
      }
      indices = newIndices;
    }
    const sorted = [];
    for (let i = 0; i < rows.length; i++) {
      sorted.push(rows[indices[i]]);
    }
    rows.length = 0;
    rows.push(...sorted);
    return;
  }
  rows.sort((a, b) => {
    for (const ob of orderBy) {
      const colIdx = colIdxMap[ob.column.toLowerCase()];
      if (colIdx === void 0) continue;
      const valA = a[colIdx];
      const valB = b[colIdx];
      const dir = ob.descending ? -1 : 1;
      if (valA == null && valB == null) continue;
      if (valA == null) return 1 * dir;
      if (valB == null) return -1 * dir;
      if (valA < valB) return -1 * dir;
      if (valA > valB) return 1 * dir;
    }
    return 0;
  });
}
function evaluateWhereExprOnRow(expr, columns, row, getValueFromExpr2) {
  if (!expr) return true;
  if (expr.type === "binary") {
    if (expr.op === "AND") {
      return evaluateWhereExprOnRow(expr.left, columns, row, getValueFromExpr2) && evaluateWhereExprOnRow(expr.right, columns, row, getValueFromExpr2);
    }
    if (expr.op === "OR") {
      return evaluateWhereExprOnRow(expr.left, columns, row, getValueFromExpr2) || evaluateWhereExprOnRow(expr.right, columns, row, getValueFromExpr2);
    }
    const leftVal = getValueFromExpr2(expr.left, columns, row);
    const rightVal = getValueFromExpr2(expr.right, columns, row);
    switch (expr.op) {
      case "=":
      case "==":
        return leftVal == rightVal;
      case "!=":
      case "<>":
        return leftVal != rightVal;
      case "<":
        return leftVal < rightVal;
      case "<=":
        return leftVal <= rightVal;
      case ">":
        return leftVal > rightVal;
      case ">=":
        return leftVal >= rightVal;
      default:
        return true;
    }
  }
  return true;
}
function getValueFromExpr(expr, columns, row) {
  if (expr.type === "literal") {
    return expr.value;
  }
  if (expr.type === "column") {
    const colName = expr.name || expr.column;
    const idx = columns.indexOf(colName) !== -1 ? columns.indexOf(colName) : columns.indexOf(colName.toLowerCase());
    return idx !== -1 ? row[idx] : null;
  }
  return null;
}
function exprToName(expr) {
  if (!expr) return "?";
  switch (expr.type) {
    case "column":
      return expr.name || expr.column || "?";
    case "literal":
      return String(expr.value);
    case "call":
      const argNames = (expr.args || []).map((a) => exprToName(a)).join(", ");
      return `${expr.name}(${argNames})`;
    case "binary":
      return `${exprToName(expr.left)} ${expr.op} ${exprToName(expr.right)}`;
    case "star":
      return "*";
    default:
      return "?";
  }
}

// src/client/sql/executor-aggregates.js
var AGG_FUNCS = ["COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "STDDEV_SAMP", "STDDEV_POP", "VARIANCE", "VAR_SAMP", "VAR_POP", "MEDIAN", "STRING_AGG", "GROUP_CONCAT"];
function hasAdvancedGroupBy(groupBy) {
  if (!groupBy || groupBy.length === 0) return false;
  if (typeof groupBy[0] === "string") return false;
  return groupBy.some(
    (item) => item.type === "ROLLUP" || item.type === "CUBE" || item.type === "GROUPING_SETS"
  );
}
function getAllGroupColumns(groupBy) {
  const columns = [];
  for (const item of groupBy) {
    if (item.type === "COLUMN") {
      if (!columns.includes(item.column)) columns.push(item.column);
    } else if (item.type === "ROLLUP" || item.type === "CUBE") {
      for (const col of item.columns) {
        if (!columns.includes(col)) columns.push(col);
      }
    } else if (item.type === "GROUPING_SETS") {
      for (const set of item.sets) {
        for (const col of set) {
          if (!columns.includes(col)) columns.push(col);
        }
      }
    }
  }
  return columns;
}
var MAX_POWERSET_COLUMNS = 12;
function powerSet(arr) {
  if (arr.length > MAX_POWERSET_COLUMNS) {
    throw new Error(
      `CUBE/ROLLUP power set limited to ${MAX_POWERSET_COLUMNS} columns (got ${arr.length}). This would generate ${Math.pow(2, arr.length)} groupings.`
    );
  }
  const result = [[]];
  for (const item of arr) {
    const len = result.length;
    for (let i = 0; i < len; i++) {
      result.push([...result[i], item]);
    }
  }
  return result;
}
function crossProductSets(sets1, sets2) {
  const result = [];
  for (const s1 of sets1) {
    for (const s2 of sets2) {
      result.push([...s1, ...s2]);
    }
  }
  return result;
}
function expandGroupBy(groupBy) {
  if (!groupBy || groupBy.length === 0) return [[]];
  if (typeof groupBy[0] === "string") {
    return [groupBy];
  }
  let result = [[]];
  for (const item of groupBy) {
    if (item.type === "COLUMN") {
      result = result.map((set) => [...set, item.column]);
    } else if (item.type === "ROLLUP") {
      const rollupSets = [];
      for (let i = item.columns.length; i >= 0; i--) {
        rollupSets.push(item.columns.slice(0, i));
      }
      result = crossProductSets(result, rollupSets);
    } else if (item.type === "CUBE") {
      const cubeSets = powerSet(item.columns);
      result = crossProductSets(result, cubeSets);
    } else if (item.type === "GROUPING_SETS") {
      result = crossProductSets(result, item.sets);
    }
  }
  const seen = /* @__PURE__ */ new Set();
  return result.filter((set) => {
    const key = JSON.stringify(set.sort());
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
function computeAggregate2(funcName, values, options = {}) {
  const nums = values.filter((v) => v != null && typeof v === "number" && !isNaN(v));
  switch (funcName.toUpperCase()) {
    case "COUNT":
      return values.filter((v) => v != null).length;
    case "SUM":
      return nums.reduce((a, b) => a + b, 0);
    case "AVG":
      return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
    case "MIN":
      return nums.length > 0 ? Math.min(...nums) : null;
    case "MAX":
      return nums.length > 0 ? Math.max(...nums) : null;
    case "STDDEV":
    case "STDDEV_SAMP": {
      if (nums.length < 2) return null;
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      const variance = nums.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (nums.length - 1);
      return Math.sqrt(variance);
    }
    case "STDDEV_POP": {
      if (nums.length === 0) return null;
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      const variance = nums.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / nums.length;
      return Math.sqrt(variance);
    }
    case "VARIANCE":
    case "VAR_SAMP": {
      if (nums.length < 2) return null;
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      return nums.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (nums.length - 1);
    }
    case "VAR_POP": {
      if (nums.length === 0) return null;
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      return nums.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / nums.length;
    }
    case "MEDIAN": {
      if (nums.length === 0) return null;
      const sorted = [...nums].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }
    case "STRING_AGG":
    case "GROUP_CONCAT": {
      const separator = options.separator ?? ",";
      return values.filter((v) => v != null).map(String).join(separator);
    }
    default:
      return null;
  }
}
function hasAggregates(ast) {
  for (const col of ast.columns) {
    if (col.type === "expr" && col.expr.type === "call") {
      if (AGG_FUNCS.includes(col.expr.name.toUpperCase())) {
        return true;
      }
    }
  }
  return false;
}
function isSimpleCountStar(ast) {
  if (ast.columns.length !== 1) return false;
  const col = ast.columns[0];
  if (col.type === "expr" && col.expr.type === "call") {
    const name = col.expr.name.toUpperCase();
    if (name === "COUNT") {
      const arg = col.expr.args[0];
      return arg?.type === "star";
    }
  }
  return false;
}
function emptyAggregateResult(ast) {
  const colNames = ast.columns.map((col) => col.alias || exprToName(col.expr || col));
  const emptyRow = ast.columns.map((col) => {
    const expr = col.expr || col;
    if (expr.type === "call" && expr.name.toUpperCase() === "COUNT") {
      return 0;
    }
    return null;
  });
  if (ast.groupBy && ast.groupBy.length > 0) {
    return {
      columns: colNames,
      rows: [],
      total: 0,
      aggregationStats: {
        scannedRows: 0,
        totalRows: 0,
        coveragePercent: "100.00",
        isPartialScan: false,
        fromSearch: true
      }
    };
  }
  return {
    columns: colNames,
    rows: [emptyRow],
    total: 1,
    aggregationStats: {
      scannedRows: 0,
      totalRows: 0,
      coveragePercent: "100.00",
      isPartialScan: false,
      fromSearch: true
    }
  };
}
function executeGroupByForSet(ast, columnData, filteredIndices, groupingSet, allGroupColumns) {
  const groups = /* @__PURE__ */ new Map();
  const groupingSetLower = groupingSet.map((c) => c.toLowerCase());
  for (const idx of filteredIndices) {
    const groupKey = groupingSet.length > 0 ? groupingSet.map((col) => {
      const val = columnData[col.toLowerCase()]?.[idx];
      return JSON.stringify(val);
    }).join("|") : "__grand_total__";
    if (!groups.has(groupKey)) {
      const groupValues = {};
      for (const col of allGroupColumns) {
        if (groupingSetLower.includes(col.toLowerCase())) {
          groupValues[col] = columnData[col.toLowerCase()]?.[idx];
        } else {
          groupValues[col] = null;
        }
      }
      groups.set(groupKey, {
        values: groupValues,
        indices: [],
        _groupingSet: groupingSet
      });
    }
    groups.get(groupKey).indices.push(idx);
  }
  if (groupingSet.length === 0 && groups.size === 0) {
    const groupValues = {};
    for (const col of allGroupColumns) {
      groupValues[col] = null;
    }
    groups.set("__grand_total__", {
      values: groupValues,
      indices: [],
      _groupingSet: groupingSet
    });
  }
  const results = [];
  for (const [, group] of groups) {
    const row = { ...group.values, _groupingSet: group._groupingSet };
    for (const col of ast.columns) {
      const expr = col.expr;
      if (expr?.type === "call" && AGG_FUNCS.includes((expr.name || "").toUpperCase())) {
        const funcName = expr.name.toUpperCase();
        const argExpr = expr.args?.[0];
        const isStar = argExpr?.type === "star";
        const colName = (argExpr?.name || argExpr?.column || "").toLowerCase();
        const alias = col.alias || `${funcName}(${isStar ? "*" : colName})`;
        const indices = group.indices;
        let values = [];
        if (isStar) {
          values = indices.map(() => 1);
        } else {
          values = indices.map((i) => columnData[colName]?.[i]).filter((v) => v != null);
        }
        row[alias] = funcName === "COUNT" && isStar ? indices.length : computeAggregate2(funcName, values);
      }
    }
    results.push(row);
  }
  return results;
}
function buildAdvancedGroupByResult(ast, allResults, allGroupColumns) {
  const resultColumns = [];
  for (const col of ast.columns) {
    if (col.alias) {
      resultColumns.push(col.alias);
    } else if (col.expr?.type === "call") {
      const argName = col.expr.args?.[0]?.name || col.expr.args?.[0]?.column || "*";
      resultColumns.push(`${col.expr.name}(${argName})`);
    } else if (col.expr?.type === "column") {
      resultColumns.push(col.expr.name || col.expr.column);
    } else {
      resultColumns.push("?");
    }
  }
  const resultRows = allResults.map((rowObj) => {
    return resultColumns.map((colName) => rowObj[colName] ?? rowObj[colName.toLowerCase()] ?? null);
  });
  if (ast.orderBy && ast.orderBy.length > 0) {
    const colIdxMap = {};
    resultColumns.forEach((name, idx) => {
      colIdxMap[name.toLowerCase()] = idx;
    });
    resultRows.sort((a, b) => {
      for (const ob of ast.orderBy) {
        const colIdx = colIdxMap[ob.column.toLowerCase()];
        if (colIdx === void 0) continue;
        const valA = a[colIdx], valB = b[colIdx];
        const dir = ob.descending || ob.direction === "DESC" ? -1 : 1;
        if (valA == null && valB == null) continue;
        if (valA == null) return 1 * dir;
        if (valB == null) return -1 * dir;
        if (valA < valB) return -1 * dir;
        if (valA > valB) return 1 * dir;
      }
      return 0;
    });
  }
  const offset = ast.offset || 0;
  let rows = resultRows;
  if (offset > 0) rows = rows.slice(offset);
  if (ast.limit) rows = rows.slice(0, ast.limit);
  return { columns: resultColumns, rows, total: allResults.length };
}
function executeAdvancedGroupBy(executor, ast, data, columnData, filteredIndices) {
  const groupingSets = expandGroupBy(ast.groupBy);
  const allGroupColumns = getAllGroupColumns(ast.groupBy);
  const allResults = [];
  for (const groupingSet of groupingSets) {
    const setResults = executeGroupByForSet(
      ast,
      columnData,
      filteredIndices,
      groupingSet,
      allGroupColumns
    );
    allResults.push(...setResults);
  }
  return buildAdvancedGroupByResult(ast, allResults, allGroupColumns);
}
function executeGroupByAggregation(executor, ast, data, columnData, filteredIndices) {
  const hasGroupByClause = ast.groupBy && ast.groupBy.length > 0;
  if (hasGroupByClause && hasAdvancedGroupBy(ast.groupBy)) {
    return executeAdvancedGroupBy(executor, ast, data, columnData, filteredIndices);
  }
  const groups = /* @__PURE__ */ new Map();
  for (const idx of filteredIndices) {
    let groupKey = "";
    if (hasGroupByClause) {
      groupKey = ast.groupBy.map((expr) => {
        const colName = (expr.column || expr.name || "").toLowerCase();
        const val = columnData[colName]?.[idx];
        return val == null ? "\0" : String(val);
      }).join("");
    }
    if (!groups.has(groupKey)) {
      groups.set(groupKey, []);
    }
    groups.get(groupKey).push(idx);
  }
  if (!hasGroupByClause && groups.size === 0) {
    groups.set("", []);
  }
  const resultColumns = [];
  for (const col of ast.columns) {
    if (col.alias) {
      resultColumns.push(col.alias);
    } else if (col.expr?.type === "call") {
      const argName = col.expr.args?.[0]?.name || col.expr.args?.[0]?.column || "*";
      resultColumns.push(`${col.expr.name}(${argName})`);
    } else if (col.expr?.type === "column") {
      resultColumns.push(col.expr.name || col.expr.column);
    } else {
      resultColumns.push("?");
    }
  }
  const resultRows = [];
  for (const [, groupIndices] of groups) {
    const row = [];
    for (const col of ast.columns) {
      const expr = col.expr;
      if (expr?.type === "call" && AGG_FUNCS.includes((expr.name || "").toUpperCase())) {
        const funcName = expr.name.toUpperCase();
        const argExpr = expr.args?.[0];
        const isStar = argExpr?.type === "star";
        const colName = (argExpr?.name || argExpr?.column || "").toLowerCase();
        if (funcName === "COUNT" && isStar) {
          row.push(groupIndices.length);
        } else {
          const values = groupIndices.map((i) => columnData[colName]?.[i]);
          const separator = expr.args?.[1]?.value;
          row.push(computeAggregate2(funcName, values, { separator }));
        }
      } else if (expr?.type === "column") {
        const colName = (expr.name || expr.column || "").toLowerCase();
        row.push(columnData[colName]?.[groupIndices[0]] ?? null);
      } else {
        row.push(evaluateInMemoryExpr(expr, columnData, groupIndices[0]));
      }
    }
    resultRows.push(row);
  }
  if (ast.orderBy && ast.orderBy.length > 0) {
    const colIdxMap = {};
    resultColumns.forEach((name, idx) => {
      colIdxMap[name.toLowerCase()] = idx;
    });
    resultRows.sort((a, b) => {
      for (const ob of ast.orderBy) {
        const colIdx = colIdxMap[ob.column.toLowerCase()];
        if (colIdx === void 0) continue;
        const valA = a[colIdx], valB = b[colIdx];
        const dir = ob.descending || ob.direction === "DESC" ? -1 : 1;
        if (valA == null && valB == null) continue;
        if (valA == null) return 1 * dir;
        if (valB == null) return -1 * dir;
        if (valA < valB) return -1 * dir;
        if (valA > valB) return 1 * dir;
      }
      return 0;
    });
  }
  const offset = ast.offset || 0;
  let rows = resultRows;
  if (offset > 0) rows = rows.slice(offset);
  if (ast.limit) rows = rows.slice(0, ast.limit);
  return { columns: resultColumns, rows, total: groups.size };
}

// src/client/sql/executor-subquery.js
function executeSubquery(executor, subqueryAst, outerColumnData, outerRowIdx) {
  const resolvedAst = structuredClone(subqueryAst);
  const subqueryTable = resolvedAst.from?.name || resolvedAst.from?.table;
  const correlatedColumns = findCorrelatedColumns(resolvedAst, subqueryTable);
  const correlationContext = {};
  for (const col of correlatedColumns) {
    const colName = col.column.toLowerCase();
    if (outerColumnData[colName]) {
      correlationContext[col.table + "." + col.column] = outerColumnData[colName][outerRowIdx];
    }
  }
  if (Object.keys(correlationContext).length > 0) {
    substituteCorrelations(resolvedAst.where, correlationContext);
  }
  const tableName = resolvedAst.from?.name?.toLowerCase() || resolvedAst.from?.table?.toLowerCase();
  if (tableName && executor._cteResults?.has(tableName)) {
    const result = executeOnInMemoryData(executor, resolvedAst, executor._cteResults.get(tableName));
    return result.rows.length > 0 ? result.rows[0][0] : null;
  }
  if (executor._database) {
    try {
      const result = executor._database._executeSingleTable(resolvedAst);
      if (result && result.then) {
        return null;
      }
      return result?.rows?.[0]?.[0] ?? null;
    } catch {
      return null;
    }
  }
  return null;
}
function findCorrelatedColumns(ast, subqueryTable) {
  const correlatedCols = [];
  const walkExpr = (expr) => {
    if (!expr) return;
    if (expr.type === "column" && expr.table && expr.table !== subqueryTable) {
      correlatedCols.push(expr);
    } else if (expr.type === "binary") {
      walkExpr(expr.left);
      walkExpr(expr.right);
    } else if (expr.type === "unary") {
      walkExpr(expr.operand);
    } else if (expr.type === "in") {
      walkExpr(expr.expr);
      expr.values?.forEach(walkExpr);
    } else if (expr.type === "between") {
      walkExpr(expr.expr);
      walkExpr(expr.low);
      walkExpr(expr.high);
    } else if (expr.type === "like") {
      walkExpr(expr.expr);
      walkExpr(expr.pattern);
    } else if (expr.type === "call") {
      expr.args?.forEach(walkExpr);
    }
  };
  walkExpr(ast.where);
  return correlatedCols;
}
function substituteCorrelations(expr, correlationContext) {
  if (!expr) return;
  if (expr.type === "column" && expr.table) {
    const key = expr.table + "." + expr.column;
    if (correlationContext.hasOwnProperty(key)) {
      expr.type = "literal";
      expr.value = correlationContext[key];
      delete expr.table;
      delete expr.column;
    }
  } else if (expr.type === "binary") {
    substituteCorrelations(expr.left, correlationContext);
    substituteCorrelations(expr.right, correlationContext);
  } else if (expr.type === "unary") {
    substituteCorrelations(expr.operand, correlationContext);
  } else if (expr.type === "in") {
    substituteCorrelations(expr.expr, correlationContext);
    expr.values?.forEach((v) => substituteCorrelations(v, correlationContext));
  } else if (expr.type === "between") {
    substituteCorrelations(expr.expr, correlationContext);
    substituteCorrelations(expr.low, correlationContext);
    substituteCorrelations(expr.high, correlationContext);
  } else if (expr.type === "like") {
    substituteCorrelations(expr.expr, correlationContext);
    substituteCorrelations(expr.pattern, correlationContext);
  } else if (expr.type === "call") {
    expr.args?.forEach((a) => substituteCorrelations(a, correlationContext));
  }
}
async function materializeCTEs(executor, ctes, db) {
  executor._database = db;
  for (const cte of ctes) {
    const cteName = cte.name.toLowerCase();
    if (cte.body.type === "RECURSIVE_CTE") {
      const anchorResult = await executeCTEBody(executor, cte.body.anchor, db);
      let result = { columns: anchorResult.columns, rows: [...anchorResult.rows] };
      for (let i = 0; i < 1e3; i++) {
        executor._cteResults.set(cteName, result);
        const recursiveResult = await executeCTEBody(executor, cte.body.recursive, db);
        if (recursiveResult.rows.length === 0) break;
        result = { columns: result.columns, rows: [...result.rows, ...recursiveResult.rows] };
      }
      executor._cteResults.set(cteName, result);
    } else {
      const result = await executeCTEBody(executor, cte.body, db);
      executor._cteResults.set(cteName, result);
    }
  }
}
async function executeCTEBody(executor, bodyAst, db) {
  const tableName = bodyAst.from?.name?.toLowerCase() || bodyAst.from?.table?.toLowerCase();
  if (tableName && executor._cteResults.has(tableName)) {
    return executeOnInMemoryData(executor, bodyAst, executor._cteResults.get(tableName));
  }
  return db._executeSingleTable(bodyAst);
}
function hasAggregatesInSelect(columns) {
  const aggFuncs = ["COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "STDDEV_SAMP", "STDDEV_POP", "VARIANCE", "VAR_SAMP", "VAR_POP", "MEDIAN", "STRING_AGG", "GROUP_CONCAT"];
  for (const col of columns) {
    if (col.expr?.type === "call") {
      if (col.expr.over) continue;
      const funcName = (col.expr.name || "").toUpperCase();
      if (aggFuncs.includes(funcName)) return true;
    }
  }
  return false;
}
function executeOnInMemoryData(executor, ast, data) {
  const columnData = {};
  for (let i = 0; i < data.columns.length; i++) {
    const colName = data.columns[i].toLowerCase();
    columnData[colName] = data.rows.map((row) => row[i]);
  }
  const filteredIndices = [];
  for (let i = 0; i < data.rows.length; i++) {
    if (!ast.where || evaluateInMemoryExpr(ast.where, columnData, i)) {
      filteredIndices.push(i);
    }
  }
  const hasGroupBy = ast.groupBy && ast.groupBy.length > 0;
  const hasAggregates2 = hasAggregatesInSelect(ast.columns);
  if (hasGroupBy || hasAggregates2) {
    return executor._executeGroupByAggregation(ast, data, columnData, filteredIndices);
  }
  if (executor.hasWindowFunctions(ast)) {
    return executor._executeWindowFunctions(ast, data, columnData, filteredIndices);
  }
  const resultColumns = [];
  const resultRows = [];
  const isSelectStar = ast.columns.length === 1 && (ast.columns[0].type === "star" || ast.columns[0].expr?.type === "star");
  if (isSelectStar) {
    for (const colName of data.columns) {
      resultColumns.push(colName);
    }
    for (const idx of filteredIndices) {
      resultRows.push([...data.rows[idx]]);
    }
  } else {
    for (const col of ast.columns) {
      resultColumns.push(col.alias || col.expr?.column || "*");
    }
    for (const idx of filteredIndices) {
      const row = ast.columns.map((col) => {
        if (col.type === "star" || col.expr?.type === "star") {
          return data.rows[idx];
        }
        return evaluateInMemoryExpr(col.expr, columnData, idx);
      });
      resultRows.push(row.flat());
    }
  }
  if (ast.orderBy && ast.orderBy.length > 0) {
    const colIdxMap = {};
    resultColumns.forEach((name, idx) => {
      colIdxMap[name.toLowerCase()] = idx;
    });
    resultRows.sort((a, b) => {
      for (const ob of ast.orderBy) {
        const colIdx = colIdxMap[ob.column.toLowerCase()];
        if (colIdx === void 0) continue;
        const valA = a[colIdx], valB = b[colIdx];
        const dir = ob.descending || ob.direction === "DESC" ? -1 : 1;
        if (valA == null && valB == null) continue;
        if (valA == null) return 1 * dir;
        if (valB == null) return -1 * dir;
        if (valA < valB) return -1 * dir;
        if (valA > valB) return 1 * dir;
      }
      return 0;
    });
  }
  const offset = ast.offset || 0;
  let rows = resultRows;
  if (offset > 0) rows = rows.slice(offset);
  if (ast.limit) rows = rows.slice(0, ast.limit);
  return { columns: resultColumns, rows, total: filteredIndices.length };
}

// src/client/sql/executor.js
var SQLExecutor = class {
  constructor(file, options = {}) {
    this.file = file;
    this.columnMap = {};
    this.columnTypes = [];
    this._cteResults = /* @__PURE__ */ new Map();
    this._database = null;
    this._ftsIndexCache = null;
    this._debug = options.debug ?? false;
    if (file.columnNames) {
      file.columnNames.forEach((name, idx) => {
        this.columnMap[name.toLowerCase()] = idx;
      });
    }
  }
  setDatabase(db) {
    this._database = db;
  }
  /**
   * Execute a SQL query
   */
  async execute(sql, onProgress = null) {
    const lexer = new SQLLexer(sql);
    const tokens = lexer.tokenize();
    const parser = new SQLParser(tokens);
    const ast = parser.parse();
    const planner = new QueryPlanner();
    const plan = planner.planSingleTable(ast);
    if (this.columnTypes.length === 0) {
      if (this.file._isRemote && this.file.detectColumnTypes) {
        this.columnTypes = await this.file.detectColumnTypes();
      } else if (this.file._columnTypes) {
        this.columnTypes = this.file._columnTypes;
      } else {
        this.columnTypes = Array(this.file.numColumns || 0).fill("unknown");
      }
    }
    const totalRows = this.file._isRemote ? await this.file.getRowCount(0) : Number(this.file.getRowCount(0));
    let columnStats = null;
    if (ast.where && plan.pushedFilters.length > 0 && this.file._isRemote) {
      columnStats = await statisticsManager.precomputeForPlan(this.file, plan);
      plan.columnStats = Object.fromEntries(columnStats);
    }
    const neededColumns = plan.scanColumns.length > 0 ? plan.scanColumns : this.collectNeededColumns(ast);
    const outputColumns = this.resolveOutputColumns(ast);
    const hasAggregates2 = plan.aggregations.length > 0 || hasAggregates(ast);
    if (hasAggregates2) {
      if (isSimpleCountStar(ast) && !ast.where && !ast.search) {
        return {
          columns: ["COUNT(*)"],
          rows: [[totalRows]],
          total: 1,
          aggregationStats: { scannedRows: 0, totalRows, coveragePercent: "100.00", isPartialScan: false, fromMetadata: true },
          queryPlan: plan
        };
      }
      if (ast.search || extractNearCondition(ast.where)) {
        return await executeAggregateWithSearch(this, ast, totalRows, onProgress);
      }
      return await this.executeAggregateQuery(ast, totalRows, onProgress);
    }
    let indices;
    const limit = ast.limit || 100;
    const offset = ast.offset || 0;
    if (!ast.where) {
      indices = [];
      const endIdx = Math.min(offset + limit, totalRows);
      for (let i = offset; i < endIdx; i++) indices.push(i);
    } else {
      indices = await this.evaluateWhere(ast.where, totalRows, onProgress);
      indices = indices.slice(offset, offset + limit);
    }
    if (onProgress) onProgress("Fetching column data...", 0, outputColumns.length);
    const columnData = {};
    for (let i = 0; i < neededColumns.length; i++) {
      const colName = neededColumns[i];
      const colIdx = this.columnMap[colName.toLowerCase()];
      if (colIdx === void 0) continue;
      if (onProgress) onProgress(`Fetching ${colName}...`, i, neededColumns.length);
      columnData[colName.toLowerCase()] = await this.readColumnData(colIdx, indices);
    }
    const rows = [];
    for (let i = 0; i < indices.length; i++) {
      const row = [];
      for (const col of outputColumns) {
        if (col.type === "star") {
          for (const name of this.file.columnNames || []) {
            row.push(columnData[name.toLowerCase()]?.[i] ?? null);
          }
        } else {
          row.push(this.evaluateExpr(col.expr, columnData, i));
        }
      }
      rows.push(row);
    }
    if (ast.pivot) {
      const pivotResult = executePivot(rows, outputColumns, ast.pivot);
      rows.length = 0;
      rows.push(...pivotResult.rows);
      outputColumns.length = 0;
      outputColumns.push(...pivotResult.columns);
    }
    if (ast.unpivot) {
      const unpivotResult = executeUnpivot(rows, outputColumns, ast.unpivot);
      rows.length = 0;
      rows.push(...unpivotResult.rows);
      outputColumns.length = 0;
      outputColumns.push(...unpivotResult.columns);
    }
    if (ast.distinct) {
      const uniqueRows = await applyDistinct(this, rows, gpuSorter);
      rows.length = 0;
      rows.push(...uniqueRows);
    }
    if (ast.orderBy && ast.orderBy.length > 0) {
      await applyOrderBy(this, rows, ast.orderBy, outputColumns, gpuSorter);
    }
    const colNames = [];
    for (const col of outputColumns) {
      if (col.type === "star") {
        colNames.push(...this.file.columnNames || []);
      } else {
        colNames.push(col.alias || exprToName(col.expr));
      }
    }
    return {
      columns: colNames,
      rows,
      total: ast.limit ? rows.length : totalRows,
      orderByOnSubset: ast.orderBy && ast.orderBy.length > 0 && rows.length < totalRows,
      orderByColumns: ast.orderBy ? ast.orderBy.map((ob) => `${ob.column} ${ob.direction}`) : [],
      queryPlan: plan,
      optimization: {
        statsComputed: columnStats?.size > 0,
        columnStats: columnStats ? Object.fromEntries(columnStats) : null,
        pushedFilters: plan.pushedFilters?.length || 0,
        estimatedSelectivity: plan.estimatedSelectivity
      }
    };
  }
  // === Column Collection ===
  collectNeededColumns(ast) {
    const columns = /* @__PURE__ */ new Set();
    for (const item of ast.columns) {
      if (item.type === "star") {
        (this.file.columnNames || []).forEach((n) => columns.add(n.toLowerCase()));
      } else {
        collectColumnsFromExpr2(item.expr, columns);
      }
    }
    if (ast.where) collectColumnsFromExpr2(ast.where, columns);
    for (const ob of ast.orderBy || []) columns.add(ob.column.toLowerCase());
    return Array.from(columns);
  }
  collectColumnsFromExpr(expr, columns) {
    collectColumnsFromExpr2(expr, columns);
  }
  resolveOutputColumns(ast) {
    return ast.columns;
  }
  // === Data Reading ===
  async readColumnData(colIdx, indices) {
    const type = this.columnTypes[colIdx] || "unknown";
    try {
      if (type === "string") {
        const data = await this.file.readStringsAtIndices(colIdx, indices);
        return Array.isArray(data) ? data : Array.from(data);
      } else if (type === "int64") {
        const data = await this.file.readInt64AtIndices(colIdx, indices);
        return Array.from(data, (v) => Number(v));
      } else if (type === "float64") {
        return Array.from(await this.file.readFloat64AtIndices(colIdx, indices));
      } else if (type === "int32") {
        return Array.from(await this.file.readInt32AtIndices(colIdx, indices));
      } else if (type === "float32") {
        return Array.from(await this.file.readFloat32AtIndices(colIdx, indices));
      } else if (type === "vector") {
        return indices.map(() => "[vector]");
      } else {
        try {
          return await this.file.readStringsAtIndices(colIdx, indices);
        } catch (e) {
          if (this._debug) console.warn(`[SQLExecutor] readColumnData col ${colIdx} fallback failed:`, e.message);
          return indices.map(() => null);
        }
      }
    } catch (e) {
      if (this._debug) console.warn(`[SQLExecutor] readColumnData col ${colIdx} failed:`, e.message);
      return indices.map(() => null);
    }
  }
  // === WHERE Evaluation ===
  async evaluateWhere(whereExpr, totalRows, onProgress) {
    const nearInfo = extractNearCondition(whereExpr);
    if (nearInfo) {
      return await evaluateWithNear(this, nearInfo, whereExpr, totalRows, onProgress);
    }
    const simpleFilter = detectSimpleFilter(whereExpr, this.columnMap, this.columnTypes);
    if (simpleFilter) {
      return await evaluateSimpleFilter(this, simpleFilter, totalRows, onProgress);
    }
    return await evaluateComplexFilter(this, whereExpr, totalRows, onProgress);
  }
  evaluateExpr(expr, columnData, rowIdx) {
    return evaluateExpr(this, expr, columnData, rowIdx);
  }
  // === Subquery/CTE Support ===
  _executeSubquery(subqueryAst, outerColumnData, outerRowIdx) {
    return executeSubquery(this, subqueryAst, outerColumnData, outerRowIdx);
  }
  async materializeCTEs(ctes, db) {
    return materializeCTEs(this, ctes, db);
  }
  _executeOnInMemoryData(ast, data) {
    return executeOnInMemoryData(this, ast, data);
  }
  _evaluateInMemoryExpr(expr, columnData, rowIdx) {
    return evaluateInMemoryExpr(expr, columnData, rowIdx);
  }
  // === Window Functions ===
  hasWindowFunctions(ast) {
    return hasWindowFunctions(ast);
  }
  computeWindowFunctions(ast, rows, columnData) {
    return computeWindowFunctions(
      ast,
      rows,
      columnData,
      (expr, colData, idx) => this._evaluateInMemoryExpr(expr, colData, idx)
    );
  }
  _executeWindowFunctions(ast, data, columnData, filteredIndices) {
    return executeWindowFunctions(this, ast, data, columnData, filteredIndices);
  }
  // === Aggregation ===
  hasAggregates(ast) {
    return hasAggregates(ast);
  }
  _executeGroupByAggregation(ast, data, columnData, filteredIndices) {
    return executeGroupByAggregation(this, ast, data, columnData, filteredIndices);
  }
  _hasAdvancedGroupBy(groupBy) {
    return hasAdvancedGroupBy(groupBy);
  }
  _emptyAggregateResult(ast) {
    return emptyAggregateResult(ast);
  }
  async executeAggregateQuery(ast, totalRows, onProgress) {
    const aggFunctions = AGG_FUNCS;
    const aggregators = [];
    const colNames = [];
    for (const col of ast.columns) {
      if (col.type === "star") {
        aggregators.push({ type: "COUNT", column: null, isStar: true });
        colNames.push("COUNT(*)");
      } else if (col.expr.type === "call" && aggFunctions.includes(col.expr.name.toUpperCase())) {
        const aggType = col.expr.name.toUpperCase();
        const argExpr = col.expr.args[0];
        aggregators.push({
          type: aggType,
          column: argExpr?.type === "column" ? argExpr.name : null,
          isStar: argExpr?.type === "star",
          sum: 0,
          count: 0,
          min: null,
          max: null,
          values: []
        });
        colNames.push(col.alias || exprToName(col.expr));
      } else {
        aggregators.push({
          type: "FIRST",
          column: col.expr.type === "column" ? col.expr.name : null,
          value: null
        });
        colNames.push(col.alias || exprToName(col.expr));
      }
    }
    const neededCols = /* @__PURE__ */ new Set();
    for (const agg of aggregators) {
      if (agg.column) neededCols.add(agg.column.toLowerCase());
    }
    if (ast.where) this.collectColumnsFromExpr(ast.where, neededCols);
    const maxRowsToScan = ast.limit ? Math.min(ast.limit, totalRows) : totalRows;
    const batchSize = 1e3;
    let scannedRows = 0;
    for (let batchStart = 0; batchStart < maxRowsToScan; batchStart += batchSize) {
      if (onProgress) onProgress(`Aggregating...`, batchStart, maxRowsToScan);
      const batchEnd = Math.min(batchStart + batchSize, maxRowsToScan);
      const batchIndices2 = Array.from({ length: batchEnd - batchStart }, (_, i) => batchStart + i);
      scannedRows += batchIndices2.length;
      const batchData = {};
      for (const colName of neededCols) {
        const colIdx = this.columnMap[colName.toLowerCase()];
        if (colIdx !== void 0) {
          batchData[colName.toLowerCase()] = await this.readColumnData(colIdx, batchIndices2);
        }
      }
      for (let i = 0; i < batchIndices2.length; i++) {
        if (ast.where && !this.evaluateExpr(ast.where, batchData, i)) continue;
        for (const agg of aggregators) {
          if (agg.type === "COUNT") {
            agg.count++;
          } else if (agg.type === "FIRST" && agg.value === null && agg.column) {
            agg.value = batchData[agg.column.toLowerCase()]?.[i];
          } else if (agg.column) {
            const val = batchData[agg.column.toLowerCase()]?.[i];
            if (val != null && !isNaN(val)) {
              agg.count++;
              if (agg.type === "SUM" || agg.type === "AVG") agg.sum += val;
              if (agg.type === "MIN") agg.min = agg.min === null ? val : Math.min(agg.min, val);
              if (agg.type === "MAX") agg.max = agg.max === null ? val : Math.max(agg.max, val);
            }
          }
        }
      }
    }
    const resultRow = aggregators.map((agg) => {
      switch (agg.type) {
        case "COUNT":
          return agg.count;
        case "SUM":
          return agg.sum;
        case "AVG":
          return agg.count > 0 ? agg.sum / agg.count : null;
        case "MIN":
          return agg.min;
        case "MAX":
          return agg.max;
        case "FIRST":
          return agg.value;
        default:
          return null;
      }
    });
    if (ast.having) {
      const havingData = {};
      colNames.forEach((name, i) => {
        havingData[name.toLowerCase()] = [resultRow[i]];
      });
      if (!this._evaluateInMemoryExpr(ast.having, havingData, 0)) {
        return { columns: colNames, rows: [], total: 0, aggregationStats: { scannedRows, totalRows } };
      }
    }
    return {
      columns: colNames,
      rows: [resultRow],
      total: 1,
      aggregationStats: {
        scannedRows,
        totalRows,
        coveragePercent: totalRows > 0 ? (scannedRows / totalRows * 100).toFixed(2) : "100.00",
        isPartialScan: scannedRows < totalRows
      }
    };
  }
  async _executeSimpleAggregateOnIndices(ast, indices, onProgress) {
    const colNames = ast.columns.map((col) => col.alias || exprToName(col.expr || col));
    const neededCols = /* @__PURE__ */ new Set();
    for (const col of ast.columns) {
      if (col.expr?.args?.[0]?.type === "column") {
        neededCols.add((col.expr.args[0].name || col.expr.args[0].column).toLowerCase());
      }
    }
    const columnData = {};
    for (const colName of neededCols) {
      const colIdx = this.columnMap[colName];
      if (colIdx !== void 0) {
        columnData[colName] = await this.readColumnData(colIdx, indices);
      }
    }
    const resultRow = ast.columns.map((col) => {
      if (col.expr?.type === "call") {
        const funcName = col.expr.name.toUpperCase();
        const argExpr = col.expr.args?.[0];
        const isStar = argExpr?.type === "star";
        if (funcName === "COUNT" && isStar) return indices.length;
        const colName = (argExpr?.name || argExpr?.column)?.toLowerCase();
        const values = indices.map((_, i) => columnData[colName]?.[i]);
        return computeAggregate2(funcName, values);
      }
      return null;
    });
    return { columns: colNames, rows: [resultRow], total: 1 };
  }
  async _executeGroupByOnIndices(ast, indices, onProgress) {
    const neededCols = /* @__PURE__ */ new Set();
    for (const expr of ast.groupBy) {
      neededCols.add((expr.column || expr.name).toLowerCase());
    }
    for (const col of ast.columns) {
      if (col.expr?.type === "column") neededCols.add((col.expr.name || col.expr.column).toLowerCase());
      if (col.expr?.args?.[0]?.type === "column") neededCols.add((col.expr.args[0].name || col.expr.args[0].column).toLowerCase());
    }
    const columnData = {};
    for (const colName of neededCols) {
      const colIdx = this.columnMap[colName];
      if (colIdx !== void 0) {
        columnData[colName] = await this.readColumnData(colIdx, indices);
      }
    }
    const groups = /* @__PURE__ */ new Map();
    for (let i = 0; i < indices.length; i++) {
      const key = ast.groupBy.map((expr) => JSON.stringify(columnData[(expr.column || expr.name).toLowerCase()]?.[i])).join("|");
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(i);
    }
    const colNames = ast.columns.map((col) => col.alias || exprToName(col.expr || col));
    const resultRows = [];
    for (const [, groupIndices] of groups) {
      const row = ast.columns.map((col) => {
        const expr = col.expr || col;
        if (expr.type === "call" && AGG_FUNCS.includes(expr.name.toUpperCase())) {
          const colName = (expr.args?.[0]?.name || expr.args?.[0]?.column)?.toLowerCase();
          const isStar = expr.args?.[0]?.type === "star";
          if (expr.name.toUpperCase() === "COUNT" && isStar) return groupIndices.length;
          const values = groupIndices.map((i) => columnData[colName]?.[i]);
          return computeAggregate2(expr.name.toUpperCase(), values);
        } else if (expr.type === "column") {
          return columnData[(expr.name || expr.column).toLowerCase()]?.[groupIndices[0]];
        }
        return null;
      });
      resultRows.push(row);
    }
    return { columns: colNames, rows: resultRows, total: resultRows.length };
  }
  // === Utility Methods ===
  exprToName(expr) {
    return exprToName(expr);
  }
  isSimpleCountStar(ast) {
    return isSimpleCountStar(ast);
  }
  async applyOrderBy(rows, orderBy, outputColumns) {
    return applyOrderBy(this, rows, orderBy, outputColumns, gpuSorter);
  }
  async applyDistinct(rows) {
    return applyDistinct(this, rows, gpuSorter);
  }
  // === Streaming ===
  async *executeStream(sql, options = {}) {
    const { chunkSize = 1e3 } = options;
    const lexer = new SQLLexer(sql);
    const tokens = lexer.tokenize();
    const parser = new SQLParser(tokens);
    const ast = parser.parse();
    if (this.columnTypes.length === 0) {
      if (this.file._isRemote && this.file.detectColumnTypes) {
        this.columnTypes = await this.file.detectColumnTypes();
      } else if (this.file._columnTypes) {
        this.columnTypes = this.file._columnTypes;
      } else {
        this.columnTypes = Array(this.file.numColumns || 0).fill("unknown");
      }
    }
    const totalRows = this.file._isRemote ? await this.file.getRowCount(0) : Number(this.file.getRowCount(0));
    const neededColumns = this.collectNeededColumns(ast);
    const limit = ast.limit || totalRows;
    let yielded = 0;
    for (let offset = 0; offset < totalRows && yielded < limit; offset += chunkSize) {
      const batchSize = Math.min(chunkSize, limit - yielded, totalRows - offset);
      const indices = Array.from({ length: batchSize }, (_, i) => offset + i);
      const columnData = [];
      for (const colName of neededColumns) {
        const colIdx = this.columnMap[colName.toLowerCase()];
        columnData.push(colIdx !== void 0 ? await this.readColumnData(colIdx, indices) : indices.map(() => null));
      }
      const rows = indices.map((_, i) => neededColumns.map((_2, c) => columnData[c][i]));
      let filteredRows = rows;
      if (ast.where) {
        filteredRows = rows.filter(
          (row) => evaluateWhereExprOnRow(ast.where, neededColumns, row, getValueFromExpr)
        );
      }
      if (filteredRows.length > 0) {
        yield { columns: neededColumns, rows: filteredRows };
        yielded += filteredRows.length;
      }
    }
  }
};

// src/client/lance/lance-file.js
init_accelerator();
var LanceFile2 = class {
  constructor(lanceql, data) {
    this.lanceql = lanceql;
    this.wasm = lanceql.wasm;
    this.memory = lanceql.memory;
    const bytes = new Uint8Array(data);
    this.dataPtr = this.wasm.alloc(bytes.length);
    if (!this.dataPtr) {
      throw new Error("Failed to allocate memory for Lance file");
    }
    this.dataLen = bytes.length;
    new Uint8Array(this.memory.buffer).set(bytes, this.dataPtr);
    const result = this.wasm.openFile(this.dataPtr, this.dataLen);
    if (result === 0) {
      this.wasm.free(this.dataPtr, this.dataLen);
      throw new Error("Failed to open Lance file");
    }
  }
  /**
   * Close the file and free memory.
   */
  close() {
    this.wasm.closeFile();
    if (this.dataPtr) {
      this.wasm.free(this.dataPtr, this.dataLen);
      this.dataPtr = null;
    }
  }
  /**
   * Get the number of columns.
   * @returns {number}
   */
  get numColumns() {
    return this.wasm.getNumColumns();
  }
  /**
   * Get the row count for a column.
   * @param {number} colIdx
   * @returns {bigint}
   */
  getRowCount(colIdx) {
    return this.wasm.getRowCount(colIdx);
  }
  /**
   * Get debug info for a column.
   * @param {number} colIdx
   * @returns {{offset: bigint, size: bigint, rows: bigint}}
   */
  getColumnDebugInfo(colIdx) {
    return {
      offset: this.wasm.getColumnBufferOffset(colIdx),
      size: this.wasm.getColumnBufferSize(colIdx),
      rows: this.wasm.getRowCount(colIdx)
    };
  }
  /**
   * Read an int64 column as a BigInt64Array.
   * @param {number} colIdx - Column index
   * @returns {BigInt64Array}
   */
  readInt64Column(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new BigInt64Array(0);
    const bufPtr = this.wasm.allocInt64Buffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate int64 buffer");
    try {
      const count = this.wasm.readInt64Column(colIdx, bufPtr, rowCount);
      const result = new BigInt64Array(count);
      const view = new BigInt64Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.freeInt64Buffer(bufPtr, rowCount);
    }
  }
  /**
   * Read a float64 column as a Float64Array.
   * @param {number} colIdx - Column index
   * @returns {Float64Array}
   */
  readFloat64Column(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Float64Array(0);
    const bufPtr = this.wasm.allocFloat64Buffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate float64 buffer");
    try {
      const count = this.wasm.readFloat64Column(colIdx, bufPtr, rowCount);
      const result = new Float64Array(count);
      const view = new Float64Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.freeFloat64Buffer(bufPtr, rowCount);
    }
  }
  // ========================================================================
  // Additional Numeric Type Column Methods
  // ========================================================================
  /**
   * Read an int32 column as an Int32Array.
   * @param {number} colIdx - Column index
   * @returns {Int32Array}
   */
  readInt32Column(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Int32Array(0);
    const bufPtr = this.wasm.allocInt32Buffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate int32 buffer");
    try {
      const count = this.wasm.readInt32Column(colIdx, bufPtr, rowCount);
      const result = new Int32Array(count);
      const view = new Int32Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(bufPtr, rowCount * 4);
    }
  }
  /**
   * Read an int16 column as an Int16Array.
   * @param {number} colIdx - Column index
   * @returns {Int16Array}
   */
  readInt16Column(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Int16Array(0);
    const bufPtr = this.wasm.allocInt16Buffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate int16 buffer");
    try {
      const count = this.wasm.readInt16Column(colIdx, bufPtr, rowCount);
      const result = new Int16Array(count);
      const view = new Int16Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(bufPtr, rowCount * 2);
    }
  }
  /**
   * Read an int8 column as an Int8Array.
   * @param {number} colIdx - Column index
   * @returns {Int8Array}
   */
  readInt8Column(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Int8Array(0);
    const bufPtr = this.wasm.allocInt8Buffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate int8 buffer");
    try {
      const count = this.wasm.readInt8Column(colIdx, bufPtr, rowCount);
      const result = new Int8Array(count);
      const view = new Int8Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(bufPtr, rowCount);
    }
  }
  /**
   * Read a uint64 column as a BigUint64Array.
   * @param {number} colIdx - Column index
   * @returns {BigUint64Array}
   */
  readUint64Column(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new BigUint64Array(0);
    const bufPtr = this.wasm.allocUint64Buffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate uint64 buffer");
    try {
      const count = this.wasm.readUint64Column(colIdx, bufPtr, rowCount);
      const result = new BigUint64Array(count);
      const view = new BigUint64Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(bufPtr, rowCount * 8);
    }
  }
  /**
   * Read a uint32 column as a Uint32Array.
   * @param {number} colIdx - Column index
   * @returns {Uint32Array}
   */
  readUint32Column(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Uint32Array(0);
    const bufPtr = this.wasm.allocIndexBuffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate uint32 buffer");
    try {
      const count = this.wasm.readUint32Column(colIdx, bufPtr, rowCount);
      const result = new Uint32Array(count);
      const view = new Uint32Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(bufPtr, rowCount * 4);
    }
  }
  /**
   * Read a uint16 column as a Uint16Array.
   * @param {number} colIdx - Column index
   * @returns {Uint16Array}
   */
  readUint16Column(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Uint16Array(0);
    const bufPtr = this.wasm.allocUint16Buffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate uint16 buffer");
    try {
      const count = this.wasm.readUint16Column(colIdx, bufPtr, rowCount);
      const result = new Uint16Array(count);
      const view = new Uint16Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(bufPtr, rowCount * 2);
    }
  }
  /**
   * Read a uint8 column as a Uint8Array.
   * @param {number} colIdx - Column index
   * @returns {Uint8Array}
   */
  readUint8Column(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Uint8Array(0);
    const bufPtr = this.wasm.allocStringBuffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate uint8 buffer");
    try {
      const count = this.wasm.readUint8Column(colIdx, bufPtr, rowCount);
      const result = new Uint8Array(count);
      const view = new Uint8Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(bufPtr, rowCount);
    }
  }
  /**
   * Read a float32 column as a Float32Array.
   * @param {number} colIdx - Column index
   * @returns {Float32Array}
   */
  readFloat32Column(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Float32Array(0);
    const bufPtr = this.wasm.allocFloat32Buffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate float32 buffer");
    try {
      const count = this.wasm.readFloat32Column(colIdx, bufPtr, rowCount);
      const result = new Float32Array(count);
      const view = new Float32Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(bufPtr, rowCount * 4);
    }
  }
  /**
   * Read a boolean column as a Uint8Array (0 or 1 values).
   * @param {number} colIdx - Column index
   * @returns {Uint8Array}
   */
  readBoolColumn(colIdx) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Uint8Array(0);
    const bufPtr = this.wasm.allocStringBuffer(rowCount);
    if (!bufPtr) throw new Error("Failed to allocate bool buffer");
    try {
      const count = this.wasm.readBoolColumn(colIdx, bufPtr, rowCount);
      const result = new Uint8Array(count);
      const view = new Uint8Array(this.memory.buffer, bufPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(bufPtr, rowCount);
    }
  }
  /**
   * Read int32 values at specific row indices.
   * @param {number} colIdx - Column index
   * @param {Uint32Array} indices - Row indices to read
   * @returns {Int32Array}
   */
  readInt32AtIndices(colIdx, indices) {
    if (indices.length === 0) return new Int32Array(0);
    const idxPtr = this.wasm.allocIndexBuffer(indices.length);
    if (!idxPtr) throw new Error("Failed to allocate index buffer");
    const outPtr = this.wasm.allocInt32Buffer(indices.length);
    if (!outPtr) {
      this.wasm.free(idxPtr, indices.length * 4);
      throw new Error("Failed to allocate output buffer");
    }
    try {
      new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
      const count = this.wasm.readInt32AtIndices(colIdx, idxPtr, indices.length, outPtr);
      const result = new Int32Array(count);
      const view = new Int32Array(this.memory.buffer, outPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(idxPtr, indices.length * 4);
      this.wasm.free(outPtr, indices.length * 4);
    }
  }
  /**
   * Read float32 values at specific row indices.
   * @param {number} colIdx - Column index
   * @param {Uint32Array} indices - Row indices to read
   * @returns {Float32Array}
   */
  readFloat32AtIndices(colIdx, indices) {
    if (indices.length === 0) return new Float32Array(0);
    const idxPtr = this.wasm.allocIndexBuffer(indices.length);
    if (!idxPtr) throw new Error("Failed to allocate index buffer");
    const outPtr = this.wasm.allocFloat32Buffer(indices.length);
    if (!outPtr) {
      this.wasm.free(idxPtr, indices.length * 4);
      throw new Error("Failed to allocate output buffer");
    }
    try {
      new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
      const count = this.wasm.readFloat32AtIndices(colIdx, idxPtr, indices.length, outPtr);
      const result = new Float32Array(count);
      const view = new Float32Array(this.memory.buffer, outPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(idxPtr, indices.length * 4);
      this.wasm.free(outPtr, indices.length * 4);
    }
  }
  /**
   * Read uint8 values at specific row indices.
   * @param {number} colIdx - Column index
   * @param {Uint32Array} indices - Row indices to read
   * @returns {Uint8Array}
   */
  readUint8AtIndices(colIdx, indices) {
    if (indices.length === 0) return new Uint8Array(0);
    const idxPtr = this.wasm.allocIndexBuffer(indices.length);
    if (!idxPtr) throw new Error("Failed to allocate index buffer");
    const outPtr = this.wasm.allocStringBuffer(indices.length);
    if (!outPtr) {
      this.wasm.free(idxPtr, indices.length * 4);
      throw new Error("Failed to allocate output buffer");
    }
    try {
      new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
      const count = this.wasm.readUint8AtIndices(colIdx, idxPtr, indices.length, outPtr);
      const result = new Uint8Array(count);
      const view = new Uint8Array(this.memory.buffer, outPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(idxPtr, indices.length * 4);
      this.wasm.free(outPtr, indices.length);
    }
  }
  /**
   * Read bool values at specific row indices.
   * @param {number} colIdx - Column index
   * @param {Uint32Array} indices - Row indices to read
   * @returns {Uint8Array}
   */
  readBoolAtIndices(colIdx, indices) {
    if (indices.length === 0) return new Uint8Array(0);
    const idxPtr = this.wasm.allocIndexBuffer(indices.length);
    if (!idxPtr) throw new Error("Failed to allocate index buffer");
    const outPtr = this.wasm.allocStringBuffer(indices.length);
    if (!outPtr) {
      this.wasm.free(idxPtr, indices.length * 4);
      throw new Error("Failed to allocate output buffer");
    }
    try {
      new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
      const count = this.wasm.readBoolAtIndices(colIdx, idxPtr, indices.length, outPtr);
      const result = new Uint8Array(count);
      const view = new Uint8Array(this.memory.buffer, outPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(idxPtr, indices.length * 4);
      this.wasm.free(outPtr, indices.length);
    }
  }
  /**
   * Filter int64 column and return matching row indices.
   * @param {number} colIdx - Column index
   * @param {number} op - Comparison operator (use LanceFile.Op)
   * @param {bigint|number} value - Value to compare against
   * @returns {Uint32Array} Array of matching row indices
   */
  filterInt64(colIdx, op, value) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Uint32Array(0);
    const idxPtr = this.wasm.allocIndexBuffer(rowCount);
    if (!idxPtr) throw new Error("Failed to allocate index buffer");
    try {
      const count = this.wasm.filterInt64Column(
        colIdx,
        op,
        BigInt(value),
        idxPtr,
        rowCount
      );
      const result = new Uint32Array(count);
      const view = new Uint32Array(this.memory.buffer, idxPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(idxPtr, rowCount * 4);
    }
  }
  /**
   * Filter float64 column and return matching row indices.
   * @param {number} colIdx - Column index
   * @param {number} op - Comparison operator (use LanceFile.Op)
   * @param {number} value - Value to compare against
   * @returns {Uint32Array} Array of matching row indices
   */
  filterFloat64(colIdx, op, value) {
    const rowCount = Number(this.getRowCount(colIdx));
    if (rowCount === 0) return new Uint32Array(0);
    const idxPtr = this.wasm.allocIndexBuffer(rowCount);
    if (!idxPtr) throw new Error("Failed to allocate index buffer");
    try {
      const count = this.wasm.filterFloat64Column(
        colIdx,
        op,
        value,
        idxPtr,
        rowCount
      );
      const result = new Uint32Array(count);
      const view = new Uint32Array(this.memory.buffer, idxPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(idxPtr, rowCount * 4);
    }
  }
  /**
   * Read int64 values at specific row indices.
   * @param {number} colIdx - Column index
   * @param {Uint32Array} indices - Row indices to read
   * @returns {BigInt64Array}
   */
  readInt64AtIndices(colIdx, indices) {
    if (indices.length === 0) return new BigInt64Array(0);
    const idxPtr = this.wasm.allocIndexBuffer(indices.length);
    if (!idxPtr) throw new Error("Failed to allocate index buffer");
    const outPtr = this.wasm.allocInt64Buffer(indices.length);
    if (!outPtr) {
      this.wasm.free(idxPtr, indices.length * 4);
      throw new Error("Failed to allocate output buffer");
    }
    try {
      new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
      const count = this.wasm.readInt64AtIndices(
        colIdx,
        idxPtr,
        indices.length,
        outPtr
      );
      const result = new BigInt64Array(count);
      const view = new BigInt64Array(this.memory.buffer, outPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(idxPtr, indices.length * 4);
      this.wasm.freeInt64Buffer(outPtr, indices.length);
    }
  }
  /**
   * Read float64 values at specific row indices.
   * @param {number} colIdx - Column index
   * @param {Uint32Array} indices - Row indices to read
   * @returns {Float64Array}
   */
  readFloat64AtIndices(colIdx, indices) {
    if (indices.length === 0) return new Float64Array(0);
    const idxPtr = this.wasm.allocIndexBuffer(indices.length);
    if (!idxPtr) throw new Error("Failed to allocate index buffer");
    const outPtr = this.wasm.allocFloat64Buffer(indices.length);
    if (!outPtr) {
      this.wasm.free(idxPtr, indices.length * 4);
      throw new Error("Failed to allocate output buffer");
    }
    try {
      new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
      const count = this.wasm.readFloat64AtIndices(
        colIdx,
        idxPtr,
        indices.length,
        outPtr
      );
      const result = new Float64Array(count);
      const view = new Float64Array(this.memory.buffer, outPtr, count);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(idxPtr, indices.length * 4);
      this.wasm.freeFloat64Buffer(outPtr, indices.length);
    }
  }
  // ========================================================================
  // Aggregation Methods
  // ========================================================================
  /**
   * Sum all values in an int64 column.
   * @param {number} colIdx - Column index
   * @returns {bigint}
   */
  sumInt64(colIdx) {
    return this.wasm.sumInt64Column(colIdx);
  }
  /**
   * Sum all values in a float64 column.
   * @param {number} colIdx - Column index
   * @returns {number}
   */
  sumFloat64(colIdx) {
    return this.wasm.sumFloat64Column(colIdx);
  }
  /**
   * Get minimum value in an int64 column.
   * @param {number} colIdx - Column index
   * @returns {bigint}
   */
  minInt64(colIdx) {
    return this.wasm.minInt64Column(colIdx);
  }
  /**
   * Get maximum value in an int64 column.
   * @param {number} colIdx - Column index
   * @returns {bigint}
   */
  maxInt64(colIdx) {
    return this.wasm.maxInt64Column(colIdx);
  }
  /**
   * Get average of a float64 column.
   * @param {number} colIdx - Column index
   * @returns {number}
   */
  avgFloat64(colIdx) {
    return this.wasm.avgFloat64Column(colIdx);
  }
  // ========================================================================
  // String Column Methods
  // ========================================================================
  /**
   * Debug: Get string column buffer info
   * @param {number} colIdx - Column index
   * @returns {{offsetsSize: number, dataSize: number}}
   */
  debugStringColInfo(colIdx) {
    const packed = this.wasm.debugStringColInfo(colIdx);
    return {
      offsetsSize: Number(BigInt(packed) >> 32n),
      dataSize: Number(BigInt(packed) & 0xFFFFFFFFn)
    };
  }
  /**
   * Debug: Get string read info for a specific row
   * @param {number} colIdx - Column index
   * @param {number} rowIdx - Row index
   * @returns {{strStart: number, strLen: number} | {error: string}}
   */
  debugReadStringInfo(colIdx, rowIdx) {
    const packed = this.wasm.debugReadStringInfo(colIdx, rowIdx);
    if ((packed & 0xFFFF0000n) === 0xDEAD0000n) {
      const errCode = Number(packed & 0xFFFFn);
      const errors = {
        1: "No file data",
        2: "No column entry",
        3: "Col meta out of bounds",
        4: "Not a string column",
        5: "Row out of bounds",
        6: "Invalid offset size"
      };
      return { error: errors[errCode] || `Unknown error ${errCode}` };
    }
    return {
      strStart: Number(BigInt(packed) >> 32n),
      strLen: Number(BigInt(packed) & 0xFFFFFFFFn)
    };
  }
  /**
   * Debug: Get data_start position for string column
   * @param {number} colIdx - Column index
   * @returns {{dataStart: number, fileLen: number}}
   */
  debugStringDataStart(colIdx) {
    const packed = this.wasm.debugStringDataStart(colIdx);
    return {
      dataStart: Number(BigInt(packed) >> 32n),
      fileLen: Number(BigInt(packed) & 0xFFFFFFFFn)
    };
  }
  /**
   * Get the number of strings in a column.
   * @param {number} colIdx - Column index
   * @returns {number}
   */
  getStringCount(colIdx) {
    return Number(this.wasm.getStringCount(colIdx));
  }
  /**
   * Read a single string at a specific row index.
   * @param {number} colIdx - Column index
   * @param {number} rowIdx - Row index
   * @returns {string}
   */
  readStringAt(colIdx, rowIdx) {
    const maxLen = 4096;
    const bufPtr = this.wasm.allocStringBuffer(maxLen);
    if (!bufPtr) throw new Error("Failed to allocate string buffer");
    try {
      const actualLen = this.wasm.readStringAt(colIdx, rowIdx, bufPtr, maxLen);
      if (actualLen === 0) return "";
      const bytes = new Uint8Array(this.memory.buffer, bufPtr, Math.min(actualLen, maxLen));
      return new TextDecoder().decode(bytes);
    } finally {
      this.wasm.free(bufPtr, maxLen);
    }
  }
  /**
   * Read all strings from a column.
   * @param {number} colIdx - Column index
   * @param {number} limit - Maximum number of strings to read
   * @returns {string[]}
   */
  readStringColumn(colIdx, limit = 1e3) {
    const count = Math.min(this.getStringCount(colIdx), limit);
    if (count === 0) return [];
    const results = [];
    for (let i = 0; i < count; i++) {
      results.push(this.readStringAt(colIdx, i));
    }
    return results;
  }
  /**
   * Read strings at specific row indices.
   * @param {number} colIdx - Column index
   * @param {Uint32Array} indices - Row indices to read
   * @returns {string[]}
   */
  readStringsAtIndices(colIdx, indices) {
    if (indices.length === 0) return [];
    const maxTotalLen = Math.min(indices.length * 256, 256 * 1024);
    const idxPtr = this.wasm.allocIndexBuffer(indices.length);
    if (!idxPtr) throw new Error("Failed to allocate index buffer");
    const strBufPtr = this.wasm.allocStringBuffer(maxTotalLen);
    if (!strBufPtr) {
      this.wasm.free(idxPtr, indices.length * 4);
      throw new Error("Failed to allocate string buffer");
    }
    const lenBufPtr = this.wasm.allocU32Buffer(indices.length);
    if (!lenBufPtr) {
      this.wasm.free(idxPtr, indices.length * 4);
      this.wasm.free(strBufPtr, maxTotalLen);
      throw new Error("Failed to allocate length buffer");
    }
    try {
      new Uint32Array(this.memory.buffer, idxPtr, indices.length).set(indices);
      const totalWritten = this.wasm.readStringsAtIndices(
        colIdx,
        idxPtr,
        indices.length,
        strBufPtr,
        maxTotalLen,
        lenBufPtr
      );
      const lengths = new Uint32Array(this.memory.buffer, lenBufPtr, indices.length);
      const results = [];
      let offset = 0;
      for (let i = 0; i < indices.length; i++) {
        const len = lengths[i];
        if (len > 0 && offset + len <= totalWritten) {
          const bytes = new Uint8Array(this.memory.buffer, strBufPtr + offset, len);
          results.push(new TextDecoder().decode(bytes));
          offset += len;
        } else {
          results.push("");
        }
      }
      return results;
    } finally {
      this.wasm.free(idxPtr, indices.length * 4);
      this.wasm.free(strBufPtr, maxTotalLen);
      this.wasm.free(lenBufPtr, indices.length * 4);
    }
  }
  // ========================================================================
  // Vector Column Support (for embeddings/semantic search)
  // ========================================================================
  /**
   * Get vector info for a column.
   * @param {number} colIdx - Column index
   * @returns {{rows: number, dimension: number}}
   */
  getVectorInfo(colIdx) {
    const packed = this.wasm.getVectorInfo(colIdx);
    return {
      rows: Number(BigInt(packed) >> 32n),
      dimension: Number(BigInt(packed) & 0xFFFFFFFFn)
    };
  }
  /**
   * Read a single vector at index.
   * @param {number} colIdx - Column index
   * @param {number} rowIdx - Row index
   * @returns {Float32Array}
   */
  readVectorAt(colIdx, rowIdx) {
    const info = this.getVectorInfo(colIdx);
    if (info.dimension === 0) return new Float32Array(0);
    const bufPtr = this.wasm.allocFloat32Buffer(info.dimension);
    if (!bufPtr) throw new Error("Failed to allocate vector buffer");
    try {
      const dim = this.wasm.readVectorAt(colIdx, rowIdx, bufPtr, info.dimension);
      const result = new Float32Array(dim);
      const view = new Float32Array(this.memory.buffer, bufPtr, dim);
      result.set(view);
      return result;
    } finally {
      this.wasm.free(bufPtr, info.dimension * 4);
    }
  }
  /**
   * Compute cosine similarity between two vectors.
   * @param {Float32Array} vecA
   * @param {Float32Array} vecB
   * @returns {number} Similarity score (-1 to 1)
   */
  cosineSimilarity(vecA, vecB) {
    if (vecA.length !== vecB.length) {
      throw new Error("Vector dimensions must match");
    }
    const ptrA = this.wasm.allocFloat32Buffer(vecA.length);
    const ptrB = this.wasm.allocFloat32Buffer(vecB.length);
    if (!ptrA || !ptrB) throw new Error("Failed to allocate buffers");
    try {
      new Float32Array(this.memory.buffer, ptrA, vecA.length).set(vecA);
      new Float32Array(this.memory.buffer, ptrB, vecB.length).set(vecB);
      return this.wasm.cosineSimilarity(ptrA, ptrB, vecA.length);
    } finally {
      this.wasm.free(ptrA, vecA.length * 4);
      this.wasm.free(ptrB, vecB.length * 4);
    }
  }
  /**
   * Batch cosine similarity using WASM SIMD.
   * Much faster than calling cosineSimilarity in a loop.
   * @param {Float32Array} queryVec - Query vector
   * @param {Float32Array[]} vectors - Array of vectors to compare
   * @param {boolean} normalized - Whether vectors are L2-normalized
   * @returns {Float32Array} - Similarity scores
   */
  batchCosineSimilarity(queryVec, vectors, normalized = true) {
    if (vectors.length === 0) return new Float32Array(0);
    const dim = queryVec.length;
    const numVectors = vectors.length;
    const queryPtr = this.wasm.allocFloat32Buffer(dim);
    const vectorsPtr = this.wasm.allocFloat32Buffer(numVectors * dim);
    const scoresPtr = this.wasm.allocFloat32Buffer(numVectors);
    if (!queryPtr || !vectorsPtr || !scoresPtr) {
      throw new Error("Failed to allocate WASM buffers");
    }
    try {
      new Float32Array(this.memory.buffer, queryPtr, dim).set(queryVec);
      const flatVectors = new Float32Array(this.memory.buffer, vectorsPtr, numVectors * dim);
      for (let i = 0; i < numVectors; i++) {
        flatVectors.set(vectors[i], i * dim);
      }
      this.wasm.batchCosineSimilarity(queryPtr, vectorsPtr, dim, numVectors, scoresPtr, normalized ? 1 : 0);
      const scores = new Float32Array(numVectors);
      scores.set(new Float32Array(this.memory.buffer, scoresPtr, numVectors));
      return scores;
    } finally {
      this.wasm.free(queryPtr, dim * 4);
      this.wasm.free(vectorsPtr, numVectors * dim * 4);
      this.wasm.free(scoresPtr, numVectors * 4);
    }
  }
  /**
   * Read all vectors from a column as array of Float32Arrays.
   * @param {number} colIdx - Column index
   * @returns {Float32Array[]} Array of vectors
   */
  readAllVectors(colIdx) {
    const info = this.getVectorInfo(colIdx);
    if (info.dimension === 0 || info.rows === 0) return [];
    const dim = info.dimension;
    const numRows = info.rows;
    const vectors = [];
    const bufPtr = this.wasm.allocFloat32Buffer(numRows * dim);
    if (!bufPtr) throw new Error("Failed to allocate vector buffer");
    try {
      if (this.wasm.readVectorColumn) {
        const count = this.wasm.readVectorColumn(colIdx, bufPtr, numRows * dim);
        const allData = new Float32Array(this.memory.buffer, bufPtr, count);
        for (let i = 0; i < numRows && i * dim < count; i++) {
          const vec = new Float32Array(dim);
          vec.set(allData.subarray(i * dim, (i + 1) * dim));
          vectors.push(vec);
        }
      } else {
        for (let i = 0; i < numRows; i++) {
          vectors.push(this.readVectorAt(colIdx, i));
        }
      }
      return vectors;
    } finally {
      this.wasm.free(bufPtr, numRows * dim * 4);
    }
  }
  /**
   * Find top-k most similar vectors to query.
   * Uses WebGPU if available, otherwise falls back to WASM SIMD.
   * GPU-accelerated top-K selection for large result sets.
   * @param {number} colIdx - Column index with vectors
   * @param {Float32Array} queryVec - Query vector
   * @param {number} topK - Number of results to return
   * @param {Function} onProgress - Progress callback (current, total)
   * @returns {Promise<{indices: Uint32Array, scores: Float32Array}>}
   */
  async vectorSearch(colIdx, queryVec, topK = 10, onProgress = null) {
    const dim = queryVec.length;
    const info = this.getVectorInfo(colIdx);
    const numRows = info.rows;
    const accelerator = getWebGPUAccelerator();
    if (accelerator.isAvailable()) {
      if (onProgress) onProgress(0, numRows);
      const allVectors2 = this.readAllVectors(colIdx);
      if (onProgress) onProgress(numRows, numRows);
      const scores2 = await accelerator.batchCosineSimilarity(queryVec, allVectors2, true);
      return await getGPUVectorSearch().topK(scores2, null, topK, true);
    }
    if (onProgress) onProgress(0, numRows);
    const allVectors = this.readAllVectors(colIdx);
    if (onProgress) onProgress(numRows, numRows);
    const scores = this.lanceql.batchCosineSimilarity(queryVec, allVectors, true);
    return await getGPUVectorSearch().topK(scores, null, topK, true);
  }
  // ========================================================================
  // DataFrame-like API
  // ========================================================================
  /**
   * Create a DataFrame-like query builder for this file.
   * @returns {DataFrame}
   */
  df() {
    return new DataFrame(this);
  }
};
// ========================================================================
// Query Methods
// ========================================================================
/**
 * Filter operator constants.
 */
__publicField(LanceFile2, "Op", {
  EQ: 0,
  // Equal
  NE: 1,
  // Not equal
  LT: 2,
  // Less than
  LE: 3,
  // Less than or equal
  GT: 4,
  // Greater than
  GE: 5
  // Greater than or equal
});

// src/client/index.js
init_remote_file();
init_remote_dataset();

// src/client/lance/lance-data-base.js
init_hot_tier_cache();
var ChunkedLanceReader2;
var LocalDatabase2;
var opfsStorage4;
var RemoteLanceFile3;
async function loadDeps() {
  if (!ChunkedLanceReader2) {
    const storageModule = await Promise.resolve().then(() => (init_lance_reader(), lance_reader_exports));
    ChunkedLanceReader2 = storageModule.ChunkedLanceReader;
  }
  if (!LocalDatabase2) {
    const dbModule = await Promise.resolve().then(() => (init_local_database(), local_database_exports));
    LocalDatabase2 = dbModule.LocalDatabase;
  }
  if (!opfsStorage4) {
    const opfsModule = await Promise.resolve().then(() => (init_opfs(), opfs_exports));
    opfsStorage4 = opfsModule.opfsStorage;
  }
  if (!RemoteLanceFile3) {
    const remoteModule = await Promise.resolve().then(() => (init_remote_file(), remote_file_exports));
    RemoteLanceFile3 = remoteModule.RemoteLanceFile;
  }
}
var LanceDataBase = class {
  constructor(type) {
    this.type = type;
  }
  // Abstract methods - must be implemented by subclasses
  async getSchema() {
    throw new Error("Not implemented");
  }
  async getRowCount() {
    throw new Error("Not implemented");
  }
  async readColumn(colIdx, start = 0, count = null) {
    throw new Error("Not implemented");
  }
  async *scan(options = {}) {
    throw new Error("Not implemented");
  }
  // Optional methods
  async insert(rows) {
    throw new Error("Write not supported for this source");
  }
  isCached() {
    return false;
  }
  async prefetch() {
  }
  async evict() {
  }
  async close() {
  }
};
var OPFSLanceData = class extends LanceDataBase {
  constructor(path, storage = null) {
    super("local");
    this.path = path;
    this.storage = storage;
    this.reader = null;
    this.database = null;
    this._isDatabase = false;
  }
  async open() {
    await loadDeps();
    this.storage = this.storage || opfsStorage4;
    const manifestPath = `${this.path}/__manifest__`;
    if (await this.storage.exists(manifestPath)) {
      this._isDatabase = true;
      this.database = new LocalDatabase2(this.path, this.storage);
      await this.database.open();
    } else {
      this.reader = await ChunkedLanceReader2.open(this.storage, this.path);
    }
    return this;
  }
  async getSchema() {
    if (this._isDatabase) {
      const tables = this.database.listTables();
      if (tables.length === 0) return [];
      return this.database.getSchema(tables[0]);
    }
    return Array.from({ length: this.reader.getNumColumns() }, (_, i) => ({
      name: `col_${i}`,
      type: "unknown"
    }));
  }
  async getRowCount() {
    if (this._isDatabase) {
      const tables = this.database.listTables();
      if (tables.length === 0) return 0;
      return this.database.count(tables[0]);
    }
    const meta = await this.reader.readColumnMetaRaw(0);
    return 0;
  }
  async readColumn(colIdx, start = 0, count = null) {
    if (this._isDatabase) {
      throw new Error("Use select() for database queries");
    }
    return this.reader.readColumnMetaRaw(colIdx);
  }
  async *scan(options = {}) {
    if (this._isDatabase) {
      const tables = this.database.listTables();
      if (tables.length === 0) return;
      yield* this.database.scan(tables[0], options);
    } else {
      throw new Error("scan() requires database, use readColumn() for single files");
    }
  }
  async insert(rows) {
    if (!this._isDatabase) {
      throw new Error("insert() requires database");
    }
    const tables = this.database.listTables();
    if (tables.length === 0) {
      throw new Error("No tables in database");
    }
    return this.database.insert(tables[0], rows);
  }
  isCached() {
    return true;
  }
  async close() {
    if (this.reader) {
      this.reader.close();
    }
    if (this.database) {
      await this.database.close();
    }
  }
};
var RemoteLanceData = class extends LanceDataBase {
  constructor(url) {
    super("remote");
    this.url = url;
    this.remoteFile = null;
    this.cachedPath = null;
  }
  async open() {
    await loadDeps();
    const cacheInfo = await getHotTierCache().getCacheInfo(this.url);
    if (cacheInfo && cacheInfo.complete) {
      this.type = "cached";
      this.cachedPath = cacheInfo.path;
    }
    if (RemoteLanceFile3) {
      this.remoteFile = await RemoteLanceFile3.open(null, this.url);
    }
    return this;
  }
  async getSchema() {
    if (!this.remoteFile) {
      return [];
    }
    const numCols = this.remoteFile.numColumns;
    const schema = [];
    for (let i = 0; i < numCols; i++) {
      const type = await this.remoteFile.getColumnType?.(i) || "unknown";
      schema.push({ name: `col_${i}`, type });
    }
    return schema;
  }
  async getRowCount() {
    if (!this.remoteFile) return 0;
    return this.remoteFile.getRowCount?.(0) || 0;
  }
  async readColumn(colIdx, start = 0, count = null) {
    if (!this.remoteFile) {
      throw new Error("Remote file not opened");
    }
    const type = await this.remoteFile.getColumnType?.(colIdx) || "unknown";
    if (type.includes("int64")) {
      return this.remoteFile.readInt64Column?.(colIdx, count);
    } else if (type.includes("float64")) {
      return this.remoteFile.readFloat64Column?.(colIdx, count);
    } else if (type.includes("string")) {
      return this.remoteFile.readStrings?.(colIdx, count);
    }
    throw new Error(`Unsupported column type: ${type}`);
  }
  async *scan(options = {}) {
    const batchSize = options.batchSize || 1e4;
    const rowCount = await this.getRowCount();
    const schema = await this.getSchema();
    for (let offset = 0; offset < rowCount; offset += batchSize) {
      const count = Math.min(batchSize, rowCount - offset);
      const batch = [];
      const columns = {};
      for (let i = 0; i < schema.length; i++) {
        columns[schema[i].name] = await this.readColumn(i, offset, count);
      }
      for (let r = 0; r < count; r++) {
        const row = {};
        for (const name of Object.keys(columns)) {
          row[name] = columns[name][r];
        }
        if (!options.where || options.where(row)) {
          batch.push(row);
        }
      }
      yield batch;
    }
  }
  isCached() {
    return this.type === "cached";
  }
  async prefetch() {
    const cache = getHotTierCache();
    await cache.cache(this.url);
    const cacheInfo = await cache.getCacheInfo(this.url);
    if (cacheInfo && cacheInfo.complete) {
      this.type = "cached";
      this.cachedPath = cacheInfo.path;
    }
  }
  async evict() {
    await getHotTierCache().evict(this.url);
    this.type = "remote";
    this.cachedPath = null;
  }
  async close() {
    if (this.remoteFile) {
      this.remoteFile.close();
    }
  }
};

// src/client/lance/lance-data-frame.js
var LanceFile3;
var DataFrame2 = class _DataFrame {
  constructor(file) {
    this.file = file;
    this._filterOps = [];
    this._selectCols = null;
    this._limitValue = null;
    this._isRemote = file._isRemote || file.baseUrl !== void 0;
  }
  /**
   * Filter rows where column matches condition.
   * Immer-style: returns new DataFrame, original unchanged.
   * @param {number} colIdx - Column index
   * @param {string} op - Operator: '=', '!=', '<', '<=', '>', '>='
   * @param {number|bigint|string} value - Value to compare
   * @param {string} type - 'int64', 'float64', or 'string'
   * @returns {DataFrame}
   */
  filter(colIdx, op, value, type = "int64") {
    const opMap = {
      "=": LanceFile3?.Op?.EQ ?? 0,
      "==": LanceFile3?.Op?.EQ ?? 0,
      "!=": LanceFile3?.Op?.NE ?? 1,
      "<>": LanceFile3?.Op?.NE ?? 1,
      "<": LanceFile3?.Op?.LT ?? 2,
      "<=": LanceFile3?.Op?.LE ?? 3,
      ">": LanceFile3?.Op?.GT ?? 4,
      ">=": LanceFile3?.Op?.GE ?? 5
    };
    const df = new _DataFrame(this.file);
    df._filterOps = [...this._filterOps, { colIdx, op: opMap[op], opStr: op, value, type }];
    df._selectCols = this._selectCols;
    df._limitValue = this._limitValue;
    df._isRemote = this._isRemote;
    return df;
  }
  /**
   * Select specific columns.
   * Immer-style: returns new DataFrame, original unchanged.
   * @param {...number} colIndices - Column indices to select
   * @returns {DataFrame}
   */
  select(...colIndices) {
    const cols = Array.isArray(colIndices[0]) ? colIndices[0] : colIndices;
    const df = new _DataFrame(this.file);
    df._filterOps = [...this._filterOps];
    df._selectCols = cols;
    df._limitValue = this._limitValue;
    df._isRemote = this._isRemote;
    return df;
  }
  /**
   * Limit number of results.
   * Immer-style: returns new DataFrame, original unchanged.
   * @param {number} n - Maximum rows
   * @returns {DataFrame}
   */
  limit(n) {
    const df = new _DataFrame(this.file);
    df._filterOps = [...this._filterOps];
    df._selectCols = this._selectCols;
    df._limitValue = n;
    df._isRemote = this._isRemote;
    return df;
  }
  /**
   * Generate SQL from DataFrame operations.
   * @returns {string}
   */
  toSQL() {
    const colNames = this.file.columnNames || this.file._schema?.map((s) => s.name) || Array.from({ length: this.file._numColumns || 6 }, (_, i) => `col_${i}`);
    let selectClause;
    if (this._selectCols && this._selectCols.length > 0) {
      selectClause = this._selectCols.map((i) => colNames[i] || `col_${i}`).join(", ");
    } else {
      selectClause = "*";
    }
    let whereClause = "";
    if (this._filterOps.length > 0) {
      const conditions = this._filterOps.map((f) => {
        const colName = colNames[f.colIdx] || `col_${f.colIdx}`;
        const val = f.type === "string" ? `'${f.value}'` : f.value;
        return `${colName} ${f.opStr} ${val}`;
      });
      whereClause = ` WHERE ${conditions.join(" AND ")}`;
    }
    const limitClause = this._limitValue ? ` LIMIT ${this._limitValue}` : "";
    return `SELECT ${selectClause} FROM dataset${whereClause}${limitClause}`;
  }
  /**
   * Execute the query and return row indices (sync, local only).
   * @returns {Uint32Array}
   */
  collectIndices() {
    if (this._isRemote) {
      throw new Error("collectIndices() is sync-only. Use collect() for remote datasets.");
    }
    let indices = null;
    for (const f of this._filterOps) {
      let newIndices;
      if (f.type === "int64") {
        newIndices = this.file.filterInt64(f.colIdx, f.op, f.value);
      } else {
        newIndices = this.file.filterFloat64(f.colIdx, f.op, f.value);
      }
      if (indices === null) {
        indices = newIndices;
      } else {
        const set = new Set(newIndices);
        indices = indices.filter((i) => set.has(i));
        indices = new Uint32Array(indices);
      }
    }
    if (indices === null) {
      const rowCount = Number(this.file.getRowCount(0));
      indices = new Uint32Array(rowCount);
      for (let i = 0; i < rowCount; i++) indices[i] = i;
    }
    if (this._limitValue !== null && indices.length > this._limitValue) {
      indices = indices.slice(0, this._limitValue);
    }
    return indices;
  }
  /**
   * Execute the query and return results.
   * Works with both local (sync) and remote (async) datasets.
   * @returns {Promise<{columns: Array[], columnNames: string[], total: number}>}
   */
  async collect() {
    if (this._isRemote) {
      const sql = this.toSQL();
      return await this.file.executeSQL(sql);
    }
    const indices = this.collectIndices();
    const cols = this._selectCols || Array.from({ length: this.file.numColumns }, (_, i) => i);
    const columns = [];
    const columnNames = [];
    for (const colIdx of cols) {
      columnNames.push(this.file.columnNames?.[colIdx] || `col_${colIdx}`);
      try {
        columns.push(Array.from(this.file.readInt64AtIndices(colIdx, indices)));
      } catch {
        try {
          columns.push(Array.from(this.file.readFloat64AtIndices(colIdx, indices)));
        } catch {
          columns.push(indices.map(() => null));
        }
      }
    }
    return {
      columns,
      columnNames,
      total: indices.length,
      _indices: indices
    };
  }
  /**
   * Count matching rows.
   * @returns {Promise<number>|number}
   */
  async count() {
    if (this._isRemote) {
      const result = await this.collect();
      return result.columns[0]?.length || 0;
    }
    return this.collectIndices().length;
  }
};

// src/client/lance/lance-data-render.js
var LanceQL2;
var RemoteLanceDataset3;
var _LanceData = class _LanceData {
  /**
   * Auto-initialize when DOM is ready.
   */
  static _autoInit() {
    if (_LanceData._initialized) return;
    _LanceData._initialized = true;
    _LanceData._registerBuiltinRenderers();
    _LanceData._injectTriggerStyles();
    _LanceData._setupObserver();
    _LanceData._processExisting();
  }
  /**
   * Get or load a dataset (cached).
   */
  static async _getDataset(url) {
    if (!url) {
      if (_LanceData._defaultDataset) return _LanceData._datasets.get(_LanceData._defaultDataset);
      throw new Error('No dataset URL. Add data-dataset="https://..." to your element.');
    }
    if (_LanceData._datasets.has(url)) {
      return _LanceData._datasets.get(url);
    }
    if (!LanceQL2) {
      const wasmModule = await Promise.resolve().then(() => (init_lanceql(), lanceql_exports));
      LanceQL2 = wasmModule.LanceQL;
    }
    if (!RemoteLanceDataset3) {
      const datasetModule = await Promise.resolve().then(() => (init_remote_dataset(), remote_dataset_exports));
      RemoteLanceDataset3 = datasetModule.RemoteLanceDataset;
    }
    if (!_LanceData._wasm) {
      const wasmUrl = document.querySelector("script[data-lanceql-wasm]")?.dataset.lanceqlWasm || "./lanceql.wasm";
      _LanceData._wasm = await LanceQL2.load(wasmUrl);
    }
    const dataset = await RemoteLanceDataset3.open(_LanceData._wasm, url);
    _LanceData._datasets.set(url, dataset);
    if (!_LanceData._defaultDataset) {
      _LanceData._defaultDataset = url;
    }
    return dataset;
  }
  /**
   * Manual init (optional).
   */
  static async init(options = {}) {
    _LanceData._autoInit();
    if (options.wasmUrl) {
      if (!LanceQL2) {
        const wasmModule = await Promise.resolve().then(() => (init_lanceql(), lanceql_exports));
        LanceQL2 = wasmModule.LanceQL;
      }
      _LanceData._wasm = await LanceQL2.load(options.wasmUrl);
    }
    if (options.dataset) {
      await _LanceData._getDataset(options.dataset);
    }
  }
  /**
   * Inject CSS that triggers JavaScript via animation events.
   */
  static _injectTriggerStyles() {
    if (document.getElementById("lance-data-triggers")) return;
    const style = document.createElement("style");
    style.id = "lance-data-triggers";
    style.textContent = `
            @keyframes lance-query-trigger {
                from { --lance-trigger: 0; }
                to { --lance-trigger: 1; }
            }

            .lance-data {
                animation: lance-query-trigger 0.001s;
            }

            .lance-data[data-refresh] {
                animation: lance-query-trigger 0.001s;
            }

            .lance-data[data-loading]::before {
                content: '';
                display: block;
                width: 20px;
                height: 20px;
                border: 2px solid #3b82f6;
                border-top-color: transparent;
                border-radius: 50%;
                animation: lance-spin 0.8s linear infinite;
            }

            @keyframes lance-spin {
                to { transform: rotate(360deg); }
            }

            .lance-data[data-error]::before {
                content: attr(data-error);
                color: #ef4444;
                font-size: 12px;
            }

            .lance-data table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }

            .lance-data th, .lance-data td {
                padding: 8px 12px;
                text-align: left;
                border-bottom: 1px solid #334155;
            }

            .lance-data th {
                background: #1e293b;
                font-weight: 500;
                color: #94a3b8;
            }

            .lance-data tr:hover td {
                background: rgba(59, 130, 246, 0.05);
            }

            .lance-data .lance-value {
                font-size: 24px;
                font-weight: 600;
                color: #3b82f6;
            }

            .lance-data .lance-list {
                list-style: none;
                padding: 0;
                margin: 0;
            }

            .lance-data .lance-list li {
                padding: 8px 0;
                border-bottom: 1px solid #334155;
            }

            .lance-data .lance-json {
                background: #0f172a;
                padding: 12px;
                border-radius: 8px;
                font-family: 'SF Mono', Monaco, monospace;
                font-size: 12px;
                white-space: pre-wrap;
                overflow-x: auto;
            }

            .lance-data .lance-images {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 16px;
            }

            .lance-data .lance-images .image-card {
                background: #1e293b;
                border-radius: 8px;
                overflow: hidden;
            }

            .lance-data .lance-images img {
                width: 100%;
                aspect-ratio: 1;
                object-fit: cover;
            }

            .lance-data .lance-images .image-meta {
                padding: 8px;
                font-size: 12px;
                color: #94a3b8;
            }
        `;
    document.head.appendChild(style);
  }
  /**
   * Set up MutationObserver for dynamic elements.
   */
  static _setupObserver() {
    if (_LanceData._observer) return;
    const hasLqAttrs = (el) => {
      return el.hasAttribute?.("lq-query") || el.hasAttribute?.("lq-src") || el.classList?.contains("lance-data");
    };
    _LanceData._observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            if (hasLqAttrs(node)) {
              _LanceData._processElement(node);
            }
            node.querySelectorAll?.("[lq-query], [lq-src], .lance-data")?.forEach((el) => {
              _LanceData._processElement(el);
            });
          }
        }
        if (mutation.type === "attributes" && hasLqAttrs(mutation.target)) {
          _LanceData._processElement(mutation.target);
        }
      }
    });
    _LanceData._observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["lq-query", "lq-src", "lq-render", "lq-bind", "data-query", "data-dataset", "data-render", "data-refresh"]
    });
    document.body.addEventListener("animationstart", (e) => {
      if (e.animationName === "lance-query-trigger" && hasLqAttrs(e.target)) {
        _LanceData._processElement(e.target);
      }
    });
  }
  /**
   * Process existing lance-data elements.
   */
  static _processExisting() {
    document.querySelectorAll("[lq-query], [lq-src], .lance-data").forEach((el) => {
      _LanceData._processElement(el);
    });
  }
  /**
   * Parse config from attributes.
   */
  static _parseConfig(el) {
    const getAttr = (lqName, dataName) => {
      return el.getAttribute(lqName) || el.dataset[dataName] || null;
    };
    return {
      dataset: getAttr("lq-src", "dataset"),
      query: getAttr("lq-query", "query"),
      render: getAttr("lq-render", "render") || "table",
      columns: (getAttr("lq-columns", "columns") || "").split(",").map((c) => c.trim()).filter(Boolean),
      bind: getAttr("lq-bind", "bind")
    };
  }
  /**
   * Render pre-computed results to an element.
   */
  static render(el, results, options = {}) {
    const element = typeof el === "string" ? document.querySelector(el) : el;
    if (!element) {
      console.error("[LanceData] Element not found:", el);
      return;
    }
    try {
      element.dispatchEvent(new CustomEvent("lq-start", {
        detail: { query: options.query || null }
      }));
      const renderType = options.render || element.dataset.render || "table";
      const renderer = _LanceData._renderers[renderType] || _LanceData._renderers.table;
      if (element.id) {
        _LanceData._queryCache.set(`rendered:${element.id}`, results);
      }
      element.innerHTML = renderer(results, { render: renderType, ...options });
      element.dispatchEvent(new CustomEvent("lq-complete", {
        detail: {
          query: options.query || null,
          columns: results.columns || [],
          total: results.total || results.rows?.length || 0
        }
      }));
    } catch (error) {
      element.dispatchEvent(new CustomEvent("lq-error", {
        detail: {
          query: options.query || null,
          message: error.message,
          error
        }
      }));
      throw error;
    }
  }
  /**
   * Extract dataset URL from SQL query.
   */
  static _extractUrlFromQuery(sql) {
    const match = sql.match(/read_lance\s*\(\s*['"]([^'"]+)['"]/i);
    return match ? match[1] : null;
  }
  /**
   * Process a single lance-data element.
   */
  static async _processElement(el) {
    if (el.dataset.processing === "true") return;
    el.dataset.processing = "true";
    let config;
    try {
      config = _LanceData._parseConfig(el);
      if (!config.query) {
        el.dataset.processing = "false";
        return;
      }
      if (config.bind) {
        _LanceData._setupBinding(el, config);
      }
      el.dataset.loading = "true";
      delete el.dataset.error;
      el.dispatchEvent(new CustomEvent("lq-start", {
        detail: { query: config.query }
      }));
      const datasetUrl = config.dataset || _LanceData._extractUrlFromQuery(config.query);
      const dataset = await _LanceData._getDataset(datasetUrl);
      const cacheKey = `${datasetUrl || "default"}:${config.query}`;
      let results = _LanceData._queryCache.get(cacheKey);
      if (!results) {
        results = await dataset.executeSQL(config.query);
        _LanceData._queryCache.set(cacheKey, results);
      }
      const renderer = _LanceData._renderers[config.render] || _LanceData._renderers.table;
      el.innerHTML = renderer(results, config);
      delete el.dataset.loading;
      el.dispatchEvent(new CustomEvent("lq-complete", {
        detail: {
          query: config.query,
          columns: results.columns || [],
          total: results.total || results.rows?.length || 0
        }
      }));
    } catch (error) {
      delete el.dataset.loading;
      el.dataset.error = error.message;
      console.error("[LanceData]", error);
      el.dispatchEvent(new CustomEvent("lq-error", {
        detail: {
          query: config?.query,
          message: error.message,
          error
        }
      }));
    } finally {
      el.dataset.processing = "false";
    }
  }
  /**
   * Set up reactive binding to an input element.
   */
  static _setupBinding(el, config) {
    const input = document.querySelector(config.bind);
    if (!input) return;
    const bindingKey = config.bind;
    if (_LanceData._bindings.has(bindingKey)) return;
    const handler = () => {
      const value = input.value;
      const newQuery = config.query.replace(/\$value/g, value);
      if (el.hasAttribute("lq-query")) {
        el.setAttribute("lq-query", newQuery);
      } else {
        el.dataset.query = newQuery;
      }
      el.dataset.refresh = Date.now();
    };
    input.addEventListener("input", handler);
    input.addEventListener("change", handler);
    _LanceData._bindings.set(bindingKey, { input, handler, element: el });
  }
  /**
   * Register a custom renderer.
   */
  static registerRenderer(name, fn) {
    _LanceData._renderers[name] = fn;
  }
  /**
   * Register built-in renderers.
   */
  static _registerBuiltinRenderers() {
    _LanceData._renderers.table = (results, config) => {
      if (!results) {
        return '<div class="lance-empty">No results</div>';
      }
      let columns, rows;
      if (results.columns && results.rows) {
        columns = config.columns?.length ? config.columns : results.columns.filter(
          (k) => !k.startsWith("_") && k !== "embedding"
        );
        rows = results.rows;
      } else if (Array.isArray(results)) {
        if (results.length === 0) {
          return '<div class="lance-empty">No results</div>';
        }
        columns = config.columns?.length ? config.columns : Object.keys(results[0]).filter(
          (k) => !k.startsWith("_") && k !== "embedding"
        );
        rows = results.map((row) => columns.map((col) => row[col]));
      } else {
        return '<div class="lance-empty">No results</div>';
      }
      if (rows.length === 0) {
        return '<div class="lance-empty">No results</div>';
      }
      let html = "<table><thead><tr>";
      for (const col of columns) {
        html += `<th>${_LanceData._escapeHtml(String(col))}</th>`;
      }
      html += "</tr></thead><tbody>";
      for (const row of rows) {
        html += "<tr>";
        for (let i = 0; i < columns.length; i++) {
          const value = row[i];
          html += `<td>${_LanceData._formatValue(value)}</td>`;
        }
        html += "</tr>";
      }
      html += "</tbody></table>";
      return html;
    };
    _LanceData._renderers.list = (results, config) => {
      if (!results || results.length === 0) {
        return '<div class="lance-empty">No results</div>';
      }
      const displayCol = config.columns?.[0] || Object.keys(results[0])[0];
      let html = '<ul class="lance-list">';
      for (const row of results) {
        html += `<li>${_LanceData._formatValue(row[displayCol])}</li>`;
      }
      html += "</ul>";
      return html;
    };
    _LanceData._renderers.value = (results, config) => {
      if (!results || results.length === 0) {
        return '<div class="lance-empty">-</div>';
      }
      const firstRow = results[0];
      const firstKey = Object.keys(firstRow)[0];
      const value = firstRow[firstKey];
      return `<div class="lance-value">${_LanceData._formatValue(value)}</div>`;
    };
    _LanceData._renderers.json = (results, config) => {
      return `<pre class="lance-json">${_LanceData._escapeHtml(JSON.stringify(results, null, 2))}</pre>`;
    };
    _LanceData._renderers.images = (results, config) => {
      if (!results || results.length === 0) {
        return '<div class="lance-empty">No images</div>';
      }
      let html = '<div class="lance-images">';
      for (const row of results) {
        const url = row.url || row.image_url || row.src;
        const text = row.text || row.caption || row.title || "";
        if (url) {
          html += `
                        <div class="image-card">
                            <img src="${_LanceData._escapeHtml(url)}" alt="${_LanceData._escapeHtml(text)}" loading="lazy">
                            ${text ? `<div class="image-meta">${_LanceData._escapeHtml(text.substring(0, 100))}</div>` : ""}
                        </div>
                    `;
        }
      }
      html += "</div>";
      return html;
    };
    _LanceData._renderers.count = (results, config) => {
      const count = results?.[0]?.count ?? results?.length ?? 0;
      return `<span class="lance-count">${count.toLocaleString()}</span>`;
    };
  }
  static _isImageUrl(str) {
    if (!str || typeof str !== "string") return false;
    const lower = str.toLowerCase();
    return (lower.startsWith("http://") || lower.startsWith("https://")) && (lower.includes(".jpg") || lower.includes(".jpeg") || lower.includes(".png") || lower.includes(".gif") || lower.includes(".webp") || lower.includes(".svg"));
  }
  static _isUrl(str) {
    if (!str || typeof str !== "string") return false;
    return str.startsWith("http://") || str.startsWith("https://");
  }
  static _formatValue(value) {
    if (value === null || value === void 0) return '<span class="null-value">NULL</span>';
    if (value === "") return '<span class="empty-value">(empty)</span>';
    if (typeof value === "number") {
      return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
    }
    if (Array.isArray(value)) {
      if (value.length > 10) return `<span class="vector-badge">[${value.length}d]</span>`;
      return `[${value.slice(0, 5).map((v) => _LanceData._formatValue(v)).join(", ")}${value.length > 5 ? "..." : ""}]`;
    }
    if (typeof value === "object") return JSON.stringify(value);
    const str = String(value);
    if (_LanceData._isImageUrl(str)) {
      const escaped = _LanceData._escapeHtml(str);
      const short = escaped.length > 40 ? escaped.substring(0, 40) + "..." : escaped;
      return `<div class="image-cell">
                <img src="${escaped}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
                <div class="image-placeholder" style="display:none"><svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg></div>
                <a href="${escaped}" target="_blank" class="url-text" title="${escaped}">${short}</a>
            </div>`;
    }
    if (_LanceData._isUrl(str)) {
      const escaped = _LanceData._escapeHtml(str);
      const short = escaped.length > 50 ? escaped.substring(0, 50) + "..." : escaped;
      return `<a href="${escaped}" target="_blank" class="url-link" title="${escaped}">${short}</a>`;
    }
    if (str.length > 100) return `<span title="${_LanceData._escapeHtml(str)}">${_LanceData._escapeHtml(str.substring(0, 100))}...</span>`;
    return _LanceData._escapeHtml(str);
  }
  static _escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
  static clearCache() {
    _LanceData._queryCache.clear();
  }
  static refresh() {
    _LanceData._queryCache.clear();
    document.querySelectorAll(".lance-data").forEach((el) => {
      el.setAttribute("data-refresh", Date.now());
    });
  }
  static destroy() {
    if (_LanceData._observer) {
      _LanceData._observer.disconnect();
      _LanceData._observer = null;
    }
    for (const [key, binding] of _LanceData._bindings) {
      binding.input.removeEventListener("input", binding.handler);
      binding.input.removeEventListener("change", binding.handler);
    }
    _LanceData._bindings.clear();
    document.getElementById("lance-data-triggers")?.remove();
    _LanceData._instance = null;
    _LanceData._dataset = null;
    _LanceData._queryCache.clear();
  }
};
__publicField(_LanceData, "_initialized", false);
__publicField(_LanceData, "_observer", null);
__publicField(_LanceData, "_wasm", null);
__publicField(_LanceData, "_datasets", /* @__PURE__ */ new Map());
__publicField(_LanceData, "_renderers", {});
__publicField(_LanceData, "_bindings", /* @__PURE__ */ new Map());
__publicField(_LanceData, "_queryCache", /* @__PURE__ */ new Map());
__publicField(_LanceData, "_defaultDataset", null);
var LanceData = _LanceData;
if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => LanceData._autoInit());
  } else {
    LanceData._autoInit();
  }
}

// src/client/lance/lance-data-sqljs.js
init_accelerator();
var LocalDatabase3;
var opfsStorage5;
var gpuAggregator;
var gpuJoiner;
var gpuGrouper;
var gpuVectorSearch;
var Statement = class {
  constructor(db, sql) {
    this.db = db;
    this.sql = sql;
    this.params = null;
    this.results = null;
    this.resultIndex = 0;
    this.done = false;
  }
  bind(params) {
    this.params = params;
    this.results = null;
    this.resultIndex = 0;
    this.done = false;
    return true;
  }
  step() {
    if (this.done) return false;
    if (this.results === null) {
      const execResult = this.db.exec(this.sql, this.params);
      if (execResult.length === 0 || execResult[0].values.length === 0) {
        this.done = true;
        return false;
      }
      this.results = execResult[0];
      this.resultIndex = 0;
    }
    if (this.resultIndex >= this.results.values.length) {
      this.done = true;
      return false;
    }
    return true;
  }
  get() {
    if (!this.results || this.resultIndex >= this.results.values.length) {
      return [];
    }
    const row = this.results.values[this.resultIndex];
    this.resultIndex++;
    return row;
  }
  getAsObject(params) {
    if (!this.results || this.resultIndex >= this.results.values.length) {
      return {};
    }
    const row = this.results.values[this.resultIndex];
    this.resultIndex++;
    const obj = {};
    this.results.columns.forEach((col, i) => {
      obj[col] = row[i];
    });
    return obj;
  }
  getColumnNames() {
    return this.results?.columns || [];
  }
  reset() {
    this.results = null;
    this.resultIndex = 0;
    this.done = false;
    return true;
  }
  free() {
    this.results = null;
    this.params = null;
    return true;
  }
  freemem() {
    return this.free();
  }
};
var Database = class {
  constructor(nameOrData, storage = null) {
    if (nameOrData instanceof Uint8Array) {
      this._inMemory = true;
      this._data = nameOrData;
      this._name = ":memory:";
      this._db = null;
    } else {
      this._inMemory = false;
      this._name = nameOrData || "default";
      this._storage = storage;
      this._db = null;
      this._pendingInit = true;
    }
    this._open = false;
    this._rowsModified = 0;
  }
  async _ensureOpen() {
    if (!this._open) {
      if (!LocalDatabase3) {
        const dbModule = await Promise.resolve().then(() => (init_local_database(), local_database_exports));
        LocalDatabase3 = dbModule.LocalDatabase;
      }
      if (!opfsStorage5) {
        const opfsModule = await Promise.resolve().then(() => (init_opfs(), opfs_exports));
        opfsStorage5 = opfsModule.opfsStorage;
      }
      if (!this._db && !this._inMemory) {
        this._db = new LocalDatabase3(this._name, this._storage || opfsStorage5);
      }
      if (this._db) {
        await this._db.open();
      }
      this._open = true;
    }
  }
  exec(sql, params) {
    return this._execAsync(sql, params);
  }
  async _execAsync(sql, params) {
    await this._ensureOpen();
    let processedSql = sql;
    if (params) {
      if (Array.isArray(params)) {
        let paramIndex = 0;
        processedSql = sql.replace(/\?/g, () => {
          const val = params[paramIndex++];
          return this._formatValue(val);
        });
      } else if (typeof params === "object") {
        for (const [key, val] of Object.entries(params)) {
          const pattern = new RegExp(`[:$@]${key}\\b`, "g");
          processedSql = processedSql.replace(pattern, this._formatValue(val));
        }
      }
    }
    const statements = processedSql.split(";").map((s) => s.trim()).filter((s) => s.length > 0);
    const results = [];
    for (const stmt of statements) {
      try {
        const lexer = new SQLLexer(stmt);
        const tokens = lexer.tokenize();
        const parser = new SQLParser(tokens);
        const ast = parser.parse();
        if (ast.type === "SELECT") {
          const rows = await this._db._executeAST(ast);
          if (rows && rows.length > 0) {
            const columns = Object.keys(rows[0]);
            const values = rows.map((row) => columns.map((c) => row[c]));
            results.push({ columns, values });
          }
          this._rowsModified = 0;
        } else {
          const result = await this._db._executeAST(ast);
          this._rowsModified = result?.inserted || result?.updated || result?.deleted || 0;
        }
      } catch (e) {
        throw new Error(`SQL error: ${e.message}
Statement: ${stmt}`);
      }
    }
    return results;
  }
  run(sql, params) {
    return this._runAsync(sql, params);
  }
  async _runAsync(sql, params) {
    await this.exec(sql, params);
    return this;
  }
  prepare(sql, params) {
    const stmt = new Statement(this, sql);
    if (params) {
      stmt.bind(params);
    }
    return stmt;
  }
  each(sql, params, callback, done) {
    this._eachAsync(sql, params, callback, done);
    return this;
  }
  async _eachAsync(sql, params, callback, done) {
    try {
      const results = await this.exec(sql, params);
      if (results.length > 0) {
        const { columns, values } = results[0];
        for (const row of values) {
          const obj = {};
          columns.forEach((col, i) => {
            obj[col] = row[i];
          });
          callback(obj);
        }
      }
      if (done) done();
    } catch (e) {
      if (done) done(e);
      else throw e;
    }
  }
  getRowsModified() {
    return this._rowsModified;
  }
  async export() {
    if (this._inMemory && this._data) {
      return this._data;
    }
    await this._ensureOpen();
    const exportData = {
      version: this._db?.version,
      tables: {}
    };
    if (this._db) {
      for (const tableName of this._db.listTables()) {
        const table = this._db.getTable(tableName);
        const rows = await this._db.select(tableName, {});
        exportData.tables[tableName] = {
          schema: table?.schema,
          rows
        };
      }
    }
    return new TextEncoder().encode(JSON.stringify(exportData));
  }
  close() {
    this._open = false;
    this._db = null;
  }
  create_function(name, func) {
    console.warn(`[LanceQL] create_function('${name}') not yet implemented`);
    return this;
  }
  create_aggregate(name, funcs) {
    console.warn(`[LanceQL] create_aggregate('${name}') not yet implemented`);
    return this;
  }
  _formatValue(val) {
    if (val === null || val === void 0) {
      return "NULL";
    }
    if (typeof val === "string") {
      return `'${val.replace(/'/g, "''")}'`;
    }
    if (typeof val === "number") {
      return String(val);
    }
    if (typeof val === "boolean") {
      return val ? "TRUE" : "FALSE";
    }
    if (Array.isArray(val)) {
      return `'[${val.join(",")}]'`;
    }
    return String(val);
  }
};
async function initSqlJs(config = {}) {
  try {
    if (!opfsStorage5) {
      const opfsModule = await Promise.resolve().then(() => (init_opfs(), opfs_exports));
      opfsStorage5 = opfsModule.opfsStorage;
    }
    await opfsStorage5.open();
  } catch (e) {
    console.warn("[LanceQL] OPFS not available:", e.message);
  }
  try {
    const accelerator = getWebGPUAccelerator();
    await accelerator.init();
    if (!gpuAggregator) gpuAggregator = new GPUAggregator();
    if (!gpuJoiner) gpuJoiner = new GPUJoiner();
    if (!gpuGrouper) gpuGrouper = new GPUGrouper();
    if (!gpuVectorSearch) gpuVectorSearch = new GPUVectorSearch();
    await gpuAggregator.init();
    await gpuJoiner.init();
    await getGPUSorter().init();
    await gpuGrouper.init();
    await gpuVectorSearch.init();
  } catch (e) {
    console.warn("[LanceQL] WebGPU not available:", e.message);
  }
  return {
    Database,
    Statement
  };
}

// src/client/index.js
init_ivf_index();

// src/client/database/lance-db-joins.js
init_local_database();
init_opfs();
async function executeJoin(db, ast) {
  const leftTableName = ast.from.name || ast.from.table;
  const leftAlias = ast.from.alias || leftTableName;
  if (ast.from.alias) {
    db.aliases.set(ast.from.alias, leftTableName);
  }
  let currentResult = null;
  let currentAlias = leftAlias;
  let currentTableName = leftTableName;
  let leftDataset = db.getTable(leftTableName);
  for (let i = 0; i < ast.joins.length; i++) {
    const join = ast.joins[i];
    const rightTableName = join.table.name || join.table.table;
    const rightAlias = join.alias || rightTableName;
    if (join.alias) {
      db.aliases.set(join.alias, rightTableName);
    }
    const rightDataset = db.getTable(rightTableName);
    const singleJoinAst = {
      ...ast,
      joins: [join],
      limit: i === ast.joins.length - 1 ? ast.limit : void 0,
      columns: i === ast.joins.length - 1 ? ast.columns : [{ type: "column", column: "*" }]
    };
    if (currentResult === null) {
      currentResult = await hashJoin(
        db,
        leftDataset,
        rightDataset,
        singleJoinAst,
        { leftAlias: currentAlias, rightAlias, leftTableName: currentTableName, rightTableName }
      );
    } else {
      currentResult = await hashJoinWithInMemoryLeft(
        db,
        currentResult,
        rightDataset,
        singleJoinAst,
        { leftAlias: currentAlias, rightAlias, leftTableName: currentTableName, rightTableName }
      );
    }
    currentAlias = `${currentAlias}_${rightAlias}`;
    currentTableName = `(${currentTableName} JOIN ${rightTableName})`;
  }
  return currentResult;
}
async function hashJoin(db, leftDataset, rightDataset, ast, context) {
  const { leftAlias, rightAlias, leftTableName, rightTableName } = context;
  const join = ast.joins[0];
  const joinType = join.type || "INNER";
  const joinCondition = join.on;
  if (joinType !== "CROSS") {
    if (!joinCondition || joinCondition.type !== "binary" || joinCondition.op !== "=") {
      throw new Error("JOIN ON condition must be an equality (e.g., table1.col1 = table2.col2)");
    }
  }
  let leftKey, rightKey, leftSQL, rightSQL, plan;
  if (joinType === "CROSS") {
    leftKey = null;
    rightKey = null;
    leftSQL = `SELECT * FROM ${leftTableName}`;
    rightSQL = `SELECT * FROM ${rightTableName}`;
  } else {
    const planner = new QueryPlanner();
    plan = planner.plan(ast, context);
    leftKey = plan.join.leftKey;
    rightKey = plan.join.rightKey;
    const leftColumns = plan.leftScan.columns;
    const rightColumns = plan.rightScan.columns;
    const leftFilters = plan.leftScan.filters;
    const rightFilters = plan.rightScan.filters;
    const leftColsWithKey = leftColumns.includes("*") ? ["*"] : [.../* @__PURE__ */ new Set([leftKey, ...leftColumns])];
    let leftWhereClause = "";
    if (leftFilters.length > 0) {
      leftWhereClause = ` WHERE ${leftFilters.map((f) => filterToSQL(f)).join(" AND ")}`;
    }
    leftSQL = `SELECT ${leftColsWithKey.join(", ")} FROM ${leftTableName}${leftWhereClause}`;
    const rightColsWithKey = rightColumns.includes("*") ? ["*"] : [.../* @__PURE__ */ new Set([rightKey, ...rightColumns])];
    let rightWhereClause = "";
    if (rightFilters.length > 0) {
      rightWhereClause = ` WHERE ${rightFilters.map((f) => filterToSQL(f)).join(" AND ")}`;
    }
    rightSQL = `SELECT ${rightColsWithKey.join(", ")} FROM ${rightTableName}${rightWhereClause}`;
  }
  await opfsStorage2.open();
  const joinExecutor = new OPFSJoinExecutor(opfsStorage2);
  const leftExecutor = new SQLExecutor(leftDataset);
  const rightExecutor = new SQLExecutor(rightDataset);
  const leftStream = leftExecutor.executeStream(leftSQL);
  const leftMeta = await joinExecutor._partitionToOPFS(leftStream, leftKey, "left", true);
  let optimizedRightSQL = rightSQL;
  const maxKeysForInClause = 1e4;
  if (leftMeta.collectedKeys && leftMeta.collectedKeys.size > 0 && leftMeta.collectedKeys.size <= maxKeysForInClause) {
    const inClause = buildInClause(rightKey, leftMeta.collectedKeys);
    optimizedRightSQL = appendWhereClause(rightSQL, inClause);
  }
  const rightStream = rightExecutor.executeStream(optimizedRightSQL);
  const results = [];
  let resultColumns = null;
  try {
    for await (const chunk of joinExecutor.executeHashJoin(
      null,
      rightStream,
      leftKey,
      rightKey,
      {
        limit: ast.limit || Infinity,
        leftAlias,
        rightAlias,
        joinType: join.type || "INNER",
        prePartitionedLeft: leftMeta
      }
    )) {
      if (!resultColumns) {
        resultColumns = chunk.columns;
      }
      results.push(...chunk.rows);
      if (ast.limit && results.length >= ast.limit) {
        break;
      }
    }
  } catch (e) {
    console.error("[LanceDatabase] OPFS join failed:", e);
    throw e;
  }
  const stats = joinExecutor.getStats();
  if (!resultColumns || results.length === 0) {
    return { columns: [], rows: [], total: 0, opfsStats: stats };
  }
  const projectedResults = applyProjection(
    results,
    resultColumns,
    plan.projection,
    leftAlias,
    rightAlias
  );
  const limitedResults = ast.limit ? projectedResults.rows.slice(0, ast.limit) : projectedResults.rows;
  return {
    columns: projectedResults.columns,
    rows: limitedResults,
    total: limitedResults.length,
    opfsStats: stats
  };
}
async function hashJoinWithInMemoryLeft(db, leftResult, rightDataset, ast, context) {
  const { leftAlias, rightAlias, leftTableName, rightTableName } = context;
  const join = ast.joins[0];
  const joinType = join.type || "INNER";
  const joinCondition = join.on;
  if (joinType !== "CROSS") {
    if (!joinCondition || joinCondition.type !== "binary" || joinCondition.op !== "=") {
      throw new Error("JOIN ON condition must be an equality (e.g., table1.col1 = table2.col2)");
    }
  }
  let leftKey, rightKey;
  if (joinType === "CROSS") {
    leftKey = null;
    rightKey = null;
  } else {
    const leftExpr = joinCondition.left;
    const rightExpr = joinCondition.right;
    const leftCol = leftExpr.column;
    const rightCol = rightExpr.column;
    const leftColsSet = new Set(leftResult.columns.map((c) => {
      const parts = c.split(".");
      return parts[parts.length - 1];
    }));
    if (leftColsSet.has(leftCol)) {
      leftKey = leftCol;
      rightKey = rightCol;
    } else {
      leftKey = rightCol;
      rightKey = leftCol;
    }
  }
  let rightSQL = `SELECT * FROM ${rightTableName}`;
  const maxKeysForInClause = 1e3;
  if (leftKey && joinType !== "CROSS") {
    const leftKeyIndex2 = findColumnIndex(leftResult.columns, leftKey);
    if (leftKeyIndex2 !== -1) {
      const leftKeys = /* @__PURE__ */ new Set();
      for (const row of leftResult.rows) {
        const key = row[leftKeyIndex2];
        if (key !== null && key !== void 0) {
          leftKeys.add(key);
        }
      }
      if (leftKeys.size > 0 && leftKeys.size <= maxKeysForInClause) {
        const inClause = buildInClause(rightKey, leftKeys);
        rightSQL = appendWhereClause(rightSQL, inClause);
      }
    }
  }
  const rightExecutor = new SQLExecutor(rightDataset);
  const rightResult = await rightExecutor.execute(new SQLParser(new SQLLexer(rightSQL).tokenize()).parse());
  const leftKeyIndex = leftKey ? findColumnIndex(leftResult.columns, leftKey) : -1;
  const rightKeyIndex = rightKey ? findColumnIndex(rightResult.columns, rightKey) : -1;
  const resultColumns = [
    ...leftResult.columns,
    ...rightResult.columns.map((c) => `${rightAlias}.${c}`)
  ];
  const results = [];
  const rightNulls = new Array(rightResult.columns.length).fill(null);
  const leftNulls = new Array(leftResult.columns.length).fill(null);
  if (joinType === "CROSS") {
    for (const leftRow of leftResult.rows) {
      for (const rightRow of rightResult.rows) {
        results.push([...leftRow, ...rightRow]);
        if (ast.limit && results.length >= ast.limit) break;
      }
      if (ast.limit && results.length >= ast.limit) break;
    }
  } else {
    const rightHash = /* @__PURE__ */ new Map();
    for (const row of rightResult.rows) {
      const key = row[rightKeyIndex];
      if (key !== null && key !== void 0) {
        if (!rightHash.has(key)) rightHash.set(key, []);
        rightHash.get(key).push(row);
      }
    }
    const matchedRightRows = joinType === "FULL" || joinType === "RIGHT" ? /* @__PURE__ */ new Set() : null;
    for (const leftRow of leftResult.rows) {
      const key = leftRow[leftKeyIndex];
      const rightMatches = rightHash.get(key) || [];
      if (rightMatches.length > 0) {
        for (const rightRow of rightMatches) {
          results.push([...leftRow, ...rightRow]);
          if (matchedRightRows) {
            matchedRightRows.add(rightResult.rows.indexOf(rightRow));
          }
          if (ast.limit && results.length >= ast.limit) break;
        }
      } else if (joinType === "LEFT" || joinType === "FULL") {
        results.push([...leftRow, ...rightNulls]);
      }
      if (ast.limit && results.length >= ast.limit) break;
    }
    if ((joinType === "RIGHT" || joinType === "FULL") && matchedRightRows) {
      for (let i = 0; i < rightResult.rows.length; i++) {
        if (!matchedRightRows.has(i)) {
          results.push([...leftNulls, ...rightResult.rows[i]]);
          if (ast.limit && results.length >= ast.limit) break;
        }
      }
    }
  }
  const limitedResults = ast.limit ? results.slice(0, ast.limit) : results;
  return {
    columns: resultColumns,
    rows: limitedResults,
    total: limitedResults.length
  };
}
function findColumnIndex(columns, columnName) {
  let idx = columns.indexOf(columnName);
  if (idx !== -1) return idx;
  for (let i = 0; i < columns.length; i++) {
    const col = columns[i];
    const parts = col.split(".");
    if (parts[parts.length - 1] === columnName) {
      return i;
    }
  }
  return -1;
}
function filterToSQL(expr) {
  if (!expr) return "";
  if (expr.type === "binary") {
    const left = filterToSQL(expr.left);
    const right = filterToSQL(expr.right);
    return `${left} ${expr.op} ${right}`;
  } else if (expr.type === "column") {
    return expr.column;
  } else if (expr.type === "literal") {
    if (typeof expr.value === "string") {
      const escaped = expr.value.replace(/'/g, "''");
      return `'${escaped}'`;
    }
    if (expr.value === null) return "NULL";
    return String(expr.value);
  } else if (expr.type === "call") {
    const args = (expr.args || []).map((a) => filterToSQL(a)).join(", ");
    return `${expr.name}(${args})`;
  } else if (expr.type === "in") {
    const col = filterToSQL(expr.expr);
    const vals = expr.values.map((v) => filterToSQL(v)).join(", ");
    return `${col} IN (${vals})`;
  } else if (expr.type === "between") {
    const col = filterToSQL(expr.expr);
    const low = filterToSQL(expr.low);
    const high = filterToSQL(expr.high);
    return `${col} BETWEEN ${low} AND ${high}`;
  } else if (expr.type === "like") {
    const col = filterToSQL(expr.expr);
    const pattern = filterToSQL(expr.pattern);
    return `${col} LIKE ${pattern}`;
  } else if (expr.type === "unary") {
    const operand = filterToSQL(expr.operand);
    if (expr.op === "NOT") return `NOT ${operand}`;
    return `${expr.op}${operand}`;
  }
  return "";
}
function applyProjection(rows, allColumns, projection, leftAlias, rightAlias) {
  if (projection.includes("*")) {
    return { columns: allColumns, rows };
  }
  const projectedColumns = [];
  const columnIndices = [];
  for (const col of projection) {
    if (col === "*") continue;
    let idx = -1;
    let outputColName = col.column;
    if (col.table) {
      const qualifiedName = `${col.table}.${col.column}`;
      idx = allColumns.indexOf(qualifiedName);
      outputColName = qualifiedName;
    }
    if (idx === -1) {
      idx = allColumns.findIndex((c) => c === col.column || c.endsWith(`.${col.column}`));
      if (idx !== -1) {
        outputColName = allColumns[idx];
      }
    }
    if (idx !== -1) {
      projectedColumns.push(col.alias || outputColName);
      columnIndices.push(idx);
    }
  }
  const projectedRows = rows.map(
    (row) => columnIndices.map((idx) => row[idx])
  );
  return { columns: projectedColumns, rows: projectedRows };
}
function buildInClause(column, keys) {
  const values = Array.from(keys).map((k) => {
    if (typeof k === "string") {
      return `'${k.replace(/'/g, "''")}'`;
    }
    if (k === null) return "NULL";
    return String(k);
  }).join(", ");
  return `${column} IN (${values})`;
}
function appendWhereClause(sql, clause) {
  const upperSQL = sql.toUpperCase();
  if (upperSQL.includes("WHERE")) {
    return sql.replace(/WHERE\s+/i, `WHERE ${clause} AND `);
  }
  return sql.replace(/FROM\s+(\w+)(\s+\w+)?/i, (match) => `${match} WHERE ${clause}`);
}

// src/client/database/lance-db-memory.js
var MemoryTable = class {
  constructor(name, schema) {
    this.name = name;
    this.schema = schema;
    this.columns = schema.map((c) => c.name);
    this.rows = [];
    this._columnIndex = /* @__PURE__ */ new Map();
    this.columns.forEach((col, idx) => {
      this._columnIndex.set(col.toLowerCase(), idx);
    });
  }
  /**
   * Convert to in-memory data format for executor
   */
  toInMemoryData() {
    const columnData = {};
    this.columns.forEach((col, idx) => {
      columnData[col.toLowerCase()] = this.rows.map((row) => row[idx]);
    });
    return { columnData, columnNames: this.columns };
  }
};
function executeCreateTable(db, ast) {
  const tableName = (ast.table || ast.name || "").toLowerCase();
  if (!tableName) {
    throw new Error("CREATE TABLE requires a table name");
  }
  if (db.memoryTables.has(tableName) || db.tables.has(tableName)) {
    if (ast.ifNotExists) {
      return { success: true, existed: true, table: tableName };
    }
    throw new Error(`Table '${tableName}' already exists`);
  }
  const schema = (ast.columns || []).map((col) => ({
    name: col.name,
    dataType: col.dataType || col.type || "TEXT",
    primaryKey: col.primaryKey || false
  }));
  if (schema.length === 0) {
    throw new Error("CREATE TABLE requires at least one column");
  }
  const table = new MemoryTable(tableName, schema);
  db.memoryTables.set(tableName, table);
  return {
    success: true,
    table: tableName,
    columns: schema.map((c) => c.name)
  };
}
function executeDropTable(db, ast) {
  const tableName = (ast.table || ast.name || "").toLowerCase();
  if (!db.memoryTables.has(tableName)) {
    if (ast.ifExists) {
      return { success: true, existed: false, table: tableName };
    }
    throw new Error(`Memory table '${tableName}' not found`);
  }
  db.memoryTables.delete(tableName);
  return { success: true, table: tableName };
}
function executeInsert(db, ast) {
  const tableName = (ast.table || "").toLowerCase();
  const table = db.memoryTables.get(tableName);
  if (!table) {
    throw new Error(`Memory table '${tableName}' not found. Use CREATE TABLE first.`);
  }
  const insertCols = ast.columns || table.columns;
  let inserted = 0;
  for (const astRow of ast.rows || ast.values || []) {
    const row = new Array(table.columns.length).fill(null);
    insertCols.forEach((colName, i) => {
      const colIdx = table._columnIndex.get(
        (typeof colName === "string" ? colName : colName.name || colName).toLowerCase()
      );
      if (colIdx !== void 0 && i < astRow.length) {
        const val = astRow[i];
        row[colIdx] = val?.value !== void 0 ? val.value : val;
      }
    });
    table.rows.push(row);
    inserted++;
  }
  return {
    success: true,
    inserted,
    total: table.rows.length
  };
}
function executeUpdate(db, ast) {
  const tableName = (ast.table || "").toLowerCase();
  const table = db.memoryTables.get(tableName);
  if (!table) {
    throw new Error(`Memory table '${tableName}' not found`);
  }
  const columnData = {};
  table.columns.forEach((col, idx) => {
    columnData[col.toLowerCase()] = table.rows.map((row) => row[idx]);
  });
  const executor = new SQLExecutor({ columnNames: table.columns });
  let updated = 0;
  for (let i = 0; i < table.rows.length; i++) {
    const matches = !ast.where || executor._evaluateInMemoryExpr(ast.where, columnData, i);
    if (matches) {
      for (const assignment of ast.assignments || ast.set || []) {
        const colName = (assignment.column || assignment.name || "").toLowerCase();
        const colIdx = table._columnIndex.get(colName);
        if (colIdx !== void 0) {
          const val = assignment.value;
          table.rows[i][colIdx] = val?.value !== void 0 ? val.value : val;
        }
      }
      updated++;
    }
  }
  return { success: true, updated };
}
function executeDelete(db, ast) {
  const tableName = (ast.table || "").toLowerCase();
  const table = db.memoryTables.get(tableName);
  if (!table) {
    throw new Error(`Memory table '${tableName}' not found`);
  }
  const originalCount = table.rows.length;
  if (ast.where) {
    const columnData = {};
    table.columns.forEach((col, idx) => {
      columnData[col.toLowerCase()] = table.rows.map((row) => row[idx]);
    });
    const executor = new SQLExecutor({ columnNames: table.columns });
    table.rows = table.rows.filter(
      (_, i) => !executor._evaluateInMemoryExpr(ast.where, columnData, i)
    );
  } else {
    table.rows = [];
  }
  return {
    success: true,
    deleted: originalCount - table.rows.length,
    remaining: table.rows.length
  };
}

// src/client/database/lance-db-optimizer.js
function getCachedPlan(db, sql) {
  const normalized = normalizeSQL(sql);
  const cached = db._planCache.get(normalized);
  if (cached) {
    cached.hits++;
    cached.lastUsed = Date.now();
    return cached.plan;
  }
  return null;
}
function setCachedPlan(db, sql, plan) {
  const normalized = normalizeSQL(sql);
  if (db._planCache.size >= db._planCacheMaxSize) {
    let oldest = null;
    let oldestTime = Infinity;
    for (const [key, value] of db._planCache) {
      if (value.lastUsed < oldestTime) {
        oldestTime = value.lastUsed;
        oldest = key;
      }
    }
    if (oldest) db._planCache.delete(oldest);
  }
  db._planCache.set(normalized, {
    plan,
    hits: 0,
    lastUsed: Date.now(),
    created: Date.now()
  });
}
function normalizeSQL(sql) {
  return sql.trim().replace(/\s+/g, " ").toLowerCase();
}
function getPlanCacheStats(db) {
  let totalHits = 0;
  for (const v of db._planCache.values()) {
    totalHits += v.hits;
  }
  return {
    size: db._planCache.size,
    maxSize: db._planCacheMaxSize,
    totalHits
  };
}
function optimizeExpr(expr) {
  if (!expr) return expr;
  if (expr.left) expr.left = optimizeExpr(expr.left);
  if (expr.right) expr.right = optimizeExpr(expr.right);
  if (expr.operand) expr.operand = optimizeExpr(expr.operand);
  if (expr.args) expr.args = expr.args.map((a) => optimizeExpr(a));
  const op = expr.op || expr.operator;
  if (expr.type === "binary" && isConstantExpr(expr.left) && isConstantExpr(expr.right)) {
    return foldBinary(expr);
  }
  if (expr.type === "binary" && op === "AND") {
    if (isTrueExpr(expr.right)) return expr.left;
    if (isTrueExpr(expr.left)) return expr.right;
    if (isFalseExpr(expr.left) || isFalseExpr(expr.right)) {
      return { type: "literal", value: false };
    }
  }
  if (expr.type === "binary" && op === "OR") {
    if (isFalseExpr(expr.right)) return expr.left;
    if (isFalseExpr(expr.left)) return expr.right;
    if (isTrueExpr(expr.left) || isTrueExpr(expr.right)) {
      return { type: "literal", value: true };
    }
  }
  return expr;
}
function isConstantExpr(expr) {
  return expr && ["literal", "number", "string"].includes(expr.type);
}
function isTrueExpr(expr) {
  return expr?.type === "literal" && expr.value === true;
}
function isFalseExpr(expr) {
  return expr?.type === "literal" && expr.value === false;
}
function foldBinary(expr) {
  const left = getConstantValueExpr(expr.left);
  const right = getConstantValueExpr(expr.right);
  const op = expr.op || expr.operator;
  let result;
  switch (op) {
    case "+":
      result = left + right;
      break;
    case "-":
      result = left - right;
      break;
    case "*":
      result = left * right;
      break;
    case "/":
      result = right !== 0 ? left / right : null;
      break;
    case "%":
      result = left % right;
      break;
    case "=":
    case "==":
      result = left === right;
      break;
    case "!=":
    case "<>":
      result = left !== right;
      break;
    case "<":
      result = left < right;
      break;
    case ">":
      result = left > right;
      break;
    case "<=":
      result = left <= right;
      break;
    case ">=":
      result = left >= right;
      break;
    default:
      return expr;
  }
  return { type: "literal", value: result };
}
function getConstantValueExpr(expr) {
  if (expr.type === "number") return expr.value;
  if (expr.type === "string") return expr.value;
  if (expr.type === "literal") return expr.value;
  return null;
}
function extractRangePredicates(where) {
  const predicates = [];
  collectRangePredicates(where, predicates);
  return predicates;
}
function collectRangePredicates(expr, predicates) {
  if (!expr) return;
  const op = expr.op || expr.operator;
  if (expr.type === "binary" && op === "AND") {
    collectRangePredicates(expr.left, predicates);
    collectRangePredicates(expr.right, predicates);
    return;
  }
  const normalizedOp = op === "==" ? "=" : op;
  if ([">", "<", ">=", "<=", "=", "!=", "<>"].includes(normalizedOp)) {
    if (isColumnRefExpr(expr.left) && isConstantExpr(expr.right)) {
      predicates.push({
        column: getColumnNameExpr(expr.left),
        operator: normalizedOp,
        value: getConstantValueExpr(expr.right)
      });
    } else if (isConstantExpr(expr.left) && isColumnRefExpr(expr.right)) {
      predicates.push({
        column: getColumnNameExpr(expr.right),
        operator: flipOperatorExpr(normalizedOp),
        value: getConstantValueExpr(expr.left)
      });
    }
  }
  if (expr.type === "between" && expr.expr) {
    const col = getColumnNameExpr(expr.expr);
    if (col && expr.low && expr.high) {
      predicates.push({
        column: col,
        operator: ">=",
        value: getConstantValueExpr(expr.low)
      });
      predicates.push({
        column: col,
        operator: "<=",
        value: getConstantValueExpr(expr.high)
      });
    }
  }
}
function flipOperatorExpr(op) {
  const flips = { ">": "<", "<": ">", ">=": "<=", "<=": ">=" };
  return flips[op] || op;
}
function isColumnRefExpr(expr) {
  return expr && (expr.type === "column" || expr.type === "identifier");
}
function getColumnNameExpr(expr) {
  if (expr.type === "column") return expr.name || expr.column;
  if (expr.type === "identifier") return expr.name || expr.value;
  return null;
}
function canPruneFragment(fragmentStats, predicates) {
  for (const pred of predicates) {
    const stats = fragmentStats[pred.column];
    if (!stats) continue;
    const { min, max, nullCount, rowCount } = stats;
    if (nullCount === rowCount) return true;
    switch (pred.operator) {
      case ">":
        if (max <= pred.value) return true;
        break;
      case ">=":
        if (max < pred.value) return true;
        break;
      case "<":
        if (min >= pred.value) return true;
        break;
      case "<=":
        if (min > pred.value) return true;
        break;
      case "=":
        if (pred.value < min || pred.value > max) return true;
        break;
      case "!=":
      case "<>":
        if (min === max && min === pred.value) return true;
        break;
    }
  }
  return false;
}
function explainQuery(db, ast) {
  const plan = {
    type: ast.type,
    tables: [],
    predicates: [],
    optimizations: []
  };
  if (ast.from) {
    plan.tables.push({
      name: ast.from.name || ast.from.table,
      alias: ast.from.alias
    });
  }
  if (ast.joins) {
    for (const join of ast.joins) {
      plan.tables.push({
        name: join.table?.name || join.table?.table,
        alias: join.table?.alias,
        joinType: join.type
      });
    }
  }
  if (ast.where) {
    plan.predicates = extractRangePredicates(ast.where);
  }
  if (ast.where) {
    plan.optimizations.push("PREDICATE_PUSHDOWN");
  }
  if (ast.groupBy) {
    plan.optimizations.push("AGGREGATE");
  }
  if (ast.orderBy) {
    plan.optimizations.push("SORT");
  }
  if (ast.limit) {
    plan.optimizations.push("LIMIT_PUSHDOWN");
  }
  return {
    columns: ["Plan"],
    rows: [[JSON.stringify(plan, null, 2)]],
    total: 1
  };
}

// src/client/database/lance-database.js
var LanceDatabase2 = class {
  constructor() {
    this.tables = /* @__PURE__ */ new Map();
    this.aliases = /* @__PURE__ */ new Map();
    this._planCache = /* @__PURE__ */ new Map();
    this._planCacheMaxSize = 100;
    this.memoryTables = /* @__PURE__ */ new Map();
  }
  /**
   * Register a table with a name
   */
  register(name, dataset) {
    this.tables.set(name, dataset);
  }
  /**
   * Register a remote dataset by URL
   */
  async registerRemote(name, url, options = {}) {
    const lanceql = window.lanceql || globalThis.lanceql;
    if (!lanceql) {
      throw new Error("LanceQL WASM module not loaded. Call LanceQL.load() first.");
    }
    const dataset = await lanceql.openDataset(url, options);
    this.register(name, dataset);
    return dataset;
  }
  /**
   * Get a table by name or alias
   */
  getTable(name) {
    const actualName = this.aliases.get(name) || name;
    const table = this.tables.get(actualName);
    if (!table) {
      throw new Error(`Table '${name}' not found. Did you forget to register it?`);
    }
    return table;
  }
  /**
   * Execute SQL query
   */
  async executeSQL(sql) {
    const cachedPlan = getCachedPlan(this, sql);
    let ast;
    if (cachedPlan) {
      ast = cachedPlan;
    } else {
      const lexer = new SQLLexer(sql);
      const tokens = lexer.tokenize();
      const parser = new SQLParser(tokens);
      ast = parser.parse();
      if (ast.type !== "EXPLAIN") {
        setCachedPlan(this, sql, ast);
      }
    }
    if (ast.type === "EXPLAIN") {
      return explainQuery(this, ast.statement);
    }
    if (ast.type === "CREATE_TABLE") {
      return executeCreateTable(this, ast);
    }
    if (ast.type === "DROP_TABLE") {
      return executeDropTable(this, ast);
    }
    if (ast.type === "INSERT") {
      return executeInsert(this, ast);
    }
    if (ast.type === "UPDATE") {
      return executeUpdate(this, ast);
    }
    if (ast.type === "DELETE") {
      return executeDelete(this, ast);
    }
    if (ast.type === "SET_OPERATION") {
      return this._executeSetOperation(ast);
    }
    if (ast.type !== "SELECT") {
      throw new Error("Only SELECT queries are supported in LanceDatabase");
    }
    if (ast.ctes && ast.ctes.length > 0) {
      return this._executeWithCTEs(ast);
    }
    if (!ast.joins || ast.joins.length === 0) {
      return this._executeSingleTable(ast);
    }
    return executeJoin(this, ast);
  }
  /**
   * Execute query with CTEs
   */
  async _executeWithCTEs(ast) {
    const cteExecutor = new SQLExecutor({ columnNames: [] });
    cteExecutor.setDatabase(this);
    await cteExecutor.materializeCTEs(ast.ctes, this);
    const mainTableName = ast.from?.name?.toLowerCase() || ast.from?.table?.toLowerCase();
    if (mainTableName && cteExecutor._cteResults.has(mainTableName)) {
      return cteExecutor._executeOnInMemoryData(ast, cteExecutor._cteResults.get(mainTableName));
    }
    if (!ast.joins || ast.joins.length === 0) {
      return this._executeSingleTable(ast);
    }
    return executeJoin(this, ast);
  }
  /**
   * Execute SET operation (UNION, INTERSECT, EXCEPT)
   */
  async _executeSetOperation(ast) {
    const leftResult = await this.executeSQL(this._astToSQL(ast.left));
    const rightResult = await this.executeSQL(this._astToSQL(ast.right));
    if (leftResult.columns.length !== rightResult.columns.length) {
      throw new Error("SET operations require same number of columns");
    }
    const rowKey = (row) => JSON.stringify(row);
    let combinedRows;
    switch (ast.operator) {
      case "UNION":
        combinedRows = [...leftResult.rows, ...rightResult.rows];
        if (!ast.all) {
          const seen = /* @__PURE__ */ new Set();
          combinedRows = combinedRows.filter((row) => {
            const key = rowKey(row);
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
          });
        }
        break;
      case "INTERSECT":
        const rightKeys = new Set(rightResult.rows.map(rowKey));
        combinedRows = leftResult.rows.filter((row) => rightKeys.has(rowKey(row)));
        if (!ast.all) {
          const seenI = /* @__PURE__ */ new Set();
          combinedRows = combinedRows.filter((row) => {
            const key = rowKey(row);
            if (seenI.has(key)) return false;
            seenI.add(key);
            return true;
          });
        }
        break;
      case "EXCEPT":
        const excludeKeys = new Set(rightResult.rows.map(rowKey));
        combinedRows = leftResult.rows.filter((row) => !excludeKeys.has(rowKey(row)));
        if (!ast.all) {
          const seenE = /* @__PURE__ */ new Set();
          combinedRows = combinedRows.filter((row) => {
            const key = rowKey(row);
            if (seenE.has(key)) return false;
            seenE.add(key);
            return true;
          });
        }
        break;
      default:
        throw new Error(`Unknown SET operator: ${ast.operator}`);
    }
    if (ast.orderBy && ast.orderBy.length > 0) {
      const colIdxMap = {};
      leftResult.columns.forEach((name, idx) => {
        colIdxMap[name.toLowerCase()] = idx;
      });
      combinedRows.sort((a, b) => {
        for (const ob of ast.orderBy) {
          const colIdx = colIdxMap[ob.column.toLowerCase()];
          if (colIdx === void 0) continue;
          const valA = a[colIdx], valB = b[colIdx];
          const dir = ob.direction === "DESC" ? -1 : 1;
          if (valA == null && valB == null) continue;
          if (valA == null) return 1 * dir;
          if (valB == null) return -1 * dir;
          if (valA < valB) return -1 * dir;
          if (valA > valB) return 1 * dir;
        }
        return 0;
      });
    }
    const offset = ast.offset || 0;
    if (offset > 0) combinedRows = combinedRows.slice(offset);
    if (ast.limit) combinedRows = combinedRows.slice(0, ast.limit);
    return { columns: leftResult.columns, rows: combinedRows, total: combinedRows.length };
  }
  /**
   * Convert AST back to SQL (for recursive SET operation execution)
   */
  _astToSQL(ast) {
    if (ast.type === "SET_OPERATION") {
      const left = this._astToSQL(ast.left);
      const right = this._astToSQL(ast.right);
      const op = ast.operator + (ast.all ? " ALL" : "");
      return `(${left}) ${op} (${right})`;
    }
    let sql = ast.distinct ? "SELECT DISTINCT " : "SELECT ";
    sql += ast.columns.map((col) => {
      if (col.expr?.type === "star") return "*";
      const expr = this._exprToSQL(col.expr);
      return col.alias ? `${expr} AS ${col.alias}` : expr;
    }).join(", ");
    if (ast.from) {
      const tableName = ast.from.name || ast.from.table;
      sql += ` FROM ${tableName}`;
      if (ast.from.alias) sql += ` AS ${ast.from.alias}`;
    }
    if (ast.joins) {
      for (const join of ast.joins) {
        const rightTable = join.table?.name || join.table?.table;
        sql += ` ${join.type} ${rightTable}`;
        if (join.alias) sql += ` AS ${join.alias}`;
        if (join.on) sql += ` ON ${this._exprToSQL(join.on)}`;
      }
    }
    if (ast.where) sql += ` WHERE ${this._exprToSQL(ast.where)}`;
    if (ast.groupBy?.length) sql += ` GROUP BY ${ast.groupBy.join(", ")}`;
    if (ast.having) sql += ` HAVING ${this._exprToSQL(ast.having)}`;
    if (ast.orderBy?.length) {
      sql += ` ORDER BY ${ast.orderBy.map((o) => `${o.column} ${o.direction || "ASC"}`).join(", ")}`;
    }
    if (ast.limit) sql += ` LIMIT ${ast.limit}`;
    if (ast.offset) sql += ` OFFSET ${ast.offset}`;
    return sql;
  }
  /**
   * Convert expression AST to SQL string
   */
  _exprToSQL(expr) {
    if (!expr) return "";
    switch (expr.type) {
      case "literal":
        if (expr.value === null) return "NULL";
        if (typeof expr.value === "string") return `'${expr.value.replace(/'/g, "''")}'`;
        return String(expr.value);
      case "column":
        return expr.table ? `${expr.table}.${expr.column}` : expr.column;
      case "star":
        return "*";
      case "binary":
        return `(${this._exprToSQL(expr.left)} ${expr.operator} ${this._exprToSQL(expr.right)})`;
      case "unary":
        return `(${expr.operator} ${this._exprToSQL(expr.operand)})`;
      case "call":
        const args = expr.args.map((a) => this._exprToSQL(a)).join(", ");
        return `${expr.name}(${expr.distinct ? "DISTINCT " : ""}${args})`;
      case "in":
        const vals = expr.values.map((v) => this._exprToSQL(v)).join(", ");
        return `${this._exprToSQL(expr.expr)} IN (${vals})`;
      case "between":
        return `${this._exprToSQL(expr.expr)} BETWEEN ${this._exprToSQL(expr.low)} AND ${this._exprToSQL(expr.high)}`;
      case "like":
        return `${this._exprToSQL(expr.expr)} LIKE ${this._exprToSQL(expr.pattern)}`;
      default:
        return "";
    }
  }
  /**
   * Execute single-table query (no joins)
   */
  async _executeSingleTable(ast) {
    if (!ast.from) {
      throw new Error("FROM clause required");
    }
    let tableName = ast.from.name || ast.from.table;
    if (!tableName && ast.from.url) {
      throw new Error("Single-table queries must use registered table names, not URLs");
    }
    const tableNameLower = tableName.toLowerCase();
    if (this.memoryTables.has(tableNameLower)) {
      const memTable = this.memoryTables.get(tableNameLower);
      const executor2 = new SQLExecutor({ columnNames: memTable.columns });
      return executor2._executeOnInMemoryData(ast, memTable.toInMemoryData());
    }
    const dataset = this.getTable(tableName);
    const executor = new SQLExecutor(dataset);
    return executor.execute(ast);
  }
  /**
   * Extract column name from expression
   */
  _extractColumnFromExpr(expr, expectedTable) {
    if (expr.type === "column") {
      if (expr.table && expr.table !== expectedTable) {
        return null;
      }
      return expr.column;
    }
    throw new Error(`Invalid join condition expression: ${JSON.stringify(expr)}`);
  }
  /**
   * Get columns needed for a specific table from SELECT list
   */
  _getColumnsForTable(selectColumns, tableAlias) {
    const columns = [];
    for (const item of selectColumns) {
      if (item.type === "star") {
        return ["*"];
      }
      if (item.type === "expr" && item.expr.type === "column") {
        const col = item.expr;
        if (!col.table || col.table === tableAlias) {
          columns.push(col.column);
        }
      }
    }
    return columns.length > 0 ? columns : ["*"];
  }
  // Optimizer delegations
  clearPlanCache() {
    this._planCache.clear();
  }
  getPlanCacheStats() {
    return getPlanCacheStats(this);
  }
  _optimizeExpr(expr) {
    return optimizeExpr(expr);
  }
  _extractRangePredicates(where) {
    return extractRangePredicates(where);
  }
  _canPruneFragment(fragmentStats, predicates) {
    return canPruneFragment(fragmentStats, predicates);
  }
  // Join helper delegations (for backward compatibility)
  _findColumnIndex(columns, columnName) {
    return findColumnIndex(columns, columnName);
  }
  _filterToSQL(expr) {
    return filterToSQL(expr);
  }
  _applyProjection(rows, allColumns, projection, leftAlias, rightAlias) {
    return applyProjection(rows, allColumns, projection, leftAlias, rightAlias);
  }
  _buildInClause(column, keys) {
    return buildInClause(column, keys);
  }
  _appendWhereClause(sql, clause) {
    return appendWhereClause(sql, clause);
  }
};

// src/client/index.js
init_local_database();

// src/client/database/memory-table.js
var MemoryTable2 = class {
  constructor(name, schema) {
    this.name = name;
    this.schema = schema;
    this.columns = schema.map((c) => c.name);
    this.rows = [];
    this._columnIndex = /* @__PURE__ */ new Map();
    this.columns.forEach((col, i) => this._columnIndex.set(col.toLowerCase(), i));
  }
  /**
   * Convert to format compatible with _executeOnInMemoryData
   */
  toInMemoryData() {
    return { columns: this.columns, rows: this.rows };
  }
  /**
   * Get row count
   */
  get rowCount() {
    return this.rows.length;
  }
};
var WorkerPool = class {
  /**
   * Create a new worker pool.
   * @param {number} size - Number of workers (default: navigator.hardwareConcurrency)
   * @param {string} workerPath - Path to worker.js
   */
  constructor(size = null, workerPath = "./worker.js") {
    this.size = size || navigator.hardwareConcurrency || 4;
    this.workerPath = workerPath;
    this.workers = [];
    this.taskQueue = [];
    this.pendingTasks = /* @__PURE__ */ new Map();
    this.nextTaskId = 0;
    this.idleWorkers = [];
    this.initialized = false;
    this.hasSharedArrayBuffer = typeof SharedArrayBuffer !== "undefined";
  }
  /**
   * Initialize all workers.
   * @returns {Promise<void>}
   */
  async init() {
    if (this.initialized) return;
    const initPromises = [];
    for (let i = 0; i < this.size; i++) {
      const worker = new Worker(this.workerPath, { type: "module" });
      this.workers.push(worker);
      worker.onmessage = (e) => this._handleMessage(i, e.data);
      worker.onerror = (e) => this._handleError(i, e);
      initPromises.push(this._initWorker(i));
    }
    await Promise.all(initPromises);
    this.initialized = true;
    console.log(`[WorkerPool] Initialized ${this.size} workers (SharedArrayBuffer: ${this.hasSharedArrayBuffer})`);
  }
  /**
   * Initialize a single worker.
   * @private
   */
  _initWorker(workerId) {
    return new Promise((resolve, reject) => {
      const taskId = this.nextTaskId++;
      this.pendingTasks.set(taskId, {
        resolve: (result) => {
          this.idleWorkers.push(workerId);
          resolve(result);
        },
        reject
      });
      this.workers[workerId].postMessage({
        type: "init",
        id: taskId,
        params: { workerId }
      });
    });
  }
  /**
   * Handle message from worker.
   * @private
   */
  _handleMessage(workerId, data) {
    if (data.type === "ready") {
      return;
    }
    const { id, success, result, error } = data;
    const task = this.pendingTasks.get(id);
    if (!task) {
      console.warn(`[WorkerPool] Unknown task ID: ${id}`);
      return;
    }
    this.pendingTasks.delete(id);
    if (success) {
      task.resolve(result);
    } else {
      task.reject(new Error(error));
    }
    this.idleWorkers.push(workerId);
    this._processQueue();
  }
  /**
   * Handle worker error.
   * @private
   */
  _handleError(workerId, error) {
    console.error(`[WorkerPool] Worker ${workerId} error:`, error);
  }
  /**
   * Process next task in queue.
   * @private
   */
  _processQueue() {
    while (this.taskQueue.length > 0 && this.idleWorkers.length > 0) {
      const task = this.taskQueue.shift();
      const workerId = this.idleWorkers.shift();
      this._sendTask(workerId, task);
    }
  }
  /**
   * Send task to worker.
   * @private
   */
  _sendTask(workerId, task) {
    const worker = this.workers[workerId];
    const transfer = task.transfer || [];
    worker.postMessage({
      type: task.type,
      id: task.id,
      params: task.params
    }, transfer);
  }
  /**
   * Submit a task to the pool.
   * @param {string} type - Task type
   * @param {Object} params - Task parameters
   * @param {Array} transfer - Transferable objects
   * @returns {Promise<any>}
   */
  submit(type, params, transfer = []) {
    return new Promise((resolve, reject) => {
      const taskId = this.nextTaskId++;
      this.pendingTasks.set(taskId, { resolve, reject });
      const task = { type, params, transfer, id: taskId };
      if (this.idleWorkers.length > 0) {
        const workerId = this.idleWorkers.shift();
        this._sendTask(workerId, task);
      } else {
        this.taskQueue.push(task);
      }
    });
  }
  /**
   * Parallel vector search across multiple data chunks.
   *
   * @param {Float32Array} query - Query vector
   * @param {Array<{vectors: Float32Array, startIndex: number}>} chunks - Data chunks
   * @param {number} dim - Vector dimension
   * @param {number} topK - Number of results per chunk
   * @param {boolean} normalized - Whether vectors are L2-normalized
   * @returns {Promise<{indices: Uint32Array, scores: Float32Array}>}
   */
  async parallelVectorSearch(query, chunks, dim, topK, normalized = false) {
    if (!this.initialized) {
      await this.init();
    }
    const searchPromises = chunks.map((chunk, i) => {
      const queryCopy = new Float32Array(query);
      return this.submit("vectorSearch", {
        vectors: chunk.vectors,
        query: queryCopy,
        dim,
        numVectors: chunk.vectors.length / dim,
        topK,
        startIndex: chunk.startIndex,
        normalized
      }, [chunk.vectors.buffer, queryCopy.buffer]);
    });
    const results = await Promise.all(searchPromises);
    return this._mergeTopK(results, topK);
  }
  /**
   * Merge top-k results from multiple workers.
   * @private
   */
  _mergeTopK(results, topK) {
    const allResults = [];
    for (const result of results) {
      for (let i = 0; i < result.count; i++) {
        allResults.push({
          index: result.indices[i],
          score: result.scores[i]
        });
      }
    }
    allResults.sort((a, b) => b.score - a.score);
    const finalK = Math.min(topK, allResults.length);
    const indices = new Uint32Array(finalK);
    const scores = new Float32Array(finalK);
    for (let i = 0; i < finalK; i++) {
      indices[i] = allResults[i].index;
      scores[i] = allResults[i].score;
    }
    return { indices, scores };
  }
  /**
   * Parallel batch similarity computation.
   *
   * @param {Float32Array} query - Query vector
   * @param {Array<Float32Array>} vectorChunks - Chunks of vectors
   * @param {number} dim - Vector dimension
   * @param {boolean} normalized - Whether vectors are L2-normalized
   * @returns {Promise<Float32Array>} - All similarity scores
   */
  async parallelBatchSimilarity(query, vectorChunks, dim, normalized = false) {
    if (!this.initialized) {
      await this.init();
    }
    const similarityPromises = vectorChunks.map((chunk) => {
      const queryCopy = new Float32Array(query);
      return this.submit("batchSimilarity", {
        query: queryCopy,
        vectors: chunk,
        dim,
        numVectors: chunk.length / dim,
        normalized
      }, [chunk.buffer, queryCopy.buffer]);
    });
    const results = await Promise.all(similarityPromises);
    const totalLength = results.reduce((sum, r) => sum + r.scores.length, 0);
    const allScores = new Float32Array(totalLength);
    let offset = 0;
    for (const result of results) {
      allScores.set(result.scores, offset);
      offset += result.scores.length;
    }
    return allScores;
  }
  /**
   * Terminate all workers.
   */
  terminate() {
    for (const worker of this.workers) {
      worker.terminate();
    }
    this.workers = [];
    this.idleWorkers = [];
    this.initialized = false;
  }
};
var SharedVectorStore = class _SharedVectorStore {
  constructor() {
    this.buffer = null;
    this.vectors = null;
    this.dim = 0;
    this.numVectors = 0;
    if (typeof SharedArrayBuffer === "undefined") {
      console.warn("[SharedVectorStore] SharedArrayBuffer not available. Using regular ArrayBuffer.");
    }
  }
  /**
   * Check if SharedArrayBuffer is available.
   */
  static isAvailable() {
    return typeof SharedArrayBuffer !== "undefined" && typeof Atomics !== "undefined";
  }
  /**
   * Allocate shared memory for vectors.
   *
   * @param {number} numVectors - Number of vectors to store
   * @param {number} dim - Vector dimension
   */
  allocate(numVectors, dim) {
    this.numVectors = numVectors;
    this.dim = dim;
    const byteLength = numVectors * dim * 4;
    if (_SharedVectorStore.isAvailable()) {
      this.buffer = new SharedArrayBuffer(byteLength);
    } else {
      this.buffer = new ArrayBuffer(byteLength);
    }
    this.vectors = new Float32Array(this.buffer);
  }
  /**
   * Copy vectors into shared memory.
   *
   * @param {Float32Array} source - Source vectors
   * @param {number} startIndex - Starting index in store
   */
  set(source, startIndex = 0) {
    this.vectors.set(source, startIndex * this.dim);
  }
  /**
   * Get a slice of vectors (view, not copy).
   *
   * @param {number} start - Start vector index
   * @param {number} count - Number of vectors
   * @returns {Float32Array}
   */
  slice(start, count) {
    const startOffset = start * this.dim;
    const length = count * this.dim;
    return new Float32Array(this.buffer, startOffset * 4, length);
  }
  /**
   * Get chunk boundaries for parallel processing.
   *
   * @param {number} numChunks - Number of chunks
   * @returns {Array<{start: number, count: number}>}
   */
  getChunks(numChunks) {
    const chunks = [];
    const chunkSize = Math.ceil(this.numVectors / numChunks);
    for (let i = 0; i < numChunks; i++) {
      const start = i * chunkSize;
      const count = Math.min(chunkSize, this.numVectors - start);
      if (count > 0) {
        chunks.push({ start, count });
      }
    }
    return chunks;
  }
};

// src/client/store/store.js
init_worker_rpc();
var Store = class {
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
    this._encryptionKeyId = null;
  }
  /**
   * Initialize the store (connects to SharedWorker).
   * @returns {Promise<Store>}
   */
  async open() {
    if (this._ready) return this;
    let encryptionConfig = null;
    if (this._getEncryptionKey) {
      const key = await this._getEncryptionKey();
      this._encryptionKeyId = `${this.name}:${Date.now()}`;
      let keyBytes;
      if (key instanceof CryptoKey) {
        keyBytes = await crypto.subtle.exportKey("raw", key);
      } else if (key instanceof ArrayBuffer || key instanceof Uint8Array) {
        keyBytes = key instanceof Uint8Array ? key : new Uint8Array(key);
      } else if (typeof key === "string") {
        const encoder = new TextEncoder();
        const data = encoder.encode(key);
        const hash = await crypto.subtle.digest("SHA-256", data);
        keyBytes = new Uint8Array(hash);
      } else {
        throw new Error("Encryption key must be CryptoKey, ArrayBuffer, Uint8Array, or string");
      }
      encryptionConfig = {
        keyId: this._encryptionKeyId,
        keyBytes: Array.from(keyBytes instanceof Uint8Array ? keyBytes : new Uint8Array(keyBytes))
      };
    }
    await workerRPC("open", {
      name: this.name,
      options: this.options,
      encryption: encryptionConfig
    });
    if (this._sessionMode && typeof window !== "undefined") {
      window.addEventListener("beforeunload", () => {
        this.clear().catch(() => {
        });
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
    return workerRPC("get", { name: this.name, key });
  }
  /**
   * Set a value. Accepts any JSON-serializable value.
   * @param {string} key
   * @param {any} value
   * @returns {Promise<void>}
   */
  async set(key, value) {
    await this._ensureOpen();
    await workerRPC("set", { name: this.name, key, value });
  }
  /**
   * Delete a key.
   * @param {string} key
   * @returns {Promise<boolean>} True if key existed
   */
  async delete(key) {
    await this._ensureOpen();
    return workerRPC("delete", { name: this.name, key });
  }
  /**
   * Check if a key exists.
   * @param {string} key
   * @returns {Promise<boolean>}
   */
  async has(key) {
    const value = await this.get(key);
    return value !== void 0;
  }
  /**
   * List all keys.
   * @returns {Promise<string[]>}
   */
  async keys() {
    await this._ensureOpen();
    return workerRPC("keys", { name: this.name });
  }
  /**
   * Clear all data.
   * @returns {Promise<void>}
   */
  async clear() {
    await this._ensureOpen();
    await workerRPC("clear", { name: this.name });
  }
  /**
   * Filter items in a collection.
   * @param {string} key - Collection key
   * @param {Object} query - Filter query (MongoDB-style operators)
   * @returns {Promise<Array>} Matching items
   */
  async filter(key, query = {}) {
    await this._ensureOpen();
    return workerRPC("filter", { name: this.name, key, query });
  }
  /**
   * Find first item matching query.
   * @param {string} key - Collection key
   * @param {Object} query - Filter query
   * @returns {Promise<Object|undefined>} First matching item
   */
  async find(key, query = {}) {
    await this._ensureOpen();
    return workerRPC("find", { name: this.name, key, query });
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
    return workerRPC("search", { name: this.name, key, text, limit });
  }
  /**
   * Count items in a collection, optionally filtered.
   * @param {string} key - Collection key
   * @param {Object} query - Optional filter query
   * @returns {Promise<number>}
   */
  async count(key, query = null) {
    await this._ensureOpen();
    return workerRPC("count", { name: this.name, key, query });
  }
  /**
   * Subscribe to changes (reactive updates).
   * @param {string} key - Key to watch
   * @param {Function} callback - Called with new value on changes
   * @returns {Function} Unsubscribe function
   */
  subscribe(key, callback) {
    console.warn("[Store] subscribe() not yet implemented");
    return () => {
    };
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
    const result = await workerRPC("enableSemanticSearch", {
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
    await workerRPC("disableSemanticSearch", { name: this.name });
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
};
async function lanceStore(name, options = {}) {
  const store = new Store(name, options);
  await store.open();
  return store;
}

// src/client/store/vault.js
init_worker_rpc();
var Vault = class {
  /**
   * @param {Function|null} getEncryptionKey - Async callback returning encryption key
   */
  constructor(getEncryptionKey = null) {
    this._getEncryptionKey = getEncryptionKey;
    this._encryptionKeyId = null;
    this._ready = false;
  }
  /**
   * Initialize the vault (connects to SharedWorker).
   * @returns {Promise<Vault>}
   */
  async _init() {
    if (this._ready) return this;
    let encryptionConfig = null;
    if (this._getEncryptionKey) {
      const key = await this._getEncryptionKey();
      this._encryptionKeyId = `vault:${Date.now()}`;
      let keyBytes;
      if (key instanceof CryptoKey) {
        keyBytes = await crypto.subtle.exportKey("raw", key);
      } else if (key instanceof ArrayBuffer || key instanceof Uint8Array) {
        keyBytes = key instanceof Uint8Array ? key : new Uint8Array(key);
      } else if (typeof key === "string") {
        const encoder = new TextEncoder();
        const data = encoder.encode(key);
        const hash = await crypto.subtle.digest("SHA-256", data);
        keyBytes = new Uint8Array(hash);
      } else {
        throw new Error("Encryption key must be CryptoKey, ArrayBuffer, Uint8Array, or string");
      }
      encryptionConfig = {
        keyId: this._encryptionKeyId,
        keyBytes: Array.from(keyBytes instanceof Uint8Array ? keyBytes : new Uint8Array(keyBytes))
      };
    }
    await workerRPC("vault:open", { encryption: encryptionConfig });
    this._ready = true;
    return this;
  }
  // =========================================================================
  // KV Operations (stored in encrypted JSON file)
  // =========================================================================
  /**
   * Get a value by key.
   * @param {string} key
   * @returns {Promise<any>} The stored value, or undefined if not found
   */
  async get(key) {
    return workerRPC("vault:get", { key });
  }
  /**
   * Set a value. Accepts any JSON-serializable value.
   * @param {string} key
   * @param {any} value
   * @returns {Promise<void>}
   */
  async set(key, value) {
    await workerRPC("vault:set", { key, value });
  }
  /**
   * Delete a key.
   * @param {string} key
   * @returns {Promise<boolean>} True if key existed
   */
  async delete(key) {
    return workerRPC("vault:delete", { key });
  }
  /**
   * List all keys.
   * @returns {Promise<string[]>}
   */
  async keys() {
    return workerRPC("vault:keys", {});
  }
  /**
   * Check if a key exists.
   * @param {string} key
   * @returns {Promise<boolean>}
   */
  async has(key) {
    const value = await this.get(key);
    return value !== void 0;
  }
  // =========================================================================
  // SQL Operations (tables in Lance format)
  // =========================================================================
  /**
   * Execute a SQL statement.
   * @param {string} sql - SQL statement
   * @returns {Promise<any>} Query results or affected row count
   *
   * @example
   * await v.exec('CREATE TABLE users (id INT, name TEXT, embedding VECTOR(384))');
   * await v.exec('INSERT INTO users VALUES (1, "Alice", [...])');
   * const results = await v.exec('SELECT * FROM users WHERE name NEAR "alice"');
   */
  async exec(sql) {
    return workerRPC("vault:exec", { sql });
  }
  /**
   * Execute a SQL query and return results as array of objects.
   * @param {string} sql - SELECT statement
   * @returns {Promise<Object[]>} Array of row objects
   */
  async query(sql) {
    const result = await this.exec(sql);
    if (!result || !result.columns || !result.rows) return [];
    return result.rows.map((row) => {
      const obj = {};
      result.columns.forEach((col, i) => {
        obj[col] = row[i];
      });
      return obj;
    });
  }
  // =========================================================================
  // DataFrame Operations
  // =========================================================================
  /**
   * Get a DataFrame reference to a table.
   * @param {string} name - Table name
   * @returns {TableRef} DataFrame-style query builder
   */
  table(name) {
    return new TableRef(this, name);
  }
  /**
   * List all tables.
   * @returns {Promise<string[]>}
   */
  async tables() {
    return workerRPC("vault:tables", {});
  }
  // =========================================================================
  // Export Operations
  // =========================================================================
  /**
   * Export a table to Lance format bytes.
   * @param {string} tableName - Name of the table to export
   * @returns {Promise<Uint8Array>} Lance file bytes
   */
  async exportToLance(tableName) {
    const schemaResult = await this.exec(`SELECT * FROM ${tableName} LIMIT 0`);
    if (!schemaResult || !schemaResult.columns) {
      throw new Error(`Table '${tableName}' not found or empty`);
    }
    const dataResult = await this.exec(`SELECT * FROM ${tableName}`);
    if (!dataResult || !dataResult.rows || dataResult.rows.length === 0) {
      throw new Error(`Table '${tableName}' is empty`);
    }
    const writer = new PureLanceWriter();
    const columns = dataResult.columns;
    const rows = dataResult.rows;
    for (let colIdx = 0; colIdx < columns.length; colIdx++) {
      const colName = columns[colIdx];
      const values = rows.map((row) => row[colName] !== void 0 ? row[colName] : row[colIdx]);
      const firstValue = values.find((v) => v !== null && v !== void 0);
      if (firstValue === void 0) {
        writer.addStringColumn(colName, values.map((v) => v === null ? "" : String(v)));
      } else if (typeof firstValue === "bigint") {
        writer.addInt64Column(colName, BigInt64Array.from(values.map((v) => v === null ? 0n : BigInt(v))));
      } else if (typeof firstValue === "number") {
        if (Number.isInteger(firstValue) && firstValue <= 2147483647 && firstValue >= -2147483648) {
          writer.addInt32Column(colName, Int32Array.from(values.map((v) => v === null ? 0 : v)));
        } else {
          writer.addFloat64Column(colName, Float64Array.from(values.map((v) => v === null ? 0 : v)));
        }
      } else if (typeof firstValue === "boolean") {
        writer.addBoolColumn(colName, values.map((v) => v === null ? false : v));
      } else if (Array.isArray(firstValue)) {
        const dim = firstValue.length;
        const flat = new Float32Array(values.length * dim);
        for (let i = 0; i < values.length; i++) {
          const vec = values[i] || new Array(dim).fill(0);
          for (let j = 0; j < dim; j++) {
            flat[i * dim + j] = vec[j] || 0;
          }
        }
        writer.addVectorColumn(colName, flat, dim);
      } else {
        writer.addStringColumn(colName, values.map((v) => v === null ? "" : String(v)));
      }
    }
    return writer.finalize();
  }
  /**
   * Upload binary data to a URL using PUT.
   * @param {Uint8Array} data - Binary data to upload
   * @param {string} url - Signed URL for upload
   * @param {Object} [options] - Upload options
   * @param {Function} [options.onProgress] - Progress callback (loaded, total)
   * @returns {Promise<Response>} Fetch response
   */
  async uploadToUrl(data, url, options = {}) {
    const { onProgress } = options;
    if (onProgress && typeof XMLHttpRequest !== "undefined") {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("PUT", url, true);
        xhr.setRequestHeader("Content-Type", "application/octet-stream");
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            onProgress(e.loaded, e.total);
          }
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve({ ok: true, status: xhr.status });
          } else {
            reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
          }
        };
        xhr.onerror = () => reject(new Error("Upload failed: network error"));
        xhr.send(data);
      });
    }
    const response = await fetch(url, {
      method: "PUT",
      body: data,
      headers: {
        "Content-Type": "application/octet-stream"
      }
    });
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    }
    return response;
  }
  /**
   * Export a table to Lance format and upload to a signed URL.
   * @param {string} tableName - Name of the table to export
   * @param {string} signedUrl - Pre-signed URL for upload (S3/R2/GCS)
   * @param {Object} [options] - Export options
   * @param {Function} [options.onProgress] - Progress callback (loaded, total)
   * @returns {Promise<{size: number, url: string}>} Upload result
   */
  async exportToRemote(tableName, signedUrl, options = {}) {
    const lanceBytes = await this.exportToLance(tableName);
    await this.uploadToUrl(lanceBytes, signedUrl, options);
    return {
      size: lanceBytes.length,
      url: signedUrl.split("?")[0]
      // Return URL without query params
    };
  }
};
var TableRef = class _TableRef {
  constructor(vault2, tableName) {
    this._vault = vault2;
    this._tableName = tableName;
    this._filters = [];
    this._similar = null;
    this._selectCols = null;
    this._limitValue = null;
    this._orderBy = null;
  }
  /**
   * Filter rows by condition.
   * @param {string} column - Column name
   * @param {string} op - Operator ('=', '!=', '<', '<=', '>', '>=')
   * @param {any} value - Value to compare
   * @returns {TableRef} New TableRef with filter applied
   */
  filter(column, op, value) {
    const ref = this._clone();
    ref._filters.push({ column, op, value });
    return ref;
  }
  /**
   * Semantic similarity search.
   * @param {string} column - Column name (text or vector)
   * @param {string} text - Search text
   * @param {number} limit - Max results (default 20)
   * @returns {TableRef} New TableRef with similarity search
   */
  similar(column, text, limit = 20) {
    const ref = this._clone();
    ref._similar = { column, text, limit };
    return ref;
  }
  /**
   * Select specific columns.
   * @param {...string} columns - Column names
   * @returns {TableRef} New TableRef with columns selected
   */
  select(...columns) {
    const ref = this._clone();
    ref._selectCols = columns.flat();
    return ref;
  }
  /**
   * Limit number of results.
   * @param {number} n - Max rows
   * @returns {TableRef} New TableRef with limit
   */
  limit(n) {
    const ref = this._clone();
    ref._limitValue = n;
    return ref;
  }
  /**
   * Order results.
   * @param {string} column - Column to order by
   * @param {string} direction - 'ASC' or 'DESC'
   * @returns {TableRef} New TableRef with ordering
   */
  orderBy(column, direction = "ASC") {
    const ref = this._clone();
    ref._orderBy = { column, direction };
    return ref;
  }
  /**
   * Execute query and return results as array of objects.
   * @returns {Promise<Object[]>}
   */
  async toArray() {
    const sql = this._toSQL();
    return this._vault.query(sql);
  }
  /**
   * Execute query and return first result.
   * @returns {Promise<Object|null>}
   */
  async first() {
    const ref = this._clone();
    ref._limitValue = 1;
    const results = await ref.toArray();
    return results[0] || null;
  }
  /**
   * Count matching rows.
   * @returns {Promise<number>}
   */
  async count() {
    const sql = this._toSQL(true);
    const result = await this._vault.exec(sql);
    return result?.rows?.[0]?.[0] || 0;
  }
  /**
   * Generate SQL from this query.
   * @param {boolean} countOnly - Generate COUNT(*) query
   * @returns {string}
   */
  _toSQL(countOnly = false) {
    const cols = countOnly ? "COUNT(*)" : this._selectCols?.join(", ") || "*";
    let sql = `SELECT ${cols} FROM ${this._tableName}`;
    const whereClauses = [];
    for (const f of this._filters) {
      const val = typeof f.value === "string" ? `'${f.value}'` : f.value;
      whereClauses.push(`${f.column} ${f.op} ${val}`);
    }
    if (this._similar) {
      whereClauses.push(`${this._similar.column} NEAR '${this._similar.text}'`);
    }
    if (whereClauses.length > 0) {
      sql += " WHERE " + whereClauses.join(" AND ");
    }
    if (this._orderBy && !countOnly) {
      sql += ` ORDER BY ${this._orderBy.column} ${this._orderBy.direction}`;
    }
    const limit = this._similar?.limit || this._limitValue;
    if (limit && !countOnly) {
      sql += ` LIMIT ${limit}`;
    }
    return sql;
  }
  _clone() {
    const ref = new _TableRef(this._vault, this._tableName);
    ref._filters = [...this._filters];
    ref._similar = this._similar;
    ref._selectCols = this._selectCols ? [...this._selectCols] : null;
    ref._limitValue = this._limitValue;
    ref._orderBy = this._orderBy;
    return ref;
  }
};
async function vault(getEncryptionKey = null) {
  const v = new Vault(getEncryptionKey);
  await v._init();
  return v;
}

// src/client/index.js
init_lanceql();
init_lanceql();
//# sourceMappingURL=lanceql.js.map
