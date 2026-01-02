/**
 * MemoryTable, WorkerPool, SharedVectorStore - Supporting classes
 */

class MemoryTable {
    constructor(name, schema) {
        this.name = name;
        this.schema = schema;  // [{ name, dataType, primaryKey }]
        this.columns = schema.map(c => c.name);
        this.rows = [];
        this._columnIndex = new Map();
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
}

/**
 * LanceDatabase manages multiple Lance datasets and executes multi-table queries.
 * Supports SQL JOINs across remote datasets with smart byte-range fetching.
 *
 * Features:
 * - Multi-table JOIN support (INNER, LEFT, RIGHT, FULL, CROSS)
 * - Hash join algorithm for efficient execution
 * - Smart column fetching (only fetch needed columns)
 * - Works with static files on CDN
 *
 * Usage:
 *   const db = await LanceQL.createDatabase();
 *   await db.registerRemote('images', 'https://cdn.example.com/images.lance');
 *   await db.registerRemote('captions', 'https://cdn.example.com/captions.lance');
 *   const results = await db.executeSQL(`
 *     SELECT i.url, c.text
 *     FROM images i
 *     JOIN captions c ON i.id = c.image_id
 *     WHERE i.aesthetic > 7.0
 *     LIMIT 20
 *   `);
 */

class WorkerPool {
    /**
     * Create a new worker pool.
     * @param {number} size - Number of workers (default: navigator.hardwareConcurrency)
     * @param {string} workerPath - Path to worker.js
     */
    constructor(size = null, workerPath = './worker.js') {
        this.size = size || navigator.hardwareConcurrency || 4;
        this.workerPath = workerPath;
        this.workers = [];
        this.taskQueue = [];
        this.pendingTasks = new Map();
        this.nextTaskId = 0;
        this.idleWorkers = [];
        this.initialized = false;

        // Check for SharedArrayBuffer support
        this.hasSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
    }

    /**
     * Initialize all workers.
     * @returns {Promise<void>}
     */
    async init() {
        if (this.initialized) return;

        const initPromises = [];

        for (let i = 0; i < this.size; i++) {
            const worker = new Worker(this.workerPath, { type: 'module' });
            this.workers.push(worker);

            // Set up message handling
            worker.onmessage = (e) => this._handleMessage(i, e.data);
            worker.onerror = (e) => this._handleError(i, e);

            // Initialize worker with WASM
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
                type: 'init',
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
        // Handle ready message (initial worker startup)
        if (data.type === 'ready') {
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

        // Worker is now idle
        this.idleWorkers.push(workerId);

        // Process next task in queue
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

        // Submit search task to each worker
        const searchPromises = chunks.map((chunk, i) => {
            // Copy query for each worker (will be transferred)
            const queryCopy = new Float32Array(query);

            return this.submit('vectorSearch', {
                vectors: chunk.vectors,
                query: queryCopy,
                dim,
                numVectors: chunk.vectors.length / dim,
                topK,
                startIndex: chunk.startIndex,
                normalized
            }, [chunk.vectors.buffer, queryCopy.buffer]);
        });

        // Wait for all workers
        const results = await Promise.all(searchPromises);

        // Merge results from all workers
        return this._mergeTopK(results, topK);
    }

    /**
     * Merge top-k results from multiple workers.
     * @private
     */
    _mergeTopK(results, topK) {
        // Collect all results
        const allResults = [];

        for (const result of results) {
            for (let i = 0; i < result.count; i++) {
                allResults.push({
                    index: result.indices[i],
                    score: result.scores[i]
                });
            }
        }

        // Sort by score descending
        allResults.sort((a, b) => b.score - a.score);

        // Take top-k
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

        const similarityPromises = vectorChunks.map(chunk => {
            const queryCopy = new Float32Array(query);
            return this.submit('batchSimilarity', {
                query: queryCopy,
                vectors: chunk,
                dim,
                numVectors: chunk.length / dim,
                normalized
            }, [chunk.buffer, queryCopy.buffer]);
        });

        const results = await Promise.all(similarityPromises);

        // Concatenate all scores
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
}

// ============================================================================
// SharedArrayBuffer Vector Store - Zero-copy data sharing
// ============================================================================

/**
 * SharedVectorStore provides zero-copy data sharing between main thread and workers.
 * Requires Cross-Origin-Isolation (COOP/COEP headers).
 */
class SharedVectorStore {
    constructor() {
        this.buffer = null;
        this.vectors = null;
        this.dim = 0;
        this.numVectors = 0;

        if (typeof SharedArrayBuffer === 'undefined') {
            console.warn('[SharedVectorStore] SharedArrayBuffer not available. Using regular ArrayBuffer.');
        }
    }

    /**
     * Check if SharedArrayBuffer is available.
     */
    static isAvailable() {
        return typeof SharedArrayBuffer !== 'undefined' &&
               typeof Atomics !== 'undefined';
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

        const byteLength = numVectors * dim * 4; // float32

        if (SharedVectorStore.isAvailable()) {
            this.buffer = new SharedArrayBuffer(byteLength);
        } else {
            // Fallback to regular ArrayBuffer
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
}

// ============================================================================
// CSS-Driven Query Engine - Zero JavaScript Data Binding
// ============================================================================

/**
 * LanceData provides CSS-driven data binding for Lance datasets.
 *
 * TRULY CSS-DRIVEN: No JavaScript initialization required!
 * Just add lq-* attributes to any element.
 *
 * Usage (pure HTML/CSS, zero JavaScript):
 * ```html
 * <div lq-query="SELECT url, text FROM read_lance('https://data.metal0.dev/laion-1m/images.lance') LIMIT 10"
 *      lq-render="table">
 * </div>
 * ```
 *
 * Attributes (supports both lq-* and data-* prefixes):
 * - lq-src / data-dataset: Dataset URL (optional if URL is in query)
 * - lq-query / data-query: SQL query string (required)
 * - lq-render / data-render: Renderer type - table, list, value, images, json (default: table)
 * - lq-columns / data-columns: Comma-separated column names to display
 * - lq-bind / data-bind: Input element selector for reactive binding
 *
 * The system auto-initializes when the script loads.
 */

export { MemoryTable, WorkerPool, SharedVectorStore };
