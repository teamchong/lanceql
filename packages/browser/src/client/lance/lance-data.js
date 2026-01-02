/**
 * LanceData - Unified Lance data interface and DataFrame
 */

class LanceDataBase {
    constructor(type) {
        this.type = type; // 'local' | 'remote' | 'cached'
    }

    // Abstract methods - must be implemented by subclasses
    async getSchema() { throw new Error('Not implemented'); }
    async getRowCount() { throw new Error('Not implemented'); }
    async readColumn(colIdx, start = 0, count = null) { throw new Error('Not implemented'); }
    async *scan(options = {}) { throw new Error('Not implemented'); }

    // Optional methods
    async insert(rows) { throw new Error('Write not supported for this source'); }
    isCached() { return false; }
    async prefetch() { }
    async evict() { }
    async close() { }
}

/**
 * OPFS-backed Lance data for local files.
 * Uses ChunkedLanceReader for efficient memory usage.
 */
class OPFSLanceData extends LanceDataBase {
    constructor(path, storage = opfsStorage) {
        super('local');
        this.path = path;
        this.storage = storage;
        this.reader = null;
        this.database = null;
        this._isDatabase = false;
    }

    /**
     * Open OPFS Lance file or database
     */
    async open() {
        // Check if it's a database (directory with manifest)
        const manifestPath = `${this.path}/__manifest__`;
        if (await this.storage.exists(manifestPath)) {
            this._isDatabase = true;
            this.database = new LocalDatabase(this.path, this.storage);
            await this.database.open();
        } else {
            // Single Lance file
            this.reader = await ChunkedLanceReader.open(this.storage, this.path);
        }
        return this;
    }

    async getSchema() {
        if (this._isDatabase) {
            const tables = this.database.listTables();
            if (tables.length === 0) return [];
            return this.database.getSchema(tables[0]);
        }
        // For single file, return column count (no schema info in simple Lance files)
        return Array.from({ length: this.reader.getNumColumns() }, (_, i) => ({
            name: `col_${i}`,
            type: 'unknown'
        }));
    }

    async getRowCount() {
        if (this._isDatabase) {
            const tables = this.database.listTables();
            if (tables.length === 0) return 0;
            return this.database.count(tables[0]);
        }
        // Read first column metadata for row count
        const meta = await this.reader.readColumnMetaRaw(0);
        // Parse row count from protobuf (simplified)
        return 0; // Would need protobuf decoder
    }

    async readColumn(colIdx, start = 0, count = null) {
        if (this._isDatabase) {
            throw new Error('Use select() for database queries');
        }
        return this.reader.readColumnMetaRaw(colIdx);
    }

    async *scan(options = {}) {
        if (this._isDatabase) {
            const tables = this.database.listTables();
            if (tables.length === 0) return;
            yield* this.database.scan(tables[0], options);
        } else {
            throw new Error('scan() requires database, use readColumn() for single files');
        }
    }

    async insert(rows) {
        if (!this._isDatabase) {
            throw new Error('insert() requires database');
        }
        const tables = this.database.listTables();
        if (tables.length === 0) {
            throw new Error('No tables in database');
        }
        return this.database.insert(tables[0], rows);
    }

    isCached() {
        return true; // OPFS is always local
    }

    async close() {
        if (this.reader) {
            this.reader.close();
        }
        if (this.database) {
            await this.database.close();
        }
    }
}

/**
 * HTTP-backed Lance data for remote files.
 * Uses HotTierCache for OPFS caching.
 */
class RemoteLanceData extends LanceDataBase {
    constructor(url) {
        super('remote');
        this.url = url;
        this.remoteFile = null;
        this.cachedPath = null;
    }

    async open() {
        // Check if already cached
        const cacheInfo = await hotTierCache.getCacheInfo(this.url);
        if (cacheInfo && cacheInfo.complete) {
            this.type = 'cached';
            this.cachedPath = cacheInfo.path;
        }

        // Open remote file (will use HTTP Range requests)
        // This assumes RemoteLanceFile exists in the codebase
        if (typeof RemoteLanceFile !== 'undefined') {
            this.remoteFile = await RemoteLanceFile.open(null, this.url);
        }

        return this;
    }

    async getSchema() {
        if (!this.remoteFile) {
            return [];
        }
        // Get column types
        const numCols = this.remoteFile.numColumns;
        const schema = [];
        for (let i = 0; i < numCols; i++) {
            const type = await this.remoteFile.getColumnType(i);
            schema.push({ name: `col_${i}`, type });
        }
        return schema;
    }

    async getRowCount() {
        if (!this.remoteFile) return 0;
        return this.remoteFile.getRowCount();
    }

    async readColumn(colIdx, start = 0, count = null) {
        if (!this.remoteFile) {
            throw new Error('Remote file not opened');
        }
        // Use remote file's column reading
        const type = await this.remoteFile.getColumnType(colIdx);
        if (type.includes('int64')) {
            return this.remoteFile.readInt64Column(colIdx, count);
        } else if (type.includes('float64')) {
            return this.remoteFile.readFloat64Column(colIdx, count);
        } else if (type.includes('string')) {
            return this.remoteFile.readStrings(colIdx, count);
        }
        throw new Error(`Unsupported column type: ${type}`);
    }

    async *scan(options = {}) {
        // For remote files, read in batches
        const batchSize = options.batchSize || 10000;
        const rowCount = await this.getRowCount();
        const schema = await this.getSchema();

        for (let offset = 0; offset < rowCount; offset += batchSize) {
            const count = Math.min(batchSize, rowCount - offset);
            const batch = [];

            // Read each column for this batch
            const columns = {};
            for (let i = 0; i < schema.length; i++) {
                columns[schema[i].name] = await this.readColumn(i, offset, count);
            }

            // Build rows
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
        return this.type === 'cached';
    }

    async prefetch() {
        // Cache entire file to OPFS
        await hotTierCache.cache(this.url);
        const cacheInfo = await hotTierCache.getCacheInfo(this.url);
        if (cacheInfo && cacheInfo.complete) {
            this.type = 'cached';
            this.cachedPath = cacheInfo.path;
        }
    }

    async evict() {
        await hotTierCache.evict(this.url);
        this.type = 'remote';
        this.cachedPath = null;
    }

    async close() {
        if (this.remoteFile) {
            this.remoteFile.close();
        }
    }
}

/**
 * Factory function to open Lance data from any source.
 * Supports:
 * - opfs://path - Local OPFS file or database
 * - https://url - Remote HTTP file (with optional caching)
 *
 * @param {string} source - Data source URI
 * @returns {Promise<LanceDataBase>}
 *
 * @example
 * // Local OPFS database
 * const local = await openLance('opfs://mydb');
 * for await (const batch of local.scan()) {
 *   processBatch(batch);
 * }
 *
 * // Remote file with caching
 * const remote = await openLance('https://example.com/data.lance');
 * await remote.prefetch(); // Cache to OPFS
 * const data = await remote.readColumn(0);
 */
async function openLance(source) {
    if (source.startsWith('opfs://')) {
        const path = source.slice(7);
        const data = new OPFSLanceData(path);
        await data.open();
        return data;
    } else if (source.startsWith('http://') || source.startsWith('https://')) {
        const data = new RemoteLanceData(source);
        await data.open();
        return data;
    } else {
        // Assume OPFS path without prefix
        const data = new OPFSLanceData(source);
        await data.open();
        return data;
    }
}

// Export unified API


class DataFrame {
    constructor(file) {
        this.file = file;
        this._filterOps = [];  // Array of {colIdx, op, value, type, opStr}
        this._selectCols = null;
        this._limitValue = null;
        this._isRemote = file._isRemote || file.baseUrl !== undefined;
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
    filter(colIdx, op, value, type = 'int64') {
        const opMap = {
            '=': LanceFile.Op?.EQ ?? 0, '==': LanceFile.Op?.EQ ?? 0,
            '!=': LanceFile.Op?.NE ?? 1, '<>': LanceFile.Op?.NE ?? 1,
            '<': LanceFile.Op?.LT ?? 2,
            '<=': LanceFile.Op?.LE ?? 3,
            '>': LanceFile.Op?.GT ?? 4,
            '>=': LanceFile.Op?.GE ?? 5
        };

        const df = new DataFrame(this.file);
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
        // Handle array passed as first arg: select([0,1,2]) or select(0,1,2)
        const cols = Array.isArray(colIndices[0]) ? colIndices[0] : colIndices;
        const df = new DataFrame(this.file);
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
        const df = new DataFrame(this.file);
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
        const colNames = this.file.columnNames || this.file._schema?.map(s => s.name) ||
            Array.from({ length: this.file._numColumns || 6 }, (_, i) => `col_${i}`);

        // SELECT clause
        let selectClause;
        if (this._selectCols && this._selectCols.length > 0) {
            selectClause = this._selectCols.map(i => colNames[i] || `col_${i}`).join(', ');
        } else {
            selectClause = '*';
        }

        // WHERE clause
        let whereClause = '';
        if (this._filterOps.length > 0) {
            const conditions = this._filterOps.map(f => {
                const colName = colNames[f.colIdx] || `col_${f.colIdx}`;
                const val = f.type === 'string' ? `'${f.value}'` : f.value;
                return `${colName} ${f.opStr} ${val}`;
            });
            whereClause = ` WHERE ${conditions.join(' AND ')}`;
        }

        // LIMIT clause
        const limitClause = this._limitValue ? ` LIMIT ${this._limitValue}` : '';

        return `SELECT ${selectClause} FROM dataset${whereClause}${limitClause}`;
    }

    /**
     * Execute the query and return row indices (sync, local only).
     * @returns {Uint32Array}
     */
    collectIndices() {
        if (this._isRemote) {
            throw new Error('collectIndices() is sync-only. Use collect() for remote datasets.');
        }

        let indices = null;

        // Apply filters
        for (const f of this._filterOps) {
            let newIndices;
            if (f.type === 'int64') {
                newIndices = this.file.filterInt64(f.colIdx, f.op, f.value);
            } else {
                newIndices = this.file.filterFloat64(f.colIdx, f.op, f.value);
            }

            if (indices === null) {
                indices = newIndices;
            } else {
                // Intersect indices
                const set = new Set(newIndices);
                indices = indices.filter(i => set.has(i));
                indices = new Uint32Array(indices);
            }
        }

        // If no filters, get all row indices
        if (indices === null) {
            const rowCount = Number(this.file.getRowCount(0));
            indices = new Uint32Array(rowCount);
            for (let i = 0; i < rowCount; i++) indices[i] = i;
        }

        // Apply limit
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
        // Remote: generate SQL and execute
        if (this._isRemote) {
            const sql = this.toSQL();
            return await this.file.executeSQL(sql);
        }

        // Local: use sync WASM methods
        const indices = this.collectIndices();
        const cols = this._selectCols ||
            Array.from({ length: this.file.numColumns }, (_, i) => i);

        const columns = [];
        const columnNames = [];

        for (const colIdx of cols) {
            columnNames.push(this.file.columnNames?.[colIdx] || `col_${colIdx}`);
            // Try int64 first, then float64
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
}

/**
 * Represents a Lance file opened from a remote URL.
 * Uses HTTP Range requests to fetch data on demand.
 */

class LanceData {
    static _initialized = false;
    static _observer = null;
    static _wasm = null;
    static _datasets = new Map(); // Cache datasets by URL
    static _renderers = {};
    static _bindings = new Map();
    static _queryCache = new Map();
    static _defaultDataset = null;

    /**
     * Auto-initialize when DOM is ready.
     * Called automatically - no user action needed.
     */
    static _autoInit() {
        if (LanceData._initialized) return;
        LanceData._initialized = true;

        // Register built-in renderers
        LanceData._registerBuiltinRenderers();

        // Inject trigger styles
        LanceData._injectTriggerStyles();

        // Set up observer for lance-data elements
        LanceData._setupObserver();

        // Process any existing elements
        LanceData._processExisting();
    }

    /**
     * Get or load a dataset (cached).
     */
    static async _getDataset(url) {
        if (!url) {
            if (LanceData._defaultDataset) return LanceData._datasets.get(LanceData._defaultDataset);
            throw new Error('No dataset URL. Add data-dataset="https://..." to your element.');
        }

        if (LanceData._datasets.has(url)) {
            return LanceData._datasets.get(url);
        }

        // Load WASM if needed
        if (!LanceData._wasm) {
            // Try to find wasm URL from script tag or use default
            const wasmUrl = document.querySelector('script[data-lanceql-wasm]')?.dataset.lanceqlWasm
                || './lanceql.wasm';
            LanceData._wasm = await LanceQL.load(wasmUrl);
        }

        const dataset = await RemoteLanceDataset.open(LanceData._wasm, url);
        LanceData._datasets.set(url, dataset);

        // First dataset becomes default
        if (!LanceData._defaultDataset) {
            LanceData._defaultDataset = url;
        }

        return dataset;
    }

    /**
     * Manual init (optional) - for advanced configuration.
     */
    static async init(options = {}) {
        LanceData._autoInit();

        if (options.wasmUrl) {
            LanceData._wasm = await LanceQL.load(options.wasmUrl);
        }
        if (options.dataset) {
            await LanceData._getDataset(options.dataset);
        }
    }

    /**
     * Inject CSS that triggers JavaScript via animation events.
     */
    static _injectTriggerStyles() {
        if (document.getElementById('lance-data-triggers')) return;

        const style = document.createElement('style');
        style.id = 'lance-data-triggers';
        style.textContent = `
            /* Lance Data CSS Trigger System */
            @keyframes lance-query-trigger {
                from { --lance-trigger: 0; }
                to { --lance-trigger: 1; }
            }

            /* Elements with lance-data class trigger on insertion */
            .lance-data {
                animation: lance-query-trigger 0.001s;
            }

            /* Re-trigger on data attribute changes */
            .lance-data[data-refresh] {
                animation: lance-query-trigger 0.001s;
            }

            /* Loading state */
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

            /* Error state */
            .lance-data[data-error]::before {
                content: attr(data-error);
                color: #ef4444;
                font-size: 12px;
            }

            /* Result container styling */
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

            /* Value renderer */
            .lance-data[style*="--render: value"] .lance-value,
            .lance-data[style*="--render:'value'"] .lance-value,
            .lance-data[style*='--render:"value"'] .lance-value {
                font-size: 24px;
                font-weight: 600;
                color: #3b82f6;
            }

            /* List renderer */
            .lance-data .lance-list {
                list-style: none;
                padding: 0;
                margin: 0;
            }

            .lance-data .lance-list li {
                padding: 8px 0;
                border-bottom: 1px solid #334155;
            }

            /* JSON renderer */
            .lance-data .lance-json {
                background: #0f172a;
                padding: 12px;
                border-radius: 8px;
                font-family: 'SF Mono', Monaco, monospace;
                font-size: 12px;
                white-space: pre-wrap;
                overflow-x: auto;
            }

            /* Image grid renderer */
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
        if (LanceData._observer) return;

        // Helper to check if element has lq-* attributes
        const hasLqAttrs = (el) => {
            return el.hasAttribute?.('lq-query') || el.hasAttribute?.('lq-src') ||
                   el.classList?.contains('lance-data');
        };

        LanceData._observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                // New nodes added
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        if (hasLqAttrs(node)) {
                            LanceData._processElement(node);
                        }
                        // Check descendants
                        node.querySelectorAll?.('[lq-query], [lq-src], .lance-data')?.forEach(el => {
                            LanceData._processElement(el);
                        });
                    }
                }

                // Attribute changes
                if (mutation.type === 'attributes' && hasLqAttrs(mutation.target)) {
                    LanceData._processElement(mutation.target);
                }
            }
        });

        LanceData._observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['lq-query', 'lq-src', 'lq-render', 'lq-bind', 'data-query', 'data-dataset', 'data-render', 'data-refresh']
        });

        // Also listen for animation events (CSS trigger)
        document.body.addEventListener('animationstart', (e) => {
            if (e.animationName === 'lance-query-trigger' && hasLqAttrs(e.target)) {
                LanceData._processElement(e.target);
            }
        });
    }

    /**
     * Process existing lance-data elements.
     */
    static _processExisting() {
        document.querySelectorAll('[lq-query], [lq-src], .lance-data').forEach(el => {
            LanceData._processElement(el);
        });
    }

    /**
     * Parse config from attributes (supports both lq-* and data-* prefixes).
     */
    static _parseConfig(el) {
        // Helper to get attribute value with fallback (lq-* takes precedence)
        const getAttr = (lqName, dataName) => {
            return el.getAttribute(lqName) || el.dataset[dataName] || null;
        };

        return {
            dataset: getAttr('lq-src', 'dataset'),
            query: getAttr('lq-query', 'query'),
            render: getAttr('lq-render', 'render') || 'table',
            columns: (getAttr('lq-columns', 'columns') || '')
                .split(',')
                .map(c => c.trim())
                .filter(Boolean),
            bind: getAttr('lq-bind', 'bind'),
        };
    }

    /**
     * Render pre-computed results to an element (CSS-driven from JS).
     * Use this when you already have query results and just want CSS-driven rendering.
     * @param {HTMLElement|string} el - Element or selector
     * @param {Object} results - Query results {columns, rows, total}
     * @param {Object} [options] - Render options
     * @param {string} [options.render] - Renderer type (table, images, json, etc.)
     */
    static render(el, results, options = {}) {
        const element = typeof el === 'string' ? document.querySelector(el) : el;
        if (!element) {
            console.error('[LanceData] Element not found:', el);
            return;
        }

        try {
            // Dispatch start event
            element.dispatchEvent(new CustomEvent('lq-start', {
                detail: { query: options.query || null }
            }));

            const renderType = options.render || element.dataset.render || 'table';
            const renderer = LanceData._renderers[renderType] || LanceData._renderers.table;

            // Store results in cache for potential re-renders
            if (element.id) {
                LanceData._queryCache.set(`rendered:${element.id}`, results);
            }

            element.innerHTML = renderer(results, { render: renderType, ...options });

            // Dispatch complete event
            element.dispatchEvent(new CustomEvent('lq-complete', {
                detail: {
                    query: options.query || null,
                    columns: results.columns || [],
                    total: results.total || results.rows?.length || 0
                }
            }));
        } catch (error) {
            // Dispatch error event
            element.dispatchEvent(new CustomEvent('lq-error', {
                detail: {
                    query: options.query || null,
                    message: error.message,
                    error: error
                }
            }));
            throw error;
        }
    }

    /**
     * Extract dataset URL from SQL query (e.g., read_lance('https://...'))
     */
    static _extractUrlFromQuery(sql) {
        const match = sql.match(/read_lance\s*\(\s*['"]([^'"]+)['"]/i);
        return match ? match[1] : null;
    }

    /**
     * Process a single lance-data element.
     */
    static async _processElement(el) {
        // Prevent double processing
        if (el.dataset.processing === 'true') return;
        el.dataset.processing = 'true';

        try {
            const config = LanceData._parseConfig(el);

            if (!config.query) {
                el.dataset.processing = 'false';
                return;
            }

            // Set up input binding if specified
            if (config.bind) {
                LanceData._setupBinding(el, config);
            }

            el.dataset.loading = 'true';
            delete el.dataset.error;

            // Dispatch start event (for Alpine.js integration)
            el.dispatchEvent(new CustomEvent('lq-start', {
                detail: { query: config.query }
            }));

            // Extract dataset URL from query if not specified
            const datasetUrl = config.dataset || LanceData._extractUrlFromQuery(config.query);

            // Get dataset (auto-loads and caches)
            const dataset = await LanceData._getDataset(datasetUrl);

            // Check cache
            const cacheKey = `${datasetUrl || 'default'}:${config.query}`;
            let results = LanceData._queryCache.get(cacheKey);

            if (!results) {
                // Execute query
                results = await dataset.executeSQL(config.query);
                LanceData._queryCache.set(cacheKey, results);
            }

            // Render results
            const renderer = LanceData._renderers[config.render] || LanceData._renderers.table;
            el.innerHTML = renderer(results, config);

            delete el.dataset.loading;

            // Dispatch complete event (for Alpine.js integration)
            el.dispatchEvent(new CustomEvent('lq-complete', {
                detail: {
                    query: config.query,
                    columns: results.columns || [],
                    total: results.total || results.rows?.length || 0
                }
            }));
        } catch (error) {
            delete el.dataset.loading;
            el.dataset.error = error.message;
            console.error('[LanceData]', error);

            // Dispatch error event (for Alpine.js integration)
            el.dispatchEvent(new CustomEvent('lq-error', {
                detail: {
                    query: config.query,
                    message: error.message,
                    error: error
                }
            }));
        } finally {
            el.dataset.processing = 'false';
        }
    }

    /**
     * Set up reactive binding to an input element.
     */
    static _setupBinding(el, config) {
        const input = document.querySelector(config.bind);
        if (!input) return;

        // Store binding reference
        const bindingKey = config.bind;
        if (LanceData._bindings.has(bindingKey)) return;

        const handler = () => {
            // Replace $value in query with input value
            const value = input.value;
            const newQuery = config.query.replace(/\$value/g, value);

            // Set via both attribute types
            if (el.hasAttribute('lq-query')) {
                el.setAttribute('lq-query', newQuery);
            } else {
                el.dataset.query = newQuery;
            }

            // Trigger refresh
            el.dataset.refresh = Date.now();
        };

        input.addEventListener('input', handler);
        input.addEventListener('change', handler);

        LanceData._bindings.set(bindingKey, { input, handler, element: el });
    }

    /**
     * Register a custom renderer.
     * @param {string} name - Renderer name
     * @param {Function} fn - Renderer function (results, config) => html
     */
    static registerRenderer(name, fn) {
        LanceData._renderers[name] = fn;
    }

    /**
     * Register built-in renderers.
     */
    static _registerBuiltinRenderers() {
        // Table renderer - handles both {columns, rows} and array-of-objects formats
        LanceData._renderers.table = (results, config) => {
            if (!results) {
                return '<div class="lance-empty">No results</div>';
            }

            // Detect format: {columns, rows} vs array of objects
            let columns, rows;
            if (results.columns && results.rows) {
                // SQLExecutor format: {columns: ['col1', 'col2'], rows: [[val1, val2], ...]}
                columns = config.columns || results.columns.filter(k =>
                    !k.startsWith('_') && k !== 'embedding'
                );
                rows = results.rows;
            } else if (Array.isArray(results)) {
                // Array of objects format: [{col1: val1, col2: val2}, ...]
                if (results.length === 0) {
                    return '<div class="lance-empty">No results</div>';
                }
                columns = config.columns || Object.keys(results[0]).filter(k =>
                    !k.startsWith('_') && k !== 'embedding'
                );
                rows = results.map(row => columns.map(col => row[col]));
            } else {
                return '<div class="lance-empty">No results</div>';
            }

            if (rows.length === 0) {
                return '<div class="lance-empty">No results</div>';
            }

            let html = '<table><thead><tr>';
            for (const col of columns) {
                html += `<th>${LanceData._escapeHtml(String(col))}</th>`;
            }
            html += '</tr></thead><tbody>';

            for (const row of rows) {
                html += '<tr>';
                for (let i = 0; i < columns.length; i++) {
                    const value = row[i];
                    html += `<td>${LanceData._formatValue(value)}</td>`;
                }
                html += '</tr>';
            }

            html += '</tbody></table>';
            return html;
        };

        // List renderer
        LanceData._renderers.list = (results, config) => {
            if (!results || results.length === 0) {
                return '<div class="lance-empty">No results</div>';
            }

            const displayCol = config.columns?.[0] || Object.keys(results[0])[0];

            let html = '<ul class="lance-list">';
            for (const row of results) {
                html += `<li>${LanceData._formatValue(row[displayCol])}</li>`;
            }
            html += '</ul>';
            return html;
        };

        // Single value renderer
        LanceData._renderers.value = (results, config) => {
            if (!results || results.length === 0) {
                return '<div class="lance-empty">-</div>';
            }

            const firstRow = results[0];
            const firstKey = Object.keys(firstRow)[0];
            const value = firstRow[firstKey];

            return `<div class="lance-value">${LanceData._formatValue(value)}</div>`;
        };

        // JSON renderer
        LanceData._renderers.json = (results, config) => {
            return `<pre class="lance-json">${LanceData._escapeHtml(JSON.stringify(results, null, 2))}</pre>`;
        };

        // Image grid renderer (for datasets with url column)
        LanceData._renderers.images = (results, config) => {
            if (!results || results.length === 0) {
                return '<div class="lance-empty">No images</div>';
            }

            let html = '<div class="lance-images">';
            for (const row of results) {
                const url = row.url || row.image_url || row.src;
                const text = row.text || row.caption || row.title || '';

                if (url) {
                    html += `
                        <div class="image-card">
                            <img src="${LanceData._escapeHtml(url)}" alt="${LanceData._escapeHtml(text)}" loading="lazy">
                            ${text ? `<div class="image-meta">${LanceData._escapeHtml(text.substring(0, 100))}</div>` : ''}
                        </div>
                    `;
                }
            }
            html += '</div>';
            return html;
        };

        // Count renderer (for aggregates)
        LanceData._renderers.count = (results, config) => {
            const count = results?.[0]?.count ?? results?.length ?? 0;
            return `<span class="lance-count">${count.toLocaleString()}</span>`;
        };
    }

    /**
     * Check if a string is an image URL.
     */
    static _isImageUrl(str) {
        if (!str || typeof str !== 'string') return false;
        const lower = str.toLowerCase();
        return (lower.startsWith('http://') || lower.startsWith('https://')) &&
               (lower.includes('.jpg') || lower.includes('.jpeg') || lower.includes('.png') ||
                lower.includes('.gif') || lower.includes('.webp') || lower.includes('.svg'));
    }

    /**
     * Check if a string is a URL.
     */
    static _isUrl(str) {
        if (!str || typeof str !== 'string') return false;
        return str.startsWith('http://') || str.startsWith('https://');
    }

    /**
     * Format a value for display.
     */
    static _formatValue(value) {
        if (value === null || value === undefined) return '<span class="null-value">NULL</span>';
        if (value === '') return '<span class="empty-value">(empty)</span>';

        if (typeof value === 'number') {
            return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
        }
        if (Array.isArray(value)) {
            if (value.length > 10) return `<span class="vector-badge">[${value.length}d]</span>`;
            return `[${value.slice(0, 5).map(v => LanceData._formatValue(v)).join(', ')}${value.length > 5 ? '...' : ''}]`;
        }
        if (typeof value === 'object') return JSON.stringify(value);

        const str = String(value);

        // Handle image URLs - show thumbnail
        if (LanceData._isImageUrl(str)) {
            const escaped = LanceData._escapeHtml(str);
            const short = escaped.length > 40 ? escaped.substring(0, 40) + '...' : escaped;
            return `<div class="image-cell">
                <img src="${escaped}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
                <div class="image-placeholder" style="display:none"><svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg></div>
                <a href="${escaped}" target="_blank" class="url-text" title="${escaped}">${short}</a>
            </div>`;
        }

        // Handle other URLs - show as clickable link
        if (LanceData._isUrl(str)) {
            const escaped = LanceData._escapeHtml(str);
            const short = escaped.length > 50 ? escaped.substring(0, 50) + '...' : escaped;
            return `<a href="${escaped}" target="_blank" class="url-link" title="${escaped}">${short}</a>`;
        }

        // Handle long strings - truncate
        if (str.length > 100) return `<span title="${LanceData._escapeHtml(str)}">${LanceData._escapeHtml(str.substring(0, 100))}...</span>`;
        return LanceData._escapeHtml(str);
    }

    /**
     * Escape HTML special characters.
     */
    static _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    /**
     * Clear the query cache.
     */
    static clearCache() {
        LanceData._queryCache.clear();
    }

    /**
     * Refresh all lance-data elements.
     */
    static refresh() {
        LanceData._queryCache.clear();
        document.querySelectorAll('.lance-data').forEach(el => {
            el.setAttribute('data-refresh', Date.now());
        });
    }

    /**
     * Destroy and clean up.
     */
    static destroy() {
        if (LanceData._observer) {
            LanceData._observer.disconnect();
            LanceData._observer = null;
        }

        // Remove bindings
        for (const [key, binding] of LanceData._bindings) {
            binding.input.removeEventListener('input', binding.handler);
            binding.input.removeEventListener('change', binding.handler);
        }
        LanceData._bindings.clear();

        // Remove injected styles
        document.getElementById('lance-data-triggers')?.remove();

        LanceData._instance = null;
        LanceData._dataset = null;
        LanceData._queryCache.clear();
    }
}

// Auto-initialize when DOM is ready (truly CSS-driven - no JS needed by user)
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => LanceData._autoInit());
    } else {
        LanceData._autoInit();
    }
}

// =============================================================================
// sql.js-Compatible API - Drop-in replacement with vector search
// =============================================================================

/**
 * Statement class - sql.js compatible prepared statement
 *
 * Thin wrapper that delegates to WASM-based SQLExecutor
 */
class Statement {
    constructor(db, sql) {
        this.db = db;
        this.sql = sql;
        this.params = null;
        this.results = null;
        this.resultIndex = 0;
        this.done = false;
    }

    /**
     * Bind parameters to the statement
     * @param {Array|Object} params - Parameters to bind
     * @returns {boolean} true on success
     */
    bind(params) {
        this.params = params;
        this.results = null;
        this.resultIndex = 0;
        this.done = false;
        return true;
    }

    /**
     * Execute and step to next row
     * @returns {boolean} true if there's a row, false if done
     */
    step() {
        if (this.done) return false;

        // Execute on first step
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

    /**
     * Get current row as array
     * @returns {Array} Current row values
     */
    get() {
        if (!this.results || this.resultIndex >= this.results.values.length) {
            return [];
        }
        const row = this.results.values[this.resultIndex];
        this.resultIndex++;
        return row;
    }

    /**
     * Get current row as object
     * @param {Object} params - Optional params (ignored, for compatibility)
     * @returns {Object} Current row as {column: value}
     */
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

    /**
     * Get column names
     * @returns {Array} Column names
     */
    getColumnNames() {
        return this.results?.columns || [];
    }

    /**
     * Reset statement for reuse
     * @returns {boolean} true on success
     */
    reset() {
        this.results = null;
        this.resultIndex = 0;
        this.done = false;
        return true;
    }

    /**
     * Free statement resources
     * @returns {boolean} true on success
     */
    free() {
        this.results = null;
        this.params = null;
        return true;
    }

    /**
     * Free and finalize (alias for free)
     */
    freemem() {
        return this.free();
    }
}

/**
 * Database class - sql.js compatible API with vector search
 *
 * Drop-in replacement for sql.js Database with:
 * - Same API: exec(), run(), prepare(), export(), close()
 * - OPFS persistence (automatic, no export/import needed)
 * - Vector search: NEAR, TOPK, embeddings
 * - Columnar Lance format for analytics
 *
 * @example
 * const SQL = await initSqlJs();
 * const db = new SQL.Database('mydb');
 *
 * // Standard SQL (same as sql.js)
 * db.exec("CREATE TABLE users (id INT, name TEXT)");
 * db.run("INSERT INTO users VALUES (?, ?)", [1, 'Alice']);
 * const results = db.exec("SELECT * FROM users");
 *
 * // Vector search (LanceQL extension)
 * db.exec("SELECT * FROM docs NEAR embedding 'search text' TOPK 10");
 */
class Database {
    /**
     * Create a new database
     * @param {string|Uint8Array} nameOrData - Database name (OPFS) or data (in-memory)
     * @param {OPFSStorage} storage - Optional storage backend
     */
    constructor(nameOrData, storage = null) {
        if (nameOrData instanceof Uint8Array) {
            // In-memory database from binary data (sql.js compatibility)
            this._inMemory = true;
            this._data = nameOrData;
            this._name = ':memory:';
            this._db = null;
        } else {
            // OPFS-persisted database
            this._inMemory = false;
            this._name = nameOrData || 'default';
            this._storage = storage;
            this._db = new LocalDatabase(this._name, storage || opfsStorage);
        }
        this._open = false;
        this._rowsModified = 0;
    }

    /**
     * Ensure database is open
     */
    async _ensureOpen() {
        if (!this._open && this._db) {
            await this._db.open();
            this._open = true;
        }
    }

    /**
     * Execute SQL and return results
     *
     * @param {string} sql - SQL statement(s)
     * @param {Array|Object} params - Optional parameters
     * @returns {Array} Array of {columns, values} result sets
     *
     * @example
     * const results = db.exec("SELECT * FROM users WHERE id = ?", [1]);
     * // [{columns: ['id', 'name'], values: [[1, 'Alice']]}]
     */
    exec(sql, params) {
        // Return promise for async operation
        return this._execAsync(sql, params);
    }

    async _execAsync(sql, params) {
        await this._ensureOpen();

        // Substitute parameters
        let processedSql = sql;
        if (params) {
            if (Array.isArray(params)) {
                let paramIndex = 0;
                processedSql = sql.replace(/\?/g, () => {
                    const val = params[paramIndex++];
                    return this._formatValue(val);
                });
            } else if (typeof params === 'object') {
                for (const [key, val] of Object.entries(params)) {
                    const pattern = new RegExp(`[:$@]${key}\\b`, 'g');
                    processedSql = processedSql.replace(pattern, this._formatValue(val));
                }
            }
        }

        // Split multiple statements
        const statements = processedSql
            .split(';')
            .map(s => s.trim())
            .filter(s => s.length > 0);

        const results = [];

        for (const stmt of statements) {
            try {
                const lexer = new SQLLexer(stmt);
                const tokens = lexer.tokenize();
                const parser = new SQLParser(tokens);
                const ast = parser.parse();

                if (ast.type === 'SELECT') {
                    // SELECT returns rows
                    const rows = await this._db._executeAST(ast);
                    if (rows && rows.length > 0) {
                        const columns = Object.keys(rows[0]);
                        const values = rows.map(row => columns.map(c => row[c]));
                        results.push({ columns, values });
                    }
                    this._rowsModified = 0;
                } else {
                    // Non-SELECT statements
                    const result = await this._db._executeAST(ast);
                    this._rowsModified = result?.inserted || result?.updated || result?.deleted || 0;
                }
            } catch (e) {
                throw new Error(`SQL error: ${e.message}\nStatement: ${stmt}`);
            }
        }

        return results;
    }

    /**
     * Execute SQL without returning results
     *
     * @param {string} sql - SQL statement
     * @param {Array|Object} params - Optional parameters
     * @returns {Database} this (for chaining)
     */
    run(sql, params) {
        return this._runAsync(sql, params);
    }

    async _runAsync(sql, params) {
        await this.exec(sql, params);
        return this;
    }

    /**
     * Prepare a statement for execution
     *
     * @param {string} sql - SQL statement
     * @param {Array|Object} params - Optional initial parameters
     * @returns {Statement} Prepared statement
     */
    prepare(sql, params) {
        const stmt = new Statement(this, sql);
        if (params) {
            stmt.bind(params);
        }
        return stmt;
    }

    /**
     * Execute SQL and call callback for each row
     *
     * @param {string} sql - SQL statement
     * @param {Array|Object} params - Parameters
     * @param {Function} callback - Called with row object for each row
     * @param {Function} done - Called when complete
     * @returns {Database} this
     */
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

    /**
     * Get number of rows modified by last statement
     * @returns {number} Rows modified
     */
    getRowsModified() {
        return this._rowsModified;
    }

    /**
     * Export database to Uint8Array
     *
     * For OPFS databases, this exports all tables as JSON.
     * For in-memory databases, returns the original data.
     *
     * @returns {Uint8Array} Database contents
     */
    async export() {
        if (this._inMemory && this._data) {
            return this._data;
        }

        await this._ensureOpen();

        // Export all tables as JSON
        const exportData = {
            version: this._db.version,
            tables: {}
        };

        for (const tableName of this._db.listTables()) {
            const table = this._db.getTable(tableName);
            const rows = await this._db.select(tableName, {});
            exportData.tables[tableName] = {
                schema: table.schema,
                rows
            };
        }

        return new TextEncoder().encode(JSON.stringify(exportData));
    }

    /**
     * Close the database
     */
    close() {
        this._open = false;
        this._db = null;
    }

    /**
     * Register a custom SQL function (stub for compatibility)
     * @param {string} name - Function name
     * @param {Function} func - Function implementation
     * @returns {Database} this
     */
    create_function(name, func) {
        console.warn(`[LanceQL] create_function('${name}') not yet implemented`);
        return this;
    }

    /**
     * Register a custom aggregate function (stub for compatibility)
     * @param {string} name - Function name
     * @param {Object} funcs - Aggregate functions {init, step, finalize}
     * @returns {Database} this
     */
    create_aggregate(name, funcs) {
        console.warn(`[LanceQL] create_aggregate('${name}') not yet implemented`);
        return this;
    }

    /**
     * Format a value for SQL
     */
    _formatValue(val) {
        if (val === null || val === undefined) {
            return 'NULL';
        }
        if (typeof val === 'string') {
            return `'${val.replace(/'/g, "''")}'`;
        }
        if (typeof val === 'number') {
            return String(val);
        }
        if (typeof val === 'boolean') {
            return val ? 'TRUE' : 'FALSE';
        }
        if (Array.isArray(val)) {
            // Vector
            return `'[${val.join(',')}]'`;
        }
        return String(val);
    }
}

/**
 * Initialize LanceQL with sql.js-compatible API
 *
 * Drop-in replacement for initSqlJs():
 *
 * @example
 * // sql.js style
 * import initSqlJs from 'sql.js';
 * const SQL = await initSqlJs();
 * const db = new SQL.Database();
 *
 * // LanceQL replacement
 * import { initSqlJs } from 'lanceql';
 * const SQL = await initSqlJs();
 * const db = new SQL.Database('mydb'); // OPFS-persisted + vector search
 *
 * @param {Object} config - Configuration (for compatibility, mostly ignored)
 * @returns {Promise<{Database: class}>} SQL namespace with Database class
 */
export async function initSqlJs(config = {}) {
    // Initialize OPFS storage
    try {
        await opfsStorage.open();
    } catch (e) {
        console.warn('[LanceQL] OPFS not available:', e.message);
    }

    // Initialize WebGPU for accelerated vector search, aggregations, joins, sorting, and grouping
    try {
        await webgpuAccelerator.init();
        await gpuAggregator.init();
        await gpuJoiner.init();
        await gpuSorter.init();
        await gpuGrouper.init();
        await gpuVectorSearch.init();
    } catch (e) {
        console.warn('[LanceQL] WebGPU not available:', e.message);
    }

    return {
        Database,
        Statement,
    };
}

// Also export as sqljs for explicit naming
export { Database as SqlJsDatabase, Statement as SqlJsStatement };

// Export WebGPU accelerator, GPU aggregator, GPU joiner, GPU sorter, and GPU grouper for direct access

// LogicTable exports (disabled - LanceDataset API not yet implemented)
// export { Table, logicTable, LogicTableQuery, loadLogicTable } from './logic-table.js';

export { LanceDataBase, OPFSLanceData, RemoteLanceData, DataFrame, LanceData, Statement, Database };
