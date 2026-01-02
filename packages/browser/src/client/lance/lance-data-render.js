/**
 * LanceData - CSS-driven rendering system
 * Extracted from lance-data.js for modularity
 */

// Forward declarations - resolved at runtime
let LanceQL, RemoteLanceDataset;

/**
 * LanceData - CSS-driven declarative data binding
 * Automatically binds SQL queries to DOM elements via CSS triggers.
 */
class LanceData {
    static _initialized = false;
    static _observer = null;
    static _wasm = null;
    static _datasets = new Map();
    static _renderers = {};
    static _bindings = new Map();
    static _queryCache = new Map();
    static _defaultDataset = null;

    /**
     * Auto-initialize when DOM is ready.
     */
    static _autoInit() {
        if (LanceData._initialized) return;
        LanceData._initialized = true;

        LanceData._registerBuiltinRenderers();
        LanceData._injectTriggerStyles();
        LanceData._setupObserver();
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

        // Load dependencies
        if (!LanceQL) {
            const wasmModule = await import('../wasm/lanceql.js');
            LanceQL = wasmModule.LanceQL;
        }
        if (!RemoteLanceDataset) {
            const datasetModule = await import('./remote-dataset.js');
            RemoteLanceDataset = datasetModule.RemoteLanceDataset;
        }

        // Load WASM if needed
        if (!LanceData._wasm) {
            const wasmUrl = document.querySelector('script[data-lanceql-wasm]')?.dataset.lanceqlWasm
                || './lanceql.wasm';
            LanceData._wasm = await LanceQL.load(wasmUrl);
        }

        const dataset = await RemoteLanceDataset.open(LanceData._wasm, url);
        LanceData._datasets.set(url, dataset);

        if (!LanceData._defaultDataset) {
            LanceData._defaultDataset = url;
        }

        return dataset;
    }

    /**
     * Manual init (optional).
     */
    static async init(options = {}) {
        LanceData._autoInit();

        if (options.wasmUrl) {
            if (!LanceQL) {
                const wasmModule = await import('../wasm/lanceql.js');
                LanceQL = wasmModule.LanceQL;
            }
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
        if (LanceData._observer) return;

        const hasLqAttrs = (el) => {
            return el.hasAttribute?.('lq-query') || el.hasAttribute?.('lq-src') ||
                   el.classList?.contains('lance-data');
        };

        LanceData._observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        if (hasLqAttrs(node)) {
                            LanceData._processElement(node);
                        }
                        node.querySelectorAll?.('[lq-query], [lq-src], .lance-data')?.forEach(el => {
                            LanceData._processElement(el);
                        });
                    }
                }

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
     * Parse config from attributes.
     */
    static _parseConfig(el) {
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
     * Render pre-computed results to an element.
     */
    static render(el, results, options = {}) {
        const element = typeof el === 'string' ? document.querySelector(el) : el;
        if (!element) {
            console.error('[LanceData] Element not found:', el);
            return;
        }

        try {
            element.dispatchEvent(new CustomEvent('lq-start', {
                detail: { query: options.query || null }
            }));

            const renderType = options.render || element.dataset.render || 'table';
            const renderer = LanceData._renderers[renderType] || LanceData._renderers.table;

            if (element.id) {
                LanceData._queryCache.set(`rendered:${element.id}`, results);
            }

            element.innerHTML = renderer(results, { render: renderType, ...options });

            element.dispatchEvent(new CustomEvent('lq-complete', {
                detail: {
                    query: options.query || null,
                    columns: results.columns || [],
                    total: results.total || results.rows?.length || 0
                }
            }));
        } catch (error) {
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
        if (el.dataset.processing === 'true') return;
        el.dataset.processing = 'true';

        let config;
        try {
            config = LanceData._parseConfig(el);

            if (!config.query) {
                el.dataset.processing = 'false';
                return;
            }

            if (config.bind) {
                LanceData._setupBinding(el, config);
            }

            el.dataset.loading = 'true';
            delete el.dataset.error;

            el.dispatchEvent(new CustomEvent('lq-start', {
                detail: { query: config.query }
            }));

            const datasetUrl = config.dataset || LanceData._extractUrlFromQuery(config.query);
            const dataset = await LanceData._getDataset(datasetUrl);

            const cacheKey = `${datasetUrl || 'default'}:${config.query}`;
            let results = LanceData._queryCache.get(cacheKey);

            if (!results) {
                results = await dataset.executeSQL(config.query);
                LanceData._queryCache.set(cacheKey, results);
            }

            const renderer = LanceData._renderers[config.render] || LanceData._renderers.table;
            el.innerHTML = renderer(results, config);

            delete el.dataset.loading;

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

            el.dispatchEvent(new CustomEvent('lq-error', {
                detail: {
                    query: config?.query,
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

        const bindingKey = config.bind;
        if (LanceData._bindings.has(bindingKey)) return;

        const handler = () => {
            const value = input.value;
            const newQuery = config.query.replace(/\$value/g, value);

            if (el.hasAttribute('lq-query')) {
                el.setAttribute('lq-query', newQuery);
            } else {
                el.dataset.query = newQuery;
            }

            el.dataset.refresh = Date.now();
        };

        input.addEventListener('input', handler);
        input.addEventListener('change', handler);

        LanceData._bindings.set(bindingKey, { input, handler, element: el });
    }

    /**
     * Register a custom renderer.
     */
    static registerRenderer(name, fn) {
        LanceData._renderers[name] = fn;
    }

    /**
     * Register built-in renderers.
     */
    static _registerBuiltinRenderers() {
        // Table renderer
        LanceData._renderers.table = (results, config) => {
            if (!results) {
                return '<div class="lance-empty">No results</div>';
            }

            let columns, rows;
            if (results.columns && results.rows) {
                columns = config.columns?.length ? config.columns : results.columns.filter(k =>
                    !k.startsWith('_') && k !== 'embedding'
                );
                rows = results.rows;
            } else if (Array.isArray(results)) {
                if (results.length === 0) {
                    return '<div class="lance-empty">No results</div>';
                }
                columns = config.columns?.length ? config.columns : Object.keys(results[0]).filter(k =>
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

        // Value renderer
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

        // Image grid renderer
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

        // Count renderer
        LanceData._renderers.count = (results, config) => {
            const count = results?.[0]?.count ?? results?.length ?? 0;
            return `<span class="lance-count">${count.toLocaleString()}</span>`;
        };
    }

    static _isImageUrl(str) {
        if (!str || typeof str !== 'string') return false;
        const lower = str.toLowerCase();
        return (lower.startsWith('http://') || lower.startsWith('https://')) &&
               (lower.includes('.jpg') || lower.includes('.jpeg') || lower.includes('.png') ||
                lower.includes('.gif') || lower.includes('.webp') || lower.includes('.svg'));
    }

    static _isUrl(str) {
        if (!str || typeof str !== 'string') return false;
        return str.startsWith('http://') || str.startsWith('https://');
    }

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

        if (LanceData._isImageUrl(str)) {
            const escaped = LanceData._escapeHtml(str);
            const short = escaped.length > 40 ? escaped.substring(0, 40) + '...' : escaped;
            return `<div class="image-cell">
                <img src="${escaped}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
                <div class="image-placeholder" style="display:none"><svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg></div>
                <a href="${escaped}" target="_blank" class="url-text" title="${escaped}">${short}</a>
            </div>`;
        }

        if (LanceData._isUrl(str)) {
            const escaped = LanceData._escapeHtml(str);
            const short = escaped.length > 50 ? escaped.substring(0, 50) + '...' : escaped;
            return `<a href="${escaped}" target="_blank" class="url-link" title="${escaped}">${short}</a>`;
        }

        if (str.length > 100) return `<span title="${LanceData._escapeHtml(str)}">${LanceData._escapeHtml(str.substring(0, 100))}...</span>`;
        return LanceData._escapeHtml(str);
    }

    static _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    static clearCache() {
        LanceData._queryCache.clear();
    }

    static refresh() {
        LanceData._queryCache.clear();
        document.querySelectorAll('.lance-data').forEach(el => {
            el.setAttribute('data-refresh', Date.now());
        });
    }

    static destroy() {
        if (LanceData._observer) {
            LanceData._observer.disconnect();
            LanceData._observer = null;
        }

        for (const [key, binding] of LanceData._bindings) {
            binding.input.removeEventListener('input', binding.handler);
            binding.input.removeEventListener('change', binding.handler);
        }
        LanceData._bindings.clear();

        document.getElementById('lance-data-triggers')?.remove();

        LanceData._instance = null;
        LanceData._dataset = null;
        LanceData._queryCache.clear();
    }
}

// Auto-initialize when DOM is ready
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => LanceData._autoInit());
    } else {
        LanceData._autoInit();
    }
}


export { LanceData };
