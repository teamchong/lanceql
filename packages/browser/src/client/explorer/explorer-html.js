/**
 * Self-contained HTML for the Explorer window
 * Includes all CSS and JS inline for popup isolation
 */

export const explorerHTML = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LanceQL Explorer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #475569;
            --row-height: 32px;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Header */
        .header {
            display: flex;
            align-items: center;
            padding: 8px 16px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            gap: 16px;
        }

        .logo {
            font-weight: 700;
            font-size: 14px;
            color: var(--accent);
        }

        .table-select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-primary);
            padding: 6px 12px;
            font-size: 13px;
            min-width: 200px;
        }

        .table-select:focus {
            outline: none;
            border-color: var(--accent);
        }

        .row-count {
            color: var(--text-muted);
            font-size: 12px;
            margin-left: auto;
        }

        /* Tabs */
        .tabs {
            display: flex;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
        }

        .tab {
            padding: 10px 20px;
            font-size: 13px;
            color: var(--text-secondary);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.15s;
        }

        .tab:hover {
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }

        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }

        /* Main content */
        .content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .tab-panel {
            display: none;
            flex: 1;
            overflow: hidden;
        }

        .tab-panel.active {
            display: flex;
            flex-direction: column;
        }

        /* SQL Tab */
        .sql-editor {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .sql-input {
            background: var(--bg-secondary);
            border: none;
            border-bottom: 1px solid var(--border);
            color: var(--text-primary);
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            font-size: 13px;
            padding: 12px;
            resize: none;
            height: 120px;
        }

        .sql-input:focus {
            outline: none;
        }

        .sql-toolbar {
            display: flex;
            padding: 8px 12px;
            gap: 8px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
        }

        .btn {
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 16px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.15s;
        }

        .btn:hover {
            background: var(--accent-hover);
        }

        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }

        .btn-secondary:hover {
            background: var(--border);
            color: var(--text-primary);
        }

        .sql-status {
            font-size: 12px;
            color: var(--text-muted);
            margin-left: auto;
        }

        .sql-status.error {
            color: var(--error);
        }

        .sql-status.success {
            color: var(--success);
        }

        /* Virtual Table */
        .virtual-table-container {
            flex: 1;
            overflow: hidden;
            position: relative;
        }

        .virtual-table {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            overflow: auto;
        }

        .table-header {
            display: flex;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .table-header-cell {
            padding: 8px 12px;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            min-width: 120px;
            border-right: 1px solid var(--border);
        }

        .table-body {
            position: relative;
        }

        .table-row {
            display: flex;
            height: var(--row-height);
            border-bottom: 1px solid var(--bg-tertiary);
        }

        .table-row:hover {
            background: var(--bg-secondary);
        }

        .table-cell {
            padding: 6px 12px;
            font-size: 12px;
            font-family: 'SF Mono', Monaco, monospace;
            min-width: 120px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            border-right: 1px solid var(--bg-tertiary);
            display: flex;
            align-items: center;
        }

        .table-cell.null {
            color: var(--text-muted);
            font-style: italic;
        }

        /* Timeline Tab */
        .timeline-container {
            padding: 20px;
            overflow-y: auto;
        }

        .timeline-slider {
            width: 100%;
            margin: 20px 0;
        }

        .timeline-versions {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .timeline-version {
            display: flex;
            align-items: center;
            padding: 12px;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border);
            cursor: pointer;
            transition: all 0.15s;
        }

        .timeline-version:hover {
            border-color: var(--accent);
        }

        .timeline-version.active {
            border-color: var(--accent);
            background: rgba(59, 130, 246, 0.1);
        }

        .version-number {
            font-weight: 700;
            font-size: 14px;
            width: 60px;
        }

        .version-meta {
            flex: 1;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .version-delta {
            font-weight: 600;
            font-size: 13px;
        }

        .version-delta.add {
            color: var(--success);
        }

        .version-delta.delete {
            color: var(--error);
        }

        /* Diff View */
        .diff-view {
            display: flex;
            gap: 20px;
            padding: 20px;
            overflow-y: auto;
        }

        .diff-panel {
            flex: 1;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }

        .diff-header {
            padding: 12px;
            font-weight: 600;
            font-size: 13px;
            border-bottom: 1px solid var(--border);
        }

        .diff-header.added {
            background: rgba(34, 197, 94, 0.1);
            color: var(--success);
        }

        .diff-header.deleted {
            background: rgba(239, 68, 68, 0.1);
            color: var(--error);
        }

        /* Loading */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-muted);
        }

        .spinner {
            width: 24px;
            height: 24px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 12px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Empty state */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-muted);
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <span class="logo">LanceQL Explorer</span>
        <select class="table-select" id="tableSelect">
            <option value="">Select a table...</option>
        </select>
        <span class="row-count" id="rowCount"></span>
    </div>

    <div class="tabs">
        <div class="tab active" data-tab="sql">SQL</div>
        <div class="tab" data-tab="dataview">DataView</div>
        <div class="tab" data-tab="timeline">Timeline</div>
    </div>

    <div class="content">
        <!-- SQL Tab -->
        <div class="tab-panel active" id="tab-sql">
            <div class="sql-editor">
                <textarea class="sql-input" id="sqlInput" placeholder="SELECT * FROM table LIMIT 100"></textarea>
                <div class="sql-toolbar">
                    <button class="btn" id="runSql">Run (Ctrl+Enter)</button>
                    <button class="btn btn-secondary" id="clearSql">Clear</button>
                    <span class="sql-status" id="sqlStatus"></span>
                </div>
                <div class="virtual-table-container" id="sqlResults">
                    <div class="empty-state">Run a query to see results</div>
                </div>
            </div>
        </div>

        <!-- DataView Tab -->
        <div class="tab-panel" id="tab-dataview">
            <div class="virtual-table-container" id="dataviewContainer">
                <div class="empty-state">Select a table to view data</div>
            </div>
        </div>

        <!-- Timeline Tab -->
        <div class="tab-panel" id="tab-timeline">
            <div class="timeline-container" id="timelineContainer">
                <div class="empty-state">Select a table to view version history</div>
            </div>
        </div>
    </div>

    <script>
        // =====================================================================
        // Explorer Logic
        // =====================================================================

        const state = {
            tables: [],
            currentTable: null,
            currentVersion: null,
            totalRows: 0,
            columns: [],
            cache: new Map(), // row cache
            pendingRequests: new Map(),
            requestId: 0
        };

        // Elements
        const tableSelect = document.getElementById('tableSelect');
        const rowCount = document.getElementById('rowCount');
        const sqlInput = document.getElementById('sqlInput');
        const sqlStatus = document.getElementById('sqlStatus');
        const sqlResults = document.getElementById('sqlResults');
        const dataviewContainer = document.getElementById('dataviewContainer');
        const timelineContainer = document.getElementById('timelineContainer');

        // =====================================================================
        // Communication with parent window
        // =====================================================================

        function sendMessage(type, params = {}) {
            return new Promise((resolve, reject) => {
                const id = ++state.requestId;
                state.pendingRequests.set(id, { resolve, reject });
                window.opener.postMessage({ type, id, ...params }, '*');

                // Timeout after 30s
                setTimeout(() => {
                    if (state.pendingRequests.has(id)) {
                        state.pendingRequests.delete(id);
                        reject(new Error('Request timeout'));
                    }
                }, 30000);
            });
        }

        window.addEventListener('message', (event) => {
            const { type, id, result, error, ...data } = event.data || {};

            if (type === 'response') {
                const pending = state.pendingRequests.get(id);
                if (pending) {
                    state.pendingRequests.delete(id);
                    if (error) {
                        pending.reject(new Error(error));
                    } else {
                        pending.resolve(result);
                    }
                }
            } else if (type === 'init') {
                initExplorer(data);
            } else if (type === 'run-sql') {
                sqlInput.value = data.sql;
                runQuery();
            } else if (type === 'select-table') {
                tableSelect.value = data.table;
                loadTable(data.table);
            }
        });

        // Signal ready
        window.opener?.postMessage({ type: 'ready' }, '*');

        // =====================================================================
        // Initialization
        // =====================================================================

        function initExplorer(data) {
            state.tables = data.tables || [];

            // Populate table select
            tableSelect.innerHTML = '<option value="">Select a table...</option>';
            for (const table of state.tables) {
                const opt = document.createElement('option');
                opt.value = table;
                opt.textContent = table;
                tableSelect.appendChild(opt);
            }

            // Auto-select table if provided
            if (data.options?.table) {
                tableSelect.value = data.options.table;
                loadTable(data.options.table);
            }
        }

        // =====================================================================
        // Tab switching
        // =====================================================================

        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

                tab.classList.add('active');
                document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
            });
        });

        // =====================================================================
        // Table selection
        // =====================================================================

        tableSelect.addEventListener('change', () => {
            const table = tableSelect.value;
            if (table) loadTable(table);
        });

        async function loadTable(table) {
            state.currentTable = table;
            state.cache.clear();

            try {
                // Get schema and count
                const [schema, count] = await Promise.all([
                    sendMessage('fetch-schema', { table }),
                    sendMessage('fetch-count', { table })
                ]);

                state.columns = schema.columns;
                state.totalRows = count;
                rowCount.textContent = count.toLocaleString() + ' rows';

                // Render DataView
                renderVirtualTable(dataviewContainer, table);

                // Load timeline
                loadTimeline(table);

            } catch (e) {
                console.error('Failed to load table:', e);
                rowCount.textContent = 'Error loading table';
            }
        }

        // =====================================================================
        // Virtual Table Rendering
        // =====================================================================

        function renderVirtualTable(container, table, data = null) {
            const ROW_HEIGHT = 32;
            const BUFFER_ROWS = 5;
            const MIN_VISIBLE = 20;
            const MAX_VISIBLE = 200;

            // Calculate visible rows based on container height
            const containerHeight = container.clientHeight || 400;
            const calculatedRows = Math.ceil(containerHeight / ROW_HEIGHT);
            const VISIBLE_ROWS = Math.min(MAX_VISIBLE, Math.max(MIN_VISIBLE, calculatedRows + BUFFER_ROWS * 2));

            if (!state.columns.length && !data) {
                container.innerHTML = '<div class="empty-state">No columns</div>';
                return;
            }

            const columns = data?.columns || state.columns;
            const totalRows = data?.rows?.length || state.totalRows;

            container.innerHTML = \`
                <div class="virtual-table" id="virtualTable">
                    <div class="table-header">
                        \${columns.map(col => \`<div class="table-header-cell">\${col}</div>\`).join('')}
                    </div>
                    <div class="table-body" style="height: \${totalRows * ROW_HEIGHT}px">
                    </div>
                </div>
            \`;

            const virtualTable = container.querySelector('#virtualTable');
            const tableBody = container.querySelector('.table-body');

            let lastScrollTop = 0;
            let renderedStart = -1;
            let renderedEnd = -1;

            async function renderRows() {
                const scrollTop = virtualTable.scrollTop;
                const startRow = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER_ROWS);
                const endRow = Math.min(totalRows, startRow + VISIBLE_ROWS);

                // Skip if same range
                if (startRow === renderedStart && endRow === renderedEnd) return;

                renderedStart = startRow;
                renderedEnd = endRow;

                // Fetch rows if needed
                let rows;
                if (data) {
                    rows = data.rows.slice(startRow, endRow);
                } else {
                    const cacheKey = \`\${startRow}-\${endRow}\`;
                    if (state.cache.has(cacheKey)) {
                        rows = state.cache.get(cacheKey);
                    } else {
                        try {
                            const result = await sendMessage('fetch-rows', {
                                table,
                                offset: startRow,
                                limit: endRow - startRow
                            });
                            rows = result?.rows || [];
                            state.cache.set(cacheKey, rows);
                        } catch (e) {
                            console.error('Failed to fetch rows:', e);
                            rows = [];
                        }
                    }
                }

                // Render rows
                const rowsHtml = rows.map((row, i) => {
                    const rowIndex = startRow + i;
                    const cells = columns.map((col, j) => {
                        const value = row[j];
                        const isNull = value === null || value === undefined;
                        return \`<div class="table-cell \${isNull ? 'null' : ''}">\${isNull ? 'NULL' : escapeHtml(String(value))}</div>\`;
                    }).join('');
                    return \`<div class="table-row" style="position: absolute; top: \${rowIndex * ROW_HEIGHT}px; width: 100%">\${cells}</div>\`;
                }).join('');

                tableBody.innerHTML = rowsHtml;
            }

            virtualTable.addEventListener('scroll', () => {
                requestAnimationFrame(renderRows);
            });

            // Re-render on resize
            const resizeObserver = new ResizeObserver(() => {
                requestAnimationFrame(renderRows);
            });
            resizeObserver.observe(container);

            // Initial render
            renderRows();
        }

        // =====================================================================
        // SQL Execution
        // =====================================================================

        async function runQuery() {
            const sql = sqlInput.value.trim();
            if (!sql) return;

            sqlStatus.textContent = 'Running...';
            sqlStatus.className = 'sql-status';

            const startTime = performance.now();

            try {
                const result = await sendMessage('exec', { sql });
                const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

                if (result?.columns && result?.rows) {
                    sqlStatus.textContent = \`\${result.rows.length} rows in \${elapsed}s\`;
                    sqlStatus.className = 'sql-status success';

                    // Render results
                    state.columns = result.columns;
                    renderVirtualTable(sqlResults, null, result);
                } else {
                    sqlStatus.textContent = \`Done in \${elapsed}s\`;
                    sqlStatus.className = 'sql-status success';
                    sqlResults.innerHTML = '<div class="empty-state">Query executed successfully</div>';
                }
            } catch (e) {
                sqlStatus.textContent = e.message;
                sqlStatus.className = 'sql-status error';
            }
        }

        document.getElementById('runSql').addEventListener('click', runQuery);
        document.getElementById('clearSql').addEventListener('click', () => {
            sqlInput.value = '';
            sqlResults.innerHTML = '<div class="empty-state">Run a query to see results</div>';
            sqlStatus.textContent = '';
        });

        sqlInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                runQuery();
            }
        });

        // =====================================================================
        // Timeline
        // =====================================================================

        async function loadTimeline(table) {
            try {
                const versions = await sendMessage('timeline', { table });
                renderTimeline(versions);
            } catch (e) {
                timelineContainer.innerHTML = '<div class="empty-state">Failed to load timeline</div>';
            }
        }

        function renderTimeline(versions) {
            if (!versions.length) {
                timelineContainer.innerHTML = '<div class="empty-state">No version history</div>';
                return;
            }

            timelineContainer.innerHTML = \`
                <h3 style="margin-bottom: 16px; font-size: 14px; color: var(--text-secondary)">Version History</h3>
                <div class="timeline-versions">
                    \${versions.map(v => \`
                        <div class="timeline-version" data-version="\${v.version}">
                            <span class="version-number">v\${v.version}</span>
                            <span class="version-meta">\${v.operation} - \${new Date(v.timestamp).toLocaleString()}</span>
                            <span class="version-delta \${v.delta.startsWith('+') ? 'add' : 'delete'}">\${v.delta}</span>
                        </div>
                    \`).join('')}
                </div>
            \`;

            // Click to view version
            timelineContainer.querySelectorAll('.timeline-version').forEach(el => {
                el.addEventListener('click', () => {
                    const version = el.dataset.version;
                    document.querySelectorAll('.timeline-version').forEach(v => v.classList.remove('active'));
                    el.classList.add('active');

                    // Switch to dataview with this version
                    state.currentVersion = parseInt(version);
                    document.querySelector('[data-tab="dataview"]').click();
                    // Re-render with version
                    if (state.currentTable) {
                        sqlInput.value = \`SELECT * FROM \${state.currentTable} VERSION AS OF \${version} LIMIT 100\`;
                    }
                });
            });
        }

        // =====================================================================
        // Utils
        // =====================================================================

        function escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }
    </script>
</body>
</html>`;
