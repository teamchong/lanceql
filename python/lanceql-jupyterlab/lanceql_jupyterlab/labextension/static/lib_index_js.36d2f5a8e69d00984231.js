"use strict";
(self["webpackChunklanceql_jupyterlab"] = self["webpackChunklanceql_jupyterlab"] || []).push([["lib_index_js"],{

/***/ "./lib/index.js"
/*!**********************!*\
  !*** ./lib/index.js ***!
  \**********************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/rendermime */ "webpack/sharing/consume/default/@jupyterlab/rendermime");
/* harmony import */ var _jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/notebook */ "webpack/sharing/consume/default/@jupyterlab/notebook");
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _lumino_widgets__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @lumino/widgets */ "webpack/sharing/consume/default/@lumino/widgets");
/* harmony import */ var _lumino_widgets__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_lumino_widgets__WEBPACK_IMPORTED_MODULE_2__);



/**
 * The MIME type for LanceQL virtual tables.
 */
const MIME_TYPE = 'application/vnd.lanceql.table+json';
/**
 * Global kernel reference - set by the plugin when notebooks are active.
 */
let _currentKernel = null;
/**
 * Global image preview element - shared across all tables.
 */
let _globalPreview = null;
function getGlobalPreview() {
    if (!_globalPreview) {
        _globalPreview = document.createElement('div');
        _globalPreview.className = 'lq-global-preview';
        _globalPreview.style.cssText = `
      display: none;
      position: fixed;
      z-index: 99999;
      background: white;
      border: 1px solid #ccc;
      border-radius: 4px;
      padding: 8px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.25);
      pointer-events: none;
    `;
        document.body.appendChild(_globalPreview);
    }
    return _globalPreview;
}
/**
 * Virtual table renderer widget.
 */
class VirtualTableWidget extends _lumino_widgets__WEBPACK_IMPORTED_MODULE_2__.Widget {
    constructor(options) {
        super();
        this._data = null;
        this._cache = new Map();
        this._comm = null;
        this._pendingRanges = new Set();
        this._scrollContainer = null;
        this._rowsContainer = null;
        this._statusEl = null;
        this.ROW_HEIGHT = 32;
        this.CHUNK_SIZE = 50;
        this.BUFFER = 10;
        this.addClass('lanceql-virtual-table');
    }
    async renderModel(model) {
        const data = model.data[MIME_TYPE];
        if (!data) {
            this.node.textContent = 'No data';
            return;
        }
        this._data = data;
        this._cache.set(0, data.rows);
        this._buildUI();
        await this._initComm();
        this._render();
    }
    _buildUI() {
        var _a;
        if (!this._data)
            return;
        const container = document.createElement('div');
        container.className = 'lq-container';
        // Header with type badges
        const header = document.createElement('div');
        header.className = 'lq-header';
        for (const col of this._data.columns) {
            const cell = document.createElement('span');
            cell.className = 'lq-cell';
            const colName = document.createElement('span');
            colName.className = 'lq-col-name';
            colName.textContent = col;
            cell.appendChild(colName);
            // Add type badge
            const colType = (_a = this._data.column_types) === null || _a === void 0 ? void 0 : _a[col];
            if (colType) {
                const badge = document.createElement('span');
                badge.className = `lq-type-badge lq-type-${colType.type}`;
                if (colType.type === 'vector' && colType.model) {
                    badge.textContent = colType.model;
                    badge.title = `${colType.dim}d vector (${colType.dtype})`;
                }
                else {
                    badge.textContent = colType.type;
                    badge.title = colType.dtype;
                }
                cell.appendChild(badge);
            }
            header.appendChild(cell);
        }
        container.appendChild(header);
        // Scroll container
        this._scrollContainer = document.createElement('div');
        this._scrollContainer.className = 'lq-scroll';
        const spacer = document.createElement('div');
        spacer.className = 'lq-spacer';
        spacer.style.height = `${this._data.total * this.ROW_HEIGHT}px`;
        this._rowsContainer = document.createElement('div');
        this._rowsContainer.className = 'lq-rows';
        spacer.appendChild(this._rowsContainer);
        this._scrollContainer.appendChild(spacer);
        container.appendChild(this._scrollContainer);
        // Status bar
        this._statusEl = document.createElement('div');
        this._statusEl.className = 'lq-status';
        this._statusEl.textContent = `${this._data.total.toLocaleString()} rows`;
        container.appendChild(this._statusEl);
        this.node.appendChild(container);
        // Scroll listener
        let scrollTimeout = null;
        this._scrollContainer.addEventListener('scroll', () => {
            if (scrollTimeout)
                window.clearTimeout(scrollTimeout);
            scrollTimeout = window.setTimeout(() => this._render(), 16);
        });
    }
    async _initComm() {
        if (!this._data) {
            this._updateStatus('(no data)');
            return;
        }
        // Try to get kernel - may need to wait for it
        let kernel = _currentKernel;
        if (!kernel) {
            // Wait a bit for kernel to be available
            await new Promise(resolve => setTimeout(resolve, 500));
            kernel = _currentKernel;
        }
        if (!kernel) {
            this._updateStatus('(static mode - no kernel)');
            console.warn('LanceQL: No kernel available for Comm');
            return;
        }
        try {
            this._comm = kernel.createComm('lanceql_display');
            this._comm.onMsg = (msg) => {
                const data = msg.content.data;
                if (data.chunks) {
                    for (const chunk of data.chunks) {
                        this._cache.set(chunk.offset, chunk.rows);
                        this._pendingRanges.delete(chunk.offset);
                    }
                    this._render();
                }
            };
            await this._comm.open({ source_id: this._data.source_id });
            this._updateStatus('');
        }
        catch (e) {
            console.warn('LanceQL: Comm init failed:', e);
            this._updateStatus('(comm failed)');
        }
    }
    _updateStatus(extra) {
        if (this._statusEl && this._data) {
            const base = `${this._data.total.toLocaleString()} rows`;
            this._statusEl.textContent = extra ? `${base} ${extra}` : base;
        }
    }
    _render() {
        var _a;
        if (!this._data || !this._scrollContainer || !this._rowsContainer)
            return;
        const scrollTop = this._scrollContainer.scrollTop;
        const viewHeight = this._scrollContainer.clientHeight;
        const startRow = Math.max(0, Math.floor(scrollTop / this.ROW_HEIGHT) - this.BUFFER);
        const visibleCount = Math.ceil(viewHeight / this.ROW_HEIGHT) + this.BUFFER * 2;
        const endRow = Math.min(this._data.total, startRow + visibleCount);
        let html = '';
        for (let i = startRow; i < endRow; i++) {
            const row = this._getRow(i);
            if (row) {
                html += `<div class="lq-row" style="top:${i * this.ROW_HEIGHT}px">`;
                for (const col of this._data.columns) {
                    const val = row[col];
                    const colType = (_a = this._data.column_types) === null || _a === void 0 ? void 0 : _a[col];
                    if (val === null || val === undefined) {
                        html += '<span class="lq-cell"></span>';
                    }
                    else if ((colType === null || colType === void 0 ? void 0 : colType.type) === 'vector' && Array.isArray(val)) {
                        // Render sparkline for vectors
                        const sparkline = this._renderSparkline(val, colType.dim || val.length);
                        html += `<span class="lq-cell lq-vector-cell" data-vector="${this._escapeHtml(JSON.stringify(val))}" data-dim="${colType.dim || val.length}" data-model="${colType.model || ''}">`;
                        html += sparkline;
                        html += '</span>';
                    }
                    else if (this._data.image_columns.includes(col) && typeof val === 'string') {
                        const shortUrl = val.length > 30 ? val.substring(0, 30) + '...' : val;
                        html += `<span class="lq-cell lq-img-cell" data-img-url="${this._escapeHtml(val)}">`;
                        html += `<a href="${this._escapeHtml(val)}" target="_blank" title="${this._escapeHtml(val)}">${this._escapeHtml(shortUrl)}</a>`;
                        html += '</span>';
                    }
                    else {
                        html += `<span class="lq-cell">${this._escapeHtml(String(val))}</span>`;
                    }
                }
                html += '</div>';
            }
            else {
                html += `<div class="lq-row lq-loading" style="top:${i * this.ROW_HEIGHT}px">`;
                html += '<span class="lq-cell">Loading...</span>';
                html += '</div>';
            }
        }
        this._rowsContainer.innerHTML = html;
        this._setupImagePreviews();
        this._setupVectorPreviews();
        this._fetchMissing(startRow, endRow);
    }
    _getRow(idx) {
        const chunkStart = Math.floor(idx / this.CHUNK_SIZE) * this.CHUNK_SIZE;
        const rows = this._cache.get(chunkStart);
        if (!rows)
            return null;
        const localIdx = idx - chunkStart;
        const result = {};
        for (const col of this._data.columns) {
            const colData = rows[col];
            result[col] = colData ? colData[localIdx] : null;
        }
        return result;
    }
    _fetchMissing(start, end) {
        if (!this._comm || !this._data)
            return;
        const ranges = [];
        for (let chunk = Math.floor(start / this.CHUNK_SIZE) * this.CHUNK_SIZE; chunk < end; chunk += this.CHUNK_SIZE) {
            if (!this._cache.has(chunk) && !this._pendingRanges.has(chunk)) {
                ranges.push({ offset: chunk, limit: this.CHUNK_SIZE });
                this._pendingRanges.add(chunk);
            }
        }
        if (ranges.length > 0) {
            this._comm.send({
                action: 'fetch_rows',
                source_id: this._data.source_id,
                ranges: ranges
            });
        }
    }
    _escapeHtml(s) {
        return s
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }
    _renderSparkline(values, dim) {
        // Sample values if too many
        const maxPoints = 60;
        let sampled = values;
        if (values.length > maxPoints) {
            const step = Math.floor(values.length / maxPoints);
            sampled = [];
            for (let i = 0; i < values.length; i += step) {
                sampled.push(values[i]);
            }
        }
        // Normalize values to 0-20 range for SVG height
        const min = Math.min(...sampled);
        const max = Math.max(...sampled);
        const range = max - min || 1;
        const normalized = sampled.map(v => 20 - ((v - min) / range) * 18);
        // Build SVG path
        const width = 80;
        const step = width / (normalized.length - 1 || 1);
        let path = `M0,${normalized[0].toFixed(1)}`;
        for (let i = 1; i < normalized.length; i++) {
            path += ` L${(i * step).toFixed(1)},${normalized[i].toFixed(1)}`;
        }
        return `<svg class="lq-sparkline" width="80" height="20" viewBox="0 0 80 20">
      <path d="${path}" fill="none" stroke="#6366f1" stroke-width="1.5"/>
    </svg><span class="lq-dim-label">${dim}d</span>`;
    }
    _setupImagePreviews() {
        if (!this._rowsContainer)
            return;
        const preview = getGlobalPreview();
        const imgCells = this._rowsContainer.querySelectorAll('.lq-img-cell');
        imgCells.forEach((cell) => {
            const url = cell.dataset.imgUrl;
            if (!url)
                return;
            cell.addEventListener('mouseenter', () => {
                preview.innerHTML = `<img src="${url}" style="max-width:300px;max-height:200px;display:block;">`;
                preview.style.display = 'block';
            });
            cell.addEventListener('mousemove', (e) => {
                const me = e;
                preview.style.left = `${me.clientX + 15}px`;
                preview.style.top = `${me.clientY + 15}px`;
            });
            cell.addEventListener('mouseleave', () => {
                preview.style.display = 'none';
            });
        });
    }
    _setupVectorPreviews() {
        if (!this._rowsContainer)
            return;
        const preview = getGlobalPreview();
        const vectorCells = this._rowsContainer.querySelectorAll('.lq-vector-cell');
        vectorCells.forEach((cell) => {
            const vectorStr = cell.dataset.vector;
            const dim = cell.dataset.dim;
            const model = cell.dataset.model;
            if (!vectorStr)
                return;
            cell.addEventListener('click', () => {
                try {
                    const vector = JSON.parse(vectorStr);
                    const stats = this._computeVectorStats(vector);
                    preview.innerHTML = `
            <div class="lq-vector-stats">
              <div class="lq-stats-header">${model || `${dim}d vector`}</div>
              <div class="lq-stats-row"><span>Dimension:</span><span>${vector.length}</span></div>
              <div class="lq-stats-row"><span>Norm (L2):</span><span>${stats.norm.toFixed(4)}</span></div>
              <div class="lq-stats-row"><span>Mean:</span><span>${stats.mean.toFixed(4)}</span></div>
              <div class="lq-stats-row"><span>Std:</span><span>${stats.std.toFixed(4)}</span></div>
              <div class="lq-stats-row"><span>Min:</span><span>${stats.min.toFixed(4)}</span></div>
              <div class="lq-stats-row"><span>Max:</span><span>${stats.max.toFixed(4)}</span></div>
              <div class="lq-stats-row"><span>Sparsity:</span><span>${(stats.sparsity * 100).toFixed(1)}%</span></div>
            </div>
          `;
                    preview.style.display = 'block';
                    // Position near the cell
                    const rect = cell.getBoundingClientRect();
                    preview.style.left = `${rect.right + 10}px`;
                    preview.style.top = `${rect.top}px`;
                }
                catch (e) {
                    console.warn('Failed to parse vector:', e);
                }
            });
            // Close on click outside
            document.addEventListener('click', (e) => {
                if (!cell.contains(e.target) && !preview.contains(e.target)) {
                    preview.style.display = 'none';
                }
            }, { once: true });
        });
    }
    _computeVectorStats(values) {
        const n = values.length;
        if (n === 0) {
            return { norm: 0, mean: 0, std: 0, min: 0, max: 0, sparsity: 1 };
        }
        let sum = 0;
        let sumSq = 0;
        let min = values[0];
        let max = values[0];
        let zeros = 0;
        for (const v of values) {
            sum += v;
            sumSq += v * v;
            if (v < min)
                min = v;
            if (v > max)
                max = v;
            if (Math.abs(v) < 1e-10)
                zeros++;
        }
        const mean = sum / n;
        const variance = sumSq / n - mean * mean;
        const std = Math.sqrt(Math.max(0, variance));
        const norm = Math.sqrt(sumSq);
        const sparsity = zeros / n;
        return { norm, mean, std, min, max, sparsity };
    }
    dispose() {
        if (this._comm) {
            this._comm.close();
            this._comm = null;
        }
        super.dispose();
    }
}
/**
 * MIME renderer factory.
 */
const rendererFactory = {
    safe: true,
    mimeTypes: [MIME_TYPE],
    createRenderer: (options) => {
        return new VirtualTableWidget(options);
    }
};
/**
 * Main plugin - registers MIME renderer and tracks kernel.
 */
const plugin = {
    id: 'lanceql-jupyterlab:plugin',
    autoStart: true,
    requires: [_jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0__.IRenderMimeRegistry],
    optional: [_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_1__.INotebookTracker],
    activate: (app, rendermime, notebookTracker) => {
        console.log('LanceQL: Extension activated');
        // Register MIME renderer
        rendermime.addFactory(rendererFactory, 0);
        console.log('LanceQL: MIME renderer registered for', MIME_TYPE);
        if (!notebookTracker) {
            console.warn('LanceQL: No notebook tracker - Comm will not work');
            return;
        }
        // Track kernel
        const updateKernel = () => {
            var _a, _b;
            const current = notebookTracker.currentWidget;
            if ((_b = (_a = current === null || current === void 0 ? void 0 : current.sessionContext) === null || _a === void 0 ? void 0 : _a.session) === null || _b === void 0 ? void 0 : _b.kernel) {
                _currentKernel = current.sessionContext.session.kernel;
                console.log('LanceQL: Kernel connected');
            }
            else {
                _currentKernel = null;
            }
        };
        notebookTracker.currentChanged.connect(updateKernel);
        notebookTracker.widgetAdded.connect((sender, panel) => {
            panel.sessionContext.kernelChanged.connect(updateKernel);
            panel.sessionContext.statusChanged.connect(updateKernel);
        });
        updateKernel();
    }
};
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (plugin);


/***/ }

}]);
//# sourceMappingURL=lib_index_js.36d2f5a8e69d00984231.js.map