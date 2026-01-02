/**
 * LanceQL - WASM loader and utilities
 */

import { getWebGPUAccelerator } from '../gpu/accelerator.js';

class LocalSQLParser {
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

            // Check for PRIMARY KEY
            if (this.match(TokenType.PRIMARY)) {
                this.expect(TokenType.KEY);
                col.primaryKey = true;
            }

            columns.push(col);
        } while (this.match(TokenType.COMMA));

        this.expect(TokenType.RPAREN);

        return { type: 'create_table', table: tableName, columns };
    }

    parseDataType() {
        const token = this.advance();
        let type = token.value || token.type;

        // Handle VECTOR(dim)
        if (type === 'VECTOR' && this.match(TokenType.LPAREN)) {
            const dim = this.expect(TokenType.NUMBER).value;
            this.expect(TokenType.RPAREN);
            return { type: 'vector', dim: parseInt(dim) };
        }

        // Handle VARCHAR(len)
        if ((type === 'VARCHAR' || type === 'TEXT') && this.match(TokenType.LPAREN)) {
            this.expect(TokenType.NUMBER); // ignore length
            this.expect(TokenType.RPAREN);
        }

        return type;
    }

    parseDrop() {
        this.expect(TokenType.DROP);
        this.expect(TokenType.TABLE);
        const tableName = this.expect(TokenType.IDENTIFIER).value;
        return { type: 'drop_table', table: tableName };
    }

    parseInsert() {
        this.expect(TokenType.INSERT);
        this.expect(TokenType.INTO);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        // Optional column list
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

            // Build row object
            if (columns) {
                const row = {};
                columns.forEach((col, i) => row[col] = values[i]);
                rows.push(row);
            } else {
                rows.push(values); // positional - needs schema lookup
            }
        } while (this.match(TokenType.COMMA));

        return { type: 'insert', table: tableName, columns, rows };
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
        // Vector literal [1.0, 2.0, 3.0]
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

        return { type: 'update', table: tableName, set, where };
    }

    parseDelete() {
        this.expect(TokenType.DELETE);
        this.expect(TokenType.FROM);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        return { type: 'delete', table: tableName, where };
    }

    parseSelect() {
        this.expect(TokenType.SELECT);

        // Columns
        const columns = [];
        if (this.match(TokenType.STAR)) {
            columns.push('*');
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

        return { type: 'select', table: tableName, columns, where, orderBy, limit, offset };
    }

    parseWhereExpr() {
        return this.parseOrExpr();
    }

    parseOrExpr() {
        let left = this.parseAndExpr();
        while (this.match(TokenType.OR)) {
            const right = this.parseAndExpr();
            left = { op: 'OR', left, right };
        }
        return left;
    }

    parseAndExpr() {
        let left = this.parseComparison();
        while (this.match(TokenType.AND)) {
            const right = this.parseComparison();
            left = { op: 'AND', left, right };
        }
        return left;
    }

    parseComparison() {
        const column = this.expect(TokenType.IDENTIFIER).value;

        let op;
        if (this.match(TokenType.EQ)) op = '=';
        else if (this.match(TokenType.NE)) op = '!=';
        else if (this.match(TokenType.LT)) op = '<';
        else if (this.match(TokenType.LE)) op = '<=';
        else if (this.match(TokenType.GT)) op = '>';
        else if (this.match(TokenType.GE)) op = '>=';
        else if (this.match(TokenType.LIKE)) op = 'LIKE';
        else throw new Error(`Expected comparison operator`);

        const value = this.parseValue();
        return { op, column, value };
    }
}

// Immer-style WASM runtime - auto string/bytes marshalling
const E = new TextEncoder();
const D = new TextDecoder();
let _w, _m, _p = 0, _M = 0;

// Get shared buffer view (lazy allocation)
const _g = () => {
    if (!_p || !_M) return null;
    return new Uint8Array(_m.buffer, _p, _M);
};

// Ensure shared buffer is large enough
const _ensure = (size) => {
    if (_p && size <= _M) return true;
    // Free old buffer if exists
    if (_p && _w.free) _w.free(_p, _M);
    _M = Math.max(size + 1024, 4096); // At least 4KB
    _p = _w.alloc(_M);
    return _p !== 0;
};

// Marshal JS value to WASM args (strings and Uint8Array auto-copied to WASM memory)
const _x = a => {
    if (a instanceof Uint8Array) {
        if (!_ensure(a.length)) return [a]; // Fallback if alloc fails
        _g().set(a);
        return [_p, a.length];
    }
    if (typeof a !== 'string') return [a];
    const b = E.encode(a);
    if (!_ensure(b.length)) return [a]; // Fallback if alloc fails
    _g().set(b);
    return [_p, b.length];
};

// Read string from WASM memory
const readStr = (ptr, len) => D.decode(new Uint8Array(_m.buffer, ptr, len));

// Read bytes from WASM memory (returns copy)
const readBytes = (ptr, len) => new Uint8Array(_m.buffer, ptr, len).slice();

/**
 * LanceFileWriter - Thin JS wrapper for WASM fragment writer
 *
 * Uses high-level WASM API: fragmentBegin -> fragmentAdd*Column -> fragmentEnd
 * All encoding logic is in WASM, JS only marshals data.
 */
class LanceFileWriter {
    constructor(schema) {
        this.schema = schema;  // [{name, type, nullable?, vectorDim?}]
        this.columns = new Map();  // columnName -> values[]
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

        // Estimate buffer size
        const estimatedSize = Math.max(64 * 1024, this.rowCount * 1024);
        if (!_w.fragmentBegin(estimatedSize)) {
            throw new Error('Failed to initialize WASM fragment writer');
        }

        // Add each column using high-level WASM API
        for (const col of this.schema) {
            const values = this.columns.get(col.name) || [];
            this._addColumn(col, values);
        }

        // Finalize - WASM writes metadata, offsets table, and footer
        const finalSize = _w.fragmentEnd();
        if (finalSize === 0) {
            throw new Error('Failed to finalize fragment');
        }

        const bufferPtr = _w.writerGetBuffer();
        if (!bufferPtr) {
            throw new Error('Failed to get writer buffer');
        }

        return new Uint8Array(_m.buffer, bufferPtr, finalSize).slice();
    }

    _addColumn(col, values) {
        const type = (col.type || col.dataType || 'string').toLowerCase();
        const nullable = col.nullable !== false;

        // Allocate name in WASM memory
        const nameBytes = E.encode(col.name);
        const namePtr = _w.alloc(nameBytes.length);
        new Uint8Array(_m.buffer, namePtr, nameBytes.length).set(nameBytes);

        let result = 0;

        switch (type) {
            case 'int64':
            case 'int':
            case 'integer':
            case 'bigint': {
                const arr = new BigInt64Array(values.map(v => BigInt(v ?? 0)));
                const ptr = _w.alloc(arr.byteLength);
                new BigInt64Array(_m.buffer, ptr, values.length).set(arr);
                result = _w.fragmentAddInt64Column(namePtr, nameBytes.length, ptr, values.length, nullable);
                _w.free(ptr, arr.byteLength);
                break;
            }

            case 'int32': {
                const arr = new Int32Array(values.map(v => v ?? 0));
                const ptr = _w.alloc(arr.byteLength);
                new Int32Array(_m.buffer, ptr, values.length).set(arr);
                result = _w.fragmentAddInt32Column(namePtr, nameBytes.length, ptr, values.length, nullable);
                _w.free(ptr, arr.byteLength);
                break;
            }

            case 'float64':
            case 'float':
            case 'double': {
                const arr = new Float64Array(values.map(v => v ?? 0.0));
                const ptr = _w.alloc(arr.byteLength);
                new Float64Array(_m.buffer, ptr, values.length).set(arr);
                result = _w.fragmentAddFloat64Column(namePtr, nameBytes.length, ptr, values.length, nullable);
                _w.free(ptr, arr.byteLength);
                break;
            }

            case 'float32': {
                const arr = new Float32Array(values.map(v => v ?? 0.0));
                const ptr = _w.alloc(arr.byteLength);
                new Float32Array(_m.buffer, ptr, values.length).set(arr);
                result = _w.fragmentAddFloat32Column(namePtr, nameBytes.length, ptr, values.length, nullable);
                _w.free(ptr, arr.byteLength);
                break;
            }

            case 'string':
            case 'text':
            case 'varchar': {
                // Build string data and offsets
                let currentOffset = 0;
                const offsets = new Uint32Array(values.length + 1);
                const allBytes = [];

                for (let i = 0; i < values.length; i++) {
                    offsets[i] = currentOffset;
                    const bytes = E.encode(String(values[i] ?? ''));
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

            case 'bool':
            case 'boolean': {
                const byteCount = Math.ceil(values.length / 8);
                const packed = new Uint8Array(byteCount);
                for (let i = 0; i < values.length; i++) {
                    if (values[i]) packed[Math.floor(i / 8)] |= (1 << (i % 8));
                }
                const ptr = _w.alloc(packed.length);
                new Uint8Array(_m.buffer, ptr, packed.length).set(packed);
                result = _w.fragmentAddBoolColumn(namePtr, nameBytes.length, ptr, packed.length, values.length, nullable);
                _w.free(ptr, packed.length);
                break;
            }

            case 'vector': {
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
                // Fallback to string
                _w.free(namePtr, nameBytes.length);
                return this._addColumn({ ...col, type: 'string' }, values);
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
            format: 'json'
        };
        return E.encode(JSON.stringify(data));
    }
}

// WASM utils exported for advanced usage

const wasmUtils = {
    readStr,
    readBytes,
    encoder: E,
    decoder: D,
    getMemory: () => _m,
    getExports: () => _w,
};

// LanceQL high-level methods factory (needs proxy reference)
const _createLanceqlMethods = (proxy) => ({
    /**
     * Get the library version.
     * @returns {string} Version string like "0.1.0"
     */
    getVersion() {
        const v = _w.getVersion();
        const major = (v >> 16) & 0xFF;
        const minor = (v >> 8) & 0xFF;
        const patch = v & 0xFF;
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
        // Ensure WebGPU is initialized for vector search
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
        // Ensure WebGPU is initialized for vector search
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
        // Store reference to lanceql instance for registerRemote()
        if (typeof window !== 'undefined') {
            window.lanceql = proxy;
        } else if (typeof globalThis !== 'undefined') {
            globalThis.lanceql = proxy;
        }
        return new LanceDatabase();
    }
});


class LanceQL {
    /**
     * Load LanceQL from a WASM file path or URL.
     * Returns Immer-style proxy with auto string/bytes marshalling.
     * @param {string} wasmPath - Path to the lanceql.wasm file
     * @returns {Promise<LanceQL>}
     */
    static async load(wasmPath = './lanceql.wasm') {
        const response = await fetch(wasmPath);
        const wasmBytes = await response.arrayBuffer();
        const wasmModule = await WebAssembly.instantiate(wasmBytes, {});

        _w = wasmModule.instance.exports;
        _m = _w.memory;

        // Create Immer-style proxy that auto-marshals string/bytes arguments
        // Also includes high-level LanceQL methods
        let _methods = null;
        const proxy = new Proxy({}, {
            get(_, n) {
                // Lazy init methods with proxy reference
                if (!_methods) _methods = _createLanceqlMethods(proxy);
                // High-level LanceQL methods
                if (n in _methods) return _methods[n];
                // Special properties
                if (n === 'memory') return _m;
                if (n === 'raw') return _w;  // Raw WASM exports
                if (n === 'wasm') return _w; // Backward compatibility
                // WASM functions with auto-marshalling
                if (typeof _w[n] === 'function') {
                    return (...a) => _w[n](...a.flatMap(_x));
                }
                return _w[n];
            }
        });
        return proxy;
    }
}

/**
 * Represents an open Lance file (loaded entirely in memory).
 */

export { LocalSQLParser, LanceFileWriter, wasmUtils, LanceQL };
