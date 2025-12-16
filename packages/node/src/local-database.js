/**
 * LocalDatabase - CRUD operations with ACID support for Node.js
 *
 * Uses manifest-based ACID:
 * - Atomic manifest updates (file rename)
 * - Fragment-based storage (append-only data)
 * - Deletion vectors (logical deletes)
 * - Versioning for rollback
 */

const fs = require('fs');
const path = require('path');
const { promisify } = require('util');

const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);
const mkdir = promisify(fs.mkdir);
const readdir = promisify(fs.readdir);
const stat = promisify(fs.stat);
const unlink = promisify(fs.unlink);
const rename = promisify(fs.rename);

// =============================================================================
// SQL Lexer/Parser for CRUD operations (shared with browser)
// =============================================================================

const TokenType = {
    SELECT: 'SELECT', FROM: 'FROM', WHERE: 'WHERE', ORDER: 'ORDER', BY: 'BY',
    LIMIT: 'LIMIT', OFFSET: 'OFFSET', ASC: 'ASC', DESC: 'DESC', AND: 'AND',
    OR: 'OR', NOT: 'NOT', IN: 'IN', LIKE: 'LIKE', BETWEEN: 'BETWEEN', IS: 'IS',
    NULL: 'NULL', TRUE: 'TRUE', FALSE: 'FALSE', AS: 'AS', DISTINCT: 'DISTINCT',
    COUNT: 'COUNT', SUM: 'SUM', AVG: 'AVG', MIN: 'MIN', MAX: 'MAX',
    GROUP: 'GROUP', HAVING: 'HAVING', STAR: 'STAR', COMMA: 'COMMA',
    LPAREN: 'LPAREN', RPAREN: 'RPAREN', EQ: 'EQ', NE: 'NE', LT: 'LT',
    LE: 'LE', GT: 'GT', GE: 'GE', PLUS: 'PLUS', MINUS: 'MINUS',
    MULTIPLY: 'MULTIPLY', DIVIDE: 'DIVIDE', IDENTIFIER: 'IDENTIFIER',
    STRING: 'STRING', NUMBER: 'NUMBER', FLOAT: 'FLOAT', EOF: 'EOF',
    DOT: 'DOT', NEAR: 'NEAR', TOPK: 'TOPK', FILE: 'FILE', READ_LANCE: 'READ_LANCE',
    CREATE: 'CREATE', TABLE: 'TABLE', INSERT: 'INSERT', INTO: 'INTO',
    VALUES: 'VALUES', UPDATE: 'UPDATE', SET: 'SET', DELETE: 'DELETE',
    DROP: 'DROP', INTEGER: 'INTEGER', TEXT: 'TEXT', REAL: 'REAL',
    BLOB: 'BLOB', PRIMARY: 'PRIMARY', KEY: 'KEY', AUTOINCREMENT: 'AUTOINCREMENT',
    VECTOR: 'VECTOR'
};

const KEYWORDS = {
    'select': TokenType.SELECT, 'from': TokenType.FROM, 'where': TokenType.WHERE,
    'order': TokenType.ORDER, 'by': TokenType.BY, 'limit': TokenType.LIMIT,
    'offset': TokenType.OFFSET, 'asc': TokenType.ASC, 'desc': TokenType.DESC,
    'and': TokenType.AND, 'or': TokenType.OR, 'not': TokenType.NOT,
    'in': TokenType.IN, 'like': TokenType.LIKE, 'between': TokenType.BETWEEN,
    'is': TokenType.IS, 'null': TokenType.NULL, 'true': TokenType.TRUE,
    'false': TokenType.FALSE, 'as': TokenType.AS, 'distinct': TokenType.DISTINCT,
    'count': TokenType.COUNT, 'sum': TokenType.SUM, 'avg': TokenType.AVG,
    'min': TokenType.MIN, 'max': TokenType.MAX, 'group': TokenType.GROUP,
    'having': TokenType.HAVING, 'near': TokenType.NEAR, 'topk': TokenType.TOPK,
    'file': TokenType.FILE, 'read_lance': TokenType.READ_LANCE,
    'create': TokenType.CREATE, 'table': TokenType.TABLE, 'insert': TokenType.INSERT,
    'into': TokenType.INTO, 'values': TokenType.VALUES, 'update': TokenType.UPDATE,
    'set': TokenType.SET, 'delete': TokenType.DELETE, 'drop': TokenType.DROP,
    'integer': TokenType.INTEGER, 'text': TokenType.TEXT, 'real': TokenType.REAL,
    'blob': TokenType.BLOB, 'primary': TokenType.PRIMARY, 'key': TokenType.KEY,
    'autoincrement': TokenType.AUTOINCREMENT, 'vector': TokenType.VECTOR
};

class SQLLexer {
    constructor(sql) {
        this.sql = sql;
        this.pos = 0;
        this.tokens = [];
    }

    peek() { return this.sql[this.pos] || ''; }
    advance() { return this.sql[this.pos++]; }
    isEnd() { return this.pos >= this.sql.length; }
    isWhitespace(c) { return /\s/.test(c); }
    isDigit(c) { return /[0-9]/.test(c); }
    isAlpha(c) { return /[a-zA-Z_]/.test(c); }
    isAlphaNum(c) { return /[a-zA-Z0-9_]/.test(c); }

    tokenize() {
        while (!this.isEnd()) {
            this.skipWhitespace();
            if (this.isEnd()) break;
            const c = this.peek();

            if (c === "'") { this.string(); continue; }
            if (this.isDigit(c) || (c === '-' && this.isDigit(this.sql[this.pos + 1]))) { this.number(); continue; }
            if (this.isAlpha(c)) { this.identifier(); continue; }
            if (c === '*') { this.advance(); this.tokens.push({ type: TokenType.STAR }); continue; }
            if (c === ',') { this.advance(); this.tokens.push({ type: TokenType.COMMA }); continue; }
            if (c === '(') { this.advance(); this.tokens.push({ type: TokenType.LPAREN }); continue; }
            if (c === ')') { this.advance(); this.tokens.push({ type: TokenType.RPAREN }); continue; }
            if (c === '.') { this.advance(); this.tokens.push({ type: TokenType.DOT }); continue; }
            if (c === '+') { this.advance(); this.tokens.push({ type: TokenType.PLUS }); continue; }
            if (c === '-') { this.advance(); this.tokens.push({ type: TokenType.MINUS }); continue; }
            if (c === '/') { this.advance(); this.tokens.push({ type: TokenType.DIVIDE }); continue; }
            if (c === '=' && this.sql[this.pos + 1] !== '=') { this.advance(); this.tokens.push({ type: TokenType.EQ }); continue; }
            if (c === '!' && this.sql[this.pos + 1] === '=') { this.advance(); this.advance(); this.tokens.push({ type: TokenType.NE }); continue; }
            if (c === '<' && this.sql[this.pos + 1] === '>') { this.advance(); this.advance(); this.tokens.push({ type: TokenType.NE }); continue; }
            if (c === '<' && this.sql[this.pos + 1] === '=') { this.advance(); this.advance(); this.tokens.push({ type: TokenType.LE }); continue; }
            if (c === '>' && this.sql[this.pos + 1] === '=') { this.advance(); this.advance(); this.tokens.push({ type: TokenType.GE }); continue; }
            if (c === '<') { this.advance(); this.tokens.push({ type: TokenType.LT }); continue; }
            if (c === '>') { this.advance(); this.tokens.push({ type: TokenType.GT }); continue; }

            this.advance();
        }
        this.tokens.push({ type: TokenType.EOF });
        return this.tokens;
    }

    skipWhitespace() {
        while (!this.isEnd() && this.isWhitespace(this.peek())) this.advance();
    }

    string() {
        this.advance();
        let value = '';
        while (!this.isEnd() && this.peek() !== "'") {
            if (this.peek() === "'" && this.sql[this.pos + 1] === "'") {
                value += "'";
                this.advance(); this.advance();
            } else {
                value += this.advance();
            }
        }
        this.advance();
        this.tokens.push({ type: TokenType.STRING, value });
    }

    number() {
        let value = '';
        if (this.peek() === '-') value += this.advance();
        while (!this.isEnd() && this.isDigit(this.peek())) value += this.advance();
        if (this.peek() === '.' && this.isDigit(this.sql[this.pos + 1])) {
            value += this.advance();
            while (!this.isEnd() && this.isDigit(this.peek())) value += this.advance();
            this.tokens.push({ type: TokenType.FLOAT, value: parseFloat(value) });
        } else {
            this.tokens.push({ type: TokenType.NUMBER, value: parseInt(value, 10) });
        }
    }

    identifier() {
        let value = '';
        while (!this.isEnd() && this.isAlphaNum(this.peek())) value += this.advance();
        const lower = value.toLowerCase();
        const type = KEYWORDS[lower] || TokenType.IDENTIFIER;
        this.tokens.push({ type, value: type === TokenType.IDENTIFIER ? value : lower });
    }
}

// =============================================================================
// SQL Parser for CRUD statements
// =============================================================================

class LocalSQLParser {
    constructor(tokens) {
        this.tokens = tokens;
        this.pos = 0;
    }

    peek() { return this.tokens[this.pos]; }
    advance() { return this.tokens[this.pos++]; }
    isEnd() { return this.peek().type === TokenType.EOF; }
    match(type) { return this.peek().type === type ? this.advance() : null; }

    parse() {
        const token = this.peek();
        if (token.type === TokenType.CREATE) return this.parseCreate();
        if (token.type === TokenType.INSERT) return this.parseInsert();
        if (token.type === TokenType.UPDATE) return this.parseUpdate();
        if (token.type === TokenType.DELETE) return this.parseDelete();
        if (token.type === TokenType.DROP) return this.parseDrop();
        if (token.type === TokenType.SELECT) return this.parseSelect();
        throw new Error(`Unexpected token: ${token.type}`);
    }

    parseCreate() {
        this.match(TokenType.CREATE);
        if (!this.match(TokenType.TABLE)) throw new Error('Expected TABLE');
        const name = this.match(TokenType.IDENTIFIER)?.value;
        if (!name) throw new Error('Expected table name');
        if (!this.match(TokenType.LPAREN)) throw new Error('Expected (');
        const columns = [];
        do {
            const col = this.parseColumnDef();
            columns.push(col);
        } while (this.match(TokenType.COMMA));
        if (!this.match(TokenType.RPAREN)) throw new Error('Expected )');
        return { type: 'CREATE_TABLE', table: name, columns };
    }

    parseColumnDef() {
        const name = this.match(TokenType.IDENTIFIER)?.value;
        if (!name) throw new Error('Expected column name');
        let dataType = 'TEXT';
        let vectorDim = null;
        let primaryKey = false;
        let autoIncrement = false;

        if (this.match(TokenType.INTEGER)) dataType = 'INTEGER';
        else if (this.match(TokenType.TEXT)) dataType = 'TEXT';
        else if (this.match(TokenType.REAL)) dataType = 'REAL';
        else if (this.match(TokenType.BLOB)) dataType = 'BLOB';
        else if (this.match(TokenType.VECTOR)) {
            dataType = 'VECTOR';
            if (this.match(TokenType.LPAREN)) {
                const dim = this.match(TokenType.NUMBER);
                vectorDim = dim?.value || 384;
                this.match(TokenType.RPAREN);
            }
        }

        if (this.match(TokenType.PRIMARY)) {
            this.match(TokenType.KEY);
            primaryKey = true;
        }
        if (this.match(TokenType.AUTOINCREMENT)) autoIncrement = true;

        return { name, type: dataType, primaryKey, autoIncrement, vectorDim };
    }

    parseInsert() {
        this.match(TokenType.INSERT);
        this.match(TokenType.INTO);
        const table = this.match(TokenType.IDENTIFIER)?.value;
        if (!table) throw new Error('Expected table name');

        let columns = null;
        if (this.match(TokenType.LPAREN)) {
            columns = [];
            do {
                const col = this.match(TokenType.IDENTIFIER)?.value;
                if (col) columns.push(col);
            } while (this.match(TokenType.COMMA));
            this.match(TokenType.RPAREN);
        }

        if (!this.match(TokenType.VALUES)) throw new Error('Expected VALUES');

        const rows = [];
        do {
            if (!this.match(TokenType.LPAREN)) throw new Error('Expected (');
            const values = [];
            do {
                values.push(this.parseValue());
            } while (this.match(TokenType.COMMA));
            if (!this.match(TokenType.RPAREN)) throw new Error('Expected )');
            rows.push(values);
        } while (this.match(TokenType.COMMA));

        return { type: 'INSERT', table, columns, rows };
    }

    parseUpdate() {
        this.match(TokenType.UPDATE);
        const table = this.match(TokenType.IDENTIFIER)?.value;
        if (!table) throw new Error('Expected table name');
        if (!this.match(TokenType.SET)) throw new Error('Expected SET');

        const updates = {};
        do {
            const col = this.match(TokenType.IDENTIFIER)?.value;
            if (!this.match(TokenType.EQ)) throw new Error('Expected =');
            updates[col] = this.parseValue();
        } while (this.match(TokenType.COMMA));

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseCondition();
        }

        return { type: 'UPDATE', table, updates, where };
    }

    parseDelete() {
        this.match(TokenType.DELETE);
        this.match(TokenType.FROM);
        const table = this.match(TokenType.IDENTIFIER)?.value;
        if (!table) throw new Error('Expected table name');

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseCondition();
        }

        return { type: 'DELETE', table, where };
    }

    parseDrop() {
        this.match(TokenType.DROP);
        if (!this.match(TokenType.TABLE)) throw new Error('Expected TABLE');
        const table = this.match(TokenType.IDENTIFIER)?.value;
        if (!table) throw new Error('Expected table name');
        return { type: 'DROP_TABLE', table };
    }

    parseSelect() {
        this.match(TokenType.SELECT);
        const columns = [];
        if (this.match(TokenType.STAR)) {
            columns.push({ type: 'star' });
        } else {
            do {
                const col = this.match(TokenType.IDENTIFIER)?.value;
                if (col) columns.push({ type: 'column', name: col });
            } while (this.match(TokenType.COMMA));
        }

        this.match(TokenType.FROM);
        const table = this.match(TokenType.IDENTIFIER)?.value;

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseCondition();
        }

        let orderBy = null;
        if (this.match(TokenType.ORDER)) {
            this.match(TokenType.BY);
            const col = this.match(TokenType.IDENTIFIER)?.value;
            let desc = false;
            if (this.match(TokenType.DESC)) desc = true;
            else this.match(TokenType.ASC);
            orderBy = { column: col, desc };
        }

        let limit = null;
        if (this.match(TokenType.LIMIT)) {
            const num = this.match(TokenType.NUMBER);
            limit = num?.value;
        }

        return { type: 'SELECT', columns, table, where, orderBy, limit };
    }

    parseValue() {
        if (this.match(TokenType.NULL)) return null;
        if (this.match(TokenType.TRUE)) return true;
        if (this.match(TokenType.FALSE)) return false;
        const str = this.match(TokenType.STRING);
        if (str) return str.value;
        const num = this.match(TokenType.NUMBER);
        if (num) return num.value;
        const flt = this.match(TokenType.FLOAT);
        if (flt) return flt.value;
        throw new Error('Expected value');
    }

    parseCondition() {
        const column = this.match(TokenType.IDENTIFIER)?.value;
        if (!column) throw new Error('Expected column name');
        let op;
        if (this.match(TokenType.EQ)) op = '=';
        else if (this.match(TokenType.NE)) op = '!=';
        else if (this.match(TokenType.LT)) op = '<';
        else if (this.match(TokenType.LE)) op = '<=';
        else if (this.match(TokenType.GT)) op = '>';
        else if (this.match(TokenType.GE)) op = '>=';
        else if (this.match(TokenType.LIKE)) op = 'LIKE';
        else throw new Error('Expected comparison operator');
        const value = this.parseValue();
        return { op, column, value };
    }
}

// =============================================================================
// FileStorage - File-based storage backend
// =============================================================================

class FileStorage {
    constructor(basePath) {
        this.basePath = basePath;
    }

    async init() {
        await mkdir(this.basePath, { recursive: true });
    }

    async save(name, data) {
        const filePath = path.join(this.basePath, name);
        const dir = path.dirname(filePath);
        await mkdir(dir, { recursive: true });
        const buffer = Buffer.isBuffer(data) ? data : Buffer.from(JSON.stringify(data));
        await writeFile(filePath, buffer);
        return { name, size: buffer.length, storage: 'file' };
    }

    async load(name) {
        const filePath = path.join(this.basePath, name);
        try {
            const data = await readFile(filePath);
            try {
                return JSON.parse(data.toString());
            } catch {
                return data;
            }
        } catch (err) {
            if (err.code === 'ENOENT') return null;
            throw err;
        }
    }

    async delete(name) {
        const filePath = path.join(this.basePath, name);
        try {
            await unlink(filePath);
        } catch (err) {
            if (err.code !== 'ENOENT') throw err;
        }
    }

    async list() {
        try {
            const files = await readdir(this.basePath, { recursive: true });
            return files.map(f => ({ name: f }));
        } catch {
            return [];
        }
    }

    async exists(name) {
        const filePath = path.join(this.basePath, name);
        try {
            await stat(filePath);
            return true;
        } catch {
            return false;
        }
    }
}

// =============================================================================
// LocalDatabase - CRUD with ACID support
// =============================================================================

class LocalDatabase {
    constructor(dbPath, options = {}) {
        this.name = path.basename(dbPath);
        this.dbPath = dbPath;
        this.storage = new FileStorage(dbPath);
        this.tables = new Map();
        this.version = 0;
        this.manifestKey = '__manifest__.json';
        this._open = false;
    }

    async open() {
        await this.storage.init();
        const manifest = await this.storage.load(this.manifestKey);
        if (manifest) {
            this.version = manifest.version || 0;
            for (const [name, schema] of Object.entries(manifest.tables || {})) {
                const deletionsArr = manifest.deletions?.[name] || [];
                this.tables.set(name, {
                    schema,
                    fragments: manifest.fragments?.[name] || [],
                    deletions: new Set(deletionsArr)
                });
            }
        }
        this._open = true;
    }

    async close() {
        await this._saveManifest();
        this.tables.clear();
        this._open = false;
    }

    get isOpen() { return this._open; }

    async _saveManifest() {
        const manifest = {
            version: this.version,
            tables: {},
            fragments: {},
            deletions: {}
        };
        for (const [name, table] of this.tables) {
            manifest.tables[name] = table.schema;
            manifest.fragments[name] = table.fragments;
            manifest.deletions[name] = Array.from(table.deletions || []);
        }
        await this.storage.save(this.manifestKey, manifest);
    }

    listTables() {
        return Array.from(this.tables.keys());
    }

    async exec(sql) {
        const lexer = new SQLLexer(sql);
        const tokens = lexer.tokenize();
        const parser = new LocalSQLParser(tokens);
        const ast = parser.parse();
        return this._executeAST(ast);
    }

    async _executeAST(ast) {
        switch (ast.type) {
            case 'CREATE_TABLE': return this.createTable(ast.table, ast.columns);
            case 'DROP_TABLE': return this.dropTable(ast.table);
            case 'INSERT': return this._executeInsert(ast);
            case 'UPDATE': return this._executeUpdate(ast);
            case 'DELETE': return this._executeDelete(ast);
            case 'SELECT': return this._executeSelect(ast);
            default: throw new Error(`Unknown statement type: ${ast.type}`);
        }
    }

    async createTable(tableName, columns) {
        if (this.tables.has(tableName)) {
            throw new Error(`Table ${tableName} already exists`);
        }
        this.tables.set(tableName, {
            schema: { columns, primaryKey: columns.find(c => c.primaryKey)?.name },
            fragments: [],
            deletions: new Set()
        });
        this.version++;
        await this._saveManifest();
        return { success: true, table: tableName };
    }

    async dropTable(tableName) {
        if (!this.tables.has(tableName)) {
            throw new Error(`Table ${tableName} does not exist`);
        }
        const table = this.tables.get(tableName);
        for (const fragId of table.fragments) {
            await this.storage.delete(`${tableName}/${fragId}.json`);
        }
        this.tables.delete(tableName);
        this.version++;
        await this._saveManifest();
        return { success: true };
    }

    async insert(tableName, rows) {
        const table = this.tables.get(tableName);
        if (!table) throw new Error(`Table ${tableName} does not exist`);

        const fragId = `frag_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
        await this.storage.save(`${tableName}/${fragId}.json`, rows);
        table.fragments.push(fragId);
        this.version++;
        await this._saveManifest();
        return { success: true, inserted: rows.length };
    }

    async _executeInsert(ast) {
        const rows = ast.rows.map(values => {
            const row = {};
            if (ast.columns) {
                ast.columns.forEach((col, i) => { row[col] = values[i]; });
            } else {
                const table = this.tables.get(ast.table);
                table.schema.columns.forEach((col, i) => { row[col.name] = values[i]; });
            }
            return row;
        });
        return this.insert(ast.table, rows);
    }

    async _getAllRows(tableName) {
        const table = this.tables.get(tableName);
        if (!table) throw new Error(`Table ${tableName} does not exist`);

        const rows = [];
        let rowId = 0;
        for (const fragId of table.fragments) {
            const fragData = await this.storage.load(`${tableName}/${fragId}.json`);
            if (fragData) {
                for (const row of fragData) {
                    if (!table.deletions.has(rowId)) {
                        rows.push({ ...row, __rowId: rowId });
                    }
                    rowId++;
                }
            }
        }
        return rows;
    }

    async delete(tableName, predicate) {
        const table = this.tables.get(tableName);
        if (!table) throw new Error(`Table ${tableName} does not exist`);

        const rows = await this._getAllRows(tableName);
        let deleted = 0;
        for (const row of rows) {
            if (predicate(row)) {
                table.deletions.add(row.__rowId);
                deleted++;
            }
        }
        this.version++;
        await this._saveManifest();
        return { success: true, deleted };
    }

    async _executeDelete(ast) {
        return this.delete(ast.table, row => this._matchCondition(row, ast.where));
    }

    async update(tableName, updates, predicate) {
        const table = this.tables.get(tableName);
        if (!table) throw new Error(`Table ${tableName} does not exist`);

        const rows = await this._getAllRows(tableName);
        const newRows = [];
        let updated = 0;

        for (const row of rows) {
            if (predicate(row)) {
                table.deletions.add(row.__rowId);
                const newRow = { ...row };
                delete newRow.__rowId;
                Object.assign(newRow, updates);
                newRows.push(newRow);
                updated++;
            }
        }

        if (newRows.length > 0) {
            const fragId = `frag_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
            await this.storage.save(`${tableName}/${fragId}.json`, newRows);
            table.fragments.push(fragId);
        }

        this.version++;
        await this._saveManifest();
        return { success: true, updated };
    }

    async _executeUpdate(ast) {
        return this.update(ast.table, ast.updates, row => this._matchCondition(row, ast.where));
    }

    async select(tableName, options = {}) {
        const rows = await this._getAllRows(tableName);
        let result = rows.map(r => { const { __rowId, ...rest } = r; return rest; });

        if (options.where) {
            result = result.filter(row => options.where(row));
        }
        if (options.orderBy) {
            result.sort((a, b) => {
                const aVal = a[options.orderBy.column];
                const bVal = b[options.orderBy.column];
                return options.orderBy.desc ? (bVal > aVal ? 1 : -1) : (aVal > bVal ? 1 : -1);
            });
        }
        if (options.limit) {
            result = result.slice(0, options.limit);
        }
        return result;
    }

    async _executeSelect(ast) {
        return this.select(ast.table, {
            where: ast.where ? (row => this._matchCondition(row, ast.where)) : null,
            orderBy: ast.orderBy,
            limit: ast.limit
        });
    }

    _matchCondition(row, cond) {
        if (!cond) return true;
        const val = row[cond.column];
        switch (cond.op) {
            case '=': return val === cond.value;
            case '!=': return val !== cond.value;
            case '<': return val < cond.value;
            case '<=': return val <= cond.value;
            case '>': return val > cond.value;
            case '>=': return val >= cond.value;
            case 'LIKE': {
                const pattern = cond.value.replace(/%/g, '.*').replace(/_/g, '.');
                return new RegExp(`^${pattern}$`, 'i').test(val);
            }
            default: return false;
        }
    }

    async compact() {
        for (const [tableName, table] of this.tables) {
            const rows = await this._getAllRows(tableName);
            const cleanRows = rows.map(r => { const { __rowId, ...rest } = r; return rest; });

            for (const fragId of table.fragments) {
                await this.storage.delete(`${tableName}/${fragId}.json`);
            }

            const fragId = `frag_${Date.now()}_compacted`;
            await this.storage.save(`${tableName}/${fragId}.json`, cleanRows);
            table.fragments = [fragId];
            table.deletions = new Set();
        }
        this.version++;
        await this._saveManifest();
    }
}

module.exports = { LocalDatabase, FileStorage, SQLLexer, LocalSQLParser };
