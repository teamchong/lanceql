/**
 * SQLLexer - SQL tokenization
 */

const TokenType = {
    // Keywords
    SELECT: 'SELECT', DISTINCT: 'DISTINCT', FROM: 'FROM', WHERE: 'WHERE',
    AND: 'AND', OR: 'OR', NOT: 'NOT', ORDER: 'ORDER', BY: 'BY',
    ASC: 'ASC', DESC: 'DESC', LIMIT: 'LIMIT', OFFSET: 'OFFSET', AS: 'AS',
    NULL: 'NULL', IS: 'IS', IN: 'IN', BETWEEN: 'BETWEEN', LIKE: 'LIKE',
    TRUE: 'TRUE', FALSE: 'FALSE', GROUP: 'GROUP', HAVING: 'HAVING',
    QUALIFY: 'QUALIFY', ROLLUP: 'ROLLUP', CUBE: 'CUBE', GROUPING: 'GROUPING', SETS: 'SETS',
    COUNT: 'COUNT', SUM: 'SUM', AVG: 'AVG', MIN: 'MIN', MAX: 'MAX',
    NEAR: 'NEAR', TOPK: 'TOPK', FILE: 'FILE',
    JOIN: 'JOIN', INNER: 'INNER', LEFT: 'LEFT', RIGHT: 'RIGHT', FULL: 'FULL',
    OUTER: 'OUTER', CROSS: 'CROSS', ON: 'ON',
    CREATE: 'CREATE', TABLE: 'TABLE', INSERT: 'INSERT', INTO: 'INTO', VALUES: 'VALUES',
    UPDATE: 'UPDATE', SET: 'SET', DELETE: 'DELETE', DROP: 'DROP', IF: 'IF', EXISTS: 'EXISTS',
    INT: 'INT', INTEGER: 'INTEGER', BIGINT: 'BIGINT', FLOAT: 'FLOAT', REAL: 'REAL',
    DOUBLE: 'DOUBLE', TEXT: 'TEXT', VARCHAR: 'VARCHAR', BOOLEAN: 'BOOLEAN', BOOL: 'BOOL',
    VECTOR: 'VECTOR', PRIMARY: 'PRIMARY', KEY: 'KEY',
    WITH: 'WITH', RECURSIVE: 'RECURSIVE', UNION: 'UNION', ALL: 'ALL',
    PIVOT: 'PIVOT', UNPIVOT: 'UNPIVOT', FOR: 'FOR',
    INTERSECT: 'INTERSECT', EXCEPT: 'EXCEPT',
    OVER: 'OVER', PARTITION: 'PARTITION', ROW_NUMBER: 'ROW_NUMBER', RANK: 'RANK',
    DENSE_RANK: 'DENSE_RANK', NTILE: 'NTILE', LAG: 'LAG', LEAD: 'LEAD',
    FIRST_VALUE: 'FIRST_VALUE', LAST_VALUE: 'LAST_VALUE', NTH_VALUE: 'NTH_VALUE',
    PERCENT_RANK: 'PERCENT_RANK', CUME_DIST: 'CUME_DIST',
    ROWS: 'ROWS', RANGE: 'RANGE', UNBOUNDED: 'UNBOUNDED', PRECEDING: 'PRECEDING',
    FOLLOWING: 'FOLLOWING', CURRENT: 'CURRENT', ROW: 'ROW',
    EXPLAIN: 'EXPLAIN', ARRAY: 'ARRAY', CASE: 'CASE', WHEN: 'WHEN', THEN: 'THEN',
    ELSE: 'ELSE', END: 'END', CAST: 'CAST', COALESCE: 'COALESCE', NULLIF: 'NULLIF',
    // Literals & Operators
    IDENTIFIER: 'IDENTIFIER', NUMBER: 'NUMBER', STRING: 'STRING',
    STAR: 'STAR', COMMA: 'COMMA', DOT: 'DOT', LPAREN: 'LPAREN', RPAREN: 'RPAREN',
    EQ: 'EQ', NE: 'NE', LT: 'LT', LE: 'LE', GT: 'GT', GE: 'GE',
    PLUS: 'PLUS', MINUS: 'MINUS', SLASH: 'SLASH', LBRACKET: 'LBRACKET', RBRACKET: 'RBRACKET',
    EOF: 'EOF',
};

const KEYWORDS = {
    'SELECT': TokenType.SELECT, 'DISTINCT': TokenType.DISTINCT, 'FROM': TokenType.FROM,
    'WHERE': TokenType.WHERE, 'AND': TokenType.AND, 'OR': TokenType.OR, 'NOT': TokenType.NOT,
    'ORDER': TokenType.ORDER, 'BY': TokenType.BY, 'ASC': TokenType.ASC, 'DESC': TokenType.DESC,
    'LIMIT': TokenType.LIMIT, 'OFFSET': TokenType.OFFSET, 'AS': TokenType.AS,
    'NULL': TokenType.NULL, 'IS': TokenType.IS, 'IN': TokenType.IN, 'BETWEEN': TokenType.BETWEEN,
    'LIKE': TokenType.LIKE, 'TRUE': TokenType.TRUE, 'FALSE': TokenType.FALSE,
    'GROUP': TokenType.GROUP, 'HAVING': TokenType.HAVING, 'QUALIFY': TokenType.QUALIFY,
    'ROLLUP': TokenType.ROLLUP, 'CUBE': TokenType.CUBE, 'GROUPING': TokenType.GROUPING, 'SETS': TokenType.SETS,
    'COUNT': TokenType.COUNT, 'SUM': TokenType.SUM, 'AVG': TokenType.AVG, 'MIN': TokenType.MIN, 'MAX': TokenType.MAX,
    'NEAR': TokenType.NEAR, 'TOPK': TokenType.TOPK, 'FILE': TokenType.FILE,
    'JOIN': TokenType.JOIN, 'INNER': TokenType.INNER, 'LEFT': TokenType.LEFT, 'RIGHT': TokenType.RIGHT,
    'FULL': TokenType.FULL, 'OUTER': TokenType.OUTER, 'CROSS': TokenType.CROSS, 'ON': TokenType.ON,
    'CREATE': TokenType.CREATE, 'TABLE': TokenType.TABLE, 'INSERT': TokenType.INSERT, 'INTO': TokenType.INTO,
    'VALUES': TokenType.VALUES, 'UPDATE': TokenType.UPDATE, 'SET': TokenType.SET, 'DELETE': TokenType.DELETE,
    'DROP': TokenType.DROP, 'IF': TokenType.IF, 'EXISTS': TokenType.EXISTS,
    'INT': TokenType.INT, 'INTEGER': TokenType.INTEGER, 'BIGINT': TokenType.BIGINT,
    'FLOAT': TokenType.FLOAT, 'REAL': TokenType.REAL, 'DOUBLE': TokenType.DOUBLE,
    'TEXT': TokenType.TEXT, 'VARCHAR': TokenType.VARCHAR, 'BOOLEAN': TokenType.BOOLEAN, 'BOOL': TokenType.BOOL,
    'VECTOR': TokenType.VECTOR, 'PRIMARY': TokenType.PRIMARY, 'KEY': TokenType.KEY,
    'WITH': TokenType.WITH, 'RECURSIVE': TokenType.RECURSIVE, 'UNION': TokenType.UNION, 'ALL': TokenType.ALL,
    'PIVOT': TokenType.PIVOT, 'UNPIVOT': TokenType.UNPIVOT, 'FOR': TokenType.FOR,
    'INTERSECT': TokenType.INTERSECT, 'EXCEPT': TokenType.EXCEPT,
    'OVER': TokenType.OVER, 'PARTITION': TokenType.PARTITION, 'ROW_NUMBER': TokenType.ROW_NUMBER,
    'RANK': TokenType.RANK, 'DENSE_RANK': TokenType.DENSE_RANK, 'NTILE': TokenType.NTILE,
    'LAG': TokenType.LAG, 'LEAD': TokenType.LEAD, 'FIRST_VALUE': TokenType.FIRST_VALUE,
    'LAST_VALUE': TokenType.LAST_VALUE, 'NTH_VALUE': TokenType.NTH_VALUE,
    'PERCENT_RANK': TokenType.PERCENT_RANK, 'CUME_DIST': TokenType.CUME_DIST,
    'ROWS': TokenType.ROWS, 'RANGE': TokenType.RANGE, 'UNBOUNDED': TokenType.UNBOUNDED,
    'PRECEDING': TokenType.PRECEDING, 'FOLLOWING': TokenType.FOLLOWING,
    'CURRENT': TokenType.CURRENT, 'ROW': TokenType.ROW,
    'EXPLAIN': TokenType.EXPLAIN, 'ARRAY': TokenType.ARRAY,
    'CASE': TokenType.CASE, 'WHEN': TokenType.WHEN, 'THEN': TokenType.THEN,
    'ELSE': TokenType.ELSE, 'END': TokenType.END, 'CAST': TokenType.CAST,
    'COALESCE': TokenType.COALESCE, 'NULLIF': TokenType.NULLIF,
};

class SQLLexer {
    constructor(sql) {
        this.sql = sql;
        this.pos = 0;
        this.length = sql.length;
    }

    peek() {
        if (this.pos >= this.length) return '\0';
        return this.sql[this.pos];
    }

    advance() {
        if (this.pos < this.length) {
            return this.sql[this.pos++];
        }
        return '\0';
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
            if (ch === '.' && !hasDecimal) {
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
        this.advance(); // Skip opening quote

        while (this.pos < this.length) {
            const ch = this.sql[this.pos];
            if (ch === quote) {
                // Check for escaped quote
                if (this.pos + 1 < this.length && this.sql[this.pos + 1] === quote) {
                    this.pos += 2;
                    continue;
                }
                this.pos++; // Skip closing quote
                break;
            }
            this.pos++;
        }

        // Return string without quotes, handling escaped quotes
        const inner = this.sql.slice(start + 1, this.pos - 1);
        return inner.replace(new RegExp(quote + quote, 'g'), quote);
    }

    nextToken() {
        this.skipWhitespace();

        if (this.pos >= this.length) {
            return { type: TokenType.EOF, value: null };
        }

        const ch = this.peek();

        // Identifiers and keywords
        if (/[a-zA-Z_]/.test(ch)) {
            const value = this.readIdentifier();
            const upper = value.toUpperCase();
            const type = KEYWORDS[upper] || TokenType.IDENTIFIER;
            return { type, value: type === TokenType.IDENTIFIER ? value : upper };
        }

        // Numbers
        if (/\d/.test(ch)) {
            const value = this.readNumber();
            return { type: TokenType.NUMBER, value };
        }

        // Strings
        if (ch === "'" || ch === '"') {
            const value = this.readString(ch);
            return { type: TokenType.STRING, value };
        }

        // Operators
        this.advance();

        switch (ch) {
            case '*': return { type: TokenType.STAR, value: '*' };
            case ',': return { type: TokenType.COMMA, value: ',' };
            case '.': return { type: TokenType.DOT, value: '.' };
            case '(': return { type: TokenType.LPAREN, value: '(' };
            case ')': return { type: TokenType.RPAREN, value: ')' };
            case '+': return { type: TokenType.PLUS, value: '+' };
            case '-': return { type: TokenType.MINUS, value: '-' };
            case '/': return { type: TokenType.SLASH, value: '/' };
            case '[': return { type: TokenType.LBRACKET, value: '[' };
            case ']': return { type: TokenType.RBRACKET, value: ']' };
            case '=': return { type: TokenType.EQ, value: '=' };
            case '<':
                if (this.peek() === '=') {
                    this.advance();
                    return { type: TokenType.LE, value: '<=' };
                }
                if (this.peek() === '>') {
                    this.advance();
                    return { type: TokenType.NE, value: '<>' };
                }
                return { type: TokenType.LT, value: '<' };
            case '>':
                if (this.peek() === '=') {
                    this.advance();
                    return { type: TokenType.GE, value: '>=' };
                }
                return { type: TokenType.GT, value: '>' };
            case '!':
                if (this.peek() === '=') {
                    this.advance();
                    return { type: TokenType.NE, value: '!=' };
                }
                throw new Error(`Unexpected character: ${ch}`);
            default:
                throw new Error(`Unexpected character: ${ch}`);
        }
    }

    tokenize() {
        const tokens = [];
        let token;
        while ((token = this.nextToken()).type !== TokenType.EOF) {
            tokens.push(token);
        }
        tokens.push(token); // Include EOF
        return tokens;
    }
}

/**
 * SQL Parser - parses tokens into AST
 */

export { SQLLexer, TokenType, KEYWORDS };
