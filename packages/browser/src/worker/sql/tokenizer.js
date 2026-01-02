/**
 * SQL Tokenizer - Lexical analysis for SQL statements
 */

export const TokenType = {
    // Keywords
    SELECT: 'SELECT', FROM: 'FROM', WHERE: 'WHERE', INSERT: 'INSERT',
    INTO: 'INTO', VALUES: 'VALUES', UPDATE: 'UPDATE', SET: 'SET',
    DELETE: 'DELETE', CREATE: 'CREATE', DROP: 'DROP', TABLE: 'TABLE',
    IF: 'IF', EXISTS: 'EXISTS', NOT: 'NOT', PRIMARY: 'PRIMARY',
    KEY: 'KEY', ORDER: 'ORDER', BY: 'BY', ASC: 'ASC', DESC: 'DESC',
    LIMIT: 'LIMIT', OFFSET: 'OFFSET', AND: 'AND', OR: 'OR',
    NULL: 'NULL', TRUE: 'TRUE', FALSE: 'FALSE', LIKE: 'LIKE',
    // JOIN keywords
    JOIN: 'JOIN', LEFT: 'LEFT', RIGHT: 'RIGHT', INNER: 'INNER', ON: 'ON', AS: 'AS',
    FULL: 'FULL', OUTER: 'OUTER', CROSS: 'CROSS',
    // GROUP BY / HAVING / QUALIFY / ROLLUP / CUBE / GROUPING SETS
    GROUP: 'GROUP', HAVING: 'HAVING', QUALIFY: 'QUALIFY',
    ROLLUP: 'ROLLUP', CUBE: 'CUBE', GROUPING: 'GROUPING', SETS: 'SETS',
    // Aggregate functions
    COUNT: 'COUNT', SUM: 'SUM', AVG: 'AVG', MIN: 'MIN', MAX: 'MAX',
    // Additional operators
    DISTINCT: 'DISTINCT', BETWEEN: 'BETWEEN', IN: 'IN',
    // Vector search
    NEAR: 'NEAR', TOPK: 'TOPK',
    // CASE expression
    CASE: 'CASE', WHEN: 'WHEN', THEN: 'THEN', ELSE: 'ELSE', END: 'END',
    // Type casting and IS NULL
    CAST: 'CAST', IS: 'IS',
    // Set operations
    UNION: 'UNION', INTERSECT: 'INTERSECT', EXCEPT: 'EXCEPT', ALL: 'ALL',
    // CTEs
    WITH: 'WITH', RECURSIVE: 'RECURSIVE',
    // Window functions
    OVER: 'OVER', PARTITION: 'PARTITION', ROW_NUMBER: 'ROW_NUMBER', RANK: 'RANK',
    DENSE_RANK: 'DENSE_RANK', LAG: 'LAG', LEAD: 'LEAD',
    NTILE: 'NTILE', PERCENT_RANK: 'PERCENT_RANK', CUME_DIST: 'CUME_DIST',
    FIRST_VALUE: 'FIRST_VALUE', LAST_VALUE: 'LAST_VALUE', NTH_VALUE: 'NTH_VALUE',
    // Window frame specifications
    ROWS: 'ROWS', RANGE: 'RANGE', UNBOUNDED: 'UNBOUNDED',
    PRECEDING: 'PRECEDING', FOLLOWING: 'FOLLOWING', CURRENT: 'CURRENT', ROW: 'ROW',
    // NULLS FIRST/LAST for ORDER BY
    NULLS: 'NULLS', FIRST: 'FIRST', LAST: 'LAST',
    // Date/Time keywords (used in EXTRACT)
    YEAR: 'YEAR', MONTH: 'MONTH', DAY: 'DAY',
    HOUR: 'HOUR', MINUTE: 'MINUTE', SECOND: 'SECOND',
    // Array keyword
    ARRAY: 'ARRAY',
    // DML enhancement keywords
    CONFLICT: 'CONFLICT', DO: 'DO', NOTHING: 'NOTHING', EXCLUDED: 'EXCLUDED', USING: 'USING',
    // EXPLAIN
    EXPLAIN: 'EXPLAIN', ANALYZE: 'ANALYZE',
    // PIVOT/UNPIVOT
    PIVOT: 'PIVOT', UNPIVOT: 'UNPIVOT', FOR: 'FOR',

    // Literals
    IDENTIFIER: 'IDENTIFIER', STRING: 'STRING', NUMBER: 'NUMBER',

    // Operators
    EQ: '=', NE: '!=', LT: '<', LE: '<=', GT: '>', GE: '>=',
    STAR: '*', COMMA: ',', LPAREN: '(', RPAREN: ')',
    LBRACKET: '[', RBRACKET: ']', DOT: '.',
    // Arithmetic operators
    PLUS: '+', MINUS: '-', SLASH: '/',
    // Bitwise operators
    AMPERSAND: '&', PIPE: '|', CARET: '^', TILDE: '~', LSHIFT: '<<', RSHIFT: '>>',

    // Special
    EOF: 'EOF',
};

export class SQLLexer {
    constructor(sql) {
        this.sql = sql;
        this.pos = 0;
    }

    tokenize() {
        const tokens = [];

        while (this.pos < this.sql.length) {
            this._skipWhitespace();
            if (this.pos >= this.sql.length) break;

            const token = this._nextToken();
            if (token) tokens.push(token);
        }

        tokens.push({ type: TokenType.EOF });
        return tokens;
    }

    _skipWhitespace() {
        while (this.pos < this.sql.length && /\s/.test(this.sql[this.pos])) {
            this.pos++;
        }
    }

    _nextToken() {
        const ch = this.sql[this.pos];

        // Single character tokens
        const singleChars = {
            '*': TokenType.STAR,
            ',': TokenType.COMMA,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            '=': TokenType.EQ,
            '<': TokenType.LT,
            '>': TokenType.GT,
            '.': TokenType.DOT,
            '+': TokenType.PLUS,
            '/': TokenType.SLASH,
            '&': TokenType.AMPERSAND,
            '|': TokenType.PIPE,
            '^': TokenType.CARET,
            '~': TokenType.TILDE,
        };

        if (singleChars[ch]) {
            this.pos++;

            // Check for multi-char operators
            if (ch === '<' && this.sql[this.pos] === '=') {
                this.pos++;
                return { type: TokenType.LE };
            }
            if (ch === '<' && this.sql[this.pos] === '<') {
                this.pos++;
                return { type: TokenType.LSHIFT };
            }
            if (ch === '>' && this.sql[this.pos] === '=') {
                this.pos++;
                return { type: TokenType.GE };
            }
            if (ch === '>' && this.sql[this.pos] === '>') {
                this.pos++;
                return { type: TokenType.RSHIFT };
            }
            if (ch === '!' && this.sql[this.pos] === '=') {
                this.pos++;
                return { type: TokenType.NE };
            }
            if (ch === '<' && this.sql[this.pos] === '>') {
                this.pos++;
                return { type: TokenType.NE };
            }

            return { type: singleChars[ch] };
        }

        if (ch === '!') {
            this.pos++;
            if (this.sql[this.pos] === '=') {
                this.pos++;
                return { type: TokenType.NE };
            }
            throw new Error(`Unexpected character: !`);
        }

        // String literal
        if (ch === "'" || ch === '"') {
            return this._readString(ch);
        }

        // Minus sign: could be negative number or subtraction operator
        if (ch === '-') {
            // Check if it's a negative number (minus followed by digit)
            // Only treat as negative number at start or after operator/open paren
            const prevChar = this.pos > 0 ? this.sql[this.pos - 1] : ' ';
            const isAfterOperand = /[a-zA-Z0-9_)\]]/.test(prevChar.trim() || ' ');
            if (!isAfterOperand && /\d/.test(this.sql[this.pos + 1])) {
                return this._readNumber();
            }
            // Otherwise treat as minus operator
            this.pos++;
            return { type: TokenType.MINUS };
        }

        // Number
        if (/\d/.test(ch)) {
            return this._readNumber();
        }

        // Identifier or keyword
        if (/[a-zA-Z_]/.test(ch)) {
            return this._readIdentifier();
        }

        throw new Error(`Unexpected character: ${ch}`);
    }

    _readString(quote) {
        this.pos++; // Skip opening quote
        let value = '';

        while (this.pos < this.sql.length && this.sql[this.pos] !== quote) {
            if (this.sql[this.pos] === '\\') {
                this.pos++;
                if (this.pos < this.sql.length) {
                    value += this.sql[this.pos];
                }
            } else {
                value += this.sql[this.pos];
            }
            this.pos++;
        }

        this.pos++; // Skip closing quote
        return { type: TokenType.STRING, value };
    }

    _readNumber() {
        let value = '';
        if (this.sql[this.pos] === '-') {
            value += this.sql[this.pos++];
        }

        while (this.pos < this.sql.length && /[\d.]/.test(this.sql[this.pos])) {
            value += this.sql[this.pos++];
        }

        return { type: TokenType.NUMBER, value };
    }

    _readIdentifier() {
        let value = '';

        while (this.pos < this.sql.length && /[a-zA-Z0-9_]/.test(this.sql[this.pos])) {
            value += this.sql[this.pos++];
        }

        const upper = value.toUpperCase();
        if (TokenType[upper]) {
            return { type: TokenType[upper], value };
        }

        return { type: TokenType.IDENTIFIER, value };
    }
}
