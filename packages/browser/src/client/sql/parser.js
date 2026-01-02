/**
 * SQLParser - Recursive descent SQL parser
 */

import { SQLLexer } from './lexer.js';

class SQLParser {
    constructor(tokens) {
        this.tokens = tokens;
        this.pos = 0;
    }

    current() {
        return this.tokens[this.pos] || { type: TokenType.EOF };
    }

    advance() {
        if (this.pos < this.tokens.length) {
            return this.tokens[this.pos++];
        }
        return { type: TokenType.EOF };
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
        // Handle EXPLAIN prefix
        if (this.check(TokenType.EXPLAIN)) {
            this.advance();  // consume EXPLAIN
            const statement = this.parse();  // Parse the inner statement
            return { type: 'EXPLAIN', statement };
        }

        // Check for WITH clause (CTEs)
        let ctes = [];
        if (this.check(TokenType.WITH)) {
            ctes = this.parseWithClause();
        }

        // Dispatch based on first keyword
        if (this.check(TokenType.SELECT)) {
            const result = this.parseSelect();
            result.ctes = ctes;  // Attach CTEs to SELECT
            return result;
        } else if (this.check(TokenType.INSERT)) {
            return this.parseInsert();
        } else if (this.check(TokenType.UPDATE)) {
            return this.parseUpdate();
        } else if (this.check(TokenType.DELETE)) {
            return this.parseDelete();
        } else if (this.check(TokenType.CREATE)) {
            return this.parseCreateTable();
        } else if (this.check(TokenType.DROP)) {
            return this.parseDropTable();
        } else {
            throw new Error(`Unexpected token: ${this.current().type}. Expected SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, or EXPLAIN`);
        }
    }

    /**
     * Parse WITH clause (Common Table Expressions)
     * Syntax: WITH [RECURSIVE] name [(columns)] AS (subquery) [, ...]
     */
    parseWithClause() {
        this.expect(TokenType.WITH);
        const isRecursive = !!this.match(TokenType.RECURSIVE);

        const ctes = [];
        do {
            const name = this.expect(TokenType.IDENTIFIER).value;

            // Optional column list
            let columns = [];
            if (this.match(TokenType.LPAREN)) {
                columns.push(this.expect(TokenType.IDENTIFIER).value);
                while (this.match(TokenType.COMMA)) {
                    columns.push(this.expect(TokenType.IDENTIFIER).value);
                }
                this.expect(TokenType.RPAREN);
            }

            this.expect(TokenType.AS);
            this.expect(TokenType.LPAREN);

            // Parse CTE body - may contain UNION ALL for recursive CTEs
            const body = this.parseCteBody(isRecursive);

            this.expect(TokenType.RPAREN);

            ctes.push({
                name,
                columns,
                body,
                recursive: isRecursive
            });
        } while (this.match(TokenType.COMMA));

        return ctes;
    }

    /**
     * Parse CTE body which may contain UNION ALL for recursive CTEs
     */
    parseCteBody(isRecursive) {
        // Parse anchor query - disable set operation parsing, we handle UNION ALL here
        const anchor = this.parseSelect(true, true);  // isSubquery=true, noSetOps=true

        // Check for UNION ALL (required for recursive CTEs)
        if (isRecursive && this.match(TokenType.UNION)) {
            this.expect(TokenType.ALL);
            const recursive = this.parseSelect(true, true);  // Same for recursive part
            return {
                type: 'RECURSIVE_CTE',
                anchor,
                recursive
            };
        }

        return anchor;
    }

    /**
     * Parse SELECT statement
     * @param {boolean} isSubquery - If true, don't require EOF at end (for subqueries)
     * @param {boolean} noSetOps - If true, don't parse set operations (for CTE body parsing)
     */
    parseSelect(isSubquery = false, noSetOps = false) {
        this.expect(TokenType.SELECT);

        // DISTINCT
        const distinct = !!this.match(TokenType.DISTINCT);

        // Select list
        const columns = this.parseSelectList();

        // FROM - supports: table_name, read_lance('url'), 'url.lance'
        let from = null;
        if (this.match(TokenType.FROM)) {
            from = this.parseFromClause();
        }

        // JOIN clauses (one or more)
        const joins = [];
        while (this.check(TokenType.JOIN) || this.check(TokenType.INNER) ||
               this.check(TokenType.LEFT) || this.check(TokenType.RIGHT) ||
               this.check(TokenType.FULL) || this.check(TokenType.CROSS)) {
            joins.push(this.parseJoinClause());
        }

        // PIVOT clause (optional)
        // Syntax: PIVOT (aggregate FOR column IN (value1, value2, ...))
        let pivot = null;
        if (this.match(TokenType.PIVOT)) {
            this.expect(TokenType.LPAREN);

            // Parse aggregate function
            const aggFunc = this.parsePrimary();
            if (aggFunc.type !== 'call') {
                throw new Error('PIVOT requires an aggregate function (e.g., SUM, COUNT, AVG)');
            }

            this.expect(TokenType.FOR);
            const forColumn = this.expect(TokenType.IDENTIFIER).value;

            this.expect(TokenType.IN);
            this.expect(TokenType.LPAREN);

            // Parse IN values
            const inValues = [];
            inValues.push(this.parsePrimary().value);
            while (this.match(TokenType.COMMA)) {
                inValues.push(this.parsePrimary().value);
            }
            this.expect(TokenType.RPAREN);
            this.expect(TokenType.RPAREN);

            pivot = {
                aggregate: aggFunc,
                forColumn,
                inValues
            };
        }

        // UNPIVOT clause (optional)
        // Syntax: UNPIVOT (valueColumn FOR nameColumn IN (col1, col2, ...))
        let unpivot = null;
        if (this.match(TokenType.UNPIVOT)) {
            this.expect(TokenType.LPAREN);

            const valueColumn = this.expect(TokenType.IDENTIFIER).value;

            this.expect(TokenType.FOR);
            const nameColumn = this.expect(TokenType.IDENTIFIER).value;

            this.expect(TokenType.IN);
            this.expect(TokenType.LPAREN);

            // Parse IN columns
            const inColumns = [];
            inColumns.push(this.expect(TokenType.IDENTIFIER).value);
            while (this.match(TokenType.COMMA)) {
                inColumns.push(this.expect(TokenType.IDENTIFIER).value);
            }
            this.expect(TokenType.RPAREN);
            this.expect(TokenType.RPAREN);

            unpivot = {
                valueColumn,
                nameColumn,
                inColumns
            };
        }

        // WHERE
        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseExpr();
        }

        // GROUP BY (supports ROLLUP, CUBE, GROUPING SETS)
        let groupBy = [];
        if (this.match(TokenType.GROUP)) {
            this.expect(TokenType.BY);
            groupBy = this.parseGroupByList();
        }

        // HAVING
        let having = null;
        if (this.match(TokenType.HAVING)) {
            having = this.parseExpr();
        }

        // QUALIFY - filter on window function results
        let qualify = null;
        if (this.match(TokenType.QUALIFY)) {
            qualify = this.parseExpr();
        }

        // NEAR - vector similarity search
        // Syntax: NEAR [column] <'text'|row_num> [TOPK n]
        let search = null;
        if (this.match(TokenType.NEAR)) {
            let column = null;
            let query = null;
            let searchRow = null;
            let topK = 20; // default
            let encoder = 'minilm'; // default

            // First token after NEAR: could be column name, string, or number
            if (this.check(TokenType.IDENTIFIER)) {
                // Could be column name - peek ahead
                const ident = this.advance().value;
                if (this.check(TokenType.STRING) || this.check(TokenType.NUMBER)) {
                    // It was a column name
                    column = ident;
                } else {
                    // It was a search term without quotes (error)
                    throw new Error(`NEAR requires quoted text or row number. Did you mean: NEAR '${ident}'?`);
                }
            }

            // Now expect string (text search) or number (row search)
            if (this.check(TokenType.STRING)) {
                query = this.advance().value;
            } else if (this.check(TokenType.NUMBER)) {
                searchRow = parseInt(this.advance().value, 10);
            } else {
                throw new Error('NEAR requires a quoted text string or row number');
            }

            // Optional TOPK
            if (this.match(TokenType.TOPK)) {
                topK = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }

            search = { query, searchRow, column, topK, encoder };
        }

        // Build the base SELECT AST (without ORDER BY/LIMIT/OFFSET yet)
        const baseAst = {
            type: 'SELECT',
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
            offset: null,
        };

        // Check for set operations (UNION, INTERSECT, EXCEPT) BEFORE parsing ORDER BY/LIMIT
        // This ensures ORDER BY/LIMIT apply to the set operation result, not individual SELECTs
        if (!noSetOps && (this.check(TokenType.UNION) || this.check(TokenType.INTERSECT) || this.check(TokenType.EXCEPT))) {
            const operator = this.advance().type;
            const all = !!this.match(TokenType.ALL);
            // Parse right side without set operations (will be handled here) and without ORDER BY/LIMIT
            const right = this.parseSelect(true, true);  // noSetOps=true for right side

            // Now parse ORDER BY/LIMIT/OFFSET for the combined result
            let orderBy = [];
            let limit = null;
            let offset = null;

            if (this.match(TokenType.ORDER)) {
                this.expect(TokenType.BY);
                orderBy = this.parseOrderByList();
            }
            if (this.match(TokenType.LIMIT)) {
                limit = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }
            if (orderBy.length === 0 && this.match(TokenType.ORDER)) {
                this.expect(TokenType.BY);
                orderBy = this.parseOrderByList();
            }
            if (this.match(TokenType.OFFSET)) {
                offset = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }

            // Check that we've consumed all tokens (unless this is a subquery)
            if (!isSubquery && this.current().type !== TokenType.EOF) {
                throw new Error(`Unexpected token after query: ${this.current().type} (${this.current().value}). Check your SQL syntax.`);
            }

            return {
                type: 'SET_OPERATION',
                operator,
                all,
                left: baseAst,
                right,
                orderBy,
                limit,
                offset,
            };
        }

        // No set operation - parse ORDER BY/LIMIT/OFFSET for this SELECT
        // UNLESS noSetOps is true (we're part of a set operation and ORDER BY/LIMIT belong to outer context)
        if (!noSetOps) {
            let orderBy = [];
            let limit = null;
            let offset = null;

            if (this.match(TokenType.ORDER)) {
                this.expect(TokenType.BY);
                orderBy = this.parseOrderByList();
            }
            if (this.match(TokenType.LIMIT)) {
                limit = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }
            if (orderBy.length === 0 && this.match(TokenType.ORDER)) {
                this.expect(TokenType.BY);
                orderBy = this.parseOrderByList();
            }
            if (this.match(TokenType.OFFSET)) {
                offset = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }

            baseAst.orderBy = orderBy;
            baseAst.limit = limit;
            baseAst.offset = offset;
        }

        // Check that we've consumed all tokens (unless this is a subquery)
        if (!isSubquery && this.current().type !== TokenType.EOF) {
            throw new Error(`Unexpected token after query: ${this.current().type} (${this.current().value}). Check your SQL syntax.`);
        }

        return baseAst;
    }

    /**
     * Parse INSERT statement
     * Syntax: INSERT INTO table_name [(col1, col2, ...)] VALUES (val1, val2, ...), ...
     */
    parseInsert() {
        this.expect(TokenType.INSERT);
        this.expect(TokenType.INTO);

        // Table name
        const table = this.expect(TokenType.IDENTIFIER).value;

        // Optional column list
        let columns = null;
        if (this.match(TokenType.LPAREN)) {
            columns = [];
            columns.push(this.expect(TokenType.IDENTIFIER).value);
            while (this.match(TokenType.COMMA)) {
                columns.push(this.expect(TokenType.IDENTIFIER).value);
            }
            this.expect(TokenType.RPAREN);
        }

        // VALUES clause
        this.expect(TokenType.VALUES);

        // Parse value rows
        const rows = [];
        do {
            this.expect(TokenType.LPAREN);
            const values = [];
            values.push(this.parseValue());
            while (this.match(TokenType.COMMA)) {
                values.push(this.parseValue());
            }
            this.expect(TokenType.RPAREN);
            rows.push(values);
        } while (this.match(TokenType.COMMA));

        return {
            type: 'INSERT',
            table,
            columns,
            rows,
        };
    }

    /**
     * Parse a single value (number, string, null, true, false)
     */
    parseValue() {
        if (this.match(TokenType.NULL)) {
            return { type: 'null', value: null };
        }
        if (this.match(TokenType.TRUE)) {
            return { type: 'boolean', value: true };
        }
        if (this.match(TokenType.FALSE)) {
            return { type: 'boolean', value: false };
        }
        if (this.check(TokenType.NUMBER)) {
            const token = this.advance();
            const value = token.value.includes('.') ? parseFloat(token.value) : parseInt(token.value, 10);
            return { type: 'number', value };
        }
        if (this.check(TokenType.STRING)) {
            const token = this.advance();
            return { type: 'string', value: token.value };
        }
        if (this.check(TokenType.MINUS)) {
            this.advance();
            const token = this.expect(TokenType.NUMBER);
            const value = token.value.includes('.') ? -parseFloat(token.value) : -parseInt(token.value, 10);
            return { type: 'number', value };
        }
        // Vector literal: [1.0, 2.0, 3.0]
        if (this.check(TokenType.LBRACKET)) {
            return this.parseArrayLiteral();
        }

        throw new Error(`Expected value, got ${this.current().type}`);
    }

    /**
     * Parse array literal: [1, 2, 3] or ARRAY[1, 2, 3]
     */
    parseArrayLiteral() {
        this.expect(TokenType.LBRACKET);
        const elements = [];

        if (!this.check(TokenType.RBRACKET)) {
            elements.push(this.parseExpr());
            while (this.match(TokenType.COMMA)) {
                elements.push(this.parseExpr());
            }
        }

        this.expect(TokenType.RBRACKET);
        return { type: 'array', elements };
    }

    /**
     * Parse UPDATE statement
     * Syntax: UPDATE table_name SET col1 = val1, col2 = val2 [WHERE condition]
     */
    parseUpdate() {
        this.expect(TokenType.UPDATE);

        // Table name
        const table = this.expect(TokenType.IDENTIFIER).value;

        // SET clause
        this.expect(TokenType.SET);

        const assignments = [];
        do {
            const column = this.expect(TokenType.IDENTIFIER).value;
            this.expect(TokenType.EQ);
            const value = this.parseValue();
            assignments.push({ column, value });
        } while (this.match(TokenType.COMMA));

        // Optional WHERE
        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseExpr();
        }

        return {
            type: 'UPDATE',
            table,
            assignments,
            where,
        };
    }

    /**
     * Parse DELETE statement
     * Syntax: DELETE FROM table_name [WHERE condition]
     */
    parseDelete() {
        this.expect(TokenType.DELETE);
        this.expect(TokenType.FROM);

        // Table name
        const table = this.expect(TokenType.IDENTIFIER).value;

        // Optional WHERE
        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseExpr();
        }

        return {
            type: 'DELETE',
            table,
            where,
        };
    }

    /**
     * Parse CREATE TABLE statement
     * Syntax: CREATE TABLE [IF NOT EXISTS] table_name (col1 TYPE, col2 TYPE, ...)
     */
    parseCreateTable() {
        this.expect(TokenType.CREATE);
        this.expect(TokenType.TABLE);

        // Check for IF NOT EXISTS
        let ifNotExists = false;
        if (this.match(TokenType.IF)) {
            this.expect(TokenType.NOT);
            this.expect(TokenType.EXISTS);
            ifNotExists = true;
        }

        // Table name
        const table = this.expect(TokenType.IDENTIFIER).value;

        // Column definitions
        this.expect(TokenType.LPAREN);

        const columns = [];
        do {
            const name = this.expect(TokenType.IDENTIFIER).value;

            // Data type
            let dataType = 'TEXT'; // default
            let primaryKey = false;
            let vectorDim = null;

            if (this.check(TokenType.INT) || this.check(TokenType.INTEGER) || this.check(TokenType.BIGINT)) {
                this.advance();
                dataType = 'INT64';
            } else if (this.check(TokenType.FLOAT) || this.check(TokenType.REAL) || this.check(TokenType.DOUBLE)) {
                this.advance();
                dataType = 'FLOAT64';
            } else if (this.check(TokenType.TEXT) || this.check(TokenType.VARCHAR)) {
                this.advance();
                dataType = 'STRING';
            } else if (this.check(TokenType.BOOLEAN) || this.check(TokenType.BOOL)) {
                this.advance();
                dataType = 'BOOL';
            } else if (this.check(TokenType.VECTOR)) {
                this.advance();
                dataType = 'VECTOR';
                // Optional dimension: VECTOR(384)
                if (this.match(TokenType.LPAREN)) {
                    vectorDim = parseInt(this.expect(TokenType.NUMBER).value, 10);
                    this.expect(TokenType.RPAREN);
                }
            }

            // Optional PRIMARY KEY
            if (this.match(TokenType.PRIMARY)) {
                this.expect(TokenType.KEY);
                primaryKey = true;
            }

            columns.push({ name, dataType, primaryKey, vectorDim });
        } while (this.match(TokenType.COMMA));

        this.expect(TokenType.RPAREN);

        return {
            type: 'CREATE_TABLE',
            table,
            columns,
            ifNotExists,
        };
    }

    /**
     * Parse DROP TABLE statement
     * Syntax: DROP TABLE [IF EXISTS] table_name
     */
    parseDropTable() {
        this.expect(TokenType.DROP);
        this.expect(TokenType.TABLE);

        // Check for IF EXISTS
        let ifExists = false;
        if (this.match(TokenType.IF)) {
            this.expect(TokenType.EXISTS);
            ifExists = true;
        }

        // Table name
        const table = this.expect(TokenType.IDENTIFIER).value;

        return {
            type: 'DROP_TABLE',
            table,
            ifExists,
        };
    }

    parseSelectList() {
        const items = [this.parseSelectItem()];

        while (this.match(TokenType.COMMA)) {
            items.push(this.parseSelectItem());
        }

        return items;
    }

    parseSelectItem() {
        // Check for *
        if (this.match(TokenType.STAR)) {
            return { type: 'star' };
        }

        // Expression
        const expr = this.parseExpr();

        // Optional AS alias
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.check(TokenType.IDENTIFIER) && !this.check(TokenType.FROM, TokenType.WHERE, TokenType.ORDER, TokenType.LIMIT, TokenType.GROUP, TokenType.JOIN, TokenType.INNER, TokenType.LEFT, TokenType.RIGHT, TokenType.COMMA)) {
            // Implicit alias (but not if next token is a keyword or comma)
            alias = this.advance().value;
        }

        return { type: 'expr', expr, alias };
    }

    /**
     * Parse FROM clause - supports:
     * - table_name (identifier)
     * - read_lance('url') (function call)
     * - 'url.lance' (string literal, auto-detect)
     */
    parseFromClause() {
        let from = null;

        // Check for string literal (direct URL/path)
        if (this.check(TokenType.STRING)) {
            const url = this.advance().value;
            from = { type: 'url', url };
        }
        // Check for function call like read_lance(), read_lance(24), read_lance('url'), read_lance('url', 24)
        else if (this.check(TokenType.IDENTIFIER)) {
            const name = this.advance().value;

            // If followed by (, it's a function call
            if (this.match(TokenType.LPAREN)) {
                const funcName = name.toLowerCase();
                if (funcName === 'read_lance') {
                    // read_lance(FILE) - local uploaded file
                    // read_lance(FILE, 24) - local file with version
                    // read_lance('url') - remote url
                    // read_lance('url', 24) - remote url with version
                    from = { type: 'url', function: 'read_lance' };

                    if (!this.check(TokenType.RPAREN)) {
                        // First arg: FILE keyword, string (url)
                        if (this.match(TokenType.FILE)) {
                            // Local file - mark as file reference
                            from.isFile = true;
                            // Check for second arg (version)
                            if (this.match(TokenType.COMMA)) {
                                from.version = parseInt(this.expect(TokenType.NUMBER).value, 10);
                            }
                        } else if (this.check(TokenType.STRING)) {
                            from.url = this.advance().value;
                            // Check for second arg (version)
                            if (this.match(TokenType.COMMA)) {
                                from.version = parseInt(this.expect(TokenType.NUMBER).value, 10);
                            }
                        }
                    }
                    this.expect(TokenType.RPAREN);
                } else {
                    throw new Error(`Unknown table function: ${name}. Supported: read_lance()`);
                }
            } else {
                // Just an identifier (table name - for future use)
                from = { type: 'table', name };
            }
        } else {
            throw new Error('Expected table name, URL string, or read_lance() after FROM');
        }

        // Parse optional alias (e.g., FROM images i or FROM images AS i)
        if (from) {
            if (this.match(TokenType.AS)) {
                from.alias = this.expect(TokenType.IDENTIFIER).value;
            } else if (this.check(TokenType.IDENTIFIER) && !this.check(TokenType.WHERE, TokenType.ORDER, TokenType.LIMIT, TokenType.GROUP, TokenType.NEAR, TokenType.JOIN, TokenType.INNER, TokenType.LEFT, TokenType.RIGHT, TokenType.COMMA)) {
                // Implicit alias (not followed by a keyword)
                from.alias = this.advance().value;
            }
        }

        return from;
    }

    /**
     * Parse JOIN clause - supports:
     * - JOIN table ON condition
     * - INNER JOIN table ON condition
     * - LEFT JOIN table ON condition
     * - RIGHT JOIN table ON condition
     * - FULL OUTER JOIN table ON condition
     * - CROSS JOIN table
     */
    parseJoinClause() {
        // Parse join type
        let joinType = 'INNER'; // default

        if (this.match(TokenType.INNER)) {
            this.expect(TokenType.JOIN);
            joinType = 'INNER';
        } else if (this.match(TokenType.LEFT)) {
            this.match(TokenType.OUTER); // optional
            this.expect(TokenType.JOIN);
            joinType = 'LEFT';
        } else if (this.match(TokenType.RIGHT)) {
            this.match(TokenType.OUTER); // optional
            this.expect(TokenType.JOIN);
            joinType = 'RIGHT';
        } else if (this.match(TokenType.FULL)) {
            this.match(TokenType.OUTER); // optional
            this.expect(TokenType.JOIN);
            joinType = 'FULL';
        } else if (this.match(TokenType.CROSS)) {
            this.expect(TokenType.JOIN);
            joinType = 'CROSS';
        } else {
            this.expect(TokenType.JOIN); // plain JOIN defaults to INNER
        }

        // Parse table reference (same as FROM clause) - includes alias parsing
        const table = this.parseFromClause();

        // Alias is already parsed by parseFromClause and stored in table.alias
        const alias = table.alias || null;

        // Parse ON condition (except for CROSS JOIN)
        let on = null;
        if (joinType !== 'CROSS') {
            this.expect(TokenType.ON);
            on = this.parseExpr();
        }

        return {
            type: joinType,
            table,
            alias,
            on
        };
    }

    parseColumnList() {
        const columns = [this.expect(TokenType.IDENTIFIER).value];

        while (this.match(TokenType.COMMA)) {
            columns.push(this.expect(TokenType.IDENTIFIER).value);
        }

        return columns;
    }

    /**
     * Parse GROUP BY list with support for ROLLUP, CUBE, GROUPING SETS
     * Returns array of items, each with { type, column/columns/sets }
     */
    parseGroupByList() {
        const items = [];

        do {
            if (this.match(TokenType.ROLLUP)) {
                // ROLLUP(col1, col2, ...)
                this.expect(TokenType.LPAREN);
                const columns = this.parseColumnList();
                this.expect(TokenType.RPAREN);
                items.push({ type: 'ROLLUP', columns });
            } else if (this.match(TokenType.CUBE)) {
                // CUBE(col1, col2, ...)
                this.expect(TokenType.LPAREN);
                const columns = this.parseColumnList();
                this.expect(TokenType.RPAREN);
                items.push({ type: 'CUBE', columns });
            } else if (this.match(TokenType.GROUPING)) {
                // GROUPING SETS((col1, col2), (col1), ())
                this.expect(TokenType.SETS);
                this.expect(TokenType.LPAREN);
                const sets = this.parseGroupingSets();
                this.expect(TokenType.RPAREN);
                items.push({ type: 'GROUPING_SETS', sets });
            } else {
                // Simple column
                items.push({ type: 'COLUMN', column: this.expect(TokenType.IDENTIFIER).value });
            }
        } while (this.match(TokenType.COMMA));

        return items;
    }

    /**
     * Parse the sets inside GROUPING SETS(...)
     * Each set is (col1, col2) or () for grand total
     */
    parseGroupingSets() {
        const sets = [];

        do {
            this.expect(TokenType.LPAREN);
            if (this.check(TokenType.RPAREN)) {
                // Empty set () = grand total
                sets.push([]);
            } else {
                sets.push(this.parseColumnList());
            }
            this.expect(TokenType.RPAREN);
        } while (this.match(TokenType.COMMA));

        return sets;
    }

    parseOrderByList() {
        const items = [this.parseOrderByItem()];

        while (this.match(TokenType.COMMA)) {
            items.push(this.parseOrderByItem());
        }

        return items;
    }

    parseOrderByItem() {
        const column = this.expect(TokenType.IDENTIFIER).value;

        let descending = false;
        if (this.match(TokenType.DESC)) {
            descending = true;
        } else {
            this.match(TokenType.ASC);
        }

        return { column, descending };
    }

    // Expression parsing with precedence
    parseExpr() {
        return this.parseOrExpr();
    }

    parseOrExpr() {
        let left = this.parseAndExpr();

        while (this.match(TokenType.OR)) {
            const right = this.parseAndExpr();
            left = { type: 'binary', op: 'OR', left, right };
        }

        return left;
    }

    parseAndExpr() {
        let left = this.parseNotExpr();

        while (this.match(TokenType.AND)) {
            const right = this.parseNotExpr();
            left = { type: 'binary', op: 'AND', left, right };
        }

        return left;
    }

    parseNotExpr() {
        if (this.match(TokenType.NOT)) {
            const operand = this.parseNotExpr();
            return { type: 'unary', op: 'NOT', operand };
        }
        return this.parseCmpExpr();
    }

    parseCmpExpr() {
        let left = this.parseAddExpr();

        // IS NULL / IS NOT NULL
        if (this.match(TokenType.IS)) {
            const negated = !!this.match(TokenType.NOT);
            this.expect(TokenType.NULL);
            return {
                type: 'binary',
                op: negated ? '!=' : '==',
                left,
                right: { type: 'literal', value: null }
            };
        }

        // IN - can be a list of values or a subquery
        if (this.match(TokenType.IN)) {
            this.expect(TokenType.LPAREN);

            // Check if this is a subquery (starts with SELECT)
            if (this.check(TokenType.SELECT)) {
                const subquery = this.parseSelect(true);  // isSubquery=true
                this.expect(TokenType.RPAREN);
                return { type: 'in', expr: left, values: [{ type: 'subquery', query: subquery }] };
            }

            // Otherwise, parse as list of values
            const values = [];
            values.push(this.parsePrimary());
            while (this.match(TokenType.COMMA)) {
                values.push(this.parsePrimary());
            }
            this.expect(TokenType.RPAREN);
            return { type: 'in', expr: left, values };
        }

        // BETWEEN
        if (this.match(TokenType.BETWEEN)) {
            const low = this.parseAddExpr();
            this.expect(TokenType.AND);
            const high = this.parseAddExpr();
            return { type: 'between', expr: left, low, high };
        }

        // LIKE
        if (this.match(TokenType.LIKE)) {
            const pattern = this.parsePrimary();
            return { type: 'like', expr: left, pattern };
        }

        // NEAR - vector similarity search in WHERE clause
        if (this.match(TokenType.NEAR)) {
            const text = this.parsePrimary();
            return { type: 'near', column: left, text };
        }

        // Comparison operators
        const opMap = {
            [TokenType.EQ]: '==',
            [TokenType.NE]: '!=',
            [TokenType.LT]: '<',
            [TokenType.LE]: '<=',
            [TokenType.GT]: '>',
            [TokenType.GE]: '>=',
        };

        const opToken = this.match(TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE);
        if (opToken) {
            const right = this.parseAddExpr();
            return { type: 'binary', op: opMap[opToken.type], left, right };
        }

        return left;
    }

    parseAddExpr() {
        let left = this.parseMulExpr();

        while (true) {
            const opToken = this.match(TokenType.PLUS, TokenType.MINUS);
            if (!opToken) break;
            const right = this.parseMulExpr();
            left = { type: 'binary', op: opToken.value, left, right };
        }

        return left;
    }

    parseMulExpr() {
        let left = this.parseUnaryExpr();

        while (true) {
            const opToken = this.match(TokenType.STAR, TokenType.SLASH);
            if (!opToken) break;
            const right = this.parseUnaryExpr();
            left = { type: 'binary', op: opToken.value, left, right };
        }

        return left;
    }

    parseUnaryExpr() {
        if (this.match(TokenType.MINUS)) {
            const operand = this.parseUnaryExpr();
            return { type: 'unary', op: '-', operand };
        }
        return this.parsePrimary();
    }

    parsePrimary() {
        // NULL
        if (this.match(TokenType.NULL)) {
            return { type: 'literal', value: null };
        }

        // TRUE/FALSE
        if (this.match(TokenType.TRUE)) {
            return { type: 'literal', value: true };
        }
        if (this.match(TokenType.FALSE)) {
            return { type: 'literal', value: false };
        }

        // ARRAY[...] literal
        if (this.match(TokenType.ARRAY)) {
            let result = this.parseArrayLiteral();
            // Check for subscript: ARRAY[1,2,3][2]
            while (this.check(TokenType.LBRACKET)) {
                this.advance();
                const index = this.parseExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // Bare bracket array [...]
        if (this.check(TokenType.LBRACKET)) {
            let result = this.parseArrayLiteral();
            // Check for subscript: [1,2,3][2]
            while (this.check(TokenType.LBRACKET)) {
                this.advance();
                const index = this.parseExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // Number
        if (this.check(TokenType.NUMBER)) {
            const value = this.advance().value;
            return { type: 'literal', value: parseFloat(value) };
        }

        // String
        if (this.check(TokenType.STRING)) {
            const value = this.advance().value;
            return { type: 'literal', value };
        }

        // Window function keywords (ROW_NUMBER, RANK, etc.)
        const windowFuncTokens = [
            TokenType.ROW_NUMBER, TokenType.RANK, TokenType.DENSE_RANK, TokenType.NTILE,
            TokenType.LAG, TokenType.LEAD, TokenType.FIRST_VALUE, TokenType.LAST_VALUE, TokenType.NTH_VALUE,
            TokenType.PERCENT_RANK, TokenType.CUME_DIST
        ];
        if (windowFuncTokens.some(t => this.check(t))) {
            const name = this.advance().type;  // Use token type as function name
            this.expect(TokenType.LPAREN);
            const args = [];
            if (!this.check(TokenType.RPAREN)) {
                args.push(this.parseExpr());
                while (this.match(TokenType.COMMA)) {
                    args.push(this.parseExpr());
                }
            }
            this.expect(TokenType.RPAREN);

            // OVER clause is required for window functions
            const over = this.parseOverClause();
            return { type: 'call', name, args, distinct: false, over };
        }

        // Function call or column reference
        if (this.check(TokenType.IDENTIFIER) || this.check(TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX, TokenType.GROUPING)) {
            const name = this.advance().value;

            // Function call
            if (this.match(TokenType.LPAREN)) {
                let distinct = !!this.match(TokenType.DISTINCT);
                const args = [];

                if (!this.check(TokenType.RPAREN)) {
                    // Handle COUNT(*)
                    if (this.check(TokenType.STAR)) {
                        this.advance();
                        args.push({ type: 'star' });
                    } else {
                        args.push(this.parseExpr());
                        while (this.match(TokenType.COMMA)) {
                            args.push(this.parseExpr());
                        }
                    }
                }

                this.expect(TokenType.RPAREN);

                // Check for OVER clause (aggregate as window function)
                let over = null;
                if (this.check(TokenType.OVER)) {
                    over = this.parseOverClause();
                }

                return { type: 'call', name: name.toUpperCase(), args, distinct, over };
            }

            // Column reference - check for table.column syntax
            if (this.match(TokenType.DOT)) {
                // table.column (column can be a keyword like "text")
                const table = name;
                const token = this.advance();
                // Allow keywords as column names (e.g., c.text where TEXT is a keyword)
                const column = token.value || token.type.toLowerCase();
                return { type: 'column', table, column };
            }

            // Simple column reference
            let result = { type: 'column', column: name };

            // Check for array subscript: column[index]
            if (this.check(TokenType.LBRACKET)) {
                this.advance();  // consume [
                const index = this.parseExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }

            return result;
        }

        // Parenthesized expression or subquery
        if (this.match(TokenType.LPAREN)) {
            // Check if this is a subquery (starts with SELECT)
            if (this.check(TokenType.SELECT)) {
                const subquery = this.parseSelect(true);  // isSubquery=true
                this.expect(TokenType.RPAREN);
                return { type: 'subquery', query: subquery };
            }
            const expr = this.parseExpr();
            this.expect(TokenType.RPAREN);
            return expr;
        }

        // Star (for SELECT *)
        if (this.match(TokenType.STAR)) {
            return { type: 'star' };
        }

        throw new Error(`Unexpected token: ${this.current().type} (${this.current().value})`);
    }

    /**
     * Parse OVER clause for window functions
     * Syntax: OVER ([PARTITION BY expr, ...] [ORDER BY expr [ASC|DESC], ...] [frame_clause])
     */
    parseOverClause() {
        this.expect(TokenType.OVER);
        this.expect(TokenType.LPAREN);

        const over = { partitionBy: [], orderBy: [], frame: null };

        // PARTITION BY clause
        if (this.match(TokenType.PARTITION)) {
            this.expect(TokenType.BY);
            over.partitionBy.push(this.parseExpr());
            while (this.match(TokenType.COMMA)) {
                over.partitionBy.push(this.parseExpr());
            }
        }

        // ORDER BY clause
        if (this.match(TokenType.ORDER)) {
            this.expect(TokenType.BY);
            over.orderBy = this.parseOrderByList();
        }

        // Optional frame clause: ROWS/RANGE BETWEEN ... AND ...
        if (this.check(TokenType.ROWS) || this.check(TokenType.RANGE)) {
            over.frame = this.parseFrameClause();
        }

        this.expect(TokenType.RPAREN);
        return over;
    }

    /**
     * Parse frame clause for window functions
     * Syntax: ROWS|RANGE BETWEEN frame_start AND frame_end
     *         or: ROWS|RANGE frame_start
     */
    parseFrameClause() {
        const frameType = this.advance().type;  // ROWS or RANGE
        const frame = { type: frameType, start: null, end: null };

        // Check for BETWEEN ... AND ... syntax
        if (this.match(TokenType.BETWEEN)) {
            frame.start = this.parseFrameBound();
            this.expect(TokenType.AND);
            frame.end = this.parseFrameBound();
        } else {
            // Single bound (implies CURRENT ROW as end for some DBs, or just start)
            frame.start = this.parseFrameBound();
            frame.end = { type: 'CURRENT ROW' };  // Default end
        }

        return frame;
    }

    /**
     * Parse a frame bound
     * Options: UNBOUNDED PRECEDING, UNBOUNDED FOLLOWING, CURRENT ROW, N PRECEDING, N FOLLOWING
     */
    parseFrameBound() {
        if (this.match(TokenType.UNBOUNDED)) {
            if (this.match(TokenType.PRECEDING)) {
                return { type: 'UNBOUNDED PRECEDING' };
            } else if (this.match(TokenType.FOLLOWING)) {
                return { type: 'UNBOUNDED FOLLOWING' };
            }
            throw new Error('Expected PRECEDING or FOLLOWING after UNBOUNDED');
        }

        if (this.match(TokenType.CURRENT)) {
            this.expect(TokenType.ROW);
            return { type: 'CURRENT ROW' };
        }

        // N PRECEDING or N FOLLOWING
        if (this.check(TokenType.NUMBER)) {
            const n = parseInt(this.advance().value, 10);
            if (this.match(TokenType.PRECEDING)) {
                return { type: 'PRECEDING', offset: n };
            } else if (this.match(TokenType.FOLLOWING)) {
                return { type: 'FOLLOWING', offset: n };
            }
            throw new Error('Expected PRECEDING or FOLLOWING after number');
        }

        throw new Error('Invalid frame bound');
    }
}

/**
 * SQL Executor - executes parsed SQL against a LanceFile
 */

export { SQLParser };
