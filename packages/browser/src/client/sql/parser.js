/**
 * SQLParser - Recursive descent SQL parser
 */

import { SQLLexer, TokenType } from './lexer.js';
import * as Expr from './parser-expr.js';
import * as Advanced from './parser-advanced.js';

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
            ctes = Advanced.parseWithClause(this);
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
        let pivot = null;
        if (this.match(TokenType.PIVOT)) {
            pivot = Advanced.parsePivotClause(this, Expr.parsePrimary);
        }

        // UNPIVOT clause (optional)
        let unpivot = null;
        if (this.match(TokenType.UNPIVOT)) {
            unpivot = Advanced.parseUnpivotClause(this);
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
            groupBy = Advanced.parseGroupByList(this);
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
        let search = null;
        if (this.match(TokenType.NEAR)) {
            search = Advanced.parseNearClause(this);
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
        if (!noSetOps && (this.check(TokenType.UNION) || this.check(TokenType.INTERSECT) || this.check(TokenType.EXCEPT))) {
            const operator = this.advance().type;
            const all = !!this.match(TokenType.ALL);
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
     */
    parseInsert() {
        this.expect(TokenType.INSERT);
        this.expect(TokenType.INTO);

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
            values.push(Expr.parseValue(this));
            while (this.match(TokenType.COMMA)) {
                values.push(Expr.parseValue(this));
            }
            this.expect(TokenType.RPAREN);
            rows.push(values);
        } while (this.match(TokenType.COMMA));

        return { type: 'INSERT', table, columns, rows };
    }

    /**
     * Parse UPDATE statement
     */
    parseUpdate() {
        this.expect(TokenType.UPDATE);

        const table = this.expect(TokenType.IDENTIFIER).value;

        this.expect(TokenType.SET);

        const assignments = [];
        do {
            const column = this.expect(TokenType.IDENTIFIER).value;
            this.expect(TokenType.EQ);
            const value = Expr.parseValue(this);
            assignments.push({ column, value });
        } while (this.match(TokenType.COMMA));

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseExpr();
        }

        return { type: 'UPDATE', table, assignments, where };
    }

    /**
     * Parse DELETE statement
     */
    parseDelete() {
        this.expect(TokenType.DELETE);
        this.expect(TokenType.FROM);

        const table = this.expect(TokenType.IDENTIFIER).value;

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseExpr();
        }

        return { type: 'DELETE', table, where };
    }

    /**
     * Parse CREATE TABLE statement
     */
    parseCreateTable() {
        this.expect(TokenType.CREATE);
        this.expect(TokenType.TABLE);

        let ifNotExists = false;
        if (this.match(TokenType.IF)) {
            this.expect(TokenType.NOT);
            this.expect(TokenType.EXISTS);
            ifNotExists = true;
        }

        const table = this.expect(TokenType.IDENTIFIER).value;

        this.expect(TokenType.LPAREN);

        const columns = [];
        do {
            const name = this.expect(TokenType.IDENTIFIER).value;

            let dataType = 'TEXT';
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
                if (this.match(TokenType.LPAREN)) {
                    vectorDim = parseInt(this.expect(TokenType.NUMBER).value, 10);
                    this.expect(TokenType.RPAREN);
                }
            }

            if (this.match(TokenType.PRIMARY)) {
                this.expect(TokenType.KEY);
                primaryKey = true;
            }

            columns.push({ name, dataType, primaryKey, vectorDim });
        } while (this.match(TokenType.COMMA));

        this.expect(TokenType.RPAREN);

        return { type: 'CREATE_TABLE', table, columns, ifNotExists };
    }

    /**
     * Parse DROP TABLE statement
     */
    parseDropTable() {
        this.expect(TokenType.DROP);
        this.expect(TokenType.TABLE);

        let ifExists = false;
        if (this.match(TokenType.IF)) {
            this.expect(TokenType.EXISTS);
            ifExists = true;
        }

        const table = this.expect(TokenType.IDENTIFIER).value;

        return { type: 'DROP_TABLE', table, ifExists };
    }

    parseSelectList() {
        const items = [this.parseSelectItem()];

        while (this.match(TokenType.COMMA)) {
            items.push(this.parseSelectItem());
        }

        return items;
    }

    parseSelectItem() {
        if (this.match(TokenType.STAR)) {
            return { type: 'star' };
        }

        const expr = this.parseExpr();

        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.check(TokenType.IDENTIFIER) && !this.check(TokenType.FROM, TokenType.WHERE, TokenType.ORDER, TokenType.LIMIT, TokenType.GROUP, TokenType.JOIN, TokenType.INNER, TokenType.LEFT, TokenType.RIGHT, TokenType.COMMA)) {
            alias = this.advance().value;
        }

        return { type: 'expr', expr, alias };
    }

    /**
     * Parse FROM clause
     */
    parseFromClause() {
        let from = null;

        if (this.check(TokenType.STRING)) {
            const url = this.advance().value;
            from = { type: 'url', url };
        } else if (this.check(TokenType.IDENTIFIER)) {
            const name = this.advance().value;

            if (this.match(TokenType.LPAREN)) {
                const funcName = name.toLowerCase();
                if (funcName === 'read_lance') {
                    from = { type: 'url', function: 'read_lance' };

                    if (!this.check(TokenType.RPAREN)) {
                        if (this.match(TokenType.FILE)) {
                            from.isFile = true;
                            if (this.match(TokenType.COMMA)) {
                                from.version = parseInt(this.expect(TokenType.NUMBER).value, 10);
                            }
                        } else if (this.check(TokenType.STRING)) {
                            from.url = this.advance().value;
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
                from = { type: 'table', name };
            }
        } else {
            throw new Error('Expected table name, URL string, or read_lance() after FROM');
        }

        if (from) {
            if (this.match(TokenType.AS)) {
                from.alias = this.expect(TokenType.IDENTIFIER).value;
            } else if (this.check(TokenType.IDENTIFIER) && !this.check(TokenType.WHERE, TokenType.ORDER, TokenType.LIMIT, TokenType.GROUP, TokenType.NEAR, TokenType.JOIN, TokenType.INNER, TokenType.LEFT, TokenType.RIGHT, TokenType.COMMA)) {
                from.alias = this.advance().value;
            }
        }

        return from;
    }

    /**
     * Parse JOIN clause
     */
    parseJoinClause() {
        let joinType = 'INNER';

        if (this.match(TokenType.INNER)) {
            this.expect(TokenType.JOIN);
            joinType = 'INNER';
        } else if (this.match(TokenType.LEFT)) {
            this.match(TokenType.OUTER);
            this.expect(TokenType.JOIN);
            joinType = 'LEFT';
        } else if (this.match(TokenType.RIGHT)) {
            this.match(TokenType.OUTER);
            this.expect(TokenType.JOIN);
            joinType = 'RIGHT';
        } else if (this.match(TokenType.FULL)) {
            this.match(TokenType.OUTER);
            this.expect(TokenType.JOIN);
            joinType = 'FULL';
        } else if (this.match(TokenType.CROSS)) {
            this.expect(TokenType.JOIN);
            joinType = 'CROSS';
        } else {
            this.expect(TokenType.JOIN);
        }

        const table = this.parseFromClause();
        const alias = table.alias || null;

        let on = null;
        if (joinType !== 'CROSS') {
            this.expect(TokenType.ON);
            on = this.parseExpr();
        }

        return { type: joinType, table, alias, on };
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

    // Expression parsing - delegate to parser-expr.js
    parseExpr() {
        return Expr.parseExpr(this);
    }

    // Window function OVER clause - delegate to parser-advanced.js
    parseOverClause() {
        return Advanced.parseOverClause(this);
    }
}

export { SQLParser };
