/**
 * SQL Parser - Recursive descent parser for SQL statements
 */

import { TokenType } from './tokenizer.js';

class SQLParser {
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

    check(type) {
        return this.peek().type === type;
    }

    isKeyword(keyword) {
        const token = this.peek();
        const upper = keyword.toUpperCase();
        return token.type === TokenType[upper] ||
               (token.type === TokenType.IDENTIFIER && token.value.toUpperCase() === upper);
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
            case TokenType.EXPLAIN:
                return this.parseExplain();
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
            case TokenType.WITH:
                return this.parseWithClause();
            default:
                throw new Error(`Unexpected token: ${token.type}`);
        }
    }

    parseExplain() {
        this.expect(TokenType.EXPLAIN);
        const analyze = !!this.match(TokenType.ANALYZE);

        // Parse the statement to explain (SELECT, UPDATE, DELETE, INSERT)
        const stmtToken = this.peek();
        let statement;
        switch (stmtToken.type) {
            case TokenType.SELECT:
                statement = this.parseSelect();
                break;
            case TokenType.WITH:
                statement = this.parseWithClause();
                break;
            case TokenType.UPDATE:
                statement = this.parseUpdate();
                break;
            case TokenType.DELETE:
                statement = this.parseDelete();
                break;
            case TokenType.INSERT:
                statement = this.parseInsert();
                break;
            default:
                throw new Error(`EXPLAIN not supported for: ${stmtToken.type}`);
        }

        return { type: 'EXPLAIN', analyze, statement };
    }

    parseWithClause() {
        this.expect(TokenType.WITH);
        const recursive = !!this.match(TokenType.RECURSIVE);

        // Parse CTEs
        const ctes = [];
        do {
            const cteName = this.expect(TokenType.IDENTIFIER).value;
            this.expect(TokenType.AS);
            this.expect(TokenType.LPAREN);
            const cteQuery = this.parseSelect();
            this.expect(TokenType.RPAREN);
            ctes.push({ name: cteName, query: cteQuery });
        } while (this.match(TokenType.COMMA));

        // Parse main SELECT
        const mainQuery = this.parseSelect();

        return {
            type: 'WITH',
            recursive,
            ctes,
            query: mainQuery
        };
    }

    parseCreate() {
        this.expect(TokenType.CREATE);
        this.expect(TokenType.TABLE);

        let ifNotExists = false;
        if (this.match(TokenType.IF)) {
            this.expect(TokenType.NOT);
            this.expect(TokenType.EXISTS);
            ifNotExists = true;
        }

        const tableName = this.expect(TokenType.IDENTIFIER).value;
        this.expect(TokenType.LPAREN);

        const columns = [];
        do {
            const colName = this.expect(TokenType.IDENTIFIER).value;
            const colType = this.parseDataType();
            const col = { name: colName, type: colType };

            if (this.match(TokenType.PRIMARY)) {
                this.expect(TokenType.KEY);
                col.primaryKey = true;
            }

            columns.push(col);
        } while (this.match(TokenType.COMMA));

        this.expect(TokenType.RPAREN);

        return { type: 'CREATE_TABLE', table: tableName, columns, ifNotExists };
    }

    parseDataType() {
        const token = this.advance();
        let type = token.value || token.type;

        // Handle VECTOR(dim)
        if (type.toUpperCase() === 'VECTOR' && this.match(TokenType.LPAREN)) {
            const dim = this.expect(TokenType.NUMBER).value;
            this.expect(TokenType.RPAREN);
            return { type: 'vector', dim: parseInt(dim) };
        }

        // Handle VARCHAR(len)
        if ((type.toUpperCase() === 'VARCHAR' || type.toUpperCase() === 'TEXT') && this.match(TokenType.LPAREN)) {
            this.expect(TokenType.NUMBER);
            this.expect(TokenType.RPAREN);
        }

        return type;
    }

    parseDrop() {
        this.expect(TokenType.DROP);
        this.expect(TokenType.TABLE);

        let ifExists = false;
        if (this.match(TokenType.IF)) {
            this.expect(TokenType.EXISTS);
            ifExists = true;
        }

        const tableName = this.expect(TokenType.IDENTIFIER).value;
        return { type: 'DROP_TABLE', table: tableName, ifExists };
    }

    parseInsert() {
        this.expect(TokenType.INSERT);
        this.expect(TokenType.INTO);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        let columns = null;
        if (this.match(TokenType.LPAREN)) {
            columns = [this.expect(TokenType.IDENTIFIER).value];
            while (this.match(TokenType.COMMA)) {
                columns.push(this.expect(TokenType.IDENTIFIER).value);
            }
            this.expect(TokenType.RPAREN);
        }

        // Check for INSERT...SELECT vs INSERT...VALUES
        if (this.check(TokenType.SELECT)) {
            const selectQuery = this.parseSelect();
            return { type: 'INSERT', table: tableName, columns, select: selectQuery };
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

            if (columns) {
                const row = {};
                columns.forEach((col, i) => row[col] = values[i]);
                rows.push(row);
            } else {
                rows.push(values);
            }
        } while (this.match(TokenType.COMMA));

        // Check for ON CONFLICT clause (UPSERT)
        let onConflict = null;
        if (this.match(TokenType.ON)) {
            this.expect(TokenType.CONFLICT);

            // Optional conflict target: (column1, column2)
            let conflictColumns = null;
            if (this.match(TokenType.LPAREN)) {
                conflictColumns = [this.expect(TokenType.IDENTIFIER).value];
                while (this.match(TokenType.COMMA)) {
                    conflictColumns.push(this.expect(TokenType.IDENTIFIER).value);
                }
                this.expect(TokenType.RPAREN);
            }

            this.expect(TokenType.DO);

            if (this.match(TokenType.NOTHING)) {
                onConflict = { action: 'nothing', columns: conflictColumns };
            } else if (this.match(TokenType.UPDATE)) {
                this.expect(TokenType.SET);
                const updates = {};
                do {
                    const col = this.expect(TokenType.IDENTIFIER).value;
                    this.expect(TokenType.EQ);
                    updates[col] = this.parseArithmeticExpr();
                } while (this.match(TokenType.COMMA));
                onConflict = { action: 'update', columns: conflictColumns, updates };
            }
        }

        return { type: 'INSERT', table: tableName, columns, rows, onConflict };
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
        // Vector literal
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

        // Optional alias
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.check(TokenType.IDENTIFIER) && !this.isKeyword('SET')) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        }

        this.expect(TokenType.SET);

        const updates = {};
        do {
            const col = this.expect(TokenType.IDENTIFIER).value;
            this.expect(TokenType.EQ);
            updates[col] = this.parseArithmeticExpr();
        } while (this.match(TokenType.COMMA));

        // Optional FROM clause for JOINs
        let from = null;
        if (this.match(TokenType.FROM)) {
            from = [];
            do {
                const tbl = this.expect(TokenType.IDENTIFIER).value;
                let tblAlias = null;
                if (this.match(TokenType.AS)) {
                    tblAlias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.check(TokenType.IDENTIFIER) && !this.isKeyword('WHERE')) {
                    tblAlias = this.expect(TokenType.IDENTIFIER).value;
                }
                from.push({ name: tbl, alias: tblAlias });
            } while (this.match(TokenType.COMMA));
        }

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        return { type: 'UPDATE', table: tableName, alias, updates, from, where };
    }

    parseDelete() {
        this.expect(TokenType.DELETE);
        this.expect(TokenType.FROM);
        const tableName = this.expect(TokenType.IDENTIFIER).value;

        // Optional alias
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.check(TokenType.IDENTIFIER) && !this.isKeyword('USING') && !this.isKeyword('WHERE')) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        }

        // Optional USING clause for JOINs
        let using = null;
        if (this.match(TokenType.USING)) {
            using = [];
            do {
                const tbl = this.expect(TokenType.IDENTIFIER).value;
                let tblAlias = null;
                if (this.match(TokenType.AS)) {
                    tblAlias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.check(TokenType.IDENTIFIER) && !this.isKeyword('WHERE')) {
                    tblAlias = this.expect(TokenType.IDENTIFIER).value;
                }
                using.push({ name: tbl, alias: tblAlias });
            } while (this.match(TokenType.COMMA));
        }

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        return { type: 'DELETE', table: tableName, alias, using, where };
    }

    parseSelect() {
        this.expect(TokenType.SELECT);

        // Check for DISTINCT
        const distinct = !!this.match(TokenType.DISTINCT);

        // Parse column list (may include aggregate functions, table.column, etc.)
        const columns = [];
        columns.push(this.parseSelectColumn());
        while (this.match(TokenType.COMMA)) {
            columns.push(this.parseSelectColumn());
        }

        // FROM clause is optional (for SELECT without tables like SELECT 1+1, SELECT JSON_OBJECT(...))
        const tables = [];
        const joins = [];
        let table = null;

        if (this.match(TokenType.FROM)) {
            // Parse table with optional alias
            tables.push(this.parseTableRef());

            // Parse JOINs
            while (this.peek().type === TokenType.JOIN ||
                   this.peek().type === TokenType.LEFT ||
                   this.peek().type === TokenType.RIGHT ||
                   this.peek().type === TokenType.INNER ||
                   this.peek().type === TokenType.FULL ||
                   this.peek().type === TokenType.CROSS) {
                joins.push(this.parseJoin());
            }
            table = tables[0].name;
        }

        let where = null;
        if (this.match(TokenType.WHERE)) {
            where = this.parseWhereExpr();
        }

        // Parse GROUP BY (with ROLLUP/CUBE/GROUPING SETS support)
        let groupBy = null;
        if (this.match(TokenType.GROUP)) {
            this.expect(TokenType.BY);

            // Check for ROLLUP
            if (this.match(TokenType.ROLLUP)) {
                this.expect(TokenType.LPAREN);
                const columns = [this.parseColumnRef()];
                while (this.match(TokenType.COMMA)) {
                    columns.push(this.parseColumnRef());
                }
                this.expect(TokenType.RPAREN);
                groupBy = { type: 'ROLLUP', columns };
            }
            // Check for CUBE
            else if (this.match(TokenType.CUBE)) {
                this.expect(TokenType.LPAREN);
                const columns = [this.parseColumnRef()];
                while (this.match(TokenType.COMMA)) {
                    columns.push(this.parseColumnRef());
                }
                this.expect(TokenType.RPAREN);
                groupBy = { type: 'CUBE', columns };
            }
            // Check for GROUPING SETS
            else if (this.match(TokenType.GROUPING)) {
                this.expect(TokenType.SETS);
                this.expect(TokenType.LPAREN);
                const sets = [];
                do {
                    this.expect(TokenType.LPAREN);
                    const setCols = [];
                    if (!this.check(TokenType.RPAREN)) {
                        setCols.push(this.parseColumnRef());
                        while (this.match(TokenType.COMMA)) {
                            setCols.push(this.parseColumnRef());
                        }
                    }
                    this.expect(TokenType.RPAREN);
                    sets.push(setCols);
                } while (this.match(TokenType.COMMA));
                this.expect(TokenType.RPAREN);
                groupBy = { type: 'GROUPING_SETS', sets };
            }
            // Standard GROUP BY
            else {
                groupBy = [this.parseColumnRef()];
                while (this.match(TokenType.COMMA)) {
                    groupBy.push(this.parseColumnRef());
                }
            }
        }

        // Parse HAVING
        let having = null;
        if (this.match(TokenType.HAVING)) {
            having = this.parseWhereExpr();
        }

        // Parse QUALIFY (filter on window function results)
        let qualify = null;
        if (this.match(TokenType.QUALIFY)) {
            qualify = this.parseWhereExpr();
        }

        let orderBy = null;
        if (this.match(TokenType.ORDER)) {
            this.expect(TokenType.BY);
            orderBy = [];
            do {
                const column = this.parseColumnRef();
                const desc = !!this.match(TokenType.DESC);
                if (!desc) this.match(TokenType.ASC);
                // Parse NULLS FIRST/LAST
                let nullsFirst = null;
                if (this.match(TokenType.NULLS)) {
                    if (this.match(TokenType.FIRST)) {
                        nullsFirst = true;
                    } else if (this.match(TokenType.LAST)) {
                        nullsFirst = false;
                    }
                }
                orderBy.push({ column, desc, nullsFirst });
            } while (this.match(TokenType.COMMA));
        }

        let limit = null;
        if (this.match(TokenType.LIMIT)) {
            limit = parseInt(this.expect(TokenType.NUMBER).value);
        }

        let offset = null;
        if (this.match(TokenType.OFFSET)) {
            offset = parseInt(this.expect(TokenType.NUMBER).value);
        }

        // For backwards compatibility, use first table name as 'table' (null if no FROM)
        const tableName = tables.length > 0 ? tables[0].name : null;

        const selectAst = {
            type: 'SELECT',
            table: tableName,
            tables,
            columns,
            distinct,
            joins,
            where,
            groupBy,
            having,
            qualify,
            orderBy,
            limit,
            offset
        };

        // Check for set operations (UNION, INTERSECT, EXCEPT)
        if (this.match(TokenType.UNION)) {
            const all = this.match(TokenType.ALL);
            const right = this.parseSelect();
            return { type: 'UNION', all: !!all, left: selectAst, right };
        }
        if (this.match(TokenType.INTERSECT)) {
            const all = this.match(TokenType.ALL);
            const right = this.parseSelect();
            return { type: 'INTERSECT', all: !!all, left: selectAst, right };
        }
        if (this.match(TokenType.EXCEPT)) {
            const all = this.match(TokenType.ALL);
            const right = this.parseSelect();
            return { type: 'EXCEPT', all: !!all, left: selectAst, right };
        }

        // Check for PIVOT/UNPIVOT transformations
        if (this.match(TokenType.PIVOT)) {
            return this.parsePivot(selectAst);
        }
        if (this.match(TokenType.UNPIVOT)) {
            return this.parseUnpivot(selectAst);
        }

        return selectAst;
    }

    parsePivot(selectAst) {
        this.expect(TokenType.LPAREN);
        const aggToken = this.advance();
        const aggFunc = (aggToken.value || aggToken.type).toUpperCase();
        this.expect(TokenType.LPAREN);
        const valueColumn = this.parseColumnRef();
        this.expect(TokenType.RPAREN);
        this.expect(TokenType.FOR);
        const pivotColumn = this.parseColumnRef();
        this.expect(TokenType.IN);
        this.expect(TokenType.LPAREN);
        const pivotValues = [];
        do {
            pivotValues.push(this.expect(TokenType.STRING).value);
        } while (this.match(TokenType.COMMA));
        this.expect(TokenType.RPAREN);
        this.expect(TokenType.RPAREN);

        return {
            type: 'PIVOT',
            select: selectAst,
            aggFunc,
            valueColumn: typeof valueColumn === 'string' ? valueColumn : valueColumn.column,
            pivotColumn: typeof pivotColumn === 'string' ? pivotColumn : pivotColumn.column,
            pivotValues
        };
    }

    parseUnpivot(selectAst) {
        // Helper to get identifier (allows keywords to be used as column names)
        const getIdentifier = () => {
            const token = this.advance();
            // Accept IDENTIFIER or any keyword that has a value (keywords store their value)
            return token.value || token.type;
        };

        this.expect(TokenType.LPAREN);
        const valueColumn = getIdentifier();
        this.expect(TokenType.FOR);
        const nameColumn = getIdentifier();
        this.expect(TokenType.IN);
        this.expect(TokenType.LPAREN);
        const unpivotColumns = [];
        do {
            unpivotColumns.push(getIdentifier());
        } while (this.match(TokenType.COMMA));
        this.expect(TokenType.RPAREN);
        this.expect(TokenType.RPAREN);

        return {
            type: 'UNPIVOT',
            select: selectAst,
            valueColumn,
            nameColumn,
            unpivotColumns
        };
    }

    // Parse a single column in SELECT clause
    parseSelectColumn() {
        // Check for * first
        if (this.match(TokenType.STAR)) {
            return { type: 'star', value: '*' };
        }

        // Check for scalar subquery: (SELECT ...)
        if (this.check(TokenType.LPAREN)) {
            const savedPos = this.pos;
            this.advance(); // consume (
            if (this.check(TokenType.SELECT)) {
                const subquery = this.parseSelect();
                this.expect(TokenType.RPAREN);
                let alias = null;
                if (this.match(TokenType.AS)) {
                    alias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.peek().type === TokenType.IDENTIFIER) {
                    const nextType = this.peek().type;
                    if (nextType !== TokenType.FROM && nextType !== TokenType.COMMA &&
                        nextType !== TokenType.WHERE && nextType !== TokenType.ORDER &&
                        nextType !== TokenType.GROUP && nextType !== TokenType.LIMIT) {
                        alias = this.advance().value;
                    }
                }
                return { type: 'scalar_subquery', subquery, alias };
            }
            // Not a subquery, restore position
            this.pos = savedPos;
        }

        // Check for CASE expression
        if (this.match(TokenType.CASE)) {
            const caseExpr = this.parseCaseExpr();
            let alias = null;
            if (this.match(TokenType.AS)) {
                alias = this.expect(TokenType.IDENTIFIER).value;
            }
            return { type: 'case', expr: caseExpr, alias };
        }

        // Check for window functions: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, NTILE, PERCENT_RANK, CUME_DIST, FIRST_VALUE, LAST_VALUE, NTH_VALUE
        const windowFuncs = [TokenType.ROW_NUMBER, TokenType.RANK, TokenType.DENSE_RANK, TokenType.LAG, TokenType.LEAD,
                             TokenType.NTILE, TokenType.PERCENT_RANK, TokenType.CUME_DIST, TokenType.FIRST_VALUE, TokenType.LAST_VALUE, TokenType.NTH_VALUE];
        for (const funcType of windowFuncs) {
            if (this.match(funcType)) {
                const funcName = funcType.toLowerCase();
                this.expect(TokenType.LPAREN);
                let arg = null;
                let args = [];
                // Handle function arguments
                if (funcName === 'lag' || funcName === 'lead' || funcName === 'first_value' || funcName === 'last_value') {
                    if (!this.check(TokenType.RPAREN)) {
                        arg = this.parseColumnRef();
                    }
                } else if (funcName === 'ntile') {
                    // NTILE takes a number argument
                    arg = parseInt(this.expect(TokenType.NUMBER).value, 10);
                } else if (funcName === 'nth_value') {
                    // NTH_VALUE(col, n)
                    arg = this.parseColumnRef();
                    if (this.match(TokenType.COMMA)) {
                        args.push(arg);
                        args.push(parseInt(this.expect(TokenType.NUMBER).value, 10));
                    }
                }
                this.expect(TokenType.RPAREN);
                this.expect(TokenType.OVER);
                const windowSpec = this.parseOverSpec();

                // Optional alias
                let alias = null;
                if (this.match(TokenType.AS)) {
                    alias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.peek().type === TokenType.IDENTIFIER) {
                    alias = this.advance().value;
                }

                return { type: 'window', func: funcName, arg, args: args.length ? args : null, over: windowSpec, alias };
            }
        }

        // Check for aggregate functions
        const aggFuncs = [TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX];
        for (const funcType of aggFuncs) {
            if (this.match(funcType)) {
                const funcName = funcType;
                this.expect(TokenType.LPAREN);
                let arg;
                if (this.match(TokenType.STAR)) {
                    arg = '*';
                } else if (this.match(TokenType.DISTINCT)) {
                    const col = this.parseColumnRef();
                    arg = { distinct: true, column: col };
                } else {
                    arg = this.parseColumnRef();
                }
                this.expect(TokenType.RPAREN);

                // Check for OVER (window function)
                if (this.match(TokenType.OVER)) {
                    const windowSpec = this.parseOverSpec();
                    let alias = null;
                    if (this.match(TokenType.AS)) {
                        alias = this.expect(TokenType.IDENTIFIER).value;
                    } else if (this.peek().type === TokenType.IDENTIFIER) {
                        alias = this.advance().value;
                    }
                    return { type: 'window', func: funcName.toLowerCase(), arg, over: windowSpec, alias };
                }

                // Optional alias
                let alias = null;
                if (this.match(TokenType.AS)) {
                    alias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.peek().type === TokenType.IDENTIFIER) {
                    alias = this.advance().value;
                }

                return { type: 'aggregate', func: funcName.toLowerCase(), arg, alias };
            }
        }

        // Check for string-based aggregate functions (STDDEV, VARIANCE, MEDIAN, STRING_AGG)
        const stringAggFuncs = ['STDDEV', 'STDDEV_SAMP', 'STDDEV_POP', 'VARIANCE', 'VAR_SAMP', 'VAR_POP', 'MEDIAN', 'STRING_AGG', 'GROUP_CONCAT'];
        if (this.peek().type === TokenType.IDENTIFIER) {
            const funcName = this.peek().value.toUpperCase();
            if (stringAggFuncs.includes(funcName)) {
                this.advance();
                this.expect(TokenType.LPAREN);
                let arg;
                let separator = null;

                if (this.match(TokenType.DISTINCT)) {
                    const col = this.parseColumnRef();
                    arg = { distinct: true, column: col };
                } else {
                    arg = this.parseColumnRef();
                }

                // STRING_AGG has a second argument for separator
                if ((funcName === 'STRING_AGG' || funcName === 'GROUP_CONCAT') && this.match(TokenType.COMMA)) {
                    separator = this.expect(TokenType.STRING).value;
                    arg = { column: arg, separator };
                }

                this.expect(TokenType.RPAREN);

                // Check for OVER (window function) - some aggregates can be windowed
                if (this.match(TokenType.OVER)) {
                    const windowSpec = this.parseOverSpec();
                    let alias = null;
                    if (this.match(TokenType.AS)) {
                        alias = this.expect(TokenType.IDENTIFIER).value;
                    } else if (this.peek().type === TokenType.IDENTIFIER) {
                        alias = this.advance().value;
                    }
                    return { type: 'window', func: funcName.toLowerCase(), arg, over: windowSpec, alias };
                }

                // Optional alias
                let alias = null;
                if (this.match(TokenType.AS)) {
                    alias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.peek().type === TokenType.IDENTIFIER) {
                    alias = this.advance().value;
                }

                return { type: 'aggregate', func: funcName.toLowerCase(), arg, alias };
            }
        }

        // Parse arithmetic expression (handles columns, literals, functions, and arithmetic operators)
        const expr = this.parseArithmeticExpr();

        // Optional alias
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.peek().type === TokenType.IDENTIFIER) {
            const nextType = this.peek().type;
            if (nextType !== TokenType.FROM && nextType !== TokenType.COMMA &&
                nextType !== TokenType.WHERE && nextType !== TokenType.ORDER &&
                nextType !== TokenType.GROUP && nextType !== TokenType.LIMIT) {
                alias = this.advance().value;
            }
        }

        // Convert result to appropriate type for backward compatibility
        if (expr.type === 'column') {
            return { type: 'column', value: expr.value, alias };
        } else if (expr.type === 'function') {
            return { type: 'function', func: expr.func, args: expr.args, alias };
        } else if (expr.type === 'literal') {
            return { type: 'literal', value: expr.value, alias };
        } else if (expr.type === 'arithmetic') {
            return { type: 'arithmetic', expr, alias };
        }

        // Fallback
        return { type: 'arithmetic', expr, alias };
    }

    // Parse function arguments (comma-separated expressions)
    parseFunctionArgs() {
        const args = [];
        if (this.peek().type !== TokenType.RPAREN) {
            args.push(this.parseFunctionArg());
            while (this.match(TokenType.COMMA)) {
                args.push(this.parseFunctionArg());
            }
        }
        return args;
    }

    // Parse a single function argument (can be column, literal, or nested function)
    parseFunctionArg() {
        // Check for AS keyword (for CAST ... AS type)
        if (this.peek().type === TokenType.AS) {
            return null; // Will be handled by CAST specially
        }

        // Check for nested function call
        const scalarFuncs = ['COALESCE', 'NULLIF', 'UPPER', 'LOWER', 'LENGTH', 'SUBSTR', 'SUBSTRING',
                            'TRIM', 'LTRIM', 'RTRIM', 'CONCAT', 'REPLACE', 'ABS', 'ROUND', 'CEIL',
                            'CEILING', 'FLOOR', 'MOD', 'POWER', 'POW', 'SQRT',
                            // Date/Time functions
                            'NOW', 'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP',
                            'DATE', 'TIME', 'STRFTIME', 'DATE_DIFF', 'DATE_ADD', 'DATE_SUB',
                            'EXTRACT', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND',
                            // Additional string functions
                            'SPLIT', 'LEFT', 'RIGHT', 'LPAD', 'RPAD', 'POSITION', 'INSTR', 'REPEAT', 'REVERSE',
                            // Conditional functions
                            'GREATEST', 'LEAST', 'IIF', 'IF',
                            // Additional math functions
                            'LOG', 'LOG10', 'LN', 'EXP', 'SIN', 'COS', 'TAN', 'ASIN', 'ACOS', 'ATAN', 'ATAN2',
                            'PI', 'RANDOM', 'RAND', 'SIGN', 'DEGREES', 'RADIANS', 'TRUNCATE', 'TRUNC',
                            // REGEXP functions
                            'REGEXP_MATCHES', 'REGEXP_REPLACE', 'REGEXP_EXTRACT', 'REGEXP_SUBSTR',
                            'REGEXP_SPLIT', 'REGEXP_COUNT',
                            // JSON functions
                            'JSON_EXTRACT', 'JSON_VALUE', 'JSON_OBJECT', 'JSON_ARRAY',
                            'JSON_KEYS', 'JSON_LENGTH', 'JSON_TYPE', 'JSON_VALID',
                            // Array functions
                            'ARRAY_LENGTH', 'ARRAY_CONTAINS', 'ARRAY_POSITION', 'ARRAY_APPEND',
                            'ARRAY_REMOVE', 'ARRAY_SLICE', 'ARRAY_CONCAT', 'UNNEST',
                            // UUID functions
                            'UUID', 'GEN_RANDOM_UUID', 'UUID_STRING', 'IS_UUID',
                            // Binary/Bit functions
                            'BIT_COUNT', 'HEX', 'UNHEX', 'ENCODE', 'DECODE'];
        if (this.peek().type === TokenType.IDENTIFIER) {
            const funcName = this.peek().value.toUpperCase();
            if (scalarFuncs.includes(funcName)) {
                this.advance();
                this.expect(TokenType.LPAREN);
                const args = this.parseFunctionArgs();
                this.expect(TokenType.RPAREN);
                return { type: 'function', func: funcName.toLowerCase(), args };
            }
        }

        // String literal
        if (this.peek().type === TokenType.STRING) {
            const left = { type: 'literal', value: this.advance().value };
            return this.tryParseComparisonExpr(left);
        }

        // Number literal
        if (this.peek().type === TokenType.NUMBER) {
            const left = { type: 'literal', value: parseFloat(this.advance().value) };
            return this.tryParseComparisonExpr(left);
        }

        // NULL
        if (this.match(TokenType.NULL)) {
            return { type: 'literal', value: null };
        }

        // ARRAY literal: ARRAY[1, 2, 3]
        if (this.match(TokenType.ARRAY)) {
            const elements = [];
            if (this.match(TokenType.LBRACKET)) {
                if (!this.check(TokenType.RBRACKET)) {
                    elements.push(this.parseArithmeticExpr());
                    while (this.match(TokenType.COMMA)) {
                        elements.push(this.parseArithmeticExpr());
                    }
                }
                this.expect(TokenType.RBRACKET);
            }
            let result = { type: 'array_literal', elements };
            // Check for subscript
            while (this.match(TokenType.LBRACKET)) {
                const index = this.parseArithmeticExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // Bare bracket array: [1, 2, 3]
        if (this.match(TokenType.LBRACKET)) {
            const elements = [];
            if (!this.check(TokenType.RBRACKET)) {
                elements.push(this.parseArithmeticExpr());
                while (this.match(TokenType.COMMA)) {
                    elements.push(this.parseArithmeticExpr());
                }
            }
            this.expect(TokenType.RBRACKET);
            let result = { type: 'array_literal', elements };
            // Check for subscript
            while (this.match(TokenType.LBRACKET)) {
                const index = this.parseArithmeticExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // Column reference
        const col = this.parseColumnRef();
        const left = { type: 'column', value: col };
        return this.tryParseComparisonExpr(left);
    }

    // Check if there's a comparison operator and parse as comparison expression
    tryParseComparisonExpr(left) {
        // Check for comparison operators
        let op = null;
        if (this.match(TokenType.EQ)) op = '=';
        else if (this.match(TokenType.NE)) op = '!=';
        else if (this.match(TokenType.LT)) op = '<';
        else if (this.match(TokenType.LE)) op = '<=';
        else if (this.match(TokenType.GT)) op = '>';
        else if (this.match(TokenType.GE)) op = '>=';

        if (!op) return left;

        // Parse right side of comparison
        let right;
        if (this.peek().type === TokenType.STRING) {
            right = { type: 'literal', value: this.advance().value };
        } else if (this.peek().type === TokenType.NUMBER) {
            right = { type: 'literal', value: parseFloat(this.advance().value) };
        } else if (this.match(TokenType.NULL)) {
            right = { type: 'literal', value: null };
        } else {
            const col = this.parseColumnRef();
            right = { type: 'column', value: col };
        }

        return { type: 'comparison', op, left, right };
    }

    // Parse CASE WHEN ... THEN ... ELSE ... END
    parseCaseExpr() {
        const branches = [];

        // Simple CASE: CASE expr WHEN value THEN result
        // Searched CASE: CASE WHEN condition THEN result
        let caseExpr = null;
        if (this.peek().type !== TokenType.WHEN) {
            caseExpr = this.parseFunctionArg();
        }

        while (this.match(TokenType.WHEN)) {
            const condition = this.parseFunctionArg();
            this.expect(TokenType.THEN);
            const result = this.parseFunctionArg();
            branches.push({ condition, result });
        }

        let elseResult = null;
        if (this.match(TokenType.ELSE)) {
            elseResult = this.parseFunctionArg();
        }

        this.expect(TokenType.END);
        return { caseExpr, branches, elseResult };
    }

    // ========== Arithmetic Expression Parsing ==========
    // Parse arithmetic expression with proper precedence: () > * / > + -
    parseArithmeticExpr() {
        return this.parseAddSub();
    }

    parseAddSub() {
        let left = this.parseMulDiv();
        while (this.peek().type === TokenType.PLUS || this.peek().type === TokenType.MINUS) {
            const op = this.advance().type === TokenType.PLUS ? '+' : '-';
            const right = this.parseMulDiv();
            left = { type: 'arithmetic', op, left, right };
        }
        return left;
    }

    parseMulDiv() {
        let left = this.parseBitwise();
        while (this.peek().type === TokenType.STAR || this.peek().type === TokenType.SLASH) {
            const op = this.advance().type === TokenType.STAR ? '*' : '/';
            const right = this.parseBitwise();
            left = { type: 'arithmetic', op, left, right };
        }
        return left;
    }

    parseBitwise() {
        let left = this.parseUnary();
        const bitwiseOps = [TokenType.AMPERSAND, TokenType.PIPE, TokenType.CARET, TokenType.LSHIFT, TokenType.RSHIFT];
        while (bitwiseOps.includes(this.peek().type)) {
            const token = this.advance();
            let op;
            switch (token.type) {
                case TokenType.AMPERSAND: op = '&'; break;
                case TokenType.PIPE: op = '|'; break;
                case TokenType.CARET: op = '^'; break;
                case TokenType.LSHIFT: op = '<<'; break;
                case TokenType.RSHIFT: op = '>>'; break;
            }
            const right = this.parseUnary();
            left = { type: 'arithmetic', op, left, right };
        }
        return left;
    }

    parseUnary() {
        // Handle unary minus
        if (this.match(TokenType.MINUS)) {
            const operand = this.parseUnary();
            return { type: 'arithmetic', op: 'unary-', operand };
        }
        // Handle bitwise NOT
        if (this.match(TokenType.TILDE)) {
            const operand = this.parseUnary();
            return { type: 'arithmetic', op: 'unary~', operand };
        }
        return this.parseArithmeticPrimary();
    }

    parseArithmeticPrimary() {
        // Parenthesized expression
        if (this.match(TokenType.LPAREN)) {
            const expr = this.parseArithmeticExpr();
            this.expect(TokenType.RPAREN);
            return expr;
        }

        // Number literal
        if (this.peek().type === TokenType.NUMBER) {
            return { type: 'literal', value: parseFloat(this.advance().value) };
        }

        // String literal
        if (this.peek().type === TokenType.STRING) {
            return { type: 'literal', value: this.advance().value };
        }

        // NULL literal
        if (this.match(TokenType.NULL)) {
            return { type: 'literal', value: null };
        }

        // ARRAY constructor: ARRAY[1, 2, 3] with optional subscript ARRAY[1,2,3][1]
        if (this.match(TokenType.ARRAY)) {
            const elements = [];
            if (this.match(TokenType.LBRACKET)) {
                if (!this.check(TokenType.RBRACKET)) {
                    elements.push(this.parseArithmeticExpr());
                    while (this.match(TokenType.COMMA)) {
                        elements.push(this.parseArithmeticExpr());
                    }
                }
                this.expect(TokenType.RBRACKET);
            }
            let result = { type: 'array_literal', elements };
            // Check for subscript
            while (this.match(TokenType.LBRACKET)) {
                const index = this.parseArithmeticExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // Bare bracket array: [1, 2, 3] with optional subscript
        if (this.match(TokenType.LBRACKET)) {
            const elements = [];
            if (!this.check(TokenType.RBRACKET)) {
                elements.push(this.parseArithmeticExpr());
                while (this.match(TokenType.COMMA)) {
                    elements.push(this.parseArithmeticExpr());
                }
            }
            this.expect(TokenType.RBRACKET);
            let result = { type: 'array_literal', elements };
            // Check for subscript
            while (this.match(TokenType.LBRACKET)) {
                const index = this.parseArithmeticExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        // EXCLUDED.column for UPSERT
        if (this.match(TokenType.EXCLUDED)) {
            this.expect(TokenType.DOT);
            const col = this.expect(TokenType.IDENTIFIER).value;
            return { type: 'column', value: `EXCLUDED.${col}` };
        }

        // CAST function has special syntax: CAST(value AS type)
        if (this.match(TokenType.CAST)) {
            this.expect(TokenType.LPAREN);
            const value = this.parseArithmeticExpr();
            this.expect(TokenType.AS);
            const targetType = this.parseDataType();
            this.expect(TokenType.RPAREN);
            return { type: 'function', func: 'cast', args: [value, { type: 'literal', value: targetType }] };
        }

        // Tokens that can also be function names (Date/Time: YEAR, MONTH, etc.; String: LEFT, RIGHT)
        const funcTokens = [TokenType.YEAR, TokenType.MONTH, TokenType.DAY, TokenType.HOUR, TokenType.MINUTE, TokenType.SECOND,
                           TokenType.LEFT, TokenType.RIGHT];
        for (const funcType of funcTokens) {
            if (this.peek().type === funcType) {
                const funcName = this.advance().type.toLowerCase();
                this.expect(TokenType.LPAREN);
                const args = this.parseFunctionArgs();
                this.expect(TokenType.RPAREN);
                return { type: 'function', func: funcName, args };
            }
        }

        // Check for function call (identifier followed by LPAREN)
        if (this.peek().type === TokenType.IDENTIFIER) {
            const nextPos = this.pos + 1;
            if (nextPos < this.tokens.length && this.tokens[nextPos].type === TokenType.LPAREN) {
                // It's a function call - use parseFunctionArg's logic
                const funcName = this.advance().value.toUpperCase();
                this.expect(TokenType.LPAREN);
                const args = this.parseFunctionArgs();
                this.expect(TokenType.RPAREN);
                return { type: 'function', func: funcName.toLowerCase(), args };
            }
        }

        // Column reference (possibly table.column) with optional array subscript
        if (this.peek().type === TokenType.IDENTIFIER) {
            const col = this.parseColumnRef();
            let result = { type: 'column', value: col };
            // Check for array subscript: col[index]
            while (this.match(TokenType.LBRACKET)) {
                const index = this.parseArithmeticExpr();
                this.expect(TokenType.RBRACKET);
                result = { type: 'subscript', array: result, index };
            }
            return result;
        }

        throw new Error(`Unexpected token in arithmetic expression: ${this.peek().type}`);
    }

    // Parse window specification: (PARTITION BY ... ORDER BY ...) - OVER already consumed
    parseOverSpec() {
        this.expect(TokenType.LPAREN);

        let partitionBy = null;
        let orderBy = null;

        // Parse PARTITION BY
        if (this.match(TokenType.PARTITION)) {
            this.expect(TokenType.BY);
            partitionBy = [this.parseColumnRef()];
            while (this.match(TokenType.COMMA)) {
                partitionBy.push(this.parseColumnRef());
            }
        }

        // Parse ORDER BY
        if (this.match(TokenType.ORDER)) {
            this.expect(TokenType.BY);
            orderBy = [];
            do {
                const column = this.parseColumnRef();
                const desc = !!this.match(TokenType.DESC);
                if (!desc) this.match(TokenType.ASC);
                orderBy.push({ column, desc });
            } while (this.match(TokenType.COMMA));
        }

        // Parse window frame specification (ROWS/RANGE BETWEEN ... AND ...)
        let frame = null;
        let frameType = null;
        if (this.match(TokenType.ROWS)) {
            frameType = 'rows';
        } else if (this.match(TokenType.RANGE)) {
            frameType = 'range';
        }
        if (frameType) {
            this.expect(TokenType.BETWEEN);
            const start = this.parseFrameBound();
            this.expect(TokenType.AND);
            const end = this.parseFrameBound();
            frame = { type: frameType, start, end };
        }

        this.expect(TokenType.RPAREN);
        return { partitionBy, orderBy, frame };
    }

    // Parse window frame bound (UNBOUNDED PRECEDING, CURRENT ROW, N PRECEDING, etc.)
    parseFrameBound() {
        if (this.match(TokenType.UNBOUNDED)) {
            if (this.match(TokenType.PRECEDING)) return { type: 'unbounded', direction: 'preceding' };
            if (this.match(TokenType.FOLLOWING)) return { type: 'unbounded', direction: 'following' };
            throw new Error('Expected PRECEDING or FOLLOWING after UNBOUNDED');
        }
        if (this.match(TokenType.CURRENT)) {
            this.expect(TokenType.ROW);
            return { type: 'current' };
        }
        // N PRECEDING or N FOLLOWING
        const n = parseInt(this.expect(TokenType.NUMBER).value, 10);
        if (this.match(TokenType.PRECEDING)) return { type: 'offset', value: n, direction: 'preceding' };
        if (this.match(TokenType.FOLLOWING)) return { type: 'offset', value: n, direction: 'following' };
        throw new Error('Expected PRECEDING or FOLLOWING after number');
    }

    // Parse column reference (may be table.column or just column)
    parseColumnRef() {
        const first = this.expect(TokenType.IDENTIFIER).value;
        if (this.match(TokenType.DOT)) {
            const second = this.expect(TokenType.IDENTIFIER).value;
            return { table: first, column: second };
        }
        return first;
    }

    // Parse table reference with optional alias (supports subqueries)
    parseTableRef() {
        // Check for subquery: (SELECT ...)
        if (this.match(TokenType.LPAREN)) {
            if (this.peek().type === TokenType.SELECT) {
                const subquery = this.parseSelect();
                this.expect(TokenType.RPAREN);
                // Alias is required for derived tables
                let alias = null;
                if (this.match(TokenType.AS)) {
                    alias = this.expect(TokenType.IDENTIFIER).value;
                } else if (this.peek().type === TokenType.IDENTIFIER) {
                    alias = this.advance().value;
                }
                return { type: 'subquery', query: subquery, alias: alias || '__derived' };
            }
            // Not a subquery, put back the LPAREN
            this.pos--;
        }

        const name = this.expect(TokenType.IDENTIFIER).value;
        let alias = null;
        if (this.match(TokenType.AS)) {
            alias = this.expect(TokenType.IDENTIFIER).value;
        } else if (this.peek().type === TokenType.IDENTIFIER) {
            // Check if next token looks like an alias (not a keyword)
            const nextType = this.peek().type;
            if (nextType !== TokenType.WHERE && nextType !== TokenType.ORDER &&
                nextType !== TokenType.GROUP && nextType !== TokenType.LIMIT &&
                nextType !== TokenType.JOIN && nextType !== TokenType.LEFT &&
                nextType !== TokenType.RIGHT && nextType !== TokenType.INNER &&
                nextType !== TokenType.FULL && nextType !== TokenType.OUTER &&
                nextType !== TokenType.CROSS && nextType !== TokenType.ON) {
                alias = this.advance().value;
            }
        }
        return { name, alias };
    }

    // Parse JOIN clause
    parseJoin() {
        let joinType = 'INNER';

        if (this.match(TokenType.FULL)) {
            this.match(TokenType.OUTER); // OUTER is optional
            this.expect(TokenType.JOIN);
            joinType = 'FULL';
        } else if (this.match(TokenType.CROSS)) {
            this.expect(TokenType.JOIN);
            joinType = 'CROSS';
        } else if (this.match(TokenType.LEFT)) {
            this.match(TokenType.OUTER); // OUTER is optional
            this.expect(TokenType.JOIN);
            joinType = 'LEFT';
        } else if (this.match(TokenType.RIGHT)) {
            this.match(TokenType.OUTER); // OUTER is optional
            this.expect(TokenType.JOIN);
            joinType = 'RIGHT';
        } else if (this.match(TokenType.INNER)) {
            this.expect(TokenType.JOIN);
            joinType = 'INNER';
        } else {
            this.expect(TokenType.JOIN); // Just JOIN
        }

        const table = this.parseTableRef();

        // CROSS JOIN has no ON clause
        let on = null;
        if (joinType !== 'CROSS') {
            this.expect(TokenType.ON);
            on = this.parseJoinCondition();
        }

        return { type: joinType, table, on };
    }

    // Parse JOIN ON condition with compound expressions (AND/OR)
    parseJoinCondition() {
        return this.parseJoinOrExpr();
    }

    parseJoinOrExpr() {
        let left = this.parseJoinAndExpr();
        while (this.match(TokenType.OR)) {
            left = { op: 'OR', left, right: this.parseJoinAndExpr() };
        }
        return left;
    }

    parseJoinAndExpr() {
        let left = this.parseJoinComparison();
        while (this.match(TokenType.AND)) {
            left = { op: 'AND', left, right: this.parseJoinComparison() };
        }
        return left;
    }

    parseJoinComparison() {
        // Handle parenthesized expressions
        if (this.match(TokenType.LPAREN)) {
            const expr = this.parseJoinOrExpr();
            this.expect(TokenType.RPAREN);
            return expr;
        }

        const left = this.parseColumnRef();
        let op;
        if (this.match(TokenType.EQ)) op = '=';
        else if (this.match(TokenType.NE)) op = '!=';
        else if (this.match(TokenType.LT)) op = '<';
        else if (this.match(TokenType.LE)) op = '<=';
        else if (this.match(TokenType.GT)) op = '>';
        else if (this.match(TokenType.GE)) op = '>=';
        else throw new Error('Expected comparison operator in JOIN condition');

        const right = this.parseColumnRef();
        return { op, left, right };
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
        // Handle NOT EXISTS first
        if (this.match(TokenType.NOT)) {
            if (this.match(TokenType.EXISTS)) {
                this.expect(TokenType.LPAREN);
                const subquery = this.parseSelect();
                this.expect(TokenType.RPAREN);
                return { op: 'NOT EXISTS', subquery };
            }
            // Put back NOT - it might be part of NOT IN, NOT LIKE, etc.
            this.pos--;
        }

        // Handle EXISTS
        if (this.match(TokenType.EXISTS)) {
            this.expect(TokenType.LPAREN);
            const subquery = this.parseSelect();
            this.expect(TokenType.RPAREN);
            return { op: 'EXISTS', subquery };
        }

        // Handle parenthesized expressions
        if (this.match(TokenType.LPAREN)) {
            const expr = this.parseOrExpr();
            this.expect(TokenType.RPAREN);
            return expr;
        }

        // Handle aggregate functions in HAVING clause
        const aggFuncs = [TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX];
        let column;
        let isAggregate = false;
        let aggFunc = null;
        let aggArg = null;

        for (const funcType of aggFuncs) {
            if (this.match(funcType)) {
                isAggregate = true;
                aggFunc = funcType.toLowerCase();
                this.expect(TokenType.LPAREN);
                if (this.match(TokenType.STAR)) {
                    aggArg = '*';
                } else {
                    aggArg = this.parseColumnRef();
                }
                this.expect(TokenType.RPAREN);
                column = { type: 'aggregate', func: aggFunc, arg: aggArg };
                break;
            }
        }

        if (!isAggregate) {
            // Handle literal values on left side (e.g., WHERE 1 = 0, WHERE 'a' = 'b')
            if (this.check(TokenType.NUMBER)) {
                const value = parseFloat(this.advance().value);
                column = { type: 'literal', value };
            } else if (this.check(TokenType.STRING)) {
                const value = this.advance().value;
                column = { type: 'literal', value };
            } else {
                // Parse column (may be dotted like table.column)
                column = this.parseColumnRef();
            }
        }

        // Handle IS NULL / IS NOT NULL
        if (this.match(TokenType.IS)) {
            const isNot = this.match(TokenType.NOT);
            this.expect(TokenType.NULL);
            return { op: isNot ? 'IS NOT NULL' : 'IS NULL', column };
        }

        // Handle NOT BETWEEN, NOT IN, NOT LIKE
        const isNot = this.match(TokenType.NOT);

        // Handle BETWEEN / NOT BETWEEN
        if (this.match(TokenType.BETWEEN)) {
            const low = this.parseValue();
            this.expect(TokenType.AND);
            const high = this.parseValue();
            return { op: isNot ? 'NOT BETWEEN' : 'BETWEEN', column, low, high };
        }

        // Handle IN / NOT IN (with subquery support)
        if (this.match(TokenType.IN)) {
            this.expect(TokenType.LPAREN);
            // Check if it's a subquery
            if (this.check(TokenType.SELECT)) {
                const subquery = this.parseSelect();
                this.expect(TokenType.RPAREN);
                return { op: isNot ? 'NOT IN SUBQUERY' : 'IN SUBQUERY', column, subquery };
            }
            // Otherwise it's a list of values
            const values = [this.parseValue()];
            while (this.match(TokenType.COMMA)) {
                values.push(this.parseValue());
            }
            this.expect(TokenType.RPAREN);
            return { op: isNot ? 'NOT IN' : 'IN', column, values };
        }

        // Handle LIKE / NOT LIKE (move before other operators)
        if (this.match(TokenType.LIKE)) {
            const value = this.parseValue();
            return { op: isNot ? 'NOT LIKE' : 'LIKE', column, value };
        }

        // If we consumed NOT but didn't match BETWEEN/IN/LIKE, error
        if (isNot) {
            throw new Error('Expected BETWEEN, IN, or LIKE after NOT');
        }

        // Handle NEAR (vector similarity search)
        // Syntax: column NEAR value [TOPK n]
        if (this.match(TokenType.NEAR)) {
            const value = this.parseValue();
            let topK = null;
            if (this.match(TokenType.TOPK)) {
                topK = parseInt(this.expect(TokenType.NUMBER).value, 10);
            }
            return { op: 'NEAR', column, value, topK };
        }

        let op;
        if (this.match(TokenType.EQ)) op = '=';
        else if (this.match(TokenType.NE)) op = '!=';
        else if (this.match(TokenType.LT)) op = '<';
        else if (this.match(TokenType.LE)) op = '<=';
        else if (this.match(TokenType.GT)) op = '>';
        else if (this.match(TokenType.GE)) op = '>=';
        else throw new Error(`Expected comparison operator`);

        // Parse value - can be literal value OR column reference (for correlated subqueries)
        let value;
        const nextToken = this.peek();
        if (nextToken.type === TokenType.IDENTIFIER) {
            // Could be a column reference (e.g., d.id in correlated subquery)
            value = this.parseColumnRef();
        } else {
            value = this.parseValue();
        }
        return { op, column, value };
    }
}

// SQL Executor helper - get column value from row

export { SQLParser };
