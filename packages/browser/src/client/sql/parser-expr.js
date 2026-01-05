/**
 * Parser Expression Handling
 * Expression parsing methods for SQLParser
 */

import { TokenType } from './lexer.js';

/**
 * Parse expression (entry point)
 */
export function parseExpr(parser) {
    return parseOrExpr(parser);
}

/**
 * Parse OR expression
 */
export function parseOrExpr(parser) {
    let left = parseAndExpr(parser);

    while (parser.match(TokenType.OR)) {
        const right = parseAndExpr(parser);
        left = { type: 'binary', op: 'OR', left, right };
    }

    return left;
}

/**
 * Parse AND expression
 */
export function parseAndExpr(parser) {
    let left = parseNotExpr(parser);

    while (parser.match(TokenType.AND)) {
        const right = parseNotExpr(parser);
        left = { type: 'binary', op: 'AND', left, right };
    }

    return left;
}

/**
 * Parse NOT expression
 */
export function parseNotExpr(parser) {
    if (parser.match(TokenType.NOT)) {
        const operand = parseNotExpr(parser);
        return { type: 'unary', op: 'NOT', operand };
    }
    return parseCmpExpr(parser);
}

/**
 * Parse comparison expression
 */
export function parseCmpExpr(parser) {
    let left = parseAddExpr(parser);

    // IS NULL / IS NOT NULL
    if (parser.match(TokenType.IS)) {
        const negated = !!parser.match(TokenType.NOT);
        parser.expect(TokenType.NULL);
        return {
            type: 'binary',
            op: negated ? '!=' : '==',
            left,
            right: { type: 'literal', value: null }
        };
    }

    // IN - can be a list of values or a subquery
    if (parser.match(TokenType.IN)) {
        parser.expect(TokenType.LPAREN);

        // Check if this is a subquery (starts with SELECT)
        if (parser.check(TokenType.SELECT)) {
            const subquery = parser.parseSelect(true);  // isSubquery=true
            parser.expect(TokenType.RPAREN);
            return { type: 'in', expr: left, values: [{ type: 'subquery', query: subquery }] };
        }

        // Otherwise, parse as list of values
        const values = [];
        values.push(parsePrimary(parser));
        while (parser.match(TokenType.COMMA)) {
            values.push(parsePrimary(parser));
        }
        parser.expect(TokenType.RPAREN);
        return { type: 'in', expr: left, values };
    }

    // BETWEEN
    if (parser.match(TokenType.BETWEEN)) {
        const low = parseAddExpr(parser);
        parser.expect(TokenType.AND);
        const high = parseAddExpr(parser);
        return { type: 'between', expr: left, low, high };
    }

    // LIKE
    if (parser.match(TokenType.LIKE)) {
        const pattern = parsePrimary(parser);
        return { type: 'like', expr: left, pattern };
    }

    // NEAR - vector similarity search in WHERE clause
    if (parser.match(TokenType.NEAR)) {
        const value = parsePrimary(parser);
        return { type: 'near', column: left, value };
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

    const opToken = parser.match(TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE);
    if (opToken) {
        const right = parseAddExpr(parser);
        return { type: 'binary', op: opMap[opToken.type], left, right };
    }

    return left;
}

/**
 * Parse additive expression (+, -)
 */
export function parseAddExpr(parser) {
    let left = parseMulExpr(parser);

    while (true) {
        const opToken = parser.match(TokenType.PLUS, TokenType.MINUS);
        if (!opToken) break;
        const right = parseMulExpr(parser);
        left = { type: 'binary', op: opToken.value, left, right };
    }

    return left;
}

/**
 * Parse multiplicative expression (*, /)
 */
export function parseMulExpr(parser) {
    let left = parseUnaryExpr(parser);

    while (true) {
        const opToken = parser.match(TokenType.STAR, TokenType.SLASH);
        if (!opToken) break;
        const right = parseUnaryExpr(parser);
        left = { type: 'binary', op: opToken.value, left, right };
    }

    return left;
}

/**
 * Parse unary expression (-)
 */
export function parseUnaryExpr(parser) {
    if (parser.match(TokenType.MINUS)) {
        const operand = parseUnaryExpr(parser);
        return { type: 'unary', op: '-', operand };
    }
    return parsePrimary(parser);
}

/**
 * Parse primary expression
 */
export function parsePrimary(parser) {
    // NULL
    if (parser.match(TokenType.NULL)) {
        return { type: 'literal', value: null };
    }

    // TRUE/FALSE
    if (parser.match(TokenType.TRUE)) {
        return { type: 'literal', value: true };
    }
    if (parser.match(TokenType.FALSE)) {
        return { type: 'literal', value: false };
    }

    // ARRAY[...] literal
    if (parser.match(TokenType.ARRAY)) {
        let result = parseArrayLiteral(parser);
        // Check for subscript: ARRAY[1,2,3][2]
        while (parser.check(TokenType.LBRACKET)) {
            parser.advance();
            const index = parseExpr(parser);
            parser.expect(TokenType.RBRACKET);
            result = { type: 'subscript', array: result, index };
        }
        return result;
    }

    // Bare bracket array [...]
    if (parser.check(TokenType.LBRACKET)) {
        let result = parseArrayLiteral(parser);
        // Check for subscript: [1,2,3][2]
        while (parser.check(TokenType.LBRACKET)) {
            parser.advance();
            const index = parseExpr(parser);
            parser.expect(TokenType.RBRACKET);
            result = { type: 'subscript', array: result, index };
        }
        return result;
    }

    // Number
    if (parser.check(TokenType.NUMBER)) {
        const value = parser.advance().value;
        return { type: 'literal', value: parseFloat(value) };
    }

    // String
    if (parser.check(TokenType.STRING)) {
        const value = parser.advance().value;
        return { type: 'literal', value };
    }

    // Window function keywords (ROW_NUMBER, RANK, etc.)
    const windowFuncTokens = [
        TokenType.ROW_NUMBER, TokenType.RANK, TokenType.DENSE_RANK, TokenType.NTILE,
        TokenType.LAG, TokenType.LEAD, TokenType.FIRST_VALUE, TokenType.LAST_VALUE, TokenType.NTH_VALUE,
        TokenType.PERCENT_RANK, TokenType.CUME_DIST
    ];
    if (windowFuncTokens.some(t => parser.check(t))) {
        const name = parser.advance().type;  // Use token type as function name
        parser.expect(TokenType.LPAREN);
        const args = [];
        if (!parser.check(TokenType.RPAREN)) {
            args.push(parseExpr(parser));
            while (parser.match(TokenType.COMMA)) {
                args.push(parseExpr(parser));
            }
        }
        parser.expect(TokenType.RPAREN);

        // OVER clause is required for window functions
        const over = parser.parseOverClause();
        return { type: 'call', name, args, distinct: false, over };
    }

    // Function call or column reference
    if (parser.check(TokenType.IDENTIFIER) || parser.check(TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX, TokenType.GROUPING)) {
        const name = parser.advance().value;

        // Function call
        if (parser.match(TokenType.LPAREN)) {
            let distinct = !!parser.match(TokenType.DISTINCT);
            const args = [];

            if (!parser.check(TokenType.RPAREN)) {
                // Handle COUNT(*)
                if (parser.check(TokenType.STAR)) {
                    parser.advance();
                    args.push({ type: 'star' });
                } else {
                    args.push(parseExpr(parser));
                    while (parser.match(TokenType.COMMA)) {
                        args.push(parseExpr(parser));
                    }
                }
            }

            parser.expect(TokenType.RPAREN);

            // Check for OVER clause (aggregate as window function)
            let over = null;
            if (parser.check(TokenType.OVER)) {
                over = parser.parseOverClause();
            }

            return { type: 'call', name: name.toUpperCase(), args, distinct, over };
        }

        // Column reference - check for table.column syntax
        if (parser.match(TokenType.DOT)) {
            // table.column (column can be a keyword like "text")
            const table = name;
            const token = parser.advance();
            // Allow keywords as column names (e.g., c.text where TEXT is a keyword)
            const column = token.value || token.type.toLowerCase();
            return { type: 'column', table, column };
        }

        // Simple column reference
        let result = { type: 'column', column: name };

        // Check for array subscript: column[index]
        if (parser.check(TokenType.LBRACKET)) {
            parser.advance();  // consume [
            const index = parseExpr(parser);
            parser.expect(TokenType.RBRACKET);
            result = { type: 'subscript', array: result, index };
        }

        return result;
    }

    // Parenthesized expression or subquery
    if (parser.match(TokenType.LPAREN)) {
        // Check if this is a subquery (starts with SELECT)
        if (parser.check(TokenType.SELECT)) {
            const subquery = parser.parseSelect(true);  // isSubquery=true
            parser.expect(TokenType.RPAREN);
            return { type: 'subquery', query: subquery };
        }
        const expr = parseExpr(parser);
        parser.expect(TokenType.RPAREN);
        return expr;
    }

    // Star (for SELECT *)
    if (parser.match(TokenType.STAR)) {
        return { type: 'star' };
    }

    throw new Error(`Unexpected token: ${parser.current().type} (${parser.current().value})`);
}

/**
 * Parse a single value (number, string, null, true, false)
 */
export function parseValue(parser) {
    if (parser.match(TokenType.NULL)) {
        return { type: 'null', value: null };
    }
    if (parser.match(TokenType.TRUE)) {
        return { type: 'boolean', value: true };
    }
    if (parser.match(TokenType.FALSE)) {
        return { type: 'boolean', value: false };
    }
    if (parser.check(TokenType.NUMBER)) {
        const token = parser.advance();
        const value = token.value.includes('.') ? parseFloat(token.value) : parseInt(token.value, 10);
        return { type: 'number', value };
    }
    if (parser.check(TokenType.STRING)) {
        const token = parser.advance();
        return { type: 'string', value: token.value };
    }
    if (parser.check(TokenType.MINUS)) {
        parser.advance();
        const token = parser.expect(TokenType.NUMBER);
        const value = token.value.includes('.') ? -parseFloat(token.value) : -parseInt(token.value, 10);
        return { type: 'number', value };
    }
    // Vector literal: [1.0, 2.0, 3.0]
    if (parser.check(TokenType.LBRACKET)) {
        return parseArrayLiteral(parser);
    }

    throw new Error(`Expected value, got ${parser.current().type}`);
}

/**
 * Parse array literal: [1, 2, 3] or ARRAY[1, 2, 3]
 */
export function parseArrayLiteral(parser) {
    parser.expect(TokenType.LBRACKET);
    const elements = [];

    if (!parser.check(TokenType.RBRACKET)) {
        elements.push(parseExpr(parser));
        while (parser.match(TokenType.COMMA)) {
            elements.push(parseExpr(parser));
        }
    }

    parser.expect(TokenType.RBRACKET);
    return { type: 'array', elements };
}
