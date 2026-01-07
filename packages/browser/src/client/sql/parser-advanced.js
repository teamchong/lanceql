/**
 * Parser Advanced Clauses
 * CTEs, window functions, GROUP BY extensions, PIVOT/UNPIVOT
 */

import { TokenType } from './lexer.js';

/**
 * Parse WITH clause (Common Table Expressions)
 * Syntax: WITH [RECURSIVE] name [(columns)] AS (subquery) [, ...]
 */
export function parseWithClause(parser) {
    parser.expect(TokenType.WITH);
    const isRecursive = !!parser.match(TokenType.RECURSIVE);

    const ctes = [];
    do {
        const name = parser.expect(TokenType.IDENTIFIER).value;

        // Optional column list
        let columns = [];
        if (parser.match(TokenType.LPAREN)) {
            columns.push(parser.expect(TokenType.IDENTIFIER).value);
            while (parser.match(TokenType.COMMA)) {
                columns.push(parser.expect(TokenType.IDENTIFIER).value);
            }
            parser.expect(TokenType.RPAREN);
        }

        parser.expect(TokenType.AS);
        parser.expect(TokenType.LPAREN);

        // Parse CTE body - may contain UNION ALL for recursive CTEs
        const body = parseCteBody(parser, isRecursive);

        parser.expect(TokenType.RPAREN);

        ctes.push({
            name,
            columns,
            body,
            recursive: isRecursive
        });
    } while (parser.match(TokenType.COMMA));

    return ctes;
}

/**
 * Parse CTE body which may contain UNION ALL for recursive CTEs
 */
export function parseCteBody(parser, isRecursive) {
    // Parse anchor query - disable set operation parsing, we handle UNION ALL here
    const anchor = parser.parseSelect(true, true);  // isSubquery=true, noSetOps=true

    // Check for UNION ALL (required for recursive CTEs)
    if (isRecursive && parser.match(TokenType.UNION)) {
        parser.expect(TokenType.ALL);
        const recursive = parser.parseSelect(true, true);  // Same for recursive part
        return {
            type: 'RECURSIVE_CTE',
            anchor,
            recursive
        };
    }

    return anchor;
}

/**
 * Parse GROUP BY list with support for ROLLUP, CUBE, GROUPING SETS
 * Returns array of items, each with { type, column/columns/sets }
 */
export function parseGroupByList(parser) {
    const items = [];

    do {
        if (parser.match(TokenType.ROLLUP)) {
            // ROLLUP(col1, col2, ...)
            parser.expect(TokenType.LPAREN);
            const columns = parseColumnList(parser);
            parser.expect(TokenType.RPAREN);
            items.push({ type: 'ROLLUP', columns });
        } else if (parser.match(TokenType.CUBE)) {
            // CUBE(col1, col2, ...)
            parser.expect(TokenType.LPAREN);
            const columns = parseColumnList(parser);
            parser.expect(TokenType.RPAREN);
            items.push({ type: 'CUBE', columns });
        } else if (parser.match(TokenType.GROUPING)) {
            // GROUPING SETS((col1, col2), (col1), ())
            parser.expect(TokenType.SETS);
            parser.expect(TokenType.LPAREN);
            const sets = parseGroupingSets(parser);
            parser.expect(TokenType.RPAREN);
            items.push({ type: 'GROUPING_SETS', sets });
        } else {
            // Simple column
            items.push({ type: 'COLUMN', column: parser.expect(TokenType.IDENTIFIER).value });
        }
    } while (parser.match(TokenType.COMMA));

    // Optional TOPK modifier for GROUP BY
    let topK = null;
    if (parser.match(TokenType.TOPK)) {
        topK = parseInt(parser.expect(TokenType.NUMBER).value, 10);
    }

    return { items, topK };
}

/**
 * Parse the sets inside GROUPING SETS(...)
 * Each set is (col1, col2) or () for grand total
 */
export function parseGroupingSets(parser) {
    const sets = [];

    do {
        parser.expect(TokenType.LPAREN);
        if (parser.check(TokenType.RPAREN)) {
            // Empty set () = grand total
            sets.push([]);
        } else {
            sets.push(parseColumnList(parser));
        }
        parser.expect(TokenType.RPAREN);
    } while (parser.match(TokenType.COMMA));

    return sets;
}

/**
 * Parse column list for GROUP BY
 */
export function parseColumnList(parser) {
    const columns = [parser.expect(TokenType.IDENTIFIER).value];

    while (parser.match(TokenType.COMMA)) {
        columns.push(parser.expect(TokenType.IDENTIFIER).value);
    }

    return columns;
}

/**
 * Parse OVER clause for window functions
 * Syntax: OVER ([PARTITION BY expr, ...] [ORDER BY expr [ASC|DESC], ...] [frame_clause])
 */
export function parseOverClause(parser) {
    parser.expect(TokenType.OVER);
    parser.expect(TokenType.LPAREN);

    const over = { partitionBy: [], orderBy: [], frame: null };

    // PARTITION BY clause
    if (parser.match(TokenType.PARTITION)) {
        parser.expect(TokenType.BY);
        over.partitionBy.push(parser.parseExpr());
        while (parser.match(TokenType.COMMA)) {
            over.partitionBy.push(parser.parseExpr());
        }
    }

    // ORDER BY clause
    if (parser.match(TokenType.ORDER)) {
        parser.expect(TokenType.BY);
        over.orderBy = parser.parseOrderByList();
    }

    // Optional frame clause: ROWS/RANGE BETWEEN ... AND ...
    if (parser.check(TokenType.ROWS) || parser.check(TokenType.RANGE)) {
        over.frame = parseFrameClause(parser);
    }

    parser.expect(TokenType.RPAREN);
    return over;
}

/**
 * Parse frame clause for window functions
 * Syntax: ROWS|RANGE BETWEEN frame_start AND frame_end
 *         or: ROWS|RANGE frame_start
 */
export function parseFrameClause(parser) {
    const frameType = parser.advance().type;  // ROWS or RANGE
    const frame = { type: frameType, start: null, end: null };

    // Check for BETWEEN ... AND ... syntax
    if (parser.match(TokenType.BETWEEN)) {
        frame.start = parseFrameBound(parser);
        parser.expect(TokenType.AND);
        frame.end = parseFrameBound(parser);
    } else {
        // Single bound (implies CURRENT ROW as end for some DBs, or just start)
        frame.start = parseFrameBound(parser);
        frame.end = { type: 'CURRENT ROW' };  // Default end
    }

    return frame;
}

/**
 * Parse a frame bound
 * Options: UNBOUNDED PRECEDING, UNBOUNDED FOLLOWING, CURRENT ROW, N PRECEDING, N FOLLOWING
 */
export function parseFrameBound(parser) {
    if (parser.match(TokenType.UNBOUNDED)) {
        if (parser.match(TokenType.PRECEDING)) {
            return { type: 'UNBOUNDED PRECEDING' };
        } else if (parser.match(TokenType.FOLLOWING)) {
            return { type: 'UNBOUNDED FOLLOWING' };
        }
        throw new Error('Expected PRECEDING or FOLLOWING after UNBOUNDED');
    }

    if (parser.match(TokenType.CURRENT)) {
        parser.expect(TokenType.ROW);
        return { type: 'CURRENT ROW' };
    }

    // N PRECEDING or N FOLLOWING
    if (parser.check(TokenType.NUMBER)) {
        const n = parseInt(parser.advance().value, 10);
        if (parser.match(TokenType.PRECEDING)) {
            return { type: 'PRECEDING', offset: n };
        } else if (parser.match(TokenType.FOLLOWING)) {
            return { type: 'FOLLOWING', offset: n };
        }
        throw new Error('Expected PRECEDING or FOLLOWING after number');
    }

    throw new Error('Invalid frame bound');
}

/**
 * Parse PIVOT clause
 * Syntax: PIVOT (aggregate FOR column IN (value1, value2, ...))
 */
export function parsePivotClause(parser, parsePrimaryFn) {
    parser.expect(TokenType.LPAREN);

    // Parse aggregate function
    const aggFunc = parsePrimaryFn(parser);
    if (aggFunc.type !== 'call') {
        throw new Error('PIVOT requires an aggregate function (e.g., SUM, COUNT, AVG)');
    }

    parser.expect(TokenType.FOR);
    const forColumn = parser.expect(TokenType.IDENTIFIER).value;

    parser.expect(TokenType.IN);
    parser.expect(TokenType.LPAREN);

    // Parse IN values
    const inValues = [];
    inValues.push(parsePrimaryFn(parser).value);
    while (parser.match(TokenType.COMMA)) {
        inValues.push(parsePrimaryFn(parser).value);
    }
    parser.expect(TokenType.RPAREN);
    parser.expect(TokenType.RPAREN);

    return {
        aggregate: aggFunc,
        forColumn,
        inValues
    };
}

/**
 * Parse UNPIVOT clause
 * Syntax: UNPIVOT (valueColumn FOR nameColumn IN (col1, col2, ...))
 */
export function parseUnpivotClause(parser) {
    parser.expect(TokenType.LPAREN);

    const valueColumn = parser.expect(TokenType.IDENTIFIER).value;

    parser.expect(TokenType.FOR);
    const nameColumn = parser.expect(TokenType.IDENTIFIER).value;

    parser.expect(TokenType.IN);
    parser.expect(TokenType.LPAREN);

    // Parse IN columns
    const inColumns = [];
    inColumns.push(parser.expect(TokenType.IDENTIFIER).value);
    while (parser.match(TokenType.COMMA)) {
        inColumns.push(parser.expect(TokenType.IDENTIFIER).value);
    }
    parser.expect(TokenType.RPAREN);
    parser.expect(TokenType.RPAREN);

    return {
        valueColumn,
        nameColumn,
        inColumns
    };
}

/**
 * Parse NEAR clause for vector similarity search
 * Syntax: NEAR [column] <'text'|row_num> [TOPK n]
 */
export function parseNearClause(parser) {
    let column = null;
    let query = null;
    let searchRow = null;
    let topK = 20; // default
    let encoder = 'minilm'; // default

    // First token after NEAR: could be column name, string, or number
    if (parser.check(TokenType.IDENTIFIER)) {
        // Could be column name - peek ahead
        const ident = parser.advance().value;
        if (parser.check(TokenType.STRING) || parser.check(TokenType.NUMBER)) {
            // It was a column name
            column = ident;
        } else {
            // It was a search term without quotes (error)
            throw new Error(`NEAR requires quoted text or row number. Did you mean: NEAR '${ident}'?`);
        }
    }

    // Now expect string (text search) or number (row search)
    if (parser.check(TokenType.STRING)) {
        query = parser.advance().value;
    } else if (parser.check(TokenType.NUMBER)) {
        searchRow = parseInt(parser.advance().value, 10);
    } else {
        throw new Error('NEAR requires a quoted text string or row number');
    }

    // Optional TOPK
    if (parser.match(TokenType.TOPK)) {
        topK = parseInt(parser.expect(TokenType.NUMBER).value, 10);
    }

    return { query, searchRow, column, topK, encoder };
}
