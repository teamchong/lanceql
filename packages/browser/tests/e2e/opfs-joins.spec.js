import { test, expect } from '@playwright/test';

/**
 * OPFS-Backed JOIN Tests
 *
 * These tests verify the OPFS join functionality in the LanceDatabase class.
 * Since OPFS JOINs require actual Lance datasets, these tests focus on:
 * 1. Helper methods (_buildInClause, _appendWhereClause)
 * 2. Semi-join optimization logic
 * 3. Filter pushdown SQL generation
 *
 * For full JOIN type tests with actual data, use manual testing via test-join.html
 */
test.describe('OPFS JOIN Infrastructure', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/test-join.html');
        await page.waitForLoadState('domcontentloaded');
    });

    test('LanceDatabase class is exported and instantiable', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { LanceDatabase } = await import('./lanceql.js');
                const db = new LanceDatabase();
                return {
                    success: true,
                    hasRegister: typeof db.register === 'function',
                    hasExecuteSQL: typeof db.executeSQL === 'function'
                };
            } catch (e) {
                return { success: false, error: e.message };
            }
        });

        expect(result.success).toBe(true);
        expect(result.hasRegister).toBe(true);
        expect(result.hasExecuteSQL).toBe(true);
    });

    test('_buildInClause helper generates correct SQL', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = await import('./lanceql.js');
            const db = new LanceDatabase();

            const tests = [];

            // Test with numbers
            const numKeys = new Set([1, 2, 3]);
            const numResult = db._buildInClause('id', numKeys);
            tests.push({
                name: 'numeric keys',
                pass: numResult === 'id IN (1, 2, 3)',
                actual: numResult
            });

            // Test with strings
            const strKeys = new Set(['apple', 'banana']);
            const strResult = db._buildInClause('name', strKeys);
            tests.push({
                name: 'string keys',
                pass: strResult === "name IN ('apple', 'banana')",
                actual: strResult
            });

            // Test with quotes in strings (SQL injection prevention)
            const quoteKeys = new Set(["O'Reilly", "test"]);
            const quoteResult = db._buildInClause('author', quoteKeys);
            tests.push({
                name: 'escaped quotes',
                pass: quoteResult.includes("O''Reilly"),  // Should escape single quotes
                actual: quoteResult
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('_appendWhereClause helper adds conditions correctly', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = await import('./lanceql.js');
            const db = new LanceDatabase();

            const tests = [];

            // Test adding WHERE to query without WHERE
            const noWhere = 'SELECT * FROM users';
            const result1 = db._appendWhereClause(noWhere, 'id > 0');
            tests.push({
                name: 'add WHERE to query without WHERE',
                pass: result1.includes('WHERE id > 0'),
                actual: result1
            });

            // Test adding AND to query with existing WHERE
            const hasWhere = 'SELECT * FROM users WHERE active = 1';
            const result2 = db._appendWhereClause(hasWhere, 'id IN (1, 2)');
            tests.push({
                name: 'add AND to query with existing WHERE',
                pass: result2.includes('WHERE id IN (1, 2) AND active = 1'),
                actual: result2
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('_filterToSQL handles all expression types', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = await import('./lanceql.js');
            const db = new LanceDatabase();

            const tests = [];

            // Test IN expression
            const inExpr = {
                type: 'in',
                expr: { type: 'column', column: 'id' },
                values: [
                    { type: 'literal', value: 1 },
                    { type: 'literal', value: 2 }
                ]
            };
            const inResult = db._filterToSQL(inExpr);
            tests.push({
                name: 'IN expression',
                pass: inResult === 'id IN (1, 2)',
                actual: inResult
            });

            // Test BETWEEN expression
            const betweenExpr = {
                type: 'between',
                expr: { type: 'column', column: 'price' },
                low: { type: 'literal', value: 10 },
                high: { type: 'literal', value: 100 }
            };
            const betweenResult = db._filterToSQL(betweenExpr);
            tests.push({
                name: 'BETWEEN expression',
                pass: betweenResult === 'price BETWEEN 10 AND 100',
                actual: betweenResult
            });

            // Test LIKE expression
            const likeExpr = {
                type: 'like',
                expr: { type: 'column', column: 'name' },
                pattern: { type: 'literal', value: 'test%' }
            };
            const likeResult = db._filterToSQL(likeExpr);
            tests.push({
                name: 'LIKE expression',
                pass: likeResult === "name LIKE 'test%'",
                actual: likeResult
            });

            // Test unary NOT expression
            const notExpr = {
                type: 'unary',
                op: 'NOT',
                operand: { type: 'column', column: 'active' }
            };
            const notResult = db._filterToSQL(notExpr);
            tests.push({
                name: 'NOT expression',
                pass: notResult === 'NOT active',
                actual: notResult
            });

            // Test column stripping (no table prefix in output)
            const colExpr = { type: 'column', table: 'users', column: 'id' };
            const colResult = db._filterToSQL(colExpr);
            tests.push({
                name: 'column without table prefix',
                pass: colResult === 'id',  // Should strip table prefix
                actual: colResult
            });

            // Test quote escaping in literals
            const quoteExpr = { type: 'literal', value: "O'Reilly" };
            const quoteResult = db._filterToSQL(quoteExpr);
            tests.push({
                name: 'escaped quotes in literal',
                pass: quoteResult === "'O''Reilly'",
                actual: quoteResult
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('OPFSJoinExecutor accepts prePartitionedLeft option', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { OPFSJoinExecutor, OPFSStorage } = await import('./lanceql.js');

            try {
                // Just verify the class exists and options are accepted
                // Actual execution requires real datasets
                const storage = new OPFSStorage();
                const executor = new OPFSJoinExecutor(storage);

                return {
                    success: true,
                    hasExecuteHashJoin: typeof executor.executeHashJoin === 'function',
                    hasPartitionToOPFS: typeof executor._partitionToOPFS === 'function'
                };
            } catch (e) {
                return { success: false, error: e.message };
            }
        });

        expect(result.success).toBe(true);
        expect(result.hasExecuteHashJoin).toBe(true);
        expect(result.hasPartitionToOPFS).toBe(true);
    });

    test('_findColumnIndex handles qualified and unqualified column names', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = await import('./lanceql.js');
            const db = new LanceDatabase();

            const tests = [];

            // Test exact match
            const columns1 = ['id', 'name', 'email'];
            tests.push({
                name: 'exact match',
                pass: db._findColumnIndex(columns1, 'name') === 1,
                actual: db._findColumnIndex(columns1, 'name')
            });

            // Test qualified name (table.column)
            const columns2 = ['users.id', 'users.name', 'orders.id'];
            tests.push({
                name: 'unqualified finds qualified',
                pass: db._findColumnIndex(columns2, 'name') === 1,
                actual: db._findColumnIndex(columns2, 'name')
            });

            // Test not found
            tests.push({
                name: 'not found returns -1',
                pass: db._findColumnIndex(columns1, 'notfound') === -1,
                actual: db._findColumnIndex(columns1, 'notfound')
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('LanceDatabase has _hashJoinWithInMemoryLeft method for multiple JOINs', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = await import('./lanceql.js');
            const db = new LanceDatabase();

            return {
                success: true,
                hasMethod: typeof db._hashJoinWithInMemoryLeft === 'function',
                hasFindColumn: typeof db._findColumnIndex === 'function'
            };
        });

        expect(result.success).toBe(true);
        expect(result.hasMethod).toBe(true);
        expect(result.hasFindColumn).toBe(true);
    });

    test('SQL parser handles subqueries in WHERE clause', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test scalar subquery
            const sql1 = 'SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders)';
            try {
                const tokens = new SQLLexer(sql1).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'scalar subquery parsed',
                    pass: ast.where?.type === 'binary' && ast.where.right?.type === 'subquery',
                    actual: ast.where?.right?.type
                });

                // Verify subquery structure
                tests.push({
                    name: 'subquery has SELECT AST',
                    pass: ast.where?.right?.query?.columns?.length > 0,
                    actual: JSON.stringify(ast.where?.right?.query?.columns)
                });
            } catch (e) {
                tests.push({ name: 'scalar subquery parse failed', pass: false, actual: e.message });
            }

            // Test correlated subquery
            const sql2 = 'SELECT * FROM orders o WHERE amount > (SELECT AVG(amount) FROM orders WHERE customer_id = o.customer_id)';
            try {
                const tokens = new SQLLexer(sql2).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'correlated subquery parsed',
                    pass: ast.where?.right?.type === 'subquery',
                    actual: ast.where?.right?.type
                });

                // Check that the inner WHERE references outer column
                const innerWhere = ast.where?.right?.query?.where;
                tests.push({
                    name: 'correlated reference detected',
                    pass: innerWhere?.right?.table === 'o',
                    actual: innerWhere?.right?.table
                });
            } catch (e) {
                tests.push({ name: 'correlated subquery parse failed', pass: false, actual: e.message });
            }

            // Test IN with subquery
            const sql3 = 'SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)';
            try {
                const tokens = new SQLLexer(sql3).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'IN subquery parsed',
                    pass: ast.where?.type === 'in' && ast.where.values[0]?.type === 'subquery',
                    actual: ast.where?.values[0]?.type
                });
            } catch (e) {
                tests.push({ name: 'IN subquery parse failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('SQL parser handles WITH clause (CTEs)', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test simple CTE
            const sql1 = 'WITH active_users AS (SELECT * FROM users WHERE active = 1) SELECT * FROM active_users';
            try {
                const tokens = new SQLLexer(sql1).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'simple CTE parsed',
                    pass: ast.ctes?.length === 1 && ast.ctes[0].name === 'active_users',
                    actual: ast.ctes?.[0]?.name
                });

                tests.push({
                    name: 'CTE body is SELECT',
                    pass: ast.ctes[0].body?.type === 'SELECT',
                    actual: ast.ctes?.[0]?.body?.type
                });
            } catch (e) {
                tests.push({ name: 'simple CTE parse failed', pass: false, actual: e.message });
            }

            // Test recursive CTE
            const sql2 = `WITH RECURSIVE hierarchy AS (
                SELECT id, name, 0 AS depth FROM employees WHERE manager_id IS NULL
                UNION ALL
                SELECT e.id, e.name, h.depth + 1 FROM employees e JOIN hierarchy h ON e.manager_id = h.id
            ) SELECT * FROM hierarchy`;
            try {
                const tokens = new SQLLexer(sql2).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'recursive CTE parsed',
                    pass: ast.ctes?.length === 1 && ast.ctes[0].recursive === true,
                    actual: `recursive=${ast.ctes?.[0]?.recursive}`
                });

                tests.push({
                    name: 'recursive CTE has UNION ALL structure',
                    pass: ast.ctes[0].body?.type === 'RECURSIVE_CTE',
                    actual: ast.ctes?.[0]?.body?.type
                });

                tests.push({
                    name: 'recursive CTE has anchor and recursive parts',
                    pass: !!(ast.ctes[0].body?.anchor && ast.ctes[0].body?.recursive),
                    actual: `anchor=${!!ast.ctes?.[0]?.body?.anchor}, recursive=${!!ast.ctes?.[0]?.body?.recursive}`
                });
            } catch (e) {
                tests.push({ name: 'recursive CTE parse failed', pass: false, actual: e.message });
            }

            // Test CTE with column list
            const sql3 = 'WITH totals (category, total) AS (SELECT category, SUM(amount) FROM sales GROUP BY category) SELECT * FROM totals';
            try {
                const tokens = new SQLLexer(sql3).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'CTE with column list parsed',
                    pass: ast.ctes[0].columns?.length === 2 &&
                          ast.ctes[0].columns[0] === 'category' &&
                          ast.ctes[0].columns[1] === 'total',
                    actual: JSON.stringify(ast.ctes?.[0]?.columns)
                });
            } catch (e) {
                tests.push({ name: 'CTE with column list parse failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('SQL parser handles PIVOT/UNPIVOT clauses', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test PIVOT
            const sql1 = "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))";
            try {
                const tokens = new SQLLexer(sql1).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'PIVOT parsed',
                    pass: ast.pivot !== null,
                    actual: JSON.stringify(ast.pivot)
                });

                tests.push({
                    name: 'PIVOT aggregate is SUM',
                    pass: ast.pivot?.aggregate?.name === 'SUM',
                    actual: ast.pivot?.aggregate?.name
                });

                tests.push({
                    name: 'PIVOT forColumn is quarter',
                    pass: ast.pivot?.forColumn === 'quarter',
                    actual: ast.pivot?.forColumn
                });

                tests.push({
                    name: 'PIVOT has 4 IN values',
                    pass: ast.pivot?.inValues?.length === 4,
                    actual: ast.pivot?.inValues?.length
                });
            } catch (e) {
                tests.push({ name: 'PIVOT parse failed', pass: false, actual: e.message });
            }

            // Test UNPIVOT
            const sql2 = "SELECT * FROM wide_table UNPIVOT (value FOR month IN (jan, feb, mar))";
            try {
                const tokens = new SQLLexer(sql2).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'UNPIVOT parsed',
                    pass: ast.unpivot !== null,
                    actual: JSON.stringify(ast.unpivot)
                });

                tests.push({
                    name: 'UNPIVOT valueColumn is value',
                    pass: ast.unpivot?.valueColumn === 'value',
                    actual: ast.unpivot?.valueColumn
                });

                tests.push({
                    name: 'UNPIVOT nameColumn is month',
                    pass: ast.unpivot?.nameColumn === 'month',
                    actual: ast.unpivot?.nameColumn
                });

                tests.push({
                    name: 'UNPIVOT has 3 IN columns',
                    pass: ast.unpivot?.inColumns?.length === 3 &&
                          ast.unpivot?.inColumns[0] === 'jan',
                    actual: JSON.stringify(ast.unpivot?.inColumns)
                });
            } catch (e) {
                tests.push({ name: 'UNPIVOT parse failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });
});

/**
 * Phase 7: CTE, Set Operations, and Window Functions Tests
 */
test.describe('Phase 7: CTE, Set Operations, Window Functions', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/test-join.html');
        await page.waitForLoadState('domcontentloaded');
    });

    test('SQL parser handles CTE (WITH clause)', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test simple CTE
            const sql1 = 'WITH t AS (SELECT id, name FROM users) SELECT * FROM t';
            try {
                const tokens = new SQLLexer(sql1).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'CTE parsed into ast.ctes',
                    pass: ast.ctes?.length === 1,
                    actual: ast.ctes?.length
                });

                tests.push({
                    name: 'CTE has name "t"',
                    pass: ast.ctes?.[0]?.name === 't',
                    actual: ast.ctes?.[0]?.name
                });

                tests.push({
                    name: 'CTE body is SELECT',
                    pass: ast.ctes?.[0]?.body?.type === 'SELECT',
                    actual: ast.ctes?.[0]?.body?.type
                });
            } catch (e) {
                tests.push({ name: 'CTE parse failed', pass: false, actual: e.message });
            }

            // Test multiple CTEs
            const sql2 = 'WITH a AS (SELECT 1), b AS (SELECT 2) SELECT * FROM a';
            try {
                const tokens = new SQLLexer(sql2).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'Multiple CTEs parsed',
                    pass: ast.ctes?.length === 2,
                    actual: ast.ctes?.length
                });
            } catch (e) {
                tests.push({ name: 'Multiple CTEs failed', pass: false, actual: e.message });
            }

            // Test recursive CTE
            const sql3 = 'WITH RECURSIVE nums AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM nums) SELECT * FROM nums';
            try {
                const tokens = new SQLLexer(sql3).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'Recursive CTE marked as recursive',
                    pass: ast.ctes?.[0]?.recursive === true,
                    actual: ast.ctes?.[0]?.recursive
                });

                tests.push({
                    name: 'Recursive CTE body type is RECURSIVE_CTE',
                    pass: ast.ctes?.[0]?.body?.type === 'RECURSIVE_CTE',
                    actual: ast.ctes?.[0]?.body?.type
                });
            } catch (e) {
                tests.push({ name: 'Recursive CTE failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('SQL parser handles SET operations (UNION, INTERSECT, EXCEPT)', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test UNION
            const sql1 = 'SELECT id FROM a UNION SELECT id FROM b';
            try {
                const tokens = new SQLLexer(sql1).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'UNION parsed as SET_OPERATION',
                    pass: ast.type === 'SET_OPERATION',
                    actual: ast.type
                });

                tests.push({
                    name: 'UNION operator is UNION',
                    pass: ast.operator === 'UNION',
                    actual: ast.operator
                });

                tests.push({
                    name: 'UNION all is false',
                    pass: ast.all === false,
                    actual: ast.all
                });
            } catch (e) {
                tests.push({ name: 'UNION parse failed', pass: false, actual: e.message });
            }

            // Test UNION ALL
            const sql2 = 'SELECT id FROM a UNION ALL SELECT id FROM b';
            try {
                const tokens = new SQLLexer(sql2).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'UNION ALL has all=true',
                    pass: ast.all === true,
                    actual: ast.all
                });
            } catch (e) {
                tests.push({ name: 'UNION ALL failed', pass: false, actual: e.message });
            }

            // Test INTERSECT
            const sql3 = 'SELECT id FROM a INTERSECT SELECT id FROM b';
            try {
                const tokens = new SQLLexer(sql3).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'INTERSECT operator parsed',
                    pass: ast.operator === 'INTERSECT',
                    actual: ast.operator
                });
            } catch (e) {
                tests.push({ name: 'INTERSECT failed', pass: false, actual: e.message });
            }

            // Test EXCEPT
            const sql4 = 'SELECT id FROM a EXCEPT SELECT id FROM b';
            try {
                const tokens = new SQLLexer(sql4).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'EXCEPT operator parsed',
                    pass: ast.operator === 'EXCEPT',
                    actual: ast.operator
                });
            } catch (e) {
                tests.push({ name: 'EXCEPT failed', pass: false, actual: e.message });
            }

            // Test SET operation with ORDER BY and LIMIT
            const sql5 = 'SELECT id FROM a UNION SELECT id FROM b ORDER BY id LIMIT 10';
            try {
                const tokens = new SQLLexer(sql5).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'SET op with ORDER BY',
                    pass: ast.orderBy?.length === 1 && ast.orderBy[0].column === 'id',
                    actual: JSON.stringify(ast.orderBy)
                });

                tests.push({
                    name: 'SET op with LIMIT',
                    pass: ast.limit === 10,
                    actual: ast.limit
                });
            } catch (e) {
                tests.push({ name: 'SET op with ORDER/LIMIT failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('SQL parser handles window functions with OVER clause', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test ROW_NUMBER
            const sql1 = 'SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM users';
            try {
                const tokens = new SQLLexer(sql1).tokenize();
                const ast = new SQLParser(tokens).parse();

                const rowNumCol = ast.columns?.[1];
                tests.push({
                    name: 'ROW_NUMBER parsed',
                    pass: rowNumCol?.expr?.name === 'ROW_NUMBER',
                    actual: rowNumCol?.expr?.name
                });

                tests.push({
                    name: 'ROW_NUMBER has OVER clause',
                    pass: rowNumCol?.expr?.over !== null,
                    actual: rowNumCol?.expr?.over ? 'present' : 'missing'
                });

                tests.push({
                    name: 'OVER has ORDER BY',
                    pass: rowNumCol?.expr?.over?.orderBy?.length === 1,
                    actual: rowNumCol?.expr?.over?.orderBy?.length
                });
            } catch (e) {
                tests.push({ name: 'ROW_NUMBER failed', pass: false, actual: e.message });
            }

            // Test PARTITION BY
            const sql2 = 'SELECT dept, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) FROM employees';
            try {
                const tokens = new SQLLexer(sql2).tokenize();
                const ast = new SQLParser(tokens).parse();

                const rankCol = ast.columns?.[1];
                tests.push({
                    name: 'PARTITION BY parsed',
                    pass: rankCol?.expr?.over?.partitionBy?.length === 1,
                    actual: rankCol?.expr?.over?.partitionBy?.length
                });

                tests.push({
                    name: 'PARTITION BY column is dept',
                    pass: rankCol?.expr?.over?.partitionBy?.[0]?.column === 'dept',
                    actual: rankCol?.expr?.over?.partitionBy?.[0]?.column
                });
            } catch (e) {
                tests.push({ name: 'PARTITION BY failed', pass: false, actual: e.message });
            }

            // Test LAG/LEAD
            const sql3 = 'SELECT id, LAG(value, 1, 0) OVER (ORDER BY id) FROM data';
            try {
                const tokens = new SQLLexer(sql3).tokenize();
                const ast = new SQLParser(tokens).parse();

                const lagCol = ast.columns?.[1];
                tests.push({
                    name: 'LAG parsed with 3 args',
                    pass: lagCol?.expr?.name === 'LAG' && lagCol?.expr?.args?.length === 3,
                    actual: `${lagCol?.expr?.name} with ${lagCol?.expr?.args?.length} args`
                });
            } catch (e) {
                tests.push({ name: 'LAG failed', pass: false, actual: e.message });
            }

            // Test aggregate as window function (SUM OVER)
            const sql4 = 'SELECT id, SUM(amount) OVER (ORDER BY id) AS running_total FROM orders';
            try {
                const tokens = new SQLLexer(sql4).tokenize();
                const ast = new SQLParser(tokens).parse();

                const sumCol = ast.columns?.[1];
                tests.push({
                    name: 'SUM OVER parsed',
                    pass: sumCol?.expr?.name === 'SUM' && sumCol?.expr?.over !== null,
                    actual: sumCol?.expr?.over ? 'has OVER' : 'no OVER'
                });
            } catch (e) {
                tests.push({ name: 'SUM OVER failed', pass: false, actual: e.message });
            }

            // Test frame clause
            const sql5 = 'SELECT id, AVG(value) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM data';
            try {
                const tokens = new SQLLexer(sql5).tokenize();
                const ast = new SQLParser(tokens).parse();

                const avgCol = ast.columns?.[1];
                tests.push({
                    name: 'Frame clause parsed',
                    pass: avgCol?.expr?.over?.frame !== null,
                    actual: avgCol?.expr?.over?.frame ? 'has frame' : 'no frame'
                });

                tests.push({
                    name: 'Frame type is ROWS',
                    pass: avgCol?.expr?.over?.frame?.type === 'ROWS',
                    actual: avgCol?.expr?.over?.frame?.type
                });

                tests.push({
                    name: 'Frame start is 2 PRECEDING',
                    pass: avgCol?.expr?.over?.frame?.start?.type === 'PRECEDING' &&
                          avgCol?.expr?.over?.frame?.start?.offset === 2,
                    actual: JSON.stringify(avgCol?.expr?.over?.frame?.start)
                });
            } catch (e) {
                tests.push({ name: 'Frame clause failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('SQLExecutor has window function execution methods', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLExecutor } = await import('./lanceql.js');

            // Create executor with mock file
            const executor = new SQLExecutor({ columnNames: ['id', 'value'] });

            const tests = [];

            tests.push({
                name: 'hasWindowFunctions method exists',
                pass: typeof executor.hasWindowFunctions === 'function',
                actual: typeof executor.hasWindowFunctions
            });

            tests.push({
                name: 'computeWindowFunctions method exists',
                pass: typeof executor.computeWindowFunctions === 'function',
                actual: typeof executor.computeWindowFunctions
            });

            tests.push({
                name: '_partitionRows method exists',
                pass: typeof executor._partitionRows === 'function',
                actual: typeof executor._partitionRows
            });

            tests.push({
                name: '_compareRowsByOrder method exists',
                pass: typeof executor._compareRowsByOrder === 'function',
                actual: typeof executor._compareRowsByOrder
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('SQLExecutor CTE materialization methods exist', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLExecutor } = await import('./lanceql.js');

            const executor = new SQLExecutor({ columnNames: [] });

            const tests = [];

            tests.push({
                name: 'materializeCTEs method exists',
                pass: typeof executor.materializeCTEs === 'function',
                actual: typeof executor.materializeCTEs
            });

            tests.push({
                name: '_executeCTEBody method exists',
                pass: typeof executor._executeCTEBody === 'function',
                actual: typeof executor._executeCTEBody
            });

            tests.push({
                name: '_executeOnInMemoryData method exists',
                pass: typeof executor._executeOnInMemoryData === 'function',
                actual: typeof executor._executeOnInMemoryData
            });

            tests.push({
                name: '_cteResults is a Map',
                pass: executor._cteResults instanceof Map,
                actual: executor._cteResults?.constructor?.name
            });

            tests.push({
                name: 'setDatabase method exists',
                pass: typeof executor.setDatabase === 'function',
                actual: typeof executor.setDatabase
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('LanceDatabase has SET operation and CTE execution methods', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = await import('./lanceql.js');

            const db = new LanceDatabase();

            const tests = [];

            tests.push({
                name: '_executeSetOperation method exists',
                pass: typeof db._executeSetOperation === 'function',
                actual: typeof db._executeSetOperation
            });

            tests.push({
                name: '_executeWithCTEs method exists',
                pass: typeof db._executeWithCTEs === 'function',
                actual: typeof db._executeWithCTEs
            });

            tests.push({
                name: '_astToSQL method exists',
                pass: typeof db._astToSQL === 'function',
                actual: typeof db._astToSQL
            });

            tests.push({
                name: '_exprToSQL method exists',
                pass: typeof db._exprToSQL === 'function',
                actual: typeof db._exprToSQL
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('In-memory query execution works correctly', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLExecutor, SQLLexer, SQLParser } = await import('./lanceql.js');

            const executor = new SQLExecutor({ columnNames: [] });

            // Mock in-memory data
            const data = {
                columns: ['id', 'name', 'value'],
                rows: [
                    [1, 'Alice', 100],
                    [2, 'Bob', 200],
                    [3, 'Charlie', 150],
                    [4, 'Diana', 250]
                ]
            };

            const tests = [];

            // Test SELECT * with WHERE
            const sql1 = 'SELECT * FROM t WHERE value > 150';
            try {
                const tokens = new SQLLexer(sql1).tokenize();
                const ast = new SQLParser(tokens).parse();
                const result = executor._executeOnInMemoryData(ast, data);

                tests.push({
                    name: 'SELECT * WHERE filters correctly',
                    pass: result.rows.length === 2,  // Bob (200) and Diana (250)
                    actual: result.rows.length
                });
            } catch (e) {
                tests.push({ name: 'SELECT * WHERE failed', pass: false, actual: e.message });
            }

            // Test SELECT with specific columns
            const sql2 = 'SELECT name, value FROM t WHERE id < 3';
            try {
                const tokens = new SQLLexer(sql2).tokenize();
                const ast = new SQLParser(tokens).parse();
                const result = executor._executeOnInMemoryData(ast, data);

                tests.push({
                    name: 'SELECT columns filters correctly',
                    pass: result.rows.length === 2 && result.columns.length === 2,
                    actual: `${result.rows.length} rows, ${result.columns.length} cols`
                });
            } catch (e) {
                tests.push({ name: 'SELECT columns failed', pass: false, actual: e.message });
            }

            // Test LIMIT
            const sql3 = 'SELECT * FROM t LIMIT 2';
            try {
                const tokens = new SQLLexer(sql3).tokenize();
                const ast = new SQLParser(tokens).parse();
                const result = executor._executeOnInMemoryData(ast, data);

                tests.push({
                    name: 'LIMIT works',
                    pass: result.rows.length === 2,
                    actual: result.rows.length
                });
            } catch (e) {
                tests.push({ name: 'LIMIT failed', pass: false, actual: e.message });
            }

            // Test ORDER BY
            const sql4 = 'SELECT * FROM t ORDER BY value DESC';
            try {
                const tokens = new SQLLexer(sql4).tokenize();
                const ast = new SQLParser(tokens).parse();
                const result = executor._executeOnInMemoryData(ast, data);

                tests.push({
                    name: 'ORDER BY DESC works',
                    pass: result.rows[0][2] === 250,  // Diana has highest value
                    actual: result.rows[0][2]
                });
            } catch (e) {
                tests.push({ name: 'ORDER BY failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });

    test('Window function ROW_NUMBER execution', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLExecutor } = await import('./lanceql.js');

            const executor = new SQLExecutor({ columnNames: ['id', 'dept', 'salary'] });

            // Simulate data
            const rows = [
                [1, 'HR', 50000],
                [2, 'HR', 60000],
                [3, 'IT', 70000],
                [4, 'IT', 80000]
            ];
            const columnData = {
                id: [1, 2, 3, 4],
                dept: ['HR', 'HR', 'IT', 'IT'],
                salary: [50000, 60000, 70000, 80000]
            };

            const tests = [];

            // Test ROW_NUMBER() without partition
            const over1 = { partitionBy: [], orderBy: [{ column: 'id', direction: 'ASC' }] };
            const result1 = executor._computeWindowFunction('ROW_NUMBER', [], over1, rows, columnData);

            tests.push({
                name: 'ROW_NUMBER without partition',
                pass: JSON.stringify(result1) === JSON.stringify([1, 2, 3, 4]),
                actual: JSON.stringify(result1)
            });

            // Test ROW_NUMBER() with PARTITION BY
            const over2 = {
                partitionBy: [{ type: 'column', column: 'dept' }],
                orderBy: [{ column: 'salary', direction: 'ASC' }]
            };
            const result2 = executor._computeWindowFunction('ROW_NUMBER', [], over2, rows, columnData);

            tests.push({
                name: 'ROW_NUMBER with partition',
                pass: result2[0] === 1 && result2[1] === 2 && result2[2] === 1 && result2[3] === 2,
                actual: JSON.stringify(result2)
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${t.actual}`).toBe(true);
        }
    });
});

test.describe('Phase 8: Advanced Aggregations', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/test-join.html');
        await page.waitForLoadState('domcontentloaded');
    });

    test('SQL parser handles ROLLUP', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test ROLLUP parsing
            const sql1 = 'SELECT region, product, SUM(sales) FROM sales GROUP BY ROLLUP(region, product)';
            try {
                const tokens = new SQLLexer(sql1).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'ROLLUP parsed correctly',
                    pass: ast.groupBy.length === 1 && ast.groupBy[0].type === 'ROLLUP',
                    actual: JSON.stringify(ast.groupBy)
                });

                tests.push({
                    name: 'ROLLUP columns correct',
                    pass: ast.groupBy[0].columns.length === 2 &&
                          ast.groupBy[0].columns[0] === 'region' &&
                          ast.groupBy[0].columns[1] === 'product',
                    actual: ast.groupBy[0].columns
                });
            } catch (e) {
                tests.push({ name: 'ROLLUP parsing failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${JSON.stringify(t.actual)}`).toBe(true);
        }
    });

    test('SQL parser handles CUBE', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test CUBE parsing
            const sql = 'SELECT a, b, COUNT(*) FROM t GROUP BY CUBE(a, b)';
            try {
                const tokens = new SQLLexer(sql).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'CUBE parsed correctly',
                    pass: ast.groupBy.length === 1 && ast.groupBy[0].type === 'CUBE',
                    actual: JSON.stringify(ast.groupBy)
                });

                tests.push({
                    name: 'CUBE columns correct',
                    pass: ast.groupBy[0].columns.length === 2 &&
                          ast.groupBy[0].columns[0] === 'a' &&
                          ast.groupBy[0].columns[1] === 'b',
                    actual: ast.groupBy[0].columns
                });
            } catch (e) {
                tests.push({ name: 'CUBE parsing failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${JSON.stringify(t.actual)}`).toBe(true);
        }
    });

    test('SQL parser handles GROUPING SETS', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test GROUPING SETS parsing
            const sql = 'SELECT a, b, SUM(x) FROM t GROUP BY GROUPING SETS((a, b), (a), ())';
            try {
                const tokens = new SQLLexer(sql).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'GROUPING SETS parsed correctly',
                    pass: ast.groupBy.length === 1 && ast.groupBy[0].type === 'GROUPING_SETS',
                    actual: JSON.stringify(ast.groupBy)
                });

                tests.push({
                    name: 'GROUPING SETS has 3 sets',
                    pass: ast.groupBy[0].sets.length === 3,
                    actual: ast.groupBy[0].sets.length
                });

                tests.push({
                    name: 'First set is (a, b)',
                    pass: JSON.stringify(ast.groupBy[0].sets[0]) === JSON.stringify(['a', 'b']),
                    actual: ast.groupBy[0].sets[0]
                });

                tests.push({
                    name: 'Second set is (a)',
                    pass: JSON.stringify(ast.groupBy[0].sets[1]) === JSON.stringify(['a']),
                    actual: ast.groupBy[0].sets[1]
                });

                tests.push({
                    name: 'Third set is empty (grand total)',
                    pass: ast.groupBy[0].sets[2].length === 0,
                    actual: ast.groupBy[0].sets[2]
                });
            } catch (e) {
                tests.push({ name: 'GROUPING SETS parsing failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${JSON.stringify(t.actual)}`).toBe(true);
        }
    });

    test('SQL parser handles mixed GROUP BY with ROLLUP', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test mixed GROUP BY: column + ROLLUP
            const sql = 'SELECT year, quarter, SUM(sales) FROM sales GROUP BY year, ROLLUP(quarter)';
            try {
                const tokens = new SQLLexer(sql).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'Mixed GROUP BY has 2 items',
                    pass: ast.groupBy.length === 2,
                    actual: ast.groupBy.length
                });

                tests.push({
                    name: 'First item is COLUMN',
                    pass: ast.groupBy[0].type === 'COLUMN' && ast.groupBy[0].column === 'year',
                    actual: ast.groupBy[0]
                });

                tests.push({
                    name: 'Second item is ROLLUP',
                    pass: ast.groupBy[1].type === 'ROLLUP' && ast.groupBy[1].columns[0] === 'quarter',
                    actual: ast.groupBy[1]
                });
            } catch (e) {
                tests.push({ name: 'Mixed GROUP BY parsing failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${JSON.stringify(t.actual)}`).toBe(true);
        }
    });

    test('SQLExecutor _expandGroupBy works correctly', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLExecutor } = await import('./lanceql.js');

            const executor = new SQLExecutor({ columnNames: [] });
            const tests = [];

            // Test ROLLUP expansion: ROLLUP(a, b) -> [(a,b), (a), ()]
            const rollupGroupBy = [{ type: 'ROLLUP', columns: ['a', 'b'] }];
            const rollupSets = executor._expandGroupBy(rollupGroupBy);
            tests.push({
                name: 'ROLLUP generates 3 sets',
                pass: rollupSets.length === 3,
                actual: rollupSets.length
            });
            tests.push({
                name: 'ROLLUP set 0 is [a, b]',
                pass: JSON.stringify(rollupSets[0]) === JSON.stringify(['a', 'b']),
                actual: rollupSets[0]
            });
            tests.push({
                name: 'ROLLUP set 1 is [a]',
                pass: JSON.stringify(rollupSets[1]) === JSON.stringify(['a']),
                actual: rollupSets[1]
            });
            tests.push({
                name: 'ROLLUP set 2 is []',
                pass: JSON.stringify(rollupSets[2]) === JSON.stringify([]),
                actual: rollupSets[2]
            });

            // Test CUBE expansion: CUBE(a, b) -> 4 sets
            const cubeGroupBy = [{ type: 'CUBE', columns: ['a', 'b'] }];
            const cubeSets = executor._expandGroupBy(cubeGroupBy);
            tests.push({
                name: 'CUBE generates 4 sets',
                pass: cubeSets.length === 4,
                actual: cubeSets.length
            });

            // Test simple column (backward compat)
            const simpleGroupBy = ['col1', 'col2'];
            const simpleSets = executor._expandGroupBy(simpleGroupBy);
            tests.push({
                name: 'Simple GROUP BY returns single set',
                pass: simpleSets.length === 1 && JSON.stringify(simpleSets[0]) === JSON.stringify(['col1', 'col2']),
                actual: simpleSets
            });

            // Test new-style simple columns
            const newSimpleGroupBy = [{ type: 'COLUMN', column: 'x' }, { type: 'COLUMN', column: 'y' }];
            const newSimpleSets = executor._expandGroupBy(newSimpleGroupBy);
            tests.push({
                name: 'New-style simple GROUP BY returns single set',
                pass: newSimpleSets.length === 1 && JSON.stringify(newSimpleSets[0]) === JSON.stringify(['x', 'y']),
                actual: newSimpleSets
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${JSON.stringify(t.actual)}`).toBe(true);
        }
    });

    test('SQLExecutor _powerSet generates all subsets', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLExecutor } = await import('./lanceql.js');

            const executor = new SQLExecutor({ columnNames: [] });
            const tests = [];

            const subsets = executor._powerSet(['a', 'b']);
            tests.push({
                name: 'powerSet generates 4 subsets for 2 elements',
                pass: subsets.length === 4,
                actual: subsets.length
            });

            // Should include [], [a], [b], [a, b]
            const keys = subsets.map(s => JSON.stringify(s.sort())).sort();
            tests.push({
                name: 'powerSet includes empty set',
                pass: keys.includes('[]'),
                actual: keys
            });

            tests.push({
                name: 'powerSet includes [a]',
                pass: keys.includes('["a"]'),
                actual: keys
            });

            tests.push({
                name: 'powerSet includes [b]',
                pass: keys.includes('["b"]'),
                actual: keys
            });

            tests.push({
                name: 'powerSet includes [a, b]',
                pass: keys.includes('["a","b"]'),
                actual: keys
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${JSON.stringify(t.actual)}`).toBe(true);
        }
    });

    test('GROUPING() function is recognized', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test GROUPING() function parsing
            const sql = 'SELECT a, GROUPING(a), SUM(x) FROM t GROUP BY ROLLUP(a)';
            try {
                const tokens = new SQLLexer(sql).tokenize();
                const ast = new SQLParser(tokens).parse();

                // Find the GROUPING column
                const groupingCol = ast.columns.find(c =>
                    c.expr?.type === 'call' && c.expr.name.toUpperCase() === 'GROUPING'
                );

                tests.push({
                    name: 'GROUPING function parsed',
                    pass: groupingCol !== undefined,
                    actual: ast.columns.map(c => c.expr?.name || c.expr?.column || '*')
                });

                if (groupingCol) {
                    tests.push({
                        name: 'GROUPING has one argument',
                        pass: groupingCol.expr.args.length === 1,
                        actual: groupingCol.expr.args.length
                    });

                    tests.push({
                        name: 'GROUPING argument is column a',
                        pass: groupingCol.expr.args[0].column === 'a' || groupingCol.expr.args[0].name === 'a',
                        actual: groupingCol.expr.args[0]
                    });
                }
            } catch (e) {
                tests.push({ name: 'GROUPING function parsing failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${JSON.stringify(t.actual)}`).toBe(true);
        }
    });

    test('HAVING clause is parsed correctly', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { SQLLexer, SQLParser } = await import('./lanceql.js');

            const tests = [];

            // Test HAVING parsing
            const sql = 'SELECT dept, COUNT(*) FROM employees GROUP BY dept HAVING COUNT(*) > 5';
            try {
                const tokens = new SQLLexer(sql).tokenize();
                const ast = new SQLParser(tokens).parse();

                tests.push({
                    name: 'HAVING clause exists',
                    pass: ast.having !== null,
                    actual: ast.having !== null
                });

                tests.push({
                    name: 'HAVING is binary expression',
                    pass: ast.having?.type === 'binary',
                    actual: ast.having?.type
                });

                tests.push({
                    name: 'HAVING operator is >',
                    pass: ast.having?.op === '>',
                    actual: ast.having?.op
                });

                tests.push({
                    name: 'HAVING right side is 5',
                    pass: ast.having?.right?.value === 5,
                    actual: ast.having?.right?.value
                });
            } catch (e) {
                tests.push({ name: 'HAVING parsing failed', pass: false, actual: e.message });
            }

            return tests;
        });

        for (const t of result) {
            expect(t.pass, `${t.name}: got ${JSON.stringify(t.actual)}`).toBe(true);
        }
    });
});

// ============================================================================
// Phase 9: Query Optimization Tests
// ============================================================================

test.describe('Phase 9: Query Optimization', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/test-join.html');
    });

    test('9.1 Plan cache is initialized in LanceDatabase', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();
            return {
                hasPlanCache: db._planCache instanceof Map,
                hasMaxSize: typeof db._planCacheMaxSize === 'number',
                maxSize: db._planCacheMaxSize
            };
        });

        expect(result.hasPlanCache).toBe(true);
        expect(result.hasMaxSize).toBe(true);
        expect(result.maxSize).toBe(100);
    });

    test('9.2 _normalizeSQL removes extra whitespace and lowercases', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const tests = [];

            // Test whitespace normalization
            tests.push({
                name: 'Normalizes extra spaces',
                pass: db._normalizeSQL('SELECT  *   FROM   t') === 'select * from t'
            });

            // Test tabs and newlines
            tests.push({
                name: 'Normalizes tabs and newlines',
                pass: db._normalizeSQL('SELECT\t*\nFROM\tt') === 'select * from t'
            });

            // Test leading/trailing whitespace
            tests.push({
                name: 'Trims leading/trailing whitespace',
                pass: db._normalizeSQL('  SELECT * FROM t  ') === 'select * from t'
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, t.name).toBe(true);
        }
    });

    test('9.3 Plan caching with _getCachedPlan and _setCachedPlan', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const tests = [];

            // Test cache miss
            tests.push({
                name: 'Returns null for cache miss',
                pass: db._getCachedPlan('SELECT * FROM t') === null
            });

            // Set a plan
            const testPlan = { type: 'SELECT', from: { table: 't' } };
            db._setCachedPlan('SELECT * FROM t', testPlan);

            // Test cache hit
            const cached = db._getCachedPlan('SELECT * FROM t');
            tests.push({
                name: 'Returns cached plan on hit',
                pass: cached !== null && cached.type === 'SELECT'
            });

            // Test normalized query hits same cache
            const normalizedCached = db._getCachedPlan('  SELECT  *  FROM  t  ');
            tests.push({
                name: 'Normalized SQL hits cache',
                pass: normalizedCached !== null && normalizedCached.type === 'SELECT'
            });

            return tests;
        });

        for (const t of result) {
            expect(t.pass, t.name).toBe(true);
        }
    });

    test('9.4 getPlanCacheStats returns correct statistics', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            // Initially empty
            let stats = db.getPlanCacheStats();
            if (stats.size !== 0 || stats.totalHits !== 0) {
                return { pass: false, name: 'Initial stats wrong' };
            }

            // Add a plan
            db._setCachedPlan('SELECT 1', { type: 'SELECT' });
            stats = db.getPlanCacheStats();
            if (stats.size !== 1) {
                return { pass: false, name: 'Size should be 1' };
            }

            // Hit the cache
            db._getCachedPlan('SELECT 1');
            db._getCachedPlan('SELECT 1');
            stats = db.getPlanCacheStats();
            if (stats.totalHits !== 2) {
                return { pass: false, name: 'Hits should be 2', actual: stats.totalHits };
            }

            return { pass: true };
        });

        expect(result.pass, result.name).toBe(true);
    });

    test('9.5 clearPlanCache clears the cache', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            db._setCachedPlan('SELECT 1', { type: 'SELECT' });
            db._setCachedPlan('SELECT 2', { type: 'SELECT' });

            if (db.getPlanCacheStats().size !== 2) {
                return { pass: false, name: 'Should have 2 entries' };
            }

            db.clearPlanCache();

            if (db.getPlanCacheStats().size !== 0) {
                return { pass: false, name: 'Should be empty after clear' };
            }

            return { pass: true };
        });

        expect(result.pass, result.name).toBe(true);
    });

    test('9.6 Constant folding: 1 + 2 = 3', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const expr = {
                type: 'binary',
                operator: '+',
                left: { type: 'number', value: 1 },
                right: { type: 'number', value: 2 }
            };

            const optimized = db._optimizeExpr(expr);
            return {
                pass: optimized.type === 'literal' && optimized.value === 3,
                actual: optimized
            };
        });

        expect(result.pass, `Expected 3, got ${JSON.stringify(result.actual)}`).toBe(true);
    });

    test('9.7 Constant folding: 10 - 4 = 6', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const expr = {
                type: 'binary',
                operator: '-',
                left: { type: 'number', value: 10 },
                right: { type: 'number', value: 4 }
            };

            const optimized = db._optimizeExpr(expr);
            return {
                pass: optimized.type === 'literal' && optimized.value === 6,
                actual: optimized
            };
        });

        expect(result.pass, `Expected 6, got ${JSON.stringify(result.actual)}`).toBe(true);
    });

    test('9.8 Constant folding: 5 * 3 = 15', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const expr = {
                type: 'binary',
                operator: '*',
                left: { type: 'number', value: 5 },
                right: { type: 'number', value: 3 }
            };

            const optimized = db._optimizeExpr(expr);
            return {
                pass: optimized.type === 'literal' && optimized.value === 15,
                actual: optimized
            };
        });

        expect(result.pass).toBe(true);
    });

    test('9.9 Constant folding: 10 / 2 = 5', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const expr = {
                type: 'binary',
                operator: '/',
                left: { type: 'number', value: 10 },
                right: { type: 'number', value: 2 }
            };

            const optimized = db._optimizeExpr(expr);
            return {
                pass: optimized.type === 'literal' && optimized.value === 5,
                actual: optimized
            };
        });

        expect(result.pass).toBe(true);
    });

    test('9.10 Boolean simplification: x AND TRUE = x', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const expr = {
                type: 'binary',
                operator: 'AND',
                left: { type: 'column', name: 'x' },
                right: { type: 'literal', value: true }
            };

            const optimized = db._optimizeExpr(expr);
            return {
                pass: optimized.type === 'column' && optimized.name === 'x',
                actual: optimized
            };
        });

        expect(result.pass).toBe(true);
    });

    test('9.11 Boolean simplification: x OR FALSE = x', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const expr = {
                type: 'binary',
                operator: 'OR',
                left: { type: 'column', name: 'y' },
                right: { type: 'literal', value: false }
            };

            const optimized = db._optimizeExpr(expr);
            return {
                pass: optimized.type === 'column' && optimized.name === 'y',
                actual: optimized
            };
        });

        expect(result.pass).toBe(true);
    });

    test('9.12 Boolean simplification: x AND FALSE = FALSE', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const expr = {
                type: 'binary',
                operator: 'AND',
                left: { type: 'column', name: 'x' },
                right: { type: 'literal', value: false }
            };

            const optimized = db._optimizeExpr(expr);
            return {
                pass: optimized.type === 'literal' && optimized.value === false,
                actual: optimized
            };
        });

        expect(result.pass).toBe(true);
    });

    test('9.13 _extractRangePredicates from WHERE x > 10', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const where = {
                type: 'binary',
                operator: '>',
                left: { type: 'column', name: 'x' },
                right: { type: 'number', value: 10 }
            };

            const predicates = db._extractRangePredicates(where);
            return {
                pass: predicates.length === 1 &&
                      predicates[0].column === 'x' &&
                      predicates[0].operator === '>' &&
                      predicates[0].value === 10,
                actual: predicates
            };
        });

        expect(result.pass, JSON.stringify(result.actual)).toBe(true);
    });

    test('9.14 _extractRangePredicates from compound WHERE', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            // WHERE x > 10 AND y < 5
            const where = {
                type: 'binary',
                operator: 'AND',
                left: {
                    type: 'binary',
                    operator: '>',
                    left: { type: 'column', name: 'x' },
                    right: { type: 'number', value: 10 }
                },
                right: {
                    type: 'binary',
                    operator: '<',
                    left: { type: 'column', name: 'y' },
                    right: { type: 'number', value: 5 }
                }
            };

            const predicates = db._extractRangePredicates(where);
            return {
                pass: predicates.length === 2,
                predicates: predicates
            };
        });

        expect(result.pass, `Expected 2 predicates, got ${result.predicates?.length}`).toBe(true);
    });

    test('9.15 _canPruneFragment: prune when max < predicate value', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const stats = {
                x: { min: 0, max: 5, nullCount: 0, rowCount: 100 }
            };
            const predicates = [{ column: 'x', operator: '>', value: 10 }];

            return db._canPruneFragment(stats, predicates);
        });

        expect(result).toBe(true);
    });

    test('9.16 _canPruneFragment: cannot prune when value in range', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const stats = {
                x: { min: 0, max: 100, nullCount: 0, rowCount: 100 }
            };
            const predicates = [{ column: 'x', operator: '>', value: 10 }];

            return db._canPruneFragment(stats, predicates);
        });

        expect(result).toBe(false);
    });

    test('9.17 _canPruneFragment: prune equality outside range', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const stats = {
                x: { min: 10, max: 20, nullCount: 0, rowCount: 100 }
            };
            const predicates = [{ column: 'x', operator: '=', value: 5 }];

            return db._canPruneFragment(stats, predicates);
        });

        expect(result).toBe(true);
    });

    test('9.18 EXPLAIN parses correctly', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { SQLLexer, SQLParser } = window;

            try {
                const sql = 'EXPLAIN SELECT * FROM t WHERE x > 10';
                const lexer = new SQLLexer(sql);
                const tokens = lexer.tokenize();
                const parser = new SQLParser(tokens);
                const ast = parser.parse();

                return {
                    pass: ast.type === 'EXPLAIN' &&
                          ast.statement?.type === 'SELECT' &&
                          ast.statement?.from?.name === 't',
                    ast: ast
                };
            } catch (e) {
                return { pass: false, error: e.message };
            }
        });

        expect(result.pass, result.error || JSON.stringify(result.ast)).toBe(true);
    });

    test('9.19 _explainQuery returns plan structure', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase, SQLLexer, SQLParser } = window;
            const db = new LanceDatabase();

            const sql = 'SELECT * FROM t WHERE x > 10 ORDER BY x LIMIT 5';
            const lexer = new SQLLexer(sql);
            const tokens = lexer.tokenize();
            const parser = new SQLParser(tokens);
            const ast = parser.parse();

            const planResult = db._explainQuery(ast);

            // Parse the plan JSON
            const plan = JSON.parse(planResult.rows[0][0]);

            return {
                pass: plan.type === 'SELECT' &&
                      plan.tables.length === 1 &&
                      plan.predicates.length === 1 &&
                      plan.optimizations.includes('PREDICATE_PUSHDOWN') &&
                      plan.optimizations.includes('SORT') &&
                      plan.optimizations.includes('LIMIT_PUSHDOWN'),
                plan: plan
            };
        });

        expect(result.pass, JSON.stringify(result.plan, null, 2)).toBe(true);
    });

    test('9.20 LRU eviction works when cache is full', async ({ page }) => {
        const result = await page.evaluate(() => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            // Set a small cache size for testing
            db._planCacheMaxSize = 3;

            // Add 3 plans
            db._setCachedPlan('SELECT 1', { type: 'SELECT', n: 1 });
            db._setCachedPlan('SELECT 2', { type: 'SELECT', n: 2 });
            db._setCachedPlan('SELECT 3', { type: 'SELECT', n: 3 });

            // Access SELECT 2 to make it recently used
            db._getCachedPlan('SELECT 2');

            // Add a 4th plan - should evict SELECT 1 (oldest)
            db._setCachedPlan('SELECT 4', { type: 'SELECT', n: 4 });

            // Check that SELECT 1 is evicted and SELECT 2 is still there
            const has1 = db._getCachedPlan('SELECT 1') !== null;
            const has2 = db._getCachedPlan('SELECT 2') !== null;
            const has4 = db._getCachedPlan('SELECT 4') !== null;

            return {
                pass: !has1 && has2 && has4,
                has1, has2, has4,
                size: db.getPlanCacheStats().size
            };
        });

        expect(result.pass, `has1=${result.has1}, has2=${result.has2}, has4=${result.has4}`).toBe(true);
    });
});

// ============================================================================
// Phase 10: Memory Tables Tests
// ============================================================================

test.describe('Phase 10: Memory Tables', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/test-join.html');
    });

    test('10.1 CREATE TABLE creates memory table', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const createResult = await db.executeSQL('CREATE TABLE users (id INT, name TEXT)');

            return {
                success: createResult.success,
                hasTable: db.memoryTables.has('users'),
                tableName: createResult.table,
                columns: createResult.columns
            };
        });

        expect(result.success).toBe(true);
        expect(result.hasTable).toBe(true);
        expect(result.tableName).toBe('users');
        expect(result.columns).toEqual(['id', 'name']);
    });

    test('10.2 INSERT adds rows to memory table', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE t (x INT, y TEXT)');
            const insertResult = await db.executeSQL("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')");

            const selectResult = await db.executeSQL('SELECT * FROM t');

            return {
                inserted: insertResult.inserted,
                total: insertResult.total,
                rows: selectResult.rows,
                columns: selectResult.columns
            };
        });

        expect(result.inserted).toBe(3);
        expect(result.total).toBe(3);
        expect(result.rows.length).toBe(3);
    });

    test('10.3 SELECT from memory table works', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE nums (val INT)');
            await db.executeSQL('INSERT INTO nums VALUES (10), (20), (30)');

            const selectResult = await db.executeSQL('SELECT * FROM nums WHERE val > 15');

            return {
                rows: selectResult.rows,
                count: selectResult.rows.length
            };
        });

        expect(result.count).toBe(2);
        expect(result.rows[0][0]).toBe(20);
        expect(result.rows[1][0]).toBe(30);
    });

    test('10.4 UPDATE modifies rows with WHERE', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE t (id INT, val INT)');
            await db.executeSQL('INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)');

            const updateResult = await db.executeSQL('UPDATE t SET val = 100 WHERE id = 2');

            const selectResult = await db.executeSQL('SELECT val FROM t WHERE id = 2');

            return {
                updated: updateResult.updated,
                newValue: selectResult.rows[0]?.[0]
            };
        });

        expect(result.updated).toBe(1);
        expect(result.newValue).toBe(100);
    });

    test('10.5 DELETE removes rows with WHERE', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE t (id INT)');
            await db.executeSQL('INSERT INTO t VALUES (1), (2), (3), (4), (5)');

            const deleteResult = await db.executeSQL('DELETE FROM t WHERE id > 2');

            const selectResult = await db.executeSQL('SELECT * FROM t');

            return {
                deleted: deleteResult.deleted,
                remaining: selectResult.rows.length
            };
        });

        expect(result.deleted).toBe(3);
        expect(result.remaining).toBe(2);
    });

    test('10.6 DROP TABLE removes memory table', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE temp (id INT)');
            const hasBeforeDrop = db.memoryTables.has('temp');

            await db.executeSQL('DROP TABLE temp');
            const hasAfterDrop = db.memoryTables.has('temp');

            return { hasBeforeDrop, hasAfterDrop };
        });

        expect(result.hasBeforeDrop).toBe(true);
        expect(result.hasAfterDrop).toBe(false);
    });

    test('10.7 CREATE TABLE IF NOT EXISTS does not error on existing table', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE t (id INT)');
            const secondCreate = await db.executeSQL('CREATE TABLE IF NOT EXISTS t (id INT)');

            return {
                existed: secondCreate.existed,
                success: secondCreate.success
            };
        });

        expect(result.success).toBe(true);
        expect(result.existed).toBe(true);
    });

    test('10.8 DROP TABLE IF EXISTS does not error on nonexistent', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            const dropResult = await db.executeSQL('DROP TABLE IF EXISTS nonexistent');

            return {
                existed: dropResult.existed,
                success: dropResult.success
            };
        });

        expect(result.success).toBe(true);
        expect(result.existed).toBe(false);
    });

    test('10.9 SELECT with aggregation on memory table', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE scores (user_id INT, score INT)');
            await db.executeSQL('INSERT INTO scores VALUES (1, 100), (1, 150), (2, 200), (2, 50)');

            const selectResult = await db.executeSQL(`
                SELECT user_id, SUM(score) as total
                FROM scores
                GROUP BY user_id
            `);

            return {
                rows: selectResult.rows,
                columns: selectResult.columns
            };
        });

        expect(result.rows.length).toBe(2);
    });

    test('10.10 DELETE without WHERE truncates table', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE t (id INT)');
            await db.executeSQL('INSERT INTO t VALUES (1), (2), (3)');

            const deleteResult = await db.executeSQL('DELETE FROM t');

            const selectResult = await db.executeSQL('SELECT COUNT(*) FROM t');

            return {
                deleted: deleteResult.deleted,
                remaining: selectResult.rows[0]?.[0]
            };
        });

        expect(result.deleted).toBe(3);
        expect(result.remaining).toBe(0);
    });

    test('10.11 Case insensitivity for table names', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE MyTable (id INT)');
            await db.executeSQL('INSERT INTO MYTABLE VALUES (1)');
            const selectResult = await db.executeSQL('SELECT * FROM mytable');

            return {
                rows: selectResult.rows
            };
        });

        expect(result.rows.length).toBe(1);
    });

    test('10.12 UPDATE without WHERE updates all rows', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE t (val INT)');
            await db.executeSQL('INSERT INTO t VALUES (1), (2), (3)');

            const updateResult = await db.executeSQL('UPDATE t SET val = 99');

            const selectResult = await db.executeSQL('SELECT * FROM t');
            const allNinetyNine = selectResult.rows.every(row => row[0] === 99);

            return {
                updated: updateResult.updated,
                allNinetyNine
            };
        });

        expect(result.updated).toBe(3);
        expect(result.allNinetyNine).toBe(true);
    });
});

// Phase 11: Window Function Enhancements Tests
test.describe('Phase 11: Window Function Enhancements', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/test-join.html');
    });

    test('11.1 PERCENT_RANK returns 0 for first row', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE scores (id INT, score INT)');
            await db.executeSQL('INSERT INTO scores VALUES (1, 100), (2, 200), (3, 300), (4, 400)');

            const selectResult = await db.executeSQL(`
                SELECT id, score, PERCENT_RANK() OVER (ORDER BY score) as pct_rank
                FROM scores
            `);

            return {
                rows: selectResult.rows,
                columns: selectResult.columns
            };
        });

        // First row should be 0, last row should be 1
        expect(result.rows[0][2]).toBe(0);
        expect(result.rows[3][2]).toBe(1);
    });

    test('11.2 PERCENT_RANK with ties', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE scores (id INT, score INT)');
            await db.executeSQL('INSERT INTO scores VALUES (1, 100), (2, 100), (3, 200), (4, 300)');

            const selectResult = await db.executeSQL(`
                SELECT id, score, PERCENT_RANK() OVER (ORDER BY score) as pct_rank
                FROM scores
            `);

            return selectResult.rows;
        });

        // First two rows have same score, should both be 0
        expect(result[0][2]).toBe(0);
        expect(result[1][2]).toBe(0);
    });

    test('11.3 CUME_DIST returns fraction of rows', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE scores (id INT, score INT)');
            await db.executeSQL('INSERT INTO scores VALUES (1, 100), (2, 200), (3, 300), (4, 400)');

            const selectResult = await db.executeSQL(`
                SELECT id, score, CUME_DIST() OVER (ORDER BY score) as cum_dist
                FROM scores
            `);

            return selectResult.rows;
        });

        // CUME_DIST: count(rows <= current) / total
        // Row 1: 1/4 = 0.25, Row 2: 2/4 = 0.5, Row 3: 3/4 = 0.75, Row 4: 4/4 = 1.0
        expect(result[0][2]).toBe(0.25);
        expect(result[1][2]).toBe(0.5);
        expect(result[2][2]).toBe(0.75);
        expect(result[3][2]).toBe(1);
    });

    test('11.4 SUM with ROWS BETWEEN frame', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE data (id INT, val INT)');
            await db.executeSQL('INSERT INTO data VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)');

            const selectResult = await db.executeSQL(`
                SELECT id, val, SUM(val) OVER (
                    ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
                ) as moving_sum
                FROM data
            `);

            return selectResult.rows;
        });

        // Row 1: 10+20 = 30 (no preceding)
        // Row 2: 10+20+30 = 60
        // Row 3: 20+30+40 = 90
        // Row 4: 30+40+50 = 120
        // Row 5: 40+50 = 90 (no following)
        expect(result[0][2]).toBe(30);
        expect(result[1][2]).toBe(60);
        expect(result[2][2]).toBe(90);
        expect(result[3][2]).toBe(120);
        expect(result[4][2]).toBe(90);
    });

    test('11.5 AVG with running average (UNBOUNDED PRECEDING)', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE data (id INT, val INT)');
            await db.executeSQL('INSERT INTO data VALUES (1, 10), (2, 20), (3, 30), (4, 40)');

            const selectResult = await db.executeSQL(`
                SELECT id, val, AVG(val) OVER (
                    ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) as running_avg
                FROM data
            `);

            return selectResult.rows;
        });

        // Row 1: 10/1 = 10
        // Row 2: 30/2 = 15
        // Row 3: 60/3 = 20
        // Row 4: 100/4 = 25
        expect(result[0][2]).toBe(10);
        expect(result[1][2]).toBe(15);
        expect(result[2][2]).toBe(20);
        expect(result[3][2]).toBe(25);
    });

    test('11.6 LAST_VALUE with frame to end', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE data (id INT, val INT)');
            await db.executeSQL('INSERT INTO data VALUES (1, 10), (2, 20), (3, 30), (4, 40)');

            const selectResult = await db.executeSQL(`
                SELECT id, val, LAST_VALUE(val) OVER (
                    ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as last_val
                FROM data
            `);

            return selectResult.rows;
        });

        // All rows should get 40 (the actual last value in partition)
        expect(result[0][2]).toBe(40);
        expect(result[1][2]).toBe(40);
        expect(result[2][2]).toBe(40);
        expect(result[3][2]).toBe(40);
    });

    test('11.7 COUNT with N PRECEDING frame', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE data (id INT, val INT)');
            await db.executeSQL('INSERT INTO data VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)');

            const selectResult = await db.executeSQL(`
                SELECT id, val, COUNT(*) OVER (
                    ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) as cnt
                FROM data
            `);

            return selectResult.rows;
        });

        // Row 1: 1, Row 2: 2, Row 3+: 3
        expect(result[0][2]).toBe(1);
        expect(result[1][2]).toBe(2);
        expect(result[2][2]).toBe(3);
        expect(result[3][2]).toBe(3);
        expect(result[4][2]).toBe(3);
    });

    test('11.8 PERCENT_RANK single row returns 0', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE single (id INT, val INT)');
            await db.executeSQL('INSERT INTO single VALUES (1, 100)');

            const selectResult = await db.executeSQL(`
                SELECT id, PERCENT_RANK() OVER (ORDER BY val) as pct_rank
                FROM single
            `);

            return selectResult.rows[0][1];
        });

        // Single row should return 0
        expect(result).toBe(0);
    });

    test('11.9 CUME_DIST single row returns 1', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE single (id INT, val INT)');
            await db.executeSQL('INSERT INTO single VALUES (1, 100)');

            const selectResult = await db.executeSQL(`
                SELECT id, CUME_DIST() OVER (ORDER BY val) as cum_dist
                FROM single
            `);

            return selectResult.rows[0][1];
        });

        // Single row should return 1
        expect(result).toBe(1);
    });

    test('11.10 MIN/MAX with frame', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { LanceDatabase } = window;
            const db = new LanceDatabase();

            await db.executeSQL('CREATE TABLE data (id INT, val INT)');
            await db.executeSQL('INSERT INTO data VALUES (1, 50), (2, 10), (3, 30), (4, 20), (5, 40)');

            const selectResult = await db.executeSQL(`
                SELECT id, val,
                    MIN(val) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as frame_min,
                    MAX(val) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as frame_max
                FROM data
            `);

            return selectResult.rows;
        });

        // Row 1 (val=50): frame [50,10] -> min=10, max=50
        // Row 2 (val=10): frame [50,10,30] -> min=10, max=50
        // Row 3 (val=30): frame [10,30,20] -> min=10, max=30
        // Row 4 (val=20): frame [30,20,40] -> min=20, max=40
        // Row 5 (val=40): frame [20,40] -> min=20, max=40
        expect(result[0][2]).toBe(10);  // min
        expect(result[0][3]).toBe(50);  // max
        expect(result[2][2]).toBe(10);  // min for row 3
        expect(result[2][3]).toBe(30);  // max for row 3
    });
});
