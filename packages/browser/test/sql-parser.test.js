/**
 * Unit tests for SQL Parser
 */

import { test, describe } from 'node:test';
import assert from 'node:assert';

import { SQLLexer } from '../src/client/sql/lexer.js';
import { SQLParser } from '../src/client/sql/parser.js';

// Helper to parse SQL
function parse(sql) {
    const lexer = new SQLLexer(sql);
    const tokens = lexer.tokenize();
    const parser = new SQLParser(tokens);
    return parser.parse();
}

describe('SQL Parser', () => {
    describe('SELECT statements', () => {
        test('parses simple SELECT *', () => {
            const ast = parse('SELECT * FROM users');
            assert.strictEqual(ast.type, 'SELECT');
            assert.ok(ast.columns);
            assert.ok(ast.from);
        });

        test('parses SELECT with specific columns', () => {
            const ast = parse('SELECT id, name, email FROM users');
            assert.strictEqual(ast.columns.length, 3);
        });

        test('parses SELECT with aliases', () => {
            const ast = parse('SELECT id AS user_id, name AS user_name FROM users');
            assert.ok(ast.columns[0].alias);
            assert.ok(ast.columns[1].alias);
        });
    });

    describe('WHERE clause', () => {
        test('parses simple WHERE', () => {
            const ast = parse('SELECT * FROM users WHERE id = 1');
            assert.ok(ast.where);
        });

        test('parses WHERE with AND/OR', () => {
            const ast = parse('SELECT * FROM users WHERE active = true AND age > 18');
            assert.ok(ast.where);
        });

        test('parses WHERE with IN', () => {
            const ast = parse('SELECT * FROM users WHERE id IN (1, 2, 3)');
            assert.ok(ast.where);
        });

        test('parses WHERE with BETWEEN', () => {
            const ast = parse('SELECT * FROM users WHERE age BETWEEN 18 AND 65');
            assert.ok(ast.where);
        });

        test('parses WHERE with LIKE', () => {
            const ast = parse("SELECT * FROM users WHERE name LIKE '%john%'");
            assert.ok(ast.where);
        });
    });

    describe('JOIN clauses', () => {
        test('parses INNER JOIN', () => {
            const ast = parse('SELECT * FROM users JOIN orders ON users.id = orders.user_id');
            assert.ok(ast.joins);
            assert.strictEqual(ast.joins.length, 1);
        });

        test('parses LEFT JOIN', () => {
            const ast = parse('SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id');
            assert.ok(ast.joins);
        });

        test('parses multiple JOINs', () => {
            const ast = parse(`
                SELECT * FROM users
                JOIN orders ON users.id = orders.user_id
                JOIN products ON orders.product_id = products.id
            `);
            assert.strictEqual(ast.joins.length, 2);
        });
    });

    describe('GROUP BY and HAVING', () => {
        test('parses GROUP BY', () => {
            const ast = parse('SELECT dept, COUNT(*) FROM employees GROUP BY dept');
            assert.ok(ast.groupBy);
        });

        test('parses GROUP BY with HAVING', () => {
            const ast = parse('SELECT dept, COUNT(*) as cnt FROM employees GROUP BY dept HAVING cnt > 5');
            assert.ok(ast.having);
        });
    });

    describe('ORDER BY and LIMIT', () => {
        test('parses ORDER BY', () => {
            const ast = parse('SELECT * FROM users ORDER BY name ASC');
            assert.ok(ast.orderBy);
        });

        test('parses LIMIT', () => {
            const ast = parse('SELECT * FROM users LIMIT 10');
            assert.strictEqual(ast.limit, 10);
        });

        test('parses LIMIT with OFFSET', () => {
            const ast = parse('SELECT * FROM users LIMIT 10 OFFSET 20');
            assert.strictEqual(ast.limit, 10);
            assert.strictEqual(ast.offset, 20);
        });
    });

    describe('Aggregate functions', () => {
        test('parses COUNT(*)', () => {
            const ast = parse('SELECT COUNT(*) FROM users');
            assert.ok(ast.columns[0]);
        });

        test('parses SUM, AVG, MIN, MAX', () => {
            const ast = parse('SELECT SUM(amount), AVG(price), MIN(id), MAX(id) FROM orders');
            assert.strictEqual(ast.columns.length, 4);
        });
    });

    describe('CTE (WITH clause)', () => {
        test('parses simple CTE', () => {
            const ast = parse(`
                WITH active_users AS (SELECT * FROM users WHERE active = true)
                SELECT * FROM active_users
            `);
            assert.ok(ast.ctes); // Parser stores CTEs in ctes array
            assert.strictEqual(ast.ctes.length, 1);
            assert.strictEqual(ast.ctes[0].name, 'active_users');
        });
    });

    describe('Window functions', () => {
        test('parses ROW_NUMBER with OVER', () => {
            const ast = parse('SELECT ROW_NUMBER() OVER (ORDER BY id) as rn FROM users');
            assert.ok(ast.columns[0]);
        });
    });

    describe('UNION', () => {
        test('parses UNION', () => {
            const ast = parse('SELECT id FROM users UNION SELECT id FROM admins');
            // Parser returns SET_OPERATION type for UNION/INTERSECT/EXCEPT
            assert.strictEqual(ast.type, 'SET_OPERATION');
            assert.strictEqual(ast.operator, 'UNION');
            assert.ok(ast.left);
            assert.ok(ast.right);
        });
    });

    describe('INSERT statements', () => {
        test('parses INSERT INTO', () => {
            const ast = parse("INSERT INTO users (name, email) VALUES ('John', 'john@example.com')");
            assert.strictEqual(ast.type, 'INSERT');
            assert.strictEqual(ast.table, 'users');
        });
    });

    describe('UPDATE statements', () => {
        test('parses UPDATE', () => {
            const ast = parse("UPDATE users SET name = 'Jane' WHERE id = 1");
            assert.strictEqual(ast.type, 'UPDATE');
            assert.strictEqual(ast.table, 'users');
        });
    });

    describe('DELETE statements', () => {
        test('parses DELETE', () => {
            const ast = parse('DELETE FROM users WHERE id = 1');
            assert.strictEqual(ast.type, 'DELETE');
            assert.strictEqual(ast.table, 'users');
        });
    });

    describe('CREATE TABLE statements', () => {
        test('parses CREATE TABLE', () => {
            const ast = parse('CREATE TABLE users (id INT, name TEXT)');
            assert.strictEqual(ast.type, 'CREATE_TABLE');
            assert.strictEqual(ast.table, 'users');
        });
    });
});
