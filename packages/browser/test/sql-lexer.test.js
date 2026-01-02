/**
 * Unit tests for SQL Lexer
 */

import { test, describe } from 'node:test';
import assert from 'node:assert';

// Import from source for unit testing
import { SQLLexer } from '../src/client/sql/lexer.js';

describe('SQL Lexer', () => {
    test('tokenizes simple SELECT', () => {
        const lexer = new SQLLexer('SELECT * FROM users');
        const tokens = lexer.tokenize();

        assert.strictEqual(tokens[0].type, 'SELECT');
        assert.strictEqual(tokens[0].value, 'SELECT');
        assert.strictEqual(tokens[1].type, 'STAR');
        assert.strictEqual(tokens[2].type, 'FROM');
        assert.strictEqual(tokens[2].value, 'FROM');
        assert.strictEqual(tokens[3].type, 'IDENTIFIER');
        assert.strictEqual(tokens[3].value, 'users');
    });

    test('tokenizes numbers', () => {
        const lexer = new SQLLexer('SELECT 42, 3.14, -100');
        const tokens = lexer.tokenize();

        const numbers = tokens.filter(t => t.type === 'NUMBER');
        assert.strictEqual(numbers.length, 3);
        assert.strictEqual(numbers[0].value, '42');
        assert.strictEqual(numbers[1].value, '3.14');
    });

    test('tokenizes strings with single quotes', () => {
        const lexer = new SQLLexer("SELECT 'hello world'");
        const tokens = lexer.tokenize();

        const strings = tokens.filter(t => t.type === 'STRING');
        assert.strictEqual(strings.length, 1);
        assert.strictEqual(strings[0].value, 'hello world');
    });

    test('tokenizes strings with double quotes', () => {
        const lexer = new SQLLexer('SELECT "hello world"');
        const tokens = lexer.tokenize();

        const strings = tokens.filter(t => t.type === 'STRING');
        assert.strictEqual(strings.length, 1);
        assert.strictEqual(strings[0].value, 'hello world');
    });

    test('tokenizes comparison operators', () => {
        const lexer = new SQLLexer('a = b AND c != d AND e >= f AND g <= h');
        const tokens = lexer.tokenize();

        const ops = tokens.filter(t => ['EQ', 'NE', 'GE', 'LE'].includes(t.type));
        assert.strictEqual(ops.length, 4);
    });

    test('tokenizes parentheses', () => {
        const lexer = new SQLLexer('(a + b) * c');
        const tokens = lexer.tokenize();

        assert.strictEqual(tokens[0].type, 'LPAREN');
        assert.strictEqual(tokens[4].type, 'RPAREN');
    });

    test('tokenizes qualified identifiers', () => {
        const lexer = new SQLLexer('SELECT users.name, orders.id');
        const tokens = lexer.tokenize();

        const identifiers = tokens.filter(t => t.type === 'IDENTIFIER');
        assert.ok(identifiers.some(i => i.value === 'users'));
        assert.ok(identifiers.some(i => i.value === 'name'));
    });

    test('tokenizes keywords case-insensitively', () => {
        const lexer = new SQLLexer('select FROM where');
        const tokens = lexer.tokenize();

        // Keywords are stored with their specific type (SELECT, FROM, WHERE)
        assert.strictEqual(tokens[0].type, 'SELECT');
        assert.strictEqual(tokens[1].type, 'FROM');
        assert.strictEqual(tokens[2].type, 'WHERE');
    });

    test('tokenizes arrays', () => {
        const lexer = new SQLLexer('SELECT [1, 2, 3]');
        const tokens = lexer.tokenize();

        assert.ok(tokens.some(t => t.type === 'LBRACKET'));
        assert.ok(tokens.some(t => t.type === 'RBRACKET'));
    });

    test('tokenizes complex query', () => {
        const sql = `
            SELECT u.name, COUNT(*) as cnt
            FROM users u
            JOIN orders o ON u.id = o.user_id
            WHERE u.active = true
            GROUP BY u.name
            HAVING COUNT(*) > 5
            ORDER BY cnt DESC
            LIMIT 10
        `;
        const lexer = new SQLLexer(sql);
        const tokens = lexer.tokenize();

        assert.ok(tokens.length > 20);
        assert.ok(tokens.some(t => t.value === 'JOIN'));
        assert.ok(tokens.some(t => t.value === 'GROUP'));
        assert.ok(tokens.some(t => t.value === 'HAVING'));
        assert.ok(tokens.some(t => t.value === 'ORDER'));
        assert.ok(tokens.some(t => t.value === 'LIMIT'));
    });

    test('handles escaped quotes in strings', () => {
        const lexer = new SQLLexer("SELECT 'it''s working'");
        const tokens = lexer.tokenize();

        const strings = tokens.filter(t => t.type === 'STRING');
        assert.strictEqual(strings.length, 1);
    });

    test('tokenizes window functions', () => {
        const lexer = new SQLLexer('ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)');
        const tokens = lexer.tokenize();

        assert.ok(tokens.some(t => t.value === 'ROW_NUMBER'));
        assert.ok(tokens.some(t => t.value === 'OVER'));
        assert.ok(tokens.some(t => t.value === 'PARTITION'));
    });
});
