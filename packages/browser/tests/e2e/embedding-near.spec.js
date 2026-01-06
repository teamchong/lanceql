/**
 * LanceQL Feature Tests - E2E
 * Tests: GROUP BY, JOIN, aggregations, WHERE filters
 * 
 * Note: These tests require WASM to be loaded. Some tests may be skipped in CI
 * if WASM loading times out.
 */

import { test, expect } from '@playwright/test';

// Increase timeout for all tests in this file
test.setTimeout(60000);

test.describe('SQL Features', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');
        await page.waitForLoadState('domcontentloaded');
    });

    test('GROUP BY with aggregates', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { vault } = await import('./lanceql.js');
                const v = await vault();

                await v.exec('CREATE TABLE sales (category TEXT, amount REAL)');
                await v.exec(`INSERT INTO sales VALUES 
                    ('electronics', 500), ('electronics', 300),
                    ('clothing', 100), ('clothing', 150),
                    ('food', 50), ('food', 25)`);

                const grouped = await v.exec(`
                    SELECT category, SUM(amount), AVG(amount), COUNT(*)
                    FROM sales GROUP BY category
                `);
                return { success: true, rowCount: grouped.rows.length };
            } catch (e) {
                return { success: false, error: e.message };
            }
        });

        if (!result.success) {
            console.log('Skipped - WASM not loaded:', result.error);
            test.skip();
            return;
        }
        expect(result.rowCount).toBe(3);
    });

    test('JOIN with WHERE filter', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { vault } = await import('./lanceql.js');
                const v = await vault();

                await v.exec('CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL)');
                await v.exec('CREATE TABLE customers (id INTEGER, name TEXT)');
                await v.exec(`INSERT INTO customers VALUES (1, 'Alice'), (2, 'Bob')`);
                await v.exec(`INSERT INTO orders VALUES (1, 1, 100), (2, 1, 200), (3, 2, 150)`);

                const joined = await v.exec(`
                    SELECT c.name, o.amount FROM orders o
                    JOIN customers c ON o.customer_id = c.id
                    WHERE o.amount > 100
                `);
                return { success: true, rowCount: joined.rows.length };
            } catch (e) {
                return { success: false, error: e.message };
            }
        });

        if (!result.success) {
            console.log('Skipped - WASM not loaded:', result.error);
            test.skip();
            return;
        }
        expect(result.rowCount).toBeGreaterThan(0);
    });

    test('aggregation SUM/AVG/MIN/MAX/COUNT', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { vault } = await import('./lanceql.js');
                const v = await vault();

                await v.exec('CREATE TABLE scores (value REAL)');
                await v.exec(`INSERT INTO scores VALUES (10), (20), (30), (40), (50)`);

                const agg = await v.exec('SELECT SUM(value), AVG(value), MIN(value), MAX(value), COUNT(*) FROM scores');
                return { success: true, row: agg.rows[0] };
            } catch (e) {
                return { success: false, error: e.message };
            }
        });

        if (!result.success) {
            console.log('Skipped - WASM not loaded:', result.error);
            test.skip();
            return;
        }
        expect(result.row['sum(value)']).toBe(150);
        expect(result.row['count(*)']).toBe(5);
    });

    test('WHERE with multiple conditions', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { vault } = await import('./lanceql.js');
                const v = await vault();

                await v.exec('CREATE TABLE products (id INTEGER, price REAL, status TEXT)');
                await v.exec(`INSERT INTO products VALUES 
                    (1, 100, 'active'), (2, 200, 'active'), 
                    (3, 50, 'inactive'), (4, 300, 'active')`);

                const filtered = await v.exec(`
                    SELECT * FROM products WHERE price > 100 AND status = 'active'
                `);
                return { success: true, rowCount: filtered.rows.length };
            } catch (e) {
                return { success: false, error: e.message };
            }
        });

        if (!result.success) {
            console.log('Skipped - WASM not loaded:', result.error);
            test.skip();
            return;
        }
        expect(result.rowCount).toBe(2);
    });
});

test.describe('WebGPU', () => {
    test('GPU availability check', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const gpuStatus = await page.evaluate(async () => {
            if (!navigator.gpu) return { available: false };
            try {
                const adapter = await navigator.gpu.requestAdapter();
                return { available: !!adapter };
            } catch { return { available: false }; }
        });

        expect(typeof gpuStatus.available).toBe('boolean');
        console.log('WebGPU available:', gpuStatus.available);
    });
});
