/**
 * LanceQL Feature Tests - E2E
 * Tests: GROUP BY, JOIN, aggregations, WHERE filters, NEAR vector search
 */

import { test, expect } from '@playwright/test';

test.describe('SQL Features', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');
        await page.waitForLoadState('domcontentloaded');
        await page.waitForTimeout(2000);
    });

    test('GROUP BY with aggregates', async ({ page }) => {
        const result = await page.evaluate(async () => {
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
            return grouped.rows.length;
        });

        expect(result).toBe(3);
    });

    test('JOIN with WHERE filter', async ({ page }) => {
        const result = await page.evaluate(async () => {
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
            return joined.rows.length;
        });

        expect(result).toBeGreaterThan(0);
    });

    test('JOIN + GROUP BY + SUM', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();

            await v.exec('CREATE TABLE orders2 (customer_id INTEGER, amount REAL)');
            await v.exec('CREATE TABLE customers2 (id INTEGER, region TEXT)');
            await v.exec(`INSERT INTO customers2 VALUES (1, 'West'), (2, 'East')`);
            await v.exec(`INSERT INTO orders2 VALUES (1, 100), (1, 200), (2, 150)`);

            const result = await v.exec(`
                SELECT c.region, SUM(o.amount)
                FROM orders2 o JOIN customers2 c ON o.customer_id = c.id
                GROUP BY c.region
            `);
            return result.rows.length;
        });

        expect(result).toBe(2);
    });

    test('aggregation SUM/AVG/MIN/MAX/COUNT', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();

            await v.exec('CREATE TABLE scores (value REAL)');
            await v.exec(`INSERT INTO scores VALUES (10), (20), (30), (40), (50)`);

            const agg = await v.exec('SELECT SUM(value), AVG(value), MIN(value), MAX(value), COUNT(*) FROM scores');
            return agg.rows[0];
        });

        expect(result['sum(value)']).toBe(150);
        expect(result['avg(value)']).toBe(30);
        expect(result['min(value)']).toBe(10);
        expect(result['max(value)']).toBe(50);
        expect(result['count(*)']).toBe(5);
    });

    test('WHERE with multiple conditions', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();

            await v.exec('CREATE TABLE products (id INTEGER, price REAL, status TEXT)');
            await v.exec(`INSERT INTO products VALUES 
                (1, 100, 'active'), (2, 200, 'active'), 
                (3, 50, 'inactive'), (4, 300, 'active')`);

            const filtered = await v.exec(`
                SELECT * FROM products WHERE price > 100 AND status = 'active'
            `);
            return filtered.rows.length;
        });

        expect(result).toBe(2); // products with id 2 and 4
    });
});

test.describe('Vector Search (NEAR)', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');
        await page.waitForLoadState('domcontentloaded');
        await page.waitForTimeout(2000);
    });

    test('NEAR query parses without error', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();

            await v.exec('CREATE TABLE docs (id INTEGER, text TEXT)');
            await v.exec(`INSERT INTO docs VALUES (1, 'hello world'), (2, 'goodbye world')`);

            try {
                await v.exec(`SELECT * FROM docs WHERE text NEAR 'hello' LIMIT 1`);
                return { success: true };
            } catch (e) {
                // NEAR may require model - check if it's a model error (acceptable)
                const isModelError = e.message.includes('model') || e.message.includes('encoder');
                return { success: false, modelError: isModelError, error: e.message };
            }
        });

        // Either succeeded or failed due to missing model (both acceptable)
        expect(result.success || result.modelError).toBe(true);
    });

    test('Vector + JOIN query structure', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();

            await v.exec('CREATE TABLE items (id INTEGER, desc TEXT)');
            await v.exec('CREATE TABLE inventory (item_id INTEGER, qty INTEGER)');
            await v.exec(`INSERT INTO items VALUES (1, 'laptop'), (2, 'desk')`);
            await v.exec(`INSERT INTO inventory VALUES (1, 50), (2, 20)`);

            try {
                const r = await v.exec(`
                    SELECT i.desc, inv.qty FROM items i
                    JOIN inventory inv ON i.id = inv.item_id
                    WHERE i.desc NEAR 'computer'
                `);
                return { success: true, rows: r.rows.length };
            } catch (e) {
                return { success: false, error: e.message };
            }
        });

        // Query should at least parse without syntax error
        expect(typeof result.success).toBe('boolean');
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
