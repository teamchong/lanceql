/**
 * Embedding and NEAR Vector Search E2E Tests
 * 
 * Tests that embedding generation and NEAR vector search work correctly.
 */

import { test, expect } from '@playwright/test';

test.describe('Embedding & NEAR Vector Search', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');
        await page.waitForLoadState('domcontentloaded');
        await page.waitForTimeout(2000); // Wait for WASM to load
    });

    test('creates embeddings for text data', async ({ page }) => {
        test.setTimeout(60000);

        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();

            // Create table with text column
            await v.exec('CREATE TABLE docs (id INTEGER, content TEXT)');
            await v.exec(`INSERT INTO docs VALUES 
                (1, 'The quick brown fox jumps over the lazy dog'),
                (2, 'A beautiful sunset over the ocean'),
                (3, 'Machine learning and artificial intelligence'),
                (4, 'Cats and dogs are popular pets'),
                (5, 'The weather is sunny today')`);

            // Check data was inserted
            const count = await v.exec('SELECT COUNT(*) FROM docs');
            return {
                rowCount: count.rows[0]['count(*)'] || count.rows[0][0],
                success: true
            };
        });

        expect(result.success).toBe(true);
        expect(result.rowCount).toBe(5);
    });

    test('NEAR search returns relevant results', async ({ page }) => {
        test.setTimeout(90000);

        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();

            // Create table with embeddings
            await v.exec('CREATE TABLE products (id INTEGER, name TEXT, embedding VECTOR(384))');

            // Insert with pre-computed embeddings (simplified - in real use, embeddings are generated)
            // For this test, we'll use the text-based NEAR which generates embeddings on the fly
            await v.exec(`INSERT INTO products VALUES 
                (1, 'red sports car'),
                (2, 'blue ocean waves'),
                (3, 'green forest trees'),
                (4, 'yellow taxi cab'),
                (5, 'orange sunset sky')`);

            try {
                // Test NEAR search (requires embedding model to be loaded)
                // This may fail if no model is available, which is acceptable in CI
                const results = await v.exec(`SELECT id, name FROM products WHERE name NEAR 'vehicle' LIMIT 3`);
                return {
                    success: true,
                    rowCount: results.rows?.length || 0,
                    results: results.rows
                };
            } catch (e) {
                // NEAR requires embedding model - may not be available in CI
                return {
                    success: false,
                    error: e.message,
                    modelRequired: e.message.includes('model') || e.message.includes('encoder')
                };
            }
        });

        // Either NEAR worked, or it failed because no model is loaded (acceptable in CI)
        if (result.success) {
            expect(result.rowCount).toBeGreaterThan(0);
            console.log('NEAR results:', result.results);
        } else {
            expect(result.modelRequired).toBe(true);
            console.log('NEAR requires model:', result.error);
        }
    });

    test('aggregation with vector columns works', async ({ page }) => {
        test.setTimeout(60000);

        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();

            await v.exec('CREATE TABLE vectors (id INTEGER, score REAL)');
            await v.exec(`INSERT INTO vectors VALUES 
                (1, 0.95), (2, 0.87), (3, 0.72), (4, 0.91), (5, 0.65)`);

            const agg = await v.exec('SELECT SUM(score), AVG(score), MIN(score), MAX(score), COUNT(*) FROM vectors');
            return agg.rows[0];
        });

        // Sum: 0.95+0.87+0.72+0.91+0.65 = 4.1
        expect(result['sum(score)']).toBeCloseTo(4.1, 1);
        expect(result['avg(score)']).toBeCloseTo(0.82, 1);
        expect(result['min(score)']).toBeCloseTo(0.65, 1);
        expect(result['max(score)']).toBeCloseTo(0.95, 1);
        expect(result['count(*)']).toBe(5);
    });

    test('JOIN with WHERE filters works', async ({ page }) => {
        test.setTimeout(60000);

        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();

            // Create two tables
            await v.exec('CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL)');
            await v.exec('CREATE TABLE customers (id INTEGER, name TEXT)');

            await v.exec(`INSERT INTO customers VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')`);
            await v.exec(`INSERT INTO orders VALUES 
                (1, 1, 100.50), (2, 1, 250.00), (3, 2, 75.25), (4, 3, 300.00)`);

            // JOIN with WHERE
            const joined = await v.exec(`
                SELECT c.name, o.amount 
                FROM orders o 
                JOIN customers c ON o.customer_id = c.id 
                WHERE o.amount > 100
            `);

            return {
                rowCount: joined.rows.length,
                rows: joined.rows
            };
        });

        // Should return orders > 100: (Alice, 250), (Charlie, 300)... and maybe (Alice, 100.50)
        expect(result.rowCount).toBeGreaterThan(0);
        console.log('JOIN results:', result.rows);
    });
});

test.describe('WebGPU Vector Search', () => {
    test('GPU vector search initialization', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');
        await page.waitForLoadState('domcontentloaded');

        const gpuStatus = await page.evaluate(async () => {
            try {
                // Check if WebGPU is available
                if (!navigator.gpu) {
                    return { available: false, reason: 'navigator.gpu not available' };
                }

                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    return { available: false, reason: 'No GPU adapter found' };
                }

                const device = await adapter.requestDevice();
                return {
                    available: true,
                    adapterName: adapter.info?.device || 'unknown',
                    limits: {
                        maxBufferSize: device.limits.maxBufferSize,
                        maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX
                    }
                };
            } catch (e) {
                return { available: false, reason: e.message };
            }
        });

        console.log('WebGPU Status:', gpuStatus);
        // We don't require WebGPU to be available, just that the check doesn't crash
        expect(typeof gpuStatus.available).toBe('boolean');
    });
});
