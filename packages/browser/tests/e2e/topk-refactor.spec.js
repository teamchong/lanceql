/**
 * E2E tests for TOPK and NEAR syntax refactor
 */

import { test, expect } from '@playwright/test';

test.describe('TOPK and NEAR Syntax', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');
        await page.waitForFunction(() => typeof window.testInit === 'function', { timeout: 30000 });
        await page.evaluate(async () => {
            if (!window.vaultInstance) await window.testInit();
        });
        await page.waitForFunction(() => !!window.vaultInstance, { timeout: 60000 });
    });

    test('TOPK keyword and default of 20', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const v = window.vaultInstance;
            await v.exec('CREATE TABLE items (id INTEGER)');
            // Insert 25 items
            const vals = Array.from({ length: 25 }, (_, i) => `(${i})`).join(',');
            await v.exec(`INSERT INTO items VALUES ${vals}`);

            // Test explicit TOPK
            const res3 = await v.exec('SELECT * FROM items TOPK 3');
            // Test default TOPK (should be 20)
            const resDefault = await v.exec('SELECT * FROM items');

            return {
                topk3: res3.rows.length,
                topkDefault: resDefault.rows.length
            };
        });
        expect(result.topk3).toBe(3);
        expect(result.topkDefault).toBe(20);
    });

    test('JOIN ON col NEAR val TOPK n', async ({ page }) => {
        page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
        page.on('pageerror', err => console.log('BROWSER ERROR:', err.message));

        const result = await page.evaluate(async () => {
            const v = window.vaultInstance;
            await v.exec('CREATE TABLE base (id INTEGER)');
            await v.exec('INSERT INTO base VALUES (1), (2)');

            await v.exec('CREATE TABLE target (id INTEGER, vec FLOAT32[3])');
            await v.exec(`INSERT INTO target (id, vec) VALUES (10, [1,0,0]), (11, [0,1,0]), (12, [0,0,1])`);

            // Join with TOPK 2 nearest on target
            const res = await v.exec(`
                SELECT b.id, t.id FROM base b
                JOIN target t ON t.vec NEAR [1, 0.1, 0.1] TOPK 2
            `);
            return {
                rowCount: res.rows.length,
                rows: res.rows
            };
        });
        expect(result.rowCount).toBe(4);
    });

    test('GROUP BY col TOPK n', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const v = window.vaultInstance;
            await v.exec('CREATE TABLE groups (cat INTEGER)');
            await v.exec('INSERT INTO groups VALUES (1), (1), (2), (2), (3), (3), (4), (4)');

            const res = await v.exec('SELECT cat, COUNT(*) FROM groups GROUP BY cat TOPK 2');
            return {
                groupCount: res.rows.length
            };
        });
        expect(result.groupCount).toBe(2);
    });

    test('JOIN ON col NEAR col with CREATE VECTOR INDEX', async ({ page }) => {
        // Higher timeout since MiniLM embedding takes time
        test.setTimeout(180000);

        page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
        page.on('pageerror', err => console.log('BROWSER ERROR:', err.message));

        const result = await page.evaluate(async () => {
            const v = window.vaultInstance;
            const logs = [];

            try {
                // Create images table with text captions
                logs.push('Creating images table');
                await v.exec('CREATE TABLE images (id INTEGER, caption TEXT)');
                await v.exec(`INSERT INTO images VALUES
                    (1, 'a cute cat playing'),
                    (2, 'a happy dog running'),
                    (3, 'a colorful bird flying')`);

                // Create emoji table with text descriptions
                logs.push('Creating emoji table');
                await v.exec('CREATE TABLE emoji (id INTEGER, emoji TEXT, description TEXT)');
                await v.exec(`INSERT INTO emoji VALUES
                    (1, 'cat', 'a fluffy cat'),
                    (2, 'dog', 'a loyal dog'),
                    (3, 'bird', 'a flying bird')`);

                // Test simple equi-join first (before vector indexes)
                logs.push('Testing simple JOIN before vector index');
                const simpleJoin1 = await v.exec(`
                    SELECT i.id, e.emoji
                    FROM images i
                    JOIN emoji e ON i.id = e.id
                `);
                logs.push(`Simple JOIN before vector index returned ${simpleJoin1.rows.length} rows`);

                // Load MiniLM model
                logs.push('Loading MiniLM model');
                await v.loadMinilmModel();
                logs.push('MiniLM model loaded');

                // Create vector index on images.caption using minilm
                logs.push('Creating vector index on images.caption');
                await v.exec('CREATE VECTOR INDEX ON images(caption) USING minilm');
                logs.push('Vector index on images created');

                // Create vector index on emoji.description using minilm
                logs.push('Creating vector index on emoji.description');
                await v.exec('CREATE VECTOR INDEX ON emoji(description) USING minilm');
                logs.push('Vector index on emoji created');

                // Test simple equi-join again (after vector indexes)
                logs.push('Testing simple JOIN after vector index');
                const simpleJoin2 = await v.exec(`
                    SELECT i.id, e.emoji
                    FROM images i
                    JOIN emoji e ON i.id = e.id
                `);
                logs.push(`Simple JOIN after vector index returned ${simpleJoin2.rows.length} rows`);

                // Now execute NEAR JOIN
                logs.push('Executing NEAR JOIN');
                const res = await v.exec(`
                    SELECT i.id, i.caption, e.emoji, e.description
                    FROM images i
                    JOIN emoji e ON i.caption NEAR e.description TOPK 2
                `);
                logs.push(`NEAR JOIN returned ${res.rows.length} rows`);

                return {
                    success: true,
                    rowCount: res.rows.length,
                    rows: res.rows,
                    logs
                };
            } catch (e) {
                logs.push(`Error: ${e.message}`);
                return {
                    success: false,
                    error: e.message,
                    logs
                };
            }
        });

        console.log('Test logs:', result.logs);
        if (!result.success) {
            console.log('Test failed with error:', result.error);
        }
        expect(result.success).toBe(true);
        expect(result.rowCount).toBeGreaterThan(0);
    });
});
