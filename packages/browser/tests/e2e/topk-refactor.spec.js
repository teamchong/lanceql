/**
 * E2E tests for TOPK and NEAR syntax refactor
 */

import { test, expect } from '@playwright/test';

test.describe('TOPK and NEAR Syntax', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');
        await page.waitForFunction(() => typeof window.testInit === 'function');
        await page.evaluate(async () => {
            if (!window.vaultInstance) await window.testInit();
        });
        await page.waitForFunction(() => !!window.vaultInstance);
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
});
