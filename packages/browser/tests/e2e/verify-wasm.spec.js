import { test, expect } from '@playwright/test';

test('verify WASM SQL execution via Vault in Worker', async ({ page }) => {
    // Navigate to a page to load modules
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    await page.goto('/examples/wasm/test-vault-sql.html');

    const result = await page.evaluate(async () => {
        try {
            const { LocalDatabase } = await import('/packages/browser/dist/lanceql.esm.js');
            const db = new LocalDatabase('verify-wasm-db');
            await db.open();

            const tableName = 'test_table';
            await db.createTable(tableName, [
                { name: 'id', type: 'int64', primaryKey: true },
                { name: 'val', type: 'float64' }
            ], true);

            const rows = [];
            for (let i = 0; i < 200; i++) {
                rows.push({ id: BigInt(i), val: Math.random() * 100 });
            }
            await db.insert(tableName, rows);

            // Explicitly flush to ensure fragments are created (WASM file path)
            await db.flush();

            const sql = 'SELECT SUM(val) as total FROM test_table WHERE id > 50';
            const result = await db.exec(sql);
            return result;
        } catch (e) {
            return { error: e.stack || e.toString() };
        }
    });

    console.log('Verification Result:', JSON.stringify(result, null, 2));

    expect(result).toBeTruthy();
    expect(result.error).toBeUndefined();

    // Robust row count check
    const rowCount = result.rowCount !== undefined ? result.rowCount : (result.rows ? result.rows.length : 0);
    expect(rowCount).toBeGreaterThan(0);

    // Robust data check
    const total = (result.data && result.data.total) !== undefined ?
        (ArrayBuffer.isView(result.data.total) ? result.data.total[0] : result.data.total) :
        (result.rows && result.rows[0] ? result.rows[0].total : undefined);
    expect(total).toBeDefined();
});
