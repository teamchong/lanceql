
import { test, expect } from '@playwright/test';

test('verify WASM 1:N Join execution', async ({ page }) => {
    // Navigate to local test page to ensure modules are loadable
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    await page.goto('/examples/wasm/test-vault-sql.html');

    const result = await page.evaluate(async () => {
        try {
            const { LocalDatabase } = await import('/packages/browser/dist/lanceql.esm.js');
            const db = new LocalDatabase('verify-join-db');
            await db.open();

            await db.createTable('customers', [
                { name: 'id', type: 'int64', primaryKey: true },
                { name: 'name', type: 'string' }
            ], true);

            await db.createTable('orders', [
                { name: 'id', type: 'int64', primaryKey: true },
                { name: 'customer_id', type: 'int64' },
                { name: 'amount', type: 'float64' }
            ], true);

            // Insert data
            await db.insert('customers', [
                { id: 1n, name: 'Alice' },
                { id: 2n, name: 'Bob' }
            ]);

            const dummyCustomers = [];
            for (let i = 0; i < 200; i++) {
                dummyCustomers.push({ id: BigInt(100 + i), name: `Cust${i}` });
            }
            await db.insert('customers', dummyCustomers);

            await db.insert('orders', [
                { id: 100n, customer_id: 1n, amount: 10.0 },
                { id: 101n, customer_id: 1n, amount: 20.0 },
                { id: 102n, customer_id: 2n, amount: 30.0 }
            ]);

            // Add dummy data to force WASM (> 100 rows)
            const dummyRows = [];
            for (let i = 0; i < 200; i++) {
                dummyRows.push({ id: BigInt(200 + i), customer_id: 999n, amount: 0.0 });
            }
            await db.insert('orders', dummyRows);

            // Explicitly flush to ensure fragments are created
            await db.flush();

            // Execute Join
            const result = await db.exec(
                "SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id"
            );

            return result;
        } catch (e) {
            return { error: e.stack || e.toString() };
        }
    });

    console.log('Join Result:', JSON.stringify(result, null, 2));

    expect(result.error).toBeUndefined();

    const rows = result.rows || [];

    // Verify values using Amount (numeric types are fixed)
    const aliceOrders = rows.filter(r => r.amount === 10 || r.amount === 20);
    const bobOrders = rows.filter(r => r.amount === 30);

    expect(rows.length).toBe(3);
    expect(aliceOrders.length).toBe(2);
    expect(bobOrders.length).toBe(1);

    const amounts = aliceOrders.map(r => r.amount).sort();
    expect(amounts).toEqual([10, 20]);
});
