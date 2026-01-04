/**
 * Browser Database Comparison Benchmarks
 * LanceQL vs sql.js
 *
 * Note: LanceQL vs DuckDB WASM benchmarks are in run-comparison-benchmark.spec.js
 * which runs directly in CI without HTML pages.
 */

import { test, expect } from '@playwright/test';

test.describe('Browser Database Benchmarks', () => {

    test('LanceQL vs sql.js', async ({ page }) => {
        test.setTimeout(120000);

        // Collect console logs
        const logs = [];
        page.on('console', msg => logs.push(`${msg.type()}: ${msg.text()}`));

        // Navigate to sql.js comparison page
        await page.goto('/examples/sqljs-compat/index.html');
        await page.waitForLoadState('networkidle');

        // Run benchmarks
        await page.click('button:has-text("Run Benchmarks")');

        // Wait for benchmark to complete - look for "Benchmark complete" text in output
        await page.waitForFunction(() => {
            const sqljsOutput = document.getElementById('sqljs-output');
            const lanceOutput = document.getElementById('lanceql-output');
            return (sqljsOutput?.textContent?.includes('Benchmark complete') ||
                    lanceOutput?.textContent?.includes('Benchmark complete') ||
                    sqljsOutput?.textContent?.includes('Error'));
        }, { timeout: 90000 });

        const results = await page.evaluate(() => {
            const rows = document.querySelectorAll('#benchmark-table tbody tr');
            return Array.from(rows).map(row => {
                const cells = row.querySelectorAll('td');
                return {
                    operation: cells[0]?.textContent || '',
                    sqljs: cells[1]?.textContent || '',
                    lanceql: cells[2]?.textContent || '',
                    winner: cells[3]?.textContent || ''
                };
            });
        });

        console.log('\n╔══════════════════════════════════════════════════════════════╗');
        console.log('║           LanceQL vs sql.js Benchmark Results                ║');
        console.log('╠════════════════════════╦════════════╦════════════╦═══════════╣');
        console.log('║ Operation              ║ sql.js     ║ LanceQL    ║ Winner    ║');
        console.log('╠════════════════════════╬════════════╬════════════╬═══════════╣');
        for (const r of results) {
            const op = r.operation.padEnd(22);
            const sj = r.sqljs.padStart(10);
            const lq = r.lanceql.padStart(10);
            const w = r.winner.padEnd(9);
            console.log(`║ ${op} ║ ${sj} ║ ${lq} ║ ${w} ║`);
        }
        console.log('╚════════════════════════╩════════════╩════════════╩═══════════╝');

        expect(results.length).toBeGreaterThan(0);
    });
});
