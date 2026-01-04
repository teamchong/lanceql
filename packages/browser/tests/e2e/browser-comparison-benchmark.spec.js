/**
 * Browser Database Comparison Benchmarks
 * LanceQL vs sql.js vs DuckDB WASM
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

    test('LanceQL vs DuckDB WASM (1K rows)', async ({ page }) => {
        test.setTimeout(180000);

        await page.goto('/examples/wasm/test-benchmark.html');

        // Set row count
        await page.selectOption('#rowCount', '1000');

        // Run benchmarks
        await page.click('#runBtn');

        // Wait for completion or error
        await page.waitForFunction(() => {
            const status = document.getElementById('status');
            return status && (status.textContent.includes('complete') || status.textContent.includes('Error'));
        }, { timeout: 120000 });

        const results = await page.evaluate(() => {
            const rows = document.querySelectorAll('#resultsBody tr');
            return Array.from(rows).map(row => {
                const cells = row.querySelectorAll('td');
                return {
                    query: cells[0]?.textContent || '',
                    rows: cells[1]?.textContent || '',
                    lanceql: cells[2]?.textContent || '',
                    duckdb: cells[3]?.textContent || '',
                    ratio: cells[4]?.textContent || '',
                    winner: cells[5]?.textContent || ''
                };
            });
        });

        console.log('\n╔══════════════════════════════════════════════════════════════════════╗');
        console.log('║              LanceQL vs DuckDB WASM Benchmark Results                ║');
        console.log('╠══════════════════╦═══════╦════════════╦════════════╦═══════╦═════════╣');
        console.log('║ Query            ║ Rows  ║ LanceQL    ║ DuckDB     ║ Ratio ║ Winner  ║');
        console.log('╠══════════════════╬═══════╬════════════╬════════════╬═══════╬═════════╣');
        for (const r of results) {
            const q = r.query.padEnd(16);
            const rows = r.rows.padStart(5);
            const lq = (r.lanceql + 'ms').padStart(10);
            const dd = (r.duckdb + 'ms').padStart(10);
            const ratio = r.ratio.padStart(5);
            const w = r.winner.padEnd(7);
            console.log(`║ ${q} ║ ${rows} ║ ${lq} ║ ${dd} ║ ${ratio} ║ ${w} ║`);
        }
        console.log('╚══════════════════╩═══════╩════════════╩════════════╩═══════╩═════════╝');

        expect(results.length).toBeGreaterThan(0);
    });

    test('LanceQL vs DuckDB WASM (10K rows)', async ({ page }) => {
        test.setTimeout(300000);

        await page.goto('/examples/wasm/test-benchmark.html');

        // Set row count to 10K
        await page.selectOption('#rowCount', '10000');

        // Run benchmarks
        await page.click('#runBtn');

        // Wait for completion or error
        await page.waitForFunction(() => {
            const status = document.getElementById('status');
            return status && (status.textContent.includes('complete') || status.textContent.includes('Error'));
        }, { timeout: 240000 });

        const results = await page.evaluate(() => {
            const rows = document.querySelectorAll('#resultsBody tr');
            return Array.from(rows).map(row => {
                const cells = row.querySelectorAll('td');
                return {
                    query: cells[0]?.textContent || '',
                    rows: cells[1]?.textContent || '',
                    lanceql: cells[2]?.textContent || '',
                    duckdb: cells[3]?.textContent || '',
                    ratio: cells[4]?.textContent || '',
                    winner: cells[5]?.textContent || ''
                };
            });
        });

        console.log('\n╔══════════════════════════════════════════════════════════════════════╗');
        console.log('║        LanceQL vs DuckDB WASM Benchmark Results (10K rows)          ║');
        console.log('╠══════════════════╦═══════╦════════════╦════════════╦═══════╦═════════╣');
        console.log('║ Query            ║ Rows  ║ LanceQL    ║ DuckDB     ║ Ratio ║ Winner  ║');
        console.log('╠══════════════════╬═══════╬════════════╬════════════╬═══════╬═════════╣');
        for (const r of results) {
            const q = r.query.padEnd(16);
            const rows = r.rows.padStart(5);
            const lq = (r.lanceql + 'ms').padStart(10);
            const dd = (r.duckdb + 'ms').padStart(10);
            const ratio = r.ratio.padStart(5);
            const w = r.winner.padEnd(7);
            console.log(`║ ${q} ║ ${rows} ║ ${lq} ║ ${dd} ║ ${ratio} ║ ${w} ║`);
        }
        console.log('╚══════════════════╩═══════╩════════════╩════════════╩═══════╩═════════╝');

        expect(results.length).toBeGreaterThan(0);
    });
});
