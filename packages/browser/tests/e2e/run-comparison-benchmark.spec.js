/**
 * LanceQL vs DuckDB WASM Comparison Benchmark
 */

import { test, expect } from '@playwright/test';

test('LanceQL vs DuckDB benchmark', async ({ page }) => {
    test.setTimeout(300000);

    await page.goto('/examples/wasm/test-benchmark.html');

    // Set 1000 rows
    await page.selectOption('#rowCount', '1000');

    // Run benchmarks
    await page.click('#runBtn');

    // Wait for completion
    await page.waitForFunction(() => {
        const status = document.getElementById('status');
        return status && status.textContent.includes('complete');
    }, { timeout: 180000 });

    // Get results
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

    console.log('\n=== LanceQL vs DuckDB WASM Benchmark (1000 rows) ===\n');
    console.log('| Query Type     | LanceQL (ms) | DuckDB (ms) | Ratio | Winner |');
    console.log('|----------------|--------------|-------------|-------|--------|');
    for (const r of results) {
        console.log(`| ${r.query.padEnd(14)} | ${r.lanceql.padStart(12)} | ${r.duckdb.padStart(11)} | ${r.ratio.padStart(5)} | ${r.winner.padEnd(6)} |`);
    }

    expect(results.length).toBeGreaterThan(0);
});
