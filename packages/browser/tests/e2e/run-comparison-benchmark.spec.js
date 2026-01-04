/**
 * LanceQL vs DuckDB WASM Comparison Benchmark
 */

import { test, expect } from '@playwright/test';

async function runBenchmark(page, rowCount) {
    await page.goto('/examples/wasm/test-benchmark.html');

    // Set row count
    await page.selectOption('#rowCount', rowCount);

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

    console.log(`\n=== LanceQL vs DuckDB WASM Benchmark (${rowCount} rows) ===\n`);
    console.log('| Query Type     | LanceQL (ms) | DuckDB (ms) | Ratio | Winner |');
    console.log('|----------------|--------------|-------------|-------|--------|');
    for (const r of results) {
        console.log(`| ${r.query.padEnd(14)} | ${r.lanceql.padStart(12)} | ${r.duckdb.padStart(11)} | ${r.ratio.padStart(5)} | ${r.winner.padEnd(6)} |`);
    }

    return results;
}

test('LanceQL vs DuckDB benchmark 1K', async ({ page }) => {
    test.setTimeout(300000);
    const results = await runBenchmark(page, '1000');
    expect(results.length).toBeGreaterThan(0);
});

test('LanceQL vs DuckDB benchmark 10K', async ({ page }) => {
    test.setTimeout(300000);
    const results = await runBenchmark(page, '10000');
    expect(results.length).toBeGreaterThan(0);
});

test('LanceQL vs DuckDB numeric only 10K', async ({ page }) => {
    test.setTimeout(300000);
    await page.goto('/examples/wasm/test-benchmark.html');

    // Inject a numeric-only test
    const result = await page.evaluate(async () => {
        const { vault } = await import('./lanceql.js');
        const lanceVault = await vault();

        // Setup numeric-only table
        await lanceVault.exec('DROP TABLE IF EXISTS bench_numeric');
        await lanceVault.exec('CREATE TABLE bench_numeric (id INTEGER, val1 REAL, val2 REAL, val3 REAL)');
        for (let i = 0; i < 10000; i++) {
            await lanceVault.exec(`INSERT INTO bench_numeric VALUES (${i}, ${Math.random()}, ${Math.random()}, ${Math.random()})`);
        }

        // Warmup
        for (let i = 0; i < 2; i++) await lanceVault.exec('SELECT * FROM bench_numeric');

        // Benchmark
        const times = [];
        for (let i = 0; i < 5; i++) {
            const start = performance.now();
            await lanceVault.exec('SELECT * FROM bench_numeric');
            times.push(performance.now() - start);
        }
        times.sort((a, b) => a - b);
        return times[2]; // median
    });

    console.log(`\n=== Numeric-only SELECT * (10K rows): ${result.toFixed(2)}ms ===\n`);
    expect(result).toBeLessThan(10);
});
