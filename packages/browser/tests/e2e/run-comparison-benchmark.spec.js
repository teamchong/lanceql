/**
 * LanceQL vs DuckDB WASM Comparison Benchmark
 * Runs entirely in CI without HTML pages
 */

import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

function saveResults(rowCount, results) {
    if (process.env.CI) {
        try {
            const resultsPath = path.join(process.cwd(), 'benchmark-results.json');
            let allResults = [];
            if (fs.existsSync(resultsPath)) {
                allResults = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
            }
            allResults.push({ 
                rowCount, 
                timestamp: new Date().toISOString(), 
                commit: process.env.GITHUB_SHA || 'local',
                results 
            });
            fs.writeFileSync(resultsPath, JSON.stringify(allResults, null, 2));
            console.log(`Saved benchmark results to ${resultsPath}`);
        } catch (e) {
            console.error('Failed to save benchmark results:', e);
        }
    }
}

async function runBenchmark(page, rowCount) {
    // Use minimal HTML page just to load JS
    await page.goto('/examples/wasm/test-vault-sql.html');

    const results = await page.evaluate(async (rows) => {
        const { vault } = await import('./lanceql.js');
        const lanceVault = await vault();

        // Load DuckDB WASM
        const duckdb = await import('https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.28.0/+esm');
        const JSDELIVR_BUNDLES = {
            mvp: { mainModule: 'https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.28.0/dist/duckdb-mvp.wasm', mainWorker: 'https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.28.0/dist/duckdb-browser-mvp.worker.js' },
            eh: { mainModule: 'https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.28.0/dist/duckdb-eh.wasm', mainWorker: 'https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.28.0/dist/duckdb-browser-eh.worker.js' }
        };
        const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);
        const worker = new Worker(bundle.mainWorker);
        const logger = new duckdb.ConsoleLogger();
        const db = new duckdb.AsyncDuckDB(logger, worker);
        await db.instantiate(bundle.mainModule);
        const duckConn = await db.connect();

        const numCustomers = Math.min(Math.floor(rows / 10), 100);
        const benchResults = [];

        // Setup tables
        await lanceVault.exec('DROP TABLE IF EXISTS bench_orders');
        await lanceVault.exec('DROP TABLE IF EXISTS bench_customers');
        await lanceVault.exec('CREATE TABLE bench_orders (id INTEGER, customer_id INTEGER, amount REAL, status TEXT)');
        await lanceVault.exec('CREATE TABLE bench_customers (id INTEGER, name TEXT)');

        const statuses = ['pending', 'shipped', 'delivered', 'cancelled'];
        for (let i = 0; i < rows; i++) {
            await lanceVault.exec(`INSERT INTO bench_orders VALUES (${i}, ${i % numCustomers}, ${(Math.random() * 1000).toFixed(2)}, '${statuses[i % 4]}')`);
        }
        for (let i = 0; i < numCustomers; i++) {
            await lanceVault.exec(`INSERT INTO bench_customers VALUES (${i}, 'Customer ${i}')`);
        }

        // DuckDB: Use batch INSERT with explicit columns to ensure proper column binding
        await duckConn.query(`CREATE TABLE bench_orders (id INTEGER, customer_id INTEGER, amount DOUBLE, status VARCHAR)`);
        // Insert in batches of 1000 to avoid query size limits
        const batchSize = 1000;
        for (let batch = 0; batch < rows; batch += batchSize) {
            const end = Math.min(batch + batchSize, rows);
            const values = [];
            for (let i = batch; i < end; i++) {
                values.push(`(${i}, ${i % numCustomers}, ${(Math.random() * 1000).toFixed(2)}, '${statuses[i % 4]}')`);
            }
            await duckConn.query(`INSERT INTO bench_orders (id, customer_id, amount, status) VALUES ${values.join(',')}`);
        }
        await duckConn.query(`CREATE TABLE bench_customers (id INTEGER, name VARCHAR)`);
        const customerValues = Array.from({length: numCustomers}, (_, i) => `(${i}, 'Customer ${i}')`).join(',');
        await duckConn.query(`INSERT INTO bench_customers (id, name) VALUES ${customerValues}`);

        async function benchmark(name, lanceSql, duckSql) {
            // Warmup
            for (let i = 0; i < 2; i++) {
                await lanceVault.exec(lanceSql);
                await duckConn.query(duckSql);
            }

            // Run multiple iterations and measure total time
            const iterations = 10;

            // LanceQL
            let lanceResult;
            const lanceStart = performance.now();
            for (let i = 0; i < iterations; i++) {
                lanceResult = await lanceVault.exec(lanceSql);
            }
            const lanceMs = (performance.now() - lanceStart) / iterations;

            // DuckDB
            let duckResult;
            const duckStart = performance.now();
            for (let i = 0; i < iterations; i++) {
                duckResult = await duckConn.query(duckSql);
            }
            const duckMs = (performance.now() - duckStart) / iterations;

            const ratio = duckMs > 0 ? lanceMs / duckMs : (lanceMs > 0 ? Infinity : 1);
            const winner = ratio < 1 ? 'LanceQL' : 'DuckDB';

            return { query: name, rows, lanceql: lanceMs.toFixed(2), duckdb: duckMs.toFixed(2), ratio: ratio.toFixed(2) + 'x', winner };
        }

        // Verify data was inserted correctly
        const lanceSelectResult = await lanceVault.exec('SELECT COUNT(*) FROM bench_orders');
        const lanceSelectRows = lanceSelectResult?.rows ?? lanceSelectResult ?? [];
        const lanceInsertCount = lanceSelectRows[0]?.['count(*)'] ?? lanceSelectRows[0]?.['COUNT(*)'] ?? lanceSelectRows[0]?.[0] ?? 0;

        if (Number(lanceInsertCount) !== rows) {
            throw new Error(`LanceQL INSERT verification failed: expected ${rows} rows, got ${lanceInsertCount}`);
        }

        // Verify aggregation works correctly before benchmarking
        const lanceAggResult = await lanceVault.exec('SELECT SUM(amount), AVG(amount), COUNT(*) FROM bench_orders');
        const duckAggResult = await duckConn.query('SELECT SUM(amount), AVG(amount), COUNT(*) FROM bench_orders');

        // Extract values for verification
        const lanceAggRows = lanceAggResult?.rows ?? lanceAggResult ?? [];
        const lanceRow = lanceAggRows[0];
        // Try various key formats: count(*), COUNT(*), or positional
        const lanceCount = lanceRow?.['count(*)'] ?? lanceRow?.['COUNT(*)'] ?? lanceRow?.[2] ?? 0;
        const duckAggArray = duckAggResult.toArray();
        const duckRow = duckAggArray[0];
        const duckCount = duckRow?.['count_star()'] ?? duckRow?.['COUNT(*)'] ?? duckRow?.[2] ?? 0;

        if (Number(lanceCount) !== rows) {
            throw new Error(`LanceQL COUNT(*) mismatch: expected ${rows}, got ${lanceCount}. Result: ${JSON.stringify(lanceRow)}`);
        }
        if (Number(duckCount) !== rows) {
            throw new Error(`DuckDB COUNT(*) mismatch: expected ${rows}, got ${duckCount}. Result: ${JSON.stringify(duckRow)}`);
        }

        benchResults.push(await benchmark('Simple SELECT', 'SELECT * FROM bench_orders', 'SELECT * FROM bench_orders'));
        benchResults.push(await benchmark('Aggregation', 'SELECT SUM(amount), AVG(amount), COUNT(*) FROM bench_orders', 'SELECT SUM(amount), AVG(amount), COUNT(*) FROM bench_orders'));
        benchResults.push(await benchmark('JOIN', 'SELECT o.id, c.name FROM bench_orders o JOIN bench_customers c ON o.customer_id = c.id', 'SELECT o.id, c.name FROM bench_orders o JOIN bench_customers c ON o.customer_id = c.id'));
        benchResults.push(await benchmark('Complex WHERE', "SELECT * FROM bench_orders WHERE amount > 500 AND status = 'shipped'", "SELECT * FROM bench_orders WHERE amount > 500 AND status = 'shipped'"));

        // Cleanup
        await lanceVault.exec('DROP TABLE bench_orders');
        await lanceVault.exec('DROP TABLE bench_customers');
        await duckConn.close();
        await db.terminate();
        worker.terminate();

        return benchResults;
    }, rowCount);

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
    const results = await runBenchmark(page, 1000);
    saveResults(1000, results);
    expect(results.length).toBeGreaterThan(0);
});

test('LanceQL vs DuckDB benchmark 10K', async ({ page }) => {
    test.setTimeout(300000);
    const results = await runBenchmark(page, 10000);
    saveResults(10000, results);
    expect(results.length).toBeGreaterThan(0);
});

test('LanceQL vs DuckDB benchmark 100K', async ({ page }) => {
    test.setTimeout(600000);
    const results = await runBenchmark(page, 100000);
    saveResults(100000, results);
    expect(results.length).toBeGreaterThan(0);
});
