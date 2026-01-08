// @ts-check
import { test, expect } from '@playwright/test';

/**
 * LanceQL SQL Performance Benchmarks
 *
 * These tests measure LanceQL SQL query performance across different query types
 * and data sizes. For DuckDB comparison, use the interactive test-benchmark.html
 * page served with proper CORS headers.
 */
test.describe('LanceQL SQL Performance Benchmarks', () => {
    // Run benchmarks sequentially
    test.describe.configure({ mode: 'serial' });
    test.setTimeout(60000);

    const WARMUP_RUNS = 2;
    const BENCHMARK_RUNS = 5;

    test.beforeEach(async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');
        await page.waitForLoadState('domcontentloaded');
    });

    test('Cold start time', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const start = performance.now();
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const time = performance.now() - start;
            return { time, success: true };
        });

        console.log(`\n=== Cold Start ===`);
        console.log(`LanceQL: ${result.time.toFixed(2)}ms`);

        expect(result.success).toBe(true);
        expect(result.time).toBeLessThan(5000); // Should init under 5s
    });

    for (const rowCount of [100, 1000, 5000]) {
        test(`Simple SELECT performance (${rowCount} rows)`, async ({ page }) => {
            const result = await page.evaluate(async ({ rows, warmup, runs }) => {
                const { vault } = await import('./lanceql.js');
                const v = await vault();

                // Setup - use batch INSERT (10 rows per statement) for better memory efficiency
                await v.exec('DROP TABLE IF EXISTS perf_select');
                await v.exec('CREATE TABLE perf_select (id INTEGER, name TEXT, value REAL)');
                const batchSize = 10;
                for (let i = 0; i < rows; i += batchSize) {
                    const values = [];
                    for (let j = i; j < Math.min(i + batchSize, rows); j++) {
                        values.push(`(${j}, 'Name${j}', ${Math.random() * 1000})`);
                    }
                    await v.exec(`INSERT INTO perf_select VALUES ${values.join(',')}`);
                }

                // Warmup
                for (let i = 0; i < warmup; i++) {
                    await v.exec('SELECT * FROM perf_select');
                }

                // Benchmark
                const times = [];
                for (let i = 0; i < runs; i++) {
                    const start = performance.now();
                    const res = await v.exec('SELECT * FROM perf_select');
                    times.push(performance.now() - start);
                }

                // Cleanup
                await v.exec('DROP TABLE perf_select');

                const sorted = times.sort((a, b) => a - b);
                const median = sorted[Math.floor(sorted.length / 2)];
                const p95 = sorted[Math.floor(sorted.length * 0.95)];

                return { median, p95, min: sorted[0], max: sorted[sorted.length - 1] };
            }, { rows: rowCount, warmup: WARMUP_RUNS, runs: BENCHMARK_RUNS });

            console.log(`\n=== Simple SELECT (${rowCount} rows) ===`);
            console.log(`Median: ${result.median.toFixed(2)}ms`);
            console.log(`P95:    ${result.p95.toFixed(2)}ms`);
            console.log(`Range:  ${result.min.toFixed(2)}-${result.max.toFixed(2)}ms`);

            expect(result.median).toBeGreaterThanOrEqual(0);
        });

        test(`Aggregation performance (${rowCount} rows)`, async ({ page }) => {
            const result = await page.evaluate(async ({ rows, warmup, runs }) => {
                const { vault } = await import('./lanceql.js');
                const v = await vault();

                // Setup - use batch INSERT for better memory efficiency
                await v.exec('DROP TABLE IF EXISTS perf_agg');
                await v.exec('CREATE TABLE perf_agg (id INTEGER, category TEXT, amount REAL)');
                const categories = ['A', 'B', 'C', 'D', 'E'];
                const batchSize = 10;
                for (let i = 0; i < rows; i += batchSize) {
                    const values = [];
                    for (let j = i; j < Math.min(i + batchSize, rows); j++) {
                        const cat = categories[j % categories.length];
                        values.push(`(${j}, '${cat}', ${Math.random() * 1000})`);
                    }
                    await v.exec(`INSERT INTO perf_agg VALUES ${values.join(',')}`);
                }

                const query = 'SELECT category, COUNT(*), SUM(amount), AVG(amount) FROM perf_agg GROUP BY category';

                // Warmup
                for (let i = 0; i < warmup; i++) {
                    await v.exec(query);
                }

                // Benchmark
                const times = [];
                for (let i = 0; i < runs; i++) {
                    const start = performance.now();
                    await v.exec(query);
                    times.push(performance.now() - start);
                }

                // Cleanup
                await v.exec('DROP TABLE perf_agg');

                const sorted = times.sort((a, b) => a - b);
                const median = sorted[Math.floor(sorted.length / 2)];
                const p95 = sorted[Math.floor(sorted.length * 0.95)];

                return { median, p95, min: sorted[0], max: sorted[sorted.length - 1] };
            }, { rows: rowCount, warmup: WARMUP_RUNS, runs: BENCHMARK_RUNS });

            console.log(`\n=== Aggregation (${rowCount} rows) ===`);
            console.log(`Median: ${result.median.toFixed(2)}ms`);
            console.log(`P95:    ${result.p95.toFixed(2)}ms`);
            console.log(`Range:  ${result.min.toFixed(2)}-${result.max.toFixed(2)}ms`);

            // Fast aggregations may measure as 0ms due to timer resolution
            expect(result.median).toBeGreaterThanOrEqual(0);
        });

        test(`JOIN performance (${rowCount} rows)`, async ({ page }) => {
            const result = await page.evaluate(async ({ rows, warmup, runs }) => {
                const { vault } = await import('./lanceql.js');
                const v = await vault();

                // Setup - use batch INSERT for better memory efficiency
                const numCustomers = Math.min(Math.floor(rows / 10), 100);
                await v.exec('DROP TABLE IF EXISTS perf_orders');
                await v.exec('DROP TABLE IF EXISTS perf_customers');
                await v.exec('CREATE TABLE perf_customers (id INTEGER, name TEXT)');
                await v.exec('CREATE TABLE perf_orders (id INTEGER, customer_id INTEGER, amount REAL)');

                // Batch insert customers
                const batchSize = 10;
                for (let i = 0; i < numCustomers; i += batchSize) {
                    const values = [];
                    for (let j = i; j < Math.min(i + batchSize, numCustomers); j++) {
                        values.push(`(${j}, 'Customer${j}')`);
                    }
                    await v.exec(`INSERT INTO perf_customers VALUES ${values.join(',')}`);
                }
                // Batch insert orders
                for (let i = 0; i < rows; i += batchSize) {
                    const values = [];
                    for (let j = i; j < Math.min(i + batchSize, rows); j++) {
                        const customerId = Math.floor(Math.random() * numCustomers);
                        values.push(`(${j}, ${customerId}, ${Math.random() * 1000})`);
                    }
                    await v.exec(`INSERT INTO perf_orders VALUES ${values.join(',')}`);
                }

                const query = 'SELECT c.name, SUM(o.amount) FROM perf_orders o JOIN perf_customers c ON o.customer_id = c.id GROUP BY c.name';

                // Warmup
                for (let i = 0; i < warmup; i++) {
                    await v.exec(query);
                }

                // Benchmark
                const times = [];
                for (let i = 0; i < runs; i++) {
                    const start = performance.now();
                    await v.exec(query);
                    times.push(performance.now() - start);
                }

                // Cleanup
                await v.exec('DROP TABLE perf_orders');
                await v.exec('DROP TABLE perf_customers');

                const sorted = times.sort((a, b) => a - b);
                const median = sorted[Math.floor(sorted.length / 2)];
                const p95 = sorted[Math.floor(sorted.length * 0.95)];

                return { median, p95, min: sorted[0], max: sorted[sorted.length - 1] };
            }, { rows: rowCount, warmup: WARMUP_RUNS, runs: BENCHMARK_RUNS });

            console.log(`\n=== JOIN (${rowCount} rows) ===`);
            console.log(`Median: ${result.median.toFixed(2)}ms`);
            console.log(`P95:    ${result.p95.toFixed(2)}ms`);
            console.log(`Range:  ${result.min.toFixed(2)}-${result.max.toFixed(2)}ms`);

            // Fast JOINs may measure as 0ms due to timer resolution
            expect(result.median).toBeGreaterThanOrEqual(0);
        });
    }

    test('Complex WHERE performance', async ({ page }) => {
        const result = await page.evaluate(async ({ warmup, runs }) => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const rows = 1000;

            // Setup - use batch INSERT for better memory efficiency
            const statuses = ['active', 'pending', 'inactive', 'archived'];
            await v.exec('DROP TABLE IF EXISTS perf_where');
            await v.exec('CREATE TABLE perf_where (id INTEGER, status TEXT, amount REAL, priority INTEGER)');

            const batchSize = 10;
            for (let i = 0; i < rows; i += batchSize) {
                const values = [];
                for (let j = i; j < Math.min(i + batchSize, rows); j++) {
                    const status = statuses[j % statuses.length];
                    const priority = (j % 5) + 1;
                    values.push(`(${j}, '${status}', ${Math.random() * 1000}, ${priority})`);
                }
                await v.exec(`INSERT INTO perf_where VALUES ${values.join(',')}`);
            }

            const query = `SELECT * FROM perf_where
                WHERE (status IN ('active', 'pending') AND amount > 500)
                OR (priority >= 4 AND amount BETWEEN 100 AND 300)`;

            // Warmup
            for (let i = 0; i < warmup; i++) {
                await v.exec(query);
            }

            // Benchmark
            const times = [];
            for (let i = 0; i < runs; i++) {
                const start = performance.now();
                await v.exec(query);
                times.push(performance.now() - start);
            }

            // Cleanup
            await v.exec('DROP TABLE perf_where');

            const sorted = times.sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)];
            const p95 = sorted[Math.floor(sorted.length * 0.95)];

            return { median, p95, min: sorted[0], max: sorted[sorted.length - 1] };
        }, { warmup: WARMUP_RUNS, runs: BENCHMARK_RUNS });

        console.log(`\n=== Complex WHERE (1000 rows) ===`);
        console.log(`Median: ${result.median.toFixed(2)}ms`);
        console.log(`P95:    ${result.p95.toFixed(2)}ms`);
        console.log(`Range:  ${result.min.toFixed(2)}-${result.max.toFixed(2)}ms`);

        expect(result.median).toBeGreaterThanOrEqual(0);
    });

    test('Window function performance', async ({ page }) => {
        const result = await page.evaluate(async ({ warmup, runs }) => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const rows = 500;

            // Setup - use batch INSERT for better memory efficiency
            await v.exec('DROP TABLE IF EXISTS perf_window');
            await v.exec('CREATE TABLE perf_window (id INTEGER, region TEXT, amount REAL)');
            const regions = ['North', 'South', 'East', 'West'];

            const batchSize = 10;
            for (let i = 0; i < rows; i += batchSize) {
                const values = [];
                for (let j = i; j < Math.min(i + batchSize, rows); j++) {
                    const region = regions[j % regions.length];
                    values.push(`(${j}, '${region}', ${Math.random() * 1000})`);
                }
                await v.exec(`INSERT INTO perf_window VALUES ${values.join(',')}`);
            }

            const query = 'SELECT id, region, ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount) AS rn, SUM(amount) OVER (PARTITION BY region) AS region_total FROM perf_window';

            // Warmup
            for (let i = 0; i < warmup; i++) {
                await v.exec(query);
            }

            // Benchmark
            const times = [];
            for (let i = 0; i < runs; i++) {
                const start = performance.now();
                await v.exec(query);
                times.push(performance.now() - start);
            }

            // Cleanup
            await v.exec('DROP TABLE perf_window');

            const sorted = times.sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)];
            const p95 = sorted[Math.floor(sorted.length * 0.95)];

            return { median, p95, min: sorted[0], max: sorted[sorted.length - 1] };
        }, { warmup: WARMUP_RUNS, runs: BENCHMARK_RUNS });

        console.log(`\n=== Window Functions (500 rows) ===`);
        console.log(`Median: ${result.median.toFixed(2)}ms`);
        console.log(`P95:    ${result.p95.toFixed(2)}ms`);
        console.log(`Range:  ${result.min.toFixed(2)}-${result.max.toFixed(2)}ms`);

        expect(result.median).toBeGreaterThanOrEqual(0);
    });

    test('CTE performance', async ({ page }) => {
        const result = await page.evaluate(async ({ warmup, runs }) => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            const rows = 500;

            // Setup - use batch INSERT for better memory efficiency
            await v.exec('DROP TABLE IF EXISTS perf_cte');
            await v.exec('CREATE TABLE perf_cte (id INTEGER, dept_id INTEGER, salary REAL)');

            const batchSize = 10;
            for (let i = 0; i < rows; i += batchSize) {
                const values = [];
                for (let j = i; j < Math.min(i + batchSize, rows); j++) {
                    const deptId = (j % 10) + 1;
                    values.push(`(${j}, ${deptId}, ${30000 + Math.random() * 70000})`);
                }
                await v.exec(`INSERT INTO perf_cte VALUES ${values.join(',')}`);
            }

            const query = `
                WITH dept_stats AS (
                    SELECT dept_id, AVG(salary) AS avg_salary, COUNT(*) AS cnt
                    FROM perf_cte
                    GROUP BY dept_id
                )
                SELECT * FROM dept_stats WHERE avg_salary > 50000
            `;

            // Warmup
            for (let i = 0; i < warmup; i++) {
                await v.exec(query);
            }

            // Benchmark
            const times = [];
            for (let i = 0; i < runs; i++) {
                const start = performance.now();
                await v.exec(query);
                times.push(performance.now() - start);
            }

            // Cleanup
            await v.exec('DROP TABLE perf_cte');

            const sorted = times.sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)];
            const p95 = sorted[Math.floor(sorted.length * 0.95)];

            return { median, p95, min: sorted[0], max: sorted[sorted.length - 1] };
        }, { warmup: WARMUP_RUNS, runs: BENCHMARK_RUNS });

        console.log(`\n=== CTE (500 rows) ===`);
        console.log(`Median: ${result.median.toFixed(2)}ms`);
        console.log(`P95:    ${result.p95.toFixed(2)}ms`);
        console.log(`Range:  ${result.min.toFixed(2)}-${result.max.toFixed(2)}ms`);

        expect(result.median).toBeGreaterThanOrEqual(0);
    });
});
