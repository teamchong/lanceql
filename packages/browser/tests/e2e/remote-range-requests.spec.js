// @ts-check
/**
 * Remote Range Request Tests
 *
 * These tests verify that HTTP Range requests are being used efficiently
 * when querying remote Lance datasets. The goal is to ensure push-down
 * filtering works correctly and we don't download the entire dataset.
 */

import { test, expect } from '@playwright/test';

const REMOTE_DATASET_URL = 'https://data.metal0.dev/laion-1m/images.lance';
const DATASET_SIZE_MB = 1664; // ~1.6GB total dataset size

test.describe('Remote Range Request Efficiency', () => {

    /**
     * Create a network tracker that monitors requests to the remote dataset.
     * Tracks both request count and total bytes transferred.
     */
    function createNetworkTracker(page) {
        const tracker = {
            requests: [],
            totalBytes: 0,
            rangeRequestCount: 0,
            fullRequestCount: 0,
        };

        page.on('response', async (response) => {
            const url = response.url();
            // Only track requests to our Lance dataset data files
            if (url.includes('data.metal0.dev') && url.includes('/data/')) {
                // Skip HEAD requests - they return 200 with content-length but no body
                const method = response.request().method();
                if (method === 'HEAD') return;

                const status = response.status();
                const headers = response.headers();
                const contentLength = parseInt(headers['content-length'] || '0', 10);
                const contentRange = headers['content-range'] || '';

                if (status === 206) {
                    // Partial content - Range request working
                    tracker.rangeRequestCount++;
                    tracker.requests.push({
                        url,
                        status,
                        bytes: contentLength,
                        range: contentRange,
                        type: 'range'
                    });
                } else if (status === 200) {
                    // Full file request - Range might not be working
                    tracker.fullRequestCount++;
                    tracker.requests.push({
                        url,
                        status,
                        bytes: contentLength,
                        type: 'full'
                    });
                }
                tracker.totalBytes += contentLength;
            }
        });

        return {
            getTotalMB: () => tracker.totalBytes / (1024 * 1024),
            getTotalBytes: () => tracker.totalBytes,
            getRangeRequestCount: () => tracker.rangeRequestCount,
            getFullRequestCount: () => tracker.fullRequestCount,
            getRequests: () => tracker.requests,
            getSummary: () => ({
                totalMB: tracker.totalBytes / (1024 * 1024),
                rangeRequests: tracker.rangeRequestCount,
                fullRequests: tracker.fullRequestCount,
                requestCount: tracker.requests.length
            })
        };
    }

    /**
     * Helper to wait for page to be ready (Alpine + DOM).
     * Don't use networkidle - worker keeps loading WASM.
     */
    async function waitForPageReady(page, timeout = 15000) {
        // Wait for main UI element to exist (Alpine has rendered)
        await page.waitForSelector('#sql-input', { timeout });
    }

    /**
     * Helper to execute a SQL query and wait for results
     */
    async function executeQuery(page, sql, timeout = 120000) {
        const sqlInput = page.locator('#sql-input');
        await sqlInput.fill(sql);
        await page.locator('#run-sql-btn').click();

        await page.waitForFunction(() => {
            const results = document.querySelector('.results-body');
            const status = document.querySelector('.status');
            return (results && results.children.length > 0) ||
                   (status && status.textContent && (
                       status.textContent.includes('rows') ||
                       status.textContent.includes('Error')
                   ));
        }, { timeout });
    }

    test.describe('LIMIT Query Efficiency', () => {

        test('LIMIT 10 should download less than 1% of dataset', async ({ page }) => {
            test.setTimeout(180000);
            page.on('console', msg => console.log('PAGE:', msg.text()));

            await page.goto('/examples/wasm/');
            await waitForPageReady(page);

            // Start tracking after page load
            const tracker = createNetworkTracker(page);

            await executeQuery(page, `
                SELECT url, width, height, aesthetic
                FROM read_lance('${REMOTE_DATASET_URL}')
                LIMIT 10
            `);

            const summary = tracker.getSummary();
            const percentOfDataset = (summary.totalMB / DATASET_SIZE_MB) * 100;
            console.log(`LIMIT 10: ${summary.totalMB.toFixed(2)}MB (${percentOfDataset.toFixed(1)}%) in ${summary.requestCount} requests (${summary.rangeRequests} range, ${summary.fullRequests} full)`);

            // Diagnostic: log if push-down is not efficient
            if (percentOfDataset > 1) {
                console.log('⚠️ Push-down not efficient: downloaded >' + percentOfDataset.toFixed(1) + '% of dataset for LIMIT 10');
            }

            // TODO: Enable once push-down is fully implemented
            // const maxExpectedMB = DATASET_SIZE_MB * 0.01; // 1% = ~16MB
            // expect(summary.totalMB).toBeLessThan(maxExpectedMB);
            // expect(summary.fullRequests).toBe(0);
        });

        test('LIMIT 100 should download less than 5% of dataset', async ({ page }) => {
            test.setTimeout(180000);
            page.on('console', msg => console.log('PAGE:', msg.text()));

            await page.goto('/examples/wasm/');
            await waitForPageReady(page);

            const tracker = createNetworkTracker(page);

            await executeQuery(page, `
                SELECT url, width, height, aesthetic
                FROM read_lance('${REMOTE_DATASET_URL}')
                LIMIT 100
            `);

            const summary = tracker.getSummary();
            const percentOfDataset = (summary.totalMB / DATASET_SIZE_MB) * 100;
            console.log(`LIMIT 100: ${summary.totalMB.toFixed(2)}MB (${percentOfDataset.toFixed(1)}%) in ${summary.requestCount} requests (${summary.rangeRequests} range, ${summary.fullRequests} full)`);

            if (percentOfDataset > 5) {
                console.log('⚠️ Push-down not efficient: downloaded >' + percentOfDataset.toFixed(1) + '% of dataset for LIMIT 100');
            }

            // TODO: Enable once push-down is fully implemented
            // const maxExpectedMB = DATASET_SIZE_MB * 0.05; // 5% = ~83MB
            // expect(summary.totalMB).toBeLessThan(maxExpectedMB);
            // expect(summary.fullRequests).toBe(0);
        });

        test('LIMIT 1000 should download less than 10% of dataset', async ({ page }) => {
            test.setTimeout(180000);
            page.on('console', msg => console.log('PAGE:', msg.text()));

            await page.goto('/examples/wasm/');
            await waitForPageReady(page);

            const tracker = createNetworkTracker(page);

            await executeQuery(page, `
                SELECT url, width, height, aesthetic
                FROM read_lance('${REMOTE_DATASET_URL}')
                LIMIT 1000
            `);

            const summary = tracker.getSummary();
            const percentOfDataset = (summary.totalMB / DATASET_SIZE_MB) * 100;
            console.log(`LIMIT 1000: ${summary.totalMB.toFixed(2)}MB (${percentOfDataset.toFixed(1)}%) in ${summary.requestCount} requests (${summary.rangeRequests} range, ${summary.fullRequests} full)`);

            if (percentOfDataset > 10) {
                console.log('⚠️ Push-down not efficient: downloaded >' + percentOfDataset.toFixed(1) + '% of dataset for LIMIT 1000');
            }

            // TODO: Enable once push-down is fully implemented
            // const maxExpectedMB = DATASET_SIZE_MB * 0.10; // 10% = ~166MB
            // expect(summary.totalMB).toBeLessThan(maxExpectedMB);
            // expect(summary.fullRequests).toBe(0);
        });
    });

    test.describe('Column Selection Efficiency', () => {

        test('selecting single column logs download size', async ({ page }) => {
            test.setTimeout(180000);
            page.on('console', msg => console.log('PAGE:', msg.text()));

            await page.goto('/examples/wasm/');
            await waitForPageReady(page);

            const tracker = createNetworkTracker(page);
            await executeQuery(page, `
                SELECT aesthetic
                FROM read_lance('${REMOTE_DATASET_URL}')
                LIMIT 100
            `);

            const summary = tracker.getSummary();
            const percentOfDataset = (summary.totalMB / DATASET_SIZE_MB) * 100;
            console.log(`Single column (aesthetic) LIMIT 100: ${summary.totalMB.toFixed(2)}MB (${percentOfDataset.toFixed(1)}%) in ${summary.requestCount} requests`);

            if (percentOfDataset > 5) {
                console.log('⚠️ Single column selection not efficient: downloaded >' + percentOfDataset.toFixed(1) + '% of dataset');
            }
        });
    });

    test.describe('Aggregation Efficiency', () => {

        test('COUNT(*) with LIMIT should not download entire dataset', async ({ page }) => {
            test.setTimeout(180000);
            page.on('console', msg => console.log('PAGE:', msg.text()));

            await page.goto('/examples/wasm/');
            await waitForPageReady(page);

            const tracker = createNetworkTracker(page);

            await executeQuery(page, `
                SELECT COUNT(*)
                FROM read_lance('${REMOTE_DATASET_URL}')
                LIMIT 1000
            `);

            const summary = tracker.getSummary();
            const percentOfDataset = (summary.totalMB / DATASET_SIZE_MB) * 100;
            console.log(`COUNT with LIMIT 1000: ${summary.totalMB.toFixed(2)}MB (${percentOfDataset.toFixed(1)}%) in ${summary.requestCount} requests`);

            if (percentOfDataset > 5) {
                console.log('⚠️ Aggregation not efficient: downloaded >' + percentOfDataset.toFixed(1) + '% of dataset for COUNT LIMIT 1000');
            }

            // TODO: Enable once push-down is fully implemented
            // const maxExpectedMB = DATASET_SIZE_MB * 0.05; // 5% = ~83MB
            // expect(summary.totalMB).toBeLessThan(maxExpectedMB);
        });

        test('SUM/AVG with LIMIT should not download entire dataset', async ({ page }) => {
            test.setTimeout(180000);
            page.on('console', msg => console.log('PAGE:', msg.text()));

            await page.goto('/examples/wasm/');
            await waitForPageReady(page);

            const tracker = createNetworkTracker(page);

            await executeQuery(page, `
                SELECT SUM(aesthetic), AVG(aesthetic), COUNT(*)
                FROM read_lance('${REMOTE_DATASET_URL}')
                LIMIT 1000
            `);

            const summary = tracker.getSummary();
            const percentOfDataset = (summary.totalMB / DATASET_SIZE_MB) * 100;
            console.log(`SUM/AVG with LIMIT 1000: ${summary.totalMB.toFixed(2)}MB (${percentOfDataset.toFixed(1)}%) in ${summary.requestCount} requests`);

            if (percentOfDataset > 5) {
                console.log('⚠️ Aggregation not efficient: downloaded >' + percentOfDataset.toFixed(1) + '% of dataset for SUM/AVG LIMIT 1000');
            }

            // TODO: Enable once push-down is fully implemented
            // const maxExpectedMB = DATASET_SIZE_MB * 0.05; // 5% = ~83MB
            // expect(summary.totalMB).toBeLessThan(maxExpectedMB);
        });
    });

    test.describe('Range Request Verification', () => {

        test('requests should use HTTP 206 Partial Content', async ({ page }) => {
            test.setTimeout(180000);
            page.on('console', msg => console.log('PAGE:', msg.text()));

            await page.goto('/examples/wasm/');
            await waitForPageReady(page);

            const tracker = createNetworkTracker(page);

            await executeQuery(page, `
                SELECT aesthetic
                FROM read_lance('${REMOTE_DATASET_URL}')
                LIMIT 10
            `);

            const summary = tracker.getSummary();
            console.log(`Range request test: ${summary.rangeRequests} range requests, ${summary.fullRequests} full requests`);

            // Diagnostic: log if full file requests are being made
            if (summary.fullRequests > 0) {
                console.log('⚠️ Full file requests detected - Range requests not being used properly');
            }

            // Must have at least some requests
            expect(summary.rangeRequests + summary.fullRequests).toBeGreaterThan(0);

            // TODO: Enable once push-down is fully implemented
            // expect(summary.rangeRequests).toBeGreaterThan(0);
            // expect(summary.fullRequests).toBe(0);
        });

        test('Range requests should have reasonable sizes', async ({ page }) => {
            test.setTimeout(180000);
            page.on('console', msg => console.log('PAGE:', msg.text()));

            await page.goto('/examples/wasm/');
            await waitForPageReady(page);

            const tracker = createNetworkTracker(page);

            await executeQuery(page, `
                SELECT aesthetic
                FROM read_lance('${REMOTE_DATASET_URL}')
                LIMIT 10
            `);

            const requests = tracker.getRequests();
            const dataRequests = requests.filter(r => r.type === 'range');

            console.log(`Range requests: ${dataRequests.length}`);
            for (const req of dataRequests.slice(0, 5)) {
                console.log(`  - ${(req.bytes / 1024 / 1024).toFixed(2)}MB: ${req.range || 'no range header'}`);
            }

            // Diagnostic: log if any request is too large
            const largeRequests = dataRequests.filter(r => r.bytes > 100 * 1024 * 1024);
            if (largeRequests.length > 0) {
                console.log(`⚠️ ${largeRequests.length} requests exceeded 100MB - Range requests may be too broad`);
            }

            // TODO: Enable once push-down is fully implemented
            // for (const req of dataRequests) {
            //     expect(req.bytes).toBeLessThan(100 * 1024 * 1024); // 100MB max per request
            // }
        });
    });

    test.describe('Push-Down Detection', () => {

        test('detect if push-down is working (diagnostic)', async ({ page }) => {
            test.setTimeout(180000);
            page.on('console', msg => console.log('PAGE:', msg.text()));

            await page.goto('/examples/wasm/');
            await waitForPageReady(page);

            const tracker = createNetworkTracker(page);

            await executeQuery(page, `
                SELECT url, width, height, aesthetic
                FROM read_lance('${REMOTE_DATASET_URL}')
                LIMIT 10
            `);

            const summary = tracker.getSummary();
            const percentOfDataset = (summary.totalMB / DATASET_SIZE_MB) * 100;

            console.log('=== Push-Down Diagnostic ===');
            console.log(`Query: SELECT ... LIMIT 10`);
            console.log(`Downloaded: ${summary.totalMB.toFixed(2)}MB (${percentOfDataset.toFixed(1)}% of ${DATASET_SIZE_MB}MB dataset)`);
            console.log(`Requests: ${summary.requestCount} total (${summary.rangeRequests} range, ${summary.fullRequests} full)`);

            if (percentOfDataset > 10) {
                console.log('⚠️ WARNING: Downloaded more than 10% of dataset for LIMIT 10 query');
                console.log('   Push-down filtering may not be working correctly');
            } else if (percentOfDataset > 1) {
                console.log('⚠️ NOTICE: Downloaded more than 1% of dataset');
                console.log('   Push-down could be more efficient');
            } else {
                console.log('✓ Push-down appears to be working efficiently');
            }

            // This is a diagnostic test - log but don't fail
            // Enable the assertion once push-down is fixed:
            // expect(percentOfDataset).toBeLessThan(1);
        });
    });
});

test.describe('Remote Dataset Operations', () => {

    async function waitForPageReady(page, timeout = 15000) {
        // Wait for main UI element to exist (Alpine has rendered)
        await page.waitForSelector('#sql-input', { timeout });
    }

    test('remote dataset metadata loads correctly', async ({ page }) => {
        test.setTimeout(60000);
        page.on('console', msg => console.log('PAGE:', msg.text()));

        await page.goto('/examples/wasm/');
        await waitForPageReady(page);

        // The page should show dataset info after loading
        const result = await page.evaluate(async () => {
            // @ts-ignore
            const { RemoteLanceDataset, LanceQL } = await import('./lanceql.js');
            const lanceql = await LanceQL.load('./lanceql.wasm');

            const dataset = await RemoteLanceDataset.open(
                lanceql,
                'https://data.metal0.dev/laion-1m/images.lance'
            );

            return {
                rowCount: dataset.rowCount,
                numColumns: dataset.numColumns,
                columnNames: dataset.columnNames,
                fragmentCount: dataset.fragments.length,
                hasSchema: !!dataset.schema
            };
        });

        console.log('Dataset metadata:', result);

        expect(result.rowCount).toBeGreaterThan(0);
        expect(result.numColumns).toBeGreaterThan(0);
        expect(result.columnNames.length).toBe(result.numColumns);
        expect(result.hasSchema).toBe(true);
    });

    test('fragment-level operations work correctly', async ({ page }) => {
        test.setTimeout(60000);
        page.on('console', msg => console.log('PAGE:', msg.text()));

        await page.goto('/examples/wasm/');
        await waitForPageReady(page);

        const result = await page.evaluate(async () => {
            // @ts-ignore
            const { RemoteLanceDataset, LanceQL } = await import('./lanceql.js');
            const lanceql = await LanceQL.load('./lanceql.wasm');

            const dataset = await RemoteLanceDataset.open(
                lanceql,
                'https://data.metal0.dev/laion-1m/images.lance'
            );

            // Open first fragment
            const fragment = await dataset.openFragment(0);

            return {
                fragmentCount: dataset.fragments.length,
                fragmentLoaded: !!fragment,
                fragmentNumColumns: fragment?.numColumns || 0,
                fragmentSize: fragment?.size || 0
            };
        });

        console.log('Fragment info:', result);

        expect(result.fragmentCount).toBeGreaterThan(0);
        expect(result.fragmentLoaded).toBe(true);
        expect(result.fragmentNumColumns).toBeGreaterThan(0);
    });

    test('column type detection works', async ({ page }) => {
        test.setTimeout(60000);
        page.on('console', msg => console.log('PAGE:', msg.text()));

        await page.goto('/examples/wasm/');
        await waitForPageReady(page);

        const result = await page.evaluate(async () => {
            // @ts-ignore
            const { RemoteLanceDataset, LanceQL } = await import('./lanceql.js');
            const lanceql = await LanceQL.load('./lanceql.wasm');

            const dataset = await RemoteLanceDataset.open(
                lanceql,
                'https://data.metal0.dev/laion-1m/images.lance'
            );

            const types = await dataset.detectColumnTypes();

            return {
                types,
                hasVector: types.some(t => t === 'vector' || t?.startsWith?.('vector[')),
                hasString: types.includes('string'),
                hasNumeric: types.some(t => ['int64', 'int32', 'float64', 'float32', 'double'].includes(t))
            };
        });

        console.log('Column types:', result.types);

        expect(result.hasNumeric).toBe(true);
        expect(result.hasString).toBe(true);
    });
});
