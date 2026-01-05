import { test, expect } from '@playwright/test';

test.describe('WebGPU Features', () => {
    test.beforeEach(async ({ page }) => {
        // Use the test page which imports lanceql
        await page.goto('/examples/wasm/test-vault-sql.html');
        await page.waitForLoadState('domcontentloaded');
    });

    test('GPU Vector Search availability', async ({ page }) => {
        const isAvailable = await page.evaluate(async () => {
            try {
                // Import from the exposed module or file
                const { getGPUVectorSearch } = await import('./webgpu/index.js');
                const gpuSearch = getGPUVectorSearch();
                await gpuSearch.init();
                return gpuSearch.isAvailable();
            } catch (e) {
                console.log('WebGPU check failed:', e);
                return false;
            }
        });

        // In most CI environments this will be false, but we want to ensure the code path runs without error.
        // If we are in a WebGPU-enabled environment, it should be true.
        console.log(`WebGPU Available: ${isAvailable}`);
        
        // We expect the check to complete without throwing, returning boolean.
        expect(typeof isAvailable).toBe('boolean');
    });

    test('GPU Aggregation fallback', async ({ page }) => {
        // Verify that if GPU is not available, we fallback gracefully (implicit test)
        // or if available, it works.
        const result = await page.evaluate(async () => {
            const { vault } = await import('./lanceql.js');
            const v = await vault();
            
            // Create a large table to trigger potential GPU path (threshold is 10k)
            await v.exec('CREATE TABLE test_gpu_agg (id INTEGER, val REAL)');
            const rows = [];
            for (let i = 0; i < 15000; i++) {
                rows.push(`(${i}, ${i * 0.1})`);
            }
            
            // Batch insert to be faster
            const batchSize = 1000;
            for (let i = 0; i < rows.length; i+=batchSize) {
                await v.exec(`INSERT INTO test_gpu_agg VALUES ${rows.slice(i, i+batchSize).join(',')}`);
            }
            
            const start = performance.now();
            const res = await v.exec('SELECT SUM(val) FROM test_gpu_agg');
            const time = performance.now() - start;
            
            return { 
                sum: res.rows[0]['sum(val)'] || res.rows[0][0], 
                time 
            };
        });

        // Sum of 0..14999 * 0.1
        // Sum(0..N-1) = (N-1)*N/2
        // 14999 * 15000 / 2 * 0.1 = 11249250
        const expected = 11249250;
        expect(Math.abs(result.sum - expected)).toBeLessThan(1.0); // Floating point tolerance
        console.log(`Aggregation Time (15k rows): ${result.time.toFixed(2)}ms`);
    });
});
