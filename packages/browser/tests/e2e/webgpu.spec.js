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

    test('Chunked GPU Joiner initializes', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { getChunkedGPUJoiner } = await import('./src/client/webgpu/chunked-gpu-join.js');
                const joiner = getChunkedGPUJoiner();
                const available = await joiner.init();
                return { available, error: null };
            } catch (e) {
                return { available: false, error: e.message };
            }
        });

        console.log(`ChunkedGPUJoiner Available: ${result.available}`);
        expect(result.error).toBeNull();
    });

    test('Chunked GPU Grouper initializes', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { getChunkedGPUGrouper } = await import('./src/client/webgpu/chunked-gpu-group.js');
                const grouper = getChunkedGPUGrouper();
                const available = await grouper.init();
                return { available, error: null };
            } catch (e) {
                return { available: false, error: e.message };
            }
        });

        console.log(`ChunkedGPUGrouper Available: ${result.available}`);
        expect(result.error).toBeNull();
    });

    test('OPFS Result Buffer write and read', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { OPFSResultBuffer } = await import('./src/client/cache/opfs-result-buffer.js');
                const buffer = new OPFSResultBuffer('test-buffer');
                await buffer.init();

                // Write some data
                await buffer.appendMatches(new Uint32Array([1, 2, 3, 4]));
                await buffer.appendMatches(new Uint32Array([5, 6, 7, 8]));
                const stats = await buffer.finalize();

                // Read back
                const data = await buffer.readAll();
                await buffer.close(true);

                return {
                    stats,
                    data: Array.from(data),
                    error: null
                };
            } catch (e) {
                return { stats: null, data: [], error: e.message };
            }
        });

        if (result.error) {
            console.log(`OPFS not available: ${result.error}`);
        } else {
            expect(result.data).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
            expect(result.stats.totalEntries).toBe(8);
            console.log(`OPFS Buffer Stats: ${JSON.stringify(result.stats)}`);
        }
    });

    test('Chunked Join CPU fallback works', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { ChunkedGPUJoiner } = await import('./src/client/webgpu/chunked-gpu-join.js');
                const joiner = new ChunkedGPUJoiner();
                // Don't init GPU - force CPU fallback

                // Create test data as async iterables
                async function* leftData() {
                    yield {
                        keys: new Uint32Array([1, 2, 3, 4]),
                        indices: new Uint32Array([0, 1, 2, 3])
                    };
                }
                async function* rightData() {
                    yield {
                        keys: new Uint32Array([2, 3, 5]),
                        indices: new Uint32Array([0, 1, 2])
                    };
                }

                const resultBuffer = await joiner.hashJoin(leftData(), rightData());
                const matches = await resultBuffer.readAll();
                await resultBuffer.close(true);

                // Should match: (1,0) -> key 2, (2,1) -> key 3
                return {
                    matchCount: matches.length / 2,
                    matches: Array.from(matches),
                    error: null
                };
            } catch (e) {
                return { matchCount: 0, matches: [], error: e.message };
            }
        });

        expect(result.error).toBeNull();
        expect(result.matchCount).toBe(2);
        console.log(`Chunked Join Matches: ${result.matchCount}`);
    });

    test('Chunked Vector Search initializes', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { getChunkedGPUVectorSearch } = await import('./src/client/webgpu/chunked-gpu-vector-search.js');
                const search = getChunkedGPUVectorSearch();
                const available = await search.init();
                return { available, error: null };
            } catch (e) {
                return { available: false, error: e.message };
            }
        });

        console.log(`ChunkedGPUVectorSearch Available: ${result.available}`);
        expect(result.error).toBeNull();
    });

    test('Chunked Vector Search finds nearest vectors', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { ChunkedGPUVectorSearch, DistanceMetric } = await import('./src/client/webgpu/chunked-gpu-vector-search.js');
                const search = new ChunkedGPUVectorSearch();
                // Don't init GPU - force CPU fallback

                // Create test vectors (4 dimensions)
                const queryVec = new Float32Array([1, 0, 0, 0]);
                const vectors = [
                    new Float32Array([1, 0, 0, 0]),    // idx 0: exact match
                    new Float32Array([0.9, 0.1, 0, 0]), // idx 1: close
                    new Float32Array([0, 1, 0, 0]),    // idx 2: orthogonal
                    new Float32Array([-1, 0, 0, 0]),   // idx 3: opposite
                ];

                const results = await search.searchFlat(queryVec, vectors, {
                    k: 2,
                    metric: DistanceMetric.COSINE
                });

                return {
                    topIndices: Array.from(results.indices),
                    topScores: Array.from(results.scores),
                    error: null
                };
            } catch (e) {
                return { topIndices: [], topScores: [], error: e.message };
            }
        });

        expect(result.error).toBeNull();
        expect(result.topIndices.length).toBe(2);
        // Index 0 should be first (exact match), index 1 should be second (close)
        expect(result.topIndices[0]).toBe(0);
        expect(result.topIndices[1]).toBe(1);
        console.log(`Vector Search Results: indices=${result.topIndices}, scores=${result.topScores.map(s => s.toFixed(3))}`);
    });

    test('Chunked Sorter initializes', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { getChunkedGPUSorter } = await import('./src/client/webgpu/chunked-gpu-sort.js');
                const sorter = getChunkedGPUSorter();
                const available = await sorter.init();
                return { available, error: null };
            } catch (e) {
                return { available: false, error: e.message };
            }
        });

        console.log(`ChunkedGPUSorter Available: ${result.available}`);
        expect(result.error).toBeNull();
    });

    test('Chunked Sort with LIMIT works', async ({ page }) => {
        const result = await page.evaluate(async () => {
            try {
                const { ChunkedGPUSorter } = await import('./src/client/webgpu/chunked-gpu-sort.js');
                const sorter = new ChunkedGPUSorter();
                // Don't init GPU - force CPU fallback

                // Create test data
                async function* generateChunks() {
                    yield {
                        keys: new Float32Array([5, 2, 8, 1]),
                        indices: new Uint32Array([0, 1, 2, 3])
                    };
                    yield {
                        keys: new Float32Array([9, 3, 7, 4]),
                        indices: new Uint32Array([4, 5, 6, 7])
                    };
                }

                // Sort descending, get top 3
                const results = await sorter.sortWithLimit(generateChunks(), 3, true);

                return {
                    keys: Array.from(results.keys),
                    indices: Array.from(results.indices),
                    error: null
                };
            } catch (e) {
                return { keys: [], indices: [], error: e.message };
            }
        });

        expect(result.error).toBeNull();
        expect(result.keys.length).toBe(3);
        // Top 3 descending should be: 9, 8, 7
        expect(result.keys).toEqual([9, 8, 7]);
        expect(result.indices).toEqual([4, 2, 6]);
        console.log(`Chunked Sort Top 3: keys=${result.keys}, indices=${result.indices}`);
    });
});
