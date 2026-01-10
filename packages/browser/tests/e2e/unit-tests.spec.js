/**
 * Browser-based unit tests using Playwright
 * These tests run in real browser context, not Node.js
 */

import { test, expect } from '@playwright/test';

// SQL Lexer and SQL Parser tests removed - JS SQL layer deleted, SQL now handled by WASM

test.describe('LRU Cache (Browser)', () => {
    test('put and get a value', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(async () => {
            const { LRUCache } = await import('./src/client/cache/lru-cache.js');
            const cache = new LRUCache({ maxSize: 1000 });
            const data = new Uint8Array([1, 2, 3, 4]);
            cache.put('key1', data);
            const retrieved = cache.get('key1');
            return Array.from(retrieved);
        });

        expect(result).toEqual([1, 2, 3, 4]);
    });

    test('returns undefined for missing key', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(async () => {
            const { LRUCache } = await import('./src/client/cache/lru-cache.js');
            const cache = new LRUCache({ maxSize: 1000 });
            return cache.get('nonexistent');
        });

        expect(result).toBeUndefined();
    });

    test('overwrites existing key', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(async () => {
            const { LRUCache } = await import('./src/client/cache/lru-cache.js');
            const cache = new LRUCache({ maxSize: 1000 });
            cache.put('key1', new Uint8Array([1, 2]));
            cache.put('key1', new Uint8Array([3, 4]));
            return Array.from(cache.get('key1'));
        });

        expect(result).toEqual([3, 4]);
    });

    test('delete removes a key', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(async () => {
            const { LRUCache } = await import('./src/client/cache/lru-cache.js');
            const cache = new LRUCache({ maxSize: 1000 });
            cache.put('key1', new Uint8Array([1, 2]));
            cache.delete('key1');
            return cache.get('key1');
        });

        expect(result).toBeUndefined();
    });

    test('clear removes all keys', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(async () => {
            const { LRUCache } = await import('./src/client/cache/lru-cache.js');
            const cache = new LRUCache({ maxSize: 1000 });
            cache.put('key1', new Uint8Array([1]));
            cache.put('key2', new Uint8Array([2]));
            cache.put('key3', new Uint8Array([3]));
            cache.clear();
            return {
                key1: cache.get('key1'),
                key2: cache.get('key2'),
                key3: cache.get('key3')
            };
        });

        expect(result.key1).toBeUndefined();
        expect(result.key2).toBeUndefined();
        expect(result.key3).toBeUndefined();
    });

    test('evicts oldest entry when max size exceeded', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(async () => {
            const { LRUCache } = await import('./src/client/cache/lru-cache.js');
            const cache = new LRUCache({ maxSize: 10 });
            cache.put('key1', new Uint8Array(5));
            cache.put('key2', new Uint8Array(5));
            cache.put('key3', new Uint8Array(5)); // Should evict key1
            return {
                key1: cache.get('key1'),
                key2: cache.get('key2') !== undefined,
                key3: cache.get('key3') !== undefined
            };
        });

        expect(result.key1).toBeUndefined();
        expect(result.key2).toBe(true);
        expect(result.key3).toBe(true);
    });

    test('recently accessed items are not evicted', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(async () => {
            const { LRUCache } = await import('./src/client/cache/lru-cache.js');
            const cache = new LRUCache({ maxSize: 10 });
            cache.put('key1', new Uint8Array(5));
            cache.put('key2', new Uint8Array(5));
            cache.get('key1'); // Access key1 to make it recently used
            cache.put('key3', new Uint8Array(5)); // Should evict key2
            return {
                key1: cache.get('key1') !== undefined,
                key2: cache.get('key2'),
                key3: cache.get('key3') !== undefined
            };
        });

        expect(result.key1).toBe(true);
        expect(result.key2).toBeUndefined();
        expect(result.key3).toBe(true);
    });

    test('tracks statistics', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(async () => {
            const { LRUCache } = await import('./src/client/cache/lru-cache.js');
            const cache = new LRUCache({ maxSize: 1000 });
            cache.put('key1', new Uint8Array(100));
            cache.put('key2', new Uint8Array(200));
            return cache.stats();
        });

        expect(result.entries).toBe(2);
        expect(result.currentSize).toBe(300);
        expect(result.maxSize).toBe(1000);
    });
});

test.describe('WebGPU Shaders (Browser)', () => {
    test('WebGPU availability check', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(async () => {
            return {
                hasNavigatorGpu: !!navigator.gpu,
                hasAdapter: navigator.gpu ? !!(await navigator.gpu.requestAdapter()) : false
            };
        });

        // WebGPU may or may not be available depending on browser/hardware
        expect(typeof result.hasNavigatorGpu).toBe('boolean');
    });

    test('GEMM CPU reference produces correct results', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(() => {
            function cpuGemm(A, B, M, N, K, alpha = 1.0) {
                const C = new Float32Array(M * N);
                for (let i = 0; i < M; i++) {
                    for (let j = 0; j < N; j++) {
                        let sum = 0;
                        for (let k = 0; k < K; k++) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = alpha * sum;
                    }
                }
                return C;
            }

            const A = new Float32Array([1, 2, 3, 4]);
            const B = new Float32Array([5, 6, 7, 8]);
            return Array.from(cpuGemm(A, B, 2, 2, 2));
        });

        expect(result).toEqual([19, 22, 43, 50]);
    });

    test('GELU CPU reference handles negative values', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(() => {
            function cpuGelu(input) {
                const output = new Float32Array(input.length);
                for (let i = 0; i < input.length; i++) {
                    const x = input[i];
                    const sigmoid = 1 / (1 + Math.exp(-1.702 * x));
                    output[i] = x * sigmoid;
                }
                return output;
            }

            const input = new Float32Array([-2, -1, 0, 1, 2]);
            return Array.from(cpuGelu(input));
        });

        expect(result[0]).toBeLessThan(0); // GELU(-2) < 0
        expect(result[1]).toBeLessThan(0); // GELU(-1) < 0
        expect(Math.abs(result[2])).toBeLessThan(1e-6); // GELU(0) ≈ 0
        expect(result[3]).toBeGreaterThan(0.5); // GELU(1) > 0.5
        expect(result[4]).toBeGreaterThan(1.5); // GELU(2) > 1.5
    });

    test('LayerNorm CPU reference normalizes to zero mean', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(() => {
            function cpuLayerNorm(input, gamma, beta, eps = 1e-5) {
                const mean = input.reduce((a, b) => a + b, 0) / input.length;
                const variance = input.reduce((a, b) => a + (b - mean) ** 2, 0) / input.length;
                const invStd = 1 / Math.sqrt(variance + eps);

                const output = new Float32Array(input.length);
                for (let i = 0; i < input.length; i++) {
                    output[i] = ((input[i] - mean) * invStd) * gamma[i] + beta[i];
                }
                return output;
            }

            const input = new Float32Array([1, 2, 3, 4, 5]);
            const gamma = new Float32Array(5).fill(1);
            const beta = new Float32Array(5).fill(0);
            const normalized = cpuLayerNorm(input, gamma, beta);

            const mean = normalized.reduce((a, b) => a + b, 0) / normalized.length;
            const variance = normalized.reduce((a, b) => a + b * b, 0) / normalized.length;

            return { mean, variance };
        });

        expect(Math.abs(result.mean)).toBeLessThan(1e-5);
        expect(Math.abs(result.variance - 1)).toBeLessThan(1e-5);
    });
});

test.describe.serial('Module Loading (Browser)', () => {
    test('LRUCache exports correctly', async ({ page }) => {
        await page.goto('/examples/wasm/test-vault-sql.html');

        const result = await page.evaluate(async () => {
            const { LRUCache } = await import('./src/client/cache/lru-cache.js');
            return typeof LRUCache === 'function';
        });

        expect(result).toBe(true);
    });

    test('WASM demo page loads', async ({ page }) => {
        await page.goto('/examples/wasm/');
        await page.waitForLoadState('networkidle');

        const result = await page.evaluate(() => {
            return document.querySelector('#tables-drop-zone') !== null;
        });

        expect(result).toBe(true);
    });

});

// Time Travel SQL tests removed - JS SQL parser deleted, time travel detection via regex in worker
