/**
 * Unit tests for LRU Cache
 */

import { test, describe } from 'node:test';
import assert from 'node:assert';

import { LRUCache } from '../src/client/cache/lru-cache.js';

describe('LRU Cache', () => {
    describe('Basic operations', () => {
        test('put and get a value', () => {
            const cache = new LRUCache({ maxSize: 1000 });
            const data = new Uint8Array([1, 2, 3, 4]);

            cache.put('key1', data);
            const result = cache.get('key1');

            assert.deepStrictEqual(result, data);
        });

        test('returns undefined for missing key', () => {
            const cache = new LRUCache({ maxSize: 1000 });

            const result = cache.get('nonexistent');

            assert.strictEqual(result, undefined);
        });

        test('overwrites existing key', () => {
            const cache = new LRUCache({ maxSize: 1000 });

            cache.put('key1', new Uint8Array([1, 2]));
            cache.put('key1', new Uint8Array([3, 4]));

            const result = cache.get('key1');
            assert.deepStrictEqual(result, new Uint8Array([3, 4]));
        });

        test('delete removes a key', () => {
            const cache = new LRUCache({ maxSize: 1000 });

            cache.put('key1', new Uint8Array([1, 2]));
            cache.delete('key1');

            assert.strictEqual(cache.get('key1'), undefined);
        });

        test('clear removes all keys', () => {
            const cache = new LRUCache({ maxSize: 1000 });

            cache.put('key1', new Uint8Array([1]));
            cache.put('key2', new Uint8Array([2]));
            cache.put('key3', new Uint8Array([3]));
            cache.clear();

            assert.strictEqual(cache.get('key1'), undefined);
            assert.strictEqual(cache.get('key2'), undefined);
            assert.strictEqual(cache.get('key3'), undefined);
        });
    });

    describe('LRU eviction', () => {
        test('evicts oldest entry when max size exceeded', () => {
            const cache = new LRUCache({ maxSize: 10 });

            // Each Uint8Array takes its length in bytes
            cache.put('key1', new Uint8Array(5)); // 5 bytes
            cache.put('key2', new Uint8Array(5)); // 5 bytes, total 10
            cache.put('key3', new Uint8Array(5)); // 5 bytes, should evict key1

            assert.strictEqual(cache.get('key1'), undefined);
            assert.ok(cache.get('key2') !== undefined);
            assert.ok(cache.get('key3') !== undefined);
        });

        test('recently accessed items are not evicted', () => {
            const cache = new LRUCache({ maxSize: 10 });

            cache.put('key1', new Uint8Array(5));
            cache.put('key2', new Uint8Array(5));

            // Access key1 to make it recently used
            cache.get('key1');

            // Add new item, should evict key2 (least recently used)
            cache.put('key3', new Uint8Array(5));

            assert.ok(cache.get('key1') !== undefined);
            assert.strictEqual(cache.get('key2'), undefined);
            assert.ok(cache.get('key3') !== undefined);
        });

        test('handles large items that exceed max size', () => {
            const cache = new LRUCache({ maxSize: 10 });

            // Item larger than cache - should either not be cached or evict everything
            cache.put('big', new Uint8Array(20));

            // Behavior depends on implementation - either cached or rejected
            const stats = cache.stats();
            assert.ok(stats.currentSize <= 20);
        });
    });

    describe('Statistics', () => {
        test('tracks entries count', () => {
            const cache = new LRUCache({ maxSize: 1000 });

            cache.put('key1', new Uint8Array(10));
            cache.put('key2', new Uint8Array(10));
            cache.put('key3', new Uint8Array(10));

            const stats = cache.stats();
            assert.strictEqual(stats.entries, 3);
        });

        test('tracks current size', () => {
            const cache = new LRUCache({ maxSize: 1000 });

            cache.put('key1', new Uint8Array(100));
            cache.put('key2', new Uint8Array(200));

            const stats = cache.stats();
            assert.strictEqual(stats.currentSize, 300);
        });

        test('tracks max size', () => {
            const cache = new LRUCache({ maxSize: 5000 });

            const stats = cache.stats();
            assert.strictEqual(stats.maxSize, 5000);
        });

        test('updates size after delete', () => {
            const cache = new LRUCache({ maxSize: 1000 });

            cache.put('key1', new Uint8Array(100));
            cache.put('key2', new Uint8Array(200));
            cache.delete('key1');

            const stats = cache.stats();
            assert.strictEqual(stats.currentSize, 200);
            assert.strictEqual(stats.entries, 1);
        });
    });

    describe('Edge cases', () => {
        test('handles empty Uint8Array', () => {
            const cache = new LRUCache({ maxSize: 1000 });

            cache.put('empty', new Uint8Array(0));
            const result = cache.get('empty');

            assert.ok(result !== undefined);
            assert.strictEqual(result.length, 0);
        });

        test('handles string values', () => {
            const cache = new LRUCache({ maxSize: 1000 });

            cache.put('str', 'hello world');
            const result = cache.get('str');

            assert.strictEqual(result, 'hello world');
        });

        test('handles object values', () => {
            const cache = new LRUCache({ maxSize: 1000 });
            const obj = { foo: 'bar', num: 42 };

            cache.put('obj', obj);
            const result = cache.get('obj');

            assert.deepStrictEqual(result, obj);
        });

        test('handles null values', () => {
            const cache = new LRUCache({ maxSize: 1000 });

            cache.put('null', null);
            const result = cache.get('null');

            assert.strictEqual(result, null);
        });
    });

    describe('Default options', () => {
        test('uses default max size if not specified', () => {
            const cache = new LRUCache();

            const stats = cache.stats();
            assert.ok(stats.maxSize > 0);
        });
    });
});
