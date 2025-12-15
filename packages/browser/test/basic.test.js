/**
 * Basic tests for lanceql npm package.
 * Tests module loading and basic API.
 */

const { test, describe } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

// Test that the module can be required
describe('Module Loading', () => {
    test('CJS module exports are present', () => {
        const lanceql = require('../dist/lanceql.js');

        // Check that key exports exist
        assert.ok(typeof lanceql.LanceQL === 'function' || typeof lanceql.LanceQL === 'object',
            'LanceQL should be exported');
    });

    test('ESM module exists', () => {
        const esmPath = path.join(__dirname, '..', 'dist', 'lanceql.esm.js');
        assert.ok(fs.existsSync(esmPath), 'ESM module should exist');
    });

    test('WASM file exists', () => {
        const wasmPath = path.join(__dirname, '..', 'dist', 'lanceql.wasm');
        assert.ok(fs.existsSync(wasmPath), 'WASM file should exist');
    });

    test('TypeScript definitions exist', () => {
        const dtsPath = path.join(__dirname, '..', 'dist', 'lanceql.d.ts');
        assert.ok(fs.existsSync(dtsPath), 'TypeScript definitions should exist');
    });
});

describe('Package Structure', () => {
    test('package.json has correct fields', () => {
        const pkg = require('../package.json');

        assert.strictEqual(pkg.name, 'lanceql');
        assert.ok(pkg.version, 'version should be set');
        assert.ok(pkg.main, 'main should be set');
        assert.ok(pkg.module, 'module should be set');
        assert.ok(pkg.types, 'types should be set');
    });

    test('exports map is correct', () => {
        const pkg = require('../package.json');

        assert.ok(pkg.exports['.'], 'default export should exist');
        assert.ok(pkg.exports['./wasm'], 'wasm export should exist');
    });
});
