/**
 * Tests for @logic_table runtime compilation.
 *
 * NOTE: These tests are skipped on Linux due to a Zig standard library
 * limitation where ChildProcess cannot forward environment variables.
 * See: https://github.com/ziglang/zig/issues/5190
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { compileLogicTable, CompilerError } from '../src/compiler.js';
import os from 'os';

// Skip all tests on Linux due to Zig ChildProcess limitation
const isLinux = os.platform() === 'linux';
const describeOrSkip = isLinux ? describe.skip : describe;

const VECTOR_OPS_SOURCE = `
from logic_table import logic_table

@logic_table
class VectorOps:
    def dot_product(self, a: list, b: list) -> float:
        result = 0.0
        for i in range(len(a)):
            result = result + a[i] * b[i]
        return result

    def sum_squares(self, a: list) -> float:
        result = 0.0
        for i in range(len(a)):
            result = result + a[i] * a[i]
        return result
`;

describeOrSkip('LogicTableCompiler', () => {
    it.skip('should compile and call dot_product', async () => {
        // FIXME: metal0 eval() codegen issue - list indexing uses Python eval instead of native code
        const ops = await compileLogicTable(VECTOR_OPS_SOURCE);

        // Test basic dot product
        const result = ops.VectorOps.dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        expect(result).toBeCloseTo(32.0, 2); // 1*4 + 2*5 + 3*6 = 32
    });

    it('should cache compilation results', async () => {
        // First compilation
        const ops1 = await compileLogicTable(VECTOR_OPS_SOURCE);
        const result1 = ops1.VectorOps.dot_product([1.0, 0.0], [0.0, 1.0]);

        // Second compilation should use cache
        const ops2 = await compileLogicTable(VECTOR_OPS_SOURCE);
        const result2 = ops2.VectorOps.dot_product([1.0, 0.0], [0.0, 1.0]);

        expect(result1).toBe(result2);
    });

    it.skip('should force recompile when requested', async () => {
        // FIXME: metal0 eval() codegen issue
        const ops = await compileLogicTable(VECTOR_OPS_SOURCE, { force: true });
        const result = ops.VectorOps.dot_product([2.0, 3.0], [4.0, 5.0]);
        expect(result).toBeCloseTo(23.0, 2); // 2*4 + 3*5 = 23
    });

    it('should provide class attribute access', async () => {
        const ops = await compileLogicTable(VECTOR_OPS_SOURCE);

        // Both access patterns should work
        expect(ops.VectorOps).toBeDefined();
        expect(ops.VectorOps.dot_product).toBeDefined();
        expect(typeof ops.VectorOps.dot_product).toBe('function');
    });

    it('should expose methods dictionary', async () => {
        const ops = await compileLogicTable(VECTOR_OPS_SOURCE);
        const methods = ops.methods;

        expect(methods).toHaveProperty('VectorOps_dot_product');
    });

    it('should expose classes dictionary', async () => {
        const ops = await compileLogicTable(VECTOR_OPS_SOURCE);
        const classes = ops.classes;

        expect(classes).toHaveProperty('VectorOps');
    });
});

describeOrSkip('CompilerError', () => {
    it('should throw on invalid syntax', async () => {
        const invalidSource = `
from logic_table import logic_table

@logic_table
class BrokenOps:
    def broken(self
        return 42  # Missing closing paren
`;

        await expect(compileLogicTable(invalidSource)).rejects.toThrow(CompilerError);
    });
});
