/**
 * Tests for @logic_table runtime compilation.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { compileLogicTable, CompilerError } from '../src/compiler.js';

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

// TODO: Re-enable when metal0 --emit-logic-table-shared is fixed
// The export wrapper generates invalid Zig code (using catch on non-error types)
describe.skip('LogicTableCompiler', () => {
    it('should compile and call dot_product', async () => {
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

    it('should force recompile when requested', async () => {
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

// TODO: Re-enable when metal0 --emit-logic-table-shared is fixed
describe.skip('CompilerError', () => {
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
