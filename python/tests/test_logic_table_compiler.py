"""Tests for @logic_table runtime compilation."""

import pytest
import sys
import os

# Add python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metal0.lanceql import compile_logic_table, CompilerError


VECTOR_OPS_SOURCE = '''
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
'''


class TestLogicTableCompiler:
    """Test @logic_table runtime compilation."""

    def test_compile_and_call_dot_product(self):
        """Test compiling and calling dot_product."""
        ops = compile_logic_table(VECTOR_OPS_SOURCE)

        # Test basic dot product
        result = ops.VectorOps.dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert abs(result - 32.0) < 0.001  # 1*4 + 2*5 + 3*6 = 32

    def test_compile_caching(self):
        """Test that compilation results are cached."""
        # First compilation
        ops1 = compile_logic_table(VECTOR_OPS_SOURCE)
        result1 = ops1.VectorOps.dot_product([1.0, 0.0], [0.0, 1.0])

        # Second compilation should use cache
        ops2 = compile_logic_table(VECTOR_OPS_SOURCE)
        result2 = ops2.VectorOps.dot_product([1.0, 0.0], [0.0, 1.0])

        assert result1 == result2

    def test_force_recompile(self):
        """Test force recompilation."""
        ops = compile_logic_table(VECTOR_OPS_SOURCE, force=True)
        result = ops.VectorOps.dot_product([2.0, 3.0], [4.0, 5.0])
        assert abs(result - 23.0) < 0.001  # 2*4 + 3*5 = 23

    def test_class_attribute_access(self):
        """Test accessing methods via class proxy."""
        ops = compile_logic_table(VECTOR_OPS_SOURCE)

        # Both access patterns should work
        assert hasattr(ops, 'VectorOps')
        assert hasattr(ops.VectorOps, 'dot_product')

    def test_methods_dict(self):
        """Test methods dictionary."""
        ops = compile_logic_table(VECTOR_OPS_SOURCE)
        methods = ops.methods

        assert 'VectorOps_dot_product' in methods

    def test_classes_dict(self):
        """Test classes dictionary."""
        ops = compile_logic_table(VECTOR_OPS_SOURCE)
        classes = ops.classes

        assert 'VectorOps' in classes


class TestCompilerError:
    """Test compiler error handling."""

    def test_invalid_syntax(self):
        """Test that invalid Python syntax raises CompilerError."""
        invalid_source = '''
from logic_table import logic_table

@logic_table
class BrokenOps:
    def broken(self
        return 42  # Missing closing paren
'''
        with pytest.raises(CompilerError):
            compile_logic_table(invalid_source)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
