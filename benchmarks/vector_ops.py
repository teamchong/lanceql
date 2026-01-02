# Vector operations for @logic_table benchmark
#
# Example Python code for future JIT compilation via metal0.
# Currently benchmarks use native Zig SIMD functions instead.

from logic_table import logic_table


@logic_table
class VectorOps:
    """Core vector operations - example @logic_table Python code."""

    def dot_product(self, a: list, b: list) -> float:
        """Compute dot product of two vectors."""
        result = 0.0
        for i in range(len(a)):
            result = result + a[i] * b[i]
        return result

    def sum_squares(self, a: list) -> float:
        """Compute sum of squares of a vector."""
        result = 0.0
        for i in range(len(a)):
            result = result + a[i] * a[i]
        return result

