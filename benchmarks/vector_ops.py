# Vector operations for @logic_table benchmark
# Compiled by metal0: metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a

@logic_table
class VectorOps:
    """Core vector operations for similarity search and ML."""

    def dot_product(self, a: list, b: list) -> float:
        """Compute dot product of two vectors."""
        result = 0.0
        for i in range(len(a)):
            result = result + a[i] * b[i]
        return result

    def sum_squares(self, a: list) -> float:
        """Compute sum of squares."""
        result = 0.0
        for i in range(len(a)):
            result = result + a[i] * a[i]
        return result

    def sum_values(self, a: list) -> float:
        """Sum all values in array."""
        result = 0.0
        for i in range(len(a)):
            result = result + a[i]
        return result
