# Vector operations for @logic_table benchmark
# Compiled by metal0: metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a
#
# Uses SIMD-vectorized built-in functions from logic_table module for maximum performance.
# These compile to @Vector operations in Zig, achieving near-native SIMD performance.

import logic_table

@logic_table.logic_table
class VectorOps:
    """Core vector operations for similarity search and ML.

    Uses SIMD-accelerated built-in functions for 4-way vectorization.
    On 384-dimensional vectors, this processes 96 iterations instead of 384.
    """

    def dot_product(self, a: list, b: list) -> float:
        """Compute dot product of two vectors using SIMD."""
        return logic_table.dot_product(a, b)

    def l2_distance(self, a: list, b: list) -> float:
        """Compute L2 (Euclidean) distance between two vectors using SIMD."""
        return logic_table.l2_distance(a, b)

    def cosine_similarity(self, a: list, b: list) -> float:
        """Compute cosine similarity between two vectors using SIMD."""
        return logic_table.cosine_similarity(a, b)
