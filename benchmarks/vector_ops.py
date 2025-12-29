# Vector operations for @logic_table benchmark
# Compiled by metal0: metal0 build --emit-logic-table benchmarks/vector_ops.py -o lib/vector_ops.a
#
# Uses SIMD @Vector(4, f64) for 4-way vectorization on CPU.
# On 384-dimensional vectors, this processes 96 SIMD iterations instead of 384 scalar.

import logic_table

@logic_table.logic_table
class VectorOps:
    """Core vector operations for similarity search and ML.

    All operations use SIMD (4-way vectorization) for high performance.
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
