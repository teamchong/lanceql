"""LanceQL - PyArrow-compatible driver for Lance columnar files."""

__version__ = "0.1.0"

from . import parquet

__all__ = ["parquet", "__version__"]
