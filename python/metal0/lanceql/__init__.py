"""LanceQL - Query engine for Lance columnar files.

Simple, unified API for working with Lance datasets:

    import lanceql

    # Connect to dataset (local or remote)
    db = lanceql.connect("data.lance")

    # DataFrame-style queries
    results = (
        db.table()
        .filter("aesthetic > 0.8")
        .similar("embedding", "red shoes", k=10)
        .select("url", "text")
        .limit(100)
        .collect()
    )

    # SQL queries (same execution engine)
    results = db.sql("SELECT url, text FROM data WHERE aesthetic > 0.8 LIMIT 10")

    # Both return List[Dict] - convert to any format
    df = db.table().filter("score > 0.5").to_polars()
    table = db.table().limit(100).to_arrow()
"""

__version__ = "0.1.0"

# Public API - clean interface only
from .api import connect, Connection, TableRef

# Convenience alias
open = connect

__all__ = [
    "connect",
    "open",
    "Connection",
    "TableRef",
    "__version__",
]


# Internal modules accessible via lanceql.internal.*
def __getattr__(name):
    """Lazy load internal modules."""
    if name == "internal":
        from types import SimpleNamespace
        from . import parquet, cache, remote, vector, compiler, logic_table

        return SimpleNamespace(
            parquet=parquet,
            HotTierCache=cache.HotTierCache,
            hot_tier_cache=cache.hot_tier_cache,
            RemoteLanceDataset=remote.RemoteLanceDataset,
            IVFIndex=remote.IVFIndex,
            VectorAccelerator=vector.VectorAccelerator,
            vector_accelerator=vector.vector_accelerator,
            LogicTableCompiler=compiler.LogicTableCompiler,
            compile_logic_table=compiler.compile_logic_table,
            compile_logic_table_file=compiler.compile_logic_table_file,
            CompilerError=compiler.CompilerError,
            CompiledLogicTable=logic_table.CompiledLogicTable,
        )

    # Display module for Jupyter integration
    if name == "display":
        from . import display
        return display

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
