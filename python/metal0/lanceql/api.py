"""Unified LanceQL API - Clean interface matching browser's vault/TableRef pattern.

Simple, unified API:

    import lanceql

    # Open dataset - returns query builder
    db = lanceql.connect("data.lance")

    # DataFrame-style queries
    results = db.table("images").filter("aesthetic > 0.8").similar("embedding", "red shoes").limit(10).collect()

    # SQL queries
    results = db.sql("SELECT url, text FROM images WHERE aesthetic > 0.8 LIMIT 10")

    # Both return the same format - list of dicts
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Iterator
import numpy as np


class TableRef:
    """DataFrame-style query builder. Immutable - each method returns a new TableRef.

    Mirrors the browser's TableRef:
        table.filter("price", "<", 100)
        table.similar("embedding", query, k=10)
        table.select("name", "price")
        table.limit(50)
        table.collect()
    """

    def __init__(self, db: "Connection", table_name: str, *, _query_state: Optional[dict] = None):
        self._db = db
        self._table_name = table_name
        self._state = _query_state or {
            "filters": [],
            "similarity": None,
            "columns": None,
            "limit": None,
            "offset": 0,
            "order_by": None,
        }

    def _clone(self, **updates) -> "TableRef":
        """Create new TableRef with updated state."""
        new_state = {**self._state, **updates}
        return TableRef(self._db, self._table_name, _query_state=new_state)

    # === Filter Methods ===

    def filter(self, expr: str) -> "TableRef":
        """Filter rows using SQL-like expression.

        Args:
            expr: Filter expression (e.g., "price < 100", "category = 'electronics'")

        Example:
            table.filter("price < 100")
            table.filter("category = 'shoes'")
            table.filter("aesthetic > 0.8 AND width > 500")
        """
        return self._clone(filters=self._state["filters"] + [expr])

    def where(self, expr: str) -> "TableRef":
        """Alias for filter()."""
        return self.filter(expr)

    # === Vector Search ===

    def similar(
        self,
        column: str,
        query: Union[str, np.ndarray, List[float]],
        k: int = 10,
    ) -> "TableRef":
        """Find similar vectors using semantic search.

        Args:
            column: Vector column name
            query: Query text (auto-encoded) or vector
            k: Number of similar results

        Example:
            table.similar("embedding", "red running shoes", k=20)
            table.similar("embedding", query_vector, k=10)
        """
        return self._clone(similarity={"column": column, "query": query, "k": k})

    def nearest(self, column: str, query: Union[str, np.ndarray, List[float]], k: int = 10) -> "TableRef":
        """Alias for similar()."""
        return self.similar(column, query, k)

    # === Projection ===

    def select(self, *columns: str) -> "TableRef":
        """Select specific columns.

        Example:
            table.select("url", "text", "score")
        """
        return self._clone(columns=list(columns))

    # === Pagination ===

    def limit(self, n: int) -> "TableRef":
        """Limit number of results."""
        return self._clone(limit=n)

    def offset(self, n: int) -> "TableRef":
        """Skip first n results."""
        return self._clone(offset=n)

    def take(self, n: int) -> "TableRef":
        """Alias for limit()."""
        return self.limit(n)

    # === Ordering ===

    def order_by(self, column: str, desc: bool = False) -> "TableRef":
        """Order results by column."""
        return self._clone(order_by={"column": column, "desc": desc})

    def sort(self, column: str, desc: bool = False) -> "TableRef":
        """Alias for order_by()."""
        return self.order_by(column, desc)

    # === Execution ===

    def collect(self) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dicts."""
        return self._db._execute_table_query(self._table_name, self._state)

    def to_list(self) -> List[Dict[str, Any]]:
        """Alias for collect()."""
        return self.collect()

    def to_arrow(self):
        """Execute and return PyArrow Table."""
        import pyarrow as pa
        rows = self.collect()
        if not rows:
            return pa.table({})
        columns = {k: [r.get(k) for r in rows] for k in rows[0].keys()}
        return pa.table(columns)

    def to_polars(self):
        """Execute and return Polars DataFrame."""
        import polars as pl
        return pl.from_arrow(self.to_arrow())

    def to_pandas(self):
        """Execute and return Pandas DataFrame."""
        return self.to_arrow().to_pandas()

    def first(self) -> Optional[Dict[str, Any]]:
        """Return first result or None."""
        results = self.limit(1).collect()
        return results[0] if results else None

    def count(self) -> int:
        """Count matching rows."""
        return len(self.collect())

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over results."""
        return iter(self.collect())

    def __repr__(self) -> str:
        parts = [f"TableRef({self._table_name!r})"]
        if self._state["filters"]:
            parts.append(f".filter({len(self._state['filters'])} conditions)")
        if self._state["similarity"]:
            parts.append(f".similar({self._state['similarity']['column']!r})")
        if self._state["columns"]:
            parts.append(f".select({len(self._state['columns'])} cols)")
        if self._state["limit"]:
            parts.append(f".limit({self._state['limit']})")
        return "".join(parts)


class Connection:
    """Database connection with unified SQL and DataFrame API.

    Both SQL and DataFrame queries share the same execution engine.

        db = lanceql.connect("data.lance")

        # SQL style
        db.sql("SELECT * FROM images WHERE aesthetic > 0.8 LIMIT 10")

        # DataFrame style
        db.table("images").filter("aesthetic > 0.8").limit(10).collect()

        # Both return List[Dict]
    """

    def __init__(self, source: str):
        self._source = source
        self._is_remote = source.startswith("http://") or source.startswith("https://")
        self._dataset = None
        self._table_names: List[str] = []

    def _ensure_open(self):
        """Lazily open the dataset."""
        if self._dataset is not None:
            return

        if self._is_remote:
            from .remote import RemoteLanceDataset
            self._dataset = RemoteLanceDataset.open(self._source)
            # Remote datasets have a single implicit table
            self._table_names = ["data"]
        else:
            import lance
            self._dataset = lance.dataset(self._source)
            self._table_names = ["data"]  # Lance files are single-table

    # === DataFrame API ===

    def table(self, name: str = "data") -> TableRef:
        """Get a table reference for DataFrame-style queries.

        Args:
            name: Table name (default "data" for single-table datasets)

        Returns:
            TableRef query builder

        Example:
            db.table("images").filter("aesthetic > 0.8").similar("embedding", "shoes").collect()
        """
        self._ensure_open()
        return TableRef(self, name)

    # === SQL API ===

    def sql(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results.

        Args:
            query: SQL SELECT statement

        Returns:
            List of row dictionaries

        Example:
            db.sql("SELECT url, text FROM images WHERE aesthetic > 0.8 LIMIT 10")
        """
        self._ensure_open()
        return self._execute_sql(query)

    def execute(self, query: str) -> List[Dict[str, Any]]:
        """Alias for sql()."""
        return self.sql(query)

    # === Internal Execution ===

    def _execute_table_query(self, table_name: str, state: dict) -> List[Dict[str, Any]]:
        """Execute a TableRef query."""
        if self._is_remote:
            return self._execute_remote_query(state)
        else:
            return self._execute_local_query(state)

    def _execute_local_query(self, state: dict) -> List[Dict[str, Any]]:
        """Execute query on local Lance dataset."""
        import pyarrow.compute as pc

        table = self._dataset.to_table()

        # Apply filters
        for expr in state["filters"]:
            table = self._apply_filter(table, expr)

        # Apply similarity search
        if state["similarity"]:
            table = self._apply_similarity(table, state["similarity"])

        # Apply column selection
        if state["columns"]:
            table = table.select(state["columns"])

        # Apply ordering
        if state["order_by"]:
            col = state["order_by"]["column"]
            desc = state["order_by"]["desc"]
            indices = pc.sort_indices(table[col], sort_keys=[(col, "descending" if desc else "ascending")])
            table = table.take(indices)

        # Apply offset and limit
        if state["offset"] > 0:
            table = table.slice(state["offset"])
        if state["limit"]:
            table = table.slice(0, state["limit"])

        return table.to_pylist()

    def _execute_remote_query(self, state: dict) -> List[Dict[str, Any]]:
        """Execute query on remote dataset with IVF index."""
        from .vector import vector_accelerator

        # Remote queries require similarity search
        if not state["similarity"]:
            raise ValueError("Remote queries require .similar() for vector search")

        sim = state["similarity"]
        query = sim["query"]
        k = sim["k"]

        # Encode text query
        if isinstance(query, str):
            query = _encode_text(query)
        elif isinstance(query, list):
            query = np.array(query, dtype=np.float32)

        # Normalize
        query = query / np.linalg.norm(query)

        # Execute vector search
        results = self._dataset.vector_search(query, top_k=k)

        return [
            {"_index": int(idx), "_score": float(score)}
            for idx, score in zip(results["indices"], results["scores"])
        ]

    def _apply_filter(self, table, expr: str):
        """Apply a filter expression to PyArrow table."""
        import pyarrow.compute as pc
        import re

        # Parse simple expressions: "column op value"
        # Supports: =, !=, <, <=, >, >=
        pattern = r"(\w+)\s*(=|!=|<>|<=|>=|<|>)\s*(.+)"
        match = re.match(pattern, expr.strip())

        if not match:
            raise ValueError(f"Cannot parse filter: {expr}")

        col, op, val = match.groups()
        val = val.strip().strip("'\"")

        # Try to convert to appropriate type
        try:
            val = float(val)
            if val == int(val):
                val = int(val)
        except ValueError:
            pass  # Keep as string

        if op == "=" or op == "==":
            mask = pc.equal(table[col], val)
        elif op == "!=" or op == "<>":
            mask = pc.not_equal(table[col], val)
        elif op == "<":
            mask = pc.less(table[col], val)
        elif op == "<=":
            mask = pc.less_equal(table[col], val)
        elif op == ">":
            mask = pc.greater(table[col], val)
        elif op == ">=":
            mask = pc.greater_equal(table[col], val)
        else:
            raise ValueError(f"Unsupported operator: {op}")

        return table.filter(mask)

    def _apply_similarity(self, table, sim_state: dict):
        """Apply similarity search to table."""
        from .vector import vector_accelerator

        col = sim_state["column"]
        query = sim_state["query"]
        k = sim_state["k"]

        # Encode text query
        if isinstance(query, str):
            query = _encode_text(query)
        elif isinstance(query, list):
            query = np.array(query, dtype=np.float32)

        # Get vectors from table
        vectors = table[col].to_numpy()
        if len(vectors) == 0:
            return table

        # Convert to 2D array
        if hasattr(vectors[0], 'tolist'):
            vectors = np.array([v.tolist() for v in vectors], dtype=np.float32)
        else:
            vectors = np.array(vectors, dtype=np.float32)

        # Normalize query
        query = query / np.linalg.norm(query)

        # Find top-k
        indices, scores = vector_accelerator.top_k_similarity(query, vectors, k=k)

        return table.take(indices)

    def _execute_sql(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query."""
        import re

        # Parse SELECT ... FROM table WHERE ... LIMIT ...
        query = query.strip()

        # Extract table name
        from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
        if not from_match:
            raise ValueError("Invalid SQL: missing FROM clause")

        table_name = from_match.group(1)
        table_ref = self.table(table_name)

        # Extract columns
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE)
        if select_match:
            cols = select_match.group(1).strip()
            if cols != "*":
                columns = [c.strip() for c in cols.split(",")]
                table_ref = table_ref.select(*columns)

        # Extract WHERE
        where_match = re.search(r'WHERE\s+(.+?)(?:ORDER|LIMIT|$)', query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1).strip()
            # Split on AND (simple case)
            conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
            for cond in conditions:
                table_ref = table_ref.filter(cond.strip())

        # Extract LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            table_ref = table_ref.limit(int(limit_match.group(1)))

        # Extract ORDER BY
        order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', query, re.IGNORECASE)
        if order_match:
            col = order_match.group(1)
            desc = order_match.group(2) and order_match.group(2).upper() == "DESC"
            table_ref = table_ref.order_by(col, desc)

        return table_ref.collect()

    @property
    def tables(self) -> List[str]:
        """List available tables."""
        self._ensure_open()
        return self._table_names

    def __repr__(self) -> str:
        return f"Connection({self._source!r})"


def connect(source: str) -> Connection:
    """Connect to a Lance dataset.

    Args:
        source: Path to local .lance file or URL to remote dataset

    Returns:
        Connection with SQL and DataFrame APIs

    Example:
        # Local file
        db = lanceql.connect("data.lance")
        results = db.table().filter("score > 0.8").limit(10).collect()

        # Remote dataset
        db = lanceql.connect("https://data.metal0.dev/laion-1m/images.lance")
        results = db.table().similar("embedding", "red shoes", k=10).collect()

        # SQL
        results = db.sql("SELECT url, text FROM data WHERE aesthetic > 0.8 LIMIT 10")
    """
    return Connection(source)


# Alias for convenience
open = connect


def _encode_text(text: str) -> np.ndarray:
    """Encode text to embedding vector using MiniLM-L6-v2."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text, convert_to_numpy=True).astype(np.float32)
    except ImportError:
        raise ImportError(
            "Text encoding requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        )
