"""
LanceQL Logic Table Protocol

This defines the contract between Python logic tables and the SQL engine.
Logic tables are Python functions that:
1. Declare their input schema (what columns they need)
2. Declare their output schema (what they return)
3. Process data in batches (not row-by-row for performance)
4. Support window functions (PARTITION BY, ORDER BY)

Example:
    @logic_table(
        input_schema={"amount": float, "vendor": str},
        output_schema={"is_fraud": bool, "confidence": float},
        batch_size=1000,
        supports_partition=True
    )
    def fraud_detector(batch: pa.RecordBatch) -> pa.RecordBatch:
        # Process batch, return results
        ...

SQL Usage:
    SELECT fraud_detector(amount, vendor) OVER(PARTITION BY user_id)
    FROM transactions
"""

from typing import Protocol, TypeVar, Generic, Callable, Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import pyarrow as pa


# =============================================================================
# Schema Types
# =============================================================================

class DataType(Enum):
    """Supported data types for logic table schemas."""
    INT64 = "int64"
    FLOAT64 = "float64"
    STRING = "string"
    BOOL = "bool"
    TIMESTAMP = "timestamp"
    BINARY = "binary"
    # Vector types for ML
    VECTOR_F32 = "vector[float32]"
    VECTOR_F64 = "vector[float64]"
    # Nested types
    LIST = "list"
    STRUCT = "struct"


@dataclass
class ColumnSchema:
    """Schema for a single column."""
    name: str
    dtype: DataType
    nullable: bool = True
    # For vector types
    dimensions: Optional[int] = None
    # For list/struct types
    element_type: Optional['ColumnSchema'] = None
    fields: Optional[List['ColumnSchema']] = None


@dataclass
class TableSchema:
    """Schema for input or output of a logic table."""
    columns: List[ColumnSchema]

    def to_arrow(self) -> pa.Schema:
        """Convert to PyArrow schema."""
        fields = []
        for col in self.columns:
            arrow_type = self._to_arrow_type(col)
            fields.append(pa.field(col.name, arrow_type, nullable=col.nullable))
        return pa.schema(fields)

    def _to_arrow_type(self, col: ColumnSchema) -> pa.DataType:
        type_map = {
            DataType.INT64: pa.int64(),
            DataType.FLOAT64: pa.float64(),
            DataType.STRING: pa.string(),
            DataType.BOOL: pa.bool_(),
            DataType.TIMESTAMP: pa.timestamp('us'),
            DataType.BINARY: pa.binary(),
        }
        if col.dtype in type_map:
            return type_map[col.dtype]
        elif col.dtype == DataType.VECTOR_F32:
            return pa.list_(pa.float32(), col.dimensions or -1)
        elif col.dtype == DataType.VECTOR_F64:
            return pa.list_(pa.float64(), col.dimensions or -1)
        elif col.dtype == DataType.LIST:
            elem_type = self._to_arrow_type(col.element_type) if col.element_type else pa.null()
            return pa.list_(elem_type)
        elif col.dtype == DataType.STRUCT:
            fields = [pa.field(f.name, self._to_arrow_type(f)) for f in (col.fields or [])]
            return pa.struct(fields)
        return pa.null()


# =============================================================================
# Window Specification
# =============================================================================

@dataclass
class WindowSpec:
    """Window function specification from SQL OVER clause."""
    partition_by: Optional[List[str]] = None  # PARTITION BY columns
    order_by: Optional[List[tuple]] = None     # [(column, 'asc'|'desc'), ...]
    frame_type: Optional[str] = None           # 'rows' | 'range'
    frame_start: Optional[Any] = None          # Frame start bound
    frame_end: Optional[Any] = None            # Frame end bound


# =============================================================================
# Logic Table Protocol
# =============================================================================

class LogicTableProtocol(Protocol):
    """Protocol that all logic tables must implement."""

    @property
    def input_schema(self) -> TableSchema:
        """Declare what columns this logic table needs."""
        ...

    @property
    def output_schema(self) -> TableSchema:
        """Declare what columns this logic table returns."""
        ...

    @property
    def batch_size(self) -> int:
        """Preferred batch size for processing."""
        ...

    @property
    def supports_partition(self) -> bool:
        """Whether this logic table can process partitions in parallel."""
        ...

    def process(
        self,
        batch: pa.RecordBatch,
        window: Optional[WindowSpec] = None
    ) -> pa.RecordBatch:
        """
        Process a batch of data.

        Args:
            batch: Input data matching input_schema
            window: Optional window specification from OVER clause

        Returns:
            Output data matching output_schema
        """
        ...


# =============================================================================
# Decorator for Easy Definition
# =============================================================================

def logic_table(
    input_schema: Dict[str, type],
    output_schema: Dict[str, type],
    batch_size: int = 10000,
    supports_partition: bool = True,
):
    """
    Decorator to create a logic table from a simple function.

    Example:
        @logic_table(
            input_schema={"amount": float, "vendor": str},
            output_schema={"is_fraud": bool, "confidence": float}
        )
        def fraud_detector(batch: pa.RecordBatch) -> pa.RecordBatch:
            amounts = batch.column("amount").to_numpy()
            # ... process ...
            return pa.RecordBatch.from_pydict({
                "is_fraud": is_fraud,
                "confidence": confidence
            })
    """
    def decorator(func: Callable[[pa.RecordBatch], pa.RecordBatch]):
        # Convert Python types to schema
        input_cols = [
            ColumnSchema(name, _python_type_to_dtype(dtype))
            for name, dtype in input_schema.items()
        ]
        output_cols = [
            ColumnSchema(name, _python_type_to_dtype(dtype))
            for name, dtype in output_schema.items()
        ]

        class LogicTableImpl:
            input_schema = TableSchema(input_cols)
            output_schema = TableSchema(output_cols)
            batch_size = batch_size
            supports_partition = supports_partition

            def process(self, batch: pa.RecordBatch, window: Optional[WindowSpec] = None):
                return func(batch)

            def __call__(self, batch: pa.RecordBatch):
                return self.process(batch)

        return LogicTableImpl()

    return decorator


def _python_type_to_dtype(t: type) -> DataType:
    """Convert Python type hints to DataType."""
    type_map = {
        int: DataType.INT64,
        float: DataType.FLOAT64,
        str: DataType.STRING,
        bool: DataType.BOOL,
        bytes: DataType.BINARY,
    }
    return type_map.get(t, DataType.STRING)


# =============================================================================
# Vector-specific Logic Tables
# =============================================================================

def vector_logic_table(
    vector_column: str,
    dimensions: int,
    output_schema: Dict[str, type],
    batch_size: int = 10000,
):
    """
    Decorator for logic tables that process vectors.

    Example:
        @vector_logic_table(
            vector_column="embedding",
            dimensions=384,
            output_schema={"similarity": float, "cluster_id": int}
        )
        def cluster_vectors(vectors: np.ndarray) -> Dict[str, np.ndarray]:
            # vectors is (N, 384) float32 array
            similarities = cosine_sim(vectors, centroid)
            clusters = assign_clusters(vectors)
            return {"similarity": similarities, "cluster_id": clusters}
    """
    def decorator(func):
        input_cols = [
            ColumnSchema(vector_column, DataType.VECTOR_F32, dimensions=dimensions)
        ]
        output_cols = [
            ColumnSchema(name, _python_type_to_dtype(dtype))
            for name, dtype in output_schema.items()
        ]

        class VectorLogicTable:
            input_schema = TableSchema(input_cols)
            output_schema = TableSchema(output_cols)
            batch_size = batch_size
            supports_partition = True

            def process(self, batch: pa.RecordBatch, window: Optional[WindowSpec] = None):
                import numpy as np
                # Extract vectors as numpy array
                vectors = np.array(batch.column(vector_column).to_pylist())
                # Call user function
                result = func(vectors)
                # Convert to RecordBatch
                return pa.RecordBatch.from_pydict(result)

        return VectorLogicTable()

    return decorator


# =============================================================================
# Time Window Logic Tables
# =============================================================================

@dataclass
class TimeWindow:
    """Specification for time-based windows."""
    timestamp_column: str
    window_duration: str  # e.g., "1 hour", "30 minutes"
    slide_duration: Optional[str] = None  # For hopping windows
    gap_duration: Optional[str] = None     # For session windows


def time_window_logic_table(
    time_window: TimeWindow,
    input_schema: Dict[str, type],
    output_schema: Dict[str, type],
):
    """
    Decorator for logic tables that process time windows.

    Example:
        @time_window_logic_table(
            time_window=TimeWindow(
                timestamp_column="event_time",
                window_duration="1 hour",
                slide_duration="15 minutes"
            ),
            input_schema={"event_time": datetime, "value": float},
            output_schema={"window_start": datetime, "avg_value": float}
        )
        def hourly_avg(batch: pa.RecordBatch) -> pa.RecordBatch:
            # batch contains all events in one time window
            ...
    """
    def decorator(func):
        # Add timestamp column to input schema
        full_input = {time_window.timestamp_column: 'timestamp', **input_schema}

        class TimeWindowLogicTable:
            input_schema = TableSchema([
                ColumnSchema(name, _python_type_to_dtype(dtype) if dtype != 'timestamp' else DataType.TIMESTAMP)
                for name, dtype in full_input.items()
            ])
            output_schema = TableSchema([
                ColumnSchema(name, _python_type_to_dtype(dtype))
                for name, dtype in output_schema.items()
            ])
            batch_size = 10000
            supports_partition = True
            window_spec = time_window

            def process(self, batch: pa.RecordBatch, window: Optional[WindowSpec] = None):
                return func(batch)

        return TimeWindowLogicTable()

    return decorator


# =============================================================================
# Protocol Registration
# =============================================================================

_registered_tables: Dict[str, LogicTableProtocol] = {}


def register_logic_table(name: str, table: LogicTableProtocol):
    """Register a logic table for use in SQL."""
    _registered_tables[name] = table


def get_logic_table(name: str) -> Optional[LogicTableProtocol]:
    """Get a registered logic table by name."""
    return _registered_tables.get(name)


def list_logic_tables() -> List[str]:
    """List all registered logic tables."""
    return list(_registered_tables.keys())


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    # Example 1: Simple fraud detection
    @logic_table(
        input_schema={"amount": float, "vendor": str},
        output_schema={"is_fraud": bool, "confidence": float}
    )
    def fraud_detector(batch: pa.RecordBatch) -> pa.RecordBatch:
        amounts = batch.column("amount").to_numpy()
        # Simple threshold-based detection
        is_fraud = amounts > 10000
        confidence = np.minimum(amounts / 10000, 1.0)
        return pa.RecordBatch.from_pydict({
            "is_fraud": is_fraud,
            "confidence": confidence
        })

    # Example 2: Vector similarity
    @vector_logic_table(
        vector_column="embedding",
        dimensions=384,
        output_schema={"similarity": float}
    )
    def cosine_similarity(vectors: np.ndarray) -> Dict[str, np.ndarray]:
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        similarities = vectors_norm @ query
        return {"similarity": similarities}

    # Register for SQL use
    register_logic_table("fraud_detector", fraud_detector)
    register_logic_table("cosine_similarity", cosine_similarity)

    # Show registered tables
    print("Registered logic tables:")
    for name in list_logic_tables():
        table = get_logic_table(name)
        print(f"  {name}:")
        print(f"    Input:  {[c.name for c in table.input_schema.columns]}")
        print(f"    Output: {[c.name for c in table.output_schema.columns]}")
        print(f"    Batch:  {table.batch_size}")
