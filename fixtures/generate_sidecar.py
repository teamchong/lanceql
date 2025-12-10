#!/usr/bin/env python3
"""
Generate sidecar manifest (.meta.json) for Lance datasets.

This script creates a JSON file with pre-calculated metadata that browsers
can fetch in parallel with the manifest, avoiding extra RTTs for column info.

Usage:
    python generate_sidecar.py <dataset_path> [output_path]

Example:
    python generate_sidecar.py /path/to/images.lance
    # Creates /path/to/images.lance/.meta.json

    python generate_sidecar.py /path/to/images.lance ./meta.json
    # Creates ./meta.json

The sidecar file contains:
- Schema with column names, types, and field IDs
- Fragment info with row counts and data file paths
- Column statistics (min/max/null_count where available)
- Version info for cache invalidation
"""

import json
import sys
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa


def arrow_type_to_string(arrow_type: pa.DataType) -> str:
    """Convert PyArrow type to simple string representation."""
    if pa.types.is_int64(arrow_type):
        return "int64"
    elif pa.types.is_int32(arrow_type):
        return "int32"
    elif pa.types.is_int16(arrow_type):
        return "int16"
    elif pa.types.is_int8(arrow_type):
        return "int8"
    elif pa.types.is_float64(arrow_type):
        return "float64"
    elif pa.types.is_float32(arrow_type):
        return "float32"
    elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    elif pa.types.is_boolean(arrow_type):
        return "boolean"
    elif pa.types.is_fixed_size_list(arrow_type):
        inner = arrow_type_to_string(arrow_type.value_type)
        return f"vector[{arrow_type.list_size}]<{inner}>"
    elif pa.types.is_list(arrow_type):
        inner = arrow_type_to_string(arrow_type.value_type)
        return f"list<{inner}>"
    elif pa.types.is_struct(arrow_type):
        return "struct"
    else:
        return str(arrow_type)


def get_column_stats(ds: lance.LanceDataset, col_name: str) -> dict[str, Any]:
    """Get statistics for a column if available."""
    stats = {}
    try:
        # Try to compute basic stats for numeric columns
        schema = ds.schema
        field = schema.field(col_name)

        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            # Sample first 1000 rows for stats
            sample = ds.head(1000).column(col_name)
            if len(sample) > 0:
                non_null = [v.as_py() for v in sample if v.is_valid]
                if non_null:
                    stats["min"] = min(non_null)
                    stats["max"] = max(non_null)
                    stats["null_count_sample"] = len(sample) - len(non_null)
    except Exception:
        pass
    return stats


def generate_sidecar(dataset_path: str, output_path: str | None = None) -> dict:
    """Generate sidecar manifest for a Lance dataset."""
    ds = lance.dataset(dataset_path)

    # Schema info
    schema_info = []
    for i, field in enumerate(ds.schema):
        col_info = {
            "name": field.name,
            "type": arrow_type_to_string(field.type),
            "index": i,
        }

        # Add stats for numeric columns
        stats = get_column_stats(ds, field.name)
        if stats:
            col_info["stats"] = stats

        schema_info.append(col_info)

    # Fragment info
    fragments_info = []
    for frag in ds.get_fragments():
        frag_info = {
            "id": frag.fragment_id,
            "num_rows": frag.count_rows(),
            "physical_rows": frag.metadata.num_rows,
        }

        # Get data file paths
        data_files = []
        for df in frag.data_files():
            data_files.append(df.path())
        if data_files:
            frag_info["data_files"] = data_files

        # Check for deletion file
        if frag.deletion_file():
            frag_info["has_deletions"] = True
            frag_info["deleted_rows"] = frag.metadata.num_rows - frag.count_rows()

        fragments_info.append(frag_info)

    # Build sidecar manifest
    sidecar = {
        "version": 1,
        "lance_version": ds.version,
        "schema": schema_info,
        "fragments": fragments_info,
        "total_rows": ds.count_rows(),
        "num_columns": len(ds.schema),
    }

    # Determine output path
    if output_path is None:
        output_path = Path(dataset_path) / ".meta.json"
    else:
        output_path = Path(output_path)

    # Write sidecar file
    with open(output_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Generated sidecar manifest: {output_path}")
    print(f"  Schema: {len(schema_info)} columns")
    print(f"  Fragments: {len(fragments_info)}")
    print(f"  Total rows: {sidecar['total_rows']:,}")

    return sidecar


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide a dataset path")
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(dataset_path).exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    generate_sidecar(dataset_path, output_path)


if __name__ == "__main__":
    main()
