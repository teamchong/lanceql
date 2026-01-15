"""Global display hook that auto-replaces DataFrame/table rendering.

Usage:
    # Opt-in to virtual scrolling for ALL tables
    import metal0.lanceql.display

    # Now any DataFrame/Table uses virtual scrolling
    import polars as pl
    df = pl.read_parquet('data.parquet')
    df  # Virtual scrolling with lazy loading (requires lanceql-jupyterlab extension)
"""
import uuid
import json
from typing import Any, Dict, List

# Custom MIME type for LanceQL virtual tables
MIME_TYPE = 'application/vnd.lanceql.table+json'

# Global registry: source_id -> DataSource wrapper
_sources: Dict[str, 'DataSource'] = {}


class DataSource:
    """Wrapper that holds data and provides row slicing."""

    def __init__(self, data: Any, source_id: str = None):
        self.id = source_id or str(uuid.uuid4())[:8]
        self.data = data
        self._total_rows = None
        self._columns = None
        self._column_types = None
        _sources[self.id] = self

    @property
    def total_rows(self) -> int:
        if self._total_rows is None:
            self._total_rows = len(self.data)
        return self._total_rows

    @property
    def columns(self) -> List[str]:
        if self._columns is None:
            # PyArrow Table has column_names
            if hasattr(self.data, 'column_names'):
                self._columns = list(self.data.column_names)
            # Polars/Pandas have .columns
            elif hasattr(self.data, 'columns'):
                self._columns = list(self.data.columns)
            # Fallback to schema.names
            elif hasattr(self.data, 'schema') and hasattr(self.data.schema, 'names'):
                names = self.data.schema.names
                if callable(names):
                    self._columns = list(names())
                else:
                    self._columns = list(names)
            else:
                self._columns = []
        return self._columns

    @property
    def column_types(self) -> Dict[str, Dict]:
        """Get column type metadata for each column."""
        if self._column_types is None:
            self._column_types = {}
            for col in self.columns:
                self._column_types[col] = self._detect_column_type(col)
        return self._column_types

    def _detect_column_type(self, col: str) -> Dict:
        """Detect type info for a column."""
        type_info = {'type': 'unknown', 'dtype': 'unknown'}

        try:
            # PyArrow Table
            if hasattr(self.data, 'schema'):
                import pyarrow as pa
                field = self.data.schema.field(col)
                arrow_type = field.type

                if pa.types.is_fixed_size_list(arrow_type):
                    dim = arrow_type.list_size
                    type_info = {
                        'type': 'vector',
                        'dtype': 'fixed_size_list',
                        'dim': dim,
                        'model': _detect_embedding_model(dim)
                    }
                elif pa.types.is_list(arrow_type):
                    type_info = {'type': 'list', 'dtype': 'list'}
                elif pa.types.is_floating(arrow_type):
                    type_info = {'type': 'float', 'dtype': str(arrow_type)}
                elif pa.types.is_integer(arrow_type):
                    type_info = {'type': 'int', 'dtype': str(arrow_type)}
                elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
                    type_info = {'type': 'string', 'dtype': 'string'}
                elif pa.types.is_boolean(arrow_type):
                    type_info = {'type': 'bool', 'dtype': 'bool'}
                elif pa.types.is_timestamp(arrow_type):
                    type_info = {'type': 'timestamp', 'dtype': str(arrow_type)}
                else:
                    type_info = {'type': 'other', 'dtype': str(arrow_type)}

            # Polars DataFrame
            elif hasattr(self.data, 'schema') and hasattr(self.data, 'dtypes'):
                import polars as pl
                dtype = self.data.schema[col]

                if isinstance(dtype, pl.Array):
                    dim = dtype.size
                    type_info = {
                        'type': 'vector',
                        'dtype': 'array',
                        'dim': dim,
                        'model': _detect_embedding_model(dim)
                    }
                elif isinstance(dtype, pl.List):
                    type_info = {'type': 'list', 'dtype': 'list'}
                elif dtype in (pl.Float32, pl.Float64):
                    type_info = {'type': 'float', 'dtype': str(dtype)}
                elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    type_info = {'type': 'int', 'dtype': str(dtype)}
                elif dtype == pl.Utf8 or dtype == pl.String:
                    type_info = {'type': 'string', 'dtype': 'string'}
                elif dtype == pl.Boolean:
                    type_info = {'type': 'bool', 'dtype': 'bool'}
                elif isinstance(dtype, pl.Datetime):
                    type_info = {'type': 'timestamp', 'dtype': str(dtype)}
                else:
                    type_info = {'type': 'other', 'dtype': str(dtype)}

            # Pandas DataFrame
            elif hasattr(self.data, 'dtypes'):
                import numpy as np
                dtype = self.data[col].dtype

                # Check if column contains lists/arrays (vector-like)
                if dtype == object:
                    first_val = self.data[col].iloc[0] if len(self.data) > 0 else None
                    if isinstance(first_val, (list, np.ndarray)):
                        dim = len(first_val) if first_val is not None else 0
                        type_info = {
                            'type': 'vector',
                            'dtype': 'array',
                            'dim': dim,
                            'model': _detect_embedding_model(dim)
                        }
                    else:
                        type_info = {'type': 'string', 'dtype': 'object'}
                elif np.issubdtype(dtype, np.floating):
                    type_info = {'type': 'float', 'dtype': str(dtype)}
                elif np.issubdtype(dtype, np.integer):
                    type_info = {'type': 'int', 'dtype': str(dtype)}
                elif np.issubdtype(dtype, np.bool_):
                    type_info = {'type': 'bool', 'dtype': 'bool'}
                elif np.issubdtype(dtype, np.datetime64):
                    type_info = {'type': 'timestamp', 'dtype': str(dtype)}
                else:
                    type_info = {'type': 'other', 'dtype': str(dtype)}

        except Exception:
            pass

        return type_info

    def get_rows(self, offset: int, limit: int) -> Dict:
        """Slice rows from data source."""
        # PyArrow Table
        if hasattr(self.data, 'slice') and hasattr(self.data, 'to_pydict'):
            sliced = self.data.slice(offset, limit)
            return sliced.to_pydict()
        # Polars DataFrame
        elif hasattr(self.data, 'to_dicts'):
            sliced = self.data[offset:offset+limit]
            # Convert to column-oriented dict
            rows = sliced.to_dicts()
            if not rows:
                return {col: [] for col in self.columns}
            return {col: [r[col] for r in rows] for col in self.columns}
        # Pandas DataFrame
        elif hasattr(self.data, 'iloc'):
            return self.data.iloc[offset:offset+limit].to_dict('list')
        else:
            raise TypeError(f"Unsupported data type: {type(self.data)}")


def _detect_embedding_model(dim: int) -> str:
    """Detect embedding model based on dimension."""
    models = {
        384: 'MiniLM-L6',
        512: 'CLIP ViT-B/32',
        768: 'BERT/MiniLM-L12',
        1024: 'CLIP ViT-L/14',
        1536: 'OpenAI Ada-002',
        3072: 'OpenAI text-embedding-3-large',
    }
    return models.get(dim, f'{dim}d')


def _detect_url_columns(columns: List[str], sample_row: Dict) -> List[str]:
    """Detect columns that contain URLs (potential images)."""
    url_cols = []
    for col in columns:
        val = sample_row.get(col)
        if isinstance(val, str) and val.startswith('http'):
            url_cols.append(col)
    return url_cols


class VirtualTableBundle:
    """Bundle that outputs custom MIME type for JupyterLab extension."""

    def __init__(self, data: Any):
        self.source = DataSource(data)

        # Get initial rows (first chunk for immediate display)
        initial_limit = min(100, self.source.total_rows)
        initial_rows = self.source.get_rows(0, initial_limit)
        columns = self.source.columns

        # Detect image columns
        image_cols = []
        if initial_rows and columns and initial_rows.get(columns[0]):
            first_row = {col: initial_rows[col][0] if initial_rows.get(col) else None for col in columns}
            image_cols = _detect_url_columns(columns, first_row)

        self._bundle = {
            'source_id': self.source.id,
            'total': self.source.total_rows,
            'columns': columns,
            'column_types': self.source.column_types,
            'image_columns': image_cols,
            'rows': initial_rows,
        }

    def _repr_mimebundle_(self, **kwargs):
        """Return MIME bundle with custom type for extension + HTML fallback."""
        return {
            MIME_TYPE: self._bundle,
            'text/html': _render_static_html(self._bundle),
        }, {}


def _format_table_bundle(data: Any):
    """Format any table-like object as VirtualTableBundle."""
    return VirtualTableBundle(data)


def _render_sparkline_svg(values: List, max_points: int = 60) -> str:
    """Render a sparkline SVG for vector data."""
    if not values or not isinstance(values, (list, tuple)):
        return ''

    # Sample if too many points
    sampled = values
    if len(values) > max_points:
        step = len(values) // max_points
        sampled = [values[i] for i in range(0, len(values), step)]

    # Normalize to 0-18 range
    min_v = min(sampled)
    max_v = max(sampled)
    range_v = max_v - min_v if max_v != min_v else 1
    normalized = [18 - ((v - min_v) / range_v) * 16 for v in sampled]

    # Build SVG path
    width = 80
    step = width / (len(normalized) - 1) if len(normalized) > 1 else width
    path = f'M0,{normalized[0]:.1f}'
    for i in range(1, len(normalized)):
        path += f' L{i * step:.1f},{normalized[i]:.1f}'

    return f'<svg width="80" height="20" style="vertical-align:middle"><path d="{path}" fill="none" stroke="#6366f1" stroke-width="1.5"/></svg>'


def _get_type_badge_html(col_type: Dict) -> str:
    """Generate HTML for type badge."""
    if not col_type:
        return ''

    type_name = col_type.get('type', 'unknown')
    colors = {
        'vector': 'background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white',
        'string': 'background:#10b981;color:white',
        'int': 'background:#3b82f6;color:white',
        'float': 'background:#f59e0b;color:white',
        'bool': 'background:#ec4899;color:white',
        'timestamp': 'background:#14b8a6;color:white',
        'list': 'background:#8b5cf6;color:white',
    }
    style = colors.get(type_name, 'background:#6b7280;color:white')

    if type_name == 'vector' and col_type.get('model'):
        label = col_type['model']
    else:
        label = type_name

    return f'<span style="{style};font-size:10px;padding:1px 6px;border-radius:3px;margin-left:6px">{label}</span>'


def _render_static_html(data: Dict) -> str:
    """Render static HTML fallback (when extension not installed)."""
    source_id = data['source_id']
    total = data['total']
    columns = data['columns']
    column_types = data.get('column_types', {})
    image_cols = data['image_columns']
    rows = data['rows']

    # Build header with type badges
    header_html = ''
    for col in columns:
        col_type = column_types.get(col, {})
        badge = _get_type_badge_html(col_type)
        header_html += f'<span class="lq-cell">{_escape_html(col)}{badge}</span>'

    # Build visible rows
    rows_html = ''
    if columns and rows.get(columns[0]):
        num_rows = min(100, len(rows[columns[0]]))
        for i in range(num_rows):
            rows_html += f'<div class="lq-row" style="top:{i * 32}px">'
            for col in columns:
                val = rows.get(col, [])[i] if i < len(rows.get(col, [])) else None
                col_type = column_types.get(col, {})

                if val is None:
                    rows_html += '<span class="lq-cell"></span>'
                elif col_type.get('type') == 'vector' and isinstance(val, (list, tuple)):
                    # Render sparkline for vectors
                    sparkline = _render_sparkline_svg(val)
                    dim = col_type.get('dim', len(val))
                    rows_html += f'<span class="lq-cell lq-vector-cell">{sparkline}<span class="lq-dim">{dim}d</span></span>'
                elif col in image_cols and isinstance(val, str):
                    short_url = val[:30] + '...' if len(val) > 30 else val
                    rows_html += f'<span class="lq-cell lq-img-cell">{_escape_html(short_url)}'
                    rows_html += f'<span class="lq-img-preview"><img src="{_escape_html(val)}" loading="lazy"></span>'
                    rows_html += '</span>'
                else:
                    rows_html += f'<span class="lq-cell">{_escape_html(str(val))}</span>'
            rows_html += '</div>'

    spacer_height = min(100, total) * 32

    return f'''
<div id="lq-{source_id}" class="lanceql-virtual-table">
<style>
.lanceql-virtual-table {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; }}
.lq-container {{ border: 1px solid #e0e0e0; border-radius: 4px; overflow: hidden; }}
.lq-header {{ display: flex; background: #f5f5f5; font-weight: 600; border-bottom: 1px solid #e0e0e0; }}
.lq-header .lq-cell {{ padding: 8px 12px; display: flex; align-items: center; }}
.lq-scroll {{ height: 400px; overflow-y: auto; }}
.lq-spacer {{ position: relative; }}
.lq-row {{ display: flex; border-bottom: 1px solid #f0f0f0; position: absolute; width: 100%; box-sizing: border-box; }}
.lq-row:hover {{ background: #f9f9f9; }}
.lq-cell {{ flex: 1; padding: 6px 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 80px; position: relative; }}
.lq-cell.lq-img-cell {{ cursor: pointer; color: #0066cc; }}
.lq-cell.lq-img-cell:hover {{ text-decoration: underline; }}
.lq-img-preview {{ display: none; position: absolute; z-index: 1000; background: white; border: 1px solid #ccc; border-radius: 4px; padding: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); left: 0; top: 100%; }}
.lq-cell.lq-img-cell:hover .lq-img-preview {{ display: block; }}
.lq-img-preview img {{ max-height: 150px; max-width: 200px; }}
.lq-vector-cell {{ display: flex; align-items: center; gap: 6px; }}
.lq-dim {{ font-size: 10px; color: #666; background: #f0f0f0; padding: 1px 4px; border-radius: 2px; }}
.lq-status {{ padding: 4px 12px; background: #fafafa; color: #666; font-size: 11px; border-top: 1px solid #e0e0e0; }}
.lq-note {{ padding: 4px 12px; background: #fff3cd; color: #856404; font-size: 11px; border-top: 1px solid #ffc107; }}
</style>
<div class="lq-container">
  <div class="lq-header">{header_html}</div>
  <div class="lq-scroll">
    <div class="lq-spacer" style="height: {spacer_height}px;">
      <div class="lq-rows">{rows_html}</div>
    </div>
  </div>
  <div class="lq-status">{total:,} rows (showing first {min(100, total)})</div>
  <div class="lq-note">Install lanceql-jupyterlab extension for full virtual scrolling</div>
</div>
</div>
'''


def _escape_html(s: str) -> str:
    """Escape HTML special characters."""
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def _register_comm_handler():
    """Register Comm target for frontend data requests."""
    try:
        ip = get_ipython()  # noqa: F821

        def handle_open(comm, open_msg):
            def handle_msg(msg):
                data = msg['content']['data']
                if data.get('action') == 'fetch_rows':
                    source = _sources.get(data['source_id'])
                    if source:
                        # Handle batch requests - multiple ranges
                        ranges = data.get('ranges', [])
                        chunks = []
                        for r in ranges:
                            rows = source.get_rows(r['offset'], r['limit'])
                            chunks.append({'offset': r['offset'], 'rows': rows})

                        comm.send({
                            'source_id': data['source_id'],
                            'total': source.total_rows,
                            'chunks': chunks,
                        })
            comm.on_msg(handle_msg)

        ip.kernel.comm_manager.register_target('lanceql_display', handle_open)
    except Exception:
        pass


def register_display_hooks():
    """Register global IPython formatters for table types."""
    try:
        ip = get_ipython()  # noqa: F821

        # Use _repr_mimebundle_ approach - works with JupyterLab MIME renderer
        # Register formatters that return VirtualTableBundle

        # Register for Polars DataFrame
        try:
            import polars as pl
            ip.display_formatter.formatters['text/html'].for_type(
                pl.DataFrame, lambda obj: _render_static_html(VirtualTableBundle(obj)._bundle)
            )
            # Also register mimebundle formatter
            ip.display_formatter.mimebundle_formatter.for_type(
                pl.DataFrame, lambda obj, **kwargs: VirtualTableBundle(obj)._repr_mimebundle_(**kwargs)
            )
        except ImportError:
            pass

        # Register for PyArrow Table
        try:
            import pyarrow as pa
            ip.display_formatter.formatters['text/html'].for_type(
                pa.Table, lambda obj: _render_static_html(VirtualTableBundle(obj)._bundle)
            )
            ip.display_formatter.mimebundle_formatter.for_type(
                pa.Table, lambda obj, **kwargs: VirtualTableBundle(obj)._repr_mimebundle_(**kwargs)
            )
        except ImportError:
            pass

        # Register for Pandas DataFrame
        try:
            import pandas as pd
            ip.display_formatter.formatters['text/html'].for_type(
                pd.DataFrame, lambda obj: _render_static_html(VirtualTableBundle(obj)._bundle)
            )
            ip.display_formatter.mimebundle_formatter.for_type(
                pd.DataFrame, lambda obj, **kwargs: VirtualTableBundle(obj)._repr_mimebundle_(**kwargs)
            )
        except ImportError:
            pass

        # Register Comm handler for lazy loading
        _register_comm_handler()

    except NameError:
        # Not in IPython
        pass


# Auto-register on import
register_display_hooks()
