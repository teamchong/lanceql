# lanceql-jupyterlab

JupyterLab extension for virtual scrolling DataFrames with lazy loading.

## Features

- **Virtual scrolling**: Only renders visible rows, handles millions of rows smoothly
- **Lazy loading**: Fetches rows on demand via Jupyter Comm as you scroll
- **Global hook**: Just `import metal0.lanceql.display` - all DataFrames auto-virtualized
- **Image preview**: URL columns show image thumbnail on hover
- **Dark mode**: Supports JupyterLab theme

## Installation

```bash
pip install lanceql-jupyterlab
```

Or for development:

```bash
cd lanceql-jupyterlab
pip install -e ".[dev]"
jupyter labextension develop . --overwrite
```

## Usage

```python
# Enable virtual scrolling for all tables
import metal0.lanceql.display

# Now any DataFrame uses virtual scrolling
import polars as pl
df = pl.DataFrame({'id': range(1_000_000), 'value': range(1_000_000)})
df  # Virtual scrolling with lazy loading
```

## How it works

1. **Python side** (`metal0.lanceql.display`):
   - Registers IPython formatters for Polars, PyArrow, Pandas
   - Outputs custom MIME type `application/vnd.lanceql.table+json`
   - Registers Comm handler to serve row data on demand

2. **Extension side** (this package):
   - Registers MIME renderer for the custom MIME type
   - Renders virtual scrolling table UI
   - Uses Jupyter Comm to fetch rows as user scrolls

## Development

```bash
# Install dependencies
jlpm install

# Build extension
jlpm build

# Watch for changes
jlpm watch
```

## Requirements

- JupyterLab >= 4.0.0
- Python >= 3.8
