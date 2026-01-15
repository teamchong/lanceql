"""LanceQL JupyterLab extension for virtual scrolling DataFrames."""

__version__ = "0.1.0"


def _jupyter_labextension_paths():
    """Return the JupyterLab extension paths."""
    return [{
        "src": "labextension",
        "dest": "lanceql-jupyterlab"
    }]
