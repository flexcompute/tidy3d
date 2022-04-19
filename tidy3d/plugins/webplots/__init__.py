"""Import post run visualization app and Simulation plotting through plotly."""

from ...log import Tidy3dImportError

# try to get the plotly packages, otherwise print a helpful error message.
try:
    from jupyter_dash import JupyterDash
    from dash import Dash
    import plotly.graph_objects as go
except ImportError as e:
    raise Tidy3dImportError(
        "Could not import plotly requirements. "
        "Ensure that tidy3d is installed with [plotly] requirements specified. "
        '``pip install "tidy3d-beta[plotly]" or `pip install -e ".[plotly]". '
        "Or, install the dependencies directly with `pip install -r requirements/plotly.txt`"
    ) from e

from .app import SimulationDataApp
from .simulation import SimulationPlotly
