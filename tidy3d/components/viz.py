# pylint: disable=invalid-name
""" utilities for plotting """
from typing import Any
from functools import wraps

import matplotlib.pylab as plt
import pydantic as pd
import plotly.graph_objects as go

from .types import Ax, PlotlyFig
from .base import Tidy3dBaseModel

""" Constants """

# add this around extents of plots
PLOT_BUFFER = 0.3

ARROW_COLOR_MONITOR = "orange"
ARROW_COLOR_SOURCE = "green"
ARROW_COLOR_POLARIZATION = "brown"
ARROW_ALPHA = 0.8


# this times the min of axis height and width gives the arrow length
ARROW_LENGTH_FACTOR = 0.4

# this times ARROW_LENGTH gives width
ARROW_WIDTH_FACTOR = 0.4


""" Decorators """


def make_ax() -> Ax:
    """makes an empty `ax`."""
    _, ax = plt.subplots(1, 1, tight_layout=True)
    return ax


def add_ax_if_none(plot):
    """Decorates `plot(*args, **kwargs, ax=None)` function.
    if ax=None in the function call, creates an ax and feeds it to rest of function.
    """

    @wraps(plot)
    def _plot(*args, **kwargs) -> Ax:
        """New plot function using a generated ax if None."""
        if kwargs.get("ax") is None:
            ax = make_ax()
            kwargs["ax"] = ax
        return plot(*args, **kwargs)

    return _plot


def equal_aspect(plot):
    """Decorates a plotting function returning a matplotlib axes.
    Ensures the aspect ratio of the returned axes is set to equal.
    Useful for 2D plots, like sim.plot() or sim_data.plot_fields()
    """

    @wraps(plot)
    def _plot(*args, **kwargs) -> Ax:
        """New plot function with equal aspect ratio axes returned."""
        ax = plot(*args, **kwargs)
        ax.set_aspect("equal")
        return ax

    return _plot


def make_fig() -> PlotlyFig:
    """makes an empty `fig`."""
    fig = go.Figure()
    return fig


def add_fig_if_none(plotly):
    """Decorates `plot(*args, **kwargs, ax=None)` function.
    if fig=None in the function call, creates a plotly fig and feeds it to rest of function.
    """

    @wraps(plotly)
    def _plotly(*args, **kwargs) -> PlotlyFig:
        """New plot function using a generated ax if None."""
        if kwargs.get("fig") is None:
            fig = make_fig()
            kwargs["fig"] = fig
        return plotly(*args, **kwargs)

    return _plotly

def equal_aspect_plotly(plotly):
    """Decorates a plotting function returning a matplotlib axes.
    Ensures the aspect ratio of the returned axes is set to equal.
    Useful for 2D plots, like sim.plotly()
    """

    @wraps(plotly)
    def _plotly(*args, **kwargs) -> PlotlyFig:
        """New plot function with equal aspect ratio axes returned."""
        fig = plotly(*args, **kwargs)
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        return fig

    return _plotly


""" plot parameters """

class PlotParams(Tidy3dBaseModel):
    """Stores plotting parameters / specifications for a given model."""

    alpha: Any = pd.Field(None, title="Opacity")
    edgecolor: Any = pd.Field(None, title="Edge Color", alias="ec")
    facecolor: Any = pd.Field(None, title="Face Color", alias="fc")
    fill: bool = pd.Field(True, title="Is Filled")
    hatch: str = pd.Field(None, title="Hatch Style")
    linewidth: pd.NonNegativeFloat = pd.Field(1, title="Line Width", alias="lw")

# defaults for different tidy3d objects
plot_params_geometry = PlotParams()
plot_params_structure = PlotParams()
plot_params_source = PlotParams(alpha=0.4, facecolor="limegreen", edgecolor="limegreen", lw=3)
plot_params_monitor = PlotParams(alpha=0.4, facecolor="orange", edgecolor="orange", lw=3)
plot_params_pml = PlotParams(alpha=0.7, facecolor="gray", edgecolor="gray", hatch="x")
plot_params_symmetry = PlotParams(alpha=0.6)
plot_params_sim_boundary = PlotParams(linewidth=0, facecolor='rgba(0,0,0,0.05)', edgecolor="black")

# stores color of simulation.structures for given index in simulation.medium_map
MEDIUM_CMAP = [
    "#689DBC",
    "#D0698E",
    "#5E6EAD",
    "#C6224E",
    "#BDB3E2",
    "#9EC3E0",
    "#616161",
    "#877EBC",
]

""" Data viz backend """
# import plotly.express as px
# from .data import SimulationData

# sim_data = SimulationData.from_file('data.hdf5')
# ex = sim_data['fields'].Ex.real
# fig = px.imshow(ex.sel(f=200e12).T, animation_frame=0, labels=dict(animation_frame="slice"));
# fig.write_json(fig.json)
# fig.show()

