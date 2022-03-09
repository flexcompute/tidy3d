# pylint: disable=invalid-name
""" utilities for plotting """
from typing import Any
from functools import wraps

import matplotlib.pylab as plt
from pydantic import BaseModel
import plotly.graph_objects as go

from .types import Ax, PlotlyFig

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


def add_fig_if_none(plot):
    """Decorates `plot(*args, **kwargs, ax=None)` function.
    if fig=None in the function call, creates a plotly fig and feeds it to rest of function.
    """

    @wraps(plot)
    def _plot(*args, **kwargs) -> PlotlyFig:
        """New plot function using a generated ax if None."""
        if kwargs.get("fig") is None:
            fig = make_fig()
            kwargs["fig"] = fig
        return plot(*args, **kwargs)

    return _plot


""" plot parameters """


class PlotParams(BaseModel):
    """Stores plotting parameters / specifications for a given model."""

    alpha: Any = None
    edgecolor: Any = None
    facecolor: Any = None
    fill: bool = True
    hatch: str = None
    lw: int = 1


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
