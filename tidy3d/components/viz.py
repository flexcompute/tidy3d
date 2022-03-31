# pylint: disable=invalid-name
""" utilities for plotting """
from typing import Any
from functools import wraps

import matplotlib.pylab as plt
import pydantic as pd

from .types import Ax
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

# arrow width cannot be larger than this factor times the max of axis height and width
MAX_ARROW_WIDTH_FACTOR = 0.02


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


""" plot parameters """


class PlotParams(Tidy3dBaseModel):
    """Stores plotting parameters / specifications for a given model."""

    alpha: Any = pd.Field(1.0, title="Opacity")
    edgecolor: Any = pd.Field(None, title="Edge Color", alias="ec")
    facecolor: Any = pd.Field(None, title="Face Color", alias="fc")
    fill: bool = pd.Field(True, title="Is Filled")
    hatch: str = pd.Field(None, title="Hatch Style")
    linewidth: pd.NonNegativeFloat = pd.Field(1, title="Line Width", alias="lw")

    def include_kwargs(self, **kwargs) -> "PlotParams":
        """Update the plot params with supplied kwargs."""
        new_plot_params = self.copy(deep=True)
        for key, value in kwargs.items():
            if key not in ("type",):
                setattr(new_plot_params, key, value)
        return new_plot_params

    def to_kwargs(self) -> dict:
        """Export the plot parameters as kwargs dict that can be supplied to plot function."""
        kwarg_dict = self.dict()
        for ignore_key in ("type",):
            kwarg_dict.pop(ignore_key)
        return kwarg_dict


# defaults for different tidy3d objects
plot_params_geometry = PlotParams()
plot_params_structure = PlotParams()
plot_params_source = PlotParams(alpha=0.4, facecolor="limegreen", edgecolor="limegreen", lw=3)
plot_params_monitor = PlotParams(alpha=0.4, facecolor="orange", edgecolor="orange", lw=3)
plot_params_pml = PlotParams(alpha=0.7, facecolor="gray", edgecolor="gray", hatch="x")
plot_params_symmetry = PlotParams(edgecolor="gray", facecolor="gray", alpha=0.6)

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
