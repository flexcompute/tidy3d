from __future__ import annotations

from typing import Any

import pydantic.v1 as pd
from numpy import inf

from tidy3d.components.base import Tidy3dBaseModel


class AbstractPlotParams(Tidy3dBaseModel):
    """Abstract class for storing plotting parameters.
    Corresponds with select properties of ``matplotlib.artist.Artist``.
    """

    alpha: Any = pd.Field(1.0, title="Opacity")
    zorder: float = pd.Field(None, title="Display Order")

    def include_kwargs(self, **kwargs) -> AbstractPlotParams:
        """Update the plot params with supplied kwargs."""
        update_dict = {
            key: value
            for key, value in kwargs.items()
            if key not in ("type",) and value is not None and key in self.__fields__
        }
        return self.copy(update=update_dict)

    def override_with_viz_spec(self, viz_spec) -> AbstractPlotParams:
        """Override plot params with supplied VisualizationSpec."""
        return self.include_kwargs(**dict(viz_spec))

    def to_kwargs(self) -> dict:
        """Export the plot parameters as kwargs dict that can be supplied to plot function."""
        kwarg_dict = self.dict()
        for ignore_key in ("type", "attrs"):
            kwarg_dict.pop(ignore_key)
        return kwarg_dict


class PathPlotParams(AbstractPlotParams):
    """Stores plotting parameters / specifications for a path.
    Corresponds with select properties of ``matplotlib.lines.Line2D``.
    """

    color: Any = pd.Field(None, title="Color", alias="c")
    linewidth: pd.NonNegativeFloat = pd.Field(2, title="Line Width", alias="lw")
    linestyle: str = pd.Field("--", title="Line Style", alias="ls")
    marker: Any = pd.Field("o", title="Marker Style")
    markeredgecolor: Any = pd.Field(None, title="Marker Edge Color", alias="mec")
    markerfacecolor: Any = pd.Field(None, title="Marker Face Color", alias="mfc")
    markersize: pd.NonNegativeFloat = pd.Field(10, title="Marker Size", alias="ms")


class PlotParams(AbstractPlotParams):
    """Stores plotting parameters / specifications for a given model.
    Corresponds with select properties of ``matplotlib.patches.Patch``.
    """

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
plot_params_pml = PlotParams(alpha=0.7, facecolor="gray", edgecolor="gray", hatch="x", zorder=inf)
plot_params_pec = PlotParams(alpha=1.0, facecolor="gold", edgecolor="black", zorder=inf)
plot_params_pmc = PlotParams(alpha=1.0, facecolor="lightsteelblue", edgecolor="black", zorder=inf)
plot_params_bloch = PlotParams(alpha=1.0, facecolor="orchid", edgecolor="black", zorder=inf)
plot_params_symmetry = PlotParams(edgecolor="gray", facecolor="gray", alpha=0.6, zorder=inf)
plot_params_override_structures = PlotParams(
    linewidth=0.4, edgecolor="black", fill=False, zorder=inf
)
plot_params_fluid = PlotParams(facecolor="white", edgecolor="lightsteelblue", lw=0.4, hatch="xx")
plot_params_grid = PlotParams(edgecolor="black", lw=0.2)
plot_params_lumped_element = PlotParams(
    alpha=0.4, facecolor="mediumblue", edgecolor="mediumblue", lw=3
)
