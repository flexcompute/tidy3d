from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from numpy import inf
from pydantic import Field, NonNegativeFloat

from tidy3d.components.base import Tidy3dBaseModel

if TYPE_CHECKING:
    from tidy3d.components.viz.visualization_spec import VisualizationSpec


class AbstractPlotParams(Tidy3dBaseModel):
    """Abstract class for storing plotting parameters.
    Corresponds with select properties of ``matplotlib.artist.Artist``.
    """

    alpha: Any = Field(1.0, title="Opacity")
    zorder: Optional[float] = Field(None, title="Display Order")

    def include_kwargs(self, **kwargs: Any) -> AbstractPlotParams:
        """Update the plot params with supplied kwargs."""
        update_dict = {
            key: value
            for key, value in kwargs.items()
            if key not in ("type",) and value is not None and key in type(self).model_fields
        }
        return self.copy(update=update_dict)

    def override_with_viz_spec(self, viz_spec: VisualizationSpec) -> AbstractPlotParams:
        """Override plot params with supplied VisualizationSpec."""
        return self.include_kwargs(**dict(viz_spec))

    def to_kwargs(self) -> dict[str, Any]:
        """Export the plot parameters as kwargs dict that can be supplied to plot function."""
        kwarg_dict = self.model_dump()
        for ignore_key in ("type", "attrs"):
            kwarg_dict.pop(ignore_key)
        return kwarg_dict


class PathPlotParams(AbstractPlotParams):
    """Stores plotting parameters / specifications for a path.
    Corresponds with select properties of ``matplotlib.lines.Line2D``.
    """

    color: Optional[Any] = Field(None, title="Color", alias="c")
    linewidth: NonNegativeFloat = Field(2, title="Line Width", alias="lw")
    linestyle: str = Field("--", title="Line Style", alias="ls")
    marker: Any = Field("o", title="Marker Style")
    markeredgecolor: Optional[Any] = Field(None, title="Marker Edge Color", alias="mec")
    markerfacecolor: Optional[Any] = Field(None, title="Marker Face Color", alias="mfc")
    markersize: NonNegativeFloat = Field(10, title="Marker Size", alias="ms")


class PlotParams(AbstractPlotParams):
    """Stores plotting parameters / specifications for a given model.
    Corresponds with select properties of ``matplotlib.patches.Patch``.
    """

    edgecolor: Optional[Any] = Field(None, title="Edge Color", alias="ec")
    facecolor: Optional[Any] = Field(None, title="Face Color", alias="fc")
    fill: bool = Field(True, title="Is Filled")
    hatch: Optional[str] = Field(None, title="Hatch Style")
    linewidth: NonNegativeFloat = Field(1, title="Line Width", alias="lw")


# defaults for different tidy3d objects
plot_params_geometry = PlotParams()
plot_params_structure = PlotParams()
plot_params_source = PlotParams(alpha=0.4, facecolor="limegreen", edgecolor="limegreen", lw=3)
plot_params_absorber = PlotParams(
    alpha=0.4, facecolor="lightskyblue", edgecolor="lightskyblue", lw=3
)
plot_params_monitor = PlotParams(alpha=0.4, facecolor="orange", edgecolor="orange", lw=3)
plot_params_pml = PlotParams(alpha=0.7, facecolor="gray", edgecolor="gray", hatch="x", zorder=inf)
plot_params_pec = PlotParams(alpha=1.0, facecolor="gold", edgecolor="black", zorder=inf)
plot_params_pmc = PlotParams(alpha=1.0, facecolor="lightsteelblue", edgecolor="black", zorder=inf)
plot_params_bloch = PlotParams(alpha=1.0, facecolor="orchid", edgecolor="black", zorder=inf)
plot_params_abc = PlotParams(alpha=1.0, facecolor="lightskyblue", edgecolor="black", zorder=inf)
plot_params_symmetry = PlotParams(edgecolor="gray", facecolor="gray", alpha=0.6, zorder=inf)
plot_params_override_structures = PlotParams(
    linewidth=0.4, edgecolor="black", fill=False, zorder=inf
)
plot_params_fluid = PlotParams(facecolor="white", edgecolor="lightsteelblue", lw=0.4, hatch="xx")
plot_params_grid = PlotParams(edgecolor="black", lw=0.2)
plot_params_lumped_element = PlotParams(
    alpha=0.4, facecolor="mediumblue", edgecolor="mediumblue", lw=3
)
plot_params_min_grid_size = PlotParams(
    alpha=0.5, facecolor="gray", edgecolor="darkred", lw=0, fill=True, hatch=".", zorder=0
)
