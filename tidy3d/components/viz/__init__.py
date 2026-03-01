from __future__ import annotations

from .axes_utils import (
    add_ax_if_none,
    add_plotter_if_none,
    equal_aspect,
    make_ax,
    set_default_labels_and_title,
)
from .descartes import Polygon, polygon_patch, polygon_path
from .flex_style import apply_tidy3d_params, restore_matplotlib_rcparams
from .plot_params import (
    AbstractPlotParams,
    PathPlotParams,
    PlotParams,
    plot_params_abc,
    plot_params_absorber,
    plot_params_bloch,
    plot_params_fluid,
    plot_params_geometry,
    plot_params_grid,
    plot_params_lumped_element,
    plot_params_min_grid_size,
    plot_params_monitor,
    plot_params_override_structures,
    plot_params_pec,
    plot_params_pmc,
    plot_params_pml,
    plot_params_source,
    plot_params_structure,
    plot_params_symmetry,
)
from .plot_sim_3d import plot_scene_3d, plot_sim_3d
from .styles import (
    ARROW_ALPHA,
    ARROW_COLOR_ABSORBER,
    ARROW_COLOR_MONITOR,
    ARROW_COLOR_POLARIZATION,
    ARROW_COLOR_SOURCE,
    ARROW_LENGTH,
    FLEXCOMPUTE_COLORS,
    MEDIUM_CMAP,
    PLOT_BUFFER,
    STRUCTURE_EPS_CMAP,
    STRUCTURE_EPS_CMAP_R,
    STRUCTURE_HEAT_COND_CMAP,
    arrow_style,
)
from .visualization_spec import MATPLOTLIB_IMPORTED, VisualizationSpec

# Note: apply_tidy3d_params() is no longer called at import time to reduce import overhead.
# It will be called automatically when matplotlib is first used for plotting.
_tidy3d_style_applied = False


def _ensure_tidy3d_style() -> None:
    """Apply tidy3d matplotlib style if not already applied."""
    global _tidy3d_style_applied
    if not _tidy3d_style_applied:
        apply_tidy3d_params()
        _tidy3d_style_applied = True


__all__ = [
    "ARROW_ALPHA",
    "ARROW_COLOR_ABSORBER",
    "ARROW_COLOR_MONITOR",
    "ARROW_COLOR_POLARIZATION",
    "ARROW_COLOR_SOURCE",
    "ARROW_LENGTH",
    "FLEXCOMPUTE_COLORS",
    "MATPLOTLIB_IMPORTED",
    "MEDIUM_CMAP",
    "PLOT_BUFFER",
    "STRUCTURE_EPS_CMAP",
    "STRUCTURE_EPS_CMAP_R",
    "STRUCTURE_HEAT_COND_CMAP",
    "AbstractPlotParams",
    "PathPlotParams",
    "PlotParams",
    "Polygon",
    "VisualizationSpec",
    "add_ax_if_none",
    "add_plotter_if_none",
    "arrow_style",
    "equal_aspect",
    "make_ax",
    "plot_params_abc",
    "plot_params_absorber",
    "plot_params_bloch",
    "plot_params_fluid",
    "plot_params_geometry",
    "plot_params_grid",
    "plot_params_lumped_element",
    "plot_params_min_grid_size",
    "plot_params_monitor",
    "plot_params_override_structures",
    "plot_params_pec",
    "plot_params_pmc",
    "plot_params_pml",
    "plot_params_source",
    "plot_params_structure",
    "plot_params_symmetry",
    "plot_scene_3d",
    "plot_sim_3d",
    "polygon_patch",
    "polygon_path",
    "restore_matplotlib_rcparams",
    "set_default_labels_and_title",
]
