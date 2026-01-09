"""Compatibility shim for :mod:`tidy3d._common.components.viz.plot_params`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.viz.plot_params import (
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
    plot_params_monitor,
    plot_params_override_structures,
    plot_params_pec,
    plot_params_pmc,
    plot_params_pml,
    plot_params_source,
    plot_params_structure,
    plot_params_symmetry,
)
