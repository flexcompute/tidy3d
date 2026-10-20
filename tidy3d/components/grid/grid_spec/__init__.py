"""Compatibility facade for historical grid_spec imports."""

from __future__ import annotations

from flexcompute.core._migration.em.grid.schema.validation_schema import (
    MIN_GRID_SPACING,
    UNITS_HELP_URL,
)

from tidy3d.components.grid.corner_finder import CornerFinderSpec

from .constants import (
    DEFAULT_REFINEMENT_FACTOR,
    DL_MIN_FROM_GAPS_FRACTION,
    GAP_MESHING_TOL,
    GAP_REFINEMENT_WARNING_THRESH,
    INPLANE_OVERRIDE_UNION_MAX_ITERS,
    MIN_STEP_BOUND_SCALE,
    CornersAndConvexity,
    _GeneratedGridSizeError,
)
from .grid_1d import (
    AbstractAutoGrid,
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridSpec1d,
    GridType,
    QuasiUniformGrid,
    UniformGrid,
)
from .refinement import GridRefinement, LayerRefinementSpec
from .spec import GridSpec

__all__ = [
    "DEFAULT_REFINEMENT_FACTOR",
    "DL_MIN_FROM_GAPS_FRACTION",
    "GAP_MESHING_TOL",
    "GAP_REFINEMENT_WARNING_THRESH",
    "INPLANE_OVERRIDE_UNION_MAX_ITERS",
    "MIN_GRID_SPACING",
    "MIN_STEP_BOUND_SCALE",
    "UNITS_HELP_URL",
    "AbstractAutoGrid",
    "AutoGrid",
    "CornerFinderSpec",
    "CornersAndConvexity",
    "CustomGrid",
    "CustomGridBoundaries",
    "GridRefinement",
    "GridSpec",
    "GridSpec1d",
    "GridType",
    "LayerRefinementSpec",
    "QuasiUniformGrid",
    "UniformGrid",
    "_GeneratedGridSizeError",
]
