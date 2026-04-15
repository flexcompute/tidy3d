from __future__ import annotations

from tidy3d.components.geometry.contour_conversion import smooth_polygon_vertices

from .contours import (
    curvature_penalty,
)
from .filters import (
    CircularFilter,
    ConicFilter,
    GaussianFilter,
    make_circular_filter,
    make_conic_filter,
    make_filter,
    make_gaussian_filter,
)
from .misc import grey_indicator
from .parametrizations import (
    FilterAndProject,
    initialize_params_from_simulation,
    make_filter_and_project,
)
from .penalties import ErosionDilationPenalty, make_curvature_penalty, make_erosion_dilation_penalty
from .polyslab_set import PolySlabSet
from .projections import ramp_projection, smoothed_projection, tanh_projection
from .symmetries import symmetrize_diagonal, symmetrize_mirror, symmetrize_rotation

__all__ = [
    "CircularFilter",
    "ConicFilter",
    "ErosionDilationPenalty",
    "FilterAndProject",
    "GaussianFilter",
    "PolySlabSet",
    "curvature_penalty",
    "grey_indicator",
    "initialize_params_from_simulation",
    "make_circular_filter",
    "make_conic_filter",
    "make_curvature_penalty",
    "make_erosion_dilation_penalty",
    "make_filter",
    "make_filter_and_project",
    "make_gaussian_filter",
    "ramp_projection",
    "smooth_polygon_vertices",
    "smoothed_projection",
    "symmetrize_diagonal",
    "symmetrize_mirror",
    "symmetrize_rotation",
    "tanh_projection",
]
