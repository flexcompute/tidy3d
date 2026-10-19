"""Base geometry classes.

The implementation is split by responsibility across this package's modules;
this module exports their public names.
"""

from __future__ import annotations

from .abstract_primitives import Centered, Circular, Planar, SimplePlaneIntersection
from .array import GeometryArray
from .box import Box
from .clip_operation import ClipOperation
from .constants import (
    GDS_MAX_COORDINATE_INDEX,
    LINEAR_TRANSFORM_TOL,
    POLY_DISTANCE_TOLERANCE,
    POLY_GRID_SIZE,
    POLY_TOLERANCE_RATIO,
)
from .core import (
    _BOX_FACES,
    Geometry,
    assert_geometry_finite,
    check_transform_invertible,
)
from .geometry_group import GeometryGroup
from .transformed import Transformed
from .utils import cleanup_shapely_object

__all__ = [
    "GDS_MAX_COORDINATE_INDEX",
    "LINEAR_TRANSFORM_TOL",
    "POLY_DISTANCE_TOLERANCE",
    "POLY_GRID_SIZE",
    "POLY_TOLERANCE_RATIO",
    "_BOX_FACES",
    "Box",
    "Centered",
    "Circular",
    "ClipOperation",
    "Geometry",
    "GeometryArray",
    "GeometryGroup",
    "Planar",
    "SimplePlaneIntersection",
    "Transformed",
    "assert_geometry_finite",
    "check_transform_invertible",
    "cleanup_shapely_object",
]
