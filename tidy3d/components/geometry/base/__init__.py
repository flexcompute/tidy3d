"""Base geometry classes and compatibility exports.

The implementation is split by responsibility while keeping the historical
tidy3d.components.geometry.base import path intact.
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

# isort: split
# GeometryType depends on the concrete geometry classes from the sibling
# geometry modules, so it is imported only after those classes are available.
from tidy3d.components.geometry.utils import (
    GeometryType,
    from_shapely,
    vertices_from_shapely,
)

# isort: split
from . import array as _array
from . import clip_operation as _clip_operation
from . import geometry_group as _geometry_group
from . import transformed as _transformed

_types_namespace = {"GeometryType": GeometryType}
for _module in (_array, _clip_operation, _geometry_group, _transformed):
    _module.GeometryType = GeometryType

for _model in (GeometryArray, ClipOperation, GeometryGroup, Transformed):
    _model.model_rebuild(force=True, _types_namespace=_types_namespace)

del (
    _array,
    _clip_operation,
    _geometry_group,
    _transformed,
    _model,
    _module,
    _types_namespace,
)

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
    "GeometryType",
    "Planar",
    "SimplePlaneIntersection",
    "Transformed",
    "assert_geometry_finite",
    "check_transform_invertible",
    "cleanup_shapely_object",
    "from_shapely",
    "vertices_from_shapely",
]
