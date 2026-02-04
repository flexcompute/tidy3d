"""2D geometry primitives for layer-based layouts."""

from typing import Union

from tidy3d.components.geometry.geometry2d.base import Geometry2D
from tidy3d.components.geometry.geometry2d.composites import Array2D, Transformed2D
from tidy3d.components.geometry.geometry2d.path import ArcSegment, Path2D
from tidy3d.components.geometry.geometry2d.polygon import Polygon2D
from tidy3d.components.geometry.geometry2d.primitives import Circle2D, Rectangle2D

# Union of all concrete Geometry2D types for pydantic discriminated union
# Update this when adding new Geometry2D subclasses
# Note: Transformed2D and Array2D must come after primitives since they
# reference Geometry2DType in their field annotations
Geometry2DType = Union[
    Array2D,
    Circle2D,
    Path2D,
    Polygon2D,
    Rectangle2D,
    Transformed2D,
]

# Resolve forward references for classes that use "Geometry2DType"
Array2D.update_forward_refs(Geometry2DType=Geometry2DType)
Polygon2D.update_forward_refs(Geometry2DType=Geometry2DType)
Transformed2D.update_forward_refs(Geometry2DType=Geometry2DType)

__all__ = [
    "ArcSegment",
    "Array2D",
    "Geometry2D",
    "Geometry2DType",
    "Circle2D",
    "Path2D",
    "Polygon2D",
    "Rectangle2D",
    "Transformed2D",
]

