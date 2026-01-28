"""2D geometry primitives for layer-based layouts."""

from typing import Union

from tidy3d.components.geometry.geometry2d.base import Geometry2D
from tidy3d.components.geometry.geometry2d.polygon import Polygon2D
from tidy3d.components.geometry.geometry2d.primitives import Circle2D, Rectangle2D

# Union of all concrete Geometry2D types for pydantic discriminated union
# Update this when adding new Geometry2D subclasses
Geometry2DType = Union[
    Circle2D,
    Polygon2D,
    Rectangle2D,
]

# Resolve forward references for Polygon2D.holes which uses "Geometry2DType"
Polygon2D.update_forward_refs(Geometry2DType=Geometry2DType)

__all__ = [
    "Geometry2D",
    "Geometry2DType",
    "Circle2D",
    "Polygon2D",
    "Rectangle2D",
]

