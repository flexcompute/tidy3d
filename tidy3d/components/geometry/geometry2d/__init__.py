"""2D geometry primitives for layer-based layouts."""

from typing import Union

from tidy3d.components.geometry.geometry2d.base import Geometry2D
from tidy3d.components.geometry.geometry2d.primitives import Circle2D, Rectangle2D

# Union of all concrete Geometry2D types for pydantic discriminated union
# Update this when adding new Geometry2D subclasses
Geometry2DType = Union[
    Circle2D,
    Rectangle2D,
]

__all__ = [
    "Geometry2D",
    "Geometry2DType",
    "Circle2D",
    "Rectangle2D",
]

