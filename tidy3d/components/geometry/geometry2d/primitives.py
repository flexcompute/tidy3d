"""Concrete 2D geometry primitives."""

from __future__ import annotations

import pydantic.v1 as pydantic
import shapely

from tidy3d.components.geometry.geometry2d.base import Bound2D, Geometry2D
from tidy3d.components.types import Coordinate2D, Shapely


class Circle2D(Geometry2D):
    """A circle defined by center and radius.

    Example
    -------
    >>> circle = Circle2D(center=(1.0, 2.0), radius=0.5)
    """

    center: Coordinate2D = pydantic.Field(
        (0.0, 0.0),
        title="Center",
        description="Center coordinate of the circle in the 2D plane.",
    )

    radius: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Radius",
        description="Radius of the circle.",
        units="um",
    )

    @property
    def bounds_2d(self) -> Bound2D:
        """Returns the 2D bounding box of the circle."""
        cx, cy = self.center
        r = self.radius
        return ((cx - r, cy - r), (cx + r, cy + r))

    def to_shapely(self) -> Shapely:
        """Convert to a shapely Point buffered to a circle polygon."""
        point = shapely.Point(self.center)
        return point.buffer(self.radius)


class Rectangle2D(Geometry2D):
    """A rectangle defined by center and size.

    Example
    -------
    >>> rect = Rectangle2D(center=(0.0, 0.0), size=(2.0, 1.0))
    """

    center: Coordinate2D = pydantic.Field(
        (0.0, 0.0),
        title="Center",
        description="Center coordinate of the rectangle in the 2D plane.",
    )

    size: tuple[pydantic.PositiveFloat, pydantic.PositiveFloat] = pydantic.Field(
        ...,
        title="Size",
        description="Size of the rectangle as (width, height).",
        units="um",
    )

    @property
    def bounds_2d(self) -> Bound2D:
        """Returns the 2D bounding box of the rectangle."""
        cx, cy = self.center
        half_w = self.size[0] / 2.0
        half_h = self.size[1] / 2.0
        return ((cx - half_w, cy - half_h), (cx + half_w, cy + half_h))

    def to_shapely(self) -> Shapely:
        """Convert to a shapely box polygon."""
        (min_x, min_y), (max_x, max_y) = self.bounds_2d
        return shapely.box(min_x, min_y, max_x, max_y)

