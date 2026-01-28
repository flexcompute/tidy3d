"""Abstract base class for 2D geometry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import Coordinate2D, Shapely

# Type alias for 2D bounding box: ((min_x, min_y), (max_x, max_y))
Bound2D = tuple[Coordinate2D, Coordinate2D]


class Geometry2D(Tidy3dBaseModel, ABC):
    """Abstract base class for 2D geometry shapes.

    This serves as the foundation for representing 2D shapes that can be
    used in layer-based layouts (PCB, GDS, ODB++). Shapes are stored in
    their analytical/compact form and converted to shapely geometries
    when needed for boolean operations or 3D conversion.
    """

    @property
    @abstractmethod
    def bounds_2d(self) -> Bound2D:
        """Returns the 2D bounding box of the geometry.

        Returns
        -------
        Bound2D
            Tuple of ((min_x, min_y), (max_x, max_y)) coordinates.
        """

    @abstractmethod
    def to_shapely(self) -> Shapely:
        """Convert to a shapely geometry object.

        Returns
        -------
        Shapely
            A shapely geometry (Point, Polygon, MultiPolygon, etc.)
            representing this 2D shape.
        """

