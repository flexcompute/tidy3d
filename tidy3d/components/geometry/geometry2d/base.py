"""Abstract base class for 2D geometry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import Axis, Coordinate2D, Shapely

if TYPE_CHECKING:
    from tidy3d.components.geometry.base import Geometry

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

    @abstractmethod
    def to_3d_geometry(
        self,
        slab_bounds: tuple[float, float],
        axis: Axis = 2,
        sidewall_angle: float = 0.0,
    ) -> "Geometry":
        """Convert to a 3D tidy3d Geometry by extrusion.

        Parameters
        ----------
        slab_bounds : tuple[float, float]
            (z_min, z_max) bounds for the extruded geometry.
        axis : Axis
            Axis perpendicular to the 2D plane (extrusion direction).
            Default is 2 (z-axis).
        sidewall_angle : float
            Angle of the sidewall in radians. Positive values create
            a narrower top than bottom. Default is 0.0.

        Returns
        -------
        Geometry
            A 3D tidy3d Geometry (Cylinder, Box, PolySlab, etc.).
        """

