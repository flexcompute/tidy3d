"""Concrete 2D geometry primitives."""

from __future__ import annotations

import pydantic.v1 as pydantic
import shapely

from tidy3d.components.geometry.base import Box, Geometry
from tidy3d.components.geometry.geometry2d.base import Bound2D, Geometry2D
from tidy3d.components.geometry.polyslab import PolySlab
from tidy3d.components.geometry.primitives import Cylinder
from tidy3d.components.types import Axis, Coordinate2D, Shapely


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

    def to_3d_geometry(
        self,
        slab_bounds: tuple[float, float],
        axis: Axis = 2,
        sidewall_angle: float = 0.0,
    ) -> Geometry:
        """Convert to a 3D Cylinder geometry.

        Parameters
        ----------
        slab_bounds : tuple[float, float]
            (z_min, z_max) bounds for the extruded geometry.
        axis : Axis
            Axis perpendicular to the 2D plane. Default is 2 (z-axis).
        sidewall_angle : float
            Angle of the sidewall in radians. Default is 0.0.

        Returns
        -------
        Cylinder
            A 3D Cylinder geometry.
        """
        z_min, z_max = slab_bounds
        z_center = (z_min + z_max) / 2.0
        length = z_max - z_min

        # Build 3D center: insert z_center at the axis position
        cx, cy = self.center
        center_3d = list((cx, cy))
        center_3d.insert(axis, z_center)

        return Cylinder(
            center=tuple(center_3d),
            radius=self.radius,
            length=length,
            axis=axis,
            sidewall_angle=sidewall_angle,
        )


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

    def to_3d_geometry(
        self,
        slab_bounds: tuple[float, float],
        axis: Axis = 2,
        sidewall_angle: float = 0.0,
    ) -> Geometry:
        """Convert to a 3D Box or PolySlab geometry.

        Uses Box when sidewall_angle is 0 (more efficient).
        Uses PolySlab when sidewall_angle is non-zero.

        Parameters
        ----------
        slab_bounds : tuple[float, float]
            (z_min, z_max) bounds for the extruded geometry.
        axis : Axis
            Axis perpendicular to the 2D plane. Default is 2 (z-axis).
        sidewall_angle : float
            Angle of the sidewall in radians. Default is 0.0.

        Returns
        -------
        Box or PolySlab
            A 3D geometry.
        """
        z_min, z_max = slab_bounds
        z_center = (z_min + z_max) / 2.0
        thickness = z_max - z_min

        cx, cy = self.center
        w, h = self.size

        if sidewall_angle == 0.0:
            # Use Box for efficiency when no sidewall angle
            # Build 3D center and size based on axis
            center_3d = list((cx, cy))
            center_3d.insert(axis, z_center)
            size_3d = list((w, h))
            size_3d.insert(axis, thickness)
            return Box(center=tuple(center_3d), size=tuple(size_3d))

        # Use PolySlab for non-zero sidewall angle
        # Get rectangle vertices in 2D (counter-clockwise)
        (min_x, min_y), (max_x, max_y) = self.bounds_2d
        vertices = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ]
        return PolySlab(
            vertices=vertices,
            slab_bounds=slab_bounds,
            axis=axis,
            sidewall_angle=sidewall_angle,
        )

