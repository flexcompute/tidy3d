"""Concrete primitive geometrical objects."""
from __future__ import annotations

from typing import List
from math import isclose

import pydantic as pydantic
import numpy as np
import shapely

from ..base import cached_property
from ..types import Axis, Bound
from ...exceptions import SetupError, ValidationError
from ...constants import MICROMETER, LARGE_NUMBER

from . import base

# for sampling conical frustum in visualization
_N_SAMPLE_CURVE_SHAPELY = 40

# for shapely circular shapes discretization in visualization
_N_SHAPELY_QUAD_SEGS = 200


class Sphere(base.Centered, base.Circular):
    """Spherical geometry.

    Example
    -------
    >>> b = Sphere(center=(1,2,3), radius=2)
    """

    def inside(
        self, x: np.ndarray[float], y: np.ndarray[float], z: np.ndarray[float]
    ) -> np.ndarray[bool]:
        """For input arrays ``x``, ``y``, ``z`` of arbitrary but identical shape, return an array
        with the same shape which is ``True`` for every point in zip(x, y, z) that is inside the
        volume of the :class:`Geometry`, and ``False`` otherwise.

        Parameters
        ----------
        x : np.ndarray[float]
            Array of point positions in x direction.
        y : np.ndarray[float]
            Array of point positions in y direction.
        z : np.ndarray[float]
            Array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            ``True`` for every point that is inside the geometry.
        """
        self._ensure_equal_shape(x, y, z)
        x0, y0, z0 = self.center
        dist_x = np.abs(x - x0)
        dist_y = np.abs(y - y0)
        dist_z = np.abs(z - z0)
        return (dist_x**2 + dist_y**2 + dist_z**2) <= (self.radius**2)

    def intersections_plane(self, x: float = None, y: float = None, z: float = None):
        """Returns shapely geometry at plane specified by one non None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        if not self.intersects_axis_position(axis, position):
            return []
        z0, (x0, y0) = self.pop_axis(self.center, axis=axis)
        intersect_dist = self._intersect_dist(position, z0)
        if not intersect_dist:
            return []
        return [shapely.Point(x0, y0).buffer(0.5 * intersect_dist, quad_segs=_N_SHAPELY_QUAD_SEGS)]

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float, float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        coord_min = tuple(c - self.radius for c in self.center)
        coord_max = tuple(c + self.radius for c in self.center)
        return (coord_min, coord_max)

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""

        volume = 4.0 / 3.0 * np.pi * self.radius**3

        # a very loose upper bound on how much of sphere is in bounds
        for axis in range(3):
            if self.center[axis] <= bounds[0][axis] or self.center[axis] >= bounds[1][axis]:
                volume *= 0.5

        return volume

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""

        area = 4.0 * np.pi * self.radius**2

        # a very loose upper bound on how much of sphere is in bounds
        for axis in range(3):
            if self.center[axis] <= bounds[0][axis] or self.center[axis] >= bounds[1][axis]:
                area *= 0.5

        return area


class Cylinder(base.Centered, base.Circular, base.Planar):
    """Cylindrical geometry with optional sidewall angle along axis
    direction. When ``sidewall_angle`` is nonzero, the shape is a
    conical frustum or a cone.

    Example
    -------
    >>> c = Cylinder(center=(1,2,3), radius=2, length=5, axis=2)
    """

    # Provide more explanations on where radius is defined
    radius: pydantic.NonNegativeFloat = pydantic.Field(
        ...,
        title="Radius",
        description="Radius of geometry at the ``reference_plane``.",
        units=MICROMETER,
    )

    length: pydantic.NonNegativeFloat = pydantic.Field(
        ...,
        title="Length",
        description="Defines thickness of cylinder along axis dimension.",
        units=MICROMETER,
    )

    @pydantic.validator("length", always=True)
    def _only_middle_for_infinite_length_slanted_cylinder(cls, val, values):
        """For a slanted cylinder of infinite length, ``reference_plane`` can only
        be ``middle``; otherwise, the radius at ``center`` is either td.inf or 0.
        """
        if isclose(values["sidewall_angle"], 0) or not np.isinf(val):
            return val
        if values["reference_plane"] != "middle":
            raise SetupError(
                "For a slanted cylinder here is of infinite length, "
                "defining the reference_plane other than 'middle' "
                "leads to undefined cylinder behaviors near 'center'."
            )
        return val

    @property
    def center_axis(self):
        """Gets the position of the center of the geometry in the out of plane dimension."""
        z0, _ = self.pop_axis(self.center, axis=self.axis)
        return z0

    @property
    def length_axis(self) -> float:
        """Gets the length of the geometry along the out of plane dimension."""
        return self.length

    @cached_property
    def _normal_2dmaterial(self) -> Axis:
        """Get the normal to the given geometry, checking that it is a 2D geometry."""
        if self.length != 0:
            raise ValidationError("'Medium2D' requires the 'Cylinder' length to be zero.")
        return self.axis

    def _intersections_normal(self, z: float):
        """Find shapely geometries intersecting cylindrical geometry with axis normal to slab.

        Parameters
        ----------
        z : float
            Position along the axis normal to slab

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        # radius at z
        radius_offset = self._radius_z(z)

        if radius_offset <= 0:
            return []

        _, (x0, y0) = self.pop_axis(self.center, axis=self.axis)
        return [shapely.Point(x0, y0).buffer(radius_offset, quad_segs=_N_SHAPELY_QUAD_SEGS)]

    def _intersections_side(self, position, axis):
        """Find shapely geometries intersecting cylindrical geometry with axis orthogonal to length.
        When ``sidewall_angle`` is nonzero, so that it's in fact a conical frustum or cone, the
        cross section can contain hyperbolic curves. This is currently approximated by a polygon
        of many vertices.

        Parameters
        ----------
        position : float
            Position along axis direction.
        axis : int
            Integer index into 'xyz' (0, 1, 2).

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        # position in the local coordinate of the cylinder
        position_local = position - self.center[axis]

        # no intersection
        if abs(position_local) >= self.radius_max:
            return []

        # half of intersection length at the top and bottom
        intersect_half_length_max = np.sqrt(self.radius_max**2 - position_local**2)
        intersect_half_length_min = -LARGE_NUMBER
        if abs(position_local) < self.radius_min:
            intersect_half_length_min = np.sqrt(self.radius_min**2 - position_local**2)

        # the vertices on the max side of top/bottom
        # The two vertices are present in all scenarios.
        vertices_max = [
            self._local_to_global_side_cross_section([-intersect_half_length_max, 0], axis),
            self._local_to_global_side_cross_section([intersect_half_length_max, 0], axis),
        ]

        # Extending to a cone, the maximal height of the cone
        h_cone = (
            LARGE_NUMBER if isclose(self.sidewall_angle, 0) else self.radius_max / abs(self._tanq)
        )
        # The maximal height of the cross section
        height_max = min(
            (1 - abs(position_local) / self.radius_max) * h_cone, self.finite_length_axis
        )

        # more vertices to add for conical frustum shape
        vertices_frustum_right = []
        vertices_frustum_left = []
        if not (isclose(position, self.center[axis]) or isclose(self.sidewall_angle, 0)):
            # The y-coordinate for the additional vertices
            y_list = height_max * np.linspace(0, 1, _N_SAMPLE_CURVE_SHAPELY)
            # `abs()` to make sure np.sqrt(0-fp_eps) goes through
            x_list = np.sqrt(
                np.abs(self.radius_max**2 * (1 - y_list / h_cone) ** 2 - position_local**2)
            )
            for i in range(_N_SAMPLE_CURVE_SHAPELY):
                vertices_frustum_right.append(
                    self._local_to_global_side_cross_section([x_list[i], y_list[i]], axis)
                )
                vertices_frustum_left.append(
                    self._local_to_global_side_cross_section(
                        [
                            -x_list[_N_SAMPLE_CURVE_SHAPELY - i - 1],
                            y_list[_N_SAMPLE_CURVE_SHAPELY - i - 1],
                        ],
                        axis,
                    )
                )

        # the vertices on the min side of top/bottom
        vertices_min = []

        ## termination at the top/bottom
        if intersect_half_length_min > 0:
            vertices_min.append(
                self._local_to_global_side_cross_section(
                    [intersect_half_length_min, self.finite_length_axis], axis
                )
            )
            vertices_min.append(
                self._local_to_global_side_cross_section(
                    [-intersect_half_length_min, self.finite_length_axis], axis
                )
            )
        ## early termination
        else:
            vertices_min.append(self._local_to_global_side_cross_section([0, height_max], axis))

        return [
            shapely.Polygon(
                vertices_max + vertices_frustum_right + vertices_min + vertices_frustum_left
            )
        ]

    def inside(
        self, x: np.ndarray[float], y: np.ndarray[float], z: np.ndarray[float]
    ) -> np.ndarray[bool]:
        """For input arrays ``x``, ``y``, ``z`` of arbitrary but identical shape, return an array
        with the same shape which is ``True`` for every point in zip(x, y, z) that is inside the
        volume of the :class:`Geometry`, and ``False`` otherwise.

        Parameters
        ----------
        x : np.ndarray[float]
            Array of point positions in x direction.
        y : np.ndarray[float]
            Array of point positions in y direction.
        z : np.ndarray[float]
            Array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            ``True`` for every point that is inside the geometry.
        """
        # radius at z
        self._ensure_equal_shape(x, y, z)
        z0, (x0, y0) = self.pop_axis(self.center, axis=self.axis)
        z, (x, y) = self.pop_axis((x, y, z), axis=self.axis)
        radius_offset = self._radius_z(z)
        positive_radius = radius_offset > 0

        dist_x = np.abs(x - x0)
        dist_y = np.abs(y - y0)
        dist_z = np.abs(z - z0)
        inside_radius = (dist_x**2 + dist_y**2) <= (radius_offset**2)
        inside_height = dist_z <= (self.finite_length_axis / 2)
        return positive_radius * inside_radius * inside_height

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float, float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        coord_min = [c - self.radius_max for c in self.center]
        coord_max = [c + self.radius_max for c in self.center]
        coord_min[self.axis] = self.center[self.axis] - self.length_axis / 2.0
        coord_max[self.axis] = self.center[self.axis] + self.length_axis / 2.0
        return (tuple(coord_min), tuple(coord_max))

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""

        coord_min = max(self.bounds[0][self.axis], bounds[0][self.axis])
        coord_max = min(self.bounds[1][self.axis], bounds[1][self.axis])

        length = coord_max - coord_min

        volume = np.pi * self.radius_max**2 * length

        # a very loose upper bound on how much of the cylinder is in bounds
        for axis in range(3):
            if axis != self.axis:
                if self.center[axis] <= bounds[0][axis] or self.center[axis] >= bounds[1][axis]:
                    volume *= 0.5

        return volume

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""

        area = 0

        coord_min = self.bounds[0][self.axis]
        coord_max = self.bounds[1][self.axis]

        if coord_min < bounds[0][self.axis]:
            coord_min = bounds[0][self.axis]
        else:
            area += np.pi * self.radius_max**2

        if coord_max > bounds[1][self.axis]:
            coord_max = bounds[1][self.axis]
        else:
            area += np.pi * self.radius_max**2

        length = coord_max - coord_min

        area += 2.0 * np.pi * self.radius_max * length

        # a very loose upper bound on how much of the cylinder is in bounds
        for axis in range(3):
            if axis != self.axis:
                if self.center[axis] <= bounds[0][axis] or self.center[axis] >= bounds[1][axis]:
                    area *= 0.5

        return area

    @cached_property
    def radius_bottom(self) -> float:
        """radius of bottom"""
        return self._radius_z(self.center_axis - self.finite_length_axis / 2)

    @cached_property
    def radius_top(self) -> float:
        """radius of bottom"""
        return self._radius_z(self.center_axis + self.finite_length_axis / 2)

    @cached_property
    def radius_max(self) -> float:
        """max(radius of top, radius of bottom)"""
        return max(self.radius_bottom, self.radius_top)

    @cached_property
    def radius_min(self) -> float:
        """min(radius of top, radius of bottom). It can be negative for a large
        sidewall angle.
        """
        return min(self.radius_bottom, self.radius_top)

    def _radius_z(self, z: float):
        """Compute the radius of the cross section at the position z.

        Parameters
        ----------
        z : float
            Position along the axis normal to slab
        """
        if isclose(self.sidewall_angle, 0):
            return self.radius

        radius_middle = self.radius
        if self.reference_plane == "top":
            radius_middle += self.finite_length_axis / 2 * self._tanq
        elif self.reference_plane == "bottom":
            radius_middle -= self.finite_length_axis / 2 * self._tanq

        return radius_middle - (z - self.center_axis) * self._tanq

    def _local_to_global_side_cross_section(self, coords: List[float], axis: int) -> List[float]:
        """Map a point (x,y) from local to global coordinate system in the
        side cross section.

        The definition of the local: y=0 lies at the base if ``sidewall_angle>=0``,
        and at the top if ``sidewall_angle<0``; x=0 aligns with the corresponding
        ``self.center``.

        Parameters
        ----------
        axis : int
            Integer index into 'xyz' (0, 1, 2).
        coords : List[float, float]
            The value in the planar coordinate.

        Returns
        -------
        Tuple[float, float]
            The point in the global coordinate for plotting `_intersection_side`.

        """

        _, (x_center, y_center) = self.pop_axis(self.center, axis=axis)
        lx_offset, ly_offset = self._order_by_axis(
            plane_val=coords[0], axis_val=-self.finite_length_axis / 2 + coords[1], axis=axis
        )
        if not isclose(self.sidewall_angle, 0):
            ly_offset *= (-1) ** (self.sidewall_angle < 0)
        return [x_center + lx_offset, y_center + ly_offset]
