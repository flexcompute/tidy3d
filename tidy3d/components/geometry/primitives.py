"""Concrete primitive geometrical objects."""

from __future__ import annotations

from math import isclose
from typing import List

import autograd.numpy as anp
import numpy as np
import pydantic.v1 as pydantic
import shapely

from ...constants import C_0, LARGE_NUMBER, MICROMETER
from ...exceptions import SetupError, ValidationError
from ...packaging import verify_packages_import
from ..autograd import AutogradFieldMap, TracedSize1D
from ..autograd.derivative_utils import DerivativeInfo
from ..base import cached_property, skip_if_fields_missing
from ..types import Axis, Bound, Coordinate, MatrixReal4x4, Shapely, Tuple
from . import base
from .polyslab import PolySlab

# for sampling conical frustum in visualization
_N_SAMPLE_CURVE_SHAPELY = 40

# for shapely circular shapes discretization in visualization
_N_SHAPELY_QUAD_SEGS = 200

# Default number of points to discretize polyslab in `Cylinder.to_polyslab()`
_N_PTS_CYLINDER_POLYSLAB = 51

# Default number of points per wvl in material for discretizing cylinder in autograd derivative
_PTS_PER_WVL_MAT_CYLINDER_DISCRETIZE = 10


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

    def intersections_tilted_plane(
        self, normal: Coordinate, origin: Coordinate, to_2D: MatrixReal4x4
    ) -> List[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        normal = np.array(normal)
        unit_normal = normal / (np.sum(normal**2) ** 0.5)
        projection = np.dot(np.array(origin) - np.array(self.center), unit_normal)
        if abs(projection) >= self.radius:
            return []

        radius = (self.radius**2 - projection**2) ** 0.5
        center = np.array(self.center) + projection * unit_normal

        v = np.zeros(3)
        v[np.argmin(np.abs(unit_normal))] = 1
        u = np.cross(unit_normal, v)
        u /= np.sum(u**2) ** 0.5
        v = np.cross(unit_normal, u)

        angles = np.linspace(0, 2 * np.pi, _N_SHAPELY_QUAD_SEGS * 4 + 1)[:-1]
        circ = center + np.outer(np.cos(angles), radius * u) + np.outer(np.sin(angles), radius * v)
        vertices = np.dot(np.hstack((circ, np.ones((angles.size, 1)))), to_2D.T)
        return [shapely.Polygon(vertices[:, :2])]

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
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
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

    See Also
    --------

    **Notebooks**

    * `THz integrated demultiplexer/filter based on a ring resonator <../../../notebooks/THzDemultiplexerFilter.html>`_
    * `Photonic crystal waveguide polarization filter <../../../notebooks/PhotonicCrystalWaveguidePolarizationFilter.html>`_
    """

    # Provide more explanations on where radius is defined
    radius: TracedSize1D = pydantic.Field(
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
    @skip_if_fields_missing(["sidewall_angle", "reference_plane"])
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

    def to_polyslab(
        self, num_pts_circumference: int = _N_PTS_CYLINDER_POLYSLAB, **kwargs
    ) -> PolySlab:
        """Convert instance of ``Cylinder`` into a discretized version using ``PolySlab``.

        Parameters
        ----------
        num_pts_circumference : int = 51
            Number of points in the circumference of the discretized polyslab.
        **kwargs:
            Extra keyword arguments passed to ``PolySlab()``, such as ``dilation``.

        Returns
        -------
        PolySlab
            Extruded polygon representing a discretized version of the cylinder.
        """

        center_axis = self.center_axis
        length_axis = self.length_axis
        slab_bounds = (center_axis - length_axis / 2.0, center_axis + length_axis / 2.0)

        if num_pts_circumference < 3:
            raise ValueError("'PolySlab' from 'Cylinder' must have 3 or more radius points.")

        _, (x0, y0) = self.pop_axis(self.center, axis=self.axis)

        xs_, ys_ = self._points_unit_circle(num_pts_circumference=num_pts_circumference)

        xs = x0 + self.radius * xs_
        ys = y0 + self.radius * ys_

        vertices = anp.stack((xs, ys), axis=-1)

        return PolySlab(
            vertices=vertices,
            axis=self.axis,
            slab_bounds=slab_bounds,
            sidewall_angle=self.sidewall_angle,
            reference_plane=self.reference_plane,
            **kwargs,
        )

    def _points_unit_circle(
        self, num_pts_circumference: int = _N_PTS_CYLINDER_POLYSLAB
    ) -> np.ndarray:
        """Set of x and y points for the unit circle when discretizing cylinder as a polyslab."""
        angles = np.linspace(0, 2 * np.pi, num_pts_circumference, endpoint=False)
        xs = np.cos(angles)
        ys = np.sin(angles)
        return np.stack((xs, ys), axis=0)

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""

        # compute number of points in the circumference of the polyslab using resolution info
        wvl0 = C_0 / derivative_info.frequency
        wvl_mat = wvl0 / max(1.0, np.sqrt(abs(derivative_info.eps_in)))

        circumference = 2 * np.pi * self.radius
        wvls_in_circumference = circumference / wvl_mat

        num_pts_circumference = int(
            np.ceil(_PTS_PER_WVL_MAT_CYLINDER_DISCRETIZE * wvls_in_circumference)
        )
        num_pts_circumference = max(3, num_pts_circumference)

        # construct equivalent polyslab and compute the derivatives
        polyslab = self.to_polyslab(num_pts_circumference=num_pts_circumference)

        derivative_info_polyslab = derivative_info.updated_copy(paths=[("vertices",)], deep=False)
        vjps_polyslab = polyslab.compute_derivatives(derivative_info_polyslab)

        vjps_vertices_xs, vjps_vertices_ys = vjps_polyslab[("vertices",)].T

        # transform polyslab vertices derivatives into Cylinder parameter derivatives
        xs_, ys_ = self._points_unit_circle(num_pts_circumference=num_pts_circumference)
        vjp_xs = np.sum(xs_ * vjps_vertices_xs)
        vjp_ys = np.sum(ys_ * vjps_vertices_ys)

        vjps = {}
        for path in derivative_info.paths:
            if path == ("radius",):
                vjps[path] = vjp_xs + vjp_ys

            elif "center" in path:
                _, center_index = path
                if center_index == self.axis:
                    raise NotImplementedError(
                        "Currently cannot differentiate Cylinder with respect to its 'center' along"
                        " the axis. If you would like this feature added, please feel free to raise"
                        " an issue on the tidy3d front end repository."
                    )

                _, (index_x, index_y) = self.pop_axis((0, 1, 2), axis=self.axis)
                if center_index == index_x:
                    vjps[path] = np.sum(vjps_vertices_xs)
                elif center_index == index_y:
                    vjps[path] = np.sum(vjps_vertices_ys)
                else:
                    raise ValueError(
                        "Something unexpected happened. Was asked to differentiate "
                        f"with respect to 'Cylinder.center[{center_index}]', but this was not "
                        "detected as being one of the parallel axis with "
                        f"'Cylinder.axis' of '{self.axis}'. If you received this error, please raise "
                        "an issue on the tidy3d front end repository with details about how you "
                        "defined your 'Cylinder' in the objective function."
                    )

            else:
                raise NotImplementedError(
                    f"Differentiation with respect to 'Cylinder' '{path}' field not supported. "
                    "If you would like this feature added, please feel free to raise "
                    "an issue on the tidy3d front end repository."
                )

        return vjps

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

    def _update_from_bounds(self, bounds: Tuple[float, float], axis: Axis) -> Cylinder:
        """Returns an updated geometry which has been transformed to fit within ``bounds``
        along the ``axis`` direction."""
        if axis != self.axis:
            raise ValueError(
                f"'_update_from_bounds' may only be applied along axis '{self.axis}', "
                f"but was given axis '{axis}'."
            )
        new_center = list(self.center)
        new_center[axis] = (bounds[0] + bounds[1]) / 2
        new_length = bounds[1] - bounds[0]
        return self.updated_copy(center=new_center, length=new_length)

    @verify_packages_import(["trimesh"])
    def _do_intersections_tilted_plane(
        self, normal: Coordinate, origin: Coordinate, to_2D: MatrixReal4x4
    ) -> List[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        import trimesh

        z0, (x0, y0) = self.pop_axis(self.center, self.axis)
        half_length = self.finite_length_axis / 2

        z_top = z0 + half_length
        z_bot = z0 - half_length

        if np.isclose(self.sidewall_angle, 0):
            r_top = self.radius
            r_bot = self.radius
        else:
            r_top = self.radius_top
            r_bot = self.radius_bottom
            if r_top < 0 or np.isclose(r_top, 0):
                r_top = 0
                z_top = z0 + self._radius_z(z0) / self._tanq
            elif r_bot < 0 or np.isclose(r_bot, 0):
                r_bot = 0
                z_bot = z0 + self._radius_z(z0) / self._tanq

        angles = np.linspace(0, 2 * np.pi, _N_SHAPELY_QUAD_SEGS * 4 + 1)

        if r_bot > 0:
            x_bot = x0 + r_bot * np.cos(angles)
            y_bot = y0 + r_bot * np.sin(angles)
            x_bot[-1] = x0
            y_bot[-1] = y0
        else:
            x_bot = np.array([x0])
            y_bot = np.array([y0])

        if r_top > 0:
            x_top = x0 + r_top * np.cos(angles)
            y_top = y0 + r_top * np.sin(angles)
            x_top[-1] = x0
            y_top[-1] = y0
        else:
            x_top = np.array([x0])
            y_top = np.array([y0])

        x = np.hstack((x_bot, x_top))
        y = np.hstack((y_bot, y_top))
        z = np.hstack((np.full_like(x_bot, z_bot), np.full_like(x_top, z_top)))
        vertices = np.vstack(self.unpop_axis(z, (x, y), self.axis)).T

        if x_bot.shape[0] == 1:
            m = 1
            n = x_top.shape[0] - 1
            faces_top = [(m + n, m + i, m + (i + 1) % n) for i in range(n)]
            faces_side = [(m + (i + 1) % n, m + i, 0) for i in range(n)]
            faces = faces_top + faces_side
        elif x_top.shape[0] == 1:
            m = x_bot.shape[0]
            n = m - 1
            faces_bot = [(n, (i + 1) % n, i) for i in range(n)]
            faces_side = [(i, (i + 1) % n, m) for i in range(n)]
            faces = faces_bot + faces_side
        else:
            m = x_bot.shape[0]
            n = m - 1
            faces_bot = [(n, (i + 1) % n, i) for i in range(n)]
            faces_top = [(m + n, m + i, m + (i + 1) % n) for i in range(n)]
            faces_side_bot = [(i, (i + 1) % n, m + (i + 1) % n) for i in range(n)]
            faces_side_top = [(m + (i + 1) % n, m + i, i) for i in range(n)]
            faces = faces_bot + faces_top + faces_side_bot + faces_side_top

        mesh = trimesh.Trimesh(vertices, faces)

        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            return []
        path, _ = section.to_planar(to_2D=to_2D)
        return path.polygons_full

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
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        static_self = self.to_static()

        # radius at z
        radius_offset = static_self._radius_z(z)

        if radius_offset <= 0:
            return []

        _, (x0, y0) = self.pop_axis(static_self.center, axis=self.axis)
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
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
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
        ``self.center``. In both cases, y-axis is pointing towards the narrowing
        direction of cylinder.

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

        # For negative sidewall angle, quantities along axis direction usually needs a flipped sign
        axis_sign = 1
        if self.sidewall_angle < 0:
            axis_sign = -1

        lx_offset, ly_offset = self._order_by_axis(
            plane_val=coords[0],
            axis_val=axis_sign * (-self.finite_length_axis / 2 + coords[1]),
            axis=axis,
        )
        _, (x_center, y_center) = self.pop_axis(self.center, axis=axis)
        return [x_center + lx_offset, y_center + ly_offset]
