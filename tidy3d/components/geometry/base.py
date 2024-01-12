"""Abstract base classes for geometry."""


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from math import isclose
import functools

import pydantic as pydantic
import numpy as np
import shapely
# from matplotlib import patches

from ..base import Tidy3dBaseModel, cached_property
from ..types import Ax, Axis, PlanePosition, Shapely, ClipOperationType, annotate_type
from ..types import Bound, Size, Coordinate, Coordinate2D, ArrayFloat2D, ArrayFloat3D
from ..viz import add_ax_if_none, equal_aspect, PLOT_BUFFER, ARROW_LENGTH
from ..viz import PlotParams, plot_params_geometry, polygon_patch, arrow_style
from ..transformation import RotationAroundAxis
from ...log import log
from ...exceptions import SetupError, ValidationError, Tidy3dKeyError, Tidy3dError
from ...constants import MICROMETER, LARGE_NUMBER, RADIAN, inf, fp_eps

POLY_GRID_SIZE = 1e-12


class Geometry(Tidy3dBaseModel, ABC):
    """Abstract base class, defines where something exists in space."""

    @cached_property
    def plot_params(self):
        """Default parameters for plotting a Geometry object."""
        return plot_params_geometry

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

        def point_inside(x: float, y: float, z: float):
            """Returns ``True`` if a single point ``(x, y, z)`` is inside."""
            shapes_intersect = self.intersections_plane(z=z)
            loc = shapely.Point(x, y)
            return any(shape.contains(loc) for shape in shapes_intersect)

        arrays = tuple(map(np.array, (x, y, z)))
        self._ensure_equal_shape(*arrays)
        inside = np.zeros((arrays[0].size,), dtype=bool)
        arrays_flat = map(np.ravel, arrays)
        for ipt, args in enumerate(zip(*arrays_flat)):
            inside[ipt] = point_inside(*args)
        return inside.reshape(arrays[0].shape)

    @staticmethod
    def _ensure_equal_shape(*arrays):
        """Ensure all input arrays have the same shape."""
        shapes = {np.array(arr).shape for arr in arrays}
        if len(shapes) > 1:
            raise ValueError("All coordinate inputs (x, y, z) must have the same shape.")

    def _inds_inside_bounds(
        self, x: np.ndarray[float], y: np.ndarray[float], z: np.ndarray[float]
    ) -> Tuple[slice, slice, slice]:
        """Return slices into the sorted input arrays that are inside the geometry bounds.

        Parameters
        ----------
        x : np.ndarray[float]
            1D array of point positions in x direction.
        y : np.ndarray[float]
            1D array of point positions in y direction.
        z : np.ndarray[float]
            1D array of point positions in z direction.

        Returns
        -------
        Tuple[slice, slice, slice]
            Slices into each of the three arrays that are inside the geometry bounds.
        """
        bounds = self.bounds
        inds_in = []
        for dim, coords in enumerate([x, y, z]):
            inds = np.nonzero((bounds[0][dim] <= coords) * (coords <= bounds[1][dim]))[0]
            inds_in.append(slice(0, 0) if inds.size == 0 else slice(inds[0], inds[-1] + 1))

        return tuple(inds_in)

    def inside_meshgrid(
        self, x: np.ndarray[float], y: np.ndarray[float], z: np.ndarray[float]
    ) -> np.ndarray[bool]:
        """Perform ``self.inside`` on a set of sorted 1D coordinates. Applies meshgrid to the
        supplied coordinates before checking inside.

        Parameters
        ----------

        x : np.ndarray[float]
            1D array of point positions in x direction.
        y : np.ndarray[float]
            1D array of point positions in y direction.
        z : np.ndarray[float]
            1D array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            Array with shape ``(x.size, y.size, z.size)``, which is ``True`` for every
            point that is inside the geometry.
        """

        arrays = tuple(map(np.array, (x, y, z)))
        if any(arr.ndim != 1 for arr in arrays):
            raise ValueError("Each of the supplied coordinates (x, y, z) must be 1D.")
        shape = tuple(arr.size for arr in arrays)
        is_inside = np.zeros(shape, dtype=bool)
        inds_inside = self._inds_inside_bounds(*arrays)
        coords_inside = tuple(arr[ind] for ind, arr in zip(inds_inside, arrays))
        coords_3d = np.meshgrid(*coords_inside, indexing="ij")
        is_inside[inds_inside] = self.inside(*coords_3d)
        return is_inside

    @abstractmethod
    def intersections_plane(
        self, x: float = None, y: float = None, z: float = None
    ) -> List[Shapely]:
        """Returns list of shapely geomtries at plane specified by one non-None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

    def intersections_2dbox(self, plane: Box) -> List[Shapely]:
        """Returns list of shapely geomtries representing the intersections of the geometry with
        a 2D box.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane. For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        log.warning(
            "'intersections_2dbox()' is deprecated and will be removed in the future. "
            "Use 'plane.intersections_with(...)' for the same functionality."
        )
        return plane.intersections_with(self)

    def intersects(self, other) -> bool:
        """Returns ``True`` if two :class:`Geometry` have intersecting `.bounds`.

        Parameters
        ----------
        other : :class:`Geometry`
            Geometry to check intersection with.

        Returns
        -------
        bool
            Whether the rectangular bounding boxes of the two geometries intersect.
        """

        self_bmin, self_bmax = self.bounds
        other_bmin, other_bmax = other.bounds

        # are all of other's minimum coordinates less than self's maximum coordinate?
        in_minus = all(o <= s for (s, o) in zip(self_bmax, other_bmin))

        # are all of other's maximum coordinates greater than self's minum coordinate?
        in_plus = all(o >= s for (s, o) in zip(self_bmin, other_bmax))

        # for intersection of bounds, both must be true
        return in_minus and in_plus

    def intersects_plane(self, x: float = None, y: float = None, z: float = None) -> bool:
        """Whether self intersects plane specified by one non-None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

        Returns
        -------
        bool
            Whether this geometry intersects the plane.
        """

        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        return self.intersects_axis_position(axis, position)

    def intersects_axis_position(self, axis: int, position: float) -> bool:
        """Whether self intersects plane specified by a given position along a normal axis.

        Parameters
        ----------
        axis : int = None
            Axis nomral to the plane.
        position : float = None
            Position of plane along the normal axis.

        Returns
        -------
        bool
            Whether this geometry intersects the plane.
        """
        return self.bounds[0][axis] <= position <= self.bounds[1][axis]

    @cached_property
    @abstractmethod
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """

    @staticmethod
    def bounds_intersection(bounds1: Bound, bounds2: Bound) -> Bound:
        """Return the bounds that are the intersection of two bounds."""
        rmin1, rmax1 = bounds1
        rmin2, rmax2 = bounds2
        rmin = tuple(max(v1, v2) for v1, v2 in zip(rmin1, rmin2))
        rmax = tuple(min(v1, v2) for v1, v2 in zip(rmax1, rmax2))
        return (rmin, rmax)

    @cached_property
    def bounding_box(self):
        """Returns :class:`Box` representation of the bounding box of a :class:`Geometry`.

        Returns
        -------
        :class:`Box`
            Geometric object representing bounding box.
        """
        return Box.from_bounds(*self.bounds)

    def _pop_bounds(self, axis: Axis) -> Tuple[Coordinate2D, Tuple[Coordinate2D, Coordinate2D]]:
        """Returns min and max bounds in plane normal to and tangential to ``axis``.

        Parameters
        ----------
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]
            Bounds along axis and a tuple of bounds in the ordered planar coordinates.
            Packed as ``(zmin, zmax), ((xmin, ymin), (xmax, ymax))``.
        """
        b_min, b_max = self.bounds
        zmin, (xmin, ymin) = self.pop_axis(b_min, axis=axis)
        zmax, (xmax, ymax) = self.pop_axis(b_max, axis=axis)
        return (zmin, zmax), ((xmin, ymin), (xmax, ymax))

    @staticmethod
    def _get_center(pt_min: float, pt_max: float) -> float:
        """Returns center point based on bounds along dimension."""
        if np.isneginf(pt_min) and np.isposinf(pt_max):
            return 0.0
        if np.isneginf(pt_min) or np.isposinf(pt_max):
            raise SetupError(
                f"Bounds of ({pt_min}, {pt_max}) supplied along one dimension. "
                "We currently don't support a single ``inf`` value in bounds for ``Box``. "
                "To construct a semi-infinite ``Box``, "
                "please supply a large enough number instead of ``inf``. "
                "For example, a location extending outside of the "
                "Simulation domain (including PML)."
            )
        return (pt_min + pt_max) / 2.0

    @cached_property
    def _normal_2dmaterial(self) -> Axis:
        """Get the normal to the given geometry, checking that it is a 2D geometry."""
        raise ValidationError("'Medium2D' is not conpatible with this geometry class.")

    @equal_aspect
    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **patch_kwargs
    ) -> Ax:
        """Plot geometry cross section at single (x,y,z) coordinate.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # find shapes that intersect self at plane
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        shapes_intersect = self.intersections_plane(x=x, y=y, z=z)

        plot_params = self.plot_params.include_kwargs(**patch_kwargs)

        # for each intersection, plot the shape
        for shape in shapes_intersect:
            ax = self.plot_shape(shape, plot_params=plot_params, ax=ax)

        # clean up the axis display
        ax = self.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_aspect("equal")
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")
        return ax

    def plot_shape(self, shape: Shapely, plot_params: PlotParams, ax: Ax) -> Ax:
        """Defines how a shape is plotted on a matplotlib axes."""
        _shape = Geometry.evaluate_inf_shape(shape)
        if _shape.geom_type == "LineString":
            xs, ys = zip(*_shape.coords)
            ax.plot(xs, ys, color=plot_params.facecolor)
        elif _shape.geom_type == "Point":
            ax.scatter(shape.x, shape.y, color=plot_params.facecolor)
        else:
            patch = polygon_patch(_shape, **plot_params.to_kwargs())
            ax.add_artist(patch)
        return ax

    def _get_plot_labels(self, axis: Axis) -> Tuple[str, str]:
        """Returns planar coordinate x and y axis labels for cross section plots.

        Parameters
        ----------
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        str, str
            Labels of plot, packaged as ``(xlabel, ylabel)``.
        """
        _, (xlabel, ylabel) = self.pop_axis("xyz", axis=axis)
        return xlabel, ylabel

    def _get_plot_limits(
        self, axis: Axis, buffer: float = PLOT_BUFFER
    ) -> Tuple[Coordinate2D, Coordinate2D]:
        """Gets planar coordinate limits for cross section plots.

        Parameters
        ----------
        axis : int
            Integer index into 'xyz' (0,1,2).
        buffer : float = 0.3
            Amount of space to add around the limits on the + and - sides.

        Returns
        -------
            Tuple[float, float], Tuple[float, float]
        The x and y plot limits, packed as ``(xmin, xmax), (ymin, ymax)``.
        """
        _, ((xmin, ymin), (xmax, ymax)) = self._pop_bounds(axis=axis)
        return (xmin - buffer, xmax + buffer), (ymin - buffer, ymax + buffer)

    def add_ax_labels_lims(self, axis: Axis, ax: Ax, buffer: float = PLOT_BUFFER) -> Ax:
        """Sets the x,y labels based on ``axis`` and the extends based on ``self.bounds``.

        Parameters
        ----------
        axis : int
            Integer index into 'xyz' (0,1,2).
        ax : matplotlib.axes._subplots.Axes
            Matplotlib axes to add labels and limits on.
        buffer : float = 0.3
            Amount of space to place around the limits on the + and - sides.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        xlabel, ylabel = self._get_plot_labels(axis=axis)
        (xmin, xmax), (ymin, ymax) = self._get_plot_limits(axis=axis, buffer=buffer)

        # note: axes limits dont like inf values, so we need to evaluate them first if present
        xmin, xmax, ymin, ymax = self._evaluate_inf((xmin, xmax, ymin, ymax))

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    @staticmethod
    def _evaluate_inf(array):
        """Processes values and evaluates any infs into large (signed) numbers."""
        return np.where(np.isinf(array), np.sign(array) * LARGE_NUMBER, array)

    @staticmethod
    def evaluate_inf_shape(shape: Shapely) -> Shapely:
        """Returns a copy of shape with inf vertices replaced by large numbers if polygon."""
        if not any(np.isinf(b) for b in shape.bounds):
            return shape

        if shape.geom_type == "Polygon":
            return shapely.Polygon(
                Geometry._evaluate_inf(np.array(shape.exterior.coords)),
                [Geometry._evaluate_inf(np.array(g.coords)) for g in shape.interiors],
            )
        if shape.geom_type in {"Point", "LineString", "LinearRing"}:
            return shape.__class__(Geometry._evaluate_inf(np.array(shape.coords)))
        if shape.geom_type in {
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
            "GeometryCollection",
        }:
            return shape.__class__([Geometry.evaluate_inf_shape(g) for g in shape.geoms])
        return shape

    @staticmethod
    def pop_axis(coord: Tuple[Any, Any, Any], axis: int) -> Tuple[Any, Tuple[Any, Any]]:
        """Separates coordinate at ``axis`` index from coordinates on the plane tangent to ``axis``.

        Parameters
        ----------
        coord : Tuple[Any, Any, Any]
            Tuple of three values in original coordinate system.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        Any, Tuple[Any, Any]
            The input coordinates are separated into the one along the axis provided
            and the two on the planar coordinates,
            like ``axis_coord, (planar_coord1, planar_coord2)``.
        """
        plane_vals = list(coord)
        axis_val = plane_vals.pop(axis)
        return axis_val, tuple(plane_vals)

    @staticmethod
    def unpop_axis(ax_coord: Any, plane_coords: Tuple[Any, Any], axis: int) -> Tuple[Any, Any, Any]:
        """Combine coordinate along axis with coordinates on the plane tangent to the axis.

        Parameters
        ----------
        ax_coord : Any
            Value along axis direction.
        plane_coords : Tuple[Any, Any]
            Values along ordered planar directions.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        Tuple[Any, Any, Any]
            The three values in the xyz coordinate system.
        """
        coords = list(plane_coords)
        coords.insert(axis, ax_coord)
        return tuple(coords)

    @staticmethod
    def parse_xyz_kwargs(**xyz) -> Tuple[Axis, float]:
        """Turns x,y,z kwargs into index of the normal axis and position along that axis.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

        Returns
        -------
        int, float
            Index into xyz axis (0,1,2) and position along that axis.
        """
        xyz_filtered = {k: v for k, v in xyz.items() if v is not None}
        assert len(xyz_filtered) == 1, "exatly one kwarg in [x,y,z] must be specified."
        axis_label, position = list(xyz_filtered.items())[0]
        axis = "xyz".index(axis_label)
        return axis, position

    @staticmethod
    def rotate_points(points: ArrayFloat3D, axis: Coordinate, angle: float) -> ArrayFloat3D:
        """Rotate a set of points in 3D.

        Parameters
        ----------
        points : ArrayLike[float]
            Array of shape ``(3, ...)``.
        axis : Coordinate
            Axis of rotation
        angle : float
            Angle of rotation counter-clockwise around the axis (rad).
        """
        rotation = RotationAroundAxis(axis=axis, angle=angle)
        return rotation.rotate_vector(points)

    def reflect_points(
        self,
        points: ArrayFloat3D,
        polar_axis: Axis,
        angle_theta: float,
        angle_phi: float,
    ) -> ArrayFloat3D:
        """Reflect a set of points in 3D at a plane passing through the coordinate origin defined
        and normal to a given axis defined in polar coordinates (theta, phi) w.r.t. the
        ``polar_axis`` which can be 0, 1, or 2.

        Parameters
        ----------
        points : ArrayLike[float]
            Array of shape ``(3, ...)``.
        polar_axis : Axis
            Cartesian axis w.r.t. which the normal axis angles are defined.
        angle_theta : float
            Polar angle w.r.t. the polar axis.
        angle_phi : float
            Azimuth angle around the polar axis.
        """

        # Rotate such that the plane normal is along the polar_axis
        axis_theta, axis_phi = [0, 0, 0], [0, 0, 0]
        axis_phi[polar_axis] = 1
        plane_axes = [0, 1, 2]
        plane_axes.pop(polar_axis)
        axis_theta[plane_axes[1]] = 1
        points_new = self.rotate_points(points, axis_phi, -angle_phi)
        points_new = self.rotate_points(points_new, axis_theta, -angle_theta)

        # Flip the ``polar_axis`` coordinate of the points, which is now normal to the plane
        points_new[polar_axis, :] *= -1

        # Rotate back
        points_new = self.rotate_points(points_new, axis_theta, angle_theta)
        points_new = self.rotate_points(points_new, axis_phi, angle_phi)

        return points_new

    def volume(self, bounds: Bound = None):
        """Returns object's volume with optional bounds.

        Parameters
        ----------
        bounds : Tuple[Tuple[float, float, float], Tuple[float, float, float]] = None
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        float
            Volume in um^3.
        """

        if not bounds:
            bounds = self.bounds

        return self._volume(bounds)

    @abstractmethod
    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""

    def surface_area(self, bounds: Bound = None):
        """Returns object's surface area with optional bounds.

        Parameters
        ----------
        bounds : Tuple[Tuple[float, float, float], Tuple[float, float, float]] = None
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        float
            Surface area in um^2.
        """

        if not bounds:
            bounds = self.bounds

        return self._surface_area(bounds)

    @abstractmethod
    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""

    """ Field and coordinate transformations """

    @staticmethod
    def car_2_sph(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert Cartesian to spherical coordinates.

        Parameters
        ----------
        x : float
            x coordinate relative to ``local_origin``.
        y : float
            y coordinate relative to ``local_origin``.
        z : float
            z coordinate relative to ``local_origin``.

        Returns
        -------
        Tuple[float, float, float]
            r, theta, and phi coordinates relative to ``local_origin``.
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    @staticmethod
    def sph_2_car(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
        """Convert spherical to Cartesian coordinates.

        Parameters
        ----------
        r : float
            radius.
        theta : float
            polar angle (rad) downward from x=y=0 line.
        phi : float
            azimuthal (rad) angle from y=z=0 line.

        Returns
        -------
        Tuple[float, float, float]
            x, y, and z coordinates relative to ``local_origin``.
        """
        r_sin_theta = r * np.sin(theta)
        x = r_sin_theta * np.cos(phi)
        y = r_sin_theta * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def sph_2_car_field(
        f_r: float, f_theta: float, f_phi: float, theta: float, phi: float
    ) -> Tuple[complex, complex, complex]:
        """Convert vector field components in spherical coordinates to cartesian.

        Parameters
        ----------
        f_r : float
            radial component of the vector field.
        f_theta : float
            polar angle component of the vector fielf.
        f_phi : float
            azimuthal angle component of the vector field.
        theta : float
            polar angle (rad) of location of the vector field.
        phi : float
            azimuthal angle (rad) of location of the vector field.

        Returns
        -------
        Tuple[float, float, float]
            x, y, and z components of the vector field in cartesian coordinates.
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        f_x = f_r * sin_theta * cos_phi + f_theta * cos_theta * cos_phi - f_phi * sin_phi
        f_y = f_r * sin_theta * sin_phi + f_theta * cos_theta * sin_phi + f_phi * cos_phi
        f_z = f_r * cos_theta - f_theta * sin_theta
        return f_x, f_y, f_z

    @staticmethod
    def car_2_sph_field(
        f_x: float, f_y: float, f_z: float, theta: float, phi: float
    ) -> Tuple[complex, complex, complex]:
        """Convert vector field components in cartesian coordinates to spherical.

        Parameters
        ----------
        f_x : float
            x component of the vector field.
        f_y : float
            y component of the vector fielf.
        f_z : float
            z component of the vector field.
        theta : float
            polar angle (rad) of location of the vector field.
        phi : float
            azimuthal angle (rad) of location of the vector field.

        Returns
        -------
        Tuple[float, float, float]
            radial (s), elevation (theta), and azimuthal (phi) components
            of the vector field in spherical coordinates.
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        f_r = f_x * sin_theta * cos_phi + f_y * sin_theta * sin_phi + f_z * cos_theta
        f_theta = f_x * cos_theta * cos_phi + f_y * cos_theta * sin_phi - f_z * sin_theta
        f_phi = -f_x * sin_phi + f_y * cos_phi
        return f_r, f_theta, f_phi

    @staticmethod
    def kspace_2_sph(ux: float, uy: float, axis: Axis) -> Tuple[float, float]:
        """Convert normalized k-space coordinates to angles.

        Parameters
        ----------
        ux : float
            normalized kx coordinate.
        uy : float
            normalized ky coordinate.
        axis : int
            axis along which the observation plane is oriented.

        Returns
        -------
        Tuple[float, float]
            theta and phi coordinates relative to ``local_origin``.
        """
        phi_local = np.arctan2(uy, ux)
        theta_local = np.arcsin(np.sqrt(ux**2 + uy**2))
        # Spherical coordinates rotation matrix reference:
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
        if axis == 2:
            return theta_local, phi_local

        x = np.cos(theta_local)
        y = np.sin(theta_local) * np.sin(phi_local)
        z = -np.sin(theta_local) * np.cos(phi_local)

        if axis == 1:
            x, y, z = -z, x, -y

        theta = np.arccos(z)
        phi = np.arctan2(y, x)
        return theta, phi

    @staticmethod
    def load_gds_vertices_gdstk(
        gds_cell, gds_layer: int, gds_dtype: int = None, gds_scale: pydantic.PositiveFloat = 1.0
    ) -> List[ArrayFloat2D]:
        """Load polygon vertices from a ``gdstk.Cell``.

        Parameters
        ----------
        gds_cell : gdstk.Cell
            ``gdstk.Cell`` or ``gdspy.Cell`` containing 2D geometric data.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``. If ``None``, imports all data for this layer into
            the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of micrometer. For example, if gds file uses
            nanometers, set ``gds_scale=1e-3``. Must be positive.

        Returns
        -------
        List[ArrayFloat2D]
            List of polygon vertices
        """

        # apply desired scaling and load the polygon vertices
        if gds_dtype is not None:
            # if both layer and datatype are specified, let gdstk do the filtering for better
            # performance on large layouts
            all_vertices = [
                polygon.scale(gds_scale).points
                for polygon in gds_cell.get_polygons(layer=gds_layer, datatype=gds_dtype)
            ]
        else:
            all_vertices = [
                polygon.scale(gds_scale).points
                for polygon in gds_cell.get_polygons()
                if polygon.layer == gds_layer
            ]
        # make sure something got loaded, otherwise error
        if not all_vertices:
            raise Tidy3dKeyError(
                f"Couldn't load gds_cell, no vertices found at gds_layer={gds_layer} "
                f"with specified gds_dtype={gds_dtype}."
            )

        return all_vertices

    @staticmethod
    def load_gds_vertices_gdspy(
        gds_cell, gds_layer: int, gds_dtype: int = None, gds_scale: pydantic.PositiveFloat = 1.0
    ) -> List[ArrayFloat2D]:
        """Load polygon vertices from a ``gdspy.Cell``.

        Parameters
        ----------
        gds_cell : gdspy.Cell
            ``gdstk.Cell`` or ``gdspy.Cell`` containing 2D geometric data.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``. If ``None``, imports all data for this layer into
            the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of micrometer. For example, if gds file uses
            nanometers, set ``gds_scale=1e-3``. Must be positive.

        Returns
        -------
        List[ArrayFloat2D]
            List of polygon vertices
        """

        # load the polygon vertices
        vert_dict = gds_cell.get_polygons(by_spec=True)
        all_vertices = []
        for (gds_layer_file, gds_dtype_file), vertices in vert_dict.items():
            if gds_layer_file == gds_layer and (gds_dtype is None or gds_dtype == gds_dtype_file):
                all_vertices.extend(iter(vertices))
        # make sure something got loaded, otherwise error
        if not all_vertices:
            raise Tidy3dKeyError(
                f"Couldn't load gds_cell, no vertices found at gds_layer={gds_layer} "
                f"with specified gds_dtype={gds_dtype}."
            )

        # apply scaling
        all_vertices = [vertices * gds_scale for vertices in all_vertices]
        return all_vertices

    @staticmethod
    def from_gds(
        gds_cell,
        axis: Axis,
        slab_bounds: Tuple[float, float],
        gds_layer: int,
        gds_dtype: int = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
        dilation: float = 0.0,
        sidewall_angle: float = 0,
        reference_plane: PlanePosition = "middle",
    ) -> GeometryGroup:
        """Import a ``gdstk.Cell`` or a ``gdspy.Cell`` and extrude it into a GeometryGroup.

        Parameters
        ----------
        gds_cell : Union[gdstk.Cell, gdspy.Cell]
            ``gdstk.Cell`` or ``gdspy.Cell`` containing 2D geometric data.
        axis : int
            Integer index defining the extrusion axis: 0 (x), 1 (y), or 2 (z).
        slab_bounds: Tuple[float, float]
            Minimal and maximal positions of the extruded slab along ``axis``.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``. If ``None``, imports all data for this layer into
            the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of micrometer. For example, if gds file uses
            nanometers, set ``gds_scale=1e-3``. Must be positive.
        dilation : float = 0.0
            Dilation (positive) or erosion (negative) amount to be applied to the original polygons.
        sidewall_angle : float = 0
            Angle of the extrusion sidewalls, away from the vertical direction, in radians. Positive
            (negative) values result in slabs larger (smaller) at the base than at the top.
        reference_plane : PlanePosition = "middle"
            Reference position of the (dilated/eroded) polygons along the slab axis. One of
            ``"middle"`` (polygons correspond to the center of the slab bounds), ``"bottom"``
            (minimal slab bound position), or ``"top"`` (maximal slab bound position). This value
            has no effect if ``sidewall_angle == 0``.

        Returns
        -------
        :class:`GeometryGroup`
            Geometry group with geometries created from the 2D data.
        """

        # switch the GDS cell loader function based on the class name string
        # TODO: make this more robust in future releases
        gds_cell_class_name = str(gds_cell.__class__)

        if "gdstk" in gds_cell_class_name:
            gds_loader_fn = Geometry.load_gds_vertices_gdstk
        elif "gdspy" in gds_cell_class_name:
            gds_loader_fn = Geometry.load_gds_vertices_gdspy
        else:
            raise ValueError(
                f"Argument 'gds_cell' of type '{gds_cell_class_name}' does not seem to be "
                "a 'Cell' instance from 'gdstk' or 'gdspy' modules and, therefore, cannot be "
                "loaded by Tidy3D."
            )

        geometries = []
        with log as consolidated_logger:
            for vertices in gds_loader_fn(gds_cell, gds_layer, gds_dtype, gds_scale):
                # buffer(0) is necessary to merge self-intersections
                shape = shapely.set_precision(shapely.Polygon(vertices).buffer(0), POLY_GRID_SIZE)
                try:
                    geometries.append(
                        from_shapely(
                            shape, axis, slab_bounds, dilation, sidewall_angle, reference_plane
                        )
                    )
                except pydantic.ValidationError as error:
                    consolidated_logger.warning(str(error))
                except Tidy3dError as error:
                    consolidated_logger.warning(str(error))
        return GeometryGroup(geometries=geometries)

    @staticmethod
    def from_shapely(
        shape: Shapely,
        axis: Axis,
        slab_bounds: Tuple[float, float],
        dilation: float = 0.0,
        sidewall_angle: float = 0,
        reference_plane: PlanePosition = "middle",
    ) -> Geometry:
        """Convert a shapely primitive into a geometry instance by extrusion.

        Parameters
        ----------
        shape : shapely.geometry.base.BaseGeometry
            Shapely primitive to be converted. It must be a linear ring, a polygon or a collection
            of any of those.
        axis : int
            Integer index defining the extrusion axis: 0 (x), 1 (y), or 2 (z).
        slab_bounds: Tuple[float, float]
            Minimal and maximal positions of the extruded slab along ``axis``.
        dilation : float
            Dilation of the polygon in the base by shifting each edge along its normal outwards
            direction by a distance; a negative value corresponds to erosion.
        sidewall_angle : float = 0
            Angle of the extrusion sidewalls, away from the vertical direction, in radians. Positive
            (negative) values result in slabs larger (smaller) at the base than at the top.
        reference_plane : PlanePosition = "middle"
            Reference position of the (dilated/eroded) polygons along the slab axis. One of
            ``"middle"`` (polygons correspond to the center of the slab bounds), ``"bottom"``
            (minimal slab bound position), or ``"top"`` (maximal slab bound position). This value
            has no effect if ``sidewall_angle == 0``.

        Returns
        -------
        :class:`Geometry`
            Geometry extruded from the 2D data.
        """
        return from_shapely(shape, axis, slab_bounds, dilation, sidewall_angle, reference_plane)

    def _as_union(self) -> List[Geometry]:
        """Return a list of geometries that, united, make up the given geometry."""
        if isinstance(self, GeometryGroup):
            return self.geometries

        if isinstance(self, ClipOperation) and self.operation == "union":
            return (self.geometry_a, self.geometry_b)
        return (self,)

    def __add__(self, other):
        """Union of geometries"""
        # This allows the user to write sum(geometries...) with the default start=0
        if isinstance(other, int):
            return self
        if not isinstance(other, Geometry):
            return NotImplemented
        return GeometryGroup(geometries=self._as_union() + other._as_union())

    def __radd__(self, other):
        """Union of geometries"""
        # This allows the user to write sum(geometries...) with the default start=0
        if isinstance(other, int):
            return self
        if not isinstance(other, Geometry):
            return NotImplemented
        return GeometryGroup(geometries=other._as_union() + self._as_union())

    def __or__(self, other):
        """Union of geometries"""
        if not isinstance(other, Geometry):
            return NotImplemented
        return GeometryGroup(geometries=self._as_union() + other._as_union())

    def __mul__(self, other):
        """Intersection of geometries"""
        if not isinstance(other, Geometry):
            return NotImplemented
        return ClipOperation(operation="intersection", geometry_a=self, geometry_b=other)

    def __and__(self, other):
        """Intersection of geometries"""
        if not isinstance(other, Geometry):
            return NotImplemented
        return ClipOperation(operation="intersection", geometry_a=self, geometry_b=other)

    def __sub__(self, other):
        """Difference of geometries"""
        if not isinstance(other, Geometry):
            return NotImplemented
        return ClipOperation(operation="difference", geometry_a=self, geometry_b=other)

    def __xor__(self, other):
        """Symmetric difference of geometries"""
        if not isinstance(other, Geometry):
            return NotImplemented
        return ClipOperation(operation="symmetric_difference", geometry_a=self, geometry_b=other)

    def __pos__(self):
        """No op"""
        return self

    def __neg__(self):
        """Opposite of a geometry"""
        return ClipOperation(
            operation="difference", geometry_a=Box(size=(inf, inf, inf)), geometry_b=self
        )

    def __invert__(self):
        """Opposite of a geometry"""
        return ClipOperation(
            operation="difference", geometry_a=Box(size=(inf, inf, inf)), geometry_b=self
        )


""" Abstract subclasses """


class Centered(Geometry, ABC):
    """Geometry with a well defined center."""

    center: Coordinate = pydantic.Field(
        (0.0, 0.0, 0.0),
        title="Center",
        description="Center of object in x, y, and z.",
        units=MICROMETER,
    )

    @pydantic.validator("center", always=True)
    def _center_not_inf(cls, val):
        """Make sure center is not infinitiy."""
        if any(np.isinf(v) for v in val):
            raise ValidationError("center can not contain td.inf terms.")
        return val


class Planar(Geometry, ABC):
    """Geometry with one ``axis`` that is slab-like with thickness ``height``."""

    axis: Axis = pydantic.Field(
        2, title="Axis", description="Specifies dimension of the planar axis (0,1,2) -> (x,y,z)."
    )

    sidewall_angle: float = pydantic.Field(
        0.0,
        title="Sidewall angle",
        description="Angle of the sidewall. "
        "``sidewall_angle=0`` (default) specifies a vertical wall; "
        "``0<sidewall_angle<np.pi/2`` specifies a shrinking cross section "
        "along the ``axis`` direction; "
        "and ``-np.pi/2<sidewall_angle<0`` specifies an expanding cross section "
        "along the ``axis`` direction.",
        gt=-np.pi / 2,
        lt=np.pi / 2,
        units=RADIAN,
    )

    reference_plane: PlanePosition = pydantic.Field(
        "middle",
        title="Reference plane for cross section",
        description="The position of the plane where the supplied cross section are "
        "defined. The plane is perpendicular to the ``axis``. "
        "The plane is located at the ``bottom``, ``middle``, or ``top`` of the "
        "geometry with respect to the axis. "
        "E.g. if ``axis=1``, ``bottom`` refers to the negative side of the y-axis, and "
        "``top`` refers to the positive side of the y-axis.",
    )

    @property
    @abstractmethod
    def center_axis(self) -> float:
        """Gets the position of the center of the geometry in the out of plane dimension."""

    @property
    @abstractmethod
    def length_axis(self) -> float:
        """Gets the length of the geometry along the out of plane dimension."""

    @property
    def finite_length_axis(self) -> float:
        """Gets the length of the geometry along the out of plane dimension.
        If the length is td.inf, return ``LARGE_NUMBER``
        """
        return min(self.length_axis, LARGE_NUMBER)

    def intersections_plane(self, x: float = None, y: float = None, z: float = None):
        """Returns shapely geometry at plane specified by one non None value of x,y,z.

        Parameters
        ----------
        x : float
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

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
        if axis == self.axis:
            return self._intersections_normal(position)
        return self._intersections_side(position, axis)

    @abstractmethod
    def _intersections_normal(self, z: float) -> list:
        """Find shapely geometries intersecting planar geometry with axis normal to slab.

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

    @abstractmethod
    def _intersections_side(self, position: float, axis: Axis) -> list:
        """Find shapely geometries intersecting planar geometry with axis orthogonal to plane.

        Parameters
        ----------
        position : float
            Position along axis.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

    def _order_axis(self, axis: int) -> int:
        """Order the axis as if self.axis is along z-direction.

        Parameters
        ----------
        axis : int
            Integer index into the structure's planar axis.

        Returns
        -------
        int
            New index of axis.
        """
        axis_index = [0, 1]
        axis_index.insert(self.axis, 2)
        return axis_index[axis]

    def _order_by_axis(self, plane_val: Any, axis_val: Any, axis: int) -> Tuple[Any, Any]:
        """Orders a value in the plane and value along axis in correct (x,y) order for plotting.
           Note: sometimes if axis=1 and we compute cross section values orthogonal to axis,
           they can either be x or y in the plots.
           This function allows one to figure out the ordering.

        Parameters
        ----------
        plane_val : Any
            The value in the planar coordinate.
        axis_val : Any
            The value in the ``axis`` coordinate.
        axis : int
            Integer index into the structure's planar axis.

        Returns
        -------
        ``(Any, Any)``
            The two planar coordinates in this new coordinate system.
        """
        vals = 3 * [plane_val]
        vals[self.axis] = axis_val
        _, (val_x, val_y) = self.pop_axis(vals, axis=axis)
        return val_x, val_y

    @cached_property
    def _tanq(self) -> float:
        """Value of ``tan(sidewall_angle)``.

        The (possibliy infinite) geometry offest is given by ``_tanq * length_axis``.
        """
        return np.tan(self.sidewall_angle)


class Circular(Geometry):
    """Geometry with circular characteristics (specified by a radius)."""

    radius: pydantic.NonNegativeFloat = pydantic.Field(
        ..., title="Radius", description="Radius of geometry.", units=MICROMETER
    )

    @pydantic.validator("radius", always=True)
    def _radius_not_inf(cls, val):
        """Make sure center is not infinitiy."""
        if np.isinf(val):
            raise ValidationError("radius can not be td.inf.")
        return val

    def _intersect_dist(self, position, z0) -> float:
        """Distance between points on circle at z=position where center of circle at z=z0.

        Parameters
        ----------
        position : float
            position along z.
        z0 : float
            center of circle in z.

        Returns
        -------
        float
            Distance between points on the circle intersecting z=z, if no points, ``None``.
        """
        dz = np.abs(z0 - position)
        if dz > self.radius:
            return None
        return 2 * np.sqrt(self.radius**2 - dz**2)


"""Primitive classes"""


class Box(Centered):
    """Rectangular prism.
       Also base class for :class:`Simulation`, :class:`Monitor`, and :class:`Source`.

    Example
    -------
    >>> b = Box(center=(1,2,3), size=(2,2,2))
    """

    size: Size = pydantic.Field(
        ...,
        title="Size",
        description="Size in x, y, and z directions.",
        units=MICROMETER,
    )

    @classmethod
    def from_bounds(cls, rmin: Coordinate, rmax: Coordinate, **kwargs):
        """Constructs a :class:`Box` from minimum and maximum coordinate bounds

        Parameters
        ----------
        rmin : Tuple[float, float, float]
            (x, y, z) coordinate of the minimum values.
        rmax : Tuple[float, float, float]
            (x, y, z) coordinate of the maximum values.

        Example
        -------
        >>> b = Box.from_bounds(rmin=(-1, -2, -3), rmax=(3, 2, 1))
        """

        center = tuple(cls._get_center(pt_min, pt_max) for pt_min, pt_max in zip(rmin, rmax))
        size = tuple((pt_max - pt_min) for pt_min, pt_max in zip(rmin, rmax))
        return cls(center=center, size=size, **kwargs)

    @classmethod
    def surfaces(cls, size: Size, center: Coordinate, **kwargs):
        """Returns a list of 6 :class:`Box` instances corresponding to each surface of a 3D volume.
        The output surfaces are stored in the order [x-, x+, y-, y+, z-, z+], where x, y, and z
        denote which axis is perpendicular to that surface, while "-" and "+" denote the direction
        of the normal vector of that surface. If a name is provided, each output surface's name
        will be that of the provided name appended with the above symbols. E.g., if the provided
        name is "box", the x+ surfaces's name will be "box_x+".

        Parameters
        ----------
        size : Tuple[float, float, float]
            Size of object in x, y, and z directions.
        center : Tuple[float, float, float]
            Center of object in x, y, and z.

        Example
        -------
        >>> b = Box.surfaces(size=(1, 2, 3), center=(3, 2, 1))
        """

        if any(s == 0.0 for s in size):
            raise SetupError(
                "Can't generate surfaces for the given object because it has zero volume."
            )

        bounds = Box(center=center, size=size).bounds

        # Set up geometry data and names for each surface:
        centers = [list(center) for _ in range(6)]
        sizes = [list(size) for _ in range(6)]

        surface_index = 0
        for dim_index in range(3):
            for min_max_index in range(2):
                new_center = centers[surface_index]
                new_size = sizes[surface_index]

                new_center[dim_index] = bounds[min_max_index][dim_index]
                new_size[dim_index] = 0.0

                centers[surface_index] = new_center
                sizes[surface_index] = new_size

                surface_index += 1

        name_base = kwargs.pop("name", "")
        kwargs.pop("normal_dir", None)

        names = []
        normal_dirs = []

        for coord in "xyz":
            for direction in "-+":
                surface_name = name_base + "_" + coord + direction
                names.append(surface_name)
                normal_dirs.append(direction)

        # ignore surfaces that are infinitely far away
        del_idx = []
        for idx, _size in enumerate(size):
            if _size == inf:
                del_idx.append(idx)
        del_idx = [[2 * i, 2 * i + 1] for i in del_idx]
        del_idx = [item for sublist in del_idx for item in sublist]

        def del_items(items, indices):
            """Delete list items at indices."""
            return [i for j, i in enumerate(items) if j not in indices]

        centers = del_items(centers, del_idx)
        sizes = del_items(sizes, del_idx)
        names = del_items(names, del_idx)
        normal_dirs = del_items(normal_dirs, del_idx)

        surfaces = []
        for _cent, _size, _name, _normal_dir in zip(centers, sizes, names, normal_dirs):
            if "normal_dir" in cls.__dict__["__fields__"]:
                kwargs["normal_dir"] = _normal_dir

            if "name" in cls.__dict__["__fields__"]:
                kwargs["name"] = _name

            surface = cls(center=_cent, size=_size, **kwargs)
            surfaces.append(surface)

        return surfaces

    @classmethod
    def surfaces_with_exclusion(cls, size: Size, center: Coordinate, **kwargs):
        """Returns a list of 6 :class:`Box` instances corresponding to each surface of a 3D volume.
        The output surfaces are stored in the order [x-, x+, y-, y+, z-, z+], where x, y, and z
        denote which axis is perpendicular to that surface, while "-" and "+" denote the direction
        of the normal vector of that surface. If a name is provided, each output surface's name
        will be that of the provided name appended with the above symbols. E.g., if the provided
        name is "box", the x+ surfaces's name will be "box_x+". If `kwargs` contains an
        `exclude_surfaces` parameter, the returned list of surfaces will not include the excluded
        surfaces. Otherwise, the behavior is identical to that of `surfaces()`.

        Parameters
        ----------
        size : Tuple[float, float, float]
            Size of object in x, y, and z directions.
        center : Tuple[float, float, float]
            Center of object in x, y, and z.

        Example
        -------
        >>> b = Box.surfaces_with_exclusion(
        ...     size=(1, 2, 3), center=(3, 2, 1), exclude_surfaces=["x-"]
        ... )
        """
        exclude_surfaces = kwargs.pop("exclude_surfaces", None)
        surfaces = cls.surfaces(size=size, center=center, **kwargs)
        if "name" in cls.__dict__["__fields__"] and exclude_surfaces:
            surfaces = [surf for surf in surfaces if surf.name[-2:] not in exclude_surfaces]
        return surfaces

    def intersections_plane(self, x: float = None, y: float = None, z: float = None):
        """Returns shapely geometry at plane specified by one non None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

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
        Lz, (Lx, Ly) = self.pop_axis(self.size, axis=axis)
        dz = np.abs(z0 - position)
        if dz > Lz / 2 + fp_eps:
            return []

        minx = x0 - Lx / 2
        miny = y0 - Ly / 2
        maxx = x0 + Lx / 2
        maxy = y0 + Ly / 2

        # handle case where the box vertices are identical
        if isclose(minx, maxx) and isclose(miny, maxy):
            return [shapely.Point(minx, miny)]

        return [shapely.box(minx, miny, maxx, maxy)]

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
        Lx, Ly, Lz = self.size
        dist_x = np.abs(x - x0)
        dist_y = np.abs(y - y0)
        dist_z = np.abs(z - z0)
        return (dist_x <= Lx / 2) * (dist_y <= Ly / 2) * (dist_z <= Lz / 2)

    def intersections_with(self, other):
        """Returns list of shapely geomtries representing the intersections of the geometry with
        this 2D box.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect this 2D box.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        # Verify 2D
        if self.size.count(0.0) != 1:
            raise ValidationError(
                "Intersections with other geometry are only calculated from a 2D box."
            )

        # dont bother if the geometry doesn't intersect the self at all
        if not other.intersects(self):
            return []

        # get list of Shapely shapes that intersect at the self
        normal_ind = self.size.index(0.0)
        dim = "xyz"[normal_ind]
        pos = self.center[normal_ind]
        xyz_kwargs = {dim: pos}
        shapes_plane = other.intersections_plane(**xyz_kwargs)

        # intersect all shapes with the input self
        bs_min, bs_max = (self.pop_axis(bounds, axis=normal_ind)[1] for bounds in self.bounds)
        shapely_box = shapely.box(bs_min[0], bs_min[1], bs_max[0], bs_max[1])
        shapely_box = Geometry.evaluate_inf_shape(shapely_box)
        return [Geometry.evaluate_inf_shape(shape) & shapely_box for shape in shapes_plane]

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        size = self.size
        center = self.center
        coord_min = tuple(c - s / 2 for (s, c) in zip(size, center))
        coord_max = tuple(c + s / 2 for (s, c) in zip(size, center))
        return (coord_min, coord_max)

    @cached_property
    def geometry(self):
        """:class:`Box` representation of self (used for subclasses of Box).

        Returns
        -------
        :class:`Box`
            Instance of :class:`Box` representing self's geometry.
        """
        return Box(center=self.center, size=self.size)

    @cached_property
    def zero_dims(self) -> List[Axis]:
        """A list of axes along which the :class:`Box` is zero-sized."""
        return [dim for dim, size in enumerate(self.size) if size == 0]

    @cached_property
    def _normal_2dmaterial(self) -> Axis:
        """Get the normal to the given geometry, checking that it is a 2D geometry."""
        if np.count_nonzero(self.size) != 2:
            raise ValidationError(
                "'Medium2D' requires exactly one of the 'Box' dimensions to have size zero."
            )
        return self.size.index(0)

    def _plot_arrow(
        self,
        direction: Tuple[float, float, float],
        x: float = None,
        y: float = None,
        z: float = None,
        color: str = None,
        alpha: float = None,
        bend_radius: float = None,
        bend_axis: Axis = None,
        both_dirs: bool = False,
        ax: Ax = None,
        arrow_base: Coordinate = None,
    ) -> Ax:
        """Adds an arrow to the axis if with options if certain conditions met.

        Parameters
        ----------
        direction: Tuple[float, float, float]
            Normalized vector describing the arrow direction.
        x : float = None
            Position of plotting plane in x direction.
        y : float = None
            Position of plotting plane in y direction.
        z : float = None
            Position of plotting plane in z direction.
        color : str = None
            Color of the arrow.
        alpha : float = None
            Opacity of the arrow (0, 1)
        bend_radius : float = None
            Radius of curvature for this arrow.
        bend_axis : Axis = None
            Axis of curvature of `bend_radius`.
        both_dirs : bool = False
            If True, plots an arrow ponting in direction and one in -direction.
        arrow_base : :class:`.Coordinate` = None
            Custom base of the arrow. Uses the geometry's center if not provided.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The matplotlib axes with the arrow added.
        """

        plot_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (dx, dy) = self.pop_axis(direction, axis=plot_axis)

        # conditions to check to determine whether to plot arrow, taking into account the
        # possibility of a custom arrow base
        arrow_intersecting_plane = len(self.intersections_plane(x=x, y=y, z=z)) > 0
        center = self.center
        if arrow_base:
            arrow_intersecting_plane = arrow_intersecting_plane and any(
                a == b for a, b in zip(arrow_base, [x, y, z])
            )
            center = arrow_base

        _, (dx, dy) = self.pop_axis(direction, axis=plot_axis)
        components_in_plane = any(not np.isclose(component, 0) for component in (dx, dy))

        # plot if arrow in plotting plane and some non-zero component can be displayed.
        if arrow_intersecting_plane and components_in_plane:
            _, (x0, y0) = self.pop_axis(center, axis=plot_axis)

            # Reasonable value for temporary arrow size.  The correct size and direction
            # have to be calculated after all transforms have been set.  That is why we
            # use a callback to do these calculations only at the drawing phase.
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            v_x = (xmax - xmin) / 10
            v_y = (ymax - ymin) / 10

            directions = (1.0, -1.0) if both_dirs else (1.0,)
            for sign in directions:
                arrow = patches.FancyArrowPatch(
                    (x0, y0),
                    (x0 + v_x, y0 + v_y),
                    arrowstyle=arrow_style,
                    color=color,
                    alpha=alpha,
                    zorder=np.inf,
                )
                # Don't draw this arrow until it's been reshaped
                arrow.set_visible(False)

                callback = self._arrow_shape_cb(
                    arrow, (x0, y0), (dx, dy), sign, bend_radius if bend_axis == plot_axis else None
                )
                callback_id = ax.figure.canvas.mpl_connect("draw_event", callback)

                # Store a reference to the callback because mpl_connect does not.
                arrow.set_shape_cb = (callback_id, callback)

                ax.add_patch(arrow)

        return ax

    @staticmethod
    def _arrow_shape_cb(arrow, pos, direction, sign, bend_radius):
        def _cb(event):
            # We only want to set the shape once, so we disconnect ourselves
            event.canvas.mpl_disconnect(arrow.set_shape_cb[0])

            transform = arrow.axes.transData.transform
            scale = transform((1, 0))[0] - transform((0, 0))[0]
            arrow_length = ARROW_LENGTH * event.canvas.figure.get_dpi() / scale

            if bend_radius:
                v_norm = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
                vx_norm = direction[0] / v_norm
                vy_norm = direction[1] / v_norm
                bend_angle = -sign * arrow_length / bend_radius
                t_x = 1 - np.cos(bend_angle)
                t_y = np.sin(bend_angle)
                v_x = -bend_radius * (vx_norm * t_y - vy_norm * t_x)
                v_y = -bend_radius * (vx_norm * t_x + vy_norm * t_y)
                tangent_angle = np.arctan2(direction[1], direction[0])
                arrow.set_connectionstyle(
                    patches.ConnectionStyle.Angle3(
                        angleA=180 / np.pi * tangent_angle,
                        angleB=180 / np.pi * (tangent_angle + bend_angle),
                    )
                )

            else:
                v_x = sign * arrow_length * direction[0]
                v_y = sign * arrow_length * direction[1]

            arrow.set_positions(pos, (pos[0] + v_x, pos[1] + v_y))
            arrow.set_visible(True)
            arrow.draw(event.renderer)

        return _cb

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""

        volume = 1

        for axis in range(3):
            min_bound = max(self.bounds[0][axis], bounds[0][axis])
            max_bound = min(self.bounds[1][axis], bounds[1][axis])

            volume *= max_bound - min_bound

        return volume

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""

        min_bounds = list(self.bounds[0])
        max_bounds = list(self.bounds[1])

        in_bounds_factor = [2, 2, 2]
        length = [0, 0, 0]

        for axis in (0, 1, 2):
            if min_bounds[axis] < bounds[0][axis]:
                min_bounds[axis] = bounds[0][axis]
                in_bounds_factor[axis] -= 1

            if max_bounds[axis] > bounds[1][axis]:
                max_bounds[axis] = bounds[1][axis]
                in_bounds_factor[axis] -= 1

            length[axis] = max_bounds[axis] - min_bounds[axis]

        return (
            length[0] * length[1] * in_bounds_factor[2]
            + length[1] * length[2] * in_bounds_factor[0]
            + length[2] * length[0] * in_bounds_factor[1]
        )


"""Compound subclasses"""


class ClipOperation(Geometry):
    """Class representing the result of a set operation between geometries."""

    operation: ClipOperationType = pydantic.Field(
        ...,
        title="Operation Type",
        description="Operation to be performed between geometries.",
    )

    geometry_a: annotate_type(GeometryType) = pydantic.Field(
        ...,
        title="Geometry A",
        description="First operand for the set operation. It can be any geometry type, including "
        ":class:`GeometryGroup`.",
    )

    geometry_b: annotate_type(GeometryType) = pydantic.Field(
        ...,
        title="Geometry B",
        description="Second operand for the set operation. It can also be any geometry type.",
    )

    @staticmethod
    def to_polygon_list(base_geometry: Shapely) -> List[Shapely]:
        """Return a list of valid polygons from a shapely geometry, discarding points, lines, and
        empty polygons.

        Parameters
        ----------
        base_geometry : shapely.geometry.base.BaseGeometry
            Base geometry for inspection.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            Valid polygons retrieved from ``base geometry``.
        """
        if base_geometry.geom_type == "GeometryCollection":
            return [p for geom in base_geometry.geoms for p in ClipOperation.to_polygon_list(geom)]
        if base_geometry.geom_type == "MultiPolygon":
            return [p for p in base_geometry.geoms if not p.is_empty]
        if base_geometry.geom_type == "Polygon" and not base_geometry.is_empty:
            return [base_geometry]
        return []

    def intersections_plane(
        self, x: float = None, y: float = None, z: float = None
    ) -> List[Shapely]:
        """Returns list of shapely geomtries at plane specified by one non-None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        geom_a = Geometry.evaluate_inf_shape(
            shapely.unary_union(self.geometry_a.intersections_plane(x, y, z))
        )
        geom_b = Geometry.evaluate_inf_shape(
            shapely.unary_union(self.geometry_b.intersections_plane(x, y, z))
        )
        if self.operation == "union":
            result = ClipOperation.to_polygon_list(shapely.union(geom_a, geom_b))
        elif self.operation == "intersection":
            result = ClipOperation.to_polygon_list(shapely.intersection(geom_a, geom_b))
        elif self.operation == "difference":
            result = ClipOperation.to_polygon_list(shapely.difference(geom_a, geom_b))
        elif self.operation == "symmetric_difference":
            result = ClipOperation.to_polygon_list(shapely.symmetric_difference(geom_a, geom_b))
        else:
            raise ValueError(
                "'operation' must be one of 'union', 'intersection', 'difference', or "
                "'symmetric_difference'."
            )
        return result

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        # Overestimates
        if self.operation == "difference":
            result = self.geometry_a.bounds
        elif self.operation == "intersection":
            bounds = (self.geometry_a.bounds, self.geometry_b.bounds)
            result = (
                tuple(max(b[i] for b, _ in bounds) for i in range(3)),
                tuple(min(b[i] for _, b in bounds) for i in range(3)),
            )
            if any(result[0][i] > result[1][i] for i in range(3)):
                result = ((0, 0, 0), (0, 0, 0))
        else:
            bounds = (self.geometry_a.bounds, self.geometry_b.bounds)
            result = (
                tuple(min(b[i] for b, _ in bounds) for i in range(3)),
                tuple(max(b[i] for _, b in bounds) for i in range(3)),
            )
        return result

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
        inside_a = self.geometry_a.inside(x, y, z)
        inside_b = self.geometry_b.inside(x, y, z)
        if self.operation == "union":
            result = inside_a | inside_b
        elif self.operation == "intersection":
            result = inside_a & inside_b
        elif self.operation == "difference":
            result = inside_a & ~inside_b
        elif self.operation == "symmetric_difference":
            result = inside_a != inside_b
        else:
            raise ValueError(
                "'operation' must be one of 'union', 'intersection', 'difference', or "
                "'symmetric_difference'."
            )
        return result

    def inside_meshgrid(
        self, x: np.ndarray[float], y: np.ndarray[float], z: np.ndarray[float]
    ) -> np.ndarray[bool]:
        """Faster way to check ``self.inside`` on a meshgrid. The input arrays are assumed sorted.

        Parameters
        ----------
        x : np.ndarray[float]
            1D array of point positions in x direction.
        y : np.ndarray[float]
            1D array of point positions in y direction.
        z : np.ndarray[float]
            1D array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            Array with shape ``(x.size, y.size, z.size)``, which is ``True`` for every
            point that is inside the geometry.
        """
        inside_a = self.geometry_a.inside_meshgrid(x, y, z)
        inside_b = self.geometry_b.inside_meshgrid(x, y, z)
        if self.operation == "union":
            result = inside_a | inside_b
        elif self.operation == "intersection":
            result = inside_a & inside_b
        elif self.operation == "difference":
            result = inside_a & ~inside_b
        else:
            result = inside_a != inside_b
        return result

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""
        # Overestimates
        if self.operation == "intersection":
            return min(self.geometry_a.surface_area(bounds), self.geometry_b.surface_area(bounds))
        if self.operation == "difference":
            return self.geometry_a.surface_area(bounds)
        return self.geometry_a.surface_area(bounds) + self.geometry_b.surface_area(bounds)

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""
        # Overestimates
        return self.geometry_a.surface_area(bounds) + self.geometry_b.surface_area(bounds)


class GeometryGroup(Geometry):
    """A collection of Geometry objects that can be called as a single geometry object."""

    geometries: Tuple[annotate_type(GeometryType), ...] = pydantic.Field(
        ...,
        title="Geometries",
        description="Tuple of geometries in a single grouping. "
        "Can provide significant performance enhancement in ``Structure`` when all geometries are "
        "assigned the same medium.",
    )

    @pydantic.validator("geometries", always=True)
    def _geometries_not_empty(cls, val):
        """make sure geometries are not empty."""
        if not len(val) > 0:
            raise ValidationError("GeometryGroup.geometries must not be empty.")
        return val

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float, float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """

        bounds = tuple(geometry.bounds for geometry in self.geometries)
        return (
            tuple(min(b[i] for b, _ in bounds) for i in range(3)),
            tuple(max(b[i] for _, b in bounds) for i in range(3)),
        )

    def intersections_plane(
        self, x: float = None, y: float = None, z: float = None
    ) -> List[Shapely]:
        """Returns list of shapely geomtries at plane specified by one non-None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        if not self.intersects_plane(x, y, z):
            return []
        return [
            intersection
            for geometry in self.geometries
            for intersection in geometry.intersections_plane(x=x, y=y, z=z)
        ]

    def intersects_axis_position(self, axis: float, position: float) -> bool:
        """Whether self intersects plane specified by a given position along a normal axis.

        Parameters
        ----------
        axis : int = None
            Axis nomral to the plane.
        position : float = None
            Position of plane along the normal axis.

        Returns
        -------
        bool
            Whether this geometry intersects the plane.
        """
        return any(geom.intersects_axis_position(axis, position) for geom in self.geometries)

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
        individual_insides = (geometry.inside(x, y, z) for geometry in self.geometries)
        return functools.reduce(lambda a, b: a | b, individual_insides)

    def inside_meshgrid(
        self, x: np.ndarray[float], y: np.ndarray[float], z: np.ndarray[float]
    ) -> np.ndarray[bool]:
        """Faster way to check ``self.inside`` on a meshgrid. The input arrays are assumed sorted.

        Parameters
        ----------
        x : np.ndarray[float]
            1D array of point positions in x direction.
        y : np.ndarray[float]
            1D array of point positions in y direction.
        z : np.ndarray[float]
            1D array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            Array with shape ``(x.size, y.size, z.size)``, which is ``True`` for every
            point that is inside the geometry.
        """
        individual_insides = (geom.inside_meshgrid(x, y, z) for geom in self.geometries)
        return functools.reduce(lambda a, b: a | b, individual_insides)

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""
        individual_volumes = (geometry.volume(bounds) for geometry in self.geometries)
        return np.sum(individual_volumes)

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""
        individual_areas = (geometry.surface_area(bounds) for geometry in self.geometries)
        return np.sum(individual_areas)


from .utils import GeometryType, from_shapely  # noqa: E402
