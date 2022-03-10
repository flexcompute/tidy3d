# pylint:disable=too-many-lines
"""Defines spatial extent of objects."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Any

import pydantic
import numpy as np

from shapely.geometry import Point, Polygon, box
from descartes import PolygonPatch

from .base import Tidy3dBaseModel
from .types import Bound, Size, Coordinate, Axis, Coordinate2D, tidynumpy, Array
from .types import Vertices, Ax, Shapely
from .viz import add_ax_if_none, equal_aspect
from .viz import PLOT_BUFFER, ARROW_LENGTH_FACTOR, ARROW_WIDTH_FACTOR, MAX_ARROW_WIDTH_FACTOR
from ..log import Tidy3dKeyError, SetupError, ValidationError
from ..constants import MICROMETER, LARGE_NUMBER, RADIAN

# for sampling polygon in slanted polyslab along  z-direction for
# validating polygon to be non_intersecting.
_N_SAMPLE_POLYGON_INTERSECT = 100


class Geometry(Tidy3dBaseModel, ABC):
    """Abstract base class, defines where something exists in space."""

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

    def inside(self, x, y, z) -> bool:
        """Returns ``True`` if point ``(x,y,z)`` is inside volume of :class:`Geometry`.

        Parameters
        ----------
        x : float
            Position of point in x direction.
        y : float
            Position of point in y direction.
        z : float
            Position of point in z direction.

        Returns
        -------
        bool
            True if point ``(x,y,z)`` is inside geometry.
        """
        shapes_intersect = self.intersections(z=z)
        loc = Point(x, y)
        return any(shape.contains(loc) for shape in shapes_intersect)

    @abstractmethod
    def intersections(self, x: float = None, y: float = None, z: float = None) -> List[Shapely]:
        """Returns list of shapely geoemtries at plane specified by one non-None value of x,y,z.

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
        intersections = self.intersections(x=x, y=y, z=z)
        return bool(intersections)

    @property
    def bounds(self) -> Bound:  # pylint:disable=too-many-locals
        """Returns bounding box min and max coordinates..

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """

        x0, y0, z0 = self.center
        shape_x = self.intersections(x=x0)[0]
        shape_y = self.intersections(y=y0)[0]
        shape_z = self.intersections(z=z0)[0]
        x_miny, x_minz, x_maxy, x_maxz = shape_x.bounds
        y_minx, y_minz, y_maxx, y_maxz = shape_y.bounds
        z_minx, z_miny, z_maxx, z_maxy = shape_z.bounds
        minx = min(y_minx, z_minx)
        maxx = max(y_maxx, z_maxx)
        miny = min(x_miny, z_miny)
        maxy = max(x_maxy, z_maxy)
        minz = min(x_minz, y_minz)
        maxz = max(x_maxz, y_maxz)
        return (minx, miny, minz), (maxx, maxy, maxz)

    @property
    def bounding_box(self):
        """Returns :class:`Box` representation of the bounding box of a :class:`Geometry`.

        Returns
        -------
        :class:`Box`
            Geometric object representing bounding box.
        """
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.bounds
        Lx = xmax - xmin
        Ly = ymax - ymin
        Lz = zmax - zmin
        x0 = (xmax + xmin) / 2.0
        y0 = (ymax + ymin) / 2.0
        z0 = (zmax + zmin) / 2.0
        return Box(center=(x0, y0, z0), size=(Lx, Ly, Lz))

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

    @equal_aspect
    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **patch_kwargs
    ) -> Ax:
        # pylint:disable=line-too-long
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
        # pylint:disable=line-too-long

        # find shapes that intersect self at plane
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        shapes_intersect = self.intersections(x=x, y=y, z=z)

        # for each intersection, plot the shape
        for shape in shapes_intersect:
            _shape = self.evaluate_inf_shape(shape)
            patch = PolygonPatch(_shape, **patch_kwargs)
            ax.add_artist(patch)

        # clean up the axis display
        ax = self.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_aspect("equal")
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")
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
        xmin, xmax, ymin, ymax = self._evaluate_infs(xmin, xmax, ymin, ymax)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    @staticmethod
    def _evaluate_infs(*values):
        """Processes values and evaluates any infs into large (signed) numbers."""
        return map(lambda v: v if not np.isinf(v) else np.sign(v) * LARGE_NUMBER, values)

    @classmethod
    def evaluate_inf_shape(cls, shape: "shapely.Geometry") -> "shapely.Geometry":
        """Returns a copy of shape with inf vertices replaced by large numbers if polygon."""

        if not isinstance(shape, Polygon):
            return shape

        coords = shape.exterior.coords[:]
        new_coords = []
        for (coord_x, coord_y) in coords:
            new_coord = tuple(cls._evaluate_infs(coord_x, coord_y))
            new_coords.append(new_coord)
        return Polygon(new_coords)

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
    def rotate_points(points: Array[float], axis: Coordinate, angle: float) -> Array[float]:
        """Rotate a set of points in 3D.

        Parameters
        ----------
        points : Array[float]
            Array of shape ``(3, ...)``.
        axis : Coordinate
            Axis of rotation
        angle : float
            Angle of rotation counter-clockwise around the axis (rad).
        """

        # Normalized axis vector components
        (ux, uy, uz) = axis / np.linalg.norm(axis)

        # General rotation matrix
        rot_mat = np.zeros((3, 3))
        cos = np.cos(angle)
        sin = np.sin(angle)
        rot_mat[0, 0] = cos + ux**2 * (1 - cos)
        rot_mat[0, 1] = ux * uy * (1 - cos) - uz * sin
        rot_mat[0, 2] = ux * uz * (1 - cos) + uy * sin
        rot_mat[1, 0] = uy * ux * (1 - cos) + uz * sin
        rot_mat[1, 1] = cos + uy**2 * (1 - cos)
        rot_mat[1, 2] = uy * uz * (1 - cos) - ux * sin
        rot_mat[2, 0] = uz * ux * (1 - cos) - uy * sin
        rot_mat[2, 1] = uz * uy * (1 - cos) + ux * sin
        rot_mat[2, 2] = cos + uz**2 * (1 - cos)

        if len(points.shape) == 1:
            return rot_mat @ points

        return np.einsum("ij,jp...->ip...", rot_mat, points)

    def reflect_points(
        self,
        points: Array[float],
        polar_axis: Axis,
        angle_theta: float,
        angle_phi: float,
    ) -> Array[float]:
        """Reflect a set of points in 3D at a plane passing through the coordinate origin defined
        and normal to a given axis defined in polar coordinates (theta, phi) w.r.t. the
        ``polar_axis`` which can be 0, 1, or 2.

        Parameters
        ----------
        points : Array[float]
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


""" Abstract subclasses """


class Planar(Geometry, ABC):
    """Geometry with one ``axis`` that is slab-like with thickness ``height``."""

    axis: Axis = pydantic.Field(
        2, title="Axis", description="Specifies dimension of the planar axis (0,1,2) -> (x,y,z)."
    )
    length: pydantic.NonNegativeFloat = pydantic.Field(
        None,
        title="Length",
        description="Defines thickness of geometry along axis dimension.",
        units=MICROMETER,
    )

    def intersections(self, x: float = None, y: float = None, z: float = None):
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
        if axis == self.axis:
            z0, _ = self.pop_axis(self.center, axis=self.axis)
            if (position < z0 - self.length / 2) or (position > z0 + self.length / 2):
                return []
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

    @property
    def bounds(self):
        """Returns bounding box for planar geometry, may implement for subclasses.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        z0, _ = self.pop_axis(self.center, axis=self.axis)
        z_min = z0 - self.length / 2.0
        z_max = z0 + self.length / 2.0
        shape_top = self.intersections(z=z0)[0]
        (xmin, ymin, xmax, ymax) = shape_top.bounds
        bounds_min = self.unpop_axis(z_min, (xmin, ymin), axis=self.axis)
        bounds_max = self.unpop_axis(z_max, (xmax, ymax), axis=self.axis)
        return bounds_min, bounds_max

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


""" importable geometries """


class Box(Geometry):
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
    def from_bounds(cls, rmin: Coordinate, rmax: Coordinate):
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

        def get_center(pt_min: float, pt_max: float) -> float:
            """Returns center point based on bounds along dimension."""
            if np.isneginf(pt_min) and np.isposinf(pt_max):
                return 0.0
            return (pt_min + pt_max) / 2.0

        center = tuple(get_center(pt_min, pt_max) for pt_min, pt_max in zip(rmin, rmax))
        size = tuple((pt_max - pt_min) for pt_min, pt_max in zip(rmin, rmax))
        return cls(center=center, size=size)

    def intersections(self, x: float = None, y: float = None, z: float = None):
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
        z0, (x0, y0) = self.pop_axis(self.center, axis=axis)
        Lz, (Lx, Ly) = self.pop_axis(self.size, axis=axis)
        dz = np.abs(z0 - position)
        if dz > Lz / 2:
            return []
        return [box(minx=x0 - Lx / 2, miny=y0 - Ly / 2, maxx=x0 + Lx / 2, maxy=y0 + Ly / 2)]

    def inside(self, x, y, z) -> bool:
        """Returns ``True`` if point ``(x,y,z)`` inside volume of geometry.

        Parameters
        ----------
        x : float
            Position of point in x direction.
        y : float
            Position of point in y direction.
        z : float
            Position of point in z direction.

        Returns
        -------
        bool
            Whether point ``(x,y,z)`` is inside geometry.
        """
        x0, y0, z0 = self.center
        Lx, Ly, Lz = self.size
        dist_x = np.abs(x - x0)
        dist_y = np.abs(y - y0)
        dist_z = np.abs(z - z0)
        return (dist_x <= Lx / 2) * (dist_y <= Ly / 2) * (dist_z <= Lz / 2)

    @property
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

    @property
    def geometry(self):
        """:class:`Box` representation of self (used for subclasses of Box).

        Returns
        -------
        :class:`Box`
            Instance of :class:`Box` representing self's geometry.
        """
        return Box(center=self.center, size=self.size)

    def _plot_arrow(  # pylint:disable=too-many-arguments, too-many-locals
        self,
        direction: Tuple[float, float, float],
        x: float = None,
        y: float = None,
        z: float = None,
        color: str = None,
        alpha: float = None,
        length_factor: float = ARROW_LENGTH_FACTOR,
        width_factor: float = ARROW_WIDTH_FACTOR,
        both_dirs: bool = False,
        ax: Ax = None,
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
        length_factor : float = None
            How long the (3D, unprojected) arrow is compared to the min(height, width) of the axes.
        width_factor : float = None
            How wide the (3D, unprojected) arrow is compared to the min(height, width) of the axes.
        both_dirs : bool = False
            If True, plots an arrow ponting in direction and one in -direction.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The matplotlib axes with the arrow added.
        """

        plot_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        arrow_axis = [component == 0 for component in direction]
        arrow_length, arrow_width = self._arrow_dims(ax, length_factor, width_factor)

        # only add arrow if the plotting plane is perpendicular to the source
        if arrow_axis.count(0.0) > 1 or arrow_axis.index(0.0) != plot_axis:
            _, (x0, y0) = self.pop_axis(self.center, axis=plot_axis)
            _, (dx, dy) = self.pop_axis(direction, axis=plot_axis)

            def add_arrow(sign=1.0):
                """Add an arrow to the axes and include a sign to direction."""
                ax.arrow(
                    x=x0,
                    y=y0,
                    dx=sign * arrow_length * dx,
                    dy=sign * arrow_length * dy,
                    width=arrow_width,
                    color=color,
                    alpha=alpha,
                    zorder=np.inf,
                )

            add_arrow(sign=1.0)
            if both_dirs:
                add_arrow(sign=-1.0)

        return ax

    def _arrow_dims(
        self,
        ax: Ax,
        length_factor: float = ARROW_LENGTH_FACTOR,
        width_factor: float = ARROW_WIDTH_FACTOR,
    ) -> Tuple[float, float]:
        """Length and width of arrow based on axes size and length and width factors."""

        # get the sizes of the matplotlib axes
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax_width = xmax - xmin
        ax_height = ymax - ymin

        # apply length factor to the minimum size to get arrow length
        arrow_length = length_factor * min(ax_width, ax_height)

        # constrain arrow width by the maximum size and the max arrow width factor
        arrow_width = width_factor * arrow_length
        arrow_width = min(arrow_width, MAX_ARROW_WIDTH_FACTOR * max(ax_width, ax_height))

        return arrow_length, arrow_width


class Sphere(Circular):
    """Spherical geometry.

    Example
    -------
    >>> b = Sphere(center=(1,2,3), radius=2)
    """

    def inside(self, x, y, z) -> bool:
        """Returns True if point ``(x,y,z)`` inside volume of geometry.

        Parameters
        ----------
        x : float
            Position of point in x direction.
        y : float
            Position of point in y direction.
        z : float
            Position of point in z direction.

        Returns
        -------
        bool
            Whether point ``(x,y,z)`` is inside geometry.
        """
        x0, y0, z0 = self.center
        dist_x = np.abs(x - x0)
        dist_y = np.abs(y - y0)
        dist_z = np.abs(z - z0)
        return (dist_x**2 + dist_y**2 + dist_z**2) <= (self.radius**2)

    def intersections(self, x: float = None, y: float = None, z: float = None):
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
        z0, (x0, y0) = self.pop_axis(self.center, axis=axis)
        intersect_dist = self._intersect_dist(position, z0)
        if not intersect_dist:
            return []
        return [Point(x0, y0).buffer(0.5 * intersect_dist)]

    @property
    def bounds(self):
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float, float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        coord_min = tuple(c - self.radius for c in self.center)
        coord_max = tuple(c + self.radius for c in self.center)
        return (coord_min, coord_max)


class Cylinder(Circular, Planar):
    """Cylindrical geometry.

    Example
    -------
    >>> c = Cylinder(center=(1,2,3), radius=2, length=5, axis=2)
    """

    length: pydantic.NonNegativeFloat = pydantic.Field(
        ...,
        title="Length",
        description="Defines thickness of cylinder along axis dimension.",
        units=MICROMETER,
    )

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
        _, (x0, y0) = self.pop_axis(self.center, axis=self.axis)
        return [Point(x0, y0).buffer(self.radius)]

    def _intersections_side(self, position, axis):
        """Find shapely geometries intersecting cylindrical geometry with axis orthogonal to length.

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
        z0_axis, _ = self.pop_axis(self.center, axis=axis)
        intersect_dist = self._intersect_dist(position, z0_axis)
        if not intersect_dist:
            return []
        Lx, Ly = self._order_by_axis(plane_val=intersect_dist, axis_val=self.length, axis=axis)
        _, (x0_plot_plane, y0_plot_plane) = self.pop_axis(self.center, axis=axis)
        int_box = box(
            minx=x0_plot_plane - Lx / 2,
            miny=y0_plot_plane - Ly / 2,
            maxx=x0_plot_plane + Lx / 2,
            maxy=y0_plot_plane + Ly / 2,
        )
        return [int_box]

    def inside(self, x, y, z) -> bool:
        """Returns True if point ``(x,y,z)`` inside volume of geometry.

        Parameters
        ----------
        x : float
            Position of point in x direction.
        y : float
            Position of point in y direction.
        z : float
            Position of point in z direction.

        Returns
        -------
        bool
            Whether point ``(x,y,z)`` is inside geometry.
        """
        z0, (x0, y0) = self.pop_axis(self.center, axis=self.axis)
        dist_x = np.abs(x - x0)
        dist_y = np.abs(y - y0)
        dist_z = np.abs(z - z0)
        inside_radius = (dist_x**2 + dist_y**2) <= (self.radius**2)
        inside_height = dist_z <= (self.length / 2)
        return inside_radius * inside_height

    @property
    def bounds(self):
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float, float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        coord_min = list(c - self.radius for c in self.center)
        coord_max = list(c + self.radius for c in self.center)
        coord_min[self.axis] = self.center[self.axis] - self.length / 2.0
        coord_max[self.axis] = self.center[self.axis] + self.length / 2.0
        return (tuple(coord_min), tuple(coord_max))


class PolySlab(Planar):
    """Polygon extruded with optional sidewall angle along axis direction.

    Example
    -------
    >>> vertices = np.array([(0,0), (1,0), (1,1)])
    >>> p = PolySlab(vertices=vertices, axis=2, slab_bounds=(-1, 1))
    """

    slab_bounds: Tuple[float, float] = pydantic.Field(
        ...,
        title="Slab Bounds",
        description="Minimum and maximum positions of the slab along axis dimension.",
        units=MICROMETER,
    )

    dilation: float = pydantic.Field(
        0.0,
        title="Dilation",
        description="Dilation of the polygon in the base by shifting each edge along its "
        "normal outwards direction by a distance; a negative value corresponds to erosion.",
        units=MICROMETER,
    )

    sidewall_angle: float = pydantic.Field(
        0.0,
        title="Sidewall angle",
        description="Angle of the sidewall. "
        "``sidewall_angle=0`` (default) specifies vertical wall, "
        "while ``0<sidewall_angle<np.pi/2`` for the base to be larger than the top.",
        # "and ``sidewall_angle<0`` for base to be smaller than the top.",
        ge=0.0,
        lt=np.pi / 2,
        units=RADIAN,
    )

    vertices: Union[Vertices, tidynumpy] = pydantic.Field(
        ...,
        title="Vertices",
        description="List of (d1, d2) defining the 2 dimensional positions of the base polygon "
        "face vertices along dimensions parallel to slab normal axis.",
        units=MICROMETER,
    )

    @pydantic.validator("axis", always=True)
    def supports_z_axis_only(cls, val):
        """PolySlab can only be oriented in z right now."""
        if val != 2:
            raise ValidationError(
                "PolySlab can only support axis=2 in this version of Tidy3D."
                "Support for slabs oriented in other axes will be available in future releases."
            )
        return val

    @pydantic.validator("slab_bounds", always=True)
    def set_length(cls, val, values):
        """sets the .length field using zmin, zmax"""
        zmin, zmax = val
        values["length"] = zmax - zmin
        return val

    @pydantic.validator("vertices", always=True)
    def correct_shape(cls, val):
        """makes sure vertices size is correct. Remove duplicate
        neighbouring vertices, orient vertices in CCW direction.
        make sure no intersecting edges.
        """

        val_np = PolySlab.vertices_to_array(val)
        shape = val_np.shape

        # overall shape of vertices
        if len(shape) != 2 or shape[1] != 2:
            raise SetupError(
                "PolySlab.vertices must be a 2 dimensional array shaped (N, 2).  "
                f"Given array with shape of {shape}."
            )

        # remove duplicate neighbouring vertices, and orient in CCW direction
        vertices = PolySlab._proper_vertices(val)
        if vertices.shape[0] < 3:
            raise SetupError(
                "At least 3 vertices (after removing duplicate neighbouring vertices) "
                "need to be supplied to setup the polygon."
            )

        if not Polygon(vertices).is_valid:
            raise SetupError(
                "A valid Polygon may not possess any intersecting or overlapping edges."
            )

        return PolySlab.array_to_vertices(val_np)

    @pydantic.validator("vertices", always=True)
    def set_center(cls, val, values):
        """sets the .center field using zmin, zmax, and polygon vertices"""
        polygon_face = Polygon(val)
        zmin, zmax = values.get("slab_bounds")
        if np.isneginf(zmin) and np.isposinf(zmax):
            z0 = 0.0
        else:
            z0 = (zmin + zmax) / 2.0
        [(x0, y0)] = list(polygon_face.centroid.coords)
        values["center"] = cls.unpop_axis(z0, (x0, y0), axis=values.get("axis"))
        return val

    @pydantic.validator("vertices", always=True)
    def no_self_intersecting_polygon(cls, val, values):
        """In this version, we don't support self-intersecting polygons yet, meaning that
        any normal cross section of the PolySlab cannot be self-intersecting.
        For PolySlab of non-zero dilation or sidewall angle values, self-intersection can
        occur even if the supplied vertices make a valid polygon. The
        non-self-intersecting criteria will be validated here.

        There are two types of self-intersection that can occur during dilation:
        1) vertex-vertex crossing. This is well treated. A maximal dilation value will be
        suggested if crossing is detected.

        2) vertex-edge crossing. The implementation of this part needs improvement. Now we
        just sample _N_SAMPLE_POLYGON_INTERSECT cross sections along the normal axis, and check
        if they are self-intersecting.
        """

        ## First, make sure no vertex-vertex crossing in the base
        ## 1) obviously, vertex-vertex crossing can occur during erosion
        ## 2) for concave polygon, the crossing can even occur during dilation
        val_np = PolySlab._proper_vertices(val)
        base = PolySlab._shift_vertices(val_np, values["dilation"])[0]

        # compute distance between vertices after dilation to detect vertex-vertex
        # crossing events
        cross_val, max_dist = PolySlab._crossing_detection(val_np, values["dilation"])
        if cross_val:
            # 1) crossing during erosion
            if values["dilation"] < 0:
                # too much erosion
                raise SetupError(
                    "Erosion value (-dilation) is too large. Some edges in the base polygon "
                    f"are fully eroded. Maximal erosion should be {max_dist:.3e} "
                    "for this polygon. Support for vertices crossing under "
                    "significant erosion will be available in future releases."
                )
            # 2) crossing during dilation in concave polygon
            raise SetupError(
                "Dilation value is too large in a concave polygon, resulting in "
                "vertices crossing. "
                f"Maximal dilation should be {max_dist:.3e} for this polygon. "
                "Support for vertices crossing under significant dilation "
                "in concave polygons will be available in future releases."
            )
        # If no vertex-vertex crossing is detected, but the polygon is still self-intersecting.
        # it is attributed to vertex-edge crossing.
        if not Polygon(base).is_valid:
            raise SetupError(
                "Dilation/Erosion value is too large, resulting in "
                "vertex-edge crossing, and thus self-intersecting polygons. "
                "Support for self-intersecting polygons under significant dilation "
                "will be available in future releases."
            )

        if np.isclose(values["sidewall_angle"], 0):
            return val

        # For Slanted PolySlab. Similar procedure to validate
        # Fist, no vertex-vertex crossing at any point during extrusion
        dist = -values["length"] * np.tan(values["sidewall_angle"])
        cross_val, max_dist = PolySlab._crossing_detection(base, dist)
        if cross_val:
            max_thick = max_dist / np.abs(dist) * values["length"]
            raise SetupError(
                "Sidewall angle or structure thickness is so large that there are "
                "vertices crossing somewhere during extrusion. "
                "Please reduce structure thickness "
                f"to be < {max_thick:.3e} to avoid the crossing of polygon vertices.  "
                "Support for vertices crossing will be available in future releases."
            )

        # sample _N_SAMPLE_POLYGON_INTERSECT cross sections along the normal axis to check
        # if there is any vertex-edge crossing event.
        dist_list = dist * np.linspace(
            1.0 / (_N_SAMPLE_POLYGON_INTERSECT + 1),
            _N_SAMPLE_POLYGON_INTERSECT / (_N_SAMPLE_POLYGON_INTERSECT + 1.0),
            _N_SAMPLE_POLYGON_INTERSECT,
        )
        for dist_i in dist_list:
            poly_i = PolySlab._shift_vertices(base, dist_i)[0]
            if not Polygon(poly_i).is_valid:
                raise SetupError(
                    "Sidewall angle or structure thickness is so large that there are "
                    "vertex-edge crossing somewhere during extrusion, resulting in an "
                    "self-intersecting polygon. "
                    "Please reduce structure thickness. "
                    "Support for self-intersecting polygon will be available "
                    "in future releases."
                )
        return val

    @classmethod
    def from_gds(  # pylint:disable=too-many-arguments
        cls,
        gds_cell,
        axis: Axis,
        slab_bounds: Tuple[float, float],
        gds_layer: int,
        gds_dtype: int = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
        dilation: float = 0.0,
        sidewall_angle: float = 0,
    ) -> List["PolySlab"]:
        """Import :class:`PolySlab` from a ``gdspy.Cell``.

        Parameters
        ----------
        gds_cell : gdspy.Cell
            ``gdspy.Cell`` containing 2D geometric data.
        axis : int
            Integer index into the polygon's slab axis. (0,1,2) -> (x,y,z).
        slab_bounds: Tuple[float, float]
            Minimum and maximum positions of the slab along ``axis``.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``.
            If ``None``, imports all data for this layer into the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of MICROMETER.
            For example, if gds file uses nanometers, set ``gds_scale=1e-3``.
            Must be positive.
        dilation : float = 0.0
            Dilation of the polygon in the base by shifting each edge along its
            normal outwards direction by a distance;
            a negative value corresponds to erosion.
        sidewall_angle : float = 0
            Angle of the sidewall.
            ``sidewall_angle=0`` (default) specifies vertical wall,
            while ``0<sidewall_angle<np.pi/2`` for the base to be larger than the top.

        Returns
        -------
        List[:class:`PolySlab`]
            List of :class:`PolySlab` objects sharing ``axis`` and  slab bound properties.
        """

        # load the polygon vertices
        vert_dict = gds_cell.get_polygons(by_spec=True)
        all_vertices = []
        for (gds_layer_file, gds_dtype_file), vertices in vert_dict.items():
            if gds_layer_file == gds_layer:
                if gds_dtype is None or gds_dtype == gds_dtype_file:
                    for shape_vertices in vertices:
                        all_vertices.append(shape_vertices)

        # make sure something got loaded, otherwise error
        if len(all_vertices) == 0:
            raise Tidy3dKeyError(
                f"Couldn't load gds_cell, no vertices found at gds_layer={gds_layer} "
                f"with specified gds_dtype={gds_dtype}."
            )

        # apply scaling and convert vertices into polyslabs
        all_vertices = [vertices * gds_scale for vertices in all_vertices]
        all_vertices = [vertices.tolist() for vertices in all_vertices]
        return [
            cls(
                vertices=verts,
                axis=axis,
                slab_bounds=slab_bounds,
                dilation=dilation,
                sidewall_angle=sidewall_angle,
            )
            for verts in all_vertices
        ]

    @property
    def _tanq(self) -> float:
        """
        tan(sidewall_angle). _tanq*height gives the offset value
        """
        return np.tan(self.sidewall_angle)

    @property
    def base_polygon(self) -> tidynumpy:
        """The polygon at the base after potential dilation operation.

        Returns
        -------
        tidynumpy
            The vertices of the polygon at the base.
        """

        return self._shift_vertices(self._proper_vertices(self.vertices), self.dilation)[0]

    @property
    def top_polygon(self) -> tidynumpy:
        """The polygon at the top after potential dilation and sidewall operation.

        Returns
        -------
        tidynumpy
            The vertices of the polygon at the top.
        """

        dist = -self.length * self._tanq
        return self._shift_vertices(self.base_polygon, dist)[0]

    def inside(self, x, y, z) -> bool:  # pylint:disable=too-many-locals
        """Returns True if point ``(x,y,z)`` inside volume of geometry.
        For slanted polyslab and x/y/z to be np.ndarray, a loop over z-axis
        is performed to find out the offsetted polygon at each z-coordinate.

        Parameters
        ----------
        x : float
            Position of point in x direction.
        y : float
            Position of point in y direction.
        z : float
            Position of point in z direction.

        Returns
        -------
        bool
            Whether point ``(x,y,z)`` is inside geometry.
        """

        z0, _ = self.pop_axis(self.center, axis=self.axis)
        dist_z = np.abs(z - z0)
        inside_height = dist_z <= (self.length / 2)

        # avoid going into face checking if no points are inside slab bounds
        if not np.any(inside_height):
            return inside_height

        # check what points are inside polygon cross section (face)
        z_local = z - z0 + self.length / 2  # distance to the base
        dist = -z_local * self._tanq

        def contains_pointwise(face_polygon):
            def fun_contain(xy_point):
                point = Point(xy_point)
                return face_polygon.covers(point)

            return fun_contain

        if isinstance(x, np.ndarray):
            inside_polygon = np.zeros_like(inside_height)
            xs_slab = x[inside_height]
            ys_slab = y[inside_height]

            # vertical sidewall
            if np.isclose(self.sidewall_angle, 0):
                face_polygon = Polygon(self.base_polygon)
                fun_contain = contains_pointwise(face_polygon)
                contains_vectorized = np.vectorize(fun_contain, signature="(n)->()")
                points_stacked = np.stack((xs_slab, ys_slab), axis=1)
                inside_polygon_slab = contains_vectorized(points_stacked)
                inside_polygon[inside_height] = inside_polygon_slab
            # slanted sidewall, offsetting vertices at each z
            else:
                for z_i in range(z.shape[2]):
                    if not inside_height[0, 0, z_i]:
                        continue
                    vertices_z = self._shift_vertices(self.base_polygon, dist[0, 0, z_i])[0]
                    face_polygon = Polygon(vertices_z)
                    fun_contain = contains_pointwise(face_polygon)
                    contains_vectorized = np.vectorize(fun_contain, signature="(n)->()")
                    points_stacked = np.stack((x[:, :, 0].flatten(), y[:, :, 0].flatten()), axis=1)
                    inside_polygon_slab = contains_vectorized(points_stacked)
                    inside_polygon[:, :, z_i] = inside_polygon_slab.reshape(x.shape[:2])
        else:
            vertices_z = self._shift_vertices(self.base_polygon, dist)[0]
            face_polygon = Polygon(vertices_z)
            point = Point(x, y)
            inside_polygon = face_polygon.covers(point)
        return inside_height * inside_polygon

    def _intersections_normal(self, z: float):
        """Find shapely geometries intersecting planar geometry with axis normal to slab.

        Parameters
        ----------
        z : float
            Position along the axis normal to slab.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        z0, _ = self.pop_axis(self.center, axis=self.axis)
        z_local = z - z0 + self.length / 2  # distance to the base
        dist = -z_local * self._tanq
        vertices_z = self._shift_vertices(self.base_polygon, dist)[0]
        return [Polygon(vertices_z)]

    def _intersections_side(self, position, axis) -> list:  # pylint:disable=too-many-locals
        """Find shapely geometries intersecting planar geometry with axis orthogonal to slab.

        For slanted polyslab, the procedure is as follows,
        1) Find out all z-coordinates where the plane will intersect directly with a vertex.
        Denote the coordinates as (z_0, z_1, z_2, ... )
        2) Find out all polygons that can be formed between z_i and z_{i+1}. There are two
        types of polygons:
            a) formed by the plane intersecting the edges
            b) formed by the plane intersecting the vertices.
            For either type, one needs to compute:
                i) intersecting position
                ii) angle between the plane and the intersecting edge
            For a), both are straightforward to compute; while for b), one needs to compute
            which edge the plane will slide into.
        3) Looping through z_i, and merge all polygons. The partition by z_i is because once
        the plane intersects the vertex, it can intersect with other edges during
        the extrusion.

        Parameters
        ----------
        position : float
            Position along ``axis``.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        # find out all z_i where the plane will intersect the vertex
        z0, _ = self.pop_axis(self.center, axis=self.axis)
        z_base = z0 - self.length / 2
        height_list = self._find_intersecting_height(position, axis)
        polys = []

        # looping through z_i to assemble the polygons
        height = 0.0
        height_list = np.append(height_list, self.length)
        for h_local in height_list:
            dist = -height * self._tanq
            vertices = self._shift_vertices(self.base_polygon, dist)[0]
            z_min, z_max = z_base + height, z_base + height + h_local
            # for vertical sidewall, no need for complications
            if np.isclose(self.sidewall_angle, 0):
                ints_y, ints_angle = self._find_intersecting_ys_angle_vertical(
                    vertices, position, axis
                )
            else:
                ints_y, ints_angle = self._find_intersecting_ys_angle_slant(
                    vertices, position, axis
                )

            # make polygon with intersections and z axis information
            for y_index in range(len(ints_y) // 2):
                y_min = ints_y[2 * y_index]
                y_max = ints_y[2 * y_index + 1]
                minx, miny = self._order_by_axis(plane_val=y_min, axis_val=z_min, axis=axis)
                maxx, maxy = self._order_by_axis(plane_val=y_max, axis_val=z_max, axis=axis)

                if np.isclose(self.sidewall_angle, 0):
                    polys.append(box(minx=minx, miny=miny, maxx=maxx, maxy=maxy))
                else:
                    angle_min = ints_angle[2 * y_index]
                    angle_max = ints_angle[2 * y_index + 1]

                    angle_min = np.arctan(np.tan(self.sidewall_angle) / np.sin(angle_min))
                    angle_max = np.arctan(np.tan(self.sidewall_angle) / np.sin(angle_max))

                    dy_min = h_local * np.tan(angle_min)
                    dy_max = h_local * np.tan(angle_max)

                    x1, y1 = self._order_by_axis(plane_val=y_min, axis_val=z_min, axis=axis)
                    x2, y2 = self._order_by_axis(plane_val=y_max, axis_val=z_min, axis=axis)

                    if y_max - y_min <= dy_min + dy_max:
                        # intersect before reaching top of polygon
                        # make triangle
                        h_mid = (y_max - y_min) / (dy_min + dy_max) * h_local
                        z_mid = z_min + h_mid
                        y_mid = y_min + dy_min / h_local * h_mid
                        x3, y3 = self._order_by_axis(plane_val=y_mid, axis_val=z_mid, axis=axis)
                        vertices = ((x1, y1), (x2, y2), (x3, y3))
                        polys.append(Polygon(vertices))
                    else:
                        x3, y3 = self._order_by_axis(
                            plane_val=y_max - dy_max, axis_val=z_max, axis=axis
                        )
                        x4, y4 = self._order_by_axis(
                            plane_val=y_min + dy_min, axis_val=z_max, axis=axis
                        )

                        vertices = ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
                        polys.append(Polygon(vertices))

            height += h_local

        return polys

    def _find_intersecting_height(self, position: float, axis: int) -> np.ndarray:
        """Found a list of height where the plane will intersect with the vertices;
        For vertical sidewall, just return np.array([]).
        Assumes axis is handles so this function works on xy plane.

        Parameters
        ----------
        position : float
            position along axis.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        np.ndarray
            Height (relative to the base) where the plane will intersect with vertices.
        """
        if np.isclose(self.sidewall_angle, 0):
            return np.array([])

        vertices = self.base_polygon.copy()

        # shift rate
        dist = 1.0
        shift_x, shift_y = PolySlab._shift_vertices(vertices, dist)[2]
        shift_val = shift_x if axis == 0 else shift_y
        shift_val[np.isclose(shift_val, 0)] = np.inf  # for static vertices

        # distance to the plane in the direction of vertex shifting
        distance = vertices[:, axis] - position
        height = distance / self._tanq / shift_val
        height = np.unique(height)
        if self.sidewall_angle < 0:
            height *= -1

        height = height[height > 0]
        height = height[height < self.length]
        return height

    def _find_intersecting_ys_angle_vertical(  # pylint:disable=too-many-locals
        self, vertices: np.ndarray, position: float, axis: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finds pairs of forward and backwards vertices where polygon intersects position at axis,
        Find intersection point (in y) assuming straight line,and intersecting angle between plane
        and edges. (For unslanted polyslab).
           Assumes axis is handles so this function works on xy plane.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        position : float
            position along axis.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        Union[np.ndarray, np.ndarray]
            List of intersection points along y direction.
            List of angles between plane and edges.
        """

        vertices_axis = vertices.copy()
        # if the first coordinate refers to bounds, need to flip the vertices x,y
        if (axis == 2) or ((self.axis == 2) and (axis == 1)):
            vertices_axis = np.roll(vertices_axis, shift=1, axis=1)

        # get the forward vertices
        vertices_f = np.roll(vertices_axis, shift=-1, axis=0)

        # find which segments intersect
        intersects_b = np.logical_and(
            (vertices_f[:, 0] <= position), (vertices_axis[:, 0] > position)
        )
        intersects_f = np.logical_and(
            (vertices_axis[:, 0] <= position), (vertices_f[:, 0] > position)
        )
        intersects_segment = np.logical_or(intersects_b, intersects_f)
        iverts_b = vertices_axis[intersects_segment]
        iverts_f = vertices_f[intersects_segment]

        # intersecting positions and angles
        ints_y = []
        ints_angle = []
        for (vertices_f_local, vertices_b_local) in zip(iverts_b, iverts_f):
            x1, y1 = vertices_f_local
            x2, y2 = vertices_b_local
            slope = (y2 - y1) / (x2 - x1)
            y = y1 + slope * (position - x1)
            ints_y.append(y)
            ints_angle.append(np.pi / 2 - np.arctan(np.abs(slope)))

        ints_y = np.array(ints_y)
        ints_angle = np.array(ints_angle)

        sort_index = np.argsort(ints_y)
        ints_y_sort = ints_y[sort_index]
        ints_angle_sort = ints_angle[sort_index]

        return ints_y_sort, ints_angle_sort

    def _find_intersecting_ys_angle_slant(  # pylint:disable=too-many-locals, too-many-statements
        self, vertices: np.ndarray, position: float, axis: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finds pairs of forward and backwards vertices where polygon intersects position at axis,
        Find intersection point (in y) assuming straight line,and intersecting angle between plane
        and edges. (For slanted polyslab)
           Assumes axis is handles so this function works on xy plane.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        position : float
            position along axis.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        Union[np.ndarray, np.ndarray]
            List of intersection points along y direction.
            List of angles between plane and edges.
        """

        vertices_axis = vertices.copy()
        # if the first coordinate refers to bounds, need to flip the vertices x,y
        if (axis == 2) or ((self.axis == 2) and (axis == 1)):
            vertices_axis = np.roll(vertices_axis, shift=1, axis=1)

        # get the forward vertices
        vertices_f = np.roll(vertices_axis, shift=-1, axis=0)
        # get the backward vertices
        vertices_b = np.roll(vertices_axis, shift=1, axis=0)

        ## First part, plane intersects with edges, same as vertical
        ints_y, ints_angle = self._find_intersecting_ys_angle_vertical(vertices, position, axis)
        ints_y = ints_y.tolist()
        ints_angle = ints_angle.tolist()

        ## Second part, plane intersects directly with vertices
        # vertices on the intersection
        intersects_on = np.isclose(vertices_axis[:, 0], position)
        iverts_on = vertices_axis[intersects_on]
        # position of the neighbouring vertices
        iverts_b = vertices_b[intersects_on]
        iverts_f = vertices_f[intersects_on]
        # shift rate
        dist = -np.sign(self.sidewall_angle)
        shift_x, shift_y = self._shift_vertices(vertices, dist)[2]
        shift_val = shift_x if axis == 0 else shift_y
        shift_val = shift_val[intersects_on]

        for (vertices_f_local, vertices_b_local, vertices_on_local, shift_local) in zip(
            iverts_f, iverts_b, iverts_on, shift_val
        ):
            x_on, y_on = vertices_on_local
            x_f, y_f = vertices_f_local
            x_b, y_b = vertices_b_local
            # case 1, neighbouring vertices on opposite side
            if (x_b - position) * (x_f - position) < 0:
                ints_y.append(y_on)
                # vertices keep on the plane
                if np.isclose(shift_local, 0):
                    ints_angle.append(np.pi / 2)
                else:
                    x1, y1 = x_b, y_b
                    # decide which edge the plane will slide into
                    if shift_local * (x_f - position) < 0:
                        x1, y1, = (
                            x_f,
                            y_f,
                        )
                    slope = (y_on - y1) / (x_on - x1)
                    ints_angle.append(np.pi / 2 - np.arctan(np.abs(slope)))
            # case 2, neightbouring vertices on the same side:
            elif (x_b - position) * (x_f - position) > 0:
                # this vertex is no longer relevant
                if (x_f - position) * shift_local > 0:
                    continue
                # the vertex will split into two
                ints_y.append(y_on)
                ints_y.append(y_on)
                slope = (y_on - y_b) / (x_on - x_b)
                ints_angle.append(np.pi / 2 - np.arctan(np.abs(slope)))
                slope = (y_on - y_f) / (x_on - x_f)
                ints_angle.append(np.pi / 2 - np.arctan(np.abs(slope)))
            # case 3, one of neightbouring vertices on the plane too:
            else:
                # now only for angle<np.pi/2, so it's not relevant
                # needs to implement here for angle>pi/2
                continue

        ints_y = np.array(ints_y)
        ints_angle = np.array(ints_angle)

        sort_index = np.argsort(ints_y)
        ints_y_sort = ints_y[sort_index]
        ints_angle_sort = ints_angle[sort_index]

        return ints_y_sort, ints_angle_sort

    @property
    def _bounds(self):

        # get the min and max points in polygon plane
        xpoints_base = tuple(c[0] for c in self.base_polygon)
        ypoints_base = tuple(c[1] for c in self.base_polygon)
        xpoints_top = tuple(c[0] for c in self.top_polygon)
        ypoints_top = tuple(c[1] for c in self.top_polygon)
        xmin = min(min(xpoints_base), min(xpoints_top))
        ymin = min(min(ypoints_base), min(ypoints_top))
        xmax = max(max(xpoints_base), max(xpoints_top))
        ymax = max(max(ypoints_base), max(ypoints_top))
        z0, _ = self.pop_axis(self.center, axis=self.axis)
        zmin = z0 - self.length / 2
        zmax = z0 + self.length / 2
        coords_min = self.unpop_axis(zmin, (xmin, ymin), axis=self.axis)
        coords_max = self.unpop_axis(zmax, (xmax, ymax), axis=self.axis)
        return (tuple(coords_min), tuple(coords_max))

    @staticmethod
    def _area(vertices: np.ndarray) -> float:
        """Compute the signed polygon area (positive for CCW orientation).

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        float
            Signed polygon area (positive for CCW orientation).
        """
        vert_shift = np.roll(vertices.copy(), axis=0, shift=-1)
        term1 = vertices[:, 0] * vert_shift[:, 1]
        term2 = vertices[:, 1] * vert_shift[:, 0]

        return np.sum(term1 - term2) * 0.5

    @staticmethod
    def _orient(vertices: np.ndarray) -> np.ndarray:
        """Return a CCW-oriented polygon.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        np.ndarray
            Vertices of a CCW-oriented polygon.
        """
        if PolySlab._area(vertices) > 0:
            return vertices

        return vertices[::-1, :]

    @staticmethod
    def _remove_duplicate_vertices(vertices: np.ndarray) -> np.ndarray:
        """Remove redundant/identical nearest neighbour vertices.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        np.ndarray
            Vertices of polygon.
        """

        vertices_f = np.roll(vertices.copy(), shift=-1, axis=0)
        vertices_diff = np.linalg.norm(vertices - vertices_f, axis=1)
        return vertices[~np.isclose(vertices_diff, 0)]

    @staticmethod
    def _crossing_detection(vertices: np.ndarray, dist: float) -> Tuple[bool, float]:
        """Detect if vertices will cross after a dilation distance dist.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        dist : float
            Distance to offset.

        Returns
        -------
        Tuple[bool,float]
            True if there are any crossings;
            if True, return the maximal allowed dilation.
        """

        # edge length
        vs_orig = vertices.T.copy()
        vs_next = np.roll(vs_orig.copy(), axis=-1, shift=-1)
        edge_length = np.linalg.norm(vs_next - vs_orig, axis=0)

        # edge length remaining
        parallel_shift = PolySlab._shift_vertices(vertices, dist)[1]
        parallel_shift_p = np.roll(parallel_shift.copy(), shift=-1)
        edge_reduction = -(parallel_shift + parallel_shift_p)
        length_remaining = edge_length - edge_reduction

        if np.any(length_remaining < 0):
            index_oversized = length_remaining < 0
            max_dist = np.min(edge_length[index_oversized] / edge_reduction[index_oversized])
            max_dist *= np.abs(dist)
            return True, max_dist
        return False, None

    @staticmethod
    def array_to_vertices(arr_vertices: np.ndarray) -> Vertices:
        """Converts a numpy array of vertices to a list of tuples."""
        return [(x, y) for (x, y) in arr_vertices]  # pylint:disable=unnecessary-comprehension

    @staticmethod
    def _proper_vertices(vertices: Vertices) -> np.ndarray:
        """convert vertices to np.array format,
        removing duplicate neighbouring vertices,
        and oriented in CCW direction.

        Returns
        -------
        tidynumpy
           The vertices of the polygon for internal use.
        """

        vertices_np = PolySlab.vertices_to_array(vertices)
        return PolySlab._orient(PolySlab._remove_duplicate_vertices(vertices_np))

    @staticmethod
    def vertices_to_array(vertices_tuple: Vertices) -> np.ndarray:
        """Converts a list of tuples (vertices) to a numpy array."""
        return np.array(vertices_tuple)

    @staticmethod
    def _shift_vertices(  # pylint:disable=too-many-locals
        vertices: np.ndarray, dist
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Shifts the vertices of a polygon outward uniformly by distances
        `dists`.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        dist : float
            Distance to offset.

        Returns
        -------
        Tuple[np.ndarray, np.narray,Tuple[np.ndarray,np.ndarray]]
            New polygon vertices;
            and the shift of vertices in direction parallel to the edges.
            Shift along x and y direction.
        """

        if np.isclose(dist, 0):
            return vertices, np.zeros(vertices.shape[0], dtype=float), None

        def rot90(v):
            """90 degree rotation of 2d vector
            vx -> vy
            vy -> -vx
            """
            vxs, vys = v
            return np.stack((-vys, vxs), axis=0)

        def cross(u, v):
            return np.cross(u, v, axis=0)

        def normalize(v):
            return v / np.linalg.norm(v, axis=0)

        vs_orig = vertices.T.copy()
        vs_next = np.roll(vs_orig.copy(), axis=-1, shift=-1)
        vs_previous = np.roll(vs_orig.copy(), axis=-1, shift=+1)

        asp = normalize(vs_next - vs_orig)
        asm = normalize(vs_orig - vs_previous)

        # the vertex shift is decomposed into parallel and perpendicular directions
        perpendicular_shift = -dist
        det = cross(asm, asp)

        tan_half_angle = np.where(
            np.isclose(det, 0), 0.0, cross(asm, rot90(asm - asp)) / (det + np.isclose(det, 0))
        )
        parallel_shift = dist * tan_half_angle

        shift_total = perpendicular_shift * rot90(asm) + parallel_shift * asm
        shift_x = shift_total[0, :]
        shift_y = shift_total[1, :]

        return np.swapaxes(vs_orig + shift_total, -2, -1), parallel_shift, (shift_x, shift_y)


# geometries that can be used to define structures.
GeometryFields = (Box, Sphere, Cylinder, PolySlab)
GeometryType = Union[GeometryFields]
