# pylint:disable=too-many-lines
"""Defines spatial extent of objects."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Any

import pydantic
import numpy as np

from shapely.geometry import Point, Polygon, box
from descartes import PolygonPatch

from .base import Tidy3dBaseModel
from .types import Bound, Size, Coordinate, Axis, Coordinate2D, ArrayLike
from .types import Vertices, Ax, Shapely
from .viz import add_ax_if_none, equal_aspect
from ..log import Tidy3dKeyError, SetupError, ValidationError
from ..constants import MICROMETER

# add this around extents of plots
PLOT_BUFFER = 0.3


class Geometry(Tidy3dBaseModel, ABC):
    """Abstract base class, defines where something exists in space."""

    center: Coordinate = pydantic.Field(
        (0.0, 0.0, 0.0),
        title="Center",
        description="Center of object in x, y, and z.",
        units=MICROMETER,
    )

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
            patch = PolygonPatch(shape, **patch_kwargs)
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
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

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
            return self._intersections_normal()
        return self._intersections_side(position, axis)

    @abstractmethod
    def _intersections_normal(self) -> list:
        """Find shapely geometries intersecting planar geometry with axis normal to slab.

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
        return 2 * np.sqrt(self.radius ** 2 - dz ** 2)


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
        center = tuple((pt_min + pt_max) / 2.0 for pt_min, pt_max in zip(rmin, rmax))
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
        return (dist_x < Lx / 2) * (dist_y < Ly / 2) * (dist_z < Lz / 2)

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
        return (dist_x ** 2 + dist_y ** 2 + dist_z ** 2) <= (self.radius ** 2)

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

    def _intersections_normal(self):
        """Find shapely geometries intersecting cylindrical geometry with axis normal to slab.

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
        inside_radius = (dist_x ** 2 + dist_y ** 2) <= (self.radius ** 2)
        inside_height = dist_z < (self.length / 2)
        return inside_radius * inside_height

    @property
    def _bounds(self):
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
    """Polygon with constant thickness (slab) along axis direction.

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

    vertices: Union[Vertices, ArrayLike] = pydantic.Field(
        ...,
        title="Vertices",
        description="List of (d1, d2) defining the 2 dimensional positions of the polygon "
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
        """makes sure vertices is correct shape if numpy array"""
        if isinstance(val, np.ndarray):
            shape = val.shape
            if len(shape) != 2 or shape[1] != 2:
                raise SetupError(
                    "PolySlab.vertices must be a 2 dimensional array shaped (N, 2).  "
                    f"Given array with shape of {shape}."
                )
        return val

    @pydantic.validator("vertices", always=True)
    def set_center(cls, val, values):
        """sets the .center field using zmin, zmax, and polygon vertices"""
        polygon_face = Polygon(val)
        zmin, zmax = values.get("slab_bounds")
        z0 = (zmin + zmax) / 2.0
        [(x0, y0)] = list(polygon_face.centroid.coords)
        values["center"] = cls.unpop_axis(z0, (x0, y0), axis=values.get("axis"))
        return val

    @classmethod
    def from_gds(  # pylint:disable=too-many-arguments
        cls,
        gds_cell,
        axis: Axis,
        slab_bounds: Tuple[float, float],
        gds_layer: int,
        gds_dtype: int,
        polygon_index: pydantic.NonNegativeInt = 0,
        gds_scale: pydantic.PositiveFloat = 1.0,
    ):
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
        gds_dtype : int
            Data type index in the ``gds_cell``.
        polygon_index : int = 0
            Index into the list of polygons at given ``gds_layer`` and ``gds_dtype``.
            Must be non-negative.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of MICROMETER.
            For example, if gds file uses nanometers, set ``gds_scale=1e-3``.
            Must be positive.

        Returns
        -------
        :class:`PolySlab`
            Geometry with slab along ``axis`` and geometry defined by gds cell in plane.
        """

        vert_dict = gds_cell.get_polygons(by_spec=True)
        try:
            key = (gds_layer, gds_dtype)
            list_of_vertices = vert_dict[key]
        except Exception as e:
            raise Tidy3dKeyError(
                f"Can't load gds_cell, gds_layer={gds_layer} and gds_dtype={gds_dtype} not found.  "
                f"Found (layer, dtype) list of {list(vert_dict.keys())} in cell."
            ) from e

        # select polygon index
        try:
            vertices = list_of_vertices[polygon_index]
        except Exception as e:
            raise Tidy3dKeyError(
                f"No polygons found at polygon_index={polygon_index}.  "
                f"{len(list_of_vertices)} polygons found at "
                f"gds_layer={gds_layer} and gds_dtype={gds_dtype}."
            ) from e

        vertices *= gds_scale
        vertices = vertices.tolist()

        return cls(vertices=vertices, axis=axis, slab_bounds=slab_bounds)

    def inside(self, x, y, z) -> bool:  # pylint:disable=too-many-locals
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
        z0, _ = self.pop_axis(self.center, axis=self.axis)
        dist_z = np.abs(z - z0)
        inside_height = dist_z < (self.length / 2)

        # avoid going into face checking if no points are inside slab bounds
        if not np.any(inside_height):
            return inside_height

        # check what points are inside polygon cross section (face)
        face_polygon = Polygon(self.vertices)
        if isinstance(x, np.ndarray):
            inside_polygon = np.zeros_like(inside_height)
            xs_slab = x[inside_height]
            ys_slab = y[inside_height]

            def contains_pointwise(xy_point):
                point = Point(xy_point)
                return face_polygon.contains(point)

            contains_vectorized = np.vectorize(contains_pointwise, signature="(n)->()")
            points_stacked = np.stack((xs_slab, ys_slab), axis=1)
            inside_polygon_slab = contains_vectorized(points_stacked)
            inside_polygon[inside_height] = inside_polygon_slab
        else:
            point = Point(x, y)
            inside_polygon = face_polygon.contains(point)
        return inside_height * inside_polygon

    def _intersections_normal(self):
        """Find shapely geometries intersecting planar geometry with axis normal to slab

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        return [Polygon(self.vertices)]

    def _intersections_side(self, position, axis) -> list:  # pylint:disable=too-many-locals
        """Find shapely geometries intersecting planar geometry with axis orthogonal to slab

        Parameters
        ----------
        position : float
            Position along ``axis``
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        z0, _ = self.pop_axis(self.center, axis=self.axis)
        z_min, z_max = z0 - self.length / 2, z0 + self.length / 2
        iverts_b, iverts_f = self._find_intersecting_vertices(position, axis)
        ints_y = self._find_intersecting_ys(iverts_b, iverts_f, position)

        # make polygon with intersections and z axis information
        polys = []
        for y_index in range(len(ints_y) // 2):
            y_min = ints_y[2 * y_index]
            y_max = ints_y[2 * y_index + 1]
            minx, miny = self._order_by_axis(plane_val=y_min, axis_val=z_min, axis=axis)
            maxx, maxy = self._order_by_axis(plane_val=y_max, axis_val=z_max, axis=axis)
            polys.append(box(minx=minx, miny=miny, maxx=maxx, maxy=maxy))

        return polys

    def _find_intersecting_vertices(
        self, position: float, axis: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Finds pairs of forward and backwards vertices where polygon intersects position at axis.
           Assumes axis is handles so this function works on xy plane.

        Parameters
        ----------
        position : float
            position along axis
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        np.ndarray, np.ndarray
            Backward (xy) vertices and forward (xy) vertices.
        """

        vertices_b = np.array(self.vertices)

        # if the first coordinate refers to bounds, need to flip the vertices x,y
        if (axis == 2) or ((self.axis == 2) and (axis == 1)):
            vertices_b = np.roll(vertices_b, shift=1, axis=1)

        # get the forward vertices
        vertices_f = np.roll(vertices_b, shift=1, axis=0)

        # find which segments intersect
        intersects_b = np.logical_and((vertices_f[:, 0] <= position), (vertices_b[:, 0] > position))
        intersects_f = np.logical_and((vertices_b[:, 0] <= position), (vertices_f[:, 0] > position))
        intersects_segment = np.logical_or(intersects_b, intersects_f)
        iverts_b = vertices_b[intersects_segment]
        iverts_f = vertices_f[intersects_segment]

        return iverts_b, iverts_f

    @staticmethod
    def _find_intersecting_ys(
        iverts_b: np.ndarray, iverts_f: np.ndarray, position: float
    ) -> List[float]:
        """For each intersecting segment, find intersection point (in y) assuming straight line.

        Parameters
        ----------
        iverts_b : np.ndarray
            Backward (x,y) vertices.
        iverts_f : np.ndarray
            Forward (x,y) vertices.
        position : float
            Position along coordinate x.

        Returns
        -------
        List[float]
            List of intersection points along y direction.
        """

        ints_y = []
        for (vertices_f, vertices_b) in zip(iverts_b, iverts_f):
            x1, y1 = vertices_f
            x2, y2 = vertices_b
            slope = (y2 - y1) / (x2 - x1)
            y = y1 + slope * (position - x1)
            ints_y.append(y)
        ints_y.sort()
        return ints_y

    @property
    def _bounds(self):

        # get the min and max points in polygon plane
        xpoints = tuple(c[0] for c in self.vertices)
        ypoints = tuple(c[1] for c in self.vertices)
        xmin, xmax = min(xpoints), max(xpoints)
        ymin, ymax = min(ypoints), max(ypoints)
        z0, _ = self.pop_axis(self.center, axis=self.axis)
        zmin = z0 - self.length / 2
        zmax = z0 + self.length / 2
        coords_min = self.unpop_axis(zmin, (xmin, ymin), axis=self.axis)
        coords_max = self.unpop_axis(zmax, (xmax, ymax), axis=self.axis)
        return (tuple(coords_min), tuple(coords_max))


# geometries that can be used to define structures.
GeometryFields = (Box, Sphere, Cylinder, PolySlab)
GeometryType = Union[GeometryFields]
