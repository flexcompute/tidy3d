# pylint:disable=too-many-lines, too-many-arguments
"""Defines spatial extent of objects."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Any, Callable, Optional
from math import isclose
import functools

import pydantic
import numpy as np
from matplotlib import patches, path
from shapely import unary_union
from shapely.geometry import Point, Polygon, box, MultiPolygon, LineString
from shapely.validation import make_valid

from .base import Tidy3dBaseModel, cached_property
from .types import Bound, Size, Coordinate, Axis, Coordinate2D, ArrayFloat3D, PlanePosition
from .types import Vertices, Ax, Shapely, annotate_type
from .viz import add_ax_if_none, equal_aspect
from .viz import PLOT_BUFFER, ARROW_LENGTH, arrow_style
from .viz import PlotParams, plot_params_geometry, polygon_patch
from ..log import log
from ..exceptions import Tidy3dKeyError, SetupError, ValidationError, DataError
from ..constants import MICROMETER, LARGE_NUMBER, RADIAN, fp_eps, inf
from .data.dataset import TriangleMeshDataset
from .data.data_array import TriangleMeshDataArray, DATA_ARRAY_MAP
from .transformation import RotationAroundAxis

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except Exception:  # pylint:disable=broad-except
    TRIMESH_AVAILABLE = False

try:
    import networkx  # pylint:disable=unused-import
    import rtree  # pylint:disable=unused-import

    NETWORKX_RTREE_AVAILABLE = True
except Exception:  # pylint:disable=broad-except
    NETWORKX_RTREE_AVAILABLE = False


# sampling polygon along dilation for validating polygon to be
# non self-intersecting during the entire dilation process
_N_SAMPLE_POLYGON_INTERSECT = 5
# for sampling conical frustum in visualization
_N_SAMPLE_CURVE_SHAPELY = 40
_IS_CLOSE_RTOL = np.finfo(float).eps
# for shapely circular shapes discretization in visualization
_N_SHAPELY_QUAD_SEGS = 200
# polygon merge
POLY_GRID_SIZE = 1e-12


Points = ArrayFloat3D


# pylint:disable=too-many-public-methods
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
            loc = Point(x, y)
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
        shapes = set(np.array(arr).shape for arr in arrays)
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
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        # Verify 2D
        if plane.size.count(0.0) != 1:
            raise ValueError("Input geometry must be a 2D Box.")

        # dont bother if the geometry doesn't intersect the plane at all
        if not self.intersects(plane):
            return []

        # get list of Shapely shapes that intersect at the plane
        normal_ind = plane.size.index(0.0)
        dim = "xyz"[normal_ind]
        pos = plane.center[normal_ind]
        xyz_kwargs = {dim: pos}
        shapes_plane = self.intersections_plane(**xyz_kwargs)

        # intersect all shapes with the input plane
        bs_min, bs_max = [plane.pop_axis(bounds, axis=normal_ind)[1] for bounds in plane.bounds]
        shapely_box = box(minx=bs_min[0], miny=bs_min[0], maxx=bs_max[1], maxy=bs_max[1])
        shapely_box = plane.evaluate_inf_shape(shapely_box)
        return [plane.evaluate_inf_shape(shape) & shapely_box for shape in shapes_plane]

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
    def bounds(self) -> Bound:  # pylint:disable=too-many-locals
        """Returns bounding box min and max coordinates..

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
        if isinstance(self, Box):
            if np.count_nonzero(self.size) != 2:
                raise ValidationError(
                    "'Medium2D' requires exactly one of the 'Box' dimensions to have size zero."
                )
            return self.size.index(0)
        if isinstance(self, PolySlab):
            if self.slab_bounds[0] != self.slab_bounds[1]:
                raise ValidationError("'Medium2D' requires the 'PolySlab' bounds to be equal.")
            return self.axis
        if isinstance(self, Cylinder):
            if self.length != 0:
                raise ValidationError("'Medium2D' requires the 'Cylinder' length to be zero.")
            return self.axis
        raise ValidationError(
            "'Medium2D' is only compatible with 'Box', 'PolySlab', and 'Cylinder' geometries."
        )

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
        _shape = self.evaluate_inf_shape(shape)
        if isinstance(_shape, LineString):
            xs, ys = zip(*_shape.coords)
            ax.plot(xs, ys, color=plot_params.edgecolor)
        elif isinstance(_shape, Point):
            ax.scatter(shape.x, shape.y, color=plot_params.facecolor)
        else:
            patch = polygon_patch(_shape, **plot_params.to_kwargs())
            ax.add_artist(patch)
        return ax

    @classmethod
    def strip_coords(
        cls, shape: Shapely
    ) -> Tuple[List[float], List[float], Tuple[List[float], List[float]]]:
        """Get the exterior and list of interior xy coords for a shape.

        Parameters
        ----------
        shape: shapely.geometry.base.BaseGeometry
            The shape that you want to strip coordinates from.

        Returns
        -------
        Tuple[List[float], List[float], Tuple[List[float], List[float]]]
            List of exterior xy coordinates
            and a list of lists of the interior xy coordinates of the "holes" in the shape.
        """

        if isinstance(shape, Polygon):
            ext_coords = shape.exterior.coords[:]
            list_int_coords = [interior.coords[:] for interior in shape.interiors]
        elif isinstance(shape, MultiPolygon):
            all_ext_coords = []
            list_all_int_coords = []
            for _shape in shape.geoms:
                all_ext_coords.append(_shape.exterior.coords[:])
                all_int_coords = [_interior.coords[:] for _interior in _shape.interiors]
                list_all_int_coords.append(all_int_coords)
            ext_coords = np.concatenate(all_ext_coords, axis=0)
            list_int_coords = [
                np.concatenate(all_int_coords, axis=0)
                for all_int_coords in list_all_int_coords
                if len(all_int_coords) > 0
            ]
        return ext_coords, list_int_coords

    @classmethod
    def map_to_coords(cls, func: Callable[[float], float], shape: Shapely) -> Shapely:
        """Maps a function to each coordinate in shape.

        Parameters
        ----------
        func : Callable[[float], float]
            Takes old coordinate and returns new coordinate.
        shape: shapely.geometry.base.BaseGeometry
            The shape to map this function to.

        Returns
        -------
        shapely.geometry.base.BaseGeometry
            A new copy of the input shape with the mapping applied to the coordinates.
        """

        if not isinstance(shape, (Polygon, MultiPolygon)):
            return shape

        def apply_func(coords):
            return [(func(coord_x), func(coord_y)) for (coord_x, coord_y) in coords]

        ext_coords, list_int_coords = cls.strip_coords(shape)
        new_ext_coords = apply_func(ext_coords)
        list_new_int_coords = [apply_func(int_coords) for int_coords in list_int_coords]

        return Polygon(new_ext_coords, holes=list_new_int_coords)

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
        xmin, xmax, ymin, ymax = (self._evaluate_inf(v) for v in (xmin, xmax, ymin, ymax))

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    @staticmethod
    def _evaluate_inf(v):
        """Processes values and evaluates any infs into large (signed) numbers."""
        return np.sign(v) * LARGE_NUMBER if np.isinf(v) else v

    @classmethod
    def evaluate_inf_shape(cls, shape: Shapely) -> Shapely:
        """Returns a copy of shape with inf vertices replaced by large numbers if polygon."""

        return cls.map_to_coords(cls._evaluate_inf, shape) if isinstance(shape, Polygon) else shape

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
    def rotate_points(points: Points, axis: Coordinate, angle: float) -> Points:
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
        points: Points,
        polar_axis: Axis,
        angle_theta: float,
        angle_phi: float,
    ) -> Points:
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
        """Returns object's volume given bounds."""

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
        """Returns object's surface area given bounds."""

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
        """
        tan(sidewall_angle). _tanq*height gives the offset value
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


""" importable geometries """


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
    def surfaces(cls, size: Size, center: Coordinate, **kwargs):  # pylint: disable=too-many-locals
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
    def surfaces_with_exclusion(
        cls, size: Size, center: Coordinate, **kwargs
    ):  # pylint: disable=too-many-locals
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

    # pylint:disable=too-many-locals
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
            return [Point(minx, miny)]

        return [box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)]

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

    def _plot_arrow(  # pylint:disable=too-many-arguments, too-many-locals
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
        """Returns object's volume given bounds."""

        volume = 1

        for axis in range(3):
            min_bound = max(self.bounds[0][axis], bounds[0][axis])
            max_bound = min(self.bounds[1][axis], bounds[1][axis])

            volume *= max_bound - min_bound

        return volume

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area given bounds."""

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


class Sphere(Centered, Circular):
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
        return [Point(x0, y0).buffer(0.5 * intersect_dist, quad_segs=_N_SHAPELY_QUAD_SEGS)]

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
        """Returns object's volume given bounds."""

        volume = 4.0 / 3.0 * np.pi * self.radius**3

        # a very loose upper bound on how much of sphere is in bounds
        for axis in range(3):
            if self.center[axis] <= bounds[0][axis] or self.center[axis] >= bounds[1][axis]:
                volume *= 0.5

        return volume

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area given bounds."""

        area = 4.0 * np.pi * self.radius**2

        # a very loose upper bound on how much of sphere is in bounds
        for axis in range(3):
            if self.center[axis] <= bounds[0][axis] or self.center[axis] >= bounds[1][axis]:
                area *= 0.5

        return area


class Cylinder(Centered, Circular, Planar):
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
        return [Point(x0, y0).buffer(radius_offset, quad_segs=_N_SHAPELY_QUAD_SEGS)]

    def _intersections_side(self, position, axis):  # pylint:disable=too-many-locals
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
            Polygon(vertices_max + vertices_frustum_right + vertices_min + vertices_frustum_left)
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
        """Returns object's volume given bounds."""

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
        """Returns object's surface area given bounds."""

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
        description="Dilation of the supplied polygon by shifting each edge along its "
        "normal outwards direction by a distance; a negative value corresponds to erosion.",
        units=MICROMETER,
    )

    vertices: Vertices = pydantic.Field(
        ...,
        title="Vertices",
        description="List of (d1, d2) defining the 2 dimensional positions of the polygon "
        "face vertices at the ``reference_plane``. "
        "The index of dimension should be in the ascending order: e.g. if "
        "the slab normal axis is ``axis=y``, the coordinate of the vertices will be in (x, z)",
        units=MICROMETER,
    )

    @property
    def center_axis(self) -> float:
        """Gets the position of the center of the geometry in the out of plane dimension."""
        zmin, zmax = self.slab_bounds
        if np.isneginf(zmin) and np.isposinf(zmax):
            return 0.0
        return (zmax + zmin) / 2.0

    @property
    def length_axis(self) -> float:
        """Gets the length of the geometry along the out of plane dimension."""
        zmin, zmax = self.slab_bounds
        return zmax - zmin

    @pydantic.validator("vertices", pre=True, always=True)
    def convert_to_numpy(cls, val):
        """Pre-convert vertices to numpy one time."""
        return np.array(val)

    @pydantic.validator("slab_bounds", always=True)
    def slab_bounds_order(cls, val):
        """Maximum position of the slab should be no smaller than its minimal position."""
        if val[1] < val[0]:
            raise SetupError(
                "Polyslab.slab_bounds must be specified in the order of "
                "minimum and maximum positions of the slab along the axis. "
                f"But now the maximum {val[1]} is smaller than the minimum {val[0]}."
            )
        return val

    @pydantic.validator("vertices", always=True)
    def correct_shape(cls, val):
        """Makes sure vertices size is correct.
        Make sure no intersecting edges.
        """

        shape = val.shape

        # overall shape of vertices
        if len(shape) != 2 or shape[1] != 2:
            raise SetupError(
                "PolySlab.vertices must be a 2 dimensional array shaped (N, 2).  "
                f"Given array with shape of {shape}."
            )

        # make sure no polygon splitting, isalands, 0 area
        poly_heal = make_valid(Polygon(val))
        if poly_heal.area < fp_eps:
            raise SetupError("The polygon almost collapses to a 1D curve.")

        if not isinstance(poly_heal, Polygon) or len(poly_heal.interiors) > 0:
            raise SetupError(
                "Polygon is self-intersecting, resulting in "
                "polygon splitting or generation of holes/islands. "
                "A general treatment to self-intersecting polygon will be available "
                "in future releases."
            )
        return val

    @pydantic.validator("vertices", always=True)
    def no_complex_self_intersecting_polygon_at_reference_plane(cls, val, values):
        """At the reference plane, check if the polygon is self-intersecting.

        There are two types of self-intersection that can occur during dilation:
        1) the one that creates holes/islands, or splits polygons, or removes everything;
        2) the one that does not.

        For 1), we issue an error since it is yet to be supported;
        For 2), we heal the polygon, and warn that the polygon has been cleaned up.
        """
        # no need to valiate anything here
        if isclose(values["dilation"], 0):
            return val

        val_np = PolySlab._proper_vertices(val)
        dist = values["dilation"]

        # 0) fully eroded
        if dist < 0 and dist < -PolySlab._maximal_erosion(val_np):
            raise SetupError("Erosion value is too large. The polygon is fully eroded.")

        # no edge events
        if not PolySlab._edge_events_detection(val_np, dist, ignore_at_dist=False):
            return val

        poly_offset = PolySlab._shift_vertices(val_np, dist)[0]
        if PolySlab._area(poly_offset) < fp_eps**2:
            raise SetupError("Erosion value is too large. The polygon is fully eroded.")

        # edge events
        poly_offset = make_valid(Polygon(poly_offset))
        # 1) polygon split or create holes/islands
        if not isinstance(poly_offset, Polygon) or len(poly_offset.interiors) > 0:
            raise SetupError(
                "Dilation/Erosion value is too large, resulting in "
                "polygon splitting or generation of holes/islands. "
                "A general treatment to self-intersecting polygon will be available "
                "in future releases."
            )

        # case 2
        log.warning(
            "The dilation/erosion value is too large. resulting in a "
            "self-intersecting polygon. "
            "The vertices have been modified to make a valid polygon."
        )
        return val

    @pydantic.validator("vertices", always=True)
    def no_self_intersecting_polygon_during_extrusion(cls, val, values):
        """In this simple polyslab, we don't support self-intersecting polygons yet, meaning that
        any normal cross section of the PolySlab cannot be self-intersecting. This part checks
        if any self-interction will occur during extrusion with non-zero sidewall angle.

        There are two types of self-intersection, known as edge events,
        that can occur during dilation:
        1) neighboring vertex-vertex crossing. This type of edge event can be treated with
        ``ComplexPolySlab`` which divides the polyslab into a list of simple polyslabs.

        2) other types of edge events that can create holes/islands or split polygons.
        To detect this, we sample _N_SAMPLE_POLYGON_INTERSECT cross sections to see if any creation
        of polygons/holes, and changes in vertices number.
        """
        if "sidewall_angle" not in values:
            raise ValidationError("``sidewall_angle`` failed validation.")

        # no need to valiate anything here
        if isclose(values["sidewall_angle"], 0):
            return val

        # apply dilation
        poly_ref = PolySlab._proper_vertices(val)
        if not isclose(values["dilation"], 0):
            poly_ref = PolySlab._shift_vertices(poly_ref, values["dilation"])[0]
            poly_ref = PolySlab._heal_polygon(poly_ref)

        # Fist, check vertex-vertex crossing at any point during extrusion
        length = values["slab_bounds"][1] - values["slab_bounds"][0]
        dist = [-length * np.tan(values["sidewall_angle"])]
        # reverse the dilation value if it's defined on the top
        if values["reference_plane"] == "top":
            dist = [-dist[0]]
        # for middle, both direction needs to be examined
        elif values["reference_plane"] == "middle":
            dist = [dist[0] / 2, -dist[0] / 2]

        # capture vertex crossing events
        max_thick = []
        for dist_val in dist:
            max_dist = PolySlab._neighbor_vertices_crossing_detection(poly_ref, dist_val)

            if max_dist is not None:
                max_thick.append(max_dist / abs(dist_val) * length)

        if len(max_thick) > 0:
            max_thick = min(max_thick)
            raise SetupError(
                "Sidewall angle or structure thickness is so large that the polygon "
                "is self-intersecting during extrusion. "
                f"Please either reduce structure thickness to be < {max_thick:.3e}, "
                "or use our plugin 'ComplexPolySlab' to divide the complex polyslab "
                "into a list of simple polyslabs."
            )

        # vertex-edge crossing event.
        for dist_val in dist:
            if PolySlab._edge_events_detection(poly_ref, dist_val):
                raise SetupError(
                    "Sidewall angle or structure thickness is too large, "
                    "resulting in polygon splitting or generation of holes/islands. "
                    "A general treatment to self-intersecting polygon will be available "
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
        reference_plane: PlanePosition = "middle",
    ) -> List[PolySlab]:
        """Import :class:`PolySlab` from a ``gdstk.Cell`` or a ``gdspy.Cell``.

        Parameters
        ----------
        gds_cell : Union[gdstk.Cell, gdspy.Cell]
            ``gdstk.Cell`` or ``gdspy.Cell`` containing 2D geometric data.
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
        reference_plane : PlanePosition = "middle"
            The position of the GDS layer. It can be at the ``bottom``, ``middle``,
            or ``top`` of the PolySlab. E.g. if ``axis=1``, ``bottom`` refers to the
            negative side of y-axis, and ``top`` refers to the positive side of y-axis.

        Returns
        -------
        List[:class:`PolySlab`]
            List of :class:`PolySlab` objects sharing ``axis`` and  slab bound properties.
        """

        all_vertices = cls._load_gds_vertices(gds_cell, gds_layer, gds_dtype, gds_scale)

        return [
            cls(
                vertices=verts,
                axis=axis,
                slab_bounds=slab_bounds,
                dilation=dilation,
                sidewall_angle=sidewall_angle,
                reference_plane=reference_plane,
            )
            for verts in all_vertices
        ]

    @classmethod
    def _load_gds_vertices(
        cls,
        gds_cell,
        gds_layer: int,
        gds_dtype: int = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
    ) -> List[Vertices]:
        """Import :class:`PolySlab` from a ``gdstk.Cell`` or a ``gdspy.Cell``.

        Parameters
        ----------
        gds_cell : Union[gdstk.Cell, gdspy.Cell]
            ``gdstk.Cell`` or ``gdspy.Cell`` containing 2D geometric data.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``.
            If ``None``, imports all data for this layer into the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of MICROMETER.
            For example, if gds file uses nanometers, set ``gds_scale=1e-3``.
            Must be positive.

        Returns
        -------
        List[Vertices]
            List of :class:`.Vertices`
        """

        # switch the GDS cell loader function based on the class name string
        # TODO: make this more robust in future releases
        gds_cell_class_name = str(gds_cell.__class__)

        if "gdstk" in gds_cell_class_name:
            gds_loader_fn = cls._load_gds_vertices_gdstk

        elif "gdspy" in gds_cell_class_name:
            gds_loader_fn = cls._load_gds_vertices_gdspy

        else:
            raise ValueError(
                f"argumeent 'gds_cell' of type '{gds_cell_class_name}' "
                "does not seem to be associated with 'gdstk' or 'gdspy' packages "
                "and therefore can't be loaded by Tidy3D."
            )

        all_vertices = gds_loader_fn(
            gds_cell=gds_cell, gds_layer=gds_layer, gds_dtype=gds_dtype, gds_scale=gds_scale
        )

        # convert vertices into polyslabs
        polygons = [Polygon(vertices).buffer(0) for vertices in all_vertices]
        polys_union = unary_union(polygons, grid_size=POLY_GRID_SIZE)

        if isinstance(polys_union, Polygon):
            all_vertices = [PolySlab.strip_coords(polys_union)[0]]
        elif isinstance(polys_union, MultiPolygon):
            all_vertices = [PolySlab.strip_coords(polygon)[0] for polygon in polys_union.geoms]
        return all_vertices

    @staticmethod
    def _load_gds_vertices_gdstk(
        gds_cell,
        gds_layer: int,
        gds_dtype: int = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
    ) -> List[Vertices]:
        """Load :class:`PolySlab` vertices from a ``gdstk.Cell``.

        Parameters
        ----------
        gds_cell : gdstk.Cell
            ``gdstk.Cell`` or ``gdspy.Cell`` containing 2D geometric data.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``.
            If ``None``, imports all data for this layer into the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of MICROMETER.
            For example, if gds file uses nanometers, set ``gds_scale=1e-3``.
            Must be positive.

        Returns
        -------
        List[Vertices]
            List of :class:`.Vertices`
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

    @classmethod
    def _load_gds_vertices_gdspy(  # pylint:disable=too-many-arguments, too-many-locals
        cls,
        gds_cell,
        gds_layer: int,
        gds_dtype: int = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
    ) -> List["PolySlab"]:
        """Load :class:`PolySlab` vertices from a ``gdspy.Cell``.

        Parameters
        ----------
        gds_cell :  gdspy.Cell
            ``gdspy.Cell`` containing 2D geometric data.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``.
            If ``None``, imports all data for this layer into the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of MICROMETER.
            For example, if gds file uses nanometers, set ``gds_scale=1e-3``.
            Must be positive.

        Returns
        -------
        List[Vertices]
            List of :class:`.Vertices`
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

        # apply scaling and convert vertices into polyslabs
        all_vertices = [vertices * gds_scale for vertices in all_vertices]
        all_vertices = [vertices.tolist() for vertices in all_vertices]

        return all_vertices

    @cached_property
    def reference_polygon(self) -> np.ndarray:
        """The polygon at the reference plane.

        Returns
        -------
        ArrayLike[float, float]
            The vertices of the polygon at the reference plane.
        """
        vertices = self._proper_vertices(self.vertices)
        if isclose(self.dilation, 0):
            return vertices
        offset_vertices = self._shift_vertices(vertices, self.dilation)[0]
        return self._heal_polygon(offset_vertices)

    @cached_property
    def middle_polygon(self) -> np.ndarray:
        """The polygon at the middle.

        Returns
        -------
        ArrayLike[float, float]
            The vertices of the polygon at the middle.
        """

        dist = self._extrusion_length_to_offset_distance(self.length_axis / 2)
        if self.reference_plane == "bottom":
            return self._shift_vertices(self.reference_polygon, dist)[0]
        if self.reference_plane == "top":
            return self._shift_vertices(self.reference_polygon, -dist)[0]
        # middle case
        return self.reference_polygon

    @cached_property
    def base_polygon(self) -> np.ndarray:
        """The polygon at the base, derived from the ``middle_polygon``.

        Returns
        -------
        ArrayLike[float, float]
            The vertices of the polygon at the base.
        """
        if self.reference_plane == "bottom":
            return self.reference_polygon
        dist = self._extrusion_length_to_offset_distance(-self.length_axis / 2)
        return self._shift_vertices(self.middle_polygon, dist)[0]

    @cached_property
    def top_polygon(self) -> np.ndarray:
        """The polygon at the top, derived from the ``middle_polygon``.

        Returns
        -------
        ArrayLike[float, float]
            The vertices of the polygon at the top.
        """
        if self.reference_plane == "top":
            return self.reference_polygon
        dist = self._extrusion_length_to_offset_distance(self.length_axis / 2)
        return self._shift_vertices(self.middle_polygon, dist)[0]

    # pylint:disable=too-many-locals
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

        z, (x, y) = self.pop_axis((x, y, z), axis=self.axis)

        z0 = self.center_axis
        dist_z = np.abs(z - z0)
        inside_height = dist_z <= (self.length_axis / 2)

        # avoid going into face checking if no points are inside slab bounds
        if not np.any(inside_height):
            return inside_height

        # check what points are inside polygon cross section (face)
        z_local = z - z0  # distance to the middle
        dist = -z_local * self._tanq

        # # Leaving this function and commented out lines using it below in case we want to revert
        # # to it at some point, e.g. if we introduce a MATPLOTLIB_INSTALLED flag.
        # def contains_pointwise(face_polygon):
        #     def fun_contain(xy_point):
        #         point = Point(xy_point)
        #         return face_polygon.covers(point)
        #     return fun_contain

        if isinstance(x, np.ndarray):
            inside_polygon = np.zeros_like(inside_height)
            xs_slab = x[inside_height]
            ys_slab = y[inside_height]

            # vertical sidewall
            if isclose(self.sidewall_angle, 0):
                # face_polygon = Polygon(self.reference_polygon)
                # fun_contain = contains_pointwise(face_polygon)
                # contains_vectorized = np.vectorize(fun_contain, signature="(n)->()")
                poly_path = path.Path(self.reference_polygon)
                contains_vectorized = poly_path.contains_points
                points_stacked = np.stack((xs_slab, ys_slab), axis=1)
                inside_polygon_slab = contains_vectorized(points_stacked)
                inside_polygon[inside_height] = inside_polygon_slab
            # slanted sidewall, offsetting vertices at each z
            else:
                # a helper function for moving axis
                def _move_axis(arr):
                    return np.moveaxis(arr, source=self.axis, destination=-1)

                def _move_axis_reverse(arr):
                    return np.moveaxis(arr, source=-1, destination=self.axis)

                inside_polygon_axis = _move_axis(inside_polygon)
                x_axis = _move_axis(x)
                y_axis = _move_axis(y)

                for z_i in range(z.shape[self.axis]):
                    if not _move_axis(inside_height)[0, 0, z_i]:
                        continue
                    vertices_z = self._shift_vertices(
                        self.middle_polygon, _move_axis(dist)[0, 0, z_i]
                    )[0]
                    # face_polygon = Polygon(vertices_z)
                    # fun_contain = contains_pointwise(face_polygon)
                    # contains_vectorized = np.vectorize(fun_contain, signature="(n)->()")
                    poly_path = path.Path(vertices_z)
                    contains_vectorized = poly_path.contains_points
                    points_stacked = np.stack(
                        (x_axis[:, :, 0].flatten(), y_axis[:, :, 0].flatten()), axis=1
                    )
                    inside_polygon_slab = contains_vectorized(points_stacked)
                    inside_polygon_axis[:, :, z_i] = inside_polygon_slab.reshape(x_axis.shape[:2])
                inside_polygon = _move_axis_reverse(inside_polygon_axis)
        else:
            vertices_z = self._shift_vertices(self.middle_polygon, dist)[0]
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
        z0 = self.center_axis
        z_local = z - z0  # distance to the middle
        dist = -z_local * self._tanq
        vertices_z = self._shift_vertices(self.middle_polygon, dist)[0]
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
        z0 = self.center_axis
        z_base = z0 - self.length_axis / 2

        axis_ordered = self._order_axis(axis)
        height_list = self._find_intersecting_height(position, axis_ordered)
        polys = []

        # looping through z_i to assemble the polygons
        height_list = np.append(height_list, self.length_axis)
        h_base = 0.0
        for h_top in height_list:
            # length within between top and bottom
            h_length = h_top - h_base

            # coordinate of each subsection
            z_min, z_max = z_base + h_base, z_base + h_top

            # vertices for the base of each subsection
            # move up by `fp_eps` in case vertices are degenerate at the base.
            dist = -(h_base - self.length_axis / 2 + fp_eps) * self._tanq
            vertices = self._shift_vertices(self.middle_polygon, dist)[0]

            # for vertical sidewall, no need for complications
            if isclose(self.sidewall_angle, 0):
                ints_y, ints_angle = self._find_intersecting_ys_angle_vertical(
                    vertices, position, axis_ordered
                )
            else:
                ints_y, ints_angle = self._find_intersecting_ys_angle_slant(
                    vertices, position, axis_ordered
                )

            # make polygon with intersections and z axis information
            for y_index in range(len(ints_y) // 2):
                y_min = ints_y[2 * y_index]
                y_max = ints_y[2 * y_index + 1]
                minx, miny = self._order_by_axis(plane_val=y_min, axis_val=z_min, axis=axis)
                maxx, maxy = self._order_by_axis(plane_val=y_max, axis_val=z_max, axis=axis)

                if isclose(self.sidewall_angle, 0):
                    polys.append(box(minx=minx, miny=miny, maxx=maxx, maxy=maxy))
                else:
                    angle_min = ints_angle[2 * y_index]
                    angle_max = ints_angle[2 * y_index + 1]

                    angle_min = np.arctan(np.tan(self.sidewall_angle) / np.sin(angle_min))
                    angle_max = np.arctan(np.tan(self.sidewall_angle) / np.sin(angle_max))

                    dy_min = h_length * np.tan(angle_min)
                    dy_max = h_length * np.tan(angle_max)

                    x1, y1 = self._order_by_axis(plane_val=y_min, axis_val=z_min, axis=axis)
                    x2, y2 = self._order_by_axis(plane_val=y_max, axis_val=z_min, axis=axis)
                    x3, y3 = self._order_by_axis(
                        plane_val=y_max - dy_max, axis_val=z_max, axis=axis
                    )
                    x4, y4 = self._order_by_axis(
                        plane_val=y_min + dy_min, axis_val=z_max, axis=axis
                    )
                    vertices = ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
                    polys.append(Polygon(vertices).buffer(0))
            # update the base coordinate for the next subsection
            h_base = h_top

        # merge touching polygons
        polys_union = unary_union(polys, grid_size=POLY_GRID_SIZE)
        if isinstance(polys_union, Polygon):
            return [polys_union]
        if isinstance(polys_union, MultiPolygon):
            return polys_union.geoms
        # in other cases, just return the original unmerged polygons
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
        if isclose(self.sidewall_angle, 0):
            return np.array([])

        # shift rate
        dist = 1.0
        shift_x, shift_y = PolySlab._shift_vertices(self.middle_polygon, dist)[2]
        shift_val = shift_x if axis == 0 else shift_y
        shift_val[np.isclose(shift_val, 0, rtol=_IS_CLOSE_RTOL)] = np.inf  # for static vertices

        # distance to the plane in the direction of vertex shifting
        distance = self.middle_polygon[:, axis] - position
        height = distance / self._tanq / shift_val + self.length_axis / 2
        height = np.unique(height)
        # further filter very close ones
        is_not_too_close = np.insert((np.diff(height) > fp_eps), 0, True)
        height = height[is_not_too_close]

        height = height[height > fp_eps]
        height = height[height < self.length_axis - fp_eps]
        return height

    def _find_intersecting_ys_angle_vertical(  # pylint:disable=too-many-locals
        self, vertices: np.ndarray, position: float, axis: int, exclude_on_vertices: bool = False
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
        exclude_on_vertices : bool = False
            Whehter to exclude those intersecting directly with the vertices.

        Returns
        -------
        Union[np.ndarray, np.ndarray]
            List of intersection points along y direction.
            List of angles between plane and edges.
        """

        vertices_axis = vertices.copy()
        # flip vertices x,y for axis = y
        if axis == 1:
            vertices_axis = np.roll(vertices_axis, shift=1, axis=1)

        # get the forward vertices
        vertices_f = np.roll(vertices_axis, shift=-1, axis=0)

        # x coordinate of the two sets of vertices
        x_vertices_f = vertices_f[:, 0]
        x_vertices_axis = vertices_axis[:, 0]

        # find which segments intersect
        f_left_to_intersect = x_vertices_f <= position
        orig_right_to_intersect = x_vertices_axis > position
        intersects_b = np.logical_and(f_left_to_intersect, orig_right_to_intersect)

        f_right_to_intersect = x_vertices_f > position
        orig_left_to_intersect = x_vertices_axis <= position
        intersects_f = np.logical_and(f_right_to_intersect, orig_left_to_intersect)

        # exclude vertices at the position if exclude_on_vertices is True
        if exclude_on_vertices:
            intersects_on = np.isclose(x_vertices_axis, position, rtol=_IS_CLOSE_RTOL)
            intersects_f_on = np.isclose(x_vertices_f, position, rtol=_IS_CLOSE_RTOL)
            intersects_both_off = np.logical_not(np.logical_or(intersects_on, intersects_f_on))
            intersects_f &= intersects_both_off
            intersects_b &= intersects_both_off
        intersects_segment = np.logical_or(intersects_b, intersects_f)

        iverts_b = vertices_axis[intersects_segment]
        iverts_f = vertices_f[intersects_segment]

        # intersecting positions and angles
        ints_y = []
        ints_angle = []
        for vertices_f_local, vertices_b_local in zip(iverts_b, iverts_f):
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
        # flip vertices x,y for axis = y
        if axis == 1:
            vertices_axis = np.roll(vertices_axis, shift=1, axis=1)

        # get the forward vertices
        vertices_f = np.roll(vertices_axis, shift=-1, axis=0)
        # get the backward vertices
        vertices_b = np.roll(vertices_axis, shift=1, axis=0)

        ## First part, plane intersects with edges, same as vertical
        ints_y, ints_angle = self._find_intersecting_ys_angle_vertical(
            vertices, position, axis, exclude_on_vertices=True
        )
        ints_y = ints_y.tolist()
        ints_angle = ints_angle.tolist()

        ## Second part, plane intersects directly with vertices
        # vertices on the intersection
        intersects_on = np.isclose(vertices_axis[:, 0], position, rtol=_IS_CLOSE_RTOL)
        iverts_on = vertices_axis[intersects_on]
        # position of the neighbouring vertices
        iverts_b = vertices_b[intersects_on]
        iverts_f = vertices_f[intersects_on]
        # shift rate
        dist = -np.sign(self.sidewall_angle)
        shift_x, shift_y = self._shift_vertices(self.middle_polygon, dist)[2]
        shift_val = shift_x if axis == 0 else shift_y
        shift_val = shift_val[intersects_on]

        for vertices_f_local, vertices_b_local, vertices_on_local, shift_local in zip(
            iverts_f, iverts_b, iverts_on, shift_val
        ):
            x_on, y_on = vertices_on_local
            x_f, y_f = vertices_f_local
            x_b, y_b = vertices_b_local

            num_added = 0  # keep track the number of added vertices
            slope = []  # list of slopes for added vertices
            # case 1, shifting velocity is 0
            if np.isclose(shift_local, 0, rtol=_IS_CLOSE_RTOL):
                ints_y.append(y_on)
                # Slope w.r.t. forward and backward should equal,
                # just pick one of them.
                slope.append((y_on - y_b) / (x_on - x_b))
                ints_angle.append(np.pi / 2 - np.arctan(np.abs(slope[0])))
                continue

            # case 2, shifting towards backward direction
            if (x_b - position) * shift_local < 0:
                ints_y.append(y_on)
                slope.append((y_on - y_b) / (x_on - x_b))
                num_added += 1

            # case 3, shifting towards forward direction
            if (x_f - position) * shift_local < 0:
                ints_y.append(y_on)
                slope.append((y_on - y_f) / (x_on - x_f))
                num_added += 1

            # in case 2, and case 3, if just num_added = 1
            if num_added == 1:
                ints_angle.append(np.pi / 2 - np.arctan(np.abs(slope[0])))
            # if num_added = 2, the order of the two new vertices needs to handled correctly;
            # it should be sorted according to the -slope * moving direction
            elif num_added == 2:
                dressed_slope = [-s_i * shift_local for s_i in slope]
                sort_index = np.argsort(np.array(dressed_slope))
                sorted_slope = np.array(slope)[sort_index]

                ints_angle.append(np.pi / 2 - np.arctan(np.abs(sorted_slope[0])))
                ints_angle.append(np.pi / 2 - np.arctan(np.abs(sorted_slope[1])))

        ints_y = np.array(ints_y)
        ints_angle = np.array(ints_angle)

        sort_index = np.argsort(ints_y)
        ints_y_sort = ints_y[sort_index]
        ints_angle_sort = ints_angle[sort_index]

        return ints_y_sort, ints_angle_sort

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates. The dilation and slant angle are not
        taken into account exactly for speed. Instead, the polygon may be slightly smaller than
        the returned bounds, but it should always be fully contained.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """

        # check for the maximum possible contribution from dilation/slant on each side
        max_offset = self.dilation
        if self.reference_plane == "bottom":
            max_offset += max(0, -self._tanq * self.length_axis)
        elif self.reference_plane == "top":
            max_offset += max(0, self._tanq * self.length_axis)
        elif self.reference_plane == "middle":
            max_offset += max(0, abs(self._tanq) * self.length_axis / 2)

        # special care when dilated
        if max_offset > 0:
            dilated_vertices = self._shift_vertices(
                self._proper_vertices(self.vertices), max_offset
            )[0]
            xmin, ymin = np.amin(dilated_vertices, axis=0)
            xmax, ymax = np.amax(dilated_vertices, axis=0)
        else:
            # otherwise, bounds are directly based on the supplied vertices
            xmin, ymin = np.amin(self.vertices, axis=0)
            xmax, ymax = np.amax(self.vertices, axis=0)

        # get bounds in (local) z
        zmin, zmax = self.slab_bounds

        # rearrange axes
        coords_min = self.unpop_axis(zmin, (xmin, ymin), axis=self.axis)
        coords_max = self.unpop_axis(zmax, (xmax, ymax), axis=self.axis)
        return (tuple(coords_min), tuple(coords_max))

    def _extrusion_length_to_offset_distance(self, extrusion: float) -> float:
        """Convert extrusion length to offset distance."""
        if isclose(self.sidewall_angle, 0):
            return 0
        return -extrusion * self._tanq

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
    def _perimeter(vertices: np.ndarray) -> float:
        """Compute the polygon perimeter.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        float
            Polygon perimeter.
        """
        vert_shift = np.roll(vertices.copy(), axis=0, shift=-1)
        dx = vertices[:, 0] - vert_shift[:, 0]
        dy = vertices[:, 1] - vert_shift[:, 1]

        return np.sum(np.sqrt(dx**2 + dy**2))

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
        return vertices if PolySlab._area(vertices) > 0 else vertices[::-1, :]

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
        return vertices[~np.isclose(vertices_diff, 0, rtol=_IS_CLOSE_RTOL)]

    @staticmethod
    def _proper_vertices(vertices: Vertices) -> np.ndarray:
        """convert vertices to np.array format,
        removing duplicate neighbouring vertices,
        and oriented in CCW direction.

        Returns
        -------
        ArrayLike[float, float]
           The vertices of the polygon for internal use.
        """

        vertices_np = PolySlab.vertices_to_array(vertices)
        return PolySlab._orient(PolySlab._remove_duplicate_vertices(vertices_np))

    @staticmethod
    def _edge_events_detection(  # pylint:disable=too-many-return-statements
        proper_vertices: np.ndarray, dilation: float, ignore_at_dist: bool = True
    ) -> bool:
        """Detect any edge events within the offset distance ``dilation``.
        If ``ignore_at_dist=True``, the edge event at ``dist`` is ignored.
        """

        # ignore the event that occurs right at the offset distance
        if ignore_at_dist:
            dilation -= fp_eps * dilation / abs(dilation)
        # number of vertices before offsetting
        num_vertices = proper_vertices.shape[0]

        # 0) fully eroded?
        if dilation < 0 and dilation < -PolySlab._maximal_erosion(proper_vertices):
            return True

        # sample at a few dilation values
        dist_list = dilation * np.linspace(0, 1, 1 + _N_SAMPLE_CURVE_SHAPELY)[1:]
        for dist in dist_list:
            # offset: we offset the vertices first, and then use shapely to make it proper
            # in principle, one can offset with shapely.buffer directly, but shapely somehow
            # automatically removes some vertices even though no change of topology.
            poly_offset = PolySlab._shift_vertices(proper_vertices, dist)[0]
            # flipped winding number
            if PolySlab._area(poly_offset) < fp_eps**2:
                return True

            poly_offset = make_valid(Polygon(poly_offset))
            # 1) polygon split or create holes/islands
            if not isinstance(poly_offset, Polygon) or len(poly_offset.interiors) > 0:
                return True

            # 2) reduction in vertex number
            offset_vertices = PolySlab._proper_vertices(list(poly_offset.exterior.coords))
            if offset_vertices.shape[0] != num_vertices:
                return True

            # 3) some splitted polygon might fully disappear after the offset, but they
            # can be detected if we offset back.
            poly_offset_back = make_valid(
                Polygon(PolySlab._shift_vertices(offset_vertices, -dist)[0])
            )
            if isinstance(poly_offset_back, MultiPolygon) or len(poly_offset_back.interiors) > 0:
                return True
            offset_back_vertices = list(poly_offset_back.exterior.coords)
            if PolySlab._proper_vertices(offset_back_vertices).shape[0] != num_vertices:
                return True

        return False

    @staticmethod
    def _neighbor_vertices_crossing_detection(
        vertices: np.ndarray, dist: float, ignore_at_dist: bool = True
    ) -> float:
        """Detect if neighboring vertices will cross after a dilation distance dist.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        dist : float
            Distance to offset.
        ignore_at_dist : bool, optional
            whether to ignore the event right at ``dist`.

        Returns
        -------
        float
            the absolute value of the maximal allowed dilation
            if there are any crossing, otherwise return ``None``.
        """
        # ignore the event that occurs right at the offset distance
        if ignore_at_dist:
            dist -= fp_eps * dist / abs(dist)

        edge_length, edge_reduction = PolySlab._edge_length_and_reduction_rate(vertices)
        length_remaining = edge_length - edge_reduction * dist

        if np.any(length_remaining < 0):
            index_oversized = length_remaining < 0
            max_dist = np.min(
                np.abs(edge_length[index_oversized] / edge_reduction[index_oversized])
            )
            return max_dist
        return None

    @staticmethod
    def array_to_vertices(arr_vertices: np.ndarray) -> Vertices:
        """Converts a numpy array of vertices to a list of tuples."""
        return list(arr_vertices)

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
        np.ndarray
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

        if isclose(dist, 0):
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
            np.isclose(det, 0, rtol=_IS_CLOSE_RTOL),
            0.0,
            cross(asm, rot90(asm - asp)) / (det + np.isclose(det, 0, rtol=_IS_CLOSE_RTOL)),
        )
        parallel_shift = dist * tan_half_angle

        shift_total = perpendicular_shift * rot90(asm) + parallel_shift * asm
        shift_x = shift_total[0, :]
        shift_y = shift_total[1, :]

        return np.swapaxes(vs_orig + shift_total, -2, -1), parallel_shift, (shift_x, shift_y)

    @staticmethod
    def _edge_length_and_reduction_rate(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Edge length of reduction rate of each edge with unit offset length.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        Tuple[np.ndarray, np.narray]
            edge length, and reduction rate
        """

        # edge length
        vs_orig = vertices.T.copy()
        vs_next = np.roll(vs_orig.copy(), axis=-1, shift=-1)
        edge_length = np.linalg.norm(vs_next - vs_orig, axis=0)

        # edge length remaining
        dist = 1
        parallel_shift = PolySlab._shift_vertices(vertices, dist)[1]
        parallel_shift_p = np.roll(parallel_shift.copy(), shift=-1)
        edge_reduction = -(parallel_shift + parallel_shift_p)
        return edge_length, edge_reduction

    @staticmethod
    def _maximal_erosion(vertices: np.ndarray) -> float:
        """The erosion value that reduces the length of
        all edges to be non-positive.
        """
        edge_length, edge_reduction = PolySlab._edge_length_and_reduction_rate(vertices)
        ind_nonzero = abs(edge_reduction) > fp_eps
        return -np.min(edge_length[ind_nonzero] / edge_reduction[ind_nonzero])

    @staticmethod
    def _heal_polygon(vertices: np.ndarray) -> np.ndarray:
        """heal a self-intersecting polygon."""
        shapely_poly = Polygon(vertices)
        if shapely_poly.is_valid:
            return vertices
        # perform healing
        poly_heal = make_valid(shapely_poly)
        return PolySlab._proper_vertices(list(poly_heal.exterior.coords))

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume given bounds."""

        z_min, z_max = self.slab_bounds

        z_min = max(z_min, bounds[0][self.axis])
        z_max = min(z_max, bounds[1][self.axis])

        length = z_max - z_min

        top_area = abs(self._area(self.top_polygon))
        base_area = abs(self._area(self.base_polygon))

        # https://mathworld.wolfram.com/PyramidalFrustum.html
        return 1.0 / 3.0 * length * (top_area + base_area + np.sqrt(top_area * base_area))

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area given bounds."""

        area = 0

        top = self.top_polygon
        base = self.base_polygon

        top_area = abs(self._area(top))
        base_area = abs(self._area(base))

        top_perim = self._perimeter(top)
        base_perim = self._perimeter(top)

        z_min, z_max = self.slab_bounds

        if z_min < bounds[0][self.axis]:
            z_min = bounds[0][self.axis]
        else:
            area += base_area

        if z_max > bounds[1][self.axis]:
            z_max = bounds[1][self.axis]
        else:
            area += top_area

        length = z_max - z_min

        area += 0.5 * (top_perim + base_perim) * length

        return area


class TriangleMesh(Geometry, ABC):
    """Custom surface geometry given by a triangle mesh, as in the STL file format.

    Example
    -------
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    >>> stl_geom = TriangleMesh.from_vertices_faces(vertices, faces)
    """

    mesh_dataset: Optional[TriangleMeshDataset] = pydantic.Field(
        ...,
        title="Surface mesh data",
        description="Surface mesh data.",
    )

    @classmethod
    def _check_trimesh_library(cls):
        """Check if the trimesh package is imported."""
        if not TRIMESH_AVAILABLE:
            raise ImportError(
                "The package 'trimesh' was not found. Please install the 'trimesh' "
                "dependencies to use 'TriangleMesh'. For example: "
                "pip install -r requirements/trimesh.txt."
            )

    @pydantic.root_validator(pre=True)
    def _validate_trimesh_library(cls, values):
        """Check if the trimesh package is imported as a validator."""
        cls._check_trimesh_library()
        return values

    @pydantic.validator("mesh_dataset", pre=True, always=True)
    def _warn_if_none(cls, val: TriangleMeshDataset) -> TriangleMeshDataset:
        """Warn if the Dataset fails to load."""
        if isinstance(val, dict):
            if any((v in DATA_ARRAY_MAP for _, v in val.items() if isinstance(v, str))):
                log.warning("Loading 'mesh_dataset' without data.")
                return None
        return val

    @pydantic.validator("mesh_dataset", always=True)
    def _check_mesh(cls, val: TriangleMeshDataset) -> TriangleMeshDataset:
        """Check that the mesh is valid."""
        if val is None:
            return None
        mesh = cls._triangles_to_trimesh(val.surface_mesh)
        if not all(np.array(mesh.area_faces) > fp_eps):
            raise ValidationError(
                "The provided mesh has triangles with zero area. "
                "Consider using 'trimesh.Trimesh.remove_degenerate_faces' to fix this."
            )
        if not mesh.is_winding_consistent:
            log.warning(
                "The provided mesh does not have consistent winding (face orientations). "
                "This can lead to incorrect permittivity distributions. You can use a "
                "'PermittivityMonitor' to check if the permittivity distribution is correct. "
                "You could repair the mesh manually, or try 'trimesh.repair.fix_winding'."
            )
        if not mesh.is_watertight:
            log.warning(
                "The provided mesh is not watertight and may require manual repair. "
                "Non-watertight meshes can generate incorrect permittivity distributions. "
                "They also cause problems with plotting. You can use a "
                "'PermittivityMonitor' to check if the permittivity distribution is correct. "
                "You can see which faces are broken using 'trimesh.repair.broken_faces'."
            )

        return val

    @classmethod
    def from_stl(
        cls,
        filename: str,
        scale: float = 1.0,
        origin: Tuple[float, float, float] = (0, 0, 0),
        solid_index: int = None,
        **kwargs,
    ) -> Union[TriangleMesh, GeometryGroup]:
        """Load a :class:`.TriangleMesh` directly from an STL file.
        The ``solid_index`` parameter can be used to select a single solid from the file.
        Otherwise, if the file contains a single solid, it will be loaded as a
        :class:`.TriangleMesh`; if the file contains multiple solids,
        they will all be loaded as a :class:`.GeometryGroup`.

        Parameters
        ----------
        filename : str
            The name of the STL file containing the surface geometry mesh data.
        scale : float = 1.0
            The length scale for the loaded geometry (um).
            For example, a scale of 10.0 means that a vertex (1, 0, 0) will be placed at
            x = 10 um.
        origin : Tuple[float, float, float] = (0, 0, 0)
            The origin of the loaded geometry, in units of ``scale``.
            Translates from (0, 0, 0) to this point after applying the scaling.
        solid_index : int = None
            If set, read a single solid with this index from the file.

        Returns
        -------
        Union[:class:`.TriangleMesh`, :class:`.GeometryGroup`]
            The geometry or geometry group from the file.
        """

        def process_single(mesh: trimesh.Trimesh) -> TriangleMesh:
            """Process a single 'trimesh.Trimesh' using scale and origin."""
            mesh.apply_scale(scale)
            mesh.apply_translation(origin)
            return cls.from_trimesh(mesh)

        cls._check_trimesh_library()

        scene = trimesh.load(filename, **kwargs)
        meshes = []
        if isinstance(scene, trimesh.Trimesh):
            meshes = [scene]
        elif isinstance(scene.trimesh.Scene):
            meshes = scene.dump()
        else:
            raise ValidationError(
                "Invalid trimesh type in file. Supported types are 'trimesh.Trimesh' "
                "and 'trimesh.Scene'."
            )

        if solid_index is None:
            if isinstance(scene, trimesh.Trimesh):
                return process_single(scene)
            if isinstance(scene, trimesh.Scene):
                geoms = [process_single(mesh) for mesh in meshes]
                return GeometryGroup(geometries=geoms)

        if solid_index < len(meshes):
            return process_single(meshes[solid_index])
        raise ValidationError("No solid found at 'solid_index' in the stl file.")

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh) -> TriangleMesh:
        """Create a :class:`.TriangleMesh` from a ``trimesh.Trimesh`` object.

        Parameters
        ----------
        trimesh : ``trimesh.Trimesh``
            The Trimesh object containing the surface geometry mesh data.

        Returns
        -------
        :class:`.TriangleMesh`
            The custom surface mesh geometry given by the ``trimesh.Trimesh`` provided.
        """
        return cls.from_vertices_faces(mesh.vertices, mesh.faces)

    @classmethod
    def from_triangles(cls, triangles: np.ndarray) -> TriangleMesh:
        """Create a :class:`.TriangleMesh` from a numpy array
        containing the triangles of a surface mesh.

        Parameters
        ----------
        triangles : ``np.ndarray``
            A numpy array of shape (N, 3, 3) storing the triangles of the surface mesh.
            The first index labels the triangle, the second index labels the vertex
            within a given triangle, and the third index is the coordinate (x, y, or z).

        Returns
        -------
        :class:`.TriangleMesh`
            The custom surface mesh geometry given by the triangles provided.

        """
        triangles = np.array(triangles)
        if len(triangles.shape) != 3 or triangles.shape[1] != 3 or triangles.shape[2] != 3:
            raise ValidationError(
                f"Provided 'triangles' must be an N x 3 x 3 array, given {triangles.shape}."
            )
        num_faces = len(triangles)
        coords = dict(
            face_index=np.arange(num_faces),
            vertex_index=np.arange(3),
            axis=np.arange(3),
        )
        vertices = TriangleMeshDataArray(triangles, coords=coords)
        mesh_dataset = TriangleMeshDataset(surface_mesh=vertices)
        return TriangleMesh(mesh_dataset=mesh_dataset)

    @classmethod
    def from_vertices_faces(cls, vertices: np.ndarray, faces: np.ndarray) -> TriangleMesh:
        """Create a :class:`.TriangleMesh` from numpy arrays containing the data
        of a surface mesh. The first array contains the vertices, and the second array contains
        faces formed from triples of the vertices.

        Parameters
        ----------
        vertices: ``np.ndarray``
            A numpy array of shape (N, 3) storing the vertices of the surface mesh.
            The first index labels the vertex, and the second index is the coordinate
            (x, y, or z).
        faces : ``np.ndarray``
            A numpy array of shape (M, 3) storing the indices of the vertices of each face
            in the surface mesh. The first index labels the face, and the second index
            labels the vertex index within the ``vertices`` array.

        Returns
        -------
        :class:`.TriangleMesh`
            The custom surface mesh geometry given by the vertices and faces provided.

        """

        cls._check_trimesh_library()

        vertices = np.array(vertices)
        faces = np.array(faces)
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValidationError(
                f"Provided 'vertices' must be an N x 3 array, given {vertices.shape}."
            )
        if len(faces.shape) != 2 or faces.shape[1] != 3:
            raise ValidationError(f"Provided 'faces' must be an M x 3 array, given {faces.shape}.")
        return cls.from_triangles(trimesh.Trimesh(vertices, faces).triangles)

    @classmethod
    def _triangles_to_trimesh(cls, triangles: np.ndarray) -> trimesh.Trimesh:
        """Convert an (N, 3, 3) numpy array of triangles to a ``trimesh.Trimesh``."""

        cls._check_trimesh_library()
        return trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles))

    @cached_property
    def trimesh(self) -> trimesh.Trimesh:
        """A ``trimesh.Trimesh`` object representing the custom surface mesh geometry."""
        return self._triangles_to_trimesh(self.triangles)

    @cached_property
    def triangles(self) -> np.ndarray:
        """The triangles of the surface mesh as an ``np.ndarray``."""
        if self.mesh_dataset is None:
            raise DataError("Can't get triangles as 'mesh_dataset' is None.")
        return self.mesh_dataset.surface_mesh.to_numpy()

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area given bounds."""
        # currently ignores bounds
        return self.trimesh.area

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume given bounds."""
        # currently ignores bounds
        return self.trimesh.volume

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        if self.mesh_dataset is None:
            return ((-inf, -inf, -inf), (inf, inf, inf))
        return self.trimesh.bounds

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

        if self.mesh_dataset is None:
            return []

        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)

        origin = self.unpop_axis(position, (0, 0), axis=axis)
        normal = self.unpop_axis(1, (0, 0), axis=axis)

        mesh = self.trimesh

        if not NETWORKX_RTREE_AVAILABLE:
            raise ImportError(
                "'TriangleMesh.intersections_plane' requires 'networkx' and 'rtree'. "
                "Please install the 'trimesh' dependencies. For example: "
                "pip install -r requirements/trimesh.txt."
            )

        section = mesh.section(plane_origin=origin, plane_normal=normal)

        if section is None:
            return []

        # homogeneous transformation matrix to map to xy plane
        mapping = np.eye(4)

        # translate to origin
        mapping[3, :3] = -np.array(origin)

        # permute so normal is aligned with z axis
        # and (y, z), (x, z), resp. (x, y) are aligned with (x, y)
        identity = np.eye(3)
        permutation = self.unpop_axis(identity[2], identity[0:2], axis=axis)
        mapping[:3, :3] = np.array(permutation).T

        section2d, _ = section.to_planar(to_2D=mapping)
        return section2d.polygons_full

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

        arrays = tuple(map(np.array, (x, y, z)))
        self._ensure_equal_shape(*arrays)
        inside = np.zeros((arrays[0].size,), dtype=bool)
        arrays_flat = map(np.ravel, arrays)
        arrays_stacked = np.stack(tuple(arrays_flat), axis=-1)
        inside = self.trimesh.contains(arrays_stacked)
        return inside.reshape(arrays[0].shape)

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

        log.warning(
            "Plotting a 'TriangleMesh' may give inconsistent results "
            "if the mesh is not unionized. We recommend unionizing all meshes before import. "
            "A 'PermittivityMonitor' can be used to check that the mesh is loaded correctly."
        )

        return Geometry.plot(self, x=x, y=y, z=z, ax=ax, **patch_kwargs)


# types of geometry including just one Geometry object (exluding group)
SingleGeometryType = Union[Box, Sphere, Cylinder, PolySlab, TriangleMesh]


class GeometryGroup(Geometry):
    """A collection of Geometry objects that can be called as a single geometry object."""

    geometries: Tuple[annotate_type(SingleGeometryType), ...] = pydantic.Field(
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
        rmins = (bound[0] for bound in bounds)
        rmaxs = (bound[1] for bound in bounds)

        rmin = functools.reduce(
            lambda x, y: (min(x[0], y[0]), min(x[1], y[1]), min(x[2], y[2])), rmins
        )
        rmax = functools.reduce(
            lambda x, y: (max(x[0], y[0]), max(x[1], y[1]), max(x[2], y[2])), rmaxs
        )

        return rmin, rmax

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
        all_intersections = (
            geometry.intersections_plane(x=x, y=y, z=z) for geometry in self.geometries
        )

        return functools.reduce(lambda a, b: a + b, all_intersections)

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
        """Returns object's volume given bounds."""

        individual_volumes = (geometry.volume(bounds) for geometry in self.geometries)

        return np.sum(individual_volumes)

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area given bounds."""

        individual_areas = (geometry.surface_area(bounds) for geometry in self.geometries)

        return np.sum(individual_areas)


# geometries usable to define a structure
GeometryType = Union[SingleGeometryType, GeometryGroup]
