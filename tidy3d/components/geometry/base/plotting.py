"""Geometry plotting, coordinate helpers, and measure methods."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import shapely

from tidy3d.components.autograd import get_static
from tidy3d.components.transformation import RotationAroundAxis
from tidy3d.components.viz import (
    PLOT_BUFFER,
    add_ax_if_none,
    equal_aspect,
    polygon_patch,
    set_default_labels_and_title,
)
from tidy3d.constants import LARGE_NUMBER
from tidy3d.exceptions import SetupError

from .constants import GDS_MAX_COORDINATE_INDEX

if TYPE_CHECKING:
    from numpy.typing import ArrayFloat3D, ArrayLike, NDArray

    from tidy3d.components.types import (
        Ax,
        Axis,
        Bound,
        Coordinate,
        Coordinate2D,
        LengthUnit,
        Shapely,
    )
    from tidy3d.components.viz import PlotParams, VisualizationSpec


@equal_aspect
@add_ax_if_none
def plot(
    self,  # pyrefly: ignore[implicit-any-parameter]
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    ax: Ax = None,
    plot_length_units: LengthUnit = None,
    viz_spec: VisualizationSpec = None,
    **patch_kwargs: Any,
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
    plot_length_units : LengthUnit = None
        Specify units to use for axis labels, tick labels, and the title.
    viz_spec : VisualizationSpec = None
        Plotting parameters associated with a medium to use instead of defaults.
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
    axis, _position = self.parse_xyz_kwargs(x=x, y=y, z=z)
    shapes_intersect = self.intersections_plane(x=x, y=y, z=z, section_tolerance_2d=True)

    plot_params = self.plot_params
    if viz_spec is not None:
        plot_params = plot_params.override_with_viz_spec(viz_spec)
    plot_params = plot_params.include_kwargs(**patch_kwargs)

    # for each intersection, plot the shape
    for shape in shapes_intersect:
        ax = self.plot_shape(shape, plot_params=plot_params, ax=ax)

    # clean up the axis display
    ax = self.add_ax_lims(axis=axis, ax=ax)
    ax.set_aspect("equal")
    # Add the default axis labels, tick labels, and title
    from .box import Box

    ax = Box.add_ax_labels_and_title(ax=ax, x=x, y=y, z=z, plot_length_units=plot_length_units)
    return ax


def plot_shape(
    self,  # pyrefly: ignore[implicit-any-parameter]
    shape: Shapely,
    plot_params: PlotParams,
    ax: Ax,
) -> Ax:
    """Defines how a shape is plotted on a matplotlib axes."""
    if shape.geom_type in (
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
        "GeometryCollection",
    ):
        for sub_shape in shape.geoms:
            ax = self.plot_shape(shape=sub_shape, plot_params=plot_params, ax=ax)

        return ax

    _shape = evaluate_inf_shape(shape)

    if _shape.geom_type == "LineString":
        xs, ys = zip(*_shape.coords)
        ax.plot(xs, ys, color=plot_params.facecolor, linewidth=plot_params.linewidth)
    elif _shape.geom_type == "Point":
        ax.scatter(shape.x, shape.y, color=plot_params.facecolor)
    else:
        patch = polygon_patch(_shape, **plot_params.to_kwargs())
        ax.add_artist(patch)
    return ax


def _do_not_intersect(bounds_a: float, bounds_b: float, shape_a: Shapely, shape_b: Shapely) -> bool:
    """Check whether two shapes intersect."""

    # do a bounding box check to see if any intersection to do anything about
    if (
        bounds_a[0] > bounds_b[2]
        or bounds_b[0] > bounds_a[2]
        or bounds_a[1] > bounds_b[3]
        or bounds_b[1] > bounds_a[3]
    ):
        return True

    # look more closely to see if intersected.
    if shape_b.is_empty or not shape_a.intersects(shape_b):
        return True

    return False


def _get_plot_labels(axis: Axis) -> tuple[str, str]:
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
    _, (xlabel, ylabel) = pop_axis("xyz", axis=axis)
    return xlabel, ylabel


def _get_plot_limits(
    self,  # pyrefly: ignore[implicit-any-parameter]
    axis: Axis,
    buffer: float = PLOT_BUFFER,
) -> tuple[Coordinate2D, Coordinate2D]:
    """Gets planar coordinate limits for cross section plots.

    Parameters
    ----------
    axis : int
        Integer index into 'xyz' (0,1,2).
    buffer : float = 0.3
        Amount of space to add around the limits on the + and - sides.

    Returns
    -------
        tuple[float, float], tuple[float, float]
    The x and y plot limits, packed as ``(xmin, xmax), (ymin, ymax)``.
    """
    _, ((xmin, ymin), (xmax, ymax)) = self._pop_bounds(axis=axis)
    return (xmin - buffer, xmax + buffer), (ymin - buffer, ymax + buffer)


def add_ax_lims(
    self,  # pyrefly: ignore[implicit-any-parameter]
    axis: Axis,
    ax: Ax,
    buffer: float = PLOT_BUFFER,
) -> Ax:
    """Sets the x,y limits based on ``self.bounds``.

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
    (xmin, xmax), (ymin, ymax) = self._get_plot_limits(axis=axis, buffer=buffer)

    # note: axes limits dont like inf values, so we need to evaluate them first if present
    xmin, xmax, ymin, ymax = self._evaluate_inf((xmin, xmax, ymin, ymax))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return ax


def add_ax_labels_and_title(
    ax: Ax,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    plot_length_units: LengthUnit = None,
) -> Ax:
    """Sets the axis labels, tick labels, and title based on ``axis``
    and an optional ``plot_length_units`` argument.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes
        Matplotlib axes to add labels and limits on.
    x : float = None
        Position of plane in x direction, only one of x,y,z can be specified to define plane.
    y : float = None
        Position of plane in y direction, only one of x,y,z can be specified to define plane.
    z : float = None
        Position of plane in z direction, only one of x,y,z can be specified to define plane.
    plot_length_units : LengthUnit = None
        When set to a supported ``LengthUnit``, plots will be produced with annotated axes
        and title with the proper units.

    Returns
    -------
    matplotlib.axes._subplots.Axes
        The supplied matplotlib axes.
    """
    from .box import Box

    axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
    axis_labels = Box._get_plot_labels(axis)
    ax = set_default_labels_and_title(
        axis_labels=axis_labels,
        axis=axis,
        position=position,
        ax=ax,
        plot_length_units=plot_length_units,
    )
    return ax


def _evaluate_inf(array: ArrayLike) -> NDArray[np.floating]:
    """Processes values and evaluates any infs into large (signed) numbers."""
    array = get_static(np.array(array))
    return np.where(np.isinf(array), np.sign(array) * LARGE_NUMBER, array)


def evaluate_inf_shape(shape: Shapely) -> Shapely:
    """Returns a copy of shape with inf vertices replaced by large numbers if polygon."""
    if not any(np.isinf(b) for b in shape.bounds):
        return shape
    return shapely.transform(shape, _evaluate_inf, include_z=None)


def pop_axis(coord: tuple[Any, Any, Any], axis: int) -> tuple[Any, tuple[Any, Any]]:
    """Separates coordinate at ``axis`` index from coordinates on the plane tangent to ``axis``.

    Parameters
    ----------
    coord : tuple[Any, Any, Any]
        Tuple of three values in original coordinate system.
    axis : int
        Integer index into 'xyz' (0,1,2).

    Returns
    -------
    Any, tuple[Any, Any]
        The input coordinates are separated into the one along the axis provided
        and the two on the planar coordinates,
        like ``axis_coord, (planar_coord1, planar_coord2)``.
    """
    plane_vals = list(coord)
    axis_val = plane_vals.pop(axis)
    return axis_val, tuple(plane_vals)


def unpop_axis(ax_coord: Any, plane_coords: tuple[Any, Any], axis: int) -> tuple[Any, Any, Any]:
    """Combine coordinate along axis with coordinates on the plane tangent to the axis.

    Parameters
    ----------
    ax_coord : Any
        Value along axis direction.
    plane_coords : tuple[Any, Any]
        Values along ordered planar directions.
    axis : int
        Integer index into 'xyz' (0,1,2).

    Returns
    -------
    tuple[Any, Any, Any]
        The three values in the xyz coordinate system.
    """
    coords = list(plane_coords)
    coords.insert(axis, ax_coord)
    return tuple(coords)


def parse_xyz_kwargs(**xyz: Any) -> tuple[Axis, float]:
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
    if len(xyz_filtered) != 1:
        raise ValueError("exactly one kwarg in [x,y,z] must be specified.")
    axis_label, position = list(xyz_filtered.items())[0]
    axis = "xyz".index(axis_label)
    return axis, position


def _validate_gds_precision(
    *,
    polygons: list[Any],
    gds_precision: float,
    context: str,
) -> float:
    """Validate that the requested GDS precision is safe for the written polygons."""
    if not np.isfinite(gds_precision) or gds_precision <= 0:
        raise SetupError(
            f"Requested 'gds_precision={gds_precision:.6g} um' in {context} must be "
            "positive and finite."
        )

    if not polygons:
        return gds_precision

    max_abs_coord = 0.0
    for polygon in polygons:
        bbox = polygon.bounding_box()
        if bbox is None:
            continue
        for point in bbox:
            for value in point:
                coordinate = float(value)
                if not np.isfinite(coordinate):
                    raise SetupError(
                        f"Cannot export non-finite GDS coordinate '{coordinate}' in "
                        f"{context}. Use finite geometry bounds before exporting to GDS."
                    )
                max_abs_coord = max(max_abs_coord, abs(coordinate))

    if max_abs_coord <= 0:
        return gds_precision

    min_safe_precision = float(np.nextafter(max_abs_coord / GDS_MAX_COORDINATE_INDEX, np.inf))
    if gds_precision >= min_safe_precision:
        return gds_precision

    raise SetupError(
        f"Requested 'gds_precision={gds_precision:.6g} um' in {context} is too fine for "
        f"the export bounds (+/-{max_abs_coord:.6g} um). The minimum safe precision is "
        f"'{min_safe_precision:.6g} um' to stay within the signed 32-bit GDS coordinate "
        "range. Use a larger 'gds_precision'."
    )


def parse_two_xyz_kwargs(**xyz: Any) -> list[tuple[Axis, float]]:
    """Turns x,y,z kwargs into indices of axes and the position along each axis.

    Parameters
    ----------
    x : float = None
        Position in x direction, only two of x,y,z can be specified to define line.
    y : float = None
        Position in y direction, only two of x,y,z can be specified to define line.
    z : float = None
        Position in z direction, only two of x,y,z can be specified to define line.

    Returns
    -------
    [(int, float), (int, float)]
        Index into xyz axis (0,1,2) and position along that axis.
    """
    xyz_filtered = {k: v for k, v in xyz.items() if v is not None}
    assert len(xyz_filtered) == 2, "exactly two kwarg in [x,y,z] must be specified."
    xyz_list = list(xyz_filtered.items())
    return [("xyz".index(axis_label), position) for axis_label, position in xyz_list]


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
    self,  # pyrefly: ignore[implicit-any-parameter]
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


def volume(self, bounds: Bound = None) -> float:  # pyrefly: ignore[implicit-any-parameter]
    """Returns object's volume with optional bounds.

    Parameters
    ----------
    bounds : tuple[tuple[float, float, float], tuple[float, float, float]] = None
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
def _volume(self, bounds: Bound) -> float:  # pyrefly: ignore[implicit-any-parameter]
    """Returns object's volume within given bounds."""


def surface_area(self, bounds: Bound = None) -> float:  # pyrefly: ignore[implicit-any-parameter]
    """Returns object's surface area with optional bounds.

    Parameters
    ----------
    bounds : tuple[tuple[float, float, float], tuple[float, float, float]] = None
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
def _surface_area(self, bounds: Bound) -> float:  # pyrefly: ignore[implicit-any-parameter]
    """Returns object's surface area within given bounds."""
