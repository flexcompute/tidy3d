"""Utilities for geometry manipulation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from math import isclose
from typing import Any, Optional, Union

import numpy as np
import pydantic.v1 as pydantic
import shapely
from shapely.geometry import (
    Polygon,
)
from shapely.geometry.base import (
    BaseMultipartGeometry,
)

from tidy3d.components.autograd.utils import get_static
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.grid import Grid
from tidy3d.components.types import (
    ArrayFloat2D,
    Axis,
    Bound,
    Coordinate,
    Direction,
    MatrixReal4x4,
    PlanePosition,
    Shapely,
)
from tidy3d.constants import fp_eps
from tidy3d.exceptions import SetupError, Tidy3dError

from . import base, mesh, polyslab, primitives

GeometryType = Union[
    base.Box,
    base.Transformed,
    base.ClipOperation,
    base.GeometryGroup,
    primitives.Sphere,
    primitives.Cylinder,
    polyslab.PolySlab,
    polyslab.ComplexPolySlabBase,
    mesh.TriangleMesh,
]


def flatten_shapely_geometries(
    geoms: Union[Shapely, Iterable[Shapely]], keep_types: tuple[type, ...] = (Polygon,)
) -> list[Shapely]:
    """
    Flatten nested geometries into a flat list, while only keeping the specified types.

    Recursively extracts and returns non-empty geometries of the given types from input geometries,
    expanding any GeometryCollections or Multi* types.

    Parameters
    ----------
    geoms : Union[Shapely, Iterable[Shapely]]
        Input geometries to flatten.

    keep_types : tuple[type, ...]
        Geometry types to keep (e.g., (Polygon, LineString)). Default is
        (Polygon).

    Returns
    -------
    list[Shapely]
        Flat list of non-empty geometries matching the specified types.
    """
    # Handle single Shapely object by wrapping it in a list
    if isinstance(geoms, Shapely):
        geoms = [geoms]

    flat = []
    for geom in geoms:
        if geom.is_empty:
            continue
        if isinstance(geom, keep_types):
            flat.append(geom)
        elif isinstance(geom, BaseMultipartGeometry):
            flat.extend(flatten_shapely_geometries(geom.geoms, keep_types))
    return flat


def merging_geometries_on_plane(
    geometries: list[GeometryType],
    plane: Box,
    property_list: list[Any],
    interior_disjoint_geometries: bool = False,
    cleanup: bool = True,
    quad_segs: Optional[int] = None,
) -> list[tuple[Any, Shapely]]:
    """Compute list of shapes on plane. Overlaps are removed or merged depending on
    provided property_list.

    Parameters
    ----------
    geometries : List[GeometryType]
        List of structures to filter on the plane.
    plane : Box
        Plane specification.
    property_list : List = None
        Property value for each structure.
    interior_disjoint_geometries: bool = False
        If ``True``, geometries of different properties on the plane must not be overlapping.
    cleanup : bool = True
        If True, removes extremely small features from each polygon's boundary.
    quad_segs : Optional[int] = None
        Number of segments used to discretize circular shapes. If ``None``, uses
        high-quality visualization settings.

    Returns
    -------
    List[Tuple[Any, Shapely]]
        List of shapes and their property value on the plane after merging.
    """

    if len(geometries) != len(property_list):
        raise SetupError(
            "Number of provided property values is not equal to the number of geometries."
        )

    shapes = []
    for geo, prop in zip(geometries, property_list):
        # get list of Shapely shapes that intersect at the plane
        shapes_plane = plane.intersections_with(geo, cleanup=cleanup, quad_segs=quad_segs)

        # Append each of them and their property information to the list of shapes
        for shape in shapes_plane:
            shapes.append((prop, shape, shape.bounds))

    if interior_disjoint_geometries:
        # No need to consider overlapping. We simply group shapes by property, and union_all
        # shapes of the same property.
        shapes_by_prop = defaultdict(list)
        for prop, shape, _ in shapes:
            shapes_by_prop[prop].append(shape)
        # union shapes of same property
        results = []
        for prop, shapes in shapes_by_prop.items():
            unionized = shapely.union_all(shapes).buffer(0).normalize()
            if not unionized.is_empty:
                results.append((prop, unionized))
        return results

    background_shapes = []
    for prop, shape, bounds in shapes:
        minx, miny, maxx, maxy = bounds

        # loop through background_shapes (note: all background are non-intersecting or merged)
        for index, (_prop, _shape, _bounds) in enumerate(background_shapes):
            _minx, _miny, _maxx, _maxy = _bounds

            # do a bounding box check to see if any intersection to do anything about
            if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                continue

            # look more closely to see if intersected.
            if shape.disjoint(_shape):
                continue

            # different prop, remove intersection from background shape
            if prop != _prop:
                diff_shape = (_shape - shape).buffer(0).normalize()
                # mark background shape for removal if nothing left
                if diff_shape.is_empty or len(diff_shape.bounds) == 0:
                    background_shapes[index] = None
                background_shapes[index] = (_prop, diff_shape, diff_shape.bounds)
            # same prop, unionize shapes and mark background shape for removal
            else:
                shape = (shape | _shape).buffer(0).normalize()
                background_shapes[index] = None

        # after doing this with all background shapes, add this shape to the background
        background_shapes.append((prop, shape, shape.bounds))

        # remove any existing background shapes that have been marked as 'None'
        background_shapes = [b for b in background_shapes if b is not None]

    # filter out any remaining None or empty shapes (shapes with area completely removed)
    return [(prop, shape) for (prop, shape, _) in background_shapes if shape]


def flatten_groups(
    *geometries: GeometryType,
    flatten_nonunion_type: bool = False,
    flatten_transformed: bool = False,
    transform: Optional[MatrixReal4x4] = None,
) -> GeometryType:
    """Iterates over all geometries, flattening groups and unions.

    Parameters
    ----------
    *geometries : GeometryType
        Geometries to flatten.
    flatten_nonunion_type : bool = False
        If ``False``, only flatten geometry unions (and ``GeometryGroup``). If ``True``, flatten
        all clip operations.
    flatten_transformed : bool = False
        If ``True``, ``Transformed`` groups are flattened into individual transformed geometries.
    transform : Optional[MatrixReal4x4]
        Accumulated transform from parents. Only used when ``flatten_transformed`` is ``True``.

    Yields
    ------
    GeometryType
        Geometries after flattening groups and unions.
    """
    for geometry in geometries:
        if isinstance(geometry, base.GeometryGroup):
            yield from flatten_groups(
                *geometry.geometries,
                flatten_nonunion_type=flatten_nonunion_type,
                flatten_transformed=flatten_transformed,
                transform=transform,
            )
        elif isinstance(geometry, base.ClipOperation) and (
            flatten_nonunion_type or geometry.operation == "union"
        ):
            yield from flatten_groups(
                geometry.geometry_a,
                geometry.geometry_b,
                flatten_nonunion_type=flatten_nonunion_type,
                flatten_transformed=flatten_transformed,
                transform=transform,
            )
        elif flatten_transformed and isinstance(geometry, base.Transformed):
            new_transform = geometry.transform
            if transform is not None:
                new_transform = np.matmul(transform, new_transform)
            yield from flatten_groups(
                geometry.geometry,
                flatten_nonunion_type=flatten_nonunion_type,
                flatten_transformed=flatten_transformed,
                transform=new_transform,
            )
        elif flatten_transformed and transform is not None:
            yield base.Transformed(geometry=geometry, transform=transform)
        else:
            yield geometry


def traverse_geometries(geometry: GeometryType) -> GeometryType:
    """Iterator over all geometries within the given geometry.

    Iterates over groups and clip operations within the given geometry, yielding each one.

    Parameters
    ----------
    geometry: GeometryType
        Base geometry to start iteration.

    Returns
    -------
    :class:`Geometry`
        Geometries within the base geometry.
    """
    if isinstance(geometry, base.GeometryGroup):
        for g in geometry.geometries:
            yield from traverse_geometries(g)
    elif isinstance(geometry, base.ClipOperation):
        yield from traverse_geometries(geometry.geometry_a)
        yield from traverse_geometries(geometry.geometry_b)
    yield geometry


def from_shapely(
    shape: Shapely,
    axis: Axis,
    slab_bounds: tuple[float, float],
    dilation: float = 0.0,
    sidewall_angle: float = 0,
    reference_plane: PlanePosition = "middle",
) -> base.Geometry:
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
    if shape.geom_type == "LinearRing":
        if sidewall_angle == 0:
            return polyslab.PolySlab(
                vertices=shape.coords[:-1],
                axis=axis,
                slab_bounds=slab_bounds,
                dilation=dilation,
                reference_plane=reference_plane,
            )
        group = polyslab.ComplexPolySlabBase(
            vertices=shape.coords[:-1],
            axis=axis,
            slab_bounds=slab_bounds,
            dilation=dilation,
            sidewall_angle=sidewall_angle,
            reference_plane=reference_plane,
        ).geometry_group
        return group.geometries[0] if len(group.geometries) == 1 else group

    if shape.geom_type == "Polygon":
        exterior = from_shapely(
            shape.exterior, axis, slab_bounds, dilation, sidewall_angle, reference_plane
        )
        interior = [
            from_shapely(hole, axis, slab_bounds, -dilation, -sidewall_angle, reference_plane)
            for hole in shape.interiors
        ]
        if len(interior) == 0:
            return exterior
        interior = interior[0] if len(interior) == 1 else base.GeometryGroup(geometries=interior)
        return base.ClipOperation(operation="difference", geometry_a=exterior, geometry_b=interior)

    if shape.geom_type in {"MultiPolygon", "GeometryCollection"}:
        return base.GeometryGroup(
            geometries=[
                from_shapely(geo, axis, slab_bounds, dilation, sidewall_angle, reference_plane)
                for geo in shape.geoms
            ]
        )

    raise Tidy3dError(f"Shape {shape} cannot be converted to Geometry.")


def vertices_from_shapely(shape: Shapely) -> ArrayFloat2D:
    """Iterate over the polygons of a shapely geometry returning the vertices.

    Parameters
    ----------
    shape : shapely.geometry.base.BaseGeometry
        Shapely primitive to have its vertices extracted. It must be a linear ring, a polygon or a
        collection of any of those.

    Returns
    -------
    List[Tuple[ArrayFloat2D]]
        List of tuples ``(exterior, *interiors)``.
    """
    if shape.geom_type == "LinearRing":
        return [(shape.coords[:-1],)]
    if shape.geom_type == "Polygon":
        return [(shape.exterior.coords[:-1], *tuple(hole.coords[:-1] for hole in shape.interiors))]
    if shape.geom_type in {"MultiPolygon", "GeometryCollection"}:
        return sum(vertices_from_shapely(geo) for geo in shape.geoms)

    raise Tidy3dError(f"Shape {shape} cannot be converted to Geometry.")


def validate_no_transformed_polyslabs(
    geometry: GeometryType, transform: MatrixReal4x4 = None
) -> None:
    """Prevents the creation of slanted polyslabs rotated out of plane."""
    if transform is None:
        transform = np.eye(4)
    if isinstance(geometry, polyslab.PolySlab):
        # sidewall_angle may be autograd-traced; unbox for the check only
        if not (
            isclose(get_static(geometry.sidewall_angle), 0)
            or base.Transformed.preserves_axis(transform, geometry.axis)
        ):
            raise Tidy3dError(
                "Slanted PolySlabs are not allowed to be rotated out of the slab plane."
            )
    elif isinstance(geometry, base.Transformed):
        transform = np.dot(transform, geometry.transform)
        validate_no_transformed_polyslabs(geometry.geometry, transform)
    elif isinstance(geometry, base.GeometryGroup):
        for geo in geometry.geometries:
            validate_no_transformed_polyslabs(geo, transform)
    elif isinstance(geometry, base.ClipOperation):
        validate_no_transformed_polyslabs(geometry.geometry_a, transform)
        validate_no_transformed_polyslabs(geometry.geometry_b, transform)


class SnapLocation(Enum):
    """Describes different methods for defining the snapping locations."""

    Boundary = 1
    """
    Choose the boundaries of Yee cells.
    """
    Center = 2
    """
    Choose the center of Yee cells.
    """


class SnapBehavior(Enum):
    """Describes different methods for snapping intervals, which are defined by two endpoints."""

    Closest = 1
    """
    Snaps the interval's endpoints to the closest grid point.
    """
    Expand = 2
    """
    Snaps the interval's endpoints to the closest grid points,
    while guaranteeing that the snapping location will never move endpoints inwards.
    """
    Contract = 3
    """
    Snaps the interval's endpoints to the closest grid points,
    while guaranteeing that the snapping location will never move endpoints outwards.
    """
    StrictExpand = 4
    """
    Same as Expand, but will always move endpoints outwards, even if already coincident with grid.
    """
    StrictContract = 5
    """
    Same as Contract, but will always move endpoints inwards, even if already coincident with grid.
    """
    Off = 6
    """
    Do not use snapping.
    """


class SnappingSpec(Tidy3dBaseModel):
    """Specifies how to apply grid snapping along each dimension."""

    location: tuple[SnapLocation, SnapLocation, SnapLocation] = pydantic.Field(
        ...,
        title="Location",
        description="Describes which positions in the grid will be considered for snapping.",
    )

    behavior: tuple[SnapBehavior, SnapBehavior, SnapBehavior] = pydantic.Field(
        ...,
        title="Behavior",
        description="Describes how snapping positions will be chosen.",
    )

    margin: Optional[
        tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt, pydantic.NonNegativeInt]
    ] = pydantic.Field(
        (0, 0, 0),
        title="Margin",
        description="Number of additional grid points to consider when expanding or contracting "
        "during snapping. Only applies when ``SnapBehavior`` is ``Expand`` or ``Contract``.",
    )


def get_closest_value(test: float, coords: np.ArrayLike, upper_bound_idx: int) -> float:
    """Helper to choose the closest value in an array to a given test value,
    using the index of the upper bound. The ``upper_bound_idx`` corresponds to the first value in
    the ``coords`` array which is greater than or equal to the test value.
    """
    # Handle corner cases first
    if upper_bound_idx == 0:
        return coords[upper_bound_idx]
    if upper_bound_idx == len(coords):
        return coords[upper_bound_idx - 1]
    # General case
    lower_bound = coords[upper_bound_idx - 1]
    upper_bound = coords[upper_bound_idx]
    dlower = abs(test - lower_bound)
    dupper = abs(test - upper_bound)
    return lower_bound if dlower < dupper else upper_bound


def snap_box_to_grid(grid: Grid, box: Box, snap_spec: SnappingSpec, rtol: float = fp_eps) -> Box:
    """Snaps a :class:`.Box` to the grid, so that the boundaries of the box are aligned with grid centers or boundaries.
    The way in which each dimension of the `box` is snapped to the grid is controlled by ``snap_spec``.
    """

    def _clamp_index(idx: int, length: int) -> int:
        return max(0, min(idx, length - 1))

    def get_lower_bound(
        test: float,
        coords: np.ArrayLike,
        upper_bound_idx: int,
        rel_tol: float,
        strict_bounds: bool,
        margin: int = 0,
    ) -> float:
        """Choose the lower bound from coords for snapping a test value downward.

        Returns a coordinate value from ``coords`` that satisfies ``result <= test`` when
        ``strict_bounds=False``, or ``result < test`` when ``strict_bounds=True``. The
        ``rel_tol`` parameter is used to determine floating-point equality: if ``test`` is
        close to a grid point (within ``rel_tol``).

        Parameters
        ----------
        test : float
            The value to snap.
        coords : np.ArrayLike
            Sorted array of coordinate values to snap to.
        upper_bound_idx : int
            Index from ``np.searchsorted(coords, test, side="left")`` - the first index where
            ``coords[upper_bound_idx] >= test``.
        rel_tol : float
            Relative tolerance for floating-point equality comparison.
        strict_bounds : bool
            If ``False``: Return value satisfies ``result <= test`` (using ``rel_tol`` for equality).
            If ``True``: Return value satisfies ``result < test`` (using ``rel_tol`` for equality).
        margin : int, optional
            Additional offset in grid cells (can be negative). Applied after determining the
            snap index. Default is 0.

        Returns
        -------
        float
            The selected coordinate value from ``coords``.
        """
        snap_idx = upper_bound_idx - 1
        if (
            not strict_bounds
            and upper_bound_idx != len(coords)
            and isclose(coords[upper_bound_idx], test, rel_tol=rel_tol)
        ):
            snap_idx = upper_bound_idx
        elif (
            strict_bounds
            and upper_bound_idx >= 2
            and isclose(coords[upper_bound_idx - 1], test, rel_tol=rel_tol)
        ):
            snap_idx = upper_bound_idx - 2

        # Apply margin and clamp
        snap_idx += margin
        snap_idx = _clamp_index(snap_idx, len(coords))
        return coords[snap_idx]

    def get_upper_bound(
        test: float,
        coords: np.ArrayLike,
        upper_bound_idx: int,
        rel_tol: float,
        strict_bounds: bool,
        margin: int = 0,
    ) -> float:
        """Choose the upper bound from coords for snapping a test value upward.

        Returns a coordinate value from ``coords`` that satisfies ``result >= test`` when
        ``strict_bounds=False``, or ``result > test`` when ``strict_bounds=True``. The
        ``rel_tol`` parameter is used to determine floating-point equality: if ``test`` is
        close to a grid point (within ``rel_tol``).

        Parameters
        ----------
        test : float
            The value to snap.
        coords : np.ArrayLike
            Sorted array of coordinate values to snap to.
        upper_bound_idx : int
            Index from ``np.searchsorted(coords, test, side="left")`` - the first index where
            ``coords[upper_bound_idx] >= test``.
        rel_tol : float
            Relative tolerance for floating-point equality comparison.
        strict_bounds : bool
            If ``False``: Return value satisfies ``result >= test`` (using ``rel_tol`` for equality).
            If ``True``: Return value satisfies ``result > test`` (using ``rel_tol`` for equality).
        margin : int, optional
            Additional offset in grid cells (can be negative). Applied after determining the
            snap index. Default is 0.

        Returns
        -------
        float
            The selected coordinate value from ``coords``.
        """
        snap_idx = upper_bound_idx

        if (
            not strict_bounds
            and upper_bound_idx > 0
            and (isclose(coords[upper_bound_idx - 1], test, rel_tol=rel_tol))
        ):
            snap_idx = upper_bound_idx - 1
        elif (
            strict_bounds
            and upper_bound_idx < len(coords)
            and isclose(coords[upper_bound_idx], test, rel_tol=rel_tol)
        ):
            snap_idx = upper_bound_idx + 1

        # Apply margin and clamp
        snap_idx += margin
        snap_idx = _clamp_index(snap_idx, len(coords))
        return coords[snap_idx]

    def find_snapping_locations(
        interval_min: float,
        interval_max: float,
        coords: np.ndarray,
        snap_type: SnapBehavior,
        snap_margin: pydantic.NonNegativeInt,
    ) -> tuple[float, float]:
        """Helper that snaps a supplied interval [interval_min, interval_max] to a
        sorted array representing coordinate values.
        """
        # Locate the interval that includes the min and max
        min_upper_bound_idx = np.searchsorted(coords, interval_min, side="left")
        max_upper_bound_idx = np.searchsorted(coords, interval_max, side="left")
        strict_bounds = (
            snap_type == SnapBehavior.StrictExpand or snap_type == SnapBehavior.StrictContract
        )
        if snap_type == SnapBehavior.Closest:
            min_snap = get_closest_value(interval_min, coords, min_upper_bound_idx)
            max_snap = get_closest_value(interval_max, coords, max_upper_bound_idx)
        elif snap_type == SnapBehavior.Expand or snap_type == SnapBehavior.StrictExpand:
            min_snap = get_lower_bound(
                interval_min,
                coords,
                min_upper_bound_idx,
                rel_tol=rtol,
                strict_bounds=strict_bounds,
                margin=-snap_margin,
            )
            max_snap = get_upper_bound(
                interval_max,
                coords,
                max_upper_bound_idx,
                rel_tol=rtol,
                strict_bounds=strict_bounds,
                margin=+snap_margin,
            )
        else:  # SnapType.Contract
            min_snap = get_upper_bound(
                interval_min,
                coords,
                min_upper_bound_idx,
                rel_tol=rtol,
                strict_bounds=strict_bounds,
                margin=+snap_margin,
            )
            max_snap = get_lower_bound(
                interval_max,
                coords,
                max_upper_bound_idx,
                rel_tol=rtol,
                strict_bounds=strict_bounds,
                margin=-snap_margin,
            )
            if max_snap < min_snap:
                raise SetupError("The supplied 'snap_margin' is too large for this contraction.")
        return (min_snap, max_snap)

    # Iterate over each axis and apply the specified snapping behavior.
    min_b, max_b = (list(f) for f in box.bounds)
    grid_bounds = grid.boundaries.to_list
    grid_centers = grid.centers.to_list
    for axis in range(3):
        snap_location = snap_spec.location[axis]
        snap_type = snap_spec.behavior[axis]
        snap_margin = snap_spec.margin[axis]
        if snap_type == SnapBehavior.Off:
            continue
        if snap_location == SnapLocation.Boundary:
            snap_coords = np.array(grid_bounds[axis])
        elif snap_location == SnapLocation.Center:
            snap_coords = np.array(grid_centers[axis])

        box_min = min_b[axis]
        box_max = max_b[axis]

        (new_min, new_max) = find_snapping_locations(
            box_min, box_max, snap_coords, snap_type, snap_margin
        )
        min_b[axis] = new_min
        max_b[axis] = new_max
    return Box.from_bounds(min_b, max_b)


def snap_point_to_grid(
    grid: Grid, point: Coordinate, snap_location: tuple[SnapLocation, SnapLocation, SnapLocation]
) -> Coordinate:
    """Snaps a :class:`.Coordinate` to the grid, so that it is coincident with grid centers or boundaries.
    The way in which each dimension of the ``point`` is snapped to the grid is controlled by ``snap_location``.
    """
    grid_bounds = grid.boundaries.to_list
    grid_centers = grid.centers.to_list
    snapped_point = 3 * [0]
    for axis in range(3):
        if snap_location[axis] == SnapLocation.Boundary:
            snap_coords = np.array(grid_bounds[axis])
        elif snap_location[axis] == SnapLocation.Center:
            snap_coords = np.array(grid_centers[axis])

        # Locate the interval that includes the test point
        min_upper_bound_idx = np.searchsorted(snap_coords, point[axis], side="left")
        snapped_point[axis] = get_closest_value(point[axis], snap_coords, min_upper_bound_idx)

    return tuple(snapped_point)


def _shift_value_signed(
    obj: Box,
    grid: Grid,
    bounds: Bound,
    direction: Direction,
    shift: int,
    name: Optional[str] = None,
) -> float:
    """Calculate the signed distance corresponding to moving the object by ``shift`` number
    of cells in the positive or negative ``direction`` along the dimension given by
    ``obj._normal_axis``.
    """
    if name is None:
        name = f"A '{obj.type}'"

    # get the grid boundaries and sizes along obj normal from the simulation
    normal_axis = obj._normal_axis
    grid_boundaries = grid.boundaries.to_list[normal_axis]
    grid_centers = grid.centers.to_list[normal_axis]

    # get the index of the grid cell where the obj lies
    obj_position = obj.center[normal_axis]
    obj_pos_gt_grid_bounds = np.flatnonzero(obj_position > grid_boundaries)

    # no obj index can be determined
    if len(obj_pos_gt_grid_bounds) == 0 or obj_position > grid_boundaries[-1]:
        raise SetupError(
            f"{name} position '{obj_position}' is outside of simulation bounds '({grid_boundaries[0]}, {grid_boundaries[-1]})' along dimension '{'xyz'[normal_axis]}'."
        )
    obj_index = obj_pos_gt_grid_bounds[-1]
    # shift the obj to the left
    signed_shift = shift if direction == "+" else -shift
    if signed_shift < 0:
        if np.isclose(obj_position, grid_boundaries[obj_index + 1]):
            obj_index += 1
        shifted_index = obj_index + signed_shift
        if shifted_index < 0 or grid_centers[shifted_index] <= bounds[0][normal_axis]:
            raise SetupError(
                f"{name} normal is less than 2 cells to the boundary "
                f"on -{'xyz'[normal_axis]} side. "
                "Please either increase the mesh resolution near the obj or "
                "move the obj away from the boundary."
            )

    # shift the obj to the right
    else:
        shifted_index = obj_index + signed_shift
        if (
            shifted_index >= len(grid_centers)
            or grid_centers[shifted_index] >= bounds[1][normal_axis]
        ):
            raise SetupError(
                f"{name} normal is less than 2 cells to the boundary "
                f"on +{'xyz'[normal_axis]} side."
                "Please either increase the mesh resolution near the obj or "
                "move the obj away from the boundary."
            )

    new_pos = grid_centers[shifted_index]
    return new_pos - obj_position


def _shift_object(obj: Box, grid: Grid, bounds: Bound, direction: Direction, shift: int) -> Box:
    """Move a plane-like object by ``shift`` number
    of cells in the positive or negative ``direction`` along the dimension given by
    ``obj._normal_axis``.
    """
    shift = _shift_value_signed(obj=obj, grid=grid, bounds=bounds, direction=direction, shift=shift)
    new_center = np.array(obj.center)
    new_center[obj._normal_axis] += shift
    # note: if this needs to be generalized beyond absorber, one would probably
    # slightly adjust the code below regarding grid_shift
    return obj.updated_copy(center=tuple(new_center), grid_shift=0)
