"""Utilities for geometry manipulation."""

from __future__ import annotations

from enum import Enum
from math import isclose
from typing import Tuple, Union

import numpy as np
import pydantic as pydantic

from ...constants import fp_eps
from ...exceptions import Tidy3dError
from ..base import Tidy3dBaseModel
from ..geometry.base import Box
from ..grid.grid import Grid
from ..types import ArrayFloat2D, Axis, MatrixReal4x4, PlanePosition, Shapely
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


def flatten_groups(*geometries: GeometryType, flatten_nonunion_type: bool = False) -> GeometryType:
    """Iterates over all geometries, flattening groups and unions.

    Parameters
    ----------
    *geometries : GeometryType
        Geometries to flatten.

    flatten_nonunion_type : bool = False
        If ``False``, only flatten geometry unions (and ``GeometryGroup``). If ``True``, flatten
        all clip operations.

    Yields
    ------
    GeometryType
        Geometries after flattening groups and unions.
    """
    for geometry in geometries:
        if isinstance(geometry, base.GeometryGroup):
            yield from flatten_groups(
                *geometry.geometries, flatten_nonunion_type=flatten_nonunion_type
            )
        elif isinstance(geometry, base.ClipOperation) and (
            flatten_nonunion_type or geometry.operation == "union"
        ):
            yield from flatten_groups(
                geometry.geometry_a,
                geometry.geometry_b,
                flatten_nonunion_type=flatten_nonunion_type,
            )
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
    slab_bounds: Tuple[float, float],
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
        return [(shape.exterior.coords[:-1],) + tuple(hole.coords[:-1] for hole in shape.interiors)]
    if shape.geom_type in {"MultiPolygon", "GeometryCollection"}:
        return sum(vertices_from_shapely(geo) for geo in shape.geoms)

    raise Tidy3dError(f"Shape {shape} cannot be converted to Geometry.")


def validate_no_transformed_polyslabs(geometry: GeometryType, transform: MatrixReal4x4 = None):
    """Prevents the creation of slanted polyslabs rotated out of plane."""
    if transform is None:
        transform = np.eye(4)
    if isinstance(geometry, polyslab.PolySlab):
        if not (
            isclose(geometry.sidewall_angle, 0)
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
    Off = 4
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


def snap_box_to_grid(grid: Grid, box: Box, snap_spec: SnappingSpec, rtol=fp_eps) -> Box:
    """Snaps a :class:`.Box` to the grid, so that the boundaries of the box are aligned with grid centers or boundaries.
    The way in which each dimension of the `box` is snapped to the grid is controlled by ``snap_spec``.
    """

    def closest_value(test: float, first: float, second: float) -> float:
        """Helper to choose the closest value to a given test value."""
        dfirst = np.abs(test - first)
        dsecond = np.abs(test - second)
        return first if dfirst < dsecond else second

    def find_snapping_locations(
        min: float, max: float, coords: np.ndarray, snap_type: SnapBehavior
    ) -> tuple[float, float]:
        """Helper that snaps a supplied interval [min, max] to a sorted array representing
        coordinate values.
        """
        # Locate the interval that includes the min and max
        min_left_idx = np.searchsorted(coords, min, side="left")
        max_left_idx = np.searchsorted(coords, max, side="left")
        if snap_type == SnapBehavior.Closest:
            if min_left_idx > 0:
                min_snap = closest_value(min, coords[min_left_idx - 1], coords[min_left_idx])
            else:
                min_snap = coords[min_left_idx]
            if max_left_idx > 0:
                max_snap = closest_value(max, coords[max_left_idx - 1], coords[max_left_idx])
            else:
                max_snap = coords[max_left_idx]
        elif snap_type == SnapBehavior.Expand:
            if min_left_idx == 0 or np.isclose(coords[min_left_idx], min, rtol=rtol):
                min_snap = coords[min_left_idx]
            else:
                min_snap = coords[min_left_idx - 1]
            if max_left_idx > 0 and np.isclose(coords[max_left_idx - 1], max, rtol=rtol):
                max_snap = coords[max_left_idx - 1]
            else:
                max_snap = coords[max_left_idx]
        else:  # SnapType.Contract
            if min_left_idx > 0 and np.isclose(coords[min_left_idx - 1], min, rtol=rtol):
                min_snap = coords[min_left_idx - 1]
            else:
                min_snap = coords[min_left_idx]
            if max_left_idx == 0 or np.isclose(coords[max_left_idx], max, rtol=rtol):
                max_snap = coords[max_left_idx]
            else:
                max_snap = coords[max_left_idx - 1]
        return (min_snap, max_snap)

    # Iterate over each axis and apply the specified snapping behavior.
    min_b = list(box.bounds[0])
    max_b = list(box.bounds[1])
    grid_bounds = grid.boundaries.to_list
    grid_centers = grid.centers.to_list
    for axis in range(3):
        snap_location = snap_spec.location[axis]
        snap_type = snap_spec.behavior[axis]
        if snap_type == SnapBehavior.Off:
            continue
        if snap_location == SnapLocation.Boundary:
            snap_coords = np.array(grid_bounds[axis])
        elif snap_location == SnapLocation.Center:
            snap_coords = np.array(grid_centers[axis])

        box_min = min_b[axis]
        box_max = max_b[axis]

        (new_min, new_max) = find_snapping_locations(box_min, box_max, snap_coords, snap_type)
        min_b[axis] = new_min
        max_b[axis] = new_max
    return Box.from_bounds(min_b, max_b)
