"""Utilities for geometry manipulation."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from math import isclose
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import shapely
from pydantic import Field, NonNegativeInt
from shapely.geometry import (
    Polygon,
)
from shapely.geometry.base import (
    BaseMultipartGeometry,
)

from tidy3d._common.components.autograd.utils import get_static
from tidy3d._common.components.base import Tidy3dBaseModel
from tidy3d._common.components.geometry import base, mesh, polyslab, primitives
from tidy3d._common.components.geometry.base import Box
from tidy3d._common.components.types.base import Shapely
from tidy3d._common.exceptions import SetupError, Tidy3dError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike

    from tidy3d._common.components.types.base import (
        ArrayFloat2D,
        Axis,
        MatrixReal4x4,
        PlanePosition,
    )

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
    geometries : list[GeometryType]
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
    list[tuple[Any, Shapely]]
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
    slab_bounds: tuple[float, float]
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
    list[tuple[ArrayFloat2D]]
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

    location: tuple[SnapLocation, SnapLocation, SnapLocation] = Field(
        title="Location",
        description="Describes which positions in the grid will be considered for snapping.",
    )

    behavior: tuple[SnapBehavior, SnapBehavior, SnapBehavior] = Field(
        title="Behavior",
        description="Describes how snapping positions will be chosen.",
    )

    margin: Optional[tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]] = Field(
        (0, 0, 0),
        title="Margin",
        description="Number of additional grid points to consider when expanding or contracting "
        "during snapping. Only applies when ``SnapBehavior`` is ``Expand`` or ``Contract``.",
    )


def get_closest_value(test: float, coords: ArrayLike, upper_bound_idx: int) -> float:
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
