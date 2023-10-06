"""Utilities for geometry manipulation."""
from __future__ import annotations
from typing import Union, Tuple

from ..types import Axis, PlanePosition, Shapely
from ...exceptions import Tidy3dError

from . import base
from . import primitives
from . import polyslab
from . import mesh

GeometryType = Union[
    base.Box,
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
