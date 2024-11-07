"""Utilities for 2D geometry manipulation."""

from math import isclose
from typing import List, Tuple

import numpy as np
import shapely

from ...constants import fp_eps, inf
from ..geometry.base import Box, ClipOperation, Geometry
from ..geometry.polyslab import _MIN_POLYGON_AREA, PolySlab
from ..grid.grid import Grid
from ..scene import Scene
from ..structure import Structure
from ..types import Axis


def increment_float(val: float, sign) -> float:
    """Applies a small positive or negative shift as though `val` is a 32bit float
    using numpy.nextafter, but additionally handles some corner cases.
    """
    # Infinity is left unchanged
    if val == inf or val == -inf:
        return val

    if sign >= 0:
        sign = 1
    else:
        sign = -1

    # Avoid small increments within subnormal values
    if np.abs(val) <= np.finfo(np.float32).tiny:
        return val + sign * np.finfo(np.float32).tiny

    # Numpy seems to skip over the increment from -0.0 and +0.0
    # which is different from c++
    val_inc = np.nextafter(val, sign * inf, dtype=np.float32)

    return np.float32(val_inc)


def snap_coordinate_to_grid(grid: Grid, center: float, axis: Axis) -> float:
    """2D materials are snapped to grid along their normal axis"""
    new_centers = grid.boundaries.to_list[axis]
    new_center = new_centers[np.argmin(abs(new_centers - center))]
    return new_center


def get_bounds(geom: Geometry, axis: Axis) -> Tuple[float, float]:
    """Get the bounds of a geometry in the axis direction."""
    return (geom.bounds[0][axis], geom.bounds[1][axis])


def get_thickened_geom(geom: Geometry, axis: Axis):
    """Helper to return a slightly thickened version of a planar geometry."""
    center = get_bounds(geom, axis)[0]
    neg_thickness = increment_float(center, -1.0)
    pos_thickness = increment_float(center, 1.0)
    return geom._update_from_bounds(bounds=(neg_thickness, pos_thickness), axis=axis)


def get_neighbors(
    geom: Geometry,
    axis: Axis,
    structures: List[Structure],
):
    """Find the neighboring structures and return the tested positions above and below."""
    center = get_bounds(geom, axis)[0]
    check_delta = [
        increment_float(center, -1.0) - center,
        increment_float(center, 1.0) - center,
    ]

    neighbors_below = []
    neighbors_above = []
    for _, position in enumerate(check_delta):
        geom_shifted = geom._update_from_bounds(
            bounds=(center + position, center + position), axis=axis
        )

        # to prevent false positives due to 2D materials touching different materials
        # along their sides, shrink the bounds along the tangential directions by
        # a tiny bit before checking for intersections
        bounds = [list(i) for i in geom_shifted.bounds]
        _, tan_dirs = Geometry.pop_axis([0, 1, 2], axis=axis)
        for dim in tan_dirs:
            # Don't shrink if the width is already close to 0
            if not isclose(bounds[0][dim], bounds[1][dim], rel_tol=2 * fp_eps):
                bounds[0][dim] = increment_float(bounds[0][dim], 1.0)
                bounds[1][dim] = increment_float(bounds[1][dim], -1.0)

        structures_side = Scene.intersecting_structures(Box.from_bounds(*bounds), structures)

        if position < 0:
            neighbors_below += list(structures_side)
        else:
            neighbors_above += list(structures_side)

    return neighbors_below, neighbors_above, check_delta


def subdivide(
    geom: Geometry, structures: List[Structure]
) -> List[Tuple[Geometry, Structure, Structure]]:
    """Subdivide geometry associated with a :class:`.Medium2D` into partitions
    that each have a homogeneous substrate / superstrate. Partitions are computed
    using ``shapely`` boolean operations on polygons.

    Parameters
    ----------
    geom : Geometry
        A 2D geometry associated with the :class:`.Medium2D`.
    structures : List[Structure]
        List of structures that are checked for intersection with ``geom``.

    Returns
    -------
    List[Tuple[Geometry, Structure, Structure]]
        List of the created partitions. Each element of the list represents a partition of the 2D geometry,
        which includes the newly created structures below and above.

    """

    def shapely_to_polyslab(polygon: shapely.Polygon, axis: Axis, center: float) -> PolySlab:
        xx, yy = polygon.exterior.coords.xy
        vertices = list(zip(xx, yy))
        return PolySlab(slab_bounds=(center, center), vertices=vertices, axis=axis)

    def to_multipolygon(shapely_geometry) -> shapely.MultiPolygon:
        return shapely.MultiPolygon(ClipOperation.to_polygon_list(shapely_geometry))

    axis = geom._normal_2dmaterial
    # Find neighbors and the small offset they were found at
    neighbors_below, neighbors_above, check_delta = get_neighbors(
        geom=geom, axis=axis, structures=structures
    )

    # Compute the plane of intersection
    center = get_bounds(geom, axis)[0]
    coord = "xyz"[axis]
    plane = {coord: center}

    # Convert input geometry into MultiPolygon shapely geometry and track the original structure that references the media properties
    geom_shapely = Geometry.evaluate_inf_shape(shapely.union_all(geom.intersections_plane(**plane)))

    plane[coord] = center + check_delta[1]
    above_shapely = [
        (
            Geometry.evaluate_inf_shape(
                shapely.union_all(structure.geometry.intersections_plane(**plane))
            ),
            structure,
        )
        for structure in neighbors_above
    ]

    plane[coord] = center + check_delta[0]
    below_shapely = [
        [
            Geometry.evaluate_inf_shape(
                shapely.union_all(structure.geometry.intersections_plane(**plane))
            ),
            structure,
        ]
        for structure in neighbors_below
    ]

    # First find the intersections of 2d material with all structures above in reverse order
    above_intersections = []
    for mp_structure in reversed(above_shapely):
        # If the 2D structure overlaps completely with all previously tested structures above then there is no more work to do
        if not geom_shapely:
            break

        intersection_res = shapely.intersection(geom_shapely, mp_structure[0])
        intersection_mp = to_multipolygon(intersection_res)
        difference_res = shapely.difference(geom_shapely, mp_structure[0])
        geom_shapely = to_multipolygon(difference_res)

        if intersection_mp:
            above_intersections.append((intersection_mp, mp_structure[1]))
    above_intersections.reverse()

    # Next find intersections of previous result with all structures below
    # List that stores a tuple of a MultiPolygon, the adjacent structure below, and the adjacent structure above
    both_intersections = []
    # Similar to above, but keep track of both differences of the previous result and the below polygons for faster termination
    for mp_structure_above in reversed(above_intersections):
        above_intersection = mp_structure_above[0]
        for mp_structure_below in reversed(below_shapely):
            # Possible to finish loops early
            if not above_intersection:
                break
            if not mp_structure_below[0]:
                continue

            intersection_res = shapely.intersection(above_intersection, mp_structure_below[0])
            intersection_mp = to_multipolygon(intersection_res)
            above_difference = to_multipolygon(
                shapely.difference(above_intersection, mp_structure_below[0])
            )
            below_difference = to_multipolygon(
                shapely.difference(mp_structure_below[0], above_intersection)
            )
            # Update polygons by subtracting the intersecting parts
            mp_structure_below[0] = below_difference
            above_intersection = above_difference
            if intersection_mp:
                both_intersections.append(
                    (intersection_mp, mp_structure_below[1], mp_structure_above[1])
                )
    both_intersections.reverse()

    # If there turns out to be only one substrate/superstrate combo, then return the original geometry
    if len(both_intersections) == 1:
        return [(geom, both_intersections[0][1], both_intersections[0][2])]

    # Flatten into array of only polygons and adjacent structures
    # The geometry produced should all be MultiPolygons
    final_polygons = []
    for element in both_intersections:
        for polygon in element[0].geoms:
            final_polygons.append((polygon, element[1], element[2]))

    # Create polyslab from subdivided geometry, while filtering out any
    # polygons with very small areas
    polyslab_result = [
        (shapely_to_polyslab(element[0], axis, center), element[1], element[2])
        for element in final_polygons
        if element[0].area >= _MIN_POLYGON_AREA
    ]

    return polyslab_result
