"""Utilities for 2D geometry manipulation."""
import numpy as np
import shapely
from typing import Tuple, List

from ..types import Axis
from ...constants import fp_eps, inf
from ...exceptions import ValidationError
from ..geometry.base import Geometry, Box, ClipOperation
from ..geometry.primitives import Cylinder
from ..geometry.polyslab import PolySlab
from ..scene import Scene
from ..structure import Structure

# for 2d materials. to find neighboring media, search a distance on either side
# equal to this times the grid size
DIST_NEIGHBOR_REL_2D_MED = 1e-5


def get_bounds(geom: Geometry, axis: Axis) -> Tuple[float, float]:
    """Get the bounds of a geometry in the axis direction."""
    return (geom.bounds[0][axis], geom.bounds[1][axis])


def set_bounds(geom: Geometry, bounds: Tuple[float, float], axis: Axis) -> Geometry:
    """Set the bounds of a geometry in the axis direction."""
    if isinstance(geom, Box):
        new_center = list(geom.center)
        new_center[axis] = (bounds[0] + bounds[1]) / 2
        new_size = list(geom.size)
        new_size[axis] = bounds[1] - bounds[0]
        return geom.updated_copy(center=new_center, size=new_size)
    if isinstance(geom, PolySlab):
        return geom.updated_copy(slab_bounds=bounds)
    if isinstance(geom, Cylinder):
        new_center = list(geom.center)
        new_center[axis] = (bounds[0] + bounds[1]) / 2
        new_length = bounds[1] - bounds[0]
        return geom.updated_copy(center=new_center, length=new_length)
    raise ValidationError(
        "'Medium2D' is only compatible with 'Box', 'PolySlab', or 'Cylinder' geometry."
    )


def get_thickened_geom(geom: Geometry, axis: Axis, axis_dl: float):
    """Helper to return a slightly thickened version of a planar geometry."""
    center = get_bounds(geom, axis)[0]
    neg_thickness = axis_dl * DIST_NEIGHBOR_REL_2D_MED
    pos_thickness = axis_dl * DIST_NEIGHBOR_REL_2D_MED
    return set_bounds(geom, bounds=(center - neg_thickness, center + pos_thickness), axis=axis)


def get_neighbors(
    geom: Geometry,
    axis: Axis,
    axis_dl: float,
    structures: List[Structure],
):
    """Find the neighboring structures and return the tested positions above and below."""
    center = get_bounds(geom, axis)[0]
    check_delta = [
        -axis_dl * DIST_NEIGHBOR_REL_2D_MED,
        axis_dl * DIST_NEIGHBOR_REL_2D_MED,
    ]

    neighbors_below = []
    neighbors_above = []
    for _, position in enumerate(check_delta):
        geom_shifted = set_bounds(geom, bounds=(center + position, center + position), axis=axis)

        # to prevent false positives due to 2D materials touching different materials
        # along their sides, shrink the bounds along the tangential directions by
        # a tiny bit before checking for intersections
        bounds = [list(i) for i in geom_shifted.bounds]
        _, tan_dirs = Geometry.pop_axis([0, 1, 2], axis=axis)
        for dim in tan_dirs:
            if bounds[0][dim] != -inf:
                bounds[0][dim] += fp_eps * max(np.abs(bounds[0][dim]), 1.0)
            if bounds[1][dim] != inf:
                bounds[1][dim] -= fp_eps * max(np.abs(bounds[1][dim]), 1.0)

        structures_side = Scene.intersecting_structures(Box.from_bounds(*bounds), structures)

        if position < 0:
            neighbors_below += list(structures_side)
        else:
            neighbors_above += list(structures_side)

    return neighbors_below, neighbors_above, check_delta


def subdivide(
    geom: Geometry, axis: Axis, axis_dl: float, structures: List[Structure]
) -> List[Tuple[Geometry, Structure, Structure]]:
    """Subdivide Medium2D into pieces with homogeneous substrate / superstrate."""
    """Use the provided average grid size along the axis to search for neighbors."""

    def shapely_to_polyslab(polygon: shapely.Polygon, axis: Axis, center: float) -> PolySlab:
        xx, yy = polygon.exterior.coords.xy
        vertices = list(zip(xx, yy))
        return PolySlab(slab_bounds=(center, center), vertices=vertices, axis=axis)

    def to_multipolygon(shapely_geometry) -> shapely.MultiPolygon:
        return shapely.MultiPolygon(ClipOperation.to_polygon_list(shapely_geometry))

    # Find neighbors and the small offset they were found at
    neighbors_below, neighbors_above, check_delta = get_neighbors(
        geom=geom, axis=axis, axis_dl=axis_dl, structures=structures
    )

    # Compute the plane of intersection
    center = get_bounds(geom, axis)[0]
    coord = "xyz"[axis]
    plane = {coord: center}

    # Convert input geometry into MultiPolygon shapely geometry and track the original structure that references the media properties
    geom_shapely = Geometry.evaluate_inf_shape(
        shapely.MultiPolygon(geom.intersections_plane(**plane))
    )

    plane[coord] = center + check_delta[1]
    above_shapely = [
        (
            Geometry.evaluate_inf_shape(
                shapely.MultiPolygon(structure.geometry.intersections_plane(**plane))
            ),
            structure,
        )
        for structure in neighbors_above
    ]

    plane[coord] = center + check_delta[0]
    below_shapely = [
        [
            Geometry.evaluate_inf_shape(
                shapely.MultiPolygon(structure.geometry.intersections_plane(**plane))
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

    # Create polyslab from subdivided geometry
    polyslab_result = [
        (shapely_to_polyslab(element[0], axis, center), element[1], element[2])
        for element in final_polygons
    ]

    return polyslab_result
