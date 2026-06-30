"""Utilities for 2D geometry manipulation."""

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING

import numpy as np
import shapely

from tidy3d.components.geometry.base import Box, ClipOperation, Geometry, GeometryGroup
from tidy3d.components.geometry.float_utils import increment_float
from tidy3d.components.geometry.polyslab import _MIN_POLYGON_AREA, PolySlab
from tidy3d.components.scene import Scene
from tidy3d.constants import fp_eps

# Relative tolerance for filtering slivers based on grid size.
# Polygons with area < min_cell_area * _SLIVER_AREA_RTOL are filtered.
# Polygons with any bounding box dimension < min_cell_size * _SLIVER_DIM_RTOL are filtered.
_SLIVER_AREA_RTOL = 1e-4
_SLIVER_DIM_RTOL = 1e-4

if TYPE_CHECKING:
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.medium import MediumType3D
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import Axis, Shapely


def snap_coordinate_to_grid(grid: Grid, center: float, axis: Axis) -> float:
    """2D materials are snapped to grid along their normal axis"""
    new_centers = grid.boundaries.to_list[axis]
    new_center = new_centers[np.argmin(abs(new_centers - center))]
    return new_center


def get_bounds(geom: Geometry, axis: Axis) -> tuple[float, float]:
    """Get the bounds of a geometry in the axis direction."""
    return (geom.bounds[0][axis], geom.bounds[1][axis])


def structures_at_offset(
    geom: Geometry,
    axis: Axis,
    structures: list[Structure],
    offset: float,
) -> tuple[Structure, ...]:
    """Structures intersecting ``geom``'s footprint collapsed onto the plane offset by ``offset``
    from its lower bound along ``axis``.

    This is the shared probe used to find the media neighbouring a zero-thickness object: the
    footprint is moved to ``lower_bound + offset`` along ``axis`` and its tangential extent is shrunk
    slightly so a probe sitting on a material seam does not pick up the structures merely touching its
    sides. Callers choose ``offset`` to suit their needs (an :func:`increment_float` nudge to read the
    media at the object's limit, or a fraction of the grid step for a robust, grid-aware sample).
    """
    center = get_bounds(geom, axis)[0]
    geom_shifted = geom._update_from_bounds(bounds=(center + offset, center + offset), axis=axis)

    # to prevent false positives due to objects touching different materials along their sides,
    # shrink the bounds along the tangential directions by a tiny bit before checking intersections
    bounds = [list(i) for i in geom_shifted.bounds]
    _, tan_dirs = Geometry.pop_axis([0, 1, 2], axis=axis)
    for dim in tan_dirs:
        # Don't shrink if the width is already close to 0
        if not isclose(bounds[0][dim], bounds[1][dim], rel_tol=2 * fp_eps):
            bounds[0][dim] = increment_float(bounds[0][dim], 1.0)
            bounds[1][dim] = increment_float(bounds[1][dim], -1.0)

    return Scene.intersecting_structures(Box.from_bounds(*bounds), structures)


def get_neighbors(
    geom: Geometry,
    axis: Axis,
    structures: list[Structure],
) -> tuple[tuple[Structure, ...], tuple[Structure, ...], tuple[float, float]]:
    """Find the neighboring structures and return the tested positions just below and above.

    The probe sits one :func:`increment_float` step to either side, with the magnitude floored at
    ``fp_eps`` (see the comment on ``check_delta``).
    """
    center = get_bounds(geom, axis)[0]
    # ``fp_eps`` is the float32 epsilon: above the float64 ULP at realistic coordinates so
    # ``center + offset`` always resolves (an unfloored nudge degenerates to ~1e-38 at the origin
    # and is lost), yet still infinitesimal next to any cell -- which keeps ``subdivide`` (which
    # reuses ``check_delta`` as its partitioning plane) reading a sloped substrate's cross-section
    # at the sheet, not away from it.
    check_delta = [
        min(increment_float(center, -1.0) - center, -fp_eps),
        max(increment_float(center, 1.0) - center, fp_eps),
    ]
    neighbors_below = list(structures_at_offset(geom, axis, structures, check_delta[0]))
    neighbors_above = list(structures_at_offset(geom, axis, structures, check_delta[1]))
    return neighbors_below, neighbors_above, check_delta


def snap_to_dual_cell(grid: Grid, coordinate: float, axis: Axis) -> tuple[float, float]:
    """Center and width of the dual grid cell along ``axis`` containing ``coordinate`` -- the span
    between the two primal-grid-cell centers straddling it.

    A lumped element's transverse edges snap to grid centers, so this dual cell is the transverse
    footprint of the smallest such element once volumetricized; its width is the mean of the two
    adjacent primal cells (the same quantity ``get_dls`` returns for the normal direction). Aligning
    the promoted element to this cell makes the later center-snap idempotent, so the volumetric width
    is exactly one dual cell rather than the over-expanded result of re-snapping a wider box."""
    centers = np.array(grid.centers.to_list[axis])
    idx = min(max(int(np.searchsorted(centers, coordinate)), 1), len(centers) - 1)
    lower, upper = float(centers[idx - 1]), float(centers[idx])
    return (lower + upper) / 2, upper - lower


def choose_line_normal_axis(
    geom: Geometry,
    voltage_axis: Axis,
    structures: list[Structure],
    background_medium: MediumType3D,
    grid: Grid,
) -> Axis:
    """Pick the normal axis used to realize a one-dimensional (line) lumped element as a
    :class:`.Medium2D` sheet, using the same neighbour finder as the planar path
    (:func:`get_neighbors`).

    Returns the first transverse axis across which the surrounding media differ, so the sheet
    normal points across the real interface; if neither does, the first zero-size axis is used
    (matching :attr:`.RectangularLumpedElement.normal_axis`'s default).
    """

    def topmost_medium(neighbors: tuple[Structure, ...]) -> MediumType3D:
        """Highest-priority (last) intersecting structure's medium, else the background."""
        return neighbors[-1].medium if neighbors else background_medium

    # Snap to the grid boundary the 2D material will land on later (without modifying the element).
    snapped = [snap_coordinate_to_grid(grid, coord, axis) for axis, coord in enumerate(geom.center)]
    for candidate_normal in (a for a in range(3) if a != voltage_axis):
        # one-dual-cell footprint along the remaining transverse axis, so the probe is a finite
        # sheet rather than a zero-area line
        footprint_axis = 3 - voltage_axis - candidate_normal
        _, footprint_width = snap_to_dual_cell(grid, snapped[footprint_axis], footprint_axis)
        size = [0.0, 0.0, 0.0]
        size[voltage_axis] = geom.size[voltage_axis]
        size[footprint_axis] = footprint_width
        dummy = Box(center=tuple(snapped), size=tuple(size))
        neighbors_below, neighbors_above, _ = get_neighbors(dummy, candidate_normal, structures)
        if topmost_medium(neighbors_below) != topmost_medium(neighbors_above):
            return candidate_normal
    # no interface found -- the default normal, matching RectangularLumpedElement.normal_axis
    return geom.size.index(0.0)


def _is_sliver_polygon(polygon: shapely.Polygon, axis: Axis, grid: Grid | None) -> bool:
    """Check if a polygon is a sliver (degenerate thin polygon) based on grid size.

    A polygon is considered a sliver if:
    - Its area is below the minimum polygon area threshold, OR
    - When grid is provided: its area is much smaller than the minimum grid cell area
      in the 2D plane, OR any of its bounding box dimensions is much smaller than
      the corresponding minimum grid cell size.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon to check.
    axis : Axis
        The normal axis of the 2D plane.
    grid : Grid | None
        Optional grid to use for computing relative thresholds based on cell sizes.

    Returns
    -------
    bool
        True if the polygon is a sliver and should be filtered out.
    """
    area = polygon.area

    # Always apply absolute minimum area threshold
    if area < _MIN_POLYGON_AREA:
        return True

    # If no grid provided, only use absolute threshold
    if grid is None:
        return False

    # Get tangential axes (the 2D plane axes)
    _, tan_axes = Geometry.pop_axis([0, 1, 2], axis=axis)

    # Get minimum cell sizes along tangential directions
    grid_sizes = grid.sizes
    sizes_list = grid_sizes.to_list
    min_cell_sizes = [np.min(sizes_list[ax]) for ax in tan_axes]
    min_cell_area = min_cell_sizes[0] * min_cell_sizes[1]

    # Check if area is too small relative to grid cell area
    if area < min_cell_area * _SLIVER_AREA_RTOL:
        return True

    # Check if any bounding box dimension is too small relative to grid cell size
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    bbox_dims = [bounds[2] - bounds[0], bounds[3] - bounds[1]]  # (width, height)

    for dim_size, min_grid_size in zip(bbox_dims, min_cell_sizes):
        if dim_size < min_grid_size * _SLIVER_DIM_RTOL:
            return True

    return False


def subdivide(
    geom: Geometry, structures: list[Structure], grid: Grid | None = None
) -> list[tuple[Geometry, Structure, Structure]]:
    """Subdivide geometry associated with a :class:`.Medium2D` into partitions
    that each have a homogeneous substrate / superstrate. Partitions are computed
    using ``shapely`` boolean operations on polygons.

    Parameters
    ----------
    geom : Geometry
        A 2D geometry associated with the :class:`.Medium2D`.
    structures : list[Structure]
        List of structures that are checked for intersection with ``geom``.
    grid : Grid, optional
        Grid to use for filtering sliver polygons based on cell sizes. When provided,
        polygons with area much smaller than the minimum grid cell area, or with any
        bounding box dimension much smaller than the corresponding minimum grid cell
        size, are filtered out. This prevents numerical artifacts from very thin
        degenerate polygons that can arise from boolean operations.

    Returns
    -------
    list[tuple[Geometry, Structure, Structure]]
        List of the created partitions. Each element of the list represents a partition of the 2D geometry,
        which includes the newly created structures below and above.

    """

    def shapely_to_polyslab(polygon: shapely.Polygon, axis: Axis, center: float) -> Geometry:
        def ring_vertices(ring: shapely.LinearRing) -> list[tuple[float, float]]:
            xx, yy = ring.coords.xy
            return list(zip(xx, yy))

        polyslab = PolySlab(
            slab_bounds=(center, center),
            vertices=ring_vertices(polygon.exterior),
            axis=axis,
        )
        if len(polygon.interiors) == 0:
            return polyslab

        interiors = [
            PolySlab(
                slab_bounds=(center, center),
                vertices=ring_vertices(interior),
                axis=axis,
            )
            for interior in polygon.interiors
        ]
        return polyslab - GeometryGroup(geometries=interiors)

    def to_multipolygon(shapely_geometry: Shapely) -> shapely.MultiPolygon:
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

    # Create polyslab from subdivided geometry, while filtering out sliver polygons
    polyslab_result = [
        (shapely_to_polyslab(element[0], axis, center), element[1], element[2])
        for element in final_polygons
        if not _is_sliver_polygon(element[0], axis, grid)
    ]

    return polyslab_result
