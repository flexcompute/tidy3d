"""Layer-refinement gap detection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tidy3d.components.geometry.base import Box, ClipOperation
from tidy3d.components.grid.grid_spec.constants import GAP_MESHING_TOL
from tidy3d.constants import inf

if TYPE_CHECKING:
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.types import ArrayFloat1D, ArrayFloat2D, CoordinateOptional, Shapely

    from .model import LayerRefinementSpec


def _find_vertical_intersections(
    self: LayerRefinementSpec,
    grid_x_coords: ArrayFloat1D,
    grid_y_coords: ArrayFloat1D,
    poly_vertices: ArrayFloat2D,
    boundary: tuple[str | None, str | None],
) -> tuple[np.typing.NDArray[np.int_], np.typing.NDArray[np.float64]]:
    """Detect intersection points of single polygon and vertical grid lines."""

    # indices of cells that contain intersection with grid lines (left edge of a cell)
    cells_ij = []
    # relative displacements of intersection from the bottom of the cell along y axis
    cells_dy = []
    # whether intersections are valid: if polygon segment is almost parallel to
    # a grid line, we mark those intersection as invalid
    cells_valid = []

    # for each polygon vertex find the index of the first grid line on the right
    # Use searchsorted for O(n log m) instead of O(n * m) with argmax broadcasting
    grid_lines_on_right = np.searchsorted(grid_x_coords, poly_vertices[:, 0], side="left")
    # once we know these indices then we can find grid lines intersected by the i-th
    # segment of the polygon as
    # [grid_lines_on_right[i], grid_lines_on_right[i+1]) for grid_lines_on_right[i] > grid_lines_on_right[i+1]
    # or
    # [grid_lines_on_right[i+1], grid_lines_on_right[i]) for grid_lines_on_right[i] < grid_lines_on_right[i+1]

    # loop over segments of the polygon and determine in which cells and where exactly they cross grid lines
    # v_beg and v_end are the starting and ending points of the segment
    # ind_beg and ind_end are starting and ending indices of vertical grid lines that the segment intersects
    # as described above
    for ind_beg, ind_end, v_beg, v_end in zip(
        grid_lines_on_right,
        np.roll(grid_lines_on_right, -1),
        poly_vertices,
        np.roll(poly_vertices, axis=0, shift=-1),
    ):
        # no intersections
        if ind_end == ind_beg:
            continue

        # intersects one grid line but almost parallel to it
        not_nearly_parallel = True
        if np.abs(ind_end - ind_beg) == 1:
            delta_x = np.abs(v_beg[0] - v_end[0])
            delta_y = np.abs(v_beg[1] - v_end[1])
            grid_size_x = np.abs(grid_x_coords[ind_beg - 1] - grid_x_coords[ind_end - 1])
            # we discard segments that are substantially vertical with respect to both grid step size
            # and its own vertical size (so that we don't discard tiny pieces of curved boundaries)
            if delta_x < 2 * GAP_MESHING_TOL * min(grid_size_x, delta_y):
                not_nearly_parallel = False

        # sort vertices in ascending order to make treatmeant unifrom
        reverse = False
        if ind_beg > ind_end:
            reverse = True
            ind_beg, ind_end, v_beg, v_end = ind_end, ind_beg, v_end, v_beg

        # x coordinates are simply x coordinates of intersected vertical grid lines
        intersections_x = grid_x_coords[ind_beg:ind_end]

        # y coordinates can be found from line equation
        intersections_y = v_beg[1] + (v_end[1] - v_beg[1]) / (v_end[0] - v_beg[0]) * (
            intersections_x - v_beg[0]
        )

        # however, some of the vertical lines might be crossed
        # outside of computational domain
        # so we need to see which ones are actually inside along y axis
        inds_inside_grid = np.logical_and(
            intersections_y >= grid_y_coords[0], intersections_y <= grid_y_coords[-1]
        )

        intersections_y = intersections_y[inds_inside_grid]

        # find i and j indices of cells which contain these intersections

        # i indices are simply indices of crossed vertical grid lines
        cell_is = np.arange(ind_beg, ind_end)[inds_inside_grid]

        # j indices can be computed by finding insertion indices
        # of y coordinates of intersection points into array of y coordinates
        # of the grid lines that preserve sorting
        cell_js = np.searchsorted(grid_y_coords, intersections_y) - 1

        # find local dy, that is, the distance between the intersection point
        # and the bottom edge of the cell
        dy = (intersections_y - grid_y_coords[cell_js]) / (
            grid_y_coords[cell_js + 1] - grid_y_coords[cell_js]
        )

        # preserve uniform ordering along perimeter of the polygon
        if reverse:
            cell_is = cell_is[::-1]
            cell_js = cell_js[::-1]
            dy = dy[::-1]

        # record info
        cells_ij.append(np.transpose([cell_is, cell_js]))
        cells_dy.append(dy)
        cells_valid.append(not_nearly_parallel * np.ones_like(dy))

    if len(cells_ij) > 0:
        cells_ij = np.concatenate(cells_ij)
        cells_dy = np.concatenate(cells_dy)
        cells_valid = np.concatenate(cells_valid)

        # Filter from re-entering subcell features. That is, we discard any consecutive
        # intersections if they are crossing the same edge. This happens, for example,
        # when a tiny feature pokes through an edge. This helps not to set dl_min
        # to a very low value, and take into account only actual gaps and strips.

        # To do that we use the fact that intersection points are recorded and stored
        # in the order as they appear along the border of the polygon.

        # first we calculate linearized indices of edges (cells) they cross
        linear_index = cells_ij[:, 0] * len(grid_y_coords) + cells_ij[:, 1]

        # then look at the differences with next and previous neighbors
        fwd_diff = linear_index - np.roll(linear_index, -1)
        bwd_diff = np.roll(fwd_diff, 1)

        # an intersection point is not a part of a "re-entering subcell feature"
        # if it doesn't cross the same edges as its neighbors
        valid = np.logical_and(fwd_diff != 0, bwd_diff != 0)
        valid = np.logical_and(valid, cells_valid)

        cells_dy = cells_dy[valid]
        cells_ij = cells_ij[valid]

        # Now we are duplicating intersection points very close to cell boundaries
        # to corresponding adjacent cells. Basically, if we have a line crossing
        # very close to a grid node, we consider that it crosses edges on both sides
        # from that node. That is, this serves as a tolerance allowance.
        # Note that duplicated intersections and their originals will be snapped to
        # cell boundaries during quantization later.
        close_to_zero = cells_dy < GAP_MESHING_TOL
        close_to_one = (1.0 - cells_dy) < GAP_MESHING_TOL

        points_to_duplicate_near_zero = cells_ij[close_to_zero]
        points_to_duplicate_near_one = cells_ij[close_to_one]

        # if we go beyond simulation domain boundary, either ignore
        # or wrap periodically depending on boundary conditions
        cells_ij_zero_side = points_to_duplicate_near_zero - np.array([0, 1])
        cells_zero_side_out = cells_ij_zero_side[:, 1] == -1
        if boundary[0] == "periodic":
            cells_ij_zero_side[cells_zero_side_out, 1] = len(grid_y_coords) - 2
        else:
            cells_ij_zero_side = cells_ij_zero_side[cells_zero_side_out == 0]

        cells_ij_one_side = points_to_duplicate_near_one + np.array([0, 1])
        cells_one_side_out = cells_ij_one_side[:, 1] == len(grid_y_coords) - 1
        if boundary[1] == "periodic":
            cells_ij_one_side[cells_one_side_out, 1] = 0
        else:
            cells_ij_one_side = cells_ij_one_side[cells_one_side_out == 0]

        cells_ij = np.concatenate(
            [
                cells_ij,
                cells_ij_zero_side,
                cells_ij_one_side,
            ]
        )
        cells_dy = np.concatenate(
            [
                cells_dy,
                np.ones(len(cells_ij_zero_side)),
                np.zeros(len(cells_ij_one_side)),
            ]
        )
    else:
        cells_ij = np.empty((0, 2), dtype=int)
        cells_dy = np.empty(0, dtype=float)

    return cells_ij, cells_dy


def _process_poly(
    self: LayerRefinementSpec,
    grid_x_coords: ArrayFloat1D,
    grid_y_coords: ArrayFloat1D,
    poly_vertices: ArrayFloat2D,
    boundaries: tuple[tuple[str | None, str | None], tuple[str | None, str | None]],
) -> tuple[
    np.typing.NDArray[np.int_],
    np.typing.NDArray[np.float64],
    np.typing.NDArray[np.int_],
    np.typing.NDArray[np.float64],
]:
    """Detect intersection points of single polygon and grid lines."""

    # find cells that contain intersections of vertical grid lines
    # and relative locations of those intersections (along y axis)
    v_cells_ij, v_cells_dy = self._find_vertical_intersections(
        grid_x_coords, grid_y_coords, poly_vertices, boundaries[1]
    )

    # find cells that contain intersections of horizontal grid lines
    # and relative locations of those intersections (along x axis)
    # reuse the same command but flip dimensions
    h_cells_ij, h_cells_dx = self._find_vertical_intersections(
        grid_y_coords, grid_x_coords, np.flip(poly_vertices, axis=1), boundaries[0]
    )
    if len(h_cells_ij) > 0:
        # flip dimensions back
        h_cells_ij = np.roll(h_cells_ij, axis=1, shift=1)

    return v_cells_ij, v_cells_dy, h_cells_ij, h_cells_dx


def _process_slice(
    self: LayerRefinementSpec,
    x: ArrayFloat1D,
    y: ArrayFloat1D,
    merged_geos: list[tuple[Any, Shapely]],
    boundaries: list[list[str | None]],
) -> tuple[
    np.typing.NDArray[np.int_],
    np.typing.NDArray[np.float64],
    np.typing.NDArray[np.int_],
    np.typing.NDArray[np.float64],
]:
    """Detect intersection points of geometries boundaries and grid lines."""

    # cells that contain intersections of vertical grid lines
    v_cells_ij = []
    # relative locations of those intersections (along y axis)
    v_cells_dy = []

    # cells that contain intersections of horizontal grid lines
    h_cells_ij = []
    # relative locations of those intersections (along x axis)
    h_cells_dx = []

    # for PEC and PMC boundary - treat them as PEC structure
    # so that gaps are resolved near boundaries if any
    nx = len(x)
    ny = len(y)

    if boundaries[0][0] == "pec/pmc":
        h_cells_ij.append(np.transpose([np.zeros(ny), np.arange(ny)]).astype(int))
        h_cells_dx.append(np.zeros(ny))

    if boundaries[0][1] == "pec/pmc":
        h_cells_ij.append(np.transpose([(nx - 2) * np.ones(ny), np.arange(ny)]).astype(int))
        h_cells_dx.append(np.ones(ny))

    if boundaries[1][0] == "pec/pmc":
        v_cells_ij.append(np.transpose([np.arange(nx), np.zeros(nx)]).astype(int))
        v_cells_dy.append(np.zeros(nx, dtype=int))

    if boundaries[1][1] == "pec/pmc":
        v_cells_ij.append(np.transpose([np.arange(nx), (ny - 2) * np.ones(nx)]).astype(int))
        v_cells_dy.append(np.ones(nx))

    # loop over all shapes
    for mat, shapes in merged_geos:
        if not mat.is_pec:
            # note that we expect LossyMetal's converted into PEC in merged_geos
            # that is why we are not checking for that separately
            continue
        polygon_list = ClipOperation.to_polygon_list(shapes)
        for poly in polygon_list:
            poly = poly.normalize().buffer(0)

            # find intersections of a polygon with grid lines
            # specifically:
            # 0. cells that contain intersections of vertical grid lines
            # 1. relative locations of those intersections along y axis
            # 2. cells that contain intersections of horizontal grid lines
            # 3. relative locations of those intersections along x axis
            data = self._process_poly(x, y, np.array(poly.exterior.coords)[:-1], boundaries)

            if len(data[0]) > 0:
                v_cells_ij.append(data[0])
                v_cells_dy.append(data[1])

            if len(data[2]) > 0:
                h_cells_ij.append(data[2])
                h_cells_dx.append(data[3])

            # in case the polygon has holes
            for poly_inner in poly.interiors:
                data = self._process_poly(x, y, np.array(poly_inner.coords)[:-1], boundaries)
                if len(data[0]) > 0:
                    v_cells_ij.append(data[0])
                    v_cells_dy.append(data[1])

                if len(data[2]) > 0:
                    h_cells_ij.append(data[2])
                    h_cells_dx.append(data[3])

    if len(v_cells_ij) > 0:
        v_cells_ij = np.concatenate(v_cells_ij)
        v_cells_dy = np.concatenate(v_cells_dy)
    else:
        v_cells_ij = np.empty((0, 2), dtype=int)
        v_cells_dy = np.empty(0, dtype=float)

    if len(h_cells_ij) > 0:
        h_cells_ij = np.concatenate(h_cells_ij)
        h_cells_dx = np.concatenate(h_cells_dx)
    else:
        h_cells_ij = np.empty((0, 2), dtype=int)
        h_cells_dx = np.empty(0, dtype=float)

    return v_cells_ij, v_cells_dy, h_cells_ij, h_cells_dx


def _generate_horizontal_snapping_lines(
    self: LayerRefinementSpec,
    grid_y_coords: ArrayFloat1D,
    intersected_cells_ij: np.typing.NDArray[np.int_],
    relative_vert_disp: np.typing.NDArray[np.float64],
) -> tuple[list[float], float]:
    """Convert a list of intersections of vertical grid lines, given as coordinates of cells
    and relative vertical displacement inside each cell, into locations of snapping lines that
    resolve thin gaps and strips.
    """
    min_gap_width = inf

    snapping_lines_y = []
    if len(intersected_cells_ij) > 0:
        # quantize intersection locations
        relative_vert_disp = np.round(relative_vert_disp / GAP_MESHING_TOL).astype(int)
        cell_linear_inds = (
            intersected_cells_ij[:, 0] * len(grid_y_coords) + intersected_cells_ij[:, 1]
        )
        cell_linear_inds_and_disps = np.transpose([cell_linear_inds, relative_vert_disp])
        # remove duplicates
        cell_linear_inds_and_disps_unique = np.unique(cell_linear_inds_and_disps, axis=0)

        # count intersections of vertical grid lines in each cell
        cell_linear_inds_unique, counts = np.unique(
            cell_linear_inds_and_disps_unique[:, 0], return_counts=True
        )
        # when we count intersections we use linearized 2d index because we really
        # need to count intersections in each cell separately

        # but when we need to decide about refinement, due to cartesian nature of grid
        # we will need to consider all cells with a given j index at a time

        # so, let's compute j index for each cell in the unique list
        cell_linear_inds_unique_j = cell_linear_inds_unique % len(grid_y_coords)

        # loop through all j rows that contain intersections
        for ind_j in np.unique(cell_linear_inds_unique_j):
            # we need to refine between two grid lines corresponding to index j
            # if at least one cell with given j contains > 1 intersections

            # get all intersected cells with given j index
            j_selection = cell_linear_inds_unique_j == ind_j
            # and number intersections in each of them
            counts_j = counts[j_selection]

            # find cell with max intersections
            max_count_el = np.argmax(counts_j)
            max_count = counts_j[max_count_el]
            if max_count > 1:
                # get its linear index
                target_cell_linear_ind = cell_linear_inds_unique[j_selection][max_count_el]
                # look up relative positions of intersections in that cells
                target_disps = np.sort(
                    cell_linear_inds_and_disps_unique[
                        cell_linear_inds_and_disps_unique[:, 0] == target_cell_linear_ind, 1
                    ]
                )

                # place a snapping line between any two neighboring intersections (in relative units)
                relative_snap_lines_pos = (
                    0.5 * (target_disps[1:] + target_disps[:-1]) * GAP_MESHING_TOL
                )
                # convert relative positions to absolute ones
                snapping_lines_y += [
                    grid_y_coords[ind_j]
                    + rel_pos * (grid_y_coords[ind_j + 1] - grid_y_coords[ind_j])
                    for rel_pos in relative_snap_lines_pos
                ]

                # compute minimal gap/strip width
                min_gap_width_current = (
                    np.min(target_disps[1:] - target_disps[:-1]) * GAP_MESHING_TOL
                )
                min_gap_width = min(
                    min_gap_width,
                    min_gap_width_current * (grid_y_coords[ind_j + 1] - grid_y_coords[ind_j]),
                )

    return snapping_lines_y, min_gap_width


def _resolve_gaps(
    self: LayerRefinementSpec,
    grid: Grid,
    merged_geos: list[tuple[Any, Shapely]],
    boundary_type: tuple[
        tuple[str | None, str | None],
        tuple[str | None, str | None],
        tuple[str | None, str | None],
    ],
) -> tuple[tuple[CoordinateOptional], float]:
    """
    Detect underresolved gaps and place snapping lines in them. Also return the detected minimal gap width.

    Parameters
    ----------
    grid : Grid
        Grid to resolve gaps on.
    merged_geos : list[tuple[Any, Shapely]]
        Merged geometries on the inplane plane.
    boundary_type : tuple[tuple[str, str], tuple[str, str], tuple[str, str]]
        Type of boundary conditions along each dimension: "pec/pmc", "periodic", or
        None for any other. This is relevant only for gap meshing.

    Returns
    -------
    list[list[CoordinateOptional], float]
        List of snapping lines and the detected minimal gap width.
    """

    # get x and y coordinates of grid lines
    _, tan_dims = Box.pop_axis([0, 1, 2], self.axis)
    x = grid.boundaries.to_list[tan_dims[0]]
    y = grid.boundaries.to_list[tan_dims[1]]

    # restrict to the size of layer spec
    rmin, rmax = self.bounds
    _, rmin = Box.pop_axis(rmin, self.axis)
    _, rmax = Box.pop_axis(rmax, self.axis)

    # extract tangential boundaries
    _, boundaries_tan = Box.pop_axis(boundary_type, self.axis)

    # new coords are expanded by a grid at both min/max
    new_coords = []
    new_boundaries = []
    for coord, cmin, cmax, bdry in zip([x, y], rmin, rmax, boundaries_tan):
        if cmax <= coord[0] or cmin >= coord[-1]:
            return [], inf
        if cmin < coord[0]:
            ind_min = 0
        else:
            ind_min = max(0, np.argmax(coord >= cmin) - 2)

        if cmax > coord[-1]:
            ind_max = len(coord) - 1
        else:
            ind_max = np.argmax(coord >= cmax) + 1

        if ind_min >= ind_max - 1:
            return [], inf

        new_coords.append(coord[ind_min : (ind_max + 1)])
        # ignore boundary conditions if we are not touching them
        new_boundaries.append(
            [
                None if ind_min > 0 else bdry[0],
                None if ind_max < len(coord) - 1 else bdry[1],
            ]
        )

    x, y = new_coords

    # find intersections of pec polygons with grid lines
    # specifically:
    # 0. cells that contain intersections of vertical grid lines
    # 1. relative locations of those intersections along y axis
    # 2. cells that contain intersections of horizontal grid lines
    # 3. relative locations of those intersections along x axis
    v_cells_ij, v_cells_dy, h_cells_ij, h_cells_dx = self._process_slice(
        x, y, merged_geos, new_boundaries
    )

    # generate horizontal snapping lines
    snapping_lines_y, min_gap_width_along_y = self._generate_horizontal_snapping_lines(
        y, v_cells_ij, v_cells_dy
    )
    detected_gap_width = min_gap_width_along_y

    # generate vertical snapping lines
    if len(h_cells_ij) > 0:  # check, otherwise np.roll fails
        snapping_lines_x, min_gap_width_along_x = self._generate_horizontal_snapping_lines(
            x, np.roll(h_cells_ij, shift=1, axis=1), h_cells_dx
        )

        detected_gap_width = min(detected_gap_width, min_gap_width_along_x)
    else:
        snapping_lines_x = []

    # convert snapping lines' coordinates into 3d coordinates
    snapping_lines_y_3d = [
        Box.unpop_axis(Y, (None, None), axis=tan_dims[1]) for Y in snapping_lines_y
    ]
    snapping_lines_x_3d = [
        Box.unpop_axis(X, (None, None), axis=tan_dims[0]) for X in snapping_lines_x
    ]

    return snapping_lines_x_3d + snapping_lines_y_3d, detected_gap_width
