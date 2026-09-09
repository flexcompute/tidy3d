"""Layer-refinement edges helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.grid_spec.constants import GAP_MESHING_TOL
from tidy3d.components.structure import MeshOverrideStructure
from tidy3d.constants import inf

if TYPE_CHECKING:
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.grid.grid_spec.constants import CornersAndConvexity
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import ArrayFloat2D, CoordinateOptional, Shapely

    from .model import LayerRefinementSpec


def _polygon_rings_2d(polygon: Shapely) -> list[ArrayFloat2D]:
    """Open in-plane rings (exterior and any interiors) of a shapely polygon."""
    rings = [np.asarray(polygon.exterior.coords, dtype=float)[:-1]]
    for interior in polygon.interiors:
        rings.append(np.asarray(interior.coords, dtype=float)[:-1])
    return rings


def _edge_run_override_structure(
    self: LayerRefinementSpec, run_vertices: ArrayFloat2D, grid_size_in_vacuum: float
) -> MeshOverrideStructure:
    """One in-plane override over the bounding box of an axis-unaligned run, grown by the edge
    refinement margin."""
    mins = run_vertices.min(axis=0)
    maxs = run_vertices.max(axis=0)
    center = self.unpop_axis(None, tuple((mins + maxs) / 2), self.axis)
    size = self.unpop_axis(0.0, tuple(maxs - mins), self.axis)
    return self._edge_refinement.override_structure(
        center, size, grid_size_in_vacuum, self.refinement_inside_sim_only
    )


def _override_structures_edges(
    self: LayerRefinementSpec, grid_size_in_vacuum: float, merged_geos: list[tuple[Any, Shapely]]
) -> list[MeshOverrideStructure]:
    """Override structures refining the mesh along axis-unaligned in-plane edges.

    Edge geometry comes from the merged metal polygons; axis-alignment is classified per
    segment from their vertices, so this is independent of the corner-detection pass.
    """
    if self._edge_refinement is None:
        return []
    override_structures = []
    for polygon in self.corner_finder._polygons_from_merged_geos(merged_geos):
        for ring in self._polygon_rings_2d(polygon):
            for run_vertices in self.corner_finder._axis_unaligned_runs(ring):
                override_structures.append(
                    self._edge_run_override_structure(run_vertices, grid_size_in_vacuum)
                )
    return override_structures


def _small_geometry_measurement_overrides(
    self: LayerRefinementSpec, grid: Grid, merged_geos: list[tuple[Any, Shapely]]
) -> tuple[list[MeshOverrideStructure], float]:
    """Overrides resolving small disjoint geometries, measured on the constructed grid.

    For each disjoint geometry and in-plane axis, count the cells of ``grid`` fully contained
    in the geometry's bounding box (the grid boundaries inside the box, minus one); where that
    count is below ``min_steps_per_geometry`` the axis is refined to
    ``extent / min_steps_per_geometry``. Unlike corner/edge refinement this runs *after* the
    grid is built, so only features the fully-refined grid leaves under-resolved are refined —
    a large geometry already spans enough cells and is skipped, with no explicit size threshold.

    Returns ``(overrides, dl_min_from_small_geometry)`` where the second value is the smallest
    emitted target grid size (``inf`` if nothing is emitted), used to lower the ``dl_min`` floor
    before the final rebuild. Each override is the geometry bounding box, refining only the
    failing in-plane axes (the passing in-plane axis and the normal axis are left ``None``),
    emitted ``shadow=False`` with ``priority=-1`` so it stacks per axis with any overlapping
    corner/edge override.
    """
    if self.min_steps_per_geometry is None or self.corner_finder is None:
        return [], inf
    _, tan_dims = Box.pop_axis((0, 1, 2), self.axis)
    grid_boundaries = grid.boundaries.to_list
    override_structures = []
    dl_min = inf
    for polygon in self.corner_finder._polygons_from_merged_geos(merged_geos):
        umin, vmin, umax, vmax = polygon.bounds
        bbox_min, bbox_max = (umin, vmin), (umax, vmax)
        # per in-plane axis, the target grid size, or None where the geometry is already resolved
        dl_2d = [None, None]
        for axis2d in range(2):
            extent = bbox_max[axis2d] - bbox_min[axis2d]
            # skip near-zero extents (e.g. slivers from Shapely boolean ops): their
            # vanishing dl would otherwise propagate to dl_min and trigger runaway refinement
            if extent < GAP_MESHING_TOL:
                continue
            # cells fully inside the bbox = grid boundaries lying within it, minus one
            coords = np.asarray(grid_boundaries[tan_dims[axis2d]])
            num_cells = (
                int(np.count_nonzero((coords >= bbox_min[axis2d]) & (coords <= bbox_max[axis2d])))
                - 1
            )
            # under-resolved axis: shrink the grid size to fit min_steps_per_geometry cells
            if num_cells < self.min_steps_per_geometry:
                dl_axis = extent / self.min_steps_per_geometry
                dl_2d[axis2d] = dl_axis
                dl_min = min(dl_min, dl_axis)
        if any(dl is not None for dl in dl_2d):
            center = self.unpop_axis(
                self.center_axis, ((umin + umax) / 2, (vmin + vmax) / 2), self.axis
            )
            size = self.unpop_axis(inf, (umax - umin, vmax - vmin), self.axis)
            dl = self.unpop_axis(None, tuple(dl_2d), self.axis)
            override_structures.append(
                MeshOverrideStructure(
                    geometry=Box(center=center, size=size),
                    dl=dl,
                    shadow=False,
                    drop_outside_sim=self.refinement_inside_sim_only,
                    priority=-1,
                )
            )
    return override_structures, dl_min


def _is_inplane_bounded(self: LayerRefinementSpec, geometry: Box) -> bool:
    """Whether the geometry is bounded in at least one of the inplane dimensions."""
    return np.isfinite(geometry.size[(self.axis + 1) % 3]) or np.isfinite(
        geometry.size[(self.axis + 2) % 3]
    )


def _unpop_axis(self: LayerRefinementSpec, ax_coord: float, plane_coord: Any) -> CoordinateOptional:
    """Combine coordinate along axis with identical coordinates on the plane tangential to the axis.

    Parameters
    ----------
    ax_coord : float
        Value self.axis direction.
    plane_coord : Any
        Values along planar directions that are identical.

    Returns
    -------
    CoordinateOptional
        The three values in the xyz coordinate system.
    """
    return self.unpop_axis(ax_coord, [plane_coord, plane_coord], self.axis)


def suggested_dl_min(
    self: LayerRefinementSpec,
    grid_size_in_vacuum: float,
    structures: list[Structure],
    sim_bounds: tuple,
    boundary_type: tuple,
    cached_merged_geos: list[tuple[Any, Shapely]] | None = None,
    cached_corners_and_convexity: CornersAndConvexity | None = None,
) -> float:
    """Suggested lower bound of grid step size for this layer.

    Parameters
    ----------
    grid_size_in_vacuum : float
        Grid step size in vaccum.
    structures : list[Structure]
        List of structures present in simulation.
    sim_bounds : tuple
        Bounds of the simulation domain excluding the PML regions, formatted as
        ``(mins, maxs)`` where each is a 3-tuple of coordinates.
    boundary_type : tuple
        Boundary type of the simulation domain.
    cached_merged_geos : Optional[list[tuple[Any, Shapely]]]
        Cached merged geometries. If None, will be computed.
    cached_corners_and_convexity : Optional[tuple[list[ArrayFloat2D], list[ArrayFloat1D]]]
        Cached corners and convexity data. If None, will be computed.

    Returns
    -------
    float
        Suggested lower bound of grid size to resolve most snapping points and
        mesh refinement structures.
    """
    dl_min = inf

    # axis dimension
    if self.length_axis > 0:
        # bounds snapping
        if self.bounds_snapping == "bounds":
            dl_min = min(dl_min, self.length_axis)
        # from min_steps along bounds
        if self.min_steps_along_axis is not None:
            dl_min = min(dl_min, self.length_axis / self.min_steps_along_axis)
        # refinement
        if self.bounds_refinement is not None:
            dl_min = min(dl_min, self.bounds_refinement._grid_size(grid_size_in_vacuum))

    # inplane dimension: corner refinement
    if self.corner_finder is not None and self.corner_refinement is not None:
        dl_min = min(dl_min, self.corner_refinement._grid_size(grid_size_in_vacuum))
    # inplane dimension: edge refinement (independent of corner detection). The min over the
    # corner and edge terms already reflects the upgraded effective corner grid size.
    edge = self._edge_refinement
    if edge is not None:
        dl_min = min(dl_min, edge._grid_size(grid_size_in_vacuum))

    # small-geometry resolution does not contribute to this static bound: its target grid size
    # is known only after the grid is measured, so it lowers ``dl_min`` dynamically in the
    # post-mesh measurement pass (see ``_small_geometry_measurement_overrides``).

    # min feature size
    if self.corner_finder is not None and not self.corner_finder._no_min_dl_override:
        dl_suggested = self._dl_min_from_smallest_feature(
            structures,
            sim_bounds=sim_bounds,
            boundary_type=boundary_type,
            cached_merged_geos=cached_merged_geos,
            cached_corners_and_convexity=cached_corners_and_convexity,
        )
        dl_min = min(dl_min, dl_suggested)

    return dl_min


def generate_snapping_points(
    self: LayerRefinementSpec,
    structure_list: list[Structure],
    sim_bounds: tuple,
    boundary_type: tuple,
    cached_corners_and_convexity: CornersAndConvexity | None = None,
    cached_merged_geos: list[tuple[Any, Shapely]] | None = None,
) -> list[CoordinateOptional]:
    """generate snapping points for mesh refinement."""
    snapping_points = self._snapping_points_along_axis
    if self.corner_snapping:
        snapping_points += self._corners(
            structure_list,
            sim_bounds=sim_bounds,
            boundary_type=boundary_type,
            cached_corners_and_convexity=cached_corners_and_convexity,
            cached_merged_geos=cached_merged_geos,
        )
    return snapping_points


def generate_override_structures(
    self: LayerRefinementSpec,
    grid_size_in_vacuum: float,
    structure_list: list[Structure],
    sim_bounds: tuple,
    boundary_type: tuple,
    cached_corners_and_convexity: CornersAndConvexity | None = None,
    cached_merged_geos: list[tuple[Any, Shapely]] | None = None,
) -> list[MeshOverrideStructure]:
    """Generate mesh override structures for mesh refinement."""
    return self._override_structures_along_axis(
        grid_size_in_vacuum
    ) + self._override_structures_inplane(
        structure_list,
        grid_size_in_vacuum,
        sim_bounds,
        boundary_type,
        cached_corners_and_convexity,
        cached_merged_geos,
    )
