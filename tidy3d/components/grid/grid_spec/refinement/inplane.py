"""Layer-refinement inplane helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely

from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.grid_spec.constants import INPLANE_OVERRIDE_UNION_MAX_ITERS
from tidy3d.components.structure import MeshOverrideStructure
from tidy3d.constants import fp_eps, inf

if TYPE_CHECKING:
    from tidy3d.components.grid.grid_spec.constants import CornersAndConvexity
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import ArrayFloat1D, ArrayFloat2D, CoordinateOptional, Shapely

    from .model import LayerRefinementSpec


def _inplane_inside(self: LayerRefinementSpec, box: Box, point: ArrayFloat2D) -> bool:
    """On the inplane cross section, whether the point is inside the box.

    Parameters
    ----------
    box : Box
        Box to check point containment.
    point : ArrayFloat2D
        Point position on inplane plane.

    Returns
    -------
    bool
        ``True`` for every point that is inside the layer.
    """

    point_3d = self.unpop_axis(ax_coord=self.center[self.axis], plane_coords=point, axis=self.axis)
    return box.inside(point_3d[0], point_3d[1], point_3d[2])


def _layer_box(self: LayerRefinementSpec, sim_bounds: tuple, boundary_type: tuple) -> Box:
    """Layer box with size slightly adjusted to boundary conditions.

    Parameters
    ----------
    sim_bounds : tuple
        Bounds of the simulation domain excluding the PML regions, formatted as
        ``(mins, maxs)`` where each is a 3-tuple of coordinates.
    boundary_type : tuple
        Boundary type of the simulation domain.

    Returns
    -------
    Box
        Layer box with size slightly adjusted to boundary conditions.
    """
    layer_geometry = Box(center=self.center, size=self.size)

    # start from the actual bounds of the layer geometry
    layer_min, layer_max = (list(b) for b in layer_geometry.bounds)
    sim_min, sim_max = sim_bounds

    for axis2d_ind in range(2):
        axis_ind = (self.axis + axis2d_ind + 1) % 3
        bc_minus, bc_plus = boundary_type[axis_ind]
        # shrink away from lower periodic boundary if the layer touches it
        if bc_minus == "periodic" and layer_min[axis_ind] <= sim_min[axis_ind] + fp_eps:
            layer_min[axis_ind] = sim_min[axis_ind] + fp_eps

        # shrink away from upper periodic boundary if the layer touches it
        if bc_plus == "periodic" and layer_max[axis_ind] >= sim_max[axis_ind] - fp_eps:
            layer_max[axis_ind] = sim_max[axis_ind] - fp_eps

    new_box = Box.from_bounds(rmin=layer_min, rmax=layer_max)
    return layer_geometry.updated_copy(center=new_box.center, size=new_box.size)


def _merged_geos(
    self: LayerRefinementSpec,
    structure_list: list[Structure],
    sim_bounds: tuple,
    boundary_type: tuple,
) -> list[tuple[Any, Shapely]]:
    """Merged geometries on the inplane plane.

    Parameters
    ----------
    structure_list : list[Structure]
        List of structures present in simulation.
    sim_bounds : tuple
        Bounds of the simulation domain excluding the PML regions.
    boundary_type : tuple
        Boundary type of the simulation domain.

    Returns
    -------
    list[tuple[Any, Shapely]]
        Merged geometries on the inplane plane.
    """
    if not self._inplane_merge_needed:
        return []

    layer_geometry = self._layer_box(sim_bounds, boundary_type)
    # filter structures outside the layer
    structures_intersect = structure_list
    if self._is_inplane_bounded(layer_geometry):
        structures_intersect = [s for s in structure_list if layer_geometry.intersects(s.geometry)]
    merged_geos = self.corner_finder._merged_pec_on_plane(
        normal_axis=self.axis,
        coord=self.center_axis,
        structure_list=structures_intersect,
        center=layer_geometry.center,
        size=layer_geometry.size,
        interior_disjoint_geometries=self.interior_disjoint_geometries,
        keep_metal_only=(
            self.interior_disjoint_geometries and self.corner_finder.medium == "metal"
        ),
    )
    return merged_geos


def _corners_and_convexity_2d(
    self: LayerRefinementSpec,
    merged_geos: list[tuple[Any, Shapely]],
    structure_list: list[Structure],
    ravel: bool,
    sim_bounds: tuple,
    boundary_type: tuple,
) -> tuple[list[ArrayFloat2D], list[ArrayFloat1D]]:
    """Raw inplane corners and their convexity.

    Parameters
    ----------
    merged_geos : list[tuple[Any, Shapely]]
        Merged geometries on the inplane plane.
    structure_list : list[Structure]
        List of structures present in simulation.
    ravel : bool
        Whether to put the resulting corners in a single list or per polygon.
    sim_bounds : tuple
        Bounds of the simulation domain excluding the PML regions.
    boundary_type : tuple
        Boundary type of the simulation domain.

    Returns
    -------
    tuple[list[ArrayFloat2D], list[ArrayFloat1D]]
        Raw inplane corners and their convexity.
    """
    if self.corner_finder is None:
        return [], []

    inplane_points, convexity = self.corner_finder._corners_and_convexity(
        merged_geos=merged_geos,
        ravel=ravel,
    )

    # filter corners outside the inplane bounds
    layer_geometry = self._layer_box(sim_bounds, boundary_type)
    layer_geometry_slightly_enlarged = layer_geometry.slightly_enlarged_copy()
    if self._is_inplane_bounded(layer_geometry) and len(inplane_points) > 0:
        # flatten temporary list of arrays for faster processing
        if not ravel:
            split_inds = np.cumsum([len(pts) for pts in inplane_points])[:-1]
            inplane_points = np.concatenate(inplane_points)
            convexity = np.concatenate(convexity)
        inds = [
            self._inplane_inside(layer_geometry_slightly_enlarged, point)
            for point in inplane_points
        ]
        inplane_points = inplane_points[inds]
        convexity = convexity[inds]
        if not ravel:
            inplane_points = np.split(inplane_points, split_inds)
            convexity = np.split(convexity, split_inds)

    return inplane_points, convexity


def _dl_min_from_smallest_feature(
    self: LayerRefinementSpec,
    structure_list: list[Structure],
    sim_bounds: tuple,
    boundary_type: tuple,
    cached_merged_geos: list[tuple[Any, Shapely]] | None = None,
    cached_corners_and_convexity: CornersAndConvexity | None = None,
) -> float:
    """Calculate `dl_min` suggestion based on smallest feature size."""

    if cached_corners_and_convexity is None:
        if cached_merged_geos is None:
            merged_geos = self._merged_geos(structure_list, sim_bounds, boundary_type)
        else:
            merged_geos = cached_merged_geos
        inplane_points, convexity = self._corners_and_convexity_2d(
            merged_geos=merged_geos,
            structure_list=structure_list,
            ravel=False,
            sim_bounds=sim_bounds,
            boundary_type=boundary_type,
        )
    else:
        inplane_points, convexity = cached_corners_and_convexity

    dl_min = inf

    if self.corner_finder is None or self.corner_finder._no_min_dl_override:
        return dl_min

    finder = self.corner_finder

    for points, conv in zip(inplane_points, convexity):
        conv_nei = np.roll(conv, -1)
        lengths = np.linalg.norm(points - np.roll(points, axis=0, shift=-1), axis=-1)

        if finder.convex_resolution is not None:
            convex_features = np.logical_and(conv, conv_nei)
            if np.any(convex_features):
                min_convex_size = np.min(lengths[convex_features])
                dl_min = min(dl_min, min_convex_size / finder.convex_resolution)

        if finder.concave_resolution is not None:
            concave_features = np.logical_not(np.logical_or(conv, conv_nei))
            if np.any(concave_features):
                min_concave_size = np.min(lengths[concave_features])
                dl_min = min(dl_min, min_concave_size / finder.concave_resolution)

        if finder.mixed_resolution is not None:
            mixed_features = np.logical_xor(conv, conv_nei)
            if np.any(mixed_features):
                min_mixed_size = np.min(lengths[mixed_features])
                dl_min = min(dl_min, min_mixed_size / finder.mixed_resolution)

    return dl_min


def _corners(
    self: LayerRefinementSpec,
    structure_list: list[Structure],
    sim_bounds: tuple,
    boundary_type: tuple,
    cached_corners_and_convexity: CornersAndConvexity | None = None,
    cached_merged_geos: list[tuple[Any, Shapely]] | None = None,
) -> list[CoordinateOptional]:
    """Inplane corners in 3D coordinate."""
    if self.corner_finder is None:
        return []
    if cached_corners_and_convexity is None:
        if cached_merged_geos is None:
            merged_geos = self._merged_geos(structure_list, sim_bounds, boundary_type)
        else:
            merged_geos = cached_merged_geos
        inplane_points, _ = self._corners_and_convexity_2d(
            merged_geos=merged_geos,
            structure_list=structure_list,
            ravel=True,
            sim_bounds=sim_bounds,
            boundary_type=boundary_type,
        )
    else:
        inplane_points, convexity = cached_corners_and_convexity
        inplane_points, _ = self.corner_finder._ravel_corners_and_convexity(
            ravel=True, corner_list=inplane_points, convexity_list=convexity
        )

    # convert 2d points to 3d
    return [
        Box.unpop_axis(ax_coord=None, plane_coords=point, axis=self.axis)
        for point in inplane_points
    ]


def _snapping_points_along_axis(self: LayerRefinementSpec) -> list[CoordinateOptional]:
    """Snapping points for layer bounds."""

    if self.bounds_snapping is None:
        return []
    if self.bounds_snapping == "center":
        return [
            self._unpop_axis(ax_coord=self.center_axis, plane_coord=None),
        ]
    if self.bounds_snapping == "lower":
        return [
            self._unpop_axis(ax_coord=self.bounds[0][self.axis], plane_coord=None),
        ]
    if self.bounds_snapping == "upper":
        return [
            self._unpop_axis(ax_coord=self.bounds[1][self.axis], plane_coord=None),
        ]

    # the rest is for "bounds"
    return [
        self._unpop_axis(ax_coord=self.bounds[index][self.axis], plane_coord=None)
        for index in range(1 + (self.length_axis > 0))
    ]


def _override_structures_inplane(
    self: LayerRefinementSpec,
    structure_list: list[Structure],
    grid_size_in_vacuum: float,
    sim_bounds: tuple,
    boundary_type: tuple,
    cached_corners_and_convexity: CornersAndConvexity | None = None,
    cached_merged_geos: list[tuple[Any, Shapely]] | None = None,
) -> list[MeshOverrideStructure]:
    """Inplane mesh override structures.

    Collects candidate overrides from the two pre-mesh in-plane sources — corner refinement
    (coupled with edge) and edge refinement — then unions overlapping candidates that share a
    grid size into one bounding box each. Candidates of different grid sizes are kept
    separate; because every override is ``shadow=False`` the mesher resolves any overlap to the
    smaller grid size per axis. Small-geometry resolution is a separate post-mesh measurement
    pass and never enters this union.
    """
    # the merge is shared by corner detection and edge refinement, so compute it at most once
    if cached_merged_geos is None:
        merged_geos = self._merged_geos(structure_list, sim_bounds, boundary_type)
    else:
        merged_geos = cached_merged_geos

    candidates = []

    # corner refinement, at the finer of the corner/edge grid size
    corner_refinement = self._corner_refinement(grid_size_in_vacuum)
    if corner_refinement is not None:
        candidates += [
            corner_refinement.override_structure(
                corner, (0, 0, 0), grid_size_in_vacuum, self.refinement_inside_sim_only
            )
            for corner in self._corners(
                structure_list,
                sim_bounds=sim_bounds,
                boundary_type=boundary_type,
                cached_corners_and_convexity=cached_corners_and_convexity,
                cached_merged_geos=merged_geos,
            )
        ]

    # edge refinement along axis-unaligned in-plane edges
    if self._edge_refinement is not None:
        candidates += self._override_structures_edges(grid_size_in_vacuum, merged_geos)

    return self._union_inplane_overrides(candidates)


def _union_inplane_overrides(
    self: LayerRefinementSpec, candidates: list[MeshOverrideStructure]
) -> list[MeshOverrideStructure]:
    """Union overlapping in-plane overrides that share a grid size into disjoint bboxes.

    Candidates are grouped by their (per-axis) target grid size; within a group, overlapping
    boxes are merged (see ``_union_same_dl_overrides``) into pairwise non-overlapping
    bounding-box overrides. Groups of different grid sizes are not merged.
    """
    if not candidates:
        return []
    # bucket candidates by their per-axis target grid size; only same-size boxes may merge
    overrides_by_dl = defaultdict(list)
    for candidate in candidates:
        overrides_by_dl[tuple(candidate._dl)].append(candidate)
    unioned = []
    for dl, dl_overrides in overrides_by_dl.items():
        unioned += self._union_same_dl_overrides(dl_overrides, dl)
    return unioned


def _union_same_dl_overrides(
    self: LayerRefinementSpec, overrides: list[MeshOverrideStructure], dl: tuple
) -> list[MeshOverrideStructure]:
    """Merge overlapping same-grid-size boxes into non-overlapping bounding-box overrides.

    Collapsing a connected component to its bounding box can itself overlap a box from another
    component that none of the component's members touched directly: e.g. a long diagonal edge
    run's bbox spans the whole patch and swallows the right-angle corner boxes. A single
    connected-components pass would leave those bounding boxes overlapping but separate, so
    iterate connected-components-then-bbox until the bounding boxes are pairwise disjoint. Each
    pass that merges anything strictly reduces the box count, so this terminates; the iteration
    is also capped at ``INPLANE_OVERRIDE_UNION_MAX_ITERS`` as a safety bound.
    """
    boxes_2d = [self._inplane_footprint_box(override.geometry) for override in overrides]
    for _ in range(INPLANE_OVERRIDE_UNION_MAX_ITERS):
        components = self._connected_components(boxes_2d)
        # nothing merged this pass (every box is its own component) => no overlaps remain
        if len(components) == len(boxes_2d):
            break
        # box_bounds[i] = (umin, vmin, umax, vmax) of box i
        box_bounds = np.array([box.bounds for box in boxes_2d])
        # each connected component collapses to the bounding box enclosing all its boxes
        merged = []
        for component in components:
            component_bounds = box_bounds[component]
            umin, vmin = component_bounds[:, 0].min(), component_bounds[:, 1].min()
            umax, vmax = component_bounds[:, 2].max(), component_bounds[:, 3].max()
            merged.append(shapely.box(umin, vmin, umax, vmax))
        boxes_2d = merged
    return [self._inplane_override_from_bbox(*box.bounds, dl) for box in boxes_2d]


def _inplane_footprint_box(self: LayerRefinementSpec, geometry: Box) -> Shapely:
    """In-plane (axis-popped) bounding-box footprint of an override geometry as a shapely box."""
    rmin, rmax = geometry.bounds
    _, (umin, vmin) = self.pop_axis(rmin, self.axis)
    _, (umax, vmax) = self.pop_axis(rmax, self.axis)
    return shapely.box(umin, vmin, umax, vmax)


def _connected_components(boxes: list[Shapely]) -> list[list[int]]:
    """Indices of connected components of touching/overlapping boxes via an RTree index."""
    # Union-find over box indices. ``component_root[i]`` links box ``i`` toward the
    # representative ("root") box of its component; boxes that share a root are one component.
    num_boxes = len(boxes)
    component_root = list(range(num_boxes))

    def find_root(box_ind: int) -> int:
        # Walk the links to the component's root, halving the path so future lookups are cheap.
        while component_root[box_ind] != box_ind:
            component_root[box_ind] = component_root[component_root[box_ind]]
            box_ind = component_root[box_ind]
        return box_ind

    def merge_components(box_a: int, box_b: int) -> None:
        # Point one component's root at the other's so both boxes end up under one root.
        root_a, root_b = find_root(box_a), find_root(box_b)
        if root_a != root_b:
            component_root[root_a] = root_b

    # The STRtree turns the all-pairs intersection test into a near-linear spatial query as the
    # box count grows; merge the two boxes of every intersecting pair into one component.
    tree = shapely.STRtree(boxes)
    query_inds, tree_inds = tree.query(boxes, predicate="intersects")
    for query_ind, tree_ind in zip(query_inds, tree_inds):
        merge_components(int(query_ind), int(tree_ind))

    # Bucket each box under its resolved root; every bucket is one connected component.
    components = defaultdict(list)
    for box_ind in range(num_boxes):
        components[find_root(box_ind)].append(box_ind)
    return list(components.values())


def _inplane_override_from_bbox(
    self: LayerRefinementSpec, umin: float, vmin: float, umax: float, vmax: float, dl: tuple
) -> MeshOverrideStructure:
    """A ``shadow=False`` in-plane override spanning a 2D bounding box at grid size ``dl``."""
    center = self.unpop_axis(self.center_axis, ((umin + umax) / 2, (vmin + vmax) / 2), self.axis)
    size = self.unpop_axis(inf, (umax - umin, vmax - vmin), self.axis)
    return MeshOverrideStructure(
        geometry=Box(center=center, size=size),
        dl=list(dl),
        shadow=False,
        drop_outside_sim=self.refinement_inside_sim_only,
        priority=-1,
    )


def _override_structures_along_axis(
    self: LayerRefinementSpec, grid_size_in_vacuum: float
) -> list[MeshOverrideStructure]:
    """Mesh override structures for refining mesh along layer axis dimension."""

    override_structures = []
    dl = inf
    # minimal number of step sizes along layer axis
    if self.min_steps_along_axis is not None and self.length_axis > 0:
        dl = self.length_axis / self.min_steps_along_axis
        override_structures.append(
            MeshOverrideStructure(
                geometry=Box(
                    center=self._unpop_axis(ax_coord=self.center_axis, plane_coord=0),
                    size=self._unpop_axis(ax_coord=self.length_axis, plane_coord=inf),
                ),
                dl=self._unpop_axis(ax_coord=dl, plane_coord=None),
                shadow=False,
                drop_outside_sim=self.refinement_inside_sim_only,
                priority=-1,
            )
        )

    # refinement at upper and lower bounds
    if self.bounds_refinement is not None:
        refinement_structures = [
            self.bounds_refinement.override_structure(
                self._unpop_axis(ax_coord=self.bounds[index][self.axis], plane_coord=None),
                (0, 0, 0),
                grid_size_in_vacuum,
                drop_outside_sim=self.refinement_inside_sim_only,
            )
            for index in range(1 + (self.length_axis > 0))
        ]
        # combine them to one if the two overlap
        if len(refinement_structures) == 2 and refinement_structures[0].geometry.intersects(
            refinement_structures[1].geometry
        ):
            rmin, rmax = Box.bounds_union(
                refinement_structures[0].geometry.bounds,
                refinement_structures[1].geometry.bounds,
            )
            combined_structure = MeshOverrideStructure(
                geometry=Box.from_bounds(rmin=rmin, rmax=rmax),
                dl=refinement_structures[0]._dl,
                shadow=False,
                drop_outside_sim=self.refinement_inside_sim_only,
            )
            refinement_structures = [
                combined_structure,
            ]

        # drop if the grid size is no greater than the one from "min_steps_along_axis"
        if refinement_structures[0]._dl[self.axis] <= dl:
            override_structures += refinement_structures
    return override_structures
