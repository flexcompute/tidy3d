"""Layer-refinement properties helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import LayerRefinementSpec


from .model import GridRefinement


def length_axis(self: LayerRefinementSpec) -> float:
    """Gets the thickness of the layer."""
    return self.size[self.axis]


def center_axis(self: LayerRefinementSpec) -> float:
    """Gets the position of the center of the layer along the layer dimension."""
    return self.center[self.axis]


def _edge_refinement(self: LayerRefinementSpec) -> GridRefinement | None:
    """In-plane edge refinement spec, or ``None`` when edge refinement is off.

    Edge refinement requires ``corner_finder``. ``'mirror_corner'`` resolves to
    ``corner_refinement`` (so the common one-knob path drives both corners and edges); ``None``
    disables edge refinement; an explicit :class:`.GridRefinement` is used as-is.
    """
    if self.corner_finder is None:
        return None
    edge = self.in_plane_edge_refinement
    if edge == "mirror_corner":
        return self.corner_refinement
    return edge


def _inplane_merge_needed(self: LayerRefinementSpec) -> bool:
    """Whether any in-plane feature needs the merged metal geometries.

    The merge requires ``corner_finder``; with it set, the merge is built when at least one
    consumer is active: corner snapping, corner refinement, edge refinement, the feature-size
    ``dl_min`` reduction (concave/convex/mixed resolution), small-geometry resolution
    (``min_steps_per_geometry``), or gap meshing (``gap_meshing_iters``).
    """
    if self.corner_finder is None:
        return False
    return (
        self.corner_snapping
        or self.corner_refinement is not None
        or self._edge_refinement is not None
        or not self.corner_finder._no_min_dl_override
        or self.min_steps_per_geometry is not None
        or self.gap_meshing_iters > 0
    )


def _corner_refinement(
    self: LayerRefinementSpec, grid_size_in_vacuum: float
) -> GridRefinement | None:
    """Corner refinement spec at the finer of the corner and edge grid sizes.

    Returns ``None`` when ``corner_finder`` is off or neither corner nor edge refinement is
    active. ``corner_snapping`` is unaffected; this governs the override grid size only.
    """
    if self.corner_finder is None:
        return None
    corner = self.corner_refinement
    edge = self._edge_refinement
    if corner is None and edge is None:
        return None
    grid_sizes = []
    if corner is not None:
        grid_sizes.append(corner._grid_size(grid_size_in_vacuum))
    if edge is not None:
        grid_sizes.append(edge._grid_size(grid_size_in_vacuum))
    # take the finer grid size; the corner's cell count drives the extent when it is active
    num_cells = corner.num_cells if corner is not None else edge.num_cells
    return GridRefinement(dl=min(grid_sizes), num_cells=num_cells)
