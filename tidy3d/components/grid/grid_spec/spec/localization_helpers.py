"""Grid-spec localization helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.bound_ops import bounds_intersection
from tidy3d.components.geometry.utils import filter_intersecting_geometries
from tidy3d.components.structure import MeshOverrideStructure

if TYPE_CHECKING:
    from tidy3d.components.grid.grid_spec.refinement import LayerRefinementSpec
    from tidy3d.components.types import CoordinateOptional

# ---------------------------------------------------------------------------
# Grid-spec localization helpers
#
# These helpers are used to speed up ModeSimulation instantiation, for
# regions much smaller than the full simulation. The grid specification must be
# localized to that region so the mode solver doesn't waste effort on mesh
# entities that are far from the port.
#
# Three kinds of mesh entity are filtered independently:
#   1. Override structures — geometry-bearing mesh hints.  Non-MeshOverride
#      structures are pruned via recursive geometry filtering (see
#      geometry/utils.py).  MeshOverrideStructures use per-axis interval
#      checks: if the structure's extent along an axis doesn't overlap the
#      region, that axis's ``dl`` is set to None (disabled).
#   2. Layer refinement specs — bounding-box objects clipped to the region.
#   3. Snapping points — (x, y, z) coordinates where individual out-of-
#      region coordinates are set to None.  The point is kept as long as at
#      least one coordinate remains.
# ---------------------------------------------------------------------------


def _filter_override_structures_to_region(
    override_structures: tuple,
    region: Box,
) -> tuple:
    """Filter override structures to the requested region, preserving axis-aware mesh hints."""

    region_bounds = region.bounds
    # Batch-filter the geometries of non-MeshOverride structures using the
    # recursive geometry filter.  MeshOverrideStructures are handled below
    # with per-axis interval checks because their dl hints are axis-specific.
    filtered_geometries = iter(
        filter_intersecting_geometries(
            [
                struct.geometry
                for struct in override_structures
                if not isinstance(struct, MeshOverrideStructure)
            ],
            region,
        )
    )
    filtered = []

    # Iterate the original tuple so override priority order is preserved;
    # filtered_geometries is consumed in lock-step for non-MeshOverride entries.
    for struct in override_structures:
        if not isinstance(struct, MeshOverrideStructure):
            geometry = next(filtered_geometries)
            if geometry is not None:
                filtered.append(struct.updated_copy(geometry=geometry, deep=False))
            continue

        # MeshOverrideStructure: disable the override along axes where the structure
        # doesn't overlap the region, rather than discarding it entirely. Freeze the
        # resolved grid size into 'dl' (clearing 'min_steps_per_size') first so disabling
        # an axis is a simple 'dl[axis] = None'.
        bounds = struct.geometry.bounds
        frozen = struct._freeze_dl()
        dl = list(frozen.dl)

        for axis in range(3):
            if dl[axis] is None:
                continue
            if bounds[1][axis] < region_bounds[0][axis] or bounds[0][axis] > region_bounds[1][axis]:
                dl[axis] = None

        if any(val is not None for val in dl):
            filtered.append(frozen.updated_copy(dl=tuple(dl)))

    return tuple(filtered)


def _filter_layer_refinement_specs_to_region(
    layer_specs: tuple[LayerRefinementSpec, ...],
    region: Box,
) -> tuple[LayerRefinementSpec, ...]:
    """Filter layer refinement specs to the requested region."""

    filtered = []
    for spec in layer_specs:
        if not region.intersects(spec):
            continue
        # Clip the spec's bounding box to the region so it only covers the
        # overlapping portion.
        clipped = Box.from_bounds(*bounds_intersection(spec.bounds, region.bounds))
        filtered.append(spec.updated_copy(center=clipped.center, size=clipped.size))
    return tuple(filtered)


def _filter_snapping_points_to_region(
    snapping_points: tuple[CoordinateOptional, ...],
    region: Box,
) -> tuple[CoordinateOptional, ...]:
    """Filter snapping points to coordinates relevant to the requested region.

    Each coordinate is checked independently: out-of-region coordinates are
    set to None.  The point is kept as long as at least one coordinate
    survives.
    """

    region_bounds = region.bounds
    filtered = []
    for point in snapping_points:
        filtered_point = list(point)
        for axis in range(3):
            coord = point[axis]
            if coord is not None and not (
                region_bounds[0][axis] <= coord <= region_bounds[1][axis]
            ):
                filtered_point[axis] = None

        if any(coord is not None for coord in filtered_point):
            filtered.append(tuple(filtered_point))

    return tuple(filtered)
