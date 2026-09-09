"""Grid-spec localization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .localization_helpers import (
    _filter_layer_refinement_specs_to_region,
    _filter_override_structures_to_region,
    _filter_snapping_points_to_region,
)

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.geometry.base import Box

    from .model import GridSpec


def _localized_copy(self: GridSpec, region: Box) -> Self:
    """Return a copy with meshing entities localized to ``region``.

    Structures that don't intersect the region are removed or trimmed.
    For axis-specific entities (snapping points, mesh-override ``dl``),
    each axis is filtered independently against ``region.bounds``.

    Parameters
    ----------
    region : :class:`.Box`
        Requested localization region.
    """

    if not self.snapped_grid_used:
        return self

    updates = {}

    override_structures = _filter_override_structures_to_region(
        self.override_structures, region=region
    )
    snapping_points = _filter_snapping_points_to_region(self.snapping_points, region=region)
    layer_refinement_specs = _filter_layer_refinement_specs_to_region(
        self.layer_refinement_specs, region=region
    )

    if override_structures != self.override_structures:
        updates["override_structures"] = override_structures
    if snapping_points != self.snapping_points:
        updates["snapping_points"] = snapping_points
    if layer_refinement_specs != self.layer_refinement_specs:
        updates["layer_refinement_specs"] = layer_refinement_specs

    if not updates:
        return self

    return self.updated_copy(**updates)
