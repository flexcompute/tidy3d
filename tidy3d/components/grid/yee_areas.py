"""Yee-grid integration-width primitives.

This module is the single source of truth for the per-axis integration widths
("diff areas") used in surface integrals, for both colocation conventions:

- :func:`colocated_widths_1d` / :func:`colocated_edges_1d` -- data colocated to
  grid boundaries (one width per boundary sample). Used by
  :meth:`tidy3d.components.data.monitor_data.ElectromagneticFieldData._diff_area`
  (colocated flux, ``dot``, mode normalization).
- :func:`yee_primal_dual_widths_1d` -- data at native Yee-staggered positions
  (one primal and one dual width per cell). Used by
  :meth:`ElectromagneticFieldData._diff_area_at_yee_positions` (non-colocated
  flux, ``dot``/``outer_dot``).

Callers pass the relevant grid boundary array directly so the helpers stay
free of any pydantic / xarray dependency.
"""

from __future__ import annotations

from flex_em.numerical.raw.grid import (
    colocated_edges_1d,
    colocated_widths_1d,
    yee_primal_dual_widths_1d,
)

__all__ = [
    "colocated_edges_1d",
    "colocated_widths_1d",
    "yee_primal_dual_widths_1d",
]
