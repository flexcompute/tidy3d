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

import numpy as np


def _clipped_widths(coords: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Consecutive widths of ``coords`` clipped to ``[lo, hi]`` -- the one primitive every
    integration-width convention in this module reduces to. Cells fully outside the clip
    range collapse to zero width; cells straddling an edge keep only the covered part."""
    clipped = np.clip(coords, lo, hi)
    return clipped[1:] - clipped[:-1]


def colocated_edges_1d(
    sample_boundaries: np.ndarray,
    mnt_min: float = -np.inf,
    mnt_max: float = np.inf,
) -> np.ndarray:
    """Integration-cell edges around boundary-colocated samples, clipped to monitor bounds.

    The cell around each sample extends to the midpoints towards its neighbors; the first
    and last cells end at the outermost samples themselves, then everything is clipped to
    ``[mnt_min, mnt_max]``. Together with sample values this realizes the trapezoidal rule
    with the edge values implicitly interpolated to the exact monitor start/end, provided
    the integrand is zero outside the monitor geometry.

    Parameters
    ----------
    sample_boundaries : np.ndarray
        Sorted-ascending colocation positions (grid boundaries); at least two entries.
    mnt_min, mnt_max : float
        Monitor bounds along this axis; infinite values disable clipping on that side.

    Returns
    -------
    np.ndarray
        Edge coordinates, length ``len(sample_boundaries) + 1``.
    """
    sample_boundaries = np.asarray(sample_boundaries)
    centers = (sample_boundaries[1:] + sample_boundaries[:-1]) / 2
    edges = np.concatenate([[sample_boundaries[0]], centers, [sample_boundaries[-1]]])
    return np.clip(edges, mnt_min, mnt_max)


def colocated_widths_1d(
    sample_boundaries: np.ndarray,
    mnt_min: float = -np.inf,
    mnt_max: float = np.inf,
) -> np.ndarray:
    """Per-sample integration widths for boundary-colocated data, clipped to monitor bounds.

    Cannot over-count past the grid: the edges are built from the sample positions
    themselves, so the integration never extends beyond the outermost sample.

    Parameters
    ----------
    sample_boundaries : np.ndarray
        Sorted-ascending colocation positions (grid boundaries).
    mnt_min, mnt_max : float
        Monitor bounds along this axis; infinite values disable clipping on that side.

    Returns
    -------
    np.ndarray
        One width per sample (length ``len(sample_boundaries)``). A single-sample axis
        (zero-sized dimension) collapses to ``np.array([1.0])`` so flux-like quantities
        come out in units of ``W/µm`` along it.
    """
    sample_boundaries = np.asarray(sample_boundaries)
    if sample_boundaries.size <= 1:
        return np.array([1.0])
    edges = colocated_edges_1d(sample_boundaries, mnt_min, mnt_max)
    return edges[1:] - edges[:-1]


def yee_primal_dual_widths_1d(
    boundaries: np.ndarray,
    mnt_min: float = -np.inf,
    mnt_max: float = np.inf,
    *,
    valid_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell primal and dual widths along one axis, clipped to monitor bounds.

    Parameters
    ----------
    boundaries : np.ndarray
        Full grid boundary array along this axis. Length ``num_cells + 1``;
        cell ``i`` extends from ``boundaries[i]`` to ``boundaries[i+1]``.
    mnt_min, mnt_max : float
        Monitor bounds along this axis; infinite values disable clipping on
        that side, so omitting both integrates the full valid extent (the
        convention used by ``dot``/``outer_dot`` against another dataset,
        whose normalization covers the full data extent).
    valid_bounds : tuple[float, float], optional
        Extent of physically valid data, clamping the monitor bounds; defaults
        to the outermost grid boundaries. ``boundaries`` typically comes from
        a grid padded by one interpolation cell on each side
        (``discretize_monitor``); flux callers MUST pass the halo-free extent
        here, otherwise a monitor reaching or exceeding the simulation domain
        integrates the padding cells and over-counts -- under periodic/Bloch
        boundaries that skews the flux (an extra halo cell for ``size=inf``,
        or large spurious edge cells for a monitor wider than the domain).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(primal_widths, dual_widths)``, each length ``num_cells``. Both
        collapse to ``np.array([1.0])`` for the 1D-degenerate case
        (one cell, i.e. ``len(boundaries) == 2``) so flux quantities come
        out in units of ``W/µm`` along zero-sized simulation dimensions --
        matching the :func:`colocated_widths_1d` convention.
    """
    boundaries = np.asarray(boundaries)
    if boundaries.size == 2:
        return np.array([1.0]), np.array([1.0])

    valid_min, valid_max = valid_bounds if valid_bounds is not None else boundaries[[0, -1]]
    integration_min = max(mnt_min, float(valid_min))
    integration_max = min(mnt_max, float(valid_max))

    field_centers = (boundaries[:-1] + boundaries[1:]) / 2
    centers = np.concatenate([[integration_min], field_centers])
    bnds = np.concatenate([boundaries[:-1], [integration_max]])

    primal_widths = _clipped_widths(bnds, integration_min, integration_max)
    dual_widths = _clipped_widths(centers, integration_min, integration_max)
    return primal_widths, dual_widths
