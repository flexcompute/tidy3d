"""Geometry operations for bounding box type with minimal imports."""

from __future__ import annotations

from math import isclose

from tidy3d.components.types import Bound
from tidy3d.constants import fp_eps


def bounds_intersection(bounds1: Bound, bounds2: Bound) -> Bound:
    """Return the bounds that are the intersection of two bounds."""
    rmin1, rmax1 = bounds1
    rmin2, rmax2 = bounds2
    rmin = tuple(max(v1, v2) for v1, v2 in zip(rmin1, rmin2))
    rmax = tuple(min(v1, v2) for v1, v2 in zip(rmax1, rmax2))
    return (rmin, rmax)


def bounds_union(bounds1: Bound, bounds2: Bound) -> Bound:
    """Return the bounds that are the union of two bounds."""
    rmin1, rmax1 = bounds1
    rmin2, rmax2 = bounds2
    rmin = tuple(min(v1, v2) for v1, v2 in zip(rmin1, rmin2))
    rmax = tuple(max(v1, v2) for v1, v2 in zip(rmax1, rmax2))
    return (rmin, rmax)


def bounds_contains(
    outer_bounds: Bound, inner_bounds: Bound, rtol: float = fp_eps, atol: float = 0.0
) -> bool:
    """Checks whether ``inner_bounds`` is contained within ``outer_bounds`` within specified tolerances.

    Parameters
    ----------
    outer_bounds : Bound
        The outer bounds to check containment against
    inner_bounds : Bound
        The inner bounds to check if contained
    rtol : float = fp_eps
        Relative tolerance for comparing bounds
    atol : float = 0.0
        Absolute tolerance for comparing bounds

    Returns
    -------
    bool
        True if ``inner_bounds`` is contained within ``outer_bounds`` within tolerances
    """
    outer_min, outer_max = outer_bounds
    inner_min, inner_max = inner_bounds
    for dim in range(3):
        outer_min_dim = outer_min[dim]
        outer_max_dim = outer_max[dim]
        inner_min_dim = inner_min[dim]
        inner_max_dim = inner_max[dim]
        within_min = (
            isclose(outer_min_dim, inner_min_dim, rel_tol=rtol, abs_tol=atol)
            or outer_min_dim <= inner_min_dim
        )
        within_max = (
            isclose(outer_max_dim, inner_max_dim, rel_tol=rtol, abs_tol=atol)
            or outer_max_dim >= inner_max_dim
        )

        if not within_min or not within_max:
            return False
    return True
