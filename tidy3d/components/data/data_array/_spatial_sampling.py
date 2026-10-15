"""Sample-count limits and Cartesian grid builders for spatial resampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tidy3d.exceptions import DataError, format_chained_exception_message

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from tidy3d.components.types import ArrayLike, Bound


# Ceiling on the Cartesian grid ``_grid_from_bounds`` will build. The array is held in memory
# whole and interpolated point by point, and a bounds/resolution pair is taken literally, so an
# innocent-looking resolution over a device-sized box can ask for terabytes.
MAX_SPATIAL_SAMPLES = 4_000_000


def _check_sample_count(coords: Sequence[ArrayLike]) -> None:
    """Reject a target grid too large to materialise."""
    counts = [len(np.atleast_1d(comp)) for comp in coords]
    total = counts[0] * counts[1] * counts[2]
    if total > MAX_SPATIAL_SAMPLES:
        raise DataError(
            f"The requested grid has {counts[0]}x{counts[1]}x{counts[2]} = {total} "
            f"points, above the {MAX_SPATIAL_SAMPLES} supported. Coarsen 'resolution' "
            "or narrow 'bounds'; the array is held in memory whole and sampled point "
            "by point."
        )


def _grid_steps(resolution: float | ArrayLike) -> NDArray:
    """Validate ``resolution`` as one positive, finite step per axis."""
    # Broadcasting rather than a size check: it accepts a scalar and a 3-vector and rejects
    # every other shape, including the nested ones a size check lets through.
    try:
        steps = np.broadcast_to(np.asarray(resolution, dtype=float), (3,))
    except (TypeError, ValueError) as e:
        raise DataError(
            format_chained_exception_message(
                "Grid spacing must be a scalar or one value per 'x', 'y', and 'z'; got "
                f"{resolution!r}.",
                e,
            )
        ) from e
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise DataError(f"Grid spacing must be finite and positive; got {resolution!r}.")
    return steps


def _sampling_counts(steps: NDArray, extents: NDArray) -> NDArray:
    """Nodes per axis that :func:`_grid_from_bounds` places over ``extents`` at ``steps``."""
    return np.maximum(2, np.ceil(extents / steps) + 1)


def _clamp_steps_to_budget(
    steps: NDArray, extents: NDArray, budget: float, fixed_total: float = 1
) -> tuple[NDArray, NDArray] | None:
    """Raise ``steps`` until the grid they build holds at most ``budget`` points.

    ``steps`` and ``extents`` cover only the axes being resampled; ``fixed_total`` is the
    number of points the remaining axes contribute. Returns the raised steps together with
    the node counts they land on, or ``None`` if the request already fits.

    Sampling a field onto a grid it cannot support invents detail, and since the result is
    allocated dense it can ask for hundreds of gigabytes on a device-scale field.
    """
    if extents.size == 0:
        return None

    # Re-counted after every pass rather than scaled once by the excess ratio: the ratio is
    # continuous and 'ceil' rounds back up, so one pass can land on the same grid it started
    # from. Every pass grows the steps, so the counts fall monotonically towards the two
    # endpoints each axis keeps; stop there even if that grid is still above budget.
    counts = _sampling_counts(steps, extents)
    if np.prod(counts) * fixed_total <= budget:
        return None

    clamped = np.array(steps, dtype=float)
    while np.prod(counts) * fixed_total > budget and np.any(counts > 2):
        clamped *= (np.prod(counts) * fixed_total / budget) ** (1 / clamped.size)
        counts = _sampling_counts(clamped, extents)
    return clamped, counts


def _grid_from_bounds(
    bounds: Bound,
    resolution: float | ArrayLike,
    keep_coords: Mapping[str, NDArray] | None = None,
    max_samples: float = MAX_SPATIAL_SAMPLES,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build per-axis coordinate arrays spanning ``bounds`` at ``resolution``.

    ``keep_coords`` maps an axis name to a coordinate returned untouched. It is how a caller
    holding data already coarser than ``resolution`` along some axis avoids resampling it; such
    an axis still counts towards ``max_samples``. A caller resampling data it already holds can
    raise ``max_samples`` to that data's own size: the grid is then refused only if it would be
    larger than what is in memory already.
    """
    # Checked before unpacking: a malformed pair would otherwise surface as a bare
    # 'not enough values to unpack' from somewhere downstream.
    if len(bounds) != 2 or any(len(corner) != 3 for corner in bounds):
        raise DataError(
            f"'bounds' must be ((xmin, ymin, zmin), (xmax, ymax, zmax)); got {bounds!r}."
        )

    rmin, rmax = bounds
    steps = _grid_steps(resolution)
    keep_coords = keep_coords or {}

    # Sizes first, allocation second: the whole point of the cap is to refuse before
    # anything large is materialised, and np.linspace on one enormous axis would
    # already have exhausted memory by the time a post-hoc check ran.
    counts = []
    for axis, (lo, hi, step) in enumerate(zip(rmin, rmax, steps)):
        dim = "xyz"[axis]
        if dim in keep_coords:
            counts.append(len(np.atleast_1d(keep_coords[dim])))
            continue
        extent = hi - lo
        if extent < 0:
            # Kept distinct from an equal pair: swapped bounds would otherwise collapse
            # to a single plane and quietly return a field over the wrong region.
            raise DataError(
                f"'bounds' is reversed along {dim}: the minimum ({lo}) is above "
                f"the maximum ({hi}). Expected ((xmin, ymin, zmin), (xmax, ymax, zmax))."
            )
        # A zero-thickness axis (a 2D simulation) collapses to one sample plane.
        counts.append(1 if extent == 0 else max(2, int(np.ceil(extent / step)) + 1))

    total = counts[0] * counts[1] * counts[2]
    if total > max_samples:
        raise DataError(
            f"'bounds' at this 'resolution' needs {counts[0]}x{counts[1]}x{counts[2]} = "
            f"{total} points, above the {int(max_samples)} supported. Coarsen "
            "'resolution' or narrow 'bounds'."
        )

    return tuple(
        np.atleast_1d(keep_coords[dim])
        if dim in keep_coords
        else (np.array([lo]) if count == 1 else np.linspace(lo, hi, count))
        for dim, lo, hi, count in zip("xyz", rmin, rmax, counts)
    )
