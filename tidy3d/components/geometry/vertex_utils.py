"""Utilities for polygon vertex arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


DEFAULT_ADJACENT_DUPLICATE_TOL = 1e-8


def remove_adjacent_duplicate_vertices(
    vertices: NDArray,
    tol: float = DEFAULT_ADJACENT_DUPLICATE_TOL,
    rtol: float = 0.0,
    drop_closing_duplicate: bool = False,
) -> NDArray:
    """Remove adjacent duplicate vertices with a vectorized nearest-neighbor check.

    Parameters
    ----------
    vertices : np.ndarray
        Shape ``(N, 2)`` array defining polygon vertices in a plane.
    tol : float
        Vertices whose distance to the next vertex is less than or equal to ``tol`` are treated
        as duplicates.
    rtol : float
        Relative tolerance passed to :func:`numpy.isclose` when comparing edge lengths to zero.
    drop_closing_duplicate : bool
        If ``True``, also remove a final vertex when it duplicates the first one within ``tol``.

    Returns
    -------
    np.ndarray
        Vertices with adjacent duplicates removed while preserving one representative point per
        duplicate run.
    """
    if len(vertices) == 0:
        return vertices

    vertices_next = np.roll(vertices, shift=-1, axis=0)
    vertices_diff = np.linalg.norm(vertices - vertices_next, axis=1)
    deduped = vertices[~np.isclose(vertices_diff, 0, atol=tol, rtol=rtol)]

    if len(deduped) == 0:
        return vertices[:1]

    if (
        drop_closing_duplicate
        and len(deduped) > 1
        and np.isclose(np.linalg.norm(deduped[0] - deduped[-1]), 0, atol=tol, rtol=rtol)
    ):
        deduped = deduped[:-1]

    return deduped
