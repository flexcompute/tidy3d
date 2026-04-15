from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
import numpy as npo

if TYPE_CHECKING:
    from typing import Callable

    from tidy3d.components.types import ArrayFloat1D, ArrayFloat2D

    from .polyslab_set import PolySlabSet


def _consecutive_triplet_penalties(
    vertices: ArrayFloat2D, penalty_fn: Callable[[ArrayFloat2D], float]
) -> ArrayFloat1D:
    """Evaluate a local 3-point penalty on each cyclic vertex window of a closed ring."""
    ring = vertices
    n_ring_vertices = int(ring.shape[0])
    if n_ring_vertices < 3:
        raise ValueError("Curvature penalty requires polygon rings with at least 3 vertices.")

    ring_triplets = np.stack((np.roll(ring, 1, axis=0), ring, np.roll(ring, -1, axis=0)), axis=1)
    penalty_values = [penalty_fn(triplet) for triplet in ring_triplets]
    return np.stack(penalty_values, axis=0)


def curvature_penalty(
    polyslab_set: PolySlabSet,
    penalty_fn: Callable[[ArrayFloat2D], float],
    *,
    ignore_boundary_vertices: bool = True,
) -> float:
    """Aggregate a local 3-point curvature-style penalty across contour rings.

    The penalty is evaluated on cyclic 3-point windows so masking boundary
    vertices preserves local adjacency for curvature-based penalties on closed rings.
    """
    weighted_sum = 0.0
    total_contributions = 0
    for polyslab, frame_mask in zip(
        polyslab_set.polyslabs, polyslab_set.frame_boundary_vertex_mask
    ):
        vertices = np.array(polyslab.vertices)
        n_ring_vertices = int(vertices.shape[0])
        if n_ring_vertices < 3:
            raise ValueError("Curvature penalty requires polygon rings with at least 3 vertices.")

        boundary_mask = npo.asarray(frame_mask, dtype=bool).reshape(-1)
        if boundary_mask.shape[0] != n_ring_vertices:
            raise ValueError("Boundary mask length must match the number of ring vertices.")

        triplet_penalties = _consecutive_triplet_penalties(vertices, penalty_fn)
        if ignore_boundary_vertices:
            contribution_mask = ~boundary_mask
        else:
            contribution_mask = npo.ones(n_ring_vertices, dtype=bool)

        contribution_count = int(npo.sum(contribution_mask))
        if contribution_count == 0:
            continue
        weighted_sum += np.sum(triplet_penalties[contribution_mask])
        total_contributions += contribution_count
    if total_contributions == 0:
        return 0.0
    return weighted_sum / float(total_contributions)
