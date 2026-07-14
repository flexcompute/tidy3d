from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString, Polygon

from tidy3d.components.autograd.utils import hasbox

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


MAX_SELF_INTERSECTION_REPAIR_TRIALS = 10


class SelfIntersectionStatus(str, Enum):
    """Status code for self-intersection handling.

    ``NO_VALID_UPDATE_FOUND``
        No valid repaired update was found, so the current vertices were kept.
    ``CORRECTED_UPDATE_FOUND``
        The returned geometry is a valid partial correction of the requested update.
        This also covers mixed multi-ring outcomes where some rings keep the full proposal
        and others revert.
    ``FULL_UPDATE_APPLIED``
        The proposed update was already valid and was kept as-is.
    """

    NO_VALID_UPDATE_FOUND = "no_valid_update_found"
    CORRECTED_UPDATE_FOUND = "corrected_update_found"
    FULL_UPDATE_APPLIED = "full_update_applied"


def repair_self_intersecting_ring(
    current_vertices: np.ndarray,
    proposed_vertices: np.ndarray,
    *,
    return_status: bool = False,
) -> np.ndarray | tuple[np.ndarray, SelfIntersectionStatus]:
    """Best-effort repair of a proposed closed ring update.

    The returned ring preserves vertex count and either accepts the original proposal,
    salvages it by scaling one candidate arc, or falls back to scaling the full-ring step.
    When ``return_status`` is ``True``, the returned :class:`SelfIntersectionStatus`
    describes the outcome for this ring only.
    """

    if hasbox(current_vertices) or hasbox(proposed_vertices):
        raise NotImplementedError(
            "Self-intersection repair is not supported when differentiating w.r.t. polygon "
            "vertices. Use the raw update path for autograd-traced updates."
        )

    base = np.asarray(current_vertices, dtype=float)
    proposed = np.asarray(proposed_vertices, dtype=float)

    if _ring_is_simple(proposed):
        if not return_status:
            return proposed_vertices
        return proposed_vertices, SelfIntersectionStatus.FULL_UPDATE_APPLIED

    delta = proposed - base
    intersections = find_ring_self_intersections(proposed)
    best_local_candidate = None
    best_local_motion = -np.inf

    for edge_i, edge_j in intersections:
        ordered_sets = ordered_repair_vertex_sets(delta, edge_i=edge_i, edge_j=edge_j)
        for vertex_set in ordered_sets:
            if not vertex_set:
                continue
            local_candidate = maximize_subset_scale(
                base_vertices=base,
                delta_vertices=delta,
                vertex_set=vertex_set,
                max_trials=MAX_SELF_INTERSECTION_REPAIR_TRIALS,
            )
            if local_candidate is None:
                continue
            local_motion = _total_displacement(local_candidate - base)
            if local_motion > best_local_motion:
                best_local_candidate = local_candidate
                best_local_motion = local_motion

    if best_local_candidate is not None:
        if not return_status:
            return best_local_candidate
        return best_local_candidate, _repair_status(
            base_vertices=base,
            proposed_vertices=proposed,
            repaired_vertices=best_local_candidate,
        )

    repaired = maximize_global_scale(
        base_vertices=base,
        delta_vertices=delta,
        max_trials=MAX_SELF_INTERSECTION_REPAIR_TRIALS,
    )
    if not return_status:
        return repaired
    return repaired, _repair_status(
        base_vertices=base,
        proposed_vertices=proposed,
        repaired_vertices=repaired,
    )


def aggregate_self_intersection_statuses(
    statuses: Iterable[SelfIntersectionStatus],
) -> SelfIntersectionStatus:
    """Aggregate per-ring statuses into one update-level status.

    Mixed outcomes are reported as ``CORRECTED_UPDATE_FOUND`` because the returned
    geometry is still a valid partial correction of the requested update.
    """
    status_set = set(statuses)
    if not status_set:
        return SelfIntersectionStatus.FULL_UPDATE_APPLIED
    if status_set == {SelfIntersectionStatus.FULL_UPDATE_APPLIED}:
        return SelfIntersectionStatus.FULL_UPDATE_APPLIED
    if status_set == {SelfIntersectionStatus.NO_VALID_UPDATE_FOUND}:
        return SelfIntersectionStatus.NO_VALID_UPDATE_FOUND
    return SelfIntersectionStatus.CORRECTED_UPDATE_FOUND


def find_ring_self_intersections(vertices: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return non-adjacent edge pairs that intersect in a closed ring."""
    ring = np.asarray(vertices, dtype=float)
    num_vertices = int(ring.shape[0])
    if num_vertices < 4:
        return ()

    segments = [
        LineString([tuple(ring[idx]), tuple(ring[(idx + 1) % num_vertices])])
        for idx in range(num_vertices)
    ]
    intersections: list[tuple[int, int]] = []
    for edge_i in range(num_vertices):
        for edge_j in range(edge_i + 1, num_vertices):
            if _edges_are_adjacent(edge_i, edge_j, num_vertices):
                continue
            if not segments[edge_i].intersects(segments[edge_j]):
                continue
            geom = segments[edge_i].intersection(segments[edge_j])
            if geom.is_empty:
                continue
            intersections.append((edge_i, edge_j))
    return tuple(intersections)


def ordered_repair_vertex_sets(
    delta_vertices: np.ndarray, *, edge_i: int, edge_j: int
) -> tuple[tuple[int, ...], ...]:
    """Return candidate cyclic arcs to shrink for one crossing edge pair."""
    delta = np.asarray(delta_vertices, dtype=float)
    num_vertices = int(delta.shape[0])
    if num_vertices == 0:
        return ()

    arc_a = _cyclic_indices((edge_i + 1) % num_vertices, edge_j % num_vertices, num_vertices)
    arc_b = _cyclic_indices((edge_j + 1) % num_vertices, edge_i % num_vertices, num_vertices)
    candidates = [tuple(arc_a), tuple(arc_b)]

    def _score(indices: tuple[int, ...]) -> tuple[float, int, float]:
        if not indices:
            return (0.0, 0, 0.0)
        disp = np.linalg.norm(delta[list(indices)], axis=1)
        mean_disp = float(np.mean(disp)) if disp.size else 0.0
        total_disp = float(np.sum(disp)) if disp.size else 0.0
        return (-mean_disp, len(indices), -total_disp)

    candidates = sorted(candidates, key=_score)
    return tuple(dict.fromkeys(candidates))


def maximize_subset_scale(
    *,
    base_vertices: np.ndarray,
    delta_vertices: np.ndarray,
    vertex_set: tuple[int, ...],
    max_trials: int,
) -> np.ndarray | None:
    """Return the best valid candidate found for one subset-scaling family."""
    candidate_lo = _apply_subset_scale(base_vertices, delta_vertices, vertex_set, scale=0.0)
    if not _ring_is_simple(candidate_lo):
        return None

    return _search_best_scale(
        candidate_at_scale=lambda scale: _apply_subset_scale(
            base_vertices, delta_vertices, vertex_set, scale=scale
        ),
        max_trials=max_trials,
        initial_scale=0.0,
        initial_candidate=candidate_lo,
    )


def maximize_global_scale(
    *,
    base_vertices: np.ndarray,
    delta_vertices: np.ndarray,
    max_trials: int,
) -> np.ndarray:
    """Return the best valid candidate found along the full-ring update direction."""
    return _search_best_scale(
        candidate_at_scale=lambda scale: np.asarray(
            base_vertices + scale * delta_vertices, dtype=float
        ),
        max_trials=max_trials,
        initial_scale=0.0,
        initial_candidate=np.asarray(base_vertices, dtype=float),
    )


def _search_best_scale(
    *,
    candidate_at_scale: Callable[[float], np.ndarray],
    max_trials: int,
    initial_scale: float,
    initial_candidate: np.ndarray,
) -> np.ndarray:
    """Search a scale family without assuming validity is monotone in the scale."""
    best_scale = initial_scale
    best_candidate = np.asarray(initial_candidate, dtype=float)
    if max_trials <= 0:
        return best_candidate

    for trial_idx in range(1, max_trials + 1):
        scale = _van_der_corput(trial_idx)
        candidate = candidate_at_scale(scale)
        if _ring_is_simple(candidate) and scale > best_scale:
            best_scale = scale
            best_candidate = candidate

    return best_candidate


def _van_der_corput(index: int) -> float:
    """Return the base-2 van der Corput sample in ``(0, 1)`` for a 1-based index."""
    scale = 0.0
    denominator = 2.0
    value = index
    while value > 0:
        value, remainder = divmod(value, 2)
        scale += remainder / denominator
        denominator *= 2.0
    return scale


def _ring_is_simple(vertices: np.ndarray) -> bool:
    polygon = Polygon(np.asarray(vertices, dtype=float))
    return polygon.is_valid and polygon.is_simple and polygon.area > 0.0


def _edges_are_adjacent(edge_i: int, edge_j: int, num_vertices: int) -> bool:
    if (edge_i + 1) % num_vertices == edge_j:
        return True
    if (edge_j + 1) % num_vertices == edge_i:
        return True
    return False


def _cyclic_indices(start: int, stop: int, num_vertices: int) -> tuple[int, ...]:
    indices = [start]
    while indices[-1] != stop:
        indices.append((indices[-1] + 1) % num_vertices)
    return tuple(indices)


def _apply_subset_scale(
    base_vertices: np.ndarray,
    delta_vertices: np.ndarray,
    vertex_set: tuple[int, ...],
    *,
    scale: float,
) -> np.ndarray:
    candidate = np.asarray(base_vertices + delta_vertices, dtype=float)
    candidate[list(vertex_set)] = np.asarray(
        base_vertices[list(vertex_set)], dtype=float
    ) + scale * np.asarray(delta_vertices[list(vertex_set)], dtype=float)
    return candidate


def _total_displacement(delta: np.ndarray) -> float:
    if delta.size == 0:
        return 0.0
    return float(np.sum(np.linalg.norm(delta, axis=1)))


def _repair_status(
    *,
    base_vertices: np.ndarray,
    proposed_vertices: np.ndarray,
    repaired_vertices: np.ndarray,
) -> SelfIntersectionStatus:
    if np.allclose(repaired_vertices, proposed_vertices):
        return SelfIntersectionStatus.FULL_UPDATE_APPLIED
    if np.allclose(repaired_vertices, base_vertices):
        return SelfIntersectionStatus.NO_VALID_UPDATE_FOUND
    return SelfIntersectionStatus.CORRECTED_UPDATE_FOUND
