"""How staged point-cloud monitors split a structure's surface sample points into chunks.

Monitor staging and the compiled shape term name and size the chunks with these helpers, so
the two cannot disagree.
"""

from __future__ import annotations


def _adjoint_point_cloud_chunk_size() -> int:
    """Maximum rows per staged point-cloud monitor (import deferred: cycle)."""
    from tidy3d.components.monitor import (
        MAX_POINT_CLOUD_FIELD_MONITOR_POINTS,
        MAX_POINT_CLOUD_PERMITTIVITY_MONITOR_POINTS,
    )

    return min(MAX_POINT_CLOUD_FIELD_MONITOR_POINTS, MAX_POINT_CLOUD_PERMITTIVITY_MONITOR_POINTS)


def adjoint_point_cloud_chunk_bounds(num_points: int) -> list[tuple[int, int]]:
    """Row ranges splitting one structure's concatenated point stream across monitors.

    The two-sided deterministic chunking contract: construction slices each staged
    stream (field query points, per-component PEC points, per-component material
    points — all ``num_points`` rows in artifact entry order) into these ranges, one
    monitor per range, and consumption processes the recorded chunks against these
    same ranges. Structures whose surface sampling exceeds a single monitor's point
    cap therefore split transparently, with no stored payload and no change to the
    per-set offset bookkeeping downstream.
    """
    chunk_size = _adjoint_point_cloud_chunk_size()
    return [
        (start, min(start + chunk_size, num_points)) for start in range(0, num_points, chunk_size)
    ]


def adjoint_point_cloud_chunk_tag(monitor_tag: str, chunk_index: int) -> str:
    """Monitor monitor tag of one chunk; chunk 0 keeps the unsplit base tag."""
    return monitor_tag if chunk_index == 0 else f"{monitor_tag}_c{chunk_index}"
