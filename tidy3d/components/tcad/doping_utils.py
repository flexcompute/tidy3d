"""Seam-aware aggregation for doping boxes (any ``AbstractDopingBox`` subtype).

``aggregate_doping_seam_aware`` is the canonical definition of how a set of doping boxes
accumulates onto node coordinates. It treats each finite box boundary as a 1-ULP shell,
preserves intentional interior stacking, and resolves multi-box boundary nodes to the
one-sided stack owned by the later box in the list (last-in-list-wins, mirroring how
overlapping tidy3d structures resolve) so abutting boxes do not double-count shared
faces. It covers ConstantDoping, GaussianDoping, CustomDoping, and any future
``AbstractDopingBox`` — each box contributes its own per-node profile. It is shared by
simulation setup and doping plots so rendered and solved fields match; any
reimplementation must reproduce its result byte-for-byte.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from tidy3d.components.tcad.doping import AbstractDopingBox, ConstantDoping

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.components.tcad.doping import DopingBoxType


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------


def _ulp(v: float) -> float:
    """Distance from v to the next representable float (1 ULP at v)."""
    if not np.isfinite(v) or v == 0.0:
        return 5e-324  # smallest positive float
    return abs(np.nextafter(v, np.inf) - v)


def _seam_axis(
    lo_a: NDArray,
    hi_a: NDArray,
    lo_b: NDArray,
    hi_b: NDArray,
) -> tuple[int, bool] | None:
    """Return ``(axis, a_is_left)`` if the boxes share exactly one face, else None.

    ``a_is_left`` means ``hi_a[axis] ≈ lo_b[axis]``. None = no touch, edge/corner
    only, or volumetric overlap.
    """
    touching: list[tuple[int, bool]] = []

    for ax in range(3):
        tol_ab = max(_ulp(float(hi_a[ax])), _ulp(float(lo_b[ax])))
        tol_ba = max(_ulp(float(hi_b[ax])), _ulp(float(lo_a[ax])))

        if abs(float(hi_a[ax]) - float(lo_b[ax])) <= tol_ab:
            touching.append((ax, True))
        elif abs(float(hi_b[ax]) - float(lo_a[ax])) <= tol_ba:
            touching.append((ax, False))

    # A face-seam has exactly one touching axis.
    if len(touching) != 1:
        return None

    ax, a_is_left = touching[0]

    # Boxes must have positive overlap on the other two axes (face, not edge).
    for other in range(3):
        if other == ax:
            continue
        if float(hi_a[other]) <= float(lo_b[other]) or float(hi_b[other]) <= float(lo_a[other]):
            return None

    return ax, a_is_left


# ---------------------------------------------------------------------------
# Node-level two-track aggregation
# ---------------------------------------------------------------------------


def _inside_box(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    lo: NDArray,
    hi: NDArray,
) -> NDArray:
    """Point-in-box test on flat node arrays, bounds expanded 1 ULP outward.

    The outward expansion captures a node that ``center ± size/2`` rounding placed up
    to 1 ULP outside a finite face. Infinite faces are left unbounded.
    """
    mask = np.ones(len(x), dtype=bool)
    for coord, lo_v, hi_v in zip((x, y, z), lo, hi):
        if np.isfinite(lo_v):
            mask &= coord >= np.nextafter(float(lo_v), -np.inf)
        if np.isfinite(hi_v):
            mask &= coord <= np.nextafter(float(hi_v), +np.inf)
    return mask


def _near_box_boundary(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    lo: NDArray,
    hi: NDArray,
) -> NDArray:
    """Points in the 1-ULP shell of any finite face of a box."""
    boundary = np.zeros(len(x), dtype=bool)
    for coord, lo_v, hi_v in zip((x, y, z), lo, hi):
        if np.isfinite(lo_v):
            lo_f = float(lo_v)
            boundary |= (coord >= np.nextafter(lo_f, -np.inf)) & (
                coord <= np.nextafter(lo_f, +np.inf)
            )
        if np.isfinite(hi_v):
            hi_f = float(hi_v)
            boundary |= (coord >= np.nextafter(hi_f, -np.inf)) & (
                coord <= np.nextafter(hi_f, +np.inf)
            )
    return boundary


def _near_bound(coord: float, bound: float) -> bool:
    """Return true when ``coord`` is within the 1-ULP shell of ``bound``."""
    if not np.isfinite(bound):
        return False
    return abs(coord - bound) <= max(_ulp(coord), _ulp(bound))


def _axis_side_options(coord: float, lo: float, hi: float) -> set[int]:
    """Directional sectors touched by a box at one coordinate.

    ``-1`` means the box interior approaches from the lower/left side, ``+1`` from
    the upper/right side. Strictly interior coordinates contribute to both sides.
    If finite bounds collapse to the same float, the result is intentionally
    ambiguous and active seam-pair orientation resolves it.
    """
    options: set[int] = set()
    if _near_bound(coord, lo):
        options.add(+1)
    if _near_bound(coord, hi):
        options.add(-1)
    return options or {-1, +1}


def _owning_side_stack(
    covering: list[int],
    los: list[NDArray],
    his: list[NDArray],
    node_contribs: list[float],
    point: tuple[float, float, float],
    active_pairs: list[tuple[int, int, int, bool]],
) -> float:
    """Doping stack on the directional sector owned by the later box (last-in-list-wins).

    Boxes covering the node are grouped into directional sectors using the same one-sided
    decomposition used elsewhere in this module. The winning value is the stack
    on the sector(s) occupied by the highest-index covering box, so the later box takes a
    shared boundary while genuine overlaps on that box's side still sum. ``node_contribs``
    is each box's per-node contribution at this node (uniform concentration for Constant,
    the evaluated profile for Gaussian/Custom), so the resolver works for any box type.
    """
    options = {
        i: [
            _axis_side_options(point[axis], float(los[i][axis]), float(his[i][axis]))
            for axis in range(3)
        ]
        for i in covering
    }

    # Active seam pairs are authoritative about which side of their shared face a
    # box owns. This matters for large coordinates where finite bounds can collapse
    # to the same representable float and coordinate-only side tests are ambiguous.
    for i, j, axis, i_is_left in active_pairs:
        if i not in options or j not in options:
            continue
        i_side, j_side = (-1, +1) if i_is_left else (+1, -1)
        options[i][axis] = {i_side}
        options[j][axis] = {j_side}

    sector_sums: dict[tuple[int, int, int], float] = {}
    for i in covering:
        for sector in product(*[sorted(axis_options) for axis_options in options[i]]):
            sector_sums[sector] = sector_sums.get(sector, 0.0) + node_contribs[i]

    # Last-in-list-wins: the latest covering box owns the node. Among the sectors it
    # occupies, take the largest stack so a genuine overlap on its side is preserved.
    last = max(covering)
    last_sectors = product(*[sorted(axis_options) for axis_options in options[last]])
    return max((sector_sums[sector] for sector in last_sectors), default=0.0)


def _box_contrib(
    box: DopingBoxType,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    lo: NDArray,
    hi: NDArray,
) -> NDArray:
    """Per-node doping contribution of a single box, evaluated as if every node were
    inside it (membership is applied separately by the caller).

    ConstantDoping is the uniform ``concentration`` (fast path). Every other doping-box
    type (GaussianDoping, CustomDoping, and any future ``AbstractDopingBox``) is evaluated
    generically through its own ``_get_contrib`` on coordinates first clamped to the box's
    bounds. The clamp makes a node that ``center ± size/2`` rounding placed up to 1 ULP
    outside a finite face evaluate to the boundary value (``ref_con`` for Gaussian, the
    edge sample for Custom) rather than 0. Interior nodes are unaffected by the clamp.
    """
    if isinstance(box, ConstantDoping):
        return np.full(len(x), box.concentration)

    xc = np.clip(x, lo[0], hi[0])
    yc = np.clip(y, lo[1], hi[1])
    zc = np.clip(z, lo[2], hi[2])
    contrib = box._get_contrib({"x": xc, "y": yc, "z": zc}, meshgrid=False)
    return np.asarray(contrib, dtype=float).reshape(len(x))


def aggregate_doping_seam_aware(
    doping_list: list[DopingBoxType],
    x: NDArray,
    y: NDArray,
    z: NDArray,
) -> NDArray:
    """Aggregate one doping field's boxes (``N_d`` or ``N_a``) onto node coords ``x, y, z``.

    Pass a single-polarity box list — all donor boxes or all acceptor boxes, never a
    mix. Abutting boxes have their shared boundary deduped against each other, so a
    donor box and an acceptor box that happen to touch would be wrongly cancelled even
    though donor and acceptor doping are physically independent quantities.

    Applies to every doping-box type uniformly (ConstantDoping, GaussianDoping,
    CustomDoping, and any future ``AbstractDopingBox``). Each box contributes its own
    per-node profile (see ``_box_contrib``): the uniform ``concentration`` for Constant,
    the Gaussian profile for Gaussian, the interpolated array for Custom.

    Membership uses each box's bounds expanded 1 ULP outward, so a node that
    ``center ± size/2`` rounding placed just outside a face is still captured — including
    when that node is also interior to an overlapping box (the expansion is part of
    membership, not a separate fallback track). For Gaussian/Custom the recovered node
    evaluates to the box's boundary value via the clamp in ``_box_contrib``. Boxes that
    genuinely overlap away from finite boundaries stack (their contributions sum).

    Multi-box boundary nodes resolve to the one-sided stack owned by the later box in the
    list (last-in-list-wins, mirroring how overlapping tidy3d structures resolve), which
    dedupes abutting seams of any box-type combination without dropping a genuine overlap
    on the winning box's side.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    coords = {"x": x, "y": y, "z": z}

    # Every doping-box type joins the seam-aware track. The fallback sum only catches
    # non-box inputs (defensive; callers pass DopingBoxType instances).
    other_sum = np.zeros(len(x))
    boxes = []
    for box in doping_list:
        if isinstance(box, AbstractDopingBox):
            boxes.append(box)
        else:
            other_sum += box._get_contrib(coords, meshgrid=False)

    if not boxes:
        return other_sum

    los = [np.asarray(b.bounds[0], dtype=float) for b in boxes]
    his = [np.asarray(b.bounds[1], dtype=float) for b in boxes]
    # Per-node contribution of each box (profile evaluated everywhere; membership masks
    # below). For Gaussian/Custom this folds in the 1-ULP boundary recovery via clamping.
    contribs = [_box_contrib(boxes[i], x, y, z, los[i], his[i]) for i in range(len(boxes))]
    # Effective membership: bounds expanded 1 ULP outward at finite faces.
    mem = [_inside_box(x, y, z, los[i], his[i]) for i in range(len(boxes))]

    any_mem = np.zeros(len(x), dtype=bool)
    cover_count = np.zeros(len(x), dtype=int)
    cover_sum = np.zeros(len(x))
    boundary_mem = np.zeros(len(x), dtype=bool)
    for i in range(len(boxes)):
        cover_sum += np.where(mem[i], contribs[i], 0.0)
        cover_count += mem[i].astype(int)
        boundary_mem |= mem[i] & _near_box_boundary(x, y, z, los[i], his[i])
        any_mem |= mem[i]

    # Face-seam orientation resolves ambiguous side ownership at large coordinates
    # where finite bounds can collapse to the same representable float.
    seam_pairs: list[tuple[int, int, int, bool]] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            seam = _seam_axis(los[i], his[i], los[j], his[j])
            if seam is None:
                continue
            axis, i_is_left = seam
            seam_pairs.append((i, j, axis, i_is_left))

    # Multi-box boundary nodes resolve to the one-sided stack owned by the later box,
    # covering stacked boxes on either side of a face and multi-box corners. Interior
    # overlaps keep their plain sum.
    needs_resolution = (cover_count > 1) & boundary_mem
    resolved = cover_sum.copy()
    for node_idx in np.flatnonzero(needs_resolution):
        covering = [i for i in range(len(boxes)) if mem[i][node_idx]]
        active_pairs = [
            (i, j, axis, i_is_left)
            for i, j, axis, i_is_left in seam_pairs
            if mem[i][node_idx] and mem[j][node_idx]
        ]
        point = (float(x[node_idx]), float(y[node_idx]), float(z[node_idx]))
        node_contribs = [contribs[i][node_idx] for i in range(len(boxes))]
        resolved[node_idx] = _owning_side_stack(
            covering, los, his, node_contribs, point, active_pairs
        )

    seam_aware = np.where(any_mem, resolved, 0.0)
    return other_sum + seam_aware


# Backward-compatible alias for external callers importing the original name.
aggregate_constant_doping_seam_aware = aggregate_doping_seam_aware
