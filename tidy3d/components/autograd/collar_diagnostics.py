"""Pre-run accuracy checks of a structure's collar (exterior stress sleeve).

Each check reports, as a warning, the share of a traced PEC surface whose collar gradient
loses accuracy: metal in front of the surface, the surface curving within the collar's
depth, sources and monitors in the collar, media the stress does not model, and structures
crossing an absorber.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flexcompute.core._migration.em.numerical.raw.collar import samples_near_cut

from tidy3d.components.structure import Structure

from .staging import effective_is_pec_at_points, is_pec_surface, sheet_opened_at

if TYPE_CHECKING:
    from tidy3d.components.medium import AbstractMedium
    from tidy3d.components.simulation import Simulation

    from .spacing import SamplingScanIndex

# room a collar column needs before metal in front of it, beyond its rigid depth, in local
# cells: every gap holding a single cell layer, where the collar's force on the face is lost,
# is narrower than this (see ``collar_column_obstacles``)
COLLAR_COLUMN_ROOM_CELLS = 1.5
# estimated collar volume per unit of flat-surface collar above which the surface turns too
# fast for the velocity extension (see ``collar_expansion``; reached by a sphere of radius ~5
# local cells, whose gradient was 4% off, 19% at three cells), and the share of a structure's
# surface either collar check must flag before it is reported
COLLAR_EXPANSION_LIMIT = 1.7
# largest normal turn between neighboring samples on one smooth patch for the expansion
# check (a three-cell sphere's samples turn ~27 degrees over 1.5 sample spacings)
COLLAR_PATCH_TURN_DEG = 40.0
COLLAR_WARN_FRACTION = 0.01


def structure_label(structure: Structure) -> str:
    """A structure's name, or its geometry type when it has none."""
    return structure.name or structure.geometry.type


def structure_medium(simulation: Simulation, structure_index: float) -> AbstractMedium:
    """The optical medium of a structure (an index, possibly stored as a float), or the
    simulation's for ``-1``; a ``MultiPhysicsMedium`` resolves to its optical medium."""
    if structure_index < 0:
        medium = simulation.medium
    else:
        medium = simulation.structures[int(structure_index)].medium
    return Structure._get_public_optical_medium(medium)


def collar_column_obstacles(
    active_points: np.ndarray,
    outward: np.ndarray,
    sample_cell: np.ndarray,
    sample_offset: np.ndarray,
    simulation: Simulation,
    scan_index: SamplingScanIndex,
    closed_sides: list[tuple[int, int]],
) -> np.ndarray:
    """Active samples whose collar column meets metal close in front of them.

    The column runs from the sample along its exterior normal ``outward`` (turned toward the
    side that is not metal by the activity probes of :func:`_structure_active_surface`) out to
    the rigid depth plus ``COLLAR_COLUMN_ROOM_CELLS``. Metal
    there (another structure, a PEC / PMC domain wall seen as the domain's edge, or the
    structure itself across a narrow gap or an inside corner) stops the velocity before it
    can taper: the foreign-surface cutoff ramps it to zero over one cell, so the column's
    whole drop is read from a cell or two. Two cell layers between the surface and the metal
    read it exactly (a flat face over flat metal matched FD at 1.5 to 3 cells); one layer
    loses it, since the cutoff is zero within half a cell of the metal and that cell's face
    on the surface is cut too. A zero-thickness PEC sheet, which no probe point lands in, counts
    where the column crosses its plane inside its outline. ``closed_sides`` are the PEC / PMC
    domain sides as ``(axis, side)`` (see ``collar_staging._domain_sides``).
    """
    if active_points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    half = 0.5 * sample_cell
    limit = sample_offset + COLLAR_COLUMN_ROOM_CELLS * sample_cell
    bounds_min, bounds_max = (np.asarray(b, dtype=float) for b in simulation.bounds)
    hit = np.zeros(active_points.shape[0], dtype=bool)
    steps = max(2, int(np.ceil(2.0 * np.max(limit / sample_cell))))
    for fraction in np.linspace(1.0 / steps, 1.0, steps):
        probes = active_points + (half + fraction * (limit - half))[:, None] * outward
        hit |= effective_is_pec_at_points(probes, simulation, scan_index) > 0
        for axis, side in closed_sides:
            hit |= (
                probes[:, axis] < bounds_min[axis]
                if side == 0
                else probes[:, axis] > bounds_max[axis]
            )
    start = active_points + half[:, None] * outward
    end = active_points + limit[:, None] * outward
    for structure_index, structure in enumerate(simulation.structures):
        sheet_min, sheet_max = (np.asarray(b, dtype=float) for b in structure.geometry.bounds)
        axis = int(np.argmin(sheet_max - sheet_min))
        if not is_pec_surface(structure.medium, simulation) or sheet_max[axis] > sheet_min[axis]:
            continue
        before, after = start[:, axis] - sheet_min[axis], end[:, axis] - sheet_min[axis]
        crosses = np.flatnonzero((before * after <= 0.0) & (before != after))
        if crosses.size == 0:
            continue
        fraction = before[crosses] / (before[crosses] - after[crosses])
        crossing = start[crosses] + fraction[:, None] * (end[crosses] - start[crosses])
        crossing[:, axis] = sheet_min[axis]
        hit[crosses] |= structure.geometry.inside(*crossing.T) & ~sheet_opened_at(
            crossing, axis, half[crosses], structure_index, simulation
        )
    return hit


def collar_expansion(
    active_points: np.ndarray,
    outward: np.ndarray,
    active_weights: np.ndarray,
    sample_depth: np.ndarray,
) -> np.ndarray:
    """Estimated collar volume per unit of flat-surface collar around each active sample.

    The collar spreads its velocity through a shell of depth ``d`` (rigid depth plus taper);
    over a surface of mean curvature ``H`` (positive where it bulges toward the collar) that
    shell holds ``1 + H d`` (plus ``O(d^2 / R^2)``) times the volume it would over a flat
    surface (``1 + d / R`` on a sphere, ``1 + d / 2R`` on a cylinder). The kernel's first
    distance moment follows that growth only to first order in ``d / R``. ``H`` comes from the
    spread of neighboring exterior normals,
    ``(n_j - n_i) . (x_j - x_i) / |x_j - x_i|^2`` over neighbors on the same smooth patch
    (within 1.5 sample spacings, turning by under ``COLLAR_PATCH_TURN_DEG``: sharp edges and
    polygon corners take their volume through the kernel's across-edge rule instead),
    averaged by area over bins one collar depth wide so a single corner does not count as a
    surface.
    The discrete estimate runs below ``1 + d / R``: 2.07, 1.72, 1.54 and 1.27 on spheres of
    radius 3.2, 4.8, 6.4 and 12.7 local cells, and exactly one on flat faces.
    """
    from scipy.spatial import cKDTree

    num = active_points.shape[0]
    tree = cKDTree(active_points)
    spacing, _ = tree.query(active_points, k=min(2, num))
    spacing = float(np.median(spacing.reshape(num, -1)[:, -1])) if num > 1 else 0.0
    curvature = np.zeros(num)
    if spacing > 0:
        near = tree.sparse_distance_matrix(
            tree, max_distance=1.5 * spacing, output_type="coo_matrix"
        )
        i, j = near.row, near.col
        dx = active_points[j] - active_points[i]
        same_patch = np.einsum("ni,ni->n", outward[i], outward[j]) > np.cos(
            np.deg2rad(COLLAR_PATCH_TURN_DEG)
        )
        terms = np.einsum("ni,ni->n", outward[j] - outward[i], dx) / np.maximum(
            np.einsum("ni,ni->n", dx, dx), np.finfo(float).tiny
        )
        count = np.bincount(i[same_patch], minlength=num)
        curvature = np.bincount(i[same_patch], weights=terms[same_patch], minlength=num)
        curvature = curvature / np.maximum(count, 1)
    # area-weighted mean over bins one collar depth wide
    _, region = np.unique(
        np.floor(active_points / float(np.max(sample_depth))).astype(np.int64),
        axis=0,
        return_inverse=True,
    )
    region = region.ravel()
    weights = np.asarray(active_weights, dtype=float)
    mean_curvature = np.bincount(region, weights=weights * curvature) / np.maximum(
        np.bincount(region, weights=weights), np.finfo(float).tiny
    )
    return 1.0 + mean_curvature[region] * sample_depth


def _warn_collar_shape(
    structure: Structure,
    weights: np.ndarray,
    obstacles: np.ndarray,
    expansion: np.ndarray,
    offset_cells: float,
) -> None:
    """Warn when metal crowds the collar or the surface turns within its depth.

    ``offset_cells`` is the collar's rigid depth in local cells, which the room in front of a
    sample (``COLLAR_COLUMN_ROOM_CELLS``) extends."""
    from tidy3d.log import log

    name = structure_label(structure)
    area = max(float(np.sum(weights)), np.finfo(float).tiny)
    crowded = float(np.sum(weights[obstacles])) / area
    if crowded >= COLLAR_WARN_FRACTION:
        log.warning(
            f"Adjoint collar: {100.0 * crowded:.1f}% of the PEC surface of structure '{name}' "
            "has metal (another structure, a domain wall, or itself across a narrow gap or "
            f"inside corner) within {offset_cells + COLLAR_COLUMN_ROOM_CELLS:g} local "
            "cells in front of it; the collar has no room to taper there, and the shape "
            "gradient of that part of the surface may be lost entirely (where the gap holds a "
            "single grid cell). Widen the gap or refine the grid across it."
        )
    tight = float(np.sum(weights[expansion > COLLAR_EXPANSION_LIMIT])) / area
    if tight >= COLLAR_WARN_FRACTION:
        log.warning(
            f"Adjoint collar: {100.0 * tight:.1f}% of the PEC surface of structure '{name}' "
            "curves within the collar's depth (a radius of curvature under about five local "
            "cells); its shape gradient there may be off by several percent (about 20% for a "
            "radius of three cells). Refine the grid over these features."
        )


def warn_near_cut(
    owner: str,
    active_points: np.ndarray,
    active_normals: np.ndarray,
    reach: np.ndarray,
    cut_centers: np.ndarray,
    cut_sizes: np.ndarray,
    what: str,
) -> None:
    """Warn when cutting cells (``cut_centers``, ``cut_sizes``) from the collar of ``owner``
    (a structure, as it should be named in the message) shortens it for part of the surface.
    A column passing within a cut cell's size of its center counts as cut."""
    from tidy3d.log import log

    if cut_centers.shape[0] == 0:
        return
    width = float(np.max(cut_sizes))
    near = samples_near_cut(active_points, active_normals, reach, width, cut_centers)
    if np.any(near):
        log.warning(
            f"Adjoint collar: {what} lies within the collar of {100.0 * np.mean(near):.1f}% of "
            f"the PEC surface of {owner}; the "
            "shape gradient of that part of the surface loses accuracy. A plane crossing a "
            "structure costs little, but one close above or beside its surface does: keep "
            f"it at least {float(np.max(reach)):.3g} um from traced PEC surfaces, or refine "
            "the grid between them."
        )


def _warn_collar_media(
    structure: Structure,
    simulation: Simulation,
    cell_centers: np.ndarray,
    cell_sizes: np.ndarray,
    medium_index: np.ndarray,
) -> None:
    """Warn when the collar's cells hold media its stress does not model.

    The stress takes one permittivity per medium (a spatially varying medium's spatial
    average, see ``shape_terms._collar_kernel``), assumes a reciprocal medium,
    and has no term for a 2D sheet's surface current where a non-PEC sheet crosses the
    collar. Sheets are found by their bounding box, so a cell beside a non-rectangular
    sheet may also count.
    """
    from tidy3d.components.medium import Medium2D
    from tidy3d.log import log

    issues = []
    media = [structure_medium(simulation, index) for index in np.unique(medium_index)]
    if any(not medium.is_spatially_uniform for medium in media):
        issues.append("a spatially varying medium (taken as uniform at its average permittivity)")
    if any(
        medium.is_time_modulated
        or (
            medium.is_fully_anisotropic
            and not np.allclose(medium.conductivity, np.transpose(medium.conductivity))
        )
        for medium in media
    ):
        issues.append("a non-reciprocal medium (time-modulated or magneto-optic)")
    for other in simulation.structures:
        if (
            other is structure
            or not isinstance(other.medium, Medium2D)
            or is_pec_surface(other.medium, simulation)
        ):
            continue
        sheet_min, sheet_max = (np.asarray(b, dtype=float) for b in other.geometry.bounds)
        axis = int(np.argmin(sheet_max - sheet_min))
        plane = np.delete(np.arange(3), axis)
        # a sheet on a cell face (where the grid usually puts it) is half a cell from both
        crossed = np.abs(cell_centers[:, axis] - sheet_min[axis]) <= 0.5 * cell_sizes[:, axis] * (
            1.0 + 1e-6
        )
        crossed &= np.all(cell_centers[:, plane] >= sheet_min[plane], axis=1)
        crossed &= np.all(cell_centers[:, plane] <= sheet_max[plane], axis=1)
        if np.any(crossed):
            issues.append(f"the 2D sheet of structure '{structure_label(other)}'")
    if issues:
        log.warning(
            "Adjoint collar: the collar around the PEC surface of structure "
            f"'{structure_label(structure)}' holds "
            f"{'; '.join(issues)}. Its stress does not model these, so the shape gradient of "
            "that surface may be inaccurate. Keep such media at least a few grid cells from "
            "traced PEC surfaces."
        )


def _warn_absorber_crossing(
    structure: Structure, simulation: Simulation, absorbing_sides: list[tuple[int, int]]
) -> None:
    """Warn when the structure runs past an absorbing domain side, where its motion is lost."""
    from tidy3d.log import log

    geometry_min, geometry_max = structure.geometry.bounds
    bounds_min, bounds_max = simulation.bounds
    if any(
        geometry_min[axis] < bounds_min[axis]
        if side == 0
        else geometry_max[axis] > bounds_max[axis]
        for axis, side in absorbing_sides
    ):
        log.warning(
            f"Adjoint collar: structure '{structure_label(structure)}' extends "
            "past the simulation boundary into an absorbing boundary; its shape gradient "
            "covers only its surface inside the simulation domain, not its motion within the "
            "absorber."
        )
