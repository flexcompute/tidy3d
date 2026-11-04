"""Collar (exterior stress sleeve) construction for structures that move a PEC surface.

Classifies each surface sample against the effective PEC surface (on it, collar owned, or on
no surface), selects the exterior Yee cells within the collar's reach, cuts the cells at
sources, finds the foreign metal around the collar, and records the medium holding each
cell. The result is the structure's ``CollarSamplingData``; the pre-run accuracy checks live
in ``collar_diagnostics``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flexcompute.core._migration.em.numerical.raw.collar import _EDGE_COS as COLLAR_EDGE_COS
from flexcompute.core._migration.em.numerical.raw.collar import (
    _EDGE_NEIGHBORS as COLLAR_EDGE_NEIGHBORS,
)
from flexcompute.core._migration.em.numerical.raw.collar import cells_near_source, mirror_images

from tidy3d.config import config
from tidy3d.em.translate.sample_sets import CollarSamplingData, IndexedDataArray
from tidy3d.exceptions import AdjointError
from tidy3d.system import get_available_memory_bytes

from .collar_diagnostics import (
    _warn_absorber_crossing,
    _warn_collar_media,
    _warn_collar_shape,
    collar_column_obstacles,
    collar_expansion,
    structure_label,
    warn_near_cut,
)
from .staging import (
    _point_data_array,
    _precedence_order,
    effective_is_pec_at_points,
    grid_widths_near,
    is_pec_surface,
    sheet_opened_at,
)

if TYPE_CHECKING:
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.structure import Structure
    from tidy3d.em.translate.sample_sets import (
        PECSamplingData,
        PointDataArray,
        SurfaceSampleSet,
    )

    from .spacing import SamplingScanIndex
    from .types import PathType

# collar (exterior stress sleeve) defaults, in cells across the surface (see
# ``_collar_cell_scale``): the velocity is rigid for one cell beyond the surface (where
# colocated Yee fields are least reliable) and tapers to zero over the next three
COLLAR_OFFSET_CELLS = 1.0
COLLAR_TAPER_CELLS = 3.0
# distance of the surface-activity probes from each sample, in its cell's extent along the
# sample normal: well above the chord error of tessellated curved surfaces, well below any
# feature the grid resolves along the normal (a thin film on a grid fine only across it)
COLLAR_PROBE_CELLS = 0.1
# a PEC boundary face of the collar farther than this many local cells from the structure's
# active samples is foreign surface (other structures' metal, or surfaces another structure's
# faces carve), where the velocity is cut; a closer one is its own unless its two cells lie
# on the same side of the structure's geometry (see ``_foreign_pec_faces``)
COLLAR_OWN_FACE_CELLS = 1.0
# a non-PEC structure's own interface samples this close to metal (in local cells) are
# collar owned, the collar's whole velocity support: near a metal edge the interface's
# derivative is singular and cancels most of the metal side's, which the collar integrates,
# so both halves must come from the same volume integral (with a one-cell handover the
# boundary integrand took the air side, and the two disagreed)
COLLAR_NEAR_METAL_CELLS = COLLAR_OFFSET_CELLS + COLLAR_TAPER_CELLS
# peak memory of collar staging per candidate cell of the structure's expanded bounding box
# (measured ~225 B on a solid PEC box): the candidates are materialized densely
COLLAR_BYTES_PER_CANDIDATE = 250


def invariant_axes(simulation: Simulation) -> tuple[int, ...]:
    """The simulation's zero-size axes, along which the fields do not vary (2D)."""
    return tuple(axis for axis, size in enumerate(simulation.size) if size == 0)


def _sheet_masks(
    normals: np.ndarray,
    pec: PECSamplingData,
    is_pec_structure: bool,
    flat: np.ndarray,
    on_sheet: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(on_surface, removed)`` for the samples of a zero-thickness structure, whose ``flat``
    axes have no extent (see :func:`_structure_active_surface`). ``on_sheet`` marks the
    samples inside the structure's geometry: the side probes of a face sample see open air
    on both sides of a hole cut out of the sheet as on the metal, so only membership tells
    them apart."""
    num = normals.shape[0]
    outside = np.asarray(pec.outside.pec_mask.values) > 0
    inside = np.asarray(pec.inside.pec_mask.values) > 0
    in_plane = np.all(np.abs(normals[:, flat]) < 0.5, axis=1)
    if not is_pec_structure:
        # a non-PEC sheet bounds no metal across its plane, but its in-plane edge can meet
        # metal continuing the plane (a PEC sheet): moving that edge moves the metal's edge, so
        # those edge samples are collar owned, as the PEC sheet's own samples are; the rest
        # keep their boundary integrands
        return in_plane & outside & ~inside, np.zeros(num, dtype=bool)
    surface = np.where(in_plane, outside ^ inside, ~outside & on_sheet)
    removed = np.where(in_plane, ~outside & ~inside, ~on_sheet)
    return surface, removed


def _structure_active_surface(
    staged: dict[PathType, SurfaceSampleSet],
    structure: Structure,
    structure_index: int,
    simulation: Simulation,
    scan_index: SamplingScanIndex,
    grid: Grid,
    near_metal_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated surface points of the staged sets, and which are on the PEC surface and
    which the collar moves.

    A sample is on the PEC surface when moving it moves the effective PEC surface: of its two probes
    ``p ± probe n`` (``probe`` a fraction of its cell's extent along ``n``), exactly one is PEC
    in the effective medium, and the structure owns the side it would grow into. For a PEC structure that is the PEC side, inside its own
    geometry. For a non-PEC structure (a dielectric carving a PEC) it is the non-PEC side,
    inside its own geometry and not overridden by a structure of higher precedence. This drops
    faces buried in other PEC (PEC on both sides), faces carved away by such a structure or
    lying in open air (no PEC on either side), and operand faces a boolean operation
    removed, independent of the orientation of an operand's normals (a subtracted box's
    normals point into the metal). Zero-thickness PEC structures have no metal across their
    own plane, whose samples keep the side-probe rule (inactive where the outside is PEC)
    and are removed outside the sheet's geometry (across a hole cut out of it);
    their in-plane edges follow the bulk rule on the in-plane probes (exactly one side PEC),
    so the edge of a hole a boolean operation cut out of a sheet, whose outside probe lies
    in the remaining metal, stays active, and an edge in open air is removed. Zero-thickness
    non-PEC structures bound no PEC across their plane, only at an in-plane edge that meets
    metal continuing the plane, whose samples are then active.

    Returns ``(points, on_surface, active, removed, outward)``: ``on_surface`` marks the samples
    on the PEC surface (the structure's own surface, used to tell it from foreign metal),
    ``active`` the samples whose forces come from the collar (the kernel's ``active_mask``), and
    ``removed`` the
    samples on no surface at all (no metal on either side for a PEC structure, not an
    interface of a non-PEC one), which must not capture collar cells. The two agree for PEC
    structures; a non-PEC structure's collar also owns its interface samples within
    ``near_metal_distance`` of the metal, whose boundary probes fall in PEC-averaged cells.
    ``outward`` is the sign turning each sample's normal toward its side that is not metal
    (the collar's side), from the same probes: -1 where the normal points into the metal (a
    carved surface's), +1 otherwise (a sheet's two sides are both open).
    """
    points, on_surface, active, removed, outward_sign = [], [], [], [], []
    bounds_min, bounds_max = (np.asarray(b, dtype=float) for b in structure.geometry.bounds)
    flat = bounds_max - bounds_min <= 0.0
    sheet = bool(np.any(flat))
    is_pec_structure = is_pec_surface(structure.medium, simulation)
    order = _precedence_order(simulation)
    later = [simulation.structures[i] for i in order[order.index(structure_index) + 1 :]]
    planar = [a for a in range(3) if a not in invariant_axes(simulation)]
    skip = list(invariant_axes(simulation))
    for sample_set in staged.values():
        if sample_set.num_points == 0:
            continue
        sample_points = np.asarray(sample_set.points.values, dtype=float)
        points.append(sample_points)
        normals = np.asarray(sample_set.normals.values, dtype=float)
        probe = (COLLAR_PROBE_CELLS * _extent_along(grid, sample_points, normals, skip))[:, None]
        if sheet:
            # on the sheet's plane, where the samples lie up to their float precision
            in_plane_points = np.where(flat, bounds_min, sample_points)
            on_sheet = structure.geometry.inside(*in_plane_points.T)
            surface, gone = _sheet_masks(
                normals, sample_set.staging.pec_sampling, is_pec_structure, flat, on_sheet
            )
            on_surface.append(surface)
            active.append(surface)
            removed.append(gone)
            outward_sign.append(np.ones(sample_set.num_points))
            continue
        sides = []
        for sign in (-1.0, 1.0):
            probes = sample_points + sign * probe * normals
            x, y, z = probes[:, 0], probes[:, 1], probes[:, 2]
            is_pec = effective_is_pec_at_points(probes, simulation, scan_index) > 0
            own = structure.geometry.inside(x, y, z)
            if not is_pec_structure:
                for other in later:
                    own &= ~other.geometry.inside(x, y, z)
            sides.append((is_pec, own))
        (pec_minus, own_minus), (pec_plus, own_plus) = sides
        outward_sign.append(np.where(pec_plus & ~pec_minus, -1.0, 1.0))
        if is_pec_structure:
            on_surface.append(
                (pec_minus & own_minus & ~pec_plus) | (pec_plus & own_plus & ~pec_minus)
            )
            active.append(on_surface[-1])
            removed.append(~(pec_minus | pec_plus))
        else:
            on_surface.append(
                (own_minus & ~pec_minus & pec_plus) | (own_plus & ~pec_plus & pec_minus)
            )
            # the structure's own non-PEC interface next to the metal: the boundary
            # integrand's probes there fall in PEC-averaged cells, so the collar (which
            # keeps the jump of the structure's own interfaces) takes these samples too
            interface = own_minus ^ own_plus
            removed.append(~interface)
            # probe from just off the face on its non-own side: probes in the face plane
            # would land on the structure's own boundary, where it takes precedence
            outward = np.where(own_minus, 1.0, -1.0)[:, None] * normals
            base = sample_points + probe * outward
            near_metal = np.zeros(sample_set.num_points, dtype=bool)
            # every cell out to the distance, so thin metal is not stepped over
            steps = max(1, int(np.ceil(COLLAR_NEAR_METAL_CELLS)))
            for distance in near_metal_distance * np.arange(1, steps + 1) / steps:
                for axis in planar:
                    for sign in (-1.0, 1.0):
                        probes = base.copy()
                        probes[:, axis] += sign * distance
                        near_metal |= effective_is_pec_at_points(probes, simulation, scan_index) > 0
            active.append(on_surface[-1] | (interface & near_metal))
    if not points:
        empty = np.zeros(0, dtype=bool)
        return np.zeros((0, 3)), empty, empty, empty, np.zeros(0)
    return (
        np.concatenate(points),
        np.concatenate(on_surface),
        np.concatenate(active),
        np.concatenate(removed),
        np.concatenate(outward_sign),
    )


def _indexed(values: np.ndarray) -> IndexedDataArray:
    """One value per sample or cell, in order."""
    return IndexedDataArray(values, coords={"index": np.arange(values.shape[0])})


def _local_cell_size(grid: Grid, bounds: tuple, skip: list[int]) -> float:
    """Coarsest grid spacing among the cells a structure's surface touches.

    The collar's rigid offset must cover at least one full cell in every direction (the
    colocated fields in the cell touching the PEC are the ones the kernel keeps out of
    the taper) and its taper must span a few cells, so on a nonuniform grid the scale is
    set by the coarsest spacing around the structure, not the refined one: a layer
    refined only in-plane still has whole coarse cells stacked against its faces. Axes in
    ``skip`` (a 2D simulation's zero-size axis) are left out.
    """
    return max(
        float(np.max(widths[touching])) if np.any(touching) else float(np.min(widths))
        for axis, (widths, touching) in enumerate(grid_widths_near(grid, bounds))
        if axis not in skip
    )


def _neighbor_widths(grid: Grid, points: np.ndarray, axis: int) -> np.ndarray:
    """``(N, 3)`` widths along ``axis`` of the cell holding each point and its two neighbors."""
    bounds = np.asarray(getattr(grid.boundaries, "xyz"[axis]), dtype=float)
    widths = np.diff(bounds)
    index = np.clip(np.searchsorted(bounds, points[:, axis]) - 1, 0, widths.size - 1)
    return np.stack([widths[np.clip(index + step, 0, widths.size - 1)] for step in (-1, 0, 1)], 1)


def _sample_cell_sizes(grid: Grid, points: np.ndarray, skip: list[int]) -> np.ndarray:
    """Coarsest grid step around each point: its cell and the neighbors, along every axis
    but those in ``skip``. The collar's reach adds it, so the taper ends inside the collar
    even where the cells along the surface are coarse.
    """
    sizes = np.zeros(points.shape[0])
    for axis in range(3):
        if axis not in skip:
            sizes = np.maximum(sizes, _neighbor_widths(grid, points, axis).max(axis=1))
    return sizes


def _collar_cell_scale(
    grid: Grid, points: np.ndarray, normals: np.ndarray, skip: list[int]
) -> np.ndarray:
    """Per sample: the cell scale of the collar's rigid depth and taper.

    The rigid depth must cover the cells touching the surface across it: the widest extent
    of the touching cells along the sample's normal and along the normals the surface turns
    to at an edge or corner next to it (the cells wedged beside an edge touch it too). The
    cells along the surface do not matter, so on an anisotropic grid (a strip refined only
    across its edges, a film refined only across its thickness) the collar stays a few fine
    cells deep instead of a few coarse ones. On a uniform grid this is the grid step. Where
    the samples lie farther apart than the cells across the surface, the scale grows so the
    rigid depth closes the gaps between them (it is a union of balls around the samples).
    """
    from scipy.spatial import cKDTree

    widths = np.zeros(points.shape)
    for axis in range(3):
        if axis not in skip:
            widths[:, axis] = _neighbor_widths(grid, points, axis).max(axis=1)

    def extent(direction: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(direction) * widths, axis=1)

    scale = extent(normals)
    spacing = np.zeros(points.shape[0])
    if points.shape[0] > 1:
        distance, neighbors = cKDTree(points).query(
            points, k=min(COLLAR_EDGE_NEIGHBORS, points.shape[0])
        )
        for column in range(1, neighbors.shape[1]):
            other = normals[neighbors[:, column]]
            turned = np.abs(np.sum(other * normals, axis=1)) < COLLAR_EDGE_COS
            scale = np.where(turned, np.maximum(scale, extent(other)), scale)
            # spacing to the nearest sample of the same surface (a sheet's opposite face
            # coincides with it)
            neighbor = ~turned & (distance[:, column] > 0.0)
            spacing = np.where(neighbor & (spacing == 0.0), distance[:, column], spacing)
    # a normal along skipped axes only (a face parallel to a 2D plane) crosses no cell
    scale = np.where(scale > 0.0, scale, _sample_cell_sizes(grid, points, skip))
    return np.maximum(scale, spacing / np.sqrt(2.0))


def _extent_along(
    grid: Grid, points: np.ndarray, normals: np.ndarray, skip: list[int]
) -> np.ndarray:
    """Each point's cell extent along its normal: the distance along it across the narrowest of
    the cells it touches, axes in ``skip`` left out. A face of a film resolved only across its
    thickness gets the fine step, not the coarse in-plane one."""
    extent = np.full(points.shape[0], np.inf)
    for axis in range(3):
        if axis in skip:
            continue
        along = np.abs(normals[:, axis])
        width = _neighbor_widths(grid, points, axis).min(axis=1)
        extent = np.where(
            along > 0.0, np.minimum(extent, width / np.maximum(along, 1e-300)), extent
        )
    # a normal along skipped axes only (a face parallel to a 2D plane) crosses no cell
    return np.where(np.isfinite(extent), extent, _sample_cell_sizes(grid, points, skip))


def near_bounds(
    cell_centers: np.ndarray,
    cell_sizes: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    simulation: Simulation,
    surface_only: bool = False,
) -> np.ndarray:
    """Collar cells near a box (see :func:`cells_near_source`) or its images across the
    simulation's symmetry planes."""
    near = np.zeros(cell_centers.shape[0], dtype=bool)
    for lower, upper in mirror_images(bounds, simulation.symmetry, simulation.center):
        near |= cells_near_source(cell_centers, cell_sizes, lower, upper, surface_only)
    return near


def source_cut(
    cell_centers: np.ndarray, cell_sizes: np.ndarray, simulation: Simulation
) -> np.ndarray:
    """Collar cells cut by the simulation's sources, mirrored ones included; a total-field /
    scattered-field source injects only on the surface of its box."""
    from tidy3d.components.source.field import TFSF

    cut = np.zeros(cell_centers.shape[0], dtype=bool)
    for source in simulation.sources:
        cut |= near_bounds(
            cell_centers, cell_sizes, source.bounds, simulation, isinstance(source, TFSF)
        )
    return cut


def _domain_sides(simulation: Simulation) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """``(closed, absorbing)`` domain sides as ``(axis, side)``, ``side`` 0 for the minimum and
    1 for the maximum: PEC / PMC walls, and absorbing boundaries (PML, absorbers, ABCs).

    A closed side is fixed metal next to the collar, a foreign surface like any other PEC
    (on a PEC or PMC wall the stress traction is normal, so cutting the velocity along the
    wall's normal removes its term). An open side (absorbing, periodic, Bloch) only stops the
    collar: a face without a neighbor sees zero stress beyond, so the face sums take the
    stress flux through that plane, the identity's term for a velocity dropping to zero
    there, which gives the gradient of the surface inside the domain. A zero-size axis has
    no sides.
    """
    from tidy3d.components.boundary import (
        AbsorberSpec,
        AbstractABCBoundary,
        PECBoundary,
        PMCBoundary,
    )

    closed, absorbing = [], []
    for axis, dim in enumerate("xyz"):
        if axis in invariant_axes(simulation):
            continue
        boundary = getattr(simulation.boundary_spec, dim)
        for side, edge in enumerate((boundary.minus, boundary.plus)):
            if isinstance(edge, (PECBoundary, PMCBoundary)):
                closed.append((axis, side))
            elif isinstance(edge, (AbsorberSpec, AbstractABCBoundary)):
                absorbing.append((axis, side))
    return closed, absorbing


def _collar_medium_structure_index(points: np.ndarray, simulation: Simulation) -> np.ndarray:
    """Index of the highest-precedence non-PEC (3D) structure containing each point, ``-1`` otherwise.

    Precedence follows the simulation: priority, then list order. PEC structures never
    hold collar cells and 2D media have no volume, so both are skipped.
    """
    from tidy3d.components.medium import Medium2D

    index = np.full(points.shape[0], -1.0)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    for structure_index in _precedence_order(simulation):
        structure = simulation.structures[structure_index]
        if is_pec_surface(structure.medium, simulation) or isinstance(structure.medium, Medium2D):
            continue
        inside = structure.geometry.inside(x, y, z)
        index[inside] = float(structure_index)
    return index


def _flat_interface_faces(
    cell_indices: np.ndarray,
    cell_centers: np.ndarray,
    cell_sizes: np.ndarray,
    medium_index: np.ndarray,
    simulation: Simulation,
    skip: list[int],
) -> np.ndarray:
    """Per collar cell and axis, ``(N, 3)``: the face to its upper neighbor along the axis lies
    on a dielectric interface flat and normal to that axis (the kernel's ``flat_interface``).

    Between two collar cells held by different structures the interface is found along the
    axis by bisection on the holding structure at the face's center and near its four
    corners; it is flat and normal to the axis where all five crossings agree. An interface
    tilted against the grid (a staircase) crosses at different heights and keeps the whole
    jump dropped.
    """
    flat = np.zeros(cell_centers.shape, dtype=bool)
    lookup = {tuple(idx): n for n, idx in enumerate(cell_indices.astype(np.int64).tolist())}
    corners = np.array([(0.0, 0.0), (-1, -1), (-1, 1), (1, -1), (1, 1)])

    def held(columns: np.ndarray, axis: int, heights: np.ndarray) -> np.ndarray:
        """The structure holding each probe column at the given heights along ``axis``."""
        points = columns.copy()
        points[:, :, axis] = heights
        return _collar_medium_structure_index(points.reshape(-1, 3), simulation).reshape(
            heights.shape
        )

    for axis in range(3):
        if axis in skip:
            continue
        step = np.eye(3, dtype=np.int64)[axis]
        upper = np.array(
            [lookup.get(tuple(idx + step), -1) for idx in cell_indices.astype(np.int64)],
            dtype=np.int64,
        )
        lower = np.nonzero(upper >= 0)[0]
        lower = lower[medium_index[lower] != medium_index[upper[lower]]]
        if lower.size == 0:
            continue
        below, above = medium_index[lower], medium_index[upper[lower]]
        others = [a for a in range(3) if a != axis]
        # probe columns through the face center and 0.45 of a cell toward each corner
        columns = np.repeat(cell_centers[lower][:, None, :], len(corners), axis=1)
        for k, other in enumerate(others):
            if other not in skip:
                columns[:, :, other] += (
                    0.45 * corners[None, :, k] * cell_sizes[lower][:, None, other]
                )
        start = cell_centers[lower][:, axis][:, None] * np.ones(len(corners))
        stop = cell_centers[upper[lower]][:, axis][:, None] * np.ones(len(corners))
        ok = np.all(held(columns, axis, start) == below[:, None], axis=1) & np.all(
            held(columns, axis, stop) == above[:, None], axis=1
        )
        lo, hi = start.copy(), stop.copy()
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            at = held(columns, axis, mid)
            ok &= np.all((at == below[:, None]) | (at == above[:, None]), axis=1)
            lo = np.where(at == below[:, None], mid, lo)
            hi = np.where(at == below[:, None], hi, mid)
        crossing = 0.5 * (lo + hi)
        spread = np.max(crossing, axis=1) - np.min(crossing, axis=1)
        flat[lower, axis] = ok & (spread <= 1e-3 * (stop[:, 0] - start[:, 0]))
    return flat


def _candidate_cells(
    grid: Grid, points: np.ndarray, reach: float, owner: str
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray] | None:
    """The grid box around ``points`` expanded by ``reach``: its shape and the indices,
    centers and sizes of its cells (C order), or ``None`` when it holds no cell.

    Raises ``AdjointError`` before materializing a box that would exceed the adjoint memory
    budget (``config.adjoint.memory_allotment_fraction`` of the available memory).
    """
    lo, hi = points.min(axis=0) - reach, points.max(axis=0) + reach
    axes_idx, axes_centers, axes_sizes = [], [], []
    for axis, dim in enumerate("xyz"):
        centers = np.asarray(getattr(grid.centers, dim), dtype=float)
        widths = np.asarray(getattr(grid.sizes, dim), dtype=float)
        select = np.nonzero((centers >= lo[axis]) & (centers <= hi[axis]))[0]
        axes_idx.append(select)
        axes_centers.append(centers[select])
        axes_sizes.append(widths[select])
    if any(sel.size == 0 for sel in axes_idx):
        return None
    shape = tuple(sel.size for sel in axes_idx)
    # ponytail: dense box, so memory scales with the enclosed volume; surface-local candidates
    # would scale with the collar's
    needed = COLLAR_BYTES_PER_CANDIDATE * int(np.prod(shape, dtype=np.int64))
    available = get_available_memory_bytes()
    if available > 0 and needed > config.adjoint.memory_allotment_fraction * available:
        raise AdjointError(
            f"The PEC collar of {owner} spans {shape[0]} x {shape[1]} x {shape[2]} grid cells "
            f"around the structure, needing about {needed / 1e9:.1f} GB to stage, more than the "
            f"{config.adjoint.memory_allotment_fraction:.0%} of available memory "
            "('config.adjoint.memory_allotment_fraction') the adjoint may use. Coarsen the grid "
            "around the structure or split it into smaller structures."
        )

    def flat(values: list[np.ndarray]) -> np.ndarray:
        return np.stack([g.ravel() for g in np.meshgrid(*values, indexing="ij")], axis=1)

    return shape, flat(axes_idx), flat(axes_centers), flat(axes_sizes)


def _cut_sources_and_flag_monitors(
    structure: Structure,
    simulation: Simulation,
    keep: np.ndarray,
    cell_centers: np.ndarray,
    cell_sizes: np.ndarray,
    active_points: np.ndarray,
    active_normals: np.ndarray,
    taper_reach: np.ndarray,
) -> np.ndarray:
    """``keep`` with the cells at sources cut. Monitors may carry adjoint sources, cut at
    consumption, so the ones in a sample's collar column, which cost accuracy, are only
    flagged here."""
    owner = f"structure '{structure_label(structure)}'"
    cut = keep & source_cut(cell_centers, cell_sizes, simulation)
    warn_near_cut(
        owner,
        active_points,
        active_normals,
        taper_reach,
        cell_centers[cut],
        cell_sizes[cut],
        "a source",
    )
    keep = keep & ~cut
    for monitor in simulation.monitors:
        near = keep & near_bounds(cell_centers, cell_sizes, monitor.bounds, simulation)
        warn_near_cut(
            owner,
            active_points,
            active_normals,
            taper_reach,
            cell_centers[near],
            cell_sizes[near],
            f"monitor '{monitor.name}' (if the objective depends on it)",
        )
    return keep


def _foreign_pec_faces(
    structure: Structure,
    shape: tuple[int, int, int],
    is_pec: np.ndarray,
    keep: np.ndarray,
    cell_centers: np.ndarray,
    cell_sizes: np.ndarray,
    closed_sides: list[tuple[int, int]],
    simulation: Simulation,
    surface_points: np.ndarray,
    local_cell: float,
) -> tuple[IndexedDataArray | None, PointDataArray | None]:
    """Distance and normal from each kept cell to the nearest foreign PEC face, or ``None``s.

    Foreign faces are faces between a collar cell and a PEC cell (or a PEC / PMC domain
    wall) that are not the structure's own surface: other structures' metal, surfaces another
    structure's faces carve into this one, walls. A face is the structure's own only within
    ``COLLAR_OWN_FACE_CELLS`` of its surface samples, and, between two cells, only where the
    cells lie on opposite sides of its geometry: metal across a gap (both cells outside) or
    a carved surface (both inside) is foreign however close, on any grid. A wall face has no
    cell across and is the structure's own only where its surface lies on the wall.
    """
    from scipy.spatial import cKDTree

    bounds_min, bounds_max = (np.asarray(b, dtype=float) for b in simulation.bounds)
    pec_grid, keep_grid = is_pec.reshape(shape), keep.reshape(shape)
    centers_grid, sizes_grid = cell_centers.reshape((*shape, 3)), cell_sizes.reshape((*shape, 3))
    flat_index = np.arange(is_pec.size).reshape(shape)
    boundary_faces, boundary_normals, collar_cells, across_cells = [], [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            neighbor_pec = np.zeros(shape, dtype=bool)
            neighbor_index = np.full(shape, -1)
            source, target = [slice(None)] * 3, [slice(None)] * 3
            source[axis] = slice(1, None) if sign > 0 else slice(0, -1)
            target[axis] = slice(0, -1) if sign > 0 else slice(1, None)
            neighbor_pec[tuple(target)] = pec_grid[tuple(source)]
            neighbor_index[tuple(target)] = flat_index[tuple(source)]
            if (axis, int(sign > 0)) in closed_sides:
                wall = bounds_max[axis] if sign > 0 else bounds_min[axis]
                face = centers_grid[..., axis] + sign * 0.5 * sizes_grid[..., axis]
                neighbor_pec |= np.isclose(face, wall, rtol=0.0, atol=1e-6 * local_cell)
            on_boundary = keep_grid & neighbor_pec
            face_centers = centers_grid[on_boundary].copy()
            face_centers[:, axis] += sign * 0.5 * sizes_grid[on_boundary][:, axis]
            boundary_faces.append(face_centers)
            collar_cells.append(flat_index[on_boundary])
            across_cells.append(neighbor_index[on_boundary])
            normal = np.zeros(3)
            normal[axis] = sign
            boundary_normals.append(np.tile(normal, (face_centers.shape[0], 1)))
    boundary_faces = np.concatenate(boundary_faces)
    boundary_normals = np.concatenate(boundary_normals)
    foreign_faces, foreign_normals = _foreign_sheet_faces(
        structure, simulation, cell_centers[keep], cell_sizes[keep], local_cell
    )
    if boundary_faces.shape[0]:
        collar_cells, across_cells = np.concatenate(collar_cells), np.concatenate(across_cells)
        between_cells = across_cells >= 0
        cells = np.unique(np.concatenate([collar_cells, across_cells[between_cells]]))
        inside = np.zeros(is_pec.size, dtype=bool)
        inside[cells] = structure.geometry.inside(*cell_centers[cells].T)
        same_side = between_cells & (inside[collar_cells] == inside[np.maximum(across_cells, 0)])
        own_distance, _ = cKDTree(surface_points).query(boundary_faces, k=1)
        is_foreign = same_side | (own_distance > COLLAR_OWN_FACE_CELLS * local_cell)
        foreign_faces = np.concatenate([foreign_faces, boundary_faces[is_foreign]])
        foreign_normals = np.concatenate([foreign_normals, boundary_normals[is_foreign]])
    if not foreign_faces.shape[0]:
        return None, None
    distance, nearest = cKDTree(foreign_faces).query(cell_centers[keep], k=1)
    return _indexed(distance), _point_data_array(foreign_normals[nearest])


def _foreign_sheet_faces(
    structure: Structure,
    simulation: Simulation,
    centers: np.ndarray,
    sizes: np.ndarray,
    local_cell: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Faces of the collar cells on another structure's zero-thickness PEC sheet, with the
    sheet's normal.

    A sheet holds no cell center, so no PEC cell finds it; without these faces the collar
    would run through it and take the sheet's traction as the structure's own. The cells on
    both sides share each face, so the normal's sign is arbitrary; the cutoff uses it only
    as an axis.
    """
    faces, normals = [np.zeros((0, 3))], [np.zeros((0, 3))]
    for other_index, other in enumerate(simulation.structures):
        if other is structure or not is_pec_surface(other.medium, simulation):
            continue
        sheet_min, sheet_max = (np.asarray(b, dtype=float) for b in other.geometry.bounds)
        axis = int(np.argmin(sheet_max - sheet_min))
        if sheet_max[axis] > sheet_min[axis]:
            continue
        toward = np.abs(sheet_min[axis] - centers[:, axis])
        near = np.flatnonzero(toward <= 0.5 * sizes[:, axis] + 1e-6 * local_cell)
        face = centers[near].copy()
        face[:, axis] = sheet_min[axis]
        on_sheet = other.geometry.inside(*face.T) & ~sheet_opened_at(
            face, axis, 0.5 * sizes[near, axis], other_index, simulation
        )
        faces.append(face[on_sheet])
        normals.append(np.tile(np.eye(3)[axis], (faces[-1].shape[0], 1)))
    return np.concatenate(faces), np.concatenate(normals)


def collar_sampling_data(
    staged: dict[PathType, SurfaceSampleSet],
    structure: Structure,
    structure_index: int,
    simulation: Simulation,
    scan_index: SamplingScanIndex,
    grid: Grid,
) -> CollarSamplingData | None:
    """Collar (exterior stress sleeve) payload for one structure, or ``None``.

    Any structure whose motion moves the effective PEC surface gets a collar: a PEC
    structure, or a non-PEC structure carving a PEC (see :func:`_structure_active_surface`).
    The collar is every Yee cell whose center lies within ``offset + taper`` (plus one
    cell so the taper reaches zero inside the collar) of the structure's active surface
    samples, is not PEC, and lies inside the physical domain (see :func:`_domain_sides` for
    its sides). PEC boundary faces of the collar far from those samples, PEC / PMC domain
    walls included, are foreign surface and cut the velocity normal to them there. Sources
    cut their cells out of the collar (adjoint sources at consumption, where the adjoint
    simulation is known); monitors next to the surface are flagged here, since an adjoint
    source on them would leave that part of the surface unresolved.
    """
    from scipy.spatial import cKDTree

    # the structure's surface samples: on the PEC surface, owned by the collar, or on none
    invariant = list(invariant_axes(simulation))
    local_cell = _local_cell_size(grid, structure.geometry.bounds, invariant)
    all_points, on_surface, active, removed, outward_sign = _structure_active_surface(
        staged,
        structure,
        structure_index,
        simulation,
        scan_index,
        grid,
        COLLAR_NEAR_METAL_CELLS * local_cell,
    )
    if all_points.shape[0] == 0 or not np.any(on_surface):
        return None
    active_points = all_points[active]
    staged_sets = [s for s in staged.values() if s.num_points > 0]
    all_normals = np.concatenate([np.asarray(s.normals.values, dtype=float) for s in staged_sets])
    active_normals = all_normals[active]
    active_weights = np.concatenate(
        [np.asarray(s.weights.values, dtype=float) for s in staged_sets]
    )[active]

    # the collar's depth and taper follow the grid across the surface at each sample, so a
    # surface shared by two structures gets the same collar from either one, whatever grid
    # their far sides see
    sample_cell = _sample_cell_sizes(grid, all_points, invariant)
    sample_scale = _collar_cell_scale(grid, all_points, all_normals, invariant)
    sample_offset = COLLAR_OFFSET_CELLS * sample_scale
    sample_taper = COLLAR_TAPER_CELLS * sample_scale
    sample_reach = sample_offset + sample_taper + sample_cell
    taper_reach = sample_offset[active] + sample_taper[active]
    closed_sides, absorbing_sides = _domain_sides(simulation)
    _warn_absorber_crossing(structure, simulation, absorbing_sides)

    # the collar: cells within reach of the active surface, inside the domain, not PEC, and
    # clear of sources
    candidates = _candidate_cells(
        grid,
        active_points,
        float(np.max(sample_reach[active])),
        f"structure '{structure_label(structure)}'",
    )
    if candidates is None:
        return None
    shape, cell_indices, cell_centers, cell_sizes = candidates
    # a zero-size axis holds one cell: place it in the simulation plane, with the samples
    cell_centers[:, invariant] = np.asarray(simulation.center, dtype=float)[invariant]
    bounds_min, bounds_max = (np.asarray(b, dtype=float) for b in simulation.bounds)
    inside = (cell_centers > bounds_min) & (cell_centers < bounds_max)
    inside[:, invariant] = True
    interior = np.all(inside, axis=1)
    is_pec = effective_is_pec_at_points(cell_centers, simulation, scan_index) > 0
    # Interior PEC cells can dominate the candidate volume but never enter the collar.
    keep = interior & ~is_pec
    exterior = np.flatnonzero(keep)
    distance, nearest = cKDTree(active_points).query(cell_centers[exterior], k=1)
    keep[exterior] &= distance <= sample_reach[active][nearest]
    keep = _cut_sources_and_flag_monitors(
        structure,
        simulation,
        keep,
        cell_centers,
        cell_sizes,
        active_points,
        active_normals,
        taper_reach,
    )
    if not np.any(keep):
        return None

    # checks before the run: metal in front of the surface, and the surface curving within
    # the collar's depth
    outward = outward_sign[active][:, None] * active_normals
    _warn_collar_shape(
        structure,
        active_weights,
        collar_column_obstacles(
            active_points,
            outward,
            sample_scale[active],
            sample_offset[active],
            simulation,
            scan_index,
            closed_sides,
        ),
        collar_expansion(active_points, outward, active_weights, taper_reach),
        COLLAR_OFFSET_CELLS,
    )

    pec_neighbor_distance, pec_neighbor_normal = _foreign_pec_faces(
        structure,
        shape,
        is_pec,
        keep,
        cell_centers,
        cell_sizes,
        closed_sides,
        simulation,
        all_points[on_surface],
        local_cell,
    )
    medium_index = _collar_medium_structure_index(cell_centers[keep], simulation)
    _warn_collar_media(structure, simulation, cell_centers[keep], cell_sizes[keep], medium_index)
    flat_interface = _flat_interface_faces(
        cell_indices[keep],
        cell_centers[keep],
        cell_sizes[keep],
        medium_index,
        simulation,
        invariant,
    )
    return CollarSamplingData(
        points=_point_data_array(cell_centers[keep]),
        cell_indices=_point_data_array(cell_indices[keep].astype(float)),
        cell_sizes=_point_data_array(cell_sizes[keep]),
        pec_neighbor_distance=pec_neighbor_distance,
        pec_neighbor_normal=pec_neighbor_normal,
        on_surface=_indexed(on_surface),
        collar_owned=_indexed(active),
        removed=_indexed(removed & ~active),
        medium_structure_index=_indexed(medium_index),
        flat_interface=_point_data_array(flat_interface.astype(float)),
        sample_offset=_indexed(sample_offset),
        sample_taper=_indexed(sample_taper),
    )
