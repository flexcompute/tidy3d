"""Per-structure electromagnetic power loss, aggregated from loss-density monitor data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flexcompute.core._migration.em.numerical.raw import power_loss as loss_numerics

from tidy3d.components.data.data_array import StructureLossDataArray
from tidy3d.components.material.multi_physics import MultiPhysicsMedium
from tidy3d.components.medium.anisotropic import FullyAnisotropicMedium
from tidy3d.components.medium.lossy_metal import LossyMetalMedium
from tidy3d.components.subpixel_spec import SurfaceImpedance
from tidy3d.exceptions import DataError, SetupError
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.components.data.data_array import SpatialDataArray
    from tidy3d.components.data.monitor_data import FieldStructureData, SurfaceFieldStructureData
    from tidy3d.components.medium.base import AbstractMedium
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.structure import Structure


# off-diagonal conductivity below this fraction of the largest entry is rounding, not coupling
_OFF_DIAGONAL_RTOL = 1e-9
# a sample lies in the mirrored half when it is this far below a symmetry plane [um]
_MIRROR_PLANE_ATOL = 1e-9
# step into the metal, as a fraction of the smallest grid step, at which a vertex's owner is found
_SURFACE_OWNER_STEP = 1e-3
# volumetric samples relabeled per block, bounding the temporary arrays the relabeling needs
_RELABEL_BLOCK_SIZE = 250_000


def _check_owner_range(owners: np.ndarray, num_structures: int, data_name: str) -> np.ndarray:
    """Round ``owners`` to the nearest int (the same canonicalization ``_integrate_by_structure``
    and ``surface_loss_density`` use for aggregation/lookup) and raise a controlled 'DataError'
    for any value outside ``{-1} | [0, num_structures)``.

    Mirrors ``optical_generation._energy_gap_map``'s guard: ``-1`` (background) is the only valid
    negative value, never indexed into ``structures`` -- anything else negative is out of range
    too, not merely "not background". Validating the rounded owners (rather than the raw stored
    values) matters because a value that looks in-range before rounding (e.g. ``1.9`` in a
    two-structure simulation) can round up to an out-of-range index during aggregation and slip
    past a raw-value check undetected. Returns the rounded owners so callers use the exact same
    values this guard validated, rather than re-deriving them (and risking a second, possibly
    inconsistent, cast).
    """
    owners = np.round(np.asarray(owners)).astype(int)
    out_of_range = owners[(owners < -1) | (owners >= num_structures)]
    if out_of_range.size:
        raise DataError(
            f"Structure-ownership index {int(out_of_range.flat[0])} in '{data_name}' is out of "
            f"range for a simulation with {num_structures} structures; the recorded "
            "'structure_index' does not match 'simulation.structures'."
        )
    return owners


def _volumetric_loss_per_structure(
    field_structure: FieldStructureData, num_structures: int, sim: Simulation
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Total volumetric loss [W] per structure, one value per recorded frequency.

    Per-component absorbed power stays on each axis's own native Yee grid (matching where
    ``structure_index_*`` is recorded), multiplied by each sample's Yee control volume (see
    ``FieldStructureData._cell_volumes``), grouped by ownership, and summed across axes. Samples
    with zero volume (control volume outside the monitor, e.g. the recording padding) are left out
    of the grouping, so an owner is reported only when one of its samples covers part of the
    monitor.
    """
    zero_dims = field_structure.monitor.zero_dims
    if zero_dims:
        zero_dim_names = ", ".join(["x", "y", "z"][d] for d in zero_dims)
        raise SetupError(
            f"'per_structure_loss' requires a volumetric 'FieldStructureMonitor' (nonzero size "
            f"along every axis); '{field_structure.monitor.name}' is zero-sized along "
            f"{zero_dim_names}. A 2D/1D monitor's cell-volume element has no real thickness to "
            "integrate over, so a total power in [W] cannot be formed -- use "
            "'absorbed_power_density' directly for a density on a plane instead."
        )
    flat_dims = [dim for dim, num_cells in zip("xyz", sim.grid.num_cells) if num_cells == 1]
    if flat_dims:
        raise SetupError(
            "'per_structure_loss' requires a 3D simulation; the simulation grid has a single "
            f"cell along {', '.join(flat_dims)}, so the recorded fields are a 2D cross-section "
            "and a total power in [W] cannot be formed -- use 'absorbed_power_density' for the "
            "loss density instead."
        )
    expanded = field_structure.symmetry_expanded
    freqs = np.atleast_1d(np.asarray(expanded.monitor.freqs, dtype=float))
    per_axis_power = expanded.per_component_absorbed_power
    cell_volumes = expanded._cell_volumes()
    structure_index = {
        "x": expanded.structure_index_x,
        "y": expanded.structure_index_y,
        "z": expanded.structure_index_z,
    }
    owners = {
        axis: _check_owner_range(
            structure_index[axis].values, num_structures, f"structure_index_{axis}"
        )
        for axis in "xyz"
    }
    if any(sim.symmetry):
        owners = {
            axis: _relabel_mirrored_volumetric(structure_index[axis], owners[axis], sim)
            for axis in "xyz"
        }

    weighted = {axis: cell_volumes[axis].values > 0 for axis in "xyz"}

    # select by position: 'freqs' follows 'monitor.freqs', which need be neither sorted nor unique
    totals: dict[int, np.ndarray] = {}
    for f_ind in range(freqs.size):
        for axis in "xyz":
            power = per_axis_power[axis].isel(f=f_ind).real
            contribution = loss_numerics.integrate_volumetric_loss(
                power.values[weighted[axis]],
                owners[axis][weighted[axis]],
                cell_volumes[axis].values[weighted[axis]],
            )
            for idx, value in contribution.items():
                totals.setdefault(idx, np.zeros(freqs.size))[f_ind] += value
    return totals, freqs


def _surface_loss_per_structure(
    surface_field_structure: SurfaceFieldStructureData, sim: Simulation, num_structures: int
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Total surface loss [W] per structure, one value per recorded frequency."""
    expanded = surface_field_structure.symmetry_expanded
    owners = _check_owner_range(
        expanded.structure_index.values.values, num_structures, "structure_index"
    )
    if any(sim.symmetry):
        owners = _relabel_mirrored_surface(expanded, owners, sim)
        # look up each vertex's Re(Z_s) from its relabeled owner too
        relabeled = expanded.structure_index.values.copy(data=owners.astype(float))
        expanded = expanded.updated_copy(
            structure_index=expanded.structure_index.updated_copy(values=relabeled)
        )

    freqs = np.atleast_1d(np.asarray(expanded.monitor.freqs, dtype=float))
    sld = expanded.surface_loss_density(sim)
    points = np.asarray(sld.points.values)
    cells = np.asarray(sld.cells.values)
    vertex_area = loss_numerics.vertex_areas(points, cells)

    totals: dict[int, np.ndarray] = {}
    for f_ind in range(freqs.size):
        sld_slice = np.asarray(sld.values.isel(f=f_ind).values).real
        contribution = loss_numerics.integrate_surface_loss(sld_slice, vertex_area, owners)
        for idx, value in contribution.items():
            totals.setdefault(idx, np.zeros(freqs.size))[f_ind] += value
    return totals, freqs


def _em_medium(medium: AbstractMedium | MultiPhysicsMedium) -> AbstractMedium | None:
    """The electromagnetic medium: a MultiPhysicsMedium carries it as ``optical``."""
    return medium.optical if isinstance(medium, MultiPhysicsMedium) else medium


def _loses_power_at_surface(medium: AbstractMedium | MultiPhysicsMedium, sim: Simulation) -> bool:
    """Whether the solver models the medium as a surface rather than a volume: a PEC, or a
    non-penetrable lossy metal under the surface impedance (SIBC) subpixel method. Under any other
    lossy-metal method the metal is solved as a regular conductor, so it loses power in its volume.
    """
    em_medium = _em_medium(medium)
    if em_medium is None or not em_medium.is_pec_like:
        return False
    if isinstance(em_medium, LossyMetalMedium):
        return isinstance(sim._subpixel.lossy_metal, SurfaceImpedance)
    return True


def _absorbs_volumetrically(
    medium: AbstractMedium | MultiPhysicsMedium, sim: Simulation, freqs: np.ndarray
) -> bool:
    """Whether a medium modeled as a volume has a positive Im(eps) at any of ``freqs``."""
    em_medium = _em_medium(medium)
    if em_medium is None or _loses_power_at_surface(medium, sim):
        return False
    try:
        return any(np.any(np.imag(np.asarray(em_medium.eps_diagonal(f))) > 0) for f in freqs)
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return False


def _has_off_diagonal_loss(medium: AbstractMedium | MultiPhysicsMedium) -> bool:
    """Whether a medium dissipates through cross terms between field components, such as
    ``E_x* sigma_xy E_y``, which the diagonal permittivity recorded by a 'FieldStructureMonitor'
    cannot account for. Only the symmetric part of the real conductivity tensor dissipates: an
    antisymmetric (e.g. magneto-optic) part adds no loss, so it is not counted."""
    em_medium = _em_medium(medium)
    if not isinstance(em_medium, FullyAnisotropicMedium):
        return False
    conductivity = np.asarray(em_medium.conductivity, dtype=float)
    dissipative = (conductivity + conductivity.T) / 2
    off_diagonal = np.abs(dissipative[~np.eye(3, dtype=bool)])
    # entries this small next to the largest one are rounding, e.g. from a rotated diagonal tensor
    return bool(np.any(off_diagonal > _OFF_DIAGONAL_RTOL * np.max(np.abs(conductivity))))


def _owner_at(
    points: np.ndarray, sim: Simulation, candidates: set[int] | None = None
) -> np.ndarray:
    """Owner of each point (N, 3) by the solver's painter rule: of the structures whose geometry
    contains the point, the one painted last, or -1 if none. Structures are painted in the order
    'Structure._sort_structures' gives, by priority under ``sim.structure_priority_mode`` with
    list order breaking ties; owners are returned as indices into ``sim.structures``.
    ``candidates`` restricts the structures considered. Only the points within a structure's
    bounds are tested against its geometry."""
    structures = sim.structures
    painted = sorted(
        range(len(structures)),
        key=lambda index: structures[index]._priority(sim.structure_priority_mode),
    )
    owners = np.full(len(points), -1, dtype=int)
    for index in painted:
        if candidates is not None and index not in candidates:
            continue
        rmin, rmax = structures[index].geometry.bounds
        in_bounds = np.flatnonzero(np.all((points >= rmin) & (points <= rmax), axis=1))
        if in_bounds.size == 0:
            continue
        candidates_in = points[in_bounds]
        inside = structures[index].geometry.inside(
            candidates_in[:, 0], candidates_in[:, 1], candidates_in[:, 2]
        )
        owners[in_bounds[np.asarray(inside, dtype=bool)]] = index
    return owners


def _mirrored_samples(points: np.ndarray, sim: Simulation) -> tuple[np.ndarray, np.ndarray]:
    """Which points (N, 3) lie in the part of the domain the solver did not record, below an
    active symmetry plane of ``sim``, and the image of each point across every plane it lies below:
    the recorded sample its values were expanded from.

    The symmetry is read from the simulation rather than the data, since data read from
    'SimulationData' arrives already expanded, with its own 'symmetry' reset."""
    mirrored = np.zeros(len(points), dtype=bool)
    images = np.array(points, dtype=float)
    for axis, plane_symmetry in enumerate(sim.symmetry):
        if plane_symmetry == 0:
            continue
        center = sim.center[axis]
        below = points[:, axis] < center - _MIRROR_PLANE_ATOL
        images[below, axis] = 2 * center - points[below, axis]
        mirrored |= below
    return mirrored, images


def _relabel_mirrored(
    owners: np.ndarray,
    mirrored_points: np.ndarray,
    recorded_points: np.ndarray,
    sim: Simulation,
    candidates: set[int] | None = None,
) -> np.ndarray:
    """Owners of mirrored samples. Symmetry expansion copies each recorded sample's owner to its
    mirror image, which is wrong when the image lies in another structure, e.g. the second of two
    mirror-image structures. Where the priority rule reproduces the solver's owner at the recorded
    point, it is trusted at the mirrored point too; elsewhere the copied owner is kept."""
    consistent = _owner_at(recorded_points, sim, candidates) == owners
    relabeled = owners.copy()
    relabeled[consistent] = _owner_at(mirrored_points[consistent], sim, candidates)
    return relabeled


def _relabel_mirrored_volumetric(
    structure_index: SpatialDataArray, owners: np.ndarray, sim: Simulation
) -> np.ndarray:
    """``owners`` of one E component's expanded samples, with the mirrored ones relabeled. The
    samples are processed in blocks of '_RELABEL_BLOCK_SIZE', so the temporary coordinate arrays
    stay bounded however large the monitor is."""
    dims = list(structure_index.dims)
    axis_coords = [np.asarray(structure_index.coords[dim].values, dtype=float) for dim in dims]
    xyz_order = [dims.index(dim) for dim in "xyz"]
    # a C-ordered copy, so that 'flat' is a view of it whatever the layout of 'owners'
    relabeled = np.array(owners, dtype=int, order="C", copy=True)
    flat = relabeled.reshape(-1)
    for start in range(0, flat.size, _RELABEL_BLOCK_SIZE):
        block = np.arange(start, min(start + _RELABEL_BLOCK_SIZE, flat.size))
        grid_index = np.unravel_index(block, relabeled.shape)
        points = np.stack([axis_coords[d][grid_index[d]] for d in xyz_order], axis=-1)
        mirrored, images = _mirrored_samples(points, sim)
        if mirrored.any():
            samples = block[mirrored]
            flat[samples] = _relabel_mirrored(
                flat[samples], points[mirrored], images[mirrored], sim
            )
    return relabeled


def _relabel_mirrored_surface(
    expanded: SurfaceFieldStructureData, owners: np.ndarray, sim: Simulation
) -> np.ndarray:
    """``owners`` of the expanded surface vertices, with the mirrored ones relabeled. A vertex
    lies on the metal surface, so its owner is found a small step into the metal along the
    outward normal, among the structures the solver models as a surface."""
    points = np.asarray(expanded.structure_index.points.values, dtype=float)
    normals = np.asarray(expanded.normal.values.values, dtype=float)
    mirrored, images = _mirrored_samples(points, sim)
    relabeled = np.array(owners, copy=True)
    if not mirrored.any():
        return relabeled
    image_normals = np.where(images != points, -normals, normals)
    step = _SURFACE_OWNER_STEP * min(np.min(sizes) for sizes in sim.grid.sizes.to_list)
    metals = {
        index
        for index, structure in enumerate(sim.structures)
        if _loses_power_at_surface(structure.medium, sim)
    }
    relabeled[mirrored] = _relabel_mirrored(
        relabeled[mirrored],
        points[mirrored] - step * normals[mirrored],
        images[mirrored] - step * image_normals[mirrored],
        sim,
        metals,
    )
    return relabeled


def _check_owners_supported(
    owners: list[int], structures: list[Structure], sim: Simulation
) -> None:
    """Reject owners whose loss this method cannot attribute or compute correctly."""
    cross_coupled = [
        i
        for i in owners
        if not _loses_power_at_surface(sim.medium if i == -1 else structures[i].medium, sim)
        and _has_off_diagonal_loss(sim.medium if i == -1 else structures[i].medium)
    ]
    if cross_coupled:
        names = ["the background medium" if i == -1 else f"structure {i}" for i in cross_coupled]
        raise SetupError(
            f"'per_structure_loss' does not support {', '.join(names)}: a "
            "'FullyAnisotropicMedium' with off-diagonal conductivity loses power through cross "
            "terms between field components, which the diagonal permittivity recorded by a "
            "'FieldStructureMonitor' does not capture."
        )


def compute_per_structure_loss(
    structures: list[Structure],
    field_structure: FieldStructureData | None,
    surface_field_structure: SurfaceFieldStructureData | None,
    sim: Simulation,
) -> StructureLossDataArray:
    """Total loss [W] per structure, taken from the monitor that records that structure's kind.

    A structure the solver models as a surface (a PEC, or a lossy metal under the surface
    impedance subpixel method) loses power at its surface, recorded by the surface monitor; any
    other structure loses it in its volume, recorded by the volumetric monitor. Each structure's
    total comes from its own kind's monitor only. Only structures sampled by a supplied monitor
    appear, and one that its own kind's monitor did not sample (not supplied, or not reaching the
    structure) is NaN.

    At least one of ``field_structure``/``surface_field_structure`` must be given; when both are,
    they must record the same frequencies. Both monitors' ``structure_index`` must index the same
    ``Simulation.structures`` (true for any two monitors recorded against the same simulation);
    that is not detectable here, so it is a caller requirement.

    The result covers every owner sampled by a supplied monitor. ``structure_index`` ``-1`` is the
    background medium (``sim.medium``), treated like any structure. Surface vertices are owned only
    by PEC-like structures, so a structure that loses power in its volume may be absent from a
    surface-only result; a warning is logged when the simulation contains a lossy one and no
    volumetric monitor was supplied.

    Under simulation symmetry, the samples of the mirrored half are attributed to the structure
    the geometry places there, not to the one their values were expanded from, so mirror-image
    structures each get their own loss; this assumes the setup is symmetric. Raises a
    ``SetupError`` for a ``FullyAnisotropicMedium`` owner with off-diagonal conductivity, whose loss
    has cross terms the recorded diagonal permittivity cannot capture.
    """
    if field_structure is None and surface_field_structure is None:
        raise SetupError(
            "'per_structure_loss' requires at least one of 'volumetric_monitor_name' or "
            "'surface_monitor_name'."
        )
    if field_structure is not None and surface_field_structure is not None:
        volumetric_freqs = np.asarray(field_structure.monitor.freqs, dtype=float)
        surface_freqs = np.asarray(surface_field_structure.monitor.freqs, dtype=float)
        if volumetric_freqs.shape != surface_freqs.shape or not np.allclose(
            volumetric_freqs, surface_freqs, rtol=1e-12, atol=0
        ):
            raise SetupError(
                "'per_structure_loss' combines the volumetric and surface monitors frequency by "
                f"frequency, so '{field_structure.monitor.name}' and "
                f"'{surface_field_structure.monitor.name}' must record the same 'freqs'."
            )

    num_structures = len(structures)
    volumetric_totals: dict[int, np.ndarray] | None = None
    surface_totals: dict[int, np.ndarray] | None = None
    freqs = None
    if field_structure is not None:
        volumetric_totals, freqs = _volumetric_loss_per_structure(
            field_structure, num_structures, sim
        )
    if surface_field_structure is not None:
        surface_totals, freqs = _surface_loss_per_structure(
            surface_field_structure, sim, num_structures
        )

    if field_structure is None:
        media = [(-1, sim.medium), *((i, s.medium) for i, s in enumerate(structures))]
        absorbers = [i for i, m in media if _absorbs_volumetrically(m, sim, freqs)]
        if absorbers:
            names = ["the background medium" if i == -1 else f"structure {i}" for i in absorbers]
            log.warning(
                "'per_structure_loss' was called without 'volumetric_monitor_name', but "
                f"{', '.join(names)} lose power in their volume, which only a "
                "'FieldStructureMonitor' records; they are absent from the result."
            )

    owners = sorted(set(volumetric_totals or {}) | set(surface_totals or {}))
    _check_owners_supported(owners, structures, sim)
    values = np.full((freqs.size, len(owners)), np.nan)
    for column, owner in enumerate(owners):
        medium = sim.medium if owner == -1 else structures[owner].medium
        totals = surface_totals if _loses_power_at_surface(medium, sim) else volumetric_totals
        # NaN unless the monitor for this owner's kind was supplied and sampled it: a lossy metal
        # outside the surface monitor's reach has no recorded loss, which zero would misreport
        if totals is not None and owner in totals:
            values[:, column] = totals[owner]
    return StructureLossDataArray(values, coords={"f": freqs, "structure_index": owners})
