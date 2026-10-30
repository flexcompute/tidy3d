"""Per-structure electromagnetic power loss, aggregated from loss-density monitor data, and the
mass-normalized loss density (specific absorption rate)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from flexcompute.core._migration.em.numerical.raw import power_loss as loss_numerics

from tidy3d.components.data.data_array import (
    ScalarFieldDataArray,
    StructureLossDataArray,
    StructureLossTimeDataArray,
)
from tidy3d.components.material.multi_physics import MultiPhysicsMedium
from tidy3d.components.material.tcad.heat import FluidMedium, SolidMedium
from tidy3d.components.medium.abstract_custom import AbstractCustomMedium
from tidy3d.components.medium.anisotropic import AnisotropicMedium, FullyAnisotropicMedium
from tidy3d.components.medium.base import AbstractMedium
from tidy3d.components.medium.lossy_metal import LossyMetalMedium
from tidy3d.components.medium.pole_residue import DispersiveMedium
from tidy3d.components.medium.two_d import Medium2D
from tidy3d.components.subpixel_spec import SurfaceImpedance
from tidy3d.constants import DERIVED_STRUCTURE_INDEX
from tidy3d.exceptions import DataError, SetupError
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tidy3d.components.data.data_array import SpatialDataArray
    from tidy3d.components.data.monitor_data import (
        FieldStructureData,
        FieldStructureTimeData,
        SurfaceFieldStructureData,
    )
    from tidy3d.components.material.types import StructureMediumType

    VolumetricLossData = FieldStructureData | FieldStructureTimeData
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.structure import Structure


# off-diagonal conductivity below this fraction of the largest entry is rounding, not coupling
_OFF_DIAGONAL_RTOL = 1e-9


def _is_time_domain(field_structure: VolumetricLossData | None) -> bool:
    """Whether volumetric loss data was recorded in the time domain."""
    # imported here: the monitor-data package imports this module while it initializes
    from tidy3d.components.data.monitor_data import FieldStructureTimeData

    return isinstance(field_structure, FieldStructureTimeData)


def _check_owner_range(owners: np.ndarray, num_structures: int, data_name: str) -> np.ndarray:
    """Round ``owners`` to the nearest int (the same canonicalization ``_integrate_by_structure``
    and ``surface_loss_density`` use for aggregation/lookup) and raise a controlled 'DataError'
    for any value outside ``{-2, -1} | [0, num_structures)``.

    Mirrors ``optical_generation._energy_gap_map``'s guard: ``-1`` (background) and
    ``DERIVED_STRUCTURE_INDEX`` (``-2``, a lumped element or an automatic PEC frame) are the only
    valid negative values, never indexed into ``structures``. Validating the rounded owners
    (rather than the raw stored values) matters because a value that looks in-range before
    rounding (e.g. ``1.9`` in a two-structure simulation) can round up to an out-of-range index
    during aggregation and slip past a raw-value check undetected. Returns the rounded owners so
    callers use the exact same values this guard validated, rather than re-deriving them (and
    risking a second, possibly inconsistent, cast).
    """
    owners = np.round(np.asarray(owners)).astype(int)
    out_of_range = owners[(owners < DERIVED_STRUCTURE_INDEX) | (owners >= num_structures)]
    if out_of_range.size:
        raise DataError(
            f"Structure-ownership index {int(out_of_range.flat[0])} in '{data_name}' is out of "
            f"range for a simulation with {num_structures} structures; the recorded "
            "'structure_index' does not match 'simulation.structures'."
        )
    return owners


def _structure_index_by_axis(field_structure: FieldStructureData) -> dict[str, SpatialDataArray]:
    """The recorded owner map of each E component, keyed by axis."""
    return {
        "x": field_structure.structure_index_x,
        "y": field_structure.structure_index_y,
        "z": field_structure.structure_index_z,
    }


def _expanded_owners(expanded: FieldStructureData, num_structures: int) -> dict[str, np.ndarray]:
    """Owner of every sample of symmetry-expanded data, per E component, as an index into
    ``sim.structures`` (-1 for the background, -2 for a lumped element or an automatic PEC frame).
    A sample of the mirrored half keeps the owner of the recorded sample it was expanded from."""
    structure_index = _structure_index_by_axis(expanded)
    return {
        axis: _check_owner_range(
            structure_index[axis].values, num_structures, f"structure_index_{axis}"
        )
        for axis in "xyz"
    }


def _volumetric_loss_per_structure(
    field_structure: VolumetricLossData, num_structures: int, sim: Simulation
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Total volumetric loss [W] per structure, one value per recorded frequency, or per recorded
    time for time-domain data (the instantaneous loss).

    Per-component loss power stays on each axis's own native Yee grid (matching where
    ``structure_index_*`` is recorded), multiplied by each sample's Yee control volume (see
    ``FieldStructureData._cell_volumes``), grouped by ownership, and summed across axes. Samples
    with zero volume (control volume outside the monitor, e.g. the recording padding) are left out
    of the grouping, so an owner is reported only when one of its samples covers part of the
    monitor.
    """
    monitor_type = type(field_structure.monitor).__name__
    density_method = (
        "loss_power_density" if _is_time_domain(field_structure) else "absorbed_power_density"
    )
    zero_dims = field_structure.monitor.zero_dims
    if zero_dims:
        zero_dim_names = ", ".join(["x", "y", "z"][d] for d in zero_dims)
        raise SetupError(
            f"'per_structure_loss' requires a volumetric '{monitor_type}' (nonzero size "
            f"along every axis); '{field_structure.monitor.name}' is zero-sized along "
            f"{zero_dim_names}. A 2D/1D monitor's cell-volume element has no real thickness to "
            "integrate over, so a total power in [W] cannot be formed -- use "
            f"'{density_method}' directly for a density on a plane instead."
        )
    flat_dims = [dim for dim, num_cells in zip("xyz", sim.grid.num_cells) if num_cells == 1]
    if flat_dims:
        raise SetupError(
            "'per_structure_loss' requires a 3D simulation; the simulation grid has a single "
            f"cell along {', '.join(flat_dims)}, so the recorded fields are a 2D cross-section "
            f"and a total power in [W] cannot be formed -- use '{density_method}' for the "
            "loss density instead."
        )
    expanded = field_structure.symmetry_expanded
    if _is_time_domain(expanded):
        sample_dim = "t"
        samples = np.atleast_1d(np.asarray(expanded.Ex.coords["t"], dtype=float))
        per_axis_power = expanded.per_component_loss_power
    else:
        sample_dim = "f"
        samples = np.atleast_1d(np.asarray(expanded.monitor.freqs, dtype=float))
        per_axis_power = expanded.per_component_absorbed_power
    cell_volumes = expanded._cell_volumes()
    owners = _expanded_owners(expanded, num_structures)

    weighted = {axis: cell_volumes[axis].values > 0 for axis in "xyz"}

    # select by position: the samples follow 'monitor.freqs', which need be neither sorted nor
    # unique, or the recorded times
    totals: dict[int, np.ndarray] = {}
    for sample_ind in range(samples.size):
        for axis in "xyz":
            power = per_axis_power[axis].isel({sample_dim: sample_ind}).real
            contribution = loss_numerics.integrate_volumetric_loss(
                power.transpose("x", "y", "z").values[weighted[axis]],
                owners[axis][weighted[axis]],
                cell_volumes[axis].values[weighted[axis]],
            )
            for idx, value in contribution.items():
                totals.setdefault(idx, np.zeros(samples.size))[sample_ind] += value
    return totals, samples


def _surface_loss_per_structure(
    surface_field_structure: SurfaceFieldStructureData, sim: Simulation, num_structures: int
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Total surface loss [W] per structure, one value per recorded frequency."""
    expanded = surface_field_structure.symmetry_expanded
    owners = _check_owner_range(
        expanded.structure_index.values.values, num_structures, "structure_index"
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


def _has_surface_loss(medium: AbstractMedium | MultiPhysicsMedium, sim: Simulation) -> bool:
    """Whether the solver models the medium as a surface that loses power: a lossy metal under the
    surface impedance (SIBC) subpixel method, unlike a lossless PEC."""
    return _loses_power_at_surface(medium, sim) and isinstance(_em_medium(medium), LossyMetalMedium)


def _absorbs_volumetrically(
    medium: AbstractMedium | MultiPhysicsMedium, sim: Simulation, freqs: np.ndarray
) -> bool:
    """Whether a medium modeled as a volume has a positive Im(eps) at any of ``freqs``, anywhere
    in it for a custom medium. A medium whose permittivity cannot be evaluated counts as absorbing,
    so that its loss is asked about rather than dropped."""
    em_medium = _em_medium(medium)
    if em_medium is None or _loses_power_at_surface(medium, sim):
        return False
    try:
        return any(
            np.any(np.imag(eps) > 0) for f in freqs for eps in _diagonal_permittivity(em_medium, f)
        )
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return True


def _diagonal_permittivity(medium: AbstractMedium, freq: float) -> list[np.ndarray]:
    """The diagonal permittivity of ``medium`` at ``freq``: at every sample of a custom medium,
    whose 'eps_diagonal' keeps only the sample of largest ``|eps|``."""
    if isinstance(medium, AbstractCustomMedium):
        return [np.asarray(component.values) for component in medium.eps_dataarray_freq(freq)]
    return [np.asarray(medium.eps_diagonal(freq))]


# |Im(eps)| at most this fraction of |eps| counts as no loss or gain: a loss tangent far below that of
# any physical dielectric.
LOSSLESS_LOSS_TANGENT = 1e-6

# Number of frequencies across the simulation band at which a dispersive medium's loss is sampled.
NUM_BAND_SAMPLES = 5


def _simulation_band_samples(sim: Simulation) -> np.ndarray:
    """Positive frequencies spanning the sources' band; empty when no source sets one."""
    freq_min, freq_max = sim.frequency_range
    freqs = np.linspace(freq_min, freq_max, NUM_BAND_SAMPLES)
    return np.unique(freqs[freqs > 0])


def _has_loss_or_gain(medium: AbstractMedium, freqs: np.ndarray) -> bool:
    """Whether ``medium`` has a nonzero ``Im(eps)`` (beyond :data:`LOSSLESS_LOSS_TANGENT`) at any of
    ``freqs``, anywhere in it for a custom medium. A medium whose permittivity cannot be evaluated,
    or that is given no frequency to evaluate it at, counts as having loss or gain."""
    if freqs.size == 0:
        return True
    try:
        for freq in freqs:
            for eps in _diagonal_permittivity(medium, freq):
                if np.any(np.abs(np.imag(eps)) > LOSSLESS_LOSS_TANGENT * np.abs(eps)):
                    return True
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return True
    return False


def _unrecorded_time_domain_loss(
    medium: AbstractMedium | MultiPhysicsMedium, freqs: np.ndarray
) -> str | None:
    """Why the time stepping exchanges power with ``medium`` outside the static conductivity a
    ``FieldStructureTimeMonitor`` records, or ``None`` when it records all of it.

    - A nonlinear medium carries its nonlinear loss or gain outside its static conductivity.
    - A dispersive medium carries all of its loss or gain, its conductivity included, in pole
      currents. One with no loss or gain at ``freqs`` is recorded exactly: its loss is zero.
    """
    em_medium = _em_medium(medium)
    if em_medium is None:
        return None
    if isinstance(em_medium, AnisotropicMedium):
        components = list(em_medium.elements.values())
    else:
        components = [em_medium]
    for component in components:
        if component.is_nonlinear:
            return "a nonlinear medium, whose nonlinear loss or gain is carried outside its conductivity"
        if isinstance(component, DispersiveMedium) and _has_loss_or_gain(component, freqs):
            return (
                "a dispersive medium with loss or gain in the simulation band, all of which, its "
                "conductivity included, is carried by its pole currents"
            )
    return None


def _medium_label(medium: AbstractMedium | MultiPhysicsMedium) -> str:
    """How an error message names a medium: by its ``name``, or by its type when unnamed."""
    return f"medium '{medium.name}'" if medium.name else f"an unnamed '{type(medium).__name__}'"


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


def _check_owners_supported(
    owners: list[int], structures: list[Structure], sim: Simulation, method_name: str
) -> None:
    """Reject owners whose loss ``method_name`` cannot attribute or compute correctly."""
    cross_coupled = [
        i
        for i in owners
        if i != DERIVED_STRUCTURE_INDEX
        and not _loses_power_at_surface(sim.medium if i == -1 else structures[i].medium, sim)
        and _has_off_diagonal_loss(sim.medium if i == -1 else structures[i].medium)
    ]
    if cross_coupled:
        names = ["the background medium" if i == -1 else f"structure {i}" for i in cross_coupled]
        raise SetupError(
            f"'{method_name}' does not support {', '.join(names)}: a "
            "'FullyAnisotropicMedium' with off-diagonal conductivity loses power through cross "
            "terms between field components, which the diagonal permittivity recorded by a "
            "'FieldStructureMonitor' does not capture."
        )


def compute_per_structure_loss(
    structures: list[Structure],
    field_structure: VolumetricLossData | None,
    surface_field_structure: SurfaceFieldStructureData | None,
    sim: Simulation,
) -> StructureLossDataArray | StructureLossTimeDataArray:
    """Total loss [W] per structure, taken from the monitor that records that structure's kind.

    Time-domain volumetric data (:class:`.FieldStructureTimeData`) gives the instantaneous loss
    over its recorded times; it cannot be combined with a surface monitor, which is
    frequency-domain. It records loss only through a static conductivity, so an owner that carries
    loss or gain outside one (a dispersive medium with loss or gain in the simulation band, or a
    nonlinear medium) is NaN, with a warning.

    A structure the solver models as a surface (a PEC, or a lossy metal under the surface
    impedance subpixel method) loses power at its surface, recorded by the surface monitor; any
    other structure loses it in its volume, recorded by the volumetric monitor. Each structure's
    total comes from its own kind's monitor only. Only structures sampled by a supplied monitor
    appear, and one that its own kind's monitor did not sample (not supplied, or not reaching the
    structure) is NaN, except a PEC: it loses no power, so it is 0 W whenever a supplied
    monitor samples it.

    At least one of ``field_structure``/``surface_field_structure`` must be given; when both are,
    they must record the same frequencies. Both monitors' ``structure_index`` must index the same
    ``Simulation.structures`` (true for any two monitors recorded against the same simulation);
    that is not detectable here, so it is a caller requirement.

    The result covers every owner sampled by a supplied monitor. ``structure_index`` ``-1`` is the
    background medium (``sim.medium``), treated like any structure, and ``-2`` collects the lumped
    elements and automatic PEC frames, which have no entry in ``sim.structures``; a lumped element
    loses power in its volume and a PEC frame loses none, so ``-2`` takes its total from the
    volumetric monitor and is NaN without one. Surface vertices are owned only by PEC-like
    structures, so a structure that loses power in its volume may be absent from a surface-only
    result; a warning is logged when the simulation contains a lossy one and no volumetric monitor
    was supplied.

    Under simulation symmetry, a sample of the mirrored half keeps the owner of the recorded
    sample it was expanded from, so a structure lying wholly in the mirrored half has no entry
    and the total of its mirror image includes the mirrored samples. Raises a
    ``SetupError`` for a ``FullyAnisotropicMedium`` owner with off-diagonal conductivity, whose loss
    has cross terms the recorded diagonal permittivity cannot capture.
    """
    if field_structure is None and surface_field_structure is None:
        raise SetupError(
            "'per_structure_loss' requires at least one of 'volumetric_monitor_name' or "
            "'surface_monitor_name'."
        )
    time_domain = _is_time_domain(field_structure)
    if time_domain and surface_field_structure is not None:
        raise SetupError(
            "'per_structure_loss' cannot combine the time-domain "
            f"'{field_structure.monitor.name}' with the frequency-domain surface monitor "
            f"'{surface_field_structure.monitor.name}'; call it with one or the other."
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
    samples = None
    if field_structure is not None:
        volumetric_totals, samples = _volumetric_loss_per_structure(
            field_structure, num_structures, sim
        )
    if surface_field_structure is not None:
        surface_totals, samples = _surface_loss_per_structure(
            surface_field_structure, sim, num_structures
        )

    if field_structure is None:
        media = [(-1, sim.medium), *((i, s.medium) for i, s in enumerate(structures))]
        absorbers = [i for i, m in media if _absorbs_volumetrically(m, sim, samples)]
        if absorbers:
            names = ["the background medium" if i == -1 else f"structure {i}" for i in absorbers]
            log.warning(
                "'per_structure_loss' was called without 'volumetric_monitor_name', but "
                f"{', '.join(names)} lose power in their volume, which only a "
                "'FieldStructureMonitor' records; they are absent from the result."
            )

    owners = sorted(set(volumetric_totals or {}) | set(surface_totals or {}))
    _check_owners_supported(owners, structures, sim, "per_structure_loss")
    if surface_field_structure is None:
        metals = [
            i
            for i in owners
            if i != DERIVED_STRUCTURE_INDEX
            and _has_surface_loss(sim.medium if i == -1 else structures[i].medium, sim)
        ]
        if metals:
            names = ["the background medium" if i == -1 else f"structure {i}" for i in metals]
            log.warning(
                "'per_structure_loss' was called without 'surface_monitor_name', but "
                f"{', '.join(names)} lose power at their surface, which only a frequency-domain "
                "'SurfaceFieldStructureMonitor' records; their loss is NaN in the result."
            )
    unrecorded = {}
    if time_domain:
        band = _simulation_band_samples(sim)
        for owner in owners:
            if owner == DERIVED_STRUCTURE_INDEX:
                continue
            medium = sim.medium if owner == -1 else structures[owner].medium
            if _loses_power_at_surface(medium, sim):
                continue
            reason = _unrecorded_time_domain_loss(medium, band)
            if reason is not None:
                unrecorded[owner] = reason
        if unrecorded:
            names = [
                f"{'the background medium' if i == -1 else f'structure {i}'} ({reason})"
                for i, reason in unrecorded.items()
            ]
            log.warning(
                "'per_structure_loss' reports NaN for "
                f"{'; '.join(names)}: the time-domain monitor records loss only through a static "
                "conductivity."
            )
    values = np.full((samples.size, len(owners)), np.nan)
    for column, owner in enumerate(owners):
        if owner in unrecorded:
            continue
        if owner == DERIVED_STRUCTURE_INDEX:
            totals = volumetric_totals
        else:
            medium = sim.medium if owner == -1 else structures[owner].medium
            if _loses_power_at_surface(medium, sim) and not _has_surface_loss(medium, sim):
                # a PEC loses no power, whichever monitor sampled it
                values[:, column] = 0.0
                continue
            totals = surface_totals if _loses_power_at_surface(medium, sim) else volumetric_totals
        # NaN unless the monitor for this owner's kind was supplied and sampled it: a lossy metal
        # outside the surface monitor's reach has no recorded loss, which zero would misreport
        if totals is not None and owner in totals:
            values[:, column] = totals[owner]
    if time_domain:
        return StructureLossTimeDataArray(values, coords={"t": samples, "structure_index": owners})
    return StructureLossDataArray(values, coords={"f": samples, "structure_index": owners})


def heat_medium_densities(sim: Simulation) -> dict[StructureMediumType, float]:
    """Mass density [kg/um^3] of each medium of ``sim`` whose heat medium sets a positive one
    (``heat`` of a :class:`.MultiPhysicsMedium`, or the medium's ``heat_spec``), keyed by the
    medium as the simulation holds it."""
    densities = {}
    for medium in [sim.medium, *(structure.medium for structure in sim.structures)]:
        heat = medium.heat_spec
        if isinstance(heat, (SolidMedium, FluidMedium)) and heat.density:
            densities[medium] = heat.density
    return densities


def compute_specific_absorption_rate(
    field_structure: FieldStructureData,
    sim: Simulation,
    density: Mapping[StructureMediumType, float],
    *,
    method_name: str,
    missing_remedy: str,
) -> ScalarFieldDataArray:
    """Point specific absorption rate ``SAR = P_abs / rho`` [W/kg] on the colocated grid.

    Each component's absorbed power is divided by the mass density of the medium owning its
    native Yee sample, and only then colocated and summed, so no sample is divided by a
    neighbour's density. ``density`` maps a medium, as the simulation holds it, to ``rho``
    [kg/um^3].

    Samples owned by the following do not absorb in their volume and contribute zero:

    - a structure modeled as a surface: a PEC, including an anisotropic medium with a PEC
      component, or a lossy metal under the surface impedance method;
    - a 2D material;
    - a lumped element or an automatic PEC frame;
    - a lossless medium, which needs no ``density`` entry.

    Every medium that absorbs in its volume needs an entry.

    Raises a ``SetupError`` for a non-positive density, for a ``density`` key that is not a medium
    of the simulation, for a volumetrically absorbing owner without a density, and for a
    ``FullyAnisotropicMedium`` owner with off-diagonal conductivity. The errors name
    ``method_name``, and the one for a missing density ends with ``missing_remedy``. Owners are
    checked over every recorded sample, including the one-cell recording margin around the
    monitor. Under simulation symmetry, a mirrored sample is divided by the density of the owner of
    the recorded sample it was expanded from, as ``per_structure_loss`` attributes its loss.
    """
    structures = list(sim.structures)
    # [background, *structures], indexed by owner + 1 so -1 is the background
    media = [sim.medium, *(structure.medium for structure in structures)]
    simulation_media = set(media)
    for medium, rho in density.items():
        if not isinstance(medium, (AbstractMedium, MultiPhysicsMedium)):
            raise SetupError(
                f"'density' is keyed by media, but has a '{type(medium).__name__}' key. Key each "
                "density by the medium as the simulation holds it ('structure.medium', or "
                "'simulation.medium' for the background)."
            )
        label = _medium_label(medium)
        if medium not in simulation_media:
            raise SetupError(
                f"'density' has an entry for {label}, which is not a medium of the simulation. "
                "Key each density by the medium as the simulation holds it ('structure.medium', "
                "or 'simulation.medium' for the background)."
            )
        if not rho > 0:
            raise SetupError(f"'density' of {label} must be positive [kg/um^3], got {rho}.")

    expanded = field_structure.symmetry_expanded
    freqs = np.atleast_1d(np.asarray(expanded.monitor.freqs, dtype=float))
    owners = _expanded_owners(expanded, len(structures))
    sampled = sorted(set().union(*(np.unique(owners[axis]).tolist() for axis in "xyz")))
    _check_owners_supported(sampled, structures, sim, method_name)

    # [lumped elements and PEC frames, background, *structures], indexed by owner + 2
    inverse_density = np.zeros(len(media) + 1)
    missing = []
    for owner in sampled:
        if owner == DERIVED_STRUCTURE_INDEX:
            continue
        medium = media[owner + 1]
        # contributes zero, whatever ``density`` holds:
        # - a surface or a 2D sheet, which has no mass;
        # - a lossless medium, whose interface samples can still record a neighbour's loss.
        if (
            _loses_power_at_surface(medium, sim)
            or isinstance(_em_medium(medium), Medium2D)
            or not _absorbs_volumetrically(medium, sim, freqs)
        ):
            continue
        if medium in density:
            inverse_density[owner + 2] = 1.0 / density[medium]
        else:
            missing.append(owner)
    if missing:
        names = [
            ("the background" if owner == -1 else f"structure {owner}")
            + f" ({_medium_label(media[owner + 1])})"
            for owner in missing
        ]
        raise SetupError(
            f"'{method_name}' needs a mass density for every medium that absorbs power in the "
            f"monitor region; it is missing for {', '.join(names)}. {missing_remedy}"
        )

    structure_index = _structure_index_by_axis(expanded)
    per_axis = {
        axis: power.real
        * xr.DataArray(
            inverse_density[owners[axis] + 2],
            coords=structure_index[axis].coords,
            dims=structure_index[axis].dims,
        )
        for axis, power in expanded.per_component_absorbed_power.items()
    }
    total = expanded._colocate_and_sum_axes(per_axis)
    ordered = total.transpose(*[dim for dim in ("x", "y", "z", "f") if dim in total.dims])
    return ScalarFieldDataArray(ordered.data, coords=ordered.coords)
