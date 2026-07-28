"""Optical carrier generation and thermalization heat from linear absorption."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, PositiveFloat

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.material.multi_physics import MultiPhysicsMedium
from tidy3d.components.material.tcad.charge import SemiconductorMedium
from tidy3d.components.tcad.generation_recombination import DistributedGeneration
from tidy3d.constants import HBAR, HERTZ, VOLUMETRIC_HEAT_RATE, Q_e
from tidy3d.exceptions import DataError, SetupError
from tidy3d.log import log

if TYPE_CHECKING:
    import xarray as xr

    from tidy3d.components.data.monitor_data import FieldStructureData
    from tidy3d.components.structure import Structure

# frequency-matching tolerance for a requested vs. recorded frequency
FREQ_RTOL = 1e-5


class OpticalGenerationData(Tidy3dBaseModel):
    """Carrier generation and thermalization heat from linear optical absorption.

    An instance holds either the whole-domain **aggregate** (``component=None``, produced by
    :meth:`.SimulationData.optical_generation`: per-component quantities colocated onto the common
    grid-boundary coordinates and summed over components) or a single electric-field
    **component's contribution** (``component="x"/"y"/"z"``, produced by
    :meth:`.SimulationData.per_component_optical_generation`: kept on that component's native Yee
    grid, not colocated and not summed). The absorbed optical power is converted into a
    band-gap-gated carrier ``generation_rate`` and the accompanying ``thermalization_heat_rate``,
    all evaluated at a single frequency and returned as :class:`.SpatialDataArray` objects (which
    the Heat/Charge solvers consume).

    Notes
    -----

        Scope is **linear, single-photon, diagonal-anisotropic** absorption. For each absorbed
        above-gap photon, ``hw = E_g`` (stored in the electron-hole pair) ``+ (hw - E_g)``
        (thermalization heat):

        - Absorbed power: ``P_abs = 1/2 w eps0 sum_i Im(eps_ii) |E_i|^2`` [W/um^3] (always
          includes sub-band-gap / charge-less absorption).
        - Generation rate: ``g = P_abs / hw`` [1/(s um^3)], nonzero only where ``hw >= E_g``.
          Internal quantum efficiency is 1 by construction; carrier loss is left to the charge
          solver.
        - Heat rate: ``Q = P_abs - sum_i g_i E_g,i`` [W/um^3]. Above-gap semiconductor reduces to
          the thermalization loss ``P_abs (1 - E_g/hw)``; below-gap / oxide / metal have ``g = 0``
          so ``Q = P_abs``.

        Each quantity is computed per component on the native Yee grid and the band-gap gate is
        applied per component (using the recorded structure-ownership index) before colocation, so
        interface absorption is not mis-attributed by interpolating the permittivity. For a
        per-component instance the sums over ``i`` above contain only that instance's component;
        the aggregate is recovered by colocating and summing the three components.
    """

    component: Literal["x", "y", "z"] | None = Field(
        None,
        title="Field component",
        description="``None`` for the whole-domain colocated aggregate; ``'x'``/``'y'``/``'z'`` "
        "when this instance holds a single electric-field component's contribution on its "
        "native Yee grid (see ``SimulationData.per_component_optical_generation``).",
    )

    absorbed_power_density: SpatialDataArray = Field(
        title="Absorbed power density",
        description="Volumetric absorbed optical power ``P_abs`` over the whole domain [W/um^3].",
        json_schema_extra={"units": VOLUMETRIC_HEAT_RATE},
    )

    generation_rate: SpatialDataArray = Field(
        title="Generation rate",
        description="Band-gap-gated carrier generation rate ``g`` [1/(um^3 s)].",
        json_schema_extra={"units": "1/(um^3 s)"},
    )

    pair_power_density: SpatialDataArray = Field(
        title="Pair power density",
        description="Power stored in generated electron-hole pairs ``sum_i g_i E_g,i`` [W/um^3].",
        json_schema_extra={"units": VOLUMETRIC_HEAT_RATE},
    )

    freq: PositiveFloat = Field(
        title="Frequency",
        description="Optical frequency at which the absorption was evaluated [Hz]. Stored rather "
        "than read back from the arrays' scalar ``f`` coordinate, which does not survive HDF5 "
        "serialization.",
        json_schema_extra={"units": HERTZ},
    )

    @property
    def thermalization_heat_rate(self) -> SpatialDataArray:
        """Volumetric thermalization + parasitic heat ``Q = P_abs - sum_i g_i E_g,i`` [W/um^3].

        Derived from the two stored powers: ``absorbed_power_density - pair_power_density``.
        """
        heat = self.absorbed_power_density - self.pair_power_density
        return SpatialDataArray(heat.data, coords=heat.coords)

    @property
    def distributed_generation(self) -> DistributedGeneration:
        """Wrap ``generation_rate`` as a :class:`.DistributedGeneration` (um^-3 -> cm^-3).

        For a per-component instance (``component`` set) this wraps only that component's
        contribution, not the total generation.
        """
        return DistributedGeneration.from_rate_um3(gen_um3=self.generation_rate)


def _resolve_frequency(field_structure: FieldStructureData, freq: float | None) -> float:
    """Resolve the single frequency to evaluate against the monitor's recorded frequencies."""
    recorded = np.atleast_1d(np.asarray(field_structure.monitor.freqs, dtype=float))
    if freq is None:
        if recorded.size != 1:
            raise SetupError(
                "'optical_generation' is single-frequency: the monitor recorded "
                f"{recorded.size} frequencies, so 'freq' must be specified."
            )
        return float(recorded[0])

    closest = int(np.argmin(np.abs(recorded - freq)))
    if not np.isclose(recorded[closest], freq, rtol=FREQ_RTOL, atol=0.0):
        raise SetupError(
            f"Requested 'freq={freq}' does not match any recorded monitor frequency within a "
            f"relative tolerance of {FREQ_RTOL}. Recorded frequencies: {recorded.tolist()}."
        )
    return float(recorded[closest])


def _energy_gap_map(
    structure_index: SpatialDataArray,
    structures: list[Structure],
    temperature: float,
) -> tuple[np.ndarray, bool]:
    """Resolve a per-Yee-point band-gap energy ``E_g`` [eV] from structure-ownership indices.

    ``-1`` (background) and any owning structure without a ``SemiconductorMedium`` ``charge`` medium
    map to ``+inf`` (gate closed). Note ``-1`` is handled explicitly, never as ``structures[-1]``.

    Returns the ``E_g`` map and a flag indicating whether any owning structure carried a
    semiconductor ``charge`` medium (used to tell "below gap" apart from "no charge media at all").
    """
    indices = np.round(np.asarray(structure_index.values)).astype(int)
    if indices.size:
        out_of_range = indices[(indices < -1) | (indices >= len(structures))]
        if out_of_range.size:
            raise DataError(
                f"Structure-ownership index {int(out_of_range.flat[0])} is out of range for a "
                f"simulation with {len(structures)} structures; the 'FieldStructureData' "
                "'structure_index_*' maps do not match 'simulation.structures'."
            )

    # ``[background, *structures]`` band-gap lookup built once: row 0 is the background and row
    # ``i + 1`` is ``structures[i]``. ``+inf`` (gate closed) for the background and any owning
    # structure without a ``SemiconductorMedium`` ``charge`` medium. Indexing this with the
    # ownership map (shifted by 1 so ``-1`` -> background) is O(structures + points).
    eg_lookup = np.full(len(structures) + 1, np.inf)
    for i, structure in enumerate(structures):
        medium = structure.medium
        if isinstance(medium, MultiPhysicsMedium) and isinstance(
            medium.charge, SemiconductorMedium
        ):
            eg_lookup[i + 1] = medium.charge.E_g.band_gap_energy(temperature=temperature)

    e_g = eg_lookup[indices + 1]
    # A finite band gap appears only where a cell is owned by a semiconductor structure, so this
    # reflects whether any *owning* structure in this region carries a 'charge' medium (used to
    # distinguish "below gap" from "no charge media at all"), not merely its presence in the sim.
    has_semiconductor = bool(np.isfinite(e_g).any())
    return e_g, has_semiconductor


def _as_spatial(data_array: xr.DataArray) -> SpatialDataArray:
    """Real part, transposed to ``(x, y, z)``, as a :class:`.SpatialDataArray` (keeps scalar ``f``)."""
    ordered = data_array.real.transpose(*[d for d in ("x", "y", "z") if d in data_array.dims])
    return SpatialDataArray(ordered.data, coords=ordered.coords)


def _gated_per_axis(
    field_structure: FieldStructureData,
    structures: list[Structure],
    *,
    freq: float | None,
    temperature: float,
    power_scale: float,
) -> tuple[
    FieldStructureData,
    dict[str, xr.DataArray],
    dict[str, xr.DataArray],
    dict[str, xr.DataArray],
    float,
]:
    """Per-component absorbed power, gated generation rate, and gated pair power on native grids.

    Shared by :func:`compute_optical_generation` (which colocates and sums) and
    :func:`compute_per_component_optical_generation` (which keeps the components separate). The
    absorbed power is ungated; the band-gap gate is applied per component using the recorded
    structure-ownership index before any colocation. Returns the symmetry-expanded data and the
    per-axis (keys ``"x"/"y"/"z"``) power/generation/pair arrays evaluated at the resolved
    frequency, which is returned alongside them.
    """
    if power_scale <= 0:
        raise SetupError(f"'power_scale' must be positive, got {power_scale}.")

    freq = _resolve_frequency(field_structure, freq)

    # symmetry-expand so the whole physical region is covered, then evaluate at the frequency
    expanded = field_structure.symmetry_expanded

    eph_ev = 2 * np.pi * HBAR * freq  # photon energy [eV]
    eph_joule = eph_ev * Q_e

    structure_index = {
        "x": expanded.structure_index_x,
        "y": expanded.structure_index_y,
        "z": expanded.structure_index_z,
    }

    p_axes: dict[str, xr.DataArray] = {}
    g_axes: dict[str, xr.DataArray] = {}
    pair_axes: dict[str, xr.DataArray] = {}
    has_charge_medium = False

    for axis, power in expanded.per_component_absorbed_power.items():
        power = (power.sel(f=freq, method="nearest") * power_scale).real
        e_g, found_semiconductor = _energy_gap_map(structure_index[axis], structures, temperature)
        has_charge_medium = has_charge_medium or found_semiconductor
        gate = eph_ev >= e_g

        g_vals = np.where(gate, power.values / eph_joule, 0.0)
        e_g_joule = np.where(gate, e_g, 0.0) * Q_e  # avoid inf * 0 where the gate is closed

        p_axes[axis] = power
        g_axes[axis] = power.copy(data=g_vals)
        pair_axes[axis] = power.copy(data=g_vals * e_g_joule)

    # Warn only when there is genuinely no semiconductor charge medium to gate against. A region
    # that *has* semiconductors but is simply below band gap legitimately yields zero generation
    # and must stay silent (the warning would send users debugging a non-issue).
    if not has_charge_medium:
        log.warning(
            "'optical_generation': no owning structure in the monitor region carries a 'charge' "
            "medium, so 'generation_rate' is zero everywhere and 'thermalization_heat_rate' equals "
            "'absorbed_power_density'. The usual cause is a missing 'MultiPhysicsMedium' on the "
            "semiconductors."
        )

    return expanded, p_axes, g_axes, pair_axes, freq


def compute_optical_generation(
    field_structure: FieldStructureData,
    structures: list[Structure],
    *,
    freq: float | None = None,
    temperature: float = 300.0,
    power_scale: float = 1.0,
) -> OpticalGenerationData:
    """Compute the whole-domain :class:`.OpticalGenerationData` from a :class:`.FieldStructureData`.

    Per-component absorbed power is formed on each native Yee grid, gated per component using the
    recorded structure-ownership index (resolving each owning structure's ``charge`` band gap), and
    only then colocated onto the common grid-boundary coordinates and summed.
    """
    expanded, p_axes, g_axes, pair_axes, freq_resolved = _gated_per_axis(
        field_structure, structures, freq=freq, temperature=temperature, power_scale=power_scale
    )

    # late colocation: colocate each per-axis scalar onto a common grid, then sum
    return OpticalGenerationData(
        absorbed_power_density=_as_spatial(expanded._colocate_and_sum_axes(p_axes)),
        generation_rate=_as_spatial(expanded._colocate_and_sum_axes(g_axes)),
        pair_power_density=_as_spatial(expanded._colocate_and_sum_axes(pair_axes)),
        freq=freq_resolved,
    )


def compute_per_component_optical_generation(
    field_structure: FieldStructureData,
    structures: list[Structure],
    *,
    freq: float | None = None,
    temperature: float = 300.0,
    power_scale: float = 1.0,
) -> dict[Literal["x", "y", "z"], OpticalGenerationData]:
    """Per-component :class:`.OpticalGenerationData` on each native Yee grid (no colocation).

    Colocation blends the three staggered components and smears the band-gap gate at material
    interfaces. This instead returns one :class:`.OpticalGenerationData` per electric-field
    component (keys ``"x"/"y"/"z"``, each instance tagged with its ``component``), each on that
    component's native grid with the exact gated values -- e.g. to feed a downstream solver per
    component for extra interface accuracy.
    """
    _expanded, p_axes, g_axes, pair_axes, freq_resolved = _gated_per_axis(
        field_structure, structures, freq=freq, temperature=temperature, power_scale=power_scale
    )
    return {
        axis: OpticalGenerationData(
            component=axis,
            absorbed_power_density=_as_spatial(p_axes[axis]),
            generation_rate=_as_spatial(g_axes[axis]),
            pair_power_density=_as_spatial(pair_axes[axis]),
            freq=freq_resolved,
        )
        for axis in p_axes
    }
