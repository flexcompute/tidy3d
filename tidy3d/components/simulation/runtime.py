"""Runtime, time-step, Courant, and simulation-size validation for ``Simulation``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np

from tidy3d.components.base import cached_property
from tidy3d.components.boundary import (
    ABCBoundary,
    Absorber,
    BlochBoundary,
    ModeABCBoundary,
    Periodic,
)
from tidy3d.components.medium import FullyAnisotropicMedium, LossyMetalMedium
from tidy3d.components.monitor import (
    FreqMonitor,
    MediumMonitor,
    ModeMonitor,
    PermittivityMonitor,
    PointCloudPermittivityMonitor,
)
from tidy3d.components.run_time_spec import RunTimeSpec
from tidy3d.components.source.field import TFSF, AbstractModeSource, FixedAngleSpec, PlaneWave
from tidy3d.config import config
from tidy3d.constants import C_0
from tidy3d.exceptions import (
    SetupError,
    Tidy3dError,
)
from tidy3d.log import log
from tidy3d.packaging import (
    _check_tidy3d_extras_available,
    tidy3d_extras,
)

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.grid.grid import Coords1D
    from tidy3d.components.types import FreqBound
    from tidy3d.components.types.time import SourceTimeType

from . import constants


@property
def _simple_bc(self: Any) -> tuple[bool, bool, bool]:
    """Return whether each axis has no Periodic, Bloch, ABC, or ModeABC boundary."""
    simple = []
    for axis_name in ("x", "y", "z"):
        boundary = self.boundary_spec[axis_name]
        simple.append(
            not any(
                isinstance(item, Periodic | BlochBoundary | ABCBoundary | ModeABCBoundary)
                for item in (boundary.plus, boundary.minus)
            )
        )
    return tuple(simple)


def _validate_relax_courant_compatibility(self: Any) -> Self:
    """Error if ``relax_courant`` is enabled with incompatible components."""

    if not self.relax_courant:
        return self

    incompatible = []

    if len(self.internal_absorbers) > 0:
        incompatible.append("Internal absorbers are not supported.")

    boundary_spec = self.boundary_spec
    if boundary_spec is not None:
        for axis_name in ("x", "y", "z"):
            boundary = boundary_spec[axis_name]
            if isinstance(boundary.plus, Absorber) or isinstance(boundary.minus, Absorber):
                incompatible.append(f"Adiabatic absorber boundary condition along {axis_name}.")

    for source in self.sources:
        if isinstance(source, TFSF):
            incompatible.append(f"TFSF source '{source.name}'.")
        elif isinstance(source, PlaneWave) and isinstance(source.angular_spec, FixedAngleSpec):
            incompatible.append(f"Fixed-angle PlaneWave source '{source.name}'.")

    mediums = [self.medium] + [structure.medium for structure in self.structures]
    for medium in mediums:
        if isinstance(medium, FullyAnisotropicMedium):
            incompatible.append("Contains a 'FullyAnisotropicMedium'.")
        if hasattr(medium, "nonlinear_spec") and medium.nonlinear_spec is not None:
            incompatible.append("Contains a nonlinear medium.")
        if hasattr(medium, "modulation_spec") and medium.modulation_spec is not None:
            incompatible.append("Contains a time-modulated medium.")

    if any(s == 0 for s in self.size):
        incompatible.append("Zero-size (collapsed) simulation dimensions.")

    for axis_name, num_cells in zip(("x", "y", "z"), self.grid.num_cells):
        if num_cells <= 1:
            incompatible.append(f"Single-cell {axis_name}-axis (quasi-2D simulation).")

    if boundary_spec is not None:
        if not any(self._simple_bc):
            incompatible.append(
                "No axis free of Periodic, Bloch, ABC, and ModeABC boundary conditions "
                "(at least one such axis is required)."
            )
        for axis_name in ("x", "y", "z"):
            axis_boundary = boundary_spec[axis_name]
            if isinstance(axis_boundary.plus, ABCBoundary | ModeABCBoundary) or isinstance(
                axis_boundary.minus, ABCBoundary | ModeABCBoundary
            ):
                incompatible.append(f"ABC or ModeABC boundary condition along {axis_name}.")

    if incompatible:
        detail = "\n".join(f"  - {item}" for item in incompatible)
        self._raise_validation_error_at_loc(
            "'relax_courant' is incompatible with the current simulation:\n" + detail,
            "relax_courant",
        )

    return self


def _validate_low_freq_smoothing(self: Any) -> Self:
    """Validate the low frequency smoothing parameters."""
    # check that all monitors are present and they are mode monitors
    val = self.low_freq_smoothing
    if val is None:
        return self
    monitors = self.monitors
    present_mode_monitor_names = [
        monitor.name for monitor in monitors if isinstance(monitor, ModeMonitor)
    ]
    for monitor_ind, monitor in enumerate(val.monitors):
        if monitor not in present_mode_monitor_names:
            self._raise_validation_error_at_loc(
                f"Low frequency smoothing specification refers to monitor '{monitor}' which either does not exist or is not a mode monitor.",
                "low_freq_smoothing",
                "monitors",
                monitor_ind,
            )
    return self


def _validate_size(self: Any) -> None:
    """Ensures the simulation is within size limits before simulation is uploaded."""

    if config.simulation.skip_size_checks:
        return

    num_domain_cells_excluding_pml = self._num_non_pml_cells()
    if num_domain_cells_excluding_pml < constants.WARN_SIM_DOMAIN_CELLS_EXCLUDING_PML:
        log.warning(
            f"Simulation has {num_domain_cells_excluding_pml} grid cells in the simulation "
            "domain excluding PML, which is below the recommended "
            f"{constants.WARN_SIM_DOMAIN_CELLS_EXCLUDING_PML}. Please double-check that the setup "
            "is intended (for example, units).",
            custom_loc=["size"],
        )

    num_comp_cells = self.num_cells / 2 ** (np.sum(np.abs(self.symmetry)))
    if num_comp_cells > constants.MAX_GRID_CELLS:
        raise SetupError(
            f"Simulation has {num_comp_cells:.2e} computational cells, "
            f"a maximum of {constants.MAX_GRID_CELLS:.2e} are allowed."
        )

    num_time_steps = self.num_time_steps
    if num_time_steps > constants.MAX_TIME_STEPS:
        raise SetupError(
            f"Simulation has {num_time_steps:.2e} time steps, "
            f"a maximum of {constants.MAX_TIME_STEPS:.2e} are allowed."
        )
    if num_time_steps > constants.WARN_TIME_STEPS:
        log.warning(
            f"Simulation has {num_time_steps:.2e} time steps. The 'run_time' may be "
            "unnecessarily large, unless there are very long-lived resonances.",
            custom_loc=["run_time"],
        )

    num_cells_times_steps = num_time_steps * num_comp_cells
    if num_cells_times_steps > constants.MAX_CELLS_TIMES_STEPS:
        raise SetupError(
            f"Simulation has {num_cells_times_steps:.2e} grid cells * time steps, "
            f"a maximum of {constants.MAX_CELLS_TIMES_STEPS:.2e} are allowed."
        )


@cached_property
def _run_time(self: Any) -> float:
    """Run time evaluated based on self.run_time."""

    if not isinstance(self.run_time, RunTimeSpec):
        return self.run_time

    return self._resolve_run_time([src.source_time for src in self.sources])


def _resolve_run_time(self: Any, source_times: list[SourceTimeType]) -> float:
    """Resolve a ``RunTimeSpec`` run time from explicit source time pulses.

    Decoupled from ``self.sources`` so callers that know the excitation a priori (such as
    a :class:`.AbstractComponentModeler`) can resolve the run time without attaching
    sources to the simulation. Assumes ``self.run_time`` is a ``RunTimeSpec``.
    """
    run_time_spec = self.run_time

    # contribution from the time of the source pulses
    if not source_times:
        source_time = 0.0
        max_ref_ind = 1
    else:
        end_times = [st.end_time() for st in source_times]
        end_times = [x for x in end_times if x is not None]
        if not end_times:
            raise SetupError(
                "Could not resolve a concrete 'run_time' from the 'RunTimeSpec': at least one "
                "excitation must have a decaying (non-DC) pulse profile, so that its end time is "
                "defined."
            )
        source_time_max = np.max(end_times)
        source_time = run_time_spec.source_factor * source_time_max

        # get the maximum refractive index evaluated over each of the source central frequencies
        all_ref_inds = [self.get_refractive_indices(st._freq0) for st in source_times]
        avg_ref_inds = [np.mean(np.array(n)) for n in all_ref_inds]
        max_ref_ind = np.max(avg_ref_inds, initial=1)

    # contribution from field decay out of the simulation
    propagation_lengths = np.array(self.bounds[1]) - np.array(self.bounds[0])
    max_propagation_length = np.max(propagation_lengths)
    propagation_time = run_time_spec.quality_factor * max_ref_ind * max_propagation_length / C_0

    return source_time + propagation_time


@cached_property
def frequency_range(self: Any) -> FreqBound:
    """Range of frequencies spanning all sources' frequency dependence.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum frequencies of the power spectrum of the sources.
    """
    source_ranges = [source.source_time._frequency_range_sigma_cached for source in self.sources]
    freq_min = min((freq_range[0] for freq_range in source_ranges), default=0.0)
    freq_max = max((freq_range[1] for freq_range in source_ranges), default=0.0)

    return (freq_min, freq_max)


@cached_property
def _dt_fixed_angle_reduction_factor(self: Any) -> float:
    """Reduction in time step due to plane wave source with ``FixedAngleSpec``."""
    if self._is_periodic_fixed_angle:
        theta = self._fixed_angle_sources[0].angle_theta
        return (
            constants.FIXED_ANGLE_DT_SAFETY_FACTOR
            * np.sqrt(3)
            * np.cos(theta) ** 2
            / np.sqrt(2 + np.cos(theta) ** 2)
        )
    return 1


@cached_property
def scaled_courant(self: Any) -> float:
    """When conformal mesh is applied, courant number is scaled down depending on `conformal_mesh_spec`."""

    mediums = self.scene.mediums
    contain_pec_structures = (
        any(medium.is_pec for medium in mediums)
        or any(
            isinstance(src, AbstractModeSource) and src.frame is not None for src in self.sources
        )
        or len(self.internal_absorbers) > 0
    )
    # A penetrable lossy metal is solved as a regular medium, so it does not impose the
    # SIBC courant restriction.
    contain_sibc_structures = any(
        isinstance(medium, LossyMetalMedium) and not medium.penetrable for medium in mediums
    )
    return self.courant * self._subpixel.courant_ratio(
        contain_pec_structures=contain_pec_structures,
        contain_sibc_structures=contain_sibc_structures,
    )


@cached_property
def dt(self: Any) -> float:
    """Simulation time step (distance).

    Returns
    -------
    float
        Time step (seconds).
    """
    dl_mins = [
        np.min(sizes)
        for dim, sizes in enumerate(self.grid.sizes.to_list)
        if self.grid.num_cells[dim] > 1
    ]
    dl_sum_inv_sq = sum(1 / dl**2 for dl in dl_mins)
    dl_avg = 1 / np.sqrt(dl_sum_inv_sq)
    # material factor
    n_cfl = min(min(mat.n_cfl for mat in self.scene.mediums), 1)

    if self.relax_courant:
        _check_tidy3d_extras_available()
        boundaries = self.grid.boundaries.to_list
        relax_ratio = tidy3d_extras["mod"].extension._relax_courant(
            coord_boundaries_x=boundaries[0],
            coord_boundaries_y=boundaries[1],
            coord_boundaries_z=boundaries[2],
            simple_bc=self._simple_bc,
        )
    else:
        relax_ratio = 1.0

    return (
        relax_ratio
        * self._dt_fixed_angle_reduction_factor
        * n_cfl
        * self.scaled_courant
        * dl_avg
        / C_0
    )


@cached_property
def tmesh(self: Any) -> Coords1D:
    """FDTD time stepping points.

    Returns
    -------
    np.ndarray
        Times (seconds) that the simulation time steps through.
    """
    dt = self.dt
    return np.arange(0.0, self._run_time + dt, dt)


@cached_property
def num_time_steps(self: Any) -> int:
    """Number of time steps in simulation."""

    return len(self.tmesh)


@cached_property
def wvl_mat_min(self: Any) -> float:
    """Minimum wavelength in the materials present throughout the simulation.

    Returns
    -------
    float
        Minimum wavelength in the material (microns).
    """
    if len(self.sources) == 0:
        raise Tidy3dError(
            "There are no sources present in the simulation. Please "
            "add sources before querying for the minimum material "
            "wavelength."
        )
    freq_max = max(source.source_time._freq0 for source in self.sources)
    wvl_min = C_0 / freq_max

    n_values = self.get_refractive_indices(freq_max)
    n_max = max(n_values)
    return wvl_min / n_max


@cached_property
def nyquist_step(self: Any) -> int:
    """Maximum number of discrete time steps to keep sampling below Nyquist limit.

    Returns
    -------
    int
        The largest ``N`` such that ``N * self.dt`` is below the Nyquist limit.
    """

    # source frequency upper bound
    freq_source_max = self.frequency_range[1]
    # monitor frequency upper bound
    freq_monitor_max = max(
        (
            monitor.frequency_range[1]
            for monitor in self.monitors
            if isinstance(monitor, FreqMonitor)
            and not isinstance(
                monitor, PermittivityMonitor | MediumMonitor | PointCloudPermittivityMonitor
            )
        ),
        default=0.0,
    )
    # combined frequency upper bound
    freq_max = max(freq_source_max, freq_monitor_max)

    # in fixed angle simulations both E and H are available at full and half steps
    fixed_angle_factor = 1
    if len(self._fixed_angle_sources) > 0:
        fixed_angle_factor = 2

    if freq_max > 0:
        nyquist_step = int(1 / (2 * freq_max) / self.dt * fixed_angle_factor) - 1
        nyquist_step = max(1, nyquist_step)
    else:
        nyquist_step = 1

    return nyquist_step
