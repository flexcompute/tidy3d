"""Source normalization, background-medium, and thin-lens validation for ``Simulation``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import autograd.numpy as np

from tidy3d.components.boundary import AbsorberSpec, BlochBoundary, Periodic
from tidy3d.components.geometry.base import Box
from tidy3d.components.medium import AnisotropicMedium, FullyAnisotropicMedium, Medium
from tidy3d.components.monitor import (
    AbstractFieldMonitor,
    AbstractGaussianOverlapMonitor,
    AbstractOverlapMonitor,
    ThinLensOverlapMonitor,
)
from tidy3d.components.scene import Scene
from tidy3d.components.source.field import (
    TFSF,
    AstigmaticGaussianBeam,
    CustomFieldSource,
    FixedAngleSpec,
    GaussianBeam,
    PlanarSource,
    PlaneWave,
    ThinLensBeam,
)
from tidy3d.components.source.time import ContinuousWave, CustomSourceTime
from tidy3d.components.structure import Structure
from tidy3d.components.thin_lens import MAX_THIN_LENS_SETUP_WORK_UNITS, thin_lens_pupil_grid_samples
from tidy3d.components.validators import named_obj_descr
from tidy3d.config import config
from tidy3d.constants import C_0
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.medium import AbstractMedium, MediumType3D
    from tidy3d.components.types import ArrayFloat1D

from . import constants


def _plane_wave_boundaries(self: Any) -> Self:
    """Error if there are plane wave sources incompatible with boundary conditions."""
    boundaries = self.boundary_spec.to_list
    sources = self.sources
    size = self.size
    sim_medium = self.medium
    structures = self.structures
    for source_ind, source in enumerate(sources):
        if not isinstance(source, PlaneWave):
            continue

        _, tan_dirs = self.pop_axis([0, 1, 2], axis=source.injection_axis)
        medium_set = Scene.intersecting_media(source, structures)
        medium = medium_set.pop() if medium_set else sim_medium

        for tan_dir in tan_dirs:
            boundary = boundaries[tan_dir]

            # check the PML/absorber + angled plane wave case
            num_pml = sum(isinstance(bnd, AbsorberSpec) for bnd in boundary)
            if num_pml > 0 and source.angle_theta != 0:
                self._raise_validation_error_at_loc(
                    "Angled plane wave sources are not compatible with the absorbing boundary "
                    f"along dimension {tan_dir}. Either set the source ``angle_theta`` to "
                    "``0``, or use Bloch boundaries that match the source angle.",
                    "sources",
                    source_ind,
                )

            # check the Bloch boundary + angled plane wave case
            if isinstance(source.angular_spec, FixedAngleSpec):
                num_bloch = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)
                if num_bloch > 0:
                    self._raise_validation_error_at_loc(
                        "Fixed angle plane wave sources ('FixedAngleSpec' and 'angle_theta' != 0) do "
                        f"not require the Bloch boundary along dimension {tan_dir}. "
                        "Either set the boundary conditions to 'Periodic' to proceed to simulate a plane "
                        "wave with frequency-independent propagation direction, or switch to "
                        "'FixedInPlaneKSpec' specification to simulate a plane wave with a fixed "
                        "in-plane Bloch vector (frequency-dependent propagation direction).",
                        "sources",
                        source_ind,
                    )
            else:
                num_bloch = sum(isinstance(bnd, Periodic | BlochBoundary) for bnd in boundary)
                if num_bloch > 0:
                    self._check_bloch_vec(
                        source=source,
                        source_ind=source_ind,
                        bloch_vec=boundary[0].bloch_vec,
                        dim=tan_dir,
                        medium=medium,
                        domain_size=size[tan_dir],
                    )
    return self


def _check_source_freq_available(
    self: Any,
    *,
    no_source_error: str,
    no_source_loc: tuple[object, ...],
    multi_freq_warning: str,
) -> None:
    """Shared source-frequency availability check.

    Used by objects that derive a single evaluation frequency from the simulation's
    sources when none is given explicitly (``ModeABCBoundary`` / ``ABCBoundary`` and
    ``ModeTimeMonitor``). Raises a loc-aware error (at ``no_source_loc``) when there are
    no sources to derive the frequency from, and warns (``multi_freq_warning``) when the
    sources do not share a common central frequency — the first source's central
    frequency is then used.
    """
    sources = self.sources
    if len(sources) == 0:
        self._raise_validation_error_at_loc(no_source_error, *no_source_loc)

    freq0s = [source.source_time._freq0 for source in sources]
    if not all(math.isclose(freq0, freq0s[0]) for freq0 in freq0s):
        log.warning(multi_freq_warning, capture=False)


def _source_homogeneous_isotropic(self: Any) -> Self:
    """Error if a plane wave or gaussian beam source is not in a homogeneous and isotropic
    region.
    """
    val = self.sources

    if val is None:
        return self

    # list of structures including background as a Box()
    structure_bg = Structure(
        geometry=Box(
            size=self.size,
            center=self.center,
        ),
        medium=self.medium,
    )

    structures = self.structures or []
    total_structures = [structure_bg, *list(structures)]

    # for each plane wave in the sources list
    with log as consolidated_logger:
        for source_id, source in enumerate(val):
            # TFSF sources are checked at their injection plane:
            # neither angular spec supports anisotropic source
            # media.
            if isinstance(source, TFSF):
                inj_size = list(source.size)
                inj_size[source.injection_axis] = 0.0
                media_probe = Box(center=source.injection_plane_center, size=tuple(inj_size))
                src_mediums = Scene.intersecting_media(media_probe, total_structures)
                if any(
                    isinstance(m, AnisotropicMedium | FullyAnisotropicMedium) for m in src_mediums
                ):
                    self._raise_validation_error_at_loc(
                        "An anisotropic medium is detected on the injection plane of "
                        f"a {source.type} source. Injection of {source.type} into "
                        "anisotropic media is not currently supported — anisotropic "
                        "structures fully inside the TFSF box are fine.",
                        "sources",
                        source_id,
                    )
            if isinstance(source, PlaneWave | GaussianBeam | AstigmaticGaussianBeam | ThinLensBeam):
                mediums = Scene.intersecting_media(source, total_structures)
                # make sure there is no more than one medium in the returned list
                if len(mediums) > 1:
                    self._raise_validation_error_at_loc(
                        f"{len(mediums)} different mediums detected on plane "
                        f"intersecting a {source.type} source. Plane must be homogeneous.",
                        "sources",
                        source_id,
                    )
                # 0 medium, something is wrong
                if len(mediums) < 1:
                    self._raise_validation_error_at_loc(
                        f"No medium detected on plane intersecting a {source.type}, "
                        "indicating an unexpected error. Please create a github issue so "
                        "that the problem can be investigated.",
                        "sources",
                        source_id,
                    )
                src_medium = list(mediums)[0]
                if isinstance(src_medium, AnisotropicMedium | FullyAnisotropicMedium):
                    self._raise_validation_error_at_loc(
                        f"An anisotropic medium is detected on plane intersecting a {source.type} "
                        f"source. Injection of {source.type} into anisotropic media currently is "
                        "not supported.",
                        "sources",
                        source_id,
                    )

                # check if the medium is spatially uniform
                if not src_medium.is_spatially_uniform:
                    consolidated_logger.warning(
                        f"Nonuniform custom medium detected on plane intersecting a {source.type}. "
                        "Plane must be homogeneous. Make sure custom medium is uniform on the plane.",
                        custom_loc=["sources", source_id],
                    )

                if isinstance(source, PlaneWave) and source._is_periodic_fixed_angle:
                    is_lossless_dieletric = (
                        isinstance(src_medium, Medium) and src_medium.conductivity == 0
                    )

                    if not is_lossless_dieletric:
                        self._raise_validation_error_at_loc(
                            "A fixed angle plane wave can only be injected into a homogeneous isotropic"
                            "dispersionless medium.",
                            "sources",
                            source_id,
                        )

                # check if broadband angled gaussian beam frequency variation is too fast
                if (
                    isinstance(source, GaussianBeam | AstigmaticGaussianBeam)
                    and np.abs(source.angle_theta) > 0
                    and source.num_freqs > 1
                ):

                    def radius(waist_radius: float, waist_distance: float, k0: float) -> float:
                        """Gaussian beam radius at a given waist distance and k0."""
                        z_r = waist_radius**2 * k0 / 2
                        return waist_radius * np.sqrt(1 + (waist_distance / z_r) ** 2)

                    # A slanted GaussianBeam will accumulate a phase that's frequency-dependent
                    # like phi = K f, with the derivative dphi / df = K = 2 * pi * n * r * sin(theta) / c_0.
                    # Here, we compute the maximum value of this coefficient computed at the waist radius
                    # and over all frequencies. Then we compare this to the frequency spacing to
                    # determine whether the frequency dependence is too fast, and issue a warning.
                    optical_path_length = []
                    freqs = source.frequency_grid
                    for freq in freqs:
                        n_freq, _ = src_medium.nk_model(frequency=freq)
                        k0 = 2 * np.pi * n_freq * freq / C_0
                        if isinstance(source, GaussianBeam):
                            rad = radius(source.waist_radius, source.waist_distance, k0)
                        else:
                            rad = max(
                                radius(source.waist_sizes[0], source.waist_distances[0], k0),
                                radius(source.waist_sizes[1], source.waist_distances[1], k0),
                            )
                        optical_path_length.append(n_freq * rad * np.sin(source.angle_theta))
                    # Maximum value of the path length over all freqs
                    max_path_length = np.max(optical_path_length)
                    # Maximum value of the phase difference
                    max_phase_diff = max_path_length * 2 * np.pi * (freqs[-1] - freqs[0]) / C_0
                    # Compare this in magnitude to the frequency spacing assuming uniform
                    # spacing. This is heuristic since in reality we use a Chebyshev grid,
                    # but it should be a good rule of thumb. Because the Chebyshev interpolation
                    # is much better than simple interpolation, we don't require << 1, just < 1
                    if not max_phase_diff / source.num_freqs < 1:
                        log.warning(
                            f"Broadband, angled {source.type} source has a phase dependence "
                            "with frequency that might be under-resolved by the provided "
                            "number of frequencies. Consider reducing the source bandwidth, "
                            "or increasing the 'num_freqs' of the source, and verify the "
                            "source injection in an empty simulation.",
                        )

    return self


def _check_normalize_index(self: Any) -> Self:
    """Check validity of normalize index in context of simulation.sources."""
    val = self.normalize_index

    # not normalizing
    if val is None:
        return self

    sources = self.sources
    num_sources = len(sources)
    if num_sources > 0:
        # No check if no sources, but it should be irrelevant anyway
        if val >= num_sources:
            self._raise_validation_error_at_loc(
                f"'normalize_index' {val} out of bounds for number of sources {num_sources}.",
                "normalize_index",
            )

        # Also error if normalizing by a zero-amplitude source
        if sources[val].source_time.amplitude == 0:
            self._raise_validation_error_at_loc(
                "Cannot set 'normalize_index' to source with zero amplitude.",
                "normalize_index",
            )

        # Warn if normalizing by a ContinuousWave or CustomSourceTime source, if frequency-domain monitors are present.
        if isinstance(sources[val].source_time, ContinuousWave):
            log.warning(
                f"'normalize_index' {val} is a source with 'ContinuousWave' "
                "time dependence. Normalizing frequency-domain monitors by this "
                "source is not meaningful because field decay does not occur. "
                "Consider setting 'normalize_index' to 'None' instead."
            )
        if isinstance(sources[val].source_time, CustomSourceTime):
            log.warning(
                f"'normalize_index' {val} is a source with 'CustomSourceTime' "
                "time dependence. Normalizing frequency-domain monitors by this "
                "source is only meaningful if field decay occurs."
            )

    return self


def _warn_source_monitor_normalization_grid(self: Any) -> None:
    """Warn when a source's use_colocated_integration doesn't match monitor settings."""
    with log as consolidated_logger:
        for src_idx, source in enumerate(self.sources):
            if not isinstance(source, PlanarSource | TFSF):
                continue
            # CustomFieldSource doesn't use flux-based normalization (flux=1),
            # so use_colocated_integration has no effect.
            if isinstance(source, CustomFieldSource):
                continue
            src_colocated = source.use_colocated_integration
            for monitor in self.monitors:
                if not isinstance(monitor, AbstractFieldMonitor | AbstractOverlapMonitor):
                    continue
                # Skip internally generated adjoint monitors (colocate=False by design)
                if monitor.name.startswith("adjoint_") or self._is_flux_adjoint_helper_monitor(
                    monitor
                ):
                    continue
                if monitor.use_colocated_integration != src_colocated:
                    consolidated_logger.warning(
                        f"Source '{source.name}' has "
                        f"'use_colocated_integration={src_colocated}', but monitor "
                        f"'{monitor.name}' has "
                        f"'use_colocated_integration={monitor.use_colocated_integration}'. "
                        "This mismatch may lead to slightly inaccurate power normalization.",
                        custom_loc=["sources", src_idx],
                    )


def _validate_custom_source_time(self: Any) -> None:
    """Warn if all simulation times are outside CustomSourceTime definition range."""
    run_time = self._run_time
    for idx, source in enumerate(self.sources):
        if isinstance(source.source_time, CustomSourceTime):
            if source.source_time._all_outside_range(run_time=run_time):
                data_times = source.source_time.data_times
                mint = np.min(data_times)
                maxt = np.max(data_times)
                obj_descr = named_obj_descr(source, "sources", idx)
                log.warning(
                    f"'CustomSourceTime': {obj_descr} is defined over a time range "
                    f"'({mint}, {maxt})' which does not include any of the 'Simulation' "
                    f"times '({0, run_time})'. The envelope will be constant extrapolated "
                    "from the first or last value in the 'CustomSourceTime', which may not "
                    "be the desired outcome."
                )


def _thin_lens_source_plane_cells(self: Any, source: ThinLensBeam) -> int:
    """Return discretized tangential source-plane cells for thin-lens setup sizing."""
    normal_axis = source.size.index(0.0)
    _, plane_inds = source.pop_axis([0, 1, 2], axis=normal_axis)
    num_cells = self.discretize(source, extend=True).num_cells
    return int(num_cells[plane_inds[0]] * num_cells[plane_inds[1]])


def _thin_lens_setup_work_units(
    self: Any,
    *,
    plane_cells: int,
    num_plane_waves: int | tuple[int, int],
    num_freqs: int,
    num_evaluations: int = 1,
) -> int:
    """Return conservative thin-lens setup work units."""
    return plane_cells * thin_lens_pupil_grid_samples(num_plane_waves) * num_freqs * num_evaluations


@staticmethod
def _thin_lens_setup_work_limit(*, num_evaluations: int) -> int:
    """Return path-specific thin-lens setup work cap."""
    return num_evaluations * MAX_THIN_LENS_SETUP_WORK_UNITS


@staticmethod
def _thin_lens_monitor_setup_evaluations(monitor: ThinLensOverlapMonitor) -> int:
    """Return number of angular-spectrum evaluations for thin-lens monitor setup."""
    if monitor.colocate:
        return constants.THIN_LENS_MONITOR_SETUP_EVALUATIONS
    return constants.THIN_LENS_MONITOR_SETUP_EVALUATIONS * constants.THIN_LENS_FIELD_COMPONENTS


@staticmethod
def _thin_lens_min_background_index(medium: AbstractMedium, freqs: ArrayFloat1D) -> float:
    """Return the minimum effective real background index used by the thin-lens profile."""
    background_n = np.asarray(medium.background_index_from_freqs(freqs), dtype=complex)
    n_real = np.real(background_n)
    n_effective = np.where(n_real <= 0, np.abs(background_n), n_real)
    return float(np.min(n_effective))


def _validate_gaussian_like_beam_background_medium(
    self: Any,
    *,
    beam_obj: ThinLensBeam | AbstractGaussianOverlapMonitor,
    mediums: set[MediumType3D],
    freqs: ArrayFloat1D,
    loc_root: str,
    loc_ind: int,
) -> None:
    """Validate background assumptions used by Gaussian-like beam formulas."""
    if len(mediums) > 1:
        self._raise_validation_error_at_loc(
            f"{len(mediums)} different mediums detected on plane intersecting a "
            f"{beam_obj.type}. Plane must be homogeneous.",
            loc_root,
            loc_ind,
        )
    if len(mediums) < 1:
        self._raise_validation_error_at_loc(
            f"No medium detected on plane intersecting a {beam_obj.type}, "
            "indicating an unexpected error. Please create a github issue so "
            "that the problem can be investigated.",
            loc_root,
            loc_ind,
        )

    medium = next(iter(mediums))
    if isinstance(medium, AnisotropicMedium | FullyAnisotropicMedium):
        self._raise_validation_error_at_loc(
            f"An anisotropic medium is detected on plane intersecting a {beam_obj.type}. "
            f"{beam_obj.type} currently supports only isotropic background media.",
            loc_root,
            loc_ind,
        )
    if not medium.is_spatially_uniform:
        log.warning(
            f"Nonuniform custom medium detected on plane intersecting a {beam_obj.type}. "
            "Gaussian-like overlap setup assumes a homogeneous background medium.",
            custom_loc=[loc_root, loc_ind],
        )

    if not isinstance(beam_obj, ThinLensBeam | ThinLensOverlapMonitor):
        return
    min_background_index = self._thin_lens_min_background_index(medium, freqs)
    if beam_obj.numerical_aperture >= min_background_index:
        self._raise_validation_error_at_loc(
            f"{beam_obj.type} 'numerical_aperture' ({beam_obj.numerical_aperture:.4g}) "
            "must be less than the real background refractive index on its plane "
            f"({min_background_index:.4g}).",
            loc_root,
            loc_ind,
            "numerical_aperture",
        )


def _validate_gaussian_like_beam_backgrounds(self: Any) -> None:
    """Validate Gaussian-like beam source and monitor background medium assumptions."""
    structure_bg = Structure(
        geometry=Box(size=self.size, center=self.center),
        medium=self.medium,
    )
    total_structures = [structure_bg, *list(self.structures or [])]

    for source_ind, source in enumerate(self.sources):
        if not isinstance(source, ThinLensBeam):
            continue
        mediums = Scene.intersecting_media(source, total_structures)
        self._validate_gaussian_like_beam_background_medium(
            beam_obj=source,
            mediums=mediums,
            freqs=np.asarray(source.frequency_grid),
            loc_root="sources",
            loc_ind=source_ind,
        )

    for monitor_ind, monitor in enumerate(self.monitors):
        if not isinstance(monitor, AbstractGaussianOverlapMonitor):
            continue
        mediums = self._call_with_validation_loc(
            ["monitors", monitor_ind],
            self._projection_monitor_mediums_in_bounds,
            center=self.center,
            size=self.size,
            monitor=monitor,
            structures=total_structures,
        )
        self._validate_gaussian_like_beam_background_medium(
            beam_obj=monitor,
            mediums=mediums,
            freqs=np.asarray(monitor.freqs),
            loc_root="monitors",
            loc_ind=monitor_ind,
        )


def _validate_thin_lens_setup_size(self: Any) -> None:
    """Reject thin-lens setups with excessive angular-spectrum preprocessing work."""

    if config.simulation.skip_size_checks:
        return

    for source_ind, source in enumerate(self.sources):
        if not isinstance(source, ThinLensBeam):
            continue
        num_freqs = max(1, np.asarray(source.frequency_grid).size)
        plane_cells = self._thin_lens_source_plane_cells(source)
        work_units = self._thin_lens_setup_work_units(
            plane_cells=plane_cells,
            num_plane_waves=source.num_plane_waves,
            num_freqs=num_freqs,
            num_evaluations=constants.THIN_LENS_SOURCE_SETUP_EVALUATIONS,
        )
        work_limit = self._thin_lens_setup_work_limit(
            num_evaluations=constants.THIN_LENS_SOURCE_SETUP_EVALUATIONS
        )
        if work_units > work_limit:
            self._raise_validation_error_at_loc(
                f"ThinLensBeam source has {work_units:.2e} estimated setup work units, "
                f"which exceeds the maximum allowed {work_limit:.2e}. "
                "Consider reducing 'num_plane_waves', source plane size, or source "
                "'num_freqs'.",
                "sources",
                source_ind,
            )

    for monitor_ind, monitor in enumerate(self.monitors):
        if not isinstance(monitor, ThinLensOverlapMonitor):
            continue
        plane_cells = self._monitor_num_cells(monitor)
        num_evaluations = self._thin_lens_monitor_setup_evaluations(monitor)
        work_units = self._thin_lens_setup_work_units(
            plane_cells=plane_cells,
            num_plane_waves=monitor.num_plane_waves,
            num_freqs=len(monitor.freqs),
            num_evaluations=num_evaluations,
        )
        work_limit = self._thin_lens_setup_work_limit(num_evaluations=num_evaluations)
        if work_units > work_limit:
            self._raise_validation_error_at_loc(
                f"ThinLensOverlapMonitor has {work_units:.2e} estimated setup work units, "
                f"which exceeds the maximum allowed {work_limit:.2e}. "
                "Consider reducing 'num_plane_waves', monitor plane size, or monitor "
                "frequencies.",
                "monitors",
                monitor_ind,
            )
