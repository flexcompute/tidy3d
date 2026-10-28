"""Source normalization, background-medium, and thin-lens validation for ``Simulation``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from flexcompute.core._migration.em.material.medium.comparison import optical_medium_key

from tidy3d.components.geometry.base import Box
from tidy3d.components.medium import AnisotropicMedium, FullyAnisotropicMedium
from tidy3d.components.monitor import (
    AbstractGaussianOverlapMonitor,
    ThinLensOverlapMonitor,
)
from tidy3d.components.scene import Scene
from tidy3d.components.source.field import ThinLensBeam
from tidy3d.components.structure import Structure
from tidy3d.components.thin_lens import MAX_THIN_LENS_SETUP_WORK_UNITS, thin_lens_pupil_grid_samples
from tidy3d.config import config
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.components.material.types import StructureMediumType
    from tidy3d.components.medium import AbstractMedium
    from tidy3d.components.source.field import PlaneWave
    from tidy3d.components.types import ArrayFloat1D

from . import constants


def _source_medium(self: Any, source: PlaneWave) -> AbstractMedium | None:
    """Return a representative medium intersecting a homogeneous plane-wave source."""
    mediums = self.scene.intersecting_media(source, self.structures)
    if len({optical_medium_key(Structure._get_optical_medium(medium)) for medium in mediums}) > 1:
        return None
    return mediums.pop() if mediums else self.medium


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
    mediums: set[StructureMediumType],
    freqs: ArrayFloat1D,
    loc_root: str,
    loc_ind: int,
) -> None:
    """Validate background assumptions used by Gaussian-like beam formulas."""
    if len(mediums) < 1:
        self._raise_validation_error_at_loc(
            f"No medium detected on plane intersecting a {beam_obj.type}, "
            "indicating an unexpected error. Please create a github issue so "
            "that the problem can be investigated.",
            loc_root,
            loc_ind,
        )

    public_mediums = {
        optical_medium_key(Structure._get_public_optical_medium(medium)) for medium in mediums
    }
    if len(public_mediums) > 1:
        self._raise_validation_error_at_loc(
            f"{len(public_mediums)} different public optical mediums detected on plane "
            f"intersecting a {beam_obj.type}. Plane must be homogeneous.",
            loc_root,
            loc_ind,
        )

    public_medium = next(iter(public_mediums))
    if public_medium is None:
        self._raise_validation_error_at_loc(
            f"No optical medium detected on plane intersecting a {beam_obj.type}.",
            loc_root,
            loc_ind,
        )

    if isinstance(public_medium, AnisotropicMedium | FullyAnisotropicMedium):
        self._raise_validation_error_at_loc(
            f"An anisotropic medium is detected on plane intersecting a {beam_obj.type}. "
            f"{beam_obj.type} currently supports only isotropic background media.",
            loc_root,
            loc_ind,
        )
    if not public_medium.is_spatially_uniform:
        log.warning(
            f"Nonuniform custom medium detected on plane intersecting a {beam_obj.type}. "
            "Gaussian-like overlap setup assumes a homogeneous background medium.",
            custom_loc=[loc_root, loc_ind],
        )

    if not isinstance(beam_obj, ThinLensBeam | ThinLensOverlapMonitor):
        return
    min_background_index = self._thin_lens_min_background_index(public_medium, freqs)
    if beam_obj.numerical_aperture >= min_background_index:
        self._raise_validation_error_at_loc(
            f"{beam_obj.type} 'numerical_aperture' ({beam_obj.numerical_aperture:.4g}) "
            "must be less than the background refractive index on its plane.",
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
