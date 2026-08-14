"""Definition-derived sampling resolutions for adjoint shape gradients.

The geometry VJP code can choose its surface quadrature spacing from monitor
permittivity data during backward postprocessing. For grouped adjoint solves, that
can make the sampling depend on how adjoint frequencies were partitioned. This
module mirrors the same material rules, but computes one per-structure resolution
from the simulation definition and the full adjoint frequency set before chunked
gradient postprocessing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from tidy3d.components.medium import AbstractCustomMedium, Medium2D
from tidy3d.components.structure import Structure
from tidy3d.config import config
from tidy3d.constants import C_0
from tidy3d.exceptions import AdjointError, Tidy3dImportError
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.medium import MediumType
    from tidy3d.components.simulation import Simulation


class SamplingResolution(NamedTuple):
    """Definition-derived resolutions for one structure's shape-gradient sampling."""

    spacing: float
    """Adaptive surface quadrature spacing in microns."""

    material_length_scale: float
    """Minimum material wavelength or skin-depth length scale in microns."""

    material_wavelength: float
    """Material discretization wavelength in microns for triangulation-based sampling."""


def adjoint_sampling_resolution(simulation: Simulation, structure_index: int) -> SamplingResolution:
    """Compute definition-derived sampling resolutions for one traced structure.

    The calculation mirrors ``DerivativeInfo.adaptive_vjp_spacing()`` and
    ``discretization_wavelength()`` but uses the simulation definition instead of
    postprocessed monitor data. It scans media whose bounding boxes overlap the
    structure's adjoint monitor region and evaluates them at the full adjoint
    frequency set, so the result is independent of solver frequency chunking or
    adjoint-source grouping.
    """

    frequencies = np.asarray(simulation._freqs_adjoint, dtype=float)
    if frequencies.size == 0:
        raise AdjointError(
            "Cannot compute adjoint sampling resolution: the simulation contains no "
            "monitor frequencies to build adjoint data for."
        )

    structure = simulation.structures[structure_index]
    plane = simulation if simulation.size.count(0.0) == 1 else None
    monitor_box = structure._adjoint_monitor_box(grid=simulation.grid, plane=plane)

    eps_values = _region_eps_values(
        simulation=simulation, monitor_box=monitor_box, frequencies=frequencies
    )
    wavelength_min = C_0 / frequencies.max()
    material_length_scale = _min_spacing_from_eps(
        eps_values=eps_values, frequencies=frequencies, wavelength_min=wavelength_min
    )

    return SamplingResolution(
        spacing=_adaptive_spacing(
            material_length_scale=material_length_scale, wavelength_min=wavelength_min
        ),
        material_length_scale=material_length_scale,
        material_wavelength=_material_wavelength(
            eps_values=eps_values, wavelength_min=wavelength_min
        ),
    )


def _region_eps_values(
    simulation: Simulation, monitor_box: Box, frequencies: np.ndarray
) -> np.ndarray:
    """Return permittivity values for media overlapping ``monitor_box``.

    Bounding-box overlap is intentionally conservative: including an extra medium can
    only make the chosen sampling resolution as fine or finer than the exact local
    monitor-derived rule. Structures use the simulation's grid-aware volumetric
    equivalents so ``Medium2D`` sheets include their local cell thickness and adjacent
    media. The simulation background and structure background media are included
    because both can define outside-side shape-gradient material data.
    """

    mediums = [simulation.medium]
    for candidate in simulation.volumetric_structures:
        if not monitor_box.intersects(candidate.geometry):
            continue
        mediums.append(candidate.medium)
        if candidate.background_medium is not None:
            mediums.append(candidate.background_medium)

    values = []
    for medium in mediums:
        optical_medium = Structure._get_optical_medium(medium)
        if optical_medium is None:
            continue
        values.append(
            _medium_eps_values(
                medium=optical_medium, frequencies=frequencies, monitor_box=monitor_box
            )
        )

    if not values:
        raise AdjointError(
            "Cannot compute adjoint sampling resolution: no optical media overlap "
            "the structure adjoint monitor region."
        )
    return np.concatenate(values)


def _medium_eps_values(medium: MediumType, frequencies: np.ndarray, monitor_box: Box) -> np.ndarray:
    """Return representative complex permittivity values at ``frequencies``.

    Spatially varying custom media are first reduced to the adjoint monitor region
    for the traced structure, preventing remote data in the same custom medium from
    setting an unrelated structure's sampling resolution. That reduction needs the
    optional vtk dependency for unstructured data; without it the full dataset is used,
    which can only choose a finer resolution.
    """

    if isinstance(medium, Medium2D):
        raise AdjointError(
            "Cannot compute adjoint sampling resolution from an unconverted Medium2D. "
            "Use the simulation's grid-aware volumetric structures."
        )

    if isinstance(medium, AbstractCustomMedium):
        try:
            medium = medium.sel_inside(bounds=monitor_box.bounds)
        except Tidy3dImportError:
            # Reducing unstructured custom data to a region needs the optional vtk
            # dependency, but the permittivity accessors below do not. Keep the unreduced
            # medium: extra remote values can only make the resolution as fine or finer,
            # which is the same conservatism as the bounding-box medium selection above.
            log.warning(
                "Could not restrict unstructured custom medium data to the adjoint monitor "
                "region because the optional 'vtk' dependency is unavailable. Falling back to "
                "the full dataset, which may choose a finer adjoint sampling resolution than "
                "necessary. Install 'vtk' for region-restricted sampling.",
                log_once=True,
            )
        values = []
        for frequency in frequencies:
            eps_min, eps_max = medium._eps_bounds(frequency=frequency)
            values.extend([complex(eps_min), complex(eps_max)])
            values.extend(complex(value) for value in medium.eps_diagonal(frequency))
        return np.asarray(values, dtype=complex)

    values = [medium.eps_diagonal(frequency) for frequency in frequencies]
    return np.asarray(values, dtype=complex).ravel()


def _min_spacing_from_eps(
    eps_values: np.ndarray, frequencies: np.ndarray, wavelength_min: float
) -> float:
    """Return the material length scale used by adaptive quadrature spacing."""

    eps_real = eps_values.real
    dx_candidates = []

    if np.any(eps_real > 0):
        eps_max = eps_real[eps_real > 0].max()
        dx_candidates.append(wavelength_min / np.sqrt(eps_max))

    if np.any(eps_real <= 0):
        omega = 2 * np.pi * frequencies.max()
        eps_neg = eps_real[eps_real <= 0]
        dx_candidates.append(C_0 / (omega * np.sqrt(np.abs(eps_neg).max())))

    return min(dx_candidates)


def _adaptive_spacing(material_length_scale: float, wavelength_min: float) -> float:
    """Return adaptive quadrature spacing with the same clipping rule as VJP code."""

    computed_spacing = config.adjoint.default_wavelength_fraction * material_length_scale
    min_allowed_spacing = wavelength_min * config.adjoint.minimum_spacing_fraction

    if computed_spacing < min_allowed_spacing:
        log.warning(
            f"Based on the material, the adaptive spacing for adjoint surface sampling "
            f"would be {computed_spacing:.3e} μm. The spacing has been clipped to "
            f"{min_allowed_spacing:.3e} μm to prevent a performance degradation.",
            log_once=True,
        )

    return max(computed_spacing, min_allowed_spacing)


def _material_wavelength(eps_values: np.ndarray, wavelength_min: float) -> float:
    """Return material discretization wavelength with the existing clipping rule."""

    max_refractive_index = max(1.0, float(np.sqrt(np.abs(eps_values).max())))
    wvl_mat = wavelength_min / max_refractive_index

    min_wvl_mat = config.adjoint.min_wvl_fraction * wavelength_min
    if wvl_mat < min_wvl_mat:
        log.warning(
            f"The minimum wavelength inside the sampled materials is {wvl_mat:.3e} μm, which "
            f"would create a large number of discretization points for computing the gradient. "
            f"To prevent performance degradation, the discretization wavelength has "
            f"been clipped to {min_wvl_mat:.3e} μm.",
            log_once=True,
        )

    return max(wvl_mat, min_wvl_mat)
