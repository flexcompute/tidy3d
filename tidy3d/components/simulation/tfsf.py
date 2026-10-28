"""TFSF, fixed-angle, and dipole-emission source validation for ``Simulation``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from flexcompute.core._migration.em.material.medium.comparison import optical_medium_key

from tidy3d.components.base import cached_property
from tidy3d.components.geometry.base import Box
from tidy3d.components.medium import (
    AbstractCustomMedium,
    FullyAnisotropicMedium,
    LossyMetalMedium,
    PECMedium,
    PMCMedium,
)
from tidy3d.components.scene import Scene
from tidy3d.components.source.base import Source
from tidy3d.components.source.field import TFSF, FixedAngleSpec, PlaneWave
from tidy3d.components.structure import Structure
from tidy3d.constants import C_0, inf
from tidy3d.exceptions import (
    SetupError,
)
from tidy3d.packaging import (
    disable_local_subpixel,
)

if TYPE_CHECKING:
    from tidy3d.components.medium import AbstractMedium
    from tidy3d.components.source.utils import SourceType
    from tidy3d.components.types import ArrayFloat1D


@staticmethod
def _get_periodic_fixed_angle_sources(
    sources: tuple[SourceType, ...],
) -> tuple[SourceType, ...]:
    """Periodic fixed-angle :class:`PlaneWave` sources.

    ``TFSF`` sources with ``FixedAngleSpec`` are intentionally
    excluded — only a fixed-angle :class:`PlaneWave` is a periodic
    fixed-angle source, and the FDTD rule ``validate_fixed_angle_components``
    prohibits combining the two.
    """

    return [
        source
        for source in sources
        if isinstance(source, PlaneWave) and source._is_periodic_fixed_angle
    ]


def _dipole_emission_tfsf_source(self: Any) -> TFSF:
    """Return the single TFSF source associated with a dipole-emission monitor."""
    if len(self.sources) != 1 or not isinstance(self.sources[0], TFSF):
        raise SetupError(
            "DipoleEmissionMonitor postprocessing requires a simulation with exactly "
            "one source, and that source must be a TFSF source."
        )
    return self.sources[0]


@staticmethod
def _dipole_emission_tfsf_injection_plane(source: TFSF) -> Box:
    """Planar TFSF injection face used as the dipole-emission collection side."""
    injection_plane_size = list(source.size)
    injection_plane_size[source.injection_axis] = 0.0
    return Box(center=tuple(source.injection_plane_center), size=tuple(injection_plane_size))


def _dipole_emission_tfsf_injection_medium(
    self: Any,
    source: TFSF,
) -> AbstractMedium:
    """Return the single public optical medium intersecting the TFSF injection face."""
    injection_plane = self._dipole_emission_tfsf_injection_plane(source)
    simulation_background = Structure(
        geometry=Box(size=self.size, center=self.center),
        medium=self.medium,
    )
    plane_media = Scene.intersecting_media(
        injection_plane, [simulation_background, *list(self.structures or [])]
    )
    public_plane_media = {
        optical_medium_key(Structure._get_public_optical_medium(medium)) for medium in plane_media
    }
    if len(public_plane_media) != 1:
        raise SetupError(
            "DipoleEmissionMonitor requires a homogeneous medium on the TFSF injection "
            f"plane; found {len(public_plane_media)} public optical media."
        )
    public_medium = next(iter(public_plane_media))
    if public_medium is None:
        raise SetupError(
            "DipoleEmissionMonitor requires an optical medium on the TFSF injection plane."
        )
    return public_medium


def _dipole_emission_background_index(
    self: Any,
    source: TFSF,
    freqs: ArrayFloat1D,
) -> float:
    """Validate and return the real nondispersive index on the TFSF injection side."""
    medium = self._dipole_emission_tfsf_injection_medium(source)
    background_n = np.asarray(medium.background_index_from_freqs(freqs), dtype=complex)
    if not np.allclose(background_n.imag, 0.0):
        raise SetupError(
            "DipoleEmissionMonitor requires real TFSF injection-side refractive index."
        )
    if not np.allclose(background_n.real, background_n.real[0], rtol=1e-12, atol=0.0):
        raise SetupError(
            "DipoleEmissionMonitor requires nondispersive TFSF injection-side refractive index."
        )
    return float(background_n.real[0])


def _aux_tfsf_source(self: Any, source: TFSF) -> PlaneWave:
    """Create the auxiliary plane wave source for a give TFSF source."""
    # center and size of the plane wave source
    source_size = [inf] * 3
    source_size[source.injection_axis] = 0
    source_center = list(source.injection_plane_center)

    # since we need to access values of the aux self at dual grid locations below the actual
    # injection plane, we need to place the aux sim's source at least one full cell below the
    # location of the injection plane; for good measure, we'll offset the source by two cells
    src_grid = self.discretize(source, extend=False)
    src_grid_sizes = src_grid.sizes.to_list
    if source.direction == "+":
        offset = -sum(src_grid_sizes[source.injection_axis][0:2])
    else:
        offset = sum(src_grid_sizes[source.injection_axis][-1:-3:-1])
    source_center[source.injection_axis] += offset

    # Make sure that the new source center is within the simulation bounds
    sim_axis_bounds = [self.bounds[i][source.injection_axis] for i in range(2)]
    if (
        source_center[source.injection_axis] < sim_axis_bounds[0]
        or source_center[source.injection_axis] > sim_axis_bounds[1]
    ):
        raise SetupError(
            "The TFSF source is too close to the simulation domain boundary along the "
            "injection axis. Slightly increase the simulation domain size along that "
            "dimension, or decrease the source size."
        )

    # Pre-compensate the source-time so the unit-amplitude
    # reference lands at the injection plane (the box face), not
    # at the aux source plane that sits ``|offset|`` along the
    # propagation direction. For lossless source-side media this
    # is purely a phase shift; for lossy media it also pre-
    # amplifies by ``exp(+Im(kz)·|offset|)`` to undo the decay
    # over ``|offset|``. The medium at the injection plane is
    # queried stacking-aware (a structure overlapping the source
    # plane changes the local ``n``); fall back to ``self.medium``
    # if multiple public optical media are visible there.
    source_time = source.source_time
    injection_plane_size = list(source.size)
    injection_plane_size[source.injection_axis] = 0.0
    injection_plane_probe = Box(
        center=tuple(source.injection_plane_center),
        size=tuple(injection_plane_size),
    )
    injection_bg = Structure(geometry=Box(size=self.size, center=self.center), medium=self.medium)
    plane_mediums = Scene.intersecting_media(
        injection_plane_probe, [injection_bg, *list(self.structures or [])]
    )
    public_plane_media = {
        optical_medium_key(Structure._get_public_optical_medium(medium)) for medium in plane_mediums
    }
    injection_medium = (
        next(iter(public_plane_media))
        if len(public_plane_media) == 1
        else Structure._get_public_optical_medium(self.medium)
    )
    try:
        f0 = float(source_time._freq0)
        n_complex = complex(injection_medium.background_index_from_freqs([f0])[0])
    except (AttributeError, NotImplementedError):
        n_complex = None
    if n_complex is not None:
        kz_continuum_at_f0 = (
            (2.0 * np.pi * f0 / C_0) * n_complex * float(np.cos(source.angle_theta))
        )
        compensation = complex(np.exp(-1j * kz_continuum_at_f0 * abs(offset)))
        amp_factor = float(np.abs(compensation))
        phase_shift = float(np.angle(compensation))
        if not (amp_factor == 1.0 and phase_shift == 0.0):
            source_time = source_time.updated_copy(
                amplitude=amp_factor * source_time.amplitude,
                phase=source_time.phase + phase_shift,
            )

    # Note: broadband injection for TFSF not currently supported.
    return PlaneWave(
        size=source_size,
        center=source_center,
        source_time=source_time,
        angle_theta=source.angle_theta,
        angle_phi=source.angle_phi,
        pol_angle=source.pol_angle,
        direction=source.direction,
        num_freqs=source.num_freqs,
        use_colocated_integration=source.use_colocated_integration,
    )


@cached_property
def aux_fields(self: Any) -> list[str]:
    """All aux fields available in the simulation."""
    fields = []
    for medium in self.scene.mediums:
        if medium.nonlinear_spec is not None:
            fields += medium.nonlinear_spec.aux_fields
    return fields


@disable_local_subpixel
def _validate_tfsf_structure_intersections(self: Any) -> None:
    """Error if the 4 sidewalls of a TFSF box don't all intersect the same structures.
    This validator may need to compute permittivities on the grid, so it is called
    pre-upload rather than at the time of definition. Also errors if any side wall
    intersects with a custom medium or a fully anisotropic media.
    """
    for source_idx, source in enumerate(self.sources):
        if not isinstance(source, TFSF):
            continue
        # get all TFSF surfaces
        tfsf_surfaces = Source.surfaces(
            center=source.center, size=source.size, source_time=source.source_time
        )
        sidewall_surfaces = []
        sidewall_structs = []
        # get the structures that intersect each sidewall
        for surface in tfsf_surfaces:
            # ignore the sidewall surface if it falls outside the simulation domain
            if not self.intersects(surface):
                continue

            if surface.name[-2] != "xyz"[source.injection_axis]:
                sidewall_surfaces.append(surface)
                intersecting_structs = Scene.intersecting_structures(
                    test_object=surface, structures=self.structures
                )

                if any(
                    isinstance(struct.medium, AbstractCustomMedium | FullyAnisotropicMedium)
                    for struct in intersecting_structs
                ):
                    raise SetupError(
                        f"The surfaces of TFSF source '{source.name}' must not intersect any "
                        "structures containing a 'CustomMedium' or a 'FullyAnisotropicMedium'."
                    )

                # Surface-BC media (``LossyMetalMedium``, ``PECMedium``, ``PMCMedium``)
                # are only rejected at sidewalls of a fixed-angle TFSF source; the
                # constant-in-plane-k TFSF supports them. Move the structure fully
                # inside the box, or switch the source to ``FixedInPlaneKSpec``.
                if isinstance(source.angular_spec, FixedAngleSpec) and any(
                    isinstance(struct.medium, PECMedium | PMCMedium)
                    or (
                        isinstance(struct.medium, LossyMetalMedium) and not struct.medium.penetrable
                    )
                    for struct in intersecting_structs
                ):
                    self._raise_validation_error_at_loc(
                        f"Fixed-angle TFSF source '{source.name}' cannot have its sidewalls "
                        "intersect a 'LossyMetalMedium', 'PECMedium', or 'PMCMedium'. Move the "
                        "structure fully inside the TFSF box, or use 'FixedInPlaneKSpec' "
                        "instead.",
                        "sources",
                        source_idx,
                    )

                # if no structures intersect, just add a phantom associated with the simulation
                # background, to prevent false positives below
                if not intersecting_structs:
                    sidewall_structs.append(
                        [
                            Structure(
                                geometry=Box(center=self.center, size=self.size),
                                medium=self.medium,
                            )
                        ]
                    )
                else:
                    sidewall_structs.append(intersecting_structs)

        # let the first wall be a reference, and compare the rest of them to the structures
        # intersected by that reference wall
        if len(sidewall_structs) > 1:
            ref_structs = sidewall_structs[0]
            test_structs = sidewall_structs[1:]
            if all(structs == ref_structs for structs in test_structs):
                continue

            # if the == test doesn't pass, that doesn't mean the materials are necessarily
            # different, because it's possible that the sidewalls encounter different
            # `Structure` objects but with an identical material profile, which is still
            # a valid setup; in this case, compute the epsilon profile on the grid for each
            # side wall - the profiles must be the same along the injection axis, so we take
            # a single "stripe" of epsilon as the reference and subtract it from all other
            # stripes, which should result in zero if all the epsilon profiles are the same
            freq0 = source.source_time._freq0
            _, plane_axs = source.pop_axis("xyz", axis=source.injection_axis)
            ref_eps = self.epsilon(box=sidewall_surfaces[0], coord_key="centers", freq=freq0)
            kwargs = {plane_axs[0]: 0, plane_axs[1]: 0}
            ref_eps = ref_eps.isel(**kwargs)
            for surface in sidewall_surfaces:
                test_eps = self.epsilon(box=surface, coord_key="centers", freq=freq0) - ref_eps
                if not np.allclose(test_eps.to_numpy(), 0):
                    raise SetupError(
                        f"All sidewalls of the TFSF source '{source.name}' must intersect "
                        "the same media along the injection axis "
                        f" '{'xyz'[source.injection_axis]}'."
                    )


@cached_property
def _fixed_angle_sources(self: Any) -> tuple[SourceType, ...]:
    """List of plane wave sources with ``FixedAngleSpec``."""
    return self._get_periodic_fixed_angle_sources(self.sources)


@cached_property
def _is_periodic_fixed_angle(self: Any) -> bool:
    """Whether the simulation contains a periodic fixed-angle source —
    i.e. a fixed-angle :class:`PlaneWave` with non-zero ``angle_theta``."""
    return len(self._fixed_angle_sources) > 0
