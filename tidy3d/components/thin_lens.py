"""Shared helpers for thin-lens beam validation and sizing."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from flexcompute.core._migration.em.thin_lens.validation import (
    validate_lens_geometry,
    validate_num_plane_waves,
)
from flexcompute.core.validation import raise_first_error
from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import MICROMETER
from tidy3d.exceptions import AdjointError
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tidy3d.compat import Self
    from tidy3d.components.monitor import ThinLensOverlapMonitor
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.source.field import ThinLensBeam

DEFAULT_THIN_LENS_NUM_PLANE_WAVES = 51
MAX_THIN_LENS_NUM_PLANE_WAVES = 1000
MAX_THIN_LENS_SETUP_WORK_UNITS = 1_000_000_000_000


class AbstractThinLens(Tidy3dBaseModel, ABC):
    """Shared thin-lens angular-spectrum parameters."""

    numerical_aperture: PositiveFloat = Field(
        title="Numerical Aperture",
        description="Numerical aperture of the focused beam in the background medium. "
        "This must be less than the real refractive index on the component plane.",
    )

    waist_distance: float = Field(
        default=0.0,
        title="Waist Distance",
        description="Signed axial distance from the thin-lens focal plane to the component "
        "plane, measured along the rotated beam propagation direction. A positive value "
        "places the focal plane behind the component plane.",
        json_schema_extra={"units": MICROMETER},
    )

    fill_lens: bool = Field(
        default=True,
        title="Fill Lens",
        description="If ``True``, use a uniform circular pupil over the numerical aperture. "
        "If ``False``, use a Gaussian under-filled pupil defined by ``lens_diameter`` "
        "and ``beam_diameter``. If not provided, the lens is filled.",
    )

    lens_diameter: PositiveFloat | None = Field(
        default=None,
        title="Lens Diameter",
        description="Physical lens diameter used for Gaussian under-filled pupils. Required "
        "when ``fill_lens=False`` and ignored when ``fill_lens=True``.",
        json_schema_extra={"units": MICROMETER},
    )

    beam_diameter: PositiveFloat | None = Field(
        default=None,
        title="Beam Diameter",
        description="Incident Gaussian beam diameter used for Gaussian under-filled pupils. "
        "Required when ``fill_lens=False`` and ignored when ``fill_lens=True``.",
        json_schema_extra={"units": MICROMETER},
    )

    num_plane_waves: PositiveInt | tuple[PositiveInt, PositiveInt] = Field(
        default=DEFAULT_THIN_LENS_NUM_PLANE_WAVES,
        title="Number of Plane Waves",
        description="Number of angular-spectrum samples in the two tangential directions. "
        "If an integer is supplied, the same value is used in both directions. If not "
        f"provided, {DEFAULT_THIN_LENS_NUM_PLANE_WAVES} samples are used in each direction. "
        f"Each direction is limited to {MAX_THIN_LENS_NUM_PLANE_WAVES} samples.",
    )

    lens_offset: tuple[float, float] = Field(
        default=(0.0, 0.0),
        title="Lens Offset",
        description="Tangential offset of the focused field relative to the beam axis, "
        "implemented as a pupil phase ramp. The two entries follow the component's "
        "tangential-axis order after removing the normal axis from ``(x, y, z)``. For "
        "example, a y-normal component uses ``(x, z)``. This shifts the ideal focal field "
        "but does not model a decentered finite aperture with clipping or changed pupil "
        "amplitude. If not provided, the focal field is centered on the optical axis.",
        json_schema_extra={"units": MICROMETER},
    )

    @field_validator("num_plane_waves")
    @classmethod
    def _validate_num_plane_waves(
        cls, val: PositiveInt | tuple[PositiveInt, PositiveInt]
    ) -> PositiveInt | tuple[PositiveInt, PositiveInt]:
        """Ensure the angular grid is large enough to sample a circular pupil."""
        raise_first_error(validate_num_plane_waves(val), ValueError)
        return val

    @model_validator(mode="after")
    def _validate_lens_geometry(self) -> Self:
        """Validate coupled lens-pupil settings."""
        for diagnostic in validate_lens_geometry(self):
            self._raise_validation_error_at_loc(diagnostic.message, *diagnostic.path)
        return self

    @property
    def _num_plane_waves_xy(self) -> tuple[int, int]:
        """Return angular-spectrum sampling counts in the two tangential directions."""
        return thin_lens_num_plane_waves_xy(self.num_plane_waves)


def thin_lens_num_plane_waves_xy(
    num_plane_waves: PositiveInt | tuple[PositiveInt, PositiveInt],
) -> tuple[int, int]:
    """Return angular-spectrum sample counts in the two tangential directions."""
    if isinstance(num_plane_waves, tuple):
        return num_plane_waves[0], num_plane_waves[1]
    return num_plane_waves, num_plane_waves


def thin_lens_pupil_grid_samples(
    num_plane_waves: PositiveInt | tuple[PositiveInt, PositiveInt],
) -> int:
    """Return the square angular-spectrum sample count before circular aperture masking."""
    num_x, num_y = thin_lens_num_plane_waves_xy(num_plane_waves)
    return num_x * num_y


def validate_thin_lens_adjoint_setup(simulation: Simulation, traced_fields: list[tuple]) -> None:
    """Validate thin-lens physical assumptions before derivative compilation removes tracers."""
    from tidy3d.components.monitor import ThinLensOverlapMonitor
    from tidy3d.components.source.field import ThinLensBeam

    monitors = [
        monitor for monitor in simulation.monitors if isinstance(monitor, ThinLensOverlapMonitor)
    ]
    for monitor in monitors:
        if monitor._strip_traced_fields():
            raise AdjointError(
                f"ThinLensOverlapMonitor '{monitor.name}' must have fixed parameters. "
                "Differentiating the overlap measurement basis is not supported."
            )
    sources = [
        (index, source)
        for index, source in enumerate(simulation.sources)
        if isinstance(source, ThinLensBeam)
    ]
    if not sources and not monitors:
        return
    if not traced_fields and not simulation.medium._strip_traced_fields():
        return

    traced_sources = {path[1] for path in traced_fields if path[0] == "sources"}
    for index, source in sources:
        validate_thin_lens_adjoint_background(
            simulation,
            source,
            np.unique(np.concatenate((simulation._freqs_adjoint, source.frequency_grid))),
            require_reciprocal_profile=index in traced_sources,
            traced_fields=traced_fields,
        )
    for monitor in monitors:
        try:
            validate_thin_lens_adjoint_background(
                simulation, monitor, np.asarray(monitor.freqs), traced_fields=traced_fields
            )
        except AdjointError as error:
            log.warning(
                f"{error} This monitor can be recorded, but cannot contribute to an adjoint objective."
            )


def _thin_lens_component_label(component: ThinLensBeam | ThinLensOverlapMonitor) -> str:
    """Return a readable component label for capability errors."""
    return f"{component.type} '{component.name}'" if component.name else component.type


def validate_thin_lens_reciprocal_profile(
    component: ThinLensBeam | ThinLensOverlapMonitor,
) -> None:
    """Require the profile options used by the reciprocal thin-lens source."""
    label = _thin_lens_component_label(component)
    if not component.use_colocated_integration or not getattr(component, "colocate", True):
        raise AdjointError(f"{label} adjoints require colocated fields and integration.")
    if not getattr(component, "conjugated_dot_product", True):
        raise AdjointError(f"{label} adjoints require conjugated_dot_product=True.")


def validate_thin_lens_adjoint_background(
    simulation: Simulation,
    component: ThinLensBeam | ThinLensOverlapMonitor,
    frequencies: np.ndarray,
    *,
    require_reciprocal_profile: bool = True,
    traced_fields: Sequence[tuple] = (),
) -> None:
    """Reject unsupported profile dependence, retaining traced ownership on static simulations.

    A fixed source only needs the background-dependency check. Differentiating its
    parameters or an overlap amplitude additionally needs the reciprocal-profile
    contract. Inspect effective owners together so occluded tracers are harmless.
    """
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.medium import (
        AnisotropicMedium,
        AnisotropicMediumFromMedium2D,
        FullyAnisotropicMedium,
        Medium2D,
    )
    from tidy3d.components.scene import Scene
    from tidy3d.components.structure import Structure

    label = _thin_lens_component_label(component)
    structures = [
        Structure(
            geometry=Box(center=simulation.center, size=simulation.size), medium=simulation.medium
        ),
        *simulation.structures,
    ]
    traced_structures = {
        i for i, structure in enumerate(structures) if structure._strip_traced_fields()
    }
    traced_structures.update(path[1] + 1 for path in traced_fields if path[0] == "structures")
    if any(path[0] == "medium" for path in traced_fields):
        traced_structures.add(0)
    traced_numerical_structures = {path[1] + 1 for path in traced_fields if path[0] == "numerical"}
    if not require_reciprocal_profile and not (traced_structures or traced_numerical_structures):
        return

    sim_box = simulation.geometry.to_static()
    clipped = Box.from_bounds(
        *Box.bounds_intersection(component.to_static().bounds, sim_box.bounds)
    )
    if clipped.size.count(0.0) == 1:
        plane = clipped
    elif clipped.size.count(0.0) == 2 and sim_box.size.count(0.0) == 1:
        plane = sim_box
    else:
        raise AdjointError(f"{label} adjoints require a nonzero in-domain integration region.")
    regions = plane.intersections_with(clipped)
    measure = "area" if clipped.size.count(0.0) == 1 else "length"

    def active_plane_structures(structures: list[Structure]) -> set[int]:
        """Return original indices of structures owning part of the integration region."""
        structure_order = [
            0,
            *sorted(
                range(1, len(structures)),
                key=lambda index: structures[index]._priority(simulation.structure_priority_mode),
            ),
        ]
        owners = Scene._filter_structures_plane(
            structures=[structures[index] for index in structure_order],
            plane=plane,
            property_list=structure_order,
        )
        active = {
            index
            for index, shape in owners
            for region in regions
            if getattr(shape.intersection(region), measure) > 0
        }
        # The scene merger's polygon cleanup can discard partially occluded lines.
        # Resolve original and finalized sheet lines while honoring the same priorities.
        for position, index in enumerate(structure_order):
            if not isinstance(
                structures[index]._optical_medium, (Medium2D, AnisotropicMediumFromMedium2D)
            ):
                continue
            for section in plane.intersections_with(structures[index].geometry.to_static()):
                if section.area > 0:
                    continue
                for region in regions:
                    intersection = section.intersection(region)
                    for later in structure_order[position + 1 :]:
                        for mask in plane.intersections_with(
                            structures[later].geometry.to_static()
                        ):
                            intersection = intersection.difference(mask)
                    if intersection.length > 0:
                        active.add(index)
        return active

    if traced_structures or traced_numerical_structures:
        active = active_plane_structures(structures)
        if active & traced_numerical_structures:
            raise AdjointError(
                f"{label} adjoints do not support traced numerical structures on the plane."
            )
        if active & traced_structures:
            raise AdjointError(
                f"{label} adjoints do not support traced background-medium or geometry parameters "
                "on the plane."
            )
    if not require_reciprocal_profile:
        return

    validate_thin_lens_reciprocal_profile(component)

    # A modal PEC frame is represented by zero-area lines on its source plane, so
    # Scene's area ownership intentionally omits it. It still changes the discrete
    # source profile and must invalidate this reciprocal construction.
    if any(
        not shape.intersection(region).is_empty
        for frame in simulation._modal_plane_frames
        for shape in plane.intersections_with(frame.geometry)
        for region in regions
    ):
        raise AdjointError(
            f"{label} adjoints require one homogeneous medium on the component plane."
        )

    # The solver turns 2D materials and lumped elements into volumetric media and
    # adds modal PEC frames. Use its finalized structures so the accepted plane
    # matches the material seen by the solver.
    effective_structures = [
        Structure(
            geometry=Box(center=simulation.center, size=simulation.size), medium=simulation.medium
        ),
        *simulation._finalized_volumetric_structures,
    ]
    effective_active = active_plane_structures(effective_structures)
    media = {
        Structure._get_optical_medium(effective_structures[index].medium)
        for index in effective_active
    }
    if len(media) != 1:
        raise AdjointError(
            f"{label} adjoints require one homogeneous medium on the component plane."
        )
    medium = next(iter(media))
    if (
        medium is None
        or isinstance(medium, (AnisotropicMedium, FullyAnisotropicMedium))
        or not medium.is_spatially_uniform
        or medium.is_time_modulated
        or medium.is_nonlinear
    ):
        raise AdjointError(
            f"{label} adjoints require a homogeneous, isotropic, linear, static plane medium."
        )
    medium = medium.to_static()
    index = np.asarray(medium.background_index_from_freqs(frequencies), dtype=complex)
    if np.any(index.imag != 0) or np.any(index.real <= 0):
        raise AdjointError(
            f"{label} adjoints require a lossless plane medium with positive refractive index."
        )
