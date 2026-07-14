"""Shared helpers for thin-lens beam validation and sizing."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import MICROMETER

if TYPE_CHECKING:
    from tidy3d.compat import Self

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
        0.0,
        title="Waist Distance",
        description="Signed axial distance from the thin-lens focal plane to the component "
        "plane, measured along the rotated beam propagation direction. A positive value "
        "places the focal plane behind the component plane.",
        json_schema_extra={"units": MICROMETER},
    )

    fill_lens: bool = Field(
        True,
        title="Fill Lens",
        description="If ``True``, use a uniform circular pupil over the numerical aperture. "
        "If ``False``, use a Gaussian under-filled pupil defined by ``lens_diameter`` "
        "and ``beam_diameter``. If not provided, the lens is filled.",
    )

    lens_diameter: PositiveFloat | None = Field(
        None,
        title="Lens Diameter",
        description="Physical lens diameter used for Gaussian under-filled pupils. Required "
        "when ``fill_lens=False`` and ignored when ``fill_lens=True``.",
        json_schema_extra={"units": MICROMETER},
    )

    beam_diameter: PositiveFloat | None = Field(
        None,
        title="Beam Diameter",
        description="Incident Gaussian beam diameter used for Gaussian under-filled pupils. "
        "Required when ``fill_lens=False`` and ignored when ``fill_lens=True``.",
        json_schema_extra={"units": MICROMETER},
    )

    num_plane_waves: PositiveInt | tuple[PositiveInt, PositiveInt] = Field(
        DEFAULT_THIN_LENS_NUM_PLANE_WAVES,
        title="Number of Plane Waves",
        description="Number of angular-spectrum samples in the two tangential directions. "
        "If an integer is supplied, the same value is used in both directions. If not "
        f"provided, {DEFAULT_THIN_LENS_NUM_PLANE_WAVES} samples are used in each direction. "
        f"Each direction is limited to {MAX_THIN_LENS_NUM_PLANE_WAVES} samples.",
    )

    lens_offset: tuple[float, float] = Field(
        (0.0, 0.0),
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
        return validate_thin_lens_num_plane_waves(val)

    @model_validator(mode="after")
    def _validate_lens_geometry(self) -> Self:
        """Validate coupled lens-pupil settings."""
        if self.fill_lens:
            return self
        if self.lens_diameter is None:
            self._raise_validation_error_at_loc(
                "'lens_diameter' is required when 'fill_lens=False'.",
                "lens_diameter",
            )
        if self.beam_diameter is None:
            self._raise_validation_error_at_loc(
                "'beam_diameter' is required when 'fill_lens=False'.",
                "beam_diameter",
            )
        if self.beam_diameter > self.lens_diameter:
            self._raise_validation_error_at_loc(
                "'beam_diameter' must not exceed 'lens_diameter'.",
                "beam_diameter",
            )
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


def validate_thin_lens_num_plane_waves(
    num_plane_waves: PositiveInt | tuple[PositiveInt, PositiveInt],
) -> PositiveInt | tuple[PositiveInt, PositiveInt]:
    """Validate thin-lens angular-spectrum sample counts."""
    values = thin_lens_num_plane_waves_xy(num_plane_waves)
    if any(num < 3 for num in values):
        raise ValueError("'num_plane_waves' must be at least 3 in each direction.")
    if any(num > MAX_THIN_LENS_NUM_PLANE_WAVES for num in values):
        raise ValueError(
            "'num_plane_waves' must not exceed "
            f"{MAX_THIN_LENS_NUM_PLANE_WAVES} in either direction."
        )
    return num_plane_waves
