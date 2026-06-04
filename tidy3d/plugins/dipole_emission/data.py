"""Result data objects for the dipole emission plugin."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.plugins.dipole_emission.data_array import (
    DipoleEmissionStudyDataArray,
    DipoleEmissionStudyPositionDataArray,
)
from tidy3d.plugins.dipole_emission.study import DipoleEmissionStudy


class DipoleEmissionStudyData(Tidy3dBaseModel):
    """Reduced result from a :class:`.DipoleEmissionStudy`.

    The default data arrays are summed over the study ``positions`` using any
    position- and axis-dependent ``position_weights`` and are indexed by
    Cartesian dipole orientation, far-field polarization, observation angle, and
    frequency. They represent angular radiation intensity per squared electric
    dipole moment, with dipole moment expressed in C*um, into the collection
    half-space defined by the study.
    ``radiation_intensity_transfer`` is derived by dividing the stored intensity
    by the study source spectrum. Optional position-resolved arrays are present
    only when the study
    ``store_position_indexes`` is nonempty. Raw ``BatchData`` is intentionally
    not stored here; use
    ``DipoleEmissionStudy.run(..., return_batch_data=True)`` or manual batch
    execution when diagnostic monitor data is needed.
    """

    radiation_intensity: DipoleEmissionStudyDataArray = Field(
        title="Radiation Intensity",
        description=(
            "Radiated angular power density per squared electric dipole moment "
            "in C*um. The ``source_time`` spectrum from the study is included."
        ),
    )

    radiation_intensity_at_positions: DipoleEmissionStudyPositionDataArray | None = Field(
        None,
        title="Position-Resolved Radiation Intensity",
        description=(
            "Radiation intensity at positions selected by the study ``store_position_indexes``."
        ),
    )

    study: DipoleEmissionStudy = Field(
        title="Study",
        description="Study definition used to generate this result.",
    )

    @property
    def radiation_intensity_transfer(self) -> DipoleEmissionStudyDataArray:
        """Radiation-intensity transfer with the study source spectrum divided out."""
        pulse_spectrum_abs2 = np.asarray(
            self.study.pulse_spectrum_abs2,
            dtype=self.radiation_intensity.dtype,
        )
        return DipoleEmissionStudyDataArray(
            self.radiation_intensity.values / pulse_spectrum_abs2[None, None, None, :],
            dims=DipoleEmissionStudyDataArray._dims,
            coords=self.radiation_intensity.coords,
        )

    @property
    def radiation_intensity_transfer_at_positions(
        self,
    ) -> DipoleEmissionStudyPositionDataArray | None:
        """Position-resolved transfer for selected stored position indexes."""
        if self.radiation_intensity_at_positions is None:
            return None
        pulse_spectrum_abs2 = np.asarray(
            self.study.pulse_spectrum_abs2,
            dtype=self.radiation_intensity_at_positions.dtype,
        )
        return DipoleEmissionStudyPositionDataArray(
            self.radiation_intensity_at_positions.values
            / pulse_spectrum_abs2[None, None, None, None, :],
            dims=DipoleEmissionStudyPositionDataArray._dims,
            coords=self.radiation_intensity_at_positions.coords,
        )
