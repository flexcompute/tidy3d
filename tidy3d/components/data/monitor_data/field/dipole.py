from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from pydantic import (
    Field,
    model_validator,
)

from tidy3d.components.data.data_array import (
    DipoleEmissionDataArray,
    DipoleEmissionPositionDataArray,
)
from tidy3d.components.data.monitor_data.base import MonitorData

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.source.base import Source
from tidy3d.components.monitor import DipoleEmissionMonitor
from tidy3d.exceptions import Tidy3dNotImplementedError


class DipoleEmissionData(MonitorData):
    """Data associated with a :class:`.DipoleEmissionMonitor`.

    The default arrays are summed over all sampled dipole positions using the
    monitor position- and axis-dependent ``position_weights``. Optional
    position-resolved arrays are present only when ``store_position_indexes`` is
    nonempty.
    """

    monitor: DipoleEmissionMonitor = Field(
        ..., title="Monitor", description="Dipole-emission monitor."
    )

    radiation_intensity: DipoleEmissionDataArray = Field(
        ...,
        title="Radiation Intensity",
        description=(
            "Radiated angular power density per squared electric dipole moment "
            "with dipole moment expressed in C*um."
        ),
    )

    radiation_intensity_at_positions: DipoleEmissionPositionDataArray | None = Field(
        default=None,
        title="Position-Resolved Radiation Intensity",
        description="Radiation intensity at selected stored position indexes.",
    )

    @model_validator(mode="after")
    def _frequencies_match_monitor(self) -> Self:
        """Ensure stored frequency coordinates match the associated monitor."""
        monitor_freqs = np.asarray(self.monitor.freqs)
        arrays = {
            "radiation_intensity": self.radiation_intensity,
            "radiation_intensity_at_positions": self.radiation_intensity_at_positions,
        }
        for array_name, array in arrays.items():
            if array is None:
                continue
            freqs = np.asarray(array.coords["f"].values)
            if not np.array_equal(freqs, monitor_freqs):
                self._raise_validation_error_at_loc(
                    "Radiation-intensity frequency coordinates must match the monitor.",
                    array_name,
                )
        return self

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[Source]:
        """Reject adjoint use until dipole-emission adjoint sources are available."""
        if not dataset_names:
            return []

        raise Tidy3dNotImplementedError(
            "Adjoint objectives depending on DipoleEmissionData are currently unsupported."
        )
