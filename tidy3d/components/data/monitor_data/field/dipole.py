from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import (
    Field,
)

from tidy3d.components.data.data_array import (
    DipoleEmissionDataArray,
    DipoleEmissionPositionDataArray,
)
from tidy3d.components.data.monitor_data.base import MonitorData

if TYPE_CHECKING:
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

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[Source]:
        """Reject adjoint use until dipole-emission adjoint sources are available."""
        if not dataset_names:
            return []

        raise Tidy3dNotImplementedError(
            "Adjoint objectives depending on DipoleEmissionData are currently unsupported."
        )
