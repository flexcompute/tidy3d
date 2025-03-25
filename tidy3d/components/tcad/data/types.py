"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from typing import Union

from tidy3d.components.tcad.data.monitor_data.charge import (
    SteadyCapacitanceData,
    SteadyEnergyBandData,
    SteadyFreeCarrierData,
    SteadyPotentialData,
)
from tidy3d.components.tcad.data.monitor_data.heat import TemperatureData

TCADMonitorDataType = Union[
    TemperatureData,
    SteadyPotentialData,
    SteadyFreeCarrierData,
    SteadyEnergyBandData,
    SteadyCapacitanceData,
]
