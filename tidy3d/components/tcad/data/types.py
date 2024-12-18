from typing import Union

from tidy3d.components.tcad.data.monitor_data.charge import (
    SteadyCapacitanceData,
    SteadyChargeCarrierData,
    # SteadyPotentialData,
    SteadyVoltageData,
)
from tidy3d.components.tcad.data.monitor_data.heat import TemperatureData

HeatChargeMonitorDataTypes = Union[
    TemperatureData,
    SteadyVoltageData,
    SteadyChargeCarrierData,
    SteadyCapacitanceData,  # SteadyPotentialData
]
