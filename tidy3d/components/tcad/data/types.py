"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from tidy3d.components.tcad.data.monitor_data.charge import (
    SelfHeatingData,
    SteadyCapacitanceData,
    SteadyChargeResidualData,
    SteadyCurrentDensityData,
    SteadyElectricFieldData,
    SteadyEnergyBandData,
    SteadyFreeCarrierData,
    SteadyGenerationRecombinationData,
    SteadyPotentialData,
)
from tidy3d.components.tcad.data.monitor_data.heat import TemperatureData

TCADMonitorDataType = (
    TemperatureData
    | SteadyPotentialData
    | SteadyFreeCarrierData
    | SteadyElectricFieldData
    | SteadyEnergyBandData
    | SteadyCapacitanceData
    | SteadyCurrentDensityData
    | SteadyChargeResidualData
    | SteadyGenerationRecombinationData
    | SelfHeatingData
)
