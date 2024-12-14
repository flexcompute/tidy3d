from enum import Enum
from typing import Union

from tidy3d.components.tcad.bandgap import SlotboomNarrowingBandGap
from tidy3d.components.tcad.boundary.charge import CurrentBC, InsulatingBC, VoltageBC
from tidy3d.components.tcad.boundary.heat import ConvectionBC, HeatFluxBC, TemperatureBC
from tidy3d.components.tcad.generation_recombination import (
    AugerRecombination,
    RadiativeRecombination,
    ShockleyReedHallRecombination,
)
from tidy3d.components.tcad.mobility import CaugheyThomasMobility
from tidy3d.components.tcad.monitors.charge import (
    StaticCapacitanceMonitor,
    StaticChargeCarrierMonitor,
    StaticVoltageMonitor,
)
from tidy3d.components.tcad.monitors.heat import TemperatureMonitor
from tidy3d.components.tcad.source.coupled import HeatFromElectricSource
from tidy3d.components.tcad.source.heat import HeatSource, UniformHeatSource

# types of monitors that are accepted by heat simulation
HeatChargeMonitorTypes = Union[
    TemperatureMonitor,
    StaticVoltageMonitor,
    StaticChargeCarrierMonitor,
    StaticCapacitanceMonitor,
]
ChargeMonitorTypes = (StaticVoltageMonitor, StaticChargeCarrierMonitor, StaticCapacitanceMonitor)
HeatChargeSourceTypes = Union[HeatSource, HeatFromElectricSource, UniformHeatSource]
ChargeSourceTypes = None
HeatSourceTypes = (UniformHeatSource, HeatSource, HeatFromElectricSource)

MobilityModelTypes = Union[CaugheyThomasMobility]
RecombinationModelTypes = Union[
    AugerRecombination, RadiativeRecombination, ShockleyReedHallRecombination
]
BandGapModelTypes = Union[SlotboomNarrowingBandGap]

HeatBCTypes = (TemperatureBC, HeatFluxBC, ConvectionBC)
ElectricBCTypes = (VoltageBC, CurrentBC, InsulatingBC)
HeatChargeBCTypes = HeatChargeBoundaryConditionTypes = Union[
    TemperatureBC, HeatFluxBC, ConvectionBC, VoltageBC, CurrentBC, InsulatingBC
]


class HeatChargeSimulationTypes(str, Enum):
    """Enumeration of the types of simulations currently supported"""

    HEAT = "HEAT"
    CONDUCTION = "CONDUCTION"
    CHARGE = "CHARGE"
