"""File containing classes required for the setup of a DEVSIM case."""

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
    SteadyCapacitanceMonitor,
    SteadyFreeChargeCarrierMonitor,
    SteadyVoltageMonitor,
)
from tidy3d.components.tcad.monitors.heat import TemperatureMonitor
from tidy3d.components.tcad.source.coupled import HeatFromElectricSource
from tidy3d.components.tcad.source.heat import HeatSource, UniformHeatSource
from tidy3d.components.types import Union

MobilityModelTypes = Union[CaugheyThomasMobility]
RecombinationModelTypes = Union[
    AugerRecombination, RadiativeRecombination, ShockleyReedHallRecombination
]
BandGapModelTypes = Union[SlotboomNarrowingBandGap]

# types of monitors that are accepted by heat simulation
TCADMonitorTypes = Union[
    TemperatureMonitor,
    SteadyVoltageMonitor,
    SteadyFreeChargeCarrierMonitor,
    SteadyCapacitanceMonitor,
]
TCADSourceTypes = Union[HeatSource, HeatFromElectricSource, UniformHeatSource]
HeatChargeBCTypes = Union[
    TemperatureBC, HeatFluxBC, ConvectionBC, VoltageBC, CurrentBC, InsulatingBC
]
