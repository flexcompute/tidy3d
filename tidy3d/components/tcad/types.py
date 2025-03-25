"""File containing classes required for the setup of a DEVSIM case."""

from tidy3d.components.tcad.bandgap import SlotboomBandGapNarrowing
from tidy3d.components.tcad.boundary.charge import CurrentBC, InsulatingBC, VoltageBC
from tidy3d.components.tcad.boundary.heat import ConvectionBC, HeatFluxBC, TemperatureBC
from tidy3d.components.tcad.generation_recombination import (
    AugerRecombination,
    RadiativeRecombination,
    ShockleyReedHallRecombination,
)
from tidy3d.components.tcad.mobility import CaugheyThomasMobility, ConstantMobilityModel
from tidy3d.components.tcad.monitors.charge import (
    SteadyCapacitanceMonitor,
    SteadyEnergyBandMonitor,
    SteadyFreeCarrierMonitor,
    SteadyPotentialMonitor,
)
from tidy3d.components.tcad.monitors.heat import TemperatureMonitor
from tidy3d.components.tcad.source.coupled import HeatFromElectricSource
from tidy3d.components.tcad.source.heat import HeatSource, UniformHeatSource
from tidy3d.components.types import Union

MobilityModelType = Union[CaugheyThomasMobility, ConstantMobilityModel]
RecombinationModelType = Union[
    AugerRecombination, RadiativeRecombination, ShockleyReedHallRecombination
]
BandGapNarrowingModelType = Union[SlotboomBandGapNarrowing]

# types of monitors that are accepted by heat simulation
HeatChargeMonitorType = Union[
    TemperatureMonitor,
    SteadyPotentialMonitor,
    SteadyFreeCarrierMonitor,
    SteadyEnergyBandMonitor,
    SteadyCapacitanceMonitor,
]
HeatChargeSourceType = Union[HeatSource, HeatFromElectricSource, UniformHeatSource]
HeatChargeBCType = Union[
    TemperatureBC, HeatFluxBC, ConvectionBC, VoltageBC, CurrentBC, InsulatingBC
]
