"""File containing classes required for the setup of a DEVSIM case."""

from tidy3d.components.tcad.bandgap import SlotboomNarrowingBandGap
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
