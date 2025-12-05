"""HeatCharge solver associated types."""

from __future__ import annotations

from typing import Union

from tidy3d.components.tcad.bandgap import SlotboomBandGapNarrowing
from tidy3d.components.tcad.bandgap_energy import (
    ConstantEnergyBandGap,
    VarshniEnergyBandGap,
)
from tidy3d.components.tcad.boundary.charge import CurrentBC, InsulatingBC, VoltageBC
from tidy3d.components.tcad.boundary.heat import (
    ConvectionBC,
    HeatFluxBC,
    TemperatureBC,
)
from tidy3d.components.tcad.effective_DOS import (
    ConstantEffectiveDOS,
    DualValleyEffectiveDOS,
    IsotropicEffectiveDOS,
    MultiValleyEffectiveDOS,
)
from tidy3d.components.tcad.generation_recombination import (
    AugerRecombination,
    DistributedGeneration,
    HurkxDirectBandToBandTunneling,
    RadiativeRecombination,
    SelberherrImpactIonization,
    ShockleyReedHallRecombination,
)
from tidy3d.components.tcad.mobility import CaugheyThomasMobility, ConstantMobilityModel
from tidy3d.components.tcad.monitors.charge import (
    SteadyCapacitanceMonitor,
    SteadyCurrentDensityMonitor,
    SteadyElectricFieldMonitor,
    SteadyEnergyBandMonitor,
    SteadyFreeCarrierMonitor,
    SteadyPotentialMonitor,
)
from tidy3d.components.tcad.monitors.heat import TemperatureMonitor
from tidy3d.components.tcad.source.coupled import HeatFromElectricSource
from tidy3d.components.tcad.source.heat import HeatSource, UniformHeatSource

EffectiveDOSModelType = Union[
    ConstantEffectiveDOS, IsotropicEffectiveDOS, MultiValleyEffectiveDOS, DualValleyEffectiveDOS
]
EnergyBandGapModelType = Union[ConstantEnergyBandGap, VarshniEnergyBandGap]
MobilityModelType = Union[CaugheyThomasMobility, ConstantMobilityModel]
RecombinationModelType = Union[
    AugerRecombination,
    DistributedGeneration,
    RadiativeRecombination,
    ShockleyReedHallRecombination,
    HurkxDirectBandToBandTunneling,
    SelberherrImpactIonization,
]
BandGapNarrowingModelType = Union[SlotboomBandGapNarrowing]

# types of monitors that are accepted by heat simulation
HeatChargeMonitorType = Union[
    TemperatureMonitor,
    SteadyPotentialMonitor,
    SteadyFreeCarrierMonitor,
    SteadyEnergyBandMonitor,
    SteadyElectricFieldMonitor,
    SteadyCapacitanceMonitor,
    SteadyCurrentDensityMonitor,
]
HeatChargeSourceType = Union[HeatSource, HeatFromElectricSource, UniformHeatSource]
HeatChargeBCType = Union[
    TemperatureBC, HeatFluxBC, ConvectionBC, VoltageBC, CurrentBC, InsulatingBC
]
