"""
Note in the future we might want to implement interpolation models here.
"""

from typing import Union

from tidy3d.components.mediums.tcad.charge import (
    ChargeConductorMedium,
    ChargeInsulatorMedium,
    SemiconductorMedium,
)
from tidy3d.components.mediums.tcad.heat import FluidSpec, SolidSpec

OpticalMediumTypes = None
ElectricalMediumTypes = None
HeatMediumTypes = ThermalSpecType = Union[FluidSpec, SolidSpec]
ChargeMediumTypes = Union[ChargeConductorMedium, ChargeInsulatorMedium, SemiconductorMedium]
