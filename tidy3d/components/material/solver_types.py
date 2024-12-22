"""
Note in the future we might want to implement interpolation models here.
"""

from typing import Union

from tidy3d.components.material.tcad.charge import (
    ChargeConductorMedium,
    ChargeInsulatorMedium,
    SemiconductorMedium,
)
from tidy3d.components.material.tcad.heat import ThermalSpecType

OpticalMediumTypes = None
ElectricalMediumTypes = None
HeatMediumTypes = ThermalSpecType
ChargeMediumTypes = Union[ChargeConductorMedium, ChargeInsulatorMedium, SemiconductorMedium]
