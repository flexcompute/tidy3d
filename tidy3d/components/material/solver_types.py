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
from tidy3d.components.medium import MediumType, MediumType3D

OpticalMediumType = MediumType
ElectricalMediumType = MediumType
HeatMediumType = ThermalSpecType
ChargeMediumType = Union[ChargeConductorMedium, ChargeInsulatorMedium, SemiconductorMedium]

OpticalMediumType3D = ElectricalMediumType3D = ChargeMediumType3D = MediumType3D
