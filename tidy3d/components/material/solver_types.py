"""
Note in the future we might want to implement interpolation models here.
"""

from __future__ import annotations

from tidy3d.components.material.tcad.charge import (
    ChargeConductorMedium,
    ChargeInsulatorMedium,
    SemiconductorMedium,
)
from tidy3d.components.material.tcad.heat import ThermalSpecType
from tidy3d.components.medium import (
    ProtectedOpticalMediumPlacementType,
    ProtectedOpticalMediumPlacementType3D,
    PublicMediumType,
    PublicMediumType3D,
)

OpticalMediumType = ProtectedOpticalMediumPlacementType
ElectricalMediumType = PublicMediumType
HeatMediumType = ThermalSpecType
ChargeMediumType = ChargeConductorMedium | ChargeInsulatorMedium | SemiconductorMedium

OpticalMediumType3D = ProtectedOpticalMediumPlacementType3D
ElectricalMediumType3D = ChargeMediumType3D = PublicMediumType3D
