from __future__ import annotations

from .multi_physics import MultiPhysicsMedium
from .solver_types import (
    ChargeMediumType,
    ChargeMediumType3D,
    ElectricalMediumType,
    ElectricalMediumType3D,
    HeatMediumType,
    OpticalMediumType,
    OpticalMediumType3D,
)

StructureMediumType = (
    MultiPhysicsMedium
    | OpticalMediumType
    | ElectricalMediumType
    | HeatMediumType
    | ChargeMediumType
)

MultiPhysicsMediumType3D = (
    MultiPhysicsMedium | OpticalMediumType3D | ElectricalMediumType3D | ChargeMediumType3D
)
