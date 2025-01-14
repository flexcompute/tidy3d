from typing import Union

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

StructureMediumType = Union[
    MultiPhysicsMedium,
    OpticalMediumType,
    ElectricalMediumType,
    HeatMediumType,
    ChargeMediumType,
]

MultiPhysicsMediumType3D = Union[
    MultiPhysicsMedium, OpticalMediumType3D, ElectricalMediumType3D, ChargeMediumType3D
]
