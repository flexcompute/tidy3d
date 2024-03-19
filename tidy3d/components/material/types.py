from typing import Union

from .multi_physics import MultiPhysicsMedium
from .solver_types import (
    ChargeMediumTypes,
    ChargeMediumTypes3D,
    ElectricalMediumTypes,
    ElectricalMediumTypes3D,
    HeatMediumTypes,
    OpticalMediumTypes,
    OpticalMediumTypes3D,
)

StructureMediumTypes = Union[
    MultiPhysicsMedium,
    OpticalMediumTypes,
    ElectricalMediumTypes,
    HeatMediumTypes,
    ChargeMediumTypes,
]

MultiPhysicsMediumTypes3D = Union[
    MultiPhysicsMedium, OpticalMediumTypes3D, ElectricalMediumTypes3D, ChargeMediumTypes3D
]
