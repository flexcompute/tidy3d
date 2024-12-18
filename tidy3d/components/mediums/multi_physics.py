from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.mediums.solver_types import (
    ChargeMediumTypes,
    ElectricalMediumTypes,
    HeatMediumTypes,
    OpticalMediumTypes,
)


class MultiPhysicsMedium(Tidy3dBaseModel):
    """
    A multi-physics medium may contain multiple multi-physical properties, defined for each solver medium.
    """

    optical: OpticalMediumTypes
    electrical: ElectricalMediumTypes
    heat: HeatMediumTypes
    charge: ChargeMediumTypes
