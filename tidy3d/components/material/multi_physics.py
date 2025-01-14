from typing import Optional

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.material.solver_types import (
    ChargeMediumType,
    ElectricalMediumType,
    HeatMediumType,
    OpticalMediumType,
)


class MultiPhysicsMedium(Tidy3dBaseModel):
    """
    A multi-physics medium may contain multiple multi-physical properties, defined for each solver medium.

    Examples
    --------
    >>> import tidy3d as td
    >>> SiO2 = td.MultiPhysicsMedium(
    ... optical=td.Medium(permittivity=3.9),
    ...  charge=td.ChargeInsulatorMedium(permittivity=3.9), # redefining permittivity
    ... name="SiO2",
    ... )
    """

    # TODO requires backwards compatibility.
    name: Optional[str] = None
    optical: Optional[OpticalMediumType] = None
    electrical: Optional[ElectricalMediumType] = None
    heat: Optional[HeatMediumType] = None
    charge: Optional[ChargeMediumType] = None

    @property
    def heat_spec(self):
        if self.heat is not None:
            return self.heat
        elif self.optical is not None:
            return self.optical.heat_spec
        else:
            return None
