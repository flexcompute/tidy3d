from typing import Union

from .dispersionless import CustomIsotropicMedium, Medium, PECMedium
from .dispersive import (
    CustomDebye,
    CustomDrude,
    CustomLorentz,
    CustomPoleResidue,
    CustomSellmeier,
    Debye,
    Drude,
    Lorentz,
    PoleResidue,
    Sellmeier,
)
from .lossy_metal import LossyMetalMedium

IsotropicUniformMediumType = Union[
    Medium, LossyMetalMedium, PoleResidue, Sellmeier, Lorentz, Debye, Drude, PECMedium
]
IsotropicCustomMediumType = Union[
    CustomPoleResidue,
    CustomSellmeier,
    CustomLorentz,
    CustomDebye,
    CustomDrude,
]
IsotropicCustomMediumInternalType = Union[IsotropicCustomMediumType, CustomIsotropicMedium]
IsotropicMediumType = Union[IsotropicCustomMediumType, IsotropicUniformMediumType]
