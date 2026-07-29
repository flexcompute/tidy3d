"""Public type aliases for electromagnetic medium models."""

from __future__ import annotations

from .anisotropic import (
    AnisotropicMedium,
    AnisotropicMediumFromMedium2D,
    FullyAnisotropicMedium,
)
from .custom import (
    CustomAnisotropicMedium,
    CustomDebye,
    CustomDrude,
    CustomIsotropicMedium,
    CustomLorentz,
    CustomMedium,
    CustomPoleResidue,
    CustomSellmeier,
)
from .debye import Debye
from .drude import Drude
from .isotropic import Medium, PECMedium, PMCMedium
from .lorentz import Lorentz
from .lossy_metal import LossyMetalMedium
from .perturbation import PerturbationMedium, PerturbationPoleResidue
from .pole_residue import PoleResidue
from .sellmeier import Sellmeier
from .two_d import Medium2D

IsotropicUniformMediumFor2DType = (
    Medium | LossyMetalMedium | PoleResidue | Sellmeier | Lorentz | Debye | Drude | PECMedium
)
IsotropicUniformMediumType = IsotropicUniformMediumFor2DType | PMCMedium

IsotropicCustomMediumType = (
    CustomPoleResidue | CustomSellmeier | CustomLorentz | CustomDebye | CustomDrude
)
IsotropicCustomMediumInternalType = IsotropicCustomMediumType | CustomIsotropicMedium
IsotropicMediumType = IsotropicCustomMediumType | IsotropicUniformMediumType

MediumType3D = (
    Medium
    | AnisotropicMedium
    | PECMedium
    | PMCMedium
    | PoleResidue
    | Sellmeier
    | Lorentz
    | Debye
    | Drude
    | FullyAnisotropicMedium
    | CustomMedium
    | CustomPoleResidue
    | CustomSellmeier
    | CustomLorentz
    | CustomDebye
    | CustomDrude
    | CustomAnisotropicMedium
    | PerturbationMedium
    | PerturbationPoleResidue
    | LossyMetalMedium
)

MediumType = MediumType3D | Medium2D | AnisotropicMediumFromMedium2D

__all__ = [
    "IsotropicCustomMediumInternalType",
    "IsotropicCustomMediumType",
    "IsotropicMediumType",
    "IsotropicUniformMediumFor2DType",
    "IsotropicUniformMediumType",
    "MediumType",
    "MediumType3D",
]
