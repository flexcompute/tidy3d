"""Defines properties of the medium / materials"""
# ruff: noqa: I001

from __future__ import annotations

from typing import Union

from .nonlinear import (
    KerrNonlinearity,
    NonlinearModel,
    NonlinearSpec,
    NonlinearSusceptibility,
    TwoPhotonAbsorption,
)
from .base import AbstractMedium
from .dispersionless import Medium, PECMedium, CustomMedium, CustomIsotropicMedium
from .anisotropic import (
    AnisotropicMedium,
    CustomAnisotropicMedium,
    FullyAnisotropicMedium,
    AnisotropicMediumFromMedium2D,
)
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
    medium_from_nk,
)
from .lossy_metal import (
    HammerstadSurfaceRoughness,
    HuraySurfaceRoughness,
    LossyMetalMedium,
    SurfaceImpedanceFitterParam,
)
from .medium_2d import Medium2D
from .perturbation import PerturbationMedium, PerturbationPoleResidue


NonlinearModelType = Union[NonlinearSusceptibility, TwoPhotonAbsorption, KerrNonlinearity]
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
MediumType3D = Union[
    Medium,
    AnisotropicMedium,
    PECMedium,
    PoleResidue,
    Sellmeier,
    Lorentz,
    Debye,
    Drude,
    FullyAnisotropicMedium,
    CustomMedium,
    CustomPoleResidue,
    CustomSellmeier,
    CustomLorentz,
    CustomDebye,
    CustomDrude,
    CustomAnisotropicMedium,
    PerturbationMedium,
    PerturbationPoleResidue,
    LossyMetalMedium,
]
MediumType = Union[MediumType3D, Medium2D, AnisotropicMediumFromMedium2D]

PEC = PECMedium(name="PEC")
PEC2D = Medium2D(ss=PEC, tt=PEC)


__all__ = [
    "PEC",
    "PEC2D",
    "AbstractMedium",
    "AnisotropicMedium",
    "AnisotropicMediumFromMedium2D",
    "CustomAnisotropicMedium",
    "CustomDebye",
    "CustomDrude",
    "CustomLorentz",
    "CustomMedium",
    "CustomPoleResidue",
    "CustomSellmeier",
    "Debye",
    "Drude",
    "FullyAnisotropicMedium",
    "HammerstadSurfaceRoughness",
    "HuraySurfaceRoughness",
    "KerrNonlinearity",
    "Lorentz",
    "LossyMetalMedium",
    "Medium",
    "Medium2D",
    "NonlinearModel",
    "NonlinearSpec",
    "NonlinearSusceptibility",
    "PECMedium",
    "PerturbationMedium",
    "PerturbationPoleResidue",
    "PoleResidue",
    "Sellmeier",
    "SurfaceImpedanceFitterParam",
    "TwoPhotonAbsorption",
    "medium_from_nk",
]
