"""Defines properties of the medium / materials"""
# ruff: noqa: I001

from __future__ import annotations

from .nonlinear import (
    KerrNonlinearity,
    NonlinearModel,
    NonlinearSpec,
    NonlinearSusceptibility,
    TwoPhotonAbsorption,
)
from .base import AbstractMedium, AbstractCustomMedium
from .dispersionless import Medium, PECMedium, CustomMedium, PEC
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
from .medium_2d import Medium2D, MediumType, MediumType3D, PEC2D
from .perturbation import PerturbationMedium, PerturbationPoleResidue, AbstractPerturbationMedium

AnisotropicMedium.update_forward_refs()

__all__ = [
    "PEC",
    "PEC2D",
    "AbstractMedium",
    "AbstractCustomMedium",
    "AbstractPerturbationMedium",
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
    "MediumType",
    "MediumType3D",
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
