"""Electromagnetic medium models and compatibility exports.

The package separates abstract foundations, uniform and dispersive model families,
spatially varying custom media, perturbation models, and public type aliases. Thermal,
charge, and multiphysics material helpers remain in :mod:`tidy3d.components.material`.
"""

from __future__ import annotations

# Related nonlinear models historically exported from this namespace.
from tidy3d.components.nonlinear import (
    KerrNonlinearity,
    NonlinearModel,
    NonlinearSpec,
    NonlinearSpecType,
    NonlinearSusceptibility,
    TwoPhotonAbsorption,
)

# isort: split

# Abstract bases and shared helpers.
from .abstract_custom import AbstractCustomMedium
from .base import (
    ALLOWED_INTERP_METHODS,
    FILL_VALUE,
    FREQ_EVAL_INF,
    AbstractMedium,
    ArrayComplex,
    ArrayFloat,
    ArrayGeneric,
    ComplexArrayOrScalar,
    FrequencyArray,
    WeightFunction,
    ensure_freq_in_range,
)

# isort: split

# Uniform, anisotropic, and dispersive model families.
from .anisotropic import (
    AnisotropicMedium,
    AnisotropicMediumFromMedium2D,
    FullyAnisotropicMedium,
)
from .debye import Debye
from .drude import Drude
from .isotropic import PEC, PMC, Medium, PECMedium, PMCMedium, medium_from_nk
from .lorentz import Lorentz
from .lossy_metal import LOSSY_METAL_SCALED_REAL_PART, LossyMetalMedium
from .pole_residue import DispersiveMedium, PoleResidue
from .roughness import (
    LOSSY_METAL_DEFAULT_MAX_POLES,
    LOSSY_METAL_DEFAULT_SAMPLING_FREQUENCY,
    LOSSY_METAL_DEFAULT_TOLERANCE_RMS,
    AbstractSurfaceRoughness,
    HammerstadSurfaceRoughness,
    HuraySurfaceRoughness,
    SurfaceImpedanceFitterParam,
    SurfaceRoughnessType,
)
from .sellmeier import Sellmeier
from .two_d import PEC2D, Medium2D

# isort: split

# Spatially varying custom model families.
from .custom import (
    CustomAnisotropicMedium,
    CustomAnisotropicMediumInternal,
    CustomDebye,
    CustomDispersiveMedium,
    CustomDrude,
    CustomIsotropicMedium,
    CustomLorentz,
    CustomMedium,
    CustomPoleResidue,
    CustomSellmeier,
)

# isort: split

# Temperature and carrier-density perturbation models.
from .perturbation import (
    AbstractPerturbationMedium,
    PerturbationMedium,
    PerturbationMediumType,
    PerturbationPoleResidue,
)

# isort: split

# Public medium unions.
from .medium_types import (
    IsotropicCustomMediumInternalType,
    IsotropicCustomMediumType,
    IsotropicMediumType,
    IsotropicUniformMediumFor2DType,
    IsotropicUniformMediumType,
    MediumType,
    MediumType3D,
)

# isort: split

# Resolve forward references only after every model family and type alias exists.
from ._rebuild import rebuild_medium_models as _rebuild_medium_models

_rebuild_medium_models()
del _rebuild_medium_models

__all__ = [
    "ALLOWED_INTERP_METHODS",
    "FILL_VALUE",
    "FREQ_EVAL_INF",
    "LOSSY_METAL_DEFAULT_MAX_POLES",
    "LOSSY_METAL_DEFAULT_SAMPLING_FREQUENCY",
    "LOSSY_METAL_DEFAULT_TOLERANCE_RMS",
    "LOSSY_METAL_SCALED_REAL_PART",
    "PEC",
    "PEC2D",
    "PMC",
    "AbstractCustomMedium",
    "AbstractMedium",
    "AbstractPerturbationMedium",
    "AbstractSurfaceRoughness",
    "AnisotropicMedium",
    "AnisotropicMediumFromMedium2D",
    "ArrayComplex",
    "ArrayFloat",
    "ArrayGeneric",
    "ComplexArrayOrScalar",
    "CustomAnisotropicMedium",
    "CustomAnisotropicMediumInternal",
    "CustomDebye",
    "CustomDispersiveMedium",
    "CustomDrude",
    "CustomIsotropicMedium",
    "CustomLorentz",
    "CustomMedium",
    "CustomPoleResidue",
    "CustomSellmeier",
    "Debye",
    "DispersiveMedium",
    "Drude",
    "FrequencyArray",
    "FullyAnisotropicMedium",
    "HammerstadSurfaceRoughness",
    "HuraySurfaceRoughness",
    "IsotropicCustomMediumInternalType",
    "IsotropicCustomMediumType",
    "IsotropicMediumType",
    "IsotropicUniformMediumFor2DType",
    "IsotropicUniformMediumType",
    "KerrNonlinearity",
    "Lorentz",
    "LossyMetalMedium",
    "Medium",
    "Medium2D",
    "MediumType",
    "MediumType3D",
    "NonlinearModel",
    "NonlinearSpec",
    "NonlinearSpecType",
    "NonlinearSusceptibility",
    "PECMedium",
    "PMCMedium",
    "PerturbationMedium",
    "PerturbationMediumType",
    "PerturbationPoleResidue",
    "PoleResidue",
    "Sellmeier",
    "SurfaceImpedanceFitterParam",
    "SurfaceRoughnessType",
    "TwoPhotonAbsorption",
    "WeightFunction",
    "ensure_freq_in_range",
    "medium_from_nk",
]
