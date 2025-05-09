from typing import Union

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
