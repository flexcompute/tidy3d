""" Defines profile of Perfectly-matched layers (absorber) """
from typing import Union, Literal

import pydantic

from .base import Tidy3dBaseModel


"""TODO: better docstrings."""


class AbsorberParams(Tidy3dBaseModel):
    """Specifies parameters for an Absorber or PML. Sigma is in units of 2*EPSILON_0/dt."""

    sigma_order: pydantic.NonNegativeInt
    sigma_min: pydantic.NonNegativeFloat
    sigma_max: pydantic.NonNegativeFloat


class PMLParams(AbsorberParams):
    """Extra parameters needed for complex frequency-shifted PML. Kappa is dimensionless, alpha
    is in the same units as sigma."""

    kappa_order: pydantic.NonNegativeInt
    kappa_min: pydantic.NonNegativeFloat
    kappa_max: pydantic.NonNegativeFloat
    alpha_order: pydantic.NonNegativeInt
    alpha_min: pydantic.NonNegativeFloat
    alpha_max: pydantic.NonNegativeFloat


AbsorberPs = AbsorberParams(sigma_order=3, sigma_min=0.0, sigma_max=6.4)
StandardPs = PMLParams(
    sigma_order=3,
    sigma_min=0.0,
    sigma_max=1.5,
    kappa_order=3,
    kappa_min=1.0,
    kappa_max=3.0,
    alpha_order=1,
    alpha_min=0.0,
    alpha_max=0.0,
)
StablePs = PMLParams(
    sigma_order=3,
    sigma_min=0.0,
    sigma_max=1.0,
    kappa_order=3,
    kappa_min=1.0,
    kappa_max=5.0,
    alpha_order=1,
    alpha_min=0.0,
    alpha_max=0.9,
)


class AbsorberSpec(Tidy3dBaseModel):
    """Specifies the absorber along a single dimension."""

    num_layers: pydantic.NonNegativeInt


class PML(AbsorberSpec):
    """Specifies a standard PML along a single dimension.

    Parameters
    ----------
    num_layers : ``int``, optional
        Number of layers of PML to add to + and - boundaries, default = 12.
    pml_params : :class:PMLParams
        Parameters of the complex frequency-shifted absorption poles.

    Example
    -------
    >>> pml = PML(num_layers=10)
    """

    num_layers: pydantic.NonNegativeInt = 12
    parameters: PMLParams = StandardPs


class StablePML(AbsorberSpec):
    """Specifies a 'stable' PML along a single dimension.

    Parameters
    ----------
    num_layers : ``int``, optional
        Number of layers of PML to add to + and - boundaries, default = 40.

    Example
    -------
    >>> pml = StablePML(num_layers=100)
    """

    num_layers: pydantic.NonNegativeInt = 40
    parameters: Literal[StablePs] = StablePs


class Absorber(AbsorberSpec):
    """Specifies an adiab absorber along a single dimension.

    Parameters
    ----------
    num_layers : ``int``, optional
        Number of layers of PML to add to + and - boundaries, default = 40.

    Example
    -------
    >>> pml = Absorber(num_layers=100)
    """

    num_layers: pydantic.NonNegativeInt = 40
    parameters: AbsorberParams = AbsorberPs


PMLTypes = Union[PML, StablePML, Absorber, None]
