"""Defines profile of Perfectly-matched layers (absorber) 

Attributes
----------
AbsorberPs : TYPE
    Description
PMLTypes : TYPE
    Description
StablePs : TYPE
    Description
StandardPs : TYPE
    Description
"""
from typing import Union, Literal
from abc import ABC

import pydantic

from .base import Tidy3dBaseModel


"""TODO: better docstrings."""


class AbsorberParams(Tidy3dBaseModel):
    """Specifies parameters common to Absorbers and PMLs.

    Parameters
    ----------
    sigma_order : int = 3
        Order of the polynomial describing the absorber profile (~dist^sigma_order).
        Must be non-negative.
    sigma_min : float = 0.0
        Minimum value of the absorber conductivity.
        Units of 2*EPSILON_0/dt.
        Must be non non-negative.        
    sigma_max : float = 1.5
        Maximum value of the absorber conductivity.
        Units of 2*EPSILON_0/dt.
        Must be non non-negative.

    Example
    -------
    >>> params = AbsorberParams(sigma_order=3, sigma_min=0.0, sigma_max=1.5)
    """

    sigma_order: pydantic.NonNegativeInt
    sigma_min: pydantic.NonNegativeFloat
    sigma_max: pydantic.NonNegativeFloat


class PMLParams(AbsorberParams):
    """Specifies full set of parameters needed for complex, frequency-shifted PML.

    Parameters
    ----------
    sigma_order : int = 3
        Order of the polynomial describing the absorber profile (sigma~dist^sigma_order).
        Must be non-negative.
    sigma_min : float = 0.0
        Minimum value of the absorber conductivity.
        Units of 2*EPSILON_0/dt.
        Must be non-negative.        
    sigma_max : float = 1.5
        Maximum value of the absorber conductivity.
        Units of 2*EPSILON_0/dt.
        Must be non-negative.
    kappa_order : int = 3
        Order of the polynomial describing the PML kappa profile (kappa~dist^kappa_order).
        Must be non-negative.
    kappa_min : float = 0.0
        Minimum value of the PML kappa.
        Dimensionless.
        Must be non-negative.        
    kappa_max : float = 1.5
        Maximum value of the PML kappa.
        Dimensionless.
        Must be non-negative.
    alpha_order : int = 3
        Order of the polynomial describing the PML alpha profile (alpha~dist^alpha_order).
        Must be non-negative.
    alpha_min : float = 0.0
        Minimum value of the PML alpha.
        Units of 2*EPSILON_0/dt.
        Must be non-negative.        
    alpha_max : float = 1.5
        Maximum value of the PML alpha.
        Units of 2*EPSILON_0/dt.
        Must be non-negative.

    Example
    -------
    >>> params = PMLParams(sigma_order=3, sigma_min=0.0, sigma_max=1.5, kappa_min=0.0)
    """

    kappa_order: pydantic.NonNegativeInt
    kappa_min: pydantic.NonNegativeFloat
    kappa_max: pydantic.NonNegativeFloat
    alpha_order: pydantic.NonNegativeInt
    alpha_min: pydantic.NonNegativeFloat
    alpha_max: pydantic.NonNegativeFloat

""" Default parameters """

DefaultAbsorberParameters = AbsorberParams(sigma_order=3, sigma_min=0.0, sigma_max=6.4)
DefaultPMLParameters = PMLParams(
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
DefaultStablePMLParameters = PMLParams(
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

""" PML specifications """

class AbsorberSpec(Tidy3dBaseModel, ABC):
    """Abstract base class.
       Specifies the generic absorber properties along a single dimension.

    Parameters
    ----------
    num_layers : int = 12
        Number of layers of standard PML to add to + and - boundaries.
        Must be non-negative.
    parameters : :class:`AbsorberParams`
        Parameters to fine tune the absorber profile and properties.
    """

    num_layers: pydantic.NonNegativeInt
    parameters: AbsorberParams


class PML(AbsorberSpec):
    """Specifies a standard PML along a single dimension.
    
    Parameters
    ----------
    num_layers : int = 12
        Number of layers of standard PML to add to + and - boundaries.
        Must be non-negative.
    parameters : :class:`PMLParams` = DefaultPMLParameters
        Parameters of the complex frequency-shifted absorption poles.
    
    Example
    -------
    >>> pml = PML(num_layers=10)
    """

    num_layers: pydantic.NonNegativeInt = 12
    parameters: PMLParams = DefaultPMLParameters


class StablePML(AbsorberSpec):
    """Specifies a 'stable' PML along a single dimension.
    This PML deals handles possbly divergent simulations better, but at the expense of more layers.
    
    Parameters
    ----------
    num_layers : int = 40
        Number of layers of stable PML to add to + and - boundaries.
        Must be non-negative.
    parameters : Literal[DefaultStablePMLParameters] = DefaultStablePMLParameters
        "Stable" parameters of the complex frequency-shifted absorption poles.

    Example
    -------
    >>> pml = StablePML(num_layers=40)
    """

    num_layers: pydantic.NonNegativeInt = 40
    parameters: Literal[DefaultStablePMLParameters] = DefaultStablePMLParameters


class Absorber(AbsorberSpec):
    """Specifies an adiabatic absorber along a single dimension.
    This absorber is well-suited for dispersive materials
    intersecting with absorbing edges of the simulation at the expense of more layers.
    
    Parameters
    ----------
    num_layers : int = 40
        Number of layers of absorber to add to + and - boundaries.
    parameters : :class:`AbsorberParams` = DefaultAbsorberParameters
        General absorber parameters.

    Example
    -------
    >>> pml = Absorber(num_layers=40)
    """

    num_layers: pydantic.NonNegativeInt = 40
    parameters: AbsorberParams = DefaultAbsorberParameters


# pml types allowed in simulation init
PMLTypes = Union[PML, StablePML, Absorber, None]
