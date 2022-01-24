"""Defines profile of Perfectly-matched layers (absorber)"""
from typing import Union
from abc import ABC

import pydantic

from .base import Tidy3dBaseModel
from ..constants import PML_SIGMA

# TODO: More explanation on parameters, when to use various PMLs.


class AbsorberParams(Tidy3dBaseModel):
    """Specifies parameters common to Absorbers and PMLs.

    Example
    -------
    >>> params = AbsorberParams(sigma_order=3, sigma_min=0.0, sigma_max=1.5)
    """

    sigma_order: pydantic.NonNegativeInt = pydantic.Field(
        3,
        title="Sigma Order",
        description="Order of the polynomial describing the absorber profile (~dist^sigma_order).",
    )

    sigma_min: pydantic.NonNegativeFloat = pydantic.Field(
        0.0,
        title="Sigma Minimum",
        description="Minimum value of the absorber conductivity.",
        units=PML_SIGMA,
    )

    sigma_max: pydantic.NonNegativeFloat = pydantic.Field(
        1.5,
        title="Sigma Maximum",
        description="Maximum value of the absorber conductivity.",
        units=PML_SIGMA,
    )


class PMLParams(AbsorberParams):
    """Specifies full set of parameters needed for complex, frequency-shifted PML.

    Example
    -------
    >>> params = PMLParams(sigma_order=3, sigma_min=0.0, sigma_max=1.5, kappa_min=0.0)
    """

    kappa_order: pydantic.NonNegativeInt = pydantic.Field(
        3,
        title="Kappa Order",
        description="Order of the polynomial describing the PML kappa profile "
        "(kappa~dist^kappa_order).",
    )

    kappa_min: pydantic.NonNegativeFloat = pydantic.Field(
        0.0, title="Kappa Minimum", description=""
    )

    kappa_max: pydantic.NonNegativeFloat = pydantic.Field(
        1.5, title="Kappa Maximum", description=""
    )

    alpha_order: pydantic.NonNegativeInt = pydantic.Field(
        3,
        title="Alpha Order",
        description="Order of the polynomial describing the PML alpha profile "
        "(alpha~dist^alpha_order).",
    )

    alpha_min: pydantic.NonNegativeFloat = pydantic.Field(
        0.0, title="Alpha Minimum", description="Minimum value of the PML alpha.", units=PML_SIGMA
    )

    alpha_max: pydantic.NonNegativeFloat = pydantic.Field(
        1.5, title="Alpha Maximum", description="Maximum value of the PML alpha.", units=PML_SIGMA
    )


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
    """Specifies the generic absorber properties along a single dimension."""

    num_layers: pydantic.NonNegativeInt = pydantic.Field(
        ...,
        title="Number of Layers",
        description="Number of layers of standard PML to add to + and - boundaries.",
    )
    parameters: AbsorberParams = pydantic.Field(
        ...,
        title="Absorber Parameters",
        description="Parameters to fine tune the absorber profile and properties.",
    )


class PML(AbsorberSpec):
    """Specifies a standard PML along a single dimension.

    Example
    -------
    >>> pml = PML(num_layers=10)
    """

    num_layers: pydantic.NonNegativeInt = pydantic.Field(
        12,
        title="Number of Layers",
        description="Number of layers of standard PML to add to + and - boundaries.",
    )

    parameters: PMLParams = pydantic.Field(
        DefaultPMLParameters,
        title="PML Parameters",
        description="Parameters of the complex frequency-shifted absorption poles.",
    )


class StablePML(AbsorberSpec):
    """Specifies a 'stable' PML along a single dimension.
    This PML deals handles possbly divergent simulations better, but at the expense of more layers.

    Example
    -------
    >>> pml = StablePML(num_layers=40)
    """

    num_layers: pydantic.NonNegativeInt = pydantic.Field(
        40, title="Number of Layers", description="Number of layers of 'stable' PML."
    )

    parameters: PMLParams = pydantic.Field(
        DefaultStablePMLParameters,
        title="Stable PML Parameters",
        description="'Stable' parameters of the complex frequency-shifted absorption poles.",
    )


class Absorber(AbsorberSpec):
    """Specifies an adiabatic absorber along a single dimension.
    This absorber is well-suited for dispersive materials
    intersecting with absorbing edges of the simulation at the expense of more layers.

    Example
    -------
    >>> pml = Absorber(num_layers=40)
    """

    num_layers: pydantic.NonNegativeInt = pydantic.Field(
        40,
        title="Number of Layers",
        description="Number of layers of absorber to add to + and - boundaries.",
    )

    parameters: AbsorberParams = pydantic.Field(
        DefaultAbsorberParameters,
        title="Absorber Parameters",
        description="Adiabatic absorber parameters.",
    )


# pml types allowed in simulation init
PMLTypes = Union[PML, StablePML, Absorber, None]
