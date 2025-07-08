from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import ELECTRON_VOLT


class ConstantEnergyBandGap(Tidy3dBaseModel):
    """Constant Energy band gap"""

    eg: pd.PositiveFloat = pd.Field(
        title="Band Gap",
        description="Energy band gap",
        units=ELECTRON_VOLT,
    )


class QuadraticEnergyBandGap(Tidy3dBaseModel):
    """
    Models the temperature dependence of the energy band gap (Eg) using a
    quadratic approximation.

    Notes
    -----
    The model uses the following formula:

        .. math::

            E_g(T) = E_g(300) + \\alpha T + \\beta T^2

    Example
    -------
    >>> model = QuadraticEnergyBandGap(
    ...     eg_300=1.12,
    ...     alpha=-4.73e-4,
    ...     beta=-2.0e-7,
    ... )
    """

    eg_300: pd.PositiveFloat = pd.Field(
        ...,
        title="Band Gap at 300 K",
        description="Energy band gap at a reference temperature of 300 K.",
        units=ELECTRON_VOLT,
    )

    alpha: float = pd.Field(
        ...,
        title="Linear Temperature Coefficient (alpha)",
        description="Linear coefficient for the temperature dependence of the band gap.",
        units="eV/K",
    )

    beta: float = pd.Field(
        ...,
        title="Quadratic Temperature Coefficient (beta)",
        description="Quadratic coefficient for the temperature dependence of the band gap.",
        units="eV/K²",
    )


class VarshniEnergyBandGap(Tidy3dBaseModel):
    """
    Models the temperature dependence of the energy band gap (Eg)
    using the Varshni formula.

    Notes
    -----
    The model implements the following formula:

        .. math::

            E_g(T) = E_g(0) - \\frac{\\alpha T^2}{T + \\beta}$

    Example
    -------
    >>> # Parameters for Silicon (Si)
    >>> si_model = VarshniBandGap(
    ...     eg_0=1.17,
    ...     alpha=4.73e-4,
    ...     beta=636.0,
    ... )

    References
    -------

        Varshni, Y. P. (1967). Temperature dependence of the energy gap in semiconductors. Physica, 34(1), 149-154.

    """

    eg_0: pd.PositiveFloat = pd.Field(
        ...,
        title="Band Gap at 0 K",
        description="Energy band gap at absolute zero (0 Kelvin).",
        units=ELECTRON_VOLT,
    )

    alpha: pd.PositiveFloat = pd.Field(
        ...,
        title="Varshni Alpha Coefficient",
        description="Empirical Varshni coefficient (α).",
        units="eV/K",
    )

    beta: pd.PositiveFloat = pd.Field(
        ...,
        title="Varshni Beta Coefficient",
        description="Empirical Varshni coefficient (β), related to the Debye temperature.",
        units="K",
    )
