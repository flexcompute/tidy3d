from __future__ import annotations

from pydantic import Field, PositiveFloat

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import ELECTRON_VOLT


class ConstantEnergyBandGap(Tidy3dBaseModel):
    """Constant Energy band gap"""

    eg: PositiveFloat = Field(
        title="Band Gap",
        description="Energy band gap",
        json_schema_extra={"units": ELECTRON_VOLT},
    )


class VarshniEnergyBandGap(Tidy3dBaseModel):
    """
    Models the temperature dependence of the energy band gap (Eg)
    using the Varshni formula.

    Notes
    -----
    The model implements the following formula:

    .. math::

        E_g(T) = E_g(0) - \\frac{\\alpha T^2}{T + \\beta}

    Example
    -------
    >>> # Parameters for Silicon (Si)
    >>> si_model = VarshniEnergyBandGap(
    ...     eg_0=1.17,
    ...     alpha=4.73e-4,
    ...     beta=636.0,
    ... )

    References
    ----------

        .. [1] Varshni, Y. P. (1967). Temperature dependence of the energy gap in semiconductors. Physica, 34(1), 149-154.

    """

    eg_0: PositiveFloat = Field(
        title="Band Gap at 0 K",
        description="Energy band gap at absolute zero (0 Kelvin).",
        json_schema_extra={"units": ELECTRON_VOLT},
    )

    alpha: PositiveFloat = Field(
        title="Varshni Alpha Coefficient",
        description="Empirical Varshni coefficient (α).",
        json_schema_extra={"units": "eV/K"},
    )

    beta: PositiveFloat = Field(
        title="Varshni Beta Coefficient",
        description="Empirical Varshni coefficient (β), related to the Debye temperature.",
        json_schema_extra={"units": "K"},
    )
