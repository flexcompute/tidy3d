import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import PERCMCUBE, VOLT


# Band-gap narrowing models
class SlotboomBandGapNarrowing(Tidy3dBaseModel):
    """
    Parameters for the Slotboom model for band-gap narrowing.

    Notes
    ------
        The Slotboom band-gap narrowing :math:`\\Delta E_G` model is discussed in [1]_ as follows:

        .. math::

            \\Delta E_G = V_{1,bgn} \\left( \\ln \\left( \\frac{N_{tot}}{N_{2,bgn}} \\right)
            + \\sqrt{\\left( \\ln \\left( \\frac{N_{tot}}{N_{2,bgn}} \\right) \\right)^2 + C_{2,bgn}} \\right)
            \\quad \\text{if} \\quad N_{tot} \\geq 10^{15} \\text{cm}^{-3},

            \\Delta E_G = 0 \\quad \\text{if} \\quad N_{tot} < 10^{15} \\text{cm}^{-3}.

        Note that :math:`N_{tot}` is the total doping as defined within a :class:`SemiconductorMedium`.

        Example
        -------
            >>> import tidy3d as td
            >>> default_Si = td.SlotboomBandGapNarrowing(
            ...    v1=6.92 * 1e-3,
            ...    n2=1.3e17,
            ...    c2=0.5,
            ...    min_N=1e15,
            ... )

        .. [1] 'UNIFIED APPARENT BANDGAP NARROWING IN n- AND p-TYPE SILICON' Solid-State Electronics Vol. 35, No. 2, pp. 125-129, 1992"""

    v1: pd.PositiveFloat = pd.Field(
        ...,
        title=r"$V_{1,bgn}$ parameter",
        description=r"$V_{1,bgn}$ parameter",
        units=VOLT,
    )

    n2: pd.PositiveFloat = pd.Field(
        ...,
        title=r"$N_{2,bgn}$ parameter",
        description=r"$N_{2,bgn}$ parameter",
        units=PERCMCUBE,
    )

    c2: float = pd.Field(
        title=r"$C_{2,bgn}$ parameter",
        description=r"$C_{2,bgn}$ parameter",
    )

    min_N: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Minimum total doping",
        description="Bandgap narrowing is applied at location where total doping "
        "is higher than 'min_N'.",
        units=PERCMCUBE,
    )
