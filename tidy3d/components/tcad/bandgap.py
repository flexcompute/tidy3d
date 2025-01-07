import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import VOLT


# Band-gap narrowing models
class SlotboomBandGapNarrowing(Tidy3dBaseModel):
    """
    This class specifies the parameters for the Slotboom model for band-gap narrowing.

    Notes
    ------
        The Slotboom band-gap narrowing model :math:`\\Delta E_G` is discussed in [1]_ as follows:

        .. math::

            \\Delta E_G = V_{1,bgn} \\left( \\ln \\left( \\frac{N_{tot}}{N_{2,bgn}} \\right)
            + \\sqrt{\\left( \\ln \\left( \\frac{N_{tot}}{N_{2,bgn}} \\right) \\right)^2 + C_{2,bgn}} \\right)
            \\quad \\text{if} \\quad N_{tot} \\geq 10^{15} \\text{cm}^{-3},

            \\Delta E_G = 0 \\quad \\text{if} \\quad N_{tot} < 10^{15} \\text{cm}^{-3}.

        Note that :math:`N_{tot}` is the total doping as defined within a :class:`SemiconductorMedium`.

        TODO define are the other parameters

        .. [1] 'UNIFIED APPARENT BANDGAP NARROWING IN n- AND p-TYPE SILICON'
                Solid-State Electronics Vol. 35, No. 2, pp. 125-129, 1992"""

    v1: pd.PositiveFloat = pd.Field(
        6.92 * 1e-3, title="V1 parameter", description=f"V1 parameter in {VOLT}", units=VOLT
    )

    n2: pd.PositiveFloat = pd.Field(
        1.3e17,
        title="n2 parameter",
        description="n2 parameter in cm^(-3)",
    )

    c2: float = pd.Field(
        0.5,
        title="c2 parameter",
        description="c2 parameter",
    )
