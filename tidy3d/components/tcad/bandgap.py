import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import VOLT


# Band-gap narrowing models
class SlotboomNarrowingModel(Tidy3dBaseModel):
    """This class specifies the parameters for the Slotboom model for band-gap narrowing.

    Reference
    ---------
    'UNIFIED APPARENT BANDGAP NARROWING IN n- AND p-TYPE SILICON'
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
