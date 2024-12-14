import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel


# Generation-Recombination models
class AugerRecombination(Tidy3dBaseModel):
    """This class defines the parameters for the Auger recombination model.
    NOTE: default parameters are those appropriate for Silicon."""

    c_n: pd.PositiveFloat = pd.Field(
        2.8e-31, title="Constant for electrons", description="Constant for electrons in cm^6/s"
    )

    c_p: pd.PositiveFloat = pd.Field(
        9.9e-32, title="Constant for holes", description="Constant for holes in cm^6/s"
    )


class RadiativeRecombination(Tidy3dBaseModel):
    """This class is used to define the parameters for the radiative recombination model.
    NOTE: default values are those appropriate for Silicon."""

    r_const: float = pd.Field(
        1.6e-14,
        title="Radiation constant in cm^3/s",
        description="Radiation constant in cm^3/s",
    )


class ShockleyReedHallRecombination(Tidy3dBaseModel):
    """This class defines the parameters for the Shockley-Reed-Hall recombination model.
    NOTE: currently, lifetimes are considered constant (not dependent on temp. nor doping)
    NOTE: default values are those appropriate for Silicon."""

    tau_n: pd.PositiveFloat = pd.Field(
        3.3e-6, title="Electron lifetime.", description="Electron lifetime."
    )

    tau_p: pd.PositiveFloat = pd.Field(4e-6, title="Hole lifetime.", description="Hole lifetime.")
