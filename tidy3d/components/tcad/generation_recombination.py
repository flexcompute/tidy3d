import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel


class AugerRecombination(Tidy3dBaseModel):
    """
    This class defines the parameters for the Auger recombination model.

    Notes
    -----

        The Auger recombination rate ``R_A`` is primarily defined by the electrons and holes Auger recombination
        coefficients, :math:`C_n` and :math:`C_p`, respectively.

        .. math::

            R_A = \\left( C_n n + C_p p \\right) \\left( np - n_0 p_0 \\right)

    Note
    -----
        The default parameters are those appropriate for Silicon."""

    c_n: pd.PositiveFloat = pd.Field(
        2.8e-31, title="Constant for electrons", description="Constant for electrons in cm^6/s"
    )

    c_p: pd.PositiveFloat = pd.Field(
        9.9e-32, title="Constant for holes", description="Constant for holes in cm^6/s"
    )


class RadiativeRecombination(Tidy3dBaseModel):
    """
    This class is used to define the parameters for the radiative recombination model.

    Notes
    -----

        This is a direct recombination model primarily defined by a radiative recombination coefficient :math:`R_{\\text{rad}}`.

        .. math::

            R_{\\text{rad}} = C \\left( np - n_0 p_0 \\right)

    Note
    ----
        The default values are those appropriate for Silicon.
    """

    r_const: float = pd.Field(
        1.6e-14,
        title="Radiation constant in cm^3/s",
        description="Radiation constant in cm^3/s",
    )


class ShockleyReedHallRecombination(Tidy3dBaseModel):
    """This class defines the parameters for the Shockley-Reed-Hall (SRH) recombination model.

    Notes
    -----
        TODO verify mid gap limitations or not.
        The recombination rate parameter from this model is defined from [1]_ as follows:

        .. math::

           R_{SRH} = \\frac{n p - n_0 p_0}{\\tau_p \\left(n + \\sqrt{n_0 p_0}\\right) + \\tau_n \\left(p + \\sqrt{n_0 p_0}\\right)}.

        Note that the electron and holes densities are defined within the :class:`SemiconductorMedium`. The electron
        lifetime :math:`\\tau_n` and hole lifetimes :math:`\\tau_p` need to be defined.


        .. [1] Schenk. A model for the field and temperature dependence of shockley-read-hall
               lifetimes in silicon. Solid-State Electronics, 35:1585–1596, 1992.


    Note
    ----
    Important considerations when using this model:

    - Currently, lifetimes are considered constant (not dependent on temperature or doping)
    - Default values are those appropriate for Silicon.
    """

    tau_n: pd.PositiveFloat = pd.Field(
        3.3e-6, title="Electron lifetime.", description="Electron lifetime."
    )

    tau_p: pd.PositiveFloat = pd.Field(4e-6, title="Hole lifetime.", description="Hole lifetime.")
