from typing import Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel


class ConstantMobilityModel(Tidy3dBaseModel):
    """Constant mobility model

    Example
    -------
    >>> import tidy3d as td
    >>> mobility_model = td.ConstantMobilityModel(mu=1500)
    """

    mu: pd.NonNegativeFloat = pd.Field(
        ..., title="Mobility", description="Mobility", units="cm²/V-s"
    )


class CaugheyThomasHighField(Tidy3dBaseModel):
    """Caughey-Thomas high-field effect coefficients.

    Notes
    -----
      This class is typically used in a `CaugheyThomasMobility` object. The formulations
      is that described in [1]_ and summarized in the equation below

      .. math::
        \\mu = \\frac{\\mu_0}{\\left[1 + \\left(\\frac{\\mu_0 E}{v_{\\rm{sat}}} \\right)^\\beta \\right]^{1/\\beta}}

      where :math:`\\mu_0` is the low-field mobility coefficient, :math:`v_{\\rm{sat}}` is the saturation
      velocity, and :math:`E` is the electric field component in the velocity direction. :math:`\\beta`
      is a coefficient that usually takes on the values 1 (holes) and 2 (electrons).


      .. [1] M. Caughey and R.E. Thomas. Carrier mobilities in silicon empirically related to doping
           and field. Proceedings of the IEEE, 55(12):2192–2193, December 1967
    """

    v_sat: pd.PositiveFloat = pd.Field(
        ..., title="Saturation velocity", description="Saturation velocity in cm/s", units="cm/s"
    )

    beta: float = pd.Field(..., title="Exponent coefficient", description="Exponent coefficient")


MobilityHighFieldEffectModel = Union[CaugheyThomasHighField]


class CaugheyThomasMobility(Tidy3dBaseModel):
    """The Caughey-Thomas temperature-dependent carrier mobility model.

    Notes
    -----
        The general form of the Caughey-Thomas mobility model [1]_ is of the form:

        .. math::

            \\mu_0 = \\frac{\\mu_{max} - \\mu_{min}}{1 + \\left(N/N_{ref}\\right)^z} + \\mu_{min}

    where :math:`\\mu_0` represents the low-field mobility and  :math:`N` is the total doping (acceptors + donors).
    :math:`\\mu_{max}`, :math:`\\mu_{min}`, :math:`z`, and :math:`N_{ref}` are temperature dependent,
    the dependence being of the form

    .. math::

        \\phi = \\phi_{ref} \\left( \\frac{T}{T_{ref}}\\right)^\\alpha

    and :math:`T_{ref}` is taken to be 300K.

    The complete form (with temperature effects) for the low-field mobility can be written as

    .. math::

        \\mu_0 = \\frac{\\mu_{max}(\\frac{T}{T_{ref}})^{\\alpha_2} - \\mu_{min}(\\frac{T}{T_{ref}})^{\\alpha_1}}{1 + \\left(N/N_{ref}(\\frac{T}{T_{ref}})^{\\alpha_3}\\right)^{\\alpha_N(\\frac{T}{T_{ref}})^{\\alpha_4}}} + \\mu_{min}(\\frac{T}{T_{ref}})^{\\alpha_1}

    The following table maps the symbols used in the equations above with the names used in the code:

    .. list-table::
       :widths: 25 25 75
       :header-rows: 1

       * - Symbol
         - Parameter Name
         - Description
       * - :math:`\\mu_{min}`
         - ``mu_min``
         - Minimum low-field mobility for :math:`n` and :math:`p`
       * - :math:`\\mu_{max}`
         - ``mu_n``
         - Maximum low-field mobility for :math:`n` and :math:`p`
       * - :math:`\\alpha_1`
         - ``exp_1``
         - Exponent for temperature dependence of the minimum mobility coefficient
       * - :math:`\\alpha_2`
         - ``exp_2``
         - Exponent for temperature dependence of the maximum mobility coefficient
       * - :math:`\\alpha_N`
         - ``exp_N``
         - Exponent for doping dependence.
       * - :math:`\\alpha_4`
         - ``exp_4``
         - Exponent for the temperature dependence of the exponent :math:`\\alpha_N`
       * - :math:`N_{ref}`
         - ``ref_N``,
         - Reference doping parameter


    .. [1] M. Caughey and R.E. Thomas. Carrier mobilities in silicon empirically related to doping
           and field. Proceedings of the IEEE, 55(12):2192–2193, December 1967

    Example
    -------
        >>> import tidy3d as td
        >>> mobility_Si_n = td.CaugheyThomasMobility(
        ...   mu_min=52.2,
        ...   mu=1471.0,
        ...   ref_N=9.68e16,
        ...   exp_N=0.68,
        ...   exp_1=-0.57,
        ...   exp_2=-2.33,
        ...   exp_3=2.4,
        ...   exp_4=-0.146,
        ... )
        >>> mobility_Si_p = td.CaugheyThomasMobility(
        ...   mu_min=44.9,
        ...   mu=470.5,
        ...   ref_N=2.23e17,
        ...   exp_N=0.719,
        ...   exp_1=-0.57,
        ...   exp_2=-2.33,
        ...   exp_3=2.4,
        ...   exp_4=-0.146,
        ... )

    """

    # mobilities
    mu_min: pd.PositiveFloat = pd.Field(
        ...,
        title=r"$\mu_{min}$ Minimum electron mobility",
        description="Minimum electron mobility at reference temperature (300K) in cm^2/V-s. ",
    )

    mu: pd.PositiveFloat = pd.Field(
        ...,
        title="Reference mobility",
        description="Reference mobility at reference temperature (300K) in cm^2/V-s",
    )

    # thermal exponent for reference mobility
    exp_2: float = pd.Field(
        ..., title="Exponent for temperature dependent behavior of reference mobility"
    )

    # doping exponent
    exp_N: pd.PositiveFloat = pd.Field(
        ...,
        title="Exponent for doping dependence of mobility.",
        description="Exponent for doping dependence of mobility at reference temperature (300K).",
    )

    # reference doping
    ref_N: pd.PositiveFloat = pd.Field(
        ...,
        title="Reference doping",
        description="Reference doping at reference temperature (300K) in #/cm^3.",
    )

    # temperature exponent
    exp_1: float = pd.Field(
        ...,
        title="Exponent of thermal dependence of minimum mobility.",
        description="Exponent of thermal dependence of minimum mobility.",
    )

    exp_3: float = pd.Field(
        ...,
        title="Exponent of thermal dependence of reference doping.",
        description="Exponent of thermal dependence of reference doping.",
    )

    exp_4: float = pd.Field(
        ...,
        title="Exponent of thermal dependence of the doping exponent effect.",
        description="Exponent of thermal dependence of the doping exponent effect.",
    )

    high_field: MobilityHighFieldEffectModel = pd.Field(
        None, title="High-field effect model", description="High-field effect model."
    )
