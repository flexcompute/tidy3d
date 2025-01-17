import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel


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

        \\mu_0 = \\frac{\\mu_{max}(\\frac{T}{T_{ref}})^{\\alpha_2} - \\mu_{min}(\\frac{T}{T_{ref}})^{\\alpha_1}}{1 + \\left(N/N_{ref}(\\frac{T}{T_{ref}})^{\\alpha_3}\\right)^{\\alpha_{n,p}(\\frac{T}{T_{ref}})^{\\alpha_4}}} + \\mu_{min}(\\frac{T}{T_{ref}})^{\\alpha_1}

    The following table maps the symbols used in the equations above with the names used in the code:

    .. list-table::
       :widths: 25 25 75
       :header-rows: 1

       * - Symbol
         - Parameter Name
         - Description
       * - :math:`\\mu_{min}`
         - ``mu_n_min``, ``mu_p_min``
         - Minimum low-field mobility for :math:`n` and :math:`p`
       * - :math:`\\mu_{max}`
         - ``mu_n``, ``mu_p``
         - Maximum low-field mobility for :math:`n` and :math:`p`
       * - :math:`\\alpha_1`
         - ``exp_t_mu_min``
         - Exponent for temperature dependence of the minimum mobility coefficient
       * - :math:`\\alpha_2`
         - ``exp_t_mu``
         - Exponent for temperature dependence of the maximum mobility coefficient
       * - :math:`\\alpha_{n,p}`
         - ``exp_d_p``, ``exp_d_n``
         - Exponent for doping dependence of hole mobility.
       * - :math:`\\alpha_4`
         - ``exp_t_d_exp``
         - Exponent for the temperature dependence of the exponent :math:`\\alpha_n` and :math:`\\alpha_p`
       * - :math:`N_{ref}`
         - ``ref_N``
         - Reference doping parameter


    .. [1] M. Caughey and R.E. Thomas. Carrier mobilities in silicon empirically related to doping
           and field. Proceedings of the IEEE, 55(12):2192–2193, December 1967

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.CaugheyThomasMobility(
        ...   mu_n_min=52.2,
        ...   mu_n=1471.0,
        ...   mu_p_min=44.9,
        ...   mu_p=470.5,
        ...   exp_t_mu=-2.33,
        ...   exp_d_n=0.68,
        ...   exp_d_p=0.719,
        ...   ref_N=2.23e17,
        ...   exp_t_mu_min=-0.57,
        ...   exp_t_d=2.4,
        ...   exp_t_d_exp=-0.146,
        ... )


    Warning
    -------
    There are some current limitations of this model:

    - High electric field effects not yet supported.
    """

    # mobilities
    mu_n_min: pd.PositiveFloat = pd.Field(
        title=r"$\mu_{min}$ Minimum electron mobility",
        description="Minimum electron mobility at reference temperature (300K) in cm^2/V-s. ",
    )

    mu_n: pd.PositiveFloat = pd.Field(
        title="Electron reference mobility",
        description="Reference electron mobility at reference temperature (300K) in cm^2/V-s",
    )

    mu_p_min: pd.PositiveFloat = pd.Field(
        title="Minimum hole mobility",
        description="Minimum hole mobility at reference temperature (300K) in cm^2/V-s. ",
    )

    mu_p: pd.PositiveFloat = pd.Field(
        title="Hole reference mobility",
        description="Reference hole mobility at reference temperature (300K) in cm^2/V-s",
    )

    # thermal exponent for reference mobility
    exp_t_mu: float = pd.Field(
        title="Exponent for temperature dependent behavior of reference mobility"
    )

    # doping exponent
    exp_d_n: pd.PositiveFloat = pd.Field(
        title="Exponent for doping dependence of electron mobility.",
        description="Exponent for doping dependence of electron mobility at reference temperature (300K).",
    )

    exp_d_p: pd.PositiveFloat = pd.Field(
        title="Exponent for doping dependence of hole mobility.",
        description="Exponent for doping dependence of hole mobility at reference temperature (300K).",
    )

    # reference doping
    ref_N: pd.PositiveFloat = pd.Field(
        title="Reference doping",
        description="Reference doping at reference temperature (300K) in #/cm^3.",
    )

    # temperature exponent
    exp_t_mu_min: float = pd.Field(
        title="Exponent of thermal dependence of minimum mobility.",
        description="Exponent of thermal dependence of minimum mobility.",
    )

    exp_t_d: float = pd.Field(
        title="Exponent of thermal dependence of reference doping.",
        description="Exponent of thermal dependence of reference doping.",
    )

    exp_t_d_exp: float = pd.Field(
        title="Exponent of thermal dependence of the doping exponent effect.",
        description="Exponent of thermal dependence of the doping exponent effect.",
    )
