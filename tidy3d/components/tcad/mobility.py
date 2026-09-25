from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import PERCMCUBE

if TYPE_CHECKING:
    from tidy3d.compat import Self


# Floor the accelerated charge solver applies to the temperature-scaled Canali
# saturation velocity 'v_sat * (T/300)**exp_v_sat' [cm/s], and the bounds it
# clamps the temperature-scaled exponent 'beta * (T/300)**exp_beta' to.
# Configurations that reach either at a known lattice temperature are rejected at
# setup time instead of being silently clamped.
CANALI_V_SAT_FLOOR = 5e5
CANALI_BETA_MIN = 0.1
CANALI_BETA_MAX = 20.0

_FIELD_DEPENDENCE_DESCRIPTION = (
    "Optional high-field velocity-saturation model applied on top of the low-field "
    "mobility computed by this model. When ``None``, the mobility does not depend on "
    "the driving force. Supported only by the accelerated charge solver, and must be "
    "set on both the electron and hole mobility models of a medium or on neither."
)


class CanaliFieldDependence(Tidy3dBaseModel):
    """High-field velocity saturation with the extended Canali model.

    Notes
    -----
        The extended Canali model [1]_ degrades the low-field mobility
        :math:`\\mu_{low}` computed by the mobility model it is attached to, so
        that the carrier drift velocity saturates at :math:`v_{sat}` for large
        driving forces:

        .. math::

            \\mu(F) = \\frac{\\mu_{low}}
            {\\left[1 + \\left(\\mu_{low} F / v_{sat}\\right)^\\beta\\right]^{1/\\beta}}

    where :math:`F` [V/cm] is the magnitude of the local driving force on that
    carrier: the electrostatic field plus the band-structure and thermal-gradient
    contributions. The saturation velocity and the Canali exponent follow
    power-law temperature dependences

    .. math::

        v_{sat}(T) = v_{sat} \\left(\\frac{T}{300K}\\right)^{\\alpha_{v}},
        \\qquad
        \\beta(T) = \\beta \\left(\\frac{T}{300K}\\right)^{\\alpha_{\\beta}}

    with :math:`\\alpha_v` given by ``exp_v_sat`` and :math:`\\alpha_\\beta` by
    ``exp_beta``. The solver floors :math:`v_{sat}(T)` at :math:`5 \\times 10^5`
    cm/s and clamps :math:`\\beta(T)` to :math:`[0.1, 20]`. Which of the two
    happens depends on whether the lattice temperature is known when the
    simulation is built: when it is — a scalar analysis temperature *or* a
    prescribed temperature field — configurations that reach the floor or leave
    the clamp are rejected at validation time; when the temperature is solved
    for (self-heating), the floor and the clamp are applied at runtime against
    the live temperature field.

    .. [1] C. Canali, G. Majni, R. Minder, and G. Ottaviani. Electron and hole
           drift velocity measurements in silicon and their empirical relation
           to electric field and temperature. IEEE Transactions on Electron
           Devices, 22(11):1045-1047, 1975.

    Note
    ----
    The power-law temperature dependence degrades above roughly 400 K; treat
    results at higher lattice temperatures with care.

    Example
    -------
    The silicon parameters measured by Canali et al. [1]_:

        >>> import tidy3d as td
        >>> canali_Si_n = td.CanaliFieldDependence(
        ...   v_sat=1.07e7,
        ...   beta=1.109,
        ...   exp_v_sat=-0.87,
        ...   exp_beta=0.66,
        ... )
        >>> canali_Si_p = td.CanaliFieldDependence(
        ...   v_sat=8.37e6,
        ...   beta=1.213,
        ...   exp_v_sat=-0.52,
        ...   exp_beta=0.17,
        ... )

    attached to the low-field model of each carrier:

        >>> mobility_n = td.CaugheyThomasMobility(
        ...   mu_min=52.2,
        ...   mu=1471.0,
        ...   ref_N=9.68e16,
        ...   exp_N=0.68,
        ...   exp_1=-0.57,
        ...   exp_2=-2.33,
        ...   exp_3=2.4,
        ...   exp_4=-0.146,
        ...   field_dependence=canali_Si_n,
        ... )

    Warning
    -------
    Use this model for silicon/germanium-family (indirect-gap) semiconductors,
    whose drift velocity rises monotonically with the driving force and
    saturates — the behavior this expression describes. Do not use it for
    direct-gap III-V semiconductors (GaAs, InGaAs, InP): there the electron
    velocity peaks and then *decreases* with increasing field
    (transferred-electron effect), which no parameter choice of a monotone
    model can represent; beyond the peak field it would overestimate the
    drift velocity.
    """

    v_sat: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Saturation velocity",
        description="Carrier saturation velocity :math:`v_{sat}` at the reference "
        "temperature (300K).",
        json_schema_extra={"units": "cm/s"},
    )

    beta: float = Field(
        default=1.0,
        ge=CANALI_BETA_MIN,
        le=CANALI_BETA_MAX,
        title="Canali exponent",
        description="Canali exponent :math:`\\beta` at the reference temperature (300K), "
        "controlling how sharply the mobility rolls off at the saturation field. "
        f"Must lie in [{CANALI_BETA_MIN}, {CANALI_BETA_MAX}].",
    )

    exp_v_sat: float = Field(
        default=0.0,
        allow_inf_nan=False,
        title="Exponent for temperature dependence of the saturation velocity",
        description="Exponent for the temperature dependence of the saturation velocity, "
        "``v_sat(T) = v_sat * (T/300)**exp_v_sat``.",
    )

    exp_beta: float = Field(
        default=0.0,
        allow_inf_nan=False,
        title="Exponent for temperature dependence of the Canali exponent",
        description="Exponent for the temperature dependence of the Canali exponent, "
        "``beta(T) = beta * (T/300)**exp_beta``.",
    )


class ConstantMobilityModel(Tidy3dBaseModel):
    """Constant mobility model

    Example
    -------
    >>> import tidy3d as td
    >>> mobility_model = td.ConstantMobilityModel(mu=1500)
    """

    mu: NonNegativeFloat = Field(
        title="Mobility",
        description="Mobility",
        json_schema_extra={"units": "cm²/V-s"},
    )

    field_dependence: CanaliFieldDependence | None = Field(
        default=None,
        title="High-field mobility dependence",
        description=_FIELD_DEPENDENCE_DESCRIPTION,
    )


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
         - ``mu``
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
       * - :math:`\\alpha_3`
         - ``exp_3``
         - Exponent for the temperature dependence of the reference doping
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


    Note
    ----
    High-field velocity saturation can be enabled by supplying a
    :class:`CanaliFieldDependence` model through ``field_dependence``.
    """

    # mobilities
    mu_min: PositiveFloat = Field(
        title="Minimum electron mobility",
        description="Minimum electron mobility  :math:`\\mu_{\\text{min}}`  at reference temperature (300K).",
        json_schema_extra={"units": "cm^2/V-s"},
    )

    mu: PositiveFloat = Field(
        title="Reference mobility",
        description="Reference mobility at reference temperature (300K).",
        json_schema_extra={"units": "cm^2/V-s"},
    )

    # thermal exponent for reference mobility
    exp_2: float = Field(
        title="Exponent for temperature dependent behavior of reference mobility",
    )

    # doping exponent
    exp_N: PositiveFloat = Field(
        title="Exponent for doping dependence of mobility.",
        description="Exponent for doping dependence of mobility at reference temperature (300K).",
    )

    # reference doping
    ref_N: PositiveFloat = Field(
        title="Reference doping",
        description="Reference doping at reference temperature (300K).",
        json_schema_extra={"units": PERCMCUBE},
    )

    # temperature exponent
    exp_1: float = Field(
        title="Exponent of thermal dependence of minimum mobility.",
        description="Exponent of thermal dependence of minimum mobility.",
    )

    exp_3: float = Field(
        title="Exponent of thermal dependence of reference doping.",
        description="Exponent of thermal dependence of reference doping.",
    )

    exp_4: float = Field(
        title="Exponent of thermal dependence of the doping exponent effect.",
        description="Exponent of thermal dependence of the doping exponent effect.",
    )

    field_dependence: CanaliFieldDependence | None = Field(
        default=None,
        title="High-field mobility dependence",
        description=_FIELD_DEPENDENCE_DESCRIPTION,
    )


class MasettiMobility(Tidy3dBaseModel):
    """The Masetti doping-dependent low-field carrier mobility model.

    Notes
    -----
        The Masetti mobility model [1]_ is

        .. math::

            \\mu(N,T) =
            \\mu_0\\left(\\frac{T}{T_{ref}}\\right)^{\\gamma_0} e^{-P_c/N}
            + \\frac{
                \\mu_{max}\\left(\\frac{T}{T_{ref}}\\right)^{\\gamma_{max}}
                - \\mu_{min2}\\left(\\frac{T}{T_{ref}}\\right)^{\\gamma_0}}
                {1 + \\left(N/C_r\\right)^{\\alpha}}
            - \\frac{\\mu_1}{1 + \\left(C_s/N\\right)^{\\beta}}

    where :math:`N` is the total ionized doping concentration (acceptors + donors) and
    :math:`T_{ref}` is 300 K. The final subtractive term captures the high-doping
    clustering behavior absent from the Caughey-Thomas model. The exponential factor
    suppresses the :math:`\\mu_0` floor at low doping; Masetti et al. use it for
    boron-doped silicon (:math:`P_c > 0`, :math:`\\mu_{min2} = 0`). The defaults
    :math:`P_c = 0` and :math:`\\mu_{min2} = \\mu_0` give their arsenic and
    phosphorus form.

    This model is supported only by the accelerated charge solver. It must be
    used for both electron and hole mobility within a semiconductor medium.

    The following table maps the symbols used in the equations above with the names used in
    the code:

    .. list-table::
       :widths: 25 25 75
       :header-rows: 1

       * - Symbol
         - Parameter Name
         - Description
       * - :math:`\\mu_{max}`
         - ``mu_max``
         - High-mobility plateau at 300 K.
       * - :math:`\\mu_0`
         - ``mu_0``
         - Mid-doping floor at 300 K.
       * - :math:`\\mu_{min2}`
         - ``mu_min2``
         - Floor subtracted from the plateau in the transition term at 300 K.
       * - :math:`\\mu_1`
         - ``mu_1``
         - High-doping clustering amplitude.
       * - :math:`P_c`
         - ``Pc``
         - Low-doping cutoff concentration of the :math:`\\mu_0` floor.
       * - :math:`C_r`
         - ``Cr``
         - First transition doping concentration.
       * - :math:`C_s`
         - ``Cs``
         - Clustering-onset doping concentration.
       * - :math:`\\alpha`
         - ``alpha``
         - First denominator exponent.
       * - :math:`\\beta`
         - ``beta``
         - Clustering denominator exponent.
       * - :math:`\\gamma_{max}`
         - ``exp_max``
         - Temperature exponent for ``mu_max``.
       * - :math:`\\gamma_0`
         - ``exp_0``
         - Temperature exponent for ``mu_0`` and ``mu_min2``.

    .. [1] G. Masetti, M. Severi, and S. Solmi. Modeling of carrier mobility against
           carrier concentration in arsenic-, phosphorus-, and boron-doped silicon.
           IEEE Transactions on Electron Devices, 30(7), 1983.

    Example
    -------
        >>> import tidy3d as td
        >>> mobility_Si_n = td.MasettiMobility(
        ...   mu_max=1417.0,
        ...   mu_0=52.2,
        ...   mu_1=43.4,
        ...   Cr=9.68e16,
        ...   Cs=3.43e20,
        ...   alpha=0.68,
        ...   beta=2.0,
        ...   exp_max=-2.5,
        ...   exp_0=-0.57,
        ... )
        >>> mobility_Si_p = td.MasettiMobility(
        ...   mu_max=470.5,
        ...   mu_0=44.9,
        ...   mu_min2=0.0,
        ...   mu_1=29.0,
        ...   Pc=9.23e16,
        ...   Cr=2.23e17,
        ...   Cs=6.10e20,
        ...   alpha=0.719,
        ...   beta=2.0,
        ...   exp_max=-2.2,
        ...   exp_0=-0.57,
        ... )
    """

    mu_max: PositiveFloat = Field(
        title="Maximum mobility",
        description="High-mobility plateau at reference temperature (300K).",
        json_schema_extra={"units": "cm^2/V-s"},
    )

    mu_0: PositiveFloat = Field(
        title="Reference low mobility",
        description="Mid-doping mobility floor at reference temperature (300K).",
        json_schema_extra={"units": "cm^2/V-s"},
    )

    mu_min2: NonNegativeFloat | None = Field(
        default=None,
        title="Transition-term low mobility",
        description="Floor subtracted from 'mu_max' in the transition term at reference "
        "temperature (300K). Scales with temperature through 'exp_0'. When ``None``, it "
        "equals 'mu_0'.",
        json_schema_extra={"units": "cm^2/V-s"},
    )

    mu_1: NonNegativeFloat = Field(
        title="Clustering mobility",
        description="High-doping clustering mobility amplitude.",
        json_schema_extra={"units": "cm^2/V-s"},
    )

    Pc: NonNegativeFloat = Field(
        default=0.0,
        title="Cutoff doping",
        description="Low-doping cutoff concentration of the 'mu_0' floor, which is "
        "weighted by 'exp(-Pc/N)'. Zero disables the cutoff.",
        json_schema_extra={"units": PERCMCUBE},
    )

    Cr: PositiveFloat = Field(
        title="Transition doping",
        description="First transition doping concentration.",
        json_schema_extra={"units": PERCMCUBE},
    )

    Cs: PositiveFloat = Field(
        title="Clustering doping",
        description="Clustering-onset doping concentration.",
        json_schema_extra={"units": PERCMCUBE},
    )

    alpha: PositiveFloat = Field(
        title="Doping exponent",
        description="First denominator doping exponent.",
    )

    beta: PositiveFloat = Field(
        title="Clustering exponent",
        description="Clustering denominator doping exponent.",
    )

    exp_max: float = Field(
        title="Exponent for maximum mobility temperature dependence.",
        description="Temperature exponent for the high-mobility plateau.",
    )

    exp_0: float = Field(
        title="Exponent for low mobility temperature dependence.",
        description="Temperature exponent for the mid-doping mobility floor.",
    )

    field_dependence: CanaliFieldDependence | None = Field(
        default=None,
        title="High-field mobility dependence",
        description=_FIELD_DEPENDENCE_DESCRIPTION,
    )

    @model_validator(mode="after")
    def _validate_reference_high_doping_limit(self) -> Self:
        """Ensure the high-doping asymptote is positive at the reference temperature."""
        if self.mu_1 >= self.mu_0:
            raise ValueError(
                "'mu_1' must be smaller than 'mu_0' so the Masetti high-doping mobility "
                "asymptote (mu_0 - mu_1) remains positive at the reference temperature "
                "(300 K). At other isothermal temperatures the asymptote scales as "
                "mu_0 * (T/300)**exp_0 - mu_1."
            )
        return self

    @model_validator(mode="after")
    def _validate_reference_low_doping_limit(self) -> Self:
        """Ensure the low-doping plateau is positive at the reference temperature."""
        if self.mu_min2_resolved >= self.mu_max:
            raise ValueError(
                "'mu_min2' must be smaller than 'mu_max' so the Masetti low-doping "
                "mobility plateau remains positive at the reference temperature (300 K)."
            )
        return self

    @property
    def mu_min2_resolved(self) -> float:
        """'mu_min2', or 'mu_0' when it is not set."""
        return self.mu_0 if self.mu_min2 is None else self.mu_min2
