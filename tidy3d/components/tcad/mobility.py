from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Literal

from pydantic import Field, NonNegativeFloat, PositiveFloat

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.constants import MICROMETER, PERCMCUBE

if TYPE_CHECKING:
    import numpy as np


# Runtime floor on every field dependence's 'v_sat * (T/300)**exp_v_sat' [cm/s] and
# clamp on the Canali 'beta * (T/300)**exp_beta'. Reaching either at a known lattice
# temperature is rejected at setup instead of being silently clamped.
V_SAT_FLOOR = 5e5
CANALI_BETA_MIN = 0.1
CANALI_BETA_MAX = 20.0
TRANSFERRED_ELECTRON_GAMMA_MIN = 2.0
TRANSFERRED_ELECTRON_GAMMA_MAX = 20.0
# Reference density [cm^-3] of the quasi-Fermi driving force when none is given.
QUASI_FERMI_REFERENCE_DENSITY = 1e10

_SURFACE_DEPENDENCE_DESCRIPTION = (
    "Optional surface mobility degradation near semiconductor/insulator interfaces, "
    "combined with the low-field mobility computed by this model by Matthiessen's rule. "
    "When ``None``, the mobility does not depend on the transverse field. Supported only "
    "by the accelerated charge solver, and must be set on both the electron and hole "
    "mobility models of a medium or on neither."
)

_FIELD_DEPENDENCE_DESCRIPTION = (
    "Optional high-field mobility model applied on top of the low-field "
    "mobility computed by this model. When ``None``, the mobility does not depend on "
    "the driving force. Supported only by the accelerated charge solver, and must be "
    "set on both the electron and hole mobility models of a medium or on neither."
)


class _HighFieldDependenceBase(Tidy3dBaseModel, ABC):
    """Fields and validation shared by the high-field mobility laws."""

    v_sat: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Saturation velocity",
        description="Carrier drift velocity :math:`v_{sat}` approached at high driving "
        "force, at the reference temperature (300K).",
        json_schema_extra={"units": "cm/s"},
    )

    exp_v_sat: float = Field(
        default=0.0,
        allow_inf_nan=False,
        title="Exponent for temperature dependence of the saturation velocity",
        description="Exponent for the temperature dependence of the saturation velocity, "
        "``v_sat(T) = v_sat * (T/300)**exp_v_sat``.",
    )

    driving_force: Literal["electric_field", "quasi_fermi_gradient"] = Field(
        default="electric_field",
        title="Driving force",
        description="Driving force :math:`F` of the high-field mobility: the electric field "
        "acting on the carrier, or the gradient of its quasi-Fermi potential. Where the "
        "carrier density is well above ``reference_density``, the quasi-Fermi gradient "
        "vanishes under the built-in fields of junctions and heterointerfaces at "
        "equilibrium, so they do not enter the high-field mobility. With "
        "``'quasi_fermi_gradient'`` the DC solve can take up to about twice as long.",
    )

    reference_density: PositiveFloat | None = Field(
        default=None,
        allow_inf_nan=False,
        title="Reference density",
        description="Density :math:`n_0` added to the carrier density in the quasi-Fermi "
        "potential of the ``'quasi_fermi_gradient'`` driving force, which keeps the driving "
        "force finite in depleted regions. Where the carrier density is comparable to or "
        "below it, the driving force approaches the electric field. Valid only with that "
        "driving force; when ``None``, 1e10 cm^-3 is used.",
        json_schema_extra={"units": PERCMCUBE},
    )

    @property
    def reference_density_resolved(self) -> float | None:
        """'reference_density' with its default for the quasi-Fermi driving force, or
        ``None`` for the electric field."""
        if self.driving_force != "quasi_fermi_gradient":
            return None
        if self.reference_density is None:
            return QUASI_FERMI_REFERENCE_DENSITY
        return self.reference_density


class CanaliFieldDependence(_HighFieldDependenceBase):
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
    carrier, selected by ``driving_force``. With ``"electric_field"`` it is the
    electrostatic field plus the band-structure and thermal-gradient
    contributions. With ``"quasi_fermi_gradient"`` it is the gradient of the
    carrier's quasi-Fermi potential [2]_, [3]_, [4]_, evaluated with the
    carrier density :math:`c` replaced by :math:`c + n_0`, where :math:`n_0` is
    ``reference_density``. Where :math:`c \\gg n_0` this gradient vanishes at
    equilibrium, so the built-in fields of junctions and heterointerfaces do not
    reduce the mobility there. Where :math:`c` is comparable to or below
    :math:`n_0`, as in depleted regions, the fraction :math:`n_0 / (c + n_0)` of
    the electric-field driving force remains, so built-in and gate fields can still
    reduce the mobility of those carriers. With either driving force the mobility
    drops wherever the carrier velocity approaches :math:`v_{sat}`, for example
    where minority carriers enter through an ohmic contact. With
    ``"quasi_fermi_gradient"`` each bias point is solved twice, so the DC solve can
    take up to about twice as long.

    The saturation velocity and the Canali exponent follow
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
    .. [2] S. Selberherr. Analysis and Simulation of Semiconductor Devices.
           Springer, Vienna, 1984.
    .. [3] S. Selberherr, W. Hänsch, M. Seavey, and J. Slotboom. The evolution of
           the MINIMOS mobility model. Solid-State Electronics, 33(11):1425-1436,
           1990.
    .. [4] T. Simlinger, H. Brech, T. Grave, and S. Selberherr. Simulation of
           submicron double-heterojunction high electron mobility transistors
           with MINIMOS-NT. IEEE Transactions on Electron Devices,
           44(5):700-707, 1997.

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
        ...   mu=1417.0,
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
    drift velocity. Use :class:`TransferredElectronFieldDependence` for them.
    """

    beta: float = Field(
        default=1.0,
        ge=CANALI_BETA_MIN,
        le=CANALI_BETA_MAX,
        title="Canali exponent",
        description="Canali exponent :math:`\\beta` at the reference temperature (300K), "
        "controlling how sharply the mobility rolls off at the saturation field. "
        f"Must lie in [{CANALI_BETA_MIN}, {CANALI_BETA_MAX}].",
    )

    exp_beta: float = Field(
        default=0.0,
        allow_inf_nan=False,
        title="Exponent for temperature dependence of the Canali exponent",
        description="Exponent for the temperature dependence of the Canali exponent, "
        "``beta(T) = beta * (T/300)**exp_beta``.",
    )


class TransferredElectronFieldDependence(_HighFieldDependenceBase):
    """High-field mobility with negative differential mobility (transferred-electron effect).

    Notes
    -----
        The transferred-electron model [1]_, [2]_, [3]_ describes electrons in
        direct-gap III-V semiconductors, whose drift velocity peaks and then
        *decreases* toward :math:`v_{sat}` as the driving force grows. It degrades
        the low-field mobility :math:`\\mu_{low}` computed by the mobility model it
        is attached to:

        .. math::

            \\mu(F) = \\frac{\\mu_{low} + \\left(v_{sat} / F\\right)
            \\left(F / F_0\\right)^\\gamma}
            {1 + \\left(F / F_0\\right)^\\gamma}

    so that :math:`\\mu \\to \\mu_{low}` for :math:`F \\ll F_0` and the drift
    velocity :math:`\\mu F \\to v_{sat}` for :math:`F \\gg F_0`. Here :math:`F`
    [V/cm] is the magnitude of the local driving force on that carrier, selected
    by ``driving_force`` as for :class:`CanaliFieldDependence`. With
    ``"electric_field"`` it is the electrostatic field plus the band-structure and
    thermal-gradient contributions. With ``"quasi_fermi_gradient"`` it is the
    gradient of the carrier's quasi-Fermi potential [2]_, [3]_, evaluated with the
    carrier density :math:`c` replaced by :math:`c + n_0`, where :math:`n_0` is
    ``reference_density``. Where :math:`c \\gg n_0` this gradient vanishes at
    equilibrium, so the built-in fields of junctions and heterointerfaces do not
    move the carriers past the velocity peak there. With ``"quasi_fermi_gradient"``
    the DC solve can take up to about twice as long.

    Published parameter sets use :math:`\\gamma = 4` for GaAs and the other
    III-V binaries [2]_, [3]_; [4]_ uses 3 for InP and 2 for GaP.

    The saturation velocity follows a power-law temperature dependence

    .. math::

        v_{sat}(T) = v_{sat} \\left(\\frac{T}{300K}\\right)^{\\alpha_{v}}

    with :math:`\\alpha_v` given by ``exp_v_sat``; :math:`F_0` and :math:`\\gamma`
    do not depend on temperature. The solver floors :math:`v_{sat}(T)` at
    :math:`5 \\times 10^5` cm/s. When the lattice temperature is known when the
    simulation is built — a scalar analysis temperature *or* a prescribed
    temperature field — configurations that reach the floor are rejected at
    validation time; when the temperature is solved for (self-heating), the floor
    is applied at runtime against the live temperature field.

    .. [1] J. J. Barnes, R. J. Lomax, and G. I. Haddad. Finite-element simulation
           of GaAs MESFET's with lateral doping profiles and submicron gates. IEEE
           Transactions on Electron Devices, 23(9):1042-1048, 1976.
    .. [2] S. Selberherr. Analysis and Simulation of Semiconductor Devices.
           Springer, Vienna, 1984.
    .. [3] V. Palankovski and R. Quay. Analysis and Simulation of Heterostructure
           Devices. Springer, Vienna, 2004.
    .. [4] V. Palankovski. Simulation of Heterojunction Bipolar Transistors.
           Dissertation, Technische Universität Wien, 2000.

    Note
    ----
    Past the velocity peak the differential mobility is negative, so the current
    is no longer a monotone function of the applied voltage: a current-driven
    contact can then match several bias points. Sweep the voltage instead. Long,
    lightly doped regions biased in this range can also be intrinsically unstable
    (Gunn domains), and a steady-state solution may be hard to reach or not exist.

    With ``"electric_field"``, the law also acts on the built-in field of an
    isotype junction at equilibrium, where no current flows: a GaAs step from
    :math:`10^{17}` to :math:`10^{15}` cm⁻³ carries about :math:`1.6 \\times 10^4`
    V/cm, four times :math:`F_0`. A 1 um, :math:`10^{15}` cm⁻³ region between two
    such contacts loses about 9 % of its zero-bias conductance, and its DC sweep
    converges at zero bias but not at the following bias steps (on a finer mesh,
    not even at zero bias). ``"quasi_fermi_gradient"`` avoids both, but only
    below the velocity peak: past it, a steeper carrier-density gradient
    along the current lowers the drift velocity and acts as a negative diffusion,
    so the DC solve does not converge once the drift velocity :math:`J / (q c)`
    reaches the peak anywhere in the device, near 0.2 V in the region above.
    For devices with such junctions, use ``"quasi_fermi_gradient"`` and stay below
    the velocity peak: past it, neither driving force currently converges. Without
    such junctions, transport past the peak converges with ``"electric_field"``.

    Example
    -------
    GaAs electrons at 300 K:

        >>> import tidy3d as td
        >>> transferred_electron_GaAs_n = td.TransferredElectronFieldDependence(
        ...   v_sat=7.7e6,
        ...   F_0=4.0e3,
        ...   gamma=4.0,
        ...   driving_force="quasi_fermi_gradient",
        ... )

    attached to the low-field model of the electrons, here the undoped GaAs
    electron mobility (the velocity then peaks at about 2.2e7 cm/s near 3.4 kV/cm):

        >>> mobility_n = td.ConstantMobilityModel(
        ...   mu=8500.0,
        ...   field_dependence=transferred_electron_GaAs_n,
        ... )
    """

    F_0: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Reference field",
        description="Reference field :math:`F_0` at which the high-field term takes over "
        "from the low-field mobility; the velocity peak lies close to it.",
        json_schema_extra={"units": "V/cm"},
    )

    gamma: float = Field(
        default=4.0,
        ge=TRANSFERRED_ELECTRON_GAMMA_MIN,
        le=TRANSFERRED_ELECTRON_GAMMA_MAX,
        title="Transition exponent",
        description="Exponent :math:`\\gamma` controlling how sharply the mobility "
        "transitions from the low-field to the high-field branch. Published parameter sets "
        "use 4 for GaAs, AlAs and InAs, 3 or 4 for InP, and 2 for GaP. "
        f"Must lie in [{TRANSFERRED_ELECTRON_GAMMA_MIN}, {TRANSFERRED_ELECTRON_GAMMA_MAX}]: "
        "below :math:`F_0` the mobility exceeds :math:`\\mu_{low}` by up to "
        ":math:`(v_{sat}/F_0)(F/F_0)^{\\gamma-1}`, which for :math:`\\gamma` close to 1 "
        "stays large even far below :math:`F_0`. The upper bound is that of the Canali "
        "exponent.",
    )


FieldDependenceType = CanaliFieldDependence | TransferredElectronFieldDependence


class LombardiSurfaceDependence(Tidy3dBaseModel):
    """Surface mobility degradation with the Lombardi model.

    Notes
    -----
        The Lombardi model [1]_ adds the acoustic-phonon and surface-roughness
        scattering of carriers confined against a semiconductor/insulator
        interface, such as a MOSFET inversion layer, to the low-field mobility
        :math:`\\mu_b` of the mobility model it is attached to:

        .. math::

            \\frac{1}{\\mu} = \\frac{1}{\\mu_b}
            + \\frac{D}{\\mu_{ac}} + \\frac{D}{\\mu_{sr}},
            \\qquad
            D = \\operatorname{sech}\\left(\\left(d / l_{crit}\\right)^2\\right)

        with

        .. math::

            \\mu_{ac} = \\frac{B}{E_\\perp}
            + \\frac{C \\left(N / 1\\,\\text{cm}^{-3}\\right)^{\\lambda}}
            {\\left(T / 300\\,\\text{K}\\right)^{k} E_\\perp^{1/3}},
            \\qquad
            \\mu_{sr} = \\frac{\\delta}{E_\\perp^{\\gamma}}

    where :math:`N` is the total doping concentration, :math:`T` the lattice
    temperature and :math:`d` the distance to the nearest semiconductor/insulator
    interface. The weight :math:`D` is the surface-to-bulk transition of [2]_,
    applied here to the terms of [1]_: it is close to 1 across the inversion layer
    and vanishes beyond a few :math:`l_{crit}`. :math:`E_\\perp` [V/cm] is the
    component, along the direction toward that interface, of the field driving
    the carrier (electrostatic plus band-structure and thermal-gradient
    contributions, without the degeneracy term of Fermi-Dirac statistics). Only
    a field pushing the carrier toward the interface degrades its mobility.

    With a ``field_dependence`` (:class:`CanaliFieldDependence` or
    :class:`TransferredElectronFieldDependence`) on the same mobility model, the
    degraded mobility is its low-field input. With the ``"electric_field"`` driving
    force, the field driving the high-field model loses its interface-normal
    component in proportion to :math:`D`: no current crosses an insulator, so near
    the interface the current runs along it. Within :math:`6.5\\,l_{crit}` of the
    interface, this model also changes the high-field mobility where :math:`D` is
    negligible. With the ``"quasi_fermi_gradient"`` driving force the whole
    quasi-Fermi gradient drives the high-field model [2]_. Where the carrier density
    is well above ``reference_density``, as across the inversion layer, the
    confining field does not enter it, since no current crosses the interface [3]_.

    Where the nearest interface switches, as across the bisector of a corner or
    the middle of a film between two insulators, the direction of
    :math:`E_\\perp`, and with it the mobility, can change abruptly.

    .. [1] C. Lombardi, S. Manzini, A. Saporito, and M. Vanzi. A physically based
           mobility model for numerical simulation of nonplanar devices. IEEE
           Transactions on Computer-Aided Design, 7(11):1164-1171, 1988.
    .. [2] S. Selberherr, W. Hänsch, M. Seavey, and J. Slotboom. The evolution of
           the MINIMOS mobility model. Solid-State Electronics, 33(11):1425-1436,
           1990.
    .. [3] V. Palankovski. Simulation of Heterojunction Bipolar Transistors.
           Dissertation, Technische Universität Wien, 2000.

    Note
    ----
    The mesh must resolve the inversion layer: about 0.1 nm normal to the
    interface in the first rows of semiconductor nodes.

    Example
    -------
    The silicon parameters of [1]_, with :math:`C` divided by 300 K for the
    :math:`T / 300\\,\\text{K}` scaling and the hole :math:`B` of
    :math:`9.93 \\times 10^6` cm/s that its fits require (its Table II prints
    :math:`9.93 \\times 10^7`):

        >>> import tidy3d as td
        >>> lombardi_Si_n = td.LombardiSurfaceDependence(
        ...   B=4.75e7,
        ...   C=580.0,
        ...   exp_N=0.125,
        ...   delta=5.82e14,
        ... )
        >>> lombardi_Si_p = td.LombardiSurfaceDependence(
        ...   B=9.93e6,
        ...   C=2947.0,
        ...   exp_N=0.0317,
        ...   delta=2.05e14,
        ... )

    attached to the low-field model of each carrier:

        >>> mobility_n = td.MasettiMobility(
        ...   mu_max=1417.0,
        ...   mu_0=52.2,
        ...   mu_1=43.4,
        ...   Cr=9.68e16,
        ...   Cs=3.43e20,
        ...   alpha=0.68,
        ...   beta=2.0,
        ...   exp_max=-2.5,
        ...   exp_0=-0.57,
        ...   surface_dependence=lombardi_Si_n,
        ... )
    """

    B: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Acoustic-phonon coefficient",
        description="Coefficient :math:`B` of the field-limited acoustic-phonon term.",
        json_schema_extra={"units": "cm/s"},
    )

    C: NonNegativeFloat = Field(
        allow_inf_nan=False,
        title="Doping-dependent acoustic-phonon coefficient",
        description="Coefficient :math:`C` of the doping- and temperature-dependent "
        "acoustic-phonon term at the reference temperature (300K).",
        json_schema_extra={"units": "cm^(5/3)/(V^(2/3) s)"},
    )

    exp_N: NonNegativeFloat = Field(
        allow_inf_nan=False,
        title="Doping exponent",
        description="Exponent :math:`\\lambda` of the doping dependence of the "
        "acoustic-phonon term.",
    )

    exp_T: float = Field(
        default=1.0,
        allow_inf_nan=False,
        title="Temperature exponent",
        description="Exponent :math:`k` of the temperature dependence of the "
        "acoustic-phonon term, which scales as ``(T/300)**(-exp_T)``.",
    )

    delta: PositiveFloat = Field(
        allow_inf_nan=False,
        title="Surface-roughness coefficient",
        description="Coefficient :math:`\\delta` of the surface-roughness term.",
        json_schema_extra={"units": "cm^2/V-s (V/cm)^exp_E"},
    )

    exp_E: float = Field(
        default=2.0,
        ge=1.0,
        allow_inf_nan=False,
        title="Surface-roughness field exponent",
        description="Exponent :math:`\\gamma` of the transverse-field dependence of the "
        "surface-roughness term. Must be at least 1.",
    )

    l_crit: PositiveFloat = Field(
        default=0.01,
        allow_inf_nan=False,
        title="Decay length",
        description="Length :math:`l_{crit}` over which the surface terms decay with the "
        "distance to the nearest semiconductor/insulator interface.",
        json_schema_extra={"units": MICROMETER},
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

    field_dependence: FieldDependenceType | None = Field(
        default=None,
        discriminator=TYPE_TAG_STR,
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
    .. [2] G. Masetti, M. Severi, and S. Solmi. Modeling of carrier mobility against
           carrier concentration in arsenic-, phosphorus-, and boron-doped silicon.
           IEEE Transactions on Electron Devices, 30(7):764-769, 1983.

    Example
    -------
    Silicon, with the doping parameters (``mu_min``, ``mu``, ``ref_N``, ``exp_N``) of
    Masetti et al. [2]_:

        >>> import tidy3d as td
        >>> mobility_Si_n = td.CaugheyThomasMobility(
        ...   mu_min=52.2,
        ...   mu=1417.0,
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
    High-field mobility can be enabled by supplying a
    :class:`CanaliFieldDependence` or :class:`TransferredElectronFieldDependence`
    model through ``field_dependence``, and
    surface degradation near insulator interfaces by supplying a
    :class:`LombardiSurfaceDependence` model through ``surface_dependence``.
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

    field_dependence: FieldDependenceType | None = Field(
        default=None,
        discriminator=TYPE_TAG_STR,
        title="High-field mobility dependence",
        description=_FIELD_DEPENDENCE_DESCRIPTION,
    )

    surface_dependence: LombardiSurfaceDependence | None = Field(
        default=None,
        title="Surface mobility dependence",
        description=_SURFACE_DEPENDENCE_DESCRIPTION,
    )

    def _low_field_mobility(self, doping: np.ndarray, temperature: float) -> np.ndarray:
        """Low-field mobility [cm^2/V-s] at the given doping [cm^-3] and temperature [K]."""
        scale = temperature / 300.0
        mu_min = self.mu_min * scale**self.exp_1
        mu_max = self.mu * scale**self.exp_2
        ref_N = self.ref_N * scale**self.exp_3
        exp_N = self.exp_N * scale**self.exp_4
        return mu_min + (mu_max - mu_min) / (1.0 + (doping / ref_N) ** exp_N)


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

    field_dependence: FieldDependenceType | None = Field(
        default=None,
        discriminator=TYPE_TAG_STR,
        title="High-field mobility dependence",
        description=_FIELD_DEPENDENCE_DESCRIPTION,
    )

    surface_dependence: LombardiSurfaceDependence | None = Field(
        default=None,
        title="Surface mobility dependence",
        description=_SURFACE_DEPENDENCE_DESCRIPTION,
    )

    @property
    def mu_min2_resolved(self) -> float:
        """'mu_min2', or 'mu_0' when it is not set."""
        return self.mu_0 if self.mu_min2 is None else self.mu_min2


class AlloyMobility(Tidy3dBaseModel):
    """Low-field carrier mobility of a III-V alloy, interpolated between two end members.

    Notes
    -----
        The mobility of a ternary or pseudo-binary alloy :math:`A_{1-x}B_x` is the harmonic
        interpolation of the doping- and temperature-dependent end-member mobilities, plus an
        alloy-scattering term [1]_:

        .. math::

            \\frac{1}{\\mu(N,T)} = \\frac{1-x}{\\mu_{x0}(N,T)} + \\frac{x}{\\mu_{x1}(N,T)}
            + \\frac{x(1-x)}{\\mu_{alloy}}

    where :math:`N` is the total doping (acceptors + donors) and :math:`\\mu_{x0}` and
    :math:`\\mu_{x1}` are the :class:`CaugheyThomasMobility` of the end members :math:`A`
    (:math:`x = 0`) and :math:`B` (:math:`x = 1`). As in [1]_, the end members are the full
    low-field mobilities, with lattice and impurity scattering. The alloy-scattering term
    vanishes at both end members. :math:`\\mu_{alloy}` is sign free: a positive value lowers
    the mobility (alloy scattering), a negative one raises it.

    The model is single-valley: for electrons it describes one conduction-band valley and
    holds only where that valley dominates transport, e.g.
    :math:`\\mathrm{Al}_x\\mathrm{Ga}_{1-x}\\mathrm{As}` electrons in the :math:`\\Gamma`-valley
    range :math:`x < 0.45`. A negative
    :math:`\\mu_{alloy}` can drive :math:`1/\\mu` to zero at low temperature. With the
    :math:`\\mathrm{Al}_x\\mathrm{Ga}_{1-x}\\mathrm{As}` :math:`\\Gamma`-valley electron values
    of [1]_ (GaAs and AlAs lattice mobilities of 8500 and 250 :math:`\\text{cm}^2/\\text{V-s}`,
    :math:`\\mu_{alloy} = -250` :math:`\\text{cm}^2/\\text{V-s}`), :math:`1/\\mu` reaches zero
    below about 220-260 K at low doping for :math:`x < 0.45`. A setup that does so is rejected
    when its temperatures are known: the prescribed temperature, or 300 K and the
    heat-boundary temperatures when the temperature is solved for. If a solved lattice
    temperature still drives :math:`1/\\mu` to zero, the solver stops with an error naming the
    zone, carrier, temperature and doping. An end member may itself be an alloy described by
    its own :class:`CaugheyThomasMobility` fit, e.g. InGaAsP lattice-matched to InP as an
    interpolation between InP and In0.53Ga0.47As.

    This model is supported only by the accelerated charge solver. It must be used for
    both electron and hole mobility within a semiconductor medium, with the same
    ``mole_fraction``.

    The following table maps the symbols used in the equations above with the names used in
    the code:

    .. list-table::
       :widths: 25 25 75
       :header-rows: 1

       * - Symbol
         - Parameter Name
         - Description
       * - :math:`\\mu_{x0}`
         - ``mobility_x0``
         - Mobility of the :math:`x = 0` end member.
       * - :math:`\\mu_{x1}`
         - ``mobility_x1``
         - Mobility of the :math:`x = 1` end member.
       * - :math:`x`
         - ``mole_fraction``
         - Mole fraction of the :math:`x = 1` end member.
       * - :math:`\\mu_{alloy}`
         - ``mu_alloy``
         - Alloy-scattering (bowing) mobility.

    .. [1] V. Palankovski and R. Quay. Analysis and Simulation of Heterostructure Devices.
           Springer, Vienna, 2004.

    Example
    -------
        >>> import tidy3d as td
        >>> mobility_GaAs_n = td.CaugheyThomasMobility(
        ...   mu_min=800.0,
        ...   mu=8500.0,
        ...   ref_N=1e17,
        ...   exp_N=0.5,
        ...   exp_1=-0.9,
        ...   exp_2=-2.2,
        ...   exp_3=6.2,
        ...   exp_4=0.0,
        ... )
        >>> mobility_InAs_n = td.CaugheyThomasMobility(
        ...   mu_min=11700.0,
        ...   mu=32500.0,
        ...   ref_N=4.4e16,
        ...   exp_N=0.5,
        ...   exp_1=-0.33,
        ...   exp_2=-1.7,
        ...   exp_3=3.6,
        ...   exp_4=0.0,
        ... )
        >>> mobility_InGaAs_n = td.AlloyMobility(
        ...   mobility_x0=mobility_GaAs_n,
        ...   mobility_x1=mobility_InAs_n,
        ...   mole_fraction=0.53,
        ... )
    """

    mobility_x0: CaugheyThomasMobility = Field(
        title="End-member mobility at mole fraction 0",
        description="Low-field mobility of the end member at ``mole_fraction = 0``. "
        "Must not set 'field_dependence' or 'surface_dependence'.",
    )

    mobility_x1: CaugheyThomasMobility = Field(
        title="End-member mobility at mole fraction 1",
        description="Low-field mobility of the end member at ``mole_fraction = 1``. "
        "Must not set 'field_dependence' or 'surface_dependence'.",
    )

    mole_fraction: float = Field(
        ge=0.0,
        le=1.0,
        title="Mole fraction",
        description="Mole fraction of the 'mobility_x1' end member.",
    )

    mu_alloy: float | None = Field(
        default=None,
        allow_inf_nan=False,
        title="Alloy-scattering mobility",
        description="Alloy-scattering (bowing) mobility :math:`\\mu_{alloy}`. Positive values "
        "lower the alloy mobility, negative values raise it. When ``None``, there is no "
        "bowing term.",
        json_schema_extra={"units": "cm^2/V-s"},
    )

    field_dependence: FieldDependenceType | None = Field(
        default=None,
        discriminator=TYPE_TAG_STR,
        title="High-field mobility dependence",
        description=_FIELD_DEPENDENCE_DESCRIPTION,
    )

    surface_dependence: LombardiSurfaceDependence | None = Field(
        default=None,
        title="Surface mobility dependence",
        description=_SURFACE_DEPENDENCE_DESCRIPTION,
    )

    def _inverse_low_field_mobility(self, doping: np.ndarray, temperature: float) -> np.ndarray:
        """Inverse low-field mobility [V-s/cm^2], whose sign tells whether the mobility is valid."""
        inverse = (1.0 - self.mole_fraction) / self.mobility_x0._low_field_mobility(
            doping, temperature
        ) + self.mole_fraction / self.mobility_x1._low_field_mobility(doping, temperature)
        if self.mu_alloy is not None:
            inverse = inverse + self.mole_fraction * (1.0 - self.mole_fraction) / self.mu_alloy
        return inverse
