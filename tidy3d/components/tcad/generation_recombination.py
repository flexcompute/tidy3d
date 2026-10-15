from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, NonNegativeFloat, PositiveFloat

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.constants import PERCMCUBE, SECOND


class FossumCarrierLifetime(Tidy3dBaseModel):
    """
    Doping- and temperature-dependent SRH carrier lifetime.

    Notes
    -----

        This model expresses the Shockley-Read-Hall carrier lifetime as a
        function of absolute temperature :math:`T` and total ionized dopant
        concentration :math:`N = N_D + N_A`:

        .. math::

            \\tau(N, T) = \\frac{\\tau_{300}\\,(T/300)^{\\alpha_T}}{A + B\\,(N/N_0) + C\\,(N/N_0)^{\\alpha}}

        The model is physically meaningful only with :math:`A = 1`; a
        warning is emitted if any other value is provided. The
        :math:`B\\,(N/N_0)` term is the linear doping form introduced by
        Fossum [1]_, which alone gives
        :math:`\\tau \\propto 1/(1 + N/N_0)`. The
        :math:`C\\,(N/N_0)^{\\alpha}` term adds a higher-order doping
        contribution; typical exponents are :math:`\\alpha = 1`
        (Fossum-shaped) or :math:`\\alpha = 2`, which reproduces the
        Auger-like high-doping behaviour obtained by collapsing the
        SRH + Auger parallel combination of Roulston et al. [2]_ into a
        single denominator. The :math:`(T/300)^{\\alpha_T}` factor is the
        empirical temperature scaling of Klaassen [3]_.

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.FossumCarrierLifetime(
        ...   tau_300=3.3e-6,
        ...   alpha_T=-0.5,
        ...   N0=7.1e15,
        ...   A=1,
        ...   B=0,
        ...   C=1,
        ...   alpha=1
        ... )

    References
    ----------

        .. [1] Fossum, J. G., and D. S. Lee. "A physical model for the
               dependence of carrier lifetime on doping density in
               nondegenerate silicon." Solid-State Electronics 25.8 (1982):
               741-747.

        .. [2] Roulston, D. J., N. D. Arora, and S. G. Chamberlain.
               "Modeling and measurement of minority-carrier lifetime versus
               doping in diffused layers of n+-p silicon diodes." IEEE
               Transactions on Electron Devices ED-29.2 (1982): 284-291.

        .. [3] Klaassen, D. B. M. "A unified mobility model for device
               simulation - II. Temperature dependence of carrier mobility
               and lifetime." Solid-State Electronics 35.7 (1992): 961-967.

    """

    tau_300: PositiveFloat = Field(
        title="Tau at 300K",
        description="Carrier lifetime at 300K",
        json_schema_extra={"units": SECOND},
    )

    alpha_T: float = Field(
        title="Exponent for thermal dependence",
        description="Exponent for thermal dependence",
    )

    N0: PositiveFloat = Field(
        title="Reference concentration",
        description="Reference concentration",
        json_schema_extra={"units": PERCMCUBE},
    )

    A: float = Field(
        title="Constant A",
        description="Constant A",
    )

    B: float = Field(
        title="Constant B",
        description="Constant B",
    )

    C: float = Field(
        title="Constant C",
        description="Constant C",
    )

    alpha: float = Field(
        title="Exponent constant",
        description="Exponent constant",
    )


class PalankovskiQuayApproxCarrierLifetime(Tidy3dBaseModel):
    """
    Doping- and temperature-dependent SRH carrier lifetime, Palankovski–Quay
    empirical (Scharfetter-style) approximation.

    Notes
    -----

        This model expresses the Shockley-Read-Hall carrier lifetime as a
        function of absolute temperature :math:`T` and total ionized dopant
        concentration :math:`N = N_D + N_A`:

        .. math::

            \\tau(N, T) = \\tau_{max}\\, \\left(\\frac{N}{N_{ref}}\\right)^{-\\gamma}\\, \\left(\\frac{300}{T}\\right)^{-\\alpha_T}

        This is the empirical Scharfetter-style form (Palankovski & Quay
        [1]_, eqs. 3.157/3.158) combined with the temperature factor of
        eqs. 3.160/3.161. The book fixes the temperature exponent at
        :math:`(300/T)^{3/2}` (i.e. :math:`\\alpha_T = -3/2`); the parameter
        is exposed here so users can override it. Material-specific values
        for :math:`\\tau_{max}`, :math:`N_{ref}`, and :math:`\\gamma` are
        tabulated for Si, SiGe, GaAs, InGaAs, and InAlAs in Table 3.38 of
        the reference.

        The trap-assisted band-to-band tunneling enhancement
        :math:`1/(1+r_\\nu)` and the surface-recombination term
        :math:`s_\\nu/y` from the book's full physics-based form are not
        included in this approximation.

        For numerical stability — the unclamped form diverges as
        :math:`N \\to 0`, which is unphysical (:math:`\\tau_{max}` is the
        intrinsic-region upper bound) and would produce NaN in the SRH
        Jacobian — the backend evaluator floors :math:`N` at :math:`N_{ref}`
        before applying the doping factor. The formula above therefore
        applies verbatim for :math:`N \\ge N_{ref}`; for :math:`N < N_{ref}`
        the lifetime saturates at
        :math:`\\tau_{max}\\,(300/T)^{-\\alpha_T}`.

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.PalankovskiQuayApproxCarrierLifetime(
        ...   tau_max=1e-5,
        ...   N_ref=1e16,
        ...   gamma=1.0,
        ...   alpha_T=-1.5,
        ... )

    References
    ----------

        .. [1] Palankovski, Vassil, and Rüdiger Quay. Analysis and
               simulation of heterostructure devices. Springer Science &
               Business Media, 2004.

    """

    tau_max: PositiveFloat = Field(
        title="Reference lifetime",
        description="Reference lifetime :math:`\\tau_{max}` from the "
        "Palankovski-Quay empirical form (book Table 3.38).",
        json_schema_extra={"units": SECOND},
    )

    N_ref: PositiveFloat = Field(
        title="Reference doping concentration",
        description="Reference doping concentration :math:`N_{ref}` in the "
        "Scharfetter doping-dependence factor.",
        json_schema_extra={"units": PERCMCUBE},
    )

    gamma: float = Field(
        default=1.0,
        title="Doping exponent",
        description="Dimensionless exponent :math:`\\gamma` of the doping-dependence factor.",
    )

    alpha_T: float = Field(
        default=-1.5,
        title="Temperature exponent",
        description="Dimensionless temperature exponent :math:`\\alpha_T`. "
        "The Palankovski-Quay model fixes this at :math:`-3/2`; users may "
        "override it.",
    )


CarrierLifetimeType = FossumCarrierLifetime | PalankovskiQuayApproxCarrierLifetime


class AugerRecombination(Tidy3dBaseModel):
    """
    Parameters for the Auger recombination model.

    Notes
    -----

        The Auger recombination rate ``R_A`` is primarily defined by the electrons and holes Auger recombination
        coefficients, :math:`C_n` and :math:`C_p`, respectively.

        .. math::

            R_A = \\left( C_n n + C_p p \\right) \\left( np - n_0 p_0 \\right)

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.AugerRecombination(
        ...   c_n=2.8e-31,
        ...   c_p=9.9e-32,
        ... )
    """

    c_n: PositiveFloat = Field(
        title="Constant for electrons",
        description="Constant for electrons.",
        json_schema_extra={"units": "cm^6/s"},
    )

    c_p: PositiveFloat = Field(
        title="Constant for holes",
        description="Constant for holes.",
        json_schema_extra={"units": "cm^6/s"},
    )


class RadiativeRecombination(Tidy3dBaseModel):
    """
    Defines the parameters for the radiative recombination model.

    Notes
    -----

        This is a direct recombination model primarily defined by a radiative recombination coefficient :math:`R_{\\text{rad}}`.

        .. math::

            R_{\\text{rad}} = C \\cdot n \\cdot p

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.RadiativeRecombination(
        ...   r_const=1.6e-14
        ... )
    """

    r_const: float = Field(
        title="Radiation constant",
        description="Radiation constant of the radiative recombination model.",
        json_schema_extra={"units": "cm^3/s"},
    )


class HurkxTrapAssistedTunneling(Tidy3dBaseModel):
    """
    Field enhancement of Shockley-Read-Hall recombination by trap-assisted tunneling.

    Notes
    -----

        This is not an additional recombination mechanism but a field
        enhancement of an existing Shockley-Read-Hall model. The process is
        *elastic* band-to-trap tunneling: a carrier crosses the triangular
        barrier tilted by the local electric field :math:`F` and is captured
        by (or emitted from) the trap at unchanged energy. The field acts on
        the trap-band matrix element, so emission and capture are enhanced
        equally, detailed balance is preserved and the trap-level
        concentrations :math:`n_1` and :math:`p_1` are unchanged. The
        enhancement therefore enters only as a shortened effective lifetime
        in the bulk,

        .. math::

            \\tau_c \\rightarrow \\frac{\\tau_c}{1 + \\Gamma_c},

        and as an increased recombination velocity at a surface,

        .. math::

            S_c \\rightarrow S_c \\left( 1 + \\Gamma_c \\right),

        for each carrier :math:`c = n, p`.

        The enhancement factor is the Hurkx field-effect integral [1]_

        .. math::

            \\Gamma_c = \\int_0^{\\tilde{E}_{n,c}}
                \\exp \\left[ w - \\frac{2}{3} \\frac{w^{3/2}}{\\tilde{E}_c} \\right] \\, dw

        in the reduced field :math:`\\tilde{E}_c = |F| / E_{0,c}`, with

        .. math::

            E_{0,c} = \\frac{\\sqrt{8 \\, m_{t,c} \\, m_0 \\, (k T)^3}}{q \\hbar}

        and the Pauli-blocked upper limit, the shorter of the band-to-trap
        barrier and the distance to the carrier's own quasi-Fermi level,

        .. math::

            \\tilde{E}_{n,n} = \\min \\left[ \\frac{E_C - E_T}{kT},
            -\\eta_n \\right], \\qquad
            \\tilde{E}_{n,p} = \\min \\left[ \\frac{E_T - E_V}{kT},
            -\\eta_p \\right],

        floored at zero. The trap sits at the intrinsic level :math:`E_i`,
        which is what the equal :math:`n_1 = p_1` convention of the
        recombination model means, so the two barriers are

        .. math::

            E_C - E_T = \\frac{E_g}{2}
            - \\frac{kT}{2} \\ln \\frac{N_V}{N_C} - E_t, \\qquad
            E_T - E_V = \\frac{E_g}{2}
            + \\frac{kT}{2} \\ln \\frac{N_V}{N_C} + E_t,

        and they still sum to :math:`E_g`: asymmetric densities of state move
        the trap within the gap, they do not change the gap. The quasi-Fermi
        coordinates :math:`\\eta_n = F_{1/2}^{-1}(n/N_C)` and
        :math:`\\eta_p = F_{1/2}^{-1}(p/N_V)` are exact under Fermi-Dirac
        statistics and reduce to :math:`\\ln(n/N_C)` and
        :math:`\\ln(p/N_V)` under Boltzmann. Tunneling into states already
        occupied below the quasi-Fermi level is forbidden, so the enhancement
        switches itself off in quasi-neutral regions and reaches its full
        value only in depletion. That window also saturates
        :math:`\\Gamma_c` at :math:`e^{\\tilde{E}_{n,c}} - 1`, a
        band-structure limit rather than a fitted one, and one that differs
        between the two carriers wherever :math:`N_C \\neq N_V`.

        The tunneling effective masses :math:`m_{t,n}` and :math:`m_{t,p}`
        are **single-carrier** masses, not the electron-hole *reduced* mass
        that appears in band-to-band tunneling: band-to-trap tunneling
        involves one carrier and one localized state, so only that carrier's
        mass enters. The reduced mass built from the same germanium band
        masses, :math:`(1/0.038 + 1/0.043)^{-1} = 0.0202 \\, m_0`, is a
        band-to-band quantity and overestimates :math:`\\Gamma` here by more
        than an order of magnitude.

        The defaults are germanium values, :math:`m_{t,n} = 0.12 \\, m_0`
        (L valley) and :math:`m_{t,p} = 0.043 \\, m_0` (light hole); a
        defensible range for :math:`m_{t,n}` in germanium is
        0.08 - 0.36 :math:`m_0`. These are material inputs, not fitting
        parameters.

        The formulation is local: the field is taken to be uniform over the
        tunneling length :math:`L_t = \\Delta E / (q F)`, so it is least
        reliable at sharp junction corners and heterointerfaces, where the
        field varies appreciably over that distance.

        On a surface, as in the bulk, the enhancement is included in both the
        steady-state and the small-signal results.

    Note
    ----
    This model is supported only by the accelerated charge solver.

    Example
    -------
        >>> import tidy3d as td
        >>> tat = td.HurkxTrapAssistedTunneling(m_t_n=0.12, m_t_p=0.043)
        >>> srh = td.ShockleyReedHallRecombination(
        ...     tau_n=1e-9,
        ...     tau_p=1e-9,
        ...     field_enhancement=tat,
        ... )

    References
    ----------
        .. [1] Hurkx, G. A. M., D. B. M. Klaassen, and M. P. G. Knuvers. "A new
               recombination model for device simulation including tunneling."
               IEEE Transactions on Electron Devices 39.2 (1992): 331-338.
    """

    m_t_n: PositiveFloat = Field(
        default=0.12,
        title="Electron tunneling effective mass",
        description="Electron tunneling effective mass :math:`m_{t,n}` in units of the "
        "free electron mass :math:`m_0`. This is the single-carrier tunneling mass, "
        "not the electron-hole reduced mass used in band-to-band tunneling. The "
        "default is the germanium L-valley value.",
        json_schema_extra={"units": "m_0"},
    )

    m_t_p: PositiveFloat = Field(
        default=0.043,
        title="Hole tunneling effective mass",
        description="Hole tunneling effective mass :math:`m_{t,p}` in units of the "
        "free electron mass :math:`m_0`. This is the single-carrier tunneling mass, "
        "not the electron-hole reduced mass used in band-to-band tunneling. The "
        "default is the germanium light-hole value.",
        json_schema_extra={"units": "m_0"},
    )


class ShockleyReedHallRecombination(Tidy3dBaseModel):
    """Defines the parameters for the Shockley-Reed-Hall (SRH) recombination model.

    Notes
    -----

        The recombination rate parameter from this model is defined from [1]_ as follows:

        .. math::

           R_{SRH} = \\frac{n p - n_0 p_0}{\\tau_p \\left(n + \\sqrt{n_0 p_0}\\right) + \\tau_n \\left(p + \\sqrt{n_0 p_0}\\right)}.

        Note that the electron and holes densities are defined within the :class:`SemiconductorMedium`. The electron
        lifetime :math:`\\tau_n` and hole lifetimes :math:`\\tau_p` need to be defined.


        .. [1] Schenk. A model for the field and temperature dependence of shockley-read-hall
               lifetimes in silicon. Solid-State Electronics, 35:1585–1596, 1992.

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.ShockleyReedHallRecombination(
        ...   tau_n=3.3e-6,
        ...   tau_p=4e-6,
        ... )

    Note
    ----
    Important considerations when using this model:

    - This model represents mid-gap traps Shockley-Reed-Hall recombination.
    """

    tau_n: PositiveFloat | CarrierLifetimeType = Field(
        title="Electron lifetime",
        description="Electron lifetime",
        json_schema_extra={"units": SECOND},
    )

    tau_p: PositiveFloat | CarrierLifetimeType = Field(
        title="Hole lifetime",
        description="Hole lifetime",
        json_schema_extra={"units": SECOND},
    )

    field_enhancement: HurkxTrapAssistedTunneling | None = Field(
        default=None,
        title="Trap-assisted tunneling field enhancement",
        description="Optional :class:`HurkxTrapAssistedTunneling` model. When set, the "
        "SRH lifetimes are field-enhanced by trap-assisted tunneling, "
        ":math:`\\tau \\rightarrow \\tau / (1 + \\Gamma)`, with :math:`\\Gamma` "
        "evaluated per carrier from the local electric field. Leaving it ``None`` gives "
        "the unenhanced SRH rate. Requires the accelerated charge solver.",
    )


class DistributedGeneration(Tidy3dBaseModel):
    """Class that allows to add a distributed generation model.

    Notes
    -----
    The generation rate will be interpolated to the simulation mesh during the setup phase.
    In places where the generation rate is not defined, it will be filled with zeros.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> coords = dict(x=x, y=y, z=z)
    >>> fd = td.SpatialDataArray(np.random.random((2,3,4)), coords=coords)
    >>> dist_g = td.DistributedGeneration(rate=fd)
    """

    rate: SpatialDataArray = Field(
        title="Generation rate",
        description="Spatially varying generation rate.",
        json_schema_extra={"units": "1/(cm^3 s)"},
    )

    @classmethod
    def from_rate_um3(cls, gen_um3: SpatialDataArray) -> DistributedGeneration:
        """Creates a DistributedGeneration from a SpatialDataArray in um^-3 s^-1."""
        gen_cm3 = np.array(gen_um3.data) * 1e12  # Convert from um^-3 to cm^-3
        new_gen = SpatialDataArray(gen_cm3, coords=gen_um3.coords)
        return cls(rate=new_gen)


class HurkxDirectBandToBandTunneling(Tidy3dBaseModel):
    """
    This class defines a direct band-to-band tunneling recombination model based on the Hurkx model.

    Notes
    -----

    The model is described in [1]_.

    The direct band-to-band tunneling recombination rate :math:`R^{\\text{BTBT}}` is primarily defined by the
    material's bandgap energy :math:`E_g` and the electric field :math:`F`.

    The default of :math:`A` is the silicon value of [2]_, where :math:`B` and :math:`F_0` of the paper
    are :math:`A` and :math:`B` here. The default of :math:`B` (1.9e6 V/cm) differs from the silicon
    value of [2]_ (1.9e7 V/cm).

    .. math::

        R^{\\text{BTBT}} = A \\cdot \\frac{n \\cdot p - n_i^2}{(n + n_i) \\cdot (p + n_i)} \\cdot \\left( \\frac{|\\mathbf{E}|}{E_0} \\right)^{\\sigma} \\cdot \\exp \\left(-\\frac{B}{|\\mathbf{E}|} \\cdot \\left( \\frac{E_g}{E_{g, 300}} \\right)^{3/2} \\right)

    where :math:`A`, :math:`B`, :math:`E_0`, and :math:`\\sigma` are material-dependent parameters.

    Example
    -------
    >>> import tidy3d as td
    >>> Si = td.HurkxDirectBandToBandTunneling(
    ...   A=4e14,
    ...   B=1.9e7,
    ...   E_0=1,
    ...   sigma=2.5
    ... )

    References
    ----------
        .. [1] Palankovski, Vassil, and Rüdiger Quay. Analysis and simulation of heterostructure devices. Springer Science & Business Media, 2004.
        .. [2] Hurkx, G. A. M., D. B. M. Klaassen, and M. P. G. Knuvers. "A new
               recombination model for device simulation including tunneling."
               IEEE Transactions on Electron Devices 39.2 (1992): 331-338.
    """

    A: PositiveFloat = Field(
        default=4e14,
        title="Parameter :math:`A`",
        description="Parameter :math:`A` in the direct BTBT Hurkx model.",
        json_schema_extra={"units": "1/(cm^3 s)"},
    )
    B: float = Field(
        default=1.9e6,
        title="Parameter :math:`B`",
        description="Parameter :math:`B` in the direct BTBT Hurkx model. The default differs "
        "from the silicon value 1.9e7 V/cm of Hurkx et al. [2].",
        json_schema_extra={"units": "V/cm"},
    )
    E_0: PositiveFloat = Field(
        default=1,
        title="Reference electric field :math:`E_0`",
        description="Reference electric field :math:`E_0` in the direct BTBT Hurkx model.",
        json_schema_extra={"units": "V/cm"},
    )
    sigma: float = Field(
        default=2.5,
        title="Exponent parameter",
        description="Exponent :math:`\\sigma` in the direct BTBT Hurkx model. For direct "
        "semiconductors :math:`\\sigma` is typically 2.0, while for indirect "
        "semiconductors :math:`\\sigma` is typically 2.5.",
    )


class SurfaceShockleyReedHallRecombination(Tidy3dBaseModel):
    """
    Surface Shockley-Read-Hall (SRH) recombination at a single trap level.

    Notes
    -----

        Extended SRH form evaluated per unit area at a semiconductor interface:

        .. math::

           R_s = \\frac{n p - n_{i,\\mathrm{eff}}^2}
                       {(n + n_1)/S_p + (p + p_1)/S_n}

        with :math:`n_1 = n_{i,\\mathrm{eff}}\\,e^{(E_t - E_F^i)/kT}` and
        :math:`p_1 = n_{i,\\mathrm{eff}}\\,e^{(E_F^i - E_t)/kT}`, where
        ``E_t`` is the trap level relative to the intrinsic Fermi level
        :math:`E_F^i`.  The default ``E_t = 0`` (mid-gap) is the
        velocity-form shorthand used in most low-injection passivation
        analyses, for which :math:`n_1 = p_1 = n_{i,\\mathrm{eff}}`.

        Conventions follow Altermatt 2011 (J. Comput. Electron. 10:314):
        :math:`n_{i,\\mathrm{eff}}` is the bandgap-narrowing-corrected
        effective intrinsic concentration, and the same value enters both the
        numerator and the trap-level concentrations :math:`n_1, p_1`.

        Typical surface recombination velocities for crystalline silicon:

        * ideal Al2O3 / SiNx passivation: ``S < 10`` cm/s
        * thermal SiO2: ``S ~ 10 - 100`` cm/s
        * bare Si surface (no passivation): ``S ~ 1e6`` cm/s

    Example
    -------
        >>> import tidy3d as td
        >>> # Si/SiO2 interface, mid-gap traps (default), moderate passivation
        >>> sr = td.SurfaceShockleyReedHallRecombination(S_n=1e3, S_p=1e3)
        >>> # Same interface with explicit near-mid-gap trap energy
        >>> sr = td.SurfaceShockleyReedHallRecombination(S_n=1e3, S_p=1e3, E_t=0.05)

    References
    ----------
        .. [1] P. P. Altermatt, "Models for numerical device simulations of
               crystalline silicon solar cells - a review," J. Comput. Electron.
               10:314 (2011).
    """

    S_n: NonNegativeFloat = Field(
        title="Electron surface recombination velocity",
        description="Electron surface recombination velocity at the interface. "
        "Zero corresponds to ideal passivation on the electron channel.",
        json_schema_extra={"units": "cm/s"},
    )
    S_p: NonNegativeFloat = Field(
        title="Hole surface recombination velocity",
        description="Hole surface recombination velocity at the interface. "
        "Zero corresponds to ideal passivation on the hole channel.",
        json_schema_extra={"units": "cm/s"},
    )
    E_t: float = Field(
        default=0.0,
        title="Trap level relative to intrinsic Fermi",
        description="Energy of the interface trap level relative to the "
        "intrinsic Fermi level. ``0`` corresponds to mid-gap (the most common "
        "case); positive values are above intrinsic. Must lie inside the "
        "band gap for the recombination rate to be physical.",
        json_schema_extra={"units": "eV"},
    )
    field_enhancement: HurkxTrapAssistedTunneling | None = Field(
        default=None,
        title="Trap-assisted tunneling field enhancement",
        description="Optional :class:`HurkxTrapAssistedTunneling` model. When set, the "
        "surface recombination velocities are field-enhanced by trap-assisted "
        "tunneling, :math:`S \\rightarrow S (1 + \\Gamma)`, with :math:`\\Gamma` "
        "evaluated per carrier from the local electric field. It does not apply on a "
        "surface that also carries a :class:`VoltageBC`, "
        "where the recombination velocities describe carrier exchange with the metal "
        "rather than capture by a trap. Leaving it ``None`` gives the unenhanced "
        "surface SRH rate. Requires the accelerated charge solver.",
    )


class HeterojunctionCaptureSide(Tidy3dBaseModel):
    """
    One side of a heterojunction interface trap: the material it captures
    carriers from, and the capture velocities from that side.

    Example
    -------
        >>> import tidy3d as td
        >>> side = td.HeterojunctionCaptureSide(name="inp", S_n=1e3, S_p=1e3)
    """

    name: str = Field(
        title="Side name",
        description="Name of the structure on this side of the heterojunction when the "
        "condition is placed with a :class:`.StructureStructureInterface`, or of its "
        "medium when placed with a :class:`.MediumMediumInterface`.",
    )
    S_n: NonNegativeFloat = Field(
        title="Electron capture velocity",
        description="Velocity at which the interface trap captures electrons from this "
        "side. Zero closes the electron channel to this side.",
        json_schema_extra={"units": "cm/s"},
    )
    S_p: NonNegativeFloat = Field(
        title="Hole capture velocity",
        description="Velocity at which the interface trap captures holes from this "
        "side. Zero closes the hole channel to this side.",
        json_schema_extra={"units": "cm/s"},
    )


class HeterojunctionSurfaceShockleyReedHallRecombination(Tidy3dBaseModel):
    """
    Shockley-Read-Hall (SRH) recombination through a single interface trap level
    shared by the two sides of a heterojunction.

    Notes
    -----

        One trap level at energy ``E_t`` exchanges carriers with both sides of
        the interface [1]_ [2]_. For side :math:`k`, with interface carrier
        densities :math:`n_k` and :math:`p_k` and trap occupancy :math:`f`,
        the net electron and hole capture rates per unit area are

        .. math::

           U_{n,k} = S_{n,k} \\left[ n_k (1 - f) - n_{1,k} f \\right],
           \\qquad
           U_{p,k} = S_{p,k} \\left[ p_k f - p_{1,k} (1 - f) \\right].

        In steady state the trap captures as many electrons as holes,
        :math:`\\sum_k U_{n,k} = \\sum_k U_{p,k}`, which fixes :math:`f` and
        gives the net recombination rate

        .. math::

           R_s = \\frac{A_n A_p - B_n B_p}{A_n + B_n + A_p + B_p}

        with

        .. math::

           A_n = \\sum_k S_{n,k} n_k, \\quad
           B_n = \\sum_k S_{n,k} n_{1,k}, \\quad
           A_p = \\sum_k S_{p,k} p_k, \\quad
           B_p = \\sum_k S_{p,k} p_{1,k}.

        ``E_t`` is measured from the intrinsic Fermi level of the side named by
        ``E_t_reference``. The trap-level concentrations :math:`n_{1,k}` and
        :math:`p_{1,k}` of each side follow from that one trap energy through
        the side's own band edges, so the rate vanishes in equilibrium. Under
        Fermi-Dirac statistics they carry each side's degeneracy factor.

        Each side captures at its own full velocities; a side with zero
        velocities drops out, leaving one-sided SRH. With two identical
        materials and equal velocities on both sides, the rate under Boltzmann
        statistics is therefore twice that of
        :class:`SurfaceShockleyReedHallRecombination` at the same velocities on a
        same-material junction: one measured interface velocity ``S`` corresponds
        to ``S / 2`` on each side. At a type-II (staggered) interface the trap
        recombines electrons of one side with holes of the other.

        Place the condition with a :class:`.StructureStructureInterface` or a
        :class:`.MediumMediumInterface` whose two names are the ``sides``
        names; their order does not matter. The two sides must be
        :class:`.SemiconductorMedium` materials that differ in ``N_c``, ``N_v``,
        ``E_g`` or ``electron_affinity``; doping alone does not make a
        heterojunction. Free surfaces, semiconductor/insulator interfaces and
        junctions between regions of the same material take
        :class:`SurfaceShockleyReedHallRecombination` instead. The model has no
        field enhancement and cannot share a placement with a :class:`VoltageBC`.

    Note
    ----
    This model is supported only by the accelerated charge solver.

    Example
    -------
        >>> import tidy3d as td
        >>> sr = td.HeterojunctionSurfaceShockleyReedHallRecombination(
        ...     E_t=0.1,
        ...     E_t_reference="inp",
        ...     sides=(
        ...         td.HeterojunctionCaptureSide(name="inp", S_n=1e3, S_p=1e3),
        ...         td.HeterojunctionCaptureSide(name="ingaas", S_n=4e2, S_p=1e3),
        ...     ),
        ... )

    References
    ----------
        .. [1] H. J. Pauwels and G. Vanhoutte, "The influence of interface state
               and energy barriers on the efficiency of heterojunction solar
               cells," J. Phys. D: Appl. Phys. 11, 649 (1978).
               DOI: 10.1088/0022-3727/11/5/009.
        .. [2] R. Stangl, C. Leendertz, and J. Haschke, "Numerical simulation of
               solar cells and solar cell characterization methods: the
               open-source on demand program AFORS-HET," in Solar Energy,
               R. D. Rugescu (ed.), InTech (2010), Sec. 4.3.3.
               DOI: 10.5772/8073.
    """

    E_t: float = Field(
        default=0.0,
        title="Trap level relative to intrinsic Fermi",
        description="Energy of the interface trap level relative to the intrinsic "
        "Fermi level of the side named by ``E_t_reference``. ``0`` corresponds to "
        "mid-gap of that side; positive values are above it.",
        json_schema_extra={"units": "eV"},
    )
    E_t_reference: str = Field(
        title="Trap level reference side",
        description="Name of the side, one of the ``sides`` names, whose intrinsic "
        "Fermi level ``E_t`` is measured from.",
    )
    sides: tuple[HeterojunctionCaptureSide, HeterojunctionCaptureSide] = Field(
        title="Capture sides",
        description="The two sides of the heterojunction, each named and with the "
        "velocities at which the interface trap captures carriers from it. The order "
        "does not matter.",
    )


SurfaceRecombinationModelType = (
    SurfaceShockleyReedHallRecombination | HeterojunctionSurfaceShockleyReedHallRecombination
)


class SelberherrImpactIonization(Tidy3dBaseModel):
    """
    This class defines the parameters for the Selberherr impact ionization model [1]_.
    Three formulations are available, distinguished by the driving force
    :math:`F_\\nu` for carrier :math:`\\nu`: ``"Selberherr"`` uses the
    conduction-band-edge gradient :math:`\\|\\nabla E_c\\|`, ``"PQ"`` projects it
    on the carrier-current direction,
    :math:`\\nabla E_c\\cdot\\mathbf J_\\nu/\\|\\mathbf J_\\nu\\|` [2]_, and
    ``"GradQuasiFermi"`` uses the quasi-Fermi-energy gradient
    :math:`\\|\\nabla E_{F\\nu}\\|`. Every gradient is taken within a single
    material, so a band offset across a heterojunction does not act as a
    driving force.

    Notes
    -----

        The impact ionization rate ``\\alpha_{\\nu}`` (for :math:`\\nu = p` (holes) and :math:`\\nu = n` (electrons)) is defined by:

        .. math::

            \\alpha_{\\nu} = \\alpha_{\\nu}^\\infty \\cdot \\exp \\left( - \\left( \\frac{E_{\\nu}^{\\text{crit}}}{F_\\nu} \\right)^{\\beta_{\\nu}} \\right)

        where :math:`\\alpha_{\\nu}^\\infty`,
        :math:`E_{\\nu}^{\\text{crit}}`, and :math:`\\beta_{\\nu}` are
        material-dependent parameters and :math:`\\mathbf J_\\nu` is the carrier
        current density. With ``"GradQuasiFermi"``, :math:`E_{F\\nu}` is
        evaluated at a carrier density of at least
        :math:`10^{8}\\,\\text{cm}^{-3}` and held constant below it: the
        quasi-Fermi level of a fully depleted region is not resolvable there and
        carries no ionization. This formulation is supported only for isothermal
        charge simulations using the accelerated solver.

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.SelberherrImpactIonization(
        ...   alpha_n_inf=7.03e5,
        ...   alpha_p_inf=1.582e6,
        ...   E_n_crit=1.23e6,
        ...   E_p_crit=2.03e6,
        ...   beta_n=1,
        ...   beta_p=1,
        ...   formulation='PQ'
        ... )

    References
    ----------
        .. [1] Selberherr, Siegfried. Analysis and simulation of semiconductor devices. Springer Science & Business Media, 1984.
        .. [2] Vassil Palankovski and Rüdiger Quay. Analysis and simulation of heterostructure devices. Springer Science & Business Media, 2004.
    """

    alpha_n_inf: PositiveFloat = Field(
        title="Electron ionization coefficient at infinite field",
        description="Electron ionization coefficient at infinite field.",
        json_schema_extra={"units": "1/cm"},
    )
    alpha_p_inf: PositiveFloat = Field(
        title="Hole ionization coefficient at infinite field",
        description="Hole ionization coefficient at infinite field.",
        json_schema_extra={"units": "1/cm"},
    )
    E_n_crit: PositiveFloat = Field(
        title="Critical electric field for electrons",
        description="Critical electric field for electrons.",
        json_schema_extra={"units": "V/cm"},
    )
    E_p_crit: PositiveFloat = Field(
        ...,
        title="Critical electric field for holes",
        description="Critical electric field for holes.",
        json_schema_extra={"units": "V/cm"},
    )
    beta_n: PositiveFloat = Field(
        title="Exponent for electrons",
        description="Exponent for electrons.",
    )
    beta_p: PositiveFloat = Field(
        title="Exponent for holes",
        description="Exponent for holes.",
    )

    formulation: Literal["Selberherr", "PQ", "GradQuasiFermi"] = Field(
        default="PQ",
        title="Formulation",
        description="Driving-force formulation used for impact ionization. "
        "'Selberherr' uses the conduction-band-edge gradient magnitude, 'PQ' "
        "projects that gradient onto the carrier-current direction, and "
        "'GradQuasiFermi' uses the quasi-Fermi-energy gradient magnitude.",
    )
