from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.constants import PERCMCUBE, SECOND
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.compat import Self


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

    @model_validator(mode="after")
    def _warn_if_A_not_one(self: Self) -> Self:
        """Warn if ``A`` is set to a value other than 1.

        All published parameterizations of this lifetime form (Fossum,
        Roulston, Klaassen, Schenk) take ``A = 1``; other values have no
        physical interpretation in the literature.
        """
        if self.A != 1:
            log.warning(
                f"'FossumCarrierLifetime.A' is set to {self.A}, but A=1 is "
                "the only value consistent with the published "
                "parameterizations of this model. Other values have no "
                "physical interpretation."
            )
        return self


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
        1.0,
        title="Doping exponent",
        description="Dimensionless exponent :math:`\\gamma` of the doping-dependence factor.",
    )

    alpha_T: float = Field(
        -1.5,
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

    @model_validator(mode="after")
    def check_spatialdataarray_dimensions(self) -> Self:
        """Check that the SpatialDataArray is at least 2D:"""

        rate = self.rate

        zero_dims = [d for d in ["x", "y", "z"] if len(rate.coords[d]) <= 1]

        if len(zero_dims) > 1:
            raise ValueError("SpatialDataArray must be at least 2D.")

        return self


class HurkxDirectBandToBandTunneling(Tidy3dBaseModel):
    """
    This class defines a direct band-to-band tunneling recombination model based on the Hurkx model.

    Notes
    -----

    The model is described in [1]_.

    The direct band-to-band tunneling recombination rate :math:`R^{\\text{BTBT}}` is primarily defined by the
    material's bandgap energy :math:`E_g` and the electric field :math:`F`.

    Default values are provided for silicon.

    .. math::

        R^{\\text{BTBT}} = A \\cdot \\frac{n \\cdot p - n_i^2}{(n + n_i) \\cdot (p + n_i)} \\cdot \\left( \\frac{|\\mathbf{E}|}{E_0} \\right)^{\\sigma} \\cdot \\exp \\left(-\\frac{B}{|\\mathbf{E}|} \\cdot \\left( \\frac{E_g}{E_{g, 300}} \\right)^{3/2} \\right)

    where :math:`A`, :math:`B`, :math:`E_0`, and :math:`\\sigma` are material-dependent parameters.

    Example
    -------
    >>> import tidy3d as td
    >>> default_Si = td.HurkxDirectBandToBandTunneling(
    ...   A=1e19,
    ...   B=1.9e6,
    ...   E_0=1,
    ...   sigma=2.5
    ... )

    References
    ----------
        .. [1] Palankovski, Vassil, and Rüdiger Quay. Analysis and simulation of heterostructure devices. Springer Science & Business Media, 2004.
    """

    A: PositiveFloat = Field(
        4e14,
        title="Parameter :math:`A`",
        description="Parameter :math:`A` in the direct BTBT Hurkx model.",
        json_schema_extra={"units": "1/(cm^3 s)"},
    )
    B: float = Field(
        1.9e6,
        title="Parameter :math:`B`",
        description="Parameter :math:`B` in the direct BTBT Hurkx model.",
        json_schema_extra={"units": "V/cm"},
    )
    E_0: PositiveFloat = Field(
        1,
        title="Reference electric field :math:`E_0`",
        description="Reference electric field :math:`E_0` in the direct BTBT Hurkx model.",
        json_schema_extra={"units": "V/cm"},
    )
    sigma: float = Field(
        2.5,
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
        0.0,
        title="Trap level relative to intrinsic Fermi",
        description="Energy of the interface trap level relative to the "
        "intrinsic Fermi level. ``0`` corresponds to mid-gap (the most common "
        "case); positive values are above intrinsic. Must lie inside the "
        "band gap for the recombination rate to be physical.",
        json_schema_extra={"units": "eV"},
    )

    @model_validator(mode="after")
    def _validate_trap_level_in_gap(self) -> Self:
        """Warn if the trap energy is outside any physically reasonable
        band gap. ``E_t`` is referenced to the intrinsic Fermi level in
        eV, so values much larger than half a typical bandgap are almost
        certainly a unit mistake (eV vs. Joules) or a sign convention
        error, producing unphysical ``n_1``, ``p_1`` values."""
        if abs(self.E_t) > 1.5:
            log.warning(
                f"SurfaceShockleyReedHallRecombination 'E_t' = {self.E_t:.3e} eV "
                "is outside the typical semiconductor band gap (|E_t| > 1.5 eV). "
                "'E_t' is referenced to the intrinsic Fermi level in eV; "
                "double-check units and sign convention."
            )
        return self

    @model_validator(mode="after")
    def _validate_at_least_one_velocity_positive(self) -> Self:
        """Warn if both surface recombination velocities are zero. The
        BC then contributes no recombination flux and reduces to an
        electrostatic interface; use this only for a charged
        ``SurfaceRecombinationBC`` with no carrier recombination flux.
        Otherwise prefer ``InsulatingBC``."""
        if self.S_n == 0.0 and self.S_p == 0.0:
            log.warning(
                "SurfaceShockleyReedHallRecombination has S_n = S_p = 0 and "
                "will not contribute any recombination flux. Use this only "
                "for a charged 'SurfaceRecombinationBC' with no carrier "
                "recombination flux; otherwise prefer 'InsulatingBC'."
            )
        return self


SurfaceRecombinationModelType = SurfaceShockleyReedHallRecombination


class SelberherrImpactIonization(Tidy3dBaseModel):
    """
    This class defines the parameters for the Selberherr impact ionization model. Two formulations are available that
    depend on the driving field, as described in [1]_ (:math:`\\| E \\|`) and [2]_ (:math:`E \\cdot J_{\\nu} / \\| E \\|` for :math:`\\nu = n,p`).

    Notes
    -----

        The impact ionization rate ``\\alpha_{\\nu}`` (for :math:`\\nu = p` (holes) and :math:`\\nu = n` (electrons)) is defined by:

        .. math::

            \\alpha_{\\nu} = \\alpha_{\\nu}^\\infty \\cdot \\exp \\left( - \\left( \\frac{E_{\\nu}^{\\text{crit}} \\cdot |\\mathbf{J}_{\\nu}|}{E \\cdot \\mathbf{J}_{\\nu}} \\right)^{\\beta_{\\nu}} \\right)

        where :math:`\\alpha_{\\nu}^\\infty`, :math:`E_{\\nu}^{\\text{crit}}`, and :math:`\\beta_{\\nu}` are material-dependent parameters.

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

    formulation: Literal["Selberherr", "PQ"] = Field(
        "PQ",
        title="Formulation",
        description="Formulation used for impact ionization. Options are 'Selberherr' "
        "or 'PQ' for Selberherr and Palankovski and Quay formulations, respectively.",
    )
