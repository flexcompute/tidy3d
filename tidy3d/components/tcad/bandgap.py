from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import Field, NonNegativeFloat, PositiveFloat

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import ELECTRON_VOLT, PERCMCUBE, VOLT

TemperatureType = float | Sequence[float] | np.ndarray


# Energy band-gap models
class ConstantEnergyBandGap(Tidy3dBaseModel):
    """Constant (temperature-independent) energy band gap.

    Example
    -------
    >>> import tidy3d as td
    >>> band_gap = td.ConstantEnergyBandGap(eg=1.11)
    >>> band_gap.band_gap_energy(temperature=400.0)
    1.11
    """

    eg: PositiveFloat = Field(
        title="Band Gap",
        description="Energy band gap",
        json_schema_extra={"units": ELECTRON_VOLT},
    )

    def band_gap_energy(self, temperature: TemperatureType) -> float | np.ndarray:
        """Energy band gap in eV at the requested temperature(s) in Kelvin."""
        temperature = np.asarray(temperature, dtype=float)
        eg = np.full(np.shape(temperature), self.eg, dtype=float)
        return eg.item() if eg.ndim == 0 else eg


class VarshniEnergyBandGap(Tidy3dBaseModel):
    """
    Models the temperature dependence of the energy band gap (Eg)
    using the Varshni formula.

    Notes
    -----
    See [1]_ for the original formulation.

    The model implements the following formula:

    .. math::

        E_g(T) = E_g(0) - \\frac{\\alpha T^2}{T + \\beta}

    Example
    -------
    >>> # Parameters for Silicon (Si)
    >>> si_model = VarshniEnergyBandGap(
    ...     eg_0=1.17,
    ...     alpha=4.73e-4,
    ...     beta=636.0,
    ... )
    >>> round(si_model.band_gap_energy(temperature=300.0), 4)
    1.1245

    References
    ----------

        .. [1] Varshni, Y. P. (1967). Temperature dependence of the energy gap in semiconductors. Physica, 34(1), 149-154.

    """

    eg_0: PositiveFloat = Field(
        title="Band Gap at 0 K",
        description="Energy band gap at absolute zero (0 Kelvin).",
        json_schema_extra={"units": ELECTRON_VOLT},
    )

    alpha: PositiveFloat = Field(
        title="Varshni Alpha Coefficient",
        description="Empirical Varshni coefficient (α).",
        json_schema_extra={"units": "eV/K"},
    )

    beta: PositiveFloat = Field(
        title="Varshni Beta Coefficient",
        description="Empirical Varshni coefficient (β), related to the Debye temperature.",
        json_schema_extra={"units": "K"},
    )

    def band_gap_energy(self, temperature: TemperatureType) -> float | np.ndarray:
        """Energy band gap in eV at the requested temperature(s) in Kelvin."""
        temperature = np.asarray(temperature, dtype=float)
        eg = self.eg_0 - self.alpha * temperature**2 / (temperature + self.beta)
        return eg.item() if eg.ndim == 0 else eg


# Band-gap narrowing models
class SlotboomBandGapNarrowing(Tidy3dBaseModel):
    """
    Parameters for the Slotboom model for band-gap narrowing.

    Notes
    ------
        The Slotboom band-gap narrowing :math:`\\Delta E_G` model is discussed in [1]_ as follows:

        .. math::

            \\Delta E_G = V_{1,bgn} \\left( \\ln \\left( \\frac{N_{tot}}{N_{2,bgn}} \\right)
            + \\sqrt{\\left( \\ln \\left( \\frac{N_{tot}}{N_{2,bgn}} \\right) \\right)^2 + C_{2,bgn}} \\right)
            \\quad \\text{if} \\quad N_{tot} \\geq 10^{15} \\text{cm}^{-3},

            \\Delta E_G = 0 \\quad \\text{if} \\quad N_{tot} < 10^{15} \\text{cm}^{-3}.

        Note that :math:`N_{tot}` is the total doping as defined within a :class:`SemiconductorMedium`.

        This is an *apparent* band-gap narrowing: its parameters fit the measured
        :math:`n_0 p_0 = n_i^2 \\exp(\\Delta E_G / k_B T)` of heavily doped silicon read with
        Boltzmann statistics, so majority-carrier degeneracy is already part of
        :math:`\\Delta E_G`. With ``fermi_dirac=True`` the solver uses the larger Fermi-Dirac
        narrowing :math:`\\Delta E_G^{FD} \\ge \\Delta E_G`, so degeneracy is not counted twice [2]_:

        .. math::

            \\Delta E_G^{FD} = \\Delta E_G - k_B T \\ln \\frac{F_{1/2}(\\eta)}{e^{\\eta}}, \\qquad
            \\eta = F_{1/2}^{-1} \\left( \\frac{|N_D - N_A|}{N_{maj}} \\right),

        where :math:`F_{1/2}` is the normalized Fermi-Dirac integral and :math:`N_{maj}` is
        :math:`N_c` where :math:`N_D > N_A` and :math:`N_v` otherwise. :math:`T` is the local lattice
        temperature, so both statistics give the same :math:`n_0 p_0` in a neutral region at every
        temperature.

        The fit also takes the majority-carrier density to be :math:`|N_D - N_A|`, which assumes
        fully ionized dopants. Where the medium sets ``donor_ionization`` or
        ``acceptor_ionization``, the neutral majority-carrier density :math:`n_{maj}` is lower,
        and under either statistics the solver uses

        .. math::

            \\Delta E_G' = \\Delta E_G + k_B T \\left(\\eta_{maj}
            - \\ln \\frac{|N_D - N_A|}{N_{maj}}\\right),

        with :math:`\\eta_{maj}` the reduced Fermi level of :math:`n_{maj}`, that is
        :math:`F_{1/2}^{-1}(n_{maj}/N_{maj})` with Fermi-Dirac statistics and
        :math:`\\ln(n_{maj}/N_{maj})` with Boltzmann statistics. The minority-carrier density of a
        neutral region is then the one the narrowing was fitted to. At full ionization this is
        the expression above.

        .. [1] 'UNIFIED APPARENT BANDGAP NARROWING IN n- AND p-TYPE SILICON' Solid-State Electronics Vol. 35, No. 2, pp. 125-129, 1992

        .. [2] D. Yan and A. Cuevas, 'Empirical determination of the energy band gap narrowing in highly doped n+ silicon', Journal of Applied Physics 114, 044508, 2013

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.SlotboomBandGapNarrowing(
        ...    v1=6.92 * 1e-3,
        ...    n2=1.3e17,
        ...    c2=0.5,
        ...    min_N=1e15,
        ... )
    """

    v1: PositiveFloat = Field(
        title=":math:`V_{1,bgn}` parameter",
        description=":math:`V_{1,bgn}` parameter",
        json_schema_extra={"units": VOLT},
    )

    n2: PositiveFloat = Field(
        title=":math:`N_{2,bgn}` parameter",
        description=":math:`N_{2,bgn}` parameter",
        json_schema_extra={"units": PERCMCUBE},
    )

    c2: float = Field(
        title=":math:`C_{2,bgn}` parameter",
        description=":math:`C_{2,bgn}` parameter",
    )

    min_N: NonNegativeFloat = Field(
        title="Minimum total doping",
        description="Bandgap narrowing is applied at location where total doping "
        "is higher than ``min_N``.",
        json_schema_extra={"units": PERCMCUBE},
    )
