"""Parametric material models."""
from abc import ABC, abstractmethod
from typing import List, Tuple
import warnings
import pydantic as pd
import numpy as np

from ..components.medium import PoleResidue, Medium2D, Drude
from ..components.base import Tidy3dBaseModel
from ..constants import EPSILON_0, Q_e, HBAR, K_B, ELECTRON_VOLT, KELVIN
from ..log import log

try:
    from scipy import integrate

    INTEGRATE_AVAILABLE = True
except ImportError:
    INTEGRATE_AVAILABLE = False

# default values of the physical parameters for graphene
# scattering rate in eV
GRAPHENE_DEF_GAMMA = 0.00041
# chemical potential in eV
GRAPHENE_DEF_MU_C = 0
# temperature in K
GRAPHENE_DEF_TEMP = 300

# constants controlling the numerical integration of the interband term in graphene
# frequency limits of integration
GRAPHENE_INT_MIN = 1e10
GRAPHENE_INT_MAX = 1e16
# integration absolute tolerance
GRAPHENE_INT_TOL = 1e-20

# constants controlling the Pade approximation of the interband term in graphene
# frequency range for fitting
GRAPHENE_FIT_FREQ_MIN = 1e12
GRAPHENE_FIT_FREQ_MAX = 1e15
GRAPHENE_FIT_NUM_FREQS = 100
GRAPHENE_FIT_ATOL = 2e-5
# parameters controlling node placement
GRAPHENE_FIT_LARGE_MULTIPLIER = 10
GRAPHENE_FIT_SMALL_MULTIPLIER = 0.25
# number of optimization iterations for fitting
GRAPHENE_FIT_NUM_ITERS = 100


class ParametricVariantItem2D(ABC, Tidy3dBaseModel):
    """A variant of a 2D material depending on parameters, that must be initialized in order to
    generate the material model."""

    @property
    @abstractmethod
    def medium(self) -> Medium2D:
        """Calculate the material model at the current parameter values."""


class Graphene(ParametricVariantItem2D):
    """Parametric surface conductivity model for graphene.

    Note
    ----
    The model contains intraband and interband terms, as described in::

        George W. Hanson, "Dyadic Green's Functions for an Anisotropic,
        Non-Local Model of Biased Graphene," IEEE Trans. Antennas Propag.
        56, 3, 747-757 (2008).

    Example
    -------
    >>> graphene_medium = Graphene(mu_c = 0.2).medium

    """

    mu_c: float = pd.Field(
        GRAPHENE_DEF_MU_C,
        title="Chemical potential in eV",
        description="Chemical potential in eV.",
        units=ELECTRON_VOLT,
    )
    temp: float = pd.Field(
        GRAPHENE_DEF_TEMP, title="Temperature in K", description="Temperature in K.", units=KELVIN
    )
    gamma: float = pd.Field(
        GRAPHENE_DEF_GAMMA,
        title="Scattering rate in eV",
        description="Scattering rate in eV. Must be small compared to the optical frequency.",
        units=ELECTRON_VOLT,
    )
    scaling: float = pd.Field(
        1,
        title="Scaling factor",
        description="Scaling factor used to model multiple layers of graphene.",
    )

    include_interband: bool = pd.Field(
        True,
        title="Include interband terms",
        description="Include interband terms, relevant at high frequency (IR). "
        "Otherwise, the intraband terms only give a simpler Drude-type model relevant "
        "only at low frequency (THz).",
    )
    interband_fit_freq_nodes: List[Tuple[float, float]] = pd.Field(
        None,
        title="Interband fitting frequency nodes",
        description="Frequency nodes for fitting interband term. "
        "Each pair of nodes in the list corresponds to a single Pade approximant of order "
        "(1, 2), which is optimized to minimize the error at these two frequencies. "
        "The default behavior is to fit a first approximant at one very low frequency and "
        "one very high frequency, and to fit a second approximant in the vicinity of the "
        "interband feature. This default behavior works for a wide range "
        "of frequencies; consider changing the nodes to obtain a better fit for a "
        "narrow-band simulation.",
    )
    interband_fit_num_iters: pd.NonNegativeInt = pd.Field(
        GRAPHENE_FIT_NUM_ITERS,
        title="Interband fitting number of iterations",
        description="Number of iterations for optimizing each Pade approximant when "
        "fitting the interband term. Making this larger might give a better fit "
        "at the cost of decreased stability in the fitting algorithm.",
    )

    @property
    def medium(self) -> Medium2D:
        """Surface conductivity model for graphene."""
        intraband = self.intraband_drude
        if self.include_interband:
            interband = self.interband_pole_residue
            intraband_poles = intraband.pole_residue.poles
            interband_poles = interband.pole_residue.poles
            poles = intraband_poles + interband_poles
            pole_residue = self._filter_poles(
                PoleResidue(
                    poles=poles, frequency_range=(GRAPHENE_FIT_FREQ_MIN, GRAPHENE_FIT_FREQ_MAX)
                )
            )
            return Medium2D(ss=pole_residue, tt=pole_residue)
        return Medium2D(ss=intraband, tt=intraband)

    @property
    def intraband_drude(self) -> Drude:
        """A Drude-type model for the intraband term of graphene.

        Returns
        -------
        :class:`.Drude`
            A Drude-type model for the intraband term of graphene.

        """
        factor1 = Q_e * K_B * self.temp / (HBAR**2 * 4 * np.pi**3 * EPSILON_0)
        factor2 = self.mu_c / (K_B * self.temp) + 2 * np.log(
            np.exp(-self.mu_c / (K_B * self.temp)) + 1
        )
        intra_f1 = np.sqrt(self.scaling * factor1 * factor2)
        intra_delta1 = (1 / HBAR) * self.gamma / np.pi
        return Drude(coeffs=[(intra_f1, intra_delta1)])

    @property
    def interband_pole_residue(self) -> PoleResidue:
        """A pole-residue model for the interband term of graphene. Note that this does not
        include the intraband term, which is added in separately.

        Returns
        -------
        :class:`.PoleResidue`
            A pole-residue model for the interband term of graphene.
        """
        mu_c = self.mu_c / (2 * np.pi * HBAR)
        temp = self.temp * K_B / (2 * np.pi * HBAR)
        resonance = max(np.sqrt(abs(mu_c**2 - temp**2)), GRAPHENE_FIT_FREQ_MIN * 1e-5)
        freqs = np.linspace(
            resonance / GRAPHENE_FIT_LARGE_MULTIPLIER,
            resonance * GRAPHENE_FIT_LARGE_MULTIPLIER,
            GRAPHENE_FIT_NUM_FREQS,
        )
        sigma = self.interband_conductivity(freqs)
        nodes = self.interband_fit_freq_nodes
        if nodes is None:
            fcenter = freqs[np.argmin(np.imag(sigma))]
            fwidth = (
                fcenter - freqs[np.nonzero(np.imag(sigma) < 0.5 * np.amin(np.imag(sigma)))[0][0]]
            )
            nodes = [
                (fcenter / GRAPHENE_FIT_LARGE_MULTIPLIER, fcenter * GRAPHENE_FIT_LARGE_MULTIPLIER),
                (fcenter, fcenter + fwidth * GRAPHENE_FIT_SMALL_MULTIPLIER),
            ]

        flattened_freqs = [freq for pair in nodes for freq in pair]
        sigma_inds = self.interband_conductivity(flattened_freqs)
        inds = [(2 * i, 2 * i + 1) for i in range(len(nodes))]

        pole_residue = self._fit_interband_conductivity(flattened_freqs, sigma_inds, inds)
        pole_residue_filtered = self._filter_poles(pole_residue)
        sigma_fit = pole_residue_filtered.sigma_model(freqs)
        if not np.allclose(sigma, sigma_fit, rtol=0, atol=GRAPHENE_FIT_ATOL):
            log.warning(
                "Graphene fit may not be good. Try changing the physical or fitting parameters."
            )
        return pole_residue_filtered

    def numerical_conductivity(self, freqs: List[float]) -> List[complex]:
        """Numerically calculate the conductivity. If this differs from the
        conductivity of the :class:`.Medium2D`, it is due to error while
        fitting the interband term, and you may try values of ``interband_fit_freq_nodes``
        different from its default (calculated) value.

        Parameters
        ----------
        freqs : List[float]
            The list of frequencies.

        Returns
        -------
        List[complex]
            The list of corresponding conductivities, in S.
        """
        intra_sigma = self.intraband_drude.sigma_model(freqs)
        inter_sigma = self.interband_conductivity(freqs)
        return intra_sigma + inter_sigma

    def interband_conductivity(self, freqs: List[float]) -> List[complex]:
        """Numerically integrate interband term.

        Parameters
        ----------
        freqs : List[float]
            The list of frequencies.

        Returns
        -------
        List[complex]
            The list of corresponding interband conductivities, in S.
        """

        def fermi(E: float) -> float:
            """Fermi distribution."""
            # catch overflow warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return 1 / (np.exp((E - self.mu_c) / (K_B * self.temp)) + 1)

        def fermi_g(E: float) -> float:
            """Difference of fermi distributions."""
            return fermi(-E) - fermi(E)

        def integrand(E: float, omega: float) -> float:
            """Integrand for interband term."""
            return (fermi_g(E * HBAR) - fermi_g(HBAR * omega / 2)) / (omega**2 - 4 * E**2)

        if not INTEGRATE_AVAILABLE:
            raise ImportError(
                "The package 'scipy' was not found. Please install the 'core' "
                "dependencies to calculate the interband term of graphene. For example: "
                "pip install -r requirements/core.txt"
            )

        omegas = 2 * np.pi * np.array(freqs)
        sigma = np.zeros(len(omegas), dtype=complex)
        integration_min = GRAPHENE_INT_MIN
        integration_max = GRAPHENE_INT_MAX
        for i, omega in enumerate(omegas):

            integral, _ = integrate.quad(
                integrand, integration_min, integration_max, args=(omega,), epsabs=GRAPHENE_INT_TOL
            )
            sigma[i] = (Q_e / (4 * HBAR)) * (
                fermi_g(HBAR * omega / 2) - (4 * omega / (1j * np.pi)) * integral
            )

        return self.scaling * sigma

    def _fit_interband_conductivity(  # pylint: disable=too-many-locals
        self,
        freqs: List[float],
        sigma: List[complex],
        indslist: List[Tuple[int, int]],
    ):
        """Fit the interband conductivity with a Pade approximation, as described in

        Stamatios Amanatiadis, Theodoros Zygiridis, Tadao Ohtani, Yasushi Kanai,
        and Nikolaos Kantartzis, "A Consistent Scheme for the Precise FDTD Modeling
        of the Graphene Interband Contribution," IEEE Trans. Magn. 57, 6 (2021).

        Parameters
        ----------
        freqs : List[float]
            The input frequencies.
        sigma : List[complex]
            The interband conductivity to fit.
        indslist : List[Tuple[int, int]]
            The indices at which to sample the data for fitting.
            The length of this list determines the number of Pade terms used.
        Returns
        -------
        :class:`.PoleResidue`
            A pole-residue model approximating the interband conductivity.
        """

        def evaluate_coeffslist(omega: List[float], coeffslist: List[List[float]]) -> List[float]:
            """Evaluate the Pade approximants given by ``coeffslist` to ``omega``.
            Each item in ``coeffslist`` is a list of four coefficients corresponding to
            a single Pade term."""
            res = np.zeros(len(omega), dtype=complex)
            for (alpha0, alpha1, beta1, beta2) in coeffslist:
                res += (alpha0 + alpha1 * 1j * omega) / (
                    1 + beta1 * 1j * omega + beta2 * (1j * omega) ** 2
                )
            return res

        def fit_single(
            omega: List[float], sigma: List[complex], inds: Tuple[int, int]
        ) -> List[float]:
            """Fit a single Pade approximant of degree (1, 2) to ``sigma``
            as a real function of i ``omega``. The method is described in

            Adam Mock, "Pade approximant spectral fit for FDTD
            simulation of graphene in the near infrared,"
            Opt. Mater. Express 2, 6, pp. 771-781 (2012).
            """
            gamma = [np.real(sigma[i]) for i in inds]
            eta = [np.imag(sigma[i]) for i in inds]
            omegas = [omega[i] for i in inds]
            matrix = np.array(
                [
                    [1, 0, omegas[0] * eta[0], omegas[0] ** 2 * gamma[0]],
                    [0, omegas[0], -omegas[0] * gamma[0], omegas[0] ** 2 * eta[0]],
                    [1, 0, omegas[1] * eta[1], omegas[1] ** 2 * gamma[1]],
                    [0, omegas[1], -omegas[1] * gamma[1], omegas[1] ** 2 * eta[1]],
                ]
            )
            return np.linalg.pinv(matrix) @ np.array([gamma[0], eta[0], gamma[1], eta[1]])

        def optimize(
            omega: List[float],
            sigma: List[complex],
            indslist: List[Tuple[int, int]],
            coeffslist: List[List[float]],
        ) -> List[float]:
            """Optimize the coefficients in ``coeffslist`` by sampling ``omega`` and ``sigma``
            at the indices in ``indslist``."""
            for _ in range(self.interband_fit_num_iters):
                old_coeffslist = coeffslist
                res = sigma - evaluate_coeffslist(omega, old_coeffslist)
                for j, coeffs in enumerate(old_coeffslist):
                    curr_res = res + evaluate_coeffslist(omega, [coeffs])
                    coeffslist[j] = fit_single(omega, curr_res, indslist[j])
            return coeffslist

        def get_pole_residue(coeffslist: List[List[float]]) -> PoleResidue:
            """Convert a list of Pade coefficients into a :class:`.PoleResidue` model."""
            poles = []
            for (alpha0, alpha1, beta1, beta2) in coeffslist:
                disc = beta1**2 - 4 * beta2
                root1 = (beta1 + np.sqrt(complex(disc))) / (2 * beta2)
                root2 = (beta1 - np.sqrt(complex(disc))) / (2 * beta2)
                res1 = (alpha0 - alpha1 * root1) / (beta2 * (root2 - root1))
                res2 = alpha1 / beta2 - res1

                if disc > 0:
                    for (root, res) in zip([root1, root2], [res1, res2]):
                        poles.append((root, res / 2))
                else:
                    poles.append((root1, res1))

            flipped_poles = []
            for (a, c) in poles:
                if np.real(a) > 0:
                    flipped_poles += [(-1j * np.conj(1j * a), c)]
                else:
                    flipped_poles += [(a, c)]
            return PoleResidue(
                poles=flipped_poles, frequency_range=(GRAPHENE_FIT_FREQ_MIN, GRAPHENE_FIT_FREQ_MAX)
            )

        # fitting works better with normalized quantities (THz and uS)
        omega_thz = 2 * np.pi * np.array(freqs) * 1e-12
        sigma_us = np.array(sigma) * 1e6
        coeffslist = []
        for inds in indslist:
            res = np.array(sigma_us) - evaluate_coeffslist(omega_thz, coeffslist)
            coeffslist.append(fit_single(omega_thz, res, inds))

        coeffslist = optimize(omega_thz, sigma_us, indslist, coeffslist)

        pole_res = get_pole_residue(coeffslist)
        # unnormalize, and convert from conductivity to permittivity
        poles = [(a * 1e12, -c / (a * EPSILON_0 * 1e6)) for (a, c) in pole_res.poles]

        return PoleResidue(
            poles=poles, frequency_range=(GRAPHENE_FIT_FREQ_MIN, GRAPHENE_FIT_FREQ_MAX)
        )

    def _filter_poles(self, medium: PoleResidue) -> PoleResidue:
        """Clean up poles, merging poles at zero frequency."""
        zero_res = 0
        poles = []
        for (a, c) in medium.poles:
            if a == 0:
                zero_res += c
            elif abs(a) > 1e17:
                continue
            else:
                poles += [(a, c)]
        return PoleResidue(
            poles=poles + [(0, zero_res)],
            frequency_range=(GRAPHENE_FIT_FREQ_MIN, GRAPHENE_FIT_FREQ_MAX),
        )
