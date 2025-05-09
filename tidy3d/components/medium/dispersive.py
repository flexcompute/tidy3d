from __future__ import annotations

import warnings
from math import isclose
from typing import Dict, Tuple, Union

import autograd as ag
import autograd.numpy as np
import numpy as npo
import pydantic.v1 as pd

from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.autograd.types import (
    AutogradFieldMap,
    TracedPoleAndResidue,
    TracedPositiveFloat,
)
from tidy3d.components.base import cached_property, skip_if_fields_missing
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.data.utils import (
    CustomSpatialDataType,
    CustomSpatialDataTypeAnnotated,
    _check_same_coordinates,
    _get_numpy_array,
    _ones_like,
    _zeros_like,
)
from tidy3d.components.data.validators import validate_no_nans
from tidy3d.components.dispersion_fitter import (
    LOSS_CHECK_MAX,
    LOSS_CHECK_MIN,
    LOSS_CHECK_NUM,
    imag_resp_extrema_locs,
)
from tidy3d.components.grid.grid import Coords
from tidy3d.components.types import ArrayComplex3D, ArrayFloat1D, Bound, Complex, PoleAndResidue
from tidy3d.constants import (
    C_0,
    EPSILON_0,
    HERTZ,
    MICROMETER,
    PERMITTIVITY,
    RADPERSEC,
    SECOND,
    fp_eps,
)
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

from .base import AbstractMedium
from .dispersionless import CustomMedium, Medium
from .dispersive_abc import CustomDispersiveMedium, DispersiveMedium
from .utils import ensure_freq_in_range


class PoleResidue(DispersiveMedium):
    """A dispersive medium described by the pole-residue pair model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(\\omega) = \\epsilon_\\infty - \\sum_i
            \\left[\\frac{c_i}{j \\omega + a_i} +
            \\frac{c_i^*}{j \\omega + a_i^*}\\right]

    Example
    -------
    >>> pole_res = PoleResidue(eps_inf=2.0, poles=[((-1+2j), (3+4j)), ((-5+6j), (7+8j))])
    >>> eps = pole_res.eps_model(200e12)

    See Also
    --------

    :class:`CustomPoleResidue`:
        A spatially varying dispersive medium described by the pole-residue pair model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: TracedPositiveFloat = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    poles: Tuple[TracedPoleAndResidue, ...] = pd.Field(
        (),
        title="Poles",
        description="Tuple of complex-valued (:math:`a_i, c_i`) poles for the model.",
        units=(RADPERSEC, RADPERSEC),
    )

    @pd.validator("poles", always=True)
    def _causality_validation(cls, val):
        """Assert causal medium."""
        for a, _ in val:
            if np.any(np.real(_get_numpy_array(a)) > 0):
                raise SetupError("For stable medium, 'Re(a_i)' must be non-positive.")
        return val

    _validate_permittivity_modulation = DispersiveMedium._permittivity_modulation_validation()
    _validate_conductivity_modulation = DispersiveMedium._conductivity_modulation_validation()

    @staticmethod
    def _eps_model(
        eps_inf: pd.PositiveFloat, poles: Tuple[PoleAndResidue, ...], frequency: float
    ) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        omega = 2 * np.pi * frequency
        eps = eps_inf + 0 * frequency + 0.0j
        for a, c in poles:
            a_cc = np.conj(a)
            c_cc = np.conj(c)
            eps = eps - c / (1j * omega + a)
            eps = eps - c_cc / (1j * omega + a_cc)
        return eps

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""
        return self._eps_model(eps_inf=self.eps_inf, poles=self.poles, frequency=frequency)

    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model."""

        return dict(
            eps_inf=self.eps_inf,
            poles=self.poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )

    def __str__(self):
        """string representation"""
        return (
            f"td.PoleResidue("
            f"\n\teps_inf={self.eps_inf}, "
            f"\n\tpoles={self.poles}, "
            f"\n\tfrequency_range={self.frequency_range})"
        )

    @classmethod
    def from_medium(cls, medium: Medium) -> PoleResidue:
        """Convert a :class:`.Medium` to a pole residue model.

        Parameters
        ----------
        medium: :class:`.Medium`
            The medium with permittivity and conductivity to convert.

        Returns
        -------
        :class:`.PoleResidue`
            The pole residue equivalent.
        """
        poles = [(0, medium.conductivity / (2 * EPSILON_0))]
        return PoleResidue(
            eps_inf=medium.permittivity, poles=poles, frequency_range=medium.frequency_range
        )

    def to_medium(self) -> Medium:
        """Convert to a :class:`.Medium`.
        Requires the pole residue model to only have a pole at 0 frequency,
        corresponding to a constant conductivity term.

        Returns
        -------
        :class:`.Medium`
            The non-dispersive equivalent with constant permittivity and conductivity.
        """
        res = 0
        for a, c in self.poles:
            if abs(a) > fp_eps:
                raise ValidationError("Cannot convert dispersive 'PoleResidue' to 'Medium'.")
            res = res + (c + np.conj(c)) / 2
        sigma = res * 2 * EPSILON_0
        return Medium(
            permittivity=self.eps_inf,
            conductivity=np.real(sigma),
            frequency_range=self.frequency_range,
        )

    @staticmethod
    def lo_to_eps_model(
        poles: Tuple[Tuple[float, float, float, float], ...],
        eps_inf: pd.PositiveFloat,
        frequency: float,
    ) -> complex:
        """Complex permittivity as a function of frequency for a given set of LO-TO coefficients.
        See ``from_lo_to`` in :class:`.PoleResidue` for the detailed form of the model
        and a reference paper.

        Parameters
        ----------
        poles : Tuple[Tuple[float, float, float, float], ...]
            The LO-TO poles, given as list of tuples of the form
            (omega_LO, gamma_LO, omega_TO, gamma_TO).
        eps_inf: pd.PositiveFloat
            The relative permittivity at infinite frequency.
        frequency: float
            Frequency at which to evaluate the permittivity.

        Returns
        -------
        complex
            The complex permittivity of the given LO-TO model at the given frequency.
        """
        omega = 2 * np.pi * frequency
        eps = eps_inf
        for omega_lo, gamma_lo, omega_to, gamma_to in poles:
            eps *= omega_lo**2 - omega**2 - 1j * omega * gamma_lo
            eps /= omega_to**2 - omega**2 - 1j * omega * gamma_to
        return eps

    @classmethod
    def from_lo_to(
        cls, poles: Tuple[Tuple[float, float, float, float], ...], eps_inf: pd.PositiveFloat = 1
    ) -> PoleResidue:
        """Construct a pole residue model from the LO-TO form
        (longitudinal and transverse optical modes).
        The LO-TO form is :math:`\\epsilon_\\infty \\prod_{i=1}^l \\frac{\\omega_{LO, i}^2 - \\omega^2 - i \\omega \\gamma_{LO, i}}{\\omega_{TO, i}^2 - \\omega^2 - i \\omega \\gamma_{TO, i}}` as given in the paper:

            M. Schubert, T. E. Tiwald, and C. M. Herzinger,
            "Infrared dielectric anisotropy and phonon modes of sapphire,"
            Phys. Rev. B 61, 8187 (2000).

        Parameters
        ----------
        poles : Tuple[Tuple[float, float, float, float], ...]
            The LO-TO poles, given as list of tuples of the form
            (omega_LO, gamma_LO, omega_TO, gamma_TO).
        eps_inf: pd.PositiveFloat
            The relative permittivity at infinite frequency.

        Returns
        -------
        :class:`.PoleResidue`
            The pole residue equivalent of the LO-TO form provided.
        """

        omegas_lo, gammas_lo, omegas_to, gammas_to = map(np.array, zip(*poles))

        # discriminants of quadratic factors of denominator
        discs = 2 * npo.emath.sqrt((gammas_to / 2) ** 2 - omegas_to**2)

        # require nondegenerate TO poles
        if len({(omega_to, gamma_to) for (_, _, omega_to, gamma_to) in poles}) != len(poles) or any(
            disc == 0 for disc in discs
        ):
            raise ValidationError(
                "Unable to construct a pole residue model "
                "from an LO-TO form with degenerate TO poles. Consider adding a "
                "perturbation to split the poles, or using "
                "'PoleResidue.lo_to_eps_model' and fitting with the 'FastDispersionFitter'."
            )

        # roots of denominator, in pairs
        roots = []
        for gamma_to, disc in zip(gammas_to, discs):
            roots.append(-gamma_to / 2 + disc / 2)
            roots.append(-gamma_to / 2 - disc / 2)

        # interpolants
        interpolants = eps_inf * np.ones(len(roots), dtype=complex)
        for i, a in enumerate(roots):
            for omega_lo, gamma_lo in zip(omegas_lo, gammas_lo):
                interpolants[i] *= omega_lo**2 + a**2 + a * gamma_lo
            for j, a2 in enumerate(roots):
                if j != i:
                    interpolants[i] /= a - a2

        a_coeffs = []
        c_coeffs = []

        for i in range(0, len(roots), 2):
            if not np.isreal(roots[i]):
                a_coeffs.append(roots[i])
                c_coeffs.append(interpolants[i])
            else:
                a_coeffs.append(roots[i])
                a_coeffs.append(roots[i + 1])
                # factor of two from adding conjugate pole of real pole
                c_coeffs.append(interpolants[i] / 2)
                c_coeffs.append(interpolants[i + 1] / 2)

        return PoleResidue(eps_inf=eps_inf, poles=list(zip(a_coeffs, c_coeffs)))

    @staticmethod
    def imag_ep_extrema(poles: Tuple[PoleAndResidue, ...]) -> ArrayFloat1D:
        """Extrema of Im[eps] in the same unit as poles.

        Parameters
        ----------
        poles: Tuple[PoleAndResidue, ...]
            Tuple of complex-valued (``a_i, c_i``) poles for the model.
        """

        poles_a = [a for (a, _) in poles]
        poles_c = [c for (_, c) in poles]
        return imag_resp_extrema_locs(poles=poles_a, residues=poles_c)

    def _imag_ep_extrema_with_samples(self) -> ArrayFloat1D:
        """Provide a list of frequencies (in unit of rad/s) to probe the possible lower and
        upper bound of Im[eps] within the ``frequency_range``. If ``frequency_range`` is None,
        it checks the entire frequency range. The returned frequencies include not only extrema,
        but also a list of sampled frequencies.
        """

        # extrema frequencies: in the intermediate stage, convert to the unit eV for
        # better numerical handling, since those quantities will be ~ 1 in photonics
        extrema_freq = self.imag_ep_extrema(self.angular_freq_to_eV(np.array(self.poles)))
        extrema_freq = self.eV_to_angular_freq(extrema_freq)

        # let's check a big range in addition to the imag_extrema
        if self.frequency_range is None:
            range_ev = np.logspace(LOSS_CHECK_MIN, LOSS_CHECK_MAX, LOSS_CHECK_NUM)
            range_omega = self.eV_to_angular_freq(range_ev)
        else:
            fmin, fmax = self.frequency_range
            fmin = max(fmin, fp_eps)
            range_freq = np.logspace(np.log10(fmin), np.log10(fmax), LOSS_CHECK_NUM)
            range_omega = self.Hz_to_angular_freq(range_freq)

            extrema_freq = extrema_freq[
                np.logical_and(extrema_freq > range_omega[0], extrema_freq < range_omega[-1])
            ]
        return np.concatenate((range_omega, extrema_freq))

    @cached_property
    def loss_upper_bound(self) -> float:
        """Upper bound of Im[eps] in `frequency_range`"""
        freq_list = self.angular_freq_to_Hz(self._imag_ep_extrema_with_samples())
        ep = self.eps_model(freq_list)
        # filter `NAN` in case some of freq_list are exactly at the pole frequency
        # of Sellmeier-type poles.
        ep = ep[~np.isnan(ep)]
        return max(ep.imag)

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint derivatives for each of the ``fields`` given the multiplied E and D."""

        # compute all derivatives beforehand
        dJ_deps = self.derivative_eps_complex_volume(
            E_der_map=derivative_info.E_der_map,
            bounds=derivative_info.bounds,
            freqs=np.atleast_1d(derivative_info.frequency),
        )

        dJ_deps = complex(dJ_deps)

        # TODO: fix for multi-frequency
        frequency = derivative_info.frequency
        poles_complex = [(complex(a), complex(c)) for a, c in self.poles]
        poles_complex = np.stack(poles_complex, axis=0)

        # compute gradients of eps_model with respect to eps_inf and poles
        grad_eps_model = ag.holomorphic_grad(self._eps_model, argnum=(0, 1))
        with warnings.catch_warnings():
            # ignore warnings about holmorphic grad being passed a non-complex input (poles)
            warnings.simplefilter("ignore")
            deps_deps_inf, deps_dpoles = grad_eps_model(
                complex(self.eps_inf), poles_complex, complex(frequency)
            )

        # multiply with partial dJ/deps to give full gradients

        dJ_deps_inf = dJ_deps * deps_deps_inf
        dJ_dpoles = [(dJ_deps * a, dJ_deps * c) for a, c in deps_dpoles]

        # get vjps w.r.t. permittivity and conductivity of the bulk
        derivative_map = {}
        for field_path in derivative_info.paths:
            field_name, *rest = field_path

            if field_name == "eps_inf":
                derivative_map[field_path] = float(np.real(dJ_deps_inf))

            elif field_name == "poles":
                pole_index, a_or_c = rest
                derivative_map[field_path] = complex(dJ_dpoles[pole_index][a_or_c])

        return derivative_map

    @classmethod
    def _real_partial_fraction_decomposition(
        cls, a: np.ndarray, b: np.ndarray, tol: pd.PositiveFloat = 1e-2
    ) -> tuple[list[tuple[Complex, Complex]], np.ndarray]:
        """Computes the complex conjugate pole residue pairs given a rational expression with
        real coefficients.

        Parameters
        ----------

        a : np.ndarray
            Coefficients of the numerator polynomial in increasing monomial order.
        b : np.ndarray
            Coefficients of the denominator polynomial in increasing monomial order.
        tol : pd.PositiveFloat
            Tolerance for pole finding. Two poles are considered equal, if their spacing is less
            than ``tol``.

        Returns
        -------
        tuple[list[tuple[Complex, Complex]], np.ndarray]
            The list of complex conjugate poles and their associated residues. The second element of the
            ``tuple`` is an array of coefficients representing any direct polynomial term.

        """
        from scipy import signal

        if a.ndim != 1 or np.any(np.iscomplex(a)):
            raise ValidationError(
                "Numerator coefficients must be a one-dimensional array of real numbers."
            )
        if b.ndim != 1 or np.any(np.iscomplex(b)):
            raise ValidationError(
                "Denominator coefficients must be a one-dimensional array of real numbers."
            )

        # Compute residues and poles using scipy
        (r, p, k) = signal.residue(np.flip(a), np.flip(b), tol=tol, rtype="avg")

        # Assuming real coefficients for the polynomials, the poles should be real or come as
        # complex conjugate pairs
        r_filtered = []
        p_filtered = []
        for res, (idx, pole) in zip(list(r), enumerate(list(p))):
            # Residue equal to zero interpreted as rational expression was not
            # in simplest form. So skip this pole.
            if res == 0:
                continue
            # Causal and stability check
            if np.real(pole) > 0:
                raise ValidationError("Transfer function is invalid. It is non-causal.")
            # Check for higher order pole, which come in consecutive order
            if idx > 0 and p[idx - 1] == pole:
                raise ValidationError(
                    "Transfer function is invalid. A higher order pole was detected. Try reducing ``tol``, "
                    "or ensure that the rational expression does not have repeated poles. "
                )
            if np.imag(pole) == 0:
                r_filtered.append(res / 2)
                p_filtered.append(pole)
            else:
                pair_found = len(np.argwhere(np.array(p) == np.conj(pole))) == 1
                if not pair_found:
                    raise ValueError(
                        "Failed to find complex-conjugate of pole in poles computed by SciPy."
                    )
                previously_added = len(np.argwhere(np.array(p_filtered) == np.conj(pole))) == 1
                if not previously_added:
                    r_filtered.append(res)
                    p_filtered.append(pole)

        poles_residues = list(zip(p_filtered, r_filtered))
        k_increasing_order = np.flip(k)
        return (poles_residues, k_increasing_order)

    @classmethod
    def from_admittance_coeffs(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        eps_inf: pd.PositiveFloat = 1,
        pole_tol: pd.PositiveFloat = 1e-2,
    ) -> PoleResidue:
        """Construct a :class:`.PoleResidue` model from an admittance function defining the
        relationship between the electric field and the polarization current density in the
        Laplace domain.

        Parameters
        ----------
        a : np.ndarray
            Coefficients of the numerator polynomial in increasing monomial order.
        b : np.ndarray
            Coefficients of the denominator polynomial in increasing monomial order.
        eps_inf: pd.PositiveFloat
            The relative permittivity at infinite frequency.
        pole_tol: pd.PositiveFloat
            Tolerance for the pole finding algorithm in Hertz. Two poles are considered equal, if their
            spacing is closer than ``pole_tol`.
        Returns
        -------
        :class:`.PoleResidue`
            The pole residue equivalent.

        Notes
        -----

            The supplied admittance function relates the electric field to the polarization current density
            in the Laplace domain and is equivalent to a frequency-dependent complex conductivity
            :math:`\\sigma(\\omega)`.

            .. math::
                J_p(s) = Y(s)E(s)

            .. math::
                Y(s) = \\frac{a_0 + a_1 s + \\dots + a_M s^M}{b_0 + b_1 s + \\dots + b_N s^N}

            An equivalent :class:`.PoleResidue` medium is constructed using an equivalent frequency-dependent
            complex permittivity defined as

            .. math::
                \\epsilon(s) = \\epsilon_\\infty - \\frac{1}{\\epsilon_0 s}
                \\frac{a_0 + a_1 s + \\dots + a_M s^M}{b_0 + b_1 s + \\dots + b_N s^N}.
        """

        if a.ndim != 1 or np.any(np.logical_or(np.iscomplex(a), a < 0)):
            raise ValidationError(
                "Numerator coefficients must be a one-dimensional array of non-negative real numbers."
            )
        if b.ndim != 1 or np.any(np.logical_or(np.iscomplex(b), b < 0)):
            raise ValidationError(
                "Denominator coefficients must be a one-dimensional array of non-negative real numbers."
            )

        # Trim any trailing zeros, so that length corresponds with polynomial order
        a = np.trim_zeros(a, "b")
        b = np.trim_zeros(b, "b")

        # Validate that transfer function will result in a proper transfer function, once converted to
        # the complex permittivity version
        # Let q equal the order of the numerator polynomial, and p equal the order
        # of the denominator polynomal. Then, q < p is strictly proper rational transfer function (RTF)
        # q <= p is a proper RTF, and q > p is an improper RTF.
        q = len(a) - 1
        p = len(b) - 1

        if q > p + 1:
            raise ValidationError(
                "Transfer function is improper, the order of the numerator polynomial must be at most "
                "one greater than the order of the denominator polynomial."
            )

        # Modify the transfer function defining a complex conductivity to match the complex
        # frequency-dependent portion of the pole residue model
        # Meaning divide by -j*omega*epsilon (s*epsilon)
        b = np.concatenate(([0], b * EPSILON_0))

        poles_and_residues, k = cls._real_partial_fraction_decomposition(
            a=a, b=b, tol=pole_tol * 2 * np.pi
        )

        # A direct polynomial term of zeroth order is interpreted as an additional contribution to eps_inf.
        # So we only handle that special case.
        if len(k) == 1:
            if np.iscomplex(k[0]) or k[0] < 0:
                raise ValidationError(
                    "Transfer function is invalid. Direct polynomial term must be real and positive for "
                    "conversion to an equivalent 'PoleResidue' medium."
                )
            else:
                # A pure capacitance will translate to an increased permittivity at infinite frequency.
                eps_inf = eps_inf + k[0]

        pole_residue_from_transfer = PoleResidue(eps_inf=eps_inf, poles=poles_and_residues)

        # Check passivity
        ang_freqs = PoleResidue._imag_ep_extrema_with_samples(pole_residue_from_transfer)
        freq_list = PoleResidue.angular_freq_to_Hz(ang_freqs)
        ep = pole_residue_from_transfer.eps_model(freq_list)
        # filter `NAN` in case some of freq_list are exactly at the pole frequency
        ep = ep[~np.isnan(ep)]

        if np.any(np.imag(ep) < -fp_eps):
            log.warning(
                "Generated 'PoleResidue' medium is not passive. Please raise an issue on the "
                "Tidy3d frontend with this message and some information about your "
                "simulation setup and we will investigate."
            )

        return pole_residue_from_transfer


class CustomPoleResidue(CustomDispersiveMedium, PoleResidue):
    """A spatially varying dispersive medium described by the pole-residue pair model.

    Notes
    -----

        In this method, the frequency-dependent permittivity :math:`\\epsilon(\\omega)` is expressed as a sum of
        resonant material poles _`[1]`.

        .. math::

            \\epsilon(\\omega) = \\epsilon_\\infty - \\sum_i
            \\left[\\frac{c_i}{j \\omega + a_i} +
            \\frac{c_i^*}{j \\omega + a_i^*}\\right]

        For each of these resonant poles identified by the index :math:`i`, an auxiliary differential equation is
        used to relate the auxiliary current :math:`J_i(t)` to the applied electric field :math:`E(t)`.
        The sum of all these auxiliary current contributions describes the total dielectric response of the material.

        .. math::

            \\frac{d}{dt} J_i (t) - a_i J_i (t) = \\epsilon_0 c_i \\frac{d}{dt} E (t)

        Hence, the computational cost increases with the number of poles.

        **References**

        .. [1]   M. Han, R.W. Dutton and S. Fan, IEEE Microwave and Wireless Component Letters, 16, 119 (2006).

        .. TODO add links to notebooks using this.

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(np.ones((5, 6, 7)), coords=coords)
    >>> a1 = SpatialDataArray(-np.random.random((5, 6, 7)), coords=coords)
    >>> c1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> a2 = SpatialDataArray(-np.random.random((5, 6, 7)), coords=coords)
    >>> c2 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> pole_res = CustomPoleResidue(eps_inf=eps_inf, poles=[(a1, c1), (a2, c2)])
    >>> eps = pole_res.eps_model(200e12)

    See Also
    --------

    **Notebooks**

    * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**

    * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = pd.Field(
        ...,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    poles: Tuple[Tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        pd.Field(
            (),
            title="Poles",
            description="Tuple of complex-valued (:math:`a_i, c_i`) poles for the model.",
            units=(RADPERSEC, RADPERSEC),
        )
    )

    _no_nans_eps_inf = validate_no_nans("eps_inf")
    _no_nans_poles = validate_no_nans("poles")
    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("poles")

    @pd.validator("eps_inf", always=True)
    def _eps_inf_positive(cls, val):
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @pd.validator("poles", always=True)
    @skip_if_fields_missing(["eps_inf"])
    def _poles_correct_shape(cls, val, values):
        """poles must have the same shape."""

        for coeffs in val:
            for coeff in coeffs:
                if not _check_same_coordinates(coeff, values["eps_inf"]):
                    raise SetupError(
                        "All pole coefficients 'a' and 'c' must have the same coordinates; "
                        "The coordinates must also be consistent with 'eps_inf'."
                    )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False

        for coeffs in self.poles:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        eps = PoleResidue.eps_model(self, frequency)
        return (eps, eps, eps)

    def poles_on_grid(self, coords: Coords) -> Tuple[Tuple[ArrayComplex3D, ArrayComplex3D], ...]:
        """Spatial profile of poles interpolated at the supplied coordinates.

        Parameters
        ----------
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        Tuple[Tuple[ArrayComplex3D, ArrayComplex3D], ...]
            The poles interpolated at the supplied coordinate.
        """

        def fun_interp(input_data: SpatialDataArray) -> ArrayComplex3D:
            return _get_numpy_array(coords.spatial_interp(input_data, self.interp_method))

        return tuple((fun_interp(a), fun_interp(c)) for (a, c) in self.poles)

    @classmethod
    def from_medium(cls, medium: CustomMedium) -> CustomPoleResidue:
        """Convert a :class:`.CustomMedium` to a pole residue model.

        Parameters
        ----------
        medium: :class:`.CustomMedium`
            The medium with permittivity and conductivity to convert.

        Returns
        -------
        :class:`.CustomPoleResidue`
            The pole residue equivalent.
        """
        poles = [(_zeros_like(medium.conductivity), medium.conductivity / (2 * EPSILON_0))]
        medium_dict = medium.dict(exclude={"type", "eps_dataset", "permittivity", "conductivity"})
        medium_dict.update({"eps_inf": medium.permittivity, "poles": poles})
        return CustomPoleResidue.parse_obj(medium_dict)

    def to_medium(self) -> CustomMedium:
        """Convert to a :class:`.CustomMedium`.
        Requires the pole residue model to only have a pole at 0 frequency,
        corresponding to a constant conductivity term.

        Returns
        -------
        :class:`.CustomMedium`
            The non-dispersive equivalent with constant permittivity and conductivity.
        """
        res = 0
        for a, c in self.poles:
            if np.any(abs(_get_numpy_array(a)) > fp_eps):
                raise ValidationError(
                    "Cannot convert dispersive 'CustomPoleResidue' to 'CustomMedium'."
                )
            res = res + (c + np.conj(c)) / 2
        sigma = res * 2 * EPSILON_0

        self_dict = self.dict(exclude={"type", "eps_inf", "poles"})
        self_dict.update({"permittivity": self.eps_inf, "conductivity": np.real(sigma)})
        return CustomMedium.parse_obj(self_dict)

    @cached_property
    def loss_upper_bound(self) -> float:
        """Not implemented yet."""
        raise SetupError("To be implemented.")

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomPoleResidue
            CustomPoleResidue with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("eps_inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        poles_reduced = []
        for pole, residue in self.poles:
            if not pole.does_cover(bounds=bounds):
                log.warning("Pole spatial data array does not fully cover the requested region.")

            if not residue.does_cover(bounds=bounds):
                log.warning("Residue spatial data array does not fully cover the requested region.")

            poles_reduced.append((pole.sel_inside(bounds), residue.sel_inside(bounds)))

        return self.updated_copy(eps_inf=eps_inf_reduced, poles=poles_reduced)

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint derivatives for each of the ``fields`` given the multiplied E and D."""

        dJ_deps = 0.0
        for dim in "xyz":
            dJ_deps += self._derivative_field_cmp(
                E_der_map=derivative_info.E_der_map,
                eps_data=self.eps_inf,
                dim=dim,
                freqs=np.atleast_1d(derivative_info.frequency),
            )

        # TODO: fix for multi-frequency
        frequency = derivative_info.frequency

        poles_complex = [
            (np.array(a.values, dtype=complex), np.array(c.values, dtype=complex))
            for a, c in self.poles
        ]
        poles_complex = np.stack(poles_complex, axis=0)

        def eps_model_r(
            eps_inf: complex, poles: list[tuple[complex, complex]], frequency: float
        ) -> float:
            """Real part of ``eps_model`` evaluated on ``self`` fields."""
            return np.real(self._eps_model(eps_inf, poles, frequency))

        def eps_model_i(
            eps_inf: complex, poles: list[tuple[complex, complex]], frequency: float
        ) -> float:
            """Real part of ``eps_model`` evaluated on ``self`` fields."""
            return np.imag(self._eps_model(eps_inf, poles, frequency))

        # compute the gradients w.r.t. each real and imaginary parts for eps_inf and poles
        grad_eps_model_r = ag.elementwise_grad(eps_model_r, argnum=(0, 1))
        grad_eps_model_i = ag.elementwise_grad(eps_model_i, argnum=(0, 1))
        deps_deps_inf_r, deps_dpoles_r = grad_eps_model_r(
            self.eps_inf.values, poles_complex, frequency
        )
        deps_deps_inf_i, deps_dpoles_i = grad_eps_model_i(
            self.eps_inf.values, poles_complex, frequency
        )

        # multiply with dJ_deps partial derivative to give full gradients

        deps_deps_inf = deps_deps_inf_r + 1j * deps_deps_inf_i
        dJ_deps_inf = dJ_deps * deps_deps_inf / 3.0  # mysterious 3

        dJ_dpoles = []
        for (da_r, dc_r), (da_i, dc_i) in zip(deps_dpoles_r, deps_dpoles_i):
            da = da_r + 1j * da_i
            dc = dc_r + 1j * dc_i
            dJ_da = dJ_deps * da / 2.0  # mysterious 2
            dJ_dc = dJ_deps * dc / 2.0  # mysterious 2
            dJ_dpoles.append((dJ_da, dJ_dc))

        derivative_map = {}
        for field_path in derivative_info.paths:
            field_name, *rest = field_path

            if field_name == "eps_inf":
                derivative_map[field_path] = np.real(dJ_deps_inf)

            elif field_name == "poles":
                pole_index, a_or_c = rest
                derivative_map[field_path] = dJ_dpoles[pole_index][a_or_c]

        return derivative_map


class Sellmeier(DispersiveMedium):
    """A dispersive medium described by the Sellmeier model.

    Notes
    -----

        The frequency-dependence of the refractive index is described by:

        .. math::

            n(\\lambda)^2 = 1 + \\sum_i \\frac{B_i \\lambda^2}{\\lambda^2 - C_i}

        For lossless, weakly dispersive materials, the best way to incorporate the dispersion without doing
        complicated fits and without slowing the simulation down significantly is to provide the value of the
        refractive index dispersion :math:`\\frac{dn}{d\\lambda}` in :meth:`tidy3d.Sellmeier.from_dispersion`. The
        value is assumed to be at the central frequency or wavelength (whichever is provided), and a one-pole model
        for the material is generated.

    Example
    -------
    >>> sellmeier_medium = Sellmeier(coeffs=[(1,2), (3,4)])
    >>> eps = sellmeier_medium.eps_model(200e12)

    See Also
    --------

    :class:`CustomSellmeier`
        A spatially varying dispersive medium described by the Sellmeier model.

    **Notebooks**

    * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**

    * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    coeffs: Tuple[Tuple[float, pd.PositiveFloat], ...] = pd.Field(
        title="Coefficients",
        description="List of Sellmeier (:math:`B_i, C_i`) coefficients.",
        units=(None, MICROMETER + "^2"),
    )

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if `allow_gain` is False."""
        if values.get("allow_gain"):
            return val
        for B, _ in val:
            if B < 0:
                raise ValidationError(
                    "For passive medium, 'B_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    @pd.validator("modulation_spec", always=True)
    def _validate_permittivity_modulation(cls, val):
        """Assert modulated permittivity cannot be <= 0."""

        if val is None or val.permittivity is None:
            return val

        min_eps_inf = 1.0
        if min_eps_inf - val.permittivity.max_modulation <= 0:
            raise ValidationError(
                "The minimum permittivity value with modulation applied was found to be negative."
            )
        return val

    _validate_conductivity_modulation = DispersiveMedium._conductivity_modulation_validation()

    def _n_model(self, frequency: float) -> complex:
        """Complex-valued refractive index as a function of frequency."""

        wvl = C_0 / np.array(frequency)
        wvl2 = wvl**2
        n_squared = 1.0
        for B, C in self.coeffs:
            n_squared = n_squared + B * wvl2 / (wvl2 - C)
        return np.sqrt(n_squared + 0j)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        n = self._n_model(frequency)
        return AbstractMedium.nk_to_eps_complex(n)

    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model"""
        poles = []
        for B, C in self.coeffs:
            beta = 2 * np.pi * C_0 / np.sqrt(C)
            alpha = -0.5 * beta * B
            a = 1j * beta
            c = 1j * alpha
            poles.append((a, c))
        return dict(eps_inf=1, poles=poles, frequency_range=self.frequency_range, name=self.name)

    @staticmethod
    def _from_dispersion_to_coeffs(n: float, freq: float, dn_dwvl: float):
        """Compute Sellmeier coefficients from dispersion."""
        wvl = C_0 / np.array(freq)
        nsqm1 = n**2 - 1
        c_coeff = -(wvl**3) * n * dn_dwvl / (nsqm1 - wvl * n * dn_dwvl)
        b_coeff = (wvl**2 - c_coeff) / wvl**2 * nsqm1
        return [(b_coeff, c_coeff)]

    @classmethod
    def from_dispersion(cls, n: float, freq: float, dn_dwvl: float = 0, **kwargs):
        """Convert ``n`` and wavelength dispersion ``dn_dwvl`` values at frequency ``freq`` to
        a single-pole :class:`Sellmeier` medium.

        Parameters
        ----------
        n : float
            Real part of refractive index. Must be larger than or equal to one.
        dn_dwvl : float = 0
            Derivative of the refractive index with wavelength (1/um). Must be negative.
        freq : float
            Frequency at which ``n`` and ``dn_dwvl`` are sampled.

        Returns
        -------
        :class:`Sellmeier`
            Single-pole Sellmeier medium with the prvoided refractive index and index dispersion
            valuesat at the prvoided frequency.
        """

        if dn_dwvl >= 0:
            raise ValidationError("Dispersion ``dn_dwvl`` must be smaller than zero.")
        if n < 1:
            raise ValidationError("Refractive index ``n`` cannot be smaller than one.")
        return cls(coeffs=cls._from_dispersion_to_coeffs(n, freq, dn_dwvl), **kwargs)


class CustomSellmeier(CustomDispersiveMedium, Sellmeier):
    """A spatially varying dispersive medium described by the Sellmeier model.

    Notes
    -----

        The frequency-dependence of the refractive index is described by:

        .. math::

            n(\\lambda)^2 = 1 + \\sum_i \\frac{B_i \\lambda^2}{\\lambda^2 - C_i}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> b1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> c1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> sellmeier_medium = CustomSellmeier(coeffs=[(b1,c1),])
    >>> eps = sellmeier_medium.eps_model(200e12)

    See Also
    --------

    :class:`Sellmeier`
        A dispersive medium described by the Sellmeier model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    coeffs: Tuple[Tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        pd.Field(
            ...,
            title="Coefficients",
            description="List of Sellmeier (:math:`B_i, C_i`) coefficients.",
            units=(None, MICROMETER + "^2"),
        )
    )

    _no_nans = validate_no_nans("coeffs")

    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @pd.validator("coeffs", always=True)
    def _correct_shape_and_sign(cls, val):
        """every term in coeffs must have the same shape, and B>=0 and C>0."""
        if len(val) == 0:
            return val
        for B, C in val:
            if not _check_same_coordinates(B, val[0][0]) or not _check_same_coordinates(
                C, val[0][0]
            ):
                raise SetupError("Every term in 'coeffs' must have the same coordinates.")
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((B, C)):
                raise SetupError("'B' and 'C' must be real.")
            if np.any(_get_numpy_array(C) <= 0):
                raise SetupError("'C' must be positive.")
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if `allow_gain` is False."""
        if values.get("allow_gain"):
            return val
        for B, _ in val:
            if np.any(_get_numpy_array(B) < 0):
                raise ValidationError(
                    "For passive medium, 'B_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model."""
        poles_dict = Sellmeier._pole_residue_dict(self)
        if len(self.coeffs) > 0:
            poles_dict.update({"eps_inf": _ones_like(self.coeffs[0][0])})
        return poles_dict

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        eps = Sellmeier.eps_model(self, frequency)
        # if `eps` is simply a float, convert it to a SpatialDataArray ; this is possible when
        # `coeffs` is empty.
        if isinstance(eps, (int, float, complex)):
            eps = SpatialDataArray(eps * np.ones((1, 1, 1)), coords=dict(x=[0], y=[0], z=[0]))
        return (eps, eps, eps)

    @classmethod
    def from_dispersion(
        cls,
        n: CustomSpatialDataType,
        freq: float,
        dn_dwvl: CustomSpatialDataType,
        interp_method="nearest",
        **kwargs,
    ):
        """Convert ``n`` and wavelength dispersion ``dn_dwvl`` values at frequency ``freq`` to
        a single-pole :class:`CustomSellmeier` medium.

        Parameters
        ----------
        n : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Real part of refractive index. Must be larger than or equal to one.
        dn_dwvl : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Derivative of the refractive index with wavelength (1/um). Must be negative.
        freq : float
            Frequency at which ``n`` and ``dn_dwvl`` are sampled.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain permittivity values that are not supplied
            at the Yee grids.

        Returns
        -------
        :class:`.CustomSellmeier`
            Single-pole Sellmeier medium with the prvoided refractive index and index dispersion
            valuesat at the prvoided frequency.
        """

        if not _check_same_coordinates(n, dn_dwvl):
            raise ValidationError("'n' and'dn_dwvl' must have the same dimension.")
        if np.any(_get_numpy_array(dn_dwvl) >= 0):
            raise ValidationError("Dispersion ``dn_dwvl`` must be smaller than zero.")
        if np.any(_get_numpy_array(n) < 1):
            raise ValidationError("Refractive index ``n`` cannot be smaller than one.")
        return cls(
            coeffs=cls._from_dispersion_to_coeffs(n, freq, dn_dwvl),
            interp_method=interp_method,
            **kwargs,
        )

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomSellmeier
            CustomSellmeier with reduced data.
        """
        coeffs_reduced = []
        for b_coeff, c_coeff in self.coeffs:
            if not b_coeff.does_cover(bounds=bounds):
                log.warning(
                    "Sellmeier B coeff spatial data array does not fully cover the requested region."
                )

            if not c_coeff.does_cover(bounds=bounds):
                log.warning(
                    "Sellmeier C coeff spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append((b_coeff.sel_inside(bounds), c_coeff.sel_inside(bounds)))

        return self.updated_copy(coeffs=coeffs_reduced)


class Lorentz(DispersiveMedium):
    """A dispersive medium described by the Lorentz model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty + \\sum_i
            \\frac{\\Delta\\epsilon_i f_i^2}{f_i^2 - 2jf\\delta_i - f^2}

    Example
    -------
    >>> lorentz_medium = Lorentz(eps_inf=2.0, coeffs=[(1,2,3), (4,5,6)])
    >>> eps = lorentz_medium.eps_model(200e12)

    See Also
    --------

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: pd.PositiveFloat = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[float, float, pd.NonNegativeFloat], ...] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, f_i, \\delta_i`) values for model.",
        units=(PERMITTIVITY, HERTZ, HERTZ),
    )

    @pd.validator("coeffs", always=True)
    def _coeffs_unequal_f_delta(cls, val):
        """f**2 and delta**2 cannot be exactly the same."""
        for _, f, delta in val:
            if f**2 == delta**2:
                raise SetupError("'f' and 'delta' cannot take equal values.")
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        if values.get("allow_gain"):
            return val
        for del_ep, _, _ in val:
            if del_ep < 0:
                raise ValidationError(
                    "For passive medium, 'Delta epsilon_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    _validate_permittivity_modulation = DispersiveMedium._permittivity_modulation_validation()
    _validate_conductivity_modulation = DispersiveMedium._conductivity_modulation_validation()

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        eps = self.eps_inf + 0.0j
        for de, f, delta in self.coeffs:
            eps = eps + (de * f**2) / (f**2 - 2j * frequency * delta - frequency**2)
        return eps

    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model."""

        poles = []
        for de, f, delta in self.coeffs:
            w = 2 * np.pi * f
            d = 2 * np.pi * delta

            if self._all_larger(d**2, w**2):
                r = np.sqrt(d * d - w * w) + 0j
                a0 = -d + r
                c0 = de * w**2 / 4 / r
                a1 = -d - r
                c1 = -c0
                poles.extend(((a0, c0), (a1, c1)))
            else:
                r = np.sqrt(w * w - d * d)
                a = -d - 1j * r
                c = 1j * de * w**2 / 2 / r
                poles.append((a, c))

        return dict(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )

    @staticmethod
    def _all_larger(coeff_a, coeff_b) -> bool:
        """``coeff_a`` and ``coeff_b`` can be either float or SpatialDataArray."""
        if isinstance(coeff_a, CustomSpatialDataType.__args__):
            return np.all(_get_numpy_array(coeff_a) > _get_numpy_array(coeff_b))
        return coeff_a > coeff_b

    @classmethod
    def from_nk(cls, n: float, k: float, freq: float, **kwargs):
        """Convert ``n`` and ``k`` values at frequency ``freq`` to a single-pole Lorentz
        medium.

        Parameters
        ----------
        n : float
            Real part of refractive index.
        k : float = 0
            Imaginary part of refrative index.
        freq : float
            Frequency to evaluate permittivity at (Hz).
        kwargs: dict
            Keyword arguments passed to the medium construction.

        Returns
        -------
        :class:`Lorentz`
            Lorentz medium having refractive index n+ik at frequency ``freq``.
        """
        eps_complex = AbstractMedium.nk_to_eps_complex(n, k)
        eps_r, eps_i = eps_complex.real, eps_complex.imag
        if eps_r >= 1:
            log.warning(
                "For 'permittivity>=1', it is more computationally efficient to "
                "use a dispersiveless medium constructed from 'Medium.from_nk()'."
            )
        # first, lossless medium
        if isclose(eps_i, 0):
            if eps_r < 1:
                fp = np.sqrt((eps_r - 1) / (eps_r - 2)) * freq
                return cls(
                    eps_inf=1,
                    coeffs=[
                        (1, fp, 0),
                    ],
                )
            return cls(
                eps_inf=1,
                coeffs=[
                    ((eps_r - 1) / 2, np.sqrt(2) * freq, 0),
                ],
            )
        # lossy medium
        alpha = (eps_r - 1) / eps_i
        delta_p = freq / 2 / (alpha**2 - alpha + 1)
        fp = np.sqrt((alpha**2 + 1) / (alpha**2 - alpha + 1)) * freq
        return cls(
            eps_inf=1,
            coeffs=[
                (eps_i, fp, delta_p),
            ],
            **kwargs,
        )


class CustomLorentz(CustomDispersiveMedium, Lorentz):
    """A spatially varying dispersive medium described by the Lorentz model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty + \\sum_i
            \\frac{\\Delta\\epsilon_i f_i^2}{f_i^2 - 2jf\\delta_i - f^2}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(np.ones((5, 6, 7)), coords=coords)
    >>> d_epsilon = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> f = SpatialDataArray(1+np.random.random((5, 6, 7)), coords=coords)
    >>> delta = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> lorentz_medium = CustomLorentz(eps_inf=eps_inf, coeffs=[(d_epsilon,f,delta),])
    >>> eps = lorentz_medium.eps_model(200e12)

    See Also
    --------

    :class:`CustomPoleResidue`:
        A spatially varying dispersive medium described by the pole-residue pair model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = pd.Field(
        ...,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[
        Tuple[
            CustomSpatialDataTypeAnnotated,
            CustomSpatialDataTypeAnnotated,
            CustomSpatialDataTypeAnnotated,
        ],
        ...,
    ] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, f_i, \\delta_i`) values for model.",
        units=(PERMITTIVITY, HERTZ, HERTZ),
    )

    _no_nans_eps_inf = validate_no_nans("eps_inf")
    _no_nans_coeffs = validate_no_nans("coeffs")

    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @pd.validator("eps_inf", always=True)
    def _eps_inf_positive(cls, val):
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @pd.validator("coeffs", always=True)
    def _coeffs_unequal_f_delta(cls, val):
        """f and delta cannot be exactly the same.
        Not needed for now because we have a more strict
        validator `_coeffs_delta_all_smaller_or_larger_than_fi`.
        """
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["eps_inf"])
    def _coeffs_correct_shape(cls, val, values):
        """coeffs must have consistent shape."""
        for de, f, delta in val:
            if (
                not _check_same_coordinates(de, values["eps_inf"])
                or not _check_same_coordinates(f, values["eps_inf"])
                or not _check_same_coordinates(delta, values["eps_inf"])
            ):
                raise SetupError(
                    "All terms in 'coeffs' must have the same coordinates; "
                    "The coordinates must also be consistent with 'eps_inf'."
                )
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((de, f, delta)):
                raise SetupError("All terms in 'coeffs' must be real.")
        return val

    @pd.validator("coeffs", always=True)
    def _coeffs_delta_all_smaller_or_larger_than_fi(cls, val):
        """We restrict either all f**2>delta**2 or all f**2<delta**2 for now."""
        for _, f, delta in val:
            f2 = f**2
            delta2 = delta**2
            if not (Lorentz._all_larger(f2, delta2) or Lorentz._all_larger(delta2, f2)):
                raise SetupError(
                    "Coefficients in 'coeffs' are restricted to have "
                    "either all 'delta**2'<'f**2' or all 'delta**2'>'f**2'."
                )
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        allow_gain = values.get("allow_gain")
        for del_ep, _, delta in val:
            if np.any(_get_numpy_array(delta) < 0):
                raise ValidationError("For stable medium, 'delta_i' must be non-negative.")
            if not allow_gain and np.any(_get_numpy_array(del_ep) < 0):
                raise ValidationError(
                    "For passive medium, 'Delta epsilon_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        eps = Lorentz.eps_model(self, frequency)
        return (eps, eps, eps)

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomLorentz
            CustomLorentz with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("Eps inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        coeffs_reduced = []
        for de, f, delta in self.coeffs:
            if not de.does_cover(bounds=bounds):
                log.warning(
                    "Lorentz 'de' spatial data array does not fully cover the requested region."
                )

            if not f.does_cover(bounds=bounds):
                log.warning(
                    "Lorentz 'f' spatial data array does not fully cover the requested region."
                )

            if not delta.does_cover(bounds=bounds):
                log.warning(
                    "Lorentz 'delta' spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append(
                (de.sel_inside(bounds), f.sel_inside(bounds), delta.sel_inside(bounds))
            )

        return self.updated_copy(eps_inf=eps_inf_reduced, coeffs=coeffs_reduced)


class Drude(DispersiveMedium):
    """A dispersive medium described by the Drude model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty - \\sum_i
            \\frac{ f_i^2}{f^2 + jf\\delta_i}

    Example
    -------
    >>> drude_medium = Drude(eps_inf=2.0, coeffs=[(1,2), (3,4)])
    >>> eps = drude_medium.eps_model(200e12)

    See Also
    --------

    :class:`CustomDrude`:
        A spatially varying dispersive medium described by the Drude model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: pd.PositiveFloat = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[float, pd.PositiveFloat], ...] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`f_i, \\delta_i`) values for model.",
        units=(HERTZ, HERTZ),
    )

    _validate_permittivity_modulation = DispersiveMedium._permittivity_modulation_validation()
    _validate_conductivity_modulation = DispersiveMedium._conductivity_modulation_validation()

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        eps = self.eps_inf + 0.0j
        for f, delta in self.coeffs:
            eps = eps - (f**2) / (frequency**2 + 1j * frequency * delta)
        return eps

    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model."""

        poles = []

        for f, delta in self.coeffs:
            w = 2 * np.pi * f
            d = 2 * np.pi * delta

            c0 = (w**2) / 2 / d + 0j
            c1 = -c0
            a1 = -d + 0j

            if isinstance(c0, complex):
                a0 = 0j
            else:
                a0 = 0 * c0

            poles.extend(((a0, c0), (a1, c1)))

        return dict(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


class CustomDrude(CustomDispersiveMedium, Drude):
    """A spatially varying dispersive medium described by the Drude model.


    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty - \\sum_i
            \\frac{ f_i^2}{f^2 + jf\\delta_i}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(np.ones((5, 6, 7)), coords=coords)
    >>> f1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> delta1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> drude_medium = CustomDrude(eps_inf=eps_inf, coeffs=[(f1,delta1),])
    >>> eps = drude_medium.eps_model(200e12)

    See Also
    --------

    :class:`Drude`:
        A dispersive medium described by the Drude model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = pd.Field(
        ...,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        pd.Field(
            ...,
            title="Coefficients",
            description="List of (:math:`f_i, \\delta_i`) values for model.",
            units=(HERTZ, HERTZ),
        )
    )

    _no_nans_eps_inf = validate_no_nans("eps_inf")
    _no_nans_coeffs = validate_no_nans("coeffs")

    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @pd.validator("eps_inf", always=True)
    def _eps_inf_positive(cls, val):
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["eps_inf"])
    def _coeffs_correct_shape_and_sign(cls, val, values):
        """coeffs must have consistent shape and sign."""
        for f, delta in val:
            if not _check_same_coordinates(f, values["eps_inf"]) or not _check_same_coordinates(
                delta, values["eps_inf"]
            ):
                raise SetupError(
                    "All terms in 'coeffs' must have the same coordinates; "
                    "The coordinates must also be consistent with 'eps_inf'."
                )
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((f, delta)):
                raise SetupError("All terms in 'coeffs' must be real.")
            if np.any(_get_numpy_array(delta) <= 0):
                raise SetupError("For stable medium, 'delta' must be positive.")
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        eps = Drude.eps_model(self, frequency)
        return (eps, eps, eps)

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomDrude
            CustomDrude with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("Eps inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        coeffs_reduced = []
        for f, delta in self.coeffs:
            if not f.does_cover(bounds=bounds):
                log.warning(
                    "Drude 'f' spatial data array does not fully cover the requested region."
                )

            if not delta.does_cover(bounds=bounds):
                log.warning(
                    "Drude 'delta' spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append((f.sel_inside(bounds), delta.sel_inside(bounds)))

        return self.updated_copy(eps_inf=eps_inf_reduced, coeffs=coeffs_reduced)


class Debye(DispersiveMedium):
    """A dispersive medium described by the Debye model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty + \\sum_i
            \\frac{\\Delta\\epsilon_i}{1 - jf\\tau_i}

    Example
    -------
    >>> debye_medium = Debye(eps_inf=2.0, coeffs=[(1,2),(3,4)])
    >>> eps = debye_medium.eps_model(200e12)

    See Also
    --------

    :class:`CustomDebye`
        A spatially varying dispersive medium described by the Debye model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: pd.PositiveFloat = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[float, pd.PositiveFloat], ...] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, \\tau_i`) values for model.",
        units=(PERMITTIVITY, SECOND),
    )

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if `allow_gain` is False."""
        if values.get("allow_gain"):
            return val
        for del_ep, _ in val:
            if del_ep < 0:
                raise ValidationError(
                    "For passive medium, 'Delta epsilon_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    _validate_permittivity_modulation = DispersiveMedium._permittivity_modulation_validation()
    _validate_conductivity_modulation = DispersiveMedium._conductivity_modulation_validation()

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        eps = self.eps_inf + 0.0j
        for de, tau in self.coeffs:
            eps = eps + de / (1 - 1j * frequency * tau)
        return eps

    def _pole_residue_dict(self):
        """Dict representation of Medium as a pole-residue model."""

        poles = []
        for de, tau in self.coeffs:
            a = -2 * np.pi / tau + 0j
            c = -0.5 * de * a

            poles.append((a, c))

        return dict(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


class CustomDebye(CustomDispersiveMedium, Debye):
    """A spatially varying dispersive medium described by the Debye model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty + \\sum_i
            \\frac{\\Delta\\epsilon_i}{1 - jf\\tau_i}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(1+np.random.random((5, 6, 7)), coords=coords)
    >>> eps1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> tau1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> debye_medium = CustomDebye(eps_inf=eps_inf, coeffs=[(eps1,tau1),])
    >>> eps = debye_medium.eps_model(200e12)

    See Also
    --------

    :class:`Debye`
        A dispersive medium described by the Debye model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = pd.Field(
        ...,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        pd.Field(
            ...,
            title="Coefficients",
            description="List of (:math:`\\Delta\\epsilon_i, \\tau_i`) values for model.",
            units=(PERMITTIVITY, SECOND),
        )
    )

    _no_nans_eps_inf = validate_no_nans("eps_inf")
    _no_nans_coeffs = validate_no_nans("coeffs")

    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @pd.validator("eps_inf", always=True)
    def _eps_inf_positive(cls, val):
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["eps_inf"])
    def _coeffs_correct_shape(cls, val, values):
        """coeffs must have consistent shape."""
        for de, tau in val:
            if not _check_same_coordinates(de, values["eps_inf"]) or not _check_same_coordinates(
                tau, values["eps_inf"]
            ):
                raise SetupError(
                    "All terms in 'coeffs' must have the same coordinates; "
                    "The coordinates must also be consistent with 'eps_inf'."
                )
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((de, tau)):
                raise SetupError("All terms in 'coeffs' must be real.")
        return val

    @pd.validator("coeffs", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        allow_gain = values.get("allow_gain")
        for del_ep, tau in val:
            if np.any(_get_numpy_array(tau) <= 0):
                raise SetupError("For stable medium, 'tau_i' must be positive.")
            if not allow_gain and np.any(_get_numpy_array(del_ep) < 0):
                raise ValidationError(
                    "For passive medium, 'Delta epsilon_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        eps = Debye.eps_model(self, frequency)
        return (eps, eps, eps)

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomDebye
            CustomDebye with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("Eps inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        coeffs_reduced = []
        for de, tau in self.coeffs:
            if not de.does_cover(bounds=bounds):
                log.warning(
                    "Debye 'f' spatial data array does not fully cover the requested region."
                )

            if not tau.does_cover(bounds=bounds):
                log.warning(
                    "Debye 'tau' spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append((de.sel_inside(bounds), tau.sel_inside(bounds)))

        return self.updated_copy(eps_inf=eps_inf_reduced, coeffs=coeffs_reduced)


def medium_from_nk(n: float, k: float, freq: float, **kwargs) -> Union[Medium, Lorentz]:
    """Convert ``n`` and ``k`` values at frequency ``freq`` to :class:`Medium` if ``Re[epsilon]>=1``,
    or :class:`Lorentz` if if ``Re[epsilon]<1``.

    Parameters
    ----------
    n : float
        Real part of refractive index.
    k : float = 0
        Imaginary part of refrative index.
    freq : float
        Frequency to evaluate permittivity at (Hz).
    kwargs: dict
        Keyword arguments passed to the medium construction.

    Returns
    -------
    Union[:class:`Medium`, :class:`Lorentz`]
        Dispersionless medium or Lorentz medium having refractive index n+ik at frequency ``freq``.
    """
    eps_complex = AbstractMedium.nk_to_eps_complex(n, k)
    if eps_complex.real >= 1:
        return Medium.from_nk(n, k, freq, **kwargs)
    return Lorentz.from_nk(n, k, freq, **kwargs)
