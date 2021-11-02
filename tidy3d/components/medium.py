# pylint: disable=invalid-name
"""Defines properties of the medium / materials"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Callable
import pydantic
import numpy as np

from .base import Tidy3dBaseModel
from .types import PoleAndResidue, Literal, Ax, FreqBound
from .viz import add_ax_if_none
from .validators import validate_name_str

from ..constants import C_0, inf
from ..log import log


# TODO: make pole residue support complex values instead of tuples of real and imaginary.
# TODO: double check all units.
# TODO: convert all to pole-residue.

""" Medium Definitions """


class AbstractMedium(ABC, Tidy3dBaseModel):
    """A medium within which electromagnetic waves propagate.

    Parameters
    ----------
    frequeuncy_range : Tuple[float, float] = (-inf, inf)
        Range of validity for the medium in Hz.
        If simulation or plotting functions use frequency out of this range, a warning is thrown.
    name : str = None
        Optional name for the medium.
    """

    name: str = None
    frequency_range: Tuple[FreqBound, FreqBound] = (-inf, inf)

    _name_validator = validate_name_str()

    @abstractmethod
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            Complex-valued relative permittivity evaluated at ``frequency``.
        """

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:  # pylint: disable=invalid-name
        """plot n, k of medium as a function of frequencies."""
        freqs = np.array(freqs)
        eps_complex = self.eps_model(freqs)
        n, k = eps_complex_to_nk(eps_complex)

        freqs_thz = freqs / 1e12
        ax.plot(freqs_thz, n, label="n")
        ax.plot(freqs_thz, k, label="k")
        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax


def ensure_freq_in_range(eps_model: Callable[[float], complex]) -> Callable[[float], complex]:
    """Decorate ``eps_model`` to log warning if frequency supplied is out of bounds."""

    def _eps_model(self, frequency: float) -> complex:
        """New eps_model function."""
        fmin, fmax = self.frequency_range
        if np.any(frequency < fmin) or np.any(frequency > fmax):
            log.warning(
                "frequency passed to `Medium.eps_model()`"
                f"is outside of `Medium.frequency_range` = {self.frequency_range}"
            )
        return eps_model(self, frequency)

    return _eps_model


""" Dispersionless Medium """


class Medium(AbstractMedium):
    """Dispersionless medium.

    Parameters
    ----------
    permittivity : float = 1.0
        Relative permittivity in dimensionless units.
        Must be greater than or equal to 1.
    conductivity : float = 0.0
        Electric conductivity in dimensions of (S/micron)
        Defined such that the imaginary part of the complex permittivity at angular frequency omega
        is given by conductivity/omega.
        Must be greater than or equal to 0.
    frequeuncy_range : Tuple[float, float] = (-inf, inf)
        Range of validity for the medium in Hz.
        If simulation or plotting functions use frequency out of this range, a warning is thrown.
    name : str = None
        Optional name for the medium.

    Example
    -------
    >>> dielectric = Medium(permittivity=4.0, name='my_medium')
    >>> eps = dielectric.eps_model(200e12)
    """

    permittivity: pydantic.confloat(ge=1.0) = 1.0
    conductivity: pydantic.confloat(ge=0.0) = 0.0
    type: Literal["Medium"] = "Medium"

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            Complex-valued relative permittivity evaluated at ``frequency``.
        """
        return eps_sigma_to_eps_complex(self.permittivity, self.conductivity, frequency)

    def __str__(self) -> str:
        """string representation."""
        return (
            f"td.Medium("
            f"permittivity={self.permittivity},"
            f"conductivity={self.conductivity},"
            f"frequency_range={self.frequency_range})"
        )


""" Dispersive Media """


class DispersiveMedium(AbstractMedium, ABC):
    """A Medium with dispersion (propagation characteristics depend on frequency)"""


class PoleResidue(DispersiveMedium):
    """A dispersive medium described by the pole-residue pair model.
    The frequency-dependence of the complex-valued permittivity is described by:

    .. math::

        \\epsilon(\\omega) = \\epsilon_\\infty - \\sum_i
        \\left[\\frac{c_i}{j \\omega + a_i} +
        \\frac{c_i^*}{j \\omega + a_i^*}\\right]

    where :math:`a_i` is in Hz and :math:`c_i` is unitless.

    Parameters
    ----------
    eps_inf : float = 1.0
        Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).
    poles : List[Tuple[Tuple[float, float], Tuple[float, float]]]
        List of (:math:`a_i, c_i`) poles for the model.
        Note that :math:`a_i` and :math:`c_i` are complex-valued and therefore must each
        be specified as a tuple containing the real and imaginary parts.
    frequeuncy_range : Tuple[float, float] = (-inf, inf)
        Range of validity for the medium in Hz.
        If simulation or plotting functions use frequency out of this range, a warning is thrown.
    name : str = None
        Optional name for the medium.

    Example
    -------
    >>> pole_res = PoleResidue(eps_inf=2.0, poles=[((1,2),(3,4)), ((5,6),(7,8))])
    >>> eps = pole_res.eps_model(200e12)
    """

    eps_inf: float = 1.0
    poles: List[PoleAndResidue] = []
    type: Literal["PoleResidue"] = "PoleResidue"

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            Complex-valued relative permittivity evaluated at the frequency.
        """
        omega = 2 * np.pi * frequency
        eps = self.eps_inf + 0.0j
        for p in self.poles:
            (ar, ai), (cr, ci) = p
            a = ar + 1j * ai
            c = cr + 1j * ci
            a_cc = np.conj(a)
            c_cc = np.conj(c)
            eps -= c / (1j * omega + a)
            eps -= c_cc / (1j * omega + a_cc)
        return eps

    def __str__(self):
        """string representation"""
        return (
            f"td.PoleResidue("
            f"\n\tpoles={self.poles}, "
            f"\n\tfrequency_range={self.frequency_range})"
        )


class Sellmeier(DispersiveMedium):
    """A dispersive medium described by the Sellmeier model.
    The frequency-dependence of the refractive index is described by:

    .. math::

        n(\\lambda)^2 = 1 + \\sum_i \\frac{B_i \\lambda^2}{\\lambda^2 - C_i}

    where :math:`\\lambda` is in microns, :math:`B_i` is unitless and :math:`C_i` is in microns^2.

    Parameters
    ----------
    coeffs : List[Tuple[float, float]]
        List of Sellmeier (:math:`B_i, C_i`) coefficients.
    frequeuncy_range : Tuple[float, float] = (-inf, inf)
        Range of validity for the medium in Hz.
        If simulation or plotting functions use frequency out of this range, a warning is thrown.
    name : str = None
        Optional name for the medium.

    Example
    -------
    >>> sellmeier_medium = Sellmeier(coeffs=[(1,2), (3,4)])
    >>> eps = sellmeier_medium.eps_model(200e12)
    """

    coeffs: List[Tuple[float, float]]
    type: Literal["Sellmeier"] = "Sellmeier"

    def _n_model(self, frequency: float) -> complex:
        """Complex-valued refractive index as a function of frequency."""
        wvl = C_0 / frequency
        wvl2 = wvl ** 2
        n_squared = 1.0
        for (B, C) in self.coeffs:
            n_squared += B * wvl2 / (wvl2 - C)
        return np.sqrt(n_squared)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            Complex-valued relative permittivity evaluated at the frequency.
        """
        n = self._n_model(frequency)
        return nk_to_eps_complex(n)


class Lorentz(DispersiveMedium):
    """A dispersive medium described by the Lorentz model.
    The frequency-dependence of the complex-valued permittivity is described by:

    .. math::
        \\epsilon(f) = \\epsilon_\\infty + \\sum_i
        \\frac{\\Delta\\epsilon_i f_i^2}{f_i^2 + 2jf\\delta_i - f^2}

    where :math:`f, f_i, \\delta_i` are in Hz.

    Parameters
    ----------
    eps_inf : float = 1.0
        Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).
    coeffs : List[Tuple[float, float, float]]
        List of (:math:`\\Delta\\epsilon_i, f_i, \\delta_i`) values for model.
    frequeuncy_range : Tuple[float, float] = (-inf, inf)
        Range of validity for the medium in Hz.
        If simulation or plotting functions use frequency out of this range, a warning is thrown.
    name : str = None
        Optional name for the medium.

    Example
    -------
    >>> lorentz_medium = Lorentz(eps_inf=2.0, coeffs=[(1,2,3), (4,5,6)])
    >>> eps = lorentz_medium.eps_model(200e12)
    """

    eps_inf: float = 1.0
    coeffs: List[Tuple[float, float, float]]
    type: Literal["Lorentz"] = "Lorentz"

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            Complex-valued relative permittivity evaluated at the frequency.
        """
        eps = self.eps_inf + 0.0j
        for (de, f, delta) in self.coeffs:
            eps += (de * f ** 2) / (f ** 2 + 2j * f * delta - frequency ** 2)
        return eps


class Debye(DispersiveMedium):
    """A dispersive medium described by the Debye model.
    The frequency-dependence of the complex-valued permittivity is described by:

    .. math::
        \\epsilon(f) = \\epsilon_\\infty + \\sum_i
        \\frac{\\Delta\\epsilon_i}{1 + jf\\tau_i}

    where :math:`f` is in Hz, and :math:`\\tau_i` is in seconds.

    Parameters
    ----------
    eps_inf : float = 1.0
        Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).
    coeffs : List[Tuple[float, float, float]]
        List of (:math:`\\Delta\\epsilon_i, \\tau_i`) values for model.
    frequeuncy_range : Tuple[float, float] = (-inf, inf)
        Range of validity for the medium in Hz.
        If simulation or plotting functions use frequency out of this range, a warning is thrown.
    name : str = None
        Optional name for the medium.

    Example
    -------
    >>> debye_medium = Debye(eps_inf=2.0, coeffs=[(1,2),(3,4)])
    >>> eps = debye_medium.eps_model(200e12)
    """

    eps_inf: float = 1.0
    coeffs: List[Tuple[float, float]]
    type: Literal["Debye"] = "Debye"

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            Complex-valued relative permittivity evaluated at the frequency.
        """
        eps = self.eps_inf + 0.0j
        for (de, tau) in self.coeffs:
            eps += de / (1 + 1j * frequency * tau)
        return eps


# types of mediums that can be used in Simulation and Structures
MediumType = Union[Medium, PoleResidue, Sellmeier, Lorentz, Debye]

""" Conversion helper functions """


def nk_to_eps_complex(n: float, k: float = 0.0) -> complex:
    """Convert n, k to complex permittivity.

    Parameters
    ----------
    n : float
        Real part of refractive index.
    k : float = 0.0
        Imaginary part of refrative index.

    Returns
    -------
    complex
        Complex-valued relative permittivty.
    """
    eps_real = n ** 2 - k ** 2
    eps_imag = 2 * n * k
    return eps_real + 1j * eps_imag


def eps_complex_to_nk(eps_c: complex) -> Tuple[float, float]:
    """Convert complex permittivity to n, k values.

    Parameters
    ----------
    eps_c : complex
        Complex-valued relative permittivity.

    Returns
    -------
    Tuple[float, float]
        Real and imaginary parts of refractive index (n & k).
    """
    ref_index = np.sqrt(eps_c)
    return ref_index.real, ref_index.imag


def nk_to_eps_sigma(n: float, k: float, freq: float) -> Tuple[float, float]:
    """Convert ``n``, ``k`` at frequency ``freq`` to permittivity and conductivity values.

    Parameters
    ----------
    n : float
        Real part of refractive index.
    k : float = 0.0
        Imaginary part of refrative index.
    frequency : float
        Frequency to evaluate permittivity at (Hz).

    Returns
    -------
    Tuple[float, float]
        Real part of relative permittivity & electric conductivity.
    """
    eps_complex = nk_to_eps_complex(n, k)
    eps_real, eps_imag = eps_complex.real, eps_complex.imag
    omega = 2 * np.pi * freq
    sigma = omega * eps_imag
    return eps_real, sigma


def nk_to_medium(n: float, k: float, freq: float) -> Medium:
    """Convert ``n`` and ``k`` values at frequency ``freq`` to :class:`Medium`.

    Parameters
    ----------
    n : float
        Real part of refractive index.
    k : float = 0
        Imaginary part of refrative index.
    frequency : float
        Frequency to evaluate permittivity at (Hz).

    Returns
    -------
    :class:`Medium`
        medium containing the corresponding ``permittivity`` and ``conductivity``.
    """
    eps, sigma = nk_to_eps_sigma(n, k, freq)
    return Medium(permittivity=eps, conductivity=sigma)


def eps_sigma_to_eps_complex(eps_real: float, sigma: float, freq: float) -> complex:
    """convert permittivity and conductivity to complex permittivity at freq

    Parameters
    ----------
    eps_real : float
        Real-valued relative permittivity.
    sigma : float
        Conductivity.
    freq : float
        Frequency to evaluate permittivity at (Hz).

    Returns
    -------
    complex
        Complex-valued relative permittivity.
    """
    omega = 2 * np.pi * freq
    return eps_real + 1j * sigma / omega
