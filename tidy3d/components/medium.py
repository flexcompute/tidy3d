# pylint: disable=invalid-name
"""Defines properties of the medium / materials 

Attributes
----------
MediumType : TYPE
    Description
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Callable
import pydantic
import numpy as np

from .base import Tidy3dBaseModel
from .types import PoleAndResidue, Literal, Ax, FreqBound
from .viz import add_ax_if_none

from ..constants import C_0, inf
from ..log import log

""" Medium Definitions """


class AbstractMedium(ABC, Tidy3dBaseModel):
    """A medium within which electromagnetic waves propagate"""

    name: str = None
    frequency_range: Tuple[FreqBound, FreqBound] = (-inf, inf)

    @abstractmethod
    def eps_model(self, frequency: float) -> complex:
        """complex permittivity as a function of frequency

        Parameters
        ----------
        frequency : float
            Description
        name: ``str``
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
    """Dispersionless medium."""

    permittivity: pydantic.confloat(ge=1.0) = 1.0
    conductivity: pydantic.confloat(ge=0.0) = 0.0
    type: Literal["Medium"] = "Medium"

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : ``float``
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        ``complex``
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
    """defines a dispersion model through pole-residue pair model.

    Parameters
    ----------
    eps_inf : ``float`` = 1.0
        Permittivity at infinite frequency.
    poles : ``[((float, float), (float, float))]`` = []
        List of poles, ``poles[0]`` contain the real and imaginary parts of
        the frequency and amplitude of the first, respecitvely.

    Example
    -------
    >>> pole_res = PoleResidue(eps_inf=2.0, poles=[((1,2),(3,4)), ((5,6),(7,8))])
    """

    eps_inf: float = 1.0
    poles: List[PoleAndResidue] = []
    type: Literal["PoleResidue"] = "PoleResidue"

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : ``float``
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        ``complex``
            Complex-valued relative permittivity evaluated at ``frequency``.
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
        """string representation

        Returns
        -------
        TYPE
            Description
        """
        return (
            f"td.PoleResidue("
            f"\n\tpoles={self.poles}, "
            f"\n\tfrequency_range={self.frequency_range})"
        )


class Sellmeier(DispersiveMedium):
    """Sellmeier model for dispersion"""

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
        frequency : ``float``
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        ``complex``
            Complex-valued relative permittivity evaluated at ``frequency``.
        """
        n = self._n_model(frequency)
        return nk_to_eps_complex(n)


class Lorentz(DispersiveMedium):
    """Lorentz model for dispersion"""

    eps_inf: float = 1.0
    coeffs: List[Tuple[float, float, float]]
    type: Literal["Lorentz"] = "Lorentz"

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : ``float``
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        ``complex``
            Complex-valued relative permittivity evaluated at ``frequency``.
        """
        eps = self.eps_inf + 0.0j
        for (de, f, delta) in self.coeffs:
            eps += (de * f ** 2) / (f ** 2 + 2j * f * delta - frequency ** 2)
        return eps


class Debye(DispersiveMedium):
    """Debye model for dispersion"""

    eps_inf: float = 1.0
    coeffs: List[Tuple[float, float]]
    type: Literal["Debye"] = "Debye"

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : ``float``
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        ``complex``
            Complex-valued relative permittivity evaluated at ``frequency``.
        """
        eps = self.eps_inf + 0.0j
        for (de, tau) in self.coeffs:
            eps += de / (1 + 1j * frequency * tau)
        return eps


MediumType = Union[Medium, PoleResidue, Sellmeier, Lorentz, Debye]

""" conversion helpers """


def nk_to_eps_complex(n: float, k: float = 0.0) -> complex:
    """Convert n, k to complex permittivity.

    Parameters
    ----------
    n : ``float``
        real part of refractive index
    k : ``float`` = 0
        imaginary part of refrative index.

    Returns
    -------
    ``complex``
        Complex-valued relative permittivty.
    """
    eps_real = n ** 2 - k ** 2
    eps_imag = 2 * n * k
    return eps_real + 1j * eps_imag


def eps_complex_to_nk(eps_c: complex) -> Tuple[float, float]:
    """convert complex permittivity to n, k

    Parameters
    ----------
    eps_c : ``complex``
        Complex-valued relative permittivity.

    Returns
    -------
    ``(float, float)``
        Real and imaginary parts of refractive index (n & k).
    """
    ref_index = np.sqrt(eps_c)
    return ref_index.real, ref_index.imag


def nk_to_eps_sigma(n: float, k: float, freq: float) -> Tuple[float, float]:
    """convert n, k at freq to permittivity and conductivity

    Parameters
    ----------
    n : ``float``
        real part of refractive index
    k : ``float`` = 0
        imaginary part of refrative index.
    frequency : ``float``
        Frequency to evaluate permittivity at (Hz).

    Returns
    -------
    ``(float, float)``
        Real part of relative permittivity & conductivity.
    """
    eps_complex = nk_to_eps_complex(n, k)
    eps_real, eps_imag = eps_complex.real, eps_complex.imag
    omega = 2 * np.pi * freq
    sigma = omega * eps_imag
    return eps_real, sigma


def nk_to_medium(n: float, k: float, freq: float) -> Medium:
    """Convert ``n`` and ``k`` values at ``frequency`` to :class:`Medium`.

    Parameters
    ----------
    n : ``float``
        real part of refractive index
    k : ``float`` = 0
        imaginary part of refrative index.
    frequency : ``float``
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
    eps_real : ``float``
        Real-valued relative permittivity.
    sigma : ``float``
        Conductivity.
    freq : ``float``
        Frequency to evaluate permittivity at (Hz).

    Returns
    -------
    complex
        Complex-valued relative permittivity.
    """
    omega = 2 * np.pi * freq
    return eps_real + 1j * sigma / omega
