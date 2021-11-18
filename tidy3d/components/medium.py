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

from ..constants import C_0, pec_val
from ..log import log


""" Medium Definitions """


class AbstractMedium(ABC, Tidy3dBaseModel):
    """A medium within which electromagnetic waves propagate.

    Parameters
    ----------
    frequeuncy_range : Tuple[float, float] = None
        Range of validity for the medium in Hz.
        If None, then all frequencies are valid.
        If simulation or plotting functions use frequency out of this range, a warning is thrown.
    name : str = None
        Optional name for the medium.
    """

    name: str = None
    frequency_range: Tuple[FreqBound, FreqBound] = None

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
        """Plot n, k of a :class:`Medium` as a function of frequency.

        Parameters
        ----------
        freqs: float
            Frequencies (Hz) to evaluate the medium properties at.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        freqs = np.array(freqs)
        eps_complex = self.eps_model(freqs)
        n, k = AbstractMedium.eps_complex_to_nk(eps_complex)

        freqs_thz = freqs / 1e12
        ax.plot(freqs_thz, n, label="n")
        ax.plot(freqs_thz, k, label="k")
        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    """ Conversion helper functions """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        eps_complex = AbstractMedium.nk_to_eps_complex(n, k)
        eps_real, eps_imag = eps_complex.real, eps_complex.imag
        omega = 2 * np.pi * freq
        sigma = omega * eps_imag
        return eps_real, sigma

    @staticmethod
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
            If not supplied, returns real part of permittivity (limit as frequency -> infinity.)

        Returns
        -------
        complex
            Complex-valued relative permittivity.
        """
        if freq is None:
            return eps_real
        omega = 2 * np.pi * freq
        return eps_real + 1j * sigma / omega


def ensure_freq_in_range(eps_model: Callable[[float], complex]) -> Callable[[float], complex]:
    """Decorate ``eps_model`` to log warning if frequency supplied is out of bounds."""

    def _eps_model(self, frequency: float) -> complex:
        """New eps_model function."""

        # if frequency is none, don't check, return original function
        if frequency is None or self.frequency_range is None:
            return eps_model(self, frequency)

        fmin, fmax = self.frequency_range
        if np.any(frequency < fmin) or np.any(frequency > fmax):
            log.warning(
                "frequency passed to `Medium.eps_model()`"
                f"is outside of `Medium.frequency_range` = {self.frequency_range}"
            )
        return eps_model(self, frequency)

    return _eps_model


""" Dispersionless Medium """

# PEC keyword
class PECMedium(AbstractMedium):
    """Perfect electrical conductor class.

    Note
    ----
    To avoid confusion from duplicate PECs,
    use the pre-defined instance ``PEC`` rather than creating your own :class:`PECMedium` instance.
    """

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

        # return something like frequency with value of pec_val + 0j
        return 0j * frequency + pec_val


# PEC instance (usable)
PEC = PECMedium(name="PEC")


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
        return AbstractMedium.eps_sigma_to_eps_complex(
            self.permittivity, self.conductivity, frequency
        )

    @classmethod
    def from_nk(cls, n: float, k: float, freq: float):
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
        eps, sigma = AbstractMedium.nk_to_eps_sigma(n, k, freq)
        return cls(permittivity=eps, conductivity=sigma)


class AnisotropicMedium(AbstractMedium):
    """Diagonally anisotripic medium.

    Parameters
    ----------
    xx : :class:`Medium`
        :class:`Medium` describing the :math:`\\epsilon_{xx}`-component of the permittivity tensor.
    yy : :class:`Medium`
        :class:`Medium` describing the :math:`\\epsilon_{yy}`-component of the permittivity tensor.
    zz : :class:`Medium`
        :class:`Medium` describing the :math:`\\epsilon_{zz}`-component of the permittivity tensor.
    name : str = None
        Optional name for the medium.

    Note
    ----
    Only diagonal anisotropy and non-dispersive components are currently supported.

    Example
    -------
    >>> medium_xx = Medium(permittivity=4.0)
    >>> medium_yy = Medium(permittivity=4.1)
    >>> medium_zz = Medium(permittivity=3.9)
    >>> anisotropic_dielectric = AnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)
    """

    xx: Medium
    yy: Medium
    zz: Medium

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[complex, complex, complex]
            Complex-valued relative permittivity for each component evaluated at ``frequency``.
        """
        eps_xx = self.xx.eps_model(frequency)
        eps_yy = self.yy.eps_model(frequency)
        eps_zz = self.zz.eps_model(frequency)
        return (eps_xx, eps_yy, eps_zz)

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`Medium` as a function of frequency.

        Parameters
        ----------
        freqs: float
            Frequencies (Hz) to evaluate the medium properties at.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        freqs = np.array(freqs)
        freqs_thz = freqs / 1e12

        for label, medium_component in zip(("xx", "yy", "zz"), (self.xx, self.yy, self.zz)):

            eps_complex = medium_component.eps_model(freqs)
            n, k = AbstractMedium.eps_complex_to_nk(eps_complex)
            ax.plot(freqs_thz, n, label=f"n, eps_{label}")
            ax.plot(freqs_thz, k, label=f"k, eps_{label}")

        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax


""" Dispersive Media """


class DispersiveMedium(AbstractMedium, ABC):
    """A Medium with dispersion (propagation characteristics depend on frequency)"""

    @property
    @abstractmethod
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""


class PoleResidue(DispersiveMedium):
    """A dispersive medium described by the pole-residue pair model.
    The frequency-dependence of the complex-valued permittivity is described by:

    .. math::

        \\epsilon(\\omega) = \\epsilon_\\infty - \\sum_i
        \\left[\\frac{c_i}{j \\omega + a_i} +
        \\frac{c_i^*}{j \\omega + a_i^*}\\right]

    where :math:`a_i` and :math:`c_i` are in units of rad/s.

    Parameters
    ----------
    eps_inf : float = 1.0
        Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).
    poles : List[Tuple[complex, complex]]
        List of complex-valued (:math:`a_i, c_i`) poles for the model.
    frequeuncy_range : Tuple[float, float] = (-inf, inf)
        Range of validity for the medium in Hz.
        If simulation or plotting functions use frequency out of this range, a warning is thrown.
    name : str = None
        Optional name for the medium.

    Example
    -------
    >>> pole_res = PoleResidue(eps_inf=2.0, poles=[(1+2j, 3+4j), (5+6j, 7+8j)])
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
        for (a, c) in self.poles:
            a_cc = np.conj(a)
            c_cc = np.conj(c)
            eps -= c / (1j * omega + a)
            eps -= c_cc / (1j * omega + a_cc)
        return eps

    @property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""
        return PoleResidue(
            eps_inf=self.eps_inf,
            poles=self.poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )

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
        return AbstractMedium.nk_to_eps_complex(n)

    @property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""

        poles = []
        for (B, C) in self.coeffs:
            beta = 2 * np.pi * C_0 / np.sqrt(C)
            alpha = -0.5 * beta * B
            a = 1j * beta
            c = 1j * alpha
            poles.append((a, c))

        return PoleResidue(
            eps_inf=1,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


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
            eps += (de * f ** 2) / (f ** 2 + 2j * frequency * delta - frequency ** 2)
        return eps

    @property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""

        poles = []
        for (de, f, delta) in self.coeffs:

            w = 2 * np.pi * f
            d = 2 * np.pi * delta

            if d > w:
                r = 1j * np.sqrt(d * d - w * w)
            else:
                r = np.sqrt(w * w - d * d)

            a = d - 1j * r
            c = 1j * de * w ** 2 / 2 / r

            poles.append((a, c))

        return PoleResidue(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


class Drude(DispersiveMedium):
    """A dispersive medium described by the Drude model.
    The frequency-dependence of the complex-valued permittivity is described by:

    .. math::
        \\epsilon(f) = \\epsilon_\\infty - \\sum_i
        \\frac{ f_i^2}{f^2 - jf\\delta_i}

    where :math:`f, f_i, \\delta_i` are in Hz.

    Parameters
    ----------
    eps_inf : float = 1.0
        Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).
    coeffs : List[Tuple[float, float]]
        List of (:math:`f_i, \\delta_i`) values for model.
    frequeuncy_range : Tuple[float, float] = (-inf, inf)
        Range of validity for the medium in Hz.
        If simulation or plotting functions use frequency out of this range, a warning is thrown.
    name : str = None
        Optional name for the medium.

    Example
    -------
    >>> drude_medium = Drude(eps_inf=2.0, coeffs=[(1,2), (3,4)])
    >>> eps = drude_medium.eps_model(200e12)
    """

    eps_inf: float = 1.0
    coeffs: List[Tuple[float, float]]
    type: Literal["Drude"] = "Drude"

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
        for (f, delta) in self.coeffs:
            eps -= (f ** 2) / (frequency ** 2 - 1j * frequency * delta)
        return eps

    @property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""

        poles = []
        for (f, delta) in self.coeffs:

            w = 2 * np.pi * f
            d = 2 * np.pi * delta

            c0 = -(w ** 2) / 2 / d + 0j
            a0 = 0j

            c1 = w ** 2 / 2 / d + 0j
            a1 = d + 0j

            poles.append((a0, c0))
            poles.append((a1, c1))

        return PoleResidue(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


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

    @property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""

        poles = []
        for (de, tau) in self.coeffs:
            a = 2 * np.pi / tau + 0j
            c = -0.5 * de * a
            poles.append((a, c))

        return PoleResidue(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


# types of mediums that can be used in Simulation and Structures
MediumType = Union[
    Literal[PEC], Medium, AnisotropicMedium, PoleResidue, Sellmeier, Lorentz, Debye, Drude
]
