# pylint: disable=invalid-name
""" Defines properties of the medium / materials """

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union
import pydantic
import numpy as np

from .base import Tidy3dBaseModel
from .types import PoleAndResidue, Literal
from ..constants import C_0, inf

""" conversion helpers """


def nk_to_eps_complex(n, k=0.0):
    """convert n, k to complex permittivity"""
    eps_real = n ** 2 - k ** 2
    eps_imag = 2 * n * k
    return eps_real + 1j * eps_imag


def eps_complex_to_nk(eps_c):
    """convert complex permittivity to n, k"""
    ref_index = np.sqrt(eps_c)
    return ref_index.real, ref_index.imag


def nk_to_eps_sigma(n, k, freq):
    """convert n, k at freq to permittivity and conductivity"""
    eps_complex = nk_to_eps_complex(n, k)
    eps_real, eps_imag = eps_complex.real, eps_complex.imag
    omega = 2 * np.pi * freq
    sigma = omega * eps_imag
    return eps_real, sigma


def nk_to_medium(n, k, freq):
    """convert n, k at freq to Medium"""
    eps, sigma = nk_to_eps_sigma(n, k, freq)
    return Medium(permittivity=eps, conductivity=sigma)


def eps_sigma_to_eps_complex(eps_real, sigma, freq):
    """convert permittivity and conductivity to complex permittivity at freq"""
    omega = 2 * np.pi * freq
    return eps_real + 1j * sigma / omega


""" Medium Definitions """


class AbstractMedium(ABC, Tidy3dBaseModel):
    """A medium within which electromagnetic waves propagate"""

    # frequencies within which the medium is valid
    frequency_range: Optional[Tuple[float, float]] = (-inf, inf)

    @abstractmethod
    def eps_model(self, frequency: float) -> complex:
        """complex permittivity as a function of frequency"""


""" Dispersionless Medium """


class Medium(AbstractMedium):
    """Dispersionless medium"""

    permittivity: pydantic.confloat(ge=1.0) = 1.0
    conductivity: pydantic.confloat(ge=0.0) = 0.0
    type: Literal["Medium"] = "Medium"

    def eps_model(self, frequency: float = None):
        if frequency is None:
            return self.permittivity
        return eps_sigma_to_eps_complex(self.permittivity, self.conductivity, frequency)

    def __str__(self):
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
    """defines a dispersion model"""

    eps_inf: float = 1.0
    poles: List[PoleAndResidue]
    type: Literal["PoleResidue"] = "PoleResidue"

    def eps_model(self, frequency):
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
        return (
            f"td.PoleResidue("
            f"\n\tpoles={self.poles}, "
            f"\n\tfrequency_range={self.frequency_range})"
        )


class Sellmeier(DispersiveMedium):
    """Sellmeier model for dispersion"""

    coeffs: List[Tuple[float, float]]
    type: Literal["Sellmeier"] = "Sellmeier"

    def _n_model(self, frequency):
        wvl = C_0 / frequency
        wvl2 = wvl ** 2
        n_squared = 1.0
        for (B, C) in self.coeffs:
            n_squared += B * wvl2 / (wvl2 - C)
        return np.sqrt(n_squared)

    def eps_model(self, frequency):
        n = self._n_model(frequency)
        return nk_to_eps_complex(n)


class Lorentz(DispersiveMedium):
    """Lorentz model for dispersion"""

    eps_inf: float = 1.0
    coeffs: List[Tuple[float, float, float]]
    type: Literal["Lorentz"] = "Lorentz"

    def eps_model(self, frequency):
        eps = self.eps_inf + 0.0j
        for (de, f, delta) in self.coeffs:
            eps += (de * f ** 2) / (f ** 2 + 2j * f * delta - frequency ** 2)
        return eps


class Debye(DispersiveMedium):
    """Debye model for dispersion"""

    eps_inf: float = 1.0
    coeffs: List[Tuple[float, float]]
    type: Literal["Debye"] = "Debye"

    def eps_model(self, frequency):
        eps = self.eps_inf + 0.0j
        for (de, tau) in self.coeffs:
            eps += de / (1 + 1j * frequency * tau)
        return eps


MediumType = Union[Medium, PoleResidue, Sellmeier, Lorentz, Debye]
