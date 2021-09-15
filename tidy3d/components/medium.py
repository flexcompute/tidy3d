import numpy as np
from abc import ABC, abstractmethod

from .base import Tidy3dBaseModel
from .validators import ensure_greater_or_equal
from .types import List, Tuple, PoleResidue
from .constants import C_0, inf

""" conversion helpers """

def nk_to_eps_complex(n, k=0.0):
    """ convert n, k to complex permittivity """
    eps_real = n**2 - k**2
    eps_imag = 2*n*k
    return eps_real + 1j*eps_imag

def nk_to_eps_sigma(n, k, freq):
    """ convert n, k at freq to permittivity and conductivity """
    eps_complex = nk_to_eps_complex(n, k)
    eps_real, eps_imag = eps_complex.real, eps_complex.imag
    omega = 2 * np.pi * freq
    sigma = omega * eps_imag
    return eps_real, sigma

def nk_to_medium(n, k, freq):
    """ convert n, k at freq to Medium """
    eps, sigma = nk_to_eps_sigma(n, k, freq)
    return Medium(permittivity=eps, conductivity=sigma)

def eps_sigma_to_eps_complex(eps_real, sigma, freq):
    """ convert permittivity and conductivity to complex permittivity at freq """
    omega = 2 * np.pi * freq
    return eps_real + 1j * sigma / omega

""" Medium Definitions """

class AbstractMedium(ABC, Tidy3dBaseModel):
    """A medium within which electromagnetic waves propagate"""
    
    # frequencies within which the medium is valid
    frequency_range: Tuple[float, float] = (-inf, inf)

    @abstractmethod
    def eps_model(self, frequency: float) -> complex:
        """ complex permittivity as a function of frequency """
        pass

""" Dispersionless Medium """

class Medium(AbstractMedium):
    """ Dispersionless medium"""

    permittivity: float = 1.0
    conductivity: float = 0.0

    _permittivity_validator = ensure_greater_or_equal("permittivity", 1.0)
    _conductivity_validator = ensure_greater_or_equal("conductivity", 0.0)

    def eps_model(self, frequency):
        return eps_sigma_to_eps_complex(self.permittivity, self.conductivity, frequency)

""" Dispersive Media """

class DispersiveMedium(AbstractMedium):
    """ A Medium with dispersion (propagation characteristics depend on frequency) """
    pass

class PoleResidue(DispersiveMedium):
    """defines a dispersion model"""

    eps_inf: float = 1.0
    poles: List[PoleResidue]

    def eps_model(self, frequency):
        omega = 2 * np.pi * frequency
        eps = self.eps_inf + 0.0j
        for p in self.poles:
            (ar, ai), (cr, ci) = p
            a = ar + 1j*ai
            c = cr + 1j*ci
            a_cc = np.conj(a)
            c_cc = np.conj(c)
            eps -= c / (1j * omega + a)
            eps -= c_cc / (1j * omega + a_cc)
        return eps

class Sellmeier(DispersiveMedium):
    """ Sellmeier model for dispersion"""

    coeffs: List[Tuple[float, float]]

    def _n_model(self, frequency):
        wvl = C_0 / frequency
        wvl2 = wvl**2
        n_squared = 1.0
        for (B, C) in self.coeffs:
            n_squared += B * wvl2 / (wvl2 - C)
        return np.sqrt(n_squared)

    def eps_model(self, frequency):
        n = self._n_model(frequency)
        return nk_to_eps_complex(n)

class Lorentz(DispersiveMedium):
    """ Lorentz model for dispersion"""
    
    eps_inf: float = 1.0
    coeffs: List[Tuple[float, float, float]]

    def eps_model(self, frequency):
        eps = self.eps_inf + 0.0j
        for (de, f, delta) in self.coeffs:
            eps += (de * f**2) / (f**2 + 2j*f*delta - frequency**2)
        return eps

class Debye(DispersiveMedium):
    """ Debye model for dispersion"""

    eps_inf: float = 1.0
    coeffs: List[Tuple[float, float]]

    def eps_model(self, frequency):
        eps = self.eps_inf + 0.0j
        for (de, tau) in self.coeffs:
            eps += de / (1 + 1j*frequency*tau)
        return eps

# class Drude(DispersiveMedium):
#     """ need to do """
#     pass
