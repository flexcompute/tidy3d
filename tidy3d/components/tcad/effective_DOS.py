from abc import ABC, abstractmethod

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import C_0, HBAR, K_B

from ...exceptions import DataError

# constants definition
m_e_C_square = 0.51099895069e6  # (electron mass * C_0^2) in eV
m_e_eV = m_e_C_square / C_0 / C_0  # equivalent electron mass in eV
um_3_to_cm_3 = 1e12  # conversion factor from micron^(-3) to cm^(-3)

DOS_aux_const = 2.0 * np.power((m_e_eV * K_B) / (2 * np.pi * HBAR * HBAR), 1.5) * um_3_to_cm_3


class EffectiveDOS(Tidy3dBaseModel, ABC):
    """Abstract class for the effective density of states"""

    @abstractmethod
    def calc_eff_dos(self, T: float):
        """Abstract method to calculate the effective density of states."""
        pass

    @abstractmethod
    def calc_eff_dos_derivative(self, T: float):
        """Abstract method to calculate the temperature derivative of the effective density of states."""
        pass

    def get_effective_DOS(self, T: float):
        if T <= 0:
            raise DataError(
                f"Incorrect temperature value ({T}) for the effectve density of states calculation."
            )

        return self.calc_eff_dos(T)

    def get_effective_DOS_derivative(self, T: float):
        if T <= 0:
            raise DataError(
                f"Incorrect temperature value ({T}) for the effectve density of states calculation."
            )

        return self.calc_eff_dos_derivative(T)


class ConstantEffectiveDOS(EffectiveDOS):
    """Constant effective density of states model."""

    N: pd.PositiveFloat = pd.Field(
        ..., title="Effective DOS", description="Effective density of states", units="cm^(-3)"
    )

    def calc_eff_dos(self, T: float):
        return self.N

    def calc_eff_dos_derivative(self, T: float):
        return 0.0


class IsotropicEffectiveDOS(EffectiveDOS):
    """Effective density of states model that assumes single valley and isotropic effective mass.
    The model assumes the standard equation for the 3D semiconductor with parabolic energy dispersion:

    .. math::

        \\begin{equation}
             \\mathbf{N_eff} = 2 * (\\frac{m_eff * m_e * k_B T}{2 \\pi \\hbar^2})^(3/2)
        \\end{equation}
    """

    m_eff: pd.PositiveFloat = pd.Field(
        ...,
        title="Effective mass",
        description="Effective mass of the carriers",
        units="Electron mass",
    )

    def calc_eff_dos(self, T: float):
        return np.power(self.m_eff * T, 1.5) * DOS_aux_const

    def calc_eff_dos_derivative(self, T: float):
        return self.calc_eff_dos(T) * 1.5 / T


class MultiValleyEffectiveDOS(EffectiveDOS):
    """Effective density of states model that assumes multiple equivalent valleys and anisotropic effective mass.
    The model assumes the standard equation for the 3D semiconductor with parabolic energy dispersion:

    .. math::

        \\begin{equation}
             \\mathbf{N_eff} = 2 * N_valley * (m_{eff_long} * m_{eff_trans} * m_{eff_trans})^(1/2) *(\\frac{m_e * k_B * T}{2 \\pi * \\hbar^2})^(3/2)
        \\end{equation}
    """

    m_eff_long: pd.PositiveFloat = pd.Field(
        ...,
        title="Longitudinal effective mass",
        description="Effective mass of the carriers in the longitudinal direction",
        units="Electron mass",
    )

    m_eff_trans: pd.PositiveFloat = pd.Field(
        ...,
        title="Longitudinal effective mass",
        description="Effective mass of the carriers in the transverse direction",
        units="Electron mass",
    )

    N_valley: pd.PositiveFloat = pd.Field(
        ..., title="Number of valleys", description="Number of effective valleys"
    )

    def calc_eff_dos(self, T: float):
        return (
            self.N_valley
            * np.power(self.m_eff_long * self.m_eff_trans * self.m_eff_trans, 0.5)
            * np.power(T, 1.5)
            * DOS_aux_const
        )

    def calc_eff_dos_derivative(self, T: float):
        return self.calc_eff_dos(T) * 1.5 / T


class DualValleyEffectiveDOS(EffectiveDOS):
    """Effective density of states model that assumes combibation of light holes and heavy holes with isotropic effective masses.
    The model assumes the standard equation for the 3D semiconductor with parabolic energy dispersion:

    .. math::

        \\begin{equation}
             \\mathbf{N_eff} = 2 * ( {\\frac{m_{eff_lh} * m_e * k_B * T}{2 \\pi \\hbar^2})^(3/2) + (\\frac{m_{eff_hh} * m_e * k_B * T}{2 \\pi \\hbar^2})^(3/2) )
        \\end{equation}
    """

    m_eff_lh: pd.PositiveFloat = pd.Field(
        ...,
        title="Light hole effective mass",
        description="Effective mass of the light holes",
        units="Electron mass",
    )

    m_eff_hh: pd.PositiveFloat = pd.Field(
        ...,
        title="Heavy hole effective mass",
        description="Effective mass of the heavy holes",
        units="Electron mass",
    )

    def calc_eff_dos(self, T: float):
        return (np.power(self.m_eff_lh * T, 1.5) + np.power(self.m_eff_hh * T, 1.5)) * DOS_aux_const

    def calc_eff_dos_derivative(self, T: float):
        return self.calc_eff_dos(T) * 1.5 / T
