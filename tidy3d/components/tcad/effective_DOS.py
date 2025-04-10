from abc import ABC, abstractmethod

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel

from ...exceptions import DataError

# constants definition
k_B = 1.380649e-23  # Boltzmann constant in J/K
m_e = 9.1093837139e-31  # electron mass in kg
Planck_hbar = 1.054571817e-34  # reduced Planck constant in J*s
m_3_to_cm_3 = 1e-6  # conversion factor from m^(-3) to cm^(-3)
DOS_aux_const = (
    2.0 * np.power((m_e * k_B) / (2 * np.pi * Planck_hbar * Planck_hbar), 1.5) * m_3_to_cm_3
)


class EffectiveDOS(Tidy3dBaseModel, ABC):
    """Abstract class for the effective density of states"""

    @abstractmethod
    def _calc_eff_DOS(self, T: float):
        """Abstract method to calculate the effective density of states."""
        pass

    @abstractmethod
    def _calc_eff_DOS_derivative(self, T: float):
        """Abstract method to calculate the temperature derivative of the effective density of states."""
        pass

    def get_effective_DOS(self, T: float):
        if T <= 0:
            raise DataError(
                f"Incorrect temperature value ({T}) for the effectve density of states calculation."
            )

        return self._calc_eff_DOS(T)

    def get_effective_DOS_derivative(self, T: float):
        if T <= 0:
            raise DataError(
                f"Incorrect temperature value ({T}) for the effectve density of states calculation."
            )

        return self._calc_eff_DOS_derivative(T)


class ConstantEffectiveDOS(EffectiveDOS):
    """Constant effective density of states model."""

    N: pd.NonNegativeFloat = pd.Field(
        ..., title="Effective DOS", description="Effective density of states", units="cm^(-3)"
    )

    def _calc_eff_DOS(self, T: float):
        return self.N

    def _calc_eff_DOS_derivative(self, T: float):
        return 0.0


class IsotropicEffectiveDOS(EffectiveDOS):
    """Effective density of states model that assumes single valley and isotropic effective mass.
    The model assumes the standard equation for the 3D semiconductor with parabolic energy dispersion:

    .. math::

        \\begin{equation}
             \\mathbf{N_eff} = 2 * (\\m_eff \\m_e \\k_B \\T / (2 \\pi \\hbar^2))^(3/2)
        \\end{equation}
    """

    m_eff: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Effective mass",
        description="Effective mass of the carriers",
        units="Electron mass",
    )

    def _calc_eff_DOS(self, T: float):
        return np.power(self.m_eff * T, 1.5) * DOS_aux_const

    def _calc_eff_DOS_derivative(self, T: float):
        return self._calc_eff_DOS(T) * 1.5 / T


class MultiValleyEffectiveDOS(EffectiveDOS):
    """Effective density of states model that assumes multiple valleys and anisotropic effective mass.
    The model assumes the standard equation for the 3D semiconductor with parabolic energy dispersion:

    .. math::

        \\begin{equation}
             \\mathbf{N_eff} = 2 * \\N_valley (\\m_eff_long * \\m_eff_trans * \\m_eff_trans)^(1/2) (\\m_e \\k_B \\T / (2 \\pi \\hbar^2))^(3/2)
        \\end{equation}
    """

    m_eff_long: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Longitudinal effective mass",
        description="Effective mass of the carriers in the longitudinal direction",
        units="Electron mass",
    )

    m_eff_trans: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Longitudinal effective mass",
        description="Effective mass of the carriers in the transverse direction",
        units="Electron mass",
    )

    N_valley: pd.NonNegativeInt = pd.Field(
        ..., title="Nnmber of valleys", description="Number of valleys in the energy band"
    )

    def _calc_eff_DOS(self, T: float):
        return (
            self.N_valley
            * np.power(self.m_eff_long * self.m_eff_trans * self.m_eff_trans, 0.5)
            * np.power(T, 1.5)
            * DOS_aux_const
        )

    def _calc_eff_DOS_derivative(self, T: float):
        return self._calc_eff_DOS(T) * 1.5 / T


class DualValleyEffectiveDOS(EffectiveDOS):
    """Effective density of states model that assumes combibation of light holes and heavy holes with isotropic effective masses.
    The model assumes the standard equation for the 3D semiconductor with parabolic energy dispersion:

    .. math::

        \\begin{equation}
             \\mathbf{N_eff} = 2 * ( (\\m_eff_lh \\m_e \\k_B \\T / (2 \\pi \\hbar^2))^(3/2) + (\\m_eff_hh \\m_e \\k_B \\T / (2 \\pi \\hbar^2))^(3/2) )
        \\end{equation}
    """

    m_eff_lh: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Light hole effective mass",
        description="Effective mass of the light holes",
        units="Electron mass",
    )

    m_eff_hh: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Heavy hole effective mass",
        description="Effective mass of the heavy holes",
        units="Electron mass",
    )

    def _calc_eff_DOS(self, T: float):
        return (np.power(self.m_eff_lh * T, 1.5) + np.power(self.m_eff_hh * T, 1.5)) * DOS_aux_const

    def _calc_eff_DOS_derivative(self, T: float):
        return self._calc_eff_DOS(T) * 1.5 / T
