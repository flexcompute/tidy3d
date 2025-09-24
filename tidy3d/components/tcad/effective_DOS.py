from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import HBAR, K_B, M_E_EV, PERCMCUBE
from tidy3d.exceptions import DataError

um_3_to_cm_3 = 1e12  # conversion factor from micron^(-3) to cm^(-3)
DOS_aux_const = 2.0 * np.power((M_E_EV * K_B) / (2 * np.pi * HBAR * HBAR), 1.5) * um_3_to_cm_3


class EffectiveDOS(Tidy3dBaseModel, ABC):
    """Abstract class for the effective density of states"""

    @abstractmethod
    def calc_eff_dos(self, T: float) -> None:
        """Abstract method to calculate the effective density of states."""

    def get_effective_DOS(self, T: float):
        if T <= 0:
            raise DataError(
                f"Incorrect temperature value ({T}) for the effective density of states calculation."
            )

        return self.calc_eff_dos(T)


class ConstantEffectiveDOS(EffectiveDOS):
    """Constant effective density of states model."""

    N: pd.PositiveFloat = pd.Field(
        ..., title="Effective DOS", description="Effective density of states", units=PERCMCUBE
    )

    def calc_eff_dos(self, T: float):
        return self.N


class IsotropicEffectiveDOS(EffectiveDOS):
    """Effective density of states model that assumes single valley and isotropic effective mass.
    The model assumes the standard equation for the 3D semiconductor with parabolic energy dispersion:

    Notes
    -----

    .. math::

        \\mathbf{N_eff} = 2 * (\\frac{m_eff * m_e * k_B T}{2 \\pi \\hbar^2})^(3/2)

    """

    m_eff: pd.PositiveFloat = pd.Field(
        ...,
        title="Effective mass",
        description="Effective mass of the carriers relative to the electron mass at rest",
    )

    def calc_eff_dos(self, T: float):
        return np.power(self.m_eff * T, 1.5) * DOS_aux_const


class MultiValleyEffectiveDOS(EffectiveDOS):
    """Effective density of states model that assumes multiple equivalent valleys and anisotropic effective mass.
    The model assumes the standard equation for the 3D semiconductor with parabolic energy dispersion:

    Notes
    -----

    .. math::

       N_{\\text{eff}} = 2 N_{\\text{valley}} \\left( m_{\\text{eff,long}} m_{\\text{eff,trans}}^2 \\right)^{1/2} \\left( \\frac{m_e k_B T}{2 \\pi \\hbar^2} \\right)^{3/2}

    """

    m_eff_long: pd.PositiveFloat = pd.Field(
        ...,
        title="Longitudinal effective mass",
        description="Relative effective mass of the carriers in the longitudinal direction. This is a relative value compared to the electron mass at rest.",
    )

    m_eff_trans: pd.PositiveFloat = pd.Field(
        ...,
        title="Transverse effective mass",
        description="Relative effective mass of the carriers in the transverse direction. This is a relative value compared to the electron mass at rest.",
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


class DualValleyEffectiveDOS(EffectiveDOS):
    """Effective density of states model that assumes combination of light holes and heavy holes with isotropic effective masses.
    The model assumes the standard equation for the 3D semiconductor with parabolic energy dispersion:

    Notes
    -----

    .. math::

       N_{eff} = 2 \\left( \\frac{m_{\\text{eff, lh}} m_e k_B T}{2 \\pi \\hbar^2} \\right)^{3/2} + 2 \\left( \\frac{m_{\\text{eff, hh}} m_e k_B T}{2 \\pi \\hbar^2} \\right)^{3/2}

    """

    m_eff_lh: pd.PositiveFloat = pd.Field(
        ...,
        title="Light hole effective mass",
        description="Relative effective mass of the light holes. This is a relative value compared to the electron mass at rest.",
    )

    m_eff_hh: pd.PositiveFloat = pd.Field(
        ...,
        title="Heavy hole effective mass",
        description="Relative effective mass of the heavy holes. This is a relative value compared to the electron mass at rest.",
    )

    def calc_eff_dos(self, T: float):
        return (np.power(self.m_eff_lh * T, 1.5) + np.power(self.m_eff_hh * T, 1.5)) * DOS_aux_const
