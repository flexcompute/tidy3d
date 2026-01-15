from __future__ import annotations

from tidy3d.components.data.data_array import FreqModeDataArray
from tidy3d.constants import NEPERPERMETER, PERMETER, RADPERMETER, VELOCITY_SI


class PropagationConstantArray(FreqModeDataArray):
    """Data array for the complex propagation constant :math:`\\gamma = -\\alpha + j\\beta` with units of 1/m.

    Notes
    -----
        In the physics convention where time-harmonic fields evolve with :math:`e^{-j\\omega t}`, a wave
        propagating in the +z direction varies as :math:`E(z) = E_0 e^{\\gamma z} = E_0 e^{-\\alpha z} e^{j\\beta z}`.
    """

    __slots__ = ()
    _data_attrs = {"units": PERMETER, "long_name": "propagation constant"}


class PhaseConstantArray(FreqModeDataArray):
    """Data array for the phase constant :math:`\\beta = \\text{Im}(\\gamma)` with units of rad/m."""

    __slots__ = ()
    _data_attrs = {"units": RADPERMETER, "long_name": "phase constant"}


class AttenuationConstantArray(FreqModeDataArray):
    """Data array for the attenuation constant :math:`\\alpha = -\\text{Re}(\\gamma)` with units of Nepers/m."""

    __slots__ = ()
    _data_attrs = {"units": NEPERPERMETER, "long_name": "attenuation constant"}


class PhaseVelocityArray(FreqModeDataArray):
    """Data array for the phase velocity :math:`v_p = c/n_{\\mathrm{eff}}` with units of m/s."""

    __slots__ = ()
    _data_attrs = {"units": VELOCITY_SI, "long_name": "phase velocity"}


class GroupVelocityArray(FreqModeDataArray):
    """Data array for the group velocity :math:`v_g = c/n_{\\mathrm{group}}` with units of m/s."""

    __slots__ = ()
    _data_attrs = {"units": VELOCITY_SI, "long_name": "group velocity"}
