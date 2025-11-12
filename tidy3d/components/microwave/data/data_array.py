from __future__ import annotations

from tidy3d.components.data.data_array import FreqModeDataArray
from tidy3d.constants import NEPERPERMETER, PERMETER, RADPERMETER, VELOCITY_SI


class PropagationConstantArray(FreqModeDataArray):
    __slots__ = ()
    _data_attrs = {"units": PERMETER, "long_name": "propagation constant"}


class PhaseConstantArray(FreqModeDataArray):
    __slots__ = ()
    _data_attrs = {"units": RADPERMETER, "long_name": "phase constant"}


class AttenuationConstantArray(FreqModeDataArray):
    __slots__ = ()
    _data_attrs = {"units": NEPERPERMETER, "long_name": "attenuation constant"}


class PhaseVelocityArray(FreqModeDataArray):
    __slots__ = ()
    _data_attrs = {"units": VELOCITY_SI, "long_name": "phase velocity"}


class GroupVelocityArray(FreqModeDataArray):
    __slots__ = ()
    _data_attrs = {"units": VELOCITY_SI, "long_name": "group velocity"}
