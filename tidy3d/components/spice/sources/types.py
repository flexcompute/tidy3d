from __future__ import annotations

from typing import Union

from .ac import SSACVoltageSource
from .dc import DCCurrentSource, DCVoltageSource, GroundVoltage

VoltageSourceType = Union[DCVoltageSource, SSACVoltageSource, GroundVoltage]
CurrentSourceType = Union[DCCurrentSource]
