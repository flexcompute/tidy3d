from __future__ import annotations

from .ac import SSACVoltageSource
from .dc import DCCurrentSource, DCVoltageSource, GroundVoltage

VoltageSourceType = DCVoltageSource | SSACVoltageSource | GroundVoltage
CurrentSourceType = DCCurrentSource
