from __future__ import annotations

from typing import Union

from .dc import DCCurrentSource, DCVoltageSource

VoltageSourceType = Union[DCVoltageSource]
CurrentSourceType = Union[DCCurrentSource]
