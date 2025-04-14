"""Backwards compatibility - import from tidy3d.components.microwave.path_integrals.integrals instead."""

from __future__ import annotations

from tidy3d.components.microwave.path_integrals.integrals.base import (
    AxisAlignedPathIntegral,
    EMScalarFieldType,
    IntegrableMonitorDataType,
)
from tidy3d.components.microwave.path_integrals.integrals.current import (
    AxisAlignedCurrentIntegral,
)
from tidy3d.components.microwave.path_integrals.integrals.voltage import (
    AxisAlignedVoltageIntegral,
)

__all__ = [
    "AxisAlignedCurrentIntegral",
    "AxisAlignedPathIntegral",
    "AxisAlignedVoltageIntegral",
    "EMScalarFieldType",
    "IntegrableMonitorDataType",
]
