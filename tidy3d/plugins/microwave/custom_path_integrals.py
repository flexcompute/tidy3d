"""Backwards compatibility - import from tidy3d.components.microwave.path_integrals.integrals instead."""

from __future__ import annotations

from tidy3d.components.data.data_array import (
    CurrentIntegralResultType,
    IntegralResultType,
    VoltageIntegralResultType,
    _make_base_result_data_array,
    _make_current_data_array,
    _make_voltage_data_array,
)
from tidy3d.components.microwave.path_integrals.integrals.base import (
    Custom2DPathIntegral,
)
from tidy3d.components.microwave.path_integrals.integrals.current import (
    CompositeCurrentIntegral,
    Custom2DCurrentIntegral,
)
from tidy3d.components.microwave.path_integrals.integrals.voltage import (
    Custom2DVoltageIntegral,
)

__all__ = [
    "CompositeCurrentIntegral",
    "CurrentIntegralResultType",
    "Custom2DCurrentIntegral",
    "Custom2DPathIntegral",
    "Custom2DVoltageIntegral",
    "IntegralResultType",
    "VoltageIntegralResultType",
    "_make_base_result_data_array",
    "_make_current_data_array",
    "_make_voltage_data_array",
]
