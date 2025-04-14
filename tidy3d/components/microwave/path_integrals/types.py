"""Common type definitions for path integral specifications."""

from __future__ import annotations

from typing import Union

from tidy3d.components.microwave.path_integrals.specs.current import (
    AxisAlignedCurrentIntegralSpec,
    CompositeCurrentIntegralSpec,
    Custom2DCurrentIntegralSpec,
)
from tidy3d.components.microwave.path_integrals.specs.voltage import (
    AxisAlignedVoltageIntegralSpec,
    Custom2DVoltageIntegralSpec,
)

VoltagePathSpecType = Union[AxisAlignedVoltageIntegralSpec, Custom2DVoltageIntegralSpec]
CurrentPathSpecType = Union[
    AxisAlignedCurrentIntegralSpec, Custom2DCurrentIntegralSpec, CompositeCurrentIntegralSpec
]
