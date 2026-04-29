"""Common type definitions for path integral specifications."""

from __future__ import annotations

from tidy3d.components.microwave.path_integrals.specs.current import (
    AxisAlignedCurrentIntegralSpec,
    CompositeCurrentIntegralSpec,
    Custom2DCurrentIntegralSpec,
)
from tidy3d.components.microwave.path_integrals.specs.voltage import (
    AxisAlignedVoltageIntegralSpec,
    Custom2DVoltageIntegralSpec,
)

VoltagePathSpecType = AxisAlignedVoltageIntegralSpec | Custom2DVoltageIntegralSpec
CurrentPathSpecType = (
    AxisAlignedCurrentIntegralSpec | Custom2DCurrentIntegralSpec | CompositeCurrentIntegralSpec
)
