from __future__ import annotations

from typing import Union

from tidy3d.components.microwave.path_integrals.current_spec import (
    CompositeCurrentIntegralSpec,
    CurrentIntegralAxisAlignedSpec,
    CustomCurrentIntegral2DSpec,
)
from tidy3d.components.microwave.path_integrals.voltage_spec import (
    CustomVoltageIntegral2DSpec,
    VoltageIntegralAxisAlignedSpec,
)

VoltagePathSpecTypes = Union[VoltageIntegralAxisAlignedSpec, CustomVoltageIntegral2DSpec]
CurrentPathSpecTypes = Union[
    CurrentIntegralAxisAlignedSpec, CustomCurrentIntegral2DSpec, CompositeCurrentIntegralSpec
]
