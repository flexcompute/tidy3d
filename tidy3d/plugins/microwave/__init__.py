"""Imports from microwave plugin."""

from __future__ import annotations

from tidy3d.components.microwave.impedance_calculator import (
    CurrentIntegralType,
    ImpedanceCalculator,
    VoltageIntegralType,
)
from tidy3d.components.microwave.path_integrals.integrals.auto import (
    path_integrals_from_lumped_element,
)
from tidy3d.components.microwave.path_integrals.integrals.base import (
    AxisAlignedPathIntegral,
    Custom2DPathIntegral,
)
from tidy3d.components.microwave.path_integrals.integrals.current import (
    AxisAlignedCurrentIntegral,
    CompositeCurrentIntegral,
    Custom2DCurrentIntegral,
)
from tidy3d.components.microwave.path_integrals.integrals.voltage import (
    AxisAlignedVoltageIntegral,
    Custom2DVoltageIntegral,
)

from . import models
from .array_factor import (
    BlackmanHarrisWindow,
    BlackmanWindow,
    ChebWindow,
    HammingWindow,
    HannWindow,
    KaiserWindow,
    RadialTaper,
    RectangularAntennaArrayCalculator,
    RectangularTaper,
    TaylorWindow,
)
from .lobe_measurer import LobeMeasurer
from .rf_material_library import rf_material_library

# Backwards compatibility
CurrentIntegralTypes = CurrentIntegralType
VoltageIntegralTypes = VoltageIntegralType

__all__ = [
    "AxisAlignedCurrentIntegral",
    "AxisAlignedPathIntegral",
    "AxisAlignedVoltageIntegral",
    "BlackmanHarrisWindow",
    "BlackmanWindow",
    "ChebWindow",
    "CompositeCurrentIntegral",
    "CurrentIntegralTypes",
    "Custom2DCurrentIntegral",
    "Custom2DPathIntegral",
    "Custom2DVoltageIntegral",
    "HammingWindow",
    "HannWindow",
    "ImpedanceCalculator",
    "KaiserWindow",
    "LobeMeasurer",
    "RadialTaper",
    "RectangularAntennaArrayCalculator",
    "RectangularTaper",
    "TaylorWindow",
    "VoltageIntegralTypes",
    "models",
    "path_integrals_from_lumped_element",
    "rf_material_library",
]
