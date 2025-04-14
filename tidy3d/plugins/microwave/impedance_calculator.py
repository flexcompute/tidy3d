"""Backwards compatibility - import from tidy3d.components.microwave instead."""

from __future__ import annotations

from tidy3d.components.microwave.impedance_calculator import (
    CurrentIntegralType,
    ImpedanceCalculator,
    VoltageIntegralType,
)

__all__ = [
    "CurrentIntegralType",
    "ImpedanceCalculator",
    "VoltageIntegralType",
]
