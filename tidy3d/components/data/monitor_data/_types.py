"""Shared type aliases for monitor-data implementations."""

from __future__ import annotations

from typing import TypeVar

from tidy3d.components.data.data_array import (
    EMEFreqModeDataArray,
    FreqDataArray,
    FreqModeDataArray,
    TimeDataArray,
)
from tidy3d.components.types import ArrayFloat1D

SourceT = TypeVar("SourceT")
DataArrayCoordValue = float | int | str
DataArrayEntry = tuple[tuple[int, ...], tuple[DataArrayCoordValue, ...], complex]
Coords1D = ArrayFloat1D
GRID_CORRECTION_TYPE = (
    float | FreqDataArray | TimeDataArray | FreqModeDataArray | EMEFreqModeDataArray
)

__all__ = [
    "GRID_CORRECTION_TYPE",
    "Coords1D",
    "DataArrayCoordValue",
    "DataArrayEntry",
    "SourceT",
]
