"""Shared constants for base geometry classes."""

from __future__ import annotations

POLY_GRID_SIZE = 1e-12
POLY_TOLERANCE_RATIO = 1e-12
POLY_DISTANCE_TOLERANCE = 8e-12
GDS_MAX_COORDINATE_INDEX = 2**31 - 1

__all__ = [
    "GDS_MAX_COORDINATE_INDEX",
    "POLY_DISTANCE_TOLERANCE",
    "POLY_GRID_SIZE",
    "POLY_TOLERANCE_RATIO",
]
