"""Compatibility shim for :mod:`tidy3d._common.components.geometry.base`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.geometry.base import (
    POLY_DISTANCE_TOLERANCE,
    POLY_GRID_SIZE,
    POLY_TOLERANCE_RATIO,
    Box,
    Centered,
    Circular,
    ClipOperation,
    Geometry,
    GeometryGroup,
    Planar,
    SimplePlaneIntersection,
    Transformed,
    _bit_operations,
    _shapely_operations,
    cleanup_shapely_object,
)
