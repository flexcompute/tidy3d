"""Compatibility shim for :mod:`tidy3d._common.components.geometry.triangulation`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.geometry.triangulation import (
    Vertex,
    is_inside,
    triangulate,
    update_convexity,
    update_ear_flag,
)
