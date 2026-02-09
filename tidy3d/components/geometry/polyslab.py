"""Compatibility shim for :mod:`tidy3d._common.components.geometry.polyslab`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.geometry.polyslab import (
    _COMPLEX_POLYSLAB_DIVISIONS_WARN,
    _IS_CLOSE_RTOL,
    _MAX_POLYSLAB_VERTICES_FOR_TRIANGULATION,
    _MIN_POLYGON_AREA,
    _N_SAMPLE_POLYGON_INTERSECT,
    ComplexPolySlabBase,
    PolySlab,
    leggauss,
)
