"""Compatibility shim for :mod:`tidy3d._common.components.geometry.primitives`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.geometry.primitives import (
    _DEFAULT_EDGE_FRACTION,
    _MAX_ICOSPHERE_SUBDIVISIONS,
    _N_PTS_CYLINDER_POLYSLAB,
    _N_SAMPLE_CURVE_SHAPELY,
    _N_SHAPELY_QUAD_SEGS_VISUALIZATION,
    UNIT_SPHERE,
    Cylinder,
    Sphere,
    _base_icosahedron,
)
