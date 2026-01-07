"""Compatibility shim for :mod:`tidy3d._common.components.autograd.derivative_utils`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.autograd.derivative_utils import (
    ArrayComplex,
    ArrayFloat,
    DerivativeInfo,
    EpsType,
    FieldData,
    LazyInterpolator,
    PermittivityData,
    integrate_within_bounds,
)
