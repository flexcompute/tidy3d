"""Compatibility shim for :mod:`tidy3d._common.components.autograd.functions`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.autograd.functions import (
    _add_at,
    _evaluate_linear,
    _evaluate_nearest,
    add_at,
    interpn,
    trapz,
)
