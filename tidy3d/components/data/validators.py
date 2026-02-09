"""Compatibility shim for :mod:`tidy3d._common.components.data.validators`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.data.validators import (
    validate_can_interpolate,
    validate_no_nans,
)
