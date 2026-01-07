"""Compatibility shim for :mod:`tidy3d._common.compat`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.compat import (
    Self,
    TypeAlias,
    _package_is_older_than,
    alignment,
)
