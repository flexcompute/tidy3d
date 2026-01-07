"""Compatibility shim for :mod:`tidy3d._common.components.autograd.boxes`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.autograd.boxes import (
    TidyArrayBox,
    _autograd_module_cache,
    from_arraybox,
    item,
)
