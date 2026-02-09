"""Compatibility shim for :mod:`tidy3d._common.components.autograd.field_map`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.autograd.field_map import (
    FieldMap,
    Tracer,
    TracerKeys,
    _encoded_path,
)
