"""Compatibility shim for :mod:`tidy3d._common.config.serializer`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.config.serializer import (
    Path,
    _apply_value,
    _describe_field,
    _iter_model_types,
    _prune_missing_keys,
    build_document,
    collect_descriptions,
)
