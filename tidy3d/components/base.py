"""Compatibility shim for :mod:`tidy3d._common.components.base`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.base import (
    FORBID_SPECIAL_CHARACTERS,
    INDENT,
    INDENT_JSON_FILE,
    JSON_TAG,
    MAX_STRING_LENGTH,
    TRACED_FIELD_KEYS_ATTR,
    TYPE_TO_CLASS_MAP,
    T,
    Tidy3dBaseModel,
    _CacheReturn,
    _fmt_ann_literal,
    _get_valid_extension,
    _GuardedReturn,
    _make_lazy_proxy,
    cache,
    cached_property,
    cached_property_guarded,
    make_json_compatible,
)
