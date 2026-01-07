"""Compatibility shim for :mod:`tidy3d._common.config.loader`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.config.loader import (
    ConfigLoader,
    _assign_path,
    _clean_data,
    _is_writable,
    _merge_into,
    _temporary_config_dir,
    _xdg_config_home,
    canonical_config_directory,
    deep_diff,
    deep_merge,
    legacy_config_directory,
    load_environment_overrides,
    migrate_legacy_config,
    resolve_config_directory,
)
