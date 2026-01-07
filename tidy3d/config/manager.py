"""Compatibility shim for :mod:`tidy3d._common.config.manager`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.config.manager import (
    BUILTIN_PROFILES,
    ConfigManager,
    PluginsAccessor,
    ProfilesAccessor,
    SectionAccessor,
    _build_config_panel,
    _build_section_panel,
    _deep_get,
    _extract_persisted,
    _model_dict,
    _prepare_for_display,
    _render_panel,
    _resolve_model_type,
    _serialize_value,
    normalize_profile_name,
)
