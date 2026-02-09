"""Compatibility shim for :mod:`tidy3d._common.config.registry`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.config.registry import (
    _HANDLERS,
    _MANAGER,
    _SECTIONS,
    ConfigManagerProtocol,
    T,
    attach_manager,
    get_handlers,
    get_manager,
    get_sections,
    register_handler,
    register_plugin,
    register_section,
)
