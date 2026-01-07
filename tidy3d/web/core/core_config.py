"""Compatibility shim for :mod:`tidy3d._common.web.core.core_config`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.web.core.core_config import (
    config_setting,
    get_logger,
    get_logger_console,
    get_version,
    set_config,
)
