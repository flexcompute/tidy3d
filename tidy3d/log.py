"""Compatibility shim for :mod:`tidy3d._common.log`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.log import (
    CONSOLE_WIDTH,
    DEFAULT_LEVEL,
    DEFAULT_LOG_STYLES,
    Logger,
    LogHandler,
    LogLevel,
    LogValue,
    NoOpProgress,
    Progress,
    _default_log_level_format,
    _get_level_int,
    _level_name,
    _level_value,
    get_aware_datetime,
    get_logging_console,
    log,
    set_log_suppression,
    set_logging_console,
    set_logging_file,
    set_logging_level,
    set_warn_once,
)
