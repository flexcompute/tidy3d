"""Compatibility shim for :mod:`tidy3d._common.config`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

import tidy3d._common.config as _common_config
from tidy3d.config import sections

_common_config.initialize_env()

from tidy3d._common.config import (  # noqa: E402 - import after Env setup
    ConfigManager,
    Env,
    Environment,
    EnvironmentConfig,
    LegacyConfigWrapper,
    LegacyEnvironment,
    LegacyEnvironmentConfig,
    _base_manager,
    _config_wrapper,
    _create_manager,
    config,
    get_handlers,
    get_manager,
    get_sections,
    register_handler,
    register_plugin,
    register_section,
    reload_config,
)
