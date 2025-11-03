"""Tidy3D configuration system public API."""

from __future__ import annotations

from typing import Any

from . import sections  # noqa: F401 - ensure builtin sections register
from .legacy import LegacyConfigWrapper, LegacyEnvironment, LegacyEnvironmentConfig
from .manager import ConfigManager
from .registry import (
    get_handlers,
    get_sections,
    register_handler,
    register_plugin,
    register_section,
)

__all__ = [
    "ConfigManager",
    "Env",
    "Environment",
    "EnvironmentConfig",
    "config",
    "get_handlers",
    "get_sections",
    "register_handler",
    "register_plugin",
    "register_section",
]


def _create_manager() -> ConfigManager:
    return ConfigManager()


_base_manager = _create_manager()
# TODO(FXC-3827): Drop LegacyConfigWrapper once legacy accessors are removed in Tidy3D 2.12.
_config_wrapper = LegacyConfigWrapper(_base_manager)
config = _config_wrapper

# TODO(FXC-3827): Remove legacy Env exports after deprecation window (planned 2.12).
Environment = LegacyEnvironment
EnvironmentConfig = LegacyEnvironmentConfig
Env = LegacyEnvironment(_base_manager)


def reload_config(*, profile: str | None = None) -> LegacyConfigWrapper:
    """Recreate the global configuration manager (primarily for tests)."""

    global _base_manager, Env
    if _base_manager is not None:
        try:
            _base_manager.apply_web_env({})
        except AttributeError:
            pass
    _base_manager = ConfigManager(profile=profile)
    _config_wrapper.reset_manager(_base_manager)
    Env.reset_manager(_base_manager)
    return _config_wrapper


def get_manager() -> ConfigManager:
    """Return the underlying configuration manager instance."""

    return _base_manager


def __getattr__(name: str) -> Any:
    return getattr(config, name)
