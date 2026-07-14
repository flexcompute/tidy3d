"""Tidy3D configuration system public API."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from tidy3d.log import log

from . import sections  # noqa: F401 - ensure builtin sections register
from .manager import ConfigManager
from .migrations import CURRENT_CONFIG_VERSION, register_migration
from .registry import (
    get_handlers,
    get_sections,
    register_handler,
    register_plugin,
    register_section,
)

__all__ = [
    "CURRENT_CONFIG_VERSION",
    "ConfigManager",
    "config",
    "get_handlers",
    "get_sections",
    "register_handler",
    "register_migration",
    "register_plugin",
    "register_section",
]

_REMOVED_ACCESSORS = {
    "Env": "`config.switch_profile(...)` and `config.profile`",
    "Environment": "`config.switch_profile(...)` and `config.profile`",
    "EnvironmentConfig": "`config.web`",
    "frozen": None,
    "logging_level": "`config.logging.level`",
    "log_suppression": "`config.logging.suppression`",
    "suppress_rf_license_warning": "`config.microwave.suppress_rf_license_warning`",
    "use_local_subpixel": "`config.simulation.use_local_subpixel`",
}


def _create_manager(*, profile: str | None = None, allow_fallback: bool = True) -> ConfigManager:
    try:
        return ConfigManager(profile=profile)
    except Exception as exc:
        if not allow_fallback:
            raise
        from .loader import _temporary_config_dir

        fallback_dir = _temporary_config_dir()
        log.warning(
            "Failed to initialize configuration from the active config directory: "
            f"{exc}. Falling back to temporary configuration at '{fallback_dir}'."
        )
        return ConfigManager(profile=profile, config_dir=fallback_dir)


def _removed_accessor_message(name: str) -> str:
    replacement = _REMOVED_ACCESSORS[name]
    message = f"`tidy3d.config.{name}` was removed in Tidy3D 2.12."
    if replacement is None:
        return f"{message} No replacement is available."
    return f"{message} This functionality has moved to {replacement}."


def _raise_removed_accessor(name: str) -> None:
    message = _removed_accessor_message(name)
    log.warning(message, log_once=True)
    raise AttributeError(message)


_REMOVED_IMPORT_FINDER_MARKER = "_tidy3d_removed_config_accessor_module"


class _RemovedConfigAccessorImportFinder:
    """Raise tailored errors for removed names imported as package children."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        setattr(self, _REMOVED_IMPORT_FINDER_MARKER, module_name)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        prefix = f"{self.module_name}."
        if not fullname.startswith(prefix):
            return None

        name = fullname.removeprefix(prefix)
        if "." in name or name not in _REMOVED_ACCESSORS:
            return None

        message = _removed_accessor_message(name)
        log.warning(message, log_once=True)
        raise ImportError(message)


class _ConfigModule(ModuleType):
    """Module type that prevents removed public names from being recreated."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _REMOVED_ACCESSORS:
            _raise_removed_accessor(name)
        super().__setattr__(name, value)


def _install_removed_accessor_import_finder() -> None:
    for finder in sys.meta_path:
        if getattr(finder, _REMOVED_IMPORT_FINDER_MARKER, None) == __name__:
            return
    sys.meta_path.insert(0, _RemovedConfigAccessorImportFinder(__name__))


class _ConfigProxy:
    """Stable public config object backed by the active ConfigManager."""

    def __init__(self, manager: ConfigManager) -> None:
        object.__setattr__(self, "_manager", manager)

    def reset_manager(self, manager: ConfigManager) -> None:
        object.__setattr__(self, "_manager", manager)

    def __getattr__(self, name: str) -> Any:
        if name in _REMOVED_ACCESSORS:
            _raise_removed_accessor(name)
        return getattr(self._manager, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in _REMOVED_ACCESSORS:
            _raise_removed_accessor(name)
        if name in self._manager._section_models:
            setattr(self._manager, name, value)
            return
        raise AttributeError(f"Config has no section '{name}'")

    def __str__(self) -> str:
        return str(self._manager)

    def __enter__(self) -> _ConfigProxy:
        self._manager.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._manager.__exit__(exc_type, exc_value, traceback)


_base_manager = _create_manager()
config = _ConfigProxy(_base_manager)


def reload_config(*, profile: str | None = None) -> _ConfigProxy:
    """Recreate the global configuration manager (primarily for tests)."""

    global _base_manager
    if _base_manager is not None:
        try:
            _base_manager.apply_web_env({})
        except AttributeError:
            pass
    _base_manager = _create_manager(profile=profile, allow_fallback=False)
    config.reset_manager(_base_manager)
    return config


def get_manager() -> ConfigManager:
    """Return the underlying configuration manager instance."""

    return _base_manager


def __getattr__(name: str) -> Any:
    if name in _REMOVED_ACCESSORS:
        _raise_removed_accessor(name)
    return getattr(config, name)


_install_removed_accessor_import_finder()
sys.modules[__name__].__class__ = _ConfigModule
