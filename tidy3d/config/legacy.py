"""Legacy compatibility layer for tidy3d.config.

This module holds (most) of the compatibility layer to the pre-2.10 tidy3d config
and is intended to be removed in a future release.
"""

from __future__ import annotations

import os
import ssl
import warnings
from pathlib import Path
from typing import Any, Optional

import toml

from tidy3d.log import log

from .manager import ConfigManager, normalize_profile_name
from .profiles import BUILTIN_PROFILES


def _warn_env_deprecated() -> None:
    message = "'tidy3d.config.Env' is deprecated; use 'config.switch_profile(...)' instead."
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    log.warning(message, log_once=True)


class LegacyConfigWrapper:
    """Provide attribute-level compatibility with the legacy config module."""

    def __init__(self, manager: ConfigManager):
        self._manager = manager
        self._frozen = False  # retained for backwards compatibility tests

    @property
    def logging_level(self):
        return self._manager.get_section("logging").level

    @logging_level.setter
    def logging_level(self, value):
        from warnings import warn

        warn(
            "'config.logging_level' is deprecated; use 'config.logging.level' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._manager.update_section("logging", level=value)

    @property
    def log_suppression(self):
        return self._manager.get_section("logging").suppression

    @log_suppression.setter
    def log_suppression(self, value):
        from warnings import warn

        warn(
            "'config.log_suppression' is deprecated; use 'config.logging.suppression'.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._manager.update_section("logging", suppression=value)

    @property
    def use_local_subpixel(self):
        return self._manager.get_section("simulation").use_local_subpixel

    @use_local_subpixel.setter
    def use_local_subpixel(self, value):
        from warnings import warn

        warn(
            "'config.use_local_subpixel' is deprecated; use 'config.simulation.use_local_subpixel'.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._manager.update_section("simulation", use_local_subpixel=value)

    @property
    def suppress_rf_license_warning(self):
        return self._manager.get_section("microwave").suppress_rf_license_warning

    @suppress_rf_license_warning.setter
    def suppress_rf_license_warning(self, value):
        from warnings import warn

        warn(
            "'config.suppress_rf_license_warning' is deprecated; "
            "use 'config.microwave.suppress_rf_license_warning'.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._manager.update_section("microwave", suppress_rf_license_warning=value)

    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def frozen(self, value):
        self._frozen = bool(value)

    def save(self, include_defaults: bool = False):
        self._manager.save(include_defaults=include_defaults)

    def reset_manager(self, manager: ConfigManager) -> None:
        """Swap the underlying manager instance."""

        self._manager = manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        elif name in {
            "logging_level",
            "log_suppression",
            "use_local_subpixel",
            "suppress_rf_license_warning",
            "frozen",
        }:
            prop = getattr(type(self), name)
            prop.fset(self, value)
        else:
            setattr(self._manager, name, value)

    def __str__(self) -> str:
        return self._manager.format()


class LegacyEnvironmentConfig:
    """Backward compatible environment config wrapper."""

    def __init__(
        self,
        manager: Optional[ConfigManager] = None,
        name: Optional[str] = None,
        *,
        web_api_endpoint: Optional[str] = None,
        website_endpoint: Optional[str] = None,
        s3_region: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        enable_caching: Optional[bool] = None,
        ssl_version: Optional[ssl.TLSVersion] = None,
        env_vars: Optional[dict[str, str]] = None,
        environment: Optional[LegacyEnvironment] = None,
    ) -> None:
        if name is None:
            raise ValueError("Environment name is required")
        name = normalize_profile_name(name)
        self._manager = manager
        self._name = name
        self._environment = environment
        self._overrides: dict[str, Any] = {}
        if web_api_endpoint is not None:
            self._overrides["api_endpoint"] = web_api_endpoint
        if website_endpoint is not None:
            self._overrides["website_endpoint"] = website_endpoint
        if s3_region is not None:
            self._overrides["s3_region"] = s3_region
        if ssl_verify is not None:
            self._overrides["ssl_verify"] = ssl_verify
        if enable_caching is not None:
            self._overrides["enable_caching"] = enable_caching
        if ssl_version is not None:
            self._overrides["ssl_version"] = ssl_version
        if env_vars is not None:
            self._overrides["env_vars"] = dict(env_vars)

    @property
    def manager(self) -> Optional[ConfigManager]:
        return self._manager

    def active(self) -> None:
        _warn_env_deprecated()
        if self._manager is not None and self._manager.profile != self._name:
            self._manager.switch_profile(self._name)

        environment = self._environment
        if environment is None:
            from tidy3d.config import Env  # local import to avoid circular

            environment = Env

        environment.set_current(self)

    @property
    def web_api_endpoint(self) -> Optional[str]:
        value = self._value("api_endpoint")
        return _maybe_str(value)

    @property
    def website_endpoint(self) -> Optional[str]:
        value = self._value("website_endpoint")
        return _maybe_str(value)

    @property
    def s3_region(self) -> Optional[str]:
        return self._value("s3_region")

    @property
    def ssl_verify(self) -> bool:
        value = self._value("ssl_verify")
        if value is None:
            return True
        return bool(value)

    @property
    def enable_caching(self) -> bool:
        value = self._value("enable_caching")
        if value is None:
            return True
        return bool(value)

    @enable_caching.setter
    def enable_caching(self, value: bool) -> None:
        self._overrides["enable_caching"] = value
        if self._manager and self._manager.profile == self._name:
            self._manager.update_section("web", enable_caching=value)

    @property
    def ssl_version(self):
        return self._value("ssl_version")

    @property
    def env_vars(self):
        value = self._value("env_vars")
        if value is None:
            return {}
        return dict(value)

    @env_vars.setter
    def env_vars(self, value: dict[str, str]) -> None:
        self._overrides["env_vars"] = dict(value)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = normalize_profile_name(value)

    def get_real_url(self, path: str) -> str:
        endpoint = self.web_api_endpoint or ""
        return "/".join([endpoint.rstrip("/"), path.lstrip("/")])

    @property
    def _web_section(self):
        section = {}
        if self._manager is not None:
            if self._manager.profile == self._name:
                source = self._manager.as_dict().get("web", {})
            else:
                source = self._manager.preview_profile(self._name).get("web", {})
            if isinstance(source, dict):
                section.update(source)
        for key, value in self._overrides.items():
            if value is not None:
                section[key] = value
        return section

    def _value(self, key: str) -> Any:
        if key in self._overrides and self._overrides[key] is not None:
            return self._overrides[key]
        return self._web_section.get(key)


class LegacyEnvironment:
    """Legacy Env wrapper that maps to profiles."""

    def __init__(self, manager: ConfigManager):
        self._previous_env_vars: dict[str, Optional[str]] = {}
        self.reset_manager(manager)

    def reset_manager(self, manager: ConfigManager) -> None:
        self._manager = manager
        self.env_map: dict[str, LegacyEnvironmentConfig] = {}
        for name in BUILTIN_PROFILES:
            self.env_map[name] = LegacyEnvironmentConfig(manager, name, environment=self)

        desired_env = os.getenv("TIDY3D_ENV")
        if desired_env:
            desired = normalize_profile_name(desired_env)
        else:
            desired = manager.profile

        if desired == "default":
            desired = "prod"

        desired = normalize_profile_name(desired)

        self._current = self.env_map.setdefault(
            desired, LegacyEnvironmentConfig(manager, desired, environment=self)
        )
        self._apply_env_vars(self._current)

    @property
    def current(self) -> LegacyEnvironmentConfig:
        return self._current

    def set_current(self, env_config: LegacyEnvironmentConfig) -> None:
        _warn_env_deprecated()
        key = normalize_profile_name(env_config.name)
        if env_config.manager is self._manager:
            if self._manager.profile != key:
                self._manager.switch_profile(key)
            stored = self.env_map.setdefault(key, env_config)
        else:
            stored = env_config
            stored.name = key
            self.env_map[key] = stored

        stored._environment = self
        self._current = stored
        self._apply_env_vars(stored)

    def enable_caching(self, enable_caching: bool = True) -> None:
        if self._current.manager is self._manager:
            self._manager.update_section("web", enable_caching=enable_caching)
        self._current.enable_caching = enable_caching

    def set_ssl_version(self, ssl_version) -> None:
        if self._current.manager is self._manager:
            self._manager.update_section("web", ssl_version=ssl_version)
        self._current._overrides["ssl_version"] = ssl_version

    def __getattr__(self, name: str) -> LegacyEnvironmentConfig:
        key = normalize_profile_name(name)
        return self.env_map.setdefault(key, LegacyEnvironmentConfig(self._manager, key))

    def _apply_env_vars(self, config: LegacyEnvironmentConfig) -> None:
        self._restore_env_vars()
        env_vars = config.env_vars or {}
        self._previous_env_vars = {}
        for key, value in env_vars.items():
            self._previous_env_vars[key] = os.environ.get(key)
            os.environ[key] = value

    def _restore_env_vars(self) -> None:
        for key, previous in self._previous_env_vars.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self._previous_env_vars = {}


def _maybe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def load_legacy_flat_config(config_dir: Path) -> dict[str, Any]:
    """Load legacy flat configuration file (pre-migration format)."""

    legacy_path = config_dir / "config"
    if not legacy_path.exists():
        return {}

    try:
        text = legacy_path.read_text(encoding="utf-8")
    except Exception as exc:
        log.warning(f"Failed to read legacy configuration file '{legacy_path}': {exc}")
        return {}

    try:
        parsed = toml.loads(text)
    except Exception as exc:
        log.warning(f"Failed to decode legacy configuration file '{legacy_path}': {exc}")
        return {}

    legacy_data: dict[str, Any] = {}
    apikey = parsed.get("apikey")
    if apikey is not None:
        legacy_data.setdefault("web", {})["apikey"] = apikey
    return legacy_data


__all__ = [
    "LegacyConfigWrapper",
    "LegacyEnvironment",
    "LegacyEnvironmentConfig",
    "finalize_legacy_migration",
    "load_legacy_flat_config",
]


def finalize_legacy_migration(config_dir: Path) -> None:
    """Promote a copied legacy configuration tree into the structured format.

    Parameters
    ----------
    config_dir : Path
        Destination directory (typically the canonical config location).
    """

    legacy_data = load_legacy_flat_config(config_dir)

    from .manager import ConfigManager  # local import to avoid circular dependency

    manager = ConfigManager(profile="default", config_dir=config_dir)
    config_path = config_dir / "config.toml"
    for section, values in legacy_data.items():
        if isinstance(values, dict):
            manager.update_section(section, **values)
    try:
        manager.save(include_defaults=True)
    except Exception:
        if config_path.exists():
            try:
                config_path.unlink()
            except Exception:
                pass
        raise

    legacy_flat_path = config_dir / "config"
    if legacy_flat_path.exists():
        try:
            legacy_flat_path.unlink()
        except Exception as exc:
            log.warning(f"Failed to remove legacy configuration file '{legacy_flat_path}': {exc}")
