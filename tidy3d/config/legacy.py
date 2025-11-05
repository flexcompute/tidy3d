"""Legacy compatibility layer for tidy3d.config.

This module holds (most) of the compatibility layer to the pre-2.10 tidy3d config
and is intended to be removed in a future release.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Optional

import toml

from tidy3d._runtime import WASM_BUILD
from tidy3d.log import LogLevel, log

# TODO(FXC-3827): Remove LegacyConfigWrapper/Environment shims and related helpers in Tidy3D 2.12.
from .manager import ConfigManager, normalize_profile_name
from .profiles import BUILTIN_PROFILES


def _warn_env_deprecated() -> None:
    message = "'tidy3d.config.Env' is deprecated; use 'config.switch_profile(...)' instead."
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    log.warning(message, log_once=True)


# TODO(FXC-3827): Delete LegacyConfigWrapper once legacy attribute access is dropped.
class LegacyConfigWrapper:
    """Provide attribute-level compatibility with the legacy config module."""

    def __init__(self, manager: ConfigManager):
        self._manager = manager
        self._frozen = False  # retained for backwards compatibility tests

    @property
    def logging_level(self) -> LogLevel:
        return self._manager.get_section("logging").level

    @logging_level.setter
    def logging_level(self, value: LogLevel) -> None:
        from warnings import warn

        warn(
            "'config.logging_level' is deprecated; use 'config.logging.level' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._manager.update_section("logging", level=value)

    @property
    def log_suppression(self) -> bool:
        return self._manager.get_section("logging").suppression

    @log_suppression.setter
    def log_suppression(self, value: bool) -> None:
        from warnings import warn

        warn(
            "'config.log_suppression' is deprecated; use 'config.logging.suppression'.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._manager.update_section("logging", suppression=value)

    @property
    def use_local_subpixel(self) -> Optional[bool]:
        return self._manager.get_section("simulation").use_local_subpixel

    @use_local_subpixel.setter
    def use_local_subpixel(self, value: Optional[bool]) -> None:
        from warnings import warn

        warn(
            "'config.use_local_subpixel' is deprecated; use 'config.simulation.use_local_subpixel'.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._manager.update_section("simulation", use_local_subpixel=value)

    @property
    def suppress_rf_license_warning(self) -> bool:
        return self._manager.get_section("microwave").suppress_rf_license_warning

    @suppress_rf_license_warning.setter
    def suppress_rf_license_warning(self, value: bool) -> None:
        from warnings import warn

        warn(
            "'config.suppress_rf_license_warning' is deprecated; "
            "use 'config.microwave.suppress_rf_license_warning'.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._manager.update_section("microwave", suppress_rf_license_warning=value)

    @property
    def frozen(self) -> bool:
        return self._frozen

    @frozen.setter
    def frozen(self, value: bool) -> None:
        self._frozen = bool(value)

    def save(self, include_defaults: bool = False) -> None:
        self._manager.save(include_defaults=include_defaults)

    def reset_manager(self, manager: ConfigManager) -> None:
        """Swap the underlying manager instance."""

        self._manager = manager

    def switch_profile(self, profile: str) -> None:
        """Switch active profile and synchronize the legacy environment proxy."""

        normalized = normalize_profile_name(profile)
        self._manager.switch_profile(normalized)
        try:
            from tidy3d.config import Env as _legacy_env
        except Exception:
            _legacy_env = None
        if _legacy_env is not None:
            _legacy_env._sync_to_manager(apply_env=True)

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


# TODO(FXC-3827): Delete LegacyEnvironmentConfig once profile-based Env shim is removed.
class LegacyEnvironmentConfig:
    """Backward compatible environment config wrapper that proxies ConfigManager."""

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
        ssl_version: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
        environment: Optional[LegacyEnvironment] = None,
    ) -> None:
        if name is None:
            raise ValueError("Environment name is required")
        self._manager = manager
        self._name = normalize_profile_name(name)
        self._environment = environment
        self._pending: dict[str, Any] = {}
        if web_api_endpoint is not None:
            self._pending["api_endpoint"] = web_api_endpoint
        if website_endpoint is not None:
            self._pending["website_endpoint"] = website_endpoint
        if s3_region is not None:
            self._pending["s3_region"] = s3_region
        if ssl_verify is not None:
            self._pending["ssl_verify"] = ssl_verify
        if enable_caching is not None:
            self._pending["enable_caching"] = enable_caching
        if ssl_version is not None:
            self._pending["ssl_version"] = ssl_version
        if env_vars is not None:
            self._pending["env_vars"] = dict(env_vars)

    def reset_manager(self, manager: ConfigManager) -> None:
        self._manager = manager

    @property
    def manager(self) -> Optional[ConfigManager]:
        if self._manager is not None:
            return self._manager
        if self._environment is not None:
            return self._environment._manager
        return None

    def active(self) -> None:
        _warn_env_deprecated()
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
    def enable_caching(self, value: Optional[bool]) -> None:
        self._set_pending("enable_caching", value)

    @property
    def ssl_version(self) -> Optional[str]:
        return self._value("ssl_version")

    @ssl_version.setter
    def ssl_version(self, value: Optional[str]) -> None:
        self._set_pending("ssl_version", value)

    @property
    def env_vars(self) -> dict[str, str]:
        value = self._value("env_vars")
        if value is None:
            return {}
        return dict(value)

    @env_vars.setter
    def env_vars(self, value: dict[str, str]) -> None:
        self._set_pending("env_vars", dict(value))

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = normalize_profile_name(value)

    def copy_state_from(self, other: LegacyEnvironmentConfig) -> None:
        if not isinstance(other, LegacyEnvironmentConfig):
            raise TypeError("Expected LegacyEnvironmentConfig instance.")
        for key, value in other._pending.items():
            if key == "env_vars" and value is not None:
                self._pending[key] = dict(value)
            else:
                self._pending[key] = value

    def get_real_url(self, path: str) -> str:
        manager = self.manager
        if manager is not None and manager.profile == self._name:
            web_section = manager.get_section("web")
            if hasattr(web_section, "build_api_url"):
                return web_section.build_api_url(path)

        endpoint = self.web_api_endpoint or ""
        if not path:
            return endpoint
        return "/".join([endpoint.rstrip("/"), str(path).lstrip("/")])

    def apply_pending_overrides(self) -> None:
        manager = self.manager
        if manager is None or manager.profile != self._name:
            return
        if not self._pending:
            return
        updates = dict(self._pending)
        manager.update_section("web", **updates)
        self._pending.clear()

    def _set_pending(self, key: str, value: Any) -> None:
        if key == "env_vars" and value is not None:
            self._pending[key] = dict(value)
        else:
            self._pending[key] = value
        self.apply_pending_overrides()

    def _web_section(self) -> dict[str, Any]:
        manager = self.manager
        if manager is None or WASM_BUILD:
            return {}
        profile = normalize_profile_name(self._name)
        if manager.profile == profile:
            section = manager.get_section("web")
            return section.model_dump(mode="python", exclude_unset=False)
        preview = manager.preview_profile(profile)
        source = preview.get("web", {})
        return dict(source) if isinstance(source, dict) else {}

    def _value(self, key: str) -> Any:
        if key in self._pending:
            return self._pending[key]
        return self._web_section().get(key)


# TODO(FXC-3827): Delete LegacyEnvironment after deprecating `tidy3d.config.Env`.
class LegacyEnvironment:
    """Legacy Env wrapper that maps to profiles."""

    def __init__(self, manager: ConfigManager):
        self._previous_env_vars: dict[str, Optional[str]] = {}
        self.env_map: dict[str, LegacyEnvironmentConfig] = {}
        self._current: Optional[LegacyEnvironmentConfig] = None
        self._manager: Optional[ConfigManager] = None
        self._applied_profile: Optional[str] = None
        self.reset_manager(manager)

    def reset_manager(self, manager: ConfigManager) -> None:
        self._manager = manager
        self.env_map = {}
        for name in BUILTIN_PROFILES:
            key = normalize_profile_name(name)
            self.env_map[key] = LegacyEnvironmentConfig(manager, key, environment=self)
        self._applied_profile = None
        self._current = None
        self._sync_to_manager(apply_env=True)

    @property
    def current(self) -> LegacyEnvironmentConfig:
        self._sync_to_manager()
        assert self._current is not None
        return self._current

    def set_current(self, env_config: LegacyEnvironmentConfig) -> None:
        _warn_env_deprecated()
        key = normalize_profile_name(env_config.name)
        stored = self._get_config(key)
        stored.copy_state_from(env_config)
        if self._manager and self._manager.profile != key:
            self._manager.switch_profile(key)
        self._sync_to_manager(apply_env=True)

    def enable_caching(self, enable_caching: Optional[bool] = True) -> None:
        config = self.current
        config.enable_caching = enable_caching
        self._sync_to_manager()

    def set_ssl_version(self, ssl_version: Optional[str]) -> None:
        config = self.current
        config.ssl_version = ssl_version
        self._sync_to_manager()

    def __getattr__(self, name: str) -> LegacyEnvironmentConfig:
        return self._get_config(name)

    def _get_config(self, name: str) -> LegacyEnvironmentConfig:
        key = normalize_profile_name(name)
        config = self.env_map.get(key)
        if config is None:
            config = LegacyEnvironmentConfig(self._manager, key, environment=self)
            self.env_map[key] = config
        else:
            manager = self._manager
            if manager is not None:
                config.reset_manager(manager)
            config._environment = self
        return config

    def _sync_to_manager(self, *, apply_env: bool = False) -> None:
        if self._manager is None:
            return
        active = normalize_profile_name(self._manager.profile)
        config = self._get_config(active)
        config.apply_pending_overrides()
        self._current = config
        if apply_env or self._applied_profile != active:
            self._apply_env_vars(config)
            self._applied_profile = active

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
    """Load legacy flat configuration file (pre-migration format).

    This function now supports both the original flat config format and
    Nexus custom deployment settings introduced in later versions.

    Legacy key mappings:
    - apikey -> web.apikey
    - web_api_endpoint -> web.api_endpoint
    - website_endpoint -> web.website_endpoint
    - s3_region -> web.s3_region
    - s3_endpoint -> web.env_vars.AWS_ENDPOINT_URL_S3
    - ssl_verify -> web.ssl_verify
    - enable_caching -> web.enable_caching
    """

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

    # Migrate API key (original functionality)
    apikey = parsed.get("apikey")
    if apikey is not None:
        legacy_data.setdefault("web", {})["apikey"] = apikey

    # Migrate Nexus API endpoint
    web_api = parsed.get("web_api_endpoint")
    if web_api is not None:
        legacy_data.setdefault("web", {})["api_endpoint"] = web_api

    # Migrate Nexus website endpoint
    website = parsed.get("website_endpoint")
    if website is not None:
        legacy_data.setdefault("web", {})["website_endpoint"] = website

    # Migrate S3 region
    s3_region = parsed.get("s3_region")
    if s3_region is not None:
        legacy_data.setdefault("web", {})["s3_region"] = s3_region

    # Migrate SSL verification setting
    ssl_verify = parsed.get("ssl_verify")
    if ssl_verify is not None:
        legacy_data.setdefault("web", {})["ssl_verify"] = ssl_verify

    # Migrate caching setting
    enable_caching = parsed.get("enable_caching")
    if enable_caching is not None:
        legacy_data.setdefault("web", {})["enable_caching"] = enable_caching

    # Migrate S3 endpoint to env_vars
    s3_endpoint = parsed.get("s3_endpoint")
    if s3_endpoint is not None:
        env_vars = legacy_data.setdefault("web", {}).setdefault("env_vars", {})
        env_vars["AWS_ENDPOINT_URL_S3"] = s3_endpoint

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
