"""Schema versioning and migration helpers for tidy3d configuration files."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import tomlkit

from tidy3d.log import log

from .registry import get_sections
from .schema_utils import TOP_LEVEL_METADATA_KEYS, _resolve_model_type

if TYPE_CHECKING:
    from typing import Optional

    from pydantic import BaseModel

CONFIG_VERSION_KEY = "config_version"
CURRENT_CONFIG_VERSION = 2

AUTO_MIGRATE_ENV = "TIDY3D_CONFIG_AUTO_MIGRATE"
FORWARD_COMPAT_ENV = "TIDY3D_CONFIG_FORWARD_COMPAT"

FORWARD_COMPAT_STRICT = "strict"
FORWARD_COMPAT_BEST_EFFORT = "best-effort"

MigrationFunc = Callable[[tomlkit.TOMLDocument], None]
_MIGRATIONS: dict[int, list[MigrationFunc]] = {}
_MIGRATION_CHAIN_VALIDATED_UP_TO = 0


def register_migration(version: int) -> Callable[[MigrationFunc], MigrationFunc]:
    """Register a schema migration from ``version`` to ``version + 1``."""

    def decorator(func: MigrationFunc) -> MigrationFunc:
        _MIGRATIONS.setdefault(version, []).append(func)
        _invalidate_migration_chain()
        return func

    return decorator


def get_config_version(source: Any) -> int:
    """Return the config version stored in a dict or TOML document."""

    if isinstance(source, tomlkit.TOMLDocument):
        raw = source.get(CONFIG_VERSION_KEY)
    elif isinstance(source, dict):
        raw = source.get(CONFIG_VERSION_KEY)
    else:
        raw = None

    if raw is None:
        return 0
    if isinstance(raw, bool):
        return _warn_invalid_version(raw)
    if isinstance(raw, int):
        version = raw
    elif isinstance(raw, str):
        try:
            version = int(raw)
        except (TypeError, ValueError):
            return _warn_invalid_version(raw)
    else:
        return _warn_invalid_version(raw)
    if version < 0:
        return _warn_invalid_version(version)
    return version


def _warn_invalid_version(value: Any) -> int:
    log.warning(f"Invalid '{CONFIG_VERSION_KEY}' value {value!r}; falling back to version 0.")
    return 0


def set_config_version(document: tomlkit.TOMLDocument, version: int) -> None:
    """Set the config version in a TOML document."""

    document[CONFIG_VERSION_KEY] = int(version)


def strip_config_version(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` without the config version key."""

    if CONFIG_VERSION_KEY not in data:
        return data
    cleaned = dict(data)
    cleaned.pop(CONFIG_VERSION_KEY, None)
    return cleaned


def inject_config_version(data: dict[str, Any], version: int) -> dict[str, Any]:
    """Return a copy of ``data`` with the config version key set."""

    updated = dict(data)
    updated[CONFIG_VERSION_KEY] = int(version)
    return updated


def auto_migrate_enabled() -> bool:
    """Return True if automatic write-back is enabled."""

    raw = os.getenv(AUTO_MIGRATE_ENV)
    if raw is None:
        return True
    value = raw.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    log.warning(f"Unrecognized '{AUTO_MIGRATE_ENV}' value {raw!r}; defaulting to auto-migrate.")
    return True


def forward_compat_mode() -> str:
    """Return the forward-compat behavior for newer config versions."""

    raw = os.getenv(FORWARD_COMPAT_ENV)
    if not raw:
        return FORWARD_COMPAT_BEST_EFFORT
    value = raw.strip().lower()
    if value in {FORWARD_COMPAT_STRICT, FORWARD_COMPAT_BEST_EFFORT}:
        return value
    log.warning(f"Unrecognized '{FORWARD_COMPAT_ENV}' value {raw!r}; defaulting to best-effort.")
    return FORWARD_COMPAT_BEST_EFFORT


def apply_migrations(document: tomlkit.TOMLDocument, from_version: int, to_version: int) -> None:
    """Apply registered migrations to the document."""

    if from_version >= to_version:
        return
    if from_version < 0:
        from_version = 0
    _ensure_migration_chain(to_version)
    for version in range(from_version, to_version):
        for migrator in _MIGRATIONS.get(version, []):
            migrator(document)


def best_effort_filter(data: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys from a config payload using the registered schemas."""

    sections = get_sections()
    if not sections:
        return strip_config_version(data)

    filtered: dict[str, Any] = {}
    for key, value in data.items():
        if key == CONFIG_VERSION_KEY:
            continue
        if key in TOP_LEVEL_METADATA_KEYS:
            filtered[key] = value
            continue
        if key == "plugins":
            filtered_plugins = _filter_plugins(value, sections)
            if filtered_plugins is not None:
                filtered["plugins"] = filtered_plugins
            continue

        schema = sections.get(key)
        if schema is None:
            log.warning(
                f"Ignoring unknown configuration section '{key}' during best-effort parsing."
            )
            continue
        if isinstance(value, dict):
            filtered[key] = _filter_section_data(schema, value)
        else:
            log.warning(
                f"Configuration section '{key}' should be a table; "
                "ignoring non-table value during best-effort parsing."
            )
            filtered[key] = {}
    return filtered


def _filter_plugins(value: Any, sections: dict[str, type[BaseModel]]) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        log.warning(
            "Configuration section 'plugins' should be a table; "
            "ignoring non-table value during best-effort parsing."
        )
        return None

    filtered: dict[str, Any] = {}
    for plugin_name, plugin_data in value.items():
        schema = sections.get(f"plugins.{plugin_name}")
        if schema is None:
            log.warning(
                f"Ignoring unknown plugin configuration section '{plugin_name}' "
                "during best-effort parsing."
            )
            continue
        if isinstance(plugin_data, dict):
            filtered[plugin_name] = _filter_section_data(schema, plugin_data)
        else:
            log.warning(
                f"Configuration plugin section '{plugin_name}' should be a table; "
                "ignoring non-table value during best-effort parsing."
            )
            filtered[plugin_name] = {}
    return filtered


def _filter_section_data(schema: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for field_name, field in schema.model_fields.items():
        if field_name not in data:
            continue
        value = data[field_name]
        nested_model = _resolve_model_type(field.annotation)
        if nested_model is not None:
            if isinstance(value, dict):
                filtered[field_name] = _filter_section_data(nested_model, value)
                continue
            if isinstance(value, list):
                filtered[field_name] = [
                    _filter_section_data(nested_model, item) if isinstance(item, dict) else item
                    for item in value
                ]
                continue
        filtered[field_name] = value
    return filtered


def _ensure_migration_chain(target_version: int) -> None:
    global _MIGRATION_CHAIN_VALIDATED_UP_TO
    if target_version <= _MIGRATION_CHAIN_VALIDATED_UP_TO:
        return
    _validate_migration_chain(target_version)
    _MIGRATION_CHAIN_VALIDATED_UP_TO = target_version


def _invalidate_migration_chain() -> None:
    global _MIGRATION_CHAIN_VALIDATED_UP_TO
    _MIGRATION_CHAIN_VALIDATED_UP_TO = 0


def _validate_migration_chain(target_version: int) -> None:
    for version in range(target_version):
        if version not in _MIGRATIONS or not _MIGRATIONS[version]:
            raise RuntimeError(f"Missing config migration step for v{version} -> v{version + 1}.")


@register_migration(0)
def _migrate_v0_to_v1(document: tomlkit.TOMLDocument) -> None:
    """Initial schema migration (no-op)."""

    return None


@register_migration(1)
def _migrate_v1_to_v2(document: tomlkit.TOMLDocument) -> None:
    """Move pay_type from vgpu defaults into the new run section."""

    vgpu_table = document.get("vgpu")
    if not isinstance(vgpu_table, tomlkit.items.Table):
        return
    if "pay_type" not in vgpu_table:
        return

    run_table = document.get("run")
    if not isinstance(run_table, tomlkit.items.Table):
        run_table = tomlkit.table()
        document["run"] = run_table

    if "pay_type" not in run_table:
        run_table["pay_type"] = vgpu_table["pay_type"]
    del vgpu_table["pay_type"]


__all__ = [
    "AUTO_MIGRATE_ENV",
    "CONFIG_VERSION_KEY",
    "CURRENT_CONFIG_VERSION",
    "FORWARD_COMPAT_BEST_EFFORT",
    "FORWARD_COMPAT_ENV",
    "FORWARD_COMPAT_STRICT",
    "apply_migrations",
    "auto_migrate_enabled",
    "best_effort_filter",
    "forward_compat_mode",
    "get_config_version",
    "inject_config_version",
    "register_migration",
    "set_config_version",
    "strip_config_version",
]
