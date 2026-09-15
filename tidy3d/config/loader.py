"""Tidy3D config loader wiring around the shared flexcompute-core mechanics."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import tomlkit
from flexcompute.core.config import ConfigApp, ConfigRegistry
from flexcompute.core.config.loader import ConfigLoader as CoreConfigLoader
from flexcompute.core.config.loader import deep_diff as _deep_diff
from flexcompute.core.config.loader import deep_merge as _deep_merge
from pydantic import BaseModel

from tidy3d.log import log

from .deprecations import check_deprecations
from .migrations import (
    CURRENT_CONFIG_VERSION,
    apply_migrations,
    auto_migrate_enabled,
    best_effort_filter,
    forward_compat_mode,
    get_config_version,
    inject_config_version,
    register_migration,
    set_config_version,
    strip_config_version,
)
from .profiles import BUILTIN_PROFILES
from .registry import get_handlers, get_sections, register_handler, register_section
from .schema_utils import TOP_LEVEL_METADATA_KEYS

if TYPE_CHECKING:
    from collections.abc import Iterable

_OPTIONAL_CORE_SECTION_NAMES = {"web", "local_cache", "batch_data_cache"}
deep_diff = _deep_diff
deep_merge = _deep_merge
SectionDecorator = Callable[[type[BaseModel]], type[BaseModel]]
HandlerDecorator = Callable[[Callable[[BaseModel], None]], Callable[[BaseModel], None]]
MigrationDecorator = Callable[
    [Callable[[tomlkit.TOMLDocument], None]], Callable[[tomlkit.TOMLDocument], None]
]


class Tidy3DRegistryAdapter(ConfigRegistry):
    """Expose Tidy3D's global registries through flexcompute-core's loader API."""

    def __init__(self) -> None:
        super().__init__(current_version=CURRENT_CONFIG_VERSION)

    @property
    def sections(self) -> dict[str, type[BaseModel]]:
        return get_sections()

    @property
    def handlers(self) -> dict[str, Any]:
        return get_handlers()

    def section(self, name: str) -> SectionDecorator:
        return register_section(name)

    def handler(self, name: str) -> HandlerDecorator:
        return register_handler(name)

    def migration(self, version: int) -> MigrationDecorator:
        return register_migration(version)

    def apply_migrations(
        self, document: tomlkit.TOMLDocument, from_version: int, to_version: int | None = None
    ) -> None:
        target = CURRENT_CONFIG_VERSION if to_version is None else to_version
        apply_migrations(document, from_version, target)


def tidy3d_config_app(config_dir: Path | None, *, resolve_default: bool = True) -> ConfigApp:
    """Return the product identity used by the Core config mechanics."""

    resolved_config_dir = config_dir
    if resolved_config_dir is None and resolve_default:
        resolved_config_dir = resolve_config_directory()

    return ConfigApp(
        app_id="tidy3d",
        env_prefix="TIDY3D",
        config_dir_name="tidy3d",
        built_in_profiles=BUILTIN_PROFILES,
        config_dir=resolved_config_dir,
        version_profile_files=True,
        tolerate_load_errors=True,
        forward_compat_default="best-effort",
    )


class ConfigLoader(CoreConfigLoader):
    """Handle Tidy3D config IO while delegating generic mechanics to flexcompute-core."""

    def __init__(self, config_dir: Path | None = None, *, app: ConfigApp | None = None) -> None:
        if app is None:
            loader_path = None if config_dir is None else Path(config_dir)
            app = tidy3d_config_app(loader_path)
        super().__init__(
            app=app,
            registry=Tidy3DRegistryAdapter(),
            env=dict(os.environ),
        )

    def _before_load_base(self, path: Path) -> None:
        _warn_legacy_flat_config_ignored(config_dir=self.config_dir, config_path=path)

    def _profile_uses_base_config_only(self, profile: str) -> bool:
        return profile in ("default", "prod")

    def _current_config_version(self) -> int:
        return CURRENT_CONFIG_VERSION

    def _get_config_version(self, source: Any) -> int:
        return get_config_version(source)

    def _set_config_version(self, document: tomlkit.TOMLDocument, version: int) -> None:
        set_config_version(document, version)

    def _strip_config_version(self, data: dict[str, Any]) -> dict[str, Any]:
        return strip_config_version(data)

    def _inject_config_version(self, data: dict[str, Any], version: int) -> dict[str, Any]:
        return inject_config_version(data, version)

    def _apply_migrations(
        self, document: tomlkit.TOMLDocument, from_version: int, to_version: int
    ) -> None:
        apply_migrations(document, from_version, to_version)

    def _auto_migrate_enabled(self) -> bool:
        return auto_migrate_enabled()

    def _forward_compat_mode(self) -> str:
        return forward_compat_mode()

    def _best_effort_filter(self, data: dict[str, Any]) -> dict[str, Any]:
        return best_effort_filter(data)

    def _validate_tree(self, tree: dict[str, Any], *, error_context: str, log_errors: bool) -> None:
        build_validated_models(tree, error_context=error_context, log_errors=log_errors)

    def _migration_failure_message(self, path: Path, version: int, exc: Exception) -> str:
        return (
            f"Automatic configuration migration failed for '{path}' "
            f"(from config_version {version} to {CURRENT_CONFIG_VERSION}): {exc}. "
            "Retry manually with 'tidy3d config upgrade' after fixing the issue."
        )

    def _wrap_migration_errors(self) -> bool:
        return True

    def _warn(self, message: str) -> None:
        log.warning(message)

    def _warn_once(self, message: str) -> None:
        log.warning(message, log_once=True)

    def _error(self, message: str) -> None:
        log.error(message)


def load_environment_overrides() -> dict[str, Any]:
    """Parse environment variables into a nested configuration dict."""

    known_roots = {name.split(".", 1)[0] for name in get_sections().keys()}
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if key == "SIMCLOUD_APIKEY":
            if "web" in known_roots:
                _assign_path(overrides, ("web", "apikey"), value)
            continue
        if not key.startswith("TIDY3D_"):
            continue
        rest = key[len("TIDY3D_") :]
        if "__" not in rest:
            continue
        segments = tuple(segment.lower() for segment in rest.split("__") if segment)
        if not segments:
            continue
        if segments[0] == "auth":
            segments = ("web", *segments[1:])
        if segments[0] not in known_roots:
            continue
        _assign_path(overrides, segments, value)
    return overrides


def _assign_path(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = target
    for segment in path[:-1]:
        node = node.setdefault(segment, {})
    node[path[-1]] = value


class SectionPayload(NamedTuple):
    name: str
    schema: type[BaseModel]
    payload: Any
    prefix: tuple[str, ...]
    plugin_name: str | None


class ValidatedModels(NamedTuple):
    sections: dict[str, BaseModel]
    plugins: dict[str, BaseModel]


def iter_section_payloads(
    data: dict[str, Any], *, coerce_non_dict: bool
) -> Iterable[SectionPayload]:
    """Iterate over configured section payloads with consistent plugin handling."""

    sections = get_sections()
    for name, schema in sections.items():
        if name == "plugins":
            continue
        if name.startswith("plugins."):
            plugin_name = name.split(".", 1)[1]
            plugins_data = data.get("plugins", {})
            if not isinstance(plugins_data, dict):
                plugins_data = {}
            payload = plugins_data.get(plugin_name, {})
            if not isinstance(payload, dict) and coerce_non_dict:
                payload = {}
            yield SectionPayload(name, schema, payload, ("plugins", plugin_name), plugin_name)
            continue

        payload = data.get(name, {})
        if not isinstance(payload, dict) and coerce_non_dict:
            payload = {}
        yield SectionPayload(name, schema, payload, (name,), None)


def build_validated_models(
    data: dict[str, Any], *, error_context: str, log_errors: bool = True
) -> ValidatedModels:
    """Validate payloads and build section/plugin models from a config tree."""

    new_sections: dict[str, BaseModel] = {}
    new_plugins: dict[str, BaseModel] = {}
    errors: list[Exception] = []
    top_level_sections = {name for name in get_sections() if "." not in name}

    for key, value in data.items():
        if key in TOP_LEVEL_METADATA_KEYS:
            continue
        if key not in top_level_sections:
            if key in _OPTIONAL_CORE_SECTION_NAMES:
                if log_errors:
                    log.warning(
                        f"Ignoring configuration section '{key}' because it is not available in this build."
                    )
                continue
            exc = ValueError(f"Unknown configuration section '{key}'.")
            if log_errors:
                log.error(f"Failed to {error_context} configuration for section '{key}': {exc}")
            errors.append(exc)
            continue
        if key == "plugins" and not isinstance(value, dict):
            exc = TypeError("Configuration section 'plugins' should be a table.")
            if log_errors:
                log.error(f"Failed to {error_context} configuration for section '{key}': {exc}")
            errors.append(exc)
            continue

    for item in iter_section_payloads(data, coerce_non_dict=False):
        try:
            if isinstance(item.payload, dict):
                check_deprecations(item.schema, item.payload, item.prefix)
            model = item.schema(**item.payload)
        except Exception as exc:
            if log_errors:
                if item.plugin_name is not None:
                    log.error(
                        f"Failed to {error_context} configuration for plugin '{item.plugin_name}': {exc}"
                    )
                else:
                    log.error(
                        f"Failed to {error_context} configuration for section '{item.name}': {exc}"
                    )
            errors.append(exc)
            continue
        if item.plugin_name is not None:
            new_plugins[item.plugin_name] = model
        else:
            new_sections[item.name] = model
    if errors:
        raise errors[0]
    return ValidatedModels(new_sections, new_plugins)


def legacy_config_directory() -> Path:
    """Return the legacy configuration directory (~/.tidy3d)."""

    return Path.home() / ".tidy3d"


def canonical_config_directory() -> Path:
    """Return the platform-dependent canonical configuration directory."""

    return _xdg_config_home() / "tidy3d"


def _warn_legacy_dir_ignored(*, canonical_dir: Path, legacy_dir: Path) -> None:
    if legacy_dir.exists():
        log.warning(
            f"Using canonical configuration directory at '{canonical_dir}'. "
            f"Found legacy directory at '{legacy_dir}', which will be ignored. "
            f"Tidy3D configuration now uses '{canonical_dir / 'config.toml'}'.",
            log_once=True,
        )


def _warn_legacy_flat_config_ignored(*, config_dir: Path, config_path: Path) -> None:
    legacy_path = config_dir / "config"
    if legacy_path.is_file():
        log.warning(
            f"Found legacy configuration file at '{legacy_path}', which is no longer loaded. "
            f"Tidy3D configuration now uses '{config_path}'.",
            log_once=True,
        )


def resolve_config_directory() -> Path:
    """Determine the directory used to store tidy3d configuration files."""

    base_override = os.getenv("TIDY3D_BASE_DIR")
    if base_override:
        base_path = Path(base_override).expanduser().resolve()
        path = base_path / "config"
        if path.is_dir():
            return path
        if _is_writable(path.parent):
            return path
        log.warning(
            "'TIDY3D_BASE_DIR' is not writable; using temporary configuration directory instead."
        )
        return _temporary_config_dir()

    canonical_dir = canonical_config_directory()
    legacy_dir = legacy_config_directory()
    if canonical_dir.is_dir():
        _warn_legacy_dir_ignored(canonical_dir=canonical_dir, legacy_dir=legacy_dir)
        return canonical_dir
    if _is_writable(canonical_dir.parent):
        _warn_legacy_dir_ignored(canonical_dir=canonical_dir, legacy_dir=legacy_dir)
        return canonical_dir

    fallback_dir = _temporary_config_dir()

    if legacy_dir.exists():
        log.warning(
            f"Configuration found in removed legacy location '{legacy_dir}', which will be "
            "ignored.",
            log_once=True,
        )

    log.warning(
        f"Unable to write to '{canonical_dir}'; falling back to temporary directory "
        f"'{fallback_dir}'."
    )
    return fallback_dir


def _xdg_config_home() -> Path:
    xdg_home = os.getenv("XDG_CONFIG_HOME")
    if xdg_home:
        return Path(xdg_home).expanduser()
    return Path.home() / ".config"


def _temporary_config_dir() -> Path:
    base = Path(tempfile.gettempdir()) / "tidy3d"
    base.mkdir(mode=0o700, exist_ok=True)
    return base / "config"


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        fd, test_path = tempfile.mkstemp(dir=path, prefix=".tidy3d_write_test_")
        os.close(fd)
        try:
            Path(test_path).unlink()
        except FileNotFoundError:
            pass
        return True
    except Exception:
        return False
