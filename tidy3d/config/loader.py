"""Filesystem helpers and persistence utilities for the configuration system."""

from __future__ import annotations

import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import toml
import tomlkit
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
    set_config_version,
    strip_config_version,
)
from .profiles import BUILTIN_PROFILES
from .registry import get_sections
from .schema_utils import TOP_LEVEL_METADATA_KEYS
from .serializer import build_document, collect_descriptions

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Optional

_BASE_ONLY_METADATA_KEYS = TOP_LEVEL_METADATA_KEYS
_OPTIONAL_CORE_SECTION_NAMES = {"web", "local_cache", "batch_data_cache"}


class ConfigLoader:
    """Handle reading and writing configuration files."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or resolve_config_directory()
        self.config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._docs: dict[Path, tomlkit.TOMLDocument] = {}
        self._pending_writes: dict[Path, tomlkit.TOMLDocument] = {}
        self._pending_legacy_moves: dict[Path, tuple[Path, Path]] = {}

    def load_base(
        self,
        *,
        commit_writes: bool = True,
        queue_migration_write: Optional[bool] = None,
        validation_profile: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load base configuration from config.toml.

        If config.toml doesn't exist but the legacy flat config does,
        automatically migrate to the new format.
        """

        config_path = self.config_dir / "config.toml"
        if queue_migration_write is None:
            queue_migration_write = commit_writes
        data = self._read_toml(
            config_path,
            queue_migration_write=queue_migration_write,
            validation_profile=validation_profile,
        )
        if commit_writes:
            self.commit_pending_writes()
        if data:
            return data

        # Check for legacy flat config
        from .legacy import load_legacy_flat_config

        legacy_path = self.config_dir / "config"
        legacy = load_legacy_flat_config(self.config_dir)

        # Auto-migrate if legacy config exists
        if legacy and legacy_path.exists():
            log.info(
                f"Detected legacy configuration at '{legacy_path}'. "
                "Automatically migrating to new format..."
            )

            try:
                migrated = self._migrate_legacy_payload(legacy)
            except Exception as exc:
                self._warn_legacy_auto_migration_failed(exc)
                return legacy

            queued_legacy_write = False
            if queue_migration_write:
                queued_legacy_write = self._queue_legacy_migration_write(
                    config_path=config_path,
                    legacy_path=legacy_path,
                    migrated=migrated,
                    validation_profile=validation_profile,
                )

            if commit_writes and queued_legacy_write:
                self.commit_pending_writes()
                if not legacy_path.exists():
                    backup_path = legacy_path.with_suffix(".migrated")
                    log.info(
                        f"Migration complete. Configuration saved to '{config_path}'. "
                        f"Legacy config backed up as '{backup_path.name}'."
                    )

            return migrated

        if legacy:
            try:
                return self._migrate_legacy_payload(legacy)
            except Exception as exc:
                self._warn_legacy_auto_migration_failed(exc)
                return legacy
        return {}

    def _warn_legacy_auto_migration_failed(self, exc: Exception) -> None:
        """Log a consistent warning when legacy payload migration fails."""

        log.warning(
            f"Failed to auto-migrate legacy configuration: {exc}. "
            "Using legacy data without migration."
        )

    def load_user_profile(
        self,
        profile: str,
        *,
        commit_writes: bool = True,
        queue_migration_write: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Load user profile overrides (if any)."""

        if profile in ("default", "prod"):
            # default and prod share the same baseline; user overrides live in config.toml
            return {}

        profile_path = self.profile_path(profile)
        if queue_migration_write is None:
            queue_migration_write = commit_writes
        data = self._read_toml(profile_path, queue_migration_write=queue_migration_write)
        try:
            self._validate_base_only_metadata(path=profile_path, data=data)
        except Exception:
            self._pending_writes.pop(profile_path, None)
            raise
        if commit_writes:
            self.commit_pending_writes()
        return data

    def get_builtin_profile(self, profile: str) -> dict[str, Any]:
        """Return builtin profile data if available."""

        return BUILTIN_PROFILES.get(profile, {})

    def save_base(self, data: dict[str, Any]) -> None:
        """Persist base configuration."""

        config_path = self.config_dir / "config.toml"
        self._atomic_write(config_path, data)

    def save_profile(self, profile: str, data: dict[str, Any]) -> None:
        """Persist profile overrides (remove file if empty)."""

        profile_path = self.profile_path(profile)
        if not data:
            if profile_path.exists():
                profile_path.unlink()
            self._docs.pop(profile_path, None)
            self._pending_writes.pop(profile_path, None)
            return
        profile_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._atomic_write(profile_path, data)

    def profile_path(self, profile: str) -> Path:
        """Return on-disk path for a profile."""

        return self.config_dir / "profiles" / f"{profile}.toml"

    def get_default_profile(self) -> Optional[str]:
        """Read the default_profile from config.toml.

        Returns
        -------
        Optional[str]
            The default profile name if set, None otherwise.
        """

        config_path = self.config_dir / "config.toml"
        if not config_path.exists():
            return None

        try:
            text = config_path.read_text(encoding="utf-8")
            data = toml.loads(text)
            return data.get("default_profile")
        except Exception as exc:
            log.warning(f"Failed to read default_profile from '{config_path}': {exc}")
        return None

    def set_default_profile(self, profile: Optional[str]) -> None:
        """Set the default_profile in config.toml.

        Parameters
        ----------
        profile : Optional[str]
            The profile name to set as default, or None to remove the setting.
        """

        config_path = self.config_dir / "config.toml"
        data = self._read_toml(config_path)

        if profile is None:
            # Remove default_profile if it exists
            if "default_profile" in data:
                del data["default_profile"]
        else:
            # Set default_profile as a top-level key
            data["default_profile"] = profile

        self._atomic_write(config_path, data)

    def commit_pending_writes(self) -> None:
        """Write back migrated configuration files after validation."""

        if not self._pending_writes:
            return
        if not auto_migrate_enabled():
            self._pending_writes.clear()
            self._pending_legacy_moves.clear()
            return

        for path, document in list(self._pending_writes.items()):
            legacy_move = self._pending_legacy_moves.get(path)
            try:
                self._atomic_write_document(path, document, keep_backup=True)
                if legacy_move is not None:
                    legacy_path, backup_path = legacy_move
                    if legacy_path.exists():
                        legacy_path.rename(backup_path)
            except Exception as exc:
                log.warning(f"Failed to write migrated configuration file '{path}': {exc}")
            finally:
                self._clear_pending_path(path)

    def write_document(self, path: Path, document: tomlkit.TOMLDocument) -> None:
        """Write a fully rendered TOML document to disk."""

        self._atomic_write_document(path, document, keep_backup=True)

    def write_documents_transactional(
        self, documents: list[tuple[Path, tomlkit.TOMLDocument]]
    ) -> None:
        """Write a batch of upgraded documents with rollback on failure."""

        if not documents:
            return

        originals: dict[Path, str | None] = {}
        original_backups: dict[Path, str | None] = {}
        for path, _ in documents:
            try:
                originals[path] = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                originals[path] = None
            except Exception as exc:
                raise ValueError(f"Failed to read '{path}' before upgrade write: {exc}") from exc
            backup_path = path.with_suffix(path.suffix + ".bak")
            try:
                original_backups[path] = backup_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                original_backups[path] = None
            except Exception as exc:
                raise ValueError(
                    f"Failed to read backup '{backup_path}' before upgrade write: {exc}"
                ) from exc

        written_paths: list[Path] = []
        try:
            for path, document in documents:
                self._atomic_write_document(path, document, keep_backup=True)
                written_paths.append(path)
        except Exception as exc:
            for path in reversed(written_paths):
                original = originals.get(path)
                original_backup = original_backups.get(path)
                backup_path = path.with_suffix(path.suffix + ".bak")
                try:
                    if original is None:
                        if path.exists():
                            path.unlink()
                        self._clear_cached_path(path)
                    else:
                        self._replace_text_atomic(path, original)
                        try:
                            self._docs[path] = tomlkit.parse(original)
                        except Exception:
                            self._docs.pop(path, None)
                        self._clear_pending_path(path)

                    if original_backup is None:
                        if backup_path.exists():
                            backup_path.unlink()
                    else:
                        self._replace_text_atomic(backup_path, original_backup)
                except Exception as rollback_exc:
                    log.error(f"Failed to rollback configuration file '{path}': {rollback_exc}")
            raise RuntimeError(f"Failed to apply configuration upgrade atomically: {exc}") from exc

    def preview_schema_upgrade(self, path: Path) -> dict[str, Any]:
        """Return schema upgrade preview information for a config file."""

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise ValueError(f"Failed to read '{path}': {exc}") from exc

        data, document = self._parse_toml_text(path, text)
        version = get_config_version(document or data)
        if version > CURRENT_CONFIG_VERSION:
            return {
                "path": path,
                "version": version,
                "forward": True,
                "changed": False,
                "before": text,
                "after": text,
                "document": None,
            }

        if version < CURRENT_CONFIG_VERSION:
            apply_migrations(document, version, CURRENT_CONFIG_VERSION)
            set_config_version(document, CURRENT_CONFIG_VERSION)
            after = tomlkit.dumps(document)
        else:
            after = text

        try:
            data = toml.loads(after)
        except Exception as exc:
            raise ValueError(f"Failed to decode migrated '{path}': {exc}") from exc

        cleaned = strip_config_version(data)
        self._validate_base_only_metadata(path=path, data=cleaned)
        self._validate_data_for_path(path, cleaned)
        return {
            "path": path,
            "version": version,
            "forward": False,
            "changed": after != text,
            "before": text,
            "after": after,
            "document": document if after != text else None,
        }

    def _read_toml(
        self,
        path: Path,
        *,
        queue_migration_write: bool = True,
        validation_profile: Optional[str] = None,
        raise_on_parse_error: bool = False,
    ) -> dict[str, Any]:
        if not path.exists():
            self._clear_cached_path(path)
            return {}

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            log.warning(f"Failed to read configuration file '{path}': {exc}")
            self._clear_cached_path(path)
            return {}

        try:
            data, document = self._parse_toml_text(path, text)
        except ValueError as exc:
            self._clear_cached_path(path)
            if raise_on_parse_error:
                raise
            log.warning(str(exc))
            return {}

        self._docs[path] = document
        return self._apply_schema_migrations(
            path,
            data,
            document,
            queue_migration_write=queue_migration_write,
            validation_profile=validation_profile,
        )

    def _parse_toml_text(
        self, path: Path, text: str
    ) -> tuple[dict[str, Any], tomlkit.TOMLDocument]:
        try:
            document = tomlkit.parse(text)
        except Exception as exc:
            raise ValueError(f"Failed to parse configuration file '{path}': {exc}") from exc

        try:
            data = toml.loads(text)
        except Exception as exc:
            raise ValueError(f"Failed to decode configuration file '{path}': {exc}") from exc

        return data, document

    def _clear_cached_path(self, path: Path) -> None:
        self._docs.pop(path, None)
        self._clear_pending_path(path)

    def _clear_pending_path(self, path: Path) -> None:
        self._pending_writes.pop(path, None)
        self._pending_legacy_moves.pop(path, None)

    def _queue_legacy_migration_write(
        self,
        *,
        config_path: Path,
        legacy_path: Path,
        migrated: dict[str, Any],
        validation_profile: Optional[str] = None,
    ) -> bool:
        if not self._should_queue_migration_write(
            config_path, migrated, validation_profile=validation_profile
        ):
            self._clear_pending_path(config_path)
            return False
        document = self._build_document(config_path, migrated)
        self._docs[config_path] = document
        self._pending_writes[config_path] = document
        self._pending_legacy_moves[config_path] = (
            legacy_path,
            legacy_path.with_suffix(".migrated"),
        )
        return True

    def _is_profile_path(self, path: Path) -> bool:
        profiles_dir = (self.config_dir / "profiles").resolve()
        try:
            path.resolve().relative_to(profiles_dir)
        except ValueError:
            return False
        return True

    def _validate_base_only_metadata(self, path: Path, data: dict[str, Any]) -> None:
        if not self._is_profile_path(path):
            return
        invalid_keys = sorted(key for key in _BASE_ONLY_METADATA_KEYS if key in data)
        if not invalid_keys:
            return
        rendered = ", ".join(f"'{key}'" for key in invalid_keys)
        raise ValueError(
            f"Configuration key(s) {rendered} are only allowed in 'config.toml', not '{path.name}'."
        )

    def _load_profile_for_validation(self, profile: str) -> dict[str, Any]:
        """Load a profile payload without triggering migration write-back."""

        if profile in ("default", "prod"):
            return {}
        profile_path = self.profile_path(profile)
        data = self._read_toml(profile_path, queue_migration_write=False)
        self._validate_base_only_metadata(path=profile_path, data=data)
        return data

    def _base_validation_profile(
        self, data: dict[str, Any], validation_profile: Optional[str]
    ) -> str:
        if validation_profile:
            candidate = validation_profile.strip()
            if candidate:
                return candidate
        default_profile = data.get("default_profile")
        if isinstance(default_profile, str):
            candidate = default_profile.strip()
            if candidate:
                return candidate
        return "default"

    def _validation_tree_for_path(
        self, path: Path, data: dict[str, Any], validation_profile: Optional[str]
    ) -> dict[str, Any]:
        """Build the runtime-equivalent validation tree for a config file payload."""

        if self._is_profile_path(path):
            profile_name = path.stem
            base_data = self.load_base(commit_writes=False, queue_migration_write=False)
            builtin_data = self.get_builtin_profile(profile_name)
            return deep_merge(builtin_data, base_data, data)

        profile_name = self._base_validation_profile(data, validation_profile)
        builtin_data = self.get_builtin_profile(profile_name)
        if profile_name in ("default", "prod"):
            return deep_merge(builtin_data, data)

        profile_data = self._load_profile_for_validation(profile_name)
        return deep_merge(builtin_data, data, profile_data)

    def _validate_data_for_path(
        self, path: Path, data: dict[str, Any], *, validation_profile: Optional[str] = None
    ) -> None:
        validation_tree = self._validation_tree_for_path(path, data, validation_profile)
        build_validated_models(validation_tree, error_context="validate", log_errors=False)

    def _migrate_legacy_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        if not data:
            return {}
        document = tomlkit.parse(toml.dumps(data))
        apply_migrations(document, 0, CURRENT_CONFIG_VERSION)
        set_config_version(document, CURRENT_CONFIG_VERSION)
        migrated = toml.loads(tomlkit.dumps(document))
        return strip_config_version(migrated)

    def _apply_schema_migrations(
        self,
        path: Path,
        data: dict[str, Any],
        document: tomlkit.TOMLDocument,
        *,
        queue_migration_write: bool = True,
        validation_profile: Optional[str] = None,
    ) -> dict[str, Any]:
        version = get_config_version(document or data)

        if version > CURRENT_CONFIG_VERSION:
            self._clear_pending_path(path)
            mode = forward_compat_mode()
            if mode == "strict":
                raise ValueError(
                    f"Configuration file '{path}' targets config_version {version}, "
                    f"which is newer than supported version {CURRENT_CONFIG_VERSION}."
                )
            log.warning(
                f"Configuration file '{path}' targets config_version {version}, "
                f"which is newer than supported version {CURRENT_CONFIG_VERSION}. "
                "Falling back to best-effort parsing; unknown keys may be ignored."
            )
            filtered = best_effort_filter(data)
            return strip_config_version(filtered)

        if version < CURRENT_CONFIG_VERSION:
            # Keep an untouched document snapshot so loader caches stay consistent
            # if an in-place migration mutates and then fails.
            original_document = tomlkit.parse(tomlkit.dumps(document))
            try:
                apply_migrations(document, version, CURRENT_CONFIG_VERSION)
                set_config_version(document, CURRENT_CONFIG_VERSION)
                migrated = toml.loads(tomlkit.dumps(document))
            except Exception as exc:
                message = (
                    f"Automatic configuration migration failed for '{path}' "
                    f"(from config_version {version} to {CURRENT_CONFIG_VERSION}): {exc}. "
                    "Retry manually with 'tidy3d config upgrade' after fixing the issue."
                )
                log.error(message)
                self._docs[path] = original_document
                self._clear_pending_path(path)
                raise ValueError(message) from exc
            self._docs[path] = document
            migrated = strip_config_version(migrated)
            if queue_migration_write:
                if self._should_queue_migration_write(
                    path, migrated, validation_profile=validation_profile
                ):
                    self._pending_writes[path] = document
                    self._pending_legacy_moves.pop(path, None)
                else:
                    self._clear_pending_path(path)
            return migrated

        self._clear_pending_path(path)
        return strip_config_version(data)

    def _should_queue_migration_write(
        self, path: Path, data: dict[str, Any], *, validation_profile: Optional[str] = None
    ) -> bool:
        try:
            self._validate_data_for_path(path, data, validation_profile=validation_profile)
        except Exception as exc:
            log.warning(
                f"Skipping auto-migration write-back for '{path}' because migrated payload "
                f"does not validate without overrides: {exc}"
            )
            return False
        return True

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        document = self._build_document(path, data)
        toml_text = tomlkit.dumps(document)

        self._write_text_atomic(path, toml_text)
        self._docs[path] = tomlkit.parse(toml_text)
        self._clear_pending_path(path)

    def _build_document(self, path: Path, data: dict[str, Any]) -> tomlkit.TOMLDocument:
        with_version = inject_config_version(data, CURRENT_CONFIG_VERSION)
        cleaned = _clean_data(deepcopy(with_version))
        descriptions = collect_descriptions()
        base_document = self._docs.get(path)
        return build_document(cleaned, base_document, descriptions)

    def _atomic_write_document(
        self, path: Path, document: tomlkit.TOMLDocument, *, keep_backup: bool = False
    ) -> None:
        toml_text = tomlkit.dumps(document)
        self._write_text_atomic(path, toml_text, keep_backup=keep_backup)
        self._docs[path] = tomlkit.parse(toml_text)
        self._clear_pending_path(path)

    def _write_text_atomic(self, path: Path, toml_text: str, *, keep_backup: bool = False) -> None:
        tmp_path = self._write_temp_file(path, toml_text)

        backup_path = path.with_suffix(path.suffix + ".bak")
        try:
            if path.exists():
                shutil.copy2(path, backup_path)
            tmp_path.replace(path)
            os.chmod(path, 0o600)
            if backup_path.exists() and not keep_backup:
                backup_path.unlink()
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if backup_path.exists():
                try:
                    backup_path.replace(path)
                except Exception:
                    log.warning("Failed to restore configuration backup")
            raise

    def _replace_text_atomic(self, path: Path, toml_text: str) -> None:
        """Atomically replace a file without creating or mutating backup files."""

        tmp_path = self._write_temp_file(path, toml_text)
        try:
            tmp_path.replace(path)
            os.chmod(path, 0o600)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _write_temp_file(self, path: Path, toml_text: str) -> Path:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp_dir = path.parent
        with tempfile.NamedTemporaryFile(
            "w", dir=tmp_dir, delete=False, encoding="utf-8"
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(toml_text)
            handle.flush()
            os.fsync(handle.fileno())
        return tmp_path


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
            segments = ("web",) + segments[1:]
        if segments[0] not in known_roots:
            continue
        _assign_path(overrides, segments, value)
    return overrides


def deep_merge(*sources: dict[str, Any]) -> dict[str, Any]:
    """Deep merge multiple dictionaries into a new dict."""

    result: dict[str, Any] = {}
    for source in sources:
        _merge_into(result, source)
    return result


def _merge_into(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            node = target.setdefault(key, {})
            if isinstance(node, dict):
                _merge_into(node, value)
            else:
                target[key] = deepcopy(value)
        else:
            target[key] = value


def deep_diff(base: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Return keys from target that differ from base."""

    diff: dict[str, Any] = {}
    keys = set(base.keys()) | set(target.keys())
    for key in keys:
        base_value = base.get(key)
        target_value = target.get(key)
        if isinstance(base_value, dict) and isinstance(target_value, dict):
            nested = deep_diff(base_value, target_value)
            if nested:
                diff[key] = nested
        elif target_value != base_value:
            if isinstance(target_value, dict):
                diff[key] = deepcopy(target_value)
            else:
                diff[key] = target_value
    return diff


def _assign_path(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = target
    for segment in path[:-1]:
        node = node.setdefault(segment, {})
    node[path[-1]] = value


def _clean_data(data: Any) -> Any:
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            cleaned_value = _clean_data(value)
            if cleaned_value is None:
                continue
            cleaned[key] = cleaned_value
        return cleaned
    if isinstance(data, list):
        cleaned_list = [_clean_data(item) for item in data]
        return [item for item in cleaned_list if item is not None]
    if data is None:
        return None
    return data


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
            prefix = ("plugins", plugin_name)
            yield SectionPayload(name, schema, payload, prefix, plugin_name)
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
        # propagate the first error; others already logged
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
            "Found legacy directory at '~/.tidy3d', which will be ignored. "
            "Remove it manually or run 'tidy3d config migrate --delete-legacy' to clean up.",
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

    if legacy_dir.exists():
        log.warning(
            "Configuration found in legacy location '~/.tidy3d'. Consider running 'tidy3d config migrate'.",
            log_once=True,
        )
        return legacy_dir

    log.warning(f"Unable to write to '{canonical_dir}'; falling back to temporary directory.")
    return _temporary_config_dir()


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


def migrate_legacy_config(*, overwrite: bool = False, remove_legacy: bool = False) -> Path:
    """Copy configuration files from the legacy ``~/.tidy3d`` directory to the canonical location.

    Parameters
    ----------
    overwrite : bool
        If ``True``, existing files in the canonical directory will be replaced.
    remove_legacy : bool
        If ``True``, the legacy directory is removed after a successful migration.

    Returns
    -------
    Path
        The path of the canonical configuration directory.

    Raises
    ------
    FileNotFoundError
        If the legacy directory does not exist.
    FileExistsError
        If the destination already exists and ``overwrite`` is ``False``.
    RuntimeError
        If the legacy and canonical directories resolve to the same location.
    """

    legacy_dir = legacy_config_directory()
    if not legacy_dir.exists():
        raise FileNotFoundError("Legacy configuration directory '~/.tidy3d' was not found.")

    canonical_dir = canonical_config_directory()
    if canonical_dir.resolve() == legacy_dir.resolve():
        raise RuntimeError(
            "Legacy and canonical configuration directories are the same path; nothing to migrate."
        )

    if canonical_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Destination '{canonical_dir}' already exists. Pass overwrite=True to replace existing files."
        )

    canonical_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(legacy_dir, canonical_dir, dirs_exist_ok=overwrite)

    from .legacy import finalize_legacy_migration  # local import to avoid circular dependency

    finalize_legacy_migration(canonical_dir)

    if remove_legacy:
        shutil.rmtree(legacy_dir)

    return canonical_dir
