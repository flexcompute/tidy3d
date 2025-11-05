"""Filesystem helpers and persistence utilities for the configuration system."""

from __future__ import annotations

import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import toml
import tomlkit

from tidy3d.log import log

from .profiles import BUILTIN_PROFILES
from .serializer import build_document, collect_descriptions


class ConfigLoader:
    """Handle reading and writing configuration files."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or resolve_config_directory()
        self.config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._docs: dict[Path, tomlkit.TOMLDocument] = {}

    def load_base(self) -> dict[str, Any]:
        """Load base configuration from config.toml.

        If config.toml doesn't exist but the legacy flat config does,
        automatically migrate to the new format.
        """

        config_path = self.config_dir / "config.toml"
        data = self._read_toml(config_path)
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
                # Save in new format
                self.save_base(legacy)

                # Rename old config to preserve it
                backup_path = legacy_path.with_suffix(".migrated")
                legacy_path.rename(backup_path)

                log.info(
                    f"Migration complete. Configuration saved to '{config_path}'. "
                    f"Legacy config backed up as '{backup_path.name}'."
                )

                # Re-read the newly created config
                return self._read_toml(config_path)
            except Exception as exc:
                log.warning(
                    f"Failed to auto-migrate legacy configuration: {exc}. "
                    "Using legacy data without migration."
                )
                return legacy

        if legacy:
            return legacy
        return {}

    def load_user_profile(self, profile: str) -> dict[str, Any]:
        """Load user profile overrides (if any)."""

        if profile in ("default", "prod"):
            # default and prod share the same baseline; user overrides live in config.toml
            return {}

        profile_path = self.profile_path(profile)
        return self._read_toml(profile_path)

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

    def _read_toml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            self._docs.pop(path, None)
            return {}

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            log.warning(f"Failed to read configuration file '{path}': {exc}")
            self._docs.pop(path, None)
            return {}

        try:
            document = tomlkit.parse(text)
        except Exception as exc:
            log.warning(f"Failed to parse configuration file '{path}': {exc}")
            document = tomlkit.document()
        self._docs[path] = document

        try:
            return toml.loads(text)
        except Exception as exc:
            log.warning(f"Failed to decode configuration file '{path}': {exc}")
            return {}

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp_dir = path.parent

        cleaned = _clean_data(deepcopy(data))
        descriptions = collect_descriptions()

        base_document = self._docs.get(path)
        document = build_document(cleaned, base_document, descriptions)
        toml_text = tomlkit.dumps(document)

        with tempfile.NamedTemporaryFile(
            "w", dir=tmp_dir, delete=False, encoding="utf-8"
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(toml_text)
            handle.flush()
            os.fsync(handle.fileno())

        backup_path = path.with_suffix(path.suffix + ".bak")
        try:
            if path.exists():
                shutil.copy2(path, backup_path)
            tmp_path.replace(path)
            os.chmod(path, 0o600)
            if backup_path.exists():
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

        self._docs[path] = tomlkit.parse(toml_text)


def load_environment_overrides() -> dict[str, Any]:
    """Parse environment variables into a nested configuration dict."""

    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if key == "SIMCLOUD_APIKEY":
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


def legacy_config_directory() -> Path:
    """Return the legacy configuration directory (~/.tidy3d)."""

    return Path.home() / ".tidy3d"


def canonical_config_directory() -> Path:
    """Return the platform-dependent canonical configuration directory."""

    return _xdg_config_home() / "tidy3d"


def resolve_config_directory() -> Path:
    """Determine the directory used to store tidy3d configuration files."""

    base_override = os.getenv("TIDY3D_BASE_DIR")
    if base_override:
        base_path = Path(base_override).expanduser().resolve()
        path = base_path / "config"
        if _is_writable(path.parent):
            return path
        log.warning(
            "'TIDY3D_BASE_DIR' is not writable; using temporary configuration directory instead."
        )
        return _temporary_config_dir()

    canonical_dir = canonical_config_directory()
    if _is_writable(canonical_dir.parent):
        legacy_dir = legacy_config_directory()
        if legacy_dir.exists():
            log.warning(
                f"Using canonical configuration directory at '{canonical_dir}'. "
                "Found legacy directory at '~/.tidy3d', which will be ignored. "
                "Remove it manually or run 'tidy3d config migrate --delete-legacy' to clean up.",
                log_once=True,
            )
        return canonical_dir

    legacy_dir = legacy_config_directory()
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
        test_file = path / ".tidy3d_write_test"
        with open(test_file, "w", encoding="utf-8"):
            pass
        test_file.unlink()
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
