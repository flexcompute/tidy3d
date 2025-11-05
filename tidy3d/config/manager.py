"""Central configuration manager implementation."""

from __future__ import annotations

import os
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text
from rich.tree import Tree

from tidy3d.log import log

from .loader import ConfigLoader, deep_diff, deep_merge, load_environment_overrides
from .profiles import BUILTIN_PROFILES
from .registry import attach_manager, get_handlers, get_sections


def normalize_profile_name(name: str) -> str:
    """Return a canonical profile name for builtin profiles."""

    normalized = name.strip()
    lowered = normalized.lower()
    if lowered in BUILTIN_PROFILES:
        return lowered
    return normalized


class SectionAccessor:
    """Attribute proxy that routes assignments back through the manager."""

    def __init__(self, manager: ConfigManager, path: str):
        self._manager = manager
        self._path = path

    def __getattr__(self, name: str) -> Any:
        model = self._manager._get_model(self._path)
        if model is None:
            raise AttributeError(f"Section '{self._path}' is not available")
        return getattr(model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._manager.update_section(self._path, **{name: value})

    def __repr__(self) -> str:
        model = self._manager._get_model(self._path)
        return f"SectionAccessor({self._path}={model!r})"

    def __rich__(self) -> Panel:
        model = self._manager._get_model(self._path)
        if model is None:
            return Panel(Text(f"Section '{self._path}' is unavailable", style="red"))
        data = _prepare_for_display(model.model_dump(exclude_unset=False))
        return _build_section_panel(self._path, data)

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        model = self._manager._get_model(self._path)
        if model is None:
            return {}
        return model.model_dump(*args, **kwargs)

    def __str__(self) -> str:
        return self._manager.format_section(self._path)


class PluginsAccessor:
    """Provides access to registered plugin configurations."""

    def __init__(self, manager: ConfigManager):
        self._manager = manager

    def __getattr__(self, plugin: str) -> SectionAccessor:
        if plugin not in self._manager._plugin_models:
            raise AttributeError(f"Plugin '{plugin}' is not registered")
        return SectionAccessor(self._manager, f"plugins.{plugin}")

    def list(self) -> Iterable[str]:
        return sorted(self._manager._plugin_models.keys())


class ProfilesAccessor:
    """Read-only profile helper."""

    def __init__(self, manager: ConfigManager):
        self._manager = manager

    def list(self) -> dict[str, list[str]]:
        return self._manager.list_profiles()

    def __getattr__(self, profile: str) -> dict[str, Any]:
        return self._manager.preview_profile(profile)


class ConfigManager:
    """High-level orchestrator for tidy3d configuration."""

    def __init__(
        self,
        profile: Optional[str] = None,
        config_dir: Optional[os.PathLike[str]] = None,
    ):
        loader_path = None if config_dir is None else Path(config_dir)
        self._loader = ConfigLoader(loader_path)
        self._runtime_overrides: dict[str, dict[str, Any]] = defaultdict(dict)
        self._plugin_models: dict[str, BaseModel] = {}
        self._section_models: dict[str, BaseModel] = {}
        self._profile = self._resolve_initial_profile(profile)
        self._builtin_data: dict[str, Any] = {}
        self._base_data: dict[str, Any] = {}
        self._profile_data: dict[str, Any] = {}
        self._raw_tree: dict[str, Any] = {}
        self._effective_tree: dict[str, Any] = {}
        self._env_overrides: dict[str, Any] = load_environment_overrides()
        self._web_env_previous: dict[str, Optional[str]] = {}

        attach_manager(self)
        self._reload()

        # Notify users when using a non-default profile
        if self._profile != "default":
            log.info(f"Using configuration profile: '{self._profile}'", log_once=True)

        self._apply_handlers()

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def config_dir(self) -> Path:
        return self._loader.config_dir

    @property
    def plugins(self) -> PluginsAccessor:
        return PluginsAccessor(self)

    @property
    def profiles(self) -> ProfilesAccessor:
        return ProfilesAccessor(self)

    def update_section(self, name: str, **updates: Any) -> None:
        if not updates:
            return
        segments = name.split(".")
        overrides = self._runtime_overrides[self._profile]
        previous = deepcopy(overrides)
        node = overrides
        for segment in segments[:-1]:
            node = node.setdefault(segment, {})
        section_key = segments[-1]
        section_payload = node.setdefault(section_key, {})
        for key, value in updates.items():
            section_payload[key] = _serialize_value(value)
        try:
            self._reload()
        except Exception:
            self._runtime_overrides[self._profile] = previous
            raise
        self._apply_handlers(section=name)

    def switch_profile(self, profile: str) -> None:
        if not profile:
            raise ValueError("Profile name cannot be empty")
        normalized = normalize_profile_name(profile)
        if not normalized:
            raise ValueError("Profile name cannot be empty")
        self._profile = normalized
        self._reload()

        # Notify users when switching to a non-default profile
        if self._profile != "default":
            log.info(f"Switched to configuration profile: '{self._profile}'")

        self._apply_handlers()

    def set_default_profile(self, profile: Optional[str]) -> None:
        """Set the default profile to be used on startup.

        Parameters
        ----------
        profile : Optional[str]
            The profile name to use as default, or None to clear the default.
            When set, this profile will be automatically loaded unless overridden
            by environment variables (TIDY3D_CONFIG_PROFILE, TIDY3D_PROFILE, or TIDY3D_ENV).

        Notes
        -----
        This setting is persisted to config.toml and survives across sessions.
        Environment variables always take precedence over the default profile.
        """

        if profile is not None:
            normalized = normalize_profile_name(profile)
            if not normalized:
                raise ValueError("Profile name cannot be empty")
            self._loader.set_default_profile(normalized)
        else:
            self._loader.set_default_profile(None)

    def get_default_profile(self) -> Optional[str]:
        """Get the currently configured default profile.

        Returns
        -------
        Optional[str]
            The default profile name if set, None otherwise.
        """

        return self._loader.get_default_profile()

    def save(self, include_defaults: bool = False) -> None:
        if self._profile == "default":
            # For base config: only save fields marked with persist=True
            base_without_env = self._filter_persisted(self._compose_without_env())
            if include_defaults:
                defaults = self._filter_persisted(self._default_tree())
                base_without_env = deep_merge(defaults, base_without_env)
            self._loader.save_base(base_without_env)
        else:
            # For profile overrides: save any field that differs from baseline
            # (don't filter by persist flag - profiles should save all customizations)
            base_without_env = self._compose_without_env()
            baseline = deep_merge(self._builtin_data, self._base_data)
            diff = deep_diff(baseline, base_without_env)
            self._loader.save_profile(self._profile, diff)
        # refresh cached base/profile data after saving
        self._base_data = self._loader.load_base()
        self._profile_data = self._loader.load_user_profile(self._profile)
        self._reload()

    def reset_to_defaults(self, *, include_profiles: bool = True) -> None:
        """Reset configuration files to their default annotated state."""

        self._runtime_overrides = defaultdict(dict)
        defaults = self._filter_persisted(self._default_tree())
        self._loader.save_base(defaults)

        if include_profiles:
            profiles_dir = self._loader.profile_path("_dummy").parent
            if profiles_dir.exists():
                shutil.rmtree(profiles_dir)
            loader_docs = getattr(self._loader, "_docs", {})
            for path in list(loader_docs.keys()):
                try:
                    path.relative_to(profiles_dir)
                except ValueError:
                    continue
                loader_docs.pop(path, None)
            self._profile = "default"

        self._reload()
        self._apply_handlers()

    def apply_web_env(self, env_vars: Mapping[str, str]) -> None:
        """Apply environment variable overrides for the web configuration section."""

        self._restore_web_env()
        for key, value in env_vars.items():
            self._web_env_previous[key] = os.environ.get(key)
            os.environ[key] = value

    def _restore_web_env(self) -> None:
        """Restore previously overridden environment variables."""

        for key, previous in self._web_env_previous.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self._web_env_previous.clear()

    def list_profiles(self) -> dict[str, list[str]]:
        profiles_dir = self._loader.config_dir / "profiles"
        user_profiles = []
        if profiles_dir.exists():
            for path in profiles_dir.glob("*.toml"):
                user_profiles.append(path.stem)
        built_in = sorted(name for name in BUILTIN_PROFILES.keys())
        return {"built_in": built_in, "user": sorted(user_profiles)}

    def preview_profile(self, profile: str) -> dict[str, Any]:
        builtin = self._loader.get_builtin_profile(profile)
        base = self._loader.load_base()
        overrides = self._loader.load_user_profile(profile)
        view = deep_merge(builtin, base, overrides)
        return deepcopy(view)

    def get_section(self, name: str) -> BaseModel:
        model = self._get_model(name)
        if model is None:
            raise AttributeError(f"Section '{name}' is not available")
        return model

    def as_dict(self, include_env: bool = True) -> dict[str, Any]:
        """Return the current configuration tree, including defaults for all sections."""

        tree = self._compose_without_env()
        if include_env:
            tree = deep_merge(tree, self._env_overrides)
        return deep_merge(self._default_tree(), tree)

    def __rich__(self) -> Panel:
        """Return a rich renderable representation of the full configuration."""

        return _build_config_panel(
            title=f"Config (profile='{self._profile}')",
            data=_prepare_for_display(self.as_dict(include_env=True)),
        )

    def format(self, *, include_env: bool = True) -> str:
        """Return a human-friendly representation of the full configuration."""

        panel = _build_config_panel(
            title=f"Config (profile='{self._profile}')",
            data=_prepare_for_display(self.as_dict(include_env=include_env)),
        )
        return _render_panel(panel)

    def format_section(self, name: str) -> str:
        """Return a string representation for an individual section."""

        model = self._get_model(name)
        if model is None:
            raise AttributeError(f"Section '{name}' is not available")
        data = _prepare_for_display(model.model_dump(exclude_unset=False))
        panel = _build_section_panel(name, data)
        return _render_panel(panel)

    def on_section_registered(self, section: str) -> None:
        self._reload()
        self._apply_handlers(section=section)

    def on_handler_registered(self, section: str) -> None:
        self._apply_handlers(section=section)

    def _resolve_initial_profile(self, profile: Optional[str]) -> str:
        if profile:
            return normalize_profile_name(str(profile))

        # Check environment variables first (highest priority)
        env_profile = (
            os.getenv("TIDY3D_CONFIG_PROFILE")
            or os.getenv("TIDY3D_PROFILE")
            or os.getenv("TIDY3D_ENV")
        )
        if env_profile:
            return normalize_profile_name(env_profile)

        # Check for default_profile in config file
        config_default = self._loader.get_default_profile()
        if config_default:
            return normalize_profile_name(config_default)

        # Fall back to "default" profile
        return "default"

    def _reload(self) -> None:
        self._env_overrides = load_environment_overrides()
        self._builtin_data = deepcopy(self._loader.get_builtin_profile(self._profile))
        self._base_data = deepcopy(self._loader.load_base())
        self._profile_data = deepcopy(self._loader.load_user_profile(self._profile))
        self._raw_tree = deep_merge(self._builtin_data, self._base_data, self._profile_data)

        runtime = deepcopy(self._runtime_overrides.get(self._profile, {}))
        effective = deep_merge(self._raw_tree, self._env_overrides, runtime)
        self._effective_tree = effective
        self._build_models()

    def _build_models(self) -> None:
        sections = get_sections()
        new_sections: dict[str, BaseModel] = {}
        new_plugins: dict[str, BaseModel] = {}

        errors: list[tuple[str, Exception]] = []
        for name, schema in sections.items():
            if name.startswith("plugins."):
                plugin_name = name.split(".", 1)[1]
                plugin_data = _deep_get(self._effective_tree, ("plugins", plugin_name)) or {}
                try:
                    new_plugins[plugin_name] = schema(**plugin_data)
                except Exception as exc:
                    log.error(f"Failed to load configuration for plugin '{plugin_name}': {exc}")
                    errors.append((name, exc))
                continue
            if name == "plugins":
                continue
            section_data = self._effective_tree.get(name, {})
            try:
                new_sections[name] = schema(**section_data)
            except Exception as exc:
                log.error(f"Failed to load configuration for section '{name}': {exc}")
                errors.append((name, exc))

        if errors:
            # propagate the first error; others already logged
            raise errors[0][1]

        self._section_models = new_sections
        self._plugin_models = new_plugins

    def _get_model(self, name: str) -> Optional[BaseModel]:
        if name.startswith("plugins."):
            plugin = name.split(".", 1)[1]
            return self._plugin_models.get(plugin)
        return self._section_models.get(name)

    def _apply_handlers(self, section: Optional[str] = None) -> None:
        handlers = get_handlers()
        targets = [section] if section else handlers.keys()
        for target in targets:
            handler = handlers.get(target)
            if handler is None:
                continue
            model = self._get_model(target)
            if model is None:
                continue
            try:
                handler(model)
            except Exception as exc:
                log.error(f"Failed to apply configuration handler for '{target}': {exc}")

    def _compose_without_env(self) -> dict[str, Any]:
        runtime = self._runtime_overrides.get(self._profile, {})
        return deep_merge(self._raw_tree, runtime)

    def _default_tree(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for name, schema in get_sections().items():
            if name.startswith("plugins."):
                plugin = name.split(".", 1)[1]
                defaults.setdefault("plugins", {})[plugin] = _model_dict(schema())
            elif name == "plugins":
                defaults.setdefault("plugins", {})
            else:
                defaults[name] = _model_dict(schema())
        return defaults

    def _filter_persisted(self, tree: dict[str, Any]) -> dict[str, Any]:
        sections = get_sections()
        filtered: dict[str, Any] = {}
        plugins_source = tree.get("plugins", {})
        plugin_filtered: dict[str, Any] = {}

        for name, schema in sections.items():
            if name == "plugins":
                continue
            if name.startswith("plugins."):
                plugin_name = name.split(".", 1)[1]
                plugin_data = plugins_source.get(plugin_name, {})
                if not isinstance(plugin_data, dict):
                    continue
                persisted_plugin = _extract_persisted(schema, plugin_data)
                if persisted_plugin:
                    plugin_filtered[plugin_name] = persisted_plugin
                continue

            section_data = tree.get(name, {})
            if not isinstance(section_data, dict):
                continue
            persisted_section = _extract_persisted(schema, section_data)
            if persisted_section:
                filtered[name] = persisted_section

        if plugin_filtered:
            filtered["plugins"] = plugin_filtered
        return filtered

    def __getattr__(self, name: str) -> Any:
        if name in self._section_models:
            return SectionAccessor(self, name)
        if name == "plugins":
            return self.plugins
        raise AttributeError(f"Config has no section '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in self._section_models:
            if isinstance(value, BaseModel):
                payload = value.model_dump(exclude_unset=False)
            else:
                payload = value
            self.update_section(name, **payload)
            return
        object.__setattr__(self, name, value)

    def __str__(self) -> str:
        return self.format()


def _deep_get(tree: dict[str, Any], path: Iterable[str]) -> Optional[dict[str, Any]]:
    node: Any = tree
    for segment in path:
        if not isinstance(node, dict):
            return None
        node = node.get(segment)
        if node is None:
            return None
    return node if isinstance(node, dict) else None


def _resolve_model_type(annotation: Any) -> Optional[type[BaseModel]]:
    """Return the first BaseModel subclass found in an annotation (if any)."""

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        return None

    for arg in get_args(annotation):
        nested = _resolve_model_type(arg)
        if nested is not None:
            return nested
    return None


def _serialize_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_unset=False)
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return value


def _prepare_for_display(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {
            k: _prepare_for_display(v) for k, v in value.model_dump(exclude_unset=False).items()
        }
    if isinstance(value, dict):
        return {str(k): _prepare_for_display(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_prepare_for_display(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "get_secret_value"):
        displayed = getattr(value, "display", None)
        if callable(displayed):
            return displayed()
        return str(value)
    return value


def _build_config_panel(title: str, data: dict[str, Any]) -> Panel:
    tree = Tree(Text(title, style="bold cyan"))
    if data:
        for key in sorted(data.keys()):
            branch = tree.add(Text(key, style="bold magenta"))
            branch.add(Pretty(data[key], expand_all=True))
    else:
        tree.add(Text("<empty>", style="dim"))
    return Panel(tree, border_style="cyan", padding=(0, 1))


def _build_section_panel(name: str, data: Any) -> Panel:
    tree = Tree(Text(name, style="bold cyan"))
    tree.add(Pretty(data, expand_all=True))
    return Panel(tree, border_style="cyan", padding=(0, 1))


def _render_panel(renderable: Panel, *, width: int = 100) -> str:
    buffer = StringIO()
    console = Console(file=buffer, record=True, force_terminal=True, width=width, color_system=None)
    console.print(renderable)
    return buffer.getvalue().rstrip()


def _model_dict(model: BaseModel) -> dict[str, Any]:
    data = model.model_dump(exclude_unset=False)
    for key, value in list(data.items()):
        if hasattr(value, "get_secret_value"):
            data[key] = value.get_secret_value()
    return data


def _extract_persisted(schema: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    persisted: dict[str, Any] = {}
    for field_name, field in schema.model_fields.items():
        schema_extra = field.json_schema_extra or {}
        annotation = field.annotation
        persist = bool(schema_extra.get("persist")) if isinstance(schema_extra, dict) else False
        if not persist:
            continue
        if field_name not in data:
            continue
        value = data[field_name]
        if value is None:
            persisted[field_name] = None
            continue

        nested_type = _resolve_model_type(annotation)
        if nested_type is not None:
            nested_source = value if isinstance(value, dict) else {}
            nested_persisted = _extract_persisted(nested_type, nested_source)
            if nested_persisted:
                persisted[field_name] = nested_persisted
            continue

        if hasattr(value, "get_secret_value"):
            persisted[field_name] = value.get_secret_value()
        else:
            persisted[field_name] = deepcopy(value)

    return persisted


__all__ = [
    "ConfigManager",
    "PluginsAccessor",
    "ProfilesAccessor",
    "SectionAccessor",
    "normalize_profile_name",
]
