"""Registry utilities for tidy3d configuration sections and handlers."""

from __future__ import annotations

from typing import Callable, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_SECTIONS: dict[str, type[BaseModel]] = {}
_HANDLERS: dict[str, Callable[[BaseModel], None]] = {}
_MANAGER: Optional[ConfigManagerProtocol] = None


class ConfigManagerProtocol:
    """Protocol-like interface for manager notifications."""

    def on_section_registered(self, section: str) -> None:
        """Called when a new section schema is registered."""

    def on_handler_registered(self, section: str) -> None:
        """Called when a handler is registered."""


def attach_manager(manager: ConfigManagerProtocol) -> None:
    """Attach the active configuration manager for registry callbacks."""

    global _MANAGER
    _MANAGER = manager


def get_manager() -> Optional[ConfigManagerProtocol]:
    """Return the currently attached configuration manager, if any."""

    return _MANAGER


def register_section(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a configuration section schema."""

    def decorator(cls: type[T]) -> type[T]:
        _SECTIONS[name] = cls
        if _MANAGER is not None:
            _MANAGER.on_section_registered(name)
        return cls

    return decorator


def register_plugin(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a plugin configuration schema."""

    return register_section(f"plugins.{name}")


def register_handler(
    name: str,
) -> Callable[[Callable[[BaseModel], None]], Callable[[BaseModel], None]]:
    """Decorator to register a handler for a configuration section."""

    def decorator(func: Callable[[BaseModel], None]) -> Callable[[BaseModel], None]:
        _HANDLERS[name] = func
        if _MANAGER is not None:
            _MANAGER.on_handler_registered(name)
        return func

    return decorator


def get_sections() -> dict[str, type[BaseModel]]:
    """Return registered section schemas."""

    return dict(_SECTIONS)


def get_handlers() -> dict[str, Callable[[BaseModel], None]]:
    """Return registered configuration handlers."""

    return dict(_HANDLERS)
