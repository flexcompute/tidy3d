"""Shared helpers for backend Tidy3D schema translation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def apply_type_field_map(payload: Any, type_field_map: Mapping[str, Mapping[str, str]]) -> Any:
    """Rename fields according to the current payload object's type."""

    if isinstance(payload, dict):
        field_map = type_field_map.get(payload.get("type"), {})
        mapped = {
            field_map.get(key, key): apply_type_field_map(value, type_field_map)
            for key, value in payload.items()
        }
        return mapped
    if isinstance(payload, list):
        return [apply_type_field_map(value, type_field_map) for value in payload]
    if isinstance(payload, tuple):
        return tuple(apply_type_field_map(value, type_field_map) for value in payload)
    return payload


def dump_for_schema(obj: Any, *, type_name: str) -> dict[str, Any]:
    """Dump a public Tidy3D model and set the copied-schema discriminator."""

    payload = obj.model_dump(mode="python")
    payload["type"] = type_name
    return payload


def dump_data_for_schema(obj: Any) -> dict[str, Any]:
    """Dump public Tidy3D result data for schema validation."""

    return obj.model_dump(mode="python")


def dump_data_for_public(obj: Any) -> dict[str, Any]:
    """Dump schema result data for public Tidy3D validation."""

    return obj.model_dump(mode="python")


def _drop_closed_file_log_handlers() -> bool:
    """Remove stale public or copied-schema file handlers left by pipeline logging."""

    removed = False
    for module_name in ("tidy3d.log", "flex_em.schema.tidy3d.log"):
        try:
            log = import_module(module_name).log
        except (ImportError, AttributeError):
            continue

        handler = log.handlers.get("file")
        file_handle = getattr(getattr(handler, "console", None), "file", None)
        if handler is not None and getattr(file_handle, "closed", False):
            del log.handlers["file"]
            removed = True
    return removed


def validate_data(data_type: type[Any], payload: dict[str, Any]) -> Any:
    """Validate translated result data without stale pipeline file logging."""

    _drop_closed_file_log_handlers()
    try:
        return data_type.model_validate(payload)
    except Exception as exc:
        if "I/O operation on closed file" not in str(exc):
            raise
        if not _drop_closed_file_log_handlers():
            raise
        return data_type.model_validate(payload)


def validate_public_data(public_type: type[Any], payload: dict[str, Any]) -> Any:
    """Validate schema result payloads as public Tidy3D data without stale file logging."""

    return validate_data(public_type, payload)


def dump_for_public(obj: Any, *, type_name: str) -> dict[str, Any]:
    """Dump a schema model and set the public Tidy3D discriminator."""

    payload = obj.model_dump(mode="python")
    payload["type"] = type_name
    return payload
