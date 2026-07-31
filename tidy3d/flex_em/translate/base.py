"""Shared helpers for backend Tidy3D schema translation."""

from __future__ import annotations

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


def _drop_closed_public_log_file_handler() -> bool:
    """Remove a stale public Tidy3D file log handler left with a closed file."""

    try:
        from tidy3d.log import log
    except Exception:
        return False

    handler = log.handlers.get("file")
    file_handle = getattr(getattr(handler, "console", None), "file", None)
    if handler is None or not getattr(file_handle, "closed", False):
        return False
    del log.handlers["file"]
    return True


def validate_public_data(public_type: type[Any], payload: dict[str, Any]) -> Any:
    """Validate schema result payloads as public Tidy3D data without stale file logging."""

    _drop_closed_public_log_file_handler()
    try:
        return public_type.model_validate(payload)
    except Exception as exc:
        if "I/O operation on closed file" not in str(exc):
            raise
        if not _drop_closed_public_log_file_handler():
            raise
        return public_type.model_validate(payload)


def dump_for_public(obj: Any, *, type_name: str) -> dict[str, Any]:
    """Dump a schema model and set the public Tidy3D discriminator."""

    payload = obj.model_dump(mode="python")
    payload["type"] = type_name
    return payload
