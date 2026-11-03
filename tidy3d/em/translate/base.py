"""Shared helpers for backend Tidy3D schema translation."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    # Maps one dumped model to the other side's payload.
    Adapter = Callable[[dict[str, Any]], dict[str, Any]]


def _drop_fields(*fields: str) -> Adapter:
    """Adapter dropping schema-only fields, which public models reject."""

    return lambda payload: {key: value for key, value in payload.items() if key not in fields}


@cache
def _translators() -> dict[str, Adapter]:
    """The translator of each registered class with one, keyed by its type tag."""

    from tidy3d.em.translate.map import COMPONENT_SCOPES, resolve

    return {
        key.rpartition(":")[2]: resolve(scope.translator)
        for key, scope in COMPONENT_SCOPES.items()
        if scope is not None and scope.translator is not None
    }


# Fields the schema adds, which public models reject, keyed by type tag. Types not listed
# translate back as copies.
SCHEMA_TO_PUBLIC_ADAPTERS: dict[str, Adapter] = {
    "BroadbandPulse": _drop_fields("gaussian_pulses", "pulse_freq0", "pulse_end_time"),
    # The backend subpixel subclasses add this field.
    "Simulation": _drop_fields("subpixel_scheme"),
    "ModeSimulation": _drop_fields("subpixel_scheme"),
}


def _adapt(payload: Any, adapters: Mapping[str, Adapter]) -> Any:
    """Apply ``adapters`` to every dumped model in the payload by its type tag, innermost first.

    ``attrs`` holds user metadata, so a ``type`` key inside it is not a type tag.
    """

    # Translators pass a fresh dump; an empty map needs neither a walk nor a copy.
    if not adapters:
        return payload
    if isinstance(payload, dict):
        payload = {
            key: value if key == "attrs" else _adapt(value, adapters)
            for key, value in payload.items()
        }
        tag = payload.get("type")
        adapter = adapters.get(tag) if isinstance(tag, str) else None
        return adapter(payload) if adapter else payload
    if isinstance(payload, list):
        return [_adapt(value, adapters) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_adapt(value, adapters) for value in payload)
    return payload


def apply_translators(payload: Any) -> Any:
    """Apply each registered class's translator to its dumped models in the payload.

    The ``BroadbandPulse`` translator needs ``tidy3d-extras``.
    """

    return _adapt(payload, _translators())


def dump_for_schema(obj: Any, *, type_name: str) -> dict[str, Any]:
    """Dump a public Tidy3D model as a schema payload with the copied-schema discriminator."""

    payload = obj.model_dump(mode="python")
    payload["type"] = type_name
    return apply_translators(payload)


def dump_for_public(obj: Any, *, type_name: str) -> dict[str, Any]:
    """Dump a schema model as a public Tidy3D payload with the public discriminator."""

    payload = obj.model_dump(mode="python")
    payload["type"] = type_name
    return _adapt(payload, SCHEMA_TO_PUBLIC_ADAPTERS)


def dump_data_for_schema(obj: Any) -> dict[str, Any]:
    """Dump public Tidy3D result data as a schema payload."""

    return apply_translators(obj.model_dump(mode="python"))


def dump_data_for_public(obj: Any) -> dict[str, Any]:
    """Dump schema result data as a public Tidy3D payload."""

    return _adapt(obj.model_dump(mode="python"), SCHEMA_TO_PUBLIC_ADAPTERS)


def _drop_closed_file_log_handlers() -> bool:
    """Remove stale public or copied-schema file handlers left by pipeline logging."""

    removed = False
    for module_name in ("tidy3d.log", "flexcompute.core.log"):
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
