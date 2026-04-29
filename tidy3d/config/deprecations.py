"""Helpers for warning about deprecated configuration keys."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tidy3d.log import log

from .schema_utils import _resolve_model_type

if TYPE_CHECKING:
    from pydantic import BaseModel


def _normalize_version(value: Any, path: str, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        log.warning(f"Ignoring invalid {label}={value!r} on '{path}'.")
        return None
    if value < 0:
        log.warning(f"Ignoring invalid {label}={value!r} on '{path}'.")
        return None
    return value


def check_deprecations(
    schema: type[BaseModel],
    data: dict[str, Any],
    prefix: tuple[str, ...],
    *,
    current_version: int | None = None,
) -> None:
    """Warn or error when deprecated/removed fields are present."""

    if current_version is None:
        from .migrations import CURRENT_CONFIG_VERSION

        active_version = CURRENT_CONFIG_VERSION
    else:
        active_version = current_version

    for field_name, field in schema.model_fields.items():
        if field_name not in data:
            continue
        field_path = ".".join((*prefix, field_name))
        schema_extra = field.json_schema_extra or {}
        if isinstance(schema_extra, dict):
            deprecated_in = _normalize_version(
                schema_extra.get("deprecated_in"), field_path, "deprecated_in"
            )
            removed_in = _normalize_version(
                schema_extra.get("removed_in"), field_path, "removed_in"
            )
            replaced_by = schema_extra.get("replaced_by")
            if deprecated_in is not None and removed_in is not None:
                if removed_in < deprecated_in + 2:
                    raise ValueError(
                        f"Deprecation metadata for '{field_path}' violates the minimum window "
                        f"(removed_in={removed_in}, deprecated_in={deprecated_in})."
                    )
            if removed_in is not None and active_version >= removed_in:
                raise ValueError(
                    f"Configuration key '{field_path}' was removed in config schema v{removed_in}."
                )
            if deprecated_in is not None and active_version >= deprecated_in:
                message = (
                    f"Configuration key '{field_path}' is deprecated in schema v{deprecated_in}."
                )
                if replaced_by:
                    message = f"{message} Use '{replaced_by}' instead."
                log.warning(message, log_once=True)

        nested_model = _resolve_model_type(field.annotation)
        nested_value = data.get(field_name)
        if nested_model is not None:
            if isinstance(nested_value, dict):
                check_deprecations(
                    nested_model,
                    nested_value,
                    (*prefix, field_name),
                    current_version=active_version,
                )
            elif isinstance(nested_value, list):
                for item in nested_value:
                    if isinstance(item, dict):
                        check_deprecations(
                            nested_model,
                            item,
                            (*prefix, field_name),
                            current_version=active_version,
                        )


__all__ = ["check_deprecations"]
