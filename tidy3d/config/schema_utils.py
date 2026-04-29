"""Shared schema helpers for configuration models."""

from __future__ import annotations

from typing import Any, get_args, get_origin

from pydantic import BaseModel

TOP_LEVEL_METADATA_KEYS = frozenset({"default_profile"})


def _resolve_model_type(annotation: Any) -> type[BaseModel] | None:
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


__all__ = ["TOP_LEVEL_METADATA_KEYS", "_resolve_model_type"]
