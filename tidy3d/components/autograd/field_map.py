"""Typed containers for autograd traced field metadata."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import Field

from tidy3d.components.autograd.types import TracedArrayLike, TracedComplex, TracedFloat
from tidy3d.components.base import Tidy3dBaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.components.autograd.types import AutogradFieldMap


class Tracer(Tidy3dBaseModel):
    """Representation of a single traced element within a model."""

    path: tuple[Any, ...] = Field(
        title="Path to the traced object in the model dictionary.",
    )
    data: TracedFloat | TracedComplex | TracedArrayLike = Field(title="Tracing data")


class FieldMap(Tidy3dBaseModel):
    """Collection of traced elements."""

    tracers: tuple[Tracer, ...] = Field(
        title="Collection of Tracers.",
    )

    @property
    def to_autograd_field_map(self) -> AutogradFieldMap:
        """Convert to ``AutogradFieldMap`` autograd dictionary."""
        return {tracer.path: tracer.data for tracer in self.tracers}

    @classmethod
    def from_autograd_field_map(cls, autograd_field_map: AutogradFieldMap) -> FieldMap:
        """Initialize from an ``AutogradFieldMap`` autograd dictionary."""
        tracers = []
        for path, data in autograd_field_map.items():
            tracers.append(Tracer(path=path, data=data))
        return cls(tracers=tuple(tracers))


def _encoded_path(path: tuple[Any, ...]) -> str:
    """Return a stable JSON representation for a traced path."""
    return json.dumps(list(path), separators=(",", ":"), ensure_ascii=True)


class TracerKeys(Tidy3dBaseModel):
    """Collection of traced field paths."""

    keys: tuple[tuple[Any, ...], ...] = Field(
        title="Collection of tracer keys.",
    )

    def encoded_keys(self) -> list[str]:
        """Return the JSON-encoded representation of keys."""
        return [_encoded_path(path) for path in self.keys]

    @classmethod
    def from_field_mapping(
        cls,
        field_mapping: AutogradFieldMap,
        *,
        sort_key: Callable[[tuple[Any, ...]], str] | None = None,
    ) -> TracerKeys:
        """Construct keys from an autograd field mapping."""
        if sort_key is None:
            sort_key = _encoded_path

        sorted_paths = tuple(sorted(field_mapping.keys(), key=sort_key))
        return cls(keys=sorted_paths)
