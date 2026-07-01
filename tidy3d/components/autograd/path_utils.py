"""Utilities for user-facing autograd path messages."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

from tidy3d.components.autograd.types import PathType
from tidy3d.exceptions import AdjointError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

PathInput = str | PathType


@dataclass(frozen=True)
class AutogradRoute:
    """Validated route metadata for a traced autograd path."""

    local_path: PathType
    """Path relative to the component that resolved it."""


def traced_paths(*paths: PathInput) -> tuple[PathType, ...]:
    """Build ordered traced paths from bare roots or explicit path tuples."""
    return tuple((path,) if isinstance(path, str) else tuple(path) for path in paths)


def indexed_traced_paths(root: str, size: int) -> tuple[PathType, ...]:
    """Build ``root[0]`` through ``root[size - 1]`` traced paths."""
    return tuple((root, index) for index in range(size))


def format_traced_path(path: tuple[Any, ...]) -> str:
    """Format an internal traced field path as a user-facing parameter name."""
    path = tuple(path)
    if not path:
        return "<empty>"

    formatted = str(path[0])
    for part in path[1:]:
        if isinstance(part, Integral):
            formatted += f"[{part}]"
        else:
            formatted += f".{part}"
    return formatted


def format_traced_paths(paths: tuple[PathType, ...]) -> tuple[str, ...]:
    """Format supported traced paths as user-facing parameter names."""
    return tuple(format_traced_path(path) for path in paths)


def _supported_parameters_message(
    *,
    parameter_kind: str,
    owner_kind: str,
    supported_parameters: tuple[str, ...],
) -> str:
    """Describe which parameters may be traced."""
    if supported_parameters:
        supported = ", ".join(f"'{parameter}'" for parameter in supported_parameters)
        return f"Supported {parameter_kind} parameters are: {supported}."
    return f"This {owner_kind} does not support traced {parameter_kind} parameters."


def raise_unsupported_traced_path(
    *,
    parameter_kind: str,
    owner_kind: str,
    owner_name: str,
    field_path: tuple[Any, ...],
    supported_parameters: tuple[str, ...] = (),
) -> None:
    """Raise a user-facing validation error for an unsupported traced path."""
    parameter = format_traced_path(field_path)
    raise AdjointError(
        f"Automatic differentiation with respect to {parameter_kind} parameter '{parameter}' "
        f"is not supported for {owner_kind} '{owner_name}'. "
        f"{_supported_parameters_message(parameter_kind=parameter_kind, owner_kind=owner_kind, supported_parameters=supported_parameters)}"
    )


def validate_traced_path(
    *,
    parameter_kind: str,
    owner_kind: str,
    owner_name: str,
    field_path: tuple[Any, ...],
    supported_paths: Collection[tuple[Any, ...]],
    supported_parameters: tuple[str, ...],
) -> AutogradRoute:
    """Validate one traced path against an explicit set of supported paths."""
    if field_path not in supported_paths:
        raise_unsupported_traced_path(
            parameter_kind=parameter_kind,
            owner_kind=owner_kind,
            owner_name=owner_name,
            field_path=field_path,
            supported_parameters=supported_parameters,
        )
    return AutogradRoute(local_path=field_path)


def raise_with_traced_path_context(
    error: AdjointError,
    *,
    parameter_kind: str,
    local_path: tuple[Any, ...],
    full_path: tuple[Any, ...],
) -> None:
    """Re-raise a delegated validation error with the full user-facing parameter path."""
    local_parameter = format_traced_path(local_path)
    full_parameter = format_traced_path(full_path)
    old = f"{parameter_kind} parameter '{local_parameter}'"
    new = f"{parameter_kind} parameter '{full_parameter}'"

    message = str(error)
    if old not in message:
        raise error

    raise AdjointError(message.replace(old, new, 1)) from error


def resolve_delegated_autograd_route(
    *,
    parameter_kind: str,
    owner_kind: str,
    owner_name: str,
    field_path: tuple[Any, ...],
    delegates: Mapping[Any, Any],
    supported_parameters: tuple[str, ...],
) -> AutogradRoute:
    """Resolve a prefixed traced path by delegating to the selected child component."""
    if len(field_path) < 2 or field_path[0] not in delegates:
        raise_unsupported_traced_path(
            parameter_kind=parameter_kind,
            owner_kind=owner_kind,
            owner_name=owner_name,
            field_path=field_path,
            supported_parameters=supported_parameters,
        )

    sub_path = field_path[1:]
    try:
        delegates[field_path[0]]._resolve_autograd_route(sub_path)
    except AdjointError as err:
        raise_with_traced_path_context(
            err,
            parameter_kind=parameter_kind,
            local_path=sub_path,
            full_path=field_path,
        )

    return AutogradRoute(local_path=field_path)
