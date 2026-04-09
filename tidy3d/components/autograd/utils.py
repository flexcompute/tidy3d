# utilities for working with autograd
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import autograd.numpy as anp
from autograd.tracer import getval, isbox

if TYPE_CHECKING:
    from typing import Union

    from autograd.numpy.numpy_boxes import ArrayBox
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "accumulate_field_map",
    "adjoint_fwidth_from_simulation",
    "asarray1d",
    "contains",
    "get_static",
    "hasbox",
    "is_tidy_box",
    "negate_vjp_map",
    "negate_vjp_value",
    "pack_complex_vec",
    "split_list",
]


def get_static(item: Any) -> Any:
    """
    Get the 'static' (untraced) version of some value by recursively calling getval
    on Box instances within a nested structure.
    """
    if isbox(item):
        return getval(item)
    elif isinstance(item, list):
        return [get_static(x) for x in item]
    elif isinstance(item, tuple):
        return tuple(get_static(x) for x in item)
    elif isinstance(item, dict):
        return {k: get_static(v) for k, v in item.items()}
    return item


def split_list(x: list[Any], index: int) -> tuple[list, list]:
    """Split a list at a given index."""
    x = list(x)
    return x[:index], x[index:]


def adjoint_fwidth_from_simulation(simulation: Any) -> float:
    """Return adjoint-source fwidth derived from the simulation normalization source."""

    sources = simulation.sources
    if not sources:
        raise ValueError(
            "Cannot determine adjoint source fwidth because the simulation has no sources."
        )

    normalize_index = simulation.normalize_index
    if normalize_index is None:
        normalize_index = 0
    if normalize_index < 0 or normalize_index >= len(sources):
        raise ValueError(
            f"Invalid normalize_index {normalize_index} for simulation with {len(sources)} sources."
        )

    return sources[normalize_index].source_time.fwidth


def is_tidy_box(x: Any) -> bool:
    """Check if a value is a tidy box."""
    return getattr(x, "_tidy", False)


def contains(target: Any, seq: Iterable[Any]) -> bool:
    """Return ``True`` if target occurs anywhere within arbitrarily nested iterables."""
    for x in seq:
        if x == target:
            return True
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            if contains(target, x):
                return True
    return False


def hasbox(obj: Any) -> bool:
    """True if any element inside obj is an autograd Box."""
    if isbox(obj):
        return True
    if is_dataclass(obj) and not isinstance(obj, type):
        return any(hasbox(getattr(obj, field.name)) for field in fields(obj))
    if isinstance(obj, Mapping):
        return any(hasbox(v) for v in obj.values())
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return any(hasbox(i) for i in obj)
    return False


def pack_complex_vec(z: Union[NDArray, ArrayBox]) -> Union[NDArray, ArrayBox]:
    """Ravel [Re(z); Im(z)] into one real vector (autograd-safe)."""
    return anp.concatenate([anp.ravel(anp.real(z)), anp.ravel(anp.imag(z))])


def asarray1d(x: Union[ArrayLike, ArrayBox]) -> Union[NDArray, ArrayBox]:
    """Autograd-friendly 1D flatten: returns ndarray of shape (-1,)."""
    x = anp.array(x)
    return x if x.ndim == 1 else anp.ravel(x)


def negate_vjp_value(value: Any) -> Any:
    """Negate a VJP value while preserving list/tuple container types."""
    if isinstance(value, tuple):
        return tuple(negate_vjp_value(v) for v in value)
    if isinstance(value, list):
        return [negate_vjp_value(v) for v in value]
    return -value


def negate_vjp_map(vjp_map: Mapping[Any, Any]) -> dict[Any, Any]:
    """Negate all values in a VJP map with type-preserving value handling."""
    return {path: negate_vjp_value(value) for path, value in vjp_map.items()}


def accumulate_field_map(target: dict, addition: dict) -> None:
    """Accumulate an autograd field map into target in-place."""
    for k, v in addition.items():
        if k in target:
            existing = target[k]
            if isinstance(existing, (list, tuple)) and isinstance(v, (list, tuple)):
                target[k] = type(existing)(x + y for x, y in zip(existing, v))
            else:
                target[k] = existing + v
        else:
            target[k] = v
