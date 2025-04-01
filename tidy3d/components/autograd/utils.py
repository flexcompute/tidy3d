# utilities for working with autograd
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from autograd.tracer import getval, isbox


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


def split_list(x: list[Any], index: int) -> (list[Any], list[Any]):
    """Split a list at a given index."""
    x = list(x)
    return x[:index], x[index:]


def is_tidy_box(x: Any) -> bool:
    """Check if a value is a tidy box."""
    return getattr(x, "_tidy", False)


def hasbox(obj: Any) -> bool:
    """True if any element inside obj is an autograd Box."""
    if isbox(obj):
        return True
    if isinstance(obj, Mapping):
        return any(hasbox(v) for v in obj.values())
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return any(hasbox(i) for i in obj)
    return False


__all__ = [
    "get_static",
    "hasbox",
    "is_tidy_box",
    "split_list",
]
