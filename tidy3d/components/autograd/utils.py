# utilities for working with autograd
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from autograd.tracer import getval


def get_static(x: Any) -> Any:
    """Get the 'static' (untraced) version of some value."""
    return getval(x)


def split_list(x: list[Any], index: int) -> (list[Any], list[Any]):
    """Split a list at a given index."""
    x = list(x)
    return x[:index], x[index:]


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


__all__ = [
    "contains",
    "get_static",
    "is_tidy_box",
    "split_list",
]
