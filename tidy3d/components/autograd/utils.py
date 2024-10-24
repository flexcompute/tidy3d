# utilities for working with autograd

import typing

import autograd.numpy as anp
import numpy as np
from autograd.tracer import getval, isbox


def _is_traced_objary(x: typing.Any) -> bool:
    """Check whether x is a traced object array."""
    return isinstance(x, np.ndarray) and x.size > 0 and isbox(x.flat[0])


def is_traced(x: typing.Any) -> bool:
    """Check whether x is traced (contains an autograd Box)"""
    return isbox(x) or _is_traced_objary(x)


def get_static(x: typing.Any) -> typing.Any:
    """Get the 'static' (untraced) version of some value."""
    return getval(x)


def get_tracer(x: typing.Any) -> typing.Any:
    """Get the 'boxed' (traced) version of some value."""
    if not is_traced(x):
        return None

    if _is_traced_objary(x):
        return anp.array(x.tolist())

    return x


def split_list(x: list[typing.Any], index: int) -> (list[typing.Any], list[typing.Any]):
    """Split a list at a given index."""
    x = list(x)
    return x[:index], x[index:]


def is_tidy_box(x: typing.Any) -> bool:
    """Check if a value is a tidy box."""
    return getattr(x, "_tidy", False)


__all__ = [
    "get_static",
    "get_tracer",
    "is_traced",
    "split_list",
    "is_tidy_box",
]
