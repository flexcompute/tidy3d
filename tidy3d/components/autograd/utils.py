# utilities for working with autograd

import typing

from autograd.tracer import getval


def get_static(x: typing.Any) -> typing.Any:
    """Get the 'static' (untraced) version of some value."""
    return getval(x)


def split_list(x: list[typing.Any], index: int) -> (list[typing.Any], list[typing.Any]):
    """Split a list at a given index."""
    x = list(x)
    return x[:index], x[index:]


__all__ = [
    "get_static",
    "split_list",
]
