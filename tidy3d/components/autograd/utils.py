# utilities for working with autograd

import typing

from autograd.tracer import getval


def get_static(x: typing.Any) -> typing.Any:
    """Get the 'static' (untraced) version of some value."""
    return getval(x)


__all__ = [
    "get_static",
]
