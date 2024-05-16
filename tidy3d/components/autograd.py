# utilities for working with autograd

from autograd.extend import Box, primitive, defvjp
from autograd.builtins import dict as dict_ag


import typing
from .types import Size1D

# TODO: should we use ArrayBox? Box is more general

# Types for floats, or collections of floats that can also be autograd tracers
TracedFloat = typing.Union[float, Box]
TracedSize1D = typing.Union[Size1D, Box]
TracedSize = typing.Union[tuple[TracedSize1D, TracedSize1D, TracedSize1D], Box]
TracedCoordinate = typing.Union[tuple[TracedFloat, TracedFloat, TracedFloat], Box]

# The data type that we pass in and out of the web.run() @autograd.primitive
AutogradFieldMap = dict_ag[tuple[str, ...], TracedFloat]


def get_static(x: typing.Any) -> typing.Any:
    """Get the 'static' (untraced) version of some value."""
    if isinstance(x, Box):
        return get_static(x._value)
    return x


__all__ = [
    "Box",
    "primitive",
    "defvjp",
    "get_static",
]
