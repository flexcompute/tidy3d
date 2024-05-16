# utilities for working with autograd

from autograd.extend import Box, primitive, defvjp
from autograd.builtins import dict as dict_ag

import typing
from .types import Size1D


# TODO: should we use ArrayBox? Box is more general
TracedFloat = typing.Union[float, Box]


TracedSize1D = typing.Union[Size1D, Box]

TracedSize = typing.Union[tuple[TracedSize1D, TracedSize1D, TracedSize1D], Box]
TracedCoordinate = typing.Union[tuple[TracedFloat, TracedFloat, TracedFloat], Box]

AutogradFieldMap = dict_ag[tuple[str, ...], TracedFloat]


def get_static(x: typing.Any) -> typing.Any:
    """Get the 'static' (untraced) version of some value."""
    if isinstance(x, Box):
        return x._value
    return x


__all__ = [
    "Box",
    "primitive",
    "defvjp",
    "get_static",
]
