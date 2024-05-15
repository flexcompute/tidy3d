# utilities for working with autograd

from autograd.extend import Box, primitive, defvjp
from autograd.builtins import dict as dict_ag
import typing

# TODO: should we use ArrayBox? Box is more general
TracedFloat = typing.Union[float, Box]

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
