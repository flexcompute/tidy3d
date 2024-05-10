# all the autograd extras

from autograd.extend import Box, primitive, defvjp
import typing

# TODO: should we use ArrayBox? Box is more general
TracedFloat = typing.Union[float, Box]


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
