# all the autograd extras

from autograd.extend import primitive, defvjp, Box
import autograd as ag
import typing

# TODO: should we use ArrayBox? Box is more general
TracedFloat = float | Box

def get_static(x: typing.Any) -> typing.Any:
    """Get the 'static' (untraced) version of some value."""
    if isinstance(x, Box):
        return x._value
    return x

