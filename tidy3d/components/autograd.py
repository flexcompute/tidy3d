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


def adjoint_mnt_fld_name(i: int) -> str:
    """Name of field monitor for adjoint gradient for structure i"""
    return f"adjoint_fld_{i}"


def adjoint_mnt_eps_name(i: int) -> str:
    """Name of permittivity monitor for adjoint gradient for structure i"""
    return f"adjoint_eps_{i}"


__all__ = [
    "Box",
    "primitive",
    "defvjp",
    "get_static",
    "adjoint_mnt_fld_name",
    "adjoint_mnt_eps_name",
]
