# all the autograd extras
import numpy as np

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


def get_structure_indices(sim_field_mapping: list[str]) -> list[int]:
    """Get unique list of structures indices from a list of field mapping paths."""
    structure_indices = [get_structure_index(path) for path in sim_field_mapping]
    return np.unique(structure_indices)


def get_structure_index(field_mapping: str) -> int:
    """Get the index of a structure from a specific field map."""
    return int(field_mapping.split("/")[1])


def get_field_key(field_mapping: str) -> tuple[str, ...]:
    """Get the key for this specific field."""
    return tuple(field_mapping.split("/")[-2:])


def make_field_path(index: int, key: str) -> str:
    """Turn an index and a field tuple into a path into the structures list."""
    field_key = "/".join(key)
    return f"structures/{index}/{field_key}"


__all__ = [
    "Box",
    "primitive",
    "defvjp",
    "get_static",
    "adjoint_mnt_fld_name",
    "adjoint_mnt_eps_name",
]
