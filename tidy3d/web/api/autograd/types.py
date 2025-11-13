from __future__ import annotations

import typing
from collections.abc import Hashable
from dataclasses import dataclass

import tidy3d as td
from tidy3d.components.autograd import AutogradFieldMap
from tidy3d.components.autograd.types import NumericalStructureInfo


@dataclass
class NumericalStructureConfig:
    create: typing.Callable
    """Function that creates the structure given an untraced version of the parameters"""

    compute_derivatives: typing.Callable
    """Function that computes the vjp for the structure given the same arguments
    that the internal _compute_derivatives function gets."""

    parameters: typing.Any
    """Parameters used for creating the structure."""

    # we could consider making this Optional and if it is not specified, we could
    # just append it to the structures list in the simulation
    structure_index: typing.Optional[int] = -1
    """Index for structure in the simulation. If not specified, assume the structure is appended into the structure list."""


@dataclass
class UserVJPConfig:
    structure_index: int
    """Index for structure to replace vjp."""

    compute_derivatives: typing.Callable
    """Function that computes the vjp for the structure given the same arguments
    that the internal _compute_derivatives function gets."""

    path_key: typing.Optional[str] = None
    """Path key this is relevant for. If not specified, assume the supplied function applies for all keys."""


class UserVjpEntry(typing.NamedTuple):
    structure_index: int
    path: tuple[Hashable, ...]
    fn: typing.Callable[..., typing.Any]


UserVjpSpec = tuple[UserVjpEntry, ...]


class SetupRunResult(typing.NamedTuple):
    sim_fields: AutogradFieldMap
    simulation: td.Simulation
    numerical_info: dict[int, NumericalStructureInfo]


__all__ = [
    "SetupRunResult",
    "UserVjpEntry",
    "UserVjpSpec",
]
