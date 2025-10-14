from __future__ import annotations

import typing
from collections.abc import Hashable

import tidy3d as td
from tidy3d.components.autograd import AutogradFieldMap
from tidy3d.components.autograd.types import NumericalStructureInfo


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
