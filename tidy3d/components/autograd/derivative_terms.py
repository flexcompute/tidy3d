"""Client-side derivative-program building blocks shared by every owner.

An owner turns its traced paths into keyed terms for the shared executor, plus bindings: the
chain rule from a term's outputs back to its traced paths. Both travel together as ``Compiled``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from flexcompute.core._migration.em.derivatives import Kernel, Key

from tidy3d.exceptions import AdjointError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .types import GeometryAdjointSettings

PathType = tuple[str | int, ...]


@dataclass(frozen=True)
class Binding:
    """The chain rule from one term's outputs to some traced paths; ``term=None`` is a zero.

    ``reduce(outputs, frequencies)`` takes the outputs and the frequencies they cover (all the
    program frequencies, or one frequency batch's) and returns each of ``paths``' VJP. One call
    fills them all, so a chain rule shared by several paths runs once. A returned VJP may share
    memory with ``outputs``, so VJPs are combined out of place.
    """

    paths: tuple[PathType, ...]
    term: Key | None
    reduce: Callable[[Mapping[Key, np.ndarray], np.ndarray], Mapping[PathType, Any]]

    @classmethod
    def one(cls, path: PathType, term: Key | None, reduce: Callable) -> Binding:
        """A binding of one path, whose ``reduce`` returns that path's VJP."""

        return cls((path,), term, partial(_one_path, path=path, reduce=reduce))


def _one_path(
    outputs: Mapping[Key, np.ndarray], frequencies: np.ndarray, *, path: PathType, reduce: Callable
) -> dict[PathType, Any]:
    return {path: reduce(outputs, frequencies)}


@dataclass(frozen=True)
class Compiled:
    """Terms, the monitors they read, and the bindings back to traced paths."""

    terms: tuple[tuple[Key, Kernel], ...] = ()
    monitors: tuple[Any, ...] = ()
    bindings: tuple[Binding, ...] = ()


@dataclass(frozen=True)
class ShapeDerivativeContext:
    """What a geometry's chain rule reads besides its sample sets; it holds no field data."""

    simulation_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    point_integrands: dict[PathType, np.ndarray]
    """Each sample set's per-point sensitivity, ``(N, 1)``, under the set's canonical key."""
    resolved_material_wavelength: float
    """The material wavelength the sample sets were generated with."""
    settings: GeometryAdjointSettings
    """The adjoint settings the sample sets were generated with.

    Compiling the derivative program takes one snapshot and its bindings carry it here,
    so generation and the chain rule read the same values."""


def combine(parts: Iterable[Compiled]) -> Compiled:
    """Concatenate compiled parts; monitors are shared by name and must agree."""

    parts = tuple(parts)
    monitors: dict[str, Any] = {}
    for monitor in (monitor for part in parts for monitor in part.monitors):
        if monitors.setdefault(monitor.name, monitor) != monitor:
            raise AdjointError(f"Two different derivative monitors are named '{monitor.name}'.")
    return Compiled(
        terms=tuple(term for part in parts for term in part.terms),
        monitors=tuple(monitors.values()),
        bindings=tuple(binding for part in parts for binding in part.bindings),
    )


__all__ = ["Binding", "Compiled", "PathType", "ShapeDerivativeContext", "combine"]
