"""Derivative-program pieces shared by the medium classes.

A structure hands its medium a ``MediumTarget``: how the medium's terms are keyed, which volume
monitor and components they read, and the box they integrate over. Parents narrow it for their
component media.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from flexcompute.core._migration.em.derivatives import (
    FieldProductKernel,
    Integrate,
    Key,
    RectilinearGrid,
)

from tidy3d.components.autograd.derivative_terms import Binding, Compiled
from tidy3d.components.autograd.utils import get_static

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from tidy3d.components.autograd.derivative_terms import PathType


@dataclass(frozen=True)
class MediumTarget:
    """How a medium's terms are keyed, what they read, and where they integrate."""

    key: Key
    """Term key and traced-path prefix, e.g. ``("structures", 2, "medium")``."""
    components: tuple[str, ...]
    """Diagonal components the terms may read."""
    monitor: str
    """Volume monitor recording those components."""
    bounds: np.ndarray
    """``(2, 3)`` integration box."""
    frequencies: np.ndarray
    interpolation: str | None = None
    """A parent's override of a custom medium's own grid interpolation."""
    resolved: bool = False
    """Keep custom-medium sensitivities per frequency (a numerical query's ``resolved`` mode)."""

    def narrowed(self, component: str) -> MediumTarget:
        """The target of the child medium of one diagonal component."""

        return replace(self, key=(*self.key, component), components=(component,))


def uniform_terms(medium: Any, paths: Iterable[PathType], target: MediumTarget) -> Compiled:
    """One box-integrated term; each path applies the medium's chain rule to its components."""

    kernel = FieldProductKernel(
        monitor=target.monitor,
        components=target.components,
        reductions=(Integrate(bounds=target.bounds, edges="clip_trapezoid"),),
    )
    bindings = tuple(
        Binding.one(
            (*target.key, *path),
            target.key,
            partial(
                _stacked,
                rule=medium._derivative_rule(path),
                components=target.components,
            ),
        )
        for path in paths
    )
    return Compiled(terms=((target.key, kernel),), bindings=bindings)


def _stacked(
    outputs: Mapping[Key, np.ndarray], frequencies: np.ndarray, *, rule: Callable, components: tuple
) -> Any:
    return rule(np.stack([outputs[(component,)] for component in components]), frequencies)


def zero_binding(path: PathType, like: Any) -> Binding:
    """A traced path whose VJP is identically zero, in the shape and dtype of ``like``."""

    zero = np.zeros_like(np.asarray(get_static(like)))
    return Binding.one(path, None, lambda _outputs, _frequencies: zero)


def projection(target: MediumTarget, sorted_data: Any, interpolation: str) -> Integrate:
    """Transpose-project onto a parameter grid with the custom-media edge rule."""

    x, y, z = (np.asarray(sorted_data.coords[axis].values, dtype=float) for axis in "xyz")
    grid = RectilinearGrid(x=x, y=y, z=z, interpolation=(interpolation,) * 3)
    return Integrate(bounds=target.bounds, edges="crop_cropped_widths", onto=grid)


def unsort(values: np.ndarray, data: Any) -> np.ndarray:
    """Values on ``data``'s spatially sorted grid (leading axes x, y, z), back in its order."""

    values = np.asarray(values)
    for axis, dim in enumerate("xyz"):
        coordinates = np.asarray(data.coords[dim].values, dtype=float)
        order = np.argsort(np.argsort(coordinates, kind="stable"), kind="stable")
        if not np.array_equal(order, np.arange(order.size)):
            values = np.take(values, order, axis=axis)
    return values


def restore(values: np.ndarray, data: Any, *, frequency_axis: bool = False) -> np.ndarray:
    """A sensitivity on the sorted parameter grid, back in the public array's order and dtype."""

    values = unsort(values, data)
    if frequency_axis:
        values = values[..., None]
    dtype = np.asarray(get_static(data.values)).dtype
    if np.issubdtype(dtype, np.complexfloating):
        return values.astype(dtype, copy=False)
    return np.real(values).astype(dtype, copy=False)


def restore_output(
    outputs: Mapping[Key, np.ndarray], frequencies: np.ndarray, *, key: Key, data: Any
) -> np.ndarray:
    """One output, restored to the public layout of ``data``."""

    return restore(outputs[key], data)


__all__ = [
    "MediumTarget",
    "projection",
    "restore",
    "restore_output",
    "uniform_terms",
    "unsort",
    "zero_binding",
]
