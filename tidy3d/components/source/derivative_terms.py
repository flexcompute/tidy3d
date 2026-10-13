"""Derivative-program pieces shared by the source classes.

The owner hands each traced source a ``SourceTarget``. The source builds one adjoint field monitor
over its own box and the ``SourcePullbackKernel`` terms that pull the adjoint field back onto its
profile grid, weighted by the source spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from flexcompute.core._migration.em.derivatives import (
    Key,
    SourceProfile,
    SourcePullback,
    SourcePullbackKernel,
    SumFrequencies,
)

from tidy3d.components.autograd.derivative_terms import Binding, Compiled
from tidy3d.components.autograd.utils import get_static
from tidy3d.components.medium.derivative_terms import unsort, zero_binding
from tidy3d.components.monitor import FieldMonitor
from tidy3d.constants import C_0

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from tidy3d.components.autograd.derivative_terms import PathType

_FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_ORDER = ("f", "x", "y", "z")


@dataclass(frozen=True)
class SourceTarget:
    """How a source's terms are keyed, the monitor they read, and the run they belong to."""

    key: Key
    """Term key and traced-path prefix, e.g. ``("sources", 1)``."""
    monitor: str
    """Name of the adjoint field monitor over the source."""
    frequencies: np.ndarray
    simulation: Any
    """The run: its time mesh weights the spectrum, its grid and background set beam profiles."""


def frequency_weights(source: Any, target: SourceTarget, source_frequency: float) -> np.ndarray:
    """Per-frequency weight of the adjoint field, from the source spectrum at its own frequency."""

    simulation, frequencies = target.simulation, target.frequencies
    spectrum = source.source_time.spectrum(
        simulation.tmesh, np.full_like(frequencies, source_frequency), simulation.dt
    )
    # - 2.0: real-objective / one-sided-frequency adjoint convention.
    # - 2*pi (f -> omega): convert Hz-based quantities to angular-frequency form.
    # - 2*pi (domega = 2*pi*df): Fourier measure conversion for the current convention.
    # - c0: wavelength/frequency conversion (omega * lambda = 2*pi*c0).
    prefactor = 2.0 * (2.0 * np.pi) * (2.0 * np.pi) * C_0
    return prefactor * np.asarray(spectrum, dtype=complex) * (source_frequency / frequencies)


def source_monitor(source: Any, target: SourceTarget, fields: Iterable[str]) -> FieldMonitor:
    """The adjoint field monitor over the source's own box."""

    wanted = set(fields)
    return FieldMonitor(
        center=tuple(get_static(value) for value in source.center),
        size=tuple(get_static(value) for value in source.size),
        freqs=tuple(target.frequencies),
        fields=tuple(field for field in _FIELD_COMPONENTS if field in wanted),
        name=target.monitor,
    )


def center_vjp(
    outputs: Mapping[Key, np.ndarray], frequencies: np.ndarray, *, path: PathType, sign: float = 1.0
) -> Any:
    """``("center",)`` gets the ``(3,)`` sensitivity; ``("center", k)`` its axis ``k``."""

    center = sign * np.asarray(outputs[("center",)], dtype=float)
    return center if len(path) == 1 else float(center[path[1]])


def dataset_source_terms(
    source: Any,
    paths: Iterable[PathType],
    target: SourceTarget,
    *,
    dataset_key: str,
    interpolation: tuple[str, str, str],
    confine_to_bounds: bool,
) -> Compiled:
    """One term per dataset component, each on that component's own grid.

    A traced component returns its cotangent, summed to the dataset frequency. A traced center
    reads every coupled component, and its bindings add up across the component terms. A traced
    component that couples to no adjoint field (a field source's normal component) is zero.
    """

    dataset = getattr(source, dataset_key)
    paths = tuple(paths)
    centers = [path for path in paths if path[0] == "center"]
    requested = {path[1] for path in paths if path[0] == dataset_key}
    coupled = {c for c in dataset.field_components if source._derivative_coupling(c)[0]}
    components = [c for c in _FIELD_COMPONENTS if c in requested | (coupled if centers else set())]
    first = next(iter(dataset.field_components.values()))
    weights = frequency_weights(source, target, float(np.asarray(first.coords["f"]).ravel()[0]))
    center = np.asarray([get_static(value) for value in source.center], dtype=float)
    size = np.asarray([get_static(value) for value in source.size], dtype=float)
    terms, bindings, read = [], [], set()
    for component in components:
        data = dataset.field_components[component]
        traced = (*target.key, dataset_key, component)
        adjoint_component, sign = source._derivative_coupling(component)
        if adjoint_component is None:
            bindings.append(zero_binding(traced, data.values))
            continue
        read.add(adjoint_component)
        # the pullback crops and interpolates on increasing axes; the cotangent is put back in
        # the dataset's own coordinate order
        order = [np.argsort(data.coords[dim].values, kind="stable") for dim in "xyz"]
        values = np.asarray(get_static(data.transpose(*_ORDER).values), dtype=complex)
        values = values[np.ix_(np.arange(values.shape[0]), *order)]
        profile = SourceProfile(
            **{dim: data.coords[dim].values[index] for dim, index in zip("xyz", order)},
            values=((component, values),) if centers else (),
        )
        pullback = SourcePullback(
            source_component=component,
            adjoint_component=adjoint_component,
            coupling_sign=sign,
            interpolation=interpolation,
            return_cotangent=component in requested,
            include_in_center=bool(centers),
        )
        reductions = []
        if component in requested:
            reductions.append(SumFrequencies(outputs=(("cotangent", component),)))
            dims = tuple(_ORDER.index(dim) for dim in data.dims)
            vjp = partial(_dataset_cotangent, component=component, dims=dims, data=data)
            bindings.append(Binding.one(traced, (*target.key, component), vjp))
        if centers:
            reductions.append(SumFrequencies(outputs=(("center",),)))
            bindings.extend(
                Binding.one(
                    (*target.key, *path), (*target.key, component), partial(center_vjp, path=path)
                )
                for path in centers
            )
        kernel = SourcePullbackKernel(
            monitor=target.monitor,
            adjoint_scale=weights,
            profile=profile,
            pullbacks=(pullback,),
            center=center,
            size=size,
            confine_to_bounds=confine_to_bounds,
            compute_center=bool(centers),
            reductions=tuple(reductions),
        )
        terms.append(((*target.key, component), kernel))
    monitors = (source_monitor(source, target, read),) if terms else ()
    return Compiled(terms=tuple(terms), monitors=monitors, bindings=tuple(bindings))


def _dataset_cotangent(
    outputs: Mapping[Key, np.ndarray],
    frequencies: np.ndarray,
    *,
    component: str,
    dims: tuple[int, ...],
    data: Any,
) -> np.ndarray:
    """The summed cotangent with its dataset's length-1 frequency axis, in the dataset's order."""

    return np.transpose(unsort(outputs[("cotangent", component)], data)[None], dims)


__all__ = [
    "SourceTarget",
    "center_vjp",
    "dataset_source_terms",
    "frequency_weights",
    "source_monitor",
]
