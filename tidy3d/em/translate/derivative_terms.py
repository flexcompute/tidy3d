"""Compile traced simulation paths into a derivative program, its monitors, and client bindings.

Components build their own terms and bindings (``_derivative_terms``). This module groups the
traced paths by owner, checks the owner, hands it a ``MediumTarget``, and stages the monitors
its terms read.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from flexcompute.core._migration.em.derivatives import DerivativeProgram

from tidy3d.components.autograd.adjoint_settings import adjoint_settings_from_config
from tidy3d.components.autograd.collection import collect_adjoint_sample_sets
from tidy3d.components.autograd.derivative_terms import combine
from tidy3d.components.autograd.utils import get_static
from tidy3d.components.bound_ops import bounds_contains, bounds_intersection
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.medium.custom.isotropic import CustomMedium
from tidy3d.components.medium.derivative_terms import MediumTarget
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.source.derivative_terms import SourceTarget
from tidy3d.config import config
from tidy3d.em.translate.shape_terms import shape_terms
from tidy3d.exceptions import AdjointError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from flexcompute.core._migration.em.derivatives import Results

    from tidy3d.components.autograd.derivative_terms import Binding, Compiled, PathType
    from tidy3d.components.autograd.types import GeometryAdjointSettings

_COMPONENTS = ("xx", "yy", "zz")


def compile_derivatives(
    simulation: Any,
    paths: Iterable[PathType],
    numerical_queries: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[DerivativeProgram, Compiled]:
    """The derivative program and its client side for every traced path of one simulation.

    The paths must already be validated (``_resolve_autograd_route``), as run setup does.
    Geometry paths, regular or queried, sample their surfaces here under one snapshot of
    the adjoint settings, which their bindings carry to the chain rule.
    ``numerical_queries`` holds each inserted numerical structure's declared queries by index;
    their results bind under ``("numerical", index, query_key, *structure_local_path)``.
    """

    frequencies = np.asarray(simulation._freqs_adjoint, dtype=float)
    settings = adjoint_settings_from_config()
    owners: dict[tuple, list[PathType]] = defaultdict(list)
    for path in map(tuple, paths):
        owner = path[:2] if path[:1] == ("sources",) else path[:3]
        if owner[:1] != ("sources",) and owner[::2] not in (
            ("structures", "medium"),
            ("structures", "geometry"),
        ):
            raise AdjointError(f"No derivative compiler handles the traced path {path}.")
        owners[owner].append(path[len(owner) :])
    compiled = combine(
        [_owner(simulation, owner, local, frequencies, settings) for owner, local in owners.items()]
        + [
            _query(simulation, index, position, key, query, frequencies, settings)
            for index, queries in (numerical_queries or {}).items()
            for position, (key, query) in enumerate(queries.items())
        ]
    )
    taken = {m.name for m in simulation.monitors} & {m.name for m in compiled.monitors}
    if taken:
        raise AdjointError(
            f"Simulation monitors already use derivative monitor names {sorted(taken)}."
        )
    return DerivativeProgram(frequencies_hz=frequencies, terms=compiled.terms), compiled


def bind(
    results: Results, bindings: Iterable[Binding], frequencies: np.ndarray
) -> dict[PathType, Any]:
    """Each traced path's VJP from results over ``frequencies``; bindings of one path add up."""

    vjps: dict[PathType, Any] = {}
    for binding in bindings:
        outputs = {}
        for term in binding.terms:
            if outputs.keys() & results[term].keys():
                raise AdjointError(f"Terms {binding.terms} bound together share output keys.")
            outputs.update(results[term])
        for path, value in binding.reduce(outputs, frequencies).items():
            vjps[path] = vjps[path] + value if path in vjps else value
    return vjps


def _owner(
    simulation: Any,
    owner: tuple,
    local: list[PathType],
    frequencies: np.ndarray,
    settings: GeometryAdjointSettings,
) -> Compiled:
    """The compiled terms of one owner: a source, a structure's medium, or its geometry."""

    if owner[0] == "sources":
        return _source(simulation, owner[1], local, frequencies)
    if owner[2] == "medium":
        return _medium(simulation, owner[1], local, frequencies)
    structure_sets = _structure_sets(simulation, owner[1], local, settings)
    return shape_terms(simulation, owner[1], local, structure_sets, frequencies, settings=settings)


def _structure_sets(
    simulation: Any, index: int, paths: list[PathType], settings: GeometryAdjointSettings
) -> Any:
    """The staged surface sample sets for one structure's traced geometry paths."""

    traced = [("structures", index, "geometry", *path) for path in paths]
    return collect_adjoint_sample_sets(simulation, traced, settings).for_structure(index)


def _query(
    simulation: Any,
    index: int,
    position: int,
    query_key: str,
    query: Any,
    frequencies: np.ndarray,
    settings: GeometryAdjointSettings,
) -> Compiled:
    """One declared numerical query, compiled by the same owners on its target structure.

    Its monitors are named by the query's position, so free-form keys never reach a name.
    """

    target_simulation, paths = _query_target(simulation, index, query_key, query)
    prefix, token = ("numerical", index, query_key), f"{index}_q{position}"
    parts = []
    if "medium" in paths:
        parts.append(
            _medium(
                target_simulation,
                index,
                paths["medium"],
                frequencies,
                key=(*prefix, "medium"),
                monitor=f"adjoint_medium_fld_{token}",
                resolved=query.frequency_mode == "resolved",
            )
        )
    if "geometry" in paths:
        structure_sets = _structure_sets(target_simulation, index, paths["geometry"], settings)
        parts.append(
            shape_terms(
                target_simulation,
                index,
                paths["geometry"],
                structure_sets,
                frequencies,
                settings=settings,
                key=(*prefix, "geometry"),
                token=token,
            )
        )
    return combine(parts)


def _query_target(
    simulation: Any, index: int, query_key: str, query: Any
) -> tuple[Any, dict[str, list[PathType]]]:
    """The simulation with the query's view substituted, and its paths by domain.

    The view only changes what is differentiated, so the substituted simulation keeps the grid
    the fields are recorded on; an automatic grid would otherwise be rebuilt from the view.
    """

    structure = simulation.structures[index]
    try:
        view = query.view_for(simulation, index)
    except AdjointError:
        raise
    except Exception as error:
        raise AdjointError(
            f"Numerical derivative query {query_key!r} view setup failed: {error}"
        ) from error
    geometry = structure.geometry if view is None or view.geometry is None else view.geometry
    medium = structure.medium if view is None or view.medium is None else view.medium
    if geometry is not structure.geometry and not bounds_contains(
        structure.geometry.bounds, geometry.bounds
    ):
        raise AdjointError(
            f"Numerical derivative query {query_key!r} geometry bounds must be contained "
            "within the inserted numerical structure bounds."
        )
    if medium.is_pec and not structure.medium.is_pec:
        raise AdjointError(
            f"Numerical derivative query {query_key!r} cannot replace a non-PEC medium with "
            "PEC because the required magnetic fields were not requested."
        )
    target = structure.updated_copy(geometry=geometry, medium=medium)
    structures = list(simulation.structures)
    structures[index] = target
    target_simulation = simulation.updated_copy(
        structures=tuple(structures), grid_spec=GridSpec.from_grid(simulation.grid), validate=False
    )
    paths: dict[str, list[PathType]] = {}
    for path in query.paths:
        try:
            target._resolve_autograd_route(tuple(path))
        except (AdjointError, IndexError, TypeError) as error:
            detail = str(error) or "the indexed parameter does not exist"
            raise AdjointError(
                f"Numerical derivative query {query_key!r} cannot resolve path {path!r}: {detail}"
            ) from error
        paths.setdefault(path[0], []).append(tuple(path[1:]))
    if query.frequency_mode == "resolved" and (
        set(paths) != {"medium"} or type(medium) is not CustomMedium
    ):
        raise AdjointError(
            f"Numerical derivative query {query_key!r} requests frequency-resolved output, "
            "which currently requires direct CustomMedium paths."
        )
    return target_simulation, paths


def _medium(
    simulation: Any,
    index: int,
    local_paths: list[PathType],
    frequencies: np.ndarray,
    *,
    key: tuple | None = None,
    monitor: str | None = None,
    resolved: bool = False,
) -> Compiled:
    """One structure's medium paths, compiled by the medium over the structure's bounding box."""

    structure = simulation.structures[index]
    medium = structure.medium
    if medium.is_time_modulated or medium.is_nonlinear:
        raise AdjointError("Medium derivatives do not support time-modulated or nonlinear media.")
    bounds = np.asarray(bounds_intersection(simulation.bounds, structure.geometry.bounds))
    active = np.asarray(simulation.size) > 0
    if np.any(bounds[0] > bounds[1]) or np.any(bounds[0][active] >= bounds[1][active]):
        raise AdjointError("Medium derivatives need the structure to overlap the simulation.")
    target = MediumTarget(
        key=("structures", index, "medium") if key is None else key,
        components=_COMPONENTS,
        monitor=f"adjoint_medium_fld_{index}" if monitor is None else monitor,
        bounds=bounds,
        frequencies=frequencies,
        resolved=resolved,
    )
    compiled = medium._derivative_terms(local_paths, target)
    if not compiled.terms:
        return compiled
    read = {c for _, kernel in compiled.terms for c in kernel.components}
    return replace(compiled, monitors=(_volume_monitor(simulation, index, target, read),))


def _source(
    simulation: Any, index: int, local_paths: list[PathType], frequencies: np.ndarray
) -> Compiled:
    """One source's paths, compiled by the source over its own adjoint field monitor."""

    target = SourceTarget(
        key=("sources", index),
        monitor=f"adjoint_source_fld_{index}",
        frequencies=frequencies,
        simulation=simulation,
    )
    return simulation.sources[index]._derivative_terms(local_paths, target)


def _volume_monitor(
    simulation: Any, index: int, target: MediumTarget, components: set[str]
) -> FieldMonitor:
    """The E-field monitor over one structure, in the box the legacy medium monitor uses."""

    plane = simulation if simulation.size.count(0.0) == 1 else None
    box = simulation.structures[index]._adjoint_monitor_box(grid=simulation.grid, plane=plane)
    return FieldMonitor(
        center=tuple(get_static(x) for x in box.center),
        size=tuple(get_static(x) for x in box.size),
        freqs=tuple(target.frequencies),
        fields=tuple(f"E{c[0]}" for c in _COMPONENTS if c in components),
        name=target.monitor,
        interval_space=config.adjoint.monitor_interval_custom,
        colocate=False,
    )


__all__ = ["bind", "compile_derivatives"]
