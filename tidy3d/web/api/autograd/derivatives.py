"""The compiled derivative run of one autograd task: its program, bindings, and callback paths.

Compiler-owned paths become a ``DerivativeProgram`` evaluated by the shared executor; paths owned
by a user callback (custom VJPs and numerical-structure parameters) keep the legacy volumetric
``DerivativeInfo`` route, fed per frequency batch with the results of any declared queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

from flexcompute.core._migration.em.derivatives import (
    PROGRAM_DIGEST_ATTR,
    ExecutionSettings,
    execute_batches,
)
from flexcompute.core._migration.em.derivatives.execute import merge_batch

import tidy3d as td
import tidy3d.system as system_utils
from tidy3d.config import config
from tidy3d.em.translate.derivative_terms import bind, compile_derivatives

from .utils import expand_custom_vjp_configs

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import numpy as np
    from flexcompute.core._migration.em.derivatives import DerivativeProgram, Results

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.derivative_terms import Binding

    from .types import CustomVJPConfig, NumericalStructureConfig


@dataclass(frozen=True)
class DerivativeRun:
    """The compiled side of one task's derivatives, and the paths left to user callbacks."""

    program: DerivativeProgram
    monitors: tuple[Any, ...]
    """Every monitor the program reads, staged in the forward simulation."""
    bindings: tuple[Binding, ...]
    """The chain rule from the program's outputs back to each compiled traced path."""
    callback_keys: tuple[tuple, ...]

    @property
    def adjoint_monitors(self) -> tuple[Any, ...]:
        """Permittivity is the same in the adjoint run, so its point clouds stay forward-only."""

        return tuple(
            m for m in self.monitors if not isinstance(m, td.PointCloudPermittivityMonitor)
        )

    @cached_property
    def digest(self) -> str:
        return self.program.digest()

    def stage_forward(self, simulation: td.Simulation) -> td.Simulation:
        """The forward simulation: callback monitors (and FluxMonitor helpers), then the program's."""

        staged = simulation._with_adjoint_monitors(list(self.callback_keys))
        return staged.updated_copy(monitors=(*staged.monitors, *self.monitors))

    def for_upload(self, simulation: td.Simulation, simulation_type: str) -> td.Simulation:
        """A staged simulation as a remote gradient uploads it, carrying the program digest."""

        attrs = {**simulation.attrs, PROGRAM_DIGEST_ATTR: self.digest}
        return simulation.updated_copy(simulation_type=simulation_type, attrs=attrs, deep=False)

    def vjps(self, results: Results) -> AutogradFieldMap:
        """Bind one adjoint simulation's results to every compiled path but the queries."""

        regular = [b for b in self.bindings if not _is_query(b)]
        return bind(results, regular, self.program.frequencies_hz)


def _is_query(binding: Binding) -> bool:
    """A binding's paths share one owner, so its first path says whether it binds a query."""

    return binding.paths[0][0] == "numerical"


def callback_keys(
    sim_fields_keys: Sequence[tuple], custom_vjp: Sequence[CustomVJPConfig] | None
) -> tuple[tuple, ...]:
    """Traced paths a user callback owns: numerical-structure parameters and custom-VJP paths."""

    lookup = expand_custom_vjp_configs(custom_vjp, list(sim_fields_keys))
    return tuple(
        key
        for key in map(tuple, sim_fields_keys)
        if key[0] == "numerical"
        or (key[0] == "structures" and tuple(key[2:4]) in lookup.get(key[1], {}))
    )


def compile_run(
    simulation: td.Simulation,
    sim_fields_keys: Sequence[tuple],
    custom_vjp: Sequence[CustomVJPConfig] | None,
    numerical_structures: Mapping[int, NumericalStructureConfig],
) -> DerivativeRun:
    """Compile every compiler-owned path and every traced numerical structure's queries."""

    owned = callback_keys(sim_fields_keys, custom_vjp)
    traced_numerical = {key[1] for key in owned if key[0] == "numerical"}
    queries = {
        index: config.derivative_queries
        for index, config in numerical_structures.items()
        if index in traced_numerical and config.derivative_queries
    }
    paths = [key for key in map(tuple, sim_fields_keys) if key not in owned]
    program, compiled = compile_derivatives(simulation, paths, queries)
    return DerivativeRun(
        program=program,
        monitors=compiled.monitors,
        bindings=compiled.bindings,
        callback_keys=owned,
    )


def execution_settings() -> ExecutionSettings:
    """The client's executor settings: gradient precision, memory budget, and logging."""

    return ExecutionSettings(
        complex_dtype=config.adjoint.gradient_dtype_complex,
        memory_allotment_fraction=config.adjoint.memory_allotment_fraction,
        solver_freq_chunk_size=config.adjoint.solver_freq_chunk_size,
        available_memory_bytes=system_utils.get_available_memory_bytes,
        warn=lambda message: td.log.warning(message, log_once=True),
    )


@dataclass(frozen=True)
class RunVjps:
    """The program's VJPs for one adjoint simulation, and its batches for numerical callbacks."""

    vjps: AutogradFieldMap
    batch_size: int | None
    query_results: dict[int, list[tuple[np.ndarray, dict[str, dict[tuple, Any]]]]]
    """Per numerical structure, one ``(batch frequencies, {query key: {structure-local path:
    value}})`` per batch, in batch order."""


def run_vjps(
    run: DerivativeRun,
    sim_data_fwd: td.SimulationData,
    sim_data_adj: td.SimulationData,
    extra_data_bytes: int = 0,
) -> RunVjps:
    """Execute the program once over its frequency batches and bind its results.

    Only terms the regular bindings read are merged; query bindings bind each batch on its own,
    so a numerical callback receives the results for exactly the frequencies of its chunk.
    ``extra_data_bytes`` is callback data processed in those same batches, so the plan budgets it.
    """

    program = run.program
    queries = [b for b in run.bindings if _is_query(b)]
    regular_terms = {b.term for b in run.bindings if not _is_query(b)}
    results: Results = {}
    query_results: dict[int, list[tuple[np.ndarray, dict[str, dict[tuple, Any]]]]] = {}
    batch_size = None
    for batch, outputs in execute_batches(
        program, sim_data_fwd, sim_data_adj, execution_settings(), extra_data_bytes
    ):
        regular = {term: o for term, o in outputs.items() if term in regular_terms}
        merge_batch(results, batch, regular, program.frequencies_hz.size)
        batch_size = batch_size or batch.size
        if not queries:
            continue
        values = {term: {key: out.values for key, out in o.items()} for term, o in outputs.items()}
        frequencies = program.frequencies_hz[batch]
        for index, chunk in bind_queries(values, queries, frequencies).items():
            query_results.setdefault(index, []).append((frequencies, chunk))
    return RunVjps(vjps=run.vjps(results), batch_size=batch_size, query_results=query_results)


def bind_queries(
    results: Results, bindings: Iterable[Binding], frequencies: np.ndarray
) -> dict[int, dict[str, dict[tuple, Any]]]:
    """Query bindings bound over ``frequencies``: per numerical structure, per query key, by
    structure-local path, as a numerical callback receives them."""

    grouped: dict[int, dict[str, dict[tuple, Any]]] = {}
    for (_, structure, key, *path), value in bind(results, bindings, frequencies).items():
        grouped.setdefault(structure, {}).setdefault(key, {})[tuple(path)] = value
    return grouped


__all__ = [
    "DerivativeRun",
    "RunVjps",
    "bind_queries",
    "callback_keys",
    "compile_run",
    "execution_settings",
    "run_vjps",
]
