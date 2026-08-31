from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import ItemsView, Mapping, Sequence

    import tidy3d as td
    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.parallel_adjoint_bases import ParallelAdjointBasis
    from tidy3d.components.autograd.spacing import SamplingResolution
    from tidy3d.flex_em.translate.sample_sets import GeometrySampleSets

    from .types import CustomVJPConfig, NumericalStructureConfig


def _require_sim_data(sim_data: td.SimulationData | None, *, field_name: str) -> td.SimulationData:
    if sim_data is None:
        raise ValueError(f"Missing required simulation data in context: {field_name}")
    return sim_data


def sampling_resolutions_for_traced_fields(
    simulation: td.Simulation,
    sim_fields_keys: Sequence[tuple],
    sample_sets: GeometrySampleSets | None = None,
) -> dict[int, SamplingResolution]:
    """Precompute one geometry sampling resolution per traced structure.

    When a pre-collected ``sample_sets`` artifact is available, the generation-time
    resolutions it carries are reused for the structures it covers, and only the
    remainder (numerical structures and custom-vjp-excluded geometry paths, which are
    deliberately absent from the artifact) pays the simulation-definition scan. This
    also guarantees generation and consumption agree on the resolution by construction.

    Structure and numerical VJP paths may contain several traced fields for the
    same structure. Deduplicating their indices here keeps the simulation-definition
    material scan outside the per-adjoint-result postprocessing loop. Plain
    structure paths only need this metadata for geometry derivatives; medium-only
    paths do not consume geometry sampling resolutions.

    Built-in medium derivatives never read the sampling helpers, so skipping medium-only
    paths is safe for them. It is deliberately visible to ``custom_vjp`` callbacks, which
    receive the same ``DerivativeInfo``: a callback on a structure with no traced geometry
    path gets ``None`` for the resolved fields and an ``AdjointError`` if it asks for them.
    See ``CustomVJPConfig.compute_derivatives``.

    Numerical paths cannot be filtered that way, because their path tails are defined by
    the ``NumericalStructureConfig`` and carry no geometry/medium distinction, so they are
    resolved unconditionally. That keeps a callback reading sampling metadata straight off
    its ``DerivativeInfo`` working, at the cost of one definition-level material scan for a
    numerical structure that turns out to want only medium derivatives.
    """

    from tidy3d.components.autograd.spacing import (
        SamplingScanIndex,
        adjoint_sampling_resolution,
    )

    structure_indices = sorted(
        {
            component_index
            for component_type, component_index, *component_path in sim_fields_keys
            if component_type == "numerical"
            or (
                component_type == "structures"
                and len(component_path) > 0
                and component_path[0] == "geometry"
            )
        }
    )
    reused = sample_sets.sampling_resolutions() if sample_sets is not None else {}
    to_scan = [index for index in structure_indices if index not in reused]
    scan_index = SamplingScanIndex.from_simulation(simulation) if to_scan else None
    return {
        structure_index: (
            reused[structure_index]
            if structure_index in reused
            else adjoint_sampling_resolution(simulation, structure_index, scan_index=scan_index)
        )
        for structure_index in structure_indices
    }


@dataclass(frozen=True)
class ParallelAdjointState:
    """Typed cache for locally precomputed parallel-adjoint runtime state."""

    task_name: str
    num_sims: int
    basis_specs: list[ParallelAdjointBasis]
    basis_maps: dict[ParallelAdjointBasis, dict[str, AutogradFieldMap]]
    basis_task_map: dict[ParallelAdjointBasis, str]

    @property
    def has_parallel_state(self) -> bool:
        """Whether the state contains populated parallel-adjoint basis information."""
        return bool(self.basis_specs) or bool(self.basis_maps) or bool(self.basis_task_map)


@dataclass
class AutogradContext:
    """Typed runtime state container for autograd execution."""

    simulation_data_original: td.SimulationData | None = None
    simulation_data_forward: td.SimulationData | None = None
    forward_task_id: str | None = None
    forward_task_from_cache: bool = False
    parallel_adjoint_state: ParallelAdjointState | None = None
    sample_sets: GeometrySampleSets | None = None


@dataclass(frozen=True)
class TaskContextBase:
    """Base per-task autograd execution context."""

    task_name: str
    context: AutogradContext
    max_num_adjoint_per_fwd: int
    numerical_structures: dict[int, NumericalStructureConfig]
    custom_vjp: tuple[CustomVJPConfig, ...] | None


def normalize_task_custom_vjp(
    task_custom_vjp: CustomVJPConfig | Sequence[CustomVJPConfig] | None,
) -> tuple[CustomVJPConfig, ...] | None:
    """Normalize per-task custom VJP config to a tuple."""
    if task_custom_vjp is None:
        return None
    if isinstance(task_custom_vjp, tuple):
        return task_custom_vjp
    if isinstance(task_custom_vjp, list):
        return tuple(task_custom_vjp)
    return (task_custom_vjp,)


@dataclass(frozen=True)
class ForwardTaskContext(TaskContextBase):
    """Per-task forward-pass context."""

    sim_fields: AutogradFieldMap
    sim_fields_keys: list[tuple]
    sim_original: td.Simulation

    @classmethod
    def from_inputs(
        cls,
        *,
        task_name: str,
        sim_fields: AutogradFieldMap,
        sim_original: td.Simulation,
        context: AutogradContext,
        max_num_adjoint_per_fwd: int,
        numerical_structures: dict[int, NumericalStructureConfig],
        custom_vjp: CustomVJPConfig | Sequence[CustomVJPConfig] | None,
    ) -> ForwardTaskContext:
        return cls(
            task_name=task_name,
            context=context,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            numerical_structures=numerical_structures,
            custom_vjp=normalize_task_custom_vjp(custom_vjp),
            sim_fields=sim_fields,
            sim_fields_keys=list(sim_fields.keys()),
            sim_original=sim_original,
        )


@dataclass(frozen=True)
class AdjointTaskContext(TaskContextBase):
    """Per-task backward-pass context."""

    sim_fields_original: AutogradFieldMap
    sim_fields_keys: list[tuple]
    sim_data_orig: td.SimulationData
    sim_data_fwd: td.SimulationData | None
    forward_task_id: str | None
    parallel_info: ParallelAdjointState | None

    @classmethod
    def from_inputs(
        cls,
        *,
        task_name: str,
        sim_fields_original: AutogradFieldMap,
        context: AutogradContext,
        max_num_adjoint_per_fwd: int,
        numerical_structures: dict[int, NumericalStructureConfig],
        custom_vjp: CustomVJPConfig | Sequence[CustomVJPConfig] | None,
        local_gradient: bool,
    ) -> AdjointTaskContext:
        parallel_info = context.parallel_adjoint_state if local_gradient else None
        sim_data_orig = _require_sim_data(
            context.simulation_data_original, field_name="simulation_data_original"
        )
        return cls(
            task_name=task_name,
            context=context,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            numerical_structures=numerical_structures,
            custom_vjp=normalize_task_custom_vjp(custom_vjp),
            sim_fields_original=sim_fields_original,
            sim_fields_keys=list(sim_fields_original.keys()),
            sim_data_orig=sim_data_orig,
            sim_data_fwd=context.simulation_data_forward,
            forward_task_id=context.forward_task_id,
            parallel_info=parallel_info,
        )


@dataclass(frozen=True, init=False)
class AdjointPostprocessInputs:
    """Typed inputs required by adjoint postprocessing."""

    sim_data_orig: td.SimulationData
    sim_data_fwd: td.SimulationData
    sim_fields_keys: list[tuple]
    numerical_structure_map: dict[int, NumericalStructureConfig]
    custom_vjp: tuple[CustomVJPConfig, ...] | None
    sampling_resolutions: Mapping[int, SamplingResolution]
    sample_sets: GeometrySampleSets | None = None

    def __init__(
        self,
        *,
        sim_data_orig: td.SimulationData,
        sim_data_fwd: td.SimulationData,
        sim_fields_keys: list[tuple],
        numerical_structure_map: dict[int, NumericalStructureConfig],
        custom_vjp: tuple[CustomVJPConfig, ...] | None,
        sampling_resolutions: Mapping[int, SamplingResolution] | None = None,
        sample_sets: GeometrySampleSets | None = None,
    ) -> None:
        """Create postprocess inputs, deriving sampling metadata when omitted."""

        if sampling_resolutions is None:
            sampling_resolutions = sampling_resolutions_for_traced_fields(
                sim_data_orig.simulation,
                sim_fields_keys,
                sample_sets=sample_sets,
            )

        object.__setattr__(self, "sim_data_orig", sim_data_orig)
        object.__setattr__(self, "sim_data_fwd", sim_data_fwd)
        object.__setattr__(self, "sim_fields_keys", sim_fields_keys)
        object.__setattr__(self, "numerical_structure_map", numerical_structure_map)
        object.__setattr__(self, "custom_vjp", custom_vjp)
        object.__setattr__(self, "sampling_resolutions", sampling_resolutions)
        object.__setattr__(self, "sample_sets", sample_sets)

    @classmethod
    def from_adjoint_task_context(
        cls, task_context: AdjointTaskContext
    ) -> AdjointPostprocessInputs:
        sim_data_fwd = _require_sim_data(task_context.sim_data_fwd, field_name="sim_data_fwd")
        return cls(
            sim_data_orig=task_context.sim_data_orig,
            sim_data_fwd=sim_data_fwd,
            sim_fields_keys=task_context.sim_fields_keys,
            numerical_structure_map=task_context.numerical_structures,
            custom_vjp=task_context.custom_vjp,
            sample_sets=task_context.context.sample_sets,
        )

    @classmethod
    def from_forward_task_context(
        cls, task_context: ForwardTaskContext
    ) -> AdjointPostprocessInputs:
        context = task_context.context
        sim_data_orig = _require_sim_data(
            context.simulation_data_original, field_name="simulation_data_original"
        )
        sim_data_fwd = _require_sim_data(
            context.simulation_data_forward, field_name="simulation_data_forward"
        )
        return cls(
            sim_data_orig=sim_data_orig,
            sim_data_fwd=sim_data_fwd,
            sim_fields_keys=task_context.sim_fields_keys,
            numerical_structure_map=task_context.numerical_structures,
            custom_vjp=task_context.custom_vjp,
            sample_sets=context.sample_sets,
        )


TaskContextT = TypeVar("TaskContextT", bound=TaskContextBase)


@dataclass(frozen=True)
class TaskBatchBase(Generic[TaskContextT]):
    """Base batch context containing a typed task map and shared run kwargs."""

    tasks: dict[str, TaskContextT]
    run_kwargs: dict[str, Any]

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(self.tasks.keys())

    def __getitem__(self, task_name: str) -> TaskContextT:
        return self.tasks[task_name]

    def items(self) -> ItemsView[str, TaskContextT]:
        return self.tasks.items()


@dataclass(frozen=True)
class ForwardTaskBatch(TaskBatchBase[ForwardTaskContext]):
    """Batch forward context keyed by task name."""

    @classmethod
    def from_single(
        cls, *, task_context: ForwardTaskContext, run_kwargs: dict[str, Any]
    ) -> ForwardTaskBatch:
        return cls(tasks={task_context.task_name: task_context}, run_kwargs=run_kwargs)

    @classmethod
    def from_inputs(
        cls,
        *,
        sim_fields_dict: Mapping[str, AutogradFieldMap],
        sims_original: Mapping[str, td.Simulation],
        contexts: Mapping[str, AutogradContext],
        max_num_adjoint_per_fwd: int,
        numerical_structures: Mapping[str, dict[int, NumericalStructureConfig]],
        custom_vjp: Mapping[str, Sequence[CustomVJPConfig]] | None,
        run_kwargs: dict[str, Any],
    ) -> ForwardTaskBatch:
        tasks = {
            task_name: ForwardTaskContext.from_inputs(
                task_name=task_name,
                sim_fields=sim_fields_dict[task_name],
                sim_original=sims_original[task_name],
                context=contexts[task_name],
                max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
                numerical_structures=numerical_structures.get(task_name, {}),
                custom_vjp=(custom_vjp or {}).get(task_name),
            )
            for task_name in sim_fields_dict.keys()
        }
        return cls(tasks=tasks, run_kwargs=run_kwargs)


@dataclass(frozen=True)
class AdjointTaskBatch(TaskBatchBase[AdjointTaskContext]):
    """Batch backward context keyed by task name."""

    @classmethod
    def from_single(
        cls, *, task_context: AdjointTaskContext, run_kwargs: dict[str, Any]
    ) -> AdjointTaskBatch:
        return cls(tasks={task_context.task_name: task_context}, run_kwargs=run_kwargs)

    @classmethod
    def from_inputs(
        cls,
        *,
        sim_fields_original_dict: Mapping[str, AutogradFieldMap],
        contexts: Mapping[str, AutogradContext],
        max_num_adjoint_per_fwd: int,
        numerical_structures: Mapping[str, dict[int, NumericalStructureConfig]],
        custom_vjp: Mapping[str, Sequence[CustomVJPConfig]] | None,
        run_kwargs: dict[str, Any],
        local_gradient: bool,
    ) -> AdjointTaskBatch:
        tasks = {
            task_name: AdjointTaskContext.from_inputs(
                task_name=task_name,
                sim_fields_original=sim_fields_original_dict[task_name],
                context=contexts[task_name],
                max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
                numerical_structures=numerical_structures.get(task_name, {}),
                custom_vjp=(custom_vjp or {}).get(task_name),
                local_gradient=local_gradient,
            )
            for task_name in sim_fields_original_dict.keys()
        }
        return cls(tasks=tasks, run_kwargs=run_kwargs)


@dataclass(frozen=True)
class PreparedAdjointBatch:
    """Prepared async adjoint work bundle."""

    sims_adj: dict[str, td.Simulation]
    task_name_mapping: dict[str, str]
    sim_fields_vjp_dict: dict[str, AutogradFieldMap]
    task_has_adj_sources: dict[str, bool]

    @property
    def any_adj_sources(self) -> bool:
        return any(self.task_has_adj_sources.values())
