from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tidy3d.components.autograd.collection import collect_adjoint_sample_sets

from .utils import custom_vjp_geometry_exclusions, expand_custom_vjp_configs

if TYPE_CHECKING:
    from collections.abc import Sequence

    import tidy3d as td
    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.em.translate.sample_sets import GeometrySampleSets

    from .context import AutogradContext, ForwardTaskContext
    from .types import CustomVJPConfig


@dataclass(frozen=True)
class PreparedForward:
    """Result of preparing one autograd forward task.

    Pairs the monitor-staged combined simulation with the exact sample-set artifact
    those monitors were staged from, so monitor staging, the consumption context,
    the remote sidecar upload, and adjoint construction all share one instance.
    """

    sim_combined: td.Simulation
    sample_sets: GeometrySampleSets


def prepare_forward(task_context: ForwardTaskContext) -> PreparedForward:
    """Collect the sample-set artifact once and stage the forward adjoint monitors.

    The single preparation point of both gradient strategies: the artifact is
    collected exactly once per task, stored on the task context (backward
    consumption and parallel-adjoint construction read it there), and returned
    alongside the monitor-staged simulation. Unsupported media are validated
    before any sampling work, so a traced CustomMedium geometry fails here
    rather than after geometry sampling and payload staging.
    """
    task_context.sim_original._check_custom_medium_geometry_overlap(task_context.sim_fields)
    sample_sets = collect_forward_sample_sets(
        task_context.sim_original,
        task_context.sim_fields_keys,
        task_context.custom_vjp,
    )
    task_context.context.sample_sets = sample_sets
    sim_combined = setup_fwd(
        sim_fields=task_context.sim_fields,
        sim_original=task_context.sim_original,
        local_gradient=True,
        custom_vjp=task_context.custom_vjp,
        sample_sets=sample_sets,
    )
    return PreparedForward(sim_combined=sim_combined, sample_sets=sample_sets)


def collect_forward_sample_sets(
    sim_original: td.Simulation,
    sim_fields_keys: list[tuple],
    custom_vjp: Sequence[CustomVJPConfig] | None = None,
) -> GeometrySampleSets:
    """Collect the adjoint sample-set artifact with custom-vjp exclusions applied.

    The single collection point of the client flow: the returned instance is
    threaded to forward monitor staging, parallel-adjoint monitor construction,
    and backward consumption, so monitor point ordering and consumption indexing
    cannot diverge.
    """
    exclusions = custom_vjp_geometry_exclusions(
        expand_custom_vjp_configs(custom_vjp, sim_fields_keys)
    )
    return collect_adjoint_sample_sets(sim_original, sim_fields_keys, exclude=exclusions)


def setup_fwd(
    sim_fields: AutogradFieldMap,
    sim_original: td.Simulation,
    local_gradient: bool = False,
    custom_vjp: Sequence[CustomVJPConfig] | None = None,
    sample_sets: GeometrySampleSets | None = None,
) -> td.Simulation:
    """Return a forward simulation with adjoint monitors attached.

    ``sample_sets`` is the pre-collected artifact from
    :func:`collect_forward_sample_sets`; when omitted, it is collected here with
    the same custom-vjp exclusions. Shape-derivative monitors are staged from it.
    """

    # Ensure there aren't any traced geometries with custom media
    sim_original._check_custom_medium_geometry_overlap(sim_fields)

    if sample_sets is None:
        sample_sets = collect_forward_sample_sets(sim_original, list(sim_fields.keys()), custom_vjp)

    # Always try to build the variant that includes adjoint monitors so that
    # errors in monitor placement are caught early.
    sim_with_adj_mon = sim_original._with_adjoint_monitors(sim_fields, sample_sets=sample_sets)
    return sim_with_adj_mon if local_gradient else sim_original


def postprocess_fwd(
    sim_data_combined: td.SimulationData,
    sim_original: td.Simulation,
    context: AutogradContext,
    sim_fields_keys: list[tuple],
    custom_vjp: Sequence[CustomVJPConfig] | None = None,
) -> AutogradFieldMap:
    """Postprocess the combined simulation data into an Autograd field map.

    ``sim_fields_keys`` is required: this is the local-gradient flow's only entry
    point, and the backward pass needs the sample sets collected here — skipping
    collection would surface later as a missing-artifact error on geometry paths.
    """
    num_mnts_original = len(sim_original.monitors)
    sim_data_original, sim_data_fwd = sim_data_combined._split_original_fwd(
        num_mnts_original=num_mnts_original
    )

    context.simulation_data_original = sim_data_original
    context.simulation_data_forward = sim_data_fwd

    # the adjoint surface sample sets consumed by backward postprocessing: reuse the
    # instance collected at forward setup when the strategy threaded it onto the
    # context; collect here otherwise (paths owned by custom vjps are excluded —
    # they never consume sample sets)
    if context.sample_sets is None:
        context.sample_sets = collect_forward_sample_sets(sim_original, sim_fields_keys, custom_vjp)

    # strip out the tracer AutogradFieldMap for the .data from the original sim
    data_traced = sim_data_original._strip_traced_fields(
        include_untraced_data_arrays=True, starting_paths=(("data",),)
    )

    # return the AutogradFieldMap that autograd registers as the "output" of the primitive
    return data_traced
