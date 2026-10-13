from __future__ import annotations

from typing import TYPE_CHECKING

from flexcompute.core._migration.em.derivatives import split_derivative_data

from tidy3d.log import log

if TYPE_CHECKING:
    import tidy3d as td
    from tidy3d.components.autograd import AutogradFieldMap

    from .context import AutogradContext, ForwardTaskContext


def prepare_compiled_forward(task_context: ForwardTaskContext) -> td.Simulation:
    """Compile the task's derivative run and stage its forward monitors.

    The compiled monitors record what the program reads; paths owned by a user callback keep
    the volumetric monitors their ``DerivativeInfo`` is built from.
    """

    from .derivatives import compile_run

    sim_original = task_context.sim_original
    sim_original._check_custom_medium_geometry_overlap(task_context.sim_fields)
    run = compile_run(
        sim_original,
        task_context.sim_fields_keys,
        task_context.custom_vjp,
        task_context.numerical_structures,
    )
    task_context.context.derivative_run = run
    return run.stage_forward(sim_original)


def postprocess_fwd(
    sim_data_combined: td.SimulationData,
    sim_original: td.Simulation,
    context: AutogradContext,
) -> AutogradFieldMap:
    """Postprocess the combined simulation data into an Autograd field map."""
    sim_data_original, sim_data_fwd = split_derivative_data(
        sim_data_combined, [monitor.name for monitor in sim_original.monitors], log
    )

    context.simulation_data_original = sim_data_original
    context.simulation_data_forward = sim_data_fwd

    # strip out the tracer AutogradFieldMap for the .data from the original sim
    data_traced = sim_data_original._strip_traced_fields(
        include_untraced_data_arrays=True, starting_paths=(("data",),)
    )

    # return the AutogradFieldMap that autograd registers as the "output" of the primitive
    return data_traced
