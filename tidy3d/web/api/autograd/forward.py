from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d.components.autograd.collection import collect_adjoint_sample_sets

from .utils import custom_vjp_geometry_exclusions, expand_custom_vjp_configs

if TYPE_CHECKING:
    from collections.abc import Sequence

    import tidy3d as td
    from tidy3d.components.autograd import AutogradFieldMap

    from .context import AutogradContext
    from .types import CustomVJPConfig


def setup_fwd(
    sim_fields: AutogradFieldMap,
    sim_original: td.Simulation,
    local_gradient: bool = False,
) -> td.Simulation:
    """Return a forward simulation with adjoint monitors attached."""

    # Ensure there aren't any traced geometries with custom media
    sim_original._check_custom_medium_geometry_overlap(sim_fields)

    # Always try to build the variant that includes adjoint monitors so that
    # errors in monitor placement are caught early.
    sim_with_adj_mon = sim_original._with_adjoint_monitors(sim_fields)
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

    # pre-collect the adjoint surface sample sets consumed by backward postprocessing;
    # paths owned by custom vjps are excluded (they never consume sample sets)
    exclusions = custom_vjp_geometry_exclusions(
        expand_custom_vjp_configs(custom_vjp, sim_fields_keys)
    )
    context.sample_sets = collect_adjoint_sample_sets(
        sim_original, sim_fields_keys, exclude=exclusions
    )

    # strip out the tracer AutogradFieldMap for the .data from the original sim
    data_traced = sim_data_original._strip_traced_fields(
        include_untraced_data_arrays=True, starting_paths=(("data",),)
    )

    # return the AutogradFieldMap that autograd registers as the "output" of the primitive
    return data_traced
