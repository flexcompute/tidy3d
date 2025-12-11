from __future__ import annotations

import tidy3d as td
from tidy3d.components.autograd import AutogradFieldMap


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
    aux_data: dict,
) -> AutogradFieldMap:
    """Postprocess the combined simulation data into an Autograd field map."""

    num_mnts_original = len(sim_original.monitors)
    sim_data_original, sim_data_fwd = sim_data_combined._split_original_fwd(
        num_mnts_original=num_mnts_original
    )

    aux_data["sim_data"] = sim_data_original
    aux_data["sim_data_fwd_adjoint"] = sim_data_fwd

    # strip out the tracer AutogradFieldMap for the .data from the original sim
    data_traced = sim_data_original._strip_traced_fields(
        include_untraced_data_arrays=True, starting_path=("data",)
    )

    # return the AutogradFieldMap that autograd registers as the "output" of the primitive
    return data_traced
