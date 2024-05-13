# test autograd integration into tidy3d

import autograd.numpy as npa
import tidy3d as td
from tidy3d.components.autograd import primitive, defvjp
import typing

from .webapi import run as run_webapi

WVL = 1.0
FREQ0 = td.C_0 / WVL
MNT_NAME = "mnt"


""" Core autograd. """


AUX_KEY_SIM_DATA = "sim_data"
AUX_KEY_DATA_EPS = "adjoint_eps"
AUX_KEY_DATA_FLD = "adjoint_fld"
AUX_KEY_SIM_FIELD_MAPPING = "traced_sim_fields"


# TODO: pass all the kwargs
def _run_static(sim: td.Simulation) -> td.SimulationData:
    """Run a simulation without any tracers (would call web API)."""
    td.log.info("running static simulation _run_static()")
    return run_webapi(sim, task_name="autograd my life", verbose=False)


def split_list(x: list[typing.Any], index: int) -> (list[typing.Any], list[typing.Any]):
    """Split a list at a given index."""
    x = list(x)
    return x[:index], x[index:]


def split_data_list(sim_data: td.SimulationData, num_mnts_original: int) -> (list, list, list):
    """Split data list into original, adjoint field, and adjoint permittivity."""

    data_all = list(sim_data.data)
    num_mnts_adjoint = (len(data_all) - num_mnts_original) // 2

    td.log.info(
        f" -> {num_mnts_original} monitors, {num_mnts_adjoint} adjoint field monitors, {num_mnts_adjoint} adjoint eps monitors."
    )

    data_original, data_adjoint = split_list(data_all, index=num_mnts_original)
    data_adjoint_fld, data_adjoint_eps = split_list(data_adjoint, index=num_mnts_adjoint)

    return data_original, data_adjoint_fld, data_adjoint_eps


@primitive
def _run(sim_fields: npa.ndarray, sim: td.Simulation, aux_data: dict) -> tuple:
    """Autograd-traced running function, runs a simulation under the hood and stitches in data."""

    td.log.info("running primitive _run()")

    adjoint_mnts_fld, adjoint_mnts_eps = sim.generate_adjoint_monitors()
    num_mnts_original = len(sim.monitors)

    monitors = list(sim.monitors) + adjoint_mnts_fld + adjoint_mnts_eps
    sim = sim.copy(update=dict(monitors=monitors))

    # NOTE: do we really need to sanitize the sim? maybe it's fine to just run it as is
    sim_static = sim  # sim.to_static()
    sim_data = _run_static(sim_static)

    data_original, data_adjoint_fld, data_adjoint_eps = split_data_list(
        sim_data=sim_data, num_mnts_original=num_mnts_original
    )
    sim_data = sim_data.copy(update=dict(data=data_original))

    # store the forward simulation data for later (can't return it directly or autograd complains)
    aux_data[AUX_KEY_SIM_DATA] = sim_data
    aux_data[AUX_KEY_DATA_FLD] = data_adjoint_fld
    aux_data[AUX_KEY_DATA_EPS] = data_adjoint_eps

    # TODO: traced_fields needs to generate a mapping to the traced data objects
    data_fields = sim_data.traced_fields()

    return npa.array(data_fields)


def run(sim: td.Simulation) -> td.SimulationData:
    """User-facing run function, passes autograd primitive _run() the array of traced fields."""
    td.log.info("running user-facing run()")

    # TODO: traced_fields needs to generate a mapping to the traced structures
    sim_fields, sim_field_mapping = sim.traced_fields()

    # TODO: if no tracers (sim_fields empty?) probably can just run normally, although should generalize

    # TODO: hide this aux data stuff?
    aux_data = {}
    aux_data[AUX_KEY_SIM_FIELD_MAPPING] = sim_field_mapping

    data_fields = _run(sim_fields, sim=sim, aux_data=aux_data)

    # TODO: hide this aux data stuff?
    if AUX_KEY_SIM_DATA not in aux_data:
        raise KeyError(
            f"Could not grab 'td.SimulationData' from 'aux_data[{AUX_KEY_SIM_DATA}]'. It might not have been properly stored."
        )
    sim_data = aux_data[AUX_KEY_SIM_DATA]

    # TODO: with_traced_fields needs a mapping to handle arbitrary data
    sim_data_traced = sim_data.with_traced_fields(data_fields=data_fields)
    return sim_data_traced


def _run_bwd(
    data_fields_fwd: npa.ndarray, sim_fields_fwd: npa.ndarray, sim: td.Simulation, aux_data: dict
) -> typing.Callable[[npa.ndarray], npa.ndarray]:
    """VJP-maker for _run(). Constructs and runs adjoint simulation, does postprocessing."""

    # get the fwd epsilon and field data from the cached aux_data
    data_fwd_fld = aux_data[AUX_KEY_DATA_FLD]
    data_fwd_eps = aux_data[AUX_KEY_DATA_EPS]
    sim_data_fwd = aux_data[AUX_KEY_SIM_DATA]
    # sim_field_mapping = aux_data[AUX_KEY_SIM_FIELD_MAPPING]

    td.log.info("constructing custom vjp")

    def vjp(data_fields_vjp: list) -> list:
        """dJ/d{sim.traced_fields()} as a function of Function of dJ/d{data.traced_fields()}"""
        td.log.info("running custom vjp")

        # make and run adjoint simulation
        sim_adj = sim_data_fwd.make_adjoint_sim(data_fields_vjp=data_fields_vjp)

        if not len(sim_adj.sources):
            td.log.warning(
                "No adjoint sources generated. "
                "There is likely zero output in the data, or you have no traceable monitors. "
                "The gradient will be 0.0 for all input elements. "
                "Skipping adjoint simulation."
            )
            return [0.0 for _ in range(len(data_fields_vjp))]

        sim_data_adj = _run_static(sim_adj)

        # split into field and epsilon values
        _, data_adj_fld, data_adj_eps = split_data_list(sim_data=sim_data_adj, num_mnts_original=0)

        # compute the VJP output for all of the traced structures using forward and adjoint fields
        vjp_iters = (data_fwd_fld, data_fwd_eps, data_adj_fld, data_adj_eps)
        vjp_values = []

        for structure_index, (fwd_fld, fwd_eps, adj_fld, adj_eps) in enumerate(zip(*vjp_iters)):
            assert npa.all(fwd_eps == adj_eps), "different forward and adjoint permittivity."

            vjp_value_i = sim_data_fwd.compute_derivative(
                structure_index=structure_index,
                fwd_fld=fwd_fld,
                fwd_eps=fwd_eps,
                adj_fld=adj_fld,
                adj_eps=adj_eps,
            )

            vjp_values.append(vjp_value_i)

        return npa.stack(vjp_values)

    return vjp


defvjp(_run, _run_bwd, argnums=[0])

"""END This is code that will need to go into the regular components eventually."""
