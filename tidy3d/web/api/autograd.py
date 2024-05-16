# test autograd integration into tidy3d

import tidy3d as td
from tidy3d.components.autograd import primitive, defvjp, AutogradFieldMap
import typing

from .webapi import run as run_webapi

WVL = 1.0
FREQ0 = td.C_0 / WVL
MNT_NAME = "mnt"


""" Core autograd. """


AUX_KEY_SIM_DATA = "sim_data"
AUX_KEY_DATA_FWD = "sim_data_fwd_adjoint"
AUX_KEY_SIM_FIELD_MAPPING = "traced_sim_fields"
AUX_KEY_DATA_FIELD_MAPPING = "traced_sim_data_fields"


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
def _run(sim_fields: AutogradFieldMap, sim: td.Simulation, aux_data: dict) -> AutogradFieldMap:
    """Autograd-traced running function, runs a simulation under the hood and stitches in data."""

    td.log.info("running primitive _run()")

    num_mnts_original = len(sim.monitors)

    sim = sim.with_adjoint_monitors(sim_fields)

    # NOTE: do we really need to sanitize the sim? maybe it's fine to just run it as is
    sim_static = sim.to_static()
    sim_data = _run_static(sim_static)

    data_original, data_adjoint_fld, data_adjoint_eps = split_data_list(
        sim_data=sim_data, num_mnts_original=num_mnts_original
    )
    sim_data = sim_data.copy(update=dict(data=data_original))

    sim_fwd = sim.updated_copy(monitors=list(sim.monitors[num_mnts_original:]))

    sim_data_fwd = sim_data.updated_copy(
        data=(data_adjoint_fld + data_adjoint_eps),
        simulation=sim_fwd,
    )

    # store the forward simulation data for later (can't return it directly or autograd complains)
    aux_data[AUX_KEY_SIM_DATA] = sim_data
    aux_data[AUX_KEY_DATA_FWD] = sim_data_fwd

    # TODO: traced_fields needs to generate a mapping to the traced data objects
    # data_fields, data_field_mapping = sim_data.traced_fields()
    # aux_data[AUX_KEY_DATA_FIELD_MAPPING] = data_field_mapping

    traced_fields_data = sim_data.strip_traced_fields()

    return traced_fields_data


def run(sim: td.Simulation) -> td.SimulationData:
    """User-facing run function, passes autograd primitive _run() the array of traced fields."""
    td.log.info("running user-facing run()")

    # TODO: traced_fields needs to generate a mapping to the traced structures
    traced_fields_sim = sim.strip_traced_fields()

    if not traced_fields_sim:
        td.log.info("no tracers found, just running normally (non-autograd).")
        return _run_static(sim)

    # TODO: hide this aux data stuff?
    aux_data = {}

    traced_fields_data = _run(traced_fields_sim, sim=sim, aux_data=aux_data)

    # TODO: hide this aux data stuff?
    if AUX_KEY_SIM_DATA not in aux_data:
        raise KeyError(
            f"Could not grab 'td.SimulationData' from 'aux_data[{AUX_KEY_SIM_DATA}]'. It might not have been properly stored."
        )

    sim_data = aux_data[AUX_KEY_SIM_DATA]

    traced_fields_data = {
        path: val for path, val in traced_fields_data.items() if path[0] == "data"
    }
    sim_data_traced = sim_data.insert_traced_fields(traced_fields_data)

    return sim_data_traced


def _run_bwd(
    data_fields_fwd: AutogradFieldMap,
    sim_fields_fwd: AutogradFieldMap,
    sim: td.Simulation,
    aux_data: dict,
) -> typing.Callable[[AutogradFieldMap], AutogradFieldMap]:
    """VJP-maker for _run(). Constructs and runs adjoint simulation, does postprocessing."""

    # get the fwd epsilon and field data from the cached aux_data
    sim_data_orig = aux_data[AUX_KEY_SIM_DATA]
    sim_data_fwd = aux_data[AUX_KEY_DATA_FWD]

    td.log.info("constructing custom vjp")

    def vjp(data_fields_vjp: AutogradFieldMap) -> AutogradFieldMap:
        """dJ/d{sim.traced_fields()} as a function of Function of dJ/d{data.traced_fields()}"""

        td.log.info("running custom vjp")

        # insert the VJP data into a SimulationData
        data_fields_vjp = {path: val for path, val in data_fields_vjp.items() if path[0] == "data"}
        sim_data_vjp = sim_data_orig.insert_traced_fields(field_mapping=data_fields_vjp)

        # make adjoint simulation from that SimulationData
        sim_adj = sim_data_vjp.make_adjoint_sim(
            sim_fields_fwd=sim_fields_fwd, data_fields_vjp=data_fields_vjp
        )

        # no adjoint sources, no gradient for you :(
        if not len(sim_adj.sources):
            td.log.warning(
                "No adjoint sources generated. "
                "There is likely zero output in the data, or you have no traceable monitors. "
                "The gradient will be 0.0 for all input elements. "
                "Skipping adjoint simulation."
            )

            # make empty VJP
            sim_fields_vjp = {}
            for path, _ in data_fields_vjp.items():
                key, *sim_path = path
                if key == "simulation":
                    sim_fields_vjp[tuple(sim_path)] = 0.0

            # TODO: add a test
            return sim_fields_vjp

        # run adjoint simulation
        sim_data_adj = _run_static(sim_adj)

        # split into field and epsilon values
        _, data_adj_fld, data_adj_eps = split_data_list(sim_data=sim_data_adj, num_mnts_original=0)

        sim_fields_vjp = {}

        for path, _ in sim_fields_fwd.items():
            # grab the correct structure and field data
            _, structure_index, *sub_path = path

            fld_fwd = sim_data_fwd.get_adjoint_data(structure_index, data_type="fld")
            eps_fwd = sim_data_fwd.get_adjoint_data(structure_index, data_type="eps")
            fld_adj = sim_data_adj.get_adjoint_data(structure_index, data_type="fld")
            eps_adj = sim_data_adj.get_adjoint_data(structure_index, data_type="eps")

            # compute and store the derivative
            structure = sim_data_fwd.simulation.structures[structure_index]
            vjp_value = structure.compute_derivative(
                path=tuple(sub_path),
                fld_fwd=fld_fwd,
                eps_fwd=eps_fwd,
                fld_adj=fld_adj,
                eps_adj=eps_adj,
                eps_sim=sim_data_orig.simulation.medium.permittivity,
            )

            sim_fields_vjp[tuple(path)] = vjp_value  # TODO: actually compute this

        return sim_fields_vjp

    return vjp


defvjp(_run, _run_bwd, argnums=[0])

"""END This is code that will need to go into the regular components eventually."""
