# autograd wrapper for web functions

import tidy3d as td
from tidy3d.components.autograd import primitive, defvjp, AutogradFieldMap, get_static  # noqa: 401
import typing

import numpy as np

from ..webapi import run as run_webapi
from ..container import Batch, BatchData

from .utils import split_list, split_data_list, get_derivative_maps


# keys for data into auxiliary dictionary
AUX_KEY_SIM_DATA_ORIGINAL = "sim_data"
AUX_KEY_SIM_DATA_FWD = "sim_data_fwd_adjoint"

""" User-facing ``run`` and `run_async`` functions, compatible with ``autograd`` """

def run(simulation: td.Simulation, task_name: str, **run_kwargs) -> td.SimulationData:
    """User-facing ``web.run`` function, compatible with ``autograd`` differentiation."""

    td.log.info("running user-facing run()")

    traced_fields_sim = setup_fwd_run(simulation=simulation)

    # if we register this as not needing adjoint at all (no tracers), call regular run function
    if not traced_fields_sim:
        td.log.info(
            "No autograd derivative tracers found in the 'Simulation' passed to 'run'. "
            "This could indicate that there is no path from your objective function arguments "
            "to the 'Simulation'. If this is unexpected, double check your objective function "
            "pre-processing. Running regular tidy3d simulation."
        )
        return _run_tidy3d(simulation, task_name=task_name, **run_kwargs)

    td.log.info("Found derivative tracers in the 'Simulation', running adjoint forward pass.")

    # will store the SimulationData for original and forward so we can access them later
    aux_data = {}

    # run our custom @primitive, passing the traced fields first to register with autograd
    traced_fields_data = _run_primitive(
        traced_fields_sim,
        simulation=simulation,
        task_name=task_name,
        aux_data=aux_data,
        **run_kwargs,
    )

    return postprocess_fwd_run(traced_fields_data=traced_fields_data, aux_data=aux_data)

def run_async(simulations: dict[str, td.Simulation], **run_kwargs) -> BatchData:
    raise NotImplementedError("I haven't implemented batching yet")

def setup_fwd_run(simulation: td.Simulation) -> AutogradFieldMap:
    """Process a user-supplied ``Simulation`` into inputs to ``_run_primitive``."""

    # get a mapping of all the traced fields in the provided simulation
    return simulation.strip_traced_fields()


def postprocess_fwd_run(traced_fields_data: AutogradFieldMap, aux_data: dict) -> td.SimulationData:
    """Process the return from ``_run_primitive`` into ``SimulationData`` for user."""

    traced_fields_data = {key: val for key, val in traced_fields_data.items() if key[0] == "data"}

    # grab the user's 'SimulationData' and return with the autograd-tracers inserted
    sim_data_original = aux_data[AUX_KEY_SIM_DATA_ORIGINAL]
    return sim_data_original.insert_traced_fields(traced_fields_data)


""" Autograd-traced Primitive for FWD pass ``run`` functions """

@primitive
def _run_primitive(
    sim_fields: AutogradFieldMap,
    simulation: td.Simulation,
    task_name: str,
    aux_data: dict,
    **run_kwargs,
) -> AutogradFieldMap:
    """Autograd-traced 'run()' function: runs simulation, strips tracer data, caches fwd data."""

    td.log.info("running primitive '_run_primitive()'")
    sim_original = simulation.to_static()
    sim_combined = setup_fwd(sim_fields=sim_fields, sim_original=sim_original)
    sim_data_combined = _run_tidy3d(sim_combined, task_name=task_name, **run_kwargs)
    return postprocess_fwd(sim_data_combined=sim_data_combined, sim_original=sim_original, aux_data=aux_data)

def setup_fwd(sim_fields: AutogradFieldMap, sim_original: td.Simulation) -> td.Simulation:
    """Set up the combined forward simulation."""

    # make and run a sim with combined original & adjoint monitors
    return sim_original.with_adjoint_monitors(sim_fields)

def postprocess_fwd(sim_data_combined: td.SimulationData, sim_original: td.Simulation, aux_data: dict) -> AutogradFieldMap:
    """Postprocess the combined simulation data into an Autograd field map."""

    sim_combined = sim_data_combined.simulation

    num_mnts_original = len(sim_original.monitors)

    # split the data and monitors into the original ones & adjoint gradient ones (for 'fwd')
    data_original, data_fwd = split_data_list(
        sim_data=sim_data_combined, num_mnts_original=num_mnts_original
    )
    _, monitors_fwd = split_list(sim_combined.monitors, index=num_mnts_original)

    # reconstruct the simulation data for the user, using original sim, and data for original mnts
    sim_data_original = sim_data_combined.updated_copy(
        simulation=sim_original, data=data_original, deep=False
    )

    # construct the 'forward' simulation and its data, which is only used for for gradient calc.
    sim_fwd = sim_combined.updated_copy(monitors=monitors_fwd)
    sim_data_fwd = sim_data_combined.updated_copy(
        simulation=sim_fwd,
        data=data_fwd,
        deep=False,
    )

    # cache these two SimulationData objects for later (note: the Simulations are already inside)
    aux_data[AUX_KEY_SIM_DATA_ORIGINAL] = sim_data_original
    aux_data[AUX_KEY_SIM_DATA_FWD] = sim_data_fwd

    # strip out the tracer AutogradFieldMap for the .data from the original sim
    data_traced = sim_data_original.strip_traced_fields()

    # need to get the static version of the arrays, otherwise get ArrayBox of ArrayBox
    # NOTE: this is a bit confusing to me, why does autograd make them ArrayBox out of _run_tidy3d?

    data_traced = {path: value for path, value in data_traced.items() if path[0] == "data"}

    # return the AutogradFieldMap that autograd registers as the "output" of the primitive
    return data_traced


""" VJP maker for ADJ pass."""

def _run_bwd(
    data_fields_original: AutogradFieldMap,
    sim_fields_original: AutogradFieldMap,
    simulation: td.Simulation,
    task_name: str,
    aux_data: dict,
    **run_kwargs,
) -> typing.Callable[[AutogradFieldMap], AutogradFieldMap]:
    """VJP-maker for ``_run_primitive()``. Constructs and runs adjoint simulation, computes grad."""

    # get the fwd epsilon and field data from the cached aux_data
    sim_data_orig = aux_data[AUX_KEY_SIM_DATA_ORIGINAL]
    sim_data_fwd = aux_data[AUX_KEY_SIM_DATA_FWD]

    td.log.info("constructing custom vjp function for backwards pass.")

    def vjp(data_fields_vjp: AutogradFieldMap) -> AutogradFieldMap:
        """dJ/d{sim.traced_fields()} as a function of Function of dJ/d{data.traced_fields()}"""


        sim_adj = setup_adj(
            data_fields_vjp=data_fields_vjp,
            sim_data_orig=sim_data_orig,
            sim_data_fwd=sim_data_fwd,
            sim_fields_original=sim_fields_original
        )

        # no adjoint sources, no gradient for you :(
        if not len(sim_adj.sources):
            td.log.warning(
                "No adjoint sources generated. "
                "There is likely zero output in the data, or you have no traceable monitors. "
                "As a result, the 'SimulationData' returned has no contribution to the gradient. "
                "Skipping the adjoint simulation. "
                "If this is unexpected, please double check the post-processing function to ensure "
                "there is a path from the 'SimulationData' to the objective function return value."
            )

            # TODO: add a test for this
            # construct a VJP of all zeros for all tracers in the original simulation
            return {path: 0 * value for path, value in sim_fields_original.items()}

        # run adjoint simulation
        task_name_adj = task_name + "_adjoint"
        sim_data_adj = _run_tidy3d(sim_adj, task_name=task_name_adj, **run_kwargs)

        return postprocess_adj(
            sim_data_adj=sim_data_adj,
            sim_data_orig=sim_data_orig,
            sim_data_fwd=sim_data_fwd,
            sim_fields_original=sim_fields_original
        )

    return vjp


def setup_adj(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    sim_fields_original: AutogradFieldMap
) -> td.Simulation:
    """Construct an adjoint simulation from a set of data_fields for the VJP."""

    td.log.info("Running custom vjp (adjoint) pipeline.")

    # immediately filter out any data_vjps with all 0's in the data
    data_fields_vjp = {
        key: get_static(value)
        for key, value in data_fields_vjp.items()
        if not np.all(value == 0.0)
    }

    # insert the raw VJP data into the .data of the original SimulationData
    sim_data_vjp = sim_data_orig.insert_traced_fields(field_mapping=data_fields_vjp)

    # make adjoint simulation from that SimulationData
    data_vjp_paths = set(data_fields_vjp.keys())
    sim_adj = sim_data_vjp.make_adjoint_sim(
        data_vjp_paths=data_vjp_paths, adjoint_monitors=sim_data_fwd.simulation.monitors
    )

    td.log.info(f"Adjoint simulation created with {len(sim_adj.sources)} sources.")

    return sim_adj

def postprocess_adj(
    sim_data_adj: td.SimulationData,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    sim_fields_original: AutogradFieldMap
) -> AutogradFieldMap:
    """Postprocess some data from the adjoint simulation into the VJP for the original sim flds."""

    # map of index into 'structures' to the list of paths we need vjps for
    sim_vjp_map = {}
    for _, structure_index, *structure_path in sim_fields_original.keys():
        structure_path = tuple(structure_path)
        if structure_index in sim_vjp_map:
            sim_vjp_map[structure_index].append(structure_path)
        else:
            sim_vjp_map[structure_index] = [structure_path]

    # store the derivative values given the forward and adjoint data
    sim_fields_vjp = {}
    for structure_index, structure_paths in sim_vjp_map.items():
        # grab the forward and adjoint data
        fld_fwd = sim_data_fwd.get_adjoint_data(structure_index, data_type="fld")
        eps_fwd = sim_data_fwd.get_adjoint_data(structure_index, data_type="eps")
        fld_adj = sim_data_adj.get_adjoint_data(structure_index, data_type="fld")
        eps_adj = sim_data_adj.get_adjoint_data(structure_index, data_type="eps")

        # maps of the E_fwd * E_adj and D_fwd * D_adj, each as as td.FieldData & 'Ex', 'Ey', 'Ez'
        der_maps = get_derivative_maps(fld_fwd=fld_fwd, eps_fwd=eps_fwd, fld_adj=fld_adj, eps_adj=eps_adj)
        E_der_map = der_maps["E"]
        D_der_map = der_maps["D"]

        # compute the derivatives for this structure
        structure = sim_data_fwd.simulation.structures[structure_index]
        vjp_value_map = structure.compute_derivatives(
            structure_paths=structure_paths,
            E_der_map=E_der_map,
            D_der_map=D_der_map,
            eps_structure=eps_fwd,
            eps_sim=sim_data_orig.simulation.medium.permittivity,
            bounds=structure.geometry.bounds,  # TODO: pass intersecting bounds with sim?
        )

        # extract VJPs and put back into sim_fields_vjp AutogradFieldMap
        for structure_path, vjp_value in vjp_value_map.items():
            sim_path = tuple(["structures", structure_index] + list(structure_path))
            sim_fields_vjp[sim_path] = vjp_value

    return sim_fields_vjp

""" Register primitives and VJP makers used by the user-facing functions."""

defvjp(_run_primitive, _run_bwd, argnums=[0])
# defvjp(_run_async_primitive, _run_async_bwd, argnums=[0])


""" The fundamental Tidy3D run and run_async functions used above. """

def _run_tidy3d(simulation: td.Simulation, task_name: str, **run_kwargs) -> td.SimulationData:
    """Run a simulation without any tracers using regular web.run()."""
    td.log.info("running regular simulation with '_run_tidy3d()'")
    # TODO: set task_type to "tidy3d adjoint autograd?"
    data = run_webapi(simulation, task_name=task_name, **run_kwargs)
    return data

def _run_async_tidy3d(simulations: dict[str, td.Simulation],  **run_kwargs) -> BatchData:
    """Run a simulation without any tracers using regular ``web.run_async``."""
    td.log.info("running batch of simulations with '_run_async_tidy3d()'")
    # TODO: set task_type to "tidy3d adjoint autograd?"
    batch_data = run_async_webapi(simulations, **run_kwargs)
    return batch_data
