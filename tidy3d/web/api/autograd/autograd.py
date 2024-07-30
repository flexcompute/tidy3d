# autograd wrapper for web functions

import tempfile
import typing

import numpy as np
from autograd.builtins import dict as dict_ag
from autograd.extend import defvjp, primitive

import tidy3d as td
from tidy3d.components.autograd import AutogradFieldMap, get_static

# from tidy3d.components.autograd.utils import split_data_list, split_list
from tidy3d.components.autograd.derivative_utils import DerivativeInfo

from ...core.s3utils import download_file, upload_file
from ..asynchronous import DEFAULT_DATA_DIR
from ..asynchronous import run_async as run_async_webapi
from ..container import BatchData, Job
from ..tidy3d_stub import SimulationDataType, SimulationType
from ..webapi import run as run_webapi
from .utils import FieldMap, get_derivative_maps

# keys for data into auxiliary dictionary
AUX_KEY_SIM_DATA_ORIGINAL = "sim_data"
AUX_KEY_SIM_DATA_FWD = "sim_data_fwd_adjoint"
AUX_KEY_FWD_TASK_ID = "task_id_fwd"
AUX_KEY_SIM_ORIGINAL = "sim_original"
# server-side auxiliary files to upload/download
SIM_VJP_FILE = "output/autograd_sim_vjp.hdf5"
SIM_FIELDS_FILE = "autograd_sim_fields.hdf5"

ISSUE_URL = (
    "https://github.com/flexcompute/tidy3d/issues/new?"
    "assignees=tylerflex&labels=adjoint&projects=&template=autograd_bug.md"
)
URL_LINK = f"[blue underline][link={ISSUE_URL}]'{ISSUE_URL}'[/link][/blue underline]"

MAX_NUM_TRACED_STRUCTURES = 500

# default value for whether to do local gradient calculation (True) or server side (False)
LOCAL_GRADIENT = True


def is_valid_for_autograd(simulation: td.Simulation) -> bool:
    """Check whether a supplied simulation can use autograd run."""

    # only support Simulations
    if not isinstance(simulation, td.Simulation):
        return False

    # if no tracers just use regular web.run()
    traced_fields = simulation.strip_traced_fields(
        include_untraced_data_arrays=False, starting_path=("structures",)
    )
    if not traced_fields:
        return False

    # if too many structures, raise an error
    structure_indices = {i for key, i, *_ in traced_fields.keys() if key == "structures"}
    num_traced_structures = len(structure_indices)
    if num_traced_structures > MAX_NUM_TRACED_STRUCTURES:
        raise ValueError(
            f"Autograd support is currently limited to {MAX_NUM_TRACED_STRUCTURES} structures with "
            f"traced fields. Found {num_traced_structures} structures with traced fields."
        )

    return True


def is_valid_for_autograd_async(simulations: dict[str, td.Simulation]) -> bool:
    """Check whether the supplied simulations dict can use autograd run_async."""
    if not isinstance(simulations, dict):
        return False
    if not all(is_valid_for_autograd(sim) for sim in simulations.values()):
        return False
    return True


def run(
    simulation: SimulationType,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
    verbose: bool = True,
    progress_callback_upload: typing.Callable[[float], None] = None,
    progress_callback_download: typing.Callable[[float], None] = None,
    solver_version: str = None,
    worker_group: str = None,
    simulation_type: str = "tidy3d",
    parent_tasks: list[str] = None,
    local_gradient: bool = LOCAL_GRADIENT,
) -> SimulationDataType:
    """
    Submits a :class:`.Simulation` to server, starts running, monitors progress, downloads,
    and loads results as a :class:`.SimulationDataType` object.

    Parameters
    ----------
    simulation : Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]
        Simulation to upload to server.
    task_name : str
        Name of task.
    folder_name : str = "default"
        Name of folder to store task on web UI.
    path : str = "simulation_data.hdf5"
        Path to download results file (.hdf5), including filename.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    simulation_type : str = "tidy3d"
        Type of simulation being uploaded.
    progress_callback_upload : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    progress_callback_download : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.
    solver_version: str = None
        target solver version.
    worker_group: str = None
        worker group
    local_gradient: bool = False
        Whether to perform gradient calculation locally, requiring more downloads but potentially
        more stable with experimental features.

    Returns
    -------
    Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
        Object containing solver results for the supplied simulation.

    Notes
    -----

        Submitting a simulation to our cloud server is very easily done by a simple web API call.

        .. code-block:: python

            sim_data = tidy3d.web.api.webapi.run(simulation, task_name='my_task', path='out/data.hdf5')

        The :meth:`tidy3d.web.api.webapi.run()` method shows the simulation progress by default.  When uploading a
        simulation to the server without running it, you can use the :meth:`tidy3d.web.api.webapi.monitor`,
        :meth:`tidy3d.web.api.container.Job.monitor`, or :meth:`tidy3d.web.api.container.Batch.monitor` methods to
        display the progress of your simulation(s).

    Examples
    --------

        To access the original :class:`.Simulation` object that created the simulation data you can use:

        .. code-block:: python

            # Run the simulation.
            sim_data = web.run(simulation, task_name='task_name', path='out/sim.hdf5')

            # Get a copy of the original simulation object.
            sim_copy = sim_data.simulation

    See Also
    --------

    :meth:`tidy3d.web.api.webapi.monitor`
        Print the real time task progress until completion.

    :meth:`tidy3d.web.api.container.Job.monitor`
        Monitor progress of running :class:`Job`.

    :meth:`tidy3d.web.api.container.Batch.monitor`
        Monitor progress of each of the running tasks.
    """

    if is_valid_for_autograd(simulation):
        return _run(
            simulation=simulation,
            task_name=task_name,
            folder_name=folder_name,
            path=path,
            callback_url=callback_url,
            verbose=verbose,
            progress_callback_upload=progress_callback_upload,
            progress_callback_download=progress_callback_download,
            solver_version=solver_version,
            worker_group=worker_group,
            simulation_type="tidy3d_autograd",
            parent_tasks=parent_tasks,
            local_gradient=local_gradient,
        )

    return run_webapi(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
        progress_callback_upload=progress_callback_upload,
        progress_callback_download=progress_callback_download,
        solver_version=solver_version,
        worker_group=worker_group,
        simulation_type=simulation_type,
        parent_tasks=parent_tasks,
    )


def run_async(
    simulations: dict[str, SimulationType],
    folder_name: str = "default",
    path_dir: str = DEFAULT_DATA_DIR,
    callback_url: str = None,
    num_workers: int = None,
    verbose: bool = True,
    simulation_type: str = "tidy3d",
    parent_tasks: dict[str, list[str]] = None,
) -> BatchData:
    """Submits a set of Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] objects to server,
    starts running, monitors progress, downloads, and loads results as a :class:`.BatchData` object.

    .. TODO add example and see also reference.

    Parameters
    ----------
    simulations : Dict[str, Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]]
        Mapping of task name to simulation.
    folder_name : str = "default"
        Name of folder to store each task on web UI.
    path_dir : str
        Base directory where data will be downloaded, by default current working directory.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    num_workers: int = None
        Number of tasks to submit at once in a batch, if None, will run all at the same time.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.

    Returns
    ------
    :class:`BatchData`
        Contains the Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] for each
        Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] in :class:`Batch`.

    See Also
    --------

    :class:`Job`:
        Interface for managing the running of a Simulation on server.

    :class:`Batch`
        Interface for submitting several :class:`Simulation` objects to sever.
    """

    if is_valid_for_autograd_async(simulations):
        return _run_async(
            simulations=simulations,
            folder_name=folder_name,
            path_dir=path_dir,
            callback_url=callback_url,
            num_workers=num_workers,
            verbose=verbose,
            simulation_type="tidy3d_autograd_async",
            parent_tasks=parent_tasks,
        )

    return run_async_webapi(
        simulations=simulations,
        folder_name=folder_name,
        path_dir=path_dir,
        callback_url=callback_url,
        num_workers=num_workers,
        verbose=verbose,
        simulation_type=simulation_type,
        parent_tasks=parent_tasks,
    )


""" User-facing ``run`` and `run_async`` functions, compatible with ``autograd`` """


def _run(
    simulation: td.Simulation, task_name: str, local_gradient: bool = LOCAL_GRADIENT, **run_kwargs
) -> td.SimulationData:
    """User-facing ``web.run`` function, compatible with ``autograd`` differentiation."""

    traced_fields_sim = setup_run(simulation=simulation)

    # if we register this as not needing adjoint at all (no tracers), call regular run function
    if not traced_fields_sim:
        td.log.warning(
            "No autograd derivative tracers found in the 'Simulation' passed to 'run'. "
            "This could indicate that there is no path from your objective function arguments "
            "to the 'Simulation'. If this is unexpected, double check your objective function "
            "pre-processing. Running regular tidy3d simulation."
        )
        data, _ = _run_tidy3d(simulation, task_name=task_name, **run_kwargs)
        return data

    # will store the SimulationData for original and forward so we can access them later
    aux_data = {}

    # run our custom @primitive, passing the traced fields first to register with autograd
    traced_fields_data = _run_primitive(
        traced_fields_sim,  # if you pass as a kwarg it will not trace :/
        sim_original=simulation.to_static(),
        task_name=task_name,
        aux_data=aux_data,
        local_gradient=local_gradient,
        **run_kwargs,
    )

    return postprocess_run(traced_fields_data=traced_fields_data, aux_data=aux_data)


def _run_async(
    simulations: dict[str, td.Simulation], **run_async_kwargs
) -> dict[str, td.SimulationData]:
    """User-facing ``web.run_async`` function, compatible with ``autograd`` differentiation."""

    task_names = simulations.keys()

    traced_fields_sim_dict = {}
    for task_name in task_names:
        traced_fields_sim_dict[task_name] = setup_run(simulation=simulations[task_name])
    traced_fields_sim_dict = dict_ag(traced_fields_sim_dict)

    # TODO: shortcut primitive running for any items with no tracers?

    aux_data_dict = {task_name: {} for task_name in task_names}
    sims_original = {
        task_name: simulation.to_static() for task_name, simulation in simulations.items()
    }
    traced_fields_data_dict = _run_async_primitive(
        traced_fields_sim_dict,  # if you pass as a kwarg it will not trace :/
        sims_original=sims_original,
        aux_data_dict=aux_data_dict,
        **run_async_kwargs,
    )

    # TODO: package this as a Batch? it might be not possible as autograd tracers lose their
    # powers when we save them to file.
    sim_data_dict = {}
    for task_name in task_names:
        traced_fields_data = traced_fields_data_dict[task_name]
        aux_data = aux_data_dict[task_name]
        sim_data = postprocess_run(traced_fields_data=traced_fields_data, aux_data=aux_data)
        sim_data_dict[task_name] = sim_data

    return sim_data_dict


def setup_run(simulation: td.Simulation) -> AutogradFieldMap:
    """Process a user-supplied ``Simulation`` into inputs to ``_run_primitive``."""

    # get a mapping of all the traced fields in the provided simulation
    return simulation.strip_traced_fields(
        include_untraced_data_arrays=False, starting_path=("structures",)
    )


def postprocess_run(traced_fields_data: AutogradFieldMap, aux_data: dict) -> td.SimulationData:
    """Process the return from ``_run_primitive`` into ``SimulationData`` for user."""

    # grab the user's 'SimulationData' and return with the autograd-tracers inserted
    sim_data_original = aux_data[AUX_KEY_SIM_DATA_ORIGINAL]
    return sim_data_original.insert_traced_fields(traced_fields_data)


""" Autograd-traced Primitive for FWD pass ``run`` functions """


@primitive
def _run_primitive(
    sim_fields: AutogradFieldMap,
    sim_original: td.Simulation,
    task_name: str,
    aux_data: dict,
    local_gradient: bool,
    **run_kwargs,
) -> AutogradFieldMap:
    """Autograd-traced 'run()' function: runs simulation, strips tracer data, caches fwd data."""

    td.log.info("running primitive '_run_primitive()'")

    if local_gradient:
        sim_combined = setup_fwd(
            sim_fields=sim_fields,
            sim_original=sim_original,
            local_gradient=local_gradient,
        )
        sim_data_combined, _ = _run_tidy3d(sim_combined, task_name=task_name, **run_kwargs)

        # TODO: put this in postprocess?
        # aux_data[AUX_KEY_FWD_TASK_ID] = task_id_fwd

        field_map = postprocess_fwd(
            sim_data_combined=sim_data_combined,
            sim_original=sim_original,
            aux_data=aux_data,
        )

    else:
        run_kwargs["simulation_type"] = "autograd_fwd"
        run_kwargs["sim_fields"] = sim_fields

        sim_data_orig, task_id_fwd = _run_tidy3d(
            sim_original,
            task_name=task_name,
            **run_kwargs,
        )

        # TODO: put this in postprocess?
        aux_data[AUX_KEY_FWD_TASK_ID] = task_id_fwd
        aux_data[AUX_KEY_SIM_DATA_ORIGINAL] = sim_data_orig
        field_map = sim_data_orig.strip_traced_fields(
            include_untraced_data_arrays=True, starting_path=("data",)
        )

    return field_map


@primitive
def _run_async_primitive(
    sim_fields_dict: dict[str, AutogradFieldMap],
    sims_original: dict[str, td.Simulation],
    aux_data_dict: dict[dict[str, typing.Any]],
    **run_async_kwargs,
) -> dict[str, AutogradFieldMap]:
    task_names = sim_fields_dict.keys()

    sims_combined = {}
    for task_name in task_names:
        sim_fields = sim_fields_dict[task_name]
        sim_original = sims_original[task_name]
        sims_combined[task_name] = setup_fwd(sim_fields=sim_fields, sim_original=sim_original)

    batch_data_combined = _run_async_tidy3d(sims_combined, **run_async_kwargs)

    field_map_fwd_dict = {}
    for task_name in task_names:
        sim_data_combined = batch_data_combined[task_name]
        sim_original = sims_original[task_name]
        aux_data = aux_data_dict[task_name]
        field_map_fwd_dict[task_name] = postprocess_fwd(
            sim_data_combined=sim_data_combined,
            sim_original=sim_original,
            aux_data=aux_data,
        )

    return field_map_fwd_dict


def setup_fwd(
    sim_fields: AutogradFieldMap,
    sim_original: td.Simulation,
    local_gradient: bool = LOCAL_GRADIENT,
) -> td.Simulation:
    """Set up the combined forward simulation."""

    # if local gradient, make and run a sim with combined original & adjoint monitors
    if local_gradient:
        return sim_original.with_adjoint_monitors(sim_fields)

    # if remote gradient, add them later
    return sim_original


def upload_sim_fields(sim_fields: AutogradFieldMap, task_id: str, verbose: bool = False):
    """Function to grab the VJP result for the simulation fields from the adjoint task ID."""
    data_file = tempfile.NamedTemporaryFile(suffix=".hdf5")
    data_file.close()
    FieldMap.from_autograd_field_map(sim_fields).to_file(data_file.name)
    upload_file(
        task_id,
        data_file.name,
        SIM_FIELDS_FILE,
        verbose=verbose,
    )


def postprocess_fwd(
    sim_data_combined: td.SimulationData,
    sim_original: td.Simulation,
    aux_data: dict,
) -> AutogradFieldMap:
    """Postprocess the combined simulation data into an Autograd field map."""

    num_mnts_original = len(sim_original.monitors)
    sim_data_original, sim_data_fwd = sim_data_combined.split_original_fwd(
        num_mnts_original=num_mnts_original
    )

    aux_data[AUX_KEY_SIM_DATA_ORIGINAL] = sim_data_original
    aux_data[AUX_KEY_SIM_DATA_FWD] = sim_data_fwd

    # strip out the tracer AutogradFieldMap for the .data from the original sim
    data_traced = sim_data_original.strip_traced_fields(
        include_untraced_data_arrays=True, starting_path=("data",)
    )

    # return the AutogradFieldMap that autograd registers as the "output" of the primitive
    return data_traced


""" VJP maker for ADJ pass."""


def get_vjp_traced_fields(task_id_adj: str, verbose: bool) -> AutogradFieldMap:
    """Function to grab the VJP result for the simulation fields from the adjoint task ID."""
    data_file = tempfile.NamedTemporaryFile(suffix=".hdf5")
    data_file.close()
    download_file(task_id_adj, SIM_VJP_FILE, to_file=data_file.name, verbose=verbose)
    field_map = FieldMap.from_file(data_file.name)
    return field_map.to_autograd_field_map


def _run_bwd(
    data_fields_original: AutogradFieldMap,
    sim_fields_original: AutogradFieldMap,
    sim_original: td.Simulation,
    task_name: str,
    aux_data: dict,
    local_gradient: bool,
    **run_kwargs,
) -> typing.Callable[[AutogradFieldMap], AutogradFieldMap]:
    """VJP-maker for ``_run_primitive()``. Constructs and runs adjoint simulation, computes grad."""

    # get the fwd epsilon and field data from the cached aux_data
    sim_data_orig = aux_data[AUX_KEY_SIM_DATA_ORIGINAL]

    if local_gradient:
        sim_data_fwd = aux_data[AUX_KEY_SIM_DATA_FWD]

    td.log.info("constructing custom vjp function for backwards pass.")

    def vjp(data_fields_vjp: AutogradFieldMap) -> AutogradFieldMap:
        """dJ/d{sim.traced_fields()} as a function of Function of dJ/d{data.traced_fields()}"""

        sim_adj = setup_adj(
            data_fields_vjp=data_fields_vjp,
            sim_data_orig=sim_data_orig,
            sim_fields_original=sim_fields_original,
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
        task_name_adj = str(task_name) + "_adjoint"

        if local_gradient:
            sim_data_adj, _ = _run_tidy3d(sim_adj, task_name=task_name_adj, **run_kwargs)

            vjp_traced_fields = postprocess_adj(
                sim_data_adj=sim_data_adj,
                sim_data_orig=sim_data_orig,
                sim_data_fwd=sim_data_fwd,
                sim_fields_original=sim_fields_original,
            )

        else:
            task_id_fwd = aux_data[AUX_KEY_FWD_TASK_ID]
            run_kwargs["parent_tasks"] = [task_id_fwd]
            run_kwargs["simulation_type"] = "autograd_bwd"

            vjp_traced_fields = _run_tidy3d_bwd(sim_adj, task_name=task_name_adj, **run_kwargs)

        return vjp_traced_fields

    return vjp


def _run_async_bwd(
    data_fields_original_dict: dict[str, AutogradFieldMap],
    sim_fields_original_dict: dict[str, AutogradFieldMap],
    sims_original: dict[str, td.Simulation],
    aux_data_dict: dict[str, dict[str, typing.Any]],
    **run_async_kwargs,
) -> typing.Callable[[dict[str, AutogradFieldMap]], dict[str, AutogradFieldMap]]:
    """VJP-maker for ``_run_primitive()``. Constructs and runs adjoint simulation, computes grad."""

    task_names = data_fields_original_dict.keys()

    # get the fwd epsilon and field data from the cached aux_data
    sim_data_orig_dict = {}
    sim_data_fwd_dict = {}
    for task_name in task_names:
        aux_data = aux_data_dict[task_name]
        sim_data_orig_dict[task_name] = aux_data[AUX_KEY_SIM_DATA_ORIGINAL]
        sim_data_fwd_dict[task_name] = aux_data[AUX_KEY_SIM_DATA_FWD]

    td.log.info("constructing custom vjp function for backwards pass.")

    def vjp(data_fields_dict_vjp: dict[str, AutogradFieldMap]) -> dict[str, AutogradFieldMap]:
        """dJ/d{sim.traced_fields()} as a function of Function of dJ/d{data.traced_fields()}"""

        task_names_adj = {task_name + "_adjoint" for task_name in task_names}

        sims_adj = {}
        for task_name, task_name_adj in zip(task_names, task_names_adj):
            data_fields_vjp = data_fields_dict_vjp[task_name]
            sim_data_orig = sim_data_orig_dict[task_name]
            sim_data_fwd = sim_data_fwd_dict[task_name]
            sim_fields_original = sim_fields_original_dict[task_name]

            sim_adj = setup_adj(
                data_fields_vjp=data_fields_vjp,
                sim_data_orig=sim_data_orig,
                sim_fields_original=sim_fields_original,
            )
            sims_adj[task_name_adj] = sim_adj

            # TODO: handle case where no adjoint sources?

        # run adjoint simulation
        batch_data_adj = _run_async_tidy3d(sims_adj, **run_async_kwargs)

        sim_fields_vjp_dict = {}
        for task_name, task_name_adj in zip(task_names, task_names_adj):
            sim_data_adj = batch_data_adj[task_name_adj]
            sim_data_orig = sim_data_orig_dict[task_name]
            sim_data_fwd = sim_data_fwd_dict[task_name]
            sim_fields_original = sim_fields_original_dict[task_name]

            sim_fields_vjp = postprocess_adj(
                sim_data_adj=sim_data_adj,
                sim_data_orig=sim_data_orig,
                sim_data_fwd=sim_data_fwd,
                sim_fields_original=sim_fields_original,
            )
            sim_fields_vjp_dict[task_name] = sim_fields_vjp

        return sim_fields_vjp_dict

    return vjp


def setup_adj(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    sim_fields_original: AutogradFieldMap,
) -> td.Simulation:
    """Construct an adjoint simulation from a set of data_fields for the VJP."""

    td.log.info("Running custom vjp (adjoint) pipeline.")

    # immediately filter out any data_vjps with all 0's in the data
    data_fields_vjp = {
        key: get_static(value) for key, value in data_fields_vjp.items() if not np.all(value == 0.0)
    }

    # insert the raw VJP data into the .data of the original SimulationData
    sim_data_vjp = sim_data_orig.insert_traced_fields(field_mapping=data_fields_vjp)

    # make adjoint simulation from that SimulationData
    data_vjp_paths = set(data_fields_vjp.keys())

    num_monitors = len(sim_data_orig.simulation.monitors)
    adjoint_monitors = sim_data_orig.simulation.with_adjoint_monitors(sim_fields_original).monitors[
        num_monitors:
    ]

    sim_adj = sim_data_vjp.make_adjoint_sim(
        data_vjp_paths=data_vjp_paths,
        adjoint_monitors=adjoint_monitors,
    )

    td.log.info(f"Adjoint simulation created with {len(sim_adj.sources)} sources.")

    return sim_adj


def postprocess_adj(
    sim_data_adj: td.SimulationData,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    sim_fields_original: AutogradFieldMap,
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
        der_maps = get_derivative_maps(
            fld_fwd=fld_fwd, eps_fwd=eps_fwd, fld_adj=fld_adj, eps_adj=eps_adj
        )
        E_der_map = der_maps["E"]
        D_der_map = der_maps["D"]

        # compute the derivatives for this structure
        structure = sim_data_fwd.simulation.structures[structure_index]

        # todo: handle multi-frequency, move to a property?
        frequencies = {src.source_time.freq0 for src in sim_data_adj.simulation.sources}
        frequencies = list(frequencies)
        if len(frequencies) != 1:
            raise RuntimeError("Multiple adjoint frequencies found.")
        freq_adj = frequencies[0]

        eps_in = np.mean(structure.medium.eps_model(freq_adj))
        eps_out = np.mean(sim_data_orig.simulation.medium.eps_model(freq_adj))

        derivative_info = DerivativeInfo(
            paths=structure_paths,
            E_der_map=E_der_map.field_components,
            D_der_map=D_der_map.field_components,
            eps_data=eps_fwd.field_components,
            eps_in=eps_in,
            eps_out=eps_out,
            frequency=freq_adj,
            bounds=structure.geometry.bounds,  # TODO: pass intersecting bounds with sim?
        )

        vjp_value_map = structure.compute_derivatives(derivative_info)

        # extract VJPs and put back into sim_fields_vjp AutogradFieldMap
        for structure_path, vjp_value in vjp_value_map.items():
            sim_path = tuple(["structures", structure_index] + list(structure_path))
            sim_fields_vjp[sim_path] = vjp_value

    return sim_fields_vjp


""" Register primitives and VJP makers used by the user-facing functions."""

defvjp(_run_primitive, _run_bwd, argnums=[0])
defvjp(_run_async_primitive, _run_async_bwd, argnums=[0])


""" The fundamental Tidy3D run and run_async functions used above. """


def parse_run_kwargs(**run_kwargs):
    """Parse the ``run_kwargs`` to extract what should be passed to the ``Job`` initialization."""
    job_fields = list(Job._upload_fields) + ["solver_version"]
    job_init_kwargs = {k: v for k, v in run_kwargs.items() if k in job_fields}
    return job_init_kwargs


def _run_tidy3d(
    simulation: td.Simulation, task_name: str, **run_kwargs
) -> (td.SimulationData, str):
    """Run a simulation without any tracers using regular web.run()."""
    job_init_kwargs = parse_run_kwargs(**run_kwargs)
    job = Job(simulation=simulation, task_name=task_name, **job_init_kwargs)
    td.log.info(f"running {job.simulation_type} simulation with '_run_tidy3d()'")
    if job.simulation_type == "autograd_fwd":
        verbose = run_kwargs.get("verbose", False)
        upload_sim_fields(run_kwargs["sim_fields"], task_id=job.task_id, verbose=verbose)
    data = job.run()
    return data, job.task_id


def _run_tidy3d_bwd(simulation: td.Simulation, task_name: str, **run_kwargs) -> AutogradFieldMap:
    """Run a simulation without any tracers using regular web.run()."""
    job_init_kwargs = parse_run_kwargs(**run_kwargs)
    job = Job(simulation=simulation, task_name=task_name, **job_init_kwargs)
    td.log.info(f"running {job.simulation_type} simulation with '_run_tidy3d_bwd()'")
    job.start()
    job.monitor()
    return get_vjp_traced_fields(task_id_adj=job.task_id, verbose=job.verbose)


def _run_async_tidy3d(simulations: dict[str, td.Simulation], **run_async_kwargs) -> BatchData:
    """Run a simulation without any tracers using regular ``web.run_async``."""
    td.log.info("running batch of simulations with '_run_async_tidy3d()'")
    # TODO: set task_type to "tidy3d adjoint autograd?"
    batch_data = run_async_webapi(simulations, **run_async_kwargs)
    return batch_data
