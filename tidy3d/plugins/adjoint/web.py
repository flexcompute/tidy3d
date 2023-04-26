"""Adjoint-specific webapi."""
from typing import Tuple, Dict, Union
from functools import partial

from jax import custom_vjp

from ...components.simulation import Simulation
from ...components.data.sim_data import SimulationData
from ...web.webapi import run as web_run
from ...web.asynchronous import run_async as web_run_async
from ...web.container import BatchData, DEFAULT_DATA_DIR

from .components.simulation import JaxSimulation, JaxInfo
from .components.data.sim_data import JaxSimulationData


# temporarily locally store forward sim data for testing purposes
# TODO: replace with our own storage system
CACHE = {}

# how we store a key to the forward sim
# note, must be jax-compatible (not string!) as it's returned by one of the jax vjp makers
FwdSimKeyType = int


def _task_name_fwd(task_name: str) -> str:
    """task name for forward run as a function of the original task name."""
    return str(task_name) + "_fwd"


def _task_name_adj(task_name: str) -> str:
    """task name for adjoint run as a function of the original task name."""
    return str(task_name) + "_adj"


def tidy3d_run_fn(simulation: Simulation, task_name: str, **kwargs) -> SimulationData:
    """Run a regular :class:`.Simulation` after conversion from jax type."""
    return web_run(simulation=simulation, task_name=task_name, **kwargs)


def tidy3d_run_async_fn(simulations: Dict[str, Simulation], **kwargs) -> BatchData:
    """Run a set of regular :class:`.Simulation` objects after conversion from jax type."""
    return web_run_async(simulations=simulations, **kwargs)


""" Running a single simulation using web.run. """


# pylint:disable=too-many-arguments
@partial(custom_vjp, nondiff_argnums=tuple(range(1, 6)))
def run(
    simulation: JaxSimulation,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
    verbose: bool = True,
) -> JaxSimulationData:
    """Submits a :class:`.JaxSimulation` to server, starts running, monitors progress, downloads,
    and loads results as a :class:`.JaxSimulationData` object.
    Can be included within a function that will have ``jax.grad`` applied.

    Parameters
    ----------
    simulation : :class:`.JaxSimulation`
        Simulation to upload to server.
    task_name : str
        Name of task.
    path : str = "simulation_data.hdf5"
        Path to download results file (.hdf5), including filename.
    folder_name : str = "default"
        Name of folder to store task on web UI.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.

    Returns
    -------
    :class:`.JaxSimulationData`
        Object containing solver results for the supplied :class:`.JaxSimulation`.
    """

    sim, jax_info = simulation.to_simulation()

    sim_data = tidy3d_run_fn(
        simulation=sim,
        task_name=str(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
    )
    return JaxSimulationData.from_sim_data(sim_data, jax_info)


# pylint:disable=too-many-arguments
def run_fwd(
    simulation: JaxSimulation,
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    verbose: bool,
) -> Tuple[JaxSimulationData, Tuple[FwdSimKeyType, float]]:
    """Run forward pass and stash extra objects for the backwards pass."""

    simulation, jax_info = simulation.to_simulation()

    resp = webapi_run_adjoint_fwd(
        simulation=simulation,
        jax_info=jax_info,
        task_name=str(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
    )

    # TODO: handle error if keys not present in response
    sim_data_orig = SimulationData.from_file(path)
    jax_sim_data_orig = JaxSimulationData.from_sim_data(sim_data_orig, jax_info)
    fwd_sim_key = resp.get("fwd_sim_key")
    fwidth_adj = resp.get("fwidth_adj")

    return jax_sim_data_orig, (fwd_sim_key,)


# pylint:disable=too-many-arguments
def run_bwd(
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    verbose: bool,
    res: tuple,
    sim_data_vjp: JaxSimulationData,
) -> Tuple[JaxSimulation]:
    """Run backward pass and return simulation storing vjp of the objective w.r.t. the sim."""

    (fwd_sim_key,) = res

    fwidth_adj = sim_data_vjp.simulation._fwidth_adjoint
    jax_sim_adj = sim_data_vjp.make_adjoint_simulation(fwidth=fwidth_adj)
    sim_adj, jax_info_adj = jax_sim_adj.to_simulation()

    resp = webapi_run_adjoint_bwd(
        sim_adj=sim_adj,
        jax_info_adj=jax_info_adj,
        fwd_sim_key=fwd_sim_key,
        task_name=_task_name_adj(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
    )

    # TODO: handle error if keys not present in response
    sim_vjp = resp.get("sim_vjp")

    sim_vjp = sim_data_vjp.simulation.updated_copy(input_structures=sim_vjp.input_structures)

    return (sim_vjp,)


"""TO DO: IMPLEMENT this section IN WEBAPI """


# pylint:disable=unused-argument
def get_fwd_sim_key(sim_fwd: Simulation) -> FwdSimKeyType:
    """Returns a unique identifier for the forward task."""

    # todo: find way to assign unique identifier to forward simulation
    return len(CACHE)


def store_fwd_data(fwd_sim_key: FwdSimKeyType, sim_data_fwd: JaxSimulationData) -> None:
    """Store the forward pass data on our servers."""

    # TODO: save this data to storage, could use pickle as we control it
    CACHE[fwd_sim_key] = sim_data_fwd


def get_fwd_data(fwd_sim_key: FwdSimKeyType) -> JaxSimulationData:
    """Grab the forward pass data from our servers."""

    # TODO: grab from storage
    return CACHE[fwd_sim_key]


# TODO: implement this as a webapi that can be called via http request
def webapi_run_adjoint_fwd(
    simulation: Simulation,
    jax_info: JaxInfo,
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    verbose: bool,
) -> Dict[str, Union[SimulationData, FwdSimKeyType, float]]:
    """Runs the forward simulation on our servers, stores the gradient data for later."""

    # Make a regular Simulation but with the gradient monitors
    # TODO: can this be done better / fewer copying around? Currently:
    # Simulation -> JaxSimulation -> JaxSimulation -> Simulation
    jax_sim = JaxSimulation.from_simulation(simulation, jax_info)
    grad_mnts = jax_sim.get_grad_monitors()
    jax_sim_fwd = jax_sim.updated_copy(**grad_mnts)
    sim_fwd, jax_info = jax_sim_fwd.to_simulation()

    sim_data = tidy3d_run_fn(
        simulation=sim_fwd,
        task_name=str(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
    )
    jax_sim_data_fwd = JaxSimulationData.from_sim_data(sim_data, jax_info)

    # simulation data (without gradient data), written to the path file
    # TODO: Why doesn't this work instead of calling a separate tidy3d_fun_fn?
    sim_data_orig, _ = JaxSimulationData.split_fwd_sim_data(sim_data=sim_data, jax_info=jax_info)
    # # Uncomment to make test pass
    # sim_data_orig = tidy3d_run_fn(
    #     simulation=simulation,
    #     task_name=_task_name_fwd(task_name),
    #     folder_name=folder_name,
    #     path=path,
    #     callback_url=callback_url,
    #     verbose=verbose,
    # )

    # store the forward data (including monitor data) on our servers
    # TODO: store this to file
    fwd_sim_key = get_fwd_sim_key(sim_fwd=sim_fwd)
    store_fwd_data(fwd_sim_key=fwd_sim_key, sim_data_fwd=jax_sim_data_fwd)

    # TODO: there should be no return here but instead things should be packaged in files
    return dict(fwd_sim_key=fwd_sim_key)


# TODO: implement this as a webapi that can be called via http request
def webapi_run_adjoint_bwd(
    sim_adj: Simulation,
    jax_info_adj: JaxInfo,
    fwd_sim_key: FwdSimKeyType,
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    verbose: bool,
) -> Dict[str, JaxSimulation]:
    """Runs adjoint simulation on our servers, grabs the gradient data from fwd for processing."""

    jax_sim_adj = JaxSimulation.from_simulation(sim_adj, jax_info_adj)

    sim_data_fwd = get_fwd_data(fwd_sim_key=fwd_sim_key)

    grad_data_fwd = sim_data_fwd.grad_data
    grad_eps_data_fwd = sim_data_fwd.grad_eps_data

    sim_data_adj = run(
        simulation=jax_sim_adj,
        task_name=_task_name_adj(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
    )
    grad_data_adj = sim_data_adj.grad_data

    # get gradient and insert into the resulting simulation structure medium
    sim_vjp = sim_data_adj.simulation.store_vjp(grad_data_fwd, grad_data_adj, grad_eps_data_fwd)

    # test that inputs and outputs are serializable
    # TODO: remove in production version.
    sim_adj.json()
    sim_vjp.json()

    # TODO: package in a file to be downloaded
    return dict(sim_vjp=sim_vjp)


""" END WEBAPI ADDITIONS """

# register the custom forward and backward functions
run.defvjp(run_fwd, run_bwd)


""" Running a batch of simulations using web.run_async. """


def _task_name_orig(index: int, task_name_suffix: str = None):
    """Task name as function of index into simulations. Note: for original must be int."""
    if task_name_suffix is not None:
        return f"{index}{task_name_suffix}"
    return int(index)


# pylint:disable=too-many-locals
@partial(custom_vjp, nondiff_argnums=tuple(range(1, 7)))
def run_async(
    simulations: Tuple[JaxSimulation, ...],
    folder_name: str = "default",
    path_dir: str = DEFAULT_DATA_DIR,
    callback_url: str = None,
    verbose: bool = True,
    num_workers: int = None,
    task_name_suffix: str = None,
) -> Tuple[JaxSimulationData, ...]:
    """Submits a set of :class:`.JaxSimulation` objects to server, starts running,
    monitors progress, downloads, and loads results
    as a tuple of :class:`.JaxSimulationData` objects.
    Uses ``asyncio`` to perform these steps asynchronously.

    Parameters
    ----------
    simulations : Tuple[:class:`.JaxSimulation`, ...]
        Collection of :class:`.JaxSimulations` to run asynchronously.
    folder_name : str = "default"
        Name of folder to store each task on web UI.
    path_dir : str
        Base directory where data will be downloaded, by default current working directory.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.
    num_workers: int = None
        Number of tasks to submit at once in a batch, if None, will run all at the same time.

    Note
    ----
    This is an experimental feature and may not work on all systems or configurations.
    For more details, see ``https://realpython.com/async-io-python/``.

    Returns
    ------
    Tuple[:class:`JaxSimulationData, ...]
        Contains the :class:`.JaxSimulationData` of each :class:`.JaxSimulation`.
    """

    simulations = {_task_name_orig(i, task_name_suffix): sim for i, sim in enumerate(simulations)}

    task_info = {task_name: jax_sim.to_simulation() for task_name, jax_sim in simulations.items()}

    # TODO: anyone know a better syntax for this?
    sims_tidy3d = {str(task_name): sim for task_name, (sim, _) in task_info.items()}
    jax_infos = {str(task_name): jax_info for task_name, (_, jax_info) in task_info.items()}

    # run using regular tidy3d simulation running fn
    batch_data_tidy3d = tidy3d_run_async_fn(
        simulations=sims_tidy3d,
        folder_name=folder_name,
        path_dir=path_dir,
        callback_url=callback_url,
        verbose=verbose,
        num_workers=num_workers,
    )

    # convert back to jax type and return
    task_name_suffix = "" if task_name_suffix is None else task_name_suffix
    jax_batch_data = []
    for i in range(len(simulations)):
        task_name = _task_name_orig(i, task_name_suffix)
        sim_data_tidy3d = batch_data_tidy3d[task_name]
        jax_info = jax_infos[str(task_name)]
        jax_sim_data = JaxSimulationData.from_sim_data(sim_data_tidy3d, jax_info=jax_info)
        jax_batch_data.append(jax_sim_data)

    return jax_batch_data


# pylint:disable=too-many-locals, unused-argument
def run_async_fwd(
    simulations: Tuple[JaxSimulation, ...],
    folder_name: str,
    path_dir: str,
    callback_url: str,
    verbose: bool,
    num_workers: int,
    task_name_suffix: str,
) -> Tuple[Dict[str, JaxSimulationData], tuple]:
    """Run forward pass and stash extra objects for the backwards pass."""

    task_name_suffix_fwd = _task_name_fwd("")

    sims_fwd = []
    for simulation in simulations:
        grad_mnts = simulation.get_grad_monitors()
        sim_fwd = simulation.updated_copy(**grad_mnts)
        sims_fwd.append(sim_fwd)

    batch_data_fwd = run_async(
        simulations=sims_fwd,
        folder_name=folder_name,
        path_dir=path_dir,
        callback_url=callback_url,
        verbose=verbose,
        num_workers=num_workers,
        task_name_suffix=task_name_suffix_fwd,
    )

    # remove the gradient data from the returned version (not needed)
    batch_data_orig = []
    for i, sim_data_fwd in enumerate(batch_data_fwd):
        sim_orig = simulations[i]
        sim_data_orig = sim_data_fwd.copy(update=dict(grad_data=(), simulation=sim_orig))
        batch_data_orig.append(sim_data_orig)

    return batch_data_orig, (batch_data_fwd,)


# pylint:disable=too-many-arguments, too-many-locals, unused-argument
def run_async_bwd(
    folder_name: str,
    path_dir: str,
    callback_url: str,
    verbose: bool,
    num_workers: int,
    task_name_suffix: str,
    res: tuple,
    batch_data_vjp: Tuple[JaxSimulationData, ...],
) -> Tuple[Dict[str, JaxSimulation]]:
    """Run backward pass and return simulation storing vjp of the objective w.r.t. the sim."""

    # grab the forward simulation and its gradient monitor data
    (batch_data_fwd,) = res

    task_name_suffix_adj = _task_name_adj("")

    grad_data_fwd = {}
    grad_eps_data_fwd = {}

    for i, sim_data_fwd in enumerate(batch_data_fwd):
        grad_data_fwd[i] = sim_data_fwd.grad_data
        grad_eps_data_fwd[i] = sim_data_fwd.grad_eps_data

    # make and run adjoint simulation
    sims_adj = []
    for i, sim_data_fwd in enumerate(batch_data_fwd):
        fwidth_adj = sim_data_fwd.simulation._fwidth_adjoint  # pylint:disable=protected-access
        sim_data_vjp = batch_data_vjp[i]
        sim_adj = sim_data_vjp.make_adjoint_simulation(fwidth=fwidth_adj)
        sims_adj.append(sim_adj)

    batch_data_adj = run_async(
        simulations=sims_adj,
        folder_name=folder_name,
        path_dir=path_dir,
        callback_url=callback_url,
        verbose=verbose,
        num_workers=num_workers,
        task_name_suffix=task_name_suffix_adj,
    )

    sims_vjp = []
    for i, (sim_data_fwd, sim_data_adj) in enumerate(zip(batch_data_fwd, batch_data_adj)):

        grad_data_fwd = sim_data_fwd.grad_data
        grad_data_adj = sim_data_adj.grad_data
        grad_data_eps_fwd = sim_data_fwd.grad_eps_data

        sim_data_vjp = batch_data_vjp[i]
        sim_vjp = sim_data_vjp.simulation.store_vjp(grad_data_fwd, grad_data_adj, grad_data_eps_fwd)
        sims_vjp.append(sim_vjp)

    return (sims_vjp,)


# register the custom forward and backward functions
run_async.defvjp(run_async_fwd, run_async_bwd)
