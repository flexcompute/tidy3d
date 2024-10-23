"""Adjoint-specific webapi."""

import os
import tempfile
from functools import partial
from typing import Dict, List, Tuple

import pydantic.v1 as pd
from jax import custom_vjp
from jax.tree_util import register_pytree_node_class

import tidy3d as td
from tidy3d.web.api.asynchronous import run_async as web_run_async
from tidy3d.web.api.webapi import run as web_run
from tidy3d.web.api.webapi import wait_for_connection
from tidy3d.web.core.s3utils import download_file, upload_file

from ...components.data.sim_data import SimulationData
from ...components.simulation import Simulation
from ...components.types import Literal
from ...web.api.container import DEFAULT_DATA_DIR, Batch, BatchData, Job
from .components.base import JaxObject
from .components.data.sim_data import JaxSimulationData
from .components.simulation import NUM_PROC_LOCAL, JaxInfo, JaxSimulation

# file names and paths for server side adjoint
SIM_VJP_FILE = "output/jax_sim_vjp.hdf5"
JAX_INFO_FILE = "jax_info.json"


@register_pytree_node_class
class RunResidual(JaxObject):
    """Class to store extra data needed to pass between the forward and backward adjoint run."""

    fwd_task_id: str = pd.Field(
        ..., title="Forward task_id", description="task_id of the forward simulation."
    )


@register_pytree_node_class
class RunResidualBatch(JaxObject):
    """Class to store extra data needed to pass between the forward and backward adjoint run."""

    fwd_task_ids: Tuple[str, ...] = pd.Field(
        ..., title="Forward task_ids", description="task_ids of the forward simulations."
    )


@register_pytree_node_class
class RunResidualAsync(JaxObject):
    """Class to store extra data needed to pass between the forward and backward adjoint run."""

    fwd_task_ids: Dict[str, str] = pd.Field(
        ..., title="Forward task_ids", description="task_ids of the forward simulation for async."
    )


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


def _run(
    simulation: JaxSimulation,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
    verbose: bool = True,
) -> JaxSimulationData:
    """Split the provided ``JaxSimulation`` into a regular ``Simulation`` and a ``JaxInfo`` part,
    run using ``tidy3d_run_fn``, which runs on the server by default but can be monkeypatched,
    and recombine into a ``JaxSimulationData``.
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

    simulation._validate_web_adjoint()

    # TODO: add task_id
    return _run(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
    )


def run_fwd(
    simulation: JaxSimulation,
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    verbose: bool,
) -> Tuple[JaxSimulationData, Tuple[RunResidual]]:
    """Run forward pass and stash extra objects for the backwards pass."""

    simulation._validate_web_adjoint()

    sim_fwd, jax_info_fwd, jax_info_orig = simulation.to_simulation_fwd()

    sim_data_orig, task_id = webapi_run_adjoint_fwd(
        simulation=sim_fwd,
        jax_info=jax_info_fwd,
        task_name=str(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
    )

    res = RunResidual(fwd_task_id=task_id)
    jax_sim_data_orig = JaxSimulationData.from_sim_data(
        sim_data_orig, jax_info_orig, task_id=task_id
    )

    return jax_sim_data_orig, (res,)


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

    fwd_task_id = res[0].fwd_task_id
    fwidth_adj = sim_data_vjp.simulation._fwidth_adjoint
    run_time_adj = sim_data_vjp.simulation._run_time_adjoint
    jax_sim_adj = sim_data_vjp.make_adjoint_simulation(fwidth=fwidth_adj, run_time=run_time_adj)
    sim_adj, jax_info_adj = jax_sim_adj.to_simulation()

    sim_vjp = webapi_run_adjoint_bwd(
        sim_adj=sim_adj,
        jax_info_adj=jax_info_adj,
        fwd_task_id=fwd_task_id,
        task_name=_task_name_adj(task_name),
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
    )

    sim_vjp = sim_data_vjp.simulation.updated_copy(input_structures=sim_vjp.input_structures)

    return (sim_vjp,)


"""TO DO: IMPLEMENT this section IN WEBAPI """


@wait_for_connection
def upload_jax_info(jax_info: JaxInfo, task_id: str, verbose: bool) -> None:
    """Upload jax_info for a task with a given task_id."""
    handle, fname = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    try:
        jax_info.to_file(fname)
        upload_file(
            task_id,
            fname,
            JAX_INFO_FILE,
            verbose=verbose,
        )
    except Exception as e:
        td.log.error(f"Error occurred while uploading 'jax_info': {e}")
        raise e
    finally:
        os.unlink(fname)


@wait_for_connection
def download_sim_vjp(task_id: str, verbose: bool) -> JaxSimulation:
    """Download the vjp loaded simulation from the server to return to jax."""
    handle, fname = tempfile.mkstemp(suffix=".hdf5")
    os.close(handle)
    try:
        download_file(task_id, SIM_VJP_FILE, to_file=fname, verbose=verbose)
        return JaxSimulation.from_file(fname)
    except Exception as e:
        td.log.error(f"Error occurred while downloading 'sim_vjp': {e}")
        raise e
    finally:
        os.unlink(fname)


AdjointSimulationType = Literal["tidy3d", "adjoint_fwd", "adjoint_bwd"]


class AdjointJob(Job):
    """Job that uploads a jax_info object and also includes new fields for adjoint tasks."""

    simulation_type: AdjointSimulationType = pd.Field(
        None,
        title="Simulation Type",
        description="Type of simulation, used internally only.",
    )

    jax_info: JaxInfo = pd.Field(
        None,
        title="Jax Info",
        description="Container of information needed to reconstruct jax simulation.",
    )

    def start(self) -> None:
        """Start running a :class:`AdjointJob`. after uploading jax info.

        Note
        ----
        To monitor progress of the :class:`Job`, call :meth:`Job.monitor` after started.
        """
        if self.jax_info is not None:
            upload_jax_info(task_id=self.task_id, jax_info=self.jax_info, verbose=self.verbose)
        super().start()


class AdjointBatch(Batch):
    """Batch that uploads a jax_info object and also includes new fields for adjoint tasks."""

    simulation_type: AdjointSimulationType = pd.Field(
        "tidy3d",
        title="Simulation Type",
        description="Type of simulation, used internally only.",
    )

    jax_infos: Dict[str, JaxInfo] = pd.Field(
        ...,
        title="Jax Info Dict",
        description="Containers of information needed to reconstruct JaxSimulation for each item.",
    )

    jobs_cached: Dict[str, AdjointJob] = pd.Field(
        None,
        title="Jobs (Cached)",
        description="Optional field to specify ``jobs``. Only used as a workaround internally "
        "so that ``jobs`` is written when ``Batch.to_file()`` and then the proper task is loaded "
        "from ``Batch.from_file()``. We recommend leaving unset as setting this field along with "
        "fields that were not used to create the task will cause errors.",
    )

    _job_type = AdjointJob

    def start(self) -> None:
        """Start running a :class:`AdjointBatch`. after uploading jax info for each job.

        Note
        ----
        To monitor progress of the :class:`Batch`, call :meth:`Batch.monitor` after started.
        """
        for task_name, job in self.jobs.items():
            jax_info = self.jax_infos.get(task_name)
            upload_jax_info(task_id=job.task_id, jax_info=jax_info, verbose=job.verbose)
        super().start()


def webapi_run_adjoint_fwd(
    simulation: Simulation,
    jax_info: JaxInfo,
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    verbose: bool,
) -> Dict[str, float]:
    """Runs the forward simulation on our servers, stores the gradient data for later."""

    job = AdjointJob(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        simulation_type="adjoint_fwd",
        jax_info=jax_info,
    )

    sim_data = job.run(path=path)
    return sim_data, job.task_id


def webapi_run_adjoint_bwd(
    sim_adj: Simulation,
    jax_info_adj: JaxInfo,
    fwd_task_id: str,
    task_name: str,
    folder_name: str,
    callback_url: str,
    verbose: bool,
) -> JaxSimulation:
    """Runs adjoint simulation on our servers, grabs the gradient data from fwd for processing."""

    job = AdjointJob(
        simulation=sim_adj,
        task_name=task_name,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        simulation_type="adjoint_bwd",
        parent_tasks=[fwd_task_id],
        jax_info=jax_info_adj,
    )

    job.start()
    job.monitor()
    sim_vjp = download_sim_vjp(task_id=job.task_id, verbose=verbose)
    return sim_vjp


""" END WEBAPI ADDITIONS """

# register the custom forward and backward functions
run.defvjp(run_fwd, run_bwd)

""" Running a batch of simulations using web.run_async. """


def _task_name_orig(index: int):
    """Task name as function of index into simulations. Note: for original must be int."""
    return int(index)


@partial(custom_vjp, nondiff_argnums=tuple(range(1, 6)))
def run_async(
    simulations: Tuple[JaxSimulation, ...],
    folder_name: str = "default",
    path_dir: str = DEFAULT_DATA_DIR,
    callback_url: str = None,
    verbose: bool = True,
    num_workers: int = None,
) -> Tuple[JaxSimulationData, ...]:
    """Submits a set of :class:`.JaxSimulation` objects to server, starts running,
    monitors progress, downloads, and loads results
    as a tuple of :class:`.JaxSimulationData` objects.

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
    Tuple[:class:`.JaxSimulationData`, ...]
        Contains the :class:`.JaxSimulationData` of each :class:`.JaxSimulation`.
    """

    for simulation in simulations:
        simulation._validate_web_adjoint()

    # get task names, the td.Simulation, and JaxInfo for all supplied simulations
    task_names = [str(_task_name_orig(i)) for i in range(len(simulations))]
    task_info = [jax_sim.to_simulation() for jax_sim in simulations]

    # process this into dictionaries of task_name -> Simulation and task_name -> JaxInfo
    sims, jax_infos = list(zip(*task_info))
    sims_tidy3d = dict(zip(task_names, sims))
    jax_infos = dict(zip(task_names, jax_infos))

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
    jax_batch_data = []
    for i in range(len(simulations)):
        task_name = str(_task_name_orig(i))
        sim_data_tidy3d = batch_data_tidy3d[task_name]
        jax_info = jax_infos[str(task_name)]
        # TODO: add task_id
        jax_sim_data = JaxSimulationData.from_sim_data(sim_data_tidy3d, jax_info=jax_info)
        jax_batch_data.append(jax_sim_data)

    return jax_batch_data


def run_async_fwd(
    simulations: Tuple[JaxSimulation, ...],
    folder_name: str,
    path_dir: str,
    callback_url: str,
    verbose: bool,
    num_workers: int,
) -> Tuple[Tuple[JaxSimulationData, ...], RunResidualBatch]:
    """Run forward pass and stash extra objects for the backwards pass."""

    for simulation in simulations:
        simulation._validate_web_adjoint()

    jax_infos_orig = []
    sims_fwd = []
    jax_infos_fwd = []
    for simulation in simulations:
        sim_fwd, jax_info_fwd, jax_info_orig = simulation.to_simulation_fwd()
        jax_infos_orig.append(jax_info_orig)
        sims_fwd.append(sim_fwd)
        jax_infos_fwd.append(jax_info_fwd)

    batch_data_orig, fwd_task_ids = webapi_run_async_adjoint_fwd(
        simulations=sims_fwd,
        jax_infos=jax_infos_fwd,
        folder_name=folder_name,
        path_dir=path_dir,
        callback_url=callback_url,
        verbose=verbose,
    )

    batch_data_orig = [sim_data for _, sim_data in batch_data_orig.items()]

    jax_batch_data_orig = []
    for sim_data_orig, jax_info_orig, task_id in zip(batch_data_orig, jax_infos_orig, fwd_task_ids):
        jax_sim_data = JaxSimulationData.from_sim_data(
            sim_data_orig, jax_info_orig, task_id=task_id
        )
        jax_batch_data_orig.append(jax_sim_data)

    residual = RunResidualBatch(fwd_task_ids=fwd_task_ids)
    return jax_batch_data_orig, (residual,)


def run_async_bwd(
    folder_name: str,
    path_dir: str,
    callback_url: str,
    verbose: bool,
    num_workers: int,
    res: tuple,
    batch_data_vjp: Tuple[JaxSimulationData, ...],
) -> Tuple[Dict[str, JaxSimulation]]:
    """Run backward pass and return simulation storing vjp of the objective w.r.t. the sim."""

    fwd_task_ids = res[0].fwd_task_ids
    sims_adj = []
    jax_infos_adj = []
    parent_tasks_adj = []

    for sim_data_vjp, fwd_task_id in zip(batch_data_vjp, fwd_task_ids):
        parent_tasks_adj.append([str(fwd_task_id)])
        fwidth_adj = sim_data_vjp.simulation._fwidth_adjoint
        run_time_adj = sim_data_vjp.simulation._run_time_adjoint
        jax_sim_adj = sim_data_vjp.make_adjoint_simulation(fwidth=fwidth_adj, run_time=run_time_adj)
        sim_adj, jax_info_adj = jax_sim_adj.to_simulation()
        sims_adj.append(sim_adj)
        jax_infos_adj.append(jax_info_adj)

    sims_vjp = webapi_run_async_adjoint_bwd(
        simulations=sims_adj,
        jax_infos=jax_infos_adj,
        folder_name=folder_name,
        path_dir=path_dir,
        callback_url=callback_url,
        verbose=verbose,
        parent_tasks=parent_tasks_adj,
    )

    # update the JaxSimulation.input_structures in the sim_data_vjps using the adjoint returned vals
    sims_vjp_updated = []
    for sim_vjp, sim_data_vjp in zip(sims_vjp, batch_data_vjp):
        sim_vjp_orig = sim_data_vjp.simulation
        sim_vjp_updated = sim_vjp_orig.updated_copy(input_structures=sim_vjp.input_structures)
        sims_vjp_updated.append(sim_vjp_updated)

    return (sims_vjp_updated,)


def webapi_run_async_adjoint_fwd(
    simulations: Tuple[Simulation, ...],
    jax_infos: Tuple[JaxInfo, ...],
    folder_name: str,
    path_dir: str,
    callback_url: str,
    verbose: bool,
) -> Tuple[BatchData, Dict[str, str]]:
    """Runs the forward simulations on our servers, stores the gradient data for later."""
    task_names = [str(_task_name_orig(i)) for i in range(len(simulations))]

    simulations = dict(zip(task_names, simulations))
    jax_infos = dict(zip(task_names, jax_infos))

    batch = AdjointBatch(
        simulations=simulations,
        jax_infos=jax_infos,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        simulation_type="adjoint_fwd",
    )

    batch_data_orig = batch.run(path_dir=path_dir)

    return batch_data_orig, tuple(batch_data_orig.task_ids.values())


def webapi_run_async_adjoint_bwd(
    simulations: Tuple[Simulation, ...],
    jax_infos: Tuple[JaxInfo, ...],
    folder_name: str,
    path_dir: str,
    callback_url: str,
    verbose: bool,
    parent_tasks: List[List[str]],
) -> List[JaxSimulation]:
    """Runs the forward simulations on our servers, stores the gradient data for later."""

    task_names = [str(i) for i in range(len(simulations))]

    simulations = dict(zip(task_names, simulations))
    jax_infos = dict(zip(task_names, jax_infos))
    parent_tasks = [tuple(task_ids) for task_ids in parent_tasks]
    parent_tasks_dict = dict(zip(task_names, parent_tasks))

    batch = AdjointBatch(
        simulations=simulations,
        jax_infos=jax_infos,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        simulation_type="adjoint_bwd",
        parent_tasks=parent_tasks_dict,
    )

    batch.start()
    batch.monitor()
    sims_vjp = []
    for _, job in batch.jobs.items():
        task_id = job.task_id
        sim_vjp = download_sim_vjp(task_id=task_id, verbose=verbose)
        sims_vjp.append(sim_vjp)

    return sims_vjp


# register the custom forward and backward functions
run_async.defvjp(run_async_fwd, run_async_bwd)


""" Options to do the previous but all client side (mainly for testing / debugging)."""


@partial(custom_vjp, nondiff_argnums=tuple(range(1, 7)))
def run_local(
    simulation: JaxSimulation,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
    verbose: bool = True,
    num_proc: int = NUM_PROC_LOCAL,
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
    num_proc: int = 1
        Number of processes to use for the gradient computations.

    Returns
    -------
    :class:`.JaxSimulationData`
        Object containing solver results for the supplied :class:`.JaxSimulation`.
    """

    # convert to regular tidy3d (and accounting info)
    sim_tidy3d, jax_info = simulation.to_simulation()

    # run using regular tidy3d simulation running fn
    sim_data_tidy3d = tidy3d_run_fn(
        simulation=sim_tidy3d,
        task_name=str(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
    )

    # convert back to jax type and return
    # TODO: add task_id
    return JaxSimulationData.from_sim_data(sim_data_tidy3d, jax_info=jax_info)


def run_local_fwd(
    simulation: JaxSimulation,
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    verbose: bool,
    num_proc: int,
) -> Tuple[JaxSimulationData, tuple]:
    """Run forward pass and stash extra objects for the backwards pass."""

    # add the gradient monitors and run the forward simulation
    grad_mnts = simulation.get_grad_monitors(
        input_structures=simulation.input_structures, freqs_adjoint=simulation.freqs_adjoint
    )
    sim_fwd = simulation.updated_copy(**grad_mnts)
    sim_data_fwd = _run(
        simulation=sim_fwd,
        task_name=_task_name_fwd(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
    )

    # remove the gradient data from the returned version (not needed)
    sim_data_orig = sim_data_fwd.copy(update=dict(grad_data=(), simulation=simulation))
    return sim_data_orig, (sim_data_fwd,)


def run_local_bwd(
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    verbose: bool,
    num_proc: int,
    res: tuple,
    sim_data_vjp: JaxSimulationData,
) -> Tuple[JaxSimulation]:
    """Run backward pass and return simulation storing vjp of the objective w.r.t. the sim."""

    # grab the forward simulation and its gradient monitor data
    (sim_data_fwd,) = res
    grad_data_fwd = sim_data_fwd.grad_data_symmetry
    grad_eps_data_fwd = sim_data_fwd.grad_eps_data_symmetry

    # make and run adjoint simulation
    fwidth_adj = sim_data_fwd.simulation._fwidth_adjoint
    run_time_adj = sim_data_fwd.simulation._run_time_adjoint
    sim_adj = sim_data_vjp.make_adjoint_simulation(fwidth=fwidth_adj, run_time=run_time_adj)
    sim_data_adj = _run(
        simulation=sim_adj,
        task_name=_task_name_adj(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
    )

    sim_data_adj = sim_data_adj.normalize_adjoint_fields()

    grad_data_adj = sim_data_adj.grad_data_symmetry

    # get gradient and insert into the resulting simulation structure medium
    sim_vjp = sim_data_vjp.simulation.store_vjp(
        grad_data_fwd=grad_data_fwd,
        grad_data_adj=grad_data_adj,
        grad_eps_data=grad_eps_data_fwd,
        num_proc=num_proc,
    )

    return (sim_vjp,)


# register the custom forward and backward functions
run_local.defvjp(run_local_fwd, run_local_bwd)


""" Running a batch of simulations using web.run_async. """


def _task_name_orig_local(index: int, task_name_suffix: str = None):
    """Task name as function of index into simulations. Note: for original must be int."""
    if task_name_suffix is not None:
        return f"{index}{task_name_suffix}"
    return int(index)


@partial(custom_vjp, nondiff_argnums=tuple(range(1, 7)))
def run_async_local(
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
    This does the adjoint processing on the client side. So more data will be required for download.

    Returns
    ------
    Tuple[:class:`.JaxSimulationData`, ...]
        Contains the :class:`.JaxSimulationData` of each :class:`.JaxSimulation`.
    """

    simulations = {
        _task_name_orig_local(i, task_name_suffix): sim for i, sim in enumerate(simulations)
    }

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
        task_name = _task_name_orig_local(i, task_name_suffix)
        sim_data_tidy3d = batch_data_tidy3d[task_name]
        jax_info = jax_infos[str(task_name)]
        # TODO: add task_id
        jax_sim_data = JaxSimulationData.from_sim_data(sim_data_tidy3d, jax_info=jax_info)
        jax_batch_data.append(jax_sim_data)

    return jax_batch_data


def run_async_local_fwd(
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
        grad_mnts = simulation.get_grad_monitors(
            input_structures=simulation.input_structures, freqs_adjoint=simulation.freqs_adjoint
        )
        sim_fwd = simulation.updated_copy(**grad_mnts)
        sims_fwd.append(sim_fwd)

    batch_data_fwd = run_async_local(
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


def run_async_local_bwd(
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
        grad_data_fwd[i] = sim_data_fwd.grad_data_symmetry
        grad_eps_data_fwd[i] = sim_data_fwd.grad_eps_data_symmetry

    # make and run adjoint simulation
    sims_adj = []
    for i, sim_data_fwd in enumerate(batch_data_fwd):
        fwidth_adj = sim_data_fwd.simulation._fwidth_adjoint
        run_time_adj = sim_data_fwd.simulation._run_time_adjoint
        sim_data_vjp = batch_data_vjp[i]
        sim_adj = sim_data_vjp.make_adjoint_simulation(fwidth=fwidth_adj, run_time=run_time_adj)
        sims_adj.append(sim_adj)

    batch_data_adj = run_async_local(
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
        sim_data_adj = sim_data_adj.normalize_adjoint_fields()

        grad_data_fwd = sim_data_fwd.grad_data_symmetry
        grad_data_adj = sim_data_adj.grad_data_symmetry
        grad_data_eps_fwd = sim_data_fwd.grad_eps_data_symmetry

        sim_data_vjp = batch_data_vjp[i]
        sim_vjp = sim_data_vjp.simulation.store_vjp(grad_data_fwd, grad_data_adj, grad_data_eps_fwd)
        sims_vjp.append(sim_vjp)

    return (sims_vjp,)


# register the custom forward and backward functions
run_async_local.defvjp(run_async_local_fwd, run_async_local_bwd)
