"""Provides lowest level, user-facing interface to server."""

import os
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List

import pytz
from requests import HTTPError
from rich.progress import Progress

from ...components.types import Literal
from ...exceptions import WebError
from ...log import get_logging_console, log
from ..core.constants import SIM_FILE_HDF5, TaskId
from ..core.environment import Env
from ..core.task_core import Folder, SimulationTask
from ..core.task_info import ChargeType, TaskInfo
from .connect_util import (
    REFRESH_TIME,
    get_grid_points_str,
    get_time_steps_str,
    wait_for_connection,
)
from .tidy3d_stub import SimulationDataType, SimulationType, Tidy3dStub, Tidy3dStubData

# time between checking run status
RUN_REFRESH_TIME = 1.0

# file names when uploading to S3
SIM_FILE_JSON = "simulation.json"

# not all solvers are supported yet in GUI
GUI_SUPPORTED_TASK_TYPES = ["FDTD", "MODE_SOLVER", "HEAT"]

# if a solver is in beta stage, cost is subject to change
BETA_TASK_TYPES = ["HEAT", "EME"]

# map task_type to solver name for display
SOLVER_NAME = {"FDTD": "FDTD", "HEAT": "Heat", "MODE_SOLVER": "Mode", "EME": "EME"}


def _get_url(task_id: str) -> str:
    """Get the URL for a task on our server."""
    return f"{Env.current.website_endpoint}/workbench?taskId={task_id}"


@wait_for_connection
def run(
    simulation: SimulationType,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
    verbose: bool = True,
    progress_callback_upload: Callable[[float], None] = None,
    progress_callback_download: Callable[[float], None] = None,
    solver_version: str = None,
    worker_group: str = None,
    simulation_type: str = "tidy3d",
    parent_tasks: list[str] = None,
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
    task_id = upload(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        progress_callback=progress_callback_upload,
        simulation_type=simulation_type,
        parent_tasks=parent_tasks,
    )
    start(
        task_id,
        solver_version=solver_version,
        worker_group=worker_group,
    )
    monitor(task_id, verbose=verbose)
    return load(
        task_id=task_id, path=path, verbose=verbose, progress_callback=progress_callback_download
    )


@wait_for_connection
def upload(
    simulation: SimulationType,
    task_name: str,
    folder_name: str = "default",
    callback_url: str = None,
    verbose: bool = True,
    progress_callback: Callable[[float], None] = None,
    simulation_type: str = "tidy3d",
    parent_tasks: List[str] = None,
    source_required: bool = True,
) -> TaskId:
    """
    Upload simulation to server, but do not start running :class:`.Simulation`.

    Parameters
    ----------
    simulation : Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]
        Simulation to upload to server.
    task_name : str
        Name of task.
    folder_name : str
        Name of folder to store task on web UI
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    simulation_type : str = "tidy3d"
        Type of simulation being uploaded.
    parent_tasks : List[str]
        List of related task ids.
    source_required: bool = True
        If ``True``, simulations without sources will raise an error before being uploaded.

    Returns
    -------
    str
        Unique identifier of task on server.


    Notes
    -----

        Once you've created a ``job`` object using :class:`tidy3d.web.api.container.Job`, you can upload it to our servers with:

        .. code-block:: python

            web.upload(simulation, task_name="task_name", verbose=verbose)

        It will not run until you explicitly tell it to do so with :meth:`tidy3d.web.api.webapi.start`.

    """
    stub = Tidy3dStub(simulation=simulation)
    stub.validate_pre_upload(source_required=source_required)
    log.debug("Creating task.")

    task_type = stub.get_type()

    task = SimulationTask.create(
        task_type, task_name, folder_name, callback_url, simulation_type, parent_tasks, "Gz"
    )
    if verbose:
        console = get_logging_console()
        console.log(
            f"Created task '{task_name}' with task_id '{task.task_id}' and task_type '{task_type}'."
        )
        if task_type in BETA_TASK_TYPES:
            solver_name = SOLVER_NAME[task_type]
            console.log(
                f"Tidy3D's {solver_name} solver is currently in the beta stage. "
                f"Cost of {solver_name} simulations is subject to change in the future."
            )
        if task_type in GUI_SUPPORTED_TASK_TYPES:
            url = _get_url(task.task_id)
            console.log(f"View task using web UI at [link={url}]'{url}'[/link].")

    task.upload_simulation(stub=stub, verbose=verbose, progress_callback=progress_callback)

    # log the url for the task in the web UI
    log.debug(f"{Env.current.website_endpoint}/folders/{task.folder_id}/tasks/{task.task_id}")
    return task.task_id


@wait_for_connection
def get_info(task_id: TaskId, verbose: bool = True) -> TaskInfo:
    """Return information about a task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    Returns
    -------
    :class:`TaskInfo`
        Object containing information about status, size, credits of task.
    """
    task = SimulationTask.get(task_id, verbose)
    if not task:
        raise ValueError("Task not found.")
    return TaskInfo(**{"taskId": task.task_id, "taskType": task.task_type, **task.dict()})


@wait_for_connection
def start(
    task_id: TaskId,
    solver_version: str = None,
    worker_group: str = None,
) -> None:
    """Start running the simulation associated with task.

    Parameters
    ----------

    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    solver_version: str = None
        target solver version.
    worker_group: str = None
        worker group
    Note
    ----
    To monitor progress, can call :meth:`monitor` after starting simulation.
    """
    task = SimulationTask.get(task_id)
    if not task:
        raise ValueError("Task not found.")
    task.submit(
        solver_version=solver_version,
        worker_group=worker_group,
    )


@wait_for_connection
def get_run_info(task_id: TaskId):
    """Gets the % done and field_decay for a running task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.

    Returns
    -------
    perc_done : float
        Percentage of run done (in terms of max number of time steps).
        Is ``None`` if run info not available.
    field_decay : float
        Average field intensity normalized to max value (1.0).
        Is ``None`` if run info not available.
    """
    task = SimulationTask(taskId=task_id)
    return task.get_running_info()


def get_status(task_id) -> str:
    """Get the status of a task. Raises an error if status is "error".

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    """
    task_info = get_info(task_id)
    status = task_info.status
    if status == "visualize":
        return "success"
    if status == "error":
        raise WebError(
            f"Error running task {task_id}! Use 'web.download_log('{task_id}')' to "
            "download and examine the solver log, and/or contact customer support for help."
        )
    return status


def monitor(task_id: TaskId, verbose: bool = True) -> None:
    """
    Print the real time task progress until completion.

    Notes
    -----

        To monitor the simulation's progress and wait for its completion, use:

        .. code-block:: python

            tidy3d.web.api.webapi.monitor(job.task_id, verbose=verbose).

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.

    Note
    ----
    To load results when finished, may call :meth:`load`.
    """

    console = get_logging_console() if verbose else None

    task_info = get_info(task_id)

    task_name = task_info.taskName

    task_type = task_info.taskType

    break_statuses = ("success", "error", "diverged", "deleted", "draft", "abort")

    def get_estimated_cost() -> float:
        """Get estimated cost, if None, is not ready."""
        task_info = get_info(task_id)
        block_info = task_info.taskBlockInfo
        if block_info and block_info.chargeType == ChargeType.FREE:
            est_flex_unit = 0
            grid_points = block_info.maxGridPoints
            time_steps = block_info.maxTimeSteps
            grid_points_str = get_grid_points_str(grid_points)
            time_steps_str = get_time_steps_str(time_steps)
            console.log(
                f"You are running this simulation for FREE. Your current plan allows"
                f" up to {block_info.maxFreeCount} free non-concurrent simulations per"
                f" day (under {grid_points_str} grid points and {time_steps_str}"
                f" time steps)"
            )
        else:
            est_flex_unit = task_info.estFlexUnit
            if est_flex_unit is not None and est_flex_unit > 0:
                console.log(
                    f"Maximum FlexCredit cost: {est_flex_unit:1.3f}. Use 'web.real_cost(task_id)'"
                    f" to get the billed FlexCredit cost after a simulation run."
                )
        return est_flex_unit

    def monitor_preprocess() -> None:
        """Periodically check the status."""
        status = get_status(task_id)
        while status not in break_statuses and status != "running":
            new_status = get_status(task_id)
            if new_status != status:
                status = new_status
                if verbose and status != "running":
                    console.log(f"status = {status}")
            time.sleep(REFRESH_TIME)

    status = get_status(task_id)

    if verbose:
        console.log(f"status = {status}")

    # already done
    if status in break_statuses:
        return

    # preprocessing
    if verbose:
        console.log(
            "To cancel the simulation, use 'web.abort(task_id)' or 'web.delete(task_id)' "
            "or abort/delete the task in the web "
            "UI. Terminating the Python script will not stop the job running on the cloud."
        )
        with console.status(f"[bold green]Waiting for '{task_name}'...", spinner="runner"):
            monitor_preprocess()
    else:
        monitor_preprocess()

    # if the estimated cost is ready, print it
    if verbose:
        get_estimated_cost()
        console.log("starting up solver")

    # while running but before the percentage done is available, keep waiting
    while get_run_info(task_id)[0] is None and get_status(task_id) == "running":
        time.sleep(REFRESH_TIME)

    # while running but percentage done is available
    if verbose:
        # verbose case, update progressbar
        console.log("running solver")
        if task_type == "FDTD":
            with Progress(console=console) as progress:
                pbar_pd = progress.add_task("% done", total=100)
                perc_done, _ = get_run_info(task_id)

                while (
                    perc_done is not None and perc_done < 100 and get_status(task_id) == "running"
                ):
                    perc_done, field_decay = get_run_info(task_id)
                    new_description = f"solver progress (field decay = {field_decay:.2e})"
                    progress.update(pbar_pd, completed=perc_done, description=new_description)
                    time.sleep(RUN_REFRESH_TIME)

                perc_done, field_decay = get_run_info(task_id)
                if perc_done is not None and perc_done < 100 and field_decay > 0:
                    console.log(f"early shutoff detected at {perc_done:1.0f}%, exiting.")

                new_description = f"solver progress (field decay = {field_decay:.2e})"
                progress.update(pbar_pd, completed=100, refresh=True, description=new_description)
        elif task_type == "EME":
            with Progress(console=console) as progress:
                pbar_pd = progress.add_task("% done", total=100)
                perc_done, _ = get_run_info(task_id)

                while (
                    perc_done is not None and perc_done < 100 and get_status(task_id) == "running"
                ):
                    perc_done, _ = get_run_info(task_id)
                    new_description = "solver progress"
                    progress.update(pbar_pd, completed=perc_done, description=new_description)
                    time.sleep(RUN_REFRESH_TIME)

                perc_done, _ = get_run_info(task_id)
                new_description = "solver progress"
                progress.update(pbar_pd, completed=100, refresh=True, description=new_description)

    else:
        # non-verbose case, just keep checking until status is not running or perc_done >= 100
        perc_done, _ = get_run_info(task_id)
        while perc_done is not None and perc_done < 100 and get_status(task_id) == "running":
            perc_done, field_decay = get_run_info(task_id)
            time.sleep(1.0)

    # post processing
    if verbose:
        status = get_status(task_id)
        if status != "running":
            console.log(f"status = {status}")

        with console.status(f"[bold green]Finishing '{task_name}'...", spinner="runner"):
            while status not in break_statuses:
                new_status = get_status(task_id)
                if new_status != status:
                    status = new_status
                    console.log(f"status = {status}")
                time.sleep(REFRESH_TIME)

        if task_type in GUI_SUPPORTED_TASK_TYPES:
            url = _get_url(task_id)
            console.log(f"View simulation result at [blue underline][link={url}]'{url}'[/link].")
    else:
        while get_status(task_id) not in break_statuses:
            time.sleep(REFRESH_TIME)


@wait_for_connection
def download(
    task_id: TaskId,
    path: str = "simulation_data.hdf5",
    verbose: bool = True,
    progress_callback: Callable[[float], None] = None,
) -> None:
    """Download results of task to file.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation_data.hdf5"
        Download path to .hdf5 data file (including filename).
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    """
    task = SimulationTask(taskId=task_id)
    task.get_sim_data_hdf5(path, verbose=verbose, progress_callback=progress_callback)


@wait_for_connection
def download_json(task_id: TaskId, path: str = SIM_FILE_JSON, verbose: bool = True) -> None:
    """Download the ``.json`` file associated with the :class:`.Simulation` of a given task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation.json"
        Download path to .json file of simulation (including filename).
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.

    """

    task = SimulationTask(taskId=task_id)
    task.get_simulation_json(path, verbose=verbose)


@wait_for_connection
def download_hdf5(
    task_id: TaskId,
    path: str = SIM_FILE_HDF5,
    verbose: bool = True,
    progress_callback: Callable[[float], None] = None,
) -> None:
    """Download the ``.hdf5`` file associated with the :class:`.Simulation` of a given task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation.hdf5"
        Download path to .hdf5 file of simulation (including filename).
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    """
    task = SimulationTask(taskId=task_id)
    task.get_simulation_hdf5(path, verbose=verbose, progress_callback=progress_callback)


@wait_for_connection
def load_simulation(
    task_id: TaskId, path: str = SIM_FILE_JSON, verbose: bool = True
) -> SimulationType:
    """Download the ``.json`` file of a task and load the associated simulation.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation.json"
        Download path to .json file of simulation (including filename).
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.

    Returns
    -------
    Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]
        Simulation loaded from downloaded json file.
    """

    task = SimulationTask.get(task_id)
    task.get_simulation_json(path, verbose=verbose)
    return Tidy3dStub.from_file(path)


@wait_for_connection
def download_log(
    task_id: TaskId,
    path: str = "tidy3d.log",
    verbose: bool = True,
    progress_callback: Callable[[float], None] = None,
) -> None:
    """Download the tidy3d log file associated with a task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "tidy3d.log"
        Download path to log file (including filename).
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    Note
    ----
    To load downloaded results into data, call :meth:`load` with option ``replace_existing=False``.
    """
    task = SimulationTask(taskId=task_id)
    task.get_log(path, verbose=verbose, progress_callback=progress_callback)


@wait_for_connection
def load(
    task_id: TaskId,
    path: str = "simulation_data.hdf5",
    replace_existing: bool = True,
    verbose: bool = True,
    progress_callback: Callable[[float], None] = None,
) -> SimulationDataType:
    """
    Download and Load simulation results into :class:`.SimulationData` object.

    Notes
    -----

        After the simulation is complete, you can load the results into a :class:`.SimulationData` object by its
        ``task_id`` using:

        .. code-block:: python

            sim_data = web.load(task_id, path="outt/sim.hdf5", verbose=verbose)

        The :meth:`tidy3d.web.api.webapi.load` method is very convenient to load and postprocess results from simulations
        created using Tidy3D GUI.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str
        Download path to .hdf5 data file (including filename).
    replace_existing: bool = True
        Downloads the data even if path exists (overwriting the existing).
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    Returns
    -------
    Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
        Object containing simulation data.
    """
    if not os.path.exists(path) or replace_existing:
        download(task_id=task_id, path=path, verbose=verbose, progress_callback=progress_callback)

    if verbose:
        console = get_logging_console()
        console.log(f"loading simulation from {path}")

    stub_data = Tidy3dStubData.postprocess(path)
    return stub_data


@wait_for_connection
def delete(task_id: TaskId) -> TaskInfo:
    """Delete server-side data associated with task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.

    Returns
    -------
    TaskInfo
        Object containing information about status, size, credits of task.
    """

    # task = SimulationTask.get(task_id)
    task = SimulationTask(taskId=task_id)
    task.delete()
    return TaskInfo(**{"taskId": task.task_id, **task.dict()})


@wait_for_connection
def delete_old(
    days_old: int = 100,
    folder: str = "default",
) -> int:
    """Delete all tasks older than a given amount of days.

    Parameters
    ----------
    folder : str
        Only allowed to delete in one folder at a time.
    days_old : int = 100
        Minimum number of days since the task creation.

    Returns
    -------
    int
        Total number of tasks deleted.
    """

    folder = Folder.get(folder)
    if not folder:
        return 0
    tasks = folder.list_tasks()
    if not tasks:
        return 0
    tasks = list(
        filter(lambda t: t.created_at < datetime.now(pytz.utc) - timedelta(days=days_old), tasks)
    )
    for task in tasks:
        task.delete()
    return len(tasks)


@wait_for_connection
def abort(task_id: TaskId):
    """Abort server-side data associated with task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.

    Returns
    -------
    TaskInfo
        Object containing information about status, size, credits of task.
    """

    task = SimulationTask.get(task_id)
    # task = SimulationTask(taskId=task_id)
    task.abort()
    return TaskInfo(**{"taskId": task.task_id, **task.dict()})


@wait_for_connection
def get_tasks(
    num_tasks: int = None, order: Literal["new", "old"] = "new", folder: str = "default"
) -> List[Dict]:
    """Get a list with the metadata of the last ``num_tasks`` tasks.

    Parameters
    ----------
    num_tasks : int = None
        The number of tasks to return, or, if ``None``, return all.
    order : Literal["new", "old"] = "new"
        Return the tasks in order of newest-first or oldest-first.
    folder: str = "default"
        Folder from which to get the tasks.

    Returns
    -------
    List[Dict]
        List of dictionaries storing the information for each of the tasks last ``num_tasks`` tasks.
    """
    folder = Folder.get(folder, create=True)
    tasks = folder.list_tasks()
    if not tasks:
        return []
    if order == "new":
        tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)
    elif order == "old":
        tasks = sorted(tasks, key=lambda t: t.created_at)
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return [task.dict() for task in tasks]


@wait_for_connection
def estimate_cost(task_id: str, verbose: bool = True, solver_version: str = None) -> float:
    """Compute the maximum FlexCredit charge for a given task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    verbose : bool = True
        Whether to log the cost and helpful messages.
    solver_version : str = None
        Target solver version.

    Returns
    -------
    float
        Estimated maximum cost for :class:`.Simulation` associated with given ``task_id``.

    Note
    ----
    Cost is calculated assuming the simulation runs for
    the full ``run_time``. If early shut-off is triggered, the cost is adjusted proportionately.
    A minimum simulation cost may also apply, which depends on the task details.

    Notes
    -----

        We can get the cost estimate of running the task before actually running it. This prevents us from
        accidentally running large jobs that we set up by mistake. The estimated cost is the maximum cost
        corresponding to running all the time steps.

    Examples
    --------

    Basic example:

    .. code-block:: python

        # initializes job, puts task on server (but doesn't run it)
        job = web.Job(simulation=sim, task_name="job", verbose=verbose)

        # estimate the maximum cost
        estimated_cost = web.estimate_cost(job.task_id)

        print(f'The estimated maximum cost is {estimated_cost:.3f} Flex Credits.')

    """
    task = SimulationTask.get(task_id)
    if not task:
        raise ValueError("Task not found.")

    task.estimate_cost(solver_version=solver_version)
    task_info = get_info(task_id)
    status = task_info.metadataStatus

    # Wait for a termination status
    while status not in ["processed", "success", "error", "failed"]:
        time.sleep(REFRESH_TIME)
        task_info = get_info(task_id)
        status = task_info.metadataStatus

    if status in ["processed", "success"]:
        if verbose:
            console = get_logging_console()
            console.log(
                f"Maximum FlexCredit cost: {task_info.estFlexUnit:1.3f}. Minimum cost depends on "
                "task execution details. Use 'web.real_cost(task_id)' to get the billed FlexCredit "
                "cost after a simulation run."
            )
            fc_mode = task_info.estFlexCreditMode
            fc_post = task_info.estFlexCreditPostProcess
            if fc_mode:
                console.log(f"  {fc_mode:1.3f} FlexCredit of the total cost from mode solves.")
            if fc_post:
                console.log(f"  {fc_post:1.3f} FlexCredit of the total cost from post-processing.")
        return task_info.estFlexUnit

    log.warning(
        "Could not get estimated cost! It will be reported during a simulation run in the "
        "preprocessing step."
    )
    return None


@wait_for_connection
def real_cost(task_id: str, verbose=True) -> float:
    """Get the billed cost for given task after it has been run.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    verbose : bool = True
        Whether to log the cost and helpful messages.

    Returns
    -------
    float
        The flex credit cost that was billed for the given ``task_id``.

    Note
    ----
        The billed cost may not be immediately available when the task status is set to ``success``,
        but should be available shortly after.

    Examples
    --------

    To obtain the cost of a simulation, you can use the function ``tidy3d.web.real_cost(task_id)``. In the example
    below, a job is created, and its cost is estimated. After running the simulation, the real cost can be obtained.

    .. code-block:: python

        import time

        # initializes job, puts task on server (but doesn't run it)
        job = web.Job(simulation=sim, task_name="job", verbose=verbose)

        # estimate the maximum cost
        estimated_cost = web.estimate_cost(job.task_id)

        print(f'The estimated maximum cost is {estimated_cost:.3f} Flex Credits.')

        # Runs the simulation.
        sim_data = job.run(path="data/sim_data.hdf5")

        time.sleep(5)

        # Get the billed FlexCredit cost after a simulation run.
        cost = web.real_cost(job.task_id)
    """
    task_info = get_info(task_id)
    flex_unit = task_info.realFlexUnit
    ori_flex_unit = task_info.oriRealFlexUnit
    if not flex_unit:
        log.warning(
            f"Billed FlexCredit for task '{task_id}' is not available. If the task has been "
            "successfully run, it should be available shortly."
        )
    else:
        if verbose:
            console = get_logging_console()
            console.log(f"Billed flex credit cost: {flex_unit:1.3f}.")
            if flex_unit != ori_flex_unit and task_info.taskType == "FDTD":
                console.log(
                    "Note: the task cost pro-rated due to early shutoff was below the minimum "
                    "threshold, due to fast shutoff. Decreasing the simulation 'run_time' should "
                    "decrease the estimated, and correspondingly the billed cost of such tasks."
                )
    return flex_unit


@wait_for_connection
def test() -> None:
    """
    Confirm whether Tidy3D authentication is configured. Raises exception if not.
    """
    try:
        # note, this is a little slow, but the only call that doesn't require providing a task id.
        get_tasks(num_tasks=0)
        console = get_logging_console()
        console.log("Authentication configured successfully!")
    except (WebError, HTTPError) as e:
        url = "https://docs.flexcompute.com/projects/tidy3d/en/latest/index.html"

        raise WebError(
            "Tidy3D not configured correctly. Please refer to our documentation for installation "
            "instructions at "
            f"[blue underline][link={url}]'{url}'[/link]."
        ) from e
