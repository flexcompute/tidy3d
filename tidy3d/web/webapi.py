"""Provides lowest level, user-facing interface to server."""

import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Callable
from functools import wraps

from requests import HTTPError, ReadTimeout
from requests.exceptions import ConnectionError as ConnErr
from requests.exceptions import JSONDecodeError
from urllib3.exceptions import NewConnectionError

import pytz
from rich.console import Console
from rich.progress import Progress

from .environment import Env
from .simulation_task import SimulationTask, SIM_FILE_HDF5, Folder
from .task import TaskId, TaskInfo, ChargeType
from ..components.data.sim_data import SimulationData
from ..components.simulation import Simulation
from ..components.types import Literal
from ..log import log
from ..exceptions import WebError

# time between checking task status
REFRESH_TIME = 0.3

# time between checking run status
RUN_REFRESH_TIME = 1.0

# file names when uploading to S3
SIM_FILE_JSON = "simulation.json"

# number of seconds to keep re-trying connection before erroring
CONNECTION_RETRY_TIME = 30


def wait_for_connection(decorated_fn=None, wait_time_sec: float = CONNECTION_RETRY_TIME):
    """Causes function to ignore connection errors and retry for ``wait_time_sec`` secs."""

    def decorator(web_fn):
        """Decorator returned by @wait_for_connection()"""

        @wraps(web_fn)
        def web_fn_wrapped(*args, **kwargs):
            """Function to return including connection waiting."""
            time_start = time.time()
            warned_previously = False

            while (time.time() - time_start) < wait_time_sec:
                try:
                    return web_fn(*args, **kwargs)
                except (ConnErr, ConnectionError, NewConnectionError, ReadTimeout, JSONDecodeError):
                    if not warned_previously:
                        log.warning(f"No connection: Retrying for {wait_time_sec} seconds.")
                        warned_previously = True
                    time.sleep(REFRESH_TIME)

            raise WebError("No internet connection: giving up on connection waiting.")

        return web_fn_wrapped

    if decorated_fn:
        return decorator(decorated_fn)

    return decorator


@wait_for_connection
def run(  # pylint:disable=too-many-arguments
    simulation: Simulation,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
    verbose: bool = True,
    progress_callback_upload: Callable[[float], None] = None,
    progress_callback_download: Callable[[float], None] = None,
    solver_version: str = None,
    worker_group: str = None,
) -> SimulationData:
    """Submits a :class:`.Simulation` to server, starts running, monitors progress, downloads,
    and loads results as a :class:`.SimulationData` object.

    Parameters
    ----------
    simulation : :class:`.Simulation`
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
        If `True`, will print progressbars and status, otherwise, will run silently.
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
    :class:`.SimulationData`
        Object containing solver results for the supplied :class:`.Simulation`.
    """
    task_id = upload(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        progress_callback=progress_callback_upload,
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
def upload(  # pylint:disable=too-many-locals,too-many-arguments
    simulation: Simulation,
    task_name: str,
    folder_name: str = "default",
    callback_url: str = None,
    verbose: bool = True,
    progress_callback: Callable[[float], None] = None,
    simulation_type: str = "tidy3d",
    parent_tasks: List[str] = None,
    source_required: bool = True,
) -> TaskId:
    """Upload simulation to server, but do not start running :class:`.Simulation`.

    Parameters
    ----------
    simulation : :class:`.Simulation`
        Simulation to upload to server.
    task_name : str
        Name of task.
    folder_name : str
        Name of folder to store task on web UI
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    simulation_type : str
        Type of simulation being uploaded.
    parent_tasks : List[str]
        List of related task ids.
    source_required: bool = True
        If ``True``, simulations without sources will raise an error before being uploaded.

    Returns
    -------
    str
        Unique identifier of task on server.

    Note
    ----
    To start the simulation running, must call :meth:`start` after uploaded.
    """

    simulation.validate_pre_upload(source_required=source_required)
    log.debug("Creating task.")

    task = SimulationTask.create(
        simulation, task_name, folder_name, callback_url, simulation_type, parent_tasks
    )
    if verbose:
        console = Console()
        console.log(f"Created task '{task_name}' with task_id '{task.task_id}'.")
        url = f"https://tidy3d.simulation.cloud/workbench?taskId={task.task_id}"
        console.log(f"View task using web UI at [link={url}]'{url}'[/link].")
    task.upload_simulation(verbose=verbose, progress_callback=progress_callback)

    # log the url for the task in the web UI
    log.debug(
        f"{Env.current.website_endpoint}/folders/{task.folder.folder_id}/tasks/{task.task_id}"
    )
    return task.task_id


@wait_for_connection
def get_info(task_id: TaskId) -> TaskInfo:
    """Return information about a task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.

    Returns
    -------
    :class:`TaskInfo`
        Object containing information about status, size, credits of task.
    """
    task = SimulationTask.get(task_id)
    if not task:
        raise ValueError("Task not found.")
    return TaskInfo(**{"taskId": task.task_id, **task.dict()})


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
    task.submit(solver_version=solver_version, worker_group=worker_group)


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
        Average field intensity normlized to max value (1.0).
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
        raise WebError("Error running task!")
    return status


# pylint: disable=too-many-statements, too-many-locals, too-many-branches
def monitor(task_id: TaskId, verbose: bool = True) -> None:
    # pylint:disable=too-many-statements
    """Print the real time task progress until completion.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.

    Note
    ----
    To load results when finished, may call :meth:`load`.
    """

    task_info = get_info(task_id)
    task_name = task_info.taskName

    break_statuses = ("success", "error", "diverged", "deleted", "draft")

    console = Console() if verbose else None

    def get_estimated_cost() -> float:
        """Get estimated cost, if None, is not ready."""
        task_info = get_info(task_id)
        block_info = task_info.taskBlockInfo
        if block_info and block_info.chargeType == ChargeType.FREE:
            est_flex_unit = 0
            grid_points = block_info.maxGridPoints
            time_steps = block_info.maxTimeSteps
            if grid_points < 1000:
                grid_points_str = f"{grid_points}"
            elif 1000 <= grid_points < 1000 * 1000:
                grid_points_str = f"{grid_points / 1000}K"
            else:
                grid_points_str = f"{grid_points / 1000 / 1000}M"

            if time_steps < 1000:
                time_steps_str = f"{time_steps}"
            elif 1000 <= grid_points < 1000 * 1000:
                time_steps_str = f"{time_steps / 1000}K"
            else:
                time_steps_str = f"{time_steps / 1000 / 1000}M"

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
        with console.status(f"[bold green]Starting '{task_name}'...", spinner="runner"):
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
        console.log(
            "To cancel the simulation, use 'web.delete(task_id)' or delete the task in the web"
            " UI. Terminating the Python script will not stop the job running on the cloud."
        )
        with Progress(console=console) as progress:

            pbar_pd = progress.add_task("% done", total=100)
            perc_done, _ = get_run_info(task_id)

            while perc_done is not None and perc_done < 100 and get_status(task_id) == "running":
                perc_done, field_decay = get_run_info(task_id)
                new_description = f"solver progress (field decay = {field_decay:.2e})"
                progress.update(pbar_pd, completed=perc_done, description=new_description)
                time.sleep(RUN_REFRESH_TIME)

            perc_done, field_decay = get_run_info(task_id)
            if perc_done is not None and perc_done < 100 and field_decay > 0:
                console.log("early shutoff detected, exiting.")

            progress.update(pbar_pd, completed=100, refresh=True)

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
    """Download results of task and log to file.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation_data.hdf5"
        Download path to .hdf5 data file (including filename).
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    """
    task = SimulationTask(taskId=task_id)
    task.get_simulation_hdf5(path, verbose=verbose, progress_callback=progress_callback)


@wait_for_connection
def download_json(
    task_id: TaskId,
    path: str = SIM_FILE_JSON,
    verbose: bool = True,
    progress_callback: Callable[[float], None] = None,
) -> None:
    """Download the `.json` file associated with the :class:`.Simulation` of a given task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation.json"
        Download path to .json file of simulation (including filename).
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    """

    task = SimulationTask(taskId=task_id)
    task.get_simulation_json(path, verbose=verbose, progress_callback=progress_callback)


@wait_for_connection
def download_hdf5(
    task_id: TaskId,
    path: str = SIM_FILE_HDF5,
    verbose: bool = True,
    progress_callback: Callable[[float], None] = None,
) -> None:
    """Download the `.hdf5` file associated with the :class:`.Simulation` of a given task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation.hdf5"
        Download path to .hdf5 file of simulation (including filename).
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    """

    task = SimulationTask(taskId=task_id)
    task.get_simulation_hdf5(path, verbose=verbose, progress_callback=progress_callback)


@wait_for_connection
def load_simulation(
    task_id: TaskId,
    path: str = SIM_FILE_JSON,
    verbose: bool = True,
    progress_callback: Callable[[float], None] = None,
) -> Simulation:
    """Download the `.json` file of a task and load the associated :class:`.Simulation`.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation.json"
        Download path to .json file of simulation (including filename).
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    Returns
    -------
    :class:`.Simulation`
        Simulation loaded from downloaded json file.
    """

    # task = SimulationTask.get(task_id)
    task = SimulationTask(taskId=task_id)
    task.get_simulation_json(path, verbose=verbose, progress_callback=progress_callback)
    return Simulation.from_file(path)


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
        If `True`, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    Note
    ----
    To load downloaded results into data, call :meth:`load` with option `replace_existing=False`.
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
) -> SimulationData:
    """Download and Load simultion results into :class:`.SimulationData` object.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str
        Download path to .hdf5 data file (including filename).
    replace_existing: bool = True
        Downloads the data even if path exists (overwriting the existing).
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    Returns
    -------
    :class:`.SimulationData`
        Object containing simulation data.
    """

    if not os.path.exists(path) or replace_existing:
        download(task_id=task_id, path=path, verbose=verbose, progress_callback=progress_callback)

    if verbose:
        console = Console()
        console.log(f"loading SimulationData from {path}")

    sim_data = SimulationData.from_file(path)

    final_decay_value = sim_data.final_decay_value
    shutoff_value = sim_data.simulation.shutoff
    if (shutoff_value != 0) and (final_decay_value > shutoff_value):
        log.warning(
            f"Simulation final field decay value of {final_decay_value} "
            f"is greater than the simulation shutoff threshold of {shutoff_value}. "
            "Consider simulation again with large run_time duration for more accurate results."
        )

    return sim_data


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


# TODO: make this return a list of TaskInfo instead?
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
    folder = Folder.get(folder)
    tasks = folder.list_tasks()
    if order == "new":
        tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)
    elif order == "old":
        tasks = sorted(tasks, key=lambda t: t.created_at)
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return [task.dict() for task in tasks]


@wait_for_connection
def estimate_cost(task_id: str) -> float:
    """Compute the maximum FlexCredit charge for a given task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.

    Returns
    -------
    float
        Estimated maximum cost for :class:`.Simulation` associated with given ``task_id``..

    Note
    ----
    Cost is calculated assuming the simulation runs for
    the full ``run_time``. If early shut-off is triggered, the cost is adjusted proportionately.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.

    Returns
    -------
    float
        Estimated cost of the task in FlexCredits.
    """

    task = SimulationTask.get(task_id)
    if not task:
        raise ValueError("Task not found.")

    task.estimate_cost()
    task_info = get_info(task_id)
    status = task_info.metadataStatus

    # Wait for a termination status
    while status not in ["processed", "success", "error", "failed"]:
        time.sleep(REFRESH_TIME)
        task_info = get_info(task_id)
        status = task_info.metadataStatus

    if status in ["processed", "success"]:
        return task_info.estFlexUnit

    log.warning(
        "Could not get estimated cost! It will be reported during a simulation run in the "
        "preprocessing step."
    )
    return None


@wait_for_connection
def real_cost(task_id: str) -> float:
    """Get the billed cost for given task after it has been run.

    Note
    ----
        The billed cost may not be immediately available when the task status is set to ``success``,
        but should be available shortly after.
    """
    task_info = get_info(task_id)
    flex_unit = task_info.realFlexUnit
    if not flex_unit:
        log.warning(
            f"Billed FlexCredit for task '{task_id}' is not available. If the task has been "
            "successfully run, it should be available shortly."
        )
    return flex_unit


@wait_for_connection
def test() -> None:
    """Confirm whether Tidy3D authentication is configured. Raises exception if not."""
    try:
        # note, this is a little slow, but the only call that doesn't require providing a task id.
        get_tasks(num_tasks=0)
        console = Console()
        console.log("Authentication configured successfully!")
    except (WebError, HTTPError) as e:
        url = "https://docs.flexcompute.com/projects/tidy3d/en/latest/quickstart.html"

        raise WebError(
            "Tidy3D not configured correctly. Please refer to our documentation for installation "
            "instructions at "
            f"[link={url}]'{url}'[/link]."
        ) from e
