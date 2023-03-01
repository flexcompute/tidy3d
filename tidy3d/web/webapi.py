"""Provides lowest level, user-facing interface to server."""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict
import tempfile

import requests
from dateutil import parser
from rich.console import Console
from rich.progress import Progress

from . import httputils as http
from .config import DEFAULT_CONFIG
from .s3utils import upload_string, get_s3_sts_token, download_file

from .s3utils import upload_file
from .task import TaskId, TaskInfo, Folder
from ..components.data.sim_data import SimulationData
from ..components.simulation import Simulation
from ..components.types import Literal
from ..log import log, WebError
from ..version import __version__

REFRESH_TIME = 0.3
TOTAL_DOTS = 3
# file names when uploading to S3
SIM_FILE_JSON = "simulation.json"
SIM_FILE_HDF5 = "simulation.hdf5"


def run(  # pylint:disable=too-many-arguments
    simulation: Simulation,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
    verbose: bool = True,
) -> SimulationData:
    """Submits a :class:`.Simulation` to server, starts running, monitors progress, downloads,
    and loads results as a :class:`.SimulationData` object.

    Parameters
    ----------
    simulation : :class:`.Simulation`
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
    :class:`.SimulationData`
        Object containing solver results for the supplied :class:`.Simulation`.
    """
    task_id = upload(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
    )
    start(task_id)
    monitor(task_id, verbose=verbose)
    return load(task_id=task_id, path=path, verbose=verbose)


def upload(  # pylint:disable=too-many-locals,too-many-arguments
    simulation: Simulation,
    task_name: str,
    folder_name: str = "default",
    callback_url: str = None,
    verbose: bool = True,
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

    Returns
    -------
    TaskId
        Unique identifier of task on server.

    Note
    ----
    To start the simulation running, must call :meth:`start` after uploaded.
    """

    simulation.validate_pre_upload()

    data = {
        "taskName": task_name,
        "callbackUrl": callback_url,
    }

    folder = _query_or_create_folder(folder_name)
    method = f"tidy3d/projects/{folder.projectId}/tasks"

    log.debug("Creating task.")
    try:
        task = http.post(method=method, data=data)
        task_id = task["taskId"]
    except requests.exceptions.HTTPError as e:
        error_json = json.loads(e.response.text)
        raise WebError(error_json["error"]) from e
    # log the task_id so users can copy and paste it from STDOUT / file if the need it later.
    if verbose:
        log.info(f"Created task '{task_name}' with task_id '{task_id}'.")

    # pylint:disable=protected-access
    upload_string(task_id, simulation._json_string, SIM_FILE_JSON, verbose=verbose)
    if len(simulation.custom_datasets) > 0:
        # Also upload hdf5 containing all data.
        # The temp file will be re-opened in `to_hdf5` which can cause an error on some systems
        # so we explicitly close it first.
        data_file = tempfile.NamedTemporaryFile()  # pylint:disable=consider-using-with
        data_file.close()
        simulation.to_hdf5(data_file.name)
        upload_file(task_id, data_file.name, SIM_FILE_HDF5, verbose=verbose)

    # log the url for the task in the web UI
    log.debug(f"{DEFAULT_CONFIG.website_endpoint}/folders/{folder.projectId}/tasks/{task_id}")

    return task_id


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
    method = f"tidy3d/tasks/{task_id}/detail"
    info_dict = http.get(method)
    if info_dict is None:
        raise WebError(f"task {task_id} not found, unable to load info.")
    return TaskInfo(**info_dict)


def start(task_id: TaskId) -> None:
    """Start running the simulation associated with task.

    Parameters
    ----------

    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    solver_version : str
        Supply or override a specific solver version to the task.
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.

    Note
    ----
    To monitor progress, can call :meth:`monitor` after starting simulation.
    """
    method = f"tidy3d/tasks/{task_id}/submit"
    data = {}

    # do not pass protocol version if mapping is missing or needs an override.
    if DEFAULT_CONFIG.solver_version:
        data["solverVersion"] = DEFAULT_CONFIG.solver_version
    else:
        data["protocolVersion"] = __version__

    if DEFAULT_CONFIG.worker_group:
        data["workerGroup"] = DEFAULT_CONFIG.worker_group

    http.post(method, data=data)


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
    try:
        token = get_s3_sts_token(task_id, "output/solver_progress.csv")
        client = token.get_client()
        progress = client.get_object(Bucket=token.get_bucket(), Key=token.get_s3_key())["Body"]
        progress_string = progress.read().split(b"\n")
        perc_done, field_decay = progress_string[-2].split(b",")[:2]
        return float(perc_done), float(field_decay)
    except Exception:  # pylint:disable=broad-except
        return None, None


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

    def get_status() -> str:
        """Get status for this task."""
        task_info = get_info(task_id)
        status = task_info.status
        if status == "visualize":
            return "success"
        if status == "error":
            raise WebError("Error running task!")
        return status

    def get_estimated_cost() -> float:
        """Get estimated cost, if None, is not ready."""
        task_info = get_info(task_id)
        return task_info.estFlexUnit

    def monitor_preprocess() -> None:
        """Periodically check the status."""
        status = get_status()
        while status not in break_statuses and status != "running":
            new_status = get_status()
            if new_status != status:
                status = new_status
                if verbose and status != "running":
                    log.info(f"status = {status}")
            time.sleep(REFRESH_TIME)

    status = get_status()

    if verbose:
        console = Console()
        log.info(f"status = {status}")

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
        est_flex_unit = get_estimated_cost()
        if est_flex_unit is not None and est_flex_unit > 0:
            log.info(
                f"Maximum FlexUnit cost: {est_flex_unit:1.3f}. Use 'web.real_cost(task_id)' to "
                "get the billed FlexUnit cost after a simulation run."
            )
        log.info("starting up solver")

    # while running but before the percentage done is available, keep waiting
    while get_run_info(task_id)[0] is None and get_status() == "running":
        time.sleep(REFRESH_TIME)

    # while running but percentage done is available
    if verbose:

        # verbose case, update progressbar
        log.info("running solver")
        with Progress(console=console) as progress:

            pbar_pd = progress.add_task("% done", total=100)
            perc_done, _ = get_run_info(task_id)

            while perc_done is not None and perc_done < 100 and get_status() == "running":
                perc_done, field_decay = get_run_info(task_id)
                new_description = f"% done (field decay = {field_decay:.2e})"
                progress.update(pbar_pd, completed=perc_done, description=new_description)
                time.sleep(1.0)

            if perc_done is not None and perc_done < 100:
                log.info("early shutoff detected, exiting.")

            progress.update(pbar_pd, completed=100, refresh=True)

    else:

        # non-verbose case, just keep checking until status is not running or perc_done >= 100
        perc_done, _ = get_run_info(task_id)
        while perc_done is not None and perc_done < 100 and get_status() == "running":
            perc_done, field_decay = get_run_info(task_id)
            time.sleep(1.0)

    # post processing
    if verbose:

        status = get_status()
        if status != "running":
            log.info(f"status = {status}")

        with console.status(f"[bold green]Finishing '{task_name}'...", spinner="runner"):
            while status not in break_statuses:
                new_status = get_status()
                if new_status != status:
                    status = new_status
                    log.info(f"status = {status}")
                time.sleep(REFRESH_TIME)
    else:
        while get_status() not in break_statuses:
            time.sleep(REFRESH_TIME)


def download(task_id: TaskId, path: str = "simulation_data.hdf5", verbose: bool = True) -> None:
    """Download results of task and log to file.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation_data.hdf5"
        Download path to .hdf5 data file (including filename).
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.

    """
    _download_file(task_id, fname="output/monitor_data.hdf5", path=path, verbose=verbose)


def download_json(task_id: TaskId, path: str = SIM_FILE_JSON, verbose: bool = True) -> None:
    """Download the `.json` file associated with the :class:`.Simulation` of a given task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation.json"
        Download path to .json file of simulation (including filename).
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.

    """
    _download_file(task_id, fname=SIM_FILE_JSON, path=path, verbose=verbose)


def download_hdf5(task_id: TaskId, path: str = SIM_FILE_HDF5, verbose: bool = True) -> None:
    """Download the `.hdf5` file associated with the :class:`.Simulation` of a given task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation.hdf5"
        Download path to .hdf5 file of simulation (including filename).
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.

    """
    _download_file(task_id, fname=SIM_FILE_HDF5, path=path, verbose=verbose)


def load_simulation(task_id: TaskId, path: str = SIM_FILE_JSON, verbose: bool = True) -> Simulation:
    """Download the `.json` file of a task and load the associated :class:`.Simulation`.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation.json"
        Download path to .json file of simulation (including filename).
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.

    Returns
    -------
    :class:`.Simulation`
        Simulation loaded from downloaded json file.
    """
    download_json(task_id, path=path, verbose=verbose)
    return Simulation.from_file(path)


def download_log(task_id: TaskId, path: str = "tidy3d.log", verbose: bool = True) -> None:
    """Download the tidy3d log file associated with a task.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "tidy3d.log"
        Download path to log file (including filename).

    Note
    ----
    To load downloaded results into data, call :meth:`load` with option `replace_existing=False`.
    """
    _download_file(task_id, fname="output/tidy3d.log", path=path, verbose=verbose)


def load(
    task_id: TaskId,
    path: str = "simulation_data.hdf5",
    replace_existing: bool = True,
    verbose: bool = True,
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

    Returns
    -------
    :class:`.SimulationData`
        Object containing simulation data.
    """

    if not os.path.exists(path) or replace_existing:
        download(task_id=task_id, path=path, verbose=verbose)

    if verbose:
        log.info(f"loading SimulationData from {path}")

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

    method = f"tidy3d/tasks/{str(task_id)}"
    return http.delete(method)


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

    folder = _query_or_create_folder(folder)
    tasks = http.get(f"tidy3d/projects/{folder.projectId}/tasks")
    count = 0
    for task in tasks:
        stime_str = task["createdAt"]
        stime = parser.parse(stime_str)
        cutoff_date = datetime.now(stime.tzinfo) - timedelta(days=days_old)
        if stime < cutoff_date:
            delete(task["taskId"])
            count += 1

    return count


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
    """
    folder = _query_or_create_folder(folder)
    tasks = http.get(f"tidy3d/projects/{folder.projectId}/tasks")
    store_dict = {
        "submit_time": [],
        "status": [],
        "task_name": [],
        "task_id": [],
    }
    for task in tasks:
        try:
            store_dict["submit_time"].append(task["createdAt"])
            store_dict["status"].append(task["status"])
            store_dict["task_name"].append(task["taskName"])
            store_dict["task_id"].append(task["taskId"])
        except KeyError:
            logging.warning(f"Error with task {task['taskId']}, skipping.")

    sort_inds = sorted(
        range(len(store_dict["submit_time"])),
        key=store_dict["submit_time"].__getitem__,
        reverse=order == "new",
    )

    if num_tasks is None or num_tasks > len(sort_inds):
        num_tasks = len(sort_inds)

    return [
        {key: item[sort_inds[ipr]] for (key, item) in store_dict.items()}
        for ipr in range(num_tasks)
    ]


def estimate_cost(task_id: str) -> float:
    """Compute the maximum flex unit charge for a given task, assuming the simulation runs for
    the full ``run_time``. If early shut-off is triggered, the cost is adjusted proporionately.
    """

    method = f"tidy3d/tasks/{task_id}/metadata"
    data = {}

    # do not pass protocol version if mapping is missing or needs an override.
    if DEFAULT_CONFIG.solver_version:
        data["solverVersion"] = DEFAULT_CONFIG.solver_version
    else:
        data["protocolVersion"] = __version__

    try:
        resp = http.post(method, data=data)
    except Exception:  # pylint:disable=broad-except
        log.warning(
            "Could not get estimated cost! It will be reported in preprocessing upon "
            "simulation run."
        )
        return None

    return resp.get("flex_unit")


def real_cost(task_id: str) -> float:
    """Get the billed cost for given task after it has been run.

    Note
    ----
        The billed cost may not be immediately available when the task status is set to ``success``,
        but should be available shortly after.
    """

    flex_unit = get_info(task_id).realFlexUnit
    if not flex_unit:
        log.warning(
            f"Billed FlexUnit for task '{task_id}'' is not available. If the task has been "
            "successfully run, it should be available shortly."
        )
    return flex_unit


def _query_or_create_folder(folder_name) -> Folder:
    log.debug("query folder")
    method = f"tidy3d/project?projectName={folder_name}"

    try:
        resp = http.get(method)
        if resp is None:
            log.debug("folder not found, create one.")
            method = "tidy3d/projects"
            resp = http.post(method, data={"projectName": folder_name})
        folder = Folder(**resp)
    except WebError as e:
        raise e
    except Exception as e:  # pylint:disable=broad-except
        raise WebError("Could not create task folder") from e

    return folder


def _download_file(task_id: TaskId, fname: str, path: str, verbose: bool = True) -> None:
    """Download a specific file from server.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    fname : str
        Name of the file on server (eg. ``monitor_data.hdf5``, ``tidy3d.log``, ``simulation.json``)
    path : str
        Path where the file will be downloaded to (including filename).
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.

    """

    task_info = get_info(task_id)
    if task_info.status in ("error", "deleted"):
        raise WebError(f"can't download task '{task_id}', status = '{task_info.status}'")

    directory, _ = os.path.split(path)
    if directory != "":
        os.makedirs(directory, exist_ok=True)

    if verbose:
        log.info(f'downloading file "{fname}" to "{path}"')

    try:
        download_file(task_id, fname, path, verbose=verbose)
    except WebError as e:
        raise e
    except Exception as e:  # pylint:disable=broad-except
        task_info = get_info(task_id)
        log.warning(str(e))
        log.error(
            "Cannot retrieve requested file, check the file name and "
            "make sure the task has run correctly. Current "
            f"task status is '{task_info.status}.'",
        )


def _rm_file(path: str):
    """Clear path if it exists."""
    if os.path.exists(path) and not os.path.isdir(path):
        log.debug(f"removing file {path}")
        os.remove(path)


def _s3_grant(task_id: TaskId):
    user = DEFAULT_CONFIG.user
    if "expiration" in user:
        exp = parser.parse(user["expiration"]).replace(tzinfo=None)
        # request new temporary credential when existing one expires in 5 minutes.
        if (exp - datetime.utcnow()).total_seconds() > 300:
            return

    method = f"tidy3d/s3upload/grant?resourceId={task_id}"
    resp = http.get(method)
    credentials = resp["userCredentials"]
    DEFAULT_CONFIG.user = {**DEFAULT_CONFIG.user, **credentials}
