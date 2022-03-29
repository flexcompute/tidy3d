"""Provides lowest level, user-facing interface to server."""

import os
import time
import json
from typing import List, Dict, Optional
from datetime import datetime
import logging

import requests
from rich.console import Console
from rich.progress import Progress

from .config import DEFAULT_CONFIG as Config
from .s3utils import get_s3_user, DownloadProgress
from .task import TaskId, TaskInfo
from . import httputils as http
from ..components.simulation import Simulation
from ..components.data import SimulationData
from ..components.types import Literal
from ..log import log, WebError

REFRESH_TIME = 0.3
TOTAL_DOTS = 3


def run(  # pylint:disable=too-many-arguments
    simulation: Simulation,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
    normalize_index: Optional[int] = 0,
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
    normalize_index : int = 0
        If specified, normalizes the frequency-domain data by the amplitude spectrum of the source
        corresponding to ``simulation.sources[normalize_index]``.
        This occurs when the data is loaded into a :class:`.SimulationData` object.
        To turn off normalization, set ``normalize_index`` to ``None``.

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
    )
    start(task_id)
    monitor(task_id)
    return load(task_id=task_id, path=path, normalize_index=normalize_index)


def upload(
    simulation: Simulation, task_name: str, folder_name: str = "default", callback_url: str = None
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

    Returns
    -------
    TaskId
        Unique identifier of task on server.

    Note
    ----
    To start the simulation running, must call :meth:`start` after uploaded.
    """

    task_id = _upload_task(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        callback_url=callback_url,
    )

    # log the task_id so users can copy and paste it from STDOUT / file if the need it later.
    log.info(f"Uploaded task '{task_name}' with task_id '{task_id}'.")

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
    method = f"fdtd/task/{task_id}"
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

    Note
    ----
    To monitor progress, can call :meth:`monitor` after starting simulation.
    """
    task = get_info(task_id)
    folder_name = task.folderId
    method = f"fdtd/model/{folder_name}/task/{task_id}"
    task.status = "queued"
    http.put(method, data=task.dict())


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
        client, bucket, user_id = get_s3_user()
        key = f"users/{user_id}/{task_id}/output/solver_progress.csv"
        progress = client.get_object(Bucket=bucket, Key=key)["Body"]
        progress_string = progress.read().split(b"\n")
        perc_done, field_decay = progress_string[-2].split(b",")
        return float(perc_done), float(field_decay)
    except Exception:  # pylint:disable=broad-except
        return None, None


def monitor(task_id: TaskId) -> None:
    """Print the real time task progress until completion.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.

    Note
    ----
    To load results when finished, may call :meth:`load`.
    """

    task_info = get_info(task_id)
    task_name = task_info.taskName

    status = None

    break_statuses = ("success", "error", "diverged", "deleted", "draft")

    def get_status():
        """Get status for this task (called many times below, so put into function)."""
        status = get_info(task_id).status
        if status == "visualize":
            return "success"
        return status

    console = Console()

    # already done
    if get_status() in break_statuses:
        console.log(f"status = {get_status()}")
        return

    # preprocessing
    with console.status(f"[bold green]Starting '{task_name}'...", spinner="runner"):
        while get_status() not in break_statuses and get_status() != "running":
            if status != get_status():
                status = get_status()
                if status != "running":
                    console.log(f"status = {status}")
            time.sleep(REFRESH_TIME)

    # startup phase where run info is not available
    console.log("starting up solver")
    while get_run_info(task_id)[0] is None and get_status() == "running":
        time.sleep(REFRESH_TIME)

    # phase where run % info is available
    console.log("running solver")
    with Progress(console=console) as progress:
        pbar_pd = progress.add_task("% done", total=100)
        perc_done, _ = get_run_info(task_id)
        while perc_done is not None and perc_done < 100 and get_status() == "running":
            perc_done, field_decay = get_run_info(task_id)
            new_description = f"% done (field decay = {field_decay:.2e})"
            progress.update(pbar_pd, completed=perc_done, description=new_description)
            time.sleep(1.0)
        if perc_done is not None and perc_done < 100:
            console.log("early shutoff detected, exiting.")
        else:
            progress.update(pbar_pd, completed=100)

    # postprocessing
    with console.status(f"[bold green]Finishing '{task_name}'...", spinner="runner"):
        while get_status() not in break_statuses:
            if status != get_status():
                status = get_status()
                console.log(f"status = {status}")
            time.sleep(REFRESH_TIME)

    # final status (if diffrent from the last printed status)
    if status != get_status():
        console.log(f"status = {get_status()}")


def download(task_id: TaskId, path: str = "simulation_data.hdf5") -> None:
    """Download results of task and log to file.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    path : str = "simulation_data.hdf5"
        Download path to .hdf5 data file (including filename).

    Note
    ----
    To load downloaded results into data, call :meth:`load` with option `replace_existing=False`.
    """

    # TODO: it should be possible to load "diverged" simulations
    task_info = get_info(task_id)
    if task_info.status in ("error", "deleted", "draft"):
        raise WebError(f"can't download task '{task_id}', status = '{task_info.status}'")

    directory, _ = os.path.split(path)
    if directory != "":
        os.makedirs(directory, exist_ok=True)

    _download_file(task_id, fname="monitor_data.hdf5", path=path)


def load(
    task_id: TaskId,
    path: str = "simulation_data.hdf5",
    replace_existing: bool = True,
    normalize_index: Optional[int] = 0,
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
    normalize_index : int = 0
        If specified, normalizes the frequency-domain data by the amplitude spectrum of the source
        corresponding to ``simulation.sources[normalize_index]``.
        This occurs when the data is loaded into a :class:`.SimulationData` object.
        To turn off normalization, set ``normalize_index`` to ``None``.

    Returns
    -------
    :class:`.SimulationData`
        Object containing simulation data.
    """

    if not os.path.exists(path) or replace_existing:
        download(task_id=task_id, path=path)

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

    if normalize_index is not None:
        return sim_data.normalize(normalize_index=normalize_index)

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

    method = f"fdtd/task/{str(task_id)}"
    return http.delete(method)


def delete_old(days_old: int = 100, folder: str = None) -> int:
    """Delete all tasks older than a given amount of days.

    Parameters
    ----------
    days_old : int = 100
        Minimum number of days since the task creation.
    folder : str = None
        If None, all folders are purged.

    Returns
    -------
    int
        Total number of tasks deleted.
    """

    tasks = http.get("fdtd/models")
    count = 0
    for pfolder in tasks:
        if pfolder["name"] == folder or folder is None:
            for task in pfolder["children"]:
                if task["status"] == "deleted":
                    continue
                stime_str = task["submitTime"]
                stime = datetime.strptime(stime_str, "%Y:%m:%d:%H:%M:%S")
                days_elapsed = (datetime.utcnow() - stime).days
                if days_elapsed >= days_old:
                    delete(task["taskId"])
                    count += 1

    return count


def get_tasks(num_tasks: int = None, order: Literal["new", "old"] = "new") -> List[Dict]:
    """Get a list with the metadata of the last ``num_tasks`` tasks.

    Parameters
    ----------
    num_tasks : int = None
        The number of tasks to return, or, if ``None``, return all.
    order : Literal["new", "old"] = "new"
        Return the tasks in order of newest-first or oldest-first.
    """

    tasks = http.get("fdtd/models")
    store_dict = {
        "submit_time": [],
        "status": [],
        "task_name": [],
        "task_id": [],
    }
    for pfolder in tasks:
        for task in pfolder["children"]:
            try:
                store_dict["submit_time"].append(task["submitTime"])
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

    out_dict = []
    for ipr in range(num_tasks):
        out_dict.append({key: item[sort_inds[ipr]] for (key, item) in store_dict.items()})

    return out_dict


def _upload_task(  # pylint:disable=too-many-locals,too-many-arguments
    simulation: Simulation,
    task_name: str,
    folder_name: str = "default",
    solver_version: str = Config.solver_version,
    worker_group: str = Config.worker_group,
    callback_url: str = None,
) -> TaskId:
    """upload with all kwargs exposed"""

    simulation.validate_pre_upload()

    json_string = simulation._json_string()  # pylint:disable=protected-access
    data = {
        "status": "uploading",
        "solverVersion": solver_version,
        "taskName": task_name,
        "workerGroup": worker_group,
        "callbackUrl": callback_url,
    }

    method = f"fdtd/model/{folder_name}/task"

    log.debug("Creating task.")
    try:
        task = http.post(method=method, data=data)
        task_id = task["taskId"]
    except requests.exceptions.HTTPError as e:
        error_json = json.loads(e.response.text)
        raise WebError(error_json["error"]) from e

    # upload the file to s3
    log.debug("Uploading the json file")

    client, bucket, user_id = get_s3_user()

    key = f"users/{user_id}/{task_id}/simulation.json"

    # size_bytes = len(json_string.encode('utf-8'))
    # TODO: add progressbar, with put_object, no callback, so no real need.
    # with Progress() as progress:
    # upload_progress = UploadProgress(size_bytes, progress)
    log.debug(f"json = {json_string}")
    client.put_object(
        Body=json_string,
        Bucket=bucket,
        Key=key,
        # Callback=upload_progress.report
    )

    task["status"] = "draft"
    http.put(f"{method}/{task_id}", data=task)

    return task_id


def _download_file(task_id: TaskId, fname: str, path: str) -> None:
    """Download a specific file from server.

    Parameters
    ----------
    task_id : str
        Unique identifier of task on server.  Returned by :meth:`upload`.
    fname : str
        Name of the file on server (eg. ``monitor_data.hdf5``, ``tidy3d.log``, ``simulation.json``)
    path : str
        Path where the file will be downloaded to (including filename).
    """
    log.info(f'downloading file "{fname}" to "{path}"')

    try:
        client, bucket, user_id = get_s3_user()

        if fname in ("monitor_data.hdf5", "tidy3d.log"):
            key = f"users/{user_id}/{task_id}/output/{fname}"
        else:
            key = f"users/{user_id}/{task_id}/{fname}"

        head_object = client.head_object(Bucket=bucket, Key=key)
        size_bytes = head_object["ContentLength"]
        with Progress() as progress:
            download_progress = DownloadProgress(size_bytes, progress)
            client.download_file(
                Bucket=bucket, Filename=path, Key=key, Callback=download_progress.report
            )

    except Exception as e:  # pylint:disable=broad-except
        task_info = get_info(task_id)
        log.warning(e)
        log.error(
            "Cannot retrieve requested file, check the file name and "
            "make sure the task has run correctly. Current "
            f"task status is '{task_info.status}.",
        )


def _rm_file(path: str):
    """Clear path if it exists."""
    if os.path.exists(path) and not os.path.isdir(path):
        log.debug(f"removing file {path}")
        os.remove(path)
