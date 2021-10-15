"""Provides lowest level, user-facing interface to server."""

import os
import sys
import time
from shutil import copyfile
import json

import numpy as np
import requests
from rich.console import Console
from rich.progress import Progress

from .config import DEFAULT_CONFIG as Config
from .s3utils import get_s3_user, DownloadProgress, UploadProgress
from .task import TaskId, Task, TaskInfo, RunInfo, TaskStatus
from . import httputils as http
from ..components.simulation import Simulation
from ..components.data import SimulationData
from ..log import log

sys.path.append("../../")
from tidy3d_core.convert import export_old_json, load_old_monitor_data
from tidy3d_core.postprocess import load_solver_results


TOTAL_DOTS = 3

""" webapi functions """

def run(simulation: Simulation, task_name: str, folder_name: str = "default", refresh_time: float = 0.3, path: str = "simulation_data.hdf5") -> SimulationData:
    """submits simulation to server, starts running, monitors progress, downloads and loads resultse.

    Parameters
    ----------
    simulation : :class:`Simulation`
        Simulation to upload to server.
    task_name : ``str``
        Name of task
    path : ``str``
        Path to download results file (.hdf5), including filename.
    folder_name : ``str``
        Name of folder to store task on web UI
    refresh_time : ``float``
        seconds between updating progress monitor

    Returns
    -------
    :class:`SimulationData`
        Object containing solver results for the supplied :class:`Simulation`.
    """
    task_id = upload(simulation=simulation, task_name=task_name, folder_name=folder_name)
    start(task_id)
    monitor(task_id, refresh_time=refresh_time)
    return load(task_id=task_id, simulation=Simulation, path=path)


def upload(simulation: Simulation, task_name: str, folder_name: str = "default") -> TaskId:
    """upload simulation to server (as draft, dont run).

    Parameters
    ----------
    simulation : :class:`Simulation`
        Simulation to upload to server.
    task_name : ``str``
        name of task
    folder_name : ``str``
        name of folder to store task on web UI


    Returns
    -------
    TaskId
        Unique identifier of task on server.
    """
    return _upload_task(simulation=simulation, task_name=task_name, folder_name=folder_name)


def get_info(task_id: TaskId) -> TaskInfo:
    """Return information about a task.

    Parameters
    ----------
    task_id : TaskId
        Unique identifier of task on server.

    Returns
    -------
    TaskInfo
        Object containing information about status, size, credits of task.
    """
    method = os.path.join("fdtd/task", task_id)
    info_dict = http.get(method)
    return TaskInfo(**info_dict)


def start(task_id: TaskId) -> None:
    """Start running the simulation associated with task.

    Parameters
    ----------
    task_id : TaskId
        Unique identifier of task on server.
    """
    task = get_info(task_id)
    folder_name = task.folderId
    method = os.path.join("fdtd/model", folder_name, "task", task_id)
    task.status = "queued"
    http.put(method, data=task.dict())


def get_run_info(task_id: TaskId):
    """gets the % done and field_decay for a running task

    Parameters
    ----------
    task_id : TaskId
        Unique identifier of task on server.

    Returns
    -------
    perc_done : float
        Percentage of run done (in terms of max number of time steps)
    field_decay : float
        Average field intensity normlized to max value (1.0).
    """

    client, bucket, user_id = get_s3_user()
    key = os.path.join("users", user_id, task_id, "output", "solver_progress.csv")
    progress = client.get_object(Bucket=bucket, Key=key)["Body"]
    progress_string = progress.read().split(b"\n")
    perc_done, field_decay = progress_string[-2].split(b",")
    return float(perc_done), float(field_decay)


def monitor(task_id: TaskId, refresh_time: float = 0.3) -> None:
    """Print the real time task progress until completion.

    Parameters
    ----------
    task_id : ``TaskId``
        Unique identifier of task on server.
    refresh_time : ``float``
        seconds between updating monitor
    """

    task_info = get_info(task_id)
    task_name = task_info.taskName
    perc_done = 0.0
    field_decay = 1.0
    status = ""

    # to do: toggle console / display on or off, might want off for Job / Batch to override
    with Progress() as progress:

        def get_description(status: str, num_dots=0) -> str:
            """ gets the progressbar description as a function of status """
            dot_string = ''.join([' ' if i >= num_dots else '.' for i in range(TOTAL_DOTS)])
            base = f"[purple]Monitoring task{dot_string}  "
            if status:
                return base + f"status='{status}'"
            return base

        pbar = progress.add_task(f"[purple]Working on task: '{task_name}'", total=100.0)
        num_dots = 0

        while status not in ("success", "error", "diverged", "deleted", "draft"):

            new_status = get_info(task_id).status
            if new_status != status:
                progress.update(pbar, description=get_description(new_status, num_dots))
                status = new_status
            time.sleep(refresh_time)
            num_dots = (num_dots + 1) % (TOTAL_DOTS + 1)
            progress.update(pbar, description=get_description(status, num_dots))

            if new_status in ("running", ):

                # try getting new percentage, if not available, just make some up
                try:
                    perc_done_new, field_decay_new = get_run_info(task_id)

                # TODO: get the perc right away so we dont have to handle this.
                except Exception as e:
                    perc_done_new = perc_done
                    field_decay_new = field_decay

                # advance the progressbar
                progress.update(pbar, description=get_description(new_status, num_dots), advance=perc_done_new - perc_done)
                perc_done = perc_done_new
                field_decay = field_decay_new


def download(task_id: TaskId, simulation: Simulation, path: str = "simulation_data.hdf5") -> None:
    """Fownload results of task and log to file.

    Parameters
    ----------
    task_id : TaskId
        Unique identifier of task on server.
    path : str
        Download path to .hdf5 data file (including filename).
    """

    task_info = get_info(task_id)
    if task_info.status in ("error", "diverged", "deleted", "draft"):
        log.error(f"can't download task '{task_id}', status = '{task_info.status}'")
        raise RuntimeError

    directory, _ = os.path.split(path)
    sim_file = os.path.join(directory, "simulation.json")
    mon_file = os.path.join(directory, "monitor_data.hdf5")
    log_file = os.path.join(directory, "tidy3d.log")

    log.info('clearing existing files before downloading')
    for _path in (sim_file, mon_file, path):
        _rm_file(_path)

    # TODO: load these server side using the new specifictaions
    _download_file(task_id, fname="simulation.json", path=sim_file)
    _download_file(task_id, fname="monitor_data.hdf5", path=mon_file)

    # TODO: do this stuff server-side
    log.info('getting log string')
    _download_file(task_id, fname="tidy3d.log", path=log_file)
    with open(log_file, "r") as f:
        log_string = f.read()


    log.info('loading old monitor data to data dict')
    # TODO: we cant convert old simulation file to new, so we'll ask for original as input instead.
    # simulation = Simulation.load(sim_file)
    mon_data_dict = load_old_monitor_data(simulation=simulation, data_file=mon_file)

    log.info('creating SimulationData from monitor data dict')
    sim_data = load_solver_results(simulation=simulation, solver_data_dict=mon_data_dict, log_string=log_string, task_info=task_info)

    log.info(f'exporting SimulationData to {path}')
    sim_data.export(path)

    log.info('clearing extraneous files')
    _rm_file(sim_file)
    _rm_file(mon_file)
    _rm_file(log_file)


def load(task_id: TaskId, simulation: Simulation, path: str = "simulation_data.hdf5") -> SimulationData:
    """Download and Load simultion results into ``SimulationData`` object.

    Parameters
    ----------
    task_id : ``TaskId``
        Unique identifier of task on server.
    path : ``str``
        Download path to .hdf5 data file (including filename).

    Returns
    -------
    :class:`SimulationData`
        Object containing simulation data.
    """
    if not os.path.exists(path):
        download(task_id=task_id, simulation=simulation, path=path)

    log.info(f'loading SimulationData from {path}')
    return SimulationData.load(path)


def delete(task_id: TaskId) -> TaskInfo:
    """Delete server-side data associated with task.

    Parameters
    ----------
    task_id : TaskId
        Unique identifier of task on server.

    Returns
    -------
    TaskInfo
        Object containing information about status, size, credits of task.
    """

    method = os.path.join("fdtd", "task", str(task_id))
    return http.delete(method)

def _upload_task(
    simulation: Simulation,
    task_name: str,
    folder_name: str = "default",
    solver_version: str = Config.solver_version,
    worker_group: str = Config.worker_group,
) -> TaskId:
    """upload with all kwargs exposed"""

    # convert to old json and get string version
    sim_dict = export_old_json(simulation)
    json_string = json.dumps(sim_dict)

    # TODO: remove node size, time steps, compute weight, worker group
    data = {
        "status": "draft",
        "solverVersion": solver_version,
        "taskName": task_name,
        "nodeSize": 10,  # int(sim_dict["parameters"]["nodes"]),
        "timeSteps": 80,  # int(sim_dict["parameters"]["time_steps"]),
        "computeWeight": 1,  # float(sim_dict["parameters"]["compute_weight"]),
        "workerGroup": worker_group,
    }

    method = os.path.join("fdtd/model", folder_name, "task")

    log.info(f"Creating task.")
    try:
        task = http.post(method=method, data=data)
        task_id = task["taskId"]
    except requests.exceptions.HTTPError as e:
        error_json = json.loads(e.response.text)
        log.error(error_json["error"])

    # upload the file to s3
    log.info("Uploading the json file")


    client, bucket, user_id = get_s3_user()

    key = os.path.join("users", user_id, task_id, "simulation.json")

    # size_bytes = len(json_string.encode('utf-8'))
    # TODO: add progressbar, with put_object, no callback, so no real need.
    # with Progress() as progress:
        # upload_progress = UploadProgress(size_bytes, progress)
    client.put_object(
        Body=json_string,
        Bucket=bucket,
        Key=key,
        # Callback=upload_progress.report
    )

    return task_id

def _download_file(task_id: TaskId, fname: str, path: str) -> None:
    """Download a specific file ``fname`` to ``path``.

    Parameters
    ----------
    task_id : ``TaskId``
        Unique identifier of task on server.
    fname : ``str``
        Name of the file on server (eg. ``monitor_data.hdf5``, ``tidy3d.log``, ``simulation.json``)
    path : ``str``
        Path where the file will be downloaded to (including filename).
    """
    log.info(f'downloading file "{fname}" to "{path}"')

    try:
        client, bucket, user_id = get_s3_user()

        if fname in ("monitor_data.hdf5", "tidy3d.log"):
            key = os.path.join("users", user_id, task_id, "output", fname)
        else:
            key = os.path.join("users", user_id, task_id, fname)

        head_object = client.head_object(Bucket=bucket, Key=key)
        size_bytes = head_object["ContentLength"]
        with Progress() as progress:
            download_progress = DownloadProgress(size_bytes, progress)
            client.download_file(
                Bucket=bucket, Filename=path, Key=key, Callback=download_progress.report
            )

    except Exception as e:
        task_info = get_info(task_id)
        log.warning(e)
        log.error(
            "Cannot retrieve requested file, check the file name and "
            "make sure the project has run correctly. Current "
            f"project status is '{task_info.status}.",
        )


def _rm_file(path: str):
    """clear path if it exists"""
    if os.path.exists(path) and not os.path.isdir(path):
        log.info(f'removing file {path}')
        os.remove(path)
