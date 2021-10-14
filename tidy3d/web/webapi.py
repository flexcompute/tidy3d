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
from . import httputils as http
from .s3utils import get_s3_client, DownloadProgress, UploadProgress
from .task import TaskId, Task, TaskInfo, RunInfo, TaskStatus

from ..components.simulation import Simulation
from ..components.data import SimulationData
from ..log import log

sys.path.append("../../")
from tidy3d_core.convert import export_old_json, load_old_monitor_data

""" webapi functions """


MONITOR_MESSAGE = {
    TaskStatus.INIT: "task hasnt been run, start with `web.run(task)`",
    TaskStatus.SUCCESS: "task finished succesfully, download with `web.download(task, path)`",
    TaskStatus.ERROR: "task errored",
}


def _upload(
    simulation: Simulation,
    task_name: str,
    folder_name: str = "default",
    solver_version: str = Config.solver_version,
    worker_group: str = Config.worker_group,
) -> TaskId:
    """upload with all kwargs exposed"""

    sim_dict = export_old_json(simulation)
    json_string = json.dumps(sim_dict)

    method = os.path.join("fdtd/model", folder_name, "task")
    data = {
        "status": "draft",
        "solverVersion": solver_version,
        "taskName": task_name,
        "nodeSize": 10,  # int(sim_dict["parameters"]["nodes"]),
        "timeSteps": 80,  # int(sim_dict["parameters"]["time_steps"]),
        "computeWeight": 1,  # float(sim_dict["parameters"]["compute_weight"]),
        "workerGroup": worker_group,
    }

    try:
        task = http.post(method=method, data=data)

    except requests.exceptions.HTTPError as e:
        error_json = json.loads(e.response.text)
        log.error(error_json["error"])

    task_id = task["taskId"]

    # upload the file to s3
    log.info("Uploading the json file...")

    key = os.path.join("users", Config.user["UserId"], task_id, "simulation.json")

    client = get_s3_client()
    client.put_object(
        Body=json_string,
        Bucket=Config.studio_bucket,
        Key=key,
    )

    return task_id


def upload(simulation: Simulation, task_name: str, folder_name: str = "default") -> TaskId:
    """upload simulation to server (as draft, dont run).

    Parameters
    ----------
    simulation : :class:`Simulation`
        Simulation to upload to server.
    task_name : ``str``
        name of task


    Returns
    -------
    TaskId
        Unique identifier of task on server.
    """
    return _upload(simulation=simulation, task_name=task_name, folder_name=folder_name)


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


def get_run_info(task_id: TaskId) -> RunInfo:
    """get information about running status of task

    Parameters
    ----------
    task_id : TaskId
        Description

    Returns
    -------
    RunInfo
        Description
    """
    # task = get_task_by_id(task_id)
    # call server
    # return make_fake_run_info(task_id)


def run(task_id: TaskId) -> None:
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


def get_run_progress(task_id: TaskId):
    """gets the % done and field_decay for a running task"""

    client = get_s3_client()
    bucket = Config.studio_bucket
    keys = Config.user
    user_id = keys["UserId"]
    key = os.path.join("users", user_id, task_id, "output", "solver_progress.csv")
    progress = client.get_object(Bucket=bucket, Key=key)["Body"]
    progress_string = progress.read().split(b"\n")
    perc_done, field_decay = progress_string[-2].split(b",")
    return RunInfo(perc_done=perc_done, field_decay=field_decay)


def monitor(task_id: TaskId, refresh_time: float = 0.5) -> None:
    """Print the real time task progress until completion.

    Parameters
    ----------
    task_id : TaskId
        Unique identifier of task on server.
    refresh_time : float
        seconds between checking status
    """

    task_info = get_info(task_id)
    task_name = task_info.taskName
    perc_done = 0.0
    field_decay = 1.0
    status = ""

    # to do: toggle console / display on or off, might want off for Job / Batch to override
    console = Console()
    with Progress() as progress:

        def get_description(status: str) -> str:
            """ gets the progressbar description as a function of status """
            base = f"[purple]Monitoring task:  "
            if status:
                return base + f"status='{status}'"
            return base

        pbar_running = progress.add_task(f"[purple]Working on task: '{task_name}'", total=100.0)

        while status not in ("success", "error", "diverged", "deleted", "draft"):

            new_status = get_info(task_id).status
            if new_status != status:
                progress.update(pbar_running, description=get_description(new_status))
                status = new_status
            time.sleep(refresh_time)

            # isn't able to get status right now for some reason
            if new_status == "running":
                try:
                    run_info = get_run_progress(task_id)
                    perc_done_new = run_info.perc_done
                    field_decay_new = run_info.field_decay
                except Exception as e:
                    perc_done_new = perc_done + 1
                    field_decay_new = field_decay - 0.01
                    # need to fix, get the status earlier!
                progress.update(pbar_running, advance=perc_done_new - perc_done)
                perc_done = perc_done_new
                field_decay = field_decay_new



def _download_file(task_id: TaskId, fname: str, path: str) -> None:
    """Download a specific file ``fname`` to ``path``.

    Parameters
    ----------
    task_id : TaskId
        Unique identifier of task on server.
    fname : str
        Name of file on server.
    path : str
        Path where the file will be downloaded to (including filename).
    """
    try:
        user_id = Config.user["UserId"]
        if fname in ("monitor_data.hdf5", "tidy3d.log"):
            key = os.path.join("users", user_id, task_id, "output", fname)
        else:
            key = os.path.join("users", user_id, task_id, fname)
        client = get_s3_client()
        bucket = Config.studio_bucket
        with Progress() as progress:
            download_progress = DownloadProgress(client, bucket, key, progress)
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
        os.remove(path)


def download(task_id: TaskId, simulation: Simulation, path: str = "simulation_data.hdf5") -> None:
    """Fownload results of task to file.

    Parameters
    ----------
    task_id : TaskId
        Unique identifier of task on server.
    path : str
        Download path to .hdf5 data file (including filename).
    """

    directory, _ = os.path.split(path)
    sim_file = os.path.join(directory, "simulation.json")
    mon_file = os.path.join(directory, "monitor_data.hdf5")

    log.info(sim_file)
    log.info(mon_file)

    for _path in (sim_file, mon_file, path):
        _rm_file(_path)

    _download_file(task_id, fname="simulation.json", path=sim_file)

    # note: we cant load this old simulation file, so we'll ask for it as input instead.
    # simulation = Simulation.load(sim_file)

    _download_file(task_id, fname="monitor_data.hdf5", path=mon_file)
    mon_data_dict = load_old_monitor_data(simulation=simulation, data_file=mon_file)
    sim_data = SimulationData(simulation=simulation, monitor_data=mon_data_dict)
    sim_data.export(path)
    os.remove(sim_file)
    os.remove(mon_file)


def load(task_id: TaskId, simulation: Simulation, path: str) -> SimulationData:
    """Download and Load simultion results into ``SimulationData`` object.

    Parameters
    ----------
    task_id : TaskId
        Unique identifier of task on server.
    path : str
        Download path to .hdf5 data file (including filename).

    Returns
    -------
    SimulationData
        Object containing simulation data.
    """
    if not os.path.exists(path):
        download(task_id=task_id, simulation=simulation, path=path)
    return SimulationData.load(path)


def _rm(path: str):
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)


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
