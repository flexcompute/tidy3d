""" Provides lowest level, user-facing interface to server """

import os
import sys
import time
from shutil import copyfile

sys.path.append("../../")

import tidy3d_core as tdcore
import numpy as np

from ..components.simulation import Simulation
from .task import TaskId, Task, TaskInfo, RunInfo, TaskStatus

""" filesystem emulation for tests """

SERVER_DIR = "tests/tmp/server"


def server_path(fname):
    """gets path to fname in server dir"""
    return os.path.join(SERVER_DIR, fname)


CLIENT_DIR = "tests/tmp/client"


def client_path(fname):
    """gets path to fname in client dir"""
    return os.path.join(CLIENT_DIR, fname)


def make_fake_task_id(num_letters=4):
    """constructs a fake task_id with `num_letters` letters"""
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    task_str_list = [np.random.choice(alphabet) for _ in range(num_letters)]
    task_str = "".join(task_str_list)
    return f"task_{task_str}"


def make_fake_task() -> Task:
    """make a fake task"""
    task_id = make_fake_task_id()
    task_info = make_fake_info()
    return Task(id=task_id, info=task_info)


def make_fake_info() -> TaskInfo:
    """make fake task info"""
    return TaskInfo(
        task_id=make_fake_task_id(),
        status=TaskStatus.INIT,
        size_bytes=1e4 * np.random.random(),
        credits=100 * np.random.random(),
    )


def make_fake_run_info(task_id: TaskId) -> RunInfo:
    """make fake run info"""
    print(task_id)
    return RunInfo(
        perc_done=100 * np.random.random(),
        field_decay=1 * np.random.random(),
    )


# global variable maps TaskID -> Task
TASKS = {}


def get_task_by_id(task_id: TaskId) -> Task:
    """look up Task by task_id in TASKS"""
    task = TASKS.get(task_id)
    assert task is not None, f"task_id {task_id} not found"
    return task


def _get_sim_path(task_id: TaskId):
    """get path to simulation file on server"""
    return server_path(f"sim_{task_id}.json")


def _get_data_path_server(task_id: TaskId):
    """get path to data file on server"""
    return server_path(f"sim_{task_id}.hdf5")


def _get_data_path_client(task_id: TaskId):
    """get path to data file on client"""
    return client_path(f"sim_{task_id}.hdf5")


""" webapi functions """


def upload(simulation: Simulation) -> TaskId:
    """upload simulation to server (as draft, dont run)."""

    # create the task
    task = make_fake_task()
    task.info.status = TaskStatus.INIT
    task_id = task.id

    # export simulation json
    sim_path = _get_sim_path(task_id)
    simulation.export(sim_path)

    # add task to 'server' and return id
    TASKS[task_id] = task
    return task_id


def get_info(task_id: TaskId) -> TaskInfo:
    """get information about task (status, size, credits, etc)."""
    task = get_task_by_id(task_id)
    # call server
    return task.info


def get_run_info(task_id: TaskId) -> RunInfo:
    """get information about running status of task"""
    # task = get_task_by_id(task_id)
    # call server
    return make_fake_run_info(task_id)


def run(task_id: TaskId) -> None:
    """start running the task."""
    task = get_task_by_id(task_id)
    task.info.status = TaskStatus.RUN

    # emulate a solve

    # load json file
    sim_path = _get_sim_path(task_id)
    sim_core = Simulation.load(sim_path)

    # get raw results in dict of dict of np.ndarray
    solver_data_dict = tdcore.solve(sim_core)

    # load these results as SimulationData server side if you want
    # sim_data_core = tdcore.load_solver_results(sim_core, solver_data_dict)

    # or, download these results to hdf5 file
    data_path = _get_data_path_server(task_id)
    tdcore.save_solver_results(data_path, sim_core, solver_data_dict)

    task.info.status = TaskStatus.SUCCESS


MONITOR_MESSAGE = {
    TaskStatus.INIT: "task hasnt been run, start with `web.run(task)`",
    TaskStatus.SUCCESS: "task finished succesfully, download with `web.download(task, path)`",
    TaskStatus.ERROR: "task errored",
}


def _print_status(task_id: TaskId) -> None:
    status = get_info(task_id).status
    print(f'status = "{status.value}"')


def monitor(task_id: TaskId) -> None:
    """monitor the task progress (% done, time step, total time step, field_decay)."""

    # emulate running the task
    task = get_task_by_id(task_id)
    task.info.status = TaskStatus.RUN

    while True:

        status = get_info(task_id).status

        _print_status(task_id)
        if status in (TaskStatus.SUCCESS, TaskStatus.ERROR):
            print("-> returning")
            return

        task.info.status = TaskStatus.QUEUE
        _print_status(task_id)
        task.info.status = TaskStatus.PRE
        _print_status(task_id)

        task.info.status = TaskStatus.RUN
        num_steps = 4
        for i in range(num_steps):
            run_info = RunInfo(
                perc_done=(i + 1) / num_steps,
                field_decay=np.exp(-i / num_steps),
            )
            _print_status(task_id)
            run_info.display()
            time.sleep(0.1 / num_steps)

        task.info.status = TaskStatus.POST
        _print_status(task_id)

        # emulate completing the task
        task.info.status = TaskStatus.SUCCESS


def download(task_id: TaskId, path: str) -> None:
    """download results of simulation run to client side"""

    # load the file into SimulationData
    data_path_server = _get_data_path_server(task_id)
    copyfile(data_path_server, path)


def _rm(path: str):
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)


def delete(task_id: TaskId) -> None:
    """delete data associated with task_id from server"""

    # remove from server directories
    sim_path = _get_sim_path(task_id)
    data_path = _get_data_path_server(task_id)
    _rm(sim_path)
    _rm(data_path)

    # remove task from emulated server queue if still there
    if task_id in TASKS:
        TASKS.pop(task_id)
