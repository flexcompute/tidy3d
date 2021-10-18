"""higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks."""
import os
from abc import ABC
from typing import Dict, Generator
import time

from rich.console import Console
from rich.progress import Progress

from . import webapi as web
from .task import TaskId, TaskInfo, RunInfo, TaskName
from ..components.simulation import Simulation
from ..components.data import SimulationData
from ..components.base import Tidy3dBaseModel


DEFAULT_DATA_PATH = "simulation_data.hdf5"
DEFAULT_DATA_DIR = "."


class WebContainer(Tidy3dBaseModel, ABC):
    """base class for job and batch, technically not used"""


class Job(WebContainer):
    """Interface for managing the running of a :class:`Simulation` on server."""

    simulation: Simulation
    task_name: TaskName
    folder_name: str = "default"
    task_id: TaskId = None

    def run(self, path: str = DEFAULT_DATA_PATH) -> SimulationData:
        """run :class:`Job` all the way through and return data.

        Parameters
        ----------
        path_dir : str
            Base directory where data will be downloaded, by default current working directory.

        Returns
        -------
        ``{TaskName: SimulationData}``
            Dictionary mapping task name to :class:`SimulationData` for :class:`Job`.
        """

        self.upload()
        self.start()
        self.monitor()
        return self.load_data(path=path)

    def upload(self) -> None:
        """Upload simulation to server without running."""
        task_id = web.upload(
            simulation=self.simulation, task_name=self.task_name, folder_name=self.folder_name
        )
        self.task_id = task_id

    def get_info(self) -> TaskInfo:
        """Return information about a :class:`Job`.

        Returns
        -------
        ``TaskInfo``
            :class:`TaskInfo` object containing info about status, size, credits of task and others.
        """
        task_info = web.get_info(task_id=self.task_id)
        return task_info

    @property
    def status(self):
        """return current status"""
        return self.get_info().status

    def start(self) -> None:
        """start running a task"""
        web.start(self.task_id)

    def get_run_info(self) -> RunInfo:
        """Return information about the running ``Job``.

        Returns
        -------
        RunInfo
            Task run information.
        """
        run_info = web.get_run_info(task_id=self.task_id)
        return run_info

    def monitor(self) -> None:
        """monitor progress of running ``Job``."""

        status = self.status
        console = Console()

        with console.status(f"[bold green]Working on '{self.task_name}'...") as _:

            while status not in ("success", "error", "diverged", "deleted", "draft"):
                new_status = self.status
                if new_status != status:
                    console.log(f"status = {new_status}")
                    status = new_status
                time.sleep(web.REFRESH_TIME)

    def download(self, path: str = DEFAULT_DATA_PATH) -> None:
        """Download results of simulation.

        Parameters
        ----------
        path : ``str``
            Path to download data as ``.hdf5`` file (including filename).
        """
        web.download(task_id=self.task_id, simulation=self.simulation, path=path)

    def load_data(self, path: str = DEFAULT_DATA_PATH) -> SimulationData:
        """Download results from simulation (if not already) and load them into ``SimulationData``
        object.

        Parameters
        ----------
        path : str
            Path to download data as ``.hdf5`` file (including filename).

        Returns
        -------
        :class:`SimulationData`
            Object containing data about simulation.
        """
        return web.load_data(task_id=self.task_id, simulation=self.simulation, path=path)

    def delete(self):
        """Delete server-side data associated with :class:`Job`."""
        web.delete(self.task_id)
        self.task_id = None


class Batch(WebContainer):
    """Interface for submitting several :class:`Simulation` objects to sever.

    Parameters
    ----------
    simulations : ``{str: :class:`Simulation`}``
        Mapping of task name to :class:`Simulation` objects.
    folder_name : ``str``, optional
        Name of folder to store member of each batch on web UI.
    """

    simulations: Dict[TaskName, Simulation]
    jobs: Dict[TaskName, Job] = None
    folder_name: str = "default"

    def run(self, path_dir: str = DEFAULT_DATA_DIR):
        """Run each :class:`Job` in :class:`Batch` all the way through and return iterator for data.

        Parameters
        ----------
        path_dir : ``str``
            Base directory where data will be downloaded, by default current working directory.

        Yields
        ------
        ``(TaskName, SimulationData)``
            Task name and Simulation data, returned one by one if iterated over.
        """

        self.upload()
        self.start()
        self.monitor()
        self.download(path_dir=path_dir)
        return self.items()

    def upload(self) -> None:
        """create jobs and upload to server"""
        self.jobs = {}
        for task_name, simulation in self.simulations.items():
            job = Job(simulation=simulation, task_name=task_name, folder_name=self.folder_name)
            self.jobs[task_name] = job
            job.upload()

    def get_info(self) -> Dict[TaskName, TaskInfo]:
        """get general information about all job's task

        Returns
        -------
        ``{str: :class:`TaskInfo`}``
            Description
        """
        info_dict = {}
        for task_name, job in self.jobs.items():
            task_info = job.get_info()
            info_dict[task_name] = task_info
        return info_dict

    def start(self) -> None:
        """start running a task"""
        for _, job in self.jobs.items():
            job.start()

    def get_run_info(self) -> Dict[TaskName, RunInfo]:
        """get information about a each of the tasks in batch.

        Returns
        -------
        ``{str: RunInfo}``
            Dictionary of task name to dictionary of run info for each task.
        """
        run_info_dict = {}
        for task_name, job in self.jobs.items():
            run_info = job.get_run_info()
            run_info_dict[task_name] = run_info
        return run_info_dict

    def monitor(self) -> None:
        """monitor progress of each of the running tasks in batch."""

        def pbar_description(task_name: str, status: str) -> str:
            return f"{task_name}: status = {status}"

        run_statuses = ["queued", "preprocess", "running", "postprocess", "visualize", "success"]
        end_statuses = ("success", "error", "diverged", "deleted", "draft")

        pbar_tasks = {}
        with Progress() as progress:
            for task_name, job in self.jobs.items():
                status = job.status
                description = pbar_description(task_name, status)
                pbar = progress.add_task(description, total=len(run_statuses) - 1)
                pbar_tasks[task_name] = pbar

            statuses = [job.status for _, job in self.jobs.items()]
            while any(s not in end_statuses for s in statuses):
                for status_index, (task_name, job) in enumerate(self.jobs.items()):
                    current_status = statuses[status_index]
                    new_status = job.status
                    if new_status != current_status:
                        completed = run_statuses.index(new_status)
                        pbar = pbar_tasks[task_name]
                        description = pbar_description(task_name, new_status)
                        progress.update(pbar, description=description, completed=completed)
                        statuses[status_index] = new_status
                statuses = [job.status for _, job in self.jobs.items()]
                time.sleep(web.REFRESH_TIME)

            for task_name, job in self.jobs.items():
                pbar = pbar_tasks[task_name]
                description = pbar_description(task_name, job.status)
                progress.update(pbar, description=description, completed=len(run_statuses))

    @staticmethod
    def _job_data_path(task_id: TaskId, path_dir: str = DEFAULT_DATA_DIR):
        """Default path to data of a single Job in Batch

        Parameters
        ----------
        task_id : ``TaskId``
            task_id corresponding to a :class:`Job`
        path_dir : ``str``, optional
            Base directory where data will be downloaded, by default current working directory.

        Returns
        -------
        str
            path of the data file
        """
        return os.path.join(path_dir, f"{str(task_id)}.hdf5")

    def download(self, path_dir: str = DEFAULT_DATA_DIR) -> None:
        """download results.

        Parameters
        ----------
        path_dir : ``str``
            Base directory where data will be downloaded, by default current working directory.
        """
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_name, path_dir)
            job.download(path=job_path)

    def load_data(self, path_dir: str = DEFAULT_DATA_DIR) -> Dict[TaskName, SimulationData]:
        """download results and load them into SimulationData object.
        Note: this will return a dictionary of :class:`SimulationData` objects, each of which can
        hold a large amount of data.
        Use `Batch.items()` to instead loop through :class:`SimulationData` objects and only store
        current iteration in memory if many simulations or large amounts of data.

        Parameters
        ----------
        path_dir : str
            Base directory where data will be downloaded, by default current working directory.

        Returns
        -------
        ``{TaskName: SimulationData}``
            Dictionary mapping task name to :class:`SimulationData` for :class:`Job`.
        """
        sim_data_dir = {}
        self.download(path_dir=path_dir)
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            sim_data = job.load_data(path=job_path)
            sim_data_dir[task_name] = sim_data
        return sim_data_dir

    def delete(self):
        """delete server-side data associated with job"""
        for _, job in self.jobs.items():
            job.delete()
            self.jobs = None

    def items(self, path_dir: str = DEFAULT_DATA_DIR) -> Generator:
        """simple iterator, ``for task_name, sim_data in batch.items(): do something``

        Parameters
        ----------
        path_dir : ``str``
            Base directory where data will be downloaded, by default current working directory.

        Yields
        ------
        ``(TaskName, SimulationData)``
            Task name and Simulation data, returned one by one if iterated over.
        """
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            sim_data = job.load_data(path=job_path)
            yield task_name, sim_data
