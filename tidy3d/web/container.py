"""higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks."""

import os
from abc import ABC
from typing import Dict, Tuple

from . import webapi as web
from .task import TaskId, TaskInfo, RunInfo
from ..components.simulation import Simulation
from ..components.data import SimulationData
from ..components.base import Tidy3dBaseModel


class WebContainer(Tidy3dBaseModel, ABC):
    """base class for job and batch, technically not used"""


# type of task_name
TaskName = str


class Job(WebContainer):
    """Interface for managing the running of a ``Simulation`` on server."""

    simulation: Simulation
    task_name: TaskName
    task_id: TaskId = None

    def upload(self) -> None:
        """Upload simulation to server without running."""
        task_id = web.upload(simulation=self.simulation)
        self.task_id = task_id

    def get_info(self) -> TaskInfo:
        """Return information about a Job.

        Returns
        -------
        TaskInfo
            Object containing information about status, size, credits of task.
        """
        task_info = web.get_info(task_id=self.task_id)
        return task_info

    def run(self) -> None:
        """start running a task"""
        web.run(self.task_id)

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
        web.monitor(task_id=self.task_id)

    def download(self, path: str) -> None:
        """Download results of simulation.

        Parameters
        ----------
        path : str
            Download path to .hdf5 data file (including filename).
        """
        web.download(task_id=self.task_id, path=path)

    def load_results(self, path: str) -> SimulationData:
        """Download results from simulation (if not already) and load them into ``SimulationData``
        object.

        Parameters
        ----------
        path : str
            Download path to .hdf5 data file (including filename).

        Returns
        -------
        SimulationData
            Object containing data about simulation.
        """
        if not os.path.exists(path):
            self.download(path=path)
        sim_data = SimulationData.load(path)
        return sim_data

    def delete(self):
        """Delete server-side data associated with Job."""
        web.delete(self.task_id)
        self.task_id = None


class Batch(WebContainer):
    """Interface for managing the running of several ``Simulation`` on server.

    Attributes
    ----------
    jobs : TYPE
        Description
    """

    simulations: Dict[TaskName, Simulation]
    jobs: Dict[TaskName, Job] = None

    def __init__(self, **kwargs):
        """Create a batch of Jobs from dictionary of named ``Simulation`` objects.

        Parameters
        ----------
        **kwargs
            Description
        simulations (Dict[str, Simulation] : dictionary of task name and ``Simulation`` for each
        job.
        """
        jobs = kwargs.get("jobs")
        if jobs is None:
            jobs = {}
            for task_name, simulation in kwargs["simulations"].items():
                job = Job(simulation=simulation, task_name=task_name)
                jobs[task_name] = job
            kwargs["jobs"] = jobs
        super().__init__(**kwargs)

    def upload(self) -> None:
        """upload simulations to server, record task ids"""
        for _, job in self.jobs.items():
            job.upload()

    def get_info(self) -> Dict[TaskName, TaskInfo]:
        """get general information about all job's task

        Returns
        -------
        Dict[TaskName, TaskInfo]
            Description
        """
        info_dict = {}
        for task_name, job in self.jobs.items():
            task_info = job.get_info()
            info_dict[task_name] = task_info
        return info_dict

    def run(self) -> None:
        """start running a task"""
        for _, job in self.jobs.items():
            job.run()

    def get_run_info(self) -> Dict[TaskName, RunInfo]:
        """get information about a each of the tasks in batch.

        Returns
        -------
        Dict[TaskName, RunInfo]
            Description
        """
        run_info_dict = {}
        for task_name, job in self.jobs.items():
            run_info = job.get_run_info()
            run_info_dict[task_name] = run_info
        return run_info_dict

    def monitor(self) -> None:
        """monitor progress of each of the running tasks in batch."""
        for task_name, job in self.jobs.items():
            print(f"\nmonitoring task: {task_name}")
            job.monitor()

    @staticmethod
    def _job_data_path(task_id: TaskId, path_dir: str):
        """Default path to data of a single Job in Batch

        Parameters
        ----------
        task_id : TaskId
            Description
        path_dir : str
            Description

        Returns
        -------
        TYPE
            Description
        """
        return os.path.join(path_dir, f"{str(task_id)}.hdf5")

    def download(self, path_dir: str) -> None:
        """download results.

        Parameters
        ----------
        path_dir : str
            Description
        """
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_name, path_dir)
            job.download(path=job_path)

    def load_results(self, path_dir: str) -> Dict[TaskName, SimulationData]:
        """download results and load them into SimulationData object.

        Parameters
        ----------
        path_dir : str
            Base directory where data will be downloaded.

        Returns
        -------
        Dict[TaskName, SimulationData]
            Description
        """
        sim_data_dir = {}
        self.download(path_dir=path_dir)
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            sim_data = job.load_results(path=job_path)
            sim_data_dir[task_name] = sim_data
        return sim_data_dir

    def delete(self):
        """delete server-side data associated with job"""
        for _, job in self.jobs.items():
            job.delete()
            self.jobs = None

    def save(self, fname: str) -> None:
        """Save ``Batch`` information to file.

        Parameters
        ----------
        fname : str
            path to save ``Batch`` as .json file (including filename).
        """
        self.export(fname=fname)

    def items(self, path_dir: str) -> Tuple[TaskName, SimulationData]:
        """simple iterator, `for task_name, sim_data in batch: `do something`

        Parameters
        ----------
        path_dir : str
            Base directory where data will be downloaded.

        Yields
        ------
        Tuple[TaskName, SimulationData]
            Task name and Simulation data, returned one by one if iterated over.
        """
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            sim_data = job.load_results(path=job_path)
            yield task_name, sim_data
