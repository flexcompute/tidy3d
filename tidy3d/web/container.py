"""higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks."""
from __future__ import annotations

import os
from abc import ABC
from typing import Dict, Tuple
import time

from rich.console import Console
from rich.progress import Progress
import pydantic as pd

from . import webapi as web
from .task import TaskId, TaskInfo, RunInfo, TaskName
from ..components.simulation import Simulation
from ..components.base import Tidy3dBaseModel
from ..components.data.sim_data import SimulationData

from ..log import DataError


DEFAULT_DATA_PATH = "simulation_data.hdf5"
DEFAULT_DATA_DIR = "."


class WebContainer(Tidy3dBaseModel, ABC):
    """Base class for :class:`Job` and :class:`Batch`, technically not used"""


class Job(WebContainer):
    """Interface for managing the running of a :class:`.Simulation` on server."""

    simulation: Simulation = pd.Field(
        ..., title="Simulation", description="Simulation to run as a 'task'."
    )

    task_name: TaskName = pd.Field(..., title="Task Name", description="Unique name of the task.")

    folder_name: str = pd.Field(
        "default", title="Folder Name", description="Name of folder to store task on web UI."
    )

    callback_url: str = pd.Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event. "
        "The body content is a json file with fields "
        "``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.",
    )

    verbose: bool = pd.Field(
        True, title="Verbose", description="Whether to print info messages and progressbars."
    )

    task_id: TaskId = pd.Field(
        None,
        title="Task Id",
        description="Task ID number, set when the task is uploaded, leave as None.",
    )

    def run(self, path: str = DEFAULT_DATA_PATH) -> SimulationData:
        """run :class:`Job` all the way through and return data.

        Parameters
        ----------
        path_dir : str = "./simulation_data.hdf5"
            Base directory where data will be downloaded, by default current working directory.

        Returns
        -------
        Dict[str: :class:`.SimulationData`]
            Dictionary mapping task name to :class:`.SimulationData` for :class:`Job`.
        """

        self.start()
        self.monitor()
        return self.load(path=path)

    @pd.validator("task_id", always=True)
    def _upload(cls, val, values) -> None:
        """Upload simulation to server without running."""
        if val is not None:
            return val

        task_id = web.upload(
            simulation=values.get("simulation"),
            task_name=values.get("task_name"),
            folder_name=values.get("folder_name"),
            callback_url=values.get("callback_url"),
            verbose=values.get("verbose"),
        )
        return task_id

    def get_info(self) -> TaskInfo:
        """Return information about a :class:`Job`.

        Returns
        -------
        :class:`TaskInfo`
            :class:`TaskInfo` object containing info about status, size, credits of task and others.
        """

        return web.get_info(task_id=self.task_id)

    @property
    def status(self):
        """Return current status of :class:`Job`."""
        return self.get_info().status

    def start(self) -> None:
        """Start running a :class:`Job`.

        Note
        ----
        To monitor progress of the :class:`Job`, call :meth:`Job.monitor` after started.
        """
        web.start(self.task_id)

    def get_run_info(self) -> RunInfo:
        """Return information about the running :class:`Job`.

        Returns
        -------
        :class:`RunInfo`
            Task run information.
        """
        return web.get_run_info(task_id=self.task_id)

    def monitor(self) -> None:
        """Monitor progress of running :class:`Job`.

        Note
        ----
        To load the output of completed simulation into :class:`.SimulationData`objets,
        call :meth:`Job.load`.
        """
        web.monitor(self.task_id, verbose=self.verbose)

    def download(self, path: str = DEFAULT_DATA_PATH) -> None:
        """Download results of simulation.

        Parameters
        ----------
        path : str = "./simulation_data.hdf5"
            Path to download data as ``.hdf5`` file (including filename).

        Note
        ----
        To load the data into :class:`.SimulationData`objets, can call :meth:`Job.load`.
        """
        web.download(task_id=self.task_id, path=path, verbose=self.verbose)

    def load(self, path: str = DEFAULT_DATA_PATH) -> SimulationData:
        """Download results from simulation (if not already) and load them into ``SimulationData``
        object.

        Parameters
        ----------
        path : str = "./simulation_data.hdf5"
            Path to download data as ``.hdf5`` file (including filename).

        Returns
        -------
        :class:`.SimulationData`
            Object containing data about simulation.
        """
        return web.load(task_id=self.task_id, path=path, verbose=self.verbose)

    def delete(self):
        """Delete server-side data associated with :class:`Job`."""
        web.delete(self.task_id)


class BatchData(Tidy3dBaseModel):
    """Holds a collection of :class:`.SimulationData` returned by :class:`.Batch`."""

    task_paths: Dict[TaskName, str] = pd.Field(
        ...,
        title="Data Paths",
        description="Mapping of task_name to path to corresponding data for each task in batch.",
    )

    task_ids: Dict[TaskName, str] = pd.Field(
        ..., title="Task IDs", description="Mapping of task_name to task_id for each task in batch."
    )

    verbose: bool = pd.Field(
        True, title="Verbose", description="Whether to print info messages and progressbars."
    )

    def load_sim_data(self, task_name: str) -> SimulationData:
        """Load a :class:`.SimulationData` from file by task name."""
        task_data_path = self.task_paths[task_name]
        task_id = self.task_ids[task_name]
        return web.load(
            task_id=task_id,
            path=task_data_path,
            replace_existing=False,
            verbose=self.verbose,
        )

    def items(self) -> Tuple[TaskName, SimulationData]:
        """Iterate through the :class:`.SimulationData` for each task_name."""
        for task_name in self.task_paths.keys():
            yield task_name, self.load_sim_data(task_name)

    def __getitem__(self, task_name: TaskName) -> SimulationData:
        """Get the :class:`.SimulationData` for a given ``task_name``."""
        return self.load_sim_data(task_name)

    @classmethod
    def load(cls, path_dir: str = DEFAULT_DATA_DIR) -> BatchData:
        """Load :class:`Batch` from file, download results, and load them.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where data will be downloaded, by default current working directory.
            A `batch.hdf5` file must be present in the directory.

        Returns
        ------
        :class:`BatchData`
            Contains the :class:`.SimulationData` of each :class:`.Simulation` in :class:`Batch`.
        """

        batch_file = Batch._batch_path(path_dir=path_dir)  # pylint:disable=protected-access
        batch = Batch.from_file(batch_file)
        return batch.load(path_dir=path_dir)


class Batch(WebContainer):
    """Interface for submitting several :class:`.Simulation` objects to sever."""

    simulations: Dict[TaskName, Simulation] = pd.Field(
        ...,
        title="Simulations",
        description="Mapping of task names to Simulations to run as a batch.",
    )

    folder_name: str = pd.Field(
        "default",
        title="Folder Name",
        description="Name of folder to store member of each batch on web UI.",
    )

    verbose: bool = pd.Field(
        True, title="Verbose", description="Whether to print info messages and progressbars."
    )

    jobs: Dict[TaskName, Job] = pd.Field(
        None,
        title="Simulations",
        description="Mapping of task names to individual Job object for each task in the batch. "
        "Set by ``Batch.upload``, leave as None.",
    )

    @staticmethod
    def _check_path_dir(path_dir: str) -> None:
        """Make sure ``path_dir`` exists and create one if not."""

        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)

    def run(self, path_dir: str = DEFAULT_DATA_DIR) -> BatchData:
        """Upload and run each simulation in :class:`Batch`.

        Parameters
        ----------
        path_dir : str
            Base directory where data will be downloaded, by default current working directory.

        Returns
        ------
        :class:`BatchData`
            Contains the :class:`.SimulationData` of each :class:`.Simulation` in :class:`Batch`.

        Note
        ----
        A typical usage might look like:

        >>> batch_data = batch.run()
        >>> for task_name, sim_data in batch_data.items():
        ...     # do something with data.

        ``bach_data`` does not store all of the :class:`.SimulationData` objects in memory,
        rather it iterates over the task names
        and loads the corresponding :class:`.SimulationData` from file one by one.
        If no file exists for that task, it downloads it.
        """
        self._check_path_dir(path_dir)
        self.start()
        self.monitor()
        return self.load(path_dir=path_dir)

    @pd.validator("jobs", always=True)
    def _upload(cls, val, values) -> None:
        """Create a series of tasks in the :class:`.Batch` and upload them to server.

        Note
        ----
        To start the simulations running, must call :meth:`Batch.start` after uploaded.
        """
        if val is not None:
            return val

        verbose = bool(values.get("verbose"))

        jobs = {}
        for task_name, simulation in values.get("simulations").items():
            job = Job(
                simulation=simulation,
                task_name=task_name,
                folder_name=values.get("folder_name"),
                verbose=verbose,
            )
            jobs[task_name] = job
        return jobs

    def get_info(self) -> Dict[TaskName, TaskInfo]:
        """Get information about each task in the :class:`Batch`.

        Returns
        -------
        Dict[str, :class:`TaskInfo`]
            Mapping of task name to data about task associated with each task.
        """
        info_dict = {}
        for task_name, job in self.jobs.items():
            task_info = job.get_info()
            info_dict[task_name] = task_info
        return info_dict

    def start(self) -> None:
        """Start running all tasks in the :class:`Batch`.

        Note
        ----
        To monitor the running simulations, can call :meth:`Batch.monitor`.
        """
        for _, job in self.jobs.items():
            job.start()

    def get_run_info(self) -> Dict[TaskName, RunInfo]:
        """get information about a each of the tasks in the :class:`Batch`.

        Returns
        -------
        Dict[str: :class:`RunInfo`]
            Maps task names to run info for each task in the :class:`Batch`.
        """
        run_info_dict = {}
        for task_name, job in self.jobs.items():
            run_info = job.get_run_info()
            run_info_dict[task_name] = run_info
        return run_info_dict

    def monitor(self) -> None:  # pylint:disable=too-many-locals
        """Monitor progress of each of the running tasks.

        Note
        ----
        To loop through the data of completed simulations, can call :meth:`Batch.items`.
        """

        def pbar_description(task_name: str, status: str) -> str:
            return f"{task_name}: status = {status}"

        run_statuses = [
            "draft",
            "queued",
            "preprocess",
            "queued_solver",
            "running",
            "postprocess",
            "visualize",
            "success",
        ]
        end_statuses = ("success", "error", "diverged", "deleted", "draft")

        if self.verbose:
            console = Console()
            console.log("Started working on Batch.")

            with Progress(console=console) as progress:

                # create progressbars
                pbar_tasks = {}
                for task_name, job in self.jobs.items():
                    status = job.status
                    description = pbar_description(task_name, status)
                    pbar = progress.add_task(description, total=len(run_statuses) - 1)
                    pbar_tasks[task_name] = pbar

                while any(job.status not in end_statuses for job in self.jobs.values()):
                    for task_name, job in self.jobs.items():
                        pbar = pbar_tasks[task_name]
                        status = job.status
                        description = pbar_description(task_name, status)
                        completed = run_statuses.index(status)
                        progress.update(pbar, description=description, completed=completed)
                    time.sleep(web.REFRESH_TIME)

                # set all to 100% completed
                for task_name, job in self.jobs.items():
                    pbar = pbar_tasks[task_name]
                    status = job.status
                    description = pbar_description(task_name, status)
                    if status == "success":
                        progress.update(
                            pbar,
                            description=description,
                            completed=len(run_statuses) - 1,
                            refresh=True,
                        )

                console.log("Batch complete.")

        else:
            while any(job.status not in end_statuses for job in self.jobs.values()):
                time.sleep(web.REFRESH_TIME)

    @staticmethod
    def _job_data_path(task_id: TaskId, path_dir: str = DEFAULT_DATA_DIR):
        """Default path to data of a single :class:`Job` in :class:`Batch`.

        Parameters
        ----------
        task_id : str
            task_id corresponding to a :class:`Job`.
        path_dir : str = './'
            Base directory where data will be downloaded, by default, the current working directory.

        Returns
        -------
        str
            Full path to the data file.
        """
        return os.path.join(path_dir, f"{str(task_id)}.hdf5")

    @staticmethod
    def _batch_path(path_dir: str = DEFAULT_DATA_DIR):
        """Default path to save :class:`Batch` hdf5 file.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where the batch.hdf5 will be downloaded,
            by default, the current working directory.

        Returns
        -------
        str
            Full path to the batch file.
        """
        return os.path.join(path_dir, "batch.hdf5")

    def download(self, path_dir: str = DEFAULT_DATA_DIR) -> None:
        """Download results of each task.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where data will be downloaded, by default the current working directory.

        Note
        ----
        To load the data into :class:`.SimulationData`objets, can call :meth:`Batch.items`.

        The data for each task will be named as ``{path_dir}/{task_name}.hdf5``.
        The :class:`Batch` hdf5 file will be automatically saved as ``{path_dir}/batch.hdf5``,
        allowing one to load this :class:`Batch` later using ``batch = Batch.from_file()``.
        """

        self.to_file(self._batch_path(path_dir=path_dir))

        for job in self.jobs.values():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            job.download(path=job_path)

    def load(self, path_dir: str = DEFAULT_DATA_DIR) -> BatchData:
        """Download results and load them into :class:`.BatchData` object.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where data will be downloaded, by default current working directory.

        Returns
        ------
        :class:`BatchData`
            Contains the :class:`.SimulationData` of each :class:`.Simulation` in :class:`Batch`.

        The :class:`Batch` hdf5 file will be automatically saved as ``{path_dir}/batch.hdf5``,
        allowing one to load this :class:`Batch` later using ``batch = Batch.from_file()``.
        """

        self.to_file(self._batch_path(path_dir=path_dir))

        if self.jobs is None:
            raise DataError("Can't load batch results, hasn't been uploaded.")

        task_paths = {}
        task_ids = {}
        for task_name, job in self.jobs.items():
            task_paths[task_name] = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            task_ids[task_name] = self.jobs[task_name].task_id

        return BatchData(task_paths=task_paths, task_ids=task_ids, verbose=self.verbose)

    def delete(self):
        """Delete server-side data associated with each task in the batch."""
        for _, job in self.jobs.items():
            job.delete()
