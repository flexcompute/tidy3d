"""higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks."""
from __future__ import annotations

import os
from abc import ABC
from typing import Dict, Tuple
import time

from rich.progress import Progress
import pydantic.v1 as pd

from .tidy3d_stub import SimulationType, SimulationDataType
from ..api import webapi as web
from ..core.task_info import TaskInfo, RunInfo
from ..core.constants import TaskId, TaskName
from ...components.base import Tidy3dBaseModel
from ...log import log, get_logging_console

from ...exceptions import DataError

DEFAULT_DATA_PATH = "simulation_data.hdf5"
DEFAULT_DATA_DIR = "."


class WebContainer(Tidy3dBaseModel, ABC):
    """Base class for :class:`Job` and :class:`Batch`, technically not used"""


class Job(WebContainer):
    """Interface for managing the task runs on the server."""

    simulation: SimulationType = pd.Field(
        ...,
        title="simulation",
        description="Simulation to run as a 'task'.",
        discriminator="type",
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

    solver_version: str = pd.Field(
        None,
        title="Solver Version",
        description_str="Custom solver version to use, "
        "otherwise uses default for the current front end version.",
    )

    verbose: bool = pd.Field(
        True, title="Verbose", description="Whether to print info messages and progressbars."
    )

    simulation_type: str = pd.Field(
        "tidy3d",
        title="Simulation Type",
        description="Type of simulation, used internally only.",
    )

    parent_tasks: Tuple[TaskId, ...] = pd.Field(
        None, title="Parent Tasks", description="Tuple of parent task ids, used internally only."
    )

    task_id: TaskId = pd.Field(
        None,
        title="Task Id",
        description="Task ID number, set when the task is uploaded, leave as None.",
    )

    _upload_fields = (
        "simulation",
        "task_name",
        "folder_name",
        "callback_url",
        "verbose",
        "simulation_type",
        "parent_tasks",
    )

    def run(self, path: str = DEFAULT_DATA_PATH) -> SimulationDataType:
        """Run :class:`Job` all the way through and return data.

        Parameters
        ----------
        path_dir : str = "./simulation_data.hdf5"
            Base directory where data will be downloaded, by default current working directory.

        Returns
        -------
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`]
            Object containing simulation results.
        """

        self.start()
        self.monitor()
        return self.load(path=path)

    @pd.root_validator()
    def _upload(cls, values) -> None:
        """Upload simulation to server without running."""

        # task_id already present, don't re-upload
        if values.get("task_id") is not None:
            return values

        # upload kwargs with all fields except task_id
        upload_kwargs = {key: values.get(key) for key in cls._upload_fields}
        task_id = web.upload(**upload_kwargs)

        # then set the task_id and return
        values["task_id"] = task_id
        return values

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
        web.start(self.task_id, solver_version=self.solver_version)

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
        To load the data after download, use :meth:`Job.load`.
        """
        web.download(task_id=self.task_id, path=path, verbose=self.verbose)

    def load(self, path: str = DEFAULT_DATA_PATH) -> SimulationDataType:
        """Download job results and load them into a data object.

        Parameters
        ----------
        path : str = "./simulation_data.hdf5"
            Path to download data as ``.hdf5`` file (including filename).

        Returns
        -------
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`]
            Object containing simulation results.
        """
        return web.load(task_id=self.task_id, path=path, verbose=self.verbose)

    def delete(self) -> None:
        """Delete server-side data associated with :class:`Job`."""
        web.delete(self.task_id)

    def real_cost(self, verbose: bool = True) -> float:
        """Get the billed cost for the task associated with this job.

        Parameters
        ----------
        verbose : bool = True
            Whether to log the cost and helpful messages.

        Returns
        -------
        float
            Billed cost of the task in FlexCredits.
        """
        return web.real_cost(self.task_id, verbose=verbose)

    def estimate_cost(self, verbose: bool = True) -> float:
        """Compute the maximum FlexCredit charge for a given :class:`.Job`.

        Parameters
        ----------
        verbose : bool = True
            Whether to log the cost and helpful messages.

        Returns
        -------
        float
            Estimated cost of the task in FlexCredits.

        Note
        ----
        Cost is calculated assuming the simulation runs for
        the full ``run_time``. If early shut-off is triggered, the cost is adjusted proportionately.
        """
        return web.estimate_cost(self.task_id, verbose=verbose)


class BatchData(Tidy3dBaseModel):
    """Holds a collection of data objects returned by :class:`.Batch`."""

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

    def load_sim_data(self, task_name: str) -> SimulationDataType:
        """Load a simulation data object from file by task name."""
        task_data_path = self.task_paths[task_name]
        task_id = self.task_ids[task_name]
        web.get_info(task_id)
        return web.load(
            task_id=task_id,
            path=task_data_path,
            replace_existing=False,
            verbose=self.verbose,
        )

    def items(self) -> Tuple[TaskName, SimulationDataType]:
        """Iterate through the simulations for each task_name."""
        for task_name in self.task_paths.keys():
            yield task_name, self.load_sim_data(task_name)

    def __getitem__(self, task_name: TaskName) -> SimulationDataType:
        """Get the simulation data object for a given ``task_name``."""
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
            Contains Union[:class:`.SimulationData`, :class:`.HeatSimulationData`]
            for each Union[:class:`.Simulation`, :class:`.HeatSimulation`] in :class:`Batch`.
        """

        batch_file = Batch._batch_path(path_dir=path_dir)
        batch = Batch.from_file(batch_file)
        return batch.load(path_dir=path_dir)


class Batch(WebContainer):
    """Interface for submitting multiple simulations to the sever."""

    simulations: Dict[TaskName, SimulationType] = pd.Field(
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

    solver_version: str = pd.Field(
        None,
        title="Solver Version",
        description_str="Custom solver version to use, "
        "otherwise uses default for the current front end version.",
    )

    callback_url: str = pd.Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event. "
        "The body content is a json file with fields "
        "``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.",
    )

    simulation_type: str = pd.Field(
        "tidy3d",
        title="Simulation Type",
        description="Type of each simulation in the batch, used internally only.",
    )

    parent_tasks: Dict[str, Tuple[TaskId, ...]] = pd.Field(
        None,
        title="Parent Tasks",
        description="Collection of parent task ids for each job in batch, used internally only.",
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
            Contains Union[:class:`.SimulationData`, :class:`.HeatSimulationData`] for
            each Union[:class:`.Simulation`, :class:`.HeatSimulation`] in :class:`Batch`.

        Note
        ----
        A typical usage might look like:

        >>> batch_data = batch.run()
        >>> for task_name, sim_data in batch_data.items():
        ...     # do something with data.

        ``bach_data`` does not store all of the data objects in memory,
        rather it iterates over the task names and loads the corresponding
        data from file one by one. If no file exists for that task, it downloads it.
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

        # the type of job to upload (to generalize to subclasses)
        JobType = cls.__fields__["jobs"].type_
        parent_tasks = values.get("parent_tasks")

        verbose = bool(values.get("verbose"))
        jobs = {}
        for task_name, simulation in values.get("simulations").items():

            upload_kwargs = {key: values.get(key) for key in JobType._upload_fields}
            upload_kwargs["task_name"] = task_name
            upload_kwargs["simulation"] = simulation
            upload_kwargs["verbose"] = verbose
            if parent_tasks and task_name in parent_tasks:
                upload_kwargs["parent_tasks"] = parent_tasks[task_name]
            job = JobType(**upload_kwargs)
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

    def monitor(self) -> None:
        """Monitor progress of each of the running tasks.

        Note
        ----
        To loop through the data of completed simulations, can call :meth:`Batch.items`.
        """

        def pbar_description(task_name: str, status: str) -> str:
            """Make a progressbar description based on the status."""
            description = f"{task_name}: status = {status}"

            # if something went wrong, make it red
            if "error" in status or "diverge" in status:
                description = f"[red]{description}"

            return description

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
        end_statuses = ("success", "error", "errored", "diverged", "diverge", "deleted", "draft")

        if self.verbose:
            console = get_logging_console()
            console.log("Started working on Batch.")

            self.estimate_cost()
            console.log(
                "Use 'Batch.real_cost()' to "
                "get the billed FlexCredit cost after the Batch has completed."
            )

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

                        # if a problem occured, update progressbar completion to 100%
                        if status not in run_statuses:
                            completed = run_statuses.index("success")
                        else:
                            completed = run_statuses.index(status)

                        progress.update(pbar, description=description, completed=completed)
                    time.sleep(web.REFRESH_TIME)

                # set all to 100% completed (if error or diverge, will be red)
                for task_name, job in self.jobs.items():
                    pbar = pbar_tasks[task_name]
                    status = job.status
                    description = pbar_description(task_name, status)

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
        To load and iterate through the data, use :meth:`Batch.items()`.

        The data for each task will be named as ``{path_dir}/{task_name}.hdf5``.
        The :class:`Batch` hdf5 file will be automatically saved as ``{path_dir}/batch.hdf5``,
        allowing one to load this :class:`Batch` later using ``batch = Batch.from_file()``.
        """

        self.to_file(self._batch_path(path_dir=path_dir))

        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)

            if "error" in job.status:
                log.warning(f"Not downloading '{task_name}' as the task errored.")
                continue

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
            Contains Union[:class:`.SimulationData`, :class:`.HeatSimulationData`] for each
            Union[:class:`.Simulation`, :class:`.HeatSimulation`] in :class:`Batch`.

        The :class:`Batch` hdf5 file will be automatically saved as ``{path_dir}/batch.hdf5``,
        allowing one to load this :class:`Batch` later using ``batch = Batch.from_file()``.
        """

        self.to_file(self._batch_path(path_dir=path_dir))

        if self.jobs is None:
            raise DataError("Can't load batch results, hasn't been uploaded.")

        task_paths = {}
        task_ids = {}
        for task_name, job in self.jobs.items():
            if "error" in job.status:
                log.warning(f"Not loading '{task_name}' as the task errored.")
                continue

            task_paths[task_name] = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            task_ids[task_name] = self.jobs[task_name].task_id

        return BatchData(task_paths=task_paths, task_ids=task_ids, verbose=self.verbose)

    def delete(self) -> None:
        """Delete server-side data associated with each task in the batch."""
        for _, job in self.jobs.items():
            job.delete()

    def real_cost(self, verbose: bool = True) -> float:
        """Get the sum of billed costs for each task associated with this batch.

        Parameters
        ----------
        verbose : bool = True
            Whether to log the cost and helpful messages.

        Returns
        -------
        float
            Billed cost for the entire :class:`.Batch`.
        """
        real_cost_sum = 0.0
        for _, job in self.jobs.items():
            cost_job = job.real_cost(verbose=False)
            if cost_job is not None:
                real_cost_sum += cost_job

        real_cost_sum = real_cost_sum or None  # convert to None if 0

        if real_cost_sum and verbose:
            console = get_logging_console()
            console.log(f"Total billed flex credit cost: {real_cost_sum:1.3f}.")
        return real_cost_sum

    def estimate_cost(self, verbose: bool = True) -> float:
        """Compute the maximum FlexCredit charge for a given :class:`.Batch`.

        Parameters
        ----------
        verbose : bool = True
            Whether to log the cost and helpful messages.

        Note
        ----
        Cost is calculated assuming the simulation runs for
        the full ``run_time``. If early shut-off is triggered, the cost is adjusted proportionately.

        Returns
        -------
        float
            Estimated total cost of the tasks in FlexCredits.
        """
        batch_cost = sum(job.estimate_cost(verbose=False) for _, job in self.jobs.items())

        if verbose:
            console = get_logging_console()
            if batch_cost is not None and batch_cost > 0:
                console.log(f"Maximum FlexCredit cost: {batch_cost:1.3f} for the whole batch.")
            else:
                console.log("Could not get estimated batch cost!")

        return batch_cost
