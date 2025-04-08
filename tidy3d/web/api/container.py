"""higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks."""

from __future__ import annotations

import concurrent
import os
import time
from abc import ABC
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import pydantic.v1 as pd
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from ...components.base import Tidy3dBaseModel, cached_property
from ...components.mode.mode_solver import ModeSolver
from ...components.types import Literal, annotate_type
from ...exceptions import DataError
from ...log import get_logging_console, log
from ..api import webapi as web
from ..core.constants import TaskId, TaskName
from ..core.task_core import Folder
from ..core.task_info import RunInfo, TaskInfo
from ..core.types import PayType
from .tidy3d_stub import SimulationDataType, SimulationType

# Max # of workers for parallel upload / download: above 10, performance is same but with warnings
DEFAULT_NUM_WORKERS = 10
DEFAULT_DATA_PATH = "simulation_data.hdf5"
DEFAULT_DATA_DIR = "."
BATCH_MONITOR_PROGRESS_REFRESH_TIME = 0.02


class WebContainer(Tidy3dBaseModel, ABC):
    """Base class for :class:`Job` and :class:`Batch`, technically not used"""

    from abc import abstractmethod

    @staticmethod
    @abstractmethod
    def _check_path_dir(path: str) -> None:
        """Make sure local output directory exists and create it if not."""
        pass

    @staticmethod
    def _check_folder(folder_name: str) -> None:
        """Make sure ``folder_name`` exists on the web UI and create it if not."""
        Folder.get(folder_name, create=True)


class Job(WebContainer):
    """
    Interface for managing the running of a :class:`.Simulation` on server.

    Notes
    -----

        This class provides a more convenient way to manage single simulations, mainly because it eliminates the need
        for keeping track of the ``task_id`` and original :class:`.Simulation`.

        We can get the cost estimate of running the task before actually running it. This prevents us from
        accidentally running large jobs that we set up by mistake. The estimated cost is the maximum cost
        corresponding to running all the time steps.

        Another convenient thing about :class:`Job` objects is that they can be saved and loaded just like other
        ``tidy3d`` components.

    Examples
    --------

        Once you've created a ``job`` object using :class:`tidy3d.web.api.container.Job`, you can upload it to our servers with:

        .. code-block:: python

            tidy3d.web.upload(simulation, task_name="task_name", verbose=verbose)`

        It will not run until you explicitly tell it to do so with:

        .. code-block:: python

            tidy3d.web.api.webapi.start(job.task_id)

        To monitor the simulation's progress and wait for its completion, use

        .. code-block:: python

            tidy3d.web.api.webapi.monitor(job.task_id, verbose=verbose)

        After running the simulation, you can load the results using for example:

        .. code-block:: python

            sim_data = tidy3d.web.api.webapi.load(job.task_id, path="out/simulation.hdf5", verbose=verbose)

        The job container has a convenient method to save and load the results of a job that has already finished,
        without needing to know the task_id, as below:

        .. code-block:: python

            # Saves the job metadata to a single file.
            job.to_file("data/job.json")

            # You can exit the session, break here, or continue in new session.

            # Load the job metadata from file.
            job_loaded = tidy3d.web.api.container.Job.from_file("data/job.json")

            # Download the data from the server and load it into a SimulationData object.
            sim_data = job_loaded.load(path="data/sim.hdf5")


    See Also
    --------

    :meth:`tidy3d.web.api.webapi.run_async`
        Submits a set of :class:`.Simulation` objects to server, starts running, monitors progress,
        downloads, and loads results as a :class:`.BatchData` object.

    :class:`Batch`
         Interface for submitting several :class:`Simulation` objects to sever.

    **Notebooks**
        *  `Running simulations through the cloud <../../notebooks/WebAPI.html>`_
        * `Performing parallel / batch processing of simulations <../../notebooks/ParameterScan.html>`_
        * `Inverse taper edge coupler <../../notebooks/EdgeCoupler.html>`_
    """

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
        description="Custom solver version to use, "
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

    task_id_cached: TaskId = pd.Field(
        None,
        title="Task ID (Cached)",
        description="Optional field to specify ``task_id``. Only used as a workaround internally "
        "so that ``task_id`` is written when ``Job.to_file()`` and then the proper task is loaded "
        "from ``Job.from_file()``. We recommend leaving unset as setting this field along with "
        "fields that were not used to create the task will cause errors.",
    )

    reduce_simulation: Literal["auto", True, False] = pd.Field(
        "auto",
        title="Reduce Simulation",
        description="Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.",
    )

    pay_type: PayType = pd.Field(
        PayType.AUTO,
        title="Payment Type",
        description="Specify the payment method.",
    )

    _upload_fields = (
        "simulation",
        "task_name",
        "folder_name",
        "callback_url",
        "verbose",
        "simulation_type",
        "parent_tasks",
        "solver_version",
        "reduce_simulation",
    )

    def to_file(self, fname: str) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        fname : str
            Full path to the .yaml or .json file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """
        task_id_cached = self._cached_properties.get("task_id")
        self = self.updated_copy(task_id_cached=task_id_cached)
        super(Job, self).to_file(fname=fname)  # noqa: UP008

    def run(self, path: str = DEFAULT_DATA_PATH) -> SimulationDataType:
        """Run :class:`Job` all the way through and return data.

        Parameters
        ----------
        path_dir : str = "./simulation_data.hdf5"
            Base directory where data will be downloaded, by default current working directory.

        Returns
        -------
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
            Object containing simulation results.
        """
        self.upload()
        self.start()
        self.monitor()
        return self.load(path=path)

    @cached_property
    def task_id(self) -> TaskId:
        """The task ID for this ``Job``. Uploads the ``Job`` if it hasn't already been uploaded."""
        if self.task_id_cached:
            return self.task_id_cached
        self._check_folder(self.folder_name)
        return self._upload()

    def _upload(self) -> TaskId:
        """Upload this job and return the task ID for handling."""
        # upload kwargs with all fields except task_id
        upload_kwargs = {key: getattr(self, key) for key in self._upload_fields}
        task_id = web.upload(**upload_kwargs)
        return task_id

    def upload(self) -> None:
        """Upload this ``Job``."""
        _ = self.task_id

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
        web.start(self.task_id, solver_version=self.solver_version, pay_type=self.pay_type)

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
        To load the output of completed simulation into :class:`.SimulationData` objects,
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
        self._check_path_dir(path=path)
        web.download(task_id=self.task_id, path=path, verbose=self.verbose)

    def load(self, path: str = DEFAULT_DATA_PATH) -> SimulationDataType:
        """Download job results and load them into a data object.

        Parameters
        ----------
        path : str = "./simulation_data.hdf5"
            Path to download data as ``.hdf5`` file (including filename).

        Returns
        -------
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
            Object containing simulation results.
        """
        self._check_path_dir(path=path)
        data = web.load(task_id=self.task_id, path=path, verbose=self.verbose)
        if isinstance(self.simulation, ModeSolver):
            self.simulation._patch_data(data=data)
        return data

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
        return web.estimate_cost(self.task_id, verbose=verbose, solver_version=self.solver_version)

    @staticmethod
    def _check_path_dir(path: str) -> None:
        """Make sure parent directory of ``path`` exists and create it if not.

        Parameters
        ----------
        path : str
            Path to file to be created (including filename).
        """
        parent_dir = os.path.dirname(path)
        if len(parent_dir) > 0 and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)


class BatchData(Tidy3dBaseModel, Mapping):
    """
    Holds a collection of :class:`.SimulationData` returned by :class:`Batch`.

    Notes
    -----

        When the batch is completed, the output is not a :class:`.SimulationData` but rather a :class:`BatchData`. The
        data within this :class:`BatchData` object can either be indexed directly ``batch_results[task_name]`` or can be looped
        through ``batch_results.items()`` to get the :class:`.SimulationData` for each task.

    See Also
    --------

    :class:`Batch`:
         Interface for submitting several :class:`.Simulation` objects to sever.

    :class:`.SimulationData`:
         Stores data from a collection of :class:`.Monitor` objects in a :class:`.Simulation`.

    **Notebooks**
        * `Running simulations through the cloud <../../notebooks/WebAPI.html>`_
        * `Performing parallel / batch processing of simulations <../../notebooks/ParameterScan.html>`_
    """

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
            verbose=False,
        )

    def __getitem__(self, task_name: TaskName) -> SimulationDataType:
        """Get the simulation data object for a given ``task_name``."""
        return self.load_sim_data(task_name)

    def __iter__(self):
        """Iterate over the task names."""
        return iter(self.task_paths)

    def __len__(self):
        """Return the number of tasks in the batch."""
        return len(self.task_paths)

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
            Contains Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
            for each Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] in :class:`Batch`.
        """

        batch_file = Batch._batch_path(path_dir=path_dir)
        batch = Batch.from_file(batch_file)
        return batch.load(path_dir=path_dir)


class Batch(WebContainer):
    """
    Interface for submitting several :class:`Simulation` objects to sever.

    Notes
    -----

        Commonly one needs to submit a batch of :class:`Simulation`. The built-in :class:`Batch` object is the best way to upload,
        start, monitor, and load a series of tasks. The batch object is like a :class:`Job`, but stores task metadata
        for a series of simulations.

    See Also
    --------

    :meth:`tidy3d.web.api.webapi.run_async`
        Submits a set of :class:`.Simulation` objects to server, starts running, monitors progress,
        downloads, and loads results as a :class:`.BatchData` object.

    :class:`Job`:
        Interface for managing the running of a Simulation on server.

    **Notebooks**
        * `Running simulations through the cloud <../../notebooks/WebAPI.html>`_
        * `Performing parallel / batch processing of simulations <../../notebooks/ParameterScan.html>`_
        * `Inverse taper edge coupler <../../notebooks/EdgeCoupler.html>`_
    """

    simulations: Dict[TaskName, annotate_type(SimulationType)] = pd.Field(
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
        description="Custom solver version to use, "
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

    num_workers: Optional[pd.PositiveInt] = pd.Field(
        DEFAULT_NUM_WORKERS,
        title="Number of Workers",
        description="Number of workers for multi-threading upload and download of batch. "
        "Corresponds to ``max_workers`` argument passed to "
        "``concurrent.futures.ThreadPoolExecutor``. When left ``None``, will pass the maximum "
        "number of threads available on the system.",
    )

    reduce_simulation: Literal["auto", True, False] = pd.Field(
        "auto",
        title="Reduce Simulation",
        description="Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.",
    )

    pay_type: PayType = pd.Field(
        PayType.AUTO,
        title="Payment Type",
        description="Specify the payment method.",
    )

    jobs_cached: Dict[TaskName, Job] = pd.Field(
        None,
        title="Jobs (Cached)",
        description="Optional field to specify ``jobs``. Only used as a workaround internally "
        "so that ``jobs`` is written when ``Batch.to_file()`` and then the proper task is loaded "
        "from ``Batch.from_file()``. We recommend leaving unset as setting this field along with "
        "fields that were not used to create the task will cause errors.",
    )

    _job_type = Job

    def run(self, path_dir: str = DEFAULT_DATA_DIR) -> BatchData:
        """Upload and run each simulation in :class:`Batch`.

        Parameters
        ----------
        path_dir : str
            Base directory where data will be downloaded, by default current working directory.

        Returns
        ------
        :class:`BatchData`
            Contains Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData] for
            each Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] in :class:`Batch`.

        Note
        ----
        A typical usage might look like:

        >>> from tidy3d.web.api.container import Batch
        >>> custom_batch = Batch()
        >>> batch_data = custom_batch.run() # doctest: +SKIP
        >>> for task_name, sim_data in batch_data.items(): # doctest: +SKIP
        ...     # do something with data. # doctest: +SKIP

        ``bach_data`` does not store all of the data objects in memory,
        rather it iterates over the task names and loads the corresponding
        data from file one by one. If no file exists for that task, it downloads it.
        """
        self._check_path_dir(path_dir)
        self.upload()
        self.start()
        self.monitor()
        self.download(path_dir=path_dir)
        return self.load(path_dir=path_dir)

    @cached_property
    def jobs(self) -> Dict[TaskName, Job]:
        """Create a series of tasks in the :class:`.Batch` and upload them to server.

        Note
        ----
        To start the simulations running, must call :meth:`Batch.start` after uploaded.
        """

        if self.jobs_cached is not None:
            return self.jobs_cached

        # the type of job to upload (to generalize to subclasses)
        JobType = self._job_type
        self_dict = self.dict()

        jobs = {}
        for task_name, simulation in self.simulations.items():
            job_kwargs = {}

            for key in JobType._upload_fields:
                if key in self_dict:
                    job_kwargs[key] = self_dict.get(key)

            job_kwargs["task_name"] = task_name
            job_kwargs["simulation"] = simulation
            job_kwargs["verbose"] = False
            job_kwargs["solver_version"] = self.solver_version
            job_kwargs["pay_type"] = self.pay_type
            job_kwargs["reduce_simulation"] = self.reduce_simulation
            if self.parent_tasks and task_name in self.parent_tasks:
                job_kwargs["parent_tasks"] = self.parent_tasks[task_name]
            job = JobType(**job_kwargs)
            jobs[task_name] = job
        return jobs

    def to_file(self, fname: str) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        fname : str
            Full path to the .yaml or .json file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """
        jobs_cached = self._cached_properties.get("jobs")
        if jobs_cached is not None:
            jobs = {}
            for key, job in jobs_cached.items():
                task_id = job._cached_properties.get("task_id")
                jobs[key] = job.updated_copy(task_id_cached=task_id)
            self = self.updated_copy(jobs_cached=jobs)
        super(Batch, self).to_file(fname=fname)  # noqa: UP008

    @property
    def num_jobs(self) -> int:
        """Number of jobs in the batch."""
        return len(self.jobs)

    def upload(self) -> None:
        """Upload a series of tasks associated with this ``Batch`` using multi-threading."""
        self._check_folder(self.folder_name)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(job.upload) for _, job in self.jobs.items()]

            # progressbar (number of tasks uploaded)
            if self.verbose:
                console = get_logging_console()
                progress_columns = (
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                )
                with Progress(*progress_columns, console=console, transient=False) as progress:
                    pbar_message = f"Uploading data for {self.num_jobs} tasks"
                    pbar = progress.add_task(pbar_message, total=self.num_jobs)
                    completed = 0
                    for _ in concurrent.futures.as_completed(futures):
                        completed += 1
                        progress.update(pbar, completed=completed)

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
        if self.verbose:
            console = get_logging_console()
            console.log(f"Started working on Batch containing {self.num_jobs} tasks.")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for _, job in self.jobs.items():
                executor.submit(job.start)

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
        """Monitor progress of each of the running tasks."""

        def pbar_description(
            task_name: str, status: str, max_name_length: int, status_width: int
        ) -> str:
            """Make a progressbar description based on the status."""
            # if task name too long, truncate and add ...
            if len(task_name) > max_name_length - 3:  # -3 to leave room for ...
                task_name = task_name[: (max_name_length - 3)] + "..."

            # right-align status
            task_part = f"{task_name:<{max_name_length}}"

            if "error" in status or "diverge" in status or "aborted" in status:
                status_part = f"→ [red]{status:<{status_width}}"
            elif status == "success":
                status_part = f"→ [green]{status:<{status_width}}"
            elif status == "queued" or status == "queued_solver" or status == "aborting":
                status_part = f"→ [yellow]{status:<{status_width}}"
            elif status in ["preprocess", "postprocess", "running"]:
                status_part = f"→ [blue]{status:<{status_width}}"
            else:
                status_part = f"→ {status:<{status_width}}"

            return f"{task_part} {status_part}"

        run_statuses = [
            "draft",
            "queued",
            "preprocess",
            "queued_solver",
            "running",
            "postprocess",
            "visualize",
            "success",
            "aborting",
        ]
        end_statuses = (
            "success",
            "error",
            "errored",
            "diverged",
            "diverge",
            "deleted",
            "draft",
            "aborted",
        )

        max_task_name = max(len(task_name) for task_name in self.jobs.keys())
        max_name_length = min(30, max(max_task_name, 15))
        status_width = max(
            max(len(status) for status in run_statuses), max(len(status) for status in end_statuses)
        )

        if self.verbose:
            console = get_logging_console()

            self.estimate_cost()
            console.log(
                "Use 'Batch.real_cost()' to "
                "get the billed FlexCredit cost after the Batch has completed."
            )

            progress_columns = (
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=25),
                TaskProgressColumn(),
                TimeElapsedColumn(),
            )

            with Progress(*progress_columns, console=console, transient=False) as progress:
                # create progress bars
                pbar_tasks = {}
                for task_name, job in self.jobs.items():
                    status = job.status
                    description = pbar_description(task_name, status, max_name_length, status_width)
                    completed = run_statuses.index(status) if status in run_statuses else 0
                    pbar = progress.add_task(
                        description, total=len(run_statuses) - 1, completed=completed
                    )
                    pbar_tasks[task_name] = pbar

                while any(job.status not in end_statuses for job in self.jobs.values()):
                    updates = []
                    for task_name, job in self.jobs.items():
                        status = job.status
                        if status in run_statuses:
                            updates.append(
                                (
                                    pbar_tasks[task_name],
                                    pbar_description(
                                        task_name, status, max_name_length, status_width
                                    ),
                                    run_statuses.index(status),
                                )
                            )

                    for pbar, description, completed in updates:
                        progress.update(
                            pbar, description=description, completed=completed, refresh=False
                        )

                    progress.refresh()
                    time.sleep(BATCH_MONITOR_PROGRESS_REFRESH_TIME)

                updates = []
                for task_name, job in self.jobs.items():
                    updates.append(
                        (
                            pbar_tasks[task_name],
                            pbar_description(task_name, job.status, max_name_length, status_width),
                            len(run_statuses) - 1,
                        )
                    )

                for pbar, description, completed in updates:
                    progress.update(
                        pbar, description=description, completed=completed, refresh=False
                    )

                progress.refresh()
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

        The data for each task will be named as ``{path_dir}/{task_id}.hdf5``.
        The :class:`Batch` hdf5 file will be automatically saved as ``{path_dir}/batch.hdf5``,
        allowing one to load this :class:`Batch` later using ``batch = Batch.from_file()``.
        """
        self._check_path_dir(path_dir=path_dir)
        self.to_file(self._batch_path(path_dir=path_dir))

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            fns = []
            for task_name, job in self.jobs.items():
                job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)

                if "error" in job.status:
                    log.warning(f"Not downloading '{task_name}' as the task errored.")
                    continue

                def fn(job=job, job_path=job_path) -> None:
                    return job.download(path=job_path)

                fns.append(fn)

            futures = [executor.submit(fn) for fn in fns]

            if self.verbose:
                console = get_logging_console()
                progress_columns = (
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                )
                with Progress(*progress_columns, console=console, transient=False) as progress:
                    pbar_message = f"Downloading data for {len(fns)} tasks"
                    pbar = progress.add_task(pbar_message, total=len(fns))
                    completed = 0
                    for _ in concurrent.futures.as_completed(futures):
                        completed += 1
                        progress.update(pbar, completed=completed)

    def load(self, path_dir: str = DEFAULT_DATA_DIR) -> BatchData:
        """Download results and load them into :class:`.BatchData` object.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where data will be downloaded, by default current working directory.

        Returns
        ------
        :class:`BatchData`
            Contains Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] for each
            Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] in :class:`Batch`.

        The :class:`Batch` hdf5 file will be automatically saved as ``{path_dir}/batch.hdf5``,
        allowing one to load this :class:`Batch` later using ``batch = Batch.from_file()``.
        """
        self._check_path_dir(path_dir=path_dir)
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

        data = BatchData(task_paths=task_paths, task_ids=task_ids, verbose=self.verbose)

        for task_name, job in self.jobs.items():
            if isinstance(job.simulation, ModeSolver):
                job_data = data[task_name]
                job.simulation._patch_data(data=job_data)

        return data

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
        job_costs = [job.estimate_cost(verbose=False) for _, job in self.jobs.items()]
        if any(cost is None for cost in job_costs):
            batch_cost = None
        else:
            batch_cost = sum(job_costs)

        if verbose:
            console = get_logging_console()
            if batch_cost is not None and batch_cost > 0:
                console.log(f"Maximum FlexCredit cost: {batch_cost:1.3f} for the whole batch.")
            else:
                console.log("Could not get estimated batch cost!")

        return batch_cost

    @staticmethod
    def _check_path_dir(path_dir: str) -> None:
        """Make sure ``path_dir`` exists and create it if not.

        Parameters
        ----------
        path_dir : str
            Directory path where files will be saved.
        """
        if len(path_dir) > 0 and not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)
