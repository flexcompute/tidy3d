"""higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks."""

from __future__ import annotations

import os
from abc import ABC
from typing import Literal, Optional

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.components.types import annotate_type
from tidy3d.log import get_logging_console
from tidy3d.web.core.constants import TaskId, TaskName
from tidy3d.web.core.task_core import Folder
from tidy3d.web.core.task_info import RunInfo, TaskInfo
from tidy3d.web.core.types import PayType

from . import webapi as web
from .asynchronous import download_async, load_async, monitor_async, start_async, upload_async
from .batch_data import DEFAULT_DATA_DIR, DEFAULT_DATA_PATH, BatchData
from .tidy3d_stub import SimulationDataType, SimulationType

# Max # of workers for parallel upload / download: above 10, performance is same but with warnings
DEFAULT_NUM_WORKERS = 10


class WebContainer(Tidy3dBaseModel, ABC):
    """Base class for :class:`Job` and :class:`Batch`, technically not used"""

    from abc import abstractmethod

    @staticmethod
    @abstractmethod
    def _check_path_dir(path: str) -> None:
        """Make sure local output directory exists and create it if not."""

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

    parent_tasks: tuple[TaskId, ...] = pd.Field(
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
        # Use lazy import to avoid circular dependency
        from .autograd.autograd import run as run_autograd

        # Use autograd-compatible run function instead of manual upload/start/monitor/load
        return run_autograd(
            simulation=self.simulation,
            task_name=self.task_name,
            path=path,
            folder_name=self.folder_name,
            callback_url=self.callback_url,
            verbose=self.verbose,
            simulation_type=self.simulation_type,
            parent_tasks=self.parent_tasks,
            solver_version=self.solver_version,
            pay_type=self.pay_type,
        )

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

    simulations: dict[TaskName, annotate_type(SimulationType)] = pd.Field(
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

    parent_tasks: dict[str, tuple[TaskId, ...]] = pd.Field(
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

    jobs_cached: dict[TaskName, Job] = pd.Field(
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
        # Use lazy import to avoid circular dependency
        from .autograd.autograd import run_async as run_async_autograd

        # Use autograd-compatible run_async function instead of manual upload/start/monitor/load
        return run_async_autograd(
            simulations=self.simulations,
            path_dir=path_dir,
            folder_name=self.folder_name,
            callback_url=self.callback_url,
            verbose=self.verbose,
            simulation_type=self.simulation_type,
            parent_tasks=self.parent_tasks,
            reduce_simulation=self.reduce_simulation,
            pay_type=self.pay_type,
        )

    @cached_property
    def jobs(self) -> dict[TaskName, Job]:
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
        task_ids = upload_async(
            simulations=self.simulations,
            folder_name=self.folder_name,
            callback_url=self.callback_url,
            num_workers=self.num_workers,
            verbose=self.verbose,
            simulation_type=self.simulation_type,
            parent_tasks=self.parent_tasks,
            reduce_simulation=self.reduce_simulation,
            pay_type=self.pay_type,
        )
        # Update jobs with the returned task_ids
        for task_name, task_id in task_ids.items():
            if task_name in self.jobs:
                self.jobs[task_name].task_id = task_id

    def get_info(self) -> dict[TaskName, TaskInfo]:
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
        task_ids = {task_name: job.task_id for task_name, job in self.jobs.items()}
        start_async(task_ids, self.num_workers, self.verbose)

    def get_run_info(self) -> dict[TaskName, RunInfo]:
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
        task_ids = {task_name: job.task_id for task_name, job in self.jobs.items()}
        monitor_async(task_ids, self.verbose)

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
        return os.path.join(path_dir, f"{task_id!s}.hdf5")

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

    def download(self, path_dir: str = DEFAULT_DATA_DIR, replace_existing: bool = False) -> None:
        """Download results of each task.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where data will be downloaded, by default the current working directory.
        replace_existing : bool = False
            Downloads the data even if path exists (overwriting the existing).

        Note
        ----
        To load and iterate through the data, use :meth:`Batch.items()`.

        The data for each task will be named as ``{path_dir}/{task_id}.hdf5``.
        The :class:`Batch` hdf5 file will be automatically saved as ``{path_dir}/batch.hdf5``,
        allowing one to load this :class:`Batch` later using ``batch = Batch.from_file()``.
        """
        task_ids = {task_name: job.task_id for task_name, job in self.jobs.items()}
        download_async(task_ids, path_dir, self.num_workers, self.verbose, replace_existing)

    def load(self, path_dir: str = DEFAULT_DATA_DIR, replace_existing: bool = False) -> BatchData:
        """Download results and load them into :class:`.BatchData` object.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where data will be downloaded, by default current working directory.
        replace_existing : bool = False
            Downloads the data even if path exists (overwriting the existing).

        Returns
        ------
        :class:`BatchData`
            Contains Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] for each
            Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] in :class:`Batch`.

        The :class:`Batch` hdf5 file will be automatically saved as ``{path_dir}/batch.hdf5``,
        allowing one to load this :class:`Batch` later using ``batch = Batch.from_file()``.
        """
        task_ids = {task_name: job.task_id for task_name, job in self.jobs.items()}
        return load_async(
            task_ids, self.simulations, path_dir, self.num_workers, self.verbose, replace_existing
        )

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
