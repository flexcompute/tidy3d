"""higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks."""

from __future__ import annotations

import atexit
import concurrent
import hashlib
import os
import shutil
import tempfile
import time
import uuid
from abc import ABC
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PositiveInt, PrivateAttr, field_validator, model_validator
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from tidy3d._runtime import WASM_BUILD
from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.types.workflow import WorkflowType
from tidy3d.config import config
from tidy3d.exceptions import DataError
from tidy3d.log import get_logging_console, log
from tidy3d.web.api import webapi as web
from tidy3d.web.api.states import (
    COMPLETED_PERCENT,
    COMPLETED_STATES,
    DRAFT_STATES,
    END_STATES,
    ERROR_STATES,
    PRE_ERROR_STATES,
    QUEUED_STATES,
    RUNNING_STATES,
    STATE_PROGRESS_PERCENTAGE,
)
from tidy3d.web.api.tidy3d_stub import Tidy3dStub, task_type_name_of
from tidy3d.web.api.webapi import _batch_detail_progress, restore_simulation_if_cached
from tidy3d.web.cache import _store_mode_solver_in_cache
from tidy3d.web.core.constants import TaskId, TaskName
from tidy3d.web.core.task_core import Folder
from tidy3d.web.core.task_info import BatchDetail
from tidy3d.web.core.types import PayType

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from os import PathLike

    from rich.progress import TaskID

    from tidy3d.components.types.workflow import WorkflowDataType
    from tidy3d.web.core.task_info import RunInfo, TaskInfo

DEFAULT_DATA_DIR = "."
BATCH_PROGRESS_REFRESH_TIME = 0.02
# Upload/start requests are network I/O-bound, so use a fixed concurrency cap.
UPLOAD_START_NUM_WORKERS = 64

BatchCategoryType = Literal[
    "tidy3d",
    "microwave",
    "tidy3d_design",
    "tidy3d_autograd",
    "tidy3d_autograd_async",
    "autograd_fwd",
    "autograd_bwd",
    "photonforge:tidy3d",
    "photonforge:mode_solver",
    "photonforge:eme",
    "photonforge:rf",
    "photonforge:heat_charge",
]


def _default_batch_num_workers() -> int:
    """Default worker count from runtime config."""
    return config.web.default_num_workers


class WebContainer(Tidy3dBaseModel, ABC):
    """Base class for :class:`Job` and :class:`Batch`, technically not used"""

    from abc import abstractmethod

    @staticmethod
    @abstractmethod
    def _check_path_dir(path: PathLike) -> None:
        """Make sure local output directory exists and create it if not."""

    @staticmethod
    def _check_folder(
        folder_name: str,
        projects_endpoint: str = "tidy3d/projects",
        project_endpoint: str = "tidy3d/project",
    ) -> None:
        """Make sure ``folder_name`` exists on the web UI and create it if not."""
        Folder.get(
            folder_name,
            create=True,
            projects_endpoint=projects_endpoint,
            project_endpoint=project_endpoint,
        )


class Job(WebContainer):
    """
    Interface for managing the running of a :class:`~tidy3d.Simulation` on server.

    Notes
    -----

        This class provides a more convenient way to manage single simulations, mainly because it eliminates the need
        for keeping track of the ``task_id`` and original :class:`~tidy3d.Simulation`.

        We can get the cost estimate of running the task before actually running it. This prevents us from
        accidentally running large jobs that we set up by mistake. The estimated cost is the maximum cost
        corresponding to running all the time steps.

        Another convenient thing about :class:`Job` objects is that they can be saved and loaded just like other
        ``tidy3d`` components.

    Examples
    --------

        Once you've created a ``job`` object using :class:`tidy3d.web.api.container.Job`, you can upload it to our servers with:

        .. code-block:: python

            tidy3d.web.upload(simulation, task_name="task_name", verbose=verbose)

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
        Submits a set of :class:`~tidy3d.Simulation` objects to server, starts running, monitors progress,
        downloads, and loads results as a :class:`.BatchData` object.

    :class:`Batch`
         Interface for submitting several :class:`~tidy3d.Simulation` objects to sever.

    **Notebooks**
        *  `Running simulations through the cloud <../../notebooks/WebAPI.html>`_
        * `Performing parallel / batch processing of simulations <../../notebooks/ParameterScan.html>`_
        * `Inverse taper edge coupler <../../notebooks/EdgeCoupler.html>`_
    """

    simulation: WorkflowType = Field(
        title="simulation",
        description="Simulation to run as a 'task'.",
        discriminator="type",
    )

    task_name: TaskName | None = Field(
        None,
        title="Task Name",
        description="Unique name of the task. Will be auto-generated if not provided.",
    )

    folder_name: str = Field(
        "default",
        title="Folder Name",
        description="Name of folder to store task on web UI.",
    )

    callback_url: str | None = Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event. "
        "The body content is a json file with fields "
        "``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.",
    )

    solver_version: str | None = Field(
        None,
        title="Solver Version",
        description="Custom solver version to use, "
        "otherwise uses default for the current front end version.",
    )

    verbose: bool = Field(
        True,
        title="Verbose",
        description="Whether to print info messages and progressbars.",
    )

    simulation_type: BatchCategoryType | None = Field(
        None,
        title="Simulation Type",
        description="Type of simulation, used internally only.",
    )

    parent_tasks: tuple[TaskId, ...] | None = Field(
        None,
        title="Parent Tasks",
        description="Tuple of parent task ids, used internally only.",
    )

    task_id_cached: TaskId | None = Field(
        None,
        title="Task ID (Cached)",
        description="Optional field to specify ``task_id``. Only used as a workaround internally "
        "so that ``task_id`` is written when ``Job.to_file()`` and then the proper task is loaded "
        "from ``Job.from_file()``. We recommend leaving unset as setting this field along with "
        "fields that were not used to create the task will cause errors.",
    )

    reduce_simulation: Literal["auto", True, False] = Field(
        "auto",
        title="Reduce Simulation",
        description="Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.",
    )

    pay_type: PayType | None = Field(
        None,
        title="Payment Type",
        description="Specify the payment method.",
    )

    lazy: bool = Field(
        False,
        title="Lazy",
        description="Whether to load the actual data (lazy=False) or return a proxy that loads the data when accessed (lazy=True).",
    )

    _upload_fields: tuple[str, ...] = PrivateAttr(
        (
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
    )

    _stash_path: str | None = PrivateAttr(default=None)
    _cached_task_id: TaskId | None = PrivateAttr(default=None)

    @cached_property
    def _stash_path_for_job(self) -> str:
        """Stash file which is a temporary location for the cached-restored file."""
        stash_dir = Path(tempfile.gettempdir()) / "tidy3d_stash"
        stash_dir.mkdir(parents=True, exist_ok=True)
        return str(Path(stash_dir / f"{uuid.uuid4()}.hdf5"))

    def _task_type_hint(self) -> str | None:
        """Best-effort task type derived from the simulation for default filename selection."""

        try:
            return task_type_name_of(self.simulation)
        except (AttributeError, TypeError, ValueError):
            return None

    def _materialize_from_stash(self, dst_path: os.PathLike) -> None:
        """Atomic copy from stash to requested path."""
        tmp = str(dst_path) + ".part"
        shutil.copy2(self._stash_path, tmp)
        os.replace(tmp, dst_path)

    def clear_stash(self) -> None:
        """Delete this job's stash file only."""
        if self._stash_path:
            try:
                if os.path.exists(self._stash_path):
                    os.remove(self._stash_path)
            finally:
                self._stash_path = None

    def to_file(self, fname: PathLike) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        fname : PathLike
            Full path to the .yaml or .json file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """
        task_id_cached = self._cached_properties.get("task_id")
        self = self.updated_copy(task_id_cached=task_id_cached)
        super(Job, self).to_file(fname=fname)  # noqa: UP008

    def run(
        self,
        path: PathLike | None = None,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
    ) -> WorkflowDataType:
        """Run :class:`Job` all the way through and return data.

        Parameters
        ----------
        path : Optional[PathLike] = None
            Path to download results file (.hdf5), including filename. When ``None``, a task-type-
            specific default filename is used.
        priority: int = None
            Priority of the simulation in the Virtual GPU (vGPU) queue (1 = lowest, 10 = highest).
            It affects only simulations from vGPU licenses and does not impact simulations using FlexCredits.
        vgpu_allocation : int = None
            Number of virtual GPUs to allocate for the simulation (1, 2, 4, or 8).
            Only applies to vGPU license users. If not specified, the system
            automatically determines the optimal GPU count.
        ignore_memory_limit : Optional[bool] = None
            If ``True``, allows the simulation to run even when estimated vGPU memory
            exceeds the allocation limit (up to 2x the limit). Only applies to
            vGPU license users. Default ``None`` leaves the server behaviour unchanged.

        Returns
        -------
        :class:`WorkflowDataType`
            Object containing simulation results.
        """
        if path is not None:
            self._check_path_dir(path=path)

        loaded_from_cache = self.load_if_cached
        if not loaded_from_cache:
            self.upload()
            self.start(
                priority=priority,
                vgpu_allocation=vgpu_allocation,
                ignore_memory_limit=ignore_memory_limit,
            )
            self.monitor()
        data = self.load(path=path)

        return data

    @cached_property
    def load_if_cached(self) -> bool:
        """Checks if results are cached and (if yes) restores them into our shared stash file."""
        # use temporary path as final destination is unknown
        stash_path = self._stash_path_for_job

        restored, cached_task_id = restore_simulation_if_cached(
            simulation=self.simulation,
            path=stash_path,
            reduce_simulation=self.reduce_simulation,
            verbose=self.verbose,
        )
        self._cached_task_id = cached_task_id

        if restored is None:
            return False

        self._stash_path = stash_path
        atexit.register(self.clear_stash)
        return True

    @cached_property
    def task_id(self) -> TaskId:
        """The task ID for this ``Job``. Uploads the ``Job`` if it hasn't already been uploaded."""
        if self.load_if_cached:
            return self._cached_task_id
        task_id = self._cached_properties.get("task_id")
        if task_id is not None:
            return task_id
        if self.task_id_cached:
            return self.task_id_cached
        self._check_folder(self.folder_name)
        return self._upload(verbose_estimate_cost=False)

    def _upload(
        self,
        verbose_estimate_cost: bool | None = None,
    ) -> TaskId:
        """Upload this job and return the task ID for handling."""
        # upload kwargs with all fields except task_id
        upload_kwargs = {key: getattr(self, key) for key in self._upload_fields}
        if verbose_estimate_cost is not None:
            upload_kwargs["verbose_estimate_cost"] = verbose_estimate_cost
        task_id = web.upload(**upload_kwargs)
        return task_id

    def upload(self) -> None:
        """Upload this ``Job`` if not already got cached results."""
        if self.load_if_cached:
            return
        cached_task_id = self._cached_properties.get("task_id")
        if cached_task_id is not None or self.task_id_cached:
            return

        self._check_folder(self.folder_name)
        task_id = self._upload(
            verbose_estimate_cost=self.verbose,
        )
        self._cached_properties["task_id"] = task_id

    def get_info(self) -> TaskInfo:
        """Return information about a :class:`Job`.

        Returns
        -------
        :class:`TaskInfo`
            :class:`TaskInfo` object containing info about status, size, credits of task and others.
        """
        return web.get_info(task_id=self.task_id)

    @property
    def status(self) -> str:
        """Return current status of :class:`Job`."""
        if self.load_if_cached:
            return "success"
        return self.get_info().status

    def start(
        self,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
    ) -> None:
        """Start running a :class:`Job`.

        Parameters
        ----------

        priority: int = None
            Priority of the simulation in the Virtual GPU (vGPU) queue (1 = lowest, 10 = highest).
            It affects only simulations from vGPU licenses and does not impact simulations using FlexCredits.
        vgpu_allocation : int = None
            Number of virtual GPUs to allocate for the simulation (1, 2, 4, or 8).
            Only applies to vGPU license users. If not specified, the system
            automatically determines the optimal GPU count.
        ignore_memory_limit : Optional[bool] = None
            If ``True``, allows the simulation to run even when estimated vGPU memory
            exceeds the allocation limit (up to 2x the limit). Only applies to
            vGPU license users. Default ``None`` leaves the server behaviour unchanged.

        Note
        ----
        To monitor progress of the :class:`Job`, call :meth:`Job.monitor` after started.
        Function has no effect if cache is enabled and data was found in cache.
        """
        loaded = self.load_if_cached
        if not loaded:
            web.start(
                self.task_id,
                solver_version=self.solver_version,
                pay_type=self.pay_type,
                priority=priority,
                vgpu_allocation=vgpu_allocation,
                ignore_memory_limit=ignore_memory_limit,
            )

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
        To load the output of completed simulation into :class:`~tidy3d.SimulationData` objects,
        call :meth:`Job.load`.
        """
        if self.load_if_cached:
            return
        web.monitor(self.task_id, verbose=self.verbose)

    def download(self, path: PathLike | None = None) -> None:
        """Download results of simulation.

        Parameters
        ----------
        path : Optional[PathLike] = None
            Path to download data as ``.hdf5`` file (including filename). When ``None``, a task-
            type-specific default filename is used.

        Note
        ----
        To load the data after download, use :meth:`Job.load`.
        """
        if self.load_if_cached:
            target_path = web._resolve_output_path(path, self._task_type_hint())
            self._check_path_dir(path=target_path)
            self._materialize_from_stash(target_path)
            return
        if path is not None:
            self._check_path_dir(path=path)
        web.download(task_id=self.task_id, path=path, verbose=self.verbose)

    def load(self, path: PathLike | None = None) -> WorkflowDataType:
        """Download job results and load them into a data object.

        Parameters
        ----------
        path : Optional[PathLike] = None
            Path to download data as ``.hdf5`` file (including filename). When ``None``, a task-
            type-specific default filename is used.

        Returns
        -------
        Union[:class:`~tidy3d.SimulationData`, :class:`~tidy3d.HeatSimulationData`,
        :class:`~tidy3d.EMESimulationData`]
            Object containing simulation results.
        """
        resolved_path = Path(path) if path is not None else None
        if self.load_if_cached:
            resolved_path = web._resolve_output_path(resolved_path, self._task_type_hint())
            self._check_path_dir(path=resolved_path)
            self._materialize_from_stash(resolved_path)
        else:
            if resolved_path is None:
                resolved_path = web._resolve_output_path(None, self._task_type_hint())
            self._check_path_dir(path=resolved_path)

        data = web.load(
            task_id=None if self.load_if_cached else self.task_id,
            path=resolved_path,
            verbose=self.verbose,
            lazy=self.lazy,
        )
        if isinstance(self.simulation, ModeSolver):
            if not self.load_if_cached:
                _store_mode_solver_in_cache(
                    self.task_id,
                    self.simulation,
                    data,
                    resolved_path,
                )
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
        if self.load_if_cached:
            return 0.0
        return web.estimate_cost(self.task_id, verbose=verbose, solver_version=self.solver_version)

    @staticmethod
    def _check_path_dir(path: PathLike) -> None:
        """Make sure parent directory of ``path`` exists and create it if not.

        Parameters
        ----------
        path : PathLike
            Path to file to be created (including filename).
        """
        path = Path(path)
        parent_dir = path.parent
        if parent_dir != Path(".") and not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="before")
    @classmethod
    def set_task_name_if_none(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Auto-assign a task_name if user did not provide one.
        """
        if not isinstance(data, dict):
            return data

        if data.get("task_name") is None:
            sim = data.get("simulation")
            stub = Tidy3dStub(simulation=sim)
            data["task_name"] = stub.get_default_task_name()
        return data


class BatchData(Tidy3dBaseModel, Mapping):
    """
    Holds a collection of :class:`~tidy3d.SimulationData` returned by :class:`Batch`.

    Notes
    -----

        When the batch is completed, the output is not a :class:`~tidy3d.SimulationData` but rather a
        :class:`BatchData`. The data within this :class:`BatchData` object can either be indexed
        directly ``batch_results[task_name]`` or can be looped through ``batch_results.items()`` to
        get the :class:`~tidy3d.SimulationData` for each task.
        Converting with ``dict(batch_results.items())`` eagerly touches all tasks and can load all
        results.

    See Also
    --------

    :class:`Batch`:
         Interface for submitting several :class:`~tidy3d.Simulation` objects to sever.

    :class:`~tidy3d.SimulationData`:
         Stores data from a collection of :class:`~tidy3d.Monitor` objects in a
         :class:`~tidy3d.Simulation`.

    **Notebooks**
        * `Running simulations through the cloud <../../notebooks/WebAPI.html>`_
        * `Performing parallel / batch processing of simulations <../../notebooks/ParameterScan.html>`_
    """

    task_paths: dict[TaskName, str] = Field(
        title="Data Paths",
        description="Mapping of task_name to path to corresponding data for each task in batch.",
    )

    task_ids: dict[TaskName, str] = Field(
        title="Task IDs",
        description="Mapping of task_name to task_id for each task in batch.",
    )

    verbose: bool = Field(
        True,
        title="Verbose",
        description="Whether to print info messages and progressbars.",
    )
    cached_tasks: dict[TaskName, bool] | None = Field(
        None,
        title="Cached Tasks",
        description="Whether the data of a task came from the cache.",
    )

    lazy: bool = Field(
        False,
        title="Lazy",
        description="Whether to load the actual data (lazy=False) or return a proxy that loads the data when accessed (lazy=True).",
    )

    is_downloaded: bool | None = Field(
        False,
        title="Is Downloaded",
        description="Whether the simulation data was downloaded before.",
    )

    _data_cache: dict[TaskName, WorkflowDataType] = PrivateAttr(default_factory=dict)
    _cache_enabled: bool | None = PrivateAttr(default=None)

    def _should_cache_data(self) -> bool:
        """Return True when in-memory caching should be enabled for batch data."""
        if self._cache_enabled is not None:
            return self._cache_enabled

        self._cache_enabled = False
        if WASM_BUILD:
            return False

        try:
            cache_config = config.batch_data_cache
        except AttributeError:
            return False
        if not cache_config.enabled:
            return False

        max_bytes = int(cache_config.max_total_size_gb * (1024**3))
        if max_bytes <= 0:
            return False

        total_size = 0
        for task_path in self.task_paths.values():
            try:
                file_size = Path(task_path).stat().st_size
            except FileNotFoundError:  # not downloaded yet
                self._cache_enabled = None
                return False
            total_size += file_size
            if total_size > max_bytes:
                return False

        self._cache_enabled = True
        return True

    def load_sim_data(self, task_name: str) -> WorkflowDataType:
        """Load a simulation data object from file by task name.

        When ``config.batch_data_cache.enabled`` is ``True`` and the total size of all task
        files stays under the configured threshold, the loaded object is cached in
        memory for subsequent accesses.
        """
        cache_enabled = self._should_cache_data()

        def _load() -> WorkflowDataType:
            task_data_path = Path(self.task_paths[task_name])
            task_id = self.task_ids[task_name]
            from_cache = self.cached_tasks[task_name] if self.cached_tasks else False

            return web.load(
                task_id=None if from_cache else task_id,
                path=task_data_path,
                verbose=False,
                replace_existing=not (from_cache or self.is_downloaded),
                lazy=self.lazy,
            )

        if cache_enabled:
            return self._get_cached_value_by_key("load_sim_data", task_name, _load)

        data = _load()

        if not cache_enabled and self._cache_enabled is None:
            cache_enabled = self._should_cache_data()
        if cache_enabled:
            return self._get_cached_value_by_key("load_sim_data", task_name, lambda: data)
        return data

    def __getitem__(self, task_name: TaskName) -> WorkflowDataType:
        """Get the simulation data object for a given ``task_name``.

        When ``config.batch_data_cache.enabled`` is `True` and the batch data size is within
        the configured threshold, the result is cached in memory.
        """
        return self.load_sim_data(task_name)

    def __iter__(self) -> Iterator[TaskName]:
        """Iterate over the task names."""
        return iter(self.task_paths)

    def __len__(self) -> int:
        """Return the number of tasks in the batch."""
        return len(self.task_paths)

    @classmethod
    def load(
        cls, path_dir: PathLike = DEFAULT_DATA_DIR, replace_existing: bool = False
    ) -> BatchData:
        """Load :class:`Batch` from file, download results, and load them.

        Parameters
        ----------
        path_dir : PathLike = './'
            Base directory where data will be downloaded, by default current working directory.
            A `batch.hdf5` file must be present in the directory.
        replace_existing : bool = False
            Downloads the data even if path exists (overwriting the existing).

        Returns
        ------
        :class:`BatchData`
            Contains Union[:class:`~tidy3d.SimulationData`, :class:`~tidy3d.HeatSimulationData`,
            :class:`~tidy3d.EMESimulationData`] for each Union[:class:`~tidy3d.Simulation`,
            :class:`~tidy3d.HeatSimulation`, :class:`~tidy3d.EMESimulation`] in :class:`Batch`.
        """
        base_dir = Path(path_dir)
        batch_file = Batch._batch_path(path_dir=base_dir)
        batch = Batch.from_file(batch_file)
        return batch.load(path_dir=base_dir, replace_existing=replace_existing)


class Batch(WebContainer):
    """
    Interface for submitting several :class:`~tidy3d.Simulation` objects to sever.

    Notes
    -----

        Commonly one needs to submit a batch of :class:`~tidy3d.Simulation`. The built-in
        :class:`Batch` object is the best way to upload, start, monitor, and load a series of
        tasks. The batch object is like a :class:`Job`, but stores task metadata for a series of
        simulations.

    See Also
    --------

    :meth:`tidy3d.web.api.webapi.run_async`
        Submits a set of :class:`~tidy3d.Simulation` objects to server, starts running, monitors progress,
        downloads, and loads results as a :class:`.BatchData` object.

    :class:`Job`:
        Interface for managing the running of a Simulation on server.

    **Notebooks**
        * `Running simulations through the cloud <../../notebooks/WebAPI.html>`_
        * `Performing parallel / batch processing of simulations <../../notebooks/ParameterScan.html>`_
        * `Inverse taper edge coupler <../../notebooks/EdgeCoupler.html>`_
    """

    simulations: (
        dict[TaskName, discriminated_union(WorkflowType)]
        | tuple[discriminated_union(WorkflowType), ...]
    ) = Field(
        title="Simulations",
        description="Mapping of task names to Simulations to run as a batch.",
    )

    folder_name: str = Field(
        "default",
        title="Folder Name",
        description="Name of folder to store member of each batch on web UI.",
    )

    verbose: bool = Field(
        True,
        title="Verbose",
        description="Whether to print info messages and progressbars.",
    )

    solver_version: str | None = Field(
        None,
        title="Solver Version",
        description="Custom solver version to use, "
        "otherwise uses default for the current front end version.",
    )

    callback_url: str | None = Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event. "
        "The body content is a json file with fields "
        "``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.",
    )

    simulation_type: BatchCategoryType | None = Field(
        None,
        title="Simulation Type",
        description="Type of each simulation in the batch, used internally only.",
    )

    parent_tasks: dict[str, tuple[TaskId, ...]] | None = Field(
        None,
        title="Parent Tasks",
        description="Collection of parent task ids for each job in batch, used internally only.",
    )

    num_workers: PositiveInt | None = Field(
        default_factory=_default_batch_num_workers,
        title="Number of Workers",
        description="Number of workers for batch multi-threading where configurable. "
        "Corresponds to ``max_workers`` argument passed to "
        "``concurrent.futures.ThreadPoolExecutor``. Upload/start use a fixed "
        f"concurrency of {UPLOAD_START_NUM_WORKERS}. Defaults to "
        "``config.web.default_num_workers``.",
    )

    reduce_simulation: Literal["auto", True, False] = Field(
        "auto",
        title="Reduce Simulation",
        description="Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.",
    )

    pay_type: PayType | None = Field(
        None,
        title="Payment Type",
        description="Specify the payment method.",
    )

    jobs_cached: dict[TaskName, Job] | None = Field(
        None,
        title="Jobs (Cached)",
        description="Optional field to specify ``jobs``. Only used as a workaround internally "
        "so that ``jobs`` is written when ``Batch.to_file()`` and then the proper task is loaded "
        "from ``Batch.from_file()``. We recommend leaving unset as setting this field along with "
        "fields that were not used to create the task will cause errors.",
    )

    lazy: bool = Field(
        False,
        title="Lazy",
        description="Whether to load the actual data (lazy=False) or return a proxy that loads the data when accessed (lazy=True).",
    )

    _job_type: type = PrivateAttr(Job)
    _terminal_status_by_task: dict[TaskName, str] = PrivateAttr(default_factory=dict)
    _terminal_task_id_by_task: dict[TaskName, TaskId] = PrivateAttr(default_factory=dict)

    @field_validator("simulations", mode="before")
    @classmethod
    def _validate_simulation_keys_are_task_names(cls, simulations: Any) -> Any:
        """Ensure mapping keys are task-name strings with a concise error message."""
        if not isinstance(simulations, Mapping):
            return simulations

        for task_name in simulations:
            if not isinstance(task_name, str):
                raise ValueError(
                    "Batch simulations keys must be strings (task names). "
                    f"Got key {task_name!r} of type {type(task_name).__name__!r}. "
                    "Use explicit string keys, for example: simulations[str(i)] = simulation."
                )

        return simulations

    def run(
        self,
        path_dir: PathLike = DEFAULT_DATA_DIR,
        priority: int | None = None,
        replace_existing: bool = False,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
    ) -> BatchData:
        """Upload and run each simulation in :class:`Batch`.

        Parameters
        ----------
        path_dir : PathLike
            Base directory where data will be downloaded, by default current working directory.
        priority: int = None
            Priority of the simulation in the Virtual GPU (vGPU) queue (1 = lowest, 10 = highest).
            It affects only simulations from vGPU licenses and does not impact simulations using FlexCredits.
        replace_existing : bool = False
            Downloads the data even if path exists (overwriting the existing). Applies when
            downloading cached results or when `download_on_success=True`.
        vgpu_allocation : int = None
            Number of virtual GPUs to allocate for the simulation (1, 2, 4, or 8).
            Only applies to vGPU license users. If not specified, the system
            automatically determines the optimal GPU count.
        ignore_memory_limit : Optional[bool] = None
            If ``True``, allows the simulation to run even when estimated vGPU memory
            exceeds the allocation limit (up to 2x the limit). Only applies to
            vGPU license users. Default ``None`` leaves the server behaviour unchanged.

        Returns
        ------
        :class:`BatchData`
            Contains Union[:class:`~tidy3d.SimulationData`, :class:`~tidy3d.HeatSimulationData`,
            :class:`~tidy3d.EMESimulationData`] for each Union[:class:`~tidy3d.Simulation`,
            :class:`~tidy3d.HeatSimulation`, :class:`~tidy3d.EMESimulation`] in :class:`Batch`.

        Note
        ----
        A typical usage might look like:

        >>> from tidy3d.web.api.container import Batch
        >>> custom_batch = Batch()
        >>> batch_data = custom_batch.run() # doctest: +SKIP
        >>> for task_name, sim_data in batch_data.items(): # doctest: +SKIP
        ...     # do something with data. # doctest: +SKIP

        ``batch_data`` iterates over task names and loads the corresponding data
        from file one by one. If no file exists for that task, it downloads it.
        When ``config.batch_data_cache.enabled`` is ``True`` and the
        total size of all task files is below `config.batch_data_cache.max_total_size_gb`,
        accessed results are cached in memory to avoid repeated loads.
        """
        loaded = [job.load_if_cached for job in self.jobs.values()]
        self._check_path_dir(path_dir)
        if not all(loaded):
            self.upload()
            self.to_file(self._batch_path(path_dir=path_dir))
            self.start(
                priority=priority,
                vgpu_allocation=vgpu_allocation,
                ignore_memory_limit=ignore_memory_limit,
            )
            self.monitor(
                path_dir=path_dir,
                download_on_success=True,
                replace_existing=replace_existing,
            )
        else:
            if self.verbose:
                console = get_logging_console()
                console.log("Found all simulations in cache.")
            self.download(path_dir=path_dir, replace_existing=replace_existing)  # moves cache files
        return self.load(path_dir=path_dir, skip_download=True)

    @cached_property
    def jobs(self) -> dict[TaskName, Job]:
        """Create a series of tasks in the :class:`.Batch` and upload them to server.

        Note
        ----
        To start the simulations running, must call :meth:`Batch.start` after uploaded.
        """

        if self.jobs_cached is not None:
            return self.jobs_cached

        if isinstance(self.simulations, tuple):
            simulations = {}
            for i, sim in enumerate(self.simulations, 1):
                stub = Tidy3dStub(simulation=sim)
                task_name = stub.get_default_task_name() + f"_{i}"
                simulations[task_name] = sim
        else:
            simulations = self.simulations

        # the type of job to upload (to generalize to subclasses)
        JobType = self._job_type
        self_dict = self.model_dump()

        jobs = {}
        for task_name, simulation in simulations.items():
            job_kwargs = {}

            for key in JobType._upload_fields.default:
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

    def to_file(self, fname: PathLike) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        fname : PathLike
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

    @classmethod
    def from_file(
        cls,
        fname: PathLike,
        group_path: str | None = None,
        lazy: bool = False,
        on_load: Callable[[Any], None] | None = None,
        **parse_obj_kwargs: Any,
    ) -> Batch:
        """Load a :class:`Batch` from file.

        Notes
        -----
        For :class:`Batch`, ``lazy=True`` also configures per-task data loading behavior used by
        :meth:`Batch.load`.
        """
        if not lazy:
            return super().from_file(
                fname=fname,
                group_path=group_path,
                lazy=lazy,
                on_load=on_load,
                **parse_obj_kwargs,
            )

        def _set_batch_lazy_and_run_callback(loaded_obj: Any) -> None:
            # Batch models are frozen; set laziness via object.__setattr__ on materialization.
            object.__setattr__(loaded_obj, "lazy", True)
            if on_load is not None:
                on_load(loaded_obj)

        return super().from_file(
            fname=fname,
            group_path=group_path,
            lazy=lazy,
            on_load=_set_batch_lazy_and_run_callback,
            **parse_obj_kwargs,
        )

    @property
    def num_jobs(self) -> int:
        """Number of jobs in the batch."""
        return len(self.jobs)

    def _partition_jobs_by_cache(self) -> tuple[list[Job], list[Job]]:
        """Return ``(cached_jobs, uncached_jobs)`` for the current batch."""
        jobs_from_cache = []
        jobs_uncached = []
        for job in self.jobs.values():
            if job.load_if_cached:
                jobs_from_cache.append(job)
            else:
                jobs_uncached.append(job)
        return jobs_from_cache, jobs_uncached

    def _log_cached_jobs(self, jobs_from_cache: list[Job]) -> None:
        """Log how many jobs were restored from cache."""
        if not self.verbose:
            return

        n_cached = len(jobs_from_cache)
        if n_cached <= 0:
            return

        console = get_logging_console()
        console.log(f"Got {n_cached} simulation{'s' if n_cached > 1 else ''} from cache.")

    def _prepare_uncached_jobs(
        self,
        *,
        check_folder: bool = False,
        log_cached_jobs: bool = False,
    ) -> list[Job]:
        """Prepare uncached jobs with shared cache/folder handling."""
        if check_folder:
            self._check_folder(self.folder_name)

        jobs_from_cache, jobs_uncached = self._partition_jobs_by_cache()
        if log_cached_jobs:
            self._log_cached_jobs(jobs_from_cache)

        return jobs_uncached

    @staticmethod
    def _cached_fallback_task_id(task_name: TaskName, job: Job) -> str:
        """Filesystem-safe fallback ID for cached jobs without known server task IDs."""
        simulation_hash = job.simulation._hash_self()
        task_name_hash = hashlib.md5(str(task_name).encode("utf-8")).hexdigest()
        return f"cached_{simulation_hash}_{task_name_hash}"

    def upload(self) -> None:
        """Upload a series of tasks associated with this ``Batch`` using multi-threading."""
        jobs_to_upload = self._prepare_uncached_jobs(
            check_folder=True,
            log_cached_jobs=True,
        )
        with ThreadPoolExecutor(max_workers=UPLOAD_START_NUM_WORKERS) as executor:
            upload_futures: dict[concurrent.futures.Future[Any], Job] = {}
            for job in jobs_to_upload:
                fut = executor.submit(job.upload)
                upload_futures[fut] = job

            if len(upload_futures) == 0:
                return

            if self.verbose:
                console = get_logging_console()
                progress_columns = (
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                )
                with Progress(*progress_columns, console=console, transient=False) as progress:
                    pbar_message = (
                        f"Uploading data for {len(jobs_to_upload)} "
                        f"task{'s' if len(jobs_to_upload) > 1 else ''}"
                    )
                    pbar = progress.add_task(pbar_message, total=len(jobs_to_upload))
                    completed = 0
                    for fut in concurrent.futures.as_completed(upload_futures):
                        job = upload_futures[fut]
                        try:
                            fut.result()
                        except Exception as exc:
                            task_name = getattr(job, "task_name", "<unknown>")
                            log.error(
                                f"Failed to upload task '{task_name}': "
                                f"{exc.__class__.__name__}: {exc}"
                            )
                            raise
                        completed += 1
                        progress.update(pbar, completed=completed)

                    progress.refresh()
                    time.sleep(BATCH_PROGRESS_REFRESH_TIME)
            else:
                for fut in concurrent.futures.as_completed(upload_futures):
                    job = upload_futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        task_name = getattr(job, "task_name", "<unknown>")
                        log.error(
                            f"Failed to upload task '{task_name}': {exc.__class__.__name__}: {exc}"
                        )
                        raise

    def get_info(self) -> dict[TaskName, TaskInfo]:
        """Get information about each task in the :class:`Batch`.

        Returns
        -------
        dict[str, :class:`TaskInfo`]
            Mapping of task name to data about task associated with each task.
        """
        info_dict = {}
        for task_name, job in self.jobs.items():
            task_info = job.get_info()
            info_dict[task_name] = task_info
        return info_dict

    def start(
        self,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
    ) -> None:
        """Start running all tasks in the :class:`Batch`.

        Parameters
        ----------

        priority: int = None
            Priority of the simulation in the Virtual GPU (vGPU) queue (1 = lowest, 10 = highest).
            It affects only simulations from vGPU licenses and does not impact simulations using FlexCredits.
        vgpu_allocation : int = None
            Number of virtual GPUs to allocate for the simulation (1, 2, 4, or 8).
            Only applies to vGPU license users. If not specified, the system
            automatically determines the optimal GPU count.
        ignore_memory_limit : Optional[bool] = None
            If ``True``, allows the simulation to run even when estimated vGPU memory
            exceeds the allocation limit (up to 2x the limit). Only applies to
            vGPU license users. Default ``None`` leaves the server behaviour unchanged.

        Note
        ----
        To monitor the running simulations, can call :meth:`Batch.monitor`.
        """
        if self.verbose:
            console = get_logging_console()
            console.log(f"Started working on Batch containing {self.num_jobs} tasks.")
        jobs_to_start = self._prepare_uncached_jobs()
        with ThreadPoolExecutor(max_workers=UPLOAD_START_NUM_WORKERS) as executor:
            start_futures: dict[concurrent.futures.Future[Any], Job] = {}
            for job in jobs_to_start:
                fut = executor.submit(
                    job.start,
                    priority=priority,
                    vgpu_allocation=vgpu_allocation,
                    ignore_memory_limit=ignore_memory_limit,
                )
                start_futures[fut] = job

            for fut in concurrent.futures.as_completed(start_futures):
                job = start_futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    task_name = getattr(job, "task_name", "<unknown>")
                    log.error(
                        f"Failed to start task '{task_name}': {exc.__class__.__name__}: {exc}"
                    )
                    raise

    def get_run_info(self) -> dict[TaskName, RunInfo]:
        """get information about a each of the tasks in the :class:`Batch`.

        Returns
        -------
        dict[str: :class:`RunInfo`]
            Maps task names to run info for each task in the :class:`Batch`.
        """
        run_info_dict = {}
        for task_name, job in self.jobs.items():
            run_info = job.get_run_info()
            run_info_dict[task_name] = run_info
        return run_info_dict

    def monitor(
        self,
        *,
        download_on_success: bool = False,
        path_dir: PathLike = DEFAULT_DATA_DIR,
        replace_existing: bool = False,
    ) -> None:
        """
        Monitor progress of each running task.

        - Optionally downloads results as soon as a job reaches final success.
        - Rich progress bars in verbose mode; quiet polling otherwise.


        Parameters
        ----------
        download_on_success : bool = False
            If ``True``, automatically start downloading the results for a job as soon as it reaches
            ``success``.
        path_dir : PathLike = './'
            Base directory where data will be downloaded, by default the current working directory.
            Only used when ``download_on_success`` is ``True``.
        replace_existing : bool = False
            Downloads the data even if path exists (overwriting the existing). Only used when
            ``download_on_success`` is ``True``.
        """
        jobs = self.jobs
        jobs_items = list(jobs.items())
        active_task_names = set(jobs)
        self._terminal_status_by_task = {
            task_name: status
            for task_name, status in self._terminal_status_by_task.items()
            if task_name in active_task_names and status in END_STATES
        }
        self._terminal_task_id_by_task = {
            task_name: task_id
            for task_name, task_id in self._terminal_task_id_by_task.items()
            if task_name in active_task_names and task_id is not None
        }
        status_by_task = dict(self._terminal_status_by_task)

        # ----- download scheduling ---------------------------------------------------
        downloads_started: set[str] = set()
        download_futures: dict[TaskId, concurrent.futures.Future] = {}
        download_executor: ThreadPoolExecutor | None = None

        if download_on_success:
            self._check_path_dir(path_dir=path_dir)
            download_executor = ThreadPoolExecutor(max_workers=self.num_workers)

        def _remember_terminal_status(task_name: TaskName, job: Job, status: str) -> None:
            if status not in END_STATES:
                return

            status_by_task[task_name] = status
            self._terminal_status_by_task[task_name] = status
            if status in ERROR_STATES:
                return

            # Keep task IDs around to avoid re-querying status/id in a following load().
            task_id = self._terminal_task_id_by_task.get(task_name)
            if task_id is None:
                task_id = job.task_id
            if task_id is None:
                self._terminal_task_id_by_task.pop(task_name, None)
                return
            self._terminal_task_id_by_task[task_name] = task_id

        def _get_status(task_name: TaskName, job: Job) -> str:
            cached_status = status_by_task.get(task_name)
            if cached_status in END_STATES:
                return cached_status

            status = job.status
            status_by_task[task_name] = status
            _remember_terminal_status(task_name, job, status)
            return status

        def schedule_download(
            task_name: TaskName,
            job: Job,
            status: str | None = None,
        ) -> None:
            if download_executor is None:
                return

            if status is None:
                status = _get_status(task_name, job)
            if status not in COMPLETED_STATES:
                return

            task_id = self._terminal_task_id_by_task.get(task_name)
            if task_id is None:
                task_id = job.task_id
            if task_id is None:
                if not job.load_if_cached:
                    return
                task_id = self._cached_fallback_task_id(task_name, job)
            if task_id in downloads_started:
                return

            job_path = self._job_data_path(task_id=task_id, path_dir=path_dir)
            if job_path.exists():
                if not replace_existing:
                    downloads_started.add(task_id)
                    log.info(
                        f"File '{job_path}' already exists. Skipping download "
                        "(set `replace_existing=True` to overwrite)."
                    )
                    return
                log.info(f"File '{job_path}' already exists. Overwriting.")

            downloads_started.add(task_id)
            download_futures[task_id] = download_executor.submit(job.download, job_path)

        # ----- continue condition & status formatting -------------------------------
        def check_continue_condition(task_name: TaskName, job: Job) -> bool:
            if job.load_if_cached:
                _remember_terminal_status(task_name, job, "success")
                return False
            return _get_status(task_name, job) not in END_STATES

        def pbar_description(
            task_name: str, status: str, max_name_length: int, status_width: int
        ) -> str:
            if len(task_name) > max_name_length - 3:
                task_name = task_name[: (max_name_length - 3)] + "..."
            task_part = f"{task_name:<{max_name_length}}"

            if status in ERROR_STATES:
                status_part = f"→ [red]{status:<{status_width}}"
            elif status in COMPLETED_STATES:
                status_part = f"→ [green]{status:<{status_width}}"
            elif status in (PRE_ERROR_STATES | DRAFT_STATES | QUEUED_STATES):
                status_part = f"→ [yellow]{status:<{status_width}}"
            elif status in RUNNING_STATES:
                status_part = f"→ [blue]{status:<{status_width}}"
            else:
                status_part = f"→ {status:<{status_width}}"
            return f"{task_part} {status_part}"

        max_task_name = max(len(task_name) for task_name in self.jobs.keys())
        max_name_length = min(30, max(max_task_name, 15))

        try:
            console = None
            progress_columns = []
            if self.verbose:
                console = get_logging_console()
                self.estimate_cost()
                console.log(
                    "Use 'Batch.real_cost()' to get the billed FlexCredit cost after completion."
                )

                progress_columns = (
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=25),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                )

            with Progress(
                *progress_columns, console=console, transient=False, disable=not self.verbose
            ) as progress:
                pbar_tasks: dict[str, TaskID] = {}
                for task_name, job in jobs_items:
                    status = status_by_task.get(task_name)
                    if self.verbose:
                        display_status = status
                        if job.load_if_cached:
                            status = "success"
                            display_status = status
                            completed = COMPLETED_PERCENT
                            _remember_terminal_status(task_name, job, status)
                        elif status in END_STATES:
                            display_status = status
                            completed = STATE_PROGRESS_PERCENTAGE.get(status, COMPLETED_PERCENT)
                            _remember_terminal_status(task_name, job, status)
                        else:
                            info = job.get_info()
                            status = info.status
                            display_status = status
                            status_by_task[task_name] = status
                            _remember_terminal_status(task_name, job, status)
                            if isinstance(info, BatchDetail):
                                display_status, _, completed = _batch_detail_progress(info)
                            else:
                                completed = STATE_PROGRESS_PERCENTAGE.get(status, 0)
                        schedule_download(task_name, job, status=status)
                        desc = pbar_description(task_name, display_status, max_name_length, 0)
                        pbar_tasks[task_name] = progress.add_task(
                            desc, total=COMPLETED_PERCENT, completed=completed
                        )
                    else:
                        if status is None:
                            if job.load_if_cached:
                                status = "success"
                                _remember_terminal_status(task_name, job, status)
                            else:
                                status = _get_status(task_name, job)
                        elif status in END_STATES:
                            _remember_terminal_status(task_name, job, status)
                        schedule_download(task_name, job, status=status)

                while any(
                    check_continue_condition(task_name, job) for task_name, job in jobs_items
                ):
                    for task_name, job in jobs_items:
                        if job.load_if_cached:
                            continue
                        status = status_by_task.get(task_name)
                        if status in END_STATES:
                            schedule_download(task_name, job, status=status)
                            continue

                        info = job.get_info()
                        status = info.status
                        status_by_task[task_name] = status
                        _remember_terminal_status(task_name, job, status)

                        schedule_download(task_name, job, status=status)

                        if self.verbose:
                            # choose display status & percent
                            if isinstance(info, BatchDetail):
                                display_status, _, pct = _batch_detail_progress(info)
                            elif status != "run_success":
                                display_status = status
                                pct = STATE_PROGRESS_PERCENTAGE.get(status, 0)
                            else:
                                post_st = getattr(job, "postprocess_status", None)
                                if post_st in END_STATES:
                                    display_status = post_st
                                    pct = STATE_PROGRESS_PERCENTAGE.get(post_st, 0)
                                else:
                                    display_status = "postprocess"
                                    pct = STATE_PROGRESS_PERCENTAGE.get("postprocess", 0)

                            pbar = pbar_tasks[task_name]
                            desc = pbar_description(task_name, display_status, max_name_length, 0)
                            progress.update(pbar, description=desc, completed=pct)
                    if self.verbose:
                        progress.refresh()
                        time.sleep(BATCH_PROGRESS_REFRESH_TIME)
                    else:
                        time.sleep(web.REFRESH_TIME)

                # final render to terminal state for all bars
                for task_name, job in jobs_items:
                    status = status_by_task.get(task_name)
                    if status is None:
                        if job.load_if_cached:
                            status = "success"
                            _remember_terminal_status(task_name, job, status)
                        else:
                            status = _get_status(task_name, job)
                    schedule_download(task_name, job, status=status)

                    if self.verbose:
                        if job.load_if_cached:
                            display_status = "success"
                            pct = COMPLETED_PERCENT
                        elif status in END_STATES:
                            display_status = status
                            pct = STATE_PROGRESS_PERCENTAGE.get(status, COMPLETED_PERCENT)
                        else:
                            info = job.get_info()
                            status = info.status
                            status_by_task[task_name] = status
                            _remember_terminal_status(task_name, job, status)
                            if isinstance(info, BatchDetail):
                                display_status, _, pct = _batch_detail_progress(info)
                            elif status != "run_success":
                                display_status = status
                                pct = STATE_PROGRESS_PERCENTAGE.get(status, COMPLETED_PERCENT)
                            else:
                                post_st = getattr(job, "postprocess_status", None)
                                if post_st in END_STATES:
                                    display_status = post_st
                                    pct = STATE_PROGRESS_PERCENTAGE.get(post_st, COMPLETED_PERCENT)
                                else:
                                    display_status = "postprocess"
                                    pct = STATE_PROGRESS_PERCENTAGE.get(
                                        "postprocess", COMPLETED_PERCENT
                                    )

                        pbar = pbar_tasks[task_name]
                        desc = pbar_description(task_name, display_status, max_name_length, 0)
                        progress.update(pbar, description=desc, completed=pct)

                if self.verbose:
                    progress.refresh()
                    console.log("Batch complete.")
        finally:
            if download_executor is not None:
                try:
                    for fut in concurrent.futures.as_completed(download_futures.values()):
                        fut.result()
                finally:
                    download_executor.shutdown(wait=True)

    @staticmethod
    def _job_data_path(task_id: TaskId, path_dir: PathLike = DEFAULT_DATA_DIR) -> Path:
        """Default path to data of a single :class:`Job` in :class:`Batch`.

        Parameters
        ----------
        task_id : str
            task_id corresponding to a :class:`Job`.
        path_dir : PathLike = './'
            Base directory where data will be downloaded, by default, the current working directory.

        Returns
        -------
        Path
            Full path to the data file.
        """
        return Path(path_dir) / f"{task_id!s}.hdf5"

    @staticmethod
    def _batch_path(path_dir: PathLike = DEFAULT_DATA_DIR) -> Path:
        """Default path to save :class:`Batch` hdf5 file.

        Parameters
        ----------
        path_dir : PathLike = './'
            Base directory where the batch.hdf5 will be downloaded,
            by default, the current working directory.

        Returns
        -------
        Path
            Full path to the batch file.
        """
        return Path(path_dir) / "batch.hdf5"

    def download(
        self, path_dir: PathLike = DEFAULT_DATA_DIR, replace_existing: bool = False
    ) -> None:
        """Download results of each task.

        Parameters
        ----------
        path_dir : PathLike = './'
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
        self._check_path_dir(path_dir=path_dir)
        self.to_file(self._batch_path(path_dir=path_dir))

        def _task_id_for_download_path(task_name: TaskName, job: Job) -> str:
            task_id = job.task_id
            if task_id is None and job.load_if_cached:
                return self._cached_fallback_task_id(task_name, job)
            return str(task_id)

        # Warn about already-existing files if we won't overwrite them
        if not replace_existing:
            num_existing = sum(
                os.path.exists(
                    self._job_data_path(
                        task_id=_task_id_for_download_path(task_name, job),
                        path_dir=path_dir,
                    )
                )
                for task_name, job in self.jobs.items()
            )
            if num_existing > 0:
                files_plural = "files have" if num_existing > 1 else "file has"
                log.info(
                    f"{num_existing} {files_plural} already been downloaded and will be skipped. "
                    "To forcibly overwrite existing files, invoke the run, load, or download "
                    "function with `replace_existing=True`.",
                    log_once=True,
                )

        fns = []

        for task_name, job in self.jobs.items():
            status = self._terminal_status_by_task.get(task_name)
            if status is None:
                status = job.status
                if status in END_STATES:
                    self._terminal_status_by_task[task_name] = status

            if status in ERROR_STATES:
                log.warning(f"Not downloading '{task_name}' as the task errored.")
                continue

            task_id_for_path = _task_id_for_download_path(task_name, job)
            job_path = self._job_data_path(task_id=task_id_for_path, path_dir=path_dir)

            if job_path.exists():
                if replace_existing:
                    log.debug(f"File '{job_path}' already exists. Overwriting.")
                else:
                    log.debug(f"File '{job_path}' already exists. Skipping.")
                    continue

            if job.load_if_cached:
                job._materialize_from_stash(job_path)
                continue

            def fn(job: Job = job, job_path: PathLike = job_path) -> None:
                job.download(path=job_path)

            fns.append(fn)

        if not fns:
            return

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
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
                    for fut in concurrent.futures.as_completed(futures):
                        fut.result()
                        completed += 1
                        progress.update(pbar, completed=completed)
            else:
                # Still ensure completion if verbose is off
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()

    def load(
        self,
        path_dir: PathLike = DEFAULT_DATA_DIR,
        replace_existing: bool = False,
        skip_download: bool = False,
    ) -> BatchData:
        """Download results and load them into :class:`.BatchData` object.

        Parameters
        ----------
        path_dir : PathLike = './'
            Base directory where data will be downloaded, by default current working directory.
        replace_existing : bool = False
            Downloads the data even if path exists (overwriting the existing).
        skip_download : bool = False
            Does not trigger download. Should be True if already downloaded.

        Returns
        ------
        :class:`BatchData`
            Contains Union[:class:`~tidy3d.SimulationData`, :class:`~tidy3d.HeatSimulationData`,
            :class:`~tidy3d.EMESimulationData`] for each Union[:class:`~tidy3d.Simulation`,
            :class:`~tidy3d.HeatSimulation`, :class:`~tidy3d.EMESimulation`] in :class:`Batch`.

        The :class:`Batch` hdf5 file will be automatically saved as ``{path_dir}/batch.hdf5``,
        allowing one to load this :class:`Batch` later using ``batch = Batch.from_file()``.
        """
        self._check_path_dir(path_dir=path_dir)

        if self.jobs is None:
            raise DataError("Can't load batch results, hasn't been uploaded.")

        task_paths = {}
        task_ids = {}
        jobs_items = list(self.jobs.items())
        terminal_status_by_task = {
            task_name: status
            for task_name, status in self._terminal_status_by_task.items()
            if task_name in self.jobs and status in END_STATES
        }
        terminal_task_id_by_task = {
            task_name: task_id
            for task_name, task_id in self._terminal_task_id_by_task.items()
            if task_name in self.jobs and task_id is not None
        }

        def _known_task_id(task_name: TaskName, job: Job) -> TaskId | None:
            """Return a known task id without triggering uploads."""
            task_id = terminal_task_id_by_task.get(task_name)
            if task_id is not None:
                return task_id

            cached_properties = getattr(job, "_cached_properties", None)
            if isinstance(cached_properties, dict):
                task_id = cached_properties.get("task_id")
                if task_id is not None:
                    return task_id

            task_id_cached = getattr(job, "task_id_cached", None)
            if task_id_cached is not None:
                return task_id_cached

            if job.load_if_cached:
                return getattr(job, "_cached_task_id", None)

            return None

        status_by_task: dict[TaskName, str] = {}
        for task_name, job in jobs_items:
            status = terminal_status_by_task.get(task_name)
            task_id = _known_task_id(task_name, job)
            loaded_from_cache_job = job.load_if_cached

            if status is None:
                if loaded_from_cache_job:
                    status = "success"
                elif task_id is None:
                    log.warning(f"Not loading '{task_name}' as the task hasn't been uploaded.")
                    continue
                else:
                    status = job.status

            status_by_task[task_name] = status
            if status in END_STATES:
                self._terminal_status_by_task[task_name] = status

            if status in ERROR_STATES:
                log.warning(f"Not loading '{task_name}' as the task errored.")
                continue

            if task_id is None and not loaded_from_cache_job:
                log.warning(f"Not loading '{task_name}' as the task hasn't been uploaded.")
                continue

            task_id_str = (
                self._cached_fallback_task_id(task_name, job) if task_id is None else str(task_id)
            )
            if task_id is not None:
                self._terminal_task_id_by_task[task_name] = task_id
            task_paths[task_name] = str(self._job_data_path(task_id=task_id_str, path_dir=path_dir))
            task_ids[task_name] = task_id_str

        loaded_from_cache = {task_name: job.load_if_cached for task_name, job in self.jobs.items()}

        if not skip_download:
            self.download(path_dir=path_dir, replace_existing=replace_existing)

        data = BatchData(
            task_paths=task_paths,
            task_ids=task_ids,
            verbose=self.verbose,
            cached_tasks=loaded_from_cache,
            lazy=self.lazy,
            is_downloaded=True,
        )

        for task_name, job in self.jobs.items():
            if task_name not in task_paths:
                continue
            if isinstance(job.simulation, ModeSolver):
                job_data = data[task_name]
                if not loaded_from_cache[task_name]:
                    _store_mode_solver_in_cache(
                        task_ids[task_name], job.simulation, job_data, task_paths[task_name]
                    )
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
            elif batch_cost == 0 and all(job.load_if_cached for job in self.jobs.values()):
                console.log(
                    "No Flexcredit cost for batch as all simulations were restored from local cache."
                )
            else:
                console.log("Could not get estimated batch cost!")

        return batch_cost

    @staticmethod
    def _check_path_dir(path_dir: PathLike) -> None:
        """Make sure ``path_dir`` exists and create it if not.

        Parameters
        ----------
        path_dir : PathLike
            Directory path where files will be saved.
        """
        path_dir = Path(path_dir)
        if path_dir != Path(".") and not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)
