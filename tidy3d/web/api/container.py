"""higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks."""

from __future__ import annotations

import atexit
import concurrent
import hashlib
import os
import shutil
import tempfile
import threading
import time
import uuid
from abc import ABC
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from pydantic import (
    Field,
    PositiveInt,
    PrivateAttr,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from tidy3d._runtime import WASM_BUILD
from tidy3d.components.base import TYPE_TO_CLASS_MAP, Tidy3dBaseModel, cached_property
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.types.workflow import WorkflowOperationType
from tidy3d.components.workflow import Workflow, resolve_workflow
from tidy3d.config import config
from tidy3d.exceptions import DataError, ValidationError
from tidy3d.exceptions import WebError as Tidy3dWebError
from tidy3d.log import get_logging_console, log
from tidy3d.web.api import task_api
from tidy3d.web.api.container_types import BatchInput, BatchTaskTree
from tidy3d.web.api.container_utils import (
    flatten_task_container,
    reconstruct_task_container,
)
from tidy3d.web.api.states import (
    COMPLETED_PERCENT,
    COMPLETED_STATES,
    DIVERGED_STATES,
    DRAFT_STATES,
    END_STATES,
    ERROR_STATES,
    PRE_ERROR_STATES,
    QUEUED_STATES,
    RUNNING_STATES,
    STATE_PROGRESS_PERCENTAGE,
    SUCCESS_STATES,
)
from tidy3d.web.api.tidy3d_stub import Tidy3dStub, task_type_name_of
from tidy3d.web.api.workflow_batch import UniformMultiStepBatchRunner, WorkflowStepJobAdapter
from tidy3d.web.api.workflow_dependencies import (
    is_supported_parent_task_input,
    supports_implicit_parent_task_reuse,
    unsupported_parent_task_dependency_message,
)
from tidy3d.web.cache import _store_mode_solver_in_cache
from tidy3d.web.core.constants import TaskId, TaskName
from tidy3d.web.core.exceptions import WebError as CoreWebError
from tidy3d.web.core.task_core import Folder
from tidy3d.web.core.task_info import BatchDetail
from tidy3d.web.core.types import PayType

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator
    from os import PathLike
    from types import TracebackType

    from rich.progress import TaskID

    from tidy3d.components.types.workflow import WorkflowDataType
    from tidy3d.components.workflow import Step, StepInput
    from tidy3d.web.api.container_types import BatchOutput
    from tidy3d.web.core.task_info import RunInfo, TaskInfo

# Backward compatibility alias for code/tests patching `container.web`.
web = task_api

# Upload/start requests are network I/O-bound, so use a fixed concurrency cap.
UPLOAD_START_NUM_WORKERS = 64
DEFAULT_DATA_DIR = "."
BATCH_PROGRESS_REFRESH_TIME = 0.02

BatchCategoryType = str


def _default_batch_num_workers() -> int:
    """Default worker count from runtime config."""
    return config.web.default_num_workers


def _validate_batch_mapping_key(key: object) -> None:
    """Ensure batch mapping keys are strings."""
    if not isinstance(key, str):
        raise ValueError(
            "Batch simulation mapping keys must be strings. "
            f"Got key {key!r} of type {type(key).__name__!r}."
        )


def _is_flat_batch_simulation_mapping(simulations: object) -> bool:
    """Return ``True`` for the historical flat ``dict[str, WorkflowOperationType]`` batch shape."""
    return isinstance(simulations, Mapping) and all(
        isinstance(task_name, str) and isinstance(simulation, WorkflowOperationType)
        for task_name, simulation in simulations.items()
    )


def _is_flat_batch_simulation_sequence(simulations: object) -> bool:
    """Return ``True`` for the historical top-level sequence-of-workflows batch shape."""
    return isinstance(simulations, tuple) and all(
        isinstance(simulation, WorkflowOperationType) for simulation in simulations
    )


def _normalize_task_tree_node(value: object) -> object:
    """Restore tuple-backed task-tree sequence nodes after file deserialization."""
    if isinstance(value, tuple):
        return tuple(_normalize_task_tree_node(item) for item in value)
    if isinstance(value, list):
        return tuple(_normalize_task_tree_node(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_task_tree_node(item) for key, item in value.items()}
    return value


def _legacy_sequence_task_mapping(
    simulations: tuple[WorkflowOperationType, ...],
) -> dict[TaskName, WorkflowOperationType]:
    """Build the historical flat task-name mapping for top-level workflow sequences."""
    flat_simulations: dict[TaskName, WorkflowOperationType] = {}
    for index, simulation in enumerate(simulations, 1):
        stub = Tidy3dStub(simulation=simulation)
        task_name = stub.get_default_task_name() + f"_{index}"
        flat_simulations[task_name] = simulation
    return flat_simulations


def _is_serialized_workflow_leaf(value: object) -> bool:
    """Return ``True`` only for serialized workflow-model leaves."""
    if not isinstance(value, Mapping):
        return False

    type_name = value.get("type")
    if not isinstance(type_name, str):
        return False

    workflow_cls = TYPE_TO_CLASS_MAP.get(type_name)
    return isinstance(workflow_cls, type) and any(
        issubclass(workflow_cls, candidate) for candidate in _workflow_leaf_classes()
    )


@lru_cache(maxsize=1)
def _workflow_leaf_classes() -> tuple[type, ...]:
    """Flatten ``WorkflowOperationType`` into its concrete runtime classes."""

    def _flatten(type_hint: object) -> tuple[type, ...]:
        args = get_args(type_hint)
        if not args:
            return (type_hint,) if isinstance(type_hint, type) else ()
        return tuple(cls for arg in args for cls in _flatten(arg))

    return _flatten(WorkflowOperationType)


_WORKFLOW_ADAPTER = TypeAdapter(discriminated_union(WorkflowOperationType))


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


class _JobStateLock:
    """Deep-copyable lock wrapper for private runtime state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def __enter__(self) -> _JobStateLock:
        self._lock.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._lock.release()

    def __deepcopy__(self, memo: dict[int, object]) -> _JobStateLock:
        return type(self)()


class JobState(Tidy3dBaseModel):
    """Serializable workflow execution state for :class:`Job`."""

    task_ids: dict[str, TaskId | None] = Field(
        default_factory=dict,
        title="Step Task IDs",
        description="Task id per step name.",
    )
    step_statuses: dict[str, str] = Field(
        default_factory=dict,
        title="Step Statuses",
        description="Execution status per step name.",
    )
    owned_task_ids: dict[str, bool] = Field(
        default_factory=dict,
        title="Owned Step Task IDs",
        description="Whether the task id for a given step was created by this job.",
    )
    cached_task_ids: dict[str, TaskId | None] = Field(
        default_factory=dict,
        title="Cached Step Task IDs",
        description="Task ids restored from local cache per step name.",
    )
    current_step_index: int = Field(
        0,
        title="Current Step Index",
        description="Index of the next step to execute.",
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

        Once you've created a ``job`` object using :class:`tidy3d.web.api.container.Job`, you can
        upload it to our servers, run it, monitor it, and load the results with:

        .. code-block:: python

            job = tidy3d.web.Job(simulation=simulation, task_name="task_name")
            sim_data = job.run(path="out/simulation.hdf5")

        Multi-step jobs expose per-step identifiers through ``job.task_ids`` and can be advanced one
        step at a time with ``job.step()``.

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

    simulation: WorkflowOperationType = Field(
        title="simulation",
        description="Simulation to run as a 'task'.",
        discriminator=TYPE_TAG_STR,
    )

    workflow: Workflow | None = Field(
        None,
        title="Workflow",
        description="Internal workflow definition. If unset, resolved from simulation type.",
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
        description="Deprecated direct option for internal use only. Internal workflows should set "
        "``td.config.run.solver_version`` instead; external users should leave unset.",
    )

    verbose: bool = Field(
        True,
        title="Verbose",
        description="Whether to print info messages and progressbars.",
    )

    simulation_type: BatchCategoryType | None = Field(
        None,
        title="Simulation Type",
        description="Internal simulation type label; external users should leave unset.",
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

    state_cached: JobState | None = Field(
        None,
        title="State (Cached)",
        description="Cached runtime workflow state used for job serialization.",
    )

    reduce_simulation: Literal["auto", True, False] = Field(
        "auto",
        title="Reduce Simulation",
        description="Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.",
    )

    pay_type: PayType | None = Field(
        None,
        title="Payment Type",
        description="Deprecated direct option for internal use only. Internal workflows should set "
        "``td.config.run.pay_type`` instead; external users should leave unset.",
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
    _resolved_workflow: Workflow | None = PrivateAttr(default=None)
    _state: JobState | None = PrivateAttr(default=None)
    _state_lock: _JobStateLock = PrivateAttr(default_factory=_JobStateLock)
    _step_stash_paths: dict[str, str] = PrivateAttr(default_factory=dict)
    _step_cached_task_ids: dict[str, TaskId | None] = PrivateAttr(default_factory=dict)
    _cache_only_restore_miss_steps: set[str] = PrivateAttr(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        """Resolve workflow and initialize runtime state."""
        self._resolved_workflow = resolve_workflow(self.simulation, self.workflow)
        self._validate_supported_workflow_dependencies()
        self._state = self._initialize_state()

    @property
    def steps(self) -> tuple[Step, ...]:
        """Resolved workflow steps for this job."""
        return self._resolved_workflow.steps

    @property
    def is_multi_step(self) -> bool:
        """Whether this job executes more than one workflow step."""
        return len(self.steps) > 1

    @property
    def state(self) -> JobState:
        """Snapshot of current runtime workflow state."""
        return self._runtime_state.model_copy(deep=True)

    @property
    def _runtime_state(self) -> JobState:
        """Live mutable runtime workflow state for internal use."""
        if self._state is None:
            self._state = self._initialize_state()
        return self._state

    @property
    def task_ids(self) -> dict[str, TaskId | None]:
        """Task ids for all resolved steps."""
        return dict(self._runtime_state.task_ids)

    def _initialize_state(self) -> JobState:
        """Create runtime state from serialized state or legacy task id cache."""
        step_names = [step.name for step in self.steps]
        default_task_ids = dict.fromkeys(step_names, None)
        default_step_statuses = dict.fromkeys(step_names, "pending")
        default_owned_task_ids = dict.fromkeys(step_names, False)

        cached_state = self.state_cached.model_copy(deep=True) if self.state_cached else None
        if cached_state is not None:
            task_ids = {name: cached_state.task_ids.get(name) for name in step_names}
            step_statuses = {
                name: cached_state.step_statuses.get(name, "pending") for name in step_names
            }
            owned_task_ids = {
                name: cached_state.owned_task_ids.get(
                    name,
                    cached_state.task_ids.get(name) is not None,
                )
                for name in step_names
            }
            cached_task_ids = {
                name: cached_state.cached_task_ids.get(name)
                for name in step_names
                if name in cached_state.cached_task_ids
            }
            current_step_index = cached_state.current_step_index
        else:
            task_ids = default_task_ids
            step_statuses = default_step_statuses
            owned_task_ids = default_owned_task_ids
            cached_task_ids = {}
            current_step_index = 0

        if not self.is_multi_step and self.task_id_cached is not None:
            single_step_name = step_names[0]
            task_ids[single_step_name] = self.task_id_cached
            owned_task_ids[single_step_name] = True
            if step_statuses[single_step_name] == "pending":
                step_statuses[single_step_name] = "draft"

        self._apply_parent_mesh_shortcut(
            task_ids=task_ids,
            step_statuses=step_statuses,
            owned_task_ids=owned_task_ids,
        )

        state = JobState(
            task_ids=task_ids,
            step_statuses=step_statuses,
            owned_task_ids=owned_task_ids,
            cached_task_ids=cached_task_ids,
            current_step_index=self._next_pending_step_index(
                JobState(
                    task_ids=task_ids,
                    step_statuses=step_statuses,
                    owned_task_ids=owned_task_ids,
                    cached_task_ids=cached_task_ids,
                    current_step_index=current_step_index,
                )
            ),
        )
        return state

    def _is_default_heat_charge_mesh_workflow(self) -> bool:
        """Whether this job uses the default HeatCharge mesh -> solve workflow."""
        return supports_implicit_parent_task_reuse(self.steps, self.workflow)

    def _apply_parent_mesh_shortcut(
        self,
        *,
        task_ids: dict[str, TaskId | None],
        step_statuses: dict[str, str],
        owned_task_ids: dict[str, bool],
    ) -> None:
        """Skip default HeatCharge mesh step when a parent mesh task is explicitly provided."""
        if not self.parent_tasks or len(self.parent_tasks) != 1:
            return
        if not self._is_default_heat_charge_mesh_workflow():
            return
        if self.parent_tasks[0] is None:
            return

        first_step_name = self.steps[0].name
        if task_ids.get(first_step_name) is not None:
            return
        if step_statuses.get(first_step_name, "pending") != "pending":
            return

        # TODO(EMCORE-0003): add explicit user controls for mesh monitors and
        # mesh-step orchestration via workflow definitions; this shortcut is a
        # temporary backward-compatibility bridge for parent mesh task reuse.
        task_ids[first_step_name] = self.parent_tasks[0]
        step_statuses[first_step_name] = "completed"
        owned_task_ids[first_step_name] = False

    def _next_pending_step_index(self, state: JobState | None = None) -> int:
        """Index of the next step that is not complete."""
        state = state or self._runtime_state
        for index, step in enumerate(self.steps):
            if not self._step_is_complete(step.name, state):
                return index
        return len(self.steps)

    def _step_is_complete(self, step_name: str, state: JobState | None = None) -> bool:
        """Whether a step has completed execution."""
        state = state or self._runtime_state
        status = state.step_statuses.get(step_name, "pending")
        if status == "completed" or status in SUCCESS_STATES:
            return True
        return step_name == self.steps[-1].name and status in DIVERGED_STATES

    def _update_current_step_index(self) -> None:
        with self._state_lock:
            self._state = self._runtime_state.updated_copy(
                current_step_index=self._next_pending_step_index()
            )

    def _workflow_terminal_status(self) -> str:
        """Return the terminal workflow status from persisted final-step state."""
        status = self._runtime_state.step_statuses.get(self.steps[-1].name, "success")
        if status in SUCCESS_STATES:
            return "success"
        return status

    def _completed_step_status(self, step: Step, status: str) -> str:
        """Preserve terminal status only for the final user-visible workflow result."""
        if status in DIVERGED_STATES:
            return status
        if step.name == self.steps[-1].name and status in COMPLETED_STATES:
            return status
        return "completed"

    def _raise_if_step_blocks_downstream(self, step: Step, status: str) -> None:
        """Raise when a non-final step cannot safely provide downstream inputs."""
        if step.name != self.steps[-1].name and status in DIVERGED_STATES:
            raise DataError(
                f"Workflow step '{step.name}' ended with status '{status}' and cannot be used "
                "as an input to downstream steps."
            )

    def _raise_if_step_failed(self, step: Step, status: str) -> None:
        """Raise when a workflow step has already reached a failed terminal status."""
        if status in ERROR_STATES:
            raise DataError(f"Workflow step '{step.name}' ended with status '{status}'.")

    def _refresh_uploaded_step_status(self, step: Step, task_id: TaskId) -> str:
        """Refresh persisted state for an uploaded step and return its server status."""
        status = task_api.get_info(task_id=task_id, verbose=False).status
        with self._state_lock:
            if status in COMPLETED_STATES:
                self._runtime_state.step_statuses[step.name] = self._completed_step_status(
                    step, status
                )
            else:
                self._runtime_state.step_statuses[step.name] = status
            self._update_current_step_index()
        return status

    def _workflow_step(self, step_index: int) -> Step:
        """Return the workflow step at ``step_index`` for batch orchestration."""
        return self.steps[step_index]

    def _workflow_step_task_id(self, step_name: str) -> TaskId | None:
        """Return the task id recorded for a workflow step, if one exists."""
        return self._runtime_state.task_ids.get(step_name)

    def _workflow_required_step_task_id(self, step_name: str) -> TaskId:
        """Return a workflow step task id or raise when the step has not been uploaded."""
        task_id = self._workflow_step_task_id(step_name)
        if task_id is None:
            raise DataError(f"Workflow step '{step_name}' has not been uploaded yet.")
        return task_id

    def _workflow_step_status(self, step_name: str) -> str:
        """Return the locally recorded status for a workflow step."""
        return self._runtime_state.step_statuses.get(step_name, "pending")

    def _workflow_set_step_status(self, step_name: str, status: str) -> None:
        """Set the locally recorded status for a workflow step."""
        with self._state_lock:
            self._runtime_state.step_statuses[step_name] = status
            self._update_current_step_index()

    def _workflow_sync_step_status(self, step: Step, status: str) -> str:
        """Persist a server status for a workflow step and return the stored status."""
        stored_status = (
            self._completed_step_status(step, status) if status in COMPLETED_STATES else status
        )
        with self._state_lock:
            self._runtime_state.step_statuses[step.name] = stored_status
            self._update_current_step_index()
        return stored_status

    def _workflow_next_pending_step_index(self) -> int:
        """Return the next workflow step index that still needs work."""
        return self._next_pending_step_index()

    def _workflow_step_is_complete(self, step_name: str) -> bool:
        """Whether a workflow step has completed execution."""
        return self._step_is_complete(step_name)

    def _workflow_restore_step_if_cached(self, step: Step) -> bool:
        """Restore step data from local cache when available."""
        return self._restore_step_if_cached(step)

    def _workflow_refresh_uploaded_step_status(self, step: Step, task_id: TaskId) -> str:
        """Refresh persisted state for an uploaded workflow step."""
        return self._refresh_uploaded_step_status(step, task_id)

    def _workflow_advance_cache_frontier(self) -> None:
        """Restore consecutive cached workflow steps before scheduling or estimating."""
        self._refresh_cache_only_completed_steps()
        while True:
            step_idx = self._next_pending_step_index()
            if step_idx >= len(self.steps):
                return
            if not self._restore_step_if_cached(self.steps[step_idx]):
                return

    def _workflow_batch_terminal_status(self) -> str:
        """Return the terminal status from persisted final-step state."""
        return self._workflow_terminal_status()

    def _workflow_upload_step(
        self,
        step: Step,
        *,
        verbose: bool | None = False,
        verbose_estimate_cost: bool | None = None,
    ) -> None:
        """Upload a workflow step through the single-step task API."""
        parent_task_ids = self._resolve_parent_tasks(step)
        self._check_folder(self.folder_name)
        verbose_estimate_cost = verbose if verbose_estimate_cost is None else verbose_estimate_cost
        self._ensure_step_uploaded(
            step,
            parent_task_ids=parent_task_ids,
            verbose=verbose,
            verbose_estimate_cost=verbose_estimate_cost,
        )

    def _workflow_download_step(self, step_name: str, path: PathLike) -> None:
        """Download a completed workflow step artifact."""
        self._download_step(step_name, path=path)

    def _step_task_name(self, step_name: str) -> TaskName:
        base_task_name = (
            self.task_name or Tidy3dStub(simulation=self.simulation).get_default_task_name()
        )
        if self.is_multi_step:
            return f"{base_task_name}_{step_name}"
        return base_task_name

    def _task_type_hint(self, step_name: str | None = None) -> str:
        """Return the task type name used for default result file naming."""
        step = self._step_from_name(step_name) if step_name is not None else self.steps[-1]
        return task_type_name_of(step.operation)

    def _default_output_path(self, step_name: str | None = None) -> Path:
        """Return the default local output path for a workflow step or final result."""
        output_path = task_api._resolve_output_path(None, self._task_type_hint(step_name))
        if self.is_multi_step and step_name is not None:
            safe_step_name = step_name.replace("/", "_").replace("\\", "_")
            return output_path.with_name(f"{safe_step_name}_{output_path.name}")
        return output_path

    def _step_from_name(self, step_name: str) -> Step:
        for step in self.steps:
            if step.name == step_name:
                return step
        raise DataError(f"Unknown workflow step '{step_name}'.")

    def _step_has_downstream_inputs(self, step_name: str) -> bool:
        """Whether another workflow step consumes this step's output."""
        return any(
            step_input.upstream_step == step_name
            for step in self.steps
            for step_input in step.inputs
        )

    @cached_property
    def _stash_path_for_job(self) -> str:
        """Stash file path used for single-step cache restoration."""
        return self._stash_path_for_step(self.steps[0].name)

    def _stash_path_for_step(self, step_name: str) -> str:
        """Stash file path used for cache restoration for a workflow step."""
        stash_path = self._step_stash_paths.get(step_name)
        if stash_path is not None:
            return stash_path
        stash_dir = Path(tempfile.gettempdir()) / "tidy3d_stash"
        stash_dir.mkdir(parents=True, exist_ok=True)
        stash_path = str(Path(stash_dir / f"{uuid.uuid4()}_{step_name}.hdf5"))
        self._step_stash_paths[step_name] = stash_path
        if not self._stash_path:
            self._stash_path = stash_path
            atexit.register(self.clear_stash)
        return stash_path

    def _materialize_from_stash(self, dst_path: os.PathLike) -> None:
        """Atomic copy from the single-step stash to requested path."""
        self._materialize_step_from_stash(self.steps[0].name, dst_path)

    def _materialize_step_from_stash(self, step_name: str, dst_path: os.PathLike) -> None:
        """Atomic copy from a step stash to a destination path."""
        stash_path = self._step_stash_paths.get(step_name)
        if stash_path is None:
            raise DataError(f"No cached stash path found for workflow step '{step_name}'.")
        tmp = str(dst_path) + ".part"
        shutil.copy2(stash_path, tmp)
        os.replace(tmp, dst_path)

    def clear_stash(self) -> None:
        """Delete all stash files for this job."""
        for step_name, stash_path in list(self._step_stash_paths.items()):
            try:
                if os.path.exists(stash_path):
                    os.remove(stash_path)
            finally:
                self._step_stash_paths.pop(step_name, None)
        self._stash_path = None

    def _clear_step_stash(self, step_name: str) -> None:
        """Delete and forget the stash file for one workflow step."""
        stash_path = self._step_stash_paths.pop(step_name, None)
        if stash_path is None:
            return
        try:
            if os.path.exists(stash_path):
                os.remove(stash_path)
        finally:
            if self._stash_path == stash_path:
                self._stash_path = next(iter(self._step_stash_paths.values()), None)

    def _reset_cache_only_step_after_restore_miss(self, step_name: str) -> None:
        """Mark a cache-only completed step pending after its local cache entry disappears."""
        with self._state_lock:
            if (
                step_name not in self._runtime_state.cached_task_ids
                or self._runtime_state.task_ids.get(step_name) is not None
                or not self._step_is_complete(step_name)
            ):
                return
            self._step_cached_task_ids.pop(step_name, None)
            self._runtime_state.cached_task_ids.pop(step_name, None)
            self._runtime_state.step_statuses[step_name] = "pending"
            self._runtime_state.owned_task_ids[step_name] = False
            self._cache_only_restore_miss_steps.add(step_name)
            self._update_current_step_index()

    def _cache_only_restore_miss_message(self, step_name: str) -> str:
        """Return the user-facing message for a missing cache-only result."""
        return (
            f"Workflow step '{step_name}' was restored from the local cache, but the cached "
            "result is no longer available. Use 'run()' to rerun the job."
        )

    def _refresh_cache_only_completed_steps(self) -> None:
        """Restore or reopen serialized cache-only steps before resuming execution."""
        for step in self.steps:
            if (
                self._step_is_complete(step.name)
                and self._runtime_state.task_ids.get(step.name) is None
                and step.name in self._runtime_state.cached_task_ids
                and step.name not in self._step_stash_paths
            ):
                self._restore_step_if_cached(step, force=True)

    def _serializable_task_id(self) -> TaskId | None:
        """Single-step task id used for backward-compatible serialization."""
        if self.is_multi_step:
            return None
        step_name = self.steps[0].name
        return self._runtime_state.task_ids.get(step_name) or self.task_id_cached

    def _copy_for_serialization(self) -> Job:
        workflow = self.workflow
        if workflow is not None and type(workflow) is not Workflow:
            workflow = Workflow(steps=workflow.steps)
        with self._state_lock:
            return self.updated_copy(
                workflow=workflow,
                task_id_cached=self._serializable_task_id(),
                state_cached=self._runtime_state.model_copy(deep=True),
            )

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
        serializable_job = self._copy_for_serialization()
        super(Job, serializable_job).to_file(fname=fname)

    def _restore_step_if_cached(self, step: Step, *, force: bool = False) -> bool:
        """Restore step data from local cache when available."""
        if not step.cacheable:
            return False
        if self._step_is_complete(step.name):
            if step.name in self._step_stash_paths:
                return True
            if self._runtime_state.task_ids.get(step.name) is not None and not force:
                return False

        stash_path = self._stash_path_for_step(step.name)
        restored, cached_task_id = task_api.restore_simulation_if_cached(
            simulation=step.operation,
            path=stash_path,
            reduce_simulation=self.reduce_simulation,
            verbose=self.verbose,
        )
        if restored is None:
            self._clear_step_stash(step.name)
            self._reset_cache_only_step_after_restore_miss(step.name)
            return False

        if self._step_has_downstream_inputs(step.name):
            if cached_task_id is None or not self._cached_parent_task_is_available(cached_task_id):
                self._clear_step_stash(step.name)
                self._reset_cache_only_step_after_restore_miss(step.name)
                return False

        with self._state_lock:
            if cached_task_id is None:
                self._step_cached_task_ids.pop(step.name, None)
            else:
                self._step_cached_task_ids[step.name] = cached_task_id
            self._runtime_state.cached_task_ids[step.name] = cached_task_id
            self._runtime_state.task_ids[step.name] = cached_task_id
            self._runtime_state.owned_task_ids[step.name] = False
            self._runtime_state.step_statuses[step.name] = "completed"
            self._update_current_step_index()
        if not self.is_multi_step:
            self._cached_task_id = cached_task_id
            self._stash_path = stash_path
        return True

    @staticmethod
    def _cached_parent_task_is_available(task_id: TaskId) -> bool:
        """Whether a cached parent task id can still feed downstream submissions."""
        try:
            status = task_api.get_info(task_id=task_id, verbose=False).status
        except (Tidy3dWebError, CoreWebError):
            return False
        except ValueError as err:
            if str(err) == "Task not found.":
                return False
            raise
        return status in SUCCESS_STATES

    def _step_data_source(self, step_name: str) -> tuple[Step, TaskId | None, bool]:
        """Return step, task ID, and whether the local stash should be used."""
        step = self._step_from_name(step_name)
        task_id = self._runtime_state.task_ids.get(step_name)
        from_stash = step_name in self._step_stash_paths
        cached_task_ids = self._runtime_state.cached_task_ids
        should_restore_cached = (
            step_name in cached_task_ids and cached_task_ids[step_name] == task_id
        )
        if (
            not from_stash
            and self._step_is_complete(step_name)
            and (task_id is None or should_restore_cached)
        ):
            self._restore_step_if_cached(step, force=should_restore_cached)
            task_id = self._runtime_state.task_ids.get(step_name)
            from_stash = step_name in self._step_stash_paths
        return step, task_id, from_stash

    def _validate_default_heat_charge_parent_tasks(self) -> None:
        """Preserve legacy HeatCharge parent mesh validation before workflow routing."""
        if not self.parent_tasks:
            return
        if not self._is_default_heat_charge_mesh_workflow():
            return
        if len(self.parent_tasks) == 1 and self.parent_tasks[0] is not None:
            return
        raise CoreWebError(
            "Provided 'parent_tasks' failed validation: A single parent 'task_id' "
            "corresponding to the task in which the meshing was run must be provided."
        )

    def _validate_parent_tasks_workflow_compatibility(self) -> None:
        """Reject top-level parent tasks for custom multi-step workflows."""
        if not self.parent_tasks or self.workflow is None or not self.is_multi_step:
            return
        raise DataError(
            "'parent_tasks' is only supported for single-step jobs or the built-in "
            "Heat/HeatCharge volume-mesh dependency. Custom workflow jobs must define "
            "dependencies in the workflow."
        )

    def _validate_supported_parent_task_input(self, step: Step, step_input: StepInput) -> None:
        """Ensure a workflow dependency maps to a supported cloud dependency."""
        upstream_step = self._step_from_name(step_input.upstream_step)
        upstream_output = upstream_step.get_output(step_input.upstream_output)
        if upstream_output is not None and is_supported_parent_task_input(
            step, upstream_step, step_input.upstream_output, upstream_output
        ):
            return

        raise DataError(
            f"Workflow step '{step.name}' uses dependency '{step_input.upstream_output}' "
            f"from step '{step_input.upstream_step}'. {unsupported_parent_task_dependency_message()}"
        )

    def _validate_supported_workflow_dependencies(self) -> None:
        """Ensure workflow dependencies can be submitted through Tidy3D web execution."""
        for step_index, step in enumerate(self.steps):
            for input_index, step_input in enumerate(step.inputs):
                upstream_step = self._step_from_name(step_input.upstream_step)
                upstream_output = upstream_step.get_output(step_input.upstream_output)
                if upstream_output is not None and is_supported_parent_task_input(
                    step, upstream_step, step_input.upstream_output, upstream_output
                ):
                    continue

                self._raise_validation_error_at_loc(
                    ValidationError(
                        f"Workflow step '{step.name}' uses dependency "
                        f"'{step_input.upstream_output}' from step '{step_input.upstream_step}'. "
                        f"{unsupported_parent_task_dependency_message()}"
                    ),
                    "workflow",
                    "steps",
                    step_index,
                    "inputs",
                    input_index,
                    "upstream_output",
                )

    def _resolve_parent_tasks(self, step: Step) -> tuple[TaskId, ...]:
        """Collect parent task ids required to submit a workflow step."""
        self._validate_parent_tasks_workflow_compatibility()
        parent_task_ids: list[TaskId] = []
        if step.name == self.steps[0].name and self.parent_tasks:
            if self.is_multi_step:
                self._validate_default_heat_charge_parent_tasks()
            else:
                parent_task_ids.extend(
                    task_id for task_id in self.parent_tasks if task_id is not None
                )

        for step_input in step.inputs:
            self._validate_supported_parent_task_input(step, step_input)
            parent_task_id = self._runtime_state.task_ids.get(step_input.upstream_step)
            if parent_task_id is None:
                raise DataError(
                    f"Workflow step '{step.name}' requires output '{step_input.upstream_output}' "
                    f"from parent step '{step_input.upstream_step}', but no parent task id is available."
                )
            parent_task_ids.append(parent_task_id)

        deduplicated: list[TaskId] = []
        seen: set[TaskId] = set()
        for task_id in parent_task_ids:
            if task_id not in seen:
                seen.add(task_id)
                deduplicated.append(task_id)
        return tuple(deduplicated)

    def _ensure_step_uploaded(
        self,
        step: Step,
        *,
        parent_task_ids: tuple[TaskId, ...] = (),
        progress_callback: Callable[[float], None] | None = None,
        verbose: bool | None = None,
        verbose_estimate_cost: bool | None = None,
        _sidecar_artifacts: Mapping[str, Tidy3dBaseModel] | None = None,
    ) -> TaskId:
        """Upload workflow step if needed and return its task id."""
        with self._state_lock:
            task_id = self._runtime_state.task_ids.get(step.name)
        if task_id is not None:
            return task_id

        upload_kwargs = {
            "simulation": step.operation,
            "task_name": self._step_task_name(step.name),
            "folder_name": self.folder_name,
            "callback_url": self.callback_url,
            "verbose": self.verbose if verbose is None else verbose,
            "progress_callback": progress_callback,
            "simulation_type": self.simulation_type,
            "parent_tasks": list(parent_task_ids) if parent_task_ids else None,
            "solver_version": self.solver_version,
            "reduce_simulation": self.reduce_simulation,
            "verbose_estimate_cost": verbose_estimate_cost,
            "_workflow_step": True,
        }
        if _sidecar_artifacts is None:
            task_id = task_api.upload(**upload_kwargs)
        else:
            task_id = task_api._upload(
                **upload_kwargs,
                _sidecar_artifacts=_sidecar_artifacts,
            )
        with self._state_lock:
            self._runtime_state.task_ids[step.name] = task_id
            self._runtime_state.owned_task_ids[step.name] = True
            self._runtime_state.cached_task_ids.pop(step.name, None)
            self._step_cached_task_ids.pop(step.name, None)
            self._cache_only_restore_miss_steps.discard(step.name)
            self._runtime_state.step_statuses[step.name] = "draft"
            self._update_current_step_index()
        return task_id

    def _complete_step(
        self,
        step: Step,
        *,
        progress_callback_upload: Callable[[float], None] | None = None,
        verbose_estimate_cost: bool | None = None,
        worker_group: str | None = None,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
        checkpoint_callback: Callable[[], None] | None = None,
    ) -> None:
        """Run one workflow step to completion without downloading or loading its results."""
        if self._step_is_complete(step.name):
            return
        self._raise_if_step_blocks_downstream(
            step, self._runtime_state.step_statuses.get(step.name, "pending")
        )

        if self._restore_step_if_cached(step):
            if checkpoint_callback is not None:
                checkpoint_callback()
            return

        parent_task_ids = self._resolve_parent_tasks(step)
        self._check_folder(self.folder_name)
        with self._state_lock:
            had_task_id = self._runtime_state.task_ids.get(step.name) is not None
        task_id = self._ensure_step_uploaded(
            step,
            parent_task_ids=parent_task_ids,
            progress_callback=progress_callback_upload,
            verbose_estimate_cost=(
                self.verbose if verbose_estimate_cost is None else verbose_estimate_cost
            ),
        )
        if checkpoint_callback is not None:
            checkpoint_callback()

        should_start = True
        if had_task_id:
            status = self._refresh_uploaded_step_status(step, task_id)
            self._raise_if_step_failed(step, status)
            self._raise_if_step_blocks_downstream(step, status)
            if status in COMPLETED_STATES:
                if checkpoint_callback is not None:
                    checkpoint_callback()
                return
            should_start = status in DRAFT_STATES

        if should_start:
            with self._state_lock:
                self._runtime_state.step_statuses[step.name] = "queued"
            start_kwargs = {
                "task_id": task_id,
                "solver_version": self.solver_version,
                "pay_type": self.pay_type,
                "priority": priority,
                "vgpu_allocation": vgpu_allocation,
                "ignore_memory_limit": ignore_memory_limit,
            }
            if worker_group is not None:
                start_kwargs["worker_group"] = worker_group
            task_api.start(**start_kwargs)
            with self._state_lock:
                self._runtime_state.step_statuses[step.name] = "running"
        monitor_kwargs = {"task_id": task_id, "verbose": self.verbose}
        if worker_group is not None:
            monitor_kwargs["worker_group"] = worker_group
        task_api.monitor(**monitor_kwargs)
        status = task_api.get_info(task_id=task_id, verbose=False).status
        with self._state_lock:
            if status in COMPLETED_STATES:
                self._runtime_state.step_statuses[step.name] = self._completed_step_status(
                    step, status
                )
            else:
                self._runtime_state.step_statuses[step.name] = status
            self._update_current_step_index()
        if checkpoint_callback is not None:
            checkpoint_callback()
        self._raise_if_step_failed(step, status)
        self._raise_if_step_blocks_downstream(step, status)

    def _download_step(
        self,
        step_name: str,
        *,
        path: PathLike | None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Download or materialize one completed workflow step without loading it."""
        resolved_path = Path(path) if path is not None else self._default_output_path(step_name)
        self._check_path_dir(path=resolved_path)
        _step, task_id, from_stash = self._step_data_source(step_name)

        if not from_stash and task_id is None:
            raise DataError(
                f"Cannot download workflow step '{step_name}' because it has not been uploaded yet."
            )

        if from_stash:
            self._materialize_step_from_stash(step_name, resolved_path)
            return

        task_api.download(
            task_id=task_id,
            path=resolved_path,
            verbose=self.verbose,
            progress_callback=progress_callback,
        )

    def _run_step(
        self,
        step: Step,
        *,
        path: PathLike | None,
        progress_callback_upload: Callable[[float], None] | None = None,
        progress_callback_download: Callable[[float], None] | None = None,
        worker_group: str | None = None,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
        checkpoint_callback: Callable[[], None] | None = None,
    ) -> WorkflowDataType:
        """Run one workflow step end-to-end and load its result."""
        self._complete_step(
            step,
            progress_callback_upload=progress_callback_upload,
            worker_group=worker_group,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
            checkpoint_callback=checkpoint_callback,
        )
        return self.load_step(step.name, path=path, progress_callback=progress_callback_download)

    def _run_to_file(
        self,
        path: PathLike,
        *,
        progress_callback_upload: Callable[[float], None] | None = None,
        progress_callback_download: Callable[[float], None] | None = None,
        verbose_estimate_cost: bool | None = None,
        worker_group: str | None = None,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
        checkpoint_callback: Callable[[], None] | None = None,
    ) -> None:
        """Run a job and materialize the final result file without loading Python data."""
        if not self.is_multi_step:
            resolved_path = Path(path)
            self._check_path_dir(path=resolved_path)
            if self.load_if_cached:
                self._materialize_from_stash(resolved_path)
                return

            self.upload(progress_callback=progress_callback_upload)
            start_kwargs = {
                "priority": priority,
                "vgpu_allocation": vgpu_allocation,
                "ignore_memory_limit": ignore_memory_limit,
            }
            if worker_group is not None:
                start_kwargs["worker_group"] = worker_group
            self.start(**start_kwargs)

            monitor_kwargs = {}
            if worker_group is not None:
                monitor_kwargs["worker_group"] = worker_group
            self.monitor(**monitor_kwargs)

            task_api.download(
                task_id=self.task_id,
                path=resolved_path,
                verbose=self.verbose,
                progress_callback=progress_callback_download,
            )
            return

        self._check_path_dir(path=path)
        self._refresh_cache_only_completed_steps()
        while self._next_pending_step_index() < len(self.steps):
            step = self.steps[self._next_pending_step_index()]
            self._complete_step(
                step,
                progress_callback_upload=progress_callback_upload,
                verbose_estimate_cost=verbose_estimate_cost,
                worker_group=worker_group,
                priority=priority,
                vgpu_allocation=vgpu_allocation,
                ignore_memory_limit=ignore_memory_limit,
                checkpoint_callback=checkpoint_callback,
            )
        self._download_step(
            self.steps[-1].name,
            path=path,
            progress_callback=progress_callback_download,
        )

    def step(
        self,
        path: PathLike | None = None,
        progress_callback_upload: Callable[[float], None] | None = None,
        progress_callback_download: Callable[[float], None] | None = None,
        worker_group: str | None = None,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
    ) -> WorkflowDataType:
        """Run one incomplete workflow step and return its default loadable data.

        For HeatSimulation and HeatChargeSimulation jobs, this means running the mesh step
        first and the solve step on the next call. Raises if all steps are complete.
        """
        if path is not None:
            self._check_path_dir(path=path)
        if self.is_multi_step:
            self._refresh_cache_only_completed_steps()
        if self._next_pending_step_index() >= len(self.steps):
            raise DataError(
                "All workflow steps are already complete. Job.step() only advances an "
                "incomplete workflow one step. Use 'Job.load()' to load the completed "
                "result, or 'Job.run()' to return the final result, including results "
                "restored from the local cache."
            )
        step = self.steps[self._next_pending_step_index()]
        return self._run_step(
            step,
            path=path,
            progress_callback_upload=progress_callback_upload,
            progress_callback_download=progress_callback_download,
            worker_group=worker_group,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
        )

    def load_step(
        self,
        step_name: str,
        path: PathLike | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> WorkflowDataType:
        """Load the default user-facing data associated with a completed workflow step."""
        resolved_path = Path(path) if path is not None else self._default_output_path(step_name)
        self._check_path_dir(path=resolved_path)
        step, task_id, from_stash = self._step_data_source(step_name)

        if not from_stash and task_id is None:
            raise DataError(
                f"Cannot load workflow step '{step_name}' because it has not been uploaded yet."
            )

        if from_stash:
            self._materialize_step_from_stash(step_name, resolved_path)

        data = task_api.load(
            task_id=None if from_stash else task_id,
            path=resolved_path,
            verbose=self.verbose,
            progress_callback=progress_callback,
            replace_existing=not from_stash,
            lazy=self.lazy,
            cache_simulation=None if from_stash else step.operation,
            store_in_cache=step.cacheable,
        )

        if isinstance(step.operation, ModeSolver):
            if not from_stash and task_id is not None:
                _store_mode_solver_in_cache(
                    task_id,
                    task_api.get_reduced_simulation(
                        step.operation,
                        self.reduce_simulation,
                        warn_auto=False,
                    ),
                    data,
                    resolved_path,
                )
            step.operation._patch_data(data=data)
        return data

    def run(
        self,
        path: PathLike | None = None,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
        *,
        progress_callback_upload: Callable[[float], None] | None = None,
        progress_callback_download: Callable[[float], None] | None = None,
        worker_group: str | None = None,
    ) -> WorkflowDataType:
        """Run :class:`Job` all the way through and return data.

        HeatSimulation and HeatChargeSimulation jobs execute through the internal
        mesh-then-solve workflow and return the final simulation data object.

        Parameters
        ----------
        path : Optional[PathLike] = None
            Path to download results file (.hdf5), including filename. When ``None``, a default
            filename is used.
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

        if self.is_multi_step:
            self._refresh_cache_only_completed_steps()
            data = None
            while self._next_pending_step_index() < len(self.steps):
                step_idx = self._next_pending_step_index()
                step = self.steps[step_idx]
                is_final_step = step_idx == len(self.steps) - 1
                if is_final_step:
                    resolved_path = Path(path) if path is not None else self._default_output_path()
                    data = self._run_step(
                        step,
                        path=resolved_path,
                        progress_callback_upload=progress_callback_upload,
                        progress_callback_download=progress_callback_download,
                        worker_group=worker_group,
                        priority=priority,
                        vgpu_allocation=vgpu_allocation,
                        ignore_memory_limit=ignore_memory_limit,
                    )
                else:
                    self._complete_step(
                        step,
                        progress_callback_upload=progress_callback_upload,
                        worker_group=worker_group,
                        priority=priority,
                        vgpu_allocation=vgpu_allocation,
                        ignore_memory_limit=ignore_memory_limit,
                    )
            if data is None:
                return self.load(path=path, progress_callback=progress_callback_download)
            return data

        loaded_from_cache = self.load_if_cached
        if not loaded_from_cache:
            self.upload(progress_callback=progress_callback_upload)
            start_kwargs = {
                "priority": priority,
                "vgpu_allocation": vgpu_allocation,
                "ignore_memory_limit": ignore_memory_limit,
            }
            if worker_group is not None:
                start_kwargs["worker_group"] = worker_group
            self.start(**start_kwargs)

            monitor_kwargs = {}
            if worker_group is not None:
                monitor_kwargs["worker_group"] = worker_group
            self.monitor(**monitor_kwargs)
        data = self.load(path=path, progress_callback=progress_callback_download)

        return data

    @cached_property
    def load_if_cached(self) -> bool:
        """Checks if results are cached and (if yes) restores them into our shared stash file."""
        if self.is_multi_step:
            return False
        if not self.steps[0].cacheable:
            return False

        # use temporary path as final destination is unknown
        stash_path = self._stash_path_for_job

        restored, cached_task_id = task_api.restore_simulation_if_cached(
            simulation=self.steps[0].operation,
            path=stash_path,
            reduce_simulation=self.reduce_simulation,
            verbose=self.verbose,
        )
        self._cached_task_id = cached_task_id

        if restored is None:
            step_name = self.steps[0].name
            self._step_stash_paths.pop(step_name, None)
            self._reset_cache_only_step_after_restore_miss(step_name)
            return False

        self._stash_path = stash_path
        step_name = self.steps[0].name
        self._step_stash_paths[step_name] = stash_path
        self._step_cached_task_ids[step_name] = cached_task_id
        with self._state_lock:
            self._runtime_state.cached_task_ids[step_name] = cached_task_id
            self._runtime_state.task_ids[step_name] = cached_task_id
            self._runtime_state.owned_task_ids[step_name] = False
            self._runtime_state.step_statuses[step_name] = "completed"
            self._update_current_step_index()
        return True

    @property
    def task_id(self) -> TaskId:
        """The task ID for this ``Job``. Uploads the ``Job`` if it hasn't already been uploaded."""
        if self.is_multi_step:
            raise DataError(
                "Multi-step jobs do not expose a single task_id. Use 'job.task_ids' instead."
            )

        step = self.steps[0]
        if self.load_if_cached:
            if self._cached_task_id is None:
                raise DataError(
                    "This job was restored from the local cache, but no server task id is "
                    "available for 'task_id'."
                )
            return self._cached_task_id
        if step.name in self._cache_only_restore_miss_steps:
            raise DataError(self._cache_only_restore_miss_message(step.name))
        task_id = self._cached_properties.get("task_id")
        if task_id is not None:
            return task_id
        task_id = self._runtime_state.task_ids.get(step.name)
        if task_id is None and self.task_id_cached is not None:
            task_id = self.task_id_cached
            with self._state_lock:
                self._runtime_state.task_ids[step.name] = task_id
                if self._runtime_state.step_statuses.get(step.name) == "pending":
                    self._runtime_state.step_statuses[step.name] = "draft"
        if task_id is None:
            self._check_folder(self.folder_name)
            task_id = self._upload(verbose_estimate_cost=False)
        return task_id

    def _upload(
        self,
        progress_callback: Callable[[float], None] | None = None,
        verbose_estimate_cost: bool | None = None,
        _sidecar_artifacts: Mapping[str, Tidy3dBaseModel] | None = None,
    ) -> TaskId:
        """Upload this job and return the task ID for handling."""
        step = self.steps[0]
        parent_task_ids = self._resolve_parent_tasks(step)
        task_id = self._ensure_step_uploaded(
            step,
            parent_task_ids=parent_task_ids,
            progress_callback=progress_callback,
            verbose_estimate_cost=verbose_estimate_cost,
            _sidecar_artifacts=_sidecar_artifacts,
        )
        self._cached_properties["task_id"] = task_id
        return task_id

    def _upload_and_cache(
        self,
        progress_callback: Callable[[float], None] | None = None,
        verbose_estimate_cost: bool | None = None,
        _sidecar_artifacts: Mapping[str, Tidy3dBaseModel] | None = None,
    ) -> None:
        """Upload this job and cache the resulting task ID."""
        if self.is_multi_step:
            raise DataError("For multi-step jobs, use 'run()' or 'step()' instead of 'upload()'.")
        if self.load_if_cached:
            return
        cached_task_id = self._cached_properties.get("task_id")
        if cached_task_id is not None or self.task_id_cached:
            return

        self._check_folder(self.folder_name)
        verbose_estimate_cost = (
            self.verbose if verbose_estimate_cost is None else verbose_estimate_cost
        )
        task_id = self._upload(
            progress_callback=progress_callback,
            verbose_estimate_cost=verbose_estimate_cost,
            _sidecar_artifacts=_sidecar_artifacts,
        )
        self._cached_properties["task_id"] = task_id

    def upload(self, progress_callback: Callable[[float], None] | None = None) -> None:
        """Upload this ``Job`` if not already got cached results."""
        self._upload_and_cache(progress_callback=progress_callback)

    def get_info(self) -> TaskInfo:
        """Return information about a :class:`Job`.

        Returns
        -------
        :class:`TaskInfo`
            :class:`TaskInfo` object containing info about status, size, credits of task and others.
        """
        if self.is_multi_step:
            self._refresh_cache_only_completed_steps()
            step_idx = self._next_pending_step_index()
            if step_idx >= len(self.steps):
                final_task_id = self._runtime_state.task_ids.get(self.steps[-1].name)
                if final_task_id is None:
                    raise DataError("No task id is available for this completed multi-step job.")
                return task_api.get_info(task_id=final_task_id, verbose=self.verbose)
            step_name = self.steps[step_idx].name
            task_id = self._runtime_state.task_ids.get(step_name)
            if task_id is None:
                raise DataError(
                    f"Workflow step '{step_name}' has not been uploaded yet. "
                    "Use 'run()' or 'step()' to progress the workflow."
                )
            return task_api.get_info(task_id=task_id, verbose=self.verbose)

        return task_api.get_info(task_id=self.task_id, verbose=self.verbose)

    @property
    def status(self) -> str:
        """Return current status of :class:`Job`."""
        if self.is_multi_step:
            self._refresh_cache_only_completed_steps()
            step_idx = self._next_pending_step_index()
            if step_idx >= len(self.steps):
                return self._workflow_terminal_status()

            step = self.steps[step_idx]
            task_id = self._runtime_state.task_ids.get(step.name)
            if task_id is None:
                return self._runtime_state.step_statuses.get(step.name, "pending")

            status = task_api.get_info(task_id=task_id, verbose=False).status
            if status in COMPLETED_STATES:
                with self._state_lock:
                    self._runtime_state.step_statuses[step.name] = self._completed_step_status(
                        step, status
                    )
                    self._update_current_step_index()
                next_step_idx = self._next_pending_step_index()
                if next_step_idx >= len(self.steps):
                    return self._workflow_terminal_status()
                if next_step_idx != step_idx:
                    return self.status
            else:
                with self._state_lock:
                    self._runtime_state.step_statuses[step.name] = status
            return status

        if self.load_if_cached:
            return "success"
        if self.steps[0].name in self._cache_only_restore_miss_steps:
            return self._runtime_state.step_statuses.get(self.steps[0].name, "pending")
        return self.get_info().status

    def start(
        self,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
        *,
        worker_group: str | None = None,
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
        if self.is_multi_step:
            raise DataError("For multi-step jobs, use 'run()' or 'step()' instead of 'start()'.")
        loaded = self.load_if_cached
        if not loaded:
            start_kwargs = {
                "task_id": self.task_id,
                "solver_version": self.solver_version,
                "pay_type": self.pay_type,
                "priority": priority,
                "vgpu_allocation": vgpu_allocation,
                "ignore_memory_limit": ignore_memory_limit,
            }
            if worker_group is not None:
                start_kwargs["worker_group"] = worker_group
            task_api.start(**start_kwargs)

    def get_run_info(self) -> RunInfo:
        """Return information about the running :class:`Job`.

        Returns
        -------
        :class:`RunInfo`
            Task run information.
        """
        if self.is_multi_step:
            self._refresh_cache_only_completed_steps()
            step_idx = self._next_pending_step_index()
            if step_idx >= len(self.steps):
                final_task_id = self._runtime_state.task_ids.get(self.steps[-1].name)
                if final_task_id is None:
                    raise DataError("No task id is available for this completed multi-step job.")
                return task_api.get_run_info(task_id=final_task_id)
            step_name = self.steps[step_idx].name
            task_id = self._runtime_state.task_ids.get(step_name)
            if task_id is None:
                raise DataError(
                    f"Workflow step '{step_name}' has not been uploaded yet. "
                    "Use 'run()' or 'step()' first."
                )
            return task_api.get_run_info(task_id=task_id)
        return task_api.get_run_info(task_id=self.task_id)

    def monitor(self, worker_group: str | None = None) -> None:
        """Monitor progress of running :class:`Job`.

        Note
        ----
        To load the output of completed simulation into :class:`~tidy3d.SimulationData` objects,
        call :meth:`Job.load`.
        """
        if self.is_multi_step:
            raise DataError("For multi-step jobs, use 'run()' or 'step()' instead of 'monitor()'.")
        if self.load_if_cached:
            return
        monitor_kwargs = {"task_id": self.task_id, "verbose": self.verbose}
        if worker_group is not None:
            monitor_kwargs["worker_group"] = worker_group
        task_api.monitor(**monitor_kwargs)

    def download(self, path: PathLike | None = None) -> None:
        """Download results of simulation.

        Parameters
        ----------
        path : Optional[PathLike] = None
            Path to download data as ``.hdf5`` file (including filename). When ``None``, a default
            filename is used.

        Note
        ----
        To load the data after download, use :meth:`Job.load`.
        """
        if self.is_multi_step:
            resolved_path = Path(path) if path is not None else self._default_output_path()
            self._download_step(self.steps[-1].name, path=resolved_path)
            return

        resolved_path = Path(path) if path is not None else self._default_output_path()
        self._check_path_dir(path=resolved_path)
        if self.load_if_cached:
            self._materialize_from_stash(resolved_path)
            return
        task_api.download(task_id=self.task_id, path=resolved_path, verbose=self.verbose)

    def load(
        self,
        path: PathLike | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> WorkflowDataType:
        """Download job results and load them into a data object.

        Parameters
        ----------
        path : Optional[PathLike] = None
            Path to download data as ``.hdf5`` file (including filename). When ``None``, a default
            filename is used.

        Returns
        -------
        Union[:class:`~tidy3d.SimulationData`, :class:`~tidy3d.HeatSimulationData`,
        :class:`~tidy3d.EMESimulationData`]
            Object containing simulation results.
        """
        if self.is_multi_step:
            resolved_path = Path(path) if path is not None else self._default_output_path()
            return self.load_step(
                self.steps[-1].name,
                path=resolved_path,
                progress_callback=progress_callback,
            )

        resolved_path = Path(path) if path is not None else self._default_output_path()
        self._check_path_dir(path=resolved_path)
        loaded_from_cache = self.load_if_cached
        if not loaded_from_cache and self.steps[0].name in self._cache_only_restore_miss_steps:
            raise DataError(self._cache_only_restore_miss_message(self.steps[0].name))
        if loaded_from_cache:
            self._materialize_from_stash(resolved_path)

        data = task_api.load(
            task_id=None if loaded_from_cache else self.task_id,
            path=resolved_path,
            verbose=self.verbose,
            progress_callback=progress_callback,
            replace_existing=not loaded_from_cache,
            lazy=self.lazy,
            cache_simulation=self.steps[0].operation,
            store_in_cache=self.steps[0].cacheable,
        )
        operation = self.steps[0].operation
        if isinstance(operation, ModeSolver):
            if not loaded_from_cache:
                _store_mode_solver_in_cache(
                    self.task_id,
                    task_api.get_reduced_simulation(
                        operation,
                        self.reduce_simulation,
                        warn_auto=False,
                    ),
                    data,
                    resolved_path,
                )
            operation._patch_data(data=data)

        return data

    def delete(self) -> None:
        """Delete server-side data associated with :class:`Job`."""
        if self.is_multi_step:
            unique_task_ids = {
                task_id
                for step_name, task_id in self._runtime_state.task_ids.items()
                if task_id is not None and self._runtime_state.owned_task_ids.get(step_name, False)
            }
            for task_id in unique_task_ids:
                task_api.delete(task_id)
            return

        if self.load_if_cached:
            if self._cached_task_id is None:
                return
            task_api.delete(self._cached_task_id)
            return

        task_id = self._runtime_state.task_ids.get(self.steps[0].name)
        if task_id is None:
            task_id = self._cached_properties.get("task_id")
        if task_id is None:
            task_id = self.task_id_cached
            if task_id is not None:
                self._runtime_state.task_ids[self.steps[0].name] = task_id
        if task_id is None:
            return
        task_api.delete(task_id)

    def real_cost(self, verbose: bool = True) -> float | None:
        """Get the billed cost for the task associated with this job.

        Parameters
        ----------
        verbose : bool = True
            Whether to log the cost and helpful messages.

        Returns
        -------
        Optional[float]
            Billed cost of the task in FlexCredits, or ``None`` if unavailable.
        """
        if self.is_multi_step:
            if self._next_pending_step_index() < len(self.steps):
                return None
            total = 0.0
            has_available_cost = False
            for step_name, task_id in self._runtime_state.task_ids.items():
                if task_id is None or not self._runtime_state.owned_task_ids.get(step_name, False):
                    continue
                cost = task_api.real_cost(task_id, verbose=False)
                if cost is None:
                    return None
                has_available_cost = True
                total += cost
            if not has_available_cost:
                return None
            if verbose:
                console = get_logging_console()
                console.log(f"Total billed flex credit cost: {total:1.3f}.")
            return total
        if self.load_if_cached:
            if self._cached_task_id is None:
                return None
            return task_api.real_cost(self._cached_task_id, verbose=verbose)
        return task_api.real_cost(self.task_id, verbose=verbose)

    def _estimate_cost_info(self, verbose: bool = True) -> task_api.FlexCreditEstimate:
        """Estimate FlexCredit charge details for this :class:`.Job`."""
        if self.is_multi_step:
            while True:
                step_idx = self._next_pending_step_index()
                if step_idx >= len(self.steps):
                    return task_api.FlexCreditEstimate(maximum=0.0)

                step = self.steps[step_idx]
                if self._restore_step_if_cached(step):
                    continue

                task_id = self._runtime_state.task_ids.get(step.name)
                if task_id is None:
                    parent_task_ids = self._resolve_parent_tasks(step)
                    self._check_folder(self.folder_name)
                    task_id = self._ensure_step_uploaded(
                        step,
                        parent_task_ids=parent_task_ids,
                        verbose_estimate_cost=False,
                    )
                else:
                    status = self._refresh_uploaded_step_status(step, task_id)
                    if self._step_is_complete(step.name):
                        continue
                    self._raise_if_step_failed(step, status)
                    self._raise_if_step_blocks_downstream(step, status)

                estimate = task_api.estimate_cost_info(
                    task_id,
                    verbose=verbose,
                    solver_version=self.solver_version,
                    is_final_billed_cost=task_api._operation_estimate_is_final_billed_cost(
                        step.operation
                    ),
                )
                if verbose:
                    console = get_logging_console()
                    console.log(
                        "The FlexCredit estimate shown above is for the next workflow "
                        f"step '{step.name}' only."
                    )
                    if step_idx == 0:
                        console.log(
                            "This is the mesh step. Run it first with 'Job.step()'; after it "
                            "completes, call 'Job.estimate_cost()' again for the solver "
                            "estimate."
                        )
                return estimate

        if self.load_if_cached:
            return task_api.FlexCreditEstimate(maximum=0.0)
        return task_api.estimate_cost_info(
            self.task_id,
            verbose=verbose,
            solver_version=self.solver_version,
            is_final_billed_cost=task_api._operation_estimate_is_final_billed_cost(self.simulation),
        )

    def estimate_cost(self, verbose: bool = True) -> float:
        """Estimate the maximum FlexCredit charge for this :class:`.Job`.

        For multi-step jobs, estimates the first incomplete workflow step only.
        Returns ``0.0`` when every step is already complete.

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
        FDTD cost is calculated assuming the simulation runs for the full ``run_time``. If
        early shutoff is triggered, the cost is adjusted proportionately. For charge
        simulations, the billed cost depends on the number of solver iterations required
        for convergence. For Mode, EME, and Heat simulations, the estimated cost is the
        final billed cost.
        """
        return self._estimate_cost_info(verbose=verbose).maximum

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


_DEFAULT_JOB_ESTIMATE_COST = Job.estimate_cost


class BatchData(Tidy3dBaseModel, Mapping):
    """
    Holds a collection of :class:`~tidy3d.SimulationData` returned by :class:`Batch`.

    Notes
    -----

        When the batch is completed, the output is not a :class:`~tidy3d.SimulationData` but rather
        a :class:`BatchData`. For flat batches created from ``dict[str, simulation]``, this behaves
        like the historical flat task-name mapping. For nested batches, indexing and iteration
        follow the original top-level container shape while flat task access remains available
        through ``load_sim_data()`` and ``task_items()``. Converting nested results with
        ``dict(batch_results.items())`` eagerly touches all top-level entries and can load all
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

    task_ids: dict[TaskName, TaskId | None] = Field(
        title="Task IDs",
        description=(
            "Mapping of task_name to task_id for each task in batch. Cached results restored "
            "without a server task id use None."
        ),
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

    task_tree: BatchTaskTree | None = Field(
        None,
        title="Task Tree",
        description="Optional nested mapping from container positions to flat batch task names.",
    )

    _cache_enabled: bool | None = PrivateAttr(default=None)
    _cache_simulations: dict[TaskName, WorkflowOperationType] | None = PrivateAttr(default=None)
    _cacheable_tasks: dict[TaskName, bool] | None = PrivateAttr(default=None)
    _unavailable_tasks: dict[TaskName, str] = PrivateAttr(default_factory=dict)

    @field_validator("task_tree", mode="before")
    @classmethod
    def _normalize_task_tree(cls, value: object) -> object:
        """Restore tuple-backed sequence nodes after file deserialization."""
        return _normalize_task_tree_node(value)

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
        if task_name not in self.task_paths or task_name not in self.task_ids:
            reason = self._unavailable_tasks.get(task_name)
            if reason is None:
                raise KeyError(task_name)
            raise DataError(
                f"Task '{task_name}' has no loaded result available in this batch. "
                f"Reason: {reason}."
            )

        cache_enabled = self._should_cache_data()

        def _load() -> WorkflowDataType:
            task_data_path = Path(self.task_paths[task_name])
            task_id = self.task_ids[task_name]
            from_cache = self.cached_tasks[task_name] if self.cached_tasks else False
            from_downloaded_batch_file = self.is_downloaded and task_data_path.exists()

            return web.load(
                task_id=None if from_cache else task_id,
                path=task_data_path,
                verbose=False,
                replace_existing=not (from_cache or self.is_downloaded),
                lazy=self.lazy,
                cache_simulation=(
                    self._cache_simulations.get(task_name)
                    if self._cache_simulations is not None
                    else None
                ),
                store_in_cache=(
                    self._cacheable_tasks.get(task_name, True)
                    if self._cacheable_tasks is not None
                    else True
                ),
                # BatchData only sets this for files it already recorded as downloaded.
                _allow_existing_path_with_task_id=from_downloaded_batch_file and not from_cache,
            )

        if cache_enabled:
            return self._get_cached_value_by_key("load_sim_data", task_name, _load)

        data = _load()

        if not cache_enabled and self._cache_enabled is None:
            cache_enabled = self._should_cache_data()
        if cache_enabled:
            return self._get_cached_value_by_key("load_sim_data", task_name, lambda: data)
        return data

    def _load_sim_data_if_available(self, task_name: str) -> WorkflowDataType | None:
        """Load task data when available, or ``None`` for skipped / errored tasks."""
        if task_name not in self.task_paths or task_name not in self.task_ids:
            log.error(
                f"Task '{task_name}' has no loaded result available in this batch; returning None."
            )
            return None
        return self.load_sim_data(task_name)

    def _load_container_node(self, node: BatchTaskTree) -> BatchOutput:
        """Load a nested container node using the stored task-name tree."""
        return reconstruct_task_container(node, self._load_sim_data_if_available)

    def __getitem__(self, key: Hashable) -> WorkflowDataType | BatchOutput:
        """Get batch results by flat task name or by nested container key/index.

        When a nested simulation container was used to create the batch, top-level dict
        keys and sequence indices are resolved against that normalized container shape
        before falling back to flat task-name lookup.
        """
        if self.task_tree is not None:
            if isinstance(self.task_tree, dict) and key in self.task_tree:
                return self._load_container_node(self.task_tree[key])
            if isinstance(self.task_tree, tuple) and isinstance(key, int):
                if 0 <= key < len(self.task_tree):
                    return self._load_container_node(self.task_tree[key])
                raise KeyError(key)

        if isinstance(key, str):
            return self.load_sim_data(key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        """Return whether a result is directly available for ``key``."""
        if isinstance(key, str) and key in self.task_paths:
            return True
        if isinstance(self.task_tree, dict):
            return key in self.task_tree
        if isinstance(self.task_tree, tuple):
            return isinstance(key, int) and 0 <= key < len(self.task_tree)
        return key in self.task_paths

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Return ``default`` for unavailable or missing keys, like a mapping."""
        is_nested_dict_key = isinstance(self.task_tree, dict) and key in self.task_tree
        if (
            isinstance(key, str)
            and key not in self.task_paths
            and key in self._unavailable_tasks
            and not is_nested_dict_key
        ):
            return default
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over top-level container keys, or flat task names for flat batches."""
        if isinstance(self.task_tree, dict):
            return iter(self.task_tree)
        if isinstance(self.task_tree, tuple):
            return iter(range(len(self.task_tree)))
        return iter(self.task_paths)

    def __len__(self) -> int:
        """Return the top-level container size, or task count for flat batches."""
        if isinstance(self.task_tree, dict | tuple):
            return len(self.task_tree)
        return len(self.task_paths)

    def task_names(self) -> Iterator[TaskName]:
        """Return an iterator over flat batch task names."""
        return iter(self.task_paths)

    def task_items(self) -> Iterator[tuple[TaskName, WorkflowDataType]]:
        """Iterate over flat ``(task_name, sim_data)`` pairs."""
        for task_name in self.task_paths:
            yield task_name, self.load_sim_data(task_name)

    def task_values(self) -> Iterator[WorkflowDataType]:
        """Iterate over flat simulation data values."""
        for _, value in self.task_items():
            yield value

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
        :class:`.BatchData`
            Returns a ``BatchData`` mapping. When initialized from a nested container,
            ``__getitem__`` also supports the original top-level dict keys and sequence
            indices.
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

    simulations: BatchInput = Field(
        title="Simulations",
        description="Simulation or nested container of simulations to run as a batch. "
        "Nested mapping containers must use string keys.",
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
        description="Deprecated direct option for internal use only. Internal workflows should set "
        "``td.config.run.solver_version`` instead; external users should leave unset.",
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
        description="Internal simulation type label; external users should leave unset.",
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
        description="Deprecated direct option for internal use only. Internal workflows should set "
        "``td.config.run.pay_type`` instead; external users should leave unset.",
    )

    jobs_cached: dict[TaskName, Job] | None = Field(
        None,
        title="Jobs (Cached)",
        description="Optional field to specify ``jobs``. Only used as a workaround internally "
        "so that ``jobs`` is written when ``Batch.to_file()`` and then the proper task is loaded "
        "from ``Batch.from_file()``. We recommend leaving unset as setting this field along with "
        "fields that were not used to create the task will cause errors.",
    )

    task_tree: BatchTaskTree | None = Field(
        None,
        title="Task Tree",
        description="Internal nested mapping from container positions to flat batch task names.",
    )

    lazy: bool = Field(
        False,
        title="Lazy",
        description="Whether to load the actual data (lazy=False) or return a proxy that loads the data when accessed (lazy=True).",
    )

    _job_type: type = PrivateAttr(Job)
    _terminal_status_by_task: dict[TaskName, str] = PrivateAttr(default_factory=dict)
    _terminal_task_id_by_task: dict[TaskName, TaskId] = PrivateAttr(default_factory=dict)
    _tolerable_error_warning_tasks: set[TaskName] = PrivateAttr(default_factory=set)

    @field_validator("simulations", mode="plain")
    @classmethod
    def _skip_simulations_field_revalidation(cls, value: object) -> BatchInput:
        """Skip recursive ``BatchInput`` validation after field-level normalization."""
        return cast(BatchInput, value)

    @field_serializer("simulations", mode="plain", return_type=Any)
    def _serialize_simulations(self, value: BatchInput) -> Any:
        """Serialize normalized workflow containers without re-walking ``BatchInput``."""
        return value

    @field_validator("simulations", mode="before")
    @classmethod
    def _normalize_simulations_field(cls, value: object) -> object:
        """Normalize nested simulation containers on the field for loc-aware errors."""
        return cls._validate_simulations_container(value)

    @field_validator("task_tree", mode="before")
    @classmethod
    def _normalize_batch_task_tree(cls, value: object) -> object:
        """Restore tuple-backed sequence nodes after batch file deserialization."""
        return _normalize_task_tree_node(value)

    @staticmethod
    def _validate_simulations_container(simulations: object) -> object:
        """Validate container structure while normalizing sequences to tuples."""

        def _recur(value: object) -> object:
            if isinstance(value, WorkflowOperationType):
                return value
            if _is_serialized_workflow_leaf(value):
                return _WORKFLOW_ADAPTER.validate_python(value)
            if isinstance(value, tuple):
                return tuple(_recur(item) for item in value)
            if isinstance(value, list):
                return tuple(_recur(item) for item in value)
            if isinstance(value, Mapping):
                result = {}
                for key, item in value.items():
                    _validate_batch_mapping_key(key)
                    result[key] = _recur(item)
                return result
            raise TypeError(f"Unsupported element in container: {type(value)!r}")

        return _recur(simulations)

    @cached_property
    def _flattened_simulations(self) -> tuple[dict[TaskName, WorkflowOperationType], BatchTaskTree]:
        """Return the flat task mapping plus tuple-backed task tree for this batch."""
        if _is_flat_batch_simulation_mapping(self.simulations):
            flat_simulations = dict(self.simulations)
            return flat_simulations, {task_name: task_name for task_name in flat_simulations}
        if _is_flat_batch_simulation_sequence(self.simulations):
            flat_simulations = _legacy_sequence_task_mapping(self.simulations)
            return flat_simulations, tuple(flat_simulations)
        return flatten_task_container(
            self.simulations,
            is_leaf=lambda value: isinstance(value, WorkflowOperationType),
            validate_dict_key=_validate_batch_mapping_key,
        )

    @property
    def _flat_simulations(self) -> dict[TaskName, WorkflowOperationType]:
        """Flat task-name mapping used by batch internals."""
        return self._flattened_simulations[0]

    @property
    def _simulation_task_tree(self) -> BatchTaskTree:
        """Transient task tree derived from ``self.simulations`` when needed."""
        return self._flattened_simulations[1]

    @property
    def _has_nested_simulation_container(self) -> bool:
        """Whether batch results should expose nested container access."""
        return not (
            _is_flat_batch_simulation_mapping(self.simulations)
            or _is_flat_batch_simulation_sequence(self.simulations)
        )

    def _workflow_batch_runner(self) -> UniformMultiStepBatchRunner:
        """Return the helper that orchestrates supported multi-step workflow batches."""
        return UniformMultiStepBatchRunner(self)

    def _uniform_multi_step_jobs(
        self, jobs: Mapping[TaskName, Job] | None = None
    ) -> dict[TaskName, Job] | None:
        """Return jobs when they form one supported uniform workflow batch."""
        uniform_jobs = self._workflow_batch_runner().uniform_jobs(jobs)
        return uniform_jobs

    def _run_uniform_multi_step_batch(
        self,
        jobs: Mapping[TaskName, Job],
        *,
        path_dir: PathLike,
        priority: int | None,
        replace_existing: bool,
        vgpu_allocation: int | None,
        ignore_memory_limit: bool | None,
    ) -> BatchData:
        """Run a supported uniform multi-step batch with rolling workflow scheduling."""
        return self._workflow_batch_runner().run_batch(
            jobs,
            path_dir=path_dir,
            priority=priority,
            replace_existing=replace_existing,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
        )

    def step(
        self,
        path_dir: PathLike = DEFAULT_DATA_DIR,
        priority: int | None = None,
        replace_existing: bool = False,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
    ) -> BatchData | None:
        """Complete the next workflow step across a uniform multi-step batch.

        Parameters
        ----------
        path_dir : PathLike = './'
            Base directory where the batch checkpoint and final data are saved.
        priority : int = None
            Priority of the simulations in the Virtual GPU (vGPU) queue (1 = lowest,
            10 = highest). It affects only simulations from vGPU licenses and does
            not impact simulations using FlexCredits.
        replace_existing : bool = False
            Downloads the final data even if the file exists, overwriting it. Only
            applies when the completed workflow step is the final step.
        vgpu_allocation : int = None
            Number of virtual GPUs to allocate for the simulations (1, 2, 4, or 8).
            Only applies to vGPU license users. If not specified, the system
            automatically determines the optimal GPU count.
        ignore_memory_limit : Optional[bool] = None
            If ``True``, allows the simulations to run even when estimated vGPU
            memory exceeds the allocation limit (up to 2x the limit). Only applies
            to vGPU license users. Default ``None`` leaves the server behaviour
            unchanged.

        Returns
        -------
        :class:`BatchData` | None
            Returns ``None`` after an intermediate workflow step and checkpoints the
            batch state to ``{path_dir}/batch.hdf5``. Returns :class:`BatchData`
            after the final workflow step completes and final results are available.

        Raises
        ------
        DataError
            If the batch is not a supported uniform Heat or HeatCharge workflow
            batch, all workflow steps are complete, or no runnable workflow steps
            remain.
        """
        return self._workflow_batch_runner().step(
            path_dir=path_dir,
            priority=priority,
            replace_existing=replace_existing,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
        )

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
            Contains the batch results. Flat batches keep the historical task-name mapping
            interface, while nested batches expose the original top-level container shape.
            Use ``task_items()`` or ``load_sim_data(task_name)`` for flat task access.

        Note
        ----
        A typical usage might look like:

        >>> from tidy3d.web.api.container import Batch
        >>> custom_batch = Batch()
        >>> batch_data = custom_batch.run() # doctest: +SKIP
        >>> for task_name, sim_data in batch_data.task_items(): # doctest: +SKIP
        ...     # do something with data. # doctest: +SKIP

        For flat batches, ``batch_data`` iterates over task names and loads the corresponding
        data from file one by one. Nested batches instead iterate over top-level container
        entries; use ``task_items()`` for flat task iteration. If no file exists for a task,
        it is downloaded when accessed. When ``config.batch_data_cache.enabled`` is ``True`` and the
        total size of all task files is below `config.batch_data_cache.max_total_size_gb`,
        accessed results are cached in memory to avoid repeated loads.

        Nested mapping containers must use string keys.
        """
        multi_step_jobs = {
            task_name: job for task_name, job in self.jobs.items() if job.is_multi_step
        }
        if multi_step_jobs:
            single_step_jobs = {
                task_name: job
                for task_name, job in self.jobs.items()
                if task_name not in multi_step_jobs
            }
            uniform_multi_step_jobs = None
            if not single_step_jobs:
                uniform_multi_step_jobs = self._uniform_multi_step_jobs(multi_step_jobs)
            if uniform_multi_step_jobs is not None:
                return self._run_uniform_multi_step_batch(
                    uniform_multi_step_jobs,
                    path_dir=path_dir,
                    priority=priority,
                    replace_existing=replace_existing,
                    vgpu_allocation=vgpu_allocation,
                    ignore_memory_limit=ignore_memory_limit,
                )
            if single_step_jobs:
                log.warning(
                    "Batches containing both regular jobs and workflow jobs run those groups separately. "
                    "For maximum parallelism, split them into separate batches."
                )
            self._check_path_dir(path_dir)
            fatal_errors: list[tuple[TaskName, Exception]] = []
            batch_path = self._batch_path(path_dir=path_dir)
            checkpoint_lock = threading.Lock()

            def checkpoint_batch() -> None:
                with checkpoint_lock:
                    self.to_file(batch_path)

            def materialize_cached_single_step_jobs() -> None:
                for task_name, job in single_step_jobs.items():
                    if not job.load_if_cached:
                        continue
                    task_id = self._known_task_id(task_name, job)
                    task_id_for_path = (
                        self._cached_fallback_task_id(task_name, job)
                        if task_id is None
                        else task_id
                    )
                    if task_id is not None:
                        self._terminal_task_id_by_task[task_name] = task_id
                    self._terminal_status_by_task[task_name] = "success"
                    job_path = self._job_data_path(task_id=task_id_for_path, path_dir=path_dir)
                    if job_path.exists() and not replace_existing:
                        continue
                    job._materialize_from_stash(job_path)

            if single_step_jobs:
                single_jobs_to_upload = self._prepare_uncached_jobs(
                    jobs=single_step_jobs,
                    check_folder=True,
                    log_cached_jobs=True,
                )
                materialize_cached_single_step_jobs()
                if single_jobs_to_upload:
                    single_jobs_to_monitor = {
                        task_name: job
                        for task_name, job in single_step_jobs.items()
                        if not job.load_if_cached
                    }
                    self._upload_jobs(single_jobs_to_upload)
                    self.to_file(batch_path)
                    self._start_jobs(
                        single_jobs_to_upload,
                        priority=priority,
                        vgpu_allocation=vgpu_allocation,
                        ignore_memory_limit=ignore_memory_limit,
                    )
                    self._monitor_jobs(
                        single_jobs_to_monitor,
                        download_on_success=True,
                        path_dir=path_dir,
                        replace_existing=replace_existing,
                    )

            def _run_job(
                task_name: TaskName, job: Job
            ) -> tuple[TaskName, Job, Path, Exception | None]:
                temp_job_path = self._multi_step_temp_path(task_name, path_dir)
                try:
                    job._run_to_file(
                        path=temp_job_path,
                        verbose_estimate_cost=self.verbose,
                        priority=priority,
                        vgpu_allocation=vgpu_allocation,
                        ignore_memory_limit=ignore_memory_limit,
                        checkpoint_callback=checkpoint_batch,
                    )
                    return task_name, job, temp_job_path, None
                except Exception as exc:  # pragma: no cover - exercised via callers
                    return task_name, job, temp_job_path, exc

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(_run_job, task_name, job): task_name
                    for task_name, job in multi_step_jobs.items()
                }
                for fut in concurrent.futures.as_completed(futures):
                    task_name, job, temp_job_path, exc = fut.result()
                    if exc is not None:
                        if temp_job_path.exists():
                            temp_job_path.unlink()
                        error_status = self._tolerable_job_run_error_status(job, exc)
                        if error_status is not None:
                            self._terminal_status_by_task[task_name] = error_status
                            self._warn_tolerable_job_run_error(task_name, error_status)
                            continue
                        fatal_errors.append((task_name, exc))
                        continue

                    final_path = self._multi_step_result_path(task_name, job, path_dir)
                    if temp_job_path != final_path and temp_job_path.exists():
                        if final_path.exists() and not replace_existing:
                            temp_job_path.unlink()
                        else:
                            os.replace(temp_job_path, final_path)
                    self._terminal_status_by_task[task_name] = job.status
                    final_server_task_id = self._known_multi_step_server_task_id(job)
                    if final_server_task_id is not None:
                        self._terminal_task_id_by_task[task_name] = final_server_task_id

            self.to_file(batch_path)
            if fatal_errors:
                if len(fatal_errors) > 1:
                    failures = "; ".join(
                        f"{task_name}: {type(exc).__name__}: {exc}"
                        for task_name, exc in fatal_errors[1:]
                    )
                    log.error(f"Additional multi-step batch job failures: {failures}")
                raise fatal_errors[0][1]
            return self.load(path_dir=path_dir, skip_download=True)

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

        simulations = self._flat_simulations

        # the type of job to upload (to generalize to subclasses)
        JobType = self._job_type
        self_dict = self.model_dump()

        jobs = {}
        for task_name, simulation in simulations.items():
            job_kwargs = {}

            for key in JobType._upload_fields.default:
                if key == "parent_tasks":
                    continue
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
                jobs[key] = job._copy_for_serialization()
            self = self.updated_copy(jobs_cached=jobs)
        self = self.updated_copy(
            task_tree=self._simulation_task_tree if self._has_nested_simulation_container else None
        )
        super(Batch, self).to_file(fname=fname)  # noqa: UP008  # pyrefly: ignore[invalid-argument]

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

    def _partition_jobs_by_cache(
        self, jobs: Mapping[TaskName, Job] | None = None
    ) -> tuple[list[Job], list[Job]]:
        """Return cached and uncached jobs for this batch."""
        jobs_from_cache = []
        jobs_uncached = []
        for job in (jobs or self.jobs).values():
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
        jobs: Mapping[TaskName, Job] | None = None,
        check_folder: bool = False,
        log_cached_jobs: bool = False,
    ) -> list[Job]:
        """Prepare uncached jobs with shared cache/folder handling."""
        if check_folder:
            self._check_folder(self.folder_name)
        jobs_from_cache, jobs_uncached = self._partition_jobs_by_cache(jobs)
        if log_cached_jobs:
            self._log_cached_jobs(jobs_from_cache)
        return jobs_uncached

    @staticmethod
    def _cached_fallback_task_id(task_name: TaskName, job: Job) -> str:
        """Filesystem-safe fallback ID for cached jobs without known server task IDs."""
        cache_operation, _ = Batch._cache_operation_for_job(job)
        operation = cache_operation if cache_operation is not None else job.simulation
        simulation_hash = operation._hash_self()
        task_name_hash = hashlib.md5(task_name.encode("utf-8")).hexdigest()
        return f"cached_{simulation_hash}_{task_name_hash}"

    @staticmethod
    def _cache_only_step_is_complete(job: Job, step_name: str) -> bool:
        """Whether a serialized step completed from cache without a server task id."""
        state = getattr(job, "state", None)
        task_ids = getattr(state, "task_ids", None)
        cached_task_ids = getattr(state, "cached_task_ids", None)
        step_statuses = getattr(state, "step_statuses", None)
        if not (
            isinstance(task_ids, Mapping)
            and isinstance(cached_task_ids, Mapping)
            and isinstance(step_statuses, Mapping)
        ):
            return False
        return (
            step_name in cached_task_ids
            and task_ids.get(step_name) is None
            and step_statuses.get(step_name) in COMPLETED_STATES
        )

    def _cached_fallback_file_is_final_result(
        self, task_name: TaskName, job: Job, path_dir: PathLike
    ) -> bool:
        """Whether the multi-step final result already exists as a fallback file."""
        return self._cached_fallback_path_exists(task_name, job, job.steps[-1].name, path_dir)

    @staticmethod
    def _single_step_name(job: Job) -> str | None:
        """Return a job's only step name when available."""
        steps = getattr(job, "steps", ())
        if not steps:
            return None
        return getattr(steps[0], "name", None)

    def _cached_fallback_path_exists(
        self, task_name: TaskName, job: Job, step_name: str | None, path_dir: PathLike
    ) -> bool:
        """Whether an already materialized cache-only fallback artifact exists."""
        if step_name is None:
            return False
        if not self._cache_only_step_is_complete(job, step_name):
            return False
        fallback_task_id = self._cached_fallback_task_id(task_name, job)
        return self._job_data_path(task_id=fallback_task_id, path_dir=path_dir).exists()

    def _known_task_id(self, task_name: TaskName, job: Job) -> TaskId | None:
        """Return a known single-step task id without triggering uploads."""
        task_id = self._terminal_task_id_by_task.get(task_name)
        if task_id is not None:
            return task_id

        cached_properties = getattr(job, "_cached_properties", None)
        if isinstance(cached_properties, Mapping):
            task_id = cached_properties.get("task_id")
            if task_id is not None:
                return task_id

        state = getattr(job, "state", None)
        task_ids = getattr(state, "task_ids", None)
        steps = getattr(job, "steps", ())
        if isinstance(task_ids, Mapping) and steps:
            step_name = getattr(steps[0], "name", None)
            if step_name is not None:
                task_id = task_ids.get(step_name)
                if task_id is not None:
                    return task_id

        task_id_cached = getattr(job, "task_id_cached", None)
        if task_id_cached is not None:
            return task_id_cached

        if job.load_if_cached:
            return getattr(job, "_cached_task_id", None)

        return None

    def _multi_step_result_task_id(self, task_name: TaskName, job: Job) -> str:
        """Return the identifier used for a multi-step job's final artifact path."""
        final_task_id = self._known_multi_step_result_task_id(task_name, job)
        if final_task_id is None:
            return self._cached_fallback_task_id(task_name, job)
        return final_task_id

    @staticmethod
    def _known_multi_step_server_task_id(job: Job) -> str | None:
        """Return a multi-step final server task id without path-only fallback ids."""
        final_step_name = job.steps[-1].name
        task_ids = getattr(job, "task_ids", {})
        final_task_id = task_ids.get(final_step_name) if isinstance(task_ids, Mapping) else None
        if final_task_id is None:
            cached_task_ids = getattr(job, "_step_cached_task_ids", {})
            if isinstance(cached_task_ids, Mapping):
                final_task_id = cached_task_ids.get(final_step_name)
        if final_task_id is not None:
            return final_task_id
        return None

    @staticmethod
    def _multi_step_final_step_is_complete(job: Job, status: str | None = None) -> bool:
        """Whether the final workflow step has a downloadable terminal result."""
        if status is not None:
            return status in COMPLETED_STATES

        final_step_name = job.steps[-1].name
        step_is_complete = getattr(job, "_workflow_step_is_complete", None)
        if callable(step_is_complete):
            return bool(step_is_complete(final_step_name))

        state = getattr(job, "state", None)
        step_statuses = getattr(state, "step_statuses", None)
        if isinstance(step_statuses, Mapping):
            return step_statuses.get(final_step_name) in COMPLETED_STATES

        job_status = getattr(job, "status", None)
        return isinstance(job_status, str) and job_status in COMPLETED_STATES

    def _known_multi_step_result_task_id(
        self,
        task_name: TaskName,
        job: Job,
        path_dir: PathLike | None = None,
        status: str | None = None,
    ) -> str | None:
        """Return a multi-step final result id without inventing one for pending jobs."""
        final_step_name = job.steps[-1].name
        if path_dir is not None and self._cached_fallback_path_exists(
            task_name, job, final_step_name, path_dir
        ):
            return self._cached_fallback_task_id(task_name, job)

        refresh_cache_only_steps = getattr(job, "_refresh_cache_only_completed_steps", None)
        if callable(refresh_cache_only_steps):
            refresh_cache_only_steps()

        final_task_id = self._known_multi_step_server_task_id(job)
        if final_task_id is not None and self._multi_step_final_step_is_complete(
            job, status=status
        ):
            return final_task_id

        stash_paths = getattr(job, "_step_stash_paths", {})
        if isinstance(stash_paths, Mapping):
            stash_path = stash_paths.get(final_step_name)
            if stash_path is not None and Path(stash_path).exists():
                return self._cached_fallback_task_id(task_name, job)
        return None

    def _multi_step_result_path(self, task_name: TaskName, job: Job, path_dir: PathLike) -> Path:
        """Return the local path for a multi-step job's final downloaded artifact."""
        return self._job_data_path(
            task_id=self._multi_step_result_task_id(task_name, job),
            path_dir=path_dir,
        )

    @staticmethod
    def _multi_step_temp_path(task_name: TaskName, path_dir: PathLike) -> Path:
        """Temporary path used while materializing a multi-step batch artifact."""
        task_name_hash = hashlib.md5(task_name.encode("utf-8")).hexdigest()
        return Path(path_dir) / f".multi_step_{task_name_hash}_{uuid.uuid4().hex}.tmp.hdf5"

    @staticmethod
    def _tolerable_job_run_error_status(job: Job, exc: Exception) -> str | None:
        """Return terminal error status when a `job.run()` failure is a task error."""
        if not isinstance(exc, CoreWebError | Tidy3dWebError | DataError):
            return None
        try:
            status = job.status
        except Exception:
            return None
        if status in ERROR_STATES or (isinstance(exc, DataError) and status in DIVERGED_STATES):
            return status
        return None

    def _warn_tolerable_job_run_error(self, task_name: TaskName, status: str) -> None:
        """Warn once when a multi-step batch job is skipped due to a tolerable run error."""
        if task_name in self._tolerable_error_warning_tasks:
            return
        self._tolerable_error_warning_tasks.add(task_name)
        if status in ERROR_STATES:
            log.warning(f"Not loading '{task_name}' as the task errored.")
        elif status in DIVERGED_STATES:
            log.warning(
                f"Not loading '{task_name}' as the workflow diverged before "
                "the final step completed."
            )

    def _upload_jobs(
        self,
        jobs_to_upload: list[Job | WorkflowStepJobAdapter] | None = None,
        _sidecar_artifacts_by_task: Mapping[TaskName, Mapping[str, Tidy3dBaseModel]] | None = None,
    ) -> None:
        """Upload already-filtered single-step jobs using the historical batch pipeline."""
        if jobs_to_upload is None:
            jobs_to_upload = self._prepare_uncached_jobs(
                check_folder=True,
                log_cached_jobs=True,
            )

        with ThreadPoolExecutor(max_workers=UPLOAD_START_NUM_WORKERS) as executor:
            upload_futures: dict[concurrent.futures.Future[Any], Job | WorkflowStepJobAdapter] = {}
            for job in jobs_to_upload:
                _sidecar_artifacts = (
                    None
                    if _sidecar_artifacts_by_task is None
                    else _sidecar_artifacts_by_task.get(job.task_name)
                )
                if hasattr(job, "_upload_and_cache"):
                    fut = executor.submit(
                        job._upload_and_cache,
                        _sidecar_artifacts=_sidecar_artifacts,
                    )
                elif isinstance(job, WorkflowStepJobAdapter):
                    fut = executor.submit(job.upload, verbose_estimate_cost=self.verbose)
                else:
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

    def upload(self) -> None:
        """Upload a series of tasks associated with this ``Batch`` using multi-threading."""
        if any(job.is_multi_step for job in self.jobs.values()):
            raise DataError(
                "Batch.upload() does not support multi-step jobs. Use Batch.run() or run jobs individually."
            )

        jobs_to_upload = self._prepare_uncached_jobs(check_folder=True, log_cached_jobs=True)
        self._upload_jobs(jobs_to_upload)

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
        if any(job.is_multi_step for job in self.jobs.values()):
            raise DataError(
                "Batch.start() does not support multi-step jobs. Use Batch.run() or run jobs individually."
            )

        if self.verbose:
            console = get_logging_console()
            console.log(f"Started working on Batch containing {self.num_jobs} tasks.")

        jobs_to_start = self._prepare_uncached_jobs()
        self._start_jobs(
            jobs_to_start,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
        )

    def _start_jobs(
        self,
        jobs_to_start: list[Job | WorkflowStepJobAdapter],
        *,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
    ) -> None:
        """Start already-filtered single-step jobs using the historical batch pipeline."""
        with ThreadPoolExecutor(max_workers=UPLOAD_START_NUM_WORKERS) as executor:
            start_futures: dict[concurrent.futures.Future[Any], Job | WorkflowStepJobAdapter] = {}
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
        if any(job.is_multi_step for job in self.jobs.values()):
            raise DataError(
                "Batch.monitor() does not support multi-step jobs. Use Batch.run() or run jobs individually."
            )

        self._monitor_jobs(
            self.jobs,
            download_on_success=download_on_success,
            path_dir=path_dir,
            replace_existing=replace_existing,
        )

    def _monitor_jobs(
        self,
        jobs: Mapping[TaskName, Job | WorkflowStepJobAdapter],
        *,
        download_on_success: bool = False,
        path_dir: PathLike = DEFAULT_DATA_DIR,
        replace_existing: bool = False,
    ) -> None:
        """Monitor already-filtered single-step jobs using the historical batch pipeline."""
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
            self._check_path_dir(path=path_dir)
            download_executor = ThreadPoolExecutor(max_workers=self.num_workers)

        def _remember_terminal_status(task_name: TaskName, job: Job, status: str) -> None:
            if status not in END_STATES:
                return

            status_by_task[task_name] = status
            self._terminal_status_by_task[task_name] = status
            if status in ERROR_STATES:
                return

            task_id = self._terminal_task_id_by_task.get(task_name)
            if task_id is None:
                task_id = self._known_task_id(task_name, job)
            if task_id is None and not job.load_if_cached:
                task_id = job.task_id
            if task_id is None:
                self._terminal_task_id_by_task.pop(task_name, None)
                return
            self._terminal_task_id_by_task[task_name] = task_id

        def _get_status(task_name: TaskName, job: Job) -> str:
            cached_status = status_by_task.get(task_name)
            if cached_status in END_STATES:
                _remember_terminal_status(task_name, job, cached_status)
                return cached_status

            status = job.get_info().status
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
                task_id = self._known_task_id(task_name, job)
            if task_id is None and not job.load_if_cached:
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

        max_task_name = max(len(task_name) for task_name in jobs.keys())
        max_name_length = min(30, max(max_task_name, 15))

        try:
            console = None
            progress_columns = []
            if self.verbose:
                console = get_logging_console()
                monitoring_full_batch = len(jobs) == len(self.jobs) and all(
                    task_name in jobs and jobs[task_name] is self.jobs[task_name]
                    for task_name in self.jobs
                )
                if monitoring_full_batch:
                    self.estimate_cost()
                else:
                    self._estimate_cost_for_jobs(jobs, cost_subject="the monitored jobs")
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
                    if self.verbose:
                        if job.load_if_cached:
                            status = "success"
                            display_status = status
                            completed = COMPLETED_PERCENT
                            _remember_terminal_status(task_name, job, status)
                        else:
                            info = job.get_info()
                            status = info.status
                            status_by_task[task_name] = status
                            _remember_terminal_status(task_name, job, status)
                            if isinstance(info, BatchDetail):
                                display_status, _, completed = task_api._batch_detail_progress(info)
                            else:
                                display_status = status
                                completed = STATE_PROGRESS_PERCENTAGE.get(status, 0)
                        desc = pbar_description(task_name, display_status, max_name_length, 0)
                        pbar_tasks[task_name] = progress.add_task(
                            desc, total=COMPLETED_PERCENT, completed=completed
                        )
                        schedule_download(task_name, job, status=status)
                    else:
                        if job.load_if_cached:
                            _remember_terminal_status(task_name, job, "success")
                            schedule_download(task_name, job, status="success")
                        else:
                            schedule_download(task_name, job)

                while any(
                    check_continue_condition(task_name, job) for task_name, job in jobs_items
                ):
                    for task_name, job in jobs_items:
                        if job.load_if_cached:
                            continue
                        cached_status = status_by_task.get(task_name)
                        if cached_status in END_STATES:
                            schedule_download(task_name, job, status=cached_status)
                            continue
                        info = job.get_info()
                        status = info.status
                        status_by_task[task_name] = status
                        _remember_terminal_status(task_name, job, status)

                        schedule_download(task_name, job, status=status)

                        if self.verbose:
                            # choose display status & percent
                            if isinstance(info, BatchDetail):
                                display_status, _, pct = task_api._batch_detail_progress(info)
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
                    if job.load_if_cached:
                        _remember_terminal_status(task_name, job, "success")
                        schedule_download(task_name, job, status="success")
                    else:
                        cached_status = status_by_task.get(task_name)
                        if cached_status in END_STATES:
                            schedule_download(task_name, job, status=cached_status)
                        else:
                            schedule_download(task_name, job)

                    if self.verbose:
                        if job.load_if_cached:
                            display_status = "success"
                            pct = COMPLETED_PERCENT
                        else:
                            info = job.get_info()
                            status = info.status
                            if isinstance(info, BatchDetail):
                                display_status, _, pct = task_api._batch_detail_progress(info)
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

    @staticmethod
    def _cache_operation_for_job(job: Job) -> tuple[WorkflowOperationType | None, bool]:
        """Return the operation that should key local-cache storage for a job result."""
        steps = getattr(job, "steps", None)
        if steps:
            step = steps[-1] if job.is_multi_step else steps[0]
            operation = getattr(step, "operation", None)
            if operation is not None:
                return operation, getattr(step, "cacheable", True)

        simulation = getattr(job, "simulation", None)
        if isinstance(simulation, WorkflowOperationType):
            return simulation, True
        return None, True

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
        To load and iterate through the data, use :meth:`BatchData.task_items()`.

        The data for each task will be named as ``{path_dir}/{task_id}.hdf5``.
        The :class:`Batch` hdf5 file will be automatically saved as ``{path_dir}/batch.hdf5``,
        allowing one to load this :class:`Batch` later using ``batch = Batch.from_file()``.
        """
        self._check_path_dir(path=path_dir)
        self.to_file(self._batch_path(path_dir=path_dir))

        if any(job.is_multi_step for job in self.jobs.values()):
            # Warn about already-existing files if we won't overwrite them
            if not replace_existing:
                num_existing = 0
                for task_name, job in self.jobs.items():
                    if job.is_multi_step:
                        final_task_id = self._known_multi_step_result_task_id(
                            task_name, job, path_dir=path_dir
                        )
                        if final_task_id is None:
                            continue
                        job_path = self._job_data_path(task_id=final_task_id, path_dir=path_dir)
                    else:
                        fallback_file_cached = self._cached_fallback_path_exists(
                            task_name, job, self._single_step_name(job), path_dir
                        )
                        task_id_for_path = (
                            None if fallback_file_cached else self._known_task_id(task_name, job)
                        )
                        if task_id_for_path is None:
                            if not (fallback_file_cached or job.load_if_cached):
                                continue
                            task_id_for_path = self._cached_fallback_task_id(task_name, job)
                        job_path = self._job_data_path(
                            task_id=task_id_for_path,
                            path_dir=path_dir,
                        )
                    if os.path.exists(job_path):
                        num_existing += 1
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
                if job.is_multi_step:
                    final_fallback_file_cached = self._cached_fallback_file_is_final_result(
                        task_name, job, path_dir
                    )
                    status = self._terminal_status_by_task.get(task_name)
                    if status is None:
                        status = "success" if final_fallback_file_cached else job.status
                    if status in END_STATES:
                        self._terminal_status_by_task[task_name] = status
                    if status in ERROR_STATES:
                        log.warning(f"Not downloading '{task_name}' as the task errored.")
                        continue
                    final_task_id = self._known_multi_step_result_task_id(
                        task_name, job, path_dir=path_dir, status=status
                    )
                    final_server_task_id = (
                        None
                        if final_fallback_file_cached
                        else self._known_multi_step_server_task_id(job)
                    )
                    if status in DIVERGED_STATES and final_task_id is None:
                        log.warning(
                            f"Not downloading '{task_name}' as the workflow diverged before "
                            "the final step completed."
                        )
                        continue
                    if final_task_id is None:
                        log.warning(
                            f"Not downloading '{task_name}' as the final workflow step "
                            "hasn't completed."
                        )
                        continue
                    job_path = self._job_data_path(task_id=final_task_id, path_dir=path_dir)
                    if final_server_task_id is not None:
                        self._terminal_task_id_by_task[task_name] = final_server_task_id
                    stash_paths = getattr(job, "_step_stash_paths", {})
                    final_step_cached = (
                        isinstance(stash_paths, Mapping) and job.steps[-1].name in stash_paths
                    )
                else:
                    fallback_file_cached = self._cached_fallback_path_exists(
                        task_name, job, self._single_step_name(job), path_dir
                    )
                    loaded_from_cache_job = fallback_file_cached or job.load_if_cached
                    task_id = None if fallback_file_cached else self._known_task_id(task_name, job)
                    status = self._terminal_status_by_task.get(task_name)
                    if status is None:
                        if loaded_from_cache_job:
                            status = "success"
                        elif task_id is None:
                            log.warning(
                                f"Not downloading '{task_name}' as the task hasn't been uploaded."
                            )
                            continue
                        else:
                            status = job.status
                    if status in END_STATES:
                        self._terminal_status_by_task[task_name] = status
                    if status in ERROR_STATES:
                        log.warning(f"Not downloading '{task_name}' as the task errored.")
                        continue
                    if task_id is None and not loaded_from_cache_job:
                        log.warning(
                            f"Not downloading '{task_name}' as the task hasn't been uploaded."
                        )
                        continue
                    task_id_for_path = (
                        self._cached_fallback_task_id(task_name, job)
                        if task_id is None
                        else task_id
                    )
                    if task_id is not None:
                        self._terminal_task_id_by_task[task_name] = task_id
                    job_path = self._job_data_path(task_id=task_id_for_path, path_dir=path_dir)
                    final_step_cached = loaded_from_cache_job

                if final_step_cached:
                    if job_path.exists():
                        if replace_existing:
                            log.debug(f"File '{job_path}' already exists. Overwriting.")
                        else:
                            log.debug(f"File '{job_path}' already exists. Skipping.")
                            continue
                    if job.is_multi_step:
                        job._materialize_step_from_stash(job.steps[-1].name, job_path)
                    else:
                        job._materialize_from_stash(job_path)
                    continue

                if job_path.exists():
                    if replace_existing:
                        log.debug(f"File '{job_path}' already exists. Overwriting.")
                    else:
                        log.debug(f"File '{job_path}' already exists. Skipping.")
                        continue

                def fn(job: Job = job, job_path: PathLike = job_path) -> None:
                    job.download(path=job_path)

                fns.append(fn)
        else:

            def _task_id_for_download_path(task_name: TaskName, job: Job) -> str:
                if self._cached_fallback_path_exists(
                    task_name, job, self._single_step_name(job), path_dir
                ):
                    return self._cached_fallback_task_id(task_name, job)
                task_id = self._terminal_task_id_by_task.get(task_name)
                if task_id is None:
                    task_id = self._known_task_id(task_name, job)
                if task_id is None and not job.load_if_cached:
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
                fallback_file_cached = self._cached_fallback_path_exists(
                    task_name, job, self._single_step_name(job), path_dir
                )
                status = self._terminal_status_by_task.get(task_name)
                if status is None:
                    if fallback_file_cached or job.load_if_cached:
                        status = "success"
                    else:
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
        self._check_path_dir(path=path_dir)

        if self.jobs is None:
            raise DataError("Can't load batch results, hasn't been uploaded.")

        unavailable_tasks = {}
        if any(job.is_multi_step for job in self.jobs.values()):
            task_paths = {}
            task_ids = {}
            loaded_from_cache = {}
            for task_name, job in self.jobs.items():
                if job.is_multi_step:
                    final_fallback_file_cached = self._cached_fallback_file_is_final_result(
                        task_name, job, path_dir
                    )
                    status = self._terminal_status_by_task.get(task_name)
                    if status is None:
                        status = "success" if final_fallback_file_cached else job.status
                    if status in END_STATES:
                        self._terminal_status_by_task[task_name] = status
                    if status in ERROR_STATES:
                        if task_name not in self._tolerable_error_warning_tasks:
                            log.warning(f"Not loading '{task_name}' as the task errored.")
                        unavailable_tasks[task_name] = f"task status is '{status}'"
                        continue
                    final_task_id = self._known_multi_step_result_task_id(
                        task_name, job, path_dir=path_dir, status=status
                    )
                    final_server_task_id = (
                        None
                        if final_fallback_file_cached
                        else self._known_multi_step_server_task_id(job)
                    )
                    if status in DIVERGED_STATES and final_task_id is None:
                        if task_name not in self._tolerable_error_warning_tasks:
                            log.warning(
                                f"Not loading '{task_name}' as the workflow diverged before "
                                "the final step completed."
                            )
                        unavailable_tasks[task_name] = (
                            "workflow diverged before the final step completed"
                        )
                        continue
                    if final_task_id is None:
                        log.warning(
                            f"Not loading '{task_name}' as the final workflow step hasn't completed."
                        )
                        unavailable_tasks[task_name] = "final workflow step has not completed"
                        continue
                    task_paths[task_name] = str(
                        self._job_data_path(task_id=final_task_id, path_dir=path_dir)
                    )
                    task_ids[task_name] = final_server_task_id
                    if final_server_task_id is not None:
                        self._terminal_task_id_by_task[task_name] = final_server_task_id
                    final_step_name = job.steps[-1].name
                    stash_paths = getattr(job, "_step_stash_paths", {})
                    cached_fallback_task_id = self._cached_fallback_task_id(task_name, job)
                    stash_path = (
                        stash_paths.get(final_step_name)
                        if isinstance(stash_paths, Mapping)
                        else None
                    )
                    loaded_from_cache[task_name] = (
                        stash_path is not None and Path(stash_path).exists()
                    ) or final_task_id == cached_fallback_task_id
                else:
                    fallback_file_cached = self._cached_fallback_path_exists(
                        task_name, job, self._single_step_name(job), path_dir
                    )
                    loaded_from_cache_job = fallback_file_cached or job.load_if_cached
                    task_id = None if fallback_file_cached else self._known_task_id(task_name, job)
                    status = self._terminal_status_by_task.get(task_name)
                    if status is None:
                        if loaded_from_cache_job:
                            status = "success"
                        elif task_id is None:
                            log.warning(
                                f"Not loading '{task_name}' as the task hasn't been uploaded."
                            )
                            unavailable_tasks[task_name] = "task has not been uploaded"
                            continue
                        else:
                            status = job.status
                    if status in END_STATES:
                        self._terminal_status_by_task[task_name] = status
                    if status in ERROR_STATES:
                        log.warning(f"Not loading '{task_name}' as the task errored.")
                        unavailable_tasks[task_name] = f"task status is '{status}'"
                        continue
                    if task_id is None and not loaded_from_cache_job:
                        log.warning(f"Not loading '{task_name}' as the task hasn't been uploaded.")
                        unavailable_tasks[task_name] = "task has not been uploaded"
                        continue
                    task_id_str = (
                        self._cached_fallback_task_id(task_name, job)
                        if task_id is None
                        else task_id
                    )
                    if task_id is not None:
                        self._terminal_task_id_by_task[task_name] = task_id
                    task_paths[task_name] = str(
                        self._job_data_path(task_id=task_id_str, path_dir=path_dir)
                    )
                    task_ids[task_name] = None if task_id is None else task_id
                    loaded_from_cache[task_name] = loaded_from_cache_job
        else:
            task_paths = {}
            task_ids = {}
            loaded_from_cache = {}
            jobs_items = list(self.jobs.items())
            terminal_status_by_task = {
                task_name: status
                for task_name, status in self._terminal_status_by_task.items()
                if task_name in self.jobs and status in END_STATES
            }

            for task_name, job in jobs_items:
                status = terminal_status_by_task.get(task_name)
                fallback_file_cached = self._cached_fallback_path_exists(
                    task_name, job, self._single_step_name(job), path_dir
                )
                loaded_from_cache_job = fallback_file_cached or job.load_if_cached
                task_id = None if fallback_file_cached else self._known_task_id(task_name, job)

                if status is None:
                    if loaded_from_cache_job:
                        status = "success"
                    elif task_id is None:
                        log.warning(f"Not loading '{task_name}' as the task hasn't been uploaded.")
                        unavailable_tasks[task_name] = "task has not been uploaded"
                        continue
                    else:
                        status = job.status

                if status in END_STATES:
                    self._terminal_status_by_task[task_name] = status

                if status in ERROR_STATES:
                    log.warning(f"Not loading '{task_name}' as the task errored.")
                    unavailable_tasks[task_name] = f"task status is '{status}'"
                    continue

                if task_id is None and not loaded_from_cache_job:
                    log.warning(f"Not loading '{task_name}' as the task hasn't been uploaded.")
                    unavailable_tasks[task_name] = "task has not been uploaded"
                    continue

                task_id_str = (
                    self._cached_fallback_task_id(task_name, job) if task_id is None else task_id
                )
                if task_id is not None:
                    self._terminal_task_id_by_task[task_name] = task_id
                task_paths[task_name] = str(
                    self._job_data_path(task_id=task_id_str, path_dir=path_dir)
                )
                task_ids[task_name] = None if task_id is None else task_id
                loaded_from_cache[task_name] = loaded_from_cache_job

        if not skip_download:
            self.download(path_dir=path_dir, replace_existing=replace_existing)

        data = BatchData(
            task_paths=task_paths,
            task_ids=task_ids,
            verbose=self.verbose,
            cached_tasks=loaded_from_cache,
            lazy=self.lazy,
            is_downloaded=True,
            task_tree=(
                self.task_tree
                if self.task_tree is not None
                else self._simulation_task_tree
                if self._has_nested_simulation_container
                else None
            ),
        )
        data._unavailable_tasks = unavailable_tasks
        cache_simulations = {}
        cacheable_tasks = {}
        for task_name, job in self.jobs.items():
            if task_name not in task_paths:
                continue
            cache_simulation, cacheable = self._cache_operation_for_job(job)
            if cache_simulation is not None:
                cache_simulations[task_name] = cache_simulation
                cacheable_tasks[task_name] = cacheable
        data._cache_simulations = cache_simulations
        data._cacheable_tasks = cacheable_tasks

        for task_name, job in self.jobs.items():
            if task_name not in task_paths:
                continue
            cache_operation, _ = self._cache_operation_for_job(job)
            if isinstance(cache_operation, ModeSolver):
                job_data = data.load_sim_data(task_name)
                if not loaded_from_cache[task_name]:
                    _store_mode_solver_in_cache(
                        task_ids[task_name],
                        task_api.get_reduced_simulation(
                            cache_operation,
                            job.reduce_simulation,
                            warn_auto=False,
                        ),
                        job_data,
                        task_paths[task_name],
                    )
                cache_operation._patch_data(data=job_data)

        return data

    def delete(self) -> None:
        """Delete server-side data associated with each task in the batch."""
        for _, job in self.jobs.items():
            job.delete()

    def real_cost(self, verbose: bool = True) -> float | None:
        """Get the sum of billed costs for each task associated with this batch.

        Parameters
        ----------
        verbose : bool = True
            Whether to log the cost and helpful messages.

        Returns
        -------
        Optional[float]
            Billed cost for the entire :class:`.Batch`, or ``None`` if unavailable.
        """
        real_cost_sum = 0.0
        found_cost = False
        for _, job in self.jobs.items():
            cost_job = job.real_cost(verbose=False)
            if cost_job is None:
                return None
            found_cost = True
            real_cost_sum += cost_job

        if not found_cost:
            return None

        if verbose:
            console = get_logging_console()
            console.log(f"Total billed flex credit cost: {real_cost_sum:1.3f}.")
        return real_cost_sum

    def _estimate_cost_for_jobs(
        self,
        jobs: Mapping[TaskName, Job | WorkflowStepJobAdapter],
        verbose: bool = True,
        *,
        cost_subject: str = "the whole batch",
    ) -> float:
        """Estimate cost for an already-filtered set of single-step jobs."""
        job_estimates = [self._estimate_cost_info_for_job(job) for _, job in jobs.items()]
        job_costs = [estimate.maximum for estimate in job_estimates]
        if any(cost is None for cost in job_costs):
            batch_cost = None
        else:
            batch_cost = sum(job_costs)
        batch_typical_cost = (
            task_api._batch_typical_flex_credit_cost(job_estimates)
            if batch_cost is not None
            else None
        )

        if verbose:
            console = get_logging_console()
            if batch_cost is not None and batch_cost > 0:
                if batch_typical_cost is not None:
                    console.log(
                        f"Estimated typical FlexCredit cost: {batch_typical_cost:1.3f} "
                        f"for {cost_subject}."
                    )
                    console.log(f"Maximum FlexCredit cost: {batch_cost:1.3f} for {cost_subject}.")
                    if any(
                        task_api._estimate_has_charge_solver_iteration_scaling(estimate)
                        for estimate in job_estimates
                    ):
                        console.log(
                            "For charge simulations, the billed cost depends on the number of "
                            "solver iterations required for convergence."
                        )
                else:
                    console.log(f"Maximum FlexCredit cost: {batch_cost:1.3f} for {cost_subject}.")
            elif batch_cost == 0 and all(job.load_if_cached for job in jobs.values()):
                console.log(
                    f"No Flexcredit cost for {cost_subject} as all simulations were restored "
                    "from local cache."
                )
            else:
                console.log("Could not get estimated batch cost!")

        return batch_cost

    @staticmethod
    def _estimate_cost_info_for_job(
        job: Job | WorkflowStepJobAdapter,
    ) -> task_api.FlexCreditEstimate:
        """Return detailed estimate info while preserving legacy estimate_cost mocks."""
        estimate_cost_info = getattr(job, "_estimate_cost_info", None)
        if isinstance(job, Job) and type(job).estimate_cost is not _DEFAULT_JOB_ESTIMATE_COST:
            estimate_cost_info = None
        if estimate_cost_info is not None:
            return estimate_cost_info(verbose=False)
        return task_api.FlexCreditEstimate(maximum=job.estimate_cost(verbose=False))

    def _estimate_uniform_multi_step_frontier_cost(
        self,
        jobs: Mapping[TaskName, Job],
        verbose: bool = True,
    ) -> float:
        """Estimate the shared next workflow step for a uniform multi-step batch."""
        return self._workflow_batch_runner().estimate_frontier_cost(jobs, verbose=verbose)

    def estimate_cost(self, verbose: bool = True) -> float:
        """Compute the maximum FlexCredit charge for a given :class:`.Batch`.

        Parameters
        ----------
        verbose : bool = True
            Whether to log the cost and helpful messages.

        Note
        ----
        FDTD cost is calculated assuming each simulation runs for the full ``run_time``. If
        early shutoff is triggered, the cost is adjusted proportionately. For charge
        simulations, the billed cost depends on the number of solver iterations required
        for convergence. For Mode, EME, and Heat simulations, the estimated cost is the
        final billed cost.

        Returns
        -------
        float
            For regular batches, estimated total cost of the tasks in FlexCredits.
            For supported uniform multi-step workflow batches, estimated cost of
            the shared next workflow step across the batch. If every workflow
            step is complete, returns ``0.0``.
        """
        if any(job.is_multi_step for job in self.jobs.values()):
            uniform_multi_step_jobs = self._uniform_multi_step_jobs()
            if uniform_multi_step_jobs is not None and len(uniform_multi_step_jobs) == len(
                self.jobs
            ):
                return self._estimate_uniform_multi_step_frontier_cost(
                    uniform_multi_step_jobs,
                    verbose=verbose,
                )
            raise DataError(
                "Batch.estimate_cost() is not supported for mixed or non-uniform multi-step "
                "batches. Use Job.estimate_cost() per job instead."
            )

        return self._estimate_cost_for_jobs(self.jobs, verbose=verbose)

    @staticmethod
    def _check_path_dir(path: PathLike) -> None:
        """Make sure ``path`` exists and create it if not.

        Parameters
        ----------
        path : PathLike
            Directory path where files will be saved.
        """
        path_dir = Path(path)
        if path_dir != Path(".") and not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)
