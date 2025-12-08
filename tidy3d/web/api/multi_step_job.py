"""Higher level wrapper for workflows that require multiple sequential server tasks."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Optional

import pydantic.v1 as pd
from pydantic.v1 import PrivateAttr

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types.workflow import WorkflowDataType, WorkflowType
from tidy3d.log import log
from tidy3d.web.api import webapi as web
from tidy3d.web.api.container import DEFAULT_DATA_PATH, Job
from tidy3d.web.core.constants import TaskId
from tidy3d.web.core.types import PayType


class StepInfo(pd.BaseModel):
    """Information about a single workflow step."""

    name: str
    task_id: Optional[TaskId] = None
    status: str = "pending"

    class Config:
        """Pydantic config."""

        # Allow mutation of fields
        allow_mutation = True


class MultiStepJobState(pd.BaseModel):
    """Serializable state of a MultiStepJob for saving/loading."""

    task_name: Optional[str] = None
    folder_name: str = "default"
    solver_version: Optional[str] = None
    simulation_type: str = "tidy3d"
    current_step_index: int = 0
    steps: list[StepInfo] = pd.Field(default_factory=list)


class MultiStepJob(Tidy3dBaseModel):
    """
    Interface for managing workflows that require multiple sequential server tasks.

    This class handles simulations that are broken into multiple steps, such as
    HeatChargeSimulation which requires a meshing step followed by a solve step.
    Each step is executed sequentially, with later steps depending on the results
    of earlier ones.

    Notes
    -----
    For single-step workflows (e.g., standard FDTD simulations), use the regular
    :class:`Job` class instead. This class is specifically designed for multi-step
    workflows like HeatCharge simulations that require explicit meshing.

    Examples
    --------
    >>> from tidy3d.web import MultiStepJob
    >>> job = MultiStepJob(simulation=heat_charge_sim, task_name="my_heat_sim")
    >>> data = job.run(path="results.hdf5")
    """

    simulation: WorkflowType = pd.Field(
        ...,
        title="Simulation",
        description="Simulation to run as a multi-step workflow.",
        discriminator="type",
    )

    task_name: str = pd.Field(
        None,
        title="Task Name",
        description="Base name for the workflow tasks. Step names will be appended.",
    )

    folder_name: str = pd.Field(
        "default",
        title="Folder Name",
        description="Name of folder to store tasks on web UI.",
    )

    callback_url: Optional[str] = pd.Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event.",
    )

    solver_version: Optional[str] = pd.Field(
        None,
        title="Solver Version",
        description="Custom solver version to use.",
    )

    verbose: bool = pd.Field(
        True,
        title="Verbose",
        description="Whether to print info messages and progressbars.",
    )

    pay_type: PayType = pd.Field(
        PayType.AUTO,
        title="Payment Type",
        description="Specify the payment method.",
    )

    reduce_simulation: bool = pd.Field(
        False,
        title="Reduce Simulation",
        description="Whether to reduce structures to the simulation domain.",
    )

    simulation_type: str = pd.Field(
        "tidy3d",
        title="Simulation Type",
        description="Type of simulation, used internally only.",
    )

    worker_group: Optional[str] = pd.Field(
        None,
        title="Worker Group",
        description="Worker group for the simulation.",
    )

    lazy: bool = pd.Field(
        False,
        title="Lazy",
        description="Whether to load data lazily.",
    )

    _steps: list[StepInfo] = PrivateAttr(default_factory=list)
    _step_jobs: dict[str, Job] = PrivateAttr(default_factory=dict)
    _current_step_index: int = PrivateAttr(default=0)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._initialize_steps()

    def _initialize_steps(self) -> None:
        """Initialize the workflow steps from the simulation."""
        workflow_steps = self.simulation.workflow_steps()
        self._steps = [
            StepInfo(name=name, task_id=None, status="pending") for name, _ in workflow_steps
        ]

    @property
    def steps(self) -> list[StepInfo]:
        """Return the list of workflow steps."""
        return self._steps

    @property
    def num_steps(self) -> int:
        """Return the number of steps in this workflow."""
        return len(self._steps)

    @property
    def is_multi_step(self) -> bool:
        """Return True if this workflow has more than one step."""
        return self.num_steps > 1

    @property
    def current_step(self) -> Optional[StepInfo]:
        """Return the current step being executed."""
        if self._current_step_index < self.num_steps:
            return self._steps[self._current_step_index]
        return None

    def _get_step_task_name(self, step_name: str) -> str:
        """Generate a task name for a specific step."""
        base_name = self.task_name or "multi_step_task"
        if self.num_steps == 1:
            return base_name
        return f"{base_name}_{step_name}"

    def _get_step_simulation(self, step_index: int) -> WorkflowType:
        """Get the simulation object for a specific step."""
        workflow_steps = self.simulation.workflow_steps()
        return workflow_steps[step_index][1]

    def _get_parent_task_ids(self, step_index: int) -> Optional[tuple[TaskId, ...]]:
        """Get parent task IDs for a step (from previous steps)."""
        if step_index == 0:
            return None
        # Collect task IDs from all previous steps
        parent_ids = []
        for i in range(step_index):
            if self._steps[i].task_id:
                parent_ids.append(self._steps[i].task_id)
        return tuple(parent_ids) if parent_ids else None

    def upload_step(self, step_index: int) -> TaskId:
        """Upload a specific step to the server.

        Parameters
        ----------
        step_index : int
            Index of the step to upload.

        Returns
        -------
        TaskId
            The server task ID for this step.
        """
        if step_index >= self.num_steps:
            raise ValueError(f"Step index {step_index} out of range (max {self.num_steps - 1})")

        step = self._steps[step_index]
        step_sim = self._get_step_simulation(step_index)
        parent_tasks = self._get_parent_task_ids(step_index)

        task_id = web.upload(
            simulation=step_sim,
            task_name=self._get_step_task_name(step.name),
            folder_name=self.folder_name,
            callback_url=self.callback_url,
            verbose=self.verbose,
            simulation_type=self.simulation_type,
            parent_tasks=list(parent_tasks) if parent_tasks else None,
            solver_version=self.solver_version,
        )

        step.task_id = task_id
        step.status = "uploaded"

        if self.verbose:
            log.info(f"Step '{step.name}' uploaded with task_id: {task_id}")

        return task_id

    def start_step(self, step_index: int, priority: Optional[int] = None) -> None:
        """Start a specific step on the server.

        Parameters
        ----------
        step_index : int
            Index of the step to start.
        priority : int, optional
            Priority in the queue (1-10).
        """
        step = self._steps[step_index]
        if not step.task_id:
            raise ValueError(f"Step '{step.name}' has not been uploaded yet")

        web.start(
            step.task_id,
            solver_version=self.solver_version,
            worker_group=self.worker_group,
            pay_type=self.pay_type,
            priority=priority,
        )
        step.status = "running"

        if self.verbose:
            log.info(f"Step '{step.name}' started")

    def monitor_step(self, step_index: int) -> None:
        """Monitor progress of a specific step.

        Parameters
        ----------
        step_index : int
            Index of the step to monitor.
        """
        step = self._steps[step_index]
        if not step.task_id:
            raise ValueError(f"Step '{step.name}' has not been uploaded yet")

        web.monitor(step.task_id, verbose=self.verbose)
        step.status = "completed"

    def download_step(self, step_index: int, path: PathLike) -> None:
        """Download results from a specific step.

        Parameters
        ----------
        step_index : int
            Index of the step to download.
        path : PathLike
            Path to save the results.
        """
        step = self._steps[step_index]
        if not step.task_id:
            raise ValueError(f"Step '{step.name}' has not been uploaded yet")

        web.download(task_id=step.task_id, path=path, verbose=self.verbose)

    def load_step(self, step_index: int, path: PathLike) -> WorkflowDataType:
        """Download and load results from a specific step.

        Parameters
        ----------
        step_index : int
            Index of the step to load.
        path : PathLike
            Path to save/load the results.

        Returns
        -------
        WorkflowDataType
            The loaded data for this step.
        """
        step = self._steps[step_index]
        if not step.task_id:
            raise ValueError(f"Step '{step.name}' has not been uploaded yet")

        return web.load(task_id=step.task_id, path=path, verbose=self.verbose, lazy=self.lazy)

    def run_step(
        self,
        step_index: int,
        path: PathLike,
        priority: Optional[int] = None,
    ) -> WorkflowDataType:
        """Run a single step completely: upload, start, monitor, and load.

        Parameters
        ----------
        step_index : int
            Index of the step to run.
        path : PathLike
            Path to save the results.
        priority : int, optional
            Priority in the queue (1-10).

        Returns
        -------
        WorkflowDataType
            The loaded data for this step.
        """
        self.upload_step(step_index)
        self.start_step(step_index, priority=priority)
        self.monitor_step(step_index)
        return self.load_step(step_index, path=path)

    def run(
        self,
        path: PathLike = DEFAULT_DATA_PATH,
        priority: Optional[int] = None,
    ) -> WorkflowDataType:
        """Run all workflow steps sequentially and return the final result.

        Parameters
        ----------
        path : PathLike
            Path to download final results file (.hdf5), including filename.
        priority : int, optional
            Priority in the queue (1-10).

        Returns
        -------
        WorkflowDataType
            Object containing the final simulation results.
        """
        path = Path(path)
        parent_dir = path.parent
        if parent_dir != Path(".") and not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose and self.num_steps > 1:
            log.info(
                f"Running multi-step workflow with {self.num_steps} steps: "
                f"{[s.name for s in self._steps]}"
            )

        # Path for saving intermediate state
        state_path = self._get_state_path(path)

        data = None
        for i, step in enumerate(self._steps):
            if self.verbose and self.num_steps > 1:
                log.info(f"Running step {i + 1}/{self.num_steps}: '{step.name}'")

            # Use intermediate path for non-final steps
            if i < self.num_steps - 1:
                step_path = parent_dir / f"{path.stem}_{step.name}{path.suffix}"
            else:
                step_path = path

            data = self.run_step(i, path=step_path, priority=priority)
            self._current_step_index = i + 1

            # Save state after each step for debugging and potential resumption
            self.save_state(state_path)

        return data

    def get_step_status(self, step_index: int) -> str:
        """Get the status of a specific step.

        Parameters
        ----------
        step_index : int
            Index of the step.

        Returns
        -------
        str
            Status string for the step.
        """
        if step_index >= self.num_steps:
            raise ValueError(f"Step index {step_index} out of range")

        step = self._steps[step_index]
        if step.task_id:
            info = web.get_info(task_id=step.task_id)
            return info.status
        return step.status

    @property
    def status(self) -> dict[str, str]:
        """Return status of all steps.

        Returns
        -------
        dict[str, str]
            Dictionary mapping step names to their status.
        """
        return {step.name: self.get_step_status(i) for i, step in enumerate(self._steps)}

    def estimate_cost(self, verbose: bool = True) -> dict[str, float]:
        """Estimate cost for all steps.

        Parameters
        ----------
        verbose : bool
            Whether to log cost information.

        Returns
        -------
        dict[str, float]
            Dictionary mapping step names to estimated costs.
        """
        costs = {}
        for i, step in enumerate(self._steps):
            if step.task_id:
                costs[step.name] = web.estimate_cost(
                    step.task_id, verbose=verbose, solver_version=self.solver_version
                )
            else:
                # Need to upload first to estimate
                self.upload_step(i)
                costs[step.name] = web.estimate_cost(
                    step.task_id, verbose=verbose, solver_version=self.solver_version
                )
        return costs

    def real_cost(self, verbose: bool = True) -> dict[str, float]:
        """Get actual billed cost for all completed steps.

        Parameters
        ----------
        verbose : bool
            Whether to log cost information.

        Returns
        -------
        dict[str, float]
            Dictionary mapping step names to actual costs.
        """
        costs = {}
        for step in self._steps:
            if step.task_id and step.status == "completed":
                costs[step.name] = web.real_cost(step.task_id, verbose=verbose)
        return costs

    def delete(self) -> None:
        """Delete all server-side data associated with this workflow."""
        for step in self._steps:
            if step.task_id:
                try:
                    web.delete(step.task_id)
                    if self.verbose:
                        log.info(f"Deleted step '{step.name}' (task_id: {step.task_id})")
                except Exception as e:
                    log.warning(f"Failed to delete step '{step.name}': {e}")

    def _get_state_path(self, data_path: PathLike) -> Path:
        """Get the path for the state file based on data path."""
        data_path = Path(data_path)
        return data_path.parent / f"{data_path.stem}_multi_step_state.json"

    def save_state(self, path: PathLike) -> Path:
        """Save the current state of the workflow to a JSON file.

        This is useful for debugging and for resuming interrupted workflows.

        Parameters
        ----------
        path : PathLike
            Path to save the state file.

        Returns
        -------
        Path
            The path where the state was saved.
        """
        path = Path(path)
        state = MultiStepJobState(
            task_name=self.task_name,
            folder_name=self.folder_name,
            solver_version=self.solver_version,
            simulation_type=self.simulation_type,
            current_step_index=self._current_step_index,
            steps=self._steps,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state.json(indent=2))
        if self.verbose:
            log.info(f"Saved MultiStepJob state to: {path}")
        return path

    def load_state(self, path: PathLike) -> None:
        """Load workflow state from a JSON file.

        This restores the step information including task IDs, allowing
        you to continue monitoring or interact with previously uploaded tasks.

        Parameters
        ----------
        path : PathLike
            Path to the state file to load.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        state = MultiStepJobState.parse_file(path)

        self._current_step_index = state.current_step_index
        for i, step_data in enumerate(state.steps):
            if i < len(self._steps):
                self._steps[i].task_id = step_data.task_id
                self._steps[i].status = step_data.status

        if self.verbose:
            log.info(f"Loaded MultiStepJob state from: {path}")
            for step in self._steps:
                log.info(f"  Step '{step.name}': task_id={step.task_id}, status={step.status}")

    @classmethod
    def from_state_file(
        cls,
        state_path: PathLike,
        simulation: WorkflowType,
        **kwargs: Any,
    ) -> MultiStepJob:
        """Create a MultiStepJob from a saved state file.

        Parameters
        ----------
        state_path : PathLike
            Path to the state file.
        simulation : WorkflowType
            The simulation object (must match the original).
        **kwargs
            Additional arguments to override from saved state.

        Returns
        -------
        MultiStepJob
            A new MultiStepJob with restored state.
        """
        state = MultiStepJobState.parse_file(state_path)

        # Use saved values as defaults, allow kwargs to override
        job_kwargs = {
            "simulation": simulation,
            "task_name": state.task_name,
            "folder_name": state.folder_name,
            "solver_version": state.solver_version,
            "simulation_type": state.simulation_type,
        }
        job_kwargs.update(kwargs)

        job = cls(**job_kwargs)
        job.load_state(state_path)
        return job
