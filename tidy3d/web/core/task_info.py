"""Defines information about a task"""

from __future__ import annotations

from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Optional

import pydantic.v1 as pydantic


class TaskBase(pydantic.BaseModel, ABC):
    """Base configuration for all task objects."""

    class Config:
        """Configuration for TaskBase"""

        arbitrary_types_allowed = True
        """Allow arbitrary types to be used within the model."""


class ChargeType(str, Enum):
    """The payment method of the task."""

    FREE = "free"
    """No payment required."""

    PAID = "paid"
    """Payment required."""


class TaskBlockInfo(TaskBase):
    """Information about the task's block status.

    This includes details about how the task can be blocked by various features
    such as user limits and insufficient balance.
    """

    chargeType: ChargeType = None
    """The type of charge applicable to the task (free or paid)."""

    maxFreeCount: int = None
    """The maximum number of free tasks allowed."""

    maxGridPoints: int = None
    """The maximum number of grid points permitted."""

    maxTimeSteps: int = None
    """The maximum number of time steps allowed."""


class TaskInfo(TaskBase):
    """General information about a task or batch.

    This unified class represents both simulation tasks and batch (modeler) tasks.
    Batch-specific fields will be None for simulation tasks and vice versa.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Common fields (used by both simulation tasks and batches)
    # ─────────────────────────────────────────────────────────────────────────
    taskId: str = None
    """Unique identifier for the task."""

    taskName: str = None
    """Name of the task."""

    status: str = None
    """Current status of the task."""

    taskType: str = None
    """Type of the task."""

    version: str = None
    """Version of the task."""

    estFlexUnit: float = None
    """Estimated flexible units for the task."""

    realFlexUnit: float = None
    """Actual flexible units used by the task."""

    taskBlockInfo: TaskBlockInfo = None
    """Blocking information for the task."""

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation-specific fields (None for batches)
    # ─────────────────────────────────────────────────────────────────────────
    nodeSize: int = None
    """Size of the node allocated for the task."""

    completedAt: Optional[datetime] = None
    """Timestamp when the task was completed."""

    realCost: float = None
    """Actual cost incurred by the task."""

    timeSteps: int = None
    """Number of time steps involved in the task."""

    solverVersion: str = None
    """Version of the solver used for the task."""

    createAt: Optional[datetime] = None
    """Timestamp when the task was created."""

    estCostMin: float = None
    """Estimated minimum cost for the task."""

    estCostMax: float = None
    """Estimated maximum cost for the task."""

    oriRealFlexUnit: float = None
    """Original real flexible units."""

    estFlexCreditTimeStepping: float = None
    """Estimated flexible credits for time stepping."""

    estFlexCreditPostProcess: float = None
    """Estimated flexible credits for post-processing."""

    estFlexCreditMode: float = None
    """Estimated flexible credits based on the mode."""

    s3Storage: float = None
    """Amount of S3 storage used by the task."""

    startSolverTime: Optional[datetime] = None
    """Timestamp when the solver started."""

    finishSolverTime: Optional[datetime] = None
    """Timestamp when the solver finished."""

    totalSolverTime: int = None
    """Total time taken by the solver."""

    callbackUrl: str = None
    """Callback URL for task notifications."""

    metadataStatus: str = None
    """Status of the metadata for the task."""

    # ─────────────────────────────────────────────────────────────────────────
    # Batch-specific fields (None for simulation tasks)
    # ─────────────────────────────────────────────────────────────────────────
    refId: str = None
    """Reference identifier for batches."""

    optimizationId: str = None
    """Identifier for optimization process."""

    groupId: str = None
    """Identifier for the group the batch belongs to."""

    totalTask: int = None
    """Total number of tasks in the batch."""

    preprocessSuccess: int = None
    """Count of tasks that completed preprocessing."""

    postprocessStatus: str = None
    """Status of batch postprocessing."""

    validateSuccess: int = None
    """Count of tasks that passed validation."""

    runSuccess: int = None
    """Count of tasks that ran successfully."""

    postprocessSuccess: int = None
    """Count of tasks that completed postprocessing."""

    totalSeconds: int = None
    """Total time in seconds the batch has taken."""

    totalCheckMillis: int = None
    """Total time in milliseconds spent on checks."""

    message: str = None
    """Status message for the batch."""

    tasks: list[BatchMember] = None
    """List of batch member tasks (None for simulation tasks)."""

    validateErrors: dict = None
    """Validation errors dictionary for batches."""


class RunInfo(TaskBase):
    """Information about the run of a task."""

    perc_done: pydantic.confloat(ge=0.0, le=100.0)
    """Percentage of the task that is completed (0 to 100)."""

    field_decay: pydantic.confloat(ge=0.0, le=1.0)
    """Field decay from the maximum value (0 to 1)."""

    def display(self) -> None:
        """Print some info about the task's progress."""
        print(f" - {self.perc_done:.2f} (%) done")
        print(f" - {self.field_decay:.2e} field decay from max")


# ---------------------- Batch (Modeler) detail schema ---------------------- #


class BatchTaskBlockInfo(TaskBlockInfo):
    """
    Extends `TaskBlockInfo` with specific details for batch task blocking.

    Attributes:
        accountLimit: A usage or cost limit imposed by the user's account.
        taskBlockMsg: A human-readable message describing the reason for the block.
        taskBlockType: The specific type of block (e.g., 'balance', 'limit').
        blockStatus: The current blocking status for the batch.
        taskStatus: The status of the task when it was blocked.
    """

    accountLimit: float = None
    taskBlockMsg: str = None
    taskBlockType: str = None
    blockStatus: str = None
    taskStatus: str = None


class BatchMember(TaskBase):
    """
    Represents a single task within a larger batch operation.

    Attributes:
        refId: A reference identifier for the member task.
        folderId: The identifier of the folder containing the task.
        sweepId: The identifier for the parameter sweep, if applicable.
        taskId: The unique identifier of the task.
        linkedTaskId: The identifier of a task linked to this one.
        groupId: The identifier of the group this task belongs to.
        taskName: The name of the individual task.
        status: The current status of this specific task.
        sweepData: Data associated with a parameter sweep.
        validateInfo: Information related to the task's validation.
        replaceData: Data used for replacements or modifications.
        protocolVersion: The version of the protocol used.
        variable: The variable parameter for this task in a sweep.
        createdAt: The timestamp when the member task was created.
        updatedAt: The timestamp when the member task was last updated.
        denormalizeStatus: The status of the data denormalization process.
        summary: A dictionary containing summary information for the task.
    """

    refId: str = None
    folderId: str = None
    sweepId: str = None
    taskId: str = None
    linkedTaskId: str = None
    groupId: str = None
    taskName: str = None
    status: str = None
    sweepData: str = None
    validateInfo: str = None
    replaceData: str = None
    protocolVersion: str = None
    variable: str = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    denormalizeStatus: str = None
    summary: dict = None


class AsyncJobDetail(TaskBase):
    """
    Provides a detailed view of an asynchronous job and its sub-tasks.

    This model represents a long-running operation. The 'result' attribute holds
    the output of a completed job, which for orchestration jobs, is often a
    JSON string mapping sub-task names to their unique IDs.

    Attributes:
        asyncId: The unique identifier for the asynchronous job.
        status: The current overall status of the job (e.g., 'RUNNING', 'COMPLETED').
        progress: The completion percentage of the job (from 0.0 to 100.0).
        createdAt: The timestamp when the job was created.
        completedAt: The timestamp when the job finished (successfully or not).
        tasks: A dictionary mapping logical task keys to their unique task IDs.
               This is often populated by parsing the 'result' of an orchestration task.
        result: The raw string output of the completed job. If the job spawns other
                tasks, this is expected to be a JSON string detailing those tasks.
        taskBlockInfo: Information on any dependencies blocking the job from running.
        message: A human-readable message about the job's status.
    """

    asyncId: str
    status: str
    progress: Optional[float] = None
    createdAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None
    tasks: Optional[dict[str, str]] = None
    result: Optional[str] = None
    taskBlockInfo: Optional[TaskBlockInfo] = None
    message: Optional[str] = None


TaskInfo.update_forward_refs()
AsyncJobDetail.update_forward_refs()
