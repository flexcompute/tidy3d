"""Defines information about a task"""

from __future__ import annotations

from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Optional

import pydantic.v1 as pydantic


class TaskStatus(Enum):
    """The statuses that the task can be in."""

    INIT = "initialized"
    """The task has been initialized."""

    QUEUE = "queued"
    """The task is in the queue."""

    PRE = "preprocessing"
    """The task is in the preprocessing stage."""

    RUN = "running"
    """The task is running."""

    POST = "postprocessing"
    """The task is in the postprocessing stage."""

    SUCCESS = "success"
    """The task has completed successfully."""

    ERROR = "error"
    """The task has completed with an error."""


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
    """General information about a task."""

    taskId: str
    """Unique identifier for the task."""

    taskName: str = None
    """Name of the task."""

    nodeSize: int = None
    """Size of the node allocated for the task."""

    completedAt: Optional[datetime] = None
    """Timestamp when the task was completed."""

    status: str = None
    """Current status of the task."""

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

    realFlexUnit: float = None
    """Actual flexible units used by the task."""

    oriRealFlexUnit: float = None
    """Original real flexible units."""

    estFlexUnit: float = None
    """Estimated flexible units for the task."""

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

    taskType: str = None
    """Type of the task."""

    metadataStatus: str = None
    """Status of the metadata for the task."""

    taskBlockInfo: TaskBlockInfo = None
    """Blocking information for the task."""


class RunInfo(TaskBase):
    """Information about the run of a task."""

    perc_done: pydantic.confloat(ge=0.0, le=100.0)
    """Percentage of the task that is completed (0 to 100)."""

    field_decay: pydantic.confloat(ge=0.0, le=1.0)
    """Field decay from the maximum value (0 to 1)."""

    def display(self):
        """Print some info about the task's progress."""
        print(f" - {self.perc_done:.2f} (%) done")
        print(f" - {self.field_decay:.2e} field decay from max")


# ---------------------- Batch (Modeler) detail schema ---------------------- #


class BatchStatus(str, Enum):
    """Enumerates the possible statuses for a batch of tasks."""

    draft = "draft"
    """The batch is being configured and has not been submitted."""
    preprocess = "preprocess"
    """The batch is undergoing preprocessing."""
    validating = "validating"
    """The tasks within the batch are being validated."""
    validate_success = "validate_success"
    """All tasks in the batch passed validation."""
    validate_warn = "validate_warn"
    """Validation passed, but with warnings."""
    validate_fail = "validate_fail"
    """Validation failed for one or more tasks."""
    blocked = "blocked"
    """The batch is blocked and cannot run."""
    running = "running"
    """The batch is currently executing."""
    aborting = "aborting"
    """The batch is in the process of being aborted."""
    run_success = "run_success"
    """The batch completed successfully."""
    postprocess = "postprocess"
    """The batch is undergoing postprocessing."""
    run_failed = "run_failed"
    """The batch execution failed."""
    diverged = "diverged"
    """The simulation in the batch diverged."""
    aborted = "aborted"
    """The batch was successfully aborted."""
    error = "error"
    """An error occurred during the solver run."""


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


class BatchDetail(TaskBase):
    """
    Provides a detailed, top-level view of a batch of tasks.

    This model serves as the main payload for retrieving comprehensive
    information about a batch operation.

    Attributes:
        refId: A reference identifier for the entire batch.
        optimizationId: Identifier for the optimization process, if any.
        groupId: Identifier for the group the batch belongs to.
        name: The user-defined name of the batch.
        status: The current status of the batch.
        totalStatus: The overall status, consolidating individual task statuses.
        totalTask: The total number of tasks in the batch.
        preprocessSuccess: The count of tasks that completed preprocessing.
        postprocessStatus: The status of the batch's postprocessing stage.
        validateSuccess: The count of tasks that passed validation.
        runSuccess: The count of tasks that ran successfully.
        postprocessSuccess: The count of tasks that completed postprocessing.
        taskBlockInfo: Information on what might be blocking the batch.
        estFlexUnit: The estimated total flexible compute units for the batch.
        totalSeconds: The total time in seconds the batch has taken.
        totalCheckMillis: Total time in milliseconds spent on checks.
        message: A general message providing information about the batch status.
        tasks: A list of `BatchMember` objects, one for each task in the batch.
    """

    refId: str = None
    optimizationId: str = None
    groupId: str = None
    name: str = None
    status: str = None
    totalStatus: BatchStatus = None
    totalTask: int = 0
    preprocessSuccess: int = 0
    postprocessStatus: str = None
    validateSuccess: int = 0
    runSuccess: int = 0
    postprocessSuccess: int = 0
    taskBlockInfo: BatchTaskBlockInfo = None
    estFlexUnit: float = None
    realFlexUnit: float = None
    totalSeconds: int = None
    totalCheckMillis: int = None
    message: str = None
    tasks: list[BatchMember] = []
    validateErrors: dict = None
