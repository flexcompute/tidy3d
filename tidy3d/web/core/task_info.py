"""Defines information about a task"""

from __future__ import annotations

from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class TaskBase(BaseModel, ABC):
    """Base configuration for all task objects."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChargeType(str, Enum):
    """The payment method of the task."""

    FREE = "free"
    """No payment required."""

    PAID = "paid"
    """Payment required."""


class TaskBlockInfo(TaskBase):
    """Information about the task's block status.

    Notes
    -----
        This includes details about how the task can be blocked by various features
        such as user limits and insufficient balance.
    """

    chargeType: ChargeType | None = None
    """The type of charge applicable to the task (free or paid)."""

    maxFreeCount: int | None = None
    """The maximum number of free tasks allowed."""

    maxGridPoints: int | None = None
    """The maximum number of grid points permitted."""

    maxTimeSteps: int | None = None
    """The maximum number of time steps allowed."""


class TaskInfo(TaskBase):
    """General information about a task."""

    taskId: str
    """Unique identifier for the task."""

    taskName: str | None = None
    """Name of the task."""

    nodeSize: int | None = None
    """Size of the node allocated for the task."""

    completedAt: datetime | None = None
    """Timestamp when the task was completed."""

    status: str | None = None
    """Current status of the task."""

    realCost: float | None = None
    """Actual cost incurred by the task."""

    timeSteps: int | None = None
    """Number of time steps involved in the task."""

    solverVersion: str | None = None
    """Version of the solver used for the task."""

    createAt: datetime | None = None
    """Timestamp when the task was created."""

    estCostMin: float | None = None
    """Estimated minimum cost for the task."""

    estCostMax: float | None = None
    """Estimated maximum cost for the task."""

    realFlexUnit: float | None = None
    """Actual flexible units used by the task."""

    oriRealFlexUnit: float | None = None
    """Original real flexible units."""

    estFlexUnit: float | None = None
    """Estimated flexible units for the task."""

    estFlexCreditTimeStepping: float | None = None
    """Estimated flexible credits for time stepping."""

    estFlexCreditPostProcess: float | None = None
    """Estimated flexible credits for post-processing."""

    estFlexCreditMode: float | None = None
    """Estimated flexible credits based on the mode."""

    s3Storage: float | None = None
    """Amount of S3 storage used by the task."""

    startSolverTime: datetime | None = None
    """Timestamp when the solver started."""

    finishSolverTime: datetime | None = None
    """Timestamp when the solver finished."""

    totalSolverTime: int | None = None
    """Total time taken by the solver."""

    callbackUrl: str | None = None
    """Callback URL for task notifications."""

    taskType: str | None = None
    """Type of the task."""

    metadataStatus: str | None = None
    """Status of the metadata for the task."""

    taskBlockInfo: TaskBlockInfo | None = None
    """Blocking information for the task."""

    version: str | None = None
    """Version of the task."""


class RunInfo(TaskBase):
    """Information about the run of a task."""

    perc_done: Annotated[float, Field(ge=0.0, le=100.0)]
    """Percentage of the task that is completed (0 to 100)."""

    field_decay: Annotated[float, Field(ge=0.0, le=1.0)]
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

    accountLimit: float | None = None
    taskBlockMsg: str | None = None
    taskBlockType: str | None = None
    blockStatus: str | None = None
    taskStatus: str | None = None


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

    refId: str | None = None
    folderId: str | None = None
    sweepId: str | None = None
    taskId: str | None = None
    linkedTaskId: str | None = None
    groupId: str | None = None
    taskName: str | None = None
    status: str | None = None
    sweepData: str | None = None
    validateInfo: str | None = None
    replaceData: str | None = None
    protocolVersion: str | None = None
    variable: str | None = None
    createdAt: datetime | None = None
    updatedAt: datetime | None = None
    denormalizeStatus: str | None = None
    summary: dict | None = None


class BatchDetail(TaskBase):
    """Provides a detailed, top-level view of a batch of tasks.

    Notes
    -----
        This model serves as the main payload for retrieving comprehensive
        information about a batch operation.

    Attributes
    ----------
    refId
        A reference identifier for the entire batch.
    optimizationId
        Identifier for the optimization process, if any.
    groupId
        Identifier for the group the batch belongs to.
    name
        The user-defined name of the batch.
    status
        The current status of the batch.
    totalTask
        The total number of tasks in the batch.
    preprocessSuccess
        The count of tasks that completed preprocessing.
    postprocessStatus
        The status of the batch's postprocessing stage.
    validateSuccess
        The count of tasks that passed validation.
    runSuccess
        The count of tasks that ran successfully.
    postprocessSuccess
        The count of tasks that completed postprocessing.
    taskBlockInfo
        Information on what might be blocking the batch.
    estFlexUnit
        The estimated total flexible compute units for the batch.
    totalSeconds
        The total time in seconds the batch has taken.
    totalCheckMillis
        Total time in milliseconds spent on checks.
    message
        A general message providing information about the batch status.
    tasks
        A list of `BatchMember` objects, one for each task in the batch.
    taskType
        The type of tasks contained in the batch.
    """

    refId: str | None = None
    optimizationId: str | None = None
    groupId: str | None = None
    name: str | None = None
    status: str | None = None
    totalTask: int = 0
    preprocessSuccess: int = 0
    postprocessStatus: str | None = None
    validateSuccess: int = 0
    runSuccess: int = 0
    postprocessSuccess: int = 0
    taskBlockInfo: BatchTaskBlockInfo | None = None
    estFlexUnit: float | None = None
    realFlexUnit: float | None = None
    totalSeconds: int | None = None
    totalCheckMillis: int | None = None
    message: str | None = None
    tasks: list[BatchMember] = []
    validateErrors: dict | None = None
    taskType: str = None
    version: str | None = None


class AsyncJobDetail(TaskBase):
    """Provides a detailed view of an asynchronous job and its sub-tasks.

    Notes
    -----
        This model represents a long-running operation. The 'result' attribute holds
        the output of a completed job, which for orchestration jobs, is often a
        JSON string mapping sub-task names to their unique IDs.

    Attributes
    ----------
    asyncId
        The unique identifier for the asynchronous job.
    status
        The current overall status of the job (e.g., 'RUNNING', 'COMPLETED').
    progress
        The completion percentage of the job (from 0.0 to 100.0).
    createdAt
        The timestamp when the job was created.
    completedAt
        The timestamp when the job finished (successfully or not).
    tasks
        A dictionary mapping logical task keys to their unique task IDs.
        This is often populated by parsing the 'result' of an orchestration task.
    result
        The raw string output of the completed job. If the job spawns other
        tasks, this is expected to be a JSON string detailing those tasks.
    taskBlockInfo
        Information on any dependencies blocking the job from running.
    message
        A human-readable message about the job's status.
    """

    asyncId: str
    status: str
    progress: float | None = None
    createdAt: datetime | None = None
    completedAt: datetime | None = None
    tasks: dict[str, str] | None = None
    result: str | None = None
    taskBlockInfo: TaskBlockInfo | None = None
    message: str | None = None


AsyncJobDetail.model_rebuild()
