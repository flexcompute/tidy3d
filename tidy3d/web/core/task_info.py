"""Defines information about a task"""

from __future__ import annotations

from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

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

    chargeType: Optional[ChargeType] = None
    """The type of charge applicable to the task (free or paid)."""

    maxFreeCount: Optional[int] = None
    """The maximum number of free tasks allowed."""

    maxGridPoints: Optional[int] = None
    """The maximum number of grid points permitted."""

    maxTimeSteps: Optional[int] = None
    """The maximum number of time steps allowed."""


class TaskInfo(TaskBase):
    """General information about a task."""

    taskId: str
    """Unique identifier for the task."""

    taskName: Optional[str] = None
    """Name of the task."""

    nodeSize: Optional[int] = None
    """Size of the node allocated for the task."""

    completedAt: Optional[datetime] = None
    """Timestamp when the task was completed."""

    status: Optional[str] = None
    """Current status of the task."""

    realCost: Optional[float] = None
    """Actual cost incurred by the task."""

    timeSteps: Optional[int] = None
    """Number of time steps involved in the task."""

    solverVersion: Optional[str] = None
    """Version of the solver used for the task."""

    createAt: Optional[datetime] = None
    """Timestamp when the task was created."""

    estCostMin: Optional[float] = None
    """Estimated minimum cost for the task."""

    estCostMax: Optional[float] = None
    """Estimated maximum cost for the task."""

    realFlexUnit: Optional[float] = None
    """Actual flexible units used by the task."""

    oriRealFlexUnit: Optional[float] = None
    """Original real flexible units."""

    estFlexUnit: Optional[float] = None
    """Estimated flexible units for the task."""

    estFlexCreditTimeStepping: Optional[float] = None
    """Estimated flexible credits for time stepping."""

    estFlexCreditPostProcess: Optional[float] = None
    """Estimated flexible credits for post-processing."""

    estFlexCreditMode: Optional[float] = None
    """Estimated flexible credits based on the mode."""

    s3Storage: Optional[float] = None
    """Amount of S3 storage used by the task."""

    startSolverTime: Optional[datetime] = None
    """Timestamp when the solver started."""

    finishSolverTime: Optional[datetime] = None
    """Timestamp when the solver finished."""

    totalSolverTime: Optional[int] = None
    """Total time taken by the solver."""

    callbackUrl: Optional[str] = None
    """Callback URL for task notifications."""

    taskType: Optional[str] = None
    """Type of the task."""

    metadataStatus: Optional[str] = None
    """Status of the metadata for the task."""

    taskBlockInfo: Optional[TaskBlockInfo] = None
    """Blocking information for the task."""

    version: Optional[str] = None
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

    accountLimit: Optional[float] = None
    taskBlockMsg: Optional[str] = None
    taskBlockType: Optional[str] = None
    blockStatus: Optional[str] = None
    taskStatus: Optional[str] = None


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

    refId: Optional[str] = None
    folderId: Optional[str] = None
    sweepId: Optional[str] = None
    taskId: Optional[str] = None
    linkedTaskId: Optional[str] = None
    groupId: Optional[str] = None
    taskName: Optional[str] = None
    status: Optional[str] = None
    sweepData: Optional[str] = None
    validateInfo: Optional[str] = None
    replaceData: Optional[str] = None
    protocolVersion: Optional[str] = None
    variable: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    denormalizeStatus: Optional[str] = None
    summary: Optional[dict] = None


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

    refId: Optional[str] = None
    optimizationId: Optional[str] = None
    groupId: Optional[str] = None
    name: Optional[str] = None
    status: Optional[str] = None
    totalTask: int = 0
    preprocessSuccess: int = 0
    postprocessStatus: Optional[str] = None
    validateSuccess: int = 0
    runSuccess: int = 0
    postprocessSuccess: int = 0
    taskBlockInfo: Optional[BatchTaskBlockInfo] = None
    estFlexUnit: Optional[float] = None
    realFlexUnit: Optional[float] = None
    totalSeconds: Optional[int] = None
    totalCheckMillis: Optional[int] = None
    message: Optional[str] = None
    tasks: list[BatchMember] = []
    validateErrors: Optional[dict] = None
    taskType: str = None
    version: Optional[str] = None


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
    progress: Optional[float] = None
    createdAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None
    tasks: Optional[dict[str, str]] = None
    result: Optional[str] = None
    taskBlockInfo: Optional[TaskBlockInfo] = None
    message: Optional[str] = None


AsyncJobDetail.model_rebuild()
