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
    draft = "draft"
    preprocess = "preprocess"
    validating = "validating"
    validate_success = "validate_success"
    validate_warn = "validate_warn"
    validate_fail = "validate_fail"
    blocked = "blocked"
    running = "running"
    aborting = "aborting"
    run_success = "run_success"
    postprocess = "postprocess"
    run_failed = "run_failed"
    diverged = "diverged"
    aborted = "aborted"


class BatchTaskBlockInfo(TaskBlockInfo):
    """Extended block info for batch detail."""

    accountLimit: float = None
    taskBlockMsg: str = None
    taskBlockType: str = None
    blockStatus: str = None
    taskStatus: str = None


class BatchMember(TaskBase):
    """Information for each member task within a batch."""

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
    """Top-level batch detail payload for component modelers."""

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
    totalSeconds: int = None
    totalCheckMillis: int = None
    message: str = None
    tasks: list[BatchMember] = []
