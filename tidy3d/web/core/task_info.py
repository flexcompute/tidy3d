"""Defines information about a task"""

from __future__ import annotations

from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field


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


class RunInfo(TaskBase):
    """Information about the run of a task."""

    perc_done: Annotated[float, Field(ge=0.0, le=100.0)]
    """Percentage of the task that is completed (0 to 100)."""

    field_decay: Annotated[float, Field(ge=0.0, le=1.0)]
    """Field decay from the maximum value (0 to 1)."""

    def display(self):
        """Print some info about the task's progress."""
        print(f" - {self.perc_done:.2f} (%) done")
        print(f" - {self.field_decay:.2e} field decay from max")
