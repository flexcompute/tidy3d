""" Defnes information about a task """
from datetime import datetime
from enum import Enum
from abc import ABC
from typing import Optional

import pydantic
from pydantic import Field, ConfigDict
from typing_extensions import Annotated


class TaskStatus(Enum):
    """The statuses that the task can be in."""

    INIT = "initialized"
    QUEUE = "queued"
    PRE = "preprocessing"
    RUN = "running"
    POST = "postprocessing"
    SUCCESS = "success"
    ERROR = "error"


class TaskBase(pydantic.BaseModel, ABC):
    """Base config for all task objects."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# type of the task_id
TaskId = str

# type of task_name
TaskName = str


class ChargeType(str, Enum):
    """The payment method of task."""

    FREE = "free"
    PAID = "paid"


class TaskBlockInfo(TaskBase):
    """The block info that task will be blocked by all three features of DE,
    User limit and Insufficient balance"""

    chargeType: ChargeType = None
    maxFreeCount: int = None
    maxGridPoints: int = None
    maxTimeSteps: int = None


class TaskInfo(TaskBase):
    """General information about task."""

    taskId: str
    taskName: Optional[str] = None
    nodeSize: Optional[int] = None
    completedAt: Optional[datetime] = None
    status: Optional[str] = None
    realCost: Optional[float] = None
    timeSteps: Optional[int] = None
    solverVersion: Optional[str] = None
    createAt: Optional[datetime] = None
    estCostMin: Optional[float] = None
    estCostMax: Optional[float] = None
    realFlexUnit: Optional[float] = None
    estFlexUnit: Optional[float] = None
    s3Storage: Optional[float] = None
    startSolverTime: Optional[datetime] = None
    finishSolverTime: Optional[datetime] = None
    totalSolverTime: Optional[int] = None
    callbackUrl: Optional[str] = None
    taskType: Optional[str] = None
    metadataStatus: Optional[str] = None
    taskBlockInfo: Optional[TaskBlockInfo] = None


class RunInfo(TaskBase):
    """Information about the run."""

    perc_done: Annotated[float, Field(ge=0.0, le=100.0)]
    field_decay: Annotated[float, Field(ge=0.0, le=1.0)]

    def display(self):
        """Print some info."""
        print(f" - {self.perc_done:.2f} (%) done")
        print(f" - {self.field_decay:.2e} field decay from max")


class Folder(pydantic.BaseModel):
    """Folder information of a task."""

    projectName: str = None
    projectId: str = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
