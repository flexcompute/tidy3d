""" Defnes information about a task """
from datetime import datetime
from enum import Enum
from abc import ABC
from typing import Optional

import pydantic


class TaskStatus(Enum):
    """the statuses that the task can be in"""

    INIT = "initialized"
    QUEUE = "queued"
    PRE = "preprocessing"
    RUN = "running"
    POST = "postprocessing"
    SUCCESS = "success"
    ERROR = "error"


class TaskBase(pydantic.BaseModel, ABC):
    """base config for all task objects"""

    class Config:
        """configure class"""

        arbitrary_types_allowed = True


# type of the task_id
TaskId = str

# type of task_name
TaskName = str


class TaskInfo(TaskBase):
    """general information about task"""

    taskId: str
    taskName: str = None
    nodeSize: int = None
    completedAt: Optional[datetime] = None
    status: str = None
    realCost: float = None
    timeSteps: int = None
    solverVersion: str = None
    createAt: Optional[datetime] = None
    estCostMin: float = None
    estCostMax: float = None
    realFlexUnit: float = None
    estFlexUnit: float = None
    s3Storage: float = None
    startSolverTime: Optional[datetime] = None
    finishSolverTime: Optional[datetime] = None
    totalSolverTime: int = None
    callbackUrl: str = None
    taskType: str = None


class RunInfo(TaskBase):
    """information about the run"""

    perc_done: pydantic.confloat(ge=0.0, le=100.0)
    field_decay: pydantic.confloat(ge=0.0, le=1.0)

    def display(self):
        """print some info"""
        print(f" - {self.perc_done:.2f} (%) done")
        print(f" - {self.field_decay:.2e} field decay from max")


class Folder(pydantic.BaseModel):
    """
    Folder information of a task
    """

    projectName: str = None
    projectId: str = None

    class Config:
        """configure class"""

        arbitrary_types_allowed = True
