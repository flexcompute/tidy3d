""" Defnes information about a task """

from enum import Enum
from abc import ABC

import pydantic

from ..components.base import Tidy3dBaseModel


class TaskStatus(Enum):
    """the statuses that the task can be in"""

    INIT = "initialized"
    QUEUE = "queued"
    PRE = "preprocessing"
    RUN = "running"
    POST = "postprocessing"
    SUCCESS = "success"
    ERROR = "error"


class TaskBase(Tidy3dBaseModel, ABC):
    """base config for all task objects"""


# type of the task_id
TaskId = str

# type of task_name
TaskName = str


class TaskInfo(TaskBase):
    """general information about task"""

    execCount: int
    s3Storage: float
    userEmail: str = None
    coreDuration: float
    coreStartTime: str = None
    rankCount: int
    submitTime: str
    updateTime: str = None
    status: str
    taskParam: str = None
    objectId: str
    folderId: str
    solverVersion: str
    worker: str = None
    userId: str
    taskType: str
    objectType: str
    taskName: str
    errorMessages: str = None
    nodeSize: int
    timeSteps: int
    computeWeight: float
    solverStartTime: str = None
    solverEndTime: str = None
    taskId: str
    workerGroup: str = None
    realCost: float
    cloudInstanceSize: int
    flow360InstanceSize: int
    estCostMin: float
    estCostMax: float
    running: bool
    objectRefId: str = None
    metdataProcessed: bool


class RunInfo(TaskBase):
    """information about the run"""

    perc_done: pydantic.confloat(ge=0.0, le=100.0)
    field_decay: pydantic.confloat(ge=0.0, le=1.0)

    def display(self):
        """print some info"""
        print(f" - {self.perc_done:.2f} (%) done")
        print(f" - {self.field_decay:.2e} field decay from max")


class Task(TaskBase):
    """container for a task"""

    id: TaskId
    info: TaskInfo
