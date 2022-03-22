""" Defnes information about a task """

from enum import Enum
from abc import ABC
from typing import Any

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

    execCount: int = None
    s3Storage: float = None
    userEmail: str = None
    # coreDuration: float
    coreStartTime: str = None
    # rankCount: int
    # submitTime: str
    updateTime: str = None
    status: str = None
    taskParam: str = None
    # objectId: str
    # folderId: str
    solverVersion: str = None
    worker: str = None
    # userId: str
    taskType: str = None
    # objectType: str
    taskName: str = None
    errorMessages: str = None
    nodeSize: int = None
    timeSteps: int = None
    computeWeight: float = None
    solverStartTime: str = None
    solverEndTime: str = None
    taskId: str
    workerGroup: str = None
    realCost: float = None
    cloudInstanceSize: int = None
    flow360InstanceSize: int = None
    estCostMin: float = None
    estCostMax: float = None
    # running: bool
    objectRefId: str = None
    metdataProcessed: bool = None
    optSolverUnit: float = None
    minSolverUnit: float = None
    maxSolverUnit: float = None
    coreStartTimeAsLong: float = None
    id: str = None
    metadataProcessed: bool = None
    parentId: str = None
    realWorkUnit: float = None
    refId: str = None
    solverEndTimeAsLong: float = None
    solverStartTimeAsLong: float = None
    submitTimeAsLong: float = None
    credential: Any = None
    loopSolverTime: Any = None
    shutoffNt: Any = None
    startSolverTime: Any = None
    totalSolverTime: Any = None


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
