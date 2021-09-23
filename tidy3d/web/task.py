import pydantic
from enum import Enum
from abc import ABC

from ..components.types import Literal
from ..components.base import Tidy3dBaseModel

""" Defnes information about a task """


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

    pass


# type of the task_id
TaskId = str


class TaskInfo(TaskBase):
    """general information about task"""

    task_id: TaskId
    status: TaskStatus
    size_bytes: int
    credits: pydantic.confloat(ge=0.0)


class RunInfo(TaskBase):
    """information about the run"""

    perc_done: pydantic.confloat(ge=0.0, le=100.0)
    field_decay: pydantic.confloat(ge=0.0, le=1.0)

    def display(self):
        print(f" - {self.perc_done:.2f} (%) done")
        print(f" - {self.field_decay:.2e} field decay from max")


class Task(TaskBase):
    """container for a task"""

    id: TaskId
    info: TaskInfo
