"""Stub class"""

from abc import abstractmethod
from enum import Enum

from pydantic.v1 import BaseModel


class TaskType(Enum):
    FDTD = "FDTD"
    NORMAL="NORMAL"
    MODE_SOLVER = "MODE_SOLVER"
    HEAT = "HEAT"


class Stub(BaseModel):

    @abstractmethod
    def from_file(self):
        pass

    @abstractmethod
    def to_file(self):
        pass

    @abstractmethod
    def get_type(self) -> TaskType:
        pass

    @abstractmethod
    def validate_pre_upload(self, source_required):
        pass


class StubData:
    pass