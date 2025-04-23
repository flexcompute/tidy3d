"""Tidy3d abstraction types for the core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic.v1 import BaseModel


class Tidy3DResource(BaseModel, ABC):
    """Abstract base class / template for a webservice that implements resource query."""

    @classmethod
    @abstractmethod
    def get(cls, *args, **kwargs) -> Tidy3DResource:
        """Get a resource from the server."""


class ResourceLifecycle(Tidy3DResource, ABC):
    """Abstract base class for a webservice that implements resource life cycle management."""

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> Tidy3DResource:
        """Create a new resource and return it."""

    @abstractmethod
    def delete(self, *args, **kwargs) -> None:
        """Delete the resource."""


class Submittable(BaseModel, ABC):
    """Abstract base class / template for a webservice that implements a submit method."""

    @abstractmethod
    def submit(self, *args, **kwargs) -> None:
        """Submit the task to the webservice."""


class Queryable(BaseModel, ABC):
    """Abstract base class / template for a webservice that implements a query method."""

    @classmethod
    @abstractmethod
    def list(cls, *args, **kwargs) -> [Queryable]:
        """List all resources of this type."""


class TaskType(str, Enum):
    FDTD = "FDTD"
    MODE_SOLVER = "MODE_SOLVER"
    HEAT = "HEAT"
    HEAT_CHARGE = "HEAT_CHARGE"
    EME = "EME"
    MODE = "MODE"


class PayType(str, Enum):
    CREDITS = "FLEX_CREDIT"
    AUTO = "AUTO"

    @classmethod
    def _missing_(cls, value: object) -> PayType:
        if isinstance(value, str):
            key = value.strip().replace(" ", "_").upper()
            if key in cls.__members__:
                return cls.__members__[key]
        return super()._missing_(value)
