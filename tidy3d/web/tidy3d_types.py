"""
Tidy3d abstraction types for the webapi
"""
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="Tidy3DResource")


class Tidy3DResource(BaseModel, ABC):
    """ "Abstract base class / template for a webservice that implements resource query."""

    @classmethod
    @abstractmethod
    def get(cls, *args, **kwargs) -> T:
        """Get a resource from the server."""


class ResourceLifecycle(Tidy3DResource, ABC):
    """
    Abstract base class / template for a webservice that implements resource life cycle management.
    """

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> T:
        """Create a new resource and return it."""

    @abstractmethod
    def delete(self, *args, **kwargs) -> None:
        """Delete the resource."""


class Submittable(BaseModel, ABC):
    """Abstract base class / template for a webservice that implements a submit method."""

    @abstractmethod
    def submit(self, *args, **kwargs) -> None:
        """Submit the task to the webservice."""


Q = TypeVar("Q", bound="Queryable")


class Queryable(BaseModel, ABC):
    """Abstract base class / template for a webservice that implements a query method."""

    @classmethod
    @abstractmethod
    def list(cls, *args, **kwargs) -> [Q]:
        """List all resources of this type."""
