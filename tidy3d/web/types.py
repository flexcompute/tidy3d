"""Tidy3d abstraction types for the webapi."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel
from ..components.base import Tidy3dBaseModel


class Tidy3DResource(Tidy3dBaseModel, ABC):
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


class Submittable(Tidy3dBaseModel, ABC):
    """Abstract base class / template for a webservice that implements a submit method."""

    @abstractmethod
    def submit(self, *args, **kwargs) -> None:
        """Submit the task to the webservice."""


class Queryable(Tidy3dBaseModel, ABC):
    """Abstract base class / template for a webservice that implements a query method."""

    @classmethod
    @abstractmethod
    def list(cls, *args, **kwargs) -> [Queryable]:
        """List all resources of this type."""
