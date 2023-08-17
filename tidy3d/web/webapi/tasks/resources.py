"""Tidy3d abstraction types for the webapi."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class Tidy3DResource(BaseModel, ABC):
    """Abstract base class / template for a webservice that implements resource query."""

    @classmethod
    @abstractmethod
    def get(cls, *args, **kwargs) -> Tidy3DResource:
        """Get a resource from the server."""
        # TODO: define a **generic** approach to each of these methods that works for any Stub.

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> Tidy3DResource:
        """Create a new resource and return it."""
        # TODO: define a **generic** approach to each of these methods that works for any Stub.

    @abstractmethod
    def delete(self, *args, **kwargs) -> None:
        """Delete the resource."""
        # TODO: define a **generic** approach to each of these methods that works for any Stub.

    @abstractmethod
    def submit(self, *args, **kwargs) -> None:
        """Submit the task to the webservice."""
        # TODO: define a **generic** approach to each of these methods that works for any Stub.

    @classmethod
    @abstractmethod
    def list(cls, *args, **kwargs) -> [Queryable]:
        """List all resources of this type."""
        # TODO: define a **generic** approach to each of these methods that works for any Stub.
