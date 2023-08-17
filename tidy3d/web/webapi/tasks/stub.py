# Generic base class for some kind of task, implement abstractmethods in your base classes.
from __future__ import annotations

from abc import ABC, abstractmethod
from .resources import Tidy3DResource


class Stub(ABC, Tidy3DResource):
    """Generic template for a task that can be handled by this webapi."""

    @abstractmethod
    def to_file(self, path: str) -> None:
        """How to write the object to file for upload."""

    @classmethod
    @abstractmethod
    def from_file(cls, path: str) -> Stub:
        """How to construct the object from a file."""
