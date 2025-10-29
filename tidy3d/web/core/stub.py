"""Defines interface that can be subclassed to use with the tidy3d webapi"""

from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike


class TaskStubData(ABC):
    @abstractmethod
    def from_file(self, file_path: PathLike) -> TaskStubData:
        """Loads a :class:`TaskStubData` from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : PathLike
            Full path to the .yaml or .json or .hdf5 file to load the :class:`Stub` from.

        Returns
        -------
        :class:`Stub`
            An instance of the component class calling ``load``.

        """

    @abstractmethod
    def to_file(self, file_path: PathLike) -> None:
        """Loads a :class:`Stub` from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : PathLike
            Full path to the .yaml or .json or .hdf5 file to load the :class:`Stub` from.

        Returns
        -------
        :class:`Stub`
            An instance of the component class calling ``load``.
        """


class TaskStub(ABC):
    @abstractmethod
    def from_file(self, file_path: PathLike) -> TaskStub:
        """Loads a :class:`TaskStubData` from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : PathLike
            Full path to the .yaml or .json or .hdf5 file to load the :class:`Stub` from.

        Returns
        -------
        :class:`TaskStubData`
            An instance of the component class calling ``load``.
        """

    @abstractmethod
    def to_file(self, file_path: PathLike) -> None:
        """Loads a :class:`TaskStub` from .yaml, .json, .hdf5 or .hdf5.gz file.

        Parameters
        ----------
        file_path : PathLike
            Full path to the .yaml or .json or .hdf5 file to load the :class:`TaskStub` from.

        Returns
        -------
        :class:`Stub`
            An instance of the component class calling ``load``.
        """

    @abstractmethod
    def to_hdf5_gz(self, fname: PathLike) -> None:
        """Exports :class:`TaskStub` instance to .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .hdf5.gz file to save the :class:`TaskStub` to.
        """
