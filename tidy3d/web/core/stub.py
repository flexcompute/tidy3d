"""Defines interface that can be subclassed to use with the tidy3d webapi"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TaskStubData(ABC):
    @abstractmethod
    def from_file(self, file_path) -> TaskStubData:
        """Loads a :class:`TaskStubData` from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the :class:`Stub` from.

        Returns
        -------
        :class:`Stub`
            An instance of the component class calling ``load``.

        """
        pass

    @abstractmethod
    def to_file(self, file_path):
        """Loads a :class:`Stub` from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the :class:`Stub` from.

        Returns
        -------
        :class:`Stub`
            An instance of the component class calling ``load``.
        """
        pass


class TaskStub(ABC):
    @abstractmethod
    def from_file(self, file_path) -> TaskStub:
        """Loads a :class:`TaskStubData` from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the :class:`Stub` from.

        Returns
        -------
        :class:`TaskStubData`
            An instance of the component class calling ``load``.
        """
        pass

    @abstractmethod
    def to_file(self, file_path):
        """Loads a :class:`TaskStub` from .yaml, .json, .hdf5 or .hdf5.gz file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the :class:`TaskStub` from.

        Returns
        -------
        :class:`Stub`
            An instance of the component class calling ``load``.
        """
        pass

    @abstractmethod
    def to_hdf5_gz(self, fname: str) -> None:
        """Exports :class:`TaskStub` instance to .hdf5.gz file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5.gz file to save the :class:`TaskStub` to.
        """
        pass
