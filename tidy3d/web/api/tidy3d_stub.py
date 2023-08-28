"""Stub for webapi"""
from __future__ import annotations

import json
from typing import Union, Callable, List

from pydantic.v1 import BaseModel

from ..core.file_util import (
    read_simulation_from_json,
    read_simulation_from_hdf5,
    read_simulation_from_hdf5_gz,
)
from ..core.stub import TaskStub, TaskStubData
from ... import log
from ...components.base import _get_valid_extension
from ...components.data.sim_data import SimulationData
from ...components.data.monitor_data import ModeSolverData
from ..core.types import TaskType
from ...components.simulation import Simulation
from ...plugins.mode.mode_solver import ModeSolver


SimulationType = Union[Simulation]
SimulationDataType = Union[SimulationData]


class Tidy3dStub(BaseModel, TaskStub):

    simulation: SimulationType

    @classmethod
    def from_file(cls, file_path: str) -> SimulationType:
        """Loads a :class:`Stub` from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the :class:`.SimulationType` from.

        Returns
        -------
        :class:`.SimulationType`
            An instance of the component class calling `load`.

        Example
        -------
        >>> simulation = Simulation.from_file(fname='folder/sim.json') # doctest: +SKIP
        """
        extension = _get_valid_extension(file_path)
        if extension == ".json":
            json_str = read_simulation_from_json(file_path)
        elif extension == ".hdf5":
            json_str = read_simulation_from_hdf5(file_path)
        elif extension == ".hdf5.gz":
            json_str = read_simulation_from_hdf5_gz(file_path)

        data = json.loads(json_str)
        type_ = data["type"]
        if "Simulation" == type_:
            sim = Simulation.from_file(file_path)
        elif "ModeSolver" == type_:
            sim = ModeSolver.from_file(file_path)

        return sim

    def to_file(
        self,
        file_path: str,
    ):
        """Exports :class:`Stub` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to save the :class:`Stub` to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """
        self.simulation.to_file(file_path)

    def to_hdf5_gz(self, fname: str, custom_encoders: List[Callable] = None) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .hdf5.gz file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5.gz file to save the :class:`Tidy3dBaseModel` to.
        custom_encoders : List[Callable]
            List of functions accepting (fname: str, group_path: str, value: Any) that take
            the ``value`` supplied and write it to the hdf5 ``fname`` at ``group_path``.

        Example
        -------
        >>> simulation.to_hdf5_gz(fname='folder/sim.hdf5.gz') # doctest: +SKIP
        """

        self.simulation.to_hdf5_gz(fname)

    def get_type(self) -> str:
        """Get simulation instance type.

        Returns
        -------
        :class:`TaskType`
            An instance Type of the component class calling `load`.
        """
        if isinstance(self.simulation, Simulation):
            return TaskType.FDTD.name
        elif isinstance(self.simulation, ModeSolver):
            return TaskType.MODE_SOLVER.name

    def validate_pre_upload(self, source_required) -> None:
        """Perform some pre-checks on instances of component"""
        self.simulation.validate_pre_upload(source_required)


class Tidy3dStubData(BaseModel, TaskStubData):
    """"""

    data: SimulationDataType

    @classmethod
    def from_file(cls, file_path: str) -> SimulationDataType:
        """Loads a :class:`SimulationDataType` from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the :class:`Stub` from.

        Returns
        -------
        :class:`.SimulationDataType`
            An instance of the component class calling `load`.
        """
        extension = _get_valid_extension(file_path)
        if extension == ".json":
            json_str = read_simulation_from_json(file_path)
        elif extension == ".hdf5":
            json_str = read_simulation_from_hdf5(file_path)
        elif extension == ".hdf5.gz":
            json_str = read_simulation_from_hdf5_gz(file_path)

        data = json.loads(json_str)
        type_ = data["type"]
        if "SimulationData" == type_:
            sim_data = SimulationData.from_file(file_path)
        elif "ModeSolverData" == type_:
            sim_data = ModeSolverData.from_file(file_path)

        return sim_data

    def to_file(
        self,
        file_path: str,
    ):
        """Exports :class:`.SimulationDataType` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to save the :class:`.SimulationDataType` to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """
        self.data.to_file(file_path)

    @classmethod
    def postprocess(cls, file_path: str) -> SimulationDataType:
        """load  .yaml, .json, or .hdf5 file to :class:`.SimulationDataType` instance.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to save the :class:`.SimulationDataType` to.

        Returns
        -------
        :class:`.SimulationDataType`
            An instance of the component class calling `load`.
        """
        stub_data = Tidy3dStubData.from_file(file_path)

        final_decay_value = stub_data.final_decay_value
        shutoff_value = stub_data.simulation.shutoff
        if (shutoff_value != 0) and (final_decay_value > shutoff_value):
            log.warning(
                f"Simulation final field decay value of {final_decay_value} "
                f"is greater than the simulation shutoff threshold of {shutoff_value}. "
                "Consider simulation again with large run_time duration for more accurate results."
            )
        return stub_data
