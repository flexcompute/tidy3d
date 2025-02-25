"""Stub for webapi"""

from __future__ import annotations

import json
from typing import Callable, List, Union

import pydantic.v1 as pd
from pydantic.v1 import BaseModel

from tidy3d.components.tcad.data.sim_data import HeatChargeSimulationData, HeatSimulationData
from tidy3d.components.tcad.simulation.heat import HeatSimulation
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation

from ... import log
from ...components.base import _get_valid_extension
from ...components.data.monitor_data import ModeSolverData
from ...components.data.sim_data import SimulationData
from ...components.eme.data.sim_data import EMESimulationData
from ...components.eme.simulation import EMESimulation
from ...components.mode.data.sim_data import ModeSimulationData
from ...components.mode.simulation import ModeSimulation
from ...components.simulation import Simulation
from ...plugins.mode.mode_solver import ModeSolver
from ..core.file_util import (
    read_simulation_from_hdf5,
    read_simulation_from_hdf5_gz,
    read_simulation_from_json,
)
from ..core.stub import TaskStub, TaskStubData
from ..core.types import TaskType

SimulationType = Union[
    Simulation, HeatChargeSimulation, HeatSimulation, EMESimulation, ModeSolver, ModeSimulation
]
SimulationDataType = Union[
    SimulationData,
    HeatChargeSimulationData,
    HeatSimulationData,
    EMESimulationData,
    ModeSolverData,
    ModeSimulationData,
]


class Tidy3dStub(BaseModel, TaskStub):
    simulation: SimulationType = pd.Field(discriminator="type")

    @classmethod
    def from_file(cls, file_path: str) -> SimulationType:
        """Loads a Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]
        from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the
            Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] from.

        Returns
        -------
        Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]
            An instance of the component class calling ``load``.

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
        elif "HeatSimulation" == type_:
            sim = HeatSimulation.from_file(file_path)
        elif "HeatChargeSimulation" == type_:
            sim = HeatChargeSimulation.from_file(file_path)
        elif "EMESimulation" == type_:
            sim = EMESimulation.from_file(file_path)
        elif "ModeSimulation" == type_:
            sim = ModeSimulation.from_file(file_path)

        return sim

    def to_file(
        self,
        file_path: str,
    ):
        """Exports Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] instance to .yaml, .json,
        or .hdf5 file

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
        """Exports Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] instance to .hdf5.gz file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5.gz file to save
            the Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] to.
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
            An instance Type of the component class calling ``load``.
        """
        if isinstance(self.simulation, Simulation):
            return TaskType.FDTD.name
        elif isinstance(self.simulation, ModeSolver):
            return TaskType.MODE_SOLVER.name
        elif isinstance(self.simulation, HeatSimulation):
            return TaskType.HEAT.name
        elif isinstance(self.simulation, HeatChargeSimulation):
            return TaskType.HEAT_CHARGE.name
        elif isinstance(self.simulation, EMESimulation):
            return TaskType.EME.name
        elif isinstance(self.simulation, ModeSimulation):
            return TaskType.MODE.name

    def validate_pre_upload(self, source_required) -> None:
        """Perform some pre-checks on instances of component"""
        if isinstance(self.simulation, Simulation):
            self.simulation.validate_pre_upload(source_required)
        elif isinstance(self.simulation, EMESimulation):
            self.simulation.validate_pre_upload()


class Tidy3dStubData(BaseModel, TaskStubData):
    """"""

    data: SimulationDataType

    @classmethod
    def from_file(cls, file_path: str) -> SimulationDataType:
        """Loads a Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
        from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the
            Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] from.

        Returns
        -------
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
            An instance of the component class calling ``load``.
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
        elif "HeatSimulationData" == type_:
            sim_data = HeatSimulationData.from_file(file_path)
        elif "HeatChargeSimulationData" == type_:
            sim_data = HeatChargeSimulationData.from_file(file_path)
        elif "EMESimulationData" == type_:
            sim_data = EMESimulationData.from_file(file_path)
        elif "ModeSimulationData" == type_:
            sim_data = ModeSimulationData.from_file(file_path)

        return sim_data

    def to_file(self, file_path: str):
        """Exports Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] instance
        to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to save the
            Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """
        self.data.to_file(file_path)

    @classmethod
    def postprocess(cls, file_path: str) -> SimulationDataType:
        """Load .yaml, .json, or .hdf5 file to
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] instance.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to save the
            Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] to.

        Returns
        -------
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
            An instance of the component class calling ``load``.
        """
        stub_data = Tidy3dStubData.from_file(file_path)

        check_log_msg = "For more information, check 'SimulationData.log' or use "
        check_log_msg += "'web.download_log(task_id)'."
        warned_about_warnings = False

        if isinstance(stub_data, SimulationData):
            final_decay_value = stub_data.final_decay_value
            shutoff_value = stub_data.simulation.shutoff
            if stub_data.diverged:
                log.warning("The simulation has diverged! " + check_log_msg)
                warned_about_warnings = True
            elif (shutoff_value != 0) and (final_decay_value > shutoff_value):
                log.warning(
                    f"Simulation final field decay value of {final_decay_value} is greater than "
                    f"the simulation shutoff threshold of {shutoff_value}. Consider running the "
                    "simulation again with a larger 'run_time' duration for more accurate results."
                )

        if (
            not isinstance(stub_data, (ModeSolverData, ModeSimulationData))
            and "WARNING" in stub_data.log
            and not warned_about_warnings
        ):
            log.warning("Warning messages were found in the solver log. " + check_log_msg)

        return stub_data
